"""
预混火焰结构验证 - 复现论文 Fig. 17, 18, S6

Fig. 17 (1x2): BurnerFlame (论文使用 Burner Flame model)
  (a) 纯 NH3, Tu=350 K, 入口速度 7.8 cm/s, phi=1, P=1 atm
  (b) X[NH3]:X[H2]=0.8:0.2, Tu=350 K, 入口速度 13.2 cm/s, phi=1
  对比: Full 与 Mechanism 2 (论文口径)
Fig. 18: NH3/NO/Ar 火焰, phi=1.46, 入口速度 60 cm/s (Vandooren [51])
  论文将实验温度分布作为输入; Cantera 不支持固定温度场,
  以 BurnerFlame + 实测入口速度近似 (README 记录该局限)
Fig. S6 (1x3): NH3/air 预混火焰 NO 分布, phi=0.9/1.0/1.2 (Brackmann [8])
输出: results/premixed_flame.csv
"""
import sys
import numpy as np
import pandas as pd
import cantera as ct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_utils import RESULTS_DIR, phi_composition

MECHS = {
    "Full": "Full mechanism 2021recon.yaml",
    "Mechanism 1": "Reduced mechanism.yaml",
    "Mechanism 2": "Optimized mechanism.yaml",
}
SPECIES_OUT = ["NH3", "H2", "O2", "H2O", "OH", "NO", "N2O", "N2", "H", "O",
               "AR", "HO2"]


def burner_flame(gas, comp, Tu, P_atm, u_cm_s, width=0.05):
    gas.TPX = Tu, P_atm * ct.one_atm, comp
    f = ct.BurnerFlame(gas, width=width)
    f.burner.mdot = gas.density * (u_cm_s / 100.0)
    f.transport_model = "mixture-averaged"
    f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
    # 分步求解增强收敛
    f.solve(loglevel=0, auto=True)
    return f


def free_flame(gas, comp, Tu, P_atm, width=0.05):
    gas.TPX = Tu, P_atm * ct.one_atm, comp
    f = ct.FreeFlame(gas, width=width)
    f.transport_model = "mixture-averaged"
    f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
    f.solve(loglevel=0, auto=True)
    return f


def record(rows, mech, config, f):
    grid = f.grid
    X = f.X
    for i in range(len(grid)):
        row = {"mechanism": mech, "config": config,
               "distance_cm": grid[i] * 100.0, "T_K": f.T[i],
               "velocity_m_s": f.velocity[i]}
        for sp in SPECIES_OUT:
            j = f.gas.species_index(sp) if sp in f.gas.species_names else None
            row[f"X_{sp}"] = X[j, i] if j is not None else np.nan
        rows.append(row)


def sweep():
    mech_dir = Path(__file__).resolve().parent.parent / "mechanisms"
    rows = []

    # ---------------- Fig. 17 ----------------
    fig17 = [
        {"label": "Fig17a_pure_NH3", "XH2": 0.0, "u": 7.8, "Tu": 350.0},
        {"label": "Fig17b_NH3H2_0.8_0.2", "XH2": 0.2, "u": 13.2, "Tu": 350.0},
    ]
    for mech_name in ("Full", "Mechanism 2"):     # 论文 Fig.17 仅对比二者
        gas = ct.Solution(str(mech_dir / MECHS[mech_name]))
        print(f"=== Fig.17 {mech_name} ===", flush=True)
        for cfg in fig17:
            comp = phi_composition(cfg["XH2"], 1.0)
            try:
                f = burner_flame(gas, comp, cfg["Tu"], 1.0, cfg["u"])
                record(rows, mech_name, cfg["label"], f)
                print(f"  {cfg['label']}: u={f.velocity[0]*100:.2f} cm/s, "
                      f"({len(f.grid)} 点)", flush=True)
            except Exception as e:
                print(f"  {cfg['label']} FAILED: {e}", flush=True)

    # ---------------- Fig. 18: NH3/NO/Ar, phi=1.46, u=60 cm/s ----------------
    # phi 定义为 NH3/NO 比 (NH3 + NO -> N2 + H2O 化学计量 1:1, phi=1.46 => NH3/NO=1.46)
    # 初始组成 (数字化): NH3≈0.0098, NO≈0.0067, Ar≈0.9835
    comp18 = {"NH3": 0.0098, "NO": 0.0067, "AR": 0.9835}
    for mech_name in ("Full", "Mechanism 2"):
        gas = ct.Solution(str(mech_dir / MECHS[mech_name]))
        print(f"=== Fig.18 {mech_name} ===", flush=True)
        for T_in, u in ((500.0, 60.0),):   # 入口温度按实验初始段 ~500 K
            gas.TPX = T_in, ct.one_atm, comp18
            try:
                f = ct.BurnerFlame(gas, width=0.05)
                f.burner.mdot = gas.density * (u / 100.0)
                f.transport_model = "mixture-averaged"
                f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
                f.solve(loglevel=0, auto=True)
                record(rows, mech_name, "Fig18_NH3_NO_Ar", f)
                print(f"  solved, ({len(f.grid)} 点)", flush=True)
            except Exception as e:
                print(f"  FAILED: {e} -> 改用 FreeFlame 近似", flush=True)
                try:
                    f = free_flame(gas, comp18, T_in, 1.0)
                    record(rows, mech_name, "Fig18_NH3_NO_Ar_FreeFlame", f)
                except Exception as e2:
                    print(f"  FreeFlame 也失败: {e2}", flush=True)

    # ---------------- Fig. S6: NH3/air NO 分布, phi=0.9/1.0/1.2 ----------------
    # Brackmann: BurnerFlame, 入口速度按化学计量 NH3 火焰 ~5-7 cm/s
    for mech_name in ("Full", "Mechanism 1", "Mechanism 2"):
        gas = ct.Solution(str(mech_dir / MECHS[mech_name]))
        print(f"=== Fig.S6 {mech_name} ===", flush=True)
        for phi in (0.9, 1.0, 1.2):
            comp = phi_composition(0.0, phi)
            label = f"FigS6_pure_NH3_phi{phi}"
            try:
                f = burner_flame(gas, comp, 298.0, 1.0, 5.5)
                record(rows, mech_name, label, f)
                print(f"  phi={phi}: ok", flush=True)
            except Exception as e:
                print(f"  phi={phi} FAILED: {e}", flush=True)

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "premixed_flame.csv"
    df.to_csv(out, index=False)
    print(f"\n预混火焰计算完成: {len(df)} 行 -> {out}")


if __name__ == "__main__":
    sweep()
