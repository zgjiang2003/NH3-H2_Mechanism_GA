"""
Fig. 18 专用: NH3/NO/Ar 预混火焰结构 (实验温度分布作为输入 - 论文方法)

论文: "The experimental temperature data were employed as input data for
the simulation." 实现: FreeFlame 关闭能量方程, 将数字化的实验温度分布
(E.FLAME_NH3_NO_AR) 强加为固定温度场, 求解组分/速度场。

输出: 追加至 results/premixed_flame.csv (config=Fig18_NH3_NO_Ar_fixedT)
"""
import sys
import numpy as np
import pandas as pd
import cantera as ct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_utils import RESULTS_DIR
import experimental_data as ED

MECHS = {
    "Full": "Full mechanism 2021recon.yaml",
    "Mechanism 2": "Optimized mechanism.yaml",
}
SPECIES_OUT = ["NH3", "H2", "O2", "H2O", "OH", "NO", "N2O", "N2", "H", "O",
               "AR", "HO2"]

COMP18 = {"NH3": 0.0098, "NO": 0.0067, "AR": 0.9835}
U_IN = 60.0    # cm/s


def solve_fixed_t(mech_path):
    gas = ct.Solution(str(mech_path))
    gas.TPX = 500.0, ct.one_atm, COMP18
    # 非均匀网格 (单位: m, 即 0-1 cm), 温升区加密
    grid = np.concatenate([
        np.linspace(0.0, 0.0015, 20),
        np.linspace(0.0015, 0.0060, 60)[1:],
        np.linspace(0.0060, 0.0100, 20)[1:]])
    # 实验距离为 cm, 网格为 m
    T_exp = np.interp(grid * 100.0, ED.FLAME_NH3_NO_AR["distance_cm"],
                      ED.FLAME_NH3_NO_AR["T_K"])

    f = ct.FreeFlame(gas, grid=grid)
    f.transport_model = "mixture-averaged"
    f.set_refine_criteria(ratio=3, slope=0.08, curve=0.16)
    f.P = ct.one_atm
    # 关键顺序: 先初始化, 再强加温度剖面, 关闭能量方程, 否则 solve() 会重置
    f.set_initial_guess()
    f.set_profile('T', grid / grid[-1], T_exp)
    f.energy_enabled = False
    for _ in range(6):
        f.solve(loglevel=0, refine_grid=False, auto=False)
    return f, gas


def main():
    mech_dir = Path(__file__).resolve().parent.parent / "mechanisms"
    rows = []
    for mech_name, mech_file in MECHS.items():
        print(f"=== Fig.18 fixed-T {mech_name} ===", flush=True)
        try:
            f, gas = solve_fixed_t(mech_dir / mech_file)
            grid = f.grid
            X = f.X
            for i in range(len(grid)):
                row = {"mechanism": mech_name,
                       "config": "Fig18_NH3_NO_Ar_fixedT",
                       "distance_cm": grid[i] * 100.0, "T_K": f.T[i],
                       "velocity_m_s": f.velocity[i]}
                for sp in SPECIES_OUT:
                    j = gas.species_index(sp) if sp in gas.species_names else None
                    row[f"X_{sp}"] = X[j, i] if j is not None else np.nan
                rows.append(row)
            print(f"  solved ({len(grid)} 点)", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)

    if rows:
        df_new = pd.DataFrame(rows)
        csv = RESULTS_DIR / "premixed_flame.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df = df[~df["config"].str.startswith("Fig18")]
            df = pd.concat([df, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(csv, index=False)
        print(f"更新 -> {csv}")


if __name__ == "__main__":
    main()
