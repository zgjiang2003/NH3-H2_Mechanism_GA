"""
IDT 验证仿真 - 复现论文 Fig. 9, 10, S1, S2

机理:
  Full = Full mechanism 2021recon.yaml  (Zhang 2021 重建版, 论文的"初始机理"替代)
  M1   = Reduced mechanism.yaml         (论文补充材料原版 Mechanism 1)
  M2   = Optimized mechanism.yaml       (论文补充材料原版 Mechanism 2)

图件与工况 (论文 Table 3 与图注):
  Fig. 9  (2x2): Chen et al. [42] 激波管, phi=1, P=1.2/10 atm, XH2=0/0.05/0.3/0.7
  Fig. 10 (1x3): (a) NH3/air [44] phi=1, P=20/40 bar;
                 (b) NH3/O2/Ar [17] phi=0.5, P=1.4 atm; (c) phi=2.0
  Fig. S1 (1x2): NH3/O2/Ar [17] phi=1, P=1.4/11 atm, 高温段
  Fig. S2 (1x2): (a) RCM [2] phi=0.35, P=43.4 bar; (b) [3] phi=1, P=20 bar
点火判据: OH 摩尔分数峰值时刻。
输出: results/idt_sweep.csv
"""
import sys
import numpy as np
import pandas as pd
import cantera as ct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_utils import (RESULTS_DIR, phi_composition, chen_composition,
                        ignition_delay, dissociated_nh3_composition)

MECHS = {
    "Full": "Full mechanism 2021recon.yaml",
    "Mechanism 1": "Reduced mechanism.yaml",
    "Mechanism 2": "Optimized mechanism.yaml",
}


def sweep():
    all_mechs = {k: (Path(__file__).resolve().parent.parent
                     / "mechanisms" / v) for k, v in MECHS.items()}

    cases = []

    # ---------------- Fig. 9: Chen et al. [42], phi=1 ----------------
    fig9 = {
        0.0:  {"T": np.arange(1430, 1860, 30), "label": "(a) 100% NH3"},
        0.05: {"T": np.arange(1500, 1930, 30), "label": "(b) 95%NH3/5%H2"},
        0.3:  {"T": np.arange(1180, 1830, 30), "label": "(c) 70%NH3/30%H2"},
        0.7:  {"T": np.arange(1000, 1550, 30), "label": "(d) 30%NH3/70%H2"},
    }
    for XH2, cfg in fig9.items():
        for P in (1.2, 10.0):
            cases.append({"fig": "Fig9", "label": cfg["label"],
                          "XH2": XH2, "P_atm": P, "phi": 1.0,
                          "diluent": "N2", "n2_o2": 3.76,
                          "T_range": cfg["T"]})

    # ---------------- Fig. 10 ----------------
    # (a) NH3/air [44] phi=1, P=20/40 bar
    for P in (20.0, 40.0):
        cases.append({"fig": "Fig10", "label": "(a) 100% NH3, phi=1",
                      "XH2": 0.0, "P_atm": P, "phi": 1.0,
                      "diluent": "N2", "n2_o2": 3.76,
                      "T_range": np.arange(1100, 2000, 40)})
    # (b)(c) NH3/O2/Ar [17]: 1.143%NH3/0.857%O2/98%Ar (phi=1 基准, Ar/O2=98/0.857)
    for phi, lab in [(0.5, "(b) phi=0.5"), (2.0, "(c) phi=2.0")]:
        cases.append({"fig": "Fig10", "label": lab + " NH3/O2/Ar",
                      "XH2": 0.0, "P_atm": 1.4, "phi": phi,
                      "diluent": "AR", "n2_o2": 98.0 / 0.857,
                      "T_range": np.arange(1500, 2500, 50)})

    # ---------------- Fig. S1: Mathieu [17], phi=1, 高温 ----------------
    for P in (1.4, 11.0):
        cases.append({"fig": "FigS1", "label": f"NH3/O2/Ar, phi=1, P={P} atm",
                      "XH2": 0.0, "P_atm": P, "phi": 1.0,
                      "diluent": "AR", "n2_o2": 98.0 / 0.857,
                      "T_range": np.arange(1600, 2550, 50)})

    # ---------------- Fig. S2: RCM 低温稀释 ----------------
    # (a) Liao et al. [2]: phi=0.35, P=43.4 bar, XH2=0/0.1/0.25
    for XH2 in (0.0, 0.1, 0.25):
        cases.append({"fig": "FigS2a", "label": f"{int((1-XH2)*100)}%NH3/{int(XH2*100)}%H2",
                      "XH2": XH2, "P_atm": 43.4, "phi": 0.35,
                      "diluent": "N2", "n2_o2": 3.76,
                      "T_range": np.arange(950, 1300, 25)})
    # (b) Pochet et al. [3]: phi=1, P=20 bar
    for XH2 in (0.0, 0.1, 0.25):
        cases.append({"fig": "FigS2b", "label": f"{int((1-XH2)*100)}%NH3/{int(XH2*100)}%H2",
                      "XH2": XH2, "P_atm": 20.0, "phi": 1.0,
                      "diluent": "N2", "n2_o2": 3.76,
                      "T_range": np.arange(1000, 1400, 25)})

    rows = []
    for mech_name, mech_path in all_mechs.items():
        gas = ct.Solution(str(mech_path))
        print(f"=== {mech_name} ({gas.n_species} sp, {gas.n_reactions} rxns) ===")
        for case in cases:
            for T in case["T_range"]:
                if case["fig"] == "Fig9":
                    # Chen 2021 激波管混合气: 4% 燃料 + 化学计量 O2 + Ar
                    comp = chen_composition(case["XH2"])
                else:
                    comp = phi_composition(case["XH2"], case["phi"],
                                           case["diluent"], case["n2_o2"])
                # 低温 RCM 条件需要更长时间
                if case["P_atm"] >= 20 and T < 1300:
                    t_end = 2000.0
                elif T < 1200:
                    t_end = 20.0
                elif T < 1500:
                    t_end = 2.0
                else:
                    t_end = 0.2
                tau = ignition_delay(gas, float(T), case["P_atm"], comp, t_end)
                rows.append({
                    "figure": case["fig"], "label": case["label"],
                    "mechanism": mech_name,
                    "XH2_fuel": case["XH2"], "P_atm": case["P_atm"],
                    "T_K": float(T), "1000/T": 1000.0 / T,
                    "phi": case["phi"], "IDT_s": tau, "IDT_ms": tau * 1000.0,
                })
            print(f"  {case['fig']} {case['label']} P={case['P_atm']} done")

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "idt_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nIDT 计算完成: {len(df)} 行 -> {out}")


if __name__ == "__main__":
    sweep()
