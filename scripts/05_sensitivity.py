"""
IDT 敏感性分析 (有限差分扰动法) - 复现论文 Fig. 6

方法: 对完整机理每个反应 i, 将其速率乘数乘以 2 (k->2k), 重新计算 IDT:
    S_i = ln(tau_perturbed / tau_base) / ln(2)
即 IDT 对速率常数的对数导数 dln(tau)/dln(k)。每个工况下按最大绝对值归一化。

工况: 论文 Table 3 Case 1-8 的中值温度 (与 GA 简化目标一致)。
输出: results/sensitivity.csv (含 reaction, condition, S, S_norm)
"""
import sys
import numpy as np
import pandas as pd
import cantera as ct
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_utils import (MECH_DIR, RESULTS_DIR, FULL_MECH, chen_composition,
                        ignition_delay)

PERTURB_FACTOR = 2.0

# 论文 Table 3 Case 1-8 中值温度
SENS_CONDITIONS = [
    {"case": 1, "XH2": 0.0, "P_atm": 1.0, "T": 1700.0},
    {"case": 2, "XH2": 0.0, "P_atm": 10.0, "T": 1600.0},
    {"case": 3, "XH2": 0.05, "P_atm": 1.0, "T": 1700.0},
    {"case": 4, "XH2": 0.05, "P_atm": 10.0, "T": 1700.0},
    {"case": 5, "XH2": 0.3, "P_atm": 1.0, "T": 1500.0},
    {"case": 6, "XH2": 0.3, "P_atm": 10.0, "T": 1550.0},
    {"case": 7, "XH2": 0.7, "P_atm": 1.0, "T": 1270.0},
    {"case": 8, "XH2": 0.7, "P_atm": 10.0, "T": 1165.0},
]


def run_condition(args):
    """单个工况: 基准 IDT + 逐反应扰动 IDT -> 敏感性系数"""
    case, XH2, P, T = args
    gas = ct.Solution(str(FULL_MECH))
    comp = chen_composition(XH2, 1.0)   # Chen 2021 Ar 稀释混合气
    n = gas.n_reactions

    # 基准 (先粗扫确定 t_end)
    tau0 = ignition_delay(gas, T, P, comp, t_end=10.0 if T < 1300 else 1.0)
    if not np.isfinite(tau0):
        return case, None
    t_end = min(max(100.0 * tau0, 1e-5), 10.0)
    tau0 = ignition_delay(gas, T, P, comp, t_end)
    if not np.isfinite(tau0):
        return case, None

    eqs = [gas.reaction(i).equation for i in range(n)]
    S = np.full(n, np.nan)
    for i in range(n):
        gas.set_multiplier(PERTURB_FACTOR, i)
        tau = ignition_delay(gas, T, P, comp, t_end)
        gas.set_multiplier(1.0, i)
        if np.isfinite(tau) and tau > 0:
            S[i] = np.log(tau / tau0) / np.log(PERTURB_FACTOR)

    print(f"  Case {case} (XH2={XH2}, P={P} atm, T={T} K): "
          f"IDT0={tau0*1000:.4f} ms, done")
    return case, {"equations": eqs, "S": S, "tau0": tau0}


def main():
    tasks = [(c["case"], c["XH2"], c["P_atm"], c["T"]) for c in SENS_CONDITIONS]
    results = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for case, res in ex.map(run_condition, tasks):
            if res is not None:
                results[case] = res

    # 汇总为长表
    rows = []
    for c in SENS_CONDITIONS:
        case = c["case"]
        if case not in results:
            continue
        res = results[case]
        S = res["S"]
        denom = np.nanmax(np.abs(S))
        for i, (eq, s) in enumerate(zip(res["equations"], S)):
            rows.append({
                "case": case,
                "XH2": c["XH2"], "P_atm": c["P_atm"], "T_K": c["T"],
                "reaction_index": i,
                "reaction": eq,
                "sensitivity": s,
                "sensitivity_normalized": s / denom if denom > 0 else np.nan,
            })
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "sensitivity.csv"
    df.to_csv(out, index=False)

    # 打印每个工况 top-5
    print("\nTop-5 reactions per condition:")
    for c in SENS_CONDITIONS:
        sub = df[df["case"] == c["case"]].copy()
        sub["abs"] = sub["sensitivity"].abs()
        top = sub.nlargest(5, "abs")
        print(f" Case {c['case']} (XH2={c['XH2']}, P={c['P_atm']} atm):")
        for _, r in top.iterrows():
            print(f"   {r['reaction']:<40s} S={r['sensitivity']:+.3f} "
                  f"(norm {r['sensitivity_normalized']:+.3f})")
    print(f"\n敏感性分析完成: {out}")


if __name__ == "__main__":
    main()
