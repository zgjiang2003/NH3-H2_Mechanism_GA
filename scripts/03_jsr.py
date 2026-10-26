"""
JSR 物种浓度验证 - 复现论文 Fig. 16 (Zhang et al. [21] 实验)

工况 (依据 ReSpecTh/ELTE 基准数据库记录的 Zhang 2021 JSR 实验条件):
  P = 1 atm, T = 800-1300 K, 燃料为稀释 NH3/H2 (~300-500 ppm 级),
  氧化剂 O2, 稀释气 N2, 掺杂 ~1.5% H2O, 停留时间 2 s
  3 个子图: XH2 = 0 / 0.05 / 0.3 (对应 phi ≈ 0.15 / 0.155 / 0.18 贫燃系列)

模型: Cantera IdealGasConstPressureReactor + MassFlowController (PSR 稳态)
机理: Full(2021recon) / M1 / M2
输出: results/jsr_sweep.csv (摩尔分数)
"""
import sys
import numpy as np
import pandas as pd
import cantera as ct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_utils import RESULTS_DIR

MECHS = {
    "Full": "Full mechanism 2021recon.yaml",
    "Mechanism 1": "Reduced mechanism.yaml",
    "Mechanism 2": "Optimized mechanism.yaml",
}

# Zhang 2021 JSR 稀释条件 (ppm 级, 固定 NH3 与 O2, H2 随 XH2 增加):
#   NH3 = 300 ppm; O2 = 1500 ppm; H2 = NH3*XH2/(1-XH2); H2O = 1.5%; N2 平衡
JSR_SERIES = [
    {"XH2": 0.0,  "NH3_ppm": 300.0, "O2_ppm": 1500.0},
    {"XH2": 0.05, "NH3_ppm": 300.0, "O2_ppm": 1500.0},
    {"XH2": 0.3,  "NH3_ppm": 300.0, "O2_ppm": 1500.0},
]
TAU = 2.0        # 停留时间 s
P_ATM = 1.0
T_RANGE = np.arange(800, 1320, 25)
H2O_FRAC = 0.015


def inlet_composition(series):
    XH2 = series["XH2"]
    nh3 = series["NH3_ppm"] * 1e-6
    h2 = nh3 * XH2 / (1.0 - XH2)
    o2 = series["O2_ppm"] * 1e-6
    h2o = H2O_FRAC
    n2 = 1.0 - nh3 - h2 - o2 - h2o
    return {"NH3": nh3, "H2": h2, "O2": o2, "H2O": h2o, "N2": n2}


def jsr_steady_state(gas, T, comp):
    """PSR: 入口 -> MFC -> 定压反应器 -> 压力控制器 -> 出口, 推进至稳态"""
    gas.TPX = T, P_ATM * ct.one_atm, comp
    inlet = ct.Reservoir(gas)
    reactor = ct.IdealGasConstPressureReactor(gas)
    reactor.volume = 1.0e-6     # 1 cm^3
    exhaust = ct.Reservoir(gas)
    mdot = reactor.mass / TAU
    mfc = ct.MassFlowController(inlet, reactor, mdot=mdot)
    ct.PressureController(reactor, exhaust, primary=mfc, K=1e-3)
    net = ct.ReactorNet([reactor])
    t = 0.0
    for _ in range(60):
        t = net.advance(t + 5.0 * TAU)
    # 稳态判定: 再推进 10 tau, 组分变化 < 0.1%
    x_prev = reactor.thermo["NH3"].X[0]
    t = net.advance(t + 10.0 * TAU)
    if abs(reactor.thermo["NH3"].X[0] - x_prev) / max(x_prev, 1e-30) > 0.01:
        t = net.advance(t + 40.0 * TAU)
    return reactor.thermo


def sweep():
    mech_dir = Path(__file__).resolve().parent.parent / "mechanisms"
    rows = []
    for mech_name, mech_file in MECHS.items():
        gas = ct.Solution(str(mech_dir / mech_file))
        print(f"=== {mech_name} ===", flush=True)
        for series in JSR_SERIES:
            comp = inlet_composition(series)
            phi = (0.75 * comp["NH3"] + 0.5 * comp["H2"]) / comp["O2"]
            print(f"  XH2={series['XH2']} (phi={phi:.3f}), "
                  f"inlet NH3={comp['NH3']*1e6:.0f} ppm, "
                  f"H2={comp['H2']*1e6:.1f} ppm", flush=True)
            for T in T_RANGE:
                try:
                    th = jsr_steady_state(gas, float(T), comp)
                    rec = {
                        "mechanism": mech_name,
                        "XH2_fuel": series["XH2"], "phi": phi,
                        "P_atm": P_ATM, "T_K": float(T),
                        "X_NH3": th["NH3"].X[0],
                        "X_H2O": th["H2O"].X[0],
                        "X_NO": th["NO"].X[0],
                        "X_N2O": th["N2O"].X[0],
                        "NH3_ppm": th["NH3"].X[0] * 1e6,
                        "NO_ppm": th["NO"].X[0] * 1e6,
                        "N2O_ppm": th["N2O"].X[0] * 1e6,
                    }
                except Exception as e:
                    print(f"    T={T} FAILED: {e}")
                    rec = {"mechanism": mech_name,
                           "XH2_fuel": series["XH2"], "T_K": float(T),
                           "X_NH3": np.nan, "X_H2O": np.nan,
                           "X_NO": np.nan, "X_N2O": np.nan}
                rows.append(rec)
        print(f"  {mech_name} JSR done", flush=True)

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "jsr_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nJSR 计算完成: {len(df)} 行 -> {out}")


if __name__ == "__main__":
    sweep()
