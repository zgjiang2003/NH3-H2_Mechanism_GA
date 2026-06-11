"""
IDT 敏感性分析 - 使用 Cantera 内置敏感性分析
复现论文中关于点火延迟时间的关键反应敏感性分析
"""
import cantera as ct
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MECH_DIR = PROJECT_ROOT / "mechanisms"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def compute_idt_sensitivity(mech_path, T, P_atm, fuel_H2_frac=0.0, phi=1.0, t_end=0.1):
    """使用 Cantera 内置敏感性分析计算 IDT 对各反应的敏感性"""
    gas = ct.Solution(str(mech_path))
    fuel = {"NH3": 1.0 - fuel_H2_frac}
    if fuel_H2_frac > 0:
        fuel["H2"] = fuel_H2_frac
    o2_stoich = fuel.get("NH3", 0.0) * 0.75 + fuel.get("H2", 0.0) * 0.5
    o2_actual = o2_stoich / phi
    X = dict(fuel)
    X["O2"] = o2_actual
    X["N2"] = o2_actual * 3.76
    gas.TPX = T, P_atm * ct.one_atm, X

    reactor = ct.IdealGasReactor(gas)
    net = ct.ReactorNet([reactor])

    # 添加所有反应的敏感性
    n_reactions = gas.n_reactions
    for i in range(n_reactions):
        reactor.add_sensitivity_reaction(i)

    # 设置敏感性参数 - 跟踪 OH 浓度
    reactor.add_sensitivity_species_enthalpy("OH")

    # 计算 IDT 并获取敏感性
    t = 0.0
    oh_max = 0.0
    idt = np.nan
    while t < t_end:
        t = net.step()
        oh = reactor.thermo["OH"].X[0]
        if oh > oh_max:
            oh_max = oh
            idt = t
        if reactor.T > T + 1500:
            print(f"    Ignition detected at t={t:.6f} s, T={reactor.T:.1f} K")
            break

    if np.isnan(idt):
        print(f"    No ignition within {t_end} s")
        return None, None

    # 获取敏感性系数
    sens = net.sensitivities()
    if sens is None or len(sens) == 0:
        print(f"    No sensitivity data available")
        return idt, None

    # 敏感性系数 shape: (n_vars, n_params)
    # n_vars = 1 (mass) + 1 (volume) + 1 (temperature) + n_species
    # OH 的索引 = 3 + species_index("OH")
    try:
        oh_idx = gas.species_index("OH")
    except Exception:
        oh_idx = None
        for i, sp in enumerate(gas.species_names):
            if sp == "OH":
                oh_idx = i
                break

    if oh_idx is None:
        print(f"    OH species not found")
        return idt, None

    state_idx = 3 + oh_idx  # mass, volume, temperature, then species
    oh_sens = sens[state_idx, :]  # OH sensitivity to all parameters

    # 敏感性系数 shape: (n_reactions,)
    sensitivities = {}
    for i in range(min(n_reactions, len(oh_sens))):
        try:
            eq = gas.reaction(i).equation
            sensitivities[eq] = float(oh_sens[i])
        except Exception:
            pass

    return idt, sensitivities


if __name__ == "__main__":
    mechs = {
        "Full": MECH_DIR / "Full mechanism.yaml",
        "Reduced": MECH_DIR / "Reduced mechanism.yaml",
        "Optimized": MECH_DIR / "Optimized mechanism.yaml",
    }

    # 论文 Table 3 Case 1-8: 不同 XH2 和 P 的条件
    sensitivity_conditions = [
        # Case 1: XH2=0, P=1 atm, T=1540-1850 K (取中值 ~1700K)
        {"XH2": 0.0, "P_atm": 1.0, "T": 1700.0},
        # Case 2: XH2=0, P=10 atm, T=1430-1820 K (取中值 ~1600K)
        {"XH2": 0.0, "P_atm": 10.0, "T": 1600.0},
        # Case 3: XH2=0.05, P=1 atm, T=1500-1920 K (取中值 ~1700K)
        {"XH2": 0.05, "P_atm": 1.0, "T": 1700.0},
        # Case 4: XH2=0.05, P=10 atm, T=1450-1920 K (取中值 ~1700K)
        {"XH2": 0.05, "P_atm": 10.0, "T": 1700.0},
        # Case 5: XH2=0.3, P=1 atm, T=1180-1820 K (取中值 ~1500K)
        {"XH2": 0.3, "P_atm": 1.0, "T": 1500.0},
        # Case 6: XH2=0.3, P=10 atm, T=1180-1920 K (取中值 ~1550K)
        {"XH2": 0.3, "P_atm": 10.0, "T": 1550.0},
        # Case 7: XH2=0.7, P=1 atm, T=1000-1540 K (取中值 ~1270K)
        {"XH2": 0.7, "P_atm": 1.0, "T": 1270.0},
        # Case 8: XH2=0.7, P=10 atm, T=1030-1300 K (取中值 ~1165K)
        {"XH2": 0.7, "P_atm": 10.0, "T": 1165.0},
    ]

    all_rows = []

    for cond in sensitivity_conditions:
        T = cond["T"]
        P_atm = cond["P_atm"]
        XH2 = cond["XH2"]
        t_end = 10.0 if T < 1200 else (1.0 if T < 1500 else 0.1)

        for mech_name, mech_path in mechs.items():
            print(f"Computing sensitivity: {mech_name}, XH2={XH2}, P={P_atm} atm, T={T} K...")
            idt, sens = compute_idt_sensitivity(
                mech_path, T=T, P_atm=P_atm,
                fuel_H2_frac=XH2, phi=1.0, t_end=t_end
            )
            if sens is None:
                print(f"  Failed for {mech_name}")
                continue

            print(f"  IDT = {idt * 1000:.3f} ms")
            # 取 top 10 最敏感的反应
            sorted_sens = sorted(sens.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            for eq, s in sorted_sens:
                all_rows.append({
                    "mechanism": mech_name,
                    "T_K": T,
                    "P_atm": P_atm,
                    "XH2_fuel": XH2,
                    "reaction": eq,
                    "sensitivity_raw": s,
                    "sensitivity_normalized": s,  # 后面归一化
                })

    # 归一化: 每个工况下的敏感性系数除以该工况下最大绝对值
    df = pd.DataFrame(all_rows)
    if len(df) > 0:
        for (T, P, XH2), group in df.groupby(["T_K", "P_atm", "XH2_fuel"]):
            max_abs = group["sensitivity_raw"].abs().max()
            if max_abs > 0:
                df.loc[group.index, "sensitivity_normalized"] = group["sensitivity_raw"] / max_abs

    output_file = RESULTS_DIR / "sensitivity.csv"
    df.to_csv(output_file, index=False)
    print(f"\n敏感性分析完成，{len(df)} 行结果已保存至：{output_file}")