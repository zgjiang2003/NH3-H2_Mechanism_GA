"""
JSR（Jet-Stirred Reactor）物种浓度模拟
复现论文 Fig. 16 - Zhang et al. (2021) JSR 实验数据对比

使用 Cantera 的 IdealGasConstPressureReactor + MassFlowController
构建正确的 JSR 模型（带停留时间）
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


def oxidizer_for_phi(fuel_moles, phi, diluent="N2", n2_o2=3.76):
    o2_stoich = 0.0
    o2_stoich += fuel_moles.get("NH3", 0.0) * 0.75
    o2_stoich += fuel_moles.get("H2", 0.0) * 0.5
    o2_actual = o2_stoich / phi
    comp = dict(fuel_moles)
    comp["O2"] = o2_actual
    comp[diluent] = o2_actual * n2_o2
    return comp


def jsr_species_profile(mech_path, T_range, P_atm, tau, fuel_H2_frac, phi=1.0):
    """
    使用 Cantera 构建正确的 JSR 模型：
    - inlet reservoir -> MassFlowController -> reactor -> PressureController -> exhaust reservoir
    - 停留时间 tau 决定质量流量
    """
    gas = ct.Solution(str(mech_path))
    fuel = {"NH3": 1.0 - fuel_H2_frac}
    if fuel_H2_frac > 0:
        fuel["H2"] = fuel_H2_frac
    X = oxidizer_for_phi(fuel, phi)

    species_data = {"T": [], "NH3": [], "H2O": [], "NO": [], "N2O": []}

    for T in T_range:
        # 入口气体状态
        inlet_gas = ct.Solution(str(mech_path))
        inlet_gas.TPX = T, P_atm * ct.one_atm, X

        # 反应器气体 - 初始设为入口状态
        gas.TPX = T, P_atm * ct.one_atm, X

        # 创建 JSR 网络
        inlet = ct.Reservoir(inlet_gas)
        reactor = ct.IdealGasConstPressureReactor(gas)
        exhaust = ct.Reservoir(gas)

        # 质量流量 = 反应器质量 / 停留时间
        # 反应器体积设为 1 cm^3 = 1e-6 m^3 (典型 JSR 体积)
        reactor.volume = 1.0e-6  # m^3
        mdot = reactor.mass / tau  # kg/s

        # 连接: inlet -> reactor (质量流量控制器)
        inlet_mfc = ct.MassFlowController(inlet, reactor, mdot=mdot)

        # 连接: reactor -> exhaust (压力控制器，保持压力平衡)
        outlet = ct.PressureController(reactor, exhaust, primary=inlet_mfc, K=0.01)

        net = ct.ReactorNet([reactor])

        # 推进到稳态
        try:
            # 多次推进确保达到稳态
            for _ in range(200):
                net.advance(net.time + 5.0 * tau)
            # 再推进一段时间确认稳态
            net.advance(net.time + 10.0 * tau)

            species_data["T"].append(T)
            species_data["NH3"].append(reactor.thermo["NH3"].X[0])
            species_data["H2O"].append(reactor.thermo["H2O"].X[0])
            species_data["NO"].append(reactor.thermo["NO"].X[0])
            species_data["N2O"].append(reactor.thermo["N2O"].X[0])
        except Exception as e:
            print(f"  Failed at T={T} K: {e}")
            species_data["T"].append(T)
            species_data["NH3"].append(np.nan)
            species_data["H2O"].append(np.nan)
            species_data["NO"].append(np.nan)
            species_data["N2O"].append(np.nan)

    return species_data


def sweep_jsr():
    """复现 Fig. 16: JSR 中 NH3, H2O, N2O, NO 浓度随温度变化"""
    mechs = {
        "Full": MECH_DIR / "Full mechanism.yaml",
        "Reduced": MECH_DIR / "Reduced mechanism.yaml",
        "Optimized": MECH_DIR / "Optimized mechanism.yaml",
    }

    # Zhang et al. (2021) 条件: phi=1.0, P=1 atm, tau=2 s, 不同 H2 含量
    T_range = np.linspace(800, 1300, 51)
    P_atm = 1.0
    tau = 2.0  # 停留时间 (s)

    all_rows = []

    for mech_name, mech_path in mechs.items():
        for XH2 in [0.0, 0.05, 0.3]:  # 不同 H2 掺混比例
            print(f"Computing JSR: {mech_name}, XH2={XH2}...")
            data = jsr_species_profile(
                mech_path, T_range, P_atm, tau,
                fuel_H2_frac=XH2, phi=1.0
            )
            for i in range(len(data["T"])):
                all_rows.append({
                    "mechanism": mech_name,
                    "XH2_fuel": XH2,
                    "P_atm": P_atm,
                    "T_K": data["T"][i],
                    "X_NH3": data["NH3"][i],
                    "X_H2O": data["H2O"][i],
                    "X_NO": data["NO"][i],
                    "X_N2O": data["N2O"][i],
                })

    df = pd.DataFrame(all_rows)
    output_file = RESULTS_DIR / "jsr_sweep.csv"
    df.to_csv(output_file, index=False)
    print(f"\nJSR 计算完成，结果已保存至：{output_file}")
    return df


if __name__ == "__main__":
    sweep_jsr()
