import cantera as ct
import numpy as np
import pandas as pd
from pathlib import Path

R = 8.31446261815324


def oxidizer_for_phi(fuel_moles, phi, diluent="N2", n2_o2=3.76):
    """
    fuel_moles: dict, e.g. {"NH3": 0.7, "H2": 0.3}
    phi: equivalence ratio
    化学计量：
      NH3 + 0.75 O2 -> 0.5 N2 + 1.5 H2O
      H2  + 0.5  O2 -> H2O
    diluent: "N2" for air, "AR" for Ar-diluted mixtures
    n2_o2: N2/O2 ratio (3.76 for air); for Ar diluent, this is Ar/O2 ratio
    """
    o2_stoich = 0.0
    o2_stoich += fuel_moles.get("NH3", 0.0) * 0.75
    o2_stoich += fuel_moles.get("H2", 0.0) * 0.5
    o2_actual = o2_stoich / phi
    comp = dict(fuel_moles)
    comp["O2"] = o2_actual
    comp[diluent] = o2_actual * n2_o2
    return comp


def ignition_delay(
        mech,
        T,
        P_atm,
        fuel_H2_frac=0.0,
        phi=1.0,
        t_end=0.1,
        dt_max=1e-5,
        criterion="OH_peak",
        diluent="N2",
        n2_o2=3.76,
):
    gas = ct.Solution(mech)
    fuel = {"NH3": 1.0 - fuel_H2_frac}
    if fuel_H2_frac > 0:
        fuel["H2"] = fuel_H2_frac
    X = oxidizer_for_phi(fuel, phi, diluent=diluent, n2_o2=n2_o2)
    gas.TPX = T, P_atm * ct.one_atm, X
    reactor = ct.IdealGasReactor(gas)
    net = ct.ReactorNet([reactor])

    times = []
    temps = []
    oh = []
    t = 0.0

    while t < t_end:
        t = net.step()
        times.append(t)
        temps.append(reactor.T)
        oh.append(reactor.thermo["OH"].X[0])
        if reactor.T > T + 1500:
            break

    times = np.array(times)
    temps = np.array(temps)
    oh = np.array(oh)

    if len(times) < 5:
        return np.nan

    if criterion == "OH_peak":
        tau = times[np.argmax(oh)]
    elif criterion == "dTdt_peak":
        dTdt = np.gradient(temps, times)
        tau = times[np.argmax(dTdt)]
    else:
        raise ValueError("criterion must be OH_peak or dTdt_peak")

    return tau


def sweep_idt():
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    MECH_DIR = PROJECT_ROOT / "mechanisms"
    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    mechs = {
        "Full": MECH_DIR / "Full mechanism.yaml",
        "Reduced": MECH_DIR / "Reduced mechanism.yaml",
        "Optimized": MECH_DIR / "Optimized mechanism.yaml",
    }

    # 论文 Table 3 的温度范围 - 不同 XH2 和 P 对应不同温度区间
    # Fig. 9: Chen et al. [42] shock tube, phi=1.0
    # Case 1: XH2=0, P=1 atm, T=1540-1850 K
    # Case 2: XH2=0, P=10 atm, T=1430-1820 K
    # Case 3: XH2=0.05, P=1 atm, T=1500-1920 K
    # Case 4: XH2=0.05, P=10 atm, T=1450-1920 K
    # Case 5: XH2=0.3, P=1 atm, T=1180-1820 K
    # Case 6: XH2=0.3, P=10 atm, T=1180-1920 K
    # Case 7: XH2=0.7, P=1 atm, T=1000-1540 K
    # Case 8: XH2=0.7, P=10 atm, T=1030-1300 K

    # 扩展温度范围以覆盖 Fig. 9 和 Fig. 10 的实验数据范围
    idt_conditions = [
        # Fig. 9: Chen et al. [42] - P=1.2 atm and 10 atm
        {"XH2": 0.0, "P_atm": 1.2, "T_range": np.linspace(1050, 2000, 20)},
        {"XH2": 0.0, "P_atm": 10.0, "T_range": np.linspace(1050, 2000, 20)},
        {"XH2": 0.05, "P_atm": 1.2, "T_range": np.linspace(1050, 2000, 20)},
        {"XH2": 0.05, "P_atm": 10.0, "T_range": np.linspace(1050, 2000, 20)},
        {"XH2": 0.3, "P_atm": 1.2, "T_range": np.linspace(1000, 2000, 20)},
        {"XH2": 0.3, "P_atm": 10.0, "T_range": np.linspace(1000, 2000, 20)},
        {"XH2": 0.7, "P_atm": 1.2, "T_range": np.linspace(900, 1800, 20)},
        {"XH2": 0.7, "P_atm": 10.0, "T_range": np.linspace(900, 1800, 20)},
        # Fig. 10: He et al. [44] - pure NH3, P=20 and 40 bar
        {"XH2": 0.0, "P_atm": 20.0, "T_range": np.linspace(1050, 2000, 20)},
        {"XH2": 0.0, "P_atm": 40.0, "T_range": np.linspace(1050, 2000, 20)},
        # Fig. 10: Mathieu et al. [17] - NH3/O2/Ar, P~1.4 atm
        # 论文: 3.43% NH3 / 0.857% O2 / 98% Ar (phi=1.0)
        # Ar/O2 = 98/0.857 ≈ 114.3 (高度稀释)
        {"XH2": 0.0, "P_atm": 1.4, "T_range": np.linspace(1050, 2000, 20),
         "phi": 0.5, "diluent": "AR", "n2_o2": 114.3},
        {"XH2": 0.0, "P_atm": 1.4, "T_range": np.linspace(1050, 2000, 20),
         "phi": 1.0, "diluent": "AR", "n2_o2": 114.3},
        {"XH2": 0.0, "P_atm": 1.4, "T_range": np.linspace(1050, 2000, 20),
         "phi": 2.0, "diluent": "AR", "n2_o2": 114.3},
        # Appendix Fig. S1: Mathieu & Petersen [1] - NH3/O2/Ar, P=1.4 and 11 atm
        # 1.143% NH3 / 0.857% O2 / 98% Ar (phi=1)
        {"XH2": 0.0, "P_atm": 1.4, "T_range": np.linspace(1500, 2500, 20),
         "phi": 1.0, "diluent": "AR", "n2_o2": 114.3, "label": "FigS1_1.4atm"},
        {"XH2": 0.0, "P_atm": 11.0, "T_range": np.linspace(1500, 2500, 20),
         "phi": 1.0, "diluent": "AR", "n2_o2": 114.3, "label": "FigS1_11atm"},
        # Appendix Fig. S2(a): Liao et al. [2] - phi=0.35, P=43.4 bar
        {"XH2": 0.0, "P_atm": 43.4, "T_range": np.linspace(900, 1300, 20),
         "phi": 0.35, "label": "FigS2a"},
        {"XH2": 0.1, "P_atm": 43.4, "T_range": np.linspace(900, 1300, 20),
         "phi": 0.35, "label": "FigS2a"},
        {"XH2": 0.25, "P_atm": 43.4, "T_range": np.linspace(900, 1300, 20),
         "phi": 0.35, "label": "FigS2a"},
        # Appendix Fig. S2(b): Pochet et al. [3] - phi=1, P=20 bar
        {"XH2": 0.0, "P_atm": 20.0, "T_range": np.linspace(900, 1400, 20),
         "phi": 1.0, "label": "FigS2b"},
        {"XH2": 0.1, "P_atm": 20.0, "T_range": np.linspace(900, 1400, 20),
         "phi": 1.0, "label": "FigS2b"},
        {"XH2": 0.25, "P_atm": 20.0, "T_range": np.linspace(900, 1400, 20),
         "phi": 1.0, "label": "FigS2b"},
    ]

    rows = []
    for mech_name, mech_path in mechs.items():
        for cond in idt_conditions:
            XH2 = cond["XH2"]
            P = cond["P_atm"]
            phi = cond.get("phi", 1.0)
            diluent = cond.get("diluent", "N2")
            n2_o2 = cond.get("n2_o2", 3.76)
            for T in cond["T_range"]:
                # 根据温度调整模拟时间
                t_end = 10.0 if T < 1200 else (1.0 if T < 1500 else 0.1)
                tau = ignition_delay(
                    mech_path, T=T, P_atm=P,
                    fuel_H2_frac=XH2,
                    phi=phi,
                    t_end=t_end,
                    diluent=diluent,
                    n2_o2=n2_o2,
                )
                rows.append({
                    "mechanism": mech_name,
                    "XH2_fuel": XH2,
                    "P_atm": P,
                    "T_K": T,
                    "1000/T": 1000.0 / T,
                    "phi": phi,
                    "diluent": diluent,
                    "label": cond.get("label", ""),
                    "IDT_s": tau,
                    "IDT_ms": tau * 1000.0,
                })

    df = pd.DataFrame(rows)
    output_file = RESULTS_DIR / "idt_sweep.csv"
    df.to_csv(output_file, index=False)

    print(f"IDT 计算完成，结果已保存至：{output_file}")
    print(df.head())


if __name__ == "__main__":
    sweep_idt()
