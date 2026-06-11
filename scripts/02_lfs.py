import cantera as ct
import numpy as np
import pandas as pd
from pathlib import Path


def oxidizer_for_phi(fuel_moles, phi, diluent="N2", n2_o2=3.76):
    o2_stoich = 0.0
    o2_stoich += fuel_moles.get("NH3", 0.0) * 0.75
    o2_stoich += fuel_moles.get("H2", 0.0) * 0.5
    o2_actual = o2_stoich / phi
    comp = dict(fuel_moles)
    comp["O2"] = o2_actual
    comp[diluent] = o2_actual * n2_o2
    return comp


def compute_lfs(mech, Tu, P_atm, fuel_H2_frac=0.0, phi=1.0, width=0.03):
    gas = ct.Solution(str(mech))
    fuel = {"NH3": 1.0 - fuel_H2_frac}
    if fuel_H2_frac > 0:
        fuel["H2"] = fuel_H2_frac

    X = oxidizer_for_phi(fuel, phi)
    gas.TPX = Tu, P_atm * ct.one_atm, X

    f = ct.FreeFlame(gas, width=width)
    f.transport_model = "mixture-averaged"

    f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)

    try:
        f.solve(loglevel=0, auto=True)
        Su = f.velocity[0]
    except Exception as e:
        print(f"Failed: Tu={Tu} P={P_atm} XH2={fuel_H2_frac} phi={phi}: {e}")
        Su = float('nan')

    return Su


def sweep_lfs():
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

    rows = []

    # -------------------------------------------------------
    # 1) Fig. 11/12: LFS vs phi, Tu=298K, P=1 atm, 各种 XH2
    #    论文 Table 3 case 9: XH2=0, P=1, T=423K, phi=0.7-1.4
    #    论文 Fig.11: XH2=0,0.2,0.4,0.6,0.8,1.0, Tu=298K, P=1atm
    # -------------------------------------------------------
    for mech_name, mech_path in mechs.items():
        for XH2 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            for phi in np.arange(0.7, 1.5, 0.1):
                Su = compute_lfs(mech_path, Tu=298.0, P_atm=1.0,
                                 fuel_H2_frac=XH2, phi=round(phi, 2))
                rows.append({
                    "mechanism": mech_name,
                    "XH2_fuel": XH2,
                    "P_atm": 1.0,
                    "T_unburned_K": 298.0,
                    "phi": round(phi, 2),
                    "Su_m_s": Su,
                })
                print(f"  {mech_name} XH2={XH2} phi={phi:.1f} Su={Su:.4f}" if not np.isnan(Su) else f"  {mech_name} XH2={XH2} phi={phi:.1f} FAILED")

    # -------------------------------------------------------
    # 2) Fig. 13: LFS vs XH2, phi=1.0, Tu=298K, P=1,3,5 atm
    #    论文 Table 3 case 11: XH2=0-1.0, P=1, T=298, phi=1
    # -------------------------------------------------------
    for mech_name, mech_path in mechs.items():
        for P in [1.0, 3.0, 5.0]:
            for XH2 in np.arange(0.0, 1.05, 0.1):
                Su = compute_lfs(mech_path, Tu=298.0, P_atm=P,
                                 fuel_H2_frac=round(XH2, 2), phi=1.0)
                rows.append({
                    "mechanism": mech_name,
                    "XH2_fuel": round(XH2, 2),
                    "P_atm": P,
                    "T_unburned_K": 298.0,
                    "phi": 1.0,
                    "Su_m_s": Su,
                })

    # -------------------------------------------------------
    # 3) Fig. 14: LFS vs P, XH2=0.4, Tu=298K, phi=0.7-1.5
    #    论文 Table 3 case 10: XH2=0.4, P=1-5, T=298, phi=0.7-1.5
    # -------------------------------------------------------
    for mech_name, mech_path in mechs.items():
        for P in [1.0, 3.0, 5.0]:
            for phi in np.arange(0.7, 1.55, 0.1):
                Su = compute_lfs(mech_path, Tu=298.0, P_atm=P,
                                 fuel_H2_frac=0.4, phi=round(phi, 2))
                rows.append({
                    "mechanism": mech_name,
                    "XH2_fuel": 0.4,
                    "P_atm": P,
                    "T_unburned_K": 298.0,
                    "phi": round(phi, 2),
                    "Su_m_s": Su,
                })

    # -------------------------------------------------------
    # 4) Fig. 15: LFS vs XH2, 不同 Tu, phi=0.8-1.4, P=1 atm
    #    论文 Table 3 case 12: XH2=0.05, P=1, T=298-473, phi=0.8-1.4
    #    论文 Fig.15 还包括 XH2=0-0.6 的数据
    # -------------------------------------------------------
    for mech_name, mech_path in mechs.items():
        for Tu in [298.0, 373.0, 423.0, 473.0]:
            for XH2 in [0.0, 0.05, 0.2, 0.4, 0.6]:
                for phi in np.arange(0.8, 1.45, 0.1):
                    Su = compute_lfs(mech_path, Tu=Tu, P_atm=1.0,
                                     fuel_H2_frac=XH2, phi=round(phi, 2))
                    rows.append({
                        "mechanism": mech_name,
                        "XH2_fuel": XH2,
                        "P_atm": 1.0,
                        "T_unburned_K": Tu,
                        "phi": round(phi, 2),
                        "Su_m_s": Su,
                    })

    # -------------------------------------------------------
    # 5) Appendix Fig. S3: 高温高压 LFS
    #    (a) 80%NH3/20%H2, P=40 bar, T=820K
    #    (b) 100%NH3, P=40 bar, T=720K
    #    注意: P=40 atm 的 FreeFlame 收敛极困难，跳过此条件
    #    如需计算，需使用更精细的网格和更小的初始步长
    # -------------------------------------------------------
    # 跳过 P=40 atm 条件 (FreeFlame 收敛困难)

    # -------------------------------------------------------
    # 6) Appendix Fig. S4: 360K下LFS随压力变化
    #    XH2=0, 0.1, 0.2, Tu=360K, P=1-5 atm
    # -------------------------------------------------------
    for mech_name, mech_path in mechs.items():
        for XH2 in [0.0, 0.1, 0.2]:
            for P in [1.0, 2.0, 3.0, 5.0]:
                for phi in np.arange(0.7, 1.5, 0.1):
                    Su = compute_lfs(mech_path, Tu=360.0, P_atm=P,
                                     fuel_H2_frac=XH2, phi=round(phi, 2))
                    rows.append({
                        "mechanism": mech_name, "XH2_fuel": XH2,
                        "P_atm": P, "T_unburned_K": 360.0,
                        "phi": round(phi, 2), "Su_m_s": Su,
                    })

    # -------------------------------------------------------
    # 7) Appendix Fig. S5: NH3/H2/N2/air 混合物 LFS
    #    alpha=0.3 and 0.5, Tu=303K, P=1 atm
    #    alpha = X_H2 / (X_NH3 + X_H2) in fuel+N2 blend
    # -------------------------------------------------------
    for mech_name, mech_path in mechs.items():
        for alpha in [0.3, 0.5]:
            for phi in np.arange(0.7, 1.5, 0.1):
                # NH3/H2/N2 blend: alpha = XH2/(XNH3+XH2)
                # fuel_H2_frac in fuel = alpha
                Su = compute_lfs(mech_path, Tu=303.0, P_atm=1.0,
                                 fuel_H2_frac=alpha, phi=round(phi, 2))
                rows.append({
                    "mechanism": mech_name, "XH2_fuel": alpha,
                    "P_atm": 1.0, "T_unburned_K": 303.0,
                    "phi": round(phi, 2), "Su_m_s": Su,
                })

    df = pd.DataFrame(rows)
    output_file = RESULTS_DIR / "lfs_sweep.csv"
    df.to_csv(output_file, index=False)
    print(f"\nLFS 计算完成，结果已保存至：{output_file}")
    print(f"共 {len(df)} 行数据")


if __name__ == "__main__":
    sweep_lfs()
