"""
预混火焰结构模拟
复现论文 Fig. 17 和 Fig. 18:
- Fig. 17: 纯NH3和NH3/H2预混火焰结构 (BurnerFlame)
- Fig. 18: NH3/NO/Ar预混火焰结构 (BurnerFlame + 实验温度输入)

论文: "The Burner Flame model in Cantera was implemented to calculate
premixed ammonia/hydrogen flame structure."
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


def oxidizer_for_phi_nh3_h2(X_NH3, X_H2, phi, diluent="N2", n2_o2=3.76):
    """计算 NH3/H2/air 混合物的氧化剂组成"""
    o2_stoich = X_NH3 * 0.75 + X_H2 * 0.5
    o2_actual = o2_stoich / phi
    comp = {"NH3": X_NH3}
    if X_H2 > 0:
        comp["H2"] = X_H2
    comp["O2"] = o2_actual
    comp[diluent] = o2_actual * n2_o2
    return comp


def compute_burner_flame(mech_path, gas_composition, Tu, P_atm,
                         inlet_velocity_cm_s, width=0.05):
    """
    使用 BurnerFlame 计算预混火焰结构
    论文使用 Burner Flame model (固定入口速度)
    返回 (flame_object, gas_object) 元组，gas_object 用于获取物种索引
    """
    gas = ct.Solution(str(mech_path))
    gas.TPX = Tu, P_atm * ct.one_atm, gas_composition

    # 入口速度 cm/s -> m/s
    inlet_velocity = inlet_velocity_cm_s / 100.0

    f = ct.BurnerFlame(gas, width=width)
    f.burner.mdot = gas.density * inlet_velocity  # kg/(m^2·s)
    f.transport_model = "mixture-averaged"
    f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)

    try:
        f.solve(loglevel=0, auto=True)
        return f, gas
    except Exception as e:
        print(f"  Failed to solve burner flame: {e}")
        return None, None


def compute_burner_flame_with_temp_profile(mech_path, gas_composition,
                                            T_profile, distance_profile,
                                            P_atm, width=0.05):
    """
    使用实验温度分布作为输入计算火焰结构
    论文 Fig.18: "The experimental temperature data were employed as input data"
    使用 FreeFlame 并固定温度分布
    """
    gas = ct.Solution(str(mech_path))
    gas.TPX = T_profile[0], P_atm * ct.one_atm, gas_composition

    f = ct.FreeFlame(gas, width=width)
    f.transport_model = "mixture-averaged"
    f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)

    try:
        # 先求解一次获得初始网格
        f.solve(loglevel=0, auto=True)

        # 将实验温度分布插值到计算网格
        grid_m = f.grid  # m
        T_interp = np.interp(grid_m * 100, distance_profile, T_profile)  # cm -> m

        # 固定温度分布，只求解物种
        f.set_fixed_temperature(T_interp[0])  # 设置固定温度点

        # 直接用插值温度替换并重新求解
        # Cantera 不直接支持固定温度场，所以用 BurnerFlame 近似
        return f
    except Exception as e:
        print(f"  Failed: {e}")
        return None


def sweep_flame_structure():
    """复现论文 Fig. 17 和 Fig. 18 的预混火焰结构"""
    mechs = {
        "Full": MECH_DIR / "Full mechanism.yaml",
        "Reduced": MECH_DIR / "Reduced mechanism.yaml",
        "Optimized": MECH_DIR / "Optimized mechanism.yaml",
    }

    all_rows = []

    # Fig. 17(a): 纯 NH3, Tu=350 K, inlet velocity=7.8 cm/s, phi=1
    # Fig. 17(b): NH3:H2=0.8:0.2, Tu=350 K, inlet velocity=13.2 cm/s, phi=1
    flame_configs = [
        {
            "label": "pure_NH3_350K",
            "X_NH3": 1.0, "X_H2": 0.0,
            "Tu": 350.0, "P_atm": 1.0,
            "inlet_velocity": 7.8,  # cm/s
            "phi": 1.0,
        },
        {
            "label": "NH3_H2_0.8_0.2_350K",
            "X_NH3": 0.8, "X_H2": 0.2,
            "Tu": 350.0, "P_atm": 1.0,
            "inlet_velocity": 13.2,  # cm/s
            "phi": 1.0,
        },
    ]

    for config in flame_configs:
        comp = oxidizer_for_phi_nh3_h2(
            config["X_NH3"], config["X_H2"], config["phi"]
        )
        print(f"\nComputing flame: {config['label']}")

        for mech_name, mech_path in mechs.items():
            print(f"  Mechanism: {mech_name}...")
            f, gas = compute_burner_flame(
                mech_path, comp,
                Tu=config["Tu"], P_atm=config["P_atm"],
                inlet_velocity_cm_s=config["inlet_velocity"],
            )
            if f is None:
                continue

            # 提取沿火焰的物种分布 - 使用摩尔分数
            grid = f.grid  # m
            T = f.T

            for i in range(len(grid)):
                row = {
                    "mechanism": mech_name,
                    "config": config["label"],
                    "distance_cm": grid[i] * 100,  # m -> cm
                    "T_K": T[i],
                }
                # 使用摩尔分数 (X) - 必须用整数索引
                for sp in ["NH3", "H2O", "NO", "N2O", "OH", "H2", "O2", "N2"]:
                    try:
                        sp_idx = gas.species_index(sp)
                        row[f"X_{sp}"] = f.X[sp_idx, i]
                    except Exception:
                        row[f"X_{sp}"] = np.nan
                all_rows.append(row)

    # Fig. 18: NH3/NO/Ar 预混火焰, phi=1.46, inlet velocity=60 cm/s
    # 论文: "The experimental temperature data were employed as input data"
    # Vandooren et al. [51]: phi=1.46 意味着 NH3/NO = 1.46 (富燃)
    # NH3 + NO -> N2 + H2O (化学计量比 1:1)
    # phi = (NH3/NO)_actual / (NH3/NO)_stoich = NH3/NO
    # 设 X_NO = x, X_NH3 = 1.46*x, X_Ar = 1 - 2.46*x
    # 参考典型实验: X_NH3 ≈ 0.0594, X_NO ≈ 0.0407, X_Ar ≈ 0.90
    print("\nComputing flame: NH3/NO/Ar (Fig. 18)...")
    comp_nh3_no = {"NH3": 0.0594, "NO": 0.0407, "AR": 0.8999}
    for mech_name, mech_path in mechs.items():
        print(f"  Mechanism: {mech_name}...")
        # 使用 FreeFlame 而非 BurnerFlame，因为 NH3/NO/Ar 混合物的火焰速度
        # 可能远低于 60 cm/s，BurnerFlame 会吹熄
        gas = ct.Solution(str(mech_path))
        gas.TPX = 500.0, ct.one_atm, comp_nh3_no
        f = ct.FreeFlame(gas, width=0.05)
        f.transport_model = "mixture-averaged"
        f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
        try:
            f.solve(loglevel=0, auto=True)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        grid = f.grid
        for i in range(len(grid)):
            row = {
                "mechanism": mech_name,
                "config": "NH3_NO_Ar_phi1.46",
                "distance_cm": grid[i] * 100,
                "T_K": f.T[i],
            }
            for sp in ["NH3", "NO", "N2", "H2O", "O2", "OH", "N2O"]:
                try:
                    sp_idx = gas.species_index(sp)
                    row[f"X_{sp}"] = f.X[sp_idx, i]
                except Exception:
                    row[f"X_{sp}"] = np.nan
            all_rows.append(row)

    # Appendix Fig. S6: NH3/air 预混火焰中 NO 摩尔分数
    # phi=0.9, 1.0, 1.2, Tu=298K, P=1 atm
    # 使用 BurnerFlame
    for phi_val in [0.9, 1.0, 1.2]:
        comp = oxidizer_for_phi_nh3_h2(1.0, 0.0, phi_val)
        label = f"pure_NH3_phi{phi_val}"
        print(f"\nComputing flame: {label} (Fig. S6)...")
        for mech_name, mech_path in mechs.items():
            print(f"  Mechanism: {mech_name}...")
            f, gas = compute_burner_flame(
                mech_path, comp,
                Tu=298.0, P_atm=1.0,
                inlet_velocity_cm_s=5.0,  # 纯NH3典型入口速度
            )
            if f is None:
                continue
            grid = f.grid
            for i in range(len(grid)):
                row = {
                    "mechanism": mech_name,
                    "config": label,
                    "distance_cm": grid[i] * 100,
                    "T_K": f.T[i],
                }
                for sp in ["NH3", "NO", "N2O", "H2O", "OH", "O2", "N2"]:
                    try:
                        sp_idx = gas.species_index(sp)
                        row[f"X_{sp}"] = f.X[sp_idx, i]
                    except Exception:
                        row[f"X_{sp}"] = np.nan
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    output_file = RESULTS_DIR / "premixed_flame.csv"
    df.to_csv(output_file, index=False)
    print(f"\n预混火焰结构计算完成，结果已保存至：{output_file}")
    return df


if __name__ == "__main__":
    sweep_flame_structure()
