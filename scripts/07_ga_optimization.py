"""
GA 机理优化算法
复现论文 Section 2.3: GA-based mechanism optimization

算法流程:
1. 实数编码 (浮点数基因): 每个基因对应一个反应的 A 因子调整系数
2. 初始种群: 从简化机理 (Mechanism 1) 出发
3. 适应度函数: f_objective = f_tau + f_LFS + f_A + f_CPU
4. 锦标赛选择 + 精英保留
5. 单点交叉 (交叉率 0.3)
6. 双向变异 (变异率 0.06)
7. 运行 1000 代

优化目标:
- 调整简化机理中除氢气子机理和压力依赖反应之外的所有反应的指前因子 A
- A 因子变化范围: [0.1 * A_i, 10 * A_i] (不确定度因子 = 10)
"""
import cantera as ct
import numpy as np
import pandas as pd
import yaml
import copy
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MECH_DIR = PROJECT_ROOT / "mechanisms"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# GA 参数 (论文 Section 2.3)
# ============================================================
POP_SIZE = 30
N_GENERATIONS = 1000
CROSSOVER_RATE = 0.3
MUTATION_RATE = 0.06
TOURNAMENT_SIZE = 3
ELITISM = True

# A 因子调整范围
A_FACTOR_MIN = 0.1   # 不确定度因子下限
A_FACTOR_MAX = 10.0  # 不确定度因子上限

# 目标函数系数 (论文 Section 2.3)
SIGMA_TAU = 4.0
SIGMA_LFS = 4.0
SIGMA_A = 0.3
SIGMA_CPU = 6.0
LAMBDA_CPU = 4.0

# 简化阶段工况 (Table 3 Case 1-8, 用于 f_tau)
IDT_CONDITIONS = [
    {"XH2": 0.0, "P_atm": 1.0, "T_range": [1600, 1700, 1800]},
    {"XH2": 0.0, "P_atm": 10.0, "T_range": [1500, 1600, 1700]},
    {"XH2": 0.05, "P_atm": 1.0, "T_range": [1600, 1700, 1800]},
    {"XH2": 0.05, "P_atm": 10.0, "T_range": [1600, 1700, 1800]},
    {"XH2": 0.3, "P_atm": 1.0, "T_range": [1300, 1500, 1700]},
    {"XH2": 0.3, "P_atm": 10.0, "T_range": [1300, 1500, 1700]},
    {"XH2": 0.7, "P_atm": 1.0, "T_range": [1100, 1270, 1400]},
    {"XH2": 0.7, "P_atm": 10.0, "T_range": [1050, 1165, 1250]},
]

# 优化阶段工况 (Table 3 Case 9-12, 用于 f_LFS)
LFS_CONDITIONS = [
    # Case 9: XH2=0, P=1 atm, T=423K, phi=0.7-1.4
    {"XH2": 0.0, "P_atm": 1.0, "Tu": 423.0, "phi_range": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]},
    # Case 10: XH2=0.4, P=1-5 atm, T=298K, phi=0.7-1.5
    {"XH2": 0.4, "P_atm": 1.0, "Tu": 298.0, "phi_range": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]},
    {"XH2": 0.4, "P_atm": 3.0, "Tu": 298.0, "phi_range": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]},
    {"XH2": 0.4, "P_atm": 5.0, "Tu": 298.0, "phi_range": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]},
    # Case 11: XH2=0-1.0, P=1 atm, T=298K, phi=1.0
    {"XH2": 0.0, "P_atm": 1.0, "Tu": 298.0, "phi_range": [1.0]},
    {"XH2": 0.2, "P_atm": 1.0, "Tu": 298.0, "phi_range": [1.0]},
    {"XH2": 0.4, "P_atm": 1.0, "Tu": 298.0, "phi_range": [1.0]},
    {"XH2": 0.6, "P_atm": 1.0, "Tu": 298.0, "phi_range": [1.0]},
    {"XH2": 0.8, "P_atm": 1.0, "Tu": 298.0, "phi_range": [1.0]},
    {"XH2": 1.0, "P_atm": 1.0, "Tu": 298.0, "phi_range": [1.0]},
    # Case 12: XH2=0.05, P=1 atm, T=298-473K, phi=0.8-1.4
    {"XH2": 0.05, "P_atm": 1.0, "Tu": 298.0, "phi_range": [0.8, 1.0, 1.2, 1.4]},
    {"XH2": 0.05, "P_atm": 1.0, "Tu": 373.0, "phi_range": [0.8, 1.0, 1.2, 1.4]},
    {"XH2": 0.05, "P_atm": 1.0, "Tu": 423.0, "phi_range": [0.8, 1.0, 1.2, 1.4]},
    {"XH2": 0.05, "P_atm": 1.0, "Tu": 473.0, "phi_range": [0.8, 1.0, 1.2, 1.4]},
]


def oxidizer_for_phi(fuel_moles, phi, diluent="N2", n2_o2=3.76):
    o2_stoich = 0.0
    o2_stoich += fuel_moles.get("NH3", 0.0) * 0.75
    o2_stoich += fuel_moles.get("H2", 0.0) * 0.5
    o2_actual = o2_stoich / phi
    comp = dict(fuel_moles)
    comp["O2"] = o2_actual
    comp[diluent] = o2_actual * n2_o2
    return comp


def compute_idt(mech_path, T, P_atm, fuel_H2_frac=0.0, phi=1.0, t_end=0.1):
    """计算点火延迟时间"""
    try:
        gas = ct.Solution(str(mech_path))
        fuel = {"NH3": 1.0 - fuel_H2_frac}
        if fuel_H2_frac > 0:
            fuel["H2"] = fuel_H2_frac
        X = oxidizer_for_phi(fuel, phi)
        gas.TPX = T, P_atm * ct.one_atm, X

        reactor = ct.IdealGasReactor(gas)
        net = ct.ReactorNet([reactor])

        times, oh_vals = [], []
        t = 0.0
        while t < t_end:
            t = net.step()
            times.append(t)
            oh_vals.append(reactor.thermo["OH"].X[0])
            if reactor.T > T + 1500:
                break

        if len(times) < 5:
            return np.nan
        times = np.array(times)
        oh_vals = np.array(oh_vals)
        return times[np.argmax(oh_vals)]
    except Exception:
        return np.nan


def compute_lfs(mech_path, Tu, P_atm, fuel_H2_frac=0.0, phi=1.0):
    """计算层流火焰速度"""
    try:
        gas = ct.Solution(str(mech_path))
        fuel = {"NH3": 1.0 - fuel_H2_frac}
        if fuel_H2_frac > 0:
            fuel["H2"] = fuel_H2_frac
        X = oxidizer_for_phi(fuel, phi)
        gas.TPX = Tu, P_atm * ct.one_atm, X

        f = ct.FreeFlame(gas, width=0.03)
        f.transport_model = "mixture-averaged"
        f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
        f.solve(loglevel=0, auto=True)
        return f.velocity[0]
    except Exception:
        return np.nan


def identify_optimizable_reactions(mech_path):
    """
    识别可优化的反应:
    - 排除氢气子机理反应 (H2/O2 体系)
    - 排除压力依赖反应 (fall-off, three-body with pressure)
    """
    gas = ct.Solution(str(mech_path))
    h2_species = {"H2", "H", "O2", "O", "OH", "HO2", "H2O2", "H2O", "AR", "N2"}

    optimizable = []
    for i in range(gas.n_reactions):
        rxn = gas.reaction(i)

        # 检查是否为压力依赖反应
        rxn_type = rxn.reaction_type
        # Cantera reaction types: 1=elementary, 2=three-body, 4=falloff, etc.
        if rxn_type in [4, 5]:  # falloff, ChemicallyActivated
            continue

        # 检查是否属于氢气子机理
        # 氢气子机理: 只涉及 H2, H, O2, O, OH, HO2, H2O2, H2O 的反应
        reactants = set(rxn.reactants.keys())
        products = set(rxn.products.keys())
        all_species = reactants | products

        # 如果所有参与物种都在氢气子机理中，则排除
        if all_species.issubset(h2_species):
            continue

        optimizable.append(i)

    return optimizable


def apply_a_factors(mech_path, a_factors, optimizable_indices, output_path):
    """将 A 因子调整系数应用到机理文件"""
    with open(mech_path, 'r') as f:
        mech_data = yaml.safe_load(f)

    reactions = mech_data.get('reactions', [])
    for idx, a_factor in zip(optimizable_indices, a_factors):
        if idx < len(reactions):
            rxn = reactions[idx]
            if 'rate-constant' in rxn:
                rxn['rate-constant']['A'] *= a_factor

    with open(output_path, 'w') as f:
        yaml.dump(mech_data, f, default_flow_style=False, allow_unicode=True)


def tournament_selection(population, fitness, tournament_size=TOURNAMENT_SIZE):
    """锦标赛选择"""
    indices = np.random.choice(len(population), tournament_size, replace=False)
    best_idx = indices[np.argmin([fitness[i] for i in indices])]
    return population[best_idx].copy()


def single_point_crossover_fp(parent1, parent2):
    """单点交叉 (浮点数编码)"""
    if np.random.random() < CROSSOVER_RATE:
        point = np.random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()


def bidirectional_mutation_fp(gene):
    """双向变异 (浮点数编码): 随机增大或减小 A 因子"""
    mutated = gene.copy()
    for i in range(len(mutated)):
        if np.random.random() < MUTATION_RATE:
            # 在 [A_FACTOR_MIN, A_FACTOR_MAX] 范围内随机取值
            mutated[i] = np.random.uniform(A_FACTOR_MIN, A_FACTOR_MAX)
    return mutated


def run_ga_optimization():
    """运行 GA 机理优化"""
    reduced_mech_path = MECH_DIR / "Reduced mechanism.yaml"
    gas = ct.Solution(str(reduced_mech_path))
    print(f"Reduced mechanism: {gas.n_species} species, {gas.n_reactions} reactions")

    # Step 1: 识别可优化的反应
    optimizable_indices = identify_optimizable_reactions(str(reduced_mech_path))
    n_optimizable = len(optimizable_indices)
    print(f"Optimizable reactions: {n_optimizable} (excluding H2 sub-mechanism and pressure-dependent)")

    # 打印被排除的反应
    for i in range(gas.n_reactions):
        if i not in optimizable_indices:
            rxn = gas.reaction(i)
            print(f"  Excluded R{i}: {rxn.equation}")

    # Step 2: 计算参考值 (简化机理的 IDT 和 LFS)
    print("\nComputing reference IDT values from reduced mechanism...")
    ref_idt = {}
    for cond in IDT_CONDITIONS:
        XH2 = cond["XH2"]
        P = cond["P_atm"]
        for T in cond["T_range"]:
            key = ("IDT", XH2, P, T)
            t_end = 10.0 if T < 1200 else (1.0 if T < 1500 else 0.1)
            ref_idt[key] = compute_idt(str(reduced_mech_path), T, P, fuel_H2_frac=XH2, t_end=t_end)
            print(f"  XH2={XH2}, P={P} atm, T={T} K: IDT={ref_idt[key]*1000:.3f} ms")

    print("\nComputing reference LFS values from reduced mechanism...")
    ref_lfs = {}
    for cond in LFS_CONDITIONS:
        XH2 = cond["XH2"]
        P = cond["P_atm"]
        Tu = cond["Tu"]
        for phi in cond["phi_range"]:
            key = ("LFS", XH2, P, Tu, phi)
            ref_lfs[key] = compute_lfs(str(reduced_mech_path), Tu, P, fuel_H2_frac=XH2, phi=phi)
            print(f"  XH2={XH2}, P={P} atm, Tu={Tu} K, phi={phi}: LFS={ref_lfs[key]*100:.2f} cm/s")

    # 计算参考 CPU 时间
    t_start = time.time()
    compute_idt(str(reduced_mech_path), 1500, 1.0, fuel_H2_frac=0.0)
    compute_lfs(str(reduced_mech_path), 298, 1.0, fuel_H2_frac=0.4, phi=1.0)
    ref_cpu_time = time.time() - t_start

    # Step 3: 初始化种群
    # 基因为 A 因子调整系数, 初始值全部为 1.0 (即不调整)
    print(f"\nInitializing population: {POP_SIZE} individuals, {n_optimizable} genes each")
    population = [np.ones(n_optimizable) for _ in range(POP_SIZE)]

    # 对部分个体添加随机扰动
    for i in range(1, POP_SIZE):
        for j in range(n_optimizable):
            population[i][j] = np.random.uniform(A_FACTOR_MIN, A_FACTOR_MAX)

    # 记录进化过程
    history = {
        "generation": [],
        "best_fitness": [],
        "worst_fitness": [],
        "avg_fitness": [],
    }

    # Step 4: GA 主循环
    print(f"\nStarting GA optimization: {N_GENERATIONS} generations")
    temp_path = RESULTS_DIR / "_temp_opt_mech.yaml"

    for gen in range(N_GENERATIONS):
        fitness = []
        for i, a_factors in enumerate(population):
            try:
                # 创建临时机理文件
                apply_a_factors(reduced_mech_path, a_factors, optimizable_indices, temp_path)

                # 计算 f_tau
                f_tau = 0.0
                n_valid_idt = 0
                for key, tau_0 in ref_idt.items():
                    XH2, P, T = key[1], key[2], key[3]
                    t_end = 10.0 if T < 1200 else (1.0 if T < 1500 else 0.1)
                    tau_r = compute_idt(str(temp_path), T, P, fuel_H2_frac=XH2, t_end=t_end)
                    if np.isnan(tau_r) or np.isnan(tau_0) or tau_0 <= 0:
                        continue
                    f_tau += SIGMA_TAU * abs((tau_r - tau_0) / tau_0)
                    n_valid_idt += 1
                if n_valid_idt > 0:
                    f_tau /= n_valid_idt

                # 计算 f_LFS (与实验数据对比)
                f_lfs = 0.0
                n_valid_lfs = 0
                for key, ul_0 in ref_lfs.items():
                    XH2, P, Tu, phi = key[1], key[2], key[3], key[4]
                    ul_r = compute_lfs(str(temp_path), Tu, P, fuel_H2_frac=XH2, phi=phi)
                    if np.isnan(ul_r) or np.isnan(ul_0) or ul_0 <= 0:
                        continue
                    # 使用简化机理的 LFS 作为参考 (论文中是与实验对比)
                    # 这里用简化机理的值近似
                    f_lfs += SIGMA_LFS * abs((ul_0 - ul_r) / ul_0)
                    n_valid_lfs += 1
                if n_valid_lfs > 0:
                    f_lfs /= n_valid_lfs

                # 计算 f_A (A 因子变化幅度约束)
                f_a = 0.0
                for af in a_factors:
                    f_a += SIGMA_A * abs(np.log(af))
                f_a /= n_optimizable

                # 计算 f_CPU
                t_start = time.time()
                compute_idt(str(temp_path), 1500, 1.0, fuel_H2_frac=0.0)
                cpu_time = time.time() - t_start
                f_cpu = 1.0 / (1.0 + np.exp(SIGMA_CPU * (1 - cpu_time / (LAMBDA_CPU * ref_cpu_time))))

                f_objective = f_tau + f_lfs + f_a + f_cpu
                fitness.append(f_objective)

            except Exception as e:
                fitness.append(float('inf'))

        fitness = np.array(fitness)

        # 记录统计
        valid_fitness = fitness[fitness < float('inf')]
        if len(valid_fitness) > 0:
            best_f = np.min(valid_fitness)
            worst_f = np.max(valid_fitness)
            avg_f = np.mean(valid_fitness)
        else:
            best_f = worst_f = avg_f = float('inf')

        history["generation"].append(gen)
        history["best_fitness"].append(best_f)
        history["worst_fitness"].append(worst_f)
        history["avg_fitness"].append(avg_f)

        if gen % 50 == 0 or gen == N_GENERATIONS - 1:
            best_idx = np.argmin(fitness)
            print(f"  Gen {gen:4d}: best={best_f:.5f}, avg={avg_f:.5f}, worst={worst_f:.5f}")

        # 选择 + 交叉 + 变异
        new_population = []

        # 精英保留
        if ELITISM:
            best_idx = np.argmin(fitness)
            new_population.append(population[best_idx].copy())

        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)
            child1, child2 = single_point_crossover_fp(parent1, parent2)
            child1 = bidirectional_mutation_fp(child1)
            child2 = bidirectional_mutation_fp(child2)
            # 确保在范围内
            child1 = np.clip(child1, A_FACTOR_MIN, A_FACTOR_MAX)
            child2 = np.clip(child2, A_FACTOR_MIN, A_FACTOR_MAX)
            new_population.append(child1)
            if len(new_population) < POP_SIZE:
                new_population.append(child2)

        population = new_population[:POP_SIZE]

    # Step 5: 输出最优个体
    best_idx = np.argmin(fitness)
    best_a_factors = population[best_idx]

    print(f"\n=== GA Optimization Complete ===")
    print(f"Best fitness: {fitness[best_idx]:.5f}")

    # 保存优化机理
    output_path = MECH_DIR / "GA_Optimized_mechanism.yaml"
    apply_a_factors(reduced_mech_path, best_a_factors, optimizable_indices, output_path)
    print(f"Optimized mechanism saved to: {output_path}")

    # 验证
    gas_opt = ct.Solution(str(output_path))
    print(f"Optimized mechanism: {gas_opt.n_species} species, {gas_opt.n_reactions} reactions")

    # 保存 A 因子变化
    gas_reduced = ct.Solution(str(reduced_mech_path))
    a_factor_data = []
    for idx, af in zip(optimizable_indices, best_a_factors):
        rxn = gas_reduced.reaction(idx)
        a_factor_data.append({
            "reaction_index": idx,
            "equation": rxn.equation,
            "a_factor": af,
            "original_A": gas_reduced.reaction(idx).rate.pre_exponential_factor if hasattr(gas_reduced.reaction(idx).rate, 'pre_exponential_factor') else "N/A",
        })
    a_factor_df = pd.DataFrame(a_factor_data)
    a_factor_df.to_csv(RESULTS_DIR / "ga_optimization_a_factors.csv", index=False)

    # 保存进化历史
    history_df = pd.DataFrame(history)
    history_df.to_csv(RESULTS_DIR / "ga_optimization_history.csv", index=False)
    print(f"Evolution history saved to: {RESULTS_DIR / 'ga_optimization_history.csv'}")

    # 清理临时文件
    if temp_path.exists():
        temp_path.unlink()

    return best_a_factors, fitness[best_idx]


if __name__ == "__main__":
    run_ga_optimization()
