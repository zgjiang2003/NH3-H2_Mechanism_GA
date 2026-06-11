"""
GA 机理简化算法
复现论文 Section 2.2: GA-based mechanism reduction

算法流程:
1. 二进制编码: 每个基因对应一个反应 (1=保留, 0=删除)
2. 初始种群: 全部基因设为1 (完整机理)
3. 适应度函数: f_objective = f_tau + f_Ts + f_N + f_CPU
4. 锦标赛选择 + 精英保留
5. 单点交叉 (交叉率 0.2)
6. 双向变异 (变异率 0.03)
7. 运行 1000 代

目标工况 (Table 3 Case 1-8):
- XH2=0, P=1/10 atm, T=1540-1850/1430-1820 K
- XH2=0.05, P=1/10 atm, T=1500-1920/1450-1920 K
- XH2=0.3, P=1/10 atm, T=1180-1820/1180-1920 K
- XH2=0.7, P=1/10 atm, T=1000-1540/1030-1300 K
"""
import cantera as ct
import numpy as np
import pandas as pd
import yaml
import copy
import time
from pathlib import Path
from multiprocessing import Pool

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MECH_DIR = PROJECT_ROOT / "mechanisms"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# GA 参数 (论文 Table 2)
# ============================================================
POP_SIZE = 60
N_GENERATIONS = 1000
CROSSOVER_RATE = 0.2
MUTATION_RATE = 0.03
TOURNAMENT_SIZE = 3
ELITISM = True

# 目标函数系数 (论文 Section 2.2)
SIGMA_TAU = 1.0
SIGMA_TS = 1.0
SIGMA_N = 6.0
LAMBDA_N = 0.8
SIGMA_CPU = 6.0
LAMBDA_CPU = 4.0

# 简化目标工况 (Table 3 Case 1-8)
# 每个工况取温度范围中点附近 3 个温度点用于评估
REDUCTION_CONDITIONS = [
    {"XH2": 0.0, "P_atm": 1.0, "T_range": [1600, 1700, 1800]},
    {"XH2": 0.0, "P_atm": 10.0, "T_range": [1500, 1600, 1700]},
    {"XH2": 0.05, "P_atm": 1.0, "T_range": [1600, 1700, 1800]},
    {"XH2": 0.05, "P_atm": 10.0, "T_range": [1600, 1700, 1800]},
    {"XH2": 0.3, "P_atm": 1.0, "T_range": [1300, 1500, 1700]},
    {"XH2": 0.3, "P_atm": 10.0, "T_range": [1300, 1500, 1700]},
    {"XH2": 0.7, "P_atm": 1.0, "T_range": [1100, 1270, 1400]},
    {"XH2": 0.7, "P_atm": 10.0, "T_range": [1050, 1165, 1250]},
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
    """计算点火延迟时间 (OH峰值准则)"""
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
        tau = times[np.argmax(oh_vals)]
        return tau
    except Exception:
        return np.nan


def compute_steady_state_temp(mech_path, T, P_atm, fuel_H2_frac=0.0, phi=1.0, t_end=1.0):
    """计算恒压反应器稳态温度"""
    try:
        gas = ct.Solution(str(mech_path))
        fuel = {"NH3": 1.0 - fuel_H2_frac}
        if fuel_H2_frac > 0:
            fuel["H2"] = fuel_H2_frac
        X = oxidizer_for_phi(fuel, phi)
        gas.TPX = T, P_atm * ct.one_atm, X

        reactor = ct.IdealGasConstPressureReactor(gas)
        net = ct.ReactorNet([reactor])

        t = 0.0
        T_steady = T
        while t < t_end:
            t = net.step()
            T_steady = reactor.T
            if abs(reactor.T - T_steady) < 0.1 and t > 0.01:
                break

        return T_steady
    except Exception:
        return np.nan


def create_sub_mechanism_yaml(full_mech_path, gene, output_path):
    """根据基因编码创建子机理 YAML 文件"""
    with open(full_mech_path, 'r') as f:
        mech_data = yaml.safe_load(f)

    # 保留基因值为1的反应
    reactions = mech_data.get('reactions', [])
    kept_reactions = [r for i, r in enumerate(reactions) if i < len(gene) and gene[i] == 1]

    mech_data['reactions'] = kept_reactions

    with open(output_path, 'w') as f:
        yaml.dump(mech_data, f, default_flow_style=False, allow_unicode=True)

    return len(kept_reactions)


def evaluate_individual(args):
    """评估单个个体的适应度"""
    gene, full_mech_path, ref_idt, ref_ts, ref_cpu_time, n_full_reactions = args

    # 检查至少保留一定数量的反应
    n_kept = np.sum(gene)
    if n_kept < 10:
        return float('inf')

    # 创建临时子机理文件
    temp_path = RESULTS_DIR / "_temp_sub_mech.yaml"
    try:
        n_reactions = create_sub_mechanism_yaml(full_mech_path, gene, temp_path)

        # 计算子机理的 IDT 和稳态温度
        t_start = time.time()
        sub_idt = {}
        sub_ts = {}
        for cond in REDUCTION_CONDITIONS:
            XH2 = cond["XH2"]
            P = cond["P_atm"]
            for T in cond["T_range"]:
                key = (XH2, P, T)
                t_end = 10.0 if T < 1200 else (1.0 if T < 1500 else 0.1)
                sub_idt[key] = compute_idt(temp_path, T, P, fuel_H2_frac=XH2, t_end=t_end)
                sub_ts[key] = compute_steady_state_temp(temp_path, T, P, fuel_H2_frac=XH2, t_end=t_end * 10)
        cpu_time = time.time() - t_start

        # 计算目标函数各项
        f_tau = 0.0
        f_ts = 0.0
        n_valid = 0
        for key in ref_idt:
            if key not in sub_idt:
                continue
            tau_r = sub_idt[key]
            tau_0 = ref_idt[key]
            ts_r = sub_ts.get(key, np.nan)
            ts_0 = ref_ts.get(key, np.nan)

            if np.isnan(tau_r) or np.isnan(tau_0) or tau_0 <= 0:
                continue

            # f_tau (公式1)
            f_tau += np.log(1 + SIGMA_TAU * abs((tau_r - tau_0) / tau_0))

            # f_Ts (公式2)
            if not np.isnan(ts_r) and not np.isnan(ts_0):
                T_in = key[2]  # 初始温度
                if ts_0 - T_in > 0:
                    f_ts += np.log(1 + SIGMA_TS * abs((ts_r - ts_0) / (ts_0 - T_in)))

            n_valid += 1

        if n_valid > 0:
            f_tau /= n_valid
            f_ts /= n_valid

        # f_N (公式3) - 机理规模
        f_N = 1.0 / (1.0 + np.exp(SIGMA_N * (1 - n_reactions / (LAMBDA_N * n_full_reactions))))

        # f_CPU (公式4) - 计算代价
        if ref_cpu_time > 0:
            f_CPU = 1.0 / (1.0 + np.exp(SIGMA_CPU * (1 - cpu_time / (LAMBDA_CPU * ref_cpu_time))))
        else:
            f_CPU = 0.0

        f_objective = f_tau + f_ts + f_N + f_CPU
        return f_objective

    except Exception as e:
        return float('inf')
    finally:
        if temp_path.exists():
            temp_path.unlink()


def tournament_selection(population, fitness, tournament_size=TOURNAMENT_SIZE):
    """锦标赛选择"""
    indices = np.random.choice(len(population), tournament_size, replace=False)
    best_idx = indices[np.argmin([fitness[i] for i in indices])]
    return population[best_idx].copy()


def single_point_crossover(parent1, parent2):
    """单点交叉"""
    if np.random.random() < CROSSOVER_RATE:
        point = np.random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()


def bidirectional_mutation(gene):
    """双向变异: 基因可以从0变1也可以从1变0"""
    mutated = gene.copy()
    for i in range(len(mutated)):
        if np.random.random() < MUTATION_RATE:
            mutated[i] = 1 - mutated[i]  # 翻转
    return mutated


def run_ga_reduction():
    """运行 GA 机理简化"""
    full_mech_path = MECH_DIR / "Full mechanism.yaml"
    gas = ct.Solution(str(full_mech_path))
    n_reactions = gas.n_reactions
    print(f"Full mechanism: {gas.n_species} species, {n_reactions} reactions")

    # Step 1: 计算完整机理的参考值
    print("Computing reference values from full mechanism...")
    ref_idt = {}
    ref_ts = {}
    for cond in REDUCTION_CONDITIONS:
        XH2 = cond["XH2"]
        P = cond["P_atm"]
        for T in cond["T_range"]:
            key = (XH2, P, T)
            t_end = 10.0 if T < 1200 else (1.0 if T < 1500 else 0.1)
            ref_idt[key] = compute_idt(str(full_mech_path), T, P, fuel_H2_frac=XH2, t_end=t_end)
            ref_ts[key] = compute_steady_state_temp(str(full_mech_path), T, P, fuel_H2_frac=XH2, t_end=t_end * 10)
            print(f"  XH2={XH2}, P={P} atm, T={T} K: IDT={ref_idt[key]*1000:.3f} ms, Ts={ref_ts[key]:.1f} K")

    # 计算参考 CPU 时间
    t_start = time.time()
    for cond in REDUCTION_CONDITIONS[:2]:  # 只用部分条件估计
        XH2 = cond["XH2"]
        P = cond["P_atm"]
        for T in cond["T_range"]:
            compute_idt(str(full_mech_path), T, P, fuel_H2_frac=XH2)
    ref_cpu_time = time.time() - t_start
    print(f"Reference CPU time: {ref_cpu_time:.3f} s")

    # Step 2: 初始化种群 (全部基因设为1)
    print(f"\nInitializing population: {POP_SIZE} individuals, {n_reactions} genes each")
    population = [np.ones(n_reactions, dtype=int) for _ in range(POP_SIZE)]

    # 对部分个体添加随机扰动以增加多样性
    for i in range(1, POP_SIZE):
        # 随机删除 10-30% 的反应
        n_remove = np.random.randint(int(0.1 * n_reactions), int(0.3 * n_reactions))
        remove_idx = np.random.choice(n_reactions, n_remove, replace=False)
        population[i][remove_idx] = 0

    # 记录进化过程
    history = {
        "generation": [],
        "best_fitness": [],
        "worst_fitness": [],
        "avg_fitness": [],
        "best_n_reactions": [],
    }

    # Step 3: GA 主循环
    print(f"\nStarting GA reduction: {N_GENERATIONS} generations")
    for gen in range(N_GENERATIONS):
        # 评估适应度
        fitness = []
        for i, gene in enumerate(population):
            f = evaluate_individual((
                gene, str(full_mech_path), ref_idt, ref_ts, ref_cpu_time, n_reactions
            ))
            fitness.append(f)

        fitness = np.array(fitness)

        # 记录统计
        valid_fitness = fitness[fitness < float('inf')]
        if len(valid_fitness) > 0:
            best_idx = np.argmin(fitness)
            best_gene = population[best_idx]
            best_f = fitness[best_idx]
            worst_f = np.max(valid_fitness)
            avg_f = np.mean(valid_fitness)
            best_n = np.sum(best_gene)
        else:
            best_f = worst_f = avg_f = float('inf')
            best_n = n_reactions

        history["generation"].append(gen)
        history["best_fitness"].append(best_f)
        history["worst_fitness"].append(worst_f)
        history["avg_fitness"].append(avg_f)
        history["best_n_reactions"].append(best_n)

        if gen % 50 == 0 or gen == N_GENERATIONS - 1:
            print(f"  Gen {gen:4d}: best={best_f:.5f}, avg={avg_f:.5f}, "
                  f"worst={worst_f:.5f}, n_reactions={best_n}")

        # 选择 + 交叉 + 变异
        new_population = []

        # 精英保留
        if ELITISM:
            best_idx = np.argmin(fitness)
            new_population.append(population[best_idx].copy())

        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)
            child1, child2 = single_point_crossover(parent1, parent2)
            child1 = bidirectional_mutation(child1)
            child2 = bidirectional_mutation(child2)
            new_population.append(child1)
            if len(new_population) < POP_SIZE:
                new_population.append(child2)

        population = new_population[:POP_SIZE]

    # Step 4: 输出最优个体
    best_idx = np.argmin(fitness)
    best_gene = population[best_idx]
    best_n_reactions = np.sum(best_gene)

    print(f"\n=== GA Reduction Complete ===")
    print(f"Best fitness: {fitness[best_idx]:.5f}")
    print(f"Number of reactions: {best_n_reactions}")

    # 保存简化机理
    output_path = MECH_DIR / "GA_Reduced_mechanism.yaml"
    n_kept = create_sub_mechanism_yaml(full_mech_path, best_gene, output_path)
    print(f"Reduced mechanism saved to: {output_path}")
    print(f"Kept {n_kept} reactions out of {n_reactions}")

    # 验证简化机理
    gas_reduced = ct.Solution(str(output_path))
    print(f"Reduced mechanism: {gas_reduced.n_species} species, {gas_reduced.n_reactions} reactions")

    # 保存进化历史
    history_df = pd.DataFrame(history)
    history_df.to_csv(RESULTS_DIR / "ga_reduction_history.csv", index=False)
    print(f"Evolution history saved to: {RESULTS_DIR / 'ga_reduction_history.csv'}")

    # 保存被保留/删除的反应列表
    gas_full = ct.Solution(str(full_mech_path))
    reaction_status = []
    for i in range(n_reactions):
        reaction_status.append({
            "index": i,
            "equation": gas_full.reaction(i).equation,
            "kept": bool(best_gene[i]),
        })
    reaction_df = pd.DataFrame(reaction_status)
    reaction_df.to_csv(RESULTS_DIR / "ga_reduction_reactions.csv", index=False)

    return best_gene, fitness[best_idx]


if __name__ == "__main__":
    run_ga_reduction()
