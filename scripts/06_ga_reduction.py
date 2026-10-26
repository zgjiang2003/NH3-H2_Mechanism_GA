"""
GA 机理简化 - 复现论文 Section 2.2 (二进制编码遗传算法)

论文参数 (Table 2):
  种群 60, 代数 1000, 单点交叉率 0.2, 双向变异率 0.03,
  锦标赛选择 + 精英保留; 初始种群全部基因 = 1 (完整机理)

目标函数 (Eq.1-5, Table 1):
  f = f_tau + f_Ts + f_N + f_CPU
  f_tau = ln[1 + 1*|tau_r - tau_0| / tau_0]                (对全机理, 按点平均)
  f_Ts  = ln[1 + 1*|Ts_r - Ts_0| / (Ts_0 - Tin)]           (对全机理, 按点平均)
  f_N   = 1 / (1 + exp[6*(1 - Nr/(0.8*NR))])
  f_CPU = 1 / (1 + exp[6*(1 - t_r/(4*t_0))])

目标工况: Table 3 Case 1-8, 每工况取 3 个温度点 (共 24 点)。

用法:
  python 06_ga_reduction.py [--gens 1000] [--pop 60] [--workers 16] [--resume]
"""
import argparse
import pickle
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_utils import (MECH_DIR, RESULTS_DIR, FULL_MECH, REDUCED_MECH,
                        MechBuilder, chen_composition, ignition_delay,
                        steady_state_temperature)

# GA 参数 (论文 Table 2)
POP_SIZE = 60
N_GENERATIONS = 1000
CROSSOVER_RATE = 0.2
MUTATION_RATE = 0.03
TOURNAMENT_SIZE = 3

# 目标函数系数 (论文 Table 1)
SIGMA_TAU = 1.0
SIGMA_TS = 1.0
SIGMA_N, LAMBDA_N = 6.0, 0.8
SIGMA_CPU, LAMBDA_CPU = 6.0, 4.0

MISSING_PENALTY = np.log(1.0 + 50.0)   # IDT 未点火惩罚 (相当于 t_end=50*tau_ref)
BROKEN_FITNESS = 1.0e3
FAIL_FRACTION_LIMIT = 0.2              # 超过 20% 工况失效 => 个体失效

# Table 3 Case 1-8 温度范围及评估点 (每工况 3 点)
REDUCTION_CONDITIONS = [
    {"case": 1, "XH2": 0.0,  "P_atm": 1.0,  "T": [1571, 1695, 1819]},
    {"case": 2, "XH2": 0.0,  "P_atm": 10.0, "T": [1469, 1625, 1781]},
    {"case": 3, "XH2": 0.05, "P_atm": 1.0,  "T": [1542, 1710, 1878]},
    {"case": 4, "XH2": 0.05, "P_atm": 10.0, "T": [1497, 1685, 1873]},
    {"case": 5, "XH2": 0.3,  "P_atm": 1.0,  "T": [1244, 1500, 1756]},
    {"case": 6, "XH2": 0.3,  "P_atm": 10.0, "T": [1244, 1550, 1856]},
    {"case": 7, "XH2": 0.7,  "P_atm": 1.0,  "T": [1054, 1270, 1486]},
    {"case": 8, "XH2": 0.7,  "P_atm": 10.0, "T": [1057, 1165, 1273]},
]


def flatten_conditions():
    """展开为 [(XH2, P, T), ...] 与工况标签"""
    conds, labels = [], []
    for c in REDUCTION_CONDITIONS:
        for T in c["T"]:
            conds.append((c["XH2"], c["P_atm"], T))
            labels.append((c["case"], c["XH2"], c["P_atm"], T))
    return conds, labels


# ---------------- 工作进程全局 ----------------
_W = {}


def _init_worker(conds):
    _W["builder"] = MechBuilder(FULL_MECH)
    _W["conds"] = conds


def _evaluate_gene(gene, ref, with_breakdown=False):
    """
    评估单个基因: 返回 (fitness, n_kept, cpu_time[, breakdown])
    ref: {"tau0": arr, "ts0": arr, "t0_cpu": float, "NR": int}
    """
    import cantera as ct
    builder = _W["builder"]
    conds = _W["conds"]
    n_kept = int(np.sum(gene))

    t_start = time.perf_counter()
    try:
        yaml_text = builder.build(gene)
        gas = ct.Solution(yaml=yaml_text)
    except Exception:
        return (BROKEN_FITNESS, n_kept, 0.0)

    tau_r = np.full(len(conds), np.nan)
    ts_r = np.full(len(conds), np.nan)
    n_fail = 0
    for i, (XH2, P, T) in enumerate(conds):
        comp = chen_composition(XH2, 1.0)   # Chen 2021 Ar 稀释混合气
        tau_ref = ref["tau0"][i]
        t_end = min(max(50.0 * tau_ref, 1e-5), 20.0)
        tau_r[i] = ignition_delay(gas, T, P, comp, t_end)
        if not np.isfinite(tau_r[i]):
            n_fail += 1
            if n_fail > FAIL_FRACTION_LIMIT * len(conds):
                return (BROKEN_FITNESS, n_kept, time.perf_counter() - t_start,
                        (np.nan,) * 4)
    for i, (XH2, P, T) in enumerate(conds):
        comp = chen_composition(XH2, 1.0)
        t_run = min(max(100.0 * ref["tau0"][i], 5e-3), 50.0)
        try:
            ts_r[i] = steady_state_temperature(gas, T, P, comp, t_run,
                                               n_extend=2)
        except Exception:
            ts_r[i] = np.nan
    cpu_time = time.perf_counter() - t_start

    # f_tau / f_Ts
    valid_tau = np.isfinite(ref["tau0"]) & (ref["tau0"] > 0)

    f_tau_terms = np.where(
        np.isfinite(tau_r) & valid_tau,
        np.log(1.0 + SIGMA_TAU * np.abs(tau_r - ref["tau0"]) / ref["tau0"]),
        MISSING_PENALTY)
    f_tau = float(np.mean(f_tau_terms[valid_tau]))

    dT0 = ref["ts0"] - np.array([c[2] for c in conds], dtype=float)
    ok_ts = valid_tau & (dT0 > 100.0) & np.isfinite(ref["ts0"])
    if np.any(np.isfinite(ts_r[ok_ts])):
        ts_terms = np.where(
            np.isfinite(ts_r) & ok_ts,
            np.log(1.0 + SIGMA_TS * np.abs(ts_r - ref["ts0"]) / dT0),
            MISSING_PENALTY)
        f_ts = float(np.mean(ts_terms[ok_ts]))
    else:
        f_ts = MISSING_PENALTY

    # f_N / f_CPU
    NR = ref["NR"]
    f_N = 1.0 / (1.0 + np.exp(SIGMA_N * (1.0 - n_kept / (LAMBDA_N * NR))))
    t0 = max(ref["t0_cpu"], 1e-6)
    f_CPU = 1.0 / (1.0 + np.exp(SIGMA_CPU * (1.0 - cpu_time / (LAMBDA_CPU * t0))))

    fitness = f_tau + f_ts + f_N + f_CPU
    if not np.isfinite(fitness):
        fitness = BROKEN_FITNESS
    return fitness, n_kept, cpu_time, (f_tau, f_ts, f_N, f_CPU)


def _worker_eval(args):
    gene, ref = args
    return _evaluate_gene(gene, ref)

def eval_batch_resilient(ex, jobs, keys, cache, conds, args):
    """批量评估; 进程池崩溃时逐个定位段错误基因并拉黑, 重建池"""
    from concurrent.futures.process import BrokenProcessPool
    try:
        results = list(ex.map(_worker_eval, jobs))
        for k, r in zip(keys, results):
            cache[k] = r
        return ex
    except (BrokenProcessPool, OSError, SystemError) as e:
        print(f"  [池崩溃: {type(e).__name__}] 逐个重评估定位崩溃基因...",
              flush=True)
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        ex1 = ProcessPoolExecutor(max_workers=1, initializer=_init_worker,
                                  initargs=(conds,))
        bad = 0
        for job, k in zip(jobs, keys):
            if k in cache:
                continue
            try:
                r = ex1.submit(_worker_eval, job).result(timeout=900)
                cache[k] = r
            except Exception:
                n = int(np.frombuffer(k, dtype=np.int8).sum())
                cache[k] = (BROKEN_FITNESS, n, 0.0,
                            (np.nan, np.nan, np.nan, np.nan))
                bad += 1
        ex1.shutdown(wait=False)
        print(f"  [定位完成] 崩溃基因 {bad} 个已拉黑, 重建进程池", flush=True)
        return ProcessPoolExecutor(max_workers=args.workers,
                                   initializer=_init_worker,
                                   initargs=(conds,))




# ---------------- 主进程 GA ----------------
def tournament(population, fitness, rng, size=TOURNAMENT_SIZE):
    idx = rng.integers(0, len(population), size=size)
    best = idx[np.argmin(fitness[idx])]
    return population[best]


def crossover(p1, p2, rng):
    if rng.random() < CROSSOVER_RATE:
        pt = rng.integers(1, len(p1) - 1)
        c1 = np.concatenate([p1[:pt], p2[pt:]])
        c2 = np.concatenate([p2[:pt], p1[pt:]])
        return c1, c2
    return p1.copy(), p2.copy()


def mutate(gene, rng):
    mask = rng.random(len(gene)) < MUTATION_RATE
    gene[mask] = 1 - gene[mask]
    return gene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=N_GENERATIONS)
    ap.add_argument("--pop", type=int, default=POP_SIZE)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    state_path = RESULTS_DIR / "ga_reduction_state.pkl"
    builder = MechBuilder(FULL_MECH)
    n_rxn = builder.n_reactions
    conds, labels = flatten_conditions()

    rng = np.random.default_rng(args.seed)
    cache = {}

    if args.resume and state_path.exists():
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        population = state["population"]
        history = state["history"]
        start_gen = state["generation"] + 1
        cache = state.get("cache", {})
        rng = state.get("rng", rng)
        _gh = RESULTS_DIR / "ga_reduction_genes.npy"
        genes_history = (list(np.load(_gh)) if _gh.exists() else [])
        print(f"恢复 GA 状态: 从第 {start_gen} 代继续, 缓存 {len(cache)} 条")
    else:
        # 初始种群: 全部基因 = 1 (论文)
        population = [np.ones(n_rxn, dtype=np.int8) for _ in range(args.pop)]
        history = {"generation": [], "best": [], "worst": [], "avg": [],
                   "n_reactions": [], "f_tau": [], "f_ts": [], "f_N": [],
                   "f_CPU": []}
        start_gen = 0

    # ---- 参考值: 完整机理 IDT + Ts (主进程计算一次) ----
    import cantera as ct
    print("计算完整机理参考值 (IDT + 稳态温度)...")
    t0 = time.perf_counter()
    gas0 = ct.Solution(str(FULL_MECH))
    tau0 = np.zeros(len(conds))
    ts0 = np.zeros(len(conds))
    for i, (XH2, P, T) in enumerate(conds):
        comp = chen_composition(XH2, 1.0)
        tau = ignition_delay(gas0, T, P, comp,
                             t_end=10.0 if T < 1300 else 1.0)
        tau = ignition_delay(gas0, T, P, comp, min(max(50 * tau, 1e-5), 20))
        tau0[i] = tau
        ts0[i] = steady_state_temperature(gas0, T, P, comp,
                                          min(max(100 * tau, 5e-3), 50),
                                          n_extend=2)
        print(f"  Case{labels[i][0]} XH2={XH2} P={P}atm T={T}K: "
              f"tau={tau*1000:.4f} ms, Ts={ts0[i]:.1f} K")
    ref_cpu = time.perf_counter() - t0
    ref = {"tau0": tau0, "ts0": ts0, "t0_cpu": ref_cpu, "NR": n_rxn}
    print(f"参考评估耗时 {ref_cpu:.2f} s (作为 t_CPU 基准)")

    best_gene = None
    best_fitness = np.inf
    if 'genes_history' not in locals():   # resume 分支已加载
        genes_history = []      # 每代最优个体基因 (Fig. 5)
    t_ga = time.perf_counter()

    ex = ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(conds,))
    try:
        for gen in range(start_gen, args.gens):
            # 评估 (带缓存)
            jobs, keys = [], []
            for g in population:
                k = g.tobytes()
                if k not in cache:
                    jobs.append((g, ref))
                    keys.append(k)
            if jobs:
                ex = eval_batch_resilient(ex, jobs, keys, cache,
                                         conds, args)
            fitness = np.array([cache[g.tobytes()][0] for g in population])

            valid = fitness[fitness < BROKEN_FITNESS]
            best_idx = int(np.argmin(fitness))
            best_f = float(fitness[best_idx])
            best_gene = population[best_idx].copy()
            best_fitness = best_f
            f_tau, f_ts, f_N, f_CPU = cache[best_gene.tobytes()][3]
            hist = {
                "generation": gen,
                "best": best_f,
                "worst": float(np.max(valid)) if len(valid) else np.nan,
                "avg": float(np.mean(valid)) if len(valid) else np.nan,
                "n_reactions": int(np.sum(best_gene)),
                "f_tau": f_tau, "f_ts": f_ts, "f_N": f_N, "f_CPU": f_CPU,
            }
            for kk, vv in hist.items():
                history[kk].append(vv)
            genes_history.append(best_gene.astype(np.int8))

            if gen % 10 == 0 or gen == args.gens - 1:
                el = time.perf_counter() - t_ga
                print(f"  Gen {gen:4d}: best={best_f:.5f} "
                      f"avg={hist['avg']:.5f} Nr={hist['n_reactions']} "
                      f"fN={f_N:.4f} fCPU={f_CPU:.4f} "
                      f"[{el:.0f}s, cache={len(cache)}]", flush=True)

            # 进化操作: 精英 + 锦标赛 + 交叉 + 变异
            new_pop = [best_gene.copy()]
            while len(new_pop) < args.pop:
                p1 = tournament(population, fitness, rng)
                p2 = tournament(population, fitness, rng)
                c1, c2 = crossover(p1, p2, rng)
                new_pop.append(mutate(c1, rng))
                if len(new_pop) < args.pop:
                    new_pop.append(mutate(c2, rng))
            population = new_pop

            # 检查点
            if (gen + 1) % 10 == 0 or gen == args.gens - 1:
                with open(state_path, "wb") as f:
                    pickle.dump({"population": population, "history": history,
                                 "generation": gen, "cache": cache,
                                 "rng": rng}, f)
                pd.DataFrame(history).to_csv(
                    RESULTS_DIR / "ga_reduction_history.csv", index=False)
                np.save(RESULTS_DIR / "ga_reduction_genes.npy",
                        np.array(genes_history))

    finally:
        ex.shutdown(wait=False)

    # ---- 输出最优个体 (取进化过程中记录的精英) ----
    best_gene = best_gene if best_gene is not None else population[0]
    best_f = best_fitness
    nr = int(np.sum(best_gene))
    print(f"\n=== GA 简化完成: 最优适应度 {best_f:.5f}, "
          f"保留 {nr}/{n_rxn} 反应 ===")

    # 保存项目 GA 结果；论文官方 Mechanism 1 保存在 Reduced mechanism.yaml，勿覆盖。
    pruned_yaml = builder.build_pruned(best_gene)
    out = MECH_DIR / "GA_Reduced_mechanism.yaml"
    out.write_text(pruned_yaml)
    g = ct.Solution(str(out))
    print(f"项目 GA 简化机理: {g.n_species} 组分, {g.n_reactions} 反应 -> {out}")

    # 保留/删除反应清单
    rows = []
    gas_full = ct.Solution(str(FULL_MECH))
    for i in range(n_rxn):
        rows.append({"index": i,
                     "equation": gas_full.reaction(i).equation,
                     "kept": bool(best_gene[i])})
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "ga_reduction_reactions.csv", index=False)

    with open(state_path, "wb") as f:
        pickle.dump({"population": population, "history": history,
                     "generation": args.gens - 1, "cache": cache,
                     "rng": rng}, f)
    pd.DataFrame(history).to_csv(
        RESULTS_DIR / "ga_reduction_history.csv", index=False)
    np.save(RESULTS_DIR / "ga_reduction_genes.npy",
            np.array(genes_history))
    print("进化历史 -> results/ga_reduction_history.csv")


if __name__ == "__main__":
    main()
