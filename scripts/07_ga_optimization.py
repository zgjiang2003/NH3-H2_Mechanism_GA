"""
GA 机理优化 - 复现论文 Section 2.3 (实数编码遗传算法)

优化对象: Reduced mechanism (论文 Mechanism 1, 原版文件) 中除氢气子机理与
压力依赖(falloff)反应外的 ~45 个反应的 Arrhenius 指前因子 A
  A ∈ [0.1*A0, 10*A0] (不确定度因子 10)

目标函数 (Eq.6-8, Table 1, Table 3 Case 1-12):
  f = f_tau + f_LFS + f_A + f_CPU
  f_tau = mean_i ln[1 + 4*|tau_sim - tau_exp| / tau_exp]     (Chen IDT 实验)
  f_LFS = mean_i 4*|u_exp - u_pre| / u_exp                   (LFS 实验, 线性式)
  f_A   = mean_j |0.3 * ln(A_j / A0_j)|
  f_CPU = 1/(1 + exp[6*(1 - t_r/(4*t_0))])

GA 参数 (Table 2): 种群 30, 代数 1000, 单点交叉 0.3, 双向变异 0.06,
锦标赛选择 + 精英保留; 初始种群全部基因 = 1 (即 Mechanism 1)

用法: python 07_ga_optimization.py [--gens 300] [--pop 30] [--workers 16]
"""
import argparse
import pickle
import sys
import time
import numpy as np
import pandas as pd
import cantera as ct
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_utils import (MECH_DIR, RESULTS_DIR, REDUCED_MECH,
                        chen_composition, phi_composition, ignition_delay)
from experimental_data import IDT_CHEN_1_2ATM, IDT_CHEN_10ATM, interp_loglog, interp_loglin

# GA 参数 (论文 Table 2)
POP_SIZE = 30
N_GENERATIONS = 1000
CROSSOVER_RATE = 0.3
MUTATION_RATE = 0.06
TOURNAMENT_SIZE = 3
A_MIN, A_MAX = 0.1, 10.0

# 目标函数系数 (论文 Table 1)
SIGMA_TAU = 4.0
SIGMA_LFS = 4.0
SIGMA_A = 0.3
SIGMA_CPU, LAMBDA_CPU = 6.0, 4.0

# 氢气子机理组分 (+ 惰性碰撞体)
H2_SPECIES = {"H2", "H", "O2", "O", "OH", "HO2", "H2O2", "H2O",
              "N2", "AR", "HE", "M"}

# f_tau 目标点: Case 1-8, 每工况 2 个温度 (位于 Chen 实验数字化范围内)
IDT_TARGETS = []   # (XH2, P_atm, T, tau_exp_s)
for _tbl, _P in [(IDT_CHEN_1_2ATM, 1.2), (IDT_CHEN_10ATM, 10.0)]:
    for _xh2, _pts in _tbl.items():
        _xs = np.array([p[0] for p in _pts])
        for _x in np.linspace(_xs.min() + 0.15, _xs.max() - 0.15, 2):
            _tau = interp_loglog(_xs, [p[1] for p in _pts], _x)
            if np.isfinite(_tau):
                IDT_TARGETS.append((_xh2, _P, 1000.0 / _x, _tau * 1e-3))

# f_LFS 目标点: (XH2, Tu, P_atm, phi, u_exp_m_s) —— 数字化实验值
_LFS_298 = {
    (0.0, 298.0, 1.0): [(0.8, 5.6), (1.0, 7.0), (1.2, 6.5)],
    (0.4, 298.0, 1.0): [(0.8, 21.0), (1.0, 26.5), (1.2, 25.5), (1.4, 18.0)],
    (0.2, 298.0, 1.0): [(1.0, 12.6)],
    (0.6, 298.0, 1.0): [(1.0, 56.0)],
    (0.8, 298.0, 1.0): [(1.0, 122.0)],
    (0.4, 298.0, 3.0): [(1.0, 18.5)],
    (0.4, 298.0, 5.0): [(1.0, 14.5)],
}
LFS_TARGETS = [(k[0], k[1], k[2], ph, u / 100.0)
               for k, pts in _LFS_298.items() for ph, u in pts]


def identify_optimizable(gas):
    """可优化反应索引: 排除 falloff/压力依赖 与 氢气子机理反应"""
    idx = []
    for i in range(gas.n_reactions):
        r = gas.reaction(i)
        t = r.reaction_type
        if ("falloff" in t or "pressure" in t or "Chebyshev" in t
                or "chemically-activated" in t):
            continue
        sp = set(r.reactants) | set(r.products)
        if sp <= H2_SPECIES:
            continue
        idx.append(i)
    return idx


# ---------------- 工作进程 ----------------
_W = {}


def _init_worker(opt_idx, idt_targets, lfs_targets):
    gas = ct.Solution(str(REDUCED_MECH))
    _W["gas"] = gas
    _W["opt_idx"] = list(opt_idx)
    _W["idt"] = list(idt_targets)
    _W["lfs"] = list(lfs_targets)
    _W["flames"] = {}   # (cond) -> FreeFlame (热启动缓存)
    _W["t0_cpu"] = None


def _set_multipliers(gene):
    gas = _W["gas"]
    for k, i in enumerate(_W["opt_idx"]):
        gas.set_multiplier(float(gene[k]), i)


def _eval_idt(gene):
    gas = _W["gas"]
    _set_multipliers(gene)
    errs = []
    for XH2, P, T, tau_exp in _W["idt"]:
        comp = chen_composition(XH2, 1.0)   # Chen 2021 Ar 稀释混合气
        t_end = min(max(50.0 * tau_exp, 1e-5), 20.0)
        tau = ignition_delay(gas, T, P, comp, t_end)
        if not np.isfinite(tau):
            errs.append(np.log(1 + SIGMA_TAU * 50.0))
        else:
            errs.append(np.log(1 + SIGMA_TAU * abs(tau - tau_exp) / tau_exp))
    return float(np.mean(errs))


def _get_flame(XH2, Tu, P, phi):
    key = (XH2, Tu, P, phi)
    if key not in _W["flames"]:
        gas = ct.Solution(str(REDUCED_MECH))
        comp = phi_composition(XH2, phi)
        gas.TPX = Tu, P * ct.one_atm, comp
        f = ct.FreeFlame(gas, width=0.03)
        f.transport_model = "mixture-averaged"
        f.set_refine_criteria(ratio=3, slope=0.12, curve=0.24)
        _W["flames"][key] = f
    return _W["flames"][key]



def _rewarm_cache(ctx, timeout=300):
    """杀池后用基线基因重建全部工况的热启动缓存 (12 个冷解)"""
    mech_path = (Path(__file__).resolve().parent.parent / "mechanisms" /
                 "Reduced mechanism.yaml")
    ones = [1.0] * len(_W["opt_idx"])
    for XH2, Tu, P, phi, _u in _W["lfs"]:
        payload = (str(mech_path), XH2, Tu, P, phi,
                   list(_W["opt_idx"]), ones)
        try:
            _W["sub_pool"].apply_async(
                _sub_flame_solve, (payload,)).get(timeout=timeout)
        except Exception:
            try:
                _W["sub_pool"].terminate()
                _W["sub_pool"].join()
            except Exception:
                pass
            _W["sub_pool"] = ctx.Pool(1)
            return
    print("    [worker] 缓存已重暖", flush=True)


def _eval_lfs(gene):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    if "sub_pool" not in _W or _W["sub_pool"] is None:
        _W["sub_pool"] = ctx.Pool(1)
    errs = []
    n_timeout = 0
    mech_path = (Path(__file__).resolve().parent.parent / "mechanisms" /
                 "Reduced mechanism.yaml")
    for XH2, Tu, P, phi, u_exp in _W["lfs"]:
        payload = (str(mech_path), XH2, Tu, P, phi,
                   list(_W["opt_idx"]),
                   np.asarray(gene, dtype=float).tolist())
        try:
            u = _W["sub_pool"].apply_async(
                _sub_flame_solve, (payload,)).get(timeout=60)
        except Exception:
            # 超时/崩溃: 硬杀子进程并重建 (热缓存丢失, 可接受)
            try:
                _W["sub_pool"].terminate()
                _W["sub_pool"].join()
            except Exception:
                pass
            _W["sub_pool"] = ctx.Pool(1)
            _rewarm_cache(ctx)
            u = float("nan")
            n_timeout += 1
        if not np.isfinite(u):
            errs.append(SIGMA_LFS * 1.0)
        else:
            errs.append(SIGMA_LFS * abs(u_exp - u) / u_exp)
        if n_timeout > 4:
            # 病态基因: 剩余工况直接按失效计, 避免整代被拖垮
            errs.extend([SIGMA_LFS * 1.0] *
                        (len(_W["lfs"]) - len(errs)))
            print(f"    [worker] gene 中止: {n_timeout} 次火焰超时",
                  flush=True)
            break
    return float(np.mean(errs))



def eval_batch_resilient(ex, jobs, keys, cache, opt_idx, idt_targets,
                         lfs_targets, args):
    """批量评估; 池崩溃时逐个定位段错误基因并拉黑, 重建池"""
    from concurrent.futures.process import BrokenProcessPool
    try:
        results = list(ex.map(_evaluate, jobs))
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
                                  initargs=(opt_idx, idt_targets, lfs_targets))
        bad = 0
        for job, k in zip(jobs, keys):
            if k in cache:
                continue
            try:
                r = ex1.submit(_evaluate, job).result(timeout=1800)
                cache[k] = r
            except Exception:
                cache[k] = (1e3, np.nan, np.nan, np.nan, np.nan, 0.0)
                bad += 1
        ex1.shutdown(wait=False)
        print(f"  [定位完成] 崩溃基因 {bad} 个已拉黑, 重建进程池", flush=True)
        return ProcessPoolExecutor(max_workers=args.workers,
                                   initializer=_init_worker,
                                   initargs=(opt_idx, idt_targets,
                                             lfs_targets))




# ---------- 子进程火焰求解 (可硬杀, 带热启动缓存) ----------
_SUB_CACHE = {}


def _sub_flame_solve(payload):
    """在子进程中求解指定工况的火焰 (gene 已含乘数); 缓存火焰对象"""
    mech_path, XH2, Tu, P, phi, opt_idx, gene = payload
    key = (XH2, Tu, P, phi)
    try:
        import cantera as ct
        if key not in _SUB_CACHE:
            gas = ct.Solution(str(mech_path))
            comp = phi_composition(XH2, phi)
            gas.TPX = Tu, P * ct.one_atm, comp
            f = ct.FreeFlame(gas, width=0.03)
            f.transport_model = "mixture-averaged"
            f.set_refine_criteria(ratio=3, slope=0.12, curve=0.24)
            f.solve(loglevel=0, auto=True)          # 冷启动一次
            _SUB_CACHE[key] = f
        f = _SUB_CACHE[key]
        # 数值稳定性: 火焰求解用夹紧乘数 (真实基因值仍进入 f_A 惩罚)
        gene_c = [min(max(v, 0.25), 4.0) for v in gene]
        for k, i in enumerate(opt_idx):
            f.gas.set_multiplier(gene_c[k], i)
        try:
            f.solve(loglevel=0, auto=False, refine_grid=False)
            u = f.velocity[0]
            if not np.isfinite(u) or u <= 0.01:
                raise RuntimeError("no flame")
            return float(u)
        except Exception:
            f.solve(loglevel=0, auto=True)
            u = f.velocity[0]
            return float(u) if (np.isfinite(u) and u > 0.01) else float("nan")
    except Exception:
        return float("nan")


def _evaluate(args):
    """评估个体: 返回 (fitness, f_tau, f_lfs, f_a, f_cpu, cpu_time)"""
    gene, = args if isinstance(args, tuple) else (args,)
    t_start = time.perf_counter()
    f_tau = _eval_idt(gene)
    f_lfs = _eval_lfs(gene)
    f_a = float(np.mean(np.abs(SIGMA_A * np.log(np.asarray(gene, dtype=float)))))
    cpu_time = time.perf_counter() - t_start
    t0 = _W.get("t0_cpu") or cpu_time
    f_cpu = 1.0 / (1.0 + np.exp(SIGMA_CPU * (1.0 - cpu_time / (LAMBDA_CPU * t0))))
    return float(f_tau + f_lfs + f_a + f_cpu), f_tau, f_lfs, f_a, f_cpu, cpu_time


def _evaluate_baseline():
    gene = np.ones(len(_W["opt_idx"]))
    return _evaluate(gene)


# ---------------- GA ----------------
def tournament(pop, fit, rng, size=TOURNAMENT_SIZE):
    idx = rng.integers(0, len(pop), size=size)
    return pop[idx[np.argmin(fit[idx])]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=N_GENERATIONS)
    ap.add_argument("--pop", type=int, default=POP_SIZE)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--serial", action="store_true",
                    help="单进程运行 (调试/稳健模式)")
    args = ap.parse_args()

    gas = ct.Solution(str(REDUCED_MECH))
    opt_idx = identify_optimizable(gas)
    n_opt = len(opt_idx)
    excluded = [i for i in range(gas.n_reactions) if i not in opt_idx]
    print(f"Mechanism 1: {gas.n_species} sp, {gas.n_reactions} rxns; "
          f"可优化反应 {n_opt} 个 (排除 {len(excluded)} 个: falloff/压力依赖 + H2 子机理)")
    for i in excluded:
        r = gas.reaction(i)
        print(f"  排除 R{i+1}: {r.equation}  [{r.reaction_type}]")

    print(f"IDT 目标点 {len(IDT_TARGETS)} 个, LFS 目标点 {len(LFS_TARGETS)} 个")

    state_path = RESULTS_DIR / "ga_optimization_state.pkl"
    rng = np.random.default_rng(args.seed)

    if args.resume and state_path.exists():
        with open(state_path, "rb") as f:
            st = pickle.load(f)
        population, history, start_gen = st["population"], st["history"], st["gen"] + 1
        print(f"恢复: 从第 {start_gen} 代继续")
    else:
        population = [np.ones(n_opt) for _ in range(args.pop)]  # 论文: 从 M1 出发
        history = {"generation": [], "best": [], "avg": [], "worst": [],
                   "f_tau": [], "f_lfs": [], "f_a": [], "f_cpu": []}
        start_gen = 0

    t_ga = time.perf_counter()
    best_gene, best_fit = None, np.inf
    cache = {}

    if args.serial:
        _init_worker(opt_idx, IDT_TARGETS, LFS_TARGETS)
        class _SerialEx:
            def map(self, fn, jobs):
                return [fn(j) for j in jobs]
            def shutdown(self, wait=False):
                pass
        ex = _SerialEx()
    else:
        ex = ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker,
                                 initargs=(opt_idx, IDT_TARGETS, LFS_TARGETS))
    try:
        # 基准 (M1, 基因全 1) 的 CPU 时间参考
        base = list(ex.map(_evaluate, [np.ones(n_opt)]))[0]
        print(f"M1 基准: fitness={base[0]:.4f} (f_tau={base[1]:.4f}, "
              f"f_lfs={base[2]:.4f}, f_a={base[3]:.4f}, f_cpu={base[4]:.4f}, "
              f"t={base[5]:.1f}s)")

        for gen in range(start_gen, args.gens):
            jobs, keys = [], []
            for g in population:
                k = np.asarray(g, dtype=float).tobytes()
                if k not in cache:
                    jobs.append(np.asarray(g, dtype=float))
                    keys.append(k)
            if jobs:
                ex = eval_batch_resilient(ex, jobs, keys, cache,
                                         opt_idx, IDT_TARGETS,
                                         LFS_TARGETS, args)
            fit = np.array([cache[np.asarray(g, dtype=float).tobytes()][0]
                            for g in population])

            best_i = int(np.argmin(fit))
            best_gene = np.array(population[best_i], dtype=float)
            best_fit = float(fit[best_i])
            br = cache[best_gene.tobytes()]
            history["generation"].append(gen)
            history["best"].append(best_fit)
            history["avg"].append(float(np.mean(fit)))
            history["worst"].append(float(np.max(fit)))
            history["f_tau"].append(br[1])
            history["f_lfs"].append(br[2])
            history["f_a"].append(br[3])
            history["f_cpu"].append(br[4])

            if gen % 5 == 0 or gen == args.gens - 1:
                print(f"  Gen {gen:4d}: best={best_fit:.4f} avg={np.mean(fit):.4f} "
                      f"f_tau={br[1]:.3f} f_lfs={br[2]:.3f} f_a={br[3]:.3f} "
                      f"[{time.perf_counter()-t_ga:.0f}s, cache={len(cache)}]",
                      flush=True)

            # 进化: 精英 + 锦标赛 + 单点交叉 + 对数双向变异
            new_pop = [best_gene.copy()]
            while len(new_pop) < args.pop:
                p1 = tournament(population, fit, rng)
                p2 = tournament(population, fit, rng)
                if rng.random() < CROSSOVER_RATE:
                    pt = rng.integers(1, n_opt - 1)
                    c1 = np.concatenate([p1[:pt], p2[pt:]])
                    c2 = np.concatenate([p2[:pt], p1[pt:]])
                else:
                    c1, c2 = p1.copy(), p2.copy()
                for c in (c1, c2):
                    m = rng.random(n_opt) < MUTATION_RATE
                    c[m] = c[m] * 10.0 ** rng.uniform(-1, 1, int(m.sum()))
                    np.clip(c, A_MIN, A_MAX, out=c)
                new_pop.append(c1)
                if len(new_pop) < args.pop:
                    new_pop.append(c2)
            population = new_pop

            if (gen + 1) % 5 == 0 or gen == args.gens - 1:
                with open(state_path, "wb") as f:
                    pickle.dump({"population": population, "history": history,
                                 "gen": gen}, f)
                pd.DataFrame(history).to_csv(
                    RESULTS_DIR / "ga_optimization_history.csv", index=False)

    finally:
        ex.shutdown(wait=False)

    # ---- 输出 ----
    print(f"\n=== GA 优化完成: 最优 fitness {best_fit:.4f} ===")
    gene = best_gene
    gas_red = ct.Solution(str(REDUCED_MECH))
    gas_opt = ct.Solution(str(REDUCED_MECH))

    # 写出优化机理: 逐反应修改 A 因子
    import re
    text = Path(REDUCED_MECH).read_text()
    lines = text.splitlines(keepends=True)
    ridx = next(i for i, l in enumerate(lines) if l.startswith("reactions:"))
    blocks = [b for b in re.split(r"(?m)^(?=- equation: )",
                                  "".join(lines[ridx + 1:])) if b.strip()]
    assert len(blocks) == gas_red.n_reactions
    for k, i in enumerate(opt_idx):
        m = re.search(r"rate-constant:\s*\{A:\s*([\deE.+-]+)", blocks[i])
        A_old = float(m.group(1))
        A_new = A_old * gene[k]
        blocks[i] = blocks[i][:m.start(1)] + f"{A_new:.6g}" + blocks[i][m.end(1):]
    out_text = "".join(lines[:ridx + 1]) + "".join(blocks)
    out_path = MECH_DIR / "GA_Optimized_mechanism.yaml"
    out_path.write_text(out_text)
    g = ct.Solution(str(out_path))
    print(f"优化机理: {g.n_species} sp, {g.n_reactions} rxns -> {out_path}")

    # A 因子表 + 与论文原版 M2 对比
    paper_opt = MECH_DIR / "Optimized mechanism.yaml"
    rows = []
    if paper_opt.exists():
        gp = ct.Solution(str(paper_opt))
        eq2i = {gp.reaction(j).equation: j for j in range(gp.n_reactions)}
        for k, i in enumerate(opt_idx):
            eq = gas_red.reaction(i).equation
            A0 = gas_red.reaction(i).rate.pre_exponential_factor
            A_ours = A0 * gene[k]
            A_paper = None
            if eq in eq2i:
                r = gp.reaction(eq2i[eq])
                if hasattr(r.rate, "pre_exponential_factor"):
                    A_paper = r.rate.pre_exponential_factor
            rows.append({"index": i, "equation": eq,
                         "A0_M1": A0, "A_factor_ours": gene[k], "A_ours": A_ours,
                         "A_paper_M2": A_paper,
                         "A_factor_paper": (A_paper / A0) if A_paper else np.nan})
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_DIR / "ga_optimization_a_factors.csv", index=False)
        valid = df.dropna(subset=["A_factor_paper"])
        valid = valid[valid["A_factor_paper"].between(0.09, 11.0)]
        corr = np.corrcoef(np.log(valid["A_factor_ours"]),
                           np.log(valid["A_factor_paper"]))[0, 1]
        print(f"与论文原版 M2 的 A 因子对数相关系数: r = {corr:.3f} "
              f"(n={len(valid)})")
    print("A 因子表 -> results/ga_optimization_a_factors.csv")
    print("进化历史 -> results/ga_optimization_history.csv")


if __name__ == "__main__":
    main()
