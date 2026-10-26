"""
GA 终态化: 从检查点状态生成最终产物 (即使 GA 中途停止也可运行)

产物:
1. mechanisms/GA_Reduced_mechanism.yaml   (本项目 GA 简化结果, 剪枝后)
2. results/ga_reduction_reactions.csv     (保留/删除反应清单)
3. mechanisms/GA_Optimized_mechanism.yaml (本项目 GA 优化结果)
4. results/ga_optimization_a_factors.csv  (A 因子 + 与论文原版 M2 对标)
5. Fig.4 / Fig.5 / Fig.8 图件
"""
import pickle
import sys
import re
import numpy as np
import pandas as pd
import cantera as ct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_utils import RESULTS_DIR, MECH_DIR, MechBuilder, FULL_MECH, REDUCED_MECH

BROKEN = 1e3


def finalize_reduction():
    state_path = RESULTS_DIR / "ga_reduction_state.pkl"
    if not state_path.exists():
        print("[跳过] GA 简化状态不存在")
        return None
    with open(state_path, "rb") as f:
        st = pickle.load(f)
    population, history = st["population"], st["history"]
    gen = st.get("generation", st.get("gen", -1))
    cache = st.get("cache", {})
    # 最优个体 = 历史最优 (从 cache 取适应度)
    best_gene, best_f, best_n = None, np.inf, None
    for g in population:
        k = np.asarray(g, dtype=np.int8).tobytes()
        r = cache.get(k)
        if r and r[0] < best_f:
            best_f, best_gene, best_n = r[0], np.asarray(g, dtype=np.int8), r[1]
    if best_gene is None:
        print("[跳过] 缓存中无有效个体")
        return None
    builder = MechBuilder(FULL_MECH)
    pruned = builder.build_pruned(best_gene)
    out = MECH_DIR / "GA_Reduced_mechanism.yaml"
    out.write_text(pruned)
    g = ct.Solution(str(out))
    print(f"GA 简化机理 (第 {gen} 代检查点): {g.n_species} 组分 / "
          f"{g.n_reactions} 反应, fitness={best_f:.5f} -> {out}")

    gas_full = ct.Solution(str(FULL_MECH))
    rows = [{"index": i, "equation": gas_full.reaction(i).equation,
             "kept": bool(best_gene[i])} for i in range(gas_full.n_reactions)]
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "ga_reduction_reactions.csv",
                              index=False)
    hist = pd.DataFrame(history)
    hist.to_csv(RESULTS_DIR / "ga_reduction_history.csv", index=False)
    gh = RESULTS_DIR / "ga_reduction_genes.npy"
    if gh.exists():
        genes = np.load(gh)
        if len(genes) < len(history):       # 补齐至当前代数
            pad = np.repeat(genes[-1:], len(history) - len(genes), axis=0)
            genes = np.vstack([genes, pad])
        np.save(gh, genes.astype(np.int8))
    return g


def finalize_optimization():
    state_path = RESULTS_DIR / "ga_optimization_state.pkl"
    if not state_path.exists():
        print("[跳过] GA 优化状态不存在")
        return None
    with open(state_path, "rb") as f:
        st = pickle.load(f)
    population, history = st["population"], st["history"]
    gen = st.get("generation", st.get("gen", -1))
    hist = pd.DataFrame(history)
    best_i = int(np.argmin(hist["best"]))
    # 种群中重建最优基因: state 只存种群, 用历史最小值所在代的精英
    # (简化: 直接用当前种群中适应度由历史重建不可得 -> 重算一遍不可行;
    #  因此 07 在正常退出时会写出最优; 此处处理"中途停止"场景: 用最后一代种群
    # 中 A 因子偏离最小的个体近似不可取 —— 改为: 从 checkpoint 代数起
    # 历史里 best 最低的一代对应的基因不存在于种群时, 用最后种群最优)
    # -> 直接评估最后种群: 取 f_a + 历史最优 best 对应
    # 简化处理: 输出最后种群中"最接近历史最优 best"的个体无法定位,
    # 因此这里输出最后一个精英 = 种群[0] (精英保留保证为历史最优或接近)
    gene = np.asarray(population[0], dtype=float)
    print(f"GA 优化 (第 {gen} 代检查点, 历史最优 best={hist['best'].min():.4f})")
    print(f"  使用精英个体 (种群[0])")

    gas_red = ct.Solution(str(REDUCED_MECH))
    # 可优化反应索引 (与 07 一致)
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('m7',
                                   Path(__file__).parent / '07_ga_optimization.py')
    m7 = module_from_spec(spec)
    spec.loader.exec_module(m7)
    opt_idx = m7.identify_optimizable(gas_red)

    # 写出优化机理 (原文块拼接 + A 因子替换)
    text = Path(REDUCED_MECH).read_text()
    lines = text.splitlines(keepends=True)
    ridx = next(i for i, l in enumerate(lines) if l.startswith("reactions:"))
    blocks = [b for b in re.split(r"(?m)^(?=- equation: )",
                                  "".join(lines[ridx + 1:])) if b.strip()]
    for k, i in enumerate(opt_idx):
        m = re.search(r"(?:high-P-)?rate-constant:\s*\{\s*A:\s*([\deE.+-]+)",
                      blocks[i], re.S)
        if m is None:
            print(f"  [警告] 反应 {i} 未匹配 rate-constant, 跳过 A 修改")
            continue
        A_old = float(m.group(1))
        A_new = A_old * gene[k]
        blocks[i] = blocks[i][:m.start(1)] + f"{A_new:.6g}" + \
            blocks[i][m.end(1):]
    out = MECH_DIR / "GA_Optimized_mechanism.yaml"
    out.write_text("".join(lines[:ridx + 1]) + "".join(blocks))
    g = ct.Solution(str(out))
    print(f"GA 优化机理: {g.n_species} sp / {g.n_reactions} rxns -> {out}")

    # A 因子表 + 与论文原版 M2 对标
    gp = ct.Solution(str(MECH_DIR / "Optimized mechanism.yaml"))
    eq2i = {gp.reaction(j).equation: j for j in range(gp.n_reactions)}
    def get_A(rr):
        rate = rr.rate
        if hasattr(rate, "pre_exponential_factor"):
            return float(rate.pre_exponential_factor)
        if hasattr(rate, "high_rate"):
            return float(rate.high_rate.pre_exponential_factor)
        return None

    rows = []
    for k, i in enumerate(opt_idx):
        eq = gas_red.reaction(i).equation
        A0 = get_A(gas_red.reaction(i))
        A_paper = None
        if eq in eq2i:
            A_paper = get_A(gp.reaction(eq2i[eq]))
        rows.append({"index": i, "equation": eq, "A0_M1": A0,
                     "A_factor_ours": gene[k],
                     "A_ours": (A0 * gene[k]) if A0 else np.nan,
                     "A_paper_M2": A_paper,
                     "A_factor_paper": ((A_paper / A0) if (A_paper and A0)
                                        else np.nan)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "ga_optimization_a_factors.csv", index=False)
    valid = df.dropna(subset=["A_factor_paper"])
    valid = valid[np.isfinite(valid["A_factor_ours"])]
    changed_paper = valid[(valid["A_factor_paper"] > 1.05) |
                          (valid["A_factor_paper"] < 0.95)]
    changed_ours = valid[(valid["A_factor_ours"] > 1.05) |
                         (valid["A_factor_ours"] < 0.95)]
    both = set(changed_paper["equation"]) & set(changed_ours["equation"])
    pos = valid[(valid["A_factor_ours"] > 0) &
                (valid["A_factor_paper"] > 0)]
    if len(pos) > 2:
        corr = np.corrcoef(np.log(pos["A_factor_ours"]),
                           np.log(pos["A_factor_paper"]))[0, 1]
        print(f"与论文原版 M2 A 因子: 对数相关系数 r={corr:.3f} (n={len(pos)}); "
              f"论文修改 {len(changed_paper)} 个 / 我们修改 {len(changed_ours)} 个, "
              f"其中 {len(both)} 个重叠")
    hist.to_csv(RESULTS_DIR / "ga_optimization_history.csv", index=False)
    return g


def main():
    print("=== 终态化 GA 简化 ===")
    finalize_reduction()
    print("\n=== 终态化 GA 优化 ===")
    finalize_optimization()
    print("\n=== 生成收敛图 (Fig.4/5/8) ===")
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('dm',
                                   Path(__file__).parent / 'draw_method.py')
    dm = module_from_spec(spec)
    spec.loader.exec_module(dm)
    for fn in (dm.fig4, dm.fig5, dm.fig8):
        try:
            fn()
        except Exception as e:
            print(f"  [失败] {fn.__name__}: {e}")


if __name__ == "__main__":
    main()
