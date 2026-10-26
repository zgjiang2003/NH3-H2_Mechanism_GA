"""
方法学图件 - Fig. 1-5, 8 (与论文一一对应)

Fig. 1  研究方法流程图 (示意)
Fig. 2  遗传算法活动图 (示意)
Fig. 3  目标函数各项随 sigma/lambda 的变化 (解析曲线, 论文 Fig.3)
Fig. 4  GA 简化进化过程 (best/worst/avg 目标函数值, 论文 Fig.4)
Fig. 5  每代最优个体基因图 (论文 Fig.5)
Fig. 8  GA 优化进化过程 (论文 Fig.8)
输出: results/figs/
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_utils import RESULTS_DIR

FIGS = RESULTS_DIR / "figs"
FIGS.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 9,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


def _save(fig, name):
    fig.savefig(FIGS / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def fig1():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axis("off")
    boxes = [
        (0.02, 0.78, 0.28, 0.16, "详细机理\n(Zhang 2021,\n34 sp / 263 rxn)"),
        (0.36, 0.78, 0.28, 0.16, "敏感性分析\n(IDT, 8 工况)"),
        (0.70, 0.78, 0.28, 0.16, "GA 简化\n(二进制编码)"),
        (0.70, 0.50, 0.28, 0.16, "Mechanism 1\n(29 sp / 63 rxn)"),
        (0.36, 0.50, 0.28, 0.16, "GA 优化\n(实数编码, A 因子)"),
        (0.02, 0.50, 0.28, 0.16, "Mechanism 2\n(优化后)"),
        (0.02, 0.18, 0.28, 0.16, "验证: IDT / LFS\nJSR / 火焰结构"),
        (0.36, 0.18, 0.28, 0.16, "验证: HCCI\n缸内压力"),
        (0.70, 0.18, 0.28, 0.16, "发动机 CFD\n适用性"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02",
                                    fc="#eaf2f8" if "机理" not in t else "#d5f5e3",
                                    ec="#2c3e50", lw=1.0))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.5)
    arrows = [(0.30, 0.86, 0.36, 0.86), (0.64, 0.86, 0.70, 0.86),
              (0.84, 0.78, 0.84, 0.66), (0.70, 0.58, 0.64, 0.58),
              (0.36, 0.58, 0.30, 0.58), (0.16, 0.50, 0.16, 0.34),
              (0.30, 0.26, 0.36, 0.26), (0.64, 0.26, 0.70, 0.26)]
    for x0, y0, x1, y1 in arrows:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color="#2c3e50"))
    ax.set_title("Fig. 1  Research method flow chart", fontsize=10)
    _save(fig, "fig_01_flowchart.png")


def fig2():
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.axis("off")
    steps = [
        ("开始", "#d5f5e3"),
        ("初始化种群\n(基因全 1 = 完整机理 / A 因子全 1)", "#eaf2f8"),
        ("评估个体\n(f = f_tau + f_Ts + f_N + f_CPU\n 或 f_tau + f_LFS + f_A + f_CPU)", "#eaf2f8"),
        ("选择 (锦标赛 + 精英保留)", "#eaf2f8"),
        ("单点交叉\n(简化: p=0.2 / 优化: p=0.3)", "#eaf2f8"),
        ("双向变异\n(简化: p=0.03 / 优化: p=0.06)", "#eaf2f8"),
        ("达到代数?", "#fdf2e9"),
        ("输出最优机理 (Mechanism 1 / 2)", "#d5f5e3"),
    ]
    y = 0.95
    coords = []
    for t, c in steps:
        ax.add_patch(FancyBboxPatch((0.12, y - 0.10), 0.76, 0.09,
                                    boxstyle="round,pad=0.012", fc=c,
                                    ec="#2c3e50", lw=0.9))
        ax.text(0.5, y - 0.055, t, ha="center", va="center", fontsize=8.2)
        coords.append(y - 0.055)
        y -= 0.125
    for i in range(len(steps) - 1):
        ax.add_patch(FancyArrowPatch((0.5, coords[i] - 0.045),
                                     (0.5, coords[i + 1] + 0.045),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color="#2c3e50"))
    # 循环回路
    ax.add_patch(FancyArrowPatch((0.88, coords[6]), (0.88, coords[2]),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color="#c0392b",
                                 connectionstyle="arc3,rad=0.35"))
    ax.text(0.92, (coords[2] + coords[6]) / 2, "否", color="#c0392b",
            fontsize=9)
    ax.text(0.5, coords[6] - 0.045, "", fontsize=8)
    ax.text(0.55, coords[6] - 0.02, "是", fontsize=8)
    ax.set_title("Fig. 2  Activity chart of the genetic algorithm",
                 fontsize=10)
    _save(fig, "fig_02_ga_activity.png")


def fig3():
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.7))
    e = np.linspace(0, 3, 200)           # |tau_r - tau_0| / tau_0
    for s in (1, 2, 4):
        axes[0].plot(e, np.log(1 + s * e), label=f"sigma={s}")
    axes[0].set_xlabel("|tau_r - tau_0| / tau_0")
    axes[0].set_ylabel("f_tau")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[0].set_title("(a) f_tau", fontsize=9)

    r = np.linspace(0, 1.2, 200)
    for s in (1, 2, 4):
        axes[1].plot(r, np.log(1 + s * r), label=f"sigma={s}")
    axes[1].set_xlabel("|Ts_r - Ts_0| / (Ts_0 - Tin)")
    axes[1].set_ylabel("f_Ts")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    axes[1].set_title("(b) f_Ts", fontsize=9)

    xr = np.linspace(0, 1.2, 200)
    for s, lam in [(4, 0.8), (6, 0.8), (6, 0.6), (8, 0.8)]:
        axes[2].plot(xr, 1 / (1 + np.exp(s * (1 - xr / lam))),
                     label=f"sigma={s}, lambda={lam}")
    axes[2].set_xlabel("Nr / NR")
    axes[2].set_ylabel("f_N")
    axes[2].legend(fontsize=6.5); axes[2].grid(alpha=0.3)
    axes[2].set_title("(c) f_N", fontsize=9)

    for s, lam in [(4, 4), (6, 4), (6, 6), (8, 4)]:
        axes[3].plot(xr, 1 / (1 + np.exp(s * (1 - xr / lam))),
                     label=f"sigma={s}, lambda={lam}")
    axes[3].set_xlabel("t_CPU(r) / t_CPU(0)")
    axes[3].set_ylabel("f_CPU")
    axes[3].legend(fontsize=6.5); axes[3].grid(alpha=0.3)
    axes[3].set_title("(d) f_CPU", fontsize=9)
    fig.suptitle("Fig. 3  Objective function terms with different "
                 "sigma and lambda", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, "fig_03_objective_terms.png")


def fig4():
    f = RESULTS_DIR / "ga_reduction_history.csv"
    if not f.exists():
        print("  [跳过] ga_reduction_history.csv 不存在"); return
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(df["generation"], df["best"], "k-", lw=1.5, label="Fittest")
    ax.plot(df["generation"], df["avg"], "b--", lw=1.2, label="Average")
    ax.plot(df["generation"], df["worst"], "r:", lw=1.2, label="Worst")
    ax.set_xlabel("Generation"); ax.set_ylabel("Overall objective value")
    ax.set_yscale("log")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Fig. 4  GA reduction convergence", fontsize=9.5)
    _save(fig, "fig_04_reduction_convergence.png")


def fig5():
    f = RESULTS_DIR / "ga_reduction_genes.npy"
    if not f.exists():
        print("  [跳过] ga_reduction_genes.npy 不存在"); return
    genes = np.load(f)      # (n_gen, n_rxn)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.imshow(genes.T.astype(float), aspect="auto", cmap="Greys",
              interpolation="nearest", vmin=0, vmax=1,
              extent=[0, len(genes), 0, genes.shape[1]])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Reaction index")
    ax.set_title("Fig. 5  Genes of the fittest individual per generation "
                 "(1=retained)", fontsize=9)
    _save(fig, "fig_05_gene_map.png")


def fig8():
    f = RESULTS_DIR / "ga_optimization_history.csv"
    if not f.exists():
        print("  [跳过] ga_optimization_history.csv 不存在"); return
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(df["generation"], df["best"], "k-", lw=1.5, label="Fittest")
    ax.plot(df["generation"], df["avg"], "b--", lw=1.2, label="Average")
    ax.plot(df["generation"], df["worst"], "r:", lw=1.0, label="Worst")
    ax.set_xlabel("Generation"); ax.set_ylabel("Overall objective value")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Fig. 8  GA optimization convergence", fontsize=9.5)
    _save(fig, "fig_08_optimization_convergence.png")

    # 分解项
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for col, lab in [("f_tau", "f_tau"), ("f_lfs", "f_LFS"),
                     ("f_a", "f_A"), ("f_cpu", "f_CPU")]:
        if col in df:
            ax.plot(df["generation"], df[col], lw=1.3, label=lab)
    ax.set_xlabel("Generation"); ax.set_ylabel("Objective term value")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Objective terms during optimization", fontsize=9.5)
    _save(fig, "fig_08b_optimization_terms.png")


def main():
    print("生成方法学图件:")
    for fn in (fig1, fig2, fig3, fig4, fig5, fig8):
        try:
            fn()
        except Exception as e:
            print(f"  [失败] {fn.__name__}: {e}")
    print(f"图件目录: {FIGS}")


if __name__ == "__main__":
    main()
