"""
JSR 绘图脚本 - 复现论文 Fig. 16
绘制 JSR 物种浓度 vs 温度，包含：
- Full / Reduced / Optimized 三条模拟曲线
- 实验数据对比 (Zhang et al. [21])
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experimental_data import JSR_ZHANG

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['mathtext.default'] = 'regular'

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
df = pd.read_csv(RESULTS / "jsr_sweep.csv")

colors = {"Full": "black", "Reduced": "blue", "Optimized": "red"}
linestyles = {"Full": "--", "Reduced": "-.", "Optimized": "-"}
species = ["NH3", "H2O", "NO", "N2O"]
XH2_list = sorted(df["XH2_fuel"].unique())

# ============================================================
# Fig. 16 风格: JSR species vs T, 每个 XH2 一个子图
# ============================================================

for XH2 in XH2_list:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, sp in enumerate(species):
        ax = axes[idx]
        for mech in ["Full", "Reduced", "Optimized"]:
            sub = df[(df["mechanism"] == mech) &
                     (df["XH2_fuel"] == XH2)]
            sub = sub.sort_values("T_K")
            if len(sub) > 0:
                col = f"X_{sp}" if f"X_{sp}" in sub.columns else None
                if col is None:
                    continue
                ax.semilogy(sub["T_K"], sub[col],
                            color=colors[mech], linestyle=linestyles[mech],
                            linewidth=1.5, alpha=0.8, label=mech)

        # 实验数据
        if XH2 in JSR_ZHANG:
            exp = JSR_ZHANG[XH2]
            if f"X_{sp}" in exp:
                ax.scatter(exp["T_K"], exp[f"X_{sp}"],
                           marker='o', facecolors='none', edgecolors='green',
                           s=40, linewidths=1.5, zorder=5, label="Exp")

        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(f"Mole fraction of {sp}")
        ax.set_title(f"{sp}, X_H2 = {XH2}")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    save_path = RESULTS / f"fig_jsr_XH2_{XH2}.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"JSR figure for XH2={XH2} saved.")

# ============================================================
# 综合对比图: 所有 XH2 在一个大图中
# ============================================================
fig, axes = plt.subplots(len(species), len(XH2_list), figsize=(16, 12))

for row, sp in enumerate(species):
    for col, XH2 in enumerate(XH2_list):
        ax = axes[row, col]

        for mech in ["Full", "Reduced", "Optimized"]:
            sub = df[(df["mechanism"] == mech) &
                     (df["XH2_fuel"] == XH2)]
            sub = sub.sort_values("T_K")
            col_name = f"X_{sp}"
            if len(sub) > 0 and col_name in sub.columns:
                ax.semilogy(sub["T_K"], sub[col_name],
                            color=colors[mech], linestyle=linestyles[mech],
                            linewidth=1.5, alpha=0.8)

        # 实验数据
        if XH2 in JSR_ZHANG:
            exp = JSR_ZHANG[XH2]
            if f"X_{sp}" in exp:
                ax.scatter(exp["T_K"], exp[f"X_{sp}"],
                           marker='o', facecolors='none', edgecolors='green',
                           s=20, linewidths=1, zorder=5)

        if row == len(species) - 1:
            ax.set_xlabel("T (K)")
        if col == 0:
            ax.set_ylabel(sp)
        if row == 0:
            ax.set_title(f"X_H2 = {XH2}")
        ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(RESULTS / "fig_jsr_comparison.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("JSR comparison figure saved.")

print("All JSR figures saved.")