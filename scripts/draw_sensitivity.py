"""
敏感性分析绘图脚本 - 复现论文 Fig. 6
绘制归一化敏感性系数条形图，标记被保留的反应 (*)
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['mathtext.default'] = 'regular'

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

df = pd.read_csv(RESULTS / "sensitivity.csv")

# 读取简化机理中的反应列表，用于标记 *
import cantera as ct
MECH_DIR = ROOT / "mechanisms"
gas_reduced = ct.Solution(str(MECH_DIR / "Reduced mechanism.yaml"))
reduced_reactions = set()
for i in range(gas_reduced.n_reactions):
    reduced_reactions.add(gas_reduced.reaction(i).equation)

# 为每个工况绘制 Fig. 6 风格的敏感性条形图
conditions = df.groupby(["T_K", "P_atm", "XH2_fuel"]).size().reset_index()[["T_K", "P_atm", "XH2_fuel"]]

for _, cond in conditions.iterrows():
    T = cond["T_K"]
    P = cond["P_atm"]
    XH2 = cond["XH2_fuel"]

    # 只绘制 Full 机理的敏感性 (论文 Fig.6 用的是详细机理)
    sub = df[(df["mechanism"] == "Full") &
             (df["T_K"] == T) &
             (df["P_atm"] == P) &
             (df["XH2_fuel"] == XH2)]

    if len(sub) == 0:
        continue

    # 按归一化敏感性绝对值排序，取 top 20
    sub = sub.sort_values("sensitivity_normalized", key=abs, ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))

    reactions = sub["reaction"].values
    sens_vals = sub["sensitivity_normalized"].values

    # 标记被保留的反应
    labels = []
    for rxn in reactions:
        if rxn in reduced_reactions:
            labels.append(f"{rxn} *")
        else:
            labels.append(rxn)

    colors = ['red' if s > 0 else 'blue' for s in sens_vals]

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, sens_vals, color=colors, alpha=0.7, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Normalized sensitivity coefficient")
    ax.set_title(f"IDT Sensitivity, XH2={XH2}, P={P} atm, T={T} K")
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig.savefig(RESULTS / f"fig_sensitivity_XH2{XH2}_P{P}atm_T{T}K.png",
                dpi=300, bbox_inches='tight')
    plt.close(fig)

# 综合图: 所有工况的 top 反应
print("Creating combined sensitivity figure...")
fig, axes = plt.subplots(2, 4, figsize=(20, 12))
axes = axes.flatten()

for idx, (_, cond) in enumerate(conditions.iterrows()):
    if idx >= 8:
        break
    ax = axes[idx]
    T = cond["T_K"]
    P = cond["P_atm"]
    XH2 = cond["XH2_fuel"]

    sub = df[(df["mechanism"] == "Full") &
             (df["T_K"] == T) &
             (df["P_atm"] == P) &
             (df["XH2_fuel"] == XH2)]

    if len(sub) == 0:
        continue

    sub = sub.sort_values("sensitivity_normalized", key=abs, ascending=False).head(10)
    reactions = sub["reaction"].values
    sens_vals = sub["sensitivity_normalized"].values

    labels = []
    for rxn in reactions:
        if rxn in reduced_reactions:
            labels.append(f"{rxn} *")
        else:
            labels.append(rxn)

    colors = ['red' if s > 0 else 'blue' for s in sens_vals]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, sens_vals, color=colors, alpha=0.7, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_title(f"XH2={XH2}, P={P}atm, T={T}K", fontsize=8)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

plt.suptitle("IDT Normalized Sensitivity Analysis (Fig.6)", fontsize=12)
plt.tight_layout()
fig.savefig(RESULTS / "fig_sensitivity_fig6.png", dpi=300, bbox_inches='tight')
plt.close(fig)

print("All sensitivity figures saved.")
