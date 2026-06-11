"""
IDT 绘图脚本 - 复现论文 Fig. 9 和 Fig. 10
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experimental_data import (
    IDT_CHEN_1_2ATM, IDT_CHEN_10ATM,
    IDT_HE, IDT_MATHIEU,
)

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['mathtext.default'] = 'regular'

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
df = pd.read_csv(RESULTS / "idt_sweep.csv")

colors = {"Full": "black", "Reduced": "blue", "Optimized": "red"}
linestyles = {"Full": "--", "Reduced": "-.", "Optimized": "-"}

# 过滤无效数据（IDT_ms > 5000 表示未点燃）
df = df[df["IDT_ms"] < 5000].copy()

# ============================================================
# Fig. 9: IDT vs 1000/T, 每个 XH2 一个子图
# ============================================================
XH2_list = [0.0, 0.05, 0.3, 0.7]
P_list = [1.2, 10.0]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, XH2 in enumerate(XH2_list):
    ax = axes[idx]
    for P in P_list:
        for mech in ["Full", "Reduced", "Optimized"]:
            sub = df[(df["mechanism"] == mech) &
                     (df["XH2_fuel"] == XH2) &
                     (df["P_atm"] == P)]
            sub = sub.sort_values("1000/T")
            if len(sub) > 0:
                ax.semilogy(
                    sub["1000/T"], sub["IDT_ms"],
                    color=colors[mech], linestyle=linestyles[mech],
                    linewidth=1.5, alpha=0.8,
                    label=f"{mech}" if P == 1.2 else None,
                )

    # 实验数据
    if XH2 in IDT_CHEN_1_2ATM:
        pts = IDT_CHEN_1_2ATM[XH2]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                   marker='o', facecolors='none', edgecolors='green',
                   s=40, linewidths=1.5, zorder=5, label="Exp 1.2 atm")
    if XH2 in IDT_CHEN_10ATM:
        pts = IDT_CHEN_10ATM[XH2]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                   marker='s', facecolors='none', edgecolors='orange',
                   s=40, linewidths=1.5, zorder=5, label="Exp 10 atm")

    ax.set_xlabel("1000/T (1/K)")
    ax.set_ylabel("Ignition delay time (ms)")
    ax.set_title(f"X_H2 in fuel = {XH2}")
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7, loc='upper left')

handles = []
for mech in ["Full", "Reduced", "Optimized"]:
    handles.append(plt.Line2D([0], [0], color=colors[mech],
                              linestyle=linestyles[mech], linewidth=1.5, label=mech))
handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                           markeredgecolor='green', markersize=8, label='Exp 1.2 atm'))
handles.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
                           markeredgecolor='orange', markersize=8, label='Exp 10 atm'))
fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
fig.savefig(RESULTS / "fig_idt_comparison.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("IDT Fig.9 saved.")

# ============================================================
# Fig. 10: 纯 NH3 IDT, 不同 P 和 phi
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# He et al. [44]: NH3/air, P=20,40 bar
ax = axes[0]
for P_bar in [20, 40]:
    for mech in ["Full", "Reduced", "Optimized"]:
        sub = df[(df["mechanism"] == mech) &
                 (df["XH2_fuel"] == 0.0) &
                 (df["P_atm"] == P_bar)]
        sub = sub.sort_values("1000/T")
        if len(sub) > 0:
            ax.semilogy(sub["1000/T"], sub["IDT_ms"],
                        color=colors[mech], linestyle=linestyles[mech],
                        linewidth=1.5, alpha=0.8,
                        label=f"{mech}" if P_bar == 20 else None)

for (P_bar, phi), pts in IDT_HE.items():
    marker = 'o' if P_bar == 20 else 's'
    ax.scatter([p[0] for p in pts], [p[1] for p in pts],
               marker=marker, facecolors='none',
               edgecolors='green' if P_bar == 20 else 'orange',
               s=40, linewidths=1.5, zorder=5, label=f"Exp {P_bar} bar")

ax.set_xlabel("1000/T (1/K)")
ax.set_ylabel("Ignition delay time (ms)")
ax.set_title("NH3/air, He et al. [44]")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Mathieu et al. [17]: NH3/O2/Ar, P~1.4 atm
ax = axes[1]
for phi_val in [0.5, 1.0, 2.0]:
    for mech in ["Full", "Reduced", "Optimized"]:
        sub = df[(df["mechanism"] == mech) &
                 (df["XH2_fuel"] == 0.0) &
                 (df["P_atm"] == 1.4) &
                 (df["phi"] == phi_val) &
                 (df["diluent"] == "AR")]
        sub = sub.sort_values("1000/T")
        if len(sub) > 0:
            ax.semilogy(sub["1000/T"], sub["IDT_ms"],
                        color=colors[mech], linestyle=linestyles[mech],
                        linewidth=1.5, alpha=0.8,
                        label=f"{mech}" if phi_val == 1.0 else None)

for phi, pts in IDT_MATHIEU.items():
    markers = {0.5: '^', 1.0: 'o', 2.0: 'v'}
    ax.scatter([p[0] for p in pts], [p[1] for p in pts],
               marker=markers[phi], facecolors='none', edgecolors='blue',
               s=40, linewidths=1.5, zorder=5, label=f"Exp phi={phi}")

ax.set_xlabel("1000/T (1/K)")
ax.set_ylabel("Ignition delay time (ms)")
ax.set_title("NH3/O2/Ar, Mathieu et al. [17]")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(RESULTS / "fig_idt_fig10.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("IDT Fig.10 saved.")

# 单独每个 XH2 的详细图
for XH2 in XH2_list:
    fig, ax = plt.subplots(figsize=(8, 6))
    for mech in ["Full", "Reduced", "Optimized"]:
        for P in P_list:
            sub = df[(df["mechanism"] == mech) &
                     (df["XH2_fuel"] == XH2) &
                     (df["P_atm"] == P)]
            sub = sub.sort_values("1000/T")
            if len(sub) > 0:
                ax.semilogy(sub["1000/T"], sub["IDT_ms"],
                            color=colors[mech],
                            linestyle="--" if P == 10.0 else linestyles[mech],
                            linewidth=1.5, label=f"{mech}, {P} atm")

    if XH2 in IDT_CHEN_1_2ATM:
        pts = IDT_CHEN_1_2ATM[XH2]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                   marker='o', facecolors='none', edgecolors='green',
                   s=50, linewidths=1.5, zorder=5, label="Exp 1.2 atm")
    if XH2 in IDT_CHEN_10ATM:
        pts = IDT_CHEN_10ATM[XH2]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                   marker='s', facecolors='none', edgecolors='orange',
                   s=50, linewidths=1.5, zorder=5, label="Exp 10 atm")

    ax.set_xlabel("1000/T (1/K)")
    ax.set_ylabel("Ignition delay time (ms)")
    ax.set_title(f"NH3/H2 Ignition Delay Times, X_H2 = {XH2}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(RESULTS / f"fig_idt_XH2_{XH2}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

print("All IDT figures saved.")
