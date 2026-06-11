"""
LFS 绘图脚本 - 复现论文 Fig. 11-15
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experimental_data import (
    LFS_LHUILLIER, LFS_WANG, LFS_ICHIKAWA,
    LFS_SHRESTHA, LFS_LHUILLIER_TEMP,
)

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['mathtext.default'] = 'regular'

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
df = pd.read_csv(RESULTS / "lfs_sweep.csv")

colors = {"Full": "black", "Reduced": "blue", "Optimized": "red"}
linestyles = {"Full": "--", "Reduced": "-.", "Optimized": "-"}

# ============================================================
# Fig. 11: LFS vs phi, 不同 XH2, Tu=298K, P=1 atm
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, XH2 in enumerate([0.0, 0.2, 0.4, 0.6, 0.8]):
    ax = axes[idx]
    for mech in ["Full", "Reduced", "Optimized"]:
        sub = df[(df["mechanism"] == mech) &
                 (df["XH2_fuel"] == XH2) &
                 (df["P_atm"] == 1.0) &
                 (df["T_unburned_K"] == 298.0)]
        sub = sub.sort_values("phi")
        if len(sub) > 0:
            ax.plot(sub["phi"], sub["Su_m_s"],
                    color=colors[mech], linestyle=linestyles[mech],
                    linewidth=1.5, label=mech)

    if XH2 in LFS_LHUILLIER:
        pts = LFS_LHUILLIER[XH2]
        ax.scatter([p[0] for p in pts], [p[1] / 100.0 for p in pts],
                   marker='o', facecolors='none', edgecolors='green',
                   s=40, linewidths=1.5, zorder=5, label="Exp")

    ax.set_xlabel("Equivalence ratio")
    ax.set_ylabel("Laminar flame speed (m/s)")
    ax.set_title(f"X_H2 = {XH2}")
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7)

ax = axes[5]
ax.text(0.5, 0.5, "H2 laminar flame speed\n(see Fig. S3)",
        ha='center', va='center', transform=ax.transAxes)
ax.set_title("H2")

plt.tight_layout()
fig.savefig(RESULTS / "fig_lfs_fig11.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("LFS Fig.11 saved.")

# ============================================================
# Fig. 13: LFS vs XH2, phi=1.0, Tu=298K, various P
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

for P in [1.0, 3.0, 5.0]:
    for mech in ["Full", "Reduced", "Optimized"]:
        sub = df[(df["mechanism"] == mech) &
                 (df["P_atm"] == P) &
                 (df["T_unburned_K"] == 298.0) &
                 (df["phi"] == 1.0)]
        sub = sub.sort_values("XH2_fuel")
        if len(sub) > 0:
            ax.plot(sub["XH2_fuel"], sub["Su_m_s"],
                    color=colors[mech], linestyle=linestyles[mech],
                    linewidth=1.5, alpha=0.8,
                    label=f"{mech}" if P == 1.0 else None)

    if P in LFS_ICHIKAWA:
        pts = LFS_ICHIKAWA[P]
        markers = {1.0: 'o', 3.0: 's', 5.0: '^'}
        ax.scatter([p[0] for p in pts], [p[1] / 100.0 for p in pts],
                   marker=markers[P], facecolors='none', edgecolors='green',
                   s=50, linewidths=1.5, zorder=5, label=f"Exp {P} atm")

ax.set_xlabel("X_H2 in fuel")
ax.set_ylabel("Laminar flame speed (m/s)")
ax.set_title("NH3/H2/air, phi=1.0, Tu=298K [48]")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(RESULTS / "fig_lfs_fig13.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("LFS Fig.13 saved.")

# ============================================================
# Fig. 14: LFS vs phi, XH2=0.4, Tu=298K, various P
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

for P in [1.0, 3.0, 5.0]:
    for mech in ["Full", "Reduced", "Optimized"]:
        sub = df[(df["mechanism"] == mech) &
                 (df["XH2_fuel"] == 0.4) &
                 (df["P_atm"] == P) &
                 (df["T_unburned_K"] == 298.0)]
        sub = sub.sort_values("phi")
        if len(sub) > 0:
            ax.plot(sub["phi"], sub["Su_m_s"],
                    color=colors[mech], linestyle=linestyles[mech],
                    linewidth=1.5, alpha=0.8,
                    label=f"{mech}" if P == 1.0 else None)

    if P in LFS_SHRESTHA:
        pts = LFS_SHRESTHA[P]
        markers = {1.0: 'o', 3.0: 's', 5.0: '^'}
        ax.scatter([p[0] for p in pts], [p[1] / 100.0 for p in pts],
                   marker=markers[P], facecolors='none', edgecolors='green',
                   s=50, linewidths=1.5, zorder=5, label=f"Exp {P} atm")

ax.set_xlabel("Equivalence ratio")
ax.set_ylabel("Laminar flame speed (m/s)")
ax.set_title("NH3/H2/air, XH2=0.4, Tu=298K [49]")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(RESULTS / "fig_lfs_fig14.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("LFS Fig.14 saved.")

# ============================================================
# Fig. 15: LFS vs XH2, 不同 Tu, phi=1.0, P=1 atm
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

for Tu in [298.0, 373.0, 423.0, 473.0]:
    for mech in ["Full", "Reduced", "Optimized"]:
        sub = df[(df["mechanism"] == mech) &
                 (df["T_unburned_K"] == Tu) &
                 (df["P_atm"] == 1.0) &
                 (df["phi"] == 1.0)]
        sub = sub.sort_values("XH2_fuel")
        if len(sub) > 0:
            ax.plot(sub["XH2_fuel"], sub["Su_m_s"],
                    color=colors[mech], linestyle=linestyles[mech],
                    linewidth=1.5, alpha=0.8,
                    label=f"{mech}" if Tu == 298.0 else None)

    if Tu in LFS_LHUILLIER_TEMP:
        pts = LFS_LHUILLIER_TEMP[Tu]
        markers = {298.0: 'o', 373.0: 's', 423.0: '^', 473.0: 'd'}
        ax.scatter([p[0] for p in pts], [p[1] / 100.0 for p in pts],
                   marker=markers[Tu], facecolors='none', edgecolors='green',
                   s=50, linewidths=1.5, zorder=5, label=f"Exp {Tu}K")

ax.set_xlabel("X_H2 in fuel")
ax.set_ylabel("Laminar flame speed (m/s)")
ax.set_title("NH3/H2/air, phi=1.0, P=1 atm [45]")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(RESULTS / "fig_lfs_fig15.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("LFS Fig.15 saved.")

# ============================================================
# 综合对比图: LFS vs Tu
# ============================================================
XH2_list = sorted(df["XH2_fuel"].unique())
P_list = sorted(df["P_atm"].unique())

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, XH2 in enumerate([0.0, 0.05, 0.3, 0.7]):
    ax = axes[idx]
    for P in P_list:
        for mech in ["Full", "Reduced", "Optimized"]:
            sub = df[(df["mechanism"] == mech) &
                     (df["XH2_fuel"] == XH2) &
                     (df["P_atm"] == P) &
                     (df["phi"] == 1.0)]
            sub = sub.sort_values("T_unburned_K")
            if len(sub) > 0:
                ax.plot(sub["T_unburned_K"], sub["Su_m_s"],
                        color=colors[mech], linestyle=linestyles[mech],
                        linewidth=1.5, marker='o', markersize=5,
                        label=f"{mech}" if P == P_list[0] else None)

    ax.set_xlabel("Unburned temperature (K)")
    ax.set_ylabel("Laminar flame speed (m/s)")
    ax.set_title(f"X_H2 = {XH2}, phi=1.0")
    ax.grid(True, alpha=0.3)

handles = []
for mech in ["Full", "Reduced", "Optimized"]:
    handles.append(plt.Line2D([0], [0], color=colors[mech],
                              linestyle=linestyles[mech], linewidth=1.5, label=mech))
fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
fig.savefig(RESULTS / "fig_lfs_comparison.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("LFS comparison figure saved.")

print("All LFS figures saved.")
