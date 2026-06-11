"""
Flame 绘图脚本 - 复现论文 Fig. 17-18
绘制预混火焰结构，包含：
- Full / Reduced / Optimized 三条模拟曲线
- 实验数据对比
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experimental_data import FLAME_PURE_NH3, FLAME_NH3_H2, FLAME_NH3_NO_AR

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['mathtext.default'] = 'regular'

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
df = pd.read_csv(RESULTS / "premixed_flame.csv")

colors = {"Full": "black", "Reduced": "blue", "Optimized": "red"}
linestyles = {"Full": "--", "Reduced": "-.", "Optimized": "-"}

species = ["X_NH3", "X_NO", "X_N2O", "X_H2O"]
titles = ["NH3", "NO", "N2O", "H2O"]
configs = sorted(df["config"].unique())

# ============================================================
# 每个火焰条件单独画图
# ============================================================
for config in configs:
    sub = df[df["config"] == config]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (sp, title) in enumerate(zip(species, titles)):
        ax = axes[idx]
        for mech in ["Full", "Reduced", "Optimized"]:
            sub2 = sub[sub["mechanism"] == mech].sort_values("distance_cm")
            if len(sub2) > 0 and sp in sub2.columns:
                ax.plot(sub2["distance_cm"], sub2[sp],
                        color=colors[mech], linestyle=linestyles[mech],
                        linewidth=1.5, alpha=0.8, label=mech)

        ax.set_xlabel("Distance (cm)")
        ax.set_ylabel(f"Mole fraction of {title}")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    plt.suptitle(f"Flame structure: {config}")
    plt.tight_layout()
    save_path = RESULTS / f"fig_flame_{config}.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Flame figure for {config} saved.")

# ============================================================
# Fig. 17 风格: 纯 NH3 火焰结构 vs 实验
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (sp, title) in enumerate(zip(species, titles)):
    ax = axes[idx]
    for mech in ["Full", "Reduced", "Optimized"]:
        sub2 = df[(df["mechanism"] == mech) &
                  (df["config"] == "pure_NH3_350K")].sort_values("distance_cm")
        if len(sub2) > 0 and sp in sub2.columns:
            ax.plot(sub2["distance_cm"], sub2[sp],
                    color=colors[mech], linestyle=linestyles[mech],
                    linewidth=1.5, alpha=0.8, label=mech)

    # 实验数据
    if sp in FLAME_PURE_NH3:
        ax.scatter(FLAME_PURE_NH3["distance_cm"], FLAME_PURE_NH3[sp],
                   marker='o', facecolors='none', edgecolors='green',
                   s=40, linewidths=1.5, zorder=5, label="Exp")

    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel(f"{title} mole fraction")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7)

plt.suptitle("Pure NH3 flame structure, phi=1.0, Tu=350K (Fig.17)")
plt.tight_layout()
fig.savefig(RESULTS / "fig_flame_pure_nh3.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("Pure NH3 flame Fig.17 saved.")

# ============================================================
# NH3/H2 (0.8/0.2) 火焰结构 vs 实验
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (sp, title) in enumerate(zip(species, titles)):
    ax = axes[idx]
    for mech in ["Full", "Reduced", "Optimized"]:
        sub2 = df[(df["mechanism"] == mech) &
                  (df["config"] == "NH3_H2_0.8_0.2_350K")].sort_values("distance_cm")
        if len(sub2) > 0 and sp in sub2.columns:
            ax.plot(sub2["distance_cm"], sub2[sp],
                    color=colors[mech], linestyle=linestyles[mech],
                    linewidth=1.5, alpha=0.8, label=mech)

    # 实验数据
    if sp in FLAME_NH3_H2:
        ax.scatter(FLAME_NH3_H2["distance_cm"], FLAME_NH3_H2[sp],
                   marker='o', facecolors='none', edgecolors='green',
                   s=40, linewidths=1.5, zorder=5, label="Exp")

    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel(f"{title} mole fraction")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7)

plt.suptitle("NH3/H2 (0.8/0.2) flame structure, phi=1.0, Tu=350K (Fig.17)")
plt.tight_layout()
fig.savefig(RESULTS / "fig_flame_nh3_h2.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("NH3/H2 flame Fig.17 saved.")

# ============================================================
# Fig. 18 风格: NH3/NO/Ar 火焰结构 - Vandooren et al. [51]
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (sp, title) in enumerate(zip(species, titles)):
    ax = axes[idx]
    for mech in ["Full", "Reduced", "Optimized"]:
        sub2 = df[(df["mechanism"] == mech) &
                  (df["config"] == "NH3_NO_Ar_phi1.46")].sort_values("distance_cm")
        if len(sub2) > 0 and sp in sub2.columns:
            ax.plot(sub2["distance_cm"], sub2[sp],
                    color=colors[mech], linestyle=linestyles[mech],
                    linewidth=1.5, alpha=0.8, label=mech)

    # 实验数据
    if sp in FLAME_NH3_NO_AR:
        ax.scatter(FLAME_NH3_NO_AR["distance_cm"], FLAME_NH3_NO_AR[sp],
                   marker='o', facecolors='none', edgecolors='green',
                   s=40, linewidths=1.5, zorder=5, label="Exp")

    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel(f"{title} mole fraction")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7)

plt.suptitle("NH3/NO/Ar flame structure, phi=1.46 [51] (Fig.18)")
plt.tight_layout()
fig.savefig(RESULTS / "fig_flame_nh3_no_ar.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("NH3/NO/Ar flame Fig.18 saved.")

# ============================================================
# 温度分布图
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

for mech in ["Full", "Reduced", "Optimized"]:
    sub2 = df[(df["mechanism"] == mech) &
              (df["config"] == "pure_NH3_350K")].sort_values("distance_cm")
    if len(sub2) > 0 and "T_K" in sub2.columns:
        ax.plot(sub2["distance_cm"], sub2["T_K"],
                color=colors[mech], linestyle=linestyles[mech],
                linewidth=1.5, label=mech)

if "T_K" in FLAME_PURE_NH3:
    ax.scatter(FLAME_PURE_NH3["distance_cm"], FLAME_PURE_NH3["T_K"],
               marker='o', facecolors='none', edgecolors='green',
               s=40, linewidths=1.5, zorder=5, label="Exp")

ax.set_xlabel("Distance (cm)")
ax.set_ylabel("Temperature (K)")
ax.set_title("Pure NH3 flame temperature profile, phi=1.0")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(RESULTS / "fig_flame_temperature.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("Flame temperature figure saved.")

print("All flame structure figures saved.")