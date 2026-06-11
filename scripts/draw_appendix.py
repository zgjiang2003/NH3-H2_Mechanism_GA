"""
Appendix 补充图绘图脚本 - 复现论文 Fig. S1-S6
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

colors = {"Full": "black", "Reduced": "blue", "Optimized": "red"}
linestyles = {"Full": "--", "Reduced": "-.", "Optimized": "-"}

# ============================================================
# Fig. S1: 纯NH3高温IDT, P=1.4 and 11 atm, NH3/O2/Ar
# ============================================================
print("Creating Fig. S1...")
df_idt = pd.read_csv(RESULTS / "idt_sweep.csv")
df_idt = df_idt[df_idt["IDT_ms"] < 5000].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, P in enumerate([1.4, 11.0]):
    ax = axes[idx]
    for mech in ["Full", "Reduced", "Optimized"]:
        sub = df_idt[(df_idt["mechanism"] == mech) &
                     (df_idt["XH2_fuel"] == 0.0) &
                     (df_idt["P_atm"] == P) &
                     (df_idt["diluent"] == "AR") &
                     (df_idt["phi"] == 1.0)]
        sub = sub.sort_values("1000/T")
        if len(sub) > 0:
            ax.semilogy(sub["1000/T"], sub["IDT_ms"],
                        color=colors[mech], linestyle=linestyles[mech],
                        linewidth=1.5, label=mech)
    ax.set_xlabel("1000/T (1/K)")
    ax.set_ylabel("Ignition delay time (ms)")
    ax.set_title(f"NH3/O2/Ar, P={P} atm (Fig. S1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(RESULTS / "fig_appendix_S1.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# ============================================================
# Fig. S2: 低温稀混合气IDT
# (a) phi=0.35, P=43.4 bar
# (b) phi=1, P=20 bar
# ============================================================
print("Creating Fig. S2...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Fig. S2(a)
ax = axes[0]
for XH2 in [0.0, 0.1, 0.25]:
    for mech in ["Full", "Reduced", "Optimized"]:
        sub = df_idt[(df_idt["mechanism"] == mech) &
                     (df_idt["XH2_fuel"] == XH2) &
                     (df_idt["P_atm"] == 43.4) &
                     (df_idt["phi"] == 0.35)]
        sub = sub.sort_values("1000/T")
        if len(sub) > 0:
            ax.semilogy(sub["1000/T"], sub["IDT_ms"],
                        color=colors[mech], linestyle=linestyles[mech],
                        linewidth=1.5, alpha=0.8,
                        label=f"{mech}, XH2={XH2}" if XH2 == 0.0 else None)
    if XH2 == 0.0:
        ax.semilogy([], [], color='gray', linestyle='-', label=f"XH2={XH2}")
    elif XH2 == 0.1:
        ax.semilogy([], [], color='gray', linestyle='--', label=f"XH2={XH2}")
    elif XH2 == 0.25:
        ax.semilogy([], [], color='gray', linestyle=':', label=f"XH2={XH2}")

ax.set_xlabel("1000/T (1/K)")
ax.set_ylabel("Ignition delay time (ms)")
ax.set_title("phi=0.35, P=43.4 bar (Fig. S2a)")
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3)

# Fig. S2(b)
ax = axes[1]
for XH2 in [0.0, 0.1, 0.25]:
    for mech in ["Full", "Reduced", "Optimized"]:
        sub = df_idt[(df_idt["mechanism"] == mech) &
                     (df_idt["XH2_fuel"] == XH2) &
                     (df_idt["P_atm"] == 20.0) &
                     (df_idt["phi"] == 1.0)]
        sub = sub.sort_values("1000/T")
        if len(sub) > 0:
            ax.semilogy(sub["1000/T"], sub["IDT_ms"],
                        color=colors[mech], linestyle=linestyles[mech],
                        linewidth=1.5, alpha=0.8)

ax.set_xlabel("1000/T (1/K)")
ax.set_ylabel("Ignition delay time (ms)")
ax.set_title("phi=1.0, P=20 bar (Fig. S2b)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(RESULTS / "fig_appendix_S2.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# ============================================================
# Fig. S3: 高温高压LFS
# (a) 80%NH3/20%H2, P=40 bar, T=820K
# (b) 100%NH3, P=40 bar, T=720K
# ============================================================
print("Creating Fig. S3...")
df_lfs = pd.read_csv(RESULTS / "lfs_sweep.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Fig. S3(a)
ax = axes[0]
for mech in ["Full", "Reduced", "Optimized"]:
    sub = df_lfs[(df_lfs["mechanism"] == mech) &
                 (df_lfs["XH2_fuel"] == 0.2) &
                 (df_lfs["P_atm"] == 40.0) &
                 (df_lfs["T_unburned_K"] == 820.0)]
    sub = sub.sort_values("phi")
    if len(sub) > 0:
        ax.plot(sub["phi"], sub["Su_m_s"] * 100,
                color=colors[mech], linestyle=linestyles[mech],
                linewidth=1.5, label=mech)
ax.set_xlabel("Equivalence ratio")
ax.set_ylabel("Laminar flame speed (cm/s)")
ax.set_title("80%NH3/20%H2, P=40 bar, Tu=820K (Fig. S3a)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Fig. S3(b)
ax = axes[1]
for mech in ["Full", "Reduced", "Optimized"]:
    sub = df_lfs[(df_lfs["mechanism"] == mech) &
                 (df_lfs["XH2_fuel"] == 0.0) &
                 (df_lfs["P_atm"] == 40.0) &
                 (df_lfs["T_unburned_K"] == 720.0)]
    sub = sub.sort_values("phi")
    if len(sub) > 0:
        ax.plot(sub["phi"], sub["Su_m_s"] * 100,
                color=colors[mech], linestyle=linestyles[mech],
                linewidth=1.5, label=mech)
ax.set_xlabel("Equivalence ratio")
ax.set_ylabel("Laminar flame speed (cm/s)")
ax.set_title("100%NH3, P=40 bar, Tu=720K (Fig. S3b)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(RESULTS / "fig_appendix_S3.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# ============================================================
# Fig. S4: 360K下LFS随压力变化
# ============================================================
print("Creating Fig. S4...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, XH2 in enumerate([0.0, 0.1, 0.2]):
    ax = axes[idx]
    for P in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
        for mech in ["Full", "Reduced", "Optimized"]:
            sub = df_lfs[(df_lfs["mechanism"] == mech) &
                         (df_lfs["XH2_fuel"] == XH2) &
                         (df_lfs["P_atm"] == P) &
                         (df_lfs["T_unburned_K"] == 360.0)]
            sub = sub.sort_values("phi")
            if len(sub) > 0:
                ax.plot(sub["phi"], sub["Su_m_s"] * 100,
                        color=colors[mech], linestyle=linestyles[mech],
                        linewidth=1.0, alpha=0.7,
                        label=mech if P == 1.0 else None)
    ax.set_xlabel("Equivalence ratio")
    ax.set_ylabel("Laminar flame speed (cm/s)")
    ax.set_title(f"XH2={XH2}, Tu=360K (Fig. S4)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(RESULTS / "fig_appendix_S4.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# ============================================================
# Fig. S5: NH3/H2/N2/air 混合物 LFS
# alpha=0.3 and 0.5, Tu=303K, P=1 atm
# ============================================================
print("Creating Fig. S5...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, alpha in enumerate([0.3, 0.5]):
    ax = axes[idx]
    for mech in ["Full", "Reduced", "Optimized"]:
        sub = df_lfs[(df_lfs["mechanism"] == mech) &
                     (df_lfs["XH2_fuel"] == alpha) &
                     (df_lfs["P_atm"] == 1.0) &
                     (df_lfs["T_unburned_K"] == 303.0)]
        sub = sub.sort_values("phi")
        if len(sub) > 0:
            ax.plot(sub["phi"], sub["Su_m_s"] * 100,
                    color=colors[mech], linestyle=linestyles[mech],
                    linewidth=1.5, label=mech)
    ax.set_xlabel("Equivalence ratio")
    ax.set_ylabel("Laminar flame speed (cm/s)")
    ax.set_title(f"alpha={alpha}, Tu=303K, P=1atm (Fig. S5)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(RESULTS / "fig_appendix_S5.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# ============================================================
# Fig. S6: NH3/air 预混火焰中 NO 摩尔分数
# phi=0.9, 1.0, 1.2
# ============================================================
print("Creating Fig. S6...")
df_flame = pd.read_csv(RESULTS / "premixed_flame.csv")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, phi_val in enumerate([0.9, 1.0, 1.2]):
    ax = axes[idx]
    config = f"pure_NH3_phi{phi_val}"
    for mech in ["Full", "Reduced", "Optimized"]:
        sub = df_flame[(df_flame["mechanism"] == mech) &
                       (df_flame["config"] == config)].sort_values("distance_cm")
        if len(sub) > 0 and "X_NO" in sub.columns:
            ax.plot(sub["distance_cm"], sub["X_NO"],
                    color=colors[mech], linestyle=linestyles[mech],
                    linewidth=1.5, label=mech)
    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel("NO mole fraction")
    ax.set_title(f"NH3/air, phi={phi_val} (Fig. S6)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(RESULTS / "fig_appendix_S6.png", dpi=300, bbox_inches='tight')
plt.close(fig)

print("All Appendix figures saved.")
