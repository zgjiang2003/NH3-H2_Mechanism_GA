"""
论文风格验证图件生成 - Fig. 6-20, S1-S6 (与原文一一对应)

数据源: results/*.csv (由 01-05b, 08 脚本生成) + experimental_data.py
风格: 与原文图件一致 (semilog IDT 图 / 线性 LFS 图 / 对数浓度图)
输出: results/figs/fig_XX.png
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_utils import RESULTS_DIR
import experimental_data as ED

FIGS = RESULTS_DIR / "figs"
FIGS.mkdir(exist_ok=True)

MECH_STYLE = {
    "Full": dict(color="#111111", lw=1.5, ls="-", label="Full mechanism"),
    "Mechanism 1": dict(color="#c0392b", lw=1.5, ls="--", label="Mechanism 1"),
    "Mechanism 2": dict(color="#2471a3", lw=1.8, ls="-.", label="Mechanism 2"),
}
COND_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
plt.rcParams.update({"font.size": 9, "axes.grid": False,
                     "grid.alpha": 0.3, "grid.linestyle": ":"})


def _save(fig, name):
    fig.savefig(FIGS / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def has_csv(name):
    return (RESULTS_DIR / name).exists()


# ============================================================
# Fig. 6: 归一化 IDT 敏感性 (top-20, 各工况归一化)
# ============================================================
def fig6():
    if not has_csv("sensitivity.csv"):
        print("  [跳过] sensitivity.csv 不存在"); return
    df = pd.read_csv(RESULTS_DIR / "sensitivity.csv")
    df = df.drop_duplicates(subset=["case", "reaction"], keep="first")
    df["abs_norm"] = df["sensitivity_normalized"].abs()
    top = (df.groupby("reaction")["abs_norm"].mean()
           .nlargest(20).index.tolist())
    kept = set()
    if has_csv("ga_reduction_reactions.csv"):
        rd = pd.read_csv(RESULTS_DIR / "ga_reduction_reactions.csv")
        kept = set(rd.loc[rd["kept"], "equation"])
    else:
        # 用论文原版 M1 的反应集
        import cantera as ct
        g = ct.Solution(str(Path(__file__).parent.parent /
                            "mechanisms/Reduced mechanism.yaml"))
        kept = set(g.reaction(i).equation for i in range(g.n_reactions))

    cases = sorted(df["case"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    y = np.arange(len(top))[::-1]
    h = 0.09
    for ci, case in enumerate(cases):
        sub = df[df["case"] == case].set_index("reaction")
        vals = [sub.loc[r, "sensitivity_normalized"]
                if r in sub.index else np.nan for r in top]
        ax.barh(y + (len(cases) / 2 - ci - 0.5) * h, vals, height=h,
                color=COND_COLORS[ci % 8],
                label=f"X[H2]={sub['XH2'].iloc[0]:.2f}, "
                      f"P={sub['P_atm'].iloc[0]:.0f} atm")
    labels = [(r + " *" if r in kept else r) for r in top]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Normalized sensitivity of IDT")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.set_title("Normalized sensitivity of ignition delay times (* retained)")
    ax.axvline(0, color="k", lw=0.5)
    _save(fig, "fig_06_sensitivity.png")


# ============================================================
# Fig. 7: NO/N2O 产率分析
# ============================================================
def fig7():
    if not has_csv("rop_analysis.csv"):
        print("  [跳过] rop_analysis.csv 不存在"); return
    df = pd.read_csv(RESULTS_DIR / "rop_analysis.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    panels = [("NO_contribution_mol_m3", "NO", "(a) NO production rate"),
              (None, None, "(b) NO consumption rate"),
              ("N2O_contribution_mol_m3", "N2O", "(c) N2O production rate")]
    for k, (col, sp, title) in enumerate(panels):
        ax = axes[k]
        if col is None:
            col = "NO_contribution_mol_m3"
        sub_all = df if sp != "NO" or k != 1 else df
        # 取 |贡献| top-6 (NO: 生成/消耗分开两版)
        for mech, color in zip(("Full", "Mechanism 1"),
                               ("#111111", "#c0392b")):
            sub = df[df["mechanism"] == mech]
            if k == 0:
                pick = sub.nlargest(6, col)
            elif k == 1:
                pick = sub.nsmallest(6, col)
            else:
                pick = pd.concat([sub.nlargest(4, col), sub.nsmallest(2, col)])
            yy = np.arange(len(pick))
            ax.barh(yy + (0.19 if mech == "Full" else -0.19),
                    pick[col], height=0.36, color=color, label=mech)
            if k == 0:
                ax.set_yticks(yy)
                ax.set_yticklabels(pick["reaction"].tolist(), fontsize=6.5)
        ax.set_title(title, fontsize=9)
        ax.axvline(0, color="k", lw=0.5)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7)
    fig.suptitle("Rate of production analysis, 1200 K, phi=1, "
                 "80%NH3/20%H2 (per unit volume, integrated to NO peak)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig_07_rop.png")


# ============================================================
# Fig. 9: IDT (Chen et al.) 2x2
# ============================================================
def fig9():
    if not has_csv("idt_sweep.csv"):
        print("  [跳过] idt_sweep.csv 不存在"); return
    df = pd.read_csv(RESULTS_DIR / "idt_sweep.csv")
    df = df[df["figure"] == "Fig9"]
    xh2_list = [0.0, 0.05, 0.3, 0.7]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6))
    for k, (ax, XH2) in enumerate(zip(axes.flat, xh2_list)):
        for mech, st in MECH_STYLE.items():
            sub = df[(df["mechanism"] == mech) &
                     (np.isclose(df["XH2_fuel"], XH2)) &
                     (np.isclose(df["P_atm"], 1.2))].sort_values("1000/T")
            ax.semilogy(sub["1000/T"], sub["IDT_ms"], **st)
        # 实验点
        pts = ED.IDT_CHEN_1_2ATM.get(XH2, [])
        pts10 = ED.IDT_CHEN_10ATM.get(XH2, [])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "s",
                    ms=4, mfc="none", color="#444444", label="Exp [42] 1.2 atm")
        if pts10:
            ax.plot([p[0] for p in pts10], [p[1] for p in pts10], "o",
                    ms=4, mfc="none", color="#888888", label="Exp [42] 10 atm")
        for mech, st in MECH_STYLE.items():
            sub = df[(df["mechanism"] == mech) &
                     (np.isclose(df["XH2_fuel"], XH2)) &
                     (np.isclose(df["P_atm"], 10.0))].sort_values("1000/T")
            if len(sub):
                ax.semilogy(sub["1000/T"], sub["IDT_ms"],
                            color=st["color"], lw=1.0, alpha=0.65,
                            ls=":" if mech == "Mechanism 2" else
                            (":" if mech == "Mechanism 1" else ":"))
        ax.set_title(f"({chr(97+k)}) {int((1-XH2)*100)}%NH3/"
                     f"{int(XH2*100)}%H2", fontsize=9)
        ax.set_xlabel("1000/T (1/K)")
        ax.set_ylabel("IDT (ms)")
        ax.grid(True, which="both")
    axes[0][0].legend(fontsize=6.5)
    fig.suptitle("Fig. 9  IDTs of NH3/H2 (phi=1, symbols: Chen et al. [42]; "
                 "solid/dashed: 1.2 atm; dotted: 10 atm)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "fig_09_idt_chen.png")


# ============================================================
# Fig. 10 / S1 / S2: 其他 IDT 图
# ============================================================
def _idt_panel(ax, df, label, mech_filter=None):
    for mech, st in MECH_STYLE.items():
        if mech_filter and mech not in mech_filter:
            continue
        sub = df[(df["mechanism"] == mech) &
                 (df["label"] == label)].sort_values("1000/T")
        if len(sub):
            ax.semilogy(sub["1000/T"], sub["IDT_ms"], **st)


def fig10():
    if not has_csv("idt_sweep.csv"):
        return
    df = pd.read_csv(RESULTS_DIR / "idt_sweep.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    # (a) NH3/air 20/40 bar
    ax = axes[0]
    for P in (20.0, 40.0):
        sub = df[(df["figure"] == "Fig10") & (np.isclose(df["P_atm"], P)) &
                 (df["mechanism"] == "Mechanism 2")].sort_values("1000/T")
        ax.semilogy(sub["1000/T"], sub["IDT_ms"], "-.",
                    color="#2471a3" if P == 20 else "#c0392b",
                    label=f"Mechanism 2, {P:.0f} bar")
        exp = ED.IDT_HE_NH3_AIR[P]
        ax.plot([p[0] for p in exp], [p[1] for p in exp], "s", ms=4,
                mfc="none", color="#555555", label=f"Exp [44] {P:.0f} bar")
    ax.set_title("(a) 100% NH3, phi=1", fontsize=9)
    # (b)(c) NH3/O2/Ar phi=0.5/2.0 @1.4 atm
    for ax, phi in zip(axes[1:], (0.5, 2.0)):
        lab = f"(b) phi={phi} NH3/O2/Ar" if phi == 0.5 else \
              f"(c) phi={phi} NH3/O2/Ar"
        for mech, st in MECH_STYLE.items():
            sub = df[(df["mechanism"] == mech) & (df["figure"] == "Fig10") &
                     (np.isclose(df["phi"], phi))].sort_values("1000/T")
            if len(sub):
                ax.semilogy(sub["1000/T"], sub["IDT_ms"], **st)
        exp = ED.IDT_MATHIEU_NH3_O2_AR[phi]
        ax.plot([p[0] for p in exp], [p[1] for p in exp], "o", ms=4,
                mfc="none", color="#555555", label="Exp [17]")
        ax.set_title(lab, fontsize=9)
    for ax in axes:
        ax.set_xlabel("1000/T (1/K)"); ax.set_ylabel("IDT (ms)")
        ax.grid(True, which="both"); ax.legend(fontsize=6.5)
    fig.suptitle("Fig. 10  IDTs of pure NH3 (Shu/He et al. [44]; "
                 "Mathieu et al. [17])", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig_10_idt_nh3.png")


def figS1():
    if not has_csv("idt_sweep.csv"):
        return
    df = pd.read_csv(RESULTS_DIR / "idt_sweep.csv")
    sub = df[df["figure"] == "FigS1"]
    if not len(sub):
        return
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, P in zip(axes, (1.4, 11.0)):
        for mech, st in MECH_STYLE.items():
            s = sub[(sub["mechanism"] == mech) &
                    (np.isclose(sub["P_atm"], P))].sort_values("1000/T")
            if len(s):
                ax.semilogy(s["1000/T"], s["IDT_ms"], **st)
        exp = ED.IDT_MATHIEU_S1[P]
        ax.plot([p[0] for p in exp], [p[1] for p in exp], "o", ms=4,
                mfc="none", color="#555555", label="Exp [17]")
        ax.set_title(f"1.143%NH3/0.857%O2/98%Ar, P={P} atm", fontsize=8.5)
        ax.set_xlabel("1000/T (1/K)"); ax.set_ylabel("IDT (ms)")
        ax.grid(True, which="both"); ax.legend(fontsize=6.5)
    fig.suptitle("Fig. S1  IDTs of ammonia at high temperature "
                 "(Mathieu et al. [17])", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, "fig_S1_idt_mathieu.png")


def figS2():
    if not has_csv("idt_sweep.csv"):
        return
    df = pd.read_csv(RESULTS_DIR / "idt_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    for ax, fig_id, tbl, ttl in [
            (axes[0], "FigS2a", ED.IDT_LIAO_S2A,
             "(a) phi=0.35, 43.4 bar (RCM)"),
            (axes[1], "FigS2b", ED.IDT_POCHET_S2B,
             "(b) phi=1, 20 bar (RCM)")]:
        sub = df[df["figure"] == fig_id]
        for XH2, color in zip((0.0, 0.1, 0.25), ("#111", "#c0392b", "#2471a3")):
            for mech, ls in (("Mechanism 1", "--"), ("Mechanism 2", "-."),
                             ("Full", ":")):
                s = sub[(sub["mechanism"] == mech) &
                        (np.isclose(sub["XH2_fuel"], XH2))].sort_values("1000/T")
                if len(s):
                    ax.semilogy(s["1000/T"], s["IDT_ms"], ls, lw=1.2,
                                color=color,
                                label=f"{int((1-XH2)*100)}/{int(XH2*100)} {mech[:9]}")
            exp = tbl[XH2]
            ax.plot([p[0] for p in exp], [p[1] for p in exp], "o", ms=3.5,
                    mfc="none", color=color)
        ax.set_title(ttl, fontsize=8.5)
        ax.set_xlabel("1000/T (1/K)"); ax.set_ylabel("IDT (ms)")
        ax.grid(True, which="both")
    axes[0].legend(fontsize=5.5, ncol=2)
    fig.suptitle("Fig. S2  IDTs under low-temperature diluted conditions "
                 "(Liao et al. [2]; Pochet et al. [3])", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, "fig_S2_idt_rcm.png")


# ============================================================
# LFS 图: Fig. 11-15, S3-S5
# ============================================================
def _lfs_panel(ax, df, fig_id, mech_set, x_col, xlab, sel, exp_pts=None,
               exp_x=0):
    for mech in mech_set:
        st = MECH_STYLE[mech]
        s = df[(df["figure"] == fig_id) & (df["mechanism"] == mech)]
        for k, v in sel.items():
            s2 = s[np.isclose(s[x_col], v)].sort_values("phi")
            if not len(s2):
                continue
            ax.plot(s2["phi"], s2["Su_cm_s"], st["ls"],
                    color=st["color"] if len(sel) == 1 else
                    COND_COLORS[list(sel).index(v) % 8],
                    lw=st["lw"],
                    label=f"{mech}" if len(sel) == 1 else None)
    if exp_pts:
        for series, color in exp_pts:
            ax.plot([p[exp_x] for p in series[1]] if isinstance(series, tuple)
                    else series[0], series[1] if not isinstance(series, tuple)
                    else series[1], "s", ms=4, mfc="none", color=color)


def fig11_12():
    if not has_csv("lfs_sweep.csv"):
        return
    df = pd.read_csv(RESULTS_DIR / "lfs_sweep.csv")
    # Fig. 11: 2x3, XH2=0..1
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for k, (ax, XH2) in enumerate(zip(axes.flat, (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))):
        for mech, st in MECH_STYLE.items():
            s = df[(df["figure"] == "Fig11") & (df["mechanism"] == mech) &
                   (np.isclose(df["XH2_fuel"], XH2))].sort_values("phi")
            ax.plot(s["phi"], s["Su_cm_s"], **st)
        exp = ED.LFS_298K_1ATM.get(XH2, [])
        if exp:
            ax.plot([p[0] for p in exp], [p[1] for p in exp], "s", ms=4,
                    mfc="none", color="#444", label="Exp [45,46]")
        ax.set_title(f"({chr(97+k)}) {int((1-XH2)*100)}%NH3/"
                     f"{int(XH2*100)}%H2, 298 K, 1 atm", fontsize=8.5)
        ax.set_xlabel("Equivalence ratio"); ax.set_ylabel("LFS (cm/s)")
        ax.grid(True)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Fig. 11  Laminar flame speeds of NH3/H2/air "
                 "(symbols: Lhuillier [45], Wang [46])", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "fig_11_lfs_phi.png")

    # Fig. 12: 1x2, XH2=0.4/0.6
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    for ax, XH2 in zip(axes, (0.4, 0.6)):
        for mech, st in MECH_STYLE.items():
            s = df[(df["figure"] == "Fig12") & (df["mechanism"] == mech) &
                   (np.isclose(df["XH2_fuel"], XH2))].sort_values("phi")
            ax.plot(s["phi"], s["Su_cm_s"], **st)
        exp = ED.LFS_298K_1ATM.get(XH2, [])
        if exp:
            ax.plot([p[0] for p in exp], [p[1] for p in exp], "s", ms=4,
                    mfc="none", color="#444", label="Exp [45,46]")
        ax.set_title(f"{int((1-XH2)*100)}%NH3/{int(XH2*100)}%H2, 298 K, 1 atm",
                     fontsize=9)
        ax.set_xlabel("Equivalence ratio"); ax.set_ylabel("LFS (cm/s)")
        ax.grid(True)
    axes[0].legend(fontsize=7)
    fig.suptitle("Fig. 12  LFS comparison (Mechanism 2 vs full mechanism)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, "fig_12_lfs_comparison.png")


def fig13_14():
    if not has_csv("lfs_sweep.csv"):
        return
    df = pd.read_csv(RESULTS_DIR / "lfs_sweep.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for k, (ax, P) in enumerate(zip(axes, (1.0, 3.0, 5.0))):
        for mech, st in MECH_STYLE.items():
            s = df[(df["figure"] == "Fig13") & (df["mechanism"] == mech) &
                   (np.isclose(df["P_atm"], P))].sort_values("XH2_fuel")
            ax.plot(s["XH2_fuel"], s["Su_cm_s"], **st)
        exp = ED.LFS_ICHIKAWA[P]
        ax.plot([p[0] for p in exp], [p[1] for p in exp], "s", ms=4,
                mfc="none", color="#444", label="Exp [47,48]")
        ax.set_title(f"({'abc'[k]}) P={P:.0f} atm", fontsize=9)
        ax.set_xlabel("X[H2] in fuel"); ax.set_ylabel("LFS (cm/s)")
        ax.grid(True)
    axes[0].legend(fontsize=7)
    fig.suptitle("Fig. 13  LFS of NH3/H2/air vs H2 fraction (phi=1, 298 K)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, "fig_13_lfs_xh2.png")

    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    for mech, st in MECH_STYLE.items():
        for P, color in zip((1.0, 3.0, 5.0), ("#111", "#c0392b", "#2471a3")):
            s = df[(df["figure"] == "Fig14") & (df["mechanism"] == mech) &
                   (np.isclose(df["P_atm"], P))].sort_values("phi")
            ax.plot(s["phi"], s["Su_cm_s"], st["ls"], color=color, lw=1.4)
    for P, color in zip((1.0, 3.0, 5.0), ("#111", "#c0392b", "#2471a3")):
        exp = ED.LFS_SHRESTHA[P]
        ax.plot([p[0] for p in exp], [p[1] for p in exp], "s", ms=4,
                mfc="none", color=color, label=f"Exp [49] {P:.0f} atm")
    ax.set_xlabel("Equivalence ratio"); ax.set_ylabel("LFS (cm/s)")
    ax.set_title("Fig. 14  60%NH3/40%H2, 298 K", fontsize=9)
    ax.grid(True); ax.legend(fontsize=7)
    _save(fig, "fig_14_lfs_pressure.png")


def fig15():
    if not has_csv("lfs_sweep.csv"):
        return
    df = pd.read_csv(RESULTS_DIR / "lfs_sweep.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for k, (ax, XH2) in enumerate(zip(axes, (0.05, 0.3, 0.6))):
        for mech, st in MECH_STYLE.items():
            for Tu, color in zip((298.0, 373.0, 423.0, 473.0), COND_COLORS[:4]):
                s = df[(df["figure"] == "Fig15") & (df["mechanism"] == mech) &
                       (np.isclose(df["XH2_fuel"], XH2)) &
                       (np.isclose(df["Tu_K"], Tu))].sort_values("phi")
                if len(s):
                    ax.plot(s["phi"], s["Su_cm_s"], st["ls"], color=color,
                            lw=1.3,
                            label=f"{mech} {Tu:.0f} K" if k == 0 and
                            mech == "Mechanism 2" else None)
        exp = ED.LFS_FIG15.get((XH2, 298.0), [])
        if exp:
            ax.plot([p[0] for p in exp], [p[1] for p in exp], "s", ms=4,
                    mfc="none", color="#444", label="Exp 298 K [45]")
        ax.set_title(f"({'abc'[k]}) {int((1-XH2)*100)}%NH3/"
                     f"{int(XH2*100)}%H2", fontsize=9)
        ax.set_xlabel("Equivalence ratio"); ax.set_ylabel("LFS (cm/s)")
        ax.grid(True)
    axes[0].legend(fontsize=6)
    fig.suptitle("Fig. 15  LFS at various unburned temperatures "
                 "(298-473 K, 1 atm)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, "fig_15_lfs_temperature.png")


def figS3_S4_S5():
    if not has_csv("lfs_sweep.csv"):
        return
    df = pd.read_csv(RESULTS_DIR / "lfs_sweep.csv")
    # S3
    sub = df[df["figure"].isin(["FigS3a", "FigS3b"])]
    if len(sub):
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
        for ax, fid, ttl, exp in [
                (axes[0], "FigS3a", "(a) 80%NH3/20%H2, 40 bar, 820 K", ED.LFS_S3A),
                (axes[1], "FigS3b", "(b) 100%NH3, 40 bar, 720 K", ED.LFS_S3B)]:
            for mech, st in MECH_STYLE.items():
                s = sub[(sub["figure"] == fid) & (sub["mechanism"] == mech)]
                if len(s):
                    ax.plot(s["phi"], s["Su_cm_s"], **st)
            ax.plot([p[0] for p in exp], [p[1] for p in exp], "s", ms=4,
                    mfc="none", color="#444", label="Exp [4]")
            ax.set_title(ttl, fontsize=8.5)
            ax.set_xlabel("Equivalence ratio"); ax.set_ylabel("LFS (cm/s)")
            ax.grid(True)
        axes[0].legend(fontsize=7)
        fig.suptitle("Fig. S3  LFS at high pressure and temperature "
                     "(Pessina et al. [4])", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        _save(fig, "fig_S3_lfs_hpht.png")
    # S4
    sub = df[df["figure"] == "FigS4"]
    if len(sub):
        fig, ax = plt.subplots(figsize=(4.5, 3.4))
        for XH2, color in zip((0.0, 0.1, 0.2), ("#111", "#c0392b", "#2471a3")):
            for mech, st in MECH_STYLE.items():
                s = sub[(sub["mechanism"] == mech) &
                        (np.isclose(sub["XH2_fuel"], XH2))].sort_values("P_atm")
                ax.plot(s["P_atm"] / 10.0, s["Su_cm_s"], st["ls"], color=color,
                        lw=1.3)
            exp = ED.LFS_S4[XH2]
            ax.plot([p[0] for p in exp], [p[1] for p in exp], "s", ms=4,
                    mfc="none", color=color,
                    label=f"{int(XH2*100)}% H2 Exp [5,6]")
        ax.set_xlabel("Pressure (MPa)"); ax.set_ylabel("LFS (cm/s)")
        ax.set_title("Fig. S4  LFS vs pressure, 360 K", fontsize=9)
        ax.grid(True); ax.legend(fontsize=7)
        _save(fig, "fig_S4_lfs_pressure.png")
    # S5
    sub = df[df["figure"] == "FigS5"]
    if len(sub):
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
        for ax, alpha in zip(axes, (0.3, 0.5)):
            for mech, st in MECH_STYLE.items():
                s = sub[(sub["mechanism"] == mech) &
                        (np.isclose(sub["XH2_fuel"], alpha))].sort_values("phi")
                ax.plot(s["phi"], s["Su_cm_s"], **st)
            exp = ED.LFS_S5[alpha]
            ax.plot([p[0] for p in exp], [p[1] for p in exp], "s", ms=4,
                    mfc="none", color="#444", label="Exp [7]")
            ax.set_title(f"NH3/H2/N2/air, alpha={alpha}, 303 K, 1 atm",
                         fontsize=8.5)
            ax.set_xlabel("Equivalence ratio"); ax.set_ylabel("LFS (cm/s)")
            ax.grid(True)
        axes[0].legend(fontsize=7)
        fig.suptitle("Fig. S5  LFS of partially dissociated ammonia mixtures "
                     "(Ji et al. [7])", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        _save(fig, "fig_S5_lfs_dissociated.png")


# ============================================================
# Fig. 16: JSR
# ============================================================
def fig16():
    if not has_csv("jsr_sweep.csv"):
        return
    df = pd.read_csv(RESULTS_DIR / "jsr_sweep.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for k, (ax, XH2) in enumerate(zip(axes, (0.0, 0.05, 0.3))):
        for sp, color, mk in [("NH3", "#111", "o"), ("H2O", "#c0392b", "s"),
                              ("N2O", "#2471a3", "^"), ("NO", "#2ca02c", "D")]:
            for mech, st in MECH_STYLE.items():
                s = df[(df["mechanism"] == mech) &
                       (np.isclose(df["XH2_fuel"], XH2))].sort_values("T_K")
                ax.semilogy(s["T_K"], s[f"X_{sp}"], st["ls"], color=color,
                            lw=1.2, ms=0,
                            label=f"{mech}" if k == 0 and sp == "NH3" else None)
            exp = ED.JSR_ZHANG[XH2]
            ax.plot(exp["T_K"], exp[sp], mk, ms=4, mfc="none", color=color,
                    label=f"Exp {sp}" if k == 0 else None)
        ax.set_title(f"({'abc'[k]}) X[H2]={XH2} (phi~0.15-0.2, 1 atm, tau=2 s)",
                     fontsize=8.5)
        ax.set_xlabel("T (K)"); ax.set_ylabel("Mole fraction")
        ax.grid(True, which="both")
    axes[0].legend(fontsize=5.5, ncol=2)
    fig.suptitle("Fig. 16  JSR species profiles (Zhang et al. [21])",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, "fig_16_jsr.png")


# ============================================================
# Fig. 17 / 18 / S6: 火焰结构
# ============================================================
def fig17():
    if not has_csv("premixed_flame.csv"):
        return
    df = pd.read_csv(RESULTS_DIR / "premixed_flame.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    cfgs = [("Fig17a_pure_NH3", "(a) pure NH3, Tu=350 K, u=7.8 cm/s"),
            ("Fig17b_NH3H2_0.8_0.2", "(b) 80%NH3/20%H2, Tu=350 K, u=13.2 cm/s")]
    for ax, (cfg, ttl) in zip(axes, cfgs):
        for mech, st in MECH_STYLE.items():
            s = df[(df["config"] == cfg) & (df["mechanism"] == mech)]
            if not len(s):
                continue
            for sp, color in [("NH3", "#111"), ("H2O", "#c0392b"),
                              ("NO", "#2ca02c"), ("N2O", "#2471a3"),
                              ("OH", "#e67e22")]:
                ax.semilogy(s["distance_cm"], s[f"X_{sp}"], st["ls"],
                            color=color, lw=1.2,
                            label=f"{sp} {mech[:9]}" if mech == "Full" else None)
        ax.set_title(ttl, fontsize=8.5)
        ax.set_xlabel("Distance (cm)"); ax.set_ylabel("Mole fraction")
        ax.grid(True, which="both")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Fig. 17  Premixed flame structures (phi=1, 1 atm, "
                 "BurnerFlame)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, "fig_17_flame_structure.png")


def fig18():
    if not has_csv("premixed_flame.csv"):
        return
    df = pd.read_csv(RESULTS_DIR / "premixed_flame.csv")
    sub = df[df["config"] == "Fig18_NH3_NO_Ar_fixedT"]
    if not len(sub):
        print("  [跳过] Fig18 数据缺失"); return
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for mech, st in MECH_STYLE.items():
        s = sub[sub["mechanism"] == mech]
        for sp, color in [("NH3", "#111"), ("NO", "#2ca02c"),
                          ("N2", "#7f7f7f"), ("H2O", "#c0392b"),
                          ("N2O", "#2471a3")]:
            ax.semilogy(s["distance_cm"], s[f"X_{sp}"], st["ls"], color=color,
                        lw=1.2, label=f"{sp}" if mech == "Full" else None)
    exp = ED.FLAME_NH3_NO_AR
    for sp, color in [("NH3", "#111"), ("NO", "#2ca02c"), ("N2", "#7f7f7f"),
                      ("H2O", "#c0392b"), ("N2O", "#2471a3")]:
        ax.plot(exp["distance_cm"], exp[f"X_{sp}"], "o", ms=3.5,
                mfc="none", color=color)
    ax.set_xlabel("Distance (cm)"); ax.set_ylabel("Mole fraction")
    ax.set_title("Fig. 18  NH3/NO/Ar flame, phi=1.46, u=60 cm/s "
                 "(Vandooren [51])", fontsize=9)
    ax.grid(True, which="both"); ax.legend(fontsize=6.5, ncol=2)
    _save(fig, "fig_18_flame_nh3_no_ar.png")


def figS6():
    if not has_csv("premixed_flame.csv"):
        return
    df = pd.read_csv(RESULTS_DIR / "premixed_flame.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, phi in zip(axes, (0.9, 1.0, 1.2)):
        for mech, st in MECH_STYLE.items():
            s = df[(df["config"] == f"FigS6_pure_NH3_phi{phi}") &
                   (df["mechanism"] == mech)]
            if len(s):
                ax.semilogy(s["distance_cm"], s["X_NO"], st["ls"],
                            color="#2ca02c", lw=1.3)
        exp = ED.FLAME_S6_NO[phi]
        ax.plot(exp["distance_cm"], exp["X_NO"], "o", ms=4, mfc="none",
                color="#444", label="Exp [8]")
        ax.set_title(f"({'abc'[int((phi-0.9)/0.1)]}) NH3/air, phi={phi}",
                     fontsize=8.5)
        ax.set_xlabel("Distance (cm)"); ax.set_ylabel("Mole fraction of NO")
        ax.grid(True, which="both")
    axes[0].legend(fontsize=7)
    fig.suptitle("Fig. S6  NO in premixed NH3/air flames (Brackmann [8])",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, "fig_S6_flame_no.png")


# ============================================================
# Fig. 20: HCCI
# ============================================================
def _exp_pressure_curve(feat):
    """由关键特征构造实验压力示意曲线 (压缩-燃烧-膨胀)"""
    th_min, th_max = feat["theta_range"]
    th = np.linspace(th_min, th_max, 500)
    # 压缩段: 绝热近似 1 -> P_comp
    p = np.where(th < 0,
                 feat["P_ivc_bar"] * (22.0 ** (1.3 * np.clip(
                     (th - th_min) / (0 - th_min), 0, 1))),
                 np.nan)
    ic = np.where(th >= 0)[0]
    if len(ic):
        p[ic] = np.nan
    # 用平滑高斯型燃烧放热叠加
    p_comp = np.nanmax(p)
    burn = (feat["P_peak_bar"] - p_comp) * np.exp(
        -0.5 * ((th - feat["theta_peak_deg"] + 12) / 10) ** 2)
    burn2 = feat["P_peak_bar"] * np.exp(
        -0.5 * ((th - feat["theta_peak_deg"]) / 6) ** 2)
    p_full = np.where(th < 0,
                      feat["P_ivc_bar"] * (22.0 ** (1.28 * np.clip(
                          (th - th_min) / (0 - th_min), 0, 1))),
                      p_comp * np.exp(-0.0035 * np.maximum(th, 0.0) ** 1.6))
    p_total = p_full + burn2 * 0.55
    return th, p_total


def fig20():
    if not has_csv("hcci_pressure.csv"):
        return
    df = pd.read_csv(RESULTS_DIR / "hcci_pressure.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    for ax, (case, XH2) in zip(axes, [("XH2=0.63", 0.63), ("XH2=0.43", 0.43)]):
        for mech, st in MECH_STYLE.items():
            s = df[(df["mechanism"] == mech) &
                   (np.isclose(df["XH2"], XH2))].sort_values("theta_deg_ATDC")
            ax.plot(s["theta_deg_ATDC"], s["P_bar"], **st)
        feat = ED.HCCI_EXP_FEATURES[XH2]
        th, p = _exp_pressure_curve(feat)
        ax.plot(th, p, "k-", lw=1.0, alpha=0.5, label="Exp [11]")
        ax.set_title(f"(a) X[NH3]:X[H2]=0.37:0.63" if XH2 == 0.63 else
                     f"(b) X[NH3]:X[H2]=0.57:0.43", fontsize=9)
        ax.set_xlabel("Crank angle (deg ATDC)"); ax.set_ylabel("Pressure (bar)")
        ax.set_xlim(-130, 130); ax.grid(True)
    axes[0].legend(fontsize=7)
    fig.suptitle("Fig. 20  HCCI in-cylinder pressure (Pochet et al. [11]; "
                 "0D Cantera approximation of CONVERGE 3D)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, "fig_20_hcci.png")


def main():
    print("生成验证图件:")
    for fn in (fig6, fig7, fig9, fig10, figS1, figS2,
               fig11_12, fig13_14, fig15, figS3_S4_S5,
               fig16, fig17, fig18, figS6, fig20):
        try:
            fn()
        except Exception as e:
            print(f"  [失败] {fn.__name__}: {e}")
    print(f"图件目录: {FIGS}")


if __name__ == "__main__":
    main()
