"""
层流火焰速度 (LFS) 验证仿真 - 复现论文 Fig. 11-15, S3-S5

机理: Full(2021recon) / M1(Reduced) / M2(Optimized) —— 同 01_idt.py
模型: Cantera FreeFlame, mixture-averaged 输运 (论文口径)

图件与工况:
  Fig. 11 (2x3): LFS vs phi, Tu=298K, P=1atm, XH2=0/0.2/0.4/0.6/0.8/1.0
  Fig. 12 (1x2): (a) XH2=0.4, (b) XH2=0.6 (Tu=298K, P=1atm)
  Fig. 13 (1x3): LFS vs XH2, phi=1, Tu=298K, P=1/3/5 atm
  Fig. 14 (1x1): LFS vs phi, XH2=0.4, Tu=298K, P=1/3/5 atm
  Fig. 15 (1x3): LFS vs phi, XH2=0.05/0.3/0.6, Tu=298/373/423/473K, P=1atm
  Fig. S3 (1x2): 高温高压: (a) XH2=0.2, P=40bar, Tu=820K; (b) XH2=0, P=40bar, Tu=720K
  Fig. S4 (1x1): LFS vs P(0.1-0.5 MPa), XH2=0/0.1/0.2, Tu=360K, phi=1
  Fig. S5 (1x2): 部分解离氨 NH3/H2/N2/air, alpha=0.3/0.5, Tu=303K, P=1atm
输出: results/lfs_sweep.csv
"""
import sys
import numpy as np
import pandas as pd
import cantera as ct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_utils import RESULTS_DIR, phi_composition, dissociated_nh3_composition

MECHS = {
    "Full": "Full mechanism 2021recon.yaml",
    "Mechanism 1": "Reduced mechanism.yaml",
    "Mechanism 2": "Optimized mechanism.yaml",
}


def compute_lfs(gas, Tu, P_atm, comp, tight=False, width=0.03):
    gas.TPX = Tu, P_atm * ct.one_atm, comp
    f = ct.FreeFlame(gas, width=width)
    f.transport_model = "mixture-averaged"
    if tight:
        f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
    else:
        f.set_refine_criteria(ratio=3, slope=0.1, curve=0.2)
    try:
        f.solve(loglevel=0, auto=True)
        return f.velocity[0]
    except Exception:
        return np.nan


def sweep():
    rows = []
    mech_dir = Path(__file__).resolve().parent.parent / "mechanisms"

    def job(fig, label, XH2, Tu, P, phis, mech_name, gas, tight=False,
            width=0.03, alpha_diss=None):
        for phi in phis:
            comp = (dissociated_nh3_composition(alpha_diss, phi)
                    if alpha_diss is not None
                    else phi_composition(XH2, phi))
            Su = compute_lfs(gas, Tu, P, comp, tight=tight, width=width)
            rows.append({"figure": fig, "label": label,
                         "mechanism": mech_name,
                         "XH2_fuel": (alpha_diss if alpha_diss is not None
                                      else XH2),
                         "Tu_K": Tu, "P_atm": P, "phi": round(float(phi), 3),
                         "Su_m_s": Su,
                         "Su_cm_s": Su * 100.0 if np.isfinite(Su) else np.nan})
            print(f"  {fig} {label} {mech_name} Tu={Tu} P={P} phi={phi:.2f} "
                  f"Su={Su*100 if np.isfinite(Su) else float('nan'):.2f} cm/s",
                  flush=True)

    phis_full = np.arange(0.7, 1.45, 0.1)
    phis_narrow = np.arange(0.8, 1.45, 0.1)

    for mech_name, mech_file in MECHS.items():
        gas = ct.Solution(str(mech_dir / mech_file))
        print(f"=== {mech_name} ===", flush=True)

        # Fig. 11: XH2 = 0/0.2/0.4/0.6/0.8/1.0, 298K, 1atm, phi 0.7-1.4
        for XH2 in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            job("Fig11", f"XH2={XH2}", XH2, 298.0, 1.0, phis_full,
                mech_name, gas)

        # Fig. 12: XH2=0.4 / 0.6 (298K, 1atm)
        for XH2 in (0.4, 0.6):
            job("Fig12", f"XH2={XH2}", XH2, 298.0, 1.0, phis_full,
                mech_name, gas)

        # Fig. 13: vs XH2 (0-1, step 0.1), phi=1, 298K, P=1/3/5
        for P in (1.0, 3.0, 5.0):
            for XH2 in np.arange(0.0, 1.01, 0.1):
                job("Fig13", f"P={P}atm", round(float(XH2), 2), 298.0, P,
                    [1.0], mech_name, gas)

        # Fig. 14: vs phi, XH2=0.4, 298K, P=1/3/5
        for P in (1.0, 3.0, 5.0):
            job("Fig14", f"P={P}atm", 0.4, 298.0, P, phis_full, mech_name, gas)

        # Fig. 15: XH2=0.05/0.3/0.6 x Tu=298/373/423/473, P=1atm, phi 0.8-1.4
        for XH2 in (0.05, 0.3, 0.6):
            for Tu in (298.0, 373.0, 423.0, 473.0):
                job("Fig15", f"XH2={XH2},Tu={Tu}K", XH2, Tu, 1.0,
                    phis_narrow, mech_name, gas)

        # Fig. S3: 高温高压 (仅 Full 与 M2, 论文口径; 40 bar 收敛难, 单独参数)
        if mech_name in ("Full", "Mechanism 2"):
            job("FigS3a", "80%NH3/20%H2", 0.2, 820.0, 40.0,
                np.arange(0.8, 1.65, 0.1), mech_name, gas, tight=True,
                width=0.01)
            job("FigS3b", "100%NH3", 0.0, 720.0, 40.0,
                np.arange(0.6, 1.65, 0.2), mech_name, gas, tight=True,
                width=0.01)

        # Fig. S4: vs P(1-5 atm), XH2=0/0.1/0.2, Tu=360K, phi=1
        for XH2 in (0.0, 0.1, 0.2):
            for P in (1.0, 2.0, 3.0, 4.0, 5.0):
                job("FigS4", f"XH2={XH2}", XH2, 360.0, P, [1.0],
                    mech_name, gas)

        # Fig. S5: 部分解离氨 alpha=0.3/0.5, Tu=303K, P=1atm, phi 0.7-1.8
        for alpha in (0.3, 0.5):
            job("FigS5", f"alpha={alpha}", None, 303.0, 1.0,
                np.arange(0.6, 1.85, 0.2), mech_name, gas,
                alpha_diss=alpha)

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "lfs_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nLFS 计算完成: {len(df)} 行 -> {out}")


if __name__ == "__main__":
    sweep()
