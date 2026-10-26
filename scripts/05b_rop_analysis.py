"""
产率分析 (ROP) - 复现论文 Fig. 7
NO 与 N2O 的生成/消耗路径分析: 1200 K, phi=1, 80%NH3/20%H2, 1 atm
均相反应器中积分各反应对 NO / N2O 净生成贡献 (至 NO 峰值时刻),
对比 Full (2021recon) 与 Mechanism 1。

输出: results/rop_analysis.csv
"""
import sys
import numpy as np
import pandas as pd
import cantera as ct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_utils import RESULTS_DIR, phi_composition

MECHS = {
    "Full": "Full mechanism 2021recon.yaml",
    "Mechanism 1": "Reduced mechanism.yaml",
}
T0, P, XH2, PHI = 1200.0, 1.0, 0.2, 1.0


def rop_analysis():
    mech_dir = Path(__file__).resolve().parent.parent / "mechanisms"
    rows = []
    for mech_name, mech_file in MECHS.items():
        gas = ct.Solution(str(mech_dir / mech_file))
        comp = phi_composition(XH2, PHI)
        gas.TPX = T0, P * ct.one_atm, comp
        reactor = ct.IdealGasReactor(gas)
        net = ct.ReactorNet([reactor])
        no_idx = gas.species_index("NO")
        n2o_idx = gas.species_index("N2O")

        # 逐步推进, 记录 NO 净生成速率峰值时刻的瞬时 ROP (论文图 7 口径)
        rop_at_peak = None
        stoich_snap = None
        no_max, dno_max = 0.0, -1.0
        x_prev, t_prev = 0.0, 0.0
        while net.time < 2.0:
            t = net.step()
            state = reactor.thermo
            gas.TPY = state.T, state.P, state.Y
            x_no = state.X[no_idx]
            dt = t - t_prev
            dno = (x_no - x_prev) / dt if dt > 0 else 0.0
            x_prev, t_prev = x_no, t
            no_max = max(no_max, x_no)
            if dno > dno_max and x_no > 0.02 * no_max:
                dno_max = dno
                rop_at_peak = gas.net_rates_of_progress.copy()
                stoich_snap = (
                    gas.product_stoich_coeffs[no_idx, :] -
                    gas.reactant_stoich_coeffs[no_idx, :],
                    gas.product_stoich_coeffs[n2o_idx, :] -
                    gas.reactant_stoich_coeffs[n2o_idx, :])
            if reactor.T > T0 + 800.0:
                break

        contrib_no = stoich_snap[0] * rop_at_peak
        contrib_n2o = stoich_snap[1] * rop_at_peak

        for i in range(gas.n_reactions):
            rows.append({
                "mechanism": mech_name,
                "reaction": gas.reaction(i).equation,
                "NO_contribution_mol_m3": contrib_no[i],
                "N2O_contribution_mol_m3": contrib_n2o[i],
            })
        print(f"{mech_name}: NO 峰值 {no_max*1e6:.1f} ppm, "
              f"ROP 快照于 NO 生成速率峰值 (dX_NO/dt = {dno_max:.2e} 1/s)")

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "rop_analysis.csv"
    df.to_csv(out, index=False)

    # 打印 top 贡献
    for sp, col in [("NO", "NO_contribution_mol_m3"),
                    ("N2O", "N2O_contribution_mol_m3")]:
        print(f"\n{sp} 主要生成/消耗反应 (正=生成, 负=消耗):")
        for mech_name in MECHS:
            sub = df[df["mechanism"] == mech_name]
            top = pd.concat([sub.nlargest(4, col), sub.nsmallest(4, col)])
            print(f"  [{mech_name}]")
            for _, r in top.iterrows():
                print(f"    {r['reaction']:<38s} {r[col]:+.4e}")
    print(f"\n输出 -> {out}")


if __name__ == "__main__":
    rop_analysis()
