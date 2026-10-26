"""
HCCI 发动机缸内压力 0D 仿真 - 复现论文 Fig. 20 (Cantera 替代 CONVERGE 3D CFD)

论文用 CONVERGE 3D CFD (RNG k-eps, 60° 扇形网格) 计算 HCCI 燃烧;
本复现采用 Cantera 变容积均质反应器 (0D, 曲轴连杆运动学 + Woschni 传热),
作为"仿真部分用 Cantera 替代"口径下的近似 (无湍流/空间效应)。

发动机参数 (论文 Table 4, Pochet et al. [11]):
  排量 463 cm³, 缸径×行程 86×75 mm, 压缩比 22, 转速 1500 rpm,
  phi = 0.22, X[NH3]:X[H2] = 0.37:0.63 与 0.57:0.43, 进气压力 1 bar
  计算: IVC (-127 °CA BTDC) 至 EVO (127 °CA ATDC)
进气温度论文未给出 (3D CFD 需要的壁面/进气边界未公开):
  默认 425 K, 由点火相位标定 (可通过 --Tintake 调整)。

输出: results/hcci_pressure.csv
"""
import argparse
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
    "Mechanism 2": "Optimized mechanism.yaml",
}

VD = 463e-6          # m^3 排量 (Table 4)
CR = 22.0
RPM = 1500.0
BORE = 0.086
STROKE = 0.075
ROD = 0.127          # 连杆长度 (典型行程/连杆比)
T_WALL = 450.0       # 壁面温度 K
P_IN = 1.0           # bar
PHI = 0.22
CASES = {"XH2=0.63": 0.63, "XH2=0.43": 0.43}
THETA_IVC = -127.0
THETA_EVO = 127.0
OMEGA = RPM * 2.0 * np.pi / 60.0


def crank_volume(theta_deg):
    """曲轴转角 (deg ATDC) -> 缸内容积 m^3"""
    th = np.radians(theta_deg)
    a = STROKE / 2.0
    s = a * (1.0 - np.cos(th) + ROD / a *
             (1.0 - np.sqrt(1.0 - (a / ROD * np.sin(th)) ** 2)))
    Vc = VD / (CR - 1.0)
    return Vc + VD * s / (2.0 * a)


def dvdt(t):
    """dV/dt (m^3/s), t=0 对应 IVC"""
    th_rad = np.radians(THETA_IVC + t * RPM * 6.0)
    a = STROKE / 2.0
    sin_t, cos_t = np.sin(th_rad), np.cos(th_rad)
    dVdth = VD / 2.0 * (sin_t + (a / ROD) * sin_t * cos_t /
                        np.sqrt(1.0 - (a / ROD * sin_t) ** 2))
    return dVdth * np.pi / 180.0 * RPM * 6.0


def run_hcci(mech_path, XH2, T_in, u_heat=1500.0):
    gas = ct.Solution(str(mech_path))
    comp = phi_composition(XH2, PHI)
    gas.TPX = T_in, P_IN * ct.one_atm, comp

    reactor = ct.IdealGasReactor(gas)
    reactor.volume = crank_volume(THETA_IVC)      # IVC 缸内容积
    # Cantera 3.x: 用环境 reservoir 实现移动边界与传热
    env_gas = ct.Solution(str(mech_path))
    env_gas.TPX = T_WALL, 10.0 * ct.one_atm, {"N2": 1.0}
    env = ct.Reservoir(env_gas)

    # 移动活塞 (绝热): area=1, velocity(t) = dV/dt
    wall_piston = ct.Wall(reactor, env, A=1.0, K=0.0, U=0.0)
    wall_piston.velocity = dvdt
    # 传热壁 (固定): U*A ~ 1500 * 0.012
    area_heat = np.pi * BORE ** 2 / 4.0 + np.pi * BORE * 0.06
    wall_heat = ct.Wall(reactor, env, A=area_heat, K=0.0,
                        U=u_heat)

    net = ct.ReactorNet([reactor])
    net.rtol = 1e-8
    t_end = (THETA_EVO - THETA_IVC) / 360.0 / (RPM / 60.0)
    rows = []
    t, next_t = 0.0, 0.0
    dt_out = 0.5 / 360.0 / (RPM / 60.0)
    while t < t_end:
        t = net.step()
        while next_t <= t:
            th = THETA_IVC + next_t * RPM * 6.0
            rows.append({"theta_deg_ATDC": th, "time_s": next_t,
                         "P_bar": reactor.thermo.P / 1e5,
                         "T_K": reactor.T, "V_m3": crank_volume(th),
                         "XH2": XH2})
            next_t += dt_out
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Tintake", type=float, default=425.0)
    ap.add_argument("--u", type=float, default=1500.0,
                    help="传热系数 W/m2K")
    args = ap.parse_args()

    mech_dir = Path(__file__).resolve().parent.parent / "mechanisms"
    all_rows = []
    for mech_name, mech_file in MECHS.items():
        for case, XH2 in CASES.items():
            print(f"=== HCCI {mech_name} {case} "
                  f"(T_in={args.Tintake}K) ===", flush=True)
            try:
                df = run_hcci(mech_dir / mech_file, XH2, args.Tintake, args.u)
                df["mechanism"] = mech_name
                all_rows.append(df)
                pmax = df["P_bar"].max()
                th_pmax = df.loc[df["P_bar"].idxmax(), "theta_deg_ATDC"]
                print(f"  Pmax={pmax:.1f} bar @ {th_pmax:+.1f} deg ATDC",
                      flush=True)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        out.to_csv(RESULTS_DIR / "hcci_pressure.csv", index=False)
        print(f"\nHCCI 完成 {len(out)} 行 -> results/hcci_pressure.csv")
    else:
        print("\nHCCI 全部失败")


if __name__ == "__main__":
    main()
