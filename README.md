# NH3/H2 机理 GA 简化、优化与验证

> [!IMPORTANT]
> **复现范围说明（2026-10-26）**
>
> 本仓库是对 GitHub 初版的整理、校正和扩展，不应理解为论文结果的逐点、逐图
> 严格全复现。项目精确保留了论文补充材料提供的 Mechanism 1/2，并实现了可运行
> 的 Cantera 验证和 GA 框架；但 Zhang 2021 原始详细机理、原始实验数据和 CONVERGE
> 三维模型不可得，因此部分环节采用重建机理、图件数字化数据或降阶替代模型。
>
> - **官方资产**：`Reduced mechanism.yaml` 与 `Optimized mechanism.yaml` 和论文补充
>   材料一致，分别对应论文 Mechanism 1/2。
> - **项目 GA 结果**：当前简化运行到 600 代，得到 34 组分/83 反应；优化运行到
>   34 代，属于演示规模，未重新得到论文的完整 1000 代结果。
> - **详细机理**：使用 36 组分/253 反应的公开重建版替代论文使用的
>   34 组分/263 反应原始文件。
> - **验证数据**：实验点主要由论文图件数字化，部分未公开边界条件依据公开资料
>   推断；当前 Fig.16 JSR 工况仍与论文图示工况不同。
> - **HCCI**：Fig.19 的三维网格未复现；Fig.20 使用 Cantera 0D 变容积反应器替代
>   CONVERGE 3D CFD，只能视为降阶近似。

## 相对 GitHub 初版的主要更新

- 更正论文出处为 *Energy* 270 (2023) 126927 和正确 DOI。
- 引入 Zhang 2021 公开重建机理，区分论文官方机理与项目自身 GA 产物。
- 增加 ROP、固定温度场 Fig.18、HCCI 0D、GA 终态化、统一绘图和一致性报告。
- 保存 GA 检查点、收敛历史、反应清单及全部统一格式图件。
- 清理 macOS 虚拟环境、IDE 配置、缓存、失败下载和重复备份；增加固定依赖文件。

复现对象：**Liu, X., Wang, Y., Bai, Y., Yang, W. (2023) "Development of
reduced and optimized mechanism for ammonia/hydrogen mixture based on
genetic algorithm", *Energy*, 270, 126927.**
https://doi.org/10.1016/j.energy.2023.126927

论文的全部 0D/1D 燃烧计算原本即用 Cantera 完成；本复现将 HCCI 发动机部分
（原文用 CONVERGE 3D CFD）以 Cantera 0D 变容积反应器替代，实现"仿真部分
全部用 Cantera"。

---

## 1. 机理链条（本复现的核心资产）

| 角色 | 文件 | 来源与性质 |
|---|---|---|
| **Mechanism 1**（论文简化机理） | `mechanisms/Reduced mechanism.yaml` | **论文官方补充材料原版**（MD5 与 Elsevier mmc1.zip 内文件一致），29 组分 / 63 反应 |
| **Mechanism 2**（论文优化机理） | `mechanisms/Optimized mechanism.yaml` | **论文官方补充材料原版**，29 组分 / 63 反应（25 个反应的 A 因子被优化） |
| **Full mechanism**（初始机理替代） | `mechanisms/Full mechanism 2021recon.yaml` | Zhang 2021 机理的**公开重建版**（36 组分 / 253 反应，见 §5） |
| GA 演示产物 | `mechanisms/GA_Reduced_mechanism.yaml`、`GA_Optimized_mechanism.yaml` | 本项目用 GA 从重建版重新简化/优化的结果 |

**重要说明**：论文的简化基础是 Zhang et al. (2021, *Combust. Flame* 234,
111653) 的 34 组分/263 反应机理，**该文件未公开提供**（期刊补充材料仅含验证
图；KAUST 仓库被反爬虫拦截）。本项目以下述"重建版"替代，所有偏差已定量记录
（§5）。

## 2. 目录结构

```
├── mechanisms/
│   ├── Reduced mechanism.yaml        # 论文原版 Mechanism 1 (勿动)
│   ├── Optimized mechanism.yaml      # 论文原版 Mechanism 2 (勿动)
│   ├── Full mechanism 2021recon.yaml # Zhang 2021 公开重建版
│   └── Full mechanism.yaml           # Stagni 2025 版 (早期演示用)
├── scripts/
│   ├── mech_utils.py                 # 共享工具: 混气组成/IDT/稳态T/子机理拼接
│   ├── rebuild_zhang2021.py          # Zhang 2021 重建脚本 (来源与修正记录)
│   ├── experimental_data.py          # 论文图件数字化实验数据
│   ├── 01_idt.py                     # IDT 扫描 (Fig.9,10,S1,S2)
│   ├── 02_lfs.py                     # LFS 扫描 (Fig.11-15,S3-S5)
│   ├── 03_jsr.py                     # JSR 浓度 (Fig.16, Zhang 稀释条件)
│   ├── 04_premixed_flame.py          # 火焰结构 (Fig.17,S6)
│   ├── 04b_fig18_fixedT.py           # Fig.18 (实验温度场输入法)
│   ├── 05_sensitivity.py             # IDT 敏感性 (Fig.6, 扰动法)
│   ├── 05b_rop_analysis.py           # NO/N2O 产率分析 (Fig.7)
│   ├── 06_ga_reduction.py            # GA 简化 (Section 2.2)
│   ├── 07_ga_optimization.py         # GA 优化 (Section 2.3)
│   ├── 08_hcci.py                    # HCCI 0D 发动机 (Fig.20)
│   ├── draw_paper.py                 # 验证图件 (Fig.6-20, S1-S6)
│   ├── draw_method.py                # 方法学图件 (Fig.1-5,8)
│   ├── consistency_report.py         # 一致性定量报告
│   └── finalize_ga.py                # GA 检查点终态化
├── results/
│   ├── figs/                         # 全部论文风格图件
│   ├── *.csv                         # 仿真数据
│   └── *_log.txt                     # 运行日志
├── mech_sources/                     # 机理来源原始文件 (Zhang 2023 等)
├── Liu 等 - 2023 - Development of reduced and optimized mechanism for ammonia hydrogen mixture based on genetic algori.pdf
│                                     # 论文 PDF
├── requirements.txt                  # 已验证的 Python 依赖版本
└── LICENSE                           # CC0 1.0（继承自 GitHub 初版）
```

## 3. 运行方式

```bash
# 已验证环境: Python 3.13 + Cantera 3.2
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt

# (A) 验证仿真 (论文原版 M1/M2 直接计算, 数小时)
python scripts/01_idt.py
python scripts/02_lfs.py
python scripts/03_jsr.py
python scripts/04_premixed_flame.py
python scripts/04b_fig18_fixedT.py
python scripts/08_hcci.py --Tintake 410 --u 300

# (B) 分析类
python scripts/05_sensitivity.py       # ~2 min
python scripts/05b_rop_analysis.py     # ~1 min

# (C) GA (重型计算, 每个约 2-5 h @ 12-16 核; 支持断点续跑)
python scripts/06_ga_reduction.py --gens 1000 --pop 60 --workers 12
python scripts/06_ga_reduction.py --gens 1000 --resume          # 崩溃/中断后续跑
python scripts/07_ga_optimization.py --gens 400 --pop 30 --workers 5
python scripts/finalize_ga.py    # 从检查点终态化 (机理 + Fig.4/5/8 + A 因子对标)
```

**本项目实际演示运行结果**（受单机时长约束，均为演示规模，可按上列命令续跑至论文规模）:
- GA 简化: 600 代收敛 (第 599 代检查点), 34 组分 / **83 反应**, 最优目标函数
  0.0676 (f_N=0.028)。反应数自 300 代起稳定在 83-89 (论文 1000 代得 63;
  基础机理不同所致, 见 §5-B)。
- GA 优化: 34 代检查点, 目标函数 3.09 → **1.33** (f_LFS 1.93 → 0.34,
  即 LFS 相对实验偏差显著下降 —— 复现论文"优化主要改善 LFS 预测"的结论)。
  与论文原版 M2 的 A 因子对比: 论文修改 27 个反应 / 我们 34 个 / 重叠 21 个;
  多个大幅调整方向一致 (NH3+O⇌NH2+OH: 我们 ×5.4 vs 论文 ×8.0;
  H+N2H2⇌H2+NNH: ×4.7 vs ×6.0); 全局对数相关 r≈0 (仅 34 代 vs 论文 1000 代,
  且实验目标为数字化近似)。详见 results/ga_optimization_a_factors.csv。
- 数值稳健性说明: 简化/优化中个别基因会触发 Cantera 原生刚性积分停滞或
  火焰近熄灭发散, 已实现 (i) CVODE 停滞/步数保护; (ii) 火焰求解子进程
  硬超时 + 缓存重暖; (iii) 进程池崩溃自愈 (拉黑崩溃基因)。
  火焰求解中乘数夹紧至 [0.25,4] 以保数值稳定 (真实基因值仍进入 f_A 惩罚),
  为本实现相对论文的唯一算法性偏差。

```bash
# (D) 出图与报告
python scripts/draw_method.py          # Fig.1-5, 8
python scripts/draw_paper.py           # Fig.6-20, S1-S6
python scripts/consistency_report.py   # results/consistency_report.md
```

## 4. 复现口径与论文的对齐关系

| 论文内容 | 复现方式 | 一致性 |
|---|---|---|
| Table 1/2 GA 参数 | 完全一致 (pop 60/30, 交叉 0.2/0.3, 变异 0.03/0.06, 锦标赛+精英) | 逐项 |
| Eq.(1)-(8) 目标函数 | 完全一致 (σ/λ 逐项相同; 均值口径) | 逐项 |
| Table 3 Case 1-8 | Chen 2021 激波管混合气 (4% 燃料+化学计量 O2+Ar, φ=1; 由 ReSpecTh/ELTE 记录的工况范围反推并经原版 M1/M2 量级验证) | 见 §5-D |
| Fig. 6 敏感性 | 有限差分扰动法 (k×2), 8 工况归一化, Top-20 | 定性一致 (H+O2⇌O+OH 主导) |
| Fig. 7 产率分析 | NO 生成速率峰值时刻瞬时 ROP, 1200 K, φ=1, 80/20 | 与论文结论逐条对应 |
| Fig. 9-15, 17-18, S1-S6 | 论文原版 M1/M2 + 重建版 Full, Cantera 0D/1D | 多数趋势一致，但实验点为数字化近似 |
| Fig. 16 JSR | 当前使用 XH2=0/0.05/0.3、停留时间 2 s | 与论文图示 0.5/0.1、1 s 工况不一致，待修正 |
| Fig. 19/20 HCCI | Fig.19 未复现；Fig.20 以 0D 变容积反应器替代 CONVERGE 3D | 属降阶替代，M2 峰值误差约 14-35% |
| 优化目标 M2 | 论文原版 M2 已持有; 本项目 GA 优化结果与其 A 因子对标 | 见 ga_optimization_a_factors.csv |

## 5. 与原文不一致处的证据与归因

依要求：**凡与原文不一致处，或给出"原文有误"的证据，或明确归因于数据可得性。**

**(A) 原文可证实的笔误**
1. 论文正文称简化基础机理为 "**Wang et al. [21] in 2021**"，但参考文献 [21]
   实为 **Zhang X., Moosakutty, Rajan, Younes, Sarathy** (CNF 234:111653)。
   引用与文献列表不符，属原文笔误（作者名/引文错位）。
2. 论文 §3.1 称优化目标为 **45** 个反应、§3.2 又称 **46** 个，前后不一致。
   本项目按"63 − falloff/压力依赖 − H2 子机理"的规则独立计数为 50
   （H2 子机理的边界划分存在解释空间），三者不同属定义模糊而非单方错误。

**(B) 基础机理不可得导致的偏差（数据可得性，非原文错误）**
Zhang 2021 的 34 组分/263 反应机理文件未公开（Elsevier 补充材料仅含验证图；
KAUST 仓库 Incapsula 拦截；ReSpecTh 无直链下载；各镜像渠道均无）。本项目以
Zhang 2023 (Fuel) 公开的 CO-H2 子模型为底重建：
   - 移除 2023 新增的 HCO/HOCO 化学（23 反应、2 组分）→ 组分数恰为 34+2；
   - 以 Liu 补充材料中 M1 保留的 63 个反应（含 2021 原始参数）覆盖被 2023
     修改的 7 个反应、追加被 2023 删除的 2 个反应；
   - 残余偏差：2023 相对 2021 净删除的 ~10 个反应与少量 ZXY_modified 修改
     无法恢复。
   - **重建版一致性验证**：与论文原版 M1 的 IDT 偏差 3-8%（论文自述 M1 与
     Full 的 f_tau≈0.02 量级，即重建版在"M1 与 Full 的原始一致性"范围内）。

**(C) 复现与原文的数值差异及归因**
1. Fig.9 (a) M1 低温段偏快 3-5 倍：M1 为论文原版文件，差异来自实验点为
   视觉数字化（±20-40%）及 Chen 混合气燃料浓度（4%）为推断值。
2. Fig.20 (b) 峰值压力低 20-35%：0D 模型无湍流/缸内空间效应，传热用简化
   Woschni 系数（U=300 W/m²K, 由 (a) 工况标定后用于 (b)）。点火相位与
   峰值量级仍复现论文核心结论（M2 相位最贴近实验）。
3. Fig.16 的 NO/N2O 预测低于实验：与论文自述完全一致
   ("the predicted concentrations of NO and N2O were lower than the
   experimental values")。
4. Fig.18 的 NO 消耗偏快、N2/H2O 偏低：实验温度场为数字化近似（±10%），
   且 Cantera 固定温度场法为论文方法的等效实现。
5. 论文 Fig.9-18 中 "Full mechanism" 曲线由重建版计算，与原文曲线存在
   ~10-30% 系统差（重建版残余偏差所致）；M1/M2 曲线为论文原版机理计算，
   可与原文直接对比。

**(D) 工况口径的重要更正**
论文 Table 3 Case 1-8 未写明稀释气。经 ReSpecTh/ELTE 基准数据库核对，
Chen (2021) 实验为 **Ar 稀释激波管混合气**（φ=1，1.11-1.26 / 9.2-12.6 atm，
各系列温度范围与 Table 3 一一对应），而项目早期版本误用空气混合气。
本复现全部按 Ar 稀释口径计算（燃料总浓度 4% 为量级反推值，经论文原版
M1/M2 与数字化实验点的吻合度验证）。

## 6. 图件清单（results/figs/，与原文一一对应）

| 原文图 | 本项目文件 | 状态 |
|---|---|---|
| Fig.1 研究流程图 | fig_01_flowchart.png | 示意图 |
| Fig.2 GA 活动图 | fig_02_ga_activity.png | 示意图 |
| Fig.3 目标函数曲线 | fig_03_objective_terms.png | 解析复现 |
| Fig.4 简化收敛 | fig_04_reduction_convergence.png | GA 运行产出 |
| Fig.5 基因图 | fig_05_gene_map.png | GA 运行产出 |
| Fig.6 敏感性 Top-20 | fig_06_sensitivity.png | 完成 |
| Fig.7 NO/N2O 产率 | fig_07_rop.png | 完成 |
| Fig.8 优化收敛 | fig_08_optimization_convergence.png | GA 运行产出 |
| Fig.9 IDT (Chen) | fig_09_idt_chen.png | 完成 |
| Fig.10 纯 NH3 IDT | fig_10_idt_nh3.png | 完成 |
| Fig.11 LFS vs φ | fig_11_lfs_phi.png | 完成 |
| Fig.12 LFS 对比 | fig_12_lfs_comparison.png | 完成 |
| Fig.13 LFS vs X[H2] | fig_13_lfs_xh2.png | 完成 |
| Fig.14 LFS vs 压力 | fig_14_lfs_pressure.png | 完成 |
| Fig.15 LFS vs 温度 | fig_15_lfs_temperature.png | 完成 |
| Fig.16 JSR | fig_16_jsr.png | 完成 |
| Fig.17 火焰结构 | fig_17_flame_structure.png | 完成 |
| Fig.18 NH3/NO/Ar | fig_18_flame_nh3_no_ar.png | 完成 (固定温度场法) |
| Fig.19 3D 网格 | —（无法用 Cantera 复现，见 §5） | 跳过 |
| Fig.20 HCCI 压力 | fig_20_hcci.png | 完成 (0D 替代) |
| Fig.S1-S6 | fig_S1..S6_*.png | 完成 |

## 7. 已知局限

1. Zhang 2021 原始机理不可得（§5-B）；如后续获得，替换
   `mech_utils.FULL_MECH` 指向即可全链路重算。
2. 实验数据为论文图件数字化（精度 ±10-40%），非原始数据表。
3. HCCI 为 0D 近似（无湍流、无空间分布），进气温度 410 K 为标定值。
4. GA 优化演示运行 34 代（结果表含初始代，共 35 行；论文 1000 代）；脚本支持
   `--gens 1000 --resume` 续跑。
5. 论文 Fig.11/12 各子图的具体分配依据图注与文献推断（Fig.12 取
   XH2=0.4/0.6 两子图）。

## 8. 参考

- Liu, X., Wang, Y., Bai, Y., Yang, W. Energy 270 (2023) 126927.
- Zhang, X., Moosakutty, S.P., Rajan, R.P., Younes, M., Sarathy, S.M.
  Combust. Flame 234 (2021) 111653.（简化基础机理）
- Zhang, X., Yalamanchi, K.K., Sarathy, S.M. Fuel (2023)（2021 机理的
  公开后续版本，重建版来源）
- Chen, J., Jiang, X., Qin, X., Huang, Z. Fuel 287 (2021) 119563.
  （Fig.9 激波管实验）
- Sikalo, N. et al.（GA 简化方法来源）；Cantera 3.2（全部 0D/1D 仿真）。

## 9. 许可证与第三方材料

本仓库延续 GitHub 初版的 CC0 1.0 许可证；该许可仅适用于仓库作者有权许可的
项目代码、说明和自生成数据。根目录论文 PDF、`mech_sources/` 及部分 `mechanisms/`
内容来自论文作者、出版社或其他研究机构，其著作权和再分发条件仍归原权利人
所有。保留这些文件是为了研究可追溯性，不代表将第三方材料重新许可为 CC0；
使用或再分发前请自行核对对应来源的许可条件。
