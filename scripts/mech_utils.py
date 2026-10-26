"""
共享工具模块: 混气组成 / IDT / 稳态温度 / 子机理构建 (原文拼接法)
所有仿真脚本与 GA 脚本共用, 保证全流程计算口径一致。
"""
import re
import time
import numpy as np
import cantera as ct
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MECH_DIR = PROJECT_ROOT / "mechanisms"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# "完整机理" = Zhang 2021 (CNF 234:111653) 的重建版 (36 组分/253 反应)
# 由 Zhang 2023 (Fuel) 公开的 CO-H2 子模型去掉 2023 新增 C1 化学、
# 并用 Liu 2023 补充材料中保留的 2021 原始反应参数修正得到。
# 详见 scripts/rebuild_zhang2021.py 与 README。
FULL_MECH = MECH_DIR / "Full mechanism 2021recon.yaml"
# 备用: Stagni 2025 版 (原项目使用的基础机理)
FULL_MECH_STAGNI = MECH_DIR / "Full mechanism.yaml"
REDUCED_MECH = MECH_DIR / "Reduced mechanism.yaml"
OPTIMIZED_MECH = MECH_DIR / "Optimized mechanism.yaml"


# ------------------------------------------------------------
# 混气组成
# ------------------------------------------------------------
def phi_composition(XH2=0.0, phi=1.0, diluent="N2", n2_o2=3.76,
                    X_NH3=None, X_H2=None):
    """
    NH3/H2 燃料 + 氧化剂 (O2 + 稀释气) 按当量比 phi 构成摩尔分数字典。
    XH2: 燃料中 H2 摩尔分数 (0-1)
    化学计量: NH3 + 0.75 O2 -> 0.5 N2 + 1.5 H2O;  H2 + 0.5 O2 -> H2O
    diluent: "N2" (空气) / "AR" (Ar 稀释)
    """
    if X_NH3 is None:
        X_NH3 = 1.0 - XH2
    if X_H2 is None:
        X_H2 = XH2
    o2_stoich = 0.75 * X_NH3 + 0.5 * X_H2
    o2 = o2_stoich / phi
    comp = {}
    if X_NH3 > 0:
        comp["NH3"] = X_NH3
    if X_H2 > 0:
        comp["H2"] = X_H2
    comp["O2"] = o2
    comp[diluent] = o2 * n2_o2
    return comp


def chen_composition(XH2, phi=1.0, fuel_frac=0.04):
    """
    Chen et al. (2021, Fuel 287:119563) 激波管混合气 (论文 Fig.9/Table 3 Case1-8):
    燃料 (NH3/H2) 总量 4% + 化学计量 O2 + Ar 平衡, phi=1。
    (燃料浓度由论文图 9 实验点量级与 Ar 稀释体系推断, 精度经原版 M1/M2 复核)
    """
    nh3 = fuel_frac * (1.0 - XH2)
    h2 = fuel_frac * XH2
    o2 = (0.75 * nh3 + 0.5 * h2) / phi
    return {"NH3": nh3, "H2": h2, "O2": o2, "AR": 1.0 - nh3 - h2 - o2}


def dissociated_nh3_composition(alpha, phi, diluent="N2", n2_o2=3.76):
    """
    部分解离氨混气 (Fig. S5, Ji et al.): 2NH3 -> N2 + 3H2
    alpha = X_H2 / (X_NH3 + X_H2) 为解离气中 H2 占比
    返回 (XH2_in_fuel, fuel_dict) 供 phi_composition 使用
    """
    # 解离气: NH3 = 2(1-d), H2 = 3d, N2 = d  (以 2 mol NH3 为基准)
    # alpha = 3d / (2(1-d) + 3d) => d = 2*alpha / (3 - alpha)
    d = 2.0 * alpha / (3.0 - alpha)
    x_nh3 = 2.0 * (1.0 - d)
    x_h2 = 3.0 * d
    x_n2 = d
    s = x_nh3 + x_h2 + x_n2
    fuel = {"NH3": x_nh3 / s, "H2": x_h2 / s, "N2": x_n2 / s}
    o2_stoich = 0.75 * fuel["NH3"] + 0.5 * fuel["H2"]
    o2 = o2_stoich / phi
    comp = dict(fuel)
    comp["O2"] = o2
    comp[diluent] = o2 * n2_o2
    return comp


# ------------------------------------------------------------
# 0D 反应器仿真
# ------------------------------------------------------------
def ignition_delay(gas, T, P_atm, comp, t_end, dT_break=400.0):
    """
    定容反应器点火延迟 (OH 摩尔分数峰值时刻)。
    gas: ct.Solution (会被设置状态); 返回 (tau_s, 达峰前步数) 或 (nan, 0)
    """
    gas.TPX = T, P_atm * ct.one_atm, comp
    reactor = ct.IdealGasReactor(gas)
    net = ct.ReactorNet([reactor])
    net.rtol = 1.0e-10
    net.atol = 1.0e-20
    oh_idx = gas.species_index("OH")
    t = -1.0
    tau = np.nan
    oh_max = -1.0
    n_stall = 0
    n_steps = 0
    try:
        while t < t_end:
            t_new = net.step()
            n_steps += 1
            if t_new - t < 1e-18:          # CVODE 停滞保护
                n_stall += 1
                if n_stall > 500:
                    return np.nan
            else:
                n_stall = 0
            t = t_new
            oh = reactor.thermo[oh_idx].X[0]
            if oh > oh_max:
                oh_max = oh
                tau = t
            if reactor.T - T > dT_break:
                break
            if n_steps > 500000:           # 步数上限保护
                return np.nan
    except Exception:
        return np.nan
    return tau


def steady_state_temperature(gas, T, P_atm, comp, t_run, n_extend=3):
    """
    定压反应器绝热稳态温度: 推进 t_run, 若末期温度仍在快速变化则延长。
    """
    gas.TPX = T, P_atm * ct.one_atm, comp
    reactor = ct.IdealGasConstPressureReactor(gas)
    net = ct.ReactorNet([reactor])
    t_final = t_run
    try:
        for _ in range(n_extend + 1):
            _advance_guarded(net, t_final)
            T1 = reactor.T
            prev = t_final
            t_final = prev * 2.0
            _advance_guarded(net, t_final)
            dT = reactor.T - T1
            if abs(dT) < 2.0:
                break
    except Exception:
        pass
    return reactor.T


def _advance_guarded(net, t_target, max_steps=200000):
    """带停滞/步数保护的 advance (避免 CVODE 停滞时死循环)"""
    t = net.time
    n_stall = 0
    for _ in range(max_steps):
        t_new = net.step()
        if t_new - t < 1e-18:
            n_stall += 1
            if n_stall > 500:
                return
        else:
            n_stall = 0
        t = t_new
        if t >= t_target:
            return


def evaluate_idt_set(gas, conditions, tau_ref=None, factor=50.0):
    """
    对一组工况计算 IDT。
    conditions: [(XH2, P_atm, T, phi, diluent, n2_o2), ...]
    tau_ref: 各工况参考 IDT (定 t_end 用); 无则先用粗略 t_end
    返回 dict: index -> tau (s)
    """
    out = {}
    for i, cond in enumerate(conditions):
        XH2, P, T, phi, dil, n2o2 = cond
        comp = phi_composition(XH2, phi, dil, n2o2)
        if tau_ref is not None and np.isfinite(tau_ref[i]) and tau_ref[i] > 0:
            t_end = min(max(factor * tau_ref[i], 1e-5), 20.0)
        else:
            t_end = 10.0 if T < 1200 else (1.0 if T < 1500 else 0.2)
        out[i] = ignition_delay(gas, T, P, comp, t_end)
    return out


def evaluate_ts_set(gas, conditions, tau_ref, factor=200.0):
    """对一组工况计算稳态温度, t_run = factor*tau_ref"""
    out = {}
    for i, cond in enumerate(conditions):
        XH2, P, T, phi, dil, n2o2 = cond
        comp = phi_composition(XH2, phi, dil, n2o2)
        t_run = min(max(factor * (tau_ref[i] if np.isfinite(tau_ref[i]) else 1e-3),
                        5e-3), 50.0)
        out[i] = steady_state_temperature(gas, T, P, comp, t_run)
    return out


# ------------------------------------------------------------
# 子机理构建 (原文块拼接, 保持 Cantera YAML 有效性)
# ------------------------------------------------------------
class MechBuilder:
    """
    从完整机理文本中按基因 (0/1 向量) 拼接子机理 YAML 字符串。
    处理: duplicate 反应只保留单侧时移除 duplicate 标记。
    """

    _DUP_RE = re.compile(r"(?m)^\s{2}duplicate: true\s*$")

    def __init__(self, full_mech_path=FULL_MECH):
        text = Path(full_mech_path).read_text()
        lines = text.splitlines(keepends=True)
        # 定位顶层 sections
        ridx = next(i for i, l in enumerate(lines) if l.startswith("reactions:"))
        sidx = next(i for i, l in enumerate(lines) if l.startswith("species:"))
        self.head = "".join(lines[:ridx + 1])            # 含 phases + species 定义
        rsec = "".join(lines[ridx + 1:])
        # 仅以 "- equation:" 起始的行为新反应块 (避免 note 文本干扰)
        self.rxn_blocks = [b for b in re.split(r"(?m)^(?=- equation: )", rsec)
                           if b.strip()]
        # species 定义块 (顶层 species: 到 reactions: 之间)
        ssec = "".join(lines[sidx + 1:ridx])
        self.sp_blocks = [b for b in re.split(r"(?m)^(?=- name: \S)", ssec)
                          if b.strip()]
        self.species_names = [self._block_name(b) for b in self.sp_blocks]
        self.duplicate_groups = self._find_duplicates()
        self.n_reactions = len(self.rxn_blocks)
        self.n_species = len(self.species_names)

    @staticmethod
    def _block_name(block):
        m = re.match(r"- name:\s*(\S+)", block)
        return m.group(1) if m else None

    def _find_duplicates(self):
        """返回 duplicate 反应的索引集合"""
        dup = set()
        for i, b in enumerate(self.rxn_blocks):
            if re.search(r"(?m)^\s{2}duplicate: true\s*$", b):
                dup.add(i)
        return dup

    def build(self, gene):
        """按基因拼接子机理 YAML 字符串 (保留全部组分定义)"""
        kept = [i for i, g in enumerate(gene) if g == 1]
        # duplicate 处理: 统计每个 duplicate 方程保留的数量
        kept_set = set(kept)
        parts = []
        from collections import defaultdict
        eq_count = defaultdict(int)
        eqs = {}
        for i in kept:
            if i in self.duplicate_groups:
                eq = self._equation_of(self.rxn_blocks[i])
                eq_count[eq] += 1
                eqs[eq] = i
        for i in kept:
            b = self.rxn_blocks[i]
            if i in self.duplicate_groups:
                eq = self._equation_of(b)
                if eq_count[eq] == 1:
                    b = self._DUP_RE.sub("", b)
            parts.append(b)
        return self.head + "".join(parts)

    @staticmethod
    def _equation_of(block):
        m = re.match(r"-\s+equation:\s*(.+?)(?:\s+#|$)", block)
        return m.group(1).strip() if m else block[:60]

    def used_species(self, gene):
        """基因保留反应所涉及的组分 (反应物+产物+效率组分) + 常驻惰性气"""
        used = set()
        kept_blocks = [self.rxn_blocks[i] for i, g in enumerate(gene) if g == 1]
        kept_text = "".join(kept_blocks)
        for sp in self.species_names:
            if sp == "M":
                continue
            # 词边界匹配组分名
            if re.search(rf"(?<![\w])({re.escape(sp)})(?![\w])", kept_text):
                used.add(sp)
        used.update({"N2", "AR", "HE"} & set(self.species_names))
        return used

    def build_pruned(self, gene, keep_species=None):
        """
        构建组分剪枝后的子机理: 移除未参与任何保留反应且非常驻惰性气的组分。
        同时剪枝 phase 中的 species 列表与 species 定义块。
        """
        keep = self.used_species(gene)
        if keep_species:
            keep |= set(keep_species)
        # 1) reactions 部分同 build()
        sub_rxns = "".join(
            b for i, g in enumerate(gene) if g == 1
        ) if not self.duplicate_groups else None
        # 重新走 build 的 duplicate 逻辑
        gene_list = [1 if g == 1 else 0 for g in gene]
        rxn_text = self._rxn_text(gene_list)
        # 2) species 定义块剪枝
        sp_text = "".join(
            b for b, name in zip(self.sp_blocks, self.species_names)
            if name in keep
        )
        sp_header = "species:\n"
        # 3) 重组装: head 含 phases + species 定义, 需要拆开
        lines = self.head.splitlines(keepends=True)
        sidx = next(i for i, l in enumerate(lines) if l.startswith("species:"))
        ridx = next(i for i, l in enumerate(lines) if l.startswith("reactions:"))
        # phase species 列表替换: 解析 phases 段的 species: [ ... ] 多行列表
        phase_part = "".join(lines[:sidx])
        rxn_header = "".join(lines[ridx:])
        phase_part = self._replace_phase_species(phase_part, keep)
        return phase_part + sp_header + sp_text + rxn_header + rxn_text

    def _rxn_text(self, gene_list):
        from collections import defaultdict
        kept = [i for i, g in enumerate(gene_list) if g == 1]
        eq_count = defaultdict(int)
        for i in kept:
            if i in self.duplicate_groups:
                eq_count[self._equation_of(self.rxn_blocks[i])] += 1
        parts = []
        for i in kept:
            b = self.rxn_blocks[i]
            if i in self.duplicate_groups:
                eq = self._equation_of(b)
                if eq_count[eq] == 1:
                    b = self._DUP_RE.sub("", b)
            parts.append(b)
        return "".join(parts)

    @staticmethod
    def _replace_phase_species(phase_text, keep):
        """替换 phase 定义中的 species: [...] 列表 (支持多行流式列表)"""
        m = re.search(r"(?ms)^(\s+species:\s*)\[.*?\]", phase_text)
        if not m:
            return phase_text
        names = [n for n in
                 sorted(keep, key=lambda s: s)]
        new_list = ", ".join(names)
        return phase_text[:m.start()] + m.group(1) + "[" + new_list + "]" + \
            phase_text[m.end():]


def solution_from_gene(builder, gene):
    """由基因构建子机理并解析为 Solution"""
    return ct.Solution(yaml=builder.build(gene))


def load_full(path=FULL_MECH):
    return ct.Solution(str(path))


if __name__ == "__main__":
    b = MechBuilder()
    print(f"Full mechanism: {b.n_species} species, {b.n_reactions} reactions, "
          f"{len(b.duplicate_groups)} duplicate-flagged")
    # 自检: 全 1 基因
    gene = np.ones(b.n_reactions, dtype=int)
    g = solution_from_gene(b, gene)
    print("all-ones gene ->", g.n_reactions, "reactions OK")
    # 随机基因 (保留 ~25%)
    rng = np.random.default_rng(0)
    gene2 = (rng.random(b.n_reactions) < 0.25).astype(int)
    g2 = solution_from_gene(b, gene2)
    print("random 25% gene ->", g2.n_reactions, "reactions OK")
    # 剪枝自检
    pruned = b.build_pruned(gene2)
    g3 = ct.Solution(yaml=pruned)
    print("pruned ->", g3.n_species, "species,", g3.n_reactions, "reactions OK")
