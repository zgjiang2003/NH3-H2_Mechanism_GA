"""
构建 Zhang 2021 (CNF 234:111653) 机理的"重建版" (2021recon)

背景: Liu et al. (2023) 的简化基础机理为 Zhang et al. 2021 (34 组分/263 反应),
该机理文件未公开提供。可获得的最近亲缘版本为 Zhang et al. 2023 (Fuel 451, 137407)
补充材料中的 KAUST_NH3-NOx-CO-H2 子模型 (40 组分/274 反应, 由 2021 版更新而来,
文件内仅 13 处 ZXY_ADDED/ZXY_modified 标记)。

重建步骤:
1. 以 2023 CO-H2 版为基础, 移除 2023 年新增的 HCO/HOCO 相关化学 (23 个反应,
   2 个组分) -> 34 组分 (与 2021 版组分计数一致)
2. 补回 2021 版作为三体碰撞体的 CH4/C2H6 组分定义 (取自 Liu 2023 补充材料中
   Reduced mechanism.yaml, 即 2021 版原始热力学/输运数据)
3. 用 Liu Reduced mechanism (2021 原始参数) 覆盖其中参数被 2023 版修改的反应
   (63 个保留反应中 49 个已一致, 14 个被 ZXY 修改 -> 全部恢复 2021 值)

已知残余偏差 (无法从公开渠道恢复):
- 2023 版相对 2021 版删除的 ~12 个反应 (ZXY_deleted 等)
- 2021 版 263 反应中, 不在 Liu 保留集 (63) 且被 ZXY_modified 的少量反应

输出: mechanisms/Full mechanism 2021recon.yaml
"""
import re
import sys
import numpy as np
import cantera as ct
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
SRC = PROJ / "mech_sources" / "zhang2023_COH2.yaml"
LIU_RED = PROJ / "mechanisms" / "Reduced mechanism.yaml"
OUT = PROJ / "mechanisms" / "Full mechanism 2021recon.yaml"


def split_blocks(text, section_header):
    """按顶层 '- ' 块切分指定 section"""
    lines = text.splitlines(keepends=True)
    idx = next(i for i, l in enumerate(lines)
               if l.startswith(section_header))
    sec = "".join(lines[idx + 1:])
    # section 结束于下一个顶层关键词
    out = []
    cur = []
    for l in sec.splitlines(keepends=True):
        if re.match(r"^[A-Za-z-]+:", l):
            break
        if l.startswith("- "):
            if cur:
                out.append("".join(cur))
            cur = [l]
        elif cur:
            cur.append(l)
    if cur:
        out.append("".join(cur))
    return lines, idx, out


def eq_key(block):
    """反应块方程式的规范键: 提取方程并规范化 (展开 2 X -> X + X)"""
    m = re.match(r"-\s+equation:\s*(.+?)(?:\s+#|$)", block)
    if not m:
        return None
    eq = m.group(1).strip()
    eq = eq.replace("(+M)", "").replace("+", " ").replace("<=>", "=").replace("=>", "=")
    toks = []
    for side in eq.split("="):
        sp = []
        for t in side.split():
            if t == "M":
                sp.append("M")
                continue
            mm = re.match(r"^(\d+)\s+(.+)$", t)
            if mm:
                sp.extend([mm.group(2)] * int(mm.group(1)))
            else:
                sp.append(t)
        toks.append(tuple(sorted(sp)))
    return toks


def main():
    src_text = SRC.read_text()
    liu_text = LIU_RED.read_text()

    # ---- 1) 解析 2023 版反应块 ----
    zlines, zidx, zblocks = split_blocks(src_text, "reactions:")
    # ---- 2) 解析 Liu 2021 参数反应块 ----
    llines, lidx, lblocks = split_blocks(liu_text, "reactions:")

    zkeys = [eq_key(b) for b in zblocks]
    lkeys = [eq_key(b) for b in lblocks]

    # ---- 3) 移除 HCO/HOCO 反应, 记录替换 ----
    remove_idx = set()
    replaced = []
    for i, b in enumerate(zblocks):
        m = re.match(r"-\s+equation:\s*(.+?)(?:\s+#|$)", b)
        eq = m.group(1)
        if re.search(r"(?<![\w])(HCO|HOCO)(?![\w])", eq):
            remove_idx.add(i)

    # Liu 反应 -> 2023 匹配 (按方程键, 一一对应), 参数覆盖; 未匹配的 (2023 已删除) 追加回
    n_same = n_replace = n_missing = 0
    appended = []
    used = set()
    for lb, lk in zip(lblocks, lkeys):
        if lk is None:
            continue
        matched = [i for i, zk in enumerate(zkeys)
                   if zk == lk and i not in remove_idx and i not in used]
        if not matched:
            # 2021 有而 2023 删除的反应 -> 追加 (2021 原始参数)
            zblocks.append(lb)
            zkeys.append(lk)
            appended.append(lb.splitlines()[0][:60])
            n_missing += 1
            continue
        i = matched[0]
        used.add(i)
        if rate_equal(zblocks[i], lb):
            n_same += 1
        else:
            zblocks[i] = lb
            replaced.append(lb.splitlines()[0][:60])
            n_replace += 1

    print(f"Liu 63 反应 vs 2023: 一致 {n_same}, 覆盖为2021参数 {n_replace}, "
          f"2023已删除而追加回 {n_missing}")
    for s in appended:
        print("  [追加2021反应]", s)

    # ---- 4) 重组 reactions 文本 ----
    # 4a) duplicate 标记一致性: 同方程组 >=2 个成员 => 全部标记; 单例 => 去除标记
    from collections import defaultdict
    groups = defaultdict(list)
    for i, b in enumerate(zblocks):
        k = eq_key(b)
        if k is not None:
            groups[tuple(map(str, k))].append(i)
    for k, idxs in groups.items():
        if len(idxs) > 1:
            for i in idxs:
                if "duplicate: true" not in zblocks[i]:
                    zblocks[i] = re.sub(r"(?m)^(- equation:.*\n)",
                                        r"\1  duplicate: true\n",
                                        zblocks[i], count=1)
        else:
            i = idxs[0]
            if "duplicate: true" in zblocks[i]:
                zblocks[i] = re.sub(r"(?m)^  duplicate: true\n", "",
                                    zblocks[i])

    new_rxns = "".join(b for i, b in enumerate(zblocks) if i not in remove_idx)

    # ---- 5) species 定义: 移除 HCO/HOCO 块, 追加 CH4/C2H6 块 (来自 Liu) ----
    slines, sidx, sblocks_src = split_blocks(src_text, "species:")
    # 修正: split_blocks 的 section 终止条件, species 段到 reactions: 为止
    zsp_text = "".join(slines[sidx + 1:])
    zsp_rxn_boundary = zsp_text.find("reactions:")
    zsp_only = zsp_text[:zsp_rxn_boundary]
    sp_blocks = [b for b in re.split(r"(?m)^(?=- name: )", zsp_only) if b.strip()]
    keep_sp, drop_sp = [], []
    for b in sp_blocks:
        name = re.match(r"- name:\s*(\S+)", b).group(1)
        (drop_sp if name in ("HCO", "HOCO") else keep_sp).append(b)

    liu_sp_text = "".join(llines[:lidx])  # head + species 定义
    lsidx = liu_sp_text.find("\nspecies:\n")
    liu_sp_only = liu_sp_text[lsidx + len("\nspecies:\n"):]
    liu_blocks = [b for b in re.split(r"(?m)^(?=- name: )", liu_sp_only) if b.strip()]
    add_sp = []
    for b in liu_blocks:
        m = re.match(r"- name:\s*(\S+)", b)
        if m and m.group(1) in ("CH4", "C2H6"):
            add_sp.append(b)

    # ---- 6) 组装输出 (head 只取 phases 部分, species/reactions 单段重建) ----
    spidx = next(i for i, l in enumerate(zlines) if l.startswith("species:"))
    phase_head = "".join(zlines[:spidx])
    phase_head = re.sub(r"species:\s*\[([^\]]*)\]",
                        lambda m: rebuild_species(m.group(1)), phase_head,
                        count=1)
    if "explicit-third-body-duplicates" not in phase_head:
        phase_head = phase_head.replace(
            "  transport: mixture-averaged",
            "  transport: mixture-averaged\n"
            "  explicit-third-body-duplicates: mark-duplicate", 1)
    out_text = (phase_head + "species:\n" + "".join(keep_sp) + "".join(add_sp)
                + "reactions:\n" + new_rxns)
    OUT.write_text(out_text)
    print(f"写出: {OUT}")

    g = ct.Solution(str(OUT))
    print(f"重建结果: {g.n_species} 组分, {g.n_reactions} 反应")
    print("组分:", " ".join(g.species_names))
    return 0


def rebuild_species(old_list):
    names = [t for t in re.split(r"[,\s]+", old_list.strip()) if t and t != "..."]
    names = [n for n in names if n not in ("HCO", "HOCO")]
    for extra in ("CH4", "C2H6"):
        if extra not in names:
            names.append(extra)
    return "species: [" + ", ".join(names) + "]"


def rate_equal(b1, b2):
    """比较两个反应块的 Arrhenius 参数 (含 falloff 的 low/high)"""
    def rates(b):
        a = re.search(r"rate-constant:\s*\{A:\s*([\deE.+-]+),\s*b:\s*([-\deE.+-]+),\s*Ea:\s*([-\deE.+-]+)", b)
        low = re.search(r"low-P-rate-constant:\s*\{A:\s*([\deE.+-]+),\s*b:\s*([-\deE.+-]+),\s*Ea:\s*([-\deE.+-]+)", b)
        return (a.groups() if a else None, low.groups() if low else None)
    r1, r2 = rates(b1), rates(b2)
    if r1[0] is None or r2[0] is None:
        return False
    def close(x, y):
        return abs(np.log(float(x) / float(y))) < 0.02
    def closev(x, y):
        return abs(float(x) - float(y)) < 0.02
    ok = close(r1[0][0], r2[0][0]) and closev(r1[0][1], r2[0][1]) and \
        abs(float(r1[0][2]) - float(r2[0][2])) < 100
    if r1[1] and r2[1]:
        ok = ok and close(r1[1][0], r2[1][0]) and closev(r1[1][1], r2[1][1]) \
            and abs(float(r1[1][2]) - float(r2[1][2])) < 100
    return ok


if __name__ == "__main__":
    sys.exit(main())
