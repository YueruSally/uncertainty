"""Paired differences in percentage-point gains over Base for the single-member panel."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from hv_joint_b5000_union_v1.recompute import BUDGETS, equal_mean, interval


DEFAULT = (Path(__file__).resolve().parent / "transformer_ppo_runs/diagnostics"
           / "ccp_icm_single_member_panel_b5000_v1/final/PER_PAIR.csv")
PPO, SHARED, CROSS = "ICM_TERMINAL_PPO", "ICM_SHARED_SUPERVISED", "C_LEARN"
METHODS = (PPO, SHARED, CROSS)
# Values supplied in the 2026-11-16 display table. Used only for a provenance check.
DISPLAY = {400: (23.36, 23.05, 17.39), 1000: (21.58, 21.58, 12.71),
           5000: (7.92, 7.82, 8.30)}


def read_panel(path):
    panel = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            method, budget = row["method"], int(row["evaluation"])
            if row["reference_method"] != "BASE" or method not in METHODS or budget not in BUDGETS:
                continue
            key = (budget, row["configuration"], row["instance"], int(row["seed"]))
            if method in panel[key]:
                raise ValueError(f"重复的配对记录：{key} / {method}")
            own, base = float(row["own"]), float(row["reference"])
            if not math.isfinite(own) or not math.isfinite(base) or base <= 0:
                raise ValueError(f"无法计算相对 Base 的提升：{key} / {method}")
            panel[key][method] = (own, base)
    return panel


def report(panel):
    configuration_names = {f"S{i}" for i in range(9)}
    first_keys = None
    for budget in BUDGETS:
        rows = []
        groups = defaultdict(list)
        for (node, cfg, instance, seed), measures in sorted(panel.items()):
            if node != budget:
                continue
            if set(measures) != set(METHODS) or len({v[1] for v in measures.values()}) != 1:
                raise ValueError(f"三种方法缺少相同的配对或 Base 不一致：{budget}/{instance}/{seed}")
            row = dict(configuration=cfg, instance=instance, seed=seed)
            for method, (own, base) in measures.items():
                row[method] = 100 * (own - base) / base
            rows.append(row)
            groups[cfg].append((instance, seed))
        if (set(groups) != configuration_names
                or any(len(keys) != 3 or len({name for name, _ in keys}) != 1
                       for keys in groups.values())):
            raise ValueError(f"{budget} 次评价未形成 S0–S8 各 1 个实例 × 3 个配对种子")
        if first_keys is None:
            first_keys = groups
        elif groups != first_keys:
            raise ValueError("三个评价节点的实例或种子不一致")

        means = [equal_mean(rows, method) for method in METHODS]
        print(f"\n{budget} 次 | PPO / 共享监督 / C Transformer 相对 Base："
              + " / ".join(f"{x:+.2f}%" for x in means))
        if any(abs(a - b) > 0.005001 for a, b in zip(means, DISPLAY[budget])):
            print("来源提示：逐次明细未复现指定截图的均值。下列区间仅对应本文件。")
        for label, a, b in (("PPO − 共享监督", PPO, SHARED), ("共享监督 − C Transformer", SHARED, CROSS)):
            for row in rows:
                row["delta"] = row[a] - row[b]
            lo, hi = interval(rows, "delta", 3000)
            direction = "区间包含零" if lo <= 0 <= hi else ("前者较高" if lo > 0 else "前者较低")
            print(f"{label} | {equal_mean(rows, 'delta'):+.4f} 个百分点 | "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}] | {direction}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("per_pair", nargs="?", type=Path, default=DEFAULT)
    path = parser.parse_args().per_pair
    print(f"正在读取 {path.resolve()}", flush=True)
    report(read_panel(path))
