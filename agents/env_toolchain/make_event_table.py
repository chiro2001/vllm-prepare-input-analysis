#!/usr/bin/env python3
"""Turn `kperfx events` / `kperfx presets` output into the delivered tables.

Runs on a3-22 inside the project root (PREPARE_INPUT_ROOT overrides the root).
Writes:
  data/toolchain/libkperfx-920b-events.csv
  data/toolchain/libkperfx-920b-presets.json
  data/toolchain/libkperfx-920b-event-table.md
"""

from __future__ import annotations

import json
import os
import re
import subprocess

ROOT = os.environ.get(
    "PREPARE_INPUT_ROOT", os.path.expanduser("~/projects/vllm/prepare-input-phase")
)
KPX = os.path.join(ROOT, "tools", "libkperfx")
OUT = os.path.join(ROOT, "data", "toolchain")


def run(*args):
    return subprocess.run(args, capture_output=True, text=True, cwd=KPX).stdout


def main():
    events_txt = run("./kperfx", "events")
    rows = []
    for line in events_txt.splitlines():
        if line.startswith("#") or line.startswith("NAME") or not line.strip():
            continue
        f = line.split(None, 3)
        if len(f) < 4:
            continue
        rows.append({"name": f[0], "type": f[1], "config": f[2], "aliases": f[3]})

    groups = {}
    for i in range(1, 10):
        t = run("./kperfx", "presets", "920b_topdown_full_%d" % i)
        m = re.search(r"group %d \((\d+) events\): (.+)" % i, t)
        if m:
            groups["920b_topdown_full_%d" % i] = {
                "n_events": int(m.group(1)),
                "events": [e.strip() for e in m.group(2).split(",")],
            }
    pm = re.search(r"issue_width = (\d+)", run("./kperfx", "presets", "920b_topdown_full_1"))
    presets = {
        "920b_topdown_full": {
            "n_groups": 9,
            "issue_width": int(pm.group(1)) if pm else None,
            "per_group_limit": 8,
            "groups": groups,
            "family_aliases": ["950b_topdown_full_*", "k147_topdown_full_*"],
            "small_presets": ["basic", "branch", "memory", "stalls", "ports", "software"],
            "merge_cmd": "kperfx merge --out metrics.txt sweep/split_*.json",
        }
    }

    with open(os.path.join(OUT, "libkperfx-920b-events.csv"), "w") as fh:
        fh.write("name,type,config,aliases\n")
        for r in rows:
            fh.write(
                "%s,%s,%s,\"%s\"\n" % (r["name"], r["type"], r["config"], r["aliases"])
            )
    with open(os.path.join(OUT, "libkperfx-920b-presets.json"), "w") as fh:
        json.dump(presets, fh, indent=2)

    md = [
        "# libkperfx 920B PMU 事件码表与预置组（a3-22 实测）",
        "",
        "生成方式：`kperfx events` + `kperfx presets 920b_topdown_full_{1..9}`，"
        "原始输出见 `data/toolchain/raw/kperfx-events.txt`、"
        "`data/toolchain/raw/preset-920b_topdown_full_*.txt`。",
        "",
        "- 命名事件总数：%d" % len(rows),
        "- 单组硬件上限：**8** 个计数器（`kperfx probe` 实测 1..8 confidence=1.0000，"
        "9/10 为 `not scheduled`）",
        "- 全量 topdown 需要 **9 组**（前 8 组各 8 个硬件计数器 + 第 9 组软件事件）",
        "- 发射宽度 `issue_width = 6`",
        "- `CONFIG` 就是 `perf_event_attr.config` 的原始值，`r<hex>` 可直接传给 `perf`/libkperfx",
        "- 第 9 组是软件事件（`context_switches` / `cpu_migrations` / `page_faults` / `cpu_clock`），"
        "不占 PMU 计数器",
        "",
        "## 预置组",
        "",
        "| 组 | 事件数 | 事件 |",
        "|---|---|---|",
    ]
    for g, v in groups.items():
        ev = ", ".join("`%s`" % e for e in v["events"])
        md.append("| `%s` | %d | %s |" % (g, v["n_events"], ev))
    md += [
        "",
        "## 全部命名事件",
        "",
        "| NAME | TYPE | CONFIG | ALIASES |",
        "|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            "| `%s` | %s | `%s` | %s |" % (r["name"], r["type"], r["config"], r["aliases"])
        )
    with open(os.path.join(OUT, "libkperfx-920b-event-table.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")

    print("events=%d groups=%d -> %s" % (len(rows), len(groups), OUT))


if __name__ == "__main__":
    main()
