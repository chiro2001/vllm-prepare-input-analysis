#!/usr/bin/env python3
"""用 libkperfx 在宿主侧对一个外部进程做 PMU 计数（target=process）。

为什么这样设计：

* 容器里不挂 NPU、没有 perf、也没有 root；实测容器默认 seccomp 会直接拒绝
  perf_event_open（kperfx probe 报 open failed），所以 PMU 一律在宿主 sidecar 里采。
* 宿主 root 侧 libkperfx 以 target=process, pid=<宿主 PID> 挂到容器进程，
  内核态计数保留，且不改变容器运行配置。
* 结果必带 time_enabled / time_running / confidence；组内事件数 <= 硬件上限
  （920B = 8）时 confidence 恒为 1.0。

典型用法（宿主 root）：

    # 测到目标进程退出（配合 STOP 门控：--ready-file 就绪后再 SIGCONT）
    sudo python3 -m pi_harness.profiling.pmu count --pid 12345 --wait-exit \
         --ready-file /tmp/x.ready --events cycles,inst_retired,inst_spec --out pmu/count.json
    # 固定窗口
    sudo ... pmu count --pid 12345 --duration 10 --preset 920b_topdown_full_1 --out x.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from . import CALIBER_VERSION, ensure_kperfx_importable
from .envcap import now_ts, write_json

# 920B 命名事件表里没有 br_retired；分支用 br_pred(=0x12)/br_mis_pred(=0x10)。
# 默认 6 个事件，留出余量避免组内 multiplex（920B 单组上限 8）。
DEFAULT_EVENTS = "cpu_cycles,inst_retired,inst_spec,br_pred,br_mis_pred,l1d_cache_refill"


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as fh:
            data = fh.read()
        state = data.rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except OSError:
        return False


def count_window(pid: int, *,
                 events: str | None = None,
                 preset: str | None = None,
                 duration: float | None = None,
                 wait_exit: bool = False,
                 timeout: float | None = None,
                 ready_file: str | None = None,
                 continue_pid: bool = False,
                 multiplex: bool = False,
                 out: str | None = None,
                 sample_interval: float = 0.02) -> dict[str, Any]:
    """对 pid 做一次计数窗口，返回结构化结果。"""
    ensure_kperfx_importable()
    import kperfx  # type: ignore

    t0 = time.time()
    kwargs: dict[str, Any] = {
        "target": "process",
        "pid": pid,
        "format": "json",
        "output": "stdout",
        "quiet": True,
    }
    if events:
        kwargs["events"] = events
    if preset:
        kwargs["preset"] = preset
    if multiplex:
        kwargs["multiplex"] = True

    result: dict[str, Any] = {
        "caliber_version": CALIBER_VERSION,
        "ts_start": now_ts(),
        "pid": pid,
        "events_spec": events,
        "preset": preset,
        "multiplex_allowed": multiplex,
    }
    sess = kperfx.Session(**kwargs)
    try:
        sess.start()
        result["ts_running"] = now_ts()
        if ready_file:
            Path(ready_file).parent.mkdir(parents=True, exist_ok=True)
            Path(ready_file).write_text("ready\n", encoding="utf-8")
        if continue_pid:
            os.kill(pid, signal.SIGCONT)
        if wait_exit:
            deadline = t0 + timeout if timeout else None
            reason = "target_exit"
            while pid_alive(pid):
                if deadline and time.time() > deadline:
                    reason = "timeout"
                    break
                time.sleep(sample_interval)
            result["stopped_reason"] = reason
        else:
            dur = duration if duration is not None else 10.0
            deadline = time.time() + dur
            reason = "duration"
            while time.time() < deadline:
                if not pid_alive(pid):
                    reason = "target_exit_early"
                    break
                time.sleep(min(sample_interval, max(0.0, deadline - time.time())))
            result["stopped_reason"] = reason
        sess.stop()
    finally:
        sess.close()

    result.update({
        "ts_end": now_ts(),
        "wall_s": round(time.time() - t0, 6),
        "counters": sess.results,
        "entries": sess.entries,
        "time_enabled": int(sess.time_enabled),
        "time_running": int(sess.time_running),
        "confidence": float(sess.confidence),
        "useronly_forced": bool(getattr(sess.result, "useronly_forced", 0)),
        "instances": int(sess.instances),
        "pmu": kperfx.pmu_name(),
        "backend": kperfx.backend_name(),
    })
    c = {str(k).upper(): float(v) for k, v in result["counters"].items()}
    cyc = c.get("CPU_CYCLES")
    ins = c.get("INST_RETIRED")
    result["derived"] = {
        "cpu_cycles": cyc,
        "inst_retired": ins,
        "ipc": (ins / cyc) if (cyc and ins) else None,
        "cycles_per_sec": (cyc / result["wall_s"]) if (cyc and result["wall_s"] > 0) else None,
    }
    if out:
        write_json(out, result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="libkperfx 外部进程计数窗口（宿主 root 运行）")
    ap.add_argument("sub", nargs="?", default="count", help="子命令，目前只有 count")
    ap.add_argument("--pid", type=int, required=True, help="宿主侧目标 PID")
    ap.add_argument("--events", default=None, help=f"事件列表（默认 {DEFAULT_EVENTS}）")
    ap.add_argument("--preset", default=None, help="libkperfx 预置组名")
    ap.add_argument("--duration", type=float, default=None, help="测量窗口秒数")
    ap.add_argument("--wait-exit", action="store_true", help="测到目标进程退出")
    ap.add_argument("--timeout", type=float, default=None, help="wait-exit 的超时上限")
    ap.add_argument("--ready-file", default=None, help="会话开始后写出的就绪标记")
    ap.add_argument("--continue", dest="continue_pid", action="store_true",
                    help="就绪后给目标发 SIGCONT（配合 STOP 门控）")
    ap.add_argument("--multiplex", action="store_true", help="事件数超上限时多组同跑")
    ap.add_argument("--out", default=None, help="输出 JSON")
    args = ap.parse_args()

    if not pid_alive(args.pid):
        print(f"pid {args.pid} 不存在或已退出", file=sys.stderr)
        return 2
    if not args.events and not args.preset:
        args.events = DEFAULT_EVENTS
    if args.duration is None and not args.wait_exit:
        args.duration = 10.0
    res = count_window(args.pid, events=args.events, preset=args.preset,
                       duration=args.duration, wait_exit=args.wait_exit,
                       timeout=args.timeout, ready_file=args.ready_file,
                       continue_pid=args.continue_pid, multiplex=args.multiplex,
                       out=args.out)
    print(json.dumps({"counters": res["counters"], "confidence": res["confidence"],
                      "ipc": res["derived"]["ipc"],
                      "stopped_reason": res.get("stopped_reason")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
