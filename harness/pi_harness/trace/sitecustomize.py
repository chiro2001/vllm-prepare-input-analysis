"""Zero-touch installer for the capture hook.

Put this directory on ``PYTHONPATH`` and every interpreter (including the vLLM
worker process spawned by the engine) executes this file at start-up through
``site``.  It does nothing unless ``PI_CAPTURE`` is set, and it swallows its own
errors so a broken capture can never take the service down.

    PYTHONPATH=/work/harness:/work/harness/pi_harness/trace \\
    PI_CAPTURE=/work/data/harness/trace_real.jsonl \\
    python -m vllm.entrypoints.openai.api_server ...
"""

from __future__ import annotations

import os
import sys


def _bootstrap() -> None:
    if not os.environ.get("PI_CAPTURE"):
        return
    here = os.path.dirname(os.path.abspath(__file__))
    harness_root = os.path.dirname(os.path.dirname(here))  # .../harness
    if harness_root not in sys.path:
        sys.path.insert(0, harness_root)
    try:
        from pi_harness.trace.capture import maybe_install
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[pi_harness.sitecustomize] import failed: {exc!r}", flush=True)
        return
    cap = maybe_install()
    if cap is not None:
        print(
            f"[pi_harness.sitecustomize] capture -> {cap.path} "
            f"(target={cap.target}, max_steps={cap.stats.max_steps or 'inf'}, "
            f"flush_every={cap.stats.flush_every})",
            flush=True,
        )


try:
    _bootstrap()
except Exception as _exc:  # pragma: no cover
    print(f"[pi_harness.sitecustomize] bootstrap failed: {_exc!r}", flush=True)
