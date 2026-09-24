"""无卡 harness 的环境引导。

顺序很关键（否则会撞上循环导入）::

    install shim            # 替换 torch.npu.* 的设备操作
    import vllm             # 触发 vllm_ascend platform plugin，先完成 vllm_ascend.ops
    打桩 distributed group  # get_pp_group/get_dcp_group/... 在无进程组时是 assert
    import vllm_ascend.worker.model_runner_v1

实测证据：直接 `import vllm_ascend.worker.model_runner_v1` 会因
`vllm_ascend.device.device_op` ↔ `vllm_ascend.ops.fused_moe` 的循环导入失败。
"""

from __future__ import annotations

import os
import torch

from . import shim as _shim

_BOOTSTRAPPED = False
_state_errors: list[str] = []


class FakeGroupCoordinator:
    """world_size=1 的通信组替身。

    prepare_input 在 TP=1/PP=1/DP=1 下不做任何集合通信，真机上这些 group 也只是
    提供 `is_last_rank` / `world_size` / `rank_in_group` 之类的标量查询，因此替身
    对 CPU 侧负载的影响是中性的（不增减算力开销）。凡是**真机 P>1 会触发的集合通信**，
    本 harness 都视为"未复现"，需要在文档里登记。
    """

    def __init__(self, rank: int = 0):
        self.rank = rank
        self.world_size = 1
        self.rank_in_group = rank
        self.local_rank = rank
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == 0

    def _collective(self, *args, **kwargs):  # pragma: no cover - 不应被调用
        raise RuntimeError(
            "FakeGroupCoordinator: collective ops are not reproduced in the no-card harness"
        )

    all_reduce = all_gather = reduce_scatter = broadcast = send = recv = _collective
    barrier = _collective
    broadcast_tensor_dict = _collective
    all_gather_object = _collective

    def destroy(self) -> None:
        return None


def _patch_distributed() -> None:
    import vllm.distributed as dist
    import vllm.distributed.parallel_state as ps

    group = FakeGroupCoordinator()
    for name in (
        "get_pp_group",
        "get_tp_group",
        "get_dp_group",
        "get_dcp_group",
        "get_pcp_group",
        "get_ep_group",
        "get_eplb_group",
    ):
        fn = lambda name=name: group  # noqa: E731
        if hasattr(ps, name):
            setattr(ps, name, fn)
        if hasattr(dist, name):
            setattr(dist, name, fn)

    # 部分模块直接从 parallel_state 里 import
    for mod_name in (
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker.utils",
        "vllm.v1.outputs",
        "vllm.sequence",
        "vllm_ascend.worker.model_runner_v1",
        "vllm_ascend.worker.block_table",
        "vllm_ascend.utils",
    ):
        mod = __import__(mod_name, fromlist=["*"]) if mod_name in __import__("sys").modules else None
        if mod is None:
            continue
        for name in ("get_pp_group", "get_tp_group", "get_dp_group", "get_dcp_group",
                     "get_pcp_group", "get_ep_group", "get_eplb_group"):
            if hasattr(mod, name):
                setattr(mod, name, lambda: group)

    # vllm_ascend 自己的 dcp helper
    try:
        import vllm_ascend.distributed.utils as au

        au.get_decode_context_model_parallel_world_size = lambda: 1
        au.get_decode_context_model_parallel_rank = lambda: 0
    except Exception:  # pragma: no cover
        pass


def bootstrap(device: str = "cpu") -> None:
    """幂等引导；返回后即可 import `vllm_ascend.worker.model_runner_v1`。"""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    _shim.install(device)

    import vllm  # noqa: F401  必须早于 worker 模块，避免 vllm_ascend 内部循环导入

    # 关键：vllm_ascend 内部存在 import 顺序敏感的循环导入
    #   worker.model_runner_v1 -> attention.attention_v1 -> device.device_op
    #     -> ops.triton.fla.* (触发 ops/__init__) -> ops.fused_moe.fused_moe
    #     -> ops.fused_moe.experts_selector -> `from device.device_op import DeviceOperator`
    #     -> device_op 尚在初始化中 => ImportError
    # 先完整 import `vllm_ascend.ops` 即可打破该环（真机上由 engine 的 import 顺序隐式满足）。
    try:
        import vllm_ascend.ops  # noqa: F401
    except Exception as exc:  # pragma: no cover
        _state_errors.append(f"pre-import vllm_ascend.ops failed: {exc!r}")

    _patch_distributed()

    # 兜底：CPU 上 pin_memory 在部分内核/容器组合下不可用
    try:
        t = torch.empty(8, dtype=torch.int32, pin_memory=True)
        del t
    except Exception:
        import vllm.utils.torch_utils as tu

        tu.PIN_MEMORY = False
        import vllm.v1.utils as vu

        vu.PIN_MEMORY = False
        os.environ.setdefault("PI_SHIM_PIN_MEMORY_FALLBACK", "1")

    _BOOTSTRAPPED = True


def import_runner_class():
    """返回真实的 `NPUModelRunner` 类。"""
    bootstrap()
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    return NPUModelRunner


def import_input_batch_class():
    bootstrap()
    from vllm_ascend.worker.npu_input_batch import NPUInputBatch

    return NPUInputBatch
