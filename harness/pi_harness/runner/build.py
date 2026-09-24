"""构造真实的 NPUModelRunner 运行状态（object.__new__ + 手工装配必要属性）。

为什么不用真的 `NPUModelRunner.__init__`
----------------------------------------
真 `__init__` 会加载权重、分配 KV cache、初始化 device 属性、注册算子等，
全都需要真机。本 harness 只复现 **prepare_input 路径**，因此手工设置该路径
真正会访问的属性，并用 `AttrAudit` 记录"没被设置却被访问"的属性，
确保没有静默跳过。

所有被跳过的初始化步骤都在 `docs/06-synthetic-load.md` 的已知偏差表里登记。
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import torch

from .. import env
from .config import RunnerConfig
from .modelcfg import PROFILES, make_local_config

logger = logging.getLogger(__name__)

_AUDIT_STORE: dict[str, Counter] = {}


class AttrAudit:
    """记录运行期"从没被装配却被访问"的属性。"""

    @staticmethod
    def store() -> dict[str, Counter]:
        return _AUDIT_STORE

    @staticmethod
    def reset() -> None:
        _AUDIT_STORE.clear()


def _make_audited_subclass(cls):
    """派生一个 __getattr__ 会记账并抛错的子类。

    `getattr(self, "x", default)` 这类调用会静默走默认值 —— 我们记下来，
    因为它们正是"未被复现的代码分支"。
    """
    store = _AUDIT_STORE

    class _Audited(cls):  # type: ignore[misc, valid-type]
        def __getattr__(self, name):
            store.setdefault("missing", Counter())[name] += 1
            raise AttributeError(name)

    _Audited.__name__ = f"Audited{cls.__name__}"
    _Audited.__qualname__ = _Audited.__name__
    return _Audited


def _build_configs(cfg: RunnerConfig):
    from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig

    profile = PROFILES.get(cfg.model_profile, PROFILES["qwen35-0.8b"])
    source_config = cfg.hf_config_path or profile.get("source_config")
    local_dir = make_local_config(
        name=f"{cfg.model_profile}-h{cfg.hidden_size or 'x'}-l{cfg.num_hidden_layers or 'x'}"
        f"-m{cfg.max_model_len or 'x'}",
        source_config=source_config,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        max_position_embeddings=cfg.max_model_len,
    )
    model_kwargs = dict(
        model=local_dir,
        dtype=cfg.dtype,
        skip_tokenizer_init=True,
        enforce_eager=True,
        trust_remote_code=False,
        disable_cascade_attn=True,
    )
    model_config = ModelConfig(**model_kwargs)

    max_model_len = model_config.max_model_len
    cache_config = CacheConfig(
        block_size=cfg.block_size,
        enable_prefix_caching=cfg.enable_prefix_caching,
    )
    parallel_config = ParallelConfig()
    scheduler_config = SchedulerConfig(
        max_model_len=max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
        runner_type="generate",
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        max_num_seqs=cfg.max_num_reqs,
        enable_chunked_prefill=cfg.enable_chunked_prefill,
    )
    # 注意：这里**不**走 `VllmConfig(model_config=...)` 的 __post_init__，
    # 因为那会触发 NPUPlatform.check_and_update_config()（要查 npu-smi / SOC 型号）。
    # 我们只装配 prepare_input 会读到的字段。
    vllm_config = VllmConfig()
    vllm_config.model_config = model_config
    vllm_config.cache_config = cache_config
    vllm_config.parallel_config = parallel_config
    vllm_config.scheduler_config = scheduler_config
    return vllm_config, model_config, cache_config, parallel_config, scheduler_config, local_dir


def _build_kv_cache_config(model_config, cfg: RunnerConfig, max_num_reqs: int):
    """最小但语义正确的 KVCacheConfig（单个 full-attention KV cache group）。

    prepare_input 只用它做两件事：
      * `_may_reorder_batch`: `len(kv_cache_groups) == 0` 判断（attention-free 模型）
      * `_get_block_table` 之后的 attention metadata（属 P1+ 扩展范围）
    本 harness 的 P1 只跑 `_update_states + _prepare_inputs`，所以这里只需要
    一个长度正确、spec 类型正确的 group。
    """
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

    hf = model_config.hf_config
    n_layers = int(getattr(hf, "num_hidden_layers", 1) or 1)
    n_kv_heads = int(
        getattr(hf, "num_key_value_heads", None) or getattr(hf, "num_attention_heads", 1) or 1
    )
    head_dim = int(
        getattr(hf, "head_dim", 0)
        or (int(getattr(hf, "hidden_size", 1024)) // max(int(getattr(hf, "num_attention_heads", 1) or 1), 1))
    )
    layer_names = [f"model.layers.{i}.self_attn" for i in range(n_layers)]
    spec = FullAttentionSpec(
        block_size=cfg.block_size,
        num_kv_heads=n_kv_heads,
        head_size=head_dim,
        dtype=model_config.dtype,
    )
    group = KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)
    max_blocks = max(1, model_config.max_model_len // cfg.block_size)
    return KVCacheConfig(num_blocks=max_blocks, kv_cache_tensors=[], kv_cache_groups=[group])


def _init_ascend_config(vllm_config, model_config, cfg: RunnerConfig) -> None:
    """让 `get_ascend_config()` 可用。

    `_prepare_inputs` 末尾会调用 `lmhead_tp_enable()` → `get_ascend_config()`。
    真机上 `AscendConfig` 由 worker 在 init 时构造；无卡 harness 里我们补这一步。
    它会读 `parallel_config.tensor_parallel_size` / 模型结构来决定若干开关，
    因此这些取值必须与真机启动参数一致（否则会走错分支）。
    """
    from vllm_ascend.ascend_config import init_ascend_config

    # AscendConfig 会读 additional_config（finegrained_tp_config 等）
    if getattr(vllm_config, "additional_config", None) is None:
        try:
            vllm_config.additional_config = {}
        except Exception:
            pass
    init_ascend_config(vllm_config)


def build_runner(cfg: RunnerConfig, audit: bool = True) -> tuple[object, dict]:
    """返回 (runner, meta)。runner 是真实 `NPUModelRunner` 的实例。"""
    env.bootstrap(cfg.device)
    # 必须早于 import_runner_class()：block_table 在 import 期就把 kernel 名字绑进
    # 自己的命名空间，晚了就换不掉了。
    from . import triton_cpu

    triton_cpu.install()
    triton_cpu.set_launch_cost_us(cfg.triton_launch_us)
    triton_cpu.set_mode(cfg.slot_mapping_mode)
    cls = env.import_runner_class()
    InputBatchCls = env.import_input_batch_class()

    from vllm.v1.pool.late_interaction_runner import LateInteractionRunner

    (vllm_config, model_config, cache_config, parallel_config, scheduler_config,
     local_dir) = _build_configs(cfg)

    max_num_tokens = scheduler_config.max_num_batched_tokens
    max_num_reqs = scheduler_config.max_num_seqs
    max_model_len = max(model_config.max_model_len, 1)

    runner_cls = _make_audited_subclass(cls) if audit else cls
    runner = object.__new__(runner_cls)

    device = torch.device(cfg.device)
    dtype = model_config.dtype
    readonly: list[str] = []

    def s(name: str, value):
        """安全赋值：只读 property 会被跳过并记账（跑下去会自己报错，避免静默偏差）。"""
        try:
            setattr(runner, name, value)
        except AttributeError as exc:
            readonly.append(f"{name}: {exc}")

    # ---- 配置对象 ----
    s("vllm_config", vllm_config)
    s("model_config", model_config)
    s("cache_config", cache_config)
    s("parallel_config", parallel_config)
    s("scheduler_config", scheduler_config)
    s("speculative_config", None)
    s("compilation_config", vllm_config.compilation_config)
    s("load_config", vllm_config.load_config)
    s("observability_config", vllm_config.observability_config)
    s("lora_config", None)
    s("device", device)
    s("dtype", dtype)
    s("pin_memory", cfg.pin_memory)

    # ---- 结构/模式开关（全部显式设为"baseline 文本模型、无并行、无 async"）----
    for name, value in (
        ("is_pooling_model", False),
        ("is_multimodal_model", False),
        ("supports_mm_inputs", False),
        ("enable_prompt_embeds", False),
        ("uses_mrope", False),
        ("uses_xdrope_dim", 0),
        ("use_async_scheduling", bool(cfg.async_scheduling)),
        ("use_async_spec_decode", False),
        ("dcp_size", 1),
        ("is_kv_consumer", False),
        ("num_spec_tokens", int(cfg.num_spec_tokens)),
        ("uniform_decode_query_len", 1 + int(cfg.num_spec_tokens)),
        ("is_mm_prefix_lm", False),
        ("routed_experts_initialized", False),
        ("cascade_attn_enabled", False),
        ("dynamic_eplb", False),
        ("calculate_kv_scales", False),
        ("broadcast_pp_output", False),
        ("use_aux_hidden_state_outputs", False),
        ("reorder_batch_threshold", None),
        ("execute_model_state", None),
        ("kv_connector_output", None),
    ):
        s(name, value)

    # ---- 请求状态 ----
    s("requests", {})
    s("num_prompt_logprobs", {})
    s("encoder_cache", {})
    s("late_interaction_runner", LateInteractionRunner())
    s("kv_caches", [])
    s("shared_kv_cache_layers", {})
    s("kv_sharing_fast_prefill_eligible_layers", set())
    s("rswa_window", None)

    # ---- 尺寸 ----
    s("max_num_tokens", max_num_tokens)
    s("max_num_reqs", max_num_reqs)
    s("max_model_len", model_config.max_model_len)
    s("max_encoder_len", 0)
    s("block_size", cfg.block_size)
    s("num_gpu_blocks", 4096)  # 仅用于 max_num_blocks_per_req 计算

    # ---- 持久缓冲区（照抄 GPUModelRunner.__init__ 的相关片段）----
    s("input_ids", _buf(max_num_tokens, torch.int32, device, cfg.pin_memory))
    s("positions", torch.zeros(max_num_tokens, dtype=torch.int64, device=device))
    s("query_start_loc", _buf(max_num_reqs + 2, torch.int32, device, cfg.pin_memory))
    s("seq_lens", torch.zeros(max_num_reqs, dtype=torch.int32, device=device))
    s("optimistic_seq_lens_cpu",
      torch.zeros(max_num_reqs, dtype=torch.int32, pin_memory=cfg.pin_memory))
    s("num_computed_tokens", torch.zeros(max_num_reqs, dtype=torch.int32, device=device))
    s("prev_num_draft_tokens", _buf(max_num_reqs, torch.int32, device, cfg.pin_memory))
    s("req_indices", _buf(max_num_tokens, torch.int64, device, cfg.pin_memory))
    s("prev_positions", _buf(max_num_reqs, torch.int64, device, cfg.pin_memory))
    s("num_scheduled_tokens", _buf(max_num_reqs, torch.int32, device, cfg.pin_memory))
    s("discard_request_mask", _buf(max_num_reqs, torch.bool, device, cfg.pin_memory))
    s("discard_request_indices", _buf(max_num_reqs, torch.int64, device, cfg.pin_memory))
    s("num_decode_draft_tokens", _buf(max_num_reqs, torch.int32, device, cfg.pin_memory))
    s("num_accepted_tokens", _buf(max_num_reqs, torch.int32, device, cfg.pin_memory))
    s("num_draft_tokens", _buf(max_num_reqs, torch.int32, device, cfg.pin_memory))
    s("inputs_embeds", None)
    s("is_token_ids", _buf(max_num_tokens, torch.bool, device, cfg.pin_memory))

    arange_size = max(max_num_reqs + 1, max_num_tokens)
    s("arange_np", np.arange(arange_size, dtype=np.int64))
    s("query_pos", _buf(arange_size, torch.int64, device, cfg.pin_memory))
    s("_arange_scratch", np.empty(arange_size, dtype=np.int64))
    s("_positions_np_buf", np.empty(max_num_tokens, dtype=np.int64))

    # Ascend 特有缓冲区
    s("group_len", _buf(max_num_tokens, torch.int32, device, cfg.pin_memory))
    s("group_key_idx", _buf(max_num_tokens, torch.int32, device, cfg.pin_memory))
    s("group_key_cache_idx", _buf(max_num_tokens, torch.int32, device, cfg.pin_memory))
    s("_has_gdn", bool(cfg.has_gdn))
    s("_has_sinks", False)
    s("attn_state", None)
    s("with_prefill", False)
    s("logits_indices", None)
    s("dcp_manager", None)
    s("_needs_seq_lens_cpu_sync", False)
    s("num_discarded_requests", 0)
    s("num_accepted_tokens_event", None)
    s("valid_sampled_token_count_gpu", None)
    s("valid_sampled_token_count_event", None)
    s("valid_sampled_token_count_cpu", None)
    s("num_rejected_tokens_cpu", None)
    s("num_rejected_tokens_event", None)
    s("num_rejected_tokens_copy_stream", None)
    s("_draft_token_ids", None)
    s("mamba_prev_last_scheduled_idx", None)
    s("sampling_done_event", None)
    s("prepare_inputs_event", None)
    s("long_seq_metadata", None)
    s("kv_cache_config", _build_kv_cache_config(model_config, cfg, max_num_reqs))
    if cfg.has_gdn:
        s("gdn_query_start_loc", _buf(max_num_reqs + 2, torch.int32, device, cfg.pin_memory))

    # ---- Ascend 全局配置（`_prepare_inputs` 末尾的 `lmhead_tp_enable()` 会读它）----
    _init_ascend_config(vllm_config, model_config, cfg)

    # ---- InputBatch（真实 NPUInputBatch）----
    runner.input_batch = InputBatchCls(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_tokens,
        device=device,
        pin_memory=cfg.pin_memory,
        vocab_size=model_config.get_vocab_size(),
        block_sizes=[cfg.block_size],
        kernel_block_sizes=[[cfg.block_size]],
        is_spec_decode=bool(runner.num_spec_tokens),
        logitsprocs=None,
        logitsprocs_need_output_token_ids=False,
        is_pooling_model=False,
        num_speculative_tokens=runner.num_spec_tokens,
        cp_kv_cache_interleave_size=1,
    )

    meta = {
        "model_profile": cfg.model_profile,
        "model_config_dir": local_dir,
        "vocab_size": model_config.get_vocab_size(),
        "max_model_len": model_config.max_model_len,
        "dtype": str(dtype),
        "max_num_tokens": max_num_tokens,
        "max_num_reqs": max_num_reqs,
        "block_size": cfg.block_size,
        "num_spec_tokens": runner.num_spec_tokens,
        "max_num_blocks_per_req": int(runner.input_batch.block_table[0].max_num_blocks_per_req),
        "use_async_scheduling": runner.use_async_scheduling,
        "use_async_spec_decode": runner.use_async_spec_decode,
        "has_gdn": bool(cfg.has_gdn),
        "readonly_attrs_skipped": readonly,
    }
    return runner, meta


def _buf(size: int, dtype: torch.dtype, device: torch.device, pin_memory: bool):
    from vllm.v1.utils import CpuGpuBuffer

    return CpuGpuBuffer(size, dtype=dtype, device=device, pin_memory=pin_memory)
