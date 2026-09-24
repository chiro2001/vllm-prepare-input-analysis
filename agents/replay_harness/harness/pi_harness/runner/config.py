from __future__ import annotations

from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# 预设：必须与真机启动参数一致，否则 CPU 侧负载形状会完全不同。
#
# 真机口径来自 `scripts/launch_phase_service.sh` 的默认值：
#   --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 2048
#   --no-enable-prefix-caching --tensor-parallel-size 1
#   --compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}
#   async scheduling **默认 ON**（vLLM 0.26 这个 build 无条件打开）
#   负载：1 个请求 / prompt 128 tokens / max_tokens 64
#
# 为什么这件事很关键：`max_model_len` 决定 `InputBatch.token_ids_cpu_tensor`
# 的形状 `(max_num_reqs, max_model_len)`，而 `_prepare_inputs` 每步都要
# `torch.index_select(token_ids_cpu_tensor.flatten(), ...)`：
#   * 真机  (8, 2048)   int32 = 64 KB   -> 常驻 L1/L2
#   * 若误用 (64, 262144) int32 = 67 MB  -> 每步从 DRAM 拉，cache 行为完全不同
# 用错预设会直接毁掉 topdown / IPC 的一致性对照。
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    # 真机 launcher 的默认口径（**做一致性对照时用这个**）
    "realmachine": dict(
        max_model_len=2048,
        max_num_reqs=8,
        max_num_batched_tokens=2048,
        block_size=128,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        async_scheduling=True,
        has_gdn=True,
        slot_mapping_mode="noop",
        batch=1,
        isl=128,
        osl=64,
    ),
    # 压力/扫描口径：放大并发与 token 预算，把 prepare_input 的斜率拉出来
    "stress": dict(
        max_model_len=32768,
        max_num_reqs=64,
        max_num_batched_tokens=16384,
        block_size=128,
        enable_prefix_caching=True,
        async_scheduling=False,
        has_gdn=True,
        batch=32,
        isl=2048,
        osl=128,
    ),
    # 模型自身最大值：**仅**用于研究 max_model_len 的影响，不要拿它做真机对照
    "modelmax": dict(
        max_model_len=None,   # None => 用 model config 的 max_position_embeddings
        max_num_reqs=64,
        max_num_batched_tokens=16384,
        block_size=128,
        enable_prefix_caching=True,
        async_scheduling=False,
        has_gdn=True,
    ),
}


@dataclass
class RunnerConfig:
    """无卡 replay 的运行配置（与真机对齐的固定身份 + 负载参数）。"""

    # --- 模型身份（决定 vocab_size / max_model_len / dtype / 结构相关分支） ---
    preset: str = "realmachine"
    model_profile: str = "qwen35-0.8b"       # qwen35-0.8b | qwen35-2b | qwen3-1.7b | synth
    model_path: str = "/models/Qwen3.5-0.8B"
    hf_config_path: str | None = None
    dtype: str = "bfloat16"
    # 结构覆盖（None = 沿用 profile 真实取值）
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None

    # --- 引擎配置（对齐真机启动参数） ---
    max_model_len: int | None = 2048        # None => 取 model config（会 warn）
    max_num_reqs: int = 8
    max_num_batched_tokens: int = 2048
    block_size: int = 128
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    num_spec_tokens: int = 0                # MTP 草稿数；>0 走 spec decode 路径
    async_scheduling: bool = True           # 对齐真机 http server（vLLM 0.26 默认 ON）
    has_gdn: bool = True                    # Qwen3.5-0.8B 含线性注意力层（layer_types 有 linear_attention）
    pin_memory: bool = False                # 无卡容器里 pin_memory 常不可用
    device: str = "cpu"

    # --- 工作负载 ---
    batch: int = 1
    isl: int = 128
    osl: int = 64
    prefix_hit_ratio: float = 0.0
    arrival: str = "simultaneous"           # simultaneous | stair | poisson
    steps: int = 200
    seed: int = 0
    shuffle_values: bool = False            # A/B：打乱 token id / block id 取值

    # --- 计费/注入模型（保真度相关，见 docs/06） ---
    op_cost_us: float = 0.0                 # 每个被 shim 的设备操作注入的经验成本
    triton_launch_us: float = 0.0           # 每次 Ascend Triton launch 的 Python 侧开销
    # 默认 noop：任何"与真机比 topdown/IPC/热点"的用途都必须是 noop
    # （cpu_fallback 的 numpy 兜底会变成假热点，且小 max_model_len 下占比更高）
    slot_mapping_mode: str = "noop"          # cpu_fallback | noop | inject:<us>
    trace_path: str | None = None           # 指定则用 record 出来的 trace 而不是合成

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def apply_preset(self) -> "RunnerConfig":
        """把 `PRESETS[preset]` 套用成默认值（显式传入的字段优先）。"""
        preset = PRESETS.get(self.preset)
        if not preset:
            return self
        defaults = RunnerConfig()
        for k, v in preset.items():
            if getattr(self, k, None) == getattr(defaults, k, None):
                setattr(self, k, v)
        return self

    def validate(self) -> list[str]:
        """返回配置错误列表（空 = 合法）。

        **真机引擎会拒绝的配置，harness 也必须拒绝**，否则会跑到 vLLM 内部
        的隐式假设上并抛出不相关的异常（例如 block table 越界 broadcast error）。

        最重要的一条：`isl + osl <= max_model_len`。一个请求要能产出 `osl` 个
        token，它的序列总长最多到 `isl + osl`，不能超过 `max_model_len`；
        否则它需要的 KV block 数会超过 `max_num_blocks_per_req`
        （= `cdiv(max_model_len, block_size)`），`BlockTable.append_row` 会越界。
        """
        errs: list[str] = []
        trace_mode = bool(self.trace_path)
        mml = self.max_model_len
        if mml is not None:
            if self.isl > mml:
                errs.append(
                    f"isl({self.isl}) > max_model_len({mml})：请求的 prompt 本身就超长"
                )
            elif not trace_mode and self.isl + self.osl > mml:
                errs.append(
                    f"isl+osl = {self.isl + self.osl} > max_model_len({mml})："
                    f"请求无法在 max_model_len 内产出 {self.osl} 个 token。"
                    f"真机引擎会拒绝该请求；请降低 isl/osl 或提高 max_model_len"
                    f"（例如 --max-model-len {self.isl + self.osl}）"
                )
            if not trace_mode:
                max_blocks = -(-mml // max(self.block_size, 1))
                need_blocks = -(-(self.isl + self.osl) // max(self.block_size, 1))
            else:
                max_blocks = need_blocks = 0
            if need_blocks > max_blocks:
                errs.append(
                    f"需要 {need_blocks} 个 KV block > max_num_blocks_per_req {max_blocks}"
                    f"（max_model_len={mml}, block_size={self.block_size}）"
                )
        if not trace_mode and self.batch > self.max_num_reqs:
            errs.append(
                f"batch({self.batch}) > max_num_reqs({self.max_num_reqs})："
                f"并发上限会把请求挡在等待队列里"
            )
        if self.num_spec_tokens and self.isl + self.osl + self.num_spec_tokens > (mml or 0):
            errs.append(
                f"spec_k={self.num_spec_tokens} 会让序列超过 max_model_len({mml})"
            )
        return errs
