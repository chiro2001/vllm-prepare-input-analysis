"""生成本地 HF config.json —— 让 `ModelConfig` 能在**离线/无卡**下构造。

为什么不直接用真模型目录
------------------------
`ModelConfig` 会去 **inspect 模型类**（import vllm.model_executor.models.<arch>）。
Qwen3.5 的类会拉起 flash-linear-attention / triton 等重依赖，无卡容器里不稳。
prepare_input 的 CPU 负载并不依赖模型实现，只依赖：
  * `max_model_len`（决定 token_ids_cpu 的宽度、block table 宽度）
  * `vocab_size`、`dtype`
  * `layer_types`（决定 `_has_gdn`，进而决定 `_prepare_inputs` 里是否维护
    `gdn_query_start_loc`）
因此我们把真模型的 config **摊平**成一份等价的最小 config，
把 `architectures` 换成一个轻量类（默认 `LlamaForCausalLM`），其余数值保持真实取值。

保真度：这条路径**跳过了模型类 import 与 inspect**。它对 prepare_input 的
CPU 负载影响为"中性"（不进 prepare_input 路径），但会让 `is_multimodal_model`
之类的标志与真机不同 —— 已在 docs 的已知偏差表登记。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_TMP_ROOT = Path(os.environ.get("PI_MODEL_TMP", "/tmp/pi_models"))


def _flatten_real_config(src: Path) -> dict:
    raw = json.loads(Path(src).read_text())
    out: dict = {}
    for k, v in raw.items():
        if isinstance(v, (dict, list)):
            continue
        out[k] = v
    text_cfg = raw.get("text_config") or {}
    out.update(text_cfg)
    # 保留 layer_types 供 check_gdn_layer() 判定（真值语义一致）
    if "layer_types" in text_cfg:
        out["layer_types"] = text_cfg["layer_types"]
    return out


def make_local_config(
    *,
    name: str,
    source_config: str | None,
    architectures: str = "LlamaForCausalLM",
    model_type: str = "llama",
    hidden_size: int | None = None,
    num_hidden_layers: int | None = None,
    num_attention_heads: int | None = None,
    num_key_value_heads: int | None = None,
    head_dim: int | None = None,
    vocab_size: int | None = None,
    max_position_embeddings: int | None = None,
) -> str:
    """写出 `<tmp>/<name>/config.json`，返回目录路径。"""
    if source_config and Path(source_config).exists():
        cfg = _flatten_real_config(Path(source_config))
    else:
        cfg = {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 3584,
            "vocab_size": 151936,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "tie_word_embeddings": True,
            "rope_theta": 1000000.0,
        }

    for key, val in (
        ("hidden_size", hidden_size),
        ("num_hidden_layers", num_hidden_layers),
        ("num_attention_heads", num_attention_heads),
        ("num_key_value_heads", num_key_value_heads),
        ("head_dim", head_dim),
        ("vocab_size", vocab_size),
        ("max_position_embeddings", max_position_embeddings),
    ):
        if val is not None:
            cfg[key] = val
    if "num_key_value_heads" not in cfg:
        cfg["num_key_value_heads"] = cfg.get("num_attention_heads", 8)
    if "head_dim" not in cfg:
        cfg["head_dim"] = cfg.get("hidden_size", 1024) // max(cfg.get("num_attention_heads", 8), 1)
    cfg["architectures"] = [architectures]
    cfg["model_type"] = model_type
    # 清掉会触发多模态/其它机器分支的字段
    for k in ("vision_config", "image_token_id", "video_token_id",
              "vision_start_token_id", "vision_end_token_id", "quantization_config"):
        cfg.pop(k, None)

    outdir = _TMP_ROOT / name
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))
    return str(outdir)


PROFILES = {
    # 真机首选模型：Qwen3.5-0.8B（BF16, TP1）
    "qwen35-0.8b": dict(source_config="/models/Qwen3.5-0.8B/config.json"),
    "qwen35-2b": dict(source_config="/models/Qwen3.5-2B/config.json"),
    "qwen3-1.7b": dict(source_config="/models/Qwen3-1.7B/config.json"),
    "synth": dict(source_config=None),
}
