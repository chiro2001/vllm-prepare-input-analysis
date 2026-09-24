# `prepare input` scope 边界（行号级证据）

> 生成时间：2026-09-23T18:12:28+0000　|　脚本：`harness/probes/scope_prepare_input.py`（sha256 `e4d70e6d28d34fd4…`）

> 文件：`/work/refs/vllm-ascend/vllm_ascend/worker/model_runner_v1.py`（sha256 `4699af75ac8b4b55283d55190365e93d13820a05888bfa65aea8746deabc4d3a`）

> 宿主函数：`execute_model()`

## 1. 起止行

- **起点** `L1859`：`with record_function_or_nullcontext("prepare input"):`　（父函数 `execute_model`）
- **终点** `L2098`：scope 的**最后一条语句**，同时也是整个 `with` 块的末行
- 证明（末尾三行的原文）：

```python
 2097:             # update global cos, sin
 2098:             update_cos_sin(positions)  # <-- scope 末行
 2099: 
 2100:         if self.dynamic_eplb:  # <-- scope 之外（同级语句）
 2101:             self.eplb_updator.forward_before()
```

> 判据：`L2098` 之后的同级语句缩进回到 `execute_model` 的 8 空格层（`L2100 if self.dynamic_eplb:`），因此 `with` 块在 `L2098` 结束，`dynamic_eplb` / `forward` / `post process` / `sample_token` 都在 scope **之外**。

## 2. scope 顶层语句（3 条）

| # | 行 | 语句 | 说明 |
|---|---|---|---|
| 1 | L1860 | `with self.synchronize_input_prep():` | **scope 的第一个语句**：async scheduling 下这里会 `prepare_inputs_event.synchronize()` 真等（真机同步点） |
| 2 | L2084 | `input_ids, inputs_embeds, positions, intermediate_tensors, model_kwargs, ec_connector_output = self._preproces` |  |
| 3 | L2098 | `update_cos_sin(positions)` | **scope 的最后一条语句**：写全局 rope cache（真机上是 3 个设备 kernel） |

## 3. scope 内调用的全部函数（按调用点去重）

| 调用点行 | callee | 定义位置 | 设备交互 |
|---|---|---|---|
| L27 | `prev_positions.clamp` | `—` | — |
| L34 | `valid_counts.int` | `—` | — |
| L37 | `torch.where` | `—` | — |
| L38 | `num_accepted_tokens.copy_` | `—` | — |
| L46 | `isinstance` | `—` | — |
| L47 | `self.passes.append` | `utils.py:52` | — |
| L56 | `TypeError` | `—` | — |
| L107 | `self._x.copy` | `utils.py:106` | — |
| L141 | `self.gpu.copy_` | `—` | non_blocking=True |
| L142 | `copy_` | `—` | non_blocking=True |
| L153 | `positions.size` | `—` | — |
| L155 | `_cos_sin_cache.index_select()()()` | `—` | — |
| L155 | `_cos_sin_cache.index_select()()` | `—` | — |
| L155 | `_cos_sin_cache.index_select()` | `—` | — |
| L155 | `_cos_sin_cache.index_select` | `—` | — |
| L170 | `self.runner._sync_metadata_across_dp` | `model_runner_v1.py:702` | — |
| L172 | `self.block_table.gpu.stride` | `—` | — |
| L173 | `get_ascend_device_type` | `—` | — |
| L182 | `get_metadata_builder` | `—` | — |
| L185 | `self.block_table.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L186 | `get_ascend_config` | `—` | — |
| L197 | `get_device_tensor` | `block_table.py:221` | — |
| L202 | `builder.build_for_graph_capture` | `attention_v1.py:397` | — |
| L213 | `multi_steps_attn_metadata.append` | `utils.py:52` | — |
| L215 | `self.token_indices_to_sample.fill_` | `—` | — |
| L230 | `self.model.precompute_and_store_context_kv` | `—` | — |
| L231 | `self.model` | `—` | — |
| L233 | `self._get_positions` | `gpu_model_runner.py:1023` | — |
| L239 | `self._runnable` | `—` | — |
| L251 | `np.asarray` | `—` | numpy |
| L251 | `self._update_full_graph_params` | `llm_base_proposer.py:2297` | — |
| L253 | `adler32` | `—` | — |
| L253 | `req_id.encode` | `—` | — |
| L259 | `offload_req_ids_tensor.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L262 | `np.diff()` | `—` | numpy |
| L262 | `np.diff` | `—` | numpy |
| L262 | `query_start_loc_cpu.numpy` | `—` | numpy |
| L263 | `np.arange` | `—` | numpy |
| L265 | `RuntimeError` | `—` | — |
| L272 | `offload_token_to_req.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L287 | `_slice_reqs` | `utils.py:280` | — |
| L299 | `self._stop_event.set` | `—` | — |
| L300 | `self.is_alive` | `utils.py:440` | — |
| L300 | `threading.current_thread` | `—` | — |
| L301 | `self._split_decodes_and_prefills` | `attention_v1.py:264` | — |
| L301 | `self.join` | `utils.py:446` | — |
| L303 | `logger.warning` | `—` | — |
| L320 | `common_attn_metadata.slot_mapping.to` | `—` | — |
| L327 | `self.attn_mask_builder.get_attention_mask` | `attention_mask.py:68` | — |
| L330 | `query_start_loc_cpu.pin_memory()` | `—` | .to(device=...),non_blocking=True,pin_memory |
| L330 | `query_start_loc_cpu.pin_memory` | `—` | pin_memory |
| L332 | `tolist` | `—` | — |
| L333 | `seq_lens.tolist` | `—` | — |
| L358 | `torch.cat` | `—` | — |
| L358 | `seq_lens.new_ones` | `—` | — |
| L363 | `block_table.new_zeros` | `—` | — |
| L368 | `self._build_backend_metadata` | `attention_v1.py:273` | — |
| L376 | `self.metadata_cls` | `—` | — |
| L718 | `should_skip_allreduce_across_dp_group` | `—` | — |
| L722 | `torch.zeros` | `—` | — |
| L725 | `dist.all_reduce` | `pyhccl.py:130` | — |
| L725 | `get_dp_group` | `—` | — |
| L729 | `num_tokens_across_dp.max()` | `—` | — |
| L729 | `num_tokens_across_dp.max` | `—` | — |
| L730 | `CUDAGraphMode` | `—` | — |
| L730 | `_post_process_cudagraph_mode` | `model_runner_v1.py:5060` | — |
| L738 | `num_tokens_across_dp.cpu` | `—` | — |
| L755 | `get_pp_group` | `—` | — |
| L759 | `self._is_pd_prefill_worker` | `model_runner_v1.py:748` | — |
| L775 | `np.nonzero` | `—` | numpy |
| L778 | `set` | `—` | — |
| L795 | `prev_req_indices.get` | `backend.py:55` | — |
| L801 | `req_state.output_token_ids.append` | `utils.py:52` | — |
| L810 | `torch.tensor()` | `—` | — |
| L810 | `torch.tensor` | `—` | — |
| L818 | `normalize_block_ids_by_group` | `config_data.py:627` | — |
| L820 | `self.allocated_block_ids_by_group.extend` | `utils.py:55` | — |
| L822 | `enumerate` | `—` | — |
| L824 | `self.update_mamba_spec_blocks` | `config_data.py:827` | — |
| L825 | `extend` | `utils.py:55` | — |
| L831 | `self._apply_pp_sampled_tokens_from_scheduler_output` | `model_runner_v1.py:751` | — |
| L832 | `super()` | `—` | — |
| L832 | `super` | `—` | — |
| L879 | `query_start_loc.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L906 | `self.input_batch.block_table.commit_block_table` | `block_table.py:184` | — |
| L917 | `scheduler_output.scheduled_spec_decode_tokens.get` | `backend.py:55` | — |
| L922 | `self._build_attn_state` | `model_runner_v1.py:1320` | — |
| L929 | `self._get_cumsum_and_arange` | `gpu_model_runner.py:1720` | numpy |
| L940 | `self.dcp_manager.init_batch_info` | `—` | — |
| L949 | `self._compute_prev_positions` | `gpu_model_runner.py:1746` | — |
| L956 | `self.prev_positions.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L960 | `self.dcp_manager.generate_dcp_mtp_input` | `—` | — |
| L975 | `torch.from_numpy` | `—` | numpy |
| L987 | `torch.index_select` | `—` | — |
| L988 | `self.input_batch.token_ids_cpu_tensor.flatten` | `—` | — |
| L994 | `self.input_batch.is_token_ids_tensor.flatten` | `—` | — |
| L1004 | `range` | `—` | — |
| L1027 | `min` | `—` | — |
| L1039 | `self.query_start_loc.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L1048 | `fill` | `—` | numpy |
| L1049 | `self.gdn_query_start_loc.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L1053 | `mamba_utils.MambaBuffers.create` | `—` | — |
| L1056 | `torch.add` | `graph_fusion_pass_manager.py:45` | numpy |
| L1061 | `fill_` | `—` | — |
| L1067 | `self._prepare_input_ids` | `gpu_model_runner.py:1761` | — |
| L1072 | `self._calc_mrope_positions` | `gpu_model_runner.py:2725` | — |
| L1072 | `self.input_batch.get_pooling_params` | `gpu_input_batch.py:939` | — |
| L1073 | `self.mrope_positions.gpu.copy_` | `—` | non_blocking=True |
| L1078 | `self._calc_xdrope_positions` | `gpu_model_runner.py:2774` | — |
| L1078 | `param.extra_kwargs.get` | `backend.py:55` | — |
| L1091 | `numpy` | `—` | numpy |
| L1093 | `token_type_id_requests.get` | `backend.py:55` | — |
| L1094 | `torch.arange` | `—` | — |
| L1095 | `token_type_ids.append` | `utils.py:52` | — |
| L1096 | `self.discard_request_indices.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L1097 | `torch.empty` | `sequence_parallelism.py:52` | pin_memory |
| L1099 | `self.discard_request_mask.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L1101 | `token_type_ids_cpu.to` | `—` | .to(device=...),non_blocking=True |
| L1104 | `self.num_accepted_tokens_event.synchronize` | `—` | synchronize |
| L1111 | `np.where` | `—` | numpy |
| L1126 | `self.num_accepted_tokens.np.fill` | `—` | numpy |
| L1127 | `self.num_accepted_tokens.gpu.fill_` | `—` | — |
| L1135 | `to` | `—` | .to(device=...),non_blocking=True |
| L1145 | `self.prev_num_draft_tokens.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L1146 | `update_num_computed_tokens_for_batch_change` | `utils.py:13` | — |
| L1161 | `self.req_indices.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L1164 | `self.query_pos.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L1166 | `self.num_scheduled_tokens.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L1169 | `getattr` | `—` | — |
| L1171 | `dcp_manager.rebuild_async_spec_decode_inputs` | `—` | numpy |
| L1178 | `bool` | `—` | — |
| L1204 | `DCPAsyncSpecDecodeRebuildResult` | `—` | — |
| L1216 | `torch.from_numpy()` | `—` | .to(device=...),numpy |
| L1244 | `self._correct_optimistic_seq_lens_cpu` | `model_runner_v1.py:1432` | — |
| L1246 | `self.input_batch.block_table.compute_slot_mapping` | `block_table.py:153` | — |
| L1268 | `np.ones` | `—` | numpy |
| L1274 | `np.zeros` | `—` | numpy |
| L1278 | `np.full` | `—` | numpy |
| L1282 | `scheduler_output.scheduled_spec_decode_tokens.items` | `—` | — |
| L1293 | `self._calc_spec_decode_metadata` | `model_runner_v1.py:1366` | — |
| L1303 | `self.num_decode_draft_tokens.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L1308 | `np.sum` | `—` | numpy |
| L1309 | `self.set_active_loras` | `—` | — |
| L1310 | `lmhead_tp_enable` | `—` | — |
| L1312 | `nn.functional.pad` | `—` | — |
| L1321 | `np.all` | `—` | numpy |
| L1357 | `any` | `—` | — |
| L1359 | `scheduled_spec_tokens.values` | `—` | — |
| L1364 | `input_ids.masked_fill_` | `—` | — |
| L1412 | `torch.from_numpy()()` | `—` | .to(device=...),non_blocking=True,pin_memory,numpy |
| L1422 | `SpecDecodeMetadata` | `—` | — |
| L1424 | `num_draft_tokens.tolist` | `—` | — |
| L1451 | `self.valid_sampled_token_count_event.synchronize` | `—` | synchronize |
| L1452 | `correct_optimistic_seq_lens_cpu` | `utils.py:41` | numpy |
| L1453 | `self.optimistic_seq_lens_cpu.numpy` | `—` | numpy |
| L1456 | `self.valid_sampled_token_count_cpu.numpy` | `—` | numpy |
| L1703 | `mm_kwargs_combined.update` | `config_data.py:812` | — |
| L1733 | `np.cumsum` | `—` | numpy |
| L1738 | `np.subtract` | `—` | numpy |
| L1755 | `prev_positions.fill` | `—` | — |
| L1759 | `prev_req_id_to_index.get` | `backend.py:55` | — |
| L1780 | `self.input_ids.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L1782 | `self.inputs_embeds.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L1783 | `self.is_token_ids.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L1803 | `prev_indices.append` | `utils.py:52` | — |
| L1807 | `scheduled_spec_tokens.get` | `backend.py:55` | — |
| L1809 | `item` | `—` | — |
| L1813 | `sample_flattened_indices.append` | `utils.py:52` | — |
| L1814 | `spec_flattened_indices.extend` | `utils.py:55` | — |
| L1824 | `prev_draft_token_indices.extend` | `utils.py:55` | — |
| L1826 | `max` | `—` | — |
| L1860 | `self.synchronize_input_prep` | `gpu_model_runner.py:3810` | synchronize,synchronize_input_prep |
| L1862 | `self.input_ids.gpu.scatter_` | `—` | — |
| L1875 | `self.requests.get` | `backend.py:55` | — |
| L1881 | `self._update_states` | `model_runner_v1.py:816` | — |
| L1884 | `self._draft_token_ids.to` | `—` | — |
| L1885 | `has_ec_transfer` | `—` | — |
| L1885 | `get_ec_transfer` | `—` | — |
| L1887 | `self.maybe_get_ec_connector_output` | `—` | — |
| L1889 | `draft_token_ids.flatten` | `—` | — |
| L1891 | `self._execute_mm_encoder` | `gpu_model_runner.py:2963` | — |
| L1892 | `self._finalize_dump_data` | `model_runner_v1.py:3753` | — |
| L1893 | `make_empty_encoder_model_runner_output` | `—` | — |
| L1906 | `self._dummy_run` | `model_runner_v1.py:3241` | — |
| L1907 | `has_kv_transfer_group` | `—` | — |
| L1910 | `self.kv_connector_no_forward` | `—` | — |
| L1922 | `sum` | `—` | — |
| L1926 | `self._start_dump_data` | `model_runner_v1.py:3747` | — |
| L1927 | `np.array` | `—` | numpy |
| L1928 | `int` | `—` | numpy |
| L1928 | `num_scheduled_tokens_np.max` | `—` | numpy |
| L1931 | `self.encoder_seq_lens.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L1933 | `self._prepare_inputs` | `model_runner_v1.py:883` | — |
| L1943 | `self._compute_cascade_attn_prefix_lens` | `gpu_model_runner.py:2590` | — |
| L1955 | `self._determine_batch_execution_and_padding` | `model_runner_v1.py:2776` | — |
| L1965 | `logger.isEnabledFor` | `—` | — |
| L1966 | `logger.debug` | `—` | — |
| L1977 | `maybe_create_ubatch_slices` | `—` | — |
| L1986 | `self.update_eplb_heat_collection_status` | `model_runner_v1.py:3601` | — |
| L1999 | `deferred_state_corrections_fn` | `—` | — |
| L2001 | `self._get_mamba_bufs` | `gpu_model_runner.py:1047` | — |
| L2003 | `mamba_utils.preprocess_mamba` | `—` | — |
| L2011 | `self.model.get_mamba_state_copy_func` | `—` | — |
| L2021 | `self.num_accepted_tokens.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L2024 | `mamba_utils.stage_postprocess_inputs_to_gpu` | `—` | — |
| L2037 | `np.repeat` | `—` | numpy |
| L2039 | `np.add` | `graph_fusion_pass_manager.py:45` | numpy |
| L2045 | `len` | `—` | — |
| L2050 | `enable_sp` | `—` | — |
| L2056 | `self._pad_query_start_loc_for_fia` | `model_runner_v1.py:834` | — |
| L2065 | `self._build_attention_metadata` | `model_runner_v1.py:2875` | — |
| L2079 | `self._sanitize_placeholder_input_ids_for_forward` | `model_runner_v1.py:1349` | — |
| L2091 | `self._preprocess` | `gpu_model_runner.py:3490` | — |
| L2616 | `self._compute_cascade_attn_prefix_len` | `gpu_model_runner.py:2628` | — |
| L2623 | `append` | `utils.py:52` | — |
| L2695 | `get_forward_context` | `—` | — |
| L2698 | `num_computed_tokens.min` | `—` | — |
| L2705 | `partial` | `—` | — |
| L2709 | `self._update_full_graph_params_if_needed` | `model_runner_v1.py:2663` | — |
| L2710 | `run_model` | `—` | — |
| L2712 | `attn_metadata_builder.use_cascade_attention` | `—` | — |
| L2716 | `self._all_gather_hidden_states_and_aux` | `model_runner_v1.py:2655` | — |
| L2723 | `enable_sp_by_pass` | `—` | — |
| L2724 | `round_up` | `utils.py:502` | — |
| L2733 | `length_from_prompt_token_ids_or_embeds` | `—` | — |
| L2764 | `MRotaryEmbedding.get_next_input_positions_tensor` | `—` | numpy |
| L2772 | `self.sync_and_slice_intermediate_tensors` | `model_runner_v1.py:2732` | — |
| L2792 | `self._pad_for_sequence_parallelism` | `model_runner_v1.py:2719` | — |
| L2812 | `XDRotaryEmbedding.get_next_input_positions_tensor` | `—` | numpy |
| L2816 | `BatchDescriptor` | `—` | — |
| L2818 | `self.cudagraph_dispatcher.dispatch` | `—` | — |
| L2827 | `dispatch_cudagraph` | `model_runner_v1.py:2814` | — |
| L2837 | `self._sync_metadata_across_dp` | `model_runner_v1.py:702` | — |
| L2842 | `oproj_tp_enable` | `—` | — |
| L2843 | `embedding_tp_enable` | `—` | — |
| L2860 | `CUDAGraphStat` | `—` | — |
| L2864 | `str` | `—` | — |
| L2900 | `update_sparse_kv_offload_metadata` | `sparse_kv_offload_manager.py:234` | — |
| L2912 | `dict` | `—` | — |
| L2920 | `max()` | `—` | numpy |
| L2934 | `self.dcp_manager.generate_dcp_metadata` | `—` | — |
| L2949 | `scheduled_encoder_inputs.items` | `—` | — |
| L2957 | `mm_hashes.append` | `utils.py:52` | — |
| L2958 | `mm_kwargs.append` | `utils.py:52` | — |
| L2959 | `mm_lora_refs.append` | `utils.py:52` | — |
| L2964 | `blk_table.get_device_tensor` | `block_table.py:221` | — |
| L2966 | `self._batch_mm_inputs_from_scheduler` | `gpu_model_runner.py:2920` | — |
| L2980 | `_get_block_table_and_slot_mapping` | `model_runner_v1.py:2945` | — |
| L2981 | `_get_dcp_metadata` | `model_runner_v1.py:2925` | — |
| L2988 | `pe_tensor.to` | `—` | .to(device=...) |
| L2989 | `self.maybe_save_ec_to_connector` | `—` | — |
| L2996 | `AscendCommonAttentionMetadata` | `—` | — |
| L3013 | `cast` | `—` | — |
| L3015 | `self.lora_manager.supports_tower_connector_lora` | `—` | — |
| L3028 | `self.model.get_num_mm_encoder_tokens` | `—` | — |
| L3029 | `pos_info.get_num_embeds` | `—` | — |
| L3031 | `prompt_lora_mapping.append` | `utils.py:52` | — |
| L3032 | `token_lora_mapping.extend` | `utils.py:55` | — |
| L3033 | `encoder_token_counts.append` | `utils.py:52` | — |
| L3036 | `self.input_batch.lora_id_to_lora_request.get` | `backend.py:55` | — |
| L3038 | `lora_requests.add` | `graph_fusion_pass_manager.py:45` | — |
| L3041 | `LoRAMapping` | `—` | — |
| L3042 | `logits_indices.size` | `—` | — |
| L3042 | `tuple` | `—` | — |
| L3043 | `self._prepare_kv_sharing_fast_prefill` | `gpu_model_runner.py:2896` | — |
| L3047 | `self.lora_manager.set_active_adapters` | `—` | — |
| L3054 | `self.model.get_mm_mapping` | `—` | — |
| L3055 | `hasattr` | `—` | — |
| L3056 | `attn_group.get_metadata_builder` | `—` | — |
| L3064 | `self.model.get_num_mm_connector_tokens` | `—` | — |
| L3067 | `common_attn_metadata.replace` | `—` | — |
| L3073 | `connector_token_mapping.tolist` | `—` | — |
| L3077 | `torch.zeros_like` | `—` | — |
| L3087 | `group_and_batch_mm_kwargs` | `—` | pin_memory |
| L3113 | `builder.build_for_cudagraph_capture` | `—` | — |
| L3115 | `builder.build` | `attention_v1.py:291` | — |
| L3118 | `next` | `—` | pin_memory |
| L3122 | `self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs` | `—` | — |
| L3126 | `model.embed_multimodal` | `—` | — |
| L3130 | `batch_outputs_lst.extend` | `utils.py:55` | — |
| L3142 | `self.timed_encoder_operation` | `gpu_model_runner.py:7788` | — |
| L3148 | `copy` | `utils.py:106` | — |
| L3148 | `self.encoder_cudagraph_manager.supports_modality` | `—` | — |
| L3150 | `self.encoder_cudagraph_manager.execute` | `—` | — |
| L3151 | `self._get_encoder_seq_lens` | `gpu_model_runner.py:1892` | — |
| L3159 | `sanity_check_mm_encoder_outputs` | `—` | — |
| L3160 | `encoder_outputs.extend` | `utils.py:55` | — |
| L3165 | `zip` | `—` | — |
| L3176 | `self.drafter.set_per_group_attn_metadata` | `dspark_proposer.py:243` | — |
| L3189 | `_build_attn_group_metadata` | `model_runner_v1.py:3045` | — |
| L3199 | `get_mm_features_in_window` | `—` | — |
| L3204 | `pos_info.extract_embeds_range` | `—` | — |
| L3205 | `image_doc_ranges.extend` | `utils.py:55` | — |
| L3211 | `ub_metadata.values` | `—` | — |
| L3214 | `attn_metadata.values` | `—` | — |
| L3217 | `pos_info.get_embeds_indices_in_range` | `—` | — |
| L3223 | `spec_decode_common_attn_metadata.unpadded` | `utils.py:277` | — |
| L3225 | `self.encoder_cache.get` | `backend.py:55` | — |
| L3253 | `set_mm_embedding_modality` | `—` | — |
| L3254 | `mm_embeds_req.append` | `utils.py:52` | — |
| L3260 | `cudagraph_runtime_mode.valid_runtime_modes` | `—` | — |
| L3261 | `self.model.recompute_mrope_positions` | `—` | — |
| L3269 | `copy_mm_embedding_modality` | `—` | — |
| L3272 | `req_state.mrope_positions.copy_` | `—` | — |
| L3275 | `mm_embeds.extend` | `utils.py:55` | — |
| L3280 | `self.mrope_positions.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L3281 | `NotImplementedError` | `—` | — |
| L3283 | `cdiv` | `—` | — |
| L3284 | `self.xdrope_positions.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L3299 | `self.eplb_updator.forward_before` | `—` | — |
| L3303 | `num_scheduled_tokens.sum` | `—` | — |
| L3333 | `self.dcp_manager.query_lens_full.copy_to_gpu` | `utils.py:139` | CpuGpuBuffer.copy_to_gpu |
| L3346 | `num_scheduled_tokens.repeat` | `—` | — |
| L3361 | `self._should_build_dummy_attn_metadata` | `model_runner_v1.py:3226` | — |
| L3383 | `using_paged_attention` | `utils.py:170` | — |
| L3389 | `self.seq_lens.copy_` | `—` | non_blocking=True |
| L3397 | `self.gdn_query_start_loc.np.fill` | `—` | numpy |
| L3422 | `blk_table.slot_mapping.gpu.fill_` | `—` | — |
| L3427 | `self.positions.fill_` | `—` | — |
| L3428 | `self._dsa_positions_cpu_buf.fill_` | `—` | — |
| L3440 | `self.maybe_dummy_run_with_lora` | `—` | — |
| L3467 | `update_cos_sin` | `rotary_embedding.py:144` | — |
| L3477 | `get_tensor_model_parallel_world_size` | `—` | — |
| L3483 | `self.model.make_empty_intermediate_tensors` | `—` | — |
| L3486 | `IntermediateTensors` | `—` | — |
| L3487 | `self.intermediate_tensors.items` | `—` | — |
| L3503 | `self.model.compute_logits` | `—` | — |
| L3509 | `self.drafter.model.compute_logits` | `—` | — |
| L3509 | `clamp_` | `—` | — |
| L3511 | `set_ascend_forward_context` | `—` | — |
| L3522 | `self._gather_mm_embeddings` | `gpu_model_runner.py:3172` | — |
| L3525 | `self._model_forward` | `model_runner_v1.py:2685` | — |
| L3532 | `dummy_compute_logits` | `—` | — |
| L3535 | `self.drafter.dummy_run` | `dflash_proposer.py:153` | — |
| L3540 | `self.model.embed_input_ids` | `—` | — |
| L3547 | `self.eplb_updator.adaptor.clear_all_moe_loads` | `—` | — |
| L3547 | `is_token_ids.unsqueeze` | `—` | — |
| L3549 | `self.eplb_updator.forward_end` | `—` | — |
| L3563 | `self._prepare_mm_inputs` | `gpu_model_runner.py:3479` | — |
| L3565 | `self._init_model_kwargs` | `gpu_model_runner.py:1065` | — |
| L3566 | `self._extract_mm_kwargs` | `gpu_model_runner.py:1683` | — |
| L3585 | `async_tensor_h2d` | `—` | — |
| L3609 | `zero_` | `—` | — |
| L3615 | `self.sync_and_gather_intermediate_tensors` | `model_runner_v1.py:2763` | — |
| L3626 | `model_kwargs.update` | `config_data.py:812` | — |
| L3750 | `self.debugger.start` | `—` | — |
| L3757 | `self.debugger.stop` | `read_thread.py:298` | — |
| L3760 | `self.debugger.step` | `—` | — |
| L3818 | `self.prepare_inputs_event.synchronize` | `—` | synchronize |
| L3822 | `self.prepare_inputs_event.record` | `—` | — |
| L7811 | `torch.accelerator.synchronize` | `—` | synchronize |
| L7812 | `time.perf_counter` | `—` | — |
| L7825 | `EncoderTimingStats` | `—` | — |

## 4. 占比分母的唯一定义

```
t_prepare_input = wall( 从 L1859 进入 with 开始 )
                  - wall( 到 L2098 执行完 update_cos_sin 为止 )
```

scope 内**包含**（必须计入分母）：

- `synchronize_input_prep` 的 event synchronize/record（scope 入口）
- `_update_states`（ascend 覆写 + 父类 `GPUModelRunner._update_states`）
- `_prepare_inputs` 全流程（含 block table / positions / slot mapping / spec metadata / attention metadata）
- `_determine_batch_execution_and_padding`、`maybe_create_ubatch_slices`、`_pad_query_start_loc_for_fia`
- `_build_attention_metadata`、`_sanitize_placeholder_input_ids_for_forward`
- `_preprocess`、`update_cos_sin`

scope **不含**（不得计入）：

- `if self.dynamic_eplb: self.eplb_updator.forward_before()`（L2100 起）
- `with record_function_or_nullcontext("forward")`（L2120 起）
- `post process`（L2147）/ `sample_token`（L2266）/ `draft_token`
