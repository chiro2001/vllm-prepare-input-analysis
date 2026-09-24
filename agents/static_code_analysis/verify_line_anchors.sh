#!/usr/bin/env bash
# 抽检 docs/01 中引用的关键行号是否与源码内容一致
set -u
V=/home/chiro/projects/vllm/HIST_PROJECT/vllm
A=/home/chiro/projects/vllm/HIST_PROJECT/vllm-ascend
chk() { # chk <file> <line> <expected-substring>
  local f="$1" l="$2" pat="$3"
  local got; got=$(sed -n "${l}p" "$f")
  if [[ "$got" == *"$pat"* ]]; then printf "OK   %s:%s  %s\n" "$(basename $f)" "$l" "$pat"
  else printf "FAIL %s:%s  want=%s got=%s\n" "$(basename $f)" "$l" "$pat" "$got"; fi
}
chk $A/vllm_ascend/worker/model_runner_v1.py 1859 'record_function_or_nullcontext("prepare input")'
chk $A/vllm_ascend/worker/model_runner_v1.py 1860 'with self.synchronize_input_prep():'
chk $A/vllm_ascend/worker/model_runner_v1.py 1881 'deferred_state_corrections_fn = self._update_states('
chk $A/vllm_ascend/worker/model_runner_v1.py 1906 '_dummy_run(1, skip_gdn_state_update=True)'
chk $A/vllm_ascend/worker/model_runner_v1.py 1933 ') = self._prepare_inputs('
chk $A/vllm_ascend/worker/model_runner_v1.py 2065 '(attn_metadata, spec_decode_common_attn_metadata) = self._build_attention_metadata('
chk $A/vllm_ascend/worker/model_runner_v1.py 2079 'self._sanitize_placeholder_input_ids_for_forward('
chk $A/vllm_ascend/worker/model_runner_v1.py 2091 ') = self._preprocess('
chk $A/vllm_ascend/worker/model_runner_v1.py 2098 'update_cos_sin(positions)'
chk $A/vllm_ascend/worker/model_runner_v1.py 883 'def _prepare_inputs('
chk $A/vllm_ascend/worker/model_runner_v1.py 906 'self.input_batch.block_table.commit_block_table(num_reqs)'
chk $A/vllm_ascend/worker/model_runner_v1.py 1104 'self.num_accepted_tokens_event.synchronize()'
chk $A/vllm_ascend/worker/model_runner_v1.py 1320 'def _build_attn_state('
chk $A/vllm_ascend/attention/attention_v1.py 291 'def build('
chk $A/vllm_ascend/attention/attention_v1.py 330 'query_start_loc = query_start_loc_cpu.pin_memory().to(self.device, non_blocking=True)'
chk $A/vllm_ascend/attention/attention_v1.py 332 'actual_seq_lengths_q = query_start_loc_cpu[1:].tolist()'
chk $A/vllm_ascend/attention/attention_v1.py 333 'seq_lens_list = seq_lens.tolist()'
chk $A/vllm_ascend/worker/block_table.py 179 '_compute_slot_mapping_kernel[(num_reqs + 1,)]('
chk $V/vllm/v1/worker/gpu_model_runner.py 4148 'with ('
chk $V/vllm/v1/worker/gpu_model_runner.py 4149 '"gpu_model_runner: preprocess"'
chk $V/vllm/v1/worker/gpu_model_runner.py 4195 'logits_indices, spec_decode_metadata = self._prepare_inputs('
chk $V/vllm/v1/worker/gpu_model_runner.py 1937 'def _prepare_inputs('
chk $V/vllm/v1/worker/gpu_model_runner.py 1169 'def _update_states('
chk $V/vllm/v1/worker/gpu_model_runner.py 3818 'self.prepare_inputs_event.synchronize()'
chk $V/vllm/v1/worker/gpu_model_runner.py 915 'if self.num_spec_tokens:'
chk $V/vllm/v1/worker/gpu_input_batch.py 698 'if not (empty_req_indices := self.batch_update_builder.removed):'
chk $V/vllm/v1/worker/gpu_input_batch.py 831 'if batch_update:'
chk $V/vllm/v1/attention/backends/utils.py 665 'def reorder_batch_to_split_decodes_and_prefills('
chk $V/vllm/v1/worker/block_table.py 166 '_compute_slot_mapping_kernel[(num_reqs + 1,)]('
