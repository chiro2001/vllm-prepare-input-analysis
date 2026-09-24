# libkperfx 920B PMU 事件码表与预置组（a3-22 实测）

生成方式：`kperfx events` + `kperfx presets 920b_topdown_full_{1..9}`，原始输出见 `data/toolchain/raw/kperfx-events.txt`、`data/toolchain/raw/preset-920b_topdown_full_*.txt`。

- 命名事件总数：93
- 单组硬件上限：**8** 个计数器（`kperfx probe` 实测 1..8 confidence=1.0000，9/10 为 `not scheduled`）
- 全量 topdown 需要 **9 组**（前 8 组各 8 个硬件计数器 + 第 9 组软件事件）
- 发射宽度 `issue_width = 6`
- `CONFIG` 就是 `perf_event_attr.config` 的原始值，`r<hex>` 可直接传给 `perf`/libkperfx
- 第 9 组是软件事件（`context_switches` / `cpu_migrations` / `page_faults` / `cpu_clock`），不占 PMU 计数器

## 预置组

| 组 | 事件数 | 事件 |
|---|---|---|
| `920b_topdown_full_1` | 8 | `cpu_cycles`, `inst_retired`, `inst_spec`, `fetch_bubble`, `fetch_bubble_max`, `total_resource_stall`, `exec_stall`, `fdiv_fsqrt_stall` |
| `920b_topdown_full_2` | 8 | `exe_stall_div`, `fsu_stall`, `mem_stall_anyload`, `mem_stall_anystore`, `memstall_l1miss`, `memstall_l2miss`, `memstall_l3miss`, `rob_stall` |
| `920b_topdown_full_3` | 8 | `br_mis_pred`, `o3_flush`, `nuke_flush`, `bp_misp_br_ind`, `bp_misp_br_blr`, `bp_misp_br_bl`, `bp_misp_br_ret`, `if_sp_flush` |
| `920b_topdown_full_4` | 8 | `l2i_tlb`, `l2i_tlb_refill`, `l2i_cache`, `l2i_cache_refill`, `pcbuf_stall`, `int_ptag_stall`, `cc_ptag_stall`, `vfp_single_ptag_stall` |
| `920b_topdown_full_5` | 8 | `vfp_pair_ptag_stall`, `vpd_ptag_stall`, `int_mpq_stall`, `vfp_mpq_stall`, `cc_mpq_stall`, `ports_0_serialize`, `ports_0_nonserialize`, `ports_1` |
| `920b_topdown_full_6` | 8 | `ports_2`, `ports_3`, `ports_4`, `ports_5`, `ports_6`, `ld_cancel_dtlb_miss`, `ld_cancel_misalign`, `ld_cancel_lq_full` |
| `920b_topdown_full_7` | 8 | `ld_cancel_inst_type`, `ld_cancel_fwd_hzd`, `ld_cancel_struct_hzd`, `ld_cancel_pipeline`, `l2_bound_buf`, `l2_bound_snp`, `l2_bound_arb`, `dram_local` |
| `920b_topdown_full_8` | 6 | `dram_remote`, `dram_remote_cache`, `st_sca_full`, `st_head_no_pgen`, `st_order_fail`, `st_bound_pipeline` |
| `920b_topdown_full_9` | 4 | `context_switches`, `cpu_migrations`, `page_faults`, `cpu_clock` |

## 全部命名事件

| NAME | TYPE | CONFIG | ALIASES |
|---|---|---|---|
| `CPU_CYCLES` | raw | `0x000000011` | cpu_cycles,cycles,CPU_CYCLES |
| `INST_RETIRED` | raw | `0x000000008` | inst_retired,instructions,INST_RETIRED,inst |
| `INST_SPEC` | raw | `0x00000001b` | inst_spec,INST_SPEC |
| `BR_MIS_PRED` | raw | `0x000000010` | br_mis_pred,BR_MIS_PRED,branch_misses |
| `BR_PRED` | raw | `0x000000012` | br_pred,BR_PRED |
| `STALL_FRONTEND` | raw | `0x000000023` | stall_frontend,STALL_FRONTEND |
| `STALL_BACKEND` | raw | `0x000000024` | stall_backend,STALL_BACKEND |
| `L1D_TLB` | raw | `0x000000025` | l1d_tlb,L1D_TLB |
| `L1I_TLB` | raw | `0x000000026` | l1i_tlb,L1I_TLB |
| `L1D_CACHE` | raw | `0x000000004` | l1d_cache,L1D_CACHE |
| `L1D_CACHE_REFILL` | raw | `0x000000003` | l1d_cache_refill,L1D_CACHE_REFILL |
| `L1I_CACHE_REFILL` | raw | `0x000000001` | l1i_cache_refill,L1I_CACHE_REFILL |
| `LD_RETIRED` | raw | `0x000000006` | ld_retired,LD_RETIRED |
| `ST_RETIRED` | raw | `0x000000007` | st_retired,ST_RETIRED |
| `MEM_ACCESS` | raw | `0x000000013` | mem_access,MEM_ACCESS |
| `TOTAL_RESOURCE_STALL` | raw | `0x000007000` | total_resource_stall,TOTAL_RESOURCE_STALL |
| `EXEC_STALL` | raw | `0x000007001` | exec_stall,EXEC_STALL |
| `FDIV_FSQRT_STALL` | raw | `0x000007002` | fdiv_fsqrt_stall,FDIV_FSQRT_STALL |
| `EXE_STALL_DIV` | raw | `0x000007003` | exe_stall_div,EXE_STALL_DIV |
| `FSU_STALL` | raw | `0x000007004` | fsu_stall,FSU_STALL |
| `MEM_STALL_ANYLOAD` | raw | `0x000007005` | mem_stall_anyload,MEM_STALL_ANYLOAD |
| `MEM_STALL_ANYSTORE` | raw | `0x000007006` | mem_stall_anystore,MEM_STALL_ANYSTORE |
| `MEMSTALL_L1MISS` | raw | `0x000007007` | memstall_l1miss,MEMSTALL_L1MISS |
| `MEMSTALL_L2MISS` | raw | `0x000007008` | memstall_l2miss,MEMSTALL_L2MISS |
| `MEMSTALL_L3MISS` | raw | `0x000007009` | memstall_l3miss,MEMSTALL_L3MISS |
| `PORTS_0_SERIALIZE` | raw | `0x00000700a` | ports_0_serialize,PORTS_0_SERIALIZE |
| `PORTS_0_NONSERIALIZE` | raw | `0x00000700b` | ports_0_nonserialize,PORTS_0_NONSERIALIZE |
| `PORTS_1` | raw | `0x00000700c` | ports_1,PORTS_1 |
| `PORTS_2` | raw | `0x00000700d` | ports_2,PORTS_2 |
| `PORTS_3` | raw | `0x00000700e` | ports_3,PORTS_3 |
| `PORTS_4` | raw | `0x00000700f` | ports_4,PORTS_4 |
| `PORTS_5` | raw | `0x000007010` | ports_5,PORTS_5 |
| `PORTS_6` | raw | `0x000007011` | ports_6,ports_6p,PORTS_6,PORTS_6P |
| `PORTS_7` | raw | `0x000007012` | ports_7,PORTS_7 |
| `PORTS_8` | raw | `0x000007013` | ports_8,PORTS_8 |
| `ROB_STALL` | raw | `0x000002004` | rob_stall,ROB_STALL |
| `PC_BUF_STALL` | raw | `0x000002005` | pcbuf_stall,PC_BUF_STALL,PC_BUF_FULL,pc_buf_stall |
| `INT_PTAG_STALL` | raw | `0x000002006` | int_ptag_stall,INT_PTAG_STALL |
| `CC_PTAG_STALL` | raw | `0x000002007` | cc_ptag_stall,CC_PTAG_STALL |
| `LD_CANCEL_DTLB_MISS` | raw | `0x000005090` | ld_cancel_dtlb_miss,LD_CANCEL_DTLB_MISS |
| `LD_CANCEL_MISALIGN` | raw | `0x000005091` | ld_cancel_misalign,LD_CANCEL_MISALIGN |
| `LD_CANCEL_LQ_FULL` | raw | `0x000005092` | ld_cancel_lq_full,LD_CANCEL_LQ_FULL |
| `LD_CANCEL_INST_TYPE` | raw | `0x000005093` | ld_cancel_inst_type,LD_CANCEL_INST_TYPE |
| `LD_CANCEL_FWD_HZD` | raw | `0x000005094` | ld_cancel_fwd_hzd,LD_CANCEL_FWD_HZD |
| `LD_CANCEL_STRUCT_HZD` | raw | `0x000005095` | ld_cancel_struct_hzd,LD_CANCEL_STRUCT_HZD |
| `LD_CANCEL_PIPELINE` | raw | `0x000005096` | ld_cancel_pipeline,LD_CANCEL_PIPELINE |
| `L2_BOUND_BUF` | raw | `0x00000701e` | l2_bound_buf,L2_BOUND_BUF |
| `L2_BOUND_SNP` | raw | `0x00000701f` | l2_bound_snp,L2_BOUND_SNP |
| `L2_BOUND_ARB` | raw | `0x000007020` | l2_bound_arb,L2_BOUND_ARB |
| `DRAM_LOCAL` | raw | `0x000007021` | dram_local,DRAM_LOCAL |
| `DRAM_REMOTE` | raw | `0x000007022` | dram_remote,DRAM_REMOTE |
| `DRAM_REMOTE_CACHE` | raw | `0x000007023` | dram_remote_cache,DRAM_REMOTE_CACHE,DRAM_REMOTE_CACHE_ACCESS |
| `ST_SCA_FULL` | raw | `0x0000050a0` | st_sca_full,ST_SCA_FULL |
| `ST_HEAD_NO_PGEN` | raw | `0x0000050a2` | st_head_no_pgen,ST_HEAD_NO_PGEN |
| `ST_ORDER_FAIL` | raw | `0x0000050a3` | st_order_fail,ST_ORDER_FAIL |
| `BP_MISP_BR_IND` | raw | `0x000001010` | bp_misp_br_ind,BP_MISP_BR_IND |
| `BP_MISP_BR_BLR` | raw | `0x000001013` | bp_misp_br_blr,BP_MISP_BR_BLR |
| `BP_MISP_BR_BL` | raw | `0x000001016` | bp_misp_br_bl,BP_MISP_BR_BL |
| `BP_MISP_BR_RET` | raw | `0x00000100d` | bp_misp_br_ret,BP_MISP_BR_RET |
| `VPD_PTAG_STALL` | raw | `0x00000200a` | vpd_ptag_stall,VPD_PTAG_STALL |
| `FETCH_BUBBLE` | raw | `0x000002011` | fetch_bubble,FETCH_BUBBLE |
| `FETCH_BUBBLE_MAX` | raw | `0x000002012` | fetch_bubble_max,FETCH_BUBBLE_MAX |
| `O3_FLUSH` | raw | `0x000002010` | o3_flush,O3_FLUSH |
| `NUKE_FLUSH` | raw | `0x00000200f` | nuke_flush,NUKE_FLUSH |
| `IF_SP_FLUSH` | raw | `0x00000104f` | if_sp_flush,IF_SP_FLUSH |
| `L2I_TLB` | raw | `0x000000030` | l2i_tlb,L2I_TLB |
| `L2I_TLB_REFILL` | raw | `0x00000002e` | l2i_tlb_refill,L2I_TLB_REFILL |
| `L2I_CACHE` | raw | `0x000000027` | l2i_cache,L2I_CACHE |
| `L2I_CACHE_REFILL` | raw | `0x000000028` | l2i_cache_refill,L2I_CACHE_REFILL |
| `VFP_SINGLE_PTAG_STALL` | raw | `0x000002008` | vfp_single_ptag_stall,VFP_SINGLE_PTAG_STALL |
| `VFP_PAIR_PTAG_STALL` | raw | `0x000002009` | vfp_pair_ptag_stall,VFP_PAIR_PTAG_STALL |
| `INT_MPQ_STALL` | raw | `0x00000200b` | int_mpq_stall,INT_MPQ_STALL |
| `VFP_MPQ_STALL` | raw | `0x00000200c` | vfp_mpq_stall,VFP_MPQ_STALL |
| `CC_MPQ_STALL` | raw | `0x00000200d` | cc_mpq_stall,CC_MPQ_STALL |
| `ST_BOUND_PIPELINE` | raw | `0x0000050a4` | st_bound_pipeline,ST_BOUND_PIPELINE |
| `CONTEXT_SWITCHES` | software | `0x000000003` | context_switches,cs,CONTEXT_SWITCHES |
| `CPU_MIGRATIONS` | software | `0x000000004` | cpu_migrations,migrations,CPU_MIGRATIONS |
| `PAGE_FAULTS` | software | `0x000000002` | page_faults,PAGE_FAULTS |
| `MINOR_FAULTS` | software | `0x000000005` | minor_faults,MINOR_FAULTS |
| `MAJOR_FAULTS` | software | `0x000000006` | major_faults,MAJOR_FAULTS |
| `CPU_CLOCK` | software | `0x000000000` | cpu_clock,CPU_CLOCK |
| `TASK_CLOCK` | software | `0x000000001` | task_clock,TASK_CLOCK |
| `ALIGNMENT_FAULTS` | software | `0x000000007` | alignment_faults,ALIGNMENT_FAULTS |
| `EMULATION_FAULTS` | software | `0x000000008` | emulation_faults,EMULATION_FAULTS |
| `HW_CPU_CYCLES` | hardware | `0x000000000` | hw_cpu_cycles,hw_cycles |
| `HW_INSTRUCTIONS` | hardware | `0x000000001` | hw_instructions |
| `HW_CACHE_REFERENCES` | hardware | `0x000000002` | hw_cache_references |
| `HW_CACHE_MISSES` | hardware | `0x000000003` | hw_cache_misses |
| `HW_BRANCH_INSTRUCTIONS` | hardware | `0x000000004` | hw_branch_instructions |
| `HW_BRANCH_MISSES` | hardware | `0x000000005` | hw_branch_misses |
| `HW_BUS_CYCLES` | hardware | `0x000000006` | hw_bus_cycles |
| `HW_STALLED_CYCLES_FRONTEND` | hardware | `0x000000007` | hw_stalled_cycles_frontend |
| `HW_STALLED_CYCLES_BACKEND` | hardware | `0x000000008` | hw_stalled_cycles_backend |
