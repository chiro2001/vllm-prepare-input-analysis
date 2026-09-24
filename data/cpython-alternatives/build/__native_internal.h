#ifndef MYPYC_LIBRT_INTERNAL_H
#define MYPYC_LIBRT_INTERNAL_H
#include <Python.h>
#include <CPy.h>
#include "__native.h"

int CPyGlobalsInit(void);

extern PyObject *CPyStatics[76];
extern const char * const CPyLit_Str[];
extern const char * const CPyLit_Bytes[];
extern const char * const CPyLit_Int[];
extern const double CPyLit_Float[];
extern const double CPyLit_Complex[];
extern const int CPyLit_Tuple[];
extern const int CPyLit_FrozenSet[];
extern CPyModule *CPyModule_mypyc_variant__internal;
extern CPyModule *CPyModule_mypyc_variant;
extern PyObject *CPyStatic_globals;
extern CPyModule *CPyModule_builtins;
extern CPyModule *CPyModule___future__;
extern CPyModule *CPyModule_dataclasses;
extern CPyModule *CPyModule_typing;
extern CPyModule *CPyModule_numpy;
extern CPyModule *CPyModule_torch;
extern int CPyExec_mypyc_variant(PyObject *module);
extern PyTypeObject *CPyType_MetaNative;
extern PyObject *CPyDef_MetaNative(CPyTagged cpy_r_num_reqs, CPyTagged cpy_r_num_actual_tokens, CPyTagged cpy_r_max_query_len, PyObject *cpy_r_query_start_loc, PyObject *cpy_r_seq_lens, PyObject *cpy_r_block_table, PyObject *cpy_r_slot_mapping, char cpy_r_spec_decode, CPyTagged cpy_r_num_decode_tokens, CPyTagged cpy_r_num_prefill_tokens, PyObject *cpy_r_num_computed_tokens, PyObject *cpy_r_seq_lens_np, CPyTagged cpy_r_max_seq_len, PyObject *cpy_r_kv_cache_dtype, CPyTagged cpy_r_block_size, CPyTagged cpy_r_num_kv_heads, CPyTagged cpy_r_head_size, char cpy_r_causal, CPyTagged cpy_r_sliding_window, PyObject *cpy_r_token_ids, PyObject *cpy_r_positions, PyObject *cpy_r_workspace);
extern PyTypeObject *CPyType_MetaDataclass;
extern PyObject *CPyDef_MetaDataclass(PyObject *cpy_r_args, PyObject *cpy_r_kwargs);
extern char CPyDef_MetaNative_____init__(PyObject *cpy_r_self, CPyTagged cpy_r_num_reqs, CPyTagged cpy_r_num_actual_tokens, CPyTagged cpy_r_max_query_len, PyObject *cpy_r_query_start_loc, PyObject *cpy_r_seq_lens, PyObject *cpy_r_block_table, PyObject *cpy_r_slot_mapping, char cpy_r_spec_decode, CPyTagged cpy_r_num_decode_tokens, CPyTagged cpy_r_num_prefill_tokens, PyObject *cpy_r_num_computed_tokens, PyObject *cpy_r_seq_lens_np, CPyTagged cpy_r_max_seq_len, PyObject *cpy_r_kv_cache_dtype, CPyTagged cpy_r_block_size, CPyTagged cpy_r_num_kv_heads, CPyTagged cpy_r_head_size, char cpy_r_causal, CPyTagged cpy_r_sliding_window, PyObject *cpy_r_token_ids, PyObject *cpy_r_positions, PyObject *cpy_r_workspace);
extern PyObject *CPyPy_MetaNative_____init__(PyObject *self, PyObject *args, PyObject *kw);
extern PyObject *CPyDef__kwargs(PyObject *cpy_r_inp);
extern PyObject *CPyPy__kwargs(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames);
extern PyObject *CPyDef_small_ops(PyObject *cpy_r_inp, PyObject *cpy_r_mode);
extern PyObject *CPyPy_small_ops(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames);
extern CPyTagged CPyDef__read_scalars(PyObject *cpy_r_m);
extern PyObject *CPyPy__read_scalars(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames);
extern CPyTagged CPyDef__read_containers(PyObject *cpy_r_m);
extern PyObject *CPyPy__read_containers(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames);
extern CPyTagged CPyDef_run_mypyc_native(PyObject *cpy_r_inp, PyObject *cpy_r_mode);
extern PyObject *CPyPy_run_mypyc_native(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames);
extern CPyTagged CPyDef_run_mypyc_dataclass(PyObject *cpy_r_inp, PyObject *cpy_r_mode);
extern PyObject *CPyPy_run_mypyc_dataclass(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames);
extern char CPyDef___top_level__(void);
#endif
