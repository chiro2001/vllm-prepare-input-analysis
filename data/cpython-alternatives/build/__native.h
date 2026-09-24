#ifndef MYPYC_NATIVE_H
#define MYPYC_NATIVE_H
#include <Python.h>
#include <CPy.h>
#ifndef MYPYC_DECLARED_tuple_T8IIFIFFCI
#define MYPYC_DECLARED_tuple_T8IIFIFFCI
typedef struct tuple_T8IIFIFFCI {
    CPyTagged f0;
    CPyTagged f1;
    double f2;
    CPyTagged f3;
    double f4;
    double f5;
    char f6;
    CPyTagged f7;
} tuple_T8IIFIFFCI;
#endif

#ifndef MYPYC_DECLARED_tuple_T0
#define MYPYC_DECLARED_tuple_T0
typedef struct tuple_T0 {
    int empty_struct_error_flag;
} tuple_T0;
#endif

typedef struct {
    PyObject_HEAD
    CPyVTableItem *vtable;
    CPyTagged _num_reqs;
    CPyTagged _num_actual_tokens;
    CPyTagged _max_query_len;
    PyObject *_query_start_loc;
    PyObject *_seq_lens;
    PyObject *_block_table;
    PyObject *_slot_mapping;
    char _spec_decode;
    CPyTagged _num_decode_tokens;
    CPyTagged _num_prefill_tokens;
    PyObject *_num_computed_tokens;
    PyObject *_seq_lens_np;
    CPyTagged _max_seq_len;
    PyObject *_kv_cache_dtype;
    CPyTagged _block_size;
    CPyTagged _num_kv_heads;
    CPyTagged _head_size;
    char _causal;
    CPyTagged _sliding_window;
    PyObject *_token_ids;
    PyObject *_positions;
    PyObject *_workspace;
} mypyc_variant___MetaNativeObject;

typedef struct {
    PyObject_HEAD
    CPyVTableItem *vtable;
    CPyTagged _num_reqs;
    CPyTagged _num_actual_tokens;
    CPyTagged _max_query_len;
    PyObject *_query_start_loc;
    PyObject *_seq_lens;
    PyObject *_block_table;
    PyObject *_slot_mapping;
    char _spec_decode;
    CPyTagged _num_decode_tokens;
    CPyTagged _num_prefill_tokens;
    PyObject *_num_computed_tokens;
    PyObject *_seq_lens_np;
    CPyTagged _max_seq_len;
    PyObject *_kv_cache_dtype;
    CPyTagged _block_size;
    CPyTagged _num_kv_heads;
    CPyTagged _head_size;
    char _causal;
    CPyTagged _sliding_window;
    PyObject *_token_ids;
    PyObject *_positions;
    PyObject *_workspace;
} mypyc_variant___MetaDataclassObject;

#endif
