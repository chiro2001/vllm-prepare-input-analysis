#include <init.c>
#include <getargs.c>
#include <getargsfast.c>
#include <int_ops.c>
#include <float_ops.c>
#include <str_ops.c>
#include <bytes_ops.c>
#include <list_ops.c>
#include <dict_ops.c>
#include <set_ops.c>
#include <tuple_ops.c>
#include <exc_ops.c>
#include <misc_ops.c>
#include <generic_ops.c>
#include <pythonsupport.c>
#include <function_wrapper.c>
#include "__native.h"
#include "__native_internal.h"

static int
MetaNative_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    return 0;
}
static int
MetaNative_traverse(mypyc_variant___MetaNativeObject *self, visitproc visit, void *arg)
{
    if (CPyTagged_CheckLong(self->_num_reqs)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_num_reqs));
    }
    if (CPyTagged_CheckLong(self->_num_actual_tokens)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_num_actual_tokens));
    }
    if (CPyTagged_CheckLong(self->_max_query_len)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_max_query_len));
    }
    Py_VISIT(self->_query_start_loc);
    Py_VISIT(self->_seq_lens);
    Py_VISIT(self->_block_table);
    Py_VISIT(self->_slot_mapping);
    if (CPyTagged_CheckLong(self->_num_decode_tokens)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_num_decode_tokens));
    }
    if (CPyTagged_CheckLong(self->_num_prefill_tokens)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_num_prefill_tokens));
    }
    Py_VISIT(self->_num_computed_tokens);
    Py_VISIT(self->_seq_lens_np);
    if (CPyTagged_CheckLong(self->_max_seq_len)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_max_seq_len));
    }
    Py_VISIT(self->_kv_cache_dtype);
    if (CPyTagged_CheckLong(self->_block_size)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_block_size));
    }
    if (CPyTagged_CheckLong(self->_num_kv_heads)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_num_kv_heads));
    }
    if (CPyTagged_CheckLong(self->_head_size)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_head_size));
    }
    if (CPyTagged_CheckLong(self->_sliding_window)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_sliding_window));
    }
    Py_VISIT(self->_token_ids);
    Py_VISIT(self->_positions);
    Py_VISIT(self->_workspace);
    int rv = 0;
    return rv;
}

static int32_t CPyDef_MetaNative_clear(PyObject *cpy_r_self)
{
    mypyc_variant___MetaNativeObject *self = (mypyc_variant___MetaNativeObject *)cpy_r_self;
    if (CPyTagged_CheckLong(self->_num_reqs)) {
        CPyTagged __tmp = self->_num_reqs;
        self->_num_reqs = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_num_actual_tokens)) {
        CPyTagged __tmp = self->_num_actual_tokens;
        self->_num_actual_tokens = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_max_query_len)) {
        CPyTagged __tmp = self->_max_query_len;
        self->_max_query_len = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    Py_CLEAR(self->_query_start_loc);
    Py_CLEAR(self->_seq_lens);
    Py_CLEAR(self->_block_table);
    Py_CLEAR(self->_slot_mapping);
    if (CPyTagged_CheckLong(self->_num_decode_tokens)) {
        CPyTagged __tmp = self->_num_decode_tokens;
        self->_num_decode_tokens = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_num_prefill_tokens)) {
        CPyTagged __tmp = self->_num_prefill_tokens;
        self->_num_prefill_tokens = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    Py_CLEAR(self->_num_computed_tokens);
    Py_CLEAR(self->_seq_lens_np);
    if (CPyTagged_CheckLong(self->_max_seq_len)) {
        CPyTagged __tmp = self->_max_seq_len;
        self->_max_seq_len = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    Py_CLEAR(self->_kv_cache_dtype);
    if (CPyTagged_CheckLong(self->_block_size)) {
        CPyTagged __tmp = self->_block_size;
        self->_block_size = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_num_kv_heads)) {
        CPyTagged __tmp = self->_num_kv_heads;
        self->_num_kv_heads = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_head_size)) {
        CPyTagged __tmp = self->_head_size;
        self->_head_size = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_sliding_window)) {
        CPyTagged __tmp = self->_sliding_window;
        self->_sliding_window = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    Py_CLEAR(self->_token_ids);
    Py_CLEAR(self->_positions);
    Py_CLEAR(self->_workspace);
    return 0;
}

static int32_t CPyDef_MetaNative_clear_on_completion(PyObject *cpy_r_self)
{
    mypyc_variant___MetaNativeObject *self = (mypyc_variant___MetaNativeObject *)cpy_r_self;
    if (CPyTagged_CheckLong(self->_num_reqs)) {
        CPyTagged __tmp = self->_num_reqs;
        self->_num_reqs = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_num_actual_tokens)) {
        CPyTagged __tmp = self->_num_actual_tokens;
        self->_num_actual_tokens = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_max_query_len)) {
        CPyTagged __tmp = self->_max_query_len;
        self->_max_query_len = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    Py_CLEAR(self->_query_start_loc);
    Py_CLEAR(self->_seq_lens);
    Py_CLEAR(self->_block_table);
    Py_CLEAR(self->_slot_mapping);
    if (CPyTagged_CheckLong(self->_num_decode_tokens)) {
        CPyTagged __tmp = self->_num_decode_tokens;
        self->_num_decode_tokens = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_num_prefill_tokens)) {
        CPyTagged __tmp = self->_num_prefill_tokens;
        self->_num_prefill_tokens = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    Py_CLEAR(self->_num_computed_tokens);
    Py_CLEAR(self->_seq_lens_np);
    if (CPyTagged_CheckLong(self->_max_seq_len)) {
        CPyTagged __tmp = self->_max_seq_len;
        self->_max_seq_len = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    Py_CLEAR(self->_kv_cache_dtype);
    if (CPyTagged_CheckLong(self->_block_size)) {
        CPyTagged __tmp = self->_block_size;
        self->_block_size = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_num_kv_heads)) {
        CPyTagged __tmp = self->_num_kv_heads;
        self->_num_kv_heads = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_head_size)) {
        CPyTagged __tmp = self->_head_size;
        self->_head_size = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_sliding_window)) {
        CPyTagged __tmp = self->_sliding_window;
        self->_sliding_window = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    Py_CLEAR(self->_token_ids);
    Py_CLEAR(self->_positions);
    Py_CLEAR(self->_workspace);
    return 0;
}

static void
MetaNative_dealloc(mypyc_variant___MetaNativeObject *self)
{
    PyObject_GC_UnTrack(self);
    CPy_TRASHCAN_BEGIN(self, MetaNative_dealloc)
    CPyDef_MetaNative_clear((PyObject *)self);
    Py_TYPE(self)->tp_free((PyObject *)self);
    CPy_TRASHCAN_END(self)
    done: ;
}

PyObject *CPyDef___mypyc__MetaNative_setup(PyObject *cpy_r_type);
PyObject *CPyDef_MetaNative(CPyTagged cpy_r_num_reqs, CPyTagged cpy_r_num_actual_tokens, CPyTagged cpy_r_max_query_len, PyObject *cpy_r_query_start_loc, PyObject *cpy_r_seq_lens, PyObject *cpy_r_block_table, PyObject *cpy_r_slot_mapping, char cpy_r_spec_decode, CPyTagged cpy_r_num_decode_tokens, CPyTagged cpy_r_num_prefill_tokens, PyObject *cpy_r_num_computed_tokens, PyObject *cpy_r_seq_lens_np, CPyTagged cpy_r_max_seq_len, PyObject *cpy_r_kv_cache_dtype, CPyTagged cpy_r_block_size, CPyTagged cpy_r_num_kv_heads, CPyTagged cpy_r_head_size, char cpy_r_causal, CPyTagged cpy_r_sliding_window, PyObject *cpy_r_token_ids, PyObject *cpy_r_positions, PyObject *cpy_r_workspace);

static PyObject *
MetaNative_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (type != CPyType_MetaNative) {
        PyErr_SetString(PyExc_TypeError, "interpreted classes cannot inherit from compiled");
        return NULL;
    }
    PyObject *self = CPyDef___mypyc__MetaNative_setup((PyObject*)type);
    if (self == NULL)
        return NULL;
    PyObject *ret = CPyPy_MetaNative_____init__(self, args, kwds);
    if (ret == NULL) {
            Py_DECREF(self);
            return NULL;
    }
    Py_DECREF(ret);
    return self;
}

static CPyVTableItem MetaNative_vtable[1];
static bool
CPyDef_MetaNative_trait_vtable_setup(void)
{
    CPyVTableItem MetaNative_vtable_scratch[] = {
        (CPyVTableItem)CPyDef_MetaNative_____init__,
    };
    memcpy(MetaNative_vtable, MetaNative_vtable_scratch, sizeof(MetaNative_vtable));
    return 1;
}

static bool
CPyDef_MetaNative_coroutine_setup(PyObject *type)
{
    return 1;
}

static PyObject *
MetaNative_get_num_reqs(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_num_reqs(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_num_actual_tokens(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_num_actual_tokens(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_max_query_len(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_max_query_len(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_query_start_loc(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_query_start_loc(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_seq_lens(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_seq_lens(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_block_table(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_block_table(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_slot_mapping(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_slot_mapping(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_spec_decode(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_spec_decode(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_num_decode_tokens(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_num_decode_tokens(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_num_prefill_tokens(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_num_prefill_tokens(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_num_computed_tokens(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_num_computed_tokens(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_seq_lens_np(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_seq_lens_np(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_max_seq_len(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_max_seq_len(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_kv_cache_dtype(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_kv_cache_dtype(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_block_size(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_block_size(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_num_kv_heads(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_num_kv_heads(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_head_size(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_head_size(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_causal(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_causal(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_sliding_window(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_sliding_window(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_token_ids(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_token_ids(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_positions(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_positions(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);
static PyObject *
MetaNative_get_workspace(mypyc_variant___MetaNativeObject *self, void *closure);
static int
MetaNative_set_workspace(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure);

static PyGetSetDef MetaNative_getseters[] = {
    {"num_reqs",
     (getter)MetaNative_get_num_reqs, (setter)MetaNative_set_num_reqs,
     NULL, NULL},
    {"num_actual_tokens",
     (getter)MetaNative_get_num_actual_tokens, (setter)MetaNative_set_num_actual_tokens,
     NULL, NULL},
    {"max_query_len",
     (getter)MetaNative_get_max_query_len, (setter)MetaNative_set_max_query_len,
     NULL, NULL},
    {"query_start_loc",
     (getter)MetaNative_get_query_start_loc, (setter)MetaNative_set_query_start_loc,
     NULL, NULL},
    {"seq_lens",
     (getter)MetaNative_get_seq_lens, (setter)MetaNative_set_seq_lens,
     NULL, NULL},
    {"block_table",
     (getter)MetaNative_get_block_table, (setter)MetaNative_set_block_table,
     NULL, NULL},
    {"slot_mapping",
     (getter)MetaNative_get_slot_mapping, (setter)MetaNative_set_slot_mapping,
     NULL, NULL},
    {"spec_decode",
     (getter)MetaNative_get_spec_decode, (setter)MetaNative_set_spec_decode,
     NULL, NULL},
    {"num_decode_tokens",
     (getter)MetaNative_get_num_decode_tokens, (setter)MetaNative_set_num_decode_tokens,
     NULL, NULL},
    {"num_prefill_tokens",
     (getter)MetaNative_get_num_prefill_tokens, (setter)MetaNative_set_num_prefill_tokens,
     NULL, NULL},
    {"num_computed_tokens",
     (getter)MetaNative_get_num_computed_tokens, (setter)MetaNative_set_num_computed_tokens,
     NULL, NULL},
    {"seq_lens_np",
     (getter)MetaNative_get_seq_lens_np, (setter)MetaNative_set_seq_lens_np,
     NULL, NULL},
    {"max_seq_len",
     (getter)MetaNative_get_max_seq_len, (setter)MetaNative_set_max_seq_len,
     NULL, NULL},
    {"kv_cache_dtype",
     (getter)MetaNative_get_kv_cache_dtype, (setter)MetaNative_set_kv_cache_dtype,
     NULL, NULL},
    {"block_size",
     (getter)MetaNative_get_block_size, (setter)MetaNative_set_block_size,
     NULL, NULL},
    {"num_kv_heads",
     (getter)MetaNative_get_num_kv_heads, (setter)MetaNative_set_num_kv_heads,
     NULL, NULL},
    {"head_size",
     (getter)MetaNative_get_head_size, (setter)MetaNative_set_head_size,
     NULL, NULL},
    {"causal",
     (getter)MetaNative_get_causal, (setter)MetaNative_set_causal,
     NULL, NULL},
    {"sliding_window",
     (getter)MetaNative_get_sliding_window, (setter)MetaNative_set_sliding_window,
     NULL, NULL},
    {"token_ids",
     (getter)MetaNative_get_token_ids, (setter)MetaNative_set_token_ids,
     NULL, NULL},
    {"positions",
     (getter)MetaNative_get_positions, (setter)MetaNative_set_positions,
     NULL, NULL},
    {"workspace",
     (getter)MetaNative_get_workspace, (setter)MetaNative_set_workspace,
     NULL, NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef MetaNative_methods[] = {
    {"__internal_mypyc_setup", (PyCFunction)CPyDef___mypyc__MetaNative_setup, METH_O, NULL},
    {"__init__",
     (PyCFunction)CPyPy_MetaNative_____init__,
     METH_FASTCALL | METH_KEYWORDS, PyDoc_STR("__init__($self, num_reqs, num_actual_tokens, max_query_len, query_start_loc, seq_lens, block_table, slot_mapping, spec_decode, num_decode_tokens, num_prefill_tokens, num_computed_tokens, seq_lens_np, max_seq_len, kv_cache_dtype, block_size, num_kv_heads, head_size, causal, sliding_window, token_ids, positions, workspace)\n--\n\n")},
    {"__setstate__", (PyCFunction)CPyPickle_SetState, METH_O, NULL},
    {"__getstate__", (PyCFunction)CPyPickle_GetState, METH_NOARGS, NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject CPyType_MetaNative_template_ = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "MetaNative",
    .tp_new = MetaNative_new,
    .tp_dealloc = (destructor)MetaNative_dealloc,
    .tp_traverse = (traverseproc)MetaNative_traverse,
    .tp_clear = (inquiry)CPyDef_MetaNative_clear,
    .tp_getset = MetaNative_getseters,
    .tp_methods = MetaNative_methods,
    .tp_init = MetaNative_init,
    .tp_basicsize = sizeof(mypyc_variant___MetaNativeObject),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_doc = PyDoc_STR("MetaNative(num_reqs, num_actual_tokens, max_query_len, query_start_loc, seq_lens, block_table, slot_mapping, spec_decode, num_decode_tokens, num_prefill_tokens, num_computed_tokens, seq_lens_np, max_seq_len, kv_cache_dtype, block_size, num_kv_heads, head_size, causal, sliding_window, token_ids, positions, workspace)\n--\n\n"),
};
static PyTypeObject *CPyType_MetaNative_template = &CPyType_MetaNative_template_;

PyObject *CPyDef___mypyc__MetaNative_setup(PyObject *cpy_r_type)
{
    PyTypeObject *type = (PyTypeObject*)cpy_r_type;
    mypyc_variant___MetaNativeObject *self;
    self = (mypyc_variant___MetaNativeObject *)type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;
    self->vtable = MetaNative_vtable;
    self->_num_reqs = CPY_INT_TAG;
    self->_num_actual_tokens = CPY_INT_TAG;
    self->_max_query_len = CPY_INT_TAG;
    self->_spec_decode = 2;
    self->_num_decode_tokens = CPY_INT_TAG;
    self->_num_prefill_tokens = CPY_INT_TAG;
    self->_max_seq_len = CPY_INT_TAG;
    self->_block_size = CPY_INT_TAG;
    self->_num_kv_heads = CPY_INT_TAG;
    self->_head_size = CPY_INT_TAG;
    self->_causal = 2;
    self->_sliding_window = CPY_INT_TAG;
    return (PyObject *)self;
}

PyObject *CPyDef_MetaNative(CPyTagged cpy_r_num_reqs, CPyTagged cpy_r_num_actual_tokens, CPyTagged cpy_r_max_query_len, PyObject *cpy_r_query_start_loc, PyObject *cpy_r_seq_lens, PyObject *cpy_r_block_table, PyObject *cpy_r_slot_mapping, char cpy_r_spec_decode, CPyTagged cpy_r_num_decode_tokens, CPyTagged cpy_r_num_prefill_tokens, PyObject *cpy_r_num_computed_tokens, PyObject *cpy_r_seq_lens_np, CPyTagged cpy_r_max_seq_len, PyObject *cpy_r_kv_cache_dtype, CPyTagged cpy_r_block_size, CPyTagged cpy_r_num_kv_heads, CPyTagged cpy_r_head_size, char cpy_r_causal, CPyTagged cpy_r_sliding_window, PyObject *cpy_r_token_ids, PyObject *cpy_r_positions, PyObject *cpy_r_workspace)
{
    PyObject *self = CPyDef___mypyc__MetaNative_setup((PyObject *)CPyType_MetaNative);
    if (self == NULL)
        return NULL;
    char res = CPyDef_MetaNative_____init__(self, cpy_r_num_reqs, cpy_r_num_actual_tokens, cpy_r_max_query_len, cpy_r_query_start_loc, cpy_r_seq_lens, cpy_r_block_table, cpy_r_slot_mapping, cpy_r_spec_decode, cpy_r_num_decode_tokens, cpy_r_num_prefill_tokens, cpy_r_num_computed_tokens, cpy_r_seq_lens_np, cpy_r_max_seq_len, cpy_r_kv_cache_dtype, cpy_r_block_size, cpy_r_num_kv_heads, cpy_r_head_size, cpy_r_causal, cpy_r_sliding_window, cpy_r_token_ids, cpy_r_positions, cpy_r_workspace);
    if (res == 2) {
        Py_DECREF(self);
        return NULL;
    }
    return self;
}

static PyObject *
MetaNative_get_num_reqs(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_num_reqs == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'num_reqs' of 'MetaNative' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_num_reqs);
    PyObject *retval = CPyTagged_StealAsObject(self->_num_reqs);
    return retval;
}

static int
MetaNative_set_num_reqs(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'num_reqs' cannot be deleted");
        return -1;
    }
    if (self->_num_reqs != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_num_reqs);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_num_reqs = tmp;
    return 0;
}

static PyObject *
MetaNative_get_num_actual_tokens(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_num_actual_tokens == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'num_actual_tokens' of 'MetaNative' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_num_actual_tokens);
    PyObject *retval = CPyTagged_StealAsObject(self->_num_actual_tokens);
    return retval;
}

static int
MetaNative_set_num_actual_tokens(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'num_actual_tokens' cannot be deleted");
        return -1;
    }
    if (self->_num_actual_tokens != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_num_actual_tokens);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_num_actual_tokens = tmp;
    return 0;
}

static PyObject *
MetaNative_get_max_query_len(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_max_query_len == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'max_query_len' of 'MetaNative' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_max_query_len);
    PyObject *retval = CPyTagged_StealAsObject(self->_max_query_len);
    return retval;
}

static int
MetaNative_set_max_query_len(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'max_query_len' cannot be deleted");
        return -1;
    }
    if (self->_max_query_len != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_max_query_len);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_max_query_len = tmp;
    return 0;
}

static PyObject *
MetaNative_get_query_start_loc(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_query_start_loc == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'query_start_loc' of 'MetaNative' undefined");
        return NULL;
    }
    CPy_INCREF(self->_query_start_loc);
    PyObject *retval = self->_query_start_loc;
    return retval;
}

static int
MetaNative_set_query_start_loc(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'query_start_loc' cannot be deleted");
        return -1;
    }
    if (self->_query_start_loc != NULL) {
        CPy_DECREF(self->_query_start_loc);
    }
    PyObject *tmp = value;
    CPy_INCREF(tmp);
    self->_query_start_loc = tmp;
    return 0;
}

static PyObject *
MetaNative_get_seq_lens(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_seq_lens == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'seq_lens' of 'MetaNative' undefined");
        return NULL;
    }
    CPy_INCREF(self->_seq_lens);
    PyObject *retval = self->_seq_lens;
    return retval;
}

static int
MetaNative_set_seq_lens(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'seq_lens' cannot be deleted");
        return -1;
    }
    if (self->_seq_lens != NULL) {
        CPy_DECREF(self->_seq_lens);
    }
    PyObject *tmp = value;
    CPy_INCREF(tmp);
    self->_seq_lens = tmp;
    return 0;
}

static PyObject *
MetaNative_get_block_table(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_block_table == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'block_table' of 'MetaNative' undefined");
        return NULL;
    }
    CPy_INCREF(self->_block_table);
    PyObject *retval = self->_block_table;
    return retval;
}

static int
MetaNative_set_block_table(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'block_table' cannot be deleted");
        return -1;
    }
    if (self->_block_table != NULL) {
        CPy_DECREF(self->_block_table);
    }
    PyObject *tmp = value;
    CPy_INCREF(tmp);
    self->_block_table = tmp;
    return 0;
}

static PyObject *
MetaNative_get_slot_mapping(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_slot_mapping == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'slot_mapping' of 'MetaNative' undefined");
        return NULL;
    }
    CPy_INCREF(self->_slot_mapping);
    PyObject *retval = self->_slot_mapping;
    return retval;
}

static int
MetaNative_set_slot_mapping(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'slot_mapping' cannot be deleted");
        return -1;
    }
    if (self->_slot_mapping != NULL) {
        CPy_DECREF(self->_slot_mapping);
    }
    PyObject *tmp = value;
    CPy_INCREF(tmp);
    self->_slot_mapping = tmp;
    return 0;
}

static PyObject *
MetaNative_get_spec_decode(mypyc_variant___MetaNativeObject *self, void *closure)
{
    PyObject *retval = self->_spec_decode ? Py_True : Py_False;
    CPy_INCREF(retval);
    return retval;
}

static int
MetaNative_set_spec_decode(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'spec_decode' cannot be deleted");
        return -1;
    }
    char tmp;
    if (unlikely(!PyBool_Check(value))) {
        CPy_TypeError("bool", value); return -1;
    } else
        tmp = value == Py_True;
    self->_spec_decode = tmp;
    return 0;
}

static PyObject *
MetaNative_get_num_decode_tokens(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_num_decode_tokens == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'num_decode_tokens' of 'MetaNative' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_num_decode_tokens);
    PyObject *retval = CPyTagged_StealAsObject(self->_num_decode_tokens);
    return retval;
}

static int
MetaNative_set_num_decode_tokens(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'num_decode_tokens' cannot be deleted");
        return -1;
    }
    if (self->_num_decode_tokens != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_num_decode_tokens);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_num_decode_tokens = tmp;
    return 0;
}

static PyObject *
MetaNative_get_num_prefill_tokens(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_num_prefill_tokens == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'num_prefill_tokens' of 'MetaNative' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_num_prefill_tokens);
    PyObject *retval = CPyTagged_StealAsObject(self->_num_prefill_tokens);
    return retval;
}

static int
MetaNative_set_num_prefill_tokens(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'num_prefill_tokens' cannot be deleted");
        return -1;
    }
    if (self->_num_prefill_tokens != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_num_prefill_tokens);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_num_prefill_tokens = tmp;
    return 0;
}

static PyObject *
MetaNative_get_num_computed_tokens(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_num_computed_tokens == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'num_computed_tokens' of 'MetaNative' undefined");
        return NULL;
    }
    CPy_INCREF(self->_num_computed_tokens);
    PyObject *retval = self->_num_computed_tokens;
    return retval;
}

static int
MetaNative_set_num_computed_tokens(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'num_computed_tokens' cannot be deleted");
        return -1;
    }
    if (self->_num_computed_tokens != NULL) {
        CPy_DECREF(self->_num_computed_tokens);
    }
    PyObject *tmp = value;
    CPy_INCREF(tmp);
    self->_num_computed_tokens = tmp;
    return 0;
}

static PyObject *
MetaNative_get_seq_lens_np(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_seq_lens_np == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'seq_lens_np' of 'MetaNative' undefined");
        return NULL;
    }
    CPy_INCREF(self->_seq_lens_np);
    PyObject *retval = self->_seq_lens_np;
    return retval;
}

static int
MetaNative_set_seq_lens_np(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'seq_lens_np' cannot be deleted");
        return -1;
    }
    if (self->_seq_lens_np != NULL) {
        CPy_DECREF(self->_seq_lens_np);
    }
    PyObject *tmp = value;
    CPy_INCREF(tmp);
    self->_seq_lens_np = tmp;
    return 0;
}

static PyObject *
MetaNative_get_max_seq_len(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_max_seq_len == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'max_seq_len' of 'MetaNative' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_max_seq_len);
    PyObject *retval = CPyTagged_StealAsObject(self->_max_seq_len);
    return retval;
}

static int
MetaNative_set_max_seq_len(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'max_seq_len' cannot be deleted");
        return -1;
    }
    if (self->_max_seq_len != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_max_seq_len);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_max_seq_len = tmp;
    return 0;
}

static PyObject *
MetaNative_get_kv_cache_dtype(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_kv_cache_dtype == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'kv_cache_dtype' of 'MetaNative' undefined");
        return NULL;
    }
    CPy_INCREF(self->_kv_cache_dtype);
    PyObject *retval = self->_kv_cache_dtype;
    return retval;
}

static int
MetaNative_set_kv_cache_dtype(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'kv_cache_dtype' cannot be deleted");
        return -1;
    }
    if (self->_kv_cache_dtype != NULL) {
        CPy_DECREF(self->_kv_cache_dtype);
    }
    PyObject *tmp;
    if (likely(PyUnicode_Check(value)))
        tmp = value;
    else {
        CPy_TypeError("str", value); 
        tmp = NULL;
    }
    if (!tmp)
        return -1;
    CPy_INCREF(tmp);
    self->_kv_cache_dtype = tmp;
    return 0;
}

static PyObject *
MetaNative_get_block_size(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_block_size == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'block_size' of 'MetaNative' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_block_size);
    PyObject *retval = CPyTagged_StealAsObject(self->_block_size);
    return retval;
}

static int
MetaNative_set_block_size(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'block_size' cannot be deleted");
        return -1;
    }
    if (self->_block_size != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_block_size);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_block_size = tmp;
    return 0;
}

static PyObject *
MetaNative_get_num_kv_heads(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_num_kv_heads == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'num_kv_heads' of 'MetaNative' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_num_kv_heads);
    PyObject *retval = CPyTagged_StealAsObject(self->_num_kv_heads);
    return retval;
}

static int
MetaNative_set_num_kv_heads(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'num_kv_heads' cannot be deleted");
        return -1;
    }
    if (self->_num_kv_heads != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_num_kv_heads);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_num_kv_heads = tmp;
    return 0;
}

static PyObject *
MetaNative_get_head_size(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_head_size == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'head_size' of 'MetaNative' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_head_size);
    PyObject *retval = CPyTagged_StealAsObject(self->_head_size);
    return retval;
}

static int
MetaNative_set_head_size(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'head_size' cannot be deleted");
        return -1;
    }
    if (self->_head_size != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_head_size);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_head_size = tmp;
    return 0;
}

static PyObject *
MetaNative_get_causal(mypyc_variant___MetaNativeObject *self, void *closure)
{
    PyObject *retval = self->_causal ? Py_True : Py_False;
    CPy_INCREF(retval);
    return retval;
}

static int
MetaNative_set_causal(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'causal' cannot be deleted");
        return -1;
    }
    char tmp;
    if (unlikely(!PyBool_Check(value))) {
        CPy_TypeError("bool", value); return -1;
    } else
        tmp = value == Py_True;
    self->_causal = tmp;
    return 0;
}

static PyObject *
MetaNative_get_sliding_window(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_sliding_window == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'sliding_window' of 'MetaNative' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_sliding_window);
    PyObject *retval = CPyTagged_StealAsObject(self->_sliding_window);
    return retval;
}

static int
MetaNative_set_sliding_window(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'sliding_window' cannot be deleted");
        return -1;
    }
    if (self->_sliding_window != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_sliding_window);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_sliding_window = tmp;
    return 0;
}

static PyObject *
MetaNative_get_token_ids(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_token_ids == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'token_ids' of 'MetaNative' undefined");
        return NULL;
    }
    CPy_INCREF(self->_token_ids);
    PyObject *retval = self->_token_ids;
    return retval;
}

static int
MetaNative_set_token_ids(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'token_ids' cannot be deleted");
        return -1;
    }
    if (self->_token_ids != NULL) {
        CPy_DECREF(self->_token_ids);
    }
    PyObject *tmp = value;
    CPy_INCREF(tmp);
    self->_token_ids = tmp;
    return 0;
}

static PyObject *
MetaNative_get_positions(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_positions == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'positions' of 'MetaNative' undefined");
        return NULL;
    }
    CPy_INCREF(self->_positions);
    PyObject *retval = self->_positions;
    return retval;
}

static int
MetaNative_set_positions(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'positions' cannot be deleted");
        return -1;
    }
    if (self->_positions != NULL) {
        CPy_DECREF(self->_positions);
    }
    PyObject *tmp = value;
    CPy_INCREF(tmp);
    self->_positions = tmp;
    return 0;
}

static PyObject *
MetaNative_get_workspace(mypyc_variant___MetaNativeObject *self, void *closure)
{
    if (unlikely(self->_workspace == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'workspace' of 'MetaNative' undefined");
        return NULL;
    }
    CPy_INCREF(self->_workspace);
    PyObject *retval = self->_workspace;
    return retval;
}

static int
MetaNative_set_workspace(mypyc_variant___MetaNativeObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaNative' object attribute 'workspace' cannot be deleted");
        return -1;
    }
    if (self->_workspace != NULL) {
        CPy_DECREF(self->_workspace);
    }
    PyObject *tmp = value;
    CPy_INCREF(tmp);
    self->_workspace = tmp;
    return 0;
}

static int
MetaDataclass_traverse(mypyc_variant___MetaDataclassObject *self, visitproc visit, void *arg)
{
    if (CPyTagged_CheckLong(self->_num_reqs)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_num_reqs));
    }
    if (CPyTagged_CheckLong(self->_num_actual_tokens)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_num_actual_tokens));
    }
    if (CPyTagged_CheckLong(self->_max_query_len)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_max_query_len));
    }
    Py_VISIT(self->_query_start_loc);
    Py_VISIT(self->_seq_lens);
    Py_VISIT(self->_block_table);
    Py_VISIT(self->_slot_mapping);
    if (CPyTagged_CheckLong(self->_num_decode_tokens)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_num_decode_tokens));
    }
    if (CPyTagged_CheckLong(self->_num_prefill_tokens)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_num_prefill_tokens));
    }
    Py_VISIT(self->_num_computed_tokens);
    Py_VISIT(self->_seq_lens_np);
    if (CPyTagged_CheckLong(self->_max_seq_len)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_max_seq_len));
    }
    Py_VISIT(self->_kv_cache_dtype);
    if (CPyTagged_CheckLong(self->_block_size)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_block_size));
    }
    if (CPyTagged_CheckLong(self->_num_kv_heads)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_num_kv_heads));
    }
    if (CPyTagged_CheckLong(self->_head_size)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_head_size));
    }
    if (CPyTagged_CheckLong(self->_sliding_window)) {
        Py_VISIT(CPyTagged_LongAsObject(self->_sliding_window));
    }
    Py_VISIT(self->_token_ids);
    Py_VISIT(self->_positions);
    Py_VISIT(self->_workspace);
    int rv = 0;
    return rv;
}

static int32_t CPyDef_MetaDataclass_clear(PyObject *cpy_r_self)
{
    mypyc_variant___MetaDataclassObject *self = (mypyc_variant___MetaDataclassObject *)cpy_r_self;
    if (CPyTagged_CheckLong(self->_num_reqs)) {
        CPyTagged __tmp = self->_num_reqs;
        self->_num_reqs = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_num_actual_tokens)) {
        CPyTagged __tmp = self->_num_actual_tokens;
        self->_num_actual_tokens = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_max_query_len)) {
        CPyTagged __tmp = self->_max_query_len;
        self->_max_query_len = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    Py_CLEAR(self->_query_start_loc);
    Py_CLEAR(self->_seq_lens);
    Py_CLEAR(self->_block_table);
    Py_CLEAR(self->_slot_mapping);
    if (CPyTagged_CheckLong(self->_num_decode_tokens)) {
        CPyTagged __tmp = self->_num_decode_tokens;
        self->_num_decode_tokens = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_num_prefill_tokens)) {
        CPyTagged __tmp = self->_num_prefill_tokens;
        self->_num_prefill_tokens = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    Py_CLEAR(self->_num_computed_tokens);
    Py_CLEAR(self->_seq_lens_np);
    if (CPyTagged_CheckLong(self->_max_seq_len)) {
        CPyTagged __tmp = self->_max_seq_len;
        self->_max_seq_len = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    Py_CLEAR(self->_kv_cache_dtype);
    if (CPyTagged_CheckLong(self->_block_size)) {
        CPyTagged __tmp = self->_block_size;
        self->_block_size = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_num_kv_heads)) {
        CPyTagged __tmp = self->_num_kv_heads;
        self->_num_kv_heads = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_head_size)) {
        CPyTagged __tmp = self->_head_size;
        self->_head_size = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_sliding_window)) {
        CPyTagged __tmp = self->_sliding_window;
        self->_sliding_window = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    Py_CLEAR(self->_token_ids);
    Py_CLEAR(self->_positions);
    Py_CLEAR(self->_workspace);
    return 0;
}

static int32_t CPyDef_MetaDataclass_clear_on_completion(PyObject *cpy_r_self)
{
    mypyc_variant___MetaDataclassObject *self = (mypyc_variant___MetaDataclassObject *)cpy_r_self;
    if (CPyTagged_CheckLong(self->_num_reqs)) {
        CPyTagged __tmp = self->_num_reqs;
        self->_num_reqs = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_num_actual_tokens)) {
        CPyTagged __tmp = self->_num_actual_tokens;
        self->_num_actual_tokens = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_max_query_len)) {
        CPyTagged __tmp = self->_max_query_len;
        self->_max_query_len = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    Py_CLEAR(self->_query_start_loc);
    Py_CLEAR(self->_seq_lens);
    Py_CLEAR(self->_block_table);
    Py_CLEAR(self->_slot_mapping);
    if (CPyTagged_CheckLong(self->_num_decode_tokens)) {
        CPyTagged __tmp = self->_num_decode_tokens;
        self->_num_decode_tokens = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_num_prefill_tokens)) {
        CPyTagged __tmp = self->_num_prefill_tokens;
        self->_num_prefill_tokens = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    Py_CLEAR(self->_num_computed_tokens);
    Py_CLEAR(self->_seq_lens_np);
    if (CPyTagged_CheckLong(self->_max_seq_len)) {
        CPyTagged __tmp = self->_max_seq_len;
        self->_max_seq_len = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    Py_CLEAR(self->_kv_cache_dtype);
    if (CPyTagged_CheckLong(self->_block_size)) {
        CPyTagged __tmp = self->_block_size;
        self->_block_size = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_num_kv_heads)) {
        CPyTagged __tmp = self->_num_kv_heads;
        self->_num_kv_heads = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_head_size)) {
        CPyTagged __tmp = self->_head_size;
        self->_head_size = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    if (CPyTagged_CheckLong(self->_sliding_window)) {
        CPyTagged __tmp = self->_sliding_window;
        self->_sliding_window = CPY_INT_TAG;
        Py_XDECREF(CPyTagged_LongAsObject(__tmp));
    }
    Py_CLEAR(self->_token_ids);
    Py_CLEAR(self->_positions);
    Py_CLEAR(self->_workspace);
    return 0;
}

static void
MetaDataclass_dealloc(mypyc_variant___MetaDataclassObject *self)
{
    PyObject_GC_UnTrack(self);
    CPy_TRASHCAN_BEGIN(self, MetaDataclass_dealloc)
    CPyDef_MetaDataclass_clear((PyObject *)self);
    Py_TYPE(self)->tp_free((PyObject *)self);
    CPy_TRASHCAN_END(self)
    done: ;
}

PyObject *CPyDef___mypyc__MetaDataclass_setup(PyObject *cpy_r_type);
PyObject *CPyDef_MetaDataclass(PyObject *cpy_r_args, PyObject *cpy_r_kwargs);

static PyObject *
MetaDataclass_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (type != CPyType_MetaDataclass) {
        PyErr_SetString(PyExc_TypeError, "interpreted classes cannot inherit from compiled");
        return NULL;
    }
    PyObject *self = CPyDef___mypyc__MetaDataclass_setup((PyObject*)type);
    if (self == NULL)
        return NULL;
    return self;
}

static CPyVTableItem MetaDataclass_vtable[1];
static bool
CPyDef_MetaDataclass_trait_vtable_setup(void)
{
    CPyVTableItem MetaDataclass_vtable_scratch[] = {
        NULL
    };
    memcpy(MetaDataclass_vtable, MetaDataclass_vtable_scratch, sizeof(MetaDataclass_vtable));
    return 1;
}

static bool
CPyDef_MetaDataclass_coroutine_setup(PyObject *type)
{
    return 1;
}

static PyObject *
MetaDataclass_get_num_reqs(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_num_reqs(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_num_actual_tokens(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_num_actual_tokens(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_max_query_len(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_max_query_len(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_query_start_loc(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_query_start_loc(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_seq_lens(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_seq_lens(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_block_table(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_block_table(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_slot_mapping(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_slot_mapping(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_spec_decode(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_spec_decode(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_num_decode_tokens(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_num_decode_tokens(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_num_prefill_tokens(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_num_prefill_tokens(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_num_computed_tokens(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_num_computed_tokens(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_seq_lens_np(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_seq_lens_np(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_max_seq_len(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_max_seq_len(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_kv_cache_dtype(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_kv_cache_dtype(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_block_size(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_block_size(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_num_kv_heads(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_num_kv_heads(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_head_size(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_head_size(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_causal(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_causal(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_sliding_window(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_sliding_window(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_token_ids(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_token_ids(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_positions(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_positions(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);
static PyObject *
MetaDataclass_get_workspace(mypyc_variant___MetaDataclassObject *self, void *closure);
static int
MetaDataclass_set_workspace(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure);

static PyGetSetDef MetaDataclass_getseters[] = {
    {"num_reqs",
     (getter)MetaDataclass_get_num_reqs, (setter)MetaDataclass_set_num_reqs,
     NULL, NULL},
    {"num_actual_tokens",
     (getter)MetaDataclass_get_num_actual_tokens, (setter)MetaDataclass_set_num_actual_tokens,
     NULL, NULL},
    {"max_query_len",
     (getter)MetaDataclass_get_max_query_len, (setter)MetaDataclass_set_max_query_len,
     NULL, NULL},
    {"query_start_loc",
     (getter)MetaDataclass_get_query_start_loc, (setter)MetaDataclass_set_query_start_loc,
     NULL, NULL},
    {"seq_lens",
     (getter)MetaDataclass_get_seq_lens, (setter)MetaDataclass_set_seq_lens,
     NULL, NULL},
    {"block_table",
     (getter)MetaDataclass_get_block_table, (setter)MetaDataclass_set_block_table,
     NULL, NULL},
    {"slot_mapping",
     (getter)MetaDataclass_get_slot_mapping, (setter)MetaDataclass_set_slot_mapping,
     NULL, NULL},
    {"spec_decode",
     (getter)MetaDataclass_get_spec_decode, (setter)MetaDataclass_set_spec_decode,
     NULL, NULL},
    {"num_decode_tokens",
     (getter)MetaDataclass_get_num_decode_tokens, (setter)MetaDataclass_set_num_decode_tokens,
     NULL, NULL},
    {"num_prefill_tokens",
     (getter)MetaDataclass_get_num_prefill_tokens, (setter)MetaDataclass_set_num_prefill_tokens,
     NULL, NULL},
    {"num_computed_tokens",
     (getter)MetaDataclass_get_num_computed_tokens, (setter)MetaDataclass_set_num_computed_tokens,
     NULL, NULL},
    {"seq_lens_np",
     (getter)MetaDataclass_get_seq_lens_np, (setter)MetaDataclass_set_seq_lens_np,
     NULL, NULL},
    {"max_seq_len",
     (getter)MetaDataclass_get_max_seq_len, (setter)MetaDataclass_set_max_seq_len,
     NULL, NULL},
    {"kv_cache_dtype",
     (getter)MetaDataclass_get_kv_cache_dtype, (setter)MetaDataclass_set_kv_cache_dtype,
     NULL, NULL},
    {"block_size",
     (getter)MetaDataclass_get_block_size, (setter)MetaDataclass_set_block_size,
     NULL, NULL},
    {"num_kv_heads",
     (getter)MetaDataclass_get_num_kv_heads, (setter)MetaDataclass_set_num_kv_heads,
     NULL, NULL},
    {"head_size",
     (getter)MetaDataclass_get_head_size, (setter)MetaDataclass_set_head_size,
     NULL, NULL},
    {"causal",
     (getter)MetaDataclass_get_causal, (setter)MetaDataclass_set_causal,
     NULL, NULL},
    {"sliding_window",
     (getter)MetaDataclass_get_sliding_window, (setter)MetaDataclass_set_sliding_window,
     NULL, NULL},
    {"token_ids",
     (getter)MetaDataclass_get_token_ids, (setter)MetaDataclass_set_token_ids,
     NULL, NULL},
    {"positions",
     (getter)MetaDataclass_get_positions, (setter)MetaDataclass_set_positions,
     NULL, NULL},
    {"workspace",
     (getter)MetaDataclass_get_workspace, (setter)MetaDataclass_set_workspace,
     NULL, NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef MetaDataclass_methods[] = {
    {"__internal_mypyc_setup", (PyCFunction)CPyDef___mypyc__MetaDataclass_setup, METH_O, NULL},
    {"__setstate__", (PyCFunction)CPyPickle_SetState, METH_O, NULL},
    {"__getstate__", (PyCFunction)CPyPickle_GetState, METH_NOARGS, NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject CPyType_MetaDataclass_template_ = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "MetaDataclass",
    .tp_new = MetaDataclass_new,
    .tp_dealloc = (destructor)MetaDataclass_dealloc,
    .tp_traverse = (traverseproc)MetaDataclass_traverse,
    .tp_clear = (inquiry)CPyDef_MetaDataclass_clear,
    .tp_getset = MetaDataclass_getseters,
    .tp_methods = MetaDataclass_methods,
    .tp_basicsize = sizeof(mypyc_variant___MetaDataclassObject),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_doc = PyDoc_STR("MetaDataclass()\n--\n\n"),
};
static PyTypeObject *CPyType_MetaDataclass_template = &CPyType_MetaDataclass_template_;

PyObject *CPyDef___mypyc__MetaDataclass_setup(PyObject *cpy_r_type)
{
    PyTypeObject *type = (PyTypeObject*)cpy_r_type;
    mypyc_variant___MetaDataclassObject *self;
    self = (mypyc_variant___MetaDataclassObject *)type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;
    self->vtable = MetaDataclass_vtable;
    self->_num_reqs = CPY_INT_TAG;
    self->_num_actual_tokens = CPY_INT_TAG;
    self->_max_query_len = CPY_INT_TAG;
    self->_spec_decode = 2;
    self->_num_decode_tokens = CPY_INT_TAG;
    self->_num_prefill_tokens = CPY_INT_TAG;
    self->_max_seq_len = CPY_INT_TAG;
    self->_block_size = CPY_INT_TAG;
    self->_num_kv_heads = CPY_INT_TAG;
    self->_head_size = CPY_INT_TAG;
    self->_causal = 2;
    self->_sliding_window = CPY_INT_TAG;
    return (PyObject *)self;
}

PyObject *CPyDef_MetaDataclass(PyObject *cpy_r_args, PyObject *cpy_r_kwargs)
{
    PyObject *self = CPyDef___mypyc__MetaDataclass_setup((PyObject *)CPyType_MetaDataclass);
    if (self == NULL)
        return NULL;
    int res = CPyType_MetaDataclass->tp_init(self, cpy_r_args, cpy_r_kwargs);
    if (res < 0) {
        Py_DECREF(self);
        return NULL;
    }
    return self;
}

static PyObject *
MetaDataclass_get_num_reqs(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_num_reqs == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'num_reqs' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_num_reqs);
    PyObject *retval = CPyTagged_StealAsObject(self->_num_reqs);
    return retval;
}

static int
MetaDataclass_set_num_reqs(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'num_reqs' cannot be deleted");
        return -1;
    }
    if (self->_num_reqs != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_num_reqs);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_num_reqs = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_num_actual_tokens(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_num_actual_tokens == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'num_actual_tokens' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_num_actual_tokens);
    PyObject *retval = CPyTagged_StealAsObject(self->_num_actual_tokens);
    return retval;
}

static int
MetaDataclass_set_num_actual_tokens(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'num_actual_tokens' cannot be deleted");
        return -1;
    }
    if (self->_num_actual_tokens != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_num_actual_tokens);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_num_actual_tokens = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_max_query_len(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_max_query_len == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'max_query_len' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_max_query_len);
    PyObject *retval = CPyTagged_StealAsObject(self->_max_query_len);
    return retval;
}

static int
MetaDataclass_set_max_query_len(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'max_query_len' cannot be deleted");
        return -1;
    }
    if (self->_max_query_len != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_max_query_len);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_max_query_len = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_query_start_loc(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_query_start_loc == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'query_start_loc' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPy_INCREF(self->_query_start_loc);
    PyObject *retval = self->_query_start_loc;
    return retval;
}

static int
MetaDataclass_set_query_start_loc(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'query_start_loc' cannot be deleted");
        return -1;
    }
    if (self->_query_start_loc != NULL) {
        CPy_DECREF(self->_query_start_loc);
    }
    PyObject *tmp = value;
    CPy_INCREF(tmp);
    self->_query_start_loc = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_seq_lens(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_seq_lens == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'seq_lens' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPy_INCREF(self->_seq_lens);
    PyObject *retval = self->_seq_lens;
    return retval;
}

static int
MetaDataclass_set_seq_lens(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'seq_lens' cannot be deleted");
        return -1;
    }
    if (self->_seq_lens != NULL) {
        CPy_DECREF(self->_seq_lens);
    }
    PyObject *tmp = value;
    CPy_INCREF(tmp);
    self->_seq_lens = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_block_table(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_block_table == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'block_table' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPy_INCREF(self->_block_table);
    PyObject *retval = self->_block_table;
    return retval;
}

static int
MetaDataclass_set_block_table(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'block_table' cannot be deleted");
        return -1;
    }
    if (self->_block_table != NULL) {
        CPy_DECREF(self->_block_table);
    }
    PyObject *tmp = value;
    CPy_INCREF(tmp);
    self->_block_table = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_slot_mapping(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_slot_mapping == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'slot_mapping' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPy_INCREF(self->_slot_mapping);
    PyObject *retval = self->_slot_mapping;
    return retval;
}

static int
MetaDataclass_set_slot_mapping(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'slot_mapping' cannot be deleted");
        return -1;
    }
    if (self->_slot_mapping != NULL) {
        CPy_DECREF(self->_slot_mapping);
    }
    PyObject *tmp = value;
    CPy_INCREF(tmp);
    self->_slot_mapping = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_spec_decode(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_spec_decode == 2)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'spec_decode' of 'MetaDataclass' undefined");
        return NULL;
    }
    PyObject *retval = self->_spec_decode ? Py_True : Py_False;
    CPy_INCREF(retval);
    return retval;
}

static int
MetaDataclass_set_spec_decode(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'spec_decode' cannot be deleted");
        return -1;
    }
    char tmp;
    if (unlikely(!PyBool_Check(value))) {
        CPy_TypeError("bool", value); return -1;
    } else
        tmp = value == Py_True;
    self->_spec_decode = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_num_decode_tokens(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_num_decode_tokens == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'num_decode_tokens' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_num_decode_tokens);
    PyObject *retval = CPyTagged_StealAsObject(self->_num_decode_tokens);
    return retval;
}

static int
MetaDataclass_set_num_decode_tokens(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'num_decode_tokens' cannot be deleted");
        return -1;
    }
    if (self->_num_decode_tokens != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_num_decode_tokens);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_num_decode_tokens = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_num_prefill_tokens(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_num_prefill_tokens == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'num_prefill_tokens' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_num_prefill_tokens);
    PyObject *retval = CPyTagged_StealAsObject(self->_num_prefill_tokens);
    return retval;
}

static int
MetaDataclass_set_num_prefill_tokens(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'num_prefill_tokens' cannot be deleted");
        return -1;
    }
    if (self->_num_prefill_tokens != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_num_prefill_tokens);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_num_prefill_tokens = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_num_computed_tokens(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_num_computed_tokens == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'num_computed_tokens' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPy_INCREF(self->_num_computed_tokens);
    PyObject *retval = self->_num_computed_tokens;
    return retval;
}

static int
MetaDataclass_set_num_computed_tokens(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'num_computed_tokens' cannot be deleted");
        return -1;
    }
    if (self->_num_computed_tokens != NULL) {
        CPy_DECREF(self->_num_computed_tokens);
    }
    PyObject *tmp = value;
    CPy_INCREF(tmp);
    self->_num_computed_tokens = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_seq_lens_np(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_seq_lens_np == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'seq_lens_np' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPy_INCREF(self->_seq_lens_np);
    PyObject *retval = self->_seq_lens_np;
    return retval;
}

static int
MetaDataclass_set_seq_lens_np(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'seq_lens_np' cannot be deleted");
        return -1;
    }
    if (self->_seq_lens_np != NULL) {
        CPy_DECREF(self->_seq_lens_np);
    }
    PyObject *tmp = value;
    CPy_INCREF(tmp);
    self->_seq_lens_np = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_max_seq_len(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_max_seq_len == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'max_seq_len' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_max_seq_len);
    PyObject *retval = CPyTagged_StealAsObject(self->_max_seq_len);
    return retval;
}

static int
MetaDataclass_set_max_seq_len(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'max_seq_len' cannot be deleted");
        return -1;
    }
    if (self->_max_seq_len != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_max_seq_len);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_max_seq_len = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_kv_cache_dtype(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_kv_cache_dtype == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'kv_cache_dtype' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPy_INCREF(self->_kv_cache_dtype);
    PyObject *retval = self->_kv_cache_dtype;
    return retval;
}

static int
MetaDataclass_set_kv_cache_dtype(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'kv_cache_dtype' cannot be deleted");
        return -1;
    }
    if (self->_kv_cache_dtype != NULL) {
        CPy_DECREF(self->_kv_cache_dtype);
    }
    PyObject *tmp;
    if (likely(PyUnicode_Check(value)))
        tmp = value;
    else {
        CPy_TypeError("str", value); 
        tmp = NULL;
    }
    if (!tmp)
        return -1;
    CPy_INCREF(tmp);
    self->_kv_cache_dtype = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_block_size(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_block_size == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'block_size' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_block_size);
    PyObject *retval = CPyTagged_StealAsObject(self->_block_size);
    return retval;
}

static int
MetaDataclass_set_block_size(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'block_size' cannot be deleted");
        return -1;
    }
    if (self->_block_size != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_block_size);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_block_size = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_num_kv_heads(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_num_kv_heads == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'num_kv_heads' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_num_kv_heads);
    PyObject *retval = CPyTagged_StealAsObject(self->_num_kv_heads);
    return retval;
}

static int
MetaDataclass_set_num_kv_heads(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'num_kv_heads' cannot be deleted");
        return -1;
    }
    if (self->_num_kv_heads != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_num_kv_heads);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_num_kv_heads = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_head_size(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_head_size == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'head_size' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_head_size);
    PyObject *retval = CPyTagged_StealAsObject(self->_head_size);
    return retval;
}

static int
MetaDataclass_set_head_size(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'head_size' cannot be deleted");
        return -1;
    }
    if (self->_head_size != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_head_size);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_head_size = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_causal(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_causal == 2)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'causal' of 'MetaDataclass' undefined");
        return NULL;
    }
    PyObject *retval = self->_causal ? Py_True : Py_False;
    CPy_INCREF(retval);
    return retval;
}

static int
MetaDataclass_set_causal(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'causal' cannot be deleted");
        return -1;
    }
    char tmp;
    if (unlikely(!PyBool_Check(value))) {
        CPy_TypeError("bool", value); return -1;
    } else
        tmp = value == Py_True;
    self->_causal = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_sliding_window(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_sliding_window == CPY_INT_TAG)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'sliding_window' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPyTagged_INCREF(self->_sliding_window);
    PyObject *retval = CPyTagged_StealAsObject(self->_sliding_window);
    return retval;
}

static int
MetaDataclass_set_sliding_window(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'sliding_window' cannot be deleted");
        return -1;
    }
    if (self->_sliding_window != CPY_INT_TAG) {
        CPyTagged_DECREF(self->_sliding_window);
    }
    CPyTagged tmp;
    if (likely(PyLong_Check(value)))
        tmp = CPyTagged_BorrowFromObject(value);
    else {
        CPy_TypeError("int", value); return -1;
    }
    CPyTagged_INCREF(tmp);
    self->_sliding_window = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_token_ids(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_token_ids == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'token_ids' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPy_INCREF(self->_token_ids);
    PyObject *retval = self->_token_ids;
    return retval;
}

static int
MetaDataclass_set_token_ids(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'token_ids' cannot be deleted");
        return -1;
    }
    if (self->_token_ids != NULL) {
        CPy_DECREF(self->_token_ids);
    }
    PyObject *tmp = value;
    CPy_INCREF(tmp);
    self->_token_ids = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_positions(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_positions == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'positions' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPy_INCREF(self->_positions);
    PyObject *retval = self->_positions;
    return retval;
}

static int
MetaDataclass_set_positions(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'positions' cannot be deleted");
        return -1;
    }
    if (self->_positions != NULL) {
        CPy_DECREF(self->_positions);
    }
    PyObject *tmp = value;
    CPy_INCREF(tmp);
    self->_positions = tmp;
    return 0;
}

static PyObject *
MetaDataclass_get_workspace(mypyc_variant___MetaDataclassObject *self, void *closure)
{
    if (unlikely(self->_workspace == NULL)) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute 'workspace' of 'MetaDataclass' undefined");
        return NULL;
    }
    CPy_INCREF(self->_workspace);
    PyObject *retval = self->_workspace;
    return retval;
}

static int
MetaDataclass_set_workspace(mypyc_variant___MetaDataclassObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
            "'MetaDataclass' object attribute 'workspace' cannot be deleted");
        return -1;
    }
    if (self->_workspace != NULL) {
        CPy_DECREF(self->_workspace);
    }
    PyObject *tmp = value;
    CPy_INCREF(tmp);
    self->_workspace = tmp;
    return 0;
}
static PyMethodDef module_methods[] = {
    {"_kwargs", (PyCFunction)CPyPy__kwargs, METH_FASTCALL | METH_KEYWORDS, PyDoc_STR("_kwargs(inp)\n--\n\n") /* docstring */},
    {"small_ops", (PyCFunction)CPyPy_small_ops, METH_FASTCALL | METH_KEYWORDS, PyDoc_STR("small_ops(inp, mode=\'torch\')\n--\n\n") /* docstring */},
    {"_read_scalars", (PyCFunction)CPyPy__read_scalars, METH_FASTCALL | METH_KEYWORDS, PyDoc_STR("_read_scalars(m)\n--\n\n") /* docstring */},
    {"_read_containers", (PyCFunction)CPyPy__read_containers, METH_FASTCALL | METH_KEYWORDS, PyDoc_STR("_read_containers(m)\n--\n\n") /* docstring */},
    {"run_mypyc_native", (PyCFunction)CPyPy_run_mypyc_native, METH_FASTCALL | METH_KEYWORDS, PyDoc_STR("run_mypyc_native(inp, mode=\'torch\')\n--\n\n") /* docstring */},
    {"run_mypyc_dataclass", (PyCFunction)CPyPy_run_mypyc_dataclass, METH_FASTCALL | METH_KEYWORDS, PyDoc_STR("run_mypyc_dataclass(inp, mode=\'torch\')\n--\n\n") /* docstring */},
    {NULL, NULL, 0, NULL}
};

int CPyExec_mypyc_variant(PyObject *module)
{
    intern_strings();
    PyObject* modname = NULL;
    modname = PyObject_GetAttrString((PyObject *)CPyModule_mypyc_variant__internal, "__name__");
    CPyStatic_globals = PyModule_GetDict(CPyModule_mypyc_variant__internal);
    if (unlikely(CPyStatic_globals == NULL))
        goto fail;
    if (CPyGlobalsInit() < 0)
        goto fail;
    char result = CPyDef___top_level__();
    if (result == 2)
        goto fail;
    Py_DECREF(modname);
    return 0;
    fail:
    Py_CLEAR(CPyModule_mypyc_variant__internal);
    Py_CLEAR(modname);
    Py_CLEAR(CPyType_MetaNative);
    Py_CLEAR(CPyType_MetaDataclass);
    return -1;
}
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "mypyc_variant",
    NULL, /* docstring */
    0,       /* size of per-interpreter state of the module */
    module_methods,
    NULL,
};

PyMODINIT_FUNC PyInit_mypyc_variant(void)
{
    PyObject* modname = NULL;
    if (CPyModule_mypyc_variant__internal) {
        Py_INCREF(CPyModule_mypyc_variant__internal);
        return CPyModule_mypyc_variant__internal;
    }
    CPyModule_mypyc_variant__internal = PyModule_Create(&module);
    if (unlikely(CPyModule_mypyc_variant__internal == NULL))
        goto fail;
    modname = PyUnicode_FromString("mypyc_variant");
    if (modname == NULL) CPyError_OutOfMemory();
    int rv = 0;
    PyObject *shared_lib_file = PyUnicode_FromString("mypyc_variant.cpython-312-x86_64-linux-gnu.so");
    if (shared_lib_file == NULL) CPyError_OutOfMemory();
    PyObject *ext_suffix = PyUnicode_FromString(".cpython-312-x86_64-linux-gnu.so");
    if (ext_suffix == NULL) CPyError_OutOfMemory();
    Py_ssize_t is_pkg = 0;
    rv = CPyImport_SetDunderAttrs(CPyModule_mypyc_variant__internal, modname, shared_lib_file, ext_suffix, is_pkg);
    Py_DECREF(ext_suffix);
    Py_DECREF(shared_lib_file);
    if (rv < 0) goto fail;
    if (PyObject_SetItem(PyImport_GetModuleDict(), modname, CPyModule_mypyc_variant__internal) < 0)
        goto fail;
    Py_CLEAR(modname);
    if (CPyExec_mypyc_variant(CPyModule_mypyc_variant__internal) != 0)
        goto fail;
    return CPyModule_mypyc_variant__internal;
    fail:
    {
            PyObject *exc_type, *exc_val, *exc_tb;
            PyErr_Fetch(&exc_type, &exc_val, &exc_tb);
            if (modname == NULL) {
                    modname = PyUnicode_FromString("mypyc_variant");
                    if (modname == NULL) CPyError_OutOfMemory();
                }
                PyObject_DelItem(PyImport_GetModuleDict(), modname);
                PyErr_Clear();
                Py_DECREF(modname);
                Py_CLEAR(CPyModule_mypyc_variant__internal);
                PyErr_Restore(exc_type, exc_val, exc_tb);
        }
        return NULL;
    }
    
char CPyDef_MetaNative_____init__(PyObject *cpy_r_self, CPyTagged cpy_r_num_reqs, CPyTagged cpy_r_num_actual_tokens, CPyTagged cpy_r_max_query_len, PyObject *cpy_r_query_start_loc, PyObject *cpy_r_seq_lens, PyObject *cpy_r_block_table, PyObject *cpy_r_slot_mapping, char cpy_r_spec_decode, CPyTagged cpy_r_num_decode_tokens, CPyTagged cpy_r_num_prefill_tokens, PyObject *cpy_r_num_computed_tokens, PyObject *cpy_r_seq_lens_np, CPyTagged cpy_r_max_seq_len, PyObject *cpy_r_kv_cache_dtype, CPyTagged cpy_r_block_size, CPyTagged cpy_r_num_kv_heads, CPyTagged cpy_r_head_size, char cpy_r_causal, CPyTagged cpy_r_sliding_window, PyObject *cpy_r_token_ids, PyObject *cpy_r_positions, PyObject *cpy_r_workspace) {
    CPyTagged_INCREF(cpy_r_num_reqs);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_num_reqs = cpy_r_num_reqs;
    CPyTagged_INCREF(cpy_r_num_actual_tokens);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_num_actual_tokens = cpy_r_num_actual_tokens;
    CPyTagged_INCREF(cpy_r_max_query_len);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_max_query_len = cpy_r_max_query_len;
    CPy_INCREF(cpy_r_query_start_loc);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_query_start_loc = cpy_r_query_start_loc;
    CPy_INCREF(cpy_r_seq_lens);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_seq_lens = cpy_r_seq_lens;
    CPy_INCREF(cpy_r_block_table);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_block_table = cpy_r_block_table;
    CPy_INCREF(cpy_r_slot_mapping);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_slot_mapping = cpy_r_slot_mapping;
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_spec_decode = cpy_r_spec_decode;
    CPyTagged_INCREF(cpy_r_num_decode_tokens);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_num_decode_tokens = cpy_r_num_decode_tokens;
    CPyTagged_INCREF(cpy_r_num_prefill_tokens);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_num_prefill_tokens = cpy_r_num_prefill_tokens;
    CPy_INCREF(cpy_r_num_computed_tokens);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_num_computed_tokens = cpy_r_num_computed_tokens;
    CPy_INCREF(cpy_r_seq_lens_np);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_seq_lens_np = cpy_r_seq_lens_np;
    CPyTagged_INCREF(cpy_r_max_seq_len);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_max_seq_len = cpy_r_max_seq_len;
    CPy_INCREF(cpy_r_kv_cache_dtype);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_kv_cache_dtype = cpy_r_kv_cache_dtype;
    CPyTagged_INCREF(cpy_r_block_size);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_block_size = cpy_r_block_size;
    CPyTagged_INCREF(cpy_r_num_kv_heads);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_num_kv_heads = cpy_r_num_kv_heads;
    CPyTagged_INCREF(cpy_r_head_size);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_head_size = cpy_r_head_size;
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_causal = cpy_r_causal;
    CPyTagged_INCREF(cpy_r_sliding_window);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_sliding_window = cpy_r_sliding_window;
    CPy_INCREF(cpy_r_token_ids);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_token_ids = cpy_r_token_ids;
    CPy_INCREF(cpy_r_positions);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_positions = cpy_r_positions;
    CPy_INCREF(cpy_r_workspace);
    ((mypyc_variant___MetaNativeObject *)cpy_r_self)->_workspace = cpy_r_workspace;
    return 1;
}
    
    PyObject *CPyPy_MetaNative_____init__(PyObject *self, PyObject *args, PyObject *kw) {
        PyObject *obj_self = self;
        static const char * const kwlist[] = {"num_reqs", "num_actual_tokens", "max_query_len", "query_start_loc", "seq_lens", "block_table", "slot_mapping", "spec_decode", "num_decode_tokens", "num_prefill_tokens", "num_computed_tokens", "seq_lens_np", "max_seq_len", "kv_cache_dtype", "block_size", "num_kv_heads", "head_size", "causal", "sliding_window", "token_ids", "positions", "workspace", 0};
        PyObject *obj_num_reqs;
        PyObject *obj_num_actual_tokens;
        PyObject *obj_max_query_len;
        PyObject *obj_query_start_loc;
        PyObject *obj_seq_lens;
        PyObject *obj_block_table;
        PyObject *obj_slot_mapping;
        PyObject *obj_spec_decode;
        PyObject *obj_num_decode_tokens;
        PyObject *obj_num_prefill_tokens;
        PyObject *obj_num_computed_tokens;
        PyObject *obj_seq_lens_np;
        PyObject *obj_max_seq_len;
        PyObject *obj_kv_cache_dtype;
        PyObject *obj_block_size;
        PyObject *obj_num_kv_heads;
        PyObject *obj_head_size;
        PyObject *obj_causal;
        PyObject *obj_sliding_window;
        PyObject *obj_token_ids;
        PyObject *obj_positions;
        PyObject *obj_workspace;
        if (!CPyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOOOOOOOOOOOOOOOO", "__init__", kwlist, &obj_num_reqs, &obj_num_actual_tokens, &obj_max_query_len, &obj_query_start_loc, &obj_seq_lens, &obj_block_table, &obj_slot_mapping, &obj_spec_decode, &obj_num_decode_tokens, &obj_num_prefill_tokens, &obj_num_computed_tokens, &obj_seq_lens_np, &obj_max_seq_len, &obj_kv_cache_dtype, &obj_block_size, &obj_num_kv_heads, &obj_head_size, &obj_causal, &obj_sliding_window, &obj_token_ids, &obj_positions, &obj_workspace)) {
            return NULL;
        }
        PyObject *arg_self;
        if (likely(Py_TYPE(obj_self) == CPyType_MetaNative))
            arg_self = obj_self;
        else {
            CPy_TypeError("mypyc_variant.MetaNative", obj_self); 
            goto fail;
        }
        CPyTagged arg_num_reqs;
        if (likely(PyLong_Check(obj_num_reqs)))
            arg_num_reqs = CPyTagged_BorrowFromObject(obj_num_reqs);
        else {
            CPy_TypeError("int", obj_num_reqs); goto fail;
        }
        CPyTagged arg_num_actual_tokens;
        if (likely(PyLong_Check(obj_num_actual_tokens)))
            arg_num_actual_tokens = CPyTagged_BorrowFromObject(obj_num_actual_tokens);
        else {
            CPy_TypeError("int", obj_num_actual_tokens); goto fail;
        }
        CPyTagged arg_max_query_len;
        if (likely(PyLong_Check(obj_max_query_len)))
            arg_max_query_len = CPyTagged_BorrowFromObject(obj_max_query_len);
        else {
            CPy_TypeError("int", obj_max_query_len); goto fail;
        }
        PyObject *arg_query_start_loc = obj_query_start_loc;
        PyObject *arg_seq_lens = obj_seq_lens;
        PyObject *arg_block_table = obj_block_table;
        PyObject *arg_slot_mapping = obj_slot_mapping;
        char arg_spec_decode;
        if (unlikely(!PyBool_Check(obj_spec_decode))) {
            CPy_TypeError("bool", obj_spec_decode); goto fail;
        } else
            arg_spec_decode = obj_spec_decode == Py_True;
        CPyTagged arg_num_decode_tokens;
        if (likely(PyLong_Check(obj_num_decode_tokens)))
            arg_num_decode_tokens = CPyTagged_BorrowFromObject(obj_num_decode_tokens);
        else {
            CPy_TypeError("int", obj_num_decode_tokens); goto fail;
        }
        CPyTagged arg_num_prefill_tokens;
        if (likely(PyLong_Check(obj_num_prefill_tokens)))
            arg_num_prefill_tokens = CPyTagged_BorrowFromObject(obj_num_prefill_tokens);
        else {
            CPy_TypeError("int", obj_num_prefill_tokens); goto fail;
        }
        PyObject *arg_num_computed_tokens = obj_num_computed_tokens;
        PyObject *arg_seq_lens_np = obj_seq_lens_np;
        CPyTagged arg_max_seq_len;
        if (likely(PyLong_Check(obj_max_seq_len)))
            arg_max_seq_len = CPyTagged_BorrowFromObject(obj_max_seq_len);
        else {
            CPy_TypeError("int", obj_max_seq_len); goto fail;
        }
        PyObject *arg_kv_cache_dtype;
        if (likely(PyUnicode_Check(obj_kv_cache_dtype)))
            arg_kv_cache_dtype = obj_kv_cache_dtype;
        else {
            CPy_TypeError("str", obj_kv_cache_dtype); 
            goto fail;
        }
        CPyTagged arg_block_size;
        if (likely(PyLong_Check(obj_block_size)))
            arg_block_size = CPyTagged_BorrowFromObject(obj_block_size);
        else {
            CPy_TypeError("int", obj_block_size); goto fail;
        }
        CPyTagged arg_num_kv_heads;
        if (likely(PyLong_Check(obj_num_kv_heads)))
            arg_num_kv_heads = CPyTagged_BorrowFromObject(obj_num_kv_heads);
        else {
            CPy_TypeError("int", obj_num_kv_heads); goto fail;
        }
        CPyTagged arg_head_size;
        if (likely(PyLong_Check(obj_head_size)))
            arg_head_size = CPyTagged_BorrowFromObject(obj_head_size);
        else {
            CPy_TypeError("int", obj_head_size); goto fail;
        }
        char arg_causal;
        if (unlikely(!PyBool_Check(obj_causal))) {
            CPy_TypeError("bool", obj_causal); goto fail;
        } else
            arg_causal = obj_causal == Py_True;
        CPyTagged arg_sliding_window;
        if (likely(PyLong_Check(obj_sliding_window)))
            arg_sliding_window = CPyTagged_BorrowFromObject(obj_sliding_window);
        else {
            CPy_TypeError("int", obj_sliding_window); goto fail;
        }
        PyObject *arg_token_ids = obj_token_ids;
        PyObject *arg_positions = obj_positions;
        PyObject *arg_workspace = obj_workspace;
        char retval = CPyDef_MetaNative_____init__(arg_self, arg_num_reqs, arg_num_actual_tokens, arg_max_query_len, arg_query_start_loc, arg_seq_lens, arg_block_table, arg_slot_mapping, arg_spec_decode, arg_num_decode_tokens, arg_num_prefill_tokens, arg_num_computed_tokens, arg_seq_lens_np, arg_max_seq_len, arg_kv_cache_dtype, arg_block_size, arg_num_kv_heads, arg_head_size, arg_causal, arg_sliding_window, arg_token_ids, arg_positions, arg_workspace);
        if (retval == 2) {
            return NULL;
        }
        PyObject *retbox = Py_None;
        CPy_INCREF(retbox);
        return retbox;
fail: ;
        CPy_AddTraceback("mypyc_variant.py", "__init__", 52, CPyStatic_globals);
        return NULL;
    }
    
PyObject *CPyDef__kwargs(PyObject *cpy_r_inp) {
    PyObject *cpy_r_r0;
    PyObject *cpy_r_r1;
    PyObject *cpy_r_r2;
    PyObject *cpy_r_r3;
    CPyPtr cpy_r_r4;
    int64_t cpy_r_r5;
    PyObject *cpy_r_r6;
    int64_t cpy_r_r7;
    CPyPtr cpy_r_r8;
    int64_t cpy_r_r9;
    char cpy_r_r10;
    CPyPtr cpy_r_r11;
    CPyPtr cpy_r_r12;
    int64_t cpy_r_r13;
    CPyPtr cpy_r_r14;
    PyObject *cpy_r_r15;
    PyObject *cpy_r_r16;
    PyObject *cpy_r_r17;
    int64_t cpy_r_r18;
    PyObject *cpy_r_r19;
    cpy_r_r0 = CPyStatic_globals;
    cpy_r_r1 = CPyStatics[3]; /* 'FIELDS' */
    cpy_r_r2 = CPyDict_GetItem(cpy_r_r0, cpy_r_r1);
    if (unlikely(cpy_r_r2 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_kwargs", 110, CPyStatic_globals);
        goto CPyL10;
    }
    if (likely(PyList_Check(cpy_r_r2)))
        cpy_r_r3 = cpy_r_r2;
    else {
        CPy_TypeErrorTraceback("mypyc_variant.py", "_kwargs", 110, CPyStatic_globals, "list", cpy_r_r2);
        goto CPyL10;
    }
    cpy_r_r4 = (CPyPtr)((CPyPtr)cpy_r_r3 + offsetof(PyVarObject, ob_size));
    cpy_r_r5 = *(int64_t *)cpy_r_r4;
    cpy_r_r6 = PyList_New(cpy_r_r5);
    if (unlikely(cpy_r_r6 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_kwargs", 110, CPyStatic_globals);
        goto CPyL11;
    }
    cpy_r_r7 = 0;
CPyL4: ;
    cpy_r_r8 = (CPyPtr)((CPyPtr)cpy_r_r3 + offsetof(PyVarObject, ob_size));
    cpy_r_r9 = *(int64_t *)cpy_r_r8;
    cpy_r_r10 = cpy_r_r7 < cpy_r_r9;
    if (!cpy_r_r10) goto CPyL12;
    cpy_r_r11 = (CPyPtr)((CPyPtr)cpy_r_r3 + offsetof(PyListObject, ob_item));
    cpy_r_r12 = *(CPyPtr *)cpy_r_r11;
    cpy_r_r13 = cpy_r_r7 * 8;
    cpy_r_r14 = cpy_r_r12 + cpy_r_r13;
    cpy_r_r15 = *(PyObject * *)cpy_r_r14;
    CPy_INCREF(cpy_r_r15);
    if (likely(PyUnicode_Check(cpy_r_r15)))
        cpy_r_r16 = cpy_r_r15;
    else {
        CPy_TypeErrorTraceback("mypyc_variant.py", "_kwargs", 110, CPyStatic_globals, "str", cpy_r_r15);
        goto CPyL13;
    }
    cpy_r_r17 = CPyDict_GetItem(cpy_r_inp, cpy_r_r16);
    CPy_DECREF(cpy_r_r16);
    if (unlikely(cpy_r_r17 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_kwargs", 110, CPyStatic_globals);
        goto CPyL13;
    }
    CPyList_SetItemUnsafe(cpy_r_r6, cpy_r_r7, cpy_r_r17);
    cpy_r_r18 = cpy_r_r7 + 1;
    cpy_r_r7 = cpy_r_r18;
    goto CPyL4;
CPyL9: ;
    return cpy_r_r6;
CPyL10: ;
    cpy_r_r19 = NULL;
    return cpy_r_r19;
CPyL11: ;
    CPy_DecRef(cpy_r_r3);
    goto CPyL10;
CPyL12: ;
    CPy_DECREF_NO_IMM(cpy_r_r3);
    goto CPyL9;
CPyL13: ;
    CPy_DecRef(cpy_r_r3);
    CPy_DecRef(cpy_r_r6);
    goto CPyL10;
}
    
    PyObject *CPyPy__kwargs(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames) {
        static const char * const kwlist[] = {"inp", 0};
        static CPyArg_Parser parser = {"O:_kwargs", kwlist, 0};
        PyObject *obj_inp;
        if (!CPyArg_ParseStackAndKeywordsOneArg(args, nargs, kwnames, &parser, &obj_inp)) {
            return NULL;
        }
        PyObject *arg_inp;
        if (likely(PyDict_Check(obj_inp)))
            arg_inp = obj_inp;
        else {
            CPy_TypeError("dict", obj_inp); 
            goto fail;
        }
        PyObject *retval = CPyDef__kwargs(arg_inp);
        return retval;
fail: ;
        CPy_AddTraceback("mypyc_variant.py", "_kwargs", 109, CPyStatic_globals);
        return NULL;
    }
    
PyObject *CPyDef_small_ops(PyObject *cpy_r_inp, PyObject *cpy_r_mode) {
    PyObject *cpy_r_r0;
    PyObject *cpy_r_r1;
    char cpy_r_r2;
    tuple_T8IIFIFFCI cpy_r_r3;
    PyObject *cpy_r_r4;
    PyObject *cpy_r_r5;
    PyObject *cpy_r_r6;
    PyObject *cpy_r_r7;
    PyObject *cpy_r_r8;
    PyObject *cpy_r_r9;
    PyObject **cpy_r_r11;
    PyObject *cpy_r_r12;
    PyObject *cpy_r_r13;
    PyObject *cpy_r_r14;
    PyObject *cpy_r_r15;
    PyObject *cpy_r_r16;
    PyObject *cpy_r_r17;
    PyObject *cpy_r_r18;
    PyObject *cpy_r_r19;
    PyObject *cpy_r_r20;
    PyObject *cpy_r_r21;
    PyObject *cpy_r_r22;
    PyObject *cpy_r_r23;
    PyObject *cpy_r_r24;
    PyObject *cpy_r_r25;
    PyObject *cpy_r_r26;
    PyObject *cpy_r_r27;
    PyObject *cpy_r_r28;
    PyObject *cpy_r_r29;
    PyObject *cpy_r_r30;
    PyObject **cpy_r_r32;
    PyObject *cpy_r_r33;
    PyObject *cpy_r_r34;
    PyObject *cpy_r_r35;
    PyObject *cpy_r_r36;
    PyObject *cpy_r_r37;
    PyObject *cpy_r_r38;
    PyObject *cpy_r_r39;
    PyObject *cpy_r_r40;
    PyObject *cpy_r_r41;
    PyObject **cpy_r_r43;
    PyObject *cpy_r_r44;
    PyObject *cpy_r_r45;
    PyObject *cpy_r_r46;
    PyObject *cpy_r_r47;
    PyObject *cpy_r_r48;
    PyObject **cpy_r_r50;
    PyObject *cpy_r_r51;
    double cpy_r_r52;
    char cpy_r_r53;
    PyObject *cpy_r_r54;
    char cpy_r_r55;
    PyObject *cpy_r_r56;
    PyObject *cpy_r_r57;
    PyObject *cpy_r_r58;
    PyObject *cpy_r_r59;
    PyObject **cpy_r_r61;
    PyObject *cpy_r_r62;
    PyObject *cpy_r_t;
    PyObject *cpy_r_r63;
    PyObject *cpy_r_r64;
    PyObject *cpy_r_t2;
    PyObject *cpy_r_r65;
    PyObject **cpy_r_r67;
    PyObject *cpy_r_r68;
    PyObject *cpy_r_r69;
    PyObject **cpy_r_r71;
    PyObject *cpy_r_r72;
    double cpy_r_r73;
    char cpy_r_r74;
    double cpy_r_s;
    PyObject *cpy_r_r75;
    PyObject *cpy_r_r76;
    PyObject *cpy_r_r77;
    PyObject *cpy_r_r78;
    PyObject *cpy_r_r79;
    PyObject *cpy_r_r80;
    PyObject *cpy_r_r81;
    PyObject *cpy_r_r82;
    PyObject **cpy_r_r84;
    PyObject *cpy_r_r85;
    PyObject *cpy_r_r86;
    PyObject *cpy_r_ar;
    PyObject *cpy_r_r87;
    PyObject **cpy_r_r89;
    PyObject *cpy_r_r90;
    PyObject *cpy_r_cl;
    PyObject *cpy_r_r91;
    PyObject *cpy_r_r92;
    PyObject *cpy_r_r93;
    PyObject *cpy_r_r94;
    PyObject **cpy_r_r96;
    PyObject *cpy_r_r97;
    PyObject *cpy_r_r98;
    PyObject *cpy_r_cs;
    PyObject *cpy_r_r99;
    PyObject *cpy_r_r100;
    PyObject *cpy_r_r101;
    PyObject *cpy_r_r102;
    PyObject *cpy_r_r103;
    PyObject *cpy_r_r104;
    PyObject *cpy_r_r105;
    PyObject **cpy_r_r107;
    PyObject *cpy_r_r108;
    PyObject *cpy_r_r109;
    PyObject *cpy_r_fz;
    PyObject *cpy_r_r110;
    PyObject *cpy_r_r111;
    PyObject *cpy_r_r112;
    PyObject *cpy_r_r113;
    PyObject *cpy_r_r114;
    PyObject *cpy_r_r115;
    PyObject *cpy_r_r116;
    PyObject *cpy_r_ws;
    PyObject *cpy_r_r117;
    PyObject *cpy_r_r118;
    PyObject *cpy_r_r119;
    PyObject **cpy_r_r121;
    PyObject *cpy_r_r122;
    CPyTagged cpy_r_r123;
    PyObject *cpy_r_r124;
    PyObject **cpy_r_r126;
    PyObject *cpy_r_r127;
    PyObject *cpy_r_r128;
    PyObject **cpy_r_r130;
    PyObject *cpy_r_r131;
    CPyTagged cpy_r_r132;
    PyObject *cpy_r_r133;
    PyObject **cpy_r_r135;
    PyObject *cpy_r_r136;
    PyObject *cpy_r_r137;
    PyObject **cpy_r_r139;
    PyObject *cpy_r_r140;
    double cpy_r_r141;
    char cpy_r_r142;
    PyObject *cpy_r_r143;
    PyObject *cpy_r_r144;
    PyObject *cpy_r_r145;
    PyObject *cpy_r_r146;
    PyObject **cpy_r_r148;
    PyObject *cpy_r_r149;
    CPyTagged cpy_r_r150;
    double cpy_r_r151;
    PyObject *cpy_r_r152;
    PyObject *cpy_r_r153;
    PyObject *cpy_r_r154;
    PyObject **cpy_r_r156;
    PyObject *cpy_r_r157;
    int32_t cpy_r_r158;
    char cpy_r_r159;
    char cpy_r_r160;
    PyObject *cpy_r_r161;
    PyObject *cpy_r_r162;
    PyObject *cpy_r_r163;
    PyObject **cpy_r_r165;
    PyObject *cpy_r_r166;
    CPyTagged cpy_r_r167;
    PyObject *cpy_r_r168;
    PyObject *cpy_r_r169;
    PyObject *cpy_r_r170;
    PyObject **cpy_r_r172;
    PyObject *cpy_r_r173;
    CPyTagged cpy_r_r174;
    CPyTagged cpy_r_r175;
    PyObject *cpy_r_r176;
    PyObject **cpy_r_r178;
    PyObject *cpy_r_r179;
    PyObject *cpy_r_r180;
    PyObject **cpy_r_r182;
    PyObject *cpy_r_r183;
    CPyTagged cpy_r_r184;
    CPyTagged cpy_r_r185;
    PyObject *cpy_r_r186;
    PyObject *cpy_r_r187;
    PyObject *cpy_r_r188;
    PyObject *cpy_r_r189;
    PyObject *cpy_r_r190;
    PyObject **cpy_r_r192;
    PyObject *cpy_r_r193;
    CPyTagged cpy_r_r194;
    CPyTagged cpy_r_r195;
    PyObject *cpy_r_r196;
    PyObject *cpy_r_r197;
    PyObject *cpy_r_r198;
    PyObject *cpy_r_r199;
    PyObject *cpy_r_r200;
    PyObject **cpy_r_r202;
    PyObject *cpy_r_r203;
    CPyTagged cpy_r_r204;
    CPyTagged cpy_r_r205;
    tuple_T8IIFIFFCI cpy_r_r206;
    PyObject *cpy_r_r207;
    PyObject *cpy_r_r208;
    PyObject *cpy_r_r209;
    PyObject *cpy_r_r210;
    PyObject **cpy_r_r212;
    PyObject *cpy_r_r213;
    PyObject *cpy_r_r214;
    PyObject *cpy_r_r215;
    PyObject **cpy_r_r217;
    PyObject *cpy_r_r218;
    PyObject *cpy_r_r219;
    PyObject **cpy_r_r221;
    PyObject *cpy_r_r222;
    PyObject *cpy_r_r223;
    PyObject **cpy_r_r225;
    PyObject *cpy_r_r226;
    double cpy_r_r227;
    char cpy_r_r228;
    PyObject *cpy_r_r229;
    PyObject *cpy_r_r230;
    PyObject *cpy_r_r231;
    PyObject *cpy_r_r232;
    PyObject *cpy_r_r233;
    PyObject *cpy_r_r234;
    PyObject *cpy_r_r235;
    PyObject *cpy_r_r236;
    PyObject **cpy_r_r238;
    PyObject *cpy_r_r239;
    PyObject *cpy_r_r240;
    PyObject *cpy_r_r241;
    PyObject **cpy_r_r243;
    PyObject *cpy_r_r244;
    PyObject *cpy_r_r245;
    PyObject *cpy_r_r246;
    PyObject *cpy_r_r247;
    PyObject *cpy_r_r248;
    PyObject **cpy_r_r250;
    PyObject *cpy_r_r251;
    PyObject *cpy_r_r252;
    PyObject *cpy_r_r253;
    PyObject *cpy_r_r254;
    PyObject *cpy_r_r255;
    PyObject *cpy_r_r256;
    PyObject *cpy_r_r257;
    PyObject *cpy_r_r258;
    PyObject **cpy_r_r260;
    PyObject *cpy_r_r261;
    PyObject *cpy_r_r262;
    PyObject *cpy_r_r263;
    PyObject *cpy_r_r264;
    PyObject *cpy_r_r265;
    PyObject *cpy_r_r266;
    PyObject *cpy_r_r267;
    PyObject *cpy_r_r268;
    PyObject *cpy_r_r269;
    PyObject *cpy_r_r270;
    PyObject *cpy_r_r271;
    PyObject *cpy_r_r272;
    PyObject **cpy_r_r274;
    PyObject *cpy_r_r275;
    CPyTagged cpy_r_r276;
    PyObject *cpy_r_r277;
    PyObject **cpy_r_r279;
    PyObject *cpy_r_r280;
    PyObject *cpy_r_r281;
    PyObject **cpy_r_r283;
    PyObject *cpy_r_r284;
    CPyTagged cpy_r_r285;
    PyObject *cpy_r_r286;
    PyObject **cpy_r_r288;
    PyObject *cpy_r_r289;
    PyObject *cpy_r_r290;
    PyObject **cpy_r_r292;
    PyObject *cpy_r_r293;
    double cpy_r_r294;
    char cpy_r_r295;
    PyObject *cpy_r_r296;
    PyObject *cpy_r_r297;
    PyObject *cpy_r_r298;
    PyObject *cpy_r_r299;
    PyObject **cpy_r_r301;
    PyObject *cpy_r_r302;
    CPyTagged cpy_r_r303;
    double cpy_r_r304;
    PyObject *cpy_r_r305;
    PyObject *cpy_r_r306;
    PyObject *cpy_r_r307;
    PyObject **cpy_r_r309;
    PyObject *cpy_r_r310;
    int32_t cpy_r_r311;
    char cpy_r_r312;
    char cpy_r_r313;
    PyObject *cpy_r_r314;
    PyObject *cpy_r_r315;
    PyObject *cpy_r_r316;
    PyObject **cpy_r_r318;
    PyObject *cpy_r_r319;
    CPyTagged cpy_r_r320;
    PyObject *cpy_r_r321;
    PyObject *cpy_r_r322;
    PyObject *cpy_r_r323;
    PyObject **cpy_r_r325;
    PyObject *cpy_r_r326;
    CPyTagged cpy_r_r327;
    CPyTagged cpy_r_r328;
    PyObject *cpy_r_r329;
    PyObject **cpy_r_r331;
    PyObject *cpy_r_r332;
    PyObject *cpy_r_r333;
    PyObject **cpy_r_r335;
    PyObject *cpy_r_r336;
    CPyTagged cpy_r_r337;
    CPyTagged cpy_r_r338;
    PyObject *cpy_r_r339;
    PyObject *cpy_r_r340;
    PyObject *cpy_r_r341;
    PyObject *cpy_r_r342;
    PyObject *cpy_r_r343;
    PyObject **cpy_r_r345;
    PyObject *cpy_r_r346;
    CPyTagged cpy_r_r347;
    CPyTagged cpy_r_r348;
    PyObject *cpy_r_r349;
    PyObject *cpy_r_r350;
    PyObject *cpy_r_r351;
    PyObject *cpy_r_r352;
    PyObject *cpy_r_r353;
    PyObject **cpy_r_r355;
    PyObject *cpy_r_r356;
    CPyTagged cpy_r_r357;
    CPyTagged cpy_r_r358;
    tuple_T8IIFIFFCI cpy_r_r359;
    PyObject *cpy_r_r360;
    PyObject *cpy_r_r361;
    if (cpy_r_mode != NULL) goto CPyL130;
    cpy_r_r0 = CPyStatics[4]; /* 'torch' */
    CPy_INCREF(cpy_r_r0);
    cpy_r_mode = cpy_r_r0;
CPyL2: ;
    cpy_r_r1 = CPyStatics[5]; /* 'none' */
    cpy_r_r2 = CPyStr_EqualLiteral(cpy_r_mode, cpy_r_r1, 4);
    if (cpy_r_r2) {
        goto CPyL131;
    } else
        goto CPyL4;
CPyL3: ;
    cpy_r_r3.f0 = 0;
    cpy_r_r3.f1 = 0;
    cpy_r_r3.f2 = 0.0;
    cpy_r_r3.f3 = 0;
    cpy_r_r3.f4 = 0.0;
    cpy_r_r3.f5 = 0.0;
    cpy_r_r3.f6 = 1;
    cpy_r_r3.f7 = 0;
    cpy_r_r4 = PyTuple_New(8);
    if (unlikely(cpy_r_r4 == NULL))
        CPyError_OutOfMemory();
    PyObject *__tmp1 = CPyTagged_StealAsObject(cpy_r_r3.f0);
    PyTuple_SET_ITEM(cpy_r_r4, 0, __tmp1);
    PyObject *__tmp2 = CPyTagged_StealAsObject(cpy_r_r3.f1);
    PyTuple_SET_ITEM(cpy_r_r4, 1, __tmp2);
    PyObject *__tmp3 = PyFloat_FromDouble(cpy_r_r3.f2);
    PyTuple_SET_ITEM(cpy_r_r4, 2, __tmp3);
    PyObject *__tmp4 = CPyTagged_StealAsObject(cpy_r_r3.f3);
    PyTuple_SET_ITEM(cpy_r_r4, 3, __tmp4);
    PyObject *__tmp5 = PyFloat_FromDouble(cpy_r_r3.f4);
    PyTuple_SET_ITEM(cpy_r_r4, 4, __tmp5);
    PyObject *__tmp6 = PyFloat_FromDouble(cpy_r_r3.f5);
    PyTuple_SET_ITEM(cpy_r_r4, 5, __tmp6);
    PyObject *__tmp7 = cpy_r_r3.f6 ? Py_True : Py_False;
    CPy_INCREF(__tmp7);
    PyTuple_SET_ITEM(cpy_r_r4, 6, __tmp7);
    PyObject *__tmp8 = CPyTagged_StealAsObject(cpy_r_r3.f7);
    PyTuple_SET_ITEM(cpy_r_r4, 7, __tmp8);
    return cpy_r_r4;
CPyL4: ;
    cpy_r_r5 = CPyStatics[6]; /* 'seq_lens_np' */
    cpy_r_r6 = CPyDict_GetItem(cpy_r_inp, cpy_r_r5);
    if (unlikely(cpy_r_r6 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 116, CPyStatic_globals);
        goto CPyL132;
    }
    cpy_r_r7 = CPyModule_numpy;
    cpy_r_r8 = CPyStatics[7]; /* 'cumsum' */
    cpy_r_r9 = CPyObject_GetAttr(cpy_r_r7, cpy_r_r8);
    if (unlikely(cpy_r_r9 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 117, CPyStatic_globals);
        goto CPyL133;
    }
    PyObject *cpy_r_r10[1] = {cpy_r_r6};
    cpy_r_r11 = (PyObject **)&cpy_r_r10;
    cpy_r_r12 = PyObject_Vectorcall(cpy_r_r9, cpy_r_r11, 1, 0);
    CPy_DECREF(cpy_r_r9);
    if (unlikely(cpy_r_r12 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 117, CPyStatic_globals);
        goto CPyL133;
    }
    cpy_r_r13 = (PyObject *)&_Py_NoneStruct;
    cpy_r_r14 = (PyObject *)&_Py_NoneStruct;
    cpy_r_r15 = CPyStatics[64]; /* 1 */
    cpy_r_r16 = PySlice_New(cpy_r_r15, cpy_r_r13, cpy_r_r14);
    if (unlikely(cpy_r_r16 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 118, CPyStatic_globals);
        goto CPyL134;
    }
    cpy_r_r17 = PyObject_GetItem(cpy_r_r6, cpy_r_r16);
    CPy_DECREF(cpy_r_r16);
    if (unlikely(cpy_r_r17 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 118, CPyStatic_globals);
        goto CPyL134;
    }
    cpy_r_r18 = (PyObject *)&_Py_NoneStruct;
    cpy_r_r19 = (PyObject *)&_Py_NoneStruct;
    cpy_r_r20 = CPyStatics[65]; /* -1 */
    cpy_r_r21 = PySlice_New(cpy_r_r18, cpy_r_r20, cpy_r_r19);
    if (unlikely(cpy_r_r21 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 118, CPyStatic_globals);
        goto CPyL135;
    }
    cpy_r_r22 = PyObject_GetItem(cpy_r_r6, cpy_r_r21);
    CPy_DECREF(cpy_r_r21);
    if (unlikely(cpy_r_r22 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 118, CPyStatic_globals);
        goto CPyL135;
    }
    cpy_r_r23 = PyNumber_Subtract(cpy_r_r17, cpy_r_r22);
    CPy_DECREF(cpy_r_r17);
    CPy_DECREF(cpy_r_r22);
    if (unlikely(cpy_r_r23 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 118, CPyStatic_globals);
        goto CPyL134;
    }
    cpy_r_r24 = CPyModule_numpy;
    cpy_r_r25 = CPyStatics[8]; /* 'int64' */
    cpy_r_r26 = CPyObject_GetAttr(cpy_r_r24, cpy_r_r25);
    if (unlikely(cpy_r_r26 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 119, CPyStatic_globals);
        goto CPyL136;
    }
    cpy_r_r27 = CPyModule_numpy;
    cpy_r_r28 = CPyStatics[9]; /* 'zeros' */
    cpy_r_r29 = CPyObject_GetAttr(cpy_r_r27, cpy_r_r28);
    if (unlikely(cpy_r_r29 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 119, CPyStatic_globals);
        goto CPyL137;
    }
    cpy_r_r30 = CPyStatics[64]; /* 1 */
    PyObject *cpy_r_r31[2] = {cpy_r_r30, cpy_r_r26};
    cpy_r_r32 = (PyObject **)&cpy_r_r31;
    cpy_r_r33 = CPyStatics[68]; /* ('dtype',) */
    cpy_r_r34 = PyObject_Vectorcall(cpy_r_r29, cpy_r_r32, 1, cpy_r_r33);
    CPy_DECREF(cpy_r_r29);
    if (unlikely(cpy_r_r34 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 119, CPyStatic_globals);
        goto CPyL137;
    }
    CPy_DECREF(cpy_r_r26);
    cpy_r_r35 = CPyModule_numpy;
    cpy_r_r36 = CPyStatics[11]; /* 'int32' */
    cpy_r_r37 = CPyObject_GetAttr(cpy_r_r35, cpy_r_r36);
    if (unlikely(cpy_r_r37 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 120, CPyStatic_globals);
        goto CPyL138;
    }
    cpy_r_r38 = CPyModule_numpy;
    cpy_r_r39 = CPyStatics[12]; /* 'arange' */
    cpy_r_r40 = CPyObject_GetAttr(cpy_r_r38, cpy_r_r39);
    if (unlikely(cpy_r_r40 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 120, CPyStatic_globals);
        goto CPyL139;
    }
    cpy_r_r41 = CPyStatics[64]; /* 1 */
    PyObject *cpy_r_r42[2] = {cpy_r_r41, cpy_r_r37};
    cpy_r_r43 = (PyObject **)&cpy_r_r42;
    cpy_r_r44 = CPyStatics[68]; /* ('dtype',) */
    cpy_r_r45 = PyObject_Vectorcall(cpy_r_r40, cpy_r_r43, 1, cpy_r_r44);
    CPy_DECREF(cpy_r_r40);
    if (unlikely(cpy_r_r45 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 120, CPyStatic_globals);
        goto CPyL139;
    }
    CPy_DECREF(cpy_r_r37);
    cpy_r_r46 = CPyStatics[65]; /* -1 */
    cpy_r_r47 = PyObject_GetItem(cpy_r_r6, cpy_r_r46);
    if (unlikely(cpy_r_r47 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 121, CPyStatic_globals);
        goto CPyL140;
    }
    cpy_r_r48 = (PyObject *)&PyFloat_Type;
    PyObject *cpy_r_r49[1] = {cpy_r_r47};
    cpy_r_r50 = (PyObject **)&cpy_r_r49;
    cpy_r_r51 = PyObject_Vectorcall(cpy_r_r48, cpy_r_r50, 1, 0);
    if (unlikely(cpy_r_r51 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 121, CPyStatic_globals);
        goto CPyL141;
    }
    CPy_DECREF(cpy_r_r47);
    cpy_r_r52 = PyFloat_AsDouble(cpy_r_r51);
    if (cpy_r_r52 == -1.0 && PyErr_Occurred()) {
        CPy_TypeError("float", cpy_r_r51); cpy_r_r52 = -113.0;
    }
    CPy_DECREF(cpy_r_r51);
    cpy_r_r53 = cpy_r_r52 == -113.0;
    if (unlikely(cpy_r_r53)) goto CPyL22;
CPyL21: ;
    cpy_r_r54 = CPyStatics[13]; /* 'numpy' */
    cpy_r_r55 = CPyStr_EqualLiteral(cpy_r_mode, cpy_r_r54, 5);
    CPy_DECREF(cpy_r_mode);
    if (cpy_r_r55) {
        goto CPyL23;
    } else
        goto CPyL76;
CPyL22: ;
    cpy_r_r56 = PyErr_Occurred();
    if (unlikely(cpy_r_r56 != NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 121, CPyStatic_globals);
        goto CPyL140;
    } else
        goto CPyL21;
CPyL23: ;
    cpy_r_r57 = CPyModule_numpy;
    cpy_r_r58 = CPyStatics[14]; /* 'asarray' */
    cpy_r_r59 = CPyObject_GetAttr(cpy_r_r57, cpy_r_r58);
    if (unlikely(cpy_r_r59 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 123, CPyStatic_globals);
        goto CPyL142;
    }
    PyObject *cpy_r_r60[1] = {cpy_r_r6};
    cpy_r_r61 = (PyObject **)&cpy_r_r60;
    cpy_r_r62 = PyObject_Vectorcall(cpy_r_r59, cpy_r_r61, 1, 0);
    CPy_DECREF(cpy_r_r59);
    if (unlikely(cpy_r_r62 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 123, CPyStatic_globals);
        goto CPyL142;
    }
    CPy_DECREF(cpy_r_r6);
    cpy_r_t = cpy_r_r62;
    cpy_r_r63 = CPyStatics[66]; /* 2 */
    cpy_r_r64 = PyNumber_Multiply(cpy_r_t, cpy_r_r63);
    if (unlikely(cpy_r_r64 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 124, CPyStatic_globals);
        goto CPyL143;
    }
    cpy_r_t2 = cpy_r_r64;
    cpy_r_r65 = CPyStatics[15]; /* 'sum' */
    PyObject *cpy_r_r66[1] = {cpy_r_t2};
    cpy_r_r67 = (PyObject **)&cpy_r_r66;
    cpy_r_r68 = PyObject_VectorcallMethod(cpy_r_r65, cpy_r_r67, 9223372036854775809ULL, 0);
    if (unlikely(cpy_r_r68 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 125, CPyStatic_globals);
        goto CPyL144;
    }
    CPy_DECREF(cpy_r_t2);
    cpy_r_r69 = (PyObject *)&PyFloat_Type;
    PyObject *cpy_r_r70[1] = {cpy_r_r68};
    cpy_r_r71 = (PyObject **)&cpy_r_r70;
    cpy_r_r72 = PyObject_Vectorcall(cpy_r_r69, cpy_r_r71, 1, 0);
    if (unlikely(cpy_r_r72 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 125, CPyStatic_globals);
        goto CPyL145;
    }
    CPy_DECREF(cpy_r_r68);
    cpy_r_r73 = PyFloat_AsDouble(cpy_r_r72);
    if (cpy_r_r73 == -1.0 && PyErr_Occurred()) {
        CPy_TypeError("float", cpy_r_r72); cpy_r_r73 = -113.0;
    }
    CPy_DECREF(cpy_r_r72);
    cpy_r_r74 = cpy_r_r73 == -113.0;
    if (unlikely(cpy_r_r74)) goto CPyL30;
CPyL29: ;
    cpy_r_s = cpy_r_r73;
    cpy_r_r75 = CPyModule_numpy;
    cpy_r_r76 = CPyStatics[11]; /* 'int32' */
    cpy_r_r77 = CPyObject_GetAttr(cpy_r_r75, cpy_r_r76);
    if (unlikely(cpy_r_r77 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 126, CPyStatic_globals);
        goto CPyL143;
    } else
        goto CPyL31;
CPyL30: ;
    cpy_r_r78 = PyErr_Occurred();
    if (unlikely(cpy_r_r78 != NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 125, CPyStatic_globals);
        goto CPyL143;
    } else
        goto CPyL29;
CPyL31: ;
    cpy_r_r79 = CPyModule_numpy;
    cpy_r_r80 = CPyStatics[12]; /* 'arange' */
    cpy_r_r81 = CPyObject_GetAttr(cpy_r_r79, cpy_r_r80);
    if (unlikely(cpy_r_r81 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 126, CPyStatic_globals);
        goto CPyL146;
    }
    cpy_r_r82 = CPyStatics[64]; /* 1 */
    PyObject *cpy_r_r83[2] = {cpy_r_r82, cpy_r_r77};
    cpy_r_r84 = (PyObject **)&cpy_r_r83;
    cpy_r_r85 = CPyStatics[68]; /* ('dtype',) */
    cpy_r_r86 = PyObject_Vectorcall(cpy_r_r81, cpy_r_r84, 1, cpy_r_r85);
    CPy_DECREF(cpy_r_r81);
    if (unlikely(cpy_r_r86 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 126, CPyStatic_globals);
        goto CPyL146;
    }
    CPy_DECREF(cpy_r_r77);
    cpy_r_ar = cpy_r_r86;
    cpy_r_r87 = CPyStatics[16]; /* 'copy' */
    PyObject *cpy_r_r88[1] = {cpy_r_t};
    cpy_r_r89 = (PyObject **)&cpy_r_r88;
    cpy_r_r90 = PyObject_VectorcallMethod(cpy_r_r87, cpy_r_r89, 9223372036854775809ULL, 0);
    if (unlikely(cpy_r_r90 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 127, CPyStatic_globals);
        goto CPyL147;
    }
    cpy_r_cl = cpy_r_r90;
    cpy_r_r91 = CPyModule_numpy;
    cpy_r_r92 = CPyStatics[7]; /* 'cumsum' */
    cpy_r_r93 = CPyObject_GetAttr(cpy_r_r91, cpy_r_r92);
    if (unlikely(cpy_r_r93 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 128, CPyStatic_globals);
        goto CPyL148;
    }
    cpy_r_r94 = CPyStatics[67]; /* 0 */
    PyObject *cpy_r_r95[2] = {cpy_r_t, cpy_r_r94};
    cpy_r_r96 = (PyObject **)&cpy_r_r95;
    cpy_r_r97 = CPyStatics[69]; /* ('axis',) */
    cpy_r_r98 = PyObject_Vectorcall(cpy_r_r93, cpy_r_r96, 1, cpy_r_r97);
    CPy_DECREF(cpy_r_r93);
    if (unlikely(cpy_r_r98 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 128, CPyStatic_globals);
        goto CPyL148;
    }
    cpy_r_cs = cpy_r_r98;
    cpy_r_r99 = CPyModule_numpy;
    cpy_r_r100 = CPyStatics[18]; /* 'float32' */
    cpy_r_r101 = CPyObject_GetAttr(cpy_r_r99, cpy_r_r100);
    if (unlikely(cpy_r_r101 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 129, CPyStatic_globals);
        goto CPyL149;
    }
    cpy_r_r102 = CPyModule_numpy;
    cpy_r_r103 = CPyStatics[9]; /* 'zeros' */
    cpy_r_r104 = CPyObject_GetAttr(cpy_r_r102, cpy_r_r103);
    if (unlikely(cpy_r_r104 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 129, CPyStatic_globals);
        goto CPyL150;
    }
    cpy_r_r105 = CPyStatics[64]; /* 1 */
    PyObject *cpy_r_r106[2] = {cpy_r_r105, cpy_r_r101};
    cpy_r_r107 = (PyObject **)&cpy_r_r106;
    cpy_r_r108 = CPyStatics[68]; /* ('dtype',) */
    cpy_r_r109 = PyObject_Vectorcall(cpy_r_r104, cpy_r_r107, 1, cpy_r_r108);
    CPy_DECREF(cpy_r_r104);
    if (unlikely(cpy_r_r109 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 129, CPyStatic_globals);
        goto CPyL150;
    }
    CPy_DECREF(cpy_r_r101);
    cpy_r_fz = cpy_r_r109;
    cpy_r_r110 = CPyStatics[19]; /* 'workspace' */
    cpy_r_r111 = CPyDict_GetItem(cpy_r_inp, cpy_r_r110);
    if (unlikely(cpy_r_r111 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 130, CPyStatic_globals);
        goto CPyL151;
    }
    cpy_r_r112 = (PyObject *)&_Py_NoneStruct;
    cpy_r_r113 = (PyObject *)&_Py_NoneStruct;
    cpy_r_r114 = CPyStatics[66]; /* 2 */
    cpy_r_r115 = PySlice_New(cpy_r_r112, cpy_r_r114, cpy_r_r113);
    if (unlikely(cpy_r_r115 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 130, CPyStatic_globals);
        goto CPyL152;
    }
    cpy_r_r116 = PyObject_GetItem(cpy_r_r111, cpy_r_r115);
    CPy_DECREF(cpy_r_r111);
    CPy_DECREF(cpy_r_r115);
    if (unlikely(cpy_r_r116 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 130, CPyStatic_globals);
        goto CPyL151;
    }
    cpy_r_ws = cpy_r_r116;
    cpy_r_r117 = CPyStatics[65]; /* -1 */
    cpy_r_r118 = PyObject_GetItem(cpy_r_r12, cpy_r_r117);
    CPy_DECREF(cpy_r_r12);
    if (unlikely(cpy_r_r118 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 131, CPyStatic_globals);
        goto CPyL153;
    }
    cpy_r_r119 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r120[1] = {cpy_r_r118};
    cpy_r_r121 = (PyObject **)&cpy_r_r120;
    cpy_r_r122 = PyObject_Vectorcall(cpy_r_r119, cpy_r_r121, 1, 0);
    if (unlikely(cpy_r_r122 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 131, CPyStatic_globals);
        goto CPyL154;
    }
    CPy_DECREF(cpy_r_r118);
    if (likely(PyLong_Check(cpy_r_r122)))
        cpy_r_r123 = CPyTagged_FromObject(cpy_r_r122);
    else {
        CPy_TypeError("int", cpy_r_r122); cpy_r_r123 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r122);
    if (unlikely(cpy_r_r123 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 131, CPyStatic_globals);
        goto CPyL153;
    }
    cpy_r_r124 = CPyStatics[15]; /* 'sum' */
    PyObject *cpy_r_r125[1] = {cpy_r_r23};
    cpy_r_r126 = (PyObject **)&cpy_r_r125;
    cpy_r_r127 = PyObject_VectorcallMethod(cpy_r_r124, cpy_r_r126, 9223372036854775809ULL, 0);
    if (unlikely(cpy_r_r127 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 131, CPyStatic_globals);
        goto CPyL155;
    }
    CPy_DECREF(cpy_r_r23);
    cpy_r_r128 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r129[1] = {cpy_r_r127};
    cpy_r_r130 = (PyObject **)&cpy_r_r129;
    cpy_r_r131 = PyObject_Vectorcall(cpy_r_r128, cpy_r_r130, 1, 0);
    if (unlikely(cpy_r_r131 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 131, CPyStatic_globals);
        goto CPyL156;
    }
    CPy_DECREF(cpy_r_r127);
    if (likely(PyLong_Check(cpy_r_r131)))
        cpy_r_r132 = CPyTagged_FromObject(cpy_r_r131);
    else {
        CPy_TypeError("int", cpy_r_r131); cpy_r_r132 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r131);
    if (unlikely(cpy_r_r132 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 131, CPyStatic_globals);
        goto CPyL157;
    }
    cpy_r_r133 = CPyStatics[15]; /* 'sum' */
    PyObject *cpy_r_r134[1] = {cpy_r_r34};
    cpy_r_r135 = (PyObject **)&cpy_r_r134;
    cpy_r_r136 = PyObject_VectorcallMethod(cpy_r_r133, cpy_r_r135, 9223372036854775809ULL, 0);
    if (unlikely(cpy_r_r136 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 131, CPyStatic_globals);
        goto CPyL158;
    }
    CPy_DECREF(cpy_r_r34);
    cpy_r_r137 = (PyObject *)&PyFloat_Type;
    PyObject *cpy_r_r138[1] = {cpy_r_r136};
    cpy_r_r139 = (PyObject **)&cpy_r_r138;
    cpy_r_r140 = PyObject_Vectorcall(cpy_r_r137, cpy_r_r139, 1, 0);
    if (unlikely(cpy_r_r140 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 131, CPyStatic_globals);
        goto CPyL159;
    }
    CPy_DECREF(cpy_r_r136);
    cpy_r_r141 = PyFloat_AsDouble(cpy_r_r140);
    if (cpy_r_r141 == -1.0 && PyErr_Occurred()) {
        CPy_TypeError("float", cpy_r_r140); cpy_r_r141 = -113.0;
    }
    CPy_DECREF(cpy_r_r140);
    cpy_r_r142 = cpy_r_r141 == -113.0;
    if (unlikely(cpy_r_r142)) goto CPyL52;
CPyL51: ;
    cpy_r_r143 = CPyStatics[65]; /* -1 */
    cpy_r_r144 = PyObject_GetItem(cpy_r_r45, cpy_r_r143);
    CPy_DECREF(cpy_r_r45);
    if (unlikely(cpy_r_r144 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 131, CPyStatic_globals);
        goto CPyL160;
    } else
        goto CPyL53;
CPyL52: ;
    cpy_r_r145 = PyErr_Occurred();
    if (unlikely(cpy_r_r145 != NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 131, CPyStatic_globals);
        goto CPyL161;
    } else
        goto CPyL51;
CPyL53: ;
    cpy_r_r146 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r147[1] = {cpy_r_r144};
    cpy_r_r148 = (PyObject **)&cpy_r_r147;
    cpy_r_r149 = PyObject_Vectorcall(cpy_r_r146, cpy_r_r148, 1, 0);
    if (unlikely(cpy_r_r149 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 131, CPyStatic_globals);
        goto CPyL162;
    }
    CPy_DECREF(cpy_r_r144);
    if (likely(PyLong_Check(cpy_r_r149)))
        cpy_r_r150 = CPyTagged_FromObject(cpy_r_r149);
    else {
        CPy_TypeError("int", cpy_r_r149); cpy_r_r150 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r149);
    if (unlikely(cpy_r_r150 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 131, CPyStatic_globals);
        goto CPyL160;
    }
    cpy_r_r151 = cpy_r_s;
    cpy_r_r152 = CPyStatics[67]; /* 0 */
    cpy_r_r153 = PyObject_RichCompare(cpy_r_t, cpy_r_r152, 4);
    CPy_DECREF(cpy_r_t);
    if (unlikely(cpy_r_r153 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 132, CPyStatic_globals);
        goto CPyL163;
    }
    cpy_r_r154 = CPyStatics[20]; /* 'any' */
    PyObject *cpy_r_r155[1] = {cpy_r_r153};
    cpy_r_r156 = (PyObject **)&cpy_r_r155;
    cpy_r_r157 = PyObject_VectorcallMethod(cpy_r_r154, cpy_r_r156, 9223372036854775809ULL, 0);
    if (unlikely(cpy_r_r157 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 132, CPyStatic_globals);
        goto CPyL164;
    }
    CPy_DECREF(cpy_r_r153);
    cpy_r_r158 = PyObject_IsTrue(cpy_r_r157);
    CPy_DECREF(cpy_r_r157);
    cpy_r_r159 = cpy_r_r158 >= 0;
    if (unlikely(!cpy_r_r159)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 132, CPyStatic_globals);
        goto CPyL163;
    }
    cpy_r_r160 = cpy_r_r158;
    cpy_r_r161 = CPyStatics[65]; /* -1 */
    cpy_r_r162 = PyObject_GetItem(cpy_r_cs, cpy_r_r161);
    CPy_DECREF(cpy_r_cs);
    if (unlikely(cpy_r_r162 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 133, CPyStatic_globals);
        goto CPyL165;
    }
    cpy_r_r163 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r164[1] = {cpy_r_r162};
    cpy_r_r165 = (PyObject **)&cpy_r_r164;
    cpy_r_r166 = PyObject_Vectorcall(cpy_r_r163, cpy_r_r165, 1, 0);
    if (unlikely(cpy_r_r166 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 133, CPyStatic_globals);
        goto CPyL166;
    }
    CPy_DECREF(cpy_r_r162);
    if (likely(PyLong_Check(cpy_r_r166)))
        cpy_r_r167 = CPyTagged_FromObject(cpy_r_r166);
    else {
        CPy_TypeError("int", cpy_r_r166); cpy_r_r167 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r166);
    if (unlikely(cpy_r_r167 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 133, CPyStatic_globals);
        goto CPyL165;
    }
    cpy_r_r168 = CPyStatics[65]; /* -1 */
    cpy_r_r169 = PyObject_GetItem(cpy_r_ar, cpy_r_r168);
    CPy_DECREF(cpy_r_ar);
    if (unlikely(cpy_r_r169 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 133, CPyStatic_globals);
        goto CPyL167;
    }
    cpy_r_r170 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r171[1] = {cpy_r_r169};
    cpy_r_r172 = (PyObject **)&cpy_r_r171;
    cpy_r_r173 = PyObject_Vectorcall(cpy_r_r170, cpy_r_r172, 1, 0);
    if (unlikely(cpy_r_r173 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 133, CPyStatic_globals);
        goto CPyL168;
    }
    CPy_DECREF(cpy_r_r169);
    if (likely(PyLong_Check(cpy_r_r173)))
        cpy_r_r174 = CPyTagged_FromObject(cpy_r_r173);
    else {
        CPy_TypeError("int", cpy_r_r173); cpy_r_r174 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r173);
    if (unlikely(cpy_r_r174 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 133, CPyStatic_globals);
        goto CPyL167;
    }
    cpy_r_r175 = CPyTagged_Add(cpy_r_r167, cpy_r_r174);
    CPyTagged_DECREF(cpy_r_r167);
    CPyTagged_DECREF(cpy_r_r174);
    cpy_r_r176 = CPyStatics[21]; /* 'max' */
    PyObject *cpy_r_r177[1] = {cpy_r_cl};
    cpy_r_r178 = (PyObject **)&cpy_r_r177;
    cpy_r_r179 = PyObject_VectorcallMethod(cpy_r_r176, cpy_r_r178, 9223372036854775809ULL, 0);
    if (unlikely(cpy_r_r179 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 133, CPyStatic_globals);
        goto CPyL169;
    }
    CPy_DECREF(cpy_r_cl);
    cpy_r_r180 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r181[1] = {cpy_r_r179};
    cpy_r_r182 = (PyObject **)&cpy_r_r181;
    cpy_r_r183 = PyObject_Vectorcall(cpy_r_r180, cpy_r_r182, 1, 0);
    if (unlikely(cpy_r_r183 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 133, CPyStatic_globals);
        goto CPyL170;
    }
    CPy_DECREF(cpy_r_r179);
    if (likely(PyLong_Check(cpy_r_r183)))
        cpy_r_r184 = CPyTagged_FromObject(cpy_r_r183);
    else {
        CPy_TypeError("int", cpy_r_r183); cpy_r_r184 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r183);
    if (unlikely(cpy_r_r184 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 133, CPyStatic_globals);
        goto CPyL171;
    }
    cpy_r_r185 = CPyTagged_Add(cpy_r_r175, cpy_r_r184);
    CPyTagged_DECREF(cpy_r_r175);
    CPyTagged_DECREF(cpy_r_r184);
    cpy_r_r186 = CPyStatics[22]; /* 'shape' */
    cpy_r_r187 = CPyObject_GetAttr(cpy_r_fz, cpy_r_r186);
    CPy_DECREF(cpy_r_fz);
    if (unlikely(cpy_r_r187 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 133, CPyStatic_globals);
        goto CPyL172;
    }
    cpy_r_r188 = CPyStatics[67]; /* 0 */
    cpy_r_r189 = PyObject_GetItem(cpy_r_r187, cpy_r_r188);
    CPy_DECREF(cpy_r_r187);
    if (unlikely(cpy_r_r189 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 133, CPyStatic_globals);
        goto CPyL172;
    }
    cpy_r_r190 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r191[1] = {cpy_r_r189};
    cpy_r_r192 = (PyObject **)&cpy_r_r191;
    cpy_r_r193 = PyObject_Vectorcall(cpy_r_r190, cpy_r_r192, 1, 0);
    if (unlikely(cpy_r_r193 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 133, CPyStatic_globals);
        goto CPyL173;
    }
    CPy_DECREF(cpy_r_r189);
    if (likely(PyLong_Check(cpy_r_r193)))
        cpy_r_r194 = CPyTagged_FromObject(cpy_r_r193);
    else {
        CPy_TypeError("int", cpy_r_r193); cpy_r_r194 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r193);
    if (unlikely(cpy_r_r194 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 133, CPyStatic_globals);
        goto CPyL172;
    }
    cpy_r_r195 = CPyTagged_Add(cpy_r_r185, cpy_r_r194);
    CPyTagged_DECREF(cpy_r_r185);
    CPyTagged_DECREF(cpy_r_r194);
    cpy_r_r196 = CPyStatics[22]; /* 'shape' */
    cpy_r_r197 = CPyObject_GetAttr(cpy_r_ws, cpy_r_r196);
    CPy_DECREF(cpy_r_ws);
    if (unlikely(cpy_r_r197 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 133, CPyStatic_globals);
        goto CPyL174;
    }
    cpy_r_r198 = CPyStatics[67]; /* 0 */
    cpy_r_r199 = PyObject_GetItem(cpy_r_r197, cpy_r_r198);
    CPy_DECREF(cpy_r_r197);
    if (unlikely(cpy_r_r199 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 133, CPyStatic_globals);
        goto CPyL174;
    }
    cpy_r_r200 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r201[1] = {cpy_r_r199};
    cpy_r_r202 = (PyObject **)&cpy_r_r201;
    cpy_r_r203 = PyObject_Vectorcall(cpy_r_r200, cpy_r_r202, 1, 0);
    if (unlikely(cpy_r_r203 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 133, CPyStatic_globals);
        goto CPyL175;
    }
    CPy_DECREF(cpy_r_r199);
    if (likely(PyLong_Check(cpy_r_r203)))
        cpy_r_r204 = CPyTagged_FromObject(cpy_r_r203);
    else {
        CPy_TypeError("int", cpy_r_r203); cpy_r_r204 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r203);
    if (unlikely(cpy_r_r204 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 133, CPyStatic_globals);
        goto CPyL174;
    }
    cpy_r_r205 = CPyTagged_Add(cpy_r_r195, cpy_r_r204);
    CPyTagged_DECREF(cpy_r_r195);
    CPyTagged_DECREF(cpy_r_r204);
    cpy_r_r206.f0 = cpy_r_r123;
    cpy_r_r206.f1 = cpy_r_r132;
    cpy_r_r206.f2 = cpy_r_r141;
    cpy_r_r206.f3 = cpy_r_r150;
    cpy_r_r206.f4 = cpy_r_r52;
    cpy_r_r206.f5 = cpy_r_r151;
    cpy_r_r206.f6 = cpy_r_r160;
    cpy_r_r206.f7 = cpy_r_r205;
    cpy_r_r207 = PyTuple_New(8);
    if (unlikely(cpy_r_r207 == NULL))
        CPyError_OutOfMemory();
    PyObject *__tmp9 = CPyTagged_StealAsObject(cpy_r_r206.f0);
    PyTuple_SET_ITEM(cpy_r_r207, 0, __tmp9);
    PyObject *__tmp10 = CPyTagged_StealAsObject(cpy_r_r206.f1);
    PyTuple_SET_ITEM(cpy_r_r207, 1, __tmp10);
    PyObject *__tmp11 = PyFloat_FromDouble(cpy_r_r206.f2);
    PyTuple_SET_ITEM(cpy_r_r207, 2, __tmp11);
    PyObject *__tmp12 = CPyTagged_StealAsObject(cpy_r_r206.f3);
    PyTuple_SET_ITEM(cpy_r_r207, 3, __tmp12);
    PyObject *__tmp13 = PyFloat_FromDouble(cpy_r_r206.f4);
    PyTuple_SET_ITEM(cpy_r_r207, 4, __tmp13);
    PyObject *__tmp14 = PyFloat_FromDouble(cpy_r_r206.f5);
    PyTuple_SET_ITEM(cpy_r_r207, 5, __tmp14);
    PyObject *__tmp15 = cpy_r_r206.f6 ? Py_True : Py_False;
    CPy_INCREF(__tmp15);
    PyTuple_SET_ITEM(cpy_r_r207, 6, __tmp15);
    PyObject *__tmp16 = CPyTagged_StealAsObject(cpy_r_r206.f7);
    PyTuple_SET_ITEM(cpy_r_r207, 7, __tmp16);
    return cpy_r_r207;
CPyL76: ;
    cpy_r_r208 = CPyModule_torch;
    cpy_r_r209 = CPyStatics[23]; /* 'from_numpy' */
    cpy_r_r210 = CPyObject_GetAttr(cpy_r_r208, cpy_r_r209);
    if (unlikely(cpy_r_r210 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 134, CPyStatic_globals);
        goto CPyL142;
    }
    PyObject *cpy_r_r211[1] = {cpy_r_r6};
    cpy_r_r212 = (PyObject **)&cpy_r_r211;
    cpy_r_r213 = PyObject_Vectorcall(cpy_r_r210, cpy_r_r212, 1, 0);
    CPy_DECREF(cpy_r_r210);
    if (unlikely(cpy_r_r213 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 134, CPyStatic_globals);
        goto CPyL142;
    }
    CPy_DECREF(cpy_r_r6);
    cpy_r_t = cpy_r_r213;
    cpy_r_r214 = CPyStatics[24]; /* 'mul' */
    cpy_r_r215 = CPyStatics[66]; /* 2 */
    PyObject *cpy_r_r216[2] = {cpy_r_t, cpy_r_r215};
    cpy_r_r217 = (PyObject **)&cpy_r_r216;
    cpy_r_r218 = PyObject_VectorcallMethod(cpy_r_r214, cpy_r_r217, 9223372036854775810ULL, 0);
    if (unlikely(cpy_r_r218 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 135, CPyStatic_globals);
        goto CPyL143;
    }
    cpy_r_t2 = cpy_r_r218;
    cpy_r_r219 = CPyStatics[15]; /* 'sum' */
    PyObject *cpy_r_r220[1] = {cpy_r_t2};
    cpy_r_r221 = (PyObject **)&cpy_r_r220;
    cpy_r_r222 = PyObject_VectorcallMethod(cpy_r_r219, cpy_r_r221, 9223372036854775809ULL, 0);
    if (unlikely(cpy_r_r222 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 136, CPyStatic_globals);
        goto CPyL144;
    }
    CPy_DECREF(cpy_r_t2);
    cpy_r_r223 = (PyObject *)&PyFloat_Type;
    PyObject *cpy_r_r224[1] = {cpy_r_r222};
    cpy_r_r225 = (PyObject **)&cpy_r_r224;
    cpy_r_r226 = PyObject_Vectorcall(cpy_r_r223, cpy_r_r225, 1, 0);
    if (unlikely(cpy_r_r226 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 136, CPyStatic_globals);
        goto CPyL176;
    }
    CPy_DECREF(cpy_r_r222);
    cpy_r_r227 = PyFloat_AsDouble(cpy_r_r226);
    if (cpy_r_r227 == -1.0 && PyErr_Occurred()) {
        CPy_TypeError("float", cpy_r_r226); cpy_r_r227 = -113.0;
    }
    CPy_DECREF(cpy_r_r226);
    cpy_r_r228 = cpy_r_r227 == -113.0;
    if (unlikely(cpy_r_r228)) goto CPyL83;
CPyL82: ;
    cpy_r_s = cpy_r_r227;
    cpy_r_r229 = CPyModule_torch;
    cpy_r_r230 = CPyStatics[11]; /* 'int32' */
    cpy_r_r231 = CPyObject_GetAttr(cpy_r_r229, cpy_r_r230);
    if (unlikely(cpy_r_r231 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 137, CPyStatic_globals);
        goto CPyL143;
    } else
        goto CPyL84;
CPyL83: ;
    cpy_r_r232 = PyErr_Occurred();
    if (unlikely(cpy_r_r232 != NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 136, CPyStatic_globals);
        goto CPyL143;
    } else
        goto CPyL82;
CPyL84: ;
    cpy_r_r233 = CPyModule_torch;
    cpy_r_r234 = CPyStatics[12]; /* 'arange' */
    cpy_r_r235 = CPyObject_GetAttr(cpy_r_r233, cpy_r_r234);
    if (unlikely(cpy_r_r235 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 137, CPyStatic_globals);
        goto CPyL177;
    }
    cpy_r_r236 = CPyStatics[64]; /* 1 */
    PyObject *cpy_r_r237[2] = {cpy_r_r236, cpy_r_r231};
    cpy_r_r238 = (PyObject **)&cpy_r_r237;
    cpy_r_r239 = CPyStatics[68]; /* ('dtype',) */
    cpy_r_r240 = PyObject_Vectorcall(cpy_r_r235, cpy_r_r238, 1, cpy_r_r239);
    CPy_DECREF(cpy_r_r235);
    if (unlikely(cpy_r_r240 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 137, CPyStatic_globals);
        goto CPyL177;
    }
    CPy_DECREF(cpy_r_r231);
    cpy_r_ar = cpy_r_r240;
    cpy_r_r241 = CPyStatics[25]; /* 'clone' */
    PyObject *cpy_r_r242[1] = {cpy_r_t};
    cpy_r_r243 = (PyObject **)&cpy_r_r242;
    cpy_r_r244 = PyObject_VectorcallMethod(cpy_r_r241, cpy_r_r243, 9223372036854775809ULL, 0);
    if (unlikely(cpy_r_r244 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 138, CPyStatic_globals);
        goto CPyL147;
    }
    cpy_r_cl = cpy_r_r244;
    cpy_r_r245 = CPyModule_torch;
    cpy_r_r246 = CPyStatics[7]; /* 'cumsum' */
    cpy_r_r247 = CPyObject_GetAttr(cpy_r_r245, cpy_r_r246);
    if (unlikely(cpy_r_r247 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 139, CPyStatic_globals);
        goto CPyL148;
    }
    cpy_r_r248 = CPyStatics[67]; /* 0 */
    PyObject *cpy_r_r249[2] = {cpy_r_t, cpy_r_r248};
    cpy_r_r250 = (PyObject **)&cpy_r_r249;
    cpy_r_r251 = PyObject_Vectorcall(cpy_r_r247, cpy_r_r250, 2, 0);
    CPy_DECREF(cpy_r_r247);
    if (unlikely(cpy_r_r251 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 139, CPyStatic_globals);
        goto CPyL148;
    }
    cpy_r_cs = cpy_r_r251;
    cpy_r_r252 = CPyModule_torch;
    cpy_r_r253 = CPyStatics[18]; /* 'float32' */
    cpy_r_r254 = CPyObject_GetAttr(cpy_r_r252, cpy_r_r253);
    if (unlikely(cpy_r_r254 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 140, CPyStatic_globals);
        goto CPyL149;
    }
    cpy_r_r255 = CPyModule_torch;
    cpy_r_r256 = CPyStatics[9]; /* 'zeros' */
    cpy_r_r257 = CPyObject_GetAttr(cpy_r_r255, cpy_r_r256);
    if (unlikely(cpy_r_r257 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 140, CPyStatic_globals);
        goto CPyL178;
    }
    cpy_r_r258 = CPyStatics[64]; /* 1 */
    PyObject *cpy_r_r259[2] = {cpy_r_r258, cpy_r_r254};
    cpy_r_r260 = (PyObject **)&cpy_r_r259;
    cpy_r_r261 = CPyStatics[68]; /* ('dtype',) */
    cpy_r_r262 = PyObject_Vectorcall(cpy_r_r257, cpy_r_r260, 1, cpy_r_r261);
    CPy_DECREF(cpy_r_r257);
    if (unlikely(cpy_r_r262 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 140, CPyStatic_globals);
        goto CPyL178;
    }
    CPy_DECREF(cpy_r_r254);
    cpy_r_fz = cpy_r_r262;
    cpy_r_r263 = CPyStatics[19]; /* 'workspace' */
    cpy_r_r264 = CPyDict_GetItem(cpy_r_inp, cpy_r_r263);
    if (unlikely(cpy_r_r264 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 141, CPyStatic_globals);
        goto CPyL151;
    }
    cpy_r_r265 = (PyObject *)&_Py_NoneStruct;
    cpy_r_r266 = (PyObject *)&_Py_NoneStruct;
    cpy_r_r267 = CPyStatics[66]; /* 2 */
    cpy_r_r268 = PySlice_New(cpy_r_r265, cpy_r_r267, cpy_r_r266);
    if (unlikely(cpy_r_r268 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 141, CPyStatic_globals);
        goto CPyL179;
    }
    cpy_r_r269 = PyObject_GetItem(cpy_r_r264, cpy_r_r268);
    CPy_DECREF(cpy_r_r264);
    CPy_DECREF(cpy_r_r268);
    if (unlikely(cpy_r_r269 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 141, CPyStatic_globals);
        goto CPyL151;
    }
    cpy_r_ws = cpy_r_r269;
    cpy_r_r270 = CPyStatics[65]; /* -1 */
    cpy_r_r271 = PyObject_GetItem(cpy_r_r12, cpy_r_r270);
    CPy_DECREF(cpy_r_r12);
    if (unlikely(cpy_r_r271 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 142, CPyStatic_globals);
        goto CPyL153;
    }
    cpy_r_r272 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r273[1] = {cpy_r_r271};
    cpy_r_r274 = (PyObject **)&cpy_r_r273;
    cpy_r_r275 = PyObject_Vectorcall(cpy_r_r272, cpy_r_r274, 1, 0);
    if (unlikely(cpy_r_r275 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 142, CPyStatic_globals);
        goto CPyL180;
    }
    CPy_DECREF(cpy_r_r271);
    if (likely(PyLong_Check(cpy_r_r275)))
        cpy_r_r276 = CPyTagged_FromObject(cpy_r_r275);
    else {
        CPy_TypeError("int", cpy_r_r275); cpy_r_r276 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r275);
    if (unlikely(cpy_r_r276 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 142, CPyStatic_globals);
        goto CPyL153;
    }
    cpy_r_r277 = CPyStatics[15]; /* 'sum' */
    PyObject *cpy_r_r278[1] = {cpy_r_r23};
    cpy_r_r279 = (PyObject **)&cpy_r_r278;
    cpy_r_r280 = PyObject_VectorcallMethod(cpy_r_r277, cpy_r_r279, 9223372036854775809ULL, 0);
    if (unlikely(cpy_r_r280 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 142, CPyStatic_globals);
        goto CPyL181;
    }
    CPy_DECREF(cpy_r_r23);
    cpy_r_r281 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r282[1] = {cpy_r_r280};
    cpy_r_r283 = (PyObject **)&cpy_r_r282;
    cpy_r_r284 = PyObject_Vectorcall(cpy_r_r281, cpy_r_r283, 1, 0);
    if (unlikely(cpy_r_r284 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 142, CPyStatic_globals);
        goto CPyL182;
    }
    CPy_DECREF(cpy_r_r280);
    if (likely(PyLong_Check(cpy_r_r284)))
        cpy_r_r285 = CPyTagged_FromObject(cpy_r_r284);
    else {
        CPy_TypeError("int", cpy_r_r284); cpy_r_r285 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r284);
    if (unlikely(cpy_r_r285 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 142, CPyStatic_globals);
        goto CPyL183;
    }
    cpy_r_r286 = CPyStatics[15]; /* 'sum' */
    PyObject *cpy_r_r287[1] = {cpy_r_r34};
    cpy_r_r288 = (PyObject **)&cpy_r_r287;
    cpy_r_r289 = PyObject_VectorcallMethod(cpy_r_r286, cpy_r_r288, 9223372036854775809ULL, 0);
    if (unlikely(cpy_r_r289 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 142, CPyStatic_globals);
        goto CPyL184;
    }
    CPy_DECREF(cpy_r_r34);
    cpy_r_r290 = (PyObject *)&PyFloat_Type;
    PyObject *cpy_r_r291[1] = {cpy_r_r289};
    cpy_r_r292 = (PyObject **)&cpy_r_r291;
    cpy_r_r293 = PyObject_Vectorcall(cpy_r_r290, cpy_r_r292, 1, 0);
    if (unlikely(cpy_r_r293 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 142, CPyStatic_globals);
        goto CPyL185;
    }
    CPy_DECREF(cpy_r_r289);
    cpy_r_r294 = PyFloat_AsDouble(cpy_r_r293);
    if (cpy_r_r294 == -1.0 && PyErr_Occurred()) {
        CPy_TypeError("float", cpy_r_r293); cpy_r_r294 = -113.0;
    }
    CPy_DECREF(cpy_r_r293);
    cpy_r_r295 = cpy_r_r294 == -113.0;
    if (unlikely(cpy_r_r295)) goto CPyL105;
CPyL104: ;
    cpy_r_r296 = CPyStatics[65]; /* -1 */
    cpy_r_r297 = PyObject_GetItem(cpy_r_r45, cpy_r_r296);
    CPy_DECREF(cpy_r_r45);
    if (unlikely(cpy_r_r297 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 142, CPyStatic_globals);
        goto CPyL186;
    } else
        goto CPyL106;
CPyL105: ;
    cpy_r_r298 = PyErr_Occurred();
    if (unlikely(cpy_r_r298 != NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 142, CPyStatic_globals);
        goto CPyL187;
    } else
        goto CPyL104;
CPyL106: ;
    cpy_r_r299 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r300[1] = {cpy_r_r297};
    cpy_r_r301 = (PyObject **)&cpy_r_r300;
    cpy_r_r302 = PyObject_Vectorcall(cpy_r_r299, cpy_r_r301, 1, 0);
    if (unlikely(cpy_r_r302 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 142, CPyStatic_globals);
        goto CPyL188;
    }
    CPy_DECREF(cpy_r_r297);
    if (likely(PyLong_Check(cpy_r_r302)))
        cpy_r_r303 = CPyTagged_FromObject(cpy_r_r302);
    else {
        CPy_TypeError("int", cpy_r_r302); cpy_r_r303 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r302);
    if (unlikely(cpy_r_r303 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 142, CPyStatic_globals);
        goto CPyL186;
    }
    cpy_r_r304 = cpy_r_s;
    cpy_r_r305 = CPyStatics[67]; /* 0 */
    cpy_r_r306 = PyObject_RichCompare(cpy_r_t, cpy_r_r305, 4);
    CPy_DECREF(cpy_r_t);
    if (unlikely(cpy_r_r306 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 143, CPyStatic_globals);
        goto CPyL189;
    }
    cpy_r_r307 = CPyStatics[20]; /* 'any' */
    PyObject *cpy_r_r308[1] = {cpy_r_r306};
    cpy_r_r309 = (PyObject **)&cpy_r_r308;
    cpy_r_r310 = PyObject_VectorcallMethod(cpy_r_r307, cpy_r_r309, 9223372036854775809ULL, 0);
    if (unlikely(cpy_r_r310 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 143, CPyStatic_globals);
        goto CPyL190;
    }
    CPy_DECREF(cpy_r_r306);
    cpy_r_r311 = PyObject_IsTrue(cpy_r_r310);
    CPy_DECREF(cpy_r_r310);
    cpy_r_r312 = cpy_r_r311 >= 0;
    if (unlikely(!cpy_r_r312)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 143, CPyStatic_globals);
        goto CPyL189;
    }
    cpy_r_r313 = cpy_r_r311;
    cpy_r_r314 = CPyStatics[65]; /* -1 */
    cpy_r_r315 = PyObject_GetItem(cpy_r_cs, cpy_r_r314);
    CPy_DECREF(cpy_r_cs);
    if (unlikely(cpy_r_r315 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 144, CPyStatic_globals);
        goto CPyL191;
    }
    cpy_r_r316 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r317[1] = {cpy_r_r315};
    cpy_r_r318 = (PyObject **)&cpy_r_r317;
    cpy_r_r319 = PyObject_Vectorcall(cpy_r_r316, cpy_r_r318, 1, 0);
    if (unlikely(cpy_r_r319 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 144, CPyStatic_globals);
        goto CPyL192;
    }
    CPy_DECREF(cpy_r_r315);
    if (likely(PyLong_Check(cpy_r_r319)))
        cpy_r_r320 = CPyTagged_FromObject(cpy_r_r319);
    else {
        CPy_TypeError("int", cpy_r_r319); cpy_r_r320 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r319);
    if (unlikely(cpy_r_r320 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 144, CPyStatic_globals);
        goto CPyL191;
    }
    cpy_r_r321 = CPyStatics[65]; /* -1 */
    cpy_r_r322 = PyObject_GetItem(cpy_r_ar, cpy_r_r321);
    CPy_DECREF(cpy_r_ar);
    if (unlikely(cpy_r_r322 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 144, CPyStatic_globals);
        goto CPyL193;
    }
    cpy_r_r323 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r324[1] = {cpy_r_r322};
    cpy_r_r325 = (PyObject **)&cpy_r_r324;
    cpy_r_r326 = PyObject_Vectorcall(cpy_r_r323, cpy_r_r325, 1, 0);
    if (unlikely(cpy_r_r326 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 144, CPyStatic_globals);
        goto CPyL194;
    }
    CPy_DECREF(cpy_r_r322);
    if (likely(PyLong_Check(cpy_r_r326)))
        cpy_r_r327 = CPyTagged_FromObject(cpy_r_r326);
    else {
        CPy_TypeError("int", cpy_r_r326); cpy_r_r327 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r326);
    if (unlikely(cpy_r_r327 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 144, CPyStatic_globals);
        goto CPyL193;
    }
    cpy_r_r328 = CPyTagged_Add(cpy_r_r320, cpy_r_r327);
    CPyTagged_DECREF(cpy_r_r320);
    CPyTagged_DECREF(cpy_r_r327);
    cpy_r_r329 = CPyStatics[21]; /* 'max' */
    PyObject *cpy_r_r330[1] = {cpy_r_cl};
    cpy_r_r331 = (PyObject **)&cpy_r_r330;
    cpy_r_r332 = PyObject_VectorcallMethod(cpy_r_r329, cpy_r_r331, 9223372036854775809ULL, 0);
    if (unlikely(cpy_r_r332 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 144, CPyStatic_globals);
        goto CPyL195;
    }
    CPy_DECREF(cpy_r_cl);
    cpy_r_r333 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r334[1] = {cpy_r_r332};
    cpy_r_r335 = (PyObject **)&cpy_r_r334;
    cpy_r_r336 = PyObject_Vectorcall(cpy_r_r333, cpy_r_r335, 1, 0);
    if (unlikely(cpy_r_r336 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 144, CPyStatic_globals);
        goto CPyL196;
    }
    CPy_DECREF(cpy_r_r332);
    if (likely(PyLong_Check(cpy_r_r336)))
        cpy_r_r337 = CPyTagged_FromObject(cpy_r_r336);
    else {
        CPy_TypeError("int", cpy_r_r336); cpy_r_r337 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r336);
    if (unlikely(cpy_r_r337 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 144, CPyStatic_globals);
        goto CPyL197;
    }
    cpy_r_r338 = CPyTagged_Add(cpy_r_r328, cpy_r_r337);
    CPyTagged_DECREF(cpy_r_r328);
    CPyTagged_DECREF(cpy_r_r337);
    cpy_r_r339 = CPyStatics[22]; /* 'shape' */
    cpy_r_r340 = CPyObject_GetAttr(cpy_r_fz, cpy_r_r339);
    CPy_DECREF(cpy_r_fz);
    if (unlikely(cpy_r_r340 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 144, CPyStatic_globals);
        goto CPyL198;
    }
    cpy_r_r341 = CPyStatics[67]; /* 0 */
    cpy_r_r342 = PyObject_GetItem(cpy_r_r340, cpy_r_r341);
    CPy_DECREF(cpy_r_r340);
    if (unlikely(cpy_r_r342 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 144, CPyStatic_globals);
        goto CPyL198;
    }
    cpy_r_r343 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r344[1] = {cpy_r_r342};
    cpy_r_r345 = (PyObject **)&cpy_r_r344;
    cpy_r_r346 = PyObject_Vectorcall(cpy_r_r343, cpy_r_r345, 1, 0);
    if (unlikely(cpy_r_r346 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 144, CPyStatic_globals);
        goto CPyL199;
    }
    CPy_DECREF(cpy_r_r342);
    if (likely(PyLong_Check(cpy_r_r346)))
        cpy_r_r347 = CPyTagged_FromObject(cpy_r_r346);
    else {
        CPy_TypeError("int", cpy_r_r346); cpy_r_r347 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r346);
    if (unlikely(cpy_r_r347 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 144, CPyStatic_globals);
        goto CPyL198;
    }
    cpy_r_r348 = CPyTagged_Add(cpy_r_r338, cpy_r_r347);
    CPyTagged_DECREF(cpy_r_r338);
    CPyTagged_DECREF(cpy_r_r347);
    cpy_r_r349 = CPyStatics[22]; /* 'shape' */
    cpy_r_r350 = CPyObject_GetAttr(cpy_r_ws, cpy_r_r349);
    CPy_DECREF(cpy_r_ws);
    if (unlikely(cpy_r_r350 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 144, CPyStatic_globals);
        goto CPyL200;
    }
    cpy_r_r351 = CPyStatics[67]; /* 0 */
    cpy_r_r352 = PyObject_GetItem(cpy_r_r350, cpy_r_r351);
    CPy_DECREF(cpy_r_r350);
    if (unlikely(cpy_r_r352 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 144, CPyStatic_globals);
        goto CPyL200;
    }
    cpy_r_r353 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r354[1] = {cpy_r_r352};
    cpy_r_r355 = (PyObject **)&cpy_r_r354;
    cpy_r_r356 = PyObject_Vectorcall(cpy_r_r353, cpy_r_r355, 1, 0);
    if (unlikely(cpy_r_r356 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 144, CPyStatic_globals);
        goto CPyL201;
    }
    CPy_DECREF(cpy_r_r352);
    if (likely(PyLong_Check(cpy_r_r356)))
        cpy_r_r357 = CPyTagged_FromObject(cpy_r_r356);
    else {
        CPy_TypeError("int", cpy_r_r356); cpy_r_r357 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r356);
    if (unlikely(cpy_r_r357 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 144, CPyStatic_globals);
        goto CPyL200;
    }
    cpy_r_r358 = CPyTagged_Add(cpy_r_r348, cpy_r_r357);
    CPyTagged_DECREF(cpy_r_r348);
    CPyTagged_DECREF(cpy_r_r357);
    cpy_r_r359.f0 = cpy_r_r276;
    cpy_r_r359.f1 = cpy_r_r285;
    cpy_r_r359.f2 = cpy_r_r294;
    cpy_r_r359.f3 = cpy_r_r303;
    cpy_r_r359.f4 = cpy_r_r52;
    cpy_r_r359.f5 = cpy_r_r304;
    cpy_r_r359.f6 = cpy_r_r313;
    cpy_r_r359.f7 = cpy_r_r358;
    cpy_r_r360 = PyTuple_New(8);
    if (unlikely(cpy_r_r360 == NULL))
        CPyError_OutOfMemory();
    PyObject *__tmp17 = CPyTagged_StealAsObject(cpy_r_r359.f0);
    PyTuple_SET_ITEM(cpy_r_r360, 0, __tmp17);
    PyObject *__tmp18 = CPyTagged_StealAsObject(cpy_r_r359.f1);
    PyTuple_SET_ITEM(cpy_r_r360, 1, __tmp18);
    PyObject *__tmp19 = PyFloat_FromDouble(cpy_r_r359.f2);
    PyTuple_SET_ITEM(cpy_r_r360, 2, __tmp19);
    PyObject *__tmp20 = CPyTagged_StealAsObject(cpy_r_r359.f3);
    PyTuple_SET_ITEM(cpy_r_r360, 3, __tmp20);
    PyObject *__tmp21 = PyFloat_FromDouble(cpy_r_r359.f4);
    PyTuple_SET_ITEM(cpy_r_r360, 4, __tmp21);
    PyObject *__tmp22 = PyFloat_FromDouble(cpy_r_r359.f5);
    PyTuple_SET_ITEM(cpy_r_r360, 5, __tmp22);
    PyObject *__tmp23 = cpy_r_r359.f6 ? Py_True : Py_False;
    CPy_INCREF(__tmp23);
    PyTuple_SET_ITEM(cpy_r_r360, 6, __tmp23);
    PyObject *__tmp24 = CPyTagged_StealAsObject(cpy_r_r359.f7);
    PyTuple_SET_ITEM(cpy_r_r360, 7, __tmp24);
    return cpy_r_r360;
CPyL129: ;
    cpy_r_r361 = NULL;
    return cpy_r_r361;
CPyL130: ;
    CPy_INCREF(cpy_r_mode);
    goto CPyL2;
CPyL131: ;
    CPy_DECREF(cpy_r_mode);
    goto CPyL3;
CPyL132: ;
    CPy_DecRef(cpy_r_mode);
    goto CPyL129;
CPyL133: ;
    CPy_DecRef(cpy_r_mode);
    CPy_DecRef(cpy_r_r6);
    goto CPyL129;
CPyL134: ;
    CPy_DecRef(cpy_r_mode);
    CPy_DecRef(cpy_r_r6);
    CPy_DecRef(cpy_r_r12);
    goto CPyL129;
CPyL135: ;
    CPy_DecRef(cpy_r_mode);
    CPy_DecRef(cpy_r_r6);
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r17);
    goto CPyL129;
CPyL136: ;
    CPy_DecRef(cpy_r_mode);
    CPy_DecRef(cpy_r_r6);
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    goto CPyL129;
CPyL137: ;
    CPy_DecRef(cpy_r_mode);
    CPy_DecRef(cpy_r_r6);
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r26);
    goto CPyL129;
CPyL138: ;
    CPy_DecRef(cpy_r_mode);
    CPy_DecRef(cpy_r_r6);
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    goto CPyL129;
CPyL139: ;
    CPy_DecRef(cpy_r_mode);
    CPy_DecRef(cpy_r_r6);
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r37);
    goto CPyL129;
CPyL140: ;
    CPy_DecRef(cpy_r_mode);
    CPy_DecRef(cpy_r_r6);
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    goto CPyL129;
CPyL141: ;
    CPy_DecRef(cpy_r_mode);
    CPy_DecRef(cpy_r_r6);
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_r47);
    goto CPyL129;
CPyL142: ;
    CPy_DecRef(cpy_r_r6);
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    goto CPyL129;
CPyL143: ;
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    goto CPyL129;
CPyL144: ;
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_t2);
    goto CPyL129;
CPyL145: ;
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_r68);
    goto CPyL129;
CPyL146: ;
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_r77);
    goto CPyL129;
CPyL147: ;
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    goto CPyL129;
CPyL148: ;
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    goto CPyL129;
CPyL149: ;
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    goto CPyL129;
CPyL150: ;
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_r101);
    goto CPyL129;
CPyL151: ;
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    goto CPyL129;
CPyL152: ;
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_r111);
    goto CPyL129;
CPyL153: ;
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    goto CPyL129;
CPyL154: ;
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPy_DecRef(cpy_r_r118);
    goto CPyL129;
CPyL155: ;
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r123);
    goto CPyL129;
CPyL156: ;
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r123);
    CPy_DecRef(cpy_r_r127);
    goto CPyL129;
CPyL157: ;
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r123);
    goto CPyL129;
CPyL158: ;
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r123);
    CPyTagged_DecRef(cpy_r_r132);
    goto CPyL129;
CPyL159: ;
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r123);
    CPyTagged_DecRef(cpy_r_r132);
    CPy_DecRef(cpy_r_r136);
    goto CPyL129;
CPyL160: ;
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r123);
    CPyTagged_DecRef(cpy_r_r132);
    goto CPyL129;
CPyL161: ;
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r123);
    CPyTagged_DecRef(cpy_r_r132);
    goto CPyL129;
CPyL162: ;
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r123);
    CPyTagged_DecRef(cpy_r_r132);
    CPy_DecRef(cpy_r_r144);
    goto CPyL129;
CPyL163: ;
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r123);
    CPyTagged_DecRef(cpy_r_r132);
    CPyTagged_DecRef(cpy_r_r150);
    goto CPyL129;
CPyL164: ;
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r123);
    CPyTagged_DecRef(cpy_r_r132);
    CPyTagged_DecRef(cpy_r_r150);
    CPy_DecRef(cpy_r_r153);
    goto CPyL129;
CPyL165: ;
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r123);
    CPyTagged_DecRef(cpy_r_r132);
    CPyTagged_DecRef(cpy_r_r150);
    goto CPyL129;
CPyL166: ;
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r123);
    CPyTagged_DecRef(cpy_r_r132);
    CPyTagged_DecRef(cpy_r_r150);
    CPy_DecRef(cpy_r_r162);
    goto CPyL129;
CPyL167: ;
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r123);
    CPyTagged_DecRef(cpy_r_r132);
    CPyTagged_DecRef(cpy_r_r150);
    CPyTagged_DecRef(cpy_r_r167);
    goto CPyL129;
CPyL168: ;
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r123);
    CPyTagged_DecRef(cpy_r_r132);
    CPyTagged_DecRef(cpy_r_r150);
    CPyTagged_DecRef(cpy_r_r167);
    CPy_DecRef(cpy_r_r169);
    goto CPyL129;
CPyL169: ;
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r123);
    CPyTagged_DecRef(cpy_r_r132);
    CPyTagged_DecRef(cpy_r_r150);
    CPyTagged_DecRef(cpy_r_r175);
    goto CPyL129;
CPyL170: ;
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r123);
    CPyTagged_DecRef(cpy_r_r132);
    CPyTagged_DecRef(cpy_r_r150);
    CPyTagged_DecRef(cpy_r_r175);
    CPy_DecRef(cpy_r_r179);
    goto CPyL129;
CPyL171: ;
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r123);
    CPyTagged_DecRef(cpy_r_r132);
    CPyTagged_DecRef(cpy_r_r150);
    CPyTagged_DecRef(cpy_r_r175);
    goto CPyL129;
CPyL172: ;
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r123);
    CPyTagged_DecRef(cpy_r_r132);
    CPyTagged_DecRef(cpy_r_r150);
    CPyTagged_DecRef(cpy_r_r185);
    goto CPyL129;
CPyL173: ;
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r123);
    CPyTagged_DecRef(cpy_r_r132);
    CPyTagged_DecRef(cpy_r_r150);
    CPyTagged_DecRef(cpy_r_r185);
    CPy_DecRef(cpy_r_r189);
    goto CPyL129;
CPyL174: ;
    CPyTagged_DecRef(cpy_r_r123);
    CPyTagged_DecRef(cpy_r_r132);
    CPyTagged_DecRef(cpy_r_r150);
    CPyTagged_DecRef(cpy_r_r195);
    goto CPyL129;
CPyL175: ;
    CPyTagged_DecRef(cpy_r_r123);
    CPyTagged_DecRef(cpy_r_r132);
    CPyTagged_DecRef(cpy_r_r150);
    CPyTagged_DecRef(cpy_r_r195);
    CPy_DecRef(cpy_r_r199);
    goto CPyL129;
CPyL176: ;
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_r222);
    goto CPyL129;
CPyL177: ;
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_r231);
    goto CPyL129;
CPyL178: ;
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_r254);
    goto CPyL129;
CPyL179: ;
    CPy_DecRef(cpy_r_r12);
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_r264);
    goto CPyL129;
CPyL180: ;
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPy_DecRef(cpy_r_r271);
    goto CPyL129;
CPyL181: ;
    CPy_DecRef(cpy_r_r23);
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r276);
    goto CPyL129;
CPyL182: ;
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r276);
    CPy_DecRef(cpy_r_r280);
    goto CPyL129;
CPyL183: ;
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r276);
    goto CPyL129;
CPyL184: ;
    CPy_DecRef(cpy_r_r34);
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r276);
    CPyTagged_DecRef(cpy_r_r285);
    goto CPyL129;
CPyL185: ;
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r276);
    CPyTagged_DecRef(cpy_r_r285);
    CPy_DecRef(cpy_r_r289);
    goto CPyL129;
CPyL186: ;
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r276);
    CPyTagged_DecRef(cpy_r_r285);
    goto CPyL129;
CPyL187: ;
    CPy_DecRef(cpy_r_r45);
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r276);
    CPyTagged_DecRef(cpy_r_r285);
    goto CPyL129;
CPyL188: ;
    CPy_DecRef(cpy_r_t);
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r276);
    CPyTagged_DecRef(cpy_r_r285);
    CPy_DecRef(cpy_r_r297);
    goto CPyL129;
CPyL189: ;
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r276);
    CPyTagged_DecRef(cpy_r_r285);
    CPyTagged_DecRef(cpy_r_r303);
    goto CPyL129;
CPyL190: ;
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_cs);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r276);
    CPyTagged_DecRef(cpy_r_r285);
    CPyTagged_DecRef(cpy_r_r303);
    CPy_DecRef(cpy_r_r306);
    goto CPyL129;
CPyL191: ;
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r276);
    CPyTagged_DecRef(cpy_r_r285);
    CPyTagged_DecRef(cpy_r_r303);
    goto CPyL129;
CPyL192: ;
    CPy_DecRef(cpy_r_ar);
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r276);
    CPyTagged_DecRef(cpy_r_r285);
    CPyTagged_DecRef(cpy_r_r303);
    CPy_DecRef(cpy_r_r315);
    goto CPyL129;
CPyL193: ;
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r276);
    CPyTagged_DecRef(cpy_r_r285);
    CPyTagged_DecRef(cpy_r_r303);
    CPyTagged_DecRef(cpy_r_r320);
    goto CPyL129;
CPyL194: ;
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r276);
    CPyTagged_DecRef(cpy_r_r285);
    CPyTagged_DecRef(cpy_r_r303);
    CPyTagged_DecRef(cpy_r_r320);
    CPy_DecRef(cpy_r_r322);
    goto CPyL129;
CPyL195: ;
    CPy_DecRef(cpy_r_cl);
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r276);
    CPyTagged_DecRef(cpy_r_r285);
    CPyTagged_DecRef(cpy_r_r303);
    CPyTagged_DecRef(cpy_r_r328);
    goto CPyL129;
CPyL196: ;
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r276);
    CPyTagged_DecRef(cpy_r_r285);
    CPyTagged_DecRef(cpy_r_r303);
    CPyTagged_DecRef(cpy_r_r328);
    CPy_DecRef(cpy_r_r332);
    goto CPyL129;
CPyL197: ;
    CPy_DecRef(cpy_r_fz);
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r276);
    CPyTagged_DecRef(cpy_r_r285);
    CPyTagged_DecRef(cpy_r_r303);
    CPyTagged_DecRef(cpy_r_r328);
    goto CPyL129;
CPyL198: ;
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r276);
    CPyTagged_DecRef(cpy_r_r285);
    CPyTagged_DecRef(cpy_r_r303);
    CPyTagged_DecRef(cpy_r_r338);
    goto CPyL129;
CPyL199: ;
    CPy_DecRef(cpy_r_ws);
    CPyTagged_DecRef(cpy_r_r276);
    CPyTagged_DecRef(cpy_r_r285);
    CPyTagged_DecRef(cpy_r_r303);
    CPyTagged_DecRef(cpy_r_r338);
    CPy_DecRef(cpy_r_r342);
    goto CPyL129;
CPyL200: ;
    CPyTagged_DecRef(cpy_r_r276);
    CPyTagged_DecRef(cpy_r_r285);
    CPyTagged_DecRef(cpy_r_r303);
    CPyTagged_DecRef(cpy_r_r348);
    goto CPyL129;
CPyL201: ;
    CPyTagged_DecRef(cpy_r_r276);
    CPyTagged_DecRef(cpy_r_r285);
    CPyTagged_DecRef(cpy_r_r303);
    CPyTagged_DecRef(cpy_r_r348);
    CPy_DecRef(cpy_r_r352);
    goto CPyL129;
}
    
    PyObject *CPyPy_small_ops(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames) {
        static const char * const kwlist[] = {"inp", "mode", 0};
        static CPyArg_Parser parser = {"O|O:small_ops", kwlist, 0};
        PyObject *obj_inp;
        PyObject *obj_mode = NULL;
        if (!CPyArg_ParseStackAndKeywordsSimple(args, nargs, kwnames, &parser, &obj_inp, &obj_mode)) {
            return NULL;
        }
        PyObject *arg_inp;
        if (likely(PyDict_Check(obj_inp)))
            arg_inp = obj_inp;
        else {
            CPy_TypeError("dict", obj_inp); 
            goto fail;
        }
        PyObject *arg_mode;
        if (obj_mode == NULL) {
            arg_mode = NULL;
        } else if (likely(PyUnicode_Check(obj_mode)))
            arg_mode = obj_mode;
        else {
            CPy_TypeError("str", obj_mode); 
            goto fail;
        }
        PyObject *retval = CPyDef_small_ops(arg_inp, arg_mode);
        return retval;
fail: ;
        CPy_AddTraceback("mypyc_variant.py", "small_ops", 113, CPyStatic_globals);
        return NULL;
    }
    
CPyTagged CPyDef__read_scalars(PyObject *cpy_r_m) {
    PyObject *cpy_r_r0;
    PyObject *cpy_r_r1;
    PyObject *cpy_r_r2;
    PyObject *cpy_r_r3;
    PyObject *cpy_r_r4;
    PyObject *cpy_r_r5;
    PyObject *cpy_r_r6;
    PyObject *cpy_r_r7;
    PyObject *cpy_r_r8;
    PyObject *cpy_r_r9;
    PyObject *cpy_r_r10;
    PyObject *cpy_r_r11;
    PyObject *cpy_r_r12;
    PyObject *cpy_r_r13;
    PyObject *cpy_r_r14;
    PyObject *cpy_r_r15;
    PyObject *cpy_r_r16;
    PyObject *cpy_r_r17;
    PyObject *cpy_r_r18;
    PyObject *cpy_r_r19;
    PyObject *cpy_r_r20;
    PyObject *cpy_r_r21;
    PyObject *cpy_r_r22;
    PyObject *cpy_r_r23;
    PyObject *cpy_r_r24;
    PyObject *cpy_r_r25;
    PyObject *cpy_r_r26;
    PyObject *cpy_r_r27;
    PyObject *cpy_r_r28;
    PyObject *cpy_r_r29;
    PyObject *cpy_r_r30;
    PyObject *cpy_r_r31;
    PyObject **cpy_r_r33;
    PyObject *cpy_r_r34;
    CPyTagged cpy_r_r35;
    PyObject *cpy_r_r36;
    PyObject *cpy_r_r37;
    PyObject *cpy_r_r38;
    PyObject *cpy_r_r39;
    PyObject *cpy_r_r40;
    PyObject **cpy_r_r42;
    PyObject *cpy_r_r43;
    CPyTagged cpy_r_r44;
    PyObject *cpy_r_r45;
    PyObject *cpy_r_r46;
    CPyTagged cpy_r_r47;
    CPyTagged cpy_r_r48;
    cpy_r_r0 = CPyStatics[26]; /* 'num_reqs' */
    cpy_r_r1 = CPyObject_GetAttr(cpy_r_m, cpy_r_r0);
    if (unlikely(cpy_r_r1 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 148, CPyStatic_globals);
        goto CPyL29;
    }
    cpy_r_r2 = CPyStatics[27]; /* 'num_actual_tokens' */
    cpy_r_r3 = CPyObject_GetAttr(cpy_r_m, cpy_r_r2);
    if (unlikely(cpy_r_r3 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 148, CPyStatic_globals);
        goto CPyL30;
    }
    cpy_r_r4 = PyNumber_Add(cpy_r_r1, cpy_r_r3);
    CPy_DECREF(cpy_r_r1);
    CPy_DECREF(cpy_r_r3);
    if (unlikely(cpy_r_r4 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 148, CPyStatic_globals);
        goto CPyL29;
    }
    cpy_r_r5 = CPyStatics[28]; /* 'max_query_len' */
    cpy_r_r6 = CPyObject_GetAttr(cpy_r_m, cpy_r_r5);
    if (unlikely(cpy_r_r6 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 148, CPyStatic_globals);
        goto CPyL31;
    }
    cpy_r_r7 = PyNumber_Add(cpy_r_r4, cpy_r_r6);
    CPy_DECREF(cpy_r_r4);
    CPy_DECREF(cpy_r_r6);
    if (unlikely(cpy_r_r7 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 148, CPyStatic_globals);
        goto CPyL29;
    }
    cpy_r_r8 = CPyStatics[29]; /* 'num_decode_tokens' */
    cpy_r_r9 = CPyObject_GetAttr(cpy_r_m, cpy_r_r8);
    if (unlikely(cpy_r_r9 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 148, CPyStatic_globals);
        goto CPyL32;
    }
    cpy_r_r10 = PyNumber_Add(cpy_r_r7, cpy_r_r9);
    CPy_DECREF(cpy_r_r7);
    CPy_DECREF(cpy_r_r9);
    if (unlikely(cpy_r_r10 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 148, CPyStatic_globals);
        goto CPyL29;
    }
    cpy_r_r11 = CPyStatics[30]; /* 'num_prefill_tokens' */
    cpy_r_r12 = CPyObject_GetAttr(cpy_r_m, cpy_r_r11);
    if (unlikely(cpy_r_r12 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 149, CPyStatic_globals);
        goto CPyL33;
    }
    cpy_r_r13 = PyNumber_Add(cpy_r_r10, cpy_r_r12);
    CPy_DECREF(cpy_r_r10);
    CPy_DECREF(cpy_r_r12);
    if (unlikely(cpy_r_r13 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 148, CPyStatic_globals);
        goto CPyL29;
    }
    cpy_r_r14 = CPyStatics[31]; /* 'max_seq_len' */
    cpy_r_r15 = CPyObject_GetAttr(cpy_r_m, cpy_r_r14);
    if (unlikely(cpy_r_r15 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 149, CPyStatic_globals);
        goto CPyL34;
    }
    cpy_r_r16 = PyNumber_Add(cpy_r_r13, cpy_r_r15);
    CPy_DECREF(cpy_r_r13);
    CPy_DECREF(cpy_r_r15);
    if (unlikely(cpy_r_r16 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 148, CPyStatic_globals);
        goto CPyL29;
    }
    cpy_r_r17 = CPyStatics[32]; /* 'block_size' */
    cpy_r_r18 = CPyObject_GetAttr(cpy_r_m, cpy_r_r17);
    if (unlikely(cpy_r_r18 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 149, CPyStatic_globals);
        goto CPyL35;
    }
    cpy_r_r19 = PyNumber_Add(cpy_r_r16, cpy_r_r18);
    CPy_DECREF(cpy_r_r16);
    CPy_DECREF(cpy_r_r18);
    if (unlikely(cpy_r_r19 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 148, CPyStatic_globals);
        goto CPyL29;
    }
    cpy_r_r20 = CPyStatics[33]; /* 'num_kv_heads' */
    cpy_r_r21 = CPyObject_GetAttr(cpy_r_m, cpy_r_r20);
    if (unlikely(cpy_r_r21 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 149, CPyStatic_globals);
        goto CPyL36;
    }
    cpy_r_r22 = PyNumber_Add(cpy_r_r19, cpy_r_r21);
    CPy_DECREF(cpy_r_r19);
    CPy_DECREF(cpy_r_r21);
    if (unlikely(cpy_r_r22 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 148, CPyStatic_globals);
        goto CPyL29;
    }
    cpy_r_r23 = CPyStatics[34]; /* 'head_size' */
    cpy_r_r24 = CPyObject_GetAttr(cpy_r_m, cpy_r_r23);
    if (unlikely(cpy_r_r24 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 150, CPyStatic_globals);
        goto CPyL37;
    }
    cpy_r_r25 = PyNumber_Add(cpy_r_r22, cpy_r_r24);
    CPy_DECREF(cpy_r_r22);
    CPy_DECREF(cpy_r_r24);
    if (unlikely(cpy_r_r25 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 148, CPyStatic_globals);
        goto CPyL29;
    }
    cpy_r_r26 = CPyStatics[35]; /* 'sliding_window' */
    cpy_r_r27 = CPyObject_GetAttr(cpy_r_m, cpy_r_r26);
    if (unlikely(cpy_r_r27 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 150, CPyStatic_globals);
        goto CPyL38;
    }
    cpy_r_r28 = PyNumber_Add(cpy_r_r25, cpy_r_r27);
    CPy_DECREF(cpy_r_r25);
    CPy_DECREF(cpy_r_r27);
    if (unlikely(cpy_r_r28 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 148, CPyStatic_globals);
        goto CPyL29;
    }
    cpy_r_r29 = CPyStatics[36]; /* 'spec_decode' */
    cpy_r_r30 = CPyObject_GetAttr(cpy_r_m, cpy_r_r29);
    if (unlikely(cpy_r_r30 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 150, CPyStatic_globals);
        goto CPyL39;
    }
    cpy_r_r31 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r32[1] = {cpy_r_r30};
    cpy_r_r33 = (PyObject **)&cpy_r_r32;
    cpy_r_r34 = PyObject_Vectorcall(cpy_r_r31, cpy_r_r33, 1, 0);
    if (unlikely(cpy_r_r34 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 150, CPyStatic_globals);
        goto CPyL40;
    }
    CPy_DECREF(cpy_r_r30);
    if (likely(PyLong_Check(cpy_r_r34)))
        cpy_r_r35 = CPyTagged_FromObject(cpy_r_r34);
    else {
        CPy_TypeError("int", cpy_r_r34); cpy_r_r35 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r34);
    if (unlikely(cpy_r_r35 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 150, CPyStatic_globals);
        goto CPyL39;
    }
    cpy_r_r36 = CPyTagged_StealAsObject(cpy_r_r35);
    cpy_r_r37 = PyNumber_Add(cpy_r_r28, cpy_r_r36);
    CPy_DECREF(cpy_r_r28);
    CPy_DECREF(cpy_r_r36);
    if (unlikely(cpy_r_r37 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 148, CPyStatic_globals);
        goto CPyL29;
    }
    cpy_r_r38 = CPyStatics[37]; /* 'causal' */
    cpy_r_r39 = CPyObject_GetAttr(cpy_r_m, cpy_r_r38);
    if (unlikely(cpy_r_r39 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 150, CPyStatic_globals);
        goto CPyL41;
    }
    cpy_r_r40 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r41[1] = {cpy_r_r39};
    cpy_r_r42 = (PyObject **)&cpy_r_r41;
    cpy_r_r43 = PyObject_Vectorcall(cpy_r_r40, cpy_r_r42, 1, 0);
    if (unlikely(cpy_r_r43 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 150, CPyStatic_globals);
        goto CPyL42;
    }
    CPy_DECREF(cpy_r_r39);
    if (likely(PyLong_Check(cpy_r_r43)))
        cpy_r_r44 = CPyTagged_FromObject(cpy_r_r43);
    else {
        CPy_TypeError("int", cpy_r_r43); cpy_r_r44 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r43);
    if (unlikely(cpy_r_r44 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 150, CPyStatic_globals);
        goto CPyL41;
    }
    cpy_r_r45 = CPyTagged_StealAsObject(cpy_r_r44);
    cpy_r_r46 = PyNumber_Add(cpy_r_r37, cpy_r_r45);
    CPy_DECREF(cpy_r_r37);
    CPy_DECREF(cpy_r_r45);
    if (unlikely(cpy_r_r46 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 148, CPyStatic_globals);
        goto CPyL29;
    }
    if (likely(PyLong_Check(cpy_r_r46)))
        cpy_r_r47 = CPyTagged_FromObject(cpy_r_r46);
    else {
        CPy_TypeError("int", cpy_r_r46); cpy_r_r47 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r46);
    if (unlikely(cpy_r_r47 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 148, CPyStatic_globals);
        goto CPyL29;
    }
    return cpy_r_r47;
CPyL29: ;
    cpy_r_r48 = CPY_INT_TAG;
    return cpy_r_r48;
CPyL30: ;
    CPy_DecRef(cpy_r_r1);
    goto CPyL29;
CPyL31: ;
    CPy_DecRef(cpy_r_r4);
    goto CPyL29;
CPyL32: ;
    CPy_DecRef(cpy_r_r7);
    goto CPyL29;
CPyL33: ;
    CPy_DecRef(cpy_r_r10);
    goto CPyL29;
CPyL34: ;
    CPy_DecRef(cpy_r_r13);
    goto CPyL29;
CPyL35: ;
    CPy_DecRef(cpy_r_r16);
    goto CPyL29;
CPyL36: ;
    CPy_DecRef(cpy_r_r19);
    goto CPyL29;
CPyL37: ;
    CPy_DecRef(cpy_r_r22);
    goto CPyL29;
CPyL38: ;
    CPy_DecRef(cpy_r_r25);
    goto CPyL29;
CPyL39: ;
    CPy_DecRef(cpy_r_r28);
    goto CPyL29;
CPyL40: ;
    CPy_DecRef(cpy_r_r28);
    CPy_DecRef(cpy_r_r30);
    goto CPyL29;
CPyL41: ;
    CPy_DecRef(cpy_r_r37);
    goto CPyL29;
CPyL42: ;
    CPy_DecRef(cpy_r_r37);
    CPy_DecRef(cpy_r_r39);
    goto CPyL29;
}
    
    PyObject *CPyPy__read_scalars(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames) {
        static const char * const kwlist[] = {"m", 0};
        static CPyArg_Parser parser = {"O:_read_scalars", kwlist, 0};
        PyObject *obj_m;
        if (!CPyArg_ParseStackAndKeywordsOneArg(args, nargs, kwnames, &parser, &obj_m)) {
            return NULL;
        }
        PyObject *arg_m = obj_m;
        CPyTagged retval = CPyDef__read_scalars(arg_m);
        if (retval == CPY_INT_TAG) {
            return NULL;
        }
        PyObject *retbox = CPyTagged_StealAsObject(retval);
        return retbox;
fail: ;
        CPy_AddTraceback("mypyc_variant.py", "_read_scalars", 147, CPyStatic_globals);
        return NULL;
    }
    
CPyTagged CPyDef__read_containers(PyObject *cpy_r_m) {
    PyObject *cpy_r_r0;
    PyObject *cpy_r_r1;
    CPyTagged cpy_r_r2;
    CPyTagged cpy_r_acc;
    PyObject *cpy_r_r3;
    PyObject *cpy_r_r4;
    PyObject *cpy_r_r5;
    PyObject *cpy_r_r6;
    PyObject *cpy_r_r7;
    PyObject **cpy_r_r9;
    PyObject *cpy_r_r10;
    CPyTagged cpy_r_r11;
    CPyTagged cpy_r_r12;
    PyObject *cpy_r_r13;
    PyObject *cpy_r_r14;
    PyObject *cpy_r_r15;
    PyObject *cpy_r_r16;
    PyObject *cpy_r_r17;
    PyObject **cpy_r_r19;
    PyObject *cpy_r_r20;
    CPyTagged cpy_r_r21;
    CPyTagged cpy_r_r22;
    PyObject *cpy_r_r23;
    PyObject *cpy_r_r24;
    PyObject *cpy_r_r25;
    PyObject *cpy_r_r26;
    PyObject *cpy_r_r27;
    PyObject **cpy_r_r29;
    PyObject *cpy_r_r30;
    CPyTagged cpy_r_r31;
    CPyTagged cpy_r_r32;
    PyObject *cpy_r_r33;
    PyObject *cpy_r_r34;
    PyObject *cpy_r_r35;
    PyObject *cpy_r_r36;
    PyObject *cpy_r_r37;
    PyObject *cpy_r_r38;
    PyObject *cpy_r_r39;
    PyObject **cpy_r_r41;
    PyObject *cpy_r_r42;
    CPyTagged cpy_r_r43;
    CPyTagged cpy_r_r44;
    PyObject *cpy_r_r45;
    PyObject *cpy_r_r46;
    PyObject *cpy_r_r47;
    PyObject *cpy_r_r48;
    PyObject *cpy_r_r49;
    PyObject *cpy_r_r50;
    PyObject *cpy_r_r51;
    PyObject *cpy_r_r52;
    PyObject *cpy_r_r53;
    PyObject *cpy_r_r54;
    CPyTagged cpy_r_r55;
    PyObject *cpy_r_r56;
    PyObject *cpy_r_r57;
    PyObject *cpy_r_r58;
    PyObject *cpy_r_r59;
    PyObject *cpy_r_r60;
    PyObject **cpy_r_r62;
    PyObject *cpy_r_r63;
    CPyTagged cpy_r_r64;
    CPyTagged cpy_r_r65;
    PyObject *cpy_r_r66;
    PyObject *cpy_r_r67;
    PyObject *cpy_r_r68;
    PyObject *cpy_r_r69;
    PyObject *cpy_r_r70;
    PyObject *cpy_r_r71;
    PyObject *cpy_r_r72;
    PyObject **cpy_r_r74;
    PyObject *cpy_r_r75;
    CPyTagged cpy_r_r76;
    CPyTagged cpy_r_r77;
    PyObject *cpy_r_r78;
    PyObject *cpy_r_r79;
    PyObject *cpy_r_r80;
    PyObject *cpy_r_r81;
    PyObject *cpy_r_r82;
    PyObject **cpy_r_r84;
    PyObject *cpy_r_r85;
    CPyTagged cpy_r_r86;
    CPyTagged cpy_r_r87;
    PyObject *cpy_r_r88;
    PyObject *cpy_r_r89;
    PyObject *cpy_r_r90;
    PyObject *cpy_r_r91;
    PyObject *cpy_r_r92;
    PyObject **cpy_r_r94;
    PyObject *cpy_r_r95;
    CPyTagged cpy_r_r96;
    CPyTagged cpy_r_r97;
    PyObject *cpy_r_r98;
    PyObject *cpy_r_r99;
    PyObject *cpy_r_r100;
    PyObject *cpy_r_r101;
    PyObject *cpy_r_r102;
    PyObject **cpy_r_r104;
    PyObject *cpy_r_r105;
    CPyTagged cpy_r_r106;
    CPyTagged cpy_r_r107;
    CPyTagged cpy_r_r108;
    cpy_r_r0 = CPyStatics[38]; /* 'kv_cache_dtype' */
    cpy_r_r1 = CPyObject_GetAttr(cpy_r_m, cpy_r_r0);
    if (unlikely(cpy_r_r1 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 155, CPyStatic_globals);
        goto CPyL48;
    }
    cpy_r_r2 = CPyObject_Size(cpy_r_r1);
    CPy_DECREF(cpy_r_r1);
    if (unlikely(cpy_r_r2 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 155, CPyStatic_globals);
        goto CPyL48;
    }
    cpy_r_acc = cpy_r_r2;
    cpy_r_r3 = CPyStatics[39]; /* 'query_start_loc' */
    cpy_r_r4 = CPyObject_GetAttr(cpy_r_m, cpy_r_r3);
    if (unlikely(cpy_r_r4 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 156, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r5 = CPyStatics[65]; /* -1 */
    cpy_r_r6 = PyObject_GetItem(cpy_r_r4, cpy_r_r5);
    CPy_DECREF(cpy_r_r4);
    if (unlikely(cpy_r_r6 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 156, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r7 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r8[1] = {cpy_r_r6};
    cpy_r_r9 = (PyObject **)&cpy_r_r8;
    cpy_r_r10 = PyObject_Vectorcall(cpy_r_r7, cpy_r_r9, 1, 0);
    if (unlikely(cpy_r_r10 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 156, CPyStatic_globals);
        goto CPyL50;
    }
    CPy_DECREF(cpy_r_r6);
    if (likely(PyLong_Check(cpy_r_r10)))
        cpy_r_r11 = CPyTagged_FromObject(cpy_r_r10);
    else {
        CPy_TypeError("int", cpy_r_r10); cpy_r_r11 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r10);
    if (unlikely(cpy_r_r11 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 156, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r12 = CPyTagged_Add(cpy_r_acc, cpy_r_r11);
    CPyTagged_DECREF(cpy_r_acc);
    CPyTagged_DECREF(cpy_r_r11);
    cpy_r_acc = cpy_r_r12;
    cpy_r_r13 = CPyStatics[40]; /* 'slot_mapping' */
    cpy_r_r14 = CPyObject_GetAttr(cpy_r_m, cpy_r_r13);
    if (unlikely(cpy_r_r14 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 157, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r15 = CPyStatics[65]; /* -1 */
    cpy_r_r16 = PyObject_GetItem(cpy_r_r14, cpy_r_r15);
    CPy_DECREF(cpy_r_r14);
    if (unlikely(cpy_r_r16 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 157, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r17 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r18[1] = {cpy_r_r16};
    cpy_r_r19 = (PyObject **)&cpy_r_r18;
    cpy_r_r20 = PyObject_Vectorcall(cpy_r_r17, cpy_r_r19, 1, 0);
    if (unlikely(cpy_r_r20 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 157, CPyStatic_globals);
        goto CPyL51;
    }
    CPy_DECREF(cpy_r_r16);
    if (likely(PyLong_Check(cpy_r_r20)))
        cpy_r_r21 = CPyTagged_FromObject(cpy_r_r20);
    else {
        CPy_TypeError("int", cpy_r_r20); cpy_r_r21 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r20);
    if (unlikely(cpy_r_r21 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 157, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r22 = CPyTagged_Add(cpy_r_acc, cpy_r_r21);
    CPyTagged_DECREF(cpy_r_acc);
    CPyTagged_DECREF(cpy_r_r21);
    cpy_r_acc = cpy_r_r22;
    cpy_r_r23 = CPyStatics[41]; /* 'token_ids' */
    cpy_r_r24 = CPyObject_GetAttr(cpy_r_m, cpy_r_r23);
    if (unlikely(cpy_r_r24 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 158, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r25 = CPyStatics[65]; /* -1 */
    cpy_r_r26 = PyObject_GetItem(cpy_r_r24, cpy_r_r25);
    CPy_DECREF(cpy_r_r24);
    if (unlikely(cpy_r_r26 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 158, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r27 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r28[1] = {cpy_r_r26};
    cpy_r_r29 = (PyObject **)&cpy_r_r28;
    cpy_r_r30 = PyObject_Vectorcall(cpy_r_r27, cpy_r_r29, 1, 0);
    if (unlikely(cpy_r_r30 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 158, CPyStatic_globals);
        goto CPyL52;
    }
    CPy_DECREF(cpy_r_r26);
    if (likely(PyLong_Check(cpy_r_r30)))
        cpy_r_r31 = CPyTagged_FromObject(cpy_r_r30);
    else {
        CPy_TypeError("int", cpy_r_r30); cpy_r_r31 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r30);
    if (unlikely(cpy_r_r31 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 158, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r32 = CPyTagged_Add(cpy_r_acc, cpy_r_r31);
    CPyTagged_DECREF(cpy_r_acc);
    CPyTagged_DECREF(cpy_r_r31);
    cpy_r_acc = cpy_r_r32;
    cpy_r_r33 = CPyStatics[19]; /* 'workspace' */
    cpy_r_r34 = CPyObject_GetAttr(cpy_r_m, cpy_r_r33);
    if (unlikely(cpy_r_r34 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 159, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r35 = CPyStatics[22]; /* 'shape' */
    cpy_r_r36 = CPyObject_GetAttr(cpy_r_r34, cpy_r_r35);
    CPy_DECREF(cpy_r_r34);
    if (unlikely(cpy_r_r36 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 159, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r37 = CPyStatics[67]; /* 0 */
    cpy_r_r38 = PyObject_GetItem(cpy_r_r36, cpy_r_r37);
    CPy_DECREF(cpy_r_r36);
    if (unlikely(cpy_r_r38 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 159, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r39 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r40[1] = {cpy_r_r38};
    cpy_r_r41 = (PyObject **)&cpy_r_r40;
    cpy_r_r42 = PyObject_Vectorcall(cpy_r_r39, cpy_r_r41, 1, 0);
    if (unlikely(cpy_r_r42 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 159, CPyStatic_globals);
        goto CPyL53;
    }
    CPy_DECREF(cpy_r_r38);
    if (likely(PyLong_Check(cpy_r_r42)))
        cpy_r_r43 = CPyTagged_FromObject(cpy_r_r42);
    else {
        CPy_TypeError("int", cpy_r_r42); cpy_r_r43 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r42);
    if (unlikely(cpy_r_r43 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 159, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r44 = CPyTagged_Add(cpy_r_acc, cpy_r_r43);
    CPyTagged_DECREF(cpy_r_acc);
    CPyTagged_DECREF(cpy_r_r43);
    cpy_r_acc = cpy_r_r44;
    cpy_r_r45 = CPyStatics[26]; /* 'num_reqs' */
    cpy_r_r46 = CPyObject_GetAttr(cpy_r_m, cpy_r_r45);
    if (unlikely(cpy_r_r46 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 160, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r47 = CPyStatics[27]; /* 'num_actual_tokens' */
    cpy_r_r48 = CPyObject_GetAttr(cpy_r_m, cpy_r_r47);
    if (unlikely(cpy_r_r48 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 160, CPyStatic_globals);
        goto CPyL54;
    }
    cpy_r_r49 = PyNumber_Add(cpy_r_r46, cpy_r_r48);
    CPy_DECREF(cpy_r_r46);
    CPy_DECREF(cpy_r_r48);
    if (unlikely(cpy_r_r49 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 160, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r50 = CPyStatics[28]; /* 'max_query_len' */
    cpy_r_r51 = CPyObject_GetAttr(cpy_r_m, cpy_r_r50);
    if (unlikely(cpy_r_r51 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 160, CPyStatic_globals);
        goto CPyL55;
    }
    cpy_r_r52 = PyNumber_Add(cpy_r_r49, cpy_r_r51);
    CPy_DECREF(cpy_r_r49);
    CPy_DECREF(cpy_r_r51);
    if (unlikely(cpy_r_r52 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 160, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r53 = CPyTagged_StealAsObject(cpy_r_acc);
    cpy_r_r54 = PyNumber_InPlaceAdd(cpy_r_r53, cpy_r_r52);
    CPy_DECREF(cpy_r_r53);
    CPy_DECREF(cpy_r_r52);
    if (unlikely(cpy_r_r54 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 160, CPyStatic_globals);
        goto CPyL48;
    }
    if (likely(PyLong_Check(cpy_r_r54)))
        cpy_r_r55 = CPyTagged_FromObject(cpy_r_r54);
    else {
        CPy_TypeError("int", cpy_r_r54); cpy_r_r55 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r54);
    if (unlikely(cpy_r_r55 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 160, CPyStatic_globals);
        goto CPyL48;
    }
    cpy_r_acc = cpy_r_r55;
    cpy_r_r56 = CPyStatics[6]; /* 'seq_lens_np' */
    cpy_r_r57 = CPyObject_GetAttr(cpy_r_m, cpy_r_r56);
    if (unlikely(cpy_r_r57 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 161, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r58 = CPyStatics[65]; /* -1 */
    cpy_r_r59 = PyObject_GetItem(cpy_r_r57, cpy_r_r58);
    CPy_DECREF(cpy_r_r57);
    if (unlikely(cpy_r_r59 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 161, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r60 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r61[1] = {cpy_r_r59};
    cpy_r_r62 = (PyObject **)&cpy_r_r61;
    cpy_r_r63 = PyObject_Vectorcall(cpy_r_r60, cpy_r_r62, 1, 0);
    if (unlikely(cpy_r_r63 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 161, CPyStatic_globals);
        goto CPyL56;
    }
    CPy_DECREF(cpy_r_r59);
    if (likely(PyLong_Check(cpy_r_r63)))
        cpy_r_r64 = CPyTagged_FromObject(cpy_r_r63);
    else {
        CPy_TypeError("int", cpy_r_r63); cpy_r_r64 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r63);
    if (unlikely(cpy_r_r64 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 161, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r65 = CPyTagged_Add(cpy_r_acc, cpy_r_r64);
    CPyTagged_DECREF(cpy_r_acc);
    CPyTagged_DECREF(cpy_r_r64);
    cpy_r_acc = cpy_r_r65;
    cpy_r_r66 = CPyStatics[42]; /* 'block_table' */
    cpy_r_r67 = CPyObject_GetAttr(cpy_r_m, cpy_r_r66);
    if (unlikely(cpy_r_r67 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 162, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r68 = CPyStatics[22]; /* 'shape' */
    cpy_r_r69 = CPyObject_GetAttr(cpy_r_r67, cpy_r_r68);
    CPy_DECREF(cpy_r_r67);
    if (unlikely(cpy_r_r69 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 162, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r70 = CPyStatics[64]; /* 1 */
    cpy_r_r71 = PyObject_GetItem(cpy_r_r69, cpy_r_r70);
    CPy_DECREF(cpy_r_r69);
    if (unlikely(cpy_r_r71 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 162, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r72 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r73[1] = {cpy_r_r71};
    cpy_r_r74 = (PyObject **)&cpy_r_r73;
    cpy_r_r75 = PyObject_Vectorcall(cpy_r_r72, cpy_r_r74, 1, 0);
    if (unlikely(cpy_r_r75 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 162, CPyStatic_globals);
        goto CPyL57;
    }
    CPy_DECREF(cpy_r_r71);
    if (likely(PyLong_Check(cpy_r_r75)))
        cpy_r_r76 = CPyTagged_FromObject(cpy_r_r75);
    else {
        CPy_TypeError("int", cpy_r_r75); cpy_r_r76 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r75);
    if (unlikely(cpy_r_r76 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 162, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r77 = CPyTagged_Add(cpy_r_acc, cpy_r_r76);
    CPyTagged_DECREF(cpy_r_acc);
    CPyTagged_DECREF(cpy_r_r76);
    cpy_r_acc = cpy_r_r77;
    cpy_r_r78 = CPyStatics[43]; /* 'positions' */
    cpy_r_r79 = CPyObject_GetAttr(cpy_r_m, cpy_r_r78);
    if (unlikely(cpy_r_r79 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 163, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r80 = CPyStatics[65]; /* -1 */
    cpy_r_r81 = PyObject_GetItem(cpy_r_r79, cpy_r_r80);
    CPy_DECREF(cpy_r_r79);
    if (unlikely(cpy_r_r81 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 163, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r82 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r83[1] = {cpy_r_r81};
    cpy_r_r84 = (PyObject **)&cpy_r_r83;
    cpy_r_r85 = PyObject_Vectorcall(cpy_r_r82, cpy_r_r84, 1, 0);
    if (unlikely(cpy_r_r85 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 163, CPyStatic_globals);
        goto CPyL58;
    }
    CPy_DECREF(cpy_r_r81);
    if (likely(PyLong_Check(cpy_r_r85)))
        cpy_r_r86 = CPyTagged_FromObject(cpy_r_r85);
    else {
        CPy_TypeError("int", cpy_r_r85); cpy_r_r86 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r85);
    if (unlikely(cpy_r_r86 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 163, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r87 = CPyTagged_Add(cpy_r_acc, cpy_r_r86);
    CPyTagged_DECREF(cpy_r_acc);
    CPyTagged_DECREF(cpy_r_r86);
    cpy_r_acc = cpy_r_r87;
    cpy_r_r88 = CPyStatics[44]; /* 'num_computed_tokens' */
    cpy_r_r89 = CPyObject_GetAttr(cpy_r_m, cpy_r_r88);
    if (unlikely(cpy_r_r89 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 164, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r90 = CPyStatics[65]; /* -1 */
    cpy_r_r91 = PyObject_GetItem(cpy_r_r89, cpy_r_r90);
    CPy_DECREF(cpy_r_r89);
    if (unlikely(cpy_r_r91 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 164, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r92 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r93[1] = {cpy_r_r91};
    cpy_r_r94 = (PyObject **)&cpy_r_r93;
    cpy_r_r95 = PyObject_Vectorcall(cpy_r_r92, cpy_r_r94, 1, 0);
    if (unlikely(cpy_r_r95 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 164, CPyStatic_globals);
        goto CPyL59;
    }
    CPy_DECREF(cpy_r_r91);
    if (likely(PyLong_Check(cpy_r_r95)))
        cpy_r_r96 = CPyTagged_FromObject(cpy_r_r95);
    else {
        CPy_TypeError("int", cpy_r_r95); cpy_r_r96 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r95);
    if (unlikely(cpy_r_r96 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 164, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r97 = CPyTagged_Add(cpy_r_acc, cpy_r_r96);
    CPyTagged_DECREF(cpy_r_acc);
    CPyTagged_DECREF(cpy_r_r96);
    cpy_r_acc = cpy_r_r97;
    cpy_r_r98 = CPyStatics[45]; /* 'seq_lens' */
    cpy_r_r99 = CPyObject_GetAttr(cpy_r_m, cpy_r_r98);
    if (unlikely(cpy_r_r99 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 165, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r100 = CPyStatics[65]; /* -1 */
    cpy_r_r101 = PyObject_GetItem(cpy_r_r99, cpy_r_r100);
    CPy_DECREF(cpy_r_r99);
    if (unlikely(cpy_r_r101 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 165, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r102 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r103[1] = {cpy_r_r101};
    cpy_r_r104 = (PyObject **)&cpy_r_r103;
    cpy_r_r105 = PyObject_Vectorcall(cpy_r_r102, cpy_r_r104, 1, 0);
    if (unlikely(cpy_r_r105 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 165, CPyStatic_globals);
        goto CPyL60;
    }
    CPy_DECREF(cpy_r_r101);
    if (likely(PyLong_Check(cpy_r_r105)))
        cpy_r_r106 = CPyTagged_FromObject(cpy_r_r105);
    else {
        CPy_TypeError("int", cpy_r_r105); cpy_r_r106 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r105);
    if (unlikely(cpy_r_r106 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 165, CPyStatic_globals);
        goto CPyL49;
    }
    cpy_r_r107 = CPyTagged_Add(cpy_r_acc, cpy_r_r106);
    CPyTagged_DECREF(cpy_r_acc);
    CPyTagged_DECREF(cpy_r_r106);
    cpy_r_acc = cpy_r_r107;
    return cpy_r_acc;
CPyL48: ;
    cpy_r_r108 = CPY_INT_TAG;
    return cpy_r_r108;
CPyL49: ;
    CPyTagged_DecRef(cpy_r_acc);
    goto CPyL48;
CPyL50: ;
    CPyTagged_DecRef(cpy_r_acc);
    CPy_DecRef(cpy_r_r6);
    goto CPyL48;
CPyL51: ;
    CPyTagged_DecRef(cpy_r_acc);
    CPy_DecRef(cpy_r_r16);
    goto CPyL48;
CPyL52: ;
    CPyTagged_DecRef(cpy_r_acc);
    CPy_DecRef(cpy_r_r26);
    goto CPyL48;
CPyL53: ;
    CPyTagged_DecRef(cpy_r_acc);
    CPy_DecRef(cpy_r_r38);
    goto CPyL48;
CPyL54: ;
    CPyTagged_DecRef(cpy_r_acc);
    CPy_DecRef(cpy_r_r46);
    goto CPyL48;
CPyL55: ;
    CPyTagged_DecRef(cpy_r_acc);
    CPy_DecRef(cpy_r_r49);
    goto CPyL48;
CPyL56: ;
    CPyTagged_DecRef(cpy_r_acc);
    CPy_DecRef(cpy_r_r59);
    goto CPyL48;
CPyL57: ;
    CPyTagged_DecRef(cpy_r_acc);
    CPy_DecRef(cpy_r_r71);
    goto CPyL48;
CPyL58: ;
    CPyTagged_DecRef(cpy_r_acc);
    CPy_DecRef(cpy_r_r81);
    goto CPyL48;
CPyL59: ;
    CPyTagged_DecRef(cpy_r_acc);
    CPy_DecRef(cpy_r_r91);
    goto CPyL48;
CPyL60: ;
    CPyTagged_DecRef(cpy_r_acc);
    CPy_DecRef(cpy_r_r101);
    goto CPyL48;
}
    
    PyObject *CPyPy__read_containers(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames) {
        static const char * const kwlist[] = {"m", 0};
        static CPyArg_Parser parser = {"O:_read_containers", kwlist, 0};
        PyObject *obj_m;
        if (!CPyArg_ParseStackAndKeywordsOneArg(args, nargs, kwnames, &parser, &obj_m)) {
            return NULL;
        }
        PyObject *arg_m = obj_m;
        CPyTagged retval = CPyDef__read_containers(arg_m);
        if (retval == CPY_INT_TAG) {
            return NULL;
        }
        PyObject *retbox = CPyTagged_StealAsObject(retval);
        return retbox;
fail: ;
        CPy_AddTraceback("mypyc_variant.py", "_read_containers", 154, CPyStatic_globals);
        return NULL;
    }
    
CPyTagged CPyDef_run_mypyc_native(PyObject *cpy_r_inp, PyObject *cpy_r_mode) {
    PyObject *cpy_r_r0;
    PyObject *cpy_r_r1;
    PyObject *cpy_r_r2;
    PyObject *cpy_r_r3;
    PyObject *cpy_r_r4;
    PyObject *cpy_r_r5;
    PyObject *cpy_r_r6;
    CPyTagged cpy_r_r7;
    CPyTagged cpy_r_r8;
    CPyTagged cpy_r_r9;
    CPyTagged cpy_r_acc;
    PyObject *cpy_r_r10;
    PyObject *cpy_r_r11;
    PyObject **cpy_r_r13;
    PyObject *cpy_r_r14;
    CPyTagged cpy_r_r15;
    PyObject *cpy_r_r16;
    PyObject *cpy_r_r17;
    PyObject **cpy_r_r19;
    PyObject *cpy_r_r20;
    CPyTagged cpy_r_r21;
    CPyTagged cpy_r_r22;
    PyObject *cpy_r_r23;
    PyObject *cpy_r_r24;
    PyObject **cpy_r_r26;
    PyObject *cpy_r_r27;
    CPyTagged cpy_r_r28;
    CPyTagged cpy_r_r29;
    PyObject *cpy_r_r30;
    PyObject *cpy_r_r31;
    PyObject **cpy_r_r33;
    PyObject *cpy_r_r34;
    CPyTagged cpy_r_r35;
    CPyTagged cpy_r_r36;
    PyObject *cpy_r_r37;
    PyObject *cpy_r_r38;
    PyObject **cpy_r_r40;
    PyObject *cpy_r_r41;
    CPyTagged cpy_r_r42;
    CPyTagged cpy_r_r43;
    CPyPtr cpy_r_r44;
    int64_t cpy_r_r45;
    CPyTagged cpy_r_r46;
    CPyTagged cpy_r_r47;
    CPyTagged cpy_r_r48;
    CPyTagged cpy_r_r49;
    if (cpy_r_mode != NULL) goto CPyL26;
    cpy_r_r0 = CPyStatics[4]; /* 'torch' */
    CPy_INCREF(cpy_r_r0);
    cpy_r_mode = cpy_r_r0;
CPyL2: ;
    cpy_r_r1 = CPyDef__kwargs(cpy_r_inp);
    if (unlikely(cpy_r_r1 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 170, CPyStatic_globals);
        goto CPyL27;
    }
    cpy_r_r2 = (PyObject *)CPyType_MetaNative;
    cpy_r_r3 = PyList_AsTuple(cpy_r_r1);
    CPy_DECREF_NO_IMM(cpy_r_r1);
    if (unlikely(cpy_r_r3 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 170, CPyStatic_globals);
        goto CPyL27;
    }
    cpy_r_r4 = PyObject_CallObject(cpy_r_r2, cpy_r_r3);
    CPy_DECREF(cpy_r_r3);
    if (unlikely(cpy_r_r4 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 170, CPyStatic_globals);
        goto CPyL27;
    }
    if (likely(Py_TYPE(cpy_r_r4) == CPyType_MetaNative))
        cpy_r_r5 = cpy_r_r4;
    else {
        CPy_TypeErrorTraceback("mypyc_variant.py", "run_mypyc_native", 170, CPyStatic_globals, "mypyc_variant.MetaNative", cpy_r_r4);
        goto CPyL27;
    }
    cpy_r_r6 = CPyDef_small_ops(cpy_r_inp, cpy_r_mode);
    CPy_DECREF(cpy_r_mode);
    if (unlikely(cpy_r_r6 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 171, CPyStatic_globals);
        goto CPyL28;
    }
    cpy_r_r7 = CPyDef__read_scalars(cpy_r_r5);
    if (unlikely(cpy_r_r7 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 172, CPyStatic_globals);
        goto CPyL29;
    }
    cpy_r_r8 = CPyDef__read_containers(cpy_r_r5);
    CPy_DECREF_NO_IMM(cpy_r_r5);
    if (unlikely(cpy_r_r8 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 172, CPyStatic_globals);
        goto CPyL30;
    }
    cpy_r_r9 = CPyTagged_Add(cpy_r_r7, cpy_r_r8);
    CPyTagged_DECREF(cpy_r_r7);
    CPyTagged_DECREF(cpy_r_r8);
    cpy_r_acc = cpy_r_r9;
    cpy_r_r10 = CPySequenceTuple_GetItem(cpy_r_r6, 0);
    if (unlikely(cpy_r_r10 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 173, CPyStatic_globals);
        goto CPyL31;
    }
    cpy_r_r11 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r12[1] = {cpy_r_r10};
    cpy_r_r13 = (PyObject **)&cpy_r_r12;
    cpy_r_r14 = PyObject_Vectorcall(cpy_r_r11, cpy_r_r13, 1, 0);
    if (unlikely(cpy_r_r14 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 173, CPyStatic_globals);
        goto CPyL32;
    }
    CPy_DECREF(cpy_r_r10);
    if (likely(PyLong_Check(cpy_r_r14)))
        cpy_r_r15 = CPyTagged_FromObject(cpy_r_r14);
    else {
        CPy_TypeError("int", cpy_r_r14); cpy_r_r15 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r14);
    if (unlikely(cpy_r_r15 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 173, CPyStatic_globals);
        goto CPyL31;
    }
    cpy_r_r16 = CPySequenceTuple_GetItem(cpy_r_r6, 2);
    if (unlikely(cpy_r_r16 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 173, CPyStatic_globals);
        goto CPyL33;
    }
    cpy_r_r17 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r18[1] = {cpy_r_r16};
    cpy_r_r19 = (PyObject **)&cpy_r_r18;
    cpy_r_r20 = PyObject_Vectorcall(cpy_r_r17, cpy_r_r19, 1, 0);
    if (unlikely(cpy_r_r20 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 173, CPyStatic_globals);
        goto CPyL34;
    }
    CPy_DECREF(cpy_r_r16);
    if (likely(PyLong_Check(cpy_r_r20)))
        cpy_r_r21 = CPyTagged_FromObject(cpy_r_r20);
    else {
        CPy_TypeError("int", cpy_r_r20); cpy_r_r21 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r20);
    if (unlikely(cpy_r_r21 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 173, CPyStatic_globals);
        goto CPyL33;
    }
    cpy_r_r22 = CPyTagged_Add(cpy_r_r15, cpy_r_r21);
    CPyTagged_DECREF(cpy_r_r15);
    CPyTagged_DECREF(cpy_r_r21);
    cpy_r_r23 = CPySequenceTuple_GetItem(cpy_r_r6, 6);
    if (unlikely(cpy_r_r23 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 173, CPyStatic_globals);
        goto CPyL35;
    }
    cpy_r_r24 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r25[1] = {cpy_r_r23};
    cpy_r_r26 = (PyObject **)&cpy_r_r25;
    cpy_r_r27 = PyObject_Vectorcall(cpy_r_r24, cpy_r_r26, 1, 0);
    if (unlikely(cpy_r_r27 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 173, CPyStatic_globals);
        goto CPyL36;
    }
    CPy_DECREF(cpy_r_r23);
    if (likely(PyLong_Check(cpy_r_r27)))
        cpy_r_r28 = CPyTagged_FromObject(cpy_r_r27);
    else {
        CPy_TypeError("int", cpy_r_r27); cpy_r_r28 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r27);
    if (unlikely(cpy_r_r28 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 173, CPyStatic_globals);
        goto CPyL35;
    }
    cpy_r_r29 = CPyTagged_Add(cpy_r_r22, cpy_r_r28);
    CPyTagged_DECREF(cpy_r_r22);
    CPyTagged_DECREF(cpy_r_r28);
    cpy_r_r30 = CPySequenceTuple_GetItem(cpy_r_r6, 8);
    if (unlikely(cpy_r_r30 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 173, CPyStatic_globals);
        goto CPyL37;
    }
    cpy_r_r31 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r32[1] = {cpy_r_r30};
    cpy_r_r33 = (PyObject **)&cpy_r_r32;
    cpy_r_r34 = PyObject_Vectorcall(cpy_r_r31, cpy_r_r33, 1, 0);
    if (unlikely(cpy_r_r34 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 173, CPyStatic_globals);
        goto CPyL38;
    }
    CPy_DECREF(cpy_r_r30);
    if (likely(PyLong_Check(cpy_r_r34)))
        cpy_r_r35 = CPyTagged_FromObject(cpy_r_r34);
    else {
        CPy_TypeError("int", cpy_r_r34); cpy_r_r35 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r34);
    if (unlikely(cpy_r_r35 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 173, CPyStatic_globals);
        goto CPyL37;
    }
    cpy_r_r36 = CPyTagged_Add(cpy_r_r29, cpy_r_r35);
    CPyTagged_DECREF(cpy_r_r29);
    CPyTagged_DECREF(cpy_r_r35);
    cpy_r_r37 = CPySequenceTuple_GetItem(cpy_r_r6, 14);
    if (unlikely(cpy_r_r37 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 173, CPyStatic_globals);
        goto CPyL39;
    }
    cpy_r_r38 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r39[1] = {cpy_r_r37};
    cpy_r_r40 = (PyObject **)&cpy_r_r39;
    cpy_r_r41 = PyObject_Vectorcall(cpy_r_r38, cpy_r_r40, 1, 0);
    if (unlikely(cpy_r_r41 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 173, CPyStatic_globals);
        goto CPyL40;
    }
    CPy_DECREF(cpy_r_r37);
    if (likely(PyLong_Check(cpy_r_r41)))
        cpy_r_r42 = CPyTagged_FromObject(cpy_r_r41);
    else {
        CPy_TypeError("int", cpy_r_r41); cpy_r_r42 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r41);
    if (unlikely(cpy_r_r42 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 173, CPyStatic_globals);
        goto CPyL39;
    }
    cpy_r_r43 = CPyTagged_Add(cpy_r_r36, cpy_r_r42);
    CPyTagged_DECREF(cpy_r_r36);
    CPyTagged_DECREF(cpy_r_r42);
    cpy_r_r44 = (CPyPtr)((CPyPtr)cpy_r_r6 + offsetof(PyVarObject, ob_size));
    cpy_r_r45 = *(int64_t *)cpy_r_r44;
    CPy_DECREF(cpy_r_r6);
    cpy_r_r46 = cpy_r_r45 << 1;
    cpy_r_r47 = CPyTagged_Add(cpy_r_r43, cpy_r_r46);
    CPyTagged_DECREF(cpy_r_r43);
    cpy_r_r48 = CPyTagged_Add(cpy_r_acc, cpy_r_r47);
    CPyTagged_DECREF(cpy_r_acc);
    CPyTagged_DECREF(cpy_r_r47);
    cpy_r_acc = cpy_r_r48;
    return cpy_r_acc;
CPyL25: ;
    cpy_r_r49 = CPY_INT_TAG;
    return cpy_r_r49;
CPyL26: ;
    CPy_INCREF(cpy_r_mode);
    goto CPyL2;
CPyL27: ;
    CPy_DecRef(cpy_r_mode);
    goto CPyL25;
CPyL28: ;
    CPy_DecRef(cpy_r_r5);
    goto CPyL25;
CPyL29: ;
    CPy_DecRef(cpy_r_r5);
    CPy_DecRef(cpy_r_r6);
    goto CPyL25;
CPyL30: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_r7);
    goto CPyL25;
CPyL31: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    goto CPyL25;
CPyL32: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    CPy_DecRef(cpy_r_r10);
    goto CPyL25;
CPyL33: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    CPyTagged_DecRef(cpy_r_r15);
    goto CPyL25;
CPyL34: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    CPyTagged_DecRef(cpy_r_r15);
    CPy_DecRef(cpy_r_r16);
    goto CPyL25;
CPyL35: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    CPyTagged_DecRef(cpy_r_r22);
    goto CPyL25;
CPyL36: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    CPyTagged_DecRef(cpy_r_r22);
    CPy_DecRef(cpy_r_r23);
    goto CPyL25;
CPyL37: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    CPyTagged_DecRef(cpy_r_r29);
    goto CPyL25;
CPyL38: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    CPyTagged_DecRef(cpy_r_r29);
    CPy_DecRef(cpy_r_r30);
    goto CPyL25;
CPyL39: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    CPyTagged_DecRef(cpy_r_r36);
    goto CPyL25;
CPyL40: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    CPyTagged_DecRef(cpy_r_r36);
    CPy_DecRef(cpy_r_r37);
    goto CPyL25;
}
    
    PyObject *CPyPy_run_mypyc_native(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames) {
        static const char * const kwlist[] = {"inp", "mode", 0};
        static CPyArg_Parser parser = {"O|O:run_mypyc_native", kwlist, 0};
        PyObject *obj_inp;
        PyObject *obj_mode = NULL;
        if (!CPyArg_ParseStackAndKeywordsSimple(args, nargs, kwnames, &parser, &obj_inp, &obj_mode)) {
            return NULL;
        }
        PyObject *arg_inp;
        if (likely(PyDict_Check(obj_inp)))
            arg_inp = obj_inp;
        else {
            CPy_TypeError("dict", obj_inp); 
            goto fail;
        }
        PyObject *arg_mode;
        if (obj_mode == NULL) {
            arg_mode = NULL;
        } else if (likely(PyUnicode_Check(obj_mode)))
            arg_mode = obj_mode;
        else {
            CPy_TypeError("str", obj_mode); 
            goto fail;
        }
        CPyTagged retval = CPyDef_run_mypyc_native(arg_inp, arg_mode);
        if (retval == CPY_INT_TAG) {
            return NULL;
        }
        PyObject *retbox = CPyTagged_StealAsObject(retval);
        return retbox;
fail: ;
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_native", 169, CPyStatic_globals);
        return NULL;
    }
    
CPyTagged CPyDef_run_mypyc_dataclass(PyObject *cpy_r_inp, PyObject *cpy_r_mode) {
    PyObject *cpy_r_r0;
    PyObject *cpy_r_r1;
    PyObject *cpy_r_r2;
    PyObject *cpy_r_r3;
    PyObject *cpy_r_r4;
    PyObject *cpy_r_r5;
    PyObject *cpy_r_r6;
    CPyTagged cpy_r_r7;
    CPyTagged cpy_r_r8;
    CPyTagged cpy_r_r9;
    CPyTagged cpy_r_acc;
    PyObject *cpy_r_r10;
    PyObject *cpy_r_r11;
    PyObject **cpy_r_r13;
    PyObject *cpy_r_r14;
    CPyTagged cpy_r_r15;
    PyObject *cpy_r_r16;
    PyObject *cpy_r_r17;
    PyObject **cpy_r_r19;
    PyObject *cpy_r_r20;
    CPyTagged cpy_r_r21;
    CPyTagged cpy_r_r22;
    PyObject *cpy_r_r23;
    PyObject *cpy_r_r24;
    PyObject **cpy_r_r26;
    PyObject *cpy_r_r27;
    CPyTagged cpy_r_r28;
    CPyTagged cpy_r_r29;
    PyObject *cpy_r_r30;
    PyObject *cpy_r_r31;
    PyObject **cpy_r_r33;
    PyObject *cpy_r_r34;
    CPyTagged cpy_r_r35;
    CPyTagged cpy_r_r36;
    PyObject *cpy_r_r37;
    PyObject *cpy_r_r38;
    PyObject **cpy_r_r40;
    PyObject *cpy_r_r41;
    CPyTagged cpy_r_r42;
    CPyTagged cpy_r_r43;
    CPyPtr cpy_r_r44;
    int64_t cpy_r_r45;
    CPyTagged cpy_r_r46;
    CPyTagged cpy_r_r47;
    CPyTagged cpy_r_r48;
    CPyTagged cpy_r_r49;
    if (cpy_r_mode != NULL) goto CPyL26;
    cpy_r_r0 = CPyStatics[4]; /* 'torch' */
    CPy_INCREF(cpy_r_r0);
    cpy_r_mode = cpy_r_r0;
CPyL2: ;
    cpy_r_r1 = CPyDef__kwargs(cpy_r_inp);
    if (unlikely(cpy_r_r1 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 178, CPyStatic_globals);
        goto CPyL27;
    }
    cpy_r_r2 = (PyObject *)CPyType_MetaDataclass;
    cpy_r_r3 = PyList_AsTuple(cpy_r_r1);
    CPy_DECREF_NO_IMM(cpy_r_r1);
    if (unlikely(cpy_r_r3 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 178, CPyStatic_globals);
        goto CPyL27;
    }
    cpy_r_r4 = PyObject_CallObject(cpy_r_r2, cpy_r_r3);
    CPy_DECREF(cpy_r_r3);
    if (unlikely(cpy_r_r4 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 178, CPyStatic_globals);
        goto CPyL27;
    }
    if (likely(Py_TYPE(cpy_r_r4) == CPyType_MetaDataclass))
        cpy_r_r5 = cpy_r_r4;
    else {
        CPy_TypeErrorTraceback("mypyc_variant.py", "run_mypyc_dataclass", 178, CPyStatic_globals, "mypyc_variant.MetaDataclass", cpy_r_r4);
        goto CPyL27;
    }
    cpy_r_r6 = CPyDef_small_ops(cpy_r_inp, cpy_r_mode);
    CPy_DECREF(cpy_r_mode);
    if (unlikely(cpy_r_r6 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 179, CPyStatic_globals);
        goto CPyL28;
    }
    cpy_r_r7 = CPyDef__read_scalars(cpy_r_r5);
    if (unlikely(cpy_r_r7 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 180, CPyStatic_globals);
        goto CPyL29;
    }
    cpy_r_r8 = CPyDef__read_containers(cpy_r_r5);
    CPy_DECREF_NO_IMM(cpy_r_r5);
    if (unlikely(cpy_r_r8 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 180, CPyStatic_globals);
        goto CPyL30;
    }
    cpy_r_r9 = CPyTagged_Add(cpy_r_r7, cpy_r_r8);
    CPyTagged_DECREF(cpy_r_r7);
    CPyTagged_DECREF(cpy_r_r8);
    cpy_r_acc = cpy_r_r9;
    cpy_r_r10 = CPySequenceTuple_GetItem(cpy_r_r6, 0);
    if (unlikely(cpy_r_r10 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 181, CPyStatic_globals);
        goto CPyL31;
    }
    cpy_r_r11 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r12[1] = {cpy_r_r10};
    cpy_r_r13 = (PyObject **)&cpy_r_r12;
    cpy_r_r14 = PyObject_Vectorcall(cpy_r_r11, cpy_r_r13, 1, 0);
    if (unlikely(cpy_r_r14 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 181, CPyStatic_globals);
        goto CPyL32;
    }
    CPy_DECREF(cpy_r_r10);
    if (likely(PyLong_Check(cpy_r_r14)))
        cpy_r_r15 = CPyTagged_FromObject(cpy_r_r14);
    else {
        CPy_TypeError("int", cpy_r_r14); cpy_r_r15 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r14);
    if (unlikely(cpy_r_r15 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 181, CPyStatic_globals);
        goto CPyL31;
    }
    cpy_r_r16 = CPySequenceTuple_GetItem(cpy_r_r6, 2);
    if (unlikely(cpy_r_r16 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 181, CPyStatic_globals);
        goto CPyL33;
    }
    cpy_r_r17 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r18[1] = {cpy_r_r16};
    cpy_r_r19 = (PyObject **)&cpy_r_r18;
    cpy_r_r20 = PyObject_Vectorcall(cpy_r_r17, cpy_r_r19, 1, 0);
    if (unlikely(cpy_r_r20 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 181, CPyStatic_globals);
        goto CPyL34;
    }
    CPy_DECREF(cpy_r_r16);
    if (likely(PyLong_Check(cpy_r_r20)))
        cpy_r_r21 = CPyTagged_FromObject(cpy_r_r20);
    else {
        CPy_TypeError("int", cpy_r_r20); cpy_r_r21 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r20);
    if (unlikely(cpy_r_r21 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 181, CPyStatic_globals);
        goto CPyL33;
    }
    cpy_r_r22 = CPyTagged_Add(cpy_r_r15, cpy_r_r21);
    CPyTagged_DECREF(cpy_r_r15);
    CPyTagged_DECREF(cpy_r_r21);
    cpy_r_r23 = CPySequenceTuple_GetItem(cpy_r_r6, 6);
    if (unlikely(cpy_r_r23 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 181, CPyStatic_globals);
        goto CPyL35;
    }
    cpy_r_r24 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r25[1] = {cpy_r_r23};
    cpy_r_r26 = (PyObject **)&cpy_r_r25;
    cpy_r_r27 = PyObject_Vectorcall(cpy_r_r24, cpy_r_r26, 1, 0);
    if (unlikely(cpy_r_r27 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 181, CPyStatic_globals);
        goto CPyL36;
    }
    CPy_DECREF(cpy_r_r23);
    if (likely(PyLong_Check(cpy_r_r27)))
        cpy_r_r28 = CPyTagged_FromObject(cpy_r_r27);
    else {
        CPy_TypeError("int", cpy_r_r27); cpy_r_r28 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r27);
    if (unlikely(cpy_r_r28 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 181, CPyStatic_globals);
        goto CPyL35;
    }
    cpy_r_r29 = CPyTagged_Add(cpy_r_r22, cpy_r_r28);
    CPyTagged_DECREF(cpy_r_r22);
    CPyTagged_DECREF(cpy_r_r28);
    cpy_r_r30 = CPySequenceTuple_GetItem(cpy_r_r6, 8);
    if (unlikely(cpy_r_r30 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 181, CPyStatic_globals);
        goto CPyL37;
    }
    cpy_r_r31 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r32[1] = {cpy_r_r30};
    cpy_r_r33 = (PyObject **)&cpy_r_r32;
    cpy_r_r34 = PyObject_Vectorcall(cpy_r_r31, cpy_r_r33, 1, 0);
    if (unlikely(cpy_r_r34 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 181, CPyStatic_globals);
        goto CPyL38;
    }
    CPy_DECREF(cpy_r_r30);
    if (likely(PyLong_Check(cpy_r_r34)))
        cpy_r_r35 = CPyTagged_FromObject(cpy_r_r34);
    else {
        CPy_TypeError("int", cpy_r_r34); cpy_r_r35 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r34);
    if (unlikely(cpy_r_r35 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 181, CPyStatic_globals);
        goto CPyL37;
    }
    cpy_r_r36 = CPyTagged_Add(cpy_r_r29, cpy_r_r35);
    CPyTagged_DECREF(cpy_r_r29);
    CPyTagged_DECREF(cpy_r_r35);
    cpy_r_r37 = CPySequenceTuple_GetItem(cpy_r_r6, 14);
    if (unlikely(cpy_r_r37 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 181, CPyStatic_globals);
        goto CPyL39;
    }
    cpy_r_r38 = (PyObject *)&PyLong_Type;
    PyObject *cpy_r_r39[1] = {cpy_r_r37};
    cpy_r_r40 = (PyObject **)&cpy_r_r39;
    cpy_r_r41 = PyObject_Vectorcall(cpy_r_r38, cpy_r_r40, 1, 0);
    if (unlikely(cpy_r_r41 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 181, CPyStatic_globals);
        goto CPyL40;
    }
    CPy_DECREF(cpy_r_r37);
    if (likely(PyLong_Check(cpy_r_r41)))
        cpy_r_r42 = CPyTagged_FromObject(cpy_r_r41);
    else {
        CPy_TypeError("int", cpy_r_r41); cpy_r_r42 = CPY_INT_TAG;
    }
    CPy_DECREF(cpy_r_r41);
    if (unlikely(cpy_r_r42 == CPY_INT_TAG)) {
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 181, CPyStatic_globals);
        goto CPyL39;
    }
    cpy_r_r43 = CPyTagged_Add(cpy_r_r36, cpy_r_r42);
    CPyTagged_DECREF(cpy_r_r36);
    CPyTagged_DECREF(cpy_r_r42);
    cpy_r_r44 = (CPyPtr)((CPyPtr)cpy_r_r6 + offsetof(PyVarObject, ob_size));
    cpy_r_r45 = *(int64_t *)cpy_r_r44;
    CPy_DECREF(cpy_r_r6);
    cpy_r_r46 = cpy_r_r45 << 1;
    cpy_r_r47 = CPyTagged_Add(cpy_r_r43, cpy_r_r46);
    CPyTagged_DECREF(cpy_r_r43);
    cpy_r_r48 = CPyTagged_Add(cpy_r_acc, cpy_r_r47);
    CPyTagged_DECREF(cpy_r_acc);
    CPyTagged_DECREF(cpy_r_r47);
    cpy_r_acc = cpy_r_r48;
    return cpy_r_acc;
CPyL25: ;
    cpy_r_r49 = CPY_INT_TAG;
    return cpy_r_r49;
CPyL26: ;
    CPy_INCREF(cpy_r_mode);
    goto CPyL2;
CPyL27: ;
    CPy_DecRef(cpy_r_mode);
    goto CPyL25;
CPyL28: ;
    CPy_DecRef(cpy_r_r5);
    goto CPyL25;
CPyL29: ;
    CPy_DecRef(cpy_r_r5);
    CPy_DecRef(cpy_r_r6);
    goto CPyL25;
CPyL30: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_r7);
    goto CPyL25;
CPyL31: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    goto CPyL25;
CPyL32: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    CPy_DecRef(cpy_r_r10);
    goto CPyL25;
CPyL33: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    CPyTagged_DecRef(cpy_r_r15);
    goto CPyL25;
CPyL34: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    CPyTagged_DecRef(cpy_r_r15);
    CPy_DecRef(cpy_r_r16);
    goto CPyL25;
CPyL35: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    CPyTagged_DecRef(cpy_r_r22);
    goto CPyL25;
CPyL36: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    CPyTagged_DecRef(cpy_r_r22);
    CPy_DecRef(cpy_r_r23);
    goto CPyL25;
CPyL37: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    CPyTagged_DecRef(cpy_r_r29);
    goto CPyL25;
CPyL38: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    CPyTagged_DecRef(cpy_r_r29);
    CPy_DecRef(cpy_r_r30);
    goto CPyL25;
CPyL39: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    CPyTagged_DecRef(cpy_r_r36);
    goto CPyL25;
CPyL40: ;
    CPy_DecRef(cpy_r_r6);
    CPyTagged_DecRef(cpy_r_acc);
    CPyTagged_DecRef(cpy_r_r36);
    CPy_DecRef(cpy_r_r37);
    goto CPyL25;
}
    
    PyObject *CPyPy_run_mypyc_dataclass(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames) {
        static const char * const kwlist[] = {"inp", "mode", 0};
        static CPyArg_Parser parser = {"O|O:run_mypyc_dataclass", kwlist, 0};
        PyObject *obj_inp;
        PyObject *obj_mode = NULL;
        if (!CPyArg_ParseStackAndKeywordsSimple(args, nargs, kwnames, &parser, &obj_inp, &obj_mode)) {
            return NULL;
        }
        PyObject *arg_inp;
        if (likely(PyDict_Check(obj_inp)))
            arg_inp = obj_inp;
        else {
            CPy_TypeError("dict", obj_inp); 
            goto fail;
        }
        PyObject *arg_mode;
        if (obj_mode == NULL) {
            arg_mode = NULL;
        } else if (likely(PyUnicode_Check(obj_mode)))
            arg_mode = obj_mode;
        else {
            CPy_TypeError("str", obj_mode); 
            goto fail;
        }
        CPyTagged retval = CPyDef_run_mypyc_dataclass(arg_inp, arg_mode);
        if (retval == CPY_INT_TAG) {
            return NULL;
        }
        PyObject *retbox = CPyTagged_StealAsObject(retval);
        return retbox;
fail: ;
        CPy_AddTraceback("mypyc_variant.py", "run_mypyc_dataclass", 177, CPyStatic_globals);
        return NULL;
    }
    
char CPyDef___top_level__(void) {
    PyObject *cpy_r_r0;
    PyObject *cpy_r_r1;
    char cpy_r_r2;
    PyObject *cpy_r_r3;
    PyObject *cpy_r_r4;
    PyObject *cpy_r_r5;
    PyObject *cpy_r_r6;
    PyObject *cpy_r_r7;
    PyObject *cpy_r_r8;
    PyObject *cpy_r_r9;
    PyObject *cpy_r_r10;
    PyObject *cpy_r_r11;
    PyObject *cpy_r_r12;
    PyObject *cpy_r_r13;
    PyObject *cpy_r_r14;
    PyObject *cpy_r_r15;
    PyObject *cpy_r_r16;
    PyObject **cpy_r_r17;
    PyObject **cpy_r_r18;
    void *cpy_r_r20;
    void *cpy_r_r22;
    PyObject *cpy_r_r23;
    PyObject *cpy_r_r24;
    PyObject *cpy_r_r25;
    PyObject *cpy_r_r26;
    char cpy_r_r27;
    PyObject *cpy_r_r28;
    PyObject *cpy_r_r29;
    PyObject *cpy_r_r30;
    PyObject *cpy_r_r31;
    PyObject *cpy_r_r32;
    PyObject *cpy_r_r33;
    PyObject *cpy_r_r34;
    PyObject *cpy_r_r35;
    PyObject *cpy_r_r36;
    PyObject *cpy_r_r37;
    PyObject *cpy_r_r38;
    PyObject *cpy_r_r39;
    PyObject *cpy_r_r40;
    PyObject *cpy_r_r41;
    PyObject *cpy_r_r42;
    PyObject *cpy_r_r43;
    PyObject *cpy_r_r44;
    PyObject *cpy_r_r45;
    PyObject *cpy_r_r46;
    PyObject *cpy_r_r47;
    PyObject *cpy_r_r48;
    PyObject *cpy_r_r49;
    PyObject *cpy_r_r50;
    PyObject *cpy_r_r51;
    PyObject *cpy_r_r52;
    int32_t cpy_r_r53;
    char cpy_r_r54;
    PyObject *cpy_r_r55;
    PyObject *cpy_r_r56;
    PyObject *cpy_r_r57;
    PyObject *cpy_r_r58;
    char cpy_r_r59;
    char cpy_r_r60;
    PyObject *cpy_r_r61;
    PyObject *cpy_r_r62;
    PyObject *cpy_r_r63;
    PyObject *cpy_r_r64;
    PyObject *cpy_r_r65;
    PyObject *cpy_r_r66;
    PyObject *cpy_r_r67;
    PyObject *cpy_r_r68;
    PyObject *cpy_r_r69;
    PyObject *cpy_r_r70;
    PyObject *cpy_r_r71;
    PyObject *cpy_r_r72;
    PyObject *cpy_r_r73;
    PyObject *cpy_r_r74;
    PyObject *cpy_r_r75;
    PyObject *cpy_r_r76;
    PyObject *cpy_r_r77;
    PyObject *cpy_r_r78;
    PyObject *cpy_r_r79;
    PyObject *cpy_r_r80;
    PyObject *cpy_r_r81;
    PyObject *cpy_r_r82;
    PyObject *cpy_r_r83;
    PyObject *cpy_r_r84;
    int32_t cpy_r_r85;
    char cpy_r_r86;
    PyObject *cpy_r_r87;
    PyObject *cpy_r_r88;
    int32_t cpy_r_r89;
    char cpy_r_r90;
    char cpy_r_r91;
    PyObject *cpy_r_r92;
    PyObject *cpy_r_r93;
    PyObject *cpy_r_r94;
    PyObject *cpy_r_r95;
    char cpy_r_r96;
    char cpy_r_r97;
    PyObject *cpy_r_r98;
    PyObject *cpy_r_r99;
    PyObject *cpy_r_r100;
    PyObject *cpy_r_r101;
    PyObject *cpy_r_r102;
    PyObject *cpy_r_r103;
    PyObject *cpy_r_r104;
    PyObject *cpy_r_r105;
    PyObject *cpy_r_r106;
    PyObject *cpy_r_r107;
    PyObject *cpy_r_r108;
    PyObject *cpy_r_r109;
    PyObject *cpy_r_r110;
    PyObject *cpy_r_r111;
    PyObject *cpy_r_r112;
    PyObject *cpy_r_r113;
    PyObject *cpy_r_r114;
    PyObject *cpy_r_r115;
    PyObject *cpy_r_r116;
    PyObject *cpy_r_r117;
    PyObject *cpy_r_r118;
    PyObject *cpy_r_r119;
    PyObject *cpy_r_r120;
    PyObject *cpy_r_r121;
    int32_t cpy_r_r122;
    char cpy_r_r123;
    PyObject *cpy_r_r124;
    PyObject *cpy_r_r125;
    int32_t cpy_r_r126;
    char cpy_r_r127;
    PyObject *cpy_r_r128;
    tuple_T0 cpy_r_r129;
    PyObject *cpy_r_r130;
    PyObject *cpy_r_r131;
    PyObject *cpy_r_r132;
    PyObject *cpy_r_r133;
    int32_t cpy_r_r134;
    char cpy_r_r135;
    PyObject *cpy_r_r136;
    PyObject *cpy_r_r137;
    int32_t cpy_r_r138;
    char cpy_r_r139;
    PyObject *cpy_r_r140;
    PyObject *cpy_r_r141;
    int32_t cpy_r_r142;
    char cpy_r_r143;
    PyObject *cpy_r_r144;
    PyObject *cpy_r_r145;
    int32_t cpy_r_r146;
    char cpy_r_r147;
    PyObject *cpy_r_r148;
    PyObject *cpy_r_r149;
    int32_t cpy_r_r150;
    char cpy_r_r151;
    PyObject *cpy_r_r152;
    PyObject *cpy_r_r153;
    int32_t cpy_r_r154;
    char cpy_r_r155;
    PyObject *cpy_r_r156;
    PyObject *cpy_r_r157;
    int32_t cpy_r_r158;
    char cpy_r_r159;
    PyObject *cpy_r_r160;
    PyObject *cpy_r_r161;
    int32_t cpy_r_r162;
    char cpy_r_r163;
    PyObject *cpy_r_r164;
    PyObject *cpy_r_r165;
    int32_t cpy_r_r166;
    char cpy_r_r167;
    PyObject *cpy_r_r168;
    PyObject *cpy_r_r169;
    int32_t cpy_r_r170;
    char cpy_r_r171;
    PyObject *cpy_r_r172;
    PyObject *cpy_r_r173;
    int32_t cpy_r_r174;
    char cpy_r_r175;
    PyObject *cpy_r_r176;
    PyObject *cpy_r_r177;
    int32_t cpy_r_r178;
    char cpy_r_r179;
    PyObject *cpy_r_r180;
    PyObject *cpy_r_r181;
    int32_t cpy_r_r182;
    char cpy_r_r183;
    PyObject *cpy_r_r184;
    PyObject *cpy_r_r185;
    int32_t cpy_r_r186;
    char cpy_r_r187;
    PyObject *cpy_r_r188;
    PyObject *cpy_r_r189;
    int32_t cpy_r_r190;
    char cpy_r_r191;
    PyObject *cpy_r_r192;
    PyObject *cpy_r_r193;
    int32_t cpy_r_r194;
    char cpy_r_r195;
    PyObject *cpy_r_r196;
    PyObject *cpy_r_r197;
    int32_t cpy_r_r198;
    char cpy_r_r199;
    PyObject *cpy_r_r200;
    PyObject *cpy_r_r201;
    int32_t cpy_r_r202;
    char cpy_r_r203;
    PyObject *cpy_r_r204;
    PyObject *cpy_r_r205;
    int32_t cpy_r_r206;
    char cpy_r_r207;
    PyObject *cpy_r_r208;
    PyObject *cpy_r_r209;
    int32_t cpy_r_r210;
    char cpy_r_r211;
    PyObject *cpy_r_r212;
    PyObject *cpy_r_r213;
    int32_t cpy_r_r214;
    char cpy_r_r215;
    PyObject *cpy_r_r216;
    PyObject *cpy_r_r217;
    int32_t cpy_r_r218;
    char cpy_r_r219;
    char cpy_r_r220;
    PyObject *cpy_r_r221;
    int32_t cpy_r_r222;
    char cpy_r_r223;
    PyObject *cpy_r_r224;
    PyObject *cpy_r_r225;
    int32_t cpy_r_r226;
    char cpy_r_r227;
    PyObject *cpy_r_r228;
    PyObject *cpy_r_r229;
    int32_t cpy_r_r230;
    char cpy_r_r231;
    PyObject *cpy_r_r232;
    PyObject *cpy_r_r233;
    PyObject *cpy_r_r234;
    PyObject *cpy_r_r235;
    char cpy_r_r236;
    char cpy_r_r237;
    cpy_r_r0 = CPyModule_builtins;
    cpy_r_r1 = (PyObject *)&_Py_NoneStruct;
    cpy_r_r2 = cpy_r_r0 != cpy_r_r1;
    if (cpy_r_r2) goto CPyL3;
    cpy_r_r3 = CPyStatics[46]; /* 'builtins' */
    cpy_r_r4 = PyImport_Import(cpy_r_r3);
    if (unlikely(cpy_r_r4 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 1, CPyStatic_globals);
        goto CPyL53;
    }
    CPyModule_builtins = cpy_r_r4;
    CPy_INCREF(CPyModule_builtins);
    CPy_DECREF(cpy_r_r4);
CPyL3: ;
    cpy_r_r5 = CPyStatics[70]; /* ('annotations',) */
    cpy_r_r6 = CPyStatics[48]; /* '__future__' */
    cpy_r_r7 = CPyStatic_globals;
    cpy_r_r8 = CPyImport_ImportFromMany(cpy_r_r6, cpy_r_r5, cpy_r_r5, cpy_r_r7);
    if (unlikely(cpy_r_r8 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 11, CPyStatic_globals);
        goto CPyL53;
    }
    CPyModule___future__ = cpy_r_r8;
    CPy_INCREF(CPyModule___future__);
    CPy_DECREF(cpy_r_r8);
    cpy_r_r9 = CPyStatics[71]; /* ('dataclass',) */
    cpy_r_r10 = CPyStatics[50]; /* 'dataclasses' */
    cpy_r_r11 = CPyStatic_globals;
    cpy_r_r12 = CPyImport_ImportFromMany(cpy_r_r10, cpy_r_r9, cpy_r_r9, cpy_r_r11);
    if (unlikely(cpy_r_r12 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 13, CPyStatic_globals);
        goto CPyL53;
    }
    CPyModule_dataclasses = cpy_r_r12;
    CPy_INCREF(CPyModule_dataclasses);
    CPy_DECREF(cpy_r_r12);
    cpy_r_r13 = CPyStatics[72]; /* ('Any',) */
    cpy_r_r14 = CPyStatics[52]; /* 'typing' */
    cpy_r_r15 = CPyStatic_globals;
    cpy_r_r16 = CPyImport_ImportFromMany(cpy_r_r14, cpy_r_r13, cpy_r_r13, cpy_r_r15);
    if (unlikely(cpy_r_r16 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 14, CPyStatic_globals);
        goto CPyL53;
    }
    CPyModule_typing = cpy_r_r16;
    CPy_INCREF(CPyModule_typing);
    CPy_DECREF(cpy_r_r16);
    cpy_r_r17 = (PyObject **)&CPyModule_numpy;
    cpy_r_r18 = (PyObject **)&CPyModule_torch;
    PyObject **cpy_r_r19[2] = {cpy_r_r17, cpy_r_r18};
    cpy_r_r20 = (void *)&cpy_r_r19;
    int64_t cpy_r_r21[2] = {16, 17};
    cpy_r_r22 = (void *)&cpy_r_r21;
    cpy_r_r23 = CPyStatics[75]; /* (('numpy', 'numpy', 'np'), ('torch', 'torch', 'torch')) */
    cpy_r_r24 = CPyStatic_globals;
    cpy_r_r25 = CPyStatics[54]; /* 'mypyc_variant.py' */
    cpy_r_r26 = CPyStatics[55]; /* '<module>' */
    cpy_r_r27 = CPyImport_ImportMany(cpy_r_r23, cpy_r_r20, cpy_r_r24, cpy_r_r25, cpy_r_r26, cpy_r_r22);
    if (!cpy_r_r27) goto CPyL53;
    cpy_r_r28 = CPyStatics[26]; /* 'num_reqs' */
    cpy_r_r29 = CPyStatics[27]; /* 'num_actual_tokens' */
    cpy_r_r30 = CPyStatics[28]; /* 'max_query_len' */
    cpy_r_r31 = CPyStatics[39]; /* 'query_start_loc' */
    cpy_r_r32 = CPyStatics[45]; /* 'seq_lens' */
    cpy_r_r33 = CPyStatics[42]; /* 'block_table' */
    cpy_r_r34 = CPyStatics[40]; /* 'slot_mapping' */
    cpy_r_r35 = CPyStatics[36]; /* 'spec_decode' */
    cpy_r_r36 = CPyStatics[29]; /* 'num_decode_tokens' */
    cpy_r_r37 = CPyStatics[30]; /* 'num_prefill_tokens' */
    cpy_r_r38 = CPyStatics[44]; /* 'num_computed_tokens' */
    cpy_r_r39 = CPyStatics[6]; /* 'seq_lens_np' */
    cpy_r_r40 = CPyStatics[31]; /* 'max_seq_len' */
    cpy_r_r41 = CPyStatics[38]; /* 'kv_cache_dtype' */
    cpy_r_r42 = CPyStatics[32]; /* 'block_size' */
    cpy_r_r43 = CPyStatics[33]; /* 'num_kv_heads' */
    cpy_r_r44 = CPyStatics[34]; /* 'head_size' */
    cpy_r_r45 = CPyStatics[37]; /* 'causal' */
    cpy_r_r46 = CPyStatics[35]; /* 'sliding_window' */
    cpy_r_r47 = CPyStatics[41]; /* 'token_ids' */
    cpy_r_r48 = CPyStatics[43]; /* 'positions' */
    cpy_r_r49 = CPyStatics[19]; /* 'workspace' */
    CPy_INCREF(cpy_r_r28);
    CPy_INCREF(cpy_r_r29);
    CPy_INCREF(cpy_r_r30);
    CPy_INCREF(cpy_r_r31);
    CPy_INCREF(cpy_r_r32);
    CPy_INCREF(cpy_r_r33);
    CPy_INCREF(cpy_r_r34);
    CPy_INCREF(cpy_r_r35);
    CPy_INCREF(cpy_r_r36);
    CPy_INCREF(cpy_r_r37);
    CPy_INCREF(cpy_r_r38);
    CPy_INCREF(cpy_r_r39);
    CPy_INCREF(cpy_r_r40);
    CPy_INCREF(cpy_r_r41);
    CPy_INCREF(cpy_r_r42);
    CPy_INCREF(cpy_r_r43);
    CPy_INCREF(cpy_r_r44);
    CPy_INCREF(cpy_r_r45);
    CPy_INCREF(cpy_r_r46);
    CPy_INCREF(cpy_r_r47);
    CPy_INCREF(cpy_r_r48);
    CPy_INCREF(cpy_r_r49);
    cpy_r_r50 = CPyList_Build(22, cpy_r_r28, cpy_r_r29, cpy_r_r30, cpy_r_r31, cpy_r_r32, cpy_r_r33, cpy_r_r34, cpy_r_r35, cpy_r_r36, cpy_r_r37, cpy_r_r38, cpy_r_r39, cpy_r_r40, cpy_r_r41, cpy_r_r42, cpy_r_r43, cpy_r_r44, cpy_r_r45, cpy_r_r46, cpy_r_r47, cpy_r_r48, cpy_r_r49);
    if (unlikely(cpy_r_r50 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 19, CPyStatic_globals);
        goto CPyL53;
    }
    cpy_r_r51 = CPyStatic_globals;
    cpy_r_r52 = CPyStatics[3]; /* 'FIELDS' */
    cpy_r_r53 = CPyDict_SetItem(cpy_r_r51, cpy_r_r52, cpy_r_r50);
    CPy_DECREF_NO_IMM(cpy_r_r50);
    cpy_r_r54 = cpy_r_r53 >= 0;
    if (unlikely(!cpy_r_r54)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 19, CPyStatic_globals);
        goto CPyL53;
    }
    cpy_r_r55 = NULL;
    cpy_r_r56 = CPyStatics[56]; /* 'mypyc_variant' */
    cpy_r_r57 = (PyObject *)CPyType_MetaNative_template;
    cpy_r_r58 = CPyType_FromTemplate(cpy_r_r57, cpy_r_r55, cpy_r_r56);
    if (unlikely(cpy_r_r58 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 28, CPyStatic_globals);
        goto CPyL53;
    }
    cpy_r_r59 = CPyDef_MetaNative_trait_vtable_setup();
    if (unlikely(cpy_r_r59 == 2)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 28, CPyStatic_globals);
        goto CPyL54;
    }
    cpy_r_r60 = CPyDef_MetaNative_coroutine_setup(cpy_r_r58);
    if (unlikely(cpy_r_r60 == 2)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 28, CPyStatic_globals);
        goto CPyL54;
    }
    cpy_r_r61 = CPyStatics[57]; /* '__mypyc_attrs__' */
    cpy_r_r62 = CPyStatics[26]; /* 'num_reqs' */
    cpy_r_r63 = CPyStatics[27]; /* 'num_actual_tokens' */
    cpy_r_r64 = CPyStatics[28]; /* 'max_query_len' */
    cpy_r_r65 = CPyStatics[39]; /* 'query_start_loc' */
    cpy_r_r66 = CPyStatics[45]; /* 'seq_lens' */
    cpy_r_r67 = CPyStatics[42]; /* 'block_table' */
    cpy_r_r68 = CPyStatics[40]; /* 'slot_mapping' */
    cpy_r_r69 = CPyStatics[36]; /* 'spec_decode' */
    cpy_r_r70 = CPyStatics[29]; /* 'num_decode_tokens' */
    cpy_r_r71 = CPyStatics[30]; /* 'num_prefill_tokens' */
    cpy_r_r72 = CPyStatics[44]; /* 'num_computed_tokens' */
    cpy_r_r73 = CPyStatics[6]; /* 'seq_lens_np' */
    cpy_r_r74 = CPyStatics[31]; /* 'max_seq_len' */
    cpy_r_r75 = CPyStatics[38]; /* 'kv_cache_dtype' */
    cpy_r_r76 = CPyStatics[32]; /* 'block_size' */
    cpy_r_r77 = CPyStatics[33]; /* 'num_kv_heads' */
    cpy_r_r78 = CPyStatics[34]; /* 'head_size' */
    cpy_r_r79 = CPyStatics[37]; /* 'causal' */
    cpy_r_r80 = CPyStatics[35]; /* 'sliding_window' */
    cpy_r_r81 = CPyStatics[41]; /* 'token_ids' */
    cpy_r_r82 = CPyStatics[43]; /* 'positions' */
    cpy_r_r83 = CPyStatics[19]; /* 'workspace' */
    cpy_r_r84 = PyTuple_Pack(22, cpy_r_r62, cpy_r_r63, cpy_r_r64, cpy_r_r65, cpy_r_r66, cpy_r_r67, cpy_r_r68, cpy_r_r69, cpy_r_r70, cpy_r_r71, cpy_r_r72, cpy_r_r73, cpy_r_r74, cpy_r_r75, cpy_r_r76, cpy_r_r77, cpy_r_r78, cpy_r_r79, cpy_r_r80, cpy_r_r81, cpy_r_r82, cpy_r_r83);
    if (unlikely(cpy_r_r84 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 28, CPyStatic_globals);
        goto CPyL54;
    }
    cpy_r_r85 = PyObject_SetAttr(cpy_r_r58, cpy_r_r61, cpy_r_r84);
    CPy_DECREF(cpy_r_r84);
    cpy_r_r86 = cpy_r_r85 >= 0;
    if (unlikely(!cpy_r_r86)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 28, CPyStatic_globals);
        goto CPyL54;
    }
    CPyType_MetaNative = (PyTypeObject *)cpy_r_r58;
    CPy_INCREF(CPyType_MetaNative);
    cpy_r_r87 = CPyStatic_globals;
    cpy_r_r88 = CPyStatics[58]; /* 'MetaNative' */
    cpy_r_r89 = PyDict_SetItem(cpy_r_r87, cpy_r_r88, cpy_r_r58);
    cpy_r_r90 = cpy_r_r89 >= 0;
    if (unlikely(!cpy_r_r90)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 28, CPyStatic_globals);
        goto CPyL54;
    }
    cpy_r_r91 = CPy_InitSubclass(cpy_r_r58);
    CPy_DECREF(cpy_r_r58);
    if (unlikely(!cpy_r_r91)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 28, CPyStatic_globals);
        goto CPyL53;
    }
    cpy_r_r92 = NULL;
    cpy_r_r93 = CPyStatics[56]; /* 'mypyc_variant' */
    cpy_r_r94 = (PyObject *)CPyType_MetaDataclass_template;
    cpy_r_r95 = CPyType_FromTemplate(cpy_r_r94, cpy_r_r92, cpy_r_r93);
    if (unlikely(cpy_r_r95 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 84, CPyStatic_globals);
        goto CPyL53;
    }
    cpy_r_r96 = CPyDef_MetaDataclass_trait_vtable_setup();
    if (unlikely(cpy_r_r96 == 2)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 84, CPyStatic_globals);
        goto CPyL55;
    }
    cpy_r_r97 = CPyDef_MetaDataclass_coroutine_setup(cpy_r_r95);
    if (unlikely(cpy_r_r97 == 2)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 84, CPyStatic_globals);
        goto CPyL55;
    }
    cpy_r_r98 = CPyStatics[57]; /* '__mypyc_attrs__' */
    cpy_r_r99 = CPyStatics[26]; /* 'num_reqs' */
    cpy_r_r100 = CPyStatics[27]; /* 'num_actual_tokens' */
    cpy_r_r101 = CPyStatics[28]; /* 'max_query_len' */
    cpy_r_r102 = CPyStatics[39]; /* 'query_start_loc' */
    cpy_r_r103 = CPyStatics[45]; /* 'seq_lens' */
    cpy_r_r104 = CPyStatics[42]; /* 'block_table' */
    cpy_r_r105 = CPyStatics[40]; /* 'slot_mapping' */
    cpy_r_r106 = CPyStatics[36]; /* 'spec_decode' */
    cpy_r_r107 = CPyStatics[29]; /* 'num_decode_tokens' */
    cpy_r_r108 = CPyStatics[30]; /* 'num_prefill_tokens' */
    cpy_r_r109 = CPyStatics[44]; /* 'num_computed_tokens' */
    cpy_r_r110 = CPyStatics[6]; /* 'seq_lens_np' */
    cpy_r_r111 = CPyStatics[31]; /* 'max_seq_len' */
    cpy_r_r112 = CPyStatics[38]; /* 'kv_cache_dtype' */
    cpy_r_r113 = CPyStatics[32]; /* 'block_size' */
    cpy_r_r114 = CPyStatics[33]; /* 'num_kv_heads' */
    cpy_r_r115 = CPyStatics[34]; /* 'head_size' */
    cpy_r_r116 = CPyStatics[37]; /* 'causal' */
    cpy_r_r117 = CPyStatics[35]; /* 'sliding_window' */
    cpy_r_r118 = CPyStatics[41]; /* 'token_ids' */
    cpy_r_r119 = CPyStatics[43]; /* 'positions' */
    cpy_r_r120 = CPyStatics[19]; /* 'workspace' */
    cpy_r_r121 = PyTuple_Pack(22, cpy_r_r99, cpy_r_r100, cpy_r_r101, cpy_r_r102, cpy_r_r103, cpy_r_r104, cpy_r_r105, cpy_r_r106, cpy_r_r107, cpy_r_r108, cpy_r_r109, cpy_r_r110, cpy_r_r111, cpy_r_r112, cpy_r_r113, cpy_r_r114, cpy_r_r115, cpy_r_r116, cpy_r_r117, cpy_r_r118, cpy_r_r119, cpy_r_r120);
    if (unlikely(cpy_r_r121 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 84, CPyStatic_globals);
        goto CPyL55;
    }
    cpy_r_r122 = PyObject_SetAttr(cpy_r_r95, cpy_r_r98, cpy_r_r121);
    CPy_DECREF(cpy_r_r121);
    cpy_r_r123 = cpy_r_r122 >= 0;
    if (unlikely(!cpy_r_r123)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 84, CPyStatic_globals);
        goto CPyL55;
    }
    CPyType_MetaDataclass = (PyTypeObject *)cpy_r_r95;
    CPy_INCREF(CPyType_MetaDataclass);
    cpy_r_r124 = CPyStatic_globals;
    cpy_r_r125 = CPyStatics[59]; /* 'MetaDataclass' */
    cpy_r_r126 = PyDict_SetItem(cpy_r_r124, cpy_r_r125, cpy_r_r95);
    cpy_r_r127 = cpy_r_r126 >= 0;
    if (unlikely(!cpy_r_r127)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 84, CPyStatic_globals);
        goto CPyL55;
    }
    cpy_r_r128 = PyDict_New();
    if (unlikely(cpy_r_r128 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 84, CPyStatic_globals);
        goto CPyL55;
    }
    cpy_r_r129.empty_struct_error_flag = 0;
    cpy_r_r130 = PyDict_New();
    if (unlikely(cpy_r_r130 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 84, CPyStatic_globals);
        goto CPyL56;
    }
    cpy_r_r131 = (PyObject *)&PyType_Type;
    cpy_r_r132 = (PyObject *)&PyLong_Type;
    cpy_r_r133 = CPyStatics[26]; /* 'num_reqs' */
    cpy_r_r134 = PyDict_SetItem(cpy_r_r130, cpy_r_r133, cpy_r_r132);
    cpy_r_r135 = cpy_r_r134 >= 0;
    if (unlikely(!cpy_r_r135)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 85, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r136 = (PyObject *)&PyLong_Type;
    cpy_r_r137 = CPyStatics[27]; /* 'num_actual_tokens' */
    cpy_r_r138 = PyDict_SetItem(cpy_r_r130, cpy_r_r137, cpy_r_r136);
    cpy_r_r139 = cpy_r_r138 >= 0;
    if (unlikely(!cpy_r_r139)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 86, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r140 = (PyObject *)&PyLong_Type;
    cpy_r_r141 = CPyStatics[28]; /* 'max_query_len' */
    cpy_r_r142 = PyDict_SetItem(cpy_r_r130, cpy_r_r141, cpy_r_r140);
    cpy_r_r143 = cpy_r_r142 >= 0;
    if (unlikely(!cpy_r_r143)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 87, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r144 = (PyObject *)&PyType_Type;
    cpy_r_r145 = CPyStatics[39]; /* 'query_start_loc' */
    cpy_r_r146 = PyDict_SetItem(cpy_r_r130, cpy_r_r145, cpy_r_r144);
    cpy_r_r147 = cpy_r_r146 >= 0;
    if (unlikely(!cpy_r_r147)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 88, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r148 = (PyObject *)&PyType_Type;
    cpy_r_r149 = CPyStatics[45]; /* 'seq_lens' */
    cpy_r_r150 = PyDict_SetItem(cpy_r_r130, cpy_r_r149, cpy_r_r148);
    cpy_r_r151 = cpy_r_r150 >= 0;
    if (unlikely(!cpy_r_r151)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 89, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r152 = (PyObject *)&PyType_Type;
    cpy_r_r153 = CPyStatics[42]; /* 'block_table' */
    cpy_r_r154 = PyDict_SetItem(cpy_r_r130, cpy_r_r153, cpy_r_r152);
    cpy_r_r155 = cpy_r_r154 >= 0;
    if (unlikely(!cpy_r_r155)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 90, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r156 = (PyObject *)&PyType_Type;
    cpy_r_r157 = CPyStatics[40]; /* 'slot_mapping' */
    cpy_r_r158 = PyDict_SetItem(cpy_r_r130, cpy_r_r157, cpy_r_r156);
    cpy_r_r159 = cpy_r_r158 >= 0;
    if (unlikely(!cpy_r_r159)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 91, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r160 = (PyObject *)&PyBool_Type;
    cpy_r_r161 = CPyStatics[36]; /* 'spec_decode' */
    cpy_r_r162 = PyDict_SetItem(cpy_r_r130, cpy_r_r161, cpy_r_r160);
    cpy_r_r163 = cpy_r_r162 >= 0;
    if (unlikely(!cpy_r_r163)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 92, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r164 = (PyObject *)&PyLong_Type;
    cpy_r_r165 = CPyStatics[29]; /* 'num_decode_tokens' */
    cpy_r_r166 = PyDict_SetItem(cpy_r_r130, cpy_r_r165, cpy_r_r164);
    cpy_r_r167 = cpy_r_r166 >= 0;
    if (unlikely(!cpy_r_r167)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 93, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r168 = (PyObject *)&PyLong_Type;
    cpy_r_r169 = CPyStatics[30]; /* 'num_prefill_tokens' */
    cpy_r_r170 = PyDict_SetItem(cpy_r_r130, cpy_r_r169, cpy_r_r168);
    cpy_r_r171 = cpy_r_r170 >= 0;
    if (unlikely(!cpy_r_r171)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 94, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r172 = (PyObject *)&PyType_Type;
    cpy_r_r173 = CPyStatics[44]; /* 'num_computed_tokens' */
    cpy_r_r174 = PyDict_SetItem(cpy_r_r130, cpy_r_r173, cpy_r_r172);
    cpy_r_r175 = cpy_r_r174 >= 0;
    if (unlikely(!cpy_r_r175)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 95, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r176 = (PyObject *)&PyType_Type;
    cpy_r_r177 = CPyStatics[6]; /* 'seq_lens_np' */
    cpy_r_r178 = PyDict_SetItem(cpy_r_r130, cpy_r_r177, cpy_r_r176);
    cpy_r_r179 = cpy_r_r178 >= 0;
    if (unlikely(!cpy_r_r179)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 96, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r180 = (PyObject *)&PyLong_Type;
    cpy_r_r181 = CPyStatics[31]; /* 'max_seq_len' */
    cpy_r_r182 = PyDict_SetItem(cpy_r_r130, cpy_r_r181, cpy_r_r180);
    cpy_r_r183 = cpy_r_r182 >= 0;
    if (unlikely(!cpy_r_r183)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 97, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r184 = (PyObject *)&PyUnicode_Type;
    cpy_r_r185 = CPyStatics[38]; /* 'kv_cache_dtype' */
    cpy_r_r186 = PyDict_SetItem(cpy_r_r130, cpy_r_r185, cpy_r_r184);
    cpy_r_r187 = cpy_r_r186 >= 0;
    if (unlikely(!cpy_r_r187)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 98, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r188 = (PyObject *)&PyLong_Type;
    cpy_r_r189 = CPyStatics[32]; /* 'block_size' */
    cpy_r_r190 = PyDict_SetItem(cpy_r_r130, cpy_r_r189, cpy_r_r188);
    cpy_r_r191 = cpy_r_r190 >= 0;
    if (unlikely(!cpy_r_r191)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 99, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r192 = (PyObject *)&PyLong_Type;
    cpy_r_r193 = CPyStatics[33]; /* 'num_kv_heads' */
    cpy_r_r194 = PyDict_SetItem(cpy_r_r130, cpy_r_r193, cpy_r_r192);
    cpy_r_r195 = cpy_r_r194 >= 0;
    if (unlikely(!cpy_r_r195)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 100, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r196 = (PyObject *)&PyLong_Type;
    cpy_r_r197 = CPyStatics[34]; /* 'head_size' */
    cpy_r_r198 = PyDict_SetItem(cpy_r_r130, cpy_r_r197, cpy_r_r196);
    cpy_r_r199 = cpy_r_r198 >= 0;
    if (unlikely(!cpy_r_r199)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 101, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r200 = (PyObject *)&PyBool_Type;
    cpy_r_r201 = CPyStatics[37]; /* 'causal' */
    cpy_r_r202 = PyDict_SetItem(cpy_r_r130, cpy_r_r201, cpy_r_r200);
    cpy_r_r203 = cpy_r_r202 >= 0;
    if (unlikely(!cpy_r_r203)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 102, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r204 = (PyObject *)&PyLong_Type;
    cpy_r_r205 = CPyStatics[35]; /* 'sliding_window' */
    cpy_r_r206 = PyDict_SetItem(cpy_r_r130, cpy_r_r205, cpy_r_r204);
    cpy_r_r207 = cpy_r_r206 >= 0;
    if (unlikely(!cpy_r_r207)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 103, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r208 = (PyObject *)&PyType_Type;
    cpy_r_r209 = CPyStatics[41]; /* 'token_ids' */
    cpy_r_r210 = PyDict_SetItem(cpy_r_r130, cpy_r_r209, cpy_r_r208);
    cpy_r_r211 = cpy_r_r210 >= 0;
    if (unlikely(!cpy_r_r211)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 104, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r212 = (PyObject *)&PyType_Type;
    cpy_r_r213 = CPyStatics[43]; /* 'positions' */
    cpy_r_r214 = PyDict_SetItem(cpy_r_r130, cpy_r_r213, cpy_r_r212);
    cpy_r_r215 = cpy_r_r214 >= 0;
    if (unlikely(!cpy_r_r215)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 105, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r216 = (PyObject *)&PyType_Type;
    cpy_r_r217 = CPyStatics[19]; /* 'workspace' */
    cpy_r_r218 = PyDict_SetItem(cpy_r_r130, cpy_r_r217, cpy_r_r216);
    cpy_r_r219 = cpy_r_r218 >= 0;
    if (unlikely(!cpy_r_r219)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 106, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r220 = CPy_InitSubclass(cpy_r_r95);
    if (unlikely(!cpy_r_r220)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 84, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r221 = CPyStatics[60]; /* '__annotations__' */
    cpy_r_r222 = CPyDict_SetItem(cpy_r_r128, cpy_r_r221, cpy_r_r130);
    cpy_r_r223 = cpy_r_r222 >= 0;
    if (unlikely(!cpy_r_r223)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 84, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r224 = CPyStatics[61]; /* 'mypyc filler docstring' */
    cpy_r_r225 = CPyStatics[62]; /* '__doc__' */
    cpy_r_r226 = CPyDict_SetItem(cpy_r_r128, cpy_r_r225, cpy_r_r224);
    cpy_r_r227 = cpy_r_r226 >= 0;
    if (unlikely(!cpy_r_r227)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 84, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r228 = CPyStatics[56]; /* 'mypyc_variant' */
    cpy_r_r229 = CPyStatics[63]; /* '__module__' */
    cpy_r_r230 = CPyDict_SetItem(cpy_r_r128, cpy_r_r229, cpy_r_r228);
    cpy_r_r231 = cpy_r_r230 >= 0;
    if (unlikely(!cpy_r_r231)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 84, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r232 = CPyStatic_globals;
    cpy_r_r233 = CPyStatics[49]; /* 'dataclass' */
    cpy_r_r234 = CPyDict_GetItem(cpy_r_r232, cpy_r_r233);
    if (unlikely(cpy_r_r234 == NULL)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 83, CPyStatic_globals);
        goto CPyL57;
    }
    cpy_r_r235 = CPyStatics[50]; /* 'dataclasses' */
    cpy_r_r236 = CPyDataclass_SleightOfHand(cpy_r_r234, cpy_r_r95, cpy_r_r128, cpy_r_r130, cpy_r_r235);
    CPy_DECREF(cpy_r_r234);
    CPy_DECREF(cpy_r_r95);
    CPy_DECREF(cpy_r_r128);
    CPy_DECREF(cpy_r_r130);
    if (unlikely(!cpy_r_r236)) {
        CPy_AddTraceback("mypyc_variant.py", "<module>", 84, CPyStatic_globals);
        goto CPyL53;
    }
    return 1;
CPyL53: ;
    cpy_r_r237 = 2;
    return cpy_r_r237;
CPyL54: ;
    CPy_DecRef(cpy_r_r58);
    goto CPyL53;
CPyL55: ;
    CPy_DecRef(cpy_r_r95);
    goto CPyL53;
CPyL56: ;
    CPy_DecRef(cpy_r_r95);
    CPy_DecRef(cpy_r_r128);
    goto CPyL53;
CPyL57: ;
    CPy_DecRef(cpy_r_r95);
    CPy_DecRef(cpy_r_r128);
    CPy_DecRef(cpy_r_r130);
    goto CPyL53;
}
    
    int CPyGlobalsInit(void)
    {
        static int is_initialized = 0;
        if (is_initialized) return 0;
        
        CPy_Init();
        CPyModule_mypyc_variant = Py_None;
        CPyModule_builtins = Py_None;
        CPyModule___future__ = Py_None;
        CPyModule_dataclasses = Py_None;
        CPyModule_typing = Py_None;
        CPyModule_numpy = Py_None;
        CPyModule_torch = Py_None;
        if (CPyStatics_Initialize(CPyStatics, CPyLit_Str, CPyLit_Bytes, CPyLit_Int, CPyLit_Float, CPyLit_Complex, CPyLit_Tuple, CPyLit_FrozenSet) < 0) {
            return -1;
        }
        is_initialized = 1;
        return 0;
    }
    
    PyObject *CPyStatics[76];
    const char * const CPyLit_Str[] = {
    "\n\006FIELDS\005torch\004none\vseq_lens_np\006cumsum\005int64\005zeros\005dtype\005int32\006arange",
    "\n\005numpy\aasarray\003sum\004copy\004axis\afloat32\tworkspace\003any\003max\005shape",
    "\006\nfrom_numpy\003mul\005clone\bnum_reqs\021num_actual_tokens\rmax_query_len",
    "\004\021num_decode_tokens\022num_prefill_tokens\vmax_seq_len\nblock_size",
    "\005\fnum_kv_heads\thead_size\016sliding_window\vspec_decode\006causal",
    "\005\016kv_cache_dtype\017query_start_loc\fslot_mapping\ttoken_ids\vblock_table",
    "\005\tpositions\023num_computed_tokens\bseq_lens\bbuiltins\vannotations",
    "\a\n__future__\tdataclass\vdataclasses\003Any\006typing\002np\020mypyc_variant.py",
    "\005\b<module>\rmypyc_variant\017__mypyc_attrs__\nMetaNative\rMetaDataclass",
    "\004\017__annotations__\026mypyc filler docstring\a__doc__\n__module__",
    "",
};
    const char * const CPyLit_Bytes[] = {
    "",
};
    const char * const CPyLit_Int[] = {
    "\0041\000-1\0002\0000",
    "",
};
    const double CPyLit_Float[] = {0};
    const double CPyLit_Complex[] = {0};
    const int CPyLit_Tuple[] = {
    8, 1, 10, 1, 17, 1, 47, 1, 49, 1, 51, 3, 13, 13, 53, 3, 4, 4, 4, 2,
    73, 74
};
    const int CPyLit_FrozenSet[] = {0};
    CPyModule *CPyModule_mypyc_variant__internal = NULL;
    CPyModule *CPyModule_mypyc_variant;
    PyObject *CPyStatic_globals;
    CPyModule *CPyModule_builtins;
    CPyModule *CPyModule___future__;
    CPyModule *CPyModule_dataclasses;
    CPyModule *CPyModule_typing;
    CPyModule *CPyModule_numpy;
    CPyModule *CPyModule_torch;
    int CPyExec_mypyc_variant(PyObject *module);
    PyTypeObject *CPyType_MetaNative;
    PyObject *CPyDef_MetaNative(CPyTagged cpy_r_num_reqs, CPyTagged cpy_r_num_actual_tokens, CPyTagged cpy_r_max_query_len, PyObject *cpy_r_query_start_loc, PyObject *cpy_r_seq_lens, PyObject *cpy_r_block_table, PyObject *cpy_r_slot_mapping, char cpy_r_spec_decode, CPyTagged cpy_r_num_decode_tokens, CPyTagged cpy_r_num_prefill_tokens, PyObject *cpy_r_num_computed_tokens, PyObject *cpy_r_seq_lens_np, CPyTagged cpy_r_max_seq_len, PyObject *cpy_r_kv_cache_dtype, CPyTagged cpy_r_block_size, CPyTagged cpy_r_num_kv_heads, CPyTagged cpy_r_head_size, char cpy_r_causal, CPyTagged cpy_r_sliding_window, PyObject *cpy_r_token_ids, PyObject *cpy_r_positions, PyObject *cpy_r_workspace);
    PyTypeObject *CPyType_MetaDataclass;
    PyObject *CPyDef_MetaDataclass(PyObject *cpy_r_args, PyObject *cpy_r_kwargs);
    char CPyDef_MetaNative_____init__(PyObject *cpy_r_self, CPyTagged cpy_r_num_reqs, CPyTagged cpy_r_num_actual_tokens, CPyTagged cpy_r_max_query_len, PyObject *cpy_r_query_start_loc, PyObject *cpy_r_seq_lens, PyObject *cpy_r_block_table, PyObject *cpy_r_slot_mapping, char cpy_r_spec_decode, CPyTagged cpy_r_num_decode_tokens, CPyTagged cpy_r_num_prefill_tokens, PyObject *cpy_r_num_computed_tokens, PyObject *cpy_r_seq_lens_np, CPyTagged cpy_r_max_seq_len, PyObject *cpy_r_kv_cache_dtype, CPyTagged cpy_r_block_size, CPyTagged cpy_r_num_kv_heads, CPyTagged cpy_r_head_size, char cpy_r_causal, CPyTagged cpy_r_sliding_window, PyObject *cpy_r_token_ids, PyObject *cpy_r_positions, PyObject *cpy_r_workspace);
    PyObject *CPyPy_MetaNative_____init__(PyObject *self, PyObject *args, PyObject *kw);
    PyObject *CPyDef__kwargs(PyObject *cpy_r_inp);
    PyObject *CPyPy__kwargs(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames);
    PyObject *CPyDef_small_ops(PyObject *cpy_r_inp, PyObject *cpy_r_mode);
    PyObject *CPyPy_small_ops(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames);
    CPyTagged CPyDef__read_scalars(PyObject *cpy_r_m);
    PyObject *CPyPy__read_scalars(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames);
    CPyTagged CPyDef__read_containers(PyObject *cpy_r_m);
    PyObject *CPyPy__read_containers(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames);
    CPyTagged CPyDef_run_mypyc_native(PyObject *cpy_r_inp, PyObject *cpy_r_mode);
    PyObject *CPyPy_run_mypyc_native(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames);
    CPyTagged CPyDef_run_mypyc_dataclass(PyObject *cpy_r_inp, PyObject *cpy_r_mode);
    PyObject *CPyPy_run_mypyc_dataclass(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames);
    char CPyDef___top_level__(void);
