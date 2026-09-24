/* probe_load.c - controlled synthetic CPU load for perf qualification.
 *
 * Purpose: give the profiling toolchain (perf stat / perf record -g /
 * perf annotate / libkperfx) a *deterministic, symbolised* victim so that the
 * collection contract can be validated without touching an NPU.
 *
 * The mix is deliberately "python-like": unpredictable branches (dict/list
 * lookup in CPython lowers to compare+branch chains), pointer chasing over a
 * small working set (object headers) and a little floating point.  Build with
 * -O2 -g -fno-omit-frame-pointer so perf annotate can map samples back to
 * source lines.
 *
 * Usage: probe_load [--duration-ms N] [--threads T] [--kind alu|mem|mixed]
 */
#define _GNU_SOURCE
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TABLE_BITS 14
#define TABLE_SIZE (1u << TABLE_BITS)
#define TABLE_MASK (TABLE_SIZE - 1u)

static volatile double g_sink;
static volatile uint64_t g_sink_u;

static uint64_t now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Keep the three hot kernels as real (non-inlined) symbols: `perf annotate
 * -s <sym>` needs samples attributed to the symbol itself, and a flame graph
 * that only shows `worker` would hide exactly the attribution we are testing.
 * (Found the hard way: at -O2 all three were inlined into worker and
 * `perf annotate` reported "data has no samples".) */
#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

/* Unpredictable branch + integer ALU chain.  This is the function that is
 * expected to dominate the flame graph; it is exported so `perf annotate -s
 * probe_hot_branch` works. */
NOINLINE uint64_t probe_hot_branch(uint64_t state, const uint32_t *tab, uint32_t n)
{
	uint64_t acc = 0;
	for (uint32_t i = 0; i < n; i++) {
		state ^= state << 13;
		state ^= state >> 7;
		state ^= state << 17;
		uint32_t idx = (uint32_t)(state & TABLE_MASK);
		uint32_t v = tab[idx];
		/* data dependent branch: mirrors CPython's type/ob_type tests */
		if ((v & 0x7u) == 0x3u)
			acc += v * 3u + i;
		else if ((v & 0x1fu) == 0x7u)
			acc ^= (uint64_t)v << (i & 7);
		else
			acc += v ^ idx;
	}
	return acc;
}

/* Pointer chasing over the same table: L1/L2 resident, feed the memory
 * bound / ptag stall view. */
NOINLINE uint64_t probe_hot_pointer_chase(const uint32_t *tab, uint32_t steps)
{
	uint32_t idx = 0;
	uint64_t acc = 0;
	for (uint32_t i = 0; i < steps; i++) {
		idx = tab[idx & TABLE_MASK] & TABLE_MASK;
		acc += tab[(idx + i) & TABLE_MASK];
	}
	return acc;
}

/* FP dominated helper (double divide is not vectorised away). */
NOINLINE double probe_hot_fp(uint64_t seed, uint32_t n)
{
	double x = (double)(seed & 0xffffu) + 1.0;
	double acc = 0.0;
	for (uint32_t i = 1; i <= n; i++) {
		acc += x / (double)(i & 0xffffu ? (i & 0xffffu) : 1);
		x = fma(x, 1.0000001, 0.5);
	}
	return acc;
}

struct thread_arg {
	uint64_t deadline;
	const uint32_t *tab;
	int kind;
	uint64_t acc;
};

static void *worker(void *p)
{
	struct thread_arg *a = (struct thread_arg *)p;
	uint64_t state = 0x123456789abcdefull ^ (uint64_t)(uintptr_t)a;
	uint64_t acc = 0;
	double facc = 0.0;
	while (now_ns() < a->deadline) {
		if (a->kind != 1) {
			acc += probe_hot_branch(state, a->tab, 4000);
			state += 0x9e3779b97f4a7c15ull;
		}
		if (a->kind != 0)
			acc ^= probe_hot_pointer_chase(a->tab, 4000);
		facc += probe_hot_fp(state, 2000);
	}
	a->acc = acc ^ (uint64_t)facc;
	return NULL;
}

int main(int argc, char **argv)
{
	long duration_ms = 2000;
	int threads = 1;
	const char *kind_s = "mixed";

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--duration-ms") && i + 1 < argc)
			duration_ms = strtol(argv[++i], NULL, 10);
		else if (!strcmp(argv[i], "--threads") && i + 1 < argc)
			threads = (int)strtol(argv[++i], NULL, 10);
		else if (!strcmp(argv[i], "--kind") && i + 1 < argc)
			kind_s = argv[++i];
		else {
			fprintf(stderr, "usage: %s [--duration-ms N] [--threads T] "
					"[--kind alu|mem|mixed]\n", argv[0]);
			return 2;
		}
	}
	int kind = strcmp(kind_s, "alu") == 0 ? 0
		 : strcmp(kind_s, "mem") == 0 ? 1 : 2;

	uint32_t *tab = malloc(sizeof(uint32_t) * TABLE_SIZE);
	if (!tab) {
		perror("malloc");
		return 1;
	}
	uint64_t s = 88172645463325252ull;
	for (uint32_t i = 0; i < TABLE_SIZE; i++) {
		s ^= s << 13;
		s ^= s >> 7;
		s ^= s << 17;
		tab[i] = (uint32_t)(s & TABLE_MASK);
	}

	uint64_t deadline = now_ns() + (uint64_t)duration_ms * 1000000ull;
	pthread_t *th = calloc((size_t)threads, sizeof(*th));
	struct thread_arg *args = calloc((size_t)threads, sizeof(*args));
	for (int t = 0; t < threads; t++) {
		args[t].deadline = deadline;
		args[t].tab = tab;
		args[t].kind = kind;
		if (pthread_create(&th[t], NULL, worker, &args[t]) != 0) {
			fprintf(stderr, "pthread_create failed\n");
			return 1;
		}
	}
	uint64_t acc = 0;
	for (int t = 0; t < threads; t++) {
		pthread_join(th[t], NULL);
		acc ^= args[t].acc;
	}
	g_sink_u = acc;
	g_sink = probe_hot_fp(acc, 1000);
	printf("probe_load done kind=%s threads=%d duration_ms=%ld acc=%llu\n",
	       kind_s, threads, duration_ms,
	       (unsigned long long)acc);
	return 0;
}
