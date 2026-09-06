**English** | [中文](bench_mark_zh.md)

Kimi Delta Attention (KDA) is the linear-attention block of Kimi Linear: Gated
DeltaNet with the forget gate carrying a channel index, so the decay inside a
chunk is a `K`-wide vector rather than one scalar per token.

### Performance Testing

Input parameter definitions:

| Parameter | Value | Description |
|-----------|-------|-------------|
| B | 1 | Batch size |
| SEQ | 4096 | Sequence length |
| H | 4 / 96 | Query heads (`HV = H`, no GVA) |
| K, V | 128 | Key and value head dimension |
| C | 64 | Chunk size |
| BC | 16 | Anchor block inside a chunk |
| dtype | float16 / bfloat16 | Gate `g` is fp32 |

`H = 96` is the head count Kimi K3 runs, and it is the row that matters.

Best performance results, float16:

| H | AscendC | tileLang | Performance Ratio (AscendC/tileLang) |
|------|------|------|------|
| 4 | 781.16u ±5.6 | 867.62u ±10.8 | 90.0% |
| 96 | 11443.26u ±3.3 | 14251.82u ±40.0 | **80.3%** |

The 80.3% is not a point estimate that happens to clear a line: the whole
collection spread clears it, 80.21% at the slowest collection and 80.67% at the
fastest, against a denominator whose own spread is 6.7u.

Both configurations, and bfloat16, at the same shapes. `route_b` and
`KDA_WY_FIXEDCORE` are both opt-in; the default column is what a caller who asks
for nothing gets:

| H | dtype | default | `route_b` | `route_b` + `KDA_WY_FIXEDCORE` |
|------|------|------|------|------|
| 4 | fp16 | 1429.69u ±5.8 (54.6%) | 867.62u ±10.8 (90.0%) | not collected |
| 96 | fp16 | 28244.60u ±48.6 (40.5%) | 14459.51u ±31.7 (79.1%) | 14251.82u ±40.0 (80.3%) |
| 4 | bf16 | 1445.55u ±9.5 | 897.42u ±8.0 | not collected |
| 96 | bf16 | 28380.11u ±2.2 | 14671.21u ±23.2 | not collected |

bf16 reaches `route_b` through a float32 round trip in stage 6, because the part
has no bfloat16 vector select. It costs 1.5% end to end at `H = 96` and 3.4% at
`H = 4`. There is no bf16 reference collection, so no ratio is given.

Per stage at `H = 96`, fp16:

| | cumsum | kkt | solve_tril | wy_fast | chunk_h | chunk_o |
|------|------|------|------|------|------|------|
| default | 963.2u | 10084.9u | 1356.9u | 1704.8u | 3106.2u | 11117.1u |
| `route_b` | 955.7u | 3307.0u | 1353.8u | 1698.5u | 3084.7u | 4053.4u |
| `+ KDA_WY_FIXEDCORE` | 937.9u | 3326.6u | 1356.9u | **1506.2u** | 3087.8u | 4009.9u |

The decode path is reported on its own terms, since the vendor package ships no
recurrent operator to divide by. At `H = HV = 96`, `K = V = 128`, fp16, one step
costs 115.50u at `B = 1` and 7152.52u at `B = 64` — linear to within 3%, so it is
work-bound rather than launch-bound. Its scalar pipe sits at 51.7% against the
vector pipe's 46.5%, which is the next thing worth optimising there.

**Measurement method.** `msprof` in its full form (`msprof op` returns Task
Duration 0.000000 on this box), device Task Duration read from `op_summary`.
Every figure is the median of **three independent collections**, each of 12
iterations at `H = 4` and 6 at `H = 96`, with the first iteration dropped — it
carries first-touch cost and is always the outlier. The half-range across
collections is quoted so the reader can see what the number is worth; collections
of an identical configuration vary by up to 25u per stage, which is why nothing
here rests on a single run. The six stages all compile to a `prim_func` named
`main` and so share one Op Name; they are told apart by launch order, one prefill
being six launches in a fixed sequence, with `Block Num` corroborating. Board is
`Ascend910_9362` (910_93), 20 AI cores.

`bench.sh` in this directory is a different instrument: it profiles each stage's
own correctness sweep, a mix of shapes, and answers "did this stage regress", not
"what is this shape worth".

### Optimization Strategies and Impact Analysis

For the KDA operator we adopted the following optimizations, in this order. Each
marker is the absolute time after that step, measured on board at `H = 4`, and
its ratio against the reference:

0. **Correct, unoptimized** — the six-stage pipeline straight from the paper's
   factorisation  **--- 5992.20u, 13.0%**
1. **Instruction Vectorization**: materialise the broadcasts that otherwise lower
   to one narrow instruction per row  **--- 2417.41u, 32.3%**
2. **Algorithm to Cube (kkt)**: an anchored `BC` decomposition puts the
   off-diagonal strips of the gated Gram matrix into a plain matmul
   **--- 1688u, 46.3%**
3. **Algorithm to Cube (solve_tril)**: a doubling Neumann series replaces 62 rows
   of serial forward substitution with 8 matmuls  **--- 1584.83u, 49.3%**
4. **Redundant Computation Elimination**: five cuts, each one a piece of work
   another stage had already done  **--- 1438.16u, 54.3%**
5. **`route_b`**: the diagonal blocks join the strips on the cube
   **--- 867.62u, 90.0%**

Notes on the two that carry most of the gain:

**Instruction vectorization.** An operand missing one of the `T.Parallel` indices
is a broadcast, and this dialect lowers it to one narrow instruction per row
inside a loop the compiler names `outer_broadcast_idx`, each preceded by a
barrier. `T.tile.broadcast` into a tile that is dead at that point spreads it in
one wide instruction instead, at no extra UB. Measured 1866.80u -> 122.46u on an
isolated micro-benchmark, bit-identical.

**Putting the Gram matrix on the cube.** With a per-channel gate the decay sits
inside the sum over `d`, so `sum_d k_i[d] k_j[d] exp(g_i[d] - g_j[d])` is not a
matmul. Splitting the exponent at an anchor row `a` factors it into a term in `i`
and a term in `j`, which fold into the two operands and leave a plain `X Y^T`.
For the off-diagonal strips both factors are bounded; for the diagonal blocks the
column factor is not, which is why they stayed on the vector unit until
`route_b`. `route_b` raises the clamp on that column factor and moves the cube
operands to bfloat16, whose exponent range holds it. It is off by default because
it is an approximation: a gate steep enough to span more than the clamp inside
one block saturates. It is also unavailable under `cu_seqlens` -- a varlen call
that asks for it falls back to route A and warns.

**Why there is no operator fusion.** Three things were tried and measured, and
each is recorded here so it is not tried again:

- **Fusing stages to keep tensors off GM does not work in this dialect.**
  `T.copy(ub, l1)` never emits a UB→L1 move: the generated AscendC contains zero
  `copy_ub_to_l1`, and the compiler silently routes it through GM instead. There
  is no UB→L1 path to fuse across.
- **There is no inter-stage gap to recover anyway.** Wall clock for the whole
  prefill is 14151.58u against a sum of the six kernel times of 14459.51u, so the
  launches already overlap; fusion would be removing a cost that is not being
  paid. The official PyPTO implementation reaches the same conclusion by
  construction — it also lands `gk`, `aqk`, `akk`, `w`, `u`, `qg`, `kg`, `v_new`
  and `h` in GM.
- **Cube-side multi-buffer is correct but worth nothing here.** `aic_mac`
  occupancy is 4.2%, so a deeper pipeline has nothing to hide behind. Occupancy
  evidence beat instruction-shape evidence.

What is actually left is synchronisation and per-block setup. Stage 6 spends
30.8% of itself on synchronisation — priced by rebuilding it with the sync
inserter off, which is numerically wrong but times the barriers — and all
eighteen of its `SetFlag` / `WaitFlag` pairs are a set followed immediately by
its own wait, so nothing overlaps anything. Across all six stages 4561u of the
14459u is on the scalar unit, and it is per-block setup rather than arithmetic:
stage 4's 6144 blocks each run 277 ns against a 164 ns prologue.
`KDA_WY_FIXEDCORE` is that observation applied to one stage; the other five have
not had it.

### Optimization Results

Rows are the configurations this operator ships. The first seven columns are the
seven optimizations of `examples/flash_attention/fa_opt/bench_mark.md`, in its
order; the last two are axes absent from that list because flash attention is a
matmul to begin with, whereas stages 2 and 3 here were not and had to be
rewritten until they were.

| Configuration | L1 Residency | Instruction Vectorization | Multi-Buffer | Sync Elimination | CV pipelined | Optimized Sync Frequency | Reduced Instructions | Algorithm to Cube | Redundancy Removal | Performance (`H = 96`) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| default | × | √ | × | × | × | × | × | strips | √ | 40.5% |
| `route_b` | × | √ | × | × | × | × | × | strips + diagonal | √ | 79.1% |
| `route_b` + `KDA_WY_FIXEDCORE` | × | √ | × | × | × | × | × | strips + diagonal | √ | **80.3%** |

Two of the crosses are measured dead ends rather than unstarted work, and are
explained above: **Multi-Buffer** (cube-side, no gain at 4.2% MAC occupancy) and
**L1 Residency** (what this operator does is per-task residency, not the
across-basic-block residency that column means). **Sync Elimination** is the
largest single item left.

Operator implementation: https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto/examples/linear_attention_and_rnn/kda

| File Name | Description |
|---|---|
| `kda_chunk_cumsum.py` | stage 1, chunk-local cumsum of the log gate |
| `kda_chunk_scaled_dot_kkt.py` | stage 2, the gated Gram matrix; carries `route_b` |
| `kda_solve_tril.py` | stage 3, the unit lower triangular inverse |
| `kda_wy_fast.py` | stage 4, the UT transform; carries `KDA_WY_FIXEDCORE` |
| `kda_chunk_h.py` | stage 5, inter-chunk state recurrence |
| `kda_chunk_o.py` | stage 6, output; carries `route_b` |
| `kda_recurrent.py` | the decode path, one token at a time |
| `kda_varlen.py` | `cu_seqlens` bookkeeping, shared by both layers |
| `kda_chunk_ref.py`, `kda_ref.py` | the two CPU goldens |
| `bench.sh` | per-stage `msprof` regression sweep |

### Reference

The AscendC operator compared against is `chunk_kda_fwd`:
https://gitcode.com/cann/ops-transformer/tree/master/attention/chunk_kda_fwd

It is not part of the CANN binary release; it is built from that source
repository. Both sides run the same shapes and the same dtype.

It is run in its `safeGate = 1` configuration — its fast path, and the
denominator of every ratio in this file. That flag switches the score operand
from fp16 to bfloat16 (`chunk_kda_fwd_prepare.h:167`), raises the triangular
solve's pipeline depth from 1 to 4 (`:224`), and takes a software-pipelined task
loop (`:2331`).

The shape used here hits `TilingKey 2`, the compile-time specialisation for
`chunkSize == 64 && kDim == 128 && vDim == 128`. On this part that key still
dispatches to the generic implementation: the arch35 specialisation is gated on
`__CCE_AICORE__ == 310`, which is Ascend950.
