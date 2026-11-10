# AVX-512 in full-text search

Which full-text search kernels gain from AVX-512 over the AVX2 build
baseline, measured on a 4-vCPU Sapphire Rapids VM (AVX-512 F/BW/VL/VBMI2,
VPOPCNTDQ). The build targets x86-64-v3; AVX-512 kernels are picked at run
time through `lucene_util::cpu::level`, and `LUCENE_RUST_SIMD=avx2` turns
them off for an A/B on the same binary.

## Where the time goes

`bench-runner` over the 87-query mix (`benchmarks/queries.tsv`) on a
1M-document merged corpus (`scripts/bench-corpus.sh --docs 1000000`), each
query time-boxed equally, sampled with `perf record`. The profile is flat:
the largest self-time entries are the term leg's batch scoring (5.3%),
`memmove` (4.1%), the impact bound refresh (3.9%), postings cursor advance
(3.4%) and shallow advance (3.2%). Doc-block decode is about 6% in all
(`decode_full_block_body` 2.3%, `ForUtil::decode` 2.0%, `expand_bitset`
1.4%), and hit compaction (`copy_from_filtered`) 2.4%. Most of the rest is
branchy, latency-bound work (cursor state machines, heaps, comparators)
that vector width does not touch.

## Kernel micro-benchmarks (AVX2 path vs AVX-512)

| Kernel | Result | Verdict |
|---|---|---|
| Hit compaction by score (`hit_compact::copy_by_score`, `compact_by_score`) | 2.2-6.7x per hit (`vcompresspd`/`vpcompressd` in registers) | **kept** |
| Bit-set block to doc ids (`simd::expand_bitset`) | 1.25-1.75x at 1/8 to full density; very sparse words keep the bit loop | **kept** |
| Delta prefix sum (`prefix_sum_256`) | 1.3x in isolation, no net gain end to end (a separate call where the AVX2 kernel inlines) | dropped |
| `ForUtil` decode, 1-16 bits, recompiled for AVX-512 | about 1.5x on even widths, mixed on odd ones; 17+ bits unchanged | not pursued: about 2% of the mix |
| BM25 batch score (`do_score_batch`) | 1.00x | no gain: the divider has the same per-lane throughput at 512 bits |
| Norm gather (`norm_inverse_batch`) | 1.00x | no gain: load-bound |
| Ordinal counting (aggregations, `vpconflictd` + gather/scatter) | 0.77-1.00x | no gain |
| `pfor_encode` (flush and realtime chunk writes), recompiled for AVX-512 | no change within noise | no gain |

A build of the whole engine with AVX-512 enabled at compile time (so LLVM
auto-vectorizes everything at 512 bits) measured +2.3% geomean over the
AVX2 build on the mix, within the VM's noise, which is why only explicit
kernels with a measured gain are dispatched.

## End to end

Same binary, `LUCENE_RUST_SIMD=avx2` against the default, results
identical (hits, top doc, top score and top set for every query).

- The 87-query mix, best of 3 alternating rounds: +2.8% geomean.
- Disjunction, MaxScore and mixed boolean shapes, median of 5 alternating
  rounds on one pinned core: q12 +26%, q14 +22%, q15 +48%, q45 +24%,
  q47 +10%, q48 +11%, q49 +25%; single-term and conjunction shapes within
  noise (q05 +1%, q08 +3%).
- Profile shares over the mix: hit compaction 1.95% -> 0.93%, bit-set
  expansion 1.44% -> 0.99%.

The VM's run-to-run noise is 10-20% per query, so single-query numbers are
indicative; the geomean and profile shares are the firmer ones.

## Frequency and short-call risk

Both kept kernels are integer/shuffle and compare work (`vpcompressd`,
`vcompresspd`, `vcmppd`, `vcvtps2pd`), the light AVX-512 class: on Ice
Lake and later Intel parts they lower the core clock little or not at all
(not the way heavy floating-point FMA work did on Skylake-SP), and AMD Zen 4
runs 512-bit work as two 256-bit halves with no clock change. Each call
handles a whole batch (up to a few thousand hits, or one 256-doc block), so
the dispatch check (one cached load) is amortised. The compress
instructions are used in their register form and stored with a plain store;
the memory form is microcoded and slow on Zen 4.
