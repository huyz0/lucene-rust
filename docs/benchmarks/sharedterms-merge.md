# Shared-terms specialized merge

The paired `cx_stsplit_merge` case measures
`STUniformSplitTermsWriter.merge` against
`StUniformSplitPostingsFormat::merge`, using Lucene 10.5.0 and the same
`fixtures/data/codecs_sharedterms_merge/unsorted` source bytes.

Three independent 300-document segments have different field numbering,
deleted documents, a wholly deleted field, shared and unique terms, sparse
frequency and position fields, offsets, payloads, and norms. There are 805
surviving documents. Both sides open the source readers and prepare document
maps and mapped norms before measurement, then allocate and finish all six
output files for each merge. Java calls the direct shared-terms consumer,
without the rest of `IndexWriter.addIndexes`. Rust uses the specialized merge;
its fallback panics if called. An untimed FNV checksum includes every output
byte and each file's length, in the same order on both sides. Java also checks
that checksum against the committed merged segment.

The first paired run used three alternating repetitions, 1,500 ms warmup and
2,000 ms measurement per engine, pinned to CPUs 0 and 1 in the capped test
container. Rust took 630,718 ns per merge and Java 588,798 ns: **0.93x**,
inside the 1.10x noise floor. This establishes the faithful port baseline;
it does not establish that Rust is faster or slower.

The finalization path now computes each field's seen-document cardinality
once when serializing metadata, as Java's `FieldMetadata.getDocCount` does.
The baseline recomputed it for every term. The equal-term grouping vector is
also reused across terms instead of allocated each time. Both changes leave
the shared-block and postings bytes unchanged; the differential test compares
every file against Java for both specialized and sorted fallback merges.

With 5,000 ms warmup and measurement, the paired run measured 570,788 ns
for Rust and 496,297 ns for Java: **0.87x**, outside the 1.09x noise floor.
This exposes a remaining cost after the first changes. Reusing each target
field's source-postings list across terms and omitting the position-pointer
arrays for fields without positions measured 537,553 ns against 481,455 ns:
**0.90x**, with a 1.03x noise floor. Those arrays were allocated even for the
fixture's 805 singleton ID terms. Keeping single-term field statistics and the
singleton document pointer inline measured 534,918 ns against 485,729 ns:
**0.91x**, with a 1.02x noise floor.

A CPU sampling profile of the remaining gap attributed 12.86% of 5,001
samples to `malloc`, `free` and `realloc`. The sample covered 20 CPU seconds
with a requested 1 ms sampling interval; kernel timer coalescing reduced the
sample count. Other system-library samples lacked source locations, so their
nearest stripped symbols cannot support precise attribution. This allocation
cost motivated streaming source occurrences into the merged term through the
existing checked postings walker, removing source per-document position
containers and per-occurrence payload allocations.

The streamed merge measured **485,460 ns** for Rust and **488,729 ns** for
Java: **1.01x**, within the **1.02x** noise floor. On this workload the final
run cannot distinguish their speed. Both checksums are `c6e164931bc40637`.

The merge buffers one source term's document IDs and frequencies, then streams
its surviving positions, offsets and payloads into the merged current term.
It keeps one seen-document bitset per target field and the completed output
buffers. It does not build a segment-wide map of merged terms or postings.

Run the comparison with:

```sh
scripts/docker-test.sh scripts/bench-micro.sh --bench stmerge \
  --warmup-ms 5000 --measure-ms 5000 --reps 3
scripts/docker-test.sh bash scripts/bench-stmerge-memory.sh
```

The memory companion reports Linux's peak process RSS for each engine in a
fresh child process. This includes each runtime, loaded readers and output
buffers; it is not an isolated measurement of the merge's allocator usage.
The streamed merge measured Rust at 11,896 KiB and Java at 405,080 KiB;
both output checksums were `c6e164931bc40637`.

This case covers a small deletion-heavy shared dictionary. It does not
measure index sorting, foreign-format fallback, a large vocabulary, or an
end-to-end index merge.

Separate correctness fixtures in `codecs_sharedterms_merge_blocks` exercise
source terms with two full position blocks and a tail, documents spanning
block boundaries, deletions in the full blocks and the tail, and a wholly
deleted positional field that also spans a full block and a tail. Both sorted
and unsorted merges must match all six Java files. These boundary fixtures
are outside the timed case above.
