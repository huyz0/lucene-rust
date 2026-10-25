# SimpleText segment info with typed index sorts

`SimpleTextSiMicro.java` and `micro_stsi.rs` measure Lucene 10.5.0's
`SimpleTextSegmentInfoFormat.read` and `write` against
`lucene_index::segment_info::{parse_simpletext, write_simpletext}`.

The shared inputs are `fixtures/data/simpletext_index_sort`: 184 Java-written
`.si` files. They include all four index-sort providers, both directions,
every numeric type, every selector, absent and explicit missing values,
signed zeros, extrema, infinities, NaNs, a four-field sort, and prerelease
segment/minimum versions. Every file was read back by Java. Rust compares
the typed sort description and both version strings with Java's manifest,
then reproduces each file byte for byte.

Reader inputs are loaded before timing: Java uses an in-memory
`ByteBuffersDirectory`, Rust borrowed byte slices. Writer inputs are decoded
before timing; both sides allocate and finish their output inside timing.
Java uses a fresh in-memory output directory for each file; Rust returns its
serialized byte buffer. The reported operation is one segment-info file,
with each loop processing the complete fixture set. Neither case measures a
filesystem or opening a complete index.

An untimed check validates every output byte against the Java fixture and
hashes all output bytes and file lengths in manifest order. Java also
compares the decoded sort and version strings against the manifest.

The final 184-case comparison used three alternating repetitions, 5,000 ms
warmup and 5,000 ms measurement, pinned to CPUs 0 and 1 in the capped
container:

| Case | Rust ns/file | Java ns/file | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Read | 2,412.781 | 4,630.164 | 1.92x | 1.03x |
| Write | 3,359.202 | 5,006.601 | 1.49x | 1.02x |

Both engines and both cases produce checksum `abebb586714dde3b`. Peak
process RSS in the memory companion was 11,828 KiB for Rust and 469,160 KiB
for Java. The faithful adapter meets the time and process-memory comparison;
no algorithm optimisation was needed.

The prerelease cases require retaining SimpleText's complete `Version`;
the binary `.si` descriptor has no prerelease component. The typed adapter
therefore specializes the existing SimpleText metadata model with
`IndexSortField`, and both segment-info formats share the same provider
decoder and serializer. No algorithm optimisation was introduced.

Reproduce the timing and peak-process-memory comparisons with:

```sh
scripts/docker-test.sh scripts/bench-micro.sh --bench stsi \
  --warmup-ms 5000 --measure-ms 5000 --reps 3
scripts/docker-test.sh bash scripts/bench-stsi-memory.sh
```

The memory companion starts a fresh process for each engine and reports
Linux's peak RSS. This includes each runtime, fixture inputs, decoded
metadata and output buffers. It does not isolate parser or writer allocator
usage, or measure memory as the number of sort fields grows.

This workload has small segment infos, mainly with one sort field and one
diagnostic/attribute entry. It does not cover large metadata maps or
end-to-end `DirectoryReader`/`IndexWriter` routing of `SimpleTextCodec`.
