# Native term-vector writer routing

The optional `Lucene104Codec.term_vectors_writer_factory` installs native binary
term-vector callbacks at configured and explicit document flushes and mapped,
deleted or index-sorted merges. `AssertingTermVectorsCodec` wraps those callbacks
with the Java assertion state machine. Codec names, native readers and the
Lucene90 `.tvm`/`.tvd`/`.tvx` format stay the same. The default path retains its
compressed-chunk merge when eligible.

## Correctness evidence

`GenTermVectorsWriterRouting` runs 44 Java scenarios. The Rust fixture replays
their explicit document inputs and another 12 configured document cases. Native
Lucene readers independently reopen all 56 exported Rust indexes. Semantic
snapshots compare every live vector's fields, terms, frequencies, positions,
offsets and payload bytes. Cases include binary terms, UTF-16 field-name order,
missing and empty fields, independent axes, empty and nonempty payloads, chunk
boundaries, deletes, imported field numbering and sorted merges. Java callback
trace and failure rows are provenance; Rust does not compare those traces.

Before any Rust merge, a separate exporter writes 24 first-segment/two-segment
snapshots spanning default/installed writers, append reopen, and absent,
non-string or empty first values. Native Java reconciles FieldInfos, opens an
APPEND writer, adds a populated document, merges, reads vectors by stored ID and
runs CheckIndex. This guards cross-segment schema compatibility that a final
single-segment readback cannot establish.

Separate lifecycle and fault tests check callback counts, returned-error
precedence, all-output close, partial acquisition, write and close failures,
owned-output cleanup and retry. A fully deleted source still has its checksum
checked before factory acquisition. A valid-checksum source with an unknown
field number fails with the typed mapping error after acquisition and close.
Failed public flushes preserve preexisting output collisions and unrelated
files; cleanup uses the complete build's successfully created outputs.

## Measurement scope

The paired benchmark uses 2,048 documents with the same token corpus, deletion
pattern and sorting keys in Rust and Lucene 10.5.0. It measures six cases:
default and asserting flush, merge and sorted merge. Flush includes insertion of
prepared documents plus commit: Java compresses term-vector chunks during
`addDocument`, whereas Rust replays the owned vectors during flush. Merge cases
include `forceMerge(1)` plus commit. Writer setup, document preparation, source
index creation, accounting samples, semantic readback and destruction are
outside the timers. These are whole indexing/merge measurements, not isolated
term-vector codec costs.

The timing script runs six alternating engine-order repetitions, pinned to CPUs
0 and 1, with 1,000 ms warmup and 1,500 ms measured operation time per case.
Native Java also reads the benchmark's Rust indexes before timing. Required
semantic checks must agree across engines and repetitions.

| Case | Rust ms | Java ms | Java/Rust | Noise floor |
|---|---:|---:|---:|---:|
| Default insertion + commit | 29.027 | 68.193 | 2.35x | 1.23x |
| Asserting insertion + commit | 34.698 | 42.921 | 1.24x | 1.03x |
| Default merge + commit | 11.414 | 55.064 | 4.82x | 1.11x |
| Asserting merge + commit | 17.592 | 54.441 | 3.09x | 1.11x |
| Default sorted merge + commit | 12.309 | 55.438 | 4.50x | 1.10x |
| Asserting sorted merge + commit | 18.400 | 55.216 | 3.00x | 1.09x |

These are medians of six repetitions. All 72 measured timing rows have positive
iteration counts; all 72 semantic checks agree, with 216 accounting rows. They
do not isolate assertion overhead or justify comparing default and asserting
rows as a wrapper-only cost.

Rust's encoded term-vector output is 230,487 bytes for flush, 197,571 bytes for
merge and 197,663 bytes for sorted merge. Java writes 139,343, 119,403 and
117,832 bytes respectively. The existing Rust encoder emits literal-only LZ4
units and uses per-array exact packed widths. The comparison therefore includes
different compression work and output sizes; it does not establish compression
size parity. The native callback adapter retains that encoder's output. No
additional hot-loop optimization was needed to meet the timing bar on this
corpus. Compression efficiency remains open.

```sh
scripts/docker-test.sh bash scripts/verify-term-vectors-writer-routing.sh
scripts/docker-test.sh bash scripts/bench-term-vectors-writer-routing.sh 1000 1500
scripts/docker-test.sh bash scripts/bench-term-vectors-writer-routing-memory.sh 8 3
```

The memory companion uses three fresh processes per engine and case, each
performing eight iterations. Linux peak process RSS includes input documents,
source buffers, native readers, verification, runtime/JIT, GC and allocator
retention. Java uses a fixed 256 MiB heap. The benchmark samples `IndexWriter::ram_bytes_used` before commit, before the
Rust callback adapter exists. That counter therefore does not measure adapter
memory. The adapter's separate RAM API excludes opaque encoder chunks and
encoded output buffers. Neither these counters nor process RSS isolate adapter
overhead or establish total memory parity.

All 36 fresh-child samples passed semantic checks and produced 108 valid
accounting rows. Median peak process RSS, in KiB:

| Case | Rust | Java |
|---|---:|---:|
| Default insertion + commit | 49,748 | 403,312 |
| Asserting insertion + commit | 50,076 | 413,220 |
| Default merge + commit | 16,288 | 314,860 |
| Asserting merge + commit | 16,124 | 309,836 |
| Default sorted merge + commit | 16,272 | 312,032 |
| Asserting sorted merge + commit | 16,120 | 318,676 |

The native schema checker was also tested with checksum-valid exported metadata
whose first segment incorrectly cleared `storeTermVector`. Java rejected the
inconsistency during APPEND and merged FieldInfos validation. This fault proof
keeps the current valid empty vector streams; it does not reproduce the old
missing-file state or prove arbitrary corruption detection. The 24 passing
snapshots cover these configured schema cases, not the full Codec/reader graph.

## Remaining scope

The configured facade retains its fixed indexed schema across segments. It
therefore acquires an empty native writer even when every configured field is
missing or non-string. This keeps the vector flag stable for Java append and
merge. Java's dynamically discovered fields and Rust's explicit document API
can omit the field and writer instead. Disabling a configured vector field
removes its prospective vector flag. A configured flush rejects a flag change
against an existing indexed field before sorting, file creation or publication;
its buffered documents remain available for retry. Concurrent writer creation
checks the frozen configuration too. Imports also validate persisted metadata.
This prevents opt-in changes and omitted reconfiguration after reopen from
publishing mixed vector schemas. These checks preserve the existing off/no-file
contract. The configured schema-validation reads are outside the explicit-API
timing cases above.

The owned Rust document model resolves metadata eagerly and validates occurrence
array shapes before term callbacks. Malformed-input error timing can therefore
differ from Java's lazy Fields/Terms/Postings graph. The native adapter buffers
the current document into the existing chunk encoder, assembles output bytes at
finish and reports sticky output I/O errors at close. It closes after returned
errors and preserves the operation error; panic cleanup is not implemented.
Lazy reader assertions, the full Codec/source-reader graph, IOContext routing
and adapter-only peak memory remain open.
