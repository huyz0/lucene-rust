# Installed asserting points writers

The optional `Lucene104Codec::points_writer_factory` routes configured fast and
general flushes, explicit-document flushes and segment merges through an
operation-local points writer. `AssertingPointsCodec` installs Java's writer
dimensionality check and repeated-close behavior while retaining native readers,
the `Lucene104` codec name and native BKD inputs and algorithms.

## Correctness and scope

`GenPointsWriterRouting` invokes Lucene 10.5.0's actual
`AssertingPointsFormat.fieldsWriter`; its reader side returns the original
`Lucene90PointsReader`. Flush readers, mutable trees and `MergeState` are passed
unchanged, preserving Java's algorithm dispatch. The checked-in manifest records
field schemas, document identity and liveness, point multiplicity, bounds,
physical document IDs, lifecycle events, source budgets and independent range
query results.

The Rust replay covers pointful and point-free indices with and without index
sorting, append/reopen, deletes and force merge. A larger case has two multileaf
1D/2D sources plus a point-free source, repeated values and a field whose points
are all deleted. It also merges that result again. Configured-path tests exercise
both fast and general inversion and verify points beside norms, numeric doc
values, stored fields, postings and vectors.

Java retains a merged field's point dimensions after deleting all its points.
The merged segment therefore owns a valid `.kdm`/`.kdi`/`.kdd` triple even when
there are no remaining BKD entries. Rust preserves this schema and accepts a
native reader returning no values for the field on a later merge. `CheckIndex`
records a passed empty-field check; missing files and checksum failures still
fail. Field metadata alone cannot establish that a missing entry accidentally
lost points. The differential manifest checks that semantic contract.

Merge sizing includes source points before deletions. A null source reader,
including a point-free source, selects Java's generic add/finish path even for
1D fields. Otherwise 1D merging retains the native k-way merge. Configured
flushes omit absent opted-in point-only fields. Fields retained for another
configured feature keep their point shape and an empty triple when no values
exist. Fields without points opt-in retain their stored-field metadata.

Flush callbacks are `open`, one `write_field` per populated field, `finish`,
then `close`. Merge is `open`, `merge`, then `close`; merge finishes internally.
Returned operation errors precede close and cleanup errors. The asserting
wrapper closes its delegate twice and stops if the first close fails. Native
output acquisition is data, metadata, index; close attempts metadata, index,
data and preserves the first error. Cleanup tracks successful creations only,
preserving unowned collision destinations. Failure tests verify unchanged
committed bytes and retry of the original buffered documents.

The following adaptations remain explicit:

- The sink receives owned packed fields and mapped merge streams. Source reader
  acquisition, integrity checks and decoding remain in native index code; their
  exception timing is not Java's reader-object callback protocol.
- Native outputs are reserved at open, but headers and flush encoding are
  buffered until finish; merge encodes in its merge callback. Per-field I/O
  failure timing therefore differs from Java's incremental writer.
- Rust flushes fields by number; Java traverses field-hash buckets. Replay
  compares the field callback multiset within each lifecycle. Merge field order
  follows merged field information.
- Points callbacks follow norms/doc-values selection and precede vector,
  stored-field and postings callbacks during flush. The existing wider
  stored-field-versus-postings flush order is not full Java sibling-order parity.
- Reader-side assertion installation, whole-codec composition, IOContext and
  panic cleanup remain incomplete. Returned-error cleanup does not promise
  cleanup during unwinding.

## Paired measurement

Run `scripts/docker-test.sh bash scripts/bench-points-writer-routing.sh 1000 1500`.
The script rebuilds the current Rust release example and fresh Java classes,
checks a Rust index with Java's native readers, then alternates engine order
across six repetitions. Java enables the incubating vector module. The report
requires matching decoded digests, counts and packed payload sizes and estimates
noise from repeated same-engine observations. Encoded point-file sizes are
reported separately: field ordering and BKD pointer encoding can change them
without changing decoded values.

The corpus has 8,192 documents and four merge sources, default 512-point leaves,
1D extrema, sparse repeated values, 3D values with two indexed dimensions and
an all-deleted field. Flush has 42,130 points and 299,592 packed-value bytes.
Merge deletes IDs divisible by seven and retains 7,021 documents, 35,106 points
and 252,760 packed-value bytes. Each iteration checks complete decoded point
multisets, metadata, bounds and stored document identities outside its timer.

The timer includes a flush commit or force-merge plus commit, including points
encoding, finish/close, sibling formats and publication. Corpus construction,
merge-source preparation, delete queuing, verification and destruction are
outside it. It measures the installed whole-writer path rather than isolated
assertion overhead.

On 2026-10-26, six alternating repetitions (1,000 ms warmup and 1,500 ms
accumulated timed work per case) produced the following medians. The ratio is
Java time divided by Rust time; values above 1 favor Rust.

| Case | Rust ms | Java ms | Ratio | Same-engine noise |
| --- | ---: | ---: | ---: | ---: |
| Default flush commit | 6.560 | 8.363 | 1.27x | 1.06x |
| Asserting flush commit | 6.623 | 8.119 | 1.23x | 1.06x |
| Default merge commit | 4.876 | 8.413 | 1.73x | 1.02x |
| Asserting merge commit | 4.880 | 8.240 | 1.69x | 1.03x |

The report's median noise floor was 1.05x. Every Rust/Java ratio exceeds it;
no measured case needs optimization to reach 1.0. These whole-writer results
do not isolate assertion overhead. Java's native readers also read the
asserted Rust merge output and reproduced the complete decoded digest.

Decoded digests, document/field/point counts and packed payload sizes matched
for every case and repetition. Rust's flush triple was 188,855 bytes versus
Java's 188,856; both merge triples were 161,883 bytes. The flush is therefore
not byte-identical. Different field order changes absolute BKD root pointers,
whose VLong widths can change encoded size; the exact one-byte contribution
was not isolated. Equal decoded values are required independently of size.
The shallow adapting codec was 0 bytes in Rust and 72 bytes in Java.

Memory rows cover encoded point triples, decoded packed payload and shallow
configuration only. Peak BKD buffers, whole-writer/directory retention, allocator
overhead and RSS remain unmeasured, so these rows do not establish whole-memory
parity.
