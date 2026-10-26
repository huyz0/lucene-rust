# Cranky codec format wrappers

These test-framework adapters reproduce Lucene 10.5.0's write-failure
guards: reads delegate without drawing randomness, and each write draws
`Random.nextInt(100)` once before entering the delegate. A zero draw throws
the format's exact fake-I/O error. Multiple wrappers can share one random
stream. Rust uses `Rc<RefCell<JavaRandom>>` for these single-threaded tests;
Java's `Random` is thread-safe.

The norms adapter additionally guards consumer opening and field addition
before delegation, and closes the delegate before drawing its close guard.
Delegate close failures therefore consume no random draw.

The explicit adapters use the production Rust codecs and real directory
operations. They do not install `CrankyCodec` in `IndexWriter` or complete
`RandomCodec` routing. The Rust directory API has no `IOContext` parameter.
Benchmark examples keep `lucene-test-framework` outside production
dependencies. Measurements use the capped Rust 1.97.1/JDK 21 container,
alternating engines across three paired repetitions on CPUs 0 and 1.

The Asserting postings, Asserting KNN vectors, Cranky doc values, Asserting
doc values, CheapBastard and Cranky postings results below enable Java's
`jdk.incubator.vector` module, matching the standard microbenchmark runner.
Lucene reports its Panama provider with 256-bit preferred vectors and FMA.
Earlier faithful and intermediate baselines quoted in those sections omitted
the module and used Lucene's scalar provider. Their JVM configuration therefore
differs from the final pairs; the measurements do not isolate a vectorization
speedup or the effect of a Rust change across those configurations.

## Live docs

`CrankyLiveDocsMicro.java` and `cranky_live_docs_micro.rs` replay seed 42
for each batch of 256 write calls. The guard case counts delegate calls and
failures. The complete-write case creates a fresh `ByteBuffersDirectory`,
writes 65 live-doc bits with one deletion through the real Lucene90 delegate,
and consumes every output byte. Both cases include construction amortized
over the batch and handling of injected errors. All required digests matched
in every engine and repetition.

Warmup was 1,000 ms and measurement 1,500 ms per case, with a fixed
128 MiB JVM heap.

| Case | Rust ns/call | Java ns/call | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Failure guard and counting delegate | 3.108 | 12.268 | 3.95x | 1.05x |
| Complete live-doc write and byte consumption | 448.741 | 3,141.235 | 7.00x | 1.14x |

The shared noise floor was 1.09x; both timing differences exceed it. The
faithful implementation needed no algorithm optimisation in this scope.
The write comparison includes each implementation's codec and directory
API costs; it does not isolate the random guard.

Estimated retained storage for one wrapper plus its shared random state,
counted once, was 56 bytes in Rust and 80 bytes in Java. Rust counts the
wrapper, `Rc` reference counters and `RefCell<JavaRandom>`; Java counts
shallow wrapper, `Random` and its `AtomicLong` state. These figures exclude
the delegate, input bits, transient directory/output buffers, allocator
metadata and process runtime. They are not RSS measurements.

`GenCrankyLiveDocs` independently checks 7,168 interleaved writes over seven
seeds, read/file callbacks and RNG tails, including 70 injected failures.
Twelve byte cases cover bitset word boundaries and base36 generations,
corrupted output, and a count error detected after the file has been closed.
The Rust differential test reproduces every `.liv` byte and replays the
callback results. Unit tests separately check delegate errors and that
injection happens before validation or output creation.

```sh
scripts/docker-test.sh bash scripts/bench-cranky-live-docs.sh
```

## Segment info

`CrankySegmentInfoMicro.java` and `cranky_segment_info_micro.rs` use the
actual cranky wrapper and real Lucene99 `.si` directory operations. Each
batch resets the shared random seed to 42 and makes 256 calls. Reads open
the same saved file and digest the decoded document count and file-set size.
Writes build fresh segment metadata, create distinct files in a fresh
`ByteBuffersDirectory`, digest every output byte and the file-set size, and
check that injection leaves the file set unchanged. Construction, error
handling and directory costs are included. All required digests matched.

Warmup was 500 ms and measurement 1,000 ms per case, with a fixed
128 MiB JVM heap.

| Case | Rust ns/call | Java ns/call | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Complete segment-info read | 153.206 | 605.454 | 3.95x | 1.14x |
| Complete segment-info write and byte consumption | 865.961 | 4,337.380 | 5.01x | 1.13x |

Both timing differences exceed each case's measured noise. No algorithm
optimisation was needed. Estimated retained wrapper, concrete delegate and
one shared RNG graph were 56 bytes in Rust and 96 bytes in Java; Rust's
concrete delegate is zero-sized. The estimator excludes directory state,
segment metadata, transient buffers, allocator metadata and process runtime.

Fresh-process peak RSS ranged from 11,732 to 11,948 KiB for Rust and from
151,988 to 152,868 KiB for Java across three repetitions. This includes
runtime startup, JVM heap/GC, directory state and temporary buffers; it does
not measure storage owned by the wrapper alone.

`GenCrankySegmentInfo` saves a real Java `.si` and records 2,048 mixed
operations across eight seed/error configurations with RNG tails. The Rust
test reproduces the file byte for byte and compares every callback outcome,
delegate count and file-set mutation. Units check delegate error identity,
shared RNG state, failed output creation and incorrect segment IDs.
The concrete writer inherits the production encoder's trusted-metadata
precondition: the version major must be at least 7 and files must belong to
the supplied segment name. It does not reproduce Java's validation errors
for invalid versions or file prefixes.

```sh
scripts/docker-test.sh bash scripts/bench-cranky-segment-info.sh 500 1000
```

## Field infos

`CrankyFieldInfosMicro.java` and `cranky_field_infos_micro.rs` make 256
explicit callbacks per batch through counting delegates, resetting seed 0
for each batch. Reads construct a new empty `FieldInfos` result in each
language. Writes draw the guard and count successful delegate callbacks.
Both cases consume suffix lengths, callback counts and RNG continuation.
This measures callbacks plus result construction, without codec or directory
I/O. Warmup was 1,000 ms and measurement 1,500 ms, with a fixed 128 MiB JVM
heap. Every repetition matched the read digest 3139499040 and write digest
2635583194 over 256 calls.

| Case | Rust ns/call | Java ns/call | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Read callback and empty result construction | 6.541 | 12.156 | 1.86x | 1.05x |
| Write guard and counting delegate | 2.113 | 8.573 | 4.06x | 1.02x |

The faithful baseline read cost 19.009 ns in Rust and 13.148 ns in Java
(0.69x). Disassembly showed that `FieldInfos::new` unconditionally entered
fallible iterator collection and consistency checking for empty input. An
empty-input branch now returns an empty vector directly and releases any
reserved input storage; nonempty validation is unchanged. The same benchmark
then measured 6.541 ns. A regression test checks both default and reserved
empty vectors. Both final timing differences exceed their measured noise.

Estimated retained wrapper, counting delegate and one shared RNG graph were
72 bytes in Rust and 112 bytes in Java. This excludes allocator metadata,
directories, empty field infos, static input strings, runtime and RSS.

`GenCrankyFieldInfos` exercises the actual Java wrapper across seven seeds,
256 mixed operations per seed, and RNG tails. The Rust differential tests
replay callbacks, injection and delegate errors, and reproduce a real Java
`.fnm` byte for byte. The concrete writer inherits the production encoder's
trusted-metadata precondition: callers that mutate field metadata must
rebuild it through `FieldInfos::new`; the writer does not repeat Java's
per-field consistency check.

```sh
scripts/docker-test.sh bash scripts/bench-cranky-field-infos.sh
```

## Norms

`CrankyNormsMicro.java` and `cranky_norms_micro.rs` use seed 0 for batches
of 256 operations. Reads open a real production `.nvm`/`.nvd` pair and consume
ten dense/sparse document lookups. The lifecycle case opens a consumer in a
fresh memory directory, adds two fields, closes it and folds every output
byte into the digest. Opening, addition and close injections are included;
both engines consume the same errors and RNG continuation. Java uses
`NumericDocValues.advanceExact`; Rust uses the production norms value reader.
All required digests matched across three alternating repetitions.

Warmup was 1,000 ms and measurement 1,500 ms per case, with a fixed
128 MiB JVM heap.

| Case | Rust ns/call | Java ns/call | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Open producer and consume ten values | 353.310 | 1,030.020 | 2.92x | 1.10x |
| Complete consumer lifecycle and output consumption | 1,006.714 | 5,959.753 | 5.92x | 1.07x |

Both ratios exceed the shared 1.09x noise floor. No algorithm optimisation
was needed. These include the APIs and directory costs, rather than isolating
the random guards. Estimated retained wrapper, concrete delegate and one
shared RNG graph were 56 bytes in Rust and 96 bytes in Java, excluding
consumer/producer buffers, outputs, directories, segment metadata, allocator
metadata and runtime.

The fresh-process memory companion measured Rust peak RSS of 12,384–12,560 KiB
(median 12,480) and Java 167,320–184,004 KiB (median 181,912). Both cases run
in each process. RSS includes transient consumer/producer buffers, directory
state, runtime/JIT, JVM heap/GC and allocator retention; it cannot isolate
the transient-buffer increment or attribute the peak to an individual case.

The native adapter buffers owned fields until close, unlike Java's streaming
callbacks, so encoding and malformed-value errors are delayed. Explicit
output closes still happen on encoding failure, and metadata validation
precedes data opening. Java fixtures cover dense/sparse/empty bytes, shared
callback traces, repeated closes and failed output creation, including the
partial `.nvd` after metadata creation fails. The producer exposes value
lookups and integrity checking, without `NumericDocValues` or a `FieldInfos`
cross-check. Codec routing and IOContext remain absent.

```sh
scripts/docker-test.sh bash scripts/bench-cranky-norms.sh 1000 1500
scripts/docker-test.sh bash scripts/bench-cranky-norms-memory.sh 1000 1500 3
```

## Compound files

`CrankyCompoundMicro.java` and `cranky_compound_micro.rs` reset seed 42 for
each 256-operation batch. Reads open the same Java archive, consume sorted
member names, lengths and every byte, and check archive integrity. Writes
install the same three distinct-length Java inputs once per batch, create
the compound pair, consume all output bytes and remove both outputs before
the next operation. Injected writes verify no outputs and contribute a
failure marker. Wrapper setup is outside timing. Every required digest
matched across three alternating repetitions, with 1,000 ms warmup and
1,500 ms measurement per case and a fixed 128 MiB Java heap.

| Case | Rust ns/call | Java ns/call | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Complete read, member consumption and integrity | 664.416 | 1,999.187 | 3.01x | 1.02x |
| Complete write and output consumption | 1,984.854 | 6,537.028 | 3.29x | 1.05x |

Both ratios exceed the shared 1.04x noise floor; no algorithm optimisation
was needed. Retained wrapper, concrete delegate and one RNG graph were
56/96 bytes Rust/Java, excluding directories and transient buffers. Peak
process RSS ranged from 11,732 to 11,948 KiB Rust and 169,764 to 174,208 KiB
Java, including runtime, JVM heap/GC, directories and transient buffers.

The adapter validates each member before opening the next; combined-failure
fixtures verify early corruption wins over a later open error. Native
encoding buffers whole files, so a failed encode retains headers/count
instead of Java's partially copied member bytes. Equal-length packing order
can differ; exact bytes are proven for distinct-length inputs. The native
reader additionally permits loose files and copies member input buffers.
These are explicit directory adapters without Codec routing or IOContext.

```sh
scripts/docker-test.sh bash scripts/bench-cranky-compound.sh 1000 1500
```

## Asserting live docs

`AssertingLiveDocsMicro.java` and `asserting_live_docs_micro.rs` run with
checks enabled in both engines (`-ea` for Java). The guard case validates
4,096 bits before calling a counting delegate. The write case validates
65 bits, writes a real Lucene90 `.liv` in a fresh memory directory and
consumes every byte. Each batch makes 256 calls and resets its counter.
Wrapper construction, default-codec setup and Java reflection are outside
timing; bitset and segment-state setup are amortized over the batch. All
required digests matched over three alternating repetitions, with 1,000 ms
warmup, 1,500 ms measurement and a fixed 128 MiB Java heap.

| Case | Rust ns/call | Java ns/call | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Check 4,096 bits and count delegate call | 109.291 | 2,602.366 | 23.81x | 1.01x |
| Check 65 bits, write and consume output | 427.819 | 3,810.534 | 8.91x | 1.21x |

The faithful scalar scan measured 4,089.012 ns against Java's 2,433.006 ns
(0.60x). The format's concrete immutable `FixedBitSet` has no access
callbacks, so its deletion count now uses range popcounts. Counting only
the logical range preserves the scalar scan's treatment of dirty padding
accepted by `from_words` in release. Generic asserting bits/live-doc wrappers
still forward masks and iterator objects to their delegates. The unchanged
benchmark reduced Rust's guard time to 109.291 ns. Java write timing varied
between runs; the final write ratio describes the measured lifecycle and
does not isolate the popcount improvement.

The format wrapper occupies zero bytes with Rust's stateless concrete
delegate and 16 shallow bytes in Java. This excludes delegates, bitsets,
directories, output buffers, allocator metadata and runtime; it is not a
whole-process memory comparison. Fixtures cover 28 real bitset cases,
out-of-range masks with partial mutation, signed bounds, corruption,
callback order and iterator forwarding. A release-only unit test compares
dirty padding at lengths 1, 63, 65 and 130 against the original scalar count.
Nullable Java bits, sparse storage selection, docIDRunEnd/intoBitSet,
IOContext and Codec routing remain outside this adapter.

```sh
scripts/docker-test.sh bash scripts/bench-asserting-live-docs.sh
```

## Asserting norms

`AssertingNormsMicro.java` and `asserting_norms_micro.rs` keep assertions
enabled in both engines. The validation case visits 4,096 values twice:
the wrapper checks every document and value, then the delegate re-acquires
the original producer and consumes it. Acquisition, next-document and value
callback counts, producer identity and the value digest are checked. The
read case opens a real 65-document dense/sparse pair and consumes both
iterators. The lifecycle case writes both fields to a fresh directory,
closes the consumer and hashes every output byte. Setup and reflection are
outside timing. Every warmup and timed batch checks its expected digest;
all three required cross-engine digests match.

Three alternating repetitions used 1,000 ms warmup, 1,500 ms measurement,
batches of 256 calls and a fixed 128 MiB Java heap.

| Case | Rust ns/call | Java ns/call | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Validate 4,096 values and consume original producer | 25,298.546 | 3,477.976 | 0.14x | 1.13x |
| Open and consume dense/sparse iterators | 1,646.734 | 1,445.808 | 0.88x | 1.02x |
| Complete write and byte consumption | 2,817.598 | 7,017.347 | 2.49x | 1.10x |

The faithful read baseline was 2,706.498 ns (0.56x). Retaining the existing
production `NormsReader` cursor for sparse iterators removes per-document
field lookup and cursor reconstruction. Dense/empty iterators use a compact
metadata-and-byte view, avoiding embedded sparse state. Generic consumer
inlining exposes the validation boundary without removing callbacks; the
faithful validation baseline was 28,424.157 ns (0.12x).

Two cases remain below 1.0. Validation retains two complete walks through
boxed `NumericDocValues` callbacks. Java can specialize its stable concrete
iterator types at runtime; the Rust seam erases them. Sparse native iteration
also probes absent documents through `advance_exact`, because the production
cursor exposes exact lookup rather than a forward document iterator. Java
advances its `IndexedDISI` directly. These are source-level explanations of
remaining work, not profiler attribution of the measured nanoseconds. A new
generic producer interface or production cursor API is outside this bounded
adapter change; callback work and assertions have not been weakened.

Format plus concrete delegate storage is 0 bytes Rust and 32 bytes Java.
The native iterator's measured shallow state is 48 bytes dense and 232 bytes
sparse, excluding asserting wrappers, shared producer bytes and allocator
metadata. Caching increases retained sparse iterator state; the format-only
comparison does not measure that cost. Fresh-process peak RSS across all
three cases was 12,576–12,700 KiB Rust (median 12,620) and 169,520–172,456 KiB
Java (median 170,756). RSS includes runtime/JIT, transient buffers,
directories, allocator retention and JVM heap/GC. It neither isolates
iterator memory nor proves a per-iterator advantage over Java.

The adapter validates FieldInfos before opening data. Merge handles share
immutable bytes and close independently, matching Java's ByteBuffers input
backend; it does not model backends where closing the original invalidates
clones. Fixtures check both close orders, exact dense/sparse/empty bytes,
callback ordering, producer identity, thread checks and corruption. The
writer buffers owned fields until close. Nullable and bulk numeric APIs,
IOContext and Codec routing remain outside the seam.

```sh
scripts/docker-test.sh bash scripts/bench-asserting-norms.sh 1000 1500
scripts/docker-test.sh bash scripts/bench-asserting-norms-memory.sh 1000 1500 3
```

## Stored fields

`CrankyStoredFieldsMicro.java` and `cranky_stored_fields_micro.rs` cover
complete reads and writes in both compression modes. Reads open real Java
triples, check integrity, visit every field value and close. Writes exercise
all seven field overloads, finish, close, hash every output byte and delete
the outputs. The two base visitor merges use real readers, deleted documents,
field-number remapping and sequential/sorted maps. The fault case checks
exact messages, untouched input, delegated close counts and RNG tails.
Reflection, fixtures and seed selection are outside timing. Every batch
checks its expected result, and all seven required engine digests match.

Three alternating repetitions used 1,000 ms warmup, 1,500 ms measurement,
batches of 32 calls and a fixed 128 MiB Java heap.

| Case | Rust ns/call | Java ns/call | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| BEST_SPEED complete read | 1,008.721 | 2,546.363 | 2.52x | 1.01x |
| BEST_COMPRESSION complete read | 21,111.116 | 5,313.709 | 0.25x | 1.03x |
| BEST_SPEED complete write | 3,230.936 | 14,604.304 | 4.52x | 1.02x |
| BEST_COMPRESSION complete write | 28,273.231 | 36,053.501 | 1.28x | 1.01x |
| Sequential visitor merge | 4,905.634 | 17,077.350 | 3.48x | 1.03x |
| Sorted visitor merge | 4,927.816 | 17,038.026 | 3.46x | 1.02x |
| Failure callbacks and input/close checks | 434.513 | 2,101.530 | 4.84x | 1.00x |

BEST_COMPRESSION reading remains below 1.0. Its actual fixture has one
three-document chunk, 49 plaintext bytes per document, a two-byte dictionary
and 15-byte sub-blocks. All eleven DEFLATE streams use fixed Huffman blocks;
reading all three documents inflates fifteen tiny streams including repeated
dictionary reads. Miniz 0.9.1's block-header path calls `start_static_table`
and `init_tree` for each fixed block, rebuilding literal/length and distance
tables. Upstream zlib's normal fixed-table path uses retained compiled tables.
The container JDK's zlib build flags were not inspected, and profiling has
not attributed the entire measured gap to this difference.

Reusing one miniz state and compressed buffer per unit measured 21,063.110 ns
against the faithful baseline's 20,831.132 ns, with no meaningful improvement.
That experiment was removed. The decoder keeps its original algorithm and
now rejects a compressed-length claim beyond the remaining input before
allocating, including when the requested output length is zero. Replacing
the decompression backend or its fixed-table implementation is outside this
adapter change; whole-chunk caching would also change Java's ordinary reader
behavior and was not used to hide the gap.

Retained format wrapper, concrete delegate and one shared RNG graph occupy
56 bytes Rust and 96 bytes Java. This excludes reader/writer instances,
directories, input values, metadata and transient codec buffers. Peak process
RSS was 11,732–11,940 KiB Rust and 159,460–176,036 KiB Java, including runtime,
JVM heap/GC, directories and transient buffers. Neither metric measures the
native writer's private buffers in isolation.

Java fixtures prove exact output triples in both modes, raw floating-point
bits, superclass visitor merge callbacks and shared RNG continuation. They
also compare partial files after automatic chunk flush then unfinished close,
count failure and retry, index creation/close failures, and metadata/data
close-error precedence. Feature-gated native writer seams publish flushed
chunks before count validation and index creation, and metadata before index
close. Document buffering still delays streaming errors; temporary index
outputs, SegmentInfo mode routing, IOContext and Codec installation remain
outside the adapter.

```sh
scripts/docker-test.sh bash scripts/bench-cranky-stored-fields.sh 1000 1500
```

## Term vectors

`CrankyTermVectorsMicro.java` and `cranky_term_vectors_micro.rs` exercise
the actual cranky wrapper. Proximity decoding re-enters guarded position
callbacks and consumes positions, offsets and payloads. The sorted default
merge consumes remapped documents through the wrapper callbacks. The write
case creates real SimpleText output and hashes every byte; the cached-reader
case consumes every field, term, occurrence, offset and payload. Native
owned-model allocation remains timed. Setup, reflection, reader opening and
merge metadata are outside timing. Every batch verifies its digest, including
RNG continuation, and all four required cross-engine checks match.

Three alternating repetitions used 1,000 ms warmup, 1,500 ms measurement and
a fixed 128 MiB Java heap.

| Case | Rust ns/call | Java ns/call | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Nested proximity guards and decoding | 34.770 | 37.604 | 1.08x | 1.02x |
| Sorted default merge callbacks | 2,133.833 | 2,394.788 | 1.12x | 1.02x |
| Cached reader, complete occurrence consumption | 1,017.577 | 1,167.818 | 1.15x | 1.02x |
| Real SimpleText write and byte consumption | 1,271.343 | 3,723.875 | 2.93x | 1.03x |

The faithful proximity baseline measured 53.508 ns against Java's 37.625 ns
(0.70x). Payloads up to 32 bytes now use stack storage; larger payloads use
a reusable per-call Vec. Empty nonnull payloads and null payloads retain
their distinct callback representations, and reads and RNG draws keep their
order. Advisory inlining on `JavaRandom::next_int_bounded`, the proximity
helper and its thin wrapper exposes constant bounds and concrete input call
sites to the compiler. The unchanged benchmark measures 34.770 ns after
those changes. No callback or digest work was removed.

Retained format/writer wrapper overhead and one shared RNG graph occupy
64 bytes Rust and 104 bytes Java. Delegate storage, input models, decoded
documents, temporary payloads, I/O buffers, allocator metadata and runtime
are excluded. This is not a whole-process or transient-buffer comparison.

Fixtures verify real Java cranky/SimpleText bytes, callback/error ordering,
nested proximity input positions, RNG tails, sorted/deleted/null-vector
merges and negative cases. Owned models cannot express unknown enum counts,
nullable term entries or iterator-produced failures. Buffered writes and
sticky output errors differ from Java streaming behavior; native accounting
reports finished bytes, excluding active private writer buffers. Binary
vectors, IOContext and Codec routing remain outside this adapter.

```sh
scripts/docker-test.sh bash scripts/bench-cranky-term-vectors.sh
```

## Asserting stored fields

`AssertingStoredFieldsMicro.java` and `asserting_stored_fields_micro.rs`
exercise document-state validation, complete real reads and writes in both
compression modes, and the base visitor merge with deletions and sequential
or sorted maps. Validation consumes all seven overloads, finish and double
close. Reads check integrity and consume every field; writes hash every output
byte and delete the outputs. Every batch checks its expected digest, and all
seven required cross-engine digests match. Assertions stay enabled.

Three alternating repetitions used 1,000 ms warmup, 1,500 ms measurement,
batches of 32 calls and a fixed 128 MiB Java heap. Inputs, reflection lookup,
formats, source readers and merge metadata are prepared outside timing.

| Case | Rust ns/call | Java ns/call | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Document-state validation | 17.421 | 39.333 | 2.26x | 1.04x |
| BEST_SPEED complete read | 730.129 | 2,234.523 | 3.06x | 1.08x |
| BEST_COMPRESSION complete read | 13,988.256 | 4,050.788 | 0.29x | 1.02x |
| BEST_SPEED complete write | 2,505.176 | 13,080.626 | 5.22x | 1.06x |
| BEST_COMPRESSION complete write | 43,019.960 | 32,543.875 | 0.76x | 1.08x |
| Sequential visitor merge | 3,686.901 | 15,484.732 | 4.20x | 1.05x |
| Sorted visitor merge | 3,803.811 | 15,134.169 | 3.98x | 1.06x |

The compression read fixture contains 47 plaintext bytes, no dictionary and
five-byte sub-blocks. Its ten final streams use fixed Huffman blocks, with
compressed lengths 7, 5, 7, 7, 7, 7, 7, 7, 7 and 4 bytes. This exercises the
same repeated miniz fixed-table construction described for cranky stored
fields above; profiling has not attributed the entire gap to that cost.

Sharing immutable encoder Huffman tables through OnceLock measured
41,674.525 ns for compression write against the baseline's 43,019.960 ns,
a 3.1% improvement within the candidate run's 5% noise. That experiment was
removed. It did not change the read backend or output bytes.
Replacing fixed-array Box construction with zeroed Vec-to-boxed-array
allocation measured 41,037.906 ns, a 4.6% improvement within that run's 6%
noise. It preserved all seven digests and was also removed. Neither
experiment established a meaningful compression-write improvement.

For this tiny write the encoder still owns a fresh 192 KiB of mutable
window/previous-chain/hash-table arrays. Constructor reset, empty-dictionary
compression and ten sub-block resets fill the 64 KiB hash table twelve times,
or 768 KiB of stores. The reset lifecycle follows Java; these counts alone
do not establish a Rust-versus-Java cause. Rust zero-initializes all three
arrays, while the JDK native allocator and build flags were not inspected.
Each stream also runs the faithful level-six match and Huffman machinery.
The remaining write gap has no isolated profiler attribution; changing
initialization contents or reset order would risk the byte-identity contract.

Writer wrapper overhead is 8 bytes Rust and 24 bytes Java; format plus concrete
delegate shallow storage is 0 and 32 bytes respectively. These exclude delegate
graphs, private codec buffers, directories, outputs, decoded values, allocator
metadata and runtime. They do not establish transient-buffer or RSS parity.

Both measured merge readers use ordinary decoding. Java's merge-instance
decompressed-chunk cache is a separate, unbenchmarked native API gap. The
native document bound comes from parsed metadata rather than a separately
supplied SegmentInfo.maxDoc. Its diagnostic Display string reports maxDoc
rather than Java's compression mode and chunk size; construction formats
these different strings during the benchmark. Borrowed immutable files,
independent clone closes, buffered writer publication/error timing and
accounting retain the explicit native seam's limits. IOContext, SegmentInfo
mode routing and Codec/IndexWriter installation remain absent.

```sh
scripts/docker-test.sh bash scripts/bench-asserting-stored-fields.sh 1000 1500
```

## Cranky points

`CrankyPointsMicro.java` and `cranky_points_micro.rs` measure the real cranky
reader/tree and writer adapters. Reads open multi-leaf files, check integrity,
consume metadata and visited values, and close. Writes use ordinary tree
addition, finish and close, then consume every output byte. Merge cases use
the specialized native one-dimensional path and the wrapped-reader generic
visitor fallback. Every batch verifies values or output bytes plus the shared
RNG tail; all four mandatory cross-engine digests match.

Three alternating repetitions used 1,000 ms warmup, 1,500 ms measurement,
batches of 16 calls and a fixed 128 MiB Java heap.

| Case | Rust ns/call | Java ns/call | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Complete read | 4,834.875 | 5,164.949 | 1.07x | 1.08x |
| Ordinary write | 8,286.334 | 18,664.584 | 2.25x | 1.07x |
| Native one-dimensional merge | 13,089.374 | 24,907.055 | 1.90x | 1.05x |
| Wrapped-reader generic merge | 14,455.304 | 25,149.846 | 1.74x | 1.06x |

The read ratio lies within its noise; it establishes no measured speedup.
The native merge baseline materialized all source points and measured
13,361.563 ns. Leaf streaming measures 13,089.374 ns, so the timing difference
is small. The retained change addresses storage: for 4,096 points and leaf
size four, source tuple/value capacities were 147,456 bytes. The streaming
source retains one leaf's document/value buffers, measuring 32 bytes plus
2,048 bytes of pending cursor shells. This excludes node-internal vectors,
merge heads, inputs, outputs, allocator metadata and runtime. A shared first
error stops iterator peers and is checked before publishing the field output.
Exact Java merge bytes and RNG tails remain unchanged.
Source dimensions, indexed dimensions and byte width must match the target
before acquiring its tree. Mismatches return an always-on typed error in the
native seam, corresponding to Java's BKD merge assertions; regression tests
ensure they cannot publish truncated or regrouped multidimensional values.

The retained format, concrete delegate and shared RNG graph occupy 56 bytes
Rust and 96 bytes Java. This excludes readers/writers, BKD working buffers,
directories, metadata and runtime. Neither this shallow measurement nor the
source-buffer comparison establishes overall memory parity with Java.

Native tree clones escape the cranky wrapper and use the current node as
their navigation root, matching Java. Ordinary writes use BKD add/finish,
without Java's mutable-tree flush seam. Native visitors are fallible locally;
borrowed bytes and owned cursor nodes replace Java IndexInputs and mutable
cursor arrays. Output buffering and sticky close-time write errors remain
explicit deviations. IOContext and Codec/IndexWriter routing remain absent.

```sh
scripts/docker-test.sh bash scripts/bench-cranky-points.sh 1000 1500
```

## Asserting term vectors

`AssertingTermVectorsMicro.java` and `asserting_term_vectors_micro.rs` run
the actual asserting writer with assertions enabled in both engines. The
validation case consumes every callback value, inherited proximity decoding,
finish and double-close delegation. The merge case uses a fresh asserting
writer, sorted document maps, deletions and complete callback consumption.
Writes hash every real SimpleText output byte; reads include clone, integrity,
every field/term/occurrence and double close. The native reader allocates its
owned model during timing, but lacks Java's lazy asserting iterator checks.
These are bounded adapter workloads, not a claim of complete reader parity.

Three alternating repetitions used 1,000 ms warmup, 1,500 ms measurement,
batches of 256 calls and a fixed 128 MiB Java heap. Reflection, source opening,
field metadata and format construction are outside timing. Java retains its
merge-state object; Rust constructs borrowed stack descriptors during timing.
Every batch asserts its expected result, and all four mandatory cross-engine
digests match.

| Case | Rust ns/call | Java ns/call | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Writer assertions and inherited proximity | 31.270 | 39.261 | 1.26x | 1.01x |
| Sorted default merge callbacks | 2,256.665 | 2,740.818 | 1.21x | 1.02x |
| Clone, integrity and complete read | 1,156.838 | 1,463.233 | 1.26x | 1.01x |
| Real SimpleText write and complete byte consumption | 1,178.092 | 3,621.694 | 3.07x | 1.12x |

The faithful callback baseline measured 44.974 ns against Java's 40.426 ns
(0.90x). Advisory inlining on the position assertion callback and a thin
forwarder to the inherited proximity helper reduced it to 31.270 ns. No
assertions, input reads, delegate callbacks or digest work were removed, and
the status/counter layout is unchanged. This is measured evidence for the
candidate, not profiler attribution to a particular compiler transformation.

Writer overhead measures 24 bytes Rust and 48 bytes Java. Rust subtracts the
embedded counting delegate from the wrapper's shallow size; Java reports the
wrapper's shallow object size, including its header and delegate reference.
Delegate objects, private buffers, reader storage, decoded models, allocator
metadata and runtime are excluded. No peak-memory or whole-process claim is
made. Reader clone groups close independently; merge aliases share close
state, and closing the last independent group releases decoded data even if
closed handles remain alive.

The generic writer reproduces Java's assertion ordering and mutations after
successful delegate calls, including permissive null/empty-payload probes.
The concrete native SimpleText delegate does not reproduce its Java twin's
internal position/offset-presence and nonempty-payload assertions. Lazy
Fields/Terms/enums/postings checks, binary vectors, IOContext and Codec routing
remain absent; buffered publication and private-buffer accounting retain the
underlying adapter's limits. Fixtures cover actual Java traces, completed
failure/retry sequences, real bytes and clone/merge close ownership.

```sh
scripts/docker-test.sh bash scripts/bench-asserting-term-vectors.sh 1000 1500
```

## Asserting points and native lifecycle

The pair runs the actual Java asserting format and package-private point visitor
with assertions enabled, against the Rust wrappers over the same Lucene90 files.
Three alternating repetitions use 1,000 ms warmup and 1,500 ms measurement,
128 MiB Java heap and CPUs 0–1. Each batch consumes 16 operations, verifies its
full digest and requires cross-engine agreement for all five cases.

| Case | Rust ns/call | Java ns/call | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Visitor validation, 13 points | 125.173 | 131.771 | 1.05x | 1.04x |
| Open, integrity and complete read | 5,665.559 | 6,023.493 | 1.06x | 1.06x |
| Write and consume all output bytes | 8,321.984 | 19,731.554 | 2.37x | 1.04x |
| Native two-source mapped merge | 13,091.092 | 26,452.147 | 2.02x | 1.03x |
| Wrapped two-source mapped merge | 15,458.338 | 26,822.933 | 1.74x | 1.08x |

The faithful adapter needs no measured algorithm change. The read result is
indistinguishable within noise. Read includes both one- and two-dimensional
metadata, values, document IDs and original/cloned trees; writes and merges
consume all three files. Fixtures, source readers and Java reflective merge-state
setup are outside timing; Rust constructs borrowed stack descriptors in timing.
Rust metadata getters borrow immutable slices while Java returns cloned arrays,
so allocation and ownership costs differ within the measured full read.

One visitor and its three arrays occupy 126 bytes in Rust and 128 bytes in Java.
Rust counts the concrete struct and vector capacities; Java includes object and
array headers. Delegate, input, reader, tree and output storage, allocator metadata,
runtime and RSS are excluded. This is not a whole-reader memory comparison.

Actual Java lifecycle fixtures show the native inherited merge instance is the
same reader: closing either alias clears subsequent values acquisition. Rust
shares close state and returns typed closed errors for integrity and new tree
acquisition; acquired metadata and existing trees remain usable as in Java's
ByteBuffers backend. Rust retains immutable parsed metadata until all handles
drop, whereas Java clears its reader map. IOContext, invalidating input backends
and Codec/IndexWriter routing remain absent, with underlying buffered writer
publication and sticky-error timing limits.

```sh
scripts/docker-test.sh bash scripts/bench-asserting-points.sh 1000 1500
```

## Asserting postings

Remeasured on October 26, 2026 after adding borrowed-term enumeration.
The standalone model retains its compatible owned enumeration path; these
results are regression measurements, not an isolated optimization comparison.

The actual Java wrapper and Rust use the same nine-document Lucene104 segment
and two-source deleted sequential/sorted merges. Frequencies, positions, offsets,
payloads, norms and empty fields are included. Three alternating repetitions use
CPUs 0–1, 1,000 ms warmup, 1,500 ms measurement, assertions and a 128 MiB Java
heap. Every sixteen-operation batch checks complete reader statistics/postings
or every byte and length of all seven files. Validation uses a no-op delegate
and its digest is a completion marker; actual callbacks, flags, reuse and errors
are independently checked against Java trace fixtures.

| Case | Rust ns/call | Java ns/call | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Open, integrity, checked walk and close | 24,913.634 | 23,260.014 | 0.93x | 1.06x |
| Write, validate and consume all bytes | 31,024.185 | 51,876.559 | 1.67x | 1.04x |
| Sequential mapped merge | 64,141.203 | 86,317.077 | 1.35x | 1.05x |
| Sorted mapped merge | 64,322.417 | 86,910.115 | 1.35x | 1.03x |
| Validation | 3,145.421 | 9,674.211 | 3.08x | 1.03x |

The faithful baseline ratios were read 0.43x, write 0.57x, sequential/sorted merge
0.30x/0.30x and validation 0.35x. A temporary phase probe over 4,096 operations
attributed 31% of read time and 46% of validation time to full decoded-field
clones. Merges rebuilt all fields for each names/terms request; the field-access
path consumed about 69% of their measured time. Nested probe timings overlap and
include accounting overhead; the probe was removed before final measurements.

The optimized native producer retains one immutable decoded snapshot and borrows
selected fields and metadata. The mapped view still invokes every source and
checks every posting/map on every request, preserving unrelated failures and
callback order, while allocating postings only for the requested field. Nothing
caches arbitrary source results. Traced-source units exercise repeated calls,
later-source failures, abort/integrity short-circuits and unrelated bad maps.
Document-count computation sorts and deduplicates at most 128 posting IDs
(512 logical scratch bytes); larger fields retain a tree set proportional to
unique documents. Boundary and duplicate-heavy cases verify both paths.

The full-read ratio is 0.93x, just outside its 1.06x repetition spread;
Rust remains slower on this acquisition and checked-walk case. This explicit
adapter eagerly decodes all fields, copies the newly decoded model into
its shared snapshot once, and computes reader statistics from that model; Java
opens lazy enums and reads stored statistics. The profile identified acquisition
and checked enumeration as the remaining work. Removing the acquisition copy
requires an ownership-transfer API in the shared native codec, while matching
Java's lazy reader requires a different production-reader seam. Both are left
outside this wrapper change. Integrity, enumeration and assertion work remain
intact on this small segment.

Source readers, field metadata, norms, maps, format configuration and reflected
Java merge/consumer setup are outside timing. Read includes open, full integrity,
checked statistics/enumeration and close. Write/merge include native encoding,
the fresh assertion walk and double close. Rust owns eager decoded snapshots
and file buffers; Java uses lazy enums and ByteBuffersDirectory. IOContext and
Directory error timing remain absent from this standalone pair. Optional
Codec/IndexWriter routing has its own [installed measurement](postings-writer-routing.md).

The validation consumer and stateless delegate occupy 8/40 bytes Rust/Java.
Native Rust reader bookkeeping occupies 136 bytes, versus 160 in the faithful
baseline, excluding unchanged FieldInfos/file/decoded payload allocations.
One decoded graph is retained; extraction temporarily holds two. Referenced
state, enums, buffers, allocator metadata, runtime and RSS are excluded from
the wrapper comparison; these figures do not measure peak or process memory.

```sh
scripts/docker-test.sh bash scripts/bench-asserting-postings.sh 1000 1500
```

## Asserting KNN vectors

Both engines consume the same sparse sixteen-document float/byte Lucene99 HNSW
segment. Three alternating repetitions use 1,000 ms warmup, 1,500 ms measurement,
CPUs 0–1, assertions and a 128 MiB Java heap. Every sixteen-operation batch checks
all vector values and graph neighbors, or all twelve query/filter cases including
score bits and visits. Java reflection configures the actual asserting format's
delegate before timing; timed reader construction uses its normal fieldsReader.
Cross-engine digests match.

| Case | Rust ns/call | Java ns/call | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Acquire and assert vector values | 94.471 | 513.649 | 5.44x | 1.09x |
| Open, integrity, consume and close | 2,393.427 | 7,972.137 | 3.33x | 1.07x |
| Twelve native searches | 9,970.030 | 14,634.014 | 1.47x | 1.08x |
| Merge reader values and finish lifecycle | 157.086 | 615.034 | 3.92x | 1.11x |

The faithful adapter needs no algorithm change for these workloads. Rust's native
reader checks all data CRCs at open and again at integrity; Java defers data CRCs
to integrity. Borrowed bytes/metadata and independent close handles model the
ByteBuffers backend. A separate actual Java manifest establishes scorer failure,
zero-k short-circuiting, cost-before-bits callbacks and cost-error short-circuiting
for both encodings. These boundary cases are correctness tests outside timing.

The reader wrapper and its two completion counters occupy 56/64 bytes Rust/Java;
Rust includes the Arc allocation header. The asserting AcceptDocs wrapper alone
occupies 24/24 bytes; the temporary guard is excluded. Delegate/filter storage, FieldInfos, input bytes, parsed
metadata, scratch, runtime and RSS are excluded. Native writer/merge construction,
IOContext and Codec routing remain absent; merge lifecycle is not merge throughput.

```sh
scripts/docker-test.sh bash scripts/bench-asserting-knn-vectors.sh 1000 1500
```

## Cranky doc values

The actual Java wrapper and Rust consume two twelve-document Lucene90 segments
with all five sparse types, empty bytes and dictionary terms, signed numeric
extremes, duplicate sorted-numeric values and range skip indexes. Both merged
outputs delete two documents per source and remove deleted-only dictionary terms;
five columns become empty. A separate case closes a consumer without fields.
Three alternating repetitions use 1,000 ms warmup, 1,500 ms measurement,
assertions and a 128 MiB Java heap. Every sixteen-operation batch checks every
value, dictionary and skip field, or all three output files and lengths, plus
RNG continuation. All cross-engine digests match.

| Case | Rust ns/call | Java ns/call | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Open, integrity, consume and close | 6,079.579 | 9,945.150 | 1.64x | 1.05x |
| Write and consume all bytes | 8,564.650 | 65,521.718 | 7.65x | 1.07x |
| Sequential mapped merge | 16,415.141 | 95,643.690 | 5.83x | 1.11x |
| Sorted mapped merge | 16,715.463 | 88,347.949 | 5.29x | 1.06x |
| No-field close | 407.716 | 8,267.289 | 20.28x | 1.05x |

The native adapter builds typed columns directly and validates their existing
slices, removing temporary per-document values and duplicate binary copies.
The Asserting doc values section records the profile behind these shared changes.
Source
readers, maps, live masks and reflective merge-state setup are outside timing.
Opening, integrity and complete value consumption are timed without textual
formatting. Adds, inherited merge callbacks, ordinal maps, encoding, close and
full output consumption are timed. Java constructs the actual wrapper directly.
Rust owns eager columns and returns buffers; Java uses lazy iterators and
ByteBuffersDirectory. Iterator specializations, streaming error timing, optimized
merge-reader caching, Directory/IOContext and Codec routing remain absent.

The native format, wrapper and one shared RNG graph occupy 64/104 bytes
Rust/Java, counting each object once. Columns, readers, consumers, file buffers,
allocator metadata, runtime and RSS are excluded; this is not a peak-memory
comparison.

```sh
scripts/docker-test.sh bash scripts/bench-cranky-doc-values.sh 1000 1500
```

## Asserting doc values

The actual Java `AssertingDocValuesFormat` and Rust consume the same real
Lucene90 files with all five types, complete dictionaries and skip indexes.
Three alternating repetitions use CPUs 0–1, 3,000 ms warmup, 3,000 ms
measurement, assertions and a fixed 128 MiB Java heap. Every sixteen-operation
batch checks complete value/dictionary/skip digests or all three output files
and lengths. All cross-engine digests match. Validation checks a fresh iterator
and delegates the original producer. Merge lifecycle acquires, reads and double
closes a merge producer; it does not measure inherited merge throughput.

| Case | Rust ns/call | Java ns/call | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Validate all five types | 741.920 | 4,131.173 | 5.57x | 1.05x |
| Open, integrity, consume and double close | 9,325.615 | 9,057.130 | 0.97x | 1.02x |
| Write and consume all bytes | 11,834.815 | 57,686.224 | 4.87x | 1.06x |
| Merge-reader lifecycle | 3,446.164 | 5,880.957 | 1.71x | 1.03x |

The faithful scalar-provider baseline measured 5.40x validation, 0.78x read,
4.85x write and 1.70x lifecycle, with 1.02–1.04x repeat noise. A native-open
probe measured 7,664 ns in opening/decoding and 3,328 ns in asserting value/skip
walking. File copying, formatting, integrity and close each cost under 82 ns.
Direct typed-column construction removed temporary per-document `Vec<Value>`
and a flatten/copy pass, improving reads from 11,659 to 10,104 ns (0.87x).
A second probe measured 1,214 ns/open in validation's temporary reconstruction.
Slice validation now checks complete shape before dictionary and per-document
value order; an oracle compares the former validator's exact error variants
and messages on valid, malformed and mixed-defect columns. Shared changes also
improve the Cranky doc values workloads above.

The remaining measured read difference is 268 ns: Rust takes 3.0% longer than
Java. It exceeds the read case's 1.02x maximum/minimum repetition spread,
although the report marks it unresolved against the aggregate 1.04x noise
floor. Neither spread is a confidence interval or proof of equal speed.
This is an explicit below-1
exception: the current profile identifies no further bounded change that
preserves the checks and producer APIs. Rust eagerly owns decoded columns;
Java uses its native lazy producer. This architectural difference limits the
comparison but does not prove the cause of the entire residual difference.
Assertions remain intact.

The shared asserting-leaf iterators also replay 163 cases produced by the
actual Java wrappers over adversarial delegates. They compare callback order,
assertion/runtime/I/O outcomes, recovery state, duplicate count calls and the
different scalar/multivalued movement checks. Running the replay against the
former implementation failed at negative advance: Rust called `docID` before
asserting while Java asserted without that callback. These are untimed
correctness cases; the benchmark retains the normal native iterators.

Additional Rust producer/consumer/skipper wrapper state occupies 24/8/24 bytes,
versus Java shallow wrappers of 32/24/24 bytes including object headers.
Delegates, field metadata, columns/files, temporary bitsets, allocator overhead,
runtime and RSS are excluded. Rust buffers output publication and does not
replay Java's native writer acquisition passes or streaming error timing.
Inherited iterator-consumer merge, bulk APIs, singleton unwrapping, null values,
fallible scalar getters and Directory/IOContext/Codec routing remain absent.
Existing non-Send leaf iterator seams omit Java's runtime iterator thread
guard; merge producers and skippers have explicit tested thread guards.

```sh
scripts/docker-test.sh bash scripts/bench-asserting-doc-values.sh 3000 3000
```

## CheapBastard codec

The actual Java codec's direct Lucene104 postings component uses 100/200
term-block settings, compared with the ordinary 25/48. Both engines process
the same nine-document segment with 1,536 prefixed terms, rich postings and
norms; mapped sequential/sorted merges produce sixteen live documents.
Codec construction is outside timing in both engines. Three alternating
repetitions use CPUs 0–1, 3,000 ms warmup, 3,000 ms measurement, assertions and
a 128 MiB Java heap. Each batch performs sixteen operations and consumes full
posting values or all seven output files and lengths. All cross-engine digests
match. Eighteen seeks use preopened dictionaries and a fresh TermsEnum per target.

| Case | Rust ns/call | Java ns/call | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Open, integrity, enumerate and close | 1,208,239.980 | 6,043,943.434 | 5.00x | 1.06x |
| Write and consume all bytes | 1,539,165.488 | 1,665,703.483 | 1.08x | 1.05x |
| Sequential mapped merge | 2,752,566.088 | 3,595,482.300 | 1.31x | 1.15x |
| Sorted mapped merge | 2,777,820.125 | 3,537,938.528 | 1.27x | 1.11x |
| Preopened seeks | 23,712.689 | 31,097.719 | 1.31x | 1.04x |

All five timing differences exceed their case's observed repetition spread.
The first vector-enabled pair used 1,000/1,500 ms and had up to 2.36x spread
for write and 1.91x for sequential merge. Its read/write/sequential/sorted/seeks
ratios were 5.63x/1.83x/2.05x/1.53x/1.20x; those noisy write and sequential
figures do not establish speedups. The table reports a symmetric longer pair
run to improve repeatability, retaining the same workloads and checks.
Shallow codec and selected-format state occupies 24 bytes Rust and 56 bytes
Java. Delegate component graphs, fields, postings, readers/writers, temporary
maps/holders/columns, buffers, allocator overhead, runtime and RSS are excluded;
there is no whole-reader or peak-memory claim.

The faithful scalar-provider baseline ratios were 4.91x read, 0.92x write, 0.72x
sequential merge, 0.71x sorted merge and 1.18x seeks. Profiling attributed
1.73–1.76 ms to the inherited owned mapping path. Reserving native encoding
columns and retaining invocation-local borrowed posting references removed
deep position/payload graph copies. The first optimized pair measured
4.46x/1.05x/0.97x/0.95x/1.07x respectively. A current-path probe then isolated
aggregation at about 710–712 µs, conversion/norms at 676–683 µs, document-count
preparation at 113–117 µs and native encoding at 919–925 µs. These instrumented
phase estimates are not additive; per-term clock overhead is included.
The second change replaced per-term tree entries and mapped vectors with
per-field flat arenas, stable term/range grouping and one reusable posting-ref
scratch vector. Source callbacks and mapping errors still occur in encounter
order; UTF-16 field order, stable document order and norms timing are unchanged.

This is an explicit component adapter. Java's ten other codec accessors have
a generic per-call callback seam, including fresh points identity; the native
default descriptor supplies no corresponding factories. Codec/SPI installation,
IndexWriter routing and Directory/IOContext are absent. Eager native reader
snapshots, buffered file publication and the per-field norms/error-timing limits
of the Cranky postings adapter remain. The codec is never installed by claiming
its name is Lucene104.

```sh
scripts/docker-test.sh bash scripts/bench-cheap-bastard.sh 3000 3000
```

## Cranky postings

The actual Java `CrankyPostingsFormat` and Rust wrapper consume the same
nine-document Lucene104 segment with frequencies, positions, offsets, payloads,
norms and an empty field. Three alternating repetitions use CPUs 0–1, 1,000 ms
warmup, 1,500 ms measurement, assertions enabled and a 128 MiB Java heap.
Every 16-operation batch checks a digest of every posting value or all seven
output files, their lengths and RNG continuation. Cross-engine digests match.

| Case | Rust ns/call | Java ns/call | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Open, integrity, enumerate and close | 8,831.665 | 19,985.607 | 2.26x | 1.09x |
| Write, close and consume all bytes | 9,337.070 | 45,956.947 | 4.92x | 1.16x |
| Sequential mapped merge | 18,089.432 | 62,100.142 | 3.43x | 1.07x |
| Sorted mapped merge | 18,616.093 | 57,525.302 | 3.09x | 1.06x |

All differences exceed measured noise; the faithful adapter requires no
algorithm change. Source readers, document maps, norms and reflective Java
merge-state construction are outside timing. Both default merges perform abort
and integrity checks, construct their mapped Fields view and invoke the cranky
guarded write during timing. Rust materializes and sorts mapped postings when
the delegate requests them; Java exposes lazy enums. The native Rust reader
eagerly decodes, and the writer returns file buffers instead of publishing through
Java's ByteBuffersDirectory. These are full adapter workloads with different
ownership and allocation costs.

Native field enumeration and mapped merges sort field names with Java's UTF-16
comparison; binary term ordering remains unsigned byte order. Separate real
Java fixtures use U+10000 and U+E000, whose order differs from Rust string order,
and assert delegate-visible field order, every ordinary and merged output byte,
and RNG continuation. Both mapped merge regressions were observed to fail before
the comparison was corrected. The timing workload above retains the same ASCII
fields for comparison with the faithful baseline.

The format, native delegate and one shared RNG graph occupy 56 bytes Rust and
112 bytes Java, counting each object once. Fields, postings, reader/writer state,
input/output buffers, allocator metadata, runtime and RSS are excluded. No
whole-reader or peak-memory claim is made.

The native norms seam snapshots one optional column per nonempty field and
defaults a missing producer/column to 1. Java acquires a norms iterator per term
and requires one for fields with norms, defaulting only missing document values
to 1. Per-term acquisition/error timing, lazy enum failures, Directory publication
and close failures, IOContext and Codec/IndexWriter routing remain outside this
explicit adapter. Generic wrapper callbacks preserve delegate identity and errors.

```sh
scripts/docker-test.sh bash scripts/bench-cranky-postings.sh 1000 1500
```

## Cranky codec composition

`scripts/docker-test.sh bash scripts/bench-cranky-codec.sh` pairs actual Java
`CrankyCodec` with the Rust typed composition adapter over explicit inert
component probes. It measures factories, KNN pass-through, shared guards and
display; it does not install a default native codec graph or perform component
I/O through IndexWriter. Existing component pairs above cover their own native
I/O surfaces.

The faithful composition first passed `GenCrankyCodec`'s complete factory,
identity, error and shared-RNG trace plus unit boundaries. Its initial paired
ratios were 1.99x component factories, 3.33x KNN, 3.51x shared guards and 1.21x
display. Memory exposed a ten-byte short-name allocation above Java's shared
name reference. The measured optimization stores complete UTF-8 names of up to
15 bytes inline, with a boxed immutable snapshot for longer names. It preserves
the single constructor name callback, future name snapshots, live delegate
display and every factory/RNG operation. Tests cover empty, 15/16-byte ASCII,
multibyte boundaries and later delegate mutation. No factory or guard algorithm
was changed.

The remeasured pair uses 1,000 ms warmup and 1,500 ms measurement for each engine
in three alternating repetitions, pinned to the same two CPUs through the
capped runner. Java uses assertions, the vector incubator module and a fixed
128 MiB heap. Times below are median ns per operation; spread is the larger
engine max/min, not a confidence interval. The aggregate report floor is 1.09x.

| Workload | Rust ns | Java ns | Java/Rust | Spread |
|---|---:|---:|---:|---:|
| Two wrappers from each of ten factories | 121.743 | 240.144 | 1.97x | 1.18x |
| Two KNN accessor calls | 3.338 | 10.627 | 3.18x | 1.10x |
| One guarded write | 3.361 | 12.057 | 3.59x | 1.04x |
| One fresh display string | 55.709 | 68.433 | 1.23x | 1.08x |

Each batch has 256 operations. The factory workload retains two fresh wrappers
per component simultaneously and checks wrapper freshness. The differential
fixture checks exact underlying delegate identity; KNN returns the original
cached handle. The guards workload holds two field-info
wrappers plus live-doc, segment-info and compound wrappers over one RNG.
Reusable Rust non-null directory/metadata/bitset arguments are constructed
outside timing, matching Java's inert probes with null arguments. Display
alternates the delegate text and consumes every result byte.

Every batch checks the ordered factory trace, operation/display counts,
success/error outcomes, complete error messages and RNG continuation. All four
cross-engine checks are mandatory in every repetition. Java wrapper objects
escape through volatile sinks; Rust owned values escape through `black_box`.
Timing includes those sinks, identity checks and digest work. Java object
allocation and Rust value construction are different ownership mechanisms, so
this is a composition workload, not an isolated allocator comparison. The
short smoke runs are excluded from these performance results.

| Selected incremental owned state | Rust bytes | Java bytes |
|---|---:|---:|
| Shallow adapter, excluding the delegate probe | 32 | 24 |
| Extra name heap payload for `CodecProbe` | 0 | 0 |
| One shared RNG allocation graph | 48 | 56 |

The inline name remains included in Rust's shallow adapter row; its allocation
was removed rather than reassigned to an excluded object. These selected rows
sum to 80 bytes for both engines. The original Rust snapshot added ten payload
bytes to that total. Longer Rust names still require a boxed payload while Java
shares the delegate's name; no general long-name memory-parity claim is made.
Rows exclude the delegate graph, returned wrappers, transient strings, allocator
metadata, runtime and RSS. They establish neither total nor peak graph memory.
`name_heap_bytes()` reports only active name payload bytes, with inline enum
state and allocator overhead explicitly excluded from that accessor.

The partial adapter leaves native default factories, SPI/Codec installation,
IndexWriter and per-field routing open. Existing SimpleText term vectors do
not substitute for Java's default Lucene90 term vectors. The complete Java
trace is replayed by `cranky_codec_fixtures.rs` without a JVM.
