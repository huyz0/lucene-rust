# Installed asserting postings writer routing

Status: compilation, the Java differential replay and routed writer failure checks pass.
The baseline and optimized timing pairs, native Java readbacks and fresh-child
RSS runs pass their required semantic checks. The remaining native merge gaps and
scope limits are recorded below.

The pair compares Lucene 10.5.0 with Rust using native and installed asserting
postings writers for flush, unsorted merge and index-sorted merge. Java invokes
the actual nested `AssertingPostingsFormat.AssertingFieldsConsumer` by reflection,
retains the native Lucene104 format identity and opens native readers. Rust uses
`AssertingPostingsCodec`. Both keep all other formats native. Native merge acquisition verifies each original
postings group checksum before decoding or opening output consumers. Source
FieldInfos preserve routing attributes, including termless groups on remerge.
CheckIndex, reader open and merge acquisition reject omitted routed groups even
when another group survives; a regression removes all group files and their
segment-file entries and was observed failing before the guard was installed.
The Rust adapter allocates nested arrays of borrowed term references, one
reference per term. Posting, position and payload buffers remain in the original
inputs; it does not eagerly own a second posting tree. Isolated native mapping timing remains
deferred; the six installed writer operations are the measured scope.

Each corpus has 8,192 documents. Merge cases prepare four source segments and
delete IDs divisible by seven before the timed operation. The numeric rank
interleaves source segments for sorted merges. Explicit identical token streams
cover docs-only, frequencies, positions, offsets, variable payloads, repeated
positions, position gaps, sparse fields and a declared empty positions field.
Norms vary with token count; one duplicate overlaps the prior token.

The interval is the entire `commit` for flush or `forceMerge(1)` plus `commit`
for merge. Preparation, deletion commits, complete decoded verification and
writer destruction are outside the interval. Java inverts each token stream in
`addDocument`, before the timer. Rust retains explicit token documents and
inverts them inside the timed flush commit; the payload corpus uses its general
inversion path. These are matching commit operations with different work
partitioning, not isolated postings-encoder or adapter timings. Verification compares stored IDs,
numeric ranks, every term, posting, frequency, position, offset and payload,
term statistics, field options and norms for the six corpus token fields against
an analytical model. Stored IDs, deletion survivors and numeric ranks are checked
separately. The helper `id` field's postings are not re-enumerated by this
benchmark. Both
engines emit identical deterministic checksums and counts. Lucene native readers
also verify each asserted Rust output. The reader API does not expose raw native
impact blocks; these are not compared by this benchmark.

```sh
scripts/docker-test.sh bash scripts/bench-postings-writer-routing.sh 1000 1500
scripts/docker-test.sh bash scripts/bench-postings-writer-routing-memory.sh 8 3
```

Measured on October 26, 2026 in the capped Linux Docker container on an x86_64
Intel Core i5-13600KF under the Microsoft hypervisor. The VM exposes 20 logical
CPUs; the wrapper applies an eight-CPU quota and 16 GiB memory limit. The pair
pins each child to CPUs 0 and 1. Toolchains are Rust 1.97.1 and Temurin
21.0.12.1+1-LTS. Timing uses six alternating Rust/Java repetitions and a fixed
256 MiB JVM heap. The shared reporter calculates median ratios and per-case max/min spreads;
its `~` marker uses the median spread across cases as a common noise floor.
The results below also compare each ratio with its own case's spread. A ratio
within that spread cannot establish a speed difference. The memory runner starts a fresh child for every case/engine/sample
and reports Linux process peak RSS in KiB over a fixed iteration count.
Compilation runs before these fresh children. Timing and RSS measurements precede
the final omitted-group metadata guard; that guard adds field-routing validation
at reader and merge acquisition, whose overhead is not separately measured.

Encoded postings lengths sum `.doc`, `.psm`, `.pos`, `.pay`, `.tim`, `.tip`
and `.tmd` files. Logical payload bytes count only decoded payload content.
Pre-operation writer-accounted bytes are also reported: Rust's
`IndexWriter::ram_bytes_used()` counts buffered documents/custom frequencies;
Java's `IndexWriter.ramBytesUsed()` accounts for the incrementally inverted
postings representation. These values
must not be treated as interchangeable whole-writer retention estimates.

This benchmark uses one postings group. Multi-group native readers retain original
`.doc`, `.pos` and `.pay` contents for integrity checks beside their concatenated
iteration copies. Let P be the sum of those three original file contents only.
For owned inputs, their raw-content retention is approximately 2P instead of P;
the extra retention is approximately P. The `.tim` and `.tip` originals were
already retained before this fix and are not newly doubled. This estimate
excludes metadata, parsed trees and capacity overhead; it is not total reader
memory. Open already held both sets transiently; the fix extends the original
`.doc`/`.pos`/`.pay` retention to reader lifetime. Mapped inputs retain their
original mappings beside concatenated heap buffers; resident pages depend on
the OS. This multi-group retention is not measured here.

Peak RSS includes corpus/source preparation, directory contents, decoded
verification, runtime/JIT, GC and allocator retention. A fresh child per case
separates workloads but cannot isolate the asserting adapter's incremental
footprint, attribute a peak to the timed interval, or prove total memory parity.

## Correctness and adaptations

The installed seam holds every postings-group consumer until the outer operation
closes them. Native source acquisition and document mapping precede opening the
sink. Buffer-backed headers and footers publish during `write`, and norms passed
to the seam are dense. Equal Rust descriptors share a group; Java groups by
format identity. Groups are visited in first-field order. Completion postings
continue through the direct path. Configured and explicit document flushes call
postings before stored-fields callbacks.

This establishes the installed postings writer seam for the covered operations.
Full reader graphs, `IOContext` propagation and panic cleanup remain partial.
It does not establish complete `Codec` parity.

## Results

All timing repetitions, native readbacks and fresh-child RSS children agree on
these decoded checks:

| Operation | Documents | Terms | Postings | Occurrences | Payload bytes | Encoded bytes, both engines |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flush | 8,192 | 39,574 | 152,916 | 243,017 | 114,681 | 624,131 |
| Unsorted merge | 7,021 | 34,110 | 131,060 | 208,288 | 98,292 | 530,993 |
| Sorted merge | 7,021 | 34,110 | 131,060 | 208,288 | 98,292 | 513,792 |

Both native and asserting modes produce the same decoded digest per operation:
`7714994368418590615` for flush, `10842432580584821407` for unsorted merge,
and `15136451210058707053` for sorted merge. Equal encoded lengths do not establish
byte identity. Term, posting, occurrence and payload counters cover the six
corpus token fields. Encoded postings lengths also include the indexed helper
`id` field; they exclude stored fields, norms and doc values.


## Baseline timing

All six Rust/Java repetitions passed the required decoded checks. The shared
reporter's common noise floor is 1.099246x. Ratios above one favor Rust;
`~` applies the reporter's common-floor rule. The spread column is the largest
max/min ratio from either engine for that case.

| Mode | Operation | Rust median ms | Java median ms | Java/Rust | Own spread |
| --- | --- | ---: | ---: | ---: | ---: |
| Native | flush | 12.148428 | 12.741455 | 1.048815~ | 1.085179x |
| Native | merge | 22.839322 | 17.249026 | 0.755234 | 1.096953x |
| Native | sorted merge | 24.531794 | 19.087484 | 0.778071 | 1.134704x |
| Asserting | flush | 16.069238 | 15.078108 | 0.938321~ | 1.057232x |
| Asserting | merge | 25.239579 | 24.994341 | 0.990284~ | 1.101539x |
| Asserting | sorted merge | 27.011595 | 27.797821 | 1.029107~ | 1.150119x |

The native unsorted and sorted merge gaps are visible before installing the
asserting wrapper. They cannot be attributed to its validation stage from these
whole-operation timings. Native merges remain below Java in this baseline;
no profile-based cause is claimed here.

The asserting flush ratio is 0.938321x. Its inverse, 1.065733x, narrowly exceeds
its own 1.057232x spread but remains inside the common noise floor. It warrants
the bounded diagnostic below rather than a robust slowdown claim. Both asserting merge
comparisons remain unresolved by their own spreads and the common floor.

| Operation | Rust asserting/native median | Java asserting/native median |
| --- | ---: | ---: |
| flush | 1.322742x | 1.183390x |
| merge | 1.105093x | 1.449029x |
| sorted merge | 1.101085x | 1.456338x |

These within-engine multipliers describe the installed path's whole-operation
cost, including routing and native input adaptation. They do not isolate the
validator. The baseline implementation has an owned term copy and a fresh postings
box per Rust term, while Java can reuse its term builder and postings enum.
That representation difference is a profiling hypothesis, not an established
cause of these totals. The bounded diagnostic below timestamps the native delegate
and validation stages; it does not attribute their cost to allocations.

<details>
<summary>All six raw timing repetitions</summary>

Each entry is nanoseconds per operation / measured iteration count. Preparation
and verification between iterations are excluded from these nanoseconds.

```text
engine	case	rep1 ns/iterations	rep2	rep3	rep4	rep5	rep6
rust	postings_routing_asserting_flush_commit	16347301.815/92	15472993.928/97	16245015.613/93	15893460.558/95	16358539.750/92	15766194.667/96
rust	postings_routing_asserting_merge_commit	25256475.933/60	25020685.183/60	25469414.678/59	25147699.817/60	25559650.797/59	25222682.533/60
rust	postings_routing_asserting_sorted_merge_commit	27302999.600/55	26720190.509/57	27473768.927/55	26363801.509/57	27552929.436/55	26450008.632/57
rust	postings_routing_default_flush_commit	12005965.563/126	12235672.252/123	11951347.413/126	12598118.500/120	12969348.060/116	12061183.336/125
rust	postings_routing_default_merge_commit	22985443.167/66	22663811.299/67	23388320.615/65	22927502.848/66	22751140.924/66	22131847.368/68
rust	postings_routing_default_sorted_merge_commit	24585775.032/62	24007530.873/63	24591764.274/62	24255312.210/62	24708142.689/61	24477813.371/62
java	postings_routing_asserting_flush_commit	14931541.960/101	15056719.760/100	15225354.566/99	14976740.238/101	15110833.870/100	15099496.810/100
java	postings_routing_asserting_merge_commit	25247421.317/60	24498571.210/62	26986139.786/56	24552368.548/62	24845660.967/61	25143021.350/60
java	postings_routing_asserting_sorted_merge_commit	28924431.038/52	27892881.130/54	31499258.979/48	27702761.291/55	27463501.855/55	27387833.109/55
java	postings_routing_default_flush_commit	12681391.336/119	12680098.134/119	12789336.720/118	12693573.076/119	12803875.085/118	12838815.744/117
java	postings_routing_default_merge_commit	17378741.080/87	17094108.136/88	18718322.358/81	17063917.545/88	17119311.852/88	17658642.506/85
java	postings_routing_default_sorted_merge_commit	19149934.139/79	19079109.076/79	21203136.873/71	19095858.013/79	18711390.481/81	18686054.247/81
```

</details>


## Baseline memory

Each cell below is peak process RSS in KiB from a fresh child running eight
iterations of one case; there are three alternating engine repetitions per case.
All 36 samples passed the same decoded checks as the timing pair. Compilation
and the measuring Python parent's own memory are excluded; child JVM startup,
runtime/JIT, setup, verification and retained allocator/GC state are included.

| Mode | Operation | Engine | Rep 1 KiB | Rep 2 KiB | Rep 3 KiB | Median KiB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Native | flush | rust | 39616 | 40060 | 39428 | 39616 |
| Native | flush | java | 385716 | 382620 | 382628 | 382628 |
| Native | merge | rust | 35784 | 35800 | 35912 | 35800 |
| Native | merge | java | 303828 | 299196 | 298852 | 299196 |
| Native | sorted merge | rust | 34800 | 35260 | 35088 | 35088 |
| Native | sorted merge | java | 300196 | 292744 | 300244 | 300196 |
| Asserting | flush | rust | 40704 | 40728 | 40780 | 40728 |
| Asserting | flush | java | 382552 | 382860 | 385048 | 382860 |
| Asserting | merge | rust | 36304 | 36272 | 36560 | 36304 |
| Asserting | merge | java | 300060 | 299788 | 300920 | 300060 |
| Asserting | sorted merge | rust | 35712 | 35124 | 35500 | 35500 |
| Asserting | sorted merge | java | 302884 | 296812 | 294208 | 296812 |

These whole-process peaks are lower for Rust on this workload. They do not
establish an adapter-only memory ratio or unmeasured total-memory parity. In
particular, Rust's transient inversion and Java's incrementally retained index
buffers occupy different stages of the operation, and the JVM's resident runtime and heap
pages are included in its process peak.

The pre-operation writer-accounted flush bytes are 9,749,437 for Rust and
3,729,908 for Java in both native and asserting modes. Both report zero before
the merge operations because document buffers have already flushed. Those
accounting scopes differ as described above; these are not comparable total
retention estimates. The encoded and logical payload byte counts in every RSS
sample match the preceding result table.

Raw timing TSVs, all RSS samples, per-sample accounting and the complete run log
were preserved under `target/postings-writer-routing-baseline-20261026/` before
the ephemeral container exited. All six raw timing repetitions and all 36 RSS
values are also recorded in this report.


## Bounded validation diagnostic, before optimization

A temporary diagnostic timestamps the native delegate's `write` separately from
the following validation. Rust timestamps both calls directly. Java places a
timestamped native consumer inside the real reflected asserting consumer and
subtracts native-write time from its outer `write` time. Close, inversion and
other commit stages are outside these stage clocks. Instrumentation was removed
and the Rust source restored byte-identical after the diagnostic.

Three fresh processes each run eight asserted flushes, with no separate warmup:
24 samples per engine. Both engines use CPU affinity 0,1; Java uses the fixed
256 MiB heap. Rust's corrected pinned run reuses the existing instrumented
baseline binary without rebuilding or changing source. Its three decoded checks
match the baseline. Java's three checks also match. An initial unpinned Rust
exploration is retained in the raw artifacts but is not the primary result here.
Cold JVM compilation and cache state prevent interpreting the native-stage
numbers as an encoder speed comparison or these samples as steady state.

| Engine | Native write median ms | Validation median ms | Validation min–max ms | Last four per process: validation median ms |
| --- | ---: | ---: | ---: | ---: |
| rust | 6.032493 | 3.218881 | 3.048152–3.688996 | 3.245773 |
| java | 19.475173 | 2.928462 | 2.472895–12.107347 | 2.571575 |

This establishes that Rust validation contributes several milliseconds per
flush on this corpus. It does not establish that allocation is the cause, that
the later samples are steady state, or that the borderline total-flush ratio
comes entirely from validation. The owned-term-copy and postings-box differences
remain representation hypotheses until an independently verified change is
remeasured. Full stage samples and the diagnostic log are preserved under
`target/postings-writer-routing-profile-20261026/`.


## Borrowed-term optimization and final measurement

The optimized validator asks for `WriteTermsEnum::next_bytes`. Its default
implementation preserves existing owned enumerators and their callback order;
the installed native enum returns a borrowed current term. The validator keeps
one previous-term byte buffer per field and reuses its capacity, matching
Java's `BytesRefBuilder` pattern. Existing owned `next()` callers retain their
owned result. Native posting, position and payload buffers stay borrowed as
before. This removes the native owned current-term byte allocation per term;
it does not reuse postings boxes or change the corpus, timers or codec bytes.

The unchanged six-pair timing run and 36-child RSS run passed all required
decoded checks. Every operation's digest, counts, encoded length and logical
payload count agrees with the baseline above. Native Java readers again verified
all three asserted Rust outputs before timing.

The optimized pair's common noise floor is 1.046501x. Ratios above
one favor Rust; `~` means unresolved by that common-floor rule.

| Mode | Operation | Rust median ms | Java median ms | Java/Rust | Own spread |
| --- | --- | ---: | ---: | ---: | ---: |
| Native | flush | 11.955181 | 12.676279 | 1.060317 | 1.050630x |
| Native | merge | 22.128123 | 17.337472 | 0.783504 | 1.055636x |
| Native | sorted merge | 24.117589 | 18.867230 | 0.782302 | 1.038547x |
| Asserting | flush | 15.375714 | 14.927231 | 0.970832~ | 1.143249x |
| Asserting | merge | 24.460918 | 25.155241 | 1.028385~ | 1.042371x |
| Asserting | sorted merge | 26.755051 | 27.624999 | 1.032515~ | 1.037491x |

All three asserting operation ratios are inside both the common floor and their
own repetition spreads. This run establishes no asserting speed advantage or
slowdown against Java. Rust's asserting flush median falls from 16.069238 ms in
the baseline to 15.375714 ms, but its optimized spread is 1.143249x. The runs do
not establish an optimization speedup. The change removes the identified term
allocation while preserving callback and error behavior; postings-box allocation
and the index-owned mapping/buffering differences remain.

The native merge cases remain below Java at 0.783504x and 0.782302x. These cases
bypass the new asserting validator. The diagnostic above does not isolate their
cause, and no encoder or mapping speed claim follows from it. Native mapping,
source decoding and whole-file buffering remain broader measured-work candidates
for a separate profile; this seam remains partial. Flush compares different
inversion timing boundaries as described above.

<details>
<summary>All six optimized timing repetitions</summary>

Each entry is nanoseconds per operation / measured iteration count.

```text
engine	case	rep1 ns/iterations	rep2	rep3	rep4	rep5	rep6
rust	postings_routing_asserting_flush_commit	14159566.472/106	15809560.105/95	15255117.909/99	16187914.183/93	15439244.296/98	15312184.439/98
rust	postings_routing_asserting_merge_commit	24827930.623/61	24504028.452/62	23818706.016/63	24665861.377/61	23855797.714/63	24417806.774/62
rust	postings_routing_asserting_sorted_merge_commit	26200524.966/58	27000344.089/56	26668554.140/57	26841548.679/56	26958574.107/56	26024655.345/58
rust	postings_routing_default_flush_commit	11783607.570/128	12030882.672/125	11949681.095/126	11960680.992/126	12126416.056/124	11542041.477/130
rust	postings_routing_default_merge_commit	21548836.114/70	22442016.791/67	21615363.329/70	22747720.773/66	22213407.309/68	22042838.826/69
rust	postings_routing_default_sorted_merge_commit	24153731.540/63	24464047.387/62	24458284.645/62	24015046.556/63	23556022.031/64	24081445.698/63
java	postings_routing_asserting_flush_commit	14797027.549/102	14912251.079/101	14799750.471/102	14942211.119/101	15229513.162/99	15036381.350/100
java	postings_routing_asserting_merge_commit	24610677.082/61	25173493.800/60	25247943.383/60	25629514.492/59	25136989.100/60	24932235.787/61
java	postings_routing_asserting_sorted_merge_commit	27287392.073/55	28065367.926/54	27542187.291/55	27973698.685/54	27707810.927/55	27256169.964/56
java	postings_routing_default_flush_commit	12585306.467/120	12732808.686/118	12688233.714/119	12664323.748/119	12909763.205/117	12654802.277/119
java	postings_routing_default_merge_commit	16955753.719/89	17293510.862/87	17357798.483/87	17734934.988/85	17329262.621/87	17345681.092/87
java	postings_routing_default_sorted_merge_commit	18733665.025/81	19139051.722/79	18848938.863/80	18970932.700/80	18885522.088/80	18612498.346/81
```

</details>

### Optimized memory

Each sample is a fresh process running eight iterations. All 36 children
passed the same decoded checks. These whole-process peaks retain the scope and
limitations described above; they do not isolate adapter memory or establish
total-memory parity.

| Mode | Operation | Engine | Rep 1 KiB | Rep 2 KiB | Rep 3 KiB | Median KiB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Native | flush | rust | 40276 | 40132 | 39912 | 40132 |
| Native | flush | java | 384128 | 386372 | 380172 | 384128 |
| Native | merge | rust | 35476 | 35944 | 36156 | 35944 |
| Native | merge | java | 299364 | 299224 | 298080 | 299224 |
| Native | sorted merge | rust | 34884 | 34584 | 34792 | 34792 |
| Native | sorted merge | java | 300608 | 294596 | 298600 | 298600 |
| Asserting | flush | rust | 40808 | 40392 | 40436 | 40436 |
| Asserting | flush | java | 383164 | 382616 | 378520 | 382616 |
| Asserting | merge | rust | 35956 | 36248 | 36012 | 36012 |
| Asserting | merge | java | 299812 | 300344 | 302516 | 300344 |
| Asserting | sorted merge | rust | 35332 | 35376 | 35436 | 35376 |
| Asserting | sorted merge | java | 297008 | 297336 | 294816 | 297008 |

Writer-accounted bytes and encoded/logical byte counts are unchanged from the
baseline. Raw timing TSVs, all 36 RSS samples, 108 accounting rows and the complete
run log are preserved under
`target/postings-writer-routing-optimized-20261026/`.
