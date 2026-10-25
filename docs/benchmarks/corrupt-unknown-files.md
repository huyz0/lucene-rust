# MockDirectoryWrapper unknown-file selection

The pair invokes Lucene 10.5.0's actual `MockDirectoryWrapper.corruptUnknownFiles`
and Rust's `corrupt_unknown_files_with` over the same Java-produced bytes in
in-memory `ByteBuffersDirectory` instances. Java's overridable `corruptFiles`
and Rust's callback both record the selected names instead of damaging files.

The snapshot retains three existing commits plus an empty fourth commit created
by Java `IndexWriter.deleteAll` with `NoDeletionPolicy` and `NoMergePolicy`.
Every original codec file is referenced only by older commits. A scan that
protects only the latest commit therefore fails the exact victim check.
The small and large cases add 8 and 2,048 deterministic orphan codec and
pending-segments names, plus ignored README and write-lock decoys.

Timing includes both directory listings, every commit and segment-info read,
referenced-file union, filename selection, sorted victim hashing and temporary
scan cleanup. Copying the snapshot, constructing wrappers and constructing the
expected victim digest are outside timing. Every iteration checks the digest
and count; the report requires agreement across engines and repetitions.
Java's callback sorts a `TreeSet`; Rust supplies an already sorted `BTreeSet`.
Both normalization costs are included.

The script alternates engines across three repetitions, pins both to the same
two CPUs, enables Java assertions and uses a 128 MiB Java heap. It rejects load
averages above 1.5 before building and before each run. Run through the capped,
serialized wrapper:

```sh
scripts/docker-test.sh bash scripts/bench-corrupt-unknown-files.sh 1000 1500
```

Measured on October 25, 2026, with 1,000 ms warmup and 1,500 ms measurement
per case and repetition. Times are nanoseconds per complete scan; ratios are
Java median divided by Rust median. The capped container used Rust 1.97.1,
Temurin 21.0.12.1, an Intel Core i5-13600KF, and WSL2 Linux 6.18.54.1;
both engines were pinned to CPUs 0 and 1.

| Case | Rust median ns | Java median ns | Java/Rust | Largest engine repetition spread |
|---|---:|---:|---:|---:|
| 8 orphans | 35,622.534 | 44,418.905 | 1.25x | 1.06x |
| 2,048 orphans | 769,778.194 | 1,034,037.106 | 1.34x | 1.03x |

The faithful implementation exceeds parity on both measured cases, without a
subsequent optimization. The timing difference exceeds the observed repetition
spread. Raw repetitions, in order:

| Case | Rust ns | Java ns |
|---|---|---|
| 8 orphans | 35593.003, 35622.534, 36033.133 | 44418.905, 42803.463, 45223.188 |
| 2,048 orphans | 769778.194, 768958.115, 770104.816 | 1062451.038, 1029919.157, 1034037.106 |

All repetitions agreed on victim digest/count: `13394536499775359349 / 8`
and `17497423089400190733 / 2048`. The differential corpus, regenerated twice
byte-identically, has SHA-256
`6b462a904b306db5f1bf6a00f29efb65da5ef4c3b9f7579eb5913617455401a8`.
Its malformed-metadata cases are outside the successful timing workload.

No retained fields are added by this method. Transient sets and callback
allocations are unmeasured. This pair does not establish peak allocation,
whole-wrapper owned memory, GC, RSS or random-damage throughput parity.

Rust requires a quiescent directory during the scan and callback. Java holds
its directory monitor across the operation; Rust's individual directory calls
retain their existing locking. Concurrent directory mutation is outside this
port's contract and this benchmark.
