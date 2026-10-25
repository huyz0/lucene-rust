# MockDirectoryWrapper close assertion

The pair invokes actual Lucene 10.5.0 `MockDirectoryWrapper.close` and Rust's
`BaseDirectoryWrapper::close`, with the unreferenced-file assertion enabled
and the normal crash/CheckIndex check disabled. Both engines use the same
Java-produced `merge_policies/commits_index` bytes for two binary cases, plus
the SimpleText flush and merged snapshots. The latest-only case keeps
exactly the newest commit's files; the retained case starts with all three
commits. Every binary iteration checks the exact final filename set and three
live documents. The retained case must lose `segments_1` and `segments_2` during
the second writer pass; the latest-only case must keep its file set.
SimpleText cases check the final filenames and live document counts recorded
by the actual Java close oracle, including updates and compound storage.
The flush snapshot must fail during the first writer's construction because
its parent field is not configured; the merged snapshot must close successfully.
Every iteration checks the expected success or failure and its error message.

For the three successful cases, timing includes both actual writer opens, the first writer's NoDeletionPolicy
rollback, sorted before/after file-set comparison, both reader opens and
document counts, the second writer's default-policy close without committing,
and their temporary resources and locks. SimpleText flush times the expected
first-writer failure and its cleanup. Fresh memory directories, wrapper
construction, final-state validation and backing-store destruction are outside
timing. Java records its delegate close through a non-closing FilterDirectory
so the final files can be inspected. Rust directories release backing resources
when their owners drop; neither engine's final backing-store destruction is
timed. Rust requires quiescence during close instead of Java's synchronized
monitor.

The script alternates engines over three repetitions, pins both to CPUs 0 and
1, enables Java assertions and uses a 128 MiB Java heap. It refuses a load
average above 1.5 before building and before each run. Each repetition reports
the exact final filename/document-count digest for comparison across engines.

```sh
scripts/docker-test.sh bash scripts/bench-no-unreferenced-files-on-close.sh 1000 1500
```

The option adds one retained boolean flag. Transient writer, reader and file-set
allocations are unmeasured. This pair does not establish whole-wrapper owned
memory, peak allocation, GC or RSS parity.

Measured on October 25, 2026, with 1,000 ms warmup and 1,500 ms measurement
per case and repetition. Times are nanoseconds per actual close call, including
its expected failure for the flush case. Ratios are Java median divided by
Rust median. The capped container used Rust 1.97.1, Temurin 21.0.12.1, an Intel
Core i5-13600KF and WSL2 Linux 6.18.54.1. Both engines were pinned to CPUs 0 and 1.
The release example was built before the run, and compiler load was allowed to
settle before the script's load checks passed.

| Case | Rust median ns | Java median ns | Java/Rust | Largest engine repetition spread |
|---|---:|---:|---:|---:|
| Binary latest | 217,917.958 | 657,420.934 | 3.02x | 1.04x |
| Binary retained | 244,533.744 | 501,547.582 | 2.05x | 1.02x |
| SimpleText flush (expected failure) | 100,645.224 | 365,617.029 | 3.63x | 1.22x |
| SimpleText merged | 3,232,209.874 | 8,183,154.262 | 2.53x | 1.07x |

The faithful implementation exceeds timing parity on every measured case,
without a subsequent optimization. Each timing difference exceeds both its
case's observed repetition spread and the report's 1.05x noise floor. This
result covers these snapshots and timing boundaries; transient memory remains
unmeasured as described above. Raw repetitions, in order:

| Case | Rust ns | Java ns |
|---|---|---|
| Binary latest | 217783.355, 218551.932, 217917.958 | 648450.045, 657420.934, 673674.194 |
| Binary retained | 244533.744, 244461.074, 244975.811 | 509100.103, 497645.469, 501547.582 |
| SimpleText flush (expected failure) | 100645.224, 101569.455, 99691.263 | 322586.712, 365617.029, 393320.380 |
| SimpleText merged | 3224207.952, 3256411.747, 3232209.874 | 7838928.045, 8396202.266, 8183154.262 |

All engines and repetitions agreed on these final-state checks:

| Case | Filename/document digest | Files | Live documents | Close outcome |
|---|---:|---:|---:|---|
| Binary latest and retained | 14723208262604097587 | 32 | 3 | success |
| SimpleText flush | 7948185347530394044 | 27 | 148 | expected parent-field failure |
| SimpleText merged | 4052136001700903980 | 6 | 178 | success |

The close oracle was regenerated twice byte-identically. Its SHA-256 is
`6a6e0b3c2c4209dbad8c8a94be5c3ce6f740ab2aff8aaef90c6958cfd7c7bb19`.
The input directories have these Git tree IDs:

- `fixtures/data/merge_policies/commits_index`: `004c5d78d17369ea9fe703215cf744b864ab106f`.
- `fixtures/data/codecs_simpletext/codec/flush/index`: `7aa1aa55c58e308f71ba083f9f0a8a39cd1faf63`.
- `fixtures/data/codecs_simpletext/codec/merged/index`: `e5990942e6c3468a8c4196e6ba918c9f0ad67299`.
