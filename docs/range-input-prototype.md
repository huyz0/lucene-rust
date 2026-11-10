# Fallible range-input prototype

This is an uncompiled source checkpoint based on public lucene-rust revision
`46e280bc2a208544fb36ecd95932e89cd8bc5618`. It is not a qualified engine,
remote DirectoryReader, completed M02.03 task, or benchmark result. Existing
materialized entry points remain available. No dependency, feature, toolchain,
Cargo lock, CI, hsearch contract, or hsearch pin is changed.

## Implemented source

`lucene-store::range_input` defines an immutable/version-bound `ReadAt` source
and fallible `RangeInput`. Clone and slice cursors are independent. Cache and
physical-request budgets are shared across them. Every source destination is
initialized. A failed or short response is never installed in cache; failed
cursor reads do not advance. Caller buffers may be partially changed on error
and cannot be interpreted as complete results. Cache eviction occurs before a
miss allocation. No dereference, conversion or destructor dispatches remote I/O.
The source contract requires version-pinned responses and a caller-enforced
I/O deadline; the generic primitive does not invent provider identity or retry.

The input envelope accepts blocks of 1 byte through 1 MiB, at most 256 blocks
and 16 MiB cache storage, at most 1,000,000 physical attempts and at most 1 GiB
charged response capacity. These are prototype bounds, not calibrated defaults.
`RangeUsage.transferred_bytes` charges the requested response capacity before
I/O, including errors and short replies; it is conservative budget accounting,
not a measurement that those bytes arrived on the wire. Caller transport traces
must separately report actual response bytes and elapsed time. Each source has
its own envelope; the eventual directory owner must also bound their sum.

`MetadataMaterializer` admits at most 128 files, 1 MiB each and 16 MiB total
reserved scratch in a fresh, exclusively owned directory. It streams through a
4 KiB initialized buffer, fully closes the file, then uses the existing mmap
backend with zero heap-read threshold. File creation is exclusive. No fallback
reads a complete file into a Vec. Failed transfers and creation attempts remain
charged and cannot overwrite/reuse a name. The caller retains scratch until all
maps die and owns explicit cleanup. The source input counts physical reads;
materializer usage counts reserved scratch. No durability claim follows from
flush or from memory mapping. Trusted scratch ownership must exclude external
mutation, truncation, namespace replacement and reparse-path substitution.

The codec additions are deliberately additive:

- Modern block-tree `.tim` headers, footers and term blocks use explicit range
  cursors. Existing term navigation, metadata decoding, lowercase-ASCII and LZ4
  decoding execute on those bytes. Materialized `.tim` retains its slice fast
  path. Range block regions are capped at 64 KiB and frame depth at 64. `.tip`
  and `.tmd` remain admitted resident metadata, each capped at 1 MiB, with at
  most 128 field-info entries. Legacy FST range opening refuses explicitly.
- `RangeDocInput` validates modern Lucene104 framing and decodes a bounded
  encoded term window with the existing postings decoder. At most 4,096 docs
  and 64 KiB encoded bytes per term are admitted. Larger/incomplete encodings
  fail; this is not a range-native lazy postings cursor. Singleton postings
  remain pulsed into the term dictionary. Read amplification is observable:
  the window may include bytes after the selected term.
- `numeric_value_range` supports dense, empty and bounded sparse presence,
  constant/table/GCD values and varying-bit-width blocks. Sparse IndexedDISI
  regions are capped at 64 KiB. Packed values use at most eight requested bytes
  through `direct_reader::get_range`. Varying-block jump/header/value reads are
  explicit. Existing numeric decoding entry points remain unchanged.
- `StoredFieldsReader::document_range` uses admitted parsed `.fdm` and resident
  `.fdx` metadata and reads only the selected encoded `.fdt` chunk. Encoded and
  inflated chunks are capped at 1 MiB, with at most 1,024 stored fields. The
  existing header, compression and value parsers execute on the chunk. A reader
  rebound to an empty `.fdt` demonstrates that resident data is not consulted.
  Metadata and the source must belong to the same immutable file version.

## Source inventory and remaining integration

Touched production formats/helpers are `range_input`, existing DataInput,
blocktree term frames and compression dispatch, Lucene104 postings decoding,
DirectReader packed values, numeric doc values/IndexedDISI/varying block
addressing, and stored-fields chunk header/compression/document parsing.
No new unsafe or dependency is introduced. The pre-existing mmap implementation
and Apache-2.0 LICENSE remain unchanged. Java wire formats are unchanged.

The original `Directory::open`, `Input` and `DirectoryReader` still expose and
retain whole-file inputs. Existing term/filter/sort/projection query executors
have not yet been wired to these new codec inputs. This source checkpoint does
not substitute helper comparisons for real selected query execution.

The next port must bind a finite immutable file inventory to `SegmentReader`
and `DirectoryReader`, route the existing selected query executors through
fallible readers and retain the materialized baseline for the same corpus.
It must account for `.si`, `segments_N`, `.fnm`, `.tip`, `.tmd`, `.dvm`, `.fdm`,
`.fdx`, `.tim`, `.doc`, `.dvd`, `.fdt`, suffix groups and per-field dispatch.
`.pos`/`.pay`, norms `.nvm`/`.nvd`, live docs `.liv`, points `.kdm`/`.kdi`/`.kdd`,
skip data `.dvs`, updated doc-value files, and compound `.cfe`/`.cfs` members
must be enumerated whenever the selected corpus/reader opens them. Unused
formats can be explicitly outside that corpus; touched formats cannot be
silently skipped. Compound input requires checked offset/length translation,
not materializing the data archive. Vectors remain unadvertised.

Remaining interfaces include range-native lazy postings and doc-value iterator
integration, resident metadata lifetime/admission ownership, shared aggregate
budgets, actual source identity/context/deadline plumbing and query error
propagation. Existing formats/functions are not presumed safe for arbitrary
corrupt remote metadata merely because these data reads are bounded.

## Authored tests and unrun validation

Source tests exercise independent/nested slices, cross-block reads, cache
exhaustion/eviction, shared concurrent budgets, short/overlong/error responses,
metadata initialization/exclusive creation and retained failed reservations.
Additional comparisons reuse the committed Java-written GenBlockTree,
GenDocValues, GenDocValuesVaryingBpv and GenStoredFields files/manifests. They
assert native term statistics/postings/frequencies, floor-block navigation,
numeric values across block boundaries and stored values, with a 32-byte cache
and explicit source faults. Fixtures and independent expected values are not
rewritten. These are unrun component tests, not remote-query qualification.

No local Cargo build, rustc, JVM, native test, Docker, WSL, workload, benchmark
or coverage command has run. The installed stable rustfmt was used only as a
source editor; the installed pinned formatter failed to start (Windows status
0xC0000135). Pinned formatting and compiler qualification remain unrun.
Compiler feedback may require repairs. Actual selected term/filter/doc-value/stored-field queries over a
working set larger than cache, physical trace/byte/scratch accounting, injected
range errors, both native targets/profiles and required coverage/review remain
pending. Upstream's 95% per-file bar is retained; hsearch's separate task gates
are not waived. No speed, memory, format fidelity or runtime pass is claimed.

## Hosted feedback preparation

The original exact 16-file source checkpoint is preserved as commit
`dd4e7078b365d2107f2fb279429f5c34da425d36`, tree
`04327d0c0d6e92b46958ee31f885f03806b4d79b`, directly parented by the pin above.
Its canonical file manifest digest is
`9b9513229bfbb5e592f679faa8e4f7c48d4569fa9d2b0f06a474ab8532f3b95d`.
Later source formatting and fixed feedback wiring do not rewrite that commit.

The fixed `range-input-feedback` push workflow selects only the development
branch. It checks out the exact push SHA with persisted credentials disabled,
and runs the fixed compiler/component commands through
`scripts/range-feedback.py` on a separate GitHub-hosted Linux x64 runner.
The Rust image is the exact digest already declared in pinned hsearch
`656b790b94b87507717165cf4aa1930a7fb5d9ae:dev/images.json`:
`rust:1.97.1-bookworm@sha256:0e2bcaef56d041a486784e54104a81aebe0da44bd03019bd70bc0401e42e4a97`.
Its build workload has 2 CPU, 4 GiB memory, zero swap, 256 PIDs, a fresh 8 GiB
ext4 scratch filesystem, read-only source/root filesystem, a 1,800-second
execution deadline and 2 MiB combined retained stdio. Guest cgroup receipts and
host filesystem/source identity are checked before compilation. Registry,
target and temporary files share that scratch; no local/shared build pool or
cache is used. Image provisioning and checkout are runner infrastructure, not
claimed as compiler cgroup measurements. The outer job timeout is 40 minutes.
Container cleanup verifies both this job's unique label and source SHA. No
preflight or cleanup prunes unrelated Docker/host resources.

The first fixed attempt performs selected production/test type checking, the
new store and Java-fixture component controls, and pinned formatting. It is
explicitly feedback, not the full both-architecture/dev-release/lint/doctest/
coverage/query qualification matrix. First failures remain evidence. No run is
claimed passed by this preparation.

While this exact branch's PR is a draft, the existing broad workflow defers its
uncapped build jobs and deliberately fails `ci-ok`. Marking the PR ready restores
the original complete workflow. Other branches and main retain their existing
behavior. Draft feedback therefore cannot qualify a merge or a new hsearch pin.

The first published head `d7d60f4f7f28918e0475fd826739e6200dbd879d` ran as
GitHub Actions `42783579364`. It verified the actual digest pull, source SHA,
cgroup caps and Rust 1.97.1, then refused before compilation because upstream
has no committed Cargo.lock. That failure is retained. The fixed feedback job
can resolve a missing lock in a job-local source copy within the same cap,
retain only the bounded public lockfile through the normal Actions artifact
channel, and exit without compiling. The resolved lock must become committed
source before a compiler run is eligible; no floating dependency graph is
claimed qualified. This is lockfile preparation, not a new artifact harness.

Hosted lock preparation `42783964921` at source
`57f98a1579d566600f1689beaa8c2b8f85229da9` resolved the public workspace graph
inside the same verified envelope and deliberately exited 78 before compilation.
Its standard Actions artifact contains 28,918-byte Cargo.lock, SHA-256
`9adde2f8fb75f2f0be1c87495c9316410ffdcfcf948835d5a16fcb884920fa69`.
The unchanged artifact bytes are committed for subsequent locked feedback;
registry packages carry checksums. No dependency manifest or feature changes.
Existing upstream Apache-2.0/MIT/third-party license declarations remain
applicable; exact-graph license/native qualification remains unrun here.

Source-pinned run `42784224009` at
`195b228a96d718fe3fd758195d27308218c7df1a` completed selected store/codecs
all-target type checking, six range-input/metadata tests and four Java-fixture
range tests with zero failures/ignored tests in those selections. It then
failed because the base image lacked the pinned cargo-fmt component. Ordinary
full suites, actual DirectoryReader/query integration, release/ARM/coverage and
independent review were not run. This result does not qualify M02.03.

Subsequent feedback copies the image's exact toolchain into the same capped
scratch filesystem and installs only Rust 1.97.1's rustfmt/clippy components
there. It retains the read-only source/root filesystems and unchanged caps;
no local installation or shared toolchain/cache mutation occurs. Selected
all-target lint and pinned formatting follow the existing component controls.

Run `42784694334` at `1f19e1a862f5a09d644066b75ac618c68c51c6e6` again passed
type checking and all ten selected controls, then pinned Clippy rejected one
new redundant `offset: offset` initializer in numeric range decoding. The
source fix uses Rust field shorthand without changing behavior or suppressing
the lint. Pinned formatting was not reached in that failed attempt.

The stored-field integration now also has `open_range` and an owning
`RangeStoredFieldsReader`. Construction reads the actual range source's bounded
header/footer directly, validates segment identity, suffix, length and resident
metadata, and retains no materialized `.fdt`. The same Java fixture control now
opens through this constructor and checks wrong-ID refusal. This next source
increment still needs fresh exact-source feedback; previous ten-test results
are predecessor evidence. Actual DirectoryReader/query wiring remains pending.

Run `42785049240` at `16bade9b53093f87ae035c9ec783662ffe19ce89` passed selected
type checking, all ten component controls and pinned store/codecs all-target
Clippy with warnings denied. Pinned formatting then rejected two whitespace/
module-order differences; both reported changes are applied without lowering
checks. The direct stored-range constructor is a subsequent source increment
and needs its own fresh feedback.
