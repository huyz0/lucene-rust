# Fallible range-input prototype

This additive prototype retains the exact public lucene-rust base
`46e280bc2a208544fb36ecd95932e89cd8bc5618`. It is not a qualified engine,
completed M02.03 task, provider transport, or benchmark result. Existing
materialized APIs and wire formats remain available. No dependency manifest,
feature, hsearch contract or hsearch pin changes. The branch adds a committed
resolved lockfile and bounded hosted feedback, as recorded below.

## Source and bounds

`ReadAt` binds immutable/version-pinned bytes to one fallible physical attempt.
The backend/caller owns the deadline. `RangeInput` has independent clone/slice
cursors, a shared finite FIFO cache and charged request budget. Failed/short
replies never enter cache or advance the cursor. Caller buffers may change
partially on error and cannot be treated as complete results. No dereference,
conversion or destructor performs remote I/O. Requested capacity is charged
before I/O; `RangeUsage.transferred_bytes` is not actual received wire bytes.
Blocks are 1 byte through 1 MiB, cache at most 256 blocks/16 MiB, attempts at
most 1,000,000 and charged capacity at most 1 GiB per source.

`RangeDirectory` admits at most 128 files/128 MiB total immutable source bytes.
It refuses whole-file materialization of `.tim`, `.doc`, `.dvd`, `.fdt`, `.cfs`.
Other metadata is streamed through an initialized 4 KiB buffer into exclusive
files in fresh owned scratch, then mapped with the existing zero-threshold
mmap backend. Metadata admission is at most 128 files, 1 MiB each and 16 MiB
reserved scratch. Failures remain charged. The caller owns cleanup after all
maps die and must exclude external mutation, truncation and path substitution.
Directory pointer wrappers and `EstimatedWrites` preserve the range capability.
Per-source envelopes still require an owner-selected aggregate request budget.
The test supplies one shared aggregate trace envelope; this is not a provider
context/deadline implementation.

## Native selected execution

`DirectoryReader`/`SegmentReader` and the existing `IndexSearcher` route modern
term search, Boolean filtering, numeric sorting and stored projection through
range-capable readers. `.tim` regions are at most 64 KiB with 64 term frames;
`.tip`/`.tmd` are admitted resident metadata with at most 128 fields.
`LazyDocsCursor` owns the range postings decoded by the existing decoder, with
at most 4,096 documents and 64 KiB encoded bytes per selected term. This is a
bounded decoded-term cursor, not a streaming postings-block implementation.
Range impacts are conservative. Position-dependent non-singleton access fails.
Numeric values use bounded sparse presence (64 KiB), constant/table/GCD and
varying-width decoding, with at most eight requested packed-value bytes.
Stored fields reuse native metadata/compression/document visitors; selected
encoded/inflated chunks are at most 1 MiB and 1,024 fields. Streaming CRC
integrity checking uses a bounded buffer rather than materializing data files.
Materialized readers retain their existing paths. Legacy slice-only doc-value
views return no resident view for range sources; typed source views provide
numeric data, and their absence must not be read as an absent column.

The selected range profile refuses compound/vector segments, more than 4,096
documents, multiple/foreign postings groups, nonnumeric or multivalued doc
values and update generations. Legacy/vector range APIs are unadvertised.
These refusals delimit this prototype, not universal native compatibility.
Compound support still needs checked member offset/length translation; positions,
norms, points, skip structures, updates and other representations need explicit
source-aware interfaces and their own corpus/negative controls. No fallback
may silently materialize a data archive. Arbitrary corrupt remote metadata is
not presumed safe merely because the selected data reads are bounded.

## Observed feedback and remaining gates

Hosted run [42785492505](https://github.com/huyz0/lucene-rust/actions/runs/42785492505)
at `3464bfa2898d342ed5a87e1a2ab923d2ba1ec59b` passed selected store/codecs
all-target type checking, six store tests, four committed Java-fixture range
tests, store/codecs all-target Clippy and pinned formatting. The fixtures and
independent expected values were not rewritten.

Run [42790441710](https://github.com/huyz0/lucene-rust/actions/runs/42790441710)
at `4ab6e6244970e45cb06d154f24bcef732a7f1a0f` additionally passed search library
type checking and the real native DirectoryReader/IndexSearcher test. It failed
later on search Clippy's large-enum warning caused by inline range numeric
state. Its original failure and physical trace remain evidence. The follow-up
boxes only that optional range state without suppressing the lint.

The native-written corpus has 384 documents, a real deletion, full postings
block plus tail, floor terms, sparse numeric values and stored payloads. The
same sorted IDs/values/totals and projections are checked against independent
expectations and a separately streamed full-materialization baseline. Data
families `.tim`, `.doc`, `.dvd`, `.fdt` receive error and short-response faults
after reader preparation; successful/empty results are rejected. Actual touched
metadata includes `segments_N`, `.si`, `.fnm`, `.tip`, `.tmd`, `.dvm`, `.fdm`,
`.fdx` and `.liv`. This corpus does not open positions, norms, points, vectors,
compound or update formats. Physical file/offset/request/response/outcome events
are retained in a finite TSV. Tiny 512-byte per-source caches deliberately
exercise eviction; this is not a speed or transfer-saving claim.

The follow-up test adds unique accessed-block accounting above aggregate data
cache capacity, sampled cache occupancy, full baseline trace events, wrapper
capability and native streaming integrity checks. These additions require a
new hosted result. Full applicable x64/ARM, dev/release, ordinary suites,
production build, lint, doctest, coverage and independent final-tree review
remain pending. Upstream's 95% per-file floor and hsearch's separate gates are
retained. No local Cargo/native build, Docker, WSL or JVM workload has run.
Installed stable rustfmt is used only as a source editor; hosted pinned
formatting supplies qualification. No release, merge or new hsearch pin follows
from partial feedback. Earlier checkpoints below preserve their original
observations and limitations.

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

## Native reader/query integration increment

The source now integrates RangeDirectory with existing SegmentReader and
DirectoryReader construction. Existing native DocInput/LazyDocsCursor query
interfaces use bounded decoded postings for range-backed terms; resident
cursors keep their original behavior. This range prototype eagerly decodes a
bounded term window and conservatively disables impacts pruning; it does not
claim lazy remote postings performance. Numeric sort comparators receive typed
resident/range doc-value inputs. Legacy byte-slice views remain materialized-
only, and do not expose empty range data as a real column. Stored projection
uses the original native visitor and range chunk decoder. Range integrity
streams CRC checks without whole-file heap buffers. Materialized APIs retain
their original path.

The immutable RangeDirectory admits at most 128 files/128 MiB logical source,
uses the existing charged 16 MiB metadata materializer, and refuses data-file
materialization through Directory.open. Selected range segments are bounded
non-compound, single-modern-postings-group, non-vector segments with at most
4,096 documents and single-valued numeric doc values without update generations.
Other representations refuse explicitly at admission. These prototype limits
are not measured production defaults or universal codec-support claims.

The new native-written 384-document corpus has full-block plus tail term
postings, floor-block unique terms, an actual deletion/live-docs file, sparse
numeric ranks and 512-byte stored payloads. Its real DirectoryReader and
IndexSearcher term/filter/numeric-sort/stored-projection queries check independent
expected IDs/ranks/values. Range caches are 512 bytes per file and the accessed
working set must exceed capacity. Errors and short responses are injected after
query preparation for each .tim/.doc/.dvd/.fdt family and must propagate as
query/projection failures. A separate full streaming disk baseline uses mmap,
with separately counted actual transfer and scratch bytes. The finite complete
file/offset/request/response/outcome trace is retained through a standard Actions
artifact; its capacity/byte exhaustion fails rather than truncating success.
This source increment and test are not yet observed passes.

Touched reader paths are commit segments_N, .si, .fnm, .tip/.tmd/.tim, .doc,
.dvm/.dvd, .fdm/.fdx/.fdt and .liv. The selected corpus has no vectors, compound,
points, term vectors, norms or additional postings/doc-value suffix groups.
Their absence must be observed from its real inventory, not assumed. The
upstream patch plan retains those additional representations and iterators;
no current-main compatibility or runtime/provider/fleet claim follows.


## Bounded qualification expansion

After component feedback passes, the same workflow serializes seven fixed
fresh-VM qualification cells: workspace lint/rustdoc on native x64 and ARM,
workspace dev suites/doctests plus production FFI build on each architecture,
search release suites/doctests plus production FFI release build on each, and
workspace x64 coverage. All use the same exact image/source and unchanged
2-CPU/4-GiB/zero-swap/256-PID/8-GiB/1,800-second/2-MiB envelope. A bound refusal
is retained evidence and requires splitting work, not raising or omitting a
gate. There is no shared cache. Only one qualification cell runs at a time,
after the component cell, with fail-fast disabled to preserve independent
failures. Coverage installs exact cargo-llvm-cov 0.8.7 with its published locked
graph into job scratch and retains the raw LLVM file export (at most 64 MiB)
for physical-line projection. Native summary totals remain diagnostics; they
do not replace physical per-file/per-crate/changed-line accounting. The
existing workspace 95% gate remains unchanged. This wiring is not a report
that those cells ran or passed.
