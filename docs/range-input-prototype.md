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
norms, points, larger skip structures, updates and other representations need explicit
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
`.fdx`, `.liv`, `.psm` postings metadata and `.dvs` numeric skip data. The
selected `.dvs` is admitted as bounded streamed resident index data, then read
by the existing native skip decoder; it is not a range-native skip cursor.
This corpus does not open positions, norms, points, vectors, compound or update
formats. Physical file/offset/request/response/outcome events
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


Run `42791762122` at `2d091f4e` passed the extended real query test, including
unique accessed-block accounting, wrapper capability, baseline trace and
streaming integrity. It still failed search Clippy: optional numeric state
boxing reduced the join column from 464 to 440 bytes but the variant remained
over the lint's size-difference threshold. The follow-up also boxes the two
numeric cursor variants within the join column. That adds one allocation per
present numeric join leaf, rather than suppressing the lint or inflating every
leaf's inline state. Existing join tests remain part of the full suite.
New source controls cover valid-materialized compound/oversized/nonnumeric
index refusal, directory identity/count/byte admission, immutable operations,
metadata reuse and streaming checksum corruption/truncation. Their hosted
results are pending at this source checkpoint.


Run `42792873608` at `637b5eca` passed the nine store range controls and four
Java-fixture controls, then the new parameterized native test helper failed to
compile because its i32 document loop passed an i32 numeric value to the i64
field constructor. The follow-up uses an explicit lossless conversion. No
native-query, lint, format or matrix pass is inferred for that failed source.


Run `42794138089` at `77443215` passed the store/Java controls and all three
valid-materialized unsupported-profile refusal cases. The native query ran,
but the complete-inventory assertion caught `.psm`, Lucene104 postings metadata
opened by SegmentReader's existing metadata-framing validation. The follow-up
explicitly enumerates `.psm` as bounded resident metadata and requires its
trace. It writes physical evidence before inventory assertions. This failure
is retained; the additional format is not silently ignored.

Component feedback now also includes the gate's excluded benchmark workspace
type check. If that workspace lacks a committed lock, the hosted job resolves
its own graph in capped scratch, retains the bounded standard Actions lock
artifact and exits 78 before benchmark compilation or full qualification.
That separate lock must become source before the locked check can pass.
Ordinary test cells explicitly select libraries/binaries/integration tests,
then doctests separately, avoiding duplicate doctest execution.


Run `42794841013` at `555479c5` passed store/Java controls and native profile
refusals, then its inventory assertion exposed `.dvs` alongside the already
accounted `.psm`. The retained full trace was inspected as a set: 15 actual
files spanning `segments_2`, `.si`, `.fnm`, `.tip`, `.tmd`, `.tim`, `.liv`,
`.fdx`, `.fdm`, `.dvm`, `.dvd`, `.dvs`, `.psm`, `.doc`, `.fdt`. The selected
`.dvs` is a small bounded streamed resident index using existing native parsing
and checksums, not silently omitted or advertised as a remote skip cursor.
Larger skip data still needs a source-aware successor. The follow-up requires
both auxiliary formats and permits no unknown touched family.

That failed source's raw trace records 2,755 cold-query attempts/690,172 received
bytes, 1,566 fault-query attempts/391,496 received bytes with eight failures,
64 integrity attempts/15,888 received bytes, and 70 separate materialization
attempts/16,437 received bytes. These are physical ReadAt backend observations,
not kernel cache/disk/network measurements or a performance win. Native query
expectations/projections and fault checks ran before the inventory assertion,
but the complete test failed and lint/format/benchmark/matrix were not reached.


Run `42796000437` at `924004cc6a7d508f8a3b40a0aa2daaae7d17f96e` passed all
15 selected controls (nine store, four Java-fixture, two native query/profile),
store/codecs all-target Clippy, search library Clippy and pinned formatting.
The complete native control measured 9,224 unique cold-query data-block bytes
against 2,048 configured data-input cache bytes and sampled 1,800 cached bytes
across all inputs after integrity. It then deliberately exited 78 before the
benchmark type check because that excluded workspace needed a committed lock.
The 8,909-byte standard Actions lock artifact has SHA-256
`72ff470ba30cf7446c480e0eff9b1c666c81279c688d851427a4635a45b660c8`.
Its 44 package records are a subset of the already pinned workspace graph plus
only the local `bench-runner` package; no new registry package/version/checksum
is introduced. The exact artifact bytes are committed for subsequent checks.

The coverage cell also produces an independent empty-profile production map
export through `scripts/range-production-coverage.py`. It builds the three
changed libraries with coverage instrumentation and no test target, inventories
every selected compiler rlib and native archive member, retains every mapped
object in one unfiltered LLVM file export and binds their hashes/source/tool
identity. Native stdout is streamed under explicit artifact limits, including
64 MiB for the export. Object hashes are streamed; test-only source cannot
supply this denominator. The ordinary workspace coverage run and its original
95-percent control stay separate. Its native runtime export can later be
projected against production coordinates for per-file/per-crate/changed-line
policy. The producer does not implement a new parser, source filter or floor
and does not declare qualification. Its actual hosted execution is pending.
The same checks cell now includes the arithmetic, structural/parity and
existing dependency/vendored-license checks within the unchanged envelope.


## Repository-gate scope reconciliation

At `af3a192c`, the component/benchmark job and native x64/ARM checks cells passed.
The ordinary/release/coverage cells still need their own outcomes. This seven-
cell matrix did not include all of `scripts/gate.sh`: the Java-reference and
required-jar inventory checks and two gate self-controls were omitted. It must
not be called the complete repository gate.

The corrective provenance cell uses the unchanged cgroup/filesystem/deadline
caps and exact checked-in public pins. It fetches the official Lucene10.5.2 tag's
commit `145d89cd30c38d25e39a0a2232cf00eee208b9cf` into fresh scratch and verifies
its clean checkout, runs `check-java-refs.py` in its declared default audited
scope (without treating a missing tree as a pass), verifies SHA-512 for every
one of the inventory's32 binary jars and runs every module with `--require-jar`.
Inventory membership is Python/ZIP inspection, not Java execution. The cell
also runs the benchmark-report self-controls and the fixture-selection self-
controls with real Temurin21.0.9+10 tools. Its exact207,157,802-byte Linux/x64
archive is SHA-256 pinned. Lucene jars/source are Apache-2.0; the GPL-2.0 with
Classpath-exception JDK is only a transient test tool, never a linked/shipped
Rust dependency. The source and checksum/license provenance are explicit in
`scripts/range-provenance-pins.json`. No local JVM/build, shared cache, image
build or new execution pool is introduced.

The provenance cell is not a full fixture-regeneration, OpenSearch/provider or
benchmark campaign. Committed fixture bytes/generators are unchanged. Native
ARM lint replaces the gate's check-only cross-target lint with actual ARM
execution; both architectures retain their ordinary/release qualification.
The original95-percent workspace/per-file floors and separate90-percent
production aggregate/per-crate/changed-line obligations remain. The new native
projection test additionally compares exact stored field numbers and payload
contents against its written input; a payload's length alone cannot qualify.
These corrections require new-source hosted results; earlier passes are not
silently rebound to a new tree.


At source `af3a192c`, the x64 ordinary cell `129770150377` failed during linking
`geo_points_fixtures` and `test_range_field_queries`: rust-lld terminated with
SIGBUS. It did not complete the ordinary test matrix. The original retained log
has no ending filesystem or cgroup-memory receipt, so disk exhaustion is a
hypothesis consistent with prior repository history, not this run's measured
cause. The next source records those exit receipts while keeping the original
8-GiB/4-GiB/CPU/PID/deadline caps. A measured bound refusal requires bounded
work splitting with the complete declared matrix retained, never a raised cap,
omitted tests or an inferred ordinary-suite pass.


### Observed storage refusal and revised declared bound

ARM ordinary job `129770150391` at `af3a192c` failed while linking
`asserting_norms_fixtures` and `uhighlight_fixtures`; its actual GNU linker
reported `No space left on device` in the original8-GiB filesystem. No ordinary
suite pass follows. This establishes the old storage envelope was insufficient;
it does not retrospectively establish the separate x64 SIGBUS cause.

The corrective source declares20GiB scratch and keeps every CPU/memory/swap/
PID/deadline/stdio/test/coverage limit unchanged. Before allocation, the fresh
GitHub host pulls the same exact image and records actual free physical bytes.
It must have20GiB plus a4-GiB host reserve. `fallocate` physically reserves the
job-owned loop file, and `mkfs.ext4 -E nodiscard` preserves that reservation.
Insufficient capacity refuses before native execution; no global pruning or
unrelated cleanup occurs. The guest still checks its real filesystem size
against the declared upper bound and records actual CFS/memory/swap/PID caps.
Ending filesystem bytes, sampled minimum availability, cgroup memory events
and peak are retained for subsequent failures. The provenance cell runs first
among the serialized qualification cells. These are grounded corrective bounds,
not a relabeling of any failed run or relaxation of a coverage/source gate.


### Admission self-review and scheduling correction

Self-review found SimpleText dispatch happened before the range admission
checks. Its existing search-form builder can construct full in-memory data
representations, so it must not be reached through the declared modern binary
range profile. The corrective source refuses SimpleText before its `.si` or
data opens. A committed real Java SimpleText fixture first opens and prepares
successfully through the materialized API, then must refuse range admission
with only the commit file touched. This does not remove any selected modern
query or change materialized SimpleText behavior. Slice-only doc-value views
are explicitly documented as resident views; only the selected numeric-sort
executor is ported to the typed range view, and other query families need their
own source-aware ports before range capability can be advertised.

At this otherwise-required source correction, the qualification matrix permits
up to3 existing cells concurrently on separate standard GitHub-hosted VMs.
Each physical job/host still runs exactly one heavy workload with unchanged
source/image/cgroup/scratch/deadline/stdio caps and unchanged cell/test counts.
No paid runner label, local pool or shared cache is added. Earlier serialized
runs and every queue/failure/cancellation remain unchanged evidence.


### Measured execution deadlines and prospective timing bound

At `5478d56b`, x64 ordinary qualification passed. ARM ordinary job
`129821930733` reached its actual1,800-second execution deadline while test
suites were still progressing, with no recorded assertion failure and
9,786,380,288 minimum sampled scratch bytes available. Its result remains a
failure, not a partial suite pass. x64 release job `129821930755` also reached
1,800 seconds: the production FFI release build completed in5minutes, but the
search test build had not completed. Minimum sampled scratch availability was
18,472,800,256bytes. Both deadline removals bypassed the shell's exit trap;
final memory counters were not retained and no OOM cause is claimed.

The prospective correction keeps all original Cargo commands, package/feature/
target/profile scopes and test cases unchanged. Tests, release and coverage
receive an explicit5,400-second execution limit and110-minute outer job;
component, checks and provenance retain1,800-second execution limits. CPU2,
RAM4GiB, zero swap, PID256, scratch20GiB, retained stdio2MiB and the maximum of
three separate hosted VMs remain unchanged. This is a bounded timing hypothesis
requiring its own observed qualification, not a claim that either timeout
passed or that the larger bound is sufficient. Existing failed outcomes,
queued/running cells and missing historical counters remain intact.

Before disposing of a verified job-owned live container, the host now reads
its process's unified cgroup membership, verifies the exact container ID in
the canonical host cgroup path, and retains bounded memory events/peak counters.
Unavailable counters are reported explicitly. This covers the diagnostic gap
when deadline removal prevents the guest shell's EXIT trap. An earlier
uncommitted target-partition draft was preserved as local evidence and was
never executed or published; it is superseded by this simpler timing revision.


### Production archive admission diagnostic

Coverage job `129821930804` at `5478d56b` failed after134.819seconds at the
producer's unexpected-nonobject-member refusal. Its old exception omitted the
member identity and no archive inventory was exported; that historical identity
is unknown. The existing exact `lib.rmeta` allowance was already present.
No production denominator, runtime export or coverage pass was produced.

A diagnostic-only source checkpoint retains bounded native archive inventories
and runs the same pinned production-library build on a genuine hosted guest.
It deliberately ends the component cell with exit78 after the probe, before
component qualification or any full matrix. Probe output is not a denominator
or coverage result. Unknown-member admission remains unchanged until actual
producer bytes establish their shape. Subsequent normal producer failures also
retain their archive inventory and include the rejected member identity.


ARM release job `129821930834` at `5478d56b` separately failed after748.789s
with compiler SIGKILL, cgroup `oom=1`, `oom_kill=1` and peak4GiB. This is an
observed memory refusal, unlike the two deadline failures with missing counters.
The prospective heavy-profile revision declares8GiB for tests/release/coverage;
component/checks/provenance remain4GiB. Fresh-host admission records physical
MemTotal/MemAvailable and every enclosing unified cgroup limit/current usage.
Effective available memory must cover the declared guest plus2GiB host reserve,
so heavy profiles require at least10GiB available. Insufficient/unknown capacity
refuses before loop allocation or native work. CPU2, zero swap, PID256, scratch
20GiB, output2MiB, exact test scopes/floors and declared deadlines remain. Guest
checks and receipts bind the actual memory cap; this prospective capacity
change supplies no qualification until observed on fresh standard hosted VMs.
