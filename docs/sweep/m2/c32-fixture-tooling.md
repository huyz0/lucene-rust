# c32-fixture-tooling — hardening the machinery the sweep's evidence rests on

Follow-up batch on the tooling, not on a port. Three items, all raised by
earlier batches: c29's "`gen-fixtures.sh` is a footgun", c28's `/tmp`-full
test-hygiene defect, and the question of whether the two gates the sweep added
are actually being run.

**No Java counterpart exists for anything in this batch.** `scripts/`,
`.githooks/`, `.github/workflows/ci.yml` and
`crates/lucene-util/src/test_support.rs` are this port's own build and test
infrastructure; Lucene's `gradle/` and `TestUtil`/`LuceneTestCase` are a
different design for a different build system and were deliberately not used as
a model. Per the protocol's rule 1, no Java path is claimed.

Files: `scripts/{gen-fixtures.sh,fixture-segment-ids.py,check-parity.py}`,
`fixtures/{README.md,segment-ids.txt}`, `.githooks/pre-commit`,
`.github/workflows/ci.yml`, `crates/lucene-util/{Cargo.toml,src/lib.rs,
src/test_support.rs}`, and the `tempdir()` call sites in
`lucene-{store,codecs,index,search,ffi}` listed in §2.3.

**The headline is finding 8**: both gate scripts the M2 sweep added —
`scripts/check-parity.py` and `scripts/check-arith-allows.py` — have **never
been committed**. `.githooks/pre-commit` *is* tracked and calls
`check-arith-allows.py` under `set -euo pipefail`, so on any fresh clone the
whole pre-commit gate aborts on a missing file. That, not a stale table, is why
the gates "are not being run as reliably as assumed".

---

## 1. `scripts/gen-fixtures.sh` — the destructive-full-run footgun

c29: *"a full run rewrites every index with a fresh random segment id and drops
the `Append*Manifest` keys. It happened once here and was fully reverted (366
files restored, appenders re-run)."*

### What the generators actually do (read before deciding the fix)

| | |
|---|---|
| `Gen*.java` | 44 programs, each writing one fixture directory under the `--out` dir. 25-odd go through a real `IndexWriter`, which stamps a fresh `StringHelper.randomId()` into every index header. |
| `Append*Manifest.java` | 6 programs, **all six** targeting `blocktree_index/manifest.properties`. Each reads the existing manifest, strips its own key prefix, and re-appends — i.e. **idempotent**, and they never regenerate the index. |
| script order | all `Gen*`, then all `Append*`. |

Two consequences settle the design of the fix:

1. Because the appenders run *after* the generators in the same invocation, a
   full run does not by itself lose their keys — it loses them only when the
   `Gen*` programs are run without the appenders, which is what c29 was reduced
   to doing by hand. **`--only` therefore always runs all six appenders
   afterwards**, and because they are idempotent this is byte-neutral for an
   index the invocation did not touch. Verified below.
2. Preserving segment ids across a regeneration is **not possible** without
   patching Lucene: `StringHelper.randomId()` is called inside `IndexWriter`,
   the id is threaded into `.si` and `segments_N` headers and into every codec
   file, and there is no seam to inject it. So "make a full run preserve segment
   ids" is not the honest fix. **Refusing the full run by default is**, plus a
   committed record of the ids so that a regeneration that does happen is
   visible as one readable line rather than 366 binary diffs.

| Finding | Class | What we had | What we have now |
|---|---|---|---|
| 1 | CORRECTNESS | A bare `scripts/gen-fixtures.sh` silently replaced every committed fixture. | It **refuses**, printing what you probably meant. `--all` is the opt-in; `--out DIR` (which cannot clobber) still needs no flag. |
| 2 | MISSING | No way to regenerate one generator; c29 had to revert hundreds of files by hand. | `--only NAME` (repeatable, `Gen` prefix optional), plus `--list`. Always runs the appenders after. |
| 3 | CORRECTNESS | `--check` could not see a dropped `Append*Manifest` key. | Manifest **key-set** comparison, per manifest, naming the dropped keys. |
| 4 | CORRECTNESS | `--check` could not see an index having been regenerated at all. | `fixtures/segment-ids.txt`, re-derived and diffed every `--check`. |
| 5 | PERF | — | `--only` is ~3 s against ~90 s for the full set. |

### 1.1 Finding 1 — a full in-place run now refuses (CORRECTNESS)

`gen-fixtures.sh` with no arguments now exits 2 with an explanation and the
four things you more likely wanted (`--only`, `--list`, `--check`, `--out`).
`--all` is the explicit override, and its help text says to expect
`segment-ids.txt` to change and to put that in the commit message.

This is deliberately a refusal rather than a prompt: the accident's shape is a
script run non-interactively (by an agent, or from a stale shell-history entry),
where a prompt is either invisible or answered by reflex.

Not covered by the refusal, and worth knowing: **several fixture directories are
not tracked by git yet** — `break_iterator/`, `facets_index/`,
`doc_values_updates_index/`, `merge_policy/`, `multi_segment_scoring_index/`,
`fst_empty_key/`, `_2.si`, `_2.manifest.properties`. c29 recovered from its
accident with `git checkout`; for these directories that recovery would not have
worked at all. Recorded, not fixed — committing them is the landing commit's job
(§4).

### 1.2 Finding 2 — `--only` (MISSING), proven by using it

Requirement: *regenerate exactly one fixture, show the rest of the tree is
byte-identical.* Done twice, against a `sha256sum` of all 684 files under
`fixtures/data` taken before and after.

**Proof A — a deterministic generator.** `scripts/gen-fixtures.sh --only
Primitives` (44 generators available; 1 run, then all 6 appenders):

```
684 files compared; 1 difference:
  - facets_index/write.lock        (a stale zero-byte lock, deleted by the run)
segment-ids.txt: unchanged
```

Nothing else moved — including `blocktree_index/manifest.properties`, which the
six appenders rewrote from scratch and reproduced **byte for byte**. That is the
idempotence claim above, checked rather than assumed.

**Proof B — a non-deterministic `IndexWriter` generator.**
`scripts/gen-fixtures.sh --only Norms`:

```
files changed: exactly the 17 files of norms_index/
files unchanged: the other 667
segment-ids.txt diff:
  -norms_index/_0.si       fe2cd58776d3a404838f4a6ff79cdf86
  -norms_index/segments_1  fe2cd58776d3a404838f4a6ff79cdf89
  +norms_index/_0.si       4697161224d7e11ede35044841660a17
  +norms_index/segments_1  4697161224d7e11ede35044841660a1a
```

Under the old script the same intent (regenerating norms) meant regenerating
everything. Restored afterwards with `git checkout -- fixtures/data/norms_index`
plus the saved baseline, and re-verified byte-identical to the pre-experiment
state. **No fixture was regenerated wholesale at any point in this batch.**

### 1.3 Finding 3 — `--check` now catches a dropped manifest key (CORRECTNESS)

The old `--check` compared bytes only for files that two consecutive
generations agree on. `blocktree_index/manifest.properties` is **not** one of
them — measured directly, by generating `GenBlockTree` twice into two scratch
dirs:

```
bt1/blocktree_index/manifest.properties and bt2/... differ: byte 289, line 11
```

So the exact file c29's accident damaged sat in `--check`'s blind spot, and a
tree missing all 229 appended keys passed every byte comparison. `--check` now
compares each manifest's **key set** — stable across runs by construction, and
precisely what an appender contributes.

### 1.4 Finding 4 — `--check` now catches a changed segment id (CORRECTNESS)

Nothing derived from the bytes can distinguish a correctly committed index from
a freshly regenerated one: same generator, same Lucene, only a new
`randomId()`. The only way to catch it is a committed record.

`scripts/fixture-segment-ids.py` parses the id straight out of the
`CodecUtil.writeIndexHeader` prologue (`magic | vint-len + codec name | version
| 16-byte id`) of every `*.si` and `segments_*` — 69 files, all parsed, none
unreadable — and `fixtures/segment-ids.txt` is that output, checked in. Only
those two file kinds are read: both are unambiguously `writeIndexHeader`, and
every index directory has at least one of each, so distinguishing
`writeIndexHeader` from `writeHeader` per codec format would add a table to
maintain for no extra detection.

A write-mode run into `fixtures/data` refreshes the baseline automatically. That
is not a way around the detector — it is the point. `--all` stays possible, and
its damage now shows up in review as a readable one-line-per-index diff instead
of 366 opaque binaries.

The baseline also writes down a coupling that was previously folklore:
`crates/lucene-ffi/src/segment.rs` hardcodes the committed `blocktree_index`
segment id, so a regeneration breaks those tests.

### 1.5 Both detectors, proven on deliberate damage

`fixtures/data` was copied to a scratch tree, damaged in the two specific ways
c29 hit, and checked with `--check --out <copy>` (so the real tree was never
touched):

- **damage 1** — dropped the six appended key families:
  `468 -> 239` keys in `blocktree_index/manifest.properties` (the ledger's
  "239 instead of 402" figure, the manifest having grown since).
- **damage 2** — flipped one byte of `doc_values_index/_0.si`'s id, exactly what
  a regeneration produces.

```
MANIFEST KEYS DROPPED: blocktree_index/manifest.properties -- 229 key(s) ...
    missing: dismax.realLuceneDocScores
    missing: dismax.termA.field
    ...
SEGMENT IDS CHANGED: the committed fixtures are not the indexes segment-ids.txt records.
    -doc_values_index/_0.si b13327c83170075fccfd10d7f050f83b
    +doc_values_index/_0.si b13327c83170075fccfd10d7f050f8c4

  manifests with a wrong key set              : 2
  segment-id baseline lines that disagree     : 2
gen-fixtures: FAILED          (exit 1)
```

Both named, neither merely "files differ".

### 1.6 A pre-existing `--check` failure, owned by an in-flight batch

The **first** `--check` of this batch, against the untouched tree, already
failed:

```
MISMATCH (deterministic file differs from committed): analysis/manifest.properties
MANIFEST KEYS DROPPED: analysis/manifest.properties -- 105 key(s) ...
    missing: edge_ngram_basic.count ...
```

Not this batch's damage: `fixtures/src/GenAnalysis.java` was being edited *while
this batch ran* (mtime moved between two `--check` runs 5 minutes apart, and the
dropped-key count moved 105 -> 108 with it), by the concurrent analysis batch.
Left to them; `--only Analysis` is the one command that fixes it, which is the
flag existing for the reason it exists — and it is what they used. The final
`--check` of this batch is **green**:

```
  deterministic files verified byte-identical : 47
  non-deterministic files (random segment id) : 629
  deterministic mismatches                    : 0
  missing from committed tree                 : 0
  unexplained extras in committed tree        : 0
  manifests with a wrong key set              : 0
  segment-id baseline lines that disagree     : 0
gen-fixtures: ok
```

### Verdict — `scripts/gen-fixtures.sh`

Swept clean. The destructive default is gone, the single-generator path exists
and is proven byte-neutral for everything it does not name, and both of c29's
specific damage modes are now detected by name rather than being invisible.

---

## 2. The per-test temp-directory leak

c28 hit `/tmp` (16 GB tmpfs) at 100% full from ~21 000 leftover
`lucene-*-test-*` directories. At the start of this batch `/tmp` held **5 744**
of them; by the end of the batch's own test runs, 11 339. Root cause: every
crate grew its own `fn tempdir() -> PathBuf` around `std::env::temp_dir()`, and
nothing removed anything.

| Finding | Class | Resolution |
|---|---|---|
| 6 | CORRECTNESS (test hygiene) | One shared RAII guard, `lucene_util::test_support::TempDir`; 30 of 33 call sites migrated. |
| 7 | INTENTIONAL | The guard **keeps** its directory when the thread is panicking. Residual leak: 2 dirs per full run, from the two `#[should_panic]` tests in `segment_writer.rs`. Correct, and the price of the property. |

### 2.1 Where it belongs (the crate-dependency rule)

The `architecture` skill's rule is a strictly downward graph, `util ← store ←
codecs ← index ← search ← core ← ffi`, siblings never depending on each other.
A helper every crate's tests need must therefore live at the **bottom**, in
`lucene-util` — a new sibling `lucene-test-support` crate would be exactly the
sibling edge the rule forbids, five times over.

`crates/lucene-util/src/test_support.rs` is gated
`#[cfg(any(test, feature = "test-support"))]`, and downstream crates enable the
feature on a `[dev-dependencies]` edge. This is not a new pattern: `lucene-search`
and `lucene-codecs` already carry a `test-support` feature for the same reason
(resolver-2 feature unification confines a dev-dependency's features to test and
bench targets, so the module never reaches a production build). Verified:
`cargo build --workspace` compiles without the module.

The skill's "no `util`/`misc`/`common` dumping ground inside a crate" rule is
satisfied — the module is one concept (test scratch directories), not a grab bag.

### 2.2 The guard

`TempDir::new(label)` creates `lucene-{label}-test-{pid}-{seq}-{nanos}` (the
`lucene-*-test-*` shape kept deliberately, so an operator's existing cleanup
glob still finds anything left behind; `{seq}` is an atomic counter, because
`cargo test`'s default parallelism makes same-nanosecond collisions between
threads ordinary and the old helpers had only pid + nanos).

`Drop` removes the directory **unless** `std::thread::panicking()`, or `keep()`
was called, or `LUCENE_KEEP_TEST_DIRS` is set. The panicking branch is the whole
design: a failing test's scratch bytes are the evidence for the failure, and
cleanup that deletes them buys disk at the price of every future
investigation. Asserted directly, by dropping a guard inside a real
`catch_unwind` and checking the file it wrote is still there.

`AsRef<Path>`, `AsRef<OsStr>` and `Deref<Target = Path>` make the guard a
drop-in at the call sites: `FsDirectory::open(&dir)` works through std's blanket
`impl<T: AsRef<OsStr>> From<&T> for PathBuf`, `Path::new(&dir)` works, and
`dir.join(..)` works. Passing `&dir` rather than the guard is what keeps
ownership with the test — handing the guard away would delete the directory out
from under whatever just took the path.

6 unit tests; `test_support.rs` at **97.92%** line coverage (`cargo llvm-cov -p
lucene-util`), above the ≥95%-per-file bar.

### 2.3 Migration — 30 of 33 sites

Migrated (13 `fn tempdir()` helpers rewritten to return the guard, 5 in-module
inline sites, 6 integration-test helpers):

- `lucene-store/src/{directory,index_output}.rs`
- `lucene-codecs/tests/fst_borrowed_seek_fixtures.rs`
- `lucene-index/src/{deletes,field_updates,index_file_deleter,index_writer,
  merge,merge_policy,points_delete,segment_infos,segment_writer,term_delete,
  update_document}.rs`
- `lucene-index/tests/{merge_policy_to_merge_integration,positions_write_path}.rs`
- `lucene-search/src/{directory_reader,multi_segment}.rs`
- `lucene-search/tests/index_writer_{postings,term_vectors,custom_freq_postings}_fixtures.rs`
- `lucene-ffi/src/{directory_reader,points_query,segment,writer}.rs`

Three signature changes fell out, all test-local: `merge.rs`'s `flush` and
`read_merged` and `merge_policy_to_merge_integration.rs`'s helpers took `&str`
paths and now take `&Path`; `positions_write_path.rs`'s `write_index` returns
the guard rather than a `PathBuf`, which is what keeps the directory alive for
the caller.

**Measured**: `cargo test -p lucene-index -p lucene-store -p lucene-search -p
lucene-ffi` leaked **26** directories, against roughly a thousand before. Of the
26: **24** are the three un-migrated files below, and **2** are the deliberate
`#[should_panic]` keeps.

### 2.4 Handoff — the three files owned by other batches

Additive change, so these keep working exactly as they do now.

- [ ] **`crates/lucene-index/src/check_index.rs`** (c30). One
      `fn tempdir() -> PathBuf` at line ~4699, **68 call sites** — the single
      biggest leaker in the workspace (21 of the 26 leaked dirs measured above).
      `use lucene_util::test_support::TempDir;` and
      `fn tempdir() -> TempDir { TempDir::new("check-index") }`; call sites need
      no change unless they pass the path as `&str`.
- [ ] **`crates/lucene-index/src/checksum_verify.rs`** (c30). Same helper shape,
      line ~269, 3 call sites. Label `"checksum-verify"`.
- [ ] **`crates/lucene-codecs/src/fst.rs`** (c31). One *inline* site at line
      ~4506 (`let mut root = std::env::temp_dir(); root.push(...)`), inside the
      mmap-directory test. Replace the five lines with
      `let root = lucene_util::test_support::TempDir::new("fst-mmap");`.
      `crates/lucene-codecs/Cargo.toml` already has the dev-dependency edge this
      batch added.

Also **not** migrated, deliberately: `benchmarks/rust-runner/src/merge_bench.rs`.
It is outside the workspace (`exclude`d in the root `Cargo.toml`), it is not a
test, and it creates one directory per benchmark run rather than one per test —
so it is not part of this defect.

### Verdict — test temp directories

Swept, with a precise three-file handoff. The leak is down from ~1 per test to
2 per full run, and the 2 that remain are the design working as intended.

---

## 3. The two gates

| Finding | Class | Resolution |
|---|---|---|
| 8 | CORRECTNESS | **Both gate scripts are untracked.** `git ls-files scripts/` lists neither `check-parity.py` nor `check-arith-allows.py`; `git log` for both is empty. They have never been committed. |
| 9 | MISSING | `check-parity.py` was in neither `.githooks/pre-commit` nor CI. Added to both. |
| 10 | MISSING | `check-arith-allows.py` was in the hook but **not in CI**. Added to CI. |

### 3.1 Finding 8 — the gates are not in the repository (CORRECTNESS)

```
$ git ls-files scripts/
scripts/bench-compare.sh   scripts/bench-corpus.sh   scripts/bench-env.sh
scripts/bench-micro-report.py   scripts/bench-micro.sh
scripts/gen-fixtures.sh   scripts/lib-lucene-jars.sh
scripts/setup-hooks.sh   scripts/verify-write-path.sh
$ git log --oneline -- scripts/check-parity.py scripts/check-arith-allows.py
(nothing)
```

`.githooks/pre-commit` **is** tracked, and it calls `python3
scripts/check-arith-allows.py` under `set -euo pipefail`. On a fresh clone that
line fails with "No such file or directory" and the **entire pre-commit gate
aborts there** — before `cargo llvm-cov` ever runs. So the gate is not merely
unreliable on other machines; it is broken on them, and broken in a way that
takes the coverage gate down with it.

This, rather than a stale table, is the real answer to the question item 3
asked. c25's stale-table failure is a symptom of the same cause: a script that
only ever exists in one working tree drifts from a `docs/` table that everyone
edits.

**Cannot be fixed by this batch** (no commits). It is the first thing the
landing commit must do — see §4.

### 3.2 Findings 9 and 10 — wiring (MISSING)

`.githooks/pre-commit` now runs `check-parity.py` after `check-arith-allows.py`
and before the coverage gate (fail fast: both are sub-second, `llvm-cov` is
minutes).

`.github/workflows/ci.yml`'s `gate` job now runs both, after `Lint` and before
the coverage step, with a comment recording why. This is the substantive half of
the fix: a hook only runs where someone ran `setup-hooks.sh` and only when
nobody passed `--no-verify`, whereas CI runs on every push and pull request. The
step name is quoted (`"Arithmetic-gate #[allow] rule"`) — unquoted, YAML would
have swallowed everything from the `#` as a comment.

`scripts/setup-hooks.sh` **does** install both correctly: it sets
`core.hooksPath` to `.githooks`, which installs every hook in that directory,
and `git config core.hooksPath` reports `.githooks` in this tree. No change
needed. Its limitation is that it is a manual step, which is what finding 10's
CI wiring compensates for.

### 3.3 Both gates run clean

```
$ python3 scripts/check-arith-allows.py
check-arith-allows: ok (3 module(s) still unaudited)     # c26's, as expected
$ python3 scripts/check-parity.py
check-parity: ok
```

`check-parity.py` immediately earned its place by flagging this batch's own new
module (`lucene-util/src/test_support.rs` had no `docs/parity.md` row). Added to
that script's `EXEMPT` table — the table exists for exactly this, "boundary or
test infrastructure rather than a port of anything in Lucene", and each entry
carries its reason so the list cannot grow silently.

### Verdict — the gates

Wired, in both the hook and CI, and both green. **Open and blocking**: the two
scripts still are not in git.

---

## 4. What the landing commit must do

- [ ] **`git add scripts/check-parity.py scripts/check-arith-allows.py`** —
      finding 8. Without this the tracked pre-commit hook is broken for everyone
      who clones, and the CI steps added in §3.2 fail on the runner.
- [ ] `git add scripts/fixture-segment-ids.py fixtures/segment-ids.txt
      crates/lucene-util/src/test_support.rs` — this batch's new files.
- [ ] `git add` the untracked fixture directories listed in §1.1
      (`break_iterator/`, `facets_index/`, `doc_values_updates_index/`,
      `merge_policy/`, `multi_segment_scoring_index/`, `fst_empty_key/`,
      `_2.si`, `_2.manifest.properties`). Until they are tracked, the `--all`
      refusal is their only protection: `git checkout` cannot restore them.

## Gates

- `cargo fmt --all` — clean. A later `--all --check` reports one diff, in
  `lucene-codecs/src/hnsw.rs` (c31's, edited after this batch formatted).
- `scripts/gen-fixtures.sh --check` — **ok**, all seven counters at zero
  (§1.6). Also exercised: `--check --only X` and `--all --only X` are both
  rejected rather than silently doing something else, and `--only Nope` names
  `--list`.
- `cargo clippy -p lucene-util -p lucene-store -p lucene-index -p lucene-search
  -p lucene-ffi --all-targets -- -D warnings` — **no diagnostic in any file this
  batch touched**. The run does report `arithmetic_side_effects` errors, all of
  them in `lucene-codecs/src/{fst,hnsw,postings_writer,vectors}.rs` — c31's
  in-flight files, exactly its ownership list.
- `scripts/verify-write-path.sh` — **22/22 passed**, run after the migration.
- `cargo test -p lucene-util -p lucene-store -p lucene-index -p lucene-search -p
  lucene-ffi` — **all green**, 0 failed.
- `cargo test --workspace` — **83 suites, 3 971 passed, 0 failed.**
  It was blocked for most of this batch by
  `crates/lucene-codecs/examples/c31_ab_bench.rs` (`no method named 'size'
  found for struct 'OnHeapHnswGraph'`), c31's in-flight file, whose `hnsw.rs`
  was being rewritten alongside. Retried at 60 s intervals rather than edited,
  per the batch instructions, and went green on the eighth attempt once c31
  landed their change. (An intermediate `cargo test --workspace --lib --tests`,
  which skips `examples/`, was already green at 75 suites / 3 970 passed.)
- `python3 scripts/check-arith-allows.py`, `python3 scripts/check-parity.py` —
  both exit zero.
- `cargo llvm-cov -p lucene-util --summary-only` — `test_support.rs` 97.92%
  lines (workspace-wide coverage not runnable while the c31 example is broken).

## Notes

- **No fixture was regenerated wholesale**, at any point. The two `--only`
  proofs touched one deterministic generator (net zero byte change) and one
  `IndexWriter` generator (17 files, restored from git and re-verified
  byte-identical). The damage demonstrations ran against a scratch copy.
- **New files**: `scripts/fixture-segment-ids.py`, `fixtures/segment-ids.txt`,
  `crates/lucene-util/src/test_support.rs`.
- **New feature**: `lucene-util`'s `test-support`, plus a `[dev-dependencies]`
  edge onto it in `lucene-{store,codecs,index,search,ffi}`. No new third-party
  dependency.
- **No public API change** outside the test-only guard.
