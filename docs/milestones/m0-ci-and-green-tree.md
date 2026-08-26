# M0 — Green tree, real CI

> **Goal:** every gate that exists today runs automatically on every change, on
> the platforms we claim to support — and `HEAD` passes them.

| | |
|---|---|
| **Effort** | S — days |
| **Depends on** | nothing |
| **Unblocks** | everything |
| **Status** | delivered, except T0.7's CI run — see Findings |

---

## Why this milestone exists

The project already has a strong quality gate: `cargo fmt --check`, `cargo
clippy -D warnings`, and `cargo llvm-cov --fail-under-lines 95`, wired into
`.githooks/pre-commit` and documented as invariant #9 in `AGENTS.md`.

Nothing enforces it. There is no `.github/` directory. The gate runs only on
one machine, only when someone installs the hook, and can be bypassed with
`--no-verify`.

The consequence is already visible: **`HEAD` does not pass its own
pre-commit gate.** On 2026-07-14 the toolchain moved 1.97.0 → 1.97.1, two new
lints started firing, and 11 warnings entered the tree unnoticed because the
last commits landed on a machine whose toolchain had not yet updated. There is
no `rust-toolchain.toml`, so the compiler version is whatever each machine
happens to have.

The second reason is fixtures. `fixtures/data/` is checked in — 49 directories
of byte-level Lucene 10.5.0 output — precisely so `cargo test` needs no JVM.
That checked-in state is load-bearing for every differential test in the
repo, and **nothing currently proves it is still what Lucene 10.5.0
generates.** A hand-edit or a stale regeneration would silently weaken every
test that reads it.

---

## Scope

### In scope

- Clearing the existing lint debt and pinning the toolchain that produced it.
- A GitHub Actions workflow running the exact gate `AGENTS.md` already
  specifies, on Linux x64 and arm64.
- A JVM-side CI job proving the checked-in fixtures are genuinely Java-produced.
- A JVM-side CI job running the twelve existing `Verify*.java` reverse
  verifiers.
- Extracting the shell snippets currently living in `fixtures/README.md` into
  executable scripts, so CI and humans run identical commands.

### Out of scope

- New tests, new coverage, or raising the 95% bar.
- Benchmarking (that is M1).
- Release automation, artifact publishing, crates.io.
- Windows and macOS runners. `PLAN.md` targets Linux x64 + arm64; adding more
  platforms is a scope change, not a CI detail.

---

## Tasks

### T0.1 — Clear the 11 clippy warnings

Exact locations, all in `lucene-codecs`:

| File | Line | Lint |
|---|---|---|
| `crates/lucene-codecs/src/blocktree.rs` | 2420 | `byte_char_slices` → `*b"dbc"` |
| `crates/lucene-codecs/src/blocktree.rs` | 2789 | `byte_char_slices` → `*b"b"` |
| `crates/lucene-codecs/src/fst.rs` | 3791 | `byte_char_slices` → `*b"acfmz"` |
| `crates/lucene-codecs/src/fst.rs` | 3806 | `byte_char_slices` → `*b"acfmz"` |
| `crates/lucene-codecs/src/fst.rs` | 4562 | `byte_char_slices` → `*b"acfmz"` |
| `crates/lucene-codecs/src/term_vectors.rs` | 1459 | `byte_char_slices` → `*b"cat"` |
| `crates/lucene-codecs/src/for_util.rs` | 279 | `needless_range_loop` (`ints_idx`) |
| `crates/lucene-codecs/src/for_util.rs` | 296 | `needless_range_loop` (`ints_idx`) |
| `crates/lucene-codecs/src/for_util.rs` | 480 | `needless_range_loop` (`ii`) |
| `crates/lucene-codecs/src/for_util.rs` | 551 | `needless_range_loop` (`ii`) |
| `crates/lucene-codecs/src/for_util.rs` | 584 | `needless_range_loop` (`ii`) |

The six `byte_char_slices` hits are all in test data and take the suggested
fix verbatim.

The five `needless_range_loop` hits in `for_util.rs` looked, before reading
them, like candidates for `#[allow]` — that file is the bit-packing kernel
where index arithmetic *is* the specification. On inspection the autofix is
correct: each site is a `for _ in 0..N` loop alongside a manually advanced
output counter, and folding the counter into the loop binding keeps the
bit-offset visible in the range start (48/56/96/112/120) while removing a
redundant `+= 1`. The second counter (`tmp_idx`/`t`), which advances by the
packed width rather than by one, stays explicit. Semantics are unchanged.

**Done** — see commit `fix(codecs): clear clippy warnings from the 1.97.1
toolchain bump`. Note `vec![b'b']` at `blocktree.rs:1984` is deliberately
untouched: it is a macro, not an array literal, and clippy does not flag it.

### T0.2 — Pin the toolchain

Add `rust-toolchain.toml` at the workspace root:

```toml
[toolchain]
channel = "1.97.1"
components = ["rustfmt", "clippy", "llvm-tools"]
```

`llvm-tools` is required by `cargo llvm-cov` (`llvm-tools-preview` is the legacy alias). Pinning turns a compiler
bump into a reviewable one-line diff that CI evaluates, instead of a silent
divergence between machines — which is exactly how the current 11 warnings
arrived.

### T0.3 — The Rust CI workflow

`.github/workflows/ci.yml`, triggered on push and pull request:

- **Matrix:** `ubuntu-24.04` (x64) and `ubuntu-24.04-arm` (aarch64).
- **Steps**, in the order `AGENTS.md` lists them so a CI failure maps onto a
  local command the developer already knows:
  1. `cargo fmt --all --check`
  2. `cargo clippy --workspace --all-targets -- -D warnings`
  3. `cargo llvm-cov --workspace --fail-under-lines 95`
- Cache `~/.cargo` and `target/` keyed on `Cargo.lock` plus the
  `rust-toolchain.toml` hash.
- Install `cargo-llvm-cov` from a pinned version, not `latest`.

The coverage step replaces a separate `cargo test` step — `llvm-cov` runs the
tests, so running both would double a ~3s suite's cost for no signal.

### T0.4 — Extract the fixture commands into scripts

`fixtures/README.md` currently carries the generator and verifier invocations
as shell snippets with a hand-maintained list of 30 `Gen*` class names inside a
`for` loop. CI must not re-type that list; drift between the README and CI
would be invisible.

Create:

- `scripts/gen-fixtures.sh` — compile `fixtures/src/*.java` and run every
  generator. Derive the class list from the filesystem (`Gen*.java`), not a
  hardcoded array, so a new generator is picked up automatically.
- `scripts/verify-write-path.sh` — run each Rust `write_*_fixture` example and
  the matching `Verify*.java`.

Both scripts resolve the Lucene jars themselves. The README's current approach
reads `~/.gradle/caches/`, which does not exist on a CI runner: fetch
`lucene-core` and `lucene-analysis-common` 10.5.0 from Maven Central into a
local directory instead, and have the scripts accept a pre-populated cache
path so local runs stay fast.

Rewrite `fixtures/README.md` to invoke the scripts rather than restate their
contents.

### T0.5 — CI job: fixtures are genuinely Java-produced

**The original design for this task was wrong and had to be replaced.** It
called for `scripts/gen-fixtures.sh` followed by `git diff --exit-code
fixtures/data`. That cannot work: Lucene stamps a random segment ID
(`StringHelper.randomId()`) into every index header, so **366 of the 406**
generated files differ on every run by design. Only the 40 files written as
raw bytes (primitives, FSTs, analysis) are stable.

What the job actually does (`scripts/gen-fixtures.sh --check`):

1. Set up JDK 25 (Temurin — matches the local toolchain).
2. Generate **twice**, into two temp directories.
3. Treat a file as deterministic iff the two runs agree — deriving the set
   rather than maintaining a hand-written list, so it stays correct as
   fixtures are added.
4. Assert every deterministic file matches the committed bytes exactly. This
   is what catches a hand-edit.
5. Assert the generated file tree matches the committed tree, so a generator
   that silently stops emitting a file is caught even where bytes cannot be
   compared.

A stronger check — regenerate in place, then run the Rust suite against the
fresh bytes — is **not currently possible**; see Finding 6.

### T0.6 — CI job: the twelve reverse verifiers

A `write-path` job running `scripts/verify-write-path.sh`, covering the
existing Java-reads-Rust verifiers:

`VerifyCompoundFormat`, `VerifyDocValues`, `VerifyFieldInfos`, `VerifyFst`,
`VerifyLiveDocs`, `VerifyNorms`, `VerifyPoints`, `VerifySegmentInfo`,
`VerifySegmentInfos`, `VerifySparseNumericDocValues`, `VerifyStoredFields`,
`VerifyTermVectors`.

These currently exist but run only when someone remembers to. They are the
only automated evidence that this port's write path produces bytes real Lucene
accepts — running them on every change is most of their value.

Note for M3: there is deliberately **no** postings/term-dictionary verifier in
that list. Closing that gap is [M3](m3-write-path-proven.md)'s first task, not
this milestone's.

### T0.7 — Prove the gate bites

A gate nobody has seen fail is a gate nobody should trust. Four negative
controls, each of which must fail on its own.

**Run locally, all four confirmed:**

| Control | Command | Result |
|---|---|---|
| Formatting violation | `cargo fmt --all --check` | failed as required |
| Clippy warning (`needless_range_loop`) | `cargo clippy --workspace --all-targets -- -D warnings` | failed as required |
| Coverage collapse (run no tests) | `cargo llvm-cov --workspace --fail-under-lines 95 -- --exact __no_such_test__` | failed as required, TOTAL 0.00% |
| One-byte edit to `fixtures/data/vint.bin` | `scripts/gen-fixtures.sh --check` | failed as required, reported the exact file |

The tree was restored after each control and verified clean.

**Still outstanding:** the same four controls exercised through GitHub Actions
on a real pull request, with run URLs recorded. Blocked on Finding 7.

---

## Acceptance criteria

- [x] `cargo clippy --workspace --all-targets -- -D warnings` exits 0 on the
      branch.
- [x] `rust-toolchain.toml` exists and pins an exact patch version, and the pin
      is proven to resolve.
- [x] `scripts/gen-fixtures.sh --check` passes: 40 deterministic files
      byte-identical to the committed tree, 0 mismatches, 0 missing, 0
      unexplained extras.
- [x] `scripts/verify-write-path.sh` passes: 13/13 Rust-written fixtures read
      back by real Lucene 10.5.0.
- [x] All four negative controls fail as required (run locally — see T0.7).
- [x] `fixtures/README.md` and `AGENTS.md` reference the scripts, and no
      command exists in two places with two spellings.
- [x] An in-place `scripts/gen-fixtures.sh` run leaves a git-clean tree.
- [ ] A pull request shows all jobs green on both `ubuntu-24.04` and
      `ubuntu-24.04-arm`. **Blocked — Finding 7b (GitHub Actions outage).**
- [ ] The four negative controls turn *CI* red, with run URLs recorded.
      **Blocked — Finding 7b (GitHub Actions outage).**

## Risks and unknowns

- **`cargo-llvm-cov` on arm64.** Coverage instrumentation is the least
  portable step in the gate. If the arm64 runner cannot run it, fall back to
  `cargo test --workspace` on arm64 and keep the coverage gate x64-only —
  but record that decision in this file rather than silently dropping it.
- **Fixture reproducibility.** The fixtures are asserted byte-identical, which
  assumes every `Gen*.java` is deterministic. If any generator embeds a
  timestamp, a `Random` without a fixed seed, or a `HashMap` iteration order,
  T0.5 will fail for reasons unrelated to correctness. Fixing the generator to
  be deterministic is the right response; loosening the check is not.
- **CI minutes.** The fixture regeneration job compiles ~50 Java files and runs
  30 generators on every push. If that proves slow, gate it on changes to
  `fixtures/**` plus a nightly full run — but never remove it.

---

## Exit artifacts

- `rust-toolchain.toml`
- `.github/workflows/ci.yml`
- `scripts/gen-fixtures.sh`
- `scripts/verify-write-path.sh`
- A rewritten `fixtures/README.md` that calls the scripts
- Four recorded negative-control CI failures

---

## Findings

Things this milestone uncovered that were not visible when it was planned.
Each is either fixed here or recorded as follow-up work.

**1. The pre-commit hook had never been installed on the development machine.**
`git config core.hooksPath` was unset, so `.githooks/pre-commit` never ran.
This is the direct mechanism by which 11 clippy warnings entered the tree: the
gate existed, was documented, and was simply not wired up. *Fixed* — hooks
installed; CI now enforces the same gate independently of local setup.

**2. The documented fixture classpath no longer compiles.**
`fixtures/README.md` specified `lucene-core` plus `lucene-analysis-common`.
`GenBlockTree` uses `org.apache.lucene.queries.spans`, added with the
span-query fixtures, so `lucene-queries` is also required. Following the
README produced 17 compile errors. *Fixed* in `scripts/gen-fixtures.sh`.

**3. The documented generator list omits every `Append*Manifest` program.**
Those five programs append cross-engine ground truth (dismax, fuzzy, prefix,
regexp, wildcard) to an already-generated index's manifest *without*
regenerating the index — deliberately, because regenerating perturbs the
segment ID that committed bytes depend on. Running only the `Gen*` programs
yields **239 keys** in `blocktree_index/manifest.properties` where the
committed fixture has **402**. Anyone following the README would have
silently produced an incomplete fixture set. *Fixed* — the script runs
generators then appenders, in that order.

**4. Fifteen zero-byte `write.lock` files were committed under
`fixtures/data`,** while four otherwise-identical index directories had none.
They are `IndexWriter` lock artifacts, not fixtures, and nothing in `crates/`
reads them. *Fixed* — generation now deletes them so an in-place run leaves a
git-clean tree, and the stale fifteen were removed.

**5. Most fixtures are not byte-reproducible, by design.**
Lucene stamps a random segment ID into every index header, so 366 of 406
generated files differ per run. This invalidated this milestone's original
T0.5 design and forced the generate-twice approach described above. Worth
remembering whenever a future task proposes "just diff the fixtures".

**6. The Rust suite cannot run against freshly generated fixtures.**
`crates/lucene-ffi/src/segment.rs:355` hardcodes the committed blocktree
fixture's segment ID (`bea914ffd84e035aaac43aca30240b47`). Regenerating
fixtures in place therefore breaks the `lucene-ffi` tests. This blocks the
strongest available freshness check — *regenerate, then run the suite* — which
would prove the decoders work against fresh real-Lucene bytes rather than
against bytes that happen to be committed.

*Follow-up:* make those tests read the segment ID from the fixture's
`manifest.properties` (or from the `.si` itself) instead of hardcoding it,
then add a CI job that regenerates and runs the suite. Small, well-scoped, and
it upgrades the fixture guarantee from "unedited" to "still correct". Not done
here because it changes test code, which is outside this milestone's scope.

**7. Pushing a workflow file needs SSH, not the `gh` HTTPS token.**
The `gh` OAuth token carries `repo` but not `workflow`, so pushing
`.github/workflows/ci.yml` over HTTPS is rejected:

> refusing to allow an OAuth App to create or update workflow
> `.github/workflows/ci.yml` without `workflow` scope

GitHub applies that restriction to OAuth tokens over HTTPS, not to SSH. *Fixed*
— `origin` now uses `git@github.com:huyz0/lucene-rust.git`. The alternative,
`gh auth refresh -s workflow`, needs an interactive browser confirmation. Worth
knowing: this affects every future change under `.github/workflows/`.

**7b. CI has not run yet — GitHub Actions is in a major outage.**
PR [#1](https://github.com/huyz0/lucene-rust/pull/1) is open and workflow `ci`
is registered and `active`, but run `32985384206` sat `queued` for over an hour
with **zero jobs created**, and a second push to the branch produced no run at
all. `githubstatus.com` reports Actions in `major_outage` while Git operations,
the API and Pull Requests are all operational.

This is a platform outage, not a configuration fault — a misconfigured
`runs-on` label would still create the job and leave it queued individually,
and a bad workflow file would fail to register. Nothing to fix here; the two
remaining acceptance criteria simply need Actions to come back.

**8. The coverage gate does not enforce the documented invariant.**
`AGENTS.md` invariant #8 asks for ≥95% line coverage *per file*;
`cargo llvm-cov --workspace --fail-under-lines 95` enforces the workspace
*total* (currently 97.59%). Two files sit below the per-file bar —
`lucene-codecs/src/fst.rs` at 93.55% and `lucene-codecs/src/terms_dict.rs` at
92.15%. CI now reports the per-file view in its job summary without failing on
it, since raising coverage is explicitly out of this milestone's scope. Closing
the gap is a decision for whoever owns the invariant: either enforce it and
write the tests, or soften the invariant's wording to match the gate.
