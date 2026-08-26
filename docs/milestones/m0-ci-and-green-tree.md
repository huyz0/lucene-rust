# M0 — Green tree, real CI

> **Goal:** every gate that exists today runs automatically on every change, on
> the platforms we claim to support — and `HEAD` passes them.

| | |
|---|---|
| **Effort** | S — days |
| **Depends on** | nothing |
| **Unblocks** | everything |
| **Status** | not started |

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

The five `needless_range_loop` hits in `for_util.rs` need judgement, not the
autofix. That file is the bit-packing kernel — the mirror of Lucene's
generated `ForUtil` — where the index arithmetic *is* the specification and
the loop bounds encode bit-width offsets. Prefer `#[allow(clippy::needless_range_loop)]`
with a one-line comment explaining that the explicit counter mirrors the
generated Java, over a rewrite that obscures the correspondence. Do not
reshape hot decode loops to satisfy a style lint; M1 will be measuring this
file.

### T0.2 — Pin the toolchain

Add `rust-toolchain.toml` at the workspace root:

```toml
[toolchain]
channel = "1.97.1"
components = ["rustfmt", "clippy", "llvm-tools-preview"]
```

`llvm-tools-preview` is required by `cargo llvm-cov`. Pinning turns a compiler
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

A `fixtures` job (x64 only — the fixtures are byte-identical across
architectures, and proving that is not this milestone's job):

1. Set up JDK 25 (Temurin — matches the local toolchain).
2. Run `scripts/gen-fixtures.sh`.
3. `git diff --exit-code fixtures/data`

Any drift between the committed fixtures and what Lucene 10.5.0 actually
produces now fails the build. This is the job that keeps the differential
testing story honest.

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

A gate nobody has seen fail is a gate nobody should trust. Before closing the
milestone, open a throwaway pull request that introduces, one at a time:

1. a formatting violation,
2. a clippy warning,
3. a deleted test (dropping coverage below 95%),
4. a one-byte edit to a file under `fixtures/data/`.

Each must turn CI red on its own. Record the four failing run URLs in the pull
request, then close it without merging.

---

## Acceptance criteria

- [ ] `cargo clippy --workspace --all-targets -- -D warnings` exits 0 on `main`.
- [ ] `rust-toolchain.toml` exists and pins an exact patch version.
- [ ] A pull request shows all jobs green on both `ubuntu-24.04` and
      `ubuntu-24.04-arm`.
- [ ] `scripts/gen-fixtures.sh` regenerates `fixtures/data/` byte-identically
      to what is committed, in CI, with no pre-existing `~/.gradle` cache.
- [ ] All twelve `Verify*.java` reverse verifiers pass unattended in CI.
- [ ] Each of the four negative controls in T0.7 turned CI red, with run URLs
      recorded.
- [ ] `fixtures/README.md` and `AGENTS.md` reference the scripts, and no
      command exists in two places with two spellings.

---

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
