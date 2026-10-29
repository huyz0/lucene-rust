# CI: how the run is shaped, and why

`.github/workflows/ci.yml` runs the gate (AGENTS.md invariant #9) and the Java
cross-checks on every pull request and every push to `main`. This page records
what each part costs and the decisions that keep that cost down, so a later
change can be weighed against them. The measurements behind them are from the
runs of 2026-10-28/29 (PR #51's run: 66 minutes end to end, ~216 runner
minutes, 10 jobs).

## The one required check: `ci-ok`

Jobs are skipped when a change cannot affect them, so no single job is
guaranteed to run. `ci-ok` runs last, always, and fails if any job failed or was
cancelled; skipped counts as passing. It is the check to require in branch
protection, and the only one: requiring an individual job would block every
change that legitimately skips it.

## What runs for which change

The `changes` job (dorny/paths-filter) sets four flags; `workflow_dispatch`
turns them all on.

| flag | turned on by | jobs |
|---|---|---|
| (always) | every change | `checks` |
| `rust` | `crates/`, `fixtures/`, `scripts/`, `benchmarks/rust-runner/`, the root manifest, toolchain, `.cargo/`, `.config/nextest.toml`, `ci.yml` | `gate (x64)`, `gate (arm64)`, `lucene-search in release`, `write-path`, `fuzz` |
| `plugin` | the `rust` paths except the benchmarks, plus `opensearch-plugin/`, `docker/` | `opensearch plugin`, `opensearch engine`, `opensearch 2.x upgrade` |
| `fixtures` | `fixtures/`, `scripts/`, `crates/lucene-analysis/`, `crates/lucene-test-framework/`, `ci.yml` | `fixtures are Java-produced` |
| `snowball` | `crates/lucene-util/`, `crates/lucene-analysis/`, `scripts/`, the root files | `snowball vocabularies` |

So a change to `docs/`, a `*.md` file or the licence files runs `checks` only:
the parity ledger and its size budget, the port inventory, the licence table,
the arithmetic and port-invariant gates. A change only under
`opensearch-plugin/` runs `checks` and the three OpenSearch jobs.

The filters err towards running: `scripts/` turns on everything that calls a
script, and any crate turns on the plugin jobs, because lucene-ffi links them
all. A path missing from a filter is a job that silently stops running for
that change, which is worse than a wasted run. Add new top-level directories
to the filters when they are created.

## Jobs, and what changed

| job | before | now |
|---|---|---|
| gate (x64) | fmt, clippy, every Python gate, rustdoc, tests under llvm-cov through `cargo test`, then a release build and test of lucene-search: 47 min | fmt, clippy, rustdoc, tests under llvm-cov through `cargo nextest`; the Python gates moved to `checks`, the release tests to their own job |
| gate (arm64) | the same as x64, coverage included: 55 min | clippy, the tests through nextest *without* instrumentation, the doctests. The coverage figure is x64's; arm64 still runs every test |
| lucene-search in release | a step at the end of each gate (~10 min, ~9 of them building) | its own job per arch, beside the gates |
| checks | (inside the gates, twice) | every tree-reading gate, once, on every change; `check-port-invariants.py` went from ~85 s to ~20 s (one rule redid its per-file preprocessing once per declared sentinel) |
| write-path | 40 min; the seeded runs were serial loops | the seeded runs run as parallel slices (`scripts/lib-seed-shards.sh`), crash-fuzz's four runs at once |
| fixtures | 17 min, 15 of them generating every fixture twice, one run after the other | the two generations run at once |
| fuzz | 18 min: `cargo install cargo-fuzz` from source, a cold ASan build of every target, then the six 60 s runs in turn | a prebuilt cargo-fuzz, a cached build on a dated nightly, the six runs three at a time |

## Timeouts

GitHub's default job timeout is six hours, and a hung test held the gate for
over three before anyone noticed. Now:

- every job has `timeout-minutes`, about twice its normal run;
- every test runs under nextest's `slow-timeout` (`.config/nextest.toml`): it
  is reported as slow at 60 s and killed, failing, at 10 minutes. The slowest
  test in the tree takes ~70 s in a debug build on a 4-core machine.

## Caching

The old `actions/cache` entry was keyed on the `Cargo.toml` hashes alone. An
exact hit is never re-saved, so the entry froze at whatever the first job to
finish after a manifest change had built (182 MB, mostly the registry) and
every job then rebuilt nearly everything; four job kinds also shared one key
while building different profiles. `Swatinem/rust-cache` replaces it, one key
per kind of build (`gate`, `release`, `write-path`, `fuzz-<nightly>`), saved
only from `main` so pull requests read one shared entry instead of each filling
the repository's 10 GB cache quota with their own. It caches dependencies, not
the workspace's own crates: a fresh checkout's mtimes make cargo rebuild those
regardless.

## The test runner: nextest

`cargo test` runs the workspace's test binaries one after another, each using
all cores only while it has tests left, so a binary with one slow test leaves
three cores idle. `cargo nextest` schedules every test of every binary across
all cores (the whole debug suite: 9,814 tests, 1,770 s of test time, 449 s of
wall time on 4 cores) and gives each test the timeout above.
`scripts/coverage.sh` takes `COVERAGE_RUNNER=nextest` to use it under
llvm-cov; the coverage figure does not change, since doctests count under
neither runner. Locally, `cargo nextest run --workspace` is the fast way to
run the suite, and `scripts/ci-slowest-tests.py target/nextest/ci/junit.xml`
(after `--profile ci`) lists the slowest tests, as every CI test job's summary
does.

## Not done, and why

- **sccache** across jobs would let unchanged crates skip compiling, but the
  test binaries (the expensive part: lucene-search's unit tests alone take
  ~9 minutes to build in release) are executables, which sccache does not
  cache. Worth measuring once the rest has settled.
- **Committing `Cargo.lock`.** It is gitignored, so every job resolves the
  newest compatible dependency versions itself ("Locking 112 packages"). A
  committed lock file would make runs reproducible and give the cache a
  precise key; it is a project decision, not a CI one.
- **Skipping CI on `main` after a green pull request.** The merge commit was
  already tested, but `main` can move between a PR's run and its merge, so
  `main` still runs; `cancel-in-progress` already stops a superseded run.
