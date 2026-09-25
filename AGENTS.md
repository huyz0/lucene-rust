# AGENTS.md

Orientation for AI agents on **lucene-rust**. A router + invariants list; it
does **not** repeat the detail in [`.agents/skills/`](.agents/skills/),
[`PLAN.md`](PLAN.md), or [`docs/`](docs/). On conflict, the skill/doc wins for
its topic — fix the drift.

## What this is

A Rust port of Apache Lucene, exposed over an FFI (JNI / Panama FFM) boundary
so OpenSearch (JVM) can use it as a native search engine. Pinned Lucene
version: **10.5.0** (matches OpenSearch's `gradle/libs.versions.toml` — see
`docs/parity.md`). The Java source of truth lives at `/home/tuong/work/lucene`;
the OpenSearch checkout at `/home/tuong/work/OpenSearch`.

Phase 1 (foundations: `lucene-util`/`lucene-store`) is in progress. The full
plan — phases, crate layout, verification strategy, effort estimates — is
[`PLAN.md`](PLAN.md). Delivery milestones are [`docs/roadmap.md`](docs/roadmap.md);
the OpenSearch plugin (M2) lives in [`opensearch-plugin/`](opensearch-plugin/),
pinned to **OpenSearch 3.8.0**.

## Invariants (don't break)

1. **Downward-only crate deps.** `util ← store ← codecs ← index ← search ←
   core ← ffi`. Siblings never depend on each other. See the **architecture**
   skill.
2. **Port by on-disk format, not by class hierarchy.** The Java class graph is
   not the target; the byte-level wire format is. See **architecture** and
   **rust-performance**.
3. **Port → benchmark → optimise, one area at a time.** Port each area as
   close to Java as possible first (the correctness baseline), benchmark it
   against Lucene on the same bytes, then optimise until it is not slower —
   and only then start the next area. A port left slower than Java is a bug;
   a redesign made before the faithful port and its benchmark exist is too.
   See **port-workflow**, then **rust-performance** for stage 3.
4. **`unsafe` only in `lucene-util`, `lucene-store`, and `lucene-ffi`.** Every
   other crate is `#![forbid(unsafe_code)]`. See **ffi-safety**.
5. **A Rust panic must never cross the FFI boundary into the JVM.** Every
   exported `lucene-ffi` function wraps in `catch_unwind`. See **ffi-safety**.
6. **No decoder ships without a Java-fixture differential test.** Byte-level
   assumptions get it wrong more often than intuition predicts — verify
   against real Lucene output, not just plausibility. See
   **differential-testing**.
7. **`docs/parity.md` updates in the same commit** as any format that gets
   ported, partially ported, or deliberately deferred. See
   **parity-tracking**.
8. **≥95% line coverage, per file, from day one.** Differential fixture tests
   prove format fidelity; unit tests (inspired by Lucene's own JUnit tests,
   not transliterated from them) prove the decoder's own boundary/error
   handling. See **test-coverage**.
9. **Keep the gates green** — `cargo fmt --check`, `cargo clippy -- -D
   warnings`, `cargo llvm-cov --fail-under-lines 95`, the four checker scripts
   and the rustdoc pass must all pass before a task is done. `scripts/gate.sh`
   is the definition; run it as `scripts/docker-test.sh gate`.
10. **A gate nobody has seen fail is a gate nobody should trust.** When you add
   one, introduce the defect it targets, watch it fail, revert — and write down
   what it *cannot* catch. This sweep found three checks that could not fail
   and one that reported "pass" over a segment it had never opened. See
   [`docs/mechanical-gates.md`](docs/mechanical-gates.md).

## Commands

**Run tests in the container.** `scripts/docker-test.sh` is the official local
way to run anything that builds or tests this workspace:

```
scripts/docker-test.sh gate                     # the whole gate
scripts/docker-test.sh cargo test -p lucene-codecs
scripts/docker-test.sh scripts/verify-write-path.sh
scripts/docker-test.sh bash                     # a shell inside it
```

It caps memory (8 GiB, swap disabled), CPUs, pids and `/tmp`, and pins both
toolchains the gate is defined against (Rust 1.97.1, JDK 21) with the Lucene
10.5.0 jars baked in — so fixture generation and write-path verification are
reproducible and offline. The cap is not bureaucracy: an unbounded local build
has repeatedly exhausted this project's WSL2 VM and killed the running session,
which only a manual `wsl --shutdown` recovers. Inside the container the kernel
kills the build instead, and the host never notices.

`.githooks/pre-commit` (install via `scripts/setup-hooks.sh`) runs the gate
through the container automatically, falling back to a native run when Docker
is unavailable. Run it before calling a task done.

The gate itself is defined once, in [`scripts/gate.sh`](scripts/gate.sh), so
the hook, the container and this table cannot drift apart:

| Step | Command |
|------|---------|
| **The gate** | `scripts/docker-test.sh gate` |
| Format | `cargo fmt --all --check` |
| Lint | `cargo clippy --workspace --all-targets -- -D warnings` (includes the arithmetic gate — see [`docs/arithmetic-gate.md`](docs/arithmetic-gate.md)) |
| Lint for arm64 (catches target-dependent defects) | `cargo clippy --workspace --all-targets --target aarch64-unknown-linux-gnu -- -D warnings` |
| Port invariants clippy cannot express | `python3 scripts/check-port-invariants.py` (FixedBitSet bounds, sentinel call sites, codec-suffix literals, blocktree `try_*`, per-document doc-values, ledger drift — see [`docs/mechanical-gates.md`](docs/mechanical-gates.md)) |
| Rustdoc link lints | `RUSTDOCFLAGS="-D warnings -A rustdoc::private_intra_doc_links" cargo doc --workspace --no-deps --document-private-items` |
| Type-check the out-of-workspace benchmarks | `cargo check --manifest-path benchmarks/rust-runner/Cargo.toml --all-targets` |
| Tests + coverage gate | `cargo llvm-cov --workspace --fail-under-lines 95` |
| Coverage report, per file | `cargo llvm-cov --workspace --summary-only` |
| Regenerate Java fixtures | `scripts/gen-fixtures.sh --only <Gen…>` (a full run rewrites every index with fresh segment ids — see [`fixtures/README.md`](fixtures/README.md)) |
| Check fixtures are still Java-produced | `scripts/gen-fixtures.sh --check` |
| Verify the write path (Lucene reads Rust bytes) | `scripts/verify-write-path.sh` |
| Interop matrix (both engines write, append, merge, delete one index) | `scripts/verify-interop.sh` |
| Crash fuzzing (power loss, kill -9, concurrent power loss; fixed seeds) | `scripts/crash-fuzz.sh` |
| Crash campaign (all crash models in parallel, varied conditions, optional `--load`) | `scripts/crash-storm.sh --duration SECS` |
| Differential op-stream fuzzing vs Java `IndexWriter` (1000 seeds) | `scripts/op-stream-fuzz.sh` |
| Concurrent writer soak (threads + merges, every commit checked exactly; RSS/fds reported) | `cargo run --release -p lucene-search --example concurrent_soak -- --dir DIR --duration SECS` |
| OpenSearch plugin, whole proof (real 3.8.0 node: native-vs-Lucene matrix, force-merge release, SIGKILL, OpenSearch's YAML suites vs a stock node) | `scripts/verify-opensearch.sh --yaml` (needs Docker; see `opensearch-plugin/README.md`) |
| Rust engine (M5), one node: Rust vs OpenSearch's engine op for op, restart/SIGKILL/panic/breaker, YAML suites with every index on the Rust engine | `scripts/verify-opensearch.sh --engine --yaml` (see `docs/opensearch-engine.md`) |
| Rust engine (M5), three nodes: document replication, recovery, failover, Rust<->Java relocation | `scripts/verify-opensearch-cluster.sh` |
| OpenSearch's own `InternalEngineTests` on the Rust engine | `scripts/opensearch-engine-tests.sh` (skips are listed with reasons in `opensearch-plugin/tools/derive_engine_tests.py`) |
| OpenSearch plugin, JVM-side self test only (native path vs Lucene `IndexSearcher`, JNI error paths) | `scripts/opensearch-dist.sh && gradle -p opensearch-plugin check` |
| FFI fuzzing (libFuzzer + ASan; nightly, outside the workspace) | `cd crates/lucene-ffi/fuzz && cargo +nightly fuzz run <target> corpus/<target> seeds/<target>` |

Prefix any of the individual commands with `scripts/docker-test.sh` to run it
capped. **CI does not use the container** — GitHub Actions runners are already
isolated VMs, so containing them again would only cost build time; CI runs the
same commands natively.

The same gate runs in CI on every push and pull request
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)), on Linux x64 and
arm64, plus the two Java jobs (`fixtures`, `write-path`), both pinned to JDK 21 to match the container. The arm64 lint is worth running locally before
touching `lucene-ffi`: `c_char` signedness differs by target, so a whole class of
defect is invisible on x86_64 alone (see the **ffi-safety** skill). It is a
check-only build, so it needs no cross C compiler and no linker. The toolchain is pinned in
[`rust-toolchain.toml`](rust-toolchain.toml) — bump it deliberately, never
implicitly.

Two caveats worth knowing about the coverage gate. `--fail-under-lines`
enforces the **workspace total** (line coverage, currently 97.98%), not
invariant #8's per-file bar. As of `c41-gates-and-record` no file sat below
that bar. Since M2 one does, by construction:
`lucene-ffi/src/jni_bridge.rs` (0%) holds the JNI entry points, which only a
JVM can call. It is marshalling only -- every decision is in `jvm_reader.rs`
(98%) -- and it is exercised by the plugin's `NativeSelfTest` under
`-Xcheck:jni` and by `scripts/verify-opensearch.sh`, not by `cargo llvm-cov`. CI reports the per-file view in its job summary
without failing on it, so the day one drops below 95% it is visible rather than
enforced.

When reading `cargo llvm-cov --summary-only` output, note that it prints three
`Cover` columns — Regions, Functions, then **Lines**. Only the third is what
`--fail-under-lines` and invariant #8 mean. Region coverage is always lower
(97.39% vs 97.98% at the workspace level) and names different files.

**Commits**: `commit-msg` allows only `feat|fix|docs|test|chore|refactor|
perf|build|ci` + optional `(scope)` + lowercase description, and requires a
`Co-Authored-By:` trailer. Single dev — work directly on `main`. See the
**git-workflow** skill.

## Where to look (task → skill)

Skills are the process source of truth; `PLAN.md`/`docs/` are the deep-dives.

| Task | Skill |
|------|-------|
| Crates / module boundaries / where code belongs | `architecture` |
| Starting to port any Java class/format/algorithm, or deciding an area is done | `port-workflow` |
| New decoder for a Lucene file format | `differential-testing` |
| Optimising a ported module (stage 3, after its benchmark) | `rust-performance` |
| Anything in `lucene-ffi`, any `unsafe` block | `ffi-safety` |
| The OpenSearch plugin (`opensearch-plugin/`), its JNI surface | `ffi-safety` + [`opensearch-plugin/README.md`](opensearch-plugin/README.md) |
| Finished a format, need to record it | `parity-tracking` |
| Committing / finishing a unit of work | `git-workflow`, `code-review` |
| Writing tests for a new/changed module | `test-coverage` |
| Arithmetic on a length/count read off disk | `code-review` + [`docs/arithmetic-gate.md`](docs/arithmetic-gate.md) |
| Indexing a `FixedBitSet`, returning a `-1` sentinel, editing `docs/sweep/m2/LEDGER.md` | [`docs/mechanical-gates.md`](docs/mechanical-gates.md) |
| Editing skills | `manage-skills` |

## Workflow

- **Read the matching skill before acting** — it encodes the rule and names
  the gate that enforces it.
- **Port → benchmark → optimise, per area** (the **port-workflow** skill):
  closest-to-Java port that passes its differential tests, then a
  `scripts/bench-micro.sh` pair against Lucene 10.5.0, then optimisation to a
  ratio ≥ 1.0 (or a written-up reason) — before the next area starts.
- **Fixture-first for new decoders**: write the `Gen*.java` generator, run it,
  write the Rust parser against real bytes, write the differential test —
  don't hand-roll expected bytes from reading the Java source alone.
- **Before declaring work done**, after the gate is green, run the Tier 2
  semantic review: spawn the `quality-reviewer` subagent or run
  `/quality-review`.
- **Update `PLAN.md`/`docs/parity.md`/skills in the same change** — drift is a
  bug.
- **Roadmap**: build in phase order, [`PLAN.md`](PLAN.md) §2.
