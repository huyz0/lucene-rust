# Cargo debug information experiment

The workspace retains its existing development and test profiles. A bounded
`lucene-util` experiment found smaller artifacts with `debug=1`, but changed-file
build times remained within variation. This is a candidate for a representative
integration-heavy measurement, not an established workspace speedup. The fixture
selection improvement in [the companion report](test-framework-fixture-generation.md)
is already adopted.

## Reproduction and scope

```sh
scripts/docker-test.sh python3 scripts/bench-cargo-debug.py \
  --output /work/target/cargo-debug-results
```

The output directory must be new. The script has a 20-minute overall deadline
and individual command deadlines. It builds a disposable copy of `lucene-util`
under the existing target volume and removes that owned temporary directory
afterward. Timeout and interruption checks verify that the owned command group
is killed and its leader reaped before scratch cleanup; a normal nonzero exit
remains unchanged. It does not edit actual sources or existing Cargo artifacts.
The copied workspace retains the root dependencies, lints, toolchain and CPU
configuration. Its lockfile contains only the original reachable package records,
with identical versions, checksums and dependency edges; Cargo uses `--locked
--offline`. Fixtures are linked for reads.

Command-local `CARGO_PROFILE_DEV_DEBUG` and `CARGO_PROFILE_TEST_DEBUG` compare
values 2 and 1. Verbose compiler invocations verify the effective debug levels
for ordinary tests and coverage and compare their other code-generation options.
Isolated test probes verify assertions, overflow checks and a stack frame with
its function, source file and line. No release or final benchmark configuration
changes. Fresh Python timing helpers record wall/user/system time and maximum
individual process RSS, which is not simultaneous process-tree memory.

Each setting gets one empty-target build of `cargo test -p lucene-util --lib
--no-run --locked --offline -vv`, a no-op control and two changed-test-literal
rebuilds. The rebuild order alternates. The cold order is debug 2 first, then
debug 1; filesystem caches are not flushed. These are empty artifact caches,
not cold OS caches. File sizes count each inode once and exclude symlinks.

## Measurements and checks

Measured October 26, 2026, at source commit
`03eca56c471db9b933f5e0c07509f8b3554a145c`, with Rust 1.97.1,
Cargo 1.97.1 and cargo-llvm-cov 0.9.1 in the standard capped container on the
Intel Core i5-13600KF/WSL2 host. Registry sources were already available.
The exact timing and artifact samples are in
[`cargo-debug-info-samples.csv`](cargo-debug-info-samples.csv).

| Measurement | Debug 2 | Debug 1 |
| --- | ---: | ---: |
| Empty-target test build, seconds | 13.164 | 12.316 |
| No-op build, seconds | 0.870 | 0.774 |
| Changed literal, repetition 1, seconds | 2.135 | 2.026 |
| Changed literal, repetition 2, seconds | 1.785 | 1.832 |
| Cold test artifacts, logical bytes | 583004005 | 503647982 |
| Cold test artifacts, allocated bytes | 585863168 | 506531840 |
| Coverage build/test/report, seconds | 14.113 | 13.121 |
| Coverage artifacts, logical bytes | 595488906 | 515860626 |
| Coverage artifacts, allocated bytes | 598339584 | 518721536 |

Logical artifact sizes fell 13.6% for ordinary tests and 13.4% for coverage.
The single cold timing pair favored debug 1 by about 6%, but its fixed order
and absent repetitions prevent a stable speed claim. Changed-file median times
were 1.960 and 1.929 seconds, a difference smaller than their observed spread.
These measurements combine compilation and linking; they do not identify which
stage dominates.

Both effective compiler debug levels matched the requested settings; the other
captured code-generation options matched, including `target-cpu=x86-64-v3`.
Both assertion/overflow and stack-frame checks passed. Each coverage run passed
327 library tests, including two isolated probes. All 149 reported source files
had identical segment hit maps and per-file line/region totals, with 36,701 of
42,407 lines covered. This library-only experiment is a fidelity comparison,
not the workspace's coverage gate. The unchanged production source had already
passed the full gate at 97.65% line coverage.

## Decision and remaining evidence

Keep the defaults until an integration-heavy case establishes the tradeoff.
Reduced debug information sacrifices debugger local/type detail; preserving
source backtraces alone does not establish equivalent interactive debugging.
The trial did not measure the whole workspace, ARM64, failure triage by a human,
the long test execution portion of the gate, or a cache-retention improvement.

The first trial launcher stopped before building because `/usr/bin/time` is
absent from the container; built-in Python accounting replaced that dependency.
A subsequent run stopped at an unsupported positional coverage filter. The
reported complete run uses all library tests through `scripts/coverage.sh`.
Partial runs are excluded from the table.

Preflight before the trial also demonstrated the existing eviction sequence:
it cleared about 20.9 GB of incremental data, found a 200-GB target, removed
34 GB of superseded artifacts, then removed the coverage target while the
remaining 166 GB still exceeded the 160-GB cap. The safeguards remain intact.
Retaining current variants safely needs a successful-build artifact inventory;
newest mtime alone does not prove another hash obsolete. This report neither
changes that policy nor claims fewer future evictions. Grouping integration
tests, splitting crates and adding a build system remain deferred because this
experiment did not measure their costs.
