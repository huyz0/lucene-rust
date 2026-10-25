# Selected test-framework fixture generation

`scripts/gen-test-framework-fixtures.sh --only` compiles the selected generator
sources and lets `javac -sourcepath` resolve their source dependencies. It uses
a fresh temporary classes directory for every invocation. Full `--check`
continues to compile every Java source and execute every discovered generator.
There is no persistent compilation cache or invalidation policy to maintain.

Measured on October 25, 2026, through `scripts/docker-test.sh`, with its existing
8-CPU/16-GiB caps, JDK 21 and Lucene 10.5.0 jars. Three alternating before/after
repetitions ran the complete `--only GenStringMockResourceLoader` command,
including compilation, JVM startup, generation and cleanup. The before script
was the version from commit `8fbda5b1115d55925686ccc6a768d063f3104025`.

| Implementation | Repetition seconds | Median seconds |
|---|---|---:|
| Compile all generators | 2.446, 2.199, 2.248 | 2.248 |
| Compile selected sources | 0.594, 0.611, 0.575 | 0.594 |

The median ratio is **3.79x**. Every run reproduced the committed
`string_mock_resource_loader.tsv` bytes, SHA-256
`8e8a9aef53c5111e3ef4e996c7850c92af1640205cc84623d5e3b7b2c78f5839`.
This measures one small generator with warm filesystem caches. It does not
establish a speedup for every generator, cold-cache behavior, memory use or
full-check execution. The full 45-generator check passed in 20.806 seconds;
no before/after full-check timing comparison was made.

Six regression tests use the production shell script and real Java tools in
a synthetic source tree. They verify selected dependency discovery, exclusion
of an unrelated broken source, repeated selection and generation, complete
full checks, drift rejection and argument handling. Replacing selected compiler
inputs with all sources failed on the unrelated broken generator. Removing
the source path failed to find the selected generator's dependency. Both
deliberate defects were observed and reverted.

A clean compilation of the five real dependency roots (`AssertingDocValues`,
`AssertingPoints`, `AssertingPostings`, `CheapBastard`, `TestFrameworkBloom`)
also produced all four referenced helper class families. The complete Java
fixture check passed without changes to committed fixture data. Runtime
dependencies loaded by reflection remain outside the synthetic test's scope.

The workflow investigation also found that target pruning above 160 GiB can
discard useful Cargo variants: it retains only the newest hash per dependency
basename, then removes the coverage target if the directory remains oversized.
This change leaves the disk safeguards intact. Safely retaining multiple live
variants needs an artifact inventory tied to successful builds; newest mtime
alone cannot prove that another hash is obsolete. No cache-retention improvement
or reduction in Rust gate time is claimed here.

The [Cargo debug-information trial](cargo-debug-info.md) separately measures
one library's artifact size, build times, backtraces and coverage fidelity.
Its changed-file timings do not justify a workspace profile change; the current
profiles and final benchmark settings remain in place.
