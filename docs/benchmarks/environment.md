# Benchmark environment

Recorded automatically by `scripts/bench-env.sh`. A published ratio without
the machine it was measured on is not reproducible.

```
date            : 2026-08-27T06:48:42Z
kernel          : Linux 6.18.33.2-microsoft-standard-WSL2
cpu             : 13th Gen Intel(R) Core(TM) i5-13600KF
cores/threads   : 10 cores / 20 threads
topology        : hybrid (P-cores + E-cores) -- runs are pinned with taskset
memory          : 31Gi
rustc           : rustc 1.97.1 (8bab26f4f 2026-07-14)
jdk             : openjdk version "25.0.4" 2026-07-21 LTS
lucene          : 10.5.0
governor        : not exposed (WSL2)
```
