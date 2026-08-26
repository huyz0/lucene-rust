#!/usr/bin/env bash
# Record the machine a benchmark ran on. A ratio without its environment is not
# a reproducible result.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
{
  echo "# Benchmark environment"
  echo
  echo "Recorded automatically by \`scripts/bench-env.sh\`. A published ratio without"
  echo "the machine it was measured on is not reproducible."
  echo
  echo '```'
  echo "date            : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "kernel          : $(uname -sr)"
  echo "cpu             : $(lscpu 2>/dev/null | grep 'Model name' | sed 's/.*: *//')"
  echo "cores/threads   : $(lscpu 2>/dev/null | grep '^Core(s) per socket' | sed 's/.*: *//') cores / $(nproc) threads"
  echo "topology        : hybrid (P-cores + E-cores) -- runs are pinned with taskset"
  echo "memory          : $(free -h | awk 'NR==2{print $2}')"
  echo "rustc           : $(rustc --version)"
  echo "jdk             : $(java -version 2>&1 | head -1)"
  echo "lucene          : 10.5.0"
  echo "governor        : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'not exposed (WSL2)')"
  echo '```'
} > docs/benchmarks/environment.md
echo "wrote docs/benchmarks/environment.md"
