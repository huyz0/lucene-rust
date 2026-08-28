#!/usr/bin/env python3
"""Aggregate `scripts/bench-micro.sh`'s repeated A/B runs into a report that
refuses to claim a difference it cannot resolve.

Why this exists. The first version of this harness ran each engine once and
printed the ratio. Investigating whether explicit SIMD was worth adding turned
up the problem with that: running the *same binary against itself* three times,
`for_decode` varied by a median of 1.21x and a worst case of 1.64x. Every
difference smaller than that had been reported as if it were real, and several
were quoted in `docs/sweep/findings.md` before anyone checked.

So this reports two things instead of one:

  * the ratio, from the median of each side's repetitions
  * the *noise floor*, measured from this very run -- for each case, how much
    each engine varied against itself across repetitions

and marks any case whose ratio sits inside the noise floor as `~`, meaning
indistinguishable. A `~` is not a small result. It is the absence of one.
"""
import statistics as st
import sys
from pathlib import Path


def load(path):
    out = {}
    for line in path.read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass  # header or comment line
    return out


def spread(values):
    """Max/min, the factor by which repetitions of one engine disagreed."""
    lo, hi = min(values), max(values)
    return hi / lo if lo > 0 else float("inf")


def main():
    out_dir, reps = Path(sys.argv[1]), int(sys.argv[2])
    rust = [load(out_dir / f"rust.{r}.tsv") for r in range(1, reps + 1)]
    java = [load(out_dir / f"java.{r}.tsv") for r in range(1, reps + 1)]

    cases = sorted(set(rust[0]) & set(java[0]))
    if not cases:
        print("bench-micro: no cases joined -- the harnesses disagree on case names",
              file=sys.stderr)
        return 1

    rows, ratios, noises = [], [], []
    for case in cases:
        rs = [m[case] for m in rust if case in m]
        js = [m[case] for m in java if case in m]
        if len(rs) < reps or len(js) < reps:
            continue
        r_med, j_med = st.median(rs), st.median(js)
        ratio = j_med / r_med if r_med > 0 else 0.0
        noise = max(spread(rs), spread(js))
        rows.append((case, r_med, j_med, ratio, noise))
        ratios.append(ratio)
        noises.append(noise)

    floor = st.median(noises)

    print()
    print(f"{'case':<10} {'rust_ns':>12} {'java_ns':>12} {'ratio':>8} {'noise':>8}")
    print(f"{'----':<10} {'-------':>12} {'-------':>12} {'-----':>8} {'-----':>8}")
    for case, r, j, ratio, noise in rows:
        # Inside the noise floor in either direction: not a result.
        resolvable = ratio > floor or ratio < 1.0 / floor
        mark = f"{ratio:7.2f}x" if resolvable else f"{ratio:7.2f}~"
        print(f"{case:<10} {r:12.3f} {j:12.3f} {mark} {noise:7.2f}x")

    med = st.median(ratios)
    unresolved = sum(1 for r in ratios if not (r > floor or r < 1.0 / floor))
    print()
    print(f"{len(rows)} cases   median {med:.2f}x   mean {st.fmean(ratios):.2f}x")
    print(f"noise floor {floor:.2f}x (same engine, {reps} repetitions, median over cases)")
    if unresolved:
        print(f"{unresolved} case(s) marked ~ : difference is inside the noise floor, "
              f"so this run cannot tell them apart.")
    print("ratio > 1 means Rust is faster than Lucene on this case.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
