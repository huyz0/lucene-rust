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

Timing case sets must agree in every repetition. Result checks observed in
any file are required in all files. A harness with known checks must also pass
--require-check CASE for each one, so simultaneous omission cannot pass.
Legacy harnesses without checks remain supported when no checks are required.
"""
import argparse
import math
import statistics as st
import sys
from pathlib import Path


def load(path):
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[0] in ("case", "name", "bench"):
            continue
        if len(parts) >= 2 and parts[0]:
            try:
                value = float(parts[1])
            except ValueError as error:
                raise ValueError(f"{path}: invalid numeric value for {parts[0]}") from error
            if parts[0] in out:
                raise ValueError(f"{path}: duplicate case {parts[0]}")
            memory_row = parts[0].endswith("_bytes")
            if not math.isfinite(value) or value < 0 or (value == 0 and not memory_row):
                raise ValueError(f"{path}: {parts[0]} must have a finite positive value "
                                 "(zero is allowed for memory bytes)")
            out[parts[0]] = value
        else:
            raise ValueError(f"{path}: malformed timing row")
    return out


def load_checks(path):
    """`#check<TAB>case<TAB>digest<TAB>n` lines: a digest of what the case
    computed (hits and score bits, written bytes), so the two engines can be
    shown to have done the same work before their times are divided."""
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0] == "#check":
            if not parts[1] or not all(parts[2:]):
                raise ValueError(f"{path}: empty result check")
            if parts[1] in out:
                raise ValueError(f"{path}: duplicate result check {parts[1]}")
            out[parts[1]] = "\t".join(parts[2:])
        elif line.startswith("#check"):
            raise ValueError(f"{path}: malformed result check (expected tab separators)")
    return out


def spread(values):
    """Max/min, the factor by which repetitions of one engine disagreed."""
    lo, hi = min(values), max(values)
    return hi / lo if lo > 0 else float("inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("reps", type=int)
    parser.add_argument("--require-check", action="append", default=[], metavar="CASE",
                        help="require this case's result check in every engine/repetition")
    args = parser.parse_args()
    out_dir, reps = args.out_dir, args.reps
    if reps < 1:
        parser.error("reps must be positive")
    rust = [load(out_dir / f"rust.{r}.tsv") for r in range(1, reps + 1)]
    java = [load(out_dir / f"java.{r}.tsv") for r in range(1, reps + 1)]

    cases = sorted(rust[0])
    if not cases:
        print("bench-micro: no cases joined -- the harnesses disagree on case names",
              file=sys.stderr)
        return 1
    for engine, repetitions in [("rust", rust), ("java", java)]:
        for rep, values in enumerate(repetitions, 1):
            if set(values) != set(cases):
                raise ValueError(f"{engine} rep {rep}: timing cases differ from rust rep 1")

    checks = [(load_checks(out_dir / f"rust.{r}.tsv"),
               load_checks(out_dir / f"java.{r}.tsv")) for r in range(1, reps + 1)]
    required = set(args.require_check)
    for rc, jc in checks:
        required.update(rc)
        required.update(jc)
    if required - set(cases):
        raise ValueError(f"result checks without timing cases: {sorted(required - set(cases))}")

    # Results first: a case whose engines disagree is timing different work,
    # and its ratio is not a result. Every repetition is checked.
    mismatched = set()
    for r, (rc, jc) in enumerate(checks, 1):
        for engine, values in [("rust", rc), ("java", jc)]:
            missing = required - set(values)
            if missing:
                raise ValueError(f"{engine} rep {r}: missing result checks {sorted(missing)}")
        for case in sorted(set(rc) & set(jc)):
            if rc[case] != jc[case] and case not in mismatched:
                mismatched.add(case)
                print(f"bench-micro: RESULT MISMATCH in {case} (rep {r}): "
                      f"rust {rc[case]} java {jc[case]}", file=sys.stderr)
    rows, ratios, noises, memory_rows = [], [], [], []
    for case in cases:
        rs = [m[case] for m in rust]
        js = [m[case] for m in java]
        r_med, j_med = st.median(rs), st.median(js)
        if case.endswith("_bytes"):
            memory_rows.append((case, r_med, j_med))
            continue
        ratio = j_med / r_med
        noise = max(spread(rs), spread(js))
        rows.append((case, r_med, j_med, ratio, noise))
        ratios.append(ratio)
        noises.append(noise)

    if memory_rows:
        print()
        print(f"{'memory_bytes':<32} {'rust_bytes':>12} {'java_bytes':>12}")
        for case, rust_bytes, java_bytes in memory_rows:
            print(f"{case:<32} {rust_bytes:12.0f} {java_bytes:12.0f}")

    if not rows:
        print("0 timing cases (memory rows are reported separately).")
        return int(bool(mismatched))
    floor = st.median(noises)

    print()
    print(f"{'case':<10} {'rust_ns':>12} {'java_ns':>12} {'ratio':>8} {'noise':>8}")
    print(f"{'----':<10} {'-------':>12} {'-------':>12} {'-----':>8} {'-----':>8}")
    for case, r, j, ratio, noise in rows:
        # Inside the noise floor in either direction: not a result.
        resolvable = ratio > floor or ratio < 1.0 / floor
        mark = f"{ratio:7.2f}x" if resolvable else f"{ratio:7.2f}~"
        bad = "  MISMATCH" if case in mismatched else ""
        print(f"{case:<10} {r:12.3f} {j:12.3f} {mark} {noise:7.2f}x{bad}")

    med = st.median(ratios)
    unresolved = sum(1 for r in ratios if not (r > floor or r < 1.0 / floor))
    print()
    print(f"{len(rows)} cases   median {med:.2f}x   mean {st.fmean(ratios):.2f}x")
    print(f"noise floor {floor:.2f}x (same engine, {reps} repetitions, median over cases)")
    if unresolved:
        print(f"{unresolved} case(s) marked ~ : difference is inside the noise floor, "
              f"so this run cannot tell them apart.")
    print("ratio > 1 means Rust is faster than Lucene on this case.")
    if mismatched:
        print(f"{len(mismatched)} case(s) computed different results on the two engines "
              f"(MISMATCH): their ratios are not comparable.")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print(f"bench-micro: {error}", file=sys.stderr)
        sys.exit(1)
