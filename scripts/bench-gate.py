#!/usr/bin/env python3
"""The nightly performance gate (M6 T6.2): the M1 query mix, both engines on
the same index, failing on

  * a recall mismatch (the engines disagree on hits: timings are meaningless),
  * a query slower than Lucene (Rust/Java qps ratio below 1.0), or
  * a query regressed from the recorded baseline ratio by more than --regress.

A ratio, not a qps number, is what is gated: both engines run in the same job
on the same machine, one after the other, so the runner's own speed cancels
out. A query that fails is measured again, both engines, up to --retries more
times, and fails the gate only if its best ratio still fails -- a single noisy
pass (runs move by about +-9% on a quiet machine) does not.

    scripts/bench-gate.py --index benchmarks/.corpus/merged \\
        --baseline benchmarks/baseline/local/merged.tsv [--history FILE] \\
        [--retries 2] [--regress 0.10] [--update-baseline] \\
        [--negative-control N] [--measure-ms MS] [--warmup-ms MS]

--negative-control N runs the Rust side N times per timed query (a known
N-fold regression, `BENCH_NEGATIVE_CONTROL` in the runner) and inverts the
exit status: 0 when the gate caught it, 1 when it passed a build it should
have failed. A gate nobody has seen fail is not trusted (AGENTS.md #10).

Exit 0 pass, 1 fail, 2 usage or harness error. Standard library only.
"""
import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_tsv(path):
    with open(path) as f:
        return {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}


def compare(args, queries, env):
    """One bench-compare pass; its joined per-query rows."""
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as t:
        out = t.name
    cmd = [os.path.join(ROOT, "scripts", "bench-compare.sh"), "--index", args.index,
           "--queries", queries, "--tsv", out,
           "--warmup-ms", str(args.warmup_ms), "--measure-ms", str(args.measure_ms)]
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        print(f"bench-gate: bench-compare exited {r.returncode}", file=sys.stderr)
        sys.exit(2)
    rows = load_tsv(out)
    os.unlink(out)
    return rows


def verdict(qid, ratio, recall, baseline, regress):
    """Why `qid` fails, or None."""
    if recall == "mismatch":
        return "recall mismatch"
    if ratio < 1.0:
        return f"slower than Lucene ({ratio:.2f}x)"
    base = baseline.get(qid)
    if base is not None and ratio < base * (1.0 - regress):
        return f"regressed from {base:.2f}x to {ratio:.2f}x"
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--index", required=True)
    p.add_argument("--queries", default=os.path.join(ROOT, "benchmarks", "queries.tsv"))
    p.add_argument("--baseline", required=True)
    p.add_argument("--history")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--regress", type=float, default=0.10)
    p.add_argument("--update-baseline", action="store_true")
    p.add_argument("--negative-control", type=int, default=0)
    p.add_argument("--warmup-ms", type=int, default=2000)
    p.add_argument("--measure-ms", type=int, default=3000)
    args = p.parse_args()

    env = dict(os.environ)
    if args.negative_control > 1:
        env["BENCH_NEGATIVE_CONTROL"] = str(args.negative_control)
    baseline = {}
    if os.path.exists(args.baseline) and not args.update_baseline:
        baseline = {k: float(v["ratio"]) for k, v in load_tsv(args.baseline).items()}
    elif not args.update_baseline:
        # Ratios move with the CPU: a baseline is only meaningful on the
        # machine class that recorded it, so none means no regression rule.
        print(f"bench-gate: no baseline at {args.baseline}; only the "
              "slower-than-Lucene and recall rules apply")

    rows = compare(args, args.queries, env)
    best = {k: float(v["ratio"]) for k, v in rows.items()}
    recall = {k: v["recall"] for k, v in rows.items()}

    # Re-measure what failed on timing alone; a recall mismatch is not noise.
    lines = {}
    with open(args.queries) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                lines[line.split("\t", 1)[0]] = line
    for attempt in range(args.retries):
        again = [q for q in best
                 if recall[q] != "mismatch" and verdict(q, best[q], recall[q], baseline, args.regress)]
        if not again:
            break
        print(f"bench-gate: re-measuring {len(again)} (attempt {attempt + 2}): {', '.join(again)}")
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as t:
            t.writelines(lines[q] for q in again if q in lines)
            subset = t.name
        for q, r in compare(args, subset, env).items():
            best[q] = max(best[q], float(r["ratio"]))
            if r["recall"] == "mismatch":
                recall[q] = "mismatch"
        os.unlink(subset)

    failures = {q: v for q in best if (v := verdict(q, best[q], recall[q], baseline, args.regress))}
    missing = sorted(set(baseline) - set(best))

    variant = os.path.basename(os.path.normpath(args.index))
    ratios = sorted(best.values())
    median = ratios[len(ratios) // 2] if ratios else 0.0
    summary = [f"### Performance gate: `{variant}`", "",
               f"{len(best)} queries, median Rust/Lucene ratio {median:.2f}x, "
               f"minimum {min(ratios, default=0):.2f}x, "
               f">=1.5x on {sum(r >= 1.5 for r in ratios)}/{len(ratios)}", ""]
    if failures:
        summary.append("| query | why |")
        summary.append("|---|---|")
        summary += [f"| {q} | {why} |" for q, why in sorted(failures.items())]
    else:
        summary.append("No query is slower than Lucene or regressed from the baseline.")
    if missing:
        summary.append(f"\nIn the baseline but not measured: {', '.join(missing)}")
    text = "\n".join(summary)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n\n")

    if args.history:
        commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        with open(args.history, "a") as f:
            f.write(json.dumps({
                "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "commit": commit, "variant": variant,
                "negative_control": args.negative_control,
                "ratios": best, "failures": failures}) + "\n")

    if args.update_baseline:
        if failures:
            print("bench-gate: not recording a baseline that fails the gate", file=sys.stderr)
            return 1
        os.makedirs(os.path.dirname(os.path.abspath(args.baseline)), exist_ok=True)
        with open(args.baseline, "w") as f:
            f.write("id\tratio\n")
            for q in rows:
                f.write(f"{q}\t{best[q]:.4f}\n")
        print(f"bench-gate: baseline written to {args.baseline}")

    if args.negative_control > 1:
        if failures:
            print(f"bench-gate: negative control caught ({len(failures)} queries failed)")
            return 0
        print("bench-gate: NEGATIVE CONTROL PASSED THE GATE -- the gate cannot see a "
              f"{args.negative_control}x regression", file=sys.stderr)
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
