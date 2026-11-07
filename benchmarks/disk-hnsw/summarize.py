#!/usr/bin/env python3
"""Summarizes disk_hnsw_bench CSV output.

    summarize.py <bench.csv> [recall floor, default 0.9] [us per round trip, default 500]

For each memory budget and index, prints the setting with the lowest
estimated latency (CPU plus round trips at the given cost) among those whose
recall@10 reaches the floor, as a Markdown table.
"""
import csv
import sys


def main():
    path = sys.argv[1]
    floor = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
    rt_us = float(sys.argv[3]) if len(sys.argv) > 3 else 500.0
    rows = list(csv.DictReader(open(path)))
    budgets = []
    for r in rows:
        if r["budget_mib"] not in budgets:
            budgets.append(r["budget_mib"])
    budgets.sort(key=lambda b: float("inf") if b == "inf" else float(b))
    indexes = []
    for r in rows:
        if r["index"] not in indexes:
            indexes.append(r["index"])
    print(f"recall@10 >= {floor}, {rt_us:.0f} us per round trip\n")
    print("| budget MiB | index | setting | pinned MB | recall@10 | CPU us | misses | round trips | est. us |")
    print("|---|---|---|---|---|---|---|---|---|")
    for b in budgets:
        for i in indexes:
            cands = [
                r for r in rows
                if r["budget_mib"] == b and r["index"] == i and float(r["recall@10"]) >= floor
            ]
            if not cands:
                ran = [r for r in rows if r["budget_mib"] == b and r["index"] == i]
                if ran:
                    best = max(ran, key=lambda r: float(r["recall@10"]))
                    print(f"| {b} | {i} | (best recall {float(best['recall@10']):.3f}) | | | | | | |")
                else:
                    print(f"| {b} | {i} | does not fit | | | | | | |")
                continue
            def est(r):
                return float(r["cpu_us"]) + float(r["round_trips"]) * rt_us
            best = min(cands, key=est)
            setting = " ".join(
                p for p in best["config"].split() if not p.startswith(("expanded", "estimated", "enc"))
            )
            print(
                f"| {b} | {i} | {setting} | {float(best['pinned_mb']):.1f} | "
                f"{float(best['recall@10']):.3f} | {float(best['cpu_us']):.0f} | "
                f"{float(best['misses']):.0f} | {float(best['round_trips']):.1f} | {est(best):.0f} |"
            )


if __name__ == "__main__":
    main()
