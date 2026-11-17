# Lucene port anchor

What upstream Apache Lucene state the port is based on, and how to list what
landed upstream since. Machine-readable form: [`lucene-port-anchor.json`](lucene-port-anchor.json).

## Anchor

| | |
|---|---|
| **Base (the pin)** | Lucene **10.5.2**, tag `releases/lucene/10.5.2`, commit `145d89cd30c38d25e39a0a2232cf00eee208b9cf` (2026-09-21). Fixtures, jars (`LUCENE_VERSION` in `scripts/lib-lucene-jars.sh`), `Version::LATEST` and `docs/parity.md` are pinned here. All 10.5.0 to 10.5.2 fixes are in (port PR #50). |
| **Partly ported** | Lucene **10.6.0**, tag `releases/lucene/10.6.0`, commit `4ae946e5a9b99d1e22c23865c99e561b8dcc9d71` (2026-10-26). The 202 commits of `10.5.2..10.6.0` were triaged; the output-neutral ones were ported (PRs #56, #57, #58, #60, 2026-10-30). Output-changing items and the read side of the 10.6 index formats were held until the pin moves. `ported_prs` and `skipped_prs` in the JSON list what was taken or rejected. |
| **Index formats** | Reads and writes both: 10.5 (`Lucene104` codec, doc values VERSION 2, `segments_N` of 10.5, the default) and 10.6 (`.dvp`, doc-values VERSION 3, `SegmentInfos` VERSION_10_6). The plugin picks the newest format the host Lucene reads and a setting overrides it (PRs #493, #495). The sandbox Faiss codec (`lucene-faiss`, opt-in) targets Lucene `main` (11.0), whose format may change without notice. |
| **OpenSearch** | The plugin targets OpenSearch **3.9.0**, which ships Lucene **10.5.1**. OpenSearch `main` (3.11.0) already ships **10.6.0**. Checked 2026-11-17. |

How sure: the base is certain (named in `AGENTS.md`, `docs/parity.md`, `scripts/lib-lucene-jars.sh`, and the pin commit #50). The 10.6.0 coverage is a best reconstruction: the ported list comes from the PR numbers in the commit messages of #56 to #60 and the triage in the project files (`plan/lucene-10.6-triage.md`), not from a per-commit ledger, so `scripts/lucene-upstream-diff.py` also reports PR numbers merely mentioned in the repo as `referenced` for a human to confirm.

## Listing what is new upstream

```
git clone --filter=blob:none --no-checkout https://github.com/apache/lucene   # commits and trees only
scripts/lucene-upstream-diff.py --lucene ./lucene                       # branch_10_6 vs the anchor
scripts/lucene-upstream-diff.py --lucene ./lucene --pending             # only what is not ported/skipped/in the pin
scripts/lucene-upstream-diff.py --lucene ./lucene --target origin/main --line 11.0 --pending   # the 11.0 line
scripts/lucene-upstream-diff.py --lucene ./lucene --summary --format md
```

Commits are grouped by area (vectors, codecs-formats, search-query, index, analysis, util-simd) and kind
(bugfix, format?, perf, feature/refactor); test-only and build/docs commits are counted but hidden. Status is
`ported`, `skipped` (with the reason), `in-pin` (the same PR number is in the pinned release's history, so a
backport), `referenced` (mentioned in the repo, unconfirmed) or `pending`. The grouping is a heuristic on paths and
subjects: read the commit before acting on its bucket.

## Keeping it current

The PR that ports or rejects an upstream change adds its PR number to `ported_prs` or `skipped_prs` (with the
measured reason). The PR that moves the pin changes `base`, `partial`, `LUCENE_VERSION`, `Version::LATEST` and the
`opensearch` block together, and re-runs the script as the worklist.
