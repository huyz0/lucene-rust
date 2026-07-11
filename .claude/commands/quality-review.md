---
description: Run the Tier 2 semantic/design review on the current diff via the quality-reviewer subagent.
argument-hint: "[optional git range, e.g. HEAD~3..HEAD]"
allowed-tools: Task, Bash(git diff:*), Bash(git status:*), Read, Grep, Glob
---

Run a Tier 2 quality review of the current changes.

Target: $ARGUMENTS (if empty, review the working tree + staged changes).

Delegate to the `quality-reviewer` subagent: spawn it via the Task tool and
have it follow `.agents/skills/code-review/SKILL.md`. The deterministic Tier 1
gates have their own enforcement (`cargo fmt`, `cargo clippy`, `cargo test`) —
this command covers only the semantic/design judgment those checks can't make.
Relay the subagent's findings, clearly separating GATING from ADVISORY items.
