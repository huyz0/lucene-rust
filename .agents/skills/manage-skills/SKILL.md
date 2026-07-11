---
name: manage-skills
description: "WHAT: Governance for this repo's agent skill system. USE WHEN: creating or editing any file under .agents/skills/, or changing AGENTS.md."
---

# Managing skills

The skill system makes lucene-rust AI-native: an agent finds the right rule by
*trigger*, not by reading the whole repo. This skill governs that system.

## Rules

- **Pointer-only, not a rulebook.** A `SKILL.md` routes to the deep-dive
  (`PLAN.md`, `docs/`, `fixtures/README.md`) and names the gate that enforces
  it. Keep it **under 100 lines** and don't duplicate doc content — on
  conflict, the doc wins; fix the drift.
- **Frontmatter is the trigger.** Exactly two keys: `name` and `description`.
  The description MUST follow `WHAT: <summary>. USE WHEN: <concrete
  triggers>` — triggers are specific crate names, file paths, or commands,
  not vague topics.
- **Bind to an enforcer.** Every skill names the mechanical check that proves
  its rule (a `cargo` command, clippy config, CI job). If a rule can't be
  mechanically checked yet, say so explicitly rather than pretending it is.
- **Self-maintenance.** A change that alters a skill's subject must update the
  skill in the same change. Doc/skill drift is a bug.

## Enforced by

- Nothing mechanical yet (no `xtask skills` linter here) — self-review against
  this checklist. A reasonable candidate for automation once the skill count
  grows past what a human/agent can eyeball.

## Layout

```
.agents/skills/<name>/SKILL.md   # one skill, one responsibility
.claude/skills -> ../.agents/skills   # symlink so Claude Code discovers them
```

Deep detail lives in `PLAN.md`/`docs/`/`fixtures/README.md`; this folder only
routes to it.
