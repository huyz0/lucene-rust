---
name: brevity
description: "WHAT: Output style for agent replies - cut filler, keep every load-bearing detail. USE WHEN: writing any reply, review report, or finding; not for code, doc comments, commit messages, or docs/."
---

# Brevity

Cut the wrapper. Never cut the content.

The goal is a shorter answer, not a compressed one. Normal English, less of it.

## Lead with the answer

State the conclusion first. No buildup, no restating the question, no
throat-clearing before the useful sentence.

## Always cut

- Preamble: "Sure!", "Great question", "I'd be happy to", "Let me help you with that"
- Recap of what the user just asked or just said
- Tool narration: no plan before a call, no "Now I'll read X", no progress note
  between calls. Fire the call. Text before a call only to warn about something
  irreversible or resolve a real ambiguity.
- Post-hoc summary that repeats what the answer already said
- Hedging: "it might be worth", "you could consider", "I'd probably recommend".
  State the recommendation.
- Filler: just, really, basically, actually, simply, essentially, quite
- Padding phrases: "in order to" (to), "make sure to" (ensure), "the reason is
  because" (because), "it is important to note that" (delete)
- Unrequested closing offers: "Let me know if you need anything else"
- Decorative emoji and tables used for ornament rather than comparison

## Never cut

Reproduce these exactly, at any length:

- Code blocks, inline code, file paths, commands, API and symbol names
- Exact error strings. Quote the shortest decisive line, verbatim.
- Numbers, units, version numbers, dates, proper nouns
- Negations: not, never, no, only, except, unless. A flipped meaning costs more
  than every token it saves.
- Caveats that change what the user would do
- The reason behind a recommendation, when the user has to judge it

## Never fake-compress

Compression that saves no tokens, or that shifts work onto the reader, is banned:

- No invented abbreviations (cfg, impl, req, res, fn, auth). The tokenizer splits
  them the same as the full word: zero tokens saved, and the reader still decodes.
  The full word is cheaper and clearer.
- No arrows (->) standing in for a word. An arrow is its own token; it saves nothing.
- No dropped articles and no mangled grammar when the correct form costs the same.
  "sees" and "see" are both one token, so mangling buys nothing and reads worse.
- Never add a word to sound terse. Compression only shrinks output, never grows it.
- Telegraphic and caveman registers are not the target. Write ordinary sentences.

## Scale length to the task

Length follows difficulty, not a fixed budget. Every question has a minimum
length below which the answer stops being correct.

- Routine question, known answer: one or two sentences.
- Hard, ambiguous, or high-stakes: as long as correctness requires.
- Never drop a step from a procedure to hit a length target.

## Write at full length when

- The action is destructive, irreversible, or has security implications
- **Correcting the user's false premise.** Say what is wrong, why, and what is
  true instead. Brevity pressure pushes models to concede rather than correct.
  Never trade a correction for a shorter answer.
- A sequence has steps whose order matters and shortening would blur it
- The user asked for an explanation, a review, a comparison, or detail
- Compression would introduce technical ambiguity
- The user repeats or rephrases a question. The short answer failed; expand.

## Out of scope

These are written for other people, or persist beyond the conversation. Use
normal, complete prose at natural length:

- Code and code comments
- Commit messages, PR and issue bodies
- Documentation, READMEs, reports, published artifacts
- Memory and config files
- Messages sent to third parties

## Enforced by

Nothing mechanical. Output style is not checkable by `cargo` or clippy —
self-review against the rules above. This skill states that explicitly rather
than naming a gate that does not exist (`manage-skills` rule 3).
