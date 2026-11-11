# lucene-pattern-text -- log messages as template plus variables

[Index](../parity.md). Not a Lucene module: an opt-in field layout of this
port's own design, after Elasticsearch's `pattern_text`. A value is written as
three ordinary Lucene fields, so stock Lucene reads the files. Measured on
10,567,225 real Loghub lines (1.58 GB): the index is 18% smaller than a
positional `text` field sorted by time, 28% smaller sorted by source and
template, and phrases run in 2.5-35 ms against 3.7-21 ms with positions
(`match_only_text`: 6-27 s); raw numbers and code in the project's
`perf/log-template/`.

| Java | Rust | Status |
|---|---|---|
| -- | `lucene-pattern-text/src/format.rs::{split, join, encode_args, decode_args, slots, VARIABLE}` | **rust-only**. A message splits on single spaces; a token holding an ASCII digit or U+0002 is a variable, replaced by U+0002 in the template. Variables are stored as a vInt byte length and the UTF-8 bytes each. Every message rebuilds exactly (runs of spaces, edge spaces, a literal U+0002). Tests: unit tests (round trips, multi-byte lengths, corrupt values refused). |
| -- | `lucene-pattern-text/src/fields.rs::fields` | **rust-only**. Field `f`: the message indexed docs-only without norms, `f.template` a `SORTED` doc value, `f.args` a stored binary value when the message has variables. Single-valued; a template over 32,766 bytes is refused, as Lucene's `SORTED` writer does. Tests: unit test; `tests/phrase_index.rs`. |
| -- | `lucene-pattern-text/src/phrase.rs::{pattern_phrase_query, PatternPhraseSource, PhraseTokens}` | **rust-only**. A phrase is its terms as `MUST` clauses plus a `FunctionMatchQuery` that verifies each candidate: once per template ordinal and segment, a template whose static text holds the phrase matches outright and one without variables is decided outright; the rest rebuild the message from the stored variables and run the field's analyzer over it. Positions follow the analyzer's increments, so a removed stopword's gap must line up as in a positional phrase; tokens stacked at one position (synonyms) match any of them. The analyzer must never join tokens across a space; `whole_messages_only` turns the template shortcut off for one that does. Tests: unit tests; `tests/phrase_index.rs` (two segments written by the index writer, with and without the shortcut: every phrase's hits equal a positional `PhraseQuery` on the same messages, and with a stopword analyzer a brute-force position check). |
