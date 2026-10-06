# M11 — Text analysis: `lucene-analysis-common`

> **Goal:** the tokenizers, token filters, char filters and analyzers of
> `lucene-analysis-common`, producing Lucene's token stream exactly, so text
> can be analysed without a JVM.

| | |
|---|---|
| **Effort** | XL |
| **Depends on** | [M7](m7-core-complete.md) (`StandardTokenizer`, the attribute model) |
| **Unblocks** | [M12](m12-language-analysis.md); a Rust engine that analyses in Rust |
| **Status** | in progress -- part 1 (see [Progress](#progress)) |

---

## Why this milestone exists

`lucene-analysis-common` is the largest module after core: 569 files, 109k
lines of code. The port has a small `lucene-analysis` crate (the pieces its
tests and the M3 write path needed). The OpenSearch engine analyses in Java
and hands Rust pre-inverted documents (M5), so none of this has blocked
integration -- but without it the library cannot index text on its own, and
every indexing call still crosses the JVM for analysis.

Much of the module is generated or data-driven: JFlex-generated tokenizers,
Snowball stemmers generated from their own language, hyphenation and
stemming dictionaries. Porting those means porting their generators or their
data, not hand-writing the output.

---

## Scope

### In scope

- Every package: `core`, `standard` (the email/URL tokenizer), `miscellaneous`,
  `ngram`, `shingle`, `pattern`, `path`, `charfilter`, `synonym` (including
  graph synonyms and `SynonymMap`), `compound` (dictionary and hyphenation
  decompounders), `hunspell`, `snowball` and the per-language packages
  (analyzers, stemmers, normalizers, stop sets), `wikipedia`, `email`,
  `boost`, `minhash`, `payloads`, `sinks`, `commongrams`, `cjk`, `th`.
- The factory SPI (`TokenizerFactory` names and arguments), so an analyzer
  can be built from the same configuration text as in Java.
- Performance: every analyzer at least as fast as Lucene's.

### Out of scope

- ICU, Kuromoji, Nori, SmartCN, Stempel, Morfologik, phonetic, OpenNLP (M12).

---

## Tasks

- **T11.1** — A differential harness: a Java generator that runs every
  analyzer over a fixed multilingual corpus and records each token's text,
  offsets, position increment, position length, type, flags and payload.
- **T11.2** — Generated tokenizers: port JFlex output as tables (or run
  a Rust scanner generator over the `.jflex` sources), never by hand.
- **T11.3** — Snowball: port the Snowball compiler's Rust backend output for
  every language Lucene ships, checked against Lucene's stems.
- **T11.4** — Hunspell (dictionary and affix parsing, suggestions).
- **T11.5** — Synonyms, including graph output and flattening.
- **T11.6** — The remaining filters, char filters and analyzers, package by
  package.
- **T11.7** — Factories and the configuration syntax.

## Progress

Part 1 (inventory, harness, the first T11.6 packages, their benchmark):

- **T11.0** -- `docs/inventory/lucene-analysis-common.tsv` lists every class
  of the 10.5.0 jar (498, the `org/tartarus/snowball` runtime included) with
  its status; `scripts/check-port-inventory.py --module analysis-common
  --require-jar` runs in `gate.sh` and CI. `--milestone M11 --summary` is the
  live count of what M11 still owes.
- **T11.1** -- `fixtures/src/GenAnalysisCommon.java` runs ~100 chains (one
  reused `Analyzer` each) over `fixtures/corpus/analysis-common.txt` and
  writes every attribute of every token, `end()`'s state and any exception
  to `fixtures/data/analysis_common/<chain>.tsv`;
  `crates/lucene-analysis/tests/analysis_common_fixtures.rs` compares row for
  row. Chains not yet built are listed in its `PENDING`; a fixture neither
  built nor pending fails. Deterministic, so `gen-fixtures.sh --check`
  covers it.
- `java.lang.Character`'s classification (`getType`, `isLetter`,
  `isWhitespace`, decimal digits) is generated from the Unicode Character
  Database 16.0.0, not the JDK (`docs/licences.md`).
- **T11.6** packages done: `core` (with `FlattenGraphFilter`) and the
  `CharTokenizer` family; `miscellaneous` but `DateRecognizerFilter` and the
  deprecated `WordDelimiterFilter`; `en` (KStem's dictionary generated from
  Lucene's sources); `ngram`, `shingle`, `pattern`, `path`, `charfilter`,
  `commongrams`, `cjk`, `payloads`, `boost`, `minhash`, `email`. All 110
  harness chains match Lucene token for token; `stems.words` (40,392 words
  through KStem and Porter) and `urls.words` (3,000 URL/email fragment
  joins) add word-level checks. Left in these packages: their factories
  (T11.7) and `util`'s helpers for the language packages (`CSVUtil`,
  `OpenStringBuilder`, `SegmentingTokenizerBase`, `StemmerUtil`,
  `CharArrayIterator`).
- **T11.2** -- the JFlex scanners (`UAX29URLEmailTokenizerImpl`,
  `HTMLStripCharFilter`) run a shared skeleton (`util/jflex.rs`) over the
  tables of Lucene's compiled classes, read back by reflection
  (`tools/ExtractJFlexTables.java`) and stored zlib-compressed; the actions
  are ported line for line.
- **Benchmark** -- `scripts/bench-micro.sh --bench analysis_common`
  (`AnalysisCommonMicro.java` / `micro_analysis_common.rs`, ns per token,
  2026-10-06, Rust/Java): ascii_folding 1.85x, cjk 1.27x, english 1.48x,
  html_strip 1.58x, kstem 1.31x, ngram_2_3 1.08~, pattern 0.90~, shingle
  1.03~, simple 1.08~, uax29_url_email 0.99~, wdgf 1.27x, whitespace 1.33x
  (`~`: inside the 1.13x noise floor). The first run had seven cases under
  1.0x; callgrind put the cost in allocation: `AttributeSource`'s derived
  `clone_from` (every `restoreState` built a new value), a `String` per
  KStem dictionary probe and per Porter step, SipHash in `CharArraySet`,
  a UTF-16 round trip per `CharTokenizer` token, and a capture allocation
  per regex match. `pattern` stays at ~0.9x: ~45% of its instructions are
  the `regex` crate's lazy DFA, which finds a match's end forwards and then
  scans back for its start, where `java.util.regex` walks `[ ,.]+` once.
- `java.util.regex` is the `regex` crate behind a Java-syntax shim (`util/java_regex.rs`: ASCII
  `\d\w\s`, `$n` replacements; no backreferences or lookaround).
- The parity ledger's 400 KB budget is shared (the large area files were
  compacted to current facts to make room for M11), so M11's rows are one per package (`docs/parity/analysis-common.md`) and the
  per-class status lives in the inventory.

## Acceptance criteria

- [ ] Every analyzer and factory in the module produces Lucene's token
      stream -- every attribute -- on the T11.1 corpus.
- [ ] Hunspell matches Lucene's stems and suggestions for every dictionary in
      Lucene's test resources.
- [ ] Each analyzer is no slower than Lucene's on the corpus benchmark.
- [ ] `check-port-inventory.py`'s allowlist holds no class of this module.

## Risks and unknowns

- **Size.** At roughly the size of `lucene-core`, this is the milestone most
  likely to overrun; it splits cleanly by package, so it can ship in parts.
- **Licences of data files.** Stop lists, dictionaries and hyphenation
  grammars carry their own licences; each one goes through
  `check-licences.py` and `NOTICE`.

## Exit artifacts

- The analysis differential corpus and generator
- `docs/parity.md` rows for every `lucene-analysis-common` class
