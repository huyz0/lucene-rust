# M12 — Language analysis modules

> **Goal:** Lucene's language-specific analysis modules -- ICU, Japanese,
> Korean, Chinese, Polish, dictionary morphology, phonetic and OpenNLP --
> with the same tokens as Java.

| | |
|---|---|
| **Effort** | L |
| **Depends on** | [M11](m11-analysis-common.md) |
| **Unblocks** | analysing CJK and other scripts without a JVM |
| **Status** | in progress -- part 1: architecture, inventory, T12.2 Stempel, T12.3 phonetic and Morfologik; part 2: T12.1 `analysis/morph`, Kuromoji and Nori (see [Progress](#progress)) |

---

## Why this milestone exists

These modules are small in code but heavy in data (and, for ICU and OpenNLP,
in external libraries). Sizes from the 10.5.0 sources jars:

| Module | Files | Lines of code | What it carries |
|---|---|---|---|
| `analysis-icu` | 26 | 1,230 | ICU4J normalisation, folding, script-aware tokenization, collation keys |
| `analysis-kuromoji` | 59 | 5,015 | Japanese morphological analysis over the IPADIC dictionary |
| `analysis-nori` | 40 | 2,523 | Korean morphological analysis over mecab-ko-dic |
| `analysis-smartcn` | 21 | 1,399 | Chinese word segmentation (HMM, bundled dictionary) |
| `analysis-stempel` | 21 | 2,093 | Polish stemming tables |
| `analysis-morfologik` | 9 | 376 | FSA dictionaries (Morfologik) |
| `analysis-phonetic` | 10 | 449 | Commons Codec phonetic encoders |
| `analysis-opennlp` | 20 | 998 | OpenNLP models (sentence, POS, chunk, lemmatize) |

---

## Scope

### In scope

- All eight modules, reading the same dictionary and model files Lucene
  ships or loads, so a user's existing dictionaries work unchanged.
- ICU through a Rust ICU implementation (ICU4X), with a differential test
  proving the output matches ICU4J's for the operations Lucene uses; where
  it cannot, the gap is recorded per operation.

### Out of scope

- Training models or building dictionaries; only loading and using them.

---

## Tasks

- **T12.1** — Kuromoji and Nori: the Viterbi lattice, user dictionaries,
  n-best output, search-mode decompounding.
- **T12.2** — SmartCN and Stempel.
- **T12.3** — Morfologik FSA reading and phonetic encoders (reimplementing
  Commons Codec's algorithms, each against its Java output).
- **T12.4** — ICU over ICU4X.
- **T12.5** — OpenNLP: model loading and inference, or a recorded decision
  that it stays JVM-only if no faithful Rust runtime exists.
- **T12.6** — The analysis-common classes M11 deferred
  (`docs/inventory/lucene-analysis-common.tsv`, `deferred:M12`): Thai
  (`ThaiAnalyzer`, `ThaiTokenizer`, `ThaiTokenizerFactory`, which today
  refuses configuration with `UnsupportedOperationException`), collation
  (`CollationKeyAnalyzer`, `CollationAttributeFactory`,
  `CollationDocValuesField`, `tokenattributes/CollatedTermAttributeImpl`) and
  `analysis/util/{SegmentingTokenizerBase, CharArrayIterator}`. Each rests on
  a JDK service no Rust crate reproduces bit for bit -- `java.text.BreakIterator`
  (Thai's dictionary word breaker, the sentence iterator
  `SegmentingTokenizerBase` drives) or `java.text.Collator` (sort keys from
  the JDK's locale data): implement it faithfully, checked against the JDK on
  the same text as `lang/final_sigma.rs` checks the word iterator, or record a
  not-supported decision naming the typed error a caller sees. T12.2's SmartCN
  (`HMMChineseTokenizer` extends `SegmentingTokenizerBase`) depends on this
  task; `analysis/morph` (the Kuromoji/Nori base, also deferred from M11) is
  T12.1's.
- **T12.7** — The other JDK text services M11's port stops short of, each
  ported faithfully against the JDK or closed with a recorded not-supported
  decision: `java.util.regex` beyond the shim (`util/java_regex.rs` refuses
  backreferences, lookaround, possessive and atomic groups, `\b`, `(?m)`
  anchors, script and block properties with `UnsupportedOperationException`;
  `pattern*` and `KeywordMarker` factories), `SimpleDateFormat` locale data
  (`DateRecognizerFilterFactory` outside `Locale.ENGLISH`), and JDK charsets
  beyond UTF-8, UTF-16, US-ASCII and the Hunspell tables (hyphenation
  grammars, `factory/xml_source.rs`).

## Architecture

One crate per Lucene module, named after it: `lucene-analysis-phonetic`,
`lucene-analysis-stempel`, `lucene-analysis-morfologik` (part 1), then
`lucene-analysis-kuromoji`, `-nori`, `-smartcn`, `-icu` and `-opennlp`.
Each depends on `lucene-util` and `lucene-analysis` only -- the edges
Lucene's own `module-info.java` declares (`lucene-core`,
`lucene-analysis-common`) -- and nothing in the workspace depends on them:
they sit beside `lucene-search` above `lucene-analysis`, siblings never
depend on each other, and each carries `#![forbid(unsafe_code)]` and the
workspace lints (the arithmetic gate). Why one crate per module rather than
one `lucene-analysis-extra`: the modules carry megabytes of dictionary data
and, for ICU and OpenNLP, heavy third-party runtimes; a user who wants Polish
stemming should not compile ICU4X, and Lucene ships them as separate jars for
the same reason. Shared machinery stays in `lucene-analysis`: the
`analysis/morph` Viterbi base of Kuromoji and Nori, `SegmentingTokenizerBase`
(SmartCN, ICU, Thai), the factory SPI each module registers into
(`register_factories()`, as `lucene-search` registers `Word2VecSynonym`), and
the `java.lang.Character` tables.

Each module's classes have a status in `docs/inventory/lucene-analysis-<m>.tsv`
(`scripts/check-port-inventory.py --module analysis-<m> --require-jar`, in
`gate.sh` and CI); `--milestone M12 --summary` is what each still owes. The
third-party code a module is built on (Commons Codec, Morfologik's FSA
reader) is not in a Lucene jar, so `docs/parity/analysis-lang-modules.md`
records it.

## Progress

Part 1:

- **Inventory** -- the eight modules' 178 top-level classes (Stempel's
  `org/egothor/stemmer/` included), each `todo:M12` with its task until
  ported.
- **T12.3 phonetic** -- `lucene-analysis-phonetic`: the four filters and
  factories (`register_factories()`), over Commons Codec 1.17.2's encoders
  reimplemented rule for rule on UTF-16 units (`Soundex` and its two variant
  mappings, `RefinedSoundex`, `Metaphone`, `DoubleMetaphone`, `Caverphone1`/`2`,
  `ColognePhonetic`, `Nysiis`, `MatchRatingApproachEncoder`,
  `DaitchMokotoffSoundex`, Beider-Morse with its 123 rule files vendored).
  `GenAnalysisPhonetic.java`: 18,076 words x 28 encoder/option columns,
  2,188 words x 16 Beider-Morse columns and 52 `CustomAnalyzer`
  configurations (11 refused), all equal. `String.toUpperCase`'s special
  casing (`ß` -> `SS`) joined `lucene-analysis` for it
  (`java_string_to_upper_case`).
- **T12.2 Stempel** -- `lucene-analysis-stempel`: Egothor's table reader
  (`Trie`, `Row`, `Cell`, `MultiTrie2`, `Diff.apply`; the Egothor licence
  kept in the source and `NOTICE`), `StempelStemmer`, `StempelFilter`,
  `PolishAnalyzer` and `stempelPolishStem`, over Lucene's
  `stemmer_20000.tbl` (vendored, zlib). Table building (`Compile` and its
  optimisers) is out of scope. `GenAnalysisStempel.java`: 30,198 words
  through the default table, eight tables Egothor's own `Compile` builds
  (every kind and optimiser) read and run, 5 chains and the factory, all
  equal.
- **T12.3 Morfologik** -- `lucene-analysis-morfologik`: Morfologik's FSA5
  and CFSA2 automata, `.info` metadata (`java.util.Properties` syntax, every
  attribute validated), `DictionaryLookup` with the four sequence decoders
  (and the reused-buffer stale tag of a tagless entry), `MorfologikFilter`
  with `MorphosyntacticTagsAttribute`, `MorfologikAnalyzer`,
  `UkrainianMorfologikAnalyzer` and `morfologik`. The Polish (BSD-2-Clause)
  and Ukrainian (Apache-2.0) dictionaries are vendored zlib-compressed so
  the default constructors work. `GenAnalysisMorfologik.java`: 47,247
  Polish and 23,248 Ukrainian lookups, eight dictionaries built with
  Morfologik's own builder (every encoder, both formats, two charsets,
  conversions, tagless entries), analyzer, filter and factory rows with
  tags, all equal.
- **Benchmark** -- `scripts/bench-micro.sh --bench analysis_m12`
  (`AnalysisM12Micro.java` / `micro_analysis_m12.rs`, ns per token,
  2026-10-07, Rust/Java, on a machine shared with another build, noise
  floor 1.12x): beider_morse_ash_exact 2.91x, beider_morse_gen_approx 3.32x,
  daitch_mokotoff 1.24x, double_metaphone_filter 1.43x, ph_caverphone2
  5.25x, ph_cologne 1.29x, ph_double_metaphone 1.50x, ph_metaphone 1.31x,
  ph_mra 2.70x, ph_nysiis 1.57x, ph_refined_soundex 0.94~, ph_soundex
  1.00~, stempel_filter 1.17x, polish_analyzer 1.22x. The faithful port read
  ph_soundex 0.64x and ph_refined_soundex 0.52x: the cost was allocation and
  table lookups for ASCII words (now an ASCII path through `clean`,
  `toUpperCase` and the refined code, and a reused term buffer).
  Morfologik: this run's morfologik_filter 0.90~ and ukrainian_analyzer
  0.99~ did not reproduce -- the Tier 2 review measured 0.70-0.77x and
  ~0.85x pinned and warmed, with `DictionaryLookup::lookup` at 81% of the
  profile (the CFSA2 arc scan through a `Result` per byte read 43%, every
  sequence materialised as a `Vec` 27%, the allocator ~20%, the UTF-16
  round trip 5-7%). After reading each arc's flag once, visiting sequences
  in one reused buffer, reusing every lookup buffer and looking the term's
  UTF-8 up directly, a rerun pinned to one core (3 reps, load < 1.5,
  noise floor 1.10x) measured morfologik_filter 2.45x and
  ukrainian_analyzer 2.11x; the remaining profile is the arc scan itself
  (`Fsa::arc`, a third of the instructions -- Morfologik scans the same
  arcs). The same rerun read ph_soundex 0.81x (noise 1.20x), below the
  1.00~ above. Re-measured in part 2, pinned to two cores, four times:
  ph_soundex 0.98~, 0.92~, 1.04~, 0.65x and ph_refined_soundex 0.91~,
  1.36x, 1.05~, 1.20~ (noise floors 1.13-1.35x; Java itself ranged
  110-155 ns/token). The profile of the ~150 ns token is the whitespace
  tokenizer, the captured state and allocation; Soundex's own share was
  `clean`'s copy, now streamed on the ASCII path (A/B best-of-6: 140 ->
  123 ns/token). What is left is shared with Java's chain, so the ratio
  follows the machine.
- **Custom attributes** -- `lucene-analysis`' `AttributeSource` now holds
  attributes outside the core set (`CustomAttribute`, `add_custom`), which
  Morfologik's tags use and Kuromoji's, Nori's and ICU's will.

Part 2 (T12.1):

- **`analysis/morph`** -- `lucene-analysis/src/morph/`: `Viterbi` and
  `ViterbiNBest` (the lattice, n-best `Lattice`, `fixupPendingList`),
  `GraphvizFormatter`, `TokenInfoFST` over a reader of Lucene's `FST<Long>`
  (all four node encodings; `lucene-codecs`' FST is out of this crate's
  reach and allocates per arc), `BinaryDictionary`, `CharacterDefinition`,
  `ConnectionCosts`. Java's abstract classes are state plus a language
  trait (`ViterbiLang`, `NBestLang`). A user dictionary is a trie over UTF-16
  units rather than an FST (a lookup observes the same arcs and sums). The
  module carries the arithmetic gate (module-scope deny).
- **Kuromoji** -- `lucene-analysis-kuromoji`: `JapaneseTokenizer` (three
  modes, punctuation, compounds, user phrases, n-best and
  `calcNBestCost`), the four attributes as `CustomAttribute`s with Java's
  reflection (`AttributeSource::reflect_with`, added for this), the system,
  unknown and user dictionaries (default instances from the vendored IPADIC,
  or a caller's files), `ToStringUtil`, every filter, the iteration-mark
  char filter, `JapaneseAnalyzer`, `JapaneseCompletionAnalyzer` and the ten
  factories. `GenAnalysisKuromoji.java`: 448 lines (145 written here, 303
  seeded stress lines with three long enough to force the 1,024-position
  backtrace) through 16 tokenizer configurations, Graphviz lattices of
  every corpus line, `calcNBestCost`, 4,881 dictionary entries, and 45
  analyzer/factory chains (8 refused), all equal; hostile-dictionary sweeps
  (`kuromoji_hostile.rs`) never panic.
- **Nori** -- `lucene-analysis-nori`: `KoreanTokenizer` (decompound modes
  `NONE`/`DISCARD`/`MIXED`, unknown unigrams, punctuation, the space
  penalty, user dictionaries with compounds), its four attributes as
  `CustomAttribute`s, `POS` tags and types, the system, unknown and user
  dictionaries (vendored mecab-ko-dic, or a caller's files),
  `KoreanPartOfSpeechStopFilter`, `KoreanReadingFormFilter`,
  `KoreanNumberFilter`, `KoreanAnalyzer` and the factories. Java's
  `Character.UnicodeScript.of` (which Nori's unknown-word grouping uses)
  is a BMP table generated from the JDK (`java_unicode_script.rs`).
  `GenAnalysisNori.java`: 424 lines (121 written here, 303 stress) through
  8 tokenizer configurations with every attribute, Graphviz lattices,
  10,236 dictionary entries and 24 analyzer/factory chains (6 refused),
  all equal; hostile-dictionary sweeps (`nori_hostile.rs`) never panic.
- **Bench** (`scripts/bench-micro.sh --bench analysis_m12`, pinned, 3
  reps): kuromoji_normal 0.93~, search 0.96~, extended 1.11~, nbest 0.96~,
  japanese_analyzer 0.89~ (noise floor 1.21x); the first faithful port read
  0.64-0.70x. Profile-backed (perf): the per-position division in
  `WrappedPositionArray`, a per-byte presence-bit count and a `Position`
  clone per backtraced token were removed; what remains is memory-bound --
  half of `add`'s samples and most of `forward`'s sit on loads from the
  3.4 MB connection-cost matrix, the target map and the entry buffer,
  which Java's lattice touches identically.
- **Dictionaries** -- licence-clean for redistribution (IPADIC's NAIST/ICOT
  terms; mecab-ko-dic is Apache-2.0), so vendored zlib-compressed: IPADIC
  9.2 MB -> 4.6 MB, mecab-ko-dic 25.0 MB -> 7.6 MB in the repository
  (`docs/licences.md`).

## Acceptance criteria

- [ ] Each module produces Lucene's token stream on a native-language corpus
      for its language(s), every attribute.
- [ ] Kuromoji and Nori match Lucene's n-best lattices, not just the best
      path.
- [ ] Every bundled dictionary and model is licence-audited and in `NOTICE`.
- [ ] Each analyzer is no slower than Lucene's.

## Risks and unknowns

- **ICU4X versus ICU4J.** The two implement the same Unicode standards but
  are separate codebases; differences in normalisation or break iteration
  are possible and must be found by the differential test, not assumed away.
- **OpenNLP has no Rust runtime.** T12.5 may end as a documented decision
  rather than a port.

## Exit artifacts

- Per-language differential corpora and generators
- `docs/parity.md` rows for all eight modules
