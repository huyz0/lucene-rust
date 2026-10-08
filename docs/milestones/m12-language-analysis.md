# M12 — Language analysis modules

> **Goal:** Lucene's language-specific analysis modules -- ICU, Japanese,
> Korean, Chinese, Polish, dictionary morphology, phonetic and OpenNLP --
> with the same tokens as Java.

| | |
|---|---|
| **Effort** | L |
| **Depends on** | [M11](m11-analysis-common.md) |
| **Unblocks** | analysing CJK and other scripts without a JVM |
| **Status** | in progress -- part 1: architecture, inventory, T12.2 Stempel, T12.3 phonetic and Morfologik; part 2: T12.1 `analysis/morph`, Kuromoji and Nori; part 3: T12.6, T12.2 SmartCN; part 4: T12.4 ICU normalization, tokenizer, collation and transforms (see [Progress](#progress)) |

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
- ICU through a port of the ICU4J runtime pieces Lucene calls, reading
  ICU's own data files (ICU4X reads neither ICU4J's data formats nor
  Lucene's own `utr30.nrm`/`.brk` files), with differential tests proving
  the output matches ICU4J's; where it cannot, the gap is recorded per
  operation.

### Out of scope

- Training models or building dictionaries; only loading and using them.

---

## Tasks

- **T12.1** — Kuromoji and Nori: the Viterbi lattice, user dictionaries,
  n-best output, search-mode decompounding.
- **T12.2** — SmartCN and Stempel.
- **T12.3** — Morfologik FSA reading and phonetic encoders (reimplementing
  Commons Codec's algorithms, each against its Java output).
- **T12.4** — ICU: a port of the ICU4J runtime Lucene calls (ICU4X cannot
  read ICU4J's or Lucene's data files).
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
  Bench (`analysis_m12`, pinned to two cores, 3 reps): nori_discard 2.33x,
  nori_mixed 2.21x, korean_analyzer 2.19x (noise floor 1.15x); the part 2
  run read 2.28x, 2.26x, 2.12x and the Tier 2 review's 2.44x, 1.58x, 1.69x
  (noise 1.17x) -- faster than Lucene in every run, by a margin the shared
  machine moves.
- **Hostile sweep** -- both generators also run a seeded corpus of 400
  lines drawn from pools no tokenizer is tuned for (`fixtures/src/HostileText.java`:
  half-width forms, supplementary kanji, emoji, combining marks, joiners,
  variation selectors, jamo, Thai, private use, C0 controls, radicals,
  specials) through every option combination -- 48 for Kuromoji (mode,
  punctuation, compounds, user dictionary, n-best 0/2000), 24 for Nori
  (decompound mode, unigrams, punctuation, user dictionary) -- with every
  attribute; `sweep.tsv` holds one row-count and FNV-1a digest per line
  (1.2 million rows in all), byte-identical under JDK 21 and 25, all equal.
- **Bench** (`--bench analysis_m12` harness, pinned to cores 2-3, 5 reps,
  2026-10-07, noise floor 1.17x): kuromoji_normal 1.13~, search 1.08~,
  extended 1.10~, nbest 1.39x, japanese_analyzer 1.14~ (Rust 1,696 vs Java
  1,939 ns/token). History: the first faithful port read 0.64-0.70x; a
  first optimisation pass (the per-position division in
  `WrappedPositionArray`, a per-byte presence-bit count, a `Position` clone
  per backtraced token, a `String` and SipHash per part-of-speech probe)
  reached 0.91-0.99~ and 0.87~. The second pass, decided on callgrind
  instruction counts (x86-64-v3, fat LTO; per pass over the corpus before
  -> after: analyzer 122.2M -> 94.5M, normal 89.5M -> 71.9M, search
  104.5M -> 84.4M, extended 107.3M -> 87.1M, nbest 246.2M -> 150.0M) and
  an interleaved Rust-vs-Rust A/B (speed-ups 1.13-1.33x): `add` reads the
  word's row of the cost matrix once and scans each position's arcs as
  8-byte (cost, right id) records apart from the rest (Java's parallel
  arrays have the same split); the word's connection data is one 4-byte
  read without a `dyn` call; `WrappedPositionArray::get` and the
  `RollingCharBuffer` (now one vector, the freed front dropped in bulk) have
  inline fast paths, and `reset` touches only the live positions, as
  Java's does; the n-best lattice walks its chains in place (it collected
  each into a `Vec`, two allocations per position); `pruneAndRescore`
  reuses its two vectors; the FST reader's one-byte varints and labels are
  inline. In the filters, `CharArraySet`'s ignore-case probe lowercased
  every non-ASCII word into a new `String` (now only one that lowercasing
  changes) and the katakana stem filter re-encoded every term to UTF-16
  (now a scan of the chars); `AttributeSource` finds a custom attribute by
  its stored type id (it made two `dyn` calls per attribute per lookup, and
  the tokenizer sets four per token: analyzer 94.5M -> 92.6M). What
  remains of the profile is the lattice
  itself -- `add`'s least-cost scan, the FST walk per start position, the
  backtrace -- the work Java's `Viterbi` does, on the same arrays.
- **Dictionaries** -- licence-clean for redistribution (IPADIC's NAIST/ICOT
  terms; mecab-ko-dic is Apache-2.0), so vendored zlib-compressed: IPADIC
  9.2 MB -> 4.6 MB, mecab-ko-dic 25.0 MB -> 7.6 MB in the repository
  (`docs/licences.md`).

Part 3 (T12.6, T12.2 SmartCN, T12.7):

- **T12.6 sentence `BreakIterator`, `SegmentingTokenizerBase`** --
  `lucene-analysis/src/util/sentence_break.rs`: the JDK's sentence iterator
  re-specified from black-box runs (no JDK code or data): thirteen classes from
  `Character.getType` plus JDK 25's 128 exception ranges
  (`tools/GenSentenceBreakClasses.java`), and an 11-state scanner learned
  from the JDK as a Mealy machine, exhaustively equal on every class string up
  to length 7 (39M strings) and on random texts, quirks included.
  `SegmentingTokenizerBase` is state plus a `Segmenter` trait;
  `CharArrayIterator` is ported. `GenAnalysisSegmenting.java`: every code
  point's class (JDK 21/25-dependent ones left out), 20,009 texts' boundaries
  and two subclasses over texts five windows long, all equal; byte-identical
  under JDK 21 and 25.
- **T12.6 Thai: not supported** -- Thai words come from the JDK's
  dictionary word iterator over its GPL-licensed `thai_dict`, which cannot
  ship here or be re-derived without copying it. The port behaves as Lucene
  on a JRE without Thai segmentation: `ThaiTokenizer::new`, `ThaiAnalyzer`'s
  streams and the `thai` factory's `create` return
  `AnalysisError::UnsupportedOperation("This JRE does not have support for
  Thai segmentation")` (a new variant); the stop set and `normalize` work.
  Part 4 re-checked the ICU route once T12.4's `thaidict` engine existed:
  ICU's Thai word breaks differ from the JDK's on 8 of the corpus's 10 Thai
  lines (JDK 21 and 25 alike), so `ThaiTokenizer` over ICU would not be
  Lucene's `ThaiTokenizer` -- it stays not supported, and `ICUTokenizer` is
  the Thai tokenizer this port offers.
- **T12.6 collation: not supported** -- `java.text.Collator`'s sort keys
  are a function of the JDK's `CollationRules` table and locale tailorings
  (GPL JDK data; `Locale.ROOT` needs the whole table), so reproducing
  `RuleBasedCollator` even for `ROOT`/`ENGLISH` means copying it. No Rust
  type; ICU collation (T12.4) is the route. The inventory gained the status
  `not-supported` (`check-port-inventory.py`); analysis-common has no
  `deferred:M12` class left.

- **T12.2 SmartCN** -- `lucene-analysis-smartcn`: `coredict.mem` and
  `bigramdict.mem` (Lucene's Java-serialized dictionaries, Apache-2.0 from
  imdict.net, vendored zlib-compressed 6.4 MB -> 3.5 MB) read by a reader of
  Java serialization's primitive-array subset, every count bounded and every
  hash slot validated; `WordDictionary`, `BigramDictionary`, the HHMM
  segment graph, bigram graph and shortest path, `WordSegmenter`,
  `HMMChineseTokenizer` (on `SegmentingTokenizerBase`),
  `SmartChineseAnalyzer` and `hmmChinese`. The `.dct` fallback loaders
  (`AnalyzerProfile`) are not ported: Java reads them only when the jar's
  `.mem` resources are missing. `GenAnalysisSmartcn.java`: 86 corpus lines
  (`corpus/analysis-chinese.txt`, written here) and 200 seeded stress lines
  through the tokenizer, three analyzers and the factory, 8,000+ dictionary
  lookups, 3,000+ bigrams and every corpus line's segmentation path with its
  weights, all equal; byte-identical under JDK 21 and 25; corrupt
  dictionaries never panic (`smartcn_hostile.rs`).
- **Bench** (`--bench analysis_m12`, 2026-10-07, noise floor 1.15x):
  sentence_segmenting 1.15~. The faithful SmartCN port read
  smartcn_tokenizer 0.89~ and smartcn_analyzer 0.91~; perf put a quarter of
  the time in the allocator (a clone of every segment for the path, of every
  bigram key for its edge) and a fifth in per-lookup hash probing. Taking the
  segments instead of cloning them, keeping no edge text and probing each
  unit's dictionary row once at load gave 1.25x and 1.26x (Rust re-measured
  against the same run's Java: 1,187 vs 1,480 and 1,530 vs 1,929 ns/token).
- **T12.7 `java.util.regex`** -- `util/java_backtrack.rs`: a backtracking
  matcher re-specified from `Pattern`'s grammar and black-box runs (no JDK
  code) runs every pattern the `regex`-crate shim refuses: backreferences,
  lookaround, atomic and possessive groups, `\b`, `(?m)`, `(?x)`, `(?U)`,
  `(?d)`, `\G`, `\R`, script/block/binary/`java*` properties (generated,
  `tools/GenJavaRegexProperties.java`), repeated-group captures. It makes
  Java's node choices (first-match `Curly`/`Ques`, `GroupCurly`, `Loop`
  with the per-`find` failed-position memo), lookbehind windows in `int`
  arithmetic and `StartS`'s code-point stepping. Still refused (typed
  `UnsupportedOperation`): `\X`, `\b{g}`, `\N{..}`, `(?c)`. Java's
  `StackOverflowError` is a typed error (first: 256 KiB of the caller's
  stack, then a retry on a 32 MiB thread; now: see below).
  `GenJavaRegex.java`: 174
  curated patterns x 30 inputs and 2,500 generated patterns x 8, every
  `find()` span and group, `replaceAll`, `matches()`; 160,000 more
  generated patterns (seven seeds) checked locally; byte-identical under JDK
  21 and 25. **Matcher, second form** (after the Tier 2 review found a
  stack overflow inside a negated lookaround or a zero-count quantifier
  read as a match, `X{0,1}` not behaving as `X?`, and the 256 KiB budget
  measured from the call site rather than against what the caller's stack
  has left): the tree is compiled to a flat program run with explicit
  stacks on the heap (`util/java_backtrack/vm.rs`) -- every choice the
  continuation-passing matcher kept on the native stack (an alternative, a
  capture to undo, a back-off position, a loop's memo to record) is an
  entry, so a match's depth is heap (Java's `StackOverflowError` past 2^20
  entries, some 200,000 iterations of `(a|b)*` against Java's 1,552); only
  lookarounds, atomic groups and quantifiers over sub-patterns run nested,
  as deep as the pattern nests them, measured against 64 KiB of the
  caller's stack (a parse: also at most 32 levels of groups and classes) and
  started again, only when that runs out, on a 32 MiB thread spawned for
  the call (24 MiB measured). **Third review** (a pattern nesting
  lookarounds or atomic groups 3,000-22,000 deep compiled, then aborted the
  process with a Rust stack overflow on that thread; 20,000 nested `(a|`
  took 113 s to compile; any pattern with more than 32 groups paid a thread
  per match): groups and classes nest at most 20,000 levels -- Java on a
  1 MiB thread fails between 700 and 3,100 levels of groups, 6,251 of
  `[`, 8,424 of `[a&&` (two levels each) -- beyond which `compile` fails with Java's own
  `PatternSyntaxException` "Stack overflow during pattern compilation"; the
  per-alternation first characters and nested quantified groups' studies
  are computed once (no longer quadratic: 20,000 levels compile in some
  0.1 s, release); a `(?i)` backreference compares code points until the
  group's units are covered (`(?i)(😀)\1`); a capture under `X{0}` routes
  to the backtracking matcher, which counts it; a matcher keeps at most
  64 KiB of backtracking stacks between `find()`s. `deep.txt` (220
  patterns over 300-1,600 characters) and `{0,1}`/`{0,2}` in the generated
  quantifiers joined the fixtures; 100,000 generated patterns x 6 inputs
  (JDK 21 and 25 answers) and 20,000 of a fresh seed, 0 differences. Bench
  (`--bench analysis_m12` harness, pinned to cores 2-3, 5 reps, noise floor
  1.17x): regex_lookaround_replace 1.33x, regex_boundary_capture 1.26x,
  regex_backref_split 1.61x. History: the faithful port read 0.93~, 0.69x,
  0.80x; pooling and one-way runs in the continuation-passing matcher
  reached 0.96~, 0.84~, 0.65x, where a third of the time was the call per
  node. The machine runs one-way runs as one instruction (a one-step
  repetition's minimum joins the run before it, and a one-way capture
  checks the run after it before it records anything to undo), picks an
  alternation's live alternatives from a table per ASCII unit, tries search
  starts inside one machine loop, and reads `\b` on ASCII without a code
  point decode: callgrind per pass over the names, before -> after,
  24.6M -> 11.4M, 19.4M -> 12.4M and 7.0M -> 3.6M instructions.
- **T12.7 `SimpleDateFormat` locales** -- `DateRecognizerFilterFactory`'s
  `locale` takes the 1,151 JDK locales without a variant on a Gregorian or
  Buddhist calendar: `miscellaneous/date_locales.rs`, generated by
  `tools/GenDateLocales.java` (CLDR data through the JDK's API, Unicode
  licence) -- the names each text field accepts (every display name tried
  through `SimpleDateFormat` itself; `M` reads stand-alone names when it is
  the pattern's only field, format names beside another), default patterns,
  number symbols. Refused (`UnsupportedOperation`): a tag the JDK resolves
  by fallback, a variant or extension, the Japanese calendar.
  `GenAnalysisDateLocales.java`: 24,570 texts over the 886 locales JDK 21
  and 25 agree on (`corpus/date-locales.txt`); JDK 25-only data in unit
  tests. Bench (`--bench analysis_misc`): date_default_ru 5.38x (date_default 4.94x, date_iso 6.69x).
- **T12.7 charsets** -- the hyphenation grammar's `encoding` decodes every
  single-byte JDK charset (104: ISO-8859-*, windows-125*, KOI8-*, IBM and
  EBCDIC code pages, Mac), `factory/xml_charsets.rs` generated by
  `tools/GenXmlCharsets.java` (identical under JDK 21 and 25). **Not
  supported:** the multi-byte ones (Shift_JIS, EUC-JP, GBK, GB18030, Big5,
  EUC-KR, ISO-2022-*, UTF-32, CESU-8): megabytes of vendor tables or
  stateful decoders, for grammars that are UTF-8 or ISO-8859 in practice;
  they stay a typed `UnsupportedOperation`.

Part 4 (T12.4 ICU):

- **The ICU4J runtime** -- `lucene-analysis-icu/src/icu4j/`, a port of the
  ICU4J 77.1 classes the module calls, reading ICU's own data files (why not
  ICU4X: `lib.rs`): `Normalizer2` over `.nrm` (ICU's five and Lucene's
  `utr30.nrm`), `UnicodeSet` patterns with properties (generated from ICU4J,
  `tools/GenIcuProperties.java`), `RuleBasedBreakIterator` over `.brk` with
  the Thai, Lao, Khmer, Burmese and CJK dictionary engines, and collation:
  the root and every locale tailoring ICU ships (`coll.pack.z`,
  `tools/GenIcuCollPack.java`), collation elements, sort keys byte for byte,
  `Collator.getInstance` with ICU's locale fallback and attribute keywords;
  and transliteration: every rule-based transliterator ICU ships
  (`translit/root.res`, in the same pack), compound/filtered/inverse IDs,
  `Any-<script>`, normalization, case (`ucase.icu`), `Null`/`Remove`, the
  Thai word-break inserter and `createFromRules`. Not ported: incremental
  transliteration, `Hex`/`Name` escapes and IDs that name a locale
  (`el-Latin`, `Any-am_FONIPA`; typed `UnsupportedOperation`).
- **Lucene's classes** -- `ICUNormalizer2Filter`, `ICUFoldingFilter`,
  `ICUNormalizer2CharFilter`, `ICUTokenizer` (all four configurations,
  `ScriptAttribute`), `ICUCollationKeyAnalyzer`, `ICUCollationAttributeFactory`,
  `ICUCollationDocValuesField` (the bytes; the `Field` is `lucene-search`'s),
  `ICUTransformFilter` and the factories. Not ported: `ICUTokenizerFactory`'s `rulefiles` (ICU's
  break-rule compiler; refused with `UnsupportedOperationException`) and
  collators from rules (`new RuleBasedCollator(rules)`, ICU's
  `CollationBuilder`).
- **Fixtures** -- `GenAnalysisIcu.java`: normalization of every assigned
  block and 600 stress strings through six normalizers x four modes, 690
  `UnicodeSet` patterns, the filters' and factories' chains, `ICUTokenizer`
  over 379 lines in four configurations (155,649 rows);
  `GenAnalysisIcuCollation.java`: 1,687 texts through 1,829 collators (every
  bundle and type, fallback IDs, keywords, setters), full keys for seven,
  Lucene's analyzer and field bytes; `GenAnalysisIcuTransform.java`: every
  transliterator ID ICU4J lists and 88 more specs over 517 texts, 41 rule
  sets both ways, 12 `ICUTransformFilter` chains -- all equal,
  byte-identical under JDK 21 and 25.
- **Thai over ICU** -- see T12.6 above: ICU's Thai breaks are not the JDK's,
  so `ThaiTokenizer` stays not supported.
- **Bench** (`--bench analysis_icu`, 2026-10-08, noise floor 1.13x):
  icu_normalizer_charfilter 1.53x, icu_normalizer_nfkc_cf 0.90~, icu_folding
  0.85x, icu_tokenizer 0.67x, icu_collation_key 0.60x,
  icu_collation_key_phonebook_identical 0.83x; icu_transform_any_latin
  1.11~, icu_transform_trad_simp 0.58x (noise 1.24x); the profiles and the
  optimisation in progress are in `docs/parity/analysis-icu.md`.

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
