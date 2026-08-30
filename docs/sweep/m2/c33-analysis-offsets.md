# c33-analysis-offsets

The **producer** half of the offset-unit defect c23 shipped and c29 half-fixed:
`OffsetAttribute` carries offsets in Java `char`s (UTF-16 code units) and this
port's analysis emitted UTF-8 **byte** offsets. c29 made the *reader*
(`lucene-search`'s highlighter) Java-`char`-based and handed the writer side
over with the exact call sites; c23 had meanwhile made `IndexWriter` write
offsets, so this port was recording offsets into `.pos`/`.pay`/`.tvd` that real
Lucene mis-slices for any non-ASCII document.

Java read from **`/home/tuong/work/lucene-10.5.0`** throughout.

Files: `crates/lucene-analysis/src/lib.rs`,
`crates/lucene-analysis/tests/analysis_fixtures.rs`,
`fixtures/src/GenAnalysis.java`, `fixtures/data/analysis/manifest.properties`.
Consumer/evidence changes required by the fix, in other batches' files:
`crates/lucene-search/src/highlighter.rs` (the compensating conversion comes
out), `crates/lucene-index/src/{indexing_chain,index_writer}.rs` (a
pass-through test + doc), `crates/lucene-codecs/src/{block_packed,
term_vectors}.rs` and `crates/lucene-codecs/examples/
write_term_vectors_fixture.rs` (a newly-reachable negative-value path, and a
real-Lucene read-back of non-ASCII offsets), plus `docs/parity.md` and
`fixtures/README.md`.

**Two prior attempts at this batch were killed by machine crashes mid-flight,
so the working tree already carried most of the producer fix when this run
started.** That state was not taken on trust: the fixture manifest was
regenerated from the real Lucene 10.5.0 jars and diffed (identical), the gate
was run (it was **red** — finding 5), and the codec-level consequence of the
unit change was chased down and pinned (findings 7 and 8, both new this run).

---

## `crates/lucene-analysis/src/lib.rs`

Java counterparts:

- `lucene/core/src/java/org/apache/lucene/analysis/tokenattributes/OffsetAttribute.java`
  + `OffsetAttributeImpl.java` (the unit itself)
- `lucene/core/src/java/org/apache/lucene/analysis/Tokenizer.java`
  (`correctOffset`)
- `lucene/core/src/java/org/apache/lucene/analysis/standard/StandardTokenizer.java`
- `lucene/analysis/common/src/java/org/apache/lucene/analysis/core/KeywordTokenizer.java`
- `lucene/analysis/common/src/java/org/apache/lucene/analysis/ngram/{NGramTokenFilter,EdgeNGramTokenFilter}.java`
- `lucene/analysis/common/src/java/org/apache/lucene/analysis/synonym/SynonymGraphFilter.java`
- `lucene/analysis/common/src/java/org/apache/lucene/analysis/miscellaneous/ASCIIFoldingFilter.java`
- `lucene/analysis/common/src/java/org/apache/lucene/analysis/en/PorterStemFilter.java`

| Rust | Java | verdict (offset unit only — the rest of these functions was swept by b8) |
|---|---|---|
| `Token.start_offset`/`end_offset` | `OffsetAttribute.startOffset()`/`endOffset()` | divergent (UTF-8 bytes) → **fixed** |
| `tokenize` | `StandardTokenizer.incrementToken` → `offsetAtt.setOffset(correctOffset(...))` | divergent → **fixed** |
| `utf16_len` | `String.length()` | **added** (no Java counterpart — Java's unit *is* its string length) |
| `Analyzer::keyword` | `KeywordTokenizer.incrementToken`'s `finalOffset = correctOffset(upto)` | divergent → **fixed** |
| `apply_ngram_filter` | `NGramTokenFilter.incrementToken`'s `restoreState(state)` | identical (copies the input token's offsets; unit-agnostic) |
| `ngrams_for_term` | `Character.codePointCount`/`offsetByCodePoints` | identical — slices by **code point** while *reporting* code units |
| `SynonymFilter::apply`/`apply_bidirectional`/`apply_multiword` | `SynonymGraphFilter`'s buffered outputs | identical (offsets copied from input tokens) |
| `AsciiFoldingFilter::apply`, `PorterStemFilter::apply`, `SnowballEnglishStemFilter::apply`, `LowerCaseFilter::apply`, `StopFilter::apply` | the corresponding `TokenFilter.incrementToken`s, none of which call `setOffset` | identical (offsets untouched, term length changes) |
| — | `CharFilter.correctOffset` (the offset-remapping half of `correctOffset`) | not ported: this crate has no `CharFilter`, so `correctOffset` is the identity, as it is for a bare `Tokenizer` |
| — | `OffsetAttribute.setOffset`'s negative/`start > end` `IllegalArgumentException` | not-in-Java-shape: no attribute object exists here to guard; the invariant is enforced where it matters, by `IndexingChain`'s own check (ported in `lucene-index`) |

### 1. `[CORRECTNESS → fixed]` `tokenize` emitted UTF-8 byte offsets

Java's `OffsetAttribute` indexes the original `String`, i.e. UTF-16 code
units. `unicode_word_indices` hands back **byte** indices and this port used
them verbatim, so the three units in play (bytes / Unicode scalars / Java
`char`s) coincided only for ASCII — which every fixture in the repo was, which
is why nothing caught it. An emoji is 1 scalar, 2 code units and 4 bytes, so it
shifts every later term by a different amount in each unit.

Consequence (not hypothetical since c23): `indexing_chain` forwards these
offsets into `.pos`/`.pay`/`.tvd` verbatim and `IndexWriter` writes them, so a
Rust-written index of non-ASCII text told real Lucene that a term sits where it
does not. `CheckIndex` never compares an offset against the text it indexes —
it only checks ordering and range — so the index is structurally perfect and
semantically wrong.

Fixed in `tokenize`: one running conversion over the text (the segmenter yields
segments in ascending byte order, so each token costs one `utf16_len` over the
gap plus one over the token — O(n) over the document, not O(n) per token),
behind a whole-text `is_ascii()` fast path where the byte index *is* the
`char` index. `utf16_len` has its own ASCII fast path (`str::is_ascii` is a
word-at-a-time scan; the byte length is the code-unit length) so only genuinely
non-ASCII text pays the per-scalar `len_utf16` sum.

### 2. `[CORRECTNESS → fixed]` `Analyzer::keyword` ended its token at the byte length

Java's `KeywordTokenizer` reads into a `char[]` and ends the single token at
`correctOffset(upto)`, where `upto` counts `char`s. This port used
`text.len()` — UTF-8 bytes. `"id-<emoji>-é"` reported an end offset of 10 where
Lucene reports 7. Fixed with `utf16_len(text)`.

### 3. `[CORRECTNESS → fixed]` the consumer that compensated — and had to stop

**This is the "double fix" check the batch was asked to make, and it found
one.** c29 could not fix the producer (different files), so
`highlighter::offsets_from_analysis` converted UTF-8 bytes → UTF-16 at the
boundary, documented as a boundary conversion rather than a fix. Leaving that
in after this batch would shift every non-ASCII highlight *the other way*: a
compensating conversion outliving its cause is exactly as wrong as the original
defect, and harder to find.

The conversion is gone, and the test that pins it asserts the two sides are
**identical** (`offsets_from_analysis`' output equals the analyzer's raw
offsets), not merely that the highlight comes out right — an equality that
fails if either side is ever converted again. It also checks the rendered
fragment for `"café naïve dog"` and for an astral case where Java `char`s,
scalars and bytes all differ (`"beta"` at char 9, scalar 8, byte 11).

### 4. `[CORRECTNESS → fixed]` the *test file* compensated too

`analysis_fixtures.rs` ran every real-Lucene expectation through a
`char_offsets_to_byte_offsets` helper before comparing. That is a second
compensation, in the one place whose job is to detect this class of defect: it
made the wrong unit invisible, and it was itself wrong for supplementary-plane
text (it converted *scalars* to bytes, so it silently mis-stated astral cases
in either direction). It is deleted; every offset is now compared verbatim, and
there is deliberately no conversion helper left in the file for a wrong unit to
hide behind.

### 5. `[CORRECTNESS → fixed]` the interrupted attempt left the gate red

The new `indexing_chain` pass-through test asserted `"omega"` at **position 4**
in `"alpha café 世 𝌆 omega"`, expecting `U+1D306` to consume a position. It does
not: it is a symbol, so neither real Lucene's `StandardTokenizer` (the new
`utf16_astral_symbol` fixture case records exactly two tokens, one position
apart across it) nor this port emits a token for it, and a skipped run of text
consumes no position. The expectation is now 3, with the reasoning and the
fixture reference in the test — and it is a *positions* assertion sitting next
to the offsets one on purpose, since "the unit change does not touch positions"
is one of this batch's claims.

### 6. `[MISSING → fixed]` no fixture separated the three units

Every analysis fixture was ASCII or near-ASCII, so at most two of the three
units ever differed. `GenAnalysis.java` gained **12 `utf16_*` cases**, all
ground truth from real Lucene 10.5.0 analyzers, covering each producer in
scope:

| case | text shape | producer |
|---|---|---|
| `utf16_latin1` | Latin-1 accented letter (1 `char`, 2 bytes) | `StandardTokenizer` |
| `utf16_cjk_offsets` | CJK ideographs (1, 3) | `StandardTokenizer` |
| `utf16_combining_mark_offsets` | decomposed combining mark (2, 3) | `StandardTokenizer` |
| `utf16_astral_symbol` | astral symbol (2 `char`s, 1 scalar, 4 bytes), untokenized | `StandardTokenizer` |
| `utf16_astral_letter` | astral *letter* — a token whose span is 2 `char`s per scalar | `StandardTokenizer` |
| `utf16_emoji` | emoji, and what it shifts | `StandardTokenizer` |
| `utf16_all_units` | all five in one string | `StandardTokenizer` |
| `utf16_keyword_astral` | `id-<emoji>-é` | `KeywordAnalyzer` |
| `utf16_fold_after_astral` | `ß`→`ss` behind an astral letter | `ASCIIFoldingFilter` |
| `utf16_porter_after_astral` | `running`/`fishes` behind an astral letter | `PorterStemFilter` |
| `utf16_ngram_offsets` | `café 𝐀𝐁cd` | `NGramTokenFilter` |
| `utf16_edge_ngram_offsets` | `𝐀bc déf` | `EdgeNGramTokenFilter` |
| `utf16_syn_multiword` | `𝐀 wi fi 世` | `SynonymGraphFilter` |

Two properties make these evidence rather than decoration:

- **Negative controls.** `assert_defeats_byte_offsets` /
  `assert_defeats_scalar_offsets` recompute, from the fixture text itself, what
  a byte-offset and a scalar-offset producer would have reported, and assert
  the case *disagrees* with them. A case that defeats neither is evidence about
  nothing; `utf16_all_units` defeats both in a single case.
- **The manifest is reproducible.** These are plain-text keys, so
  `scripts/gen-fixtures.sh --only GenAnalysis --out <scratch>` regenerates them
  from the real jars; this run diffed the result against the committed file and
  it is **identical**, which is the check that distinguishes real-Lucene ground
  truth from a plausible hand-edit. (`--only`, never a full run: c29 triggered
  a full regeneration once and had to restore 366 files.)

Two behaviours worth recording because the fixtures pin them and they read like
bugs otherwise: the n-gram filters slice by **code point** while *reporting*
code units (the first 2-gram of `𝐀bc` is 2 code points and 4 `char`s), and
`SynonymGraphFilter`'s collapsed match spans the first matched token's
`startOffset` to the last one's `endOffset` while the originals keep their own —
b8's fix for a **decreasing** `startOffset`, which `IndexingChain` rejects
outright. The synonym test asserts non-decreasing `startOffset`s directly, so
the unit change cannot regress it silently.

### 7. `[CORRECTNESS in the record → fixed]` the unit change makes a "latent" encoder path live

`block_packed::encode_all`'s doc comment said its negative-value handling was
latent because "no current caller feeds negatives (term-vector lengths,
frequencies, positions and offsets are all non-negative)". **That is false as
of this batch.** `Lucene90CompressingTermVectorsWriter.flushOffsets` stores a
length as `(endOffset - startOffset) - prefixLength - suffixLength`, i.e. the
occurrence's span minus the term's length *in UTF-8 bytes* (Java subtracts a
`BytesRef` length from a `char` span; its reader adds the same quantity back —
verified in both `flushOffsets` and the reader's delta-decode loop, and this
port does the same on both sides). With byte offsets the two were equal and the
value was always 0 for a real token; with Java `char` offsets **every
multi-byte term produces a negative length** (`café`: a 4-`char` span over 5
bytes → −1; `世` → −2).

The encoder is correct (a previous batch had fixed the min-value framing), but
nothing exercised it from this direction. Fixed: the comment now states the
live caller, and `term_vectors.rs` gained a round-trip test over `café`/`世界`
with Java `char` offsets that asserts the exact offsets come back — an
assertion the pre-fix encoder (which truncated negatives to the low
`bitsPerValue` bits) fails.

### 8. `[MISSING → fixed]` real Lucene had never read a non-ASCII offset this port wrote

c29's handoff asked for this explicitly. Every write-path verifier case was
ASCII, so the whole non-ASCII offset path — the negative length delta of
finding 7 included — was evidenced only by this port's own reader agreeing with
its own writer, the evidence shape that let b4's FST framing and b11's invented
`.si` sort encoding round-trip perfectly while being wrong.

`write_term_vectors_fixture`'s multi-chunk segment `_2` gained **document 400**:
four non-ASCII terms (`café`, `dog`, `世`, `界`) carrying the Java `char`
offsets a real `StandardAnalyzer` reports for `"café 世界 dog"`, with the
field's flags unchanged so the chunk still takes `flushFlags`'
non-changing-flags encoding. `VerifyTermVectors` needs no change (it walks
`0..max_doc` comparing rendered documents), and real Lucene 10.5.0 reads the
document back with **exactly** those offsets.

Negative control, run: rewriting that one manifest line with the UTF-8 **byte**
offsets the pre-c33 producer would have emitted fails with
`MISMATCH _2 doc 400`, so the case genuinely discriminates the unit rather than
merely passing.

### 9. `[PERF → measured]` what the unit costs

`crates/lucene-analysis/examples/c33_offset_ab.rs` is an interleaved A/B (arms
alternate by round, min-of-25) between the shipped producer and the pre-c33
byte-offset one, which differ in exactly one thing. 200 documents of ~1.4 KB;
three runs:

| corpus | UTF-16 (shipped) | bytes (pre-c33) | delta |
|---|---|---|---|
| ASCII body, 192 tokens/doc | 4.93 / 4.99 / 4.96 µs/doc | 5.09 / 5.04 / 5.13 | **−0.05 … −0.17 µs/doc** (free; the sign is negative in all three runs, i.e. below this harness's noise) |
| ASCII + one accented word, 193 tokens/doc | 13.70 / 13.66 / 13.57 | 13.53 / 13.55 / 13.57 | **+0.00 … +0.17 µs/doc (+0.0 … +1.2%)** |
| Latin-1 + CJK + astral, 128 tokens/doc | 8.77 / 8.78 / 8.75 | 8.60 / 8.62 / 8.59 | **+0.16 … +0.17 µs/doc (+1.8 … +2.0%)** |

Reading: **ASCII text pays nothing** — the `is_ascii()` fast path is a
word-at-a-time scan whose cost disappears into the segmenter's own work, and no
per-token conversion happens at all. Non-ASCII text pays ~0.17 µs per 1.4 KB
document, ~1-2% of tokenization and **under 1% of the ~21 µs/doc this port
spends indexing a document**. The middle row is deliberately the fast path's
worst case (one accented character makes a whole ASCII document take the slow
path); the honest reading of it against the first row is that non-ASCII text is
2.7x more expensive to *segment*, and the offset conversion is a rounding error
next to that.

Better or worse than Java: Java has no conversion at all — its `char[]` buffer
*is* the unit — so this is a cost Java does not pay. The alternative that would
match Java (carry the text as UTF-16) would be far worse everywhere else in a
Rust port whose strings are UTF-8. The chosen shape confines the cost to
non-ASCII documents and keeps it O(n) in the text rather than O(tokens × n).

### 10. `[INTENTIONAL]` the filters that needed no change, pinned anyway

`NGramTokenFilter`, `EdgeNGramTokenFilter`, all three `SynonymFilter` entry
points, `AsciiFoldingFilter`, `PorterStemFilter`,
`SnowballEnglishStemFilter`, `LowerCaseFilter` and `StopFilter` all *copy* the
input token's offsets (Java `restoreState`s or leaves `OffsetAttribute`
untouched), so they were correct in whatever unit the tokenizer produced. They
are pinned by fixtures anyway, because "the filter copies its input's offsets"
is a claim that a term-length-changing filter can silently break — the fold and
stem cases put the length change *behind* a supplementary-plane token so a
producer that re-derived the span from the rewritten term is visibly wrong at
both ends.

`position_increment`/`position_length` are unaffected: every fixture comparison
is over `(term, position_increment, start_offset, end_offset)` tuples, the
synonym-graph cases carry `position_length` as well, and finding 5's test
asserts a specific position next to the offsets.

### 11. `[MISSING → recorded]` real Lucene tokenizes an emoji; this port does not

Not new (b8's F40) but newly *measurable*: `StandardTokenizer` emits an emoji as
its own token, and `unicode_word_indices` emits no token for any run without
alphanumerics. The `utf16_emoji` case records the exact shape of the gap rather
than hiding it — the tokens this port *does* produce must match Lucene's
verbatim, offsets included, and the count of dropped tokens must be exactly one.
The consequence is that positions after an emoji are one lower here than in
Lucene; closing it needs `split_word_bounds` plus an Extended_Pictographic pass,
i.e. a tokenizer rewrite, not an offset fix.

### 12. `[MISSING → recorded, other batch]` `invertState.offset` accumulation

`IndexingChain.processField` does `invertState.offset += offsetAttribute
.endOffset()` after each value of a multi-valued field, so the second value's
offsets continue where the first ended. `invert_documents*` restarts at each
`(doc, field, text)` triple. It is currently **unreachable**: `IndexWriter`'s
`build_*` helpers select a field's value with `.find(...)`, so only the first
value of a repeated field is indexed at all. Recorded for whoever adds
multi-valued fields (b9/c23's files, not this batch's); it is a positions
problem as much as an offsets one (`positionIncrementGap`/`offsetGap` are
unported too).

### Verdict

Swept clean. c23's F13 and c29's §2.2 handoff are closed on the producer side,
the consumer compensation c29 installed is removed with an equality test that
keeps it removed, and the unit is pinned at both ends: against real Lucene's own
analyzers where it is produced, and against real Lucene's own term-vectors
reader where it is written.

---

## `crates/lucene-analysis/tests/analysis_fixtures.rs` + `fixtures/src/GenAnalysis.java`

No Java counterpart (test harness + fixture generator). Covered by findings 4
and 6. 19 fixture tests, 7 of them new `utf16_*` ones; the file now contains no
offset-conversion helper at all, only the two negative-control predicates that
prove each case separates the units it claims to.

### Verdict

Swept clean; manifest verified reproducible from the real 10.5.0 jars this run.

---

## Gates

```
cargo fmt -p lucene-analysis
cargo clippy -p lucene-analysis --all-targets --jobs 2 -- -D warnings
cargo test -p lucene-analysis -p lucene-index -p lucene-search --jobs 2 -- --test-threads=4
```
all green (40 test binaries, 0 failures). `lucene-codecs` (touched for findings
7 and 8): `cargo fmt`, `term_vectors::`/`block_packed::` tests green, and
clippy clean **for the files this batch touched**.

> Observation, not a finding of this batch: `cargo clippy -p lucene-codecs
> --all-targets` currently reports 197 `arithmetic_side_effects` errors in
> `fst.rs`, `postings_writer.rs` and `vectors.rs` — files this batch does not
> touch, left mid-flight in the shared working tree by other in-progress
> batches. Named here so the next batch to run that command does not attribute
> them to c33.

Coverage (`cargo llvm-cov -p <crate> --summary-only`):

| file | lines |
|---|---|
| `lucene-analysis/src/lib.rs` | **99.27%** (regions 99.13%, functions 100%) |
| `lucene-codecs/src/block_packed.rs` | 98.68% |
| `lucene-codecs/src/term_vectors.rs` | 97.98% |

All at or above the 95%-per-file bar.
