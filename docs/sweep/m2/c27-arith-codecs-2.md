# c27 — the arithmetic gate, burned down across the seven reachable `lucene-codecs` modules

Follow-up to c24, which audited 14 modules and left **12** carrying a
`#[allow(clippy::arithmetic_side_effects)] // TODO(arith-audit)` marker on
their `mod` declaration in `lib.rs`. This batch audits the **seven c24 named
as reachable** — the modules that parse the metadata of every index this port
opens: `doc_values`, `blocktree`, `postings`, `stored_fields`,
`term_vectors`, `points`, `for_util`.

Java read from **`/home/tuong/work/lucene-10.5.0`**, the pinned tag.

## The scope correction that shaped this batch

Mid-batch, c25 reported that its re-signed `.tvd` byte-flip sweep found two
defects in an **already-lint-clean** file: a `prefixLength` off disk slicing
the previous term (panic), and a claimed decompressed length sizing
`vec![0u8; n]` (SIGABRT at ~1 PB). Its diagnosis is the one that matters:

> `clippy::arithmetic_side_effects` covers arithmetic and shifts — **not
> indexing, not slicing, not allocation.**

So this batch did not treat a module as audited when the lint went quiet.
Each module also got a hand-check of every `slice[i]` / `&slice[a..b]` whose
index came off disk, every `Vec::with_capacity(n)` / `vec![0; n]` whose `n`
did, and a **re-signed byte-flip sweep** over the files it parses. The
sharpest finding of the batch came out of that half, not the lint half:
`doc_values::sorted_numeric_values` reproduced as a **SIGABRT**, not a wrong
answer.

## Burn-down

| | count |
|---|---|
| modules carrying the marker after c24 | 12 |
| **audited this batch** | **7** |
| lint sites resolved (lib, non-test) | **679** of the 949 remaining |
| **remaining marked** | **5** (`fst`, `hnsw`, `hnsw_vectors`, `postings_writer`, `vectors`) |

`docs/arithmetic-gate.md`'s table updated 12 → 5.

**Findings: 55 `CORRECTNESS`/`MISSING`, all fixed; 5 `PERF`; 8
`INTENTIONAL`.** Of the 55, **9 were found by the non-lint half of the audit**
— indexing, slicing and allocation, which `clippy::arithmetic_side_effects`
does not cover — and those include every one of the batch's four verified
`SIGABRT`s and its one release-mode infinite loop. Two more came from the
Tier-2 review of this batch's own work.

| what it does when it fires | count |
|---|---|
| **abort** (allocation failure; `catch_unwind` cannot intercept it) | 4 verified under `ulimit -v`, plus 6 more of the same shape |
| **hang** (`ForUtil::encode` at width 0; worse than an abort — no timeout) | 1 |
| **silent wrong answer** in a release build | 8 |
| **panic** (debug, or an index/slice in both) | the rest |

Every fix has a test that fails against the unfixed code, except nine
explicitly listed as defensive hardening whose trigger needs a segment near
`i32::MAX` documents or a 32-bit target.

| Rust file | Java counterpart (10.5.0) | sites |
|---|---|---|
| `for_util.rs` | `codecs/lucene104/ForUtil.java`, `ForDeltaUtil.java`, `PForUtil.java` | 166 |
| `points.rs` | `util/bkd/BKD{Reader,Writer,Config}.java`, `codecs/lucene90/Lucene90Points{Reader,Writer}.java` | 144 |
| `blocktree.rs` | `codecs/lucene103/blocktree/*` | 93 |
| `stored_fields.rs` | `codecs/lucene90/compressing/Lucene90CompressingStoredFields{Reader,Writer}.java`, `FieldsIndex{Reader,Writer}.java` | 83 |
| `term_vectors.rs` | `codecs/lucene90/compressing/Lucene90CompressingTermVectors{Reader,Writer}.java` | 72 |
| `doc_values.rs` | `codecs/lucene90/Lucene90DocValues{Producer,Consumer}.java` | 70 |
| `postings.rs` | `codecs/lucene104/Lucene104Postings{Reader,Format}.java` | 51 |

---

## Findings

One section per module, each naming its Java counterpart, its findings and
its byte-flip sweep result.

## `term_vectors.rs`

Java: `codecs/lucene90/compressing/Lucene90CompressingTermVectorsReader.java`,
`...Writer.java`.

72 lint sites: **11** resolved with `checked_*`/a typed rejection, **3** with
`wrapping_*` where Java's `int` arithmetic wraps and the reader replays it,
**58** with a tightly-scoped `#[allow]` carrying an `// ARITH:` proof.

### F1 `[CORRECTNESS]` `read_chunk` — `totalDistinctFields` accumulated into a `u32`, sizing a ~16 GB `vec![0u8; n]`

The instance c24 flagged live at `term_vectors.rs:470`. The count arrives as a
3-bit inline field with a vint escape; `read_vint` returns `i32`, so a negative
escape became ~4 billion through `as u32`, the `+=` overflowed (debug) or
wrapped (release), and `packed_ints::byte_count` then sized `vec![0u8; n]` at
up to ~16 GB — the **abort** shape.

Fixed: `i64` throughout, a negative escape rejected by name, and the count
bounded twice over — by the writer's own invariant
(`Lucene90CompressingTermVectorsWriter.flushFieldNums` emits the
*deduplicated* field numbers of the chunk's fields, so there are never more
distinct numbers than field instances) and by the bytes the packed array must
occupy in the stream. Tests:
`negative_distinct_field_count_extension_is_a_decode_error_not_an_allocation`
(reverted → `attempt to add with overflow` at the `+=`) and
`more_distinct_field_numbers_than_fields_is_a_decode_error` — the positive,
plausible case a sign check and an EOF would both miss.

### F2 `[CORRECTNESS]` `read_chunk` — `chunkDocs` was decoded with a *signed* shift

`Lucene90CompressingTermVectorsReader:380` is
`vectorsStream.readVInt() >>> 1`; the writer's `copyChunks:858` is
`code >>> 1`. Both were ported as `>>`. The difference only shows for a
negative vint, i.e. only on a corrupt `.tvd` — but it is the difference
between `chunk_docs` being an unconstrained negative and being a number as
large as `i32::MAX`, which is what makes `docBase + chunkDocs` genuinely
overflow. Found by taking `blocktree`'s F4 (the same shape in
`SegmentTermsEnumFrame`'s `(start + end) >>> 1`) back through this file.

### F3 `[CORRECTNESS]` `read_chunk`/`copy_chunks` — `docBase + chunkDocs` overflowed the guard that was meant to catch it

With F2 fixed, a corrupt token names `chunkDocs = 0x7FFF_FFFF`, and a chunk
whose `docBase` is anything but 0 overflows the sum. Java's is an `int` add
that wraps; here it panicked in a debug build, and in a release build the wrap
carried a corrupt pair **past all three guards** — `doc_base = 128` with
`chunk_docs = i32::MAX` wraps to a small negative, which is `<= max_doc`. The
bulk-copy loop was worse: it advanced both `doc` and the writer's own
`num_docs` with a plain `+=` *before* testing `doc > to_doc`, so the wrap put
the cursor back inside the range and copied a chunk body two billion documents
long. Both now `checked_add` → `CorruptChunkBounds`. Tests:
`a_chunk_claiming_two_billion_documents_is_a_decode_error_not_an_overflow`,
`copy_chunks_rejects_a_chunk_claiming_two_billion_documents`.

### F4 `[CORRECTNESS]` `read_chunk` — `totalFields` was a sum of unbounded `i64`s sizing six reservations

`numFieldsPerDoc` is `block_packed`-decoded, so an entry is whatever the chunk
body says. A negative one turned `total_fields` into a huge `usize` through
the `as` cast, and that number then sized six `Vec::with_capacity` calls — an
abort. The sum could also overflow `i64` outright. Both rejected once, outside
every loop that consumes the result, and the reservation is now sized from the
vector that was actually decoded rather than from `chunk_docs`. `total_fields`
itself is capped by `input.remaining() * 8`: every field costs at least one
bit in the `allFieldNumOffs` array that follows
(`direct_writer_bits_required` never returns 0), and that array lies inside
the bytes still ahead of the cursor — so the input is a hard ceiling and a
well-formed 4 KB chunk loses nothing (it admits 32 768 fields against Java's
per-chunk maximum of 128 documents × their field counts). Test:
`absurd_per_document_field_count_is_a_decode_error_not_an_allocation`, both
the negative and the merely-enormous case.

### F5 `[CORRECTNESS]` `read_chunk` — `numTerms` was unbounded, and `term_offsets` non-monotonic

`num_terms_bits` can name 64 bits, so an entry is an arbitrary `i64`. A
negative one made `term_offsets` non-monotonic (and its `as usize` casts
astronomic), which is what the three `block_packed` streams are sliced by; the
running sum could also overflow. Both rejected in the accumulation loop, which
is also where the running sum is built — so `total_terms` costs one pass, not
two. Test: `absurd_per_field_term_count_is_a_decode_error_not_an_allocation`.

### F6 `[CORRECTNESS]` `read_chunk` — `freq = termFreqMinus1 + 1` overflowed, and a negative frequency poisoned every stream length

`term_freqs_minus1` is block-packed, so `v + 1` overflows at `i64::MAX` and a
negative frequency makes `total_positions`/`total_offsets`/`total_payloads`
useless as stream lengths. Factored into `sum_freqs`, checked once per term,
outside every consumer. Test:
`absurd_term_frequency_is_a_decode_error_not_an_overflow`.

### F7 `[CORRECTNESS]` `read_chunk`/`build_field` — three `.max(0)`s turned a rejection into a plausible wrong term

c25 added the LZ4 expansion ceiling that bounds `vec![0u8; decompressed_len]`.
Its two sums used `.map(|&v| v.max(0))`, which silently *accepted* a negative
length and then let the per-document byte cursor add it as a huge `usize`,
indexing the decompressed buffer from nowhere. A byte length off disk is never
negative in anything Lucene wrote, so a negative one is now rejected outright
— which is what makes every partial sum below it a prefix of a total that has
already been bounded, and lets the whole per-document cursor loop carry one
`ARITH:` proof pinned by a `debug_assert_eq!(cursor, decompressed_len)`.
Deliberately not `saturating_add`: a saturated total would still be rejected
by the ceiling, but it would report a length nobody wrote.

**The Tier-2 review found the sharper half of this, which the first pass
walked past**: `build_field`'s `prefix_lengths` entry was *also* `.max(0)`-ed,
and `prefix_lengths` is the one stream `read_chunk` did not validate. Before
this batch a negative prefix sign-extended to `usize::MAX` and
`previous_term.get(..prefix_len)` returned `None`, i.e. a typed error. With
`.max(0)` it decoded as prefix 0 and the term was materialised **from its
suffix alone** — accepted, self-consistent, and wrong, where Java throws
(`System.arraycopy` with a negative `destPos`). Exactly what
`docs/arithmetic-gate.md` singles saturation out for. `prefix_lengths` is now
validated once per chunk alongside the other two, and all three `.max(0)`s
and the `.saturating_add(1).max(0)` in `build_field` are gone — replaced by
plain casts under a `debug_assert!`, so the next reader cannot conclude
negatives are tolerated there. Tests:
`negative_term_suffix_length_is_a_decode_error`,
`negative_term_prefix_length_is_a_decode_error_not_a_shorter_term`.

### F8 `[CORRECTNESS]` `open` — `numChunks + 1`, and three counts Java only *asserts* are non-negative

`numChunks + 1` is a `long` add in Java. Here it panicked in a debug build for
a `.tvm` vlong near `i64::MAX`. Separately, Java asserts `numChunks >= 0`,
`numDirtyChunks >= 0` and `numDirtyDocs >= 0` in its three getters — and
assertions are not enough, because a negative `numDirtyChunks` slips through
all four cross-checks (it is "nonzero", it is not greater than `numChunks`,
and it is not greater than a positive `numDirtyDocs`) and then reaches
`too_dirty`'s `* 100`, which underflows at `i64::MIN`. Made real checks, which
is also what turns `too_dirty`'s multiply into provably-bounded arithmetic
rather than a fourth `checked_mul`. Tests:
`a_num_chunks_at_i64_max_is_a_mismatch_not_an_overflow`,
`a_negative_dirty_chunk_count_is_rejected_at_open`.

### F9 `[CORRECTNESS]` `open` — `maxPointer as usize + FOOTER_LENGTH`

A raw `i64` off `.tvm`: `as usize` sign-extends a negative one into a huge
length and the add then overflows. Folded into one fallible expression, and
the length equality that follows now carries the *upper* bound too — so from
`open` onward `max_pointer` is known to lie in `0..=tvd.len()`, which is what
makes every later `max_pointer as usize` (`check_integrity`'s and
`copy_chunks`') an in-bounds offset by construction. Test:
`a_negative_max_pointer_is_a_decode_error_not_an_overflow`.

### F10 `[CORRECTNESS]` `read_chunk`/`document` — `fieldNumOff` indexed `field_nums` out of range

Java only *asserts* `fieldNumOff >= 0 && fieldNumOff < fieldNums.length`, so a
corrupt `allFieldNumOffs` entry is an `ArrayIndexOutOfBoundsException` in a
production JVM and was an index panic here — reachable from any `.tvd` byte
flip. Now a typed error, checked **once per chunk** rather than on the
per-document path, which also lets `DecodedChunk::document` carry a single
`ARITH:` proof instead of a bounds check per field. Test:
`a_field_number_offset_past_the_distinct_field_numbers_is_a_decode_error`.

### F11 `[CORRECTNESS]` `flush_offsets` — a `freq == 0` term underflowed `pos + freq - 1`

`validate_field` accepts a term with `freq == 0` and empty occurrence arrays,
and `charsPerTerm`'s accumulation then evaluates `pos + freq as usize - 1`,
which underflows when `pos` is 0. Java has the same shape and throws
`ArrayIndexOutOfBoundsException`. Guarded rather than proved: the term is
skipped, which keeps `charsPerTerm` a ratio of the occurrences that exist.

### F12 `[CORRECTNESS]` `add_document` — `numDocs` wrapped past `i32::MAX`

`.tvm` records `maxDoc` as an `i32`. Java writes `++numDocs` and leaves the
ceiling to `IndexWriter`'s document limit; here a wrap would emit a segment
whose `.tvx` disagrees with its own `maxDoc`. `add_document` already panics on
a caller wiring bug (`validate_field`), so it panics on this one too — stated
in its `# Panics` section — rather than silently writing a broken segment.

### INTENTIONAL — the writer's deltas are `wrapping_sub`

`endOffset - startOffset`, `position - previous`,
`startOffset - previousOff - correction` and
`length - prefixLength - suffixLength` are all `int` arithmetic in Java, and
the reader replays every one of them with `wrapping_add`. Wrapping is what
round-trips a caller's absurd offset pair, so these are `wrapping_sub` rather
than checks that would reject one.

### Byte-flip sweep

`every_resigned_single_byte_tvd_and_tvm_corruption_is_an_error_or_a_clean_decode`
flips bit 0 and bit 7 of every seventh `.tvd` chunk-region byte and of every
`.tvm` byte, re-signs the footer so the CRC cannot "catch" the corruption,
and requires a typed error or a clean decode from `open` plus a `ChunkCursor`
walk of **every** document.

**The first version of this sweep was measuring its fixture, and the Tier-2
review caught it.** It ran over the hand-built one-document, one-field,
two-term, single-chunk `.tvd` the rest of this module's error tests use, and
scored 260/452 — but a one-document chunk takes `read_chunk`'s
`chunk_docs == 1` shortcut (so it never reaches F4's `block_packed` per-
document field counts), has one distinct field number (so F10's range check
is vacuous), and gives the `.tvx` monotonic arrays nothing to discriminate,
which puts F3 out of reach entirely. That is the same error this batch had
already documented for `.fdm` — and then repeated one file over. The fixture
is now a **writer-produced 400-document, multi-chunk, two-field,
payload-and-offset-carrying** segment, asserted to have at least three chunks
and to not take the `chunk_docs == 1` path.

**688 of 3 724 rejected, zero panics, zero aborts.** The lower rate is the
fixture being right, not the decoder being weak: a real `.tvd` chunk is mostly
its LZ4 literal run — term suffixes and payload bytes — and flipping one of
those yields a different but perfectly well-formed vector, which is what the
checksum exists to catch. Compare `.kdd` 204/438 and `.fdt` 624/904, both for
the same reason, against `.fdm` 269/282 and `.tim`+`.tip`+`.tmd` 391/436.

---

## `doc_values.rs`

Java: `codecs/lucene90/Lucene90DocValuesProducer.java`,
`Lucene90DocValuesConsumer.java`.

70 lint sites: **2** resolved with a typed rejection, **68** with `ARITH:`
proofs — 48 of them on the encode half, where every operand is an in-memory
slice length or an offset recorded from an earlier `data.len()` of the same
append-only buffer, so the proofs sit at function scope with one shared
statement of that invariant.

### F13 `[CORRECTNESS]` `sorted_numeric_values` — a corrupt address range built an 8 TB `Vec`

**The sharpest finding of the batch, and the lint never saw it.** A
multi-valued field's address array names a `[start, end)` range in the field's
flat value array; both ends come out of a `direct_monotonic` array in `.dvd`
and nothing tied them to the `numValues` the entry declares. For a
*constant*-encoded field (`bitsPerValue == 0`) `decode_value` never indexes
anything and so never fails, and `(start..end).map(decode_value).collect()`
therefore builds one `i64` per element of whatever range the file named.

Reproduced as a real **SIGABRT** — `memory allocation of 4294967296 bytes
failed`, `signal: 6` — by running the unfixed code under `ulimit -v`. That is
the dead-JVM case: `catch_unwind` at the FFI boundary cannot intercept an
allocation failure. Fixed with a `CorruptAddressRange` error bounding the
range by `numValues`. Test:
`a_sorted_numeric_address_range_past_the_value_array_is_a_decode_error_not_a_terabyte_vec`,
which builds the address array through `direct_monotonic`'s own public writer,
so it is a range a real `.dvd` can carry rather than a hand-forged `Meta`.

### F57 `[CORRECTNESS]` `1i64 << shift` at the one value the range check lets through

**c24's own failure mode, reproduced verbatim, and caught only by this
batch's Tier-2 review.** The varying-bits-per-value mask was
`(1i64 << shift) - 1` under a proof reading "`read_numeric_entry` rejects the
entry unless `shift` is in `0..=63`". The cited range is exactly the range
that breaks it: `1i64 << 63` is `i64::MIN`, whose `- 1` overflows — a
debug-build panic. In release it wraps to `i64::MAX`, which happens to be
Java's answer, so the *conclusion* survived and the reasoning did not, which
is the shape to watch for: the next person to touch that code trusts the
comment.

It is file-reachable: `blockShift` is `-2 - tableSize` off `.dvm`, so a
`tableSize` of `-65` yields 63, and `read_numeric_entry` accepts it. A
single-bit flip cannot reach it from Java's only written value (`-16`), which
is why the sweep does not. Now computed in `u64` — exact for every accepted
shift, and the reinterpretation back to `i64` is the bit pattern Java's `long`
arithmetic produces — in one `low_bits_mask` helper with a `debug_assert!`
pinning the invariant. Test: `a_block_shift_of_63_masks_without_overflowing`,
which also asserts `parse_meta` really does accept `tableSize = -65`.

### F14 `[CORRECTNESS]` eight `offset + length` region slices off `.dvm`

`docsWithFieldOffset`/`docsWithFieldLength`, `valuesOffset`/`valuesLength`,
`dataOffset`/`dataLength` and both address arrays' `offset`/`length` are pairs
of *independent* `i64`s with nothing on the wire relating them —
`norms::read_value_at_ordinal`'s c24 shape, eight times over. `offset + length`
panics in a debug build and, in a release build, wraps to a *plausible
in-range* offset that decodes valid-looking doc values out of the wrong part
of `.dvd`. Folded into one `region()` helper (plus `slice_at` for the
`[start, end)` form), so there is one parser of this shape rather than eight.
Tests: `an_overflowing_values_region_is_a_decode_error_not_an_overflow`,
`an_overflowing_docs_with_field_region_is_a_decode_error_not_an_overflow`.

The variable-width BINARY path also *subtracted* two address-array `i64`s and
then added the difference straight back (`(start, end - start)` →
`start + len`); restructuring it to carry `[start, end)` removes both
operations rather than checking them.

### F15 `[MISSING]` no sparse `DenseField` variant for BINARY / SORTED / SORTED_NUMERIC / SORTED_SET

Handed over by c26: `execute_merge` **fails** a merge whenever a source
segment has no doc-values column for a field, for those four types, because
the writer side could not express a sparse column for them.
`DocValuesConsumer.getMerged*DocValues` simply drops that reader's sub — the
field is all-missing for those documents — so an ordinary index that starts
writing a doc-values field partway through its life was un-mergeable, and the
failure surfaced to the caller as an error on a perfectly valid index. c26
fixed the NUMERIC case, which `DenseField::SparseNumeric` (added by c17 for
the index-sort path) already supported.

Added `SparseBinary`, `SparseSorted`, `SparseSortedNumeric` and
`SparseSortedSet`, and rewrote the four `write_single_sparse_*` functions as
thin wrappers over `write_dense_fields` — the same precedent the five dense
ones already follow, so there is one encoder per shape rather than two.
Tests: `write_dense_fields_all_four_sparse_types_together_round_trip`,
`sparse_variants_reject_a_duplicate_or_out_of_range_doc_id`,
`sparse_multi_valued_fields_with_no_values_write_javas_empty_marker`.

**The "thin wrappers" description understates what the refactor changed**, as
the Tier-2 review pointed out. Routing the four single-field sparse writers
through the shared bodies also moved three bytes of the wire for a *genuinely*
sparse column: `indexed_disi::write` → `write_with_dense_rank_power(...,
DEFAULT_DENSE_RANK_POWER)` with the metadata byte `0xFF` → `9`; BINARY's and
SORTED_SET's address-array `block_shift` `0` → `DIRECT_MONOTONIC_BLOCK_SHIFT`
(16); and BINARY's empty-column `minLength` `0` → `i32::MAX`. All three move
*toward* Java (`IndexedDISI.writeBitSet(..., DEFAULT_DENSE_RANK_POWER)`,
`DIRECT_MONOTONIC_BLOCK_SHIFT`, and `addBinaryField`'s
`minLength = Integer.MAX_VALUE` initial value), so none is a defect — but they
are behavioural changes, not a pure factoring, and saying otherwise would be
the kind of quiet drift this sweep exists to find.

### F16 `[CORRECTNESS]` the four sparse writers never wrote Java's `-1` dense marker

Found while doing F15. `Lucene90DocValuesConsumer` writes the `-1`
docs-with-field marker whenever `numDocsWithField == maxDoc` —
`addBinaryField` inline (line 442) and the other four types through
`writeValues` (line 443). Only the NUMERIC sparse writer collapsed; the other
four emitted a full `IndexedDISI` region for a complete column. That is
readable, and round-trips through this port's own reader, which is exactly why
nothing caught it — but it is not a byte sequence real Lucene ever produces,
and the write-path verifier is the only thing that would ever notice. Now
shared in `write_docs_with_field`. Test:
`a_sparse_column_that_covers_every_document_writes_javas_dense_marker`, which
asserts the sparse and dense writers produce **identical bytes** for a
complete column — for all four types, not just the two the first version
covered.

### INTENTIONAL — a sparse SORTED_SET still never collapses to `multiValued = 0`

Real Lucene collapses a sorted set whose every doc has exactly one ordinal to
the plain `SortedEntry` shape. This port's `SortedSetKind::Multi` decodes
correctly either way, and `write_single_sparse_sorted_set_field` already
documented the divergence before it delegated to the shared body. Left as it
was, and recorded here so the docs-with-field collapse above is not mistaken
for the same decision.

### Byte-flip sweep

`every_resigned_single_byte_dvm_and_dvd_corruption_is_an_error_or_a_clean_decode`
sweeps `.dvm` and `.dvd` for all five doc-values types in their dense shape
**and four of them in their sparse shape** — six containers, and the sparse
one is the only path that emits an `IndexedDISI` region, so without it neither
the four new encoders nor the sparse *reader* (`DisiCursor` + rank ordinals)
got any coverage at all. That gap was the Tier-2 review's. **2 151 of 4 298
rejected (50%), zero panics, zero aborts.**

### Allocations sized off disk

Hand-checked, per the scope correction. Only two exist on the read side and
both are already bounded by a validated constant: the value table by
`tableSize <= 256`, and the skip-index level array by
`levels <= SKIP_INDEX_MAX_LEVEL` (4). `parse_skip_index`'s interval loop is
bounded by the byte window, each interval costing at least 29 bytes.

### PERF

`read_chunk`'s per-term work was measured A/B (min-of-40, three rounds, four
alternations — criterion is unusable here, and this machine is bimodal: the
*same binary* measured 2 371 ns and 4 753 ns in different runs). Alternating
A/B: 3 398/4 610, 4 647/3 256, 2 371/2 374, 4 682/4 524 ns. **Neutral** —
run-to-run variance dwarfs the change. In isolation the checked `sum_freqs`
loop is ~4x slower than the plain `map(|v| v + 1).sum()` it replaced
(0.047 vs 0.006 ns/value, the checked form defeating auto-vectorisation), but
c27 also **removed a second full pass** over the same array, and for a maximal
Lucene chunk the whole loop is tens of nanoseconds against a 2.4–4.7 µs chunk
decode. Every other check added is per-chunk or per-field, never per value.

---

## `blocktree.rs`

Java: `codecs/lucene103/blocktree/{Lucene103BlockTreeTermsReader, Lucene103BlockTreeTermsWriter, SegmentTermsEnum, SegmentTermsEnumFrame, IntersectTermsEnum, FieldReader, TrieReader, TrieBuilder}.java`.

93 lint sites: **7** `checked_*` reported as corruption, **2** `saturating_*`
(the `Intersect` skip heuristic's counters — saturation can only turn the
heuristic on or off, never change which terms are yielded), **84** proved.

### F17 `[CORRECTNESS]` `read_u64_at` — a self-defeating bound the whole trie decoder rests on

`if fp + 8 > slice.len()`. `rootFP` is a `.tmd` vlong cast with `as usize`, so
a negative one arrives as `usize::MAX`: `fp + 8` panics in debug and **wraps to
7 — passing the check** — in release, then panics on the slice index. This is
the shape to look for everywhere: any `a + b > len` guard whose `a` came off
disk forms the very sum it is meant to guard. Fixed with
`checked_add(8).filter(...)`, and the `fp <= len - 8` invariant every `fp + k`
offset in `load_node` depends on is now stated. Test:
`absurd_root_fp_is_a_decode_error_not_an_overflow`.

### F18 `[CORRECTNESS]` `open_shared` — `Vec::with_capacity(numFields)` straight off a vint

`(String, FieldTerms)` measures 232 bytes, so `i32::MAX` is a **498 GB**
reservation. Java presizes an `IntObjectHashMap` and survives with a catchable
`OutOfMemoryError`; an allocation failure here aborts. Bounded by
`tmd_input.remaining() / MIN_FIELD_RECORD_BYTES` (9, from the record's nine
one-byte-minimum values). Test:
`absurd_num_fields_errors_instead_of_reserving_for_it`. Honest caveat: Linux
overcommit lets the 498 GB mapping succeed on this box, so the unfixed run
fails on a later error rather than aborting; it aborts under
`vm.overcommit_memory=2` or a container cap.

### F19 `[CORRECTNESS]` `read_bytes_ref` — `minTerm`/`maxTerm` length unbounded

`vec![0u8; vint]`, up to 2 GB. Bounded by `input.remaining()`, which cannot
reject a Lucene-written file since Java's `readBytesRef` reads `numBytes`
immediately after. Test:
`absurd_min_term_length_errors_instead_of_allocating_for_it`.

### F20 `[CORRECTNESS]` `binary_search_term_leaf` — Java's `>>>` ported as `>>`

`SegmentTermsEnumFrame.java:701` is `int mid = (start + end) >>> 1;`. The
*unsigned* shift is load-bearing: it keeps the midpoint correct once the sum
passes `Integer.MAX_VALUE`. The port had `>>` — a debug panic, a negative
`mid` in release. Fixed as `((start as u32 + end as u32) >> 1) as i32`, with
the entry guard tightened so `start >= 0` is established rather than assumed.
This finding is what sent the rest of the batch looking for `>>>`, and it
turned up F2 in `term_vectors`, F32 in `points` and F36 in `stored_fields`.
Test: `binary_search_midpoint_does_not_overflow_for_a_huge_ent_count` — which
drives `Frame` directly and, **now that F22 is in, constructs an `ent_count`
`load_block` can no longer produce** (`ent_count == i32::MAX` would need a
2 GB suffix-lengths region). It is a sound unit-level guard on
`binary_search_term_leaf`, but F20 and F22 are not independent findings: F22
is what makes F20 unreachable from a file today.

### F21 `[CORRECTNESS]` `decode_meta_data` — `totalTermFreq = docFreq + readVLong()`

A wrapping `long` add in Java; in Rust a debug panic, and in release a
**negative** frequency every scorer downstream treats as real. Now
`checked_add` → `Corrupted`. Test: `total_term_freq_overflow_is_a_decode_error`.

### F22 `[MISSING]` `load_block` — `entCount` was bounded by nothing

`Lucene103BlockTreeTermsWriter.writeBlock` pushes exactly one vint per entry
into `suffixLengthsWriter` (`allEqual` included — it replicates the blob's
bytes rather than shortening it), so `numSuffixLengthBytes >= entCount` always
holds. Checked once per block load and hoisted out of every per-entry loop;
this is what bounds `ent_count` for the eight proofs below it. Test:
`ent_count_past_the_suffix_lengths_region_is_a_decode_error`.

### INTENTIONAL — `scan_to_floor_frame`'s `wrapping_add` became `checked_add`

Being honest: a vlong delta is at most `2^63 - 1` and `fp_orig` at most
`isize::MAX`, so on a 64-bit target it cannot actually wrap. No test, and no
defect claimed. The change removes a `wrapping_` that reads as if wrapping
were intended, when a wrap here would land on a valid-looking *different*
block.

### PERF — duplicate-field detection is `O(n^2)`, left alone

`fields.iter().any(...)` where Java uses a `HashMap`. For realistic field
counts (tens) the linear memcmp beats building a hash set on the 0.175 ms
segment-open path, and F18's `numFields` cap now bounds the corrupt-input
tail.

### Byte-flip sweep

`every_resigned_single_byte_terms_dict_corruption_is_an_error_or_a_clean_decode`
covers the `.tim` block body, the `.tip` trie region and the `.tmd` record
region, driving open + `seek_exact` for every term + `seek_ceil` + a full
`next()` walk. **391 of 436 rejected, zero panics, zero aborts.** It found no
new defects — F17–F22 all came from the hand audit and were fixed before it
ran. The `prefixLength` shape c25 found in `term_vectors` does not reproduce
here: `take_suffix` already does `checked_add` plus an explicit end rejection.

### PERF measurement

A/B min-of-60, alternated three rounds, over the 2 000-term deep-nesting
fixture: `seek_all` **592.4 µs (fixed) / 594.2 µs (checks reverted)**;
`next_all` **138.9 / 141.0 µs**. Noise reached 900 µs on both variants. No
measurable regression: every new check is per-block or per-frame-push, never
per-entry, except one predictable branch on `totalTermFreq`.

---

## `points.rs`

Java: `util/bkd/{BKDReader, BKDWriter, BKDConfig, DocIdsWriter}.java`,
`index/PointValues.java`, `search/DocBaseBitSetIterator.java`.
(`codecs/lucene90/Lucene90Points{Reader,Writer}` are thin wrappers; the
substance is in `bkd/`.)

144 lint sites: **10** `checked_*`/`try_from` reported as corruption, **9**
`wrapping_*` (faithful Java `int`/`byte` truncation — the
`firstDiffByteDelta` byte write and the `delta16`/`continuous`/`legacy`
doc-id accumulators), **14** `saturating_*` in "bytes needed" ceilings where
saturation can only *reject*, never accept, **91** proved, the rest eliminated
by hoisting into three new readers.

### F23 `[CORRECTNESS]` `decode_leaf_pointers` — `.kdm` `numLeaves` sized a 17 GB reservation

b7's sibling. Java only `assert numLeaves > 0`. Reproduced as a real
**SIGABRT** under `ulimit -v`: `memory allocation of 17179869176 bytes
failed`. Capped by the packed index's own length — each leaf past the first
costs at least one FP-delta vlong byte, so `b` bytes carry at most `b` leaves,
and a well-formed file reserves exactly what it did. Test:
`leaf_pointer_reservation_is_capped_by_the_packed_index_length`.

### F24 `[CORRECTNESS]` `read_leaf_block` — the leaf point `count` was unbounded

Java bounds it *absolutely but implicitly*: `DocIdsWriter.readInts` decodes
into `new int[maxPointsInLeafNode]`, so a larger count throws before a doc id
lands. This port allocates per leaf, so the same vint sized a fresh `Vec` and
a negative one became `usize::MAX`. New `read_leaf_count` restates Java's
invariant. Test: `leaf_count_outside_max_points_in_leaf_node_is_a_decode_error`.

### F25 `[CORRECTNESS]` `read_leaf_block` — a common prefix wider than `bytesPerDim`

No writer emits one (it *is* a prefix of that dimension's bytes). Java's
`readBytes` spills into the next dimension; here the slice bound left
`scratch_value` entirely, and for an all-equal leaf the spill decoded
**silently wrong point values**. Test:
`leaf_common_prefix_outside_bytes_per_dim_is_a_decode_error`.

### F26 `[CORRECTNESS]` `read_leaf_block` — full-width common prefix on the compressed dimension

`BKDWriter.writeLeafBlockPackedValues` asserts
`commonPrefixLengths[sortedDim] < bytesPerDim` before it can pick a
non-negative marker. Without it `compressed_byte_offset` addressed one past
`scratch_value` and the `+= 1` inverted every later suffix range. Test:
`full_width_common_prefix_on_the_compressed_dim_is_a_decode_error`.

### F27 `[CORRECTNESS]` `read_leaf_block` — `i + length > count` was self-defeating

The same shape as F17: the guard formed the sum it was meant to guard. Now
`length > count - i`, whose subtraction the loop condition proves. Test:
`negative_low_cardinality_run_length_is_a_decode_error`.

### F28 `[CORRECTNESS]` `read_bitset_ids` — three unbounded header values

`longLen` sized `vec![0i64; n]` with nothing between it and the allocator
(Java grows a reusable array — an OOME, not an abort); `offsetWords * 64`
overflowed an `i32`; the per-set-bit doc id could run past `i32::MAX`. All
three bounds established **once, before the scan**, so the popcount loop is
unchanged. Test: `corrupt_bitset_doc_id_header_is_a_decode_error`.

### F29 `[CORRECTNESS]` `PointsReader::{decode_leaves,intersect}` — the `.kdi` slice bound

`index_start_pointer + num_index_bytes` on two unbounded `.kdm` values: a
debug panic, and in release a wrap to a *plausible in-range* end offset
handing the walker another field's bytes. Factored into
`PointsReader::inner_nodes`. Test:
`a_packed_index_range_that_overflows_is_a_decode_error`.

### F30 `[CORRECTNESS]` the tree walk — `nodeID * 2`, and unbounded recursion

Java lets the multiply wrap; a wrapped id then compares `< numLeaves` again,
and since the walk only stops at a leaf the depth was bounded by nothing but
the `.kdi`'s length — i.e. a large enough packed index overflows the **stack**.
`child_ids`' `checked_mul` caps the depth at 31 levels. Test:
`a_packed_index_deeper_than_the_node_id_space_is_a_decode_error`.

### F31 `[CORRECTNESS]` `intersect_node`/`add_all`/`walk_node` — the split descriptor

A negative `code` gave Java a negative `splitDim` (an AIOOBE one line later)
and gave Rust an astronomic `usize` indexing `negative_deltas`; `1 +
bytes_per_dim` can itself overflow an `i32`. Extracted into
`read_split_descriptor`, which establishes `split_dim < num_index_dims` and
`prefix <= bytes_per_dim` for all three walks. Plus `leftNumBytes` (negative →
a release-mode wrap onto the wrong node) and the leaf FP `fp + right_delta`.
Tests: `a_negative_split_descriptor_is_a_decode_error`,
`a_negative_left_num_bytes_is_a_decode_error`,
`a_leaf_pointer_that_overflows_is_a_decode_error`.

### F32 `[CORRECTNESS]` `read_bpv21` — Java's `>>>` ported as `>>`

`DocIdsWriter.readInts21` decodes its top field as `(int) (l >>> 42)`, a
22-bit **zero-extended** field. The port had `(l >> 42) as i32`, so a corrupt
block's negative word decoded to a *negative doc id* instead of the in-range
value Java yields — a different answer, not a rejected one. Every `>>>` in
`BKDReader`/`BKDWriter`/`DocIdsWriter`/`BKDUtil` was then swept; this was the
only divergence. Test: `bpv21_top_field_is_zero_extended_like_java`.

### F33 `[CORRECTNESS]` `range_query` — the caller's box was never width-checked

`RangeVisitor` slices `lower`/`upper` per index dimension against the *field's*
shape, so a short box panicked mid-traversal. Java's `PointRangeQuery`
constructor checks it up front. Test:
`range_query_rejects_bounds_of_the_wrong_width`.

### F34 `[MISSING]` write side — `PointValues.MAX_NUM_BYTES`, and a `+ 1` on a doc id

`FieldInfo`/`FieldType.setDimensions` cap `bytesPerDim` at 16, which is what
keeps `pack_index`'s
`(delta * (1 + bytesPerDim) + prefix) * numIndexDims + splitDim` inside an
`i32`; `BKDConfig` does not check it, so `check_config` (shared with the read
side, which must accept exactly what `BKDReader` accepts) still does not — the
bound is applied in `write_field` only. Separately `write_leaf_doc_ids`'
`w[1] == w[0] + 1` overflowed on a run ending at `i32::MAX`. Tests:
`write_rejects_bytes_per_dim_past_max_num_bytes`,
`write_handles_a_doc_id_run_ending_at_i32_max`.

### Byte-flip sweep

`resigned_byte_flip_sweep_never_panics`: 3 bit positions × every payload byte
of `.kdm`/`.kdi`/`.kdd`, driving open + `decode_leaves` + `decode_all_points`
+ `range_query` + `intersect` (1 014 flips). **Before the fixes: 8 panics**
(1 `.kdm`, 7 `.kdd`) plus F23's abort under `ulimit -v`. **After: 0 panics**,
rejection rates `.kdm` 270/378, `.kdi` 169/198, `.kdd` 204/438. The low `.kdd`
figure is correct rather than a gap: most `.kdd` payload bytes are packed
point values, and flipping one yields a different but perfectly well-formed
point. The whole points suite also passes under `ulimit -v 4000000`, so no
reachable path reserves past 4 GB.

### PERF measurement

Min-of-60, four alternating A/B rounds, release, 200 k points / 391 leaves:
`decode_all_points` **5 029 µs (fixed) / 4 887 µs (before)**; `range_query`
**236 / 216 µs**. Neutral within this machine's ±6% band on identical code
(the same code varied 4 851–7 764 µs across runs). Every check is per-leaf or
per-inner-node; nothing was added inside the per-point loop.

### Open

A doubly-corrupt `.kdm` + `.kdd` can still ask for a large leaf `Vec` —
reaching a multi-GB leaf now needs `maxPointsInLeafNode` *and* the leaf
`count` corrupted consistently. That is strictly better than Java, which
eagerly allocates `new int[maxPointsInLeafNode]` from a single value.
Recorded rather than invented: no tighter input-derived bound exists, because
`CONTINUOUS_IDS` genuinely encodes `count` doc ids in ~5 bytes.

---

## `stored_fields.rs`

Java: `codecs/lucene90/compressing/{Lucene90CompressingStoredFieldsReader, Lucene90CompressingStoredFieldsWriter, FieldsIndexReader, FieldsIndexWriter, StoredFieldsInts}.java`,
`codecs/lucene90/{LZ4WithPresetDictCompressionMode, Lucene90StoredFieldsFormat}.java`.

83 lint sites: **6** `checked_*` reported as corruption, **4**
`saturating_*`/`wrapping_*` where that is the honest semantics, **73** proved
under 31 tightly-scoped `#[allow]`s. About 8 sites disappeared outright rather
than being allowed (a duplicated `want_end - want_start` collapsed to one
`wanted`; a second validation pass over the offsets merged away).

### F35 `[CORRECTNESS]` `chunkDocs` was bounded only by `maxDoc`, itself a raw `.fdm` `i32` — a 60-byte file could reserve 17 GB

Java bounds `chunkDocs` by `numDocs`, which it takes from `SegmentInfo`; this
port takes `maxDoc` from `.fdm`, so nothing bounded it. `chunk_docs as usize`
sizes `read_bulk_ints`' `vec![0i64; count]` and the `offsets` reservation.
Bounded by `max_docs_per_chunk(mode)` — 1024/4096 — which is exact rather than
defensive: the data codec name `open` matched
(`Lucene90StoredFieldsFastData`/`HighData`) is written by exactly one Java
class, `Lucene90StoredFieldsFormat.impl`, whose two
`new Lucene90CompressingStoredFieldsFormat(...)` calls pin `maxDocsPerChunk`,
and the writer re-checks `numBufferedDocs >= maxDocsPerChunk` after every
document. **Verified under `ulimit -v`: unfixed, a token claiming
`i32::MAX - 1` documents gives `memory allocation of 8589934576 bytes failed`
— a SIGABRT, not an unwind.** Test:
`a_chunk_claiming_more_documents_than_the_format_allows_is_rejected` (using
2 000 000, so the suite stays survivable).

### F36 `[CORRECTNESS]` Java's `token >>> 2` ported as `>>`

In both `read_chunk_header` and `copy_chunks`. A corrupt token with its sign
bit set gave a *negative* `chunk_docs` here where Java gets a large positive —
and `chunk_docs as usize` on a negative is ~2^64. `infoAndBits >>> TYPE_BITS`
was signed here too; the two are provably equal after the `as i32` (they
differ only in the top three bits) but it was changed anyway with a comment,
so nobody has to re-derive it.

### F37 `[CORRECTNESS]` `maxPointer as usize + FOOTER_LENGTH`

`(-1i64) as usize + 16` wraps to 15, so the `.fdt`-length cross-check was not
merely bypassable — it panicked in debug before it could reject anything. Now
`usize::try_from(...).and_then(checked_add)`, which is also what establishes
`0 <= max_pointer <= fdt.len() - 16` for `check_integrity`. Test:
`negative_max_pointer_is_a_decode_error_not_an_overflow`.

### F38 `[CORRECTNESS]` `index_num_chunks != num_chunks + 1`, and a negative `numChunks`

Overflows for a `numChunks` vlong of `i64::MAX`; `num_chunks` also feeds
`direct_monotonic::floor_index` as a search bound (c24's F2 territory) and
could be negative. Test:
`num_chunks_at_the_top_of_the_vlong_range_is_a_decode_error_not_an_overflow`.

### F39 `[CORRECTNESS]` a negative `maxDoc` was accepted

Which made every `doc_base + chunk_docs > max_doc` check vacuous. Test:
`negative_max_doc_is_rejected_by_open`.

### F40 `[CORRECTNESS]` a negative per-document length made `offsets` decrease

Only the `bpv == 0` bulk-int shape can carry a negative (the 8/16/32-bit
shapes are masked), and Java's `(len == 0) != (storedFields == 0)` check
passes it. `serialized_document`'s `offsets[i+1] as usize - doc_offset` then
underflowed to ~2^64 and `Vec::with_capacity` panicked with capacity overflow.
Fixed at the source, in `read_bulk_ints`' `bpv == 0` arm — **one comparison
per array, not per document** (see PERF-3). Test:
`a_negative_document_length_is_a_decode_error_not_a_backwards_offset`.

### F41 `[CORRECTNESS]` a chunk's decompressed length was bounded by nothing

It sizes both `decompress_range`'s output `Vec` and `decompress_unit`'s
`vec![0u8; dictLength + blockLength]` — up to 1024 × `u32::MAX` ≈ 2 TB from a
hundred-byte file. Bounded by c25's pattern (`remaining × max_expansion`),
checked once per chunk header, with 255 for LZ4 and 1032 for DEFLATE (zlib's
documented worst case — deliberately the loose end, so it can never reject a
real file). Test:
`a_chunk_claiming_more_decompressed_bytes_than_could_be_produced_is_rejected`.

### F42 `[CORRECTNESS]` `read_field`'s `TYPE_BYTE_ARR` length fed `vec![0u8; length]`

`read_vint() as usize`; negative → ~2^64. Java allocates nothing here
(`new StoredFieldDataInput(in, length)` is lazy), so bounding by
bytes-remaining is the closest this port's owning `Vec<u8>` gets to the same
exposure. Test:
`an_oversized_binary_field_length_is_a_decode_error_not_an_allocation`.

### F43 `[CORRECTNESS]` `parse_document` reserved one `StoredField` slot per *claimed* field

`numStoredFields` comes off the 32-bit bulk array read unsigned, so ~4.3e9 ×
40 bytes. Capped by `bytes.len()`, which is free — every field costs at least
its `infoAndBits` byte. **Verified under `ulimit -v`: unfixed gives
`memory allocation of 171798691800 bytes failed`.** Test:
`an_absurd_stored_field_count_does_not_reserve_a_slot_per_claimed_field`.

### F44 `[CORRECTNESS]` sub-block compressed lengths were unchecked

A negative vint became ~2^64 and overflowed the running total the
`length == 0` skip path sums — which in release wraps to a *small* skip and
leaves the reader mid-unit, so the next `sliced` unit parses compressed bytes
as a framing header. Test:
`a_negative_sub_block_compressed_length_is_a_decode_error`.

### F45 `[CORRECTNESS]` `copy_chunks` advanced `doc`/`doc_base` before the guard meant to catch it

Exactly `term_vectors`' F3. A wrap could land a chunk claiming ~2^29 documents
back inside the requested range. **Untested** — reaching it needs a segment
with `maxDoc` near `i32::MAX`; recorded as reasoned hardening, not a tested
fix.

### F46 `[CORRECTNESS]` `flush`'s `doc_base += chunk_docs` wrapped silently

As Java's does — writing a negative `docBase` vint and a non-monotonic `.fdx`,
i.e. an unopenable segment written without complaint. Now
`checked_add(...).expect(...)`: more than 2^31 documents in a segment is a
caller bug (`IndexWriter.MAX_DOCS` is 2^31 − 128) and `add_document` has no
`Result`. **Untested** (not constructible).

### INTENTIONAL — `too_dirty`'s `* 100`, and `read_tlong`'s multiplies

`too_dirty` is now `saturating_mul`, but the honest statement — and what the
comment says — is that it **cannot** overflow: `open` rejects
`numDirtyChunks > numChunks`, and `numChunks` is pinned to
`indexNumChunks - 1` where `indexNumChunks` is a plain `readInt`. Saturating
makes the future-proof direction ("too dirty" → refuse the bulk copy) the safe
one, where Java's `long` wrap to a negative product reads as "clean". Not
claimed as a live defect. `read_tlong`'s `l * SECOND|HOUR|DAY` are
`wrapping_mul`: Java's `long` multiply wraps, `writeTLong` only ever emits an
`l` small enough for the product to be exact, and a corrupt vlong should
decode to the same garbage `long` Java hands its visitor rather than to a
`Result` this port's callers would then have to differ from Java about.
Nothing downstream uses the value as a length or an index.

### PERF-3 — a check inside the per-document loop cost 2.6%, and moving it up recovered all of it

The first cut of F40 put the `len < 0` check inside the per-document offsets
loop: a measured **2.6% on both `cursor_scan` and `read_chunk`**. Moving it
into `read_bulk_ints`' `bpv == 0` arm — one check per array — took that back
to zero. Separately, `read_bulk_ints` was allocating a fresh
`vec![0i64; num_words]` **per 128-value block** (64 allocations per
4096-document chunk header); hoisted out of the loop, matching Java's reusable
`long[]`. Java's two separate passes over `numStoredFields`/`offsets` are also
merged into one.

Min-of-40, A/B alternating three times, release, 40 000 documents / 40 chunks:
`document()` random access 40.65–40.76 ms vs 40.63–40.68; `ChunkCursor` scan
3.085–3.104 ms vs 3.075–3.100; `read_chunk` × 40 2.798–2.803 ms vs
2.793–2.800; `copy_chunks` 21.9–22.4 µs vs 22.1–22.6; `open` × 100
13.83–14.15 µs vs 14.00–14.31. **Neutral.**

### Byte-flip sweep

`.fdm` **269/282 (95.4%)**, `.fdx` **114/126 (90.5%)**, `.fdt` **624/904
(69.0%)**, all re-signed across all three footers, driving open + every
document through `ChunkCursor` + `parse_document` + `check_integrity`. The 13
accepted `.fdm` flips are individually accounted for: the codec version (which
`open` legitimately accepts a range of), `chunkSize` (only a `sliced` chunk
consults it), the unused `offset` of a `bpv == 0` monotonic block, and
`numDirtyDocs` (which Java only sanity-checks).

**A methodology note worth keeping**: the first run used a *single-chunk*
segment and scored 211/282 on `.fdm`. A one-chunk segment gives the monotonic
index arrays nothing to discriminate, so that rate was measuring the fixture,
not the decoder. Anyone repeating this elsewhere should build at least three
chunks.

---

## `for_util.rs`

Java: `codecs/lucene104/{ForUtil, PForUtil}.java`,
`internal/vectorization/PostingDecodingUtil.java` (`splitInts`). Note there is
**no `ForDeltaUtil` in the `lucene104` package** the port targets — that class
exists only in `backward_codecs/lucene{84,99,101,103,912}`. Nothing missing.

**Apart from F47 and F48 this module is clean, and that is the finding.** 164
of the 166 lint sites are arithmetic over *decode-shape constants* — literal
counts, strides, shifts and lane offsets baked into the format — not values
off disk. The one disk-derived width (`bits_per_value`) was already
range-checked by a prior batch, and the one disk-derived *index*
(`ints[idx]` in `pfor_decode`) is a single byte. **0** sites warranted
`checked_*` and **0** warranted `saturating_*`; manufacturing either would
have been noise. All 166 are proved, under 22 tightly-scoped `#[allow]`s, six
of them pinned with `debug_assert!`.

### F47 `[CORRECTNESS]` `ForUtil::encode` at `bitsPerValue == 0` is a release-mode infinite loop

Java's `ForUtil.encode` runs
`for (shift = shift - bitsPerValue; shift >= 0; shift -= bitsPerValue)`; at
`bitsPerValue == 0` the induction variable never moves and the encoder spins
forever. This port transliterated it faithfully, hang included. **A hang is
the one failure `catch_unwind` at the FFI boundary cannot convert back into a
Java exception** — it takes the calling JVM thread with it, with no timeout,
which makes it strictly worse than the aborts the rest of this batch is about.
Fixed with `assert!((1..=32).contains(&bits_per_value))` — one compare per
256-value block, unmeasurable. Verified against the unfixed code: the debug
build fails both new tests, and the **release** build of the zero-width test
never returns (killed at 90 s under `timeout`). Not reachable from today's
writers (`postings_writer.rs:1071` uses `.max(1)`, and `pfor_encode`'s width
search provably cannot select 0 — `bits_required` never returns 0, so the
histogram buckets `1..=max` already hold all 256 values; pinned with a
`debug_assert!`). Tests:
`encode_refuses_a_zero_bit_width_instead_of_spinning_forever`,
`encode_refuses_a_width_above_32`.

### F48 `[MISSING]` `PForUtil`'s `static { assert ForUtil.BLOCK_SIZE <= 256 }` was never ported

That assertion is what makes `pfor_decode`'s `ints[read_byte() as usize]` —
the only index in the whole file taken from a byte off disk — in bounds, and
what makes `pfor_encode`'s `i as u8` exception index lossless. Ported as
`const _: () = assert!(BLOCK_SIZE == 256)`, tightened to equality because the
decode side additionally needs `>= 256` where Java only needs `<= 256`. A
compile-time assertion is its own test.

### F49 `[PERF]` the obvious way to make these sites disappear regresses the kernel 9–48%

The first pass rewrote `expand8`/`expand16`/`collapse8`/`collapse16` with
`split_at_mut` and `decode3..decode15`'s tails with
`chunks_exact`/`chunks_exact_mut` — which removes the index arithmetic *and*
the bounds checks, and reads better. It is substantially slower: on a
`[u32; 256]` the array's **static** length is what lets LLVM fold the bounds
checks and unroll; `split_at_mut`/`chunks_exact` hand it runtime-length slices
and it stops. The iterator form measured **+38.6 / +47.9 / +43.4 / +41.4 /
+40.0 / +9.9%** at widths 3/5/9/11/13/15, and the `split_at_mut`
expand/collapse form +9% to +28% at widths 1/3/5/6/9/11/13. All 15 functions
reverted; every hot-path body is byte-identical to the pre-audit code except
one dead `let _ = m8;` removed from `decode9`.

**This is the most transferable result in the batch**: the natural way to
satisfy `arithmetic_side_effects` on a fixed-size-array kernel is a 9–48%
regression, and it looks like a cleanup.

Shipped vs. pre-audit baseline (min-of-60 × 4000 ops per width, A/B alternated
over 5 rounds, widths 1/3/5/6/7/9/11/13/15/16/21/31): decode **+0.0% to
+0.6%** (10 of 12 widths within ±0.1%), encode **−1.7% to +0.2%**.

### INTENTIONAL — `pfor_decode`'s `read_vint()? as u32`

Java is `Arrays.fill(ints, 0, BLOCK_SIZE, in.readVInt())` into an `int[]`,
storing exactly the same 32 bits. This port models Lucene's `int` as `u32`
through the decode kernel, so the cast is the identity on the bit pattern, not
a widening.

### Open — needs a change outside `for_util.rs`

`pfor_encode` given a value needing 32 bits writes
`token = (numExceptions << 5) | 32`, whose low 5 bits are `0` — the all-equal
marker — silently corrupting the block. Java is protected because
`PackedInts.bitsRequired` throws on a negative `int`; here it is only a
`debug_assert!`. Unreachable in practice (Lucene doc deltas and freqs are
non-negative Java `int`s). Making it a hard failure needs `pfor_encode` to
return `Result`, which would touch `postings_writer.rs` — a module still
carrying its `TODO(arith-audit)` marker, so it belongs to whoever burns that
one down.

---

## `postings.rs`

Java: `codecs/lucene104/{Lucene104PostingsReader, PostingsUtil, Lucene104PostingsFormat}.java`.

51 lint sites: **6** `checked_*` reported as corruption, **1** `wrapping_*` as
the honest Java-`int` semantics, **44** proved under 30 allows. Nothing got a
`saturating_*` it did not already have. Four proofs are pinned with
`debug_assert!`. c15 had already capped all five allocations in the file by
the input's own length, so the non-lint half yielded exactly one defect.

### F50 `[CORRECTNESS]` `read_positions` — a negative `.pay` payload length slices `payload_bytes` backwards

Java's `refillOffsetsOrPayloads` keeps payload lengths in an `int[]` and never
indexes a byte run with them directly. Here `payload_lengths` holds a
`PForUtil`-decoded `u32` stored as `i32`, so a corrupt block yields a
**negative** length; `negative as usize` sign-extends to ~2^64 and
`payload_upto + len` either panicked outright (debug: `attempt to add with
overflow`) or **wrapped below `payload_upto`, passed the
`end > payload_bytes.len()` guard**, and panicked in
`payload_bytes[start..end]` (release: `slice index starts at 256 but ends at
255`). That is the `a + b > len` shape again — F17's twin, in a file nobody
had connected to it. Reachable through the public `read_positions` on any
payload-carrying field with at least two full `.pos` blocks. Test:
`a_payload_length_block_claiming_a_negative_length_is_rejected_not_a_panic`,
verified failing in **both** debug and release, with the two different panics
above.

### F51 `[CORRECTNESS]` `decode_impacts_into` — the `1 +` in front of the accumulator overflowed

Java's `readImpacts` is `freq += 1 + (freqDelta >>> 1)` in an `int`, which
wraps. The port had the `>>>` right but the `1 +` was a plain `i32` add: a
five-byte varint decoding to `-2` makes the shift exactly `i32::MAX`, so
`1 + that` panics in a debug build *before* the accumulator's existing
`wrapping_add` ever sees it — on a `.doc` impacts run, i.e. on every
freqs-indexing term. Now `1i32.wrapping_add(..)`. Test:
`an_impacts_freq_delta_at_the_top_of_i32_wraps_rather_than_panicking`, which
asserts the exact wrapped value Java's `int` produces. **A single-bit flip
cannot produce that five-byte pattern, which is why neither byte-flip sweep
reaches it** — a useful reminder that the sweep and the hand audit find
different things.

### F52–F55 `[CORRECTNESS]` four defensive hardenings, no failing test possible

`wanted_ranges`' occurrence-range *end* (`from + freq as usize`: unreachable
on 64-bit, a live wrap on a 32-bit target, where `wire_count`'s `u32::MAX` cap
*is* `usize::MAX`); `read_level1_entry`'s `skip1EndFP` (a negative `readShort`
became ~2^64 and merely *looked* like a wire mismatch); the three
payload-byte-run `resize(start + len)` sites; and
`read_positions`/`read_positions_flat`'s `total_term_freq as usize`, which now
goes through `wire_count` locally instead of relying on a check two calls away.
**Called out explicitly so they are not mistaken for tested fixes.**

### F56 `[INTENTIONAL]` the eager path amplifies `.doc` ~256x into memory

`read_postings_with_flags` caps its *reservation* by `buf.len()` (c15), but
`extend_from_slice` still grows to `docFreq` entries: ~8 bytes of `.doc` buys
2 KiB of `docs` + `freqs`, so a 67 MB corrupt `.doc` can ask for ~17 GB. It is
bounded by the input and inherent to the eager design this file documents
(Java streams instead), and `docFreq <= maxDoc` is not knowable here. Recorded
rather than invented around — the same call c24 made for `terms_dict`'s
`maxBlockLength`. Capping it needs `maxDoc`, which this module does not have.

### Byte-flip sweeps

Both flip bit 0 and bit 7 of every body byte, re-sign the footer, and require
a typed error or a clean decode from every reader.

* `every_resigned_single_byte_doc_corruption_is_an_error_or_a_clean_decode` —
  a level-1 entry with impacts + its 32 level-0 blocks + a group-varint tail
  (`docFreq` 8 200), through `read_postings`,
  `read_postings_with_flags(DocsOnly)`, a full `next_doc` walk and a strided
  `advance_shallow`/`advance` walk: **589/744 (79.2%)**.
* `every_resigned_single_byte_positional_doc_corruption_is_an_error_or_a_clean_decode`
  — a positions-indexing `.doc` (two level-0 headers carrying
  `readLevel0PosData`'s four sub-fields + tail), through
  `read_occurrences_for_doc` at nine targets with `.pos`/`.pay` left intact so
  a rejection is attributable to `.doc` alone: **49/60 (81.7%)**.

Zero panics and zero aborts in either, debug and release.

### PERF measurement

Min-of-80, four A/B alternations, in-tree file swap (not `git HEAD` — the
working tree already carried 3 200 lines of uncommitted `postings.rs` changes
from earlier batches, so HEAD is not "before"):

| arm | after | before | |
|---|---|---|---|
| `read_postings`, 8 200 docs | 4.933–4.952 µs | 4.931–4.952 µs | neutral |
| lazy `next_doc` walk, 8 200 docs | 5.782–5.809 µs | 5.773–5.781 µs | +0.5%, noise |
| `read_positions_for_docs`, 1 024 of 4 096 | 7.869–7.912 µs | 7.706–7.740 µs | **+2.2%** |
| `read_positions`, 4 096 occurrences with payloads | 278.2–278.9 µs | 278.7–279.8 µs | neutral |

The only real cost is F52's `checked_add` in `wanted_ranges`, ~0.04 ns per
wanted document. `decode_full_block_body`'s delta prefix sum — the hottest
loop in the file — is byte-for-byte unchanged, and c20's highlighting path is
untouched. F50's fix measures marginally *faster* than the manual bound it
replaced.

---

## `clippy::cast_sign_loss`, assessed per module as c24 recommended

c24 left this open: the lint "would have caught F6's second half directly, and
c19 rejected it only as a *workspace-wide* deny (1 036 sites)". All seven
modules assessed it independently:

| module | sites | live defects found | verdict |
|---|---|---|---|
| `points` | 37 | **1** (a leaf pointer still sign-extending into `seek`) | audit pass, don't leave on |
| `blocktree` | 44 | 0 (~5 candidates, all already guarded) | don't adopt |
| `stored_fields` | 56 | 0 | don't adopt |
| `term_vectors` | 52 | 0 | don't adopt |
| `postings` | 28 | 0 (2 interesting shapes fixed by hand anyway) | don't adopt |
| `doc_values` | 7 | 0 | don't adopt |
| `for_util` | **3** | 0 | **adopted** — `#![deny]` at module scope |

The result is a clear recommendation, now written into
`docs/arithmetic-gate.md`: **run it once during a module's burn-down and fix
what it finds, then leave it off.** It earns its keep — it found a live defect
in `points` that the arithmetic gate did not — but in every module where the
count is 28–56, the remaining sites are deliberate bit reinterpretations
(`as u32`/`as u64` is how this port spells Java's `>>>`, `as u8` its byte
truncation), so a standing deny would cost 30–50 `#[allow]`s whose proofs
restate proofs the arithmetic gate already carries. `for_util` is the
exception worth making: 3 sites, all same-width `i32 as u32`, and the deny
locks the decode kernel against a future `i32 as usize`.

---

## Verdicts

| module | verdict |
|---|---|
| `term_vectors.rs` | swept — 13 defects (F1–F12 + F7's prefix half), 13 tests, 97.94% lines |
| `points.rs` | swept — 12 defects (F23–F34), sweep found 8 panics + 1 SIGABRT, 99.15% |
| `stored_fields.rs` | swept — 13 defects (F35–F46), two verified SIGABRTs, 98.29% |
| `doc_values.rs` | swept — 5 defects (F13–F16, F57) incl. one SIGABRT and c26's merge blocker, 97.89% |
| `blocktree.rs` | swept — 6 defects (F17–F22), 95.81% (with integration tests) |
| `postings.rs` | swept — 7 defects (F50–F56), 97.90% |
| `for_util.rs` | **swept-clean apart from two findings** — 164 of 166 sites are decode-shape constants, and no `checked_*` or `saturating_*` was warranted anywhere. The two are a release-mode infinite loop (F47) and an unported `static` assertion (F48). 96.48% |

## Gates

* `cargo fmt --all` clean.
* `cargo clippy -p lucene-codecs --all-targets -- -D warnings` clean.
* `cargo test -p lucene-codecs`: 33 test binaries, 0 failures.
* `cargo test -p lucene-index`: green (656 lib + integration) — the new
  corruption checks reject nothing a real writer produces.
* `scripts/verify-write-path.sh`: **22/22**, run after all seven modules
  changed. `VerifyTermVectors`, `VerifyDocValues`, `VerifyPoints`,
  `VerifyStoredFields`, `VerifyBlockSegment` and `VerifyPositionsSegment` are
  real Lucene 10.5.0 reading this port's changed writers' bytes.
* `python3 scripts/check-parity.py`: ok.
* `python3 scripts/check-arith-allows.py`: ok, 8 modules still unaudited
  (5 codecs + 3 index).
* Per-file line coverage, all seven: 95.81%, 96.48%, 97.89%, 97.90%, 97.94%,
  98.29%, 99.15% — every one above the 95% bar.

## Tier-2 review

The `quality-reviewer` pass over this diff found **two gating defects**, both
in the two modules this batch's own author wrote, and both fixed here:

1. **`doc_values`' shift proof was c24's failure mode verbatim** — a proof
   whose stated bound (`0..=63`) includes the value that breaks it. See F57.
   It also found the `#[allow]` and its proof stacked **three times** on one
   statement, an editing artifact from a scripted replace; removed. Worth a
   mechanical guard: `clippy::duplicated_attributes` (in `suspicious`) catches
   it, or `check-arith-allows.py` could reject two of these attributes on one
   item.
2. **`term_vectors`' `.max(0)` on `prefixLength`** turned a rejection into a
   plausible wrong term. See F7. The batch's own report argued the point about
   the sibling array in the very same function and still shipped the reflex —
   which is the argument for a review pass that reads the proofs rather than
   counting them.

Three advisory findings were also acted on: the `term_vectors` byte-flip
sweep was measuring a single-document fixture (rebuilt on a 400-document,
multi-chunk, two-field segment — see that section); the `doc_values` sweep
drove only the five *dense* writers, so the four sparse encoders this batch
added had no coverage (added); and
`a_sparse_column_that_covers_every_document_writes_javas_dense_marker`
asserted byte-identity for two of the four types (now all four). It confirmed
category (c) — **bounds that could reject a valid file** — completely clean,
having traced each of the twelve new bounds to the Java *writer* rather than
the reader, which is the check that matters most and the one this batch could
most easily have got wrong.

Two things it raised that were **not** changed, recorded here rather than
silently dropped:

* `points::MAX_POINTS_IN_LEAF_NODE` hard-codes `i32::MAX - 16`
  (`ArrayUtil.MAX_ARRAY_LENGTH`), which is JVM-header-size dependent — so the
  two implementations could disagree over an ~8-value window of absurd inputs.
* `stored_fields::too_dirty`'s `saturating_mul(100)` is the one saturation in
  the batch whose own comment says it cannot saturate. Left as future-proofing
  with the disclosure attached, which is the honest form.

## Open

* **5 modules still marked** in `lucene-codecs`: `fst`, `hnsw`,
  `hnsw_vectors`, `postings_writer`, `vectors`. None of them parse the
  per-segment metadata every index open goes through, which is why they were
  not this batch's priority — but `fst` is on the terms-index path and sits at
  86.58% line coverage, the lowest in the crate.
* **`pfor_encode` cannot report a 32-bit-wide value** (F49's open item). It
  writes a token whose low 5 bits are the all-equal marker, silently
  corrupting the block. Unreachable from Lucene's non-negative doc deltas and
  freqs; fixing it needs a `Result` signature and a `postings_writer.rs`
  call-site change, and `postings_writer` still carries its marker.
* **`postings`' eager path can still ask for ~17 GB** from a 67 MB corrupt
  `.doc` (F56). Bounded by the input, inherent to the eager design, and not
  fixable without `maxDoc`.
* **`points`' doubly-corrupt case**: reaching a multi-GB leaf `Vec` now needs
  `maxPointsInLeafNode` *and* the leaf `count` corrupted consistently — still
  strictly better than Java, which allocates `new int[maxPointsInLeafNode]`
  from a single value.
* **`stored_fields` F45/F46 and `postings` F52–F55 are untested** — hardening
  whose trigger needs a segment near `i32::MAX` documents, or a 32-bit target.
  Listed so they are not counted as tested fixes.
