//! Writes a whole index -- one `IndexWriter`, `add_document`, `commit` --
//! whose postings carry **positions, offsets and payloads**, for
//! `VerifyPositionsSegment` to read back with real Lucene 10.5.0.
//!
//! # Why this fixture exists
//!
//! Batch c20 ported the `.pos`/`.pay` skip machinery on both sides and then
//! recorded, in its own carry-over list, that no `CheckIndex` run and no Java
//! reader had ever seen a `.pos`/`.pay` file this port *wrote*: the evidence
//! for the whole positional write path was two of this port's own readers
//! agreeing with each other. This sweep has watched that exact evidence shape
//! fail twice (b4's FST framing, b11's invented `.si` sort encoding), because
//! a writer and a reader that share a misreading of the spec round-trip
//! perfectly.
//!
//! `write_full_segment_fixture` -- the one other whole-index case -- indexes
//! `DocsAndFreqs`, so it writes no `.pos` at all.
//!
//! # What it is built to exercise
//!
//! `Lucene104PostingsFormat`'s block sizes are the design constraint:
//! `BLOCK_SIZE` is 256 and `LEVEL1_NUM_DOCS` is 8192, and every interesting
//! path in the format is chosen by where a term's postings sit relative to
//! them. So:
//!
//! - **20 000 documents**, and `dense` occurs in every one of them: 78 full
//!   level-0 doc blocks, a group-varint tail, and **two complete level-1
//!   spans** (at doc 8191 and 16383) -- so the level-1 `.pos`/`.pay` skip
//!   record, the one c20 singled out as load-bearing, is written more than
//!   once and read back by Lucene's own `advance`.
//! - **Per-document frequency cycles 1..5.** The period matters and is not
//!   decoration: c20's Tier-2 review found its first fixture cycling with
//!   period 4, which made `sum(freq)` over a level-1 span an exact multiple
//!   of 256 and the level-1 `posBufferUpto` byte indistinguishable from a
//!   hardcoded zero. 5 is coprime with 256, so `.pos` block boundaries drift
//!   against `.doc` block boundaries and the skip records carry non-zero
//!   in-block offsets.
//! - **60 000 `dense` occurrences**, so `.pos` holds 234 full `PForUtil`
//!   blocks plus a vint tail, and a single document's occurrences straddle a
//!   `.pos` block boundary (`DENSE_BLOCK_CROSSING_DOC` in the manifest names
//!   one, and the verifier requires it to be sampled).
//! - **Payload lengths cycle with period 5 too** (`0..=4`), so a block's
//!   payload byte run is not uniform and a zero-length payload -- Lucene's
//!   `null`-payload equivalent -- occurs regularly.
//! - **Four fields, one per `IndexOptions` rung that this writer supports**:
//!   `tag` (`Docs`), `count` (`DocsAndFreqs`), `title`
//!   (`DocsAndFreqsAndPositions`, no offsets and no payloads) and `body`
//!   (`DocsAndFreqsAndPositionsAndOffsets` **with** payloads). They share one
//!   `.doc`/`.pos`/`.pay` file set, which is what makes the per-field framing
//!   real: `title`'s blocks must not read `.pay` even though `body`'s do, and
//!   `tag`'s must not read `.pos`.
//!
//! # The manifest, and why it is not derived from the inverted index
//!
//! `positions-manifest.properties` is written from an **independent**
//! whitespace re-scan of the very text handed to `add_document`, not from
//! `indexing_chain::invert_documents`' output. That is deliberate: a manifest
//! derived from the structure under test would agree with it however wrong
//! both were. Three separately-built things have to agree for this fixture to
//! pass -- the intended token layout, this port's invert-and-encode path, and
//! real Lucene's decoder.
//!
//! Usage: `write_positions_segment_fixture <output-dir>`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;
use std::io::Write;

/// Past `2 * LEVEL1_NUM_DOCS` (8192) so a term in every document closes two
/// whole level-1 spans, and past `78 * BLOCK_SIZE` so the level-0 path and
/// the group-varint tail both fire.
const NUM_DOCS: usize = 20_000;

/// Size of the shared filler vocabulary. Kept well under the document count
/// so filler terms have real document frequencies of their own (~33 docs
/// each here) rather than degenerating into per-document singletons.
const VOCAB: usize = 600;

/// Field numbers, fixed so the manifest and the stored-field values agree.
const F_ID: i32 = 0;
const F_BODY: i32 = 1;
const F_TITLE: i32 = 2;
const F_TAG: i32 = 3;
const F_COUNT: i32 = 4;
const F_HEAD: i32 = 5;
const F_NOTES: i32 = 6;

fn base_field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: true,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: IndexOptions::None,
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Byte,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

/// This document's `dense` term frequency. Period 5, coprime with the
/// 256-occurrence `.pos` block: see the module doc.
fn dense_freq(doc: usize) -> usize {
    1 + doc % 5
}

/// The payload attached to the token at absolute `position` of document
/// `doc`. Length cycles `0..=4` (period 5, again coprime with 256, so a
/// `.pos`/`.pay` block's payload byte run is never uniform); a zero-length
/// payload is Lucene's `null`-payload equivalent and is deliberately common.
///
/// `VerifyPositionsSegment` recomputes this rule independently, so a payload
/// that survives the round trip byte-identical but is attached to the wrong
/// occurrence still fails.
fn payload_for(doc: usize, position: usize) -> Vec<u8> {
    let len = (doc * 7 + position) % 5;
    (0..len)
        .map(|i| ((doc + position + i) & 0xFF) as u8)
        .collect()
}

/// The `body` text of document `doc`, as a plain space-separated ASCII token
/// list. ASCII and single spaces on purpose: the manifest re-derives every
/// token's byte offsets by scanning this string, and for ASCII the analyzer's
/// UTF-8 byte offsets and Lucene's character offsets coincide (see
/// `indexing_chain::Occurrence`'s note on that latent unit divergence).
fn body_text(doc: usize) -> String {
    let mut tokens: Vec<String> = Vec::new();
    for k in 0..dense_freq(doc) {
        tokens.push("dense".to_string());
        tokens.push(format!("w{:03}", (doc * 3 + k) % VOCAB));
    }
    if doc.is_multiple_of(3) {
        tokens.push("gap".to_string());
    }
    if doc.is_multiple_of(97) {
        tokens.push("sparse".to_string());
    }
    if doc.is_multiple_of(11) {
        // An adjacent pair, so the verifier can run a real PhraseQuery: a
        // segment whose positions decode individually but sit at the wrong
        // absolute values still matches every term query and no phrase.
        tokens.push("alpha".to_string());
        tokens.push("beta".to_string());
    }
    // Terms sitting exactly on the format's own boundaries. `blk256` has a
    // total term frequency of exactly BLOCK_SIZE, which is the third branch of
    // `Lucene104PostingsWriter`'s `lastPosBlockOffset` rule -- the one where
    // there is no vint tail at all and Java writes its `-1` sentinel. `blk257`
    // is the same shape one occurrence over, so the two together separate "no
    // tail" from "a one-occurrence tail". `solo` is a singleton (`docFreq ==
    // 1`, which the term dictionary pulses inline and which therefore has no
    // `.doc` skip data at all) and `duo` is the smallest non-singleton.
    if doc < 256 {
        tokens.push("blk256".to_string());
    }
    if doc < 257 {
        tokens.push("blk257".to_string());
    }
    if doc == 5 {
        tokens.push("solo".to_string());
    }
    if doc == 6 || doc == 7 {
        tokens.push("duo".to_string());
    }
    tokens.join(" ")
}

/// `head`: offsets **without** payloads. Together with `notes` (payloads
/// without offsets) and `body` (both), this makes the segment cover every
/// combination that changes how `.pay` is framed -- Lucene creates `.pay` when
/// either axis is present but writes a different record for each.
fn head_text(doc: usize) -> String {
    format!("h{:02} head h{:02}", doc % 17, doc % 5)
}

/// `notes`: payloads **without** offsets, and with payload byte runs an order
/// of magnitude longer than `body`'s, so a `.pay` payload block spans many
/// bytes per occurrence rather than the 0..4 `body` uses.
fn notes_text(doc: usize) -> String {
    format!("note n{:02} note", doc % 23)
}

/// `notes`' payload rule: lengths 0, 37 or 74 bytes, so one occurrence's
/// payload alone is wider than `body`'s whole per-block run.
fn notes_payload_for(doc: usize, position: usize) -> Vec<u8> {
    let len = (doc % 3) * 37;
    (0..len)
        .map(|i| ((doc * 3 + position * 5 + i) & 0xFF) as u8)
        .collect()
}

/// `title`: positions but no offsets and no payloads, so its terms exercise
/// the `.pos`-without-`.pay` framing inside a segment whose `.pay` exists for
/// another field.
fn title_text(doc: usize) -> String {
    format!("common t{:02} common", doc % 37)
}

/// `tag`: `IndexOptions::Docs` -- no frequencies, no positions at all.
fn tag_text(doc: usize) -> String {
    format!("always tag{:02}", doc % 13)
}

/// `count`: `IndexOptions::DocsAndFreqs`, the rung `write_full_segment_fixture`
/// already covers, present here so all four live in one segment.
fn count_text(doc: usize) -> String {
    format!("c{:01} c{:01}", doc % 4, doc % 7)
}

/// Byte offsets of each whitespace-separated token in `text`, re-derived by
/// scanning rather than taken from the analyzer.
fn scan_tokens(text: &str) -> Vec<(String, usize, usize)> {
    let mut out = Vec::new();
    let mut start = 0usize;
    for (i, ch) in text.char_indices() {
        if ch == ' ' {
            if i > start {
                out.push((text[start..i].to_string(), start, i));
            }
            start = i + 1;
        }
    }
    if text.len() > start {
        out.push((text[start..].to_string(), start, text.len()));
    }
    out
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_positions_segment_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).expect("create output dir");

    let dir = FsDirectory::open(&out_dir);
    let fields = vec![
        base_field("id", F_ID),
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            store_payloads: true,
            store_term_vectors: true,
            // The one field carrying norms: Lucene writes norms for any
            // indexed field that does not omit them and refuses a segment
            // whose `.fnm` claims norms the files do not carry.
            omit_norms: false,
            ..base_field("body", F_BODY)
        },
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            ..base_field("title", F_TITLE)
        },
        FieldInfo {
            index_options: IndexOptions::Docs,
            ..base_field("tag", F_TAG)
        },
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqs,
            ..base_field("count", F_COUNT)
        },
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            ..base_field("head", F_HEAD)
        },
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            store_payloads: true,
            ..base_field("notes", F_NOTES)
        },
    ];
    let mut writer = IndexWriter::open(
        &dir,
        fields,
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .expect("open writer");
    // One segment, so `dense`'s document frequency really is NUM_DOCS and the
    // level-1 spans really are crossed. The default 16 MB RAM buffer would
    // flush this corpus into several segments and each term's postings would
    // fall back below every block threshold this fixture exists to cross.
    writer
        .set_max_buffered_docs(NUM_DOCS as i32 + 1)
        .expect("max buffered docs");
    writer
        .set_ram_buffer_size_mb(4096.0)
        .expect("ram buffer size");
    writer
        .set_postings_field(Some("body"))
        .expect("set postings field");
    for name in ["title", "tag", "count", "head", "notes"] {
        writer.add_postings_field(name).expect("add postings field");
    }
    // Term vectors over the same field whose postings carry offsets and
    // payloads. This is what makes `CheckIndex.testTermVectors` bite: at
    // MIN_LEVEL_FOR_SLOW_CHECKS it walks every document's vector against that
    // document's postings, occurrence by occurrence, and compares positions,
    // offsets and payloads wherever both carry the axis. Before this batch the
    // writer recorded positions only in a vector, whatever the field's
    // options, so the offset and payload halves of that check could not fire.
    writer
        .set_term_vector_field(Some("body"))
        .expect("set term vector field");
    writer
        .set_norms_field(Some("body"))
        .expect("set norms field");
    writer
        .set_payload_source(Some(Box::new(|ctx| {
            let payload = match ctx.field {
                "body" => payload_for(ctx.doc_id as usize, ctx.position as usize),
                "notes" => notes_payload_for(ctx.doc_id as usize, ctx.position as usize),
                // Every other field's tokens never reach here: the source is
                // consulted only for `store_payloads` fields.
                other => panic!("payload source consulted for field {other:?}"),
            };
            if payload.is_empty() {
                None
            } else {
                Some(payload)
            }
        })))
        .expect("set payload source");

    for doc in 0..NUM_DOCS {
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: F_ID,
                        value: FieldValue::String(format!("doc{doc}")),
                    },
                    StoredField {
                        field_number: F_BODY,
                        value: FieldValue::String(body_text(doc)),
                    },
                    StoredField {
                        field_number: F_TITLE,
                        value: FieldValue::String(title_text(doc)),
                    },
                    StoredField {
                        field_number: F_TAG,
                        value: FieldValue::String(tag_text(doc)),
                    },
                    StoredField {
                        field_number: F_COUNT,
                        value: FieldValue::String(count_text(doc)),
                    },
                    StoredField {
                        field_number: F_HEAD,
                        value: FieldValue::String(head_text(doc)),
                    },
                    StoredField {
                        field_number: F_NOTES,
                        value: FieldValue::String(notes_text(doc)),
                    },
                ],
            })
            .expect("add document");
    }
    let sis = writer.commit().expect("commit");
    assert_eq!(
        sis.segments.len(),
        1,
        "the fixture needs one segment for its terms to reach the block thresholds"
    );

    write_manifest(&out_dir);
    println!("wrote positions fixture ({NUM_DOCS} docs) to {out_dir}");
}

/// Documents sampled for occurrence-by-occurrence comparison. Chosen around
/// the format's own boundaries rather than at random: either side of each
/// level-1 span end (8191/8192, 16383/16384), either side of a level-0 block
/// end (255/256, 511/512, ...), the first document of the group-varint tail,
/// the first and last documents, the document whose `dense` occurrences
/// straddle a `.pos` block boundary, and an irregular stride so nothing about
/// the set is a multiple of 256.
fn sample_docs() -> Vec<usize> {
    let mut docs: Vec<usize> = vec![0, 1, 2, NUM_DOCS - 1, NUM_DOCS - 2];
    for boundary in [256usize, 512, 1024, 8192, 16384] {
        docs.push(boundary - 1);
        docs.push(boundary);
        docs.push(boundary + 1);
    }
    // The tail: 20000 = 78 * 256 + 32, so the last full level-0 block ends at
    // document 19967 and the group-varint tail starts at 19968.
    docs.push(78 * 256 - 1);
    docs.push(78 * 256);
    docs.push(dense_block_crossing_doc());
    let mut d = 3usize;
    while d < NUM_DOCS {
        docs.push(d);
        d += 719; // coprime with 256 and with 8192
    }
    docs.retain(|d| *d < NUM_DOCS);
    docs.sort_unstable();
    docs.dedup();
    docs
}

/// The first document whose `dense` occurrences straddle a `.pos` block
/// boundary: its occurrence ordinals span a multiple of 256. Derived by
/// running the same cumulative sum the writer's `.pos` block schedule uses
/// (one block closes every 256 occurrences, doc-boundary-agnostic).
fn dense_block_crossing_doc() -> usize {
    let mut occurrences = 0usize;
    for doc in 0..NUM_DOCS {
        let freq = dense_freq(doc);
        // Crossing means this document opens in one block and closes in the
        // next, i.e. some multiple of 256 lies strictly inside its span.
        if occurrences / 256 != (occurrences + freq - 1) / 256 && doc > 512 {
            return doc;
        }
        occurrences += freq;
    }
    unreachable!("a 60 000-occurrence term must straddle a 256-occurrence block")
}

fn write_manifest(out_dir: &str) {
    let path = format!("{out_dir}/positions-manifest.properties");
    let mut m = std::fs::File::create(&path).expect("create manifest");
    writeln!(m, "num_docs={NUM_DOCS}").unwrap();

    // Term statistics, re-derived by scanning the intended text. These are
    // what pin the fixture as non-degenerate: `dense.doc_freq` below
    // 2 * 8192 would mean no second level-1 span was ever written, and
    // `dense.total_term_freq` below 256 would mean `.pos` never left its
    // vint tail.
    let mut stats: std::collections::BTreeMap<(&str, String), (usize, usize)> =
        std::collections::BTreeMap::new();
    for doc in 0..NUM_DOCS {
        for (field, text) in [
            ("body", body_text(doc)),
            ("title", title_text(doc)),
            ("tag", tag_text(doc)),
            ("count", count_text(doc)),
            ("head", head_text(doc)),
            ("notes", notes_text(doc)),
        ] {
            let mut per_doc: std::collections::BTreeMap<String, usize> =
                std::collections::BTreeMap::new();
            for (term, _, _) in scan_tokens(&text) {
                *per_doc.entry(term).or_default() += 1;
            }
            for (term, freq) in per_doc {
                let entry = stats.entry((field, term)).or_default();
                entry.0 += 1;
                entry.1 += freq;
            }
        }
    }
    for (field, term) in [
        ("body", "dense"),
        ("body", "gap"),
        ("body", "sparse"),
        ("body", "alpha"),
        ("body", "blk256"),
        ("body", "blk257"),
        ("body", "solo"),
        ("body", "duo"),
        ("title", "common"),
        ("tag", "always"),
        ("count", "c0"),
        ("head", "head"),
        ("notes", "note"),
    ] {
        let (doc_freq, ttf) = stats[&(field, term.to_string())];
        writeln!(m, "stat.{field}.{term}.doc_freq={doc_freq}").unwrap();
        writeln!(m, "stat.{field}.{term}.total_term_freq={ttf}").unwrap();
    }
    writeln!(m, "term_count.body={}", count_terms(&stats, "body")).unwrap();
    writeln!(m, "dense_block_crossing_doc={}", dense_block_crossing_doc()).unwrap();
    // Documents containing the adjacent "alpha beta" pair, for a PhraseQuery
    // whose count is a property of the positions and of nothing else.
    writeln!(m, "phrase_doc_count={}", NUM_DOCS.div_ceil(11)).unwrap();

    let samples = sample_docs();
    writeln!(m, "sample_count={}", samples.len()).unwrap();
    for (i, doc) in samples.iter().enumerate() {
        writeln!(m, "sample.{i}.doc={doc}").unwrap();
        let mut entries: Vec<String> = Vec::new();
        for (field, text) in [
            ("body", body_text(*doc)),
            ("title", title_text(*doc)),
            ("head", head_text(*doc)),
            ("notes", notes_text(*doc)),
        ] {
            let with_offsets = matches!(field, "body" | "head");
            let mut per_term: std::collections::BTreeMap<String, Vec<String>> =
                std::collections::BTreeMap::new();
            for (position, (term, start, end)) in scan_tokens(&text).into_iter().enumerate() {
                let (start, end) = if with_offsets {
                    (start as i64, end as i64)
                } else {
                    // Lucene reports -1 from startOffset()/endOffset() when
                    // the field does not index offsets, so that is what the
                    // manifest has to expect: a writer that emitted an offset
                    // region for a positions-only field would otherwise be
                    // invisible here.
                    (-1, -1)
                };
                let payload = match field {
                    "body" => hex(&payload_for(*doc, position)),
                    "notes" => hex(&notes_payload_for(*doc, position)),
                    _ => String::new(),
                };
                per_term
                    .entry(term)
                    .or_default()
                    .push(format!("{position}:{start}:{end}:{payload}"));
            }
            for (term, occurrences) in per_term {
                entries.push(format!("{field}|{term}|{}", occurrences.join(",")));
            }
        }
        writeln!(m, "sample.{i}.entry_count={}", entries.len()).unwrap();
        for (j, entry) in entries.iter().enumerate() {
            writeln!(m, "sample.{i}.{j}={entry}").unwrap();
        }
    }
}

fn count_terms(
    stats: &std::collections::BTreeMap<(&str, String), (usize, usize)>,
    field: &str,
) -> usize {
    stats.keys().filter(|(f, _)| *f == field).count()
}
