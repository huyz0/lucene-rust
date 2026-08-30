//! Differential test for the highlighter's **postings** offset source
//! (`PostingsOffsetStrategy`) against real Lucene's own per-occurrence
//! offsets.
//!
//! `fixtures/src/GenBlockTree.java`'s `"pos"` field is indexed with
//! `IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS`, and its manifest
//! records, per term, `postingsDocs` / `postingsFreqs` / `occurrences`, where
//! each occurrence is `position,startOffset,endOffset,payload` read straight
//! off a real `PostingsEnum`. That is the ground truth `offsets_from_postings`
//! has to reproduce -- a highlighter that reads offsets from the wrong place
//! produces snippets that are silently off by a few characters, which no
//! self-round-trip can catch.

use lucene_codecs::blocktree;
use lucene_codecs::field_infos;
use lucene_codecs::postings::{DocInput, PayInput, PosInput};
use lucene_search::highlighter::{
    offsets_from_phrase, offsets_from_postings, phrase_match_offsets,
};

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenBlockTree)");
        Manifest {
            kv: text
                .lines()
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    fn get(&self, key: &str) -> &str {
        self.kv
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
    }

    fn ints(&self, key: &str) -> Vec<i32> {
        self.get(key)
            .split(',')
            .map(|v| v.parse().unwrap())
            .collect()
    }

    /// Real Lucene's `(startOffset, endOffset)` pairs for `term` in `doc`,
    /// sliced out of the flat per-term occurrence list using that term's own
    /// `postingsDocs`/`postingsFreqs`.
    fn offsets(&self, term: &str, doc: i32) -> Vec<(i32, i32)> {
        let docs = self.ints(&format!("field.pos.term.{term}.postingsDocs"));
        let freqs = self.ints(&format!("field.pos.term.{term}.postingsFreqs"));
        let occurrences: Vec<&str> = self
            .get(&format!("field.pos.term.{term}.occurrences"))
            .split(';')
            .collect();
        let mut cursor = 0usize;
        for (d, f) in docs.iter().zip(freqs.iter()) {
            let take = *f as usize;
            if *d == doc {
                return occurrences[cursor..cursor + take]
                    .iter()
                    .map(|o| {
                        let parts: Vec<&str> = o.split(',').collect();
                        (parts[1].parse().unwrap(), parts[2].parse().unwrap())
                    })
                    .collect();
            }
            cursor += take;
        }
        Vec::new()
    }
}

fn id_from_hex(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

fn read_raw(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}{}.raw", dir(), name)).unwrap_or_else(|_| panic!("missing {name}.raw"))
}

type SegmentFixture = (
    blocktree::BlockTreeFields,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    [u8; 16],
    String,
    Manifest,
);

fn open_segment() -> SegmentFixture {
    let m = Manifest::load();
    let id = id_from_hex(m.get("id_hex"));
    let suffix = m.get("segment_suffix").to_string();
    let max_doc: i32 = m.get("max_doc").parse().unwrap();

    let fnm = read_raw(m.get("fnm_file_name"));
    let field_infos = field_infos::parse(&fnm, &id, "").expect("parse .fnm");
    let tim = read_raw(m.get("tim_file_name"));
    let tip = read_raw(m.get("tip_file_name"));
    let tmd = read_raw(m.get("tmd_file_name"));
    let fields = blocktree::open(&tim, &tip, &tmd, &field_infos, &id, &suffix, max_doc)
        .expect("open blocktree");

    let doc = read_raw(m.get("doc_file_name"));
    let pos = read_raw(m.get("pos_file_name"));
    let pay = read_raw(m.get("pay_file_name"));
    (fields, doc, pos, pay, id, suffix, m)
}

#[test]
fn postings_offset_strategy_matches_real_lucenes_offsets() {
    let (fields, doc, pos, pay, id, suffix, m) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");
    let pos_in = PosInput::open(&pos, &id, &suffix).expect("open .pos");
    let pay_in = PayInput::open(&pay, &id, &suffix).expect("open .pay");

    // doc 8555 carries both terms; alpha occurs twice there, which is what
    // makes the flat-occurrence-list slicing worth checking.
    for doc_id in [8555i32, 8556, 8557] {
        let spans = offsets_from_postings(
            &fields,
            Some(&doc_in),
            &pos_in,
            Some(&pay_in),
            "pos",
            &["alpha", "beta"],
            doc_id,
        )
        .expect("read postings offsets");

        let mut want: Vec<(String, i32, i32)> = Vec::new();
        for term in ["alpha", "beta"] {
            for (s, e) in m.offsets(term, doc_id) {
                want.push((term.to_string(), s, e));
            }
        }
        // `OffsetsEnum.compareTo`: start, then end, then term.
        want.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.cmp(&b.2)).then(a.0.cmp(&b.0)));

        let got: Vec<(String, i32, i32)> = spans
            .iter()
            .map(|s| (s.term.clone(), s.start_offset, s.end_offset))
            .collect();
        assert_eq!(got, want, "doc {doc_id}");
        assert!(!got.is_empty(), "doc {doc_id} must actually have offsets");
    }
}

/// Every `highlight.*` case real Lucene's own `PhraseHelper` recorded
/// (`fixtures/src/AppendHighlightManifest.java`), checked against
/// `offsets_from_phrase`.
///
/// This is the only ground truth that can settle what the highlighter should
/// do with a *reordered* sloppy phrase, because the answer is not the scorer's:
/// `WeightedSpanTermExtractor` rewrites the phrase as
/// `SpanNearQuery(clauses, slop, inOrder = slop == 0)`, whose budget is
/// `maxEnd - minStart - totalSpanLength` rather than `SloppyPhraseMatcher`'s
/// slot-shifted window width. The two disagree by two positions on a
/// transposition, they disagree about whether two slots may share one
/// occurrence, and only Lucene knows which one `PhraseHelper` runs. This port
/// enumerated in order at every slop, so every one of the reordered cases
/// below produced *no* highlight where Lucene produces one.
#[test]
fn phrase_helper_offsets_match_real_lucene() {
    let (fields, doc, pos, pay, id, suffix, m) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");
    let pos_in = PosInput::open(&pos, &id, &suffix).expect("open .pos");
    let pay_in = PayInput::open(&pay, &id, &suffix).expect("open .pay");

    let cases = [
        "exact",
        "reordered_slop0",
        "reordered_slop1",
        "reordered_slop2",
        "reordered_gammadelta",
        "gap_in_order_slop0",
        "gap_in_order_slop2",
        "gap_reordered_slop2",
        "gap_reordered_slop4",
        "repeat_two_occurrences",
        "repeat_single_occurrence",
        "absent_term",
        "single_term",
    ];
    let mut saw_a_reordered_highlight = false;
    for case in cases {
        let doc_id: i32 = m.get(&format!("highlight.{case}.doc")).parse().unwrap();
        let field = m.get(&format!("highlight.{case}.field")).to_string();
        let phrase_text = m.get(&format!("highlight.{case}.phrase")).to_string();
        let phrase: Vec<&str> = phrase_text.split(' ').collect();
        let slop: u32 = m.get(&format!("highlight.{case}.slop")).parse().unwrap();
        let expected = m.get(&format!("highlight.{case}.offsets")).to_string();

        let spans = offsets_from_phrase(
            &fields,
            Some(&doc_in),
            &pos_in,
            Some(&pay_in),
            &field,
            &phrase,
            slop,
            doc_id,
        )
        .unwrap_or_else(|e| panic!("case {case}: {e}"));
        let got = spans
            .iter()
            .map(|s| format!("{}:{},{}", s.term, s.start_offset, s.end_offset))
            .collect::<Vec<_>>()
            .join(";");
        assert_eq!(
            got, expected,
            "case {case}: phrase {phrase:?} at slop {slop} on doc {doc_id}"
        );
        if case.contains("reordered") && !expected.is_empty() {
            saw_a_reordered_highlight = true;
        }
    }
    assert!(
        saw_a_reordered_highlight,
        "the reordered cases must actually produce highlights, or this test proves nothing"
    );
}

#[test]
fn a_term_absent_from_the_document_contributes_no_offsets() {
    let (fields, doc, pos, pay, id, suffix, _m) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");
    let pos_in = PosInput::open(&pos, &id, &suffix).expect("open .pos");
    let pay_in = PayInput::open(&pay, &id, &suffix).expect("open .pay");

    // "beta" is not in doc 8556 (its postingsDocs are 8555, 8557).
    let spans = offsets_from_postings(
        &fields,
        Some(&doc_in),
        &pos_in,
        Some(&pay_in),
        "pos",
        &["beta"],
        8556,
    )
    .unwrap();
    assert!(spans.is_empty());

    // Neither is a term that is not in the dictionary at all, nor a field
    // that does not exist.
    assert!(offsets_from_postings(
        &fields,
        Some(&doc_in),
        &pos_in,
        Some(&pay_in),
        "pos",
        &["no_such_term"],
        8555,
    )
    .unwrap()
    .is_empty());
    assert!(offsets_from_postings(
        &fields,
        Some(&doc_in),
        &pos_in,
        Some(&pay_in),
        "no_such_field",
        &["alpha"],
        8555,
    )
    .unwrap()
    .is_empty());
}

/// A field indexed without offsets yields `-1`/`-1` from `PostingsEnum`; those
/// are not places in the text and must be dropped rather than handed to the
/// formatter. `"body"` is `DOCS_AND_FREQS` in this fixture, so it has no
/// positions at all -- which surfaces as an error from the positions reader,
/// the honest answer for "you asked the wrong strategy for this field".
#[test]
fn a_field_without_positions_is_an_error_not_a_silent_wrong_offset() {
    let (fields, doc, pos, pay, id, suffix, _m) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");
    let pos_in = PosInput::open(&pos, &id, &suffix).expect("open .pos");
    let pay_in = PayInput::open(&pay, &id, &suffix).expect("open .pay");

    let result = offsets_from_postings(
        &fields,
        Some(&doc_in),
        &pos_in,
        Some(&pay_in),
        "body",
        &["cat"],
        0,
    );
    assert!(
        result.is_err(),
        "a DOCS_AND_FREQS field cannot answer an offsets query"
    );
}

// ---------------------------------------------------------------------------
// `PhraseHelper` -- position-sensitive offsets (c12 §3.4)
// ---------------------------------------------------------------------------

/// The whole point of `PhraseHelper`: an occurrence of a phrase term that is
/// *not* inside a phrase match must not be highlighted.
///
/// Doc 8557 has `alpha` at position 0 and `beta` at position 3, so
/// `"alpha beta"` at slop 0 does not match it -- and real Lucene says so, in
/// this fixture's own `field.pos.sloppyGap.realLuceneSlopResults`. The
/// position-insensitive source (`offsets_from_postings`, what this port had
/// before) highlights both terms there regardless.
#[test]
fn phrase_offsets_drop_occurrences_that_are_not_in_a_phrase_match() {
    let (fields, doc, pos, pay, id, suffix, m) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");
    let pos_in = PosInput::open(&pos, &id, &suffix).expect("open .pos");
    let pay_in = PayInput::open(&pay, &id, &suffix).expect("open .pay");

    let gap_doc: i32 = m.get("field.pos.sloppyGapDoc").parse().unwrap();
    let phrase = [
        m.get("field.pos.sloppyGap.termA"),
        m.get("field.pos.sloppyGap.termB"),
    ];

    // Position-insensitive: both terms, every occurrence.
    let insensitive = offsets_from_postings(
        &fields,
        Some(&doc_in),
        &pos_in,
        Some(&pay_in),
        "pos",
        &phrase,
        gap_doc,
    )
    .unwrap();
    assert_eq!(insensitive.len(), 2, "both terms occur in the gap document");

    // Position-sensitive at slop 0: no phrase match, so nothing to highlight.
    let sensitive = offsets_from_phrase(
        &fields,
        Some(&doc_in),
        &pos_in,
        Some(&pay_in),
        "pos",
        &phrase,
        0,
        gap_doc,
    )
    .unwrap();
    assert!(
        sensitive.is_empty(),
        "the terms are {} positions apart, so there is no slop-0 phrase here",
        m.get("field.pos.sloppyGap.movesNeeded")
    );
}

/// ...and which slops *do* match is real Lucene's own answer, recorded per
/// slop in the fixture. The offsets that come back once it matches are exactly
/// the position-insensitive ones, since in this document every occurrence of
/// each term takes part in the single match.
///
/// The recorded answer here is `PhraseQuery`'s (`IndexSearcher` over the real
/// query), and it coincides with the *highlighter's* only because this case is
/// in order: `SpanNearQuery`'s window and `SloppyPhraseMatcher`'s
/// `matchLength` agree when the terms appear in phrase order and do not
/// overlap. Where they disagree -- every reordered case, and a repeated term
/// -- the highlighter's own answer is what `phrase_helper_offsets_match_real_lucene`
/// records, from real Lucene's `PhraseHelper` rather than from `IndexSearcher`.
#[test]
fn phrase_offsets_appear_at_exactly_the_slops_real_lucene_matches_at() {
    let (fields, doc, pos, pay, id, suffix, m) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");
    let pos_in = PosInput::open(&pos, &id, &suffix).expect("open .pos");
    let pay_in = PayInput::open(&pay, &id, &suffix).expect("open .pay");

    let gap_doc: i32 = m.get("field.pos.sloppyGapDoc").parse().unwrap();
    let phrase = [
        m.get("field.pos.sloppyGap.termA"),
        m.get("field.pos.sloppyGap.termB"),
    ];
    let insensitive = offsets_from_postings(
        &fields,
        Some(&doc_in),
        &pos_in,
        Some(&pay_in),
        "pos",
        &phrase,
        gap_doc,
    )
    .unwrap();

    let mut checked = 0;
    for entry in m
        .get("field.pos.sloppyGap.realLuceneSlopResults")
        .split(',')
    {
        let (slop, matches) = entry.split_once(':').unwrap();
        let slop: u32 = slop.parse().unwrap();
        let matches: bool = matches.parse().unwrap();
        let spans = offsets_from_phrase(
            &fields,
            Some(&doc_in),
            &pos_in,
            Some(&pay_in),
            "pos",
            &phrase,
            slop,
            gap_doc,
        )
        .unwrap();
        if matches {
            assert_eq!(spans, insensitive, "slop {slop}");
        } else {
            assert!(spans.is_empty(), "slop {slop}");
        }
        checked += 1;
    }
    assert!(checked >= 5);
}

/// A document where the phrase *does* match adjacently: every span comes back,
/// in `OffsetsEnum` order, with the offsets real Lucene recorded.
#[test]
fn phrase_offsets_of_an_adjacent_match_are_real_lucenes_offsets() {
    let (fields, doc, pos, pay, id, suffix, m) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");
    let pos_in = PosInput::open(&pos, &id, &suffix).expect("open .pos");
    let pay_in = PayInput::open(&pay, &id, &suffix).expect("open .pay");

    // doc 8555: alpha at position 0 (offsets 0..5), beta at position 1.
    let spans = offsets_from_phrase(
        &fields,
        Some(&doc_in),
        &pos_in,
        Some(&pay_in),
        "pos",
        &["alpha", "beta"],
        0,
        8555,
    )
    .unwrap();
    let got: Vec<(String, i32, i32)> = spans
        .iter()
        .map(|s| (s.term.clone(), s.start_offset, s.end_offset))
        .collect();
    let mut want: Vec<(String, i32, i32)> = Vec::new();
    for term in ["alpha", "beta"] {
        for (s, e) in m.offsets(term, 8555) {
            want.push((term.to_string(), s, e));
        }
    }
    want.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.cmp(&b.2)).then(a.0.cmp(&b.0)));
    assert_eq!(got, want);
    assert_eq!(got.len(), 2);
}

/// A phrase term absent from the document means no match at all -- not "the
/// terms that are present", which is what a position-insensitive source
/// returns.
#[test]
fn a_phrase_with_a_missing_term_highlights_nothing() {
    let (fields, doc, pos, pay, id, suffix, _m) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");
    let pos_in = PosInput::open(&pos, &id, &suffix).expect("open .pos");
    let pay_in = PayInput::open(&pay, &id, &suffix).expect("open .pay");

    // "beta" is not in doc 8556, where "alpha" occurs twice.
    assert!(!offsets_from_postings(
        &fields,
        Some(&doc_in),
        &pos_in,
        Some(&pay_in),
        "pos",
        &["alpha"],
        8556,
    )
    .unwrap()
    .is_empty());
    assert!(offsets_from_phrase(
        &fields,
        Some(&doc_in),
        &pos_in,
        Some(&pay_in),
        "pos",
        &["alpha", "beta"],
        99,
        8556,
    )
    .unwrap()
    .is_empty());
    // An unknown field is empty, not an error, like every other source here.
    assert!(offsets_from_phrase(
        &fields,
        Some(&doc_in),
        &pos_in,
        Some(&pay_in),
        "no_such_field",
        &["alpha", "beta"],
        0,
        8555,
    )
    .unwrap()
    .is_empty());
}

/// The collector's own rules, against hand-built occurrences: a repeated term
/// shares one output enum, an offset pair collected by two overlapping
/// matches appears once, and an occurrence outside every match is dropped.
#[test]
fn the_span_collector_deduplicates_and_shares_one_enum_per_term() {
    use lucene_codecs::postings::Position;

    let at = |position: i32, start: i32, end: i32| Position {
        position,
        start_offset: start,
        end_offset: end,
        payload: Vec::new(),
    };

    // Text: "a b a b a". Phrase "a b" matches at positions (0,1) and (2,3);
    // the trailing "a" at position 4 is in no match.
    let a = vec![at(0, 0, 1), at(2, 4, 5), at(4, 8, 9)];
    let b = vec![at(1, 2, 3), at(3, 6, 7)];
    let spans = phrase_match_offsets(&["a", "b"], &[a.clone(), b.clone()], 0);
    let got: Vec<(&str, i32, i32)> = spans
        .iter()
        .map(|s| (s.term.as_str(), s.start_offset, s.end_offset))
        .collect();
    assert_eq!(
        got,
        vec![("a", 0, 1), ("b", 2, 3), ("a", 4, 5), ("b", 6, 7)],
        "the position-4 'a' takes part in no match and must not be highlighted"
    );

    // A repeated term: "a a" over "a a a" matches at (0,1) and (1,2), so the
    // middle occurrence is collected twice and must appear once.
    let aaa = vec![at(0, 0, 1), at(1, 2, 3), at(2, 4, 5)];
    let spans = phrase_match_offsets(&["a", "a"], &[aaa.clone(), aaa.clone()], 0);
    let got: Vec<(&str, i32, i32)> = spans
        .iter()
        .map(|s| (s.term.as_str(), s.start_offset, s.end_offset))
        .collect();
    assert_eq!(got, vec![("a", 0, 1), ("a", 2, 3), ("a", 4, 5)]);

    // Degenerate inputs are empty, not a panic.
    assert!(phrase_match_offsets(&[], &[], 0).is_empty());
    assert!(phrase_match_offsets(&["a"], &[], 0).is_empty());
    assert!(phrase_match_offsets(&["a", "b"], &[a.clone(), Vec::new()], 0).is_empty());

    // A field with positions but no offsets reports -1, which is not a place
    // in the text: dropped rather than passed on.
    let no_offsets = vec![at(0, -1, -1)];
    let no_offsets_b = vec![at(1, -1, -1)];
    assert!(phrase_match_offsets(&["a", "b"], &[no_offsets, no_offsets_b], 0).is_empty());

    // A single-term "phrase" degenerates to every occurrence of that term.
    let spans = phrase_match_offsets(&["a"], &[a], 0);
    assert_eq!(spans.len(), 3);
}
