//! Cross-engine verification for task #41's multi-segment fan-out/merge
//! (`lucene_search::multi_segment`).
//!
//! **Approach taken, and why**: this crate has no genuine 2+-segment real-Lucene
//! fixture yet (`fixtures/data/` has no multi-segment `IndexWriter` session for
//! a scored query -- `segments_index`'s `_0.si`/`_1.si` fixture is
//! `segment_infos.rs`'s own commit-file fixture, unrelated to term dictionaries/
//! postings). Building one is real work (a new `Gen*.java` generator) that this
//! task's brief explicitly allows deferring in favor of the documented
//! next-best alternative: **combine two already-differentially-verified
//! single-segment fixtures' real recorded scores as if they were two segments
//! of one index**, and confirm the merge produces the same globally-sorted
//! order a human computing max/sort over the concatenated real scores would.
//!
//! Concretely: `fixtures/data/blocktree_index/` (the same real
//! `IndexWriter`-produced segment `tests/scoring_fixtures.rs` already
//! differentially verifies BM25 scores against) is opened **twice** -- once as
//! "segment 0" (`doc_base = 0`) and once as "segment 1" (`doc_base = max_doc`,
//! that same segment's own real `maxDoc`, read from its manifest) -- modeling a
//! genuine two-segment index where segment 1 happens to be a byte-identical
//! copy of segment 0. Because both "segments" are the literal same real,
//! Java-written bytes, every per-segment score is already real, fixture-proven
//! BM25 output (`scoring_fixtures.rs`'s own cross-checks establish that); this
//! test's job is purely to prove the *merge* step (doc-base translation +
//! global re-ranking + truncation) is correct, which is exactly the
//! "looks locally correct, wrong globally" bug class this task's brief warns
//! about. The expected merged order is computed independently in this test by
//! hand-concatenating each segment's own real top-n hits (translated by
//! `doc_base`) and sorting by the same score-desc/doc-id-asc rule real
//! Lucene's `HitQueue` uses -- not by calling the code under test a second
//! time.

use lucene_codecs::blocktree;
use lucene_codecs::field_infos;
use lucene_codecs::postings::DocInput;
use lucene_search::collector::TopDocsCollector;
use lucene_search::{search_term_query_multi_segment, OpenSegment, ScoreDoc, TermQuery};
use lucene_util::fixed_bit_set::FixedBitSet;

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
        let kv = text
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Manifest { kv }
    }

    fn get(&self, key: &str) -> &str {
        self.kv
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
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

/// Opens the same real fixture segment used by `scoring_fixtures.rs`, returning
/// everything needed to build an [`OpenSegment`] plus that segment's real
/// `max_doc` (needed to compute the second copy's `doc_base`).
fn open_real_segment() -> (blocktree::BlockTreeFields, Vec<u8>, [u8; 16], String, i32) {
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
    (fields, doc, id, suffix, max_doc)
}

/// Real Lucene's `HitQueue`/`TopDocs.merge` rank rule (score descending, lower
/// doc ID wins an exact tie) -- reimplemented independently here (not calling
/// into `lucene_search::collector`'s private `rank_order`) so the expected
/// order in this test is derived, not borrowed from the code under test.
fn expected_rank_order(a: &ScoreDoc, b: &ScoreDoc) -> std::cmp::Ordering {
    b.score
        .total_cmp(&a.score)
        .then_with(|| a.doc_id.cmp(&b.doc_id))
}

#[test]
fn multi_segment_merge_matches_real_lucene_scores_from_two_real_segment_copies() {
    let (fields0, doc0, id0, suffix0, max_doc0) = open_real_segment();
    let (fields1, doc1, id1, suffix1, _max_doc1) = open_real_segment();

    let doc_in0 = DocInput::open(&doc0, &id0, &suffix0).expect("open .doc (segment 0)");
    let doc_in1 = DocInput::open(&doc1, &id1, &suffix1).expect("open .doc (segment 1)");

    let query = TermQuery::new("big", "everywhere"); // docFreq == 300, real multi-block postings.
    let top_n = 8;

    // Independently compute each "segment"'s own real top-n hits by calling the
    // already-fixture-verified single-segment scored search directly (this is
    // the ground truth this test cross-checks the multi-segment merge against,
    // not a re-implementation of the scoring math).
    let mut local0 = TopDocsCollector::new(top_n);
    lucene_search::search_term_query_scored(
        &fields0,
        Some(&doc_in0),
        None,
        &query,
        None,
        &mut local0,
    )
    .unwrap();
    let mut local1 = TopDocsCollector::new(top_n);
    lucene_search::search_term_query_scored(
        &fields1,
        Some(&doc_in1),
        None,
        &query,
        None,
        &mut local1,
    )
    .unwrap();

    let doc_base1 = max_doc0; // segment 1 starts right after segment 0's real maxDoc.
    let mut expected: Vec<ScoreDoc> = Vec::new();
    expected.extend(local0.top_docs().iter().copied());
    expected.extend(local1.top_docs().iter().map(|h| ScoreDoc {
        doc_id: h.doc_id + doc_base1,
        score: h.score,
    }));
    expected.sort_by(expected_rank_order);
    expected.truncate(top_n);

    let segments = [
        OpenSegment {
            fields: &fields0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
            live_docs: None,
            doc_base: 0,
        },
        OpenSegment {
            fields: &fields1,
            doc_in: Some(&doc_in1),
            pos_in: None,
            pay_in: None,
            live_docs: None,
            doc_base: doc_base1,
        },
    ];
    let norms = [None, None];

    let actual = search_term_query_multi_segment(&segments, &query, &norms, top_n).unwrap();

    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_eq!(a.doc_id, e.doc_id);
        assert!((a.score - e.score).abs() < 1e-6, "a={a:?} e={e:?}");
    }

    // Since segment 1 is a byte-identical copy of segment 0, every doc ID in
    // segment 1's contribution must be `>= doc_base1`, and the top hit overall
    // must come from whichever segment offers the higher real score at a tie
    // in the *lower* global doc ID -- since both segments hold identical
    // per-local-doc-id scores, doc IDs from segment 0 must rank strictly ahead
    // of their score-identical segment-1 counterparts.
    for hit in &actual {
        let from_seg0 = hit.doc_id < doc_base1;
        let local_id = if from_seg0 {
            hit.doc_id
        } else {
            hit.doc_id - doc_base1
        };
        let counterpart_global = if from_seg0 {
            local_id + doc_base1
        } else {
            local_id
        };
        if let Some(counterpart) = actual.iter().find(|h| h.doc_id == counterpart_global) {
            assert!((counterpart.score - hit.score).abs() < 1e-6);
            // Whichever of the pair appears first in `actual` must be the one
            // with the lower global doc ID (identical scores by construction).
            let hit_pos = actual.iter().position(|h| h.doc_id == hit.doc_id).unwrap();
            let counterpart_pos = actual
                .iter()
                .position(|h| h.doc_id == counterpart.doc_id)
                .unwrap();
            if hit_pos < counterpart_pos {
                assert!(hit.doc_id < counterpart.doc_id);
            }
        }
    }
}

/// The two prior tests use byte-identical copies of the same real segment for
/// both "segments" -- a legitimate way to prove doc-base translation/merge
/// math against real recorded scores, but structurally unable to catch a bug
/// where segment `i`'s own state (fields/doc_in/live_docs) gets swapped with
/// segment `j`'s, since identical content makes such a swap invisible. This
/// test closes that gap cheaply (no new Java fixture needed): it gives
/// segment 1 a distinct `live_docs` bitset that excludes exactly one of the
/// real docs matching the query, so segment 1's real contribution is no
/// longer content-identical to segment 0's. If `search_term_query_multi_segment`
/// ever applied segment 0's `live_docs` (`None`/all-live) to segment 1's data
/// or vice versa, the excluded doc would still appear (or a live one would
/// wrongly vanish) at the wrong global doc id -- this test would then fail.
#[test]
fn multi_segment_merge_respects_each_segments_own_distinct_live_docs() {
    let (fields0, doc0, id0, suffix0, max_doc0) = open_real_segment();
    let (fields1, doc1, id1, suffix1, _max_doc1) = open_real_segment();

    let doc_in0 = DocInput::open(&doc0, &id0, &suffix0).expect("open .doc (segment 0)");
    let doc_in1 = DocInput::open(&doc1, &id1, &suffix1).expect("open .doc (segment 1)");

    let query = TermQuery::new("big", "everywhere");
    let top_n = 8;

    // Find segment 0's real top hit (all docs live) to identify one concrete
    // local doc id known to match the query in this fixture.
    let mut local0 = TopDocsCollector::new(1);
    lucene_search::search_term_query_scored(
        &fields0,
        Some(&doc_in0),
        None,
        &query,
        None,
        &mut local0,
    )
    .unwrap();
    let excluded_local_doc = local0.top_docs()[0].doc_id;

    // Segment 1's live_docs excludes exactly that one doc id (every other doc
    // stays live) -- segment 1's real content now genuinely differs from
    // segment 0's, unlike the identical-copy tests above.
    let mut live1 = FixedBitSet::new(max_doc0 as usize);
    for d in 0..max_doc0 as usize {
        if d != excluded_local_doc as usize {
            live1.set(d);
        }
    }

    let doc_base1 = max_doc0;
    let segments = [
        OpenSegment {
            fields: &fields0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
            live_docs: None, // segment 0: every doc live, unchanged from the tests above.
            doc_base: 0,
        },
        OpenSegment {
            fields: &fields1,
            doc_in: Some(&doc_in1),
            pos_in: None,
            pay_in: None,
            live_docs: Some(&live1), // segment 1: one doc excluded, distinct from segment 0.
            doc_base: doc_base1,
        },
    ];
    let norms = [None, None];

    let actual = search_term_query_multi_segment(&segments, &query, &norms, top_n).unwrap();

    // Segment 0's excluded-in-segment-1 doc must still appear at its own
    // (segment 0) global doc id -- segment 0's live_docs (None) must never be
    // overridden by segment 1's.
    assert!(
        actual.iter().any(|h| h.doc_id == excluded_local_doc),
        "segment 0's live_docs must not be affected by segment 1's exclusion: {actual:?}"
    );
    // Segment 1's counterpart doc (same local id, offset by doc_base1) must be
    // ABSENT -- if live_docs were ever applied to the wrong segment (a swap
    // bug), this doc would wrongly appear, or segment 0's copy would wrongly
    // be missing (already checked above).
    let excluded_global_in_seg1 = excluded_local_doc + doc_base1;
    assert!(
        !actual.iter().any(|h| h.doc_id == excluded_global_in_seg1),
        "segment 1's live_docs exclusion must be honored, not lost or misapplied: {actual:?}"
    );
}

/// Two separate zero-match scenarios, both against the real fixture (no new
/// Java generator needed):
/// (a) a term absent from the fixture entirely -- both segments contribute
///     zero matches, confirming the merge doesn't panic on an all-empty
///     input;
/// (b) a term present in the fixture, but segment 1's `live_docs` excludes
///     every doc that matches it -- segment 1 genuinely contributes zero
///     matches while segment 0 contributes real ones, the actual "one
///     segment empty, the other not" case the test name promises (unlike a
///     query that's the same on both identical-copy segments, which can
///     never be zero-on-one-side-only).
#[test]
fn multi_segment_zero_match_segment_does_not_break_real_fixture_merge() {
    let (fields0, doc0, id0, suffix0, max_doc0) = open_real_segment();
    let (fields1, doc1, id1, suffix1, _) = open_real_segment();

    let doc_in0 = DocInput::open(&doc0, &id0, &suffix0).unwrap();
    let doc_in1 = DocInput::open(&doc1, &id1, &suffix1).unwrap();

    let segments_all_live = [
        OpenSegment {
            fields: &fields0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
            live_docs: None,
            doc_base: 0,
        },
        OpenSegment {
            fields: &fields1,
            doc_in: Some(&doc_in1),
            pos_in: None,
            pay_in: None,
            live_docs: None,
            doc_base: max_doc0,
        },
    ];
    let norms = [None, None];

    // (a) A term that doesn't exist in this fixture at all -- both segments
    // contribute zero matches, and the merge returns empty without panicking.
    let empty_query = TermQuery::new("body", "nonexistent-term-xyz");
    let empty =
        search_term_query_multi_segment(&segments_all_live, &empty_query, &norms, 10).unwrap();
    assert!(empty.is_empty());

    // (b) A term that DOES match in this fixture ("cat" -> docs {0, 2}, per
    // GenBlockTree.java's known contents), but segment 1's live_docs excludes
    // every doc it matches -- segment 1 genuinely contributes zero matches
    // here, segment 0 contributes real ones.
    let real_query = TermQuery::new("body", "cat");
    let mut live1 = FixedBitSet::new(max_doc0 as usize);
    for d in 0..max_doc0 as usize {
        if d != 0 && d != 2 {
            live1.set(d);
        }
    }
    let segments_seg1_excludes_matches = [
        OpenSegment {
            fields: &fields0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
            live_docs: None,
            doc_base: 0,
        },
        OpenSegment {
            fields: &fields1,
            doc_in: Some(&doc_in1),
            pos_in: None,
            pay_in: None,
            live_docs: Some(&live1),
            doc_base: max_doc0,
        },
    ];

    let result =
        search_term_query_multi_segment(&segments_seg1_excludes_matches, &real_query, &norms, 10)
            .unwrap();
    assert!(!result.is_empty());
    // Every hit must come from segment 0's global range only -- segment 1
    // (matches excluded via live_docs) must contribute nothing.
    for hit in &result {
        assert!(
            hit.doc_id < max_doc0,
            "segment 1 must contribute zero matches once its live_docs excludes them: {hit:?}"
        );
    }
}
