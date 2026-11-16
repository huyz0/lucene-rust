#![cfg(test)]
use super::*;
use crate::query::TermQuery as TQ;
use crate::{BooleanQuery, TermQuery};
use lucene_codecs::{blocktree, field_infos};

fn fixture_dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_index/"
    )
    .to_string()
}

/// The fan-out functions are collector plumbing; these two tests drive them
/// with a synthetic per-segment closure so the merge, the doc-base
/// translation and the shared accumulator are exercised without an index.
fn stub_segments(hits: &[Vec<(i32, f32)>]) -> Vec<i32> {
    // Doc bases 0, 100, 200, ...: wide enough that a global doc id can only
    // come out right if the translation happened.
    (0..hits.len() as i32)
        .map(|i| i.saturating_mul(100))
        .collect()
}

#[test]
fn the_after_fan_out_translates_the_global_doc_id_into_each_leafs_own_space() {
    // Two leaves, all scores tied, so the page boundary is decided purely
    // by the doc-id half of the rule -- which is the half the doc-base
    // subtraction feeds.
    let per_leaf = vec![
        vec![(0i32, 1.0f32), (1, 1.0), (2, 1.0)],
        vec![(0i32, 1.0f32), (1, 1.0), (2, 1.0)],
    ];
    let doc_bases = stub_segments(&per_leaf);
    let run = |after: Option<ScoreDoc>| {
        let search = |i: usize, local: &mut TopDocsCollector| {
            for &(doc, score) in &per_leaf[i] {
                local.collect(doc, score);
            }
            Ok(())
        };
        match after {
            None => merge_multi_segment_scored(&doc_bases, 2, search).unwrap(),
            Some(a) => merge_multi_segment_scored_after(&doc_bases, 2, a, search).unwrap(),
        }
    };
    let page1 = run(None);
    assert_eq!(
        page1.iter().map(|h| h.doc_id).collect::<Vec<_>>(),
        vec![0, 1]
    );
    let page2 = run(Some(page1[1]));
    assert_eq!(
        page2.iter().map(|h| h.doc_id).collect::<Vec<_>>(),
        vec![2, 100]
    );
    // Page 3 straddles the boundary the other way: `after` is now in the
    // second leaf, so the first leaf's translated `afterDoc` is 100 and
    // every one of its documents is correctly dropped.
    let page3 = run(Some(page2[1]));
    assert_eq!(
        page3.iter().map(|h| h.doc_id).collect::<Vec<_>>(),
        vec![101, 102]
    );
}

#[test]
fn the_shared_max_score_fan_out_returns_what_the_plain_one_does() {
    // The accumulator may only *raise* a leaf's pruning threshold to one
    // another leaf has already proved competitive, so it can change how
    // much work is done and never which hits come back. That result
    // invariant is what this test asserts, and it holds however the two
    // leaves interleave.
    //
    // **It deliberately does not assert that leaf 1 observed leaf 0's
    // bar.** The design does not guarantee that: the accumulator is
    // opportunistic, so whether leaf 1 sees 8.0 depends on whether leaf 0
    // got there first, which under `rayon` it need not. The mechanism
    // itself is pinned
    // deterministically at its seam by
    // `a_leaf_sees_a_bar_another_leaf_published_through_the_accumulator`.
    let per_leaf = vec![
        vec![(0i32, 9.0f32), (1, 8.0), (2, 7.0)],
        vec![(0i32, 3.0f32), (1, 2.0), (2, 1.0)],
    ];
    let doc_bases = stub_segments(&per_leaf);
    let run = |shared: bool| {
        let search = |i: usize, local: &mut TopDocsCollector| {
            for &(doc, score) in &per_leaf[i] {
                local.collect(doc, score);
            }
            Ok(())
        };
        if shared {
            merge_multi_segment_scored_concurrent_shared_max_score(&doc_bases, 2, 0, search)
                .unwrap()
        } else {
            merge_multi_segment_scored_concurrent(&doc_bases, 2, search).unwrap()
        }
    };
    let shared = run(true);
    let plain = run(false);

    assert_eq!(shared, plain);
    assert_eq!(
        shared.iter().map(|h| h.doc_id).collect::<Vec<_>>(),
        vec![0, 1]
    );
}

/// The accumulator's own contract, at the seam
/// [`merge_multi_segment_scored_concurrent_shared_max_score`] wires up --
/// with no threads, so there is no interleaving for it to depend on.
///
/// Leaf 1's own hits top out at 3.0, so a threshold of 8.0 is a value its
/// own queue can never produce: observing it is the only evidence the
/// accumulator was consulted at all.
#[test]
fn a_leaf_sees_a_bar_another_leaf_published_through_the_accumulator() {
    use crate::collector::MaxScoreAccumulator;
    use std::sync::Arc;

    let acc = Arc::new(MaxScoreAccumulator::new());

    // Leaf 0 fills a top-2 queue whose bottom is 8.0 and publishes it.
    let mut leaf0 = TopDocsCollector::with_total_hits_threshold(2, 0)
        .with_shared_max_score(Arc::clone(&acc), 0);
    for (doc, score) in [(0i32, 9.0f32), (1, 8.0), (2, 7.0)] {
        leaf0.collect(doc, score);
    }
    assert_eq!(leaf0.min_competitive_score(), Some(8.0));

    // Leaf 1, sharing the accumulator, is handed that bar even though its
    // own queue is nowhere near it.
    let mut leaf1 = TopDocsCollector::with_total_hits_threshold(2, 0)
        .with_shared_max_score(Arc::clone(&acc), 100);
    // Three hits: a threshold is published only past `top_n`, the floor
    // Java's `Math.max(totalHitsThreshold, numHits)` puts under it.
    for (doc, score) in [(0i32, 3.0f32), (1, 2.0), (2, 1.0)] {
        leaf1.collect(doc, score);
    }
    assert_eq!(
        leaf1.min_competitive_score(),
        Some(8.0),
        "the shared accumulator must raise this leaf's bar to the other leaf's"
    );

    // The negative control: the same leaf without the accumulator can only
    // ever see its own queue's bottom.
    let mut alone = TopDocsCollector::with_total_hits_threshold(2, 0);
    for (doc, score) in [(0i32, 3.0f32), (1, 2.0), (2, 1.0)] {
        alone.collect(doc, score);
    }
    assert_eq!(alone.min_competitive_score(), Some(2.0));
}

#[test]
fn an_exhaustive_shared_fan_out_publishes_no_threshold_at_all() {
    let per_leaf = vec![vec![(0i32, 5.0f32), (1, 4.0)]];
    let doc_bases = stub_segments(&per_leaf);
    let seen = std::sync::Mutex::new(Vec::<Option<f32>>::new());
    let hits = merge_multi_segment_scored_concurrent_shared_max_score(
        &doc_bases,
        1,
        u64::MAX,
        |i, local| {
            for &(doc, score) in &per_leaf[i] {
                local.collect(doc, score);
                seen.lock()
                    .expect("no test thread panics while holding this")
                    .push(local.pruning_threshold());
            }
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(hits.len(), 1);
    assert!(seen
        .lock()
        .expect("no test thread panics while holding this")
        .iter()
        .all(Option::is_none));
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", fixture_dir()))
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
    std::fs::read(format!("{}{}.raw", fixture_dir(), name))
        .unwrap_or_else(|_| panic!("missing {name}.raw"))
}

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

/// End-to-end exercise of [`search_term_query_multi_segment`] itself (not
/// just the generic merge core) against two real, `IndexWriter`-produced
/// segment copies, proving the thin wrapper actually calls
/// `search_term_query_scored` per segment and merges the real result.
#[test]
fn search_term_query_multi_segment_merges_two_real_segments() {
    let (fields0, doc0, id0, suffix0, max_doc0) = open_real_segment();
    let (fields1, doc1, id1, suffix1, _) = open_real_segment();
    let doc_in0 = lucene_codecs::postings::DocInput::open(&doc0, &id0, &suffix0).unwrap();
    let doc_in1 = lucene_codecs::postings::DocInput::open(&doc1, &id1, &suffix1).unwrap();

    let query = TermQuery::new("body", "cat");
    let segments = [
        OpenSegment {
            fields: &fields0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
            live_docs: None,
            doc_base: 0,
            max_doc: None,
            cache: None,
            reader: None,
            index_sort_prefix: false,
            termless: None,
            similarity: None,
            points: None,
        },
        OpenSegment {
            fields: &fields1,
            doc_in: Some(&doc_in1),
            pos_in: None,
            pay_in: None,
            live_docs: None,
            doc_base: max_doc0,
            cache: None,
            reader: None,
            index_sort_prefix: false,
            termless: None,
            similarity: None,
            points: None,
            max_doc: None,
        },
    ];
    let norms = [None, None];

    let merged = search_term_query_multi_segment(&segments, &query, &norms, 10).unwrap();
    assert!(!merged.is_empty());
    // Every hit from segment 1 must have a doc id >= doc_base (translated).
    for hit in &merged {
        assert!(hit.doc_id >= 0);
    }
    // Ranking must be non-increasing by score.
    for pair in merged.windows(2) {
        assert!(pair[0].score >= pair[1].score);
    }
}

/// End-to-end exercise of [`search_boolean_query_multi_segment_maxscore`]
/// against two real, `IndexWriter`-produced segment copies
/// (the same "open the one real fixture segment twice" two-segment
/// simulation `search_term_query_multi_segment_merges_two_real_segments`
/// and `search_boolean_query_multi_segment_merges_two_real_segments`
/// above already use, documented in this crate's `docs/parity.md` as the
/// documented next-best alternative to a genuine multi-segment Java
/// fixture). Uses the exact same three-clause query
/// (`big`/`"everywhere"` -- `docFreq == 300`, spanning a real level-0
/// block plus a tail -- and `body`/`"cat"`, `body`/`"dog"`) `lib.rs`'s own
/// `boolean_maxscore_lazy_path_matches_eager_path_on_real_fixture`/
/// `test_only_boolean_maxscore_block_skip_counter_records_real_skips`
/// tests use for the single-segment case, so this test can both prove
/// identical output *and* prove real per-clause block-decode skipping
/// happened, exactly like those two single-segment tests do individually.
///
/// Asserts, for several `top_n` values:
/// - the merged, globally-ranked output of
///   [`search_boolean_query_multi_segment_maxscore`] is byte-identical to
///   [`search_boolean_query_multi_segment`] (the exhaustive multi-segment
///   path) -- MAXSCORE pruning must never change the merged result, only
///   skip unnecessary per-segment scoring work;
/// - [`test_only_maxscore_block_skip_counter`] recorded at least one real
///   skip across the two segments for small `top_n`, ruling out a
///   vacuously-passing test where the fast path is silently never taken
///   (mirroring the single-segment skip-counter proof one level down).
#[test]
fn search_boolean_query_multi_segment_maxscore_matches_exhaustive_and_skips_blocks() {
    let (fields0, doc0, id0, suffix0, max_doc0) = open_real_segment();
    let (fields1, doc1, id1, suffix1, _) = open_real_segment();
    let doc_in0 = lucene_codecs::postings::DocInput::open(&doc0, &id0, &suffix0).unwrap();
    let doc_in1 = lucene_codecs::postings::DocInput::open(&doc1, &id1, &suffix1).unwrap();

    let query = BooleanQuery::new().with_should([
        TQ::new("big", b"everywhere".as_slice()),
        TQ::new("body", "cat"),
        TQ::new("body", "dog"),
    ]);
    let segments = [
        OpenSegment {
            fields: &fields0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
            live_docs: None,
            doc_base: 0,
            max_doc: None,
            cache: None,
            reader: None,
            index_sort_prefix: false,
            termless: None,
            similarity: None,
            points: None,
        },
        OpenSegment {
            fields: &fields1,
            doc_in: Some(&doc_in1),
            pos_in: None,
            pay_in: None,
            live_docs: None,
            doc_base: max_doc0,
            cache: None,
            reader: None,
            index_sort_prefix: false,
            termless: None,
            similarity: None,
            points: None,
            max_doc: None,
        },
    ];
    let norms = [None, None];

    for &top_n in &[1usize, 2, 5, 20, 300] {
        let exhaustive =
            search_boolean_query_multi_segment(&segments, &query, &norms, top_n).unwrap();

        crate::test_only_maxscore_block_skip_counter::reset();
        let maxscore =
            search_boolean_query_multi_segment_maxscore(&segments, &query, &norms, top_n).unwrap();
        let skips = crate::test_only_maxscore_block_skip_counter::count();

        assert_eq!(
            exhaustive, maxscore,
            "top_{top_n}: multi-segment MAXSCORE output must match the exhaustive \
                 multi-segment path byte-for-byte"
        );

        if top_n < 300 {
            assert!(
                skips > 0,
                "top_{top_n}: expected real per-clause block skipping across the two \
                     segments (got {skips} skips)"
            );
        }
    }
}

/// Concurrent sibling of the test above:
/// [`search_boolean_query_multi_segment_maxscore_concurrent`] must
/// produce byte-for-byte identical output to the sequential
/// [`search_boolean_query_multi_segment_maxscore`] for the same input --
/// same "merge already-ranked per-segment output is agnostic to fan-out
/// order" argument [`merge_multi_segment_scored_concurrent`]'s own doc
/// comment already makes.
#[test]
fn search_boolean_query_multi_segment_maxscore_concurrent_matches_sequential() {
    let (fields0, doc0, id0, suffix0, max_doc0) = open_real_segment();
    let (fields1, doc1, id1, suffix1, _) = open_real_segment();
    let doc_in0 = lucene_codecs::postings::DocInput::open(&doc0, &id0, &suffix0).unwrap();
    let doc_in1 = lucene_codecs::postings::DocInput::open(&doc1, &id1, &suffix1).unwrap();

    let query = BooleanQuery::new().with_should([
        TQ::new("big", b"everywhere".as_slice()),
        TQ::new("body", "cat"),
        TQ::new("body", "dog"),
    ]);
    let segments = [
        OpenSegment {
            fields: &fields0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
            live_docs: None,
            doc_base: 0,
            max_doc: None,
            cache: None,
            reader: None,
            index_sort_prefix: false,
            termless: None,
            similarity: None,
            points: None,
        },
        OpenSegment {
            fields: &fields1,
            doc_in: Some(&doc_in1),
            pos_in: None,
            pay_in: None,
            live_docs: None,
            doc_base: max_doc0,
            cache: None,
            reader: None,
            index_sort_prefix: false,
            termless: None,
            similarity: None,
            points: None,
            max_doc: None,
        },
    ];
    let norms = [None, None];

    let sequential =
        search_boolean_query_multi_segment_maxscore(&segments, &query, &norms, 5).unwrap();
    let concurrent =
        search_boolean_query_multi_segment_maxscore_concurrent(&segments, &query, &norms, 5)
            .unwrap();
    assert!(!sequential.is_empty());
    assert_eq!(sequential, concurrent);
}

/// Same end-to-end exercise for [`search_boolean_query_multi_segment`].
#[test]
fn search_boolean_query_multi_segment_merges_two_real_segments() {
    let (fields0, doc0, id0, suffix0, max_doc0) = open_real_segment();
    let (fields1, doc1, id1, suffix1, _) = open_real_segment();
    let doc_in0 = lucene_codecs::postings::DocInput::open(&doc0, &id0, &suffix0).unwrap();
    let doc_in1 = lucene_codecs::postings::DocInput::open(&doc1, &id1, &suffix1).unwrap();

    let query = BooleanQuery::new().with_should([TQ::new("body", "cat"), TQ::new("body", "bird")]);
    let segments = [
        OpenSegment {
            fields: &fields0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
            live_docs: None,
            doc_base: 0,
            max_doc: None,
            cache: None,
            reader: None,
            index_sort_prefix: false,
            termless: None,
            similarity: None,
            points: None,
        },
        OpenSegment {
            fields: &fields1,
            doc_in: Some(&doc_in1),
            pos_in: None,
            pay_in: None,
            live_docs: None,
            doc_base: max_doc0,
            cache: None,
            reader: None,
            index_sort_prefix: false,
            termless: None,
            similarity: None,
            points: None,
            max_doc: None,
        },
    ];
    let norms = [None, None];

    let merged = search_boolean_query_multi_segment(&segments, &query, &norms, 10).unwrap();
    assert!(!merged.is_empty());
    for pair in merged.windows(2) {
        assert!(pair[0].score >= pair[1].score);
    }
}

/// A single-segment call (the degenerate `segments.len() == 1` case) must
/// still translate/merge correctly -- `doc_base != 0` here to prove the
/// translation isn't accidentally skipped for the one-segment case.
#[test]
fn single_segment_with_nonzero_doc_base_still_translates_correctly() {
    let (fields0, doc0, id0, suffix0, _max_doc0) = open_real_segment();
    let doc_in0 = lucene_codecs::postings::DocInput::open(&doc0, &id0, &suffix0).unwrap();
    let query = TermQuery::new("body", "cat");
    let segments = [OpenSegment {
        fields: &fields0,
        doc_in: Some(&doc_in0),
        pos_in: None,
        pay_in: None,
        live_docs: None,
        doc_base: 1000,
        max_doc: None,
        cache: None,
        reader: None,
        index_sort_prefix: false,
        termless: None,
        similarity: None,
        points: None,
    }];
    let norms = [None];
    let merged = search_term_query_multi_segment(&segments, &query, &norms, 10).unwrap();
    assert!(!merged.is_empty());
    for hit in &merged {
        assert!(hit.doc_id >= 1000);
    }
}

/// Synthetic per-segment "search": a closure-free stand-in producing
/// preset `(local_doc_id, score)` pairs, feeding them straight into the
/// per-segment collector `merge_multi_segment_scored` supplies -- this
/// tests the merge core in isolation from any real blocktree/postings
/// decoding, since the merge/doc-base-translation/tie-break logic is
/// exactly the highest-risk "looks locally correct, wrong globally" class
/// of bug.
fn fake_segment_search(hits: Vec<(i32, f32)>) -> impl FnMut(&mut TopDocsCollector) {
    move |local: &mut TopDocsCollector| {
        for &(doc_id, score) in &hits {
            local.collect(doc_id, score);
        }
    }
}

/// The shared-collector driver must return exactly what the
/// collector-per-leaf merge returns -- same hits, same order, same tie-break
/// across leaves -- on inputs with many cross-leaf score ties.
#[test]
fn shared_collector_matches_the_per_leaf_merge() {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for round in 0..200 {
        let leaves = 1 + (next() % 6) as usize;
        let mut doc_bases = Vec::new();
        let mut hits: Vec<Vec<(i32, f32)>> = Vec::new();
        let mut base = 0i32;
        for _ in 0..leaves {
            doc_bases.push(base);
            let max_doc = (next() % 40) as i32;
            // Coarse scores, so ties across leaves are common.
            let mut leaf = Vec::new();
            for d in 0..max_doc {
                if next() % 3 != 0 {
                    leaf.push((d, (next() % 5) as f32));
                }
            }
            hits.push(leaf);
            base += max_doc;
        }
        for top_n in [0usize, 1, 3, 10, 100] {
            let per_leaf = merge_multi_segment_scored(&doc_bases, top_n, |i, local| {
                hits[i].iter().for_each(|&(d, s)| local.collect(d, s));
                Ok(())
            })
            .unwrap();
            let shared = search_leaves_shared(&doc_bases, top_n, |i, leaf| {
                hits[i].iter().for_each(|&(d, s)| leaf.collect(d, s));
                Ok(())
            })
            .unwrap();
            assert_eq!(per_leaf, shared, "round {round} top_n {top_n}");
        }
    }
}

/// The point of sharing: a later leaf sees the threshold the earlier ones
/// raised, and a leaf's error still aborts the search.
#[test]
fn shared_collector_carries_the_threshold_into_later_leaves() {
    let mut seen = Vec::new();
    search_leaves_shared(&[0, 100], 2, |i, leaf| {
        seen.push(leaf.min_competitive_score());
        assert_eq!(leaf.score_mode(), TopDocsCollector::new(2).score_mode());
        let _ = leaf.pruning_threshold();
        let _ = leaf.constant_score_hits_needed();
        if i == 0 {
            for d in 0..2000 {
                leaf.collect(d % 100, 3.0);
            }
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(seen[0], None);
    assert!(seen[1].is_some(), "{seen:?}");

    let err = search_leaves_shared(&[0, 5], 2, |i, _| {
        if i == 1 {
            Err(crate::Error::MissingPointsInput("x".into()))
        } else {
            Ok(())
        }
    });
    assert!(err.is_err());
}

#[test]
fn merges_interleaved_scores_across_three_segments_in_global_order() {
    // Segment 0: local docs 0,1,2 -> doc_base 0 -> global 0,1,2.
    // Segment 1: local docs 0,1   -> doc_base 3 -> global 3,4.
    // Segment 2: local docs 0,1,2 -> doc_base 5 -> global 5,6,7.
    let doc_bases = [0, 3, 5];
    let mut seg0 = fake_segment_search(vec![(0, 1.0), (1, 5.0), (2, 3.0)]);
    let mut seg1 = fake_segment_search(vec![(0, 4.0), (1, 2.0)]);
    let mut seg2 = fake_segment_search(vec![(0, 6.0), (1, 0.5), (2, 4.5)]);
    let result = merge_multi_segment_scored(&doc_bases, 10, |i, local| {
        match i {
            0 => seg0(local),
            1 => seg1(local),
            2 => seg2(local),
            _ => unreachable!(),
        }
        Ok(())
    })
    .unwrap();
    // Global doc IDs: seg0 -> {0:1.0, 1:5.0, 2:3.0}, seg1 -> {3:4.0, 4:2.0},
    // seg2 -> {5:6.0, 6:0.5, 7:4.5}.
    // Expected score-descending order: 5(6.0), 1(5.0), 7(4.5), 3(4.0),
    // 2(3.0), 4(2.0), 0(1.0), 6(0.5).
    let expected: Vec<(i32, f32)> = vec![
        (5, 6.0),
        (1, 5.0),
        (7, 4.5),
        (3, 4.0),
        (2, 3.0),
        (4, 2.0),
        (0, 1.0),
        (6, 0.5),
    ];
    let actual: Vec<(i32, f32)> = result.iter().map(|d| (d.doc_id, d.score)).collect();
    assert_eq!(actual, expected);
}

#[test]
fn truncates_merged_result_to_top_n() {
    let doc_bases = [0, 10];
    let mut seg0 = fake_segment_search(vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
    let mut seg1 = fake_segment_search(vec![(0, 4.0), (1, 5.0)]);
    let result = merge_multi_segment_scored(&doc_bases, 2, |i, local| {
        match i {
            0 => seg0(local),
            1 => seg1(local),
            _ => unreachable!(),
        }
        Ok(())
    })
    .unwrap();
    let actual: Vec<(i32, f32)> = result.iter().map(|d| (d.doc_id, d.score)).collect();
    // Global scores: 0:1.0, 1:2.0, 2:3.0, 10:4.0, 11:5.0 -- top 2 by score.
    assert_eq!(actual, vec![(11, 5.0), (10, 4.0)]);
}

#[test]
fn segment_with_zero_matches_does_not_break_the_merge() {
    let doc_bases = [0, 2, 3];
    let mut seg0 = fake_segment_search(vec![(0, 1.0), (1, 2.0)]);
    let mut seg1 = fake_segment_search(vec![]); // no matches in this segment.
    let mut seg2 = fake_segment_search(vec![(0, 3.0)]);
    let result = merge_multi_segment_scored(&doc_bases, 10, |i, local| {
        match i {
            0 => seg0(local),
            1 => seg1(local),
            2 => seg2(local),
            _ => unreachable!(),
        }
        Ok(())
    })
    .unwrap();
    let actual: Vec<(i32, f32)> = result.iter().map(|d| (d.doc_id, d.score)).collect();
    assert_eq!(actual, vec![(3, 3.0), (1, 2.0), (0, 1.0)]);
}

#[test]
fn tie_break_prefers_lower_global_doc_id_across_segments() {
    // Two segments each contribute one doc at the exact same score;
    // real Lucene's HitQueue tie-break (lower doc ID wins) must apply
    // globally, not just within one segment.
    let doc_bases = [0, 100];
    let mut seg0 = fake_segment_search(vec![(5, 3.0)]); // global doc 5.
    let mut seg1 = fake_segment_search(vec![(2, 3.0)]); // global doc 102.
    let result = merge_multi_segment_scored(&doc_bases, 10, |i, local| {
        match i {
            0 => seg0(local),
            1 => seg1(local),
            _ => unreachable!(),
        }
        Ok(())
    })
    .unwrap();
    let actual: Vec<(i32, f32)> = result.iter().map(|d| (d.doc_id, d.score)).collect();
    assert_eq!(actual, vec![(5, 3.0), (102, 3.0)]);
}

#[test]
fn top_n_zero_returns_nothing() {
    let doc_bases = [0, 5];
    let mut seg0 = fake_segment_search(vec![(0, 1.0)]);
    let mut seg1 = fake_segment_search(vec![(0, 2.0)]);
    let result = merge_multi_segment_scored(&doc_bases, 0, |i, local| {
        match i {
            0 => seg0(local),
            1 => seg1(local),
            _ => unreachable!(),
        }
        Ok(())
    })
    .unwrap();
    assert!(result.is_empty());
}

#[test]
fn no_segments_returns_empty() {
    let doc_bases: [i32; 0] = [];
    let result: Vec<ScoreDoc> = merge_multi_segment_scored(&doc_bases, 10, |_, _| Ok(())).unwrap();
    assert!(result.is_empty());
}

// --- Concurrent (rayon) path: must be byte-for-byte identical to the
// sequential path for the same input. ---

/// Builds `n` synthetic segments, each contributing a small, distinctly
/// scored set of hits, with doc bases spaced far enough apart that global
/// doc IDs never collide -- shared by every concurrent-vs-sequential test
/// below so both paths run over the exact same fan-out.
fn synthetic_doc_bases_and_hits(n: usize) -> (Vec<i32>, Vec<Vec<(i32, f32)>>) {
    let mut doc_bases = Vec::with_capacity(n);
    let mut hits = Vec::with_capacity(n);
    for i in 0..n {
        doc_bases.push((i as i32) * 100);
        // Distinct, deterministic scores per segment/doc so tie-breaks
        // and ordering are exercised the same way every run.
        hits.push(vec![
            (0, (i as f32) * 0.37 + 1.0),
            (1, (i as f32) * 0.11 + 2.0),
            (2, (i as f32 % 3.0) + 0.5),
        ]);
    }
    (doc_bases, hits)
}

fn run_sequential(doc_bases: &[i32], hits: &[Vec<(i32, f32)>], top_n: usize) -> Vec<ScoreDoc> {
    merge_multi_segment_scored(doc_bases, top_n, |i, local| {
        for &(doc_id, score) in &hits[i] {
            local.collect(doc_id, score);
        }
        Ok(())
    })
    .unwrap()
}

fn run_concurrent(doc_bases: &[i32], hits: &[Vec<(i32, f32)>], top_n: usize) -> Vec<ScoreDoc> {
    merge_multi_segment_scored_concurrent(doc_bases, top_n, |i, local| {
        for &(doc_id, score) in &hits[i] {
            local.collect(doc_id, score);
        }
        Ok(())
    })
    .unwrap()
}

fn assert_identical(a: &[ScoreDoc], b: &[ScoreDoc]) {
    let a: Vec<(i32, f32)> = a.iter().map(|d| (d.doc_id, d.score)).collect();
    let b: Vec<(i32, f32)> = b.iter().map(|d| (d.doc_id, d.score)).collect();
    assert_eq!(a, b, "sequential and concurrent results must be identical");
}

#[test]
fn sequential_merge_propagates_per_segment_search_error() {
    let doc_bases = vec![0, 10, 20];
    let err = merge_multi_segment_scored(&doc_bases, 10, |i, _local| {
        if i == 1 {
            Err(crate::Error::MissingPosInput)
        } else {
            Ok(())
        }
    })
    .unwrap_err();
    assert!(matches!(err, crate::Error::MissingPosInput));
}

#[test]
fn concurrent_merge_propagates_per_segment_search_error() {
    let doc_bases = vec![0, 10, 20];
    let err = merge_multi_segment_scored_concurrent(&doc_bases, 10, |i, _local| {
        if i == 1 {
            Err(crate::Error::MissingPosInput)
        } else {
            Ok(())
        }
    })
    .unwrap_err();
    assert!(matches!(err, crate::Error::MissingPosInput));
}

#[test]
fn concurrent_matches_sequential_empty_index() {
    let doc_bases: Vec<i32> = vec![];
    let hits: Vec<Vec<(i32, f32)>> = vec![];
    let seq = run_sequential(&doc_bases, &hits, 10);
    let con = run_concurrent(&doc_bases, &hits, 10);
    assert!(seq.is_empty());
    assert_identical(&seq, &con);
}

#[test]
fn concurrent_matches_sequential_single_segment() {
    let (doc_bases, hits) = synthetic_doc_bases_and_hits(1);
    let seq = run_sequential(&doc_bases, &hits, 10);
    let con = run_concurrent(&doc_bases, &hits, 10);
    assert!(!seq.is_empty());
    assert_identical(&seq, &con);
}

#[test]
fn concurrent_matches_sequential_many_segments() {
    // 16 segments -- enough for rayon's global pool to plausibly run
    // more than one in parallel.
    let (doc_bases, hits) = synthetic_doc_bases_and_hits(16);
    let seq = run_sequential(&doc_bases, &hits, 10);
    let con = run_concurrent(&doc_bases, &hits, 10);
    assert!(!seq.is_empty());
    assert_identical(&seq, &con);
}

#[test]
fn concurrent_matches_sequential_with_ties_across_segments() {
    // Every segment contributes the exact same score at local doc 0 --
    // forces the same score-tie, ordered-by-global-doc-id path the
    // sequential merge already covers, now also through the concurrent
    // merge.
    let doc_bases: Vec<i32> = (0..8).map(|i| i * 10).collect();
    let hits: Vec<Vec<(i32, f32)>> = (0..8).map(|_| vec![(0, 3.0)]).collect();
    let seq = run_sequential(&doc_bases, &hits, 100);
    let con = run_concurrent(&doc_bases, &hits, 100);
    assert_identical(&seq, &con);
}

#[test]
fn concurrent_matches_sequential_with_top_n_truncation() {
    let (doc_bases, hits) = synthetic_doc_bases_and_hits(10);
    let seq = run_sequential(&doc_bases, &hits, 3);
    let con = run_concurrent(&doc_bases, &hits, 3);
    assert_eq!(seq.len(), 3);
    assert_identical(&seq, &con);
}

/// End-to-end: [`search_term_query_multi_segment_concurrent`] against the
/// same two real segments [`search_term_query_multi_segment_merges_two_real_segments`]
/// uses, proving the concurrent wrapper (not just the generic merge core)
/// matches the sequential wrapper exactly.
#[test]
fn search_term_query_multi_segment_concurrent_matches_sequential() {
    let (fields0, doc0, id0, suffix0, max_doc0) = open_real_segment();
    let (fields1, doc1, id1, suffix1, _) = open_real_segment();
    let doc_in0 = lucene_codecs::postings::DocInput::open(&doc0, &id0, &suffix0).unwrap();
    let doc_in1 = lucene_codecs::postings::DocInput::open(&doc1, &id1, &suffix1).unwrap();

    let query = TermQuery::new("body", "cat");
    let segments = [
        OpenSegment {
            fields: &fields0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
            live_docs: None,
            doc_base: 0,
            max_doc: None,
            cache: None,
            reader: None,
            index_sort_prefix: false,
            termless: None,
            similarity: None,
            points: None,
        },
        OpenSegment {
            fields: &fields1,
            doc_in: Some(&doc_in1),
            pos_in: None,
            pay_in: None,
            live_docs: None,
            doc_base: max_doc0,
            cache: None,
            reader: None,
            index_sort_prefix: false,
            termless: None,
            similarity: None,
            points: None,
            max_doc: None,
        },
    ];
    let norms = [None, None];

    let seq = search_term_query_multi_segment(&segments, &query, &norms, 10).unwrap();
    let con = search_term_query_multi_segment_concurrent(&segments, &query, &norms, 10).unwrap();
    assert!(!seq.is_empty());
    assert_identical(&seq, &con);
}

/// Same end-to-end check for [`search_boolean_query_multi_segment_concurrent`].
#[test]
fn search_boolean_query_multi_segment_concurrent_matches_sequential() {
    let (fields0, doc0, id0, suffix0, max_doc0) = open_real_segment();
    let (fields1, doc1, id1, suffix1, _) = open_real_segment();
    let doc_in0 = lucene_codecs::postings::DocInput::open(&doc0, &id0, &suffix0).unwrap();
    let doc_in1 = lucene_codecs::postings::DocInput::open(&doc1, &id1, &suffix1).unwrap();

    let query = BooleanQuery::new().with_should([TQ::new("body", "cat"), TQ::new("body", "bird")]);
    let segments = [
        OpenSegment {
            fields: &fields0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
            live_docs: None,
            doc_base: 0,
            max_doc: None,
            cache: None,
            reader: None,
            index_sort_prefix: false,
            termless: None,
            similarity: None,
            points: None,
        },
        OpenSegment {
            fields: &fields1,
            doc_in: Some(&doc_in1),
            pos_in: None,
            pay_in: None,
            live_docs: None,
            doc_base: max_doc0,
            cache: None,
            reader: None,
            index_sort_prefix: false,
            termless: None,
            similarity: None,
            points: None,
            max_doc: None,
        },
    ];
    let norms = [None, None];

    let seq = search_boolean_query_multi_segment(&segments, &query, &norms, 10).unwrap();
    let con = search_boolean_query_multi_segment_concurrent(&segments, &query, &norms, 10).unwrap();
    assert!(!seq.is_empty());
    assert_identical(&seq, &con);
}

// --- Sort-by-field multi-segment fan-out/merge ---

fn numeric_field_infos(field_number: i32) -> lucene_codecs::field_infos::FieldInfos {
    use lucene_codecs::field_infos::{
        DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
        VectorSimilarityFunction,
    };
    lucene_codecs::field_infos::FieldInfos {
        fields: vec![FieldInfo {
            name: "score".to_string(),
            number: field_number,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::Numeric,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        }],
    }
}

/// Writes one synthetic segment's dense NUMERIC doc-values field and
/// returns `(dvd_bytes, entry)` -- the same shape `DocValueSegment`
/// needs, built via the real (unmodified) `doc_values::write_single_dense_numeric_field`/
/// `parse_meta` round trip rather than hand-crafted bytes.
fn write_numeric_segment(values: &[i64]) -> (Vec<u8>, NumericEntry) {
    let seg_id = [7u8; lucene_store::codec_util::ID_LENGTH];
    let (dvm, dvd, _dvs) = lucene_codecs::doc_values::write_single_dense_numeric_field(
        0,
        values,
        values.len() as i32,
        &seg_id,
        "",
    )
    .unwrap();
    let fis = numeric_field_infos(0);
    let (_, meta) = lucene_codecs::doc_values::parse_meta(&dvm, &seg_id, "", &fis).unwrap();
    (dvd, meta.numeric_entry(0).unwrap().clone())
}

/// The points-pruned sort returns exactly what scanning every segment
/// returns, whichever segments carry points, in both directions, for
/// several page sizes and ranges -- values drawn from a small domain so
/// ties across segments are everywhere.
#[test]
fn sort_by_field_with_points_matches_the_full_scan() {
    let seg_id = [7u8; lucene_store::codec_util::ID_LENGTH];
    let mut x = 0x0DDB_1A5E_5BAD_5EEDu64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut built = Vec::new();
    for _ in 0..6 {
        let n = 1 + (next() % 400) as usize;
        let values: Vec<i64> = (0..n).map(|_| (next() % 500) as i64 - 100).collect();
        let (dvd, entry) = write_numeric_segment(&values);
        let points: Vec<(i32, Vec<u8>)> = values
            .iter()
            .enumerate()
            .map(|(d, &v)| (d as i32, crate::points_query::pack_i64(v).to_vec()))
            .collect();
        let (kdm, kdi, kdd) = lucene_codecs::points::write(
            &[lucene_codecs::points::WritePointsField {
                field_number: 0,
                num_dims: 1,
                num_index_dims: 1,
                bytes_per_dim: 8,
                points,
            }],
            64,
            &seg_id,
            "",
        )
        .unwrap();
        built.push((n, dvd, entry, kdm, kdi, kdd));
    }
    let readers: Vec<_> = built
        .iter()
        .map(|(_, _, _, kdm, kdi, kdd)| {
            lucene_codecs::points::open(kdm, kdi, kdd, &seg_id, "").unwrap()
        })
        .collect();
    let mut base = 0i32;
    let segments: Vec<DocValueSegment<'_>> = built
        .iter()
        .map(|(n, dvd, entry, ..)| {
            let s = DocValueSegment {
                range_data: dvd,
                range_entry: entry,
                sort_data: dvd,
                sort_entry: entry,
                live_docs: None,
                max_doc: *n as i32,
                doc_base: base,
            };
            base += *n as i32;
            s
        })
        .collect();
    for mask in [0b111111u32, 0b101010, 0] {
        let points: Vec<Option<SortPoints<'_>>> = readers
            .iter()
            .enumerate()
            .map(|(i, r)| {
                (mask >> i & 1 == 1).then_some(SortPoints {
                    reader: r,
                    field_number: 0,
                })
            })
            .collect();
        for direction in [SortDirection::Ascending, SortDirection::Descending] {
            for top_n in [1usize, 3, 10, 50, 5000] {
                for (min, max) in [(i64::MIN, i64::MAX), (-50, 120), (390, 399), (1000, 2000)] {
                    for missing in [MissingValue::Exclude, MissingValue::Default(7)] {
                        let want = search_numeric_range_sorted_by_field_multi_segment(
                            &segments, min, max, direction, missing, top_n,
                        )
                        .unwrap();
                        let got = search_numeric_range_sorted_by_field_multi_segment_with_points(
                            &segments, &points, min, max, direction, missing, top_n,
                        )
                        .unwrap();
                        assert_eq!(
                            got, want,
                            "mask {mask:b} {direction:?} n={top_n} [{min},{max}]"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn sort_by_field_multi_segment_translates_doc_ids_across_segments() {
    // Segment 0: 3 docs, values [30, 10, 20]. Segment 1 (doc_base=3): 2
    // docs, values [5, 25].
    let (dvd0, entry0) = write_numeric_segment(&[30, 10, 20]);
    let (dvd1, entry1) = write_numeric_segment(&[5, 25]);

    let segments = [
        DocValueSegment {
            range_data: &dvd0,
            range_entry: &entry0,
            sort_data: &dvd0,
            sort_entry: &entry0,
            live_docs: None,
            max_doc: 3,
            doc_base: 0,
        },
        DocValueSegment {
            range_data: &dvd1,
            range_entry: &entry1,
            sort_data: &dvd1,
            sort_entry: &entry1,
            live_docs: None,
            max_doc: 2,
            doc_base: 3,
        },
    ];

    let ascending = search_numeric_range_sorted_by_field_multi_segment(
        &segments,
        i64::MIN,
        i64::MAX,
        SortDirection::Ascending,
        MissingValue::Exclude,
        10,
    )
    .unwrap();
    // Global values by doc id: 0->30, 1->10, 2->20, 3(=0+3)->5, 4->25.
    let got: Vec<(i32, i64)> = ascending.iter().map(|d| (d.doc_id, d.value)).collect();
    assert_eq!(got, vec![(3, 5), (1, 10), (2, 20), (4, 25), (0, 30)]);

    let descending = search_numeric_range_sorted_by_field_multi_segment(
        &segments,
        i64::MIN,
        i64::MAX,
        SortDirection::Descending,
        MissingValue::Exclude,
        10,
    )
    .unwrap();
    let got_desc: Vec<(i32, i64)> = descending.iter().map(|d| (d.doc_id, d.value)).collect();
    assert_eq!(got_desc, vec![(0, 30), (4, 25), (2, 20), (1, 10), (3, 5)]);
}

#[test]
fn sort_by_field_multi_segment_tie_break_is_global_ascending_doc_id() {
    // Segment 0 (doc_base=0): 2 docs, values [50, 50] (doc 0 and doc 1
    // tie). Segment 1 (doc_base=2): 1 doc, value 50 (global doc id 2,
    // also ties). All three docs tie on value -- the merged order must
    // be ascending GLOBAL doc id (0, 1, 2), proving the tie-break
    // survives translation across segment boundaries, not just within
    // one segment.
    let (dvd0, entry0) = write_numeric_segment(&[50, 50]);
    let (dvd1, entry1) = write_numeric_segment(&[50]);

    let segments = [
        DocValueSegment {
            range_data: &dvd0,
            range_entry: &entry0,
            sort_data: &dvd0,
            sort_entry: &entry0,
            live_docs: None,
            max_doc: 2,
            doc_base: 0,
        },
        DocValueSegment {
            range_data: &dvd1,
            range_entry: &entry1,
            sort_data: &dvd1,
            sort_entry: &entry1,
            live_docs: None,
            max_doc: 1,
            doc_base: 2,
        },
    ];

    let result = search_numeric_range_sorted_by_field_multi_segment(
        &segments,
        i64::MIN,
        i64::MAX,
        SortDirection::Ascending,
        MissingValue::Exclude,
        10,
    )
    .unwrap();
    let doc_ids: Vec<i32> = result.iter().map(|d| d.doc_id).collect();
    assert_eq!(doc_ids, vec![0, 1, 2]);
}

#[test]
fn sort_by_field_multi_segment_top_n_truncates_globally() {
    let (dvd0, entry0) = write_numeric_segment(&[30, 10, 20]);
    let (dvd1, entry1) = write_numeric_segment(&[5, 25]);
    let segments = [
        DocValueSegment {
            range_data: &dvd0,
            range_entry: &entry0,
            sort_data: &dvd0,
            sort_entry: &entry0,
            live_docs: None,
            max_doc: 3,
            doc_base: 0,
        },
        DocValueSegment {
            range_data: &dvd1,
            range_entry: &entry1,
            sort_data: &dvd1,
            sort_entry: &entry1,
            live_docs: None,
            max_doc: 2,
            doc_base: 3,
        },
    ];
    let top2 = search_numeric_range_sorted_by_field_multi_segment(
        &segments,
        i64::MIN,
        i64::MAX,
        SortDirection::Ascending,
        MissingValue::Exclude,
        2,
    )
    .unwrap();
    let doc_ids: Vec<i32> = top2.iter().map(|d| d.doc_id).collect();
    assert_eq!(doc_ids, vec![3, 1]);
}

#[test]
fn sort_by_field_multi_segment_no_segments_returns_empty() {
    let segments: [DocValueSegment<'_>; 0] = [];
    let result = search_numeric_range_sorted_by_field_multi_segment(
        &segments,
        i64::MIN,
        i64::MAX,
        SortDirection::Ascending,
        MissingValue::Exclude,
        10,
    )
    .unwrap();
    assert!(result.is_empty());
}

#[test]
fn sort_by_field_multi_segment_top_n_zero_returns_nothing() {
    let (dvd0, entry0) = write_numeric_segment(&[30, 10, 20]);
    let segments = [DocValueSegment {
        range_data: &dvd0,
        range_entry: &entry0,
        sort_data: &dvd0,
        sort_entry: &entry0,
        live_docs: None,
        max_doc: 3,
        doc_base: 0,
    }];
    let result = search_numeric_range_sorted_by_field_multi_segment(
        &segments,
        i64::MIN,
        i64::MAX,
        SortDirection::Ascending,
        MissingValue::Exclude,
        0,
    )
    .unwrap();
    assert!(result.is_empty());
}

#[test]
fn sort_by_field_multi_segment_segment_with_zero_matches_does_not_break_the_merge() {
    // Segment 0's values are all outside [100, 200] -- zero matches --
    // segment 1's are inside; the merge must still return segment 1's
    // hits correctly, not error or drop them.
    let (dvd0, entry0) = write_numeric_segment(&[1, 2, 3]);
    let (dvd1, entry1) = write_numeric_segment(&[150, 175]);
    let segments = [
        DocValueSegment {
            range_data: &dvd0,
            range_entry: &entry0,
            sort_data: &dvd0,
            sort_entry: &entry0,
            live_docs: None,
            max_doc: 3,
            doc_base: 0,
        },
        DocValueSegment {
            range_data: &dvd1,
            range_entry: &entry1,
            sort_data: &dvd1,
            sort_entry: &entry1,
            live_docs: None,
            max_doc: 2,
            doc_base: 3,
        },
    ];
    let result = search_numeric_range_sorted_by_field_multi_segment(
        &segments,
        100,
        200,
        SortDirection::Ascending,
        MissingValue::Exclude,
        10,
    )
    .unwrap();
    let doc_ids: Vec<i32> = result.iter().map(|d| d.doc_id).collect();
    assert_eq!(doc_ids, vec![3, 4]);
}

// --- Deadline / timeout (task: caller-supplied wall-clock deadline) ---

#[test]
fn expired_deadline_returns_partial_results_without_hanging() {
    let doc_bases = [0, 10, 20];
    let mut seg0 = fake_segment_search(vec![(0, 1.0)]);
    let mut seg1 = fake_segment_search(vec![(0, 2.0)]);
    let mut seg2 = fake_segment_search(vec![(0, 3.0)]);
    // A deadline already in the past: the very first per-segment check
    // must trip immediately, contributing nothing rather than hanging or
    // erroring.
    let past = Instant::now() - std::time::Duration::from_secs(3600);
    let (result, timed_out) =
        merge_multi_segment_scored_with_deadline(&doc_bases, 10, Some(past), |i, local| {
            match i {
                0 => seg0(local),
                1 => seg1(local),
                2 => seg2(local),
                _ => unreachable!(),
            }
            Ok(())
        })
        .unwrap();
    assert!(timed_out);
    assert!(result.is_empty());
}

#[test]
fn generous_deadline_matches_no_timeout_path() {
    let doc_bases = [0, 3, 5];
    let seg0 = vec![(0, 1.0), (1, 5.0), (2, 3.0)];
    let seg1 = vec![(0, 4.0), (1, 2.0)];
    let seg2 = vec![(0, 6.0), (1, 0.5), (2, 4.5)];
    let hits = [seg0, seg1, seg2];
    let no_timeout = merge_multi_segment_scored(&doc_bases, 10, |i, local| {
        for &(doc_id, score) in &hits[i] {
            local.collect(doc_id, score);
        }
        Ok(())
    })
    .unwrap();
    let generous = Instant::now() + std::time::Duration::from_secs(3600);
    let (with_deadline, timed_out) =
        merge_multi_segment_scored_with_deadline(&doc_bases, 10, Some(generous), |i, local| {
            for &(doc_id, score) in &hits[i] {
                local.collect(doc_id, score);
            }
            Ok(())
        })
        .unwrap();
    assert!(!timed_out);
    assert_identical(&no_timeout, &with_deadline);
}

#[test]
fn none_deadline_behaves_identically_to_no_timeout_path() {
    let doc_bases = [0, 10];
    let hits = [vec![(0, 1.0), (1, 2.0)], vec![(0, 4.0), (1, 5.0)]];
    let no_timeout = merge_multi_segment_scored(&doc_bases, 2, |i, local| {
        for &(doc_id, score) in &hits[i] {
            local.collect(doc_id, score);
        }
        Ok(())
    })
    .unwrap();
    let (with_none, timed_out) =
        merge_multi_segment_scored_with_deadline(&doc_bases, 2, None, |i, local| {
            for &(doc_id, score) in &hits[i] {
                local.collect(doc_id, score);
            }
            Ok(())
        })
        .unwrap();
    assert!(!timed_out);
    assert_identical(&no_timeout, &with_none);
}

#[test]
fn expired_deadline_returns_partial_results_concurrent() {
    let (doc_bases, hits) = synthetic_doc_bases_and_hits(16);
    let past = Instant::now() - std::time::Duration::from_secs(3600);
    let (result, timed_out) = merge_multi_segment_scored_concurrent_with_deadline(
        &doc_bases,
        10,
        Some(past),
        |i, local| {
            for &(doc_id, score) in &hits[i] {
                local.collect(doc_id, score);
            }
            Ok(())
        },
    )
    .unwrap();
    assert!(timed_out);
    assert!(result.is_empty());
}

#[test]
fn generous_deadline_matches_no_timeout_path_concurrent() {
    let (doc_bases, hits) = synthetic_doc_bases_and_hits(16);
    let no_timeout = run_concurrent(&doc_bases, &hits, 10);
    let generous = Instant::now() + std::time::Duration::from_secs(3600);
    let (with_deadline, timed_out) = merge_multi_segment_scored_concurrent_with_deadline(
        &doc_bases,
        10,
        Some(generous),
        |i, local| {
            for &(doc_id, score) in &hits[i] {
                local.collect(doc_id, score);
            }
            Ok(())
        },
    )
    .unwrap();
    assert!(!timed_out);
    assert_identical(&no_timeout, &with_deadline);
}

#[test]
fn none_deadline_behaves_identically_to_no_timeout_path_concurrent() {
    let (doc_bases, hits) = synthetic_doc_bases_and_hits(16);
    let no_timeout = run_concurrent(&doc_bases, &hits, 10);
    let (with_none, timed_out) =
        merge_multi_segment_scored_concurrent_with_deadline(&doc_bases, 10, None, |i, local| {
            for &(doc_id, score) in &hits[i] {
                local.collect(doc_id, score);
            }
            Ok(())
        })
        .unwrap();
    assert!(!timed_out);
    assert_identical(&no_timeout, &with_none);
}

#[test]
fn sort_by_field_multi_segment_propagates_decode_errors() {
    // max_doc claims more docs than the dense entry actually holds --
    // numeric_value's Error::DocOutOfRange must surface through the
    // multi-segment wrapper, not be swallowed or panic.
    let (dvd, entry) = write_numeric_segment(&[1, 2, 3]);
    let segments = [DocValueSegment {
        range_data: &dvd,
        range_entry: &entry,
        sort_data: &dvd,
        sort_entry: &entry,
        live_docs: None,
        max_doc: 10, // beyond the entry's real 3 values
        doc_base: 0,
    }];
    let err = search_numeric_range_sorted_by_field_multi_segment(
        &segments,
        i64::MIN,
        i64::MAX,
        SortDirection::Ascending,
        MissingValue::Exclude,
        10,
    )
    .unwrap_err();
    assert!(matches!(err, crate::Error::DocValues(_)));
}
/// The concurrent fan-out must score from **reader-wide** term statistics,
/// exactly as the sequential one does.
///
/// It did not: `search_term_query_multi_segment_concurrent` called the
/// no-stats scorer, so every segment computed idf from its own
/// `docFreq`/`docCount` while the sequential path used the summed pair.
/// That is the very bug this module's doc comment records (a 1.6x idf spread
/// across the 15-segment corpus, every benchmark query disagreeing with
/// Java), reintroduced on the concurrent path alone.
///
/// The existing "concurrent matches sequential" test cannot catch it: it
/// duplicates one fixture segment whose `docFreq` is exactly half its
/// `docCount`, and `idf(d, 2d) == idf(2d, 4d) == ln 2`, so summing the
/// counters is numerically invisible there. This test therefore builds two
/// deliberately *lopsided* segments -- "fox" in 1 of 1 documents in one and
/// 1 of 4 in the other -- where the three candidate idfs (0.288 per segment
/// A, 1.204 per segment B, 0.876 reader-wide) are far apart.
mod global_stats_across_concurrent_leaves {
    use super::*;
    use lucene_codecs::field_infos::{
        DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
        VectorSimilarityFunction,
    };
    use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::segment_info::LuceneVersion;
    use lucene_store::FsDirectory;

    fn body_field(number: i32) -> FieldInfo {
        FieldInfo {
            name: "body".to_string(),
            number,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::DocsAndFreqs,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        }
    }

    fn doc(body: &str) -> Document {
        Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String(body.to_string()),
            }],
        }
    }

    use lucene_util::test_support::TempDir;

    /// A scratch directory that removes itself when the test ends -- unless
    /// the test is panicking, in which case its bytes stay for inspection.
    fn tempdir() -> TempDir {
        TempDir::new("multi-segment-stats")
    }

    #[test]
    fn concurrent_term_query_uses_reader_wide_idf_like_the_sequential_path() {
        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);
        let mut writer = IndexWriter::open(
            &dir,
            vec![body_field(0)],
            "Lucene104",
            LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
        )
        .unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        // Segment 0: one document, and it contains "fox".
        writer.add_document(doc("fox")).unwrap();
        writer.commit().unwrap();
        // Segment 1: four documents, only one of which contains "fox".
        for body in ["fox", "cat", "dog", "bird"] {
            writer.add_document(doc(body)).unwrap();
        }
        writer.commit().unwrap();

        let reader = crate::directory_reader::DirectoryReader::open(&dir).unwrap();
        assert_eq!(
            reader.segment_readers().len(),
            2,
            "the two commits must stay two segments for this test to mean anything"
        );
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let query = TermQuery::new("body", "fox");
        let norms = [None, None];

        let seq = search_term_query_multi_segment(&segments, &query, &norms, 10).unwrap();
        let con =
            search_term_query_multi_segment_concurrent(&segments, &query, &norms, 10).unwrap();
        assert_eq!(seq.len(), 2, "both segments contribute a \"fox\" document");
        assert_identical(&seq, &con);

        // And the score really is the reader-wide idf, not either segment's
        // own -- otherwise "sequential == concurrent" could be satisfied by
        // both being wrong the same way.
        let reader_wide = crate::similarity::idf(2, 5);
        let segment_a_only = crate::similarity::idf(1, 1);
        let segment_b_only = crate::similarity::idf(1, 4);
        let tf = crate::similarity::tf_norm(
            1.0,
            crate::similarity::UNNORMED_FIELD_LENGTH,
            crate::similarity::UNNORMED_FIELD_LENGTH,
            crate::similarity::DEFAULT_K1,
            crate::similarity::DEFAULT_B,
        );
        for hit in &con {
            assert!(
                (hit.score - reader_wide * tf).abs() < 1e-5,
                "doc {} scored {}, reader-wide idf predicts {} (segment-local \
                     alternatives would be {} and {})",
                hit.doc_id,
                hit.score,
                reader_wide * tf,
                segment_a_only * tf,
                segment_b_only * tf
            );
        }

        std::fs::remove_dir_all(&dir_path).ok();
    }

    /// Tripwire for `search_term_query_scored_maxscore_with_stats`'
    /// fallback paths.
    ///
    /// That function declines the pruned MAXSCORE loop on three conditions
    /// -- no `.doc` input, `docFreq <= 1`, and an index option
    /// `LazyDocsCursor` rejects -- and each fallback must score with the
    /// `global` it was handed, not the leaf's own `docFreq`/`docCount`, bit
    /// for bit, exactly as the pruned path does.
    ///
    /// The third fallback is unreachable for a scored term query --
    /// `LazyDocsCursor` rejects only `IndexOptions::None`, which carries no
    /// postings to score in the first place -- so it is asserted by
    /// inspection, not here.
    #[test]
    fn every_maxscore_fallback_honours_the_global_stats_it_is_handed() {
        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);
        let mut writer = IndexWriter::open(
            &dir,
            vec![body_field(0)],
            "Lucene104",
            LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
        )
        .unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        // "fox" is a singleton (fallback 2); "cat" is in three documents,
        // so it takes the pruned loop and acts as the control.
        for body in ["fox", "cat", "cat dog", "cat bird"] {
            writer.add_document(doc(body)).unwrap();
        }
        writer.commit().unwrap();

        let reader = crate::directory_reader::DirectoryReader::open(&dir).unwrap();
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let seg = &segments[0];

        // A `global` that is nothing like this segment's own counters, so
        // "used the segment's own" and "used the reader-wide" cannot
        // coincide numerically.
        let global = crate::CollectionStats {
            doc_freq: 1,
            doc_count: 1_000_000,
            ..Default::default()
        };
        let expected = |freq: f32| {
            crate::similarity::score(
                global.doc_freq,
                global.doc_count,
                freq,
                crate::similarity::UNNORMED_FIELD_LENGTH,
                crate::similarity::UNNORMED_FIELD_LENGTH,
            )
        };

        let fox = TermQuery::new("body", "fox");
        let cat = TermQuery::new("body", "cat");
        let run = |query: &TermQuery, doc_in| {
            let mut collector = TopDocsCollector::new(10);
            crate::search_term_query_scored_maxscore_with_stats(
                seg.fields,
                doc_in,
                seg.live_docs,
                query,
                None,
                Some(global),
                &mut collector,
            )
            .unwrap();
            collector.top_docs().to_vec()
        };

        // Fallback 1: no `.doc` input at all. (`fox` again, because the
        // eager path can only resolve a `docFreq > 1` term through an
        // opened `.doc` -- so this fallback is only ever reachable for a
        // singleton term, which is what makes it the *first* check.)
        let hits = run(&fox, None);
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].score.to_bits(),
            expected(1.0).to_bits(),
            "the no-`.doc`-input fallback must score with the \
                 reader-wide idf it was handed, not this segment's own"
        );

        // Fallback 2: `docFreq <= 1`.
        let hits = run(&fox, seg.doc_in);
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].score.to_bits(),
            expected(1.0).to_bits(),
            "the `docFreq <= 1` fallback must score with the reader-wide \
                 idf it was handed, not this segment's own idf(1, 4)"
        );

        // Control: the pruned path, which always honoured `global`, must
        // agree with the fallbacks to the bit.
        let hits = run(&cat, seg.doc_in);
        assert_eq!(hits.len(), 3);
        for hit in &hits {
            assert_eq!(hit.score.to_bits(), expected(1.0).to_bits());
        }

        std::fs::remove_dir_all(&dir_path).ok();
    }
}
