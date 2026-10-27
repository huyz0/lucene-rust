#![allow(clippy::arithmetic_side_effects)]
//! `TestShardSearching.testSimple` over [`ShardSearching`]: while
//! `ChangeIndices` adds, updates, deletes, commits and reopens on random
//! nodes, the main thread runs fresh and follow-on (`searchAfter`) queries
//! on a random node's [`ShardIndexSearcher`], and each must return what one
//! searcher over every node's reader at the same versions (Java's
//! `MultiReader` mock) returns: the same documents, rebased, with the same
//! scores and sort values (`TestUtil.assertConsistent`).
//!
//! Differs: a few fixed seeds, each running about a second; the sorts are
//! by score, by `page_views` (a `long`; `LineFileDocs` no longer adds
//! Java's `docid_intDV`) and by `title` (Java's `titleDV`); nothing is
//! printed.

use std::sync::Arc;
use std::time::{Duration, Instant};

use lucene_search::collector::{ScoreDoc, TotalHits, TotalHitsRelation};
use lucene_search::query::{BooleanQuery, Clause, PrefixQuery, TermQuery};
use lucene_search::reader::LeafReader as _;
use lucene_search::top_docs::ShardScoreDoc;
use lucene_search::top_field::{SortField, SortType};
use lucene_store::{ByteBuffersDirectory, Directory};
use lucene_test_framework::analysis::mock_analyzer::MockAnalyzer;
use lucene_test_framework::search::shard_searching::{
    with_multi_searcher, ShardError, ShardIndexSearcher, ShardSearching,
};
use lucene_test_framework::util::line_file_docs::LineFileDocs;
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;
use lucene_test_framework::util::test_util;
use lucene_util::java_random::JavaRandom;

const SEEDS: [i64; 3] = [0x5A4D, -0x0123_4567_89AB, 7];

/// `PreviousSearchState`.
#[derive(Clone)]
struct PreviousSearchState {
    query: BooleanQuery,
    sort: Option<Vec<SortField>>,
    search_after_local: Option<ShardScoreDoc>,
    search_after_shard: Option<ShardScoreDoc>,
    versions: Vec<i64>,
    num_hits_paged: u64,
}

/// What was exercised, across a run.
#[derive(Debug, Default)]
struct Counts {
    searches: u64,
    sorted: u64,
    follow_ons: u64,
    expired: u64,
    hits: u64,
}

/// One hit as `assertConsistent` compares it: the (rebased) document, the
/// score bits (`None` for a sorted hit) and the sort values.
type Hit = (i32, Option<u32>, Vec<i64>, Vec<Option<Vec<u8>>>);

/// `TestUtil.assertConsistent(expected, actual)`.
fn assert_consistent(
    what: &str,
    (expected_total, expected): (TotalHits, &[Hit]),
    (actual_total, actual): (TotalHits, &[Hit]),
) {
    assert_eq!(
        expected_total.value == 0,
        actual_total.value == 0,
        "{what}: wrong total hits"
    );
    let eq = TotalHitsRelation::EqualTo;
    match (expected_total.relation == eq, actual_total.relation == eq) {
        (true, true) => assert_eq!(expected_total.value, actual_total.value, "{what}"),
        (true, false) => assert!(expected_total.value >= actual_total.value, "{what}"),
        (false, true) => assert!(expected_total.value <= actual_total.value, "{what}"),
        (false, false) => {}
    }
    assert_eq!(expected, actual, "{what}");
}

fn query_text(q: &BooleanQuery) -> String {
    format!("{q:?}")
}

/// `assertSame`: the mock searcher's and the shard searcher's hits for
/// `q`; a continuation when there are more to page through.
fn assert_same(
    random: &mut JavaRandom,
    counts: &mut Counts,
    readers: &[Arc<lucene_search::directory_reader::DirectoryReader>],
    shard: &ShardIndexSearcher<'_, '_>,
    q: &BooleanQuery,
    sort: Option<&[SortField]>,
    state: Option<&PreviousSearchState>,
) -> Result<Option<PreviousSearchState>, ShardError> {
    let mut num_hits = test_util::next_int(random, 1, 100) as usize;
    if let Some(s) = state.filter(|s| s.search_after_local.is_none()) {
        // In addition to what we last searched.
        num_hits += s.num_hits_paged as usize;
    }
    let after_local = state.and_then(|s| s.search_after_local);
    let after_shard = state.and_then(|s| s.search_after_shard);
    let what = format!("{} sort={sort:?} n={num_hits}", query_text(q));

    // Single (mock local) searcher.
    let (total, hits, bases) = with_multi_searcher(readers, |s, leaves, bases| {
        let fail = |e: lucene_search::Error| ShardError::Failed(e.to_string());
        let (total, hits): (TotalHits, Vec<Hit>) = match sort {
            None => {
                let td = match after_local {
                    Some(a) => s.search_after(
                        ScoreDoc {
                            doc_id: a.doc,
                            score: a.score,
                        },
                        q,
                        num_hits,
                    ),
                    None => s.search(q, num_hits),
                }
                .map_err(fail)?;
                let hits = td
                    .score_docs
                    .iter()
                    .map(|h| (h.doc, Some(h.score.to_bits()), Vec::new(), Vec::new()))
                    .collect();
                (td.total_hits, hits)
            }
            Some(sort) => {
                let td = s
                    .search_sorted(leaves, q, num_hits, sort, None)
                    .map_err(fail)?;
                let hits = td
                    .hits
                    .into_iter()
                    .map(|h| (h.doc, None, h.values, h.terms))
                    .collect();
                (td.total, hits)
            }
        };
        Ok((total, hits, bases.to_vec()))
    })?;

    // Shard searcher.
    let (shard_total, shard_hits, shard_bottom) = match sort {
        None => {
            let td = match after_shard {
                Some(a) => shard.search_after(a, q, num_hits)?,
                None => shard.search(q, num_hits)?,
            };
            let rebased: Vec<Hit> = td
                .score_docs
                .iter()
                .map(|h| {
                    (
                        h.doc + bases[h.shard_index as usize],
                        Some(h.score.to_bits()),
                        Vec::new(),
                        Vec::new(),
                    )
                })
                .collect();
            (td.total_hits, rebased, td.score_docs.last().copied())
        }
        Some(sort) => {
            let td = shard.search_sorted(q, num_hits, sort)?;
            let rebased: Vec<Hit> = td
                .hits
                .into_iter()
                .map(|h| {
                    (
                        h.fields.doc + bases[h.shard_index as usize],
                        None,
                        h.fields.values,
                        h.fields.terms,
                    )
                })
                .collect();
            (td.total_hits, rebased, None)
        }
    };

    let mut num_hits_paged = hits.len() as u64;
    if let Some(s) = state.filter(|s| s.search_after_local.is_some()) {
        num_hits_paged += s.num_hits_paged;
    }
    counts.searches += 1;
    counts.sorted += u64::from(sort.is_some());
    counts.hits += hits.len() as u64;

    let more_hits = num_hits_paged < total.value;
    let (bottom_hit, bottom_hit_shards) = if more_hits && sort.is_none() {
        let last = hits.last().expect("a page of hits");
        (
            Some(ShardScoreDoc::new(
                last.0,
                f32::from_bits(last.1.expect("a score")),
            )),
            shard_bottom,
        )
    } else {
        if !more_hits {
            assert_eq!(total.value, num_hits_paged, "{what}");
        }
        (None, None)
    };

    assert_consistent(&what, (total, &hits), (shard_total, &shard_hits));

    Ok(more_hits.then(|| PreviousSearchState {
        query: q.clone(),
        sort: sort.map(<[SortField]>::to_vec),
        search_after_local: bottom_hit,
        search_after_shard: bottom_hit_shards,
        versions: shard.node_versions.clone(),
        num_hits_paged,
    }))
}

/// `testSimple` under one seed.
fn test_simple(seed: i64) -> Counts {
    let mut t = LuceneTestCase::with_config("testSimple", seed, false, 1);
    let r = t.random();
    let num_nodes = test_util::next_int(r, 1, 10) as usize;
    let run_time = Duration::from_millis(800);
    let min_docs_to_make_terms = test_util::next_int(r, 5, 20);
    let max_searcher_age_seconds = f64::from(test_util::next_int(r, 1, 3));
    let max_token_length = test_util::next_int(r, 1, 255);
    let analyzer = Arc::new(
        MockAnalyzer::new(r)
            .max_token_length(max_token_length)
            .build(),
    );
    let change_seed = r.next_long();
    let mut random = JavaRandom::new(r.next_long());

    let dirs: Vec<ByteBuffersDirectory> = (0..num_nodes)
        .map(|_| ByteBuffersDirectory::new())
        .collect();
    let dir_refs: Vec<&dyn Directory> = dirs.iter().map(|d| d as &dyn Directory).collect();
    let cluster =
        ShardSearching::start(&dir_refs, &analyzer, max_searcher_age_seconds).expect("start");
    let end = Instant::now() + run_time;
    let mut counts = Counts::default();

    std::thread::scope(|scope| {
        let changes = scope.spawn(|| {
            let mut r = JavaRandom::new(change_seed);
            let mut docs = LineFileDocs::new(&mut r).expect("LineFileDocs");
            cluster.change_indices(&mut r, &mut docs, end)
        });

        let mut prior_searches: Vec<PreviousSearchState> = Vec::new();
        let mut terms: Option<Vec<Vec<u8>>> = None;
        while Instant::now() < end {
            let do_followon = !prior_searches.is_empty() && random.next_int_bounded(7) == 1;
            // Pick a random node; we will run the query on this node.
            let my_node_id = random.next_int_bounded(num_nodes as i32) as usize;
            let (local, prev) = if do_followon {
                // Pretend the user issued a follow-on query.
                let i = random.next_int_bounded(prior_searches.len() as i32) as usize;
                let prev = prior_searches[i].clone();
                match cluster.acquire_versions(my_node_id, &prev.versions) {
                    Ok(s) => (s, Some((i, prev))),
                    Err(ShardError::Expired { .. }) => {
                        counts.expired += 1;
                        prior_searches.remove(i);
                        continue;
                    }
                    Err(e) => panic!("{e}"),
                }
            } else {
                (cluster.acquire(my_node_id), None)
            };

            // Mock: one reader over every node's searcher at the shard
            // searcher's versions.
            let mut subs = Vec::with_capacity(num_nodes);
            for (node_id, &version) in local.node_versions.iter().enumerate() {
                match cluster.nodes[node_id].searchers.acquire(version).unwrap() {
                    Some(s) => subs.push(s),
                    None => break,
                }
            }
            if subs.len() < num_nodes {
                counts.expired += 1;
                continue;
            }
            let doc_count: i32 = subs.iter().map(|s| s.max_doc()).sum();

            let (query, sort) = match &prev {
                Some((_, p)) => (Some(p.query.clone()), p.sort.clone()),
                None => {
                    if terms.is_none() && doc_count > min_docs_to_make_terms {
                        let mut all = std::collections::BTreeSet::new();
                        for s in &subs {
                            for leaf in s.segment_readers() {
                                let Some(lt) = leaf.terms("body").unwrap() else {
                                    continue;
                                };
                                let mut te = lt.iterator().unwrap();
                                while let Some(term) = te.next().unwrap() {
                                    all.insert(term.to_vec());
                                }
                            }
                        }
                        terms = (!all.is_empty()).then(|| all.into_iter().collect());
                    }
                    match &terms {
                        Some(terms) => {
                            let t = &terms[random.next_int_bounded(terms.len() as i32) as usize];
                            let clause = if random.next_boolean() {
                                Clause::Term(TermQuery::new("body", t.clone()))
                            } else {
                                let text = String::from_utf8_lossy(t).into_owned();
                                let chars: Vec<char> = text.chars().collect();
                                let prefix: String = if chars.len() <= 1 {
                                    text
                                } else {
                                    let n = test_util::next_int(&mut random, 1, 2) as usize;
                                    chars[..n].iter().collect()
                                };
                                Clause::Prefix(PrefixQuery::new("body", prefix.into_bytes()))
                            };
                            let sort = if random.next_boolean() {
                                None
                            } else {
                                match random.next_int_bounded(3) {
                                    0 => Some(vec![SortField::score()]),
                                    // Java's FIELD_DOC sort does not merge.
                                    1 => None,
                                    _ if random.next_boolean() => Some(vec![SortField::numeric(
                                        "page_views",
                                        SortType::Long,
                                        random.next_boolean(),
                                    )]),
                                    _ => Some(vec![SortField::string(
                                        "title",
                                        random.next_boolean(),
                                    )]),
                                }
                            };
                            let q = BooleanQuery {
                                must: vec![clause],
                                ..BooleanQuery::default()
                            };
                            (Some(q), sort)
                        }
                        None => (None, None),
                    }
                }
            };

            let mut search_state = None;
            if let Some(q) = query {
                counts.follow_ons += u64::from(prev.is_some());
                match assert_same(
                    &mut random,
                    &mut counts,
                    &subs,
                    &local,
                    &q,
                    sort.as_deref(),
                    prev.as_ref().map(|(_, p)| p),
                ) {
                    Ok(s) => search_state = s,
                    Err(ShardError::Expired { .. }) => {
                        counts.expired += 1;
                        if let Some((i, _)) = prev {
                            prior_searches.remove(i);
                        }
                    }
                    Err(e) => panic!("{e}"),
                }
            }
            if let Some(s) = search_state {
                if s.search_after_local.is_some() && random.next_int_bounded(5) == 3 {
                    prior_searches.push(s);
                    if prior_searches.len() > 200 {
                        prior_searches.truncate(100);
                    }
                }
            }
        }
        changes
            .join()
            .expect("ChangeIndices")
            .expect("ChangeIndices");
    });
    counts
}

#[test]
fn shard_searching_matches_one_searcher_over_every_node() {
    let all: Vec<Counts> = SEEDS.iter().map(|&s| test_simple(s)).collect();
    let sum = |f: fn(&Counts) -> u64| all.iter().map(f).sum::<u64>();
    // Gate rule #10: what the checks verify must have happened.
    assert!(sum(|c| c.searches) > 50, "{all:?}");
    assert!(sum(|c| c.sorted) > 0, "{all:?}");
    assert!(sum(|c| c.hits) > 0, "{all:?}");
    assert!(sum(|c| c.follow_ons) > 0, "{all:?}");
}

/// A node prunes a superseded searcher once it is older than the maximum
/// age (here `0`): a shard searcher still at that version then fails with
/// `SearcherExpiredException`, both when gathering that node's term
/// statistics and when acquiring its searcher, while a fresh one sees the
/// new document.
#[test]
fn a_pruned_version_has_expired() {
    let mut r = JavaRandom::new(11);
    let analyzer = Arc::new(MockAnalyzer::new(&mut r).build());
    let dirs = [ByteBuffersDirectory::new(), ByteBuffersDirectory::new()];
    let refs: Vec<&dyn Directory> = dirs.iter().map(|d| d as &dyn Directory).collect();
    let cluster = ShardSearching::start(&refs, &analyzer, 0.0).expect("start");
    let line = LineFileDocs::new(&mut r).unwrap().next_doc().unwrap();
    let word = line.body.split_whitespace().next().unwrap().to_lowercase();
    let q = BooleanQuery {
        must: vec![Clause::Term(TermQuery::new(
            "body",
            word.as_bytes().to_vec(),
        ))],
        ..BooleanQuery::default()
    };

    let old = cluster.acquire(0);
    assert_eq!(old.search(&q, 10).unwrap().total_hits.value, 0);
    // nothing changed: no new version
    cluster.reopen(1).unwrap();
    assert_eq!(cluster.acquire(0).node_versions, old.node_versions);

    cluster.nodes[1]
        .writer
        .lock()
        .unwrap()
        .add_fields_document(&line.to_document())
        .unwrap();
    std::thread::sleep(Duration::from_millis(5));
    cluster.reopen(1).unwrap();
    let expired = ShardError::Expired {
        node: 1,
        version: old.node_versions[1],
    };
    assert_eq!(old.search(&q, 10).unwrap_err(), expired);
    assert_eq!(
        cluster
            .acquire_versions(1, &old.node_versions)
            .err()
            .unwrap()
            .to_string(),
        format!("nodeID=1 version={}", old.node_versions[1])
    );

    let fresh = cluster.acquire(0);
    assert!(fresh.node_versions[1] > old.node_versions[1]);
    assert_eq!(fresh.reader().max_doc(), 0);
    let hits = fresh.search(&q, 10).unwrap();
    assert_eq!(hits.total_hits.value, 1);
    assert_eq!(hits.score_docs[0].shard_index, 1);
    // node 1's own statistics: the same score from either node
    let on_one = cluster.acquire(1).search(&q, 10).unwrap();
    assert_eq!(on_one.score_docs, hits.score_docs);
    assert_eq!(ShardError::Failed("x".into()).to_string(), "x");
}
