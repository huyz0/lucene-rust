//! The native API called directly: the baseline every other path is
//! compared to, and the code the fast-path router calls.

use std::collections::HashMap;
use std::sync::Arc;

use lucene_aggs::bucket_aggs::{aggregate_tree, AggNode, AggResult, Globals, TermsOrder};
use lucene_aggs::metric_aggs::{Source, ValueKind, NEED_SUM};
use lucene_codecs::doc_values::NumericReader;
use lucene_search::global_ords::GlobalOrds;
use lucene_search::knn_search::{knn_top_hits, KnnTarget};
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermQuery};

use crate::index::SpikeIndex;

pub const KNN_CANDIDATES: usize = 50;

pub fn term_query(field: &str, term: &str) -> BooleanQuery {
    BooleanQuery {
        must: vec![Clause::Term(TermQuery::new(
            field,
            term.as_bytes().to_vec(),
        ))],
        ..BooleanQuery::default()
    }
}

pub fn match_query(words: &[&str]) -> BooleanQuery {
    BooleanQuery {
        should: words
            .iter()
            .map(|w| Clause::Term(TermQuery::new("body", w.as_bytes().to_vec())))
            .collect(),
        minimum_should_match: 1,
        ..BooleanQuery::default()
    }
}

/// Value of the `price` doc value of a segment-local doc.
pub fn price(ix: &SpikeIndex, seg: usize, local: i32) -> Option<i64> {
    let leaf = &ix.leaves()[seg];
    let (meta, data) = leaf.doc_values_for_field(ix.fields[seg].price)?;
    let entry = meta.numeric_entry(ix.fields[seg].price)?;
    NumericReader::new(data, entry).value(local).unwrap()
}

/// `SELECT id, price FROM t WHERE id = ?`
pub fn point_read(ix: &SpikeIndex, id: &str) -> Option<(String, Option<i64>)> {
    let top = ix.searcher.search(&term_query("id", id), 1).unwrap();
    let hit = top.score_docs.first()?;
    let (seg, local) = ix.locate(hit.doc);
    Some((ix.stored_id(seg, local), price(ix, seg, local)))
}

/// `SELECT id, _score FROM t WHERE match(body, ?) ORDER BY _score DESC LIMIT k`
pub fn match_topk(ix: &SpikeIndex, words: &[&str], k: usize) -> Vec<(String, f32)> {
    let top = ix.searcher.search(&match_query(words), k).unwrap();
    top.score_docs
        .iter()
        .map(|h| {
            let (seg, local) = ix.locate(h.doc);
            (ix.stored_id(seg, local), h.score)
        })
        .collect()
}

/// `SELECT id, _score FROM t WHERE knn(vec, ?) ORDER BY _score DESC LIMIT k`
pub fn knn_topk(ix: &SpikeIndex, vector: &[f32], k: usize) -> Vec<(String, f32)> {
    let hits = knn_top_hits(
        ix.segments,
        "vec",
        &KnnTarget::Float(vector.to_vec()),
        k,
        KNN_CANDIDATES.max(k),
        None,
    )
    .unwrap();
    hits.iter()
        .map(|h| {
            let (seg, local) = ix.locate(h.doc_id);
            (ix.stored_id(seg, local), h.score)
        })
        .collect()
}

/// `SELECT cat, count(*), sum(price) FROM t GROUP BY cat` through the
/// aggregation kernels (a keyword `terms` with a `sum` sub-aggregation).
pub fn group_by(ix: &SpikeIndex) -> Vec<(String, u64, f64)> {
    let node = AggNode::Terms {
        field: "cat".into(),
        missing: None,
        accepted: None,
        zero_buckets: false,
        shard_size: 1000,
        order: TermsOrder::KeyAsc,
        subs: vec![AggNode::Metric {
            field: "price".into(),
            missing: None,
            kind: ValueKind::Long,
            source: Source::DocValues,
            needs: NEED_SUM,
            script: None,
        }],
    };
    let query = BooleanQuery {
        must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(ix.max_doc()))],
        ..BooleanQuery::default()
    };
    let ords: HashMap<String, Arc<GlobalOrds>> =
        HashMap::from([("cat".to_string(), ix.cat_global.clone())]);
    let slices = vec![(0..ix.segments.len()).collect::<Vec<_>>()];
    let mut out = aggregate_tree(
        ix.segments,
        ix.leaves(),
        &query,
        &[node],
        &Globals::new(&ords),
        &slices,
        None,
    )
    .unwrap();
    let AggResult::Terms { buckets, subs } = out.remove(0).remove(0) else {
        panic!("terms result")
    };
    let AggResult::Metric(states) = &subs[0] else {
        panic!("metric result")
    };
    buckets[0]
        .1
        .iter()
        .zip(states)
        .map(|((k, n), m)| (String::from_utf8(k.clone()).unwrap(), *n, m.sum))
        .collect()
}
