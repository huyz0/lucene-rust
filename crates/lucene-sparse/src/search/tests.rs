#![cfg(test)]
use super::*;
use crate::postings::Cluster;

fn info(attrs: &[(&str, &str)]) -> FieldInfo {
    FieldInfo::new("emb", 1).with_attributes(
        attrs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
    )
}

fn field() -> SeismicField {
    SeismicField::from_field_info(&info(&[("sparse_vector_field", "true")]))
        .unwrap()
        .unwrap()
}

#[test]
fn field_attributes_and_defaults() {
    assert_eq!(SeismicField::from_field_info(&info(&[])).unwrap(), None);
    let f = field();
    assert_eq!(f.approximate_threshold, DEFAULT_THRESHOLD);
    assert_eq!(f.ceiling_ingest, 3.0);
    assert_eq!(f.ceiling_search, 16.0);
    assert!(!f.native_engine);
    assert!(f.is_seismic_segment(DEFAULT_THRESHOLD));
    assert!(!f.is_seismic_segment(DEFAULT_THRESHOLD - 1));
    let f = SeismicField::from_field_info(&info(&[
        ("sparse_vector_field", "TRUE"),
        ("engine", "Native"),
        ("approximate_threshold", "5"),
        ("quantization_ceiling_ingest", "2.5f"),
        ("quantization_ceiling_search", ""),
    ]))
    .unwrap()
    .unwrap();
    assert!(f.native_engine);
    assert_eq!(f.approximate_threshold, 5);
    assert_eq!(f.ceiling_ingest, 2.5);
    assert_eq!(f.ceiling_search, 16.0);
    for bad in [
        ("approximate_threshold", "x"),
        ("quantization_ceiling_ingest", "x"),
        ("quantization_ceiling_search", "x"),
    ] {
        assert!(
            SeismicField::from_field_info(&info(&[("sparse_vector_field", "true"), bad])).is_err()
        );
    }
}

#[test]
fn query_folds_tokens_and_cuts_by_weight() {
    let f = field();
    let q = SeismicQuery::new(
        &[(1, 0.5), (32769, 2.0), (3, 1.0), (4, 1.0)],
        2,
        0,
        1.0,
        2.0,
        &f,
    )
    .unwrap();
    // 32769 folds onto 1 and its larger weight wins; 3 and 4 tie, the lower token first.
    assert_eq!(q.top_tokens, vec!["1", "3"]);
    assert_eq!(q.k, 10);
    assert_eq!(q.dense[1], quantize_for(2.0));
    assert_eq!(q.score_scale, 2.0 * 3.0 * 16.0 / 255.0 / 255.0);
    let nan = SeismicQuery::new(&[(1, 0.5), (1, f32::NAN)], 1, 1, 1.0, 1.0, &f).unwrap();
    assert_eq!(nan.dense, vec![0, 0]);
    for hf in [0.0, -1.0, f32::NAN] {
        assert!(SeismicQuery::new(&[(1, 1.0)], 1, 1, hf, 1.0, &f).is_err());
    }
}

fn quantize_for(v: f32) -> u8 {
    crate::vector::quantize(v, 16.0)
}

#[test]
fn score_heap_matches_java_priority_queue() {
    // java.util.PriorityQueue keeps, of equal scores, the one its heap
    // order leaves at the root longest: (1,5) is evicted before (2,5).
    let mut h = ScoreHeap::new(2);
    h.add(1, 5);
    h.add(2, 5);
    h.add(3, 7);
    assert_eq!(h.into_sorted_by_doc(), vec![(2, 5), (3, 7)]);
    let mut h = ScoreHeap::new(3);
    for (d, s) in [(0, 9), (1, 3), (2, 6), (3, 1), (4, 8), (5, 2)] {
        h.add(d, s);
    }
    // A score at or below the evicted minimum is not admitted.
    h.add(6, 6);
    assert_eq!(h.into_sorted_by_doc(), vec![(0, 9), (2, 6), (4, 8)]);
    let mut empty = ScoreHeap::new(1);
    empty.poll();
    assert_eq!(empty.min_score(), None);
    let mut none = ScoreHeap::new(0);
    none.add(1, 1);
    assert!(none.into_sorted_by_doc().is_empty());
}

#[test]
fn search_without_doc_values_or_past_bounds_finds_nothing() {
    let f = field();
    let postings = ClusteredPostings::from_clusters(
        1,
        vec![(
            "2",
            vec![Cluster {
                docs: vec![-1, 0, 1, 9],
                weights: vec![1, 1, 1, 1],
                should_not_skip: true,
                summary: Some(SparseVector::from_items(vec![(2, 1)])),
            }],
        )],
    );
    let mut live = FixedBitSet::new(2);
    live.set(1);
    let view = SegmentView {
        max_doc: 2,
        live_docs: Some(&live),
        forward: None,
        postings: &postings,
    };
    let q = SeismicQuery::new(&[(2, 1.0), (5, 0.5)], 2, 10, 1.0, 1.0, &f).unwrap();
    assert!(search(&view, &f, &q, None).unwrap().is_empty());
    let mut filter = FixedBitSet::new(2);
    filter.set(0);
    // The filter holds fewer docs than k: exact scoring, a doc without a vector scores 0.
    assert_eq!(
        search(&view, &f, &q, Some(&filter)).unwrap(),
        vec![(0, 0.0)]
    );
    let view = SegmentView {
        live_docs: None,
        max_doc: 1,
        ..view
    };
    let q = SeismicQuery::new(&[(2, 1.0)], 1, 1, 1.0, 1.0, &f).unwrap();
    let mut wide = FixedBitSet::new(1);
    wide.set(0);
    let none = FixedBitSet::new(1);
    // A filter wider than k becomes a post-filter.
    let q0 = SeismicQuery { k: 0, ..q };
    assert!(search(&view, &f, &q0, Some(&none)).unwrap().is_empty());
    assert!(search(&view, &f, &q0, Some(&wide)).unwrap().is_empty());
}
