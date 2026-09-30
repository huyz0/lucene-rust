//! The document queries' own edges: argument checks, schema mismatches,
//! rewrites, the collectors they drive -- over small indexes this port
//! writes. Their agreement with Lucene is `tests/document_fields_fixtures.rs`.

use std::net::IpAddr;

use lucene_index::document::{self as d, Document, IndexableField, Store};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

use super::*;
use crate::collector::TopDocsCollector;
use crate::directory_reader::DirectoryReader;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// Writes `docs` (one segment per inner list) and opens the index.
fn index(tmp: &TempDir, segments: Vec<Vec<Vec<Box<dyn IndexableField>>>>) -> DirectoryReader {
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    for seg in segments {
        for fields in seg {
            let mut doc = Document::new();
            for f in fields {
                doc.add_boxed(f);
            }
            w.add_fields_document(&doc).unwrap();
        }
        w.commit().unwrap();
    }
    drop(w);
    DirectoryReader::open(&dir).unwrap()
}

fn docs(leaves: &[OpenSegment<'_>], q: &dyn DocumentQuery) -> Vec<i32> {
    search_all(leaves, q)
        .unwrap()
        .into_iter()
        .map(|h| h.doc_id)
        .collect()
}

fn err(leaves: &[OpenSegment<'_>], q: &dyn DocumentQuery) -> String {
    search_all(leaves, q).unwrap_err().to_string()
}

fn b(f: impl IndexableField + 'static) -> Box<dyn IndexableField> {
    Box::new(f)
}

fn ip(s: &str) -> IpAddr {
    s.parse().unwrap()
}

#[test]
fn point_queries_check_their_arguments_and_the_fields_shape() {
    assert!(PointRangeQuery::new("f", vec![1], vec![2], 0).is_err());
    assert!(PointRangeQuery::new("f", vec![], vec![], 1).is_err());
    assert!(PointRangeQuery::new("f", vec![1, 2, 3], vec![1, 2, 3], 2).is_err());
    assert!(PointRangeQuery::new("f", vec![1, 2], vec![1], 1).is_err());
    assert!(PointInSetQuery::new("f", 1, 0, vec![]).is_err());
    assert!(PointInSetQuery::new("f", 1, 17, vec![]).is_err());
    assert!(PointInSetQuery::new("f", 0, 4, vec![]).is_err());
    assert!(PointInSetQuery::new("f", 1, 4, vec![vec![1]]).is_err());
    assert!(int_point::new_range_query_nd("f", &[1, 2], &[3]).is_err());
    assert!(int_point::new_range_query_nd("f", &[], &[]).is_err());
    assert!(binary_point::new_range_query_nd("f", &[], &[]).is_err());
    assert!(binary_point::new_range_query_nd("f", &[b"a"], &[b""]).is_err());
    assert!(binary_point::new_set_query("f", &[b"a", b"bc"]).is_err());
    assert!(inet_address_point::new_prefix_query("f", ip("1.1.1.1"), 40).is_err());

    let tmp = TempDir::new("doc-points-shape");
    let r = index(
        &tmp,
        vec![vec![
            vec![
                b(d::IntPoint::new("p", &[1, 2]).unwrap()),
                b(d::IntField::new("i", 5, Store::No)),
            ],
            vec![b(d::LongPoint::new("lp", &[7]).unwrap())],
        ]],
    );
    let opened = r.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    assert!(err(&leaves, &int_point::new_range_query("p", 0, 9).unwrap()).contains("numDims=1"));
    assert!(err(
        &leaves,
        &long_point::new_range_query_nd("p", &[0, 0], &[9, 9]).unwrap()
    )
    .contains("bytesPerDim"));
    assert!(err(&leaves, &int_point::new_set_query("p", &[1]).unwrap()).contains("numIndexDims"));
    assert_eq!(
        docs(
            &leaves,
            &int_point::new_range_query_nd("p", &[0, 0], &[9, 9]).unwrap()
        ),
        vec![0]
    );
    assert!(docs(&leaves, &int_point::new_range_query("nope", 0, 9).unwrap()).is_empty());
    assert!(docs(&leaves, &int_point::new_range_query("i", 9, 10).unwrap()).is_empty());
    assert!(docs(
        &leaves,
        binary_point::new_set_query("f", &[]).unwrap().as_ref()
    )
    .is_empty());
    // Points without doc values: the `IndexOrDocValuesQuery` has no
    // doc-values side and matches nothing, as Java's does.
    assert!(docs(&leaves, &long_field::new_range_query("lp", 0, 9).unwrap()).is_empty());
    assert_eq!(
        docs(&leaves, &int_field::new_exact_query("i", 5).unwrap()),
        vec![0]
    );
    assert!(docs(&leaves, &int_field::new_exact_query("zz", 5).unwrap()).is_empty());
    let dv_only = IndexOrDocValuesQuery::new("i", Box::new(MatchNoDocs), Box::new(MatchAllDocs));
    assert!(format!("{dv_only:?}").contains("IndexOrDocValuesQuery"));
    let s = int_field::new_sort_field_with_missing("i", true, NumericSelector::Max, 3);
    assert_eq!((s.missing, s.reverse), (3, true));
    let s = keyword_field::new_sort_field("k", false, SortedSetSelector::Max);
    assert_eq!(s.selector, crate::top_field::Selector::Max);
}

#[test]
fn doc_values_queries_refuse_the_wrong_doc_values_type() {
    let tmp = TempDir::new("doc-dv-types");
    let r = index(
        &tmp,
        vec![vec![vec![
            b(d::SortedDocValuesField::new("s", "a")),
            b(d::NumericDocValuesField::new("n", 1)),
            b(d::KeywordField::new("k", "a", Store::No)),
        ]]],
    );
    let opened = r.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    assert!(err(
        &leaves,
        &numeric_doc_values_field::new_slow_range_query("s", 0, 1)
    )
    .contains("unexpected docvalues type"));
    assert!(err(
        &leaves,
        &numeric_doc_values_field::new_slow_set_query("s", &[1])
    )
    .contains("unexpected"));
    assert!(err(
        &leaves,
        &sorted_doc_values_field::new_slow_exact_query("n", b"a")
    )
    .contains("unexpected"));
    assert!(err(
        &leaves,
        &sorted_doc_values_field::new_slow_set_query("n", &[b"a"])
    )
    .contains("unexpected"));
    assert!(err(
        &leaves,
        &int_range_doc_values_field::new_slow_intersects_query("n", &[1], &[2]).unwrap()
    )
    .contains("expected=BINARY"));
    // Absent fields and fields without doc values match nothing.
    for q in [
        Box::new(numeric_doc_values_field::new_slow_range_query("zz", 0, 1))
            as Box<dyn DocumentQuery>,
        Box::new(numeric_doc_values_field::new_slow_set_query("zz", &[1])),
        Box::new(sorted_doc_values_field::new_slow_exact_query("zz", b"a")),
        Box::new(sorted_doc_values_field::new_slow_set_query("zz", &[b"a"])),
        Box::new(int_range_doc_values_field::new_slow_intersects_query("zz", &[1], &[2]).unwrap()),
        Box::new(keyword_field::new_set_query("zz", &[b"a"])),
        Box::new(sorted_doc_values_field::new_slow_set_query("s", &[b"zz"])),
    ] {
        assert!(docs(&leaves, q.as_ref()).is_empty(), "{q:?}");
    }
    assert!(docs(&leaves, &keyword_field::new_set_query("k", &[])).is_empty());
    assert_eq!(
        docs(&leaves, &keyword_field::new_set_query("k", &[b"a", b"a"])),
        vec![0]
    );
    assert!(docs(
        &leaves,
        &numeric_doc_values_field::new_slow_set_query("n", &[])
    )
    .is_empty());
    assert!(docs(
        &leaves,
        &sorted_set_doc_values_field::new_slow_set_query("s", &[])
    )
    .is_empty());
}

#[test]
fn range_queries_check_their_arguments() {
    assert!(RangeFieldQuery::new("f", vec![1; 8], 5, d::RangeQueryType::Within).is_err());
    assert!(RangeFieldQuery::new("f", vec![], 1, d::RangeQueryType::Within).is_err());
    assert!(RangeFieldQuery::new("f", vec![1], 0, d::RangeQueryType::Within).is_err());
    assert!(int_range::new_intersects_query("f", &[], &[]).is_err());
    assert!(int_range::new_intersects_query("f", &[1], &[1, 2]).is_err());
    assert!(int_range::new_intersects_query("f", &[5], &[1]).is_err());
    assert!(int_range_doc_values_field::new_slow_intersects_query("f", &[5], &[1]).is_err());
    assert!(int_range_doc_values_field::new_slow_intersects_query("f", &[], &[]).is_err());
    assert!(inet_address_range::new_within_query("f", ip("1.1.1.2"), ip("1.1.1.1")).is_err());
    assert!(BinaryRangeFieldRangeQuery::new("f", vec![], 4, 1, d::RangeQueryType::Within).is_err());

    let tmp = TempDir::new("doc-ranges");
    let r = index(
        &tmp,
        vec![vec![
            vec![b(d::IntRange::new("r", &[1, 1], &[5, 5]).unwrap())],
            vec![b(d::IntRange::new("r", &[2, 2], &[3, 3]).unwrap())],
        ]],
    );
    let opened = r.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    assert!(err(
        &leaves,
        &int_range::new_intersects_query("r", &[1], &[2]).unwrap()
    )
    .contains("numDims=2"));
    // Every document has a box and the field's bounds are inside the query:
    // all match without walking the tree.
    assert_eq!(
        docs(
            &leaves,
            &int_range::new_within_query("r", &[0, 0], &[9, 9]).unwrap()
        ),
        vec![0, 1]
    );
    assert!(docs(
        &leaves,
        &int_range::new_crosses_query("zz", &[0, 0], &[9, 9]).unwrap()
    )
    .is_empty());
}

#[test]
fn feature_factories_check_their_arguments() {
    use feature_field::*;
    assert!(new_linear_query("f", "x", 0.0).is_err());
    assert!(new_linear_query("f", "x", 65.0).is_err());
    assert!(new_linear_query("f", "x", f32::NAN).is_err());
    assert!(new_log_query("f", "x", 1.0, 0.5).is_err());
    assert!(new_log_query("f", "x", 1.0, f32::INFINITY).is_err());
    assert!(new_saturation_query("f", "x", 1.0, 0.0).is_err());
    assert!(new_saturation_query("f", "x", 1.0, f32::NAN).is_err());
    assert!(new_sigmoid_query("f", "x", 1.0, 0.0, 1.0).is_err());
    assert!(new_sigmoid_query("f", "x", 1.0, 1.0, 0.0).is_err());
    assert!(new_sigmoid_query("f", "x", 0.0, 1.0, 1.0).is_err());
    assert!(FeatureFunction::Saturation { pivot: None }
        .score(1.0, 5.0)
        .is_err());
    let mut s = new_feature_sort("f", "x");
    assert!(s.set_missing_value().is_err());
    assert_eq!(s.feature, "x");
}

#[test]
fn feature_values_are_forward_only() {
    let tmp = TempDir::new("doc-features");
    let r = index(
        &tmp,
        vec![vec![
            vec![b(d::FeatureField::new("f", "x", 2.0).unwrap())],
            vec![b(d::StringField::new("id", "1", Store::No))],
            vec![b(d::FeatureField::new("f", "x", 8.0).unwrap())],
        ]],
    );
    let opened = r.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    let mut v = feature_field::new_double_values("f", "x")
        .get_values(&leaves[0])
        .unwrap();
    assert!(v.advance_exact(0));
    assert_eq!(v.double_value(), 2.0);
    assert!(!v.advance_exact(1));
    assert!(v.advance_exact(2));
    assert!(!v.advance_exact(1), "backwards");
    assert!(!v.advance_exact(3));
    assert_eq!(v.value_for_doc(5), 0.0);
    let none = feature_field::new_double_values("zz", "x")
        .get_values(&leaves[0])
        .unwrap();
    assert_eq!(none.clone().value_for_doc(0), 0.0);
    let missing = feature_field::new_double_values("f", "zz")
        .get_values(&leaves[0])
        .unwrap();
    assert_eq!(missing.clone().value_for_doc(0), 0.0);
    // The sort over a query's hits: highest value first.
    let top = feature_field::new_feature_sort("f", "x")
        .search(&leaves, &MatchAllDocs, 2)
        .unwrap();
    assert_eq!(top, vec![(2, 8.0), (0, 2.0)]);
    assert!(docs(
        &leaves,
        feature_field::new_linear_query("zz", "x", 1.0)
            .unwrap()
            .as_ref()
    )
    .is_empty());
    assert_eq!(
        compute_pivot_for_test(&leaves),
        d::FeatureField::decode_feature_value(
            ((2.0f32.to_bits() >> 15) as f32 + (8.0f32.to_bits() >> 15) as f32) / 2.0
        )
    );
}

fn compute_pivot_for_test(leaves: &[OpenSegment<'_>]) -> f32 {
    feature::compute_pivot_feature_value(leaves, "f", "x").unwrap()
}

#[test]
fn generic_queries_and_the_search_loop() {
    let tmp = TempDir::new("doc-generic");
    let r = index(
        &tmp,
        vec![
            vec![
                vec![b(d::TextField::new("t", "a b", Store::No))],
                vec![b(d::NumericDocValuesField::new("n", 3))],
            ],
            vec![vec![b(d::TextField::new("t", "c", Store::No))]],
        ],
    );
    let opened = r.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    assert_eq!(docs(&leaves, &MatchAllDocs), vec![0, 1, 2]);
    assert!(docs(&leaves, &MatchNoDocs).is_empty());
    assert_eq!(
        docs(&leaves, &FieldExists { field: "t".into() }),
        vec![0, 2]
    );
    assert_eq!(docs(&leaves, &FieldExists { field: "n".into() }), vec![1]);
    let boosted = Boosted::new(Box::new(MatchAllDocs), 2.5);
    let top = search_top_docs(&leaves, &boosted, 2).unwrap();
    assert_eq!(top.total_hits.value, 3);
    assert_eq!(top.score_docs.len(), 2);
    assert_eq!(top.score_docs[0].score, 2.5);
    // A boosted query rewrites its inner query and keeps the boost.
    let q = Boosted::new(
        Box::new(numeric_doc_values_field::new_slow_range_query(
            "n",
            i64::MIN,
            i64::MAX,
        )),
        2.0,
    );
    let hits = search_all(&leaves, &q).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].score, 2.0);
    // A leaf without its reader cannot be searched.
    let bare = OpenSegment {
        reader: None,
        ..leaves[0]
    };
    assert!(MatchAllDocs
        .score_leaf(&bare, 1.0, &mut TopDocsCollector::new(1))
        .is_err());
}

#[test]
fn the_long_hash_set_is_lucenes() {
    let s = DocValuesLongHashSet::new(&[i64::MIN, -5, 3, 3, 1 << 40]);
    assert_eq!(s.size(), 4);
    assert!(s.contains(i64::MIN) && s.contains(-5) && s.contains(1 << 40));
    assert!(!s.contains(4));
    assert_eq!(s.min_value, i64::MIN);
    assert_eq!(s.max_value, 1 << 40);
    let mut v: Vec<i64> = s.values().collect();
    v.sort_unstable();
    assert_eq!(v, vec![i64::MIN, -5, 3, 1 << 40]);
    let empty = DocValuesLongHashSet::new(&[]);
    assert_eq!(empty.size(), 0);
    assert!(!empty.contains(0));
    assert!(!empty.contains(i64::MIN));
    assert_eq!((empty.min_value, empty.max_value), (i64::MAX, i64::MIN));
    // Many values: the table grows to a power of two past 3/2 of them.
    let many: Vec<i64> = (0..1000).map(|i| i * 7919).collect();
    let s = DocValuesLongHashSet::new(&many);
    assert!(many.iter().all(|&x| s.contains(x)));
    assert!(!s.contains(1));
}

#[test]
fn the_range_bulk_scorer_collects_its_window() {
    assert!(RangeBulkScorer::new(3, 3, 1.0).is_err());
    let s = RangeBulkScorer::new(2, 6, 1.5).unwrap();
    assert_eq!(s.cost(), 4);
    let mut all = TopDocsCollector::new(10);
    assert_eq!(s.score(&mut all, None, 0, 2), 2, "window before the range");
    assert_eq!(s.score(&mut all, None, 6, 9), NO_MORE, "window after it");
    let mut live = lucene_util::fixed_bit_set::FixedBitSet::new(10);
    for doc in [2, 4, 5] {
        // FBS: constant indexes below the bitset's length of 10.
        live.set(doc);
    }
    assert_eq!(s.score(&mut all, Some(&live), 3, 5), 5);
    assert_eq!(s.score(&mut all, Some(&live), 5, 100), NO_MORE);
    let got: Vec<i32> = all.top_docs().iter().map(|d| d.doc_id).collect();
    assert_eq!(got, vec![4, 5]);
    // An empty range scores nothing.
    let mut none = TopDocsCollector::new(10);
    RangeBulkScorer::score_range((4, 4), None, 1.0, &mut none);
    assert!(none.top_docs().is_empty());
}

const NO_MORE: i32 = i32::MAX;

#[test]
fn the_skipper_supplier_finds_the_run() {
    use lucene_codecs::doc_values::{
        DocValuesSkipIndex, SkipIndexInterval, SkipIndexLevelInterval,
    };
    // Values ascending by doc: doc d has value d / 2, over 8 docs, in two
    // intervals of 4.
    // doc d has value d / 2 (ascending) or 3 - d / 2 (descending), over 8
    // docs in two intervals of 4.
    let value = |doc: i32, reverse: bool| {
        let v = i64::from(doc / 2);
        if reverse {
            3 - v
        } else {
            v
        }
    };
    let index_of = |reverse: bool| {
        let level = |min_doc: i32, max_doc: i32| {
            let (a, b) = (value(min_doc, reverse), value(max_doc, reverse));
            SkipIndexLevelInterval {
                max_doc_id: max_doc,
                min_doc_id: min_doc,
                max_value: a.max(b),
                min_value: a.min(b),
                doc_count: max_doc - min_doc + 1,
            }
        };
        DocValuesSkipIndex {
            min_value: 0,
            max_value: 3,
            doc_count: 8,
            max_doc_id: 7,
            max_value_count: 1,
            intervals: vec![
                SkipIndexInterval {
                    levels: vec![level(0, 3)],
                },
                SkipIndexInterval {
                    levels: vec![level(4, 7)],
                },
            ],
        }
    };
    let (asc, desc) = (index_of(false), index_of(true));
    let run = |lo: i64, hi: i64, reverse: bool| {
        let index = if reverse { &desc } else { &asc };
        let mut s = SortedSkipperScorerSupplier::new(index, reverse, lo, hi);
        let cost = s.cost();
        let mut cursor = -1i32;
        let range = s
            .range(|start, pred| {
                if start > cursor {
                    cursor = start;
                }
                while cursor < 8 {
                    if pred(value(cursor, reverse)) {
                        return Ok(cursor);
                    }
                    cursor += 1;
                }
                Ok(NO_MORE)
            })
            .unwrap();
        (range, cost)
    };
    assert_eq!(run(0, 3, false).0, (0, 8), "everything");
    assert_eq!(run(5, 9, false).0, (NO_MORE, NO_MORE), "nothing");
    assert_eq!(run(2, 1, false).0, (NO_MORE, NO_MORE), "empty range");
    assert_eq!(run(1, 2, false).0, (2, 6));
    assert_eq!(run(2, 3, false).0, (4, 8));
    assert_eq!(run(0, 0, false).0, (0, 2));
    assert!(run(1, 2, false).1 > 0);
    assert_eq!(run(0, 3, true).0, (0, 8));
    assert_eq!(run(1, 2, true).0, (2, 6));
    assert_eq!(run(0, 1, true).0, (4, 8));
    assert_eq!(run(3, 3, true).0, (0, 2));
    assert_eq!(run(2, 3, true).0, (0, 4));
}

#[test]
fn distance_feature_prunes_without_changing_the_top_hits() {
    let tmp = TempDir::new("doc-distance");
    let seg: Vec<Vec<Box<dyn IndexableField>>> = (0..3000i64)
        .map(|i| vec![b(d::LongField::new("l", (i * 37) % 3001, Store::No))])
        .collect();
    let mut with_multi = seg;
    with_multi.push(vec![
        b(d::LongField::new("l", -50, Store::No)),
        b(d::LongField::new("l", 40, Store::No)),
    ]);
    with_multi.push(vec![b(d::StringField::new("id", "x", Store::No))]);
    let r = index(&tmp, vec![with_multi]);
    let opened = r.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    let q = long_field::new_distance_feature_query("l", 2.0, 1500, 100).unwrap();
    let mut all = search_all(&leaves, q.as_ref()).unwrap();
    all.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc_id.cmp(&b.doc_id)));
    for n in [1, 3, 10] {
        let mut top = TopDocsCollector::new(n);
        q.score_leaf(&leaves[0], 1.0, &mut top).unwrap();
        assert_eq!(top.top_docs(), &all[..n], "top {n}");
    }
    assert!(LongDistanceFeatureQuery::new("l", 0, 0).is_err());
    assert!(docs(&leaves, &LongDistanceFeatureQuery::new("id", 0, 1).unwrap()).is_empty());
    assert!(docs(&leaves, &LongDistanceFeatureQuery::new("zz", 0, 1).unwrap()).is_empty());
}

#[test]
fn fields_missing_a_structure_match_nothing() {
    let tmp = TempDir::new("doc-missing-structures");
    // Every name below exists only as a stored field, so each query finds its
    // field in the infos without the structure it reads.
    let stored = |name: &str| b(d::Field::stored(name, d::StoredValue::Int(1)));
    let r = index(
        &tmp,
        vec![vec![vec![
            stored("n"),
            stored("r"),
            stored("s"),
            b(d::StringField::new("k", "a", Store::No)),
            b(d::LongPoint::new("lp", &[1]).unwrap()),
        ]]],
    );
    let opened = r.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    let queries: Vec<Box<dyn DocumentQuery>> = vec![
        Box::new(numeric_doc_values_field::new_slow_range_query("n", 0, 1)),
        Box::new(numeric_doc_values_field::new_slow_range_query(
            "absent", 0, 1,
        )),
        Box::new(numeric_doc_values_field::new_slow_set_query("n", &[1])),
        Box::new(numeric_doc_values_field::new_slow_set_query("absent", &[1])),
        Box::new(sorted_doc_values_field::new_slow_exact_query("s", b"a")),
        Box::new(sorted_doc_values_field::new_slow_exact_query(
            "absent", b"a",
        )),
        Box::new(sorted_set_doc_values_field::new_slow_set_query(
            "s",
            &[b"a"],
        )),
        Box::new(sorted_set_doc_values_field::new_slow_set_query(
            "absent",
            &[b"a"],
        )),
        Box::new(keyword_field::new_set_query("k", &[b"a"])),
        Box::new(int_range::new_intersects_query("r", &[0], &[1]).unwrap()),
        Box::new(int_range_doc_values_field::new_slow_intersects_query("r", &[0], &[1]).unwrap()),
        Box::new(int_point::new_range_query("n", 0, 1).unwrap()),
        Box::new(int_field::new_range_query("n", 0, 1).unwrap()),
        Box::new(LongDistanceFeatureQuery::new("lp", 0, 1).unwrap()),
    ];
    for q in &queries {
        assert!(docs(&leaves, q.as_ref()).is_empty(), "{q:?}");
    }
    assert_eq!(
        docs(&leaves, &keyword_field::new_exact_query("k", b"a")),
        vec![0]
    );
}

#[test]
fn skip_indexes_answer_whole_segments() {
    let tmp = TempDir::new("doc-skip-whole");
    let seg: Vec<Vec<Box<dyn IndexableField>>> = (0..40i64)
        .map(|i| {
            vec![
                b(d::NumericDocValuesField::indexed_field("n", i)),
                b(d::SortedSetDocValuesField::indexed_field(
                    "s",
                    format!("v{:02}", i % 7),
                )),
                b(d::SortedNumericDocValuesField::new("m", i)),
                b(d::SortedNumericDocValuesField::new("m", i + 100)),
                b(d::SortedSetDocValuesField::new(
                    "one",
                    format!("o{}", i % 3),
                )),
            ]
        })
        .collect();
    let r = index(&tmp, vec![seg]);
    let opened = r.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    let all: Vec<i32> = (0..40).collect();
    assert_eq!(
        docs(
            &leaves,
            &numeric_doc_values_field::new_slow_range_query("n", -5, 100)
        ),
        all
    );
    assert!(docs(
        &leaves,
        &numeric_doc_values_field::new_slow_range_query("n", 50, 60)
    )
    .is_empty());
    assert_eq!(
        docs(
            &leaves,
            &numeric_doc_values_field::new_slow_range_query("n", 3, 4)
        ),
        vec![3, 4]
    );
    assert_eq!(
        docs(
            &leaves,
            &sorted_set_doc_values_field::new_slow_range_query(
                "s",
                Some(b"a"),
                Some(b"z"),
                true,
                true
            )
        ),
        all
    );
    assert!(docs(
        &leaves,
        &sorted_set_doc_values_field::new_slow_range_query("s", Some(b"x"), Some(b"z"), true, true)
    )
    .is_empty());
    assert!(docs(
        &leaves,
        &sorted_set_doc_values_field::new_slow_range_query(
            "s",
            Some(b"v03"),
            Some(b"v02"),
            true,
            true
        )
    )
    .is_empty());
    // Multi-valued sets: values below the set's minimum are skipped, one
    // above its maximum ends the document.
    assert_eq!(
        docs(
            &leaves,
            &sorted_numeric_doc_values_field::new_slow_set_query("m", &[5, 6])
        ),
        vec![5, 6]
    );
    assert_eq!(
        docs(
            &leaves,
            &sorted_numeric_doc_values_field::new_slow_set_query("m", &[105, 1000])
        ),
        vec![5]
    );
    // A single-valued SORTED_SET field.
    assert_eq!(
        docs(
            &leaves,
            &sorted_set_doc_values_field::new_slow_exact_query("one", b"o1")
        )
        .len(),
        13
    );
    // `advance_until` past the last document.
    let mut cursor = -1;
    assert_eq!(
        doc_values_queries::advance_until(0, &mut cursor, 3, |_| Ok(false)).unwrap(),
        NO_MORE
    );
    assert_eq!(cursor, NO_MORE);
    let mut cursor = 5;
    assert_eq!(
        doc_values_queries::advance_until(2, &mut cursor, 9, |d| Ok(d == 7)).unwrap(),
        7,
        "starts where the cursor already is"
    );
}

#[test]
fn range_trees_with_inside_cells() {
    let tmp = TempDir::new("doc-range-tree");
    let seg: Vec<Vec<Box<dyn IndexableField>>> = (0..3000i32)
        .map(|i| {
            vec![
                b(d::IntRange::new("r", &[i], &[i + 5]).unwrap()),
                b(d::IntRangeDocValuesField::new("rdv", &[i], &[i + 5]).unwrap()),
            ]
        })
        .collect();
    let r = index(&tmp, vec![seg]);
    let opened = r.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    for (q, want) in [
        (
            int_range::new_within_query("r", &[100], &[2000]).unwrap(),
            1896,
        ),
        (
            int_range::new_intersects_query("r", &[100], &[2000]).unwrap(),
            1906,
        ),
        (
            int_range::new_contains_query("r", &[100], &[101]).unwrap(),
            5,
        ),
        (
            int_range::new_crosses_query("r", &[100], &[2000]).unwrap(),
            10,
        ),
    ] {
        assert_eq!(docs(&leaves, &q).len(), want, "{q:?}");
    }
    let dv = int_range_doc_values_field::new_slow_intersects_query("rdv", &[100], &[2000]).unwrap();
    assert_eq!(docs(&leaves, &dv).len(), 1906);
    let r0 = leaves[0].reader.unwrap();
    let info = r0
        .field_infos()
        .fields
        .iter()
        .find(|f| f.name == "rdv")
        .unwrap();
    let (meta, data) = r0.doc_values_for_field(info.number).unwrap();
    let mut values = BinaryRangeDocValues::new(
        lucene_codecs::doc_values::BinaryReader::new(data, meta.binary_entry(info.number).unwrap()),
        1,
        4,
    );
    assert!(format!("{values:?}").contains("packed_len"));
    assert_eq!(values.packed_value(0).unwrap().unwrap().len(), 8);
    let mut too_wide = BinaryRangeDocValues::new(
        lucene_codecs::doc_values::BinaryReader::new(data, meta.binary_entry(info.number).unwrap()),
        2,
        4,
    );
    assert!(too_wide.packed_value(0).is_err());
}

#[test]
fn distance_feature_over_numeric_values_and_edge_origins() {
    let tmp = TempDir::new("doc-distance-edges");
    let seg: Vec<Vec<Box<dyn IndexableField>>> = (0..2000i64)
        .map(|i| {
            vec![
                b(d::LongPoint::new("x", &[i]).unwrap()),
                b(d::NumericDocValuesField::new("x", i)),
                b(d::LongPoint::new("s", &[i]).unwrap()),
                b(d::SortedDocValuesField::new("s", "v")),
                b(d::LongField::new("up", i - 1000, Store::No)),
            ]
        })
        .collect();
    let r = index(&tmp, vec![seg]);
    let opened = r.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    assert!(
        err(&leaves, &LongDistanceFeatureQuery::new("s", 0, 1).unwrap())
            .contains("unexpected docvalues")
    );
    // Every document scores closer to the origin than the one before, so the
    // collector's threshold rises on every hit and the scorer keeps pruning.
    for (origin, pivot) in [
        (2000, 10),
        (i64::MIN + 3, 1_000_000),
        (i64::MAX - 3, 1_000_000),
    ] {
        for field in ["x", "up"] {
            let q = LongDistanceFeatureQuery::new(field, origin, pivot).unwrap();
            let mut all = search_all(&leaves, &q).unwrap();
            all.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc_id.cmp(&b.doc_id)));
            for n in [1, 4] {
                let mut top = TopDocsCollector::new(n);
                q.score_leaf(&leaves[0], 1.0, &mut top).unwrap();
                assert_eq!(top.top_docs(), &all[..n], "{field} {origin} top {n}");
            }
        }
    }
}

#[test]
fn point_fields_without_points_or_doc_values() {
    let tmp = TempDir::new("doc-point-shapes");
    let r = index(
        &tmp,
        vec![vec![vec![
            b(d::Field::stored("n", d::StoredValue::Int(1))),
            b(d::NumericDocValuesField::new("dvonly", 1)),
        ]]],
    );
    let opened = r.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    assert!(docs(&leaves, &int_point::new_set_query("n", &[1]).unwrap()).is_empty());
    assert!(docs(&leaves, &int_field::new_set_query("n", &[1]).unwrap()).is_empty());
    assert!(docs(&leaves, &int_field::new_set_query("dvonly", &[1]).unwrap()).is_empty());
    assert!(docs(
        &leaves,
        &double_field::new_exact_query("absent", 1.0).unwrap()
    )
    .is_empty());
}
