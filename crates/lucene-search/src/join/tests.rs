// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::directory_reader::DirectoryReader;
use crate::query::{Clause, MatchAllDocsQuery, TermQuery};
use lucene_index::buffered_updates::Term;
use lucene_index::document::{Document, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn doc(id: &str, kind: &str, block: usize) -> Document {
    let mut d = Document::new();
    d.add(StringField::new("id", id, Store::Yes));
    d.add(StringField::new("type", kind, Store::No));
    d.add(StringField::new("block", block.to_string(), Store::No));
    d
}

fn block(b: usize, children: usize) -> Vec<Document> {
    let mut docs: Vec<Document> = (0..children)
        .map(|j| doc(&format!("c{b}_{j}"), "child", b))
        .collect();
    docs.push(doc(&format!("p{b}"), "parent", b));
    docs
}

fn term(field: &str, value: &str) -> BooleanQuery {
    BooleanQuery {
        must: vec![Clause::Term(TermQuery::new(
            field,
            value.as_bytes().to_vec(),
        ))],
        ..Default::default()
    }
}

fn parents() -> QueryBitSetProducer {
    QueryBitSetProducer::new(term("type", "parent"))
}

/// Writes `ops` into a fresh index and runs `check_join_index` over it.
fn check(name: &str, ops: impl FnOnce(&mut IndexWriter<'_>)) -> Result<()> {
    let tmp = TempDir::new(name);
    let dir = FsDirectory::open(tmp.path());
    {
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        ops(&mut w);
        w.commit().unwrap();
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    check_join_index(&segments, &parents())
}

#[test]
fn a_well_formed_block_index_passes() {
    check("cji-ok", |w| {
        for b in 0..20 {
            w.add_fields_documents(&block(b, b % 4)).unwrap();
            if b % 7 == 6 {
                w.commit().unwrap();
            }
        }
        // Whole blocks deleted, by a term every document of the block has.
        w.delete_documents_by_term(&[Term::new("block", "3"), Term::new("block", "12")])
            .unwrap();
    })
    .unwrap();
}

#[test]
fn every_segment_needs_a_parent() {
    let err = check("cji-no-parent", |w| {
        w.add_fields_document(&doc("c0", "child", 0)).unwrap();
    })
    .unwrap_err();
    assert!(
        err.to_string()
            .starts_with("Every segment should have at least one parent, but "),
        "{err}"
    );
}

#[test]
fn a_segment_must_end_in_a_parent() {
    let err = check("cji-child-last", |w| {
        w.add_fields_documents(&block(0, 2)).unwrap();
        w.add_fields_document(&doc("c1_0", "child", 1)).unwrap();
    })
    .unwrap_err();
    assert!(
        err.to_string().ends_with("has a child as a last doc"),
        "{err}"
    );
}

#[test]
fn blocks_are_deleted_whole() {
    let err = check("cji-deleted-child", |w| {
        w.add_fields_documents(&block(0, 2)).unwrap();
        w.add_fields_documents(&block(1, 2)).unwrap();
        w.delete_documents_by_term(&[Term::new("id", "c1_1")])
            .unwrap();
    })
    .unwrap_err();
    assert_eq!(
        err.to_string()
            .split(" of segment ")
            .next()
            .map(str::to_string),
        Some("Parent doc 5".to_string()),
        "{err}"
    );
    assert!(err
        .to_string()
        .ends_with("is live but has a deleted child document 4"));

    let err = check("cji-deleted-parent", |w| {
        w.add_fields_documents(&block(0, 2)).unwrap();
        w.delete_documents_by_term(&[Term::new("id", "p0")])
            .unwrap();
    })
    .unwrap_err();
    assert!(
        err.to_string()
            .ends_with("is deleted but has a live child document 0"),
        "{err}"
    );
}

/// `QueryBitSetProducer`: the query's matches with deleted documents, one
/// set per segment core, cached; `None` without a scorer; a match-all is
/// every document.
#[test]
fn the_producer_caches_each_segment_and_keeps_deleted_parents() {
    let tmp = TempDir::new("qbsp");
    let dir = FsDirectory::open(tmp.path());
    {
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        for b in 0..3 {
            w.add_fields_documents(&block(b, 1)).unwrap();
        }
        w.delete_documents_by_term(&[Term::new("block", "1")])
            .unwrap();
        w.commit().unwrap();
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let seg = &segments[0];
    assert!(seg.live_docs.is_some());

    let producer = parents();
    let bits = producer.bit_set(seg).unwrap().unwrap();
    let set: Vec<usize> = (0..bits.len()).filter(|&i| bits.get(i)).collect();
    assert_eq!(set, [1, 3, 5], "the deleted parent 3 included");
    let again = producer.bit_set(seg).unwrap().unwrap();
    assert!(Arc::ptr_eq(&bits, &again), "cached");
    producer.clear();
    let fresh = producer.bit_set(seg).unwrap().unwrap();
    assert!(!Arc::ptr_eq(&bits, &fresh));
    assert_eq!(*fresh, *bits);
    assert_eq!(producer.query(), &term("type", "parent"));
    assert!(producer.key().starts_with("QueryBitSetProducer("));

    assert!(QueryBitSetProducer::new(term("type", "nothing"))
        .bit_set(seg)
        .unwrap()
        .is_none());
    let all = QueryBitSetProducer::new(BooleanQuery {
        must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(i32::MAX))],
        ..Default::default()
    })
    .bit_set(seg)
    .unwrap()
    .unwrap();
    assert_eq!(all.cardinality(), 6);
}

// ---------------------------------------------------------------------------
// The queries over a small fixed index
// ---------------------------------------------------------------------------

use crate::index_searcher::IndexSearcher;
use lucene_index::document::{
    NumericDocValuesField, SortedNumericDocValuesField, SortedSetDocValuesField, TextField,
};

/// Blocks: `p0` (children `a`, `a b`), `p1` (no children), `p2` (child
/// `b`); every child has a `price` and `color`, the parent `rank`.
fn small_index(name: &str) -> TempDir {
    let tmp = TempDir::new(name);
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_parent_field(Some("_parent")).unwrap();
    let child = |id: &str, body: &str, price: i64, color: &str| {
        let mut d = doc(id, "child", 0);
        d.add(TextField::new("body", body, Store::No));
        d.add(SortedNumericDocValuesField::new("price", price));
        d.add(SortedSetDocValuesField::new(
            "color",
            color.as_bytes().to_vec(),
        ));
        d
    };
    let parent = |id: &str, body: &str| {
        let mut d = doc(id, "parent", 0);
        d.add(TextField::new("body", body, Store::No));
        d.add(NumericDocValuesField::new("rank", 1));
        d
    };
    w.add_fields_documents(&[
        child("c0", "a", 5, "red"),
        child("c1", "a b", 3, "blue"),
        parent("p0", "x"),
    ])
    .unwrap();
    w.add_fields_documents(&[parent("p1", "x y")]).unwrap();
    w.add_fields_documents(&[child("c2", "b", 7, "green"), parent("p2", "y")])
        .unwrap();
    w.commit().unwrap();
    tmp
}

fn producer() -> Arc<dyn BitSetProducer> {
    Arc::new(parents())
}

fn body(t: &str) -> Clause {
    Clause::Term(TermQuery::new("body", t.as_bytes().to_vec()))
}

fn one(c: Clause) -> BooleanQuery {
    BooleanQuery {
        must: vec![c],
        ..Default::default()
    }
}

/// Runs `f` with a searcher over [`small_index`].
fn with_searcher(name: &str, f: impl FnOnce(&IndexSearcher<'_, '_>, &DirectoryReader)) {
    let tmp = small_index(name);
    let reader = DirectoryReader::open(&FsDirectory::open(tmp.path())).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<_> = owned.iter().map(Some).collect();
    let s = IndexSearcher::new(&segments, &norms).unwrap();
    f(&s, &reader);
}

fn hits(s: &IndexSearcher<'_, '_>, c: Clause) -> Vec<i32> {
    let mut d: Vec<i32> = s
        .search(&one(c), 100)
        .unwrap()
        .score_docs
        .iter()
        .map(|h| h.doc)
        .collect();
    d.sort_unstable();
    d
}

#[test]
fn queries_compare_and_print_as_java_identifies_them() {
    let p = producer();
    let tp = ToParentBlockJoinQuery::new(body("a"), Arc::clone(&p), ScoreMode::Max);
    assert_eq!(tp, tp.clone());
    assert_ne!(
        tp,
        ToParentBlockJoinQuery::new(body("a"), Arc::clone(&p), ScoreMode::Min)
    );
    assert!(format!("{tp:?}").contains("QueryBitSetProducer"));
    let tc = ToChildBlockJoinQuery::new(body("x"), Arc::clone(&p));
    assert_eq!(tc, tc.clone());
    assert!(format!("{tc:?}").starts_with("ToChildBlockJoinQuery("));
    let pc = ParentChildrenBlockJoinQuery::new(Arc::clone(&p), body("a"), 2);
    assert_eq!(pc, pc.clone());
    assert!(format!("{pc:?}").ends_with(", 2)"));
    let err =
        ParentsChildrenBlockJoinQuery::new(Arc::clone(&p), body("x"), body("a"), 0).unwrap_err();
    assert_eq!(err.to_string(), "childLimitPerParent must be > 0, got 0");
    let pcs = ParentsChildrenBlockJoinQuery::new(Arc::clone(&p), body("x"), body("a"), 2).unwrap();
    let custom = pcs.clone().with_combiner(ScoreCombiner::Custom {
        name: "max".into(),
        combine: Arc::new(f32::max),
    });
    assert_ne!(pcs, custom);
    assert!(format!("{custom:?}").ends_with(", 2, max)"));
    assert_eq!(
        [
            ScoreMode::None,
            ScoreMode::Avg,
            ScoreMode::Max,
            ScoreMode::Total,
            ScoreMode::Min
        ]
        .map(|m| m.to_string()),
        ["None", "Avg", "Max", "Total", "Min"]
    );
    let c: Clause = tp.into();
    assert!(crate::segment_cacheable::is_cacheable(&c, None));
    let c: Clause = pc.into();
    assert!(!crate::segment_cacheable::is_cacheable(&c, None));
    if let Clause::Extended(e) = Clause::from(pcs) {
        assert_eq!(e.name(), "ParentsChildrenBlockJoinQuery");
        assert_eq!(e.children().len(), 2);
    }
}

#[test]
fn every_query_finds_its_blocks() {
    with_searcher("join-small", |s, _| {
        let p = producer();
        for mode in [
            ScoreMode::None,
            ScoreMode::Avg,
            ScoreMode::Max,
            ScoreMode::Total,
            ScoreMode::Min,
        ] {
            assert_eq!(
                hits(
                    s,
                    ToParentBlockJoinQuery::new(body("a"), Arc::clone(&p), mode).into()
                ),
                [2],
                "{mode}"
            );
            assert_eq!(
                hits(
                    s,
                    ToParentBlockJoinQuery::new(body("b"), Arc::clone(&p), mode).into()
                ),
                [2, 5]
            );
        }
        assert_eq!(
            hits(
                s,
                ToChildBlockJoinQuery::new(body("x"), Arc::clone(&p)).into()
            ),
            [0, 1]
        );
        assert_eq!(
            hits(
                s,
                ToChildBlockJoinQuery::new(body("y"), Arc::clone(&p)).into()
            ),
            [4]
        );
        assert_eq!(
            hits(
                s,
                ParentChildrenBlockJoinQuery::new(Arc::clone(&p), body("a"), 2).into()
            ),
            [0, 1]
        );
        // A parent without children, one at doc 0's position, one outside.
        for parent in [3, 0, 99] {
            assert!(hits(
                s,
                ParentChildrenBlockJoinQuery::new(Arc::clone(&p), body("a"), parent).into()
            )
            .is_empty());
        }
        let pcs = ParentsChildrenBlockJoinQuery::new(Arc::clone(&p), body("x"), body("a"), 1)
            .unwrap()
            .with_combiner(ScoreCombiner::Custom {
                name: "zero".into(),
                combine: Arc::new(|_, _| 0.0),
            });
        let top = s.search(&one(pcs.into()), 10).unwrap();
        assert_eq!(top.score_docs.len(), 1, "one child per parent");
        assert_eq!(top.score_docs[0].score, 0.0);
        let count = s
            .count(&one(ParentsChildrenBlockJoinQuery::new(
                Arc::clone(&p),
                body("x"),
                body("a"),
                5,
            )
            .unwrap()
            .into()))
            .unwrap();
        assert_eq!(count, 2);
        // Misuse: a child query matching parents, a parent query children.
        assert!(s
            .search(
                &one(ToParentBlockJoinQuery::new(
                    Clause::MatchAllDocs(MatchAllDocsQuery::new(i32::MAX)),
                    Arc::clone(&p),
                    ScoreMode::Avg
                )
                .into()),
                10
            )
            .is_err());
        assert!(s
            .search(
                &one(ToChildBlockJoinQuery::new(body("a"), Arc::clone(&p)).into()),
                10
            )
            .is_err());
    });
}

#[test]
fn explanations_follow_java() {
    with_searcher("join-explain", |s, _| {
        let p = producer();
        for (mode, best) in [
            (ScoreMode::Max, true),
            (ScoreMode::Min, false),
            (ScoreMode::None, true),
        ] {
            let q = one(ToParentBlockJoinQuery::new(body("a"), Arc::clone(&p), mode).into());
            let score = s.search(&q, 1).unwrap().score_docs[0].score;
            let e = s.explain(&q, 2).unwrap();
            assert!(e.matched);
            assert_eq!(e.value, score, "{mode}");
            assert_eq!(
                e.description,
                format!(
                    "Score based on 2 child docs in range from 0 to 1, using score mode {mode}"
                )
            );
            assert_eq!(e.details.len(), 1);
            let _ = best;
            assert!(!s.explain(&q, 5).unwrap().matched);
        }
        let q = one(ToChildBlockJoinQuery::new(body("x"), Arc::clone(&p)).into());
        let e = s.explain(&q, 1).unwrap();
        assert_eq!(e.description, "Score based on parent document 2");
        assert_eq!(e.details.len(), 1);
        assert!(!s.explain(&q, 4).unwrap().matched);
        let q = one(ParentChildrenBlockJoinQuery::new(Arc::clone(&p), body("a"), 2).into());
        assert!(!s.explain(&q, 0).unwrap().matched);
        let q = one(
            ParentsChildrenBlockJoinQuery::new(Arc::clone(&p), body("x"), body("a"), 5)
                .unwrap()
                .into(),
        );
        let e = s.explain(&q, 1).unwrap();
        assert_eq!(
            e.description,
            "Score based on parent document 2 and child document 1 "
        );
        assert_eq!(e.details.len(), 2);
        assert!(!s.explain(&q, 4).unwrap().matched);
        // Inside a boolean, through the generic path with the segment set.
        let q = BooleanQuery {
            must: vec![
                ToParentBlockJoinQuery::new(body("b"), Arc::clone(&p), ScoreMode::Avg).into(),
            ],
            should: vec![body("y")],
            ..Default::default()
        };
        let top = s.search(&q, 2).unwrap();
        let e = s.explain(&q, top.score_docs[0].doc).unwrap();
        assert!(e.matched);
        assert_eq!(e.value, top.score_docs[0].score);
    });
}

#[test]
fn matches_follow_java() {
    with_searcher("join-matches", |s, _| {
        let p = producer();
        let tp: Clause =
            ToParentBlockJoinQuery::new(body("a"), Arc::clone(&p), ScoreMode::Avg).into();
        assert!(crate::matches::matches(s, &tp, 2).unwrap().is_some());
        assert!(crate::matches::matches(s, &tp, 5).unwrap().is_none());
        // `FilterWeight.matches`: the parent query on the child document.
        let tc: Clause = ToChildBlockJoinQuery::new(body("x"), Arc::clone(&p)).into();
        assert!(crate::matches::matches(s, &tc, 0).unwrap().is_none());
        assert!(crate::matches::matches(s, &tc, 2).unwrap().is_some());
        let pcs: Clause =
            ParentsChildrenBlockJoinQuery::new(Arc::clone(&p), body("x"), body("a"), 5)
                .unwrap()
                .into();
        assert!(crate::matches::matches(s, &pcs, 0).unwrap().is_some());
        assert!(crate::matches::matches(s, &pcs, 4).unwrap().is_none());
    });
}

#[test]
fn parents_sort_by_their_children() {
    with_searcher("join-sort", |s, reader| {
        let p = producer();
        let kids: Arc<dyn BitSetProducer> =
            Arc::new(QueryBitSetProducer::new(term("type", "child")));
        let all_parents = term("type", "parent");
        let sorted = |ty, reverse_children, child_missing| {
            let sf = ToParentBlockJoinSortField::new(
                if ty == JoinSortType::String {
                    "color"
                } else {
                    "price"
                },
                ty,
                false,
                reverse_children,
                None,
                child_missing,
                Arc::clone(&p),
                Arc::clone(&kids),
            )
            .unwrap();
            s.search_sorted(
                reader.segment_readers(),
                &all_parents,
                10,
                &[sf.sort_field()],
                None,
            )
        };
        // p1 has no children: missing (0) sorts first; p0's lowest price 3, p2's 7.
        let top = sorted(JoinSortType::Long, false, None).unwrap();
        assert_eq!(
            top.hits
                .iter()
                .map(|h| (h.doc, h.values[0]))
                .collect::<Vec<_>>(),
            [(3, 0), (2, 3), (5, 7)]
        );
        let top = sorted(JoinSortType::Long, true, None).unwrap();
        assert_eq!(
            top.hits.iter().map(|h| h.values[0]).collect::<Vec<_>>(),
            [0, 5, 7]
        );
        let top = sorted(JoinSortType::String, false, None).unwrap();
        let terms: Vec<Option<Vec<u8>>> = top.hits.iter().map(|h| h.terms[0].clone()).collect();
        assert_eq!(
            terms,
            [None, Some(b"blue".to_vec()), Some(b"green".to_vec())]
        );
        assert!(JoinMissing::Long(1) != JoinMissing::StringLast);
        let err = ToParentBlockJoinSortField::new(
            "price",
            JoinSortType::Int,
            false,
            false,
            Some(JoinMissing::Long(1)),
            None,
            Arc::clone(&p),
            Arc::clone(&kids),
        )
        .unwrap_err();
        assert!(err.to_string().contains("does not fit"));
        // The wrong doc-values type for the sort.
        let wrong = ToParentBlockJoinSortField::new(
            "rank",
            JoinSortType::String,
            false,
            false,
            None,
            None,
            Arc::clone(&p),
            Arc::clone(&kids),
        )
        .unwrap();
        assert!(s
            .search_sorted(
                reader.segment_readers(),
                &all_parents,
                10,
                &[wrong.sort_field()],
                None
            )
            .is_err());
        let wrong = ToParentBlockJoinSortField::new(
            "color",
            JoinSortType::Long,
            false,
            false,
            None,
            None,
            Arc::clone(&p),
            Arc::clone(&kids),
        )
        .unwrap();
        assert!(s
            .search_sorted(
                reader.segment_readers(),
                &all_parents,
                10,
                &[wrong.sort_field()],
                None
            )
            .is_err());
        let absent = ToParentBlockJoinSortField::new(
            "nope",
            JoinSortType::Double,
            false,
            false,
            Some(JoinMissing::Double(1.5)),
            Some(JoinMissing::Double(2.0)),
            Arc::clone(&p),
            Arc::clone(&kids),
        )
        .unwrap();
        assert!(format!("{absent:?}").starts_with("ToParentBlockJoinSortField(nope"));
        let top = s
            .search_sorted(
                reader.segment_readers(),
                &all_parents,
                10,
                &[absent.sort_field()],
                None,
            )
            .unwrap();
        assert_eq!(top.hits.len(), 3);
    });
}

#[test]
fn sort_edge_cases_follow_java() {
    use crate::top_field::{FieldComparatorSource, SortValue};
    with_searcher("join-sort-edges", |s, reader| {
        let p = producer();
        let kids: Arc<dyn BitSetProducer> =
            Arc::new(QueryBitSetProducer::new(term("type", "child")));
        let nobody: Arc<dyn BitSetProducer> =
            Arc::new(QueryBitSetProducer::new(term("type", "zz")));
        let q = term("type", "parent");
        let run = |field: &str, ty, pm, cm, children: &Arc<dyn BitSetProducer>| {
            let sf = ToParentBlockJoinSortField::new(
                field,
                ty,
                false,
                true,
                pm,
                cm,
                Arc::clone(&p),
                Arc::clone(children),
            )
            .unwrap();
            s.search_sorted(reader.segment_readers(), &q, 10, &[sf.sort_field()], None)
        };
        // `rank` is NUMERIC, on parents only: no child has one.
        let top = run(
            "rank",
            JoinSortType::Long,
            Some(JoinMissing::Long(-3)),
            None,
            &kids,
        )
        .unwrap();
        assert!(top.hits.iter().all(|h| h.values[0] == -3));
        // `id` has no doc values at all; neither child filter matches.
        assert_eq!(
            run("id", JoinSortType::Int, None, None, &kids)
                .unwrap()
                .hits
                .len(),
            3
        );
        assert_eq!(
            run("price", JoinSortType::Float, None, None, &nobody)
                .unwrap()
                .hits
                .len(),
            3
        );
        assert_eq!(
            run("nope", JoinSortType::String, None, None, &kids)
                .unwrap()
                .hits
                .len(),
            3
        );
        // Missing children last: p0 and p2 have only valued children, p1 none.
        let top = run(
            "color",
            JoinSortType::String,
            Some(JoinMissing::StringLast),
            Some(JoinMissing::StringLast),
            &kids,
        )
        .unwrap();
        assert_eq!(top.hits.last().map(|h| h.doc), Some(3));
        // A parent block with a document lacking a value (the child filter
        // matches parents too here) takes the child missing ordinal: for
        // `STRING_LAST`, Lucene's `lookupOrd(Integer.MAX_VALUE)` failure.
        let all: Arc<dyn BitSetProducer> = Arc::new(QueryBitSetProducer::new(BooleanQuery {
            should: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(i32::MAX))],
            ..Default::default()
        }));
        let two_level: Arc<dyn BitSetProducer> =
            Arc::new(QueryBitSetProducer::new(term("id", "p2")));
        let sf = ToParentBlockJoinSortField::new(
            "color",
            JoinSortType::String,
            false,
            true,
            None,
            Some(JoinMissing::StringLast),
            Arc::clone(&two_level),
            Arc::clone(&all),
        )
        .unwrap();
        let err = s
            .search_sorted(
                reader.segment_readers(),
                &term("id", "p2"),
                10,
                &[sf.sort_field()],
                None,
            )
            .unwrap_err();
        assert!(err.to_string().contains("out of bounds"), "{err}");
        // `compareValues`.
        let cmp = ToParentBlockJoinSortField::new(
            "color",
            JoinSortType::String,
            false,
            false,
            Some(JoinMissing::StringLast),
            None,
            Arc::clone(&p),
            Arc::clone(&kids),
        )
        .unwrap()
        .new_comparator("color", 1, false);
        let b = |v: Option<&str>| SortValue::Bytes(v.map(|s| s.as_bytes().to_vec()));
        use std::cmp::Ordering::*;
        assert_eq!(cmp.compare_values(&b(None), &b(Some("a"))), Greater);
        assert_eq!(cmp.compare_values(&b(Some("a")), &b(None)), Less);
        assert_eq!(cmp.compare_values(&b(None), &b(None)), Equal);
        assert_eq!(cmp.compare_values(&SortValue::Long(1), &b(None)), Equal);
        let first = ToParentBlockJoinSortField::new(
            "color",
            JoinSortType::String,
            false,
            false,
            None,
            None,
            Arc::clone(&p),
            Arc::clone(&kids),
        )
        .unwrap()
        .new_comparator("color", 1, false);
        assert_eq!(first.compare_values(&b(None), &b(Some("a"))), Less);
        assert_eq!(first.compare_values(&b(Some("a")), &b(None)), Greater);
    });
}

#[test]
fn explanations_of_nothing_are_no_matches() {
    with_searcher("join-explain-none", |s, _| {
        let p = producer();
        let nobody: Arc<dyn BitSetProducer> =
            Arc::new(QueryBitSetProducer::new(term("type", "zz")));
        for q in [
            Clause::from(ToParentBlockJoinQuery::new(
                body("zzz"),
                Arc::clone(&p),
                ScoreMode::Avg,
            )),
            ToParentBlockJoinQuery::new(body("a"), Arc::clone(&nobody), ScoreMode::Avg).into(),
            ToChildBlockJoinQuery::new(body("zzz"), Arc::clone(&p)).into(),
            ToChildBlockJoinQuery::new(body("x"), Arc::clone(&nobody)).into(),
            ParentsChildrenBlockJoinQuery::new(Arc::clone(&p), body("zzz"), body("a"), 1)
                .unwrap()
                .into(),
        ] {
            let e = s.explain(&one(q.clone()), 0).unwrap();
            assert!(!e.matched, "{q:?}");
            assert!(s.search(&one(q), 10).unwrap().score_docs.is_empty());
        }
    });
}

/// `SegmentReader::with_open_segment` before any search validated the
/// postings, over a segment with points: the inputs it opens answer a query.
#[test]
fn a_reader_opens_one_segment_for_a_producer() {
    let tmp = TempDir::new("join-open-one");
    let dir = FsDirectory::open(tmp.path());
    {
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let mut d = doc("p0", "parent", 0);
        d.add(lucene_index::document::LongPoint::new("n", &[7]).unwrap());
        w.add_fields_document(&d).unwrap();
        w.commit().unwrap();
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    let r = &reader.segment_readers()[0];
    let n = r
        .with_open_segment(0, |seg| {
            assert!(seg.points.is_some());
            Ok(parents().bit_set(seg)?.map(|b| b.cardinality()))
        })
        .unwrap();
    assert_eq!(n, Some(1));
    // Again, now that the points are parsed.
    r.with_open_segment(0, |seg| Ok(seg.max_doc)).unwrap();
}
