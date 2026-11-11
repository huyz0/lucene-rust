//! The bounded search keeps exactly the hits the ordinary one keeps, over
//! an index of several segments built to defeat it: ties, documents without
//! a value, several values a document, a segment in no time order, a skip
//! index without points, an `INT` key, deletions, filters that stop
//! matching before the newest documents, and paging.

use std::cell::Cell;

use lucene_index::buffered_updates::Term;
use lucene_index::document::{self as d, Document, IndexableField, Store};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

use crate::collector::TotalHitsRelation;
use crate::directory_reader::DirectoryReader;
use crate::query::{BooleanQuery, Clause, MatchAllDocsQuery, PointsRangeQuery, TermQuery};
use crate::top_field::{search_sorted, FieldDoc, Selector, SortField, SortType, TopFieldDocs};

thread_local! {
    /// Turns the bounded search off, for the ordinary search to compare with.
    pub(crate) static DISABLED: Cell<bool> = const { Cell::new(false) };
    /// The slices the bounded search ran.
    pub(crate) static SLICES: Cell<u32> = const { Cell::new(0) };
}

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn b(f: impl IndexableField + 'static) -> Box<dyn IndexableField> {
    Box::new(f)
}

/// A small linear congruential generator: the same index every run.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, below: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % below
    }
}

/// Six segments: time-ordered ones, one shuffled, one of a single document.
/// `t` has points and doc values, `s` doc values with a skip index, `i` an
/// `INT` key's points and doc values, `m` two values. One in seven documents
/// has no `t`; every fifth of the first three segments is deleted.
fn index(tmp: &TempDir) -> DirectoryReader {
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let mut rng = Lcg(42);
    let mut g = 0i64;
    for (seg, size) in [3000, 2500, 1, 4000, 2000, 600].into_iter().enumerate() {
        for _ in 0..size {
            let t = if seg == 3 {
                i64::try_from(rng.next(6000)).unwrap()
            } else {
                g / 3
            };
            let mut doc = Document::new();
            let mut fields: Vec<Box<dyn IndexableField>> = Vec::new();
            if g % 7 != 3 {
                fields.push(b(d::LongPoint::new("t", &[t]).unwrap()));
                fields.push(b(d::NumericDocValuesField::new("t", t)));
                fields.push(b(d::NumericDocValuesField::indexed_field("s", t)));
                let other = t + i64::try_from(rng.next(5000)).unwrap() - 2500;
                fields.push(b(d::LongField::new("m", t, Store::No)));
                fields.push(b(d::LongField::new("m", other, Store::No)));
                let i = i32::try_from(-t / 2).unwrap();
                fields.push(b(d::IntPoint::new("i", &[i]).unwrap()));
                fields.push(b(d::NumericDocValuesField::new("i", i64::from(i))));
            }
            let c = ["a", "b", "c"][usize::try_from(rng.next(3)).unwrap()];
            fields.push(b(d::StringField::new("c", c, Store::No)));
            // Only in the older documents: the newest never match.
            if g % 97 == 0 && g < 4000 {
                fields.push(b(d::StringField::new("c", "rare", Store::No)));
            }
            if g % 5 == 0 && seg < 3 {
                fields.push(b(d::StringField::new("del", "x", Store::No)));
            }
            for f in fields {
                doc.add_boxed(f);
            }
            w.add_fields_document(&doc).unwrap();
            g += 1;
        }
        w.commit().unwrap();
    }
    w.delete_documents_by_term(&[Term::new("del", "x")])
        .unwrap();
    w.commit().unwrap();
    drop(w);
    DirectoryReader::open(&dir).unwrap()
}

fn term(f: &str, v: &str) -> Clause {
    Clause::Term(TermQuery::new(f, v.as_bytes().to_vec()))
}

fn queries() -> Vec<(&'static str, BooleanQuery)> {
    let all = || Clause::MatchAllDocs(MatchAllDocsQuery::new(0));
    let range = || Clause::PointsRange(PointsRangeQuery::new("t", 900, 2600));
    vec![
        (
            "all",
            BooleanQuery {
                must: vec![all()],
                ..Default::default()
            },
        ),
        (
            "c=a",
            BooleanQuery {
                filter: vec![term("c", "a")],
                ..Default::default()
            },
        ),
        (
            "rare",
            BooleanQuery {
                filter: vec![term("c", "rare")],
                ..Default::default()
            },
        ),
        (
            "range",
            BooleanQuery {
                filter: vec![range()],
                ..Default::default()
            },
        ),
        (
            "range&b",
            BooleanQuery {
                filter: vec![range(), term("c", "b")],
                ..Default::default()
            },
        ),
        (
            "a|rare",
            BooleanQuery {
                should: vec![term("c", "a"), term("c", "rare")],
                ..Default::default()
            },
        ),
        (
            "all-c",
            BooleanQuery {
                must: vec![all()],
                must_not: vec![term("c", "c")],
                ..Default::default()
            },
        ),
        (
            "nothing",
            BooleanQuery {
                filter: vec![term("c", "nope")],
                ..Default::default()
            },
        ),
    ]
}

fn key(field: &str, ty: SortType, reverse: bool, missing: i64, selector: Selector) -> SortField {
    let mut k = SortField::numeric(field, ty, reverse);
    k.missing = missing;
    k.selector = selector;
    k
}

fn sorts() -> Vec<SortField> {
    let mut out = Vec::new();
    for reverse in [true, false] {
        for missing in [0, i64::MIN, i64::MAX] {
            out.push(key("t", SortType::Long, reverse, missing, Selector::Min));
        }
        for missing in [i64::MIN, i64::MAX] {
            out.push(key("s", SortType::Long, reverse, missing, Selector::Min));
            out.push(key(
                "i",
                SortType::Int,
                reverse,
                if missing < 0 {
                    i64::from(i32::MIN)
                } else {
                    i64::from(i32::MAX)
                },
                Selector::Min,
            ));
        }
        for selector in [Selector::Min, Selector::Max, Selector::Sum] {
            out.push(key("m", SortType::Long, reverse, i64::MIN, selector));
        }
    }
    out
}

fn run(
    reader: &DirectoryReader,
    q: &BooleanQuery,
    sort: &SortField,
    n: usize,
    threshold: u64,
    after: Option<&FieldDoc>,
    ordinary: bool,
) -> TopFieldDocs {
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let norms = vec![None; segments.len()];
    DISABLED.with(|d| d.set(ordinary));
    let top = search_sorted(
        &segments,
        reader.segment_readers(),
        q,
        &norms,
        std::slice::from_ref(sort),
        n,
        threshold,
        after,
    );
    DISABLED.with(|d| d.set(false));
    top.unwrap()
}

#[test]
fn the_bounded_search_keeps_the_ordinary_hits() {
    let tmp = TempDir::new("newest-n");
    let reader = index(&tmp);
    let mut bounded = 0u32;
    for (name, q) in queries() {
        for sort in sorts() {
            for n in [1usize, 7, 100, 1500] {
                let exact = run(&reader, &q, &sort, n, u64::MAX, None, true);
                assert_eq!(exact.total.relation, TotalHitsRelation::EqualTo);
                let mut afters = vec![None];
                if let Some(h) = exact.hits.get(n / 2) {
                    afters.push(Some(FieldDoc {
                        doc: h.doc,
                        values: h.values.clone(),
                        terms: Vec::new(),
                    }));
                }
                for after in &afters {
                    for threshold in [0, 100, u64::MAX] {
                        let ctx =
                            format!("{name} {sort:?} n={n} after={after:?} threshold={threshold}");
                        let want = run(&reader, &q, &sort, n, threshold, after.as_ref(), true);
                        SLICES.with(|c| c.set(0));
                        let got = run(&reader, &q, &sort, n, threshold, after.as_ref(), false);
                        bounded = bounded.saturating_add(SLICES.with(Cell::get).min(1));
                        assert_eq!(got.hits, want.hits, "{ctx}");
                        match got.total.relation {
                            TotalHitsRelation::EqualTo => {
                                assert_eq!(got.total.value, exact.total.value, "{ctx}");
                            }
                            TotalHitsRelation::GreaterThanOrEqualTo => {
                                assert!(threshold != u64::MAX, "{ctx}");
                                assert!(got.total.value > threshold.max(n as u64), "{ctx}");
                                assert!(got.total.value <= exact.total.value, "{ctx}");
                            }
                        }
                    }
                }
            }
        }
    }
    // Most shapes take the bounded search; the rest are the ones it refuses.
    assert!(bounded > 1000, "bounded searches: {bounded}");
}

#[test]
fn newest_hits_take_one_slice_over_time_ordered_segments() {
    let tmp = TempDir::new("newest-n-one");
    let reader = index(&tmp);
    let (_, q) = queries().swap_remove(0);
    let sort = key("t", SortType::Long, true, i64::MIN, Selector::Min);
    SLICES.with(|c| c.set(0));
    let top = run(&reader, &q, &sort, 10, 1000, None, false);
    assert_eq!(SLICES.with(Cell::get), 1);
    assert_eq!(top.hits.len(), 10);
    assert_eq!(top.total.relation, TotalHitsRelation::GreaterThanOrEqualTo);
    // An ascending key whose missing value sorts first is the ordinary
    // search's: the range would leave those documents out.
    SLICES.with(|c| c.set(0));
    let asc = key("t", SortType::Long, false, 0, Selector::Min);
    let _ = run(&reader, &q, &asc, 10, 1000, None, false);
    assert_eq!(SLICES.with(Cell::get), 0);
}
