//! `AssertingLeafReader`'s points: walks over a real index pass through the
//! checks with the answers of the unwrapped reader, and every check fires on
//! a faulty inner `PointValues`.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use lucene_index::document::{Document, IntPoint};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::{IntersectVisitor, LeafReader, PointValues, Relation};
use lucene_search::Result;
use lucene_store::ByteBuffersDirectory;
use lucene_test_framework::index::asserting_leaf_reader::{
    AssertingIntersectVisitor, AssertingLeafReader, AssertingPointValues,
};
use lucene_util::java_random::JavaRandom;

const DOCS: i32 = 3000;

fn index(dir: &ByteBuffersDirectory) {
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", version).unwrap();
    let mut r = JavaRandom::new(3);
    for _ in 0..DOCS {
        let mut d = Document::new();
        // Duplicate values, so equal values arrive with ascending doc ids.
        d.add(IntPoint::new("one", &[r.next_int_bounded(500) - 250]).unwrap());
        let (x, y) = (r.next_int_bounded(1000), r.next_int_bounded(1000));
        d.add(IntPoint::new("two", &[x, y]).unwrap());
        if r.next_int_bounded(3) == 0 {
            d.add(IntPoint::new("one", &[r.next_int_bounded(500) - 250]).unwrap());
        }
        w.add_fields_document(&d).unwrap();
    }
    w.commit().unwrap();
    w.close().unwrap();
}

fn encode(v: i32) -> [u8; 4] {
    (v as u32 ^ 0x8000_0000).to_be_bytes()
}

/// A box query over `IntPoint`s: every document with a point in
/// `[lower, upper]` in each dimension.
struct BoxVisitor {
    lower: Vec<[u8; 4]>,
    upper: Vec<[u8; 4]>,
    docs: Vec<i32>,
}

impl BoxVisitor {
    fn new(lower: &[i32], upper: &[i32]) -> Self {
        Self {
            lower: lower.iter().map(|&v| encode(v)).collect(),
            upper: upper.iter().map(|&v| encode(v)).collect(),
            docs: Vec::new(),
        }
    }

    fn dims(&self) -> usize {
        self.lower.len()
    }
}

impl IntersectVisitor for BoxVisitor {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> Relation {
        let mut crosses = false;
        for d in 0..self.dims() {
            let (lo, hi) = (&min[d * 4..d * 4 + 4], &max[d * 4..d * 4 + 4]);
            if hi < &self.lower[d][..] || lo > &self.upper[d][..] {
                return Relation::CellOutsideQuery;
            }
            crosses |= lo < &self.lower[d][..] || hi > &self.upper[d][..];
        }
        if crosses {
            Relation::CellCrossesQuery
        } else {
            Relation::CellInsideQuery
        }
    }
    fn visit(&mut self, doc_id: i32) {
        self.docs.push(doc_id);
    }
    fn visit_with_value(&mut self, doc_id: i32, packed: &[u8]) {
        if (0..self.dims()).all(|d| {
            let v = &packed[d * 4..d * 4 + 4];
            v >= &self.lower[d][..] && v <= &self.upper[d][..]
        }) {
            self.docs.push(doc_id);
        }
    }
}

fn run(values: &dyn PointValues, lower: &[i32], upper: &[i32]) -> (Vec<i32>, i64) {
    let mut v = BoxVisitor::new(lower, upper);
    values.intersect(&mut v).unwrap();
    v.docs.sort_unstable();
    let mut e = BoxVisitor::new(lower, upper);
    let estimate = values.estimate_point_count(&mut e).unwrap();
    (v.docs, estimate)
}

#[test]
fn walks_answer_as_the_unwrapped_reader() {
    let dir = ByteBuffersDirectory::new();
    index(&dir);
    let reader = DirectoryReader::open(&dir).unwrap();
    let leaf: Arc<dyn LeafReader> = Arc::new(reader.segment_readers()[0].clone());
    let asserting = AssertingLeafReader::new(leaf.clone());
    assert!(asserting.point_values("absent").unwrap().is_none());
    let mut r = JavaRandom::new(11);
    for (field, dims, span) in [("one", 1, 500), ("two", 2, 1000)] {
        let plain = leaf.point_values(field).unwrap().unwrap();
        let checked = asserting.point_values(field).unwrap().unwrap();
        assert_eq!(checked.size(), plain.size());
        assert_eq!(checked.doc_count(), plain.doc_count());
        assert_eq!(checked.num_dimensions(), dims);
        assert_eq!(checked.num_index_dimensions(), plain.num_index_dimensions());
        assert_eq!(checked.bytes_per_dimension(), 4);
        assert_eq!(checked.min_packed_value(), plain.min_packed_value());
        assert_eq!(checked.max_packed_value(), plain.max_packed_value());
        // Everything, nothing, and random boxes.
        let base = if dims == 1 { -250 } else { 0 };
        let all = (vec![i32::MIN; dims as usize], vec![i32::MAX; dims as usize]);
        let none = (vec![i32::MAX; dims as usize], vec![i32::MAX; dims as usize]);
        let mut boxes = vec![all, none];
        for _ in 0..40 {
            let (mut lo, mut hi) = (Vec::new(), Vec::new());
            for _ in 0..dims {
                let a = base + r.next_int_bounded(span);
                let b = a + r.next_int_bounded(span / 4 + 1);
                lo.push(a);
                hi.push(b);
            }
            boxes.push((lo, hi));
        }
        for (lo, hi) in &boxes {
            let want = run(plain.as_ref(), lo, hi);
            assert_eq!(
                run(checked.as_ref(), lo, hi),
                want,
                "{field} {lo:?}..{hi:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// A faulty inner PointValues
// ---------------------------------------------------------------------------

type Walk = fn(&mut dyn IntersectVisitor);

struct Fake {
    dims: i32,
    size: i64,
    doc_count: i32,
    walk: Walk,
}

impl Fake {
    fn new(dims: i32, walk: Walk) -> Self {
        Self {
            dims,
            size: 4,
            doc_count: 2,
            walk,
        }
    }
}

impl PointValues for Fake {
    fn intersect(&self, visitor: &mut dyn IntersectVisitor) -> Result<()> {
        (self.walk)(visitor);
        Ok(())
    }
    fn estimate_point_count(&self, visitor: &mut dyn IntersectVisitor) -> Result<i64> {
        (self.walk)(visitor);
        Ok(self.size)
    }
    fn min_packed_value(&self) -> &[u8] {
        &[0; 2]
    }
    fn max_packed_value(&self) -> &[u8] {
        &[9; 2]
    }
    fn num_dimensions(&self) -> i32 {
        self.dims
    }
    fn num_index_dimensions(&self) -> i32 {
        self.dims
    }
    fn bytes_per_dimension(&self) -> i32 {
        1
    }
    fn size(&self) -> i64 {
        self.size
    }
    fn doc_count(&self) -> i32 {
        self.doc_count
    }
}

/// Accepts every cell as crossing and every callback.
struct Sink;

impl IntersectVisitor for Sink {
    fn compare(&mut self, _: &[u8], _: &[u8]) -> Relation {
        Relation::CellCrossesQuery
    }
    fn visit(&mut self, _: i32) {}
    fn visit_with_value(&mut self, _: i32, _: &[u8]) {}
}

/// Accepts every cell as inside.
struct Inside;

impl IntersectVisitor for Inside {
    fn compare(&mut self, _: &[u8], _: &[u8]) -> Relation {
        Relation::CellInsideQuery
    }
    fn visit(&mut self, _: i32) {}
    fn visit_with_value(&mut self, _: i32, _: &[u8]) {}
}

#[track_caller]
fn fires(expected: &str, f: impl FnOnce()) {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("the check did not fire");
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap();
    assert!(msg.contains(expected), "{msg:?} lacks {expected:?}");
}

fn wrap(fake: Fake) -> AssertingPointValues<'static> {
    AssertingPointValues::new(Box::new(fake), 10)
}

fn walk(dims: i32, sink: &mut dyn IntersectVisitor, w: Walk) {
    wrap(Fake::new(dims, w)).intersect(sink).unwrap();
}

#[test]
fn faulty_statistics_fire() {
    let ok = |_: &mut dyn IntersectVisitor| {};
    let mut f = Fake::new(1, ok);
    f.size = 0;
    fires("size=0", || drop(wrap(f)));
    let mut f = Fake::new(1, ok);
    f.doc_count = 0;
    fires("docCount=0", || drop(wrap(f)));
    let mut f = Fake::new(1, ok);
    f.doc_count = 5;
    fires("> size=4", || drop(wrap(f)));
    let mut f = Fake::new(1, ok);
    (f.size, f.doc_count) = (40, 11);
    fires("> maxDoc=10", || drop(wrap(f)));
    let w = wrap(Fake::new(2, ok));
    assert_eq!(w.wrapped().num_dimensions(), 2);
    assert_eq!(w.min_packed_value(), &[0, 0]);
    assert_eq!(w.max_packed_value(), &[9, 9]);
    assert_eq!(w.estimate_point_count(&mut Sink).unwrap(), 4);
}

#[test]
fn a_well_behaved_fake_walk_passes() {
    walk(1, &mut Sink, |v| {
        v.grow(4);
        v.visit_with_value(3, &[1]);
        v.visit_with_value(5, &[1]);
        v.visit_with_value(0, &[2]);
        v.visit(7);
        assert_eq!(v.compare(&[0], &[4]), Relation::CellCrossesQuery);
        v.grow(2);
        v.visit_with_value(1, &[3]);
        // Only the first value after a compare must lie in its cell.
        v.visit_with_value(1, &[8]);
    });
    walk(2, &mut Inside, |v| {
        assert_eq!(v.compare(&[0, 0], &[4, 4]), Relation::CellInsideQuery);
        v.grow(3);
        v.visit(2);
        v.visit_many(&[4, 6]);
    });
}

#[test]
fn faulty_walks_fire() {
    fires("more times than the last call to grow()", || {
        walk(1, &mut Sink, |v| v.visit(1))
    });
    fires("more times than the last call to grow()", || {
        walk(1, &mut Sink, |v| {
            v.grow(1);
            v.visit_with_value(1, &[1]);
            v.visit_with_value(2, &[1]);
        })
    });
    fires("more times than the last call to grow()", || {
        walk(1, &mut Inside, |v| {
            v.grow(1);
            v.visit_many(&[1, 2]);
        })
    });
    fires("the cell's min is above its max", || {
        walk(2, &mut Sink, |v| {
            v.compare(&[0, 5], &[4, 4]);
        })
    });
    fires(
        "visit(docID) after compare returned Some(CellCrossesQuery)",
        || {
            walk(1, &mut Sink, |v| {
                v.compare(&[0], &[4]);
                v.grow(1);
                v.visit(1);
            })
        },
    );
    fires(
        "visit(docID) after compare returned Some(CellCrossesQuery)",
        || {
            walk(1, &mut Sink, |v| {
                v.compare(&[0], &[4]);
                v.grow(2);
                v.visit_many(&[1, 2]);
            })
        },
    );
    fires(
        "visit(docID, packedValue) after compare returned Some(CellInsideQuery)",
        || {
            walk(1, &mut Inside, |v| {
                v.compare(&[0], &[4]);
                v.grow(1);
                v.visit_with_value(1, &[1]);
            })
        },
    );
    fires("dim=1 of 2", || {
        walk(2, &mut Sink, |v| {
            v.compare(&[0, 2], &[4, 4]);
            v.grow(1);
            v.visit_with_value(1, &[1, 5]);
        })
    });
    fires("dim=0 of 1", || {
        walk(1, &mut Sink, |v| {
            v.compare(&[2], &[4]);
            v.grow(1);
            v.visit_with_value(1, &[1]);
        })
    });
    fires("packed value length", || {
        walk(2, &mut Sink, |v| {
            v.grow(1);
            v.visit_with_value(1, &[1]);
        })
    });
    fires(
        "doc ids are out of order when point values are the same!",
        || {
            walk(1, &mut Sink, |v| {
                v.grow(2);
                v.visit_with_value(4, &[1]);
                v.visit_with_value(3, &[1]);
            })
        },
    );
    fires("point values are out of order", || {
        walk(1, &mut Sink, |v| {
            v.grow(2);
            v.visit_with_value(4, &[2]);
            v.visit_with_value(5, &[1]);
        })
    });
    // Built directly, over any visitor.
    let mut sink = Sink;
    let mut v = AssertingIntersectVisitor::new(1, 1, 1, &mut sink);
    v.grow(1);
    v.visit_with_value(0, &[0]);
}
