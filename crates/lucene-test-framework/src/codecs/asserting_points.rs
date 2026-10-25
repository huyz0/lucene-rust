//! Java's `AssertingPointsFormat` and `AssertingLeafReader` point wrappers.
//! Assertions stay enabled in release, including statistics, acquisition-thread,
//! tree-size and visitor budget/relation/bounds/one-dimensional ordering checks.
//! Writer merge delegates directly; cloned trees remain wrapped.
//!
//! This explicit format seam reuses cranky_points' fallible visitors and native
//! Lucene90 byte reader/writer without randomness. Read state supplies maxDoc
//! separately, as Java SegmentInfo does. A narrow lifecycle extension adds merge
//! acquisition and Display. Native merge handles alias Java's inherited identity
//! instance: closing either handle makes subsequent values acquisition return None.
//! Acquired metadata and existing ByteBuffers-like trees survive; new tree acquisition
//! and integrity return typed closed errors instead of AlreadyClosedException.
//! Immutable parsed metadata remains retained until all handles drop, unlike Java's
//! cleared reader map. Native Display uses a stable label instead
//! of Java's default identity string. Native min/max metadata getters return borrowed
//! slices where Java clones byte arrays; callers cannot mutate them. Nullable
//! arrays/delegates are unrepresentable;
//! grow uses usize instead of signed Java counts. The underlying native buffered
//! writer retains publication/sticky-write-error timing and accounting limits.
//! IOContext, Codec/IndexWriter routing and invalidating input backends are absent.

use super::cranky_points::{self as byte, PointsFormat as ByteFormat};
use byte::PointsReader as ByteReader;
pub use byte::{
    Error, IntersectVisitor, MergeSource, MergeState, PointTree, PointValues, PointsWriter, Result,
    WriteState,
};
use lucene_codecs::field_infos::{FieldInfo, FieldInfos};
use lucene_codecs::points::Relation;
use std::fmt;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread::{self, ThreadId};

pub trait PointsReader: ByteReader + fmt::Display {
    fn get_merge_instance(&self) -> Self
    where
        Self: Sized;
}
#[derive(Clone, Copy)]
pub struct ReadState<'a> {
    pub bytes: byte::ReadState<'a>,
    pub max_doc: i32,
}
pub trait PointsFormat {
    type Writer<'a>: PointsWriter;
    type Reader<'a>: PointsReader;
    fn fields_writer<'a>(&mut self, state: WriteState<'a>) -> Result<Self::Writer<'a>>;
    fn fields_reader<'a>(&mut self, state: ReadState<'a>) -> Result<Self::Reader<'a>>;
}
pub struct AssertingPointsFormat<F> {
    delegate: F,
}
impl<F> AssertingPointsFormat<F> {
    pub fn new(delegate: F) -> Self {
        Self { delegate }
    }
    pub fn delegate(&self) -> &F {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut F {
        &mut self.delegate
    }
}
impl<F: PointsFormat> PointsFormat for AssertingPointsFormat<F> {
    type Writer<'a> = AssertingPointsWriter<F::Writer<'a>>;
    type Reader<'a> = AssertingPointsReader<'a, F::Reader<'a>>;
    fn fields_writer<'a>(&mut self, state: WriteState<'a>) -> Result<Self::Writer<'a>> {
        Ok(AssertingPointsWriter::new(
            self.delegate.fields_writer(state)?,
        ))
    }
    fn fields_reader<'a>(&mut self, state: ReadState<'a>) -> Result<Self::Reader<'a>> {
        Ok(AssertingPointsReader::new(
            self.delegate.fields_reader(state)?,
            state.max_doc,
            state.bytes.fields,
            false,
        ))
    }
}
pub struct AssertingPointsWriter<W> {
    delegate: W,
}
impl<W> AssertingPointsWriter<W> {
    pub fn new(delegate: W) -> Self {
        Self { delegate }
    }
    pub fn delegate(&self) -> &W {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut W {
        &mut self.delegate
    }
}
impl<W: PointsWriter> PointsWriter for AssertingPointsWriter<W> {
    /// Java: `AssertingPointsWriter.writeField`.
    fn write_field<R: ByteReader>(&mut self, field: &FieldInfo, reader: &R) -> Result<()> {
        if field.point_dimension_count == 0 {
            return Err(Error::Message(format!(
                "writing field=\"{}\" but pointDimensionalCount is 0",
                field.name
            )));
        }
        self.delegate.write_field(field, reader)
    }
    fn merge<R: ByteReader>(&mut self, state: &MergeState<'_, R>) -> Result<()> {
        self.delegate.merge(state)
    }
    fn finish(&mut self) -> Result<()> {
        self.delegate.finish()
    }
    fn close(&mut self) -> Result<()> {
        self.delegate.close()?;
        self.delegate.close()
    }
}
fn assert_thread(kind: &str, creation: ThreadId) {
    assert_eq!(
        creation,
        thread::current().id(),
        "{kind} are only supposed to be consumed in the thread in which they have been acquired."
    );
}
pub struct AssertingPointsReader<'a, R> {
    delegate: R,
    max_doc: i32,
    fields: &'a FieldInfos,
    merging: bool,
    creation_thread: ThreadId,
}
impl<'a, R: PointsReader> AssertingPointsReader<'a, R> {
    /// Java: `AssertingPointsReader(maxDoc,in,fis,merging)`.
    pub fn new(delegate: R, max_doc: i32, fields: &'a FieldInfos, merging: bool) -> Self {
        let reader = Self {
            delegate,
            max_doc,
            fields,
            merging,
            creation_thread: thread::current().id(),
        };
        let _ = reader.to_string();
        reader
    }
    pub fn delegate(&self) -> &R {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut R {
        &mut self.delegate
    }
}
impl<R: PointsReader> ByteReader for AssertingPointsReader<'_, R> {
    type Values<'a>
        = AssertingPointValues<R::Values<'a>>
    where
        Self: 'a;
    /// Java: `AssertingPointsReader.getValues`, field check before merge-thread check.
    fn values(&self, name: &str) -> Result<Option<Self::Values<'_>>> {
        assert!(self
            .fields
            .field_by_name(name)
            .is_some_and(|f| f.point_dimension_count > 0));
        if self.merging {
            assert_thread("PointsReader", self.creation_thread);
        }
        self.delegate
            .values(name)?
            .map(|v| AssertingPointValues::new(v, self.max_doc))
            .transpose()
    }
    fn check_integrity(&self) -> Result<()> {
        self.delegate.check_integrity()
    }
    fn close(&mut self) -> Result<()> {
        self.delegate.close()?;
        self.delegate.close()
    }
}
impl<R: PointsReader> PointsReader for AssertingPointsReader<'_, R> {
    fn get_merge_instance(&self) -> Self {
        Self::new(
            self.delegate.get_merge_instance(),
            self.max_doc,
            self.fields,
            true,
        )
    }
}
impl<R: fmt::Display> fmt::Display for AssertingPointsReader<'_, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AssertingPointsReader({})", self.delegate)
    }
}
pub struct AssertingPointValues<V> {
    delegate: V,
    creation_thread: ThreadId,
}
impl<V: PointValues> AssertingPointValues<V> {
    /// Java: `AssertingPointValues` / `assertStats`, including repeated callbacks.
    pub fn new(delegate: V, max_doc: i32) -> Result<Self> {
        assert!(delegate.size() > 0);
        assert!(delegate.doc_count() > 0);
        assert!(i64::from(delegate.doc_count()) <= delegate.size());
        assert!(delegate.doc_count() <= max_doc);
        Ok(Self {
            delegate,
            creation_thread: thread::current().id(),
        })
    }
    pub fn wrapped(&self) -> &V {
        &self.delegate
    }
    fn check_thread(&self) {
        assert_thread("Points", self.creation_thread);
    }
}
impl<V: PointValues> PointValues for AssertingPointValues<V> {
    type Tree<'a>
        = AssertingPointTree<'a, V, V::Tree<'a>>
    where
        Self: 'a;
    fn point_tree(&self) -> Result<Self::Tree<'_>> {
        self.check_thread();
        Ok(AssertingPointTree {
            values: &self.delegate,
            delegate: self.delegate.point_tree()?,
        })
    }
    fn min_packed_value(&self) -> Result<&[u8]> {
        self.check_thread();
        self.delegate.min_packed_value()
    }
    fn max_packed_value(&self) -> Result<&[u8]> {
        self.check_thread();
        self.delegate.max_packed_value()
    }
    fn num_dimensions(&self) -> Result<i32> {
        self.check_thread();
        self.delegate.num_dimensions()
    }
    fn num_index_dimensions(&self) -> Result<i32> {
        self.check_thread();
        self.delegate.num_index_dimensions()
    }
    fn bytes_per_dimension(&self) -> Result<i32> {
        self.check_thread();
        self.delegate.bytes_per_dimension()
    }
    fn size(&self) -> i64 {
        self.check_thread();
        self.delegate.size()
    }
    fn doc_count(&self) -> i32 {
        self.check_thread();
        self.delegate.doc_count()
    }
}
pub struct AssertingPointTree<'a, V, T> {
    values: &'a V,
    delegate: T,
}
impl<'a, V: PointValues, T: PointTree> PointTree for AssertingPointTree<'a, V, T> {
    type CloneTree = AssertingPointTree<'a, V, T::CloneTree>;
    fn clone_tree(&self) -> Self::CloneTree {
        AssertingPointTree {
            values: self.values,
            delegate: self.delegate.clone_tree(),
        }
    }
    fn move_to_child(&mut self) -> Result<bool> {
        self.delegate.move_to_child()
    }
    fn move_to_sibling(&mut self) -> Result<bool> {
        self.delegate.move_to_sibling()
    }
    fn move_to_parent(&mut self) -> Result<bool> {
        self.delegate.move_to_parent()
    }
    fn min_packed_value(&self) -> &[u8] {
        self.delegate.min_packed_value()
    }
    fn max_packed_value(&self) -> &[u8] {
        self.delegate.max_packed_value()
    }
    fn size(&self) -> i64 {
        let size = self.delegate.size();
        assert!(size > 0);
        size
    }
    fn visit_doc_ids<I: IntersectVisitor>(&mut self, visitor: &mut I) -> Result<()> {
        let mut checked = AssertingIntersectVisitor::new(
            self.values.num_dimensions()?,
            self.values.num_index_dimensions()?,
            self.values.bytes_per_dimension()?,
            visitor,
        );
        self.delegate.visit_doc_ids(&mut checked)
    }
    fn visit_doc_values<I: IntersectVisitor>(&mut self, visitor: &mut I) -> Result<()> {
        let mut checked = AssertingIntersectVisitor::new(
            self.values.num_dimensions()?,
            self.values.num_index_dimensions()?,
            self.values.bytes_per_dimension()?,
            visitor,
        );
        self.delegate.visit_doc_values(&mut checked)
    }
}
/// Java: `AssertingLeafReader.AssertingIntersectVisitor`.
/// The last compare applies only to the first subsequent value callback.
pub struct AssertingIntersectVisitor<'a, I> {
    delegate: &'a mut I,
    num_data_dims: usize,
    num_index_dims: usize,
    bytes_per_dim: usize,
    last_value: Option<Vec<u8>>,
    last_min: Vec<u8>,
    last_max: Vec<u8>,
    last_compare: Option<Relation>,
    last_doc: i32,
    budget: i32,
}
impl<'a, I: IntersectVisitor> AssertingIntersectVisitor<'a, I> {
    pub fn new(dims: i32, index_dims: i32, width: i32, delegate: &'a mut I) -> Self {
        let dims = usize::try_from(dims).expect("negative dimensions");
        let index_dims = usize::try_from(index_dims).expect("negative index dimensions");
        let width = usize::try_from(width).expect("negative bytes per dimension");
        let length = dims.checked_mul(width).expect("packed width overflow");
        Self {
            delegate,
            num_data_dims: dims,
            num_index_dims: index_dims,
            bytes_per_dim: width,
            last_value: (dims == 1).then(|| vec![0; width]),
            last_min: vec![0; length],
            last_max: vec![0; length],
            last_compare: None,
            last_doc: -1,
            budget: 0,
        }
    }
}
impl<I: IntersectVisitor> IntersectVisitor for AssertingIntersectVisitor<'_, I> {
    fn visit(&mut self, doc: i32) -> Result<()> {
        self.budget = self.budget.wrapping_sub(1);
        assert!(
            self.budget >= 0,
            "called add() more times than the last call to grow() reserved"
        );
        assert!(self
            .last_compare
            .is_none_or(|r| r == Relation::CellInsideQuery));
        self.delegate.visit(doc)
    }
    fn visit_value(&mut self, doc: i32, value: &[u8]) -> Result<()> {
        self.budget = self.budget.wrapping_sub(1);
        assert!(
            self.budget >= 0,
            "called add() more times than the last call to grow() reserved"
        );
        assert!(self
            .last_compare
            .is_none_or(|r| r == Relation::CellCrossesQuery));
        if self.last_compare.is_some() {
            for dim in 0..self.num_index_dims {
                let start = dim * self.bytes_per_dim;
                let end = start + self.bytes_per_dim;
                assert!(self.last_min[start..end] <= value[start..end]);
                assert!(self.last_max[start..end] >= value[start..end]);
            }
            self.last_compare = None;
        }
        assert_eq!(value.len(), self.num_data_dims * self.bytes_per_dim);
        if let Some(previous) = self.last_value.as_mut() {
            match previous.as_slice().cmp(value) {
                std::cmp::Ordering::Less => (),
                std::cmp::Ordering::Equal => assert!(
                    self.last_doc <= doc,
                    "doc ids are out of order when point values are the same!"
                ),
                std::cmp::Ordering::Greater => panic!("point values are out of order"),
            }
            previous.copy_from_slice(value);
            self.last_doc = doc;
        }
        self.delegate.visit_value(doc, value)
    }
    fn grow(&mut self, count: usize) {
        self.delegate.grow(count);
        self.budget = count as i32;
    }
    fn compare(&mut self, min: &[u8], max: &[u8]) -> Relation {
        for dim in 0..self.num_index_dims {
            let start = dim * self.bytes_per_dim;
            let end = start + self.bytes_per_dim;
            assert!(min[start..end] <= max[start..end]);
        }
        let len = self.num_index_dims * self.bytes_per_dim;
        self.last_max[..len].copy_from_slice(&max[..len]);
        self.last_min[..len].copy_from_slice(&min[..len]);
        let relation = self.delegate.compare(min, max);
        self.last_compare = Some(relation);
        relation
    }
}

/// Native adapter preserving Java PointsReader's inherited identity merge instance.
#[derive(Default)]
pub struct Lucene90PointsFormat;
struct NativeState<'a> {
    reader: byte::Lucene90PointsReader<'a>,
    closed: AtomicBool,
}
pub struct Lucene90PointsReader<'a> {
    state: Arc<NativeState<'a>>,
}
/// Acquired metadata survives native reader close, but a new tree cannot be
/// acquired afterwards; already acquired trees retain their independent inputs.
pub struct NativeValues<'r, 'd> {
    delegate: byte::NativeValues<'r, 'd>,
    closed: &'r AtomicBool,
}
fn closed_points() -> Error {
    Error::Message("closed points reader".into())
}
impl PointValues for NativeValues<'_, '_> {
    type Tree<'a>
        = byte::NativeTree<'a, 'a>
    where
        Self: 'a;
    fn point_tree(&self) -> Result<Self::Tree<'_>> {
        if self.closed.load(Ordering::Relaxed) {
            return Err(closed_points());
        }
        self.delegate.point_tree()
    }
    fn min_packed_value(&self) -> Result<&[u8]> {
        self.delegate.min_packed_value()
    }
    fn max_packed_value(&self) -> Result<&[u8]> {
        self.delegate.max_packed_value()
    }
    fn num_dimensions(&self) -> Result<i32> {
        self.delegate.num_dimensions()
    }
    fn num_index_dimensions(&self) -> Result<i32> {
        self.delegate.num_index_dimensions()
    }
    fn bytes_per_dimension(&self) -> Result<i32> {
        self.delegate.bytes_per_dimension()
    }
    fn size(&self) -> i64 {
        self.delegate.size()
    }
    fn doc_count(&self) -> i32 {
        self.delegate.doc_count()
    }
}
impl PointsFormat for Lucene90PointsFormat {
    type Writer<'a> = byte::Lucene90PointsWriter<'a>;
    type Reader<'a> = Lucene90PointsReader<'a>;
    fn fields_writer<'a>(&mut self, state: WriteState<'a>) -> Result<Self::Writer<'a>> {
        byte::Lucene90PointsFormat.fields_writer(state)
    }
    fn fields_reader<'a>(&mut self, state: ReadState<'a>) -> Result<Self::Reader<'a>> {
        Ok(Lucene90PointsReader {
            state: Arc::new(NativeState {
                reader: byte::Lucene90PointsFormat.fields_reader(state.bytes)?,
                closed: AtomicBool::new(false),
            }),
        })
    }
}
impl ByteReader for Lucene90PointsReader<'_> {
    type Values<'a>
        = NativeValues<'a, 'a>
    where
        Self: 'a;
    fn values(&self, name: &str) -> Result<Option<Self::Values<'_>>> {
        // Lucene90 checks FieldInfos before consulting its cleared reader map.
        let values = self.state.reader.values(name)?;
        if self.state.closed.load(Ordering::Relaxed) {
            return Ok(None);
        }
        Ok(values.map(|delegate| NativeValues {
            delegate,
            closed: &self.state.closed,
        }))
    }
    fn check_integrity(&self) -> Result<()> {
        if self.state.closed.load(Ordering::Relaxed) {
            return Err(closed_points());
        }
        self.state.reader.check_integrity()
    }
    fn close(&mut self) -> Result<()> {
        self.state.closed.store(true, Ordering::Relaxed);
        Ok(())
    }
    fn native(&self) -> Option<&lucene_codecs::points::PointsReader<'_>> {
        if self.state.closed.load(Ordering::Relaxed) {
            None
        } else {
            self.state.reader.native()
        }
    }
}
impl PointsReader for Lucene90PointsReader<'_> {
    fn get_merge_instance(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}
impl fmt::Display for Lucene90PointsReader<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Lucene90PointsReader")
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    #[derive(Default)]
    struct Recorder {
        values: Vec<(i32, Vec<u8>)>,
        fail_grow: bool,
    }
    impl IntersectVisitor for Recorder {
        fn visit(&mut self, doc: i32) -> Result<()> {
            self.values.push((doc, Vec::new()));
            Ok(())
        }
        fn visit_value(&mut self, doc: i32, value: &[u8]) -> Result<()> {
            self.values.push((doc, value.to_vec()));
            Ok(())
        }
        fn grow(&mut self, _: usize) {
            assert!(!self.fail_grow, "grow failure");
        }
        fn compare(&mut self, _: &[u8], _: &[u8]) -> Relation {
            Relation::CellCrossesQuery
        }
    }
    #[test]
    fn scalar_document_visit_consumes_budget_before_failing() {
        let mut recorder = Recorder::default();
        let mut visitor = AssertingIntersectVisitor::new(1, 1, 1, &mut recorder);
        visitor.grow(1);
        visitor.visit(7).unwrap();
        assert!(catch_unwind(AssertUnwindSafe(|| visitor.visit(8))).is_err());
        assert_eq!(visitor.budget, -1);
        assert_eq!(recorder.values, [(7, Vec::new())]);
    }
    #[test]
    fn multidimensional_bounds_only_cover_index_dimensions_and_do_not_order_points() {
        let mut recorder = Recorder::default();
        let mut visitor = AssertingIntersectVisitor::new(2, 1, 1, &mut recorder);
        visitor.grow(2);
        assert_eq!(visitor.compare(&[10], &[20]), Relation::CellCrossesQuery);
        visitor.visit_value(8, &[15, 255]).unwrap();
        visitor.visit_value(0, &[0, 0]).unwrap();
        assert_eq!(recorder.values, [(8, vec![15, 255]), (0, vec![0, 0])]);
    }
    #[test]
    fn failed_delegate_grow_preserves_previous_budget_and_failed_compare_preserves_previous_relation(
    ) {
        let mut recorder = Recorder {
            fail_grow: true,
            ..Recorder::default()
        };
        let mut visitor = AssertingIntersectVisitor::new(1, 1, 1, &mut recorder);
        visitor.budget = 2;
        assert!(catch_unwind(AssertUnwindSafe(|| visitor.grow(1))).is_err());
        assert_eq!(visitor.budget, 2);
        assert_eq!(visitor.compare(&[0], &[255]), Relation::CellCrossesQuery);
        assert!(catch_unwind(AssertUnwindSafe(|| visitor.compare(&[2], &[1]))).is_err());
        assert_eq!(visitor.last_compare, Some(Relation::CellCrossesQuery));
        visitor.visit_value(0, &[5]).unwrap();
        visitor.visit_value(1, &[6]).unwrap();
        assert_eq!(recorder.values, [(0, vec![5]), (1, vec![6])]);
    }
}
