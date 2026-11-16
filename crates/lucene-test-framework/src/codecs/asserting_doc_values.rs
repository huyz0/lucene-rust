//! Java `AssertingDocValuesFormat`: validate one fresh iterator, then pass the
//! original producer to the delegate. Producer getters reuse the five existing
//! asserting leaf iterators. Assertions remain enabled in release. Merge readers
//! check their acquisition thread; successful closes call the delegate twice.
//!
//! Iterator producers are separate from cranky's owned-column seam so iteration
//! failures and callback order remain observable. Rust slices replace BytesRef;
//! Returned leaf iterators retain their existing non-Send seam and omit Java's
//! runtime iterator thread guard; merge producer and skipper guards are explicit.
//! Null values, fallible long/binary/ord getters, singleton unwrapping and Java's
//! unavailable bulk APIs cannot be represented. Inherited DocValuesConsumer merge
//! is outside this bounded seam (the cranky column seam has its own tested merge).
//!
//! The explicit Lucene90 adapter reuses the production byte readers/writer through
//! cranky's adapter. It eagerly owns decoded columns and publishes buffered files
//! on close; it does not replay Java's native writer acquisition passes or their
//! streaming error timing. Native merge handles share immutable state and close
//! independently, matching ByteBuffers inputs, not invalidating backends.
//! Directory/IOContext and Codec/IndexWriter routing are absent. A local skipper
//! trait preserves Java checks without changing the production skipper enum.
//! Unknown field membership raises a named Rust panic where Java dereferences null.
//! Native merge acquisition after close maps Java AlreadyClosedException to a panic;
//! existing iterators borrow their producer, preventing close while borrowed.

use super::cranky_doc_values::{self as bytes, Column};
pub use super::cranky_doc_values::{Files, ReadState};
use crate::index::asserting_leaf_reader::{
    AssertingBinaryDocValues, AssertingNumericDocValues, AssertingSortedDocValues,
    AssertingSortedNumericDocValues, AssertingSortedSetDocValues,
};
use lucene_codecs::doc_values::{
    BinaryColumn, SortedNumericColumn, SortedOrdsColumn, SortedSetOrdsColumn,
};
use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos};
use lucene_codecs::segment_state::SegmentWriteState;
use lucene_search::reader::{
    BinaryDocValues, DocIdSetIterator, DocValuesIterator, NumericDocValues, SortedDocValues,
    SortedNumericDocValues, SortedSetDocValues, NO_MORE_DOCS,
};
use lucene_search::Result;
use lucene_util::fixed_bit_set::FixedBitSet;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::thread::{self, ThreadId};
fn error(message: impl fmt::Display) -> lucene_search::Error {
    lucene_store::Error::Io(std::io::Error::other(message.to_string())).into()
}

/// Java's abstract skipper, including fallible advance and observable getters.
pub trait DocValuesSkipper: Send {
    fn advance(&mut self, target: i32) -> Result<()>;
    fn num_levels(&self) -> usize;
    fn min_doc_id(&self, level: usize) -> i32;
    fn max_doc_id(&self, level: usize) -> i32;
    fn min_value(&self, level: usize) -> i64;
    fn max_value(&self, level: usize) -> i64;
    fn doc_count(&self, level: usize) -> i32;
    fn global_min_value(&self) -> i64;
    fn global_max_value(&self) -> i64;
    fn global_doc_count(&self) -> i32;
    fn max_value_count(&self) -> i32;
}
/// Fresh iterators retain source errors and producer identity across validation.
pub trait DocValuesProducer: fmt::Display {
    fn get_numeric(&self, _: &FieldInfo) -> Result<Box<dyn NumericDocValues + '_>> {
        Err(error("unsupported numeric"))
    }
    fn get_binary(&self, _: &FieldInfo) -> Result<Box<dyn BinaryDocValues + '_>> {
        Err(error("unsupported binary"))
    }
    fn get_sorted(&self, _: &FieldInfo) -> Result<Box<dyn SortedDocValues + '_>> {
        Err(error("unsupported sorted"))
    }
    fn get_sorted_numeric(&self, _: &FieldInfo) -> Result<Box<dyn SortedNumericDocValues + '_>> {
        Err(error("unsupported sorted numeric"))
    }
    fn get_sorted_set(&self, _: &FieldInfo) -> Result<Box<dyn SortedSetDocValues + '_>> {
        Err(error("unsupported sorted set"))
    }
    fn get_skipper(&self, _: &FieldInfo) -> Result<Option<Box<dyn DocValuesSkipper + '_>>> {
        Ok(None)
    }
    fn close(&mut self) -> Result<()>;
    fn check_integrity(&self) -> Result<()>;
    fn get_merge_instance(&self) -> Self
    where
        Self: Sized;
}
pub trait DocValuesConsumer {
    fn add_numeric_field(
        &mut self,
        field: &FieldInfo,
        producer: &dyn DocValuesProducer,
    ) -> Result<()>;
    fn add_binary_field(
        &mut self,
        field: &FieldInfo,
        producer: &dyn DocValuesProducer,
    ) -> Result<()>;
    fn add_sorted_field(
        &mut self,
        field: &FieldInfo,
        producer: &dyn DocValuesProducer,
    ) -> Result<()>;
    fn add_sorted_numeric_field(
        &mut self,
        field: &FieldInfo,
        producer: &dyn DocValuesProducer,
    ) -> Result<()>;
    fn add_sorted_set_field(
        &mut self,
        field: &FieldInfo,
        producer: &dyn DocValuesProducer,
    ) -> Result<()>;
    fn close(&mut self) -> Result<()>;
}
pub trait DocValuesFormat {
    type Consumer<'a>: DocValuesConsumer
    where
        Self: 'a;
    type Producer<'a>: DocValuesProducer
    where
        Self: 'a;
    fn name(&self) -> &str;
    fn fields_consumer<'a>(&'a self, state: SegmentWriteState<'a>) -> Result<Self::Consumer<'a>>;
    fn fields_producer<'a>(&'a self, state: ReadState<'a>) -> Result<Self::Producer<'a>>;
}
pub struct AssertingDocValuesFormat<F> {
    delegate: F,
}
impl<F> AssertingDocValuesFormat<F> {
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
// FieldInfos are borrowed by producer wrappers, as they are by segment readers.
impl<F: DocValuesFormat> DocValuesFormat for AssertingDocValuesFormat<F> {
    type Consumer<'a>
        = AssertingDocValuesConsumer<F::Consumer<'a>>
    where
        Self: 'a;
    type Producer<'a>
        = AssertingDocValuesProducer<'a, F::Producer<'a>>
    where
        Self: 'a;
    fn name(&self) -> &str {
        "Asserting"
    }
    /// Java: AssertingDocValuesFormat.fieldsConsumer.
    fn fields_consumer<'a>(&'a self, state: SegmentWriteState<'a>) -> Result<Self::Consumer<'a>> {
        Ok(AssertingDocValuesConsumer::new(
            self.delegate.fields_consumer(state)?,
            state.max_doc,
        ))
    }
    /// Java: AssertingDocValuesFormat.fieldsProducer: hasDocValues before delegate.
    fn fields_producer<'a>(&'a self, state: ReadState<'a>) -> Result<Self::Producer<'a>> {
        assert!(state
            .segment
            .field_infos
            .fields
            .iter()
            .any(|f| f.doc_values_type != DocValuesType::None));
        Ok(AssertingDocValuesProducer::new(
            self.delegate.fields_producer(state)?,
            state.segment.field_infos,
            state.segment.max_doc,
        ))
    }
}
pub struct AssertingDocValuesConsumer<C> {
    delegate: C,
    max_doc: i32,
}
impl<C> AssertingDocValuesConsumer<C> {
    pub fn new(delegate: C, max_doc: i32) -> Self {
        Self { delegate, max_doc }
    }
    pub fn delegate(&self) -> &C {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut C {
        &mut self.delegate
    }
}
impl<C: DocValuesConsumer> DocValuesConsumer for AssertingDocValuesConsumer<C> {
    /// Java: AssertingDocValuesConsumer.addNumericField.
    fn add_numeric_field(
        &mut self,
        field: &FieldInfo,
        producer: &dyn DocValuesProducer,
    ) -> Result<()> {
        {
            let mut values = producer.get_numeric(field)?;
            let mut last = -1;
            loop {
                let doc = values.next_doc()?;
                if doc == NO_MORE_DOCS {
                    break;
                }
                assert!(doc >= 0 && doc < self.max_doc);
                assert!(doc > last);
                last = doc;
                values.long_value();
            }
        }
        self.delegate.add_numeric_field(field, producer)
    }
    /// Java: AssertingDocValuesConsumer.addBinaryField. A Rust slice is valid.
    fn add_binary_field(
        &mut self,
        field: &FieldInfo,
        producer: &dyn DocValuesProducer,
    ) -> Result<()> {
        {
            let mut values = producer.get_binary(field)?;
            let mut last = -1;
            loop {
                let doc = values.next_doc()?;
                if doc == NO_MORE_DOCS {
                    break;
                }
                assert!(doc >= 0 && doc < self.max_doc);
                assert!(doc > last);
                last = doc;
                values.binary_value();
            }
        }
        self.delegate.add_binary_field(field, producer)
    }
    /// Java: AssertingDocValuesConsumer.addSortedField.
    fn add_sorted_field(
        &mut self,
        field: &FieldInfo,
        producer: &dyn DocValuesProducer,
    ) -> Result<()> {
        {
            let mut values = producer.get_sorted(field)?;
            let count = values.value_count();
            assert!(count <= self.max_doc);
            let mut prior = None;
            for ord in 0..count {
                let term = values.lookup_ord(ord)?;
                if let Some(prior) = prior {
                    assert!(term > prior);
                }
                prior = Some(term);
            }
            // Java's bitset cannot represent a negative value count either.
            let mut seen = FixedBitSet::new(usize::try_from(count).expect("negative value count"));
            let mut last = -1;
            loop {
                let doc = values.next_doc()?;
                if doc == NO_MORE_DOCS {
                    break;
                }
                assert!(doc >= 0 && doc < self.max_doc);
                assert!(doc > last);
                last = doc;
                let ord = values.ord_value();
                // The length was constructed from nonnegative i32 count above.
                assert!(ord >= 0 && ord < seen.len() as i32);
                seen.set(ord as usize);
            }
            assert_eq!(seen.cardinality(), count as usize);
        }
        self.delegate.add_sorted_field(field, producer)
    }
    /// Java checks increasing docID and sorted values here, not maxDoc bounds.
    fn add_sorted_numeric_field(
        &mut self,
        field: &FieldInfo,
        producer: &dyn DocValuesProducer,
    ) -> Result<()> {
        {
            let mut values = producer.get_sorted_numeric(field)?;
            let mut last = -1;
            loop {
                if values.next_doc()? == NO_MORE_DOCS {
                    break;
                }
                assert!(values.doc_id() > last);
                last = values.doc_id();
                let count = values.doc_value_count();
                assert!(count > 0);
                let mut prior = i64::MIN;
                for _ in 0..count {
                    let value = values.next_value()?;
                    assert!(value >= prior);
                    prior = value;
                }
            }
        }
        self.delegate.add_sorted_numeric_field(field, producer)
    }
    /// Java checks dictionary/ordinals here, not document order, bounds or count>0.
    fn add_sorted_set_field(
        &mut self,
        field: &FieldInfo,
        producer: &dyn DocValuesProducer,
    ) -> Result<()> {
        {
            let mut values = producer.get_sorted_set(field)?;
            let count = values.value_count();
            let mut prior = None;
            for ord in 0..count {
                let term = values.lookup_ord(ord)?;
                if let Some(prior) = prior {
                    assert!(term > prior);
                }
                prior = Some(term);
            }
            // LongBitSet's length must fit the host's addressable memory.
            let mut seen =
                FixedBitSet::new(usize::try_from(count).expect("unrepresentable value count"));
            loop {
                if values.next_doc()? == NO_MORE_DOCS {
                    break;
                }
                let mut last = -1;
                let mut i = 0;
                while i < values.doc_value_count() {
                    let ord = values.next_ord()?;
                    // The length was constructed from representable i64 count above.
                    assert!(ord >= 0 && ord < seen.len() as i64);
                    assert!(ord > last);
                    seen.set(ord as usize);
                    last = ord;
                    i += 1;
                }
            }
            assert_eq!(seen.cardinality(), count as usize);
        }
        self.delegate.add_sorted_set_field(field, producer)
    }
    /// Java: close again only after the first delegate close succeeds.
    fn close(&mut self) -> Result<()> {
        self.delegate.close()?;
        self.delegate.close()
    }
}

pub struct AssertingDocValuesProducer<'a, P> {
    delegate: P,
    fields: &'a FieldInfos,
    max_doc: i32,
    merging: bool,
    creation_thread: ThreadId,
}
impl<'a, P: DocValuesProducer> AssertingDocValuesProducer<'a, P> {
    pub fn new(delegate: P, fields: &'a FieldInfos, max_doc: i32) -> Self {
        let result = Self {
            delegate,
            fields,
            max_doc,
            merging: false,
            creation_thread: thread::current().id(),
        };
        // Java constructor asserts toString()!=null: keep the delegate formatting callback.
        let _ = result.to_string();
        result
    }
    pub fn delegate(&self) -> &P {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut P {
        &mut self.delegate
    }
    fn member(&self, field: &FieldInfo) {
        let Some(info) = self.fields.field_by_name(&field.name) else {
            panic!("null field membership");
        };
        assert_eq!(info.number, field.number);
    }
    fn typed(&self, field: &FieldInfo, kind: DocValuesType) {
        self.member(field);
        if self.merging {
            assert_eq!(self.creation_thread, thread::current().id(), "DocValuesProducer are only supposed to be consumed in the thread in which they have been acquired.");
        }
        assert_eq!(field.doc_values_type, kind);
    }
}
impl<P: DocValuesProducer> fmt::Display for AssertingDocValuesProducer<'_, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AssertingDocValuesProducer({})", self.delegate)
    }
}
impl<P: DocValuesProducer> DocValuesProducer for AssertingDocValuesProducer<'_, P> {
    fn get_numeric(&self, f: &FieldInfo) -> Result<Box<dyn NumericDocValues + '_>> {
        self.typed(f, DocValuesType::Numeric);
        Ok(Box::new(AssertingNumericDocValues::new(
            self.delegate.get_numeric(f)?,
            self.max_doc,
        )))
    }
    fn get_binary(&self, f: &FieldInfo) -> Result<Box<dyn BinaryDocValues + '_>> {
        self.typed(f, DocValuesType::Binary);
        Ok(Box::new(AssertingBinaryDocValues::new(
            self.delegate.get_binary(f)?,
            self.max_doc,
        )))
    }
    fn get_sorted(&self, f: &FieldInfo) -> Result<Box<dyn SortedDocValues + '_>> {
        self.typed(f, DocValuesType::Sorted);
        Ok(Box::new(AssertingSortedDocValues::new(
            self.delegate.get_sorted(f)?,
            self.max_doc,
        )))
    }
    fn get_sorted_numeric(&self, f: &FieldInfo) -> Result<Box<dyn SortedNumericDocValues + '_>> {
        self.typed(f, DocValuesType::SortedNumeric);
        Ok(Box::new(AssertingSortedNumericDocValues::new(
            self.delegate.get_sorted_numeric(f)?,
            self.max_doc,
        )))
    }
    fn get_sorted_set(&self, f: &FieldInfo) -> Result<Box<dyn SortedSetDocValues + '_>> {
        self.typed(f, DocValuesType::SortedSet);
        Ok(Box::new(AssertingSortedSetDocValues::new(
            self.delegate.get_sorted_set(f)?,
            self.max_doc,
        )))
    }
    fn get_skipper(&self, f: &FieldInfo) -> Result<Option<Box<dyn DocValuesSkipper + '_>>> {
        self.member(f);
        assert_ne!(f.doc_values_skip_index_type, DocValuesSkipIndexType::None);
        Ok(Some(Box::new(AssertingDocValuesSkipper::new(
            self.delegate.get_skipper(f)?.expect("null skipper"),
        ))))
    }
    fn close(&mut self) -> Result<()> {
        self.delegate.close()?;
        self.delegate.close()
    }
    fn check_integrity(&self) -> Result<()> {
        self.delegate.check_integrity()
    }
    fn get_merge_instance(&self) -> Self {
        let mut result = Self::new(
            self.delegate.get_merge_instance(),
            self.fields,
            self.max_doc,
        );
        result.merging = true;
        result
    }
}

pub struct AssertingDocValuesSkipper<'a> {
    delegate: Box<dyn DocValuesSkipper + 'a>,
    creation_thread: ThreadId,
}
impl<'a> AssertingDocValuesSkipper<'a> {
    pub fn new(delegate: Box<dyn DocValuesSkipper + 'a>) -> Self {
        let result = Self {
            delegate,
            creation_thread: thread::current().id(),
        };
        assert_eq!(result.min_doc_id(0), -1);
        assert_eq!(result.max_doc_id(0), -1);
        result
    }
    fn thread(&self) {
        assert_eq!(self.creation_thread, thread::current().id(), "Doc values skipper are only supposed to be consumed in the thread in which they have been acquired.");
    }
    fn iterating(&self) -> bool {
        self.max_doc_id(0) != -1
            && self.min_doc_id(0) != -1
            && self.max_doc_id(0) != NO_MORE_DOCS
            && self.min_doc_id(0) != NO_MORE_DOCS
    }
}
impl DocValuesSkipper for AssertingDocValuesSkipper<'_> {
    fn advance(&mut self, target: i32) -> Result<()> {
        self.thread();
        assert!(
            target > self.max_doc_id(0),
            "Illegal to call advance() on a target that is not beyond the current interval"
        );
        self.delegate.advance(target)?;
        assert!(self.delegate.min_doc_id(0) <= self.delegate.max_doc_id(0));
        Ok(())
    }
    fn num_levels(&self) -> usize {
        self.thread();
        self.delegate.num_levels()
    }
    fn min_doc_id(&self, level: usize) -> i32 {
        self.thread();
        assert!(level < self.num_levels());
        let min = self.delegate.min_doc_id(level);
        assert!(min <= self.delegate.max_doc_id(level));
        if level > 0 {
            assert!(min <= self.delegate.min_doc_id(level - 1));
        }
        min
    }
    fn max_doc_id(&self, level: usize) -> i32 {
        self.thread();
        assert!(level < self.num_levels());
        let max = self.delegate.max_doc_id(level);
        assert!(max >= self.delegate.min_doc_id(level));
        if level > 0 {
            assert!(max >= self.delegate.max_doc_id(level - 1));
        }
        max
    }
    fn min_value(&self, level: usize) -> i64 {
        self.thread();
        assert!(self.iterating(), "Unpositioned iterator");
        assert!(level < self.num_levels());
        self.delegate.min_value(level)
    }
    fn max_value(&self, level: usize) -> i64 {
        self.thread();
        assert!(self.iterating(), "Unpositioned iterator");
        assert!(level < self.num_levels());
        self.delegate.max_value(level)
    }
    fn doc_count(&self, level: usize) -> i32 {
        self.thread();
        assert!(self.iterating(), "Unpositioned iterator");
        assert!(level < self.num_levels());
        self.delegate.doc_count(level)
    }
    fn global_min_value(&self) -> i64 {
        self.thread();
        self.delegate.global_min_value()
    }
    fn global_max_value(&self) -> i64 {
        self.thread();
        self.delegate.global_max_value()
    }
    fn global_doc_count(&self) -> i32 {
        self.thread();
        self.delegate.global_doc_count()
    }
    fn max_value_count(&self) -> i32 {
        self.thread();
        self.delegate.max_value_count()
    }
}
impl DocValuesSkipper for lucene_search::reader::DocValuesSkipper<'_> {
    fn advance(&mut self, target: i32) -> Result<()> {
        self.advance(target);
        Ok(())
    }
    fn num_levels(&self) -> usize {
        self.num_levels()
    }
    fn min_doc_id(&self, level: usize) -> i32 {
        self.min_doc_id(level)
    }
    fn max_doc_id(&self, level: usize) -> i32 {
        self.max_doc_id(level)
    }
    fn min_value(&self, level: usize) -> i64 {
        self.min_value(level)
    }
    fn max_value(&self, level: usize) -> i64 {
        self.max_value(level)
    }
    fn doc_count(&self, level: usize) -> i32 {
        self.doc_count(level)
    }
    fn global_min_value(&self) -> i64 {
        self.global_min_value()
    }
    fn global_max_value(&self) -> i64 {
        self.global_max_value()
    }
    fn global_doc_count(&self) -> i32 {
        self.global_doc_count()
    }
    fn max_value_count(&self) -> i32 {
        self.max_value_count()
    }
}

/// Native decoded columns, shared by independent producer/merge handles.
pub struct Lucene90DocValuesProducer {
    delegate: Arc<bytes::OwnedDocValuesProducer>,
    closed: bool,
}
impl Lucene90DocValuesProducer {
    pub fn from_columns(columns: BTreeMap<i32, Column>) -> Self {
        Self {
            delegate: Arc::new(bytes::OwnedDocValuesProducer::new(columns)),
            closed: false,
        }
    }
    fn column(&self, field: &FieldInfo) -> Result<&Column> {
        if self.closed {
            return Err(error("closed doc values producer"));
        }
        self.delegate
            .columns
            .get(&field.number)
            .filter(|c| {
                matches!(
                    (c, field.doc_values_type),
                    (Column::Numeric(_), DocValuesType::Numeric)
                        | (Column::Binary(_), DocValuesType::Binary)
                        | (Column::Sorted(_), DocValuesType::Sorted)
                        | (Column::SortedNumeric(_), DocValuesType::SortedNumeric)
                        | (Column::SortedSet(_), DocValuesType::SortedSet)
                )
            })
            .ok_or_else(|| error("missing or incompatible field"))
    }
}
impl fmt::Display for Lucene90DocValuesProducer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Lucene90DocValuesProducer")
    }
}
struct Values<'a> {
    column: &'a Column,
    row: Option<usize>,
    doc: i32,
    upto: usize,
}
impl<'a> Values<'a> {
    fn new(column: &'a Column) -> Self {
        Self {
            column,
            row: None,
            doc: -1,
            upto: 0,
        }
    }
    fn len(&self) -> usize {
        match self.column {
            Column::Numeric(c) => c.len(),
            Column::Binary(c) => c.docs.len(),
            Column::Sorted(c) => c.docs.len(),
            Column::SortedNumeric(c) => c.docs.len(),
            Column::SortedSet(c) => c.docs.len(),
        }
    }
    fn doc_at(&self, i: usize) -> i32 {
        match self.column {
            Column::Numeric(c) => c[i].0,
            Column::Binary(c) => c.docs[i],
            Column::Sorted(c) => c.docs[i],
            Column::SortedNumeric(c) => c.docs[i],
            Column::SortedSet(c) => c.docs[i],
        }
    }
    fn count(&self) -> i32 {
        let i = self.row.expect("unpositioned values");
        match self.column {
            Column::SortedNumeric(c) => {
                i32::try_from(c.counts[i]).expect("doc value count exceeds i32")
            }
            Column::SortedSet(c) => {
                i32::try_from(c.counts[i]).expect("doc value count exceeds i32")
            }
            _ => 1,
        }
    }
    fn offset(&self) -> usize {
        let i = self.row.expect("unpositioned values");
        match self.column {
            Column::Binary(c) => c.lengths[..i].iter().map(|n| *n as usize).sum(),
            Column::SortedNumeric(c) => c.counts[..i].iter().map(|n| *n as usize).sum(),
            Column::SortedSet(c) => c.counts[..i].iter().map(|n| *n as usize).sum(),
            _ => i,
        }
    }
    fn dictionary(&self) -> &[Vec<u8>] {
        match self.column {
            Column::Sorted(c) => &c.dict,
            Column::SortedSet(c) => &c.dict,
            _ => unreachable!(),
        }
    }
    fn lookup(&self, ord: i64) -> Result<Vec<u8>> {
        let ord = usize::try_from(ord).map_err(error)?;
        self.dictionary()
            .get(ord)
            .cloned()
            .ok_or_else(|| error("ordinal out of bounds"))
    }
}
impl DocIdSetIterator for Values<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        if self.doc == NO_MORE_DOCS {
            return Ok(self.doc);
        }
        let i = self.row.map_or(0, |i| i + 1);
        self.row = Some(i);
        self.upto = 0;
        self.doc = if i < self.len() {
            self.doc_at(i)
        } else {
            NO_MORE_DOCS
        };
        Ok(self.doc)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        while self.next_doc()? < target {}
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        self.len() as i64
    }
}
impl DocValuesIterator for Values<'_> {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        let i = (0..self.len())
            .find(|&i| self.doc_at(i) >= target)
            .unwrap_or(self.len());
        let found = i < self.len() && self.doc_at(i) == target;
        self.row = if found { Some(i) } else { i.checked_sub(1) };
        self.doc = target;
        self.upto = 0;
        Ok(found)
    }
}
impl NumericDocValues for Values<'_> {
    fn long_value(&self) -> i64 {
        match self.column {
            Column::Numeric(c) => c[self.row.unwrap()].1,
            _ => unreachable!(),
        }
    }
}
impl BinaryDocValues for Values<'_> {
    fn binary_value(&self) -> &[u8] {
        match self.column {
            Column::Binary(c) => {
                let start = self.offset();
                &c.bytes[start..start + c.lengths[self.row.unwrap()] as usize]
            }
            _ => unreachable!(),
        }
    }
}
impl SortedDocValues for Values<'_> {
    fn ord_value(&self) -> i32 {
        match self.column {
            Column::Sorted(c) => c.ords[self.row.unwrap()] as i32,
            _ => unreachable!(),
        }
    }
    fn lookup_ord(&mut self, ord: i32) -> Result<Vec<u8>> {
        self.lookup(i64::from(ord))
    }
    fn value_count(&self) -> i32 {
        self.dictionary()
            .len()
            .try_into()
            .expect("too many sorted values")
    }
}
impl SortedNumericDocValues for Values<'_> {
    fn doc_value_count(&self) -> i32 {
        self.count()
    }
    fn next_value(&mut self) -> Result<i64> {
        match self.column {
            Column::SortedNumeric(c) => {
                let value = c
                    .values
                    .get(self.offset() + self.upto)
                    .copied()
                    .ok_or_else(|| error("value out of bounds"))?;
                self.upto += 1;
                Ok(value)
            }
            _ => unreachable!(),
        }
    }
}
impl SortedSetDocValues for Values<'_> {
    fn doc_value_count(&self) -> i32 {
        self.count()
    }
    fn next_ord(&mut self) -> Result<i64> {
        match self.column {
            Column::SortedSet(c) => {
                let value = c
                    .ords
                    .get(self.offset() + self.upto)
                    .copied()
                    .ok_or_else(|| error("ordinal out of bounds"))?;
                self.upto += 1;
                Ok(value)
            }
            _ => unreachable!(),
        }
    }
    fn lookup_ord(&mut self, ord: i64) -> Result<Vec<u8>> {
        self.lookup(ord)
    }
    fn value_count(&self) -> i64 {
        self.dictionary().len() as i64
    }
}
macro_rules! native_get {
    ($method:ident, $trait:ident, $kind:ident) => {
        fn $method(&self, field: &FieldInfo) -> Result<Box<dyn $trait + '_>> {
            if field.doc_values_type != DocValuesType::$kind {
                return Err(error("unsupported getter"));
            }
            Ok(Box::new(Values::new(self.column(field)?)))
        }
    };
}
impl DocValuesProducer for Lucene90DocValuesProducer {
    native_get!(get_numeric, NumericDocValues, Numeric);
    native_get!(get_binary, BinaryDocValues, Binary);
    native_get!(get_sorted, SortedDocValues, Sorted);
    native_get!(get_sorted_numeric, SortedNumericDocValues, SortedNumeric);
    native_get!(get_sorted_set, SortedSetDocValues, SortedSet);
    fn get_skipper(&self, field: &FieldInfo) -> Result<Option<Box<dyn DocValuesSkipper + '_>>> {
        self.column(field)?;
        Ok(self.delegate.skippers.get(&field.number).map(|index| {
            Box::new(lucene_search::reader::DocValuesSkipper::new(index))
                as Box<dyn DocValuesSkipper>
        }))
    }
    fn check_integrity(&self) -> Result<()> {
        if self.closed {
            return Err(error("closed doc values producer"));
        }
        bytes::DocValuesProducer::check_integrity(&*self.delegate).map_err(error)
    }
    fn close(&mut self) -> Result<()> {
        self.closed = true;
        Ok(())
    }
    fn get_merge_instance(&self) -> Self {
        assert!(!self.closed, "closed doc values producer");
        Self {
            delegate: self.delegate.clone(),
            closed: false,
        }
    }
}

/// Explicit buffered Lucene90 adapter; wrapper behavior remains iterator based.
#[derive(Default)]
pub struct Lucene90DocValuesFormat;
pub struct Lucene90DocValuesConsumer<'a> {
    delegate: bytes::Lucene90DocValuesConsumer<'a>,
}
impl Lucene90DocValuesConsumer<'_> {
    pub fn output(&self) -> Option<&Files> {
        self.delegate.output()
    }
}
impl DocValuesFormat for Lucene90DocValuesFormat {
    type Consumer<'a> = Lucene90DocValuesConsumer<'a>;
    type Producer<'a> = Lucene90DocValuesProducer;
    fn name(&self) -> &str {
        "Lucene90"
    }
    fn fields_consumer<'a>(&'a self, state: SegmentWriteState<'a>) -> Result<Self::Consumer<'a>> {
        use bytes::DocValuesFormat;
        Ok(Lucene90DocValuesConsumer {
            delegate: bytes::Lucene90DocValuesFormat::default()
                .fields_consumer(state)
                .map_err(error)?,
        })
    }
    fn fields_producer<'a>(&'a self, state: ReadState<'a>) -> Result<Self::Producer<'a>> {
        use bytes::DocValuesFormat;
        Ok(Lucene90DocValuesProducer {
            delegate: Arc::new(
                bytes::Lucene90DocValuesFormat::default()
                    .fields_producer(state)
                    .map_err(error)?,
            ),
            closed: false,
        })
    }
}
/// Drain a fresh iterator into the existing native writer's column boundary.
pub fn snapshot(field: &FieldInfo, producer: &dyn DocValuesProducer) -> Result<Column> {
    Ok(match field.doc_values_type {
        DocValuesType::Numeric => {
            let mut v = producer.get_numeric(field)?;
            let mut rows = Vec::new();
            loop {
                let doc = v.next_doc()?;
                if doc == NO_MORE_DOCS {
                    break;
                }
                rows.push((doc, v.long_value()));
            }
            Column::Numeric(rows)
        }
        DocValuesType::Binary => {
            let mut v = producer.get_binary(field)?;
            let mut c = BinaryColumn::default();
            loop {
                let doc = v.next_doc()?;
                if doc == NO_MORE_DOCS {
                    break;
                }
                c.docs.push(doc);
                let value = v.binary_value();
                c.lengths.push(value.len().try_into().map_err(error)?);
                c.bytes.extend_from_slice(value);
            }
            Column::Binary(c)
        }
        DocValuesType::Sorted => {
            let mut v = producer.get_sorted(field)?;
            let mut c = SortedOrdsColumn::default();
            for ord in 0..v.value_count() {
                c.dict.push(v.lookup_ord(ord)?);
            }
            loop {
                let doc = v.next_doc()?;
                if doc == NO_MORE_DOCS {
                    break;
                }
                c.docs.push(doc);
                c.ords.push(i64::from(v.ord_value()));
            }
            Column::Sorted(c)
        }
        DocValuesType::SortedNumeric => {
            let mut v = producer.get_sorted_numeric(field)?;
            let mut c = SortedNumericColumn::default();
            loop {
                let doc = v.next_doc()?;
                if doc == NO_MORE_DOCS {
                    break;
                }
                c.docs.push(doc);
                let count = v.doc_value_count();
                c.counts.push(u32::try_from(count).map_err(error)?);
                for _ in 0..count {
                    c.values.push(v.next_value()?);
                }
            }
            Column::SortedNumeric(c)
        }
        DocValuesType::SortedSet => {
            let mut v = producer.get_sorted_set(field)?;
            let mut c = SortedSetOrdsColumn::default();
            for ord in 0..v.value_count() {
                c.dict.push(v.lookup_ord(ord)?);
            }
            loop {
                let doc = v.next_doc()?;
                if doc == NO_MORE_DOCS {
                    break;
                }
                c.docs.push(doc);
                let count = v.doc_value_count();
                c.counts.push(u32::try_from(count).map_err(error)?);
                for _ in 0..count {
                    c.ords.push(v.next_ord()?);
                }
            }
            Column::SortedSet(c)
        }
        DocValuesType::None => return Err(error("no doc values")),
    })
}
macro_rules! native_add {
    ($method:ident) => {
        fn $method(&mut self, field: &FieldInfo, producer: &dyn DocValuesProducer) -> Result<()> {
            let column = snapshot(field, producer)?;
            let source =
                bytes::OwnedDocValuesProducer::new(BTreeMap::from([(field.number, column)]));
            bytes::DocValuesConsumer::$method(&mut self.delegate, field, &source).map_err(error)
        }
    };
}
impl DocValuesConsumer for Lucene90DocValuesConsumer<'_> {
    native_add!(add_numeric_field);
    native_add!(add_binary_field);
    native_add!(add_sorted_field);
    native_add!(add_sorted_numeric_field);
    native_add!(add_sorted_set_field);
    fn close(&mut self) -> Result<()> {
        bytes::DocValuesConsumer::close(&mut self.delegate).map_err(error)
    }
}
#[cfg(test)]
mod tests;
