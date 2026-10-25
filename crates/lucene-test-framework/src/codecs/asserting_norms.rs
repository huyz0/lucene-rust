//! Java's `AssertingNormsFormat`, consumer and producer over a producer-aware
//! test seam. Validation visits the original values producer before passing
//! that same object to the delegate. Successful closes call the delegate twice;
//! a failure on the first close prevents the second. Merge producers enforce
//! their acquisition thread before obtaining any values.
//!
//! Returned iterators reuse `AssertingNumericDocValues`. Native numeric values
//! have infallible `long_value` and no `intoBitSet`, `docIDRunEnd` or bulk
//! `longValues`; nullable producers/values are unrepresentable. Assertions stay
//! enabled in release. There is no randomness in this asserting wrapper.
//!
//! The concrete Lucene90 adapter uses the production byte reader/writer,
//! without installing Codec/IndexWriter routing
//! or IOContext. The writer retains owned field buffers until close. It replays
//! Java's statistics, sparse-document and nonconstant-value producer passes,
//! then encodes the collected values with the whole-pair native writer. Input
//! producers must return the same values on repeated acquisition, as Java's
//! writer requires. Dense and empty iterators borrow their field metadata and
//! bytes; sparse iterators retain a native NormsReader and its forward cursor.
//! This keeps the cursor's state out of dense iterators. The measured baseline
//! recreated a point-lookup cursor on every document (0.56x Java); retaining it
//! changes only internal lookup state, keeping the same iterator callbacks.
//! Metadata is parsed and cross-validated against FieldInfos before opening
//! data. Merge adapters share immutable bytes but close independently, matching
//! Lucene90's cloned ByteBuffersIndexInput fixture backend. The owned-byte seam
//! does not model backends that invalidate clones when the original closes.

use std::sync::Arc;
use std::thread::{self, ThreadId};

use lucene_codecs::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use lucene_codecs::norms::{self, NormsField};
use lucene_search::reader::{DocIdSetIterator, DocValuesIterator, NumericDocValues, NO_MORE_DOCS};
use lucene_search::Result;
use lucene_store::codec_util;

use super::cranky_norms::{self, NormsState};
use super::cranky_norms::{NormsConsumer as ByteConsumer, NormsFormat as ByteFormat};
use crate::index::asserting_leaf_reader::AssertingNumericDocValues;

fn has_norms(field: &FieldInfo) -> bool {
    field.index_options != IndexOptions::None && !field.omit_norms
}

/// A fresh iterator for each acquisition, plus the producer lifecycle callbacks.
pub trait NormsProducer {
    fn get_norms(&self, field: &FieldInfo) -> Result<Box<dyn NumericDocValues + '_>>;
    fn close(&mut self) -> Result<()>;
    fn check_integrity(&self) -> Result<()>;
    fn get_merge_instance(&self) -> Self
    where
        Self: Sized;
}

/// Producer identity is retained through the add-field callback.
pub trait NormsConsumer {
    fn add_norms_field(&mut self, field: &FieldInfo, producer: &dyn NormsProducer) -> Result<()>;
    fn close(&mut self) -> Result<()>;
}

/// Segment fields are explicit because the byte-level NormsState has none.
pub trait NormsFormat {
    type Consumer: NormsConsumer;
    type Producer: NormsProducer;
    fn norms_consumer(&mut self, state: &NormsState<'_>) -> Result<Self::Consumer>;
    fn norms_producer(
        &mut self,
        state: &NormsState<'_>,
        fields: &FieldInfos,
    ) -> Result<Self::Producer>;
}

pub struct AssertingNormsFormat<F> {
    delegate: F,
}
impl<F> AssertingNormsFormat<F> {
    /// Java: `AssertingNormsFormat`, with an explicit delegate test seam.
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
impl<F: NormsFormat> NormsFormat for AssertingNormsFormat<F> {
    type Consumer = AssertingNormsConsumer<F::Consumer>;
    type Producer = AssertingNormsProducer<F::Producer>;
    /// Java: `AssertingNormsFormat.normsConsumer`.
    fn norms_consumer(&mut self, state: &NormsState<'_>) -> Result<Self::Consumer> {
        Ok(AssertingNormsConsumer::new(
            self.delegate.norms_consumer(state)?,
            state.max_doc,
        ))
    }
    /// Java: `AssertingNormsFormat.normsProducer`, checking metadata first.
    fn norms_producer(
        &mut self,
        state: &NormsState<'_>,
        fields: &FieldInfos,
    ) -> Result<Self::Producer> {
        assert!(fields.fields.iter().any(has_norms));
        Ok(AssertingNormsProducer::new(
            self.delegate.norms_producer(state, fields)?,
            state.max_doc,
        ))
    }
}

pub struct AssertingNormsConsumer<C> {
    delegate: C,
    max_doc: i32,
}
impl<C> AssertingNormsConsumer<C> {
    /// Java: `AssertingNormsConsumer(in,maxDoc)`.
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
impl<C: NormsConsumer> NormsConsumer for AssertingNormsConsumer<C> {
    /// Java: `AssertingNormsConsumer.addNormsField`.
    // The validation baseline (0.12x Java) crossed boxed iterator callbacks on
    // every document. Expose this generic wrapper to callers so known producer
    // vtables can specialize; opaque producers retain the original dyn seam.
    #[inline(always)]
    fn add_norms_field(&mut self, field: &FieldInfo, producer: &dyn NormsProducer) -> Result<()> {
        {
            let mut values = producer.get_norms(field)?;
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
        self.delegate.add_norms_field(field, producer)
    }
    /// Java: `AssertingNormsConsumer.close`, including short-circuit on error.
    fn close(&mut self) -> Result<()> {
        self.delegate.close()?;
        self.delegate.close()
    }
}

pub struct AssertingNormsProducer<P> {
    delegate: P,
    max_doc: i32,
    merging: bool,
    creation_thread: ThreadId,
}
impl<P> AssertingNormsProducer<P> {
    /// Java: `AssertingNormsProducer(in,maxDoc,false)`.
    pub fn new(delegate: P, max_doc: i32) -> Self {
        Self {
            delegate,
            max_doc,
            merging: false,
            creation_thread: thread::current().id(),
        }
    }
    pub fn delegate(&self) -> &P {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut P {
        &mut self.delegate
    }
}
impl<P: NormsProducer> NormsProducer for AssertingNormsProducer<P> {
    /// Java: `AssertingNormsProducer.getNorms`.
    fn get_norms(&self, field: &FieldInfo) -> Result<Box<dyn NumericDocValues + '_>> {
        if self.merging {
            assert_eq!(self.creation_thread,thread::current().id(),
                "NormsProducer are only supposed to be consumed in the thread in which they have been acquired.");
        }
        assert!(has_norms(field));
        Ok(Box::new(AssertingNumericDocValues::new(
            self.delegate.get_norms(field)?,
            self.max_doc,
        )))
    }
    /// Java: `AssertingNormsProducer.close`.
    fn close(&mut self) -> Result<()> {
        self.delegate.close()?;
        self.delegate.close()
    }
    /// Java: `AssertingNormsProducer.checkIntegrity`.
    fn check_integrity(&self) -> Result<()> {
        self.delegate.check_integrity()
    }
    /// Java: `AssertingNormsProducer.getMergeInstance`; acquisition thread resets.
    fn get_merge_instance(&self) -> Self {
        Self {
            delegate: self.delegate.get_merge_instance(),
            max_doc: self.max_doc,
            merging: true,
            creation_thread: thread::current().id(),
        }
    }
}
impl<P: std::fmt::Display> std::fmt::Display for AssertingNormsProducer<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AssertingNormsProducer({})", self.delegate)
    }
}

/// Real Lucene90 format adapter over the production byte seam.
pub struct Lucene90NormsFormat;
pub struct Lucene90NormsConsumer {
    delegate: cranky_norms::Lucene90NormsConsumer,
    max_doc: i32,
}
pub struct Lucene90NormsProducer {
    delegate: Option<Arc<NativeNorms>>,
    max_doc: i32,
    num_fields: usize,
}

struct NativeNorms {
    metadata: norms::Norms,
    data: Vec<u8>,
}
impl NativeNorms {
    fn check_integrity(&self) -> Result<()> {
        let end = self
            .data
            .len()
            .checked_sub(codec_util::FOOTER_LENGTH)
            .ok_or_else(|| {
                lucene_store::Error::Io(std::io::Error::other("missing norms data footer"))
            })?;
        codec_util::check_whole_file_footer(&self.data, end)?;
        Ok(())
    }
}

fn closed() -> lucene_search::Error {
    lucene_store::Error::Io(std::io::Error::other("closed norms producer")).into()
}

impl NormsFormat for Lucene90NormsFormat {
    type Consumer = Lucene90NormsConsumer;
    type Producer = Lucene90NormsProducer;
    fn norms_consumer(&mut self, state: &NormsState<'_>) -> Result<Self::Consumer> {
        Ok(Lucene90NormsConsumer {
            delegate: cranky_norms::Lucene90NormsFormat.norms_consumer(state)?,
            max_doc: state.max_doc,
        })
    }
    fn norms_producer(
        &mut self,
        state: &NormsState<'_>,
        fields: &FieldInfos,
    ) -> Result<Self::Producer> {
        let name = if state.suffix.is_empty() {
            state.segment.to_owned()
        } else {
            format!("{}_{}", state.segment, state.suffix)
        };
        let meta = state.directory.open(&format!("{name}.nvm"))?;
        let (_, metadata) = norms::parse_meta(&meta, state.id, state.suffix)?;
        norms::validate_fields(&metadata, fields)?;
        let data = state.directory.open(&format!("{name}.nvd"))?;
        norms::check_data_header_footer(&data, state.id, state.suffix)?;
        let num_fields = metadata.entries.len();
        Ok(Lucene90NormsProducer {
            delegate: Some(Arc::new(NativeNorms {
                metadata,
                data: data.to_vec(),
            })),
            max_doc: state.max_doc,
            num_fields,
        })
    }
}

impl NormsConsumer for Lucene90NormsConsumer {
    fn add_norms_field(&mut self, field: &FieldInfo, producer: &dyn NormsProducer) -> Result<()> {
        let mut docs = Vec::new();
        let (mut min, mut max) = (i64::MAX, i64::MIN);
        {
            let mut values = producer.get_norms(field)?;
            loop {
                let doc = values.next_doc()?;
                if doc == NO_MORE_DOCS {
                    break;
                }
                let value = values.long_value();
                min = min.min(value);
                max = max.max(value);
                docs.push((doc, value));
            }
        }
        if !docs.is_empty() && docs.len() != self.max_doc as usize {
            let mut values = producer.get_norms(field)?;
            while values.next_doc()? != NO_MORE_DOCS {}
        }
        if min < max {
            let mut values = producer.get_norms(field)?;
            while values.next_doc()? != NO_MORE_DOCS {
                values.long_value();
            }
        }
        self.delegate
            .add_norms_field(NormsField::Sparse(field.number, &docs))?;
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        Ok(self.delegate.close()?)
    }
}

// The first cached-cursor candidate measured 0.91x Java for the read case.
// Its NormsReader embedded sparse state even for dense fields. Specializing
// the dense/empty shape retains the production decoder without that state.
trait Values {
    fn value(&mut self, doc: i32) -> norms::Result<Option<i64>>;
}

struct PlainValues<'a> {
    data: &'a [u8],
    entry: &'a norms::NormsEntry,
}
impl Values for PlainValues<'_> {
    #[inline]
    fn value(&mut self, doc: i32) -> norms::Result<Option<i64>> {
        norms::norm_value(self.data, self.entry, doc)
    }
}
impl Values for norms::NormsReader<'_> {
    #[inline]
    fn value(&mut self, doc: i32) -> norms::Result<Option<i64>> {
        norms::NormsReader::value(self, doc)
    }
}

struct Numeric<V> {
    reader: V,
    max_doc: i32,
    doc: i32,
    value: i64,
    cost: i64,
}
impl<V: Values> DocIdSetIterator for Numeric<V> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.advance(self.doc.saturating_add(1))
    }
    fn advance(&mut self, mut target: i32) -> Result<i32> {
        assert!(target >= 0);
        while target < self.max_doc {
            if let Some(value) = self.reader.value(target)? {
                self.doc = target;
                self.value = value;
                return Ok(target);
            }
            target += 1;
        }
        self.doc = NO_MORE_DOCS;
        self.value = 0;
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        self.cost
    }
}
impl<V: Values> DocValuesIterator for Numeric<V> {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        if target < 0 || target >= self.max_doc {
            return Err(norms::Error::DocOutOfRange(target, self.max_doc).into());
        }
        self.doc = target;
        let value = self.reader.value(target)?;
        self.value = value.unwrap_or(0);
        Ok(value.is_some())
    }
}
impl<V: Values> NumericDocValues for Numeric<V> {
    fn long_value(&self) -> i64 {
        self.value
    }
}
impl NormsProducer for Lucene90NormsProducer {
    fn get_norms(&self, field: &FieldInfo) -> Result<Box<dyn NumericDocValues + '_>> {
        let producer = self.delegate.as_deref().ok_or_else(closed)?;
        let entry = producer
            .metadata
            .entry(field.number)
            .ok_or(norms::Error::UnknownFieldNumber(field.number))?;
        if entry.is_dense() || entry.is_empty_field() {
            return Ok(Box::new(Numeric {
                reader: PlainValues {
                    data: &producer.data,
                    entry,
                },
                max_doc: self.max_doc,
                doc: -1,
                value: 0,
                cost: i64::from(entry.num_docs_with_field),
            }));
        }
        Ok(Box::new(Numeric {
            reader: norms::NormsReader::new(&producer.data, entry),
            max_doc: self.max_doc,
            doc: -1,
            value: 0,
            cost: i64::from(entry.num_docs_with_field),
        }))
    }
    fn close(&mut self) -> Result<()> {
        self.delegate = None;
        Ok(())
    }
    fn check_integrity(&self) -> Result<()> {
        self.delegate
            .as_deref()
            .ok_or_else(closed)?
            .check_integrity()
    }
    fn get_merge_instance(&self) -> Self {
        Self {
            delegate: self.delegate.clone(),
            max_doc: self.max_doc,
            num_fields: self.num_fields,
        }
    }
}
impl std::fmt::Display for Lucene90NormsProducer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lucene90NormsProducer(fields={})", self.num_fields)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::ByteBuffersDirectory;
    use std::cell::Cell;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn field(number: i32) -> FieldInfo {
        let mut field = FieldInfo::new(format!("norm{number}"), number);
        field.index_options = IndexOptions::Docs;
        field
    }
    fn error() -> lucene_search::Error {
        lucene_store::Error::Io(std::io::Error::other("source failure")).into()
    }
    struct Source {
        gets: Cell<usize>,
        fail_get: usize,
        fail_next: bool,
        docs: Vec<(i32, i64)>,
    }
    impl Source {
        fn sparse() -> Self {
            Self {
                gets: Cell::new(0),
                fail_get: 0,
                fail_next: false,
                docs: vec![(0, 7), (2, 9)],
            }
        }
    }
    struct Cursor<'a> {
        source: &'a Source,
        position: usize,
        doc: i32,
    }
    impl DocIdSetIterator for Cursor<'_> {
        fn doc_id(&self) -> i32 {
            self.doc
        }
        fn next_doc(&mut self) -> Result<i32> {
            if self.source.fail_next {
                return Err(error());
            }
            self.doc = self
                .source
                .docs
                .get(self.position)
                .map_or(NO_MORE_DOCS, |pair| pair.0);
            self.position += 1;
            Ok(self.doc)
        }
        fn advance(&mut self, target: i32) -> Result<i32> {
            while self.next_doc()? < target {}
            Ok(self.doc)
        }
        fn cost(&self) -> i64 {
            self.source.docs.len() as i64
        }
    }
    impl DocValuesIterator for Cursor<'_> {
        fn advance_exact(&mut self, target: i32) -> Result<bool> {
            Ok(self.advance(target)? == target)
        }
    }
    impl NumericDocValues for Cursor<'_> {
        fn long_value(&self) -> i64 {
            self.source.docs[self.position - 1].1
        }
    }
    impl NormsProducer for Source {
        fn get_norms(&self, _: &FieldInfo) -> Result<Box<dyn NumericDocValues + '_>> {
            let gets = self.gets.get() + 1;
            self.gets.set(gets);
            if gets == self.fail_get {
                Err(error())
            } else {
                Ok(Box::new(Cursor {
                    source: self,
                    position: 0,
                    doc: -1,
                }))
            }
        }
        fn close(&mut self) -> Result<()> {
            Ok(())
        }
        fn check_integrity(&self) -> Result<()> {
            Ok(())
        }
        fn get_merge_instance(&self) -> Self {
            Self::sparse()
        }
    }
    #[derive(Default)]
    struct Target {
        adds: usize,
        closes: usize,
    }
    impl NormsConsumer for Target {
        fn add_norms_field(&mut self, _: &FieldInfo, _: &dyn NormsProducer) -> Result<()> {
            self.adds += 1;
            Ok(())
        }
        fn close(&mut self) -> Result<()> {
            self.closes += 1;
            Ok(())
        }
    }

    #[test]
    fn exposed_delegates_control_and_observe_the_same_wrapped_objects() {
        let mut source = AssertingNormsProducer::new(Source::sparse(), 3);
        source.delegate_mut().fail_get = 1;
        assert_eq!(
            source.get_norms(&field(0)).err().unwrap().to_string(),
            "source failure"
        );
        assert_eq!(source.delegate().gets.get(), 1);
        let mut target = AssertingNormsConsumer::new(Target::default(), 3);
        target.delegate_mut().closes = 4;
        target
            .add_norms_field(&field(0), &source.delegate().get_merge_instance())
            .unwrap();
        target.close().unwrap();
        assert_eq!((target.delegate().adds, target.delegate().closes), (1, 6));
        let mut unindexed = field(0);
        unindexed.index_options = IndexOptions::None;
        assert!(catch_unwind(AssertUnwindSafe(|| source.get_norms(&unindexed))).is_err());
        assert_eq!(
            source.delegate().gets.get(),
            1,
            "field check precedes delegate"
        );
    }

    #[test]
    fn native_writer_replays_all_passes_and_propagates_each_source_failure() {
        for fail_get in 1..=3 {
            let directory = ByteBuffersDirectory::new();
            let state = NormsState {
                directory: &directory,
                segment: "_n",
                id: &[0; 16],
                suffix: "",
                max_doc: 3,
            };
            let mut writer = Lucene90NormsFormat.norms_consumer(&state).unwrap();
            let source = Source {
                fail_get,
                ..Source::sparse()
            };
            assert_eq!(
                writer
                    .add_norms_field(&field(0), &source)
                    .unwrap_err()
                    .to_string(),
                "source failure"
            );
            assert_eq!(source.gets.get(), fail_get);
            writer.close().unwrap();
        }
        let directory = ByteBuffersDirectory::new();
        let state = NormsState {
            directory: &directory,
            segment: "_n",
            id: &[0; 16],
            suffix: "",
            max_doc: 3,
        };
        let mut writer = Lucene90NormsFormat.norms_consumer(&state).unwrap();
        let source = Source {
            fail_next: true,
            ..Source::sparse()
        };
        assert_eq!(
            writer
                .add_norms_field(&field(0), &source)
                .unwrap_err()
                .to_string(),
            "source failure"
        );
        let source = Source::sparse();
        writer.add_norms_field(&field(0), &source).unwrap();
        assert_eq!(source.gets.get(), 3);
        writer.close().unwrap();
        assert!(writer.add_norms_field(&field(1), &source).is_err());
        let mut producer = Lucene90NormsFormat
            .norms_producer(
                &state,
                &FieldInfos {
                    fields: vec![field(0)],
                },
            )
            .unwrap();
        assert!(matches!(
            producer.get_norms(&field(9)).err(),
            Some(lucene_search::Error::Norms(
                norms::Error::UnknownFieldNumber(9)
            ))
        ));
        let mut cursor = producer.get_norms(&field(0)).unwrap();
        assert_eq!(cursor.advance(1).unwrap(), 2);
        assert_eq!(cursor.long_value(), 9);
        assert!(!cursor.advance_exact(1).unwrap());
        assert_eq!(cursor.doc_id(), 1);
        assert_eq!(cursor.next_doc().unwrap(), 2);
        assert_eq!(cursor.long_value(), 9);
        assert!(cursor.advance_exact(2).unwrap());
        assert_eq!(cursor.long_value(), 9);
        assert!(cursor.advance_exact(0).unwrap());
        assert_eq!(cursor.long_value(), 7);
        assert!(cursor.advance_exact(-1).is_err());
        assert!(cursor.advance_exact(3).is_err());
        assert_eq!(cursor.advance(3).unwrap(), NO_MORE_DOCS);
        drop(cursor);
        let merging = producer.get_merge_instance();
        producer.close().unwrap();
        producer.close().unwrap();
        assert_eq!(producer.to_string(), "Lucene90NormsProducer(fields=1)");
        let mut independent = merging.get_norms(&field(0)).unwrap();
        assert_eq!(independent.next_doc().unwrap(), 0);
        assert_eq!(independent.long_value(), 7);
        merging.check_integrity().unwrap();
        let clone = producer.get_merge_instance();
        assert_eq!(
            clone.get_norms(&field(0)).err().unwrap().to_string(),
            "closed norms producer"
        );
        assert_eq!(
            producer.check_integrity().unwrap_err().to_string(),
            "closed norms producer"
        );
    }

    #[test]
    fn native_open_and_close_errors_are_returned() {
        let directory = ByteBuffersDirectory::new();
        let state = NormsState {
            directory: &directory,
            segment: "_n",
            id: &[0; 16],
            suffix: "",
            max_doc: 3,
        };
        assert!(Lucene90NormsFormat
            .norms_producer(&state, &FieldInfos { fields: vec![] })
            .is_err());
        let mut writer = Lucene90NormsFormat.norms_consumer(&state).unwrap();
        let invalid = Source {
            docs: vec![(3, 1)],
            ..Source::sparse()
        };
        writer.add_norms_field(&field(0), &invalid).unwrap();
        assert!(writer.close().is_err());
        writer.close().unwrap();
    }

    #[test]
    fn metadata_field_errors_precede_opening_missing_data() {
        use lucene_store::{DataOutput, Directory};
        let directory = ByteBuffersDirectory::new();
        let (metadata, _) =
            norms::write_fields(&[NormsField::Dense(0, &[1, 2, 3])], 3, &[0; 16], "").unwrap();
        let mut output = directory.create_output("_n.nvm").unwrap();
        output.write_bytes(&metadata);
        output.close().unwrap();
        let state = NormsState {
            directory: &directory,
            segment: "_n",
            id: &[0; 16],
            suffix: "",
            max_doc: 3,
        };
        let unknown = Lucene90NormsFormat
            .norms_producer(&state, &FieldInfos { fields: vec![] })
            .err()
            .unwrap();
        assert!(matches!(
            unknown,
            lucene_search::Error::Norms(norms::Error::UnknownFieldNumber(0))
        ));
        let mut omitted = field(0);
        omitted.omit_norms = true;
        let invalid = Lucene90NormsFormat
            .norms_producer(
                &state,
                &FieldInfos {
                    fields: vec![omitted],
                },
            )
            .err()
            .unwrap();
        assert!(matches!(
            invalid,
            lucene_search::Error::Norms(norms::Error::FieldHasNoNorms(_))
        ));
        assert!(Lucene90NormsFormat
            .norms_producer(
                &state,
                &FieldInfos {
                    fields: vec![field(0)]
                }
            )
            .is_err());
    }

    #[test]
    fn incomplete_native_data_returns_an_integrity_error() {
        let incomplete = NativeNorms {
            metadata: norms::Norms { entries: vec![] },
            data: vec![],
        };
        assert_eq!(
            incomplete.check_integrity().unwrap_err().to_string(),
            "missing norms data footer"
        );
    }
}
