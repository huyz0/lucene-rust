//! Asserting norms consumers installed in the native codec's write path.
//!
//! Borrowed dense/sparse columns supply a fresh iterator on every acquisition.
//! The existing asserting consumer forwards that same producer to its native
//! delegate, including its repeated statistics/value passes and double close.
//! The native reader graph and `Lucene104` name remain fixed. This adapter does
//! not install reader wrappers, whole-codec composition or IOContext routing.

use std::io;

use lucene_codecs::codec::Lucene104Codec;
use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::norms::{self, NormsField};
use lucene_codecs::norms_writer::{NormsWriteSink, NormsWriteState, NormsWriterFactory};
use lucene_search::reader::{DocIdSetIterator, DocValuesIterator, NumericDocValues, NO_MORE_DOCS};
use lucene_search::Result;

use super::asserting_norms::{
    AssertingNormsConsumer, AssertingNormsFormat, Lucene90NormsFormat, NormsConsumer, NormsFormat,
    NormsProducer,
};
use super::cranky_norms::NormsState;

/// Native `Lucene104` routing with assertions on each norms consumer.
#[derive(Debug, Default, Clone, Copy)]
pub struct AssertingNormsCodec;

impl Lucene104Codec for AssertingNormsCodec {
    fn norms_writer_factory(&self) -> Option<&dyn NormsWriterFactory> {
        Some(self)
    }
}

fn lower_error(error: lucene_search::Error) -> norms::Error {
    match error {
        lucene_search::Error::Norms(error) => error,
        lucene_search::Error::Store(error) => error.into(),
        other => lucene_store::Error::Io(io::Error::other(other)).into(),
    }
}

impl NormsWriterFactory for AssertingNormsCodec {
    fn norms_consumer<'a>(
        &self,
        state: NormsWriteState<'a>,
    ) -> norms::Result<Box<dyn NormsWriteSink + 'a>> {
        if !state.suffix.is_empty() {
            return Err(lucene_store::Error::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "asserting norms routing requires an empty suffix",
            ))
            .into());
        }
        let consumer = AssertingNormsFormat::new(Lucene90NormsFormat)
            .norms_consumer(&NormsState {
                directory: state.directory,
                segment: state.segment,
                id: state.id,
                suffix: state.suffix,
                max_doc: state.max_doc,
            })
            .map_err(lower_error)?;
        Ok(Box::new(NativeSink {
            consumer,
            max_doc: state.max_doc,
        }))
    }
}

struct NativeSink<C> {
    consumer: AssertingNormsConsumer<C>,
    max_doc: i32,
}

impl<C: NormsConsumer> NormsWriteSink for NativeSink<C> {
    fn add_norms_field(&mut self, field: &FieldInfo, values: NormsField<'_>) -> norms::Result<()> {
        // The lower Dense contract is a complete contiguous column. Sparse
        // iteration deliberately retains caller order for the existing guard.
        if let NormsField::Dense(_, values) = values {
            assert_eq!(usize::try_from(self.max_doc).ok(), Some(values.len()));
        }
        self.consumer
            .add_norms_field(
                field,
                &ColumnProducer {
                    values,
                    max_doc: self.max_doc,
                },
            )
            .map_err(lower_error)
    }
    fn close(&mut self) -> norms::Result<()> {
        self.consumer.close().map_err(lower_error)
    }
}

#[derive(Clone, Copy)]
struct ColumnProducer<'a> {
    values: NormsField<'a>,
    max_doc: i32,
}

impl NormsProducer for ColumnProducer<'_> {
    fn get_norms(&self, field: &FieldInfo) -> Result<Box<dyn NumericDocValues + '_>> {
        let number = match self.values {
            NormsField::Dense(n, _) | NormsField::Sparse(n, _) => n,
        };
        if field.number != number {
            return Err(norms::Error::UnknownFieldNumber(field.number).into());
        }
        if field.index_options == IndexOptions::None || field.omit_norms {
            return Err(norms::Error::FieldHasNoNorms(field.name.clone()).into());
        }
        Ok(Box::new(ColumnValues {
            values: self.values,
            max_doc: self.max_doc,
            next: 0,
            doc: -1,
            value: 0,
        }))
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
    fn check_integrity(&self) -> Result<()> {
        Ok(())
    }
    fn get_merge_instance(&self) -> Self {
        *self
    }
}

struct ColumnValues<'a> {
    values: NormsField<'a>,
    max_doc: i32,
    next: usize,
    doc: i32,
    value: i64,
}

impl DocIdSetIterator for ColumnValues<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        let value = match self.values {
            NormsField::Dense(_, values) => values.get(self.next).map(|&v| (self.next as i32, v)),
            NormsField::Sparse(_, values) => values.get(self.next).copied(),
        };
        self.next = self.next.saturating_add(1);
        (self.doc, self.value) = value.unwrap_or((NO_MORE_DOCS, 0));
        Ok(self.doc)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        while self.next_doc()? < target {}
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        let len = match self.values {
            NormsField::Dense(_, v) => v.len(),
            NormsField::Sparse(_, v) => v.len(),
        };
        i64::try_from(len).unwrap_or(i64::MAX)
    }
}
impl DocValuesIterator for ColumnValues<'_> {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        if target < 0 || target >= self.max_doc {
            return Err(norms::Error::DocOutOfRange(target, self.max_doc).into());
        }
        // Forward iterator state remains just past a matching value, or at
        // the first later sparse entry when the target is absent.
        let (next, value) = match self.values {
            NormsField::Dense(_, values) => {
                let at = target as usize;
                (at.saturating_add(1), values.get(at).copied())
            }
            NormsField::Sparse(_, values) => {
                let at = values.partition_point(|(doc, _)| *doc < target);
                match values.get(at).filter(|(doc, _)| *doc == target) {
                    Some((_, value)) => (at.saturating_add(1), Some(*value)),
                    None => (at, None),
                }
            }
        };
        self.next = next;
        self.doc = target;
        self.value = value.unwrap_or(0);
        Ok(value.is_some())
    }
}
impl NumericDocValues for ColumnValues<'_> {
    fn long_value(&self) -> i64 {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::rc::Rc;

    use lucene_codecs::norms_writer::with_writer;
    use lucene_store::{ByteBuffersDirectory, Directory};

    use super::*;

    const ID: [u8; 16] = [51; 16];

    fn field(number: i32) -> FieldInfo {
        let mut info = FieldInfo::new(format!("f{number}"), number);
        info.index_options = IndexOptions::Docs;
        info
    }
    fn state(directory: &dyn Directory) -> NormsWriteState<'_> {
        NormsWriteState {
            directory,
            segment: "_test",
            id: &ID,
            suffix: "",
            max_doc: 4,
        }
    }
    fn io_kind(error: norms::Error) -> io::ErrorKind {
        match error {
            norms::Error::Store(lucene_store::Error::Io(error)) => error.kind(),
            other => panic!("unexpected error {other}"),
        }
    }

    #[test]
    fn installed_consumer_writes_native_dense_sparse_empty_and_zero_values() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AssertingNormsCodec>();
        let codec: &dyn Lucene104Codec = &AssertingNormsCodec;
        assert_eq!(codec.name(), "Lucene104");
        assert!(codec.live_docs_writer().is_none());
        assert!(codec.stored_fields_writer_factory().is_none());
        let directory = ByteBuffersDirectory::new();
        let columns = [
            NormsField::Dense(0, &[7, 7, 7, 7]),
            NormsField::Dense(1, &[0, -129, 32768, i64::MAX]),
            NormsField::Sparse(2, &[(0, 0), (3, 19)]),
            NormsField::Sparse(3, &[]),
        ];
        let result: norms::Result<()> = with_writer(
            codec.norms_writer_factory().unwrap(),
            state(&directory),
            |sink| {
                for (number, values) in (0..).zip(columns) {
                    sink.add_norms_field(&field(number), values)?;
                }
                Ok(())
            },
        );
        result.unwrap();
        let expected = norms::write_fields(&columns, 4, &ID, "").unwrap();
        let meta = directory.open("_test.nvm").unwrap();
        let data = directory.open("_test.nvd").unwrap();
        assert_eq!(&*meta, expected.0);
        assert_eq!(&*data, expected.1);
        let (_, metadata) = norms::parse_meta(&meta, &ID, "").unwrap();
        norms::check_data_header_footer(&data, &ID, "").unwrap();
        assert_eq!(
            norms::norm_value(&data, &metadata.entries[2], 0).unwrap(),
            Some(0)
        );
        assert_eq!(
            norms::norm_value(&data, &metadata.entries[2], 1).unwrap(),
            None
        );
        assert_eq!(
            norms::norm_value(&data, &metadata.entries[3], 0).unwrap(),
            None
        );
    }

    #[test]
    fn empty_consumer_and_repeated_close_write_valid_native_headers() {
        let directory = ByteBuffersDirectory::new();
        let mut sink = AssertingNormsCodec
            .norms_consumer(state(&directory))
            .unwrap();
        sink.close().unwrap();
        sink.close().unwrap();
        let meta = directory.open("_test.nvm").unwrap();
        let data = directory.open("_test.nvd").unwrap();
        let (_, metadata) = norms::parse_meta(&meta, &ID, "").unwrap();
        assert!(metadata.entries.is_empty());
        norms::check_data_header_footer(&data, &ID, "").unwrap();
        let error = sink
            .add_norms_field(&field(0), NormsField::Dense(0, &[1; 4]))
            .unwrap_err();
        assert_eq!(error.to_string(), "closed consumer");
    }

    #[test]
    fn suffix_and_output_open_failures_preserve_typed_errors() {
        let directory = ByteBuffersDirectory::new();
        let mut write_state = state(&directory);
        write_state.suffix = "s";
        let error = AssertingNormsCodec
            .norms_consumer(write_state)
            .err()
            .unwrap();
        assert_eq!(io_kind(error), io::ErrorKind::InvalidInput);
        assert!(directory.list_all().unwrap().is_empty());
        // The native delegate opens data before metadata. Collision on either
        // file remains AlreadyExists, including its failed-constructor path.
        for name in ["_test.nvd", "_test.nvm"] {
            let directory = ByteBuffersDirectory::new();
            directory.create_output(name).unwrap().close().unwrap();
            let error = AssertingNormsCodec
                .norms_consumer(state(&directory))
                .err()
                .unwrap();
            assert_eq!(io_kind(error), io::ErrorKind::AlreadyExists);
        }
    }

    #[test]
    fn fields_and_columns_keep_their_identity_and_bounds() {
        let directory = ByteBuffersDirectory::new();
        let mut sink = AssertingNormsCodec
            .norms_consumer(state(&directory))
            .unwrap();
        let error = sink
            .add_norms_field(&field(1), NormsField::Sparse(0, &[]))
            .unwrap_err();
        assert!(matches!(error, norms::Error::UnknownFieldNumber(1)));
        let mut no_norms = field(0);
        for omitted in [false, true] {
            no_norms.index_options = if omitted {
                IndexOptions::Docs
            } else {
                IndexOptions::None
            };
            no_norms.omit_norms = omitted;
            assert!(matches!(
                sink.add_norms_field(&no_norms, NormsField::Sparse(0, &[])),
                Err(norms::Error::FieldHasNoNorms(_))
            ));
        }
        for values in [
            NormsField::Dense(0, &[1]),
            NormsField::Sparse(0, &[(-1, 1)]),
            NormsField::Sparse(0, &[(4, 1)]),
            NormsField::Sparse(0, &[(1, 1), (1, 2)]),
            NormsField::Sparse(0, &[(2, 1), (0, 2)]),
        ] {
            assert!(
                catch_unwind(AssertUnwindSafe(|| sink.add_norms_field(&field(0), values))).is_err()
            );
        }
        sink.close().unwrap();
    }

    #[test]
    fn fresh_column_iterators_support_forward_and_exact_access() {
        let field = field(0);
        for values in [
            NormsField::Dense(0, &[0, 11, 12, 13]),
            NormsField::Sparse(0, &[(0, 0), (2, 12), (3, 13)]),
        ] {
            let mut source = ColumnProducer { values, max_doc: 4 };
            source.check_integrity().unwrap();
            for _ in 0..3 {
                let merge = source.get_merge_instance();
                let mut iterator = merge.get_norms(&field).unwrap();
                assert_eq!(iterator.doc_id(), -1);
                assert!(iterator.cost() >= 3);
                assert_eq!(iterator.next_doc().unwrap(), 0);
                assert_eq!(iterator.long_value(), 0);
                assert_eq!(iterator.advance(2).unwrap(), 2);
                assert_eq!(iterator.long_value(), 12);
                assert_eq!(iterator.next_doc().unwrap(), 3);
                assert_eq!(iterator.next_doc().unwrap(), NO_MORE_DOCS);
                assert_eq!(iterator.next_doc().unwrap(), NO_MORE_DOCS);
            }
            let mut iterator = source.get_norms(&field).unwrap();
            assert!(iterator.advance_exact(0).unwrap());
            assert_eq!(iterator.long_value(), 0);
            let present = iterator.advance_exact(1).unwrap();
            assert_eq!(present, matches!(values, NormsField::Dense(..)));
            assert_eq!(iterator.next_doc().unwrap(), 2);
            assert!(iterator.advance_exact(3).unwrap());
            assert_eq!(iterator.long_value(), 13);
            assert!(matches!(
                iterator.advance_exact(-1),
                Err(lucene_search::Error::Norms(norms::Error::DocOutOfRange(
                    -1, 4
                )))
            ));
            assert!(iterator.advance_exact(4).is_err());
            drop(iterator);
            source.close().unwrap();
        }
    }

    struct Recording {
        closes: Rc<RefCell<usize>>,
        fail_close: usize,
    }
    impl NormsConsumer for Recording {
        fn add_norms_field(
            &mut self,
            field: &FieldInfo,
            producer: &dyn NormsProducer,
        ) -> Result<()> {
            for _ in 0..3 {
                let mut values = producer.get_norms(field)?;
                assert_eq!(values.doc_id(), -1);
                assert_eq!(values.next_doc()?, 1);
                assert_eq!(values.long_value(), 0);
                assert_eq!(values.next_doc()?, 3);
                assert_eq!(values.long_value(), 19);
                assert_eq!(values.next_doc()?, NO_MORE_DOCS);
            }
            Ok(())
        }
        fn close(&mut self) -> Result<()> {
            let mut closes = self.closes.borrow_mut();
            *closes += 1;
            if *closes == self.fail_close {
                Err(lucene_store::Error::Io(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "close failed",
                ))
                .into())
            } else {
                Ok(())
            }
        }
    }
    #[test]
    fn existing_wrapper_forwards_repeatable_source_and_double_close_short_circuits() {
        for fail_close in [0, 1, 2] {
            let closes = Rc::new(RefCell::new(0));
            let mut sink = NativeSink {
                consumer: AssertingNormsConsumer::new(
                    Recording {
                        closes: closes.clone(),
                        fail_close,
                    },
                    4,
                ),
                max_doc: 4,
            };
            sink.add_norms_field(&field(0), NormsField::Sparse(0, &[(1, 0), (3, 19)]))
                .unwrap();
            let result = sink.close();
            if fail_close == 0 {
                result.unwrap();
            } else {
                assert_eq!(io_kind(result.unwrap_err()), io::ErrorKind::BrokenPipe);
            }
            assert_eq!(*closes.borrow(), if fail_close == 1 { 1 } else { 2 });
        }
    }

    #[test]
    fn lower_error_retains_norms_and_store_types_and_wraps_other_sources() {
        assert!(matches!(
            lower_error(norms::Error::UnknownFieldNumber(9).into()),
            norms::Error::UnknownFieldNumber(9)
        ));
        assert_eq!(
            io_kind(lower_error(
                lucene_store::Error::Io(io::Error::new(io::ErrorKind::PermissionDenied, "denied"))
                    .into()
            )),
            io::ErrorKind::PermissionDenied
        );
        let error = lower_error(lucene_search::Error::MissingPosInput);
        match error {
            norms::Error::Store(lucene_store::Error::Io(error)) => {
                assert_eq!(error.kind(), io::ErrorKind::Other);
                assert!(error
                    .get_ref()
                    .unwrap()
                    .downcast_ref::<lucene_search::Error>()
                    .is_some());
            }
            other => panic!("unexpected {other}"),
        }
    }
}
