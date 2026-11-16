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
mod tests;
