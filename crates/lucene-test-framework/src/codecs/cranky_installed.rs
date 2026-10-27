//! Java `CrankyCodec` installed in [`lucene_index::index_writer::IndexWriter`]
//! (`IndexWriter::set_codec`): the segment keeps its delegate's name and
//! files (`Lucene104`), and its stored-fields, term-vectors, norms, points,
//! postings and live-docs writers fail with Java's fake `IOException`s at
//! Java's draws (`random.nextInt(100) == 0` and friends), so the writer's
//! failure paths run against a real index. Its doc-values, field-infos,
//! segment-info and compound writes fail the same way through
//! [`Lucene104Codec::before_write`].
//!
//! The guards and their bounds are those of the explicit adapters
//! ([`super::cranky_stored_fields`], [`super::cranky_term_vectors`],
//! [`super::cranky_norms`], [`super::cranky_points`],
//! [`super::cranky_postings`], [`super::cranky_live_docs`]); a failing
//! close still closes its delegate first. The writers under them are the
//! native term-vectors, points and postings writers and, where this port
//! exports no plain native factory, the same-wire asserting stored-fields,
//! norms and live-docs writers.
//!
//! Differences from Java: one `Mutex<JavaRandom>` replaces the shared
//! `Random`, so the codec is `Send + Sync` and draws in the order this
//! port's writer calls its sinks (not Java's `IndexingChain` order); the
//! doc-values, field-infos, segment-info and compound files are the native
//! writer's, failed before or around their writes rather than inside a
//! wrapped format.

use std::io;
use std::sync::{Arc, Mutex};

use lucene_codecs::codec::{Lucene104Codec, WritePoint};
use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::live_docs_writer::{LiveDocsWriteState, LiveDocsWriter};
use lucene_codecs::norms::{self, NormsField};
use lucene_codecs::norms_writer::{NormsWriteSink, NormsWriteState, NormsWriterFactory};
use lucene_codecs::points::{self, MergePointsField, PackedPointsField};
use lucene_codecs::points_writer::{
    NativePointsWriter, PointsWriteSink, PointsWriteState, PointsWriterFactory,
};
use lucene_codecs::postings_consumer::{
    NativePostingsWriter, PostingsWriteSink, PostingsWriteState, PostingsWriterFactory,
};
use lucene_codecs::postings_writer::{self, FieldNorms, FieldPostingsInput, TermSource};
use lucene_codecs::stored_fields::{self, FieldValue};
use lucene_codecs::stored_fields_writer::{
    StoredFieldsWriteSink, StoredFieldsWriteState, StoredFieldsWriterFactory,
};
use lucene_codecs::term_vectors;
use lucene_codecs::term_vectors_writer::{
    NativeTermVectorsWriter, TermVectorsWriteSink, TermVectorsWriteState, TermVectorsWriterFactory,
};
use lucene_store::data_input::DataInput;
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::java_random::JavaRandom;

use super::asserting_live_docs_writer::AssertingLiveDocsCodec;
use super::asserting_norms_writer::AssertingNormsCodec;
use super::asserting_stored_fields_writer::AssertingStoredFieldsCodec;
use super::{
    cranky_compound, cranky_field_infos, cranky_norms, cranky_points, cranky_postings,
    cranky_stored_fields,
};

type Random = Arc<Mutex<JavaRandom>>;

/// `random.nextInt(bound) == 0`.
fn draw(random: &Random, bound: i32) -> bool {
    random
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .next_int_bounded(bound)
        == 0
}

fn fake(message: impl Into<String>) -> lucene_store::Error {
    lucene_store::Error::Io(io::Error::other(message.into()))
}

/// `new CrankyCodec(TestUtil.getDefaultCodec(), random)` through
/// IndexWriter.
#[derive(Debug, Clone)]
pub struct InstalledCrankyCodec {
    random: Random,
}

impl InstalledCrankyCodec {
    /// A codec whose faults draw from `new Random(seed)`.
    pub fn new(seed: i64) -> Self {
        InstalledCrankyCodec {
            random: Arc::new(Mutex::new(JavaRandom::new(seed))),
        }
    }
}

impl Lucene104Codec for InstalledCrankyCodec {
    fn stored_fields_writer_factory(&self) -> Option<&dyn StoredFieldsWriterFactory> {
        Some(self)
    }

    fn live_docs_writer(&self) -> Option<&dyn LiveDocsWriter> {
        Some(self)
    }

    fn norms_writer_factory(&self) -> Option<&dyn NormsWriterFactory> {
        Some(self)
    }

    fn points_writer_factory(&self) -> Option<&dyn PointsWriterFactory> {
        Some(self)
    }

    fn postings_writer_factory(&self) -> Option<&dyn PostingsWriterFactory> {
        Some(self)
    }

    fn term_vectors_writer_factory(&self) -> Option<&dyn TermVectorsWriterFactory> {
        Some(self)
    }

    /// `CrankyDocValuesFormat`, `CrankyFieldInfosFormat`,
    /// `CrankySegmentInfoFormat` and `CrankyCompoundFormat`: one
    /// `nextInt(100)` draw at each write. The doc-values field draw cannot
    /// tell the `add*Field` overloads apart and names the numeric one.
    fn before_write(&self, point: WritePoint) -> io::Result<()> {
        if !draw(&self.random, 100) {
            return Ok(());
        }
        let message = match point {
            WritePoint::DocValuesConsumer => {
                "Fake IOException from DocValuesFormat.fieldsConsumer()"
            }
            WritePoint::DocValuesField => {
                "Fake IOException from DocValuesConsumer.addNumericField()"
            }
            WritePoint::DocValuesClose => "Fake IOException from DocValuesConsumer.close()",
            WritePoint::FieldInfos => cranky_field_infos::FAKE_WRITE_ERROR,
            WritePoint::SegmentInfo => "Fake IOException from SegmentInfoFormat.write()",
            WritePoint::Compound => cranky_compound::FAKE_WRITE_ERROR,
        };
        Err(io::Error::other(message))
    }
}

impl StoredFieldsWriterFactory for InstalledCrankyCodec {
    fn fields_writer<'a>(
        &self,
        state: StoredFieldsWriteState<'a>,
    ) -> stored_fields::Result<Box<dyn StoredFieldsWriteSink + 'a>> {
        if draw(&self.random, 100) {
            return Err(fake(cranky_stored_fields::FAKE_OPEN_ERROR).into());
        }
        Ok(Box::new(StoredFields {
            delegate: AssertingStoredFieldsCodec.fields_writer(state)?,
            random: self.random.clone(),
        }))
    }
}

struct StoredFields<'a> {
    delegate: Box<dyn StoredFieldsWriteSink + 'a>,
    random: Random,
}

impl StoredFields<'_> {
    fn guard(&self, bound: i32, message: &str) -> stored_fields::Result<()> {
        if draw(&self.random, bound) {
            return Err(fake(message).into());
        }
        Ok(())
    }
}

impl StoredFieldsWriteSink for StoredFields<'_> {
    fn start_document(&mut self) -> stored_fields::Result<()> {
        self.guard(10000, cranky_stored_fields::FAKE_START_ERROR)?;
        self.delegate.start_document()
    }
    fn write_field(&mut self, number: i32, value: &FieldValue) -> stored_fields::Result<()> {
        self.guard(10000, cranky_stored_fields::FAKE_FIELD_ERROR)?;
        self.delegate.write_field(number, value)
    }
    fn write_field_data_input(
        &mut self,
        number: i32,
        input: &mut dyn DataInput,
        length: usize,
    ) -> stored_fields::Result<()> {
        self.guard(10000, cranky_stored_fields::FAKE_FIELD_ERROR)?;
        self.delegate.write_field_data_input(number, input, length)
    }
    fn finish_document(&mut self) -> stored_fields::Result<()> {
        self.guard(10000, cranky_stored_fields::FAKE_FINISH_DOCUMENT_ERROR)?;
        self.delegate.finish_document()
    }
    fn finish(&mut self, num_docs: i32) -> stored_fields::Result<()> {
        self.guard(100, cranky_stored_fields::FAKE_FINISH_ERROR)?;
        self.delegate.finish(num_docs)
    }
    fn close(&mut self) -> stored_fields::Result<()> {
        self.delegate.close()?;
        self.guard(1000, cranky_stored_fields::FAKE_CLOSE_ERROR)
    }
}

impl TermVectorsWriterFactory for InstalledCrankyCodec {
    fn vectors_writer<'a>(
        &self,
        state: TermVectorsWriteState<'a>,
    ) -> term_vectors::Result<Box<dyn TermVectorsWriteSink + 'a>> {
        if draw(&self.random, 100) {
            return Err(fake("Fake IOException from TermVectorsFormat.vectorsWriter()").into());
        }
        Ok(Box::new(TermVectors {
            delegate: NativeTermVectorsWriter::new(state)?,
            random: self.random.clone(),
        }))
    }
}

struct TermVectors<W> {
    delegate: W,
    random: Random,
}

impl<W> TermVectors<W> {
    fn guard(&self, bound: i32, operation: &str) -> term_vectors::Result<()> {
        if draw(&self.random, bound) {
            return Err(fake(format!(
                "Fake IOException from TermVectorsWriter.{operation}()"
            ))
            .into());
        }
        Ok(())
    }
}

impl<W: TermVectorsWriteSink> TermVectorsWriteSink for TermVectors<W> {
    fn start_document(&mut self, fields: i32) -> term_vectors::Result<()> {
        self.guard(10000, "startDocument")?;
        self.delegate.start_document(fields)
    }
    fn finish_document(&mut self) -> term_vectors::Result<()> {
        self.guard(10000, "finishDocument")?;
        self.delegate.finish_document()
    }
    fn start_field(
        &mut self,
        info: &FieldInfo,
        terms: i32,
        positions: bool,
        offsets: bool,
        payloads: bool,
    ) -> term_vectors::Result<()> {
        self.guard(10000, "startField")?;
        self.delegate
            .start_field(info, terms, positions, offsets, payloads)
    }
    fn finish_field(&mut self) -> term_vectors::Result<()> {
        self.guard(10000, "finishField")?;
        self.delegate.finish_field()
    }
    fn start_term(&mut self, term: &[u8], frequency: i32) -> term_vectors::Result<()> {
        self.guard(10000, "startTerm")?;
        self.delegate.start_term(term, frequency)
    }
    fn finish_term(&mut self) -> term_vectors::Result<()> {
        self.guard(10000, "finishTerm")?;
        self.delegate.finish_term()
    }
    fn add_position(
        &mut self,
        position: i32,
        start: i32,
        end: i32,
        payload: Option<&[u8]>,
    ) -> term_vectors::Result<()> {
        self.guard(10000, "addPosition")?;
        self.delegate.add_position(position, start, end, payload)
    }
    fn finish(&mut self, documents: i32) -> term_vectors::Result<()> {
        self.guard(100, "finish")?;
        self.delegate.finish(documents)
    }
    fn close(&mut self) -> term_vectors::Result<()> {
        self.delegate.close()?;
        self.guard(100, "close")
    }
    fn ram_bytes_used(&self) -> usize {
        self.delegate.ram_bytes_used()
    }
}

impl NormsWriterFactory for InstalledCrankyCodec {
    fn norms_consumer<'a>(
        &self,
        state: NormsWriteState<'a>,
    ) -> norms::Result<Box<dyn NormsWriteSink + 'a>> {
        if draw(&self.random, 100) {
            return Err(fake(cranky_norms::FAKE_OPEN_ERROR).into());
        }
        Ok(Box::new(Norms {
            delegate: AssertingNormsCodec.norms_consumer(state)?,
            random: self.random.clone(),
        }))
    }
}

struct Norms<'a> {
    delegate: Box<dyn NormsWriteSink + 'a>,
    random: Random,
}

impl NormsWriteSink for Norms<'_> {
    fn add_norms_field(&mut self, field: &FieldInfo, values: NormsField<'_>) -> norms::Result<()> {
        if draw(&self.random, 100) {
            return Err(fake(cranky_norms::FAKE_ADD_ERROR).into());
        }
        self.delegate.add_norms_field(field, values)
    }
    fn close(&mut self) -> norms::Result<()> {
        self.delegate.close()?;
        if draw(&self.random, 100) {
            return Err(fake(cranky_norms::FAKE_CLOSE_ERROR).into());
        }
        Ok(())
    }
}

impl PointsWriterFactory for InstalledCrankyCodec {
    fn fields_writer<'a>(
        &self,
        state: PointsWriteState<'a>,
    ) -> points::Result<Box<dyn PointsWriteSink + 'a>> {
        Ok(Box::new(Points {
            delegate: NativePointsWriter::new(state)?,
            random: self.random.clone(),
        }))
    }
}

struct Points<W> {
    delegate: W,
    random: Random,
}

impl<W> Points<W> {
    fn guard(&self) -> points::Result<()> {
        if draw(&self.random, 100) {
            return Err(fake(cranky_points::FAKE_ERROR).into());
        }
        Ok(())
    }
}

impl<W: PointsWriteSink> PointsWriteSink for Points<W> {
    fn write_field(&mut self, field: &FieldInfo, values: PackedPointsField) -> points::Result<()> {
        self.guard()?;
        self.delegate.write_field(field, values)
    }
    fn merge(&mut self, fields: Vec<MergePointsField>, native_readers: bool) -> points::Result<()> {
        self.guard()?;
        self.delegate.merge(fields, native_readers)?;
        self.guard()
    }
    fn finish(&mut self) -> points::Result<()> {
        self.guard()?;
        self.delegate.finish()?;
        self.guard()
    }
    fn close(&mut self) -> points::Result<()> {
        self.delegate.close()?;
        self.guard()
    }
}

impl PostingsWriterFactory for InstalledCrankyCodec {
    fn fields_consumer<'a>(
        &self,
        state: PostingsWriteState<'a>,
    ) -> postings_writer::Result<Box<dyn PostingsWriteSink + 'a>> {
        if draw(&self.random, 100) {
            return Err(fake(cranky_postings::FAKE_OPEN_ERROR).into());
        }
        Ok(Box::new(Postings {
            delegate: NativePostingsWriter::new(state)?,
            random: self.random.clone(),
        }))
    }
}

struct Postings<W> {
    delegate: W,
    random: Random,
}

impl<W: PostingsWriteSink> PostingsWriteSink for Postings<W> {
    fn write(
        &mut self,
        inputs: &[FieldPostingsInput<'_, &dyn TermSource>],
        norms: &[FieldNorms<'_>],
    ) -> postings_writer::Result<()> {
        if draw(&self.random, 100) {
            return Err(fake(cranky_postings::FAKE_WRITE_ERROR).into());
        }
        self.delegate.write(inputs, norms)
    }
    // `FieldsConsumer.merge` writes through `write`, whose guard it shares.
    fn merge(
        &mut self,
        inputs: &[FieldPostingsInput<'_, &dyn TermSource>],
        norms: &[FieldNorms<'_>],
    ) -> postings_writer::Result<()> {
        if draw(&self.random, 100) {
            return Err(fake(cranky_postings::FAKE_WRITE_ERROR).into());
        }
        self.delegate.merge(inputs, norms)
    }
    fn close(&mut self) -> postings_writer::Result<()> {
        self.delegate.close()?;
        if draw(&self.random, 100) {
            return Err(fake(cranky_postings::FAKE_CLOSE_ERROR).into());
        }
        Ok(())
    }
}

impl LiveDocsWriter for InstalledCrankyCodec {
    fn write_live_docs(
        &self,
        bits: &FixedBitSet,
        state: LiveDocsWriteState<'_>,
        new_del_count: usize,
    ) -> io::Result<()> {
        if draw(&self.random, 100) {
            return Err(io::Error::other(
                "Fake IOException from LiveDocsFormat.writeLiveDocs()",
            ));
        }
        AssertingLiveDocsCodec.write_live_docs(bits, state, new_del_count)
    }
}
