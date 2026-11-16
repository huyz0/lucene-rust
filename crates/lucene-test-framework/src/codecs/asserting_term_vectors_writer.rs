//! Asserting term-vector callbacks installed in the native codec's write path.
//!
//! The existing Java writer state machine wraps a native Lucene90 binary sink.
//! Assertion state changes only after successful delegation; successful close
//! calls the delegate twice, stopping if the first close fails. Native errors
//! retain their concrete variants through the framework's I/O callback seam.
//! The native reader graph, `Lucene104` name and binary format remain fixed.
//! Native chunk buffering, publication and accounting limits remain unchanged;
//! this adapter does not install lazy reader assertions or IOContext routing.

use std::io;

use lucene_codecs::codec::Lucene104Codec;
use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::term_vectors;
use lucene_codecs::term_vectors_writer::{
    NativeTermVectorsWriter, TermVectorsWriteSink, TermVectorsWriteState, TermVectorsWriterFactory,
};

use super::asserting_term_vectors::AssertingTermVectorsWriter;
use super::cranky_term_vectors::TermVectorsWriter;

/// Native `Lucene104` routing with assertions on term-vector writer callbacks.
#[derive(Debug, Default, Clone, Copy)]
pub struct AssertingTermVectorsCodec;

impl Lucene104Codec for AssertingTermVectorsCodec {
    fn term_vectors_writer_factory(&self) -> Option<&dyn TermVectorsWriterFactory> {
        Some(self)
    }
}

impl TermVectorsWriterFactory for AssertingTermVectorsCodec {
    fn vectors_writer<'a>(
        &self,
        state: TermVectorsWriteState<'a>,
    ) -> term_vectors::Result<Box<dyn TermVectorsWriteSink + 'a>> {
        Ok(Box::new(AssertingSink::new(NativeTermVectorsWriter::new(
            state,
        )?)))
    }
}

fn framework_error(error: term_vectors::Error) -> io::Error {
    // Preserve the native variant, including its underlying I/O error, rather
    // than flattening it into a message while crossing the framework seam.
    io::Error::other(error)
}

fn native_error(error: io::Error) -> term_vectors::Error {
    if error
        .get_ref()
        .is_some_and(|cause| cause.is::<term_vectors::Error>())
    {
        // Both unwraps follow the concrete type check above. Consuming the
        // wrapper returns the exact native error that entered this bridge.
        *error
            .into_inner()
            .unwrap()
            .downcast::<term_vectors::Error>()
            .unwrap()
    } else {
        lucene_store::Error::Io(error).into()
    }
}

struct FrameworkWriter<W>(W);

impl<W: TermVectorsWriteSink> TermVectorsWriter for FrameworkWriter<W> {
    fn start_document(&mut self, fields: i32) -> io::Result<()> {
        self.0.start_document(fields).map_err(framework_error)
    }
    fn finish_document(&mut self) -> io::Result<()> {
        self.0.finish_document().map_err(framework_error)
    }
    fn start_field(
        &mut self,
        field: &FieldInfo,
        terms: i32,
        positions: bool,
        offsets: bool,
        payloads: bool,
    ) -> io::Result<()> {
        self.0
            .start_field(field, terms, positions, offsets, payloads)
            .map_err(framework_error)
    }
    fn finish_field(&mut self) -> io::Result<()> {
        self.0.finish_field().map_err(framework_error)
    }
    fn start_term(&mut self, term: &[u8], frequency: i32) -> io::Result<()> {
        self.0.start_term(term, frequency).map_err(framework_error)
    }
    fn finish_term(&mut self) -> io::Result<()> {
        self.0.finish_term().map_err(framework_error)
    }
    fn add_position(
        &mut self,
        position: i32,
        start: i32,
        end: i32,
        payload: Option<&[u8]>,
    ) -> io::Result<()> {
        self.0
            .add_position(position, start, end, payload)
            .map_err(framework_error)
    }
    fn finish(&mut self, documents: i32) -> io::Result<()> {
        self.0.finish(documents).map_err(framework_error)
    }
    fn close(&mut self) -> io::Result<()> {
        self.0.close().map_err(framework_error)
    }
    fn ram_bytes_used(&self) -> usize {
        self.0.ram_bytes_used()
    }
}

struct AssertingSink<W> {
    writer: AssertingTermVectorsWriter<FrameworkWriter<W>>,
}

impl<W> AssertingSink<W> {
    fn new(writer: W) -> Self {
        Self {
            writer: AssertingTermVectorsWriter::new(FrameworkWriter(writer)),
        }
    }
}

impl<W: TermVectorsWriteSink> TermVectorsWriteSink for AssertingSink<W> {
    fn start_document(&mut self, fields: i32) -> term_vectors::Result<()> {
        self.writer.start_document(fields).map_err(native_error)
    }
    fn finish_document(&mut self) -> term_vectors::Result<()> {
        self.writer.finish_document().map_err(native_error)
    }
    fn start_field(
        &mut self,
        field: &FieldInfo,
        terms: i32,
        positions: bool,
        offsets: bool,
        payloads: bool,
    ) -> term_vectors::Result<()> {
        self.writer
            .start_field(field, terms, positions, offsets, payloads)
            .map_err(native_error)
    }
    fn finish_field(&mut self) -> term_vectors::Result<()> {
        self.writer.finish_field().map_err(native_error)
    }
    fn start_term(&mut self, term: &[u8], frequency: i32) -> term_vectors::Result<()> {
        self.writer
            .start_term(term, frequency)
            .map_err(native_error)
    }
    fn finish_term(&mut self) -> term_vectors::Result<()> {
        self.writer.finish_term().map_err(native_error)
    }
    fn add_position(
        &mut self,
        position: i32,
        start: i32,
        end: i32,
        payload: Option<&[u8]>,
    ) -> term_vectors::Result<()> {
        self.writer
            .add_position(position, start, end, payload)
            .map_err(native_error)
    }
    fn finish(&mut self, documents: i32) -> term_vectors::Result<()> {
        self.writer.finish(documents).map_err(native_error)
    }
    fn close(&mut self) -> term_vectors::Result<()> {
        self.writer.close().map_err(native_error)
    }
    fn ram_bytes_used(&self) -> usize {
        self.writer.ram_bytes_used()
    }
}

#[cfg(test)]
mod tests;
