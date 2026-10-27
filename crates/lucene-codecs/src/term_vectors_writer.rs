//! Optional same-wire Lucene90 term-vector writer lifecycle.
//!
//! Java: `TermVectorsFormat.vectorsWriter` and `TermVectorsWriter` callbacks.
//! The shared factory is `Send + Sync`; each sink borrows its operation's
//! directory and identity. The native adapter buffers the current document,
//! then feeds the existing chunk encoder. Bytes are assembled at `finish` and
//! output I/O failures are reported at `close`, unlike Java's streaming writes.
//! The driver closes after returned errors, preserving the first operation
//! error. Panics are not caught. Deletion belongs to the caller's tracked
//! directory: only successful output creation establishes ownership.

use std::{fmt, io, mem};

use lucene_store::{DataOutput, Directory, FsIndexOutput};

use crate::compression::CompressingFormat;
use crate::field_infos::FieldInfo;
use crate::term_vectors::{self, TermVectorField, TermVectorTerm, TermVectorsDocument};

/// Native codec identity and expected document count for one operation.
#[derive(Clone, Copy)]
pub struct TermVectorsWriteState<'a> {
    pub directory: &'a dyn Directory,
    pub segment: &'a str,
    pub id: &'a [u8; 16],
    pub suffix: &'a str,
    pub max_doc: i32,
    /// A `CompressingCodec`'s format, whose name, suffix and settings the
    /// files take (`suffix` is then the format's); `None` is
    /// `Lucene90TermVectorsFormat`.
    pub format: Option<&'static CompressingFormat>,
}

/// Java's document, field, term, occurrence, finish and close callbacks.
/// The caller finishes each open scope before opening the next one.
pub trait TermVectorsWriteSink {
    fn start_document(&mut self, fields: i32) -> term_vectors::Result<()>;
    fn finish_document(&mut self) -> term_vectors::Result<()>;
    fn start_field(
        &mut self,
        info: &FieldInfo,
        terms: i32,
        positions: bool,
        offsets: bool,
        payloads: bool,
    ) -> term_vectors::Result<()>;
    fn finish_field(&mut self) -> term_vectors::Result<()>;
    fn start_term(&mut self, term: &[u8], frequency: i32) -> term_vectors::Result<()>;
    fn finish_term(&mut self) -> term_vectors::Result<()>;
    fn add_position(
        &mut self,
        position: i32,
        start: i32,
        end: i32,
        payload: Option<&[u8]>,
    ) -> term_vectors::Result<()>;
    fn finish(&mut self, documents: i32) -> term_vectors::Result<()>;
    fn close(&mut self) -> term_vectors::Result<()>;
    /// Implementation-specific accounting; see the native adapter's limits.
    fn ram_bytes_used(&self) -> usize;
}

/// Shared codec configuration opening an operation-local native-wire writer.
pub trait TermVectorsWriterFactory: Send + Sync + fmt::Debug {
    fn vectors_writer<'a>(
        &self,
        state: TermVectorsWriteState<'a>,
    ) -> term_vectors::Result<Box<dyn TermVectorsWriteSink + 'a>>;
}

/// Runs the callback traversal and closes after success or returned error.
/// Java: `IOUtils.closeWhileHandlingException`; close errors fail otherwise
/// successful operations, but cannot replace an earlier traversal error.
pub fn with_writer<T, E>(
    factory: &dyn TermVectorsWriterFactory,
    state: TermVectorsWriteState<'_>,
    operation: impl FnOnce(&mut dyn TermVectorsWriteSink) -> Result<T, E>,
) -> Result<T, E>
where
    E: From<term_vectors::Error>,
{
    let mut writer = factory.vectors_writer(state).map_err(E::from)?;
    let result = operation(writer.as_mut());
    let closed = writer.close().map_err(E::from);
    match result {
        Ok(value) => closed.map(|()| value),
        Err(error) => Err(error),
    }
}

fn invalid(message: &str) -> term_vectors::Error {
    lucene_store::Error::Io(io::Error::new(io::ErrorKind::InvalidInput, message)).into()
}

/// The ordinary native encoder behind the callback lifecycle.
#[derive(Debug, Default, Clone, Copy)]
pub struct NativeTermVectorsWriterFactory;

impl TermVectorsWriterFactory for NativeTermVectorsWriterFactory {
    fn vectors_writer<'a>(
        &self,
        state: TermVectorsWriteState<'a>,
    ) -> term_vectors::Result<Box<dyn TermVectorsWriteSink + 'a>> {
        Ok(Box::new(NativeTermVectorsWriter::new(state)?))
    }
}

struct PendingDocument {
    value: TermVectorsDocument,
    fields: usize,
}
struct PendingField {
    value: TermVectorField,
    terms: usize,
}
struct PendingTerm {
    value: TermVectorTerm,
    occurrences: usize,
}

/// Buffered adapter for `Lucene90CompressingTermVectorsWriter` callbacks.
///
/// Callback shape/count errors are returned before the underlying encoder's
/// assertions can run. `finish` assembles the three native files; `close`
/// flushes their sticky output errors, attempts every acquired output and is
/// idempotent. The caller remains responsible for deleting owned files after
/// a returned error. Unfinished close does not synthesize a finished segment.
pub struct NativeTermVectorsWriter<'a> {
    state: TermVectorsWriteState<'a>,
    encoder: Option<term_vectors::TermVectorsWriter>,
    document: Option<PendingDocument>,
    field: Option<PendingField>,
    term: Option<PendingTerm>,
    meta: Option<FsIndexOutput>,
    data: Option<FsIndexOutput>,
    index: Option<FsIndexOutput>,
    finished: bool,
    closed: bool,
}

impl<'a> NativeTermVectorsWriter<'a> {
    /// Java constructor order: metadata, vectors data, then fields index.
    /// Failed acquisition closes acquired outputs without deleting a name
    /// whose creation failed; the original acquisition error wins.
    pub fn new(state: TermVectorsWriteState<'a>) -> term_vectors::Result<Self> {
        if state.max_doc < 0 {
            return Err(invalid("negative term-vector max_doc"));
        }
        if !state.suffix.is_ascii() || state.suffix.len() >= 256 {
            return Err(invalid(
                "term-vector suffix must be ASCII and shorter than 256 bytes",
            ));
        }
        let encoder = match state.format {
            None => term_vectors::TermVectorsWriter::new(state.id, state.suffix),
            Some(format) if format.segment_suffix == state.suffix => {
                term_vectors::TermVectorsWriter::with_format(state.id, format)
            }
            Some(_) => return Err(invalid("term-vector suffix differs from its format's")),
        };
        let mut writer = Self {
            state,
            encoder: Some(encoder),
            document: None,
            field: None,
            term: None,
            meta: None,
            data: None,
            index: None,
            finished: false,
            closed: false,
        };
        let stem = if state.suffix.is_empty() {
            state.segment.to_owned()
        } else {
            format!("{}_{}", state.segment, state.suffix)
        };
        let opened = (|| -> lucene_store::Result<()> {
            writer.meta = Some(state.directory.create_output(&format!("{stem}.tvm"))?);
            writer.data = Some(state.directory.create_output(&format!("{stem}.tvd"))?);
            writer.index = Some(state.directory.create_output(&format!("{stem}.tvx"))?);
            Ok(())
        })();
        if let Err(error) = opened {
            let _ = writer.close();
            return Err(error.into());
        }
        Ok(writer)
    }

    fn writable(&self) -> term_vectors::Result<()> {
        if self.closed {
            Err(invalid("closed term-vector writer"))
        } else if self.finished {
            Err(invalid("already finished term-vector writer"))
        } else {
            Ok(())
        }
    }
}

impl TermVectorsWriteSink for NativeTermVectorsWriter<'_> {
    fn start_document(&mut self, fields: i32) -> term_vectors::Result<()> {
        self.writable()?;
        if fields < 0 || self.document.is_some() {
            return Err(invalid("invalid term-vector document scope/count"));
        }
        if self
            .encoder
            .as_ref()
            .expect("unfinished encoder")
            .num_docs()
            >= self.state.max_doc
        {
            return Err(invalid("too many term-vector documents"));
        }
        self.document = Some(PendingDocument {
            value: TermVectorsDocument::default(),
            fields: fields as usize,
        });
        Ok(())
    }

    fn finish_document(&mut self) -> term_vectors::Result<()> {
        self.writable()?;
        let document = self
            .document
            .as_ref()
            .ok_or_else(|| invalid("no open document"))?;
        if self.field.is_some() || document.value.fields.len() != document.fields {
            return Err(invalid("unfinished term-vector fields"));
        }
        let document = self.document.take().expect("checked document");
        self.encoder
            .as_mut()
            .expect("unfinished encoder")
            .add_document(&document.value);
        Ok(())
    }

    fn start_field(
        &mut self,
        info: &FieldInfo,
        terms: i32,
        positions: bool,
        offsets: bool,
        payloads: bool,
    ) -> term_vectors::Result<()> {
        self.writable()?;
        let document = self
            .document
            .as_ref()
            .ok_or_else(|| invalid("no open document"))?;
        if self.field.is_some()
            || document.value.fields.len() >= document.fields
            || terms < 0
            || info.number < 0
            || (payloads && !positions)
        {
            return Err(invalid("invalid term-vector field scope/count/flags"));
        }
        self.field = Some(PendingField {
            value: TermVectorField {
                field_number: info.number,
                has_positions: positions,
                has_offsets: offsets,
                has_payloads: payloads,
                terms: Vec::new(),
            },
            terms: terms as usize,
        });
        Ok(())
    }

    fn finish_field(&mut self) -> term_vectors::Result<()> {
        self.writable()?;
        let field = self
            .field
            .as_ref()
            .ok_or_else(|| invalid("no open field"))?;
        if self.term.is_some() || field.value.terms.len() != field.terms {
            return Err(invalid("unfinished term-vector terms"));
        }
        let field = self.field.take().expect("checked field");
        self.document
            .as_mut()
            .expect("field requires document")
            .value
            .fields
            .push(field.value);
        Ok(())
    }

    fn start_term(&mut self, term: &[u8], frequency: i32) -> term_vectors::Result<()> {
        self.writable()?;
        let field = self
            .field
            .as_ref()
            .ok_or_else(|| invalid("no open field"))?;
        if self.term.is_some()
            || field.value.terms.len() >= field.terms
            || frequency <= 0
            || term.len() > i32::MAX as usize
        {
            return Err(invalid("invalid term-vector term scope/frequency/length"));
        }
        self.term = Some(PendingTerm {
            value: TermVectorTerm {
                term: term.to_vec(),
                freq: frequency,
                positions: field.value.has_positions.then(Vec::new),
                start_offsets: field.value.has_offsets.then(Vec::new),
                end_offsets: field.value.has_offsets.then(Vec::new),
                payloads: field.value.has_payloads.then(Vec::new),
            },
            occurrences: 0,
        });
        Ok(())
    }

    fn finish_term(&mut self) -> term_vectors::Result<()> {
        self.writable()?;
        let term = self.term.as_ref().ok_or_else(|| invalid("no open term"))?;
        let field = &self.field.as_ref().expect("term requires field").value;
        let expected = if field.has_positions || field.has_offsets {
            term.value.freq as usize
        } else {
            0
        };
        if term.occurrences != expected {
            return Err(invalid(
                "term-vector occurrence count differs from frequency",
            ));
        }
        let term = self.term.take().expect("checked term");
        self.field
            .as_mut()
            .expect("term requires field")
            .value
            .terms
            .push(term.value);
        Ok(())
    }

    fn add_position(
        &mut self,
        position: i32,
        start: i32,
        end: i32,
        payload: Option<&[u8]>,
    ) -> term_vectors::Result<()> {
        self.writable()?;
        let term = self.term.as_mut().ok_or_else(|| invalid("no open term"))?;
        let field = &self.field.as_ref().expect("term requires field").value;
        if (!field.has_positions && !field.has_offsets)
            || term.occurrences >= term.value.freq as usize
            || (!field.has_payloads && payload.is_some_and(|bytes| !bytes.is_empty()))
            || payload.is_some_and(|bytes| bytes.len() > i32::MAX as usize)
        {
            return Err(invalid("invalid term-vector occurrence count/flags"));
        }
        if let Some(values) = &mut term.value.positions {
            values.push(position);
        }
        if let Some(values) = &mut term.value.start_offsets {
            values.push(start);
        }
        if let Some(values) = &mut term.value.end_offsets {
            values.push(end);
        }
        if let Some(values) = &mut term.value.payloads {
            values.push(payload.unwrap_or_default().to_vec());
        }
        term.occurrences = term.occurrences.saturating_add(1);
        Ok(())
    }

    fn finish(&mut self, documents: i32) -> term_vectors::Result<()> {
        self.writable()?;
        if self.document.is_some()
            || documents != self.state.max_doc
            || documents
                != self
                    .encoder
                    .as_ref()
                    .expect("unfinished encoder")
                    .num_docs()
        {
            return Err(invalid(
                "unfinished term-vector document or document count mismatch",
            ));
        }
        self.finished = true;
        let (data, index, meta) = self.encoder.take().expect("unfinished encoder").finish();
        self.data.as_mut().expect("open data").write_bytes(&data);
        self.index.as_mut().expect("open index").write_bytes(&index);
        self.meta
            .as_mut()
            .expect("open metadata")
            .write_bytes(&meta);
        Ok(())
    }

    /// Java: `IOUtils.close(metaStream, vectorsStream, indexWriter)`.
    fn close(&mut self) -> term_vectors::Result<()> {
        self.closed = true;
        self.document = None;
        self.field = None;
        self.term = None;
        self.encoder = None;
        let mut first = None;
        for output in [&mut self.meta, &mut self.data, &mut self.index] {
            if let Some(output) = output.take() {
                if let Err(error) = output.close() {
                    first.get_or_insert(error);
                }
            }
        }
        first.map_or(Ok(()), |error| Err(error.into()))
    }

    /// Adapter shallow size and current callback-owned vector capacities only.
    /// Opaque encoder chunks, encoded outputs and allocator overhead are excluded;
    /// this is not whole-writer peak memory or a Java-comparable allocation count.
    fn ram_bytes_used(&self) -> usize {
        let mut bytes = mem::size_of::<Self>();
        if let Some(document) = &self.document {
            bytes = bytes.saturating_add(
                document
                    .value
                    .fields
                    .capacity()
                    .saturating_mul(mem::size_of::<TermVectorField>()),
            );
            for field in &document.value.fields {
                bytes = bytes.saturating_add(field_bytes(field));
            }
        }
        if let Some(field) = &self.field {
            bytes = bytes.saturating_add(field_bytes(&field.value));
        }
        if let Some(term) = &self.term {
            bytes = bytes.saturating_add(term_bytes(&term.value));
        }
        bytes
    }
}

fn field_bytes(field: &TermVectorField) -> usize {
    field.terms.iter().fold(
        field
            .terms
            .capacity()
            .saturating_mul(mem::size_of::<TermVectorTerm>()),
        |bytes, term| bytes.saturating_add(term_bytes(term)),
    )
}

fn term_bytes(term: &TermVectorTerm) -> usize {
    let mut bytes = term.term.capacity();
    for values in [&term.positions, &term.start_offsets, &term.end_offsets]
        .into_iter()
        .flatten()
    {
        bytes = bytes.saturating_add(values.capacity().saturating_mul(mem::size_of::<i32>()));
    }
    if let Some(payloads) = &term.payloads {
        bytes = bytes.saturating_add(
            payloads
                .capacity()
                .saturating_mul(mem::size_of::<Vec<u8>>()),
        );
        for payload in payloads {
            bytes = bytes.saturating_add(payload.capacity());
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use lucene_store::ByteBuffersDirectory;

    use super::*;

    const ID: [u8; 16] = [87; 16];
    fn state(directory: &dyn Directory) -> TermVectorsWriteState<'_> {
        TermVectorsWriteState {
            directory,
            segment: "_tv",
            id: &ID,
            suffix: "",
            max_doc: 1,
            format: None,
        }
    }
    fn field() -> FieldInfo {
        FieldInfo::new("f", 3)
    }
    fn document(writer: &mut dyn TermVectorsWriteSink, p: bool, o: bool, y: bool) {
        writer.start_document(1).unwrap();
        writer.start_field(&field(), 1, p, o, y).unwrap();
        writer.start_term(&[0, 255], 2).unwrap();
        if p || o {
            writer
                .add_position(
                    if p { 1 } else { -1 },
                    if o { 2 } else { -1 },
                    if o { 5 } else { -1 },
                    y.then_some(&[7, 8]),
                )
                .unwrap();
            writer
                .add_position(
                    if p { 4 } else { -1 },
                    if o { 6 } else { -1 },
                    if o { 9 } else { -1 },
                    None,
                )
                .unwrap();
        }
        writer.finish_term().unwrap();
        writer.finish_field().unwrap();
        writer.finish_document().unwrap();
    }

    #[test]
    fn native_callbacks_preserve_flags_payloads_suffix_and_exact_encoder_bytes() {
        for (p, o, y) in [
            (false, false, false),
            (true, false, false),
            (false, true, false),
            (true, true, false),
            (true, false, true),
            (true, true, true),
        ] {
            for suffix in ["", "s"] {
                let directory = ByteBuffersDirectory::new();
                let mut state = state(&directory);
                state.suffix = suffix;
                let mut writer = NativeTermVectorsWriter::new(state).unwrap();
                let shallow = writer.ram_bytes_used();
                writer.start_document(1).unwrap();
                writer.start_field(&field(), 1, p, o, y).unwrap();
                writer.start_term(&[0, 255], 2).unwrap();
                if p || o {
                    writer
                        .add_position(
                            if p { 1 } else { -1 },
                            if o { 2 } else { -1 },
                            if o { 5 } else { -1 },
                            y.then_some(&[7, 8]),
                        )
                        .unwrap();
                    writer
                        .add_position(
                            if p { 4 } else { -1 },
                            if o { 6 } else { -1 },
                            if o { 9 } else { -1 },
                            None,
                        )
                        .unwrap();
                }
                assert!(writer.ram_bytes_used() > shallow);
                writer.finish_term().unwrap();
                assert!(writer.ram_bytes_used() > shallow);
                writer.finish_field().unwrap();
                assert!(writer.ram_bytes_used() > shallow);
                let expected_doc = writer.document.as_ref().unwrap().value.clone();
                writer.finish_document().unwrap();
                assert_eq!(writer.ram_bytes_used(), shallow);
                let mut expected = term_vectors::TermVectorsWriter::new(&ID, suffix);
                expected.add_document(&expected_doc);
                let expected = expected.finish();
                writer.finish(1).unwrap();
                assert_eq!(
                    writer.finish(1).unwrap_err().to_string(),
                    "already finished term-vector writer"
                );
                writer.close().unwrap();
                writer.close().unwrap();
                assert_eq!(
                    writer.start_document(0).unwrap_err().to_string(),
                    "closed term-vector writer"
                );
                let stem = if suffix.is_empty() { "_tv" } else { "_tv_s" };
                for (extension, bytes) in [
                    ("tvd", expected.0),
                    ("tvx", expected.1),
                    ("tvm", expected.2),
                ] {
                    assert_eq!(
                        &*directory.open(&format!("{stem}.{extension}")).unwrap(),
                        bytes
                    );
                }
                let tvd = directory.open(&format!("{stem}.tvd")).unwrap();
                let tvx = directory.open(&format!("{stem}.tvx")).unwrap();
                let tvm = directory.open(&format!("{stem}.tvm")).unwrap();
                let reader = term_vectors::open(&tvd, &tvx, &tvm, &ID, suffix).unwrap();
                reader.check_integrity().unwrap();
                assert_eq!(reader.document(0).unwrap(), Some(expected_doc));
            }
        }
    }

    #[test]
    fn empty_documents_empty_fields_and_chunk_boundaries_remain_native() {
        for count in [0, 1, 129] {
            let directory = ByteBuffersDirectory::new();
            let mut state = state(&directory);
            state.max_doc = count;
            let mut writer = NativeTermVectorsWriterFactory
                .vectors_writer(state)
                .unwrap();
            for doc in 0..count {
                writer
                    .start_document(if doc % 2 == 0 { 0 } else { 1 })
                    .unwrap();
                if doc % 2 != 0 {
                    writer.start_field(&field(), 0, true, true, true).unwrap();
                    writer.finish_field().unwrap();
                }
                writer.finish_document().unwrap();
            }
            writer.finish(count).unwrap();
            writer.close().unwrap();
            let tvd = directory.open("_tv.tvd").unwrap();
            let tvx = directory.open("_tv.tvx").unwrap();
            let tvm = directory.open("_tv.tvm").unwrap();
            let reader = term_vectors::open(&tvd, &tvx, &tvm, &ID, "").unwrap();
            reader.check_integrity().unwrap();
            for doc in 0..count {
                let actual = reader.document(doc).unwrap();
                if doc % 2 == 0 {
                    assert!(actual.is_none());
                } else {
                    assert_eq!(actual.unwrap().fields[0].terms.len(), 0);
                }
            }
        }
    }

    #[test]
    fn malformed_callback_counts_scopes_and_flags_return_errors_without_panicking() {
        let directory = ByteBuffersDirectory::new();
        let mut negative = state(&directory);
        negative.max_doc = -1;
        assert!(NativeTermVectorsWriter::new(negative).is_err());
        for suffix in ["é".to_owned(), "x".repeat(256)] {
            let mut bad_suffix = state(&directory);
            bad_suffix.suffix = &suffix;
            assert!(NativeTermVectorsWriter::new(bad_suffix).is_err());
        }
        assert!(directory.list_all().unwrap().is_empty());
        let mut writer = NativeTermVectorsWriter::new(state(&directory)).unwrap();
        assert!(writer.finish_document().is_err());
        assert!(writer
            .start_field(&field(), 0, false, false, false)
            .is_err());
        assert!(writer.finish_field().is_err());
        assert!(writer.start_term(b"x", 1).is_err());
        assert!(writer.finish_term().is_err());
        assert!(writer.add_position(0, 0, 0, None).is_err());
        assert!(writer.finish(1).is_err());
        assert!(writer.start_document(-1).is_err());
        writer.start_document(1).unwrap();
        assert!(writer.start_document(0).is_err());
        assert!(writer.finish(1).is_err());
        assert!(writer.finish_document().is_err());
        assert!(writer
            .start_field(&field(), -1, false, false, false)
            .is_err());
        assert!(writer
            .start_field(&FieldInfo::new("invalid", -1), 0, false, false, false)
            .is_err());
        assert!(writer.start_field(&field(), 0, false, false, true).is_err());
        writer.start_field(&field(), 1, true, true, true).unwrap();
        assert!(writer.start_field(&field(), 1, true, true, true).is_err());
        assert!(writer.finish_document().is_err());
        assert!(writer.finish_field().is_err());
        assert!(writer.start_term(b"x", 0).is_err());
        assert!(writer.start_term(b"x", -1).is_err());
        writer.start_term(b"x", 1).unwrap();
        assert!(writer.start_term(b"x", 1).is_err());
        assert!(writer.finish_field().is_err());
        assert!(writer.finish_term().is_err());
        writer.add_position(0, 0, 1, None).unwrap();
        assert!(writer.add_position(0, 0, 1, None).is_err());
        writer.finish_term().unwrap();
        assert!(writer.start_term(b"x", 1).is_err());
        writer.finish_field().unwrap();
        assert!(writer
            .start_field(&field(), 0, false, false, false)
            .is_err());
        writer.finish_document().unwrap();
        assert!(writer.start_document(0).is_err());
        assert!(writer.finish(0).is_err());
        writer.finish(1).unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn occurrence_integer_values_retain_the_native_java_wrapping_semantics() {
        let directory = ByteBuffersDirectory::new();
        let mut writer = NativeTermVectorsWriter::new(state(&directory)).unwrap();
        writer.start_document(1).unwrap();
        writer.start_field(&field(), 1, true, true, false).unwrap();
        writer.start_term(b"t", 2).unwrap();
        writer
            .add_position(i32::MAX, i32::MIN, i32::MAX, None)
            .unwrap();
        writer
            .add_position(i32::MIN, i32::MAX, i32::MIN, None)
            .unwrap();
        writer.finish_term().unwrap();
        writer.finish_field().unwrap();
        let document = writer.document.as_ref().unwrap().value.clone();
        writer.finish_document().unwrap();
        writer.finish(1).unwrap();
        writer.close().unwrap();
        let mut encoder = term_vectors::TermVectorsWriter::new(&ID, "");
        encoder.add_document(&document);
        let expected = encoder.finish();
        for (extension, bytes) in [
            ("tvd", expected.0),
            ("tvx", expected.1),
            ("tvm", expected.2),
        ] {
            assert_eq!(
                &*directory.open(&format!("_tv.{extension}")).unwrap(),
                bytes
            );
        }
    }

    #[test]
    fn payload_flags_and_nonpositional_occurrence_calls_are_checked() {
        for positions in [false, true] {
            let directory = ByteBuffersDirectory::new();
            let mut writer = NativeTermVectorsWriter::new(state(&directory)).unwrap();
            writer.start_document(1).unwrap();
            writer
                .start_field(&field(), 1, positions, false, false)
                .unwrap();
            writer.start_term(b"t", 1).unwrap();
            assert!(writer.add_position(0, -1, -1, Some(&[1])).is_err());
            if positions {
                writer.add_position(0, -1, -1, Some(&[])).unwrap();
            } else {
                assert!(writer.add_position(0, -1, -1, None).is_err());
            }
            writer.finish_term().unwrap();
            writer.finish_field().unwrap();
            writer.finish_document().unwrap();
            writer.finish(1).unwrap();
            writer.close().unwrap();
        }
    }

    #[test]
    fn constructor_closes_successful_acquisitions_and_preserves_colliding_bytes() {
        for occupied in ["_tv.tvm", "_tv.tvd", "_tv.tvx"] {
            let directory = ByteBuffersDirectory::new();
            let mut output = directory.create_output(occupied).unwrap();
            output.write_byte(19);
            output.close().unwrap();
            let error = NativeTermVectorsWriter::new(state(&directory))
                .err()
                .unwrap();
            assert!(
                matches!(error, term_vectors::Error::Store(lucene_store::Error::Io(ref e)) if e.kind() == io::ErrorKind::AlreadyExists)
            );
            assert_eq!(&*directory.open(occupied).unwrap(), [19]);
            for name in directory.list_all().unwrap() {
                if name != occupied {
                    assert_eq!(directory.file_length(&name).unwrap(), 0);
                }
            }
        }
    }

    #[derive(Debug, Default)]
    struct Factory {
        events: Arc<Mutex<Vec<&'static str>>>,
        fail_open: bool,
        fail_close: bool,
    }
    struct Sink {
        events: Arc<Mutex<Vec<&'static str>>>,
        fail_close: bool,
    }
    fn failure(message: &str) -> term_vectors::Error {
        lucene_store::Error::Io(io::Error::other(message.to_owned())).into()
    }
    impl TermVectorsWriterFactory for Factory {
        fn vectors_writer<'a>(
            &self,
            state: TermVectorsWriteState<'a>,
        ) -> term_vectors::Result<Box<dyn TermVectorsWriteSink + 'a>> {
            assert_eq!(state.segment, "_tv");
            assert_eq!(state.id, &ID);
            assert_eq!(state.suffix, "");
            assert_eq!(state.max_doc, 1);
            self.events.lock().unwrap().push("open");
            if self.fail_open {
                return Err(failure("open"));
            }
            Ok(Box::new(Sink {
                events: self.events.clone(),
                fail_close: self.fail_close,
            }))
        }
    }
    impl TermVectorsWriteSink for Sink {
        fn start_document(&mut self, _: i32) -> term_vectors::Result<()> {
            self.events.lock().unwrap().push("start");
            Ok(())
        }
        fn finish_document(&mut self) -> term_vectors::Result<()> {
            Ok(())
        }
        fn start_field(
            &mut self,
            _: &FieldInfo,
            _: i32,
            _: bool,
            _: bool,
            _: bool,
        ) -> term_vectors::Result<()> {
            Ok(())
        }
        fn finish_field(&mut self) -> term_vectors::Result<()> {
            Ok(())
        }
        fn start_term(&mut self, _: &[u8], _: i32) -> term_vectors::Result<()> {
            Ok(())
        }
        fn finish_term(&mut self) -> term_vectors::Result<()> {
            Ok(())
        }
        fn add_position(
            &mut self,
            _: i32,
            _: i32,
            _: i32,
            _: Option<&[u8]>,
        ) -> term_vectors::Result<()> {
            Ok(())
        }
        fn finish(&mut self, _: i32) -> term_vectors::Result<()> {
            Ok(())
        }
        fn close(&mut self) -> term_vectors::Result<()> {
            self.events.lock().unwrap().push("close");
            if self.fail_close {
                Err(failure("close"))
            } else {
                Ok(())
            }
        }
        fn ram_bytes_used(&self) -> usize {
            0
        }
    }
    #[test]
    fn helper_closes_after_success_or_error_and_preserves_first_error() {
        let directory = ByteBuffersDirectory::new();
        for fail_close in [false, true] {
            let factory = Factory {
                fail_close,
                ..Factory::default()
            };
            let result: term_vectors::Result<_> =
                with_writer(&factory, state(&directory), |sink| {
                    sink.start_document(0)?;
                    Ok(19)
                });
            if fail_close {
                assert_eq!(result.unwrap_err().to_string(), "close");
            } else {
                assert_eq!(result.unwrap(), 19);
            }
            assert_eq!(*factory.events.lock().unwrap(), ["open", "start", "close"]);
            factory.events.lock().unwrap().clear();
            let result: term_vectors::Result<()> =
                with_writer(&factory, state(&directory), |_| Err(failure("primary")));
            assert_eq!(result.unwrap_err().to_string(), "primary");
            assert_eq!(*factory.events.lock().unwrap(), ["open", "close"]);
        }
        let factory = Factory {
            fail_open: true,
            ..Factory::default()
        };
        let result: term_vectors::Result<()> =
            with_writer(&factory, state(&directory), |_| panic!("not acquired"));
        assert_eq!(result.unwrap_err().to_string(), "open");
        assert_eq!(*factory.events.lock().unwrap(), ["open"]);
    }

    #[test]
    fn unfinished_close_drops_callback_buffers_without_finishing_a_segment() {
        let directory = ByteBuffersDirectory::new();
        let mut writer = NativeTermVectorsWriter::new(state(&directory)).unwrap();
        document(&mut writer, true, true, true);
        writer.close().unwrap();
        writer.close().unwrap();
        for name in directory.list_all().unwrap() {
            assert_eq!(directory.file_length(&name).unwrap(), 0);
        }
    }
}
