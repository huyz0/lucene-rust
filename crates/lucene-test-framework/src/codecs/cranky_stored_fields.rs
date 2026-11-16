//! Java's cranky stored-fields format and nested writer, sharing one RNG.
//! Writer opening, finish and merge draw nextInt(100); close draws
//! nextInt(1000) only after successful delegation. Per-document and all field
//! overloads draw nextInt(10000) before delegation. Merge deliberately runs
//! the base visitor merge through this wrapper, rather than delegate.merge.
//!
//! The concrete adapter uses the production fdt/fdx/fdm encoder and reader.
//! Fields are buffered in the current Document until finishDocument. Native
//! chunk buffers are copied to outputs after automatic/final flushes. Sticky
//! output write errors surface at close. Outputs open fdm then fdt; fdx opens only
//! at finish. Close without finish preserves unfinished headers. Java's two
//! temporary index files are absent because the native encoder owns index
//! arrays. SegmentInfo mode attributes, IOContext and Codec/IndexWriter routing
//! are replaced by explicit state. Borrowed input bytes replace reader-owned
//! IndexInputs; caller-provided field-number remaps replace FieldInfos names.
//! Native accounting reports current document field slots only, excluding
//! value payloads and private codec buffers; generic accounting/resource callbacks forward intact.
//! `Rc<RefCell<JavaRandom>>` is single-threaded, unlike Java Random.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::rc::Rc;

use lucene_codecs::stored_fields::{
    self, Document, FieldValue, Mode, StoredField, StoredFieldVisitor, StoredFieldsReader,
    VisitStatus,
};
use lucene_store::codec_util;
use lucene_store::data_input::DataInput;
use lucene_store::index_output::IndexOutput;
use lucene_store::{DataOutput, Directory, FsIndexOutput};
use lucene_util::java_random::JavaRandom;

pub const FAKE_OPEN_ERROR: &str = "Fake IOException from StoredFieldsFormat.fieldsWriter()";
pub const FAKE_FINISH_ERROR: &str = "Fake IOException from StoredFieldsWriter.finish()";
pub const FAKE_MERGE_ERROR: &str = "Fake IOException from StoredFieldsWriter.merge()";
pub const FAKE_CLOSE_ERROR: &str = "Fake IOException from StoredFieldsWriter.close()";
pub const FAKE_START_ERROR: &str = "Fake IOException from StoredFieldsWriter.startDocument()";
pub const FAKE_FINISH_DOCUMENT_ERROR: &str =
    "Fake IOException from StoredFieldsWriter.finishDocument()";
pub const FAKE_FIELD_ERROR: &str = "Fake IOException from StoredFieldsWriter.writeField()";
fn error(message: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> stored_fields::Error {
    lucene_store::Error::Io(std::io::Error::other(message)).into()
}

#[derive(Clone, Copy)]
pub struct StoredFieldsState<'a> {
    pub directory: &'a dyn Directory,
    pub segment: &'a str,
    pub id: &'a [u8; 16],
    pub suffix: &'a str,
    pub mode: Mode,
}
#[derive(Clone, Copy)]
pub struct StoredFieldsReadState<'a> {
    pub fdt: &'a [u8],
    pub fdx: &'a [u8],
    pub fdm: &'a [u8],
    pub id: &'a [u8; 16],
    pub suffix: &'a str,
}

/// Accountable's child labels, returned as a borrowed collection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildResource {
    pub label: String,
    pub bytes: usize,
}

pub trait StoredFieldsWriter {
    fn start_document(&mut self) -> stored_fields::Result<()>;
    fn finish_document(&mut self) -> stored_fields::Result<()>;
    /// Six Java overloads: int, long, float, double, BytesRef and String.
    fn write_field(&mut self, number: i32, value: &FieldValue) -> stored_fields::Result<()>;
    /// The distinct StoredFieldDataInput overload; consumes input after guarding.
    fn write_field_data_input(
        &mut self,
        number: i32,
        input: &mut dyn DataInput,
        length: usize,
    ) -> stored_fields::Result<()>;
    fn finish(&mut self, num_docs: i32) -> stored_fields::Result<()>;
    fn close(&mut self) -> stored_fields::Result<()>;
    fn ram_bytes_used(&self) -> usize;
    fn child_resources(&self) -> &[ChildResource];
    fn merge(&mut self, state: &MergeState<'_>) -> stored_fields::Result<i32>
    where
        Self: Sized,
    {
        merge_visitors(self, state)
    }
}
pub trait StoredFieldsFormat {
    type Writer<'a>: StoredFieldsWriter;
    type Reader<'a>;
    fn fields_writer<'a>(
        &mut self,
        state: StoredFieldsState<'a>,
    ) -> stored_fields::Result<Self::Writer<'a>>;
    fn fields_reader<'a>(
        &mut self,
        state: StoredFieldsReadState<'a>,
    ) -> stored_fields::Result<Self::Reader<'a>>;
}

pub struct CrankyStoredFieldsFormat<F> {
    delegate: F,
    random: Rc<RefCell<JavaRandom>>,
}
impl<F> CrankyStoredFieldsFormat<F> {
    pub fn new(delegate: F, random: Rc<RefCell<JavaRandom>>) -> Self {
        Self { delegate, random }
    }
    pub fn delegate(&self) -> &F {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut F {
        &mut self.delegate
    }
}
impl<F: StoredFieldsFormat> StoredFieldsFormat for CrankyStoredFieldsFormat<F> {
    type Writer<'a> = CrankyStoredFieldsWriter<F::Writer<'a>>;
    type Reader<'a> = F::Reader<'a>;
    fn fields_writer<'a>(
        &mut self,
        state: StoredFieldsState<'a>,
    ) -> stored_fields::Result<Self::Writer<'a>> {
        if self.random.borrow_mut().next_int_bounded(100) == 0 {
            return Err(error(FAKE_OPEN_ERROR));
        }
        Ok(CrankyStoredFieldsWriter::new(
            self.delegate.fields_writer(state)?,
            self.random.clone(),
        ))
    }
    fn fields_reader<'a>(
        &mut self,
        state: StoredFieldsReadState<'a>,
    ) -> stored_fields::Result<Self::Reader<'a>> {
        self.delegate.fields_reader(state)
    }
}
pub struct CrankyStoredFieldsWriter<W> {
    delegate: W,
    random: Rc<RefCell<JavaRandom>>,
}
impl<W> CrankyStoredFieldsWriter<W> {
    pub fn new(delegate: W, random: Rc<RefCell<JavaRandom>>) -> Self {
        Self { delegate, random }
    }
    pub fn delegate(&self) -> &W {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut W {
        &mut self.delegate
    }
    fn guard(&self, bound: i32, message: &'static str) -> stored_fields::Result<()> {
        if self.random.borrow_mut().next_int_bounded(bound) == 0 {
            Err(error(message))
        } else {
            Ok(())
        }
    }
}
impl<W: StoredFieldsWriter> StoredFieldsWriter for CrankyStoredFieldsWriter<W> {
    fn start_document(&mut self) -> stored_fields::Result<()> {
        self.guard(10000, FAKE_START_ERROR)?;
        self.delegate.start_document()
    }
    fn finish_document(&mut self) -> stored_fields::Result<()> {
        self.guard(10000, FAKE_FINISH_DOCUMENT_ERROR)?;
        self.delegate.finish_document()
    }
    fn write_field(&mut self, number: i32, value: &FieldValue) -> stored_fields::Result<()> {
        self.guard(10000, FAKE_FIELD_ERROR)?;
        self.delegate.write_field(number, value)
    }
    fn write_field_data_input(
        &mut self,
        number: i32,
        input: &mut dyn DataInput,
        length: usize,
    ) -> stored_fields::Result<()> {
        self.guard(10000, FAKE_FIELD_ERROR)?;
        self.delegate.write_field_data_input(number, input, length)
    }
    fn finish(&mut self, num_docs: i32) -> stored_fields::Result<()> {
        self.guard(100, FAKE_FINISH_ERROR)?;
        self.delegate.finish(num_docs)
    }
    fn merge(&mut self, state: &MergeState<'_>) -> stored_fields::Result<i32> {
        self.guard(100, FAKE_MERGE_ERROR)?;
        merge_visitors(self, state)
    }
    fn close(&mut self) -> stored_fields::Result<()> {
        self.delegate.close()?;
        self.guard(1000, FAKE_CLOSE_ERROR)
    }
    fn ram_bytes_used(&self) -> usize {
        self.delegate.ram_bytes_used()
    }
    fn child_resources(&self) -> &[ChildResource] {
        self.delegate.child_resources()
    }
}

/// Reader callbacks required by StoredFieldsWriter's base merge algorithm.
pub trait StoredFieldsSource {
    fn check_integrity(&self) -> stored_fields::Result<()>;
    fn visit_document(
        &self,
        doc: i32,
        visitor: &mut dyn StoredFieldVisitor,
    ) -> stored_fields::Result<()>;
}
impl StoredFieldsSource for StoredFieldsReader<'_> {
    fn check_integrity(&self) -> stored_fields::Result<()> {
        self.check_integrity()
    }
    fn visit_document(
        &self,
        doc: i32,
        visitor: &mut dyn StoredFieldVisitor,
    ) -> stored_fields::Result<()> {
        self.visit_document(doc, visitor)
    }
}
/// doc_map is Java DocMap(oldDoc): -1 means deleted. Each live mapped ID
/// must occur exactly once and increase within its source. field_map is the
/// number remapping obtained by matching source/target FieldInfos names.
pub struct MergeSource<'a> {
    pub reader: &'a dyn StoredFieldsSource,
    pub doc_map: &'a [i32],
    pub field_map: &'a [(i32, i32)],
}
pub struct MergeState<'a> {
    pub sources: &'a [MergeSource<'a>],
    pub needs_index_sort: bool,
    pub check_aborted: &'a dyn Fn() -> stored_fields::Result<()>,
}
struct MergeVisitor<'a, W> {
    writer: &'a mut W,
    fields: &'a [(i32, i32)],
}
impl<W: StoredFieldsWriter> MergeVisitor<'_, W> {
    fn write(&mut self, field: i32, value: FieldValue) -> stored_fields::Result<()> {
        let number = self
            .fields
            .iter()
            .find(|&&(old, _)| old == field)
            .map_or(field, |&(_, new)| new);
        self.writer.write_field(number, &value)
    }
}
impl<W: StoredFieldsWriter> StoredFieldVisitor for MergeVisitor<'_, W> {
    fn needs_field(&mut self, _: i32) -> stored_fields::Result<VisitStatus> {
        Ok(VisitStatus::Yes)
    }
    fn string_field(&mut self, field: i32, value: &str) -> stored_fields::Result<()> {
        self.write(field, FieldValue::String(value.into()))
    }
    fn binary_field(&mut self, field: i32, value: &[u8]) -> stored_fields::Result<()> {
        self.write(field, FieldValue::Binary(value.into()))
    }
    fn int_field(&mut self, field: i32, value: i32) -> stored_fields::Result<()> {
        self.write(field, FieldValue::Int(value))
    }
    fn long_field(&mut self, field: i32, value: i64) -> stored_fields::Result<()> {
        self.write(field, FieldValue::Long(value))
    }
    fn float_field(&mut self, field: i32, value: f32) -> stored_fields::Result<()> {
        self.write(field, FieldValue::Float(value))
    }
    fn double_field(&mut self, field: i32, value: f64) -> stored_fields::Result<()> {
        self.write(field, FieldValue::Double(value))
    }
}
/// Java StoredFieldsWriter.merge: all source integrity checks precede copying;
/// callbacks dispatch through the wrapper, including its final finish guard.
pub fn merge_visitors<W: StoredFieldsWriter>(
    writer: &mut W,
    state: &MergeState<'_>,
) -> stored_fields::Result<i32> {
    for source in state.sources {
        (state.check_aborted)()?;
        source.reader.check_integrity()?;
    }
    let mut positions = vec![0; state.sources.len()];
    // Java DocIDMerger keeps the current sub outside the priority queue. Its
    // common sequential run avoids touching the heap until another sub wins.
    let mut queue = BinaryHeap::new();
    let mut current = 0;
    if state.needs_index_sort && state.sources.len() > 1 {
        for index in 1..state.sources.len() {
            if let Some(next) = next_mapped(state, &mut positions, index) {
                queue.push(Reverse(next));
            }
        }
    }
    let mut count = 0;
    while current < state.sources.len() {
        let candidate = next_mapped(state, &mut positions, current);
        let selected = if state.needs_index_sort && state.sources.len() > 1 {
            if candidate
                .is_some_and(|next| queue.peek().is_none_or(|&Reverse(best)| next.0 < best.0))
            {
                candidate
            } else {
                if let Some(next) = candidate {
                    queue.push(Reverse(next));
                }
                queue.pop().map(|Reverse(next)| next)
            }
        } else {
            candidate
        };
        let Some((mapped, index, doc)) = selected else {
            if state.needs_index_sort {
                break;
            }
            current += 1;
            continue;
        };
        current = index;
        assert_eq!(mapped, count, "merge maps must enumerate target documents");
        let source = &state.sources[index];
        writer.start_document()?;
        source.reader.visit_document(
            doc as i32,
            &mut MergeVisitor {
                writer,
                fields: source.field_map,
            },
        )?;
        writer.finish_document()?;
        count += 1;
    }
    writer.finish(count)?;
    Ok(count)
}

fn next_mapped(
    state: &MergeState<'_>,
    positions: &mut [usize],
    index: usize,
) -> Option<(i32, usize, usize)> {
    let source = &state.sources[index];
    while positions[index] < source.doc_map.len() {
        let doc = positions[index];
        positions[index] += 1;
        let mapped = source.doc_map[doc];
        // SENTINEL: MergeState.DocMap returns -1 for a deleted source document.
        if mapped != -1 {
            return Some((mapped, index, doc));
        }
    }
    None
}
#[derive(Clone, Copy, Default)]
pub struct Lucene90StoredFieldsFormat;
pub struct Lucene90StoredFieldsWriter<'a> {
    directory: &'a dyn Directory,
    segment: String,
    suffix: String,
    writer: Option<stored_fields::StoredFieldsWriter>,
    outputs: Option<(FsIndexOutput, FsIndexOutput)>,
    document: Document,
    completed: i32,
    fdm_prefix: usize,
    fdt_prefix: usize,
}
fn name(segment: &str, suffix: &str, extension: &str) -> String {
    if suffix.is_empty() {
        format!("{segment}.{extension}")
    } else {
        format!("{segment}_{suffix}.{extension}")
    }
}
impl StoredFieldsFormat for Lucene90StoredFieldsFormat {
    type Writer<'a> = Lucene90StoredFieldsWriter<'a>;
    type Reader<'a> = StoredFieldsReader<'a>;
    fn fields_writer<'a>(
        &mut self,
        state: StoredFieldsState<'a>,
    ) -> stored_fields::Result<Self::Writer<'a>> {
        let mut meta = state
            .directory
            .create_output(&name(state.segment, state.suffix, "fdm"))?;
        let mut prefix = Vec::new();
        codec_util::write_index_header(
            &mut prefix,
            "Lucene90FieldsIndexMeta",
            1,
            state.id,
            state.suffix,
        );
        meta.write_bytes(&prefix);
        let data = match state
            .directory
            .create_output(&name(state.segment, state.suffix, "fdt"))
        {
            Ok(data) => data,
            Err(failure) => {
                let _ = meta.close();
                return Err(failure.into());
            }
        };
        let mut result = Lucene90StoredFieldsWriter {
            directory: state.directory,
            segment: state.segment.into(),
            suffix: state.suffix.into(),
            writer: Some(stored_fields::StoredFieldsWriter::new(
                state.mode,
                state.id,
                state.suffix,
            )),
            outputs: Some((meta, data)),
            document: Document::default(),
            completed: 0,
            fdm_prefix: prefix.len(),
            fdt_prefix: 0,
        };
        prefix.clear();
        codec_util::write_index_header(
            &mut prefix,
            state.mode.data_codec(),
            1,
            state.id,
            state.suffix,
        );
        result.outputs.as_mut().unwrap().1.write_bytes(&prefix);
        result.fdt_prefix = prefix.len();
        result
            .outputs
            .as_mut()
            .unwrap()
            .0
            .write_vint(state.mode.chunk_size());
        // Chunk size follows the metadata header in both production encoders.
        result.fdm_prefix = result.outputs.as_ref().unwrap().0.file_pointer() as usize;
        Ok(result)
    }
    fn fields_reader<'a>(
        &mut self,
        state: StoredFieldsReadState<'a>,
    ) -> stored_fields::Result<Self::Reader<'a>> {
        stored_fields::open(state.fdt, state.fdx, state.fdm, state.id, state.suffix)
    }
}
impl StoredFieldsWriter for Lucene90StoredFieldsWriter<'_> {
    fn start_document(&mut self) -> stored_fields::Result<()> {
        Ok(())
    }
    fn finish_document(&mut self) -> stored_fields::Result<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| error("closed or finished writer"))?;
        writer.add_document(&self.document);
        let data = writer.flushed_data();
        self.outputs
            .as_mut()
            .ok_or_else(|| error("closed writer"))?
            .1
            .write_bytes(&data[self.fdt_prefix..]);
        self.fdt_prefix = data.len();
        self.document.fields.clear();
        self.completed += 1;
        Ok(())
    }
    fn write_field(&mut self, number: i32, value: &FieldValue) -> stored_fields::Result<()> {
        if self.writer.is_none() {
            return Err(error("closed or finished writer"));
        }
        self.document.fields.push(StoredField {
            field_number: number,
            value: value.clone(),
        });
        Ok(())
    }
    fn write_field_data_input(
        &mut self,
        number: i32,
        input: &mut dyn DataInput,
        length: usize,
    ) -> stored_fields::Result<()> {
        if self.writer.is_none() {
            return Err(error("closed or finished writer"));
        }
        let mut bytes = vec![0; length];
        input.read_bytes(&mut bytes)?;
        self.write_field(number, &FieldValue::Binary(bytes))
    }
    fn finish(&mut self, num_docs: i32) -> stored_fields::Result<()> {
        // Java flushes its pending dirty chunk before validating numDocs or
        // opening the index output; errors must preserve those partial bytes.
        let data = self
            .writer
            .as_mut()
            .ok_or_else(|| error("closed or finished writer"))?
            .flush_for_finish();
        self.outputs
            .as_mut()
            .ok_or_else(|| error("closed writer"))?
            .1
            .write_bytes(&data[self.fdt_prefix..]);
        self.fdt_prefix = data.len();
        if num_docs != self.completed {
            return Err(error(format!(
                "Wrote {} docs, finish called with numDocs={num_docs}",
                self.completed
            )));
        }
        let writer = self
            .writer
            .take()
            .ok_or_else(|| error("closed or finished writer"))?;
        let (fdt, fdx, fdm, index_end) = writer.finish_staged();
        let mut index = self
            .directory
            .create_output(&name(&self.segment, &self.suffix, "fdx"))?;
        index.write_bytes(&fdx);
        let (meta, data) = self
            .outputs
            .as_mut()
            .ok_or_else(|| error("closed writer"))?;
        meta.write_bytes(&fdm[self.fdm_prefix..index_end]);
        self.fdm_prefix = index_end;
        index.close()?;
        meta.write_bytes(&fdm[self.fdm_prefix..]);
        data.write_bytes(&fdt[self.fdt_prefix..]);
        Ok(())
    }
    fn close(&mut self) -> stored_fields::Result<()> {
        self.writer = None;
        self.document.fields.clear();
        let Some((meta, data)) = self.outputs.take() else {
            return Ok(());
        };
        let meta_result = meta.close();
        let data_result = data.close();
        meta_result?;
        data_result?;
        Ok(())
    }
    fn ram_bytes_used(&self) -> usize {
        self.document.fields.capacity() * std::mem::size_of::<StoredField>()
    }
    fn child_resources(&self) -> &[ChildResource] {
        &[]
    }
}

#[cfg(test)]
mod tests;
