//! Java's `CrankyTermVectorsFormat` and callback writer, including superclass
//! proximity decoding and document-map merge rather than delegate bulk methods.
//!
//! Rust owns decoded fields instead of Java iterator/postings objects; malformed
//! owned models return I/O errors instead of Java assertions/index exceptions.
//! `Rc<RefCell<JavaRandom>>` shares Java's random stream on one thread. SimpleText
//! output buffers until finish; sticky output write errors surface at close, which publishes the
//! output. Concrete accounting reports finished output bytes, excluding the active
//! writer buffer; the generic wrapper forwards delegate accounting exactly.
//! An empty decoded payload means null, as in Java's SimpleText reader; direct
//! addProx decoding preserves explicit nonnull empty payloads.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io;
use std::rc::Rc;

use lucene_codecs::field_infos::{FieldInfo, FieldInfos};
use lucene_codecs::simpletext::term_vectors_reader::SimpleTextTermVectorsReader;
use lucene_codecs::simpletext::term_vectors_writer::SimpleTextTermVectorsWriter;
use lucene_codecs::term_vectors::TermVectorField;
use lucene_store::{DataInput, DataOutput, Directory, FsIndexOutput};
use lucene_util::java_random::JavaRandom;

pub type VectorDocument = Vec<(String, TermVectorField)>;
fn other(error: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::other(error)
}
fn input_error(error: lucene_store::Error) -> io::Error {
    match error {
        lucene_store::Error::Eof { .. } => io::Error::new(io::ErrorKind::UnexpectedEof, "input"),
        lucene_store::Error::Io(error) => error,
        error => other(error),
    }
}

pub struct VectorState<'a> {
    pub directory: &'a dyn Directory,
    pub segment: &'a str,
    pub max_doc: i32,
}
pub trait TermVectorsFormat {
    type Writer: TermVectorsWriter;
    type Reader;
    fn vectors_writer(&mut self, state: &VectorState<'_>) -> io::Result<Self::Writer>;
    fn vectors_reader(&mut self, state: &VectorState<'_>) -> io::Result<Self::Reader>;
}
pub trait TermVectorsReader {
    fn get(&self, doc: i32) -> io::Result<Option<VectorDocument>>;
    fn check_integrity(&self) -> io::Result<()>;
}
pub struct MergeSource<'a> {
    pub reader: Option<&'a dyn TermVectorsReader>,
    pub doc_map: &'a [i32],
}
pub struct MergeState<'a> {
    pub sources: &'a [MergeSource<'a>],
    pub field_infos: &'a FieldInfos,
    pub needs_index_sort: bool,
    pub check_aborted: &'a mut dyn FnMut() -> io::Result<()>,
}
pub trait TermVectorsWriter {
    fn start_document(&mut self, fields: i32) -> io::Result<()>;
    fn finish_document(&mut self) -> io::Result<()>;
    fn start_field(
        &mut self,
        info: &FieldInfo,
        terms: i32,
        positions: bool,
        offsets: bool,
        payloads: bool,
    ) -> io::Result<()>;
    fn finish_field(&mut self) -> io::Result<()>;
    fn start_term(&mut self, term: &[u8], frequency: i32) -> io::Result<()>;
    fn finish_term(&mut self) -> io::Result<()>;
    fn add_position(
        &mut self,
        position: i32,
        start: i32,
        end: i32,
        payload: Option<&[u8]>,
    ) -> io::Result<()>;
    fn finish(&mut self, documents: i32) -> io::Result<()>;
    fn close(&mut self) -> io::Result<()>;
    fn ram_bytes_used(&self) -> usize;
    fn add_prox(
        &mut self,
        count: i32,
        positions: Option<&mut dyn DataInput>,
        offsets: Option<&mut dyn DataInput>,
    ) -> io::Result<()> {
        add_prox_default(self, count, positions, offsets)
    }
    fn merge(&mut self, state: &mut MergeState<'_>) -> io::Result<i32> {
        merge_default(self, state)
    }
}
/// Java: `TermVectorsWriter.addProx`, including calls back through this writer.
// With small payloads on the stack and inlined RNG, the paired prox case is
// 0.89x Java. Expose the decoder so concrete input vtables can devirtualize.
#[inline]
pub fn add_prox_default<W: TermVectorsWriter + ?Sized>(
    writer: &mut W,
    count: i32,
    mut positions: Option<&mut dyn DataInput>,
    mut offsets: Option<&mut dyn DataInput>,
) -> io::Result<()> {
    let mut position = 0_i32;
    let mut last_offset = 0_i32;
    // The paired baseline was 0.70x Java: each addProx allocated a Vec even
    // for its two-byte payload. Keep short payloads on the stack; spill only
    // larger payloads, retaining the same input reads and callback borrows.
    let mut short_payload = [0_u8; 32];
    let mut long_payload = Vec::new();
    for _ in 0..count {
        let value = if let Some(input) = positions.as_deref_mut() {
            let code = input.read_vint().map_err(input_error)?;
            position = position.wrapping_add(((code as u32) >> 1) as i32);
            if code & 1 != 0 {
                let length =
                    usize::try_from(input.read_vint().map_err(input_error)?).map_err(other)?;
                // ALLOC: payload length must fit the remaining input before allocation.
                if length > input.remaining() {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "input"));
                }
                let payload = if length <= short_payload.len() {
                    &mut short_payload[..length]
                } else {
                    long_payload.resize(length, 0);
                    long_payload.as_mut_slice()
                };
                input.read_bytes(payload).map_err(input_error)?;
                Some(&*payload)
            } else {
                None
            }
        } else {
            position = -1;
            None
        };
        let (start, end) = if let Some(input) = offsets.as_deref_mut() {
            let start = last_offset.wrapping_add(input.read_vint().map_err(input_error)?);
            let end = start.wrapping_add(input.read_vint().map_err(input_error)?);
            last_offset = end;
            (start, end)
        } else {
            (-1, -1)
        };
        writer.add_position(position, start, end, value)?;
    }
    Ok(())
}
fn occurrence(values: &Option<Vec<i32>>, index: usize) -> io::Result<i32> {
    values
        .as_ref()
        .and_then(|v| v.get(index))
        .copied()
        .ok_or_else(|| other("missing term-vector occurrence"))
}
fn add_document<W: TermVectorsWriter + ?Sized>(
    writer: &mut W,
    document: Option<VectorDocument>,
    infos: &FieldInfos,
) -> io::Result<()> {
    let fields = document.unwrap_or_default();
    writer.start_document(i32::try_from(fields.len()).map_err(other)?)?;
    for (name, field) in fields {
        let info = infos
            .field_by_name(&name)
            .ok_or_else(|| other("missing destination field"))?;
        writer.start_field(
            info,
            i32::try_from(field.terms.len()).map_err(other)?,
            field.has_positions,
            field.has_offsets,
            field.has_payloads,
        )?;
        for term in field.terms {
            writer.start_term(&term.term, term.freq)?;
            if field.has_positions || field.has_offsets {
                for index in 0..usize::try_from(term.freq).map_err(other)? {
                    let position = if field.has_positions {
                        occurrence(&term.positions, index)?
                    } else {
                        -1
                    };
                    let (start, end) = if field.has_offsets {
                        (
                            occurrence(&term.start_offsets, index)?,
                            occurrence(&term.end_offsets, index)?,
                        )
                    } else {
                        (-1, -1)
                    };
                    let payload = term
                        .payloads
                        .as_ref()
                        .and_then(|p| p.get(index))
                        .filter(|p| !p.is_empty())
                        .map(Vec::as_slice);
                    writer.add_position(position, start, end, payload)?;
                }
            }
            writer.finish_term()?;
        }
        writer.finish_field()?;
    }
    writer.finish_document()
}
/// Java: `TermVectorsWriter.merge`, with its integrity pass before document callbacks.
pub fn merge_default<W: TermVectorsWriter + ?Sized>(
    writer: &mut W,
    state: &mut MergeState<'_>,
) -> io::Result<i32> {
    for source in state.sources {
        if let Some(reader) = source.reader {
            (state.check_aborted)()?;
            reader.check_integrity()?;
        }
    }
    let mut count = 0_i32;
    let mut copy = |source: usize, doc: usize| -> io::Result<()> {
        let document = state.sources[source]
            .reader
            .map(|r| r.get(i32::try_from(doc).map_err(other)?))
            .transpose()?
            .flatten();
        add_document(writer, document, state.field_infos)?;
        count = count
            .checked_add(1)
            .ok_or_else(|| other("merged document count overflow"))?;
        Ok(())
    };
    if state.needs_index_sort {
        // Java: SortedDocIDMerger holds current outside the queue, avoiding a
        // queue update while that source remains below the next source's ID.
        let mut heap = BinaryHeap::new();
        for (source, data) in state.sources.iter().enumerate().skip(1) {
            if let Some((doc, mapped)) = data.doc_map.iter().enumerate().find(|(_, id)| **id != -1)
            {
                heap.push(Reverse((*mapped, source, doc)));
            }
        }
        let mut current = (!state.sources.is_empty()).then_some((0, 0));
        while let Some((source, start)) = current {
            let candidate = state.sources[source]
                .doc_map
                .iter()
                .enumerate()
                .skip(start)
                .find(|(_, id)| **id != -1)
                .map(|(doc, mapped)| (*mapped, source, doc));
            let chosen = match candidate {
                Some(candidate) if heap.peek().is_none_or(|top| candidate.0 < top.0 .0) => {
                    Some(candidate)
                }
                Some(candidate) => {
                    let mut top = heap
                        .peek_mut()
                        .expect("nonempty queue below current document");
                    let chosen = top.0;
                    *top = Reverse(candidate);
                    Some(chosen)
                }
                None => heap.pop().map(|item| item.0),
            };
            if let Some((_, source, doc)) = chosen {
                copy(source, doc)?;
                let start = doc
                    .checked_add(1)
                    .ok_or_else(|| other("document index overflow"))?;
                current = Some((source, start));
            } else {
                current = None;
            }
        }
    } else {
        for (source, data) in state.sources.iter().enumerate() {
            for (doc, mapped) in data.doc_map.iter().enumerate() {
                if *mapped != -1 {
                    copy(source, doc)?;
                }
            }
        }
    }
    writer.finish(count)?;
    Ok(count)
}

pub struct CrankyTermVectorsFormat<F> {
    delegate: F,
    random: Rc<RefCell<JavaRandom>>,
}
impl<F> CrankyTermVectorsFormat<F> {
    pub fn new(delegate: F, random: Rc<RefCell<JavaRandom>>) -> Self {
        Self { delegate, random }
    }
    pub fn delegate(&self) -> &F {
        &self.delegate
    }
}
impl<F: TermVectorsFormat> TermVectorsFormat for CrankyTermVectorsFormat<F> {
    type Writer = CrankyTermVectorsWriter<F::Writer>;
    type Reader = F::Reader;
    fn vectors_writer(&mut self, state: &VectorState<'_>) -> io::Result<Self::Writer> {
        if self.random.borrow_mut().next_int_bounded(100) == 0 {
            return Err(other(
                "Fake IOException from TermVectorsFormat.vectorsWriter()",
            ));
        }
        Ok(CrankyTermVectorsWriter::new(
            self.delegate.vectors_writer(state)?,
            self.random.clone(),
        ))
    }
    fn vectors_reader(&mut self, state: &VectorState<'_>) -> io::Result<Self::Reader> {
        self.delegate.vectors_reader(state)
    }
}
pub struct CrankyTermVectorsWriter<W> {
    delegate: W,
    random: Rc<RefCell<JavaRandom>>,
}
impl<W> CrankyTermVectorsWriter<W> {
    pub fn new(delegate: W, random: Rc<RefCell<JavaRandom>>) -> Self {
        Self { delegate, random }
    }
    pub fn delegate(&self) -> &W {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut W {
        &mut self.delegate
    }
    pub fn child_resources(&self) -> impl Iterator<Item = &W> {
        std::iter::once(&self.delegate)
    }
    fn guard(&self, bound: i32, operation: &str) -> io::Result<()> {
        if self.random.borrow_mut().next_int_bounded(bound) == 0 {
            Err(other(format!(
                "Fake IOException from TermVectorsWriter.{operation}()"
            )))
        } else {
            Ok(())
        }
    }
}
impl<W: TermVectorsWriter> TermVectorsWriter for CrankyTermVectorsWriter<W> {
    fn start_document(&mut self, fields: i32) -> io::Result<()> {
        self.guard(10000, "startDocument")?;
        self.delegate.start_document(fields)
    }
    fn finish_document(&mut self) -> io::Result<()> {
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
    ) -> io::Result<()> {
        self.guard(10000, "startField")?;
        self.delegate
            .start_field(info, terms, positions, offsets, payloads)
    }
    fn finish_field(&mut self) -> io::Result<()> {
        self.guard(10000, "finishField")?;
        self.delegate.finish_field()
    }
    fn start_term(&mut self, term: &[u8], frequency: i32) -> io::Result<()> {
        self.guard(10000, "startTerm")?;
        self.delegate.start_term(term, frequency)
    }
    fn finish_term(&mut self) -> io::Result<()> {
        self.guard(10000, "finishTerm")?;
        self.delegate.finish_term()
    }
    fn add_position(
        &mut self,
        position: i32,
        start: i32,
        end: i32,
        payload: Option<&[u8]>,
    ) -> io::Result<()> {
        self.guard(10000, "addPosition")?;
        self.delegate.add_position(position, start, end, payload)
    }
    fn finish(&mut self, documents: i32) -> io::Result<()> {
        self.guard(100, "finish")?;
        self.delegate.finish(documents)
    }
    fn close(&mut self) -> io::Result<()> {
        self.delegate.close()?;
        self.guard(100, "close")
    }
    fn ram_bytes_used(&self) -> usize {
        self.delegate.ram_bytes_used()
    }
    // Keep this thin wrapper visible with the decoder for input specialization.
    #[inline]
    fn add_prox(
        &mut self,
        count: i32,
        positions: Option<&mut dyn DataInput>,
        offsets: Option<&mut dyn DataInput>,
    ) -> io::Result<()> {
        self.guard(10000, "addProx")?;
        add_prox_default(self, count, positions, offsets)
    }
    fn merge(&mut self, state: &mut MergeState<'_>) -> io::Result<i32> {
        self.guard(100, "merge")?;
        merge_default(self, state)
    }
}

pub struct SimpleTextVectorsFormat;
pub struct SimpleTextVectorsWriter {
    writer: Option<SimpleTextTermVectorsWriter>,
    output: Option<FsIndexOutput>,
    bytes: usize,
}
pub struct SimpleTextVectorsReader {
    bytes: Vec<u8>,
    reader: SimpleTextTermVectorsReader<'static>,
}
impl SimpleTextVectorsReader {
    pub fn open(bytes: Vec<u8>, max_doc: i32) -> io::Result<Self> {
        let reader = SimpleTextTermVectorsReader::open(&bytes, max_doc)
            .map_err(other)?
            .rebind(&[]);
        Ok(Self { bytes, reader })
    }
}
impl TermVectorsReader for SimpleTextVectorsReader {
    fn get(&self, doc: i32) -> io::Result<Option<VectorDocument>> {
        self.reader.rebind(&self.bytes).get(doc).map_err(other)
    }
    fn check_integrity(&self) -> io::Result<()> {
        Ok(())
    }
}
impl TermVectorsFormat for SimpleTextVectorsFormat {
    type Writer = SimpleTextVectorsWriter;
    type Reader = SimpleTextVectorsReader;
    fn vectors_writer(&mut self, state: &VectorState<'_>) -> io::Result<Self::Writer> {
        let output = state
            .directory
            .create_output(&format!("{}.tvc", state.segment))
            .map_err(input_error)?;
        Ok(SimpleTextVectorsWriter {
            writer: Some(SimpleTextTermVectorsWriter::new()),
            output: Some(output),
            bytes: 0,
        })
    }
    fn vectors_reader(&mut self, state: &VectorState<'_>) -> io::Result<Self::Reader> {
        let input = state
            .directory
            .open(&format!("{}.tvc", state.segment))
            .map_err(input_error)?;
        SimpleTextVectorsReader::open(input.as_ref().to_vec(), state.max_doc)
    }
}
impl SimpleTextVectorsWriter {
    fn writer(&mut self) -> io::Result<&mut SimpleTextTermVectorsWriter> {
        self.writer
            .as_mut()
            .ok_or_else(|| other("term-vector writer finished or closed"))
    }
}
impl TermVectorsWriter for SimpleTextVectorsWriter {
    fn start_document(&mut self, fields: i32) -> io::Result<()> {
        self.writer()?.start_document(fields);
        Ok(())
    }
    fn finish_document(&mut self) -> io::Result<()> {
        Ok(())
    }
    fn start_field(
        &mut self,
        info: &FieldInfo,
        terms: i32,
        positions: bool,
        offsets: bool,
        payloads: bool,
    ) -> io::Result<()> {
        self.writer()?
            .start_field(info, terms, positions, offsets, payloads);
        Ok(())
    }
    fn finish_field(&mut self) -> io::Result<()> {
        Ok(())
    }
    fn start_term(&mut self, term: &[u8], frequency: i32) -> io::Result<()> {
        self.writer()?.start_term(term, frequency);
        Ok(())
    }
    fn finish_term(&mut self) -> io::Result<()> {
        Ok(())
    }
    fn add_position(
        &mut self,
        position: i32,
        start: i32,
        end: i32,
        payload: Option<&[u8]>,
    ) -> io::Result<()> {
        self.writer()?.add_position(position, start, end, payload);
        Ok(())
    }
    fn finish(&mut self, documents: i32) -> io::Result<()> {
        let writer = self
            .writer
            .take()
            .ok_or_else(|| other("term-vector writer finished or closed"))?;
        let bytes = writer.finish(documents).map_err(other)?;
        self.bytes = bytes.len();
        self.output
            .as_mut()
            .ok_or_else(|| other("term-vector writer closed"))?
            .write_bytes(&bytes);
        Ok(())
    }
    fn close(&mut self) -> io::Result<()> {
        self.writer = None;
        if let Some(output) = self.output.take() {
            output.close().map_err(input_error)?;
        }
        Ok(())
    }
    fn ram_bytes_used(&self) -> usize {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::term_vectors::TermVectorTerm;
    use lucene_store::{ByteBuffersDirectory, SliceInput};

    #[derive(Default)]
    struct Probe {
        positions: Vec<(i32, i32, i32, Option<Vec<u8>>)>,
        calls: Vec<&'static str>,
    }
    impl TermVectorsWriter for Probe {
        fn start_document(&mut self, _: i32) -> io::Result<()> {
            self.calls.push("doc");
            Ok(())
        }
        fn finish_document(&mut self) -> io::Result<()> {
            self.calls.push("enddoc");
            Ok(())
        }
        fn start_field(
            &mut self,
            _: &FieldInfo,
            _: i32,
            _: bool,
            _: bool,
            _: bool,
        ) -> io::Result<()> {
            self.calls.push("field");
            Ok(())
        }
        fn finish_field(&mut self) -> io::Result<()> {
            self.calls.push("endfield");
            Ok(())
        }
        fn start_term(&mut self, _: &[u8], _: i32) -> io::Result<()> {
            self.calls.push("term");
            Ok(())
        }
        fn finish_term(&mut self) -> io::Result<()> {
            self.calls.push("endterm");
            Ok(())
        }
        fn add_position(&mut self, p: i32, s: i32, e: i32, y: Option<&[u8]>) -> io::Result<()> {
            self.positions.push((p, s, e, y.map(<[u8]>::to_vec)));
            Ok(())
        }
        fn finish(&mut self, _: i32) -> io::Result<()> {
            self.calls.push("finish");
            Ok(())
        }
        fn close(&mut self) -> io::Result<()> {
            Ok(())
        }
        fn ram_bytes_used(&self) -> usize {
            0
        }
    }
    fn term(freq: i32) -> TermVectorTerm {
        TermVectorTerm {
            term: vec![],
            freq,
            positions: None,
            start_offsets: None,
            end_offsets: None,
            payloads: None,
        }
    }
    fn field(term: TermVectorTerm, p: bool, o: bool) -> VectorDocument {
        vec![(
            "f".into(),
            TermVectorField {
                field_number: 9,
                has_positions: p,
                has_offsets: o,
                has_payloads: false,
                terms: vec![term],
            },
        )]
    }
    #[test]
    fn superclass_no_positions_and_invalid_occurrences() {
        let infos = FieldInfos::new(vec![FieldInfo::new("f", 7)]).unwrap();
        let mut p = Probe::default();
        add_document(&mut p, Some(field(term(3), false, false)), &infos).unwrap();
        assert!(p.positions.is_empty());
        assert!(p.calls.contains(&"endterm"));
        let mut t = term(1);
        t.start_offsets = Some(vec![2]);
        t.end_offsets = Some(vec![5]);
        add_document(&mut p, Some(field(t, false, true)), &infos).unwrap();
        assert_eq!(p.positions, vec![(-1, 2, 5, None)]);
        assert_eq!(
            add_document(&mut p, Some(field(term(1), true, false)), &infos)
                .unwrap_err()
                .to_string(),
            "missing term-vector occurrence"
        );
        assert!(add_document(&mut p, Some(field(term(-1), true, false)), &infos).is_err());
        let empty = FieldInfos::new(vec![]).unwrap();
        assert_eq!(
            add_document(&mut p, Some(field(term(0), false, false)), &empty)
                .unwrap_err()
                .to_string(),
            "missing destination field"
        );
    }
    #[test]
    fn proximity_wrapping_truncation_and_payload_distinction() {
        let mut p = Probe::default();
        // Java's unsigned position delta, then wrapping int addition.
        p.add_prox(
            3,
            Some(&mut SliceInput::new(&[
                0xfe, 0xff, 0xff, 0xff, 0x0f, 2, 1, 0,
            ])),
            None,
        )
        .unwrap();
        assert_eq!(
            p.positions,
            vec![
                (i32::MAX, -1, -1, None),
                (i32::MIN, -1, -1, None),
                (i32::MIN, -1, -1, Some(vec![]))
            ]
        );
        let mut large = vec![1, 33];
        large.extend(0_u8..33);
        p.add_prox(1, Some(&mut SliceInput::new(&large)), None)
            .unwrap();
        assert_eq!(
            p.positions.last().unwrap(),
            &(0, -1, -1, Some((0_u8..33).collect()))
        );
        let mut input = SliceInput::new(&[1, 2, 9]);
        assert_eq!(
            p.add_prox(1, Some(&mut input), None).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
        assert_eq!(input.position(), 2);
        let mut input = SliceInput::new(&[1, 0xff, 0xff, 0xff, 0xff, 0x0f]);
        assert!(p.add_prox(1, Some(&mut input), None).is_err());
        let mut input = SliceInput::new(&[0x80, 0x80, 0x80, 0x80, 0x80, 0]);
        assert!(p.add_prox(1, Some(&mut input), None).is_err());
        p.add_prox(1, None, None).unwrap();
        assert_eq!(p.positions.last().unwrap(), &(-1, -1, -1, None));
        assert_eq!(
            input_error(lucene_store::Error::Io(io::Error::other("typed"))).to_string(),
            "typed"
        );
    }
    #[test]
    fn sorted_merge_keeps_current_source_below_queue_minimum() {
        let infos = FieldInfos::new(vec![]).unwrap();
        let sources = [
            MergeSource {
                reader: None,
                doc_map: &[0, 1],
            },
            MergeSource {
                reader: None,
                doc_map: &[2, 3],
            },
        ];
        let mut abort = || Ok(());
        let mut state = MergeState {
            sources: &sources,
            field_infos: &infos,
            needs_index_sort: true,
            check_aborted: &mut abort,
        };
        let mut writer = Probe::default();
        assert_eq!(writer.merge(&mut state).unwrap(), 4);
        assert_eq!(
            writer.calls,
            vec!["doc", "enddoc", "doc", "enddoc", "doc", "enddoc", "doc", "enddoc", "finish"]
        );
    }
    #[test]
    fn empty_merge_and_simpletext_lifecycle() {
        let infos = FieldInfos::new(vec![]).unwrap();
        let sources = [MergeSource {
            reader: None,
            doc_map: &[-1, -1],
        }];
        let mut abort = || Ok(());
        let mut p = Probe::default();
        let mut state = MergeState {
            sources: &sources,
            field_infos: &infos,
            needs_index_sort: true,
            check_aborted: &mut abort,
        };
        assert_eq!(p.merge(&mut state).unwrap(), 0);
        assert_eq!(p.calls, vec!["finish"]);
        let dir = ByteBuffersDirectory::new();
        let state = VectorState {
            directory: &dir,
            segment: "_x",
            max_doc: 1,
        };
        let mut format = SimpleTextVectorsFormat;
        assert!(format.vectors_reader(&state).is_err());
        let mut writer = format.vectors_writer(&state).unwrap();
        assert_eq!(writer.ram_bytes_used(), 0);
        writer.start_document(0).unwrap();
        assert!(writer.finish(2).is_err());
        writer.close().unwrap();
        writer.close().unwrap();
        assert!(writer.start_document(0).is_err());
        dir.delete_file("_x.tvc").unwrap();
        let mut writer = format.vectors_writer(&state).unwrap();
        writer.close().unwrap();
        assert!(writer.finish(0).is_err());
        assert!(SimpleTextVectorsReader::open(vec![], 0).is_err());
    }
}
