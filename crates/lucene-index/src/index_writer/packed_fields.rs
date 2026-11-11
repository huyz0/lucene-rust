//! A buffer's explicit document fields, packed into byte blocks while they
//! wait for the flush.
//!
//! Once a document's terms are in the postings, what is left of its
//! [`ExplicitFields`] is a norm per inverted field, its doc values, its
//! points and its registered field numbers: a few dozen bytes, which as
//! structs cost several hundred (a `Vec` per role, a 32-byte value per entry,
//! one allocation per binary value). Lucene's `IndexingChain` keeps the same
//! values in per-field columns (`NormValuesWriter`, `NumericDocValuesWriter`,
//! `PointValuesWriter`), so its buffer, and the RAM it charges against
//! `ramBufferSizeMB`, grows by about that few dozen bytes per document. These
//! blocks keep this writer's charge near Lucene's, and so its flushes as
//! frequent as Lucene's. The flush reads them back one document at a time
//! ([`PackedFields::for_each`]), a role per pass, so it never holds every
//! document's fields as structs at once; only an index sort, which permutes
//! whole documents, unpacks them all.
//!
//! A document's record is its shape (which fields, which kinds of value)
//! followed by its values. Documents of one index mostly share a shape, so a
//! record shaped like the block's previous packed one leaves the shape out;
//! each block's first packed record states it, so a read can start at any
//! block.
//!
//! A document whose inverted fields still hold terms, or that carries term
//! vectors, is kept as it is: those are rare and their shape is deep.
//!
//! Stored fields go further, as Lucene's `StoredFieldsConsumer` streams each
//! document into its buffer's stored-fields writer: [`StreamedStored`]
//! compresses them a chunk at a time as documents arrive and, in a
//! concurrent writer, writes the chunks to the `.fdt` of the segment the
//! buffer reserved, so stored values neither wait in memory nor count
//! against the RAM buffer.

use std::collections::BTreeSet;

use lucene_codecs::stored_fields::{self, Document, FieldValue, StoredField};
use lucene_store::codec_util::ID_LENGTH;

use super::explicit::{ExplicitFields, InvertedField};
use super::inverted_terms::InvertedTerms;
use super::{explicit_error, Error, FinishedStored, Result};

/// Bytes per block, the size `ByteBlockPool` allocates its blocks in.
const BLOCK: usize = 32 * 1024;

/// A packed document: its shape, then its values.
const PACKED: u8 = 0;
const KEPT: u8 = 1;
/// A packed document shaped like the block's last packed one: its values
/// only.
const SAME_SHAPE: u8 = 2;

const LONG: u8 = 0;
const INT: u8 = 1;
const FLOAT: u8 = 2;
const DOUBLE: u8 = 3;
const BINARY: u8 = 4;
const STRING: u8 = 5;

/// A buffer's [`ExplicitFields`], in the order they were added.
#[derive(Default)]
pub(crate) struct PackedFields {
    /// Every block but the last is full and never changes, so a
    /// [`Self::snapshot`] shares them.
    blocks: Vec<std::sync::Arc<Vec<u8>>>,
    /// Per block, its first document and how many kept documents come
    /// before it: where a read from a document on starts.
    block_starts: Vec<(usize, usize)>,
    kept: Vec<ExplicitFields>,
    len: usize,
    /// The encoded shape of the last block's last packed document.
    last_shape: Vec<u8>,
    shape: Vec<u8>,
    scratch: Vec<u8>,
}

impl PackedFields {
    /// How many documents' fields are buffered.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Heap bytes held.
    pub(crate) fn ram_bytes(&self) -> usize {
        self.blocks
            .iter()
            .map(|b| b.capacity())
            .chain(self.kept.iter().map(ExplicitFields::ram_bytes))
            .fold(0, usize::saturating_add)
    }

    /// Calls `f` with each document's fields in the order they were added,
    /// decoded one at a time into the same struct: `f` may take what it
    /// needs out of it.
    pub(crate) fn for_each(
        &self,
        f: impl FnMut(usize, &mut ExplicitFields) -> Result<()>,
    ) -> Result<()> {
        self.for_each_range(0, self.len, f)
    }

    /// [`Self::for_each`] over documents `start..limit` only, numbered
    /// from `start`: document `start` is `f`'s 0.
    pub(crate) fn for_each_range(
        &self,
        start: usize,
        limit: usize,
        mut f: impl FnMut(usize, &mut ExplicitFields) -> Result<()>,
    ) -> Result<()> {
        let limit = limit.min(self.len);
        let first = self
            .block_starts
            .partition_point(|&(d, _)| d <= start)
            .saturating_sub(1);
        let (mut i, kept_before) = self.block_starts.get(first).copied().unwrap_or((0, 0));
        let mut doc = ExplicitFields::default();
        let mut shape = Shape::default();
        let mut kept = self.kept.get(kept_before..).unwrap_or_default().iter();
        for block in self.blocks.get(first..).unwrap_or_default() {
            let mut r = Reader {
                bytes: block,
                pos: 0,
            };
            shape.clear();
            while r.pos < block.len() {
                if i == limit {
                    return Ok(());
                }
                let wanted = i >= start;
                match r.byte()? {
                    PACKED => {
                        shape.read(&mut r)?;
                        if wanted {
                            shape.decode_into(&mut r, &mut doc)?;
                        } else {
                            shape.skip(&mut r)?;
                        }
                    }
                    SAME_SHAPE if wanted => shape.decode_into(&mut r, &mut doc)?,
                    SAME_SHAPE => shape.skip(&mut r)?,
                    KEPT => {
                        let k = kept.next().ok_or_else(|| {
                            explicit_error("packed fields: a kept document is missing")
                        })?;
                        if wanted {
                            doc.clone_from(k);
                        }
                    }
                    _ => return Err(corrupt()),
                }
                if wanted {
                    f(i.wrapping_sub(start), &mut doc)?;
                }
                i = i.saturating_add(1);
            }
        }
        if i != limit {
            return Err(corrupt());
        }
        Ok(())
    }

    /// Appends one document's fields and returns the bytes this allocated:
    /// a new block, or what a kept document holds.
    pub(crate) fn push(&mut self, fields: ExplicitFields) -> usize {
        self.len = self.len.saturating_add(1);
        let keep =
            !fields.term_vectors.is_empty() || fields.inverted.iter().any(|f| !f.terms.is_empty());
        let mut scratch = std::mem::take(&mut self.scratch);
        let mut shape = std::mem::take(&mut self.shape);
        scratch.clear();
        let mut allocated = 0;
        if keep {
            scratch.push(KEPT);
            allocated = fields.ram_bytes();
            self.kept.push(fields);
        } else {
            shape.clear();
            encode_shape(&fields, &mut shape);
            // Most documents of an index have the same fields: those after
            // the first in a block skip the shape.
            let room = self
                .blocks
                .last()
                .map_or(0, |b| b.capacity().saturating_sub(b.len()));
            scratch.push(SAME_SHAPE);
            encode_values(&fields, &mut scratch);
            if shape != self.last_shape || scratch.len() > room {
                scratch[0] = PACKED;
                scratch.splice(1..1, shape.iter().copied());
                std::mem::swap(&mut self.last_shape, &mut shape);
            }
        }
        let at = (
            self.len.saturating_sub(1),
            self.kept.len().saturating_sub(usize::from(keep)),
        );
        let blocks = self.blocks.len();
        allocated = allocated.saturating_add(self.append(&scratch, at));
        if keep && self.blocks.len() != blocks {
            // A block's first packed document carries its shape.
            self.last_shape.clear();
        }
        self.scratch = scratch;
        self.shape = shape;
        allocated
    }

    /// Appends one document's record; `at` is that document's number and
    /// the kept documents before it.
    fn append(&mut self, record: &[u8], at: (usize, usize)) -> usize {
        if let Some(block) = self.blocks.last_mut() {
            if block.capacity().saturating_sub(block.len()) >= record.len() {
                // The last block is this buffer's alone: a snapshot copies it.
                std::sync::Arc::make_mut(block).extend_from_slice(record);
                return 0;
            }
        }
        let mut block = Vec::with_capacity(BLOCK.max(record.len()));
        block.extend_from_slice(record);
        let allocated = block.capacity();
        self.blocks.push(std::sync::Arc::new(block));
        self.block_starts.push(at);
        allocated
    }

    /// The documents buffered now, apart from this buffer: the full blocks
    /// shared, the last one copied.
    pub(crate) fn snapshot(&self) -> Self {
        let mut blocks = self.blocks.clone();
        if let Some(last) = blocks.last_mut() {
            *last = std::sync::Arc::new(last.as_slice().to_vec());
        }
        PackedFields {
            blocks,
            block_starts: self.block_starts.clone(),
            kept: self.kept.clone(),
            len: self.len,
            last_shape: Vec::new(),
            shape: Vec::new(),
            scratch: Vec::new(),
        }
    }

    /// The fields back, in the order they were added.
    pub(crate) fn unpack(self) -> Result<Vec<ExplicitFields>> {
        let mut out = Vec::with_capacity(self.len);
        let mut kept = self.kept.into_iter();
        for block in &self.blocks {
            let mut r = Reader {
                bytes: block,
                pos: 0,
            };
            let mut shape = Shape::default();
            while r.pos < block.len() {
                match r.byte()? {
                    PACKED => {
                        shape.read(&mut r)?;
                        let mut doc = ExplicitFields::default();
                        shape.decode_into(&mut r, &mut doc)?;
                        out.push(doc);
                    }
                    SAME_SHAPE => {
                        let mut doc = ExplicitFields::default();
                        shape.decode_into(&mut r, &mut doc)?;
                        out.push(doc);
                    }
                    KEPT => out.push(kept.next().ok_or_else(|| {
                        explicit_error("packed fields: a kept document is missing")
                    })?),
                    _ => return Err(corrupt()),
                }
            }
        }
        if out.len() != self.len {
            return Err(corrupt());
        }
        Ok(out)
    }
}

/// A buffer's stored fields, compressed as its documents arrive.
pub(crate) struct StreamedStored {
    writer: stored_fields::StoredFieldsWriter,
    mode: stored_fields::Mode,
    fields: BTreeSet<i32>,
    /// The segment name and id reserved for the buffer once its `.fdt` goes
    /// to that segment's file ([`Self::move_to_disk`]).
    reserved: Option<(String, [u8; ID_LENGTH])>,
}

impl StreamedStored {
    pub(crate) fn new(mode: stored_fields::Mode) -> Self {
        Self {
            writer: stored_fields::StoredFieldsWriter::new(
                mode,
                &[0; ID_LENGTH],
                mode.segment_suffix(),
            ),
            mode,
            fields: BTreeSet::new(),
            reserved: None,
        }
    }

    /// Whether the buffered `.fdt` has grown enough to go to disk
    /// ([`Self::move_to_disk`]).
    pub(crate) fn wants_disk(&self) -> bool {
        self.reserved.is_none() && self.writer.buffered_fdt_bytes() >= stored_fields::SPILL_BYTES
    }

    /// The `.fdt` of the segment `segment_name` in this buffer's mode.
    pub(crate) fn fdt_name(&self, segment_name: &str) -> String {
        super::segment_file_name(segment_name, self.mode.segment_suffix(), "fdt")
    }

    /// Writes the `.fdt` to `out`, the file of the segment `segment_name`
    /// with id `segment_id` the buffer will flush to: what is buffered now,
    /// then a chunk at a time as documents arrive, so the buffer holds about
    /// a chunk of stored fields rather than all of them.
    pub(crate) fn move_to_disk(
        &mut self,
        out: lucene_store::FsIndexOutput,
        segment_name: String,
        segment_id: [u8; ID_LENGTH],
    ) {
        self.writer.set_segment_id(&segment_id);
        self.writer.stream_to(out);
        self.reserved = Some((segment_name, segment_id));
    }

    /// The segment name and id [`Self::move_to_disk`] reserved.
    pub(crate) fn reserved(&self) -> Option<&(String, [u8; ID_LENGTH])> {
        self.reserved.as_ref()
    }

    /// Adds the next document's stored values. With `keep`, a document
    /// of a chunk's size or more comes back as the compressed chunk that
    /// holds it, which it closes, for live search to read it from rather
    /// than from a copy of its values.
    pub(crate) fn add(&mut self, doc: &Document, keep: bool) -> Option<ChunkDocument> {
        for f in &doc.fields {
            self.fields.insert(f.field_number);
        }
        let big = keep
            && stored_bytes(doc) >= usize::try_from(self.mode.chunk_size()).unwrap_or(usize::MAX);
        if !big {
            self.writer.add_document(doc);
            return None;
        }
        let at = self.writer.num_docs();
        self.writer.capture_chunks();
        self.writer.add_document(doc);
        let chunk = self
            .writer
            .take_captured()
            .into_iter()
            .find(|c| c.contains(at))?;
        Some(ChunkDocument {
            chunk: std::sync::Arc::new(chunk),
            doc: at,
        })
    }

    /// The heap the writer holds now: the chunk being filled, the `.fdt`
    /// bytes not yet on disk and the chunk index -- what Lucene's
    /// `StoredFieldsConsumer` counts toward the RAM buffer. It shrinks when
    /// buffered bytes go to disk.
    pub(crate) fn ram_bytes(&self) -> usize {
        self.writer.ram_bytes()
    }

    /// The stored-fields mode the documents were compressed in.
    pub(crate) fn mode(&self) -> stored_fields::Mode {
        self.mode
    }

    /// The `.fdt`, `.fdx` and `.fdm` bytes for the segment `segment_id`,
    /// with the field numbers any document stored. A buffer moved to disk
    /// finishes and closes its file, and hands back an empty `.fdt`.
    ///
    /// # Errors
    /// Closing the `.fdt` file.
    pub(crate) fn finish(mut self, segment_id: &[u8; ID_LENGTH]) -> Result<FinishedStored> {
        let bytes = if let Some((_, reserved)) = &self.reserved {
            debug_assert_eq!(reserved, segment_id, "the segment the buffer reserved");
            let (out, fdx, fdm) = self.writer.finish_streamed();
            if let Some(out) = out {
                out.close()?;
            }
            (Vec::new(), fdx, fdm)
        } else {
            self.writer.set_segment_id(segment_id);
            self.writer.finish()
        };
        Ok(FinishedStored {
            fields: self.fields,
            bytes: std::sync::Mutex::new(Some(bytes)),
        })
    }

    /// Drops a buffer that will never flush, returning the `.fdt` file it
    /// was writing, for the caller to release.
    pub(crate) fn discard(self) -> Option<String> {
        let name = self.reserved.as_ref().map(|(name, _)| self.fdt_name(name));
        drop(self.writer);
        name
    }
}

/// A buffered document as [`StreamedStored::add`] kept it: the compressed
/// chunk that holds it and its number there.
#[derive(Debug, Clone)]
pub(crate) struct ChunkDocument {
    pub(crate) chunk: std::sync::Arc<stored_fields::WrittenChunk>,
    pub(crate) doc: i32,
}

impl ChunkDocument {
    /// The document's stored fields.
    pub(crate) fn document(&self) -> Option<Vec<StoredField>> {
        self.chunk.document(self.doc).ok().map(|d| d.fields)
    }
}

/// About how many bytes `doc`'s stored values take.
fn stored_bytes(doc: &Document) -> usize {
    doc.fields
        .iter()
        .map(|f| match &f.value {
            FieldValue::Binary(b) => b.len(),
            FieldValue::String(s) => s.len(),
            _ => 8,
        })
        .fold(0usize, usize::saturating_add)
}

fn corrupt() -> Error {
    explicit_error("packed fields: the buffer does not decode")
}

/// A packed document's shape: its inverted fields and whether each has a
/// norm, its doc values' and points' fields and value kinds, and its
/// registered fields.
fn encode_shape(fields: &ExplicitFields, out: &mut Vec<u8>) {
    write_len(out, fields.inverted.len());
    for f in &fields.inverted {
        write_vlong(out, zigzag(i64::from(f.field_number)));
        out.push(u8::from(f.norm.is_some()));
    }
    for values in [&fields.doc_values, &fields.points] {
        write_len(out, values.len());
        for v in values {
            write_vlong(out, zigzag(i64::from(v.field_number)));
            out.push(match &v.value {
                FieldValue::Long(_) => LONG,
                FieldValue::Int(_) => INT,
                FieldValue::Float(_) => FLOAT,
                FieldValue::Double(_) => DOUBLE,
                FieldValue::Binary(_) => BINARY,
                FieldValue::String(_) => STRING,
            });
        }
    }
    write_len(out, fields.registered.len());
    for &n in &fields.registered {
        write_vlong(out, zigzag(i64::from(n)));
    }
}

/// A packed document's values, in its shape's order.
fn encode_values(fields: &ExplicitFields, out: &mut Vec<u8>) {
    for norm in fields.inverted.iter().filter_map(|f| f.norm) {
        write_vlong(out, zigzag(norm));
    }
    for v in fields.doc_values.iter().chain(&fields.points) {
        match &v.value {
            FieldValue::Long(x) => write_vlong(out, zigzag(*x)),
            FieldValue::Int(x) => write_vlong(out, zigzag(i64::from(*x))),
            FieldValue::Float(x) => out.extend_from_slice(&x.to_bits().to_le_bytes()),
            FieldValue::Double(x) => out.extend_from_slice(&x.to_bits().to_le_bytes()),
            FieldValue::Binary(b) => {
                write_len(out, b.len());
                out.extend_from_slice(b);
            }
            FieldValue::String(s) => {
                write_len(out, s.len());
                out.extend_from_slice(s.as_bytes());
            }
        }
    }
}

/// [`encode_shape`] read back, for the documents that follow it.
#[derive(Default)]
struct Shape {
    inverted: Vec<(i32, bool)>,
    doc_values: Vec<(i32, u8)>,
    points: Vec<(i32, u8)>,
    registered: Vec<i32>,
    /// Whether a shape was read in the current block.
    set: bool,
}

impl Shape {
    fn clear(&mut self) {
        self.inverted.clear();
        self.doc_values.clear();
        self.points.clear();
        self.registered.clear();
        self.set = false;
    }

    fn read(&mut self, r: &mut Reader<'_>) -> Result<()> {
        self.clear();
        let n = r.len()?;
        for _ in 0..n {
            let field_number = r.i32()?;
            let norm = match r.byte()? {
                0 => false,
                1 => true,
                _ => return Err(corrupt()),
            };
            self.inverted.push((field_number, norm));
        }
        for values in [&mut self.doc_values, &mut self.points] {
            let n = r.len()?;
            for _ in 0..n {
                let field_number = r.i32()?;
                let kind = r.byte()?;
                if kind > STRING {
                    return Err(corrupt());
                }
                values.push((field_number, kind));
            }
        }
        let n = r.len()?;
        for _ in 0..n {
            self.registered.push(r.i32()?);
        }
        self.set = true;
        Ok(())
    }

    /// Moves `r` past one document's values, checked as
    /// [`Self::decode_into`] checks them but not materialized.
    fn skip(&self, r: &mut Reader<'_>) -> Result<()> {
        if !self.set {
            return Err(corrupt());
        }
        for _ in self.inverted.iter().filter(|&&(_, has_norm)| has_norm) {
            r.vlong()?;
        }
        for &(_, kind) in self.doc_values.iter().chain(&self.points) {
            read_value(r, kind)?;
        }
        Ok(())
    }

    /// Decodes one document's values over `out`, reusing its vectors.
    fn decode_into(&self, r: &mut Reader<'_>, out: &mut ExplicitFields) -> Result<()> {
        if !self.set {
            return Err(corrupt());
        }
        out.inverted.clear();
        out.inverted.reserve(self.inverted.len());
        for &(field_number, has_norm) in &self.inverted {
            let norm = if has_norm {
                Some(unzigzag(r.vlong()?))
            } else {
                None
            };
            out.inverted.push(InvertedField {
                field_number,
                terms: InvertedTerms::default(),
                norm,
            });
        }
        for (values, shape) in [
            (&mut out.doc_values, &self.doc_values),
            (&mut out.points, &self.points),
        ] {
            values.clear();
            values.reserve(shape.len());
            for &(field_number, kind) in shape {
                let value = match read_value(r, kind)? {
                    Value::Long(x) => FieldValue::Long(x),
                    Value::Int(x) => FieldValue::Int(x),
                    Value::Float(x) => FieldValue::Float(x),
                    Value::Double(x) => FieldValue::Double(x),
                    Value::Binary(b) => FieldValue::Binary(b.to_vec()),
                    Value::String(s) => FieldValue::String(s.to_string()),
                };
                values.push(StoredField {
                    field_number,
                    value,
                });
            }
        }
        out.registered.clone_from(&self.registered);
        out.term_vectors.clear();
        Ok(())
    }
}

/// A packed value, its bytes borrowed from the block.
enum Value<'a> {
    Long(i64),
    Int(i32),
    Float(f32),
    Double(f64),
    Binary(&'a [u8]),
    String(&'a str),
}

/// Reads one value of `kind`, as [`encode_values`] wrote it.
fn read_value<'a>(r: &mut Reader<'a>, kind: u8) -> Result<Value<'a>> {
    Ok(match kind {
        LONG => Value::Long(unzigzag(r.vlong()?)),
        INT => Value::Int(i32::try_from(unzigzag(r.vlong()?)).map_err(|_| corrupt())?),
        FLOAT => Value::Float(f32::from_bits(u32::from_le_bytes(r.array()?))),
        DOUBLE => Value::Double(f64::from_bits(u64::from_le_bytes(r.array()?))),
        BINARY => {
            let len = r.len()?;
            Value::Binary(r.bytes(len)?)
        }
        STRING => {
            let len = r.len()?;
            Value::String(std::str::from_utf8(r.bytes(len)?).map_err(|_| corrupt())?)
        }
        _ => return Err(corrupt()),
    })
}

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

fn unzigzag(v: u64) -> i64 {
    ((v >> 1) as i64) ^ (v & 1).wrapping_neg() as i64
}

fn write_len(out: &mut Vec<u8>, n: usize) {
    write_vlong(out, n as u64);
}

fn write_vlong(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn byte(&mut self) -> Result<u8> {
        let b = *self.bytes.get(self.pos).ok_or_else(corrupt)?;
        self.pos = self.pos.saturating_add(1);
        Ok(b)
    }

    fn bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(len).ok_or_else(corrupt)?;
        let s = self.bytes.get(self.pos..end).ok_or_else(corrupt)?;
        self.pos = end;
        Ok(s)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.bytes(N)?.try_into().map_err(|_| corrupt())
    }

    fn vlong(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = self.byte()?;
            v |= u64::from(b & 0x7f) << shift;
            if b < 0x80 {
                return Ok(v);
            }
        }
        Err(corrupt())
    }

    fn len(&mut self) -> Result<usize> {
        usize::try_from(self.vlong()?).map_err(|_| corrupt())
    }

    fn i32(&mut self) -> Result<i32> {
        i32::try_from(unzigzag(self.vlong()?)).map_err(|_| corrupt())
    }
}

#[cfg(test)]
mod tests {
    // The arithmetic gate is about values read off disk; a test's `i - 3` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use lucene_codecs::term_vectors::TermVectorField;

    fn sample(i: i64) -> ExplicitFields {
        ExplicitFields {
            inverted: vec![
                InvertedField {
                    field_number: 1,
                    terms: InvertedTerms::default(),
                    norm: Some(i - 3),
                },
                InvertedField {
                    field_number: 7,
                    terms: InvertedTerms::default(),
                    norm: None,
                },
            ],
            doc_values: vec![
                StoredField {
                    field_number: 2,
                    value: FieldValue::Long(i64::MIN + i),
                },
                StoredField {
                    field_number: 3,
                    value: FieldValue::Binary(vec![i as u8; (i % 5) as usize]),
                },
                StoredField {
                    field_number: 4,
                    value: FieldValue::Double(i as f64 * -1.5),
                },
                StoredField {
                    field_number: 5,
                    value: FieldValue::Float(f32::NAN),
                },
                StoredField {
                    field_number: 6,
                    value: FieldValue::Int(i32::MAX - i as i32),
                },
                StoredField {
                    field_number: 8,
                    value: FieldValue::String(format!("s{i}é")),
                },
            ],
            points: vec![StoredField {
                field_number: 2,
                value: FieldValue::Binary((i as u64).to_be_bytes().to_vec()),
            }],
            registered: vec![9, i32::MAX],
            term_vectors: Vec::new(),
        }
    }

    fn same(a: &ExplicitFields, b: &ExplicitFields) {
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }

    /// Every value kind comes back as it went in, in order, across blocks,
    /// with documents kept whole between packed ones.
    #[test]
    fn fields_unpack_as_they_were_packed() {
        let mut packed = PackedFields::default();
        let mut expected = Vec::new();
        let mut allocated = 0;
        for i in 0..5000 {
            let mut f = sample(i);
            if i % 997 == 0 {
                f.term_vectors.push(TermVectorField {
                    field_number: 1,
                    has_positions: false,
                    has_offsets: false,
                    has_payloads: false,
                    terms: Vec::new(),
                });
            }
            expected.push(f.clone());
            allocated += packed.push(f);
        }
        assert_eq!(packed.len(), 5000);
        assert!(packed.blocks.len() > 1, "the records span several blocks");
        assert!(allocated >= packed.blocks.len() * BLOCK);
        // one document at a time, in order, each over the last
        let mut seen = 0;
        packed
            .for_each(|i, f| {
                same(f, &expected[i]);
                assert_eq!(i, seen);
                seen += 1;
                f.doc_values.clear();
                Ok(())
            })
            .unwrap();
        assert_eq!(seen, expected.len());
        // a range, from the block holding its first document
        for (start, end) in [(0, 1), (996, 998), (1500, 4100), (4999, 5000), (2000, 2000)] {
            let mut seen = Vec::new();
            packed
                .for_each_range(start, end, |i, f| {
                    same(f, &expected[start + i]);
                    seen.push(i);
                    Ok(())
                })
                .unwrap();
            assert_eq!(seen, (0..end - start).collect::<Vec<_>>());
        }
        let unpacked = packed.unpack().unwrap();
        assert_eq!(unpacked.len(), expected.len());
        for (a, b) in unpacked.iter().zip(&expected) {
            same(a, b);
        }
    }

    /// A kept document that opens a block is followed by a packed one with
    /// its shape, which the block's reader has not seen yet.
    #[test]
    fn a_block_opened_by_a_kept_document_restates_the_shape() {
        let mut packed = PackedFields::default();
        let mut expected = Vec::new();
        let mut kept = sample(0);
        kept.term_vectors.push(TermVectorField {
            field_number: 1,
            has_positions: false,
            has_offsets: false,
            has_payloads: false,
            terms: Vec::new(),
        });
        let mut opened = 0;
        let mut i = 0;
        while opened < 3 {
            let room = packed.blocks.last().map_or(0, |b| b.capacity() - b.len());
            // Fill the block to its last byte, then a kept document opens
            // the next one.
            let mut f = sample(i);
            let mut probe = Vec::new();
            f.doc_values[1].value = FieldValue::Binary(Vec::new());
            encode_values(&f, &mut probe);
            let fill = (1 + probe.len()..1 + probe.len() + 100).contains(&room);
            if fill {
                f.doc_values[1].value = FieldValue::Binary(vec![9; room - 1 - probe.len()]);
            }
            packed.push(f.clone());
            expected.push(f);
            if fill
                && packed
                    .blocks
                    .last()
                    .is_some_and(|b| b.len() == b.capacity())
            {
                let blocks = packed.blocks.len();
                packed.push(kept.clone());
                expected.push(kept.clone());
                assert_eq!(packed.blocks.len(), blocks + 1);
                opened += 1;
            }
            i += 1;
        }
        let mut n = 0;
        packed
            .for_each(|i, f| {
                same(f, &expected[i]);
                n += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(n, expected.len());
        for (a, b) in packed.unpack().unwrap().iter().zip(&expected) {
            same(a, b);
        }
    }

    /// A record larger than a block gets a block of its own.
    #[test]
    fn a_large_record_fits() {
        let mut packed = PackedFields::default();
        let mut f = sample(1);
        f.doc_values[1].value = FieldValue::Binary(vec![7; 3 * BLOCK]);
        packed.push(sample(0));
        packed.push(f.clone());
        packed.push(sample(2));
        let unpacked = packed.unpack().unwrap();
        same(&unpacked[1], &f);
        same(&unpacked[2], &sample(2));
    }

    /// A packed document is a few dozen bytes, where the struct costs
    /// hundreds.
    #[test]
    fn a_packed_document_is_small() {
        let mut f = sample(1000);
        f.doc_values.truncate(1);
        let mut out = Vec::new();
        encode_shape(&f, &mut out);
        encode_values(&f, &mut out);
        assert!(out.len() < 40, "{} bytes", out.len());
        assert!(f.ram_bytes() > 4 * out.len());
    }
}
