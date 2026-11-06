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
//! A document whose inverted fields still hold terms, or that carries term
//! vectors, is kept as it is: those are rare and their shape is deep.
//!
//! Stored fields go further, as Lucene's `StoredFieldsConsumer` streams each
//! document into its buffer's stored-fields writer: [`StreamedStored`]
//! compresses them a chunk at a time as documents arrive, and the flush
//! writes the finished bytes.

use std::collections::BTreeSet;

use lucene_codecs::stored_fields::{self, Document, FieldValue, StoredField};
use lucene_store::codec_util::ID_LENGTH;

use super::explicit::{ExplicitFields, InvertedField};
use super::inverted_terms::InvertedTerms;
use super::{explicit_error, Error, FinishedStored, Result};

/// Bytes per block, the size `ByteBlockPool` allocates its blocks in.
const BLOCK: usize = 32 * 1024;

const PACKED: u8 = 0;
const KEPT: u8 = 1;

const LONG: u8 = 0;
const INT: u8 = 1;
const FLOAT: u8 = 2;
const DOUBLE: u8 = 3;
const BINARY: u8 = 4;
const STRING: u8 = 5;

/// A buffer's [`ExplicitFields`], in the order they were added.
#[derive(Default)]
pub(crate) struct PackedFields {
    blocks: Vec<Vec<u8>>,
    kept: Vec<ExplicitFields>,
    len: usize,
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
            .map(Vec::capacity)
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
        self.for_each_upto(self.len, f)
    }

    /// [`Self::for_each`] over the first `limit` documents only.
    pub(crate) fn for_each_upto(
        &self,
        limit: usize,
        mut f: impl FnMut(usize, &mut ExplicitFields) -> Result<()>,
    ) -> Result<()> {
        let limit = limit.min(self.len);
        let mut doc = ExplicitFields::default();
        let mut kept = self.kept.iter();
        let mut i = 0usize;
        for block in &self.blocks {
            let mut r = Reader {
                bytes: block,
                pos: 0,
            };
            while r.pos < block.len() {
                if i == limit {
                    return Ok(());
                }
                match r.byte()? {
                    PACKED => decode_into(&mut r, &mut doc)?,
                    KEPT => doc.clone_from(kept.next().ok_or_else(|| {
                        explicit_error("packed fields: a kept document is missing")
                    })?),
                    _ => return Err(corrupt()),
                }
                f(i, &mut doc)?;
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
        scratch.clear();
        let mut allocated = 0;
        if keep {
            scratch.push(KEPT);
            allocated = fields.ram_bytes();
            self.kept.push(fields);
        } else {
            scratch.push(PACKED);
            encode(&fields, &mut scratch);
        }
        allocated = allocated.saturating_add(self.append(&scratch));
        self.scratch = scratch;
        allocated
    }

    fn append(&mut self, record: &[u8]) -> usize {
        if let Some(block) = self.blocks.last_mut() {
            if block.capacity().saturating_sub(block.len()) >= record.len() {
                block.extend_from_slice(record);
                return 0;
            }
        }
        let mut block = Vec::with_capacity(BLOCK.max(record.len()));
        block.extend_from_slice(record);
        let allocated = block.capacity();
        self.blocks.push(block);
        allocated
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
            while r.pos < block.len() {
                match r.byte()? {
                    PACKED => out.push(decode(&mut r)?),
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
        }
    }

    /// Adds the next document's stored values and returns how many bytes the
    /// writer grew by.
    pub(crate) fn add(&mut self, doc: &Document) -> usize {
        let before = self.writer.ram_bytes();
        for f in &doc.fields {
            self.fields.insert(f.field_number);
        }
        self.writer.add_document(doc);
        self.writer.ram_bytes().saturating_sub(before)
    }

    /// The stored-fields mode the documents were compressed in.
    pub(crate) fn mode(&self) -> stored_fields::Mode {
        self.mode
    }

    /// The `.fdt`, `.fdx` and `.fdm` bytes for the segment `segment_id`,
    /// with the field numbers any document stored.
    pub(crate) fn finish(mut self, segment_id: &[u8; ID_LENGTH]) -> FinishedStored {
        self.writer.set_segment_id(segment_id);
        FinishedStored {
            fields: self.fields,
            bytes: std::sync::Mutex::new(Some(self.writer.finish())),
        }
    }
}

fn corrupt() -> Error {
    explicit_error("packed fields: the buffer does not decode")
}

fn encode(fields: &ExplicitFields, out: &mut Vec<u8>) {
    write_len(out, fields.inverted.len());
    for f in &fields.inverted {
        write_vlong(out, zigzag(i64::from(f.field_number)));
        match f.norm {
            None => out.push(0),
            Some(norm) => {
                out.push(1);
                write_vlong(out, zigzag(norm));
            }
        }
    }
    for values in [&fields.doc_values, &fields.points] {
        write_len(out, values.len());
        for v in values {
            write_vlong(out, zigzag(i64::from(v.field_number)));
            match &v.value {
                FieldValue::Long(x) => {
                    out.push(LONG);
                    write_vlong(out, zigzag(*x));
                }
                FieldValue::Int(x) => {
                    out.push(INT);
                    write_vlong(out, zigzag(i64::from(*x)));
                }
                FieldValue::Float(x) => {
                    out.push(FLOAT);
                    out.extend_from_slice(&x.to_bits().to_le_bytes());
                }
                FieldValue::Double(x) => {
                    out.push(DOUBLE);
                    out.extend_from_slice(&x.to_bits().to_le_bytes());
                }
                FieldValue::Binary(b) => {
                    out.push(BINARY);
                    write_len(out, b.len());
                    out.extend_from_slice(b);
                }
                FieldValue::String(s) => {
                    out.push(STRING);
                    write_len(out, s.len());
                    out.extend_from_slice(s.as_bytes());
                }
            }
        }
    }
    write_len(out, fields.registered.len());
    for &n in &fields.registered {
        write_vlong(out, zigzag(i64::from(n)));
    }
}

fn decode(r: &mut Reader<'_>) -> Result<ExplicitFields> {
    let mut out = ExplicitFields::default();
    decode_into(r, &mut out)?;
    Ok(out)
}

/// Decodes one packed record over `out`, reusing its vectors.
fn decode_into(r: &mut Reader<'_>, out: &mut ExplicitFields) -> Result<()> {
    let n = r.len()?;
    let inverted = &mut out.inverted;
    inverted.clear();
    inverted.reserve(n);
    for _ in 0..n {
        let field_number = r.i32()?;
        let norm = match r.byte()? {
            0 => None,
            1 => Some(unzigzag(r.vlong()?)),
            _ => return Err(corrupt()),
        };
        inverted.push(InvertedField {
            field_number,
            terms: InvertedTerms::default(),
            norm,
        });
    }
    for values in [&mut out.doc_values, &mut out.points] {
        values.clear();
        let n = r.len()?;
        values.reserve(n);
        for _ in 0..n {
            let field_number = r.i32()?;
            let value = match r.byte()? {
                LONG => FieldValue::Long(unzigzag(r.vlong()?)),
                INT => FieldValue::Int(i32::try_from(unzigzag(r.vlong()?)).map_err(|_| corrupt())?),
                FLOAT => FieldValue::Float(f32::from_bits(u32::from_le_bytes(r.array()?))),
                DOUBLE => FieldValue::Double(f64::from_bits(u64::from_le_bytes(r.array()?))),
                BINARY => {
                    let len = r.len()?;
                    FieldValue::Binary(r.bytes(len)?.to_vec())
                }
                STRING => {
                    let len = r.len()?;
                    let s = std::str::from_utf8(r.bytes(len)?).map_err(|_| corrupt())?;
                    FieldValue::String(s.to_string())
                }
                _ => return Err(corrupt()),
            };
            values.push(StoredField {
                field_number,
                value,
            });
        }
    }
    let n = r.len()?;
    out.registered.clear();
    out.registered.reserve(n);
    for _ in 0..n {
        out.registered.push(r.i32()?);
    }
    out.term_vectors.clear();
    Ok(())
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
        let unpacked = packed.unpack().unwrap();
        assert_eq!(unpacked.len(), expected.len());
        for (a, b) in unpacked.iter().zip(&expected) {
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
        encode(&f, &mut out);
        assert!(out.len() < 40, "{} bytes", out.len());
        assert!(f.ram_bytes() > 4 * out.len());
    }
}
