//! Decoders for the byte blobs the JVM hands the engine writer: an operation
//! ([`decode_op`]), a field registration ([`decode_field`]), commit user data
//! ([`decode_user_data`]), and the field attributes that keep a vector or
//! substring field's format across a reopen. Malformed input is
//! [`Status::Decode`], never a panic.

use std::sync::Arc;

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::index_writer::{
    DocumentVector, ExplicitDocument, ExplicitFields, IndexWriter, InvertedField, InvertedTerms,
};
use lucene_util::quantization::ScalarEncoding;

use crate::alloc::try_with_capacity;
use crate::status::{set_last_error, Status};

/// Operation kinds of [`decode_op`].
pub const OP_ADD: u8 = 0;
pub const OP_SOFT_UPDATE: u8 = 1;
/// [`OP_ADD`] and [`OP_SOFT_UPDATE`] whose documents belong to a partition
/// (the value of the index's partition field): the kind is followed by the
/// partition (`i32 len, utf8`), then the operation as without one.
pub const OP_ADD_PARTITIONED: u8 = 2;
pub const OP_SOFT_UPDATE_PARTITIONED: u8 = 3;
/// Panics inside the writer -- the fault the "a panic fails exactly one
/// shard" test injects. Refused unless the handle was opened with fault
/// injection enabled.
pub const OP_PANIC: u8 = 0xFF;

/// A little-endian reader over a caller's blob; running off the end is
/// [`Status::Decode`], never a panic.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

fn decode_error(what: &str) -> Status {
    set_last_error(format!("engine writer: malformed blob: {what}"));
    Status::Decode
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], Status> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| decode_error(what))?;
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self, what: &str) -> Result<u8, Status> {
        Ok(self.take(1, what)?[0])
    }

    fn i32(&mut self, what: &str) -> Result<i32, Status> {
        let b = self.take(4, what)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i64(&mut self, what: &str) -> Result<i64, Status> {
        let b: [u8; 8] = self
            .take(8, what)?
            .try_into()
            .map_err(|_| decode_error(what))?;
        Ok(i64::from_le_bytes(b))
    }

    /// A count, refused when negative or when even `min_bytes` per element
    /// could not fit in what is left -- which bounds every allocation sized
    /// by a count to the blob's own length.
    fn count(&mut self, min_bytes: usize, what: &str) -> Result<usize, Status> {
        let n = usize::try_from(self.i32(what)?).map_err(|_| decode_error(what))?;
        if n.saturating_mul(min_bytes.max(1)) > self.remaining() {
            return Err(decode_error(what));
        }
        Ok(n)
    }

    fn bytes(&mut self, what: &str) -> Result<&'a [u8], Status> {
        let n = self.count(1, what)?;
        self.take(n, what)
    }

    fn string(&mut self, what: &str) -> Result<String, Status> {
        let b = self.bytes(what)?;
        std::str::from_utf8(b)
            .map(str::to_string)
            .map_err(|_| decode_error(what))
    }

    fn finish(&self, what: &str) -> Result<(), Status> {
        if self.remaining() != 0 {
            return Err(decode_error(what));
        }
        Ok(())
    }
}

fn stored_value(c: &mut Cursor<'_>) -> Result<FieldValue, Status> {
    let kind = c.u8("stored kind")?;
    let bytes = c.bytes("stored value")?;
    let fixed = |n: usize| -> Result<&[u8], Status> {
        if bytes.len() == n {
            Ok(bytes)
        } else {
            Err(decode_error("stored value width"))
        }
    };
    Ok(match kind {
        0 => FieldValue::String(
            std::str::from_utf8(bytes)
                .map_err(|_| decode_error("stored string"))?
                .to_string(),
        ),
        1 => FieldValue::Binary(bytes.to_vec()),
        2 => FieldValue::Int(i32::from_le_bytes(
            fixed(4)?.try_into().map_err(|_| decode_error("int"))?,
        )),
        3 => FieldValue::Long(i64::from_le_bytes(
            fixed(8)?.try_into().map_err(|_| decode_error("long"))?,
        )),
        4 => FieldValue::Float(f32::from_le_bytes(
            fixed(4)?.try_into().map_err(|_| decode_error("float"))?,
        )),
        5 => FieldValue::Double(f64::from_le_bytes(
            fixed(8)?.try_into().map_err(|_| decode_error("double"))?,
        )),
        _ => return Err(decode_error("stored kind")),
    })
}

fn inverted_field(c: &mut Cursor<'_>) -> Result<InvertedField, Status> {
    let field_number = c.i32("inverted field number")?;
    let norm = match c.u8("norm flag")? {
        0 => None,
        1 => Some(c.i64("norm")?),
        2 => return analyzed_field(c, field_number),
        _ => return Err(decode_error("norm flag")),
    };
    // A term is at least its length, its freq and its flags.
    let n = c.count(9, "term count")?;
    let mut terms = InvertedTerms::new();
    for _ in 0..n {
        let term = c.bytes("term")?;
        let freq = c.i32("freq")?;
        let flags = c.u8("term flags")?;
        if flags & !7 != 0 {
            return Err(decode_error("term flags"));
        }
        let occurrences = usize::try_from(freq).map_err(|_| decode_error("freq"))?;
        // Each occurrence's position, offsets and payload length are at
        // least 4 bytes apiece.
        let per = [1u8, 2, 4]
            .iter()
            .map(|&bit| if flags & bit != 0 { 4usize } else { 0 })
            .sum::<usize>();
        if occurrences.saturating_mul(per) > c.remaining() {
            return Err(decode_error("occurrences"));
        }
        let mut t = terms.start(term);
        t.freq(freq);
        if flags & 1 != 0 {
            for _ in 0..occurrences {
                t.position(c.i32("position")?);
            }
        }
        if flags & 2 != 0 {
            for _ in 0..occurrences {
                t.offsets(c.i32("start offset")?, c.i32("end offset")?);
            }
        }
        if flags & 4 != 0 {
            for _ in 0..occurrences {
                t.payload(c.bytes("payload")?);
            }
        }
    }
    Ok(InvertedField {
        field_number,
        terms,
        norm,
    })
}

/// An inverted field the JVM left to Rust to analyze ([`crate::invert`]):
/// `i32 analyzer` (a [`crate::analysis::register`] id), `u8 index
/// options` (ordinal), `u8 norm` ([`crate::invert::Norm`]), `i32 len,
/// name`, `i32 instances`, then each instance's text (`i32 len`, UTF-8).
fn analyzed_field(c: &mut Cursor<'_>, field_number: i32) -> Result<InvertedField, Status> {
    use crate::invert::{invert, Norm, Options};
    let registered = crate::analysis::registered(c.i32("analyzer")?)?;
    let options = Options::from_ordinal(c.u8("index options")?)
        .ok_or_else(|| decode_error("index options"))?;
    let norm = Norm::from_byte(c.u8("norm mode")?).ok_or_else(|| decode_error("norm mode"))?;
    let name = c.string("field name")?;
    let n = c.count(4, "instances")?;
    let mut texts = try_with_capacity(n)?;
    for _ in 0..n {
        texts.push(
            std::str::from_utf8(c.bytes("instance text")?)
                .map_err(|_| decode_error("instance text"))?,
        );
    }
    invert(field_number, &name, &registered, options, norm, &texts)
}

/// One document: its stored values, inverted fields, doc values and points,
/// each a count followed by that many entries.
///
/// - stored: `i32 field, u8 kind (0 string, 1 binary, 2 int, 3 long,
///   4 float, 5 double), i32 len, bytes`
/// - inverted: `i32 field, u8 has_norm, [i64 norm], i32 terms`, then per term
///   `i32 len, bytes, i32 freq, u8 flags (1 positions, 2 offsets,
///   4 payloads), [freq x i32 position], [freq x (i32 start, i32 end)],
///   [freq x (i32 len, bytes) payload]` -- a zero-length payload is an
///   occurrence without one; or `i32 field, u8 2` and a field to analyze
///   here ([`analyzed_field`])
/// - doc values: `i32 field, u8 kind (0 long, 1 bytes), i64 | (i32 len, bytes)`
/// - points: `i32 field, i32 len, packed bytes`
///
/// A document's KNN vectors follow the operation's documents ([`decode_op`]).
fn document(c: &mut Cursor<'_>) -> Result<ExplicitDocument, Status> {
    let n = c.count(9, "stored count")?;
    let mut stored = try_with_capacity(n)?;
    for _ in 0..n {
        let field_number = c.i32("stored field number")?;
        stored.push(StoredField {
            field_number,
            value: stored_value(c)?,
        });
    }
    let n = c.count(9, "inverted count")?;
    let mut inverted = try_with_capacity(n)?;
    for _ in 0..n {
        inverted.push(inverted_field(c)?);
    }
    let n = c.count(9, "doc values count")?;
    let mut doc_values = try_with_capacity(n)?;
    for _ in 0..n {
        let field_number = c.i32("doc values field number")?;
        let value = match c.u8("doc values kind")? {
            0 => FieldValue::Long(c.i64("doc value")?),
            1 => FieldValue::Binary(c.bytes("doc value")?.to_vec()),
            _ => return Err(decode_error("doc values kind")),
        };
        doc_values.push(StoredField {
            field_number,
            value,
        });
    }
    let n = c.count(8, "points count")?;
    let mut points = try_with_capacity(n)?;
    for _ in 0..n {
        let field_number = c.i32("points field number")?;
        points.push(StoredField {
            field_number,
            value: FieldValue::Binary(c.bytes("packed point")?.to_vec()),
        });
    }
    Ok(ExplicitDocument {
        stored,
        fields: ExplicitFields {
            inverted,
            doc_values,
            points,
            registered: Vec::new(),
            term_vectors: Vec::new(),
        },
    })
}

/// One document's value of one vector field: `i32 len, utf8 field name,
/// u8 encoding (0 float32, 1 byte), i32 dimension`, then the components
/// (`f32` bits as `i32` each, or one byte each) -- a
/// `KnnFloatVectorField`/`KnnByteVectorField`.
fn document_vector(c: &mut Cursor<'_>) -> Result<DocumentVector, Status> {
    let name = c.string("vector field name")?;
    let encoding = c.u8("vector encoding")?;
    let dim = usize::try_from(c.i32("vector dimension")?)
        .map_err(|_| decode_error("vector dimension"))?;
    Ok(match encoding {
        0 => {
            let raw = c.take(
                dim.checked_mul(4)
                    .ok_or_else(|| decode_error("vector dimension"))?,
                "vector",
            )?;
            let mut v = try_with_capacity(dim)?;
            v.extend(
                raw.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            );
            DocumentVector::float32(name, v)
        }
        1 => DocumentVector::byte(name, c.take(dim, "vector")?.to_vec()),
        _ => return Err(decode_error("vector encoding")),
    })
}

/// An operation blob, decoded.
pub enum Op {
    Add {
        partition: Option<Arc<str>>,
        docs: Vec<ExplicitDocument>,
        /// One list per document; `None` when no document has a vector.
        vectors: Option<Vec<Vec<DocumentVector>>>,
    },
    SoftUpdate {
        partition: Option<Arc<str>>,
        term: Term,
        soft_field: String,
        soft_value: i64,
        docs: Vec<ExplicitDocument>,
        vectors: Option<Vec<Vec<DocumentVector>>>,
    },
    Panic,
}

/// `u8 kind`, then for a soft update `term field (i32 len, utf8), term bytes
/// (i32 len, bytes), soft-deletes field (i32 len, utf8), i64 value`, then
/// `i32 count` documents ([`document`]), then, when any of them has a KNN
/// vector, per document `i32 n` and `n` vectors ([`document_vector`]). The
/// whole blob must be consumed.
///
/// A soft update is Lucene's `softUpdateDocuments(term, docs, softDeletes)`:
/// every live document matching `term` gets `soft_field = soft_value` as a
/// doc-values update, and `docs` are added, atomically. A delete's tombstone
/// is a soft update whose one document carries the soft-deletes value itself;
/// a stale operation is an add whose documents do.
pub fn decode_op(blob: &[u8]) -> Result<Op, Status> {
    let mut c = Cursor::new(blob);
    let kind = c.u8("op kind")?;
    let op = match kind {
        OP_PANIC => Op::Panic,
        OP_ADD | OP_SOFT_UPDATE | OP_ADD_PARTITIONED | OP_SOFT_UPDATE_PARTITIONED => {
            let partition = if matches!(kind, OP_ADD_PARTITIONED | OP_SOFT_UPDATE_PARTITIONED) {
                Some(Arc::<str>::from(c.string("partition")?))
            } else {
                None
            };
            let update = if matches!(kind, OP_SOFT_UPDATE | OP_SOFT_UPDATE_PARTITIONED) {
                let field = c.string("term field")?;
                let bytes = c.bytes("term bytes")?.to_vec();
                let soft_field = c.string("soft-deletes field")?;
                let soft_value = c.i64("soft-deletes value")?;
                Some((Term { field, bytes }, soft_field, soft_value))
            } else {
                None
            };
            let n = c.count(16, "document count")?;
            if n == 0 {
                return Err(decode_error("an operation needs at least one document"));
            }
            let mut docs = try_with_capacity(n)?;
            for _ in 0..n {
                docs.push(document(&mut c)?);
            }
            // The documents' vectors, when any has one: per document a count
            // and that many [`document_vector`]s.
            let vectors = if c.remaining() > 0 {
                let mut vectors: Vec<Vec<DocumentVector>> = try_with_capacity(n)?;
                for _ in 0..n {
                    let m = c.count(9, "vector count")?;
                    let mut doc_vectors = try_with_capacity(m)?;
                    for _ in 0..m {
                        doc_vectors.push(document_vector(&mut c)?);
                    }
                    vectors.push(doc_vectors);
                }
                Some(vectors)
            } else {
                None
            };
            match update {
                None => Op::Add {
                    partition,
                    docs,
                    vectors,
                },
                Some((term, soft_field, soft_value)) => Op::SoftUpdate {
                    partition,
                    term,
                    soft_field,
                    soft_value,
                    docs,
                    vectors,
                },
            }
        }
        _ => return Err(decode_error("op kind")),
    };
    c.finish("trailing bytes after the operation")?;
    Ok(op)
}

/// `i32 len, utf8 name, u8 index_options (0 none .. 4 docs+freqs+positions+
/// offsets), u8 omit_norms, u8 store_term_vectors, u8 store_payloads,
/// u8 doc_values_type (0 none, 1 numeric, 2 binary, 3 sorted, 4 sorted_set,
/// 5 sorted_numeric), u8 doc_values_skip_index, i32 point_dimension_count,
/// i32 point_index_dimension_count, i32 point_num_bytes, i32 vector_dimension,
/// and for a vector field (a non-zero dimension) its [`decode_vector_schema`],
/// then u8 soft_deletes_field, i32 attribute count, then (i32 len, utf8) key and
/// value pairs` -- Lucene's `FieldInfo`. The attributes are the field type's
/// own (`FieldType.putAttribute`); the writer adds its per-field format ones
/// per segment.
pub fn decode_field(blob: &[u8]) -> Result<(FieldInfo, Option<KnnVectorsFormat>), Status> {
    let mut c = Cursor::new(blob);
    let name = c.string("field name")?;
    let index_options = match c.u8("index options")? {
        0 => IndexOptions::None,
        1 => IndexOptions::Docs,
        2 => IndexOptions::DocsAndFreqs,
        3 => IndexOptions::DocsAndFreqsAndPositions,
        4 => IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        _ => return Err(decode_error("index options")),
    };
    let flag = |c: &mut Cursor<'_>, what: &str| -> Result<bool, Status> {
        match c.u8(what)? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(decode_error(what)),
        }
    };
    let omit_norms = flag(&mut c, "omit norms")?;
    let store_term_vectors = flag(&mut c, "store term vectors")?;
    let store_payloads = flag(&mut c, "store payloads")?;
    let doc_values_type = match c.u8("doc values type")? {
        0 => DocValuesType::None,
        1 => DocValuesType::Numeric,
        2 => DocValuesType::Binary,
        3 => DocValuesType::Sorted,
        4 => DocValuesType::SortedSet,
        5 => DocValuesType::SortedNumeric,
        _ => return Err(decode_error("doc values type")),
    };
    let skip = if flag(&mut c, "doc values skip index")? {
        DocValuesSkipIndexType::Range
    } else {
        DocValuesSkipIndexType::None
    };
    let dims = c.i32("point dimensions")?;
    let index_dims = c.i32("point index dimensions")?;
    let num_bytes = c.i32("point bytes")?;
    let vector_dimension = c.i32("vector dimension")?;
    let vector = if vector_dimension != 0 {
        Some(decode_vector_schema(&mut c)?)
    } else {
        None
    };
    let soft = flag(&mut c, "soft deletes")?;
    let attributes = decode_user_data_pairs(
        &mut c,
        [
            "field attribute count",
            "field attribute key",
            "field attribute value",
        ],
    )?;
    c.finish("trailing bytes after the field")?;
    let mut info = FieldInfo::new(name, 0)
        .with_index_options(index_options)
        .with_omit_norms(omit_norms)
        .with_store_term_vectors(store_term_vectors)
        .with_store_payloads(store_payloads)
        .with_doc_values(doc_values_type, skip, -1)
        .with_points(dims, index_dims, num_bytes)
        .with_soft_deletes_field(soft)
        .with_attributes(attributes);
    let format = match vector {
        Some((encoding, similarity, format)) => {
            info = info.with_vectors(vector_dimension, encoding, similarity);
            format
        }
        None => None,
    };
    Ok((info, format))
}

/// The field attribute naming the `KnnVectorsFormat` the plugin's mapping
/// chose for a vector field ([`parse_vectors_format`]), kept in the
/// segments' `FieldInfos` so a writer reopened on the index merges the field
/// in the same format before any document registers it again.
pub const VECTORS_FORMAT_ATTRIBUTE: &str = "accelerated.vectors_format";

/// A [`VECTORS_FORMAT_ATTRIBUTE`] value: `hnsw:<max_conn>:<beam_width>`,
/// `hnsw_hamming:<max_conn>:<beam_width>`, `hnsw_half:<max_conn>:<beam_width>`,
/// `hnsw_sq:<scalar encoding wire number>:<max_conn>:<beam_width>`,
/// `sq:<scalar encoding wire number>` or
/// `ivf:<target_cluster_size>:<rerank encoding>` (1-bit codes),
/// `ivf:<target_cluster_size>:<rerank encoding>:<quantizer>` or
/// `hnsw_rq:<quantizer>:<max_conn>:<beam_width>` or
/// `hnsw_disk:<quantizer>:<max_conn>:<beam_width>` (`IVF_QUANTIZERS` wire
/// number); `None` for anything else.
pub fn parse_vectors_format(value: &str) -> Option<KnnVectorsFormat> {
    let parts: Vec<&str> = value.split(':').collect();
    let int = |s: &str| s.parse::<i32>().ok();
    match parts.as_slice() {
        ["hnsw", m, beam] => KnnVectorsFormat::hnsw(int(m)?, int(beam)?).ok(),
        ["hnsw_hamming", m, beam] => KnnVectorsFormat::hnsw_hamming(int(m)?, int(beam)?).ok(),
        ["hnsw_half", m, beam] => KnnVectorsFormat::hnsw_half_float(int(m)?, int(beam)?).ok(),
        ["hnsw_sq", e, m, beam] => KnnVectorsFormat::hnsw_scalar_quantized(
            ScalarEncoding::from_wire_number(int(e)?)?,
            int(m)?,
            int(beam)?,
        )
        .ok(),
        ["sq", e] => Some(KnnVectorsFormat::scalar_quantized(
            ScalarEncoding::from_wire_number(int(e)?)?,
        )),
        ["hnsw_rq", q, m, beam] => {
            KnnVectorsFormat::hnsw_quantized(q.parse().ok()?, int(m)?, int(beam)?).ok()
        }
        ["hnsw_disk", q, m, beam] => {
            KnnVectorsFormat::hnsw_disk(q.parse().ok()?, int(m)?, int(beam)?).ok()
        }
        ["ivf", size, e] => KnnVectorsFormat::ivf(size.parse().ok()?, e.parse().ok()?).ok(),
        ["ivf", size, e, q] => {
            KnnVectorsFormat::ivf_quantized(size.parse().ok()?, e.parse().ok()?, q.parse().ok()?)
                .ok()
        }
        _ => None,
    }
}

/// The field attribute marking a vector field whose `_source` value the plugin stores masked, as
/// k-NN's derived source does: `true` for a field of the root document, `nested` for one of nested
/// documents. Every segment flushed with the field lists it under
/// [`lucene_search::derived_source::DERIVED_VECTOR_FIELDS`] or
/// [`lucene_search::derived_source::NESTED_DERIVED_VECTOR_FIELDS`], so a reader (the plugin's or
/// stock k-NN's) puts the vector back.
pub const DERIVED_SOURCE_ATTRIBUTE: &str = "accelerated.derived_source";

/// The `.si` list that names `info` when it carries [`DERIVED_SOURCE_ATTRIBUTE`].
pub fn derived_source_list(info: &FieldInfo) -> Option<&'static str> {
    use lucene_search::derived_source::{DERIVED_VECTOR_FIELDS, NESTED_DERIVED_VECTOR_FIELDS};
    info.attributes
        .iter()
        .find(|(k, _)| k == DERIVED_SOURCE_ATTRIBUTE)
        .and_then(|(_, v)| match v.as_str() {
            "true" => Some(DERIVED_VECTOR_FIELDS),
            "nested" => Some(NESTED_DERIVED_VECTOR_FIELDS),
            _ => None,
        })
}

/// k-NN's derived-source `.si` lists, for a k-NN index (`OPEN_KNN`): flushed segments list the
/// derived fields, and a merge keeps every name its sources list, so masked `_source` values
/// never lose their vectors.
pub fn derived_source_attributes(writer: &mut IndexWriter<'_>) {
    use lucene_search::derived_source::{DERIVED_VECTOR_FIELDS, NESTED_DERIVED_VECTOR_FIELDS};
    writer.set_merge_union_attributes(vec![
        DERIVED_VECTOR_FIELDS.to_string(),
        NESTED_DERIVED_VECTOR_FIELDS.to_string(),
    ]);
    let derived: Vec<(&'static str, String)> = writer
        .fields()
        .iter()
        .filter_map(|f| derived_source_list(f).map(|list| (list, f.name.clone())))
        .collect();
    for (list, name) in derived {
        writer.add_segment_list_value(list, &name);
    }
}

/// The field attribute of a `compact_wildcard` field: its BINARY doc values are
/// written in the `Substring90` format, in blocks of `2^value` documents
/// ([`lucene_codecs::doc_values::Lucene90DocValuesFormat::rust_substring`]).
/// Kept in the segments' `FieldInfos`, so a writer reopened on the index
/// routes the field to the format before it merges a segment, and a merge
/// never rewrites the field as plain BINARY.
pub const SUBSTRING_ATTRIBUTE: &str = "accelerated.substring";

/// The doc-values format [`SUBSTRING_ATTRIBUTE`] names on `info`, if any.
pub fn substring_format(
    info: &FieldInfo,
) -> Option<lucene_codecs::doc_values::Lucene90DocValuesFormat> {
    let shift = info
        .attributes
        .iter()
        .find(|(k, _)| k == SUBSTRING_ATTRIBUTE)
        .and_then(|(_, v)| v.parse::<u8>().ok())?;
    Some(lucene_codecs::doc_values::Lucene90DocValuesFormat::rust_substring(shift))
}

/// Routes every `compact_wildcard` field of `writer`'s existing segments to the
/// `Substring90` format their [`SUBSTRING_ATTRIBUTE`] names.
pub fn restore_substring_formats(writer: &mut IndexWriter<'_>, fields: &[FieldInfo]) {
    for f in fields {
        if let Some(format) = substring_format(f) {
            writer.set_doc_values_format_for_field(&f.name, format);
        }
    }
}

/// Routes every vector field of `writer`'s existing segments to the format
/// its [`VECTORS_FORMAT_ATTRIBUTE`] names.
pub fn restore_vectors_formats(writer: &mut IndexWriter<'_>, fields: &[FieldInfo]) {
    let found: Vec<(String, KnnVectorsFormat)> = fields
        .iter()
        .filter(|f| f.vector_dimension > 0)
        .filter_map(|f| {
            let value = f
                .attributes
                .iter()
                .find(|(k, _)| k == VECTORS_FORMAT_ATTRIBUTE)
                .map(|(_, v)| v)?;
            Some((f.name.clone(), parse_vectors_format(value)?))
        })
        .collect();
    for (name, format) in found {
        writer.set_knn_vectors_format_for_field(&name, format);
    }
}

/// A vector field's `u8 encoding (0 float32, 1 byte), u8 similarity
/// (VectorSimilarityFunction ordinal: 0 euclidean, 1 dot product, 2 cosine,
/// 3 maximum inner product)`, then its `KnnVectorsFormat`: `u8 kind` --
/// `0` the codec's default (`Lucene99HnswVectorsFormat()`), `1`
/// `Lucene99HnswVectorsFormat(max_conn, beam_width)`, `2`
/// `Lucene104HnswScalarQuantizedVectorsFormat(encoding, max_conn,
/// beam_width)`, `3` `Lucene104ScalarQuantizedVectorsFormat(encoding)`, `4`
/// k-NN's `KNN9120HnswBinaryVectorsFormat(max_conn, beam_width)` (byte
/// fields only; recorded as `KNN990HnswBinaryVectorsFormat`) or `5` k-NN's
/// `KNN1040HnswHalfFloatVectorsFormat(max_conn, beam_width)` (float fields
/// only) or `6` `IvfVectorsFormat(target_cluster_size,
/// rerank_encoding, quantizer)`; kinds `1`, `2`, `4` and `5` are followed by `i32 max_conn,
/// i32 beam_width`, kinds `2` and `3` by `u8 scalar encoding`
/// (`ScalarEncoding.getWireNumber`), kind `6` by `i32 target_cluster_size, u8
/// rerank_encoding` (`0` f32, `1` f16, `2` bf16, `3` int8), `u8 quantizer`
/// (`IVF_QUANTIZERS` wire number).
fn decode_vector_schema(
    c: &mut Cursor<'_>,
) -> Result<
    (
        VectorEncoding,
        VectorSimilarityFunction,
        Option<KnnVectorsFormat>,
    ),
    Status,
> {
    let encoding = match c.u8("vector encoding")? {
        0 => VectorEncoding::Float32,
        1 => VectorEncoding::Byte,
        _ => return Err(decode_error("vector encoding")),
    };
    let similarity = match c.u8("vector similarity")? {
        0 => VectorSimilarityFunction::Euclidean,
        1 => VectorSimilarityFunction::DotProduct,
        2 => VectorSimilarityFunction::Cosine,
        3 => VectorSimilarityFunction::MaximumInnerProduct,
        _ => return Err(decode_error("vector similarity")),
    };
    let kind = c.u8("vectors format")?;
    let graph = |c: &mut Cursor<'_>| -> Result<(i32, i32), Status> {
        Ok((c.i32("max conn")?, c.i32("beam width")?))
    };
    let scalar = |c: &mut Cursor<'_>| -> Result<ScalarEncoding, Status> {
        ScalarEncoding::from_wire_number(i32::from(c.u8("scalar encoding")?))
            .ok_or_else(|| decode_error("scalar encoding"))
    };
    let invalid = |e: lucene_codecs::per_field_knn_vectors::InvalidFormat| {
        set_last_error(e.0);
        Status::InvalidArgument
    };
    let format = match kind {
        0 => None,
        1 => {
            let (m, beam) = graph(c)?;
            Some(KnnVectorsFormat::hnsw(m, beam).map_err(invalid)?)
        }
        2 => {
            let (m, beam) = graph(c)?;
            let e = scalar(c)?;
            Some(KnnVectorsFormat::hnsw_scalar_quantized(e, m, beam).map_err(invalid)?)
        }
        3 => Some(KnnVectorsFormat::scalar_quantized(scalar(c)?)),
        4 => {
            let (m, beam) = graph(c)?;
            Some(KnnVectorsFormat::hnsw_hamming(m, beam).map_err(invalid)?)
        }
        5 => {
            let (m, beam) = graph(c)?;
            Some(KnnVectorsFormat::hnsw_half_float(m, beam).map_err(invalid)?)
        }
        6 => {
            let size = u32::try_from(c.i32("target cluster size")?)
                .map_err(|_| decode_error("target cluster size"))?;
            let rerank = c.u8("rerank encoding")?;
            let quantizer = c.u8("IVF quantizer")?;
            Some(KnnVectorsFormat::ivf_quantized(size, rerank, quantizer).map_err(invalid)?)
        }
        7 => {
            let (m, beam) = graph(c)?;
            let quantizer = c.u8("HNSW quantizer")?;
            Some(KnnVectorsFormat::hnsw_quantized(quantizer, m, beam).map_err(invalid)?)
        }
        8 => {
            let (m, beam) = graph(c)?;
            let quantizer = c.u8("disk HNSW quantizer")?;
            Some(KnnVectorsFormat::hnsw_disk(quantizer, m, beam).map_err(invalid)?)
        }
        _ => return Err(decode_error("vectors format")),
    };
    if encoding == VectorEncoding::Byte
        && matches!(format, Some(KnnVectorsFormat::HnswHalfFloat { .. }))
    {
        set_last_error("a byte vector field cannot be stored as half floats");
        return Err(Status::InvalidArgument);
    }
    if encoding == VectorEncoding::Float32
        && matches!(format, Some(KnnVectorsFormat::HnswHamming { .. }))
    {
        set_last_error("a float vector field cannot be scored by Hamming distance");
        return Err(Status::InvalidArgument);
    }
    if format.as_ref().is_some_and(|f| {
        encoding == VectorEncoding::Byte
            && matches!(
                f,
                KnnVectorsFormat::HnswScalarQuantized { .. }
                    | KnnVectorsFormat::ScalarQuantized { .. }
            )
    }) {
        set_last_error("a byte vector field cannot be scalar quantized");
        return Err(Status::InvalidArgument);
    }
    Ok((encoding, similarity, format))
}

/// `i32 count`, then `count` pairs of `(i32 len, utf8)` key and value.
pub fn decode_user_data(blob: &[u8]) -> Result<Vec<(String, String)>, Status> {
    let mut c = Cursor::new(blob);
    let out = decode_user_data_pairs(
        &mut c,
        ["user data count", "user data key", "user data value"],
    )?;
    c.finish("trailing bytes after the user data")?;
    Ok(out)
}

/// `i32 count`, then `count` pairs of `(i32 len, utf8)` key and value;
/// `what` names the count, a key and a value in errors.
fn decode_user_data_pairs(
    c: &mut Cursor<'_>,
    [count, key, value]: [&str; 3],
) -> Result<Vec<(String, String)>, Status> {
    let n = c.count(8, count)?;
    let mut out = try_with_capacity(n)?;
    for _ in 0..n {
        let k = c.string(key)?;
        let v = c.string(value)?;
        out.push((k, v));
    }
    Ok(out)
}

/// Blob builders the decode tests, and the writer's reader-driven tests, share.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    /// A little-endian blob builder mirroring the JVM side's.
    #[derive(Default, Clone)]
    pub struct Blob(pub Vec<u8>);

    impl Blob {
        pub fn u8(mut self, v: u8) -> Self {
            self.0.push(v);
            self
        }
        pub fn i32(mut self, v: i32) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        pub fn i64(mut self, v: i64) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        pub fn bytes(self, b: &[u8]) -> Self {
            let mut s = self.i32(b.len() as i32);
            s.0.extend_from_slice(b);
            s
        }
        pub fn f32(self, v: f32) -> Self {
            self.i32(v.to_bits() as i32)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::test_support::Blob;
    use super::*;

    #[test]
    fn every_value_kind_decodes() {
        let d = Blob::default()
            .i32(6)
            .i32(0)
            .u8(0)
            .bytes(b"s")
            .i32(0)
            .u8(1)
            .bytes(&[1, 2])
            .i32(0)
            .u8(2)
            .bytes(&7i32.to_le_bytes())
            .i32(0)
            .u8(3)
            .bytes(&7i64.to_le_bytes())
            .i32(0)
            .u8(4)
            .bytes(&1.5f32.to_le_bytes())
            .i32(0)
            .u8(5)
            .bytes(&1.5f64.to_le_bytes())
            .i32(0)
            .i32(2)
            .i32(1)
            .u8(0)
            .i64(7)
            .i32(1)
            .u8(1)
            .bytes(&[9])
            .i32(1)
            .i32(2)
            .bytes(&[0, 1, 2, 3]);
        let op = decode_op(
            &Blob::default()
                .u8(OP_ADD)
                .i32(1)
                .0
                .into_iter()
                .chain(d.0)
                .collect::<Vec<_>>(),
        );
        let Ok(Op::Add { docs, .. }) = op else {
            panic!("decoded");
        };
        let doc = &docs[0];
        assert_eq!(doc.stored.len(), 6);
        assert!(matches!(doc.stored[5].value, FieldValue::Double(v) if v == 1.5));
        assert!(matches!(doc.fields.doc_values[1].value, FieldValue::Binary(ref b) if b == &[9]));
        assert_eq!(doc.fields.points.len(), 1);
        for bad in [
            Blob::default()
                .i32(1)
                .i32(0)
                .u8(9)
                .bytes(b"x")
                .i32(0)
                .i32(0)
                .i32(0),
            Blob::default()
                .i32(1)
                .i32(0)
                .u8(2)
                .bytes(b"x")
                .i32(0)
                .i32(0)
                .i32(0),
            Blob::default()
                .i32(1)
                .i32(0)
                .u8(0)
                .bytes(&[0xff])
                .i32(0)
                .i32(0)
                .i32(0),
            Blob::default()
                .i32(0)
                .i32(1)
                .i32(0)
                .u8(3)
                .i32(0)
                .i32(0)
                .i32(0),
            Blob::default().i32(0).i32(0).i32(1).i32(0).u8(9).i32(0),
            Blob::default().i32(-1),
        ] {
            let blob: Vec<u8> = Blob::default()
                .u8(OP_ADD)
                .i32(1)
                .0
                .into_iter()
                .chain(bad.0)
                .collect();
            assert!(matches!(decode_op(&blob), Err(Status::Decode)));
        }
        // A freq whose positions or offsets cannot fit.
        let short = Blob::default()
            .i32(0)
            .i32(1)
            .i32(0)
            .u8(0)
            .i32(1)
            .bytes(b"t")
            .i32(1000)
            .u8(1);
        let blob: Vec<u8> = Blob::default()
            .u8(OP_ADD)
            .i32(1)
            .0
            .into_iter()
            .chain(short.0)
            .collect();
        assert!(matches!(decode_op(&blob), Err(Status::Decode)));
        let short = Blob::default()
            .i32(0)
            .i32(1)
            .i32(0)
            .u8(0)
            .i32(1)
            .bytes(b"t")
            .i32(1000)
            .u8(2);
        let blob: Vec<u8> = Blob::default()
            .u8(OP_ADD)
            .i32(1)
            .0
            .into_iter()
            .chain(short.0)
            .collect();
        assert!(matches!(decode_op(&blob), Err(Status::Decode)));
        let vector = Blob::default()
            .bytes(b"v")
            .u8(0)
            .u8(0)
            .u8(0)
            .u8(0)
            .u8(0)
            .u8(0)
            .i32(0)
            .i32(0)
            .i32(0)
            .i32(4)
            // float32, cosine, Lucene104HnswScalarQuantizedVectorsFormat(7 bits, 16, 100)
            .u8(0)
            .u8(2)
            .u8(2)
            .i32(16)
            .i32(100)
            .u8(2)
            .u8(0);
        let attributed = vector.clone().i32(1).bytes(b"k").bytes(b"v");
        assert_eq!(
            decode_field(&attributed.0).unwrap().0.attributes,
            vec![("k".to_string(), "v".to_string())]
        );
        assert!(decode_field(&vector.clone().i32(1).bytes(b"k").0).is_err());
        let vector = vector.i32(0);
        let (info, format) = decode_field(&vector.0).unwrap();
        assert_eq!(info.vector_dimension, 4);
        assert_eq!(info.vector_encoding, VectorEncoding::Float32);
        assert_eq!(
            info.vector_similarity_function,
            VectorSimilarityFunction::Cosine
        );
        assert_eq!(
            format,
            Some(
                KnnVectorsFormat::hnsw_scalar_quantized(ScalarEncoding::SevenBit, 16, 100).unwrap()
            )
        );
        // A byte field is never quantized; a graph's bounds are Lucene's.
        let schema = |encoding: u8, kind: u8, m: i32| {
            let b = Blob::default()
                .bytes(b"v")
                .u8(0)
                .u8(0)
                .u8(0)
                .u8(0)
                .u8(0)
                .u8(0)
                .i32(0)
                .i32(0)
                .i32(0)
                .i32(4)
                .u8(encoding)
                .u8(0)
                .u8(kind);
            let b = match kind {
                1 | 4 | 5 => b.i32(m).i32(100),
                2 => b.i32(m).i32(100).u8(2),
                3 => b.u8(2),
                // `m` is the cluster size, the rerank encoding f16, the
                // quantizer bbq.
                6 => b.i32(m).u8(1).u8(0),
                // The quantizer erq2.
                7 | 8 => b.i32(m).i32(100).u8(4),
                _ => b,
            };
            b.u8(0).i32(0).0
        };
        // IVF: any field (a byte one keeps its raw vectors alone), a checked
        // cluster size.
        for encoding in [0, 1] {
            assert_eq!(
                decode_field(&schema(encoding, 6, 384)).unwrap().1,
                Some(KnnVectorsFormat::ivf(384, 1).unwrap())
            );
        }
        assert!(decode_field(&schema(0, 6, 8)).is_err());
        // Quantized HNSW: any field (a byte one gets the graph alone).
        for encoding in [0, 1] {
            assert_eq!(
                decode_field(&schema(encoding, 7, 16)).unwrap().1,
                Some(KnnVectorsFormat::hnsw_quantized(4, 16, 100).unwrap())
            );
        }
        assert!(decode_field(&schema(0, 7, 0)).is_err());
        // Disk HNSW likewise.
        for encoding in [0, 1] {
            assert_eq!(
                decode_field(&schema(encoding, 8, 16)).unwrap().1,
                Some(KnnVectorsFormat::hnsw_disk(4, 16, 100).unwrap())
            );
        }
        assert!(decode_field(&schema(0, 8, 0)).is_err());
        assert!(decode_field(&schema(0, 6, -1)).is_err());
        assert!(decode_field(&schema(1, 2, 16)).is_err());
        assert!(decode_field(&schema(1, 3, 16)).is_err());
        assert!(decode_field(&schema(1, 1, 16)).is_ok());
        // Hamming scoring is for byte (packed bit) fields only.
        assert_eq!(
            decode_field(&schema(1, 4, 16)).unwrap().1,
            Some(KnnVectorsFormat::hnsw_hamming(16, 100).unwrap())
        );
        assert!(decode_field(&schema(0, 4, 16)).is_err());
        // Half floats are for float fields only.
        assert_eq!(
            decode_field(&schema(0, 5, 16)).unwrap().1,
            Some(KnnVectorsFormat::hnsw_half_float(16, 100).unwrap())
        );
        assert!(decode_field(&schema(1, 5, 16)).is_err());
        assert!(decode_field(&schema(0, 1, 0)).is_err());
        assert!(decode_field(&schema(0, 1, 513)).is_err());
        assert_eq!(decode_field(&schema(0, 0, 0)).unwrap().1, None);
        assert!(decode_field(&schema(0, 9, 0)).is_err());
        assert_eq!(
            parse_vectors_format("hnsw:16:100"),
            Some(KnnVectorsFormat::hnsw(16, 100).unwrap())
        );
        assert_eq!(
            parse_vectors_format("hnsw_sq:3:32:200"),
            Some(
                KnnVectorsFormat::hnsw_scalar_quantized(
                    ScalarEncoding::SingleBitQueryNibble,
                    32,
                    200
                )
                .unwrap()
            )
        );
        assert_eq!(
            parse_vectors_format("hnsw_hamming:8:64"),
            Some(KnnVectorsFormat::hnsw_hamming(8, 64).unwrap())
        );
        assert_eq!(
            parse_vectors_format("hnsw_half:8:64"),
            Some(KnnVectorsFormat::hnsw_half_float(8, 64).unwrap())
        );
        assert_eq!(
            parse_vectors_format("ivf:384:1"),
            Some(KnnVectorsFormat::ivf(384, 1).unwrap())
        );
        assert_eq!(
            parse_vectors_format("ivf:384:1:6"),
            Some(KnnVectorsFormat::ivf_quantized(384, 1, 6).unwrap())
        );
        assert_eq!(
            parse_vectors_format("hnsw_rq:11:16:100"),
            Some(KnnVectorsFormat::hnsw_quantized(11, 16, 100).unwrap())
        );
        assert_eq!(
            parse_vectors_format("hnsw_disk:0:16:100"),
            Some(KnnVectorsFormat::hnsw_disk(0, 16, 100).unwrap())
        );
        assert_eq!(
            parse_vectors_format("sq:1"),
            Some(KnnVectorsFormat::scalar_quantized(
                ScalarEncoding::PackedNibble
            ))
        );
        for bad in [
            "",
            "hnsw",
            "hnsw:0:100",
            "hnsw_sq:9:16:100",
            "flat:1",
            "hnsw:a:b",
            "ivf:8:0",
            "ivf:384:4",
            "ivf:384",
            "ivf:384:1:13",
            "ivf:384:1:x",
            "hnsw_rq:12:16:100",
            "hnsw_rq:4:0:100",
            "hnsw_rq:4:16",
            "hnsw_disk:12:16:100",
            "hnsw_disk:0:0:100",
            "hnsw_disk:0:16",
        ] {
            assert_eq!(parse_vectors_format(bad), None, "{bad}");
        }
        assert!(decode_user_data(&Blob::default().i32(0).u8(0).0).is_err());
    }
}
