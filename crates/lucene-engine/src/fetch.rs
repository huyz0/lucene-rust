//! A hit's stored fields for the plugin's fetch phase and get API (read path
//! R6): `StoredFields.document(docID, visitor)` answered natively, the
//! document's fields decoded here and handed over in one call for the
//! plugin's `StoredFields` to replay into OpenSearch's visitor.
//!
//! # The document blob
//!
//! Little-endian: `count: i32`, then per field, in stored order, `number: i32`
//! (the field number), `type: u8` and the value -- [`STRING`] and [`BINARY`]
//! as `len: i32` and the bytes (UTF-8 for a string), [`INT`] an `i32`,
//! [`LONG`] an `i64`, [`FLOAT`] an `f32`'s bits, [`DOUBLE`] an `f64`'s bits.
//!
//! In a segment whose `_source` has k-NN vectors masked out (derived source), the `_source`
//! field is [`DERIVED_SOURCE`] with no value: splicing the vectors back in costs far more than
//! the rest of the document, and most fetches (`_source: false`, `_id` only) never read it. The
//! plugin asks [`source_blob`] for it when its visitor does.
//!
//! A string or binary value of at least [`DEFER_BYTES`] is left out too, as [`DEFERRED`]: its
//! type ([`STRING`] or [`BINARY`]) as a `u8`, `len: i32` and `ordinal: i32` (its position among
//! the document's fields). Such a value is mostly a large `_source`, which a `_source: false` or
//! `stored_fields` fetch never reads, and the codec leaves its bytes compressed until someone does.
//! The plugin asks [`stored_value`] for it when its visitor wants it.
//!
//! # The doc-values blob
//!
//! One document's values of the fields the plugin names
//! ([`doc_values_blob`]), for the reads OpenSearch makes per hit -- a
//! derived `_source`, `docvalue_fields`. Little-endian: `count: i32`, then
//! per named field, in the order named, `number: i32`, `kind: u8` and the
//! values: [`DV_NONE`] (the document has none, or the field no doc values
//! of either kind) nothing more; [`DV_SORTED_NUMERIC`] `n: i32` and `n`
//! `i64`s, ascending; [`DV_SORTED_SET`] `n: i32` and per value its ordinal
//! (`i64`), `len: i32` and the bytes, ordinals ascending.

use lucene_codecs::stored_fields::{self, StoredFieldVisitor, VisitStatus};

use crate::reader_handle::JvmReaderHandle;
use crate::status::{set_last_error, Status};

pub const STRING: u8 = 0;
pub const BINARY: u8 = 1;
pub const INT: u8 = 2;
pub const LONG: u8 = 3;
pub const FLOAT: u8 = 4;
pub const DOUBLE: u8 = 5;
/// A derived `_source`, its value left out: [`source_blob`] reads it.
pub const DERIVED_SOURCE: u8 = 6;
/// A large string or binary value left out: [`stored_value`] reads it.
pub const DEFERRED: u8 = 7;
/// The shortest value a document blob leaves out ([`DEFERRED`]). At this size a value spans
/// many compression units, so reading it on its own later costs the decompression of at most the
/// one unit it shares with the fields before it.
pub const DEFER_BYTES: usize = 1 << 20;

/// Segment `segment` of the handle's reader, checked.
fn segment_reader<'h>(
    h: &'h JvmReaderHandle,
    segment: i32,
    what: &str,
) -> Result<&'h lucene_search::directory_reader::SegmentReader, Status> {
    let readers = h.reader.segment_readers();
    usize::try_from(segment)
        .ok()
        .and_then(|s| readers.get(s))
        .ok_or_else(|| {
            set_last_error(format!("{what}: segment {segment} of {}", readers.len()));
            Status::InvalidArgument
        })
}

fn check_doc(
    reader: &lucene_search::directory_reader::SegmentReader,
    doc: i32,
    what: &str,
) -> Result<(), Status> {
    if doc < 0 || doc >= reader.max_doc {
        set_last_error(format!("{what}: document {doc} of {}", reader.max_doc));
        return Err(Status::InvalidArgument);
    }
    Ok(())
}

pub const DV_NONE: u8 = 0;
pub const DV_SORTED_NUMERIC: u8 = 1;
pub const DV_SORTED_SET: u8 = 2;

/// Segment `segment`'s document `doc` (segment-local) of the handle's reader,
/// encoded (see the module doc).
pub fn document_blob(h: &JvmReaderHandle, segment: i32, doc: i32) -> Result<Vec<u8>, Status> {
    const WHAT: &str = "ffi_jvm_reader_document";
    let reader = segment_reader(h, segment, WHAT)?;
    check_doc(reader, doc, WHAT)?;
    let mut blob = Blob {
        out: 0i32.to_le_bytes().to_vec(),
        count: 0,
        derived_source: reader.derived_source_number(),
        ordinal: -1,
    };
    let stored = reader.visit_stored_document(doc, &mut blob).map_err(|e| {
        set_last_error(format!("reading stored fields: {e}"));
        Status::Decode
    })?;
    if !stored {
        // Java's reader has stored-fields files for every segment (an empty
        // one when nothing is stored); without them the plugin must ask
        // Lucene rather than replay a document with no fields.
        set_last_error(format!(
            "ffi_jvm_reader_document: segment {segment} has no stored-fields files"
        ));
        return Err(Status::Decode);
    }
    let Blob { mut out, count, .. } = blob;
    out[..4].copy_from_slice(&count.to_le_bytes());
    Ok(out)
}

/// Segment `segment`'s document `doc` (segment-local) of the handle's reader:
/// its values of the fields numbered `fields`, encoded as the module doc's
/// doc-values blob. A field is read as `LeafReader.getSortedNumericDocValues`
/// or `getSortedSetDocValues` read it: by its exact doc-values type.
pub fn doc_values_blob(
    h: &JvmReaderHandle,
    segment: i32,
    doc: i32,
    fields: &[i32],
) -> Result<Vec<u8>, Status> {
    const WHAT: &str = "ffi_jvm_reader_doc_values";
    let reader = segment_reader(h, segment, WHAT)?;
    check_doc(reader, doc, WHAT)?;
    encode_doc_values(reader, doc, fields).map_err(|e| {
        set_last_error(format!("reading doc values: {e}"));
        Status::Decode
    })
}

/// [`doc_values_blob`]'s encoding of document `doc` of any leaf.
pub fn encode_doc_values<R: lucene_search::reader::LeafReader + ?Sized>(
    reader: &R,
    doc: i32,
    fields: &[i32],
) -> lucene_search::Result<Vec<u8>> {
    use lucene_codecs::field_infos::DocValuesType;
    let len = |n: usize| {
        i32::try_from(n)
            .map_err(|_| lucene_search::Error::IllegalArgument("too many values".into()))
    };
    let mut out = len(fields.len())?.to_le_bytes().to_vec();
    let infos = reader.field_infos();
    for &number in fields {
        out.extend_from_slice(&number.to_le_bytes());
        let info = infos.field_by_number(number);
        let name = info.map(|i| i.name.as_str()).unwrap_or_default();
        match info.map(|i| i.doc_values_type) {
            Some(DocValuesType::SortedNumeric) => {
                let Some(mut dv) = reader.sorted_numeric_doc_values(name)? else {
                    out.push(DV_NONE);
                    continue;
                };
                if !dv.advance_exact(doc)? {
                    out.push(DV_NONE);
                    continue;
                }
                let n = dv.doc_value_count();
                out.push(DV_SORTED_NUMERIC);
                out.extend_from_slice(&n.to_le_bytes());
                for _ in 0..n {
                    out.extend_from_slice(&dv.next_value()?.to_le_bytes());
                }
            }
            Some(DocValuesType::SortedSet) => {
                let Some(mut dv) = reader.sorted_set_doc_values(name)? else {
                    out.push(DV_NONE);
                    continue;
                };
                if !dv.advance_exact(doc)? {
                    out.push(DV_NONE);
                    continue;
                }
                let n = dv.doc_value_count();
                out.push(DV_SORTED_SET);
                out.extend_from_slice(&n.to_le_bytes());
                for _ in 0..n {
                    let ord = dv.next_ord()?;
                    let term = dv.lookup_ord(ord)?;
                    out.extend_from_slice(&ord.to_le_bytes());
                    out.extend_from_slice(&len(term.len())?.to_le_bytes());
                    out.extend_from_slice(&term);
                }
            }
            _ => out.push(DV_NONE),
        }
    }
    Ok(out)
}

/// The `_source` of segment `segment`'s document `doc` with its k-NN vectors put back
/// ([`lucene_search::directory_reader::SegmentReader::visit_source_document`]): the value a
/// [`DERIVED_SOURCE`] field of [`document_blob`] leaves out. Without `vectors`, the `_source` as
/// stored, each vector's masked value (the number `1`) where the vector would go: for a fetch
/// whose `_source` filter drops every vector field, whose output is then the same.
pub fn source_blob(
    h: &JvmReaderHandle,
    segment: i32,
    doc: i32,
    vectors: bool,
) -> Result<Vec<u8>, Status> {
    const WHAT: &str = "ffi_jvm_reader_derived_source";
    let reader = segment_reader(h, segment, WHAT)?;
    check_doc(reader, doc, WHAT)?;
    let Some(number) = reader.derived_source_number() else {
        set_last_error(format!("{WHAT}: segment {segment} has no derived source"));
        return Err(Status::InvalidArgument);
    };
    let mut source = SourceOnly {
        number,
        value: None,
    };
    let read = if vectors {
        reader.visit_source_document(doc, &mut source)
    } else {
        reader.visit_stored_document(doc, &mut source)
    };
    read.map_err(|e| {
        set_last_error(format!("reading stored fields: {e}"));
        Status::Decode
    })?;
    source.value.ok_or_else(|| {
        set_last_error(format!("{WHAT}: document {doc} stores no _source"));
        Status::Decode
    })
}

/// Segment `segment`'s document `doc` (segment-local): the value of its field at position
/// `ordinal`, which a document blob left out as [`DEFERRED`], as raw bytes (UTF-8 for a string).
pub fn stored_value(
    h: &JvmReaderHandle,
    segment: i32,
    doc: i32,
    ordinal: i32,
) -> Result<Vec<u8>, Status> {
    const WHAT: &str = "ffi_jvm_reader_stored_value";
    let reader = segment_reader(h, segment, WHAT)?;
    check_doc(reader, doc, WHAT)?;
    let mut value = NthValue {
        ordinal,
        at: 0,
        value: None,
    };
    reader.visit_stored_document(doc, &mut value).map_err(|e| {
        set_last_error(format!("reading stored fields: {e}"));
        Status::Decode
    })?;
    value.value.ok_or_else(|| {
        set_last_error(format!(
            "{WHAT}: document {doc} has no string or binary field at {ordinal}"
        ));
        Status::InvalidArgument
    })
}

/// Keeps the string or binary value of the field at position `ordinal` and stops there.
struct NthValue {
    ordinal: i32,
    at: i32,
    value: Option<Vec<u8>>,
}

impl StoredFieldVisitor for NthValue {
    fn needs_field(&mut self, _field_number: i32) -> stored_fields::Result<VisitStatus> {
        let at = self.at;
        self.at = self.at.saturating_add(1);
        Ok(match at.cmp(&self.ordinal) {
            std::cmp::Ordering::Less => VisitStatus::No,
            std::cmp::Ordering::Equal => VisitStatus::Yes,
            std::cmp::Ordering::Greater => VisitStatus::Stop,
        })
    }

    fn binary_field(&mut self, _: i32, value: &[u8]) -> stored_fields::Result<()> {
        self.value = Some(value.to_vec());
        Ok(())
    }

    fn string_field(&mut self, _: i32, value: &str) -> stored_fields::Result<()> {
        self.value = Some(value.as_bytes().to_vec());
        Ok(())
    }
}

/// Keeps field `number`'s binary value and stops there.
struct SourceOnly {
    number: i32,
    value: Option<Vec<u8>>,
}

impl StoredFieldVisitor for SourceOnly {
    fn needs_field(&mut self, field_number: i32) -> stored_fields::Result<VisitStatus> {
        Ok(match (self.value.is_some(), field_number == self.number) {
            (true, _) => VisitStatus::Stop,
            (false, true) => VisitStatus::Yes,
            (false, false) => VisitStatus::No,
        })
    }

    fn binary_field(&mut self, _field_number: i32, value: &[u8]) -> stored_fields::Result<()> {
        self.value = Some(value.to_vec());
        Ok(())
    }

    fn string_field(&mut self, _: i32, _: &str) -> stored_fields::Result<()> {
        Ok(())
    }
    fn int_field(&mut self, _: i32, _: i32) -> stored_fields::Result<()> {
        Ok(())
    }
    fn long_field(&mut self, _: i32, _: i64) -> stored_fields::Result<()> {
        Ok(())
    }
    fn float_field(&mut self, _: i32, _: f32) -> stored_fields::Result<()> {
        Ok(())
    }
    fn double_field(&mut self, _: i32, _: f64) -> stored_fields::Result<()> {
        Ok(())
    }
}

/// Encodes every field it is shown (see the module doc), straight from the
/// decompressed bytes: no [`lucene_codecs::stored_fields::Document`] in
/// between. `out` starts with room for the count, written last.
pub struct Blob {
    pub out: Vec<u8>,
    pub count: i32,
    /// The `_source` field number in a derived-source segment: written as [`DERIVED_SOURCE`].
    pub derived_source: Option<i32>,
    /// The position of the field being read, counted from 0.
    pub ordinal: i32,
}

impl Blob {
    fn field(&mut self, number: i32, ty: u8) -> stored_fields::Result<()> {
        self.count = self.count.checked_add(1).ok_or_else(too_large)?;
        self.out.extend_from_slice(&number.to_le_bytes());
        self.out.push(ty);
        Ok(())
    }

    fn bytes(&mut self, number: i32, ty: u8, b: &[u8]) -> stored_fields::Result<()> {
        self.field(number, ty)?;
        let len = i32::try_from(b.len()).map_err(|_| too_large())?;
        self.out.extend_from_slice(&len.to_le_bytes());
        self.out.extend_from_slice(b);
        Ok(())
    }
}

fn too_large() -> stored_fields::Error {
    lucene_store::Error::Corrupted("a document too large for the JVM".into()).into()
}

impl StoredFieldVisitor for Blob {
    fn needs_field(&mut self, _field_number: i32) -> stored_fields::Result<VisitStatus> {
        self.ordinal = self.ordinal.checked_add(1).ok_or_else(too_large)?;
        Ok(VisitStatus::Yes)
    }

    fn defer_values_from(&self) -> usize {
        DEFER_BYTES
    }

    /// A large value as [`DEFERRED`], or a large derived `_source` as [`DERIVED_SOURCE`]: the
    /// plugin reads either only when its visitor wants it.
    fn deferred_field(
        &mut self,
        number: i32,
        string: bool,
        length: usize,
    ) -> stored_fields::Result<()> {
        if self.derived_source == Some(number) {
            return self.field(number, DERIVED_SOURCE);
        }
        self.field(number, DEFERRED)?;
        self.out.push(if string { STRING } else { BINARY });
        let len = i32::try_from(length).map_err(|_| too_large())?;
        self.out.extend_from_slice(&len.to_le_bytes());
        self.out.extend_from_slice(&self.ordinal.to_le_bytes());
        Ok(())
    }

    /// One allocation for the whole blob: a field is at most 12 bytes longer
    /// here than serialized (5 bytes of number and type against a 1-byte
    /// vlong header, and an 8-byte long or double against a 1-byte
    /// encoding, or a 4-byte length against a 1-byte vint). A count above
    /// the length is corrupt (every field takes a byte), so it is clamped
    /// rather than trusted.
    fn document_size(&mut self, serialized_len: usize, num_stored_fields: i64) {
        let fields = usize::try_from(num_stored_fields)
            .unwrap_or(0)
            .min(serialized_len);
        self.out
            .reserve(serialized_len.saturating_add(fields.saturating_mul(12)));
    }

    fn string_field(&mut self, number: i32, value: &str) -> stored_fields::Result<()> {
        self.bytes(number, STRING, value.as_bytes())
    }

    fn binary_field(&mut self, number: i32, value: &[u8]) -> stored_fields::Result<()> {
        // A `_source` the splice would refuse is refused here too, so the plugin reads the whole
        // document with Lucene as before; one the splice leaves unchanged stays inline.
        if self.derived_source == Some(number)
            && lucene_search::derived_source::needs_injection(value)?
        {
            return self.field(number, DERIVED_SOURCE);
        }
        self.bytes(number, BINARY, value)
    }

    fn int_field(&mut self, number: i32, value: i32) -> stored_fields::Result<()> {
        self.field(number, INT)?;
        self.out.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn long_field(&mut self, number: i32, value: i64) -> stored_fields::Result<()> {
        self.field(number, LONG)?;
        self.out.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn float_field(&mut self, number: i32, value: f32) -> stored_fields::Result<()> {
        self.field(number, FLOAT)?;
        self.out.extend_from_slice(&value.to_bits().to_le_bytes());
        Ok(())
    }

    fn double_field(&mut self, number: i32, value: f64) -> stored_fields::Result<()> {
        self.field(number, DOUBLE)?;
        self.out.extend_from_slice(&value.to_bits().to_le_bytes());
        Ok(())
    }
}

/// `StoredFields.prefetch` for a page of hits: the stored-fields chunks of
/// segment `segment`'s documents `docs` (segment-local) of `handle`'s
/// reader, fetched together where the segment is read by range
/// ([`lucene_search::directory_reader::SegmentReader::prefetch_stored_documents`]);
/// nothing to do over files in memory or mapped. The plugin calls it before
/// its fetch phase reads the documents one at a time.
pub fn prefetch_documents(h: &JvmReaderHandle, segment: i32, docs: &[i32]) -> Result<(), Status> {
    let readers = h.reader.segment_readers();
    let Some(reader) = usize::try_from(segment).ok().and_then(|s| readers.get(s)) else {
        set_last_error(format!(
            "ffi_jvm_reader_prefetch_documents: segment {segment} of {}",
            readers.len()
        ));
        return Err(Status::InvalidArgument);
    };
    reader.prefetch_stored_documents(docs).map_err(|e| {
        set_last_error(format!("prefetching stored fields: {e}"));
        Status::Decode
    })
}
