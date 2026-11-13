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

use crate::error::{set_last_error, FfiStatus};
use crate::jvm_reader;

pub(crate) const STRING: u8 = 0;
pub(crate) const BINARY: u8 = 1;
pub(crate) const INT: u8 = 2;
pub(crate) const LONG: u8 = 3;
pub(crate) const FLOAT: u8 = 4;
pub(crate) const DOUBLE: u8 = 5;
/// A derived `_source`, its value left out: [`source_blob`] reads it.
pub(crate) const DERIVED_SOURCE: u8 = 6;
/// A large string or binary value left out: [`stored_value`] reads it.
pub(crate) const DEFERRED: u8 = 7;
/// The shortest value a document blob leaves out ([`DEFERRED`]). At this size a value spans
/// many compression units, so reading it on its own later costs the decompression of at most the
/// one unit it shares with the fields before it.
pub(crate) const DEFER_BYTES: usize = 1 << 20;

/// Segment `segment` of `handle`'s reader, checked.
fn segment_reader<'h>(
    h: &'h crate::registry::JvmReaderHandle,
    segment: i32,
    what: &str,
) -> Result<&'h lucene_search::directory_reader::SegmentReader, FfiStatus> {
    let readers = h.reader.segment_readers();
    usize::try_from(segment)
        .ok()
        .and_then(|s| readers.get(s))
        .ok_or_else(|| {
            set_last_error(format!("{what}: segment {segment} of {}", readers.len()));
            FfiStatus::InvalidArgument
        })
}

fn check_doc(
    reader: &lucene_search::directory_reader::SegmentReader,
    doc: i32,
    what: &str,
) -> Result<(), FfiStatus> {
    if doc < 0 || doc >= reader.max_doc {
        set_last_error(format!("{what}: document {doc} of {}", reader.max_doc));
        return Err(FfiStatus::InvalidArgument);
    }
    Ok(())
}

pub(crate) const DV_NONE: u8 = 0;
pub(crate) const DV_SORTED_NUMERIC: u8 = 1;
pub(crate) const DV_SORTED_SET: u8 = 2;

/// Segment `segment`'s document `doc` (segment-local) of `handle`'s reader,
/// encoded (see the module doc).
pub(crate) fn document_blob(handle: u64, segment: i32, doc: i32) -> Result<Vec<u8>, FfiStatus> {
    const WHAT: &str = "ffi_jvm_reader_document";
    let h = jvm_reader::lookup(
        handle,
        "ffi_jvm_reader_document: unknown or already-closed handle",
    )?;
    let reader = segment_reader(&h, segment, WHAT)?;
    check_doc(reader, doc, WHAT)?;
    let mut blob = Blob {
        out: 0i32.to_le_bytes().to_vec(),
        count: 0,
        derived_source: reader.derived_source_number(),
        ordinal: -1,
    };
    let stored = reader.visit_stored_document(doc, &mut blob).map_err(|e| {
        set_last_error(format!("reading stored fields: {e}"));
        FfiStatus::Decode
    })?;
    if !stored {
        // Java's reader has stored-fields files for every segment (an empty
        // one when nothing is stored); without them the plugin must ask
        // Lucene rather than replay a document with no fields.
        set_last_error(format!(
            "ffi_jvm_reader_document: segment {segment} has no stored-fields files"
        ));
        return Err(FfiStatus::Decode);
    }
    let Blob { mut out, count, .. } = blob;
    out[..4].copy_from_slice(&count.to_le_bytes());
    Ok(out)
}

/// Segment `segment`'s document `doc` (segment-local) of `handle`'s reader:
/// its values of the fields numbered `fields`, encoded as the module doc's
/// doc-values blob. A field is read as `LeafReader.getSortedNumericDocValues`
/// or `getSortedSetDocValues` read it: by its exact doc-values type.
pub(crate) fn doc_values_blob(
    handle: u64,
    segment: i32,
    doc: i32,
    fields: &[i32],
) -> Result<Vec<u8>, FfiStatus> {
    const WHAT: &str = "ffi_jvm_reader_doc_values";
    let h = jvm_reader::lookup(
        handle,
        "ffi_jvm_reader_doc_values: unknown or already-closed handle",
    )?;
    let reader = segment_reader(&h, segment, WHAT)?;
    check_doc(reader, doc, WHAT)?;
    encode_doc_values(reader, doc, fields).map_err(|e| {
        set_last_error(format!("reading doc values: {e}"));
        FfiStatus::Decode
    })
}

/// [`doc_values_blob`]'s encoding of document `doc` of any leaf.
pub(crate) fn encode_doc_values<R: lucene_search::reader::LeafReader + ?Sized>(
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
pub(crate) fn source_blob(
    handle: u64,
    segment: i32,
    doc: i32,
    vectors: bool,
) -> Result<Vec<u8>, FfiStatus> {
    const WHAT: &str = "ffi_jvm_reader_derived_source";
    let h = jvm_reader::lookup(
        handle,
        "ffi_jvm_reader_derived_source: unknown or already-closed handle",
    )?;
    let reader = segment_reader(&h, segment, WHAT)?;
    check_doc(reader, doc, WHAT)?;
    let Some(number) = reader.derived_source_number() else {
        set_last_error(format!("{WHAT}: segment {segment} has no derived source"));
        return Err(FfiStatus::InvalidArgument);
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
        FfiStatus::Decode
    })?;
    source.value.ok_or_else(|| {
        set_last_error(format!("{WHAT}: document {doc} stores no _source"));
        FfiStatus::Decode
    })
}

/// Segment `segment`'s document `doc` (segment-local): the value of its field at position
/// `ordinal`, which a document blob left out as [`DEFERRED`], as raw bytes (UTF-8 for a string).
pub(crate) fn stored_value(
    handle: u64,
    segment: i32,
    doc: i32,
    ordinal: i32,
) -> Result<Vec<u8>, FfiStatus> {
    const WHAT: &str = "ffi_jvm_reader_stored_value";
    let h = jvm_reader::lookup(
        handle,
        "ffi_jvm_reader_stored_value: unknown or already-closed handle",
    )?;
    let reader = segment_reader(&h, segment, WHAT)?;
    check_doc(reader, doc, WHAT)?;
    let mut value = NthValue {
        ordinal,
        at: 0,
        value: None,
    };
    reader.visit_stored_document(doc, &mut value).map_err(|e| {
        set_last_error(format!("reading stored fields: {e}"));
        FfiStatus::Decode
    })?;
    value.value.ok_or_else(|| {
        set_last_error(format!(
            "{WHAT}: document {doc} has no string or binary field at {ordinal}"
        ));
        FfiStatus::InvalidArgument
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
struct Blob {
    out: Vec<u8>,
    count: i32,
    /// The `_source` field number in a derived-source segment: written as [`DERIVED_SOURCE`].
    derived_source: Option<i32>,
    /// The position of the field being read, counted from 0.
    ordinal: i32,
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

/// [`document_blob`] into `out` (`cap` bytes, null when 0), its length into
/// `out_len` either way -- a too-small `out` is
/// [`FfiStatus::BufferTooSmall`], so a caller can size a second call.
///
/// # Safety
/// `out` must be valid for `cap` bytes (null when 0) and `out_len` writable.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_reader_document(
    handle: u64,
    segment: i32,
    doc: i32,
    out: *mut u8,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    crate::error::guard(|| {
        if out_len.is_null() || (out.is_null() && cap > 0) {
            return Err(FfiStatus::NullPointer);
        }
        let encoded = document_blob(handle, segment, doc)?;
        // SAFETY: `out_len` is non-null and writable (caller contract).
        unsafe { *out_len = encoded.len() };
        if encoded.len() > cap {
            set_last_error(format!(
                "document: {} bytes, buffer holds {cap}",
                encoded.len()
            ));
            return Err(FfiStatus::BufferTooSmall);
        }
        // SAFETY: `out` is valid for `cap >= encoded.len()` bytes; `encoded`
        // is never empty (it holds the count).
        unsafe { std::ptr::copy_nonoverlapping(encoded.as_ptr(), out, encoded.len()) };
        Ok(())
    })
}

/// `StoredFields.prefetch` for a page of hits: the stored-fields chunks of
/// segment `segment`'s documents `docs` (segment-local) of `handle`'s
/// reader, fetched together where the segment is read by range
/// ([`lucene_search::directory_reader::SegmentReader::prefetch_stored_documents`]);
/// nothing to do over files in memory or mapped. The plugin calls it before
/// its fetch phase reads the documents one at a time.
pub(crate) fn prefetch_documents(handle: u64, segment: i32, docs: &[i32]) -> Result<(), FfiStatus> {
    let h = jvm_reader::lookup(
        handle,
        "ffi_jvm_reader_prefetch_documents: unknown or already-closed handle",
    )?;
    let readers = h.reader.segment_readers();
    let Some(reader) = usize::try_from(segment).ok().and_then(|s| readers.get(s)) else {
        set_last_error(format!(
            "ffi_jvm_reader_prefetch_documents: segment {segment} of {}",
            readers.len()
        ));
        return Err(FfiStatus::InvalidArgument);
    };
    reader.prefetch_stored_documents(docs).map_err(|e| {
        set_last_error(format!("prefetching stored fields: {e}"));
        FfiStatus::Decode
    })
}

/// [`prefetch_documents`] of the `count` documents at `docs`.
///
/// # Safety
/// `docs` must be valid for `count` `i32` reads (null when 0).
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_reader_prefetch_documents(
    handle: u64,
    segment: i32,
    docs: *const i32,
    count: usize,
) -> i32 {
    crate::error::guard(|| {
        if docs.is_null() && count > 0 {
            return Err(FfiStatus::NullPointer);
        }
        let docs: &[i32] = if count == 0 {
            &[]
        } else {
            // SAFETY: caller contract, non-null checked above.
            unsafe { std::slice::from_raw_parts(docs, count) }
        };
        prefetch_documents(handle, segment, docs)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jvm_reader::tests::open;
    use lucene_codecs::stored_fields::FieldValue;

    fn i32_at(b: &[u8], at: &mut usize) -> i32 {
        let v = i32::from_le_bytes(b[*at..*at + 4].try_into().unwrap());
        *at += 4;
        v
    }

    #[test]
    fn a_document_comes_back_with_its_stored_fields() {
        let h = open();
        // The fixture's documents store no fields: each answers an empty list
        // (the next test reads a fixture that stores them).
        for segment in 0..2 {
            for doc in 0..4 {
                assert!(decode(&document_blob(h, segment, doc).unwrap()).is_empty());
            }
        }
        // Out of range, and through the C entry point with a small buffer.
        assert_eq!(
            document_blob(h, 9, 0).err(),
            Some(FfiStatus::InvalidArgument)
        );
        assert_eq!(
            document_blob(h, 0, 99).err(),
            Some(FfiStatus::InvalidArgument)
        );
        assert_eq!(
            document_blob(h, 0, -1).err(),
            Some(FfiStatus::InvalidArgument)
        );
        let mut len = 0usize;
        let rc = unsafe { ffi_jvm_reader_document(h, 0, 0, std::ptr::null_mut(), 0, &mut len) };
        assert_eq!(rc, FfiStatus::BufferTooSmall.code());
        let mut buf = vec![0u8; len];
        let rc = unsafe { ffi_jvm_reader_document(h, 0, 0, buf.as_mut_ptr(), len, &mut len) };
        assert_eq!(rc, 0);
        assert_eq!(buf, document_blob(h, 0, 0).unwrap());
        let rc = unsafe { ffi_jvm_reader_document(h, 0, 0, std::ptr::null_mut(), 8, &mut len) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        crate::jvm_reader::ffi_close_jvm_reader(h);
        assert!(document_blob(h, 0, 0).is_err());
    }

    #[test]
    fn a_documents_doc_values_come_back_as_its_leaf_reads_them() {
        use lucene_codecs::field_infos::DocValuesType;
        use lucene_search::reader::LeafReader;
        let h = crate::jvm_reader::tests::open_bwc("10.4.0");
        let held = jvm_reader::lookup(h, "test").unwrap();
        let mut seen = [0usize; 3];
        for (segment, reader) in held.reader.segment_readers().iter().enumerate() {
            let segment = i32::try_from(segment).unwrap();
            let infos = reader.field_infos();
            // Every field, and a number no field has.
            let mut fields: Vec<i32> = infos.fields.iter().map(|f| f.number).collect();
            fields.push(9999);
            for doc in 0..reader.max_doc {
                let blob = doc_values_blob(h, segment, doc, &fields).unwrap();
                let mut at = 0;
                assert_eq!(i32_at(&blob, &mut at), i32::try_from(fields.len()).unwrap());
                for &number in &fields {
                    assert_eq!(i32_at(&blob, &mut at), number);
                    let kind = blob[at];
                    at += 1;
                    seen[usize::from(kind)] += 1;
                    let info = infos.field_by_number(number);
                    match kind {
                        DV_NONE => {
                            let has = match info.map(|i| i.doc_values_type) {
                                Some(DocValuesType::SortedNumeric) => reader
                                    .sorted_numeric_doc_values(&info.unwrap().name)
                                    .unwrap()
                                    .unwrap()
                                    .advance_exact(doc)
                                    .unwrap(),
                                Some(DocValuesType::SortedSet) => reader
                                    .sorted_set_doc_values(&info.unwrap().name)
                                    .unwrap()
                                    .unwrap()
                                    .advance_exact(doc)
                                    .unwrap(),
                                _ => false,
                            };
                            assert!(!has, "field {number} doc {doc}");
                        }
                        DV_SORTED_NUMERIC => {
                            let mut dv = reader
                                .sorted_numeric_doc_values(&info.unwrap().name)
                                .unwrap()
                                .unwrap();
                            assert!(dv.advance_exact(doc).unwrap());
                            let n = i32_at(&blob, &mut at);
                            assert_eq!(n, dv.doc_value_count());
                            for _ in 0..n {
                                let v = i64::from_le_bytes(blob[at..at + 8].try_into().unwrap());
                                at += 8;
                                assert_eq!(v, dv.next_value().unwrap());
                            }
                        }
                        DV_SORTED_SET => {
                            let mut dv = reader
                                .sorted_set_doc_values(&info.unwrap().name)
                                .unwrap()
                                .unwrap();
                            assert!(dv.advance_exact(doc).unwrap());
                            let n = i32_at(&blob, &mut at);
                            assert_eq!(n, dv.doc_value_count());
                            for _ in 0..n {
                                let ord = i64::from_le_bytes(blob[at..at + 8].try_into().unwrap());
                                at += 8;
                                assert_eq!(ord, dv.next_ord().unwrap());
                                let len = usize::try_from(i32_at(&blob, &mut at)).unwrap();
                                assert_eq!(blob[at..at + len], dv.lookup_ord(ord).unwrap()[..]);
                                at += len;
                            }
                        }
                        other => panic!("kind {other}"),
                    }
                }
                assert_eq!(at, blob.len());
            }
        }
        assert!(seen.iter().all(|&n| n > 0), "every kind read: {seen:?}");
        assert_eq!(
            doc_values_blob(h, 0, -1, &[]).err(),
            Some(FfiStatus::InvalidArgument)
        );
        assert_eq!(
            doc_values_blob(h, 99, 0, &[]).err(),
            Some(FfiStatus::InvalidArgument)
        );
        drop(held);
        crate::jvm_reader::ffi_close_jvm_reader(h);
    }

    /// A blob decoded back into `(number, value)`s.
    fn decode(blob: &[u8]) -> Vec<(i32, FieldValue)> {
        let mut at = 0;
        let n = i32_at(blob, &mut at);
        let mut fields = Vec::new();
        for _ in 0..n {
            let number = i32_at(blob, &mut at);
            let ty = blob[at];
            at += 1;
            let (value, len) = match ty {
                DERIVED_SOURCE => (FieldValue::Binary(b"derived".to_vec()), 0),
                STRING | BINARY => {
                    let len = i32_at(blob, &mut at) as usize;
                    let b = blob[at..at + len].to_vec();
                    if ty == STRING {
                        (FieldValue::String(String::from_utf8(b).unwrap()), len)
                    } else {
                        (FieldValue::Binary(b), len)
                    }
                }
                INT => (FieldValue::Int(i32_at(blob, &mut at)), 0),
                FLOAT => (
                    FieldValue::Float(f32::from_bits(i32_at(blob, &mut at) as u32)),
                    0,
                ),
                // LONG or DOUBLE: every other type is a writer bug.
                _ => {
                    assert!(ty == LONG || ty == DOUBLE, "type {ty}");
                    let v = u64::from_le_bytes(blob[at..at + 8].try_into().unwrap());
                    let value = if ty == LONG {
                        FieldValue::Long(v as i64)
                    } else {
                        FieldValue::Double(f64::from_bits(v))
                    };
                    (value, 8)
                }
            };
            at += len;
            fields.push((number, value));
        }
        assert_eq!(at, blob.len());
        fields
    }

    #[test]
    fn prefetching_documents_checks_its_arguments_and_reads_the_same_documents() {
        let (_tmp, h) = open_copy(|_, _| {});
        let before = document_blob(h, 0, 0).unwrap();
        let docs = [0, 1, 2];
        // SAFETY: `docs` holds three `i32`s.
        let rc = unsafe { ffi_jvm_reader_prefetch_documents(h, 0, docs.as_ptr(), docs.len()) };
        assert_eq!(rc, 0);
        assert_eq!(document_blob(h, 0, 0).unwrap(), before);
        // SAFETY: a null pointer with no documents is allowed.
        assert_eq!(
            unsafe { ffi_jvm_reader_prefetch_documents(h, 0, std::ptr::null(), 0) },
            0
        );
        // SAFETY: as above; the null with a count is refused.
        let rc = unsafe { ffi_jvm_reader_prefetch_documents(h, 0, std::ptr::null(), 1) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        assert!(prefetch_documents(h, 9, &docs).is_err());
        assert!(prefetch_documents(u64::MAX, 0, &docs).is_err());
        crate::jvm_reader::ffi_close_jvm_reader(h);
    }

    #[test]
    fn stored_documents_come_back_as_the_segment_reader_reads_them() {
        // A Lucene-written index whose documents store strings, binaries and
        // every numeric type (`fixtures/src/GenStoredFields.java`).
        let (tmp, h) = open_copy(|_, _| {});
        let reader = lucene_search::directory_reader::DirectoryReader::open(
            &lucene_store::directory::FsDirectory::open(tmp.path_str()),
        )
        .unwrap();
        let seg = &reader.segment_readers()[0];
        let mut types = std::collections::HashSet::new();
        for doc in 0..seg.max_doc {
            let got = decode(&document_blob(h, 0, doc).unwrap());
            let want: Vec<_> = seg
                .stored_document(doc)
                .unwrap()
                .unwrap()
                .fields
                .into_iter()
                .map(|f| (f.field_number, f.value))
                .collect();
            assert_eq!(got, want, "document {doc}");
            types.extend(got.iter().map(|(_, v)| std::mem::discriminant(v)));
        }
        assert!(types.len() >= 2, "the fixture stores several field types");
        crate::jvm_reader::ffi_close_jvm_reader(h);
    }

    /// A copy of the stored-fields fixture, changed by `mutate` (given the
    /// copy's path and the segment's id), opened as a handle.
    fn open_copy(
        mutate: impl FnOnce(&std::path::Path, [u8; 16]),
    ) -> (lucene_util::test_support::TempDir, u64) {
        let src = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/stored_fields_index"
        );
        let tmp = lucene_util::test_support::TempDir::new("jvm-fetch");
        for e in std::fs::read_dir(src).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), tmp.path().join(e.file_name())).unwrap();
        }
        let dir = tmp.path_str().to_string();
        let reader = lucene_search::directory_reader::DirectoryReader::open(
            &lucene_store::directory::FsDirectory::open(&dir),
        )
        .unwrap();
        let seg = &reader.segment_readers()[0];
        let max_docs = [seg.max_doc];
        mutate(tmp.path(), seg.segment_id());
        let infos = std::fs::read(format!("{dir}/segments_1")).unwrap();
        let counts = [0usize];
        let mut h = 0u64;
        let rc = unsafe {
            crate::jvm_reader::ffi_open_jvm_reader(
                dir.as_ptr().cast(),
                dir.len(),
                infos.as_ptr(),
                infos.len(),
                1,
                0,
                max_docs.as_ptr(),
                1,
                std::ptr::null(),
                counts.as_ptr(),
                &mut h,
            )
        };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        (tmp, h)
    }

    #[test]
    fn large_values_are_deferred_and_read_on_their_own() {
        use lucene_codecs::stored_fields::{Document, Mode, StoredField, StoredFieldsWriter};
        let big = |n: usize, seed: u8| {
            (0..n)
                .map(|i| (i as u8).wrapping_mul(seed))
                .collect::<Vec<u8>>()
        };
        let mut docs = Vec::new();
        // The fixture's stored-fields files rewritten: the segment's document count, each
        // document with small values around a large binary and a large string.
        let (_tmp, h) = open_copy(|dir, id| {
            let max_doc = lucene_search::directory_reader::DirectoryReader::open(
                &lucene_store::directory::FsDirectory::open(dir.to_str().unwrap()),
            )
            .unwrap()
            .segment_readers()[0]
                .max_doc;
            let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id, "");
            for d in 0..max_doc {
                let doc = Document {
                    fields: vec![
                        StoredField {
                            field_number: 0,
                            value: FieldValue::String(format!("id-{d}")),
                        },
                        StoredField {
                            field_number: 1,
                            value: FieldValue::Binary(big(DEFER_BYTES + d as usize, 7)),
                        },
                        StoredField {
                            field_number: 2,
                            value: FieldValue::Long(i64::from(d)),
                        },
                        StoredField {
                            field_number: 3,
                            value: FieldValue::String("x".repeat(DEFER_BYTES + 3)),
                        },
                        StoredField {
                            field_number: 4,
                            value: FieldValue::Binary(big(DEFER_BYTES - 1, 3)),
                        },
                    ],
                };
                w.add_document(&doc);
                docs.push(doc);
            }
            let (fdt, fdx, fdm) = w.finish();
            std::fs::write(dir.join("_0.fdt"), fdt).unwrap();
            std::fs::write(dir.join("_0.fdx"), fdx).unwrap();
            std::fs::write(dir.join("_0.fdm"), fdm).unwrap();
        });
        let doc = 1;
        let blob = document_blob(h, 0, doc).unwrap();
        // count, then: id string, deferred binary, long, deferred string, inline binary.
        let mut at = 0;
        assert_eq!(i32_at(&blob, &mut at), 5);
        assert_eq!(i32_at(&blob, &mut at), 0);
        assert_eq!(blob[at], STRING);
        at += 1;
        let len = i32_at(&blob, &mut at) as usize;
        assert_eq!(&blob[at..at + len], b"id-1");
        at += len;
        assert_eq!(i32_at(&blob, &mut at), 1);
        assert_eq!(&blob[at..at + 2], &[DEFERRED, BINARY]);
        at += 2;
        assert_eq!(i32_at(&blob, &mut at) as usize, DEFER_BYTES + 1);
        assert_eq!(i32_at(&blob, &mut at), 1);
        assert_eq!(i32_at(&blob, &mut at), 2);
        assert_eq!(blob[at], LONG);
        at += 1 + 8;
        assert_eq!(i32_at(&blob, &mut at), 3);
        assert_eq!(&blob[at..at + 2], &[DEFERRED, STRING]);
        at += 2;
        assert_eq!(i32_at(&blob, &mut at) as usize, DEFER_BYTES + 3);
        assert_eq!(i32_at(&blob, &mut at), 3);
        assert_eq!(i32_at(&blob, &mut at), 4);
        assert_eq!(blob[at], BINARY);
        at += 1;
        assert_eq!(i32_at(&blob, &mut at) as usize, DEFER_BYTES - 1);
        at += DEFER_BYTES - 1;
        assert_eq!(at, blob.len(), "the large values stay out of the blob");
        // Each deferred value on its own, as stored.
        let FieldValue::Binary(want) = &docs[doc as usize].fields[1].value else {
            unreachable!()
        };
        assert_eq!(&stored_value(h, 0, doc, 1).unwrap(), want);
        assert_eq!(
            stored_value(h, 0, doc, 3).unwrap(),
            "x".repeat(DEFER_BYTES + 3).into_bytes()
        );
        // A numeric field, or a position past the document, has no such value.
        assert_eq!(
            stored_value(h, 0, doc, 2).err(),
            Some(FfiStatus::InvalidArgument)
        );
        assert_eq!(
            stored_value(h, 0, doc, 9).err(),
            Some(FfiStatus::InvalidArgument)
        );
        assert_eq!(
            stored_value(h, 0, -1, 1).err(),
            Some(FfiStatus::InvalidArgument)
        );
        let (mut ptr, mut len) = (std::ptr::null_mut(), 0usize);
        let rc = unsafe {
            crate::ffm_bridge::ffi_jvm_reader_stored_value_alloc(h, 0, doc, 1, &mut ptr, &mut len)
        };
        assert_eq!(rc, 0);
        assert_eq!(len, want.len());
        unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
        crate::jvm_reader::ffi_close_jvm_reader(h);
    }

    #[test]
    fn a_document_that_does_not_decode_is_a_decode_error() {
        // The first chunk claims to start at document 127: no document of
        // the segment is in it.
        let (_tmp, h) = open_copy(|dir, _| {
            let fdt = dir.join("_0.fdt");
            let mut bytes = std::fs::read(&fdt).unwrap();
            // The index header (magic, `Lucene90StoredFieldsFastData`,
            // version, id, empty suffix) is 54 bytes; the chunk's `docBase`
            // vint follows.
            assert_eq!(&bytes[5..33], b"Lucene90StoredFieldsFastData");
            assert_eq!(bytes[54], 0, "the first chunk starts at document 0");
            bytes[54] = 127;
            std::fs::write(&fdt, bytes).unwrap();
        });
        assert_eq!(document_blob(h, 0, 0).err(), Some(FfiStatus::Decode));
        assert!(crate::error::last_error().contains("reading stored fields"));
        crate::jvm_reader::ffi_close_jvm_reader(h);
    }

    #[test]
    fn a_segment_without_stored_fields_files_is_refused_not_empty() {
        // The segment's `.si` no longer lists `.fdt`/`.fdx`/`.fdm`: the
        // plugin must read the document with Lucene, not replay nothing.
        let (_tmp, h) = open_copy(|dir, id| {
            let si_path = dir.join("_0.si");
            let mut si =
                lucene_index::segment_info::parse(&std::fs::read(&si_path).unwrap(), &id).unwrap();
            si.files.retain(|f| !f.contains(".fd"));
            std::fs::write(&si_path, lucene_index::segment_info::write(&si, "")).unwrap();
        });
        assert_eq!(document_blob(h, 0, 0).err(), Some(FfiStatus::Decode));
        assert!(crate::error::last_error().contains("no stored-fields files"));
        crate::jvm_reader::ffi_close_jvm_reader(h);
    }

    #[test]
    fn a_document_with_more_fields_than_an_i32_counts_is_refused() {
        let mut b = Blob {
            out: Vec::new(),
            count: i32::MAX,
            derived_source: None,
            ordinal: -1,
        };
        let e = b.int_field(1, 1).unwrap_err();
        assert!(e.to_string().contains("too large for the JVM"), "{e}");
    }

    #[test]
    fn every_value_type_encodes_as_documented() {
        let mut b = Blob {
            out: 0i32.to_le_bytes().to_vec(),
            count: 0,
            derived_source: Some(7),
            ordinal: -1,
        };
        assert_eq!(b.needs_field(3).unwrap(), VisitStatus::Yes);
        b.string_field(1, "héllo").unwrap();
        b.binary_field(2, &[0, 255]).unwrap();
        b.int_field(3, -7).unwrap();
        b.long_field(4, i64::MIN).unwrap();
        b.float_field(5, -0.5).unwrap();
        b.double_field(6, f64::MAX).unwrap();
        // A derived `_source` leaves its value out; one that is not content stays inline, and
        // one in another content type is refused, so the plugin reads the document with Lucene.
        b.binary_field(7, br#" {"v":1}"#).unwrap();
        b.binary_field(7, b"").unwrap();
        assert!(b.binary_field(7, b"---\nv: 1").is_err());
        let Blob { mut out, count, .. } = b;
        out[..4].copy_from_slice(&count.to_le_bytes());
        assert_eq!(
            decode(&out),
            vec![
                (1, FieldValue::String("héllo".into())),
                (2, FieldValue::Binary(vec![0, 255])),
                (3, FieldValue::Int(-7)),
                (4, FieldValue::Long(i64::MIN)),
                (5, FieldValue::Float(-0.5)),
                (6, FieldValue::Double(f64::MAX)),
                (7, FieldValue::Binary(b"derived".to_vec())),
                (7, FieldValue::Binary(Vec::new())),
            ]
        );
    }
}
