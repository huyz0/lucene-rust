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

use lucene_codecs::stored_fields::{self, StoredFieldVisitor, VisitStatus};

use crate::error::{set_last_error, FfiStatus};
use crate::jvm_reader;

pub(crate) const STRING: u8 = 0;
pub(crate) const BINARY: u8 = 1;
pub(crate) const INT: u8 = 2;
pub(crate) const LONG: u8 = 3;
pub(crate) const FLOAT: u8 = 4;
pub(crate) const DOUBLE: u8 = 5;

/// Segment `segment`'s document `doc` (segment-local) of `handle`'s reader,
/// encoded (see the module doc).
pub(crate) fn document_blob(handle: u64, segment: i32, doc: i32) -> Result<Vec<u8>, FfiStatus> {
    let h = jvm_reader::lookup(
        handle,
        "ffi_jvm_reader_document: unknown or already-closed handle",
    )?;
    let readers = h.reader.segment_readers();
    let Some(reader) = usize::try_from(segment).ok().and_then(|s| readers.get(s)) else {
        set_last_error(format!(
            "ffi_jvm_reader_document: segment {segment} of {}",
            readers.len()
        ));
        return Err(FfiStatus::InvalidArgument);
    };
    if doc < 0 || doc >= reader.max_doc {
        set_last_error(format!(
            "ffi_jvm_reader_document: document {doc} of {}",
            reader.max_doc
        ));
        return Err(FfiStatus::InvalidArgument);
    }
    let mut blob = Blob {
        out: 0i32.to_le_bytes().to_vec(),
        count: 0,
    };
    reader.visit_stored_document(doc, &mut blob).map_err(|e| {
        set_last_error(format!("reading stored fields: {e}"));
        FfiStatus::Decode
    })?;
    let Blob { mut out, count } = blob;
    out[..4].copy_from_slice(&count.to_le_bytes());
    Ok(out)
}

/// Encodes every field it is shown (see the module doc), straight from the
/// decompressed bytes: no [`lucene_codecs::stored_fields::Document`] in
/// between. `out` starts with room for the count, written last.
struct Blob {
    out: Vec<u8>,
    count: i32,
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
        Ok(VisitStatus::Yes)
    }

    fn string_field(&mut self, number: i32, value: &str) -> stored_fields::Result<()> {
        self.bytes(number, STRING, value.as_bytes())
    }

    fn binary_field(&mut self, number: i32, value: &[u8]) -> stored_fields::Result<()> {
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
        // The fixture's documents store their fields; every segment answers.
        let mut seen = 0;
        for segment in 0..2 {
            for doc in 0..4 {
                let blob = document_blob(h, segment, doc).unwrap();
                let mut at = 0;
                let n = i32_at(&blob, &mut at);
                for _ in 0..n {
                    let _number = i32_at(&blob, &mut at);
                    let ty = blob[at];
                    at += 1;
                    at += match ty {
                        STRING | BINARY => i32_at(&blob, &mut at) as usize,
                        INT | FLOAT => 4,
                        LONG | DOUBLE => 8,
                        other => panic!("type {other}"),
                    };
                    seen += 1;
                }
                assert_eq!(at, blob.len());
            }
        }
        // The fixture's documents store no fields: each answers an empty list
        // (`stored_document`'s own decoding is tested on a stored-fields
        // fixture in lucene-search).
        assert_eq!(seen, 0);
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
                LONG | DOUBLE => {
                    let v = u64::from_le_bytes(blob[at..at + 8].try_into().unwrap());
                    let value = if ty == LONG {
                        FieldValue::Long(v as i64)
                    } else {
                        FieldValue::Double(f64::from_bits(v))
                    };
                    (value, 8)
                }
                other => panic!("type {other}"),
            };
            at += len;
            fields.push((number, value));
        }
        assert_eq!(at, blob.len());
        fields
    }

    #[test]
    fn stored_documents_come_back_as_the_segment_reader_reads_them() {
        // A Lucene-written index whose documents store strings, binaries and
        // every numeric type (`fixtures/src/GenStoredFields.java`).
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/stored_fields_index"
        );
        let reader = lucene_search::directory_reader::DirectoryReader::open(
            &lucene_store::directory::FsDirectory::open(dir),
        )
        .unwrap();
        let seg = &reader.segment_readers()[0];
        let infos = std::fs::read(format!("{dir}/segments_1")).unwrap();
        let max_docs = [seg.max_doc];
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

    #[test]
    fn every_value_type_encodes_as_documented() {
        let mut b = Blob {
            out: 0i32.to_le_bytes().to_vec(),
            count: 0,
        };
        assert_eq!(b.needs_field(3).unwrap(), VisitStatus::Yes);
        b.string_field(1, "héllo").unwrap();
        b.binary_field(2, &[0, 255]).unwrap();
        b.int_field(3, -7).unwrap();
        b.long_field(4, i64::MIN).unwrap();
        b.float_field(5, -0.5).unwrap();
        b.double_field(6, f64::MAX).unwrap();
        let Blob { mut out, count } = b;
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
            ]
        );
    }
}
