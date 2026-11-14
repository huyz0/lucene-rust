//! A hit's stored fields for the plugin's fetch phase and get API: the entry
//! points over [`lucene_engine::fetch`], which holds the blob layouts and the
//! decoding.

use crate::error::{set_last_error, FfiStatus};
use crate::jvm_reader;

pub(crate) use lucene_engine::fetch::{BINARY, DOUBLE, FLOAT, INT, LONG, STRING};

fn handle(
    handle: u64,
    what: &str,
) -> Result<std::sync::Arc<crate::registry::JvmReaderHandle>, FfiStatus> {
    jvm_reader::lookup(handle, &format!("{what}: unknown or already-closed handle"))
}

/// Segment `segment`'s document `doc` (segment-local) of `handle`'s reader,
/// encoded ([`lucene_engine::fetch::document_blob`]).
pub(crate) fn document_blob(h: u64, segment: i32, doc: i32) -> Result<Vec<u8>, FfiStatus> {
    lucene_engine::fetch::document_blob(&*handle(h, "ffi_jvm_reader_document")?, segment, doc)
}

/// [`lucene_engine::fetch::doc_values_blob`] over `handle`'s reader.
pub(crate) fn doc_values_blob(
    h: u64,
    segment: i32,
    doc: i32,
    fields: &[i32],
) -> Result<Vec<u8>, FfiStatus> {
    lucene_engine::fetch::doc_values_blob(
        &*handle(h, "ffi_jvm_reader_doc_values")?,
        segment,
        doc,
        fields,
    )
}

/// [`lucene_engine::fetch::source_blob`] over `handle`'s reader.
pub(crate) fn source_blob(
    h: u64,
    segment: i32,
    doc: i32,
    vectors: bool,
) -> Result<Vec<u8>, FfiStatus> {
    lucene_engine::fetch::source_blob(
        &*handle(h, "ffi_jvm_reader_derived_source")?,
        segment,
        doc,
        vectors,
    )
}

/// [`lucene_engine::fetch::stored_value`] over `handle`'s reader.
pub(crate) fn stored_value(
    h: u64,
    segment: i32,
    doc: i32,
    ordinal: i32,
) -> Result<Vec<u8>, FfiStatus> {
    lucene_engine::fetch::stored_value(
        &*handle(h, "ffi_jvm_reader_stored_value")?,
        segment,
        doc,
        ordinal,
    )
}

/// [`lucene_engine::fetch::prefetch_documents`] over `handle`'s reader.
pub(crate) fn prefetch_documents(h: u64, segment: i32, docs: &[i32]) -> Result<(), FfiStatus> {
    lucene_engine::fetch::prefetch_documents(
        &*handle(h, "ffi_jvm_reader_prefetch_documents")?,
        segment,
        docs,
    )
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
    use lucene_codecs::stored_fields::{StoredFieldVisitor, VisitStatus};
    use lucene_engine::fetch::{Blob, DEFERRED, DEFER_BYTES, DERIVED_SOURCE};
    use lucene_engine::fetch::{DV_NONE, DV_SORTED_NUMERIC, DV_SORTED_SET};

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
