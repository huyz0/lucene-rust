//! A hit's stored fields for the plugin's fetch phase and get API: the entry
//! points over [`lucene_engine::fetch`], which holds the blob layouts and the
//! decoding.

use crate::error::{set_last_error, FfiStatus};
use crate::jvm_reader;

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
mod tests;
