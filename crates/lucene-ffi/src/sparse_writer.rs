//! Clustered postings for the segments OpenSearch's own engine writes in an
//! `index.sparse` index: the plugin's Java codec hands each `sparse_vector`
//! field's forward index to a [`FieldClusterer`] at flush and merge, then has
//! the segment's `.sit`/`.sip` encoded here, as the Rust writer writes them.

use std::sync::{Mutex, OnceLock, PoisonError, RwLock};

use lucene_codecs::field_infos::FieldInfo;
use lucene_sparse::postings::FieldClusters;
use lucene_sparse::writer::FieldClusterer;
use lucene_store::codec_util::ID_LENGTH;

use crate::error::{guard, set_last_error, FfiStatus};
use crate::ffm_bridge::{clear_out, give};
use crate::raw::{bytes_from_raw, str_from_raw, try_to_vec, try_with_capacity};
use crate::registry::{lock_recovering, read_recovering};
use lucene_engine::handle::{RegistryTag, SlotMap};

/// The open clusterers, by handle.
fn clusterers() -> &'static RwLock<SlotMap<Mutex<FieldClusterer>>> {
    static REGISTRY: OnceLock<RwLock<SlotMap<Mutex<FieldClusterer>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(SlotMap::new(RegistryTag::SparseClusterer)))
}

fn invalid(e: impl std::fmt::Display) -> FfiStatus {
    set_last_error(e.to_string());
    FfiStatus::InvalidArgument
}

/// Decodes field attributes: a little-endian `u32` count, then per entry
/// a `u32` key length, the UTF-8 key, a `u32` value length and the value.
fn decode_attributes(mut bytes: &[u8]) -> Result<Vec<(String, String)>, FfiStatus> {
    fn take<'a>(bytes: &mut &'a [u8], n: usize) -> Result<&'a [u8], FfiStatus> {
        if bytes.len() < n {
            return Err(invalid("truncated field attributes"));
        }
        let (head, tail) = bytes.split_at(n);
        *bytes = tail;
        Ok(head)
    }
    fn len(bytes: &mut &[u8]) -> Result<usize, FfiStatus> {
        let b = take(bytes, 4)?;
        usize::try_from(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .map_err(|_| invalid("attribute length overflow"))
    }
    fn string(bytes: &mut &[u8]) -> Result<String, FfiStatus> {
        let n = len(bytes)?;
        String::from_utf8(try_to_vec(take(bytes, n)?)?).map_err(|_| FfiStatus::InvalidUtf8)
    }
    let count = len(&mut bytes)?;
    let mut out = try_with_capacity(count.min(64))?;
    for _ in 0..count {
        let k = string(&mut bytes)?;
        let v = string(&mut bytes)?;
        out.push((k, v));
    }
    if !bytes.is_empty() {
        return Err(invalid("trailing bytes after field attributes"));
    }
    Ok(out)
}

/// Opens a clusterer for field `number` named `name` (`name_len` UTF-8
/// bytes) with the attributes at `attrs` ([`decode_attributes`]' layout), in
/// a segment of `max_doc` documents; its handle goes to `*out_handle`.
/// Release it with [`ffi_sparse_clusterer_close`] unless
/// [`ffi_sparse_postings_write_alloc`] took it.
///
/// # Safety
/// Each pointer must be valid for its length; `out_handle` for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_sparse_clusterer_open(
    name: *const u8,
    name_len: usize,
    number: i32,
    attrs: *const u8,
    attrs_len: usize,
    max_doc: i32,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let name = unsafe { str_from_raw(name.cast(), name_len) }?;
        // SAFETY: caller contract.
        let attrs = decode_attributes(unsafe { bytes_from_raw(attrs, attrs_len) }?)?;
        let info = FieldInfo::new(name, number).with_attributes(attrs);
        let clusterer = FieldClusterer::new(&info, max_doc).map_err(invalid)?;
        let handle = lock_recovering(clusterers()).insert_checked(Mutex::new(clusterer))?;
        // SAFETY: caller contract.
        unsafe { *out_handle = handle };
        Ok(())
    })
}

/// Adds `doc`'s forward-index value (`len` bytes at `bytes`: big-endian
/// `int` token and `float` weight pairs) to clusterer `handle`. Documents
/// come in ascending order.
///
/// # Safety
/// `bytes` must be valid for `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ffi_sparse_clusterer_add(
    handle: u64,
    doc: i32,
    bytes: *const u8,
    len: usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let bytes = unsafe { bytes_from_raw(bytes, len) }?;
        let registry = read_recovering(clusterers());
        let clusterer = registry.get(handle).ok_or(FfiStatus::InvalidHandle)?;
        let added = clusterer
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .add(doc, bytes)
            .map_err(invalid);
        added
    })
}

/// Releases clusterer `handle` without writing it.
#[no_mangle]
pub extern "C" fn ffi_sparse_clusterer_close(handle: u64) -> i32 {
    guard(|| {
        lock_recovering(clusterers())
            .remove(handle)
            .map(drop)
            .ok_or(FfiStatus::InvalidHandle)
    })
}

/// Clusters the `count` fields at `handles` (in field-name order; each
/// handle is released, whatever the outcome) and encodes the segment's
/// `.sit` and `.sip` for segment id `segment_id` (16 bytes). The owned
/// buffer left at `*out_ptr`/`*out_len` holds the `.sit` length as a
/// little-endian `u64`, then `.sit`, then `.sip`; free it with
/// `ffi_jvm_free_bytes`.
///
/// # Safety
/// `handles` must be valid for `count` reads, `segment_id` for
/// [`ID_LENGTH`] bytes, `out_ptr` and `out_len` for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_sparse_postings_write_alloc(
    handles: *const u64,
    count: usize,
    segment_id: *const u8,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len) }?;
        let handles: &[u64] = if count == 0 {
            &[]
        } else if handles.is_null() {
            return Err(FfiStatus::NullPointer);
        } else {
            // SAFETY: caller contract; non-null checked above.
            unsafe { std::slice::from_raw_parts(handles, count) }
        };
        let taken: Vec<Option<FieldClusterer>> = {
            let mut registry = lock_recovering(clusterers());
            handles
                .iter()
                .map(|&h| {
                    registry
                        .remove(h)
                        .map(|m| m.into_inner().unwrap_or_else(PoisonError::into_inner))
                })
                .collect()
        };
        // SAFETY: caller contract.
        let id = unsafe { bytes_from_raw(segment_id, ID_LENGTH) }?;
        let id: &[u8; ID_LENGTH] = id.try_into().map_err(|_| invalid("bad segment id"))?;
        let mut fields: Vec<FieldClusters> = try_with_capacity(taken.len())?;
        for clusterer in taken {
            fields.push(clusterer.ok_or(FfiStatus::InvalidHandle)?.finish(id));
        }
        let (sit, sip) = lucene_sparse::postings::write(id, &fields);
        let mut out = try_with_capacity(sit.len().saturating_add(sip.len()).saturating_add(8))?;
        out.extend_from_slice(&(sit.len() as u64).to_le_bytes());
        out.extend_from_slice(&sit);
        out.extend_from_slice(&sip);
        // SAFETY: caller contract.
        unsafe { give(out, out_ptr, out_len) };
        Ok(())
    })
}

#[cfg(test)]
mod tests;
