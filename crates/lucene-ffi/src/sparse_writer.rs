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
        let mut out = try_with_capacity(8 + sit.len() + sip.len())?;
        out.extend_from_slice(&(sit.len() as u64).to_le_bytes());
        out.extend_from_slice(&sit);
        out.extend_from_slice(&sip);
        // SAFETY: caller contract.
        unsafe { give(out, out_ptr, out_len) };
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attrs(pairs: &[(&str, &str)]) -> Vec<u8> {
        let mut out = (pairs.len() as u32).to_le_bytes().to_vec();
        for (k, v) in pairs {
            for s in [k, v] {
                out.extend_from_slice(&(s.len() as u32).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
        }
        out
    }

    fn forward(pairs: &[(i32, f32)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (t, w) in pairs {
            out.extend_from_slice(&t.to_be_bytes());
            out.extend_from_slice(&w.to_be_bytes());
        }
        out
    }

    unsafe fn open(name: &str, number: i32, a: &[u8], max_doc: i32) -> (i32, u64) {
        let mut h = 0u64;
        // SAFETY: valid buffers.
        let rc = unsafe {
            ffi_sparse_clusterer_open(
                name.as_ptr(),
                name.len(),
                number,
                a.as_ptr(),
                a.len(),
                max_doc,
                &mut h,
            )
        };
        (rc, h)
    }

    #[test]
    fn attributes_decode_and_refuse_bad_layouts() {
        let a = attrs(&[("k", "v"), ("x", "")]);
        assert_eq!(
            decode_attributes(&a).unwrap(),
            vec![("k".into(), "v".into()), ("x".into(), String::new())]
        );
        assert!(decode_attributes(&a[..a.len() - 1]).is_err());
        let mut long = a.clone();
        long.push(0);
        assert!(decode_attributes(&long).is_err());
        assert!(decode_attributes(&[1, 0, 0, 0, 1, 0, 0, 0, 0xFF, 0, 0, 0, 0]).is_err());
    }

    #[test]
    fn segment_files_match_the_rust_writer_s() {
        let field_attrs = [
            ("sparse_vector_field", "true"),
            ("approximate_threshold", "4"),
            ("cluster_ratio", "0.5"),
        ];
        let docs: Vec<Vec<(i32, f32)>> = (0..30)
            .map(|d| vec![(d % 3, 0.5 + d as f32 / 30.0), (7, 1.25), (100 + d, 0.1)])
            .collect();
        let id = [9u8; ID_LENGTH];

        // The same field through the Rust writer's clusterer.
        let info = FieldInfo::new("emb", 2).with_attributes(
            field_attrs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        );
        let mut direct = FieldClusterer::new(&info, 31).unwrap();
        for (d, v) in docs.iter().enumerate() {
            if d != 5 {
                direct.add(d as i32, &forward(v)).unwrap();
            }
        }
        let (want_sit, want_sip) = lucene_sparse::postings::write(&id, &[direct.finish(&id)]);

        let a = attrs(&field_attrs);
        // SAFETY: valid buffers.
        let (rc, h) = unsafe { open("emb", 2, &a, 31) };
        assert_eq!(rc, 0);
        for (d, v) in docs.iter().enumerate() {
            if d == 5 {
                continue;
            }
            let bytes = forward(v);
            // SAFETY: valid buffers.
            let rc = unsafe { ffi_sparse_clusterer_add(h, d as i32, bytes.as_ptr(), bytes.len()) };
            assert_eq!(rc, 0);
        }
        // Out of order is refused.
        // SAFETY: valid buffers.
        let rc = unsafe { ffi_sparse_clusterer_add(h, 3, std::ptr::null(), 0) };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());

        let mut ptr = std::ptr::null_mut();
        let mut len = 0usize;
        // SAFETY: valid buffers.
        let rc = unsafe { ffi_sparse_postings_write_alloc(&h, 1, id.as_ptr(), &mut ptr, &mut len) };
        assert_eq!(rc, 0);
        // SAFETY: the call above left `len` bytes at `ptr`.
        let out = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
        // SAFETY: the buffer came from the call above.
        unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
        let sit_len = u64::from_le_bytes(out[..8].try_into().unwrap()) as usize;
        assert_eq!(&out[8..8 + sit_len], want_sit.as_slice());
        assert_eq!(&out[8 + sit_len..], want_sip.as_slice());
        // The write took the handle.
        assert_eq!(
            ffi_sparse_clusterer_close(h),
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn no_fields_still_write_a_pair_and_bad_handles_fail() {
        let id = [1u8; ID_LENGTH];
        let mut ptr = std::ptr::null_mut();
        let mut len = 0usize;
        // SAFETY: valid buffers.
        let rc = unsafe {
            ffi_sparse_postings_write_alloc(std::ptr::null(), 0, id.as_ptr(), &mut ptr, &mut len)
        };
        assert_eq!(rc, 0);
        assert!(len > 8);
        // SAFETY: the buffer came from the call above.
        unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };

        let bogus = 12345u64;
        // SAFETY: valid buffers.
        let rc =
            unsafe { ffi_sparse_postings_write_alloc(&bogus, 1, id.as_ptr(), &mut ptr, &mut len) };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
        // SAFETY: valid buffers.
        let rc = unsafe { ffi_sparse_clusterer_add(bogus, 0, std::ptr::null(), 0) };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());

        let bad = attrs(&[("sparse_vector_field", "true"), ("n_postings", "x")]);
        // SAFETY: valid buffers.
        let (rc, _) = unsafe { open("emb", 0, &bad, 10) };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        let ok = attrs(&[("sparse_vector_field", "true")]);
        // SAFETY: valid buffers.
        let (rc, h) = unsafe { open("emb", 0, &ok, 10) };
        assert_eq!(rc, 0);
        assert_eq!(ffi_sparse_clusterer_close(h), 0);
    }
}
