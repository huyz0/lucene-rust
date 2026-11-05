//! SPIKE (real-time study, not for merge): a searchable in-memory copy of the
//! documents indexed since the last refresh, so a search can see them without
//! one.
//!
//! Every document the engine writer applies is also added here. The writer
//! keeps its own buffers and flushes them at a refresh as before; the Java
//! side then starts a fresh buffer ([`ffi_live_reset`]) under its exclusive
//! indexing lock, so a buffer holds exactly the documents after one commit.
//!
//! A search takes a snapshot: the buffer and its visible document count at
//! that moment ([`ffi_live_acquire`]). Documents are only appended, so every
//! read below the snapshot's count is stable. This spike guards the buffer
//! with one `RwLock`; the design in `research/realtime-index.md` makes each
//! indexing thread's buffer single-writer and lock-free instead.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, PoisonError, RwLock};

use lucene_codecs::stored_fields::FieldValue;
use lucene_index::index_writer::ExplicitDocument;

use crate::error::{guard, set_last_error, FfiStatus};
use crate::jvm_fetch::{BINARY, DOUBLE, FLOAT, INT, LONG, STRING};
use crate::raw::bytes_from_raw;

#[derive(Default)]
struct Inner {
    /// Each document's stored fields, in `jvm_fetch`'s document-blob format.
    stored: Vec<Vec<u8>>,
    /// Field number -> term -> `doc << 32 | freq`, docs ascending.
    postings: HashMap<i32, HashMap<Vec<u8>, Vec<u64>>>,
    /// Field number -> `(doc, norm)`.
    norms: HashMap<i32, Vec<(u32, i64)>>,
    /// Field number -> `(doc, value)` for numeric doc values.
    numeric: HashMap<i32, Vec<(u32, i64)>>,
    /// Field number -> `[sum_total_term_freq, sum_doc_freq, doc_count]`.
    stats: HashMap<i32, [i64; 3]>,
    /// Deleted docs (replaced by a later version, or soft-deleted on arrival).
    deleted: Vec<u32>,
    /// `_id` term -> its newest doc in this buffer.
    by_id: HashMap<Vec<u8>, u32>,
}

/// One buffer: the documents since the last refresh.
#[derive(Default)]
pub(crate) struct LiveBuffer {
    inner: RwLock<Inner>,
    visible: AtomicU32,
}

/// The shard's live state, held by its engine writer.
#[derive(Default)]
pub(crate) struct Live {
    current: RwLock<Arc<LiveBuffer>>,
    id_field: AtomicI32,
    soft_field: AtomicI32,
    enabled: std::sync::atomic::AtomicBool,
}

fn blob(doc: &ExplicitDocument) -> Vec<u8> {
    let mut out = (doc.stored.len() as i32).to_le_bytes().to_vec();
    for f in &doc.stored {
        out.extend_from_slice(&f.field_number.to_le_bytes());
        match &f.value {
            FieldValue::String(s) => {
                out.push(STRING);
                out.extend_from_slice(&(s.len() as i32).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
            FieldValue::Binary(b) => {
                out.push(BINARY);
                out.extend_from_slice(&(b.len() as i32).to_le_bytes());
                out.extend_from_slice(b);
            }
            FieldValue::Int(v) => {
                out.push(INT);
                out.extend_from_slice(&v.to_le_bytes());
            }
            FieldValue::Long(v) => {
                out.push(LONG);
                out.extend_from_slice(&v.to_le_bytes());
            }
            FieldValue::Float(v) => {
                out.push(FLOAT);
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            FieldValue::Double(v) => {
                out.push(DOUBLE);
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
    }
    out
}

impl Live {
    /// Adds `docs` (after the writer took them); `replaces` is a soft update's
    /// `_id`, whose earlier version in this buffer is deleted.
    pub(crate) fn add(&self, docs: &[ExplicitDocument], replaces: Option<&[u8]>) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        let id_field = self.id_field.load(Ordering::Relaxed);
        let soft_field = self.soft_field.load(Ordering::Relaxed);
        // Encode outside the lock.
        let blobs: Vec<Vec<u8>> = docs.iter().map(blob).collect();
        let buffer = self
            .current
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let mut inner = buffer.inner.write().unwrap_or_else(PoisonError::into_inner);
        if let Some(id) = replaces {
            if let Some(&old) = inner.by_id.get(id) {
                inner.deleted.push(old);
            }
        }
        for (d, b) in docs.iter().zip(blobs) {
            let doc = inner.stored.len() as u32;
            inner.stored.push(b);
            let mut soft_deleted = false;
            for f in &d.fields.inverted {
                let mut ttf = 0i64;
                let mut df = 0i64;
                {
                    let field = inner.postings.entry(f.field_number).or_default();
                    for t in f.terms.iter() {
                        let freq = t.freq.max(1) as u32;
                        match field.get_mut(t.term) {
                            Some(list) => list.push(u64::from(doc) << 32 | u64::from(freq)),
                            None => {
                                field.insert(t.term.to_vec(), vec![u64::from(doc) << 32 | u64::from(freq)]);
                            }
                        }
                        ttf += i64::from(freq);
                        df += 1;
                    }
                }
                if f.field_number == id_field {
                    if let Some(t) = f.terms.iter().next() {
                        let id = t.term.to_vec();
                        if let Some(old) = inner.by_id.insert(id, doc) {
                            if replaces.is_none() {
                                inner.deleted.push(old);
                            }
                        }
                    }
                }
                let s = inner.stats.entry(f.field_number).or_default();
                s[0] += ttf;
                s[1] += df;
                s[2] += 1;
                if let Some(n) = f.norm {
                    inner.norms.entry(f.field_number).or_default().push((doc, n));
                }
            }
            for dv in &d.fields.doc_values {
                if let FieldValue::Long(v) = dv.value {
                    if dv.field_number == soft_field {
                        soft_deleted = true;
                    }
                    inner.numeric.entry(dv.field_number).or_default().push((doc, v));
                }
            }
            if soft_deleted {
                inner.deleted.push(doc);
            }
        }
        let n = inner.stored.len() as u32;
        buffer.visible.store(n, Ordering::Release);
    }
}

static SNAPSHOTS: LazyLock<Mutex<HashMap<u64, Arc<LiveBuffer>>>> = LazyLock::new(Default::default);
static NEXT: AtomicU64 = AtomicU64::new(1);

fn snapshot(id: u64) -> Result<Arc<LiveBuffer>, FfiStatus> {
    SNAPSHOTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&id)
        .cloned()
        .ok_or_else(|| {
            set_last_error(format!("live: unknown snapshot {id}"));
            FfiStatus::InvalidHandle
        })
}

/// # Safety
/// `out_ptr`/`out_len` valid for one write each.
unsafe fn give(bytes: Vec<u8>, out_ptr: *mut *mut u8, out_len: *mut usize) -> Result<(), FfiStatus> {
    if out_ptr.is_null() || out_len.is_null() {
        return Err(FfiStatus::NullPointer);
    }
    // SAFETY: caller contract.
    unsafe { crate::ffm_bridge::give(bytes, out_ptr, out_len) };
    Ok(())
}

/// Turns the live buffer on, with the `_id` and soft-deletes field numbers.
#[no_mangle]
pub extern "C" fn ffi_live_configure(handle: u64, id_field: i32, soft_field: i32) -> i32 {
    guard(|| {
        let w = crate::engine_writer::live_of(handle)?;
        let live = &w.live;
        live.id_field.store(id_field, Ordering::Relaxed);
        live.soft_field.store(soft_field, Ordering::Relaxed);
        live.enabled.store(true, Ordering::Relaxed);
        Ok(())
    })
}

/// Starts a new, empty buffer (the caller holds its exclusive indexing lock,
/// right after a commit).
#[no_mangle]
pub extern "C" fn ffi_live_reset(handle: u64) -> i32 {
    guard(|| {
        let w = crate::engine_writer::live_of(handle)?;
        let live = &w.live;
        *live.current.write().unwrap_or_else(PoisonError::into_inner) = Arc::new(LiveBuffer::default());
        Ok(())
    })
}

/// Pins the current buffer: `*out_id` names it, `*out_visible` is its visible
/// document count now. Release with [`ffi_live_release`].
///
/// # Safety
/// Out pointers valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_acquire(handle: u64, out_id: *mut u64, out_visible: *mut i32) -> i32 {
    guard(|| {
        if out_id.is_null() || out_visible.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let w = crate::engine_writer::live_of(handle)?;
        let live = &w.live;
        let buffer = live.current.read().unwrap_or_else(PoisonError::into_inner).clone();
        let visible = buffer.visible.load(Ordering::Acquire);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        SNAPSHOTS.lock().unwrap_or_else(PoisonError::into_inner).insert(id, buffer);
        // SAFETY: caller contract.
        unsafe {
            *out_id = id;
            *out_visible = visible as i32;
        }
        Ok(())
    })
}

/// The visible count of a pinned buffer now (it keeps growing until the next reset).
#[no_mangle]
pub extern "C" fn ffi_live_visible(id: u64) -> i32 {
    match snapshot(id) {
        Ok(b) => b.visible.load(Ordering::Acquire) as i32,
        Err(_) => -1,
    }
}

/// Pins the buffer snapshot `id` pins, under a new id.
///
/// # Safety
/// `out_id` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_pin(id: u64, out_id: *mut u64) -> i32 {
    guard(|| {
        if out_id.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let b = snapshot(id)?;
        let new = NEXT.fetch_add(1, Ordering::Relaxed);
        SNAPSHOTS.lock().unwrap_or_else(PoisonError::into_inner).insert(new, b);
        // SAFETY: caller contract.
        unsafe { *out_id = new };
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn ffi_live_release(id: u64) -> i32 {
    guard(|| {
        SNAPSHOTS.lock().unwrap_or_else(PoisonError::into_inner).remove(&id);
        Ok(())
    })
}

/// A term's postings below `max_doc`: little-endian `(doc: u32, freq: u32)`
/// pairs; empty when the term is not there.
///
/// # Safety
/// `term` valid for `term_len` bytes; out pointers valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_postings(
    id: u64,
    field: i32,
    term: *const u8,
    term_len: usize,
    max_doc: i32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let term = unsafe { bytes_from_raw(term, term_len)? };
        let b = snapshot(id)?;
        let inner = b.inner.read().unwrap_or_else(PoisonError::into_inner);
        let mut out = Vec::new();
        if let Some(list) = inner.postings.get(&field).and_then(|f| f.get(term)) {
            out.reserve(list.len() * 8);
            for &p in list {
                let doc = (p >> 32) as u32;
                if doc as i32 >= max_doc {
                    break;
                }
                out.extend_from_slice(&doc.to_le_bytes());
                out.extend_from_slice(&(p as u32).to_le_bytes());
            }
        }
        drop(inner);
        // SAFETY: caller contract.
        unsafe { give(out, out_ptr, out_len) }
    })
}

/// `[sum_total_term_freq, sum_doc_freq, doc_count]` of a field (may run a
/// little ahead of a snapshot).
///
/// # Safety
/// `out` valid for 3 writes.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_field_stats(id: u64, field: i32, out: *mut i64) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let b = snapshot(id)?;
        let s = b
            .inner
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .stats
            .get(&field)
            .copied()
            .unwrap_or([0; 3]);
        // SAFETY: caller contract.
        unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), out, 3) };
        Ok(())
    })
}

/// A column below `max_doc` as little-endian `(doc: u32, value: i64)`
/// entries: `kind` 0 is norms, 1 numeric doc values. Empty when absent.
///
/// # Safety
/// Out pointers valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_column(
    id: u64,
    kind: i32,
    field: i32,
    max_doc: i32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        let b = snapshot(id)?;
        let inner = b.inner.read().unwrap_or_else(PoisonError::into_inner);
        let col = if kind == 0 { inner.norms.get(&field) } else { inner.numeric.get(&field) };
        let mut out = Vec::new();
        if let Some(col) = col {
            for &(doc, v) in col {
                if doc as i32 >= max_doc {
                    break;
                }
                out.extend_from_slice(&doc.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        drop(inner);
        // SAFETY: caller contract.
        unsafe { give(out, out_ptr, out_len) }
    })
}

/// Deleted docs below `max_doc`, as little-endian `u32`s.
///
/// # Safety
/// Out pointers valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_deleted(
    id: u64,
    max_doc: i32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        let b = snapshot(id)?;
        let inner = b.inner.read().unwrap_or_else(PoisonError::into_inner);
        let mut out = Vec::new();
        for &d in &inner.deleted {
            if (d as i32) < max_doc {
                out.extend_from_slice(&d.to_le_bytes());
            }
        }
        drop(inner);
        // SAFETY: caller contract.
        unsafe { give(out, out_ptr, out_len) }
    })
}

/// One document's stored fields, as `ffi_jvm_reader_document_alloc` returns them.
///
/// # Safety
/// Out pointers valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_document(id: u64, doc: i32, out_ptr: *mut *mut u8, out_len: *mut usize) -> i32 {
    guard(|| {
        let b = snapshot(id)?;
        let inner = b.inner.read().unwrap_or_else(PoisonError::into_inner);
        let Some(blob) = usize::try_from(doc).ok().and_then(|d| inner.stored.get(d)) else {
            set_last_error(format!("live: no document {doc}"));
            return Err(FfiStatus::InvalidArgument);
        };
        let blob = blob.clone();
        drop(inner);
        // SAFETY: caller contract.
        unsafe { give(blob, out_ptr, out_len) }
    })
}
