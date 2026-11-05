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
//! read below the snapshot's count is stable. Each indexing thread
//! writes one stripe (one leaf), so threads on different stripes never
//! contend; the design makes each stripe single-writer and lock-free.
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
}

/// One stripe: the documents one group of indexing threads added since the
/// last refresh. Each stripe is its own leaf, so indexing threads on
/// different stripes never contend (the design's per-DWPT buffers).
#[derive(Default)]
pub(crate) struct Stripe {
    inner: RwLock<Inner>,
    visible: AtomicU32,
}

/// One buffer generation: the stripes plus the `_id` map and deletes, which
/// span stripes.
pub(crate) struct LiveBuffer {
    stripes: Vec<Stripe>,
    ids: Mutex<HashMap<Vec<u8>, (u32, u32)>>,
    deleted: Mutex<Vec<(u32, u32)>>,
}

impl Default for LiveBuffer {
    fn default() -> Self {
        let n = std::thread::available_parallelism().map_or(1, |n| n.get());
        LiveBuffer {
            stripes: (0..n).map(|_| Stripe::default()).collect(),
            ids: Mutex::default(),
            deleted: Mutex::default(),
        }
    }
}

static NEXT_STRIPE: AtomicU32 = AtomicU32::new(0);
thread_local! {
    static MY_STRIPE: u32 = NEXT_STRIPE.fetch_add(1, Ordering::Relaxed);
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
        let si = MY_STRIPE.with(|s| *s) % buffer.stripes.len() as u32;
        let stripe = &buffer.stripes[si as usize];
        let mut added: Vec<(u32, Option<Vec<u8>>, bool)> = Vec::with_capacity(docs.len());
        {
            let mut inner = stripe.inner.write().unwrap_or_else(PoisonError::into_inner);
            for (d, b) in docs.iter().zip(blobs) {
                let doc = inner.stored.len() as u32;
                inner.stored.push(b);
                let mut soft_deleted = false;
                let mut id = None;
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
                        id = f.terms.iter().next().map(|t| t.term.to_vec());
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
                added.push((doc, id, soft_deleted));
            }
        }
        {
            let mut ids = buffer.ids.lock().unwrap_or_else(PoisonError::into_inner);
            let mut deleted = buffer.deleted.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(id) = replaces {
                if let Some(old) = ids.remove(id) {
                    deleted.push(old);
                }
            }
            for (doc, id, soft_deleted) in added {
                if let Some(id) = id {
                    if let Some(old) = ids.insert(id, (si, doc)) {
                        deleted.push(old);
                    }
                }
                if soft_deleted {
                    deleted.push((si, doc));
                }
            }
        }
        let n = stripe.inner.read().unwrap_or_else(PoisonError::into_inner).stored.len() as u32;
        stripe.visible.fetch_max(n, Ordering::Release);
    }
}

/// A pinned buffer, and the stripe it names (`WHOLE` for the buffer itself).
type Snap = (Arc<LiveBuffer>, u32);
const WHOLE: u32 = u32::MAX;

static SNAPSHOTS: LazyLock<Mutex<HashMap<u64, Snap>>> = LazyLock::new(Default::default);
static NEXT: AtomicU64 = AtomicU64::new(1);

fn snapshot(id: u64) -> Result<Snap, FfiStatus> {
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
        let visible: u32 = buffer.stripes.iter().map(|s| s.visible.load(Ordering::Acquire)).sum();
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        SNAPSHOTS.lock().unwrap_or_else(PoisonError::into_inner).insert(id, (buffer, WHOLE));
        // SAFETY: caller contract.
        unsafe {
            *out_id = id;
            *out_visible = visible as i32;
        }
        Ok(())
    })
}

/// The visible count of a pinned stripe (or the sum over a whole buffer's
/// stripes) now; it keeps growing until the next reset.
#[no_mangle]
pub extern "C" fn ffi_live_visible(id: u64) -> i32 {
    match snapshot(id) {
        Ok((b, WHOLE)) => b.stripes.iter().map(|s| s.visible.load(Ordering::Acquire)).sum::<u32>() as i32,
        Ok((b, s)) => b.stripes[s as usize].visible.load(Ordering::Acquire) as i32,
        Err(_) => -1,
    }
}

/// The number of stripes in a pinned buffer.
#[no_mangle]
pub extern "C" fn ffi_live_stripes(id: u64) -> i32 {
    match snapshot(id) {
        Ok((b, _)) => b.stripes.len() as i32,
        Err(_) => -1,
    }
}

/// Pins stripe `stripe` of the buffer `id` pins, under a new id; the postings,
/// column, deleted and document calls read one stripe.
///
/// # Safety
/// `out_id` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_stripe(id: u64, stripe: i32, out_id: *mut u64) -> i32 {
    guard(|| {
        if out_id.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let (b, _) = snapshot(id)?;
        let Some(si) = u32::try_from(stripe).ok().filter(|&s| (s as usize) < b.stripes.len()) else {
            set_last_error(format!("live: no stripe {stripe}"));
            return Err(FfiStatus::InvalidArgument);
        };
        let new = NEXT.fetch_add(1, Ordering::Relaxed);
        SNAPSHOTS.lock().unwrap_or_else(PoisonError::into_inner).insert(new, (b, si));
        // SAFETY: caller contract.
        unsafe { *out_id = new };
        Ok(())
    })
}

fn stripe_of(id: u64) -> Result<(Arc<LiveBuffer>, usize), FfiStatus> {
    match snapshot(id)? {
        (_, WHOLE) => {
            set_last_error("live: snapshot is a whole buffer, not a stripe".to_string());
            Err(FfiStatus::InvalidArgument)
        }
        (b, s) => Ok((b, s as usize)),
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
        let (b, si) = stripe_of(id)?;
        let inner = b.stripes[si].inner.read().unwrap_or_else(PoisonError::into_inner);
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
        let (b, si) = stripe_of(id)?;
        let s = b.stripes[si]
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
        let (b, si) = stripe_of(id)?;
        let inner = b.stripes[si].inner.read().unwrap_or_else(PoisonError::into_inner);
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
        let (b, si) = stripe_of(id)?;
        let deleted = b.deleted.lock().unwrap_or_else(PoisonError::into_inner);
        let mut out = Vec::new();
        for &(s, d) in deleted.iter() {
            if s as usize == si && (d as i32) < max_doc {
                out.extend_from_slice(&d.to_le_bytes());
            }
        }
        drop(deleted);
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
        let (b, si) = stripe_of(id)?;
        let inner = b.stripes[si].inner.read().unwrap_or_else(PoisonError::into_inner);
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
