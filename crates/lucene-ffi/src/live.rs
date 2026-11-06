//! Live search for the plugin: documents still in the engine writer's
//! buffers, searched and fetched without a refresh
//! ([`lucene_index::concurrent_writer`]'s live views).
//!
//! A searcher opens a view ([`ffi_engine_writer_live_view_open`]) beside its
//! reader: the buffers' documents and deletes as of then. Its queries run
//! over the reader's segments and the view's buffers together
//! ([`ffi_jvm_reader_search_live`]), a buffered document numbered after the
//! reader's `maxDoc`, and its fetches read the buffered documents' stored
//! fields ([`ffi_live_view_document_alloc`]). Closing the view
//! ([`ffi_live_view_close`]) lets its buffers go.
//!
//! # The view blob
//!
//! Little-endian: `published: i64`, `count: i32`, then per buffer `max_doc:
//! i32`, `deleted: i32` and that many deleted document numbers (`i32`,
//! ascending).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use lucene_codecs::stored_fields::FieldValue;
use lucene_index::buffered_updates::Term;
use lucene_index::concurrent_writer::LiveView;
use lucene_search::collector::VecCollector;
use lucene_search::live_segment::{live_request, live_segments};
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::{Clause, TermQuery};
use lucene_search::similarities::Similarity;
use lucene_search::{
    search_boolean_query_multi_segment_maxscore_counting, search_term_query, TotalHitsRelation,
};
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::clauses::map_search_error;
use crate::engine_writer::{self, EngineWriter};
use crate::error::{guard, set_last_error, FfiStatus};
use crate::jvm_fetch::{BINARY, DOUBLE, FLOAT, INT, LONG, STRING};
use crate::jvm_reader::{boolean_of, decode_request, lookup};
use crate::raw::bytes_from_raw;
use crate::registry::JvmReaderHandle;

/// An open view and the writer it reads.
struct ViewHandle {
    writer: Arc<EngineWriter>,
    view: LiveView,
    /// The reader handle [`ffi_live_view_reader`] made over a reader and
    /// this view's buffers, closed with the view.
    whole: Mutex<Whole>,
}

/// What [`ffi_live_view_reader`] made of a view.
enum Whole {
    /// Nothing yet.
    None,
    /// The reader's handle, then the new one's.
    Open(u64, u64),
    /// Why its buffers cannot be written: they will not come back.
    Failed(String),
    /// The view is closed.
    Closed,
}

/// Per segment, the documents live once the deletes since the commit are
/// applied; `None` when every document is.
type Masks = Vec<Option<Arc<FixedBitSet>>>;

/// A reader's [`Masks`] after the first `applied` of the delete terms its
/// views list ([`LiveView::commit_deletes`]): every view of one commit
/// lists the same terms first, so the next search applies only the new
/// ones. `last` is the last of them, to tell a list that does not extend
/// these.
pub(crate) struct CommitLive {
    applied: usize,
    last: Term,
    masks: Arc<Masks>,
}

fn views() -> &'static Mutex<HashMap<u64, Arc<ViewHandle>>> {
    static VIEWS: OnceLock<Mutex<HashMap<u64, Arc<ViewHandle>>>> = OnceLock::new();
    VIEWS.get_or_init(Default::default)
}

static NEXT_VIEW: AtomicU64 = AtomicU64::new(1);

fn view(id: u64) -> Result<Arc<ViewHandle>, FfiStatus> {
    views()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&id)
        .cloned()
        .ok_or_else(|| {
            set_last_error("live view: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })
}

fn live_error(e: lucene_index::index_writer::Error) -> FfiStatus {
    set_last_error(format!("live search: {e}"));
    FfiStatus::InvalidArgument
}

/// Opens a view of every buffered document a reader of the commit at
/// `commit_generation` lacks (`-1`: the documents in no published segment),
/// its handle into `*out_view` and its blob (see the module doc) as an owned
/// buffer.
///
/// # Safety
/// `out_view`, `out_ptr` and `out_len` valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_live_view_open(
    writer: u64,
    commit_generation: i64,
    out_view: *mut u64,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if out_view.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::clear_out(out_ptr, out_len)? };
        let w = engine_writer::lookup(writer)?;
        let commit = (commit_generation >= 0).then_some(commit_generation);
        let view = w.writer().open_live_view(commit).map_err(live_error)?;
        let blob = match view_blob(&view) {
            Ok(b) => b,
            Err(e) => {
                w.writer().close_live_view(&view);
                return Err(e);
            }
        };
        let id = NEXT_VIEW.fetch_add(1, Ordering::Relaxed);
        views()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                id,
                Arc::new(ViewHandle {
                    writer: w,
                    view,
                    whole: Mutex::new(Whole::None),
                }),
            );
        // SAFETY: caller contract.
        unsafe {
            *out_view = id;
            crate::ffm_bridge::give(blob, out_ptr, out_len);
        }
        Ok(())
    })
}

fn view_blob(view: &LiveView) -> Result<Vec<u8>, FfiStatus> {
    let mut blob = Vec::new();
    blob.extend_from_slice(
        &i64::try_from(view.published)
            .unwrap_or(i64::MAX)
            .to_le_bytes(),
    );
    blob.extend_from_slice(&len_i32(view.buffers.len())?.to_le_bytes());
    for b in &view.buffers {
        blob.extend_from_slice(&b.max_doc.to_le_bytes());
        blob.extend_from_slice(&len_i32(b.deleted.len())?.to_le_bytes());
        for d in &b.deleted {
            blob.extend_from_slice(&d.to_le_bytes());
        }
    }
    Ok(blob)
}

fn len_i32(n: usize) -> Result<i32, FfiStatus> {
    i32::try_from(n).map_err(|_| {
        set_last_error("live view: too many entries for the JVM");
        FfiStatus::InvalidArgument
    })
}

/// Closes a view: its buffers may go once no other view reads them.
#[no_mangle]
pub extern "C" fn ffi_live_view_close(view: u64) -> i32 {
    guard(|| {
        let h = views()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&view)
            .ok_or_else(|| {
                set_last_error("live view: unknown or already-closed handle");
                FfiStatus::InvalidHandle
            })?;
        h.writer.writer().close_live_view(&h.view);
        let whole = std::mem::replace(
            &mut *h
                .whole
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            Whole::Closed,
        );
        if let Whole::Open(_, whole) = whole {
            crate::jvm_reader::ffi_close_jvm_reader(whole);
        }
        Ok(())
    })
}

/// A reader handle over the reader at `handle` and `view`'s buffers as
/// segments after its own, into `*out_handle`: every search the JVM reader
/// takes ([`crate::jvm_reader`], aggregations, sorts) then reads the
/// buffered documents too, numbered as [`ffi_jvm_reader_search_live`]
/// numbers them. The buffers are written into memory as their flush would
/// write them, once per view; the deletes since the reader's commit, and
/// the buffers' own, are its deleted documents. The handle belongs to the
/// view: closing the view closes it, and the caller must not.
///
/// # Safety
/// `out_handle` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_view_reader(
    handle: u64,
    view_id: u64,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let v = view(view_id)?;
        let mut whole = v
            .whole
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match &*whole {
            Whole::None => {}
            Whole::Open(base, id) => {
                if *base != handle {
                    set_last_error("live view: its reader handle is over another reader");
                    return Err(FfiStatus::InvalidArgument);
                }
                // SAFETY: caller contract.
                unsafe { *out_handle = *id };
                return Ok(());
            }
            Whole::Failed(e) => {
                set_last_error(e.clone());
                return Err(FfiStatus::InvalidArgument);
            }
            Whole::Closed => {
                set_last_error("live view: unknown or already-closed handle");
                return Err(FfiStatus::InvalidHandle);
            }
        }
        let h = lookup(handle, "ffi_live_view_reader: unknown or closed reader")?;
        let dir = lucene_store::ByteBuffersDirectory::new();
        let infos = match v.writer.writer().live_view_segments(&v.view, &dir) {
            Ok(infos) => infos,
            Err(e @ lucene_index::index_writer::Error::LiveSearch(_)) => {
                let msg = format!("live search: {e}");
                *whole = Whole::Failed(msg.clone());
                set_last_error(msg);
                return Err(FfiStatus::InvalidArgument);
            }
            Err(e) => return Err(live_error(e)),
        };
        let reader = h.reader.with_segments_appended(&dir, &infos).map_err(|e| {
            set_last_error(format!("opening the buffered documents: {e}"));
            FfiStatus::Decode
        })?;
        let mut live_docs: Vec<Option<FixedBitSet>> = if v.view.commit_deletes.is_empty() {
            h.live_docs.clone()
        } else {
            let opened = h.reader.open_segments().map_err(|e| {
                set_last_error(format!("opening segment postings: {e}"));
                FfiStatus::Decode
            })?;
            let segments = opened.as_open_segments();
            match commit_masks(&v.view.commit_deletes, &h, &segments)? {
                Some(m) => m.iter().map(|l| l.as_deref().cloned()).collect(),
                None => h.live_docs.clone(),
            }
        };
        let readers = reader.segment_readers();
        let first = h.live_docs.len();
        for (b, r) in v.view.buffers.iter().zip(readers.iter().skip(first)) {
            live_docs.push(buffer_mask(b.max_doc, &b.deleted, r.max_doc));
        }
        let deleted = readers
            .iter()
            .zip(&live_docs)
            .map(|(r, l)| {
                l.as_ref().map_or(0, |l| {
                    i64::from(r.max_doc).saturating_sub(i64::try_from(l.cardinality()).unwrap_or(0))
                })
            })
            .collect();
        let id = crate::registry::lock_recovering(crate::registry::jvm_readers()).insert_checked(
            Arc::new(JvmReaderHandle {
                reader,
                live_docs,
                deleted,
                commit_live: Mutex::new(None),
            }),
        )?;
        *whole = Whole::Open(handle, id);
        // SAFETY: caller contract.
        unsafe { *out_handle = id };
        Ok(())
    })
}

/// A buffer segment's live documents: the first `max_doc` of the
/// `written`, less `deleted`; `None` when that is all of them.
fn buffer_mask(max_doc: i32, deleted: &[i32], written: i32) -> Option<FixedBitSet> {
    if deleted.is_empty() && max_doc >= written {
        return None;
    }
    let len = usize::try_from(written).unwrap_or(0);
    let mut bits = FixedBitSet::new(len);
    bits.set_range(0, usize::try_from(max_doc).unwrap_or(0).min(len));
    for &d in deleted {
        if let Ok(d) = usize::try_from(d) {
            if d < bits.len() {
                bits.clear(d);
            }
        }
    }
    Some(bits)
}

/// Stops keeping the published buffers the commit at `generation` holds,
/// once no open view reads them.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_release_live(writer: u64, generation: i64) -> i32 {
    guard(|| {
        let w = engine_writer::lookup(writer)?;
        w.writer().release_live(generation).map_err(live_error)
    })
}

/// Document `doc` of buffer `buffer` of `view`, its stored fields encoded
/// as [`crate::jvm_fetch`]'s document blob, as an owned buffer.
///
/// # Safety
/// `out_ptr`/`out_len` valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_view_document_alloc(
    view_id: u64,
    buffer: i32,
    doc: i32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::clear_out(out_ptr, out_len)? };
        let h = view(view_id)?;
        let buffer = usize::try_from(buffer).map_err(|_| FfiStatus::IndexOutOfBounds)?;
        let fields = h
            .writer
            .writer()
            .live_document(&h.view, buffer, doc)
            .map_err(live_error)?;
        let mut out = len_i32(fields.len())?.to_le_bytes().to_vec();
        for f in &fields {
            out.extend_from_slice(&f.field_number.to_le_bytes());
            match &f.value {
                FieldValue::String(s) => {
                    out.push(STRING);
                    out.extend_from_slice(&len_i32(s.len())?.to_le_bytes());
                    out.extend_from_slice(s.as_bytes());
                }
                FieldValue::Binary(b) => {
                    out.push(BINARY);
                    out.extend_from_slice(&len_i32(b.len())?.to_le_bytes());
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
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::give(out, out_ptr, out_len) };
        Ok(())
    })
}

/// [`crate::jvm_reader::ffi_jvm_reader_search`] over the reader at
/// `handle` and the buffers of `view` together: a buffered hit is
/// numbered after the reader's `maxDoc`, the buffers in the view's order.
/// Every field's average length is over both. `*out_supported` is false,
/// and nothing else is written, when the query reads something the buffers
/// cannot answer (see [`live_request`]) or carries a `min_score`; the
/// caller then makes the buffers part of a reader first.
///
/// # Safety
/// As [`crate::jvm_reader::ffi_jvm_reader_search`]; `out_supported` valid
/// for one write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_jvm_reader_search_live(
    handle: u64,
    view_id: u64,
    query: *const u8,
    query_len: usize,
    top_n: usize,
    count_limit: i64,
    out_docs: *mut i32,
    out_scores: *mut f32,
    buf_len: usize,
    out_hit_count: *mut usize,
    out_total: *mut i64,
    out_total_is_lower_bound: *mut bool,
    out_supported: *mut bool,
) -> i32 {
    guard(|| {
        if out_hit_count.is_null()
            || out_total.is_null()
            || out_total_is_lower_bound.is_null()
            || out_supported.is_null()
        {
            return Err(FfiStatus::NullPointer);
        }
        if buf_len < top_n {
            return Err(FfiStatus::BufferTooSmall);
        }
        if top_n > 0 && (out_docs.is_null() || out_scores.is_null()) {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(query, query_len)? };
        let (query, min_score, sim) = decode_request(blob)?;
        let q = boolean_of(&query);
        let clause = Clause::Boolean(Box::new(q.clone()));
        let request = match (min_score, live_request(&clause)) {
            (None, Some(r)) => r,
            _ => {
                // SAFETY: caller contract.
                unsafe { *out_supported = false };
                return Ok(());
            }
        };
        let h = lookup(
            handle,
            "ffi_jvm_reader_search_live: unknown or closed reader",
        )?;
        let v = view(view_id)?;
        let snapshot = v
            .writer
            .writer()
            .live_view_snapshot(&v.view, &request)
            .map_err(live_error)?;
        let live = live_segments(snapshot).map_err(map_search_error)?;
        let inputs = live
            .iter()
            .map(|l| l.inputs())
            .collect::<lucene_search::Result<Vec<_>>>()
            .map_err(map_search_error)?;

        let opened = h.reader.open_segments().map_err(|e| {
            set_last_error(format!("opening segment postings: {e}"));
            FfiStatus::Decode
        })?;
        let sim = sim.as_ref();
        let mut segments: Vec<OpenSegment<'_>> = opened.as_open_segments();
        let masks = commit_masks(&v.view.commit_deletes, &h, &segments)?;
        for (i, s) in segments.iter_mut().enumerate() {
            s.live_docs = match &masks {
                Some(m) => m.get(i).and_then(Option::as_deref),
                None => h.live_docs.get(i).and_then(Option::as_ref),
            };
            s.similarity = sim.map(|p| p as &dyn Similarity);
        }
        let mut doc_base = h.reader.max_doc();
        for (l, i) in live.iter().zip(&inputs) {
            let mut s = l.segment(i, doc_base);
            s.similarity = sim.map(|p| p as &dyn Similarity);
            segments.push(s);
            doc_base = doc_base.saturating_add(l.max_doc());
        }

        let fields: Vec<String> = crate::clauses::clause_field_names(&q)
            .into_iter()
            .map(str::to_string)
            .collect();
        let extra = |field: &str| {
            let mut found = None;
            for l in &live {
                if let Some([_, ttf, _, dc]) = l.field_stats(field) {
                    let (t, d) = found.unwrap_or((0i64, 0i64));
                    found = Some((t.saturating_add(ttf), d.saturating_add(dc)));
                }
            }
            found
        };
        let (reader_norms, averages) = h
            .reader
            .field_norms_by_field_with(&fields, &extra)
            .map_err(|e| map_search_error(e.into()))?;
        let live_norms: Vec<_> = live
            .iter()
            .map(|l| l.norms_with_average(&|f| averages.get(f).copied()))
            .collect();
        let norms: Vec<_> = reader_norms
            .iter()
            .chain(&live_norms)
            .map(|m| (!m.is_empty()).then_some(m))
            .collect();
        let threshold = u64::try_from(count_limit).unwrap_or(0);
        let (hits, total) = search_boolean_query_multi_segment_maxscore_counting(
            &segments,
            &q,
            &norms,
            top_n.max(1),
            threshold,
        )
        .map_err(map_search_error)?;
        let hits = if top_n == 0 { Vec::new() } else { hits };
        let (total, lower_bound) = if count_limit <= 0 {
            (-1, false)
        } else {
            (
                i64::try_from(total.value).unwrap_or(i64::MAX),
                total.relation == TotalHitsRelation::GreaterThanOrEqualTo,
            )
        };
        // SAFETY: caller contract; `hits.len() <= top_n <= buf_len`.
        unsafe {
            for (i, hit) in hits.iter().enumerate() {
                *out_docs.add(i) = hit.doc_id;
                *out_scores.add(i) = hit.score;
            }
            *out_hit_count = hits.len();
            *out_total = total;
            *out_total_is_lower_bound = lower_bound;
            *out_supported = true;
        }
        Ok(())
    })
}

/// The reader's live documents with `terms` (a view's
/// [`LiveView::commit_deletes`]) applied to `segments`, its segments; `None`
/// when there are none to apply. Kept on the reader for the next search.
fn commit_masks(
    terms: &[Term],
    h: &JvmReaderHandle,
    segments: &[OpenSegment<'_>],
) -> Result<Option<Arc<Masks>>, FfiStatus> {
    let Some(last) = terms.last() else {
        return Ok(None);
    };
    let mut cached = h
        .commit_live
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let extends = cached.as_ref().filter(|c| {
        c.applied <= terms.len()
            && c.applied
                .checked_sub(1)
                .and_then(|i| terms.get(i))
                .is_some_and(|t| *t == c.last)
    });
    let (from, mut masks) = match extends {
        Some(c) if c.applied == terms.len() => return Ok(Some(Arc::clone(&c.masks))),
        Some(c) => (c.applied, Masks::clone(&c.masks)),
        None => (
            0,
            h.live_docs
                .iter()
                .map(|l| l.clone().map(Arc::new))
                .collect(),
        ),
    };
    let mut hits = VecCollector::default();
    for t in &terms[from..] {
        let query = TermQuery::new(t.field.as_str(), t.bytes.as_slice());
        for (seg, mask) in segments.iter().zip(masks.iter_mut()) {
            hits.docs.clear();
            search_term_query(seg.fields, seg.doc_in, None, &query, &mut hits)
                .map_err(map_search_error)?;
            if hits.docs.is_empty() {
                continue;
            }
            let max_doc = usize::try_from(seg.max_doc.unwrap_or(0)).unwrap_or(0);
            let mask = mask.get_or_insert_with(|| {
                let mut all = FixedBitSet::new(max_doc);
                all.set_range(0, max_doc);
                Arc::new(all)
            });
            let mask = Arc::make_mut(mask);
            let len = mask.len();
            for &d in &hits.docs {
                if let Ok(d) = usize::try_from(d) {
                    if d < len {
                        mask.clear(d);
                    }
                }
            }
        }
    }
    let masks = Arc::new(masks);
    // A view older than the cached masks keeps its own.
    if cached.as_ref().is_none_or(|c| c.applied <= terms.len()) {
        *cached = Some(CommitLive {
            applied: terms.len(),
            last: last.clone(),
            masks: Arc::clone(&masks),
        });
    }
    Ok(Some(masks))
}
