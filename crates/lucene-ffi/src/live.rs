//! Live search entry points for the plugin, over [`lucene_engine::live`]:
//! documents still in the engine writer's buffers, searched and fetched
//! without a refresh. The view machinery is in the engine crate; the
//! functions below decode the caller's pointers and call it.

use std::sync::atomic::Ordering;

use lucene_codecs::stored_fields::FieldValue;
use lucene_search::live_segment::live_request;
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::Clause;
use lucene_search::similarities::Similarity;
use lucene_search::{
    search_boolean_query_multi_segment_maxscore_counting, search_term_query_multi_segment_counting,
    TotalHitsRelation,
};

use crate::engine_writer::{self, EngineWriter};
use crate::error::{guard, set_last_error, FfiStatus};
use crate::jvm_fetch::{BINARY, DOUBLE, FLOAT, INT, LONG, STRING};
use crate::jvm_reader::lookup;
use crate::raw::bytes_from_raw;
use lucene_engine::clauses::map_search_error;
use lucene_engine::query_blob::{boolean_of, decode_request, JvmQuery};

pub(crate) use lucene_engine::live::*;

/// Lets searches of `writer`'s buffers read a view up to `millis`
/// milliseconds old: the writer's warming thread opens one every few
/// milliseconds, its chunks written, and the searches meanwhile share it,
/// its readers and their caches, writing nothing themselves. A search then
/// misses the writes of at most that long. 0 (the default) gives each search
/// a view of every write acknowledged before it.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_set_live_staleness(writer: u64, millis: i64) -> i32 {
    guard(|| {
        let w = engine_writer::lookup(writer)?;
        let ms = u64::try_from(millis).map_err(|_| {
            set_last_error("live staleness: negative");
            FfiStatus::InvalidArgument
        })?;
        if ms == 0 {
            w.live_warmer().stop_sharing();
        } else {
            w.live_warmer()
                .max_staleness_ms
                .store(ms, Ordering::Relaxed);
        }
        Ok(())
    })
}

/// Under a staleness bound, makes every later search read a view opened
/// from now on, so it holds every write acknowledged before this call:
/// what lets `refresh=wait_for` answer at once.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_live_refresh_shared(writer: u64) -> i32 {
    guard(|| {
        engine_writer::lookup(writer)?
            .live_warmer()
            .refresh_shared();
        Ok(())
    })
}

/// Opens a view of every buffered document a reader of the segment list at
/// version `reader_version` (a commit's or an NRT snapshot's) lacks (`-1`:
/// the documents in no published segment),
/// its handle into `*out_view` and its blob (see the module doc) as an owned
/// buffer.
///
/// # Safety
/// `out_view`, `out_ptr` and `out_len` valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_live_view_open(
    writer: u64,
    reader_version: i64,
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
        let commit = (reader_version >= 0).then_some(reader_version);
        let warmer = w.live_warmer();
        if warmer.max_staleness().is_some() {
            // The warmer opens the views searches share, and forgets the
            // shared one once no search comes.
            searched_lean(&w);
        }
        let h = match warmer.shared_view(commit) {
            Some(h) => h,
            None => {
                warmer.asked(commit);
                open_handle(&w, commit, false)?
            }
        };
        let blob = h.blob.clone();
        // A view searches share keeps its handle: each open of it counts.
        let id = *h
            .id
            .get_or_init(|| NEXT_VIEW.fetch_add(1, Ordering::Relaxed));
        let mut open = views()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = open.entry(id).or_insert((h, 0));
        entry.1 = entry.1.saturating_add(1);
        drop(open);
        // SAFETY: caller contract.
        unsafe {
            *out_view = id;
            crate::ffm_bridge::give(blob, out_ptr, out_len);
        }
        Ok(())
    })
}

/// Closes one open of a view: a view searches share has one handle for all
/// its opens, closed as many times. Its buffers may go once no other view
/// reads them.
#[no_mangle]
pub extern "C" fn ffi_live_view_close(view: u64) -> i32 {
    guard(|| {
        let mut open = views()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(entry) = open.get_mut(&view) else {
            set_last_error("live view: unknown or already-closed handle");
            return Err(FfiStatus::InvalidHandle);
        };
        entry.1 = entry.1.saturating_sub(1);
        let last = (entry.1 == 0).then(|| open.remove(&view));
        drop(open);
        // The view closes when its last handle goes, outside the lock: see
        // `ViewHandle`'s drop.
        drop(last);
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
/// With a request (`query` as [`decode_request`] reads it, `tree` an
/// aggregation tree or empty, and `extra` more queries the search runs, a
/// rescore's, each an `i32` length then the blob, or empty), the buffers
/// are written without their postings and the handle holds only the terms
/// the request reads: it serves searches of that request alone. Without
/// one (`query_len` 0, when `extra` is not read), or for a request reading
/// what that cannot give, the handle holds every term.
///
/// # Safety
/// `out_handle` valid for one write; `query`/`tree`/`extra` readable for
/// their lengths when non-zero.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_view_reader(
    handle: u64,
    view_id: u64,
    query: *const u8,
    query_len: usize,
    tree: *const u8,
    tree_len: usize,
    extra: *const u8,
    extra_len: usize,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        let _timer = lucene_index::live_timing::start(lucene_index::live_timing::Phase::ViewReader);
        if out_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let v = view(view_id)?;
        // SAFETY: caller contract.
        let extra = (extra_len > 0 && query_len > 0)
            .then(|| unsafe { bytes_from_raw(extra, extra_len) })
            .transpose()?;
        // SAFETY: caller contract.
        let query = (query_len > 0)
            .then(|| unsafe { bytes_from_raw(query, query_len) })
            .transpose()?;
        // SAFETY: caller contract.
        let tree = (tree_len > 0)
            .then(|| unsafe { bytes_from_raw(tree, tree_len) })
            .transpose()?;
        if query.is_some() && v.writer.live_warmer().max_staleness().is_some() {
            v.writer
                .live_warmer()
                .note_reader(handle, v.commit, query, tree, extra);
        }
        let id = view_reader_of(handle, &v, query, tree, extra)?;
        // SAFETY: caller contract.
        unsafe { *out_handle = id };
        Ok(())
    })
}

/// [`crate::jvm_knn::ffi_jvm_reader_knn`] over the reader at `handle` and
/// `view_id`'s buffers, numbered as [`ffi_jvm_reader_search_live`] numbers
/// them: the request, outputs and errors are that call's. The buffers are
/// read through a handle of [`ffi_live_view_reader`]'s that holds only the
/// filter's terms (every term for a nested field, whose parent filter it
/// does not know), cached with the view; their vectors are searched
/// exactly, as a realtime chunk keeps no graph.
///
/// # Safety
/// As [`crate::jvm_knn::ffi_jvm_reader_knn`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_live_view_knn(
    handle: u64,
    view_id: u64,
    request: *const u8,
    request_len: usize,
    out_docs: *mut i32,
    out_scores: *mut f32,
    buf_len: usize,
    out_count: *mut usize,
) -> i32 {
    guard(|| {
        if out_docs.is_null() || out_scores.is_null() || out_count.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(request, request_len)? };
        let knn = lucene_engine::knn::decode(blob)?;
        if buf_len < knn.k {
            return Err(FfiStatus::BufferTooSmall);
        }
        let v = view(view_id)?;
        let (request, key) = if knn.whole_segments() {
            (None, None)
        } else {
            let all = Clause::MatchAllDocs(lucene_search::query::MatchAllDocsQuery::new(0));
            let clause = knn.filter.as_ref().unwrap_or(&all);
            let refs = [clause];
            match lucene_search::live_segment::whole_request(&refs) {
                Some(r) => {
                    let points = lucene_search::live_segment::reads_points(&refs);
                    // Keyed by the filter's own encoding, which names a
                    // filter exactly. The view's request budget is shared
                    // with the query phase: a view searched with many
                    // distinct filters leaves later requests the reader of
                    // every term.
                    let mut key = b"knn-filter:".to_vec();
                    key.extend_from_slice(&blob[knn.filter_bytes.clone()]);
                    (Some((r, points)), Some(key))
                }
                None => (None, None),
            }
        };
        let id =
            lucene_index::live_timing::timed(lucene_index::live_timing::Phase::KnnReader, || {
                view_reader(handle, &v, request, key)
            })?;
        let h = lookup(id, "ffi_live_view_knn: the view's reader closed")?;
        let base = lookup(handle, "ffi_live_view_knn: unknown or closed reader")?
            .reader
            .segment_readers()
            .len();
        let buffers = if knn.whole_segments() {
            Vec::new()
        } else {
            buffer_graphs(&v, &h, base, &knn.field)
        };
        let hits =
            lucene_index::live_timing::timed(lucene_index::live_timing::Phase::KnnSearch, || {
                lucene_engine::knn::search(&h, &knn, &buffers)
            })?;
        // SAFETY: caller contract.
        unsafe { crate::jvm_knn::write_hits(&hits, out_docs, out_scores, buf_len, out_count) }
    })
}

/// What a search Lucene answers reads the view's buffers through, into
/// `*out_ptr`/`*out_len`: the deletes since the commit of the reader at
/// `handle` (the view's base), and each leaf of the view as a segment with
/// its postings, for the JVM to open.
///
/// Little-endian: `segments: i32`, then per segment of the reader its
/// live documents as of the view (`words: i32`, `-1` when none were
/// deleted since the commit, then that many `i64`); then `leaves: i32`, and
/// per leaf `kind: u8` (`0` a segment in memory, `1` the segment its buffer
/// was flushed as, in the index's directory), the segment's name (`i32`
/// length, UTF-8), its 16-byte id, its codec's name (as the name), the leaf's first document in it
/// (`i32`), and with kind `0` its files (`count: i32`, each a name as above,
/// `len: i64` and the bytes). A leaf of a flushed buffer may be a part of
/// that segment, which also holds documents added after the view.
///
/// A buffer taken for a flush since the view opened is waited for up to
/// `wait_ms` to publish; [`FfiStatus::InvalidArgument`] when it has not.
///
/// # Safety
/// `out_ptr`/`out_len` valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_view_lucene(
    handle: u64,
    view_id: u64,
    wait_ms: i64,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::clear_out(out_ptr, out_len)? };
        let v = view(view_id)?;
        let h = lookup(handle, "ffi_live_view_lucene: unknown or closed reader")?;
        let w: &EngineWriter = &v.writer;
        let wait = std::time::Duration::from_millis(u64::try_from(wait_ms).unwrap_or(0));
        let deadline = std::time::Instant::now()
            .checked_add(wait)
            .unwrap_or_else(std::time::Instant::now);
        let mut leaves: Vec<(ChunkFiles, i32)> = Vec::new();
        for (b, plan) in v.view.buffers.iter().zip(&v.plans) {
            match buffer_segment_files(w, b, plan, true, false, None) {
                Ok(parts) => leaves.extend(parts.into_iter().map(|(_, _, files)| (files, 0))),
                Err(lucene_index::index_writer::Error::LiveSearch(_)) => {
                    // Taken for a flush since the view opened: its documents
                    // are read from the segment it is flushed as.
                    let sci = loop {
                        if let Some(sci) = w.writer().live_published_segment(b.id) {
                            break sci;
                        }
                        if std::time::Instant::now() >= deadline {
                            set_last_error(format!(
                                "live view: buffer {} is still being flushed",
                                b.id
                            ));
                            return Err(FfiStatus::InvalidArgument);
                        }
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    };
                    leaves.extend(
                        plan.iter()
                            .map(|p| (ChunkFiles::Published(sci.clone()), p.start)),
                    );
                }
                Err(e) => return Err(live_error(e)),
            }
        }
        let masks = if v.view.commit_deletes.is_empty() {
            None
        } else {
            let opened = h.reader.open_segments().map_err(|e| {
                set_last_error(format!("opening segment postings: {e}"));
                FfiStatus::Decode
            })?;
            let segments = opened.as_open_segments();
            commit_masks(&v.view.commit_deletes, &h, &segments)?
        };
        let mut out = Vec::new();
        let segments = h.reader.segment_readers().len();
        put_len(&mut out, segments)?;
        for i in 0..segments {
            match masks
                .as_ref()
                .and_then(|m| m.get(i))
                .and_then(Option::as_ref)
            {
                Some(mask) => {
                    put_len(&mut out, mask.words().len())?;
                    mask.words()
                        .iter()
                        .for_each(|w| out.extend_from_slice(&w.to_le_bytes()));
                }
                None => out.extend_from_slice(&(-1i32).to_le_bytes()),
            }
        }
        put_len(&mut out, leaves.len())?;
        for (files, offset) in &leaves {
            let (kind, sci, dir) = match files {
                ChunkFiles::Memory(dir, sci) => (0u8, sci, Some(dir)),
                ChunkFiles::Published(sci) => (1u8, sci, None),
                ChunkFiles::None => {
                    set_last_error("live view: a buffered chunk without postings");
                    return Err(FfiStatus::Decode);
                }
            };
            out.push(kind);
            put_bytes(&mut out, sci.segment_name.as_bytes())?;
            out.extend_from_slice(&sci.segment_id);
            put_bytes(&mut out, sci.codec_name.as_bytes())?;
            out.extend_from_slice(&offset.to_le_bytes());
            if let Some(dir) = dir {
                let names = lucene_store::Directory::list_all(&**dir).map_err(|e| {
                    set_last_error(format!("live view: listing a chunk's files: {e}"));
                    FfiStatus::Decode
                })?;
                put_len(&mut out, names.len())?;
                for name in names {
                    let input = lucene_store::Directory::open(&**dir, &name).map_err(|e| {
                        set_last_error(format!("live view: reading {name}: {e}"));
                        FfiStatus::Decode
                    })?;
                    put_bytes(&mut out, name.as_bytes())?;
                    out.extend_from_slice(
                        &u64::try_from(input.len()).unwrap_or(u64::MAX).to_le_bytes(),
                    );
                    out.extend_from_slice(&input);
                }
            }
        }
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::give(out, out_ptr, out_len) };
        Ok(())
    })
}

/// Stops keeping the published buffers the segment list at version
/// `reader_version` (a commit's or an NRT snapshot's) holds, once no open
/// view reads them.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_release_live(writer: u64, reader_version: i64) -> i32 {
    guard(|| {
        let w = engine_writer::lookup(writer)?;
        w.live_warmer().unshare_before(reader_version);
        w.writer().release_live(reader_version).map_err(live_error)
    })
}

/// The bounds of field `field`'s points in buffer `buffer` of `view` as the view
/// opened ([`lucene_index::concurrent_writer::ConcurrentIndexWriter::live_point_bounds`]),
/// as an owned buffer: the point count as a little-endian `u64`, then the
/// packed least values, then the packed greatest; empty when the buffer
/// holds no point of the field.
///
/// # Safety
/// `out_ptr`/`out_len` valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_view_point_bounds_alloc(
    view_id: u64,
    buffer: i32,
    field: i32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::clear_out(out_ptr, out_len)? };
        let h = view(view_id)?;
        let buffer = usize::try_from(buffer).map_err(|_| FfiStatus::IndexOutOfBounds)?;
        let bounds = h
            .writer
            .writer()
            .live_point_bounds(&h.view, buffer, field)
            .map_err(live_error)?;
        let mut out = Vec::new();
        if let Some(b) = bounds {
            out.extend_from_slice(&b.count.to_le_bytes());
            out.extend_from_slice(&b.min);
            out.extend_from_slice(&b.max);
        }
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::give(out, out_ptr, out_len) };
        Ok(())
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
        let mut fields = h
            .writer
            .writer()
            .live_document(&h.view, buffer, doc)
            .map_err(live_error)?;
        if let Some(derived) = h.writer.live_derived() {
            derived.inject(&h.writer, &h.view, buffer, doc, &mut fields)?;
        }
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

/// Document `doc` of buffer `buffer` of `view`: its doc values of the
/// `count` fields numbered at `fields`, encoded as [`crate::jvm_fetch`]'s
/// doc-values blob, as an owned buffer. A buffer has no terms dictionary:
/// a document's terms are numbered `0..n` in order, ordinals of that
/// document alone.
///
/// # Safety
/// `fields` valid for `count` `i32` reads (null when 0); `out_ptr`/`out_len`
/// valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_view_doc_values_alloc(
    view_id: u64,
    buffer: i32,
    doc: i32,
    fields: *const i32,
    count: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::clear_out(out_ptr, out_len)? };
        if fields.is_null() && count > 0 {
            return Err(FfiStatus::NullPointer);
        }
        let fields: &[i32] = if count == 0 {
            &[]
        } else {
            // SAFETY: caller contract, non-null checked above.
            unsafe { std::slice::from_raw_parts(fields, count) }
        };
        let h = view(view_id)?;
        let buffer = usize::try_from(buffer).map_err(|_| FfiStatus::IndexOutOfBounds)?;
        let values = h
            .writer
            .writer()
            .live_doc_values(&h.view, buffer, doc, fields)
            .map_err(live_error)?;
        let out = encode_live_doc_values(fields, &values)?;
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
        let _timer =
            lucene_index::live_timing::start(lucene_index::live_timing::Phase::PlainSearch);
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
        searched_lean(&v.writer);
        let mut laps = lucene_index::live_timing::Laps::start();
        // Each buffer is read as the chunks its view's plan starts with,
        // with this request's terms (their overlays kept on the chunks),
        // then a segment of the documents after them.
        let key = {
            let mut k = vec![PLAIN_KEY];
            k.extend(request_key(blob, None, None));
            k
        };
        if v.writer.live_warmer().max_staleness().is_some() {
            v.writer.live_warmer().note_plain(blob);
        }
        let prepared = prepare_plain(&v, &request, &key)?;
        let Prepared {
            chunk_readers,
            chunk_masks,
            chunk_docs,
            live,
            prefix_lens,
        } = &*prepared;
        laps.lap(lucene_index::live_timing::Phase::PlainPrepare);
        let inputs = live
            .iter()
            .map(|l| l.inputs())
            .collect::<lucene_search::Result<Vec<_>>>()
            .map_err(map_search_error)?;

        let chunked;
        let reader = if chunk_readers.is_empty() {
            &h.reader
        } else {
            chunked = h
                .reader
                .with_readers_appended(chunk_readers.clone())
                .map_err(|e| {
                    set_last_error(format!("opening the buffered documents: {e}"));
                    FfiStatus::Decode
                })?;
            &chunked
        };
        let opened = reader.open_segments().map_err(|e| {
            set_last_error(format!("opening segment postings: {e}"));
            FfiStatus::Decode
        })?;
        laps.lap(lucene_index::live_timing::Phase::PlainOpen);
        let sim = sim.as_ref();
        let committed = h.live_docs.len();
        let mut opened_segments: Vec<OpenSegment<'_>> = opened.as_open_segments();
        let masks = commit_masks(
            &v.view.commit_deletes,
            &h,
            opened_segments.get(..committed).unwrap_or_default(),
        )?;
        for (i, s) in opened_segments.iter_mut().enumerate() {
            s.live_docs = match i.checked_sub(committed) {
                Some(c) => chunk_masks.get(c).and_then(Option::as_ref),
                None => match &masks {
                    Some(m) => m.get(i).and_then(Option::as_deref),
                    None => h.live_docs.get(i).and_then(Option::as_ref),
                },
            };
            s.similarity = sim.map(|p| p as &dyn Similarity);
        }

        let fields: Vec<String> = lucene_engine::clauses::clause_field_names(&q)
            .into_iter()
            .map(str::to_string)
            .collect();
        let extra = |field: &str| {
            let mut found = None;
            for l in live {
                if let Some([_, ttf, _, dc]) = l.field_stats(field) {
                    let (t, d) = found.unwrap_or((0i64, 0i64));
                    found = Some((t.saturating_add(ttf), d.saturating_add(dc)));
                }
            }
            found
        };
        let (reader_norms, averages) = reader
            .field_norms_by_field_with(&fields, &extra)
            .map_err(|e| map_search_error(e.into()))?;
        let live_norms: Vec<_> = live
            .iter()
            .map(|l| l.norms_with_average(&|f| averages.get(f).copied()))
            .collect();
        laps.lap(lucene_index::live_timing::Phase::PlainNorms);
        // In document order: the committed segments, then each buffer's
        // chunks and the segment after them.
        let mut segments: Vec<OpenSegment<'_>> = Vec::new();
        let mut norms = Vec::new();
        let mut opened_iter = opened_segments.into_iter().zip(reader_norms.iter());
        for _ in 0..committed {
            if let Some((s, n)) = opened_iter.next() {
                segments.push(s);
                norms.push(n);
            }
        }
        let mut doc_base = h.reader.max_doc();
        let mut chunk_docs = chunk_docs.iter().copied();
        for (((&n, l), i), ln) in prefix_lens.iter().zip(live).zip(&inputs).zip(&live_norms) {
            for _ in 0..n {
                if let (Some((mut s, n)), Some(docs)) = (opened_iter.next(), chunk_docs.next()) {
                    s.doc_base = doc_base;
                    doc_base = doc_base.saturating_add(docs);
                    segments.push(s);
                    norms.push(n);
                }
            }
            let mut s = l.segment(i, doc_base);
            s.similarity = sim.map(|p| p as &dyn Similarity);
            segments.push(s);
            norms.push(ln);
            doc_base = doc_base.saturating_add(l.max_doc());
        }
        let norms: Vec<_> = norms
            .into_iter()
            .map(|m| (!m.is_empty()).then_some(m))
            .collect();
        let threshold = u64::try_from(count_limit).unwrap_or(0);
        // A lone term is scored as the plain reader scores it, with the term
        // scorer: Lucene rewrites a one-clause boolean to that clause, so the
        // hits are the same, without a boolean scorer per leaf.
        let searched = match &query {
            JvmQuery::Term(t) => {
                let term_norms: Vec<_> = norms
                    .iter()
                    .map(|m| m.and_then(|m| m.get(&t.field)))
                    .collect();
                search_term_query_multi_segment_counting(
                    &segments,
                    t,
                    &term_norms,
                    top_n.max(1),
                    threshold,
                )
            }
            JvmQuery::Boolean(_) => search_boolean_query_multi_segment_maxscore_counting(
                &segments,
                &q,
                &norms,
                top_n.max(1),
                threshold,
            ),
        }
        .map_err(map_search_error);
        let (hits, total) = lucene_engine::query_blob::similarity_checked(sim, searched)?;
        laps.lap(lucene_index::live_timing::Phase::PlainQuery);
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
