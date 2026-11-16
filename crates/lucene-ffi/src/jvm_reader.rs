//! The reader handle the OpenSearch plugin searches, over
//! [`lucene_engine::reader_search`]: opening a handle from the JVM reader's
//! segment list and live docs, closing it, and the `extern "C"` search entry
//! points. The decoders and the searches are in the engine crate; the
//! functions below that take a `u64` resolve the handle and call them.

use std::os::raw::c_char;

use lucene_aggs::metric_aggs::MetricState;

use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::{bytes_from_raw, str_from_raw};
use crate::registry::{jvm_readers, lock_recovering, JvmReaderHandle};
use lucene_engine::query_blob::{decode_request, similarity_checked};
use std::sync::Arc;

pub(crate) use lucene_engine::reader_search::*;

/// Opens a reader over the segments listed in `infos` (a whole `segments_N`
/// file's bytes, as `SegmentInfos.write(IndexOutput)` writes them, at
/// generation `generation`) under the directory at `path`, checks that
/// segment `i` has exactly `expected_max_docs[i]` documents, and masks each
/// segment with the JVM's live docs.
///
/// **Live docs:** `live_word_counts[i]` is how many words of segment `i`'s
/// live-docs bitset follow in `live_words` (the segments' words concatenated,
/// in order) -- `FixedBitSet.getBits()` of the Java leaf's live docs, bit `d`
/// set when doc `d` is live. `0` means the segment has no deletions; any other
/// count must be exactly `bits2words(maxDoc)`, with no bit set past `maxDoc`.
/// A null `live_word_counts` means no segment has deletions. They replace
/// whatever the segment has on disk: an NRT reader's deletions are in memory.
///
/// Everything a search reads is fixed here, so the handle is immutable once
/// published and a search holds no lock while it runs (see
/// [`ffi_jvm_reader_search`]).
///
/// `previous` is a handle from an earlier call over the same directory, or
/// `0`: its unchanged segments are shared instead of re-read. It stays open
/// -- the caller closes it when its own Java reader closes.
///
/// # Safety
/// `path` must be valid for `path_len` bytes, `infos` for `infos_len` bytes,
/// `expected_max_docs` and (when non-null) `live_word_counts` for
/// `segment_count` elements each, `live_words` for the sum of
/// `live_word_counts` `u64`s, and `out_handle` for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_open_jvm_reader(
    path: *const c_char,
    path_len: usize,
    infos: *const u8,
    infos_len: usize,
    generation: i64,
    previous: u64,
    expected_max_docs: *const i32,
    segment_count: usize,
    live_words: *const u64,
    live_word_counts: *const usize,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_handle.is_null() || (expected_max_docs.is_null() && segment_count > 0) {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let (path, infos) = unsafe {
            (
                str_from_raw(path, path_len)?,
                bytes_from_raw(infos, infos_len)?,
            )
        };
        let (expected, counts): (&[i32], Option<&[usize]>) = if segment_count == 0 {
            (&[], None)
        } else {
            // SAFETY: caller contract, non-null checked above / here.
            unsafe {
                (
                    std::slice::from_raw_parts(expected_max_docs, segment_count),
                    (!live_word_counts.is_null())
                        .then(|| std::slice::from_raw_parts(live_word_counts, segment_count)),
                )
            }
        };
        let total_words = counts
            .unwrap_or(&[])
            .iter()
            .try_fold(0usize, |acc, &c| acc.checked_add(c))
            .ok_or_else(|| {
                set_last_error("live-docs word counts overflow");
                FfiStatus::InvalidArgument
            })?;
        if total_words > 0 && live_words.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let words: &[u64] = if total_words == 0 {
            &[]
        } else {
            // SAFETY: caller contract: `live_words` holds the sum of the counts.
            unsafe { std::slice::from_raw_parts(live_words, total_words) }
        };

        // Mapped, not copied: a shard's postings are gigabytes, and Lucene's own
        // `MMapDirectory` -- what OpenSearch searches with -- maps them too. The
        // mapping lives as long as the segment reader, which is why closing the
        // handle matters.
        // A `encrypted` shard's path is bound to an encrypting directory
        // over the mapping (`store.rs`), which decrypts each file on open.
        let dir = crate::store::shard_directory(path).map_err(|e| {
            set_last_error(format!("opening the shard directory [{path}]: {e}"));
            FfiStatus::Io
        })?;
        // Cloned out under a short read lock: the reopen reads files and
        // must not hold up every other search and close on the node.
        let prev = (previous != 0)
            .then(|| {
                lookup(
                    previous,
                    "ffi_open_jvm_reader: unknown or already-closed previous handle",
                )
            })
            .transpose()?;
        let opened = lucene_engine::reader_handle::open_reader(
            path,
            &*dir,
            infos,
            generation,
            prev.as_deref(),
            expected,
            counts,
            words,
        )?;
        let handle = lock_recovering(jvm_readers()).insert_checked(Arc::new(opened))?;
        // SAFETY: caller contract.
        unsafe { *out_handle = handle };
        Ok(())
    })
}

/// The host's mask of a field's values: writes the masked form of the `input_len` bytes at `input`,
/// for the field named by the `name_len` bytes at `name`, into `out` (room for `out_cap` bytes), and
/// answers its length -- which may exceed `out_cap`, when nothing was written and the caller asks
/// again with room for that many -- or a negative number when the mask fails.
pub type MaskFn = unsafe extern "C" fn(
    name: *const u8,
    name_len: usize,
    input: *const u8,
    input_len: usize,
    out: *mut u8,
    out_cap: usize,
) -> i64;

/// One masked field's mask: the host's [`MaskFn`] and the field's name.
struct HostMask {
    f: MaskFn,
    name: Vec<u8>,
}

impl std::fmt::Debug for HostMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostMask")
            .field("field", &String::from_utf8_lossy(&self.name))
            .finish_non_exhaustive()
    }
}

impl lucene_codecs::value_mask::ValueMask for HostMask {
    fn apply(&self, raw: &[u8]) -> lucene_store::Result<Vec<u8>> {
        let failed = |what: &str| {
            lucene_store::Error::IllegalArgument(format!(
                "the mask of field {} {what}",
                String::from_utf8_lossy(&self.name)
            ))
        };
        let mut out =
            crate::raw::try_with_capacity::<u8>(raw.len().saturating_mul(2).saturating_add(160))
                .map_err(|_| failed("is too large to answer"))?;
        out.resize(out.capacity(), 0);
        for _ in 0..2 {
            // SAFETY: the host's function, given buffers valid for the lengths passed; it is kept
            // alive for as long as the view is open, which the view's owner guarantees.
            let n = unsafe {
                (self.f)(
                    self.name.as_ptr(),
                    self.name.len(),
                    raw.as_ptr(),
                    raw.len(),
                    out.as_mut_ptr(),
                    out.len(),
                )
            };
            let n = usize::try_from(n).map_err(|_| failed("failed"))?;
            if n <= out.len() {
                out.truncate(n);
                return Ok(out);
            }
            out = crate::raw::try_with_capacity::<u8>(n)
                .map_err(|_| failed("is too large to answer"))?;
            out.resize(n, 0);
        }
        Err(failed("answered two different lengths"))
    }
}

/// Opens a view of the reader `base` names: its segments, each hiding the fields the `hidden`
/// blob lists for it, showing the fields the `masked` blob lists through `mask`, and masked with
/// these live docs -- what a search sees through a reader wrapper that filters fields and
/// documents and masks values (the security plugin's field- and document-level security and field
/// masking). Nothing is read from disk. `base` stays open and independent: closing either leaves
/// the other searchable.
///
/// `masked` holds a little-endian `u32` count of names, then each name as a little-endian `u32`
/// byte length and its UTF-8 bytes: the fields whose keyword and binary doc values the view shows
/// as `mask` makes them (a name a segment has no such field for is ignored). `mask` is called
/// whenever a masked value is shown, from the searching thread, and must stay valid until the
/// returned handle is closed; it may be null only when `masked` lists nothing.
///
/// `hidden` holds, for each of the `segment_count` segments in order, a little-endian `u32` count
/// of names, then each name as a little-endian `u32` byte length and its UTF-8 bytes; a name the
/// segment has no field for is ignored. `expected_max_docs`, `live_words` and `live_word_counts`
/// are [`ffi_open_jvm_reader`]'s.
///
/// # Safety
/// `hidden` must be valid for `hidden_len` bytes, `expected_max_docs` and (when non-null)
/// `live_word_counts` for `segment_count` elements each, `live_words` for the sum of
/// `live_word_counts` `u64`s, and `out_handle` for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_open_jvm_reader_view(
    base: u64,
    hidden: *const u8,
    hidden_len: usize,
    masked: *const u8,
    masked_len: usize,
    mask: Option<MaskFn>,
    expected_max_docs: *const i32,
    segment_count: usize,
    live_words: *const u64,
    live_word_counts: *const usize,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_handle.is_null() || (expected_max_docs.is_null() && segment_count > 0) {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let hidden_bytes = unsafe { bytes_from_raw(hidden, hidden_len)? };
        let hidden_sets = parse_hidden_fields(hidden_bytes, segment_count)?;
        // SAFETY: caller contract.
        let masked_bytes = unsafe { bytes_from_raw(masked, masked_len)? };
        let masks = parse_masks(masked_bytes, mask)?;
        let (expected, counts): (&[i32], Option<&[usize]>) = if segment_count == 0 {
            (&[], None)
        } else {
            // SAFETY: caller contract, non-null checked above / here.
            unsafe {
                (
                    std::slice::from_raw_parts(expected_max_docs, segment_count),
                    (!live_word_counts.is_null())
                        .then(|| std::slice::from_raw_parts(live_word_counts, segment_count)),
                )
            }
        };
        let total_words = counts
            .unwrap_or(&[])
            .iter()
            .try_fold(0usize, |acc, &c| acc.checked_add(c))
            .ok_or_else(|| {
                set_last_error("live-docs word counts overflow");
                FfiStatus::InvalidArgument
            })?;
        if total_words > 0 && live_words.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let words: &[u64] = if total_words == 0 {
            &[]
        } else {
            // SAFETY: caller contract: `live_words` holds the sum of the counts.
            unsafe { std::slice::from_raw_parts(live_words, total_words) }
        };
        let base = lookup(
            base,
            "ffi_open_jvm_reader_view: unknown or already-closed base handle",
        )?;
        let opened = lucene_engine::reader_handle::open_view(
            &base,
            &hidden_sets,
            &masks,
            expected,
            counts,
            words,
        )?;
        let handle = lock_recovering(jvm_readers()).insert_checked(Arc::new(opened))?;
        // SAFETY: caller contract.
        unsafe { *out_handle = handle };
        Ok(())
    })
}

/// [`ffi_open_jvm_reader_view`]'s `masked` blob: the names of the masked fields, each with `mask`.
fn parse_masks(
    bytes: &[u8],
    mask: Option<MaskFn>,
) -> Result<std::collections::HashMap<String, lucene_codecs::value_mask::SharedMask>, FfiStatus> {
    let bad = |what: &str| {
        set_last_error(format!("the masked-fields blob is malformed: {what}"));
        FfiStatus::InvalidArgument
    };
    let mut masks = std::collections::HashMap::new();
    if bytes.is_empty() {
        return Ok(masks);
    }
    let Some(f) = mask else {
        return Err(bad("fields to mask without a mask function"));
    };
    let u32_at = |at: &mut usize| -> Result<usize, FfiStatus> {
        let end = at.checked_add(4).ok_or_else(|| bad("offset"))?;
        let raw: [u8; 4] = bytes
            .get(*at..end)
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| bad("truncated"))?;
        *at = end;
        usize::try_from(u32::from_le_bytes(raw)).map_err(|_| bad("length"))
    };
    let mut at = 0usize;
    let n = u32_at(&mut at)?;
    for _ in 0..n {
        let len = u32_at(&mut at)?;
        let end = at.checked_add(len).ok_or_else(|| bad("name length"))?;
        let name = bytes.get(at..end).ok_or_else(|| bad("truncated name"))?;
        let text = std::str::from_utf8(name)
            .map_err(|_| bad("a name that is not UTF-8"))?
            .to_owned();
        masks.insert(
            text,
            std::sync::Arc::new(HostMask {
                f,
                name: name.to_vec(),
            }) as lucene_codecs::value_mask::SharedMask,
        );
        at = end;
    }
    if at != bytes.len() {
        return Err(bad("bytes after the last name"));
    }
    Ok(masks)
}

/// [`ffi_open_jvm_reader_view`]'s `hidden` blob: per segment, the names of its hidden fields.
fn parse_hidden_fields(
    bytes: &[u8],
    segment_count: usize,
) -> Result<Vec<std::collections::HashSet<String>>, FfiStatus> {
    fn bad(what: &str) -> FfiStatus {
        set_last_error(format!("the hidden-fields blob is malformed: {what}"));
        FfiStatus::InvalidArgument
    }
    fn u32_at(bytes: &[u8], at: &mut usize) -> Result<usize, FfiStatus> {
        let end = at.checked_add(4).ok_or_else(|| bad("offset"))?;
        let raw: [u8; 4] = bytes
            .get(*at..end)
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| bad("truncated"))?;
        *at = end;
        usize::try_from(u32::from_le_bytes(raw)).map_err(|_| bad("length"))
    }
    let mut at = 0usize;
    let mut sets = Vec::new();
    for _ in 0..segment_count {
        let n = u32_at(bytes, &mut at)?;
        let mut set = std::collections::HashSet::new();
        for _ in 0..n {
            let len = u32_at(bytes, &mut at)?;
            let end = at.checked_add(len).ok_or_else(|| bad("name length"))?;
            let name = bytes.get(at..end).ok_or_else(|| bad("truncated name"))?;
            set.insert(
                std::str::from_utf8(name)
                    .map_err(|_| bad("a name that is not UTF-8"))?
                    .to_owned(),
            );
            at = end;
        }
        sets.push(set);
    }
    if at != bytes.len() {
        return Err(bad("bytes after the last segment"));
    }
    Ok(sets)
}

/// A shared reference to an open handle, taken under a read lock held only
/// for the lookup.
pub(crate) fn lookup(handle: u64, missing: &str) -> Result<Arc<JvmReaderHandle>, FfiStatus> {
    lucene_engine::registry::lookup_jvm_reader(handle, missing)
}

/// Runs the query in the `query_len`-byte blob `query` ([`lucene_engine::query_blob::decode_query`])
/// against `handle`, writing up to `top_n` hits -- global doc ID and score,
/// best first, ties by ascending doc ID -- into `out_docs`/`out_scores` and
/// their number into `*out_hit_count`.
///
/// Total hits follow Lucene's `totalHitsThreshold` (OpenSearch's
/// `track_total_hits`): with `count_limit > 0`, `*out_total` is the number of
/// live matching documents, exact (and `*out_total_is_lower_bound` false)
/// whenever it is at most `count_limit`; once it exceeds `count_limit` it may
/// instead be any lower bound that itself exceeds `count_limit`, with
/// `*out_total_is_lower_bound` true -- Lucene's `GREATER_THAN_OR_EQUAL_TO`,
/// which `TopScoreDocCollector` switches to only when the count passes the
/// threshold, not when it reaches it.
/// `count_limit == i64::MAX` asks for an exact count. `count_limit <= 0`
/// counts nothing: `*out_total` is `-1`.
///
/// With `top_n > 0` the count comes from the search itself, exactly as
/// Lucene's collector keeps it; with `top_n == 0` see [`total_hits`].
///
/// # Safety
/// `query` must be valid for `query_len` bytes; `out_docs`/`out_scores` for
/// `buf_len` elements each, with `buf_len >= top_n`; `out_hit_count`,
/// `out_total` and `out_total_is_lower_bound` for one write each.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_jvm_reader_search(
    handle: u64,
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
) -> i32 {
    guard(|| {
        if out_hit_count.is_null() || out_total.is_null() || out_total_is_lower_bound.is_null() {
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
        let sim = sim.as_ref();
        // No lock is held while searching: a slow query must not block a
        // refresh's open or close, and through it every other search.
        let h = lookup(
            handle,
            "ffi_jvm_reader_search: unknown or already-closed handle",
        )?;
        let query = similarity_checked(sim, lucene_engine::join::resolve(&h, query, sim))?;
        let (hits, total, lower_bound) = similarity_checked(
            sim,
            match min_score {
                Some(min) => search_min_score(&h, &query, top_n, count_limit, min, sim),
                None => search(&h, &query, top_n, count_limit, sim),
            },
        )?;
        // SAFETY: caller contract; `hits.len() <= top_n <= buf_len`.
        unsafe {
            for (i, hit) in hits.iter().enumerate() {
                *out_docs.add(i) = hit.doc_id;
                *out_scores.add(i) = hit.score;
            }
            *out_hit_count = hits.len();
            *out_total = total;
            *out_total_is_lower_bound = lower_bound;
        }
        Ok(())
    })
}

/// `out_hit_count`, `out_total`, `out_total_is_lower_bound`, `out_terms_len`
/// and `out_max_score` (the tracked max score, `NaN` untracked) for one write
/// each.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_jvm_reader_search_sorted(
    handle: u64,
    query: *const u8,
    query_len: usize,
    sort: *const u8,
    sort_len: usize,
    top_n: usize,
    count_limit: i64,
    out_docs: *mut i32,
    out_values: *mut i64,
    buf_len: usize,
    out_terms: *mut u8,
    terms_cap: usize,
    out_hit_count: *mut usize,
    out_total: *mut i64,
    out_total_is_lower_bound: *mut bool,
    out_terms_len: *mut usize,
    out_max_score: *mut f32,
    out_terminated: *mut bool,
) -> i32 {
    guard(|| {
        if out_hit_count.is_null()
            || out_total.is_null()
            || out_total_is_lower_bound.is_null()
            || out_terms_len.is_null()
            || out_max_score.is_null()
            || out_terminated.is_null()
            || out_docs.is_null()
            || out_values.is_null()
            || (out_terms.is_null() && terms_cap != 0)
        {
            return Err(FfiStatus::NullPointer);
        }
        if top_n == 0 {
            set_last_error("ffi_jvm_reader_search_sorted: top_n must be at least 1");
            return Err(FfiStatus::InvalidArgument);
        }
        if buf_len < top_n {
            return Err(FfiStatus::BufferTooSmall);
        }
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(query, query_len)? };
        // SAFETY: caller contract.
        let sort_blob = unsafe { bytes_from_raw(sort, sort_len)? };
        let SortedOut {
            keys,
            hits,
            total,
            lower_bound,
            terms,
            max_score,
            terminated,
            ..
        } = search_sorted_blobs(handle, blob, sort_blob, top_n, count_limit)?;
        // SAFETY: caller contract.
        unsafe { *out_terms_len = terms.len() };
        if terms.len() > terms_cap {
            set_last_error(format!(
                "sort terms need {} bytes, the buffer holds {terms_cap}",
                terms.len()
            ));
            return Err(FfiStatus::BufferTooSmall);
        }
        // SAFETY: caller contract; `hits.len() <= top_n <= buf_len`, each hit
        // carries `keys.len()` values, and `terms` fits `terms_cap`.
        unsafe {
            if !terms.is_empty() {
                std::ptr::copy_nonoverlapping(terms.as_ptr(), out_terms, terms.len());
            }
            for (i, hit) in hits.iter().enumerate() {
                *out_docs.add(i) = hit.doc;
                for (k, &v) in hit.values.iter().enumerate() {
                    *out_values.add(i.wrapping_mul(keys).wrapping_add(k)) = v;
                }
            }
            *out_hit_count = hits.len();
            *out_total = total;
            *out_total_is_lower_bound = lower_bound;
            *out_max_score = max_score;
            *out_terminated = terminated;
        }
        Ok(())
    })
}

/// [`rescore_hits`] across the boundary: `docs`/`scores` hold `n` hits and
/// are rewritten in place.
///
/// # Safety
/// `query` must be valid for `query_len` bytes, `docs`/`scores` for `n`
/// reads and writes each.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_jvm_reader_rescore(
    handle: u64,
    query: *const u8,
    query_len: usize,
    window: i32,
    query_weight: f32,
    rescore_query_weight: f32,
    mode: i32,
    docs: *mut i32,
    scores: *mut f32,
    n: usize,
) -> i32 {
    guard(|| {
        if docs.is_null() || scores.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let window = usize::try_from(window).map_err(|_| {
            set_last_error(format!("rescore: window {window}"));
            FfiStatus::InvalidArgument
        })?;
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(query, query_len)? };
        // SAFETY: caller contract: `docs` and `scores` hold `n` slots each.
        let (docs, scores) = unsafe {
            (
                std::slice::from_raw_parts_mut(docs, n),
                std::slice::from_raw_parts_mut(scores, n),
            )
        };
        rescore_hits(
            handle,
            blob,
            window,
            query_weight,
            rescore_query_weight,
            mode,
            docs,
            scores,
        )?;
        Ok(())
    })
}

/// `out_counts` for `n` elements and `out_values` for `n *`
/// [`METRIC_VALUES`], `n` being at least the blob's field count times its
/// slice count (or times one without slices): the states come slice by
/// slice, each slice's fields in blob order; `out_terms` for `terms_cap`
/// bytes (null when 0), `out_terms_len` writable.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_jvm_reader_aggregate(
    handle: u64,
    query: *const u8,
    query_len: usize,
    aggs: *const u8,
    aggs_len: usize,
    out_counts: *mut i64,
    out_values: *mut f64,
    n: usize,
    out_terms: *mut u8,
    terms_cap: usize,
    out_terms_len: *mut usize,
) -> i32 {
    guard(|| {
        if out_counts.is_null()
            || out_values.is_null()
            || out_terms_len.is_null()
            || (out_terms.is_null() && terms_cap > 0)
        {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(query, query_len)? };
        // SAFETY: caller contract.
        let aggs_blob = unsafe { bytes_from_raw(aggs, aggs_len)? };
        let (states, terms) = aggregate_blobs(handle, blob, aggs_blob)?;
        // SAFETY: caller contract.
        unsafe { *out_terms_len = terms.len() };
        if n < states.len() || terms_cap < terms.len() {
            return Err(FfiStatus::BufferTooSmall);
        }
        if !terms.is_empty() {
            // SAFETY: `out_terms` is valid for `terms_cap >= terms.len()` bytes.
            unsafe { std::ptr::copy_nonoverlapping(terms.as_ptr(), out_terms, terms.len()) };
        }
        // SAFETY: caller contract; `states.len() <= n`.
        unsafe {
            for (i, s) in states.iter().enumerate() {
                *out_counts.add(i) = i64::try_from(s.count).unwrap_or(i64::MAX);
                for (j, v) in metric_values(s).into_iter().enumerate() {
                    *out_values.add(i.wrapping_mul(METRIC_VALUES).wrapping_add(j)) = v;
                }
            }
        }
        Ok(())
    })
}

/// Whether a concurrent `size: 0` search's count stops early
/// ([`lucene_search::terminate::count_terminates`]): the query blob, and a
/// count blob ([`decode_count_spec`]) of the limit, which segments the count
/// iterates, and the slices; `*out_terminated` receives the answer.
///
/// # Safety
/// `query`/`spec` must be valid for `query_len`/`spec_len` bytes and
/// `out_terminated` for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_reader_count_terminates(
    handle: u64,
    query: *const u8,
    query_len: usize,
    spec: *const u8,
    spec_len: usize,
    out_terminated: *mut bool,
) -> i32 {
    guard(|| {
        if out_terminated.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(query, query_len)? };
        // SAFETY: caller contract.
        let spec_blob = unsafe { bytes_from_raw(spec, spec_len)? };
        let terminated = count_terminates_blobs(handle, blob, spec_blob)?;
        // SAFETY: caller contract.
        unsafe { *out_terminated = terminated };
        Ok(())
    })
}

/// Closes a JVM reader handle. Segments a later handle reused stay open
/// until that handle closes too.
#[no_mangle]
pub extern "C" fn ffi_close_jvm_reader(handle: u64) -> i32 {
    guard(|| lucene_engine::registry::close_jvm_reader(handle))
}

/// [`lucene_engine::reader_search::search_sorted_blobs`] over `handle`'s reader.
pub(crate) fn search_sorted_blobs(
    handle: u64,
    query_blob: &[u8],
    sort_blob: &[u8],
    top_n: usize,
    count_limit: i64,
) -> Result<SortedOut, FfiStatus> {
    let h = lookup(
        handle,
        "ffi_jvm_reader_search_sorted: unknown or already-closed handle",
    )?;
    lucene_engine::reader_search::search_sorted_blobs(&h, query_blob, sort_blob, top_n, count_limit)
}

/// [`lucene_engine::reader_search::search_collapsed_blobs`] over `handle`'s reader.
pub(crate) fn search_collapsed_blobs(
    handle: u64,
    query_blob: &[u8],
    sort_blob: &[u8],
    collapse_blob: &[u8],
    top_n: usize,
) -> Result<Vec<u8>, FfiStatus> {
    let h = lookup(
        handle,
        "ffi_jvm_reader_search_collapsed: unknown or already-closed handle",
    )?;
    lucene_engine::reader_search::search_collapsed_blobs(
        &h,
        query_blob,
        sort_blob,
        collapse_blob,
        top_n,
    )
}

/// [`lucene_engine::reader_search::rescore_hits`] over `handle`'s reader.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rescore_hits(
    handle: u64,
    query_blob: &[u8],
    window: usize,
    query_weight: f32,
    rescore_query_weight: f32,
    mode: i32,
    docs: &mut [i32],
    scores: &mut [f32],
) -> Result<usize, FfiStatus> {
    let h = lookup(
        handle,
        "ffi_jvm_reader_rescore: unknown or already-closed handle",
    )?;
    lucene_engine::reader_search::rescore_hits(
        &h,
        query_blob,
        window,
        query_weight,
        rescore_query_weight,
        mode,
        docs,
        scores,
    )
}

/// [`lucene_engine::reader_search::aggregate_blobs`] over `handle`'s reader.
pub(crate) fn aggregate_blobs(
    handle: u64,
    query_blob: &[u8],
    aggs_blob: &[u8],
) -> Result<(Vec<MetricState>, Vec<u8>), FfiStatus> {
    let h = lookup(
        handle,
        "ffi_jvm_reader_aggregate: unknown or already-closed handle",
    )?;
    lucene_engine::reader_search::aggregate_blobs(&h, query_blob, aggs_blob)
}

/// [`lucene_engine::reader_search::aggregate_counting_blobs`] over `handle`'s reader.
#[cfg(test)]
#[allow(clippy::type_complexity)]
pub(crate) fn aggregate_counting_blobs(
    handle: u64,
    query_blob: &[u8],
    aggs_blob: &[u8],
    count_limit: i64,
) -> Result<(Vec<MetricState>, Vec<u8>, Option<(i64, bool)>), FfiStatus> {
    let h = lookup(
        handle,
        "ffi_jvm_reader_aggregate: unknown or already-closed handle",
    )?;
    lucene_engine::reader_search::aggregate_counting_blobs(&h, query_blob, aggs_blob, count_limit)
}

/// [`lucene_engine::reader_search::aggregate_counting_blobs_seen`] over `handle`'s reader.
#[allow(clippy::type_complexity)]
pub(crate) fn aggregate_counting_blobs_seen(
    handle: u64,
    query_blob: &[u8],
    aggs_blob: &[u8],
    count_limit: i64,
) -> Result<
    (
        Vec<MetricState>,
        Vec<u8>,
        Option<(i64, bool)>,
        Vec<Option<u64>>,
    ),
    FfiStatus,
> {
    let h = lookup(
        handle,
        "ffi_jvm_reader_aggregate: unknown or already-closed handle",
    )?;
    lucene_engine::reader_search::aggregate_counting_blobs_seen(
        &h,
        query_blob,
        aggs_blob,
        count_limit,
    )
}

/// [`lucene_engine::reader_search::doc_freq`] over `handle`'s reader.
pub(crate) fn doc_freq(handle: u64, field: &[u8], term: &[u8]) -> Result<i64, FfiStatus> {
    let h = lookup(handle, "doc_freq: unknown or already-closed handle")?;
    lucene_engine::reader_search::doc_freq(&h, field, term)
}

/// [`lucene_engine::reader_search::count_terminates_blobs`] over `handle`'s reader.
pub(crate) fn count_terminates_blobs(
    handle: u64,
    query_blob: &[u8],
    spec_blob: &[u8],
) -> Result<bool, FfiStatus> {
    let h = lookup(
        handle,
        "ffi_jvm_reader_count_terminates: unknown or already-closed handle",
    )?;
    lucene_engine::reader_search::count_terminates_blobs(&h, query_blob, spec_blob)
}

/// [`lucene_engine::reader_search::search_hybrid_blobs`] over `handle`'s reader.
pub(crate) fn search_hybrid_blobs(handle: u64, blob: &[u8]) -> Result<Vec<u8>, FfiStatus> {
    let h = lookup(
        handle,
        "ffi_jvm_reader_search_hybrid: unknown or already-closed handle",
    )?;
    lucene_engine::reader_search::search_hybrid_blobs(&h, blob)
}

/// [`lucene_engine::profile::profile_blob`] over `handle`'s reader.
pub(crate) fn profile_blobs(
    handle: u64,
    blob: &[u8],
    collect: u8,
    top_n: usize,
    count_limit: i64,
) -> Result<Vec<u8>, FfiStatus> {
    let h = lookup(
        handle,
        "ffi_jvm_reader_profile: unknown or already-closed handle",
    )?;
    lucene_engine::profile::profile_blob(&h, blob, collect, top_n, count_limit)
}

#[cfg(test)]
pub(crate) mod tests;
