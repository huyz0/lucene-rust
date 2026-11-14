//! The reader handle the OpenSearch plugin searches, over
//! [`lucene_engine::reader_search`]: opening a handle from the JVM reader's
//! segment list and live docs, closing it, and the `extern "C"` search entry
//! points. The decoders and the searches are in the engine crate; the
//! functions below that take a `u64` resolve the handle and call them.

use std::os::raw::c_char;

use lucene_aggs::metric_aggs::MetricState;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::Directory;
use lucene_util::fixed_bit_set::{bits2words, FixedBitSet};

use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::{bytes_from_raw, str_from_raw, try_with_capacity};
use crate::registry::{jvm_readers, lock_recovering, read_recovering, JvmReaderHandle};
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

        let segment_infos = lucene_index::segment_infos::parse(infos, generation).map_err(|e| {
            set_last_error(format!("parsing the JVM reader's SegmentInfos: {e}"));
            FfiStatus::Decode
        })?;
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
        let dir: &dyn Directory = &*dir;
        let reader = if previous == 0 {
            DirectoryReader::open_at(dir, segment_infos)
        } else {
            // Cloned out under a short read lock: the reopen reads files and
            // must not hold up every other search and close on the node.
            let prev = lookup(
                previous,
                "ffi_open_jvm_reader: unknown or already-closed previous handle",
            )?;
            prev.reader.reopen_at(dir, segment_infos)
        }
        .map_err(|e| {
            set_last_error(format!("opening the JVM reader's segments: {e}"));
            FfiStatus::Decode
        })?;

        let opened: Vec<i32> = reader.segment_readers().iter().map(|s| s.max_doc).collect();
        if opened != expected {
            set_last_error(format!(
                "segment maxDocs {opened:?} do not match the JVM reader's {expected:?}"
            ));
            return Err(FfiStatus::InvalidArgument);
        }
        let mut live_docs = try_with_capacity(opened.len())?;
        let mut deleted = try_with_capacity(opened.len())?;
        let mut at = 0usize;
        for (segment, &max_doc) in opened.iter().enumerate() {
            let n = counts.map_or(0, |c| c[segment]);
            let live = live_from_words(segment, max_doc, &words[at..at + n])?;
            at += n;
            deleted.push(live.as_ref().map_or(0, |l| {
                (usize::try_from(max_doc).unwrap_or(0) - l.cardinality()) as i64
            }));
            live_docs.push(live);
        }
        let cacheable = live_docs.len();
        let handle = lock_recovering(jvm_readers()).insert_checked(Arc::new(JvmReaderHandle {
            reader,
            live_docs,
            deleted,
            commit_live: std::sync::Mutex::new(None),
            cacheable,
            agg_identities: std::sync::OnceLock::new(),
            path: std::sync::Arc::from(path),
        }))?;
        // SAFETY: caller contract.
        unsafe { *out_handle = handle };
        Ok(())
    })
}

/// A shared reference to an open handle, taken under a read lock held only
/// for the lookup.
pub(crate) fn lookup(handle: u64, missing: &str) -> Result<Arc<JvmReaderHandle>, FfiStatus> {
    let reader = read_recovering(jvm_readers())
        .get(handle)
        .cloned()
        .ok_or_else(|| {
            set_last_error(missing);
            FfiStatus::InvalidHandle
        })?;
    crate::error::note_shard(&reader.path);
    Ok(reader)
}

/// Segment `segment`'s live docs from its `words` ([`ffi_open_jvm_reader`]'s
/// contract): `None` for no words, else exactly `bits2words(max_doc)` words
/// with no bit past `max_doc`.
fn live_from_words(
    segment: usize,
    max_doc: i32,
    words: &[u64],
) -> Result<Option<FixedBitSet>, FfiStatus> {
    if words.is_empty() {
        return Ok(None);
    }
    let max_doc = usize::try_from(max_doc).unwrap_or(0);
    if words.len() != bits2words(max_doc) {
        set_last_error(format!(
            "segment {segment}: {} live-docs words for maxDoc {max_doc}, expected {}",
            words.len(),
            bits2words(max_doc)
        ));
        return Err(FfiStatus::InvalidArgument);
    }
    let tail_bits = max_doc % 64;
    if tail_bits != 0 && words[words.len() - 1] >> tail_bits != 0 {
        set_last_error(format!(
            "segment {segment}: live-docs bits set past maxDoc {max_doc}"
        ));
        return Err(FfiStatus::InvalidArgument);
    }
    let mut owned = try_with_capacity::<u64>(words.len())?;
    owned.extend_from_slice(words);
    Ok(Some(FixedBitSet::from_words(owned, max_doc)))
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
                    *out_values.add(i * keys + k) = v;
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
                    *out_values.add(i * METRIC_VALUES + j) = v;
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
    guard(|| {
        let mut readers = lock_recovering(jvm_readers());
        readers.remove(handle).ok_or_else(|| {
            set_last_error("ffi_close_jvm_reader: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })?;
        // The nested queries' parent bit sets of segments no open reader
        // holds any more go with them.
        let live = readers
            .values()
            .flat_map(|h| h.reader.segment_readers())
            .map(|r| r.segment_id())
            .collect();
        drop(readers);
        lucene_engine::nodes::retain_parent_sets(&live);
        Ok(())
    })
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

#[cfg(test)]
pub(crate) mod tests {
    #![allow(unused_imports)]
    use std::os::raw::c_char;

    use lucene_aggs::metric_aggs::{MetricSpec, MetricState, Source, ValueKind, NEED_ALL};
    use lucene_aggs::terms_agg::TermsSpec;
    use lucene_search::directory_reader::DirectoryReader;
    use lucene_search::field_norms::FieldNorms;
    use lucene_search::multi_segment::OpenSegment;
    use lucene_search::query::{BooleanQuery, Clause, TermQuery};
    use lucene_search::similarities::{PerFieldSimilarity, Similarity};
    use lucene_search::top_field::{
        CollapseKind, CollapseTopDocs, CollapseValue, FieldDoc, NestedSort, Selector, SortField,
        SortType,
    };
    use lucene_search::weight_count::count_term_query;
    use lucene_search::{
        count_boolean_query_segment_with, search_boolean_query_multi_segment_maxscore_counting,
        search_term_query_multi_segment_counting, ScoreDoc, TotalHits, TotalHitsRelation,
    };
    use lucene_store::Directory;
    use lucene_util::fixed_bit_set::{bits2words, FixedBitSet};

    use crate::error::{set_last_error, FfiStatus};
    use crate::raw::{bytes_from_raw, str_from_raw, try_with_capacity};
    use crate::registry::{jvm_readers, lock_recovering, read_recovering, JvmReaderHandle};
    use lucene_engine::clauses::map_search_error;
    use lucene_engine::query_blob::{
        boolean_of, decode_query, decode_request, invalid, query_uses_points, similarity_checked,
        Cursor, JvmQuery,
    };
    use std::sync::Arc;

    use super::*;
    use lucene_engine::query_blob::{
        GEOMETRY_CIRCLE, GEOMETRY_LINE, GEOMETRY_POINT, GEOMETRY_POLYGON, GEOMETRY_RECTANGLE,
        NODE_BOOLEAN, NODE_BOOST, NODE_CONSTANT_SCORE, NODE_DISMAX, NODE_EXISTS, NODE_GEO_DISTANCE,
        NODE_GEO_POINT, NODE_GEO_SHAPE, NODE_GEO_SHAPE_BOX, NODE_MATCH_ALL, NODE_MATCH_NONE,
        NODE_PHRASE, NODE_POINTS_BOX, NODE_POINT_RANGE, NODE_POINT_SET, NODE_PREFIX, NODE_REGEXP,
        NODE_TERM, NODE_TERM_RANGE, NODE_TERM_SET, NODE_TERM_STATS, NODE_WILDCARD, QUERY_BOOLEAN,
        QUERY_MIN_SCORE, QUERY_TERM, QUERY_TREE,
    };

    /// A real two-segment Java-written index whose `manifest.properties`
    /// records Lucene's own top hits and scores (`GenMultiSegmentScoring`).
    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/multi_segment_scoring_index"
    );

    fn infos() -> Vec<u8> {
        std::fs::read(format!("{FIXTURE}/segments_2")).expect("segments_2")
    }

    pub(crate) fn term_blob(field: &str, term: &str) -> Vec<u8> {
        let mut b = vec![QUERY_TERM];
        for part in [field.as_bytes(), term.as_bytes()] {
            b.extend_from_slice(&(part.len() as i32).to_le_bytes());
            b.extend_from_slice(part);
        }
        b
    }

    /// `(occur, kind, parent, param, field, term)` clauses under a root with
    /// `msm`.
    fn bool_blob(msm: i32, clauses: &[(u8, u8, i32, i32, &str, &str)]) -> Vec<u8> {
        let mut b = vec![QUERY_BOOLEAN];
        b.extend_from_slice(&msm.to_le_bytes());
        b.extend_from_slice(&(clauses.len() as i32).to_le_bytes());
        for &(occur, kind, parent, param, field, term) in clauses {
            b.push(occur);
            b.push(kind);
            b.extend_from_slice(&parent.to_le_bytes());
            b.extend_from_slice(&param.to_le_bytes());
            for part in [field.as_bytes(), term.as_bytes()] {
                b.extend_from_slice(&(part.len() as i32).to_le_bytes());
                b.extend_from_slice(part);
            }
        }
        b
    }

    fn open_with(max_docs: &[i32], previous: u64) -> (i32, u64) {
        open_live(max_docs, previous, &[])
    }

    /// Opens the fixture with `live[i]` as segment `i`'s live-docs words
    /// (missing or empty: no deletions).
    pub(crate) fn open_live(max_docs: &[i32], previous: u64, live: &[&[u64]]) -> (i32, u64) {
        let infos = infos();
        let counts: Vec<usize> = (0..max_docs.len())
            .map(|i| live.get(i).map_or(0, |w| w.len()))
            .collect();
        let words: Vec<u64> = live.iter().flat_map(|w| w.iter().copied()).collect();
        let mut handle = 0u64;
        let rc = unsafe {
            ffi_open_jvm_reader(
                FIXTURE.as_ptr().cast(),
                FIXTURE.len(),
                infos.as_ptr(),
                infos.len(),
                2,
                previous,
                max_docs.as_ptr(),
                max_docs.len(),
                words.as_ptr(),
                counts.as_ptr(),
                &mut handle,
            )
        };
        (rc, handle)
    }

    pub(crate) fn open() -> u64 {
        let (rc, handle) = open_with(&[4, 4], 0);
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        handle
    }

    /// Opens `fixtures/data/bwc/<version>` the way the plugin opens a
    /// searcher's reader: its `segments_N` bytes, every segment's `maxDoc`
    /// and its live docs as words.
    pub(crate) fn open_bwc(version: &str) -> u64 {
        open_fixture(&format!("bwc/{version}"))
    }

    /// Opens `fixtures/data/<rel>` as [`open_bwc`] opens a version.
    pub(crate) fn open_fixture(rel: &str) -> u64 {
        let path = format!("{}/../../fixtures/data/{rel}", env!("CARGO_MANIFEST_DIR"));
        let dir = lucene_store::FsDirectory::open(&path);
        let files: Vec<String> = std::fs::read_dir(&path)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        let generation = lucene_store::directory::last_commit_generation(&files).unwrap();
        let infos = std::fs::read(format!(
            "{path}/{}",
            lucene_store::directory::segments_file_name(generation).unwrap()
        ))
        .unwrap();
        let reader = DirectoryReader::open(&dir).unwrap();
        let max_docs: Vec<i32> = reader.segment_readers().iter().map(|s| s.max_doc).collect();
        let live: Vec<Vec<u64>> = reader
            .segment_readers()
            .iter()
            .map(|s| s.live_docs().map_or(Vec::new(), |l| l.words().to_vec()))
            .collect();
        let counts: Vec<usize> = live.iter().map(Vec::len).collect();
        let words: Vec<u64> = live.concat();
        let mut handle = 0u64;
        let rc = unsafe {
            ffi_open_jvm_reader(
                path.as_ptr().cast(),
                path.len(),
                infos.as_ptr(),
                infos.len(),
                generation,
                0,
                max_docs.as_ptr(),
                max_docs.len(),
                words.as_ptr(),
                counts.as_ptr(),
                &mut handle,
            )
        };
        assert_eq!(rc, 0, "{rel}: {}", crate::error::last_error());
        handle
    }

    /// An index Lucene 9.0-10.3 wrote -- every retired postings,
    /// term-dictionary, points and segment-info generation -- opens through
    /// the plugin's own entry point and answers term and boolean queries
    /// with the current codec's (10.4.0's) hits, scores and totals -- the
    /// fixture writes the same documents in every version.
    /// The plugin's `NativeReaders` no longer falls back to Lucene for these
    /// postings formats; this is the native half of that.
    #[test]
    fn every_bwc_version_is_served_natively_like_the_current_codec() {
        let queries = [
            term_blob("body", "alpha"),
            term_blob("docs", "zeta"),
            bool_blob(
                0,
                &[
                    (
                        lucene_engine::clauses::OCCUR_SHOULD,
                        lucene_engine::clauses::CLAUSE_KIND_TERM,
                        -1,
                        0,
                        "body",
                        "alpha",
                    ),
                    (
                        lucene_engine::clauses::OCCUR_SHOULD,
                        lucene_engine::clauses::CLAUSE_KIND_TERM,
                        -1,
                        0,
                        "title",
                        "delta",
                    ),
                ],
            ),
        ];
        let answer = |version: &str| {
            let h = open_bwc(version);
            let out: Vec<_> = queries
                .iter()
                .map(|q| run(h, q, 10, true).unwrap_or_else(|rc| panic!("{version}: {rc}")))
                .collect();
            assert_eq!(ffi_close_jvm_reader(h), 0);
            out
        };
        let reference = answer("10.4.0");
        assert!(reference
            .iter()
            .all(|(hits, total)| !hits.is_empty() && *total > 0));
        for version in [
            "9.0.0", "9.1.0", "9.3.0", "9.4.2", "9.8.0", "9.11.1", "9.12.2", "10.0.0", "10.2.2",
        ] {
            assert_eq!(answer(version), reference, "{version}");
        }
    }

    /// An index whose fields are written in the completion
    /// postings formats (OpenSearch's `completion` fields; `Completion104`,
    /// and 9.0-10.3's under `bwc-completion`) opens through the plugin's entry
    /// point and answers queries on its other fields, deletions applied --
    /// the plugin's `NativeReaders` no longer falls back for these formats.
    #[test]
    fn completion_formats_are_served_natively() {
        let answer = |rel: &str| {
            let h = open_fixture(rel);
            let mut out = Vec::new();
            for id in ["0", "5", "41", "44", "89", "nosuch"] {
                out.push(run(h, &term_blob("id", id), 10, true).unwrap());
            }
            let (_, live) = run(h, &tree(N::All), 0, true).unwrap();
            // The completion field's own terms and postings: every live
            // document holds a `suggest` value.
            let (_, suggest) =
                run(h, &tree(N::Tr("suggest", None, None, true, true)), 0, true).unwrap();
            assert_eq!(suggest, live, "{rel}");
            assert_eq!(ffi_close_jvm_reader(h), 0);
            (out, live)
        };
        let (current, live) = answer("completion_query/index");
        // 90 documents, ten of them deleted (41, 44, ..., 68).
        assert_eq!(live, 80);
        let hits = |i: usize| current[i].0.len();
        assert_eq!(
            (hits(0), hits(1), hits(2), hits(3), hits(4), hits(5)),
            (1, 1, 0, 0, 1, 0)
        );
        for version in ["9.0.0", "9.11.1", "9.12.2", "10.2.2", "10.3.2"] {
            let (got, got_live) = answer(&format!("bwc-completion/{version}/index"));
            assert_eq!(got_live, live, "{version}");
            let found: Vec<usize> = got.iter().map(|(h, _)| h.len()).collect();
            assert_eq!(found, vec![1, 1, 0, 0, 1, 0], "{version}");
        }
    }

    /// `count` true is an exact count (`i64::MAX`), false none.
    fn run(
        handle: u64,
        blob: &[u8],
        top_n: usize,
        count: bool,
    ) -> Result<(Vec<(i32, f32)>, i64), i32> {
        let limit = if count { i64::MAX } else { 0 };
        run_limit(handle, blob, top_n, limit).map(|(h, t, _)| (h, t))
    }

    /// `(hits as (doc, score), total, total_is_lower_bound)`, or the status.
    pub(crate) type Searched = (Vec<(i32, f32)>, i64, bool);

    pub(crate) fn run_limit(
        handle: u64,
        blob: &[u8],
        top_n: usize,
        limit: i64,
    ) -> Result<Searched, i32> {
        let mut docs = vec![0i32; top_n];
        let mut scores = vec![0f32; top_n];
        let (mut n, mut total, mut lower) = (0usize, 0i64, false);
        let rc = unsafe {
            ffi_jvm_reader_search(
                handle,
                blob.as_ptr(),
                blob.len(),
                top_n,
                limit,
                docs.as_mut_ptr(),
                scores.as_mut_ptr(),
                top_n,
                &mut n,
                &mut total,
                &mut lower,
            )
        };
        if rc != 0 {
            return Err(rc);
        }
        Ok((
            docs[..n]
                .iter()
                .copied()
                .zip(scores[..n].iter().copied())
                .collect(),
            total,
            lower,
        ))
    }

    /// A handle that is certainly invalid: opened and closed. Its slot's
    /// generation has moved on, so no reader a parallel test opens can make
    /// it valid again -- unlike a fabricated value, which may land on one.
    fn closed_handle() -> u64 {
        let h = open();
        assert_eq!(ffi_close_jvm_reader(h), 0);
        h
    }

    /// The fixture with global doc 4 (segment 1, local 0, which holds fox)
    /// deleted.
    fn open_doc4_deleted() -> u64 {
        let (rc, h) = open_live(&[4, 4], 0, &[&[], &[0b1110]]);
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        h
    }

    /// A sort blob: `(type, flags, field, missing)` keys and an optional
    /// `(doc, values)` search-after.
    fn sort_blob(keys: &[(u8, u8, &str, i64)], after: Option<(i32, &[i64])>) -> Vec<u8> {
        let terms = vec![None; keys.len()];
        sort_blob_terms(keys, after.map(|(d, v)| (d, v, &terms[..])))
    }

    /// [`sort_blob`] with a search-after term per keyword key.
    #[allow(clippy::type_complexity)]
    pub(crate) fn sort_blob_terms(
        keys: &[(u8, u8, &str, i64)],
        after: Option<(i32, &[i64], &[Option<&[u8]>])>,
    ) -> Vec<u8> {
        let mut b = vec![keys.len() as u8];
        for &(ty, flags, field, missing) in keys {
            b.push(ty);
            b.push(flags);
            if ty != SORT_SCORE && ty != SORT_DOC {
                b.extend_from_slice(&(field.len() as i32).to_le_bytes());
                b.extend_from_slice(field.as_bytes());
                b.extend_from_slice(&missing.to_le_bytes());
            }
        }
        match after {
            None => b.push(0),
            Some((doc, values, terms)) => {
                b.push(1);
                b.extend_from_slice(&doc.to_le_bytes());
                for (i, v) in values.iter().enumerate() {
                    if keys[i].0 != SORT_STRING {
                        b.extend_from_slice(&v.to_le_bytes());
                        continue;
                    }
                    match terms[i] {
                        None => b.push(0),
                        Some(t) => {
                            b.push(1);
                            b.extend_from_slice(&(t.len() as i32).to_le_bytes());
                            b.extend_from_slice(t);
                        }
                    }
                }
            }
        }
        b.push(0); // options
        b.extend_from_slice(&0i32.to_le_bytes()); // no slices
        b
    }

    /// A one-key sort blob of a [`SORT_GEO_DISTANCE`] key, then `after`.
    fn geo_sort_blob(
        flags: u8,
        comparator: u8,
        origins: &[(f64, f64)],
        unit: f64,
        after: Option<i64>,
    ) -> Vec<u8> {
        let mut b = vec![1, SORT_GEO_DISTANCE, flags];
        b.extend_from_slice(&3i32.to_le_bytes());
        b.extend_from_slice(b"loc");
        b.extend_from_slice(&0i64.to_le_bytes());
        b.push(comparator);
        b.extend_from_slice(&(origins.len() as i32).to_le_bytes());
        for &(lat, lon) in origins {
            b.extend_from_slice(&lat.to_bits().to_le_bytes());
            b.extend_from_slice(&lon.to_bits().to_le_bytes());
        }
        b.extend_from_slice(&unit.to_bits().to_le_bytes());
        match after {
            None => b.push(0),
            Some(v) => {
                b.push(1);
                b.extend_from_slice(&7i32.to_le_bytes());
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        b.push(0); // options
        b.extend_from_slice(&0i32.to_le_bytes()); // no slices
        b
    }

    /// The geo-distance sort key: Lucene's comparator (one origin, metres,
    /// ascending, no `search_after`) and OpenSearch's (every mode, order and
    /// unit, `search_after` too); malformed keys and every truncation are
    /// errors; and a search sorted by one runs, a document without the field
    /// at `+Infinity`.
    #[test]
    fn decode_geo_sort_keys() {
        let invalid = Err(FfiStatus::InvalidArgument);
        let status = |b: &[u8]| decode_sort(b).map(|_| ());
        let o = [(10.0, 20.0)];
        let lucene = geo_sort_blob(0, 0, &o, 1.0, None);
        let (keys, after, ..) = decode_sort(&lucene).unwrap();
        assert!(matches!(keys[0].ty, SortType::Custom(_)) && !keys[0].reverse && after.is_none());
        for cut in 1..lucene.len() {
            assert!(decode_sort(&lucene[..cut]).is_err(), "cut at {cut}");
        }
        assert_eq!(
            status(&geo_sort_blob(0, 0, &o, 1.0, Some(5))),
            invalid,
            "Lucene's key and search_after"
        );
        assert_eq!(
            status(&geo_sort_blob(SORT_REVERSE, 0, &o, 1.0, None)),
            invalid,
            "Lucene's key reversed"
        );
        assert_eq!(
            status(&geo_sort_blob(0, 0, &[(0.0, 0.0), (1.0, 1.0)], 1.0, None)),
            invalid,
            "two origins"
        );
        assert_eq!(
            status(&geo_sort_blob(0, 0, &o, 1000.0, None)),
            invalid,
            "kilometres"
        );
        assert_eq!(
            status(&geo_sort_blob(0, 0, &[(95.0, 0.0)], 1.0, None)),
            invalid,
            "a latitude past the pole"
        );
        for flags in [
            0,
            SORT_MAX,
            SORT_REVERSE,
            SORT_MODE_SUM,
            SORT_MODE_AVG | SORT_REVERSE,
            SORT_MODE_MEDIAN,
        ] {
            let b = geo_sort_blob(flags, 1, &[(95.0, 0.0), (1.0, 1.0)], 1609.344, Some(42));
            let (keys, after, ..) = decode_sort(&b).unwrap();
            assert_eq!(keys[0].reverse, flags & SORT_REVERSE != 0);
            assert_eq!(after.unwrap().values, vec![42]);
        }
        assert_eq!(
            status(&geo_sort_blob(SORT_MAX | SORT_MODE_SUM, 1, &o, 1.0, None)),
            invalid,
            "max and a mode"
        );
        assert_eq!(
            status(&geo_sort_blob(SORT_NESTED, 1, &o, 1.0, None)),
            invalid,
            "nested"
        );
        assert_eq!(
            status(&geo_sort_blob(0, 1, &[], 1.0, None)),
            invalid,
            "no origins"
        );
        let many = vec![(0.0, 0.0); MAX_GEO_ORIGINS + 1];
        assert_eq!(
            status(&geo_sort_blob(0, 1, &many, 1.0, None)),
            invalid,
            "too many origins"
        );
        assert_eq!(
            status(&geo_sort_blob(0, 1, &[(f64::NAN, 0.0)], 1.0, None)),
            invalid,
            "NaN origin"
        );
        assert_eq!(
            status(&geo_sort_blob(0, 1, &[(0.0, f64::INFINITY)], 1.0, None)),
            invalid,
            "infinite origin"
        );
        assert_eq!(
            status(&geo_sort_blob(0, 1, &o, 0.0, None)),
            invalid,
            "a unit of 0 m"
        );
        assert_eq!(
            status(&geo_sort_blob(0, 2, &o, 1.0, None)),
            invalid,
            "unknown comparator"
        );
        let mut bad_utf8 = geo_sort_blob(0, 1, &o, 1.0, None);
        bad_utf8[7] = 0xff;
        assert_eq!(
            decode_sort(&bad_utf8).map(|_| ()),
            Err(FfiStatus::InvalidUtf8)
        );

        // Sorted by a field the index lacks: every hit at +Infinity, by doc.
        let h = open();
        let fox = term_blob("body", "fox");
        let inf = lucene_facet::double_to_sortable_long(f64::INFINITY);
        for blob in [
            lucene,
            geo_sort_blob(SORT_REVERSE | SORT_MODE_AVG, 1, &o, 1000.0, None),
        ] {
            let (hits, total, _) = run_sorted(h, &fox, &blob, 3, i64::MAX).unwrap();
            assert!(total > 0 && !hits.is_empty());
            assert!(hits.iter().all(|(_, v)| v == &vec![inf]), "{hits:?}");
            assert!(hits.windows(2).all(|w| w[0].0 < w[1].0), "{hits:?}");
        }
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    /// `(hits as (doc, values), total, lower_bound)`, or the status.
    pub(crate) type SortedRun = (Vec<(i32, Vec<i64>)>, i64, bool);

    pub(crate) fn run_sorted(
        handle: u64,
        query: &[u8],
        sort: &[u8],
        top_n: usize,
        limit: i64,
    ) -> Result<SortedRun, i32> {
        let keys = usize::from(sort.first().copied().unwrap_or(0));
        let mut docs = vec![0i32; top_n.max(1)];
        let mut values = vec![0i64; top_n.max(1) * keys.max(1)];
        let (mut n, mut total, mut lower) = (0usize, 0i64, false);
        let mut terms = vec![0u8; 1 << 16];
        let mut terms_len = 0usize;
        let mut max_score = 0f32;
        let rc = unsafe {
            ffi_jvm_reader_search_sorted(
                handle,
                query.as_ptr(),
                query.len(),
                sort.as_ptr(),
                sort.len(),
                top_n,
                limit,
                docs.as_mut_ptr(),
                values.as_mut_ptr(),
                top_n,
                terms.as_mut_ptr(),
                terms.len(),
                &mut n,
                &mut total,
                &mut lower,
                &mut terms_len,
                &mut max_score,
                &mut false,
            )
        };
        if rc != 0 {
            return Err(rc);
        }
        Ok((
            (0..n)
                .map(|i| (docs[i], values[i * keys..(i + 1) * keys].to_vec()))
                .collect(),
            total,
            lower,
        ))
    }

    /// A hybrid blob over `subs` keeping `top_n`, with `slices` (none = one).
    fn hybrid_blob(subs: &[Vec<u8>], top_n: i32, slices: &[&[i32]]) -> Vec<u8> {
        let mut b = vec![subs.len() as u8];
        for q in subs {
            b.extend_from_slice(&(q.len() as i32).to_le_bytes());
            b.extend_from_slice(q);
        }
        b.extend_from_slice(&top_n.to_le_bytes());
        b.extend_from_slice(&(slices.len() as i32).to_le_bytes());
        for slice in slices {
            b.extend_from_slice(&(slice.len() as i32).to_le_bytes());
            for s in *slice {
                b.extend_from_slice(&s.to_le_bytes());
            }
        }
        b
    }

    /// One slice of a hybrid answer: its total and each sub-query's
    /// `(doc, score)` hits.
    type HybridSlice = (i64, Vec<Vec<(i32, f32)>>);

    /// Every slice of a hybrid answer.
    fn decode_hybrid(out: &[u8], subs: usize) -> Vec<HybridSlice> {
        let mut c = Cursor::new(out);
        let slices = c.i32().unwrap();
        let mut res = Vec::new();
        for _ in 0..slices {
            let total = c.i64().unwrap();
            let mut per = Vec::new();
            for _ in 0..subs {
                let n = c.i32().unwrap();
                per.push(
                    (0..n)
                        .map(|_| (c.i32().unwrap(), f32::from_bits(c.i32().unwrap() as u32)))
                        .collect(),
                );
            }
            res.push((total, per));
        }
        assert_eq!(c.remaining(), 0);
        res
    }

    #[test]
    fn the_hybrid_collector_counts_only_what_beats_each_windows_threshold() {
        let sd = |doc_id, score| lucene_search::collector::ScoreDoc { doc_id, score };
        // Sub-query 0 scores across two windows; sub-query 1 only matches,
        // scoring 0 (a filter), which stock never collects.
        let scored = vec![
            vec![
                sd(0, 1.0),
                sd(1, 2.0),
                sd(4096, 1.5),
                sd(4097, 3.0),
                sd(4098, 0.5),
            ],
            vec![sd(2, 0.0)],
        ];
        let mut c = HybridCollector::new(2, 1);
        c.segment(&scored, 10);
        // Window 0: docs 0 and 1 (threshold 0), 1 pushes 0 out (threshold
        // 1.0). Window 1: 4096 and 4097 beat 1.0, 4098 does not.
        assert_eq!(c.total, 4);
        let hits: Vec<(i32, f32)> = c
            .queues
            .remove(0)
            .into_sorted()
            .iter()
            .map(|h| (h.doc, h.score))
            .collect();
        assert_eq!(hits, vec![(4107, 3.0)]);
        assert!(c.queues[0].heap.is_empty());
        // The next segment starts from threshold 0 again, the queue kept.
        let mut c = HybridCollector::new(1, 1);
        c.segment(&[vec![sd(0, 5.0)]], 0);
        c.segment(&[vec![sd(0, 1.0)]], 1);
        assert_eq!(c.total, 2);
        let hits = c.queues.remove(0).into_sorted();
        assert_eq!((hits[0].doc, hits[0].score), (0, 5.0));
    }

    #[test]
    fn a_hybrid_search_keeps_each_sub_querys_top_hits_per_slice() {
        let h = open();
        let subs = [term_blob("body", "fox"), term_blob("body", "dog")];
        let all: Vec<Vec<(i32, f32)>> = subs
            .iter()
            .map(|q| run_limit(h, q, 100, i64::MAX).unwrap().0)
            .collect();
        let mut union: Vec<i32> = all.iter().flatten().map(|&(d, _)| d).collect();
        union.sort_unstable();
        union.dedup();
        // One slice: each sub-query's own top hits, the total the union's.
        let one = decode_hybrid(
            &search_hybrid_blobs(h, &hybrid_blob(&subs, 2, &[])).unwrap(),
            2,
        );
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].0, union.len() as i64);
        for (got, want) in one[0].1.iter().zip(&all) {
            assert_eq!(got, &want[..want.len().min(2)]);
        }
        // Two slices, the second segment first: each keeps its own segment's
        // hits, scored with the whole reader's statistics.
        let two = decode_hybrid(
            &search_hybrid_blobs(h, &hybrid_blob(&subs, 100, &[&[1], &[0]])).unwrap(),
            2,
        );
        assert_eq!(two.len(), 2);
        assert_eq!(two[0].0 + two[1].0, union.len() as i64);
        for (q, want) in all.iter().enumerate() {
            let mut got: Vec<(i32, f32)> = two.iter().flat_map(|s| s.1[q].clone()).collect();
            got.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            assert_eq!(&got, want);
        }
        assert!(two[0].1.iter().flatten().all(|&(d, _)| d >= 4));
        // Refused: no sub-query, no hits kept, a segment in two slices,
        // trailing bytes.
        for bad in [
            hybrid_blob(&[], 2, &[]),
            hybrid_blob(&subs, 0, &[]),
            hybrid_blob(&subs, 2, &[&[0], &[0, 1]]),
            [hybrid_blob(&subs, 2, &[]), vec![0]].concat(),
        ] {
            assert_eq!(
                search_hybrid_blobs(h, &bad).err(),
                Some(FfiStatus::InvalidArgument)
            );
        }
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn a_nested_key_carries_its_filters_and_runs_over_a_handle() {
        let nested_key = |ty: u8, flags: u8| {
            let mut b = vec![2u8, ty, flags | SORT_NESTED];
            b.extend_from_slice(&1i32.to_le_bytes());
            b.push(b'n');
            b.extend_from_slice(&i64::MAX.to_le_bytes());
            for q in [term_blob("body", "fox"), term_blob("body", "dog")] {
                b.extend_from_slice(&(q.len() as i32).to_le_bytes());
                b.extend_from_slice(&q);
            }
            b.extend_from_slice(&3i32.to_le_bytes());
            b.extend_from_slice(&[SORT_DOC, 0]);
            b.push(0); // no search-after
            b.push(0); // options
            b.extend_from_slice(&0i32.to_le_bytes()); // no slices
            b
        };
        let (keys, ..) = decode_sort(&nested_key(SORT_LONG, SORT_MODE_SUM)).unwrap();
        let n = keys[0].nested.as_ref().expect("nested");
        assert_eq!(n.max_children, 3);
        assert_eq!(keys[0].selector, Selector::Sum);
        assert!(keys[1].nested.is_none());
        // Not for the score, a keyword or a median (no nested pick).
        for (ty, flags) in [(SORT_STRING, 0), (SORT_LONG, SORT_MODE_MEDIAN)] {
            let mut b = nested_key(ty, flags);
            if ty == SORT_STRING {
                // A keyword's missing value is 0 or 1.
                b[8..16].copy_from_slice(&0i64.to_le_bytes());
            }
            assert_eq!(decode_sort(&b).err(), Some(FfiStatus::InvalidArgument));
        }
        // Over a handle: the roots (documents matching "fox") sort by their
        // children's (dog documents') values.
        let h = open();
        let q = term_blob("body", "fox");
        let (hits, total, _) = run_sorted(h, &q, &nested_key(SORT_LONG, 0), 5, i64::MAX).unwrap();
        assert!(!hits.is_empty() && total >= hits.len() as i64);
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn decode_sort_reads_every_key_kind_and_rejects_malformed_blobs() {
        let blob = sort_blob(
            &[
                (SORT_SCORE, SORT_REVERSE, "", 0),
                (SORT_DOC, 0, "", 0),
                (SORT_LONG, SORT_MAX, "a", -5),
                (SORT_INT, 0, "b", 7),
                (SORT_DOUBLE, SORT_REVERSE, "c", 1),
                (SORT_FLOAT, 0, "d", 2),
            ],
            Some((9, &[1, 2, 3, 4, 5, 6])),
        );
        let (keys, after, _, _, _, _) = decode_sort(&blob).unwrap();
        assert_eq!(
            keys.iter().map(|k| k.ty).collect::<Vec<_>>(),
            [
                SortType::Score,
                SortType::Doc,
                SortType::Long,
                SortType::Int,
                SortType::Double,
                SortType::Float
            ]
        );
        assert!(keys[0].reverse && !keys[1].reverse && keys[4].reverse);
        assert_eq!(keys[2].selector, Selector::Max);
        assert_eq!(keys[3].selector, Selector::Min);
        assert_eq!((keys[2].field.as_str(), keys[2].missing), ("a", -5));
        assert_eq!(
            after,
            Some(FieldDoc {
                doc: 9,
                values: vec![1, 2, 3, 4, 5, 6],
                terms: Vec::new(),
            })
        );

        let invalid = Err(FfiStatus::InvalidArgument);
        let status = |b: &[u8]| decode_sort(b).map(|_| ());
        assert_eq!(status(&[]), invalid, "empty");
        assert_eq!(status(&[0, 0]), invalid, "no keys");
        assert_eq!(status(&[17]), invalid, "too many keys");
        assert_eq!(status(&[1, SORT_DOC, 4, 0]), invalid, "unknown flag");
        assert_eq!(status(&[1, 9, 0, 0]), invalid, "unknown type");
        assert_eq!(
            status(&[1, SORT_DOC, 0, 2]),
            invalid,
            "has_after is not a bool"
        );
        assert_eq!(status(&[1, SORT_DOC, 0, 0]), invalid, "no options byte");
        assert_eq!(
            status(&[1, SORT_DOC, 0, 0, 0, 0]),
            invalid,
            "trailing bytes"
        );
        assert_eq!(status(&[1, SORT_DOC, 0, 1, 0]), invalid, "truncated after");
        let mut bad_utf8 = vec![1, SORT_LONG, 0];
        bad_utf8.extend_from_slice(&1i32.to_le_bytes());
        bad_utf8.push(0xff);
        bad_utf8.extend_from_slice(&0i64.to_le_bytes());
        bad_utf8.push(0);
        assert_eq!(
            decode_sort(&bad_utf8).map(|_| ()),
            Err(FfiStatus::InvalidUtf8)
        );
    }

    /// `blob` (a [`sort_blob`]) with its options byte set to `options` and,
    /// when non-zero, `terminate_after` following it.
    fn with_terminate_after(mut blob: Vec<u8>, options: u8, n: i32) -> Vec<u8> {
        let at = blob.len() - 5;
        blob[at] = options;
        if options & SORT_TERMINATE_AFTER != 0 {
            blob.splice(at + 1..at + 1, n.to_le_bytes());
        }
        blob
    }

    /// `blob` (options last but the slices) with per-segment index-sort flags.
    fn with_index_sorted(mut blob: Vec<u8>, flags: &[u8]) -> Vec<u8> {
        let at = blob.len() - 5;
        blob[at] |= SORT_INDEX_SORTED;
        blob.extend_from_slice(&(flags.len() as i32).to_le_bytes());
        blob.extend_from_slice(flags);
        blob
    }

    #[test]
    fn index_sort_flags_decode_and_leave_an_unsorted_prefix_alone() {
        let h = open();
        let fox = term_blob("body", "fox");
        let by_score = sort_blob(&[(SORT_SCORE, 0, "", 0)], None);
        assert!(decode_sort(&by_score).unwrap().5.is_empty());
        let flagged = with_index_sorted(by_score.clone(), &[1, 0]);
        assert_eq!(decode_sort(&flagged).unwrap().5, vec![Some(0), None]);
        assert_eq!(
            decode_sort(&with_index_sorted(by_score.clone(), &[3]))
                .unwrap()
                .5,
            vec![Some(2)]
        );
        assert_eq!(
            decode_sort(&with_index_sorted(by_score.clone(), &[128])).err(),
            Some(FfiStatus::InvalidArgument)
        );
        // Flags saying no segment's index sort begins with the search's: the
        // answer without them.
        let plain = search_sorted_blobs(h, &fox, &by_score, 8, i64::MAX).unwrap();
        let off = search_sorted_blobs(h, &fox, &with_index_sorted(by_score, &[0, 0]), 8, i64::MAX)
            .unwrap();
        assert_eq!(
            off.hits
                .iter()
                .map(|f| (f.doc, f.values.clone()))
                .collect::<Vec<_>>(),
            plain
                .hits
                .iter()
                .map(|f| (f.doc, f.values.clone()))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            (off.total, off.lower_bound),
            (plain.total, plain.lower_bound)
        );
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn doc_freq_is_the_exact_count_of_a_term() {
        let h = open();
        let (_, total) = run(h, &term_blob("body", "fox"), 8, true).unwrap();
        assert_eq!(doc_freq(h, b"body", b"fox"), Ok(total));
        assert_eq!(doc_freq(h, b"body", b"no-such-term"), Ok(0));
        assert_eq!(doc_freq(h, b"no-such-field", b"fox"), Ok(0));
        assert_eq!(
            doc_freq(h, &[0xff], b"fox"),
            Err(FfiStatus::InvalidArgument)
        );
        assert_eq!(
            doc_freq(u64::MAX, b"body", b"fox"),
            Err(FfiStatus::InvalidHandle)
        );
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn terminate_after_keeps_the_first_matches_and_says_whether_it_stopped() {
        let h = open();
        let fox = term_blob("body", "fox");
        let (scored, total) = run(h, &fox, 8, true).unwrap();
        assert!(
            total > 2,
            "the fixture's fox matches more than two documents"
        );
        let mut in_order: Vec<i32> = scored.iter().map(|(d, _)| *d).collect();
        in_order.sort_unstable();
        let by_doc = sort_blob(&[(SORT_DOC, 0, "", 0)], None);
        // Two let through, of more: the first two, counted, and stopped early.
        let two = with_terminate_after(by_doc.clone(), SORT_TERMINATE_AFTER, 2);
        let out = search_sorted_blobs(h, &fox, &two, 8, i64::MAX).unwrap();
        assert_eq!(
            out.hits.iter().map(|f| f.doc).collect::<Vec<_>>(),
            in_order[..2]
        );
        assert_eq!(
            (out.total, out.lower_bound, out.terminated),
            (2, false, true)
        );
        // More than match: every match, not stopped; uncounted, -1.
        let all = with_terminate_after(by_doc.clone(), SORT_TERMINATE_AFTER, 1_000_000);
        let out = search_sorted_blobs(h, &fox, &all, 8, i64::MAX).unwrap();
        assert_eq!((out.total, out.terminated), (total, false));
        let out = search_sorted_blobs(h, &fox, &all, 8, 0).unwrap();
        assert_eq!(out.total, -1);
        // size: 0 over a term: the segments' counts, no hits.
        let counted = with_terminate_after(
            by_doc.clone(),
            SORT_TERMINATE_AFTER | SORT_COUNT_SEGMENTS,
            1,
        );
        let out = search_sorted_blobs(h, &fox, &counted, 1, i64::MAX).unwrap();
        assert!(out.hits.is_empty() && out.terminated);
        assert!(out.total >= 1);
        let out = search_sorted_blobs(h, &fox, &counted, 1, 0).unwrap();
        assert_eq!(out.total, -1);
        // A query with no segment count is refused, not answered wrongly.
        let both = bool_blob(
            0,
            &[(0, 0, -1, 0, "body", "fox"), (0, 0, -1, 0, "body", "dog")],
        );
        assert_eq!(
            search_sorted_blobs(h, &both, &counted, 1, i64::MAX).err(),
            Some(FfiStatus::InvalidArgument)
        );
        // Malformed: no count at all, and segment counts without one.
        let zero = with_terminate_after(by_doc.clone(), SORT_TERMINATE_AFTER, 0);
        assert_eq!(decode_sort(&zero).err(), Some(FfiStatus::InvalidArgument));
        let stray = with_terminate_after(by_doc, SORT_COUNT_SEGMENTS, 0);
        assert_eq!(decode_sort(&stray).err(), Some(FfiStatus::InvalidArgument));
        assert_eq!(decode_sort(&two).unwrap().4, Some((2, false)));
        assert_eq!(decode_sort(&counted).unwrap().4, Some((1, true)));
    }

    /// A count blob ([`decode_count_spec`]): `n`, the iterate flags, one slice
    /// of `slice`.
    fn count_spec(n: i32, iterate: &[u8], slice: &[i32]) -> Vec<u8> {
        let mut b = n.to_le_bytes().to_vec();
        b.extend_from_slice(&(iterate.len() as i32).to_le_bytes());
        b.extend_from_slice(iterate);
        b.extend_from_slice(&1i32.to_le_bytes());
        b.extend_from_slice(&(slice.len() as i32).to_le_bytes());
        for i in slice {
            b.extend_from_slice(&i.to_le_bytes());
        }
        b
    }

    /// The aggregations' per-segment counts replay a concurrent `size: 0`
    /// count's early termination exactly as [`count_terminates_blobs`] runs
    /// it: what the plugin computes from them (`countTerminatedEarly`)
    /// instead of searching again. Every limit, iterate pattern and slicing
    /// (a slice in reverse doc-base order included).
    #[test]
    fn the_aggregations_counts_replay_the_count_collector() {
        {
            let shape: &[i32] = &[4, 4];
            let (rc, h) = open_with(shape, 0);
            assert_eq!(rc, 0, "{}", crate::error::last_error());
            let segments = lookup(h, "test").unwrap().reader.segment_readers().len();
            let segs: Vec<i32> = (0..segments as i32).collect();
            let mut slicings: Vec<Vec<Vec<i32>>> = vec![vec![segs.clone()]];
            slicings.push(segs.iter().map(|&i| vec![i]).collect());
            slicings.push(vec![segs.iter().rev().copied().collect()]);
            let patterns: Vec<Vec<u8>> = vec![
                vec![1; segments],
                vec![0; segments],
                (0..segments).map(|i| (i % 2) as u8).collect(),
            ];
            let mut checked = 0;
            for term in ["fox", "dog", "the", "no-such-term"] {
                let q = term_blob("body", term);
                let aggs = metrics_blob(&[(METRIC_LONG, "n")]);
                let (_, _, total, seen) = aggregate_counting_blobs_seen(h, &q, &aggs, 1).unwrap();
                assert!(total.is_some());
                let seen: Vec<u64> = seen.into_iter().map(Option::unwrap).collect();
                let all: u64 = seen.iter().sum();
                for n in 1..=(all as i32 + 2) {
                    for iterate in &patterns {
                        for slices in &slicings {
                            let mut spec = n.to_le_bytes().to_vec();
                            spec.extend_from_slice(&(segments as i32).to_le_bytes());
                            spec.extend_from_slice(iterate);
                            spec.extend_from_slice(&(slices.len() as i32).to_le_bytes());
                            for slice in slices {
                                spec.extend_from_slice(&(slice.len() as i32).to_le_bytes());
                                for i in slice {
                                    spec.extend_from_slice(&i.to_le_bytes());
                                }
                            }
                            let want = count_terminates_blobs(h, &q, &spec).unwrap();
                            // `RustQueryPhaseSearcher.replayCountTerminated`
                            let replay = slices.iter().any(|slice| {
                                let mut order = slice.clone();
                                order.sort_unstable(); // doc-base order is segment order here
                                let mut collected = 0u64;
                                for &i in &order {
                                    if collected >= n as u64 {
                                        return true;
                                    }
                                    if iterate[i as usize] == 0 {
                                        continue;
                                    }
                                    let c = seen[i as usize];
                                    if collected + c > n as u64 {
                                        return true;
                                    }
                                    collected += c;
                                }
                                false
                            });
                            assert_eq!(replay, want, "{term} n={n} {iterate:?} {slices:?}");
                            // monotone in the iterated segments: the plugin skips Lucene's
                            // weight when iterating everywhere does not stop
                            let everywhere = slices.iter().any(|slice| {
                                let mut collected = 0u64;
                                slice.iter().any(|&i| {
                                    let stop = collected >= n as u64
                                        || collected + seen[i as usize] > n as u64;
                                    collected += seen[i as usize];
                                    stop
                                })
                            });
                            assert!(everywhere || !want, "{term} n={n} {slices:?}");
                            checked += 1;
                        }
                    }
                }
            }
            assert!(checked > 50, "{checked}");
            assert_eq!(ffi_close_jvm_reader(h), 0);
        }
    }

    #[test]
    fn a_concurrent_count_stops_past_its_limit_in_the_segments_it_iterates() {
        let h = open();
        let fox = term_blob("body", "fox");
        let (_, total) = run(h, &fox, 8, true).unwrap();
        let segments = lookup(h, "test").unwrap().reader.segment_readers().len();
        let all: Vec<i32> = (0..segments as i32).collect();
        let mut out = false;
        let call = |spec: &[u8], out: *mut bool| unsafe {
            ffi_jvm_reader_count_terminates(
                h,
                fox.as_ptr(),
                fox.len(),
                spec.as_ptr(),
                spec.len(),
                out,
            )
        };
        let iterate_all = vec![1u8; segments];
        let answered = vec![0u8; segments];
        // Iterated, more than the limit: stopped.
        assert_eq!(call(&count_spec(1, &iterate_all, &all), &mut out), 0);
        assert!(out);
        // Every match within the limit: not stopped.
        assert_eq!(
            call(&count_spec(total as i32, &iterate_all, &all), &mut out),
            0
        );
        assert!(!out);
        // Every segment answered from Weight.count: nothing reaches the collector.
        assert_eq!(call(&count_spec(1, &answered, &all), &mut out), 0);
        assert!(!out);
        // Malformed: a limit of 0, a bad flag, trailing bytes, a missing segment, no output.
        assert_eq!(
            call(&count_spec(0, &iterate_all, &all), &mut out),
            FfiStatus::InvalidArgument.code()
        );
        assert_eq!(
            call(&count_spec(1, &[2], &all), &mut out),
            FfiStatus::InvalidArgument.code()
        );
        let mut trailing = count_spec(1, &iterate_all, &all);
        trailing.push(0);
        assert_eq!(call(&trailing, &mut out), FfiStatus::InvalidArgument.code());
        assert_ne!(
            call(&count_spec(1, &iterate_all, &[segments as i32]), &mut out),
            0
        );
        assert_eq!(
            call(&count_spec(1, &iterate_all, &all), std::ptr::null_mut()),
            FfiStatus::NullPointer.code()
        );
    }

    #[test]
    fn min_score_keeps_the_documents_scoring_at_least_it() {
        let h = open();
        let fox = term_blob("body", "fox");
        let (scored, total) = run(h, &fox, 8, true).unwrap();
        let min = scored[scored.len() / 2].1;
        let mut blob = vec![QUERY_MIN_SCORE];
        blob.extend_from_slice(&min.to_bits().to_le_bytes());
        blob.extend_from_slice(&fox);
        let want: Vec<(i32, f32)> = scored.iter().copied().filter(|&(_, s)| s >= min).collect();
        let (hits, got_total) = run(h, &blob, 8, true).unwrap();
        // The documents at or above the minimum lead, in the same order (ties at the
        // minimum past the top 8 may follow).
        assert_eq!(hits[..want.len()], want[..]);
        assert!(hits.iter().all(|&(_, s)| s >= min));
        assert!(got_total <= total);
        // The aggregations' pass counts what min_score's size-0 count does, at
        // every limit.
        for limit in [1, 2, 3, i64::MAX] {
            let (_, _, fused) =
                aggregate_counting_blobs(h, &blob, &metrics_blob(&[(METRIC_LONG, "n")]), limit)
                    .unwrap();
            let (_, count, lower) = run_limit(h, &blob, 0, limit).unwrap();
            assert_eq!(fused, Some((count, lower)), "min_score up to {limit}");
        }
        // A segment not visited (its aggregations answered from points) leaves
        // the count to the search; one past the limit before it does not.
        assert_eq!(min_score_total(&[Some(2), None], 5), None);
        assert_eq!(min_score_total(&[Some(6), None], 5), Some((6, true)));
        assert_eq!(min_score_total(&[Some(2), Some(3)], 5), Some((5, false)));
        // size: 0 counts the same documents (fox matches at most 8 here).
        if total <= 8 {
            let (none, count) = run(h, &blob, 0, true).unwrap();
            assert!(none.is_empty());
            assert_eq!(count, want.len() as i64);
        }
        // Behind a sort: every document scoring at least the minimum, in the
        // sort's order, counted exactly; by score, the unsorted search's hits.
        let (all, _) = run(h, &fox, 10_000, true).unwrap();
        let mut passing: Vec<(i32, f32)> = all.iter().copied().filter(|&(_, s)| s >= min).collect();
        let by_score = run_sorted(
            h,
            &blob,
            &sort_blob(&[(SORT_SCORE, 0, "", 0)], None),
            8,
            i64::MAX,
        )
        .unwrap();
        let docs: Vec<i32> = by_score.0.iter().map(|&(d, _)| d).collect();
        let top: Vec<i32> = passing.iter().take(8).map(|&(d, _)| d).collect();
        assert_eq!(docs, top);
        assert_eq!(by_score.1, passing.len() as i64);
        passing.sort_by_key(|&(d, _)| d);
        let by_doc = sort_blob(&[(SORT_DOC, 0, "", 0)], None);
        let (hits, count, lower) = run_sorted(h, &blob, &by_doc, 10_000, i64::MAX).unwrap();
        assert_eq!(
            hits.iter().map(|&(d, _)| d).collect::<Vec<_>>(),
            passing.iter().map(|&(d, _)| d).collect::<Vec<_>>()
        );
        assert_eq!((count, lower), (passing.len() as i64, false));
        // Tracked: the max is the best passing score.
        let mut tracked = by_doc.clone();
        let options = tracked.len() - 5;
        tracked[options] = SORT_TRACK_MAX_SCORE;
        let out = search_sorted_blobs(h, &blob, &tracked, 2, i64::MAX).unwrap();
        assert_eq!(out.max_score, all[0].1);
        assert_eq!(out.total, passing.len() as i64);
        // Nothing passes: nothing collected, no max.
        let mut high = vec![QUERY_MIN_SCORE];
        high.extend_from_slice(&1.0e9f32.to_bits().to_le_bytes());
        high.extend_from_slice(&fox);
        let out = search_sorted_blobs(h, &high, &tracked, 2, i64::MAX).unwrap();
        assert!(out.hits.is_empty() && out.total == 0 && out.max_score.is_nan());
        // Behind terminate_after it is refused (the plugin leaves that to Lucene).
        assert_eq!(
            search_sorted_blobs(
                h,
                &blob,
                &with_terminate_after(by_doc, SORT_TERMINATE_AFTER, 2),
                4,
                i64::MAX
            )
            .err(),
            Some(FfiStatus::InvalidArgument)
        );
        // Malformed: a truncated minimum, the tag twice.
        assert_eq!(
            decode_request(&[QUERY_MIN_SCORE, 0, 0]).err(),
            Some(FfiStatus::InvalidArgument)
        );
        let mut twice = vec![QUERY_MIN_SCORE, 0, 0, 0, 0];
        twice.extend_from_slice(&blob);
        assert_eq!(
            decode_request(&twice).err(),
            Some(FfiStatus::InvalidArgument)
        );
        // Beyond every score: nothing.
        let mut high = vec![QUERY_MIN_SCORE];
        high.extend_from_slice(&1.0e9f32.to_bits().to_le_bytes());
        high.extend_from_slice(&fox);
        let (none, count) = run(h, &high, 8, true).unwrap();
        assert!(none.is_empty());
        assert_eq!(count, 0);
    }

    #[test]
    fn a_sorted_search_orders_pages_and_counts_as_lucene_does() {
        let h = open();
        let fox = term_blob("body", "fox");
        let (scored, total) = run(h, &fox, 8, true).unwrap();
        // By score: the unsorted search's order, the score's bits as the value.
        let by_score = run_sorted(
            h,
            &fox,
            &sort_blob(&[(SORT_SCORE, 0, "", 0)], None),
            8,
            i64::MAX,
        )
        .unwrap();
        assert_eq!(by_score.1, total);
        assert!(!by_score.2);
        assert_eq!(
            by_score
                .0
                .iter()
                .map(|(d, v)| (*d, f32::from_bits(v[0] as u32)))
                .collect::<Vec<_>>(),
            scored
        );
        // By document, descending, then the page after its second hit.
        let by_doc = run_sorted(
            h,
            &fox,
            &sort_blob(&[(SORT_DOC, SORT_REVERSE, "", 0)], None),
            2,
            i64::MAX,
        )
        .unwrap();
        let docs: Vec<i32> = by_doc.0.iter().map(|(d, _)| *d).collect();
        let mut want: Vec<i32> = scored.iter().map(|(d, _)| *d).collect();
        want.sort_unstable_by(|a, b| b.cmp(a));
        assert_eq!(docs, want[..2]);
        let last = by_doc.0[1].clone();
        let page2 = run_sorted(
            h,
            &fox,
            &sort_blob(&[(SORT_DOC, SORT_REVERSE, "", 0)], Some((last.0, &last.1))),
            8,
            i64::MAX,
        )
        .unwrap();
        assert_eq!(
            page2.0.iter().map(|(d, _)| *d).collect::<Vec<_>>(),
            want[2..]
        );
        // A numeric key on a field with no doc values: every hit is missing,
        // so ties break by document; with no count, the total is -1.
        let by_absent = run_sorted(
            h,
            &fox,
            &sort_blob(&[(SORT_LONG, 0, "nope", 42)], None),
            8,
            0,
        )
        .unwrap();
        assert_eq!(by_absent.1, -1);
        let mut asc = want.clone();
        asc.sort_unstable();
        assert_eq!(
            by_absent
                .0
                .iter()
                .map(|(d, v)| (*d, v[0]))
                .collect::<Vec<_>>(),
            asc.iter().map(|&d| (d, 42)).collect::<Vec<_>>()
        );
        // A boolean query through the same path.
        let both = bool_blob(
            0,
            &[(0, 0, -1, 0, "body", "fox"), (0, 0, -1, 0, "body", "dog")],
        );
        assert!(run_sorted(h, &both, &sort_blob(&[(SORT_DOC, 0, "", 0)], None), 4, 1).is_ok());
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn keyword_keys_take_a_term_after_and_hand_back_terms() {
        let blob = sort_blob_terms(
            &[
                (SORT_STRING, SORT_REVERSE | SORT_MAX, "k", 1),
                (SORT_STRING, 0, "j", 0),
                (SORT_LONG, 0, "n", 3),
            ],
            Some((4, &[0, 0, 9], &[Some(b"abc".as_slice()), None, None])),
        );
        let (keys, after, _, _, _, _) = decode_sort(&blob).unwrap();
        assert_eq!(keys[0].ty, SortType::String);
        assert!(keys[0].reverse && keys[0].selector == Selector::Max && keys[0].missing == 1);
        assert_eq!((keys[1].missing, keys[1].reverse), (0, false));
        let after = after.unwrap();
        assert_eq!(after.terms, [Some(b"abc".to_vec()), None, None]);
        assert_eq!(after.values, [0, 0, 9]);
        // A missing value is first or last, nothing else; a term is present
        // or not.
        assert_eq!(
            decode_sort(&sort_blob(&[(SORT_STRING, 0, "k", 2)], None)).unwrap_err(),
            FfiStatus::InvalidArgument
        );
        let mut bad = sort_blob_terms(&[(SORT_STRING, 0, "k", 0)], Some((1, &[0], &[None])));
        let at = bad.len() - 6; // the term's present flag, before the options byte and slices
        bad[at] = 2;
        assert_eq!(decode_sort(&bad).unwrap_err(), FfiStatus::InvalidArgument);
        // Options: bit 0 tracks the max score; any other bit is refused.
        let mut tracked = sort_blob(&[(SORT_DOC, 0, "", 0)], None);
        let options = tracked.len() - 5;
        tracked[options] = SORT_TRACK_MAX_SCORE;
        assert!(decode_sort(&tracked).unwrap().2);
        tracked[options] = 2;
        assert_eq!(
            decode_sort(&tracked).unwrap_err(),
            FfiStatus::InvalidArgument
        );
        // Terms out: per hit, per keyword key, a length or -1.
        let hits = [
            FieldDoc {
                doc: 1,
                values: vec![0, 0, 7],
                terms: vec![Some(b"xy".to_vec()), None, None],
            },
            FieldDoc {
                doc: 2,
                values: vec![0, 0, 8],
                terms: vec![None, Some(Vec::new()), None],
            },
        ];
        let mut want = Vec::new();
        for part in [
            &2i32.to_le_bytes()[..],
            b"xy",
            &(-1i32).to_le_bytes(),
            &(-1i32).to_le_bytes(),
            &0i32.to_le_bytes(),
        ] {
            want.extend_from_slice(part);
        }
        assert_eq!(encode_terms(&keys, &hits).unwrap(), want);
        assert!(encode_terms(&keys[2..], &hits).unwrap().is_empty());
        // Through the search: a field no segment has sorts all as missing,
        // and a buffer without room reports the room needed.
        let h = open();
        let q = term_blob("body", "fox");
        let sort = sort_blob(&[(SORT_STRING, 0, "nosuch", 1)], None);
        let (hits, _, _) = run_sorted(h, &q, &sort, 3, i64::MAX).unwrap();
        assert!(!hits.is_empty());
        let (mut n, mut total, mut lower, mut terms_len) = (0usize, 0i64, false, 0usize);
        let mut max_score = 0f32;
        let (mut docs, mut values) = ([0i32; 3], [0i64; 3]);
        let rc = unsafe {
            ffi_jvm_reader_search_sorted(
                h,
                q.as_ptr(),
                q.len(),
                sort.as_ptr(),
                sort.len(),
                3,
                i64::MAX,
                docs.as_mut_ptr(),
                values.as_mut_ptr(),
                3,
                std::ptr::null_mut(),
                0,
                &mut n,
                &mut total,
                &mut lower,
                &mut terms_len,
                &mut max_score,
                &mut false,
            )
        };
        assert_eq!(rc, FfiStatus::BufferTooSmall.code());
        assert_eq!(terms_len, hits.len() * 4);
        let mut terms = vec![0u8; terms_len];
        let rc = unsafe {
            ffi_jvm_reader_search_sorted(
                h,
                q.as_ptr(),
                q.len(),
                sort.as_ptr(),
                sort.len(),
                3,
                i64::MAX,
                docs.as_mut_ptr(),
                values.as_mut_ptr(),
                3,
                terms.as_mut_ptr(),
                terms.len(),
                &mut n,
                &mut total,
                &mut lower,
                &mut terms_len,
                &mut max_score,
                &mut false,
            )
        };
        assert_eq!(rc, 0);
        assert_eq!(
            terms,
            (0..n)
                .flat_map(|_| (-1i32).to_le_bytes())
                .collect::<Vec<_>>()
        );
        // A null terms buffer claiming room is refused.
        let rc = unsafe {
            ffi_jvm_reader_search_sorted(
                h,
                q.as_ptr(),
                q.len(),
                sort.as_ptr(),
                sort.len(),
                3,
                i64::MAX,
                docs.as_mut_ptr(),
                values.as_mut_ptr(),
                3,
                std::ptr::null_mut(),
                8,
                &mut n,
                &mut total,
                &mut lower,
                &mut terms_len,
                &mut max_score,
                &mut false,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn a_tracked_sort_reports_the_top_score() {
        let h = open();
        let q = term_blob("body", "fox");
        let (best, _) = run(h, &q, 1, true).unwrap();
        let mut sort = sort_blob(&[(SORT_DOC, 0, "", 0)], None);
        let options = sort.len() - 5;
        sort[options] = SORT_TRACK_MAX_SCORE;
        let (mut n, mut total, mut lower, mut terms_len) = (0usize, 0i64, false, 0usize);
        let mut max_score = 0f32;
        let (mut docs, mut values) = ([0i32; 3], [0i64; 3]);
        let call = |sort: &[u8],
                    max_score: &mut f32,
                    n: &mut usize,
                    total: &mut i64,
                    lower: &mut bool,
                    terms_len: &mut usize,
                    docs: &mut [i32; 3],
                    values: &mut [i64; 3]| unsafe {
            ffi_jvm_reader_search_sorted(
                h,
                q.as_ptr(),
                q.len(),
                sort.as_ptr(),
                sort.len(),
                3,
                i64::MAX,
                docs.as_mut_ptr(),
                values.as_mut_ptr(),
                3,
                std::ptr::null_mut(),
                0,
                n,
                total,
                lower,
                terms_len,
                max_score,
                &mut false,
            )
        };
        assert_eq!(
            call(
                &sort,
                &mut max_score,
                &mut n,
                &mut total,
                &mut lower,
                &mut terms_len,
                &mut docs,
                &mut values
            ),
            0
        );
        assert_eq!(max_score.to_bits(), best[0].1.to_bits());
        // Untracked: no max score.
        sort[options] = 0;
        assert_eq!(
            call(
                &sort,
                &mut max_score,
                &mut n,
                &mut total,
                &mut lower,
                &mut terms_len,
                &mut docs,
                &mut values
            ),
            0
        );
        assert!(max_score.is_nan());
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn a_sliced_sort_merges_its_slices() {
        let h = open();
        let q = term_blob("body", "fox");
        let keys = [
            (SORT_LONG, SORT_REVERSE, "n", i64::MIN),
            (SORT_DOC, 0, "", 0),
        ];
        let plain = sort_blob(&keys, None);
        // Every segment its own slice, last first: the one-collector answer.
        let n = lookup(h, "test").unwrap().reader.segment_readers().len() as i32;
        assert!(n > 1, "the fixture has {n} segments");
        let mut sliced = plain.clone();
        sliced.truncate(sliced.len() - 4);
        sliced.extend_from_slice(&n.to_le_bytes());
        for seg in (0..n).rev() {
            sliced.extend_from_slice(&1i32.to_le_bytes());
            sliced.extend_from_slice(&seg.to_le_bytes());
        }
        let (_, _, _, slices, _, _) = decode_sort(&sliced).unwrap();
        assert_eq!(slices.len(), n as usize);
        assert_eq!(
            run_sorted(h, &q, &sliced, 3, i64::MAX),
            run_sorted(h, &q, &plain, 3, i64::MAX)
        );
        // A slice naming a segment the reader lacks is a search error.
        let mut missing = plain.clone();
        missing.truncate(missing.len() - 4);
        for v in [2i32, 1, 7, 0] {
            missing.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(
            run_sorted(h, &q, &missing, 3, i64::MAX).unwrap_err(),
            FfiStatus::Search.code()
        );
        // A segment in two slices would be counted twice (a fuzz finding:
        // 17 total hits over an 8-document index): refused at decode.
        let mut twice = plain.clone();
        twice.truncate(twice.len() - 4);
        for v in [2i32, 1, 0, 1, 0] {
            twice.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(
            decode_sort(&twice).map(|_| ()),
            Err(FfiStatus::InvalidArgument)
        );
        ffi_close_jvm_reader(h);
    }

    /// An aggregations blob of `terms` alone.
    fn terms_blob(terms: &[(&str, i32)], slices: &[&[i32]]) -> Vec<u8> {
        aggs_blob(&[], terms, slices)
    }

    #[test]
    fn terms_run_in_the_aggregate_call() {
        let invalid = Err(FfiStatus::InvalidArgument);
        assert_eq!(
            decode_metrics(&terms_blob(&[("f", 0)], &[])).map(|_| ()),
            invalid,
            "shard_size 0"
        );
        assert_eq!(
            decode_metrics(&terms_blob(&[], &[])).map(|_| ()),
            invalid,
            "no aggregation at all"
        );
        let mut utf8 = terms_blob(&[("f", 3)], &[]);
        utf8[6] = 0xff;
        assert_eq!(
            decode_metrics(&utf8).map(|_| ()),
            Err(FfiStatus::InvalidUtf8)
        );
        let (specs, terms, slices) = decode_metrics(&terms_blob(&[("f", 3)], &[&[1, 0]])).unwrap();
        assert!(specs.is_empty());
        assert_eq!(
            terms,
            vec![TermsSpec {
                field: "f".to_string(),
                shard_size: 3
            }]
        );
        assert_eq!(slices, vec![vec![1, 0]]);
        let enc = encode_terms_results(&[lucene_aggs::terms_agg::TermsResult {
            buckets: vec![(b"ab".to_vec(), 2)],
            other_doc_count: 5,
        }])
        .unwrap();
        let mut want = 5i64.to_le_bytes().to_vec();
        want.extend_from_slice(&1i32.to_le_bytes());
        want.extend_from_slice(&2i64.to_le_bytes());
        want.extend_from_slice(&2i32.to_le_bytes());
        want.extend_from_slice(b"ab");
        assert_eq!(enc, want);

        // A field without keyword doc values: nothing counted, one slice;
        // with a metric beside it, both come back.
        let h = open();
        let q = term_blob("body", "fox");
        let aggs = aggs_blob(&[(METRIC_LONG, 0, "nosuch")], &[("nosuch", 10)], &[]);
        let (mut counts, mut values) = ([7i64; 1], [7f64; METRIC_VALUES]);
        let mut len = 0usize;
        let mut buf = [1u8; 12];
        let mut call = |out: *mut u8, cap: usize, len: &mut usize| unsafe {
            ffi_jvm_reader_aggregate(
                h,
                q.as_ptr(),
                q.len(),
                aggs.as_ptr(),
                aggs.len(),
                counts.as_mut_ptr(),
                values.as_mut_ptr(),
                1,
                out,
                cap,
                len,
            )
        };
        assert_eq!(call(buf.as_mut_ptr(), buf.len(), &mut len), 0);
        assert_eq!(len, 12);
        assert_eq!(buf, [0u8; 12]);
        assert_eq!(
            call(std::ptr::null_mut(), 0, &mut len),
            FfiStatus::BufferTooSmall.code()
        );
        assert_eq!(len, 12);
        assert_eq!(
            call(std::ptr::null_mut(), 4, &mut len),
            FfiStatus::NullPointer.code()
        );
        assert_eq!(counts[0], 0);
        ffi_close_jvm_reader(h);
    }

    thread_local! {
        /// Set: terms aggregations read buffered keyword runs encoded.
        pub(crate) static ENCODED_RUNS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// [`ENCODED_RUNS`] set until dropped, then cleared, a panic included.
    pub(crate) struct EncodedRuns;

    impl EncodedRuns {
        pub(crate) fn set(encoded: bool) -> Self {
            ENCODED_RUNS.with(|c| c.set(encoded));
            Self
        }
    }

    impl Drop for EncodedRuns {
        fn drop(&mut self) {
            ENCODED_RUNS.with(|c| c.set(false));
        }
    }

    fn metrics_blob(fields: &[(u8, &str)]) -> Vec<u8> {
        metrics_blob_from(&fields.iter().map(|&(k, f)| (k, 0, f)).collect::<Vec<_>>())
    }

    fn metrics_blob_from(fields: &[(u8, u8, &str)]) -> Vec<u8> {
        metrics_blob_sliced(fields, &[])
    }

    fn metrics_blob_sliced(fields: &[(u8, u8, &str)], slices: &[&[i32]]) -> Vec<u8> {
        aggs_blob(fields, &[], slices)
    }

    /// An aggregations blob: metrics, `terms` (field, shard size), slices.
    pub(crate) fn aggs_blob(
        fields: &[(u8, u8, &str)],
        terms: &[(&str, i32)],
        slices: &[&[i32]],
    ) -> Vec<u8> {
        let mut b = vec![fields.len() as u8];
        for &(kind, source, f) in fields {
            b.push(kind);
            b.push(source);
            b.push(NEED_ALL);
            b.extend_from_slice(&(f.len() as i32).to_le_bytes());
            b.extend_from_slice(f.as_bytes());
        }
        b.push(terms.len() as u8);
        for &(f, shard_size) in terms {
            b.extend_from_slice(&(f.len() as i32).to_le_bytes());
            b.extend_from_slice(f.as_bytes());
            b.extend_from_slice(&shard_size.to_le_bytes());
        }
        b.extend_from_slice(&(slices.len() as i32).to_le_bytes());
        for slice in slices {
            b.extend_from_slice(&(slice.len() as i32).to_le_bytes());
            for &seg in *slice {
                b.extend_from_slice(&seg.to_le_bytes());
            }
        }
        b
    }

    #[test]
    fn the_aggregations_count_is_the_plain_count_under_deletions() {
        // Segment 1's document 0 deleted: the lower bound and the live count
        // part ways there.
        let (rc, h) = open_live(&[4, 4], 0, &[&[], &[0b1110]]);
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        for term in ["fox", "dog", "the"] {
            let q = term_blob("body", term);
            for limit in [1, 2, 3, 4, 6, i64::MAX] {
                let (_, _, fused) =
                    aggregate_counting_blobs(h, &q, &metrics_blob(&[(METRIC_LONG, "n")]), limit)
                        .unwrap();
                let (_, total, lower) = run_limit(h, &q, 0, limit).unwrap();
                assert_eq!(fused, Some((total, lower)), "{term} up to {limit}");
            }
        }
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn metrics_blobs_decode_and_aggregate() {
        let (specs, _, slices) = decode_metrics(&metrics_blob(&[
            (METRIC_LONG, "a"),
            (METRIC_DOUBLE, "b"),
            (METRIC_FLOAT, "c"),
        ]))
        .unwrap();
        assert!(slices.is_empty());
        assert_eq!(
            specs
                .iter()
                .map(|s| (s.field.as_str(), s.kind))
                .collect::<Vec<_>>(),
            [
                ("a", ValueKind::Long),
                ("b", ValueKind::Double),
                ("c", ValueKind::Float)
            ]
        );
        let invalid = Err(FfiStatus::InvalidArgument);
        assert_eq!(decode_metrics(&[0]).map(|_| ()), invalid, "no fields");
        assert_eq!(
            decode_metrics(&metrics_blob(&[(9, "a")])).map(|_| ()),
            invalid
        );
        let mut trailing = metrics_blob(&[(METRIC_LONG, "a")]);
        trailing.push(0);
        assert_eq!(decode_metrics(&trailing).map(|_| ()), invalid);
        assert_eq!(
            decode_metrics(&metrics_blob_from(&[(METRIC_LONG, 3, "a")])).map(|_| ()),
            invalid,
            "unknown source"
        );
        let sources = decode_metrics(&metrics_blob_from(&[
            (METRIC_LONG, METRIC_POINTS_MIN, "a"),
            (METRIC_LONG, METRIC_POINTS_MAX, "a"),
        ]))
        .unwrap()
        .0;
        assert_eq!(sources[0].source, Source::PointsMin);
        assert_eq!(sources[1].source, Source::PointsMax);
        let sliced = |slices: &[&[i32]]| {
            decode_metrics(&metrics_blob_sliced(&[(METRIC_LONG, 0, "a")], slices)).map(|d| d.2)
        };
        assert_eq!(sliced(&[&[2, 0], &[1]]), Ok(vec![vec![2, 0], vec![1]]));
        assert_eq!(sliced(&[&[-1]]).map(|_| ()), invalid, "negative segment");
        let mut negative = metrics_blob(&[(METRIC_LONG, "a")]);
        negative.truncate(negative.len() - 4);
        assert_eq!(
            decode_metrics(&[negative.clone(), (-1i32).to_le_bytes().to_vec()].concat())
                .map(|_| ()),
            invalid,
            "negative slice count"
        );
        assert_eq!(
            decode_metrics(
                &[
                    negative.clone(),
                    1i32.to_le_bytes().to_vec(),
                    (-1i32).to_le_bytes().to_vec()
                ]
                .concat()
            )
            .map(|_| ()),
            invalid,
            "negative slice length"
        );
        assert_eq!(
            decode_metrics(
                &[
                    negative.clone(),
                    ((MAX_SLICES + 1) as i32).to_le_bytes().to_vec()
                ]
                .concat()
            )
            .map(|_| ()),
            invalid,
            "too many slices"
        );
        assert_eq!(
            decode_metrics(
                &[
                    negative,
                    1i32.to_le_bytes().to_vec(),
                    ((MAX_SLICED_SEGMENTS + 1) as i32).to_le_bytes().to_vec()
                ]
                .concat()
            )
            .map(|_| ()),
            invalid,
            "too many sliced segments"
        );
        let mut no_needs = metrics_blob(&[(METRIC_LONG, "a")]);
        no_needs[3] = 0;
        assert_eq!(
            decode_metrics(&no_needs).map(|_| ()),
            invalid,
            "no parts asked for"
        );
        no_needs[3] = 0x40;
        assert_eq!(
            decode_metrics(&no_needs).map(|_| ()),
            invalid,
            "unknown part"
        );
        let mut utf8 = vec![1, METRIC_LONG, METRIC_DOC_VALUES, NEED_ALL];
        utf8.extend_from_slice(&1i32.to_le_bytes());
        utf8.push(0xff);
        assert_eq!(
            decode_metrics(&utf8).map(|_| ()),
            Err(FfiStatus::InvalidUtf8)
        );

        // A field the index does not have: no values, the empty state.
        let h = open();
        let q = term_blob("body", "fox");
        let aggs = metrics_blob(&[(METRIC_LONG, "nosuch")]);
        let (mut counts, mut values) = ([7i64; 1], [7f64; METRIC_VALUES]);
        let mut terms_len = 0usize;
        let mut call = |handle: u64, counts: *mut i64, values: *mut f64, n: usize| unsafe {
            ffi_jvm_reader_aggregate(
                handle,
                q.as_ptr(),
                q.len(),
                aggs.as_ptr(),
                aggs.len(),
                counts,
                values,
                n,
                std::ptr::null_mut(),
                0,
                &mut terms_len,
            )
        };
        assert_eq!(call(h, counts.as_mut_ptr(), values.as_mut_ptr(), 1), 0);
        assert_eq!(counts[0], 0);
        assert_eq!(values[..4], [0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY]);
        assert_eq!(
            call(h, counts.as_mut_ptr(), values.as_mut_ptr(), 0),
            FfiStatus::BufferTooSmall.code()
        );
        assert_eq!(
            call(h, std::ptr::null_mut(), values.as_mut_ptr(), 1),
            FfiStatus::NullPointer.code()
        );
        assert_eq!(
            call(h, counts.as_mut_ptr(), std::ptr::null_mut(), 1),
            FfiStatus::NullPointer.code()
        );
        assert_eq!(
            call(closed_handle(), counts.as_mut_ptr(), values.as_mut_ptr(), 1),
            FfiStatus::InvalidHandle.code()
        );

        // The size-0 count from the aggregation's own pass: the plain count's
        // answer at every limit, and none asked for, none given.
        for term in ["fox", "dog", "no-such-term"] {
            let q = term_blob("body", term);
            for limit in [1, 2, 3, 5, i64::MAX] {
                let (_, _, fused) =
                    aggregate_counting_blobs(h, &q, &metrics_blob(&[(METRIC_LONG, "n")]), limit)
                        .unwrap();
                let (_, total, lower) = run_limit(h, &q, 0, limit).unwrap();
                assert_eq!(fused, Some((total, lower)), "{term} up to {limit}");
            }
            let (_, _, none) =
                aggregate_counting_blobs(h, &q, &metrics_blob(&[(METRIC_LONG, "n")]), 0).unwrap();
            assert_eq!(none, None);
        }

        // Slices: each keeps its own state; the same segment twice gives the
        // one-slice answer twice. A segment the reader lacks is an error.
        let q_all = term_blob("body", "fox");
        let one = aggregate_blobs(h, &q_all, &metrics_blob(&[(METRIC_LONG, "n")])).unwrap();
        let two = aggregate_blobs(
            h,
            &q_all,
            &metrics_blob_sliced(&[(METRIC_LONG, 0, "n")], &[&[0], &[0]]),
        )
        .unwrap();
        assert_eq!(two.0, [one.0[0], one.0[0]]);
        assert_eq!(
            aggregate_blobs(
                h,
                &q_all,
                &metrics_blob_sliced(&[(METRIC_LONG, 0, "n")], &[&[99]])
            ),
            Err(FfiStatus::Search)
        );
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn a_sorted_search_rejects_bad_arguments() {
        let h = open();
        let q = term_blob("body", "fox");
        let sort = sort_blob(&[(SORT_DOC, 0, "", 0)], None);
        let (mut n, mut total, mut lower, mut terms_len) = (0usize, 0i64, false, 0usize);
        let mut max_score = 0f32;
        let mut docs = [0i32; 1];
        let mut values = [0i64; 1];
        let mut call =
            |handle: u64, top_n: usize, docs: *mut i32, values: *mut i64, n: *mut usize| unsafe {
                ffi_jvm_reader_search_sorted(
                    handle,
                    q.as_ptr(),
                    q.len(),
                    sort.as_ptr(),
                    sort.len(),
                    top_n,
                    i64::MAX,
                    docs,
                    values,
                    1,
                    std::ptr::null_mut(),
                    0,
                    n,
                    &mut total,
                    &mut lower,
                    &mut terms_len,
                    &mut max_score,
                    &mut false,
                )
            };
        let (d, v) = (docs.as_mut_ptr(), values.as_mut_ptr());
        assert_eq!(call(h, 2, d, v, &mut n), FfiStatus::BufferTooSmall.code());
        assert_eq!(call(h, 0, d, v, &mut n), FfiStatus::InvalidArgument.code());
        assert_eq!(
            call(h, 1, std::ptr::null_mut(), v, &mut n),
            FfiStatus::NullPointer.code()
        );
        assert_eq!(
            call(h, 1, d, std::ptr::null_mut(), &mut n),
            FfiStatus::NullPointer.code()
        );
        assert_eq!(
            call(h, 1, d, v, std::ptr::null_mut()),
            FfiStatus::NullPointer.code()
        );
        assert_eq!(
            call(closed_handle(), 1, d, v, &mut n),
            FfiStatus::InvalidHandle.code()
        );
        assert_eq!(call(h, 1, d, v, &mut n), 0);
        assert_eq!(
            run_sorted(h, &q, &[0], 1, 1),
            Err(FfiStatus::InvalidArgument.code())
        );
        assert_eq!(
            run_sorted(h, &[9], &sort, 1, 1),
            Err(FfiStatus::InvalidArgument.code())
        );
        ffi_close_jvm_reader(h);
    }

    /// Hits and scores are Lucene's own, bit for bit, from the fixture's
    /// manifest -- the JVM reader is the multi-segment searcher, not a
    /// re-derivation of it.
    #[test]
    fn term_and_boolean_hits_match_lucene_bit_for_bit() {
        let h = open();
        let (hits, total) = run(h, &term_blob("body", "fox"), 10, true).unwrap();
        let want = [
            (4, 1059136106u32),
            (5, 1058735855),
            (6, 1058247114),
            (0, 1057234298),
        ];
        let got: Vec<(i32, u32)> = hits.iter().map(|&(d, s)| (d, s.to_bits())).collect();
        assert_eq!(got, want);
        assert_eq!(total, 4);

        let should = bool_blob(
            0,
            &[(2, 0, -1, 0, "body", "fox"), (2, 0, -1, 0, "body", "dog")],
        );
        let (hits, total) = run(h, &should, 10, true).unwrap();
        let want = [
            (4, 1063370973u32),
            (6, 1062787560),
            (5, 1058735855),
            (0, 1057234298),
            (7, 1049727197),
            (3, 1048058560),
            (1, 1047552856),
            (2, 1047077660),
        ];
        let got: Vec<(i32, u32)> = hits.iter().map(|&(d, s)| (d, s.to_bits())).collect();
        assert_eq!(got, want);
        assert_eq!(total, 8);
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    /// The total is counted, not inferred, once the top hits are full -- and
    /// with `top_n == 0` it is the only thing computed.
    #[test]
    fn total_is_counted_when_top_hits_are_full() {
        let h = open();
        let (hits, total) = run(h, &term_blob("body", "fox"), 2, true).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(total, 4);
        let (hits, total) = run(h, &term_blob("body", "fox"), 0, true).unwrap();
        assert!(hits.is_empty());
        assert_eq!(total, 4);
        let both = bool_blob(
            0,
            &[(0, 0, -1, 0, "body", "fox"), (0, 0, -1, 0, "body", "dog")],
        );
        let (hits, total) = run(h, &both, 1, true).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(total, 2, "docs 4 and 6 hold both terms");
        let (_, total) = run(h, &term_blob("body", "fox"), 2, false).unwrap();
        assert_eq!(total, -1);
        ffi_close_jvm_reader(h);
    }

    /// `count_limit` is Lucene's `totalHitsThreshold`: exact up to it, a lower
    /// bound above it -- never at or below the limit. These are the `size: 0`
    /// path's numbers ([`total_hits`]): from `docFreq` alone when that
    /// suffices, else counted segment by segment.
    #[test]
    fn count_limit_reports_a_lower_bound_at_the_threshold() {
        let h = open();
        let totals = |blob: &[u8], limit: i64| {
            let (hits, total, lower) = run_limit(h, blob, 0, limit).unwrap();
            assert!(hits.is_empty());
            (total, lower)
        };
        // fox: docFreq 1 + 3 over the two segments, no deletions.
        let fox = term_blob("body", "fox");
        assert_eq!(totals(&fox, 3), (4, true), "docFreq alone passes the limit");
        assert_eq!(
            totals(&fox, 4),
            (4, false),
            "reaching the limit is still exact"
        );
        assert_eq!(totals(&fox, 0), (-1, false), "no counting");
        // dog: docFreq 3 + 3; fox OR dog matches all 8. A disjunction's bound
        // is its best clause per segment: 3 + 3.
        let either = bool_blob(
            0,
            &[(2, 0, -1, 0, "body", "fox"), (2, 0, -1, 0, "body", "dog")],
        );
        assert_eq!(totals(&either, 5), (6, true));
        assert_eq!(totals(&either, 6), (8, true), "counted past the bound");
        assert_eq!(totals(&either, 8), (8, false));
        // fox AND dog: docs 4 and 6, both in segment 1. No cheap bound, so it
        // is counted, stopping at the first segment that passes the limit.
        let both = bool_blob(
            0,
            &[(0, 0, -1, 0, "body", "fox"), (0, 0, -1, 0, "body", "dog")],
        );
        assert_eq!(totals(&both, 1), (2, true));
        assert_eq!(totals(&both, 2), (2, false));
        // A MUST_NOT bounds nothing: counted.
        let not = bool_blob(
            0,
            &[(2, 0, -1, 0, "body", "dog"), (3, 0, -1, 0, "body", "fox")],
        );
        assert_eq!(totals(&not, 100), (4, false));
        // Wrappers bound as their inner clause.
        let one = 1.0f32.to_bits() as i32;
        let wrapped = bool_blob(0, &[(0, 2, -1, one, "", ""), (0, 0, 0, 0, "body", "fox")]);
        assert_eq!(totals(&wrapped, 3), (4, true));
        // A required clause alone bounds as itself.
        let filtered = bool_blob(0, &[(1, 0, -1, 0, "body", "fox")]);
        assert_eq!(totals(&filtered, 3), (4, true));

        // Deletions lower the bound: delete global doc 4 (segment 1, local 0),
        // which holds fox. fox's bound is now 1 + (3 - 1) = 3, its count 3.
        ffi_close_jvm_reader(h);
        let h = open_doc4_deleted();
        let totals = |blob: &[u8], limit: i64| {
            let (_, total, lower) = run_limit(h, blob, 0, limit).unwrap();
            (total, lower)
        };
        assert_eq!(totals(&fox, 2), (3, true));
        assert_eq!(totals(&fox, 3), (3, false));
        // More SHOULD clauses required than exist: Lucene matches nothing, and
        // the bound must not claim otherwise.
        let impossible = bool_blob(1, &[(0, 0, -1, 0, "body", "fox")]);
        assert_eq!(totals(&impossible, 100), (0, false));
        ffi_close_jvm_reader(h);
    }

    /// With top hits to collect, the collector counts -- Lucene's
    /// `TopScoreDocCollector` -- and the same contract holds whatever path
    /// scored the query: exact up to the limit, and past it either exact or a
    /// lower bound above the limit.
    #[test]
    fn collector_counts_obey_the_threshold_contract() {
        let h = open();
        let one = 1.0f32.to_bits() as i32;
        let queries = [
            (term_blob("body", "fox"), 4),
            (term_blob("body", "dog"), 6),
            (
                bool_blob(
                    0,
                    &[(2, 0, -1, 0, "body", "fox"), (2, 0, -1, 0, "body", "dog")],
                ),
                8,
            ),
            (
                bool_blob(
                    0,
                    &[(0, 0, -1, 0, "body", "fox"), (0, 0, -1, 0, "body", "dog")],
                ),
                2,
            ),
            (
                bool_blob(0, &[(0, 2, -1, one, "", ""), (0, 0, 0, 0, "body", "dog")]),
                6,
            ),
        ];
        for (blob, exact) in &queries {
            for limit in 1..=10 {
                for top_n in [1, 3] {
                    let (_, total, lower) = run_limit(h, blob, top_n, limit).unwrap();
                    if *exact <= limit {
                        assert_eq!((total, lower), (*exact, false), "limit {limit} top {top_n}");
                    } else {
                        assert!(
                            (!lower && total == *exact)
                                || (lower && total > limit && total <= *exact),
                            "limit {limit} top {top_n}: {total} {lower}, exact {exact}"
                        );
                    }
                }
            }
            let (_, total, lower) = run_limit(h, blob, 1, i64::MAX).unwrap();
            assert_eq!((total, lower), (*exact, false), "an exact count");
        }
        ffi_close_jvm_reader(h);
    }

    /// The JVM's live docs are the only deletions a search sees -- including
    /// in the count -- and clearing them restores every document.
    #[test]
    fn java_live_docs_mask_hits_and_counts() {
        // Segment 1 holds global docs 4..8; local doc 0 (global 4) is deleted.
        let h = open_doc4_deleted();
        let (hits, total) = run(h, &term_blob("body", "fox"), 2, true).unwrap();
        assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), [5, 6]);
        assert_eq!(total, 3);
        let should = bool_blob(
            0,
            &[(2, 0, -1, 0, "body", "fox"), (2, 0, -1, 0, "body", "dog")],
        );
        let (_, total) = run(h, &should, 1, true).unwrap();
        assert_eq!(total, 7);

        // A refresh that drops the deletion, reusing the segments: every
        // document is back, and the old handle still sees its own view.
        let (rc, h2) = open_live(&[4, 4], h, &[]);
        assert_eq!(rc, 0);
        let (_, total) = run(h2, &term_blob("body", "fox"), 2, true).unwrap();
        assert_eq!(total, 4);
        let (_, total) = run(h, &term_blob("body", "fox"), 2, true).unwrap();
        assert_eq!(total, 3);
        ffi_close_jvm_reader(h);
        ffi_close_jvm_reader(h2);
    }

    #[test]
    fn live_docs_are_validated() {
        let invalid = FfiStatus::InvalidArgument.code();
        let rc = open_live(&[4, 4], 0, &[&[0b1111, 0]]).0;
        assert_eq!(rc, invalid, "two words for maxDoc 4");
        let rc = open_live(&[4, 4], 0, &[&[0b1_0000]]).0;
        assert_eq!(rc, invalid, "a bit past maxDoc");
        let rc = open_live(&[4, 4], closed_handle(), &[]).0;
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
        // Counts claiming words that were not passed.
        let infos = infos();
        let docs = [4, 4];
        let counts = [1usize, 0];
        let mut h = 0u64;
        let rc = unsafe {
            ffi_open_jvm_reader(
                FIXTURE.as_ptr().cast(),
                FIXTURE.len(),
                infos.as_ptr(),
                infos.len(),
                2,
                0,
                docs.as_ptr(),
                docs.len(),
                std::ptr::null(),
                counts.as_ptr(),
                &mut h,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        let overflow = [usize::MAX, 1];
        let rc = unsafe {
            ffi_open_jvm_reader(
                FIXTURE.as_ptr().cast(),
                FIXTURE.len(),
                infos.as_ptr(),
                infos.len(),
                2,
                0,
                docs.as_ptr(),
                docs.len(),
                std::ptr::null(),
                overflow.as_ptr(),
                &mut h,
            )
        };
        assert_eq!(rc, invalid, "word counts that overflow");
    }

    /// A reader whose segment sizes disagree with the JVM's is refused: its
    /// doc IDs would name the wrong documents.
    #[test]
    fn open_refuses_a_segment_list_that_disagrees_with_the_jvm() {
        let (rc, _) = open_with(&[4, 5], 0);
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert!(crate::error::last_error().contains("do not match"));
        let (rc, _) = open_with(&[4], 0);
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
    }

    #[test]
    fn open_rejects_bad_arguments() {
        let infos = infos();
        let docs = [4, 4];
        let mut handle = 0u64;
        let call = |infos: &[u8], generation: i64, previous: u64, out: *mut u64| unsafe {
            ffi_open_jvm_reader(
                FIXTURE.as_ptr().cast(),
                FIXTURE.len(),
                infos.as_ptr(),
                infos.len(),
                generation,
                previous,
                docs.as_ptr(),
                docs.len(),
                std::ptr::null(),
                std::ptr::null(),
                out,
            )
        };
        assert_eq!(
            call(&infos, 2, 0, std::ptr::null_mut()),
            FfiStatus::NullPointer.code()
        );
        assert_eq!(
            call(&infos, 3, 0, &mut handle),
            FfiStatus::Decode.code(),
            "wrong generation"
        );
        assert_eq!(
            call(&infos[..20], 2, 0, &mut handle),
            FfiStatus::Decode.code()
        );
        assert_eq!(
            call(&infos, 2, 12345, &mut handle),
            FfiStatus::InvalidHandle.code()
        );
        let rc = unsafe {
            ffi_open_jvm_reader(
                FIXTURE.as_ptr().cast(),
                FIXTURE.len(),
                infos.as_ptr(),
                infos.len(),
                2,
                0,
                std::ptr::null(),
                2,
                std::ptr::null(),
                std::ptr::null(),
                &mut handle,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        let missing = "/nonexistent/lucene-rust-jvm-reader";
        let rc = unsafe {
            ffi_open_jvm_reader(
                missing.as_ptr().cast(),
                missing.len(),
                infos.as_ptr(),
                infos.len(),
                2,
                0,
                docs.as_ptr(),
                docs.len(),
                std::ptr::null(),
                std::ptr::null(),
                &mut handle,
            )
        };
        assert_eq!(rc, FfiStatus::Decode.code());
    }

    /// A refresh shares the previous handle's segments, and each handle then
    /// lives and closes on its own.
    #[test]
    fn reopening_from_a_previous_handle_shares_segments() {
        let first = open();
        let (rc, second) = open_with(&[4, 4], first);
        assert_eq!(rc, 0);
        {
            let readers = read_recovering(jvm_readers());
            let (a, b) = (readers.get(first).unwrap(), readers.get(second).unwrap());
            // `field_infos()` borrows from the segment's shared core, so one
            // address means one decoded copy.
            assert!(std::ptr::eq(
                a.reader.segment_readers()[1].field_infos(),
                b.reader.segment_readers()[1].field_infos()
            ));
        }
        assert_eq!(ffi_close_jvm_reader(first), 0);
        let (hits, _) = run(second, &term_blob("body", "fox"), 10, false).unwrap();
        assert_eq!(hits.len(), 4);
        assert_eq!(ffi_close_jvm_reader(second), 0);
        assert_eq!(
            ffi_close_jvm_reader(second),
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn search_rejects_bad_arguments() {
        let h = open();
        let blob = term_blob("body", "fox");
        let (mut n, mut total, mut lower) = (0usize, 0i64, false);
        let mut docs = [0i32; 1];
        let mut scores = [0f32; 1];
        let mut call = |handle: u64, top_n: usize, docs: *mut i32, n: *mut usize| unsafe {
            ffi_jvm_reader_search(
                handle,
                blob.as_ptr(),
                blob.len(),
                top_n,
                i64::MAX,
                docs,
                scores.as_mut_ptr(),
                1,
                n,
                &mut total,
                &mut lower,
            )
        };
        assert_eq!(
            call(h, 2, docs.as_mut_ptr(), &mut n),
            FfiStatus::BufferTooSmall.code()
        );
        assert_eq!(
            call(h, 1, std::ptr::null_mut(), &mut n),
            FfiStatus::NullPointer.code()
        );
        assert_eq!(
            call(h, 1, docs.as_mut_ptr(), std::ptr::null_mut()),
            FfiStatus::NullPointer.code()
        );
        assert_eq!(
            call(closed_handle(), 1, docs.as_mut_ptr(), &mut n),
            FfiStatus::InvalidHandle.code()
        );
        assert_eq!(
            run(h, &[9], 1, true),
            Err(FfiStatus::InvalidArgument.code())
        );
        ffi_close_jvm_reader(h);
    }

    /// A doc-and-score query whose documents lie past the segment it lands
    /// on was built against another reader: refused, never answered with
    /// doc ids outside the index.
    #[test]
    fn a_doc_and_score_query_from_another_reader_is_refused() {
        let blob = [
            2, 40, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 49, 0, 0, 0, 98, 0, 0, 1, 0, 0, 0, 0,
            111, 89, 100,
        ];
        let h = open();
        let (mut docs, mut scores) = ([0i32; 10], [0f32; 10]);
        let (mut n, mut total, mut lower) = (0usize, 0i64, false);
        let rc = unsafe {
            ffi_jvm_reader_search(
                h,
                blob.as_ptr(),
                blob.len(),
                10,
                10,
                docs.as_mut_ptr(),
                scores.as_mut_ptr(),
                10,
                &mut n,
                &mut total,
                &mut lower,
            )
        };
        assert_ne!(rc, 0);
        assert!(crate::error::last_error().contains("different reader"));
        ffi_close_jvm_reader(h);
    }

    /// A doc-and-score query listing a document twice is malformed: refused
    /// at decode rather than reaching a scorer that needs ascending docs.
    #[test]
    fn a_doc_and_score_query_listing_a_doc_twice_is_refused() {
        let mut blob = vec![2, 40, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0];
        for score in [1.0f32, 2.0] {
            blob.extend_from_slice(&3i32.to_le_bytes());
            blob.extend_from_slice(&score.to_le_bytes());
        }
        assert!(decode_request(&blob).is_err());
        assert!(crate::error::last_error().contains("listed twice"));
    }

    #[test]
    fn decode_query_rejects_malformed_blobs() {
        let invalid = Err(FfiStatus::InvalidArgument);
        let status = |b: &[u8]| decode_query(b).map(|_| ());
        assert_eq!(status(&[]), invalid, "empty");
        assert_eq!(status(&[7]), invalid, "unknown tag");
        let term = term_blob("body", "fox");
        assert_eq!(status(&term[..term.len() - 1]), invalid, "truncated");
        let mut trailing = term.clone();
        trailing.push(0);
        assert_eq!(status(&trailing), invalid, "trailing bytes");
        let mut negative = vec![QUERY_TERM];
        negative.extend_from_slice(&(-1i32).to_le_bytes());
        assert_eq!(status(&negative), invalid, "negative length");
        let mut huge = vec![QUERY_TERM];
        huge.extend_from_slice(&i32::MAX.to_le_bytes());
        assert_eq!(status(&huge), invalid, "length past the end");
        let mut bad_utf8 = vec![QUERY_TERM];
        bad_utf8.extend_from_slice(&1i32.to_le_bytes());
        bad_utf8.push(0xff);
        bad_utf8.extend_from_slice(&0i32.to_le_bytes());
        assert_eq!(status(&bad_utf8), Err(FfiStatus::InvalidUtf8));
        // The clause rules are `read_boolean_query`'s: an unknown occur, a
        // forward parent reference, and too many clauses.
        assert_eq!(status(&bool_blob(0, &[(9, 0, -1, 0, "f", "t")])), invalid);
        assert_eq!(status(&bool_blob(0, &[(2, 0, 3, 0, "f", "t")])), invalid);
        let mut many = vec![QUERY_BOOLEAN];
        many.extend_from_slice(&0i32.to_le_bytes());
        many.extend_from_slice(&i32::MAX.to_le_bytes());
        assert!(decode_query(&many).is_err());
    }

    /// A geo node's bytes, as `GeoEncoder.java` writes them.
    struct Geo(Vec<u8>);

    impl Geo {
        fn new(kind: u8, field: &str) -> Self {
            let mut g = Geo(vec![kind]);
            g.bytes(field.as_bytes());
            g
        }
        fn bytes(&mut self, b: &[u8]) -> &mut Self {
            self.int(b.len() as i32);
            self.0.extend_from_slice(b);
            self
        }
        fn int(&mut self, v: i32) -> &mut Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn byte(&mut self, v: u8) -> &mut Self {
            self.0.push(v);
            self
        }
        fn f(&mut self, v: f64) -> &mut Self {
            self.0.extend_from_slice(&v.to_bits().to_le_bytes());
            self
        }
        fn ring(&mut self, lats: &[f64], lons: &[f64]) -> &mut Self {
            self.int(lats.len() as i32);
            for &v in lats.iter().chain(lons) {
                self.f(v);
            }
            self
        }
        fn blob(&self) -> Vec<u8> {
            let mut b = vec![QUERY_TREE];
            b.extend_from_slice(&self.0);
            b
        }
    }

    /// The one clause a [`QUERY_TREE`] blob of a single node decodes to.
    fn single(blob: &[u8]) -> Result<Clause, FfiStatus> {
        match decode_query(blob)? {
            JvmQuery::Boolean(mut b) if b.must.len() == 1 => Ok(b.must.remove(0)),
            other => panic!("not one node: {other:?}"),
        }
    }

    /// The points-box and geo nodes (16-19): every geometry and relation
    /// decodes, the key is the node's bytes (equal queries equal, any change
    /// in a value a different query), and malformed or invalid input --
    /// every truncation included -- is an error, never a panic.
    #[test]
    fn decode_geo_nodes() {
        use lucene_search::extended_query::ExtendedQuery;
        let invalid = Err(FfiStatus::InvalidArgument);
        let status = |b: &[u8]| decode_query(b).map(|_| ());
        // Points box: `LatLonPoint.newBoxQuery`'s two 4-byte dimensions.
        let mut bx = Geo::new(NODE_POINTS_BOX, "loc");
        bx.int(2)
            .bytes(&[0, 0, 0, 1, 0, 0, 0, 2])
            .bytes(&[9, 0, 0, 1, 9, 0, 0, 2]);
        match single(&bx.blob()).unwrap() {
            Clause::Extended(e) => match *e {
                ExtendedQuery::PointRange(p) => {
                    assert_eq!((p.num_dims, p.bytes_per_dim), (2, 4));
                    assert_eq!(p.field, "loc");
                }
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
        assert!(query_uses_points(&decode_query(&bx.blob()).unwrap()));
        for dims in [0, 17] {
            let mut b = Geo::new(NODE_POINTS_BOX, "loc");
            b.int(dims).bytes(&[0; 8]).bytes(&[0; 8]);
            assert_eq!(status(&b.blob()), invalid, "{dims} dimensions");
        }
        let mut odd = Geo::new(NODE_POINTS_BOX, "loc");
        odd.int(2).bytes(&[0; 3]).bytes(&[0; 3]);
        assert_eq!(
            status(&odd.blob()),
            invalid,
            "corners not a multiple of the dimensions"
        );
        // A points box of one 16-byte dimension (`ip`), and a point set.
        let mut ip = Geo::new(NODE_POINTS_BOX, "ip");
        ip.int(1).bytes(&[0; 16]).bytes(&[0xff; 16]);
        assert!(query_uses_points(&decode_query(&ip.blob()).unwrap()));
        let set = |dims: i32, width: i32, points: &[&[u8]]| {
            let mut g = Geo::new(NODE_POINT_SET, "si");
            g.int(dims).int(width).int(points.len() as i32);
            for p in points {
                g.bytes(p);
            }
            g
        };
        let ints = set(
            1,
            4,
            &[&[0x80, 0, 0, 7], &[0x80, 0, 0, 3], &[0x80, 0, 0, 7]],
        );
        match single(&ints.blob()).unwrap() {
            Clause::Extended(e) => match *e {
                ExtendedQuery::PointInSet(p) => {
                    assert_eq!(
                        (p.field.as_str(), p.num_dims, p.bytes_per_dim),
                        ("si", 1, 4)
                    );
                    assert_eq!(p.points, vec![vec![0x80, 0, 0, 3], vec![0x80, 0, 0, 7]]);
                }
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
        assert!(query_uses_points(&decode_query(&ints.blob()).unwrap()));
        assert!(decode_query(&set(1, 4, &[]).blob()).is_ok(), "an empty set");
        assert_eq!(
            status(&set(1, 4, &[&[0; 3]]).blob()),
            invalid,
            "a point of the wrong width"
        );
        assert_eq!(
            status(&set(1, 17, &[&[0; 17]]).blob()),
            invalid,
            "17 bytes per dimension"
        );
        assert_eq!(status(&set(0, 4, &[]).blob()), invalid, "no dimensions");
        let full = set(1, 4, &[&[0x80, 0, 0, 7]]).blob();
        for n in 1..full.len() {
            assert!(decode_query(&full[..n]).is_err(), "truncated at {n}");
        }

        // Distance, and its key.
        let distance = |radius: f64| {
            let mut g = Geo::new(NODE_GEO_DISTANCE, "loc");
            g.f(10.0).f(-179.5).f(radius);
            g
        };
        let a = single(&distance(1000.0).blob()).unwrap();
        assert_eq!(
            a,
            single(&distance(1000.0).blob()).unwrap(),
            "equal queries"
        );
        assert_ne!(
            a,
            single(&distance(1000.0000001).blob()).unwrap(),
            "a different radius"
        );
        assert!(format!("{a:?}").contains(&format!("{:02x}", NODE_GEO_DISTANCE)));
        assert!(!query_uses_points(
            &decode_query(&distance(1.0).blob()).unwrap()
        ));
        assert_eq!(status(&distance(-1.0).blob()), invalid, "negative radius");
        let mut bad_lat = Geo::new(NODE_GEO_DISTANCE, "loc");
        bad_lat.f(91.0).f(0.0).f(1.0);
        assert_eq!(status(&bad_lat.blob()), invalid, "latitude past the pole");

        // Every geometry, under every relation it allows.
        let square_lats = [0.0, 0.0, 1.0, 1.0, 0.0];
        let square_lons = [0.0, 1.0, 1.0, 0.0, 0.0];
        let hole_lats = [0.25, 0.25, 0.75, 0.75, 0.25];
        let hole_lons = [0.25, 0.75, 0.75, 0.25, 0.25];
        let geometries = |g: &mut Geo, line: bool| {
            g.int(if line { 5 } else { 4 });
            g.byte(GEOMETRY_POINT).f(0.5).f(0.5);
            g.byte(GEOMETRY_POLYGON)
                .ring(&square_lats, &square_lons)
                .int(1);
            g.ring(&hole_lats, &hole_lons).int(0);
            g.byte(GEOMETRY_RECTANGLE).f(-1.0).f(1.0).f(179.0).f(-179.0);
            g.byte(GEOMETRY_CIRCLE).f(89.9).f(180.0).f(5000.0);
            if line {
                g.byte(GEOMETRY_LINE).ring(&[0.0, 1.0], &[0.0, 1.0]);
            }
        };
        for kind in [NODE_GEO_POINT, NODE_GEO_SHAPE] {
            for rel in 0..3u8 {
                let mut g = Geo::new(kind, "shape");
                g.byte(rel);
                geometries(&mut g, rel != 1);
                let blob = g.blob();
                assert!(
                    single(&blob).is_ok(),
                    "kind {kind} relation {rel}: {}",
                    crate::error::last_error()
                );
                // Every truncation is an error.
                for cut in 1..blob.len() {
                    assert!(
                        decode_query(&blob[..cut]).is_err(),
                        "kind {kind} cut at {cut}"
                    );
                }
            }
            // A line under WITHIN is refused, as Java's constructor refuses it.
            let mut within_line = Geo::new(kind, "shape");
            within_line
                .byte(1)
                .int(1)
                .byte(GEOMETRY_LINE)
                .ring(&[0.0, 1.0], &[0.0, 1.0]);
            assert_eq!(
                status(&within_line.blob()),
                invalid,
                "kind {kind}: a line under WITHIN"
            );
            let mut rel = Geo::new(kind, "shape");
            rel.byte(4).int(1).byte(GEOMETRY_POINT).f(0.0).f(0.0);
            assert_eq!(status(&rel.blob()), invalid, "unknown relation");
            let mut none = Geo::new(kind, "shape");
            none.byte(0).int(0);
            assert_eq!(status(&none.blob()), invalid, "no geometries");
            let mut tag = Geo::new(kind, "shape");
            tag.byte(0).int(1).byte(9);
            assert_eq!(status(&tag.blob()), invalid, "unknown geometry tag");
            let mut nested = Geo::new(kind, "shape");
            nested
                .byte(0)
                .int(1)
                .byte(GEOMETRY_POLYGON)
                .ring(&square_lats, &square_lons)
                .int(1);
            nested
                .ring(&hole_lats, &hole_lons)
                .int(1)
                .ring(&hole_lats, &hole_lons)
                .int(0);
            assert_eq!(status(&nested.blob()), invalid, "a hole with a hole");
            let mut open = Geo::new(kind, "shape");
            open.byte(0)
                .int(1)
                .byte(GEOMETRY_POLYGON)
                .ring(&[0.0, 0.0, 1.0, 1.0], &[0.0, 1.0, 1.0, 0.0])
                .int(0);
            assert_eq!(status(&open.blob()), invalid, "a ring that does not close");
            let mut huge = Geo::new(kind, "shape");
            huge.byte(0).int(1).byte(GEOMETRY_LINE).int(i32::MAX);
            assert_eq!(status(&huge.blob()), invalid, "more vertices than bytes");
            for (tag, bad) in [
                (GEOMETRY_POINT, vec![100.0, 0.0]),
                (GEOMETRY_RECTANGLE, vec![0.0, 1.0, 0.0, 200.0]),
                (GEOMETRY_CIRCLE, vec![0.0, 0.0, -5.0]),
            ] {
                let mut g = Geo::new(kind, "shape");
                g.byte(0).int(1).byte(tag);
                for v in bad {
                    g.f(v);
                }
                assert_eq!(status(&g.blob()), invalid, "invalid geometry {tag}");
            }
            let mut line = Geo::new(kind, "shape");
            line.byte(0).int(1).byte(GEOMETRY_LINE).ring(&[0.0], &[0.0]);
            assert_eq!(status(&line.blob()), invalid, "a one-point line");
        }
        // CONTAINS: points only for points; anything for shapes.
        let mut contains = Geo::new(NODE_GEO_POINT, "loc");
        contains.byte(3).int(1).byte(GEOMETRY_POINT).f(1.0).f(2.0);
        assert!(single(&contains.blob()).is_ok());
        let mut contains_poly = Geo::new(NODE_GEO_POINT, "loc");
        contains_poly
            .byte(3)
            .int(1)
            .byte(GEOMETRY_POLYGON)
            .ring(&square_lats, &square_lons)
            .int(0);
        assert_eq!(
            status(&contains_poly.blob()),
            invalid,
            "points CONTAINS a polygon"
        );
        let mut shape_contains = Geo::new(NODE_GEO_SHAPE, "shape");
        shape_contains
            .byte(3)
            .int(1)
            .byte(GEOMETRY_POLYGON)
            .ring(&square_lats, &square_lons)
            .int(0);
        assert!(single(&shape_contains.blob()).is_ok());

        // Shapes in a box, across the dateline; and an invalid box.
        let mut sbox = Geo::new(NODE_GEO_SHAPE_BOX, "shape");
        sbox.byte(2).f(-10.0).f(10.0).f(170.0).f(-170.0);
        assert!(single(&sbox.blob()).is_ok());
        let mut bad_box = Geo::new(NODE_GEO_SHAPE_BOX, "shape");
        bad_box.byte(0).f(-100.0).f(10.0).f(0.0).f(1.0);
        assert_eq!(status(&bad_box.blob()), invalid, "box past the pole");
        let mut bad_rel = Geo::new(NODE_GEO_SHAPE_BOX, "shape");
        bad_rel.byte(7).f(0.0).f(1.0).f(0.0).f(1.0);
        assert_eq!(status(&bad_rel.blob()), invalid, "box relation");
        let mut bad_field = Geo(vec![NODE_GEO_DISTANCE]);
        bad_field.bytes(&[0xff]).f(0.0).f(0.0).f(1.0);
        assert_eq!(
            decode_query(&bad_field.blob()).map(|_| ()),
            Err(FfiStatus::InvalidUtf8)
        );
        assert_eq!(status(&[QUERY_TREE, 20]), invalid, "unknown node kind");

        // Run against a real index without the fields: no hits, no error.
        let h = open();
        for blob in [distance(1e6).blob(), sbox.blob(), bx.blob()] {
            let (hits, total) = run(h, &blob, 10, true).unwrap();
            assert!(hits.is_empty() && total == 0);
        }
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    /// The fuzzer's `jvm_search` finding (seed `geo_distance_dateline`): a
    /// distance query centred on the dateline with a subnormal radius,
    /// whose bounding box crosses it a few quantization steps wide. Its
    /// grid's `int` arithmetic wraps past `i32::MAX` in Java; here it
    /// overflowed, and the caught panic came back as `FfiStatus::Panic`.
    #[test]
    fn geo_distance_on_the_dateline_answers_without_a_panic() {
        let input = include_bytes!("../fuzz/seeds/jvm_search/geo_distance_dateline");
        // The harness's framing: `top_n`, the count limit, then the blob.
        let (top_n, limit) = (usize::from(input[0] % 12), i64::from(input[1] % 12) - 1);
        let blob = &input[2..];
        assert_eq!(&blob[..2], &[QUERY_TREE, NODE_GEO_DISTANCE]);
        let h = open();
        // As found, the node is followed by bytes no query reads: an error.
        assert_eq!(
            run_limit(h, blob, top_n, limit),
            Err(FfiStatus::InvalidArgument.code())
        );
        assert!(crate::error::last_error().contains("trailing bytes"));
        // The node alone (kind, field `loc`, three `f64`s) is a valid query;
        // the fixture has no `loc` field, so it matches nothing.
        let node = &blob[..2 + 4 + 3 + 3 * 8];
        let (hits, total, _) = run_limit(h, node, top_n, limit)
            .unwrap_or_else(|rc| panic!("status {rc}: {}", crate::error::last_error()));
        assert!(hits.is_empty() && total == 0);
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    /// `ConstantScoreQuery` and `BoostQuery` -- what OpenSearch builds for a
    /// `term` on a keyword field and for a boosted `match` -- score as Lucene
    /// defines them: the constant, and the inner score times the boost.
    #[test]
    fn constant_score_and_boost_wrappers_score_as_lucene_defines() {
        let h = open();
        let one = 1.0f32.to_bits() as i32;
        let constant = bool_blob(0, &[(0, 2, -1, one, "", ""), (0, 0, 0, 0, "body", "fox")]);
        let (hits, total) = run(h, &constant, 10, true).unwrap();
        assert_eq!(
            hits,
            [(0, 1.0), (4, 1.0), (5, 1.0), (6, 1.0)],
            "constant score, doc order"
        );
        assert_eq!(total, 4);

        let two = 2.0f32.to_bits() as i32;
        let boosted = bool_blob(0, &[(0, 3, -1, two, "", ""), (0, 0, 0, 0, "body", "fox")]);
        let (hits, _) = run(h, &boosted, 10, true).unwrap();
        let (plain, _) = run(h, &term_blob("body", "fox"), 10, true).unwrap();
        assert_eq!(hits.len(), plain.len());
        for (b, p) in hits.iter().zip(&plain) {
            assert_eq!(b.0, p.0);
            assert!((b.1 - 2.0 * p.1).abs() <= 1e-6, "{b:?} vs 2 x {p:?}");
        }

        // A wrapper nested in a boolean: SHOULD constant(tag) + SHOULD fox.
        let nested = bool_blob(
            0,
            &[
                (2, 2, -1, one, "", ""),
                (0, 0, 0, 0, "body", "dog"),
                (2, 0, -1, 0, "body", "fox"),
            ],
        );
        let (hits, total) = run(h, &nested, 10, true).unwrap();
        assert_eq!(total, 8);
        let doc4 = hits.iter().find(|h| h.0 == 4).unwrap().1;
        assert!(
            (doc4 - (1.0 + f32::from_bits(1059136106))).abs() <= 1e-6,
            "{doc4}"
        );
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn wrapper_clauses_are_validated() {
        let invalid = Err(FfiStatus::InvalidArgument);
        let status = |b: &[u8]| decode_query(b).map(|_| ());
        let one = 1.0f32.to_bits() as i32;
        assert_eq!(
            status(&bool_blob(0, &[(0, 2, -1, one, "", "")])),
            invalid,
            "no child"
        );
        assert_eq!(
            status(&bool_blob(
                0,
                &[
                    (0, 3, -1, one, "", ""),
                    (0, 0, 0, 0, "f", "a"),
                    (0, 0, 0, 0, "f", "b")
                ]
            )),
            invalid,
            "two children"
        );
        assert_eq!(
            status(&bool_blob(
                0,
                &[(0, 2, -1, one, "", ""), (2, 0, 0, 0, "f", "a")]
            )),
            invalid,
            "a SHOULD child"
        );
        for bad in [f32::NAN, f32::INFINITY, -1.0] {
            assert_eq!(
                status(&bool_blob(
                    0,
                    &[
                        (0, 3, -1, bad.to_bits() as i32, "", ""),
                        (0, 0, 0, 0, "f", "a")
                    ]
                )),
                invalid,
                "boost {bad}"
            );
        }
        assert_eq!(
            status(&bool_blob(0, &[(0, 4, -1, 0, "", "")])),
            invalid,
            "unknown kind"
        );
        // Wrappers nest in each other and in booleans.
        let two = 2.0f32.to_bits() as i32;
        let ok = bool_blob(
            0,
            &[
                (0, 3, -1, two, "", ""),
                (0, 2, 0, one, "", ""),
                (0, 1, 1, 0, "", ""),
                (2, 0, 2, 0, "f", "a"),
            ],
        );
        let JvmQuery::Boolean(q) = decode_query(&ok).unwrap() else {
            panic!("expected a boolean");
        };
        let lucene_search::query::Clause::Boost(b) = &q.must[0] else {
            panic!("expected a boost");
        };
        assert_eq!(b.boost, 2.0);
        assert!(matches!(
            &*b.inner,
            lucene_search::query::Clause::ConstantScore(_)
        ));
        // A constant score's inner query never scores: no norms for it.
        assert!(lucene_engine::clauses::clause_field_names(&q).is_empty());
    }

    #[test]
    fn decode_query_builds_nested_booleans() {
        // +(a b)~1 -c
        let blob = bool_blob(
            0,
            &[
                (0, 1, -1, 1, "", ""),
                (2, 0, 0, 0, "f", "a"),
                (2, 0, 0, 0, "f", "b"),
                (3, 0, -1, 0, "f", "c"),
            ],
        );
        let JvmQuery::Boolean(q) = decode_query(&blob).unwrap() else {
            panic!("expected a boolean");
        };
        assert_eq!(q.must.len(), 1);
        assert_eq!(q.must_not.len(), 1);
        let lucene_search::query::Clause::Boolean(inner) = &q.must[0] else {
            panic!("expected a nested boolean");
        };
        assert_eq!(inner.should.len(), 2);
        assert_eq!(inner.minimum_should_match, 1);
        assert!(matches!(
            decode_query(&term_blob("f", "t")).unwrap(),
            JvmQuery::Term(_)
        ));
    }

    // ---- the query-tree blob ---------------------------------------------

    /// A query-tree node, for building blobs in tests.
    enum N<'a> {
        T(&'a str, &'a str),
        B(i32, Vec<(u8, N<'a>)>),
        C(f32, Box<N<'a>>),
        Boost(f32, Box<N<'a>>),
        D(f32, Vec<N<'a>>),
        All,
        None,
        /// `(field, slop, [(position, term)])`.
        P(&'a str, i32, Vec<(i32, &'a str)>),
        Ts(&'a str, Vec<&'a str>),
        R(&'a str, i64, i64),
        Pre(&'a str, &'a str),
        Wc(&'a str, &'a str),
        Re(&'a str, &'a str),
        Ex(&'a str),
        Ts1(&'a str, &'a str, i64),
        /// `(field, lower, upper, include_lower, include_upper)`.
        Tr(&'a str, Option<&'a str>, Option<&'a str>, bool, bool),
    }

    fn enc(n: &N<'_>, b: &mut Vec<u8>) {
        let bytes = |b: &mut Vec<u8>, x: &[u8]| {
            b.extend_from_slice(&(x.len() as i32).to_le_bytes());
            b.extend_from_slice(x);
        };
        match n {
            N::T(f, t) => {
                b.push(NODE_TERM);
                bytes(b, f.as_bytes());
                bytes(b, t.as_bytes());
            }
            N::B(msm, clauses) => {
                b.push(NODE_BOOLEAN);
                b.extend_from_slice(&msm.to_le_bytes());
                b.extend_from_slice(&(clauses.len() as i32).to_le_bytes());
                for (occur, c) in clauses {
                    b.push(*occur);
                    enc(c, b);
                }
            }
            N::C(score, c) => {
                b.push(NODE_CONSTANT_SCORE);
                b.extend_from_slice(&score.to_bits().to_le_bytes());
                enc(c, b);
            }
            N::Boost(boost, c) => {
                b.push(NODE_BOOST);
                b.extend_from_slice(&boost.to_bits().to_le_bytes());
                enc(c, b);
            }
            N::D(tie, ds) => {
                b.push(NODE_DISMAX);
                b.extend_from_slice(&tie.to_bits().to_le_bytes());
                b.extend_from_slice(&(ds.len() as i32).to_le_bytes());
                for d in ds {
                    enc(d, b);
                }
            }
            N::All => b.push(NODE_MATCH_ALL),
            N::None => b.push(NODE_MATCH_NONE),
            N::R(field, min, max) => {
                b.push(NODE_POINT_RANGE);
                bytes(b, field.as_bytes());
                b.extend_from_slice(&min.to_le_bytes());
                b.extend_from_slice(&max.to_le_bytes());
            }
            N::Ts(field, terms) => {
                b.push(NODE_TERM_SET);
                bytes(b, field.as_bytes());
                b.extend_from_slice(&(terms.len() as i32).to_le_bytes());
                for t in terms {
                    bytes(b, t.as_bytes());
                }
            }
            N::Pre(field, prefix) => {
                b.push(NODE_PREFIX);
                bytes(b, field.as_bytes());
                bytes(b, prefix.as_bytes());
            }
            N::Wc(field, pattern) => {
                b.push(NODE_WILDCARD);
                bytes(b, field.as_bytes());
                bytes(b, pattern.as_bytes());
            }
            N::Ts1(field, term, df) => {
                b.push(NODE_TERM_STATS);
                bytes(b, field.as_bytes());
                bytes(b, term.as_bytes());
                b.extend_from_slice(&df.to_le_bytes());
            }
            N::Ex(field) => {
                b.push(NODE_EXISTS);
                bytes(b, field.as_bytes());
            }
            N::Re(field, pattern) => {
                b.push(NODE_REGEXP);
                bytes(b, field.as_bytes());
                bytes(b, pattern.as_bytes());
            }
            N::Tr(field, lower, upper, include_lower, include_upper) => {
                b.push(NODE_TERM_RANGE);
                bytes(b, field.as_bytes());
                let flags = u8::from(lower.is_some())
                    | (u8::from(upper.is_some()) << 1)
                    | (u8::from(*include_lower) << 2)
                    | (u8::from(*include_upper) << 3);
                b.push(flags);
                for bound in [lower, upper].into_iter().flatten() {
                    bytes(b, bound.as_bytes());
                }
            }
            N::P(field, slop, terms) => {
                b.push(NODE_PHRASE);
                bytes(b, field.as_bytes());
                b.extend_from_slice(&slop.to_le_bytes());
                b.extend_from_slice(&(terms.len() as i32).to_le_bytes());
                for (position, term) in terms {
                    b.extend_from_slice(&position.to_le_bytes());
                    bytes(b, term.as_bytes());
                }
            }
        }
    }

    /// A prefix query's blob.
    pub(crate) fn prefix_blob(field: &str, prefix: &str) -> Vec<u8> {
        tree(N::Pre(field, prefix))
    }

    /// `+field:term #range(numeric, min, max)`'s blob.
    pub(crate) fn term_and_range_blob(
        field: &str,
        term: &str,
        numeric: &str,
        min: i64,
        max: i64,
    ) -> Vec<u8> {
        tree(N::B(
            0,
            vec![
                (lucene_engine::clauses::OCCUR_MUST, N::T(field, term)),
                (
                    lucene_engine::clauses::OCCUR_FILTER,
                    N::R(numeric, min, max),
                ),
            ],
        ))
    }

    /// `#range(numeric, min, max)`'s blob.
    pub(crate) fn range_blob(numeric: &str, min: i64, max: i64) -> Vec<u8> {
        tree(N::R(numeric, min, max))
    }

    fn tree(n: N<'_>) -> Vec<u8> {
        let mut b = vec![QUERY_TREE];
        enc(&n, &mut b);
        b
    }

    /// `#w0 ?(w5 w6)` as the JVM might send it, un-rewritten: the bridge
    /// rewrites it to `#w0 ?w5 ?w6`, so the pruned run's top ten and its
    /// lower-bound total are Lucene's (`GenMixedBooleanScoring`'s query 16:
    /// 10,343, where the nested disjunction scored as written counts 177).
    /// A lone `?t` rewrites to the term query itself, which scores as the
    /// term blob does.
    #[test]
    fn a_request_is_rewritten_before_it_is_scored() {
        let h = open_fixture("mixed_boolean_scoring_index");
        let nested = tree(N::B(
            0,
            vec![
                (1, N::T("body", "w0")),
                (
                    2,
                    N::B(0, vec![(2, N::T("body", "w5")), (2, N::T("body", "w6"))]),
                ),
            ],
        ));
        let (hits, total, lower) = run_limit(h, &nested, 10, 100).unwrap();
        let docs: Vec<i32> = hits.iter().map(|&(d, _)| d).collect();
        assert_eq!(
            docs,
            [9156, 17605, 7376, 12786, 13304, 22369, 13455, 15217, 15458, 16280]
        );
        assert_eq!(hits[0].1.to_bits(), 1_071_013_260);
        assert_eq!((total, lower), (10_343, true));

        let lone = tree(N::B(0, vec![(2, N::T("body", "w5"))]));
        let JvmQuery::Boolean(b) = decode_request(&lone).unwrap().0 else {
            panic!("a rewritten tree stays a boolean");
        };
        assert!(matches!(b.must[..], [Clause::Term(_)]) && b.should.is_empty());
        let (via_bool, ..) = run_limit(h, &lone, 10, i64::MAX).unwrap();
        let (via_term, ..) = run_limit(h, &term_blob("body", "w5"), 10, i64::MAX).unwrap();
        assert_eq!(via_bool, via_term);

        // A pure negative query rewrites to `MatchNoDocsQuery`, kept as the
        // boolean's one required clause: no hits, and a count of zero.
        let negative = tree(N::B(0, vec![(3, N::T("body", "w5"))]));
        let JvmQuery::Boolean(b) = decode_request(&negative).unwrap().0 else {
            panic!("a pure negative query stays a boolean");
        };
        assert!(matches!(b.must[..], [Clause::MatchNoDocs(_)]));
        assert_eq!(
            run_limit(h, &negative, 10, i64::MAX).unwrap(),
            (vec![], 0, false)
        );
        ffi_close_jvm_reader(h);
    }

    /// The tree blob and the clause-list blob describe the same queries, so
    /// they must search identically: same hits, same score bits, same count.
    #[test]
    fn a_query_tree_searches_exactly_like_the_clause_list() {
        let h = open();
        let pairs = [
            (
                tree(N::B(
                    0,
                    vec![(0, N::T("body", "fox")), (2, N::T("body", "dog"))],
                )),
                bool_blob(
                    0,
                    &[(0, 0, -1, 0, "body", "fox"), (2, 0, -1, 0, "body", "dog")],
                ),
            ),
            (
                tree(N::B(
                    0,
                    vec![(2, N::T("body", "dog")), (3, N::T("body", "fox"))],
                )),
                bool_blob(
                    0,
                    &[(2, 0, -1, 0, "body", "dog"), (3, 0, -1, 0, "body", "fox")],
                ),
            ),
            (
                tree(N::Boost(2.0, Box::new(N::T("body", "fox")))),
                bool_blob(
                    0,
                    &[
                        (0, 3, -1, 2.0f32.to_bits() as i32, "", ""),
                        (0, 0, 0, 0, "body", "fox"),
                    ],
                ),
            ),
            (
                tree(N::C(1.0, Box::new(N::T("body", "dog")))),
                bool_blob(
                    0,
                    &[
                        (0, 2, -1, 1.0f32.to_bits() as i32, "", ""),
                        (0, 0, 0, 0, "body", "dog"),
                    ],
                ),
            ),
        ];
        for (t, b) in &pairs {
            let a = run(h, t, 10, true).unwrap();
            let e = run(h, b, 10, true).unwrap();
            assert_eq!(a.1, e.1, "count");
            assert_eq!(a.0.len(), e.0.len());
            for (x, y) in a.0.iter().zip(&e.0) {
                assert_eq!((x.0, x.1.to_bits()), (y.0, y.1.to_bits()));
            }
            // `size: 0`, the counting path.
            assert_eq!(run(h, t, 0, true).unwrap().1, e.1);
        }
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    /// `match_all` covers every live document of every segment -- each
    /// segment's own `maxDoc`, which the blob does not carry -- `match_none`
    /// none, and dismax scores as `max + tie * rest`.
    #[test]
    fn phrase_trees_search_and_count() {
        let h = open();
        let cat_dog = || N::P("body", 0, vec![(0, "cat"), (1, "dog")]);
        let (hits, total) = run(h, &tree(cat_dog()), 10, true).unwrap();
        let mut docs: Vec<i32> = hits.iter().map(|&(d, _)| d).collect();
        docs.sort_unstable();
        assert!(docs.starts_with(&[1, 2]), "both short documents: {docs:?}");
        assert_eq!(total, hits.len() as i64);
        // The reversed phrase is nowhere in the short documents.
        let dog_cat = tree(N::P("body", 0, vec![(0, "dog"), (1, "cat")]));
        let (reversed, _) = run(h, &dog_cat, 10, true).unwrap();
        assert!(reversed.iter().all(|&(d, _)| d >= 4), "{reversed:?}");
        // Inside a tree: `cat` except where `cat dog` occurs, and the count
        // path agrees with the search.
        let except = tree(N::B(0, vec![(0, N::T("body", "cat")), (3, cat_dog())]));
        let (hits, total) = run(h, &except, 10, true).unwrap();
        assert!(hits.iter().all(|&(d, _)| d != 1 && d != 2));
        assert_eq!(run(h, &except, 0, true).unwrap().1, total);
        // A sloppy phrase matches the reversed order within slop 2.
        let sloppy = tree(N::P("body", 2, vec![(0, "dog"), (1, "cat")]));
        let (hits, _) = run(h, &sloppy, 10, true).unwrap();
        assert!(hits.iter().any(|&(d, _)| d == 1 || d == 2), "{hits:?}");
    }

    #[test]
    fn a_term_with_its_own_doc_freq_scores_from_it() {
        let h = open();
        let plain = tree(N::B(
            0,
            vec![(2, N::T("body", "fox")), (2, N::T("body", "dog"))],
        ));
        let (want, want_total) = run(h, &plain, 10, true).unwrap();
        let df = doc_freq(h, b"body", b"fox").unwrap();
        // Its own docFreq equal to the reader's: the same hits, bit for bit.
        let same = tree(N::B(
            0,
            vec![(2, N::Ts1("body", "fox", df)), (2, N::T("body", "dog"))],
        ));
        let (got, total) = run(h, &same, 10, true).unwrap();
        assert_eq!(total, want_total);
        let bits = |v: &[(i32, f32)]| v.iter().map(|&(d, s)| (d, s.to_bits())).collect::<Vec<_>>();
        assert_eq!(bits(&got), bits(&want));
        // A rarer one: the same documents, and fox's weigh more.
        let rare = tree(N::Ts1("body", "fox", 1));
        let (rare_hits, _) = run(h, &rare, 10, true).unwrap();
        let (plain_hits, _) =
            run(h, &tree(N::B(0, vec![(2, N::T("body", "fox"))])), 10, true).unwrap();
        let docs = |v: &[(i32, f32)]| {
            let mut d: Vec<i32> = v.iter().map(|&(d, _)| d).collect();
            d.sort_unstable();
            d
        };
        assert_eq!(docs(&rare_hits), docs(&plain_hits));
        assert!(rare_hits[0].1 > plain_hits[0].1);
        // A docFreq below 1 is refused.
        assert_eq!(
            run(h, &tree(N::Ts1("body", "fox", 0)), 10, true).err(),
            Some(FfiStatus::InvalidArgument.code())
        );
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn multi_term_trees_match_their_terms_at_a_constant_score() {
        let h = open();
        let docs = |blob: &[u8]| {
            let (hits, total) = run(h, blob, 10, true).unwrap();
            assert!(
                hits.iter().all(|&(_, s)| s == 1.0),
                "constant scores: {hits:?}"
            );
            let mut d: Vec<i32> = hits.iter().map(|&(d, _)| d).collect();
            d.sort_unstable();
            assert_eq!(total, d.len() as i64);
            d
        };
        let dog = docs(&tree(N::C(1.0, Box::new(N::T("body", "dog")))));
        let fox = docs(&tree(N::C(1.0, Box::new(N::T("body", "fox")))));
        assert_eq!(docs(&tree(N::Pre("body", "do"))), dog);
        assert_eq!(docs(&tree(N::Wc("body", "d?g"))), dog);
        assert_eq!(docs(&tree(N::Re("body", "d[aeiou]g"))), dog);
        assert_eq!(docs(&tree(N::Re("body", "(dog|nosuch)"))), dog);
        assert_eq!(docs(&tree(N::Re("body", "#"))), Vec::<i32>::new());
        // A term range, each end open, closed or exclusive.
        assert_eq!(
            docs(&tree(N::Tr("body", Some("dog"), Some("dog"), true, true))),
            dog
        );
        assert_eq!(
            docs(&tree(N::Tr("body", Some("dog"), Some("dog"), false, true))),
            Vec::<i32>::new()
        );
        assert_eq!(
            docs(&tree(N::Tr("body", Some("do"), Some("doh"), false, false))),
            dog
        );
        let all_body = docs(&tree(N::Tr("body", None, None, false, false)));
        assert_eq!(all_body, docs(&tree(N::Ex("body"))));
        let below = docs(&tree(N::Tr("body", None, Some("dog"), false, false)));
        let above = docs(&tree(N::Tr("body", Some("dog"), None, true, false)));
        assert!(below.iter().chain(&above).all(|d| all_body.contains(d)));
        assert!(above.iter().any(|d| dog.contains(d)));
        // An unknown flag bit fails the decode.
        let mut bad = tree(N::Tr("body", None, None, false, false));
        *bad.last_mut().unwrap() = 0x10;
        assert_eq!(
            decode_query(&bad).map(|_| ()),
            Err(FfiStatus::InvalidArgument)
        );
        assert!(crate::error::last_error().contains("term range flags"));
        // exists: every document has a body (its norms, dense here); a field
        // no segment has matches nothing. The sparse and doc-values sources
        // are `weight_count_fixtures`' against real Lucene.
        let all = docs(&tree(N::All));
        assert_eq!(docs(&tree(N::Ex("body"))), all);
        assert_eq!(docs(&tree(N::Ex("no_such_field"))), Vec::<i32>::new());
        // A pattern the grammar rejects fails the decode, not the search.
        assert_eq!(
            run(h, &tree(N::Re("body", "d[og")), 10, true).err(),
            Some(FfiStatus::InvalidArgument.code())
        );
        let mut either = [dog.clone(), fox].concat();
        either.sort_unstable();
        either.dedup();
        assert_eq!(
            docs(&tree(N::Ts("body", vec!["fox", "dog", "nosuch"]))),
            either
        );
        assert_eq!(docs(&tree(N::Ts("body", vec![]))), Vec::<i32>::new());
        // Under a boost, the boost is the score.
        let (hits, _) = run(
            h,
            &tree(N::Boost(2.0, Box::new(N::Pre("body", "do")))),
            10,
            true,
        )
        .unwrap();
        assert!(hits.iter().all(|&(_, s)| s == 2.0));
        let long: Vec<&str> = (0..1100).map(|_| "t").collect();
        assert_eq!(
            decode_query(&tree(N::Ts("body", long))).map(|_| ()),
            Err(FfiStatus::InvalidArgument),
            "a term set over the clause cap"
        );
    }

    #[test]
    fn a_points_range_opens_points_and_matches_nothing_where_there_are_none() {
        let h = open();
        // This fixture indexes no points: every segment gets an empty reader.
        let range = tree(N::R("n", 0, 10));
        assert_eq!(run(h, &range, 10, true).unwrap(), (vec![], 0));
        let nested = tree(N::B(
            0,
            vec![(0, N::T("body", "dog")), (3, N::R("n", 0, 10))],
        ));
        let (with_range, _) = run(h, &nested, 10, true).unwrap();
        let (dog, _) = run(h, &term_blob("body", "dog"), 10, true).unwrap();
        assert_eq!(with_range, dog, "excluding nothing");
        let mut short = vec![QUERY_TREE, NODE_POINT_RANGE];
        short.extend_from_slice(&1i32.to_le_bytes());
        short.push(b'n');
        short.extend_from_slice(&[0; 5]);
        assert_eq!(
            decode_query(&short).map(|_| ()),
            Err(FfiStatus::InvalidArgument),
            "truncated"
        );
    }

    #[test]
    fn malformed_phrase_nodes_are_invalid_arguments() {
        let invalid = Err(FfiStatus::InvalidArgument);
        let status = |b: &[u8]| decode_query(b).map(|_| ());
        assert_eq!(status(&tree(N::P("body", 0, vec![]))), invalid, "no terms");
        assert_eq!(
            status(&tree(N::P("body", 0, vec![(0, "a"), (2, "b")]))),
            invalid,
            "a position gap"
        );
        assert_eq!(
            status(&tree(N::P("body", -1, vec![(0, "a"), (1, "b")]))),
            invalid,
            "negative slop"
        );
        let long: Vec<(i32, &str)> = (0..1100).map(|i| (i, "t")).collect();
        assert_eq!(
            status(&tree(N::P("body", 0, long))),
            invalid,
            "over the clause cap"
        );
        assert_eq!(
            status(&tree(N::P("body", 0, vec![(0, "a"), (1, "b")]))),
            Ok(())
        );
    }

    /// A phrase over a field indexed without positions is `PhraseWeight`'s
    /// `IllegalStateException` at this boundary too: a search error carrying
    /// Lucene's message (`GenSandboxQueries`' `errors.tsv` pins it against
    /// Java), from the search and the count alike, and never a panic. The
    /// plugin re-runs a failed native search on Lucene, which throws it.
    #[test]
    fn a_phrase_over_a_field_without_positions_is_a_search_error() {
        let h = open_fixture("sandbox_queries/index");
        // The status's own prefix, then Lucene's message.
        let want = "search failed: field \"id\" was indexed without position data; \
                    cannot run PhraseQuery (phrase=id:\"1 2\")";
        let phrase = || N::P("id", 0, vec![(0, "1"), (1, "2")]);
        for (top_n, count) in [(10, true), (0, true), (10, false)] {
            assert_eq!(
                run(h, &tree(phrase()), top_n, count).err(),
                Some(FfiStatus::Search.code()),
                "top_n {top_n}, count {count}"
            );
            assert_eq!(crate::error::last_error(), want);
        }
        // Under a boolean, and over a term no segment holds: the field's
        // missing positions are checked before its terms are looked up.
        let nested = tree(N::B(0, vec![(2, N::T("body", "w1")), (2, phrase())]));
        assert_eq!(
            run(h, &nested, 10, true).err(),
            Some(FfiStatus::Search.code())
        );
        assert_eq!(crate::error::last_error(), want);
        let absent = tree(N::P("id", 0, vec![(0, "zz"), (1, "yy")]));
        assert_eq!(
            run(h, &absent, 10, true).err(),
            Some(FfiStatus::Search.code())
        );
        // One term rewrites to a term query, which needs no positions.
        assert!(run(h, &tree(N::P("id", 0, vec![(0, "1")])), 10, true).is_ok());
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn match_all_counts_the_live_documents_of_each_segment() {
        // Segment 0 has document 2 deleted; the match-all spans each
        // segment's own maxDoc and skips it.
        let (rc, h) = open_live(&[4, 4], 0, &[&[0b1011]]);
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        let (hits, total) = run(h, &tree(N::All), 10, true).unwrap();
        assert_eq!(total, 7);
        let docs: Vec<i32> = hits.iter().map(|&(d, _)| d).collect();
        assert_eq!(docs, [0, 1, 3, 4, 5, 6, 7]);
        let (_, count) = run(h, &tree(N::C(1.0, Box::new(N::All))), 0, true).unwrap();
        assert_eq!(count, 7, "the count-only path agrees");
    }

    #[test]
    fn match_all_match_none_and_dismax_trees() {
        let h = open();
        let (hits, total) = run(h, &tree(N::All), 10, true).unwrap();
        assert_eq!(total, 8, "two segments of four documents");
        assert_eq!(hits.len(), 8);
        assert!(hits.iter().all(|&(_, s)| s == 1.0));
        assert_eq!(run(h, &tree(N::All), 0, true).unwrap().1, 8);
        let boosted_all = tree(N::Boost(3.0, Box::new(N::All)));
        assert!(run(h, &boosted_all, 3, true)
            .unwrap()
            .0
            .iter()
            .all(|&(_, s)| s == 3.0));
        assert_eq!(run(h, &tree(N::None), 10, true).unwrap(), (vec![], 0));
        assert_eq!(
            run(h, &tree(N::B(0, vec![])), 10, true).unwrap(),
            (vec![], 0)
        );
        assert_eq!(
            run(h, &tree(N::B(0, vec![(3, N::T("body", "fox"))])), 10, true).unwrap(),
            (vec![], 0),
            "a pure negative boolean matches nothing"
        );
        let (fox, _) = run(h, &term_blob("body", "fox"), 10, true).unwrap();
        let (dog, _) = run(h, &term_blob("body", "dog"), 10, true).unwrap();
        let (dm, total) = run(
            h,
            &tree(N::D(0.5, vec![N::T("body", "fox"), N::T("body", "dog")])),
            10,
            true,
        )
        .unwrap();
        let mut docs: Vec<i32> = fox.iter().chain(&dog).map(|h| h.0).collect();
        docs.sort_unstable();
        docs.dedup();
        assert_eq!(total, docs.len() as i64);
        for (doc, score) in dm {
            let f = fox.iter().find(|h| h.0 == doc).map_or(0.0, |h| h.1);
            let d = dog.iter().find(|h| h.0 == doc).map_or(0.0, |h| h.1);
            let (hi, lo) = if f >= d { (f, d) } else { (d, f) };
            let want = (f64::from(hi) + f64::from(lo) * 0.5) as f32;
            assert_eq!(score.to_bits(), want.to_bits(), "doc {doc}");
        }
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn malformed_query_trees_are_invalid_arguments() {
        let invalid = Err(FfiStatus::InvalidArgument);
        let status = |b: &[u8]| decode_query(b).map(|_| ());
        assert_eq!(status(&[QUERY_TREE, 99]), invalid, "unknown node kind");
        let mut bad_occur = vec![QUERY_TREE, NODE_BOOLEAN];
        bad_occur.extend_from_slice(&0i32.to_le_bytes());
        bad_occur.extend_from_slice(&1i32.to_le_bytes());
        bad_occur.push(7);
        enc(&N::All, &mut bad_occur);
        assert_eq!(status(&bad_occur), invalid, "unknown occur");
        for bad in [f32::NAN, f32::INFINITY, -1.0] {
            assert_eq!(
                status(&tree(N::Boost(bad, Box::new(N::All)))),
                invalid,
                "boost {bad}"
            );
        }
        assert_eq!(
            status(&tree(N::D(1.5, vec![N::All]))),
            invalid,
            "tie breaker above 1"
        );
        // The Java encoder's `MAX_DEPTH`: the root at depth 0, so 32 levels
        // decode and a 33rd does not.
        let chain = |levels: usize| {
            let mut n = N::All;
            for _ in 1..levels {
                n = N::Boost(1.0, Box::new(n));
            }
            tree(n)
        };
        assert_eq!(status(&chain(32)), Ok(()), "32 levels");
        assert_eq!(status(&chain(33)), invalid, "33 levels is too deep");
        // And its `MAX_NODES`: 1024 nodes decode, 1025 do not.
        let wide = |nodes: usize| tree(N::B(0, (1..nodes).map(|_| (2, N::All)).collect()));
        assert_eq!(status(&wide(1024)), Ok(()), "1024 nodes");
        assert_eq!(status(&wide(1025)), invalid, "1025 nodes");
        let mut negative_msm = vec![QUERY_TREE, NODE_BOOLEAN];
        negative_msm.extend_from_slice(&(-1i32).to_le_bytes());
        assert_eq!(
            status(&negative_msm),
            invalid,
            "negative minimum_should_match"
        );
        let mut bad_utf8 = vec![QUERY_TREE, NODE_TERM];
        bad_utf8.extend_from_slice(&1i32.to_le_bytes());
        bad_utf8.push(0xff);
        bad_utf8.extend_from_slice(&0i32.to_le_bytes());
        assert_eq!(status(&bad_utf8), Err(FfiStatus::InvalidUtf8));
        let mut trailing = tree(N::All);
        trailing.push(0);
        assert_eq!(status(&trailing), invalid, "trailing bytes");
    }

    fn collapse_blob(field: &str, kind: u8) -> Vec<u8> {
        let mut b = (field.len() as i32).to_le_bytes().to_vec();
        b.extend_from_slice(field.as_bytes());
        b.push(kind);
        b
    }

    /// A collapsed answer's one slice: `(total, max_score, [(doc, keys, value)])`
    /// with `string` saying which keys are terms, and the terminated flag.
    #[allow(clippy::type_complexity)]
    fn parse_collapsed(
        out: &[u8],
        string: &[bool],
    ) -> (i64, f32, Vec<(i32, Vec<i64>, Option<Vec<u8>>)>, bool) {
        let mut c = Cursor { buf: out, pos: 0 };
        assert_eq!(c.i32().unwrap(), 1);
        let total = c.i64().unwrap();
        let max = f32::from_bits(c.i32().unwrap() as u32);
        let n = c.i32().unwrap();
        let mut hits = Vec::new();
        for _ in 0..n {
            let doc = c.i32().unwrap();
            let mut keys = Vec::new();
            for &is_term in string {
                if is_term {
                    let len = c.i32().unwrap();
                    if len >= 0 {
                        c.pos += len as usize;
                    }
                    keys.push(i64::from(len));
                } else {
                    keys.push(c.i64().unwrap());
                }
            }
            let value = match c.u8().unwrap() {
                COLLAPSE_VALUE_MISSING => None,
                COLLAPSE_VALUE_LONG => Some(c.i64().unwrap().to_le_bytes().to_vec()),
                COLLAPSE_VALUE_BYTES => Some(c.bytes().unwrap().to_vec()),
                other => panic!("value kind {other}"),
            };
            hits.push((doc, keys, value));
        }
        let terminated = c.u8().unwrap() != 0;
        assert_eq!(c.pos, out.len());
        (total, max, hits, terminated)
    }

    #[test]
    fn collapsed_searches_keep_a_group_per_value() {
        let h = open_fixture("sorted_search_index");
        let q = term_blob("body", "w0");
        let by_l = sort_blob(&[(SORT_LONG, 0, "l", 0)], None);
        let on_i = collapse_blob("i", COLLAPSE_NUMERIC);
        let out = search_collapsed_blobs(h, &q, &by_l, &on_i, 10).unwrap();
        let (total, max, hits, terminated) = parse_collapsed(&out, &[false]);
        assert!(total > 10 && max.is_nan() && !terminated);
        assert_eq!(hits.len(), 10);
        let mut values: Vec<_> = hits.iter().map(|h| h.2.clone()).collect();
        values.dedup();
        assert_eq!(values.len(), 10, "one hit per group");
        assert!(
            hits.windows(2).all(|w| w[0].1[0] <= w[1].1[0]),
            "sorted by l"
        );
        // Behind terminate_after: the first five matches only, stopped early.
        let five = with_terminate_after(by_l.clone(), SORT_TERMINATE_AFTER, 5);
        let out = search_collapsed_blobs(h, &q, &five, &on_i, 10).unwrap();
        let (total, _, hits, terminated) = parse_collapsed(&out, &[false]);
        assert_eq!(total, 5);
        assert!(terminated && hits.len() <= 5);
        let all = with_terminate_after(by_l.clone(), SORT_TERMINATE_AFTER, i32::MAX);
        let (whole, ..) = parse_collapsed(
            &search_collapsed_blobs(h, &q, &by_l, &on_i, 3).unwrap(),
            &[false],
        );
        let (until, _, _, stopped) = parse_collapsed(
            &search_collapsed_blobs(h, &q, &all, &on_i, 3).unwrap(),
            &[false],
        );
        assert_eq!((until, stopped), (whole, false));
        // Scores tracked behind another key.
        let tracked = with_terminate_after(by_l.clone(), SORT_TRACK_MAX_SCORE, 0);
        let (_, max, ..) = parse_collapsed(
            &search_collapsed_blobs(h, &q, &tracked, &on_i, 3).unwrap(),
            &[false],
        );
        assert!(max > 0.0);
        let invalid = Err(FfiStatus::InvalidArgument);
        let counted =
            with_terminate_after(by_l.clone(), SORT_TERMINATE_AFTER | SORT_COUNT_SEGMENTS, 5);
        assert_eq!(search_collapsed_blobs(h, &q, &counted, &on_i, 10), invalid);
        assert_eq!(search_collapsed_blobs(h, &q, &by_l, &on_i, 0), invalid);
        assert_eq!(
            search_collapsed_blobs(h, &q, &by_l, &collapse_blob("i", 9), 5),
            invalid
        );
        let mut trailing = on_i.clone();
        trailing.push(0);
        assert_eq!(search_collapsed_blobs(h, &q, &by_l, &trailing, 5), invalid);
        // A multi-valued field: the collector's own refusal.
        assert!(
            search_collapsed_blobs(h, &q, &by_l, &collapse_blob("m", COLLAPSE_NUMERIC), 50)
                .is_err()
        );
        assert!(crate::error::last_error().contains("failed to collapse"));
        assert_eq!(
            search_collapsed_blobs(u64::MAX, &q, &by_l, &on_i, 5),
            Err(FfiStatus::InvalidHandle)
        );
        assert_eq!(ffi_close_jvm_reader(h), 0);

        let h = open_fixture("keyword_sort_index");
        let all = tree(N::All);
        let by_k = sort_blob(&[(SORT_STRING, 0, "kd", 1)], None);
        let out = search_collapsed_blobs(h, &all, &by_k, &collapse_blob("k", COLLAPSE_KEYWORD), 4)
            .unwrap();
        let (_, _, hits, _) = parse_collapsed(&out, &[true]);
        assert_eq!(hits.len(), 4);
        assert!(hits.iter().filter(|h| h.2.is_some()).count() >= 3);
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    fn opensearch_key(ty: u8, flags: u8, field: &str, missing: i64) -> Vec<u8> {
        let mut b = vec![1, ty, flags];
        if ty == SORT_SHARD_DOC {
            b.extend_from_slice(&missing.to_le_bytes());
        } else {
            b.extend_from_slice(&(field.len() as i32).to_le_bytes());
            b.extend_from_slice(field.as_bytes());
            b.extend_from_slice(&missing.to_le_bytes());
        }
        b.push(0);
        b.push(0);
        b.extend_from_slice(&0i32.to_le_bytes());
        b
    }

    #[test]
    fn opensearchs_own_sort_keys_decode_and_sort() {
        let prefix = 2i64 << 32;
        let shard = opensearch_key(SORT_SHARD_DOC, SORT_REVERSE, "", prefix);
        let (keys, ..) = decode_sort(&shard).unwrap();
        assert!(matches!(keys[0].ty, SortType::Custom(_)) && keys[0].reverse);
        let half = lucene_util::numeric_utils::float_to_sortable_int(1.5);
        for (ty, flags, missing) in [
            (SORT_HALF_FLOAT, 0, i64::from(half)),
            (SORT_HALF_FLOAT, SORT_MAX, 0),
            (SORT_HALF_FLOAT, SORT_MODE_MEDIAN, 0),
            (SORT_UNSIGNED_LONG, SORT_MODE_SUM, -1),
            (SORT_UNSIGNED_LONG, SORT_MODE_AVG | SORT_REVERSE, 0),
        ] {
            assert!(
                decode_sort(&opensearch_key(ty, flags, "l", missing)).is_ok(),
                "{ty} {flags}"
            );
        }
        let invalid = Some(FfiStatus::InvalidArgument);
        for blob in [
            opensearch_key(SORT_SHARD_DOC, SORT_MAX, "", 0),
            opensearch_key(SORT_UNSIGNED_LONG, SORT_MODE_MEDIAN, "l", 0),
            opensearch_key(SORT_HALF_FLOAT, SORT_MAX | SORT_MODE_SUM, "l", 0),
            opensearch_key(SORT_HALF_FLOAT, SORT_NESTED, "l", 0),
            opensearch_key(SORT_HALF_FLOAT, 0, "l", i64::MAX),
        ] {
            assert_eq!(decode_sort(&blob).err(), invalid, "{blob:?}");
        }
        let h = open_fixture("sorted_search_index");
        let q = term_blob("body", "w1");
        let out = search_sorted_blobs(h, &q, &shard, 5, i64::MAX).unwrap();
        assert_eq!(out.hits.len(), 5);
        assert!(out.hits.windows(2).all(|w| w[0].doc > w[1].doc));
        assert!(out
            .hits
            .iter()
            .all(|f| f.values[0] == prefix | i64::from(f.doc)));
        let ul = opensearch_key(SORT_UNSIGNED_LONG, 0, "m", -1);
        let out = search_sorted_blobs(h, &q, &ul, 5, i64::MAX).unwrap();
        assert!(out
            .hits
            .windows(2)
            .all(|w| (w[0].values[0] as u64) <= (w[1].values[0] as u64)));
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn rescoring_reorders_the_window() {
        let h = open();
        let (hits, _) = run(h, &term_blob("body", "fox"), 10, true).unwrap();
        assert!(hits.len() > 3);
        let rescore = term_blob("body", "dog");
        for mode in 0..5 {
            let mut docs: Vec<i32> = hits.iter().map(|h| h.0).collect();
            let mut scores: Vec<f32> = hits.iter().map(|h| h.1).collect();
            let rc = unsafe {
                ffi_jvm_reader_rescore(
                    h,
                    rescore.as_ptr(),
                    rescore.len(),
                    3,
                    0.5,
                    2.0,
                    mode,
                    docs.as_mut_ptr(),
                    scores.as_mut_ptr(),
                    docs.len(),
                )
            };
            assert_eq!(rc, 0, "{}", crate::error::last_error());
            let mut sorted = docs.clone();
            sorted.sort_unstable();
            let mut want: Vec<i32> = hits.iter().map(|h| h.0).collect();
            want.sort_unstable();
            assert_eq!(sorted, want, "the same documents");
            assert!(scores.windows(2).all(|w| w[0] >= w[1]), "best first");
        }
        let mut docs = vec![hits[0].0];
        let mut scores = vec![1.0f32, 2.0];
        let invalid = Err(FfiStatus::InvalidArgument);
        assert_eq!(
            rescore_hits(h, &rescore, 3, 1.0, 1.0, 9, &mut docs, &mut scores[..1]),
            invalid
        );
        assert_eq!(
            rescore_hits(h, &rescore, 3, 1.0, 1.0, 0, &mut docs, &mut scores),
            invalid
        );
        assert_eq!(
            rescore_hits(
                u64::MAX,
                &rescore,
                3,
                1.0,
                1.0,
                0,
                &mut docs,
                &mut scores[..1]
            ),
            Err(FfiStatus::InvalidHandle)
        );
        let rc = unsafe {
            ffi_jvm_reader_rescore(
                h,
                rescore.as_ptr(),
                rescore.len(),
                -1,
                1.0,
                1.0,
                0,
                docs.as_mut_ptr(),
                scores.as_mut_ptr(),
                1,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        let rc = unsafe {
            ffi_jvm_reader_rescore(
                h,
                rescore.as_ptr(),
                rescore.len(),
                3,
                1.0,
                1.0,
                0,
                std::ptr::null_mut(),
                scores.as_mut_ptr(),
                1,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }
}
