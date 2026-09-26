//! The reader the OpenSearch plugin (`opensearch-plugin/`, M2) searches
//! through: one handle per *Java* `DirectoryReader`, opened from the exact
//! segment list that reader sees and masked with the exact live docs it
//! sees, so that a hit's doc ID means the same document on both sides of the
//! boundary.
//!
//! ## Why not [`crate::directory_reader::ffi_open_directory_reader`]
//!
//! That entry point opens the latest `segments_N` on disk. An OpenSearch
//! searcher is almost never that commit: a refresh opens an NRT reader from
//! the `IndexWriter` (`DirectoryReader.open(writer, applyAllDeletes = true,
//! writeAllDeletes = false)`), which lists segments flushed since the last
//! commit and carries deletions -- hard and soft -- that exist only in the
//! JVM's memory. Searching the last commit instead would miss documents and
//! serve deleted ones, and every doc ID after the first new segment would
//! point at the wrong document.
//!
//! So the JVM hands over what it actually has:
//!
//! - **the segment list**, as the bytes `SegmentInfos.write(IndexOutput)`
//!   produces for the reader's own `SegmentInfos` -- the `segments_N` format
//!   `lucene_index::segment_infos::parse` already reads, so there is no second
//!   encoding of a commit to keep in step with Java's;
//! - **each segment's `maxDoc`**, which [`ffi_open_jvm_reader`] checks against
//!   what it opened. A mismatch is an error, not a best effort: a wrong doc
//!   base silently returns the wrong documents;
//! - **each segment's live docs**, in the same call,
//!   copied from the Java leaf's `getLiveDocs()`. These replace whatever this
//!   port read from `.liv` (or derived from a soft-deletes field) *entirely*,
//!   because the JVM's view is the one OpenSearch answers with.
//!
//! A refresh passes the previous handle as `previous`, and every segment
//! that is unchanged is shared rather than re-read
//! ([`DirectoryReader::reopen_at`]).
//!
//! ## One crossing per query
//!
//! [`ffi_jvm_reader_search`] decodes the query, runs it, counts the total
//! hits when the caller asks, and writes the top hits straight into the
//! caller's buffers -- no results handle to read back and close. The query
//! arrives as one byte blob ([`decode_query`] documents the layout), so a JNI
//! caller passes one `byte[]` rather than seven parallel arrays of strings.
//!
//! ## Scope
//!
//! `TermQuery`, and `BooleanQuery` trees whose clauses are `TermQuery`,
//! `BooleanQuery`, `ConstantScoreQuery` or `BoostQuery` -- the shapes the
//! occur-tagged clause format ([`crate::query::read_boolean_query`])
//! carries. Everything else is the caller's to fall back on.

use std::os::raw::c_char;

use lucene_search::aggs::{MetricSpec, MetricState, ValueKind};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_search::top_field::{FieldDoc, Selector, SortField, SortType};
use lucene_search::weight_count::count_term_query;
use lucene_search::{
    count_boolean_query_segment, search_boolean_query_multi_segment_maxscore_counting,
    search_term_query_multi_segment_counting, ScoreDoc, TotalHits, TotalHitsRelation,
};
use lucene_store::MmapDirectory;
use lucene_util::fixed_bit_set::{bits2words, FixedBitSet};

use crate::error::{guard, set_last_error, FfiStatus};
use crate::query::{check_clause_count, map_search_error, read_boolean_query};
use crate::raw::{bytes_from_raw, str_from_raw, try_with_capacity};
use crate::registry::{jvm_readers, lock_recovering, read_recovering, JvmReaderHandle};
use std::sync::Arc;

/// The version of the contract between this library and the Java classes in
/// `opensearch-plugin/`: the entry points in this module and `jni_bridge.rs`,
/// their argument order, and the query blob layout. The plugin refuses to
/// start against a library reporting any other number -- a jar carrying a
/// stale `.so` must not get as far as reading an index.
///
/// Bump it on any change a Java caller could observe. History: 1, the first
/// plugin; 2, `CONSTANT_SCORE` and `BOOST` clause kinds; 3, `count_limit`;
/// 4, live docs passed to `ffi_open_jvm_reader` (no `set_live_docs`);
/// 5, the engine writer (`engine_writer.rs`); 6, the writer's `max_docs`;
/// 7, the [`QUERY_TREE`] blob (read path R2); 8, its phrase node (R3); 9,
/// its term-set, prefix and wildcard nodes (R3); 10, its points range node;
/// 11, sorted search ([`ffi_jvm_reader_search_sorted`], read path R4); 12,
/// its keyword keys (terms in, terms out); 13, its options byte and max
/// score (`track_scores`); 14, metric aggregations
/// ([`ffi_jvm_reader_aggregate`], read path R5).
pub const JVM_ABI_VERSION: u32 = 14;

/// Blob tag for a single `TermQuery`.
pub const QUERY_TERM: u8 = 0;
/// Blob tag for a `BooleanQuery` in the occur-tagged clause format.
pub const QUERY_BOOLEAN: u8 = 1;
/// Blob tag for a query tree: one recursive node, see [`decode_node`].
pub const QUERY_TREE: u8 = 2;

/// Query-tree node kinds ([`QUERY_TREE`]).
const NODE_TERM: u8 = 0;
const NODE_BOOLEAN: u8 = 1;
const NODE_CONSTANT_SCORE: u8 = 2;
const NODE_BOOST: u8 = 3;
const NODE_DISMAX: u8 = 4;
const NODE_MATCH_ALL: u8 = 5;
const NODE_MATCH_NONE: u8 = 6;
const NODE_PHRASE: u8 = 7;
const NODE_TERM_SET: u8 = 8;
const NODE_PREFIX: u8 = 9;
const NODE_WILDCARD: u8 = 10;
const NODE_POINT_RANGE: u8 = 11;

/// [`JVM_ABI_VERSION`], for the plugin's load-time handshake.
#[no_mangle]
pub extern "C" fn ffi_jvm_abi_version() -> u32 {
    JVM_ABI_VERSION
}

/// One decoded query blob.
#[derive(Debug)]
pub(crate) enum JvmQuery {
    Term(TermQuery),
    Boolean(BooleanQuery),
}

/// A bounds-checked little-endian reader over a caller's blob. Every read
/// that would run past the end is [`FfiStatus::InvalidArgument`] -- the blob
/// is caller-supplied, so a short one is a bad argument, not a panic.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], FfiStatus> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        let Some(end) = end else {
            set_last_error(format!(
                "query blob truncated: need {n} bytes at offset {}, have {}",
                self.pos,
                self.buf.len()
            ));
            return Err(FfiStatus::InvalidArgument);
        };
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, FfiStatus> {
        Ok(self.take(1)?[0])
    }

    fn i32(&mut self) -> Result<i32, FfiStatus> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i64(&mut self) -> Result<i64, FfiStatus> {
        let b = self.take(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn len(&mut self) -> Result<usize, FfiStatus> {
        let v = self.i32()?;
        usize::try_from(v).map_err(|_| {
            set_last_error(format!("query blob: negative length {v}"));
            FfiStatus::InvalidArgument
        })
    }

    fn bytes(&mut self) -> Result<&'a [u8], FfiStatus> {
        let n = self.len()?;
        self.take(n)
    }
}

/// Decodes a query blob. Little-endian throughout; every length is an `i32`
/// and must be non-negative.
///
/// | tag | layout after the tag byte |
/// |---|---|
/// | [`QUERY_TERM`] | `field_len`, field (UTF-8), `term_len`, term bytes |
/// | [`QUERY_BOOLEAN`] | `minimum_should_match`, `clause_count`, then per clause: `occur: u8`, `kind: u8`, `parent: i32`, `param: i32`, `field_len`, field, `term_len`, term |
///
/// A boolean clause's fields mean exactly what the parallel arrays of
/// [`read_boolean_query`] mean -- this decoder builds those arrays and hands
/// them over, so the nesting, depth and clause-count rules (and their error
/// messages) are that function's, not a second copy of them. A `BOOLEAN`
/// clause carries empty field and term bytes.
///
/// Trailing bytes are an error: a blob that decodes with bytes left over was
/// built by a writer that disagrees with this layout.
pub(crate) fn decode_query(blob: &[u8]) -> Result<JvmQuery, FfiStatus> {
    let mut c = Cursor { buf: blob, pos: 0 };
    let query = match c.u8()? {
        QUERY_TERM => {
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| FfiStatus::InvalidUtf8)?;
            let term = c.bytes()?;
            JvmQuery::Term(TermQuery::new(field, term.to_vec()))
        }
        QUERY_BOOLEAN => {
            let msm = c.i32()?;
            let count = c.len()?;
            check_clause_count(count)?;
            let mut occurs = try_with_capacity::<u8>(count)?;
            let mut kinds = try_with_capacity::<u8>(count)?;
            let mut parents = try_with_capacity::<i32>(count)?;
            let mut params = try_with_capacity::<i32>(count)?;
            let mut fields = try_with_capacity::<*const c_char>(count)?;
            let mut field_lens = try_with_capacity::<usize>(count)?;
            let mut terms = try_with_capacity::<*const u8>(count)?;
            let mut term_lens = try_with_capacity::<usize>(count)?;
            for _ in 0..count {
                occurs.push(c.u8()?);
                kinds.push(c.u8()?);
                parents.push(c.i32()?);
                params.push(c.i32()?);
                let field = c.bytes()?;
                fields.push(field.as_ptr().cast::<c_char>());
                field_lens.push(field.len());
                let term = c.bytes()?;
                terms.push(term.as_ptr());
                term_lens.push(term.len());
            }
            // SAFETY: all eight arrays hold exactly `count` elements, and every
            // (pointer, length) pair points into `blob`, which outlives this
            // call.
            let query = unsafe {
                read_boolean_query(
                    occurs.as_ptr(),
                    kinds.as_ptr(),
                    fields.as_ptr(),
                    field_lens.as_ptr(),
                    terms.as_ptr(),
                    term_lens.as_ptr(),
                    parents.as_ptr(),
                    params.as_ptr(),
                    count,
                    msm,
                )?
            };
            JvmQuery::Boolean(query)
        }
        QUERY_TREE => {
            let mut nodes = 0usize;
            match decode_node(&mut c, 0, &mut nodes)? {
                Clause::Boolean(b) => JvmQuery::Boolean(*b),
                other => JvmQuery::Boolean(BooleanQuery {
                    must: vec![other],
                    ..Default::default()
                }),
            }
        }
        other => {
            set_last_error(format!(
                "query blob: unknown query tag {other} (expected 0=TERM, 1=BOOLEAN, 2=TREE)"
            ));
            return Err(FfiStatus::InvalidArgument);
        }
    };
    if c.pos != blob.len() {
        set_last_error(format!(
            "query blob: {} trailing bytes after the query",
            blob.len() - c.pos
        ));
        return Err(FfiStatus::InvalidArgument);
    }
    Ok(query)
}

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
        // handle matters (M2 T2.6).
        let dir = MmapDirectory::open(path);
        let reader = if previous == 0 {
            DirectoryReader::open_at(&dir, segment_infos)
        } else {
            // Cloned out under a short read lock: the reopen reads files and
            // must not hold up every other search and close on the node.
            let prev = lookup(
                previous,
                "ffi_open_jvm_reader: unknown or already-closed previous handle",
            )?;
            prev.reader.reopen_at(&dir, segment_infos)
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
        let handle = lock_recovering(jvm_readers()).insert_checked(Arc::new(JvmReaderHandle {
            reader,
            live_docs,
            deleted,
        }))?;
        // SAFETY: caller contract.
        unsafe { *out_handle = handle };
        Ok(())
    })
}

/// A shared reference to an open handle, taken under a read lock held only
/// for the lookup.
fn lookup(handle: u64, missing: &str) -> Result<Arc<JvmReaderHandle>, FfiStatus> {
    read_recovering(jvm_readers())
        .get(handle)
        .cloned()
        .ok_or_else(|| {
            set_last_error(missing);
            FfiStatus::InvalidHandle
        })
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

/// Runs the query in the `query_len`-byte blob `query` ([`decode_query`])
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
        let query = decode_query(blob)?;
        // No lock is held while searching: a slow query must not block a
        // refresh's open or close, and through it every other search.
        let h = lookup(
            handle,
            "ffi_jvm_reader_search: unknown or already-closed handle",
        )?;
        let (hits, total, lower_bound) = search(&h, &query, top_n, count_limit)?;
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

/// Sort-key types in a sort blob ([`decode_sort`]).
const SORT_SCORE: u8 = 0;
const SORT_DOC: u8 = 1;
const SORT_LONG: u8 = 2;
const SORT_INT: u8 = 3;
const SORT_DOUBLE: u8 = 4;
const SORT_FLOAT: u8 = 5;
const SORT_STRING: u8 = 6;
/// Sort-key flags.
const SORT_REVERSE: u8 = 1;
const SORT_MAX: u8 = 2;
/// Sort-blob options: track the max score over every match.
const SORT_TRACK_MAX_SCORE: u8 = 1;
/// At most this many keys: OpenSearch's sorts are a handful, and each key
/// costs a value per hit on the way back.
const MAX_SORT_KEYS: usize = 16;

/// Decodes a sort blob: `key_count: u8`, then per key `type: u8`,
/// `flags: u8` ([`SORT_REVERSE`], [`SORT_MAX`]) and, for a field's type, the
/// field (`len: i32`, UTF-8) and the missing value: a comparable `i64` for a
/// numeric key (see [`lucene_search::top_field`]), `1` (`STRING_LAST`) or `0`
/// (`STRING_FIRST`) for a keyword one; then `has_after: u8` and, when 1, the
/// search-after document (`doc: i32`) and one value per key, encoded as the
/// search returns them -- an `i64`, or for a keyword key `present: u8` and,
/// when 1, the term (`len: i32`, bytes); and last `options: u8`
/// ([`SORT_TRACK_MAX_SCORE`]). Little-endian, and trailing bytes are an error.
#[allow(clippy::type_complexity)]
pub(crate) fn decode_sort(
    blob: &[u8],
) -> Result<(Vec<SortField>, Option<FieldDoc>, bool), FfiStatus> {
    let mut c = Cursor { buf: blob, pos: 0 };
    let bad = |msg: String| {
        set_last_error(msg);
        FfiStatus::InvalidArgument
    };
    let n = usize::from(c.u8()?);
    if n == 0 || n > MAX_SORT_KEYS {
        return Err(bad(format!(
            "sort blob: {n} keys, want 1..={MAX_SORT_KEYS}"
        )));
    }
    let mut keys = Vec::new();
    for _ in 0..n {
        let ty = c.u8()?;
        let flags = c.u8()?;
        if flags & !(SORT_REVERSE | SORT_MAX) != 0 {
            return Err(bad(format!("sort blob: unknown flags {flags:#x}")));
        }
        let ty = match ty {
            SORT_SCORE => SortType::Score,
            SORT_DOC => SortType::Doc,
            SORT_LONG => SortType::Long,
            SORT_INT => SortType::Int,
            SORT_DOUBLE => SortType::Double,
            SORT_FLOAT => SortType::Float,
            SORT_STRING => SortType::String,
            other => return Err(bad(format!("sort blob: unknown key type {other}"))),
        };
        let (field, missing) = match ty {
            SortType::Score | SortType::Doc => (String::new(), 0),
            _ => {
                let field = std::str::from_utf8(c.bytes()?).map_err(|_| FfiStatus::InvalidUtf8)?;
                let missing = c.i64()?;
                if ty == SortType::String && !(0..=1).contains(&missing) {
                    return Err(bad(format!(
                        "sort blob: keyword missing value {missing}, want 0 or 1"
                    )));
                }
                (field.to_string(), missing)
            }
        };
        keys.push(SortField {
            field,
            ty,
            reverse: flags & SORT_REVERSE != 0,
            selector: if flags & SORT_MAX != 0 {
                Selector::Max
            } else {
                Selector::Min
            },
            missing,
        });
    }
    let after = match c.u8()? {
        0 => None,
        1 => {
            let doc = c.i32()?;
            let mut values = Vec::new();
            let mut terms = Vec::new();
            for k in &keys {
                if k.ty != SortType::String {
                    values.push(c.i64()?);
                    terms.push(None);
                    continue;
                }
                values.push(0);
                terms.push(match c.u8()? {
                    0 => None,
                    1 => Some(c.bytes()?.to_vec()),
                    other => return Err(bad(format!("sort blob: term present is {other}"))),
                });
            }
            if keys.iter().all(|k| k.ty != SortType::String) {
                terms.clear();
            }
            Some(FieldDoc { doc, values, terms })
        }
        other => return Err(bad(format!("sort blob: has_after is {other}"))),
    };
    let options = c.u8()?;
    if options & !SORT_TRACK_MAX_SCORE != 0 {
        return Err(bad(format!("sort blob: unknown options {options:#x}")));
    }
    if c.pos != blob.len() {
        return Err(bad(format!(
            "sort blob: {} trailing bytes",
            blob.len() - c.pos
        )));
    }
    Ok((keys, after, options & SORT_TRACK_MAX_SCORE != 0))
}

/// Runs the query blob `query` sorted by the sort blob `sort`
/// ([`decode_sort`]) -- Lucene's `TopFieldCollectorManager(sort, top_n,
/// after, count_limit)` -- writing up to `top_n` hits' global doc ids into
/// `out_docs` and their sort values, `keys` per hit in key order, into
/// `out_values` (a keyword key's value there is 0), and the keyword keys'
/// terms into `out_terms`: per hit, per keyword key in key order, a
/// little-endian `i32` length (`-1` for a missing value) and the bytes.
/// `out_terms_len` receives their length; when that exceeds `terms_cap`
/// nothing is written but it, and the call returns
/// [`FfiStatus::BufferTooSmall`] for the caller to retry with room.
///
/// Total hits as [`ffi_jvm_reader_search`] reports them; `top_n` must be at
/// least 1 (a `size: 0` search has no order to keep).
///
/// # Safety
/// `query`/`sort` must be valid for `query_len`/`sort_len` bytes;
/// `out_docs` for `buf_len` elements and `out_values` for `buf_len * keys`
/// (`keys` being the sort's key count), with `buf_len >= top_n`; `out_terms`
/// for `terms_cap` bytes (it may be null when `terms_cap` is 0);
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
) -> i32 {
    guard(|| {
        if out_hit_count.is_null()
            || out_total.is_null()
            || out_total_is_lower_bound.is_null()
            || out_terms_len.is_null()
            || out_max_score.is_null()
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
        }
        Ok(())
    })
}

/// A sorted search's answer, ready to hand back.
pub(crate) struct SortedOut {
    /// The sort's key count.
    pub(crate) keys: usize,
    pub(crate) hits: Vec<FieldDoc>,
    pub(crate) total: i64,
    pub(crate) lower_bound: bool,
    /// [`encode_terms`] of `hits`.
    pub(crate) terms: Vec<u8>,
    /// Whether any key is a keyword key (the terms are then an answer,
    /// empty or not).
    pub(crate) has_terms: bool,
    /// The max score over every match when tracked, else `NaN`.
    pub(crate) max_score: f32,
}

/// [`ffi_jvm_reader_search_sorted`] up to its output buffers: the blobs
/// decoded, the search run, the terms encoded (the JNI bridge sizes its
/// Java array from them).
pub(crate) fn search_sorted_blobs(
    handle: u64,
    query_blob: &[u8],
    sort_blob: &[u8],
    top_n: usize,
    count_limit: i64,
) -> Result<SortedOut, FfiStatus> {
    let query = decode_query(query_blob)?;
    let (keys, after, track) = decode_sort(sort_blob)?;
    let h = lookup(
        handle,
        "ffi_jvm_reader_search_sorted: unknown or already-closed handle",
    )?;
    let (hits, total, lower_bound, max_score) =
        search_sorted(&h, &query, &keys, after.as_ref(), top_n, count_limit, track)?;
    let terms = encode_terms(&keys, &hits)?;
    Ok(SortedOut {
        max_score,
        has_terms: keys.iter().any(|k| k.ty == SortType::String),
        keys: keys.len(),
        hits,
        total,
        lower_bound,
        terms,
    })
}

/// Value kinds in a metrics blob ([`decode_metrics`]).
const METRIC_LONG: u8 = 0;
const METRIC_DOUBLE: u8 = 1;
const METRIC_FLOAT: u8 = 2;
/// At most this many fields in one metrics blob.
const MAX_METRICS: usize = 64;
/// Doubles per field in [`ffi_jvm_reader_aggregate`]'s output.
pub const METRIC_VALUES: usize = 6;

/// Decodes a metrics blob: `count: u8`, then per field `kind: u8`
/// ([`METRIC_LONG`], [`METRIC_DOUBLE`], [`METRIC_FLOAT`]) and the field
/// (`len: i32`, UTF-8). Trailing bytes are an error.
pub(crate) fn decode_metrics(blob: &[u8]) -> Result<Vec<MetricSpec>, FfiStatus> {
    let mut c = Cursor { buf: blob, pos: 0 };
    let bad = |msg: String| {
        set_last_error(msg);
        FfiStatus::InvalidArgument
    };
    let n = usize::from(c.u8()?);
    if n == 0 || n > MAX_METRICS {
        return Err(bad(format!(
            "metrics blob: {n} fields, want 1..={MAX_METRICS}"
        )));
    }
    let mut specs = Vec::with_capacity(n);
    for _ in 0..n {
        let kind = match c.u8()? {
            METRIC_LONG => ValueKind::Long,
            METRIC_DOUBLE => ValueKind::Double,
            METRIC_FLOAT => ValueKind::Float,
            other => return Err(bad(format!("metrics blob: unknown kind {other}"))),
        };
        let field = std::str::from_utf8(c.bytes()?).map_err(|_| FfiStatus::InvalidUtf8)?;
        specs.push(MetricSpec {
            field: field.to_string(),
            kind,
        });
    }
    if c.pos != blob.len() {
        return Err(bad(format!(
            "metrics blob: {} trailing bytes",
            blob.len() - c.pos
        )));
    }
    Ok(specs)
}

/// The numeric metric aggregations of a metrics blob ([`decode_metrics`])
/// over the live matches of the query blob `query`: per field, its value
/// count into `out_counts` and [`METRIC_VALUES`] doubles into `out_values`
/// -- the compensated sum and its delta, the minimum and maximum over every
/// value, and the minimum of each document's first value and the maximum of
/// each document's last (see [`lucene_search::aggs`]).
///
/// # Safety
/// `query`/`aggs` must be valid for `query_len`/`aggs_len` bytes;
/// `out_counts` for `n` elements and `out_values` for `n *`
/// [`METRIC_VALUES`], `n` being at least the blob's field count.
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
) -> i32 {
    guard(|| {
        if out_counts.is_null() || out_values.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(query, query_len)? };
        // SAFETY: caller contract.
        let aggs_blob = unsafe { bytes_from_raw(aggs, aggs_len)? };
        let states = aggregate_blobs(handle, blob, aggs_blob)?;
        if n < states.len() {
            return Err(FfiStatus::BufferTooSmall);
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

/// A field's doubles, in [`ffi_jvm_reader_aggregate`]'s order.
pub(crate) fn metric_values(s: &MetricState) -> [f64; METRIC_VALUES] {
    [s.sum, s.delta, s.min, s.max, s.min_of_mins, s.max_of_maxes]
}

/// [`ffi_jvm_reader_aggregate`] up to its output buffers.
pub(crate) fn aggregate_blobs(
    handle: u64,
    query_blob: &[u8],
    aggs_blob: &[u8],
) -> Result<Vec<MetricState>, FfiStatus> {
    let query = decode_query(query_blob)?;
    let specs = decode_metrics(aggs_blob)?;
    let h = lookup(
        handle,
        "ffi_jvm_reader_aggregate: unknown or already-closed handle",
    )?;
    let mut opened = h.reader.open_segments().map_err(|e| {
        set_last_error(format!("opening segment postings: {e}"));
        FfiStatus::Decode
    })?;
    if query_uses_points(&query) {
        opened.open_points().map_err(|e| {
            set_last_error(format!("opening segment points: {e}"));
            FfiStatus::Decode
        })?;
    }
    let segments: Vec<OpenSegment<'_>> = opened
        .as_open_segments()
        .into_iter()
        .zip(&h.live_docs)
        .map(|(mut s, live)| {
            s.live_docs = live.as_ref();
            s
        })
        .collect();
    let q = match &query {
        JvmQuery::Term(t) => BooleanQuery {
            must: vec![Clause::Term(t.clone())],
            ..Default::default()
        },
        JvmQuery::Boolean(b) => b.clone(),
    };
    lucene_search::aggs::metric_states(&segments, h.reader.segment_readers(), &q, &specs)
        .map_err(map_search_error)
}

/// The keyword keys' terms of `hits`, as [`ffi_jvm_reader_search_sorted`]
/// hands them back.
pub(crate) fn encode_terms(keys: &[SortField], hits: &[FieldDoc]) -> Result<Vec<u8>, FfiStatus> {
    let mut out = Vec::new();
    if keys.iter().all(|k| k.ty != SortType::String) {
        return Ok(out);
    }
    for hit in hits {
        for (k, key) in keys.iter().enumerate() {
            if key.ty != SortType::String {
                continue;
            }
            match hit.terms.get(k).and_then(Option::as_ref) {
                None => out.extend_from_slice(&(-1i32).to_le_bytes()),
                Some(t) => {
                    let len = i32::try_from(t.len()).map_err(|_| {
                        set_last_error(format!("a {}-byte sort term", t.len()));
                        FfiStatus::InvalidArgument
                    })?;
                    out.extend_from_slice(&len.to_le_bytes());
                    out.extend_from_slice(t);
                }
            }
        }
    }
    Ok(out)
}

/// The search behind [`ffi_jvm_reader_search_sorted`].
#[allow(clippy::type_complexity)]
pub(crate) fn search_sorted(
    h: &JvmReaderHandle,
    query: &JvmQuery,
    keys: &[SortField],
    after: Option<&FieldDoc>,
    top_n: usize,
    count_limit: i64,
    track_max_score: bool,
) -> Result<(Vec<FieldDoc>, i64, bool, f32), FfiStatus> {
    let mut opened = h.reader.open_segments().map_err(|e| {
        set_last_error(format!("opening segment postings: {e}"));
        FfiStatus::Decode
    })?;
    // A numeric key skips with its field's points, as `NumericComparator` does.
    let numeric_key = keys
        .iter()
        .any(|k| !matches!(k.ty, SortType::Score | SortType::Doc | SortType::String));
    if numeric_key || query_uses_points(query) {
        opened.open_points().map_err(|e| {
            set_last_error(format!("opening segment points: {e}"));
            FfiStatus::Decode
        })?;
    }
    let segments: Vec<OpenSegment<'_>> = opened
        .as_open_segments()
        .into_iter()
        .zip(&h.live_docs)
        .map(|(mut s, live)| {
            s.live_docs = live.as_ref();
            s
        })
        .collect();
    let q = match query {
        JvmQuery::Term(t) => BooleanQuery {
            must: vec![Clause::Term(t.clone())],
            ..Default::default()
        },
        JvmQuery::Boolean(b) => b.clone(),
    };
    let needs_scores = track_max_score || keys.iter().any(|k| k.ty == SortType::Score);
    let fields: Vec<String> = if needs_scores {
        crate::query::clause_field_names(&q)
            .into_iter()
            .map(str::to_string)
            .collect()
    } else {
        Vec::new()
    };
    let owned = h.reader.field_norms_by_field(&fields);
    let norms: Vec<Option<&std::collections::HashMap<String, FieldNorms<'_>>>> =
        owned.iter().map(|m| (!m.is_empty()).then_some(m)).collect();
    // `i64::MAX` is Java's `Integer.MAX_VALUE` threshold: count exactly, and
    // so never prune (`TopFieldCollector`'s exhaustive score modes).
    let threshold = match count_limit {
        i64::MAX => u64::MAX,
        n => u64::try_from(n).unwrap_or(0),
    };
    let top = lucene_search::top_field::search_sorted_tracking(
        &segments,
        h.reader.segment_readers(),
        &q,
        &norms,
        keys,
        top_n,
        threshold,
        after,
        track_max_score,
    )
    .map_err(map_search_error)?;
    if count_limit <= 0 {
        return Ok((top.hits, -1, false, top.max_score));
    }
    Ok((
        top.hits,
        i64::try_from(top.total.value).unwrap_or(i64::MAX),
        top.total.relation == TotalHitsRelation::GreaterThanOrEqualTo,
        top.max_score,
    ))
}

/// Whether `query` has a points clause anywhere, so the search opens the
/// segments' points (a cost a query without one should not pay).
fn query_uses_points(query: &JvmQuery) -> bool {
    fn clause(c: &Clause) -> bool {
        match c {
            Clause::PointsRange(_) => true,
            Clause::Boolean(b) => boolean(b),
            Clause::ConstantScore(c) => clause(&c.inner),
            Clause::Boost(b) => clause(&b.inner),
            Clause::DisjunctionMax(d) => d.disjuncts.iter().any(clause),
            _ => false,
        }
    }
    fn boolean(b: &BooleanQuery) -> bool {
        b.must
            .iter()
            .chain(&b.filter)
            .chain(&b.should)
            .chain(&b.must_not)
            .any(clause)
    }
    match query {
        JvmQuery::Term(_) => false,
        JvmQuery::Boolean(b) => boolean(b),
    }
}

/// The search behind [`ffi_jvm_reader_search`], on an already-validated
/// handle: the reader's segments with the JVM's live docs swapped in.
pub(crate) fn search(
    h: &JvmReaderHandle,
    query: &JvmQuery,
    top_n: usize,
    count_limit: i64,
) -> Result<(Vec<ScoreDoc>, i64, bool), FfiStatus> {
    let mut opened = h.reader.open_segments().map_err(|e| {
        set_last_error(format!("opening segment postings: {e}"));
        FfiStatus::Decode
    })?;
    if query_uses_points(query) {
        opened.open_points().map_err(|e| {
            set_last_error(format!("opening segment points: {e}"));
            FfiStatus::Decode
        })?;
    }
    let segments: Vec<OpenSegment<'_>> = opened
        .as_open_segments()
        .into_iter()
        .zip(&h.live_docs)
        .map(|(mut s, live)| {
            s.live_docs = live.as_ref();
            s
        })
        .collect();

    // Lucene's `TopScoreDocCollectorManager(n, totalHitsThreshold)`: the
    // search itself counts, exactly up to the threshold, and may prune past
    // it. `u64::MAX` (an exact count) disables pruning, as Java's
    // `Integer.MAX_VALUE` does; no count at all lets it prune at once.
    let threshold = u64::try_from(count_limit).unwrap_or(0);
    let searched: Option<(Vec<ScoreDoc>, TotalHits)> = if top_n == 0 {
        None
    } else {
        Some(match query {
            JvmQuery::Term(q) => {
                let owned = h.reader.field_norms(&q.field);
                let norms: Vec<Option<&FieldNorms<'_>>> =
                    owned.iter().map(Option::as_ref).collect();
                search_term_query_multi_segment_counting(&segments, q, &norms, top_n, threshold)
                    .map_err(map_search_error)?
            }
            JvmQuery::Boolean(q) => {
                let fields: Vec<String> = crate::query::clause_field_names(q)
                    .into_iter()
                    .map(str::to_string)
                    .collect();
                let owned = h.reader.field_norms_by_field(&fields);
                let norms: Vec<Option<&std::collections::HashMap<String, FieldNorms<'_>>>> =
                    owned.iter().map(|m| (!m.is_empty()).then_some(m)).collect();
                search_boolean_query_multi_segment_maxscore_counting(
                    &segments, q, &norms, top_n, threshold,
                )
                .map_err(map_search_error)?
            }
        })
    };

    Ok(match searched {
        _ if count_limit <= 0 => (searched.map(|(h, _)| h).unwrap_or_default(), -1, false),
        Some((hits, total)) => {
            let value = i64::try_from(total.value).unwrap_or(i64::MAX);
            (
                hits,
                value,
                total.relation == TotalHitsRelation::GreaterThanOrEqualTo,
            )
        }
        // `size: 0`: no top hits to collect, so the count is its own pass.
        None => {
            let (total, lower_bound) = total_hits(h, &segments, query, count_limit)?;
            (Vec::new(), total, lower_bound)
        }
    })
}

/// The total-hits count behind a `size: 0` [`ffi_jvm_reader_search`] -- one
/// with no top hits, and so no collector to count them -- as `(count,
/// is_lower_bound)`.
///
/// Lucene stops counting at `totalHitsThreshold` and reports "at least"; so
/// does this, in two steps, cheapest first:
///
/// 1. **A lower bound from the term dictionary alone** ([`lower_bound`]): a
///    term matches at least `docFreq - deletedDocs` live documents of a
///    segment, and a pure disjunction at least as many as its best clause.
///    When that already exceeds `count_limit`, no postings are read at all.
/// 2. **Counting segment by segment**, stopping at the first segment
///    boundary where the running total exceeds `count_limit`.
///
/// Either way the answer is exact whenever it is at most `count_limit`,
/// which is all Lucene promises.
fn total_hits(
    h: &JvmReaderHandle,
    segments: &[OpenSegment<'_>],
    query: &JvmQuery,
    count_limit: i64,
) -> Result<(i64, bool), FfiStatus> {
    let mut bound = 0i64;
    for (i, seg) in segments.iter().enumerate() {
        let deleted = h.deleted.get(i).copied().unwrap_or(0);
        let lb = match query {
            JvmQuery::Term(q) => term_lower_bound(seg, deleted, q)?,
            JvmQuery::Boolean(q) => lower_bound_boolean(seg, deleted, q)?,
        };
        bound = bound.saturating_add(lb);
    }
    if bound > count_limit {
        return Ok((bound, true));
    }
    let mut total = 0i64;
    for seg in segments {
        total = total.saturating_add(count_segment(seg, query)?);
        if total > count_limit {
            return Ok((total, true));
        }
    }
    Ok((total, false))
}

/// Live documents of `seg` certainly matching `q`: its `docFreq` less every
/// deleted document of the segment (each could have been one of them).
fn term_lower_bound(seg: &OpenSegment<'_>, deleted: i64, q: &TermQuery) -> Result<i64, FfiStatus> {
    let Some(field) = seg.fields.field(&q.field) else {
        return Ok(0);
    };
    let df = field
        .try_seek_exact(&q.term)
        .map_err(|e| map_search_error(e.into()))?
        .map_or(0, |s| i64::from(s.doc_freq));
    Ok(df.saturating_sub(deleted).max(0))
}

/// [`lower_bound`] for a boolean: a pure disjunction (only `SHOULD` clauses,
/// `minimum_should_match <= 1`) as its best clause; exactly one
/// `MUST`/`FILTER` clause and nothing else as that clause; anything else --
/// a conjunction, anything with a `MUST_NOT` -- `0`.
fn lower_bound_boolean(
    seg: &OpenSegment<'_>,
    deleted: i64,
    b: &BooleanQuery,
) -> Result<i64, FfiStatus> {
    if !b.must_not.is_empty() {
        return Ok(0);
    }
    if b.minimum_should_match > b.should.len() {
        // Lucene matches nothing when fewer SHOULD clauses exist than must match.
        return Ok(0);
    }
    let required = b.must.len() + b.filter.len();
    if required == 0 && b.minimum_should_match <= 1 {
        let mut best = 0;
        for c in &b.should {
            best = best.max(lower_bound(seg, deleted, c)?);
        }
        Ok(best)
    } else if required == 1 && b.should.is_empty() {
        let only = b
            .must
            .first()
            .or(b.filter.first())
            .expect("one required clause");
        lower_bound(seg, deleted, only)
    } else {
        Ok(0)
    }
}

/// A cheap lower bound on the live documents of `seg` matching `clause`, from
/// the term dictionary only. Deliberately narrow: a term (see
/// [`term_lower_bound`]); a wrapper, as its inner clause; a boolean, see
/// [`lower_bound_boolean`]. Anything else is `0`, which is always a valid
/// lower bound. Recursion depth is the clause tree's, which the decoder caps
/// at `MAX_CLAUSE_DEPTH`.
fn lower_bound(seg: &OpenSegment<'_>, deleted: i64, clause: &Clause) -> Result<i64, FfiStatus> {
    match clause {
        Clause::Term(t) => term_lower_bound(seg, deleted, t),
        Clause::ConstantScore(c) => lower_bound(seg, deleted, &c.inner),
        Clause::Boost(b) => lower_bound(seg, deleted, &b.inner),
        Clause::Boolean(b) => lower_bound_boolean(seg, deleted, b),
        _ => Ok(0),
    }
}

/// `IndexSearcher.count` for one segment.
fn count_segment(seg: &OpenSegment<'_>, query: &JvmQuery) -> Result<i64, FfiStatus> {
    Ok({
        match query {
            JvmQuery::Term(q) => count_term_query(seg.fields, seg.doc_in, seg.live_docs, q),
            // The scored search's scorer tree, in `COMPLETE_NO_SCORES`.
            JvmQuery::Boolean(q) => {
                count_boolean_query_segment(seg, q).map(|n| i64::try_from(n).unwrap_or(i64::MAX))
            }
        }
        .map_err(map_search_error)?
    })
}

/// One node of a [`QUERY_TREE`] blob, little-endian like the rest:
///
/// | kind | layout after the kind byte | Lucene query |
/// |---|---|---|
/// | `0` term | `field_len`, field (UTF-8), `term_len`, term | `TermQuery` |
/// | `1` boolean | `minimum_should_match: i32`, `count: i32`, then per clause `occur: u8` (0 `MUST`, 1 `FILTER`, 2 `SHOULD`, 3 `MUST_NOT`) and a node | `BooleanQuery` |
/// | `2` constant score | `score: f32` bits, node | `ConstantScoreQuery` (its score, normally 1) |
/// | `3` boost | `boost: f32` bits, node | `BoostQuery` |
/// | `4` dismax | `tie_breaker: f32` bits, `count: i32`, nodes | `DisjunctionMaxQuery` |
/// | `5` match all | nothing | `MatchAllDocsQuery` |
/// | `6` match none | nothing | `MatchNoDocsQuery` |
///
/// Depth is capped at `MAX_CLAUSE_DEPTH` and the whole tree at the clause
/// count limit, so the recursion is bounded by the blob, not trusted to it.
/// A score, boost or tie-breaker must be finite and non-negative (and a tie
/// breaker at most 1), as Lucene's constructors require.
fn decode_node(c: &mut Cursor<'_>, depth: usize, nodes: &mut usize) -> Result<Clause, FfiStatus> {
    use crate::query::MAX_CLAUSE_DEPTH;
    use lucene_search::query::{
        BoostQuery, ConstantScoreQuery, DisjunctionMaxQuery, MatchAllDocsQuery, MatchNoDocsQuery,
        PhraseQuery, PointsRangeQuery, PrefixQuery, TermInSetQuery, WildcardQuery,
    };
    if depth >= MAX_CLAUSE_DEPTH {
        set_last_error(format!(
            "query tree: nesting depth exceeds the maximum of {MAX_CLAUSE_DEPTH}"
        ));
        return Err(FfiStatus::InvalidArgument);
    }
    *nodes = nodes.saturating_add(1);
    check_clause_count(*nodes)?;
    let float = |c: &mut Cursor<'_>, what: &str, max: f32| -> Result<f32, FfiStatus> {
        let v = f32::from_bits(c.i32()? as u32);
        if !v.is_finite() || !(0.0..=max).contains(&v) {
            set_last_error(format!("query tree: {what} {v} is out of range"));
            return Err(FfiStatus::InvalidArgument);
        }
        Ok(v)
    };
    Ok(match c.u8()? {
        NODE_TERM => {
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| FfiStatus::InvalidUtf8)?;
            let term = c.bytes()?;
            Clause::Term(TermQuery::new(field, term.to_vec()))
        }
        NODE_BOOLEAN => {
            let msm = c.len()?;
            let count = c.len()?;
            check_clause_count(count)?;
            let mut b = BooleanQuery::new().with_minimum_should_match(msm);
            for _ in 0..count {
                let occur = c.u8()?;
                let child = decode_node(c, depth + 1, nodes)?;
                match occur {
                    0 => b.must.push(child),
                    1 => b.filter.push(child),
                    2 => b.should.push(child),
                    3 => b.must_not.push(child),
                    other => {
                        set_last_error(format!(
                            "query tree: unknown Occur tag {other} (expected 0..=3)"
                        ));
                        return Err(FfiStatus::InvalidArgument);
                    }
                }
            }
            Clause::Boolean(Box::new(b))
        }
        NODE_CONSTANT_SCORE => {
            let score = float(c, "constant score", f32::MAX)?;
            ConstantScoreQuery::new(decode_node(c, depth + 1, nodes)?, score).into()
        }
        NODE_BOOST => {
            let boost = float(c, "boost", f32::MAX)?;
            BoostQuery::new(decode_node(c, depth + 1, nodes)?, boost).into()
        }
        NODE_DISMAX => {
            let tie = float(c, "tie breaker", 1.0)?;
            let count = c.len()?;
            check_clause_count(count)?;
            let mut disjuncts = Vec::new();
            for _ in 0..count {
                disjuncts.push(decode_node(c, depth + 1, nodes)?);
            }
            DisjunctionMaxQuery::new(disjuncts, tie).into()
        }
        // The segment's own `maxDoc` is supplied per segment at search time
        // (`OpenSegment::max_doc`); the query carries none.
        NODE_MATCH_ALL => Clause::MatchAllDocs(MatchAllDocsQuery::new(i32::MAX)),
        NODE_MATCH_NONE => Clause::MatchNoDocs(MatchNoDocsQuery::new()),
        NODE_PHRASE => {
            // `field`, `slop`, then each term with its position. Only
            // consecutive positions from 0: the encoder falls a phrase with
            // gaps (a removed stopword) back to Lucene.
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| FfiStatus::InvalidUtf8)?;
            let slop = c.len()?;
            let count = c.len()?;
            *nodes = nodes.saturating_add(count);
            check_clause_count(*nodes)?;
            if count == 0 {
                set_last_error("query tree: a phrase with no terms".to_string());
                return Err(FfiStatus::InvalidArgument);
            }
            // Grown as read, not sized from `count`: the blob bounds it.
            let mut terms = Vec::new();
            for i in 0..count {
                let position = c.i32()?;
                if usize::try_from(position).ok() != Some(i) {
                    set_last_error(format!(
                        "query tree: phrase term {i} is at position {position}, not {i}"
                    ));
                    return Err(FfiStatus::InvalidArgument);
                }
                terms.push(c.bytes()?.to_vec());
            }
            let slop = u32::try_from(slop).map_err(|_| FfiStatus::InvalidArgument)?;
            Clause::Phrase(PhraseQuery::new(field, terms).with_slop(slop))
        }
        NODE_TERM_SET => {
            // `TermInSetQuery`: `field`, then the terms, each counted against
            // the node limit as `IndexSearcher`'s clause visitor counts them.
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| FfiStatus::InvalidUtf8)?;
            let count = c.len()?;
            *nodes = nodes.saturating_add(count);
            check_clause_count(*nodes)?;
            let mut terms = Vec::new();
            for _ in 0..count {
                terms.push(c.bytes()?.to_vec());
            }
            Clause::TermInSet(TermInSetQuery::new(field, terms))
        }
        NODE_PREFIX => {
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| FfiStatus::InvalidUtf8)?;
            Clause::Prefix(PrefixQuery::new(field, c.bytes()?.to_vec()))
        }
        NODE_WILDCARD => {
            // Lucene's syntax without its `\` escape, which the encoder never
            // sends (such a pattern falls back).
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| FfiStatus::InvalidUtf8)?;
            Clause::Wildcard(WildcardQuery::new(field, c.bytes()?.to_vec()))
        }
        NODE_POINT_RANGE => {
            // A one-dimensional 8-byte `PointRangeQuery` (`long`, `date`,
            // `double`): its inclusive bounds as the sortable longs
            // `NumericUtils.sortableBytesToLong` reads from the packed bytes.
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| FfiStatus::InvalidUtf8)?;
            let (min, max) = (c.i64()?, c.i64()?);
            Clause::PointsRange(PointsRangeQuery::new(field, min, max))
        }
        other => {
            set_last_error(format!(
                "query tree: unknown node kind {other} (expected 0..=11)"
            ));
            return Err(FfiStatus::InvalidArgument);
        }
    })
}

/// Closes a JVM reader handle. Segments a later handle reused stay open
/// until that handle closes too.
#[no_mangle]
pub extern "C" fn ffi_close_jvm_reader(handle: u64) -> i32 {
    guard(|| {
        lock_recovering(jvm_readers())
            .remove(handle)
            .map(|_| ())
            .ok_or_else(|| {
                set_last_error("ffi_close_jvm_reader: unknown or already-closed handle");
                FfiStatus::InvalidHandle
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real two-segment Java-written index whose `manifest.properties`
    /// records Lucene's own top hits and scores (`GenMultiSegmentScoring`).
    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/multi_segment_scoring_index"
    );

    fn infos() -> Vec<u8> {
        std::fs::read(format!("{FIXTURE}/segments_2")).expect("segments_2")
    }

    fn term_blob(field: &str, term: &str) -> Vec<u8> {
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
    fn open_live(max_docs: &[i32], previous: u64, live: &[&[u64]]) -> (i32, u64) {
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

    fn open() -> u64 {
        let (rc, handle) = open_with(&[4, 4], 0);
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        handle
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
    type Searched = (Vec<(i32, f32)>, i64, bool);

    fn run_limit(handle: u64, blob: &[u8], top_n: usize, limit: i64) -> Result<Searched, i32> {
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
    fn sort_blob_terms(
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
        b
    }

    /// `(hits as (doc, values), total, lower_bound)`, or the status.
    type SortedRun = (Vec<(i32, Vec<i64>)>, i64, bool);

    fn run_sorted(
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
        let (keys, after, _) = decode_sort(&blob).unwrap();
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
        let (keys, after, _) = decode_sort(&blob).unwrap();
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
        let at = bad.len() - 2; // the term's present flag, before the options byte
        bad[at] = 2;
        assert_eq!(decode_sort(&bad).unwrap_err(), FfiStatus::InvalidArgument);
        // Options: bit 0 tracks the max score; any other bit is refused.
        let mut tracked = sort_blob(&[(SORT_DOC, 0, "", 0)], None);
        *tracked.last_mut().unwrap() = SORT_TRACK_MAX_SCORE;
        assert!(decode_sort(&tracked).unwrap().2);
        *tracked.last_mut().unwrap() = 2;
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
        *sort.last_mut().unwrap() = SORT_TRACK_MAX_SCORE;
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
        *sort.last_mut().unwrap() = 0;
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

    fn metrics_blob(fields: &[(u8, &str)]) -> Vec<u8> {
        let mut b = vec![fields.len() as u8];
        for &(kind, f) in fields {
            b.push(kind);
            b.extend_from_slice(&(f.len() as i32).to_le_bytes());
            b.extend_from_slice(f.as_bytes());
        }
        b
    }

    #[test]
    fn metrics_blobs_decode_and_aggregate() {
        let specs = decode_metrics(&metrics_blob(&[
            (METRIC_LONG, "a"),
            (METRIC_DOUBLE, "b"),
            (METRIC_FLOAT, "c"),
        ]))
        .unwrap();
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
        let mut utf8 = vec![1, METRIC_LONG];
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
        let call = |handle: u64, counts: *mut i64, values: *mut f64, n: usize| unsafe {
            ffi_jvm_reader_aggregate(
                handle,
                q.as_ptr(),
                q.len(),
                aggs.as_ptr(),
                aggs.len(),
                counts,
                values,
                n,
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

    #[test]
    fn abi_version_is_the_constant() {
        assert_eq!(ffi_jvm_abi_version(), JVM_ABI_VERSION);
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
        assert_eq!(crate::query::clause_field_names(&q), ["f"]);
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

    // ---- the query-tree blob (ABI 7) ---------------------------------------

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

    fn tree(n: N<'_>) -> Vec<u8> {
        let mut b = vec![QUERY_TREE];
        enc(&n, &mut b);
        b
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
}
