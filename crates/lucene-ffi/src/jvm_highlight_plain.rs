//! OpenSearch's `plain` highlighter for the fetch phase
//! ([`lucene_highlighter::opensearch_plain`]): the plugin hands over a
//! field's request (query, analyzer, options) and one hit's loaded values,
//! and gets back the fragments of each value that scored and, when there
//! are none, where the `no_match_size` summary ends.
//!
//! Request layout (little-endian, lengths `i32`):
//!
//! | field | |
//! |---|---|
//! | `version: u8` | [`REQUEST_VERSION`] |
//! | `field: bytes` | UTF-8 |
//! | `query: bytes` | a query blob ([`lucene_engine::query_blob::decode_query`]) |
//! | `analyzer: bytes` | the field's index analyzer ([`lucene_engine::analysis`]) |
//! | `require_field_match: u8` | `1`: only the field's terms count |
//! | `fragmenter: u8`, `fragment_size: i32` | `0` none (`number_of_fragments: 0`), `1` span, `2` simple |
//! | `number_of_fragments: i32` | |
//! | `pre_tag: bytes`, `post_tag: bytes` | UTF-8 |
//! | `html: u8` | the `html` encoder |
//!
//! Result layout: `status: u8` ([`STATUS_FRAGMENTS`], or [`STATUS_STOCK`]
//! alone), `count: i32` and per
//! fragment `score: f32`, `len: i32` and `len` UTF-16 units, every value's
//! fragments in value order; then `summary_end: i32`, the end of the
//! summary when there is no fragment and a summary was asked for, else -1;
//! then `wait_for_pos: i32`, the span fragmenter's state the call leaves
//! (OpenSearch keeps one fragmenter per field for the whole fetch phase:
//! the plugin hands it to the field's next call).

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_highlighter::opensearch_plain::{
    PlainFragment, PlainFragmenter, PlainHighlighter, PlainOptions,
};

use crate::error::{set_last_error, FfiStatus};
use crate::jvm_highlight::{decode_clause, invalid, put_i32, utf8, Limited};
use lucene_engine::query_blob::Cursor;

/// The request layout's version.
pub(crate) const REQUEST_VERSION: u8 = 1;

/// The fragments follow.
pub(crate) const STATUS_FRAGMENTS: u8 = 0;
/// The hit is one stock's highlighter throws on (a multi-term query expanding
/// past the clause count): nothing follows.
pub(crate) const STATUS_STOCK: u8 = 1;

/// At most this many highlighters are kept per thread.
const CACHED_PER_THREAD: usize = 16;

/// A request and its highlighter.
type Entry = (Box<[u8]>, Arc<PlainHighlighter>);

thread_local! {
    /// The thread's recent requests and their highlighters, the latest last.
    static CACHED: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
}

fn decode(blob: &[u8]) -> Result<PlainHighlighter, FfiStatus> {
    let mut c = Cursor::new(blob);
    let version = c.u8()?;
    if version != REQUEST_VERSION {
        return Err(invalid(format!(
            "plain highlight request: version {version}, expected {REQUEST_VERSION}"
        )));
    }
    let field = utf8(c.bytes()?, "the field")?.to_string();
    let query = decode_clause(c.bytes()?)?;
    let analyzer = lucene_engine::analysis::analyzer(c.bytes()?)?;
    let require_field_match = c.u8()? != 0;
    let kind = c.u8()?;
    let fragment_size = c.i32()?;
    let fragmenter = match kind {
        0 => PlainFragmenter::Null,
        1 => PlainFragmenter::Span(fragment_size),
        2 => PlainFragmenter::Simple(fragment_size),
        k => return Err(invalid(format!("plain highlight request: fragmenter {k}"))),
    };
    let number_of_fragments = c.i32()?;
    let pre_tag = utf8(c.bytes()?, "the pre tag")?.to_string();
    let post_tag = utf8(c.bytes()?, "the post tag")?.to_string();
    let html = c.u8()? != 0;
    if c.remaining() != 0 {
        return Err(invalid(format!(
            "plain highlight request: {} trailing bytes",
            c.remaining()
        )));
    }
    let mut by_field = HashMap::new();
    by_field.insert(field.clone(), analyzer);
    let analyzer = Arc::new(Analyzer::new(Limited {
        by_field,
        missing: Analyzer::keyword(),
        limit: None,
    }));
    let options = PlainOptions {
        field,
        require_field_match,
        fragmenter,
        number_of_fragments,
        pre_tag,
        post_tag,
        html,
    };
    Ok(PlainHighlighter::new(options, query, analyzer))
}

/// The request's highlighter, built on its first use on this thread.
fn prepared(request: &[u8]) -> Result<Arc<PlainHighlighter>, FfiStatus> {
    CACHED.with(|cell| {
        let mut cache = cell.borrow_mut();
        if let Some(i) = cache.iter().position(|(k, _)| &**k == request) {
            let entry = cache.remove(i);
            let h = Arc::clone(&entry.1);
            cache.push(entry);
            return Ok(h);
        }
        let h = Arc::new(decode(request)?);
        if cache.len() >= CACHED_PER_THREAD {
            cache.remove(0);
        }
        cache.push((request.into(), Arc::clone(&h)));
        Ok(h)
    })
}

fn failed(e: &lucene_highlighter::Error) -> FfiStatus {
    set_last_error(format!("{}: {}", e.class(), e.message()));
    FfiStatus::Search
}

/// One hit's fragments: each of `values` highlighted as `request` asks, the
/// span fragmenter waiting for `wait_for_pos` first, and with none, where
/// `summary`'s first `no_match_size` characters end at a token (when
/// `no_match_size` is above 0 and there is a summary).
pub(crate) fn highlight(
    request: &[u8],
    values: &[&[u16]],
    no_match_size: i32,
    summary: Option<&[u16]>,
    mut wait_for_pos: i32,
) -> Result<Vec<u8>, FfiStatus> {
    let h = prepared(request)?;
    let mut fragments: Vec<PlainFragment> = Vec::new();
    for v in values {
        let text = String::from_utf16_lossy(v);
        match h.fragments_carrying(&text, &mut wait_for_pos) {
            Ok(f) => fragments.extend(f),
            // stock throws on the hit: it is stock's to answer
            Err(lucene_highlighter::Error::TooManyClauses(_)) => return Ok(vec![STATUS_STOCK]),
            Err(e) => return Err(failed(&e)),
        }
    }
    let mut summary_end = -1;
    if fragments.is_empty() && no_match_size > 0 {
        if let Some(s) = summary {
            let text = String::from_utf16_lossy(s);
            summary_end = h
                .no_match_end(no_match_size, &text)
                .map_err(|e| failed(&e))?;
        }
    }
    let encoded: Vec<Vec<u16>> = fragments
        .iter()
        .map(|f| f.text.encode_utf16().collect())
        .collect();
    let units: usize = encoded.iter().map(Vec::len).sum();
    let mut out = crate::raw::try_with_capacity::<u8>(13 + fragments.len() * 8 + units * 2)?;
    out.push(STATUS_FRAGMENTS);
    put_i32(&mut out, i32::try_from(fragments.len()).unwrap_or(i32::MAX));
    for (f, text) in fragments.iter().zip(&encoded) {
        out.extend_from_slice(&f.score.to_bits().to_le_bytes());
        let len = i32::try_from(text.len())
            .map_err(|_| invalid("highlight: a fragment longer than 2^31 units"))?;
        put_i32(&mut out, len);
        for u in text {
            out.extend_from_slice(&u.to_le_bytes());
        }
    }
    put_i32(&mut out, summary_end);
    put_i32(&mut out, wait_for_pos);
    Ok(out)
}

/// A UTF-16 buffer from the JVM: `len` units at `ptr` (null when 0).
///
/// # Safety
/// `ptr` must be valid for `len` `u16` reads.
pub(crate) unsafe fn units<'a>(
    ptr: *const u16,
    len: usize,
    what: &str,
) -> Result<&'a [u16], FfiStatus> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(FfiStatus::NullPointer);
    }
    if !ptr.is_aligned() {
        return Err(invalid(format!("plain highlight: {what} not aligned to 2")));
    }
    // SAFETY: caller contract, non-null and aligned checked above.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// [`highlight`]: one hit's fragments as an owned buffer
/// ([`crate::ffm_bridge::ffi_jvm_free_bytes`]). `content` holds the hit's
/// values one after the other (`content_len` UTF-16 units), `lengths` each
/// one's length (`count` of them); `summary` is the value the
/// `no_match_size` summary is cut from (`summary_len` units; null with a
/// negative length when there is none); `wait_for_pos` the span
/// fragmenter's state the field's previous call left, -1 at first.
///
/// # Safety
/// `request` must be valid for `request_len` bytes, `content` for
/// `content_len` `u16` reads and aligned to 2 (null when 0), `lengths` for
/// `count` `i32` reads and aligned to 4 (null when 0), `summary` for
/// `summary_len` `u16` reads and aligned to 2 (null when not above 0),
/// `out_ptr`/`out_len` for one write each.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_jvm_highlight_plain_alloc(
    request: *const u8,
    request_len: usize,
    content: *const u16,
    content_len: usize,
    lengths: *const i32,
    count: usize,
    no_match_size: i32,
    summary: *const u16,
    summary_len: i64,
    wait_for_pos: i32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    crate::error::guard(|| {
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::clear_out(out_ptr, out_len)? };
        // SAFETY: caller contract.
        let request = unsafe { crate::raw::bytes_from_raw(request, request_len)? };
        // SAFETY: caller contract.
        let content = unsafe { units(content, content_len, "content")? };
        let lengths: &[i32] = if count == 0 {
            &[]
        } else if lengths.is_null() {
            return Err(FfiStatus::NullPointer);
        } else if !lengths.is_aligned() {
            return Err(invalid("plain highlight: lengths not aligned to 4"));
        } else {
            // SAFETY: caller contract, non-null and aligned checked above.
            unsafe { std::slice::from_raw_parts(lengths, count) }
        };
        let mut values = crate::raw::try_with_capacity::<&[u16]>(count)?;
        let mut at = 0usize;
        for &len in lengths {
            let len = usize::try_from(len)
                .map_err(|_| invalid(format!("plain highlight: a value of length {len}")))?;
            let end = at
                .checked_add(len)
                .filter(|&e| e <= content.len())
                .ok_or_else(|| invalid("plain highlight: the lengths overrun the content"))?;
            values.push(&content[at..end]);
            at = end;
        }
        if at != content.len() {
            return Err(invalid(
                "plain highlight: the lengths do not cover the content",
            ));
        }
        let summary = match usize::try_from(summary_len) {
            // SAFETY: caller contract.
            Ok(len) => Some(unsafe { units(summary, len, "summary")? }),
            Err(_) => None,
        };
        let encoded = highlight(request, &values, no_match_size, summary, wait_for_pos)?;
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::give(encoded, out_ptr, out_len) };
        Ok(())
    })
}

#[cfg(test)]
mod tests;
