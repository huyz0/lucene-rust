//! OpenSearch's `fvh` highlighter for the fetch phase
//! ([`lucene_highlighter::vectorhighlight`]): the plugin hands over the
//! field query OpenSearch built (`CustomFieldQuery`'s phrase tries and term
//! sets), the field's options and one hit's values; the matches come from
//! the hit's term vectors in the reader's segment, and the call gives back
//! the fragments, or the `no_match_size` summary when nothing matched.
//!
//! Request layout (little-endian, lengths `i32`):
//!
//! | field | |
//! |---|---|
//! | `version: u8` | [`REQUEST_VERSION`] |
//! | `field: bytes` | UTF-8, the field the values are of |
//! | `matched: i32`, then that many `bytes` | `matched_fields`, in OpenSearch's order; 0 for none |
//! | `field_match: u8` | the field query keys its maps by field |
//! | `roots: i32`, then per root `key`, `node` | `rootMaps` |
//! | `term_sets: i32`, then per set `key`, `n: i32`, `n` × `bytes` | `termSetMap` |
//! | `frag_list: u8`, `margin: i32` | `0` single, `1` simple (`margin` -1 for the default) |
//! | `phrase_limit: i32`, `frag_char_size: i32`, `max_fragments: i32` | |
//! | `score_order: u8`, `fix_broken_analysis: u8`, `discrete_values: u8` | |
//! | `scanner: u8`, `max_scan: i32`, `chars: i32` × `u16` | `0` chars, `1` sentence, `2` word |
//! | `pre_tags: i32` × `bytes`, `post_tags: i32` × `bytes` | UTF-8 |
//! | `html: u8`, `no_match_size: i32` | |
//!
//! A `key` is `present: u8` then, when present, `bytes`; a `node` is
//! `terminal: u8`, `slop: i32`, `boost: f32` bits, `number: i32`, then
//! `subs: i32` × (`term: bytes`, `node`).
//!
//! Result layout: `status: u8`; with [`STATUS_FRAGMENTS`], `count: i32` and
//! per fragment `len: i32` and `len` UTF-16 units. [`STATUS_STOCK`] means
//! Java's highlighter throws for this hit (or gives text this one cannot
//! give exactly): the plugin runs OpenSearch's for it; the rest of the
//! result says why (UTF-8).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use lucene_codecs::term_vectors::ChunkCursor;
use lucene_highlighter::unified::break_iterators::BreakIterator;
use lucene_highlighter::vectorhighlight::{
    BoundaryScanner, BreakIteratorBoundaryScanner, FieldFragList, FieldPhraseList, FieldQuery,
    FieldTermStack, FragListBuilder, FragListKind, FragmentsBuilder, FragmentsOrder,
    QueryPhraseMap, SimpleBoundaryScanner, SimpleFragListBuilder, SingleFragListBuilder,
};
use lucene_highlighter::{
    BreakKind, DefaultEncoder, Encoder, RuleBreakIterator, SimpleHtmlEncoder,
};

use crate::error::{set_last_error, FfiStatus};
use crate::jvm_highlight::{invalid, put_i32, utf8};
use crate::jvm_reader::{self, Cursor};

/// The request layout's version.
pub(crate) const REQUEST_VERSION: u8 = 1;

/// The fragments follow.
pub(crate) const STATUS_FRAGMENTS: u8 = 0;
/// Run OpenSearch's highlighter for this hit.
pub(crate) const STATUS_STOCK: u8 = 1;

/// At most this many requests are kept per thread.
const CACHED_PER_THREAD: usize = 16;

/// Deeper phrase tries than this are refused.
const MAX_DEPTH: usize = 1024;

/// How the boundary scanner is made for each hit.
#[derive(Debug, Clone)]
enum Scanner {
    Chars(i32, Vec<u16>),
    Break(BreakKind),
}

/// A decoded request.
#[derive(Debug)]
struct Fvh {
    field: String,
    matched: Vec<String>,
    query: FieldQuery,
    /// `None`: `SingleFragListBuilder`; `Some(margin)`: `SimpleFragListBuilder`.
    margin: Option<i32>,
    phrase_limit: usize,
    frag_char_size: i32,
    max_fragments: i32,
    score_order: bool,
    fix_broken_analysis: bool,
    discrete_values: bool,
    scanner: Scanner,
    pre_tags: Vec<String>,
    post_tags: Vec<String>,
    html: bool,
    no_match_size: i32,
}

type Entry = (Box<[u8]>, Arc<Fvh>);

thread_local! {
    /// The thread's recent requests, the latest last.
    static CACHED: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
}

fn string(c: &mut Cursor<'_>, what: &str) -> Result<String, FfiStatus> {
    Ok(utf8(c.bytes()?, what)?.to_string())
}

fn count(c: &mut Cursor<'_>, what: &str) -> Result<usize, FfiStatus> {
    let n = c.i32()?;
    let n = usize::try_from(n).map_err(|_| invalid(format!("fvh request: {n} {what}")))?;
    if n > c.remaining() {
        return Err(invalid(format!(
            "fvh request: {n} {what} overrun the request"
        )));
    }
    Ok(n)
}

fn key(c: &mut Cursor<'_>) -> Result<Option<String>, FfiStatus> {
    Ok(match c.u8()? {
        0 => None,
        _ => Some(string(c, "a field")?),
    })
}

fn node(c: &mut Cursor<'_>, depth: usize) -> Result<QueryPhraseMap, FfiStatus> {
    if depth > MAX_DEPTH {
        return Err(invalid("fvh request: the phrase trie is too deep"));
    }
    let terminal = c.u8()? != 0;
    let slop = c.i32()?;
    let boost = f32::from_bits(c.i32()?.cast_unsigned());
    let number = c.i32()?;
    let n = count(c, "sub terms")?;
    let mut subs = HashMap::with_capacity(n); // alloc-ok: count() bounds it by the request's bytes
    for _ in 0..n {
        let term = string(c, "a term")?;
        let sub = node(c, depth + 1)?;
        subs.insert(term, sub);
    }
    Ok(QueryPhraseMap::from_parts(
        terminal, slop, boost, number, subs,
    ))
}

fn tags(c: &mut Cursor<'_>, what: &str) -> Result<Vec<String>, FfiStatus> {
    let n = count(c, what)?;
    (0..n).map(|_| string(c, what)).collect()
}

fn decode(blob: &[u8]) -> Result<Fvh, FfiStatus> {
    let mut c = Cursor::new(blob);
    let version = c.u8()?;
    if version != REQUEST_VERSION {
        return Err(invalid(format!(
            "fvh highlight request: version {version}, expected {REQUEST_VERSION}"
        )));
    }
    let field = string(&mut c, "the field")?;
    let n = count(&mut c, "matched fields")?;
    let matched = (0..n)
        .map(|_| string(&mut c, "a matched field"))
        .collect::<Result<Vec<_>, _>>()?;
    let field_match = c.u8()? != 0;
    let n = count(&mut c, "root maps")?;
    let mut roots = HashMap::with_capacity(n); // alloc-ok: count() bounds it by the request's bytes
    for _ in 0..n {
        let k = key(&mut c)?;
        roots.insert(k, node(&mut c, 0)?);
    }
    let n = count(&mut c, "term sets")?;
    let mut term_sets = HashMap::with_capacity(n); // alloc-ok: count() bounds it by the request's bytes
    for _ in 0..n {
        let k = key(&mut c)?;
        let terms = count(&mut c, "terms")?;
        let mut set = HashSet::with_capacity(terms); // alloc-ok: count() bounds it by the request's bytes
        for _ in 0..terms {
            set.insert(string(&mut c, "a term")?);
        }
        term_sets.insert(k, set);
    }
    let query = FieldQuery::from_parts(field_match, roots, term_sets);
    let frag_list = c.u8()?;
    let margin = c.i32()?;
    let margin = match frag_list {
        0 => None,
        1 => Some(margin),
        k => return Err(invalid(format!("fvh highlight request: frag list {k}"))),
    };
    let phrase_limit = usize::try_from(c.i32()?).unwrap_or(0);
    let frag_char_size = c.i32()?;
    let max_fragments = c.i32()?;
    let score_order = c.u8()? != 0;
    let fix_broken_analysis = c.u8()? != 0;
    let discrete_values = c.u8()? != 0;
    let kind = c.u8()?;
    let max_scan = c.i32()?;
    let n = count(&mut c, "boundary chars")?;
    let mut chars = Vec::with_capacity(n); // alloc-ok: count() bounds it by the request's bytes
    for _ in 0..n {
        let lo = c.u8()?;
        let hi = c.u8()?;
        chars.push(u16::from_le_bytes([lo, hi]));
    }
    let scanner = match kind {
        0 => Scanner::Chars(max_scan, chars),
        1 => Scanner::Break(BreakKind::Sentence),
        2 => Scanner::Break(BreakKind::Word),
        k => return Err(invalid(format!("fvh highlight request: scanner {k}"))),
    };
    let pre_tags = tags(&mut c, "pre tags")?;
    let post_tags = tags(&mut c, "post tags")?;
    let html = c.u8()? != 0;
    let no_match_size = c.i32()?;
    if c.remaining() != 0 {
        return Err(invalid(format!(
            "fvh highlight request: {} trailing bytes",
            c.remaining()
        )));
    }
    Ok(Fvh {
        field,
        matched,
        query,
        margin,
        phrase_limit,
        frag_char_size,
        max_fragments,
        score_order,
        fix_broken_analysis,
        discrete_values,
        scanner,
        pre_tags,
        post_tags,
        html,
        no_match_size,
    })
}

/// The request, decoded on its first use on this thread.
fn prepared(request: &[u8]) -> Result<Arc<Fvh>, FfiStatus> {
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

impl Fvh {
    fn fragments_builder(&self) -> FragmentsBuilder {
        let scanner: Box<dyn BoundaryScanner> = match &self.scanner {
            Scanner::Chars(max_scan, chars) => {
                Box::new(SimpleBoundaryScanner::with_chars(*max_scan, chars))
            }
            Scanner::Break(kind) => {
                let bi: Box<dyn BreakIterator> =
                    Box::new(RuleBreakIterator::from_utf16(*kind, Vec::new()));
                Box::new(BreakIteratorBoundaryScanner::new(bi))
            }
        };
        let order = if self.score_order {
            FragmentsOrder::ScoreOrder
        } else {
            FragmentsOrder::Simple
        };
        let mut b = FragmentsBuilder::new(order).with_boundary_scanner(scanner);
        b.set_discrete_multi_value_highlighting(self.discrete_values);
        b.set_fix_broken_analysis(self.fix_broken_analysis);
        b.set_strict_utf16(true);
        b
    }
}

/// One hit's fragments, `None` for none (or Java's error, as `Err`).
fn fragments(
    h: &Fvh,
    segment: &lucene_search::directory_reader::SegmentReader,
    doc: i32,
    values: &[Vec<u16>],
) -> Result<Result<Option<Vec<String>>, lucene_highlighter::Error>, FfiStatus> {
    let read = |e: &dyn std::fmt::Display| {
        set_last_error(format!("fvh highlight: reading term vectors: {e}"));
        FfiStatus::Decode
    };
    let tvs = segment.term_vectors_reader().map_err(|e| read(&e))?;
    let mut cursor = ChunkCursor::new();
    let fields: Vec<&str> = if h.matched.is_empty() {
        vec![h.field.as_str()]
    } else {
        h.matched.iter().map(String::as_str).collect()
    };
    let num_docs = f64::from(segment.max_doc);
    let mut lists = Vec::with_capacity(fields.len()); // alloc-ok: the fields already read
    for field in &fields {
        let vector = match (&tvs, segment.field_infos().field_by_name(field)) {
            (Some(tvs), Some(info)) if h.query.term_set(field).is_some() => {
                cursor.field(tvs, doc, info.number).map_err(|e| read(&e))?
            }
            _ => None,
        };
        let mut doc_freq = |term: &[u8]| {
            segment
                .doc_freq(field, term)
                .map(i64::from)
                .map_err(|e| lucene_highlighter::Error::Io(e.to_string()))
        };
        let mut stack = match FieldTermStack::from_term_vector(
            field,
            vector.as_ref(),
            &h.query,
            num_docs,
            &mut doc_freq,
        ) {
            Ok(s) => s,
            Err(e) => return Ok(Err(e)),
        };
        lists.push(FieldPhraseList::new(&mut stack, &h.query, h.phrase_limit));
    }
    let phrases = match lists.as_mut_slice() {
        [only] if h.matched.is_empty() => std::mem::take(only),
        _ => FieldPhraseList::merge(&lists),
    };
    let encoder: Box<dyn Encoder> = if h.html {
        Box::new(SimpleHtmlEncoder)
    } else {
        Box::new(DefaultEncoder)
    };
    let mut builder = h.fragments_builder();
    let mut run = || -> Result<Option<Vec<String>>, lucene_highlighter::Error> {
        let frag_list = match h.margin {
            None => SingleFragListBuilder.create_field_frag_list(&phrases, h.frag_char_size)?,
            Some(-1) => {
                SimpleFragListBuilder::new().create_field_frag_list(&phrases, h.frag_char_size)?
            }
            Some(m) => SimpleFragListBuilder::with_margin(m)?
                .create_field_frag_list(&phrases, h.frag_char_size)?,
        };
        let found = builder.create_fragments_from_values(
            values,
            &frag_list,
            h.max_fragments,
            &h.pre_tags,
            &h.post_tags,
            &*encoder,
        )?;
        if found.as_ref().is_some_and(|f| !f.is_empty()) || h.no_match_size <= 0 {
            return Ok(found);
        }
        let mut summary = FieldFragList::new(FragListKind::Simple);
        summary.add(0, h.no_match_size, &[]);
        builder.create_fragments_from_values(
            values,
            &summary,
            1,
            &h.pre_tags,
            &h.post_tags,
            &*encoder,
        )
    };
    Ok(run())
}

/// One hit's fragments: segment `segment`'s document `doc` (leaf-local) of
/// `handle`'s reader, whose values are `values`, highlighted as `request`
/// asks.
pub(crate) fn highlight(
    handle: u64,
    request: &[u8],
    segment: i32,
    doc: i32,
    values: &[Vec<u16>],
) -> Result<Vec<u8>, FfiStatus> {
    let h = prepared(request)?;
    let reader = jvm_reader::lookup(
        handle,
        "ffi_jvm_highlight_fvh_alloc: unknown or already-closed handle",
    )?;
    let readers = reader.reader.segment_readers();
    let Some(leaf) = usize::try_from(segment).ok().filter(|&s| s < readers.len()) else {
        return Err(invalid(format!(
            "fvh highlight: segment {segment} of {}",
            readers.len()
        )));
    };
    let seg = &readers[leaf];
    if doc < 0 || doc >= seg.max_doc {
        return Err(invalid(format!(
            "fvh highlight: doc {doc} of segment {segment} with maxDoc {}",
            seg.max_doc
        )));
    }
    let found = match fragments(&h, seg, doc, values)? {
        Ok(found) => found.unwrap_or_default(),
        Err(e) => {
            let mut out = vec![STATUS_STOCK];
            out.extend_from_slice(format!("{}: {}", e.class(), e.message()).as_bytes());
            return Ok(out);
        }
    };
    let encoded: Vec<Vec<u16>> = found.iter().map(|f| f.encode_utf16().collect()).collect();
    let units: usize = encoded.iter().map(Vec::len).sum();
    let mut out = crate::raw::try_with_capacity::<u8>(5 + encoded.len() * 4 + units * 2)?;
    out.push(STATUS_FRAGMENTS);
    put_i32(&mut out, i32::try_from(encoded.len()).unwrap_or(i32::MAX));
    for text in &encoded {
        let len = i32::try_from(text.len())
            .map_err(|_| invalid("fvh highlight: a fragment longer than 2^31 units"))?;
        put_i32(&mut out, len);
        for u in text {
            out.extend_from_slice(&u.to_le_bytes());
        }
    }
    Ok(out)
}

/// [`highlight`]: one hit's fragments as an owned buffer
/// ([`crate::ffm_bridge::ffi_jvm_free_bytes`]). `content` holds the hit's
/// values one after the other (`content_len` UTF-16 units), `lengths` each
/// one's length (`count` of them).
///
/// # Safety
/// `request` must be valid for `request_len` bytes, `content` for
/// `content_len` `u16` reads and aligned to 2 (null when 0), `lengths` for
/// `count` `i32` reads and aligned to 4 (null when 0), `out_ptr`/`out_len`
/// for one write each.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_jvm_highlight_fvh_alloc(
    handle: u64,
    request: *const u8,
    request_len: usize,
    segment: i32,
    doc: i32,
    content: *const u16,
    content_len: usize,
    lengths: *const i32,
    count: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    crate::error::guard(|| {
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::clear_out(out_ptr, out_len)? };
        // SAFETY: caller contract.
        let request = unsafe { crate::raw::bytes_from_raw(request, request_len)? };
        // SAFETY: caller contract.
        let content =
            unsafe { crate::jvm_highlight_plain::units(content, content_len, "content")? };
        let lengths: &[i32] = if count == 0 {
            &[]
        } else if lengths.is_null() {
            return Err(FfiStatus::NullPointer);
        } else if !lengths.is_aligned() {
            return Err(invalid("fvh highlight: lengths not aligned to 4"));
        } else {
            // SAFETY: caller contract, non-null and aligned checked above.
            unsafe { std::slice::from_raw_parts(lengths, count) }
        };
        let mut values = crate::raw::try_with_capacity::<Vec<u16>>(count)?;
        let mut at = 0usize;
        for &len in lengths {
            let len = usize::try_from(len)
                .map_err(|_| invalid(format!("fvh highlight: a value of length {len}")))?;
            let end = at
                .checked_add(len)
                .filter(|&e| e <= content.len())
                .ok_or_else(|| invalid("fvh highlight: the lengths overrun the content"))?;
            values.push(content[at..end].to_vec());
            at = end;
        }
        if at != content.len() {
            return Err(invalid(
                "fvh highlight: the lengths do not cover the content",
            ));
        }
        let encoded = highlight(handle, request, segment, doc, &values)?;
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::give(encoded, out_ptr, out_len) };
        Ok(())
    })
}

#[cfg(test)]
mod tests;
