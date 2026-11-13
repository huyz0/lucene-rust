//! OpenSearch's `unified` highlighter for the fetch phase
//! ([`lucene_highlighter::unified::opensearch`]): the plugin hands over a
//! field's request (query, offset source, analyzer, options) and one hit's
//! loaded value, and gets the hit's snippets back.
//!
//! A request is built once per field per search on the JVM side and sent
//! with every hit; the highlighter it describes (the query's terms,
//! automata and phrase spans) is built on its first use on a thread and kept
//! for the next hits ([`CACHED`]).
//!
//! Request layout (little-endian, lengths `i32`):
//!
//! | field | |
//! |---|---|
//! | `version: u8` | [`REQUEST_VERSION`] |
//! | `field: bytes` | UTF-8 |
//! | `query: bytes` | a query blob ([`lucene_engine::query_blob::decode_query`]) |
//! | `offset_source: u8` | `0` analysis, `1` postings, `2` postings with term vectors, `3` term vectors |
//! | `analyzers: i32`, then per analyzer `field: bytes`, `spec: bytes` | the index analyzers of the field and its `matched_fields` that are described ([`lucene_engine::analysis`]) |
//! | `offset_limit: i32` | `max_analyzer_offset` (a `LimitTokenOffsetFilter`), `-1` for none |
//! | `breaks: u8`, `max_len: i32` | `0` one passage per value, `1` sentences, `2` sentences bounded to `max_len`, `3` words |
//! | `max_passages: i32` | `number_of_fragments` (`i32::MAX - 1` for all) |
//! | `no_match_size: i32` | |
//! | `pre_tag: bytes`, `post_tag: bytes` | UTF-8 |
//! | `html: u8` | the `html` encoder |
//! | `require_field_match: u8` | `1`: only the field and `matched_fields` count; `0`: every field but `_id` |
//! | `matched: i32`, then `matched` names (`bytes`) | `matched_fields` |
//! | `max_analyzed_offset: i32` | `index.highlight.max_analyzed_offset` |
//!
//! Result layout: `status: u8` ([`STATUS_SNIPPETS`], [`STATUS_NOOP`],
//! [`STATUS_OFFSET_LIMIT`], [`STATUS_TOO_LONG`], [`STATUS_NEEDS_ANALYZER`]), then for snippets `count:
//! i32` and per snippet `score: f32`, `highlighted: u8`, `len: i32` and
//! `len` UTF-16 units.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use lucene_analysis::{Analyzer, AnalyzerWrapper, TokenStreamComponents};
use lucene_codecs::term_vectors::ChunkCursor;
use lucene_highlighter::highlighter::OffsetSource;
use lucene_highlighter::unified::opensearch::{Breaks, CustomUnifiedHighlighter, Options, Snippet};
use lucene_highlighter::unified::strategies::LeafCtx;
use lucene_search::query::Clause;

use crate::error::{set_last_error, FfiStatus};
use crate::jvm_reader::{self};
use lucene_engine::query_blob::{Cursor, JvmQuery};

/// The request layout's version.
pub(crate) const REQUEST_VERSION: u8 = 1;

/// The snippets follow.
pub(crate) const STATUS_SNIPPETS: u8 = 0;
/// The query matches nothing in the field and `no_match_size` is 0: no
/// document of this request has anything to show.
pub(crate) const STATUS_NOOP: u8 = 1;
/// `max_analyzer_offset` is above `index.highlight.max_analyzed_offset`.
pub(crate) const STATUS_OFFSET_LIMIT: u8 = 2;
/// The value is longer than `index.highlight.max_analyzed_offset` and would
/// be re-analyzed whole.
pub(crate) const STATUS_TOO_LONG: u8 = 3;
/// The content has to be re-analyzed (the offset source, or the query, asks
/// for `ANALYSIS`) with an analyzer the request does not describe.
pub(crate) const STATUS_NEEDS_ANALYZER: u8 = 4;

/// At most this many highlighters are kept per thread.
const CACHED_PER_THREAD: usize = 16;

/// A request's highlighter, built.
struct Prepared {
    highlighter: CustomUnifiedHighlighter,
    analysis: bool,
    needs_analyzer: bool,
    offset_limit: Option<i32>,
    max_analyzed_offset: i32,
}

thread_local! {
    /// The thread's recent requests and their highlighters, the latest last.
    static CACHED: RefCell<Vec<(Box<[u8]>, Prepared)>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn invalid(message: impl Into<String>) -> FfiStatus {
    set_last_error(message.into());
    FfiStatus::InvalidArgument
}

pub(crate) fn utf8<'a>(b: &'a [u8], what: &str) -> Result<&'a str, FfiStatus> {
    std::str::from_utf8(b).map_err(|_| invalid(format!("highlight request: {what} is not UTF-8")))
}

/// The index analyzers of the described fields, with `max_analyzer_offset`'s
/// `LimitTokenOffsetFilter` behind each chain
/// (`UnifiedHighlighter.getLimitedOffsetAnalyzer`). A field without one
/// gets `missing`, which never analyzes: a request whose highlighter would
/// re-analyze such a field answers [`STATUS_NEEDS_ANALYZER`].
pub(crate) struct Limited {
    pub(crate) by_field: HashMap<String, Arc<lucene_analysis::factory::CustomAnalyzer>>,
    pub(crate) missing: Analyzer,
    pub(crate) limit: Option<i32>,
}

impl AnalyzerWrapper for Limited {
    fn wrapped_analyzer(&self, field_name: &str) -> &Analyzer {
        match self.by_field.get(field_name) {
            Some(a) => a.analyzer(),
            None => &self.missing,
        }
    }

    fn wrap_components(
        &self,
        _field_name: &str,
        components: TokenStreamComponents,
    ) -> TokenStreamComponents {
        let Some(limit) = self.limit else {
            return components;
        };
        let (source, sink) = components.into_parts();
        match lucene_analysis::miscellaneous::LimitTokenOffsetFilter::new(sink, limit, false) {
            Ok(f) => TokenStreamComponents::from_parts(source, Box::new(f)),
            // a negative limit: refused when the request is decoded
            Err(_) => unreachable!("limit checked non-negative"),
        }
    }
}

/// The query of a request, as the clause tree the highlighter visits.
pub(crate) fn decode_clause(blob: &[u8]) -> Result<Clause, FfiStatus> {
    if blob.first() == Some(&lucene_engine::query_blob::QUERY_TREE) {
        let mut c = Cursor::new(&blob[1..]);
        let mut nodes = 0usize;
        let clause = lucene_engine::query_blob::decode_node(&mut c, 0, &mut nodes)?;
        if c.remaining() != 0 {
            return Err(invalid(format!(
                "highlight request: {} trailing bytes after the query",
                c.remaining()
            )));
        }
        return Ok(clause);
    }
    Ok(match lucene_engine::query_blob::decode_query(blob)? {
        JvmQuery::Term(t) => Clause::Term(t),
        JvmQuery::Boolean(b) => Clause::Boolean(Box::new(b)),
    })
}

fn decode(blob: &[u8]) -> Result<Prepared, FfiStatus> {
    let mut c = Cursor::new(blob);
    let version = c.u8()?;
    if version != REQUEST_VERSION {
        return Err(invalid(format!(
            "highlight request: version {version}, expected {REQUEST_VERSION}"
        )));
    }
    let field = utf8(c.bytes()?, "the field")?.to_string();
    let query = decode_clause(c.bytes()?)?;
    let offset_source = match c.u8()? {
        0 => OffsetSource::Analysis,
        1 => OffsetSource::Postings,
        2 => OffsetSource::PostingsWithTermVectors,
        3 => OffsetSource::TermVectors,
        s => return Err(invalid(format!("highlight request: offset source {s}"))),
    };
    let mut specs = Vec::new();
    for _ in 0..c.length()? {
        let f = utf8(c.bytes()?, "an analyzer's field")?.to_string();
        specs.push((f, c.bytes()?));
    }
    let offset_limit = match c.i32()? {
        -1 => None,
        l if l >= 0 => Some(l),
        l => return Err(invalid(format!("highlight request: offset limit {l}"))),
    };
    let kind = c.u8()?;
    let max_len = c.i32()?;
    let breaks = match kind {
        0 => Breaks::Values,
        1 => Breaks::Sentence,
        2 => Breaks::BoundedSentence(max_len),
        3 => Breaks::Word,
        k => return Err(invalid(format!("highlight request: break iterator {k}"))),
    };
    let max_passages = c.i32()?;
    let no_match_size = c.i32()?;
    let pre_tag = utf8(c.bytes()?, "the pre tag")?.to_string();
    let post_tag = utf8(c.bytes()?, "the post tag")?.to_string();
    let html = c.u8()? != 0;
    let require_field_match = c.u8()? != 0;
    let mut matched = BTreeSet::new();
    for _ in 0..c.length()? {
        matched.insert(utf8(c.bytes()?, "a matched field")?.to_string());
    }
    let max_analyzed_offset = c.i32()?;
    if c.remaining() != 0 {
        return Err(invalid(format!(
            "highlight request: {} trailing bytes",
            c.remaining()
        )));
    }
    let analysis = offset_source == OffsetSource::Analysis;
    let mut by_field = HashMap::new();
    for (f, spec) in specs {
        by_field.insert(f, lucene_engine::analysis::analyzer(spec)?);
    }
    let described: BTreeSet<String> = by_field.keys().cloned().collect();
    let analyzer = Some(Arc::new(Analyzer::new(Limited {
        by_field,
        missing: Analyzer::keyword(),
        limit: offset_limit,
    })));
    // UnifiedHighlighter.fieldMatcher
    let field_matcher: lucene_highlighter::unified::strategies::FieldMatcher =
        if require_field_match {
            let own = field.clone();
            let matched = matched.clone();
            Arc::new(move |f: &str| f == own || matched.contains(f))
        } else {
            // terms of the _id field use another encoding, not UTF-8
            Arc::new(|f: &str| f != "_id")
        };
    let masked_fields: Option<lucene_highlighter::unified::MaskedFields> = if matched.is_empty() {
        None
    } else {
        let own = field.clone();
        Some(Arc::new(move |f: &str| {
            if f == own {
                Some(matched.clone())
            } else {
                Some(BTreeSet::new())
            }
        }))
    };
    let options = Options {
        field,
        offset_source,
        breaks,
        max_passages,
        no_match_size,
        pre_tag,
        post_tag,
        html,
        field_matcher,
        masked_fields,
    };
    let highlighter = CustomUnifiedHighlighter::new(options, &query, analyzer)
        .map_err(|e| invalid(format!("highlight request: {}", e.message())))?;
    let needs_analyzer = highlighter
        .analysis_fields()
        .any(|f| !described.contains(f));
    Ok(Prepared {
        highlighter,
        analysis,
        needs_analyzer,
        offset_limit,
        max_analyzed_offset,
    })
}

/// Runs `f` on the request's highlighter, built on its first use on this
/// thread.
fn with_prepared<T>(
    request: &[u8],
    f: impl FnOnce(&mut Prepared) -> Result<T, FfiStatus>,
) -> Result<T, FfiStatus> {
    CACHED.with(|cell| {
        let mut cache = cell.borrow_mut();
        let found = cache.iter().position(|(k, _)| &**k == request);
        let idx = match found {
            Some(i) => {
                let last = cache.len() - 1;
                if i != last {
                    let entry = cache.remove(i);
                    cache.push(entry);
                }
                last
            }
            None => {
                let prepared = decode(request)?;
                if cache.len() >= CACHED_PER_THREAD {
                    cache.remove(0);
                }
                cache.push((request.into(), prepared));
                cache.len() - 1
            }
        };
        // taken out while it runs: a failed or panicking hit leaves no
        // half-used highlighter behind
        let (key, mut prepared) = cache.remove(idx);
        drop(cache);
        let result = f(&mut prepared);
        if result.is_ok() {
            cell.borrow_mut().push((key, prepared));
        }
        result
    })
}

pub(crate) fn put_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn encode(snippets: &[Snippet]) -> Result<Vec<u8>, FfiStatus> {
    let units: usize = snippets.iter().map(|s| s.text.len()).sum();
    let mut out = crate::raw::try_with_capacity::<u8>(5 + snippets.len() * 9 + units * 2)?;
    out.push(STATUS_SNIPPETS);
    put_i32(&mut out, i32::try_from(snippets.len()).unwrap_or(i32::MAX));
    for s in snippets {
        out.extend_from_slice(&s.score.to_bits().to_le_bytes());
        out.push(u8::from(s.highlighted));
        let len = i32::try_from(s.text.len())
            .map_err(|_| invalid("highlight: a snippet longer than 2^31 units"))?;
        put_i32(&mut out, len);
        for u in &s.text {
            out.extend_from_slice(&u.to_le_bytes());
        }
    }
    Ok(out)
}

/// One hit's snippets: segment `segment`'s document `doc` (leaf-local) of
/// `handle`'s reader, whose loaded value is `content`, highlighted as
/// `request` asks.
pub(crate) fn highlight(
    handle: u64,
    request: &[u8],
    segment: i32,
    doc: i32,
    content: &[u16],
) -> Result<Vec<u8>, FfiStatus> {
    with_prepared(request, |p| {
        if p.needs_analyzer {
            return Ok(vec![STATUS_NEEDS_ANALYZER]);
        }
        // CustomUnifiedHighlighter.highlightField
        if p.highlighter.is_noop() && p.no_match_size_is_zero() {
            return Ok(vec![STATUS_NOOP]);
        }
        if p.offset_limit.is_some_and(|l| l > p.max_analyzed_offset) {
            return Ok(vec![STATUS_OFFSET_LIMIT]);
        }
        let too_long = i32::try_from(content.len()).map_or(true, |n| n > p.max_analyzed_offset);
        if p.analysis && too_long && p.offset_limit.is_none() {
            return Ok(vec![STATUS_TOO_LONG]);
        }
        let h = jvm_reader::lookup(
            handle,
            "ffi_jvm_highlight_unified_alloc: unknown or already-closed handle",
        )?;
        let readers = h.reader.segment_readers();
        let Some(leaf) = usize::try_from(segment).ok().filter(|&s| s < readers.len()) else {
            return Err(invalid(format!(
                "highlight: segment {segment} of {}",
                readers.len()
            )));
        };
        let reader = &readers[leaf];
        if doc < 0 || doc >= reader.max_doc {
            return Err(invalid(format!(
                "highlight: doc {doc} of segment {segment} with maxDoc {}",
                reader.max_doc
            )));
        }
        let opened = h.reader.open_segments().map_err(|e| {
            set_last_error(format!("opening segment postings: {e}"));
            FfiStatus::Decode
        })?;
        let segments = opened.as_open_segments();
        let seg = &segments[leaf];
        let matches =
            |c: &Clause| lucene_search::matches::leaf_matches(seg, &c.clone().rewrite(), doc);
        let cursor = RefCell::new(ChunkCursor::new());
        let term_vector = |field: i32| -> lucene_search::Result<_> {
            let Some(tvs) = reader.term_vectors_reader()? else {
                return Ok(None);
            };
            Ok(cursor.borrow_mut().field(&tvs, doc, field)?)
        };
        let ctx = LeafCtx {
            reader,
            segment: seg,
            doc,
            matches: &matches,
            term_vector: &term_vector,
        };
        let snippets = p.highlighter.highlight(Some(&ctx), content).map_err(|e| {
            set_last_error(format!("{}: {}", e.class(), e.message()));
            FfiStatus::Search
        })?;
        encode(&snippets)
    })
}

impl Prepared {
    fn no_match_size_is_zero(&self) -> bool {
        self.highlighter.no_match_size() == 0
    }
}

/// [`highlight`]: one hit's snippets as an owned buffer
/// ([`crate::ffm_bridge::ffi_jvm_free_bytes`]). `request` is the field's
/// request, `content` the hit's value (`content_len` UTF-16 units).
///
/// # Safety
/// `request` must be valid for `request_len` bytes, `content` for
/// `content_len` `u16` reads and aligned to 2 (null when 0), `out_ptr`/`out_len` for one write
/// each.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_jvm_highlight_unified_alloc(
    handle: u64,
    request: *const u8,
    request_len: usize,
    segment: i32,
    doc: i32,
    content: *const u16,
    content_len: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    crate::error::guard(|| {
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::clear_out(out_ptr, out_len)? };
        // SAFETY: caller contract.
        let request = unsafe { crate::raw::bytes_from_raw(request, request_len)? };
        if content.is_null() && content_len > 0 {
            return Err(FfiStatus::NullPointer);
        }
        if content_len > 0 && !content.is_aligned() {
            return Err(invalid("highlight: content not aligned to 2"));
        }
        let content: &[u16] = if content_len == 0 {
            &[]
        } else {
            // SAFETY: caller contract, non-null checked above.
            unsafe { std::slice::from_raw_parts(content, content_len) }
        };
        let encoded = highlight(handle, request, segment, doc, content)?;
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::give(encoded, out_ptr, out_len) };
        Ok(())
    })
}

#[cfg(test)]
mod tests;
