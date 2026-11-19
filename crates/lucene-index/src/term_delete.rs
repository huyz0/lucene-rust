//! Single-segment delete-by-term resolution: "which live doc IDs does
//! `deleteDocuments(new Term(field, bytes))` actually name, in one
//! already-opened segment" — the resolve half of real Lucene's two-phase
//! `BufferedUpdates` (resolve) + `ReadersAndUpdates.writeLiveDocs` (apply)
//! delete flow. [`deletes.rs`](crate::deletes) is the apply half; this module
//! is the resolve half, scoped to one segment.
//!
//! # Why this lives in `lucene-index`, not `lucene-search`
//!
//! The dependency graph is strictly downward: `util ← store ← codecs ← index
//! ← search ← core ← ffi` (see the `architecture` skill).
//! `crates/lucene-search/Cargo.toml` depends on `lucene-index`, so
//! `lucene-index` depending back on `lucene-search` would invert that edge
//! into a cycle. `lucene-search`'s own `term_doc_ids` (in
//! `crates/lucene-search/src/lib.rs`) already does exactly this lookup —
//! `field.postings(term, doc_in)` filtered by `live_docs` — but every
//! primitive it calls (`lucene_codecs::blocktree::BlockTreeFields`,
//! `lucene_codecs::postings::DocInput`) lives in `lucene-codecs`, which
//! `lucene-index` already depends on directly. So rather than duplicate a
//! shared primitive one layer down (which would need a `lucene-codecs`
//! change) or depend upward on `lucene-search` (which would invert the DAG),
//! this module reimplements the same handful of lines directly against
//! `lucene-codecs`: it is a real primitive already living at the right layer,
//! not something that needs promoting or duplicating.
//!
//! # Scope
//!
//! Resolution is scoped to **one already-opened segment** — same shape
//! `lucene-search`'s functions take (`BlockTreeFields` + opened `.doc` file).
//! A real `IndexWriter.deleteDocuments(Term)` resolves against *every*
//! currently-open segment via `BufferedUpdates`/`ReaderPool`; this port has
//! no multi-segment reader/writer orchestration, so that part is still
//! deferred (see `docs/parity.md`). Delete-by-query beyond a single exact
//! term, and real `updateDocument`, are also out of scope here — see
//! `deletes.rs`'s module doc for why.

use lucene_codecs::blocktree::BlockTreeFields;
use lucene_codecs::postings::{DocInput, PostingsFlags};
use lucene_util::fixed_bit_set::FixedBitSet;

use lucene_store::directory::Directory;

use crate::deletes;
use crate::segment_infos::SegmentCommitInfo;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    BlockTree(#[from] lucene_codecs::blocktree::Error),
    #[error(transparent)]
    Deletes(#[from] deletes::Error),
    /// The live docs a caller passed in disagree with the commit's
    /// `delCount` (a reader opened before a later delete): writing them as the
    /// next `.liv` would bring deleted documents back.
    #[error(
        "live docs of segment {segment} have {live} live of {max_doc} documents, but the \
         commit records {del_count} deleted"
    )]
    StaleLiveDocs {
        segment: String,
        live: usize,
        max_doc: usize,
        del_count: i32,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

/// Resolves `(field, term)` to every **live** doc ID matching it in this
/// segment, ascending (same order `Postings::docs` and
/// `lucene-search::term_doc_ids` both already guarantee).
///
/// Mirrors `lucene-search::term_doc_ids`'s exact semantics: an unknown
/// `field` or a `term` not present in that field's dictionary yields an
/// empty `Vec`, not an error (matches `TermQuery.createWeight`'s `null`
/// `Scorer` "no matches" outcome — a delete for a term nothing currently
/// matches is a legitimate no-op, not a caller bug).
///
/// `doc_in` is `None` for segments with no `.doc` file opened; only actually
/// consulted when the term's `docFreq > 1` (see
/// `BlockTreeFields::field`/`FieldTerms::postings`) — passing `None` for a
/// term that turns out to need it surfaces as an [`Error`].
///
/// `live_docs` is the segment's current `.liv` bitset (`None` means "no
/// deletions yet, every doc is live"), same convention `deletes::apply_deletes`
/// and `lucene-search::search_term_query` both already use. Only live docs
/// are returned — a term match on an already-deleted doc contributes nothing
/// new to delete.
pub fn resolve_term_doc_ids(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    field: &str,
    term: &[u8],
) -> Result<Vec<i32>> {
    let Some(field_terms) = fields.field(field) else {
        return Ok(Vec::new());
    };
    // Doc ids only: this path never looks at a frequency, so Java's
    // `PostingsEnum.NONE` is the honest flag mask -- and it turns every
    // 256-value `PForUtil` frequency unpack in the term's `.doc` blocks into
    // a token byte plus a seek (`PForUtil.skip`).
    let Some(postings) = field_terms.postings_with_flags(term, doc_in, PostingsFlags::DocsOnly)?
    else {
        return Ok(Vec::new());
    };
    let Some(bits) = live_docs else {
        // No `.liv` yet: every doc is live, and an out-of-range doc ID is
        // `deletes::mark_deleted`'s to reject (it has `max_doc`; this function
        // does not).
        return Ok(postings.docs);
    };
    // `FixedBitSet::get` indexes `words[index >> 6]`, so a doc ID the postings
    // decoder produced out of range -- negative, or past this segment's
    // `maxDoc` -- is an **index panic** here, in release as well as debug,
    // rather than the `DocOutOfRange` error the apply half would have raised
    // for the identical value had `live_docs` been `None`. Checking the bound
    // makes both paths report the same corruption. The check is a compare
    // against a value hoisted out of the loop, not a per-doc reload.
    let max_doc = bits.len();
    let mut out = Vec::new();
    for &doc_id in &postings.docs {
        let idx = usize::try_from(doc_id)
            .ok()
            .filter(|idx| *idx < max_doc)
            .ok_or(deletes::Error::DocOutOfRange { doc_id, max_doc })?;
        if bits.get(idx) {
            out.push(doc_id);
        }
    }
    Ok(out)
}

/// The full single-segment "resolve then apply" delete-by-term flow: resolves
/// `(field, term)` to its matching live doc IDs via [`resolve_term_doc_ids`],
/// then applies them via [`deletes::apply_deletes`] exactly as if a caller had
/// hand-picked those doc IDs — closing the gap between "here's a term to
/// delete" and "here's the actual `mark_deleted`/`apply_deletes` call" for one
/// segment. Matches real Lucene's per-segment resolve-then-apply shape;
/// resolving across an `IndexWriter`'s whole open segment set is still out of
/// scope (see this module's doc comment).
///
/// Parameters are the union of [`resolve_term_doc_ids`]'s and
/// [`deletes::apply_deletes`]'s: `fields`/`doc_in` to resolve the term,
/// `current_live_docs`/`max_doc` for both resolution (live-doc filtering) and
/// application (bitset sizing/bounds-checking), `dir`/`sci` to write the
/// updated `.liv` file and produce the next `SegmentCommitInfo`.
#[allow(clippy::too_many_arguments)]
pub fn resolve_and_apply_term_delete(
    dir: &dyn Directory,
    sci: &SegmentCommitInfo,
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    current_live_docs: Option<&FixedBitSet>,
    max_doc: usize,
    field: &str,
    term: &[u8],
) -> Result<SegmentCommitInfo> {
    let live = current_live_docs.map_or(max_doc, FixedBitSet::cardinality);
    let expected = usize::try_from(sci.del_count)
        .ok()
        .and_then(|d| max_doc.checked_sub(d));
    if current_live_docs.is_some_and(|bits| bits.len() != max_doc) || expected != Some(live) {
        return Err(Error::StaleLiveDocs {
            segment: sci.segment_name.clone(),
            live,
            max_doc,
            del_count: sci.del_count,
        });
    }
    let doc_ids = resolve_term_doc_ids(fields, doc_in, current_live_docs, field, term)?;
    Ok(deletes::apply_deletes(
        dir,
        sci,
        current_live_docs,
        max_doc,
        doc_ids,
    )?)
}

#[cfg(test)]
mod tests;
