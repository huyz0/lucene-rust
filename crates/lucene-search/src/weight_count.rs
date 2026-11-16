//! `Weight.count(LeafReaderContext)`
//! (`lucene/core/src/java/org/apache/lucene/search/Weight.java`)
//! and the two reader-level shortcuts built on it.
//!
//! ## What Java does with this
//!
//! `Weight.count` answers "how many documents in this leaf match" **without
//! producing them**, or `-1` for "no shortcut, go and iterate".
//! `TotalHitCountCollector.getLeafCollector` asks first and throws
//! `CollectionTerminatedException` when the answer is not `-1`, so
//! `IndexSearcher.count(query)` never opens a postings list for a query that
//! can answer from metadata. Three queries override it:
//!
//! | query | shortcut |
//! |---|---|
//! | `TermQuery` | `termsEnum.docFreq()`, but **only** when the leaf has no deletions |
//! | `MatchAllDocsQuery` | `reader.numDocs()` |
//! | `FieldExistsQuery` | a per-source doc count, reconciled against `maxDoc`/`numDocs` |
//!
//! This port has no `Weight`, so each override is a free function taking the
//! same inputs its query's search function already takes. `Option<i64>` stands
//! in for Java's `-1` sentinel.
//!
//! ## The `Option` is the whole feature
//!
//! `Some(n)` is an exact answer derived from the terms dictionary's own
//! `docFreq`/`docCount` -- no `.doc` file is opened, no block is decoded.
//! `None` is Java's `-1`: the caller must iterate. [`count_term_query`] does
//! both halves, so a caller that only wants a number never has to know which
//! branch it got.

use lucene_codecs::blocktree::BlockTreeFields;
use lucene_codecs::postings::DocInput;
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::collector::CountCollector;
use crate::doc_value_query::{field_exists_leaf_is_complete, FieldExistsSource};
use crate::query::TermQuery;
use crate::Result;

/// `TermQuery.TermWeight.count(LeafReaderContext)`:
///
/// ```java
/// if (context.reader().hasDeletions() == false) {
///   TermsEnum termsEnum = getTermsEnum(context);
///   return termsEnum != null ? termsEnum.docFreq() : 0;
/// }
/// return super.count(context);
/// ```
///
/// `live_docs` **is** this port's `hasDeletions()`: every search function in
/// this crate takes the leaf's live-document bitset as `Option<&FixedBitSet>`,
/// and `None` means "no document is deleted". A caller that has deletions and
/// passes `None` is already wrong everywhere else in this crate, so the same
/// contract is used rather than a second, redundant flag.
///
/// A field that is not in this segment, or a term that is not in its
/// dictionary, is `Ok(Some(0))` -- Java's "the term cannot be found in the
/// dictionary so the count is 0", not `Ok(None)`. Nothing has to be iterated
/// to know that nothing matches. `Ok(None)` means only "no shortcut applies".
///
/// A corrupt `.tim` block is an `Err`, not a count of zero: `getTermsEnum`
/// throws `IOException` in Java and `count` propagates it, and "this term is
/// absent" is a legitimate answer here, so degrading corruption to it would
/// hand back a plausible wrong number.
pub fn count_term_query_shortcut(
    fields: &BlockTreeFields,
    live_docs: Option<&FixedBitSet>,
    query: &TermQuery,
) -> Result<Option<i64>> {
    if live_docs.is_some() {
        // `super.count(context)` -- the docFreq counts deleted documents, so it
        // is not the answer, and there is no cheaper one.
        return Ok(None);
    }
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Some(0));
    };
    Ok(Some(
        field_terms
            .try_seek_exact(&query.term)?
            .map_or(0, |stats| i64::from(stats.doc_freq)),
    ))
}

/// `IndexSearcher.count(new TermQuery(...))` for one leaf: the
/// [`count_term_query_shortcut`] when there is one, and
/// `TotalHitCountCollector`'s own per-document count otherwise.
///
/// `doc_in` is only touched on the iterating branch, so a caller with no
/// deletions may pass `None` and never open `.doc` at all.
pub fn count_term_query(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermQuery,
) -> Result<i64> {
    if let Some(n) = count_term_query_shortcut(fields, live_docs, query)? {
        return Ok(n);
    }
    let mut counter = CountCollector::default();
    crate::search_term_query(fields, doc_in, live_docs, query, &mut counter)?;
    Ok(i64::from(counter.count))
}

/// `MatchAllDocsQuery`'s `count`: `context.reader().numDocs()`, which is
/// `maxDoc()` minus the deleted documents.
///
/// Note that Java's `MatchAllDocsQuery` weight is one of only two places in
/// `lucene-core` where `count` needs no `-1` branch at all -- there is nothing
/// a scan could discover that `numDocs()` does not already say.
pub fn count_match_all_docs(max_doc: i32, live_docs: Option<&FixedBitSet>) -> i64 {
    match live_docs {
        None => i64::from(max_doc.max(0)),
        // `FixedBitSet.cardinality()`, exactly as `SegmentReader.numDocs()`
        // derives its answer from `maxDoc - delCount`.
        Some(bits) => bits.cardinality() as i64,
    }
}

/// Everything one leaf contributes to `FieldExistsQuery`'s `count` and
/// `rewrite` decisions -- the counts Java reads off the `LeafReader` inside
/// both loops, gathered once so the two rules can be written as pure
/// functions.
///
/// `source: None` is Java's `fieldInfos.fieldInfo(field) == null`: the field is
/// not in this leaf at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldExistsLeaf {
    /// Which of the three sources this leaf's `FieldInfo` selects --
    /// [`crate::doc_value_query::field_exists_source`].
    pub source: Option<FieldExistsSource>,
    /// `reader.maxDoc()`.
    pub max_doc: i32,
    /// `reader.numDocs()`; `max_doc` when the leaf has no deletions.
    pub num_docs: i32,
    /// `fieldInfo.docValuesSkipIndexType() != DocValuesSkipIndexType.NONE`.
    ///
    /// The three flags below are `FieldInfo` bits, not "did the lookup
    /// succeed": Java's `count` selects its doc-count proxy by the *declared*
    /// structure and then treats a missing reader as a count of `0`, which is a
    /// different answer from "try the next proxy". Carrying the flags is what
    /// lets this reproduce that.
    pub has_doc_values_skip_index: bool,
    /// `fieldInfo.getPointDimensionCount() > 0`.
    pub has_points: bool,
    /// `fieldInfo.getIndexOptions() != IndexOptions.NONE`.
    pub is_indexed: bool,
    /// `reader.terms(field).getDocCount()`, `None` when the field has no terms.
    pub terms_doc_count: Option<i32>,
    /// `reader.getFloatVectorValues(field).size()` / the byte equivalent.
    pub vector_size: Option<i32>,
    /// `reader.getPointValues(field).getDocCount()`.
    pub points_doc_count: Option<i32>,
    /// `reader.getDocValuesSkipper(field).docCount()`.
    pub skipper_doc_count: Option<i32>,
}

impl FieldExistsLeaf {
    /// `reader.hasDeletions()`.
    fn has_deletions(&self) -> bool {
        self.num_docs != self.max_doc
    }

    /// Java's `int count = -1; ...` block, verbatim, before the reconciliation
    /// against `maxDoc`/`numDocs` below. `-1` is "nothing to count from".
    ///
    /// ```java
    /// if (fieldInfo.hasVectorValues())        count = getVectorValuesSize(fieldInfo, reader);
    /// else if (dvType != NONE) {
    ///   if (skipIndexType != NONE)            count = skipper == null ? 0 : skipper.docCount();
    ///   else if (reader.hasDeletions() == false) {
    ///     if (pointDimensionCount > 0)        count = pv == null ? 0 : pv.getDocCount();
    ///     else if (indexOptions != NONE)      count = terms == null ? 0 : terms.getDocCount();
    ///   }
    /// } else throw new IllegalStateException(...);
    /// ```
    ///
    /// Two details that are easy to lose and both change an answer. The arms
    /// are chosen by **`FieldInfo` flags**, not by which structure happens to
    /// resolve, so a field that declares a skip index but whose skipper cannot
    /// be read counts `0` rather than falling through to points or terms. And
    /// the points/terms proxies are gated on `hasDeletions() == false`: a
    /// *doc*-count says nothing about which of those documents are live, so
    /// with deletions there is nothing to shortcut from and the query has to be
    /// run.
    ///
    /// The one place this cannot follow Java is a binary vector field whose
    /// `vector_size` the caller did not supply (see
    /// [`crate::directory_reader::SegmentReader::field_exists_leaf`], where it
    /// is a parameter). Java asserts the values are non-null and so never has
    /// the case; this answers `-1`, "go and scan", because the alternative --
    /// reading it as `0` -- would turn a caller's omission into "no document
    /// has this field", a wrong count with no signal.
    //
    // SENTINEL: `-1` = Java's "no shortcut, run the query", outside the domain
    // of a document count.
    fn raw_count(&self) -> i32 {
        match self.source {
            Some(FieldExistsSource::Vectors) => self.vector_size.unwrap_or(-1),
            Some(FieldExistsSource::DocValues) => {
                if self.has_doc_values_skip_index {
                    self.skipper_doc_count.unwrap_or(0)
                } else if self.has_deletions() {
                    -1
                } else if self.has_points {
                    self.points_doc_count.unwrap_or(0)
                } else if self.is_indexed {
                    self.terms_doc_count.unwrap_or(0)
                } else {
                    -1
                }
            }
            // Norms is handled before this is reached, and a field with no
            // source at all is Java's `fieldInfo == null`.
            None | Some(FieldExistsSource::Norms) => -1,
        }
    }
}

/// `FieldExistsQuery`'s `ConstantScoreWeight.count(LeafReaderContext)`, whole.
///
/// ```java
/// if (fieldInfo == null) return 0;
/// if (fieldInfo.hasNorms()) {
///   if (reader.getDocCount(field) == reader.maxDoc()) return reader.numDocs();
///   return super.count(context);
/// }
/// int count = ...;                       // vectors / skipper / points / terms
/// if (count == 0) return 0;
/// else if (count == reader.maxDoc()) return reader.numDocs();
/// else if (count >= 0 && reader.hasDeletions() == false) return count;
/// return super.count(context);
/// ```
///
/// In the last three lines, a leaf where every
/// document has the field answers `numDocs()` (deletions and all), a leaf with
/// no deletions answers the raw count directly, and only the "some documents
/// lack the field *and* some are deleted" case has to intersect the two sets by
/// scanning. `None` is Java's `-1` for that case.
///
/// The **norms** branch has no `count` of its own: Java only shortcuts it when
/// the field is complete, because `getDocCount(field)` there is the *terms*
/// dictionary's count and Java does not treat it as a norms count.
pub fn count_field_exists_leaf(leaf: &FieldExistsLeaf) -> Option<i64> {
    let Some(source) = leaf.source else {
        return Some(0);
    };
    if source == FieldExistsSource::Norms {
        if leaf.terms_doc_count == Some(leaf.max_doc) {
            return Some(i64::from(leaf.num_docs));
        }
        return None;
    }
    let count = leaf.raw_count();
    if count == 0 {
        // One of the sources says the field is not present on this leaf at all.
        Some(0)
    } else if count == leaf.max_doc {
        // Every document (live or deleted) has the field.
        Some(i64::from(leaf.num_docs))
    } else if count >= 0 && !leaf.has_deletions() {
        Some(i64::from(count))
    } else {
        None
    }
}

/// `FieldExistsQuery.rewrite(IndexSearcher)`'s whole-reader decision: `true`
/// when every leaf has the field on every one of its documents, in which case
/// Java returns `MatchAllDocsQuery.INSTANCE`.
///
/// **Two asymmetries of Java's that are reproduced here rather than tidied.**
///
/// 1. The norms branch reads `reader.getDocCount(field)` and `reader.maxDoc()`
///    off the **top-level** `IndexReader` while the vector and doc-values
///    branches read the *leaf*'s -- inside the same per-leaf loop. So a
///    norms-sourced field is decided once, reader-wide, and the per-leaf
///    numbers are never consulted; hence `reader_terms_doc_count`/
///    `reader_max_doc`, which are the only two values a caller has to gather
///    beyond the leaves themselves. (For a single-leaf reader the two
///    coincide, which is presumably why it has survived.)
/// 2. A leaf whose `FieldInfo` is missing breaks the loop with
///    `allReadersRewritable = false`, so a field absent from one segment blocks
///    the rewrite even if every other segment is complete -- correct, since
///    those documents genuinely do not match.
///
/// An **empty** `leaves` rewrites to `MatchAllDocsQuery`, which is Java's own
/// answer (`allReadersRewritable` starts `true` and the loop never runs). A
/// reader with no leaves has no documents, so both queries match nothing.
pub fn field_exists_rewrites_to_match_all_docs(
    leaves: &[FieldExistsLeaf],
    reader_terms_doc_count: Option<i32>,
    reader_max_doc: i32,
) -> bool {
    leaves.iter().all(|leaf| match leaf.source {
        None => false,
        Some(FieldExistsSource::Norms) => field_exists_leaf_is_complete(
            FieldExistsSource::Norms,
            reader_max_doc,
            reader_terms_doc_count,
            None,
            None,
            None,
        ),
        Some(source) => field_exists_leaf_is_complete(
            source,
            leaf.max_doc,
            leaf.terms_doc_count,
            leaf.vector_size,
            leaf.points_doc_count,
            leaf.skipper_doc_count,
        ),
    })
}

#[cfg(test)]
mod tests;
