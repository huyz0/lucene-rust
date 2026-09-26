//! `terminate_after`: OpenSearch's `EarlyTerminatingCollector` in front of a
//! sequential search's top-docs collector.
//!
//! OpenSearch never runs a `terminate_after` request concurrently
//! (`DefaultSearchContext.evaluateRequestShouldUseConcurrentSearch`), so its
//! collector sees the query's live matches in index order -- segment by
//! segment, each in doc-id order -- and lets the first `n` through. The
//! `n + 1`th match, or the next segment once `n` are in, throws the
//! `EarlyTerminationException` that ends the search (`getLeafCollector` checks
//! the count before a segment's first match, so a later segment ends it even
//! with no match of its own).
//!
//! That makes the collected documents a prefix of the index: every match up to
//! the `n`th. [`terminate_after`] finds where it ends, and
//! [`search_sorted_until`] runs the ordinary sorted search over that prefix
//! only -- the segments before the cut, and the cut segment's documents up to
//! its last collected match -- with collection statistics still taken from
//! every segment, as Lucene's are.

use lucene_util::fixed_bit_set::FixedBitSet;

use crate::directory_reader::SegmentReader;
use crate::exec::{self, Mode, NO_MORE_DOCS};
use crate::field_norms::FieldNorms;
use crate::multi_segment::OpenSegment;
use crate::query::BooleanQuery;
use crate::top_field::{FieldDoc, SortField, TopFieldDocs};
use crate::Result;

use std::collections::HashMap;

/// Where `terminate_after` ends a sequential search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cut {
    /// Matches let through: `n`, or every match when there are fewer.
    pub collected: u64,
    /// Whether the search ends early (`QuerySearchResult.terminatedEarly`):
    /// a match past the `n`th, or a segment after the one holding it.
    pub terminated: bool,
    /// The segments searched: `0..leaves`.
    pub leaves: usize,
    /// The last searched segment's last collected match, when the cut falls
    /// inside it (`n` reached); `None` when every match is collected.
    pub last: Option<(usize, i32)>,
    /// Matches let through per searched segment.
    pub per_leaf: Vec<u64>,
}

/// The first `n` live matches of `query` over `segments` in index order, and
/// whether a search collecting them stops early.
///
/// # Errors
/// A postings or points read failure.
pub fn terminate_after(segments: &[OpenSegment<'_>], query: &BooleanQuery, n: u64) -> Result<Cut> {
    let rewritten = crate::multi_segment::rewrite_points_ranges(query, segments);
    let query = rewritten.as_ref().unwrap_or(query);
    let mut collected = 0u64;
    let mut per_leaf = Vec::new();
    for (i, seg) in segments.iter().enumerate() {
        per_leaf.push(0u64);
        let ctx = exec::LeafContext {
            fields: seg.fields,
            doc_in: seg.doc_in,
            pos_in: seg.pos_in,
            pay_in: seg.pay_in,
            live_docs: seg.live_docs,
            points: seg.points,
            norms: None,
            global: None,
            max_doc: seg.max_doc,
            cache: seg.cache,
        };
        let Some(mut scorer) = exec::build::build_boolean(&ctx, query, 1.0, Mode::NoScores, true)?
        else {
            // No scorer, but `getLeafCollector` is still asked: once `n` are in,
            // this segment ends the search -- which the return below already
            // said for the segment that reached `n`.
            continue;
        };
        let mut last = -1;
        let mut here = 0u64;
        let mut doc = exec::exact_next(scorer.as_mut())?;
        while doc != NO_MORE_DOCS {
            if seg.live_docs.is_none_or(|l| l.get_doc(doc)) {
                if collected == n {
                    per_leaf[i] = here;
                    return Ok(Cut {
                        collected,
                        terminated: true,
                        leaves: i + 1,
                        last: Some((i, last)),
                        per_leaf,
                    });
                }
                // ARITH: here <= collected < n <= u64::MAX here.
                collected += 1;
                here += 1;
                last = doc;
            }
            doc = exec::exact_next(scorer.as_mut())?;
        }
        per_leaf[i] = here;
        if collected == n && last >= 0 {
            return Ok(Cut {
                collected,
                // ARITH: i < segments.len().
                terminated: i + 1 < segments.len(),
                leaves: i + 1,
                last: Some((i, last)),
                per_leaf,
            });
        }
    }
    Ok(Cut {
        collected,
        terminated: false,
        leaves: segments.len(),
        last: None,
        per_leaf,
    })
}

/// The total a `size: 0` search behind `terminate_after` reports:
/// OpenSearch's `TotalHitCountCollector` sits beside the terminating
/// collector, and on each segment it is asked for takes `Weight.count` when
/// there is one -- the whole segment's matches, past the cut -- and counts the
/// documents let through otherwise. `None` when `query` is not one whose
/// `Weight.count` this port has ([`leaf_count`]).
///
/// # Errors
/// A terms-dictionary read failure.
pub fn count_until(
    segments: &[OpenSegment<'_>],
    query: &BooleanQuery,
    cut: &Cut,
) -> Result<Option<u64>> {
    let mut total = 0u64;
    for (seg, &let_through) in segments.iter().zip(&cut.per_leaf) {
        let n = match leaf_count(seg, query)? {
            LeafCount::Whole(n) => n,
            LeafCount::Iterate => let_through,
            LeafCount::Unknown => return Ok(None),
        };
        total = total.saturating_add(n);
    }
    Ok(Some(total))
}

/// `Weight.count(leaf)` for the queries it is ported for.
enum LeafCount {
    /// Answered without iterating.
    Whole(u64),
    /// `-1`: the collector counts what reaches it.
    Iterate,
    /// A query whose `Weight.count` is not ported here.
    Unknown,
}

/// `TermWeight.count` (the term's `docFreq`, without deletions),
/// `MatchAllDocsQuery`'s (`numDocs`) and `MatchNoDocsQuery`'s (0), through the
/// wrappers whose weights hand `count` to the one they wrap
/// (`ConstantScoreQuery`, a `BoostQuery`) -- and a boolean of one required
/// clause, the form a lone clause arrives in here (Lucene rewrites it away).
fn leaf_count(seg: &OpenSegment<'_>, query: &BooleanQuery) -> Result<LeafCount> {
    use crate::query::Clause;
    fn clause(seg: &OpenSegment<'_>, c: &Clause) -> Result<LeafCount> {
        Ok(match c {
            Clause::Term(t) => {
                match crate::weight_count::count_term_query_shortcut(seg.fields, seg.live_docs, t)?
                {
                    Some(n) => LeafCount::Whole(u64::try_from(n).unwrap_or(0)),
                    None => LeafCount::Iterate,
                }
            }
            Clause::MatchAllDocs(q) => {
                let max_doc = seg.max_doc.unwrap_or(q.max_doc);
                let n = crate::weight_count::count_match_all_docs(max_doc, seg.live_docs);
                LeafCount::Whole(u64::try_from(n).unwrap_or(0))
            }
            Clause::MatchNoDocs(_) => LeafCount::Whole(0),
            Clause::ConstantScore(c) => clause(seg, &c.inner)?,
            Clause::Boost(b) => clause(seg, &b.inner)?,
            Clause::Boolean(b) => leaf_count(seg, b)?,
            _ => LeafCount::Unknown,
        })
    }
    let lone = match (&query.must[..], &query.filter[..]) {
        ([only], []) | ([], [only]) => Some(only),
        _ => None,
    };
    match lone {
        Some(only)
            if query.should.is_empty()
                && query.must_not.is_empty()
                && query.minimum_should_match == 0 =>
        {
            clause(seg, only)
        }
        _ => Ok(LeafCount::Unknown),
    }
}

/// `seg`'s live documents up to and including `last`, as a bit set of the
/// segment's size: its live-docs words copied (or all set) and every bit past
/// `last` cleared -- a word at a time, not a document at a time.
fn prefix_live(seg: &OpenSegment<'_>, last: i32) -> FixedBitSet {
    let n = usize::try_from(seg.max_doc.unwrap_or(last.saturating_add(1))).unwrap_or(0);
    let words_len = lucene_util::fixed_bit_set::bits2words(n);
    let mut words = match seg.live_docs {
        Some(live) => {
            let mut w = live.words().to_vec();
            w.resize(words_len, 0);
            w
        }
        None => vec![u64::MAX; words_len],
    };
    // Bits [0, keep) stay; keep <= n, so no bit past n survives either.
    let keep = usize::try_from(last)
        .map_or(0, |l| l.saturating_add(1))
        .min(n);
    let (full, rest) = (keep / 64, keep % 64);
    if let Some(w) = words.get_mut(full) {
        // ARITH: rest < 64.
        *w &= if rest == 0 { 0 } else { (1u64 << rest) - 1 };
    }
    for w in words.iter_mut().skip(full.saturating_add(1)) {
        *w = 0;
    }
    FixedBitSet::from_words(words, n)
}

/// [`crate::top_field::search_sorted_tracking`] behind `terminate_after(n)`: the top
/// `top_n` of the first `n` matches, as OpenSearch's collector chain keeps
/// them, and the [`Cut`] (its `collected` is the total: the top-docs
/// collector counts exactly the documents let through).
///
/// # Errors
/// What [`crate::top_field::search_sorted_tracking`] reports.
#[allow(clippy::too_many_arguments)]
pub fn search_sorted_until(
    segments: &[OpenSegment<'_>],
    readers: &[SegmentReader],
    query: &BooleanQuery,
    norms: &[Option<&HashMap<String, FieldNorms<'_>>>],
    sort: &[SortField],
    top_n: usize,
    after: Option<&FieldDoc>,
    track_max_score: bool,
    n: u64,
) -> Result<(TopFieldDocs, Cut)> {
    let cut = terminate_after(segments, query, n)?;
    // The cut segment's documents up to its last collected match, live.
    let masked = cut
        .last
        .map(|(i, last)| (i, prefix_live(&segments[i], last)));
    let view: Vec<OpenSegment<'_>> = segments
        .iter()
        .enumerate()
        .map(|(i, s)| OpenSegment {
            fields: s.fields,
            doc_in: s.doc_in,
            pos_in: s.pos_in,
            pay_in: s.pay_in,
            live_docs: match &masked {
                Some((m, bits)) if *m == i => Some(bits),
                _ => s.live_docs,
            },
            doc_base: s.doc_base,
            max_doc: s.max_doc,
            cache: s.cache,
            points: s.points,
        })
        .collect();
    let searched: Vec<usize> = (0..cut.leaves).collect();
    let top = crate::top_field::search_sorted_leaves(
        &view,
        readers,
        query,
        norms,
        sort,
        top_n,
        // Every let-through document counts: no pruning, an exact total.
        u64::MAX,
        after,
        track_max_score,
        Some(&searched),
    )?;
    Ok((top, cut))
}

/// Whether a concurrent `size: 0` search's hit count stops early: OpenSearch
/// counts each slice through an `EarlyTerminatingCollector(TotalHitCountCollector,
/// n)` (not forced; `EmptyTopDocsCollectorContext.createManager`), whose
/// reduce reports `terminated_early` when any slice's collector stopped.
///
/// A slice's collector is asked for each of its segments in doc-base order.
/// Once it holds `n` documents it stops at the next segment; otherwise the
/// count collector either answers the segment from `Weight.count` -- nothing
/// reaches the terminating collector -- or iterates its live matches, and the
/// `n + 1`th stops it. `iterate[i]` says which: `true` where `Weight.count`
/// gave `-1` (the caller asks Lucene's own weight, query cache included).
///
/// # Errors
/// A postings or points read failure, or a slice naming a segment not in
/// `segments`.
pub fn count_terminates(
    segments: &[OpenSegment<'_>],
    query: &BooleanQuery,
    slices: &[Vec<usize>],
    iterate: &[bool],
    n: u64,
) -> Result<bool> {
    let rewritten = crate::multi_segment::rewrite_points_ranges(query, segments);
    let query = rewritten.as_ref().unwrap_or(query);
    for slice in slices {
        let mut leaves = slice.clone();
        leaves.sort_unstable_by_key(|&i| segments.get(i).map_or(i32::MAX, |s| s.doc_base));
        let mut collected = 0u64;
        for i in leaves {
            let Some(seg) = segments.get(i) else {
                return Err(crate::Error::SliceOutOfRange {
                    segment: i,
                    segments: segments.len(),
                });
            };
            if collected >= n {
                return Ok(true);
            }
            if !iterate.get(i).copied().unwrap_or(true) {
                continue;
            }
            let ctx = exec::LeafContext {
                fields: seg.fields,
                doc_in: seg.doc_in,
                pos_in: seg.pos_in,
                pay_in: seg.pay_in,
                live_docs: seg.live_docs,
                points: seg.points,
                norms: None,
                global: None,
                max_doc: seg.max_doc,
                cache: seg.cache,
            };
            let Some(mut scorer) =
                exec::build::build_boolean(&ctx, query, 1.0, Mode::NoScores, true)?
            else {
                continue;
            };
            let mut doc = exec::exact_next(scorer.as_mut())?;
            while doc != NO_MORE_DOCS {
                if seg.live_docs.is_none_or(|l| l.get_doc(doc)) {
                    if collected == n {
                        return Ok(true);
                    }
                    // ARITH: collected < n.
                    collected += 1;
                }
                doc = exec::exact_next(scorer.as_mut())?;
            }
        }
    }
    Ok(false)
}
