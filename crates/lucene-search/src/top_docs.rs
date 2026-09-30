//! `org.apache.lucene.search.TopDocs`'s static half: [`merge`] (by score) and
//! [`merge_field_docs`] (by a sort), each with a start offset, shard indices
//! and a pluggable tie-breaker, and [`rrf`] (reciprocal rank fusion).
//!
//! A port of Lucene 10.5.0's `TopDocs.mergeAux` with its `ScoreMergeSortQueue`
//! and `MergeSortQueue`, driven through the same binary heap as
//! `org.apache.lucene.util.PriorityQueue` ([`MergeQueue`]): when a custom
//! tie-breaker calls two hits of different shards equal, Java falls back to
//! their positions within their shards (`tieBreakLessThan`), which is not a
//! total order across shards -- so which of them comes first depends on the
//! heap's shape, and only the same heap reproduces it.
//!
//! The in-crate multi-segment merges ([`crate::multi_segment`],
//! `top_field::merge_top_docs`) merge the leaves of one index and need none of
//! this; these functions are the shard-level API a caller federating several
//! searches (OpenSearch's coordinating node, a hybrid query's fusion) uses.
//!
//! Verified against Lucene by `tests/top_docs_fixtures.rs`
//! (`fixtures/src/GenTopDocs.java`).

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::collector::{TotalHits, TotalHitsRelation};
use crate::top_field::{compare_keys, FieldDoc, SortField};
use crate::{Error, Result};

/// `ScoreDoc` with its `shardIndex`: `-1` when every hit came from one
/// searcher (Java's default).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShardScoreDoc {
    pub doc: i32,
    pub score: f32,
    pub shard_index: i32,
}

impl ShardScoreDoc {
    /// `new ScoreDoc(doc, score)`: shard index `-1`.
    pub fn new(doc: i32, score: f32) -> Self {
        Self {
            doc,
            score,
            shard_index: -1,
        }
    }

    /// `new ScoreDoc(doc, score, shardIndex)`.
    pub fn with_shard(doc: i32, score: f32, shard_index: i32) -> Self {
        Self {
            doc,
            score,
            shard_index,
        }
    }
}

/// `TopDocs`: the total and the hits, best first.
#[derive(Debug, Clone, PartialEq)]
pub struct TopDocs {
    pub total_hits: TotalHits,
    pub score_docs: Vec<ShardScoreDoc>,
}

/// `FieldDoc` with the score and shard index of its `ScoreDoc` half.
#[derive(Debug, Clone, PartialEq)]
pub struct ShardFieldDoc {
    /// The document and its sort values, in [`crate::top_field`]'s encoding.
    pub fields: FieldDoc,
    pub score: f32,
    pub shard_index: i32,
}

impl ShardFieldDoc {
    fn score_doc(&self) -> ShardScoreDoc {
        ShardScoreDoc {
            doc: self.fields.doc,
            score: self.score,
            shard_index: self.shard_index,
        }
    }
}

/// `TopFieldDocs`: the total and the hits in the sort's order.
#[derive(Debug, Clone, PartialEq)]
pub struct ShardTopFieldDocs {
    pub total_hits: TotalHits,
    pub hits: Vec<ShardFieldDoc>,
}

/// `Comparator<ScoreDoc>`: how two hits the merge's keys call equal are
/// ordered.
pub enum TieBreaker<'a> {
    /// `DEFAULT_TIE_BREAKER`: the shard index, then the document.
    Default,
    /// `Comparator.comparingInt(d -> d.doc)`.
    DocId,
    /// `Comparator.comparingInt(d -> d.shardIndex)`.
    ShardIndex,
    /// Any other comparator.
    Custom(&'a dyn Fn(&ShardScoreDoc, &ShardScoreDoc) -> Ordering),
}

impl TieBreaker<'_> {
    fn compare(&self, a: &ShardScoreDoc, b: &ShardScoreDoc) -> Ordering {
        match self {
            TieBreaker::Default => a.shard_index.cmp(&b.shard_index).then(a.doc.cmp(&b.doc)),
            TieBreaker::DocId => a.doc.cmp(&b.doc),
            TieBreaker::ShardIndex => a.shard_index.cmp(&b.shard_index),
            TieBreaker::Custom(f) => f(a, b),
        }
    }
}

/// `TopDocs.ShardRef`: which shard, and which hit within it.
#[derive(Debug, Clone, Copy)]
struct ShardRef {
    shard_index: usize,
    hit_index: usize,
}

/// `org.apache.lucene.util.PriorityQueue`'s heap, 1-based, with `lessThan`
/// supplied per call: `add`/`top`/`updateTop`/`pop` move elements exactly as
/// Java's do, which [`merge`]'s tie-breaking depends on (see the module doc).
struct MergeQueue {
    heap: Vec<ShardRef>,
}

impl MergeQueue {
    fn new(capacity: usize) -> Self {
        let mut heap = Vec::with_capacity(capacity.saturating_add(1));
        // Slot 0 is unused, as Java's.
        heap.push(ShardRef {
            shard_index: 0,
            hit_index: 0,
        });
        Self { heap }
    }

    fn size(&self) -> usize {
        self.heap.len() - 1
    }

    fn add(&mut self, e: ShardRef, less: &impl Fn(&ShardRef, &ShardRef) -> bool) {
        self.heap.push(e);
        let at = self.size();
        self.up_heap(at, less);
    }

    fn top_mut(&mut self) -> &mut ShardRef {
        &mut self.heap[1]
    }

    fn update_top(&mut self, less: &impl Fn(&ShardRef, &ShardRef) -> bool) {
        self.down_heap(1, less);
    }

    fn pop(&mut self, less: &impl Fn(&ShardRef, &ShardRef) -> bool) {
        let size = self.size();
        if size == 0 {
            return;
        }
        self.heap.swap(1, size);
        self.heap.pop();
        if self.size() > 0 {
            self.down_heap(1, less);
        }
    }

    /// `PriorityQueue.upHeap`.
    fn up_heap(&mut self, orig: usize, less: &impl Fn(&ShardRef, &ShardRef) -> bool) {
        let mut i = orig;
        let node = self.heap[i];
        let mut j = i >> 1;
        while j > 0 && less(&node, &self.heap[j]) {
            self.heap[i] = self.heap[j];
            i = j;
            j >>= 1;
        }
        self.heap[i] = node;
    }

    /// `PriorityQueue.downHeap`.
    fn down_heap(&mut self, mut i: usize, less: &impl Fn(&ShardRef, &ShardRef) -> bool) {
        let size = self.size();
        let node = self.heap[i];
        let mut j = i << 1;
        let mut k = j + 1;
        if k <= size && less(&self.heap[k], &self.heap[j]) {
            j = k;
        }
        while j <= size && less(&self.heap[j], &node) {
            self.heap[i] = self.heap[j];
            i = j;
            j = i << 1;
            k = j + 1;
            if k <= size && less(&self.heap[k], &self.heap[j]) {
                j = k;
            }
        }
        self.heap[i] = node;
    }
}

/// `TopDocs.tieBreakLessThan`: the tie-breaker, then (for two hits it calls
/// equal) their positions within their shards.
fn tie_break_less_than(
    first: &ShardRef,
    first_doc: &ShardScoreDoc,
    second: &ShardRef,
    second_doc: &ShardScoreDoc,
    tie_breaker: &TieBreaker<'_>,
) -> bool {
    match tie_breaker.compare(first_doc, second_doc) {
        Ordering::Equal => first.hit_index < second.hit_index,
        o => o == Ordering::Less,
    }
}

/// The part of `mergeAux` shared by both queues: totals, the heap walk from
/// `start` to `start + size`, and the shard-index consistency check. Returns
/// the chosen `(shard, hit)` positions.
fn merge_aux(
    start: usize,
    size: usize,
    shard_totals: &[TotalHits],
    shard_lens: &[usize],
    shard_index_of: impl Fn(usize, usize) -> i32,
    less: impl Fn(&ShardRef, &ShardRef) -> bool,
) -> Result<(TotalHits, Vec<(usize, usize)>)> {
    let mut queue = MergeQueue::new(shard_lens.len());
    let mut total_value: u64 = 0;
    let mut relation = TotalHitsRelation::EqualTo;
    let mut avail: usize = 0;
    for (shard, (total, &len)) in shard_totals.iter().zip(shard_lens).enumerate() {
        total_value = total_value.saturating_add(total.value);
        if total.relation == TotalHitsRelation::GreaterThanOrEqualTo {
            relation = TotalHitsRelation::GreaterThanOrEqualTo;
        }
        if len > 0 {
            avail = avail.saturating_add(len);
            queue.add(
                ShardRef {
                    shard_index: shard,
                    hit_index: 0,
                },
                &less,
            );
        }
    }
    let mut hits = Vec::new();
    if avail > start {
        let window = start.saturating_add(size);
        let iterations = avail.min(window);
        let mut unset_shard_index = false;
        for hit_upto in 0..iterations {
            let top = queue.top_mut();
            let (shard, hit) = (top.shard_index, top.hit_index);
            top.hit_index += 1;
            let shard_index = shard_index_of(shard, hit);
            if hit_upto > 0 && unset_shard_index != (shard_index == -1) {
                return Err(Error::IllegalArgument(
                    "Inconsistent order of shard indices".to_string(),
                ));
            }
            unset_shard_index |= shard_index == -1;
            if hit_upto >= start {
                hits.push((shard, hit));
            }
            if hit + 1 < shard_lens[shard] {
                queue.update_top(&less);
            } else {
                queue.pop(&less);
            }
        }
    }
    Ok((
        TotalHits {
            value: total_value,
            relation,
        },
        hits,
    ))
}

/// `TopDocs.merge(start, topN, shardHits, tieBreaker)`: the hits ranked
/// `start..start + top_n` across `shard_hits`, each already sorted by score
/// descending, by score and then `tie_breaker`; the totals summed, a lower
/// bound if any shard's is.
///
/// # Errors
/// [`Error::IllegalArgument`] when some hits have a shard index and others
/// `-1` (Lucene's "Inconsistent order of shard indices").
pub fn merge(
    start: usize,
    top_n: usize,
    shard_hits: &[TopDocs],
    tie_breaker: &TieBreaker<'_>,
) -> Result<TopDocs> {
    let totals: Vec<TotalHits> = shard_hits.iter().map(|s| s.total_hits).collect();
    let lens: Vec<usize> = shard_hits.iter().map(|s| s.score_docs.len()).collect();
    let doc = |r: &ShardRef| &shard_hits[r.shard_index].score_docs[r.hit_index];
    // `ScoreMergeSortQueue.lessThan`: the higher score first.
    let less = |a: &ShardRef, b: &ShardRef| {
        let (x, y) = (doc(a), doc(b));
        if x.score < y.score {
            false
        } else if x.score > y.score {
            true
        } else {
            tie_break_less_than(a, x, b, y, tie_breaker)
        }
    };
    let (total_hits, picked) = merge_aux(
        start,
        top_n,
        &totals,
        &lens,
        |s, h| shard_hits[s].score_docs[h].shard_index,
        less,
    )?;
    Ok(TopDocs {
        total_hits,
        score_docs: picked
            .into_iter()
            .map(|(s, h)| shard_hits[s].score_docs[h])
            .collect(),
    })
}

/// `TopDocs.merge(sort, start, topN, shardHits, tieBreaker)`: as [`merge`],
/// ranked by `sort`'s keys (`FieldComparator.compareValues` times the
/// reverse multiplier), then `tie_breaker`.
///
/// # Errors
/// [`Error::IllegalArgument`] when `sort` is empty (Java's null sort), a hit
/// carries fewer sort values than `sort` has keys (a shard "not sorted by
/// the provided Sort"), or the shard indices are inconsistent.
pub fn merge_field_docs(
    sort: &[SortField],
    start: usize,
    top_n: usize,
    shard_hits: &[ShardTopFieldDocs],
    tie_breaker: &TieBreaker<'_>,
) -> Result<ShardTopFieldDocs> {
    if sort.is_empty() {
        return Err(Error::IllegalArgument(
            "sort must be non-null when merging field-docs".to_string(),
        ));
    }
    for (i, shard) in shard_hits.iter().enumerate() {
        if shard
            .hits
            .iter()
            .any(|h| h.fields.values.len() < sort.len())
        {
            return Err(Error::IllegalArgument(format!(
                "shard {i} did not set sort field values (FieldDoc.fields is null)"
            )));
        }
    }
    let totals: Vec<TotalHits> = shard_hits.iter().map(|s| s.total_hits).collect();
    let lens: Vec<usize> = shard_hits.iter().map(|s| s.hits.len()).collect();
    let hit = |r: &ShardRef| &shard_hits[r.shard_index].hits[r.hit_index];
    // `MergeSortQueue.lessThan`.
    let less = |a: &ShardRef, b: &ShardRef| {
        let (x, y) = (hit(a), hit(b));
        match compare_keys(sort, &x.fields, &y.fields) {
            Ordering::Equal => {
                tie_break_less_than(a, &x.score_doc(), b, &y.score_doc(), tie_breaker)
            }
            o => o == Ordering::Less,
        }
    };
    let (total_hits, picked) = merge_aux(
        start,
        top_n,
        &totals,
        &lens,
        |s, h| shard_hits[s].hits[h].shard_index,
        less,
    )?;
    Ok(ShardTopFieldDocs {
        total_hits,
        hits: picked
            .into_iter()
            .map(|(s, h)| shard_hits[s].hits[h].clone())
            .collect(),
    })
}

/// `TopDocs.rrf(topN, k, hits)`: reciprocal rank fusion. Each document scores
/// the sum, in `f64` and in the order the lists are given, of `1 / (k +
/// rank)` over the lists it appears in (rank from 1); the best `top_n` are
/// returned by that score descending, then document, then shard index, the
/// score narrowed to `f32`. The total is the largest of the lists' totals, as
/// a lower bound.
///
/// # Errors
/// [`Error::IllegalArgument`] when `top_n < 1`, `k < 1`, some hits have a
/// shard index and others `-1`, or `k + rank` overflows an `i32`
/// (`Math.addExact`).
pub fn rrf(top_n: i32, k: i32, hits: &[TopDocs]) -> Result<TopDocs> {
    if top_n < 1 {
        return Err(Error::IllegalArgument(format!(
            "topN must be >= 1, got {top_n}"
        )));
    }
    if k < 1 {
        return Err(Error::IllegalArgument(format!("k must be >= 1, got {k}")));
    }
    let mut shard_index_set: Option<bool> = None;
    for sd in hits.iter().flat_map(|t| &t.score_docs) {
        let this = sd.shard_index != -1;
        match shard_index_set {
            None => shard_index_set = Some(this),
            Some(s) if s != this => {
                return Err(Error::IllegalArgument(
                    "All hits must either have their ScoreDoc#shardIndex set, or unset (-1), \
                     not a mix of both."
                        .to_string(),
                ))
            }
            Some(_) => {}
        }
    }
    let mut scores: HashMap<(i32, i32), f64> = HashMap::new();
    let mut total: u64 = 0;
    for top in hits {
        total = total.max(top.total_hits.value);
        for (i, sd) in top.score_docs.iter().enumerate() {
            let rank = i32::try_from(i)
                .ok()
                .and_then(|i| i.checked_add(1))
                .and_then(|r| r.checked_add(k))
                .ok_or_else(|| Error::IllegalArgument("integer overflow".to_string()))?;
            let contribution = 1.0 / f64::from(rank);
            *scores.entry((sd.shard_index, sd.doc)).or_insert(0.0) += contribution;
        }
    }
    let mut ranked: Vec<((i32, i32), f64)> = scores.into_iter().collect();
    ranked.sort_by(|((sa, da), a), ((sb, db), b)| b.total_cmp(a).then(da.cmp(db)).then(sa.cmp(sb)));
    // `top_n >= 1` was checked above.
    let keep = usize::try_from(top_n).unwrap_or(usize::MAX);
    ranked.truncate(keep);
    Ok(TopDocs {
        total_hits: TotalHits {
            value: total,
            relation: TotalHitsRelation::GreaterThanOrEqualTo,
        },
        score_docs: ranked
            .into_iter()
            .map(|((shard_index, doc), score)| ShardScoreDoc {
                doc,
                score: score as f32,
                shard_index,
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eq(n: u64) -> TotalHits {
        TotalHits {
            value: n,
            relation: TotalHitsRelation::EqualTo,
        }
    }

    #[test]
    fn merge_of_nothing_is_empty() {
        let out = merge(0, 10, &[], &TieBreaker::Default).unwrap();
        assert!(out.score_docs.is_empty());
        assert_eq!(out.total_hits, eq(0));
    }

    #[test]
    fn start_past_the_hits_returns_none_but_keeps_the_total() {
        let a = TopDocs {
            total_hits: eq(7),
            score_docs: vec![ShardScoreDoc::new(1, 2.0), ShardScoreDoc::new(3, 1.0)],
        };
        let out = merge(5, 10, &[a], &TieBreaker::Default).unwrap();
        assert!(out.score_docs.is_empty());
        assert_eq!(out.total_hits.value, 7);
    }

    #[test]
    fn custom_tie_breaker_orders_equal_scores() {
        let a = TopDocs {
            total_hits: eq(1),
            score_docs: vec![ShardScoreDoc::with_shard(9, 1.0, 0)],
        };
        let b = TopDocs {
            total_hits: eq(1),
            score_docs: vec![ShardScoreDoc::with_shard(2, 1.0, 1)],
        };
        let by_doc_desc = |x: &ShardScoreDoc, y: &ShardScoreDoc| y.doc.cmp(&x.doc);
        let out = merge(
            0,
            2,
            &[a.clone(), b.clone()],
            &TieBreaker::Custom(&by_doc_desc),
        )
        .unwrap();
        assert_eq!(out.score_docs[0].doc, 9);
        let out = merge(0, 2, &[a, b], &TieBreaker::DocId).unwrap();
        assert_eq!(out.score_docs[0].doc, 2);
    }

    #[test]
    fn field_merge_rejects_an_empty_sort_and_missing_values() {
        let shard = ShardTopFieldDocs {
            total_hits: eq(1),
            hits: vec![ShardFieldDoc {
                fields: FieldDoc {
                    doc: 1,
                    values: vec![],
                    terms: vec![],
                },
                score: 1.0,
                shard_index: -1,
            }],
        };
        assert!(merge_field_docs(
            &[],
            0,
            1,
            std::slice::from_ref(&shard),
            &TieBreaker::Default
        )
        .is_err());
        let sort = [SortField::doc()];
        assert!(merge_field_docs(&sort, 0, 1, &[shard], &TieBreaker::Default).is_err());
    }

    #[test]
    fn rrf_rejects_bad_arguments() {
        assert!(rrf(0, 60, &[]).is_err());
        assert!(rrf(1, 0, &[]).is_err());
        let a = TopDocs {
            total_hits: eq(1),
            score_docs: vec![ShardScoreDoc::new(1, 1.0)],
        };
        assert!(rrf(1, i32::MAX, &[a]).is_err());
    }

    #[test]
    fn rrf_sums_ranks() {
        let a = TopDocs {
            total_hits: eq(2),
            score_docs: vec![ShardScoreDoc::new(1, 9.0), ShardScoreDoc::new(2, 8.0)],
        };
        let b = TopDocs {
            total_hits: eq(5),
            score_docs: vec![ShardScoreDoc::new(2, 1.0)],
        };
        let out = rrf(10, 1, &[a, b]).unwrap();
        assert_eq!(out.total_hits.value, 5);
        assert_eq!(out.score_docs[0].doc, 2);
        assert_eq!(out.score_docs[0].score, (1.0f64 / 3.0 + 0.5) as f32);
        assert_eq!(out.score_docs[1].doc, 1);
    }
}
