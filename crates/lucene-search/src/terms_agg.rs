//! OpenSearch's `terms` aggregation on a keyword field, as a shard computes
//! it before the coordinator reduces it (read path R5).
//!
//! `GlobalOrdinalsStringTermsAggregator` (and each of its collection
//! strategies, and `MapStringTermsAggregator`) counts, per term, the live
//! matching documents holding it -- a multi-valued document once per distinct
//! term -- and then keeps the top `shard_size` terms by count, highest first,
//! ties by term (unsigned bytes, ascending): `InternalOrder.compound(count
//! desc)` with the `key asc` tie-break `TermsAggregationBuilder` appends. The
//! rest go to `otherDocCount`, the sum of the counts not kept. The kept
//! buckets are listed by term, ascending (`reduceOrder` is `KEY_ASC`).
//!
//! Only terms with a count of at least one are candidates: with a
//! `min_doc_count` of 1 or more (the default), `forEach` never offers the
//! zero-count ordinals, and a `shard_min_doc_count` of 0 keeps every offered
//! one. Requests outside that -- and other orders -- are the caller's to
//! route elsewhere.
//!
//! Under concurrent segment search every slice counts on its own and keeps
//! its own top `shard_size` ([`terms_sliced`]); the shard then reduces the
//! slices with `InternalTerms.reduce`, which is the caller's (OpenSearch's
//! own reduce, in the plugin).
//!
//! Counting by term bytes rather than by global ordinal gives the same counts
//! (a global ordinal is a term); the order of the terms is the ordinals'.

use std::collections::HashMap;

use lucene_codecs::doc_values::{NumericReader, SortedNumericReader, SortedSetKind};
use lucene_codecs::terms_dict::TermsDict;

use crate::aggs::{lone_clause, segment_matches};
use crate::directory_reader::SegmentReader;
use crate::exec;
use crate::multi_segment::OpenSegment;
use crate::query::BooleanQuery;
use crate::Result;

/// One slice's shard result: the kept buckets, by term ascending, and the
/// documents counted in terms not kept.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TermsResult {
    pub buckets: Vec<(Vec<u8>, u64)>,
    pub other_doc_count: u64,
}

/// A segment's ordinal column for the field.
enum Ords<'a> {
    Absent,
    Single(Box<NumericReader<'a>>),
    Multi(SortedNumericReader<'a>),
}

fn open_ords<'a>(
    reader: &'a SegmentReader,
    field: &str,
) -> Result<(Ords<'a>, Option<TermsDict<'a>>)> {
    let store =
        |e: lucene_store::Error| crate::Error::from(lucene_codecs::doc_values::Error::from(e));
    let Some(info) = reader.field_infos().fields.iter().find(|i| i.name == field) else {
        return Ok((Ords::Absent, None));
    };
    let Some((meta, data)) = reader.doc_values_for_field(info.number) else {
        return Ok((Ords::Absent, None));
    };
    if let Some(e) = meta.sorted_set_entry(info.number) {
        return Ok(match &e.kind {
            SortedSetKind::Single(se) => (
                Ords::Single(Box::new(NumericReader::new(data, &se.ords))),
                Some(TermsDict::open(data, &se.terms).map_err(store)?),
            ),
            SortedSetKind::Multi { ords, terms } => (
                Ords::Multi(SortedNumericReader::new(data, ords)),
                Some(TermsDict::open(data, terms).map_err(store)?),
            ),
        });
    }
    if let Some(se) = meta.sorted_entry(info.number) {
        return Ok((
            Ords::Single(Box::new(NumericReader::new(data, &se.ords))),
            Some(TermsDict::open(data, &se.terms).map_err(store)?),
        ));
    }
    Err(crate::Error::TermsAggType(field.to_string()))
}

/// The `terms` shard result of `field` over `query`'s live matches, the
/// segments searched as one: [`terms_sliced`] with a single slice of every
/// segment.
///
/// # Errors
/// A field whose doc values are not keyword ones (`SORTED`/`SORTED_SET`), or
/// what reading the index reports.
pub fn terms(
    segments: &[OpenSegment<'_>],
    readers: &[SegmentReader],
    query: &BooleanQuery,
    field: &str,
    shard_size: usize,
) -> Result<TermsResult> {
    let all: Vec<usize> = (0..segments.len().min(readers.len())).collect();
    let mut sliced = terms_sliced(segments, readers, query, field, shard_size, &[all])?;
    Ok(sliced.pop().unwrap_or_default())
}

/// [`terms`] per slice of a concurrent segment search, each from scratch;
/// slices run in parallel on rayon's pool.
///
/// # Errors
/// As [`terms`], and a slice naming a segment the reader does not have.
pub fn terms_sliced(
    segments: &[OpenSegment<'_>],
    readers: &[SegmentReader],
    query: &BooleanQuery,
    field: &str,
    shard_size: usize,
    slices: &[Vec<usize>],
) -> Result<Vec<TermsResult>> {
    let rewritten = crate::multi_segment::rewrite_points_ranges(query, segments);
    let query = rewritten.as_ref().unwrap_or(query);
    let clause = lone_clause(query);
    let one = |slice: &Vec<usize>| {
        let counts = slice_counts(segments, readers, &clause, field, slice)?;
        Ok(select(counts, shard_size))
    };
    if slices.len() > 1 {
        use rayon::prelude::*;
        return slices.par_iter().map(one).collect();
    }
    slices.iter().map(one).collect()
}

/// Every term's count over one slice's segments.
fn slice_counts(
    segments: &[OpenSegment<'_>],
    readers: &[SegmentReader],
    clause: &crate::query::Clause,
    field: &str,
    slice: &[usize],
) -> Result<HashMap<Vec<u8>, u64>> {
    let mut counts: HashMap<Vec<u8>, u64> = HashMap::new();
    let mut docs_buf = Vec::new();
    let mut ords_buf = Vec::new();
    let mut per_ord: Vec<u64> = Vec::new();
    for &i in slice {
        let (Some(seg), Some(reader)) = (segments.get(i), readers.get(i)) else {
            return Err(crate::Error::SliceOutOfRange {
                segment: i,
                segments: segments.len().min(readers.len()),
            });
        };
        let (ords, dict) = open_ords(reader, field)?;
        let Some(mut dict) = dict else {
            continue;
        };
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
        let live = seg.live_docs;
        let Some(docs) = segment_matches(&ctx, clause, live, &mut docs_buf)? else {
            continue;
        };
        let is_live = |doc: i32| live.is_none_or(|l| l.get_doc(doc));
        per_ord.clear();
        per_ord.resize(usize::try_from(dict.size()).unwrap_or(0), 0);
        // An ordinal past the dictionary is corrupt; it is reported by the
        // term lookup below rather than counted.
        let mut bad_ord: Option<i64> = None;
        let mut bump = |ord: i64| match usize::try_from(ord).ok().and_then(|o| per_ord.get_mut(o)) {
            Some(c) => *c += 1,
            None => bad_ord = Some(ord),
        };
        match (ords, docs) {
            (Ords::Absent, _) => {}
            (Ords::Single(mut r), None) => r.for_each_value(0, reader.max_doc, |doc, ord| {
                if is_live(doc) {
                    bump(ord);
                }
            })?,
            (Ords::Single(mut r), Some(docs)) => {
                for &doc in docs {
                    if let Some(ord) = r.value(doc)? {
                        bump(ord);
                    }
                }
            }
            (Ords::Multi(mut r), None) => r.for_each_doc(0, reader.max_doc, |doc, ords| {
                if is_live(doc) {
                    for &ord in ords {
                        bump(ord);
                    }
                }
            })?,
            (Ords::Multi(mut r), Some(docs)) => {
                for &doc in docs {
                    r.values(doc, &mut ords_buf)?;
                    for &ord in &ords_buf {
                        bump(ord);
                    }
                }
            }
        }
        if let Some(ord) = bad_ord {
            // Raises the dictionary's own out-of-range error.
            dict.seek_ord(ord).map_err(store_err)?;
        }
        for (ord, &n) in per_ord.iter().enumerate() {
            if n > 0 {
                let term = dict.seek_ord(ord as i64).map_err(store_err)?;
                *counts.entry(term.to_vec()).or_insert(0) += n;
            }
        }
    }
    Ok(counts)
}

fn store_err(e: lucene_store::Error) -> crate::Error {
    crate::Error::from(lucene_codecs::doc_values::Error::from(e))
}

/// `buildAggregations`: the top `shard_size` by count desc, term asc; the
/// others' counts in `other_doc_count`; the kept listed by term.
fn select(counts: HashMap<Vec<u8>, u64>, shard_size: usize) -> TermsResult {
    let total: u64 = counts.values().sum();
    let mut all: Vec<(Vec<u8>, u64)> = counts.into_iter().collect();
    let by_rank =
        |a: &(Vec<u8>, u64), b: &(Vec<u8>, u64)| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0));
    if all.len() > shard_size {
        if shard_size > 0 {
            all.select_nth_unstable_by(shard_size - 1, by_rank);
        }
        all.truncate(shard_size);
    }
    let kept: u64 = all.iter().map(|b| b.1).sum();
    all.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    TermsResult {
        buckets: all,
        other_doc_count: total - kept,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(pairs: &[(&str, u64)]) -> HashMap<Vec<u8>, u64> {
        pairs
            .iter()
            .map(|&(t, n)| (t.as_bytes().to_vec(), n))
            .collect()
    }

    #[test]
    fn the_top_terms_are_kept_by_count_then_term_and_listed_by_term() {
        let r = select(
            counts(&[("b", 3), ("a", 3), ("c", 5), ("d", 1), ("e", 3)]),
            3,
        );
        assert_eq!(
            r.buckets,
            vec![(b"a".to_vec(), 3), (b"b".to_vec(), 3), (b"c".to_vec(), 5)]
        );
        assert_eq!(r.other_doc_count, 4);
        // Room for every term: nothing left over.
        let r = select(counts(&[("x", 2), ("y", 1)]), 10);
        assert_eq!(r.buckets.len(), 2);
        assert_eq!(r.other_doc_count, 0);
        // Unsigned byte order: 0xff after 'z'.
        let mut c = counts(&[("z", 1)]);
        c.insert(vec![0xff], 1);
        let r = select(c, 1);
        assert_eq!(r.buckets, vec![(b"z".to_vec(), 1)]);
        assert_eq!(select(HashMap::new(), 0), TermsResult::default());
    }
}
