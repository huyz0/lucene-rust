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
//! Counting is by global ordinal ([`GlobalOrds`]), as OpenSearch counts: a
//! global ordinal is a term, and the ordinals are in term order.

use lucene_codecs::doc_values::{NumericReader, SortedNumericReader, SortedSetKind};
use lucene_codecs::terms_dict::{TermsCursor, TermsDict, TermsDictEntry};

use crate::ordinal_map::{OrdinalMap, TermCursor};

use crate::aggs::ColumnRead;
use crate::directory_reader::SegmentReader;
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
    Multi(Box<SortedNumericReader<'a>>),
}

/// `field`'s keyword doc values in `reader`: its ordinal column and terms
/// dictionary, or `None` when the segment has no doc values for it.
fn keyword_column<'a>(
    reader: &'a SegmentReader,
    field: &str,
) -> Result<Option<(Ords<'a>, &'a [u8], &'a TermsDictEntry)>> {
    let Some(info) = reader.field_infos().fields.iter().find(|i| i.name == field) else {
        return Ok(None);
    };
    let Some((meta, data)) = reader.doc_values_for_field(info.number) else {
        return Ok(None);
    };
    if let Some(e) = meta.sorted_set_entry(info.number) {
        return Ok(Some(match &e.kind {
            SortedSetKind::Single(se) => (
                Ords::Single(Box::new(NumericReader::new(data, &se.ords))),
                data,
                &se.terms,
            ),
            SortedSetKind::Multi { ords, terms } => (
                Ords::Multi(Box::new(SortedNumericReader::new(data, ords))),
                data,
                terms,
            ),
        }));
    }
    if let Some(se) = meta.sorted_entry(info.number) {
        return Ok(Some((
            Ords::Single(Box::new(NumericReader::new(data, &se.ords))),
            data,
            &se.terms,
        )));
    }
    // Doc values of another kind are a mapping error; none at all (a field
    // indexed without doc values here) counts nothing.
    if meta.numeric_entry(info.number).is_some()
        || meta.sorted_numeric_entry(info.number).is_some()
        || meta.binary_entry(info.number).is_some()
    {
        return Err(crate::Error::TermsAggType(field.to_string()));
    }
    Ok(None)
}

/// The dictionary half of [`keyword_column`].
fn terms_entry<'a>(
    reader: &'a SegmentReader,
    field: &str,
) -> Result<Option<(&'a [u8], &'a TermsDictEntry)>> {
    Ok(keyword_column(reader, field)?.map(|(_, data, entry)| (data, entry)))
}

/// [`keyword_column`] with its dictionary opened.
fn open_ords<'a>(
    reader: &'a SegmentReader,
    field: &str,
) -> Result<(Ords<'a>, Option<TermsDict<'a>>)> {
    match keyword_column(reader, field)? {
        Some((ords, data, entry)) => {
            Ok((ords, Some(TermsDict::open(data, entry).map_err(store_err)?)))
        }
        None => Ok((Ords::Absent, None)),
    }
}

/// A keyword field's global ordinals over a reader's segments: Lucene's
/// `OrdinalMap` ([`OrdinalMap`], this port's one), which OpenSearch's
/// `GlobalOrdinalsStringTermsAggregator` counts into -- every distinct term of
/// the field across the segments has an ordinal in term order, and each
/// segment's ordinals map onto them. Built once per reader and field (see
/// [`crate::directory_reader::DirectoryReader::global_ords`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalOrds {
    map: OrdinalMap,
}

impl GlobalOrds {
    /// Merges the segments' dictionaries of `field`, streamed
    /// ([`OrdinalMap::build_streaming`]): nothing is sized by a count read
    /// off disk.
    ///
    /// # Errors
    /// A field whose doc values are not keyword ones, or a dictionary that
    /// cannot be read.
    pub fn build(readers: &[SegmentReader], field: &str) -> Result<Self> {
        let mut cursors = Vec::with_capacity(readers.len());
        for reader in readers {
            cursors.push(match terms_entry(reader, field)? {
                Some((data, entry)) => Some(TermsCursor::open(data, entry).map_err(store_err)?),
                None => None,
            });
        }
        let mut empty: Vec<NoTerms> = std::iter::repeat_with(|| NoTerms)
            .take(readers.len())
            .collect();
        let mut refs: Vec<&mut dyn TermCursor> = cursors
            .iter_mut()
            .zip(&mut empty)
            .map(|(c, e)| match c {
                Some(c) => c as &mut dyn TermCursor,
                None => e as &mut dyn TermCursor,
            })
            .collect();
        Ok(GlobalOrds {
            map: OrdinalMap::build_streaming(&mut refs).map_err(store_err)?,
        })
    }

    /// The number of distinct terms (`OrdinalMap.getValueCount`).
    pub fn value_count(&self) -> usize {
        usize::try_from(self.map.value_count()).unwrap_or(0)
    }
}

/// A segment without the field: no terms.
struct NoTerms;

impl TermCursor for NoTerms {
    fn next_term(&mut self) -> lucene_store::Result<Option<&[u8]>> {
        Ok(None)
    }
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
    let spec = [TermsSpec {
        field: field.to_string(),
        shard_size,
    }];
    let global = [std::sync::Arc::new(GlobalOrds::build(readers, field)?)];
    let sliced =
        crate::aggs::aggregate_sliced(segments, readers, query, &[], &spec, &global, slices)?;
    Ok(sliced
        .into_iter()
        .filter_map(|(_, mut terms)| terms.pop())
        .collect())
}

/// One `terms` aggregation: its field and `shard_size`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TermsSpec {
    pub field: String,
    pub shard_size: usize,
}

/// `search.aggregations.terms.max_precompute_cardinality`'s default: the most
/// terms a segment may have for its counts to be read from its postings.
const MAX_PRECOMPUTE_CARDINALITY: usize = 30_000;

/// Per-segment scratch for [`segment_counts`], reused across segments.
#[derive(Default)]
pub(crate) struct TermsScratch {
    ords: Vec<i64>,
}

/// Adds segment `seg`'s matches to `counts`, indexed by global ordinal: `read`
/// says how the matches meet the column (see [`crate::aggs::column_read`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn segment_counts(
    reader: &SegmentReader,
    postings: &lucene_codecs::blocktree::BlockTreeFields,
    seg: usize,
    field: &str,
    global: &GlobalOrds,
    read: &ColumnRead<'_>,
    counts: &mut [u64],
    scratch: &mut TermsScratch,
) -> Result<()> {
    let map = global.map.segment_ords(seg).unwrap_or(&[]);
    // `tryCollectFromTermFrequencies`: a segment every document of which
    // matches (a match-all, nothing deleted) is counted from its postings --
    // each term's `docFreq`, the i-th term being ordinal i -- when the field
    // has postings and at most `MAX_PRECOMPUTE_CARDINALITY` terms.
    if matches!(read, ColumnRead::Stream(crate::aggs::Accept::Live(None))) {
        if let Some(terms) = postings.field(field) {
            let n = usize::try_from(terms.num_terms).unwrap_or(usize::MAX);
            if n <= MAX_PRECOMPUTE_CARDINALITY && n == map.len() {
                let mut e = terms.iter();
                let mut ord = 0usize;
                while let Some((_, stats)) = e.next() {
                    let slot = map
                        .get(ord)
                        .and_then(|&g| usize::try_from(g).ok())
                        .and_then(|g| counts.get_mut(g));
                    if let Some(c) = slot {
                        *c += u64::try_from(stats.doc_freq).unwrap_or(0);
                    }
                    ord += 1;
                }
                return Ok(());
            }
        }
    }
    let (ords, _) = open_ords(reader, field)?;
    // An ordinal past the dictionary is corrupt, and reported.
    let mut bad_ord: Option<i64> = None;
    let mut bump = |ord: i64| match usize::try_from(ord)
        .ok()
        .and_then(|o| map.get(o))
        .and_then(|&g| counts.get_mut(g as usize))
    {
        Some(c) => *c += 1,
        None => bad_ord = Some(ord),
    };
    let max_doc = reader.max_doc;
    match (ords, read) {
        (Ords::Absent, _) => {}
        (Ords::Single(mut r), ColumnRead::Stream(accept)) => {
            r.for_each_value(0, max_doc, |doc, ord| {
                if accept.test(doc) {
                    bump(ord);
                }
            })?
        }
        (Ords::Single(mut r), ColumnRead::Seek(docs)) => {
            for &doc in *docs {
                if let Some(ord) = r.value(doc)? {
                    bump(ord);
                }
            }
        }
        (Ords::Multi(mut r), ColumnRead::Stream(accept)) => {
            r.for_each_doc(0, max_doc, |doc, ords| {
                if accept.test(doc) {
                    for &ord in ords {
                        bump(ord);
                    }
                }
            })?
        }
        (Ords::Multi(mut r), ColumnRead::Seek(docs)) => {
            for &doc in *docs {
                r.values(doc, &mut scratch.ords)?;
                for &ord in scratch.ords.iter() {
                    bump(ord);
                }
            }
        }
    }
    match bad_ord {
        Some(ord) => Err(store_err(lucene_store::Error::Corrupted(format!(
            "{field}: ordinal {ord} outside the segment's dictionary"
        )))),
        None => Ok(()),
    }
}

fn store_err(e: lucene_store::Error) -> crate::Error {
    crate::Error::from(lucene_codecs::doc_values::Error::from(e))
}

/// `buildAggregations`: the top `shard_size` by count desc, then global
/// ordinal (term) asc; the others' counts in `other_doc_count`; the kept
/// listed by term, their bytes read from a segment holding each.
pub(crate) fn select(
    counts: &[u64],
    shard_size: usize,
    global: &GlobalOrds,
    readers: &[SegmentReader],
    field: &str,
) -> Result<TermsResult> {
    let total: u64 = counts.iter().sum();
    let mut kept: Vec<(u32, u64)> = counts
        .iter()
        .enumerate()
        .filter(|&(_, &n)| n > 0)
        .map(|(g, &n)| (g as u32, n))
        .collect();
    let by_rank = |a: &(u32, u64), b: &(u32, u64)| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0));
    if kept.len() > shard_size {
        if shard_size > 0 {
            kept.select_nth_unstable_by(shard_size - 1, by_rank);
        }
        kept.truncate(shard_size);
    }
    let kept_docs: u64 = kept.iter().map(|b| b.1).sum();
    kept.sort_unstable_by_key(|b| b.0);
    let mut dicts: Vec<Option<TermsDict<'_>>> = Vec::new();
    let mut buckets = Vec::with_capacity(kept.len());
    for (g, n) in kept {
        let (Some(seg), Some(ord)) = (
            global.map.first_segment(i64::from(g)),
            global.map.first_segment_ord(i64::from(g)),
        ) else {
            continue;
        };
        if dicts.len() <= seg {
            dicts.resize_with(seg + 1, || None);
        }
        if dicts[seg].is_none() {
            if let Some(reader) = readers.get(seg) {
                dicts[seg] = open_ords(reader, field)?.1;
            }
        }
        let Some(dict) = dicts[seg].as_mut() else {
            continue;
        };
        buckets.push((dict.seek_ord(ord).map_err(store_err)?.to_vec(), n));
    }
    Ok(TermsResult {
        buckets,
        other_doc_count: total - kept_docs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_top_terms_are_kept_by_count_then_term_and_listed_by_term() {
        // No segments to read terms from: the buckets are empty, but the
        // counts kept and left over are what selection decided.
        let global = GlobalOrds::build(&[], "f").unwrap();
        let r = select(&[3, 3, 5, 1, 3], 3, &global, &[], "f").unwrap();
        assert_eq!(
            r.other_doc_count, 4,
            "ordinals 0, 1, 2 kept (5 and the two lowest 3s)"
        );
        let r = select(&[2, 1], 10, &global, &[], "f").unwrap();
        assert_eq!(r.other_doc_count, 0);
        assert_eq!(
            select(&[], 0, &global, &[], "f").unwrap(),
            TermsResult::default()
        );
        assert_eq!(global.value_count(), 0);
    }
}
