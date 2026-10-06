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
pub(crate) enum Ords<'a> {
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
pub(crate) fn open_ords<'a>(
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
        let readers: Vec<&SegmentReader> = readers.iter().collect();
        Ok(GlobalOrds {
            map: ordinal_map_of(&readers, field)?,
        })
    }

    /// Segment `seg`'s ordinals to global ones.
    pub(crate) fn segment_map(&self, seg: usize) -> Option<&[i64]> {
        self.map.segment_ords(seg)
    }

    /// The first segment holding global ordinal `g`.
    pub(crate) fn first_segment(&self, g: i64) -> Option<usize> {
        self.map.first_segment(g)
    }

    /// Global ordinal `g`'s ordinal in [`Self::first_segment`].
    pub(crate) fn first_segment_ord(&self, g: i64) -> Option<i64> {
        self.map.first_segment_ord(g)
    }

    /// The number of distinct terms (`OrdinalMap.getValueCount`).
    pub fn value_count(&self) -> usize {
        usize::try_from(self.map.value_count()).unwrap_or(0)
    }
}

/// `OrdinalMap.build(null, values, PackedInts.DEFAULT)` over `field`'s
/// `SORTED`/`SORTED_SET` dictionaries of `readers` (in order; a reader
/// without the field contributes no terms), streamed as
/// [`GlobalOrds::build`] streams them.
///
/// # Errors
/// A field whose doc values are not keyword ones, or a dictionary that
/// cannot be read.
pub(crate) fn ordinal_map_of(readers: &[&SegmentReader], field: &str) -> Result<OrdinalMap> {
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
    OrdinalMap::build_streaming(&mut refs).map_err(store_err)
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

/// Whether a segment's counts for `field` can come from its postings alone
/// ([`segment_counts`]' `tryCollectFromTermFrequencies`), given that every
/// document matches: the field has postings with at most
/// [`MAX_PRECOMPUTE_CARDINALITY`] terms.
pub(crate) fn precomputable(
    postings: &lucene_codecs::blocktree::BlockTreeFields,
    field: &str,
) -> bool {
    postings.field(field).is_some_and(|t| {
        usize::try_from(t.num_terms).is_ok_and(|n| n <= MAX_PRECOMPUTE_CARDINALITY)
    })
}

/// Per-segment scratch for [`segment_counts`], reused across segments.
#[derive(Default)]
pub(crate) struct TermsScratch {
    ords: Vec<i64>,
    /// A window of a single-valued column ([`count_window_stream`]).
    window: Vec<i64>,
    /// The segment's counts by segment ordinal.
    local: Vec<u32>,
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
            bad_ord = count_window_stream(&mut r, accept, max_doc, map.len(), scratch)?.or(bad_ord);
            // The segment's counts into the global ones, ordinal by ordinal:
            // the same sums as one per document. (`map` holds one global
            // ordinal per segment ordinal, each inside `counts`.)
            for (&g, &n) in map.iter().zip(&scratch.local) {
                if let Some(c) = counts.get_mut(g as usize) {
                    *c += u64::from(n);
                }
            }
        }
        (Ords::Single(mut r), ColumnRead::Seek(docs)) => {
            for &doc in *docs {
                if let Some(ord) = r.value(doc)? {
                    bump(ord);
                }
            }
        }
        (Ords::Multi(mut r), ColumnRead::Stream(accept)) => r.for_each_accepted(
            0,
            max_doc,
            |doc| accept.test(doc),
            |_, ords| {
                for &ord in ords {
                    bump(ord);
                }
            },
        )?,
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

/// Documents per [`count_window_stream`] window: a multiple of 64, so each
/// window's presence words line up with the segment's bit sets.
const COUNT_WINDOW: usize = 1024;

/// A single-valued ordinal column streamed a window at a time
/// ([`NumericReader::fill_window`]), each window's documents with a value
/// and `accept`ed taken a word at a time, into `scratch.local` -- the
/// segment's counts by segment ordinal, OpenSearch's `LowCardinality`
/// shape: one array index per match rather than a global-ordinal lookup
/// too; `ords` is the segment's dictionary size. `Some(ordinal)` for an
/// ordinal past the dictionary (counted nowhere).
fn count_window_stream(
    r: &mut NumericReader<'_>,
    accept: &crate::aggs::Accept<'_>,
    max_doc: i32,
    ords: usize,
    scratch: &mut TermsScratch,
) -> Result<Option<i64>> {
    // A few ordinals take most matches each: their counters are spread over
    // lanes (summed after), so consecutive matches of one ordinal do not
    // each wait on the previous increment's store.
    if ords <= LANED_ORDS {
        count_lanes::<COUNT_LANES>(r, accept, max_doc, ords, scratch)
    } else {
        count_lanes::<1>(r, accept, max_doc, ords, scratch)
    }
}

/// Counter lanes per ordinal for a small dictionary ([`count_window_stream`]).
const COUNT_LANES: usize = 4;

/// The most ordinals counted in [`COUNT_LANES`] lanes: their counters stay
/// in the first-level cache.
const LANED_ORDS: usize = 2048;

/// [`count_window_stream`] with `L` counters per ordinal (`local[o * L +
/// lane]`, the lane cycling per match), folded to one per ordinal at the end.
/// An ordinal past the dictionary (or negative) lands in one extra
/// ordinal's counters, so the loop has no branch for it; only when that
/// is not empty is the column read again for the ordinal to report.
#[inline(never)]
fn count_lanes<const L: usize>(
    r: &mut NumericReader<'_>,
    accept: &crate::aggs::Accept<'_>,
    max_doc: i32,
    ords: usize,
    scratch: &mut TermsScratch,
) -> Result<Option<i64>> {
    let TermsScratch { window, local, .. } = scratch;
    local.clear();
    local.resize(ords.saturating_add(1).saturating_mul(L), 0);
    let counters = local.as_mut_slice();
    window.resize(COUNT_WINDOW, 0);
    let window = window.as_mut_slice();
    let mut present = [0u64; COUNT_WINDOW / 64];
    let mut lane = 0usize;
    let mut start = 0i32;
    while start < max_doc {
        // ARITH: `start < max_doc`, so the difference is positive; `start`
        // stays a multiple of `COUNT_WINDOW` below `max_doc + COUNT_WINDOW`,
        // and `base + w` indexes the segment's words; `o <= ords`, so
        // `o * L + lane` is below the counters' `(ords + 1) * L`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            let len = ((max_doc - start) as usize).min(COUNT_WINDOW);
            r.fill_window(start, &mut window[..len], &mut present)?;
            let base = start as usize >> 6;
            let words = len.div_ceil(64);
            for (w, have) in present[..words].iter_mut().enumerate() {
                *have &= accept.word(base + w);
            }
            for (w, &have) in present[..words].iter().enumerate() {
                let mut bits = have;
                let docs = &window[w * 64..(w * 64 + 64).min(len)];
                while bits != 0 {
                    let i = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let o = (docs.get(i).copied().unwrap_or(-1) as u64).min(ords as u64) as usize;
                    if let Some(c) = counters.get_mut(o * L + lane) {
                        *c += 1;
                    }
                    lane = (lane + 1) % L;
                }
            }
            start += len as i32;
        }
    }
    // ARITH: `ords * L + L` is the counters' length.
    #[allow(clippy::arithmetic_side_effects)]
    let bad = counters[ords * L..].iter().any(|&n| n > 0);
    if L > 1 {
        // Ordinal `o`'s lanes sit at `o * L..`, at or past `o`: folding in
        // ascending order reads each before it is overwritten.
        for o in 0..ords {
            // ARITH: `o < ords`, so `o * L + L <= ords * L`, inside.
            #[allow(clippy::arithmetic_side_effects)]
            let sum = counters[o * L..o * L + L].iter().sum();
            counters[o] = sum;
        }
    }
    local.truncate(ords);
    if !bad {
        return Ok(None);
    }
    // A corrupt column: the first ordinal outside the dictionary, read again.
    let mut first = None;
    r.for_each_value(0, max_doc, |doc, ord| {
        let outside = usize::try_from(ord).map_or(true, |o| o >= ords);
        if first.is_none() && outside && accept.test(doc) {
            first = Some(ord);
        }
    })?;
    Ok(first)
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

    /// The streamed count (laned for a small dictionary, plain for a large
    /// one) against one ordinal read per accepted document, and an ordinal
    /// past a (here: deliberately understated) dictionary reported as the
    /// first such accepted ordinal.
    #[test]
    fn streamed_counts_match_a_per_document_read_and_report_a_bad_ordinal() {
        let dir = lucene_store::FsDirectory::open(std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/terms_aggs_index"
        )));
        let reader = crate::directory_reader::DirectoryReader::open(&dir).unwrap();
        // Laned (a small dictionary) and plain both met, each with a bad ordinal.
        let mut seen = [false; 2];
        for field in ["kw", "sk", "hk"] {
            for seg in reader.segment_readers() {
                let Some((Ords::Single(mut r), _, entry)) = keyword_column(seg, field).unwrap()
                else {
                    continue;
                };
                let ords = usize::try_from(entry.terms_dict_size).unwrap();
                let max_doc = seg.max_doc;
                // Every third document accepted.
                let words: Vec<u64> = (0..usize::try_from(max_doc).unwrap().div_ceil(64))
                    .map(|_| 0x9249_2492_4924_9249)
                    .collect();
                let accept = crate::aggs::Accept::Marked(&words);
                let mut want = vec![0u32; ords];
                for doc in 0..max_doc {
                    if accept.test(doc) {
                        if let Some(o) = r.value(doc).unwrap() {
                            want[usize::try_from(o).unwrap()] += 1;
                        }
                    }
                }
                let mut scratch = TermsScratch::default();
                let bad = count_window_stream(&mut r, &accept, max_doc, ords, &mut scratch);
                assert_eq!(bad.unwrap(), None, "{field}");
                assert_eq!(scratch.local, want, "{field}");
                // One ordinal fewer than the dictionary holds: the top one
                // is outside it, counted nowhere and reported.
                let top = i64::try_from(ords - 1).unwrap();
                if want[ords - 1] > 0 {
                    let bad = count_window_stream(&mut r, &accept, max_doc, ords - 1, &mut scratch);
                    assert_eq!(bad.unwrap(), Some(top), "{field}");
                    assert_eq!(scratch.local[..], want[..ords - 1], "{field}");
                    seen[usize::from(ords - 1 > LANED_ORDS)] = true;
                }
            }
        }
        assert_eq!(seen, [true, true]);
    }

    /// A field no segment has doc values for merges to no ordinals, and the
    /// aggregation over it has no buckets and nothing left over.
    #[test]
    fn a_field_without_doc_values_has_no_ordinals_or_buckets() {
        let dir = lucene_store::FsDirectory::open(std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/terms_aggs_index"
        )));
        let reader = crate::directory_reader::DirectoryReader::open(&dir).unwrap();
        let readers = reader.segment_readers();
        assert!(readers.len() > 1);
        let global = GlobalOrds::build(readers, "no_such_field").unwrap();
        assert_eq!(global.value_count(), 0);
        assert!(global.segment_map(0).is_some_and(<[i64]>::is_empty));
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let all = BooleanQuery {
            must: vec![crate::query::Clause::MatchAllDocs(
                crate::query::MatchAllDocsQuery::new(reader.max_doc()),
            )],
            ..Default::default()
        };
        let r = terms(&segments, readers, &all, "no_such_field", 10).unwrap();
        assert_eq!(r, TermsResult::default());
        // The keyword field itself does have them.
        assert!(GlobalOrds::build(readers, "kw").unwrap().value_count() > 0);
    }

    /// Counts come off the postings only for a field with postings and at
    /// most `MAX_PRECOMPUTE_CARDINALITY` terms.
    #[test]
    fn only_a_small_indexed_field_is_precomputable() {
        let dir = lucene_store::FsDirectory::open(std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/terms_aggs_index"
        )));
        let reader = crate::directory_reader::DirectoryReader::open(&dir).unwrap();
        let opened = reader.open_segments().unwrap();
        for seg in opened.as_open_segments() {
            assert!(precomputable(seg.fields, "body"));
            assert!(!precomputable(seg.fields, "kw"), "doc values only");
            assert!(!precomputable(seg.fields, "no_such_field"));
        }
    }
}
