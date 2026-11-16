//! Port of the keying half of `org.apache.lucene.index.IndexSorter` for the
//! three index sorts whose per-document key is a **byte string** rather than
//! a number: `IndexSorter.StringSorter` (a `SortField.Type.STRING` over a
//! SORTED column, and `SortedSetSortField` over a SORTED_SET column through
//! `SortedSetSelector.wrap`) and `IndexSorter.BinarySorter`
//! (`BinarySortField` over a BINARY column).
//!
//! # One key shape for every sort
//!
//! Every consumer of an index sort in this port -- the sort-on-flush
//! (`IndexWriter::sort_buffer`), the sort-preserving merge
//! (`merge::merge_segments_mapped` via `MultiSorter`'s k-way order) and
//! `CheckIndex.testSort` -- compares one `Option<i64>` per document through
//! [`crate::segment_info::SortKeyComparator`]. The byte-keyed sorts reduce to
//! that shape exactly as Java reduces them:
//!
//! - `StringSorter.getDocComparator` compares a segment's **term ordinals**
//!   (`Integer.compare(ords[d1], ords[d2])`), and a SORTED dictionary's
//!   ordinals are the terms' ranks in unsigned-byte order. Its
//!   `getComparableProviders` compares **global** ordinals out of an
//!   `OrdinalMap` over every merging segment's dictionary -- again the rank
//!   of the term in the union of the dictionaries.
//! - `BinarySorter` compares the `BytesRef`s themselves (`BytesRef.compareTo`,
//!   unsigned bytes). Replacing each value by its rank among the distinct
//!   values being compared preserves every comparison, so the same
//!   comparator applies.
//!
//! A missing value is the same in all three: `missingOrd` is
//! `Integer.MAX_VALUE` for `STRING_LAST` and `Integer.MIN_VALUE` otherwise
//! (`StringSorter`), and `BinarySortField.comparator()` orders a `null`
//! first unless the missing value is `STRING_LAST` -- after which `reverse`
//! multiplies the whole comparison in both. So one sentinel per
//! [`crate::segment_info::StringMissingValue`] serves all three
//! ([`crate::segment_info::IndexSortField::key_comparison`]).
//!
//! [`rank_terms`] is that reduction, and it is the one place it happens:
//! ranking a single segment's values reproduces its dictionary's ordinals,
//! and ranking several segments' values together reproduces the
//! `OrdinalMap`'s global ordinals, which is why the flush, the merge and the
//! check cannot disagree about what "sorted by this string" means.
//!
//! # Rust-only shape
//!
//! Java's `SortedDocValuesWriter` hands `maybeSortSegment` the ordinals of
//! its `BytesRefHash` after `hash.sort()`; this port keeps no hash in the
//! document buffer, so the flush collects each document's bytes and ranks
//! them here, which yields the same ordinals.

use lucene_codecs::doc_values::{self, DocValuesMeta, SortedSetEntry, SortedSetKind};
use lucene_codecs::terms_dict;

use crate::segment_info::{IndexSortField, IndexSortKind, SortedSetSelector};

/// `SortedSetSelector.wrap`'s choice among one document's ordinals (or
/// terms), which are **ascending and unique** -- the shape a SORTED_SET
/// column stores and `SortedSetDocValuesWriter` builds. `None` for a
/// document with no value.
///
/// Java: `MinValue` reads the first `nextOrd()`, `MaxValue` the last,
/// `MiddleMinValue` the ordinal at `(docValueCount - 1) >>> 1` and
/// `MiddleMaxValue` the one at `docValueCount >>> 1` -- which agree for an
/// odd count and pick the lower / upper of the two middle values for an even
/// one.
pub fn select_sorted_set<T>(values: &[T], selector: SortedSetSelector) -> Option<&T> {
    let last = values.len().checked_sub(1)?;
    let at = match selector {
        SortedSetSelector::Min => 0,
        SortedSetSelector::Max => last,
        SortedSetSelector::MiddleMin => last / 2,
        SortedSetSelector::MiddleMax => values.len() / 2,
    };
    values.get(at)
}

/// Replaces every present value by its **rank** among the distinct values
/// present across all of `per_source` (unsigned-byte order, rank 0 the
/// smallest): the ordinal a SORTED dictionary over those values would give
/// it, and for several sources the global ordinal `OrdinalMap` gives it. A
/// missing value stays `None`, for the comparator's sentinel.
///
/// Ranks fit an `i32` whenever the values do: Lucene caps a segment at
/// `IndexWriter.MAX_DOCS` documents, and a SORTED column at as many
/// distinct terms.
pub fn rank_terms(per_source: &[Vec<Option<Vec<u8>>>]) -> Vec<Vec<Option<i64>>> {
    let mut distinct: Vec<&[u8]> = per_source
        .iter()
        .flatten()
        .filter_map(|v| v.as_deref())
        .collect();
    distinct.sort_unstable();
    distinct.dedup();
    per_source
        .iter()
        .map(|values| {
            values
                .iter()
                .map(|v| {
                    v.as_deref().map(|v| {
                        // Every value was inserted above, so the search
                        // always finds it.
                        distinct.binary_search(&v).unwrap_or_else(|at| at) as i64
                    })
                })
                .collect()
        })
        .collect()
}

/// The byte key of every document `0..max_doc` of one segment for one
/// byte-keyed sort tier, read out of the segment's own doc-values column:
/// the SORTED value for `STRING`, the `selector`'s pick among the SORTED_SET
/// values for `SortedSetSortField`, the BINARY value for `BinarySortField`.
///
/// `Ok(None)` when `sort` is a numeric kind (its key is read as a number
/// elsewhere) or the segment has no column of the right type for the field
/// -- a caller decides whether that is an all-missing column or an error.
pub fn read_segment_terms(
    dvd: &[u8],
    meta: &DocValuesMeta,
    sort: &IndexSortField,
    field_number: i32,
    max_doc: i32,
) -> doc_values::Result<Option<Vec<Option<Vec<u8>>>>> {
    let len = usize::try_from(max_doc).unwrap_or(0);
    match &sort.kind {
        IndexSortKind::Numeric(_) | IndexSortKind::SortedNumeric { .. } => Ok(None),
        IndexSortKind::String(_) => {
            let Some(entry) = meta.sorted_entry(field_number) else {
                return Ok(None);
            };
            let dict = terms_dict::decode_all_terms(dvd, &entry.terms)?;
            let mut ords = doc_values::NumericReader::new(dvd, &entry.ords);
            let mut out = Vec::with_capacity(len);
            for doc in 0..max_doc {
                out.push(term_at(&dict, ords.value(doc)?)?);
            }
            Ok(Some(out))
        }
        IndexSortKind::SortedSet { selector, .. } => {
            let Some(entry) = meta.sorted_set_entry(field_number) else {
                return Ok(None);
            };
            let dict = sorted_set_dict(dvd, entry)?;
            let mut ords = SortedSetOrds::new(dvd, entry);
            let mut scratch = Vec::new();
            let mut out = Vec::with_capacity(len);
            for doc in 0..max_doc {
                ords.ords(doc, &mut scratch)?;
                out.push(term_at(
                    &dict,
                    select_sorted_set(&scratch, *selector).copied(),
                )?);
            }
            Ok(Some(out))
        }
        IndexSortKind::Binary(_) => {
            let Some(entry) = meta.binary_entry(field_number) else {
                return Ok(None);
            };
            let mut reader = doc_values::BinaryReader::new(dvd, entry);
            let mut out = Vec::with_capacity(len);
            for doc in 0..max_doc {
                out.push(reader.value_ref(doc)?.map(<[u8]>::to_vec));
            }
            Ok(Some(out))
        }
    }
}

/// The term an ordinal names, refusing one past the dictionary rather than
/// inventing a key for it.
fn term_at(dict: &[Vec<u8>], ord: Option<i64>) -> doc_values::Result<Option<Vec<u8>>> {
    let Some(ord) = ord else {
        return Ok(None);
    };
    usize::try_from(ord)
        .ok()
        .and_then(|at| dict.get(at))
        .cloned()
        .map(Some)
        .ok_or_else(|| {
            doc_values::Error::Store(lucene_store::Error::Corrupted(format!(
                "doc-values ordinal {ord} is outside a dictionary of {} terms",
                dict.len()
            )))
        })
}

/// A SORTED_SET column's whole dictionary, in ordinal order.
fn sorted_set_dict(dvd: &[u8], entry: &SortedSetEntry) -> doc_values::Result<Vec<Vec<u8>>> {
    Ok(match &entry.kind {
        SortedSetKind::Single(sorted) => terms_dict::decode_all_terms(dvd, &sorted.terms)?,
        SortedSetKind::Multi { terms, .. } => terms_dict::decode_all_terms(dvd, terms)?,
    })
}

/// A forward cursor over one SORTED_SET column's per-document ordinals,
/// whichever of the two shapes it was written in.
enum SortedSetOrds<'a> {
    Single(Box<doc_values::NumericReader<'a>>),
    Multi(Box<doc_values::SortedNumericReader<'a>>),
}

impl<'a> SortedSetOrds<'a> {
    fn new(dvd: &'a [u8], entry: &'a SortedSetEntry) -> Self {
        match &entry.kind {
            SortedSetKind::Single(sorted) => {
                Self::Single(Box::new(doc_values::NumericReader::new(dvd, &sorted.ords)))
            }
            SortedSetKind::Multi { ords, .. } => {
                Self::Multi(Box::new(doc_values::SortedNumericReader::new(dvd, ords)))
            }
        }
    }

    fn ords(&mut self, doc: i32, out: &mut Vec<i64>) -> doc_values::Result<()> {
        match self {
            Self::Single(reader) => {
                out.clear();
                out.extend(reader.value(doc)?);
            }
            Self::Multi(reader) => reader.values(doc, out)?,
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
