//! Keyword doc-values columns and the global ordinals built over them: a
//! segment's ordinal column and terms dictionary, and a reader's
//! [`GlobalOrds`] (every distinct term of a field across the segments, in term
//! order, with each segment's ordinals mapped onto them) and its
//! [`GlobalOrdsCache`]. Built once per reader and field by
//! [`crate::directory_reader::DirectoryReader::global_ords`], and counted
//! into by the aggregations.

use std::collections::HashMap;
use std::sync::Arc;

use lucene_codecs::doc_values::{NumericReader, SortedNumericReader, SortedSetKind};
use lucene_codecs::terms_dict::{TermsCursor, TermsDict, TermsDictEntry};

use crate::directory_reader::SegmentReader;
use crate::ordinal_map::{KeptTerms, OrdinalMap, SegmentSource, TermCursor};
use crate::Result;

/// A segment's ordinal column for the field.
pub enum Ords<'a> {
    Absent,
    Single(Box<NumericReader<'a>>),
    Multi(Box<SortedNumericReader<'a>>),
}

/// `field`'s keyword doc values in `reader`: its ordinal column and terms
/// dictionary, or `None` when the segment has no doc values for it.
pub fn keyword_column<'a>(
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

/// [`terms_entry`], but none for a segment with a keyword run of `field`
/// when `skip_runs`: its terms are found by bytes instead.
fn dict_entry<'a>(
    reader: &'a SegmentReader,
    field: &str,
    skip_runs: bool,
) -> Result<Option<(&'a [u8], &'a TermsDictEntry)>> {
    if skip_runs && reader.keyword_run(field).is_some() {
        return Ok(None);
    }
    terms_entry(reader, field)
}

/// [`keyword_column`] with its dictionary opened.
pub fn open_ords<'a>(
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
    map: Arc<OrdinalMap>,
    /// The distinct terms in global-ordinal order, while there are few
    /// ([`KEPT_TERMS`]): what [`Self::remapped`] looks terms up in.
    terms: Option<Arc<[Box<[u8]>]>>,
    /// Whether the segments with a keyword run of the field
    /// ([`SegmentReader::keyword_run`]) are left out: their documents are
    /// counted from the run, their terms found by bytes.
    skips_runs: bool,
}

/// At most this many terms, of at most as many bytes together, are kept
/// with a [`GlobalOrds`].
const KEPT_TERMS: usize = 1 << 16;

impl GlobalOrds {
    /// The merged ordinal map: global ordinal to segment and segment ordinal, and back.
    pub fn map(&self) -> &OrdinalMap {
        &self.map
    }

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
            map: Arc::new(ordinal_map_of(&readers, field)?),
            terms: None,
            skips_runs: false,
        })
    }

    /// [`Self::build`], also keeping the distinct terms while few, for a
    /// [`GlobalOrdsCache`] to look the terms of other segments up in.
    ///
    /// # Errors
    /// As [`Self::build`].
    pub fn build_keeping(readers: &[SegmentReader], field: &str, skip_runs: bool) -> Result<Self> {
        let readers: Vec<&SegmentReader> = readers.iter().collect();
        let (map, terms) = ordinal_map_keeping(&readers, field, skip_runs)?;
        Ok(GlobalOrds {
            map: Arc::new(map),
            terms: terms.map(Arc::from),
            skips_runs: skip_runs,
        })
    }

    /// The ordinal map itself, shared: what a global-ordinals join
    /// ([`crate::join::resolve_late_joins`]) maps its segments by.
    pub fn ordinal_map(&self) -> Arc<OrdinalMap> {
        Arc::clone(&self.map)
    }

    /// Whether these ordinals kept their terms.
    pub fn has_terms(&self) -> bool {
        self.terms.is_some()
    }

    /// Whether the segments with a keyword run are left out.
    pub fn skips_runs(&self) -> bool {
        self.skips_runs
    }

    /// Term `term`'s global ordinal, when these ordinals kept their terms
    /// and it is one of them.
    pub fn ord_of(&self, term: &[u8]) -> Option<usize> {
        let terms = self.terms.as_ref()?;
        terms.binary_search_by(|t| (**t).cmp(term)).ok()
    }

    /// These ordinals' terms over `readers`, where `kept[i]` is the
    /// segment of these ordinals that reader `i` is, if any: those
    /// segments' maps shared, the other readers' terms looked up. `None`
    /// when a looked-up term is new or one of theirs is in no reader, or
    /// the terms were too many to keep.
    ///
    /// # Errors
    /// As [`Self::build`].
    fn remapped(
        &self,
        readers: &[SegmentReader],
        kept: &[Option<usize>],
        field: &str,
    ) -> Result<Option<Self>> {
        let skip_runs = self.skips_runs;
        let Some(terms) = &self.terms else {
            return Ok(None);
        };
        let mut cursors = Vec::with_capacity(readers.len());
        for (reader, k) in readers.iter().zip(kept) {
            cursors.push(match (k, dict_entry(reader, field, skip_runs)?) {
                (None, Some((data, entry))) => {
                    Some(TermsCursor::open(data, entry).map_err(store_err)?)
                }
                _ => None,
            });
        }
        let mut empty: Vec<NoTerms> = std::iter::repeat_with(|| NoTerms)
            .take(readers.len())
            .collect();
        let mut sources: Vec<SegmentSource<'_>> = cursors
            .iter_mut()
            .zip(&mut empty)
            .zip(kept)
            .map(|((c, e), k)| match (k, c) {
                (Some(old), _) => SegmentSource::Kept(*old),
                (None, Some(c)) => SegmentSource::Terms(c),
                (None, None) => SegmentSource::Terms(e),
            })
            .collect();
        Ok(self
            .map
            .remapped(terms, &mut sources)
            .map_err(store_err)?
            .map(|map| GlobalOrds {
                map: Arc::new(map),
                terms: Some(Arc::clone(terms)),
                skips_runs: skip_runs,
            }))
    }

    /// Segment `seg`'s ordinals to global ones.
    pub fn segment_map(&self, seg: usize) -> Option<&[i64]> {
        self.map.segment_ords(seg)
    }

    /// The first segment holding global ordinal `g`.
    pub fn first_segment(&self, g: i64) -> Option<usize> {
        self.map.first_segment(g)
    }

    /// Global ordinal `g`'s ordinal in [`Self::first_segment`].
    pub fn first_segment_ord(&self, g: i64) -> Option<i64> {
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
    with_cursors(readers, field, false, |refs| {
        OrdinalMap::build_streaming(refs).map_err(store_err)
    })
}

/// [`ordinal_map_of`] with the distinct terms, while few.
fn ordinal_map_keeping(
    readers: &[&SegmentReader],
    field: &str,
    skip_runs: bool,
) -> Result<(OrdinalMap, Option<KeptTerms>)> {
    with_cursors(readers, field, skip_runs, |refs| {
        OrdinalMap::build_keeping_terms(refs, KEPT_TERMS).map_err(store_err)
    })
}

/// `f` over a cursor of each reader's dictionary of `field`, an empty one
/// for a reader without it.
fn with_cursors<T>(
    readers: &[&SegmentReader],
    field: &str,
    skip_runs: bool,
    f: impl FnOnce(&mut [&mut dyn TermCursor]) -> Result<T>,
) -> Result<T> {
    let mut cursors = Vec::with_capacity(readers.len());
    for reader in readers {
        cursors.push(match dict_entry(reader, field, skip_runs)? {
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
    f(&mut refs)
}

/// Global ordinals of readers that share most of their segments -- views
/// of one reader with buffered documents appended -- kept by each segment's
/// identity, so a reader holding a kept map's segments takes their maps
/// and looks up only its other segments' terms instead of merging every
/// dictionary again.
#[derive(Debug, Default)]
pub struct GlobalOrdsCache {
    by_field: std::sync::Mutex<HashMap<String, Vec<KeptMap>>>,
    /// Fields whose terms are too many to keep: their runs are not left
    /// out, so a map leaving them out is not built only to be dropped.
    unkept: std::sync::Mutex<std::collections::HashSet<String>>,
}

/// How many maps a cache keeps per field, the newest.
const CACHED_MAPS: usize = 4;

/// A kept map and its segments' keys.
type KeptMap = (Vec<SegmentKey>, Arc<GlobalOrds>);

/// A segment's doc values' identity: its id, unique to the segment, and
/// the field's doc-values generation, which a doc-values update moves on.
type SegmentKey = ([u8; lucene_util::string_helper::ID_LENGTH], i64);

/// The key of every segment left out for its keyword run.
const RUN_KEY: SegmentKey = ([0; lucene_util::string_helper::ID_LENGTH], -2);

/// Where each of `keys` sits in `old`: its position, or for the n-th
/// segment left out for its keyword run, the n-th such one's.
fn kept_at(keys: &[SegmentKey], old: &[SegmentKey]) -> Vec<Option<usize>> {
    let mut runs = old
        .iter()
        .enumerate()
        .filter(|(_, k)| **k == RUN_KEY)
        .map(|(i, _)| i);
    keys.iter()
        .map(|key| {
            if *key == RUN_KEY {
                runs.next()
            } else {
                old.iter().position(|o| o == key)
            }
        })
        .collect()
}

fn segment_key(reader: &SegmentReader, field: &str, skip_runs: bool) -> SegmentKey {
    // Every segment left out adds no terms: they all have one key.
    if skip_runs && reader.keyword_run(field).is_some() {
        return RUN_KEY;
    }
    let gen = reader
        .field_infos()
        .fields
        .iter()
        .find(|i| i.name == field)
        .map_or(-1, |i| i.doc_values_gen);
    (reader.segment_id(), gen)
}

impl GlobalOrdsCache {
    /// `readers`' global ordinals of `field`: a kept map's when one is over
    /// the same segments, else one taking the segments the kept map that
    /// shares the most with them holds, else built.
    ///
    /// # Errors
    /// As [`GlobalOrds::build`].
    pub fn ords(&self, readers: &[SegmentReader], field: &str) -> Result<Arc<GlobalOrds>> {
        self.ords_with(readers, field, false)
    }

    /// [`Self::ords`], leaving out the segments with a keyword run when
    /// `skip_runs`.
    ///
    /// # Errors
    /// As [`GlobalOrds::build`].
    pub fn ords_with(
        &self,
        readers: &[SegmentReader],
        field: &str,
        skip_runs: bool,
    ) -> Result<Arc<GlobalOrds>> {
        let unkept = |c: &Self| {
            c.unkept
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains(field)
        };
        if skip_runs && unkept(self) {
            return self.ords_with(readers, field, false);
        }
        let keys: Vec<SegmentKey> = readers
            .iter()
            .map(|r| segment_key(r, field, skip_runs))
            .collect();
        let slot = if skip_runs {
            format!("\u{1}{field}")
        } else {
            field.to_string()
        };
        let best = {
            let kept = self.by_field.lock().unwrap_or_else(|e| e.into_inner());
            kept.get(&slot).and_then(|maps| {
                maps.iter()
                    .map(|(k, g)| (kept_at(&keys, k), Arc::clone(g), k.len()))
                    .filter(|(at, _, _)| {
                        at.iter()
                            .zip(&keys)
                            .any(|(a, k)| a.is_some() && *k != RUN_KEY)
                    })
                    .max_by_key(|(at, _, _)| {
                        at.iter()
                            .zip(&keys)
                            .filter(|(a, k)| a.is_some() && **k != RUN_KEY)
                            .count()
                    })
            })
        };
        let built = match best {
            Some((at, g, n))
                if n == keys.len() && at.iter().enumerate().all(|(i, a)| *a == Some(i)) =>
            {
                return Ok(g)
            }
            Some((at, g, _)) => match g.remapped(readers, &at, field)? {
                Some(e) => e,
                None => GlobalOrds::build_keeping(readers, field, skip_runs)?,
            },
            None => GlobalOrds::build_keeping(readers, field, skip_runs)?,
        };
        let built = Arc::new(built);
        if skip_runs && built.terms.is_none() {
            self.unkept
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(field.to_string());
            return self.ords_with(readers, field, false);
        }
        if built.terms.is_some() {
            let mut kept = self.by_field.lock().unwrap_or_else(|e| e.into_inner());
            let maps = kept.entry(slot).or_default();
            if maps.len() >= CACHED_MAPS {
                maps.remove(0);
            }
            maps.push((keys, Arc::clone(&built)));
        }
        Ok(built)
    }
}

/// A segment without the field: no terms.
struct NoTerms;

impl TermCursor for NoTerms {
    fn next_term(&mut self) -> lucene_store::Result<Option<&[u8]>> {
        Ok(None)
    }
}

pub fn store_err(e: lucene_store::Error) -> crate::Error {
    crate::Error::from(lucene_codecs::doc_values::Error::from(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_left_out_for_their_runs_match_in_order() {
        let seg = |b: u8| ([b; lucene_util::string_helper::ID_LENGTH], -1);
        let old = [seg(1), RUN_KEY, seg(2), RUN_KEY];
        // The same segments: an exact hit, each at its own place.
        assert_eq!(
            kept_at(&old, &old),
            vec![Some(0), Some(1), Some(2), Some(3)]
        );
        // One more run than kept: the extra one has no match.
        let new = [seg(1), RUN_KEY, RUN_KEY, RUN_KEY, seg(3)];
        assert_eq!(
            kept_at(&new, &old),
            vec![Some(0), Some(1), Some(3), None, None]
        );
    }
}
