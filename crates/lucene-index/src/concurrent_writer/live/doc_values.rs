//! A buffer's doc values kept as columns while its documents arrive, so a
//! chunk of its documents is written from slices of them rather than from
//! the documents decoded one by one.

use std::collections::HashMap;
use std::sync::Arc;

use lucene_codecs::doc_values::{
    BinaryColumn, SortedNumericColumn, SortedOrdsColumn, SortedSetOrdsColumn,
};
use lucene_codecs::field_infos::DocValuesType;
use lucene_codecs::stored_fields::{FieldValue, StoredField};

use crate::index_writer::PrebuiltDocValues;

/// One field's values, by document, in the order the documents arrived.
/// `ends[k]` is where the values of document `docs[k]` end.
#[derive(Debug, Clone)]
enum Column {
    Numeric {
        docs: Vec<i32>,
        values: Vec<i64>,
    },
    SortedNumeric {
        docs: Vec<i32>,
        ends: Vec<u32>,
        values: Vec<i64>,
        /// How many of `values` are sorted within their document.
        sorted: usize,
    },
    Binary {
        docs: Vec<i32>,
        ends: Vec<u32>,
        bytes: Vec<u8>,
    },
    /// SORTED and SORTED_SET: each value as the id of its bytes in
    /// `terms`, in arrival order.
    Terms {
        set: bool,
        terms: Terms,
        docs: Vec<i32>,
        ends: Vec<u32>,
        ids: Vec<u32>,
    },
}

/// Distinct byte strings, numbered as first seen.
#[derive(Debug, Clone, Default)]
struct Terms {
    ids: HashMap<Arc<[u8]>, u32>,
    list: TermList,
    bytes: usize,
}

/// How many terms a [`TermList`] block holds.
const TERM_BLOCK: usize = 1024;

/// Terms by id, in blocks that stop changing once full, so a copy of the
/// list shares them and copies only the last, open one.
#[derive(Debug, Clone, Default)]
struct TermList {
    sealed: Vec<Arc<[Arc<[u8]>]>>,
    open: Vec<Arc<[u8]>>,
}

impl TermList {
    // ARITH: a count of in-memory terms.
    #[allow(clippy::arithmetic_side_effects)]
    fn len(&self) -> usize {
        self.sealed.len() * TERM_BLOCK + self.open.len()
    }

    fn push(&mut self, term: Arc<[u8]>) {
        self.open.push(term);
        if self.open.len() == TERM_BLOCK {
            self.sealed.push(std::mem::take(&mut self.open).into());
        }
    }

    /// Term `id`'s bytes; empty for an id past the list's end.
    fn get(&self, id: u32) -> &[u8] {
        let id = id as usize;
        let (block, at) = (id / TERM_BLOCK, id % TERM_BLOCK);
        let term = match self.sealed.get(block) {
            Some(b) => b.get(at),
            None if block == self.sealed.len() => self.open.get(at),
            None => None,
        };
        term.map_or(&[], |t| t)
    }

    // ARITH: capacities of in-memory vectors.
    #[allow(clippy::arithmetic_side_effects)]
    fn capacity(&self) -> usize {
        self.sealed.len() * TERM_BLOCK + self.open.capacity()
    }
}

impl Terms {
    fn id(&mut self, value: &[u8]) -> Option<u32> {
        if let Some(&id) = self.ids.get(value) {
            return Some(id);
        }
        let id = u32::try_from(self.list.len()).ok()?;
        let owned: Arc<[u8]> = value.into();
        self.bytes = self.bytes.saturating_add(owned.len());
        self.list.push(Arc::clone(&owned));
        self.ids.insert(owned, id);
        Some(id)
    }
}

fn number(v: &FieldValue) -> Option<i64> {
    match v {
        FieldValue::Long(x) => Some(*x),
        FieldValue::Int(x) => Some(i64::from(*x)),
        _ => None,
    }
}

fn bytes(v: &FieldValue) -> Option<&[u8]> {
    match v {
        FieldValue::String(s) => Some(s.as_bytes()),
        FieldValue::Binary(b) => Some(b),
        _ => None,
    }
}

/// Records that document `doc`'s values now end at `end`.
fn bump(docs: &mut Vec<i32>, ends: &mut Vec<u32>, doc: i32, end: usize) -> bool {
    let Ok(end) = u32::try_from(end) else {
        return false;
    };
    match (docs.last(), ends.last_mut()) {
        (Some(&d), Some(e)) if d == doc => *e = end,
        _ => {
            docs.push(doc);
            ends.push(end);
        }
    }
    true
}

/// Where the values of the documents at `lo..hi` in `docs` start and end.
fn value_span(ends: &[u32], lo: usize, hi: usize) -> (usize, usize) {
    let at = |k: usize| {
        k.checked_sub(1)
            .and_then(|k| ends.get(k))
            .map_or(0, |&e| e as usize)
    };
    (at(lo), at(hi))
}

/// How many of `docs` (ascending) are below `doc`, searched from the end:
/// views read the newest documents, so the answer is mostly near it.
pub(super) fn before_from_end(docs: &[i32], doc: i32) -> usize {
    let mut hi = docs.len();
    let mut step = 1usize;
    // Gallop back to a position whose document is below `doc`.
    let lo = loop {
        let at = hi.saturating_sub(step);
        if docs.get(at).is_none_or(|&d| d < doc) || at == 0 {
            break at;
        }
        hi = at;
        step = step.saturating_mul(2);
    };
    let window = docs.get(lo..hi).unwrap_or_default();
    lo.saturating_add(window.partition_point(|&d| d < doc))
}

/// Each document's value count, from where its values end.
fn counts(ends: &[u32], from: usize) -> Vec<u32> {
    let mut last = from;
    ends.iter()
        .map(|&e| {
            let n = (e as usize).saturating_sub(last);
            last = e as usize;
            u32::try_from(n).unwrap_or(u32::MAX)
        })
        .collect()
}

impl Column {
    fn new(kind: DocValuesType) -> Option<Column> {
        Some(match kind {
            DocValuesType::Numeric => Column::Numeric {
                docs: Vec::new(),
                values: Vec::new(),
            },
            DocValuesType::SortedNumeric => Column::SortedNumeric {
                docs: Vec::new(),
                ends: Vec::new(),
                values: Vec::new(),
                sorted: 0,
            },
            DocValuesType::Binary => Column::Binary {
                docs: Vec::new(),
                ends: Vec::new(),
                bytes: Vec::new(),
            },
            DocValuesType::Sorted | DocValuesType::SortedSet => Column::Terms {
                set: kind == DocValuesType::SortedSet,
                terms: Terms::default(),
                docs: Vec::new(),
                ends: Vec::new(),
                ids: Vec::new(),
            },
            _ => return None,
        })
    }

    /// Adds one of document `doc`'s values; a document's values arrive
    /// together, after every earlier document's. `false` when the field
    /// does not take it: a value of another kind.
    fn add(&mut self, doc: i32, value: &FieldValue) -> bool {
        match self {
            // A single-valued field keeps a document's first value, as the
            // documents are read.
            Column::Numeric { docs, .. } | Column::Binary { docs, .. }
                if docs.last() == Some(&doc) =>
            {
                true
            }
            Column::Terms {
                set: false, docs, ..
            } if docs.last() == Some(&doc) => true,
            Column::Numeric { docs, values } => match number(value) {
                Some(v) => {
                    docs.push(doc);
                    values.push(v);
                    true
                }
                None => false,
            },
            Column::SortedNumeric {
                docs, ends, values, ..
            } => match number(value) {
                Some(v) => {
                    values.push(v);
                    bump(docs, ends, doc, values.len())
                }
                None => false,
            },
            Column::Binary {
                docs,
                ends,
                bytes: all,
            } => match bytes(value) {
                Some(b) => {
                    all.extend_from_slice(b);
                    bump(docs, ends, doc, all.len())
                }
                None => false,
            },
            Column::Terms {
                set: _,
                terms,
                docs,
                ends,
                ids,
            } => match bytes(value) {
                Some(b) => match terms.id(b) {
                    Some(id) => {
                        ids.push(id);
                        bump(docs, ends, doc, ids.len())
                    }
                    None => false,
                },
                _ => false,
            },
        }
    }

    /// Sorts the last document's SORTED_NUMERIC values, once they are all
    /// in; a no-op until it has more.
    fn finish_doc(&mut self) {
        if let Column::SortedNumeric { values, sorted, .. } = self {
            if let Some(tail) = values.get_mut(*sorted..) {
                tail.sort_unstable();
            }
            *sorted = values.len();
        }
    }

    // ARITH: capacities of in-memory vectors times their element sizes.
    #[allow(clippy::arithmetic_side_effects)]
    fn ram_bytes(&self) -> usize {
        match self {
            Column::Numeric { docs, values } => docs.capacity() * 4 + values.capacity() * 8,
            Column::SortedNumeric {
                docs, ends, values, ..
            } => docs.capacity() * 4 + ends.capacity() * 4 + values.capacity() * 8,
            Column::Binary { docs, ends, bytes } => {
                docs.capacity() * 4 + ends.capacity() * 4 + bytes.capacity()
            }
            Column::Terms {
                terms,
                docs,
                ends,
                ids,
                ..
            } => {
                // Each distinct term: its bytes, an `Arc` header, a list
                // slot and a map entry.
                terms.bytes.saturating_add(terms.list.capacity() * 64)
                    + docs.capacity() * 4
                    + ends.capacity() * 4
                    + ids.capacity() * 4
            }
        }
    }

    /// Where documents `start..end` are in the column: their positions in
    /// `docs` and where their values start and end.
    fn span(&self, start: i32, end: i32) -> ((usize, usize), (usize, usize)) {
        let docs_of = |docs: &[i32]| {
            let hi = before_from_end(docs, end);
            let lo = before_from_end(docs.get(..hi).unwrap_or_default(), start);
            (lo, hi)
        };
        match self {
            Column::Numeric { docs, .. } => (docs_of(docs), (0, 0)),
            Column::SortedNumeric { docs, ends, .. }
            | Column::Binary { docs, ends, .. }
            | Column::Terms { docs, ends, .. } => {
                let (lo, hi) = docs_of(docs);
                ((lo, hi), value_span(ends, lo, hi))
            }
        }
    }

    /// Documents `start..end`'s values, numbered from `start`, in the
    /// shape the doc-values writer takes.
    fn range(&self, start: i32, end: i32) -> Option<PrebuiltDocValues> {
        let (docs, values) = self.span(start, end);
        self.prebuilt(start, end, docs, values, 0)
    }

    /// Documents `start..end`'s values copied out as they are kept, for
    /// [`DocValuesRun::build`] to put in the writer's shape later.
    fn run(&self, start: i32, end: i32) -> Option<DocValuesRun> {
        let ((lo, hi), (from, to)) = self.span(start, end);
        let column = match self {
            Column::Numeric { docs, values } => Column::Numeric {
                docs: docs.get(lo..hi)?.to_vec(),
                values: values.get(lo..hi)?.to_vec(),
            },
            Column::SortedNumeric {
                docs, ends, values, ..
            } => Column::SortedNumeric {
                docs: docs.get(lo..hi)?.to_vec(),
                ends: ends.get(lo..hi)?.to_vec(),
                values: values.get(from..to)?.to_vec(),
                sorted: 0,
            },
            Column::Binary { docs, ends, bytes } => Column::Binary {
                docs: docs.get(lo..hi)?.to_vec(),
                ends: ends.get(lo..hi)?.to_vec(),
                bytes: bytes.get(from..to)?.to_vec(),
            },
            Column::Terms {
                set,
                terms,
                docs,
                ends,
                ids,
            } => Column::Terms {
                set: *set,
                terms: Terms {
                    ids: HashMap::new(),
                    list: terms.list.clone(),
                    bytes: 0,
                },
                docs: docs.get(lo..hi)?.to_vec(),
                ends: ends.get(lo..hi)?.to_vec(),
                ids: ids.get(from..to)?.to_vec(),
            },
        };
        Some(DocValuesRun {
            start,
            end,
            from,
            column,
        })
    }

    /// The documents at `docs` in the column, numbered from `start`, in the
    /// shape the doc-values writer takes; `values` is where their values
    /// start and end, of which the column holds those from `base` on.
    // ARITH: indices within in-memory columns; document numbers at or
    // above `start`.
    #[allow(clippy::arithmetic_side_effects)]
    fn prebuilt(
        &self,
        start: i32,
        end: i32,
        (lo, hi): (usize, usize),
        (from, to): (usize, usize),
        base: usize,
    ) -> Option<PrebuiltDocValues> {
        let shift = |d: &[i32]| d.iter().map(|&x| x - start).collect::<Vec<i32>>();
        let (at, until) = (from.checked_sub(base)?, to.checked_sub(base)?);
        Some(match self {
            Column::Numeric { docs, values } => {
                let docs = docs.get(lo..hi)?;
                let values = values.get(lo..hi)?;
                if i64::try_from(docs.len()).ok() == Some(i64::from(end - start)) {
                    PrebuiltDocValues::Numeric(values.to_vec())
                } else {
                    PrebuiltDocValues::SparseNumeric(
                        docs.iter()
                            .map(|&d| d - start)
                            .zip(values.iter().copied())
                            .collect(),
                    )
                }
            }
            Column::SortedNumeric {
                docs, ends, values, ..
            } => PrebuiltDocValues::SortedNumeric(SortedNumericColumn {
                docs: shift(docs.get(lo..hi)?),
                counts: counts(ends.get(lo..hi)?, from),
                values: values.get(at..until)?.to_vec(),
            }),
            Column::Binary { docs, ends, bytes } => PrebuiltDocValues::Binary(BinaryColumn {
                docs: shift(docs.get(lo..hi)?),
                lengths: counts(ends.get(lo..hi)?, from),
                bytes: bytes.get(at..until)?.to_vec(),
            }),
            Column::Terms {
                set,
                terms,
                docs,
                ends,
                ids: all_ids,
            } => {
                let ids = all_ids.get(at..until)?;
                // The range's distinct terms in byte order, and each id's
                // ordinal among them.
                let mut used: Vec<u32> = ids.to_vec();
                used.sort_unstable();
                used.dedup();
                if used
                    .last()
                    .is_some_and(|&id| id as usize >= terms.list.len())
                {
                    return None;
                }
                used.sort_unstable_by(|&a, &b| terms.list.get(a).cmp(terms.list.get(b)));
                let mut ord_of: HashMap<u32, i64> = HashMap::with_capacity(used.len());
                for (o, &id) in used.iter().enumerate() {
                    ord_of.insert(id, o as i64);
                }
                let dict: Vec<Vec<u8>> =
                    used.iter().map(|&id| terms.list.get(id).to_vec()).collect();
                let counts = counts(ends.get(lo..hi)?, from);
                let docs = shift(docs.get(lo..hi)?);
                if *set {
                    let mut ords = Vec::with_capacity(ids.len());
                    let mut out_counts = Vec::with_capacity(counts.len());
                    let mut k = 0usize;
                    for &c in &counts {
                        let c = c as usize;
                        let at = ords.len();
                        ords.extend(ids.get(k..k + c)?.iter().map(|id| ord_of[id]));
                        k += c;
                        let doc_ords = &mut ords[at..];
                        doc_ords.sort_unstable();
                        let mut w = 0usize;
                        for r in 0..doc_ords.len() {
                            if w == 0 || doc_ords[r] != doc_ords[w - 1] {
                                doc_ords[w] = doc_ords[r];
                                w += 1;
                            }
                        }
                        ords.truncate(at + w);
                        out_counts.push(u32::try_from(w).ok()?);
                    }
                    PrebuiltDocValues::SortedSet(SortedSetOrdsColumn {
                        dict,
                        docs,
                        counts: out_counts,
                        ords,
                    })
                } else {
                    PrebuiltDocValues::Sorted(SortedOrdsColumn {
                        dict,
                        docs,
                        ords: ids.iter().map(|id| ord_of[id]).collect(),
                    })
                }
            }
        })
    }

    fn doc_values_type(&self) -> DocValuesType {
        match self {
            Column::Numeric { .. } => DocValuesType::Numeric,
            Column::SortedNumeric { .. } => DocValuesType::SortedNumeric,
            Column::Binary { .. } => DocValuesType::Binary,
            Column::Terms { set: true, .. } => DocValuesType::SortedSet,
            Column::Terms { set: false, .. } => DocValuesType::Sorted,
        }
    }
}

/// One field's values for a run of documents, copied out of its column as
/// kept and put in the doc-values writer's shape only when read.
#[derive(Debug, Clone)]
pub(crate) struct DocValuesRun {
    start: i32,
    end: i32,
    /// Where the run's values start among the column's.
    from: usize,
    column: Column,
}

impl DocValuesRun {
    /// The field's type; a written field of another type cannot take it.
    pub(crate) fn doc_values_type(&self) -> DocValuesType {
        self.column.doc_values_type()
    }

    /// The run's values, numbered from its first document, in the shape
    /// the doc-values writer takes.
    pub(crate) fn build(&self) -> Option<PrebuiltDocValues> {
        let (docs, values) = match &self.column {
            Column::Numeric { docs, .. } => (docs.len(), 0),
            Column::SortedNumeric { docs, values, .. } => (docs.len(), values.len()),
            Column::Binary { docs, bytes, .. } => (docs.len(), bytes.len()),
            Column::Terms { docs, ids, .. } => (docs.len(), ids.len()),
        };
        let values = (self.from, self.from.saturating_add(values));
        self.column
            .prebuilt(self.start, self.end, (0, docs), values, self.from)
    }
}

/// Every doc-values field's column.
#[derive(Debug, Clone, Default)]
pub(in crate::concurrent_writer) struct DocValuesColumns {
    columns: HashMap<i32, Column>,
    /// A value no column takes: chunks are written from the documents,
    /// which report it, and nothing more is kept.
    refused: bool,
}

impl DocValuesColumns {
    /// Whether the columns stopped: the buffer's chunks are decoded.
    pub(in crate::concurrent_writer) fn refused(&self) -> bool {
        self.refused
    }

    /// Stops keeping columns: the buffer's chunks are decoded.
    pub(in crate::concurrent_writer) fn refuse(&mut self) {
        self.refused = true;
        self.columns = HashMap::new();
    }

    /// Adds document `doc`'s values; `kind` gives a field's type.
    pub(in crate::concurrent_writer) fn record(
        &mut self,
        doc: i32,
        values: &[StoredField],
        kind: impl Fn(i32) -> Option<DocValuesType>,
    ) {
        if self.refused {
            return;
        }
        for v in values {
            let column = match self.columns.entry(v.field_number) {
                std::collections::hash_map::Entry::Occupied(e) => Some(e.into_mut()),
                std::collections::hash_map::Entry::Vacant(e) => kind(v.field_number)
                    .and_then(Column::new)
                    .map(|c| e.insert(c)),
            };
            if !column.is_some_and(|c| c.add(doc, &v.value)) {
                return self.refuse();
            }
        }
        for v in values {
            if let Some(c) = self.columns.get_mut(&v.field_number) {
                c.finish_doc();
            }
        }
    }

    pub(in crate::concurrent_writer) fn ram_bytes(&self) -> usize {
        self.columns
            .values()
            .map(Column::ram_bytes)
            .fold(0, usize::saturating_add)
    }

    /// Every field's values for documents `start..end`, numbered from
    /// `start`; `None` once a value was refused.
    pub(in crate::concurrent_writer) fn range(
        &self,
        start: i32,
        end: i32,
    ) -> Option<Vec<(i32, PrebuiltDocValues)>> {
        if self.refused {
            return None;
        }
        let mut out = Vec::with_capacity(self.columns.len());
        // A field with no values in the range keeps its empty column: a
        // document of the range may still have the field.
        for (&field, c) in &self.columns {
            out.push((field, c.range(start, end)?));
        }
        Some(out)
    }

    /// [`Self::range`], each field's values copied out as kept, to be put
    /// in the writer's shape only for the fields read.
    pub(in crate::concurrent_writer) fn runs(
        &self,
        start: i32,
        end: i32,
    ) -> Option<Vec<(i32, DocValuesRun)>> {
        if self.refused {
            return None;
        }
        self.columns
            .iter()
            .map(|(&field, c)| Some((field, c.run(start, end)?)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use lucene_codecs::field_infos::{DocValuesSkipIndexType, FieldInfo};
    use lucene_store::FsDirectory;
    use lucene_util::test_support::TempDir;

    use super::*;
    use crate::concurrent_writer::ConcurrentIndexWriter;
    use crate::index_writer::IndexWriter;
    use crate::segment_info::LuceneVersion;

    fn value(field_number: i32, value: FieldValue) -> StoredField {
        StoredField {
            field_number,
            value,
        }
    }

    /// Doc values written from columns are byte for byte the ones written
    /// from the documents, for every type, dense or sparse, and over any
    /// run of the buffer.
    #[test]
    fn columns_write_the_documents_doc_values() {
        let tmp = TempDir::new("live-dv-columns");
        let dir = FsDirectory::open(tmp.path());
        let version = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
        w.enable_explicit_documents().unwrap();
        let mut field = |name: &str, kind: DocValuesType, skip: bool| {
            let skip = if skip {
                DocValuesSkipIndexType::Range
            } else {
                DocValuesSkipIndexType::None
            };
            w.register_field(FieldInfo::new(name, 0).with_doc_values(kind, skip, -1))
                .unwrap()
        };
        let num = field("num", DocValuesType::Numeric, true);
        let sparse = field("sparse", DocValuesType::Numeric, false);
        let multi = field("multi", DocValuesType::SortedNumeric, false);
        let bin = field("bin", DocValuesType::Binary, false);
        let kw = field("kw", DocValuesType::Sorted, true);
        let tags = field("tags", DocValuesType::SortedSet, false);
        let w = ConcurrentIndexWriter::new(w, 1).unwrap();
        let cfg = w.cfg();

        let words = ["delta", "alpha", "echo", "bravo", "charlie"];
        let docs: Vec<Vec<StoredField>> = (0..300i64)
            .map(|i| {
                let mut d = vec![value(num, FieldValue::Long(i * 7 - 1000))];
                if i % 3 == 0 {
                    d.push(value(sparse, FieldValue::Long(i)));
                }
                for k in 0..(i % 4) {
                    d.push(value(multi, FieldValue::Long((i * 31 + k * 17) % 50 - 20)));
                }
                if i % 5 != 1 {
                    d.push(value(bin, FieldValue::Binary(format!("b{i}").into_bytes())));
                }
                d.push(value(
                    kw,
                    FieldValue::String(words[(i % 5) as usize].to_string()),
                ));
                for k in 0..(i % 3) {
                    let w = words[((i + k * 2) % 5) as usize];
                    d.push(value(tags, FieldValue::Binary(w.as_bytes().to_vec())));
                    d.push(value(tags, FieldValue::Binary(w.as_bytes().to_vec())));
                }
                d
            })
            .collect();
        let mut columns = DocValuesColumns::default();
        for (doc, values) in docs.iter().enumerate() {
            columns.record(i32::try_from(doc).unwrap(), values, |n| {
                cfg.field_by_number(n).map(|f| f.doc_values_type)
            });
        }
        let mut compared = 0;
        for (start, end) in [(0usize, 300usize), (0, 128), (37, 211), (299, 300), (5, 6)] {
            let run: Vec<&[StoredField]> = docs[start..end].iter().map(Vec::as_slice).collect();
            let prebuilt = columns
                .range(i32::try_from(start).unwrap(), i32::try_from(end).unwrap())
                .unwrap();
            // A field none of the run's documents has keeps its empty column.
            let present: Vec<i32> = prebuilt.iter().map(|(n, _)| *n).collect();
            let all = [num, sparse, multi, bin, kw, tags];
            assert!(
                all.iter().all(|n| present.contains(n)),
                "documents {start}..{end}"
            );
            let (decoded, from_columns) = cfg.doc_values_both_ways(&run, &prebuilt);
            assert_eq!(decoded, from_columns, "documents {start}..{end}");
            // Copied out first and built later, the same files.
            let built: Vec<(i32, PrebuiltDocValues)> = columns
                .runs(i32::try_from(start).unwrap(), i32::try_from(end).unwrap())
                .unwrap()
                .iter()
                .map(|(n, r)| (*n, r.build().unwrap()))
                .collect();
            let (_, from_runs) = cfg.doc_values_both_ways(&run, &built);
            assert_eq!(decoded, from_runs, "documents {start}..{end}, built later");
            compared += 1;
        }
        assert_eq!(compared, 5);
    }

    /// Terms past a block's end are found by id, in a copy taken before
    /// and after more arrive.
    #[test]
    fn a_term_list_finds_terms_across_blocks() {
        let mut terms = Terms::default();
        let n = TERM_BLOCK * 2 + 5;
        for k in 0..n {
            assert_eq!(
                terms.id(format!("t{k}").as_bytes()),
                Some(u32::try_from(k).unwrap())
            );
        }
        assert_eq!(terms.id(b"t3"), Some(3));
        let copy = terms.list.clone();
        terms.id(b"later");
        assert_eq!(copy.len(), n);
        for k in [0, TERM_BLOCK - 1, TERM_BLOCK, 2 * TERM_BLOCK, n - 1] {
            let id = u32::try_from(k).unwrap();
            assert_eq!(copy.get(id), format!("t{k}").as_bytes());
            assert_eq!(terms.list.get(id), format!("t{k}").as_bytes());
        }
        assert_eq!(terms.list.get(u32::try_from(n).unwrap()), b"later");
    }

    #[test]
    fn a_value_of_the_wrong_kind_refuses_the_columns() {
        let mut columns = DocValuesColumns::default();
        let kind = |_| Some(DocValuesType::Numeric);
        columns.record(0, &[value(1, FieldValue::Long(3))], kind);
        assert!(columns.range(0, 1).is_some());
        columns.record(1, &[value(1, FieldValue::Binary(vec![1]))], kind);
        assert!(columns.range(0, 2).is_none());
        let mut columns = DocValuesColumns::default();
        columns.record(
            0,
            &[value(1, FieldValue::Long(3)), value(1, FieldValue::Long(4))],
            kind,
        );
        assert!(
            matches!(
                columns.range(0, 1).as_deref(),
                Some([(1, PrebuiltDocValues::Numeric(v))]) if v == &[3]
            ),
            "a NUMERIC field keeps a document's first value"
        );
    }
}

#[cfg(test)]
mod before_from_end_tests {
    use super::before_from_end;

    /// The search from the end finds what a binary search finds.
    #[test]
    fn searching_from_the_end_matches_a_binary_search() {
        let docs: Vec<i32> = (0..300).filter(|d| d % 3 != 1).collect();
        for n in [0usize, 1, 2, 7, docs.len()] {
            let docs = &docs[..n];
            for doc in -2..305 {
                assert_eq!(
                    before_from_end(docs, doc),
                    docs.partition_point(|&d| d < doc),
                    "{n} {doc}"
                );
            }
        }
    }
}
