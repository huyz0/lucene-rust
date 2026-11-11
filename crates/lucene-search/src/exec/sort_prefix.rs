//! A conjunction's filters on the leading keys of a segment's index sort,
//! matched as one run of doc ids.
//!
//! In a segment sorted by keys `k0, k1, ...`, the documents with one value
//! of `k0` are one run, inside which they are in `k1` order; the documents
//! of that run with one value of `k1` are a shorter run in `k2` order, and so
//! on. A boolean whose required, non-scoring clauses pin the first keys
//! (`k0 = a AND k1 = b AND k2 in [x, y]`) therefore matches one run, found
//! by binary searches over the sort keys' doc values instead of by reading
//! and intersecting the clauses' postings or points. A range on a key ends
//! the prefix: past it the next key is no longer in order.
//!
//! In a segment of document blocks only the parents follow the sort, each
//! block's children before its parent; a key whose field no child holds
//! orders the blocks, and the clauses on it match the parents of one run of
//! whole blocks. Inside a run pinned to one value per key the documents
//! come in the next keys' order, so a search sorted by those keys can stop
//! a segment at its first non-competitive hit ([`pinned_sort_keys`]).
//!
//! A key is a keyword (a `STRING` or single-valued `SortedSetSortField`
//! sort), pinned by a term, or a `LONG` sort, pinned by a one-dimensional
//! long points range (`long` and `date` fields) or a doc-values range. The
//! first key's term run is proved from the term's postings alone
//! ([`super::build::sorted_term_run`]); the runs of the keys after it are
//! found in the doc values and checked against the clause at both ends.
//!
//! Like Lucene's `IndexSortSortedNumericDocValuesRangeQuery`, which
//! OpenSearch puts in front of a sorted field's range, this answers a
//! clause from the sort key's doc values, so it relies on the field's
//! terms or points holding the same values as its doc values -- as a
//! keyword, `long` or `date` field indexes them. The folded clauses are not
//! looked up in the query cache: the run costs two binary searches a key.
//! This is no Lucene class; Lucene matches only a range on the first key
//! this way.

use lucene_codecs::doc_values::{self, NumericEntry, SortedNumericEntry, SortedSetKind};
use lucene_codecs::postings::PostingsFlags;
use lucene_index::segment_info::{IndexSortField, IndexSortKind, NumericSortKey};

use super::build::{seek_term, sorted_term_run, LeafContext};
use crate::directory_reader::SegmentReader;
use crate::extended_query::ExtendedQuery;
use crate::multi_segment::OpenSegment;
use crate::query::{BooleanQuery, Clause, TermQuery};
use crate::Result;
use lucene_util::fixed_bit_set::FixedBitSet;

/// What a clause pins one sort key to.
enum Bound<'q> {
    /// One term of a keyword key.
    Term(&'q TermQuery),
    /// The inclusive range `[lower, upper]` of a long key.
    Longs(i64, i64),
}

impl Bound<'_> {
    /// Whether the keys after this one stay in order inside its run.
    fn is_point(&self) -> bool {
        match *self {
            Bound::Term(_) => true,
            Bound::Longs(l, u) => l == u,
        }
    }
}

fn is_long_key(sort: &IndexSortField) -> bool {
    matches!(
        sort.kind,
        IndexSortKind::Numeric(NumericSortKey::Long(_))
            | IndexSortKind::SortedNumeric {
                key: NumericSortKey::Long(_),
                ..
            }
    )
}

fn is_keyword_key(sort: &IndexSortField) -> bool {
    matches!(
        sort.kind,
        IndexSortKind::String(_) | IndexSortKind::SortedSet { .. }
    )
}

/// The bound `clause` sets on `sort`'s key, when it sets one this module
/// can match as a run.
fn bound<'q>(
    reader: &SegmentReader,
    clause: &'q Clause,
    sort: &IndexSortField,
) -> Option<Bound<'q>> {
    match clause {
        Clause::Term(t) if t.field == sort.field && is_keyword_key(sort) => Some(Bound::Term(t)),
        Clause::PointsRange(p) if p.field == sort.field && is_long_key(sort) => {
            // A range over a long's points: one dimension of eight bytes.
            let fi = reader.field_infos().field_by_name(&p.field)?;
            (fi.point_dimension_count == 1 && fi.point_num_bytes == 8)
                .then_some(Bound::Longs(p.min, p.max))
        }
        Clause::Extended(e) => match e.as_ref() {
            ExtendedQuery::NumericDocValuesRange(r)
                if r.field == sort.field && is_long_key(sort) =>
            {
                Some(Bound::Longs(r.lower, r.upper))
            }
            ExtendedQuery::IndexSortRange(r) if r.field == sort.field && is_long_key(sort) => {
                Some(Bound::Longs(r.lower, r.upper))
            }
            ExtendedQuery::IndexOrDocValues(q) => bound(reader, &q.dv_query, sort),
            _ => None,
        },
        _ => None,
    }
}

/// One sort key's value for any document, read at random: a long, or a
/// keyword's ordinal. Only single-valued columns, where the key is the value.
/// In a segment of document blocks a document's key is its parent's -- the
/// next parent at or after it -- which is where the sort puts it.
struct Keys<'r> {
    column: Column<'r>,
    parents: Option<&'r FixedBitSet>,
}

enum Column<'r> {
    Numeric(&'r [u8], &'r NumericEntry),
    Sorted(&'r [u8], &'r SortedNumericEntry),
}

impl<'r> Keys<'r> {
    fn open(
        reader: &'r SegmentReader,
        field: &str,
        parents: Option<&'r FixedBitSet>,
    ) -> Option<Self> {
        let fi = reader.field_infos().field_by_name(field)?;
        let (meta, data) = reader.doc_values_for_field(fi.number)?;
        let column = if let Some(e) = meta.numeric_entry(fi.number) {
            Column::Numeric(data, e)
        } else if let Some(e) = meta.sorted_numeric_entry(fi.number) {
            Column::Sorted(data, e)
        } else if let Some(e) = meta.sorted_entry(fi.number) {
            Column::Numeric(data, &e.ords)
        } else {
            match &meta.sorted_set_entry(fi.number)?.kind {
                SortedSetKind::Single(e) => Column::Numeric(data, &e.ords),
                SortedSetKind::Multi { ords, .. } => Column::Sorted(data, ords),
            }
        };
        match column {
            Column::Sorted(_, e) if e.addresses.is_some() => None,
            column => Some(Keys { column, parents }),
        }
    }

    /// The document whose key `doc` sorts by: itself, or its block's parent.
    fn holder(&self, doc: i32) -> i32 {
        match self.parents {
            None => doc,
            Some(p) => p.next_set_bit(doc as usize).map_or(doc, |d| d as i32),
        }
    }

    /// The last document of `start..end` (`start < end`) a clause on the
    /// key can match: `end - 1`, or the last parent before `end`.
    fn last_holder(&self, end: i32) -> i32 {
        match self.parents {
            None => end - 1,
            Some(p) => p
                .prev_set_bit((end - 1) as usize)
                .map_or(end - 1, |d| d as i32),
        }
    }

    fn get(&self, doc: i32) -> Result<Option<i64>> {
        let doc = self.holder(doc);
        Ok(match self.column {
            Column::Numeric(data, e) => doc_values::numeric_value(data, e, doc)?,
            Column::Sorted(data, e) => doc_values::sorted_numeric_values(data, e, doc)?
                .first()
                .copied(),
        })
    }
}

/// The documents of `run` whose key is in `[lower, upper]`, `run` being in
/// `sort`'s key order: `start..end`, possibly empty.
fn key_range(
    keys: &Keys<'_>,
    sort: &IndexSortField,
    run: (i32, i32),
    lower: i64,
    upper: i64,
) -> Result<(i32, i32)> {
    let (_, sentinel) = sort.key_comparison();
    let direction: i32 = if sort.reverse { -1 } else { 1 };
    // `direction * compare(key(doc), bound)`, a document without a value
    // keyed by the sort's sentinel.
    let compare = |doc: i32, bound: i64| -> Result<i32> {
        let v = keys.get(doc)?.unwrap_or(sentinel);
        Ok(direction * v.cmp(&bound) as i32)
    };
    let (lower, upper) = if sort.reverse {
        (upper, lower)
    } else {
        (lower, upper)
    };
    // The first document of `from..run.1` for which `past` holds, `past`
    // being false then true along the run.
    let first = |from: i32, past: &dyn Fn(i32) -> Result<bool>| -> Result<i32> {
        let (mut low, mut high) = (from, run.1);
        while low < high {
            let mid = low + (high - low) / 2;
            if past(mid)? {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        Ok(low)
    };
    let start = first(run.0, &|d| Ok(compare(d, lower)? >= 0))?;
    let end = first(start, &|d| Ok(compare(d, upper)? > 0))?;
    Ok((start, end))
}

/// The run of `t`'s documents inside `run`, a run in `sort`'s key order
/// (`sort` a keyword key): `Some(None)` when `t` has none there, `None`
/// when its postings and the key's doc values do not agree at the run's
/// ends.
fn term_run(
    ctx: &LeafContext<'_>,
    keys: &Keys<'_>,
    sort: &IndexSortField,
    t: &TermQuery,
    run: (i32, i32),
) -> Result<Option<Option<(i32, i32)>>> {
    let (Some(doc_in), Some(field_terms)) = (ctx.doc_in, ctx.fields.field(&t.field)) else {
        return Ok(None);
    };
    let Some(seeked) = seek_term(ctx, field_terms, t)? else {
        return Ok(Some(None));
    };
    let mut cursor = field_terms.lazy_postings_for(&seeked, doc_in, PostingsFlags::DocsOnly)?;
    let first = cursor.advance(run.0)?;
    if first >= run.1 {
        return Ok(Some(None));
    }
    let Some(ord) = keys.get(first)? else {
        return Ok(None);
    };
    let (start, end) = key_range(keys, sort, run, ord, ord)?;
    if end <= start || keys.holder(start) != first {
        return Ok(None);
    }
    // The term's postings inside `run` start at `start` (its first parent
    // in a segment of blocks), hold the last document (or parent) before
    // `end` and stop there.
    let last = keys.last_holder(end);
    if last > first && cursor.advance(last)? != last {
        return Ok(None);
    }
    if cursor.next_doc()? < run.1 {
        return Ok(None);
    }
    Ok(Some(Some((start, end))))
}

/// The run of documents `[lower, upper]` keeps inside `run` on a long key,
/// `Some(None)` when it keeps none; `None` when a document without a value
/// would sort inside it (its sentinel is in range), which the run cannot
/// tell from a match.
fn long_run(
    keys: &Keys<'_>,
    sort: &IndexSortField,
    run: (i32, i32),
    lower: i64,
    upper: i64,
) -> Result<Option<Option<(i32, i32)>>> {
    if lower > upper {
        return Ok(Some(None));
    }
    let (_, sentinel) = sort.key_comparison();
    if (lower..=upper).contains(&sentinel) {
        return Ok(None);
    }
    let (start, end) = key_range(keys, sort, run, lower, upper)?;
    Ok(Some((start < end).then_some((start, end))))
}

/// What [`sort_prefix_run`] found.
pub(crate) struct PrefixRun<'r> {
    /// The documents `min..max` the clauses keep, `None` when none. In a
    /// segment of document blocks the run holds whole blocks and the
    /// clauses keep only its documents in `parents`.
    pub(crate) run: Option<(i32, i32)>,
    /// The positions of the clauses the run stands for.
    pub(crate) clauses: Vec<usize>,
    /// A segment of document blocks' parent documents.
    pub(crate) parents: Option<&'r FixedBitSet>,
    /// How many leading sort keys the clauses pin to one value each, with
    /// the run in the order of the key after them: inside the run the
    /// documents come in the order of the index sort's keys from there on.
    pub(crate) pinned: usize,
}

/// The run of doc ids that the clauses at `clauses` (required, scored or
/// not alike) keep together on the segment's leading sort keys, with the
/// positions of the clauses it stands for: `Some((None, _))` when they keep
/// no document. `None` when no clause bounds the first sort key, or the
/// segment is unsorted, or has document blocks without a parent field.
///
/// In a segment of document blocks only the parents follow the sort, each
/// after its children; a key whose field only the parents hold then orders
/// the blocks, and the run is of whole blocks, the clauses matching its
/// parents.
pub(crate) fn sort_prefix_run<'r>(
    ctx: &LeafContext<'r>,
    clauses: &[&Clause],
) -> Result<Option<PrefixRun<'r>>> {
    let Some(reader) = ctx.reader else {
        return Ok(None);
    };
    if reader.simpletext.is_some() {
        return Ok(None);
    }
    let blocks = if reader.has_blocks() {
        match reader.block_parents()? {
            Some(b) => Some(b),
            None => return Ok(None),
        }
    } else {
        None
    };
    let parents = blocks.map(|b| &b.parents);
    let Some(sort) = reader.index_sort() else {
        return Ok(None);
    };
    let mut run = (0, reader.max_doc);
    let mut used: Vec<usize> = Vec::new();
    let mut pinned = 0;
    for (k, key) in sort.iter().enumerate() {
        if blocks.is_some_and(|b| !b.keys_on_parents.get(k).copied().unwrap_or(false)) {
            break;
        }
        let Some((i, b)) = clauses
            .iter()
            .enumerate()
            .filter(|(i, _)| !used.contains(i))
            .find_map(|(i, c)| bound(reader, c, key).map(|b| (i, b)))
        else {
            break;
        };
        let keys = Keys::open(reader, &key.field, parents);
        let found = match (&b, &keys) {
            (Bound::Term(t), _) if k == 0 && parents.is_none() => sorted_term_run(ctx, t)?,
            (Bound::Term(t), Some(keys)) => term_run(ctx, keys, key, t, run)?,
            (Bound::Longs(l, u), Some(keys)) => long_run(keys, key, run, *l, *u)?,
            (_, None) => None,
        };
        let Some(found) = found else {
            break;
        };
        used.push(i);
        let Some(found) = found else {
            return Ok(Some(PrefixRun {
                run: None,
                clauses: used,
                parents,
                pinned: sort.len(),
            }));
        };
        run = found;
        // The next key is in order inside the run only when this one is
        // pinned to one value, which a multi-valued keyword's run (proved
        // from its postings) does not promise: a document's key is its
        // smallest value.
        if !b.is_point() || keys.is_none() {
            break;
        }
        pinned = k + 1;
    }
    if used.is_empty() {
        return Ok(None);
    }
    Ok(Some(PrefixRun {
        run: Some(run),
        clauses: used,
        parents,
        pinned,
    }))
}

/// The required clauses of `q`, those of its required booleans' included:
/// every document `q` matches matches each of them.
fn required_clauses<'q>(q: &'q BooleanQuery, out: &mut Vec<&'q Clause>) {
    for c in q.must.iter().chain(&q.filter) {
        match c {
            Clause::Boolean(b) => required_clauses(b, out),
            c => out.push(c),
        }
    }
}

/// How many of the segment's leading index sort keys `q`'s required clauses
/// pin to one value each such that its matches come in the order of the
/// sort's keys after them (`0` when none): a search sorted by those keys
/// sees its matches in its sort's order, as `TopFieldCollector` does a
/// search whose sort begins the index sort, and can end the segment at the
/// first one that is not competitive.
///
/// # Errors
/// A corrupt postings or doc-values region.
pub fn pinned_sort_keys(seg: &OpenSegment<'_>, q: &BooleanQuery) -> Result<usize> {
    let ctx = LeafContext::for_spans(seg);
    let mut clauses = Vec::new();
    required_clauses(q, &mut clauses);
    Ok(sort_prefix_run(&ctx, &clauses)?.map_or(0, |r| r.pinned))
}
