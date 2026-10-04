//! Sorting parents by their children's doc values: `ToParentBlockJoinSortField`,
//! `BlockJoinSelector` and `ToParentDocValues` (Lucene 10.5.0).
//!
//! A [`ToParentBlockJoinSortField`] is a custom sort key
//! ([`crate::top_field::SortField::with_source`]) whose comparator reads, for
//! each parent, the minimum (`MIN`, ascending children) or maximum (`MAX`,
//! `reverseChildren`) of its children's values: `DocValues.getSortedNumeric`
//! or `getSortedSet` of the field, each child's own value picked by the same
//! selector (`SortedNumericSelector`/`SortedSetSelector`), over the children
//! the child filter names. A parent without such a child has no value (the
//! parent missing value); one with a child that has none -- any document
//! between the previous parent and it, in the child filter or not -- takes
//! the child missing value into the selection when one is set
//! (`ToParentDocValues.hasChildWithMissingValue`).
//!
//! # Values
//!
//! A key's [`SortValue`] is the comparable long Lucene's comparator compares:
//! the value for `LONG`/`INT`, `NumericUtils.floatToSortableInt` for `FLOAT`
//! and `doubleToSortableLong` for `DOUBLE` -- what the field stores, as
//! `ToParentBlockJoinSortField` requires -- and the term for `STRING`.
//!
//! # A Lucene quirk kept
//!
//! For `STRING`, a parent with a child missing a value takes the child
//! missing *ordinal* -- `-1`, or `Integer.MAX_VALUE` for `STRING_LAST` --
//! into its selection. `-1` reads as no value; `Integer.MAX_VALUE` is not an
//! ordinal, and Lucene's `TermOrdValComparator.copy` throws
//! `IndexOutOfBoundsException` looking it up: this comparator reports
//! [`Error::IllegalState`] at the same point (a competitive document's value
//! being read), and compares such a parent after every term as Lucene's
//! `compareBottom` does.

use std::cmp::Ordering;
use std::sync::Arc;

use lucene_codecs::doc_values::{self, SortedSetKind};
use lucene_codecs::terms_dict::TermsDict;
use lucene_util::fixed_bit_set::FixedBitSet;

use super::BitSetProducer;
use crate::directory_reader::SegmentReader;
use crate::top_field::{
    FieldComparator, FieldComparatorSource, LeafCtx, LeafFieldComparator, SortField, SortValue,
};
use crate::{Error, Result};

/// `BlockJoinSelector.Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockJoinSelector {
    Min,
    Max,
}

impl BlockJoinSelector {
    fn pick(self, a: i64, b: i64) -> i64 {
        match self {
            BlockJoinSelector::Min => a.min(b),
            BlockJoinSelector::Max => a.max(b),
        }
    }
}

/// `SortField.Type` of a [`ToParentBlockJoinSortField`]: the five it supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinSortType {
    Long,
    Int,
    Float,
    Double,
    String,
}

/// A missing value of a [`ToParentBlockJoinSortField`], typed as its sort
/// type requires (`validateMissingValue`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JoinMissing {
    Long(i64),
    Int(i32),
    Float(f32),
    Double(f64),
    /// `SortField.STRING_FIRST`.
    StringFirst,
    /// `SortField.STRING_LAST`.
    StringLast,
}

impl JoinMissing {
    /// The missing value as the comparable long the comparator reads.
    fn comparable(self) -> Option<i64> {
        match self {
            JoinMissing::Long(v) => Some(v),
            JoinMissing::Int(v) => Some(i64::from(v)),
            JoinMissing::Float(v) => {
                Some(i64::from(lucene_index::document::float_to_sortable_int(v)))
            }
            JoinMissing::Double(v) => Some(lucene_index::document::double_to_sortable_long(v)),
            JoinMissing::StringFirst | JoinMissing::StringLast => None,
        }
    }

    fn fits(self, ty: JoinSortType) -> bool {
        matches!(
            (self, ty),
            (JoinMissing::Long(_), JoinSortType::Long)
                | (JoinMissing::Int(_), JoinSortType::Int)
                | (JoinMissing::Float(_), JoinSortType::Float)
                | (JoinMissing::Double(_), JoinSortType::Double)
                | (
                    JoinMissing::StringFirst | JoinMissing::StringLast,
                    JoinSortType::String
                )
        )
    }
}

/// `ToParentBlockJoinSortField`: a parent sorted by its children's values of
/// `field` -- see the module doc.
#[derive(Clone)]
pub struct ToParentBlockJoinSortField {
    pub field: String,
    pub ty: JoinSortType,
    /// `reverse` of the sort (the parents).
    pub reverse: bool,
    /// Picks the maximum child value when set, the minimum otherwise.
    pub reverse_children: bool,
    pub parent_missing: Option<JoinMissing>,
    pub child_missing: Option<JoinMissing>,
    pub parents: Arc<dyn BitSetProducer>,
    pub children: Arc<dyn BitSetProducer>,
}

impl std::fmt::Debug for ToParentBlockJoinSortField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ToParentBlockJoinSortField({}, {:?}, {}, {}, {:?}, {:?}, {}, {})",
            self.field,
            self.ty,
            self.reverse,
            self.reverse_children,
            self.parent_missing,
            self.child_missing,
            self.parents.key(),
            self.children.key()
        )
    }
}

impl ToParentBlockJoinSortField {
    /// `new ToParentBlockJoinSortField(field, type, reverseParents,
    /// reverseChildren, parentMissingValue, childMissingValue, parentFilter,
    /// childFilter)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a missing value of the wrong type
    /// (`validateMissingValue`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        field: impl Into<String>,
        ty: JoinSortType,
        reverse: bool,
        reverse_children: bool,
        parent_missing: Option<JoinMissing>,
        child_missing: Option<JoinMissing>,
        parents: Arc<dyn BitSetProducer>,
        children: Arc<dyn BitSetProducer>,
    ) -> Result<Self> {
        for m in [parent_missing, child_missing].into_iter().flatten() {
            if !m.fits(ty) {
                return Err(Error::IllegalArgument(format!(
                    "missing value {m:?} does not fit sort type {ty:?}"
                )));
            }
        }
        Ok(Self {
            field: field.into(),
            ty,
            reverse,
            reverse_children,
            parent_missing,
            child_missing,
            parents,
            children,
        })
    }

    /// The sort key (`SortField`) this field sorts by.
    pub fn sort_field(&self) -> SortField {
        SortField::with_source(&self.field, Arc::new(self.clone()), self.reverse)
    }

    fn selector(&self) -> BlockJoinSelector {
        if self.reverse_children {
            BlockJoinSelector::Max
        } else {
            BlockJoinSelector::Min
        }
    }
}

impl FieldComparatorSource for ToParentBlockJoinSortField {
    fn new_comparator(
        &self,
        _field: &str,
        _num_hits: usize,
        _reverse: bool,
    ) -> Box<dyn FieldComparator> {
        Box::new(Comparator { sort: self.clone() })
    }
}

struct Comparator {
    sort: ToParentBlockJoinSortField,
}

impl FieldComparator for Comparator {
    fn leaf<'a>(&self, ctx: LeafCtx<'a>) -> Result<Box<dyn LeafFieldComparator + 'a>> {
        let s = &self.sort;
        let (parents, children) = ctx.reader.with_open_segment(ctx.doc_base, |seg| {
            Ok((s.parents.bit_set(seg)?, s.children.bit_set(seg)?))
        })?;
        let column = match s.ty {
            JoinSortType::String => {
                let missing_ord = if s.child_missing == Some(JoinMissing::StringLast) {
                    i64::from(i32::MAX)
                } else {
                    -1
                };
                Column::Ords(OrdColumn::open(ctx.reader, &s.field)?, missing_ord)
            }
            _ => Column::Longs(
                numeric_column(ctx.reader, &s.field)?,
                s.child_missing.and_then(JoinMissing::comparable),
            ),
        };
        let parent_missing = s
            .parent_missing
            .and_then(JoinMissing::comparable)
            .unwrap_or(0);
        Ok(Box::new(Leaf {
            blocks: Blocks {
                parents,
                children,
                selector: s.selector(),
            },
            column,
            buf: Vec::new(),
            last: None,
            term: None,
            parent_missing,
            missing_last: s.parent_missing == Some(JoinMissing::StringLast),
        }))
    }

    fn compare_values(&self, a: &SortValue, b: &SortValue) -> Ordering {
        match (a, b) {
            (SortValue::Long(a), SortValue::Long(b)) => a.cmp(b),
            (SortValue::Bytes(a), SortValue::Bytes(b)) => {
                // `TermOrdValComparator.compareValues`.
                let missing_last = self.sort.parent_missing == Some(JoinMissing::StringLast);
                match (a, b) {
                    (None, None) => Ordering::Equal,
                    (None, Some(_)) => {
                        if missing_last {
                            Ordering::Greater
                        } else {
                            Ordering::Less
                        }
                    }
                    (Some(_), None) => {
                        if missing_last {
                            Ordering::Less
                        } else {
                            Ordering::Greater
                        }
                    }
                    (Some(a), Some(b)) => a.cmp(b),
                }
            }
            _ => Ordering::Equal,
        }
    }

    fn values_are_bytes(&self) -> bool {
        self.sort.ty == JoinSortType::String
    }
}

/// A segment's parents and children, and the selection over a block.
struct Blocks {
    parents: Option<Arc<FixedBitSet>>,
    children: Option<Arc<FixedBitSet>>,
    selector: BlockJoinSelector,
}

impl Blocks {
    /// `ToParentDocValues.advanceExact(parent)` and the selector's value, for
    /// a value per child: the selection over the parent's children that have
    /// a value, the child missing value taken into it when some document of
    /// the block has none, and `None` when no child has one -- or when `doc`
    /// is not a parent (`advanceExact` returns `false`). A missing child or
    /// parent filter is `DocValues.emptyNumeric()`: no value.
    fn fold(
        &self,
        doc: i32,
        mut child_value: impl FnMut(i32) -> Result<Option<i64>>,
        child_missing: Option<i64>,
    ) -> Result<Option<i64>> {
        let (Some(parents), Some(children)) = (self.parents.as_deref(), self.children.as_deref())
        else {
            return Ok(None);
        };
        let Ok(p) = usize::try_from(doc) else {
            return Ok(None);
        };
        // FBS: `p` checked against the parent set's own length.
        if p >= parents.len() || !parents.get(p) {
            return Ok(None);
        }
        let start = p
            .checked_sub(1)
            .and_then(|q| parents.prev_set_bit(q))
            .map_or(0, |q| q.saturating_add(1));
        let mut acc: Option<i64> = None;
        let mut with_values = 0usize;
        for child in start..p {
            // FBS: `child < p`, checked against the child set's own length.
            if child >= children.len() || !children.get(child) {
                continue;
            }
            let Some(v) = child_value(i32::try_from(child).unwrap_or(i32::MAX))? else {
                continue;
            };
            acc = Some(match acc {
                None => v,
                Some(a) => self.selector.pick(a, v),
            });
            with_values += 1;
        }
        Ok(acc.map(|v| match child_missing {
            Some(m) if with_values < p - start => self.selector.pick(v, m),
            _ => v,
        }))
    }
}

/// The column a segment's parents are sorted by.
enum Column<'a> {
    /// `getSortedNumeric`, with the comparable child missing value.
    Longs(NumericColumn<'a>, Option<i64>),
    /// `getSortedSet`, with the child missing ordinal (`-1`, or
    /// `Integer.MAX_VALUE` for `STRING_LAST`).
    Ords(OrdColumn<'a>, i64),
}

struct Leaf<'a> {
    blocks: Blocks,
    column: Column<'a>,
    buf: Vec<i64>,
    /// The last document folded and its value: a competitive document is
    /// read twice (`compareBottom`, then `copy`).
    last: Option<(i32, Option<i64>)>,
    /// The last term looked up, by ordinal.
    term: Option<(i64, Vec<u8>)>,
    parent_missing: i64,
    missing_last: bool,
}

/// Where a `STRING` value sits among a segment's ordinals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Position {
    First,
    Term,
    Last,
}

impl Leaf<'_> {
    /// The parent's selected value -- for `STRING` its ordinal.
    fn selected(&mut self, doc: i32) -> Result<Option<i64>> {
        if let Some((d, v)) = self.last {
            if d == doc {
                return Ok(v);
            }
        }
        let selector = self.blocks.selector;
        let buf = &mut self.buf;
        let v = match &mut self.column {
            Column::Longs(column, missing) => self.blocks.fold(
                doc,
                |child| child_long(column, child, selector, buf),
                *missing,
            )?,
            Column::Ords(column, missing_ord) => self.blocks.fold(
                doc,
                |child| column.ord(child, selector, buf),
                Some(*missing_ord),
            )?,
        };
        self.last = Some((doc, v));
        Ok(v)
    }

    /// The parent's selected ordinal.
    // SENTINEL: `-1` = "no value" (`getOrdForDoc`'s missing ordinal),
    // outside the ordinals; callers test it (`value`) or map it through
    // `position` (`compare_bottom`).
    fn ord(&mut self, doc: i32) -> Result<i64> {
        Ok(self.selected(doc)?.unwrap_or(-1))
    }

    /// `lookupOrd(ord)`, Lucene's `IndexOutOfBoundsException` for an
    /// ordinal past the terms (the child missing ordinal of `STRING_LAST`).
    fn term(&mut self, ord: i64) -> Result<&[u8]> {
        let Column::Ords(column, _) = &mut self.column else {
            return Ok(&[]);
        };
        if self.term.as_ref().is_none_or(|(o, _)| *o != ord) {
            let size = column.size();
            if ord < 0 || ord >= size {
                return Err(Error::IllegalState(format!(
                    "Index {ord} out of bounds for length {size}"
                )));
            }
            let t = column.lookup(ord)?.to_vec();
            self.term = Some((ord, t));
        }
        Ok(self.term.as_ref().map_or(&[][..], |(_, t)| t.as_slice()))
    }

    /// `getOrdForDoc` with `missingOrd` substituted, as `compareBottom` reads it.
    fn position(&self, ord: i64) -> Position {
        if ord == -1 {
            if self.missing_last {
                Position::Last
            } else {
                Position::First
            }
        } else if ord >= i64::from(i32::MAX) {
            Position::Last
        } else {
            Position::Term
        }
    }
}

impl LeafFieldComparator for Leaf<'_> {
    fn value(&mut self, doc: i32, _score: f32) -> Result<SortValue> {
        if matches!(self.column, Column::Longs(..)) {
            return Ok(SortValue::Long(
                self.selected(doc)?.unwrap_or(self.parent_missing),
            ));
        }
        let ord = self.ord(doc)?;
        if ord == -1 {
            return Ok(SortValue::Bytes(None));
        }
        Ok(SortValue::Bytes(Some(self.term(ord)?.to_vec())))
    }

    /// `TermOrdValComparator.compareBottom` for `STRING`: by position among
    /// the ordinals, then by term -- without reading the term of the child
    /// missing ordinal, which only a copy does.
    fn compare_bottom(
        &mut self,
        bottom: &SortValue,
        doc: i32,
        _score: f32,
    ) -> Result<Option<Ordering>> {
        if !matches!(self.column, Column::Ords(..)) {
            return Ok(None);
        }
        let SortValue::Bytes(bottom) = bottom else {
            return Ok(None);
        };
        // SENTINEL-OK: `position` maps `-1` to the missing position.
        let ord = self.ord(doc)?;
        let doc_pos = self.position(ord);
        let bottom_pos = match bottom {
            None => self.position(-1),
            Some(_) => Position::Term,
        };
        if doc_pos != bottom_pos || doc_pos != Position::Term {
            return Ok(Some(bottom_pos.cmp(&doc_pos)));
        }
        let Some(b) = bottom.as_deref() else {
            return Ok(Some(Ordering::Equal));
        };
        Ok(Some(b.cmp(self.term(ord)?)))
    }
}

/// A segment's column of `field` as `DocValues.getSortedNumeric` reads it:
/// each document's values, ascending.
enum NumericColumn<'a> {
    Absent,
    Single(doc_values::NumericReader<'a>),
    Multi(doc_values::SortedNumericReader<'a>),
}

fn numeric_column<'a>(reader: &'a SegmentReader, field: &str) -> Result<NumericColumn<'a>> {
    let Some(info) = reader.field_infos().field_by_name(field) else {
        return Ok(NumericColumn::Absent);
    };
    let Some((meta, data)) = reader.doc_values_for_field(info.number) else {
        return Ok(NumericColumn::Absent);
    };
    if let Some(e) = meta.sorted_numeric_entry(info.number) {
        return Ok(NumericColumn::Multi(doc_values::SortedNumericReader::new(
            data, e,
        )));
    }
    if let Some(e) = meta.numeric_entry(info.number) {
        return Ok(NumericColumn::Single(doc_values::NumericReader::new(
            data, e,
        )));
    }
    if info.doc_values_type == lucene_codecs::field_infos::DocValuesType::None {
        return Ok(NumericColumn::Absent);
    }
    Err(Error::IllegalState(format!(
        "unexpected docvalues type {:?} for field '{field}' (expected one of [SORTED_NUMERIC, \
         NUMERIC]). Re-index with correct docvalues type.",
        info.doc_values_type
    )))
}

/// Each child's value picked by `selector`, `None` without one.
fn child_long(
    column: &mut NumericColumn<'_>,
    doc: i32,
    selector: BlockJoinSelector,
    buf: &mut Vec<i64>,
) -> Result<Option<i64>> {
    Ok(match column {
        NumericColumn::Absent => None,
        NumericColumn::Single(r) => r.value(doc)?,
        NumericColumn::Multi(r) => {
            buf.clear();
            r.values(doc, buf)?;
            match selector {
                BlockJoinSelector::Min => buf.first().copied(),
                BlockJoinSelector::Max => buf.last().copied(),
            }
        }
    })
}

/// A segment's column of `field` as `DocValues.getSortedSet` reads it, with
/// `SortedSetSelector`'s pick and the terms dictionary for `lookupOrd`.
enum OrdColumn<'a> {
    Absent,
    Single(&'a [u8], &'a doc_values::SortedEntry, TermsDict<'a>),
    Multi(&'a [u8], &'a doc_values::SortedNumericEntry, TermsDict<'a>),
}

impl<'a> OrdColumn<'a> {
    fn open(reader: &'a SegmentReader, field: &str) -> Result<Self> {
        let Some(info) = reader.field_infos().field_by_name(field) else {
            return Ok(OrdColumn::Absent);
        };
        let Some((meta, data)) = reader.doc_values_for_field(info.number) else {
            return Ok(OrdColumn::Absent);
        };
        let store = |e| Error::from(lucene_codecs::blocktree::Error::Store(e));
        let dict = |entry| TermsDict::open(data, entry).map_err(store);
        if let Some(e) = meta.sorted_set_entry(info.number) {
            return Ok(match &e.kind {
                SortedSetKind::Single(s) => OrdColumn::Single(data, s, dict(&s.terms)?),
                SortedSetKind::Multi { ords, terms } => OrdColumn::Multi(data, ords, dict(terms)?),
            });
        }
        if let Some(s) = meta.sorted_entry(info.number) {
            return Ok(OrdColumn::Single(data, s, dict(&s.terms)?));
        }
        if info.doc_values_type == lucene_codecs::field_infos::DocValuesType::None {
            return Ok(OrdColumn::Absent);
        }
        Err(Error::IllegalState(format!(
            "unexpected docvalues type {:?} for field '{field}' (expected one of [SORTED, \
             SORTED_SET]). Re-index with correct docvalues type.",
            info.doc_values_type
        )))
    }

    /// The child's ordinal picked by `selector`, `None` without one.
    fn ord(
        &mut self,
        doc: i32,
        selector: BlockJoinSelector,
        buf: &mut Vec<i64>,
    ) -> Result<Option<i64>> {
        Ok(match self {
            OrdColumn::Absent => None,
            OrdColumn::Single(data, s, _) => doc_values::sorted_ord(data, s, doc)?,
            OrdColumn::Multi(data, m, _) => {
                *buf = doc_values::sorted_numeric_values(data, m, doc)?;
                match selector {
                    BlockJoinSelector::Min => buf.first().copied(),
                    BlockJoinSelector::Max => buf.last().copied(),
                }
            }
        })
    }

    /// `getValueCount()`.
    fn size(&self) -> i64 {
        match self {
            OrdColumn::Absent => 0,
            OrdColumn::Single(_, _, d) | OrdColumn::Multi(_, _, d) => d.size(),
        }
    }

    /// `lookupOrd(ord)` for `0 <= ord < size()`.
    fn lookup(&mut self, ord: i64) -> Result<&[u8]> {
        let store = |e| Error::from(lucene_codecs::blocktree::Error::Store(e));
        match self {
            OrdColumn::Absent => Ok(&[]),
            OrdColumn::Single(_, _, d) | OrdColumn::Multi(_, _, d) => {
                d.seek_ord(ord).map_err(store)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(len: usize, set: &[usize]) -> Arc<FixedBitSet> {
        let mut b = FixedBitSet::new(len);
        for &i in set {
            b.set(i);
        }
        Arc::new(b)
    }

    /// Blocks {0, 1 | 2}, {3 | 4} (3 not a child), {| 5}: parents 2, 4, 5.
    fn blocks(selector: BlockJoinSelector) -> Blocks {
        Blocks {
            parents: Some(bits(6, &[2, 4, 5])),
            children: Some(bits(6, &[0, 1])),
            selector,
        }
    }

    #[test]
    fn a_block_folds_like_to_parent_doc_values() {
        let value = |doc: i32| Ok(Some(i64::from(10 - doc)));
        let min = blocks(BlockJoinSelector::Min);
        let max = blocks(BlockJoinSelector::Max);
        assert_eq!(min.fold(2, value, None).unwrap(), Some(9));
        assert_eq!(max.fold(2, value, None).unwrap(), Some(10));
        // A child missing value joins the selection only when some document
        // of the block has no value -- doc 3 is not even a child.
        assert_eq!(min.fold(2, value, Some(-1)).unwrap(), Some(9));
        assert_eq!(min.fold(4, value, Some(-1)).unwrap(), None);
        let some = |doc: i32| Ok((doc == 1).then_some(7));
        assert_eq!(min.fold(2, some, Some(3)).unwrap(), Some(3));
        assert_eq!(max.fold(2, some, Some(3)).unwrap(), Some(7));
        // No children, not a parent, outside the set: no value.
        for doc in [5, 1, -1, 6, 99] {
            assert_eq!(min.fold(doc, value, Some(0)).unwrap(), None, "{doc}");
        }
        // Either filter missing is `emptyNumeric()`.
        let none = Blocks {
            parents: None,
            children: Some(bits(6, &[0])),
            selector: BlockJoinSelector::Min,
        };
        assert_eq!(none.fold(2, value, None).unwrap(), None);
        // A child set shorter than the block reads as no child there.
        let short = Blocks {
            parents: Some(bits(6, &[2])),
            children: Some(bits(1, &[0])),
            selector: BlockJoinSelector::Max,
        };
        assert_eq!(short.fold(2, value, None).unwrap(), Some(10));
        // A failing read is the sort's error.
        let broken = |_: i32| Err(Error::IllegalState("read".into()));
        assert!(min.fold(2, broken, None).is_err());
    }

    fn leaf(column: Column<'static>) -> Leaf<'static> {
        Leaf {
            blocks: blocks(BlockJoinSelector::Min),
            column,
            buf: Vec::new(),
            last: None,
            term: None,
            parent_missing: 42,
            missing_last: false,
        }
    }

    #[test]
    fn a_field_without_values_sorts_every_parent_as_missing() {
        let mut longs = leaf(Column::Longs(NumericColumn::Absent, Some(1)));
        assert_eq!(longs.value(2, 0.0).unwrap(), SortValue::Long(42));
        // Read twice: the second answer comes from the memo.
        assert_eq!(longs.value(2, 0.0).unwrap(), SortValue::Long(42));
        assert_eq!(
            longs.compare_bottom(&SortValue::Long(0), 2, 0.0).unwrap(),
            None
        );
        assert_eq!(longs.term(0).unwrap(), b"");

        let mut ords = leaf(Column::Ords(OrdColumn::Absent, -1));
        assert_eq!(ords.value(2, 0.0).unwrap(), SortValue::Bytes(None));
        assert_eq!(
            ords.compare_bottom(&SortValue::Long(0), 2, 0.0).unwrap(),
            None
        );
        // Missing sorts first (no `STRING_LAST`): against a term bottom the
        // document wins, against a missing bottom it ties.
        assert_eq!(
            ords.compare_bottom(&SortValue::Bytes(Some(b"a".to_vec())), 2, 0.0)
                .unwrap(),
            Some(Ordering::Greater)
        );
        assert_eq!(
            ords.compare_bottom(&SortValue::Bytes(None), 4, 0.0)
                .unwrap(),
            Some(Ordering::Equal)
        );
        // No child has a value, so the child missing ordinal never joins a
        // selection (`advanceExact` is `false`); looked up, `STRING_LAST`'s
        // is past every term.
        let mut last = leaf(Column::Ords(OrdColumn::Absent, i64::from(i32::MAX)));
        assert_eq!(last.value(2, 0.0).unwrap(), SortValue::Bytes(None));
        let e = last.term(i64::from(i32::MAX)).unwrap_err();
        assert!(e.to_string().contains("out of bounds for length 0"), "{e}");
        assert!(last.term(-1).is_err());
        let mut absent = OrdColumn::Absent;
        assert_eq!(absent.size(), 0);
        assert_eq!(absent.lookup(0).unwrap(), b"");
        assert_eq!(
            absent
                .ord(0, BlockJoinSelector::Min, &mut Vec::new())
                .unwrap(),
            None
        );
    }
}
