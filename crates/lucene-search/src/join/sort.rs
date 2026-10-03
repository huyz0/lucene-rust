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
        let values = match s.ty {
            JoinSortType::String => {
                let missing_ord = if s.child_missing == Some(JoinMissing::StringLast) {
                    i64::from(i32::MAX)
                } else {
                    -1
                };
                let (ords, terms) = parent_ords(
                    ctx.reader,
                    &s.field,
                    s.selector(),
                    parents.as_deref(),
                    children.as_deref(),
                    missing_ord,
                )?;
                Values::Ords(ords, terms)
            }
            _ => Values::Longs(parent_longs(
                ctx.reader,
                &s.field,
                s.selector(),
                parents.as_deref(),
                children.as_deref(),
                s.child_missing.and_then(JoinMissing::comparable),
            )?),
        };
        let parent_missing = s
            .parent_missing
            .and_then(JoinMissing::comparable)
            .unwrap_or(0);
        Ok(Box::new(Leaf {
            values,
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

/// One segment's parent values.
enum Values {
    /// Per document, the comparable long, `None` without one.
    Longs(Vec<Option<i64>>),
    /// Per document, the selected ordinal (`-1` none; `i32::MAX` the child
    /// missing ordinal), and the segment's terms by ordinal.
    Ords(Vec<i64>, Vec<Vec<u8>>),
}

struct Leaf {
    values: Values,
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

impl Leaf {
    /// The parent's selected ordinal.
    // SENTINEL: `-1` = "no value" (`getOrdForDoc`'s missing ordinal),
    // outside the ordinals; callers test it (`value`) or map it through
    // `position` (`compare_bottom`).
    fn ord(&self, doc: i32) -> i64 {
        match &self.values {
            Values::Ords(ords, _) => usize::try_from(doc)
                .ok()
                .and_then(|d| ords.get(d))
                .copied()
                .unwrap_or(-1),
            Values::Longs(_) => -1,
        }
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

impl LeafFieldComparator for Leaf {
    fn value(&mut self, doc: i32, _score: f32) -> Result<SortValue> {
        match &self.values {
            Values::Longs(v) => Ok(SortValue::Long(
                usize::try_from(doc)
                    .ok()
                    .and_then(|d| v.get(d).copied().flatten())
                    .unwrap_or(self.parent_missing),
            )),
            Values::Ords(_, terms) => {
                let ord = self.ord(doc);
                if ord == -1 {
                    return Ok(SortValue::Bytes(None));
                }
                usize::try_from(ord)
                    .ok()
                    .and_then(|o| terms.get(o))
                    .map(|t| SortValue::Bytes(Some(t.clone())))
                    .ok_or_else(|| {
                        Error::IllegalState(format!(
                            "Index {ord} out of bounds for length {}",
                            terms.len()
                        ))
                    })
            }
        }
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
        let Values::Ords(_, terms) = &self.values else {
            return Ok(None);
        };
        let SortValue::Bytes(bottom) = bottom else {
            return Ok(None);
        };
        // SENTINEL-OK: `position` maps `-1` to the missing position.
        let ord = self.ord(doc);
        let doc_pos = self.position(ord);
        let bottom_pos = match bottom {
            None => self.position(-1),
            Some(_) => Position::Term,
        };
        if doc_pos != bottom_pos || doc_pos != Position::Term {
            return Ok(Some(bottom_pos.cmp(&doc_pos)));
        }
        let (Some(b), Some(t)) = (
            bottom.as_deref(),
            usize::try_from(ord).ok().and_then(|o| terms.get(o)),
        ) else {
            return Ok(Some(Ordering::Equal));
        };
        Ok(Some(b.cmp(t.as_slice())))
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

/// `ToParentDocValues.advanceExact(parent)` over every parent of a segment,
/// for a value per child: the selection over the parent's children that
/// have a value, whether some document of the block has none, and the
/// parent's value (`None` when no child has one).
fn fold_parents(
    max_doc: i32,
    parents: Option<&FixedBitSet>,
    children: Option<&FixedBitSet>,
    selector: BlockJoinSelector,
    mut child_value: impl FnMut(i32) -> Result<Option<i64>>,
    child_missing: Option<i64>,
) -> Result<Vec<Option<i64>>> {
    let len = usize::try_from(max_doc).unwrap_or(0);
    let mut out = vec![None; len];
    // A missing child filter is `DocValues.emptyNumeric()`: no parent has a
    // value; so is a missing parent filter.
    let (Some(parents), Some(children)) = (parents, children) else {
        return Ok(out);
    };
    let mut prev_parent: i64 = -1;
    let mut parent = parents.next_set_bit(0);
    while let Some(p) = parent {
        if p >= len {
            break;
        }
        let start = usize::try_from(prev_parent.saturating_add(1)).unwrap_or(0);
        let mut acc: Option<i64> = None;
        let mut with_values = 0usize;
        for child in start..p {
            // FBS: `child < p < len`, checked against the child set's own length.
            if child >= children.len() || !children.get(child) {
                continue;
            }
            let Some(v) = child_value(i32::try_from(child).unwrap_or(i32::MAX))? else {
                continue;
            };
            acc = Some(match acc {
                None => v,
                Some(a) => selector.pick(a, v),
            });
            with_values += 1;
        }
        if let Some(mut v) = acc {
            let total = p - start;
            if let Some(m) = child_missing {
                if with_values < total {
                    v = selector.pick(v, m);
                }
            }
            out[p] = Some(v);
        }
        prev_parent = i64::try_from(p).unwrap_or(i64::MAX);
        parent = p.checked_add(1).and_then(|n| parents.next_set_bit(n));
    }
    Ok(out)
}

/// `BlockJoinSelector.wrap(sortedNumeric, selection, parents, children,
/// childMissingValue)` read for every parent of the segment.
fn parent_longs(
    reader: &SegmentReader,
    field: &str,
    selector: BlockJoinSelector,
    parents: Option<&FixedBitSet>,
    children: Option<&FixedBitSet>,
    child_missing: Option<i64>,
) -> Result<Vec<Option<i64>>> {
    let mut column = numeric_column(reader, field)?;
    let mut buf = Vec::new();
    fold_parents(
        reader.max_doc,
        parents,
        children,
        selector,
        |doc| child_long(&mut column, doc, selector, &mut buf),
        child_missing,
    )
}

/// `BlockJoinSelector.wrap(sortedSet, selection, parents, children,
/// sortMissingLast)` read for every parent of the segment: the selected
/// ordinal (`-1` without one; the child missing ordinal taken into the
/// selection whenever a document of the block has no value, which is `-1`
/// too unless `STRING_LAST`), and the segment's terms.
fn parent_ords(
    reader: &SegmentReader,
    field: &str,
    selector: BlockJoinSelector,
    parents: Option<&FixedBitSet>,
    children: Option<&FixedBitSet>,
    missing_ord: i64,
) -> Result<(Vec<i64>, Vec<Vec<u8>>)> {
    let len = usize::try_from(reader.max_doc).unwrap_or(0);
    let column = match reader.field_infos().field_by_name(field) {
        Some(info) => reader
            .doc_values_for_field(info.number)
            .map(|(meta, data)| (info.number, info.doc_values_type, meta, data)),
        None => None,
    };
    let Some((number, ty, meta, data)) = column else {
        return Ok((vec![-1; len], Vec::new()));
    };
    let store = |e| Error::from(lucene_codecs::blocktree::Error::Store(e));
    enum Ords<'e> {
        Single(&'e doc_values::SortedEntry),
        Multi(&'e doc_values::SortedNumericEntry),
    }
    let (ords, terms_entry) = if let Some(e) = meta.sorted_set_entry(number) {
        match &e.kind {
            SortedSetKind::Single(s) => (Ords::Single(s), &s.terms),
            SortedSetKind::Multi { ords, terms } => (Ords::Multi(ords), terms),
        }
    } else if let Some(s) = meta.sorted_entry(number) {
        (Ords::Single(s), &s.terms)
    } else if ty == lucene_codecs::field_infos::DocValuesType::None {
        return Ok((vec![-1; len], Vec::new()));
    } else {
        return Err(Error::IllegalState(format!(
            "unexpected docvalues type {ty:?} for field '{field}' (expected one of [SORTED, \
             SORTED_SET]). Re-index with correct docvalues type."
        )));
    };
    let terms = lucene_codecs::terms_dict::decode_all_terms(data, terms_entry).map_err(store)?;
    let selected = fold_parents(
        reader.max_doc,
        parents,
        children,
        selector,
        |doc| -> Result<Option<i64>> {
            Ok(match &ords {
                Ords::Single(s) => doc_values::sorted_ord(data, s, doc)?,
                Ords::Multi(m) => {
                    let values = doc_values::sorted_numeric_values(data, m, doc)?;
                    match selector {
                        BlockJoinSelector::Min => values.first().copied(),
                        BlockJoinSelector::Max => values.last().copied(),
                    }
                }
            })
        },
        Some(missing_ord),
    )?;
    Ok((
        selected.into_iter().map(|o| o.unwrap_or(-1)).collect(),
        terms,
    ))
}
