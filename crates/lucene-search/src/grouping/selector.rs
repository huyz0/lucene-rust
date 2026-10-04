//! The group selectors: `TermGroupSelector` (a `SORTED` field's term),
//! `LongRangeGroupSelector`/`DoubleRangeGroupSelector` (the range a values
//! source's value falls in, `LongRangeFactory`/`DoubleRangeFactory`) and
//! `ValueSourceGroupSelector` (a function value source's value).

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use super::{FxHashMap, GroupSelector, GroupState, SearchGroup};
use crate::function::{BoxValues, FunctionContext, MutableValue, TopLevel, ValueLeaf, ValueSource};
use crate::multi_segment::OpenSegment;
use crate::reader::doc_values as dv;
use crate::reader::SortedDocValues;
use crate::values_source::{
    BoxDoubleValues, BoxLongValues, DoubleValues, DoubleValuesSource, LongValuesSource,
    ValuesContext,
};
use crate::{Error, Result};

/// A range selector moved before `setScorer` gave it values (Java's
/// `NullPointerException` on `values`).
fn no_values() -> Error {
    Error::IllegalState("the group selector has no values: setScorer was never called".into())
}

/// `setScorer` before `setNextReader` (Java's `NullPointerException` on the
/// context).
fn no_context() -> Error {
    Error::IllegalState("the group selector was given a scorer before a segment".into())
}

fn leaf_reader<'a>(leaf: &OpenSegment<'a>) -> Result<&'a crate::directory_reader::SegmentReader> {
    leaf.reader
        .ok_or_else(|| Error::MissingSegmentReader("a group selector".into()))
}

/// `TermGroupSelector`: groups by a `SORTED` field's term. Terms get ids in
/// the order first seen (`BytesRefHash`), and each segment maps its
/// ordinals to them as it goes (`ordsToGroupIds`).
///
/// Stage 3, same groups: the segment's ordinal map is a table indexed by
/// ordinal (Java's `IntIntHashMap`), filled in the first pass as ordinals
/// show up (or, once many have, by one forward walk of the dictionary)
/// rather than by seeking every known term per segment.
pub struct TermGroupSelector<'a> {
    field: String,
    values: Vec<Vec<u8>>,
    ids: FxHashMap<Vec<u8>, usize>,
    ords_to_group_ids: OrdTable,
    doc_values: Option<Box<dyn SortedDocValues + 'a>>,
    group_id: Option<usize>,
    second_pass: bool,
    include_empty: bool,
    /// This segment's documents whose ordinal was not mapped yet, and
    /// whether the dictionary was walked for the known terms.
    misses: i32,
    walked: bool,
}

/// A segment's ordinals to group ids: a table, or a map for a dictionary
/// too large to tabulate.
#[derive(Debug, Default)]
struct OrdTable {
    table: Vec<u32>,
    map: FxHashMap<i32, usize>,
    tabulated: bool,
}

impl OrdTable {
    const NONE: u32 = u32::MAX;
    /// The most ordinals a table holds.
    const MAX: i32 = 1 << 22;

    fn reset(&mut self, value_count: i32) {
        self.map.clear();
        self.table.clear();
        self.tabulated = (0..=Self::MAX).contains(&value_count);
        if self.tabulated {
            self.table
                .resize(usize::try_from(value_count).unwrap_or(0), Self::NONE);
        }
    }

    fn get(&self, ord: i32) -> Option<usize> {
        if self.tabulated {
            if let Some(&id) = usize::try_from(ord).ok().and_then(|o| self.table.get(o)) {
                return (id != Self::NONE).then(|| usize::try_from(id).ok())?;
            }
        }
        self.map.get(&ord).copied()
    }

    fn insert(&mut self, ord: i32, id: usize) {
        if self.tabulated {
            let slot = usize::try_from(ord)
                .ok()
                .and_then(|o| self.table.get_mut(o));
            if let (Some(slot), Ok(id)) = (slot, u32::try_from(id)) {
                *slot = id;
                return;
            }
        }
        self.map.insert(ord, id);
    }
}

impl<'a> TermGroupSelector<'a> {
    /// `new TermGroupSelector(field)`.
    pub fn new(field: &str) -> Self {
        Self {
            field: field.to_string(),
            values: Vec::new(),
            ids: FxHashMap::default(),
            ords_to_group_ids: OrdTable::default(),
            doc_values: None,
            group_id: None,
            second_pass: false,
            include_empty: false,
            misses: 0,
            walked: false,
        }
    }
}

impl<'a> GroupSelector<'a> for TermGroupSelector<'a> {
    type Value = Vec<u8>;

    fn set_next_reader(&mut self, _ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        let mut values = dv::get_sorted(leaf_reader(leaf)?, &self.field)?;
        let count = values.value_count();
        self.ords_to_group_ids.reset(count);
        self.misses = 0;
        self.walked = false;
        // The second pass maps its few groups' terms up front, as Java
        // does; the first pass maps an ordinal when a document first shows
        // it (its term looked up among the known), where Java seeks every
        // known term up front -- the same ids either way, and the first
        // pass reaches few documents once its top groups fill.
        if self.second_pass {
            for (i, v) in self.values.iter().enumerate() {
                let ord = values.lookup_term(v)?;
                if ord >= 0 {
                    self.ords_to_group_ids.insert(ord, i);
                }
            }
        }
        self.doc_values = Some(values);
        Ok(())
    }

    fn advance_to(&mut self, doc: i32, _score: f32) -> Result<GroupState> {
        let Some(values) = self.doc_values.as_mut() else {
            return Ok(GroupState::Skip);
        };
        if !values.advance_exact(doc)? {
            self.group_id = None;
            return Ok(if self.include_empty {
                GroupState::Accept
            } else {
                GroupState::Skip
            });
        }
        let ord = values.ord_value();
        if let Some(id) = self.ords_to_group_ids.get(ord) {
            self.group_id = Some(id);
            return Ok(GroupState::Accept);
        }
        if self.second_pass {
            return Ok(GroupState::Skip);
        }
        // Many documents with unmapped ordinals (a collector that sees every
        // document, `AllGroupsCollector`): map every known term at once by
        // walking the dictionary forward, rather than one random lookup
        // each.
        self.misses = self.misses.saturating_add(1);
        let count = values.value_count();
        let known = i32::try_from(self.values.len()).unwrap_or(i32::MAX);
        if !self.walked && self.misses > 32 && known.saturating_mul(8) >= count {
            self.walked = true;
            for o in 0..count {
                let term = values.lookup_ord(o)?;
                if let Some(&id) = self.ids.get(&term) {
                    self.ords_to_group_ids.insert(o, id);
                }
            }
            if let Some(id) = self.ords_to_group_ids.get(ord) {
                self.group_id = Some(id);
                return Ok(GroupState::Accept);
            }
        }
        let term = values.lookup_ord(ord)?;
        let id = match self.ids.get(&term) {
            Some(&id) => id,
            None => {
                let id = self.values.len();
                self.ids.insert(term.clone(), id);
                self.values.push(term);
                id
            }
        };
        self.group_id = Some(id);
        self.ords_to_group_ids.insert(ord, id);
        Ok(GroupState::Accept)
    }

    fn current_value(&self) -> Option<&Vec<u8>> {
        self.group_id.and_then(|i| self.values.get(i))
    }

    fn set_groups(&mut self, groups: &[SearchGroup<Vec<u8>>]) {
        self.values.clear();
        self.ids.clear();
        for g in groups {
            match &g.group_value {
                None => self.include_empty = true,
                Some(v) => {
                    if !self.ids.contains_key(v) {
                        self.ids.insert(v.clone(), self.values.len());
                        self.values.push(v.clone());
                    }
                }
            }
        }
        self.second_pass = true;
    }
}

/// `LongRange`: `[min, max)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LongRange {
    pub min: i64,
    pub max: i64,
}

/// `LongRangeFactory(min, width, max)`: buckets of `width` from `min` to
/// `max`, with everything below `min` and from `max` up in two open-ended
/// ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LongRangeFactory {
    pub min: i64,
    pub width: i64,
    pub max: i64,
}

impl LongRangeFactory {
    /// `getRange(value, reuse)`, in Java's `long` arithmetic.
    ///
    /// # Errors
    /// A `width` of `0` for a value inside `[min, max)`: Java's
    /// `ArithmeticException` (`/ by zero`), as [`Error::IllegalArgument`].
    pub fn get_range(&self, value: i64) -> Result<LongRange> {
        if value < self.min {
            return Ok(LongRange {
                min: i64::MIN,
                max: self.min,
            });
        }
        if value >= self.max {
            return Ok(LongRange {
                min: self.max,
                max: i64::MAX,
            });
        }
        if self.width == 0 {
            return Err(Error::IllegalArgument("/ by zero".into()));
        }
        let bucket = value.wrapping_sub(self.min).wrapping_div(self.width);
        let min = self.min.wrapping_add(bucket.wrapping_mul(self.width));
        Ok(LongRange {
            min,
            max: min.wrapping_add(self.width),
        })
    }
}

/// `DoubleRange`: `[min, max)`, equal as `Double.compare` says (one `NaN`,
/// `-0.0` apart from `0.0`).
#[derive(Debug, Clone, Copy)]
pub struct DoubleRange {
    pub min: f64,
    pub max: f64,
}

/// `Double.doubleToLongBits`: one `NaN`.
fn double_bits(d: f64) -> u64 {
    if d.is_nan() {
        0x7ff8_0000_0000_0000
    } else {
        d.to_bits()
    }
}

impl PartialEq for DoubleRange {
    fn eq(&self, o: &Self) -> bool {
        double_bits(self.min) == double_bits(o.min) && double_bits(self.max) == double_bits(o.max)
    }
}

impl Eq for DoubleRange {}

impl Hash for DoubleRange {
    fn hash<H: Hasher>(&self, state: &mut H) {
        double_bits(self.min).hash(state);
        double_bits(self.max).hash(state);
    }
}

/// `DoubleRangeFactory(min, width, max)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoubleRangeFactory {
    pub min: f64,
    pub width: f64,
    pub max: f64,
}

impl DoubleRangeFactory {
    /// `getRange(value, reuse)`. Below `min` the range is
    /// `[Double.MIN_VALUE, min)` -- the smallest *positive* double, as Java
    /// writes it.
    pub fn get_range(&self, value: f64) -> DoubleRange {
        if value < self.min {
            return DoubleRange {
                min: f64::from_bits(1),
                max: self.min,
            };
        }
        if value >= self.max {
            return DoubleRange {
                min: self.max,
                max: f64::MAX,
            };
        }
        let bucket = ((value - self.min) / self.width).floor();
        let min = self.min + bucket * self.width;
        DoubleRange {
            min,
            max: min + self.width,
        }
    }
}

/// `DoubleValuesSource.fromScorer(scorer)`: the score of the document being
/// collected, which the collector sets before each `advance_to`.
#[derive(Debug, Clone, Default)]
struct CurrentScore(Arc<AtomicU32>);

impl CurrentScore {
    fn set(&self, score: f32) {
        self.0.store(score.to_bits(), Ordering::Relaxed);
    }
}

impl DoubleValues for CurrentScore {
    fn advance_exact(&mut self, _doc: i32) -> Result<bool> {
        Ok(true)
    }
    fn double_value(&mut self) -> Result<f64> {
        Ok(f64::from(f32::from_bits(self.0.load(Ordering::Relaxed))))
    }
}

/// The selector shared by both range kinds.
struct RangeState<R> {
    in_second_pass: Option<HashSet<R>>,
    include_empty: bool,
    current: Option<R>,
}

impl<R: Clone + Eq + Hash> RangeState<R> {
    fn new() -> Self {
        Self {
            in_second_pass: None,
            include_empty: false,
            current: None,
        }
    }

    fn position(&mut self, range: Option<R>) -> GroupState {
        let Some(range) = range else {
            self.current = None;
            return if self.include_empty {
                GroupState::Accept
            } else {
                GroupState::Skip
            };
        };
        let state = match &self.in_second_pass {
            None => GroupState::Accept,
            Some(set) if set.contains(&range) => GroupState::Accept,
            Some(_) => GroupState::Skip,
        };
        self.current = Some(range);
        state
    }

    fn set_groups(&mut self, groups: &[SearchGroup<R>]) {
        let mut set = HashSet::new();
        for g in groups {
            match &g.group_value {
                None => self.include_empty = true,
                Some(r) => {
                    set.insert(r.clone());
                }
            }
        }
        self.in_second_pass = Some(set);
    }
}

/// `LongRangeGroupSelector`: groups by the [`LongRange`] a
/// [`LongValuesSource`]'s value falls in; a document without a value is in
/// the `None` group.
pub struct LongRangeGroupSelector<'a> {
    source: Arc<dyn LongValuesSource>,
    factory: LongRangeFactory,
    reader: Option<&'a crate::directory_reader::SegmentReader>,
    values: Option<BoxLongValues<'a>>,
    score: CurrentScore,
    state: RangeState<LongRange>,
}

impl<'a> LongRangeGroupSelector<'a> {
    /// `new LongRangeGroupSelector(source, rangeFactory)`.
    pub fn new(source: Arc<dyn LongValuesSource>, factory: LongRangeFactory) -> Self {
        Self {
            source,
            factory,
            reader: None,
            values: None,
            score: CurrentScore::default(),
            state: RangeState::new(),
        }
    }
}

impl<'a> GroupSelector<'a> for LongRangeGroupSelector<'a> {
    type Value = LongRange;

    /// `setNextReader`: remembers the segment (`this.context = ...`).
    fn set_next_reader(&mut self, _ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        self.reader = Some(leaf_reader(leaf)?);
        Ok(())
    }

    /// `setScorer`: the source's values for the segment, the score read
    /// from the document being collected.
    fn set_scorer(&mut self) -> Result<()> {
        let reader = self.reader.ok_or_else(no_context)?;
        let ctx = ValuesContext::for_reader(reader);
        let scores: BoxDoubleValues<'a> = Box::new(self.score.clone());
        self.values = Some(self.source.get_values(&ctx, 0, Some(scores))?);
        Ok(())
    }

    fn advance_to(&mut self, doc: i32, score: f32) -> Result<GroupState> {
        self.score.set(score);
        let values = self.values.as_mut().ok_or_else(no_values)?;
        let range = if values.advance_exact(doc)? {
            Some(self.factory.get_range(values.long_value()?)?)
        } else {
            None
        };
        Ok(self.state.position(range))
    }

    fn current_value(&self) -> Option<&LongRange> {
        self.state.current.as_ref()
    }

    fn set_groups(&mut self, groups: &[SearchGroup<LongRange>]) {
        self.state.set_groups(groups);
    }
}

/// `DoubleRangeGroupSelector`: groups by the [`DoubleRange`] a
/// [`DoubleValuesSource`]'s value falls in.
pub struct DoubleRangeGroupSelector<'a> {
    source: Arc<dyn DoubleValuesSource>,
    factory: DoubleRangeFactory,
    reader: Option<&'a crate::directory_reader::SegmentReader>,
    values: Option<BoxDoubleValues<'a>>,
    score: CurrentScore,
    state: RangeState<DoubleRange>,
}

impl<'a> DoubleRangeGroupSelector<'a> {
    /// `new DoubleRangeGroupSelector(source, rangeFactory)`.
    pub fn new(source: Arc<dyn DoubleValuesSource>, factory: DoubleRangeFactory) -> Self {
        Self {
            source,
            factory,
            reader: None,
            values: None,
            score: CurrentScore::default(),
            state: RangeState::new(),
        }
    }
}

impl<'a> GroupSelector<'a> for DoubleRangeGroupSelector<'a> {
    type Value = DoubleRange;

    /// `setNextReader`: remembers the segment (`this.context = ...`).
    fn set_next_reader(&mut self, _ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        self.reader = Some(leaf_reader(leaf)?);
        Ok(())
    }

    /// `setScorer`: the source's values for the segment, the score read
    /// from the document being collected.
    fn set_scorer(&mut self) -> Result<()> {
        let reader = self.reader.ok_or_else(no_context)?;
        let ctx = ValuesContext::for_reader(reader);
        let scores: BoxDoubleValues<'a> = Box::new(self.score.clone());
        self.values = Some(self.source.get_values(&ctx, 0, Some(scores))?);
        Ok(())
    }

    fn advance_to(&mut self, doc: i32, score: f32) -> Result<GroupState> {
        self.score.set(score);
        let values = self.values.as_mut().ok_or_else(no_values)?;
        let range = if values.advance_exact(doc)? {
            Some(self.factory.get_range(values.double_value()?))
        } else {
            None
        };
        Ok(self.state.position(range))
    }

    fn current_value(&self) -> Option<&DoubleRange> {
        self.state.current.as_ref()
    }

    fn set_groups(&mut self, groups: &[SearchGroup<DoubleRange>]) {
        self.state.set_groups(groups);
    }
}

/// `ValueSourceGroupSelector`: groups by a value source's value (its
/// `ValueFiller`'s [`MutableValue`]); a document whose value does not exist
/// is in the `None` group (Java's group of a non-existing mutable value).
///
/// `context` is Java's `Map<Object,Object> context` -- built by
/// [`FunctionContext::create`] for a source with reader-wide state -- and
/// `top` the searcher's leaves as Java's `IndexSearcher` sees them (norms,
/// similarity, and the statistics a `query()` source inside scores with);
/// without it each segment stands alone, without norms, under the default
/// similarity.
pub struct ValueSourceGroupSelector<'a> {
    source: Arc<dyn ValueSource>,
    context: Arc<FunctionContext>,
    top: Option<TopLevel<'a>>,
    values: Option<BoxValues<'a>>,
    value: MutableValue,
    second_pass: Option<HashSet<MutableValue>>,
    include_empty: bool,
}

impl<'a> ValueSourceGroupSelector<'a> {
    /// `new ValueSourceGroupSelector(valueSource, context)`.
    pub fn new(source: Arc<dyn ValueSource>, context: Arc<FunctionContext>) -> Self {
        Self {
            source,
            context,
            top: None,
            values: None,
            value: MutableValue::float(),
            second_pass: None,
            include_empty: false,
        }
    }

    /// The searcher's leaves ([`TopLevel::of_searcher`]): segment `ord`
    /// is read with leaf `ord`'s norms, similarity and reader-wide
    /// statistics.
    pub fn with_top_level(mut self, top: TopLevel<'a>) -> Self {
        self.top = Some(top);
        self
    }
}

impl<'a> GroupSelector<'a> for ValueSourceGroupSelector<'a> {
    type Value = MutableValue;

    /// `setNextReader`: the segment's values and their filler.
    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        let leaf = match self.top.as_ref().and_then(|t| t.leaves.get(ord)) {
            Some(ctx) => ValueLeaf::new(*ctx),
            None => ValueLeaf::of_segment(leaf),
        };
        let values = self.source.get_values(&self.context, &leaf)?;
        self.value = values.new_value();
        self.values = Some(values);
        Ok(())
    }

    /// `advanceTo(doc)`: `fillValue(doc)`, then a value that does not exist
    /// is accepted only when the empty group is wanted, and in the second
    /// pass only the chosen groups' values are.
    fn advance_to(&mut self, doc: i32, _score: f32) -> Result<GroupState> {
        let values = self.values.as_mut().ok_or_else(|| {
            Error::IllegalState("the group selector was moved before a segment".into())
        })?;
        values.fill_value(doc, &mut self.value)?;
        if !self.value.exists() {
            return Ok(if self.include_empty {
                GroupState::Accept
            } else {
                GroupState::Skip
            });
        }
        if let Some(groups) = &self.second_pass {
            if !groups.contains(&self.value) {
                return Ok(GroupState::Skip);
            }
        }
        Ok(GroupState::Accept)
    }

    fn current_value(&self) -> Option<&MutableValue> {
        self.value.exists().then_some(&self.value)
    }

    fn set_groups(&mut self, groups: &[SearchGroup<MutableValue>]) {
        let mut set = HashSet::new();
        for g in groups {
            match &g.group_value {
                None => self.include_empty = true,
                Some(v) => {
                    set.insert(v.clone());
                }
            }
        }
        self.second_pass = Some(set);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ordinal_table_falls_back_to_a_map() {
        let mut t = OrdTable::default();
        t.reset(4);
        t.insert(2, 7);
        t.insert(9, 1);
        assert_eq!(
            (t.get(2), t.get(9), t.get(3), t.get(-1)),
            (Some(7), Some(1), None, None)
        );
        // Too many ordinals to tabulate: a map.
        t.reset(OrdTable::MAX.saturating_add(1));
        assert_eq!(t.get(2), None);
        t.insert(1 << 30, 3);
        assert_eq!(t.get(1 << 30), Some(3));
    }
}
