//! The group selectors: `TermGroupSelector` (a `SORTED` field's term),
//! `LongRangeGroupSelector`/`DoubleRangeGroupSelector` (the range a values
//! source's value falls in, `LongRangeFactory`/`DoubleRangeFactory`).

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use super::{GroupSelector, GroupState, SearchGroup};
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
pub struct TermGroupSelector<'a> {
    field: String,
    values: Vec<Vec<u8>>,
    ids: HashMap<Vec<u8>, usize>,
    ords_to_group_ids: HashMap<i32, usize>,
    doc_values: Option<Box<dyn SortedDocValues + 'a>>,
    group_id: Option<usize>,
    second_pass: bool,
    include_empty: bool,
}

impl<'a> TermGroupSelector<'a> {
    /// `new TermGroupSelector(field)`.
    pub fn new(field: &str) -> Self {
        Self {
            field: field.to_string(),
            values: Vec::new(),
            ids: HashMap::new(),
            ords_to_group_ids: HashMap::new(),
            doc_values: None,
            group_id: None,
            second_pass: false,
            include_empty: false,
        }
    }
}

impl<'a> GroupSelector<'a> for TermGroupSelector<'a> {
    type Value = Vec<u8>;

    fn set_next_reader(&mut self, _ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        let mut values = dv::get_sorted(leaf_reader(leaf)?, &self.field)?;
        self.ords_to_group_ids.clear();
        for (i, v) in self.values.iter().enumerate() {
            let ord = values.lookup_term(v)?;
            if ord >= 0 {
                self.ords_to_group_ids.insert(ord, i);
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
        if let Some(&id) = self.ords_to_group_ids.get(&ord) {
            self.group_id = Some(id);
            return Ok(GroupState::Accept);
        }
        if self.second_pass {
            return Ok(GroupState::Skip);
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
