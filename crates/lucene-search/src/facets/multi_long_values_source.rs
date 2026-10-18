//! `MultiLongValues` and `MultiLongValuesSource`: per-document sets of
//! `long` values, as the value and range facet counters read them.
//!
//! The static factories are this module's functions ([`from_long_field`],
//! [`from_single_valued`], ...); `toMultiDoubleValuesSource()` is
//! [`to_multi_double_values_source`]. A source reads a leaf through a
//! [`ValuesContext`], as [`LongValuesSource`] does. `hashCode`/`equals` are
//! not ported; `toString` is [`MultiLongValuesSource::describe`].

use std::sync::Arc;

use super::multi_double_values_source::{
    BoxMultiDoubleValues, MultiDoubleValues, MultiDoubleValuesSource,
};
use crate::reader::doc_values::get_sorted_numeric;
use crate::reader::SortedNumericDocValues;
use crate::values_source::{doc_values_cacheable, BoxLongValues, LongValuesSource, ValuesContext};
use crate::Result;

/// `MultiLongValues`: a per-leaf cursor over each document's `long` values.
pub trait MultiLongValues {
    /// `getValueCount()`: how many values the current document has.
    fn get_value_count(&self) -> i64;
    /// `nextValue()`: the current document's next value, in order.
    fn next_value(&mut self) -> Result<i64>;
    /// `advanceExact(doc)`: whether `doc` (not before the last target) has
    /// a value.
    fn advance_exact(&mut self, doc: i32) -> Result<bool>;
}

/// A boxed per-leaf cursor.
pub type BoxMultiLongValues<'c> = Box<dyn MultiLongValues + 'c>;

/// `MultiLongValuesSource`.
pub trait MultiLongValuesSource: Send + Sync {
    /// `getValues(ctx)`: the values of leaf `leaf`.
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
    ) -> Result<BoxMultiLongValues<'c>>;

    /// `SegmentCacheable.isCacheable(ctx)`.
    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool;

    /// `toString()`.
    fn describe(&self) -> String;

    /// The single-valued source this one wraps ([`from_single_valued`]),
    /// which [`unwrap_singleton`] returns.
    fn single_valued(&self) -> Option<Arc<dyn LongValuesSource>> {
        None
    }
}

impl std::fmt::Debug for dyn MultiLongValuesSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.describe())
    }
}

/// `MultiLongValuesSource.fromLongField(field)`: the field's
/// `SORTED_NUMERIC` (or `NUMERIC`) doc values.
pub fn from_long_field(field: &str) -> Arc<dyn MultiLongValuesSource> {
    Arc::new(FieldMultiValueSource(field.to_string()))
}

/// `MultiLongValuesSource.fromIntField(field)`.
pub fn from_int_field(field: &str) -> Arc<dyn MultiLongValuesSource> {
    from_long_field(field)
}

/// `MultiLongValuesSource.fromSingleValued(singleValued)`: each document's
/// one value.
pub fn from_single_valued(
    single_valued: Arc<dyn LongValuesSource>,
) -> Arc<dyn MultiLongValuesSource> {
    Arc::new(SingleValuedAsMultiValued(single_valued))
}

/// `MultiLongValuesSource.unwrapSingleton(in)`: the single-valued source
/// [`from_single_valued`] wrapped, else `None`.
pub fn unwrap_singleton(source: &dyn MultiLongValuesSource) -> Option<Arc<dyn LongValuesSource>> {
    source.single_valued()
}

/// `toMultiDoubleValuesSource()`: each value cast to `double`.
pub fn to_multi_double_values_source(
    source: Arc<dyn MultiLongValuesSource>,
) -> Arc<dyn MultiDoubleValuesSource> {
    Arc::new(DoubleLongValuesSources(source))
}

/// `FieldMultiValueSource`.
struct FieldMultiValueSource(String);

/// Its values: the field's sorted numeric doc values.
struct FieldMultiValues<'c>(Box<dyn SortedNumericDocValues + 'c>);

impl MultiLongValues for FieldMultiValues<'_> {
    fn get_value_count(&self) -> i64 {
        i64::from(self.0.doc_value_count())
    }
    fn next_value(&mut self) -> Result<i64> {
        self.0.next_value()
    }
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.0.advance_exact(doc)
    }
}

impl MultiLongValuesSource for FieldMultiValueSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
    ) -> Result<BoxMultiLongValues<'c>> {
        let reader = ctx.leaf_reader(leaf)?;
        Ok(Box::new(FieldMultiValues(get_sorted_numeric(
            reader, &self.0,
        )?)))
    }
    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        doc_values_cacheable(ctx, leaf, &self.0)
    }
    fn describe(&self) -> String {
        format!("multi-long({})", self.0)
    }
}

/// `SingleValuedAsMultiValued`.
struct SingleValuedAsMultiValued(Arc<dyn LongValuesSource>);

/// Its values: one per document with a value.
struct SingleValues<'c>(BoxLongValues<'c>);

impl MultiLongValues for SingleValues<'_> {
    fn get_value_count(&self) -> i64 {
        1
    }
    fn next_value(&mut self) -> Result<i64> {
        self.0.long_value()
    }
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.0.advance_exact(doc)
    }
}

impl MultiLongValuesSource for SingleValuedAsMultiValued {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
    ) -> Result<BoxMultiLongValues<'c>> {
        Ok(Box::new(SingleValues(self.0.get_values(ctx, leaf, None)?)))
    }
    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        self.0.is_cacheable(ctx, leaf)
    }
    fn describe(&self) -> String {
        format!("multi-long({})", self.0.describe())
    }
    fn single_valued(&self) -> Option<Arc<dyn LongValuesSource>> {
        Some(Arc::clone(&self.0))
    }
}

/// `DoubleLongValuesSources`: a long source's values as doubles.
struct DoubleLongValuesSources(Arc<dyn MultiLongValuesSource>);

/// Its values.
struct LongsAsDoubles<'c>(BoxMultiLongValues<'c>);

impl MultiDoubleValues for LongsAsDoubles<'_> {
    fn get_value_count(&self) -> i64 {
        self.0.get_value_count()
    }
    fn next_value(&mut self) -> Result<f64> {
        // `(double) longValues.nextValue()`.
        #[allow(clippy::cast_precision_loss)]
        Ok(self.0.next_value()? as f64)
    }
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.0.advance_exact(doc)
    }
}

impl MultiDoubleValuesSource for DoubleLongValuesSources {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
    ) -> Result<BoxMultiDoubleValues<'c>> {
        Ok(Box::new(LongsAsDoubles(self.0.get_values(ctx, leaf)?)))
    }
    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        self.0.is_cacheable(ctx, leaf)
    }
    fn describe(&self) -> String {
        format!("multi-long({})", self.0.describe())
    }
}

#[cfg(test)]
mod tests;
