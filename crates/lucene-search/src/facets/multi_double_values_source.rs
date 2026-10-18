//! `MultiDoubleValues` and `MultiDoubleValuesSource`: per-document sets of
//! `double` values, as the range facet counters read them.
//!
//! The static factories are this module's functions ([`from_field`],
//! [`from_double_field`], ...); `toMultiLongValuesSource()` and
//! `toSortableMultiLongValuesSource()` are [`to_multi_long_values_source`]
//! and [`to_sortable_multi_long_values_source`]. `hashCode`/`equals` are not
//! ported; `toString` is [`MultiDoubleValuesSource::describe`].

use std::sync::Arc;

use super::multi_long_values_source::{BoxMultiLongValues, MultiLongValues, MultiLongValuesSource};
use crate::reader::doc_values::get_sorted_numeric;
use crate::reader::SortedNumericDocValues;
use crate::values_source::{
    doc_values_cacheable, double_to_sortable_long, BoxDoubleValues, Decoder, DoubleValuesSource,
    ValuesContext,
};
use crate::Result;

/// `MultiDoubleValues`: a per-leaf cursor over each document's `double`
/// values.
pub trait MultiDoubleValues {
    /// `getValueCount()`.
    fn get_value_count(&self) -> i64;
    /// `nextValue()`.
    fn next_value(&mut self) -> Result<f64>;
    /// `advanceExact(doc)`.
    fn advance_exact(&mut self, doc: i32) -> Result<bool>;
}

/// A boxed per-leaf cursor.
pub type BoxMultiDoubleValues<'c> = Box<dyn MultiDoubleValues + 'c>;

/// `MultiDoubleValuesSource`.
pub trait MultiDoubleValuesSource: Send + Sync {
    /// `getValues(ctx)`: the values of leaf `leaf`.
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
    ) -> Result<BoxMultiDoubleValues<'c>>;

    /// `SegmentCacheable.isCacheable(ctx)`.
    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool;

    /// `toString()`.
    fn describe(&self) -> String;

    /// The single-valued source this one wraps ([`from_single_valued`]),
    /// which [`unwrap_singleton`] returns.
    fn single_valued(&self) -> Option<Arc<dyn DoubleValuesSource>> {
        None
    }
}

impl std::fmt::Debug for dyn MultiDoubleValuesSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.describe())
    }
}

/// `MultiDoubleValuesSource.fromField(field, decoder)`.
pub fn from_field(field: &str, decoder: Decoder) -> Arc<dyn MultiDoubleValuesSource> {
    Arc::new(FieldMultiValuedSource {
        field: field.to_string(),
        decoder,
    })
}

/// `MultiDoubleValuesSource.fromDoubleField(field)`.
pub fn from_double_field(field: &str) -> Arc<dyn MultiDoubleValuesSource> {
    from_field(field, Decoder::Double)
}

/// `MultiDoubleValuesSource.fromFloatField(field)`.
pub fn from_float_field(field: &str) -> Arc<dyn MultiDoubleValuesSource> {
    from_field(field, Decoder::Float)
}

/// `MultiDoubleValuesSource.fromLongField(field)`.
pub fn from_long_field(field: &str) -> Arc<dyn MultiDoubleValuesSource> {
    from_field(field, Decoder::Long)
}

/// `MultiDoubleValuesSource.fromIntField(field)`.
pub fn from_int_field(field: &str) -> Arc<dyn MultiDoubleValuesSource> {
    from_long_field(field)
}

/// `MultiDoubleValuesSource.fromSingleValued(singleValued)`.
pub fn from_single_valued(
    single_valued: Arc<dyn DoubleValuesSource>,
) -> Arc<dyn MultiDoubleValuesSource> {
    Arc::new(SingleValuedAsMultiValued(single_valued))
}

/// `MultiDoubleValuesSource.unwrapSingleton(in)`.
pub fn unwrap_singleton(
    source: &dyn MultiDoubleValuesSource,
) -> Option<Arc<dyn DoubleValuesSource>> {
    source.single_valued()
}

/// `toMultiLongValuesSource()`: each value cast to `long` (Java's
/// saturating, NaN-to-zero cast, which `as` is).
pub fn to_multi_long_values_source(
    source: Arc<dyn MultiDoubleValuesSource>,
) -> Arc<dyn MultiLongValuesSource> {
    Arc::new(LongDoubleValuesSource(source))
}

/// `toSortableMultiLongValuesSource()`: each value as
/// `NumericUtils.doubleToSortableLong`.
pub fn to_sortable_multi_long_values_source(
    source: Arc<dyn MultiDoubleValuesSource>,
) -> Arc<dyn MultiLongValuesSource> {
    Arc::new(SortableMultiLongValuesSource(source))
}

/// `values` as `NumericUtils.doubleToSortableLong`s.
pub(crate) fn sortable_longs(values: BoxMultiDoubleValues<'_>) -> BoxMultiLongValues<'_> {
    Box::new(DoublesAsLongs {
        values,
        sortable: true,
    })
}

/// `FieldMultiValuedSource`.
struct FieldMultiValuedSource {
    field: String,
    decoder: Decoder,
}

/// Its values.
struct FieldValues<'c> {
    values: Box<dyn SortedNumericDocValues + 'c>,
    decoder: Decoder,
}

impl MultiDoubleValues for FieldValues<'_> {
    fn get_value_count(&self) -> i64 {
        i64::from(self.values.doc_value_count())
    }
    fn next_value(&mut self) -> Result<f64> {
        Ok(self.decoder.apply(self.values.next_value()?))
    }
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.values.advance_exact(doc)
    }
}

impl MultiDoubleValuesSource for FieldMultiValuedSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
    ) -> Result<BoxMultiDoubleValues<'c>> {
        let reader = ctx.leaf_reader(leaf)?;
        Ok(Box::new(FieldValues {
            values: get_sorted_numeric(reader, &self.field)?,
            decoder: self.decoder.clone(),
        }))
    }
    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        doc_values_cacheable(ctx, leaf, &self.field)
    }
    fn describe(&self) -> String {
        format!("multi-double({})", self.field)
    }
}

/// `SingleValuedAsMultiValued`.
struct SingleValuedAsMultiValued(Arc<dyn DoubleValuesSource>);

/// Its values.
struct SingleValues<'c>(BoxDoubleValues<'c>);

impl MultiDoubleValues for SingleValues<'_> {
    fn get_value_count(&self) -> i64 {
        1
    }
    fn next_value(&mut self) -> Result<f64> {
        self.0.double_value()
    }
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.0.advance_exact(doc)
    }
}

impl MultiDoubleValuesSource for SingleValuedAsMultiValued {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
    ) -> Result<BoxMultiDoubleValues<'c>> {
        Ok(Box::new(SingleValues(self.0.get_values(ctx, leaf, None)?)))
    }
    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        self.0.is_cacheable(ctx, leaf)
    }
    fn describe(&self) -> String {
        format!("multi-double({})", self.0.describe())
    }
    fn single_valued(&self) -> Option<Arc<dyn DoubleValuesSource>> {
        Some(Arc::clone(&self.0))
    }
}

/// `LongDoubleValuesSource`: values cast to `long`.
struct LongDoubleValuesSource(Arc<dyn MultiDoubleValuesSource>);

/// `SortableMultiLongValuesSource`: values as sortable longs.
struct SortableMultiLongValuesSource(Arc<dyn MultiDoubleValuesSource>);

/// The values of either: `sortable` picks the conversion.
struct DoublesAsLongs<'c> {
    values: BoxMultiDoubleValues<'c>,
    sortable: bool,
}

impl MultiLongValues for DoublesAsLongs<'_> {
    fn get_value_count(&self) -> i64 {
        self.values.get_value_count()
    }
    fn next_value(&mut self) -> Result<i64> {
        let v = self.values.next_value()?;
        // `(long) v`: Java's cast saturates and maps NaN to 0, as `as` does.
        #[allow(clippy::cast_possible_truncation)]
        Ok(if self.sortable {
            double_to_sortable_long(v)
        } else {
            v as i64
        })
    }
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.values.advance_exact(doc)
    }
}

impl MultiLongValuesSource for LongDoubleValuesSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
    ) -> Result<BoxMultiLongValues<'c>> {
        Ok(Box::new(DoublesAsLongs {
            values: self.0.get_values(ctx, leaf)?,
            sortable: false,
        }))
    }
    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        self.0.is_cacheable(ctx, leaf)
    }
    fn describe(&self) -> String {
        format!("multi-double({})", self.0.describe())
    }
}

impl MultiLongValuesSource for SortableMultiLongValuesSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
    ) -> Result<BoxMultiLongValues<'c>> {
        Ok(Box::new(DoublesAsLongs {
            values: self.0.get_values(ctx, leaf)?,
            sortable: true,
        }))
    }
    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        self.0.is_cacheable(ctx, leaf)
    }
    fn describe(&self) -> String {
        format!("sortableMultiLong({})", self.0.describe())
    }
}

#[cfg(test)]
mod tests;
