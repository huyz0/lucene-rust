//! `org.apache.lucene.spatial.util` and the shape value API of
//! `org.apache.lucene.spatial`: [`ShapeValues`]/[`ShapeValuesSource`] (a
//! shape per document), [`ShapeValuesPredicate`] (a `SpatialOperation`
//! against a query shape, per document), the value sources over shapes
//! ([`ShapeAreaValueSource`], [`DistanceToShapeValueSource`]),
//! [`ReciprocalDoubleValuesSource`], [`CachingDoubleValueSource`], and the
//! prefix-tree field cache ([`ShapeFieldCache`],
//! [`PointPrefixTreeFieldCacheProvider`],
//! [`ShapeFieldCacheDistanceValueSource`]).

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use lucene_util::geo::java_double_string;
use lucene_util::spatial4j::{DistanceCalculator, Point, Shape, SpatialContext};
use lucene_util::spatial_extras::prefix_tree::SpatialPrefixTree;
use lucene_util::spatial_extras::query::SpatialOperation;

use crate::explain::Explanation;
use crate::reader::{LeafReader, PostingsFlags};
use crate::values_source::{BoxDoubleValues, DoubleValues, DoubleValuesSource, ValuesContext};
use crate::Result;

/// `ShapeValues`: a per-leaf cursor over each document's shape.
pub trait ShapeValues {
    /// `advanceExact(doc)`: whether `doc` has a shape.
    ///
    /// # Errors
    /// A doc-values decode error.
    fn advance_exact(&mut self, doc: i32) -> Result<bool>;

    /// `value()`: the current document's shape (after `advance_exact`
    /// returned `true`).
    ///
    /// # Errors
    /// A shape that cannot be decoded.
    fn value(&mut self) -> Result<Arc<dyn Shape>>;
}

/// `ShapeValuesSource`: where a document's shape comes from. `Display` is
/// Java's `toString()`.
pub trait ShapeValuesSource: fmt::Display + Send + Sync {
    /// `getValues(ctx)`.
    ///
    /// # Errors
    /// The leaf cannot be read.
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
    ) -> Result<Box<dyn ShapeValues + 'c>>;

    /// `isCacheable(ctx)`.
    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool;
}

/// `DoubleValues.withDefault(in, missingValue)`: every document has a
/// value, `missing` where `inner` has none.
pub(crate) struct WithDefault<'c> {
    pub(crate) inner: BoxDoubleValues<'c>,
    pub(crate) missing: f64,
    pub(crate) has_value: bool,
}

impl<'c> WithDefault<'c> {
    pub(crate) fn boxed(inner: BoxDoubleValues<'c>, missing: f64) -> BoxDoubleValues<'c> {
        Box::new(WithDefault {
            inner,
            missing,
            has_value: false,
        })
    }
}

impl DoubleValues for WithDefault<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.has_value = self.inner.advance_exact(doc)?;
        Ok(true)
    }

    fn double_value(&mut self) -> Result<f64> {
        if self.has_value {
            self.inner.double_value()
        } else {
            Ok(self.missing)
        }
    }
}

/// `ShapeValuesPredicate`: `op.evaluate(docShape, queryShape)` per
/// document. `Display` is `source op queryShape`.
#[derive(Clone)]
pub struct ShapeValuesPredicate {
    pub source: Arc<dyn ShapeValuesSource>,
    pub op: SpatialOperation,
    pub query_shape: Arc<dyn Shape>,
}

impl fmt::Debug for ShapeValuesPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for ShapeValuesPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.source, self.op, self.query_shape)
    }
}

impl ShapeValuesPredicate {
    pub fn new(
        source: Arc<dyn ShapeValuesSource>,
        op: SpatialOperation,
        query_shape: Arc<dyn Shape>,
    ) -> Self {
        ShapeValuesPredicate {
            source,
            op,
            query_shape,
        }
    }

    /// `iterator(ctx, approximation)`'s `matches()` over a leaf: a cursor
    /// answering for documents in increasing order.
    ///
    /// # Errors
    /// The leaf cannot be read.
    pub fn matcher<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
    ) -> Result<PredicateMatcher<'c, '_>> {
        Ok(PredicateMatcher {
            values: self.source.get_values(ctx, leaf)?,
            predicate: self,
        })
    }

    /// `isCacheable(ctx)`.
    pub fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        self.source.is_cacheable(ctx, leaf)
    }
}

/// A leaf's [`ShapeValuesPredicate`] (the `TwoPhaseIterator`'s `matches`).
pub struct PredicateMatcher<'c, 'p> {
    values: Box<dyn ShapeValues + 'c>,
    predicate: &'p ShapeValuesPredicate,
}

impl PredicateMatcher<'_, '_> {
    /// `matches()` for `doc`: it has a shape and the operation holds.
    ///
    /// # Errors
    /// A decode or geometry error.
    pub fn matches(&mut self, doc: i32) -> Result<bool> {
        Ok(self.values.advance_exact(doc)?
            && self
                .predicate
                .op
                .evaluate(&*self.values.value()?, &*self.predicate.query_shape)?)
    }
}

/// `ShapeAreaValueSource`: each document's shape's area (geodetic, in
/// square degrees, when `geo_area`), times a multiplier; 0 without a shape.
pub struct ShapeAreaValueSource {
    source: Arc<dyn ShapeValuesSource>,
    ctx: Arc<SpatialContext>,
    geo_area: bool,
    multiplier: f64,
}

impl ShapeAreaValueSource {
    pub fn new(
        source: Arc<dyn ShapeValuesSource>,
        ctx: Arc<SpatialContext>,
        geo_area: bool,
        multiplier: f64,
    ) -> Self {
        ShapeAreaValueSource {
            source,
            ctx,
            geo_area,
            multiplier,
        }
    }
}

struct AreaValues<'c> {
    shapes: Box<dyn ShapeValues + 'c>,
    ctx: Option<Arc<SpatialContext>>,
    multiplier: f64,
}

impl DoubleValues for AreaValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.shapes.advance_exact(doc)
    }

    fn double_value(&mut self) -> Result<f64> {
        Ok(self.shapes.value()?.area(self.ctx.as_deref())? * self.multiplier)
    }
}

impl DoubleValuesSource for ShapeAreaValueSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        let shapes = self.source.get_values(ctx, leaf)?;
        Ok(WithDefault::boxed(
            Box::new(AreaValues {
                shapes,
                ctx: self.geo_area.then(|| self.ctx.clone()),
                multiplier: self.multiplier,
            }),
            0.0,
        ))
    }

    fn needs_scores(&self) -> bool {
        false
    }

    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        self.source.is_cacheable(ctx, leaf)
    }

    fn describe(&self) -> String {
        format!("area({},geo={})", self.source, self.geo_area)
    }
}

/// `DistanceToShapeValueSource`: the distance from a point to each
/// document's shape's center, times a multiplier; for a document without a
/// shape, 180 times the multiplier (geodetic) or `Double.MAX_VALUE`.
pub struct DistanceToShapeValueSource {
    source: Arc<dyn ShapeValuesSource>,
    query_point: Arc<dyn Point>,
    multiplier: f64,
    dist_calc: Arc<dyn DistanceCalculator>,
    null_value: f64,
}

impl DistanceToShapeValueSource {
    pub fn new(
        source: Arc<dyn ShapeValuesSource>,
        query_point: Arc<dyn Point>,
        multiplier: f64,
        ctx: &SpatialContext,
    ) -> Self {
        DistanceToShapeValueSource {
            source,
            query_point,
            multiplier,
            dist_calc: ctx.dist_calc().clone(),
            null_value: if ctx.is_geo() {
                180.0 * multiplier
            } else {
                f64::MAX
            },
        }
    }
}

struct DistanceToShapeValues<'c> {
    shapes: Box<dyn ShapeValues + 'c>,
    query_point: Arc<dyn Point>,
    dist_calc: Arc<dyn DistanceCalculator>,
    multiplier: f64,
}

impl DoubleValues for DistanceToShapeValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.shapes.advance_exact(doc)
    }

    fn double_value(&mut self) -> Result<f64> {
        let center = self.shapes.value()?.center()?;
        Ok(self.dist_calc.distance(&*self.query_point, &*center)? * self.multiplier)
    }
}

impl DoubleValuesSource for DistanceToShapeValueSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        let shapes = self.source.get_values(ctx, leaf)?;
        Ok(WithDefault::boxed(
            Box::new(DistanceToShapeValues {
                shapes,
                query_point: self.query_point.clone(),
                dist_calc: self.dist_calc.clone(),
                multiplier: self.multiplier,
            }),
            self.null_value,
        ))
    }

    fn needs_scores(&self) -> bool {
        false
    }

    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        self.source.is_cacheable(ctx, leaf)
    }

    fn describe(&self) -> String {
        format!(
            "distance({} to {})*{})",
            self.query_point,
            self.source,
            java_double_string(self.multiplier)
        )
    }
}

/// `ReciprocalDoubleValuesSource`: `c / (v + c)` of another source's
/// values.
pub struct ReciprocalDoubleValuesSource {
    dist_to_edge: f64,
    input: Arc<dyn DoubleValuesSource>,
}

impl ReciprocalDoubleValuesSource {
    pub fn new(dist_to_edge: f64, input: Arc<dyn DoubleValuesSource>) -> Self {
        ReciprocalDoubleValuesSource {
            dist_to_edge,
            input,
        }
    }

    fn recip(&self, v: f64) -> f64 {
        self.dist_to_edge / (v + self.dist_to_edge)
    }
}

struct RecipValues<'c> {
    input: BoxDoubleValues<'c>,
    dist_to_edge: f64,
}

impl DoubleValues for RecipValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.input.advance_exact(doc)
    }

    fn double_value(&mut self) -> Result<f64> {
        Ok(self.dist_to_edge / (self.input.double_value()? + self.dist_to_edge))
    }
}

impl DoubleValuesSource for ReciprocalDoubleValuesSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        Ok(Box::new(RecipValues {
            input: self.input.get_values(ctx, leaf, scores)?,
            dist_to_edge: self.dist_to_edge,
        }))
    }

    fn needs_scores(&self) -> bool {
        self.input.needs_scores()
    }

    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        self.input.is_cacheable(ctx, leaf)
    }

    fn describe(&self) -> String {
        format!(
            "recip({}, {})",
            java_double_string(self.dist_to_edge),
            self.input.describe()
        )
    }

    /// `explain`: the input's explanation, reciprocated.
    fn explain(
        &self,
        ctx: &ValuesContext<'_>,
        leaf: usize,
        doc: i32,
        score_explanation: &Explanation,
    ) -> Result<Explanation> {
        let expl = self.input.explain(ctx, leaf, doc, score_explanation)?;
        let d = java_double_string(self.dist_to_edge);
        Ok(Explanation::match_(
            self.recip(f64::from(expl.value)) as f32,
            format!("{d} / (v + {d}), computed from:"),
        )
        .with_details(vec![expl]))
    }
}

/// `CachingDoubleValueSource`: another source's values, cached by global
/// doc id (the leaf's doc base plus the doc) for the source's lifetime.
pub struct CachingDoubleValueSource {
    source: Arc<dyn DoubleValuesSource>,
    cache: Arc<Mutex<HashMap<i32, f64>>>,
}

impl CachingDoubleValueSource {
    pub fn new(source: Arc<dyn DoubleValuesSource>) -> Self {
        CachingDoubleValueSource {
            source,
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

struct CachingValues<'c> {
    vals: BoxDoubleValues<'c>,
    cache: Arc<Mutex<HashMap<i32, f64>>>,
    base: i32,
    doc: i32,
}

impl DoubleValues for CachingValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.doc = doc;
        self.vals.advance_exact(doc)
    }

    fn double_value(&mut self) -> Result<f64> {
        let key = self.base.wrapping_add(self.doc);
        let cached = self
            .cache
            .lock()
            .map_err(|_| crate::Error::IllegalState("value cache poisoned".into()))?
            .get(&key)
            .copied();
        if let Some(v) = cached {
            return Ok(v);
        }
        let v = self.vals.double_value()?;
        self.cache
            .lock()
            .map_err(|_| crate::Error::IllegalState("value cache poisoned".into()))?
            .insert(key, v);
        Ok(v)
    }
}

impl DoubleValuesSource for CachingDoubleValueSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        let base = ctx.doc_base(leaf);
        Ok(Box::new(CachingValues {
            vals: self.source.get_values(ctx, leaf, scores)?,
            cache: self.cache.clone(),
            base,
            doc: -1,
        }))
    }

    fn needs_scores(&self) -> bool {
        false
    }

    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        self.source.is_cacheable(ctx, leaf)
    }

    fn describe(&self) -> String {
        format!("Cached[{}]", self.source.describe())
    }

    fn explain(
        &self,
        ctx: &ValuesContext<'_>,
        leaf: usize,
        doc: i32,
        score_explanation: &Explanation,
    ) -> Result<Explanation> {
        self.source.explain(ctx, leaf, doc, score_explanation)
    }
}

/// `ShapeFieldCache`: each document's shapes (`None` for none).
#[derive(Debug, Clone)]
pub struct ShapeFieldCache<T> {
    cache: Vec<Option<Vec<T>>>,
    /// `defaultLength`: the initial capacity of a document's list.
    pub default_length: usize,
}

impl<T> ShapeFieldCache<T> {
    pub fn new(length: usize, default_length: usize) -> Self {
        ShapeFieldCache {
            cache: (0..length).map(|_| None).collect(),
            default_length,
        }
    }

    /// `add(docid, s)`. A doc id outside the cache is ignored (Java's array
    /// would throw; the postings only name the segment's documents).
    pub fn add(&mut self, docid: i32, s: T) {
        let default_length = self.default_length;
        if let Some(slot) = usize::try_from(docid)
            .ok()
            .and_then(|d| self.cache.get_mut(d))
        {
            slot.get_or_insert_with(|| Vec::with_capacity(default_length))
                .push(s);
        }
    }

    /// `getShapes(docid)`.
    pub fn get_shapes(&self, docid: i32) -> Option<&[T]> {
        usize::try_from(docid)
            .ok()
            .and_then(|d| self.cache.get(d))
            .and_then(|s| s.as_deref())
    }
}

/// `ShapeFieldCacheProvider<Point>` with `PointPrefixTreeFieldCacheProvider`'s
/// `readShape`: the center of every term's cell at the tree's last level,
/// for each document with that term (deleted ones included).
///
/// Java keeps one cache per `IndexReader` (a `WeakHashMap`); this one is
/// built per [`Self::get_cache`] call.
pub struct PointPrefixTreeFieldCacheProvider {
    grid: Arc<dyn SpatialPrefixTree>,
    shape_field: String,
    default_size: usize,
}

impl fmt::Display for PointPrefixTreeFieldCacheProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PointPrefixTreeFieldCacheProvider({})", self.shape_field)
    }
}

impl PointPrefixTreeFieldCacheProvider {
    pub fn new(grid: Arc<dyn SpatialPrefixTree>, shape_field: &str, default_size: usize) -> Self {
        PointPrefixTreeFieldCacheProvider {
            grid,
            shape_field: shape_field.to_string(),
            default_size,
        }
    }

    /// `readShape(term)`: a last-level cell's center, else nothing.
    fn read_shape(&self, term: &[u8]) -> Result<Option<Arc<dyn Point>>> {
        let cell = self.grid.read_cell(term)?;
        if cell.level() == self.grid.max_levels() {
            return Ok(Some(cell.shape()?.center()?));
        }
        Ok(None)
    }

    /// `getCache(reader)`: every document's points.
    ///
    /// # Errors
    /// A terms or postings decode error, or a cell that cannot be read
    /// (Java's `Terms.getTerms` refuses a field without terms).
    pub fn get_cache(&self, reader: &dyn LeafReader) -> Result<ShapeFieldCache<Arc<dyn Point>>> {
        let mut idx = ShapeFieldCache::new(
            usize::try_from(reader.max_doc()).unwrap_or(0),
            self.default_size,
        );
        let Some(terms) = reader.terms(&self.shape_field)? else {
            // `Terms.getTerms` returns `Terms.EMPTY` for a missing field
            return Ok(idx);
        };
        let mut te = terms.iterator()?;
        while let Some(term) = te.next()? {
            let term = term.to_vec();
            if let Some(shape) = self.read_shape(&term)? {
                let mut docs = te.postings(PostingsFlags::None)?;
                loop {
                    let doc = docs.next_doc()?;
                    if doc == crate::reader::NO_MORE_DOCS {
                        break;
                    }
                    idx.add(doc, shape.clone());
                }
            }
        }
        Ok(idx)
    }
}

/// `ShapeFieldCacheDistanceValueSource`: the distance from a point to a
/// document's nearest cached point, times a multiplier; for a document
/// without one, 180 times the multiplier (geodetic) or `Double.MAX_VALUE`.
pub struct ShapeFieldCacheDistanceValueSource {
    ctx: Arc<SpatialContext>,
    provider: Arc<PointPrefixTreeFieldCacheProvider>,
    from: Arc<dyn Point>,
    multiplier: f64,
}

impl ShapeFieldCacheDistanceValueSource {
    pub fn new(
        ctx: Arc<SpatialContext>,
        provider: Arc<PointPrefixTreeFieldCacheProvider>,
        from: Arc<dyn Point>,
        multiplier: f64,
    ) -> Self {
        ShapeFieldCacheDistanceValueSource {
            ctx,
            provider,
            from,
            multiplier,
        }
    }
}

struct CacheDistanceValues {
    cache: ShapeFieldCache<Arc<dyn Point>>,
    from: Arc<dyn Point>,
    calculator: Arc<dyn DistanceCalculator>,
    multiplier: f64,
    doc: i32,
}

impl DoubleValues for CacheDistanceValues {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.doc = doc;
        Ok(self.cache.get_shapes(doc).is_some())
    }

    fn double_value(&mut self) -> Result<f64> {
        let vals = self.cache.get_shapes(self.doc).unwrap_or(&[]);
        let Some((first, rest)) = vals.split_first() else {
            return Ok(0.0);
        };
        let mut v = self.calculator.distance(&*self.from, &**first)?;
        for p in rest {
            v = super::prefix::query::java_min(v, self.calculator.distance(&*self.from, &**p)?);
        }
        Ok(v * self.multiplier)
    }
}

impl DoubleValuesSource for ShapeFieldCacheDistanceValueSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        let null_value = if self.ctx.is_geo() {
            180.0 * self.multiplier
        } else {
            f64::MAX
        };
        let reader = ctx.leaf_reader(leaf)?;
        Ok(WithDefault::boxed(
            Box::new(CacheDistanceValues {
                cache: self.provider.get_cache(reader)?,
                from: self.from.clone(),
                calculator: self.ctx.dist_calc().clone(),
                multiplier: self.multiplier,
                doc: -1,
            }),
            null_value,
        ))
    }

    fn needs_scores(&self) -> bool {
        false
    }

    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        true
    }

    fn describe(&self) -> String {
        format!(
            "ShapeFieldCacheDistanceValueSource({}, {})",
            self.provider, self.from
        )
    }
}
