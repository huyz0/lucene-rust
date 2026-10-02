//! `org.apache.lucene.spatial.bbox`: [`BBoxStrategy`] -- a shape indexes as
//! its bounding box, four doubles (`__minX`, `__minY`, `__maxX`, `__maxY`)
//! stored, as doc values and/or as points, plus `__xdl` ("T" when the box
//! crosses the dateline) -- with its rectangle source and the
//! [`BBoxOverlapRatioValueSource`] similarity.

use std::fmt;
use std::sync::Arc;

use lucene_index::document::{
    DocValuesType, DoublePoint, Field, FieldType, IndexOptions, IndexableField,
    NumericDocValuesField, StoredValue, StringField,
};
use lucene_util::spatial4j::{Point, Rectangle, RectangleImpl, Shape, SpatialContext};
use lucene_util::spatial_extras::query::{SpatialArgs, SpatialOperation};

use super::bool_query::{BoolQuery, ConstantScoreBool, Occur};
use super::util::{DistanceToShapeValueSource, ShapeValues, ShapeValuesSource, WithDefault};
use super::{check_field_name, Fields, SpatialStrategy};
use crate::document::{double_point, DocumentQuery, TermConstantScoreQuery};
use crate::explain::Explanation;
use crate::reader::{LeafReader, NumericDocValues};
use crate::values_source::{
    doc_values_cacheable, BoxDoubleValues, DoubleValues, DoubleValuesSource, ValuesContext,
};
use crate::{Error, Result};

/// `BBoxStrategy.SUFFIX_MINX` and the others.
pub const SUFFIX_MINX: &str = "__minX";
pub const SUFFIX_MAXX: &str = "__maxX";
pub const SUFFIX_MINY: &str = "__minY";
pub const SUFFIX_MAXY: &str = "__maxY";
pub const SUFFIX_XDL: &str = "__xdl";

/// `BBoxStrategy.DEFAULT_FIELDTYPE` (and `PointVectorStrategy`'s): points
/// (one eight-byte dimension) and numeric doc values, not stored.
pub fn default_field_type() -> FieldType {
    let mut t = FieldType::new();
    // A fresh type is unfrozen: the setters cannot fail.
    let _ = t.set_dimensions(1, 8);
    let _ = t.set_doc_values_type(DocValuesType::Numeric);
    let _ = t.set_stored(false);
    t.frozen()
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::Spatial(lucene_util::spatial4j::Error::UnsupportedOperation(Some(
        message.into(),
    )))
}

/// `BBoxStrategy`.
#[derive(Clone)]
pub struct BBoxStrategy {
    ctx: Arc<SpatialContext>,
    field_bbox: String,
    pub(crate) field_min_x: String,
    pub(crate) field_max_x: String,
    pub(crate) field_min_y: String,
    pub(crate) field_max_y: String,
    field_xdl: String,
    options_field_type: FieldType,
    has_stored: bool,
    has_doc_vals: bool,
    has_point_vals: bool,
    /// `xdlFieldType`: `StringField.TYPE_NOT_STORED` with `DOCS`, when
    /// there is an index.
    xdl_field_type: Option<FieldType>,
}

impl fmt::Debug for BBoxStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl BBoxStrategy {
    /// `BBoxStrategy.newInstance(ctx, fieldNamePrefix)`: the default field
    /// type.
    ///
    /// # Errors
    /// An empty field name.
    pub fn new_instance(ctx: Arc<SpatialContext>, field_name_prefix: &str) -> Result<Self> {
        Self::new(ctx, field_name_prefix, default_field_type())
    }

    /// `new BBoxStrategy(ctx, fieldNamePrefix, fieldType)`: stored, doc
    /// values and points as the field type says.
    ///
    /// # Errors
    /// An empty field name.
    pub fn new(
        ctx: Arc<SpatialContext>,
        field_name_prefix: &str,
        field_type: FieldType,
    ) -> Result<Self> {
        check_field_name(field_name_prefix)?;
        let has_point_vals = field_type.point_dimension_count() > 0;
        let xdl_field_type = has_point_vals.then(|| {
            let mut t = FieldType::copy_of(&StringField::type_not_stored());
            let _ = t.set_index_options(IndexOptions::Docs);
            t.frozen()
        });
        Ok(BBoxStrategy {
            ctx,
            field_bbox: field_name_prefix.to_string(),
            field_min_x: format!("{field_name_prefix}{SUFFIX_MINX}"),
            field_max_x: format!("{field_name_prefix}{SUFFIX_MAXX}"),
            field_min_y: format!("{field_name_prefix}{SUFFIX_MINY}"),
            field_max_y: format!("{field_name_prefix}{SUFFIX_MAXY}"),
            field_xdl: format!("{field_name_prefix}{SUFFIX_XDL}"),
            has_stored: field_type.stored(),
            has_doc_vals: field_type.doc_values_type() != DocValuesType::None,
            has_point_vals,
            options_field_type: field_type.frozen(),
            xdl_field_type,
        })
    }

    /// `getFieldType()`.
    pub fn field_type(&self) -> &FieldType {
        &self.options_field_type
    }

    /// `createIndexableFields(bbox)`: stored, doc values, points, in that
    /// order, then the dateline flag.
    fn fields_of(&self, bbox: &dyn Rectangle) -> Result<Fields> {
        let illegal = |e: lucene_index::document::Error| Error::IllegalArgument(e.to_string());
        let mut fields: Fields = Vec::new();
        let values = [
            (&self.field_min_x, bbox.min_x()),
            (&self.field_min_y, bbox.min_y()),
            (&self.field_max_x, bbox.max_x()),
            (&self.field_max_y, bbox.max_y()),
        ];
        if self.has_stored {
            for (name, v) in values {
                fields.push(Box::new(Field::stored(
                    name.clone(),
                    StoredValue::Double(v),
                )));
            }
        }
        if self.has_doc_vals {
            // `DoubleDocValuesField`: the raw bits
            for (name, v) in values {
                fields.push(Box::new(NumericDocValuesField::new(
                    name.clone(),
                    v.to_bits() as i64,
                )));
            }
        }
        if self.has_point_vals {
            for (name, v) in values {
                fields.push(Box::new(
                    DoublePoint::new(name.clone(), &[v]).map_err(illegal)?,
                ));
            }
        }
        if let Some(t) = &self.xdl_field_type {
            let value = if bbox.crosses_date_line() { "T" } else { "F" };
            let f: Box<dyn IndexableField> = Box::new(
                Field::from_string(self.field_xdl.clone(), value, t.clone()).map_err(illegal)?,
            );
            fields.push(f);
        }
        Ok(fields)
    }

    /// `makeShapeValueSource()`: each document's box, from the doc values.
    pub fn make_shape_value_source(&self) -> Arc<dyn ShapeValuesSource> {
        Arc::new(BBoxValueSource {
            strategy: self.clone(),
        })
    }

    /// `makeOverlapRatioValueSource(queryBox, queryTargetProportion)`.
    ///
    /// # Errors
    /// A proportion outside `0..=1`.
    pub fn make_overlap_ratio_value_source(
        &self,
        query_box: Arc<dyn Rectangle>,
        query_target_proportion: f64,
    ) -> Result<Arc<dyn DoubleValuesSource>> {
        Ok(Arc::new(BBoxOverlapRatioValueSource::new(
            self.make_shape_value_source(),
            self.ctx.is_geo(),
            query_box,
            query_target_proportion,
            0.0,
        )?))
    }

    /// `makeNumberTermQuery(field, number)`: an exact point query.
    fn number_term(&self, field: &str, number: f64) -> Result<BoolQuery> {
        if self.has_point_vals {
            return Ok(BoolQuery::Clause(Box::new(double_point::new_exact_query(
                field, number,
            )?)));
        }
        Err(unsupported("An index is required for this operation."))
    }

    /// `makeNumericRangeQuery(field, min, max, minInclusive, maxInclusive)`:
    /// an absent bound is infinite; an exclusive one steps (`Math.nextUp` /
    /// `nextDown`).
    fn range(
        &self,
        field: &str,
        min: Option<f64>,
        max: Option<f64>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Result<BoolQuery> {
        if self.has_point_vals {
            let mut min = min.unwrap_or(f64::NEG_INFINITY);
            let mut max = max.unwrap_or(f64::INFINITY);
            if !min_inclusive {
                min = min.next_up();
            }
            if !max_inclusive {
                max = max.next_down();
            }
            return Ok(BoolQuery::Clause(Box::new(double_point::new_range_query(
                field, min, max,
            )?)));
        }
        Err(unsupported("An index is required for this operation."))
    }

    /// `makeXDL(crossedDateLine)`: the flag's term.
    fn xdl(&self, crossed: bool) -> BoolQuery {
        BoolQuery::Clause(Box::new(TermConstantScoreQuery {
            field: self.field_xdl.clone(),
            term: if crossed {
                b"T".to_vec()
            } else {
                b"F".to_vec()
            },
        }))
    }

    /// `makeXDL(crossedDateLine, query)`: the flag and the query (just the
    /// query when not geodetic).
    fn xdl_and(&self, crossed: bool, query: BoolQuery) -> BoolQuery {
        if !self.ctx.is_geo() {
            return query;
        }
        BoolQuery::all(Occur::Must, vec![Some(self.xdl(crossed)), Some(query)])
    }

    fn bool(occur: Occur, clauses: Vec<BoolQuery>) -> BoolQuery {
        BoolQuery::all(occur, clauses.into_iter().map(Some).collect())
    }

    /// `makeContains(bbox)`.
    fn make_contains(&self, bbox: &dyn Rectangle) -> Result<BoolQuery> {
        // docMinY <= minY AND docMaxY >= maxY
        let q_min_y = self.range(&self.field_min_y, None, Some(bbox.min_y()), false, true)?;
        let q_max_y = self.range(&self.field_max_y, Some(bbox.max_y()), None, true, false)?;
        let y_conditions = Self::bool(Occur::Must, vec![q_min_y, q_max_y]);
        let x_conditions = if !bbox.crosses_date_line() {
            // docMinX <= minX AND docMaxX >= maxX
            let q_min_x = self.range(&self.field_min_x, None, Some(bbox.min_x()), false, true)?;
            let q_max_x = self.range(&self.field_max_x, Some(bbox.max_x()), None, true, false)?;
            let q_min_max = Self::bool(Occur::Must, vec![q_min_x, q_max_x]);
            let q_non_xdl = self.xdl_and(false, q_min_max);
            if !self.ctx.is_geo() {
                q_non_xdl
            } else {
                // docMinXLeft <= minX OR docMaxXRight >= maxX
                let q_xdl_left =
                    self.range(&self.field_min_x, None, Some(bbox.min_x()), false, true)?;
                let q_xdl_right =
                    self.range(&self.field_max_x, Some(bbox.max_x()), None, true, false)?;
                let q_xdl_left_right = Self::bool(Occur::Should, vec![q_xdl_left, q_xdl_right]);
                let q_xdl = self.xdl_and(true, q_xdl_left_right);
                #[allow(clippy::float_cmp)] // Java's exact edge test
                let q_edge_dl = if bbox.min_x() == bbox.max_x() && bbox.min_x().abs() == 180.0 {
                    let edge = -bbox.min_x(); // opposite dateline edge (Java's `* -1`)
                    Some(Self::bool(
                        Occur::Should,
                        vec![
                            self.number_term(&self.field_min_x, edge)?,
                            self.number_term(&self.field_max_x, edge)?,
                        ],
                    ))
                } else {
                    None
                };
                BoolQuery::all(Occur::Should, vec![Some(q_non_xdl), Some(q_xdl), q_edge_dl])
            }
        } else {
            // docMinXLeft <= minX AND docMaxXRight >= maxX
            let q_xdl_left =
                self.range(&self.field_min_x, None, Some(bbox.min_x()), false, true)?;
            let q_xdl_right =
                self.range(&self.field_max_x, Some(bbox.max_x()), None, true, false)?;
            let q_xdl_left_right =
                self.xdl_and(true, Self::bool(Occur::Must, vec![q_xdl_left, q_xdl_right]));
            let q_world = Self::bool(
                Occur::Must,
                vec![
                    self.number_term(&self.field_min_x, -180.0)?,
                    self.number_term(&self.field_max_x, 180.0)?,
                ],
            );
            Self::bool(Occur::Should, vec![q_xdl_left_right, q_world])
        };
        Ok(Self::bool(Occur::Must, vec![x_conditions, y_conditions]))
    }

    /// `makeDisjoint(bbox)`.
    fn make_disjoint(&self, bbox: &dyn Rectangle) -> Result<BoolQuery> {
        // docMinY > maxY OR docMaxY < minY
        let q_min_y = self.range(&self.field_min_y, Some(bbox.max_y()), None, false, false)?;
        let q_max_y = self.range(&self.field_max_y, None, Some(bbox.min_y()), false, false)?;
        let y_conditions = Self::bool(Occur::Should, vec![q_min_y, q_max_y]);
        let x_conditions = if !bbox.crosses_date_line() {
            // docMinX > maxX OR docMaxX < minX
            let mut q_min_x =
                self.range(&self.field_min_x, Some(bbox.max_x()), None, false, false)?;
            #[allow(clippy::float_cmp)] // touches the dateline exactly
            if bbox.min_x() == -180.0 && self.ctx.is_geo() {
                // -180 == 180
                q_min_x = BoolQuery::Bool {
                    must: vec![q_min_x],
                    should: Vec::new(),
                    must_not: vec![self.number_term(&self.field_max_x, 180.0)?],
                    min_should_match: 0,
                };
            }
            let mut q_max_x =
                self.range(&self.field_max_x, None, Some(bbox.min_x()), false, false)?;
            #[allow(clippy::float_cmp)] // touches the dateline exactly
            if bbox.max_x() == 180.0 && self.ctx.is_geo() {
                q_max_x = BoolQuery::Bool {
                    must: vec![q_max_x],
                    should: Vec::new(),
                    must_not: vec![self.number_term(&self.field_min_x, -180.0)?],
                    min_should_match: 0,
                };
            }
            let q_min_max = Self::bool(Occur::Should, vec![q_min_x, q_max_x]);
            let q_non_xdl = self.xdl_and(false, q_min_max);
            if !self.ctx.is_geo() {
                q_non_xdl
            } else {
                // both the left and right portions of a crossing document
                // must be disjoint
                let q_min_x_left =
                    self.range(&self.field_min_x, Some(bbox.max_x()), None, false, false)?;
                let q_max_x_right =
                    self.range(&self.field_max_x, None, Some(bbox.min_x()), false, false)?;
                let q_left_right = Self::bool(Occur::Must, vec![q_min_x_left, q_max_x_right]);
                let q_xdl = self.xdl_and(true, q_left_right);
                Self::bool(Occur::Should, vec![q_non_xdl, q_xdl])
            }
        } else {
            // disjoint to both the left and right query portions
            let q_min_x_left = self.range(&self.field_min_x, Some(180.0), None, false, false)?;
            let q_max_x_left =
                self.range(&self.field_max_x, None, Some(bbox.min_x()), false, false)?;
            let q_min_x_right =
                self.range(&self.field_min_x, Some(bbox.max_x()), None, false, false)?;
            let q_max_x_right = self.range(&self.field_max_x, None, Some(-180.0), false, false)?;
            let q_left = Self::bool(Occur::Should, vec![q_min_x_left, q_max_x_left]);
            let q_right = Self::bool(Occur::Should, vec![q_min_x_right, q_max_x_right]);
            let q_left_right = Self::bool(Occur::Must, vec![q_left, q_right]);
            self.xdl_and(false, q_left_right)
        };
        Ok(Self::bool(Occur::Should, vec![x_conditions, y_conditions]))
    }

    /// `makeEquals(bbox)`.
    fn make_equals(&self, bbox: &dyn Rectangle) -> Result<BoolQuery> {
        Ok(Self::bool(
            Occur::Must,
            vec![
                self.number_term(&self.field_min_x, bbox.min_x())?,
                self.number_term(&self.field_min_y, bbox.min_y())?,
                self.number_term(&self.field_max_x, bbox.max_x())?,
                self.number_term(&self.field_max_y, bbox.max_y())?,
            ],
        ))
    }

    /// `makeIntersects(bbox)`: not disjoint, among documents with a box.
    fn make_intersects(&self, bbox: &dyn Rectangle) -> Result<BoolQuery> {
        let q_has_env = if self.ctx.is_geo() {
            Self::bool(Occur::Should, vec![self.xdl(false), self.xdl(true)])
        } else {
            self.xdl(false)
        };
        Ok(BoolQuery::Bool {
            must: vec![q_has_env],
            should: Vec::new(),
            must_not: vec![self.make_disjoint(bbox)?],
            min_should_match: 0,
        })
    }

    /// `makeWithin(bbox)`.
    fn make_within(&self, bbox: &dyn Rectangle) -> Result<BoolQuery> {
        // docMinY >= minY AND docMaxY <= maxY
        let q_min_y = self.range(&self.field_min_y, Some(bbox.min_y()), None, true, false)?;
        let q_max_y = self.range(&self.field_max_y, None, Some(bbox.max_y()), false, true)?;
        let y_conditions = Self::bool(Occur::Must, vec![q_min_y, q_max_y]);
        #[allow(clippy::float_cmp)] // the world's exact edges
        let world = self.ctx.is_geo() && bbox.min_x() == -180.0 && bbox.max_x() == 180.0;
        let x_conditions = if world {
            // if query world-wraps, only the y condition matters
            return Ok(y_conditions);
        } else if !bbox.crosses_date_line() {
            // docMinX >= minX AND docMaxX <= maxX
            let q_min_x = self.range(&self.field_min_x, Some(bbox.min_x()), None, true, false)?;
            let q_max_x = self.range(&self.field_max_x, None, Some(bbox.max_x()), false, true)?;
            let mut q_min_max = Self::bool(Occur::Must, vec![q_min_x, q_max_x]);
            // the opposite dateline of the query, if any
            #[allow(clippy::float_cmp)] // exact edges
            let edge = if bbox.min_x() == -180.0 {
                180.0
            } else if bbox.max_x() == 180.0 {
                -180.0
            } else {
                0.0
            };
            if edge != 0.0 && self.ctx.is_geo() {
                let edge_q = Self::bool(
                    Occur::Must,
                    vec![
                        self.number_term(&self.field_min_x, edge)?,
                        self.number_term(&self.field_max_x, edge)?,
                    ],
                );
                q_min_max = Self::bool(Occur::Should, vec![q_min_max, edge_q]);
            }
            self.xdl_and(false, q_min_max)
        } else {
            // a non-crossing document within the left portion of the query
            let q_min_x_left =
                self.range(&self.field_min_x, Some(bbox.min_x()), None, true, false)?;
            let q_max_x_left = self.range(&self.field_max_x, None, Some(180.0), false, true)?;
            let q_left = Self::bool(Occur::Must, vec![q_min_x_left, q_max_x_left]);
            // ... or within the right portion
            let q_min_x_right = self.range(&self.field_min_x, Some(-180.0), None, true, false)?;
            let q_max_x_right =
                self.range(&self.field_max_x, None, Some(bbox.max_x()), false, true)?;
            let q_right = Self::bool(Occur::Must, vec![q_min_x_right, q_max_x_right]);
            let q_left_right = Self::bool(Occur::Should, vec![q_left, q_right]);
            let q_non_xdl = self.xdl_and(false, q_left_right);
            // a crossing document: its left portion within the query's left,
            // its right within the query's right
            let q_xdl_left =
                self.range(&self.field_min_x, Some(bbox.min_x()), None, true, false)?;
            let q_xdl_right =
                self.range(&self.field_max_x, None, Some(bbox.max_x()), false, true)?;
            let q_xdl_left_right = Self::bool(Occur::Must, vec![q_xdl_left, q_xdl_right]);
            let q_xdl = self.xdl_and(true, q_xdl_left_right);
            Self::bool(Occur::Should, vec![q_non_xdl, q_xdl])
        };
        Ok(Self::bool(Occur::Must, vec![x_conditions, y_conditions]))
    }
}

impl fmt::Display for BBoxStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&super::strategy_string(
            "BBoxStrategy",
            &self.field_bbox,
            &self.ctx,
        ))
    }
}

impl SpatialStrategy for BBoxStrategy {
    fn spatial_context(&self) -> &Arc<SpatialContext> {
        &self.ctx
    }

    fn field_name(&self) -> &str {
        &self.field_bbox
    }

    fn create_indexable_fields(&self, shape: &Arc<dyn Shape>) -> Result<Fields> {
        self.fields_of(&*shape.bounding_box()?)
    }

    fn make_distance_value_source(
        &self,
        query_point: &Arc<dyn Point>,
        multiplier: f64,
    ) -> Result<Arc<dyn DoubleValuesSource>> {
        Ok(Arc::new(DistanceToShapeValueSource::new(
            self.make_shape_value_source(),
            query_point.clone(),
            multiplier,
            &self.ctx,
        )))
    }

    fn make_query(&self, args: &SpatialArgs) -> Result<Box<dyn DocumentQuery>> {
        let shape = &args.shape;
        let Some(bbox) = shape.as_rectangle() else {
            return Err(unsupported(format!(
                "Can only query by Rectangle, not {shape}"
            )));
        };
        let spatial = match args.operation {
            SpatialOperation::BBoxIntersects | SpatialOperation::Intersects => {
                self.make_intersects(bbox)?
            }
            SpatialOperation::BBoxWithin | SpatialOperation::IsWithin => self.make_within(bbox)?,
            SpatialOperation::Contains => self.make_contains(bbox)?,
            SpatialOperation::IsEqualTo => self.make_equals(bbox)?,
            SpatialOperation::IsDisjointTo => self.make_disjoint(bbox)?,
            // no Overlaps support yet
            op => return Err(Error::Spatial(op.unsupported())),
        };
        Ok(Box::new(ConstantScoreBool(spatial)))
    }
}

/// `BBoxValueSource`: each document's box from its four doc values
/// (`Double.longBitsToDouble`), without normalisation.
#[derive(Clone)]
pub struct BBoxValueSource {
    strategy: BBoxStrategy,
}

impl fmt::Display for BBoxValueSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bboxShape({})", self.strategy.field_bbox)
    }
}

/// `DocValues.getNumeric(reader, field)`: the field's numeric column, or
/// none.
pub(crate) fn numeric<'c>(
    reader: &'c crate::directory_reader::SegmentReader,
    field: &str,
) -> Result<Option<Box<dyn NumericDocValues + 'c>>> {
    reader.numeric_doc_values(field)
}

struct BBoxShapeValues<'c> {
    min_x: Option<Box<dyn NumericDocValues + 'c>>,
    min_y: Option<Box<dyn NumericDocValues + 'c>>,
    max_x: Option<Box<dyn NumericDocValues + 'c>>,
    max_y: Option<Box<dyn NumericDocValues + 'c>>,
    ctx: Arc<SpatialContext>,
}

fn advance(dv: &mut Option<Box<dyn NumericDocValues + '_>>, doc: i32) -> Result<bool> {
    match dv {
        Some(dv) => dv.advance_exact(doc),
        None => Ok(false),
    }
}

fn double_of(dv: &Option<Box<dyn NumericDocValues + '_>>) -> f64 {
    f64::from_bits(dv.as_ref().map_or(0, |d| d.long_value()) as u64)
}

impl ShapeValues for BBoxShapeValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        Ok(advance(&mut self.min_x, doc)?
            && advance(&mut self.min_y, doc)?
            && advance(&mut self.max_x, doc)?
            && advance(&mut self.max_y, doc)?)
    }

    fn value(&mut self) -> Result<Arc<dyn Shape>> {
        Ok(Arc::new(RectangleImpl::new(
            double_of(&self.min_x),
            double_of(&self.max_x),
            double_of(&self.min_y),
            double_of(&self.max_y),
            self.ctx.clone(),
        )))
    }
}

impl ShapeValuesSource for BBoxValueSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
    ) -> Result<Box<dyn ShapeValues + 'c>> {
        let reader = ctx.leaf_reader(leaf)?;
        let s = &self.strategy;
        Ok(Box::new(BBoxShapeValues {
            min_x: numeric(reader, &s.field_min_x)?,
            min_y: numeric(reader, &s.field_min_y)?,
            max_x: numeric(reader, &s.field_max_x)?,
            max_y: numeric(reader, &s.field_max_y)?,
            ctx: s.ctx.clone(),
        }))
    }

    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        let s = &self.strategy;
        [
            &s.field_min_x,
            &s.field_min_y,
            &s.field_max_x,
            &s.field_max_y,
        ]
        .iter()
        .all(|f| doc_values_cacheable(ctx, leaf, f))
    }
}

/// `BBoxOverlapRatioValueSource` (a `BBoxSimilarityValueSource`): how much
/// a document's box and the query box overlap, as a 0-1 blend of the
/// overlap's share of the query (weighted `queryTargetProportion`) and of
/// the document's box; lines and points by their length or by
/// intersection. 0 for a document without a box.
#[derive(Clone)]
pub struct BBoxOverlapRatioValueSource {
    bbox_value_source: Arc<dyn ShapeValuesSource>,
    is_geo: bool,
    query_extent: Arc<dyn Rectangle>,
    query_area: f64,
    min_side_length: f64,
    query_target_proportion: f64,
}

impl BBoxOverlapRatioValueSource {
    /// `new BBoxOverlapRatioValueSource(rectValueSource, isGeo, queryExtent,
    /// queryTargetProportion, minSideLength)`.
    ///
    /// # Errors
    /// A proportion outside `0..=1`.
    pub fn new(
        bbox_value_source: Arc<dyn ShapeValuesSource>,
        is_geo: bool,
        query_extent: Arc<dyn Rectangle>,
        query_target_proportion: f64,
        min_side_length: f64,
    ) -> Result<Self> {
        let mut s = BBoxOverlapRatioValueSource {
            bbox_value_source,
            is_geo,
            query_area: 0.0,
            min_side_length,
            query_target_proportion,
            query_extent,
        };
        s.query_area = s.calc_area(s.query_extent.width(), s.query_extent.height());
        // Not a range check: Java lets NaN through.
        #[allow(clippy::manual_range_contains)]
        if query_target_proportion < 0.0 || query_target_proportion > 1.0 {
            return Err(Error::IllegalArgument(
                "queryTargetProportion must be >= 0 and <= 1".into(),
            ));
        }
        Ok(s)
    }

    /// `new BBoxOverlapRatioValueSource(rectValueSource, queryExtent)`:
    /// geodetic, 25% towards the query, no minimum side length.
    ///
    /// # Errors
    /// Never (the proportion is in range).
    pub fn with_defaults(
        bbox_value_source: Arc<dyn ShapeValuesSource>,
        query_extent: Arc<dyn Rectangle>,
    ) -> Result<Self> {
        Self::new(bbox_value_source, true, query_extent, 0.25, 0.0)
    }

    /// `calcArea(width, height)`, with the minimum side length.
    fn calc_area(&self, width: f64, height: f64) -> f64 {
        super::prefix::query::java_max(self.min_side_length, width)
            * super::prefix::query::java_max(self.min_side_length, height)
    }

    /// `score(target, exp)`: the similarity, and (asked for) its
    /// explanation.
    ///
    /// # Errors
    /// A geometry error relating a point-like box.
    pub fn score(
        &self,
        target: &dyn Rectangle,
        want_exp: bool,
    ) -> Result<(f64, Option<Explanation>)> {
        use super::prefix::query::{java_max as max, java_min as min};
        let no = |want: bool| -> Result<(f64, Option<Explanation>)> {
            Ok((0.0, want.then(|| Explanation::no_match("No intersection"))))
        };
        let q = &*self.query_extent;
        // calculate "height": the intersection height between two boxes.
        let top = min(q.max_y(), target.max_y());
        let bottom = max(q.min_y(), target.min_y());
        let height = top - bottom;
        if height < 0.0 {
            return no(want_exp); // no intersection
        }
        // calculate "width": the intersection width between two boxes.
        let mut width = 0.0;
        {
            let (mut a, mut b): (&dyn Rectangle, &dyn Rectangle) = (q, target);
            if a.crosses_date_line() == b.crosses_date_line() {
                // both either cross or don't
                let left = max(a.min_x(), b.min_x());
                let right = min(a.max_x(), b.max_x());
                if !a.crosses_date_line() {
                    // both don't
                    #[allow(clippy::float_cmp)] // the dateline's exact edges
                    let adjacent = self.is_geo
                        && (a.min_x().abs() == 180.0 || a.max_x().abs() == 180.0)
                        && (b.min_x().abs() == 180.0 || b.max_x().abs() == 180.0);
                    if left <= right {
                        width = right - left;
                    } else if adjacent {
                        width = 0.0; // both adjacent to dateline
                    } else {
                        return no(want_exp); // no intersection
                    }
                } else {
                    // both cross
                    width = right - left + 360.0;
                }
            } else {
                if !a.crosses_date_line() {
                    // then flip
                    a = target;
                    b = q;
                }
                // a crosses, b doesn't
                let qry_west_left = max(a.min_x(), b.min_x());
                let qry_west_right = b.max_x();
                if qry_west_left < qry_west_right {
                    width += qry_west_right - qry_west_left;
                }
                let qry_east_left = b.min_x();
                let qry_east_right = min(a.max_x(), b.max_x());
                if qry_east_left < qry_east_right {
                    width += qry_east_right - qry_east_left;
                }
                if qry_west_left > qry_west_right && qry_east_left > qry_east_right {
                    return no(want_exp); // no intersection
                }
            }
        }
        // calculate queryRatio and targetRatio
        let intersection_area = self.calc_area(width, height);
        let query_ratio = if self.query_area > 0.0 {
            intersection_area / self.query_area
        } else if q.height() > 0.0 {
            // vert line
            height / q.height()
        } else if q.width() > 0.0 {
            // horiz line
            width / q.width()
        } else if q.relate(target)?.intersects() {
            1.0
        } else {
            0.0
        };
        let target_area = self.calc_area(target.width(), target.height());
        let target_ratio = if target_area > 0.0 {
            intersection_area / target_area
        } else if target.height() > 0.0 {
            // vert line
            height / target.height()
        } else if target.width() > 0.0 {
            // horiz line
            width / target.width()
        } else if target.relate(q)?.intersects() {
            1.0
        } else {
            0.0
        };
        // combine ratios into a score
        let query_factor = query_ratio * self.query_target_proportion;
        let target_factor = target_ratio * (1.0 - self.query_target_proportion);
        let score = query_factor + target_factor;
        let exp = want_exp.then(|| {
            let min_side_desc = if self.min_side_length > 0.0 {
                format!(
                    " (minSide={})",
                    lucene_util::geo::java_double_string(self.min_side_length)
                )
            } else {
                String::new()
            };
            Explanation::match_(
                score as f32,
                "BBoxOverlapRatioValueSource: queryFactor + targetFactor",
            )
            .with_details(vec![
                Explanation::match_(
                    intersection_area as f32,
                    format!("IntersectionArea{min_side_desc}"),
                )
                .with_details(vec![
                    Explanation::match_(width as f32, "width"),
                    Explanation::match_(height as f32, "height"),
                    Explanation::match_(
                        self.query_target_proportion as f32,
                        "queryTargetProportion",
                    ),
                ]),
                Explanation::match_(query_factor as f32, "queryFactor").with_details(vec![
                    Explanation::match_(target_ratio as f32, "ratio"),
                    Explanation::match_(
                        self.query_area as f32,
                        format!("area of {}{min_side_desc}", self.query_extent),
                    ),
                ]),
                Explanation::match_(target_factor as f32, "targetFactor").with_details(vec![
                    Explanation::match_(target_ratio as f32, "ratio"),
                    Explanation::match_(
                        target_area as f32,
                        format!("area of {target}{min_side_desc}"),
                    ),
                ]),
            ])
        });
        Ok((score, exp))
    }
}

struct OverlapValues<'c> {
    shapes: Box<dyn ShapeValues + 'c>,
    source: BBoxOverlapRatioValueSource,
}

impl DoubleValues for OverlapValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.shapes.advance_exact(doc)
    }

    fn double_value(&mut self) -> Result<f64> {
        let shape = self.shapes.value()?;
        let rect = shape.as_rectangle().ok_or_else(|| {
            super::class_cast(&*shape, "org.locationtech.spatial4j.shape.Rectangle")
        })?;
        Ok(self.source.score(rect, false)?.0)
    }
}

impl DoubleValuesSource for BBoxOverlapRatioValueSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        let shapes = self.bbox_value_source.get_values(ctx, leaf)?;
        Ok(WithDefault::boxed(
            Box::new(OverlapValues {
                shapes,
                source: self.clone(),
            }),
            0.0,
        ))
    }

    fn needs_scores(&self) -> bool {
        false
    }

    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        self.bbox_value_source.is_cacheable(ctx, leaf)
    }

    /// `BBoxSimilarityValueSource.toString()`.
    fn describe(&self) -> String {
        format!(
            "BBoxOverlapRatioValueSource({},{},{})",
            self.bbox_value_source,
            self.query_extent,
            lucene_util::geo::java_double_string(self.query_target_proportion)
        )
    }

    /// `BBoxSimilarityValueSource.explain`: the score's own explanation.
    fn explain(
        &self,
        ctx: &ValuesContext<'_>,
        leaf: usize,
        doc: i32,
        _score_explanation: &Explanation,
    ) -> Result<Explanation> {
        let mut shapes = self.bbox_value_source.get_values(ctx, leaf)?;
        if shapes.advance_exact(doc)? {
            let shape = shapes.value()?;
            if let Some(rect) = shape.as_rectangle() {
                if let (_, Some(e)) = self.score(rect, true)? {
                    return Ok(e);
                }
            }
        }
        Ok(Explanation::no_match(self.describe()))
    }
}
