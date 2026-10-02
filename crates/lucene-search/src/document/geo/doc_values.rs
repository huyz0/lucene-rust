//! The slow (doc-values) geo point queries: `LatLonDocValuesBoxQuery`,
//! `LatLonDocValuesQuery` and `XYDocValuesPointInGeometryQuery`, each a
//! `TwoPhaseIterator` over a `SORTED_NUMERIC` field's packed points.

use std::sync::Arc;

use lucene_index::document::{doc_value_high, doc_value_low};
use lucene_util::geo::{
    Component2D, Component2DPredicate, GeoEncodingUtils, GeoUtils, LatLonGeometry, WithinRelation,
    XYEncodingUtils, XYGeometry,
};

use super::{geo, illegal, sorted_numeric_only, GeoValues, QueryRelation};
use crate::collector::ScoringCollector;
use crate::document::{collect_live, reader, DocumentQuery};
use crate::multi_segment::OpenSegment;
use crate::Result;

/// Every live document of `leaf` with values for `field` that `matches`
/// accepts, at `score`.
fn two_phase(
    leaf: &OpenSegment<'_>,
    field: &str,
    score: f32,
    collector: &mut dyn ScoringCollector,
    mut matches: impl FnMut(&[i64]) -> Result<bool>,
) -> Result<()> {
    let Some(mut values) = sorted_numeric_only(leaf, field)? else {
        return Ok(());
    };
    let max_doc = reader(leaf)?.max_doc;
    let mut buf = Vec::new();
    for doc in 0..max_doc {
        values_of(&mut values, doc, &mut buf)?;
        if !buf.is_empty() && matches(&buf)? {
            collect_live(leaf, doc, score, collector);
        }
    }
    Ok(())
}

#[inline]
fn values_of(values: &mut GeoValues<'_>, doc: i32, out: &mut Vec<i64>) -> Result<()> {
    values.values(doc, out)
}

/// `LatLonDocValuesBoxQuery` (`LatLonDocValuesField.newSlowBoxQuery`): the
/// documents with a value inside the box (crossing the dateline when
/// `min_longitude > max_longitude`), at a constant score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatLonDocValuesBoxQuery {
    pub field: String,
    pub min_latitude: i32,
    pub max_latitude: i32,
    pub min_longitude: i32,
    pub max_longitude: i32,
    pub crosses_dateline: bool,
}

impl LatLonDocValuesBoxQuery {
    /// `LatLonDocValuesBoxQuery(field, minLatitude, maxLatitude,
    /// minLongitude, maxLongitude)`: the bounds encoded, the minimums
    /// rounded up.
    ///
    /// # Errors
    /// An invalid coordinate, with Java's message.
    pub fn new(
        field: impl Into<String>,
        min_latitude: f64,
        max_latitude: f64,
        min_longitude: f64,
        max_longitude: f64,
    ) -> Result<Self> {
        GeoUtils::check_latitude(min_latitude).map_err(geo)?;
        GeoUtils::check_latitude(max_latitude).map_err(geo)?;
        GeoUtils::check_longitude(min_longitude).map_err(geo)?;
        GeoUtils::check_longitude(max_longitude).map_err(geo)?;
        Ok(LatLonDocValuesBoxQuery {
            field: field.into(),
            crosses_dateline: min_longitude > max_longitude,
            min_latitude: GeoEncodingUtils::encode_latitude_ceil(min_latitude).map_err(geo)?,
            max_latitude: GeoEncodingUtils::encode_latitude(max_latitude).map_err(geo)?,
            min_longitude: GeoEncodingUtils::encode_longitude_ceil(min_longitude).map_err(geo)?,
            max_longitude: GeoEncodingUtils::encode_longitude(max_longitude).map_err(geo)?,
        })
    }

    /// The two-phase `matches()` over one document's values.
    fn matches(&self, values: &[i64]) -> bool {
        values.iter().any(|&value| {
            let lat = doc_value_high(value);
            if lat < self.min_latitude || lat > self.max_latitude {
                return false;
            }
            let lon = doc_value_low(value);
            if self.crosses_dateline {
                !(lon > self.max_longitude && lon < self.min_longitude)
            } else {
                !(lon < self.min_longitude || lon > self.max_longitude)
            }
        })
    }
}

impl DocumentQuery for LatLonDocValuesBoxQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        two_phase(leaf, &self.field, boost, collector, |v| Ok(self.matches(v)))
    }
}

/// `LatLonDocValuesQuery` (`LatLonDocValuesField.newSlowDistanceQuery`,
/// `newSlowPolygonQuery`, `newSlowGeometryQuery`): the documents whose
/// values relate to the geometries as `query_relation` says, at a constant
/// score.
#[derive(Debug, Clone)]
pub struct LatLonDocValuesQuery {
    pub field: String,
    pub query_relation: QueryRelation,
    pub geometries: Vec<LatLonGeometry>,
    state: Arc<DvState>,
}

/// `createWeight`'s state: the component predicate, or (for `CONTAINS`)
/// one component per geometry.
enum DvState {
    Predicate(Component2DPredicate<'static>),
    Contains(Vec<Box<dyn Component2D>>),
}

impl std::fmt::Debug for DvState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DvState::Predicate(_) => "Predicate",
            DvState::Contains(_) => "Contains",
        })
    }
}

impl LatLonDocValuesQuery {
    /// `LatLonDocValuesQuery(field, queryRelation, geometries...)`.
    ///
    /// # Errors
    /// A line under `WITHIN`, anything but points under `CONTAINS`, or a
    /// geometry `LatLonGeometry.create` rejects, with Java's message.
    pub fn new(
        field: impl Into<String>,
        query_relation: QueryRelation,
        geometries: &[LatLonGeometry],
    ) -> Result<Self> {
        if query_relation == QueryRelation::Within
            && geometries
                .iter()
                .any(|g| matches!(g, LatLonGeometry::Line(_)))
        {
            return Err(illegal(
                "LatLonDocValuesPointQuery does not support WITHIN queries with line geometries",
            ));
        }
        if query_relation == QueryRelation::Contains
            && geometries
                .iter()
                .any(|g| !matches!(g, LatLonGeometry::Point(_)))
        {
            return Err(illegal(
                "LatLonDocValuesPointQuery does not support CONTAINS queries with non-points \
                 geometries",
            ));
        }
        let component: Arc<dyn Component2D> =
            Arc::from(LatLonGeometry::create(geometries).map_err(geo)?);
        let state = if query_relation == QueryRelation::Contains {
            DvState::Contains(
                geometries
                    .iter()
                    .map(|g| LatLonGeometry::create(std::slice::from_ref(g)))
                    .collect::<std::result::Result<_, _>>()
                    .map_err(geo)?,
            )
        } else {
            DvState::Predicate(
                GeoEncodingUtils::create_component_predicate_shared(component).map_err(geo)?,
            )
        };
        Ok(LatLonDocValuesQuery {
            field: field.into(),
            query_relation,
            geometries: geometries.to_vec(),
            state: Arc::new(state),
        })
    }

    /// The two-phase `matches()` of `intersects`/`within`/`disjoint`/
    /// `contains` over one document's values.
    fn matches(&self, values: &[i64]) -> Result<bool> {
        match self.state.as_ref() {
            DvState::Predicate(p) => {
                let test = |v: &i64| p.test(doc_value_high(*v), doc_value_low(*v));
                Ok(match self.query_relation {
                    QueryRelation::Intersects => values.iter().any(test),
                    QueryRelation::Within => values.iter().all(test),
                    _ => !values.iter().any(test),
                })
            }
            DvState::Contains(components) => {
                let mut answer = WithinRelation::Disjoint;
                for &value in values {
                    let lat = GeoEncodingUtils::decode_latitude(doc_value_high(value));
                    let lon = GeoEncodingUtils::decode_longitude(doc_value_low(value));
                    for c in components {
                        match c.within_point(lon, lat).map_err(geo)? {
                            WithinRelation::NotWithin => return Ok(false),
                            WithinRelation::Disjoint => {}
                            r => answer = r,
                        }
                    }
                }
                Ok(answer == WithinRelation::Candidate)
            }
        }
    }
}

impl DocumentQuery for LatLonDocValuesQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        two_phase(leaf, &self.field, boost, collector, |v| self.matches(v))
    }
}

/// `XYDocValuesPointInGeometryQuery` (every `XYDocValuesField` query): the
/// documents with a value inside the geometries, at a constant score.
#[derive(Debug, Clone)]
pub struct XYDocValuesPointInGeometryQuery {
    pub field: String,
    pub geometries: Vec<XYGeometry>,
    component: Arc<dyn Component2D>,
}

impl XYDocValuesPointInGeometryQuery {
    /// `XYDocValuesPointInGeometryQuery(field, geometries...)`.
    ///
    /// # Errors
    /// No geometries, or one `XYGeometry.create` rejects, with Java's
    /// message.
    pub fn new(field: impl Into<String>, geometries: &[XYGeometry]) -> Result<Self> {
        if geometries.is_empty() {
            return Err(illegal("geometries must not be empty"));
        }
        Ok(XYDocValuesPointInGeometryQuery {
            field: field.into(),
            geometries: geometries.to_vec(),
            component: Arc::from(XYGeometry::create(geometries).map_err(geo)?),
        })
    }
}

impl DocumentQuery for XYDocValuesPointInGeometryQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        two_phase(leaf, &self.field, boost, collector, |values| {
            Ok(values.iter().any(|&v| {
                let x = f64::from(XYEncodingUtils::decode(doc_value_high(v)));
                let y = f64::from(XYEncodingUtils::decode(doc_value_low(v)));
                self.component.contains(x, y)
            }))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_index::document::LatLonDocValuesField;
    use lucene_util::geo::{Line, Point};

    #[test]
    fn box_matches_across_the_dateline() {
        let q = LatLonDocValuesBoxQuery::new("f", -10.0, 10.0, 170.0, -170.0).unwrap();
        assert!(q.crosses_dateline);
        let v = |lat, lon| LatLonDocValuesField::encode(lat, lon).unwrap();
        assert!(q.matches(&[v(0.0, 175.0)]));
        assert!(q.matches(&[v(0.0, -175.0)]));
        assert!(!q.matches(&[v(0.0, 0.0)]));
        assert!(!q.matches(&[v(20.0, 175.0)]));
        assert!(q.matches(&[v(20.0, 175.0), v(5.0, -179.0)]));
        assert!(LatLonDocValuesBoxQuery::new("f", -91.0, 0.0, 0.0, 0.0).is_err());
        assert!(LatLonDocValuesBoxQuery::new("f", 0.0, 0.0, 0.0, 181.0).is_err());
    }

    #[test]
    fn relations_over_several_values() {
        let v = |lat, lon| LatLonDocValuesField::encode(lat, lon).unwrap();
        let point = |lat, lon| LatLonGeometry::Point(Point::new(lat, lon).unwrap());
        let rect = LatLonGeometry::Rectangle(
            lucene_util::geo::Rectangle::new(0.0, 10.0, 0.0, 10.0).unwrap(),
        );
        let inside = v(5.0, 5.0);
        let outside = v(50.0, 50.0);
        let within =
            LatLonDocValuesQuery::new("f", QueryRelation::Within, std::slice::from_ref(&rect))
                .unwrap();
        assert!(within.matches(&[inside]).unwrap());
        assert!(!within.matches(&[inside, outside]).unwrap());
        let dis =
            LatLonDocValuesQuery::new("f", QueryRelation::Disjoint, std::slice::from_ref(&rect))
                .unwrap();
        assert!(dis.matches(&[outside]).unwrap());
        assert!(!dis.matches(&[inside, outside]).unwrap());
        let int = LatLonDocValuesQuery::new("f", QueryRelation::Intersects, &[rect]).unwrap();
        assert!(int.matches(&[outside, inside]).unwrap());
        let p = GeoEncodingUtils::decode_latitude(doc_value_high(inside));
        let l = GeoEncodingUtils::decode_longitude(doc_value_low(inside));
        let con = LatLonDocValuesQuery::new("f", QueryRelation::Contains, &[point(p, l)]).unwrap();
        assert!(con.matches(&[inside]).unwrap());
        assert!(!con.matches(&[outside]).unwrap());
        assert!(format!("{:?}", con.state).contains("Contains"));
        assert!(format!("{:?}", int.state).contains("Predicate"));
        let line = LatLonGeometry::Line(Line::new(&[0.0, 1.0], &[0.0, 1.0]).unwrap());
        let e = LatLonDocValuesQuery::new("f", QueryRelation::Within, std::slice::from_ref(&line))
            .unwrap_err();
        assert!(e.to_string().contains("WITHIN queries with line"), "{e}");
        let e = LatLonDocValuesQuery::new("f", QueryRelation::Contains, &[line]).unwrap_err();
        assert!(e.to_string().contains("non-points"), "{e}");
        assert!(XYDocValuesPointInGeometryQuery::new("f", &[]).is_err());
    }
}
