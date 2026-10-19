//! `LatLonBoundingBox`: a geo bounding box indexed as a two-dimensional
//! range (four `LatLonPoint`-encoded ints: min lat, min lon, max lat, max
//! lon), with its `RangeFieldQuery` factories. Neither dateline nor pole
//! crossing is supported, as in Java.

use lucene_util::geo::{java_double_string, GeoEncodingUtils};

use super::{illegal, packed_indexable, point_type, PackedField};
use crate::document::RangeFieldQuery;
use crate::Result;
use lucene_index::document::RangeQueryType;

/// `LatLonBoundingBox`.
#[derive(Debug, Clone, PartialEq)]
pub struct LatLonBoundingBox(PackedField);

impl LatLonBoundingBox {
    /// `BYTES`: `LatLonPoint.BYTES`.
    pub const BYTES: usize = 4;

    /// `new LatLonBoundingBox(name, minLat, minLon, maxLat, maxLon)`.
    pub fn new(
        name: impl Into<String>,
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
    ) -> Result<Self> {
        Ok(LatLonBoundingBox(PackedField {
            name: name.into(),
            field_type: point_type(4, 4)?,
            packed: encode(min_lat, min_lon, max_lat, max_lon)?,
        }))
    }

    /// `setRangeValues(minLat, minLon, maxLat, maxLon)`.
    pub fn set_range_values(
        &mut self,
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
    ) -> Result<()> {
        self.0.packed = encode(min_lat, min_lon, max_lat, max_lon)?;
        Ok(())
    }

    /// The packed box (`binaryValue()`).
    pub fn packed(&self) -> &[u8] {
        &self.0.packed
    }

    /// `newIntersectsQuery(field, minLat, minLon, maxLat, maxLon)`.
    pub fn new_intersects_query(
        field: &str,
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
    ) -> Result<RangeFieldQuery> {
        new_range_query(
            field,
            [min_lat, min_lon, max_lat, max_lon],
            RangeQueryType::Intersects,
        )
    }

    /// `newWithinQuery(field, minLat, minLon, maxLat, maxLon)`.
    pub fn new_within_query(
        field: &str,
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
    ) -> Result<RangeFieldQuery> {
        new_range_query(
            field,
            [min_lat, min_lon, max_lat, max_lon],
            RangeQueryType::Within,
        )
    }

    /// `newContainsQuery(field, minLat, minLon, maxLat, maxLon)`.
    pub fn new_contains_query(
        field: &str,
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
    ) -> Result<RangeFieldQuery> {
        new_range_query(
            field,
            [min_lat, min_lon, max_lat, max_lon],
            RangeQueryType::Contains,
        )
    }

    /// `newCrossesQuery(field, minLat, minLon, maxLat, maxLon)`.
    pub fn new_crosses_query(
        field: &str,
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
    ) -> Result<RangeFieldQuery> {
        new_range_query(
            field,
            [min_lat, min_lon, max_lat, max_lon],
            RangeQueryType::Crosses,
        )
    }
}

packed_indexable!(LatLonBoundingBox, |_: &LatLonBoundingBox| None);

/// `newRangeQuery(field, minLat, minLon, maxLat, maxLon, queryType)`.
fn new_range_query(
    field: &str,
    b: [f64; 4],
    query_type: RangeQueryType,
) -> Result<RangeFieldQuery> {
    RangeFieldQuery::new(field, encode(b[0], b[1], b[2], b[3])?, 2, query_type)
}

/// `checkArgs(minLat, minLon, maxLat, maxLon)`.
fn check_args(min_lat: f64, min_lon: f64, max_lat: f64, max_lon: f64) -> Result<()> {
    if min_lon > max_lon {
        return Err(illegal(format!(
            "cannot have minLon [{}] exceed maxLon [{}].",
            java_double_string(min_lon),
            java_double_string(max_lon)
        )));
    }
    if min_lat > max_lat {
        return Err(illegal(format!(
            "cannot have minLat [{}] exceed maxLat [{}].",
            java_double_string(min_lat),
            java_double_string(max_lat)
        )));
    }
    Ok(())
}

/// `encode(minLat, minLon, maxLat, maxLon)`, after `checkArgs`: each corner
/// as `LatLonPoint` encodes it, sortable.
fn encode(min_lat: f64, min_lon: f64, max_lat: f64, max_lon: f64) -> Result<Vec<u8>> {
    check_args(min_lat, min_lon, max_lat, max_lon)?;
    let geo = |e: lucene_util::geo::GeoError| illegal(e.to_string());
    let mut out = Vec::with_capacity(16);
    for (lat, lon) in [(min_lat, min_lon), (max_lat, max_lon)] {
        let lat = GeoEncodingUtils::encode_latitude(lat).map_err(geo)?;
        let lon = GeoEncodingUtils::encode_longitude(lon).map_err(geo)?;
        out.extend_from_slice(&sortable(lat));
        out.extend_from_slice(&sortable(lon));
    }
    Ok(out)
}

/// `NumericUtils.intToSortableBytes`.
fn sortable(v: i32) -> [u8; 4] {
    ((v as u32) ^ 0x8000_0000).to_be_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_and_checks() {
        let mut b = LatLonBoundingBox::new("bb", -10.0, -20.0, 30.0, 40.0).unwrap();
        assert_eq!(b.packed().len(), 16);
        let lat = GeoEncodingUtils::encode_latitude(-10.0).unwrap();
        assert_eq!(&b.packed()[..4], &sortable(lat));
        b.set_range_values(0.0, 0.0, 1.0, 1.0).unwrap();
        assert_eq!(&b.packed()[..4], &sortable(0));
        let e = LatLonBoundingBox::new("bb", 0.0, 10.0, 1.0, 5.0).unwrap_err();
        assert_eq!(
            e.to_string(),
            "document query: cannot have minLon [10.0] exceed maxLon [5.0]."
        );
        let e = b.set_range_values(2.0, 0.0, 1.0, 5.0).unwrap_err();
        assert_eq!(
            e.to_string(),
            "document query: cannot have minLat [2.0] exceed maxLat [1.0]."
        );
        assert!(LatLonBoundingBox::new("bb", 95.0, 0.0, 96.0, 1.0).is_err());
        assert!(LatLonBoundingBox::new("bb", 0.0, -190.0, 1.0, 1.0).is_err());
        assert_eq!(
            lucene_index::document::IndexableField::numeric_value(&b),
            None
        );
        for q in [
            LatLonBoundingBox::new_intersects_query("bb", 0.0, 0.0, 1.0, 1.0),
            LatLonBoundingBox::new_within_query("bb", 0.0, 0.0, 1.0, 1.0),
            LatLonBoundingBox::new_contains_query("bb", 0.0, 0.0, 1.0, 1.0),
            LatLonBoundingBox::new_crosses_query("bb", 0.0, 0.0, 1.0, 1.0),
        ] {
            let q = q.unwrap();
            assert_eq!((q.num_dims, q.bytes_per_dim), (2, 4));
        }
        assert!(LatLonBoundingBox::new_within_query("bb", 1.0, 0.0, 0.0, 1.0).is_err());
    }
}
