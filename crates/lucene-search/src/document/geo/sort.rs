//! Distance sorts: `LatLonPointSortField` with `LatLonPointDistanceComparator`
//! (`LatLonDocValuesField.newDistanceSort`) and `XYPointSortField` with
//! `XYPointDistanceComparator` (`XYDocValuesField.newDistanceSort`).
//!
//! [`LatLonPointSortField::search`] is `IndexSearcher.search(query, n, new
//! Sort(sortField))`: `TopFieldCollector` with this one comparator, run as
//! Java runs it -- every match is compared against the queue's bottom
//! (`compareBottom`, which first rejects a value outside the bottom's
//! bounding box, rebuilt on every `setBottom` for the first 1024 and then
//! every 64th), copied when competitive, and the hits come back nearest
//! first, ties by doc id, each with its distance (`haversin2` of the sort
//! key; the cartesian distance itself for `XY`).
//!
//! Both sort fields are also [`FieldComparatorSource`]s
//! ([`LatLonPointSortField::comparator_source`]), so a distance can be one
//! key of a [`crate::top_field::search_sorted`] sort: there a hit's value is
//! the comparable long of the sort key
//! (`lucene_index::document::double_to_sortable_long`).

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use lucene_index::document::{doc_value_high, doc_value_low, double_to_sortable_long};
use lucene_util::geo::{GeoEncodingUtils, GeoUtils, Rectangle, XYEncodingUtils, XYRectangle};
use lucene_util::sloppy_math;
use lucene_util::spatial3d::{
    DistanceStyle, GeoDistanceShape, GeoOutsideDistance, PlanetModel, XYZBounds,
};

use lucene_codecs::field_infos::FieldInfo;

use super::{geo, idx, illegal, sorted_numeric_in, GeoValues};
use crate::collector::{ScoreMode, ScoringCollector};
use crate::directory_reader::SegmentReader;
use crate::document::{reader, DocumentQuery};
use crate::multi_segment::OpenSegment;
use crate::top_field::{
    FieldComparator, FieldComparatorSource, LeafCtx, LeafFieldComparator, SortValue,
};
use crate::{Error, Result};

/// A distance-sorted search's answer.
#[derive(Debug, Clone, PartialEq)]
pub struct SortedDistance {
    /// How many documents matched (exact: nothing is skipped).
    pub total_hits: u64,
    /// `(global doc id, distance)`, nearest first, ties by doc id; a
    /// document without a value sorts last at `f64::INFINITY`.
    pub hits: Vec<(i32, f64)>,
}

/// One comparator's distance logic: what `compareBottom`, `setBottom` and
/// `sortKey` differ in between the two sorts.
trait Distance: Send + Sync {
    /// `setBottom(slot)`'s bounding-box update for a new bottom key.
    fn set_bottom(&mut self, bottom: f64) -> Result<()>;
    /// `compareBottom(doc)` over a document's values (non-empty).
    fn compare_bottom(&self, bottom: f64, values: &[i64]) -> Ordering;
    /// `sortKey(doc)` over a document's values (non-empty).
    fn sort_key(&self, values: &[i64]) -> f64;
    /// `value(slot)`: the reported value of a key.
    fn value(&self, key: f64) -> f64;
    /// `getLeafComparator`'s field check.
    fn check(&self, r: &SegmentReader, field: &str) -> Result<()>;
    /// [`Self::sort_key`] where the comparator is driven from outside
    /// [`search_distance`]: a geo3d shape method's exception (raised, see
    /// `lucene_util::spatial3d::errors`) becomes the key's error.
    fn sort_key_checked(&self, values: &[i64]) -> Result<f64> {
        Ok(self.sort_key(values))
    }
}

/// `Double.compare(a, b)`, as an ordering.
#[inline]
fn double_compare(a: f64, b: f64) -> Ordering {
    a.total_cmp(&b)
}

/// `LatLonDocValuesField.checkCompatible` / `XYDocValuesField`'s.
fn check_dv(r: &SegmentReader, field: &str, kind: &str) -> Result<()> {
    use lucene_codecs::field_infos::DocValuesType;
    if let Some(info) = info_of(r, field) {
        if info.doc_values_type != DocValuesType::None
            && info.doc_values_type != DocValuesType::SortedNumeric
        {
            return Err(illegal(format!(
                "field=\"{field}\" was indexed with docValuesType={} but this type has \
                 docValuesType=SORTED_NUMERIC, is the field really a {kind}?",
                lucene_index::document::doc_values_type_name(info.doc_values_type)
            )));
        }
    }
    Ok(())
}

/// `LatLonPointDistanceComparator`'s state beyond the slots.
#[derive(Debug, Clone)]
struct LatLonDistance {
    latitude: f64,
    longitude: f64,
    min_lat: i32,
    max_lat: i32,
    min_lon: i32,
    max_lon: i32,
    min_lon2: i32,
    set_bottom_counter: i32,
}

impl LatLonDistance {
    fn new(latitude: f64, longitude: f64) -> Self {
        LatLonDistance {
            latitude,
            longitude,
            min_lat: i32::MIN,
            max_lat: i32::MAX,
            min_lon: i32::MIN,
            max_lon: i32::MAX,
            min_lon2: i32::MAX,
            set_bottom_counter: 0,
        }
    }
}

/// `LatLonPointDistanceComparator.haversin2`.
fn haversin2(partial: f64) -> f64 {
    if partial.is_infinite() {
        return partial;
    }
    sloppy_math::haversin_meters_from_sort_key(partial)
}

impl Distance for LatLonDistance {
    fn set_bottom(&mut self, bottom: f64) -> Result<()> {
        let c = self.set_bottom_counter;
        if c < 1024 || (c & 0x3F) == 0x3F {
            let b =
                Rectangle::from_point_distance(self.latitude, self.longitude, haversin2(bottom))
                    .map_err(geo)?;
            self.min_lat = GeoEncodingUtils::encode_latitude(b.min_lat).map_err(geo)?;
            self.max_lat = GeoEncodingUtils::encode_latitude(b.max_lat).map_err(geo)?;
            if b.crosses_dateline() {
                self.min_lon = i32::MIN;
                self.max_lon = GeoEncodingUtils::encode_longitude(b.max_lon).map_err(geo)?;
                self.min_lon2 = GeoEncodingUtils::encode_longitude(b.min_lon).map_err(geo)?;
            } else {
                self.min_lon = GeoEncodingUtils::encode_longitude(b.min_lon).map_err(geo)?;
                self.max_lon = GeoEncodingUtils::encode_longitude(b.max_lon).map_err(geo)?;
                self.min_lon2 = i32::MAX;
            }
        }
        self.set_bottom_counter = c.saturating_add(1);
        Ok(())
    }

    #[inline]
    fn compare_bottom(&self, bottom: f64, values: &[i64]) -> Ordering {
        let mut cmp = Ordering::Less;
        for &encoded in values {
            let lat = doc_value_high(encoded);
            if lat < self.min_lat || lat > self.max_lat {
                continue;
            }
            let lon = doc_value_low(encoded);
            if (lon < self.min_lon || lon > self.max_lon) && lon < self.min_lon2 {
                continue;
            }
            let key = sloppy_math::haversin_sort_key(
                self.latitude,
                self.longitude,
                GeoEncodingUtils::decode_latitude(lat),
                GeoEncodingUtils::decode_longitude(lon),
            );
            cmp = cmp.max(double_compare(bottom, key));
            if cmp.is_gt() {
                return cmp;
            }
        }
        cmp
    }

    #[inline]
    fn sort_key(&self, values: &[i64]) -> f64 {
        let mut min = f64::INFINITY;
        for &encoded in values {
            let key = sloppy_math::haversin_sort_key(
                self.latitude,
                self.longitude,
                GeoEncodingUtils::decode_latitude(doc_value_high(encoded)),
                GeoEncodingUtils::decode_longitude(doc_value_low(encoded)),
            );
            min = java_min(min, key);
        }
        min
    }

    fn value(&self, key: f64) -> f64 {
        haversin2(key)
    }

    fn check(&self, r: &SegmentReader, field: &str) -> Result<()> {
        check_dv(r, field, "LatLonDocValuesField")
    }
}

/// `Math.min(double, double)`.
#[inline]
fn java_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_negative() { a } else { b };
    }
    if a <= b {
        a
    } else {
        b
    }
}

/// `XYPointDistanceComparator`'s state beyond the slots.
#[derive(Debug, Clone)]
struct XYDistance {
    x: f64,
    y: f64,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
    set_bottom_counter: i32,
}

impl XYDistance {
    fn new(x: f32, y: f32) -> Self {
        XYDistance {
            x: f64::from(x),
            y: f64::from(y),
            min_x: i32::MIN,
            max_x: i32::MAX,
            min_y: i32::MIN,
            max_y: i32::MAX,
            set_bottom_counter: 0,
        }
    }

    #[inline]
    fn distance(&self, encoded: i64) -> f64 {
        let dx = self.x - f64::from(XYEncodingUtils::decode(doc_value_high(encoded)));
        let dy = self.y - f64::from(XYEncodingUtils::decode(doc_value_low(encoded)));
        (dx * dx + dy * dy).sqrt()
    }
}

impl Distance for XYDistance {
    fn set_bottom(&mut self, bottom: f64) -> Result<()> {
        let c = self.set_bottom_counter;
        if bottom < f64::from(f32::MAX) && (c < 1024 || (c & 0x3F) == 0x3F) {
            let r = XYRectangle::from_point_distance(self.x as f32, self.y as f32, bottom as f32)
                .map_err(geo)?;
            self.min_x = XYEncodingUtils::encode(r.min_x).map_err(geo)?;
            self.max_x = XYEncodingUtils::encode(r.max_x).map_err(geo)?;
            self.min_y = XYEncodingUtils::encode(r.min_y).map_err(geo)?;
            self.max_y = XYEncodingUtils::encode(r.max_y).map_err(geo)?;
        }
        self.set_bottom_counter = c.saturating_add(1);
        Ok(())
    }

    #[inline]
    fn compare_bottom(&self, bottom: f64, values: &[i64]) -> Ordering {
        let mut cmp = Ordering::Less;
        for &encoded in values {
            let x = doc_value_high(encoded);
            if x < self.min_x || x > self.max_x {
                continue;
            }
            let y = doc_value_low(encoded);
            if y < self.min_y || y > self.max_y {
                continue;
            }
            cmp = cmp.max(double_compare(bottom, self.distance(encoded)));
            if cmp.is_gt() {
                return cmp;
            }
        }
        cmp
    }

    #[inline]
    fn sort_key(&self, values: &[i64]) -> f64 {
        values
            .iter()
            .fold(f64::INFINITY, |m, &v| java_min(m, self.distance(v)))
    }

    fn value(&self, key: f64) -> f64 {
        key
    }

    fn check(&self, r: &SegmentReader, field: &str) -> Result<()> {
        check_dv(r, field, "XYDocValuesField")
    }
}

/// `Geo3DPointDistanceComparator`'s state beyond the slots: the distance
/// shape and the bounds of the queue's bottom.
#[derive(Clone)]
struct Geo3DDistance {
    planet_model: Arc<PlanetModel>,
    shape: Arc<dyn GeoDistanceShape>,
    /// `priorityQueueBounds`: `None` until the first `setBottom`.
    bounds: Option<[f64; 6]>,
    set_bottom_counter: i32,
}

impl Geo3DDistance {
    fn decode(&self, encoded: i64) -> (f64, f64, f64) {
        let e = self.planet_model.doc_value_encoder();
        (
            e.decode_x_value(encoded),
            e.decode_y_value(encoded),
            e.decode_z_value(encoded),
        )
    }
}

/// An `XYZBounds` getter Java unboxes: an unset side (which a distance
/// shape's bounds never have) bounds nothing here, where Java would throw.
fn side(v: Option<f64>, unset: f64) -> f64 {
    v.unwrap_or(unset)
}

impl Distance for Geo3DDistance {
    fn set_bottom(&mut self, bottom: f64) -> Result<()> {
        let c = self.set_bottom_counter;
        if c < 1024 || (c & 0x3F) == 0x3F {
            let mut b = XYZBounds::new();
            self.shape
                .get_distance_bounds(&mut b, DistanceStyle::Arc, bottom)
                .map_err(s3d)?;
            self.bounds = Some([
                side(b.minimum_x(), f64::NEG_INFINITY),
                side(b.maximum_x(), f64::INFINITY),
                side(b.minimum_y(), f64::NEG_INFINITY),
                side(b.maximum_y(), f64::INFINITY),
                side(b.minimum_z(), f64::NEG_INFINITY),
                side(b.maximum_z(), f64::INFINITY),
            ]);
        }
        self.set_bottom_counter = c.saturating_add(1);
        Ok(())
    }

    fn compare_bottom(&self, bottom: f64, values: &[i64]) -> Ordering {
        let mut cmp = Ordering::Less;
        for &encoded in values {
            let (x, y, z) = self.decode(encoded);
            if let Some(b) = &self.bounds {
                if x > b[1] || x < b[0] || y > b[3] || y < b[2] || z > b[5] || z < b[4] {
                    continue;
                }
            }
            let d = self.shape.compute_distance(DistanceStyle::Arc, x, y, z);
            cmp = cmp.max(double_compare(bottom, d));
        }
        cmp
    }

    fn sort_key(&self, values: &[i64]) -> f64 {
        let mut min = f64::INFINITY;
        for &encoded in values {
            let (x, y, z) = self.decode(encoded);
            min = java_min(
                min,
                self.shape.compute_distance(DistanceStyle::Arc, x, y, z),
            );
        }
        min
    }

    fn value(&self, key: f64) -> f64 {
        key * self.planet_model.mean_radius()
    }

    fn check(&self, r: &SegmentReader, field: &str) -> Result<()> {
        check_dv(r, field, "Geo3DDocValuesField")
    }

    fn sort_key_checked(&self, values: &[i64]) -> Result<f64> {
        lucene_util::spatial3d::errors::catch(|| self.sort_key(values)).map_err(s3d)
    }
}

/// `Geo3DPointOutsideDistanceComparator`'s state: no bounds, the outside
/// distance of every value.
#[derive(Clone)]
struct Geo3DOutsideDistance {
    planet_model: Arc<PlanetModel>,
    shape: Arc<dyn GeoOutsideDistance>,
}

impl Geo3DOutsideDistance {
    fn distance(&self, encoded: i64) -> f64 {
        let e = self.planet_model.doc_value_encoder();
        self.shape.compute_outside_distance(
            DistanceStyle::Arc,
            e.decode_x_value(encoded),
            e.decode_y_value(encoded),
            e.decode_z_value(encoded),
        )
    }
}

impl Distance for Geo3DOutsideDistance {
    fn set_bottom(&mut self, _bottom: f64) -> Result<()> {
        Ok(())
    }

    fn compare_bottom(&self, bottom: f64, values: &[i64]) -> Ordering {
        let mut cmp = Ordering::Less;
        for &encoded in values {
            cmp = cmp.max(double_compare(bottom, self.distance(encoded)));
        }
        cmp
    }

    fn sort_key(&self, values: &[i64]) -> f64 {
        let mut min = f64::INFINITY;
        for &encoded in values {
            min = java_min(min, self.distance(encoded));
        }
        min
    }

    fn value(&self, key: f64) -> f64 {
        key * self.planet_model.mean_radius()
    }

    fn check(&self, r: &SegmentReader, field: &str) -> Result<()> {
        check_dv(r, field, "Geo3DDocValuesField")
    }

    fn sort_key_checked(&self, values: &[i64]) -> Result<f64> {
        lucene_util::spatial3d::errors::catch(|| self.sort_key(values)).map_err(s3d)
    }
}

fn s3d(e: lucene_util::spatial3d::Error) -> Error {
    illegal(e.to_string())
}

/// The segment's `FieldInfo` for `field`.
fn info_of<'a>(r: &'a SegmentReader, field: &str) -> Option<&'a FieldInfo> {
    r.field_infos().fields.iter().find(|f| f.name == field)
}

/// A slot of the hit queue: the key and the global doc, ordered worst
/// first (`FieldValueHitQueue`: larger key, then larger doc).
#[derive(Debug, Clone, Copy)]
struct Slot {
    key: f64,
    doc: i32,
}

impl PartialEq for Slot {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Slot {}
impl PartialOrd for Slot {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Slot {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key
            .total_cmp(&other.key)
            .then(self.doc.cmp(&other.doc))
    }
}

/// `TopFieldCollector` over one distance comparator.
struct DistanceTopN<D> {
    n: usize,
    queue: BinaryHeap<Slot>,
    distance: D,
    total: u64,
    error: Option<Error>,
}

impl<D: Distance> DistanceTopN<D> {
    fn bottom(&self) -> f64 {
        self.queue.peek().map_or(f64::INFINITY, |s| s.key)
    }

    /// `collect(doc)` with the document's values in hand.
    fn collect(&mut self, global_doc: i32, values: &[i64]) -> Result<()> {
        self.total = self.total.saturating_add(1);
        if self.queue.len() >= self.n {
            // `compareBottom(doc) <= 0`: not competitive (ties lose to the
            // earlier document).
            let bottom = self.bottom();
            let cmp = if values.is_empty() {
                double_compare(bottom, f64::INFINITY)
            } else {
                self.distance.compare_bottom(bottom, values)
            };
            if cmp.is_le() {
                return Ok(());
            }
            let key = self.key(values);
            if let Some(mut top) = self.queue.peek_mut() {
                *top = Slot {
                    key,
                    doc: global_doc,
                };
            }
            let b = self.bottom();
            self.distance.set_bottom(b)?;
        } else if self.n > 0 {
            let key = self.key(values);
            self.queue.push(Slot {
                key,
                doc: global_doc,
            });
            if self.queue.len() == self.n {
                let b = self.bottom();
                self.distance.set_bottom(b)?;
            }
        }
        Ok(())
    }

    fn key(&self, values: &[i64]) -> f64 {
        if values.is_empty() {
            f64::INFINITY
        } else {
            self.distance.sort_key(values)
        }
    }

    fn finish(self) -> SortedDistance {
        let distance = self.distance;
        SortedDistance {
            total_hits: self.total,
            hits: self
                .queue
                .into_sorted_vec()
                .into_iter()
                .map(|s| (s.doc, distance.value(s.key)))
                .collect(),
        }
    }
}

/// One leaf's view of a [`DistanceTopN`]: the query's matches arrive here.
struct LeafTopN<'a, 'v, D> {
    top: &'a mut DistanceTopN<D>,
    values: Option<GeoValues<'v>>,
    doc_base: i32,
    buf: Vec<i64>,
}

impl<D: Distance> ScoringCollector for LeafTopN<'_, '_, D> {
    fn collect(&mut self, doc_id: i32, _score: f32) {
        if self.top.error.is_some() {
            return;
        }
        self.buf.clear();
        if let Some(v) = self.values.as_mut() {
            if let Err(e) = v.values(doc_id, &mut self.buf) {
                self.top.error = Some(e);
                return;
            }
        }
        let global = self.doc_base.saturating_add(doc_id);
        if let Err(e) = self.top.collect(global, &self.buf) {
            self.top.error = Some(e);
        }
    }

    fn score_mode(&self) -> ScoreMode {
        ScoreMode::CompleteNoScores
    }
}

/// `IndexSearcher.search(query, n, sort)` with one distance key.
fn search_distance<D: Distance>(
    leaves: &[OpenSegment<'_>],
    query: &dyn DocumentQuery,
    field: &str,
    n: usize,
    distance: D,
) -> Result<SortedDistance> {
    // `TopFieldCollectorManager`'s constructor, before anything is searched
    // (`searchAfter` caps `n` at `max(1, maxDoc)` first, which leaves 0 at 0).
    if n == 0 {
        return Err(illegal(
            "numHits must be > 0; please use TotalHitCountCollector if you just need the total hit count",
        ));
    }
    let rewritten = crate::document::rewrite(query, leaves)?;
    let query: &dyn DocumentQuery = rewritten.as_deref().unwrap_or(query);
    let mut top = DistanceTopN {
        n,
        queue: BinaryHeap::with_capacity(n.min(1 << 16)),
        distance,
        total: 0,
        error: None,
    };
    for leaf in leaves {
        let r = reader(leaf)?;
        top.distance.check(r, field)?;
        let values = match info_of(r, field) {
            Some(info) => sorted_numeric_in(r, info)?,
            None => None,
        };
        let mut lc = LeafTopN {
            top: &mut top,
            values,
            doc_base: leaf.doc_base,
            buf: Vec::new(),
        };
        query.score_leaf(leaf, 1.0, &mut lc)?;
        if let Some(e) = top.error.take() {
            return Err(e);
        }
    }
    Ok(top.finish())
}

/// `LatLonPointSortField` (`LatLonDocValuesField.newDistanceSort`): by the
/// distance of a document's closest `LatLonDocValuesField` value to an
/// origin, nearest first; documents without one last.
#[derive(Debug, Clone, PartialEq)]
pub struct LatLonPointSortField {
    pub field: String,
    pub latitude: f64,
    pub longitude: f64,
}

impl LatLonPointSortField {
    /// `LatLonPointSortField(field, latitude, longitude)`.
    ///
    /// # Errors
    /// An invalid origin, with Java's message.
    pub fn new(field: impl Into<String>, latitude: f64, longitude: f64) -> Result<Self> {
        GeoUtils::check_latitude(latitude).map_err(geo)?;
        GeoUtils::check_longitude(longitude).map_err(geo)?;
        Ok(LatLonPointSortField {
            field: field.into(),
            latitude,
            longitude,
        })
    }

    /// `setMissingValue(missingValue)`: only `+Infinity` (missing last).
    ///
    /// # Errors
    /// Any other value, with Java's message.
    pub fn set_missing_value(&mut self, missing: f64) -> Result<()> {
        check_missing(missing)
    }

    /// `IndexSearcher.search(query, n, new Sort(this))`.
    ///
    /// # Errors
    /// `n == 0` (Java's `numHits must be > 0`), the field's doc values are
    /// of another type, or the index does not decode.
    pub fn search(
        &self,
        leaves: &[OpenSegment<'_>],
        query: &dyn DocumentQuery,
        n: usize,
    ) -> Result<SortedDistance> {
        search_distance(
            leaves,
            query,
            &self.field,
            n,
            LatLonDistance::new(self.latitude, self.longitude),
        )
    }

    /// This sort as a `CUSTOM` key's comparator source, for
    /// [`crate::top_field::register_comparator_source`].
    pub fn comparator_source(&self) -> Arc<dyn FieldComparatorSource> {
        Arc::new(Source::LatLon(self.clone()))
    }
}

/// `XYPointSortField` (`XYDocValuesField.newDistanceSort`): by the
/// cartesian distance of a document's closest `XYDocValuesField` value to
/// an origin, nearest first; documents without one last.
#[derive(Debug, Clone, PartialEq)]
pub struct XYPointSortField {
    pub field: String,
    pub x: f32,
    pub y: f32,
}

impl XYPointSortField {
    /// `XYPointSortField(field, x, y)`.
    pub fn new(field: impl Into<String>, x: f32, y: f32) -> Self {
        XYPointSortField {
            field: field.into(),
            x,
            y,
        }
    }

    /// `setMissingValue(missingValue)`: only `+Infinity` (missing last).
    ///
    /// # Errors
    /// Any other value, with Java's message.
    pub fn set_missing_value(&mut self, missing: f64) -> Result<()> {
        check_missing(missing)
    }

    /// `IndexSearcher.search(query, n, new Sort(this))`.
    ///
    /// # Errors
    /// `n == 0` (Java's `numHits must be > 0`), the field's doc values are
    /// of another type, or the index does not decode.
    pub fn search(
        &self,
        leaves: &[OpenSegment<'_>],
        query: &dyn DocumentQuery,
        n: usize,
    ) -> Result<SortedDistance> {
        search_distance(
            leaves,
            query,
            &self.field,
            n,
            XYDistance::new(self.x, self.y),
        )
    }

    /// This sort as a `CUSTOM` key's comparator source.
    pub fn comparator_source(&self) -> Arc<dyn FieldComparatorSource> {
        Arc::new(Source::XY(self.clone()))
    }
}

/// `Geo3DPointSortField` (`Geo3DDocValuesField.newDistanceSort` /
/// `newPathSort`): by the arc distance (`computeDistance`) of a document's
/// closest `Geo3DDocValuesField` value inside a distance shape, reported in
/// meters (times the planet's mean radius); documents without a value or
/// outside the shape last, at `f64::INFINITY`.
#[derive(Clone)]
pub struct Geo3DPointSortField {
    pub field: String,
    pub planet_model: Arc<PlanetModel>,
    pub shape: Arc<dyn GeoDistanceShape>,
}

impl std::fmt::Debug for Geo3DPointSortField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Geo3DPointSortField({:?})", self.field)
    }
}

impl Geo3DPointSortField {
    /// `Geo3DPointSortField(field, planetModel, distanceShape)`.
    pub fn new(
        field: impl Into<String>,
        planet_model: &Arc<PlanetModel>,
        shape: Arc<dyn GeoDistanceShape>,
    ) -> Self {
        Geo3DPointSortField {
            field: field.into(),
            planet_model: planet_model.clone(),
            shape,
        }
    }

    /// `setMissingValue(missingValue)`: only `+Infinity` (missing last).
    ///
    /// # Errors
    /// Any other value, with Java's message.
    pub fn set_missing_value(&mut self, missing: f64) -> Result<()> {
        check_missing(missing)
    }

    fn distance(&self) -> Geo3DDistance {
        Geo3DDistance {
            planet_model: self.planet_model.clone(),
            shape: self.shape.clone(),
            bounds: None,
            set_bottom_counter: 0,
        }
    }

    /// `IndexSearcher.search(query, n, new Sort(this))`.
    ///
    /// # Errors
    /// `n == 0`, the field's doc values are of another type, the index does
    /// not decode, or a shape method throws (as Java's would).
    pub fn search(
        &self,
        leaves: &[OpenSegment<'_>],
        query: &dyn DocumentQuery,
        n: usize,
    ) -> Result<SortedDistance> {
        let d = self.distance();
        caught(|| search_distance(leaves, query, &self.field, n, d))
    }

    /// This sort as a `CUSTOM` key's comparator source.
    pub fn comparator_source(&self) -> Arc<dyn FieldComparatorSource> {
        Arc::new(Source::Geo3D(self.clone()))
    }
}

/// `Geo3DPointOutsideSortField` (`Geo3DDocValuesField.newOutside*Sort`):
/// by the arc distance of a document's closest value to the outside of a
/// shape (0 inside), in meters; documents without a value last.
#[derive(Clone)]
pub struct Geo3DPointOutsideSortField {
    pub field: String,
    pub planet_model: Arc<PlanetModel>,
    pub shape: Arc<dyn GeoOutsideDistance>,
}

impl std::fmt::Debug for Geo3DPointOutsideSortField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Geo3DPointOutsideSortField({:?})", self.field)
    }
}

impl Geo3DPointOutsideSortField {
    /// `Geo3DPointOutsideSortField(field, planetModel, distanceShape)`.
    pub fn new(
        field: impl Into<String>,
        planet_model: &Arc<PlanetModel>,
        shape: Arc<dyn GeoOutsideDistance>,
    ) -> Self {
        Geo3DPointOutsideSortField {
            field: field.into(),
            planet_model: planet_model.clone(),
            shape,
        }
    }

    /// `setMissingValue(missingValue)`: only `+Infinity` (missing last).
    ///
    /// # Errors
    /// Any other value, with Java's message.
    pub fn set_missing_value(&mut self, missing: f64) -> Result<()> {
        check_missing(missing)
    }

    fn distance(&self) -> Geo3DOutsideDistance {
        Geo3DOutsideDistance {
            planet_model: self.planet_model.clone(),
            shape: self.shape.clone(),
        }
    }

    /// `IndexSearcher.search(query, n, new Sort(this))`.
    ///
    /// # Errors
    /// As [`Geo3DPointSortField::search`].
    pub fn search(
        &self,
        leaves: &[OpenSegment<'_>],
        query: &dyn DocumentQuery,
        n: usize,
    ) -> Result<SortedDistance> {
        let d = self.distance();
        caught(|| search_distance(leaves, query, &self.field, n, d))
    }

    /// This sort as a `CUSTOM` key's comparator source.
    pub fn comparator_source(&self) -> Arc<dyn FieldComparatorSource> {
        Arc::new(Source::Geo3DOutside(self.clone()))
    }
}

/// Runs `f` under `spatial3d::errors::catch`: an exception a shape method
/// raised while it ran is the search's error, as Java's would propagate.
fn caught<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    lucene_util::spatial3d::errors::catch(f).map_err(s3d)?
}

fn check_missing(missing: f64) -> Result<()> {
    if missing == f64::INFINITY {
        return Ok(());
    }
    Err(illegal(format!(
        "Missing value can only be Double.POSITIVE_INFINITY (missing values last), but got {}",
        lucene_util::geo::java_double_string(missing)
    )))
}

/// The two sorts as comparator sources.
#[derive(Debug, Clone)]
enum Source {
    LatLon(LatLonPointSortField),
    XY(XYPointSortField),
    Geo3D(Geo3DPointSortField),
    Geo3DOutside(Geo3DPointOutsideSortField),
}

impl FieldComparatorSource for Source {
    fn new_comparator(
        &self,
        field: &str,
        _num_hits: usize,
        _reverse: bool,
    ) -> Box<dyn FieldComparator> {
        Box::new(Comparator {
            field: field.to_string(),
            source: self.clone(),
        })
    }
}

/// The comparator a [`Source`] makes: values are the comparable longs of
/// the sort keys.
struct Comparator {
    field: String,
    source: Source,
}

impl FieldComparator for Comparator {
    fn leaf<'a>(&self, ctx: LeafCtx<'a>) -> Result<Box<dyn LeafFieldComparator + 'a>> {
        let r = ctx.reader;
        let distance: Box<dyn Distance> = match &self.source {
            Source::LatLon(s) => Box::new(LatLonDistance::new(s.latitude, s.longitude)),
            Source::XY(s) => Box::new(XYDistance::new(s.x, s.y)),
            Source::Geo3D(s) => Box::new(s.distance()),
            Source::Geo3DOutside(s) => Box::new(s.distance()),
        };
        distance.check(r, &self.field)?;
        let values = GeoColumn::open(r, &self.field)?;
        Ok(Box::new(LeafComparator {
            // Lucene's own `compareBottom` (the bottom's bounding box) for
            // the lat/lon sort, the one the plugin runs.
            bounded: matches!(self.source, Source::LatLon(_)),
            distance,
            values,
            buf: Vec::new(),
        }))
    }

    fn compare_values(&self, a: &SortValue, b: &SortValue) -> Ordering {
        match (a, b) {
            (SortValue::Long(a), SortValue::Long(b)) => a.cmp(b),
            _ => Ordering::Equal,
        }
    }
}

struct LeafComparator<'a> {
    /// Whether `compareBottom` is the distance's own ([`Distance::compare_bottom`]).
    bounded: bool,
    distance: Box<dyn Distance>,
    values: GeoColumn<'a>,
    buf: Vec<i64>,
}

impl LeafComparator<'_> {
    fn read(&mut self, doc: i32) -> Result<()> {
        self.values.read(doc, &mut self.buf)
    }
}

/// A segment's `SORTED_NUMERIC` geo points for a distance comparator: read
/// per document (`DocValues.getSortedNumeric`), or -- in a segment of
/// 10,000 documents or more, from a sort's second use on -- from the
/// segment's decoded copy ([`crate::exec::cache::SortColumn::Multi`], the
/// cache numeric sort keys use: the same values, decoded once).
enum GeoColumn<'a> {
    Live(Option<Box<GeoValues<'a>>>),
    Decoded(Arc<crate::exec::cache::SortColumn>),
}

impl<'a> GeoColumn<'a> {
    /// `field`'s points in `r`: none without the field, an error for one of
    /// another doc-values type.
    fn open(r: &'a SegmentReader, field: &str) -> Result<Self> {
        let Some(info) = info_of(r, field) else {
            return Ok(GeoColumn::Live(None));
        };
        let Some(values) = sorted_numeric_in(r, info)? else {
            return Ok(GeoColumn::Live(None));
        };
        let key = format!("geo\0{field}");
        let built = r.query_cache().sort_column(&key, r.max_doc, &mut || {
            let mut fresh = sorted_numeric_in(r, info)?;
            let n = idx(r.max_doc);
            let mut starts = Vec::with_capacity(n + 1);
            let mut all = Vec::new();
            let mut buf = Vec::new();
            for d in 0..r.max_doc {
                starts.push(all.len());
                if let Some(v) = fresh.as_mut() {
                    v.values(d, &mut buf)?;
                    all.extend_from_slice(&buf);
                }
            }
            starts.push(all.len());
            Ok(crate::exec::cache::SortColumn::Multi {
                starts,
                values: all,
            })
        })?;
        Ok(match built {
            Some(c) => GeoColumn::Decoded(c),
            None => GeoColumn::Live(Some(Box::new(values))),
        })
    }

    /// Replaces `out` with `doc`'s values.
    #[inline]
    fn read(&mut self, doc: i32, out: &mut Vec<i64>) -> Result<()> {
        out.clear();
        match self {
            GeoColumn::Live(Some(v)) => v.values(doc, out)?,
            GeoColumn::Live(None) => {}
            GeoColumn::Decoded(c) => {
                if let crate::exec::cache::SortColumn::Multi { starts, values } = &**c {
                    let d = idx(doc);
                    if let (Some(&a), Some(&b)) = (starts.get(d), starts.get(d + 1)) {
                        out.extend_from_slice(values.get(a..b).unwrap_or(&[]));
                    }
                }
            }
        }
        Ok(())
    }
}

/// A key's sort value back as the double it encodes.
fn key_of(v: &SortValue) -> Option<f64> {
    match v {
        SortValue::Long(l) => Some(lucene_util::numeric_utils::sortable_long_to_double(*l)),
        SortValue::Bytes(_) => None,
    }
}

impl LeafFieldComparator for LeafComparator<'_> {
    fn value(&mut self, doc: i32, _score: f32) -> Result<SortValue> {
        self.read(doc)?;
        let key = if self.buf.is_empty() {
            f64::INFINITY
        } else {
            self.distance.sort_key_checked(&self.buf)?
        };
        Ok(SortValue::Long(double_to_sortable_long(key)))
    }

    /// `setBottom(slot)`: the bottom's bounding box.
    fn set_bottom(&mut self, bottom: &SortValue) -> Result<()> {
        match key_of(bottom) {
            Some(key) if self.bounded => self.distance.set_bottom(key),
            _ => Ok(()),
        }
    }

    /// `compareBottom(doc)`: a value outside the bottom's box is not
    /// measured, and a document without one is `+Infinity`.
    fn compare_bottom(
        &mut self,
        bottom: &SortValue,
        doc: i32,
        _score: f32,
    ) -> Result<Option<Ordering>> {
        let Some(bottom) = key_of(bottom).filter(|_| self.bounded) else {
            return Ok(None);
        };
        self.read(doc)?;
        if self.buf.is_empty() {
            return Ok(Some(double_compare(bottom, f64::INFINITY)));
        }
        Ok(Some(self.distance.compare_bottom(bottom, &self.buf)))
    }
}

/// OpenSearch's `_geo_distance` sort where it does not hand the sort to
/// Lucene (several origins, a mode other than `min`, a unit other than
/// metres, descending): `GeoDistanceSortBuilder`'s own
/// `XFieldComparatorSource` -- a `DoubleComparator` over
/// `MultiValueMode.select(GeoUtils.distanceValues(ARC, unit, values,
/// origins))` with missing documents at `+Infinity`.
///
/// Per document: every value of the field (decoded as OpenSearch's
/// `LatLonPointDVLeafFieldData` decodes it) against every origin, values
/// outer and origins inner, each `SloppyMath.haversinMeters(origin, point)`
/// divided by the unit's metres (`DistanceUnit.convert`), the lot sorted
/// (`SortingNumericDoubleValues`), then the mode's pick. Only `GeoDistance.ARC`:
/// `PLANE` goes through `Math.cos`, whose HotSpot intrinsic no portable
/// code reproduces bit for bit, so the plugin leaves it to OpenSearch. The
/// comparator is built with no field, so it never skips; nor does this.
///
/// A hit's value is `double_to_sortable_long` of the distance, compared as
/// `Double.compare` compares the doubles (`search_after` included).
#[derive(Debug, Clone, PartialEq)]
pub struct OpenSearchGeoDistanceSort {
    /// `(lat, lon)` origins, at least one.
    pub origins: Vec<(f64, f64)>,
    /// The unit's metres (`DistanceUnit.meters`; 1 for metres).
    pub unit_meters: f64,
    pub mode: DistanceMode,
}

/// OpenSearch's `MultiValueMode` over a document's sorted distances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMode {
    Min,
    Max,
    Sum,
    Avg,
    Median,
}

impl OpenSearchGeoDistanceSort {
    /// A document's value from its encoded points; `+Infinity` (OpenSearch's
    /// `replaceMissing`) for a document without one.
    pub fn value(&self, encoded: &[i64], scratch: &mut Vec<f64>) -> f64 {
        if encoded.is_empty() {
            return f64::INFINITY;
        }
        scratch.clear();
        for &e in encoded {
            // `(int) (encoded >>> 32)` and `(int) encoded`.
            let lat = GeoEncodingUtils::decode_latitude(doc_value_high(e));
            let lon = GeoEncodingUtils::decode_longitude(doc_value_low(e));
            for &(o_lat, o_lon) in &self.origins {
                scratch
                    .push(sloppy_math::haversin_meters(o_lat, o_lon, lat, lon) / self.unit_meters);
            }
        }
        // `Arrays.sort(double[])`: `Double.compare`'s order.
        scratch.sort_by(|a, b| double_compare(*a, *b));
        let n = scratch.len();
        match self.mode {
            DistanceMode::Min => scratch[0],
            DistanceMode::Max => scratch[n - 1],
            DistanceMode::Sum => scratch.iter().fold(0.0, |t, v| t + v),
            // `total / count`, the count an int widened to double.
            DistanceMode::Avg => scratch.iter().fold(0.0, |t, v| t + v) / n as f64,
            DistanceMode::Median => {
                let mid = (n - 1) / 2;
                if n.is_multiple_of(2) {
                    (scratch[mid] + scratch[mid + 1]) / 2.0
                } else {
                    scratch[mid]
                }
            }
        }
    }
}

impl FieldComparatorSource for OpenSearchGeoDistanceSort {
    fn new_comparator(
        &self,
        field: &str,
        _num_hits: usize,
        _reverse: bool,
    ) -> Box<dyn FieldComparator> {
        Box::new(OpenSearchComparator {
            field: field.to_string(),
            sort: self.clone(),
        })
    }
}

struct OpenSearchComparator {
    field: String,
    sort: OpenSearchGeoDistanceSort,
}

impl FieldComparator for OpenSearchComparator {
    fn leaf<'a>(&self, ctx: LeafCtx<'a>) -> Result<Box<dyn LeafFieldComparator + 'a>> {
        // `DocValues.getSortedNumeric(reader, field)`: empty without the
        // field, an error for another doc-values type.
        let values = GeoColumn::open(ctx.reader, &self.field)?;
        Ok(Box::new(OpenSearchLeaf {
            sort: self.sort.clone(),
            values,
            buf: Vec::new(),
            scratch: Vec::new(),
        }))
    }

    fn compare_values(&self, a: &SortValue, b: &SortValue) -> Ordering {
        match (a, b) {
            (SortValue::Long(a), SortValue::Long(b)) => a.cmp(b),
            _ => Ordering::Equal,
        }
    }
}

struct OpenSearchLeaf<'a> {
    sort: OpenSearchGeoDistanceSort,
    values: GeoColumn<'a>,
    buf: Vec<i64>,
    scratch: Vec<f64>,
}

impl LeafFieldComparator for OpenSearchLeaf<'_> {
    fn value(&mut self, doc: i32, _score: f32) -> Result<SortValue> {
        self.values.read(doc, &mut self.buf)?;
        let d = self.sort.value(&self.buf, &mut self.scratch);
        Ok(SortValue::Long(double_to_sortable_long(d)))
    }

    /// `DoubleComparator.compareBottom`: the bottom against the document's
    /// value (`Double.compare`, as the sortable longs compare).
    fn compare_bottom(
        &mut self,
        bottom: &SortValue,
        doc: i32,
        score: f32,
    ) -> Result<Option<Ordering>> {
        let SortValue::Long(bottom) = bottom else {
            return Ok(None);
        };
        let SortValue::Long(v) = self.value(doc, score)? else {
            return Ok(None);
        };
        Ok(Some(bottom.cmp(&v)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_index::document::{LatLonDocValuesField, XYDocValuesField};

    #[test]
    fn missing_values_and_validation() {
        let mut s = LatLonPointSortField::new("f", 1.0, 2.0).unwrap();
        assert!(s.set_missing_value(f64::INFINITY).is_ok());
        let e = s.set_missing_value(0.0).unwrap_err();
        assert!(e.to_string().contains("but got 0.0"), "{e}");
        assert!(LatLonPointSortField::new("f", 100.0, 0.0).is_err());
        assert!(LatLonPointSortField::new("f", 0.0, -200.0).is_err());
        let mut x = XYPointSortField::new("f", 1.0, 2.0);
        assert!(x.set_missing_value(f64::INFINITY).is_ok());
        assert!(x.set_missing_value(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn helpers_follow_java() {
        assert_eq!(double_compare(1.0, 2.0), Ordering::Less);
        assert_eq!(double_compare(2.0, 2.0), Ordering::Equal);
        assert_eq!(double_compare(f64::INFINITY, 2.0), Ordering::Greater);
        assert!(java_min(f64::NAN, 1.0).is_nan());
        assert!(java_min(0.0, -0.0).is_sign_negative());
        assert!(java_min(-0.0, 0.0).is_sign_negative());
        assert_eq!(java_min(3.0, 2.0), 2.0);
        assert_eq!(haversin2(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn top_n_keeps_the_nearest_and_breaks_ties_by_doc() {
        let v = |lat, lon| LatLonDocValuesField::encode(lat, lon).unwrap();
        let mut top = DistanceTopN {
            n: 2,
            queue: BinaryHeap::new(),
            distance: LatLonDistance::new(0.0, 0.0),
            total: 0,
            error: None,
        };
        top.collect(0, &[v(0.0, 3.0)]).unwrap();
        top.collect(1, &[]).unwrap();
        top.collect(2, &[v(0.0, 1.0), v(0.0, 50.0)]).unwrap();
        top.collect(3, &[v(0.0, 1.0)]).unwrap();
        top.collect(4, &[v(0.0, 2.0)]).unwrap();
        top.collect(5, &[]).unwrap();
        let r = top.finish();
        assert_eq!(r.total_hits, 6);
        assert_eq!(r.hits.iter().map(|h| h.0).collect::<Vec<_>>(), vec![2, 3]);
        assert!((r.hits[0].1 - 111_195.0).abs() < 100.0, "{:?}", r.hits);

        let mut xy = DistanceTopN {
            n: 1,
            queue: BinaryHeap::new(),
            distance: XYDistance::new(0.0, 0.0),
            total: 0,
            error: None,
        };
        let p = |x, y| XYDocValuesField::encode(x, y).unwrap();
        xy.collect(0, &[]).unwrap();
        xy.collect(1, &[p(3.0, 4.0)]).unwrap();
        xy.collect(2, &[p(30.0, 40.0), p(0.0, 1.0)]).unwrap();
        xy.collect(3, &[p(100.0, 100.0)]).unwrap();
        let r = xy.finish();
        assert_eq!(r.hits, vec![(2, 1.0)]);
        let empty = DistanceTopN {
            n: 0,
            queue: BinaryHeap::new(),
            distance: XYDistance::new(0.0, 0.0),
            total: 0,
            error: None,
        };
        assert!(empty.finish().hits.is_empty());
    }

    #[test]
    fn geo3d_sorts_and_their_distances() {
        use crate::document::geo::geo3d::{from_distance, from_polygon};
        use lucene_util::geo::Polygon;
        let pm = PlanetModel::wgs84();
        let circle = from_distance(&pm, 0.0, 0.0, 100_000.0).unwrap();
        let mut s = Geo3DPointSortField::new("f", &pm, circle.clone());
        assert!(s.set_missing_value(f64::INFINITY).is_ok());
        assert!(s.set_missing_value(1.0).is_err());
        assert_eq!(format!("{s:?}"), "Geo3DPointSortField(\"f\")");
        let square = Polygon::new(
            &[0.0, 1.0, 1.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0, 1.0, 0.0],
            vec![],
        )
        .unwrap();
        let polygon = from_polygon(&pm, &[square]).unwrap();
        let mut o = Geo3DPointOutsideSortField::new("f", &pm, polygon);
        assert!(o.set_missing_value(f64::INFINITY).is_ok());
        assert!(o.set_missing_value(f64::NEG_INFINITY).is_err());
        assert_eq!(format!("{o:?}"), "Geo3DPointOutsideSortField(\"f\")");

        let enc = pm.doc_value_encoder();
        let value = |lat: f64, lon: f64| {
            let p = lucene_util::spatial3d::GeoPoint::from_lat_lon(
                &pm,
                lat.to_radians(),
                lon.to_radians(),
            )
            .unwrap();
            enc.encode_point(&p).unwrap()
        };
        // Past the bottom's bounds (here the circle's own) a value is
        // skipped, not computed.
        let mut d = s.distance();
        d.set_bottom(1e-4).unwrap();
        assert!(d.bounds.is_some_and(|b| b[3] < 0.05), "{:?}", d.bounds);
        assert_eq!(d.compare_bottom(1e-4, &[value(0.0, 5.0)]), Ordering::Less);
        assert_eq!(
            d.compare_bottom(1e-4, &[value(0.0, 0.0001)]),
            Ordering::Greater
        );
        let key = d
            .sort_key_checked(&[value(0.0, 0.5), value(0.0, 0.0)])
            .unwrap();
        assert!(key < 1e-5, "{key}");
        let od = o.distance();
        assert_eq!(od.sort_key_checked(&[value(0.5, 0.5)]).unwrap(), 0.0);
        assert!(od.sort_key_checked(&[value(5.0, 5.0)]).unwrap() > 0.0);
        let e = s3d(lucene_util::spatial3d::Error::Runtime("boom".into()));
        assert!(e.to_string().contains("boom"), "{e}");
    }

    /// OpenSearch's comparator: every value against every origin, sorted,
    /// then the mode's pick; missing documents at `+Infinity`.
    #[test]
    fn opensearch_geo_distance_modes() {
        let v = |lat, lon| LatLonDocValuesField::encode(lat, lon).unwrap();
        let point = |e: i64| {
            (
                GeoEncodingUtils::decode_latitude(doc_value_high(e)),
                GeoEncodingUtils::decode_longitude(doc_value_low(e)),
            )
        };
        let values = [v(0.0, 1.0), v(0.0, 3.0), v(10.0, 0.0)];
        let origins = vec![(0.0, 0.0), (0.0, 2.0)];
        let mut want: Vec<f64> = values
            .iter()
            .flat_map(|&e| {
                let (lat, lon) = point(e);
                origins
                    .iter()
                    .map(move |&(a, b)| sloppy_math::haversin_meters(a, b, lat, lon) / 1000.0)
            })
            .collect();
        want.sort_by(f64::total_cmp);
        let sort = |mode| OpenSearchGeoDistanceSort {
            origins: origins.clone(),
            unit_meters: 1000.0,
            mode,
        };
        let mut scratch = Vec::new();
        let at = |mode| sort(mode).value(&values, &mut Vec::new());
        assert_eq!(at(DistanceMode::Min), want[0]);
        assert_eq!(at(DistanceMode::Max), want[5]);
        let sum = want.iter().fold(0.0, |t, v| t + v);
        assert_eq!(at(DistanceMode::Sum), sum);
        assert_eq!(at(DistanceMode::Avg), sum / 6.0);
        assert_eq!(at(DistanceMode::Median), (want[2] + want[3]) / 2.0);
        // Two values: the mean of both; three (below): the middle one.
        let odd = sort(DistanceMode::Median).value(&values[..1], &mut scratch);
        let (lat, lon) = point(values[0]);
        let mut two = [
            sloppy_math::haversin_meters(0.0, 0.0, lat, lon) / 1000.0,
            sloppy_math::haversin_meters(0.0, 2.0, lat, lon) / 1000.0,
        ];
        two.sort_by(f64::total_cmp);
        assert_eq!(odd, (two[0] + two[1]) / 2.0);
        let one = OpenSearchGeoDistanceSort {
            origins: vec![(0.0, 0.0)],
            unit_meters: 1.0,
            mode: DistanceMode::Median,
        };
        assert_eq!(
            one.value(&values, &mut scratch),
            sloppy_math::haversin_meters(0.0, 0.0, point(values[1]).0, point(values[1]).1)
        );
        assert_eq!(one.value(&[], &mut scratch), f64::INFINITY);
    }

    /// The comparator over a real segment: values as sortable longs, a
    /// document without the field at `+Infinity`, a field of another
    /// doc-values type an error.
    #[test]
    fn opensearch_geo_distance_comparator_reads_the_segment() {
        use crate::directory_reader::DirectoryReader;
        use lucene_index::document::{self as d, Document};
        use lucene_index::index_writer::IndexWriter;
        use lucene_index::segment_info::LuceneVersion;
        use lucene_store::FsDirectory;
        use lucene_util::test_support::TempDir;
        let tmp = TempDir::new("os-geo-sort");
        let dir = FsDirectory::open(tmp.path());
        let version = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
        for lat in [Some(1.0), None, Some(-3.0)] {
            let mut doc = Document::new();
            if let Some(lat) = lat {
                doc.add_boxed(Box::new(
                    d::LatLonDocValuesField::new("loc", lat, 0.0).unwrap(),
                ));
            }
            doc.add_boxed(Box::new(d::SortedDocValuesField::new("kw", "x")));
            w.add_fields_document(&doc).unwrap();
        }
        w.commit().unwrap();
        drop(w);
        let r = DirectoryReader::open(&dir).unwrap();
        let seg = &r.segment_readers()[0];
        let sort = OpenSearchGeoDistanceSort {
            origins: vec![(0.0, 0.0)],
            unit_meters: 1.0,
            mode: DistanceMode::Min,
        };
        let src = sort.clone();
        let cmp = src.new_comparator("loc", 3, false);
        let mut leaf = cmp
            .leaf(LeafCtx {
                reader: seg,
                doc_base: 0,
            })
            .unwrap();
        let got: Vec<SortValue> = (0..3).map(|d| leaf.value(d, 0.0).unwrap()).collect();
        let SortValue::Long(missing) = got[1] else {
            panic!("{got:?}")
        };
        assert_eq!(missing, double_to_sortable_long(f64::INFINITY));
        assert_eq!(cmp.compare_values(&got[0], &got[2]), Ordering::Less);
        assert_eq!(
            cmp.compare_values(&got[0], &SortValue::Bytes(None)),
            Ordering::Equal
        );
        // No such field: every document missing.
        let mut none = src
            .new_comparator("nope", 3, false)
            .leaf(LeafCtx {
                reader: seg,
                doc_base: 0,
            })
            .unwrap();
        assert_eq!(none.value(0, 0.0).unwrap(), SortValue::Long(missing));
        // A field of another doc-values type.
        assert!(src
            .new_comparator("kw", 3, false)
            .leaf(LeafCtx {
                reader: seg,
                doc_base: 0,
            })
            .is_err());
    }

    /// In a segment of 10,000 documents a distance comparator reads its
    /// points from the segment's decoded copy from the second use on: the
    /// same values as the per-document read, documents without any
    /// included; and `compareBottom` answers as `value` compares.
    #[test]
    fn decoded_points_column_reads_as_the_doc_values() {
        use crate::directory_reader::DirectoryReader;
        use lucene_index::document::{self as d, Document};
        use lucene_index::index_writer::IndexWriter;
        use lucene_index::segment_info::LuceneVersion;
        use lucene_store::FsDirectory;
        use lucene_util::test_support::TempDir;
        let tmp = TempDir::new("geo-sort-column");
        let dir = FsDirectory::open(tmp.path());
        let version = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
        for i in 0..10_050u32 {
            let mut doc = Document::new();
            for k in 0..(i % 3) {
                let lat = f64::from((i * 7 + k) % 170) - 85.0;
                let lon = f64::from((i * 13 + k) % 350) - 175.0;
                doc.add_boxed(Box::new(
                    d::LatLonDocValuesField::new("loc", lat, lon).unwrap(),
                ));
            }
            w.add_fields_document(&doc).unwrap();
        }
        w.commit().unwrap();
        drop(w);
        let r = DirectoryReader::open(&dir).unwrap();
        let seg = &r.segment_readers()[0];
        assert!(seg.max_doc >= 10_000);
        let ctx = LeafCtx {
            reader: seg,
            doc_base: 0,
        };
        let lucene = LatLonPointSortField::new("loc", 10.0, 20.0)
            .unwrap()
            .comparator_source()
            .new_comparator("loc", 10, false);
        let os = OpenSearchGeoDistanceSort {
            origins: vec![(10.0, 20.0), (-5.0, 170.0)],
            unit_meters: 1609.344,
            mode: DistanceMode::Avg,
        };
        let os = os.new_comparator("loc", 10, false);
        for cmp in [lucene, os] {
            let all = |cmp: &dyn FieldComparator| {
                let mut leaf = cmp.leaf(ctx).unwrap();
                (0..seg.max_doc)
                    .map(|d| leaf.value(d, 0.0).unwrap())
                    .collect::<Vec<_>>()
            };
            // First use: per document; second: the column is built; third:
            // read from it.
            let first = all(cmp.as_ref());
            assert_eq!(all(cmp.as_ref()), first);
            assert_eq!(all(cmp.as_ref()), first);
            assert!(matches!(
                GeoColumn::open(seg, "loc").unwrap(),
                GeoColumn::Decoded(_)
            ));
            // `compareBottom` against a middling bottom, as `value` compares.
            let bottom = first[5000].clone();
            let mut leaf = cmp.leaf(ctx).unwrap();
            leaf.set_bottom(&bottom).unwrap();
            for d in (0..seg.max_doc).step_by(7) {
                let want = cmp.compare_values(&bottom, &first[idx(d)]);
                let got = leaf.compare_bottom(&bottom, d, 0.0).unwrap().unwrap();
                // Outside the bottom's box Lucene's comparator answers
                // "not competitive" unmeasured, which `value` agrees with.
                assert_eq!(got, want, "doc {d}");
            }
            assert_eq!(
                leaf.compare_bottom(&SortValue::Bytes(None), 0, 0.0)
                    .unwrap(),
                None
            );
        }
        // The other distance sorts compare by value.
        let xy = XYPointSortField::new("nope", 1.0, 2.0)
            .comparator_source()
            .new_comparator("nope", 10, false);
        let mut leaf = xy.leaf(ctx).unwrap();
        let bottom = SortValue::Long(0);
        leaf.set_bottom(&bottom).unwrap();
        assert_eq!(leaf.compare_bottom(&bottom, 0, 0.0).unwrap(), None);
    }

    #[test]
    fn lat_lon_bottom_box_across_the_dateline() {
        let mut d = LatLonDistance::new(0.0, 179.9);
        let key = sloppy_math::haversin_sort_key(0.0, 179.9, 0.0, -179.0);
        d.set_bottom(key).unwrap();
        assert_eq!(d.min_lon, i32::MIN);
        assert_ne!(d.min_lon2, i32::MAX);
        let v = |lat, lon| LatLonDocValuesField::encode(lat, lon).unwrap();
        assert_eq!(d.compare_bottom(key, &[v(0.0, 0.0)]), Ordering::Less);
        assert_eq!(d.compare_bottom(key, &[v(0.0, 179.95)]), Ordering::Greater);
        for _ in 0..1100 {
            d.set_bottom(key).unwrap();
        }
        assert_eq!(d.set_bottom_counter, 1101);
    }
}
