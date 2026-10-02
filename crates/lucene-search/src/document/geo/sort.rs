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

use lucene_codecs::field_infos::FieldInfo;

use super::{geo, illegal, sorted_numeric_in, GeoValues};
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
    /// The field's doc values are of another type, or the index does not
    /// decode.
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
    /// The field's doc values are of another type, or the index does not
    /// decode.
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
        };
        distance.check(r, &self.field)?;
        let values = match info_of(r, &self.field) {
            Some(info) => sorted_numeric_in(r, info)?,
            None => None,
        };
        Ok(Box::new(LeafComparator {
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
    distance: Box<dyn Distance>,
    values: Option<GeoValues<'a>>,
    buf: Vec<i64>,
}

impl LeafFieldComparator for LeafComparator<'_> {
    fn value(&mut self, doc: i32, _score: f32) -> Result<SortValue> {
        self.buf.clear();
        if let Some(v) = self.values.as_mut() {
            v.values(doc, &mut self.buf)?;
        }
        let key = if self.buf.is_empty() {
            f64::INFINITY
        } else {
            self.distance.sort_key(&self.buf)
        };
        Ok(SortValue::Long(double_to_sortable_long(key)))
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
