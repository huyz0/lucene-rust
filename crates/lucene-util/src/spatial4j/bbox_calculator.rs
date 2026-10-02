//! `BBoxCalculator` (`org.locationtech.spatial4j.shape.impl`): the minimum
//! bounding box of a set of rectangles. Planar is a plain min/max; geodetic
//! keeps a sorted set of disjoint longitude ranges and, at the end, puts the
//! box opposite the biggest gap.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::Arc;

use super::context::SpatialContext;
use super::shape::Rectangle;
use super::Result;
use crate::geo::{java_max, java_min};

/// A `Double` map key ordered as `Double.compareTo` orders (`-0.0 < 0.0`,
/// NaN last).
#[derive(Debug, Clone, Copy)]
struct Key(f64);

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Key {}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.0.is_nan(), other.0.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => self.0.total_cmp(&other.0),
        }
    }
}

/// `BBoxCalculator`.
#[derive(Debug, Clone)]
pub struct BBoxCalculator {
    ctx: Arc<SpatialContext>,
    min_y: f64,
    max_y: f64,
    min_x: f64,
    max_x: f64,
    /// Disjoint x ranges keyed by maxX, valued minX; `None` once processed
    /// (or world-wrapped).
    ranges: Option<BTreeMap<Key, f64>>,
}

/// `rangeContains(minX, maxX, x)`.
fn range_contains(min_x: f64, max_x: f64, x: f64) -> bool {
    if min_x <= max_x {
        x >= min_x && x <= max_x
    } else {
        x >= min_x || x <= max_x
    }
}

impl BBoxCalculator {
    /// `new BBoxCalculator(ctx)`.
    pub fn new(ctx: Arc<SpatialContext>) -> Self {
        BBoxCalculator {
            ctx,
            min_y: f64::INFINITY,
            max_y: f64::NEG_INFINITY,
            min_x: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            ranges: None,
        }
    }

    /// `expandRange(rect)`.
    pub fn expand_range_rect(&mut self, rect: &dyn Rectangle) {
        self.expand_range(rect.min_x(), rect.max_x(), rect.min_y(), rect.max_y());
    }

    /// `expandRange(minX, maxX, minY, maxY)`.
    pub fn expand_range(&mut self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) {
        self.min_y = java_min(self.min_y, min_y);
        self.max_y = java_max(self.max_y, max_y);
        self.expand_x_range(min_x, max_x);
    }

    /// The first entry at or after `key` (wrapping to the first entry).
    fn entry_from(ranges: &BTreeMap<Key, f64>, key: f64, inclusive: bool) -> Option<(f64, f64)> {
        let mut it = if inclusive {
            ranges.range(Key(key)..)
        } else {
            ranges.range((
                std::ops::Bound::Excluded(Key(key)),
                std::ops::Bound::Unbounded,
            ))
        };
        it.next().map(|(k, v)| (k.0, *v))
    }

    /// `expandXRange(minX, maxX)`.
    pub fn expand_x_range(&mut self, min_x: f64, max_x: f64) {
        if !self.ctx.is_geo() {
            self.min_x = java_min(self.min_x, min_x);
            self.max_x = java_max(self.max_x, max_x);
            return;
        }
        if self.does_x_world_wrap() {
            return;
        }
        let Some(ranges) = self.ranges.as_mut() else {
            let mut m = BTreeMap::new();
            m.insert(Key(max_x), min_x);
            self.ranges = Some(m);
            return;
        };
        // An iterator starting from the first entry that either contains
        // minX or is to the right of it (wrapping across the dateline).
        let (mut entry_max, mut entry_min) = match Self::entry_from(ranges, min_x, true) {
            Some(e) => e,
            None => {
                let (k, v) = ranges.iter().next().expect("ranges is never empty");
                (k.0, *v)
            }
        };
        if range_contains(entry_min, entry_max, max_x) {
            if range_contains(entry_min, entry_max, min_x) {
                // This entry & the new range together might wrap the world.
                if (min_x != entry_min || max_x != entry_max)
                    && range_contains(min_x, max_x, entry_min)
                    && range_contains(min_x, max_x, entry_max)
                {
                    self.min_x = -180.0;
                    self.max_x = 180.0;
                    self.ranges = None;
                }
            } else {
                // Update entry's start to be minX.
                ranges.insert(Key(entry_max), min_x);
            }
        } else {
            // Insert an entry, removing the ones it overlaps.
            let new_min_x = if range_contains(entry_min, entry_max, min_x) {
                entry_min
            } else {
                min_x
            };
            let mut new_max_x = max_x;
            while range_contains(new_min_x, new_max_x, entry_min) {
                ranges.remove(&Key(entry_max));
                if !range_contains(min_x, max_x, entry_max) {
                    new_max_x = entry_max;
                    break;
                }
                // get new entry (wrap around, which can only happen once)
                let next = Self::entry_from(ranges, entry_max, false)
                    .or_else(|| ranges.iter().next().map(|(k, v)| (k.0, *v)));
                match next {
                    Some((k, v)) => {
                        entry_max = k;
                        entry_min = v;
                    }
                    None => break,
                }
            }
            ranges.insert(Key(new_max_x), new_min_x);
        }
    }

    /// `processRanges()`.
    fn process_ranges(&mut self) {
        let Some(ranges) = self.ranges.take() else {
            return;
        };
        if ranges.len() == 1 {
            let (k, v) = ranges.iter().next().expect("one range");
            self.min_x = *v;
            self.max_x = k.0;
        } else {
            // Find the biggest gap; the box is opposite it.
            let (mut prev_max, _) = ranges
                .iter()
                .next_back()
                .map(|(k, v)| (k.0, *v))
                .expect("ranges is never empty");
            let mut biggest_gap = 0.0;
            let mut possible_remaining_gap = 360.0;
            for (k, v) in ranges.iter() {
                let (range_max, range_min) = (k.0, *v);
                let mut width_plus_gap = range_max - prev_max;
                if width_plus_gap < 0.0 {
                    width_plus_gap += 360.0;
                }
                let mut gap = range_min - prev_max;
                if gap < 0.0 {
                    gap += 360.0;
                }
                possible_remaining_gap -= width_plus_gap;
                if gap > biggest_gap {
                    biggest_gap = gap;
                    self.min_x = range_min;
                    self.max_x = prev_max;
                    if possible_remaining_gap <= biggest_gap {
                        break;
                    }
                }
                prev_max = range_max;
            }
        }
    }

    /// `doesXWorldWrap()`.
    pub fn does_x_world_wrap(&self) -> bool {
        self.min_x == -180.0 && self.max_x == 180.0
    }

    /// `getBoundary()`.
    pub fn boundary(&mut self) -> Result<Arc<dyn Rectangle>> {
        let (a, b) = (self.min_x(), self.max_x());
        self.ctx.rect(a, b, self.min_y, self.max_y)
    }

    /// `getMinX()`.
    pub fn min_x(&mut self) -> f64 {
        self.process_ranges();
        self.min_x
    }

    /// `getMaxX()`.
    pub fn max_x(&mut self) -> f64 {
        self.process_ranges();
        self.max_x
    }

    /// `getMinY()`.
    pub fn min_y(&self) -> f64 {
        self.min_y
    }

    /// `getMaxY()`.
    pub fn max_y(&self) -> f64 {
        self.max_y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_order_as_double_compare() {
        assert!(Key(-0.0) < Key(0.0));
        assert!(Key(f64::NAN) > Key(f64::INFINITY));
        assert_eq!(Key(f64::NAN), Key(-f64::NAN));
        assert!(Key(f64::NEG_INFINITY) < Key(f64::NAN));
        assert_eq!(Key(1.0).partial_cmp(&Key(2.0)), Some(Ordering::Less));
    }
}
