//! `DateRangePrefixTree`: a [`NumberRangePrefixTree`] over instants --
//! millions of years, thousands of years, years, then month, day, hour,
//! minute, second and millisecond -- built on a `java.util.Calendar`
//! template ([`Calendar`]: Java's Julian/Gregorian hybrid by default, or
//! proleptic Gregorian for `java.time` compatibility).

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use super::java_calendar::{
    Calendar, BCE, DAY_OF_MONTH, ERA, HOUR_OF_DAY, MILLISECOND, MINUTE, MONTH, SECOND, YEAR,
};
use super::number_range::{
    compare_prefix, java_index, NRShape, NrBase, NumberRangePrefixTree, NumberRangeTree,
    UnitNRShape,
};
use super::{Cell, CellIterator, SpatialPrefixTree};
use crate::spatial4j::{Error, Result, Shape, SpatialContext};

/// `FIELD_BY_LEVEL` (`-1` for the levels above the year).
const FIELD_BY_LEVEL: [i32; 10] = [
    -1,
    -1,
    -1,
    YEAR as i32,
    MONTH as i32,
    DAY_OF_MONTH as i32,
    HOUR_OF_DAY as i32,
    MINUTE as i32,
    SECOND as i32,
    MILLISECOND as i32,
];

/// `YEAR_LEVEL`.
pub const YEAR_LEVEL: i32 = 3;

/// `NUM_MYEARS`: how many million years `long` milliseconds span.
const NUM_MYEARS: i32 = 586;

/// `DateRangePrefixTree`.
#[derive(Debug, Clone)]
pub struct DateRangePrefixTree {
    inner: Arc<DateInner>,
    nr: NumberRangePrefixTree,
}

#[derive(Debug)]
struct DateInner {
    base: NrBase,
    cal_tmp: Calendar,
    mincal: Calendar,
    ad_year_base: i32,
    min_lv: Vec<i32>,
    max_lv: Vec<i32>,
    gregorian_change_date_lv: Vec<i32>,
}

/// `calFieldLen(cal, field)`.
fn cal_field_len(field: usize) -> i32 {
    Calendar::maximum(field) - Calendar::minimum(field) + 1
}

impl DateRangePrefixTree {
    /// `new DateRangePrefixTree(templateCal)`.
    pub fn new(template: &Calendar) -> Result<Self> {
        let base = NrBase::new(vec![
            NUM_MYEARS,
            1000,
            1000,
            cal_field_len(MONTH),
            cal_field_len(DAY_OF_MONTH),
            cal_field_len(HOUR_OF_DAY),
            cal_field_len(MINUTE),
            cal_field_len(SECOND),
            cal_field_len(MILLISECOND),
        ])
        .expect("the date tree's level sizes are all in range");
        let mut cal_tmp = template.clone();
        cal_tmp.clear();
        let mut mincal = cal_tmp.clone();
        mincal.set_time_in_millis(i64::MIN);
        let mut maxcal = cal_tmp.clone();
        maxcal.set_time_in_millis(i64::MAX);
        // BC years count down: the earliest BC year is the year of
        // Long.MIN_VALUE. Align year 0 at an even million years.
        let bc_firstyear = mincal.get(YEAR);
        let bc_years = bc_firstyear - 1 + 1;
        let ad_year_base = ((bc_years - 1) / 1_000_000 + 1) * 1_000_000;
        let mut inner = DateInner {
            base,
            cal_tmp,
            mincal: mincal.clone(),
            ad_year_base,
            min_lv: Vec::new(),
            max_lv: Vec::new(),
            gregorian_change_date_lv: Vec::new(),
        };
        inner.max_lv = inner.to_vals(&mut maxcal.clone());
        inner.min_lv = inner.to_vals(&mut mincal.clone());
        let mut gc = inner.cal_tmp.clone();
        gc.set_time_in_millis(template.gregorian_change());
        inner.gregorian_change_date_lv = inner.to_vals(&mut gc);
        let inner = Arc::new(inner);
        let nr = NumberRangePrefixTree::new(inner.clone());
        Ok(DateRangePrefixTree { inner, nr })
    }

    /// The tree as a [`NumberRangePrefixTree`] (for its strategy).
    pub fn number_range_tree(&self) -> &NumberRangePrefixTree {
        &self.nr
    }

    /// `newCal()`: a cleared copy of the template.
    pub fn new_cal(&self) -> Calendar {
        self.inner.cal_tmp.clone()
    }

    /// `getTreeLevelForCalendarField(calField)`: the level of a field, or
    /// minus the level of the next finer one when the field has none.
    pub fn tree_level_for_calendar_field(&self, cal_field: i32) -> Result<i32> {
        for (i, &field) in FIELD_BY_LEVEL.iter().enumerate().skip(YEAR_LEVEL as usize) {
            if field == cal_field {
                return Ok(i as i32);
            } else if field > cal_field {
                return Ok(-(i as i32));
            }
        }
        Err(Error::IllegalArgument(format!(
            "Bad calendar field?: {cal_field}"
        )))
    }

    /// `getCalPrecisionField(cal)`: the finest field set, in order, or -1.
    pub fn cal_precision_field(cal: &Calendar) -> i32 {
        let mut last_field = -1;
        for &field in &FIELD_BY_LEVEL[YEAR_LEVEL as usize..] {
            if !cal.is_set(field as usize) {
                break;
            }
            last_field = field;
        }
        last_field
    }

    /// `clearFieldsAfter(cal, field)`.
    pub fn clear_fields_after(cal: &mut Calendar, field: i32) {
        for f in (field + 1)..=(MILLISECOND as i32) {
            cal.clear_field(f as usize);
        }
    }

    /// `toUnitShape(Calendar)`/`toShape(cal)`.
    pub fn to_shape(&self, cal: &mut Calendar) -> UnitNRShape {
        self.nr.unit(self.inner.to_vals(cal))
    }

    /// `toUnitShape(Date)`: the instant at full precision.
    pub fn to_unit_shape_millis(&self, millis: i64) -> UnitNRShape {
        let mut cal = self.new_cal();
        cal.set_time_in_millis(millis);
        self.to_shape(&mut cal)
    }

    /// `toObject(shape)` / `toCalendar(lv)`.
    pub fn to_calendar(&self, lv: &UnitNRShape) -> Calendar {
        self.inner.to_calendar(lv.vals())
    }

    /// `toString(cal)`: `[-+]yyyy-MM-ddTHH:mm:ss.SSS`, to the precision set.
    pub fn cal_to_string(&self, cal: &mut Calendar) -> String {
        DateInner::cal_to_string(cal)
    }

    /// `parseCalendar(str)`.
    pub fn parse_calendar(&self, s: &str) -> Result<Calendar> {
        self.inner.parse_calendar(s)
    }

    /// `parseShape(str)`.
    pub fn parse_shape(&self, s: &str) -> Result<NRShape> {
        self.nr.parse_shape(s)
    }

    /// `toRangeShape(start, end)`.
    pub fn to_range_shape(&self, start: &UnitNRShape, end: &UnitNRShape) -> Result<NRShape> {
        self.nr.to_range_shape(start, end)
    }

    /// `getNumSubCells(lv)`.
    pub fn num_sub_cells(&self, lv: &UnitNRShape) -> Result<i32> {
        self.nr.num_sub_cells(lv)
    }
}

impl DateInner {
    /// `toShape(cal)`'s cell numbers.
    fn to_vals(&self, cal: &mut Calendar) -> Vec<i32> {
        let cal_prec_field = DateRangePrefixTree::cal_precision_field(cal);
        let mut vals = Vec::new();
        if cal_prec_field >= YEAR as i32 {
            let year = cal.get(YEAR);
            let mut year_adj = if cal.get(ERA) == BCE {
                self.ad_year_base - (year - 1)
            } else {
                self.ad_year_base + year
            };
            let v0 = year_adj / 1_000_000;
            vals.push(v0);
            year_adj -= v0 * 1_000_000;
            let v1 = year_adj / 1000;
            vals.push(v1);
            year_adj -= v1 * 1000;
            vals.push(year_adj);
            for &field in &FIELD_BY_LEVEL[YEAR_LEVEL as usize + 1..] {
                if field > cal_prec_field {
                    break;
                }
                vals.push(cal.get(field as usize) - cal.actual_minimum(field as usize));
            }
        }
        DateRangePrefixTree::clear_fields_after(cal, cal_prec_field);
        vals
    }

    /// `toCalendar(lv)`.
    fn to_calendar(&self, lv: &[i32]) -> Calendar {
        let mut cal = self.cal_tmp.clone();
        if lv.is_empty() {
            return cal;
        }
        if compare_prefix(lv, &self.min_lv) <= 0 {
            return self.mincal.clone();
        }
        // Java's `int` arithmetic, which wraps: a term's level values are
        // small unless the term is corrupt.
        let mut year_adj = lv[0].wrapping_mul(1_000_000);
        if lv.len() > 1 {
            year_adj = year_adj.wrapping_add(lv[1].wrapping_mul(1000));
            if lv.len() > 2 {
                year_adj = year_adj.wrapping_add(lv[2]);
            }
        }
        if year_adj > self.ad_year_base {
            cal.set(ERA, 1);
            cal.set(YEAR, year_adj.wrapping_sub(self.ad_year_base));
        } else {
            cal.set(ERA, 0);
            cal.set(
                YEAR,
                self.ad_year_base.wrapping_sub(year_adj).wrapping_add(1),
            );
        }
        for level in (YEAR_LEVEL as usize + 1)..=lv.len() {
            let field = FIELD_BY_LEVEL[level] as usize;
            cal.set(field, lv[level - 1].wrapping_add(cal.actual_minimum(field)));
        }
        cal
    }

    /// `fastSubCells(lv)`.
    fn fast_sub_cells(&self, lv: &[i32]) -> Result<i32> {
        if lv.len() as i32 == YEAR_LEVEL + 1 {
            Ok(match lv[lv.len() - 1] {
                8 | 3 | 5 | 10 => 30,
                1 => {
                    let year_adj = lv[0]
                        .wrapping_mul(1_000_000)
                        .wrapping_add(lv[1].wrapping_mul(1000))
                        .wrapping_add(lv[2]);
                    let year = year_adj.wrapping_sub(self.ad_year_base);
                    if year % 4 == 0 && !(year % 100 == 0 && year % 400 != 0) {
                        29
                    } else {
                        28
                    }
                }
                _ => 31,
            })
        } else {
            java_index(&self.base.max_sub_cells_by_level, lv.len())
        }
    }

    /// `slowSubCells(lv)`.
    fn slow_sub_cells(&self, lv: &[i32]) -> Result<i32> {
        let field = java_index(&FIELD_BY_LEVEL, lv.len() + 1)?;
        if field == -1 || field == YEAR as i32 || field >= HOUR_OF_DAY as i32 {
            return java_index(&self.base.max_sub_cells_by_level, lv.len());
        }
        let cal = self.to_calendar(lv);
        Ok(cal.actual_maximum(field as usize) - cal.actual_minimum(field as usize) + 1)
    }

    /// `toString(cal)`.
    fn cal_to_string(cal: &mut Calendar) -> String {
        let cal_prec_field = DateRangePrefixTree::cal_precision_field(cal);
        if cal_prec_field == -1 {
            return "*".into();
        }
        let mut b = String::with_capacity(24);
        let mut year = cal.get(YEAR);
        if cal.get(ERA) == BCE {
            year -= 1;
            if year > 0 {
                b.push('-');
            }
        } else if year > 9999 {
            b.push('+');
        }
        append_padded(&mut b, year, 4);
        if cal_prec_field >= MONTH as i32 {
            b.push('-');
            append_padded(&mut b, cal.get(MONTH) + 1, 2);
        }
        if cal_prec_field >= DAY_OF_MONTH as i32 {
            b.push('-');
            append_padded(&mut b, cal.get(DAY_OF_MONTH), 2);
        }
        if cal_prec_field >= HOUR_OF_DAY as i32 {
            b.push('T');
            append_padded(&mut b, cal.get(HOUR_OF_DAY), 2);
        }
        if cal_prec_field >= MINUTE as i32 {
            b.push(':');
            append_padded(&mut b, cal.get(MINUTE), 2);
        }
        if cal_prec_field >= SECOND as i32 {
            b.push(':');
            append_padded(&mut b, cal.get(SECOND), 2);
        }
        if cal_prec_field >= MILLISECOND as i32 && cal.get(MILLISECOND) > 0 {
            b.push('.');
            append_padded(&mut b, cal.get(MILLISECOND), 3);
        }
        DateRangePrefixTree::clear_fields_after(cal, cal_prec_field);
        b
    }

    /// `parseCalendar(str)`.
    fn parse_calendar(&self, s: &str) -> Result<Calendar> {
        if s.is_empty() {
            return Err(Error::IllegalArgument("str is null or blank".into()));
        }
        let mut cal = self.cal_tmp.clone();
        if s == "*" {
            return Ok(cal);
        }
        let chars: Vec<char> = s.chars().collect();
        let mut offset = 0usize;
        let fail = |offset: usize| Error::Parse {
            message: format!("Improperly formatted datetime: {s}"),
            offset: offset as i32,
        };
        let sub = |a: usize, b: usize| -> Option<String> {
            (a <= b && b <= chars.len()).then(|| chars[a..b].iter().collect())
        };
        let last_offset = if chars[chars.len() - 1] == 'Z' {
            chars.len() - 1
        } else {
            chars.len()
        };
        let hyphen_idx = chars
            .iter()
            .skip(1)
            .position(|&c| c == '-')
            .map(|p| p + 1)
            .unwrap_or(last_offset);
        let year = sub(offset, hyphen_idx)
            .and_then(|t| java_parse_int(&t))
            .ok_or_else(|| fail(offset))?;
        cal.set(ERA, if year <= 0 { 0 } else { 1 });
        cal.set(YEAR, if year <= 0 { -year + 1 } else { year });
        offset = hyphen_idx + 1;
        if last_offset < offset {
            return Ok(cal);
        }
        // Each two-digit field: parse, range-check, set; then the
        // delimiter before the next.
        let steps: [(usize, i32, i32, Option<char>); 5] = [
            (MONTH, 1, 12, Some('-')),
            (DAY_OF_MONTH, 1, 31, Some('T')),
            (HOUR_OF_DAY, 0, 24, Some(':')),
            (MINUTE, 0, 59, Some(':')),
            (SECOND, 0, 59, Some('.')),
        ];
        for (i, &(field, min, max, delim_after)) in steps.iter().enumerate() {
            if i > 0 {
                let delim = steps[i - 1].3.expect("every step has a delimiter");
                if chars.get(offset - 1) != Some(&delim) {
                    return Err(fail(offset));
                }
            }
            let val = sub(offset, offset + 2)
                .and_then(|t| java_parse_int(&t))
                .ok_or_else(|| fail(offset))?;
            if val < min || val > max {
                return Err(fail(offset));
            }
            cal.set(field, if field == MONTH { val - 1 } else { val });
            offset += 3;
            if last_offset < offset {
                return Ok(cal);
            }
            let _ = delim_after;
        }
        if chars.get(offset - 1) != Some(&'.') {
            return Err(fail(offset));
        }
        // ms: the remaining digits, truncated to milliseconds.
        let max_offset = last_offset as i64 - offset as i64;
        let digits = sub(offset, (offset as i64 + max_offset).max(0) as usize)
            .and_then(|t| java_parse_int(&t))
            .ok_or_else(|| fail(offset))?;
        let e = max_offset - 3;
        let pow = if e >= 0 {
            10f64.powi(e as i32)
        } else {
            1.0 / 10f64.powi(-e as i32)
        };
        let millis = (digits as f64 / pow) as i32;
        cal.set(MILLISECOND, millis);
        Ok(cal)
    }
}

/// `appendPadded(builder, integer, positions)`.
fn append_padded(b: &mut String, integer: i32, positions: usize) {
    let int_str_len = if integer > 999 {
        4
    } else if integer > 99 {
        3
    } else if integer > 9 {
        2
    } else {
        1
    };
    for _ in int_str_len..positions {
        b.push('0');
    }
    b.push_str(&integer.to_string());
}

/// `Integer.parseInt(s)`: an optional sign, then decimal digits, within
/// `int`.
pub(crate) fn java_parse_int(s: &str) -> Option<i32> {
    let (neg, digits) = match s.as_bytes().first()? {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut v: i64 = 0;
    for b in digits.bytes() {
        v = v * 10 + (b - b'0') as i64;
        if v > i32::MAX as i64 + 1 {
            return None;
        }
    }
    let v = if neg { -v } else { v };
    i32::try_from(v).ok()
}

impl NumberRangeTree for DateInner {
    fn base(&self) -> &NrBase {
        &self.base
    }

    /// `getNumSubCells(lv)`: the maximum's own count at its edge; the fast
    /// Gregorian month lengths from the Gregorian change on; the
    /// calendar's below it.
    fn num_sub_cells(&self, lv: &[i32]) -> Result<i32> {
        let cmp = compare_prefix(lv, &self.max_lv);
        if cmp == 0 {
            return Ok(java_index(&self.max_lv, lv.len())? + 1);
        }
        let cmp = compare_prefix(lv, &self.gregorian_change_date_lv);
        if cmp >= 0 {
            self.fast_sub_cells(lv)
        } else {
            self.slow_sub_cells(lv)
        }
    }

    fn unit_to_string(&self, lv: &[i32]) -> String {
        Self::cal_to_string(&mut self.to_calendar(lv))
    }

    fn parse_unit_shape(&self, s: &str) -> Result<Vec<i32>> {
        let mut cal = self.parse_calendar(s)?;
        Ok(self.to_vals(&mut cal))
    }

    fn name(&self) -> &'static str {
        "DateRangePrefixTree"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl fmt::Display for DateRangePrefixTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DateRangePrefixTree")
    }
}

impl SpatialPrefixTree for DateRangePrefixTree {
    fn spatial_context(&self) -> &Arc<SpatialContext> {
        self.nr.spatial_context()
    }

    fn max_levels(&self) -> i32 {
        self.nr.max_levels()
    }

    fn level_for_distance(&self, dist: f64) -> i32 {
        self.nr.level_for_distance(dist)
    }

    fn distance_for_level(&self, level: i32) -> Result<f64> {
        self.nr.distance_for_level(level)
    }

    fn world_cell(&self) -> Box<dyn Cell> {
        self.nr.world_cell()
    }

    fn read_cell(&self, term: &[u8]) -> Result<Box<dyn Cell>> {
        self.nr.read_cell(term)
    }

    fn tree_cell_iterator(
        &self,
        shape: &Arc<dyn Shape>,
        detail_level: i32,
    ) -> Result<Box<dyn CellIterator>> {
        self.nr.tree_cell_iterator(shape, detail_level)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
