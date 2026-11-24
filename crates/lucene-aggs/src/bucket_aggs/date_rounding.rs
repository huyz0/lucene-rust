//! `Rounding.Prepared` for a `date_histogram` in a fixed-offset or region zone.

use super::region_zone::RegionZone;

/// A calendar unit of `Rounding.DateTimeUnit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DateUnit {
    Week,
    Year,
    Quarter,
    Month,
    Day,
    Hour,
    Minute,
    Second,
}

/// A `date_histogram`'s rounding: a calendar unit or a fixed interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingKind {
    /// `TimeUnitRounding`.
    Unit(DateUnit),
    /// `TimeIntervalRounding`, in milliseconds (at least 1).
    Interval(i64),
}

/// `Rounding.Prepared` for a zone whose offset is fixed (`UTC`, `+05:30`):
/// `FixedToMidnightRounding`/`FixedNotToMidnightRounding`/`FixedRounding`
/// behind `OffsetRounding`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DateRounding {
    pub kind: RoundingKind,
    /// The zone's offset from UTC, in milliseconds; 0 when `region` is set.
    pub zone_ms: i64,
    /// An IANA region zone, whose offset changes: it replaces `zone_ms`.
    pub region: Option<RegionZone>,
    /// `OffsetRounding`'s offset, in milliseconds.
    pub offset: i64,
}

pub(super) const MILLIS_PER_DAY: i64 = 86_400_000;
pub(super) const DAYS_0000_TO_1970: i64 = 719_527;
pub(super) const MILLIS_PER_YEAR: i64 = 31_556_952_000;
pub(super) const MIN_TOTAL_MILLIS_BY_MONTH: [i64; 12] = month_starts(false);
pub(super) const MAX_TOTAL_MILLIS_BY_MONTH: [i64; 12] = month_starts(true);

/// `DateUtilsRounding`'s `MIN_`/`MAX_TOTAL_MILLIS_BY_MONTH_ARRAY`.
pub(super) const fn month_starts(leap: bool) -> [i64; 12] {
    let days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut out = [0i64; 12];
    let mut i = 0;
    while i < 11 {
        out[i + 1] = out[i] + days[i] * MILLIS_PER_DAY;
        i += 1;
    }
    out
}

/// `DateUtils.roundFloor` (units of a day or less).
pub(super) fn round_floor(utc: i64, unit: i64) -> i64 {
    if utc >= 0 {
        utc - utc % unit
    } else {
        let u = utc.wrapping_add(1);
        u.wrapping_sub(u % unit).wrapping_sub(unit)
    }
}

/// `DateUtilsRounding.isLeapYear`.
pub(super) fn is_leap_year(year: i32) -> bool {
    if year & 3 != 0 {
        return false;
    }
    if year % 100 != 0 {
        return true;
    }
    (year / 100) & 3 == 0
}

/// `DateUtilsRounding.utcMillisAtStartOfYear`.
pub(super) fn utc_millis_at_start_of_year(year: i32) -> i64 {
    let mut leap_years = year / 100;
    if year < 0 {
        leap_years = ((year + 3) >> 2) - leap_years + ((leap_years + 3) >> 2) - 1;
    } else {
        leap_years = (year >> 2) - leap_years + (leap_years >> 2);
        if is_leap_year(year) {
            leap_years -= 1;
        }
    }
    // Java's long arithmetic, which wraps at the far ends of the range.
    i64::from(year)
        .wrapping_mul(365)
        .wrapping_add(i64::from(leap_years) - DAYS_0000_TO_1970)
        .wrapping_mul(MILLIS_PER_DAY)
}

/// `DateUtilsRounding.getYear`.
pub(super) fn get_year(utc: i64) -> i32 {
    let unit = MILLIS_PER_YEAR / 2;
    let mut i2 = (utc >> 1) + (1970 * MILLIS_PER_YEAR) / 2;
    if i2 < 0 {
        i2 = i2 - unit + 1;
    }
    let mut year = (i2 / unit) as i32;
    let mut year_start = utc_millis_at_start_of_year(year);
    let diff = utc.wrapping_sub(year_start);
    if diff < 0 {
        year -= 1;
    } else if diff >= MILLIS_PER_DAY * 365 {
        let one_year = if is_leap_year(year) {
            MILLIS_PER_DAY * 366
        } else {
            MILLIS_PER_DAY * 365
        };
        year_start = year_start.wrapping_add(one_year);
        if year_start <= utc {
            year += 1;
        }
    }
    year
}

/// `DateUtilsRounding.getMonthOfYear`: 1 to 12.
pub(super) fn get_month_of_year(utc: i64, year: i32) -> usize {
    let i = (utc.wrapping_sub(utc_millis_at_start_of_year(year)) >> 10) as i32;
    let d = 84_375;
    let bounds: [i32; 11] = if is_leap_year(year) {
        [31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    } else {
        [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    };
    1 + bounds.iter().take_while(|&&b| i >= b * d).count()
}

/// `DateUtils.of(year, month)`: the first millisecond of the month.
pub(super) fn start_of_month(year: i32, month: usize) -> i64 {
    let table = if is_leap_year(year) {
        &MAX_TOTAL_MILLIS_BY_MONTH
    } else {
        &MIN_TOTAL_MILLIS_BY_MONTH
    };
    utc_millis_at_start_of_year(year).wrapping_add(table.get(month - 1).copied().unwrap_or(0))
}

impl DateUnit {
    /// `DateTimeUnit.roundFloor`.
    pub fn round_floor(self, utc: i64) -> i64 {
        match self {
            DateUnit::Week => {
                let three_days = 3 * MILLIS_PER_DAY;
                round_floor(utc.wrapping_add(three_days), 7 * MILLIS_PER_DAY)
                    .wrapping_sub(three_days)
            }
            DateUnit::Year => utc_millis_at_start_of_year(get_year(utc)),
            DateUnit::Quarter => {
                let year = get_year(utc);
                let month = get_month_of_year(utc, year);
                start_of_month(year, (month - 1) / 3 * 3 + 1)
            }
            DateUnit::Month => {
                let year = get_year(utc);
                start_of_month(year, get_month_of_year(utc, year))
            }
            DateUnit::Day => round_floor(utc, MILLIS_PER_DAY),
            DateUnit::Hour => round_floor(utc, 3_600_000),
            DateUnit::Minute => round_floor(utc, 60_000),
            DateUnit::Second => round_floor(utc, 1_000),
        }
    }
}

impl DateUnit {
    /// The first millisecond of the unit after the one starting at `start`
    /// (a value [`Self::round_floor`] returned); `None` past the range.
    pub(super) fn next_start(self, start: i64) -> Option<i64> {
        let months = match self {
            DateUnit::Week => return start.checked_add(7 * MILLIS_PER_DAY),
            DateUnit::Day => return start.checked_add(MILLIS_PER_DAY),
            DateUnit::Hour => return start.checked_add(3_600_000),
            DateUnit::Minute => return start.checked_add(60_000),
            DateUnit::Second => return start.checked_add(1_000),
            DateUnit::Month => 1,
            DateUnit::Quarter => 3,
            DateUnit::Year => 12,
        };
        let year = get_year(start);
        let month = get_month_of_year(start, year) + months;
        let (year, month) = if month > 12 {
            (year.checked_add(1)?, month - 12)
        } else {
            (year, month)
        };
        Some(start_of_month(year, month))
    }
}

impl DateRounding {
    /// The rounding of `kind` in a region zone, with `offset`.
    pub fn in_region(kind: RoundingKind, region: RegionZone, offset: i64) -> Self {
        DateRounding {
            kind,
            zone_ms: 0,
            region: Some(region),
            offset,
        }
    }

    /// `utc`'s bucket: its rounded value and the first value past it that
    /// rounds elsewhere (every value in between rounds the same); `None`
    /// where that bound would overflow.
    pub fn bucket(&self, utc: i64) -> Option<(i64, i64)> {
        if let Some(region) = self.region {
            // `OffsetRounding`: the zone's rounding of `utc - offset`, shifted back.
            let inner = utc.checked_sub(self.offset)?;
            let floor = region.round(self.kind, inner);
            let next = region.next(self.kind, inner)?;
            if floor > inner || next <= inner {
                return None;
            }
            return Some((
                floor.checked_add(self.offset)?,
                next.checked_add(self.offset)?,
            ));
        }
        let shift = self.zone_ms.checked_sub(self.offset)?;
        let local = utc.checked_add(shift)?;
        let (floor, next) = match self.kind {
            RoundingKind::Unit(u) => {
                let floor = u.round_floor(local);
                (floor, u.next_start(floor)?)
            }
            RoundingKind::Interval(interval) => {
                let floor = self.round(utc).checked_add(shift)?;
                (floor, floor.checked_add(interval)?)
            }
        };
        Some((floor.checked_sub(shift)?, next.checked_sub(shift)?))
    }

    /// `Prepared.round`: `offset.localToUtcInThisOffset(roundFloor(
    /// offset.utcToLocalTime(utc - offset))) + offset`.
    pub fn round(&self, utc: i64) -> i64 {
        if let Some(region) = self.region {
            return region
                .round(self.kind, utc.wrapping_sub(self.offset))
                .wrapping_add(self.offset);
        }
        let local = utc.wrapping_sub(self.offset).wrapping_add(self.zone_ms);
        let rounded = match self.kind {
            RoundingKind::Unit(u) => u.round_floor(local),
            RoundingKind::Interval(interval) => {
                // `TimeIntervalRounding.roundKey`.
                let key = if local < 0 {
                    local.wrapping_sub(interval).wrapping_add(1) / interval
                } else {
                    local / interval
                };
                key.wrapping_mul(interval)
            }
        };
        rounded.wrapping_sub(self.zone_ms).wrapping_add(self.offset)
    }
}
