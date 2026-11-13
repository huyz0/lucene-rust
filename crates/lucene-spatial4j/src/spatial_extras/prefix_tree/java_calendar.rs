//! The subset of `java.util.GregorianCalendar` (UTC, lenient)
//! `DateRangePrefixTree` uses: the era, year, month, day of month, hour of
//! day, minute, second and millisecond fields with Java's "is set" state,
//! the Julian/Gregorian hybrid (or, with the cutover at `Long.MIN_VALUE`,
//! the proleptic Gregorian) conversion between fields and milliseconds, and
//! the minimum/maximum queries the tree asks.
//!
//! Ported from OpenJDK 21's `GregorianCalendar`, `Calendar.selectFields`,
//! `sun.util.calendar.{BaseCalendar, JulianCalendar, CalendarUtils}`.
//! `DateRangePrefixTree` only ever sets these eight fields, in the order a
//! date is written, so field resolution always takes Java's
//! month-and-day-of-month path and the hour-of-day path; the week and
//! day-of-year fields are not modelled (after `get`, Java computes and marks
//! them set; nothing here reads them). `getActualMaximum(DAY_OF_MONTH)` in
//! the cutover month itself (October 1582), which the tree never asks, is
//! the plain month length.

use std::fmt;

/// `Calendar.ERA`.
pub const ERA: usize = 0;
/// `Calendar.YEAR`.
pub const YEAR: usize = 1;
/// `Calendar.MONTH` (0-based).
pub const MONTH: usize = 2;
/// `Calendar.DAY_OF_MONTH`.
pub const DAY_OF_MONTH: usize = 5;
/// `Calendar.HOUR_OF_DAY`.
pub const HOUR_OF_DAY: usize = 11;
/// `Calendar.MINUTE`.
pub const MINUTE: usize = 12;
/// `Calendar.SECOND`.
pub const SECOND: usize = 13;
/// `Calendar.MILLISECOND`.
pub const MILLISECOND: usize = 14;

/// The fields modelled, in order.
const FIELDS: [usize; 8] = [
    ERA,
    YEAR,
    MONTH,
    DAY_OF_MONTH,
    HOUR_OF_DAY,
    MINUTE,
    SECOND,
    MILLISECOND,
];

/// `GregorianCalendar.BCE`.
pub const BCE: i32 = 0;
/// `GregorianCalendar.CE`.
pub const CE: i32 = 1;

const ONE_DAY: i64 = 24 * 60 * 60 * 1000;
/// `GregorianCalendar.EPOCH_OFFSET`: the fixed date of 1970-01-01.
const EPOCH_OFFSET: i64 = 719_163;
/// `GregorianCalendar.EPOCH_YEAR`.
const EPOCH_YEAR: i32 = 1970;
/// `GregorianCalendar.DEFAULT_GREGORIAN_CUTOVER`: 1582-10-15T00:00Z.
pub const DEFAULT_GREGORIAN_CUTOVER: i64 = -12_219_292_800_000;
/// `JulianCalendar.JULIAN_EPOCH`.
const JULIAN_EPOCH: i64 = -1;

/// `BaseCalendar.DAYS_IN_MONTH` (index 0 is the previous December).
const DAYS_IN_MONTH: [i32; 13] = [31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
/// `BaseCalendar.ACCUMULATED_DAYS_IN_MONTH`.
const ACCUMULATED_DAYS_IN_MONTH: [i64; 13] =
    [-30, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];

/// `CalendarUtils.floorDivide` on `long`.
fn floor_div(n: i64, d: i64) -> i64 {
    if n >= 0 {
        n / d
    } else {
        ((n + 1) / d) - 1
    }
}

/// `CalendarUtils.mod` on `long`.
fn floor_mod(x: i64, y: i64) -> i64 {
    x - y * floor_div(x, y)
}

/// `CalendarUtils.isGregorianLeapYear`.
fn is_gregorian_leap_year(y: i64) -> bool {
    (y % 4) == 0 && ((y % 100) != 0 || (y % 400) == 0)
}

/// `CalendarUtils.isJulianLeapYear`.
fn is_julian_leap_year(y: i64) -> bool {
    (y % 4) == 0
}

/// A calendar system's date: normalized year (1 BC is 0), month 1-12, day.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CalDate {
    year: i64,
    month: i64,
    day: i64,
}

/// Which calendar system a date is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CalSys {
    Gregorian,
    Julian,
}

impl CalSys {
    /// `getFixedDate(year, month, dayOfMonth)`.
    fn fixed_date(self, year: i64, month: i64, day: i64) -> i64 {
        match self {
            CalSys::Gregorian => {
                let prevyear = year - 1;
                let mut days = day;
                if prevyear >= 0 {
                    days += 365 * prevyear + prevyear / 4 - prevyear / 100
                        + prevyear / 400
                        + (367 * month - 362) / 12;
                } else {
                    days += 365 * prevyear + floor_div(prevyear, 4) - floor_div(prevyear, 100)
                        + floor_div(prevyear, 400)
                        + floor_div(367 * month - 362, 12);
                }
                if month > 2 {
                    days -= if is_gregorian_leap_year(year) { 1 } else { 2 };
                }
                days
            }
            CalSys::Julian => {
                let y = year;
                let mut days = JULIAN_EPOCH - 1 + 365 * (y - 1) + day;
                if y > 0 {
                    days += (y - 1) / 4;
                } else {
                    days += floor_div(y - 1, 4);
                }
                // Java floor-divides a month <= 0; every caller passes 1-12.
                days += (367 * month - 362) / 12;
                if month > 2 {
                    days -= if is_julian_leap_year(year) { 1 } else { 2 };
                }
                days
            }
        }
    }

    /// `isLeapYear(normalizedYear)`.
    fn is_leap(self, year: i64) -> bool {
        match self {
            CalSys::Gregorian => is_gregorian_leap_year(year),
            CalSys::Julian => is_julian_leap_year(year),
        }
    }

    /// `getMonthLength(year, month)`.
    fn month_length(self, year: i64, month: i64) -> i32 {
        let mut days = DAYS_IN_MONTH[month as usize];
        if month == 2 && self.is_leap(year) {
            days += 1;
        }
        days
    }

    /// `getCalendarDateFromFixedDate(date, fixedDate)`.
    fn date(self, fixed_date: i64) -> CalDate {
        match self {
            CalSys::Gregorian => {
                let year = gregorian_year_from_fixed_date(fixed_date);
                let jan1 = self.fixed_date(year, 1, 1);
                let is_leap = is_gregorian_leap_year(year);
                let mut prior_days = fixed_date - jan1;
                let mut mar1 = jan1 + 31 + 28;
                if is_leap {
                    mar1 += 1;
                }
                if fixed_date >= mar1 {
                    prior_days += if is_leap { 1 } else { 2 };
                }
                // Java floor-divides a negative numerator; `prior_days` is
                // never negative, so the plain quotient is the same.
                let month = (12 * prior_days + 373) / 367;
                let mut month1 = jan1 + ACCUMULATED_DAYS_IN_MONTH[month as usize];
                if is_leap && month >= 3 {
                    month1 += 1;
                }
                CalDate {
                    year,
                    month,
                    day: fixed_date - month1 + 1,
                }
            }
            CalSys::Julian => {
                let fd = 4 * (fixed_date - JULIAN_EPOCH) + 1464;
                let year = if fd >= 0 {
                    fd / 1461
                } else {
                    floor_div(fd, 1461)
                };
                let mut prior_days = fixed_date - self.fixed_date(year, 1, 1);
                let is_leap = is_julian_leap_year(year);
                if fixed_date >= self.fixed_date(year, 3, 1) {
                    prior_days += if is_leap { 1 } else { 2 };
                }
                // Java floor-divides a negative numerator; `prior_days` is
                // never negative, so the plain quotient is the same.
                let month = (12 * prior_days + 373) / 367;
                CalDate {
                    year,
                    month,
                    day: fixed_date - self.fixed_date(year, month, 1) + 1,
                }
            }
        }
    }
}

/// `BaseCalendar.getGregorianYearFromFixedDate(fixedDate)`.
fn gregorian_year_from_fixed_date(fixed_date: i64) -> i64 {
    let d0 = fixed_date - 1;
    let (n400, d1) = (floor_div(d0, 146_097), floor_mod(d0, 146_097));
    let (n100, d2) = (floor_div(d1, 36_524), floor_mod(d1, 36_524));
    let (n4, d3) = (floor_div(d2, 1461), floor_mod(d2, 1461));
    let n1 = floor_div(d3, 365);
    let mut year = 400 * n400 + 100 * n100 + 4 * n4 + n1;
    if !(n100 == 4 || n1 == 4) {
        year += 1;
    }
    year
}

/// A `GregorianCalendar` in UTC, lenient, with the modelled fields.
#[derive(Clone, PartialEq, Eq)]
pub struct Calendar {
    gregorian_cutover: i64,
    cutover_date: i64,
    cutover_year: i64,
    cutover_year_julian: i64,
    fields: [i32; 15],
    is_set: [bool; 15],
    /// `isTimeSet`/`time`.
    time: Option<i64>,
    /// Whether every field has been computed from `time` (`areFieldsSet`).
    fields_computed: bool,
    /// `calsys` after the last field computation.
    calsys: CalSys,
}

impl fmt::Debug for Calendar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Calendar")
            .field("cutover", &self.gregorian_cutover)
            .field("fields", &FIELDS.map(|i| (self.is_set[i], self.fields[i])))
            .finish()
    }
}

impl Calendar {
    /// `Calendar.getInstance(UTC, Locale.ROOT)` then `clear()`: the
    /// Julian/Gregorian hybrid with Java's default cutover (1582-10-15).
    pub fn new_default() -> Self {
        Self::with_cutover(DEFAULT_GREGORIAN_CUTOVER)
    }

    /// `DateRangePrefixTree.JAVA_UTIL_TIME_COMPAT_CAL`: proleptic Gregorian
    /// (the cutover at `Long.MIN_VALUE`), cleared.
    pub fn new_proleptic() -> Self {
        Self::with_cutover(i64::MIN)
    }

    /// A cleared calendar with `setGregorianChange(new Date(cutover))`.
    pub fn with_cutover(cutover: i64) -> Self {
        let mut cutover_date = floor_div(cutover, ONE_DAY) + EPOCH_OFFSET;
        if cutover == i64::MAX {
            cutover_date += 1;
        }
        // `getGregorianCutoverDate()`: the cutover day is Gregorian.
        let cutover_year = CalSys::Gregorian.date(cutover_date).year;
        let cutover_year_julian = CalSys::Julian.date(cutover_date - 1).year;
        Calendar {
            gregorian_cutover: cutover,
            cutover_date,
            cutover_year,
            cutover_year_julian,
            fields: [0; 15],
            is_set: [false; 15],
            time: None,
            fields_computed: false,
            calsys: CalSys::Gregorian,
        }
    }

    /// `getGregorianChange().getTime()`.
    pub fn gregorian_change(&self) -> i64 {
        self.gregorian_cutover
    }

    /// `clear()`.
    pub fn clear(&mut self) {
        self.fields = [0; 15];
        self.is_set = [false; 15];
        self.time = None;
        self.fields_computed = false;
    }

    /// `clear(field)`.
    pub fn clear_field(&mut self, field: usize) {
        self.fields[field] = 0;
        self.is_set[field] = false;
        self.time = None;
        self.fields_computed = false;
    }

    /// `set(field, value)`.
    pub fn set(&mut self, field: usize, value: i32) {
        self.fields[field] = value;
        self.is_set[field] = true;
        self.time = None;
        self.fields_computed = false;
    }

    /// `isSet(field)`.
    pub fn is_set(&self, field: usize) -> bool {
        self.is_set[field]
    }

    /// `get(field)`: completes the calendar first (every field computed and
    /// set).
    pub fn get(&mut self, field: usize) -> i32 {
        self.complete();
        self.fields[field]
    }

    /// `setTimeInMillis(millis)`.
    pub fn set_time_in_millis(&mut self, millis: i64) {
        self.time = Some(millis);
        self.compute_fields();
    }

    /// `getTimeInMillis()`.
    pub fn time_in_millis(&mut self) -> i64 {
        self.complete();
        self.time.expect("completed")
    }

    /// `complete()`.
    fn complete(&mut self) {
        if self.time.is_none() {
            self.compute_time();
        }
        if !self.fields_computed {
            self.compute_fields();
        }
    }

    /// `computeFields()`: every field from `time`.
    fn compute_fields(&mut self) {
        let time = self.time.expect("time is set before fields are computed");
        let mut fixed_date = time / ONE_DAY;
        let mut time_of_day = time % ONE_DAY;
        while time_of_day < 0 {
            time_of_day += ONE_DAY;
            fixed_date -= 1;
        }
        fixed_date += EPOCH_OFFSET;
        let sys = if fixed_date >= self.cutover_date {
            CalSys::Gregorian
        } else {
            CalSys::Julian
        };
        let d = sys.date(fixed_date);
        let (era, year) = if d.year <= 0 {
            (BCE, 1 - d.year)
        } else {
            (CE, d.year)
        };
        self.calsys = sys;
        self.fields[ERA] = era;
        self.fields[YEAR] = year as i32;
        self.fields[MONTH] = (d.month - 1) as i32;
        self.fields[DAY_OF_MONTH] = d.day as i32;
        let hours = time_of_day / 3_600_000;
        let r = time_of_day % 3_600_000;
        self.fields[HOUR_OF_DAY] = hours as i32;
        self.fields[MINUTE] = (r / 60_000) as i32;
        self.fields[SECOND] = (r % 60_000 / 1000) as i32;
        self.fields[MILLISECOND] = (r % 1000) as i32;
        // Java computes and marks every field, the unmodelled ones too.
        self.is_set = [true; 15];
        self.fields_computed = true;
    }

    /// `getFixedDate(cal, year, fieldMask)` on the month-and-day path.
    fn fixed_date_of(&self, sys: CalSys, year: i64) -> i64 {
        let mut year = year;
        let mut month = self.fields[MONTH] as i64;
        if month > 11 {
            year += month / 12;
            month %= 12;
        } else if month < 0 {
            let q = floor_div(month, 12);
            year += q;
            month -= q * 12;
        }
        let mut fixed_date = sys.fixed_date(year, month + 1, 1);
        if self.is_set[DAY_OF_MONTH] {
            fixed_date += self.fields[DAY_OF_MONTH] as i64;
            fixed_date -= 1;
        }
        fixed_date
    }

    /// `computeTime()` (lenient): the instant the set fields name.
    fn compute_time(&mut self) {
        let mut year = if self.is_set[YEAR] {
            self.fields[YEAR] as i64
        } else {
            EPOCH_YEAR as i64
        };
        let era = if self.is_set[ERA] {
            self.fields[ERA]
        } else {
            CE
        };
        if era == BCE {
            year = 1 - year;
        }
        // (Java throws "Invalid era" for any other era value; the tree only
        // ever sets 0 or 1.)
        let mut time_of_day: i64 = self.fields[HOUR_OF_DAY] as i64;
        time_of_day = time_of_day * 60 + self.fields[MINUTE] as i64;
        time_of_day = time_of_day * 60 + self.fields[SECOND] as i64;
        time_of_day = time_of_day * 1000 + self.fields[MILLISECOND] as i64;
        let mut fixed_date = time_of_day / ONE_DAY;
        time_of_day %= ONE_DAY;
        while time_of_day < 0 {
            time_of_day += ONE_DAY;
            fixed_date -= 1;
        }
        let (gfd, jfd);
        if year > self.cutover_year && year > self.cutover_year_julian {
            let g = fixed_date + self.fixed_date_of(CalSys::Gregorian, year);
            if g >= self.cutover_date {
                return self.finish_time(g, time_of_day);
            }
            gfd = g;
            jfd = fixed_date + self.fixed_date_of(CalSys::Julian, year);
        } else if year < self.cutover_year && year < self.cutover_year_julian {
            let j = fixed_date + self.fixed_date_of(CalSys::Julian, year);
            if j < self.cutover_date {
                return self.finish_time(j, time_of_day);
            }
            jfd = j;
            gfd = j;
        } else {
            jfd = fixed_date + self.fixed_date_of(CalSys::Julian, year);
            gfd = fixed_date + self.fixed_date_of(CalSys::Gregorian, year);
        }
        let fd = if gfd >= self.cutover_date {
            if jfd >= self.cutover_date || self.calsys == CalSys::Gregorian {
                gfd
            } else {
                jfd
            }
        } else {
            // a "missing" date is taken as Julian (lenient)
            jfd
        };
        self.finish_time(fd, time_of_day);
    }

    fn finish_time(&mut self, fixed_date: i64, time_of_day: i64) {
        let millis = (fixed_date - EPOCH_OFFSET)
            .wrapping_mul(ONE_DAY)
            .wrapping_add(time_of_day);
        self.time = Some(millis);
        self.compute_fields();
    }

    /// `getMinimum(field)`.
    pub fn minimum(field: usize) -> i32 {
        match field {
            ERA => BCE,
            YEAR | DAY_OF_MONTH => 1,
            _ => 0,
        }
    }

    /// `getMaximum(field)`.
    pub fn maximum(field: usize) -> i32 {
        match field {
            ERA => CE,
            YEAR => 292_278_994,
            MONTH => 11,
            DAY_OF_MONTH => 31,
            HOUR_OF_DAY => 23,
            MINUTE | SECOND => 59,
            _ => 999,
        }
    }

    /// `getActualMinimum(field)`: the minimum, except the day of month in
    /// the cutover year's months, which start at the first day that exists.
    pub fn actual_minimum(&self, field: usize) -> i32 {
        if field == DAY_OF_MONTH {
            let (gc, date, fd) = self.normalized();
            if date.year == gc.cutover_year || date.year == gc.cutover_year_julian {
                let month1 = gc.fixed_date_month1(date, fd);
                return gc.calendar_date(month1).day as i32;
            }
        }
        Self::minimum(field)
    }

    /// `getActualMaximum(field)` for the fields `DateRangePrefixTree` asks
    /// (the month and the day of month), with the cutover year's gaps.
    pub fn actual_maximum(&self, field: usize) -> i32 {
        match field {
            MONTH => {
                let (gc, date, _) = self.normalized();
                if !gc.is_cutover_year(date.year) {
                    return 11;
                }
                // January 1 of the next year may or may not exist.
                let mut year = date.year;
                let next_jan1 = loop {
                    year += 1;
                    let fd = CalSys::Gregorian.fixed_date(year, 1, 1);
                    if fd >= gc.cutover_date {
                        break fd;
                    }
                };
                (gc.calsys.date(next_jan1 - 1).month - 1) as i32
            }
            DAY_OF_MONTH => {
                let (gc, date, fd) = self.normalized();
                let value = gc.calsys.month_length(date.year, date.month);
                if !gc.is_cutover_year(date.year)
                    || date.day == i64::from(value)
                    || fd >= gc.cutover_date
                {
                    return value;
                }
                let month_end = gc.fixed_date_month1(date, fd) + gc.actual_month_length(date) - 1;
                gc.calendar_date(month_end).day as i32
            }
            _ => Self::maximum(field),
        }
    }

    /// `getNormalizedCalendar()`, with its `cdate` and that date's fixed date.
    fn normalized(&self) -> (Calendar, CalDate, i64) {
        let mut gc = self.clone();
        gc.complete();
        let year = if gc.fields[ERA] == BCE {
            1 - i64::from(gc.fields[YEAR])
        } else {
            i64::from(gc.fields[YEAR])
        };
        let date = CalDate {
            year,
            month: i64::from(gc.fields[MONTH]) + 1,
            day: i64::from(gc.fields[DAY_OF_MONTH]),
        };
        let fd = gc.calsys.fixed_date(date.year, date.month, date.day);
        (gc, date, fd)
    }

    /// `isCutoverYear(normalizedYear)` on a normalized calendar.
    fn is_cutover_year(&self, normalized_year: i64) -> bool {
        let cutover_year = if self.calsys == CalSys::Gregorian {
            self.cutover_year
        } else {
            self.cutover_year_julian
        };
        normalized_year == cutover_year
    }

    /// `getCalendarDate(fd)`: Gregorian from the cutover on, Julian before.
    fn calendar_date(&self, fd: i64) -> CalDate {
        if fd >= self.cutover_date {
            CalSys::Gregorian.date(fd)
        } else {
            CalSys::Julian.date(fd)
        }
    }

    /// `getFixedDateMonth1(date, fixedDate)`: the first day of `date`'s
    /// month that exists, in the cutover year.
    fn fixed_date_month1(&self, date: CalDate, fixed_date: i64) -> i64 {
        let g_cutover = self.calendar_date(self.cutover_date);
        if g_cutover.month == 1 && g_cutover.day == 1 {
            return fixed_date - date.day + 1;
        }
        if date.month == g_cutover.month {
            let j_last = self.calendar_date(self.cutover_date - 1);
            if self.cutover_year == self.cutover_year_julian && g_cutover.month == j_last.month {
                // The "gap" fits in the same month.
                CalSys::Julian.fixed_date(date.year, date.month, 1)
            } else {
                self.cutover_date
            }
        } else {
            fixed_date - date.day + 1
        }
    }

    /// `actualMonthLength()`: the days `date`'s month really has.
    fn actual_month_length(&self, date: CalDate) -> i64 {
        let month_length = i64::from(self.calsys.month_length(date.year, date.month));
        if date.year != self.cutover_year && date.year != self.cutover_year_julian {
            return month_length;
        }
        let fd = self.calsys.fixed_date(date.year, date.month, date.day);
        let month1 = self.fixed_date_month1(date, fd);
        let next1 = month1 + month_length;
        if next1 < self.cutover_date {
            return next1 - month1;
        }
        let next_date = CalSys::Gregorian.date(next1);
        self.fixed_date_month1(next_date, next1) - month1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_and_cutover() {
        let mut c = Calendar::new_default();
        c.set_time_in_millis(0);
        assert_eq!(
            (c.get(ERA), c.get(YEAR), c.get(MONTH), c.get(DAY_OF_MONTH)),
            (CE, 1970, 0, 1)
        );
        c.set_time_in_millis(DEFAULT_GREGORIAN_CUTOVER);
        assert_eq!(
            (c.get(YEAR), c.get(MONTH), c.get(DAY_OF_MONTH)),
            (1582, 9, 15)
        );
        c.set_time_in_millis(DEFAULT_GREGORIAN_CUTOVER - 1);
        assert_eq!(
            (c.get(MONTH), c.get(DAY_OF_MONTH), c.get(HOUR_OF_DAY)),
            (9, 4, 23)
        );
        let mut p = Calendar::new_proleptic();
        p.set_time_in_millis(DEFAULT_GREGORIAN_CUTOVER - 1);
        assert_eq!((p.get(MONTH), p.get(DAY_OF_MONTH)), (9, 14));
        assert_eq!(p.gregorian_change(), i64::MIN);
    }

    #[test]
    fn lenient_fields_roll_over() {
        let mut c = Calendar::new_default();
        c.set(ERA, CE);
        c.set(YEAR, 2014);
        c.set(MONTH, 3);
        c.set(DAY_OF_MONTH, 31);
        c.set(HOUR_OF_DAY, 24);
        assert_eq!(
            (c.get(MONTH), c.get(DAY_OF_MONTH), c.get(HOUR_OF_DAY)),
            (4, 2, 0)
        );
        let mut d = Calendar::new_default();
        d.set(YEAR, 2000);
        d.set(MONTH, 13);
        d.set(MONTH, -1);
        assert_eq!((d.get(YEAR), d.get(MONTH)), (1999, 11));
        d.clear_field(MONTH);
        assert!(!d.is_set(MONTH));
        let t = d.time_in_millis();
        assert_eq!(t, 915_148_800_000, "1999-01-01");
        d.clear();
        assert!(!d.is_set(YEAR));
    }

    #[test]
    fn month_lengths_follow_the_calendar_system() {
        let mut c = Calendar::new_default();
        c.set(YEAR, 1500);
        c.set(MONTH, 1);
        assert_eq!(
            c.actual_maximum(DAY_OF_MONTH),
            29,
            "1500 is a Julian leap year"
        );
        let mut g = Calendar::new_proleptic();
        g.set(YEAR, 1500);
        g.set(MONTH, 1);
        assert_eq!(g.actual_maximum(DAY_OF_MONTH), 28);
        assert_eq!(c.actual_maximum(MONTH), 11);
        assert_eq!(c.actual_maximum(HOUR_OF_DAY), 23);
        assert_eq!(Calendar::maximum(MILLISECOND), 999);
        assert_eq!(c.actual_minimum(DAY_OF_MONTH), 1);
        assert!(!format!("{c:?}").is_empty());
        let mut bc = Calendar::new_default();
        bc.set(ERA, BCE);
        bc.set(YEAR, 1);
        bc.set(MONTH, 1);
        assert_eq!(bc.actual_maximum(DAY_OF_MONTH), 29, "1 BC is Julian year 0");
        let max = Calendar::with_cutover(i64::MAX);
        assert!(max.cutover_year > 200_000_000);
    }
}
