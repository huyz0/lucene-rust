//! `DateTools`: dates as lexicographically sortable `yyyyMMddHHmmssSSS`
//! strings, cut to a [`Resolution`], always in GMT.
//!
//! Java formats and parses through `SimpleDateFormat` over a
//! `GregorianCalendar`, and that calendar is not the proleptic Gregorian one:
//! dates before the 15 October 1582 cutover are **Julian**, and a year before
//! 1 AD is printed as its year of era (`yyyy` of 2 BC is `0002`). This port
//! reproduces both, so a pre-1582 time formats to the digits Lucene wrote.
//! Parsing is `SimpleDateFormat`'s lenient parse of abutting digit fields: a
//! string of one of the seven resolution lengths, all ASCII digits, whose
//! out-of-range fields (month 13, day 0) roll over as `Calendar` rolls them.

use super::{illegal, Result};

/// `DateTools.Resolution`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Resolution {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
    Millisecond,
}

impl Resolution {
    /// `formatLen`: the length of this resolution's string.
    pub fn format_len(self) -> usize {
        match self {
            Resolution::Year => 4,
            Resolution::Month => 6,
            Resolution::Day => 8,
            Resolution::Hour => 10,
            Resolution::Minute => 12,
            Resolution::Second => 14,
            Resolution::Millisecond => 17,
        }
    }

    fn of_len(len: usize) -> Option<Self> {
        [
            Resolution::Year,
            Resolution::Month,
            Resolution::Day,
            Resolution::Hour,
            Resolution::Minute,
            Resolution::Second,
            Resolution::Millisecond,
        ]
        .into_iter()
        .find(|r| r.format_len() == len)
    }
}

/// `DateTools`.
#[derive(Debug, Clone, Copy)]
pub struct DateTools;

const MS_PER_DAY: i64 = 86_400_000;
/// 1582-10-15 (the first Gregorian day), days since 1970-01-01.
const GREGORIAN_CUTOVER_DAY: i64 = -141_427;

/// A broken-down GMT time: proleptic astronomical year (1 BC is `0`),
/// month `1..=12`, day, and time of day.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fields {
    year: i64,
    month: i64,
    day: i64,
    hour: i64,
    minute: i64,
    second: i64,
    milli: i64,
}

// ARITH: (the date arithmetic below) every input is a Java `long` of
// milliseconds or a 4..=17-digit string's fields; the day counts involved are
// below 2^40 and every product and sum stays far inside `i64`.

/// Days since the epoch of a proleptic Gregorian date (`days_from_civil`).
// ARITH: day counts stay below 2^40 and milliseconds within
// `i64` for every input Java accepts (see the note above `gregorian_days`).
#[allow(clippy::arithmetic_side_effects)]
fn gregorian_days(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The proleptic Gregorian date of an epoch day (`civil_from_days`).
// ARITH: day counts stay below 2^40 and milliseconds within
// `i64` for every input Java accepts (see the note above `gregorian_days`).
#[allow(clippy::arithmetic_side_effects)]
fn gregorian_date(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Days since the epoch of a proleptic Julian date.
// ARITH: day counts stay below 2^40 and milliseconds within
// `i64` for every input Java accepts (see the note above `gregorian_days`).
#[allow(clippy::arithmetic_side_effects)]
fn julian_days(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(4);
    let yoe = y - era * 4;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + doy;
    // 0000-03-01 Julian is 719_470 days before 1970-01-01 Gregorian.
    era * 1461 + doe - 719_470
}

/// The proleptic Julian date of an epoch day.
// ARITH: day counts stay below 2^40 and milliseconds within
// `i64` for every input Java accepts (see the note above `gregorian_days`).
#[allow(clippy::arithmetic_side_effects)]
fn julian_date(z: i64) -> (i64, i64, i64) {
    let z = z + 719_470;
    let era = z.div_euclid(1461);
    let doe = z - era * 1461;
    let yoe = (doe - doe / 1460) / 365;
    let y = yoe + era * 4;
    let doy = doe - 365 * yoe;
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `GregorianCalendar`'s date for an epoch day: Gregorian from the cutover
/// on, Julian before it.
fn calendar_date(day: i64) -> (i64, i64, i64) {
    if day >= GREGORIAN_CUTOVER_DAY {
        gregorian_date(day)
    } else {
        julian_date(day)
    }
}

/// `GregorianCalendar`'s epoch day for a date whose day of month may be out
/// of range (lenient): the Gregorian reading when it falls on or after the
/// cutover, the Julian one otherwise.
// ARITH: day counts stay below 2^40 and milliseconds within
// `i64` for every input Java accepts (see the note above `gregorian_days`).
#[allow(clippy::arithmetic_side_effects)]
fn calendar_days(y: i64, m: i64, d: i64) -> i64 {
    // ARITH: see the note above `gregorian_days`.
    let g = gregorian_days(y, m, 1) + d - 1;
    if g >= GREGORIAN_CUTOVER_DAY {
        g
    } else {
        julian_days(y, m, 1) + d - 1
    }
}

// ARITH: day counts stay below 2^40 and milliseconds within
// `i64` for every input Java accepts (see the note above `gregorian_days`).
#[allow(clippy::arithmetic_side_effects)]
fn fields_of(time: i64) -> Fields {
    let day = time.div_euclid(MS_PER_DAY);
    let ms = time.rem_euclid(MS_PER_DAY);
    let (year, month, d) = calendar_date(day);
    Fields {
        year,
        month,
        day: d,
        hour: ms / 3_600_000,
        minute: ms / 60_000 % 60,
        second: ms / 1000 % 60,
        milli: ms % 1000,
    }
}

/// `Calendar`'s lenient `getTimeInMillis` over set fields: months roll into
/// years, days and times add on.
// ARITH: day counts stay below 2^40 and milliseconds within
// `i64` for every input Java accepts (see the note above `gregorian_days`).
#[allow(clippy::arithmetic_side_effects)]
fn time_of(f: Fields) -> i64 {
    let month0 = f.month - 1;
    let year = f.year + month0.div_euclid(12);
    let month = month0.rem_euclid(12) + 1;
    let day = calendar_days(year, month, f.day);
    // A time near the ends of `long` rounds past them and wraps, as Java's
    // `long` arithmetic does.
    day.wrapping_mul(MS_PER_DAY)
        .wrapping_add(f.hour * 3_600_000 + f.minute * 60_000 + f.second * 1000 + f.milli)
}

impl DateTools {
    /// `round(time, resolution)`: every field finer than `resolution` reset
    /// (`MONTH` to January, `DAY_OF_MONTH` to 1, the time of day to zero).
    pub fn round(time: i64, resolution: Resolution) -> i64 {
        let mut f = fields_of(time);
        if resolution <= Resolution::Year {
            f.month = 1;
        }
        if resolution <= Resolution::Month {
            f.day = 1;
        }
        if resolution <= Resolution::Day {
            f.hour = 0;
        }
        if resolution <= Resolution::Hour {
            f.minute = 0;
        }
        if resolution <= Resolution::Minute {
            f.second = 0;
        }
        if resolution <= Resolution::Second {
            f.milli = 0;
        }
        time_of(f)
    }

    /// `timeToString(time, resolution)`.
    // ARITH: day counts stay below 2^40 and milliseconds within
    // `i64` for every input Java accepts (see the note above `gregorian_days`).
    #[allow(clippy::arithmetic_side_effects)]
    pub fn time_to_string(time: i64, resolution: Resolution) -> String {
        let f = fields_of(Self::round(time, resolution));
        // `yyyy` is the year of era: 1 BC (astronomical 0) prints as 1.
        let year_of_era = if f.year <= 0 { 1 - f.year } else { f.year };
        let full = format!(
            "{year_of_era:04}{:02}{:02}{:02}{:02}{:02}{:03}",
            f.month, f.day, f.hour, f.minute, f.second, f.milli
        );
        // A year past 9999 is printed in full, and the tail cut as Java
        // cuts it: to the pattern, not to the string.
        let extra = full.len() - 17;
        full[..resolution.format_len() + extra].to_string()
    }

    /// `stringToTime(dateString)`.
    pub fn string_to_time(date_string: &str) -> Result<i64> {
        let err = || illegal(format!("Input is not a valid date string: {date_string}"));
        let resolution = Resolution::of_len(date_string.len()).ok_or_else(err)?;
        if !date_string.bytes().all(|b| b.is_ascii_digit()) {
            return Err(err());
        }
        let num = |from: usize, to: usize| -> i64 {
            date_string
                .get(from..to)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0)
        };
        let len = resolution.format_len();
        let field = |from: usize, to: usize, default: i64| {
            if len >= to {
                num(from, to)
            } else {
                default
            }
        };
        Ok(time_of(Fields {
            year: num(0, 4),
            month: field(4, 6, 1),
            day: field(6, 8, 1),
            hour: field(8, 10, 0),
            minute: field(10, 12, 0),
            second: field(12, 14, 0),
            milli: field(14, 17, 0),
        }))
    }

    /// `GregorianCalendar`'s lenient epoch day of a year, a 0-based month and
    /// a day of month, the month and day possibly out of range (they roll
    /// on). Java picks the Julian or the Gregorian reading by the year as set,
    /// before the month rolls into it: before 1582 the Julian one, from 1582
    /// on the Gregorian one unless that falls before the cutover. The month
    /// rolls into the year in `int` arithmetic (`getFixedDate`'s
    /// `year += month / 12`), which wraps as Java's does.
    // ARITH: the rolled year is an `int` and the day an `int`: every
    // intermediate of the day count stays below 2^45.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn lenient_epoch_day(year: i32, month0: i32, day: i32) -> i64 {
        let y = i64::from(year.wrapping_add(month0.div_euclid(12)));
        let m = i64::from(month0.rem_euclid(12)) + 1;
        let day = i64::from(day);
        let gregorian = gregorian_days(y, m, 1) + day - 1;
        if year >= 1582 && gregorian >= GREGORIAN_CUTOVER_DAY {
            gregorian
        } else {
            julian_days(y, m, 1) + day - 1
        }
    }

    /// `GregorianCalendar.computeFields`' local epoch day and millisecond of
    /// the day of `time` in a zone `zone` milliseconds from GMT. Like Java,
    /// it divides the time and the offset separately and carries their
    /// remainders, so no sum of the two wraps.
    // ARITH: each quotient is below 2^37 and the remainders below 2 days.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn local_epoch_day(time: i64, zone: i64) -> (i64, i64) {
        let ms = time.rem_euclid(MS_PER_DAY) + zone.rem_euclid(MS_PER_DAY);
        let day =
            time.div_euclid(MS_PER_DAY) + zone.div_euclid(MS_PER_DAY) + ms.div_euclid(MS_PER_DAY);
        (day, ms.rem_euclid(MS_PER_DAY))
    }

    /// The `GregorianCalendar` fields of `time` in a zone `zone` milliseconds
    /// from GMT ([`Self::local_epoch_day`]): its year (astronomical: 1 BC is
    /// `0`), month `1..=12`, day of month and millisecond of the day.
    pub fn calendar_fields(time: i64, zone: i64) -> (i64, i64, i64, i64) {
        let (day, ms) = Self::local_epoch_day(time, zone);
        let (year, month, d) = calendar_date(day);
        (year, month, d, ms)
    }

    /// `round(Date, resolution)` over a millisecond time is [`Self::round`];
    /// `dateToString` is [`Self::time_to_string`].
    pub fn date_to_string(time: i64, resolution: Resolution) -> String {
        Self::time_to_string(time, resolution)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;

    #[test]
    fn calendars_agree_with_known_dates() {
        assert_eq!(gregorian_days(1970, 1, 1), 0);
        assert_eq!(gregorian_days(1582, 10, 15), GREGORIAN_CUTOVER_DAY);
        assert_eq!(julian_days(1582, 10, 4), GREGORIAN_CUTOVER_DAY - 1);
        for d in [-800_000, -141_428, -141_427, 0, 20_000, 3_000_000] {
            let (y, m, dd) = gregorian_date(d);
            assert_eq!(gregorian_days(y, m, dd), d);
            let (y, m, dd) = julian_date(d);
            assert_eq!(julian_days(y, m, dd), d);
        }
        assert_eq!(calendar_date(GREGORIAN_CUTOVER_DAY - 1), (1582, 10, 4));
    }

    #[test]
    fn formats_and_rounds() {
        // 2004-09-21 13:50:11.123 GMT.
        let t = 1_095_774_611_123;
        assert_eq!(DateTools::time_to_string(t, Resolution::Year), "2004");
        assert_eq!(DateTools::time_to_string(t, Resolution::Month), "200409");
        assert_eq!(DateTools::time_to_string(t, Resolution::Day), "20040921");
        assert_eq!(DateTools::time_to_string(t, Resolution::Hour), "2004092113");
        assert_eq!(
            DateTools::time_to_string(t, Resolution::Minute),
            "200409211350"
        );
        assert_eq!(
            DateTools::date_to_string(t, Resolution::Second),
            "20040921135011"
        );
        assert_eq!(
            DateTools::time_to_string(t, Resolution::Millisecond),
            "20040921135011123"
        );
        assert_eq!(DateTools::round(t, Resolution::Millisecond), t);
        assert_eq!(DateTools::string_to_time("20040921135011123").unwrap(), t);
        assert_eq!(DateTools::round(t, Resolution::Year), 1_072_915_200_000);
        assert_eq!(
            DateTools::time_to_string(-1, Resolution::Millisecond),
            "19691231235959999"
        );
    }

    #[test]
    fn parses_leniently_and_rejects_bad_strings() {
        assert_eq!(
            DateTools::string_to_time("200413").unwrap(),
            DateTools::string_to_time("200501").unwrap()
        );
        assert_eq!(
            DateTools::string_to_time("20040100").unwrap(),
            DateTools::string_to_time("20031231").unwrap()
        );
        assert!(DateTools::string_to_time("20041").is_err());
        assert!(DateTools::string_to_time("2004a").is_err());
        assert!(DateTools::string_to_time("abcd").is_err());
        assert!(DateTools::string_to_time("").is_err());
        assert_eq!(DateTools::string_to_time("1970").unwrap(), 0);
    }

    /// `GregorianCalendar`'s lenient reading. The JDK's own answers for
    /// rolled dates (`13/40/2020`, `0/0/0`, a wrapping year) are pinned end to
    /// end by `analysis_date_locales/numbers_jdk25.tsv` and the queryparser
    /// `date_*.tsv` fixtures; these check the calendar choice and the wrap.
    #[test]
    fn lenient_epoch_days_roll_and_pick_a_calendar_by_the_year_set() {
        let day = DateTools::lenient_epoch_day;
        assert_eq!(day(1970, 0, 1), 0);
        assert_eq!(day(2020, 12, 40), gregorian_days(2021, 1, 40));
        assert_eq!(day(0, -1, 0), julian_days(-1, 12, 0));
        // 1582-10-04 (Julian) is the day before 1582-10-15 (Gregorian).
        assert_eq!(day(1582, 9, 15), GREGORIAN_CUTOVER_DAY);
        assert_eq!(day(1582, 9, 4), GREGORIAN_CUTOVER_DAY - 1);
        // A year before 1582 stays Julian however far its days roll.
        assert_eq!(day(1581, 0, 1000), julian_days(1581, 1, 1) + 999);
        // From 1582, a date rolling back before the cutover is Julian.
        assert_eq!(day(1583, 0, -400), julian_days(1583, 1, 1) - 401);
        // The rolled year wraps in `int`, the calendar still chosen by the
        // year as set.
        assert_eq!(day(i32::MAX, 12, 1), julian_days(i64::from(i32::MIN), 1, 1));
        assert_eq!(
            day(i32::MIN, -1, 1),
            julian_days(i64::from(i32::MAX), 12, 1)
        );
    }

    #[test]
    fn local_days_carry_the_zone_without_wrapping() {
        assert_eq!(
            DateTools::calendar_fields(-1, 0),
            (1969, 12, 31, 86_399_999)
        );
        assert_eq!(DateTools::local_epoch_day(-1, 1), (0, 0));
        assert_eq!(DateTools::local_epoch_day(0, -1), (-1, MS_PER_DAY - 1));
        let hour = 3_600_000;
        let (day, ms) = DateTools::local_epoch_day(i64::MAX - hour, 2 * hour);
        assert_eq!(
            i128::from(day) * i128::from(MS_PER_DAY) + i128::from(ms),
            i128::from(i64::MAX) + i128::from(hour)
        );
        let (day, ms) = DateTools::local_epoch_day(i64::MIN, -hour);
        assert_eq!(
            i128::from(day) * i128::from(MS_PER_DAY) + i128::from(ms),
            i128::from(i64::MIN) - i128::from(hour)
        );
    }

    #[test]
    fn early_dates_are_julian_and_years_are_of_era() {
        // 1000-01-01 Julian.
        let t = julian_days(1000, 1, 1) * MS_PER_DAY;
        assert_eq!(DateTools::time_to_string(t, Resolution::Day), "10000101");
        assert_eq!(DateTools::string_to_time("10000101").unwrap(), t);
        let bc = julian_days(-1, 6, 1) * MS_PER_DAY;
        assert_eq!(DateTools::time_to_string(bc, Resolution::Year), "0002");
        let far = gregorian_days(12345, 1, 1) * MS_PER_DAY;
        assert_eq!(DateTools::time_to_string(far, Resolution::Year), "12345");
    }
}
