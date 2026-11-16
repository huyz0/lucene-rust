//! The date bounds of a range query: `QueryParserBase.getRangeQuery`'s and
//! `TermRangeQueryNodeProcessor`'s
//! `DateTools.dateToString(DateFormat.getDateInstance(SHORT, locale).parse(part), resolution)`.
//!
//! The format is the locale's lenient `SimpleDateFormat` over a
//! `GregorianCalendar` in `TimeZone.getDefault()`
//! ([`DateParsing::default_time_zone`]); an inclusive upper bound moves to
//! 23:59:59.999 of its day in the parser's own time zone. Every Gregorian
//! locale's short pattern has only numeric `y`, `M` and `d` fields, whose
//! values [`SimpleDateFormat::parse_fields`] reads; here they are set on
//! the calendar as `CalendarBuilder` sets them: a one- or two-letter year
//! of exactly two digits lands in the 100 years from 80 years before now
//! (`SimpleDateFormat.set2DigitYearStart`'s default window), the month and
//! day roll on leniently, and a date before the Gregorian cutover is
//! Julian.
//!
//! A year in the hundreds of millions (an infinity, an exponent) overflows
//! the milliseconds, which wrap into some other date, in Java's `long`
//! arithmetic and here alike; a month rolling past `Integer.MAX_VALUE`'s
//! year wraps the `int` year as `GregorianCalendar.getFixedDate` does.
//!
//! Differs: time zones are fixed offsets (no daylight saving); a locale
//! whose calendar is not Gregorian (Thai's Buddhist), or whose short
//! pattern has a field other than `y`, `M` and `d` (none of the table's
//! does), is refused -- when a bound is read as a date, as Java only then
//! builds the format.

use lucene_analysis::miscellaneous::{DateLocale, SimpleDateFormat};
use lucene_index::document::{DateTools, Resolution};

const MS_PER_DAY: i64 = 86_400_000;

/// The date settings a parser's range bounds are read with.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DateParsing {
    locale: DateLocale,
    format: SimpleDateFormat,
    /// The locale's dates are ones this port computes: a Gregorian calendar
    /// and a short pattern of `y`, `M` and `d` only.
    supported: bool,
    /// The parser's `TimeZone` as its fixed offset in milliseconds; `None`
    /// is [`Self::default_time_zone`] (`TimeZone.getDefault()`, which the
    /// parsers start with).
    pub(crate) time_zone: Option<i32>,
    /// `TimeZone.getDefault()`'s offset: the zone the format parses in.
    pub(crate) default_time_zone: i32,
    /// The time the two-digit-year window is computed from; `None` is the
    /// system clock.
    pub(crate) now: Option<i64>,
}

impl Default for DateParsing {
    fn default() -> Self {
        let locale = DateLocale::for_language_tag("en-US").unwrap_or_else(DateLocale::english);
        DateParsing {
            locale,
            format: SimpleDateFormat::short_date_instance(locale),
            supported: supported(locale),
            time_zone: None,
            default_time_zone: 0,
            now: None,
        }
    }
}

impl DateParsing {
    /// `setLocale(locale)`.
    pub(crate) fn set_locale(&mut self, locale: DateLocale) {
        self.locale = locale;
        self.format = SimpleDateFormat::short_date_instance(locale);
        self.supported = supported(locale);
    }

    /// `getLocale()`.
    pub(crate) fn locale(&self) -> DateLocale {
        self.locale
    }

    fn now(&self) -> i64 {
        self.now.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .ok()
                .and_then(|d| i64::try_from(d.as_millis()).ok())
                .unwrap_or(0)
        })
    }

    /// `df.parse(text)`: the time, or `None` where Java throws
    /// `ParseException` (nothing parsed).
    pub(crate) fn parse(&self, text: &str) -> Option<i64> {
        let fields = self.format.parse_fields(text)?;
        if fields.end == 0 {
            return None;
        }
        let zone = i64::from(self.default_time_zone);
        let (start, start_year) = century_start(self.now(), zone);
        // `CalendarBuilder`'s fields, Java `int`s, unset ones the epoch's.
        let (mut year, mut month0, mut day) = (1970_i32, 0_i32, 1_i32);
        let mut ambiguous = false;
        for f in &fields.numeric {
            match f.letter {
                b'y' => {
                    year = f.value;
                    if f.count <= 2 && f.two_digits {
                        let two = start_year % 100;
                        ambiguous = year == two;
                        let shift = start_year / 100 * 100 + if year < two { 100 } else { 0 };
                        year = year.wrapping_add(shift);
                    }
                }
                b'M' => month0 = f.value.wrapping_sub(1),
                // `d`: [`supported`] admits no other letter.
                _ => day = f.value,
            }
        }
        let time = establish(year, month0, day, zone);
        Some(if ambiguous && time < start {
            establish(year.wrapping_add(100), month0, day, zone)
        } else {
            time
        })
    }

    /// The locale's dates are ones this port computes (see [`supported`]).
    pub(crate) fn is_supported(&self) -> bool {
        self.supported
    }

    /// `df.format(new Date(time))`: the date of `time` in the default zone,
    /// in the short pattern's `y`, `M` and `d` fields (the year of its era).
    pub(crate) fn format(&self, time: i64) -> String {
        let (year, month, day, _) =
            DateTools::calendar_fields(time, i64::from(self.default_time_zone));
        let year = if year <= 0 { 1 - year } else { year };
        // [`supported`] admits no other field.
        self.format
            .format_year_month_day(year, month, day)
            .unwrap_or_default()
    }

    /// The parser's time zone offset.
    fn zone(&self) -> i64 {
        i64::from(self.time_zone.unwrap_or(self.default_time_zone))
    }

    /// `Calendar.getInstance(timeZone, locale)` at `time`, its time of day
    /// set to 23:59:59.999 (in Java's `long` arithmetic, which wraps).
    pub(crate) fn end_of_day(&self, time: i64) -> i64 {
        let zone = self.zone();
        let (day, _) = DateTools::local_epoch_day(time, zone);
        day.wrapping_mul(MS_PER_DAY)
            .wrapping_add(MS_PER_DAY - 1)
            .wrapping_sub(zone)
    }

    /// `QueryParserBase.getRangeQuery`'s bound: `DateTools.dateToString` of
    /// the parsed date (at the end of its day when `end_inclusive`), or
    /// `Ok(None)` -- the bound kept -- where Java's `try` catches (no
    /// resolution, or not a date); `Err` (the reason) for a locale this
    /// port does not compute dates in.
    pub(crate) fn bound(
        &self,
        text: &str,
        resolution: Option<Resolution>,
        end_inclusive: bool,
    ) -> Result<Option<String>, String> {
        // Java parses before `dateToString` throws on a null resolution; the
        // outcome is the same, so no resolution skips the parse.
        let Some(resolution) = resolution else {
            return Ok(None);
        };
        if !self.supported {
            return Err(format!(
                "unsupported date locale {}: its calendar is not Gregorian",
                self.locale.tag()
            ));
        }
        let Some(mut time) = self.parse(text) else {
            return Ok(None);
        };
        if end_inclusive {
            time = self.end_of_day(time);
        }
        Ok(Some(DateTools::date_to_string(time, resolution)))
    }
}

/// The locale's dates are ones [`DateParsing::parse`] computes: a Gregorian
/// calendar, and a short pattern whose only fields are numeric `y`, `M` and
/// `d` (letters outside quotes).
fn supported(locale: DateLocale) -> bool {
    let mut quoted = false;
    locale.is_gregorian()
        && SimpleDateFormat::short_date_instance(locale).is_numeric()
        && locale.short_pattern().chars().all(|c| {
            if c == '\'' {
                quoted = !quoted;
            }
            quoted || !c.is_ascii_alphabetic() || matches!(c, 'y' | 'M' | 'd')
        })
}

/// `SimpleDateFormat`'s default two-digit-year start: now, 80 years back
/// (`Calendar.add(YEAR, -80)`, the day pinned to the month's length), and
/// its year.
fn century_start(now: i64, zone: i64) -> (i64, i32) {
    let (year, month, day, ms) = DateTools::calendar_fields(now, zone);
    // A `long` of milliseconds is within 300 million years: every field
    // fits an `int`.
    let int = |v: i64| i32::try_from(v).unwrap_or(i32::MAX);
    let year = int(year).saturating_sub(80);
    let (m, day) = (int(month).saturating_sub(1), int(day));
    let length = DateTools::lenient_epoch_day(year, m.saturating_add(1), 1)
        .saturating_sub(DateTools::lenient_epoch_day(year, m, 1));
    let day = i32::try_from(i64::from(day).min(length)).unwrap_or(day);
    let start = DateTools::lenient_epoch_day(year, m, day)
        .wrapping_mul(MS_PER_DAY)
        .wrapping_add(ms)
        .wrapping_sub(zone);
    (start, year)
}

/// `GregorianCalendar.computeTime` of a lenient date's midnight in `zone`:
/// a `long` of milliseconds, which a year in the hundreds of millions
/// overflows -- and wraps, as Java's does.
fn establish(year: i32, month0: i32, day: i32, zone: i64) -> i64 {
    DateTools::lenient_epoch_day(year, month0, day)
        .wrapping_mul(MS_PER_DAY)
        .wrapping_sub(zone)
}

#[cfg(test)]
mod tests;
