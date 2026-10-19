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
#[derive(Debug, Clone)]
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
mod tests {
    use super::*;

    const HOUR: i32 = 3_600_000;

    fn parsing(tag: &str) -> DateParsing {
        let mut d = DateParsing::default();
        d.set_locale(DateLocale::for_language_tag(tag).unwrap());
        d
    }

    /// A file of `fixtures/data`, as text.
    fn fixture(rel: &str) -> String {
        let path = format!("{}/../../fixtures/data/{rel}", env!("CARGO_MANIFEST_DIR"));
        std::fs::read_to_string(path).unwrap()
    }

    /// `GenQueryParserClassic`'s `System.currentTimeMillis()`, which Java's
    /// two-digit years were read against.
    fn generator_clock() -> i64 {
        fixture("queryparser_classic/date_clock.txt")
            .trim()
            .parse()
            .unwrap()
    }

    /// `GenQueryParserClassic.writeDateParse`: the short formats of a
    /// sample of locales, in GMT, over formatted dates and their mutations,
    /// the two-digit-year window the generator's clock's. A row whose
    /// pattern JDK 21 (the generator's) words otherwise than JDK 25 (the
    /// table's) is skipped.
    #[test]
    fn short_formats_parse_as_the_jdk() {
        let text = fixture("queryparser_classic/date_parse.tsv");
        let now = generator_clock();
        let (mut compared, mut skipped) = (0, 0);
        let mut bad = Vec::new();
        for line in text.lines() {
            let c: Vec<&str> = line.split('\t').collect();
            assert!(!line.contains('\\'), "an escaped row: {line}");
            let Some(locale) = DateLocale::for_language_tag(c[0]) else {
                skipped += 1;
                continue;
            };
            if locale.short_pattern() != c[1] {
                skipped += 1;
                continue;
            }
            let mut d = DateParsing::default();
            d.set_locale(locale);
            d.now = Some(now);
            let got = d.parse(c[2]).map_or("-".to_string(), |t| t.to_string());
            compared += 1;
            if got != c[3] {
                bad.push(format!(
                    "{} {:?} {:?}: want {} got {got}",
                    c[0], c[1], c[2], c[3]
                ));
            }
        }
        assert!(compared > 1000, "{compared} compared, {skipped} skipped");
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }

    /// `GenAnalysisDateLocales --jdk25`'s `numbers_jdk25.tsv`: `en-US`'s
    /// short format in GMT over numeric extremes (exponents, infinities,
    /// `long` and `int` wraps, rolled months and days) as JDK 25 parses
    /// them; JDK 21 reads a `1E` exponent of twenty digits otherwise.
    #[test]
    fn numeric_extremes_parse_as_jdk_25() {
        let text = fixture("analysis_date_locales/numbers_jdk25.tsv");
        let d = parsing("en-US");
        let mut rows = 0;
        for line in text.lines() {
            let (input, want) = line.split_once('\t').unwrap();
            assert!(!input.contains('\\'), "an escaped row: {line}");
            let got = d.parse(input).map_or("-".to_string(), |t| t.to_string());
            assert_eq!(got, want, "{input:?}");
            rows += 1;
        }
        assert!(rows > 15, "{rows}");
    }

    #[test]
    fn a_two_digit_year_falls_in_the_hundred_years_from_eighty_ago() {
        let mut d = parsing("en-US");
        // 2026-10-20T12:00Z: the window starts 1946-10-20T12:00Z.
        d.now = Some(1_792_497_600_000);
        let day =
            |d: &DateParsing, text: &str| d.bound(text, Some(Resolution::Day), false).unwrap();
        assert_eq!(day(&d, "1/2/45").as_deref(), Some("20450102"));
        assert_eq!(day(&d, "1/2/47").as_deref(), Some("19470102"));
        // 46 is the window's own two digits: 1946 unless before its start.
        assert_eq!(day(&d, "1/2/46").as_deref(), Some("20460102"));
        assert_eq!(day(&d, "12/2/46").as_deref(), Some("19461202"));
        // Three digits, or one, are the year as written.
        assert_eq!(day(&d, "1/2/046").as_deref(), Some("00460102"));
        assert_eq!(day(&d, "1/2/5").as_deref(), Some("00050102"));
        // On 1980-02-29 the window starts 1900-02-28, the leap day pinned
        // to the month's end: that day is 1900's, the one before 2000's.
        d.now = Some(320_630_400_000);
        assert_eq!(day(&d, "2/28/00").as_deref(), Some("19000228"));
        assert_eq!(day(&d, "2/27/00").as_deref(), Some("20000227"));
        // The system clock when unset.
        d.now = None;
        assert_eq!(day(&d, "1/2/2020").as_deref(), Some("20200102"));
    }

    #[test]
    fn bounds_and_their_zones() {
        let mut d = parsing("en-US");
        d.default_time_zone = -5 * HOUR;
        let ms = |d: &DateParsing, text: &str, end: bool| {
            d.bound(text, Some(Resolution::Millisecond), end).unwrap()
        };
        // Parsed at midnight in the default zone; the day ends in the
        // parser's, which is the default one until set.
        assert_eq!(
            ms(&d, "1/2/2020", false).as_deref(),
            Some("20200102050000000")
        );
        assert_eq!(
            ms(&d, "1/2/2020", true).as_deref(),
            Some("20200103045959999")
        );
        d.time_zone = Some(2 * HOUR);
        assert_eq!(
            ms(&d, "1/2/2020", true).as_deref(),
            Some("20200102215959999")
        );
        // No resolution, or not a date: the bound stays.
        assert_eq!(d.bound("1/2/2020", None, false), Ok(None));
        assert_eq!(d.bound("x", Some(Resolution::Day), false), Ok(None));
        assert_eq!(d.parse(""), None);
        // An infinite year wraps as Java's milliseconds do.
        assert_eq!(
            ms(&d, "1/2/\u{221E}", false).as_deref(),
            Some("1907286350622191633536")
        );
        assert_eq!(d.locale().tag(), "en-US");
    }

    /// Java builds the format only to read a bound as a date: a locale this
    /// port does not compute is refused then, not when set.
    #[test]
    fn a_buddhist_locale_is_refused_when_a_bound_is_a_date() {
        let mut d = DateParsing::default();
        let thai = DateLocale::for_language_tag("th-TH").unwrap();
        assert!(!thai.is_gregorian());
        d.set_locale(thai);
        assert_eq!(d.locale().tag(), "th-TH");
        assert_eq!(d.bound("1/2/2563", None, false), Ok(None));
        let err = d.bound("1/2/2563", Some(Resolution::Day), false);
        assert!(err.unwrap_err().contains("th-TH"));
    }

    /// Only `y`, `M` and `d` outside quotes are admitted.
    #[test]
    fn supported_patterns_have_only_year_month_and_day() {
        for tag in ["en-US", "de-DE", "ja-JP", "und", "fr-FR", "zh-CN", "ko-KR"] {
            let locale = DateLocale::for_language_tag(tag).unwrap();
            assert!(supported(locale), "{tag}");
        }
        assert!(!supported(DateLocale::for_language_tag("th-TH").unwrap()));
    }

    #[test]
    fn the_extremes_wrap() {
        let mut d = parsing("en-US");
        d.time_zone = Some(0);
        assert_eq!(d.end_of_day(-1), -1);
        assert!(d.end_of_day(i64::MAX) < 0);
        // Java divides the time and the zone separately: an hour short of
        // the end of `long`, two hours east, the day is the true one (past
        // `long`'s end), not a wrapped sum's; only its end then wraps.
        d.time_zone = Some(2 * HOUR);
        let hour = i64::from(HOUR);
        let day = i64::try_from((i128::from(i64::MAX) + i128::from(hour)) / 86_400_000).unwrap();
        let want = day
            .wrapping_mul(MS_PER_DAY)
            .wrapping_add(MS_PER_DAY - 1)
            .wrapping_sub(2 * hour);
        assert_eq!(d.end_of_day(i64::MAX - hour), want);
        assert!(century_start(i64::MAX, 0).1 > 200_000_000);
        assert_eq!(
            century_start(i64::MAX - hour, 2 * hour).1,
            century_start(i64::MAX, 0).1
        );
        d.now = Some(i64::MAX);
        assert!(d.parse("1/2/20").is_some());
    }
}
