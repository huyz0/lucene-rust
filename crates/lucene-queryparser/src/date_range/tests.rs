#![cfg(test)]
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
    let day = |d: &DateParsing, text: &str| d.bound(text, Some(Resolution::Day), false).unwrap();
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
