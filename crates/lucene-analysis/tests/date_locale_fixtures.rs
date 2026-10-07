//! M12 T12.7: `SimpleDateFormat`'s parse in every locale JDK 21 and 25 agree
//! on (`GenAnalysisDateLocales.java`), against the generated table
//! (`miscellaneous/date_locales.rs`).

mod support;

use lucene_analysis::miscellaneous::{DateLocale, SimpleDateFormat};
use support::unesc;

#[test]
fn every_locale_parses_as_java() {
    let text =
        std::fs::read_to_string(support::data_dir("analysis_date_locales") + "dates.txt").unwrap();
    let (mut rows, mut failures) = (0, Vec::new());
    let mut tags = std::collections::BTreeSet::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let (tag, pattern, input, want) = (f[0], unesc(f[1]), unesc(f[2]), f[3]);
        let locale =
            DateLocale::for_language_tag(tag).unwrap_or_else(|| panic!("{tag} has no data"));
        assert_eq!(locale.tag(), tag);
        tags.insert(tag);
        let format = if pattern == "DEFAULT" {
            SimpleDateFormat::date_instance(locale)
        } else {
            SimpleDateFormat::with_locale(&pattern, locale).unwrap()
        };
        let got = format.parse(&input).map_or(-1, |e| e as i64).to_string();
        if got != want && failures.len() < 40 {
            failures.push(format!(
                "{tag} {pattern:?} {input:?}: java {want}, rust {got}"
            ));
        }
        rows += 1;
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(rows > 20_000, "{rows}");
    // Every locale of the list resolves (a few have no row: their one text
    // was left out and their record had its battery under another tag).
    let corpus = support::corpus("date-locales.txt");
    assert!(corpus
        .iter()
        .all(|t| DateLocale::for_language_tag(t).is_some_and(|l| l.tag() == t)));
    assert!(
        tags.len() + 40 > corpus.len(),
        "{} of {}",
        tags.len(),
        corpus.len()
    );
}

/// What a tag resolves to: `Locale.Builder`'s canonical form, the table's
/// locales only.
#[test]
fn tags_resolve_as_the_locale_builder_does() {
    for (tag, want) in [
        ("EN", Some("en")),
        ("en-us", Some("en-US")),
        ("zh-hant-tw", Some("zh-Hant-TW")),
        ("zh-yue-hk", Some("yue-HK")),
        ("iw", Some("he")),
        ("in-id", Some("id-ID")),
        ("ji", Some("yi")),
        ("und", Some("und")),
        ("es-419", Some("es-419")),
        ("th-TH", Some("th-TH")),
        ("en-Latn", None),
        ("en-US-u-nu-thai", None),
        ("de-1996", None),
        ("x-foo", None),
        ("ja-JP-u-ca-japanese", None),
    ] {
        assert_eq!(
            DateLocale::for_language_tag(tag).map(|l| l.tag()),
            want,
            "{tag}"
        );
    }
    assert_eq!(DateLocale::english().default_pattern(), "MMM d, y");
    assert_eq!(format!("{:?}", DateLocale::english()), "DateLocale(\"en\")");
}

/// JDK 25's data where JDK 21's differs (left out of the fixture).
#[test]
fn jdk_25_data() {
    let de = DateLocale::for_language_tag("de").unwrap();
    // CLDR's stand-alone abbreviations for `L`, the format ones' for `M`.
    let l = SimpleDateFormat::with_locale("LLL", de).unwrap();
    let m = SimpleDateFormat::with_locale("MMM", de).unwrap();
    let dm = SimpleDateFormat::with_locale("d MMM", de).unwrap();
    assert_eq!((l.parse("Sept."), m.parse("Sept.")), (Some(5), Some(3)));
    assert_eq!((dm.parse("1 Sept."), dm.parse("1 Sep")), (Some(7), None));
    // Adlam's case pairs are supplementary: compared as code points.
    let ff = SimpleDateFormat::with_locale("a", DateLocale::for_language_tag("ff-Adlm").unwrap())
        .unwrap();
    assert_eq!(ff.parse("\u{1E922}\u{1E930}"), Some(4));
    let sv = DateLocale::for_language_tag("sv").unwrap();
    let y = SimpleDateFormat::with_locale("y", sv).unwrap();
    assert_eq!((y.parse("\u{2212}12"), y.parse("-12")), (Some(3), None));
    assert_eq!(y.parse("1\u{d7}10^\u{2212}2"), Some(7));
}
