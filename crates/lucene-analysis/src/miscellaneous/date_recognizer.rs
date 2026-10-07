//! `DateRecognizerFilter`, and the part of `java.text.SimpleDateFormat` it
//! calls: `DateFormat.parse(String)` succeeding or throwing
//! `ParseException`.
//!
//! The filter keeps a term exactly when `parse` succeeds, which needs no
//! calendar: Java's formats are lenient (`isLenient()` is true), so every
//! field value is accepted and only the text's shape decides. `parse(String)`
//! is `parse(text, new ParsePosition(0))`: the pattern's elements must match
//! one after another from the start of the term, and anything after the last
//! one is ignored. What each element accepts is written here from
//! `SimpleDateFormat`'s and `DecimalFormat`'s specifications and checked
//! against the JDK, input by input (`GenAnalysisMisc.java`'s `date_formats.txt`
//! runs ~95 patterns, 48 of them abutting fields and their neighbours, over
//! seeded valid, mutated and glued-together texts); no JDK code is
//! copied (`docs/licences.md`). Over UTF-16 units, as Java compares `char`s:
//!
//! - A literal (text between quotes, `''` for a quote, or any non-letter)
//!   must equal the next `char`; a space separator (`Zs`) matches any (JDK
//!   23's lenient space matching, which JDK 25 -- the plugin's -- has and
//!   CI's JDK 21 does not: the fixture leaves those texts out and the unit
//!   tests pin JDK 25's answers).
//! - A numeric field (`y Y d D F w W u H k K h m s S`, and `M`/`L` under
//!   three letters) skips spaces and tabs, then reads `DecimalFormat`'s
//!   integer: `NaN`; or an optional `-`, then `∞` or digits (every BMP
//!   decimal digit, `Character.digit`), then an optional exponent `E`,
//!   optional `-`, digits. When the next element is also a numeric field
//!   (abutting fields: `yyyyMMdd`) the field fails if its start (before the
//!   blanks) plus its letter count is past the text's end, and otherwise its
//!   number may not run past that index.
//! - A text field matches the longest of its English names, ignoring case
//!   the way `String.regionMatches(true, ...)` does, with no skipping: `G`
//!   (`AD`, `BC`, `Anno Domini`, `Before Christ`), `M`/`L` from three letters
//!   (full and abbreviated months), `E` (full and abbreviated weekdays), `a`
//!   (`AM`, `PM`).
//! - `X` skips spaces and tabs, then takes `Z`, or a sign and ASCII digits:
//!   `hh` (`X`), `hhmm` (`XX`) or `hh:mm` (`XXX`), hours to 23 and minutes
//!   to 59.
//!
//! Differs: `Locale.ENGLISH` only (`DateRecognizerFilterFactory`'s `locale`
//! is Java's other locales' CLDR names, not ported), and the general and
//! RFC 822 time zone letters `z`/`Z` are refused at construction, since they
//! match the JDK's localized time zone names (`PST`, `Pacific Standard
//! Time`, ...), which are not ported.

use crate::attributes::AttributeSource;
use crate::java_character;
use crate::token_stream::{Accept, FilteringTokenFilter, TokenFilter, TokenStream};
use crate::AnalysisError;

/// `DateRecognizerFilter.DATE_TYPE`.
pub const DATE_TYPE: &str = "date";

/// One compiled pattern element.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Element {
    /// A literal `char`.
    Literal(u16),
    /// A pattern letter repeated `count` times.
    Field { letter: u8, count: usize },
}

/// The subset of `java.text.SimpleDateFormat` (with `Locale.ENGLISH`) that
/// `DateRecognizerFilter` needs: compiling a pattern and deciding whether a
/// text parses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimpleDateFormat {
    elements: Vec<Element>,
}

/// `DateFormat.getDateInstance(DateFormat.DEFAULT, Locale.ENGLISH)`'s pattern.
pub const ENGLISH_DEFAULT_DATE_PATTERN: &str = "MMM d, y";

const MONTHS: [&str; 24] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
];
const WEEKDAYS: [&str; 14] = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sun",
    "Mon",
    "Tue",
    "Wed",
    "Thu",
    "Fri",
    "Sat",
];
const ERAS: [&str; 4] = ["BC", "AD", "Before Christ", "Anno Domini"];
const AM_PM: [&str; 2] = ["AM", "PM"];

/// `SimpleDateFormat`'s pattern letters.
const PATTERN_LETTERS: &[u8] = b"GyMdkHmsSEDFwWahKzZYuXL";

impl SimpleDateFormat {
    /// `new SimpleDateFormat(pattern, Locale.ENGLISH)`: `IllegalArgument`
    /// for what Java refuses (an unknown letter, an unterminated quote,
    /// `XXXX`), and for the time zone letters `z`/`Z` (see the module docs).
    pub fn new(pattern: &str) -> Result<Self, AnalysisError> {
        let p: Vec<u16> = pattern.encode_utf16().collect();
        let mut elements = Vec::new();
        let mut i = 0;
        let mut quoted = false;
        while i < p.len() {
            let c = p[i];
            if c == u16::from(b'\'') {
                if p.get(i + 1) == Some(&c) {
                    elements.push(Element::Literal(c));
                    i += 2;
                } else {
                    quoted = !quoted;
                    i += 1;
                }
                continue;
            }
            let letter = u8::try_from(c).ok().filter(u8::is_ascii_alphabetic);
            match letter {
                Some(l) if !quoted => {
                    if !PATTERN_LETTERS.contains(&l) {
                        return Err(AnalysisError::IllegalArgument(format!(
                            "Illegal pattern character '{}'",
                            char::from(l)
                        )));
                    }
                    let count = p[i..].iter().take_while(|&&x| x == c).count();
                    if l == b'z' || l == b'Z' {
                        return Err(AnalysisError::IllegalArgument(format!(
                            "unsupported SimpleDateFormat pattern letter '{}': time zone names are \
                             the JDK's locale data, not ported",
                            char::from(l)
                        )));
                    }
                    if l == b'X' && count > 3 {
                        return Err(AnalysisError::IllegalArgument(format!(
                            "invalid ISO 8601 format: length={count}"
                        )));
                    }
                    elements.push(Element::Field { letter: l, count });
                    i += count;
                }
                _ => {
                    elements.push(Element::Literal(c));
                    i += 1;
                }
            }
        }
        if quoted {
            return Err(AnalysisError::IllegalArgument(
                "Unterminated quote".to_string(),
            ));
        }
        Ok(SimpleDateFormat { elements })
    }

    /// `DateFormat.getDateInstance(DateFormat.DEFAULT, Locale.ENGLISH)`.
    pub fn english_default() -> Self {
        // The constant pattern compiles.
        SimpleDateFormat::new(ENGLISH_DEFAULT_DATE_PATTERN).unwrap_or(SimpleDateFormat {
            elements: Vec::new(),
        })
    }

    /// `parse(text, new ParsePosition(0))`: the UTF-16 index the parse ended
    /// at, or `None` where Java returns `null` (and `parse(String)` throws
    /// `ParseException`).
    pub fn parse(&self, text: &str) -> Option<usize> {
        let t: Vec<u16> = text.encode_utf16().collect();
        let mut pos = 0;
        for (i, e) in self.elements.iter().enumerate() {
            pos = match *e {
                Element::Literal(c) => t
                    .get(pos)
                    .filter(|&&x| x == c || (is_space(c) && is_space(x)))
                    .map(|_| pos + 1)?,
                Element::Field { letter, count } => {
                    let abutting = matches!(
                        self.elements.get(i + 1),
                        Some(&Element::Field { letter, count }) if is_numeric(letter, count)
                    );
                    parse_field(&t, pos, letter, count, abutting)?
                }
            };
        }
        Some(pos)
    }
}

impl SimpleDateFormat {
    /// `DateFormat.parse(String)` returns rather than throwing
    /// `ParseException`: [`Self::parse`] succeeds past index 0.
    pub fn parses(&self, text: &str) -> bool {
        self.parse(text).is_some_and(|end| end > 0)
    }
}

/// A space separator (`Zs`): in a lenient parse, one in the pattern
/// matches any in the text (CLDR's patterns use U+00A0 and U+202F).
fn is_space(c: u16) -> bool {
    java_character::get_type(u32::from(c)) == java_character::SPACE_SEPARATOR
}

/// A field `SimpleDateFormat` reads as a number.
fn is_numeric(letter: u8, count: usize) -> bool {
    match letter {
        b'M' | b'L' => count < 3,
        _ => b"yYdDFwWuHkKhmsS".contains(&letter),
    }
}

/// Skips the spaces and tabs numeric and zone fields allow before them.
fn skip_blanks(t: &[u16], mut pos: usize) -> usize {
    while matches!(t.get(pos), Some(&0x20 | &0x09)) {
        pos += 1;
    }
    pos
}

fn parse_field(t: &[u16], start: usize, letter: u8, count: usize, abutting: bool) -> Option<usize> {
    match letter {
        b'G' => match_names(t, start, &ERAS),
        b'E' => match_names(t, start, &WEEKDAYS),
        b'a' => match_names(t, start, &AM_PM),
        b'M' | b'L' if count >= 3 => match_names(t, start, &MONTHS),
        b'X' => parse_iso_zone(t, skip_blanks(t, start), count),
        _ => {
            let pos = skip_blanks(t, start);
            // An abutting field's number ends `count` chars after where the
            // field began (before the blanks), and the field fails outright
            // when those chars run past the text's end, however short the
            // number in them.
            let limit = if abutting {
                let limit = start.checked_add(count)?;
                if limit > t.len() {
                    return None;
                }
                limit
            } else {
                t.len()
            };
            parse_number(&t[..limit.max(pos)], pos)
        }
    }
}

/// The longest of `names` matching at `pos`, ignoring case.
fn match_names(t: &[u16], pos: usize, names: &[&str]) -> Option<usize> {
    names
        .iter()
        .filter(|n| region_matches_ignore_case(t, pos, n))
        .map(|n| pos + n.len())
        .max()
}

/// `String.regionMatches(true, pos, name, 0, name.length())` against an
/// ASCII name: equal `char`s, or equal after `Character.toUpperCase`, or
/// after `toLowerCase` of that.
fn region_matches_ignore_case(t: &[u16], pos: usize, name: &str) -> bool {
    let Some(region) = t.get(pos..pos + name.len()) else {
        return false;
    };
    region.iter().zip(name.bytes()).all(|(&c, n)| {
        let (c, n) = (u32::from(c), u32::from(n));
        if c == n {
            return true;
        }
        let (uc, un) = (java_upper(c), java_upper(n));
        uc == un || java_character::to_lower_case(uc) == java_character::to_lower_case(un)
    })
}

/// `Character.toUpperCase(char)`: a BMP mapping, kept to a `char`.
fn java_upper(c: u32) -> u32 {
    java_character::to_upper_case(c) & 0xFFFF
}

/// A `char` `Character.digit(ch, 10)` reads: a BMP decimal digit.
fn is_digit(c: u16) -> bool {
    !(0xD800..=0xDFFF).contains(&c) && java_character::decimal_digit_value(u32::from(c)).is_some()
}

/// `DecimalFormat.parse` of an integer, English symbols, no grouping: the
/// index after it, or `None` if nothing parses at `pos`.
fn parse_number(t: &[u16], pos: usize) -> Option<usize> {
    let at = |i: usize, c: u8| t.get(i) == Some(&u16::from(c));
    if at(pos, b'N') && at(pos + 1, b'a') && at(pos + 2, b'N') {
        return Some(pos + 3);
    }
    let mut p = pos;
    if at(p, b'-') {
        p += 1;
    }
    if t.get(p) == Some(&0x221E) {
        return Some(p + 1);
    }
    let digits = |from: usize| from + t[from..].iter().take_while(|&&c| is_digit(c)).count();
    let end = digits(p);
    if end == p {
        return None;
    }
    p = end;
    if at(p, b'E') {
        let mut q = p + 1;
        if at(q, b'-') {
            q += 1;
        }
        let e = digits(q.min(t.len()));
        if e > q {
            p = e;
        }
    }
    Some(p)
}

/// An ISO 8601 zone (`X`, `XX`, `XXX`) at `pos`.
fn parse_iso_zone(t: &[u16], pos: usize, count: usize) -> Option<usize> {
    let ch = |i: usize| t.get(i).copied();
    if ch(pos) == Some(u16::from(b'Z')) {
        return Some(pos + 1);
    }
    if !matches!(ch(pos), Some(0x2B | 0x2D)) {
        return None;
    }
    let two = |i: usize| -> Option<u16> {
        let d = |k: usize| {
            ch(i + k)
                .filter(|c| (0x30..=0x39).contains(c))
                .map(|c| c - 0x30)
        };
        Some(d(0)? * 10 + d(1)?)
    };
    two(pos + 1).filter(|&h| h <= 23)?;
    match count {
        1 => Some(pos + 3),
        2 => two(pos + 3).filter(|&m| m <= 59).map(|_| pos + 5),
        _ => {
            if ch(pos + 3) != Some(u16::from(b':')) {
                return None;
            }
            two(pos + 4).filter(|&m| m <= 59).map(|_| pos + 6)
        }
    }
}

/// `DateRecognizerFilter.accept()`: the term parses with the format.
pub struct DateAccept(SimpleDateFormat);

impl Accept for DateAccept {
    fn accept(&mut self, a: &AttributeSource) -> Result<bool, AnalysisError> {
        Ok(self.0.parses(a.term()))
    }
}

/// `org.apache.lucene.analysis.miscellaneous.DateRecognizerFilter`: keeps
/// the tokens a [`SimpleDateFormat`] parses.
pub struct DateRecognizerFilter<I> {
    inner: FilteringTokenFilter<I, DateAccept>,
}

impl<I: TokenStream> DateRecognizerFilter<I> {
    /// `new DateRecognizerFilter(TokenStream)`:
    /// `DateFormat.getDateInstance(DateFormat.DEFAULT, Locale.ENGLISH)`.
    pub fn new(input: I) -> Self {
        Self::with_format(input, SimpleDateFormat::english_default())
    }

    /// `new DateRecognizerFilter(TokenStream, DateFormat)`.
    pub fn with_format(input: I, format: SimpleDateFormat) -> Self {
        DateRecognizerFilter {
            inner: FilteringTokenFilter::new(input, DateAccept(format)),
        }
    }
}

impl<I: TokenStream> TokenFilter for DateRecognizerFilter<I> {
    type Input = FilteringTokenFilter<I, DateAccept>;
    fn input(&self) -> &Self::Input {
        &self.inner
    }
    fn input_mut(&mut self) -> &mut Self::Input {
        &mut self.inner
    }
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        self.inner.increment_token()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::canned::{render, Canned};

    fn err(pattern: &str) -> String {
        match SimpleDateFormat::new(pattern) {
            Err(AnalysisError::IllegalArgument(m)) => m,
            other => panic!("{pattern}: {other:?}"),
        }
    }

    #[test]
    fn patterns_java_refuses_and_the_zone_letters() {
        assert_eq!(err("yyyy-q"), "Illegal pattern character 'q'");
        assert_eq!(err("'abc"), "Unterminated quote");
        assert_eq!(err("XXXX"), "invalid ISO 8601 format: length=4");
        assert!(err("HH z").contains("'z'"));
        assert!(err("Z").contains("'Z'"));
        // Letters inside quotes are literals.
        assert!(SimpleDateFormat::new("'q'd").unwrap().parses("q5"));
    }

    #[test]
    fn parse_ends_where_the_pattern_does() {
        let f = SimpleDateFormat::english_default();
        assert_eq!(f.parse("Jan 5, 2020 and more"), Some(11));
        assert_eq!(f.parse("Jan 5 2020"), None);
        assert!(!SimpleDateFormat::new("").unwrap().parses("x"));
        assert_eq!(SimpleDateFormat::new("").unwrap().parse("x"), Some(0));
        assert_eq!(SimpleDateFormat::new("dd").unwrap().parse("5"), Some(1));
        assert_eq!(SimpleDateFormat::new("ddMM").unwrap().parse("  512"), None);
        assert_eq!(SimpleDateFormat::new("XXX").unwrap().parse("+01"), None);
        assert_eq!(SimpleDateFormat::new("XX").unwrap().parse("+0160"), None);
        assert_eq!(SimpleDateFormat::new("X").unwrap().parse("x"), None);
    }

    /// JDK 25's answers (JDK 21 has no lenient space matching).
    #[test]
    fn space_separators_match_as_in_jdk_25() {
        let f = SimpleDateFormat::new("MMM d").unwrap();
        for zs in [
            '\u{A0}', '\u{202F}', '\u{2009}', '\u{3000}', '\u{2007}', '\u{1680}',
        ] {
            assert!(f.parses(&format!("Jan{zs}5")), "{zs:?}");
        }
        for other in ['\t', '\u{85}', '\u{200B}', '\u{2028}', '\u{B}'] {
            assert!(!f.parses(&format!("Jan{other}5")), "{other:?}");
        }
        let nbsp = SimpleDateFormat::new("d\u{A0}M").unwrap();
        assert!(nbsp.parses("5 1") && nbsp.parses("5\u{202F}1") && !nbsp.parses("5\t1"));
        assert!(!SimpleDateFormat::new("d-M").unwrap().parses("5\u{A0}-1"));
    }

    #[test]
    fn filter_keeps_dates() {
        let mut f = DateRecognizerFilter::with_format(
            Canned::parse("2020-01-05:0:10:1:1 x:11:12:1:1 1999-1-1:13:21:1:1|21|0"),
            SimpleDateFormat::new("yyyy-MM-dd").unwrap(),
        );
        assert_eq!(
            render(&mut f),
            "2020-01-05:0:10:1:1 1999-1-1:13:21:2:1|21|0"
        );
        let mut f = DateRecognizerFilter::new(Canned::parse("x:0:1:1:1|1|0"));
        assert_eq!(render(&mut f), "|1|1");
        assert_eq!(DATE_TYPE, "date");
    }
}
