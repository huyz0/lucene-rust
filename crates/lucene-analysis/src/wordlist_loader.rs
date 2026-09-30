//! `org.apache.lucene.analysis.WordlistLoader`: the stopword and dictionary
//! file formats analyzers load.
//!
//! Java reads a `Reader` (or an `InputStream` plus a `Charset`) through a
//! `BufferedReader`. The port takes any `std::io::Read` of **UTF-8** -- every
//! Lucene resource file is UTF-8, and Java's decoding reader rejects malformed
//! input just as this does -- and reproduces the three details that make a
//! hand-rolled loader drift from Java's:
//!
//! - lines end at `\n`, `\r` or `\r\n` (`BufferedReader.readLine`), not only
//!   `\n`;
//! - `String.trim()` strips every char `<= U+0020` (ASCII controls and the
//!   space), not Unicode whitespace, and nothing else;
//! - the Snowball format splits on the regex `\s+`, whose `\s` is ASCII-only
//!   (`[ \t\n\x0B\f\r]`).
//!
//! The `Charset`-taking overloads have no counterpart: decode to UTF-8 first.
//! `getStemDict` returns a `HashMap` where Java returns a `CharArrayMap` (see
//! [`crate::CharArraySet`] for why).

use std::collections::HashMap;
use std::io::Read;

use crate::{AnalysisError, CharArraySet};

/// `WordlistLoader.INITIAL_CAPACITY`.
const INITIAL_CAPACITY: usize = 16;

fn read_all(mut reader: impl Read) -> Result<String, AnalysisError> {
    let mut s = String::new();
    reader
        .read_to_string(&mut s)
        .map_err(|e| AnalysisError::Io(e.to_string()))?;
    Ok(s)
}

/// `BufferedReader.readLine()` over the whole text: splits at `\n`, `\r` and
/// `\r\n`; a final line without a terminator is still a line, an empty
/// trailing one is not.
fn java_lines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let b = text.as_bytes();
    let mut start = 0;
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'\n' => {
                lines.push(&text[start..i]);
                i += 1;
                start = i;
            }
            b'\r' => {
                lines.push(&text[start..i]);
                i += 1;
                if i < b.len() && b[i] == b'\n' {
                    i += 1;
                }
                start = i;
            }
            _ => i += 1,
        }
    }
    if start < b.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// `String.trim()`: strips chars `<= ' '` from both ends.
fn java_trim(s: &str) -> &str {
    s.trim_matches(|c: char| c <= ' ')
}

/// `getWordSet(Reader, CharArraySet)`: one word per line, trimmed, blank
/// lines skipped, added to `result`.
pub fn get_word_set_into(
    reader: impl Read,
    result: &mut CharArraySet,
) -> Result<(), AnalysisError> {
    let text = read_all(reader)?;
    for line in java_lines(&text) {
        let word = java_trim(line);
        if word.is_empty() {
            continue;
        }
        result.add(word);
    }
    Ok(())
}

/// `getWordSet(Reader)`: a new case-sensitive set.
pub fn get_word_set(reader: impl Read) -> Result<CharArraySet, AnalysisError> {
    let mut set = CharArraySet::with_capacity(INITIAL_CAPACITY, false);
    get_word_set_into(reader, &mut set)?;
    Ok(set)
}

/// `getWordSet(Reader, String comment, CharArraySet)`: as
/// [`get_word_set_into`], skipping lines that start with `comment` (tested
/// before trimming, as Java does).
pub fn get_word_set_with_comment_into(
    reader: impl Read,
    comment: &str,
    result: &mut CharArraySet,
) -> Result<(), AnalysisError> {
    let text = read_all(reader)?;
    for line in java_lines(&text) {
        if !line.starts_with(comment) {
            let word = java_trim(line);
            if word.is_empty() {
                continue;
            }
            result.add(word);
        }
    }
    Ok(())
}

/// `getWordSet(Reader, String comment)`.
pub fn get_word_set_with_comment(
    reader: impl Read,
    comment: &str,
) -> Result<CharArraySet, AnalysisError> {
    let mut set = CharArraySet::with_capacity(INITIAL_CAPACITY, false);
    get_word_set_with_comment_into(reader, comment, &mut set)?;
    Ok(set)
}

/// `getSnowballWordSet(Reader, CharArraySet)`: the Snowball stopword format
/// -- `|` starts a comment, several whitespace-separated words per line.
pub fn get_snowball_word_set_into(
    reader: impl Read,
    result: &mut CharArraySet,
) -> Result<(), AnalysisError> {
    let text = read_all(reader)?;
    for mut line in java_lines(&text) {
        if let Some(comment) = line.find('|') {
            line = &line[..comment];
        }
        // `line.split("\\s+")`, `\s` being ASCII-only in Java regexes.
        for word in line.split([' ', '\t', '\n', '\u{0B}', '\u{0C}', '\r']) {
            if !word.is_empty() {
                result.add(word);
            }
        }
    }
    Ok(())
}

/// `getSnowballWordSet(Reader)`.
pub fn get_snowball_word_set(reader: impl Read) -> Result<CharArraySet, AnalysisError> {
    let mut set = CharArraySet::with_capacity(INITIAL_CAPACITY, false);
    get_snowball_word_set_into(reader, &mut set)?;
    Ok(set)
}

/// `getStemDict(Reader, CharArrayMap)`: `word<TAB>stem` per line. A line
/// without a tab is an error, as Java's `wordstem[1]` is an
/// `ArrayIndexOutOfBoundsException`.
pub fn get_stem_dict(reader: impl Read) -> Result<HashMap<String, String>, AnalysisError> {
    let text = read_all(reader)?;
    let mut result = HashMap::new();
    for line in java_lines(&text) {
        let Some((word, stem)) = line.split_once('\t') else {
            return Err(AnalysisError::IllegalArgument(format!(
                "stem dictionary line has no tab: {line:?}"
            )));
        };
        result.insert(word.to_string(), stem.to_string());
    }
    Ok(result)
}

/// `getLines(InputStream, Charset)`: non-blank, non-`#` lines, trimmed, with
/// a leading byte-order mark stripped (Java checks for it on every line read
/// until the first line is kept).
pub fn get_lines(reader: impl Read) -> Result<Vec<String>, AnalysisError> {
    let text = read_all(reader)?;
    let mut lines: Vec<String> = Vec::new();
    for mut word in java_lines(&text) {
        // skip initial bom marker
        if lines.is_empty() {
            if let Some(rest) = word.strip_prefix('\u{FEFF}') {
                word = rest;
            }
        }
        // skip comments
        if word.starts_with('#') {
            continue;
        }
        let word = java_trim(word);
        // skip blank lines
        if word.is_empty() {
            continue;
        }
        lines.push(word.to_string());
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(set: &CharArraySet) -> Vec<&str> {
        let mut v: Vec<&str> = set.iter().collect();
        v.sort();
        v
    }

    #[test]
    fn word_set_trims_like_java_and_splits_every_line_ending() {
        let text = " alpha \r\nbeta\rgamma\n\n\t\u{2003}delta\u{2003}\n";
        let set = get_word_set(text.as_bytes()).unwrap();
        // U+2003 EM SPACE is not <= ' ', so Java's trim keeps it.
        assert_eq!(
            sorted(&set),
            vec!["alpha", "beta", "gamma", "\u{2003}delta\u{2003}"]
        );
    }

    #[test]
    fn comment_lines_are_tested_before_trimming() {
        let text = "#c\n  #kept\nword\n";
        let set = get_word_set_with_comment(text.as_bytes(), "#").unwrap();
        assert_eq!(sorted(&set), vec!["#kept", "word"]);
    }

    #[test]
    fn snowball_format() {
        let text = "i | me\nme my  myself\t| comment\n | only\u{00A0}x\n";
        let set = get_snowball_word_set(text.as_bytes()).unwrap();
        // NBSP is not in Java's ASCII \s.
        assert_eq!(sorted(&set), vec!["i", "me", "my", "myself"]);
        let set = get_snowball_word_set("a\u{00A0}b c".as_bytes()).unwrap();
        assert_eq!(sorted(&set), vec!["a\u{00A0}b", "c"]);
    }

    #[test]
    fn stem_dict_and_its_error() {
        let d = get_stem_dict("walked\twalk\nran\trun\textra".as_bytes()).unwrap();
        assert_eq!(d["walked"], "walk");
        assert_eq!(d["ran"], "run\textra");
        assert!(get_stem_dict("notab".as_bytes()).is_err());
    }

    #[test]
    fn lines_strip_bom_comments_and_blanks() {
        let text = "\u{FEFF}# c\n\u{FEFF}first \n\n#x\n second\n\u{FEFF}third";
        let lines = get_lines(text.as_bytes()).unwrap();
        assert_eq!(lines, vec!["first", "second", "\u{FEFF}third"]);
    }

    #[test]
    fn invalid_utf8_is_an_error() {
        assert!(get_word_set(&[0xffu8, 0xfe][..]).is_err());
        let mut into = CharArraySet::new(true);
        get_word_set_into("A\n".as_bytes(), &mut into).unwrap();
        assert!(into.contains("a"));
        let mut into = CharArraySet::new(false);
        get_word_set_with_comment_into("x\n".as_bytes(), "#", &mut into).unwrap();
        get_snowball_word_set_into("y z".as_bytes(), &mut into).unwrap();
        assert_eq!(sorted(&into), vec!["x", "y", "z"]);
    }
}
