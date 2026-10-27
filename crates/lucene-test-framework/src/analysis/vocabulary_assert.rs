//! `VocabularyAssert`: vocabulary-based stemming tests -- every input word
//! must analyze to its expected stem.
//!
//! Java's zip variants unzip into a temp dir and read the named files back;
//! [`assert_vocabulary_zip`] and [`assert_vocabulary_tsv_zip`] read the same
//! entries straight out of the zip ([`crate::util::zip`]): the last entry of
//! a name wins, as the last write to that file does in Java. Text is decoded
//! as Java's `InputStreamReader(UTF_8)`, malformed bytes as U+FFFD.

use std::path::Path;

use lucene_analysis::{AnalysisError, Analyzer};
use lucene_util::java_random::JavaRandom;

use super::base_token_stream::check_one_term;

/// `BufferedReader.readLine()` over `text`: lines split at `\n`, `\r` or
/// `\r\n`, with no trailing empty line.
fn lines(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = text;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let end = rest.find(['\n', '\r']).unwrap_or(rest.len());
        let line = &rest[..end];
        rest = &rest[end..];
        if rest.starts_with("\r\n") {
            rest = &rest[2..];
        } else if !rest.is_empty() {
            rest = &rest[1..];
        }
        Some(line)
    })
}

/// `assertVocabulary(a, voc, out)`: line `i` of `voc` analyzes to line `i`
/// of `out`.
///
/// # Panics
/// When `out` runs out of lines, or a word analyzes to something else.
///
/// # Errors
/// What the analyzer returns.
pub fn assert_vocabulary(
    random: &mut JavaRandom,
    a: &Analyzer,
    voc: &str,
    out: &str,
) -> Result<(), AnalysisError> {
    let mut expected = lines(out);
    for input_word in lines(voc) {
        let expected_word = expected.next().expect("expected word is null");
        check_one_term(random, a, input_word, expected_word)?;
    }
    Ok(())
}

/// `assertVocabulary(a, vocOut)`: tab-separated `input\texpected` lines;
/// `#` comments and blank lines are skipped.
///
/// # Panics
/// On a line without a tab, or a word that analyzes to something else.
///
/// # Errors
/// What the analyzer returns.
pub fn assert_vocabulary_tsv(
    random: &mut JavaRandom,
    a: &Analyzer,
    voc_out: &str,
) -> Result<(), AnalysisError> {
    for line in lines(voc_out) {
        if line.starts_with('#') || line.trim().is_empty() {
            continue; // comment
        }
        let mut words = line.split('\t');
        let input = words.next().expect("split yields one part");
        let expected = words.next().expect("a tab-separated line");
        check_one_term(random, a, input, expected)?;
    }
    Ok(())
}

/// The text of entry `name` in the zip at `zip_file`.
fn zip_entry_text(zip_file: &Path, name: &str) -> Result<String, AnalysisError> {
    let io = |e: std::io::Error| AnalysisError::Io(format!("{}: {e}", zip_file.display()));
    let bytes = std::fs::read(zip_file).map_err(io)?;
    let entries = crate::util::zip::entries(&bytes).map_err(io)?;
    let entry = entries
        .iter()
        .rev()
        .find(|e| e.name == name && !e.is_directory())
        .ok_or_else(|| AnalysisError::Io(format!("NoSuchFileException: {name}")))?;
    Ok(String::from_utf8_lossy(&entry.data).into_owned())
}

/// `assertVocabulary(a, zipFile, voc, out)`: [`assert_vocabulary`] over the
/// zip's entries `voc` and `out`.
///
/// # Panics
/// As [`assert_vocabulary`].
///
/// # Errors
/// An unreadable or corrupt zip, a missing entry, or what the analyzer
/// returns.
pub fn assert_vocabulary_zip(
    random: &mut JavaRandom,
    a: &Analyzer,
    zip_file: &Path,
    voc: &str,
    out: &str,
) -> Result<(), AnalysisError> {
    let voc = zip_entry_text(zip_file, voc)?;
    let out = zip_entry_text(zip_file, out)?;
    assert_vocabulary(random, a, &voc, &out)
}

/// `assertVocabulary(a, zipFile, vocOut)`: [`assert_vocabulary_tsv`] over
/// the zip's entry `voc_out`.
///
/// # Panics
/// As [`assert_vocabulary_tsv`].
///
/// # Errors
/// An unreadable or corrupt zip, a missing entry, or what the analyzer
/// returns.
pub fn assert_vocabulary_tsv_zip(
    random: &mut JavaRandom,
    a: &Analyzer,
    zip_file: &Path,
    voc_out: &str,
) -> Result<(), AnalysisError> {
    let voc_out = zip_entry_text(zip_file, voc_out)?;
    assert_vocabulary_tsv(random, a, &voc_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::mock_analyzer::MockAnalyzer;

    #[test]
    fn line_splitting_matches_read_line() {
        let got: Vec<_> = lines("a\nb\r\nc\rd\n").collect();
        assert_eq!(got, ["a", "b", "c", "d"]);
        assert_eq!(lines("").count(), 0);
        assert_eq!(lines("\n\nx").collect::<Vec<_>>(), ["", "", "x"]);
    }

    #[test]
    fn vocabularies_pass_and_fail() {
        let mut r = JavaRandom::new(3);
        let a = MockAnalyzer::new(&mut r).build();
        assert_vocabulary(&mut r, &a, "Dogs\nCAT\n", "dogs\ncat\n").unwrap();
        assert_vocabulary_tsv(&mut r, &a, "# comment\n\nDogs\tdogs\n  \nX\tx").unwrap();
        let bad = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_vocabulary(&mut r, &a, "Dogs\n", "dog\n")
        }));
        assert!(bad.is_err());
        let short = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_vocabulary(&mut r, &a, "a\nb\n", "a\n")
        }));
        assert!(short.is_err());
    }
}
