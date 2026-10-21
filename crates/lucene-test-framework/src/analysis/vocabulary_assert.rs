//! `VocabularyAssert`: vocabulary-based stemming tests -- every input word
//! must analyze to its expected stem.
//!
//! The text variants are ported. Java's zip variants unzip into a temp dir
//! and run these on the files; a caller here reads the zip entry itself
//! and passes its text.

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
