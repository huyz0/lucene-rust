#![cfg(test)]
use super::*;
use crate::reader::StrReader;
use crate::token_stream::consume;

fn run<T: Tokenizer>(t: &mut T, text: &str) -> (Vec<(String, i32, i32)>, i32) {
    t.set_reader(Box::new(StrReader::new(text))).unwrap();
    let mut out = Vec::new();
    let end = consume(t, |a| {
        out.push((a.term().to_string(), a.start_offset(), a.end_offset()))
    })
    .unwrap();
    (out, end.end_offset())
}

#[test]
fn whitespace_letter_and_unicode_whitespace() {
    let mut ws = WhitespaceTokenizer::new();
    let (toks, end) = run(&mut ws, " a\u{00A0}b\tc😀d ");
    assert_eq!(
        toks,
        vec![("a\u{00A0}b".into(), 1, 4), ("c😀d".into(), 5, 9)]
    );
    assert_eq!(end, 10);
    let mut uw = UnicodeWhitespaceTokenizer::new();
    let (toks, _) = run(&mut uw, "a\u{00A0}b");
    assert_eq!(toks.len(), 2);
    let mut letters = LetterTokenizer::new();
    let (toks, _) = run(&mut letters, "ab1cd");
    assert_eq!(toks, vec![("ab".into(), 0, 2), ("cd".into(), 3, 5)]);
}

#[test]
fn max_token_len_cuts_and_validates() {
    let mut t = WhitespaceTokenizer::with_max_token_len(3).unwrap();
    let (toks, _) = run(&mut t, "abcdefg h");
    let terms: Vec<&str> = toks.iter().map(|t| t.0.as_str()).collect();
    assert_eq!(terms, vec!["abc", "def", "g", "h"]);
    assert!(WhitespaceTokenizer::with_max_token_len(0).is_err());
    let msg = LetterTokenizer::with_max_token_len(MAX_TOKEN_LENGTH_LIMIT + 1)
        .err()
        .unwrap()
        .to_string();
    assert!(msg.contains("maxTokenLen must be greater than 0"), "{msg}");
}

#[test]
fn surrogate_pairs_survive_the_io_buffer_boundary() {
    // 4095 'a's then an emoji: the high surrogate lands at index 4095 and
    // is held back for the next fill.
    let text = format!("{}😀 b", "a".repeat(4095));
    let mut t = CharTokenizer::from_predicate_with_max(NotJavaWhitespace, 10_000).unwrap();
    let (toks, end) = run(&mut t, &text);
    assert_eq!(toks[0].0.chars().last(), Some('😀'));
    assert_eq!(toks[0].2, 4097);
    assert_eq!(toks[1], ("b".into(), 4098, 4099));
    assert_eq!(end, 4099);
}

#[test]
fn predicates_from_closures() {
    let mut t = from_separator_char_predicate(|c| c == u32::from(b','));
    let (toks, _) = run(&mut t, "a,b c,,d");
    let terms: Vec<&str> = toks.iter().map(|t| t.0.as_str()).collect();
    assert_eq!(terms, vec!["a", "b c", "d"]);
    let mut t = CharTokenizer::from_predicate(|c: u32| c != u32::from(b' '));
    assert_eq!(run(&mut t, "x y").0.len(), 2);
}

/// A string reader that hides its text, so the tokenizer reads UTF-16
/// units a buffer at a time.
struct Hidden(StrReader);

impl CharReader for Hidden {
    fn read(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
        self.0.read(buf)
    }
}

/// The UTF-8 scan of a plain string is the buffered UTF-16 read, token
/// for token: terms, offsets, final offset, over supplementary
/// characters, no-break spaces and every cut length.
#[test]
fn the_utf8_scan_equals_the_buffered_read() {
    const ALPHABET: [&str; 16] = [
        "a",
        "b",
        "z",
        "A",
        "1",
        "-",
        "\u{7F}",
        " ",
        "\t",
        "\u{00A0}",
        "\u{3000}",
        "\u{1F600}",
        "\u{00E9}",
        "\n",
        "\u{2003}",
        "\u{10400}",
    ];
    fn tokens<P: TokenChar + Default>(
        text: &str,
        max: usize,
        hide: bool,
    ) -> (Vec<(String, i32, i32)>, i32) {
        let mut t = CharTokenizer::<P>::with_max_token_len(max).unwrap();
        let reader: Box<dyn CharReader> = if hide {
            Box::new(Hidden(StrReader::new(text)))
        } else {
            Box::new(StrReader::new(text))
        };
        t.set_reader(reader).unwrap();
        let mut out = Vec::new();
        let end = consume(&mut t, |a| {
            out.push((a.term().to_string(), a.start_offset(), a.end_offset()))
        })
        .unwrap();
        (out, end.end_offset())
    }
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for round in 0..400 {
        let len = (next() % 40) as usize + usize::from(round % 40 == 0) * 5000;
        let text: String = (0..len)
            .map(|_| ALPHABET[(next() % ALPHABET.len() as u64) as usize])
            .collect();
        for max in [1, 2, 3, 7, 255] {
            assert_eq!(
                tokens::<NotJavaWhitespace>(&text, max, false),
                tokens::<NotJavaWhitespace>(&text, max, true),
                "whitespace {max} {text:?}"
            );
            assert_eq!(
                tokens::<JavaLetter>(&text, max, false),
                tokens::<JavaLetter>(&text, max, true),
                "letter {max} {text:?}"
            );
            assert_eq!(
                tokens::<NotUnicodeWhitespace>(&text, max, false),
                tokens::<NotUnicodeWhitespace>(&text, max, true),
                "unicode whitespace {max} {text:?}"
            );
        }
    }
}

#[test]
fn reading_without_reset_is_an_error() {
    let mut t = WhitespaceTokenizer::new();
    assert!(t.increment_token().is_err());
}
