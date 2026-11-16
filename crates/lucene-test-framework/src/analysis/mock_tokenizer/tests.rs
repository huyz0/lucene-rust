#![cfg(test)]
use super::*;
use lucene_analysis::token_stream::consume;

fn tokens(t: &mut MockTokenizer) -> Vec<(String, i32, i32)> {
    let mut out = Vec::new();
    consume(t, |a| {
        out.push((a.term().to_string(), a.start_offset(), a.end_offset()))
    })
    .unwrap();
    out
}

#[test]
fn whitespace_lower_cases_and_reports_utf16_offsets() {
    let mut t = MockTokenizer::default()
        .with_text("Ab \u{1F600}x  C")
        .unwrap();
    assert_eq!(
        tokens(&mut t),
        vec![
            ("ab".to_string(), 0, 2),
            ("\u{1F600}x".to_string(), 3, 6),
            ("c".to_string(), 8, 9)
        ]
    );
    assert_eq!(t.stream_state(), StreamState::Close);
}

#[test]
fn the_state_machine_catches_workflow_violations() {
    let mut t = MockTokenizer::default().with_text("a").unwrap();
    let e = t.increment_token().unwrap_err();
    assert!(e
        .to_string()
        .contains("incrementToken() called while in wrong state: SETREADER"));
    t.reset().unwrap();
    assert!(t
        .reset()
        .unwrap_err()
        .to_string()
        .contains("double reset()"));
    // as Java's Tokenizer, the second reset swapped in the illegal-state
    // reader
    assert!(t
        .increment_token()
        .unwrap_err()
        .to_string()
        .contains("reset() called multiple times"));
    let mut t = MockTokenizer::default().with_text("a b").unwrap();
    t.reset().unwrap();
    assert!(t.increment_token().unwrap());
    assert!(t
        .end()
        .unwrap_err()
        .to_string()
        .contains("end() called in wrong state=INCREMENT!"));
    let mut t = MockTokenizer::default().with_text("a").unwrap();
    t.reset().unwrap();
    assert!(t
        .close()
        .unwrap_err()
        .to_string()
        .contains("close() called in wrong state: RESET"));
    let mut t = MockTokenizer::default().with_text("a").unwrap();
    t.reset().unwrap();
    // a second reader while the first is live
    assert!(t.set_reader(Box::new(StrReader::new("b"))).is_err());
    t.input = TokenizerInput::new();
    assert!(t
        .set_reader(Box::new(StrReader::new("b")))
        .unwrap_err()
        .to_string()
        .contains("setReader() called in wrong state: RESET"));
    for s in [
        StreamState::Increment,
        StreamState::IncrementFalse,
        StreamState::End,
    ] {
        assert!(!s.to_string().is_empty());
    }
}

#[test]
fn checks_can_be_disabled() {
    let mut t = MockTokenizer::default().with_text("a b").unwrap();
    t.set_enable_checks(false);
    // out-of-order calls pass: increment before reset (no reader yet is
    // the reader's error, not the state machine's), end mid-stream
    assert!(!t
        .increment_token()
        .unwrap_err()
        .to_string()
        .contains("wrong state"));
    t.reset().unwrap();
    assert!(t.increment_token().unwrap());
    t.end().unwrap();
    t.close().unwrap();
    t.set_reader(Box::new(StrReader::new("y"))).unwrap();
    t.reset().unwrap();
    t.reset().unwrap();
    t.close().unwrap();
    // a closed tokenizer without a reader still answers (no input: an error from the reader)
    t.set_reader(Box::new(StrReader::new("x"))).unwrap();
    t.close().unwrap();
    assert!(t.increment_token().is_err());
}

/// A reader serving fixed UTF-16 units (unpaired surrogates included).
struct Units(Vec<u16>, usize);

impl CharReader for Units {
    fn read(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
        let n = buf.len().min(self.0.len() - self.1);
        buf[..n].copy_from_slice(&self.0[self.1..self.1 + n]);
        self.1 += n;
        Ok(n)
    }
}

fn first_error(units: &[u16]) -> String {
    let mut t = MockTokenizer::default();
    t.set_reader(Box::new(Units(units.to_vec(), 0))).unwrap();
    t.reset().unwrap();
    loop {
        match t.increment_token() {
            Ok(true) => continue,
            Ok(false) => return String::new(),
            Err(e) => return e.to_string(),
        }
    }
}

#[test]
fn unpaired_surrogates_always_fail() {
    assert!(first_error(&[0xDC00]).contains("unpaired low surrogate: dc00"));
    assert!(first_error(&[0xD800, 0x61]).contains("unpaired high surrogate: d800, followed by: 61"));
    assert!(first_error(&[0xD800]).contains("stream ends with unpaired high surrogate: d800"));
    assert_eq!(first_error(&[0x61]), "");
}

/// A char filter whose corrections are arbitrary: maps every offset
/// through `f`.
struct Correcting(StrReader, fn(i32) -> i32);

impl CharReader for Correcting {
    fn read(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
        self.0.read(buf)
    }
    fn correct_offset(&self, off: i32) -> i32 {
        (self.1)(off)
    }
}

fn offset_error(text: &str, f: fn(i32) -> i32) -> String {
    let mut t = MockTokenizer::default();
    t.set_reader(Box::new(Correcting(StrReader::new(text), f)))
        .unwrap();
    t.reset().unwrap();
    loop {
        match t.increment_token() {
            Ok(true) => continue,
            Ok(false) => return String::new(),
            Err(e) => return e.to_string(),
        }
    }
}

#[test]
fn bad_offset_corrections_always_fail() {
    assert!(offset_error("ab", |o| o - 1).contains("invalid start offset: -1"));
    assert!(offset_error("ab", |o| if o == 0 { 0 } else { -5 }).contains("invalid end offset: -5"));
    assert!(offset_error("a b", |o| if o == 2 { 0 } else { o + 1 })
        .contains("start offset went backwards: 0"));
    assert!(offset_error("ab", |o| if o == 0 { 3 } else { 1 })
        .contains("end offset: 1 is before start offset: 3"));
}

#[test]
fn max_token_length_splits_and_the_automaton_state_carries_over() {
    let mut t = MockTokenizer::with_max_token_length(whitespace(), false, 3)
        .with_text("abcdefg h")
        .unwrap();
    let got: Vec<String> = tokens(&mut t).into_iter().map(|t| t.0).collect();
    assert_eq!(got, ["abc", "def", "g", "h"]);
    // KEYWORD over empty input: no token
    let mut t = MockTokenizer::new(keyword(), false).with_text("").unwrap();
    assert!(tokens(&mut t).is_empty());
    let mut t = MockTokenizer::new(simple(), true)
        .with_text("Ab1\u{4e00}")
        .unwrap();
    let got: Vec<String> = tokens(&mut t).into_iter().map(|t| t.0).collect();
    assert_eq!(got, ["ab", "\u{4e00}"]);
}
