//! Batch c33's A/B harness for the offset **unit**: what it costs to emit
//! `OffsetAttribute`'s Java `char` (UTF-16 code unit) offsets instead of the
//! UTF-8 byte offsets this crate used to emit.
//!
//! The two arms differ in exactly one thing -- how a token's `start_offset`/
//! `end_offset` are derived from the segmenter's byte indices -- so the delta
//! is the fix's own cost and nothing else. Arms alternate and the reported
//! figure is the **min of N** per arm, which is the standard shape for a
//! measurement whose noise is one-sided (scheduler, frequency).
//!
//! Run: `cargo run -p lucene-analysis --release --example c33_offset_ab`
#![allow(clippy::arithmetic_side_effects)]

use std::time::Instant;

use lucene_analysis::{tokenize, Token};
use unicode_segmentation::UnicodeSegmentation;

/// The **pre-c33** producer, kept here and nowhere else: byte offsets straight
/// off `unicode_word_indices`. This is the arm the fix replaced.
fn tokenize_byte_offsets(text: &str) -> Vec<Token> {
    text.unicode_word_indices()
        .map(|(start, word)| Token {
            term: word.to_string(),
            start_offset: start as i32,
            end_offset: (start + word.len()) as i32,
            position_increment: 1,
            position_length: 1,
        })
        .collect()
}

fn bench(label: &str, docs: &[String], rounds: usize) {
    let mut utf16 = u128::MAX;
    let mut bytes = u128::MAX;
    let mut tokens = 0usize;
    // Arm A is the shipped (UTF-16) producer, arm B the pre-c33 byte-offset
    // one. Which arm runs first alternates by round, so neither pays a
    // systematic first-touch/cache-warm penalty.
    let run = |f: fn(&str) -> Vec<Token>| -> (u128, usize) {
        let t = Instant::now();
        let (mut acc, mut n) = (0i64, 0usize);
        for d in docs {
            for tok in f(d) {
                acc += i64::from(tok.end_offset);
                n += 1;
            }
        }
        let ns = t.elapsed().as_nanos();
        std::hint::black_box(acc);
        (ns, n)
    };
    for round in 0..rounds {
        let (a, b) = if round % 2 == 0 {
            let a = run(tokenize);
            let b = run(tokenize_byte_offsets);
            (a, b)
        } else {
            let b = run(tokenize_byte_offsets);
            let a = run(tokenize);
            (a, b)
        };
        utf16 = utf16.min(a.0);
        bytes = bytes.min(b.0);
        tokens = a.1 / docs.len();
    }
    let per_doc = |ns: u128| ns as f64 / docs.len() as f64 / 1000.0;
    println!(
        "{label:<28} utf16 {:>8.3} us/doc | bytes {:>8.3} us/doc | delta {:+.3} us/doc \
         ({:+.1}%) | {tokens} tokens/doc",
        per_doc(utf16),
        per_doc(bytes),
        per_doc(utf16) - per_doc(bytes),
        (utf16 as f64 / bytes as f64 - 1.0) * 100.0,
    );
}

fn main() {
    const DOCS: usize = 200;
    const ROUNDS: usize = 25;

    // A realistic ASCII body -- the overwhelmingly common case, and the one
    // `tokenize`'s `is_ascii()` fast path exists for.
    let ascii_sentence = "the quick brown fox jumps over the lazy dog while a search engine \
         indexes every single one of these terms into its postings list ";
    let ascii: Vec<String> = (0..DOCS).map(|_| ascii_sentence.repeat(8)).collect();

    // The same body with one accented word, so `is_ascii()` fails for the
    // whole document and every token pays the per-scalar sum: the worst case
    // for the fast path, not the average non-ASCII document.
    let one_accent: Vec<String> = (0..DOCS)
        .map(|_| format!("caf\u{e9} {}", ascii_sentence.repeat(8)))
        .collect();

    // Genuinely non-ASCII text: French plus CJK plus an astral letter.
    let mixed_sentence = "le renard brun saute par-dessus le chien paresseux \u{4e16}\u{754c} \
         caf\u{e9} na\u{ef}ve \u{1d400}\u{1d401} r\u{e9}sum\u{e9} stra\u{df}e ";
    let mixed: Vec<String> = (0..DOCS).map(|_| mixed_sentence.repeat(8)).collect();

    bench("ascii body", &ascii, ROUNDS);
    bench("ascii + one accent", &one_accent, ROUNDS);
    bench("mixed latin1/cjk/astral", &mixed, ROUNDS);
}
