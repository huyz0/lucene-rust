//! `ConcatenateGraphFilter` produces its finite strings one at a time, as
//! Java's `LimitedFiniteStringsIterator` does: a peak-memory bound over an
//! input whose graph has 10 000 paths of ~6 000 labels each (draining them
//! into a list first held over 300 MB for 8 KB of text).
//!
//! Its own test binary, so the process's peak resident set (`VmHWM`) is
//! this test's alone.

#![cfg(target_os = "linux")]

use lucene_analysis::miscellaneous::{self as m, ConcatenateGraphFilter, WordDelimiterGraphFilter};
use lucene_analysis::reader::StrReader;
use lucene_analysis::token_stream::{TokenStream, Tokenizer};
use lucene_analysis::util::WhitespaceTokenizer;

/// The peak resident set so far, in KiB.
fn peak_rss_kib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with("VmHWM:")).unwrap();
    line.split_whitespace().nth(1).unwrap().parse().unwrap()
}

#[test]
fn finite_strings_are_produced_lazily() {
    let mut t = WhitespaceTokenizer::new();
    t.set_reader(Box::new(StrReader::new("a-b ".repeat(2000))))
        .unwrap();
    let w =
        WordDelimiterGraphFilter::new(t, m::GENERATE_WORD_PARTS | m::CATENATE_ALL, None).unwrap();
    let mut f = ConcatenateGraphFilter::new(w);
    f.reset().unwrap();
    let before = peak_rss_kib();
    let mut n = 0;
    while f.increment_token().unwrap() {
        n += 1;
    }
    f.end().unwrap();
    let grown = peak_rss_kib() - before;
    assert_eq!(n, 10_000, "DEFAULT_MAX_GRAPH_EXPANSIONS strings");
    assert!(
        grown < 64 * 1024,
        "peak resident set grew {} MB",
        grown / 1024
    );
}
