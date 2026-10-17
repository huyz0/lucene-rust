//! M13 T13.3c against Lucene 10.5.0 (`fixtures/src/GenFreeTextSuggest.java`):
//! `FreeTextSuggester` answers as Lucene's does over a free-text corpus for
//! unigram to four-gram models, writes Lucene's bytes, and reads them back.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_search::suggest::free_text::FreeTextSuggester;
use lucene_search::suggest::input::InputIterator;
use lucene_search::suggest::lookup::{Lookup, LookupResult};
use lucene_store::{SliceInput, VecDataOutput};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/suggest_freetext")
}

/// `AnalysisRows.esc` undone.
fn unesc(s: &str) -> String {
    let mut out: Vec<u16> = Vec::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            let mut buf = [0u16; 2];
            out.extend_from_slice(c.encode_utf16(&mut buf));
            continue;
        }
        match it.next() {
            Some('u') => {
                let h: String = it.by_ref().take(4).collect();
                out.push(u16::from_str_radix(&h, 16).unwrap());
            }
            Some('t') => out.push(9),
            Some('n') => out.push(10),
            Some('r') => out.push(13),
            Some(o) => out.push(o as u16),
            None => out.push(u16::from(b'\\')),
        }
    }
    String::from_utf16_lossy(&out)
}

/// Rows as an `InputIterator`.
struct Rows {
    rows: Vec<String>,
    payloads: bool,
    contexts: Option<BTreeSet<Vec<u8>>>,
    next: usize,
}

impl Rows {
    fn new(rows: Vec<String>) -> Self {
        Rows {
            rows,
            payloads: false,
            contexts: None,
            next: 0,
        }
    }

    fn corpus() -> Self {
        let text = fs::read_to_string(dir().join("corpus.tsv")).unwrap();
        Self::new(text.lines().map(unesc).collect())
    }

    fn of(rows: &[&str]) -> Self {
        Self::new(rows.iter().map(|s| s.to_string()).collect())
    }
}

impl InputIterator for Rows {
    fn next(&mut self) -> Option<Vec<u8>> {
        let row = self.rows.get(self.next)?;
        self.next += 1;
        Some(row.as_bytes().to_vec())
    }
    fn weight(&self) -> i64 {
        1
    }
    fn payload(&self) -> Option<&[u8]> {
        self.payloads.then_some(b"p".as_slice())
    }
    fn has_payloads(&self) -> bool {
        self.payloads
    }
    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
        self.contexts.as_ref()
    }
    fn has_contexts(&self) -> bool {
        self.contexts.is_some()
    }
}

fn plain() -> Arc<Analyzer> {
    Arc::new(Analyzer::standard(None))
}

fn stopping() -> Arc<Analyzer> {
    let stops: HashSet<String> = ["the", "of", "a", "is", "to"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    Arc::new(Analyzer::standard(Some(&stops)))
}

fn keyword() -> Arc<Analyzer> {
    Arc::new(Analyzer::keyword())
}

fn free(index: Arc<Analyzer>, query: Arc<Analyzer>, grams: i32) -> FreeTextSuggester {
    FreeTextSuggester::with_options(index, query, grams, 0x1e).unwrap()
}

fn rendered(res: &[LookupResult]) -> String {
    res.iter()
        .map(|r| format!("{}\u{1e}{}", r.key, r.value))
        .collect::<Vec<_>>()
        .join("\u{1f}")
}

/// Every row of a lookup answer file against `lk`; the row count.
fn check_lookups(lk: &FreeTextSuggester, file: &str) -> usize {
    let text = fs::read_to_string(dir().join(file)).unwrap();
    let (mut checked, mut bad) = (0, Vec::new());
    for line in text.lines() {
        let c: Vec<&str> = line.splitn(4, '\t').collect();
        let query = unesc(c[0]);
        let num: usize = c[1].parse().unwrap();
        let got = lk.lookup(&query, None, false, num);
        checked += 1;
        if c[2] == "X" {
            // c[3] is `Class\tmessage`
            let f: Vec<&str> = c[3].splitn(2, '\t').collect();
            let message = unesc(f[1]);
            match got {
                Err(e) if e.to_string().contains(&message) => {}
                other => bad.push(format!("{file} {query:?}: want {c:?}, got {other:?}")),
            }
            continue;
        }
        let want: String = c[3]
            .split('\u{1f}')
            .filter(|r| !r.is_empty())
            .map(|r| {
                let f: Vec<&str> = r.split('\u{1e}').collect();
                format!("{}\u{1e}{}", unesc(f[0]), f[1])
            })
            .collect::<Vec<_>>()
            .join("\u{1f}");
        let got = rendered(&got.unwrap());
        if got != want {
            bad.push(format!(
                "{file} {query:?} n={num}\n  want {want:?}\n  got  {got:?}"
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {checked}:\n{}",
        bad.len(),
        bad[..bad.len().min(6)].join("\n")
    );
    checked
}

fn stored(lk: &FreeTextSuggester) -> Vec<u8> {
    let mut out = VecDataOutput::new();
    assert!(lk.store(&mut out).unwrap());
    out.into_inner()
}

/// Build, answer every query, write the same bytes as Lucene, and answer the
/// same after loading them.
fn check_config(name: &str, make: impl Fn() -> FreeTextSuggester) {
    let queries = fs::read_to_string(dir().join("queries.tsv"))
        .unwrap()
        .lines()
        .count();
    let mut lk = make();
    lk.build(&mut Rows::corpus()).unwrap();
    assert_eq!(check_lookups(&lk, &format!("{name}.tsv")), queries * 3);
    let count: u64 = fs::read_to_string(dir().join(format!("{name}.count")))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(lk.get_count().unwrap(), count, "{name} count");
    let bytes = stored(&lk);
    let expected = fs::read(dir().join(format!("{name}.bin"))).unwrap();
    assert_eq!(bytes, expected, "{name} store()");
    let mut again = make();
    assert!(again.load(&mut SliceInput::new(&expected)).unwrap());
    assert_eq!(again.get_count().unwrap(), count);
    assert_eq!(
        check_lookups(&again, &format!("{name}_reload.tsv")),
        queries * 3
    );
    assert_eq!(stored(&again), bytes, "{name} store() after load()");
}

#[test]
fn bigram_default() {
    check_config("bigram", || FreeTextSuggester::new(plain()));
}

#[test]
fn unigram() {
    check_config("unigram", || free(plain(), plain(), 1));
}

#[test]
fn trigram() {
    check_config("trigram", || free(plain(), plain(), 3));
}

#[test]
fn fourgram() {
    check_config("fourgram", || free(plain(), plain(), 4));
}

#[test]
fn stop_words_leave_holes_that_shingles_fill() {
    check_config("stop_bigram", || free(stopping(), stopping(), 2));
    check_config("stop_trigram", || free(stopping(), stopping(), 3));
}

#[test]
fn different_index_and_query_analyzers_and_another_separator() {
    check_config("mixed", || {
        FreeTextSuggester::with_options(plain(), stopping(), 3, b'|').unwrap()
    });
}

#[test]
fn refuses_what_lucene_refuses() {
    let text = fs::read_to_string(dir().join("errors.tsv")).unwrap();
    let rows: HashMap<&str, Vec<&str>> = text
        .lines()
        .map(|l| {
            let mut c = l.split('\t');
            (c.next().unwrap(), c.collect())
        })
        .collect();
    let message = |name: &str| unesc(rows[name].last().unwrap());
    let class = |name: &str| rows[name][0];
    let refused = |name: &str, e: lucene_store::Error| {
        let text = e.to_string();
        assert!(text.contains(&message(name)), "{name}: {text}");
    };

    for (name, grams, sep) in [
        ("grams_0", 0, 0x1e),
        ("grams_-1", -1, 0x1e),
        ("separator_high", 2, 0x80),
    ] {
        assert_eq!(class(name), "IllegalArgumentException");
        match FreeTextSuggester::with_options(plain(), plain(), grams, sep) {
            Err(e) => refused(name, e),
            Ok(_) => panic!("{name} was accepted"),
        }
    }
    assert_eq!(rows["separator_nul"], ["ok"]);
    FreeTextSuggester::with_options(plain(), plain(), 2, 0).unwrap();

    let mut s = FreeTextSuggester::new(plain());
    let mut payloads = Rows::of(&["a b"]);
    payloads.payloads = true;
    refused("payloads", s.build(&mut payloads).unwrap_err());
    let mut contexts = Rows::of(&["a b"]);
    contexts.contexts = Some(BTreeSet::new());
    refused("contexts", s.build(&mut contexts).unwrap_err());
    refused("no_input", s.build(&mut Rows::of(&[])).unwrap_err());
    refused(
        "no_tokens",
        s.build(&mut Rows::of(&["!!!", ""])).unwrap_err(),
    );

    // one token: built, and a lookup of its prefix finds it
    assert_eq!(rows["single_token"][0], "built");
    s.build(&mut Rows::of(&["a"])).unwrap();
    assert_eq!(s.get_count().unwrap(), 1);
    assert_eq!(
        rendered(&s.lookup("a", None, false, 3).unwrap()),
        format!("a\u{1e}{}", i64::MAX)
    );

    let mut k = FreeTextSuggester::with_options(keyword(), keyword(), 2, 0x1e).unwrap();
    k.build(&mut Rows::of(&["a\u{1e}b"])).unwrap();
    assert_eq!(rows["separator_in_token"][0], "built");
    assert_eq!(k.lookup("a", None, false, 3).unwrap().len(), 0);
    refused(
        "separator_in_tokens",
        k.build(&mut Rows::of(&["a\u{1e}b\u{1e}c d"])).unwrap_err(),
    );
    k.build(&mut Rows::of(&["a b c"])).unwrap();
    refused(
        "separator_in_query",
        k.lookup("x\u{1e}y z", None, false, 3).unwrap_err(),
    );

    let unbuilt = FreeTextSuggester::new(plain());
    refused("unbuilt", unbuilt.lookup("a", None, false, 3).unwrap_err());
    assert_eq!(unbuilt.get_count().unwrap(), 0);
    assert!(unbuilt.store(&mut VecDataOutput::new()).is_err());

    let mut ok = FreeTextSuggester::new(plain());
    ok.build(&mut Rows::corpus()).unwrap();
    let ctx: BTreeSet<Vec<u8>> = BTreeSet::new();
    refused(
        "lookup_contexts",
        ok.lookup("a", Some(&ctx), false, 3).unwrap_err(),
    );
    assert_eq!(class("get"), "UnsupportedOperationException");
    assert!(ok.get("net").is_err());
    // Java's TopNSearcher throws NoSuchElementException for a depth of 0
    assert_eq!(rows["lookup_zero"][1], "NoSuchElementException");
    assert!(ok.lookup("the", None, false, 0).unwrap().is_empty());

    // a model built with other settings
    let bin = fs::read(dir().join("bigram.bin")).unwrap();
    let mut other = free(plain(), plain(), 3);
    refused(
        "load_grams",
        other.load(&mut SliceInput::new(&bin)).unwrap_err(),
    );
    let mut other = FreeTextSuggester::with_options(plain(), plain(), 2, b'|').unwrap();
    refused(
        "load_separator",
        other.load(&mut SliceInput::new(&bin)).unwrap_err(),
    );
}
