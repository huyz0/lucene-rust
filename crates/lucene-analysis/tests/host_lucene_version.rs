//! Analysis behaviour that changed between Lucene 10.5 and 10.6 follows the
//! Lucene of the node ([`Version::host`]): what `for_lucene` selects per
//! component, and what a component built with no say takes from the host.

mod support;

use std::sync::Arc;

use lucene_analysis::compound::{CompoundSizes, DictionaryCompoundWordTokenFilter};
use lucene_analysis::lang::th::{default_stop_set, ThaiAnalyzer};
use lucene_analysis::token_stream::consume;
use lucene_analysis::util::WhitespaceTokenizer;
use lucene_analysis::{Analyzer, CharArraySet};
use lucene_util::version::Version;
use support::{chain, comps};

fn types(analyzer: &Analyzer, text: &str) -> Vec<(String, String)> {
    let mut ts = analyzer.token_stream("f", text).unwrap();
    let mut out = Vec::new();
    consume(&mut ts, |a| {
        out.push((a.term().to_string(), a.token_type().to_string()));
    })
    .unwrap();
    out
}

fn dictionary() -> Arc<CharArraySet> {
    Arc::new(CharArraySet::from_words(
        ["rind", "fleisch", "wurst"],
        true,
    ))
}

fn compound(host: Option<Version>) -> Analyzer {
    chain(move || {
        let f = DictionaryCompoundWordTokenFilter::dictionary_with(
            WhitespaceTokenizer::new(),
            dictionary(),
            CompoundSizes::default(),
            false,
        );
        comps(match host {
            Some(h) => f.for_lucene(h),
            None => f,
        })
    })
}

#[test]
fn compound_subwords_are_typed_from_10_6() {
    let text = "rindfleischwurst";
    let v10_6 = types(&compound(Some(Version::LUCENE_10_6_0)), text);
    assert_eq!(
        v10_6,
        [
            ("rindfleischwurst", "word"),
            ("rind", "subword"),
            ("fleisch", "subword"),
            ("wurst", "subword"),
        ]
        .map(|(a, b)| (a.to_string(), b.to_string()))
    );
    // 10.5.1: every subword keeps the compound's own type
    let v10_5 = types(&compound(Some(Version::LUCENE_10_5_1)), text);
    assert_eq!(v10_5.len(), 4);
    assert!(v10_5.iter().all(|(_, t)| t == "word"), "{v10_5:?}");
}

#[test]
fn thai_analyzer_chain_and_stop_words_follow_the_version() {
    let old = default_stop_set(Version::LUCENE_10_5_1);
    let new = default_stop_set(Version::LUCENE_10_6_0);
    // 10.6 curated the list (#16718): `ไป` (go) went, `สำหรับ` (sara am) came
    assert!(old.contains("ไป") && !new.contains("ไป"));
    assert!(!old.contains("สำหรับ") && new.contains("สำหรับ"));
    assert_eq!(
        ThaiAnalyzer::for_lucene(Version::LUCENE_10_5_1)
            .stopwords()
            .len(),
        old.len()
    );
    assert_eq!(
        ThaiAnalyzer::for_lucene(Version::LUCENE_10_6_0)
            .stopwords()
            .len(),
        new.len()
    );
    // normalisation: 10.6 recomposes sara am from U+0E4D U+0E32 (for the index
    // and the query alike); 10.5 leaves the text alone
    let norm = |host| {
        let a = Analyzer::new(ThaiAnalyzer::for_lucene(host));
        a.normalize("f", "ท\u{0E4D}\u{0E32}").unwrap()
    };
    assert_eq!(norm(Version::LUCENE_10_5_1), "ท\u{0E4D}\u{0E32}".as_bytes());
    assert_eq!(norm(Version::LUCENE_10_6_0), "ท\u{0E33}".as_bytes());
}

#[test]
fn defaults_follow_the_host_lucene() {
    // This binary is its own process: it may set the host once.
    Version::set_host(Version::LUCENE_10_5_1);
    assert_eq!(Version::host(), Version::LUCENE_10_5_1);

    let v = types(&compound(None), "rindfleischwurst");
    assert!(v.iter().all(|(_, t)| t == "word"), "{v:?}");
    assert_eq!(
        ThaiAnalyzer::default().stopwords().len(),
        default_stop_set(Version::LUCENE_10_5_1).len()
    );
    // A release newer than the library knows counts as the newest known
    Version::set_host(Version::from_bits(11, 0, 0).unwrap());
    assert_eq!(Version::host(), Version::LATEST);
    Version::set_host(Version::LUCENE_10_6_0);
    let v = types(&compound(None), "rindfleischwurst");
    assert_eq!(v[1].1, "subword");
}
