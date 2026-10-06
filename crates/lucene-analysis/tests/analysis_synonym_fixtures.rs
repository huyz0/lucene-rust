//! M11 T11.5: synonyms against `fixtures/src/GenAnalysisSynonym.java`.
//!
//! - `maps.tsv`: the Solr and WordNet rule files of `fixtures/corpus/` parsed
//!   with each option set, every map entry (key, keepOrig, outputs) in FST
//!   order.
//! - `parse.tsv`: small rule texts, bad ones included, through both parsers
//!   and every `dedup`/`expand`: the map, or the exception.
//! - `<chain>.tsv`: `SynonymGraphFilter`/`SynonymFilter` chains (with
//!   `FlattenGraphFilter`, stop-word holes, ignoreCase) over
//!   `fixtures/corpus/analysis-synonym.txt`, row for row.

mod support;

use std::sync::Arc;

use lucene_analysis::core_analysis::FlattenGraphFilter;
use lucene_analysis::synonym::{
    SolrSynonymParser, SynonymFilter, SynonymGraphFilter, SynonymMap, SynonymParseError,
    WordnetSynonymParser,
};
use lucene_analysis::util::WhitespaceTokenizer;
use lucene_analysis::{
    Analyzer, CharArraySet, LowerCaseFilter, StandardTokenizer, StopFilter, ENGLISH_STOP_WORDS,
};
use support::{chain, comps, esc, exception_name, unesc};

fn ws_lower() -> Analyzer {
    chain(|| comps(LowerCaseFilter::new(WhitespaceTokenizer::new())))
}

fn ws_plain() -> Analyzer {
    chain(|| comps(WhitespaceTokenizer::new()))
}

fn std_lower() -> Analyzer {
    chain(|| comps(LowerCaseFilter::new(StandardTokenizer::new())))
}

fn rules(name: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/../../fixtures/corpus/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

fn solr(rules: &str, dedup: bool, expand: bool, a: &Analyzer) -> Result<SynonymMap, String> {
    let mut p = SolrSynonymParser::new(dedup, expand, a);
    p.parse(rules).map_err(|e| parse_error(&e))?;
    p.build().map_err(|e| exception_name(&e).to_string())
}

fn wordnet(rules: &str, dedup: bool, expand: bool, a: &Analyzer) -> Result<SynonymMap, String> {
    let mut p = WordnetSynonymParser::new(dedup, expand, a);
    p.parse(rules).map_err(|e| parse_error(&e))?;
    p.build().map_err(|e| exception_name(&e).to_string())
}

/// The generator's exception row for a parse failure.
fn parse_error(e: &SynonymParseError) -> String {
    match e {
        SynonymParseError::InvalidRule { line, .. } => {
            format!("ParseException\tInvalid synonym rule at line {line}")
        }
        SynonymParseError::Malformed { .. } => "StringIndexOutOfBoundsException".to_string(),
    }
}

/// The generator's `dump`.
fn dump(m: &SynonymMap) -> Vec<String> {
    let mut out = vec![format!(
        "#\t{}\t{}",
        m.max_horizontal_context,
        m.word_count()
    )];
    for (key, e) in m.entries() {
        let outs: Vec<String> = e.ords.iter().map(|&o| esc(m.word(o))).collect();
        out.push(format!(
            "{}\t{}\t{}",
            esc(&key),
            u8::from(e.keep_orig),
            outs.join("|")
        ));
    }
    out
}

struct Maps {
    solr_expand: Arc<SynonymMap>,
    solr_collapse: Arc<SynonymMap>,
    solr_nodedup: Arc<SynonymMap>,
    solr_cased: Arc<SynonymMap>,
    wordnet_expand: Arc<SynonymMap>,
    wordnet_collapse: Arc<SynonymMap>,
}

fn maps() -> Maps {
    let s = rules("synonyms-solr.txt");
    let w = rules("synonyms-wordnet.txt");
    let m = |r: Result<SynonymMap, String>| Arc::new(r.unwrap());
    Maps {
        solr_expand: m(solr(&s, true, true, &ws_lower())),
        solr_collapse: m(solr(&s, true, false, &ws_lower())),
        solr_nodedup: m(solr(&s, false, true, &ws_lower())),
        solr_cased: m(solr(&s, true, true, &ws_plain())),
        wordnet_expand: m(wordnet(&w, true, true, &std_lower())),
        wordnet_collapse: m(wordnet(&w, true, false, &std_lower())),
    }
}

#[test]
fn parsed_maps_match_lucene_entry_for_entry() {
    let m = maps();
    let mut actual = Vec::new();
    for (name, map) in [
        ("solr_cased", &m.solr_cased),
        ("solr_collapse", &m.solr_collapse),
        ("solr_expand", &m.solr_expand),
        ("solr_nodedup", &m.solr_nodedup),
        ("wordnet_collapse", &m.wordnet_collapse),
        ("wordnet_expand", &m.wordnet_expand),
    ] {
        actual.push(format!("M\t{name}"));
        actual.extend(dump(map));
    }
    let expected =
        std::fs::read_to_string(support::data_dir("analysis_synonym") + "maps.tsv").unwrap();
    let expected: Vec<&str> = expected.lines().collect();
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(a, e, "maps.tsv row {i}");
    }
    assert_eq!(actual.len(), expected.len());
    assert!(actual.len() > 150);
}

#[test]
fn parsers_match_lucene_on_good_and_bad_rules() {
    let text =
        std::fs::read_to_string(support::data_dir("analysis_synonym") + "parse.tsv").unwrap();
    let analyzer = chain(|| {
        comps(StopFilter::new(
            LowerCaseFilter::new(WhitespaceTokenizer::new()),
            Arc::new(CharArraySet::from_words(["the"], false)),
        ))
    });
    let mut actual = Vec::new();
    let mut cases = 0;
    for row in text.lines().filter(|r| r.starts_with("C\t")) {
        let f: Vec<&str> = row.splitn(5, '\t').collect();
        let (format, dedup, expand, rules) = (f[1], f[2] == "1", f[3] == "1", unesc(f[4]));
        actual.push(row.to_string());
        let built = if format == "solr" {
            solr(&rules, dedup, expand, &analyzer)
        } else {
            wordnet(&rules, dedup, expand, &analyzer)
        };
        match built {
            Ok(m) => actual.extend(dump(&m)),
            Err(e) => actual.push(format!("X\t{e}")),
        }
        cases += 1;
    }
    let expected: Vec<&str> = text.lines().collect();
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(a, e, "parse.tsv row {i}");
    }
    assert_eq!(actual.len(), expected.len());
    assert_eq!(cases, 68);
}

fn english() -> Arc<CharArraySet> {
    Arc::new(CharArraySet::from_words(ENGLISH_STOP_WORDS, false))
}

fn build(name: &str, m: &Maps) -> Option<Analyzer> {
    let (se, sc, sn, scs, we, wc) = (
        Arc::clone(&m.solr_expand),
        Arc::clone(&m.solr_collapse),
        Arc::clone(&m.solr_nodedup),
        Arc::clone(&m.solr_cased),
        Arc::clone(&m.wordnet_expand),
        Arc::clone(&m.wordnet_collapse),
    );
    let ws_lower = || LowerCaseFilter::new(WhitespaceTokenizer::new());
    let std_lower = || LowerCaseFilter::new(StandardTokenizer::new());
    Some(match name {
        "ws_lower_graph_expand" => {
            chain(move || comps(SynonymGraphFilter::new(ws_lower(), Arc::clone(&se), true)))
        }
        "ws_lower_graph_collapse" => {
            chain(move || comps(SynonymGraphFilter::new(ws_lower(), Arc::clone(&sc), true)))
        }
        "ws_lower_graph_nodedup" => {
            chain(move || comps(SynonymGraphFilter::new(ws_lower(), Arc::clone(&sn), false)))
        }
        "ws_lower_graph_expand_flatten" => chain(move || {
            comps(FlattenGraphFilter::new(SynonymGraphFilter::new(
                ws_lower(),
                Arc::clone(&se),
                true,
            )))
        }),
        "ws_lower_graph_collapse_flatten" => chain(move || {
            comps(FlattenGraphFilter::new(SynonymGraphFilter::new(
                ws_lower(),
                Arc::clone(&sc),
                false,
            )))
        }),
        "ws_graph_cased" => chain(move || {
            comps(SynonymGraphFilter::new(
                WhitespaceTokenizer::new(),
                Arc::clone(&scs),
                false,
            ))
        }),
        "ws_graph_ignore_case" => chain(move || {
            comps(SynonymGraphFilter::new(
                WhitespaceTokenizer::new(),
                Arc::clone(&se),
                true,
            ))
        }),
        "std_stop_graph_expand" => chain(move || {
            comps(SynonymGraphFilter::new(
                StopFilter::new(std_lower(), english()),
                Arc::clone(&we),
                false,
            ))
        }),
        "std_lower_graph_wordnet" => {
            chain(move || comps(SynonymGraphFilter::new(std_lower(), Arc::clone(&we), false)))
        }
        "std_lower_graph_wordnet_collapse_flatten" => chain(move || {
            comps(FlattenGraphFilter::new(SynonymGraphFilter::new(
                std_lower(),
                Arc::clone(&wc),
                false,
            )))
        }),
        "ws_lower_legacy_expand" => {
            chain(move || comps(SynonymFilter::new(ws_lower(), Arc::clone(&se), true)))
        }
        "ws_lower_legacy_collapse" => {
            chain(move || comps(SynonymFilter::new(ws_lower(), Arc::clone(&sc), false)))
        }
        "ws_legacy_ignore_case" => chain(move || {
            comps(SynonymFilter::new(
                WhitespaceTokenizer::new(),
                Arc::clone(&se),
                true,
            ))
        }),
        "std_stop_legacy_wordnet" => chain(move || {
            comps(SynonymFilter::new(
                StopFilter::new(std_lower(), english()),
                Arc::clone(&we),
                false,
            ))
        }),
        "std_lower_legacy_wordnet_collapse" => {
            chain(move || comps(SynonymFilter::new(std_lower(), Arc::clone(&wc), false)))
        }
        "ws_lower_graph_twice" => chain(move || {
            comps(SynonymGraphFilter::new(
                SynonymGraphFilter::new(ws_lower(), Arc::clone(&sc), false),
                Arc::clone(&we),
                false,
            ))
        }),
        _ => return None,
    })
}

#[test]
fn synonym_chains_match_lucene_token_for_token() {
    let m = maps();
    let mut names = support::fixture_names("analysis_synonym");
    assert!(names.remove("maps") && names.remove("parse"));
    let checked = support::check_chains("analysis_synonym", "analysis-synonym.txt", &names, |n| {
        build(n, &m)
    });
    assert_eq!(checked, 16);
}
