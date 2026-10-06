//! M11 T11.2: `ClassicTokenizer`, `ClassicFilter`, `ClassicAnalyzer` and
//! `WikipediaTokenizer` against `fixtures/src/GenAnalysisClassic.java`: the
//! chains over `fixtures/corpus/analysis-classic.txt`, and four of them over
//! 2,000 seeded fragment joins (`fragments.txt`), row for row.

mod support;

use std::collections::{BTreeSet, HashSet};

use lucene_analysis::classic::{ClassicAnalyzer, ClassicFilter, ClassicTokenizer};
use lucene_analysis::wikipedia::{WikipediaTokenizer, BOTH, UNTOKENIZED_ONLY};
use lucene_analysis::Analyzer;
use support::{chain, comps};

fn types(t: &[&str]) -> HashSet<String> {
    t.iter().map(|s| s.to_string()).collect()
}

fn build(name: &str) -> Option<Analyzer> {
    let some = || types(&["il", "c", "b", "h"]);
    let all = || types(&["il", "el", "elu", "ci", "c", "b", "i", "bi", "h", "sh"]);
    Some(match name.strip_prefix("frag_").unwrap_or(name) {
        "classic_tokenizer" => chain(|| comps(ClassicTokenizer::new())),
        "classic_tokenizer_max5" => chain(|| {
            let mut t = ClassicTokenizer::new();
            t.set_max_token_length(5)?;
            comps(t)
        }),
        "classic_filter" => chain(|| comps(ClassicFilter::new(ClassicTokenizer::new()))),
        "classic_analyzer" => Analyzer::new(ClassicAnalyzer::default()),
        "classic_analyzer_max4" => {
            Analyzer::new(ClassicAnalyzer::default().with_max_token_length(4))
        }
        "wikipedia_tokens_only" => chain(|| comps(WikipediaTokenizer::default())),
        "wikipedia_untokenized_some" => {
            chain(move || comps(WikipediaTokenizer::new(UNTOKENIZED_ONLY, some())?))
        }
        "wikipedia_both_some" => chain(move || comps(WikipediaTokenizer::new(BOTH, some())?)),
        "wikipedia_both_all" => chain(move || comps(WikipediaTokenizer::new(BOTH, all())?)),
        "wikipedia_untokenized_all" => {
            chain(move || comps(WikipediaTokenizer::new(UNTOKENIZED_ONLY, all())?))
        }
        _ => return None,
    })
}

#[test]
fn classic_and_wikipedia_chains_match_lucene() {
    let names = support::fixture_names("analysis_classic");
    let (frag, plain): (BTreeSet<String>, BTreeSet<String>) =
        names.into_iter().partition(|n| n.starts_with("frag_"));
    assert_eq!(
        support::check_chains("analysis_classic", "analysis-classic.txt", &plain, build),
        10
    );
    let lines: Vec<String> =
        std::fs::read_to_string(support::data_dir("analysis_classic") + "fragments.txt")
            .unwrap()
            .lines()
            .map(support::unesc)
            .collect();
    assert_eq!(lines.len(), 2000);
    assert_eq!(
        support::check_chains_over("analysis_classic", &lines, &frag, build),
        4
    );
}
