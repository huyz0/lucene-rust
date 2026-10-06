//! M11 T11.4's Hunspell pair (`benchmarks/micro/java/HunspellMicro.java` is
//! the Java twin): `WhitespaceTokenizer` + `HunspellStemFilter` (dedup on) per
//! dictionary of `fixtures/corpus/hunspell` over the words of its fixture
//! (`fixtures/data/hunspell/<name>.tsv`, `W` lines, 100 to a document).

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use lucene_analysis::hunspell::{Dictionary, HunspellStemFilter};
use lucene_analysis::util::WhitespaceTokenizer;
use lucene_analysis::{AnalysisError, Analyzer, AnalyzerDefinition, TokenStreamComponents};

use super::{consume_stream, measure};

/// `HunspellMicro.DICTIONARIES`.
const DICTIONARIES: [&str; 3] = ["affixes", "compound", "features"];

struct Chain(Arc<Dictionary>);

impl AnalyzerDefinition for Chain {
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        Ok(TokenStreamComponents::new(HunspellStemFilter::new(
            WhitespaceTokenizer::new(),
            Arc::clone(&self.0),
            true,
            false,
        )))
    }
}

/// `HunspellMicro.docs`.
fn docs(name: &str) -> Vec<String> {
    let text = std::fs::read_to_string(format!("fixtures/data/hunspell/{name}.tsv"))
        .expect("run from the repository root");
    let words: Vec<&str> = text
        .lines()
        .filter_map(|l| {
            let mut f = l.split('\t');
            (f.next() == Some("W")).then(|| f.next().unwrap())
        })
        .filter(|w| !w.chars().any(char::is_whitespace))
        .collect();
    words.chunks(100).map(|c| c.join(" ")).collect()
}

pub(super) fn bench_hunspell(w: Duration, mt: Duration) {
    for name in DICTIONARIES {
        let dir = "fixtures/corpus/hunspell/";
        let aff = std::fs::read(format!("{dir}{name}.aff")).expect("run from the repository root");
        let dic = std::fs::read(format!("{dir}{name}.dic")).expect("run from the repository root");
        let d = Arc::new(Dictionary::new(&aff, &[&dic], false).expect("a valid dictionary"));
        let docs = docs(name);
        let a = Analyzer::new(Chain(d));
        measure(&format!("hunspell_{name}"), w, mt, || {
            let mut tokens = 0u64;
            for text in &docs {
                let mut ts = a.token_stream("body", black_box(text)).unwrap();
                tokens += consume_stream(&mut ts);
            }
            tokens
        });
    }
}
