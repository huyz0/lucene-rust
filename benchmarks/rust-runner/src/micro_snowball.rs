//! M11 T11.3's Snowball pair (`benchmarks/micro/java/SnowballMicro.java` is
//! the Java twin): `WhitespaceTokenizer` + `SnowballFilter` per language over
//! that language's fixture vocabulary (`fixtures/data/snowball/<Language>.words`,
//! the words only, 100 to a document).

use std::hint::black_box;
use std::time::Duration;

use lucene_analysis::snowball::SnowballFilter;
use lucene_analysis::util::WhitespaceTokenizer;
use lucene_analysis::{AnalysisError, Analyzer, AnalyzerDefinition, TokenStreamComponents};

use super::{consume_stream, measure};

/// `SnowballMicro.LANGUAGES`.
const LANGUAGES: [&str; 6] = ["English", "German", "French", "Russian", "Arabic", "Turkish"];

struct Chain(&'static str);

impl AnalyzerDefinition for Chain {
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        Ok(TokenStreamComponents::new(SnowballFilter::with_name(
            WhitespaceTokenizer::new(),
            self.0,
        )?))
    }
}

/// `SnowballMicro.docs`.
fn docs(lang: &str) -> Vec<String> {
    let text = std::fs::read_to_string(format!("fixtures/data/snowball/{lang}.words"))
        .expect("run from the repository root");
    let words: Vec<&str> = text
        .lines()
        .map(|l| l.split('\t').next().unwrap())
        .filter(|w| !w.is_empty() && !w.chars().any(char::is_whitespace))
        .collect();
    words.chunks(100).map(|c| c.join(" ")).collect()
}

pub(super) fn bench_snowball(w: Duration, mt: Duration) {
    for lang in LANGUAGES {
        let docs = docs(lang);
        let a = Analyzer::new(Chain(lang));
        measure(&format!("snowball_{}", lang.to_lowercase()), w, mt, || {
            let mut tokens = 0u64;
            for text in &docs {
                let mut ts = a.token_stream("body", black_box(text)).unwrap();
                tokens += consume_stream(&mut ts);
            }
            tokens
        });
    }
}
