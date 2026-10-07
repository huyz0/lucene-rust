//! `org.apache.lucene.analysis.pl.PolishAnalyzer`.

use std::sync::{Arc, LazyLock};

use lucene_analysis::miscellaneous::SetKeywordMarkerFilter;
use lucene_analysis::token_stream::TokenStream;
use lucene_analysis::{
    AnalysisError, CharArraySet, LowerCaseFilter, StandardTokenizer, StopFilter,
};
use lucene_analysis::{AnalyzerDefinition, TokenStreamComponents};

use crate::egothor::Table;
use crate::filter::StempelFilter;
use crate::stemmer::{default_table, StempelStemmer};

/// `PolishAnalyzer.DEFAULT_STOPWORD_FILE` (`stopwords.txt`, Carrot2's
/// list, BSD; vendored from the 10.5.0 jar).
const STOPWORDS: &str = include_str!("resources/stopwords.txt");

/// `PolishAnalyzer.getDefaultStopSet()`: `WordlistLoader.getWordSet(..., "#")`.
pub fn default_stop_set() -> Arc<CharArraySet> {
    static SET: LazyLock<Arc<CharArraySet>> = LazyLock::new(|| {
        Arc::new(
            lucene_analysis::wordlist_loader::get_word_set_with_comment(STOPWORDS.as_bytes(), "#")
                .expect("the vendored stop file reads"),
        )
    });
    Arc::clone(&SET)
}

/// `CharArraySet.copy(set)`.
fn copy(set: &CharArraySet) -> Arc<CharArraySet> {
    let mut c = CharArraySet::with_capacity(set.len(), set.ignore_case());
    for w in set.iter() {
        c.add(w);
    }
    Arc::new(c)
}

/// `org.apache.lucene.analysis.pl.PolishAnalyzer`: `StandardTokenizer`,
/// `LowerCaseFilter`, `StopFilter`, `SetKeywordMarkerFilter` (when there are
/// exclusions), `StempelFilter` over the default table.
#[derive(Debug, Clone)]
pub struct PolishAnalyzer {
    stopwords: Arc<CharArraySet>,
    exclusion: Arc<CharArraySet>,
    table: Arc<Table>,
}

impl Default for PolishAnalyzer {
    /// `new PolishAnalyzer()`.
    fn default() -> Self {
        PolishAnalyzer {
            stopwords: default_stop_set(),
            exclusion: Arc::new(CharArraySet::empty()),
            table: default_table(),
        }
    }
}

impl PolishAnalyzer {
    /// `new PolishAnalyzer(CharArraySet stopwords)`.
    pub fn new(stopwords: &CharArraySet) -> Self {
        Self::with_exclusions(stopwords, &CharArraySet::empty())
    }

    /// `new PolishAnalyzer(stopwords, stemExclusionSet)`.
    pub fn with_exclusions(stopwords: &CharArraySet, exclusions: &CharArraySet) -> Self {
        PolishAnalyzer {
            stopwords: copy(stopwords),
            exclusion: copy(exclusions),
            table: default_table(),
        }
    }
}

impl AnalyzerDefinition for PolishAnalyzer {
    // Java: PolishAnalyzer.createComponents
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let stop = StopFilter::new(
            LowerCaseFilter::new(StandardTokenizer::new()),
            Arc::clone(&self.stopwords),
        );
        let marked: Box<dyn TokenStream> = if self.exclusion.is_empty() {
            Box::new(stop)
        } else {
            Box::new(SetKeywordMarkerFilter::new(
                stop,
                Arc::clone(&self.exclusion),
            ))
        };
        let stemmer = StempelStemmer::new(Arc::clone(&self.table));
        Ok(TokenStreamComponents::new(StempelFilter::new(
            marked, stemmer,
        )))
    }

    // Java: PolishAnalyzer.normalize
    fn normalize(&self, _field: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        Box::new(LowerCaseFilter::new(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::Analyzer;

    fn terms(a: PolishAnalyzer, text: &str) -> Vec<String> {
        let analyzer = Analyzer::new(a);
        let mut ts = analyzer.token_stream("f", text).unwrap();
        ts.reset().unwrap();
        let mut out = Vec::new();
        while ts.increment_token().unwrap() {
            out.push(ts.attributes().term().to_string());
        }
        ts.end().unwrap();
        out
    }

    #[test]
    fn analyzes() {
        assert!(default_stop_set().contains("się"));
        let t = terms(PolishAnalyzer::default(), "Kot i pies mieszkają");
        assert_eq!(t, ["kot", "pies", "mieszkać"]);
        let mut ex = CharArraySet::with_capacity(1, false);
        ex.add("mieszkają");
        let t = terms(
            PolishAnalyzer::with_exclusions(&CharArraySet::empty(), &ex),
            "Mieszkają i",
        );
        assert_eq!(t, ["mieszkają", "i"]);
        let t = terms(PolishAnalyzer::new(&CharArraySet::empty()), "I");
        assert_eq!(t, ["i"]);
        let n = Analyzer::new(PolishAnalyzer::default())
            .normalize("f", "KOTAMI")
            .unwrap();
        assert_eq!(n, b"kotami");
    }
}
