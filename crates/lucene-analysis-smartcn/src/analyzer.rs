//! `org.apache.lucene.analysis.cn.smart.SmartChineseAnalyzer`:
//! `HMMChineseTokenizer`, `PorterStemFilter`, and a `StopFilter` when the
//! stop set is not empty; `normalize` is `LowerCaseFilter`.

use std::sync::{Arc, LazyLock};

use lucene_analysis::en::PorterStemFilter;
use lucene_analysis::{AnalysisError, CharArraySet, LowerCaseFilter, StopFilter, TokenStream};
use lucene_analysis::{AnalyzerDefinition, TokenStreamComponents};

use crate::tokenizer::hmm_chinese_tokenizer;

/// `SmartChineseAnalyzer.getDefaultStopSet()`: `stopwords.txt`, `//`
/// comments.
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> = LazyLock::new(|| {
    Arc::new(
        lucene_analysis::wordlist_loader::get_word_set_with_comment(
            include_bytes!("resources/stopwords.txt").as_slice(),
            "//",
        )
        .expect("the vendored stop file reads"),
    )
});

/// `SmartChineseAnalyzer`.
#[derive(Debug, Clone)]
pub struct SmartChineseAnalyzer {
    stop_words: Arc<CharArraySet>,
}

impl Default for SmartChineseAnalyzer {
    /// `new SmartChineseAnalyzer()`: the default stop set.
    fn default() -> Self {
        Self::with_default_stop_words(true)
    }
}

impl SmartChineseAnalyzer {
    /// `new SmartChineseAnalyzer(boolean useDefaultStopWords)`.
    pub fn with_default_stop_words(use_default: bool) -> Self {
        SmartChineseAnalyzer {
            stop_words: if use_default {
                Arc::clone(&DEFAULT_STOP_SET)
            } else {
                Arc::new(CharArraySet::empty())
            },
        }
    }

    /// `new SmartChineseAnalyzer(CharArraySet stopWords)` (`None`: Java's
    /// `null`, no stop words).
    pub fn new(stop_words: Option<Arc<CharArraySet>>) -> Self {
        SmartChineseAnalyzer {
            stop_words: stop_words.unwrap_or_else(|| Arc::new(CharArraySet::empty())),
        }
    }
}

impl AnalyzerDefinition for SmartChineseAnalyzer {
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let result = PorterStemFilter::new(hmm_chinese_tokenizer());
        Ok(if self.stop_words.is_empty() {
            TokenStreamComponents::new(result)
        } else {
            TokenStreamComponents::new(StopFilter::new(result, Arc::clone(&self.stop_words)))
        })
    }

    fn normalize(&self, _field: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        Box::new(LowerCaseFilter::new(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::Analyzer;

    fn terms(a: &Analyzer, text: &str) -> Vec<String> {
        let mut ts = a.token_stream("f", text).unwrap();
        let mut out = Vec::new();
        lucene_analysis::token_stream::consume(&mut ts, |a| out.push(a.term().to_string()))
            .unwrap();
        out
    }

    #[test]
    fn stems_and_stops() {
        let a = Analyzer::new(SmartChineseAnalyzer::default());
        assert_eq!(
            terms(&a, "我是中国人。running"),
            ["我", "是", "中国", "人", "run"]
        );
        let a = Analyzer::new(SmartChineseAnalyzer::with_default_stop_words(false));
        assert_eq!(terms(&a, "中国。"), ["中国", ","]);
        let a = Analyzer::new(SmartChineseAnalyzer::new(None));
        assert_eq!(terms(&a, "中国。"), ["中国", ","]);
        assert_eq!(a.normalize("f", "ABC").unwrap(), b"abc");
        assert!(DEFAULT_STOP_SET.contains("。") || DEFAULT_STOP_SET.contains(","));
    }
}
