//! `org.apache.lucene.analysis.ja.JapaneseAnalyzer` and
//! `JapaneseCompletionAnalyzer`.

use std::sync::{Arc, LazyLock};

use lucene_analysis::cjk::CJKWidthCharFilter;
use lucene_analysis::reader::CharReader;
use lucene_analysis::wordlist_loader;
use lucene_analysis::{
    AnalysisError, AnalyzerDefinition, CharArraySet, LowerCaseFilter, StopFilter, TokenStream,
    TokenStreamComponents,
};

use crate::base_form::JapaneseBaseFormFilter;
use crate::completion::{CompletionMode, JapaneseCompletionFilter};
use crate::dict::UserDictionary;
use crate::katakana_stem::{JapaneseKatakanaStemFilter, DEFAULT_MINIMUM_LENGTH};
use crate::pos_stop::japanese_part_of_speech_stop_filter;
use crate::tokenizer::{JapaneseTokenizer, Mode};

/// `JapaneseAnalyzer.getDefaultStopSet()`: `stopwords.txt`, ignoring case.
pub fn default_stop_set() -> Arc<CharArraySet> {
    static SET: LazyLock<Arc<CharArraySet>> = LazyLock::new(|| {
        let mut set = CharArraySet::with_capacity(16, true);
        wordlist_loader::get_word_set_with_comment_into(
            include_str!("resources/stopwords.txt").as_bytes(),
            "#",
            &mut set,
        )
        .expect("the vendored stop words read");
        Arc::new(set)
    });
    Arc::clone(&SET)
}

/// `JapaneseAnalyzer.getDefaultStopTags()`: `stoptags.txt`.
pub fn default_stop_tags() -> Arc<CharArraySet> {
    static TAGS: LazyLock<Arc<CharArraySet>> = LazyLock::new(|| {
        let set = wordlist_loader::get_word_set_with_comment(
            include_str!("resources/stoptags.txt").as_bytes(),
            "#",
        )
        .expect("the vendored stop tags read");
        Arc::new(set)
    });
    Arc::clone(&TAGS)
}

/// `JapaneseAnalyzer`: `JapaneseTokenizer` (punctuation and compounds
/// discarded), `JapaneseBaseFormFilter`, `JapanesePartOfSpeechStopFilter`,
/// `StopFilter`, `JapaneseKatakanaStemFilter`, `LowerCaseFilter`, over a
/// `CJKWidthCharFilter`.
#[derive(Debug, Clone)]
pub struct JapaneseAnalyzer {
    user_dict: Option<Arc<UserDictionary>>,
    mode: Mode,
    stopwords: Arc<CharArraySet>,
    stoptags: Arc<CharArraySet>,
}

impl Default for JapaneseAnalyzer {
    /// `new JapaneseAnalyzer()`.
    fn default() -> Self {
        JapaneseAnalyzer {
            user_dict: None,
            mode: Mode::default(),
            stopwords: default_stop_set(),
            stoptags: default_stop_tags(),
        }
    }
}

impl JapaneseAnalyzer {
    /// `new JapaneseAnalyzer(userDict, mode, stopwords, stoptags)`.
    pub fn new(
        user_dict: Option<Arc<UserDictionary>>,
        mode: Mode,
        stopwords: Arc<CharArraySet>,
        stoptags: Arc<CharArraySet>,
    ) -> Self {
        JapaneseAnalyzer {
            user_dict,
            mode,
            stopwords,
            stoptags,
        }
    }
}

impl AnalyzerDefinition for JapaneseAnalyzer {
    // Java: JapaneseAnalyzer.createComponents
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let tokenizer =
            JapaneseTokenizer::with_options(self.user_dict.clone(), true, true, self.mode);
        let stream = JapaneseBaseFormFilter::new(tokenizer);
        let stream = japanese_part_of_speech_stop_filter(stream, Arc::clone(&self.stoptags));
        let stream = StopFilter::new(stream, Arc::clone(&self.stopwords));
        let stream = JapaneseKatakanaStemFilter::new(stream, DEFAULT_MINIMUM_LENGTH)?;
        Ok(TokenStreamComponents::new(LowerCaseFilter::new(stream)))
    }

    // Java: JapaneseAnalyzer.normalize
    fn normalize(&self, _field: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        Box::new(LowerCaseFilter::new(input))
    }

    // Java: JapaneseAnalyzer.initReader
    fn init_reader(&self, _field: &str, reader: Box<dyn CharReader>) -> Box<dyn CharReader> {
        Box::new(CJKWidthCharFilter::new(reader))
    }

    // Java: JapaneseAnalyzer.initReaderForNormalization
    fn init_reader_for_normalization(
        &self,
        _field: &str,
        reader: Box<dyn CharReader>,
    ) -> Box<dyn CharReader> {
        Box::new(CJKWidthCharFilter::new(reader))
    }
}

/// `JapaneseCompletionAnalyzer`: `JapaneseTokenizer` in normal mode,
/// `JapaneseCompletionFilter`, `LowerCaseFilter`, over a
/// `CJKWidthCharFilter`.
#[derive(Debug, Clone, Default)]
pub struct JapaneseCompletionAnalyzer {
    user_dict: Option<Arc<UserDictionary>>,
    mode: CompletionMode,
}

impl JapaneseCompletionAnalyzer {
    /// `new JapaneseCompletionAnalyzer(userDict, mode)`.
    pub fn new(user_dict: Option<Arc<UserDictionary>>, mode: CompletionMode) -> Self {
        JapaneseCompletionAnalyzer { user_dict, mode }
    }
}

impl AnalyzerDefinition for JapaneseCompletionAnalyzer {
    // Java: JapaneseCompletionAnalyzer.createComponents
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let tokenizer =
            JapaneseTokenizer::with_options(self.user_dict.clone(), true, true, Mode::Normal);
        let stream = JapaneseCompletionFilter::new(tokenizer, self.mode);
        Ok(TokenStreamComponents::new(LowerCaseFilter::new(stream)))
    }

    fn init_reader(&self, _field: &str, reader: Box<dyn CharReader>) -> Box<dyn CharReader> {
        Box::new(CJKWidthCharFilter::new(reader))
    }

    fn init_reader_for_normalization(
        &self,
        _field: &str,
        reader: Box<dyn CharReader>,
    ) -> Box<dyn CharReader> {
        Box::new(CJKWidthCharFilter::new(reader))
    }
}
