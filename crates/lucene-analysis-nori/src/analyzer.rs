//! `org.apache.lucene.analysis.ko.KoreanAnalyzer`.

use std::collections::HashSet;
use std::sync::Arc;

use lucene_analysis::{
    AnalysisError, AnalyzerDefinition, LowerCaseFilter, TokenStream, TokenStreamComponents,
};

use crate::dict::UserDictionary;
use crate::pos::Tag;
use crate::pos_stop::{default_stop_tags, korean_part_of_speech_stop_filter};
use crate::reading_form::KoreanReadingFormFilter;
use crate::tokenizer::KoreanTokenizer;
use crate::viterbi::DecompoundMode;

/// `KoreanAnalyzer`: `KoreanTokenizer` (punctuation discarded),
/// `KoreanPartOfSpeechStopFilter`, `KoreanReadingFormFilter`,
/// `LowerCaseFilter`.
#[derive(Debug, Clone)]
pub struct KoreanAnalyzer {
    user_dict: Option<Arc<UserDictionary>>,
    mode: DecompoundMode,
    stop_tags: Arc<HashSet<Tag>>,
    output_unknown_unigrams: bool,
}

impl Default for KoreanAnalyzer {
    /// `new KoreanAnalyzer()`.
    fn default() -> Self {
        KoreanAnalyzer {
            user_dict: None,
            mode: DecompoundMode::Discard,
            stop_tags: default_stop_tags(),
            output_unknown_unigrams: false,
        }
    }
}

impl KoreanAnalyzer {
    /// `new KoreanAnalyzer(userDict, mode, stopTags, outputUnknownUnigrams)`.
    pub fn new(
        user_dict: Option<Arc<UserDictionary>>,
        mode: DecompoundMode,
        stop_tags: Arc<HashSet<Tag>>,
        output_unknown_unigrams: bool,
    ) -> Self {
        KoreanAnalyzer {
            user_dict,
            mode,
            stop_tags,
            output_unknown_unigrams,
        }
    }
}

impl AnalyzerDefinition for KoreanAnalyzer {
    // Java: KoreanAnalyzer.createComponents
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let tokenizer = KoreanTokenizer::new(
            self.user_dict.clone(),
            self.mode,
            self.output_unknown_unigrams,
            true,
        );
        let stream = korean_part_of_speech_stop_filter(tokenizer, Arc::clone(&self.stop_tags));
        let stream = KoreanReadingFormFilter::new(stream);
        Ok(TokenStreamComponents::new(LowerCaseFilter::new(stream)))
    }

    // Java: KoreanAnalyzer.normalize
    fn normalize(&self, _field: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        Box::new(LowerCaseFilter::new(input))
    }
}
