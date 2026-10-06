//! `org.apache.lucene.analysis.ne`: `NepaliAnalyzer` (Indic normalization, Snowball
//! `NepaliStemmer`).

use std::sync::{Arc, LazyLock};

use crate::core_analysis::DecimalDigitFilter;
use crate::{CharArraySet, LowerCaseFilter, StandardTokenizer, StopFilter};

use super::in_::IndicNormalizationFilter;
use super::{comment_set, mark_exclusions, snowball};

/// `NepaliAnalyzer.getDefaultStopSet()` (`stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/ne_stopwords.txt")));

language_analyzer! {
    /// `NepaliAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`,
    /// `DecimalDigitFilter`, exclusions, `IndicNormalizationFilter`,
    /// `StopFilter`, Snowball `NepaliStemmer`.
    NepaliAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        let r = DecimalDigitFilter::new(LowerCaseFilter::new(StandardTokenizer::new()));
        let r = IndicNormalizationFilter::new(mark_exclusions(r, &s.exclusion));
        snowball(StopFilter::new(r, Arc::clone(&s.stopwords)), "Nepali")
    }
    normalize(s, input) {
        IndicNormalizationFilter::new(DecimalDigitFilter::new(LowerCaseFilter::new(input)))
    }
}
