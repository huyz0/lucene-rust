//! `org.apache.lucene.analysis.nl`: `DutchAnalyzer` (stem overrides, Snowball
//! `DutchStemmer`).

use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock};

use crate::analyzer::{AnalyzerDefinition, TokenStreamComponents};
use crate::miscellaneous::{StemmerOverrideBuilder, StemmerOverrideFilter, StemmerOverrideMap};
use crate::token_stream::TokenStream;
use crate::{AnalysisError, CharArraySet, LowerCaseFilter};

use super::{copy_set, mark_exclusions, snowball, snowball_set, std_lower_stop};

/// `DutchAnalyzer.getDefaultStopSet()` (`snowball/dutch_stop.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| snowball_set(include_str!("stopwords/dutch_stop.txt")));

/// `DutchAnalyzer`'s `DEFAULT_STEM_DICT`.
pub fn default_stem_dict() -> BTreeMap<String, String> {
    [
        ("fiets", "fiets"),
        ("bromfiets", "bromfiets"),
        ("ei", "eier"),
        ("kind", "kinder"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// `DutchAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`, `StopFilter`,
/// exclusions, `StemmerOverrideFilter` (when the dictionary is not empty),
/// Snowball `DutchStemmer`.
#[derive(Debug, Clone)]
pub struct DutchAnalyzer {
    stopwords: Arc<CharArraySet>,
    exclusion: Arc<CharArraySet>,
    stemdict: Option<Arc<StemmerOverrideMap>>,
}

impl Default for DutchAnalyzer {
    fn default() -> Self {
        Self::new(&DEFAULT_STOP_SET)
    }
}

impl DutchAnalyzer {
    /// `new DutchAnalyzer(stopwords)`.
    pub fn new(stopwords: &CharArraySet) -> Self {
        Self::with_options(stopwords, &CharArraySet::empty(), &default_stem_dict())
    }

    /// `new DutchAnalyzer(stopwords, stemExclusionTable, stemOverrideDict)`.
    pub fn with_options(
        stopwords: &CharArraySet,
        exclusions: &CharArraySet,
        stem_override: &BTreeMap<String, String>,
    ) -> Self {
        let stemdict = (!stem_override.is_empty()).then(|| {
            let mut b = StemmerOverrideBuilder::new(false);
            for (k, v) in stem_override {
                b.add(k, v);
            }
            b.build()
        });
        DutchAnalyzer {
            stopwords: copy_set(stopwords),
            exclusion: copy_set(exclusions),
            stemdict,
        }
    }
}

impl AnalyzerDefinition for DutchAnalyzer {
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let r = mark_exclusions(std_lower_stop(&self.stopwords), &self.exclusion);
        let r: Box<dyn TokenStream> = match &self.stemdict {
            Some(d) => Box::new(StemmerOverrideFilter::new(r, Arc::clone(d))),
            None => r,
        };
        Ok(TokenStreamComponents::new(snowball(r, "Dutch")))
    }

    fn normalize(&self, _field: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        Box::new(LowerCaseFilter::new(input))
    }
}
