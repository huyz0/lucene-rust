//! `MorfologikAnalyzer` and `UkrainianMorfologikAnalyzer`, with the default
//! dictionaries: Morfologik's Polish dictionary (`morfologik-polish` 2.1.9,
//! BSD-2-Clause, `resources/polish.LICENSE.txt`) and the Ukrainian one
//! (`morfologik-ukrainian-search` 4.9.1, Apache-2.0), vendored
//! zlib-compressed with their `.info` files.

use std::sync::{Arc, LazyLock};

use lucene_analysis::charfilter::{MappingCharFilter, NormalizeCharMap, NormalizeCharMapBuilder};
use lucene_analysis::miscellaneous::SetKeywordMarkerFilter;
use lucene_analysis::reader::CharReader;
use lucene_analysis::token_stream::TokenStream;
use lucene_analysis::{
    AnalysisError, CharArraySet, LowerCaseFilter, StandardTokenizer, StopFilter,
};
use lucene_analysis::{AnalyzerDefinition, TokenStreamComponents};

use crate::dictionary::Dictionary;
use crate::filter::MorfologikFilter;

fn inflate(z: &[u8]) -> Vec<u8> {
    miniz_oxide::inflate::decompress_to_vec_zlib(z).expect("a vendored dictionary inflates")
}

/// `new PolishStemmer().getDictionary()`: Morfologik's Polish dictionary.
pub fn polish_dictionary() -> Arc<Dictionary> {
    static DICT: LazyLock<Arc<Dictionary>> = LazyLock::new(|| {
        let fsa = inflate(include_bytes!("resources/polish.dict.z"));
        Arc::new(
            Dictionary::from_vec(fsa, include_str!("resources/polish.info"))
                .expect("the vendored Polish dictionary reads"),
        )
    });
    Arc::clone(&DICT)
}

/// `UkrainianMorfologikAnalyzer`'s dictionary (`ua/net/nlp/ukrainian.dict`).
pub fn ukrainian_dictionary() -> Arc<Dictionary> {
    static DICT: LazyLock<Arc<Dictionary>> = LazyLock::new(|| {
        let fsa = inflate(include_bytes!("resources/ukrainian.dict.z"));
        Arc::new(
            Dictionary::from_vec(fsa, include_str!("resources/ukrainian.info"))
                .expect("the vendored Ukrainian dictionary reads"),
        )
    });
    Arc::clone(&DICT)
}

/// `org.apache.lucene.analysis.morfologik.MorfologikAnalyzer`:
/// `StandardTokenizer` then `MorfologikFilter`.
#[derive(Debug, Clone)]
pub struct MorfologikAnalyzer {
    dictionary: Arc<Dictionary>,
}

impl Default for MorfologikAnalyzer {
    /// `new MorfologikAnalyzer()`: the Polish dictionary.
    fn default() -> Self {
        Self::new(polish_dictionary())
    }
}

impl MorfologikAnalyzer {
    /// `new MorfologikAnalyzer(dictionary)`.
    pub fn new(dictionary: Arc<Dictionary>) -> Self {
        MorfologikAnalyzer { dictionary }
    }
}

impl AnalyzerDefinition for MorfologikAnalyzer {
    // Java: MorfologikAnalyzer.createComponents
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        Ok(TokenStreamComponents::new(MorfologikFilter::new(
            StandardTokenizer::new(),
            Arc::clone(&self.dictionary),
        )))
    }
}

/// `UkrainianMorfologikAnalyzer`'s stop words (`uk/stopwords.txt`, vendored
/// from the Lucene 10.5.0 jar, Snowball format).
pub fn ukrainian_stop_set() -> Arc<CharArraySet> {
    static SET: LazyLock<Arc<CharArraySet>> = LazyLock::new(|| {
        Arc::new(
            lucene_analysis::wordlist_loader::get_snowball_word_set(
                include_str!("resources/uk_stopwords.txt").as_bytes(),
            )
            .expect("the vendored stop file reads"),
        )
    });
    Arc::clone(&SET)
}

/// `UkrainianMorfologikAnalyzer.NORMALIZER_MAP`.
fn normalizer_map() -> Arc<NormalizeCharMap> {
    static MAP: LazyLock<Arc<NormalizeCharMap>> = LazyLock::new(|| {
        let mut b = NormalizeCharMapBuilder::new();
        for (from, to) in [
            ("\u{2019}", "'"),
            ("\u{2018}", "'"),
            ("\u{02BC}", "'"),
            ("`", "'"),
            ("\u{00B4}", "'"),
            ("\u{0301}", ""),
            ("\u{00AD}", ""),
            ("ґ", "г"),
            ("Ґ", "Г"),
        ] {
            b.add(from, to).expect("a valid mapping");
        }
        Arc::new(b.build())
    });
    Arc::clone(&MAP)
}

/// `CharArraySet.copy(set)`.
fn copy(set: &CharArraySet) -> Arc<CharArraySet> {
    let mut c = CharArraySet::with_capacity(set.len(), set.ignore_case());
    for w in set.iter() {
        c.add(w);
    }
    Arc::new(c)
}

/// `org.apache.lucene.analysis.uk.UkrainianMorfologikAnalyzer`: a mapping
/// char filter (apostrophes, the acute accent, soft hyphen, `ґ` -> `г`),
/// `StandardTokenizer`, `LowerCaseFilter`, `StopFilter`,
/// `SetKeywordMarkerFilter` (with exclusions), `MorfologikFilter` over the
/// Ukrainian dictionary.
#[derive(Debug, Clone)]
pub struct UkrainianMorfologikAnalyzer {
    stopwords: Arc<CharArraySet>,
    exclusion: Arc<CharArraySet>,
    dictionary: Arc<Dictionary>,
}

impl Default for UkrainianMorfologikAnalyzer {
    /// `new UkrainianMorfologikAnalyzer()`.
    fn default() -> Self {
        UkrainianMorfologikAnalyzer {
            stopwords: ukrainian_stop_set(),
            exclusion: Arc::new(CharArraySet::empty()),
            dictionary: ukrainian_dictionary(),
        }
    }
}

impl UkrainianMorfologikAnalyzer {
    /// `new UkrainianMorfologikAnalyzer(stopwords)`.
    pub fn new(stopwords: &CharArraySet) -> Self {
        Self::with_exclusions(stopwords, &CharArraySet::empty())
    }

    /// `new UkrainianMorfologikAnalyzer(stopwords, stemExclusionSet)`.
    pub fn with_exclusions(stopwords: &CharArraySet, exclusions: &CharArraySet) -> Self {
        UkrainianMorfologikAnalyzer {
            stopwords: copy(stopwords),
            exclusion: copy(exclusions),
            dictionary: ukrainian_dictionary(),
        }
    }
}

impl AnalyzerDefinition for UkrainianMorfologikAnalyzer {
    // Java: UkrainianMorfologikAnalyzer.createComponents
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
        Ok(TokenStreamComponents::new(MorfologikFilter::new(
            marked,
            Arc::clone(&self.dictionary),
        )))
    }

    // Java: UkrainianMorfologikAnalyzer.initReader
    fn init_reader(&self, _field: &str, reader: Box<dyn CharReader>) -> Box<dyn CharReader> {
        Box::new(MappingCharFilter::new(normalizer_map(), reader))
    }
}
