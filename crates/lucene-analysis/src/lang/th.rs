//! `org.apache.lucene.analysis.th`: `ThaiTokenizer` and `ThaiAnalyzer` --
//! **not supported by design**: they refuse with the
//! `UnsupportedOperationException` Lucene raises on a JRE without Thai
//! dictionary segmentation.
//!
//! `ThaiTokenizer` splits sentences with the JDK's sentence iterator
//! ([`crate::util::sentence_break`], ported) and words with
//! `BreakIterator.getWordInstance(th)`, the JDK's dictionary-based iterator.
//! Its output is a function of the JDK's Thai word list (`thai_dict`, in
//! `java.base`), which is GPL-2.0-with-Classpath-Exception data: it cannot be
//! vendored into this Apache-2.0 library, nor re-derived from the JDK's
//! answers without copying it, and no other dictionary (ICU's, LibThai's)
//! splits Thai the same way. So the port does what Lucene does on a JRE whose
//! word iterator does not segment Thai: [`DBBI_AVAILABLE`] is `false`, and
//! constructing the tokenizer -- directly, through [`ThaiAnalyzer`] or the
//! `thai` factory -- is an [`AnalysisError::UnsupportedOperation`] with
//! Java's message. `ThaiAnalyzer`'s stop set and `normalize` chain need no
//! dictionary and work. Thai text can be segmented with ICU's tokenizer
//! (`analysis-icu`, M12 T12.4) instead, as Lucene's own documentation
//! suggests.

use std::sync::{Arc, LazyLock};

use crate::core_analysis::DecimalDigitFilter;
use crate::{AnalysisError, CharArraySet, LowerCaseFilter};

use super::comment_set;

/// `ThaiTokenizer.DBBI_AVAILABLE`: whether the JRE's word iterator
/// segments Thai with a dictionary -- never, here (see the module docs).
pub const DBBI_AVAILABLE: bool = false;

/// The message of `ThaiTokenizer`'s `UnsupportedOperationException`.
pub const UNSUPPORTED: &str = "This JRE does not have support for Thai segmentation";

/// `ThaiAnalyzer.getDefaultStopSet()` (`stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/th_stopwords.txt")));

/// `ThaiTokenizer`: never constructed (see the module docs).
#[derive(Debug)]
pub struct ThaiTokenizer {
    _never: (),
}

impl ThaiTokenizer {
    /// `new ThaiTokenizer()`: Java's `UnsupportedOperationException` for a
    /// JRE without Thai segmentation.
    pub fn new() -> Result<Self, AnalysisError> {
        if DBBI_AVAILABLE {
            unreachable!("no Thai dictionary is shipped");
        }
        Err(AnalysisError::UnsupportedOperation(UNSUPPORTED.to_string()))
    }
}

/// `ThaiAnalyzer`: `ThaiTokenizer`, `LowerCaseFilter`,
/// `DecimalDigitFilter`, `StopFilter`. Its token streams refuse (see the
/// module docs); `normalize` works.
#[derive(Debug, Clone)]
pub struct ThaiAnalyzer {
    stopwords: Arc<CharArraySet>,
}

impl Default for ThaiAnalyzer {
    /// `new ThaiAnalyzer()`: the default stop set.
    fn default() -> Self {
        Self::new(&DEFAULT_STOP_SET)
    }
}

impl ThaiAnalyzer {
    /// `new ThaiAnalyzer(CharArraySet stopwords)`.
    pub fn new(stopwords: &CharArraySet) -> Self {
        ThaiAnalyzer {
            stopwords: super::copy_set(stopwords),
        }
    }

    /// `getStopwordSet()`.
    pub fn stopwords(&self) -> &CharArraySet {
        &self.stopwords
    }
}

impl crate::analyzer::AnalyzerDefinition for ThaiAnalyzer {
    fn create_components(
        &self,
        _field: &str,
    ) -> Result<crate::analyzer::TokenStreamComponents, AnalysisError> {
        let _ = ThaiTokenizer::new()?;
        unreachable!("ThaiTokenizer::new always refuses")
    }

    fn normalize(
        &self,
        _field: &str,
        input: Box<dyn crate::token_stream::TokenStream>,
    ) -> Box<dyn crate::token_stream::TokenStream> {
        Box::new(DecimalDigitFilter::new(LowerCaseFilter::new(input)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Analyzer;

    #[test]
    fn the_tokenizer_refuses_as_on_a_jre_without_thai_segmentation() {
        let e = ThaiTokenizer::new().unwrap_err();
        assert_eq!(e, AnalysisError::UnsupportedOperation(UNSUPPORTED.into()));
        assert_eq!(
            e.to_string(),
            "unsupported operation: This JRE does not have support for Thai segmentation"
        );
        let a = Analyzer::new(ThaiAnalyzer::default());
        let e = a.token_stream("f", "ภาษาไทย").err().unwrap();
        assert!(matches!(e, AnalysisError::UnsupportedOperation(_)), "{e}");
    }

    #[test]
    fn stop_set_and_normalize_work() {
        let a = ThaiAnalyzer::default();
        assert!(a.stopwords().contains("และ"));
        assert_eq!(a.stopwords().len(), DEFAULT_STOP_SET.len());
        let a = Analyzer::new(ThaiAnalyzer::new(&CharArraySet::empty()));
        assert_eq!(a.normalize("f", "ABC ๑๒๓").unwrap(), b"abc 123");
    }
}
