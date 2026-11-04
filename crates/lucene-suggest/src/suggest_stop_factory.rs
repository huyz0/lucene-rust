//! `org.apache.lucene.search.suggest.analyzing.SuggestStopFilterFactory`
//! (`suggestStop`) and [`register_factories`].

use std::sync::Arc;

use lucene_analysis::factory::args::{self, JavaArgs};
use lucene_analysis::factory::spi;
use lucene_analysis::factory::{
    get_snowball_word_set, get_word_set, AnalysisFactory, FactoryBase, FactoryClass, FactoryError,
    JavaException, ResourceLoader, TokenFilterFactory,
};
use lucene_analysis::{AnalysisError, CharArraySet, TokenStream, ENGLISH_STOP_WORDS};

use super::analyzing::SuggestStopFilter;

/// `SuggestStopFilterFactory`: `words` (comma-separated files), `format`
/// (`wordset` or `snowball`), `ignoreCase`; without `words`, the English stop
/// words.
pub struct SuggestStopFilterFactory {
    base: FactoryBase,
    stop_words: Option<Arc<CharArraySet>>,
    stop_word_files: Option<String>,
    format: Option<String>,
    ignore_case: bool,
}

impl SuggestStopFilterFactory {
    /// `FORMAT_WORDSET`.
    pub const FORMAT_WORDSET: &'static str = "wordset";
    /// `FORMAT_SNOWBALL`.
    pub const FORMAT_SNOWBALL: &'static str = "snowball";

    /// `isIgnoreCase()`.
    pub fn is_ignore_case(&self) -> bool {
        self.ignore_case
    }

    /// `getStopWords()`: `None` until the factory is informed.
    pub fn stop_words(&self) -> Option<&Arc<CharArraySet>> {
        self.stop_words.as_ref()
    }
}

impl FactoryClass for SuggestStopFilterFactory {
    const NAME: &'static str = "suggestStop";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.search.suggest.analyzing.SuggestStopFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let stop_word_files = args::get(args, "words");
        let format = match args::get(args, "format") {
            Some(f) => Some(f),
            None => stop_word_files
                .as_ref()
                .map(|_| Self::FORMAT_WORDSET.to_string()),
        };
        let ignore_case = args::get_boolean(args, "ignoreCase", false);
        args::reject_unknown(args)?;
        Ok(SuggestStopFilterFactory {
            base,
            stop_words: None,
            stop_word_files,
            format,
            ignore_case,
        })
    }
}

impl AnalysisFactory for SuggestStopFilterFactory {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
    fn is_resource_loader_aware(&self) -> bool {
        true
    }
    // Java: SuggestStopFilterFactory.inform
    fn inform(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        let words = match &self.stop_word_files {
            Some(files) => {
                let format = self.format.as_deref().unwrap_or("");
                if args::java_equals_ignore_case(format, Self::FORMAT_WORDSET) {
                    get_word_set(loader, files, self.ignore_case)?
                } else if args::java_equals_ignore_case(format, Self::FORMAT_SNOWBALL) {
                    get_snowball_word_set(loader, files, self.ignore_case)?
                } else {
                    return Err(FactoryError::illegal_argument(format!(
                        "Unknown 'format' specified for 'words' file: {}",
                        self.format.as_deref().unwrap_or("null")
                    )));
                }
            }
            None => {
                if let Some(format) = &self.format {
                    return Err(FactoryError::illegal_argument(format!(
                        "'format' can not be specified w/o an explicit 'words' file: {format}"
                    )));
                }
                Some(CharArraySet::from_words(
                    ENGLISH_STOP_WORDS,
                    self.ignore_case,
                ))
            }
        };
        self.stop_words = words.map(Arc::new);
        Ok(())
    }
}

impl TokenFilterFactory for SuggestStopFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        // Java's `new SuggestStopFilter(input, null)` fails on first use
        let stop_words = self.stop_words.clone().ok_or_else(|| {
            AnalysisError::from(FactoryError::new(
                JavaException::NullPointer,
                "SuggestStopFilterFactory was not informed of its stop words.",
            ))
        })?;
        Ok(Box::new(SuggestStopFilter::new(input, stop_words)))
    }
}

/// Registers [`SuggestStopFilterFactory`] with `lucene-analysis`' SPI
/// registry (idempotent).
pub fn register_factories() -> Result<(), FactoryError> {
    spi::register_token_filter(spi::token_filter_entry::<SuggestStopFilterFactory>())
}
