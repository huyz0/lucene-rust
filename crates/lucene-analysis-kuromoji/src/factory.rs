//! The analysis-kuromoji factories (`japanese`, `japaneseBaseForm`,
//! `japaneseCompletion`, `japaneseHiraganaUppercase`,
//! `japaneseIterationMark`, `japaneseKatakanaStem`,
//! `japaneseKatakanaUppercase`, `japaneseNumber`,
//! `japanesePartOfSpeechStop`, `japaneseReadingForm`) and
//! [`register_factories`].
//!
//! Differs: `userDictionaryEncoding` other than UTF-8 is refused with
//! `UnsupportedEncodingException` (Java decodes any JDK charset).

use std::collections::HashSet;
use std::sync::Arc;

use lucene_analysis::factory::args::{self, JavaArgs};
use lucene_analysis::factory::{
    decode_utf8, get_word_set, spi, AnalysisFactory, CharFilterFactory, FactoryBase, FactoryClass,
    FactoryError, JavaException, ResourceLoader, TokenFilterFactory, TokenizerFactory,
};
use lucene_analysis::lang::java_string_to_upper_case;
use lucene_analysis::reader::CharReader;
use lucene_analysis::{AnalysisError, TokenStream};

use crate::analyzer::default_stop_tags;
use crate::base_form::JapaneseBaseFormFilter;
use crate::completion::{CompletionMode, JapaneseCompletionFilter};
use crate::dict::UserDictionary;
use crate::iteration_mark::{
    JapaneseIterationMarkCharFilter, NORMALIZE_KANA_DEFAULT, NORMALIZE_KANJI_DEFAULT,
};
use crate::katakana_stem::{JapaneseKatakanaStemFilter, DEFAULT_MINIMUM_LENGTH};
use crate::number::JapaneseNumberFilter;
use crate::pos_stop::japanese_part_of_speech_stop_filter;
use crate::reading_form::JapaneseReadingFormFilter;
use crate::tokenizer::{JapaneseTokenizer, Mode};
use crate::uppercase::{JapaneseHiraganaUppercaseFilter, JapaneseKatakanaUppercaseFilter};

macro_rules! base {
    ($t:ty) => {
        impl AnalysisFactory for $t {
            fn base(&self) -> &FactoryBase {
                &self.base
            }
            fn base_mut(&mut self) -> &mut FactoryBase {
                &mut self.base
            }
        }
    };
}

/// `Enum.valueOf(get(args, name, default).toUpperCase(Locale.ROOT))`.
fn upper(s: &str) -> String {
    String::from_utf16_lossy(&java_string_to_upper_case(
        &s.encode_utf16().collect::<Vec<_>>(),
    ))
}

/// `JapaneseTokenizerFactory` (`japanese`).
pub struct JapaneseTokenizerFactory {
    base: FactoryBase,
    mode: Mode,
    user_dictionary_path: Option<String>,
    user_dictionary_encoding: Option<String>,
    discard_punctuation: bool,
    discard_compound_token: bool,
    nbest_cost: i32,
    nbest_examples: Option<String>,
    user_dictionary: Option<Arc<UserDictionary>>,
}

impl FactoryClass for JapaneseTokenizerFactory {
    const NAME: &'static str = "japanese";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.ja.JapaneseTokenizerFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let mode = Mode::value_of(&upper(&args::get_or(args, "mode", "SEARCH")))?;
        let user_dictionary_path = args::get(args, "userDictionary");
        let user_dictionary_encoding = args::get(args, "userDictionaryEncoding");
        let discard_punctuation = args::get_boolean(args, "discardPunctuation", true);
        let discard_compound_token = args::get_boolean(args, "discardCompoundToken", true);
        let nbest_cost = args::get_int(args, "nBestCost", 0)?;
        let nbest_examples = args::get(args, "nBestExamples");
        args::reject_unknown(args)?;
        Ok(JapaneseTokenizerFactory {
            base,
            mode,
            user_dictionary_path,
            user_dictionary_encoding,
            discard_punctuation,
            discard_compound_token,
            nbest_cost,
            nbest_examples,
            user_dictionary: None,
        })
    }
}

/// Decodes a user dictionary resource in `encoding` (`UTF-8` when unset).
pub(crate) fn decode_user_dictionary(
    bytes: Vec<u8>,
    encoding: Option<&str>,
) -> Result<String, FactoryError> {
    match encoding.map(|e| e.to_ascii_uppercase()) {
        None => decode_utf8(bytes),
        Some(e) if e == "UTF-8" || e == "UTF8" => decode_utf8(bytes),
        Some(_) => Err(FactoryError::new(
            JavaException::UnsupportedEncoding,
            encoding.unwrap_or_default(),
        )),
    }
}

impl AnalysisFactory for JapaneseTokenizerFactory {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
    fn is_resource_loader_aware(&self) -> bool {
        true
    }
    // Java: JapaneseTokenizerFactory.inform
    fn inform(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        self.user_dictionary = match &self.user_dictionary_path {
            Some(path) => {
                let text = decode_user_dictionary(
                    loader.open_resource(path)?,
                    self.user_dictionary_encoding.as_deref(),
                )?;
                UserDictionary::open(&text)?.map(Arc::new)
            }
            None => None,
        };
        Ok(())
    }
}

impl TokenizerFactory for JapaneseTokenizerFactory {
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let mut t = JapaneseTokenizer::with_options(
            self.user_dictionary.clone(),
            self.discard_punctuation,
            self.discard_compound_token,
            self.mode,
        );
        let mut cost = self.nbest_cost;
        if let Some(examples) = &self.nbest_examples {
            cost = cost.max(t.calc_n_best_cost(examples)?);
        }
        t.set_n_best_cost(cost);
        Ok(Box::new(t))
    }
}

macro_rules! simple_filter {
    ($(#[$m:meta])* $name:ident, $java:literal, $class:literal, |$input:ident| $make:expr) => {
        $(#[$m])*
        pub struct $name {
            base: FactoryBase,
        }
        base!($name);
        impl FactoryClass for $name {
            const NAME: &'static str = $java;
            const CLASS_NAME: &'static str = $class;
            fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
                let base = FactoryBase::new(Self::CLASS_NAME, args)?;
                args::reject_unknown(args)?;
                Ok($name { base })
            }
        }
        impl TokenFilterFactory for $name {
            fn create(&self, $input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
                Ok(Box::new($make))
            }
        }
    };
}

simple_filter!(
    /// `JapaneseBaseFormFilterFactory` (`japaneseBaseForm`).
    JapaneseBaseFormFilterFactory,
    "japaneseBaseForm",
    "org.apache.lucene.analysis.ja.JapaneseBaseFormFilterFactory",
    |input| JapaneseBaseFormFilter::new(input)
);
simple_filter!(
    /// `JapaneseHiraganaUppercaseFilterFactory` (`japaneseHiraganaUppercase`).
    JapaneseHiraganaUppercaseFilterFactory,
    "japaneseHiraganaUppercase",
    "org.apache.lucene.analysis.ja.JapaneseHiraganaUppercaseFilterFactory",
    |input| JapaneseHiraganaUppercaseFilter::new(input)
);
simple_filter!(
    /// `JapaneseKatakanaUppercaseFilterFactory` (`japaneseKatakanaUppercase`).
    JapaneseKatakanaUppercaseFilterFactory,
    "japaneseKatakanaUppercase",
    "org.apache.lucene.analysis.ja.JapaneseKatakanaUppercaseFilterFactory",
    |input| JapaneseKatakanaUppercaseFilter::new(input)
);
simple_filter!(
    /// `JapaneseNumberFilterFactory` (`japaneseNumber`).
    JapaneseNumberFilterFactory,
    "japaneseNumber",
    "org.apache.lucene.analysis.ja.JapaneseNumberFilterFactory",
    |input| JapaneseNumberFilter::new(input)
);

/// `JapaneseCompletionFilterFactory` (`japaneseCompletion`).
pub struct JapaneseCompletionFilterFactory {
    base: FactoryBase,
    mode: CompletionMode,
}
base!(JapaneseCompletionFilterFactory);

impl FactoryClass for JapaneseCompletionFilterFactory {
    const NAME: &'static str = "japaneseCompletion";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.ja.JapaneseCompletionFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let mode = CompletionMode::value_of(&args::get_or(args, "mode", "INDEX"))?;
        args::reject_unknown(args)?;
        Ok(JapaneseCompletionFilterFactory { base, mode })
    }
}

impl TokenFilterFactory for JapaneseCompletionFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(JapaneseCompletionFilter::new(input, self.mode)))
    }
}

/// `JapaneseKatakanaStemFilterFactory` (`japaneseKatakanaStem`).
pub struct JapaneseKatakanaStemFilterFactory {
    base: FactoryBase,
    minimum_length: i32,
}
base!(JapaneseKatakanaStemFilterFactory);

impl FactoryClass for JapaneseKatakanaStemFilterFactory {
    const NAME: &'static str = "japaneseKatakanaStem";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.ja.JapaneseKatakanaStemFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let minimum_length = args::get_int(args, "minimumLength", DEFAULT_MINIMUM_LENGTH)?;
        if minimum_length < 2 {
            return Err(FactoryError::illegal_argument(format!(
                "Illegal minimumLength {minimum_length} (must be 2 or greater)"
            )));
        }
        args::reject_unknown(args)?;
        Ok(JapaneseKatakanaStemFilterFactory {
            base,
            minimum_length,
        })
    }
}

impl TokenFilterFactory for JapaneseKatakanaStemFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(JapaneseKatakanaStemFilter::new(
            input,
            self.minimum_length,
        )?))
    }
}

/// `JapaneseReadingFormFilterFactory` (`japaneseReadingForm`).
pub struct JapaneseReadingFormFilterFactory {
    base: FactoryBase,
    use_romaji: bool,
}
base!(JapaneseReadingFormFilterFactory);

impl FactoryClass for JapaneseReadingFormFilterFactory {
    const NAME: &'static str = "japaneseReadingForm";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.ja.JapaneseReadingFormFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let use_romaji = args::get_boolean(args, "useRomaji", false);
        args::reject_unknown(args)?;
        Ok(JapaneseReadingFormFilterFactory { base, use_romaji })
    }
}

impl TokenFilterFactory for JapaneseReadingFormFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(JapaneseReadingFormFilter::new(
            input,
            self.use_romaji,
        )))
    }
}

/// `JapanesePartOfSpeechStopFilterFactory` (`japanesePartOfSpeechStop`):
/// `tags` names the stop-tag files (default: the analyzer's); an empty
/// file set means no filter.
pub struct JapanesePartOfSpeechStopFilterFactory {
    base: FactoryBase,
    stop_tag_files: Option<String>,
    stop_tags: Option<Arc<HashSet<String>>>,
}

impl FactoryClass for JapanesePartOfSpeechStopFilterFactory {
    const NAME: &'static str = "japanesePartOfSpeechStop";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.ja.JapanesePartOfSpeechStopFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let stop_tag_files = args::get(args, "tags");
        let stop_tags = stop_tag_files.is_none().then(default_stop_tags);
        args::reject_unknown(args)?;
        Ok(JapanesePartOfSpeechStopFilterFactory {
            base,
            stop_tag_files,
            stop_tags,
        })
    }
}

impl AnalysisFactory for JapanesePartOfSpeechStopFilterFactory {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
    fn is_resource_loader_aware(&self) -> bool {
        true
    }
    // Java: JapanesePartOfSpeechStopFilterFactory.inform
    fn inform(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        if let Some(files) = &self.stop_tag_files {
            self.stop_tags = get_word_set(loader, files, false)?
                .map(|cas| Arc::new(cas.iter().map(str::to_string).collect()));
        }
        Ok(())
    }
}

impl TokenFilterFactory for JapanesePartOfSpeechStopFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        // if stoptags is null, it means the file is empty
        Ok(match &self.stop_tags {
            Some(tags) => Box::new(japanese_part_of_speech_stop_filter(input, Arc::clone(tags))),
            None => input,
        })
    }
}

/// `JapaneseIterationMarkCharFilterFactory` (`japaneseIterationMark`).
pub struct JapaneseIterationMarkCharFilterFactory {
    base: FactoryBase,
    normalize_kanji: bool,
    normalize_kana: bool,
}
base!(JapaneseIterationMarkCharFilterFactory);

impl FactoryClass for JapaneseIterationMarkCharFilterFactory {
    const NAME: &'static str = "japaneseIterationMark";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.ja.JapaneseIterationMarkCharFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let normalize_kanji = args::get_boolean(args, "normalizeKanji", NORMALIZE_KANJI_DEFAULT);
        let normalize_kana = args::get_boolean(args, "normalizeKana", NORMALIZE_KANA_DEFAULT);
        args::reject_unknown(args)?;
        Ok(JapaneseIterationMarkCharFilterFactory {
            base,
            normalize_kanji,
            normalize_kana,
        })
    }
}

impl CharFilterFactory for JapaneseIterationMarkCharFilterFactory {
    fn create(&self, input: Box<dyn CharReader>) -> Box<dyn CharReader> {
        Box::new(JapaneseIterationMarkCharFilter::new(
            input,
            self.normalize_kanji,
            self.normalize_kana,
        ))
    }
    fn normalize(&self, input: Box<dyn CharReader>) -> Box<dyn CharReader> {
        self.create(input)
    }
}

/// Registers the analysis-kuromoji factories with `lucene-analysis`' SPI
/// registry (idempotent).
pub fn register_factories() -> Result<(), FactoryError> {
    spi::register_tokenizer(spi::tokenizer_entry::<JapaneseTokenizerFactory>())?;
    spi::register_char_filter(spi::char_filter_entry::<
        JapaneseIterationMarkCharFilterFactory,
    >())?;
    spi::register_token_filter(spi::token_filter_entry::<JapaneseBaseFormFilterFactory>())?;
    spi::register_token_filter(spi::token_filter_entry::<JapaneseCompletionFilterFactory>())?;
    spi::register_token_filter(spi::token_filter_entry::<JapaneseKatakanaStemFilterFactory>())?;
    spi::register_token_filter(spi::token_filter_entry::<JapaneseNumberFilterFactory>())?;
    spi::register_token_filter(spi::token_filter_entry::<
        JapanesePartOfSpeechStopFilterFactory,
    >())?;
    spi::register_token_filter(spi::token_filter_entry::<JapaneseReadingFormFilterFactory>())?;
    spi::register_token_filter(spi::token_filter_entry::<
        JapaneseHiraganaUppercaseFilterFactory,
    >())?;
    spi::register_token_filter(spi::token_filter_entry::<
        JapaneseKatakanaUppercaseFilterFactory,
    >())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodings_and_unconfigured_factories() {
        assert_eq!(
            decode_user_dictionary(b"a".to_vec(), Some("utf8")).unwrap(),
            "a"
        );
        let e = decode_user_dictionary(b"a".to_vec(), Some("Shift_JIS")).unwrap_err();
        assert_eq!(e.kind, JavaException::UnsupportedEncoding);
        assert!(decode_user_dictionary(vec![0xFF], None).is_err());
        let mut f =
            JapaneseTokenizerFactory::from_args(&mut JavaArgs::from_pairs::<&str, &str>(&[]))
                .unwrap();
        assert!(f.is_resource_loader_aware());
        assert!(f
            .base_mut()
            .class_name()
            .ends_with("JapaneseTokenizerFactory"));
        assert!(f.base().class_name().ends_with("Factory"));
        let mut p = JapanesePartOfSpeechStopFilterFactory::from_args(&mut JavaArgs::from_pairs::<
            &str,
            &str,
        >(&[]))
        .unwrap();
        assert!(p.is_resource_loader_aware());
        assert!(p.base_mut().class_name().ends_with("StopFilterFactory"));
        assert!(p.base().class_name().ends_with("Factory"));
        let mut i = JapaneseIterationMarkCharFilterFactory::from_args(&mut JavaArgs::from_pairs::<
            &str,
            &str,
        >(&[]))
        .unwrap();
        assert!(i.base_mut().class_name().ends_with("Factory"));
        register_factories().unwrap();
    }
}
