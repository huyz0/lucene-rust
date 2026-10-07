//! The analysis-nori factories (`korean`, `koreanPartOfSpeechStop`,
//! `koreanReadingForm`, `koreanNumber`) and [`register_factories`].
//!
//! Differs: `userDictionaryEncoding` other than UTF-8 is refused with
//! `UnsupportedEncodingException` (Java decodes any JDK charset).

use std::collections::HashSet;
use std::sync::Arc;

use lucene_analysis::factory::args::{self, JavaArgs};
use lucene_analysis::factory::{
    decode_utf8, spi, AnalysisFactory, FactoryBase, FactoryClass, FactoryError, JavaException,
    ResourceLoader, TokenFilterFactory, TokenizerFactory,
};
use lucene_analysis::lang::java_string_to_upper_case;
use lucene_analysis::{AnalysisError, TokenStream};

use crate::dict::UserDictionary;
use crate::number::KoreanNumberFilter;
use crate::pos::Tag;
use crate::pos_stop::{default_stop_tags, korean_part_of_speech_stop_filter};
use crate::reading_form::KoreanReadingFormFilter;
use crate::tokenizer::KoreanTokenizer;
use crate::viterbi::DecompoundMode;

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

/// Decodes a user dictionary resource in `encoding` (`UTF-8` when unset).
fn decode_user_dictionary(bytes: Vec<u8>, encoding: Option<&str>) -> Result<String, FactoryError> {
    match encoding.map(|e| e.to_ascii_uppercase()) {
        None => decode_utf8(bytes),
        Some(e) if e == "UTF-8" || e == "UTF8" => decode_utf8(bytes),
        Some(_) => Err(FactoryError::new(
            JavaException::UnsupportedEncoding,
            encoding.unwrap_or_default(),
        )),
    }
}

/// `KoreanTokenizerFactory` (`korean`).
pub struct KoreanTokenizerFactory {
    base: FactoryBase,
    user_dictionary_path: Option<String>,
    user_dictionary_encoding: Option<String>,
    mode: DecompoundMode,
    output_unknown_unigrams: bool,
    discard_punctuation: bool,
    user_dictionary: Option<Arc<UserDictionary>>,
}

impl FactoryClass for KoreanTokenizerFactory {
    const NAME: &'static str = "korean";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.ko.KoreanTokenizerFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let user_dictionary_path = args::get(args, "userDictionary");
        let user_dictionary_encoding = args::get(args, "userDictionaryEncoding");
        let mode = args::get_or(args, "decompoundMode", "DISCARD");
        let mode = String::from_utf16_lossy(&java_string_to_upper_case(
            &mode.encode_utf16().collect::<Vec<_>>(),
        ));
        let mode = DecompoundMode::value_of(&mode)?;
        let output_unknown_unigrams = args::get_boolean(args, "outputUnknownUnigrams", false);
        let discard_punctuation = args::get_boolean(args, "discardPunctuation", true);
        args::reject_unknown(args)?;
        Ok(KoreanTokenizerFactory {
            base,
            user_dictionary_path,
            user_dictionary_encoding,
            mode,
            output_unknown_unigrams,
            discard_punctuation,
            user_dictionary: None,
        })
    }
}

impl AnalysisFactory for KoreanTokenizerFactory {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
    fn is_resource_loader_aware(&self) -> bool {
        true
    }
    // Java: KoreanTokenizerFactory.inform
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

impl TokenizerFactory for KoreanTokenizerFactory {
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(KoreanTokenizer::new(
            self.user_dictionary.clone(),
            self.mode,
            self.output_unknown_unigrams,
            self.discard_punctuation,
        )))
    }
}

/// `KoreanPartOfSpeechStopFilterFactory` (`koreanPartOfSpeechStop`): `tags`
/// lists the stop tags (default: the filter's).
pub struct KoreanPartOfSpeechStopFilterFactory {
    base: FactoryBase,
    stop_tags: Arc<HashSet<Tag>>,
}
base!(KoreanPartOfSpeechStopFilterFactory);

impl FactoryClass for KoreanPartOfSpeechStopFilterFactory {
    const NAME: &'static str = "koreanPartOfSpeechStop";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.ko.KoreanPartOfSpeechStopFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let stop_tags = match args::get_set(args, "tags") {
            None => default_stop_tags(),
            Some(names) => Arc::new(
                names
                    .iter()
                    .map(|n| Tag::resolve_name(n))
                    .collect::<Result<HashSet<_>, _>>()?,
            ),
        };
        args::reject_unknown(args)?;
        Ok(KoreanPartOfSpeechStopFilterFactory { base, stop_tags })
    }
}

impl TokenFilterFactory for KoreanPartOfSpeechStopFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(korean_part_of_speech_stop_filter(
            input,
            Arc::clone(&self.stop_tags),
        )))
    }
}

/// `KoreanReadingFormFilterFactory` (`koreanReadingForm`).
pub struct KoreanReadingFormFilterFactory {
    base: FactoryBase,
}
base!(KoreanReadingFormFilterFactory);

impl FactoryClass for KoreanReadingFormFilterFactory {
    const NAME: &'static str = "koreanReadingForm";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.ko.KoreanReadingFormFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        args::reject_unknown(args)?;
        Ok(KoreanReadingFormFilterFactory { base })
    }
}

impl TokenFilterFactory for KoreanReadingFormFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(KoreanReadingFormFilter::new(input)))
    }
}

/// `KoreanNumberFilterFactory` (`koreanNumber`).
pub struct KoreanNumberFilterFactory {
    base: FactoryBase,
}
base!(KoreanNumberFilterFactory);

impl FactoryClass for KoreanNumberFilterFactory {
    const NAME: &'static str = "koreanNumber";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.ko.KoreanNumberFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        args::reject_unknown(args)?;
        Ok(KoreanNumberFilterFactory { base })
    }
}

impl TokenFilterFactory for KoreanNumberFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(KoreanNumberFilter::new(input)))
    }
}

/// Registers the analysis-nori factories with `lucene-analysis`' SPI
/// registry (idempotent).
pub fn register_factories() -> Result<(), FactoryError> {
    spi::register_tokenizer(spi::tokenizer_entry::<KoreanTokenizerFactory>())?;
    spi::register_token_filter(spi::token_filter_entry::<KoreanPartOfSpeechStopFilterFactory>())?;
    spi::register_token_filter(spi::token_filter_entry::<KoreanReadingFormFilterFactory>())?;
    spi::register_token_filter(spi::token_filter_entry::<KoreanNumberFilterFactory>())
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
        assert_eq!(decode_user_dictionary(b"b".to_vec(), None).unwrap(), "b");
        let e = decode_user_dictionary(b"a".to_vec(), Some("EUC-KR")).unwrap_err();
        assert_eq!(e.kind, JavaException::UnsupportedEncoding);
        let none = || JavaArgs::from_pairs::<&str, &str>(&[]);
        let mut t = KoreanTokenizerFactory::from_args(&mut none()).unwrap();
        assert!(t.is_resource_loader_aware());
        assert!(t
            .base_mut()
            .class_name()
            .ends_with("KoreanTokenizerFactory"));
        assert!(t.base().class_name().ends_with("Factory"));
        let mut p = KoreanPartOfSpeechStopFilterFactory::from_args(&mut none()).unwrap();
        assert!(p.base_mut().class_name().ends_with("StopFilterFactory"));
        let mut r = KoreanReadingFormFilterFactory::from_args(&mut none()).unwrap();
        assert!(r
            .base_mut()
            .class_name()
            .ends_with("ReadingFormFilterFactory"));
        let mut n = KoreanNumberFilterFactory::from_args(&mut none()).unwrap();
        assert!(n.base_mut().class_name().ends_with("NumberFilterFactory"));
        register_factories().unwrap();
    }
}
