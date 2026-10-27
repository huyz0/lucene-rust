//! The module's SPI factories -- `OpenNLPTokenizerFactory` (`openNlp`),
//! `OpenNLPPOSFilterFactory` (`openNlppos`), `OpenNLPChunkerFilterFactory`
//! (`openNlpChunker`), `OpenNLPLemmatizerFilterFactory`
//! (`openNlpLemmatizer`) -- [`register_factories`], and
//! [`ner_tagger_model`] (`OpenNLPOpsFactory.getNERTaggerModel`, which no
//! factory calls).
//!
//! Each reads its models from the resource loader in `inform`; the
//! taggers themselves are created in `create`, as Lucene's are (so a
//! problem with a model's beam size or feature generators surfaces when
//! the analyzer builds its components, not when the factory is informed).
//! The POS and chunker factories wrap an `IOException` from reading a
//! model in an `IllegalArgumentException`, the tokenizer and lemmatizer
//! factories let it through, as Lucene's do.
//!
//! Differs: Lucene's `OpenNLPOpsFactory` caches every model in a static map
//! keyed by its resource name alone, so a second analyzer naming the same
//! file gets the first one's model even through another loader; here each
//! factory reads its own (the same model whenever the name names the same
//! bytes).

use std::sync::Arc;

use lucene_analysis::factory::args::{self, JavaArgs};
use lucene_analysis::factory::spi;
use lucene_analysis::factory::{
    AnalysisFactory, FactoryBase, FactoryClass, FactoryError, JavaException, ResourceLoader,
    TokenFilterFactory, TokenizerFactory,
};
use lucene_analysis::{AnalysisError, TokenStream};

use crate::filters::{OpenNlpChunkerFilter, OpenNlpLemmatizerFilter, OpenNlpPosFilter};
use crate::ops::LemmatizerOp;
use crate::tokenizer::open_nlp_tokenizer;
use crate::tools::chunker::{Chunker, ChunkerModel};
use crate::tools::dictionary::DictionaryLemmatizer;
use crate::tools::lemmatizer::{Lemmatizer, LemmatizerModel};
use crate::tools::namefind::NameFinderModel;
use crate::tools::postag::{PosModel, PosTagger};
use crate::tools::sentdetect::{SentenceDetector, SentenceModel};
use crate::tools::tokenize::{Tokenizer, TokenizerModel};

macro_rules! aware_factory {
    ($t:ty) => {
        impl AnalysisFactory for $t {
            fn base(&self) -> &FactoryBase {
                &self.base
            }
            fn base_mut(&mut self) -> &mut FactoryBase {
                &mut self.base
            }
            fn is_resource_loader_aware(&self) -> bool {
                true
            }
            fn inform(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
                self.inform_impl(loader)
            }
        }
    };
}

/// `catch (IOException e) { throw new IllegalArgumentException(e); }`.
fn io_as_illegal_argument(e: FactoryError) -> FactoryError {
    match e.kind {
        JavaException::Io => FactoryError::illegal_argument(e.message),
        _ => e,
    }
}

/// A model's absence where Java dereferences it (`create` before
/// `inform`).
fn no_model(what: &str) -> AnalysisError {
    AnalysisError::IllegalState(format!("NullPointerException: no {what} model was loaded"))
}

/// `OpenNLPTokenizerFactory`: `sentenceModel` and `tokenizerModel`, both
/// required.
pub struct OpenNlpTokenizerFactory {
    base: FactoryBase,
    sentence_model_file: String,
    tokenizer_model_file: String,
    sentence_model: Option<SentenceModel>,
    tokenizer_model: Option<TokenizerModel>,
}
aware_factory!(OpenNlpTokenizerFactory);

impl FactoryClass for OpenNlpTokenizerFactory {
    const NAME: &'static str = "openNlp";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.opennlp.OpenNLPTokenizerFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let sentence_model_file = args::require(args, "sentenceModel")?;
        let tokenizer_model_file = args::require(args, "tokenizerModel")?;
        args::reject_unknown(args)?;
        Ok(OpenNlpTokenizerFactory {
            base,
            sentence_model_file,
            tokenizer_model_file,
            sentence_model: None,
            tokenizer_model: None,
        })
    }
}

impl OpenNlpTokenizerFactory {
    fn inform_impl(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        let bytes = loader.open_resource(&self.sentence_model_file)?;
        self.sentence_model = Some(SentenceModel::load(&bytes)?);
        let bytes = loader.open_resource(&self.tokenizer_model_file)?;
        self.tokenizer_model = Some(TokenizerModel::load(&bytes)?);
        Ok(())
    }
}

impl TokenizerFactory for OpenNlpTokenizerFactory {
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let sentence = self
            .sentence_model
            .clone()
            .ok_or_else(|| no_model("sentence"))?;
        let tokenizer = self
            .tokenizer_model
            .clone()
            .ok_or_else(|| no_model("tokenizer"))?;
        Ok(Box::new(open_nlp_tokenizer(
            SentenceDetector::new(sentence),
            Tokenizer::new(tokenizer)?,
        )))
    }
}

/// `OpenNLPPOSFilterFactory`: `posTaggerModel`, required.
pub struct OpenNlpPosFilterFactory {
    base: FactoryBase,
    model_file: String,
    model: Option<PosModel>,
}
aware_factory!(OpenNlpPosFilterFactory);

impl FactoryClass for OpenNlpPosFilterFactory {
    const NAME: &'static str = "openNlppos";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.opennlp.OpenNLPPOSFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let model_file = args::require(args, "posTaggerModel")?;
        args::reject_unknown(args)?;
        Ok(OpenNlpPosFilterFactory {
            base,
            model_file,
            model: None,
        })
    }
}

impl OpenNlpPosFilterFactory {
    fn inform_impl(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        let bytes = loader
            .open_resource(&self.model_file)
            .map_err(io_as_illegal_argument)?;
        self.model = Some(PosModel::load(&bytes).map_err(io_as_illegal_argument)?);
        Ok(())
    }
}

impl TokenFilterFactory for OpenNlpPosFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let model = self.model.as_ref().ok_or_else(|| no_model("POS tagger"))?;
        Ok(Box::new(OpenNlpPosFilter::new(
            input,
            PosTagger::new(model)?,
        )))
    }
}

/// `OpenNLPChunkerFilterFactory`: `chunkerModel`, optional (without one
/// the filter fails on its first sentence, as Lucene's).
pub struct OpenNlpChunkerFilterFactory {
    base: FactoryBase,
    model_file: Option<String>,
    model: Option<ChunkerModel>,
}
aware_factory!(OpenNlpChunkerFilterFactory);

impl FactoryClass for OpenNlpChunkerFilterFactory {
    const NAME: &'static str = "openNlpChunker";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.opennlp.OpenNLPChunkerFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let model_file = args::get(args, "chunkerModel");
        args::reject_unknown(args)?;
        Ok(OpenNlpChunkerFilterFactory {
            base,
            model_file,
            model: None,
        })
    }
}

impl OpenNlpChunkerFilterFactory {
    fn inform_impl(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        if let Some(file) = &self.model_file {
            let bytes = loader.open_resource(file).map_err(io_as_illegal_argument)?;
            self.model = Some(ChunkerModel::load(&bytes).map_err(io_as_illegal_argument)?);
        }
        Ok(())
    }
}

impl TokenFilterFactory for OpenNlpChunkerFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let chunker = match (&self.model_file, &self.model) {
            (None, _) => None,
            (Some(_), Some(m)) => Some(Chunker::new(m)?),
            (Some(_), None) => return Err(no_model("chunker")),
        };
        Ok(Box::new(OpenNlpChunkerFilter::new(input, chunker)))
    }
}

/// `OpenNLPLemmatizerFilterFactory`: `dictionary` and/or
/// `lemmatizerModel`.
pub struct OpenNlpLemmatizerFilterFactory {
    base: FactoryBase,
    dictionary_file: Option<String>,
    model_file: Option<String>,
    dictionary: Option<Arc<DictionaryLemmatizer>>,
    model: Option<LemmatizerModel>,
}
aware_factory!(OpenNlpLemmatizerFilterFactory);

impl FactoryClass for OpenNlpLemmatizerFilterFactory {
    const NAME: &'static str = "openNlpLemmatizer";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.opennlp.OpenNLPLemmatizerFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let dictionary_file = args::get(args, "dictionary");
        let model_file = args::get(args, "lemmatizerModel");
        if dictionary_file.is_none() && model_file.is_none() {
            return Err(FactoryError::illegal_argument(
                "Configuration Error: missing parameter: at least one of 'dictionary' and 'lemmatizerModel' must be provided.",
            ));
        }
        args::reject_unknown(args)?;
        Ok(OpenNlpLemmatizerFilterFactory {
            base,
            dictionary_file,
            model_file,
            dictionary: None,
            model: None,
        })
    }
}

impl OpenNlpLemmatizerFilterFactory {
    fn inform_impl(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        if let Some(file) = &self.dictionary_file {
            let bytes = loader.open_resource(file)?;
            self.dictionary = Some(Arc::new(DictionaryLemmatizer::read(&bytes)?));
        }
        if let Some(file) = &self.model_file {
            let bytes = loader.open_resource(file)?;
            self.model = Some(LemmatizerModel::load(&bytes)?);
        }
        Ok(())
    }
}

impl TokenFilterFactory for OpenNlpLemmatizerFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let lemmatizer = self.model.as_ref().map(Lemmatizer::new).transpose()?;
        let op = LemmatizerOp::new(self.dictionary.clone(), lemmatizer);
        Ok(Box::new(OpenNlpLemmatizerFilter::new(input, op)))
    }
}

/// `OpenNLPOpsFactory.getNERTaggerModel(modelName, loader)` (read anew,
/// not cached by name: see the module notes); its `IOException`s pass
/// through. `NerTaggerOp::new` over it is `getNERTagger`.
pub fn ner_tagger_model(
    model_name: &str,
    loader: &dyn ResourceLoader,
) -> Result<NameFinderModel, FactoryError> {
    NameFinderModel::load(&loader.open_resource(model_name)?)
}

/// Adds the module's factories to `lucene-analysis`' SPI registry (Java
/// finds them on the classpath).
pub fn register_factories() -> Result<(), FactoryError> {
    spi::register_tokenizer(spi::tokenizer_entry::<OpenNlpTokenizerFactory>())?;
    spi::register_token_filter(spi::token_filter_entry::<OpenNlpPosFilterFactory>())?;
    spi::register_token_filter(spi::token_filter_entry::<OpenNlpChunkerFilterFactory>())?;
    spi::register_token_filter(spi::token_filter_entry::<OpenNlpLemmatizerFilterFactory>())?;
    Ok(())
}
