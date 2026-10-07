//! `Word2VecSynonymFilterFactory` (`Word2VecSynonym`) and
//! `Word2VecSynonymProviderFactory`: the factory of the filter this module
//! ports, registered into `lucene-analysis`' SPI registry by
//! [`register_factories`] (Java finds it on the classpath with the rest of
//! analysis-common).
//!
//! As in Java, providers are cached for the life of the process by model
//! file name, whatever loader read them.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use lucene_analysis::factory::args::{self, JavaArgs};
use lucene_analysis::factory::spi;
use lucene_analysis::factory::{
    AnalysisFactory, FactoryBase, FactoryClass, FactoryError, ResourceLoader, TokenFilterFactory,
};
use lucene_analysis::{AnalysisError, TokenStream};

use super::{read_dl4j_model, Word2VecSynonymFilter, Word2VecSynonymProvider};
use crate::Error;

/// `Word2VecSynonymFilterFactory.DEFAULT_MAX_SYNONYMS_PER_TERM`.
pub const DEFAULT_MAX_SYNONYMS_PER_TERM: i32 = 5;
/// `Word2VecSynonymFilterFactory.DEFAULT_MIN_ACCEPTED_SIMILARITY`.
pub const DEFAULT_MIN_ACCEPTED_SIMILARITY: f32 = 0.8;

/// `Word2VecSynonymProviderFactory.word2vecSynonymProviders`.
static PROVIDERS: LazyLock<Mutex<HashMap<String, Arc<Word2VecSynonymProvider>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn factory_error(e: Error) -> FactoryError {
    match e {
        Error::IllegalArgument(m) => FactoryError::illegal_argument(m),
        other => FactoryError::io(other.to_string()),
    }
}

/// `Word2VecSynonymProviderFactory.getSynonymProvider(loader, modelFileName,
/// DL4J)`: the cached provider of the model, read on first use.
pub fn get_synonym_provider(
    loader: &dyn ResourceLoader,
    model_file_name: &str,
) -> Result<Arc<Word2VecSynonymProvider>, FactoryError> {
    let mut cache = PROVIDERS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(p) = cache.get(model_file_name) {
        return Ok(Arc::clone(p));
    }
    let bytes = loader.open_resource(model_file_name)?;
    let model = read_dl4j_model(&bytes).map_err(factory_error)?;
    let provider = Arc::new(Word2VecSynonymProvider::new(model).map_err(factory_error)?);
    cache.insert(model_file_name.to_string(), Arc::clone(&provider));
    Ok(provider)
}

/// `org.apache.lucene.analysis.synonym.word2vec.Word2VecSynonymFilterFactory`
/// (`Word2VecSynonym`): `model` (a DL4J zip), `format` (`dl4j`, any case),
/// `maxSynonymsPerTerm`, `minAcceptedSimilarity`. Differs: a similarity out
/// of range is printed with Rust's float formatting where Java's
/// `Float.toString` would use an exponent (below 10^-3 or from 10^7).
pub struct Word2VecSynonymFilterFactory {
    base: FactoryBase,
    max_synonyms_per_term: i32,
    min_accepted_similarity: f32,
    model_file_name: String,
    provider: Option<Arc<Word2VecSynonymProvider>>,
}

impl FactoryClass for Word2VecSynonymFilterFactory {
    const NAME: &'static str = "Word2VecSynonym";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.synonym.word2vec.Word2VecSynonymFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let max_synonyms_per_term =
            args::get_int(args, "maxSynonymsPerTerm", DEFAULT_MAX_SYNONYMS_PER_TERM)?;
        let min_accepted_similarity = args::get_float(
            args,
            "minAcceptedSimilarity",
            DEFAULT_MIN_ACCEPTED_SIMILARITY,
        )?;
        let model_file_name = args::require(args, "model")?;
        // .toUpperCase(Locale.ROOT), then Word2VecSupportedFormats.valueOf
        let format = args::get_or(args, "format", "dl4j").to_uppercase();
        if format != "DL4J" {
            return Err(FactoryError::illegal_argument(format!(
                "Model format '{format}' not supported"
            )));
        }
        args::reject_unknown(args)?;
        if min_accepted_similarity <= 0.0 || min_accepted_similarity > 1.0 {
            return Err(FactoryError::illegal_argument(format!(
                "minAcceptedSimilarity must be in the range (0, 1]. Found: {min_accepted_similarity:?}"
            )));
        }
        if max_synonyms_per_term <= 0 {
            return Err(FactoryError::illegal_argument(format!(
                "maxSynonymsPerTerm must be a positive integer greater than 0. Found: {max_synonyms_per_term}"
            )));
        }
        Ok(Word2VecSynonymFilterFactory {
            base,
            max_synonyms_per_term,
            min_accepted_similarity,
            model_file_name,
            provider: None,
        })
    }
}

impl AnalysisFactory for Word2VecSynonymFilterFactory {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
    fn is_resource_loader_aware(&self) -> bool {
        true
    }
    // Java: Word2VecSynonymFilterFactory.inform
    fn inform(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        self.provider = Some(get_synonym_provider(loader, &self.model_file_name)?);
        Ok(())
    }
}

impl TokenFilterFactory for Word2VecSynonymFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(match &self.provider {
            None => input,
            Some(p) => Box::new(Word2VecSynonymFilter::new(
                input,
                Arc::clone(p),
                self.max_synonyms_per_term,
                self.min_accepted_similarity,
            )),
        })
    }
}

/// Registers [`Word2VecSynonymFilterFactory`] with `lucene-analysis`' SPI
/// registry (idempotent).
pub fn register_factories() -> Result<(), FactoryError> {
    spi::register_token_filter(spi::token_filter_entry::<Word2VecSynonymFilterFactory>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::factory::{JavaException, MapResourceLoader};

    fn build(pairs: &[(&str, &str)]) -> Result<Word2VecSynonymFilterFactory, FactoryError> {
        Word2VecSynonymFilterFactory::from_args(&mut JavaArgs::from_pairs(pairs))
    }

    #[test]
    fn arguments_are_javas() {
        let f = build(&[("model", "m.zip"), ("format", "Dl4j")]).unwrap();
        assert_eq!(f.max_synonyms_per_term, 5);
        assert_eq!(f.min_accepted_similarity, 0.8);
        assert!(f.is_resource_loader_aware());
        assert_eq!(
            build(&[("model", "m"), ("format", "bin")])
                .err()
                .unwrap()
                .message,
            "Model format 'BIN' not supported"
        );
        assert_eq!(
            build(&[("model", "m"), ("minAcceptedSimilarity", "0")])
                .err()
                .unwrap()
                .message,
            "minAcceptedSimilarity must be in the range (0, 1]. Found: 0.0"
        );
        assert!(build(&[("model", "m"), ("maxSynonymsPerTerm", "-1")]).is_err());
        assert!(build(&[]).is_err());
        assert!(build(&[("model", "m"), ("x", "1")]).is_err());
    }

    #[test]
    fn a_bad_model_is_an_error_and_an_uninformed_factory_passes_through() {
        let loader = MapResourceLoader::new().with("bad.zip", b"not a zip");
        let mut f = build(&[("model", "bad.zip")]).unwrap();
        assert!(f.inform(&loader).is_err());
        let mut g = build(&[("model", "missing.zip")]).unwrap();
        assert_eq!(g.inform(&loader).err().unwrap().kind, JavaException::Io);
        let input: Box<dyn TokenStream> = Box::new(lucene_analysis::KeywordTokenizer::new());
        assert!(f.create(input).is_ok());
        assert!(f
            .base_mut()
            .class_name()
            .ends_with("Word2VecSynonymFilterFactory"));
        assert_eq!(
            factory_error(Error::IllegalArgument("x".into())).kind,
            JavaException::IllegalArgument
        );
        register_factories().unwrap();
        assert!(spi::lookup_token_filter("word2vecsynonym").is_ok());
    }
}
