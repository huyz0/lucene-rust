//! `org.apache.lucene.analysis.morfologik.MorfologikFilterFactory`
//! (`morfologik`) and [`register_factories`].

use std::sync::Arc;

use lucene_analysis::factory::args::{self, JavaArgs};
use lucene_analysis::factory::spi;
use lucene_analysis::factory::{
    AnalysisFactory, FactoryBase, FactoryClass, FactoryError, JavaException, ResourceLoader,
    TokenFilterFactory,
};
use lucene_analysis::{AnalysisError, TokenStream};

use crate::analyzer::polish_dictionary;
use crate::dictionary::Dictionary;
use crate::filter::MorfologikFilter;

/// `DictionaryMetadata.getExpectedMetadataFileName`: the name with its
/// extension replaced by `.info`.
pub fn expected_metadata_file_name(dictionary_file: &str) -> String {
    match dictionary_file.rfind('.') {
        Some(dot) => format!("{}.info", &dictionary_file[..dot]),
        None => format!("{dictionary_file}.info"),
    }
}

/// `MorfologikFilterFactory`: `dictionary` names a dictionary resource
/// (its `.info` beside it); without it, the Polish dictionary.
pub struct MorfologikFilterFactory {
    base: FactoryBase,
    resource_name: Option<String>,
    dictionary: Option<Arc<Dictionary>>,
}

impl FactoryClass for MorfologikFilterFactory {
    const NAME: &'static str = "morfologik";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.morfologik.MorfologikFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        if args::get(args, "dictionary-resource").is_some_and(|r| !r.is_empty()) {
            return Err(FactoryError::illegal_argument(
                "The dictionary-resource attribute is no longer supported. Use the 'dictionary' attribute instead (see LUCENE-6833).",
            ));
        }
        let resource_name = args::get(args, "dictionary");
        args::reject_unknown(args)?;
        Ok(MorfologikFilterFactory {
            base,
            resource_name,
            dictionary: None,
        })
    }
}

impl AnalysisFactory for MorfologikFilterFactory {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
    fn is_resource_loader_aware(&self) -> bool {
        true
    }
    // Java: MorfologikFilterFactory.inform
    fn inform(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        self.dictionary = Some(match &self.resource_name {
            None => polish_dictionary(),
            Some(name) => {
                let fsa = loader.open_resource(name)?;
                let meta = loader.open_resource(&expected_metadata_file_name(name))?;
                let meta = String::from_utf8_lossy(&meta);
                Arc::new(Dictionary::read(&fsa, &meta).map_err(|e| FactoryError::io(e.message()))?)
            }
        });
        Ok(())
    }
}

impl TokenFilterFactory for MorfologikFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let dictionary = self.dictionary.clone().ok_or_else(|| {
            AnalysisError::from(FactoryError::new(
                JavaException::NullPointer,
                "MorfologikFilterFactory was not fully initialized.",
            ))
        })?;
        Ok(Box::new(MorfologikFilter::new(input, dictionary)))
    }
}

/// Registers [`MorfologikFilterFactory`] with `lucene-analysis`' SPI
/// registry (idempotent).
pub fn register_factories() -> Result<(), FactoryError> {
    spi::register_token_filter(spi::token_filter_entry::<MorfologikFilterFactory>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::factory::MapResourceLoader;

    #[test]
    fn arguments_and_resources() {
        assert_eq!(expected_metadata_file_name("a/b.dict"), "a/b.info");
        assert_eq!(expected_metadata_file_name("dict"), "dict.info");
        let build = |pairs: &[(&str, &str)]| {
            MorfologikFilterFactory::from_args(&mut JavaArgs::from_pairs(pairs))
        };
        assert!(build(&[("dictionary-resource", "x")]).is_err());
        assert!(build(&[("dictionary-resource", "")]).is_ok());
        assert!(build(&[("y", "1")]).is_err());
        let mut f = build(&[("dictionary", "missing.dict")]).unwrap();
        assert!(f.is_resource_loader_aware());
        assert!(f.base().class_name().ends_with("MorfologikFilterFactory"));
        let input =
            || -> Box<dyn TokenStream> { Box::new(lucene_analysis::KeywordTokenizer::new()) };
        assert!(f.create(input()).is_err());
        assert!(f.inform(&MapResourceLoader::new()).is_err());
        let bad = MapResourceLoader::new()
            .with("bad.dict", b"nope")
            .with("bad.info", b"x=1");
        let mut g = build(&[("dictionary", "bad.dict")]).unwrap();
        assert_eq!(g.inform(&bad).unwrap_err().kind, JavaException::Io);
        assert!(g.base_mut().class_name().ends_with("Factory"));
        register_factories().unwrap();
        assert!(spi::lookup_token_filter("morfologik").is_ok());
    }
}
