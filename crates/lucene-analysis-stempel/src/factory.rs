//! `org.apache.lucene.analysis.stempel.StempelPolishStemFilterFactory`
//! (`stempelPolishStem`) and [`register_factories`].

use lucene_analysis::factory::args::{self, JavaArgs};
use lucene_analysis::factory::spi;
use lucene_analysis::factory::{
    AnalysisFactory, FactoryBase, FactoryClass, FactoryError, TokenFilterFactory,
};
use lucene_analysis::{AnalysisError, TokenStream};

use crate::filter::StempelFilter;
use crate::stemmer::{default_table, StempelStemmer};

/// `StempelPolishStemFilterFactory`: no arguments; a `StempelFilter` over
/// the default Polish table.
pub struct StempelPolishStemFilterFactory {
    base: FactoryBase,
}

impl FactoryClass for StempelPolishStemFilterFactory {
    const NAME: &'static str = "stempelPolishStem";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.stempel.StempelPolishStemFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        args::reject_unknown(args)?;
        Ok(StempelPolishStemFilterFactory { base })
    }
}

impl AnalysisFactory for StempelPolishStemFilterFactory {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
}

impl TokenFilterFactory for StempelPolishStemFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(StempelFilter::new(
            input,
            StempelStemmer::new(default_table()),
        )))
    }
}

/// Registers [`StempelPolishStemFilterFactory`] with `lucene-analysis`' SPI
/// registry (idempotent).
pub fn register_factories() -> Result<(), FactoryError> {
    spi::register_token_filter(spi::token_filter_entry::<StempelPolishStemFilterFactory>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_and_registration() {
        let mut f = StempelPolishStemFilterFactory::from_args(&mut JavaArgs::new()).unwrap();
        assert!(f.base_mut().class_name().ends_with("Factory"));
        assert!(f
            .base()
            .class_name()
            .ends_with("StempelPolishStemFilterFactory"));
        let e = StempelPolishStemFilterFactory::from_args(&mut JavaArgs::from_pairs(&[("x", "1")]));
        assert_eq!(e.err().unwrap().message, "Unknown parameters: {x=1}");
        let input: Box<dyn TokenStream> = Box::new(lucene_analysis::KeywordTokenizer::new());
        assert!(f.create(input).is_ok());
        register_factories().unwrap();
        assert!(spi::lookup_token_filter("stempelpolishstem").is_ok());
    }
}
