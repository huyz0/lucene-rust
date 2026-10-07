//! `org.apache.lucene.analysis.cn.smart.HMMChineseTokenizerFactory`
//! (`hmmChinese`) and [`register_factories`].

use lucene_analysis::factory::args::{self, JavaArgs};
use lucene_analysis::factory::spi;
use lucene_analysis::factory::{
    AnalysisFactory, FactoryBase, FactoryClass, FactoryError, TokenizerFactory,
};
use lucene_analysis::{AnalysisError, TokenStream};

use crate::tokenizer::hmm_chinese_tokenizer;

/// `HMMChineseTokenizerFactory`: no arguments.
pub struct HMMChineseTokenizerFactory {
    base: FactoryBase,
}

impl FactoryClass for HMMChineseTokenizerFactory {
    const NAME: &'static str = "hmmChinese";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.cn.smart.HMMChineseTokenizerFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        args::reject_unknown(args)?;
        Ok(HMMChineseTokenizerFactory { base })
    }
}

impl AnalysisFactory for HMMChineseTokenizerFactory {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
}

impl TokenizerFactory for HMMChineseTokenizerFactory {
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(hmm_chinese_tokenizer()))
    }
}

/// Registers [`HMMChineseTokenizerFactory`] with `lucene-analysis`' SPI
/// registry (idempotent).
pub fn register_factories() -> Result<(), FactoryError> {
    spi::register_tokenizer(spi::tokenizer_entry::<HMMChineseTokenizerFactory>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_and_registration() {
        let mut f = HMMChineseTokenizerFactory::from_args(&mut JavaArgs::new()).unwrap();
        assert!(f
            .base_mut()
            .class_name()
            .ends_with("HMMChineseTokenizerFactory"));
        assert!(f.base().class_name().ends_with("Factory"));
        let e = HMMChineseTokenizerFactory::from_args(&mut JavaArgs::from_pairs(&[("x", "1")]));
        assert_eq!(e.err().unwrap().message, "Unknown parameters: {x=1}");
        assert!(f.create().is_ok());
        register_factories().unwrap();
        assert!(spi::lookup_tokenizer("hmmchinese").is_ok());
    }
}
