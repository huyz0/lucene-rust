//! The module's SPI factories -- `ICUNormalizer2CharFilterFactory`
//! (`icuNormalizer2`), `ICUNormalizer2FilterFactory` (`icuNormalizer2`),
//! `ICUFoldingFilterFactory` (`icuFolding`), `ICUTokenizerFactory` (`icu`)
//! -- and [`register_factories`].

use lucene_analysis::factory::args::{self, JavaArgs};
use lucene_analysis::factory::spi;
use std::sync::Arc;

use lucene_analysis::factory::{
    AnalysisFactory, CharFilterFactory, FactoryBase, FactoryClass, FactoryError, JavaException,
    ResourceLoader, TokenFilterFactory, TokenizerFactory,
};
use lucene_analysis::{AnalysisError, CharReader, TokenStream};

use crate::folding_filter::ICUFoldingFilter;
use crate::icu4j::normalizer2::{Mode, Normalizer2};
use crate::icu4j::unicode_set::UnicodeSet;
use crate::normalizer2_char_filter::ICUNormalizer2CharFilter;
use crate::normalizer2_filter::ICUNormalizer2Filter;
use crate::segmentation::{DefaultICUTokenizerConfig, ICUTokenizer};

/// The `filter` argument: a non-empty `UnicodeSet` pattern wraps the
/// normalizer in a `FilteredNormalizer2`.
fn apply_filter(args: &mut JavaArgs, normalizer: Normalizer2) -> Result<Normalizer2, FactoryError> {
    match args::get(args, "filter") {
        Some(filter) => {
            let set = UnicodeSet::from_pattern(&filter)?;
            if set.is_empty() {
                Ok(normalizer)
            } else {
                Ok(Normalizer2::filtered(normalizer, set)?)
            }
        }
        None => Ok(normalizer),
    }
}

/// `form` (default `nfkc_cf`), `mode` (`compose`/`decompose`) and
/// `filter`, as both `icuNormalizer2` factories read them.
fn normalizer_from_args(args: &mut JavaArgs) -> Result<Normalizer2, FactoryError> {
    let form = args::get_or(args, "form", "nfkc_cf");
    let mode = args::get_one_of(
        args,
        "mode",
        &["compose", "decompose"],
        Some("compose"),
        true,
    )?;
    let mode = if mode.as_deref() == Some("compose") {
        Mode::Compose
    } else {
        Mode::Decompose
    };
    let normalizer = Normalizer2::get_instance(&form, mode)?;
    let normalizer = apply_filter(args, normalizer)?;
    args::reject_unknown(args)?;
    Ok(normalizer)
}

macro_rules! base_impl {
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

/// `ICUNormalizer2CharFilterFactory`.
pub struct ICUNormalizer2CharFilterFactory {
    base: FactoryBase,
    normalizer: Normalizer2,
}
base_impl!(ICUNormalizer2CharFilterFactory);

impl FactoryClass for ICUNormalizer2CharFilterFactory {
    const NAME: &'static str = "icuNormalizer2";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.icu.ICUNormalizer2CharFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let normalizer = normalizer_from_args(args)?;
        Ok(ICUNormalizer2CharFilterFactory { base, normalizer })
    }
}

impl CharFilterFactory for ICUNormalizer2CharFilterFactory {
    fn create(&self, input: Box<dyn CharReader>) -> Box<dyn CharReader> {
        Box::new(ICUNormalizer2CharFilter::with_normalizer(
            input,
            self.normalizer.clone(),
        ))
    }

    fn normalize(&self, input: Box<dyn CharReader>) -> Box<dyn CharReader> {
        self.create(input)
    }
}

/// `ICUNormalizer2FilterFactory`.
pub struct ICUNormalizer2FilterFactory {
    base: FactoryBase,
    normalizer: Normalizer2,
}
base_impl!(ICUNormalizer2FilterFactory);

impl FactoryClass for ICUNormalizer2FilterFactory {
    const NAME: &'static str = "icuNormalizer2";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.icu.ICUNormalizer2FilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let normalizer = normalizer_from_args(args)?;
        Ok(ICUNormalizer2FilterFactory { base, normalizer })
    }
}

impl TokenFilterFactory for ICUNormalizer2FilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(ICUNormalizer2Filter::with_normalizer(
            input,
            self.normalizer.clone(),
        )))
    }

    fn normalize(&self, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        Box::new(ICUNormalizer2Filter::with_normalizer(
            input,
            self.normalizer.clone(),
        ))
    }
}

/// `ICUFoldingFilterFactory`.
pub struct ICUFoldingFilterFactory {
    base: FactoryBase,
    normalizer: Normalizer2,
}
base_impl!(ICUFoldingFilterFactory);

impl FactoryClass for ICUFoldingFilterFactory {
    const NAME: &'static str = "icuFolding";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.icu.ICUFoldingFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let normalizer = apply_filter(args, ICUFoldingFilter::normalizer())?;
        args::reject_unknown(args)?;
        Ok(ICUFoldingFilterFactory { base, normalizer })
    }
}

impl TokenFilterFactory for ICUFoldingFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(ICUFoldingFilter::with_normalizer(
            input,
            self.normalizer.clone(),
        )))
    }

    fn normalize(&self, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        Box::new(ICUFoldingFilter::with_normalizer(
            input,
            self.normalizer.clone(),
        ))
    }
}

/// `ICUTokenizerFactory`: `cjkAsWords`, `myanmarAsWords` (both default
/// `true`) and `rulefiles` (`Latn:my.rbbi,...`). **Differs:** `rulefiles`
/// needs ICU's rule compiler (`RuleBasedBreakIterator(String)`), which is
/// not ported: each entry's script is validated as Java validates it, then
/// `inform` fails with `UnsupportedOperationException`.
pub struct ICUTokenizerFactory {
    base: FactoryBase,
    tailored: Vec<(i32, String)>,
    config: Option<Arc<DefaultICUTokenizerConfig>>,
    cjk_as_words: bool,
    myanmar_as_words: bool,
}

impl AnalysisFactory for ICUTokenizerFactory {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
    fn is_resource_loader_aware(&self) -> bool {
        true
    }
    // Java: ICUTokenizerFactory.inform
    fn inform(&mut self, _loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        if let Some((_, path)) = self.tailored.first() {
            return Err(FactoryError::new(
                JavaException::UnsupportedOperation,
                format!(
                    "ICUTokenizerFactory rulefiles ({path}): compiling break rules from source is not ported"
                ),
            ));
        }
        self.config = Some(Arc::new(DefaultICUTokenizerConfig::new(
            self.cjk_as_words,
            self.myanmar_as_words,
        )));
        Ok(())
    }
}

impl FactoryClass for ICUTokenizerFactory {
    const NAME: &'static str = "icu";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.icu.segmentation.ICUTokenizerFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let mut tailored = Vec::new();
        if let Some(rulefiles) = args::get(args, "rulefiles") {
            let p = crate::icu4j::uprops::uprops();
            let sc = p
                .property(crate::icu4j::uprops::SCRIPT)
                .ok_or_else(|| FactoryError::illegal_argument("no Script data"))?;
            for entry in args::split_file_names(Some(&rulefiles)) {
                let Some(colon) = entry.find(':') else {
                    return Err(FactoryError::new(
                        JavaException::StringIndexOutOfBounds,
                        format!("begin 0, end -1, length {}", entry.encode_utf16().count()),
                    ));
                };
                let code = entry[..colon].trim();
                let path = entry[colon..].get(1..).unwrap_or("").trim().to_string();
                let script = p.value_by_alias(sc, code).ok_or_else(|| {
                    FactoryError::new(
                        JavaException::IllegalIcuArgument,
                        format!("Invalid name: {code}"),
                    )
                })?;
                tailored.push((script, path));
            }
        }
        let cjk_as_words = args::get_boolean(args, "cjkAsWords", true);
        let myanmar_as_words = args::get_boolean(args, "myanmarAsWords", true);
        args::reject_unknown(args)?;
        Ok(ICUTokenizerFactory {
            base,
            tailored,
            config: None,
            cjk_as_words,
            myanmar_as_words,
        })
    }
}

impl TokenizerFactory for ICUTokenizerFactory {
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let config = self.config.clone().unwrap_or_else(|| {
            // Java asserts inform() ran; an uninformed factory uses the
            // arguments' configuration.
            Arc::new(DefaultICUTokenizerConfig::new(
                self.cjk_as_words,
                self.myanmar_as_words,
            ))
        });
        Ok(Box::new(ICUTokenizer::with_config(config)))
    }
}

/// Registers the module's factories with `lucene-analysis`' SPI registry
/// (idempotent).
pub fn register_factories() -> Result<(), FactoryError> {
    spi::register_char_filter(spi::char_filter_entry::<ICUNormalizer2CharFilterFactory>())?;
    spi::register_token_filter(spi::token_filter_entry::<ICUNormalizer2FilterFactory>())?;
    spi::register_token_filter(spi::token_filter_entry::<ICUFoldingFilterFactory>())?;
    spi::register_tokenizer(spi::tokenizer_entry::<ICUTokenizerFactory>())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err<T: FactoryClass>(pairs: &[(&str, &str)]) -> FactoryError {
        T::from_args(&mut JavaArgs::from_pairs(pairs))
            .err()
            .unwrap()
    }

    #[test]
    fn tokenizer_factory() {
        struct NoLoader;
        impl ResourceLoader for NoLoader {
            fn open_resource(&self, name: &str) -> Result<Vec<u8>, FactoryError> {
                Err(FactoryError::io(name.to_string()))
            }
        }
        let e = err::<ICUTokenizerFactory>(&[("rulefiles", "Latn")]);
        assert_eq!(e.kind, JavaException::StringIndexOutOfBounds);
        let e = err::<ICUTokenizerFactory>(&[("rulefiles", "Nope:x.rbbi")]);
        assert_eq!(e.kind, JavaException::IllegalIcuArgument);
        let e = err::<ICUTokenizerFactory>(&[("bogus", "1")]);
        assert_eq!(e.kind, JavaException::IllegalArgument);
        let mut f = ICUTokenizerFactory::from_args(&mut JavaArgs::from_pairs(&[(
            "rulefiles",
            "Latn:Latin-break-only-on-whitespace.rbbi",
        )]))
        .unwrap();
        assert!(f.is_resource_loader_aware());
        let e = f.inform(&NoLoader).unwrap_err();
        assert_eq!(e.kind, JavaException::UnsupportedOperation);
        let mut f = ICUTokenizerFactory::from_args(&mut JavaArgs::from_pairs(&[
            ("cjkAsWords", "false"),
            ("myanmarAsWords", "false"),
        ]))
        .unwrap();
        assert!(f.base_mut().class_name().ends_with("ICUTokenizerFactory"));
        assert!(f.create().is_ok());
        f.inform(&NoLoader).unwrap();
        assert!(f.create().is_ok());
    }

    #[test]
    fn arguments() {
        let e = err::<ICUNormalizer2FilterFactory>(&[("form", "bogus")]);
        assert_eq!(e.kind, JavaException::MissingResource);
        let e = err::<ICUNormalizer2FilterFactory>(&[("mode", "Compose")]);
        assert_eq!(e.kind, JavaException::IllegalArgument);
        let e = err::<ICUNormalizer2CharFilterFactory>(&[("x", "1")]);
        assert_eq!(e.message, "Unknown parameters: {x=1}");
        let e = err::<ICUFoldingFilterFactory>(&[("filter", "[a")]);
        assert_eq!(e.kind, JavaException::IllegalArgument);
        let e = err::<ICUFoldingFilterFactory>(&[("y", "1")]);
        assert_eq!(e.kind, JavaException::IllegalArgument);
        let mut f =
            ICUFoldingFilterFactory::from_args(&mut JavaArgs::from_pairs(&[("filter", "[]")]))
                .unwrap();
        assert!(f
            .base_mut()
            .class_name()
            .ends_with("ICUFoldingFilterFactory"));
        let mut c = ICUNormalizer2CharFilterFactory::from_args(&mut JavaArgs::from_pairs(&[
            ("mode", "decompose"),
            ("filter", "[^a]"),
        ]))
        .unwrap();
        assert!(c.base_mut().class_name().ends_with("CharFilterFactory"));
        let mut t = ICUNormalizer2FilterFactory::from_args(&mut JavaArgs::new()).unwrap();
        assert!(t.base_mut().class_name().ends_with("FilterFactory"));
        // create and normalize over each factory.
        use lucene_analysis::{KeywordTokenizer, StrReader, Tokenizer};
        let mut r = c.create(Box::new(StrReader::new("A\u{301}")));
        let out = lucene_analysis::reader::read_to_string(&mut r).unwrap();
        assert_eq!(out, "a\u{301}");
        let mut r = c.normalize(Box::new(StrReader::new("\u{e9}")));
        assert_eq!(
            lucene_analysis::reader::read_to_string(&mut r).unwrap(),
            "e\u{301}"
        );
        let term = |s: Box<dyn TokenStream>| {
            let mut s = s;
            s.reset().unwrap();
            assert!(s.increment_token().unwrap());
            s.attributes().term().to_string()
        };
        let kw = || {
            let mut k = KeywordTokenizer::new();
            k.set_reader(Box::new(StrReader::new("ÅB"))).unwrap();
            Box::new(k) as Box<dyn TokenStream>
        };
        assert_eq!(term(t.create(kw()).unwrap()), "åb");
        assert_eq!(term(t.normalize(kw())), "åb");
        assert_eq!(term(f.create(kw()).unwrap()), "ab");
        assert_eq!(term(f.normalize(kw())), "ab");
        register_factories().unwrap();
        assert!(spi::lookup_char_filter("icunormalizer2").is_ok());
        assert!(spi::lookup_tokenizer("icu").is_ok());
        assert!(spi::lookup_token_filter("icufolding").is_ok());
    }
}
