//! The module's factories -- `PhoneticFilterFactory` (`phonetic`),
//! `DoubleMetaphoneFilterFactory` (`doubleMetaphone`),
//! `BeiderMorseFilterFactory` (`beiderMorse`) and
//! `DaitchMokotoffSoundexFilterFactory` (`daitchMokotoffSoundex`) -- and
//! [`register_factories`], which adds them to `lucene-analysis`' SPI
//! registry (Java's `module-info.java` `provides`).

use lucene_analysis::factory::args::{self, JavaArgs};
use lucene_analysis::factory::spi;
use lucene_analysis::factory::{
    AnalysisFactory, FactoryBase, FactoryClass, FactoryError, ResourceLoader, TokenFilterFactory,
};
use lucene_analysis::{AnalysisError, TokenStream};

use crate::bm::{NameType, PhoneticEngine, RuleType};
use crate::encoder::{Encoder, LANGUAGE_PACKAGE};
use crate::filters::{
    BeiderMorseFilter, DaitchMokotoffSoundexFilter, DoubleMetaphoneFilter, PhoneticFilter,
};

/// `PhoneticFilterFactory.registry`: the short names (matched upper-cased)
/// and the class each stands for.
const REGISTRY: [(&str, &str); 7] = [
    ("DOUBLEMETAPHONE", "DoubleMetaphone"),
    ("METAPHONE", "Metaphone"),
    ("SOUNDEX", "Soundex"),
    ("REFINEDSOUNDEX", "RefinedSoundex"),
    ("CAVERPHONE", "Caverphone2"),
    ("COLOGNEPHONETIC", "ColognePhonetic"),
    ("NYSIIS", "Nysiis"),
];

/// `org.apache.lucene.analysis.phonetic.PhoneticFilterFactory` (`phonetic`):
/// `encoder` (a registry name, any case, or a class name -- a simple one in
/// `org.apache.commons.codec.language`), `inject` (default true),
/// `maxCodeLength` (only for `Metaphone` and `DoubleMetaphone`).
///
/// Differs: Java's "must be full class name or one of" message lists the
/// registry's keys in `Map.of`'s per-JVM-run order; the port lists them in
/// declaration order.
pub struct PhoneticFilterFactory {
    base: FactoryBase,
    inject: bool,
    name: String,
    max_code_length: Option<i32>,
    encoder: Option<Encoder>,
}

impl PhoneticFilterFactory {
    /// Whether encoded tokens are added beside the originals.
    pub fn inject(&self) -> bool {
        self.inject
    }

    // Java: PhoneticFilterFactory.resolveEncoder
    fn resolve_encoder(name: &str) -> Result<Encoder, FactoryError> {
        let lookup = if name.contains('.') {
            name.to_string()
        } else {
            format!("{LANGUAGE_PACKAGE}{name}")
        };
        Encoder::for_class_name(&lookup).ok_or_else(|| {
            let keys: Vec<&str> = REGISTRY.iter().map(|(k, _)| *k).collect();
            FactoryError::illegal_argument(format!(
                "Error loading encoder '{name}': must be full class name or one of [{}]",
                keys.join(", ")
            ))
        })
    }

    // Java: PhoneticFilterFactory.getEncoder -- a fresh encoder with the
    // maximum code length set.
    fn encoder(&self) -> Result<Encoder, AnalysisError> {
        let mut e = self.encoder.clone().ok_or_else(|| {
            AnalysisError::IllegalState(
                "NullPointerException: PhoneticFilterFactory was not informed".into(),
            )
        })?;
        if let Some(m) = self.max_code_length {
            e.set_max_code_len(m);
        }
        Ok(e)
    }
}

impl FactoryClass for PhoneticFilterFactory {
    const NAME: &'static str = "phonetic";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.phonetic.PhoneticFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let inject = args::get_boolean(args, "inject", true);
        let name = args::require(args, "encoder")?;
        let max_code_length = match args::get(args, "maxCodeLength") {
            Some(v) => Some(args::parse_java_int(&v)?),
            None => None,
        };
        args::reject_unknown(args)?;
        Ok(PhoneticFilterFactory {
            base,
            inject,
            name,
            max_code_length,
            encoder: None,
        })
    }
}

impl AnalysisFactory for PhoneticFilterFactory {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
    fn is_resource_loader_aware(&self) -> bool {
        true
    }
    // Java: PhoneticFilterFactory.inform
    fn inform(&mut self, _loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        let upper = crate::java::string(&crate::java::to_upper(&crate::java::units(&self.name)));
        let encoder = match REGISTRY.iter().find(|(k, _)| *k == upper) {
            Some((_, class)) => Encoder::for_class_name(&format!("{LANGUAGE_PACKAGE}{class}"))
                .expect("every registry class is ported"),
            None => Self::resolve_encoder(&self.name)?,
        };
        if self.max_code_length.is_some() && !encoder.supports_max_code_len() {
            return Err(FactoryError::illegal_argument(format!(
                "Encoder {} / class {} does not support maxCodeLength",
                self.name,
                encoder.class_name()
            )));
        }
        self.encoder = Some(encoder);
        Ok(())
    }
}

impl TokenFilterFactory for PhoneticFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(PhoneticFilter::new(
            input,
            self.encoder()?,
            self.inject,
        )))
    }
}

/// `org.apache.lucene.analysis.phonetic.DoubleMetaphoneFilterFactory`
/// (`doubleMetaphone`): `inject` (default true), `maxCodeLength` (default 4;
/// below 1 fails when the filter is created, as in Java).
pub struct DoubleMetaphoneFilterFactory {
    base: FactoryBase,
    inject: bool,
    max_code_length: i32,
}

/// `DoubleMetaphoneFilterFactory.DEFAULT_MAX_CODE_LENGTH`.
pub const DEFAULT_MAX_CODE_LENGTH: i32 = 4;

impl FactoryClass for DoubleMetaphoneFilterFactory {
    const NAME: &'static str = "doubleMetaphone";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.phonetic.DoubleMetaphoneFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let inject = args::get_boolean(args, "inject", true);
        let max_code_length = args::get_int(args, "maxCodeLength", DEFAULT_MAX_CODE_LENGTH)?;
        args::reject_unknown(args)?;
        Ok(DoubleMetaphoneFilterFactory {
            base,
            inject,
            max_code_length,
        })
    }
}

impl AnalysisFactory for DoubleMetaphoneFilterFactory {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
}

impl TokenFilterFactory for DoubleMetaphoneFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(DoubleMetaphoneFilter::new(
            input,
            self.max_code_length,
            self.inject,
        )?))
    }
}

/// `org.apache.lucene.analysis.phonetic.BeiderMorseFilterFactory`
/// (`beiderMorse`): `nameType` (`GENERIC`), `ruleType` (`APPROX`),
/// `concat` (true), `languageSet` (`auto`: guess per term).
pub struct BeiderMorseFilterFactory {
    base: FactoryBase,
    engine: PhoneticEngine,
    language_set: Option<Vec<String>>,
}

impl FactoryClass for BeiderMorseFilterFactory {
    const NAME: &'static str = "beiderMorse";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.phonetic.BeiderMorseFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let name_type = args::get_or(args, "nameType", "GENERIC");
        let name_type = NameType::value_of(&name_type).ok_or_else(|| {
            FactoryError::illegal_argument(format!(
                "No enum constant org.apache.commons.codec.language.bm.NameType.{name_type}"
            ))
        })?;
        let rule_type = args::get_or(args, "ruleType", "APPROX");
        let rule_type = RuleType::value_of(&rule_type).ok_or_else(|| {
            FactoryError::illegal_argument(format!(
                "No enum constant org.apache.commons.codec.language.bm.RuleType.{rule_type}"
            ))
        })?;
        let concat = args::get_boolean(args, "concat", true);
        let engine = PhoneticEngine::new(name_type, rule_type, concat)
            .map_err(|e| FactoryError::illegal_argument(e.message()))?;
        let language_set = args::get_set(args, "languageSet")
            .filter(|langs| !(langs.len() == 1 && langs.contains("auto")))
            .map(|langs| langs.into_iter().collect());
        args::reject_unknown(args)?;
        Ok(BeiderMorseFilterFactory {
            base,
            engine,
            language_set,
        })
    }
}

impl AnalysisFactory for BeiderMorseFilterFactory {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
}

impl TokenFilterFactory for BeiderMorseFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(BeiderMorseFilter::with_languages(
            input,
            self.engine.clone(),
            self.language_set.as_deref(),
        )))
    }
}

/// `org.apache.lucene.analysis.phonetic.DaitchMokotoffSoundexFilterFactory`
/// (`daitchMokotoffSoundex`): `inject` (default true).
pub struct DaitchMokotoffSoundexFilterFactory {
    base: FactoryBase,
    inject: bool,
}

impl FactoryClass for DaitchMokotoffSoundexFilterFactory {
    const NAME: &'static str = "daitchMokotoffSoundex";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.phonetic.DaitchMokotoffSoundexFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let inject = args::get_boolean(args, "inject", true);
        args::reject_unknown(args)?;
        Ok(DaitchMokotoffSoundexFilterFactory { base, inject })
    }
}

impl AnalysisFactory for DaitchMokotoffSoundexFilterFactory {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
}

impl TokenFilterFactory for DaitchMokotoffSoundexFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(DaitchMokotoffSoundexFilter::new(
            input,
            self.inject,
        )))
    }
}

/// Registers the module's four factories with `lucene-analysis`' SPI
/// registry, in `module-info.java`'s order (idempotent).
pub fn register_factories() -> Result<(), FactoryError> {
    spi::register_token_filter(spi::token_filter_entry::<BeiderMorseFilterFactory>())?;
    spi::register_token_filter(spi::token_filter_entry::<DaitchMokotoffSoundexFilterFactory>())?;
    spi::register_token_filter(spi::token_filter_entry::<DoubleMetaphoneFilterFactory>())?;
    spi::register_token_filter(spi::token_filter_entry::<PhoneticFilterFactory>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::factory::{JavaException, MapResourceLoader};

    fn phonetic(pairs: &[(&str, &str)]) -> Result<PhoneticFilterFactory, FactoryError> {
        let mut f = PhoneticFilterFactory::from_args(&mut JavaArgs::from_pairs(pairs))?;
        f.inform(&MapResourceLoader::new())?;
        Ok(f)
    }

    #[test]
    fn phonetic_arguments() {
        let f = phonetic(&[("encoder", "soundex")]).unwrap();
        assert!(f.inject() && f.is_resource_loader_aware());
        assert!(f.base().class_name().ends_with("PhoneticFilterFactory"));
        assert!(phonetic(&[("encoder", "Caverphone1"), ("inject", "false")]).is_ok());
        assert!(phonetic(&[(
            "encoder",
            "org.apache.commons.codec.language.bm.BeiderMorseEncoder"
        )])
        .is_ok());
        let e = phonetic(&[("encoder", "Nope")]).err().unwrap();
        assert!(e.message.starts_with(
            "Error loading encoder 'Nope': must be full class name or one of [DOUBLEMETAPHONE"
        ));
        let e = phonetic(&[("encoder", "Soundex"), ("maxCodeLength", "3")])
            .err()
            .unwrap();
        assert_eq!(
            e.message,
            "Encoder Soundex / class org.apache.commons.codec.language.Soundex does not support maxCodeLength"
        );
        assert!(phonetic(&[("encoder", "Metaphone"), ("maxCodeLength", "3")]).is_ok());
        assert_eq!(
            phonetic(&[("encoder", "Metaphone"), ("maxCodeLength", "x")])
                .err()
                .unwrap()
                .kind,
            JavaException::NumberFormat
        );
        assert!(phonetic(&[]).is_err());
        assert!(phonetic(&[("encoder", "Soundex"), ("bogus", "1")]).is_err());
        let uninformed =
            PhoneticFilterFactory::from_args(&mut JavaArgs::from_pairs(&[("encoder", "Soundex")]))
                .unwrap();
        let input: Box<dyn TokenStream> = Box::new(lucene_analysis::KeywordTokenizer::new());
        assert!(uninformed.create(input).is_err());
    }

    #[test]
    fn other_factories() {
        let dm = DoubleMetaphoneFilterFactory::from_args(&mut JavaArgs::from_pairs(&[(
            "maxCodeLength",
            "0",
        )]))
        .unwrap();
        assert!(dm
            .base()
            .class_name()
            .ends_with("DoubleMetaphoneFilterFactory"));
        let input: Box<dyn TokenStream> = Box::new(lucene_analysis::KeywordTokenizer::new());
        assert!(dm.create(input).is_err());
        assert!(
            DoubleMetaphoneFilterFactory::from_args(&mut JavaArgs::from_pairs(&[("x", "0")]))
                .is_err()
        );
        let bm = |pairs: &[(&str, &str)]| {
            BeiderMorseFilterFactory::from_args(&mut JavaArgs::from_pairs(pairs))
        };
        assert!(bm(&[]).unwrap().language_set.is_none());
        assert!(bm(&[("languageSet", "auto")])
            .unwrap()
            .language_set
            .is_none());
        assert_eq!(
            bm(&[("languageSet", "polish, german")])
                .unwrap()
                .language_set
                .unwrap()
                .len(),
            2
        );
        assert!(bm(&[("nameType", "generic")])
            .err()
            .unwrap()
            .message
            .ends_with("NameType.generic"));
        assert!(bm(&[("ruleType", "x")])
            .err()
            .unwrap()
            .message
            .ends_with("RuleType.x"));
        assert_eq!(
            bm(&[("ruleType", "RULES")]).err().unwrap().message,
            "ruleType must not be RULES"
        );
        assert!(bm(&[("y", "1")]).is_err());
        let f = bm(&[]).unwrap();
        assert!(f.base().class_name().ends_with("BeiderMorseFilterFactory"));
        let d = DaitchMokotoffSoundexFilterFactory::from_args(&mut JavaArgs::from_pairs(&[(
            "inject", "false",
        )]))
        .unwrap();
        assert!(
            !d.inject
                && d.base()
                    .class_name()
                    .ends_with("DaitchMokotoffSoundexFilterFactory")
        );
        assert!(
            DaitchMokotoffSoundexFilterFactory::from_args(&mut JavaArgs::from_pairs(&[("z", "1")]))
                .is_err()
        );
        register_factories().unwrap();
        for n in [
            "phonetic",
            "doubleMetaphone",
            "beiderMorse",
            "daitchMokotoffSoundex",
        ] {
            assert!(spi::lookup_token_filter(n).is_ok(), "{n}");
        }
    }
}
