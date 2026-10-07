//! The token filter factories that take no argument (the language
//! normalizers and stemmers, the core case filters, a few miscellaneous
//! ones), and the language factories with one: `indonesianStem`
//! (`stemDerivational`), `norwegianLightStem`/`norwegianMinimalStem`
//! (`variant`), `serbianNormalization` (`haircut`).
//!
//! An argument-free factory accepts only `luceneMatchVersion`, `class` and
//! `name`; `normalize` is `create` exactly where Java's factory overrides it
//! (the normalizers, not the stemmers).

use super::args::{self, JavaArgs};
use super::{analysis_factory, FactoryBase, FactoryClass, FactoryError, TokenFilterFactory};
use crate::lang::*;
use crate::token_stream::TokenStream;
use crate::AnalysisError;

/// Declares an argument-free token filter factory: its struct, `NAME`,
/// class, the filter it builds, and whether `normalize` builds it too.
macro_rules! simple_filter_factory {
    (
        $(#[$m:meta])* $name:ident, $spi:literal, $class:literal,
        |$input:ident| $make:expr, normalize = $norm:tt
        $(, unknown = $prefix:tt)?
    ) => {
        $(#[$m])*
        pub struct $name {
            base: FactoryBase,
        }
        analysis_factory!($name);

        impl FactoryClass for $name {
            const NAME: &'static str = $spi;
            const CLASS_NAME: &'static str = $class;
            fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
                let base = FactoryBase::new(Self::CLASS_NAME, args)?;
                simple_filter_factory!(@unknown args $(, $prefix)?);
                Ok($name { base })
            }
        }

        impl $name {
            fn build($input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
                Box::new($make)
            }
        }

        impl TokenFilterFactory for $name {
            fn create(
                &self,
                input: Box<dyn TokenStream>,
            ) -> Result<Box<dyn TokenStream>, AnalysisError> {
                Ok(Self::build(input))
            }
            simple_filter_factory!(@normalize $norm);
        }
    };
    (@unknown $args:ident) => { args::reject_unknown($args)?; };
    (@unknown $args:ident, "none") => { let _ = &$args; };
    (@unknown $args:ident, $prefix:literal) => { args::reject_unknown_with($args, $prefix)?; };
    (@normalize true) => {
        fn normalize(&self, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
            Self::build(input)
        }
    };
    (@normalize false) => {};
}

simple_filter_factory!(
    /// `org.apache.lucene.analysis.ar.ArabicNormalizationFilterFactory`.
    ArabicNormalizationFilterFactory, "arabicNormalization",
    "org.apache.lucene.analysis.ar.ArabicNormalizationFilterFactory",
    |input| ar::ArabicNormalizationFilter::new(input), normalize = true
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.ar.ArabicStemFilterFactory`.
    ArabicStemFilterFactory, "arabicStem",
    "org.apache.lucene.analysis.ar.ArabicStemFilterFactory",
    |input| ar::ArabicStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.bg.BulgarianStemFilterFactory`.
    BulgarianStemFilterFactory, "bulgarianStem",
    "org.apache.lucene.analysis.bg.BulgarianStemFilterFactory",
    |input| bg::BulgarianStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.bn.BengaliNormalizationFilterFactory`.
    BengaliNormalizationFilterFactory, "bengaliNormalization",
    "org.apache.lucene.analysis.bn.BengaliNormalizationFilterFactory",
    |input| bn::BengaliNormalizationFilter::new(input), normalize = true
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.bn.BengaliStemFilterFactory`.
    BengaliStemFilterFactory, "bengaliStem",
    "org.apache.lucene.analysis.bn.BengaliStemFilterFactory",
    |input| bn::BengaliStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.br.BrazilianStemFilterFactory`.
    BrazilianStemFilterFactory, "brazilianStem",
    "org.apache.lucene.analysis.br.BrazilianStemFilterFactory",
    |input| br::BrazilianStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.cjk.CJKWidthFilterFactory`.
    CJKWidthFilterFactory, "cjkWidth",
    "org.apache.lucene.analysis.cjk.CJKWidthFilterFactory",
    |input| crate::cjk::CJKWidthFilter::new(input), normalize = true
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.ckb.SoraniNormalizationFilterFactory`.
    SoraniNormalizationFilterFactory, "soraniNormalization",
    "org.apache.lucene.analysis.ckb.SoraniNormalizationFilterFactory",
    |input| ckb::SoraniNormalizationFilter::new(input), normalize = true
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.ckb.SoraniStemFilterFactory`.
    SoraniStemFilterFactory, "soraniStem",
    "org.apache.lucene.analysis.ckb.SoraniStemFilterFactory",
    |input| ckb::SoraniStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.classic.ClassicFilterFactory`.
    ClassicFilterFactory, "classic",
    "org.apache.lucene.analysis.classic.ClassicFilterFactory",
    |input| crate::classic::ClassicFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.core.DecimalDigitFilterFactory`.
    DecimalDigitFilterFactory, "decimalDigit",
    "org.apache.lucene.analysis.core.DecimalDigitFilterFactory",
    |input| crate::core_analysis::DecimalDigitFilter::new(input), normalize = true
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.core.FlattenGraphFilterFactory`.
    FlattenGraphFilterFactory, "flattenGraph",
    "org.apache.lucene.analysis.core.FlattenGraphFilterFactory",
    |input| crate::core_analysis::FlattenGraphFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.core.LowerCaseFilterFactory`.
    LowerCaseFilterFactory, "lowercase",
    "org.apache.lucene.analysis.core.LowerCaseFilterFactory",
    |input| crate::LowerCaseFilter::new(input), normalize = true
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.core.UpperCaseFilterFactory`.
    UpperCaseFilterFactory, "uppercase",
    "org.apache.lucene.analysis.core.UpperCaseFilterFactory",
    |input| crate::core_analysis::UpperCaseFilter::new(input), normalize = true
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.cz.CzechStemFilterFactory`.
    CzechStemFilterFactory, "czechStem",
    "org.apache.lucene.analysis.cz.CzechStemFilterFactory",
    |input| cz::CzechStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.de.GermanLightStemFilterFactory`.
    GermanLightStemFilterFactory, "germanLightStem",
    "org.apache.lucene.analysis.de.GermanLightStemFilterFactory",
    |input| de::GermanLightStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.de.GermanMinimalStemFilterFactory`.
    GermanMinimalStemFilterFactory, "germanMinimalStem",
    "org.apache.lucene.analysis.de.GermanMinimalStemFilterFactory",
    |input| de::GermanMinimalStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.de.GermanNormalizationFilterFactory`.
    GermanNormalizationFilterFactory, "germanNormalization",
    "org.apache.lucene.analysis.de.GermanNormalizationFilterFactory",
    |input| de::GermanNormalizationFilter::new(input), normalize = true
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.de.GermanStemFilterFactory`.
    GermanStemFilterFactory, "germanStem",
    "org.apache.lucene.analysis.de.GermanStemFilterFactory",
    |input| de::GermanStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.el.GreekLowerCaseFilterFactory`.
    GreekLowerCaseFilterFactory, "greekLowercase",
    "org.apache.lucene.analysis.el.GreekLowerCaseFilterFactory",
    |input| el::GreekLowerCaseFilter::new(input), normalize = true
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.el.GreekStemFilterFactory`.
    GreekStemFilterFactory, "greekStem",
    "org.apache.lucene.analysis.el.GreekStemFilterFactory",
    |input| el::GreekStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.en.EnglishMinimalStemFilterFactory`.
    EnglishMinimalStemFilterFactory, "englishMinimalStem",
    "org.apache.lucene.analysis.en.EnglishMinimalStemFilterFactory",
    |input| crate::en::EnglishMinimalStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.en.EnglishPossessiveFilterFactory`.
    EnglishPossessiveFilterFactory, "englishPossessive",
    "org.apache.lucene.analysis.en.EnglishPossessiveFilterFactory",
    |input| crate::en::EnglishPossessiveFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.en.KStemFilterFactory`.
    KStemFilterFactory, "kStem",
    "org.apache.lucene.analysis.en.KStemFilterFactory",
    |input| crate::en::KStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.en.PorterStemFilterFactory`.
    PorterStemFilterFactory, "porterStem",
    "org.apache.lucene.analysis.en.PorterStemFilterFactory",
    |input| crate::en::PorterStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.es.SpanishLightStemFilterFactory`.
    SpanishLightStemFilterFactory, "spanishLightStem",
    "org.apache.lucene.analysis.es.SpanishLightStemFilterFactory",
    |input| es::SpanishLightStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.es.SpanishMinimalStemFilterFactory`.
    SpanishMinimalStemFilterFactory, "spanishMinimalStem",
    "org.apache.lucene.analysis.es.SpanishMinimalStemFilterFactory",
    |input| es::SpanishMinimalStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.es.SpanishPluralStemFilterFactory`.
    SpanishPluralStemFilterFactory, "spanishPluralStem",
    "org.apache.lucene.analysis.es.SpanishPluralStemFilterFactory",
    |input| es::SpanishPluralStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.fa.PersianNormalizationFilterFactory`.
    PersianNormalizationFilterFactory, "persianNormalization",
    "org.apache.lucene.analysis.fa.PersianNormalizationFilterFactory",
    |input| fa::PersianNormalizationFilter::new(input), normalize = true
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.fa.PersianStemFilterFactory`.
    PersianStemFilterFactory, "persianStem",
    "org.apache.lucene.analysis.fa.PersianStemFilterFactory",
    |input| fa::PersianStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.fi.FinnishLightStemFilterFactory`.
    FinnishLightStemFilterFactory, "finnishLightStem",
    "org.apache.lucene.analysis.fi.FinnishLightStemFilterFactory",
    |input| fi::FinnishLightStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.fr.FrenchLightStemFilterFactory`.
    FrenchLightStemFilterFactory, "frenchLightStem",
    "org.apache.lucene.analysis.fr.FrenchLightStemFilterFactory",
    |input| fr::FrenchLightStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.fr.FrenchMinimalStemFilterFactory`.
    FrenchMinimalStemFilterFactory, "frenchMinimalStem",
    "org.apache.lucene.analysis.fr.FrenchMinimalStemFilterFactory",
    |input| fr::FrenchMinimalStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.ga.IrishLowerCaseFilterFactory`.
    IrishLowerCaseFilterFactory, "irishLowercase",
    "org.apache.lucene.analysis.ga.IrishLowerCaseFilterFactory",
    |input| ga::IrishLowerCaseFilter::new(input), normalize = true
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.gl.GalicianMinimalStemFilterFactory`.
    GalicianMinimalStemFilterFactory, "galicianMinimalStem",
    "org.apache.lucene.analysis.gl.GalicianMinimalStemFilterFactory",
    |input| gl::GalicianMinimalStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.gl.GalicianStemFilterFactory`.
    GalicianStemFilterFactory, "galicianStem",
    "org.apache.lucene.analysis.gl.GalicianStemFilterFactory",
    |input| gl::GalicianStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.hi.HindiNormalizationFilterFactory`.
    HindiNormalizationFilterFactory, "hindiNormalization",
    "org.apache.lucene.analysis.hi.HindiNormalizationFilterFactory",
    |input| hi::HindiNormalizationFilter::new(input), normalize = true
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.hi.HindiStemFilterFactory`.
    HindiStemFilterFactory, "hindiStem",
    "org.apache.lucene.analysis.hi.HindiStemFilterFactory",
    |input| hi::HindiStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.hu.HungarianLightStemFilterFactory`.
    HungarianLightStemFilterFactory, "hungarianLightStem",
    "org.apache.lucene.analysis.hu.HungarianLightStemFilterFactory",
    |input| hu::HungarianLightStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.in.IndicNormalizationFilterFactory`.
    IndicNormalizationFilterFactory, "indicNormalization",
    "org.apache.lucene.analysis.in.IndicNormalizationFilterFactory",
    |input| in_::IndicNormalizationFilter::new(input), normalize = true
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.it.ItalianLightStemFilterFactory`.
    ItalianLightStemFilterFactory, "italianLightStem",
    "org.apache.lucene.analysis.it.ItalianLightStemFilterFactory",
    |input| it::ItalianLightStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.lv.LatvianStemFilterFactory`.
    LatvianStemFilterFactory, "latvianStem",
    "org.apache.lucene.analysis.lv.LatvianStemFilterFactory",
    |input| lv::LatvianStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.miscellaneous.HyphenatedWordsFilterFactory`.
    HyphenatedWordsFilterFactory, "hyphenatedWords",
    "org.apache.lucene.analysis.miscellaneous.HyphenatedWordsFilterFactory",
    |input| crate::miscellaneous::HyphenatedWordsFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.miscellaneous.KeywordRepeatFilterFactory`.
    KeywordRepeatFilterFactory, "keywordRepeat",
    "org.apache.lucene.analysis.miscellaneous.KeywordRepeatFilterFactory",
    |input| crate::miscellaneous::KeywordRepeatFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.miscellaneous.RemoveDuplicatesTokenFilterFactory`.
    RemoveDuplicatesTokenFilterFactory, "removeDuplicates",
    "org.apache.lucene.analysis.miscellaneous.RemoveDuplicatesTokenFilterFactory",
    |input| crate::miscellaneous::RemoveDuplicatesTokenFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.miscellaneous.ScandinavianFoldingFilterFactory`.
    ScandinavianFoldingFilterFactory, "scandinavianFolding",
    "org.apache.lucene.analysis.miscellaneous.ScandinavianFoldingFilterFactory",
    |input| crate::miscellaneous::ScandinavianFoldingFilter::new(input), normalize = true
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.miscellaneous.ScandinavianNormalizationFilterFactory`.
    ScandinavianNormalizationFilterFactory, "scandinavianNormalization",
    "org.apache.lucene.analysis.miscellaneous.ScandinavianNormalizationFilterFactory",
    |input| crate::miscellaneous::ScandinavianNormalizationFilter::new(input), normalize = true
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.miscellaneous.TrimFilterFactory`.
    TrimFilterFactory, "trim",
    "org.apache.lucene.analysis.miscellaneous.TrimFilterFactory",
    |input| crate::miscellaneous::TrimFilter::new(input), normalize = true
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.miscellaneous.FixBrokenOffsetsFilterFactory`
    /// (Java does not check for unknown arguments).
    FixBrokenOffsetsFilterFactory, "fixBrokenOffsets",
    "org.apache.lucene.analysis.miscellaneous.FixBrokenOffsetsFilterFactory",
    |input| crate::miscellaneous::FixBrokenOffsetsFilter::new(input), normalize = false,
    unknown = "none"
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.no.NorwegianNormalizationFilterFactory`.
    NorwegianNormalizationFilterFactory, "norwegianNormalization",
    "org.apache.lucene.analysis.no.NorwegianNormalizationFilterFactory",
    |input| no::NorwegianNormalizationFilter::new(input), normalize = true
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.payloads.TokenOffsetPayloadTokenFilterFactory`.
    TokenOffsetPayloadTokenFilterFactory, "tokenOffsetPayload",
    "org.apache.lucene.analysis.payloads.TokenOffsetPayloadTokenFilterFactory",
    |input| crate::payloads::TokenOffsetPayloadTokenFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.payloads.TypeAsPayloadTokenFilterFactory`.
    TypeAsPayloadTokenFilterFactory, "typeAsPayload",
    "org.apache.lucene.analysis.payloads.TypeAsPayloadTokenFilterFactory",
    |input| crate::payloads::TypeAsPayloadTokenFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.pt.PortugueseLightStemFilterFactory`.
    PortugueseLightStemFilterFactory, "portugueseLightStem",
    "org.apache.lucene.analysis.pt.PortugueseLightStemFilterFactory",
    |input| pt::PortugueseLightStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.pt.PortugueseMinimalStemFilterFactory`.
    PortugueseMinimalStemFilterFactory, "portugueseMinimalStem",
    "org.apache.lucene.analysis.pt.PortugueseMinimalStemFilterFactory",
    |input| pt::PortugueseMinimalStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.pt.PortugueseStemFilterFactory`.
    PortugueseStemFilterFactory, "portugueseStem",
    "org.apache.lucene.analysis.pt.PortugueseStemFilterFactory",
    |input| pt::PortugueseStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.reverse.ReverseStringFilterFactory`.
    ReverseStringFilterFactory, "reverseString",
    "org.apache.lucene.analysis.reverse.ReverseStringFilterFactory",
    |input| crate::reverse::ReverseStringFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.ro.RomanianNormalizationFilterFactory`.
    RomanianNormalizationFilterFactory, "romanianNormalization",
    "org.apache.lucene.analysis.ro.RomanianNormalizationFilterFactory",
    |input| ro::RomanianNormalizationFilter::new(input), normalize = true
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.ru.RussianLightStemFilterFactory`.
    RussianLightStemFilterFactory, "russianLightStem",
    "org.apache.lucene.analysis.ru.RussianLightStemFilterFactory",
    |input| ru::RussianLightStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.sv.SwedishLightStemFilterFactory`.
    SwedishLightStemFilterFactory, "swedishLightStem",
    "org.apache.lucene.analysis.sv.SwedishLightStemFilterFactory",
    |input| sv::SwedishLightStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.sv.SwedishMinimalStemFilterFactory`.
    SwedishMinimalStemFilterFactory, "swedishMinimalStem",
    "org.apache.lucene.analysis.sv.SwedishMinimalStemFilterFactory",
    |input| sv::SwedishMinimalStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.te.TeluguNormalizationFilterFactory`.
    TeluguNormalizationFilterFactory, "teluguNormalization",
    "org.apache.lucene.analysis.te.TeluguNormalizationFilterFactory",
    |input| te::TeluguNormalizationFilter::new(input), normalize = true
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.te.TeluguStemFilterFactory`.
    TeluguStemFilterFactory, "teluguStem",
    "org.apache.lucene.analysis.te.TeluguStemFilterFactory",
    |input| te::TeluguStemFilter::new(input), normalize = false
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.tr.ApostropheFilterFactory`.
    ApostropheFilterFactory, "apostrophe",
    "org.apache.lucene.analysis.tr.ApostropheFilterFactory",
    |input| tr::ApostropheFilter::new(input), normalize = false,
    unknown = "Unknown parameter(s): "
);
simple_filter_factory!(
    /// `org.apache.lucene.analysis.tr.TurkishLowerCaseFilterFactory`.
    TurkishLowerCaseFilterFactory, "turkishLowercase",
    "org.apache.lucene.analysis.tr.TurkishLowerCaseFilterFactory",
    |input| tr::TurkishLowerCaseFilter::new(input), normalize = true
);

/// `org.apache.lucene.analysis.id.IndonesianStemFilterFactory` (`indonesianStem`).
pub struct IndonesianStemFilterFactory {
    base: FactoryBase,
    stem_derivational: bool,
}
analysis_factory!(IndonesianStemFilterFactory);

impl FactoryClass for IndonesianStemFilterFactory {
    const NAME: &'static str = "indonesianStem";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.id.IndonesianStemFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let stem_derivational = args::get_boolean(args, "stemDerivational", true);
        args::reject_unknown(args)?;
        Ok(IndonesianStemFilterFactory {
            base,
            stem_derivational,
        })
    }
}

impl TokenFilterFactory for IndonesianStemFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(id::IndonesianStemFilter::with_stemmer(
            input,
            id::IndonesianStemmer {
                stem_derivational: self.stem_derivational,
            },
        )))
    }
}

/// The `variant` argument of the Norwegian stem filter factories.
fn norwegian_variant(args: &mut JavaArgs) -> Result<i32, FactoryError> {
    match args::get(args, "variant").as_deref() {
        None | Some("nb") => Ok(no::BOKMAAL),
        Some("nn") => Ok(no::NYNORSK),
        Some("no") => Ok(no::BOKMAAL | no::NYNORSK),
        Some(v) => Err(FactoryError::illegal_argument(format!(
            "invalid variant: {v}"
        ))),
    }
}

/// `org.apache.lucene.analysis.no.NorwegianLightStemFilterFactory` (`norwegianLightStem`).
pub struct NorwegianLightStemFilterFactory {
    base: FactoryBase,
    flags: i32,
}
analysis_factory!(NorwegianLightStemFilterFactory);

impl FactoryClass for NorwegianLightStemFilterFactory {
    const NAME: &'static str = "norwegianLightStem";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.no.NorwegianLightStemFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let flags = norwegian_variant(args)?;
        args::reject_unknown(args)?;
        Ok(NorwegianLightStemFilterFactory { base, flags })
    }
}

impl TokenFilterFactory for NorwegianLightStemFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(no::NorwegianLightStemFilter::with_stemmer(
            input,
            no::NorwegianLightStemmer::new(self.flags)?,
        )))
    }
}

/// `org.apache.lucene.analysis.no.NorwegianMinimalStemFilterFactory` (`norwegianMinimalStem`).
pub struct NorwegianMinimalStemFilterFactory {
    base: FactoryBase,
    flags: i32,
}
analysis_factory!(NorwegianMinimalStemFilterFactory);

impl FactoryClass for NorwegianMinimalStemFilterFactory {
    const NAME: &'static str = "norwegianMinimalStem";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.no.NorwegianMinimalStemFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let flags = norwegian_variant(args)?;
        args::reject_unknown(args)?;
        Ok(NorwegianMinimalStemFilterFactory { base, flags })
    }
}

impl TokenFilterFactory for NorwegianMinimalStemFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(no::NorwegianMinimalStemFilter::with_stemmer(
            input,
            no::NorwegianMinimalStemmer::new(self.flags)?,
        )))
    }
}

/// `org.apache.lucene.analysis.sr.SerbianNormalizationFilterFactory`
/// (`serbianNormalization`): `haircut` `bald` (the default) or `regular`.
pub struct SerbianNormalizationFilterFactory {
    base: FactoryBase,
    regular: bool,
}
analysis_factory!(SerbianNormalizationFilterFactory);

impl FactoryClass for SerbianNormalizationFilterFactory {
    const NAME: &'static str = "serbianNormalization";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.sr.SerbianNormalizationFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let haircut = args::get_one_of(args, "haircut", &["bald", "regular"], Some("bald"), true)?;
        args::reject_unknown(args)?;
        Ok(SerbianNormalizationFilterFactory {
            base,
            regular: haircut.as_deref() == Some("regular"),
        })
    }
}

impl TokenFilterFactory for SerbianNormalizationFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(if self.regular {
            Box::new(sr::SerbianNormalizationRegularFilter::new(input))
        } else {
            Box::new(sr::SerbianNormalizationFilter::new(input))
        })
    }

    fn normalize(&self, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        if self.regular {
            Box::new(sr::SerbianNormalizationRegularFilter::new(input))
        } else {
            Box::new(sr::SerbianNormalizationFilter::new(input))
        }
    }
}
