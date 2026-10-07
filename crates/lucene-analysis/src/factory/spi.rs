//! `AnalysisSPILoader`: the factories by SPI name, and the classes
//! configuration can name.
//!
//! Java's `ServiceLoader` finds the factories its classpath registers in
//! `META-INF/services` (analysis-common's `module-info.java` `provides`
//! lists, plus lucene-core's `StandardTokenizerFactory`); a name is looked up
//! lowercased (`Locale.ROOT`), so case-insensitively, and the first class
//! registered under a name wins. The port's registry holds the same
//! factories in the same order; [`register_token_filter`] (and its two
//! siblings) is Java's `reload` with another jar on the classpath -- how
//! `lucene-search` adds `Word2VecSynonymFilterFactory`, whose filter lives
//! there. Differs: `availableTokenizers()` & co. are a `Set.copyOf` in
//! Java, whose order changes from JVM to JVM; the port lists names in
//! registration order, and so does the "does not exist" message.

use std::sync::{LazyLock, RwLock};

use super::args::JavaArgs;
use super::char_filters::*;
use super::conditional::ProtectedTermFilterFactory;
use super::filters_lang::*;
use super::filters_misc::*;
use super::filters_resource::*;
use super::tokenizers::*;
use super::{
    CharFilterFactory, FactoryClass, FactoryError, JavaException, TokenFilterFactory,
    TokenizerFactory,
};
use crate::Analyzer;

/// A `Map` constructor of a tokenizer factory.
pub type TokenizerCtor = fn(&mut JavaArgs) -> Result<Box<dyn TokenizerFactory>, FactoryError>;
/// A `Map` constructor of a token filter factory.
pub type TokenFilterCtor = fn(&mut JavaArgs) -> Result<Box<dyn TokenFilterFactory>, FactoryError>;
/// A `Map` constructor of a char filter factory.
pub type CharFilterCtor = fn(&mut JavaArgs) -> Result<Box<dyn CharFilterFactory>, FactoryError>;

/// One registered factory class.
#[derive(Clone, Copy)]
pub struct SpiEntry<C> {
    /// `NAME`.
    pub name: &'static str,
    /// The Java class name.
    pub class_name: &'static str,
    /// The `Map` constructor.
    pub ctor: C,
    /// Whether the class is a `ConditionalTokenFilterFactory`.
    pub conditional: bool,
}

fn tokenizer_ctor<T: FactoryClass + TokenizerFactory + 'static>(
    args: &mut JavaArgs,
) -> Result<Box<dyn TokenizerFactory>, FactoryError> {
    Ok(Box::new(T::from_args(args)?))
}

fn token_filter_ctor<T: FactoryClass + TokenFilterFactory + 'static>(
    args: &mut JavaArgs,
) -> Result<Box<dyn TokenFilterFactory>, FactoryError> {
    Ok(Box::new(T::from_args(args)?))
}

fn char_filter_ctor<T: FactoryClass + CharFilterFactory + 'static>(
    args: &mut JavaArgs,
) -> Result<Box<dyn CharFilterFactory>, FactoryError> {
    Ok(Box::new(T::from_args(args)?))
}

/// The registry entry of a tokenizer factory class.
pub fn tokenizer_entry<T: FactoryClass + TokenizerFactory + 'static>() -> SpiEntry<TokenizerCtor> {
    SpiEntry {
        name: T::NAME,
        class_name: T::CLASS_NAME,
        ctor: tokenizer_ctor::<T>,
        conditional: false,
    }
}

/// The registry entry of a token filter factory class.
pub fn token_filter_entry<T: FactoryClass + TokenFilterFactory + 'static>(
) -> SpiEntry<TokenFilterCtor> {
    SpiEntry {
        name: T::NAME,
        class_name: T::CLASS_NAME,
        ctor: token_filter_ctor::<T>,
        conditional: false,
    }
}

/// The registry entry of a char filter factory class.
pub fn char_filter_entry<T: FactoryClass + CharFilterFactory + 'static>() -> SpiEntry<CharFilterCtor>
{
    SpiEntry {
        name: T::NAME,
        class_name: T::CLASS_NAME,
        ctor: char_filter_ctor::<T>,
        conditional: false,
    }
}

struct Registry {
    tokenizers: Vec<SpiEntry<TokenizerCtor>>,
    token_filters: Vec<SpiEntry<TokenFilterCtor>>,
    char_filters: Vec<SpiEntry<CharFilterCtor>>,
}

/// `SERVICE_NAME_PATTERN`: `^[a-zA-Z][a-zA-Z0-9_]+$`.
fn is_valid_name(name: &str) -> bool {
    let b = name.as_bytes();
    b.len() >= 2
        && b[0].is_ascii_alphabetic()
        && b[1..]
            .iter()
            .all(|c| c.is_ascii_alphanumeric() || *c == b'_')
}

/// `reload`: adds the entries whose lowercased name is not taken yet.
fn add<C>(list: &mut Vec<SpiEntry<C>>, entry: SpiEntry<C>) -> Result<(), FactoryError> {
    if !is_valid_name(entry.name) {
        return Err(FactoryError::illegal_argument(format!(
            "The name {} for {} is invalid: Allowed characters are (English) alphabet, digits, and underscore. It should be started with an alphabet.",
            entry.name, entry.class_name
        )));
    }
    let lower = entry.name.to_ascii_lowercase();
    if !list.iter().any(|e| e.name.to_ascii_lowercase() == lower) {
        list.push(entry);
    }
    Ok(())
}

static REGISTRY: LazyLock<RwLock<Registry>> = LazyLock::new(|| {
    let char_filters = vec![
        char_filter_entry::<HTMLStripCharFilterFactory>(),
        char_filter_entry::<MappingCharFilterFactory>(),
        char_filter_entry::<CJKWidthCharFilterFactory>(),
        char_filter_entry::<PersianCharFilterFactory>(),
        char_filter_entry::<PatternReplaceCharFilterFactory>(),
    ];
    let mut token_filters = vec![
        token_filter_entry::<ApostropheFilterFactory>(),
        token_filter_entry::<ArabicNormalizationFilterFactory>(),
        token_filter_entry::<ArabicStemFilterFactory>(),
        token_filter_entry::<BulgarianStemFilterFactory>(),
        token_filter_entry::<DelimitedBoostTokenFilterFactory>(),
        token_filter_entry::<BengaliNormalizationFilterFactory>(),
        token_filter_entry::<BengaliStemFilterFactory>(),
        token_filter_entry::<BrazilianStemFilterFactory>(),
        token_filter_entry::<CJKBigramFilterFactory>(),
        token_filter_entry::<CJKWidthFilterFactory>(),
        token_filter_entry::<SoraniNormalizationFilterFactory>(),
        token_filter_entry::<SoraniStemFilterFactory>(),
        token_filter_entry::<ClassicFilterFactory>(),
        token_filter_entry::<CommonGramsFilterFactory>(),
        token_filter_entry::<CommonGramsQueryFilterFactory>(),
        token_filter_entry::<DictionaryCompoundWordTokenFilterFactory>(),
        token_filter_entry::<HyphenationCompoundWordTokenFilterFactory>(),
        token_filter_entry::<DecimalDigitFilterFactory>(),
        token_filter_entry::<LowerCaseFilterFactory>(),
        token_filter_entry::<StopFilterFactory>(),
        token_filter_entry::<TypeTokenFilterFactory>(),
        token_filter_entry::<UpperCaseFilterFactory>(),
        token_filter_entry::<CzechStemFilterFactory>(),
        token_filter_entry::<GermanLightStemFilterFactory>(),
        token_filter_entry::<GermanMinimalStemFilterFactory>(),
        token_filter_entry::<GermanNormalizationFilterFactory>(),
        token_filter_entry::<GermanStemFilterFactory>(),
        token_filter_entry::<GreekLowerCaseFilterFactory>(),
        token_filter_entry::<GreekStemFilterFactory>(),
        token_filter_entry::<EnglishMinimalStemFilterFactory>(),
        token_filter_entry::<EnglishPossessiveFilterFactory>(),
        token_filter_entry::<KStemFilterFactory>(),
        token_filter_entry::<PorterStemFilterFactory>(),
        token_filter_entry::<SpanishLightStemFilterFactory>(),
        token_filter_entry::<SpanishMinimalStemFilterFactory>(),
        token_filter_entry::<SpanishPluralStemFilterFactory>(),
        token_filter_entry::<PersianNormalizationFilterFactory>(),
        token_filter_entry::<PersianStemFilterFactory>(),
        token_filter_entry::<FinnishLightStemFilterFactory>(),
        token_filter_entry::<FrenchLightStemFilterFactory>(),
        token_filter_entry::<FrenchMinimalStemFilterFactory>(),
        token_filter_entry::<IrishLowerCaseFilterFactory>(),
        token_filter_entry::<GalicianMinimalStemFilterFactory>(),
        token_filter_entry::<GalicianStemFilterFactory>(),
        token_filter_entry::<HindiNormalizationFilterFactory>(),
        token_filter_entry::<HindiStemFilterFactory>(),
        token_filter_entry::<HungarianLightStemFilterFactory>(),
        token_filter_entry::<HunspellStemFilterFactory>(),
        token_filter_entry::<IndonesianStemFilterFactory>(),
        token_filter_entry::<IndicNormalizationFilterFactory>(),
        token_filter_entry::<ItalianLightStemFilterFactory>(),
        token_filter_entry::<LatvianStemFilterFactory>(),
        token_filter_entry::<MinHashFilterFactory>(),
        token_filter_entry::<ASCIIFoldingFilterFactory>(),
        token_filter_entry::<CapitalizationFilterFactory>(),
        token_filter_entry::<CodepointCountFilterFactory>(),
        token_filter_entry::<ConcatenateGraphFilterFactory>(),
        token_filter_entry::<DateRecognizerFilterFactory>(),
        token_filter_entry::<DelimitedTermFrequencyTokenFilterFactory>(),
        token_filter_entry::<DropIfFlaggedFilterFactory>(),
        token_filter_entry::<FingerprintFilterFactory>(),
        token_filter_entry::<FixBrokenOffsetsFilterFactory>(),
        token_filter_entry::<HyphenatedWordsFilterFactory>(),
        token_filter_entry::<KeepWordFilterFactory>(),
        token_filter_entry::<KeywordMarkerFilterFactory>(),
        token_filter_entry::<KeywordRepeatFilterFactory>(),
        token_filter_entry::<LengthFilterFactory>(),
        token_filter_entry::<LimitTokenCountFilterFactory>(),
        token_filter_entry::<LimitTokenOffsetFilterFactory>(),
        token_filter_entry::<LimitTokenPositionFilterFactory>(),
        token_filter_entry::<RemoveDuplicatesTokenFilterFactory>(),
        token_filter_entry::<StemmerOverrideFilterFactory>(),
        SpiEntry {
            conditional: true,
            ..token_filter_entry::<ProtectedTermFilterFactory>()
        },
        token_filter_entry::<TrimFilterFactory>(),
        token_filter_entry::<TruncateTokenFilterFactory>(),
        token_filter_entry::<TypeAsSynonymFilterFactory>(),
        token_filter_entry::<WordDelimiterFilterFactory>(),
        token_filter_entry::<WordDelimiterGraphFilterFactory>(),
        token_filter_entry::<ScandinavianFoldingFilterFactory>(),
        token_filter_entry::<ScandinavianNormalizationFilterFactory>(),
        token_filter_entry::<EdgeNGramFilterFactory>(),
        token_filter_entry::<NGramFilterFactory>(),
        token_filter_entry::<NorwegianLightStemFilterFactory>(),
        token_filter_entry::<NorwegianMinimalStemFilterFactory>(),
        token_filter_entry::<NorwegianNormalizationFilterFactory>(),
        token_filter_entry::<PatternReplaceFilterFactory>(),
        token_filter_entry::<PatternCaptureGroupFilterFactory>(),
        token_filter_entry::<PatternTypingFilterFactory>(),
        token_filter_entry::<DelimitedPayloadTokenFilterFactory>(),
        token_filter_entry::<NumericPayloadTokenFilterFactory>(),
        token_filter_entry::<TokenOffsetPayloadTokenFilterFactory>(),
        token_filter_entry::<TypeAsPayloadTokenFilterFactory>(),
        token_filter_entry::<PortugueseLightStemFilterFactory>(),
        token_filter_entry::<PortugueseMinimalStemFilterFactory>(),
        token_filter_entry::<PortugueseStemFilterFactory>(),
        token_filter_entry::<ReverseStringFilterFactory>(),
        token_filter_entry::<RomanianNormalizationFilterFactory>(),
        token_filter_entry::<RussianLightStemFilterFactory>(),
        token_filter_entry::<ShingleFilterFactory>(),
        token_filter_entry::<FixedShingleFilterFactory>(),
        token_filter_entry::<SnowballPorterFilterFactory>(),
        token_filter_entry::<SerbianNormalizationFilterFactory>(),
        token_filter_entry::<SwedishLightStemFilterFactory>(),
        token_filter_entry::<SwedishMinimalStemFilterFactory>(),
        token_filter_entry::<SynonymFilterFactory>(),
        token_filter_entry::<SynonymGraphFilterFactory>(),
    ];
    // Word2VecSynonymFilterFactory comes next in Java's list: it is
    // registered by lucene-search (see the module docs).
    token_filters.extend([
        token_filter_entry::<FlattenGraphFilterFactory>(),
        token_filter_entry::<TeluguNormalizationFilterFactory>(),
        token_filter_entry::<TeluguStemFilterFactory>(),
        token_filter_entry::<TurkishLowerCaseFilterFactory>(),
        token_filter_entry::<ElisionFilterFactory>(),
    ]);
    let tokenizers = vec![
        tokenizer_entry::<StandardTokenizerFactory>(),
        tokenizer_entry::<ClassicTokenizerFactory>(),
        tokenizer_entry::<KeywordTokenizerFactory>(),
        tokenizer_entry::<LetterTokenizerFactory>(),
        tokenizer_entry::<WhitespaceTokenizerFactory>(),
        tokenizer_entry::<UAX29URLEmailTokenizerFactory>(),
        tokenizer_entry::<EdgeNGramTokenizerFactory>(),
        tokenizer_entry::<NGramTokenizerFactory>(),
        tokenizer_entry::<PathHierarchyTokenizerFactory>(),
        tokenizer_entry::<PatternTokenizerFactory>(),
        tokenizer_entry::<SimplePatternSplitTokenizerFactory>(),
        tokenizer_entry::<SimplePatternTokenizerFactory>(),
        tokenizer_entry::<ThaiTokenizerFactory>(),
        tokenizer_entry::<WikipediaTokenizerFactory>(),
    ];
    RwLock::new(Registry {
        tokenizers,
        token_filters,
        char_filters,
    })
});

fn read() -> std::sync::RwLockReadGuard<'static, Registry> {
    REGISTRY.read().unwrap_or_else(|e| e.into_inner())
}

fn write() -> std::sync::RwLockWriteGuard<'static, Registry> {
    REGISTRY.write().unwrap_or_else(|e| e.into_inner())
}

fn lookup<C: Copy>(
    list: &[SpiEntry<C>],
    kind: &str,
    name: &str,
) -> Result<SpiEntry<C>, FactoryError> {
    // name.toLowerCase(Locale.ROOT): registered names are ASCII.
    let lower: String = name.chars().map(crate::simple_to_lowercase).collect();
    list.iter()
        .find(|e| e.name.to_ascii_lowercase() == lower)
        .copied()
        .ok_or_else(|| {
            let names: Vec<&str> = list.iter().map(|e| e.name).collect();
            FactoryError::illegal_argument(format!(
                "A SPI class of type org.apache.lucene.analysis.{kind} with name '{name}' does not exist. \
                 You need to add the corresponding JAR file supporting this SPI to your classpath. \
                 The current classpath supports the following names: [{}]",
                names.join(", ")
            ))
        })
}

/// `TokenizerFactory.lookupClass(name)`.
pub fn lookup_tokenizer(name: &str) -> Result<SpiEntry<TokenizerCtor>, FactoryError> {
    lookup(&read().tokenizers, "TokenizerFactory", name)
}

/// `TokenFilterFactory.lookupClass(name)`.
pub fn lookup_token_filter(name: &str) -> Result<SpiEntry<TokenFilterCtor>, FactoryError> {
    lookup(&read().token_filters, "TokenFilterFactory", name)
}

/// `CharFilterFactory.lookupClass(name)`.
pub fn lookup_char_filter(name: &str) -> Result<SpiEntry<CharFilterCtor>, FactoryError> {
    lookup(&read().char_filters, "CharFilterFactory", name)
}

/// `TokenizerFactory.forName(name, args)`.
pub fn tokenizer_for_name(
    name: &str,
    args: &mut JavaArgs,
) -> Result<Box<dyn TokenizerFactory>, FactoryError> {
    (lookup_tokenizer(name)?.ctor)(args)
}

/// `TokenFilterFactory.forName(name, args)`.
pub fn token_filter_for_name(
    name: &str,
    args: &mut JavaArgs,
) -> Result<Box<dyn TokenFilterFactory>, FactoryError> {
    (lookup_token_filter(name)?.ctor)(args)
}

/// `CharFilterFactory.forName(name, args)`.
pub fn char_filter_for_name(
    name: &str,
    args: &mut JavaArgs,
) -> Result<Box<dyn CharFilterFactory>, FactoryError> {
    (lookup_char_filter(name)?.ctor)(args)
}

/// `TokenizerFactory.availableTokenizers()`.
pub fn available_tokenizers() -> Vec<&'static str> {
    read().tokenizers.iter().map(|e| e.name).collect()
}

/// `TokenFilterFactory.availableTokenFilters()`.
pub fn available_token_filters() -> Vec<&'static str> {
    read().token_filters.iter().map(|e| e.name).collect()
}

/// `CharFilterFactory.availableCharFilters()`.
pub fn available_char_filters() -> Vec<&'static str> {
    read().char_filters.iter().map(|e| e.name).collect()
}

/// `TokenizerFactory.reloadTokenizers`: registers a factory class from
/// another crate (ignored when its name is taken).
pub fn register_tokenizer(entry: SpiEntry<TokenizerCtor>) -> Result<(), FactoryError> {
    add(&mut write().tokenizers, entry)
}

/// `TokenFilterFactory.reloadTokenFilters`.
pub fn register_token_filter(entry: SpiEntry<TokenFilterCtor>) -> Result<(), FactoryError> {
    add(&mut write().token_filters, entry)
}

/// `CharFilterFactory.reloadCharFilters`.
pub fn register_char_filter(entry: SpiEntry<CharFilterCtor>) -> Result<(), FactoryError> {
    add(&mut write().char_filters, entry)
}

/// `findSPIName(Class)`: the SPI name of a registered class.
pub fn find_spi_name(class_name: &str) -> Option<&'static str> {
    let r = read();
    r.tokenizers
        .iter()
        .map(|e| (e.name, e.class_name))
        .chain(r.token_filters.iter().map(|e| (e.name, e.class_name)))
        .chain(r.char_filters.iter().map(|e| (e.name, e.class_name)))
        .find(|(_, c)| *c == class_name)
        .map(|(n, _)| n)
}

/// `ResourceLoader.findClass(name, TokenizerFactory.class)`: a registered
/// tokenizer factory by class name.
pub fn tokenizer_class(class_name: &str) -> Option<TokenizerCtor> {
    read()
        .tokenizers
        .iter()
        .find(|e| e.class_name == class_name)
        .map(|e| e.ctor)
}

/// `ResourceLoader.findClass(name, TokenFilterFactory.class)`.
pub fn token_filter_class(class_name: &str) -> Option<SpiEntry<TokenFilterCtor>> {
    read()
        .token_filters
        .iter()
        .find(|e| e.class_name == class_name)
        .copied()
}

/// `ResourceLoader.findClass(name, CharFilterFactory.class)`.
pub fn char_filter_class(class_name: &str) -> Option<CharFilterCtor> {
    read()
        .char_filters
        .iter()
        .find(|e| e.class_name == class_name)
        .map(|e| e.ctor)
}

/// The `RuntimeException` `ClasspathResourceLoader.findClass` throws for a
/// class it cannot load.
pub fn cannot_load_class(class_name: &str) -> FactoryError {
    FactoryError::new(
        JavaException::Runtime,
        format!("Cannot load class: {class_name}"),
    )
}

/// The analyzers a `SynonymFilterFactory`'s `analyzer` may name: the no-arg
/// constructors of lucene-core's and analysis-common's core analyzers.
pub fn new_analyzer(class_name: &str) -> Result<Analyzer, FactoryError> {
    use crate::core_analysis::{SimpleAnalyzer, UnicodeWhitespaceAnalyzer, WhitespaceAnalyzer};
    Ok(match class_name {
        "org.apache.lucene.analysis.standard.StandardAnalyzer" => {
            Analyzer::new(crate::StandardAnalyzer::default())
        }
        "org.apache.lucene.analysis.core.WhitespaceAnalyzer" => {
            Analyzer::new(WhitespaceAnalyzer::default())
        }
        "org.apache.lucene.analysis.core.SimpleAnalyzer" => Analyzer::new(SimpleAnalyzer),
        "org.apache.lucene.analysis.core.UnicodeWhitespaceAnalyzer" => {
            Analyzer::new(UnicodeWhitespaceAnalyzer)
        }
        "org.apache.lucene.analysis.core.KeywordAnalyzer" => Analyzer::keyword(),
        _ => return Err(cannot_load_class(class_name)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_counts_and_lookup() {
        assert_eq!(available_tokenizers().len(), 14);
        assert_eq!(available_char_filters().len(), 5);
        assert_eq!(available_token_filters().len(), 111);
        let f = token_filter_for_name("LOWERCASE", &mut JavaArgs::new()).unwrap();
        assert_eq!(
            f.base().class_name(),
            "org.apache.lucene.analysis.core.LowerCaseFilterFactory"
        );
        assert!(lookup_token_filter("protectedterm").unwrap().conditional);
        let e = lookup_tokenizer("nope").err().unwrap();
        assert!(e.message.starts_with(
            "A SPI class of type org.apache.lucene.analysis.TokenizerFactory with name 'nope' does not exist."
        ));
        assert!(lookup_char_filter("x").is_err());
        assert!(char_filter_for_name("htmlstrip", &mut JavaArgs::new()).is_ok());
        assert!(tokenizer_for_name("Whitespace", &mut JavaArgs::new()).is_ok());
        assert_eq!(
            find_spi_name("org.apache.lucene.analysis.core.KeywordTokenizerFactory"),
            Some("keyword")
        );
        assert_eq!(find_spi_name("x"), None);
        assert!(
            tokenizer_class("org.apache.lucene.analysis.core.KeywordTokenizerFactory").is_some()
        );
        assert!(token_filter_class("org.apache.lucene.analysis.core.StopFilterFactory").is_some());
        assert!(
            char_filter_class("org.apache.lucene.analysis.fa.PersianCharFilterFactory").is_some()
        );
        assert!(char_filter_class("x").is_none());
    }

    #[test]
    fn registration_keeps_the_first_name() {
        let before = available_char_filters().len();
        register_char_filter(char_filter_entry::<PersianCharFilterFactory>()).unwrap();
        assert_eq!(available_char_filters().len(), before);
        let bad = SpiEntry {
            name: "x",
            ..char_filter_entry::<PersianCharFilterFactory>()
        };
        assert!(register_char_filter(bad).is_err());
        assert!(register_tokenizer(tokenizer_entry::<ThaiTokenizerFactory>()).is_ok());
        assert!(register_token_filter(token_filter_entry::<TrimFilterFactory>()).is_ok());
        assert!(is_valid_name("a_1") && !is_valid_name("1a") && !is_valid_name("a-b"));
    }

    #[test]
    fn analyzers_by_class_name() {
        for c in [
            "org.apache.lucene.analysis.standard.StandardAnalyzer",
            "org.apache.lucene.analysis.core.WhitespaceAnalyzer",
            "org.apache.lucene.analysis.core.SimpleAnalyzer",
            "org.apache.lucene.analysis.core.UnicodeWhitespaceAnalyzer",
            "org.apache.lucene.analysis.core.KeywordAnalyzer",
        ] {
            assert!(new_analyzer(c).is_ok(), "{c}");
        }
        assert_eq!(
            new_analyzer("x.Y").err().unwrap().message,
            "Cannot load class: x.Y"
        );
    }
}
