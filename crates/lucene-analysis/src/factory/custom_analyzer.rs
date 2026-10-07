//! `org.apache.lucene.analysis.custom.CustomAnalyzer`: an analyzer built
//! from factories -- char filters, a tokenizer, token filters -- by SPI name
//! or by class, with optional position increment and offset gaps.
//!
//! Java's `CustomAnalyzer` *is* an `Analyzer`; the port's derefs to one
//! ([`CustomAnalyzer::analyzer`]) whose definition shares the factories with
//! the accessors. The builder is by value: each step returns it (or the
//! error Java throws there), and `when` hands over to a [`ConditionBuilder`]
//! whose `endwhen` hands it back. Java's `String... params` are a flat
//! key/value list here too (`&[&str]`); the `_args` variants take a
//! [`JavaArgs`] map.

use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;

use lucene_util::version::Version;

use super::args::JavaArgs;
use super::conditional::TermPredicateFactory;
use super::loader::{ClasspathResourceLoader, FilesystemResourceLoader, ResourceLoader};
use super::{
    spi, CharFilterFactory, FactoryClass, FactoryError, JavaException, TokenFilterFactory,
    TokenizerFactory, LUCENE_MATCH_VERSION_PARAM,
};
use crate::reader::CharReader;
use crate::token_stream::TokenStream;
use crate::{AnalysisError, Analyzer, AnalyzerDefinition, TokenStreamComponents};

/// The factories a [`CustomAnalyzer`] runs.
struct Parts {
    char_filters: Vec<Arc<dyn CharFilterFactory>>,
    tokenizer: Arc<dyn TokenizerFactory>,
    token_filters: Vec<Arc<dyn TokenFilterFactory>>,
    pos_inc_gap: Option<i32>,
    offset_gap: Option<i32>,
}

struct Definition(Arc<Parts>);

impl AnalyzerDefinition for Definition {
    // Java: CustomAnalyzer.createComponents
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let mut ts = self.0.tokenizer.create()?;
        for filter in &self.0.token_filters {
            ts = filter.create(ts)?;
        }
        Ok(TokenStreamComponents::new(ts))
    }

    // Java: CustomAnalyzer.normalize
    fn normalize(&self, _field: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        let mut ts = input;
        for filter in &self.0.token_filters {
            ts = filter.normalize(ts);
        }
        ts
    }

    // Java: CustomAnalyzer.initReader
    fn init_reader(&self, _field: &str, reader: Box<dyn CharReader>) -> Box<dyn CharReader> {
        let mut r = reader;
        for cf in &self.0.char_filters {
            r = cf.create(r);
        }
        r
    }

    // Java: CustomAnalyzer.initReaderForNormalization
    fn init_reader_for_normalization(
        &self,
        _field: &str,
        reader: Box<dyn CharReader>,
    ) -> Box<dyn CharReader> {
        let mut r = reader;
        for cf in &self.0.char_filters {
            r = cf.normalize(r);
        }
        r
    }

    fn position_increment_gap(&self, _field: &str) -> i32 {
        self.0.pos_inc_gap.unwrap_or(0)
    }

    fn offset_gap(&self, _field: &str) -> i32 {
        self.0.offset_gap.unwrap_or(1)
    }
}

/// `org.apache.lucene.analysis.custom.CustomAnalyzer`.
pub struct CustomAnalyzer {
    parts: Arc<Parts>,
    analyzer: Analyzer,
}

impl CustomAnalyzer {
    /// `CustomAnalyzer.builder()`: resources from the vendored jar files.
    pub fn builder() -> CustomAnalyzerBuilder {
        Self::builder_with_loader(Box::new(ClasspathResourceLoader))
    }

    /// `CustomAnalyzer.builder(Path configDir)`: resources from a directory,
    /// then the vendored jar files.
    pub fn builder_with_dir(
        config_dir: impl AsRef<Path>,
    ) -> Result<CustomAnalyzerBuilder, FactoryError> {
        Ok(Self::builder_with_loader(Box::new(
            FilesystemResourceLoader::with_classpath(config_dir.as_ref())?,
        )))
    }

    /// `CustomAnalyzer.builder(ResourceLoader)`.
    pub fn builder_with_loader(loader: Box<dyn ResourceLoader>) -> CustomAnalyzerBuilder {
        CustomAnalyzerBuilder {
            loader,
            default_match_version: None,
            char_filters: Vec::new(),
            tokenizer: None,
            token_filters: Vec::new(),
            pos_inc_gap: None,
            offset_gap: None,
            components_added: false,
        }
    }

    /// The analyzer.
    pub fn analyzer(&self) -> &Analyzer {
        &self.analyzer
    }

    /// The analyzer, by value.
    pub fn into_analyzer(self) -> Analyzer {
        self.analyzer
    }

    /// `getCharFilterFactories()`.
    pub fn char_filter_factories(&self) -> &[Arc<dyn CharFilterFactory>] {
        &self.parts.char_filters
    }

    /// `getTokenizerFactory()`.
    pub fn tokenizer_factory(&self) -> &Arc<dyn TokenizerFactory> {
        &self.parts.tokenizer
    }

    /// `getTokenFilterFactories()`.
    pub fn token_filter_factories(&self) -> &[Arc<dyn TokenFilterFactory>] {
        &self.parts.token_filters
    }
}

impl Deref for CustomAnalyzer {
    type Target = Analyzer;
    fn deref(&self) -> &Analyzer {
        &self.analyzer
    }
}

/// `CustomAnalyzer.toString()`: `CustomAnalyzer(charFilter,...,tokenizer,
/// filter,...)`, each factory printed as its class name (Java appends
/// `@identityHashCode`).
impl std::fmt::Display for CustomAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CustomAnalyzer(")?;
        for cf in &self.parts.char_filters {
            write!(f, "{},", cf.base().class_name())?;
        }
        f.write_str(self.parts.tokenizer.base().class_name())?;
        for tf in &self.parts.token_filters {
            write!(f, ",{}", tf.base().class_name())?;
        }
        f.write_str(")")
    }
}

/// `SetOnce.AlreadySetException`.
fn already_set() -> FactoryError {
    FactoryError::new(JavaException::AlreadySet, "The object cannot be set twice!")
}

/// `paramsToMap(String...)`.
fn params_to_map(params: &[&str]) -> Result<JavaArgs, FactoryError> {
    if !params.len().is_multiple_of(2) {
        return Err(FactoryError::illegal_argument(
            "Key-value pairs expected, so the number of params must be even.",
        ));
    }
    let pairs: Vec<(&str, &str)> = params.chunks(2).map(|p| (p[0], p[1])).collect();
    Ok(JavaArgs::from_pairs(&pairs))
}

/// `CustomAnalyzer.Builder`.
pub struct CustomAnalyzerBuilder {
    loader: Box<dyn ResourceLoader>,
    default_match_version: Option<Version>,
    char_filters: Vec<Box<dyn CharFilterFactory>>,
    tokenizer: Option<Box<dyn TokenizerFactory>>,
    token_filters: Vec<Box<dyn TokenFilterFactory>>,
    pos_inc_gap: Option<i32>,
    offset_gap: Option<i32>,
    components_added: bool,
}

impl CustomAnalyzerBuilder {
    /// `withDefaultMatchVersion(Version)`: only before any component.
    pub fn with_default_match_version(mut self, version: Version) -> Result<Self, FactoryError> {
        if self.components_added {
            return Err(FactoryError::new(
                JavaException::IllegalState,
                "You may only set the default match version before adding tokenizers, token filters, or char filters.",
            ));
        }
        if self.default_match_version.is_some() {
            return Err(already_set());
        }
        self.default_match_version = Some(version);
        Ok(self)
    }

    /// `withPositionIncrementGap(int)`.
    pub fn with_position_increment_gap(mut self, gap: i32) -> Result<Self, FactoryError> {
        if gap < 0 {
            return Err(FactoryError::illegal_argument("posIncGap must be >= 0"));
        }
        if self.pos_inc_gap.is_some() {
            return Err(already_set());
        }
        self.pos_inc_gap = Some(gap);
        Ok(self)
    }

    /// `withOffsetGap(int)`.
    pub fn with_offset_gap(mut self, gap: i32) -> Result<Self, FactoryError> {
        if gap < 0 {
            return Err(FactoryError::illegal_argument("offsetGap must be >= 0"));
        }
        if self.offset_gap.is_some() {
            return Err(already_set());
        }
        self.offset_gap = Some(gap);
        Ok(self)
    }

    /// `applyDefaultParams(map)`.
    fn apply_default_params(&self, mut map: JavaArgs) -> JavaArgs {
        if let Some(v) = self.default_match_version {
            map.put_if_absent(LUCENE_MATCH_VERSION_PARAM, &v.to_string());
        }
        map
    }

    fn set_tokenizer(mut self, mut t: Box<dyn TokenizerFactory>) -> Result<Self, FactoryError> {
        if t.is_resource_loader_aware() {
            t.inform(&*self.loader)?;
        }
        if self.tokenizer.is_some() {
            return Err(already_set());
        }
        self.tokenizer = Some(t);
        self.components_added = true;
        Ok(self)
    }

    /// `withTokenizer(String name, String... params)`.
    pub fn with_tokenizer(self, name: &str, params: &[&str]) -> Result<Self, FactoryError> {
        let map = params_to_map(params)?;
        self.with_tokenizer_args(name, map)
    }

    /// `withTokenizer(String name, Map<String, String> params)`.
    pub fn with_tokenizer_args(self, name: &str, params: JavaArgs) -> Result<Self, FactoryError> {
        let mut map = self.apply_default_params(params);
        let t = spi::tokenizer_for_name(name, &mut map)?;
        self.set_tokenizer(t)
    }

    /// `withTokenizer(Class<? extends TokenizerFactory>, String... params)`.
    pub fn with_tokenizer_class<T: FactoryClass + TokenizerFactory + 'static>(
        self,
        params: &[&str],
    ) -> Result<Self, FactoryError> {
        let mut map = self.apply_default_params(params_to_map(params)?);
        let t = T::from_args(&mut map)?;
        self.set_tokenizer(Box::new(t))
    }

    fn push_token_filter(
        mut self,
        mut f: Box<dyn TokenFilterFactory>,
    ) -> Result<Self, FactoryError> {
        if f.is_resource_loader_aware() {
            f.inform(&*self.loader)?;
        }
        self.token_filters.push(f);
        self.components_added = true;
        Ok(self)
    }

    /// `addTokenFilter(String name, String... params)`.
    pub fn add_token_filter(self, name: &str, params: &[&str]) -> Result<Self, FactoryError> {
        let map = params_to_map(params)?;
        self.add_token_filter_args(name, map)
    }

    /// `addTokenFilter(String name, Map<String, String> params)`.
    pub fn add_token_filter_args(self, name: &str, params: JavaArgs) -> Result<Self, FactoryError> {
        let mut map = self.apply_default_params(params);
        let f = spi::token_filter_for_name(name, &mut map)?;
        self.push_token_filter(f)
    }

    /// `addTokenFilter(Class<? extends TokenFilterFactory>, String... params)`.
    pub fn add_token_filter_class<T: FactoryClass + TokenFilterFactory + 'static>(
        self,
        params: &[&str],
    ) -> Result<Self, FactoryError> {
        let mut map = self.apply_default_params(params_to_map(params)?);
        let f = T::from_args(&mut map)?;
        self.push_token_filter(Box::new(f))
    }

    fn push_char_filter(mut self, mut f: Box<dyn CharFilterFactory>) -> Result<Self, FactoryError> {
        if f.is_resource_loader_aware() {
            f.inform(&*self.loader)?;
        }
        self.char_filters.push(f);
        self.components_added = true;
        Ok(self)
    }

    /// `addCharFilter(String name, String... params)`.
    pub fn add_char_filter(self, name: &str, params: &[&str]) -> Result<Self, FactoryError> {
        let map = params_to_map(params)?;
        self.add_char_filter_args(name, map)
    }

    /// `addCharFilter(String name, Map<String, String> params)`.
    pub fn add_char_filter_args(self, name: &str, params: JavaArgs) -> Result<Self, FactoryError> {
        let mut map = self.apply_default_params(params);
        let f = spi::char_filter_for_name(name, &mut map)?;
        self.push_char_filter(f)
    }

    /// `addCharFilter(Class<? extends CharFilterFactory>, String... params)`.
    pub fn add_char_filter_class<T: FactoryClass + CharFilterFactory + 'static>(
        self,
        params: &[&str],
    ) -> Result<Self, FactoryError> {
        let mut map = self.apply_default_params(params_to_map(params)?);
        let f = T::from_args(&mut map)?;
        self.push_char_filter(Box::new(f))
    }

    /// `when(String name, String... params)`: a conditional token filter
    /// factory by name; the filters added until `endwhen` run on the tokens
    /// it selects.
    pub fn when(self, name: &str, params: &[&str]) -> Result<ConditionBuilder, FactoryError> {
        let map = params_to_map(params)?;
        self.when_args(name, map)
    }

    /// `when(String name, Map<String, String> params)`.
    pub fn when_args(self, name: &str, params: JavaArgs) -> Result<ConditionBuilder, FactoryError> {
        let entry = spi::lookup_token_filter(name)?;
        if !entry.conditional {
            return Err(FactoryError::illegal_argument(format!(
                "TokenFilterFactory {name} is not a ConditionalTokenFilterFactory"
            )));
        }
        let mut map = self.apply_default_params(params);
        let factory = (entry.ctor)(&mut map)?;
        Ok(self.when_factory(factory))
    }

    /// `when(ConditionalTokenFilterFactory factory)`: `factory` must be a
    /// conditional factory ([`TokenFilterFactory::as_conditional`]); another
    /// one is refused at `endwhen`.
    pub fn when_factory(self, factory: Box<dyn TokenFilterFactory>) -> ConditionBuilder {
        ConditionBuilder {
            inner_filters: Vec::new(),
            factory,
            parent: self,
        }
    }

    /// `whenTerm(Predicate<CharSequence>)`.
    pub fn when_term(
        self,
        predicate: impl Fn(&str) -> bool + Send + Sync + 'static,
    ) -> ConditionBuilder {
        self.when_factory(Box::new(TermPredicateFactory::new(predicate)))
    }

    /// `build()`.
    pub fn build(self) -> Result<CustomAnalyzer, FactoryError> {
        let tokenizer = self.tokenizer.ok_or_else(|| {
            FactoryError::new(
                JavaException::IllegalState,
                "You have to set at least a tokenizer.",
            )
        })?;
        let parts = Arc::new(Parts {
            char_filters: self.char_filters.into_iter().map(Arc::from).collect(),
            tokenizer: Arc::from(tokenizer),
            token_filters: self.token_filters.into_iter().map(Arc::from).collect(),
            pos_inc_gap: self.pos_inc_gap,
            offset_gap: self.offset_gap,
        });
        let analyzer = Analyzer::new(Definition(Arc::clone(&parts)));
        Ok(CustomAnalyzer { parts, analyzer })
    }
}

/// `CustomAnalyzer.ConditionBuilder`.
pub struct ConditionBuilder {
    inner_filters: Vec<Box<dyn TokenFilterFactory>>,
    factory: Box<dyn TokenFilterFactory>,
    parent: CustomAnalyzerBuilder,
}

impl ConditionBuilder {
    /// `addTokenFilter(String name, String... params)`: not informed until
    /// `endwhen` informs the conditional factory.
    pub fn add_token_filter(self, name: &str, params: &[&str]) -> Result<Self, FactoryError> {
        let map = params_to_map(params)?;
        self.add_token_filter_args(name, map)
    }

    /// `addTokenFilter(String name, Map<String, String> params)`.
    pub fn add_token_filter_args(
        mut self,
        name: &str,
        params: JavaArgs,
    ) -> Result<Self, FactoryError> {
        let mut map = self.parent.apply_default_params(params);
        self.inner_filters
            .push(spi::token_filter_for_name(name, &mut map)?);
        Ok(self)
    }

    /// `addTokenFilter(Class<? extends TokenFilterFactory>, String... params)`.
    pub fn add_token_filter_class<T: FactoryClass + TokenFilterFactory + 'static>(
        mut self,
        params: &[&str],
    ) -> Result<Self, FactoryError> {
        let mut map = self.parent.apply_default_params(params_to_map(params)?);
        self.inner_filters.push(Box::new(T::from_args(&mut map)?));
        Ok(self)
    }

    /// `endwhen()`: hands the inner filters to the conditional factory,
    /// informs it, and adds it to the analyzer.
    pub fn endwhen(mut self) -> Result<CustomAnalyzerBuilder, FactoryError> {
        let class_name = self.factory.base().class_name();
        let Some(conditional) = self.factory.as_conditional() else {
            return Err(FactoryError::illegal_argument(format!(
                "TokenFilterFactory {class_name} is not a ConditionalTokenFilterFactory"
            )));
        };
        conditional.set_inner_filters(self.inner_filters);
        self.parent.push_token_filter(self.factory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factory::{LowerCaseFilterFactory, MapResourceLoader, WhitespaceTokenizerFactory};

    fn terms(a: &Analyzer, text: &str) -> Vec<String> {
        let mut ts = a.token_stream("f", text).unwrap();
        ts.reset().unwrap();
        let mut out = Vec::new();
        while ts.increment_token().unwrap() {
            out.push(ts.attributes().term().to_string());
        }
        ts.end().unwrap();
        ts.close().unwrap();
        out
    }

    #[test]
    fn builds_by_name_and_class() {
        let a = CustomAnalyzer::builder()
            .with_default_match_version(Version::LUCENE_10_0_0)
            .unwrap()
            .add_char_filter("htmlStrip", &[])
            .unwrap()
            .with_tokenizer("whitespace", &["maxTokenLen", "10"])
            .unwrap()
            .add_token_filter("lowercase", &[])
            .unwrap()
            .add_token_filter_class::<crate::factory::ASCIIFoldingFilterFactory>(&[])
            .unwrap()
            .with_position_increment_gap(100)
            .unwrap()
            .with_offset_gap(5)
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(terms(&a, "<b>Héllo</b> WORLD"), vec!["hello", "world"]);
        assert_eq!(a.position_increment_gap_for_field("f"), 100);
        assert_eq!(a.offset_gap_for_field("f"), 5);
        assert_eq!(
            a.normalize("f", "<i>ÀB</i>").unwrap(),
            b"<i>ab</i>".to_vec()
        );
        assert_eq!(
            a.to_string(),
            "CustomAnalyzer(org.apache.lucene.analysis.charfilter.HTMLStripCharFilterFactory,\
             org.apache.lucene.analysis.core.WhitespaceTokenizerFactory,\
             org.apache.lucene.analysis.core.LowerCaseFilterFactory,\
             org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilterFactory)"
        );
        assert_eq!(a.char_filter_factories().len(), 1);
        assert_eq!(a.token_filter_factories().len(), 2);
        assert_eq!(
            a.tokenizer_factory().base().lucene_match_version(),
            Version::LUCENE_10_0_0
        );
        let b = CustomAnalyzer::builder()
            .with_tokenizer_class::<WhitespaceTokenizerFactory>(&[])
            .unwrap()
            .add_char_filter_class::<crate::factory::CJKWidthCharFilterFactory>(&[])
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(terms(b.analyzer(), "Ａ b"), vec!["A", "b"]);
        assert_eq!(b.analyzer().position_increment_gap_for_field("f"), 0);
        assert_eq!(b.into_analyzer().offset_gap_for_field("f"), 1);
    }

    #[test]
    fn javas_builder_errors() {
        let e = CustomAnalyzer::builder().build().err().unwrap();
        assert_eq!(e.message, "You have to set at least a tokenizer.");
        let e = CustomAnalyzer::builder()
            .with_tokenizer("whitespace", &[])
            .unwrap()
            .with_tokenizer("keyword", &[])
            .err()
            .unwrap();
        assert_eq!(e.java_class(), "AlreadySetException");
        let e = CustomAnalyzer::builder()
            .with_tokenizer("whitespace", &[])
            .unwrap()
            .with_default_match_version(Version::LATEST)
            .err()
            .unwrap();
        assert_eq!(e.java_class(), "IllegalStateException");
        let b = CustomAnalyzer::builder()
            .with_default_match_version(Version::LATEST)
            .unwrap();
        assert!(b.with_default_match_version(Version::LATEST).is_err());
        assert!(CustomAnalyzer::builder()
            .with_position_increment_gap(-1)
            .is_err());
        assert!(CustomAnalyzer::builder().with_offset_gap(-1).is_err());
        let b = CustomAnalyzer::builder().with_offset_gap(1).unwrap();
        assert!(b.with_offset_gap(1).is_err());
        let b = CustomAnalyzer::builder()
            .with_position_increment_gap(1)
            .unwrap();
        assert!(b.with_position_increment_gap(1).is_err());
        let e = CustomAnalyzer::builder()
            .with_tokenizer("whitespace", &["x"])
            .err()
            .unwrap();
        assert_eq!(
            e.message,
            "Key-value pairs expected, so the number of params must be even."
        );
        let e = CustomAnalyzer::builder()
            .when("lowercase", &[])
            .err()
            .unwrap();
        assert_eq!(
            e.message,
            "TokenFilterFactory lowercase is not a ConditionalTokenFilterFactory"
        );
        let e = CustomAnalyzer::builder()
            .when_factory(Box::new(
                LowerCaseFilterFactory::from_args(&mut JavaArgs::new()).unwrap(),
            ))
            .endwhen()
            .err()
            .unwrap();
        assert!(e
            .message
            .ends_with("is not a ConditionalTokenFilterFactory"));
        assert!(CustomAnalyzer::builder_with_dir("/nonexistent/dir").is_err());
        assert!(CustomAnalyzer::builder()
            .add_char_filter("nope", &[])
            .is_err());
        assert!(CustomAnalyzer::builder()
            .add_token_filter("nope", &["a"])
            .is_err());
        assert!(CustomAnalyzer::builder()
            .add_char_filter("nope", &["a"])
            .is_err());
    }

    #[test]
    fn conditions() {
        let loader = MapResourceLoader::new().with("p.txt", "FOO\n");
        let a = CustomAnalyzer::builder_with_loader(Box::new(loader))
            .with_tokenizer("whitespace", &[])
            .unwrap()
            .when("protectedTerm", &["protected", "p.txt"])
            .unwrap()
            .add_token_filter("lowercase", &[])
            .unwrap()
            .add_token_filter_class::<crate::factory::ReverseStringFilterFactory>(&[])
            .unwrap()
            .endwhen()
            .unwrap()
            .when_term(|t| t.len() > 3)
            .add_token_filter("uppercase", &[])
            .unwrap()
            .endwhen()
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(terms(&a, "FOO BAR Quux"), vec!["FOO", "rab", "XUUQ"]);
        let e = CustomAnalyzer::builder()
            .when_term(|_| true)
            .add_token_filter("nope", &[])
            .err()
            .unwrap();
        assert_eq!(e.java_class(), "IllegalArgumentException");
        assert!(CustomAnalyzer::builder().when("nope", &[]).is_err());
        assert!(CustomAnalyzer::builder()
            .when("protectedTerm", &["x"])
            .is_err());
        let c = CustomAnalyzer::builder().when_term(|_| true);
        assert!(c.add_token_filter("lowercase", &["x"]).is_err());
        let c = CustomAnalyzer::builder().when_term(|_| true);
        assert!(c
            .add_token_filter_class::<LowerCaseFilterFactory>(&["x"])
            .is_err());
    }
}
