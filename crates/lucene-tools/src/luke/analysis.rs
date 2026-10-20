//! `org.apache.lucene.luke.models.analysis`: the analysis components on
//! offer, an analyzer by class name or built from components, the tokens it
//! makes of a text, and -- for a built one -- what each component in turn
//! makes of it.
//!
//! Differs: `addExternalJars` is not ported (there are no jars to load
//! components from); an analyzer by class name is one of those
//! [`spi::new_analyzer`] knows. A token's attributes are the port's: the
//! core attributes are always present (see [`AttributeSource`]), so every
//! token lists all of them where Java lists those its chain added.

use std::collections::BTreeSet;

use lucene_analysis::factory::args::JavaArgs;
use lucene_analysis::factory::spi;
use lucene_analysis::factory::CustomAnalyzer;
use lucene_analysis::{
    AnalysisError, Analyzer, AttrValue, AttributeSource, CharReader, State, StrReader, TokenStream,
};

use super::{LukeError, Result};

/// Luke's classpath: `lucene-suggest`'s `SuggestStopFilterFactory` and
/// analysis-common's `Word2VecSynonymFilterFactory`, both ported in
/// `lucene-search`, registered once, as `ServiceLoader` finds them.
fn classpath() {
    static REGISTERED: std::sync::Once = std::sync::Once::new();
    REGISTERED.call_once(|| {
        // a name already taken is ignored, not an error
        let _ = lucene_search::word2vec::register_factories();
        let _ = lucene_search::suggest::suggest_stop_factory::register_factories();
    });
}

/// `Analysis.TokenAttribute`: one attribute implementation of a token and
/// its values, in reflection order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenAttribute {
    /// The implementation's simple class name.
    pub att_class: String,
    pub att_values: Vec<(String, String)>,
}

/// `Analysis.Token`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub term: String,
    pub attributes: Vec<TokenAttribute>,
}

/// `Analysis.CharfilteredText`: a char filter's output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CharfilteredText {
    pub name: String,
    pub text: String,
}

/// `Analysis.NamedTokens`: a tokenizer's or a token filter's tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedTokens {
    pub name: String,
    pub tokens: Vec<Token>,
}

/// `Analysis.StepByStepResult`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StepByStepResult {
    pub charfiltered_texts: Vec<CharfilteredText>,
    pub named_tokens: Vec<NamedTokens>,
}

/// `CustomAnalyzerConfig.ComponentConfig`: a component's SPI name and its
/// parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentConfig {
    pub name: String,
    pub params: Vec<(String, String)>,
}

impl ComponentConfig {
    pub fn new(name: &str, params: Vec<(String, String)>) -> Self {
        ComponentConfig {
            name: name.to_string(),
            params,
        }
    }

    fn args(&self) -> JavaArgs {
        JavaArgs::from_pairs(&self.params)
    }
}

/// `CustomAnalyzerConfig` (without a config directory: Luke's own front end
/// sets one only to load resource files).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomAnalyzerConfig {
    pub tokenizer: ComponentConfig,
    pub char_filters: Vec<ComponentConfig>,
    pub token_filters: Vec<ComponentConfig>,
}

impl CustomAnalyzerConfig {
    /// `new CustomAnalyzerConfig.Builder(tokenizerName, tokenizerParams)`.
    pub fn new(tokenizer: ComponentConfig) -> Self {
        CustomAnalyzerConfig {
            tokenizer,
            char_filters: Vec::new(),
            token_filters: Vec::new(),
        }
    }
}

/// The current analyzer.
enum Current {
    /// `analyzer` with its class name.
    Plain(Analyzer, String),
    Custom(CustomAnalyzer),
}

/// `CustomAnalyzer`'s class name.
pub const CUSTOM_ANALYZER_CLASS: &str = "org.apache.lucene.analysis.custom.CustomAnalyzer";

/// `AnalysisImpl`.
pub struct Analysis {
    analyzer: Current,
}

impl Default for Analysis {
    fn default() -> Self {
        Self::new()
    }
}

fn exception(e: AnalysisError) -> LukeError {
    LukeError::luke(e.to_string())
}

impl Analysis {
    /// `new AnalysisImpl()`: a `StandardAnalyzer`.
    pub fn new() -> Self {
        classpath();
        Analysis {
            analyzer: Current::Plain(
                Analyzer::new(lucene_analysis::StandardAnalyzer::default()),
                "org.apache.lucene.analysis.standard.StandardAnalyzer".to_string(),
            ),
        }
    }

    /// `getAvailableCharFilters()`, by name.
    pub fn available_char_filters() -> Vec<String> {
        sorted(spi::available_char_filters())
    }

    /// `getAvailableTokenizers()`, by name.
    pub fn available_tokenizers() -> Vec<String> {
        sorted(spi::available_tokenizers())
    }

    /// `getAvailableTokenFilters()`, by name.
    pub fn available_token_filters() -> Vec<String> {
        classpath();
        sorted(spi::available_token_filters())
    }

    /// `createAnalyzerFromClassName(analyzerType)`.
    ///
    /// # Errors
    /// `LukeException` for a class the port cannot instantiate.
    pub fn create_analyzer_from_class_name(&mut self, class_name: &str) -> Result<()> {
        let analyzer = spi::new_analyzer(class_name)
            .map_err(|_| LukeError::luke(format!("Failed to instantiate class: {class_name}")))?;
        self.analyzer = Current::Plain(analyzer, class_name.to_string());
        Ok(())
    }

    /// `buildCustomAnalyzer(config)`.
    ///
    /// # Errors
    /// `LukeException` when a component is unknown or refuses its parameters.
    pub fn build_custom_analyzer(&mut self, config: &CustomAnalyzerConfig) -> Result<()> {
        let build = || {
            let mut b = CustomAnalyzer::builder()
                .with_tokenizer_args(&config.tokenizer.name, config.tokenizer.args())?;
            for cf in &config.char_filters {
                b = b.add_char_filter_args(&cf.name, cf.args())?;
            }
            for tf in &config.token_filters {
                b = b.add_token_filter_args(&tf.name, tf.args())?;
            }
            b.build()
        };
        let custom = build().map_err(|_| LukeError::luke("Failed to build custom analyzer."))?;
        self.analyzer = Current::Custom(custom);
        Ok(())
    }

    /// `currentAnalyzer().getClass().getName()`.
    pub fn current_analyzer_class(&self) -> &str {
        match &self.analyzer {
            Current::Plain(_, class) => class,
            Current::Custom(_) => CUSTOM_ANALYZER_CLASS,
        }
    }

    fn current(&self) -> &Analyzer {
        match &self.analyzer {
            Current::Plain(a, _) => a,
            Current::Custom(c) => c.analyzer(),
        }
    }

    /// `analyze(text)`: the analyzer's tokens of `text`.
    ///
    /// # Errors
    /// `LukeException` when the analysis fails.
    pub fn analyze(&self, text: &str) -> Result<Vec<Token>> {
        let mut stream = self.current().token_stream("", text).map_err(exception)?;
        stream.reset().map_err(exception)?;
        let mut tokens = Vec::new();
        while stream.increment_token().map_err(exception)? {
            tokens.push(token(stream.attributes()));
        }
        stream.close().map_err(exception)?;
        Ok(tokens)
    }

    /// `analyzeStepByStep(text)`: each char filter's output, then the
    /// tokenizer's tokens and each token filter's, every filter reading the
    /// tokens before it.
    ///
    /// As Java's does, the tokenizer reads the text through the *last* char
    /// filter alone (each one is created over the original reader), while
    /// the char filters' outputs shown are chained.
    ///
    /// # Errors
    /// `LukeException` for an analyzer not built from components, or when
    /// the analysis fails.
    pub fn analyze_step_by_step(&self, text: &str) -> Result<StepByStepResult> {
        let Current::Custom(custom) = &self.analyzer else {
            return Err(LukeError::luke("Analyzer is not CustomAnalyzer."));
        };
        let mut result = StepByStepResult::default();
        let mut reader: Box<dyn CharReader> = Box::new(StrReader::new(text));
        let mut source = text.to_string();
        for cf in custom.char_filter_factories() {
            reader = cf.create(Box::new(StrReader::new(text)));
            source = write_char_stream(cf.create(Box::new(StrReader::new(source.as_str()))))
                .map_err(exception)?;
            result.charfiltered_texts.push(CharfilteredText {
                name: spi_name(cf.base().class_name()),
                text: source.clone(),
            });
        }
        let tokenizer = custom.tokenizer_factory();
        let mut stream = tokenizer.create().map_err(exception)?;
        stream
            .as_tokenizer()
            .ok_or_else(|| LukeError::luke("not a Tokenizer"))?
            .set_reader(reader)
            .map_err(exception)?;
        let (tokens, mut states, mut attributes) = analyze_token_stream(stream)?;
        result.named_tokens.push(NamedTokens {
            name: spi_name(tokenizer.base().class_name()),
            tokens,
        });
        for tf in custom.token_filter_factories() {
            let input = ListBasedTokenStream {
                attributes,
                tokens: states,
                next: 0,
            };
            let stream = tf.create(Box::new(input)).map_err(exception)?;
            let (tokens, s, a) = analyze_token_stream(stream)?;
            states = s;
            attributes = a;
            result.named_tokens.push(NamedTokens {
                name: spi_name(tf.base().class_name()),
                tokens,
            });
        }
        Ok(result)
    }
}

fn sorted(names: Vec<&str>) -> Vec<String> {
    names
        .into_iter()
        .map(str::to_string)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// `findSPIName(factory.getClass())`.
fn spi_name(class_name: &str) -> String {
    spi::find_spi_name(class_name)
        .unwrap_or(class_name)
        .to_string()
}

/// The simple class name of the implementation Java's default attribute
/// factory gives the attribute `interface`: `PackedTokenAttributeImpl` for
/// the core token attributes, `<Interface>Impl` for the others.
pub fn impl_class(interface: &str) -> String {
    let simple = interface.rsplit('.').next().unwrap_or(interface);
    match simple {
        "CharTermAttribute"
        | "TermToBytesRefAttribute"
        | "OffsetAttribute"
        | "PositionIncrementAttribute"
        | "PositionLengthAttribute"
        | "TypeAttribute"
        | "TermFrequencyAttribute" => "PackedTokenAttributeImpl".to_string(),
        other => format!("{other}Impl"),
    }
}

/// `copyAttributes`: each attribute implementation's non-null values.
fn token(attributes: &AttributeSource) -> Token {
    let mut atts: Vec<TokenAttribute> = Vec::new();
    attributes.reflect_with(&mut |interface, key, value| {
        let class = impl_class(interface);
        let idx = match atts.iter().position(|a| a.att_class == class) {
            Some(i) => i,
            None => {
                atts.push(TokenAttribute {
                    att_class: class,
                    att_values: Vec::new(),
                });
                atts.len().saturating_sub(1)
            }
        };
        let is_null = matches!(
            value,
            AttrValue::Str(None) | AttrValue::Bytes(None) | AttrValue::List(None)
        );
        if !is_null {
            atts[idx]
                .att_values
                .push((key.to_string(), value.to_string()));
        }
    });
    Token {
        term: attributes.term().to_string(),
        attributes: atts,
    }
}

/// `analyzeTokenStream`: the stream's tokens, each one's captured state,
/// and the stream's attributes for the next stage to start from.
fn analyze_token_stream(
    mut stream: Box<dyn TokenStream>,
) -> Result<(Vec<Token>, Vec<State>, AttributeSource)> {
    let failed = |_| LukeError::luke("Error occurred while iterating over TokenStream");
    let mut tokens = Vec::new();
    let mut states = Vec::new();
    stream.reset().map_err(failed)?;
    while stream.increment_token().map_err(failed)? {
        states.push(stream.attributes().capture_state());
        tokens.push(token(stream.attributes()));
    }
    stream.end().map_err(failed)?;
    let attributes = stream.attributes().clone();
    // `IOUtils.closeWhileHandlingException`
    let _ = stream.close();
    Ok((tokens, states, attributes))
}

/// `writeCharStream`: reads 1024 chars at a time while a read fills the
/// buffer.
fn write_char_stream(mut input: Box<dyn CharReader>) -> std::result::Result<String, AnalysisError> {
    const BUFFER_SIZE: usize = 1024;
    let mut buf = [0u16; BUFFER_SIZE];
    let mut out: Vec<u16> = Vec::new();
    loop {
        let len = input.read(&mut buf)?;
        out.extend_from_slice(&buf[..len.min(BUFFER_SIZE)]);
        if len != BUFFER_SIZE {
            break;
        }
    }
    Ok(String::from_utf16_lossy(&out))
}

/// `ListBasedTokenStream`: replays the tokens a stage made, as the next
/// stage's input.
struct ListBasedTokenStream {
    attributes: AttributeSource,
    tokens: Vec<State>,
    next: usize,
}

impl TokenStream for ListBasedTokenStream {
    fn attributes(&self) -> &AttributeSource {
        &self.attributes
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.attributes
    }

    fn reset(&mut self) -> std::result::Result<(), AnalysisError> {
        self.next = 0;
        Ok(())
    }

    fn increment_token(&mut self) -> std::result::Result<bool, AnalysisError> {
        let Some(state) = self.tokens.get(self.next) else {
            return Ok(false);
        };
        self.attributes.clear_attributes();
        self.attributes.restore_state(state);
        self.next = self.next.saturating_add(1);
        Ok(true)
    }

    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attribute_classes_are_their_default_implementations() {
        assert_eq!(
            impl_class("org.apache.lucene.analysis.tokenattributes.OffsetAttribute"),
            "PackedTokenAttributeImpl"
        );
        assert_eq!(
            impl_class("org.apache.lucene.analysis.tokenattributes.KeywordAttribute"),
            "KeywordAttributeImpl"
        );
        assert_eq!(impl_class("Plain"), "PlainImpl");
    }

    #[test]
    fn a_long_char_stream_is_read_whole() {
        let text = "x".repeat(2048 + 5);
        assert_eq!(
            write_char_stream(Box::new(StrReader::new(text.as_str()))).unwrap(),
            text
        );
        assert_eq!(write_char_stream(Box::new(StrReader::new(""))).unwrap(), "");
    }

    #[test]
    fn step_by_step_needs_a_custom_analyzer() {
        let a = Analysis::default();
        assert_eq!(
            a.analyze_step_by_step("x").unwrap_err().to_string(),
            "LukeException: Analyzer is not CustomAnalyzer."
        );
        assert_eq!(
            a.current_analyzer_class(),
            "org.apache.lucene.analysis.standard.StandardAnalyzer"
        );
    }
}
