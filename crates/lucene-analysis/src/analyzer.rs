//! `org.apache.lucene.analysis.Analyzer` and its wrappers:
//! `TokenStreamComponents`, `ReuseStrategy`, `AnalyzerWrapper`,
//! `DelegatingAnalyzerWrapper`.
//!
//! # Shape
//!
//! Java's `Analyzer` is one abstract class holding two kinds of method: the
//! ones a subclass overrides (`createComponents`, `normalize(String,
//! TokenStream)`, `initReader`, the two gaps) and the `final` machinery built
//! on them (`tokenStream`, `normalize(String, String)`, component reuse,
//! `close`). The port splits them along that line:
//!
//! - [`AnalyzerDefinition`] is the overridable half -- implement it to define
//!   an analyzer (as [`crate::StandardAnalyzer`] does);
//! - [`Analyzer`] is the final half: it owns a definition plus its reuse
//!   cache, and is what every caller analyzes with.
//!
//! # Reuse
//!
//! Java caches `TokenStreamComponents` in a `CloseableThreadLocal`, one set
//! per thread (per field, under `PER_FIELD_REUSE_STRATEGY`). A Rust
//! `Analyzer` is shared by reference across threads, so the cache is a pool
//! behind a `Mutex`: [`Analyzer::token_stream`] takes a free set of
//! components (creating one on a miss, as Java does), and the returned
//! [`AnalyzerTokenStream`] puts it back when dropped. The observable
//! behaviour is Java's -- components are created once per key and reused
//! after `close()` -- and two concurrent streams simply use two sets, as two
//! Java threads would.
//!
//! Dropping an [`AnalyzerTokenStream`] closes it first if the consumer did
//! not, so a stream abandoned on an error path cannot poison the pool with
//! the "close() call missing" state Java would throw on next use.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::attributes::AttributeSource;
use crate::reader::{read_to_string, CharReader, StrReader};
use crate::token_stream::{TokenStream, Tokenizer};
use crate::{
    AnalysisError, AnalyzedTokens, CharArraySet, KeywordTokenizer, LowerCaseFilter, StopFilter,
    Token,
};

/// `Analyzer.TokenStreamComponents`: the chain's source (what receives the
/// reader) and its sink (what is consumed).
pub struct TokenStreamComponents {
    source: Option<SourceFn>,
    sink: Box<dyn TokenStream>,
}

/// `TokenStreamComponents.source`: a `Consumer<Reader>`. It receives the
/// sink too, because in Rust the tokenizer that takes the reader is owned
/// *inside* the sink chain.
pub type SourceFn =
    Box<dyn FnMut(&mut dyn TokenStream, Box<dyn CharReader>) -> Result<(), AnalysisError> + Send>;

impl TokenStreamComponents {
    /// `TokenStreamComponents(Tokenizer, TokenStream)` /
    /// `TokenStreamComponents(Tokenizer)`: the reader goes to the
    /// [`Tokenizer`] at the source of `sink` (found with
    /// [`TokenStream::as_tokenizer`]).
    pub fn new(sink: impl TokenStream + 'static) -> Self {
        TokenStreamComponents {
            source: None,
            sink: Box::new(sink),
        }
    }

    /// `TokenStreamComponents(Consumer<Reader>, TokenStream)`: a custom
    /// source hook.
    pub fn with_source(
        source: impl FnMut(&mut dyn TokenStream, Box<dyn CharReader>) -> Result<(), AnalysisError>
            + Send
            + 'static,
        sink: impl TokenStream + 'static,
    ) -> Self {
        TokenStreamComponents {
            source: Some(Box::new(source)),
            sink: Box::new(sink),
        }
    }

    /// `setReader(Reader)`.
    pub fn set_reader(&mut self, reader: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        match &mut self.source {
            Some(source) => source(&mut *self.sink, reader),
            None => match self.sink.as_tokenizer() {
                Some(tokenizer) => tokenizer.set_reader(reader),
                None => Err(AnalysisError::IllegalState(
                    "TokenStreamComponents has no Tokenizer source to receive the reader"
                        .to_string(),
                )),
            },
        }
    }

    /// `getTokenStream()`.
    pub fn token_stream(&mut self) -> &mut dyn TokenStream {
        &mut *self.sink
    }

    /// The sink, by value (for a wrapper's `wrapComponents`).
    pub fn into_parts(self) -> (Option<SourceFn>, Box<dyn TokenStream>) {
        (self.source, self.sink)
    }

    /// Reassembles [`Self::into_parts`]' output around a new sink.
    pub fn from_parts(source: Option<SourceFn>, sink: Box<dyn TokenStream>) -> Self {
        TokenStreamComponents { source, sink }
    }
}

/// `Analyzer.ReuseStrategy`: which components a field reuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReuseStrategy {
    /// `GLOBAL_REUSE_STRATEGY`: one set of components for every field.
    #[default]
    Global,
    /// `PER_FIELD_REUSE_STRATEGY`: one set per field name.
    PerField,
}

/// The overridable half of Java's `Analyzer` (see the module docs).
pub trait AnalyzerDefinition: Send + Sync {
    /// `createComponents(String fieldName)`.
    fn create_components(&self, field_name: &str) -> Result<TokenStreamComponents, AnalysisError>;

    /// `normalize(String fieldName, TokenStream in)`: the filters that apply
    /// to a query term (no tokenization). Java's default is the identity.
    fn normalize(&self, _field_name: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        input
    }

    /// `initReader(String, Reader)`: where `CharFilter`s go.
    fn init_reader(&self, _field_name: &str, reader: Box<dyn CharReader>) -> Box<dyn CharReader> {
        reader
    }

    /// `initReaderForNormalization(String, Reader)`.
    fn init_reader_for_normalization(
        &self,
        _field_name: &str,
        reader: Box<dyn CharReader>,
    ) -> Box<dyn CharReader> {
        reader
    }

    /// `getPositionIncrementGap(String)`: Java's default is `0`.
    fn position_increment_gap(&self, _field_name: &str) -> i32 {
        0
    }

    /// `getOffsetGap(String)`: Java's default is `1`.
    fn offset_gap(&self, _field_name: &str) -> i32 {
        1
    }
}

/// `org.apache.lucene.analysis.AnalyzerWrapper`: an analyzer defined by
/// another, per field, optionally wrapping its components, readers and
/// normalization chain.
///
/// Every `AnalyzerWrapper` is an [`AnalyzerDefinition`] whose methods are
/// Java's `final` overrides: `createComponents` wraps the wrapped analyzer's,
/// `initReader` applies [`Self::wrap_reader`] and then the wrapped analyzer's
/// own `initReader`, and the gaps are the wrapped analyzer's. Wrap it with
/// [`Analyzer::with_reuse_strategy`] (Java's constructor argument).
pub trait AnalyzerWrapper: Send + Sync {
    /// `getWrappedAnalyzer(String)`.
    fn wrapped_analyzer(&self, field_name: &str) -> &Analyzer;

    /// `wrapComponents(String, TokenStreamComponents)`.
    fn wrap_components(
        &self,
        _field_name: &str,
        components: TokenStreamComponents,
    ) -> TokenStreamComponents {
        components
    }

    /// `wrapTokenStreamForNormalization(String, TokenStream)`.
    fn wrap_token_stream_for_normalization(
        &self,
        _field_name: &str,
        input: Box<dyn TokenStream>,
    ) -> Box<dyn TokenStream> {
        input
    }

    /// `wrapReader(String, Reader)`.
    fn wrap_reader(&self, _field_name: &str, reader: Box<dyn CharReader>) -> Box<dyn CharReader> {
        reader
    }

    /// `wrapReaderForNormalization(String, Reader)`.
    fn wrap_reader_for_normalization(
        &self,
        _field_name: &str,
        reader: Box<dyn CharReader>,
    ) -> Box<dyn CharReader> {
        reader
    }
}

impl<W: AnalyzerWrapper> AnalyzerDefinition for W {
    fn create_components(&self, field_name: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let inner = self
            .wrapped_analyzer(field_name)
            .create_components(field_name)?;
        Ok(self.wrap_components(field_name, inner))
    }

    fn normalize(&self, field_name: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        let inner = self
            .wrapped_analyzer(field_name)
            .normalize_stream(field_name, input);
        self.wrap_token_stream_for_normalization(field_name, inner)
    }

    fn init_reader(&self, field_name: &str, reader: Box<dyn CharReader>) -> Box<dyn CharReader> {
        self.wrapped_analyzer(field_name)
            .init_reader(field_name, self.wrap_reader(field_name, reader))
    }

    fn init_reader_for_normalization(
        &self,
        field_name: &str,
        reader: Box<dyn CharReader>,
    ) -> Box<dyn CharReader> {
        self.wrapped_analyzer(field_name)
            .init_reader_for_normalization(
                field_name,
                self.wrap_reader_for_normalization(field_name, reader),
            )
    }

    fn position_increment_gap(&self, field_name: &str) -> i32 {
        self.wrapped_analyzer(field_name)
            .position_increment_gap_for_field(field_name)
    }

    fn offset_gap(&self, field_name: &str) -> i32 {
        self.wrapped_analyzer(field_name)
            .offset_gap_for_field(field_name)
    }
}

/// `org.apache.lucene.analysis.DelegatingAnalyzerWrapper`: an
/// [`AnalyzerWrapper`] that wraps nothing, so it can reuse the wrapped
/// analyzers' own components instead of caching its own (Java's
/// `DelegatingReuseStrategy`). Build one with [`Analyzer::delegating`].
pub trait DelegatingAnalyzerWrapper: Send + Sync {
    /// `getWrappedAnalyzer(String)`.
    fn wrapped_analyzer(&self, field_name: &str) -> &Analyzer;
}

/// A [`DelegatingAnalyzerWrapper`] seen as the [`AnalyzerWrapper`] Java
/// makes it: every `wrap*` method is the identity (and `final`).
struct Delegating(Box<dyn DelegatingAnalyzerWrapper>);

impl AnalyzerWrapper for Delegating {
    fn wrapped_analyzer(&self, field_name: &str) -> &Analyzer {
        self.0.wrapped_analyzer(field_name)
    }
}

/// What an [`Analyzer`] analyzes with.
enum Kind {
    Definition(Box<dyn AnalyzerDefinition>),
    Delegating(Delegating),
    /// The configurable built-in chain behind [`Analyzer::standard`] /
    /// [`Analyzer::keyword`] and their `with_*` builders.
    Builtin(BuiltinChain),
}

type Pool = HashMap<Option<String>, Vec<TokenStreamComponents>>;

/// `org.apache.lucene.analysis.Analyzer`: the final machinery over an
/// [`AnalyzerDefinition`] -- component reuse, `tokenStream`, `normalize`,
/// `close` -- plus this crate's built-in configurable chains.
pub struct Analyzer {
    kind: Kind,
    reuse_strategy: ReuseStrategy,
    /// The reuse cache; `None` once closed (Java nulls `storedValue`).
    pool: Mutex<Option<Pool>>,
    /// Override of the definition's `getPositionIncrementGap`, for every
    /// field (see [`Self::with_position_increment_gap`]).
    position_increment_gap: Option<i32>,
    /// Override of the definition's `getOffsetGap`.
    offset_gap: Option<i32>,
}

impl Analyzer {
    fn from_kind(kind: Kind, reuse_strategy: ReuseStrategy) -> Self {
        Analyzer {
            kind,
            reuse_strategy,
            pool: Mutex::new(Some(HashMap::new())),
            position_increment_gap: None,
            offset_gap: None,
        }
    }

    /// An analyzer over `definition`, with `GLOBAL_REUSE_STRATEGY` (Java's
    /// no-argument `Analyzer()` constructor).
    pub fn new(definition: impl AnalyzerDefinition + 'static) -> Self {
        Self::with_reuse_strategy(definition, ReuseStrategy::Global)
    }

    /// `Analyzer(ReuseStrategy)`.
    pub fn with_reuse_strategy(
        definition: impl AnalyzerDefinition + 'static,
        reuse_strategy: ReuseStrategy,
    ) -> Self {
        Self::from_kind(Kind::Definition(Box::new(definition)), reuse_strategy)
    }

    /// A `DelegatingAnalyzerWrapper`: token streams come from (and reuse the
    /// components of) the wrapped analyzers. `fallback` is Java's
    /// `fallbackStrategy`, used only when this analyzer's own components are
    /// asked for directly (it never caches any while delegating).
    pub fn delegating(
        wrapper: impl DelegatingAnalyzerWrapper + 'static,
        fallback: ReuseStrategy,
    ) -> Self {
        Self::from_kind(Kind::Delegating(Delegating(Box::new(wrapper))), fallback)
    }

    fn definition(&self) -> &dyn AnalyzerDefinition {
        match &self.kind {
            Kind::Definition(d) => &**d,
            Kind::Delegating(d) => d,
            Kind::Builtin(b) => b,
        }
    }

    /// `getReuseStrategy()`.
    pub fn reuse_strategy(&self) -> ReuseStrategy {
        self.reuse_strategy
    }

    /// The definition's `createComponents` (what an [`AnalyzerWrapper`]
    /// wraps).
    pub fn create_components(
        &self,
        field_name: &str,
    ) -> Result<TokenStreamComponents, AnalysisError> {
        self.definition().create_components(field_name)
    }

    /// The definition's `normalize(String, TokenStream)`.
    pub fn normalize_stream(
        &self,
        field_name: &str,
        input: Box<dyn TokenStream>,
    ) -> Box<dyn TokenStream> {
        self.definition().normalize(field_name, input)
    }

    /// `initReader(String, Reader)`.
    pub fn init_reader(
        &self,
        field_name: &str,
        reader: Box<dyn CharReader>,
    ) -> Box<dyn CharReader> {
        self.definition().init_reader(field_name, reader)
    }

    /// `initReaderForNormalization(String, Reader)`.
    pub fn init_reader_for_normalization(
        &self,
        field_name: &str,
        reader: Box<dyn CharReader>,
    ) -> Box<dyn CharReader> {
        self.definition()
            .init_reader_for_normalization(field_name, reader)
    }

    /// `getPositionIncrementGap(fieldName)`.
    pub fn position_increment_gap_for_field(&self, field_name: &str) -> i32 {
        self.position_increment_gap
            .unwrap_or_else(|| self.definition().position_increment_gap(field_name))
    }

    /// `getOffsetGap(fieldName)`.
    pub fn offset_gap_for_field(&self, field_name: &str) -> i32 {
        self.offset_gap
            .unwrap_or_else(|| self.definition().offset_gap(field_name))
    }

    /// `getPositionIncrementGap` for a caller with no field name (this
    /// crate's field-agnostic API; the definition sees `""`).
    pub fn position_increment_gap(&self) -> i32 {
        self.position_increment_gap_for_field("")
    }

    /// `getOffsetGap` for a caller with no field name.
    pub fn offset_gap(&self) -> i32 {
        self.offset_gap_for_field("")
    }

    /// Sets this analyzer's `getPositionIncrementGap` for every field.
    ///
    /// The number of positions inserted **between two values of the same
    /// multi-valued field**. Java's base `Analyzer` returns `0` from it --
    /// so a phrase query *can* match across a value boundary -- which is why
    /// every consumer of Lucene (OpenSearch's `position_increment_gap`,
    /// default 100) exposes an override; Java overrides it by subclassing.
    pub fn with_position_increment_gap(mut self, gap: i32) -> Self {
        self.position_increment_gap = Some(gap);
        self
    }

    /// Sets this analyzer's `getOffsetGap` for every field: the character
    /// offsets inserted between two values of the same multi-valued field.
    /// Java's default is **`1`**, not `0`.
    pub fn with_offset_gap(mut self, gap: i32) -> Self {
        self.offset_gap = Some(gap);
        self
    }

    /// `close()`: drops the reuse cache; later `token_stream` calls fail
    /// with Java's `AlreadyClosedException` message.
    pub fn close(&self) {
        *self.pool.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    fn pool_key(&self, field_name: &str) -> Option<String> {
        match self.reuse_strategy {
            ReuseStrategy::Global => None,
            ReuseStrategy::PerField => Some(field_name.to_string()),
        }
    }

    /// `ReuseStrategy.getReusableComponents`.
    fn take_components(
        &self,
        key: &Option<String>,
    ) -> Result<Option<TokenStreamComponents>, AnalysisError> {
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        match pool.as_mut() {
            None => Err(AnalysisError::AlreadyClosed(
                "this Analyzer is closed".to_string(),
            )),
            Some(pool) => Ok(pool.get_mut(key).and_then(Vec::pop)),
        }
    }

    /// `ReuseStrategy.setReusableComponents`.
    fn put_components(&self, key: Option<String>, components: TokenStreamComponents) {
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pool) = pool.as_mut() {
            pool.entry(key).or_default().push(components);
        }
    }

    /// `tokenStream(String fieldName, Reader reader)`.
    pub fn token_stream_from_reader(
        &self,
        field_name: &str,
        reader: Box<dyn CharReader>,
    ) -> Result<AnalyzerTokenStream<'_>, AnalysisError> {
        if let Kind::Delegating(d) = &self.kind {
            return d
                .wrapped_analyzer(field_name)
                .token_stream_from_reader(field_name, reader);
        }
        let key = self.pool_key(field_name);
        let components = self.take_components(&key)?;
        let r = self.init_reader(field_name, reader);
        let mut components = match components {
            Some(c) => c,
            None => self.create_components(field_name)?,
        };
        components.set_reader(r)?;
        Ok(AnalyzerTokenStream {
            analyzer: self,
            key,
            components: Some(components),
            closed: false,
        })
    }

    /// `tokenStream(String fieldName, String text)`.
    pub fn token_stream(
        &self,
        field_name: &str,
        text: &str,
    ) -> Result<AnalyzerTokenStream<'_>, AnalysisError> {
        self.token_stream_from_reader(field_name, Box::new(StrReader::new(text)))
    }

    /// `normalize(String fieldName, String text)`: the term bytes a query
    /// term normalizes to -- the char filters of
    /// `initReaderForNormalization`, then the definition's `normalize`
    /// chain over the whole text as one token.
    pub fn normalize(&self, field_name: &str, text: &str) -> Result<Vec<u8>, AnalysisError> {
        // apply char filters
        let mut filter_reader =
            self.init_reader_for_normalization(field_name, Box::new(StrReader::new(text)));
        let filtered_text = read_to_string(&mut *filter_reader)?;
        filter_reader.close()?;
        let length = crate::utf16_len(text) as i32;
        let mut ts = self.normalize_stream(
            field_name,
            Box::new(StringTokenStream::new(filtered_text, length)),
        );
        ts.reset()?;
        if !ts.increment_token()? {
            return Err(AnalysisError::IllegalState(format!(
                "The normalization token stream is expected to produce exactly 1 token, but got 0 for analyzer and input \"{text}\""
            )));
        }
        let term = ts.attributes().term_bytes().to_vec();
        if ts.increment_token()? {
            return Err(AnalysisError::IllegalState(format!(
                "The normalization token stream is expected to produce exactly 1 token, but got 2+ for analyzer and input \"{text}\""
            )));
        }
        ts.end()?;
        ts.close()?;
        Ok(term)
    }

    // ----------------------------------------------------- built-in chains

    /// A "standard"-style analyzer: Lucene's `StandardAnalyzer` chain --
    /// [`crate::StandardTokenizer`] (maxTokenLength 255) +
    /// [`LowerCaseFilter`] + an optional case-sensitive [`StopFilter`].
    /// ASCII folding, stemming and synonyms are off by default; the `with_*`
    /// builders below add them to this chain.
    pub fn standard(stopwords: Option<&std::collections::HashSet<String>>) -> Self {
        let chain = BuiltinChain {
            stopwords: stopwords.map(|s| Arc::new(CharArraySet::from(s))),
            ..BuiltinChain::default()
        };
        Self::from_kind(Kind::Builtin(chain), ReuseStrategy::Global)
    }

    /// Mirrors analysis-common's `KeywordAnalyzer`: the entire input becomes
    /// **exactly one token**, as given -- no segmentation, lowercasing,
    /// stopwords, stemming, folding or synonyms. Empty input still produces
    /// one empty token spanning `0..0`, as `KeywordTokenizer` does.
    ///
    /// The `with_*` chain builders have no effect on a keyword analyzer.
    pub fn keyword() -> Self {
        let chain = BuiltinChain {
            keyword: true,
            ..BuiltinChain::default()
        };
        Self::from_kind(Kind::Builtin(chain), ReuseStrategy::Global)
    }

    fn builtin_mut(&mut self) -> Option<&mut BuiltinChain> {
        // A builder changes what `create_components` builds, so cached
        // components are stale.
        if let Some(pool) = self.pool.get_mut().unwrap_or_else(|e| e.into_inner()) {
            pool.clear();
        }
        match &mut self.kind {
            Kind::Builtin(b) => Some(b),
            _ => None,
        }
    }

    /// Adds [`crate::AsciiFoldingFilter`] to a built-in chain. Order:
    /// tokenize -> **fold** -> lowercase -> stopwords -> stemming.
    /// No effect on an analyzer built from a definition.
    pub fn with_ascii_folding(mut self) -> Self {
        if let Some(b) = self.builtin_mut() {
            b.ascii_folding = true;
        }
        self
    }

    /// Adds [`crate::PorterStemFilter`] as the built-in chain's last stage
    /// before synonyms (stopwords see unstemmed terms, as in
    /// `EnglishAnalyzer`).
    pub fn with_stemming(mut self) -> Self {
        if let Some(b) = self.builtin_mut() {
            b.stemming = true;
        }
        self
    }

    /// Adds [`crate::SnowballEnglishStemFilter`] in the Porter stemmer's
    /// place; takes precedence over [`Self::with_stemming`].
    pub fn with_snowball_stemming(mut self) -> Self {
        if let Some(b) = self.builtin_mut() {
            b.snowball_stemming = true;
        }
        self
    }

    /// Adds [`crate::SynonymFilter::apply`] as the built-in chain's last
    /// stage (after stopwords, so a removed stopword is never expanded).
    pub fn with_synonyms(mut self, synonyms: HashMap<String, Vec<String>>) -> Self {
        if let Some(b) = self.builtin_mut() {
            b.synonyms = Some(Arc::new(synonyms));
            b.synonyms_bidirectional = false;
        }
        self
    }

    /// [`Self::with_synonyms`] with [`crate::SynonymFilter::apply_bidirectional`].
    pub fn with_bidirectional_synonyms(mut self, synonyms: HashMap<String, Vec<String>>) -> Self {
        if let Some(b) = self.builtin_mut() {
            b.synonyms = Some(Arc::new(synonyms));
            b.synonyms_bidirectional = true;
        }
        self
    }

    // -------------------------------------------- materialising helpers

    /// Runs the whole lifecycle over `text` for field `""`, calling
    /// `f(term, start_offset, end_offset, position_increment)` per token,
    /// and returns `end()`'s `(position_increment, end_offset)` -- the two
    /// values `IndexingChain.invertTokenStream` reads after `stream.end()`.
    pub fn try_for_each_token(
        &self,
        text: &str,
        mut f: impl FnMut(&str, i32, i32, i32),
    ) -> Result<(i32, i32), AnalysisError> {
        let mut ts = self.token_stream("", text)?;
        ts.reset()?;
        while ts.increment_token()? {
            let a = ts.attributes();
            f(
                a.term(),
                a.start_offset(),
                a.end_offset(),
                a.position_increment(),
            );
        }
        ts.end()?;
        let a = ts.attributes();
        let end = (a.position_increment(), a.end_offset());
        ts.close()?;
        Ok(end)
    }

    /// [`Self::try_for_each_token`] for the built-in chains, which cannot
    /// fail over a `&str`.
    ///
    /// # Panics
    ///
    /// If a custom [`AnalyzerDefinition`]'s chain returns an error; use
    /// [`Self::try_for_each_token`] for those.
    pub fn for_each_token(&self, text: &str, f: impl FnMut(&str, i32, i32, i32)) -> (i32, i32) {
        self.try_for_each_token(text, f)
            .expect("analysis of an in-memory string failed; use try_for_each_token")
    }

    /// The tokens of `text` (field `""`) with the `end()` values.
    pub fn try_analyze_stream(&self, text: &str) -> Result<AnalyzedTokens, AnalysisError> {
        let mut ts = self.token_stream("", text)?;
        collect_tokens(&mut ts)
    }

    /// [`Self::try_analyze_stream`].
    ///
    /// # Panics
    ///
    /// As [`Self::for_each_token`].
    pub fn analyze_stream(&self, text: &str) -> AnalyzedTokens {
        self.try_analyze_stream(text)
            .expect("analysis of an in-memory string failed; use try_analyze_stream")
    }

    /// The tokens of `text` (field `""`).
    ///
    /// # Panics
    ///
    /// As [`Self::for_each_token`].
    pub fn analyze(&self, text: &str) -> Vec<Token> {
        self.analyze_stream(text).tokens
    }
}

/// Consumes `ts` (`reset`, every token, `end`, `close`) into an
/// [`AnalyzedTokens`].
pub fn collect_tokens(ts: &mut dyn TokenStream) -> Result<AnalyzedTokens, AnalysisError> {
    let mut tokens = Vec::new();
    let end = crate::token_stream::consume(ts, |a| {
        tokens.push(Token {
            term: a.term().to_string(),
            start_offset: a.start_offset(),
            end_offset: a.end_offset(),
            position_increment: a.position_increment(),
            position_length: a.position_length(),
        })
    })?;
    Ok(AnalyzedTokens {
        tokens,
        final_position_increment: end.position_increment(),
        final_offset: end.end_offset(),
    })
}

/// The stream [`Analyzer::token_stream`] returns: the reused components'
/// sink, returned to the analyzer's pool on drop (closing it first if the
/// consumer did not).
pub struct AnalyzerTokenStream<'a> {
    analyzer: &'a Analyzer,
    key: Option<String>,
    components: Option<TokenStreamComponents>,
    closed: bool,
}

impl AnalyzerTokenStream<'_> {
    fn sink(&mut self) -> &mut dyn TokenStream {
        self.components
            .as_mut()
            .expect("components present until drop")
            .token_stream()
    }
}

impl TokenStream for AnalyzerTokenStream<'_> {
    fn attributes(&self) -> &AttributeSource {
        self.components
            .as_ref()
            .expect("components present until drop")
            .sink
            .attributes()
    }
    fn attributes_mut(&mut self) -> &mut AttributeSource {
        self.sink().attributes_mut()
    }
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        self.sink().increment_token()
    }
    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.sink().reset()
    }
    fn end(&mut self) -> Result<(), AnalysisError> {
        self.sink().end()
    }
    fn close(&mut self) -> Result<(), AnalysisError> {
        self.closed = true;
        self.sink().close()
    }
    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        self.sink().as_tokenizer()
    }
}

impl Drop for AnalyzerTokenStream<'_> {
    fn drop(&mut self) {
        let Some(mut components) = self.components.take() else {
            return;
        };
        if !self.closed && components.token_stream().close().is_err() {
            // A chain that cannot close is not reusable.
            return;
        }
        self.analyzer.put_components(self.key.take(), components);
    }
}

/// `Analyzer.StringTokenStream`: the whole (char-filtered) text as a single
/// token, the input of the `normalize` chain.
struct StringTokenStream {
    atts: AttributeSource,
    value: String,
    length: i32,
    used: bool,
}

impl StringTokenStream {
    fn new(value: String, length: i32) -> Self {
        StringTokenStream {
            atts: AttributeSource::new(),
            value,
            length,
            used: true,
        }
    }
}

impl TokenStream for StringTokenStream {
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }
    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }
    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.used = false;
        Ok(())
    }
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        if self.used {
            return Ok(false);
        }
        self.atts.clear_attributes();
        self.atts.set_term(&self.value);
        self.atts.set_offset(0, self.length)?;
        self.used = true;
        Ok(true)
    }
    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        self.atts.set_offset(self.length, self.length)
    }
}

/// The configurable chain behind [`Analyzer::standard`]/[`Analyzer::keyword`].
#[derive(Default)]
struct BuiltinChain {
    stopwords: Option<Arc<CharArraySet>>,
    ascii_folding: bool,
    stemming: bool,
    snowball_stemming: bool,
    synonyms: Option<Arc<HashMap<String, Vec<String>>>>,
    synonyms_bidirectional: bool,
    keyword: bool,
}

impl AnalyzerDefinition for BuiltinChain {
    fn create_components(&self, _field_name: &str) -> Result<TokenStreamComponents, AnalysisError> {
        use crate::legacy::{SynonymAdapter, TermRewriteFilter};
        use crate::StandardTokenizer;
        if self.keyword {
            return Ok(TokenStreamComponents::new(KeywordTokenizer::new()));
        }
        let plain = !self.ascii_folding
            && !self.stemming
            && !self.snowball_stemming
            && self.synonyms.is_none();
        if plain {
            // The common chains, monomorphised: StandardAnalyzer's own shape.
            let lower = LowerCaseFilter::new(StandardTokenizer::new());
            return Ok(match &self.stopwords {
                Some(stop) => TokenStreamComponents::new(StopFilter::new(lower, Arc::clone(stop))),
                None => TokenStreamComponents::new(lower),
            });
        }
        let mut ts: Box<dyn TokenStream> = Box::new(StandardTokenizer::new());
        if self.ascii_folding {
            ts = Box::new(TermRewriteFilter::ascii_folding(ts));
        }
        ts = Box::new(LowerCaseFilter::new(ts));
        if let Some(stop) = &self.stopwords {
            ts = Box::new(StopFilter::new(ts, Arc::clone(stop)));
        }
        if self.snowball_stemming {
            ts = Box::new(TermRewriteFilter::snowball_english(ts));
        } else if self.stemming {
            ts = Box::new(TermRewriteFilter::porter(ts));
        }
        if let Some(synonyms) = &self.synonyms {
            ts = Box::new(SynonymAdapter::new(
                ts,
                Arc::clone(synonyms),
                self.synonyms_bidirectional,
            ));
        }
        Ok(TokenStreamComponents::new(ts))
    }

    fn normalize(&self, _field_name: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        if self.keyword {
            return input;
        }
        Box::new(LowerCaseFilter::new(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_stream::consume;
    use crate::{StandardAnalyzer, StandardTokenizer};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn terms(a: &Analyzer, field: &str, text: &str) -> Vec<String> {
        let mut ts = a.token_stream(field, text).unwrap();
        let mut out = Vec::new();
        consume(&mut ts, |x| out.push(x.term().to_string())).unwrap();
        out
    }

    /// Counts `create_components` calls; per-field gaps.
    struct Counting(Arc<AtomicUsize>);

    impl AnalyzerDefinition for Counting {
        fn create_components(&self, _f: &str) -> Result<TokenStreamComponents, AnalysisError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(TokenStreamComponents::new(StandardTokenizer::new()))
        }
        fn position_increment_gap(&self, field: &str) -> i32 {
            if field == "wide" {
                100
            } else {
                0
            }
        }
    }

    #[test]
    fn components_are_reused_per_strategy() {
        let n = Arc::new(AtomicUsize::new(0));
        let a = Analyzer::new(Counting(Arc::clone(&n)));
        assert_eq!(a.reuse_strategy(), ReuseStrategy::Global);
        assert_eq!(terms(&a, "f", "a b"), vec!["a", "b"]);
        assert_eq!(terms(&a, "g", "c"), vec!["c"]);
        assert_eq!(
            n.load(Ordering::SeqCst),
            1,
            "global: one set for every field"
        );
        // two live streams need two sets
        let s1 = a.token_stream("f", "x").unwrap();
        let s2 = a.token_stream("f", "y").unwrap();
        drop((s1, s2));
        assert_eq!(n.load(Ordering::SeqCst), 2);

        let n = Arc::new(AtomicUsize::new(0));
        let a = Analyzer::with_reuse_strategy(Counting(Arc::clone(&n)), ReuseStrategy::PerField);
        terms(&a, "f", "a");
        terms(&a, "g", "a");
        terms(&a, "f", "a");
        assert_eq!(n.load(Ordering::SeqCst), 2, "per field: one set per field");
        assert_eq!(a.position_increment_gap_for_field("wide"), 100);
        assert_eq!(a.position_increment_gap_for_field("f"), 0);
        assert_eq!(a.offset_gap_for_field("f"), 1);
    }

    #[test]
    fn close_makes_the_analyzer_unusable() {
        let a = Analyzer::new(StandardAnalyzer::new());
        a.close();
        let err = a.token_stream("f", "x").err().unwrap();
        assert!(matches!(err, AnalysisError::AlreadyClosed(_)), "{err}");
        // a stream dropped after close does not resurrect the pool
        let b = Analyzer::new(StandardAnalyzer::new());
        let ts = b.token_stream("f", "x").unwrap();
        b.close();
        drop(ts);
        assert!(b.token_stream("f", "x").is_err());
    }

    #[test]
    fn an_abandoned_stream_is_closed_and_reusable() {
        let a = Analyzer::new(StandardAnalyzer::new());
        {
            let mut ts = a.token_stream("f", "one two").unwrap();
            ts.reset().unwrap();
            assert!(ts.increment_token().unwrap());
            // dropped mid-stream, never closed
        }
        assert_eq!(terms(&a, "f", "three"), vec!["three"]);
    }

    /// Emits its input's first token, then `extra` empty ones.
    struct Repeat {
        input: Box<dyn TokenStream>,
        n: usize,
        extra: usize,
    }

    impl crate::TokenFilter for Repeat {
        type Input = Box<dyn TokenStream>;
        fn input(&self) -> &Self::Input {
            &self.input
        }
        fn input_mut(&mut self) -> &mut Self::Input {
            &mut self.input
        }
        fn increment(&mut self) -> Result<bool, AnalysisError> {
            self.n += 1;
            if self.n == 1 {
                return self.input.increment_token();
            }
            Ok(self.n <= 1 + self.extra)
        }
    }

    struct Norm(usize);

    impl AnalyzerDefinition for Norm {
        fn create_components(&self, _f: &str) -> Result<TokenStreamComponents, AnalysisError> {
            Ok(TokenStreamComponents::new(StandardTokenizer::new()))
        }
        fn normalize(&self, _f: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
            if self.0 == 0 {
                Box::new(crate::FilteringTokenFilter::new(
                    input,
                    |_: &AttributeSource| false,
                ))
            } else {
                Box::new(Repeat {
                    input,
                    n: 0,
                    extra: self.0,
                })
            }
        }
    }

    #[test]
    fn normalize_runs_the_normalize_chain_over_one_token() {
        let a = Analyzer::new(StandardAnalyzer::new());
        assert_eq!(a.normalize("f", "HeLLo World").unwrap(), b"hello world");
        assert_eq!(a.normalize("f", "").unwrap(), b"");
        // a normalize chain that drops the token, or adds one, is an error
        let err = Analyzer::new(Norm(0)).normalize("f", "x").unwrap_err();
        assert!(err.to_string().contains("but got 0"), "{err}");
        let err = Analyzer::new(Norm(1)).normalize("f", "x").unwrap_err();
        assert!(err.to_string().contains("but got 2+"), "{err}");
    }

    /// Uppercases ASCII letters.
    struct Upper(Box<dyn CharReader>);

    impl crate::CharFilter for Upper {
        fn input(&self) -> &dyn CharReader {
            &*self.0
        }
        fn input_mut(&mut self) -> &mut dyn CharReader {
            &mut *self.0
        }
        fn read_filtered(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
            let n = self.0.read(buf)?;
            for u in &mut buf[..n] {
                if (u16::from(b'a')..=u16::from(b'z')).contains(u) {
                    *u -= 32;
                }
            }
            Ok(n)
        }
        fn correct(&self, off: i32) -> i32 {
            off
        }
    }

    /// Lowercases field "lower"'s tokens; uppercases field "shout"'s reader.
    struct PerField {
        inner: Analyzer,
    }

    impl AnalyzerWrapper for PerField {
        fn wrapped_analyzer(&self, _f: &str) -> &Analyzer {
            &self.inner
        }
        fn wrap_components(&self, field: &str, c: TokenStreamComponents) -> TokenStreamComponents {
            if field != "lower" {
                return c;
            }
            let (source, sink) = c.into_parts();
            TokenStreamComponents::from_parts(source, Box::new(LowerCaseFilter::new(sink)))
        }
        fn wrap_reader(&self, field: &str, r: Box<dyn CharReader>) -> Box<dyn CharReader> {
            if field == "shout" {
                Box::new(Upper(r))
            } else {
                r
            }
        }
        fn wrap_reader_for_normalization(
            &self,
            field: &str,
            r: Box<dyn CharReader>,
        ) -> Box<dyn CharReader> {
            self.wrap_reader(field, r)
        }
        fn wrap_token_stream_for_normalization(
            &self,
            _f: &str,
            input: Box<dyn TokenStream>,
        ) -> Box<dyn TokenStream> {
            input
        }
    }

    #[test]
    fn analyzer_wrapper_wraps_components_readers_and_gaps() {
        let inner = Analyzer::new(Counting(Arc::new(AtomicUsize::new(0))));
        let a = Analyzer::with_reuse_strategy(PerField { inner }, ReuseStrategy::PerField);
        assert_eq!(terms(&a, "lower", "AbC d"), vec!["abc", "d"]);
        assert_eq!(terms(&a, "other", "AbC"), vec!["AbC"]);
        assert_eq!(terms(&a, "shout", "AbC"), vec!["ABC"]);
        assert_eq!(a.position_increment_gap_for_field("wide"), 100);
        assert_eq!(a.offset_gap_for_field("x"), 1);
        assert_eq!(a.normalize("shout", "ab").unwrap(), b"AB");
        assert_eq!(a.normalize("x", "ab").unwrap(), b"ab");
    }

    struct Delegate {
        a: Analyzer,
        b: Analyzer,
    }

    impl DelegatingAnalyzerWrapper for Delegate {
        fn wrapped_analyzer(&self, field: &str) -> &Analyzer {
            if field == "b" {
                &self.b
            } else {
                &self.a
            }
        }
    }

    #[test]
    fn delegating_wrapper_reuses_the_wrapped_components() {
        let na = Arc::new(AtomicUsize::new(0));
        let d = Analyzer::delegating(
            Delegate {
                a: Analyzer::new(Counting(Arc::clone(&na))),
                b: Analyzer::new(StandardAnalyzer::new()),
            },
            ReuseStrategy::PerField,
        );
        assert_eq!(terms(&d, "a", "X y"), vec!["X", "y"]);
        assert_eq!(terms(&d, "a", "z"), vec!["z"]);
        assert_eq!(terms(&d, "b", "X y"), vec!["x", "y"]);
        assert_eq!(na.load(Ordering::SeqCst), 1);
        assert_eq!(d.position_increment_gap_for_field("wide"), 100);
        assert_eq!(d.normalize("b", "Q").unwrap(), b"q");
        // its own components, asked for directly, are the wrapped analyzer's
        let mut c = d.create_components("b").unwrap();
        c.set_reader(Box::new(StrReader::new("Hi"))).unwrap();
        let mut out = Vec::new();
        consume(c.token_stream(), |x| out.push(x.term().to_string())).unwrap();
        assert_eq!(out, vec!["hi"]);
    }

    struct NoTokenizer;

    impl AnalyzerDefinition for NoTokenizer {
        fn create_components(&self, _f: &str) -> Result<TokenStreamComponents, AnalysisError> {
            Ok(TokenStreamComponents::new(StringTokenStream::new(
                "x".into(),
                1,
            )))
        }
    }

    #[test]
    fn components_without_a_tokenizer_cannot_take_a_reader() {
        let err = Analyzer::new(NoTokenizer)
            .token_stream("f", "x")
            .err()
            .unwrap();
        assert!(err.to_string().contains("no Tokenizer source"), "{err}");
    }

    #[test]
    fn materialising_helpers_and_builders() {
        let a = Analyzer::standard(None)
            .with_position_increment_gap(7)
            .with_offset_gap(2);
        assert_eq!(a.position_increment_gap(), 7);
        assert_eq!(a.offset_gap(), 2);
        let s = a.try_analyze_stream("Fox  ").unwrap();
        assert_eq!(s.tokens.len(), 1);
        assert_eq!(s.final_offset, 5);
        let mut seen = Vec::new();
        let end = a
            .try_for_each_token("A b", |t, s, e, i| seen.push((t.to_string(), s, e, i)))
            .unwrap();
        assert_eq!(
            seen,
            vec![("a".to_string(), 0, 1, 1), ("b".to_string(), 2, 3, 1)]
        );
        assert_eq!(end, (0, 3));
        // builders on a definition-backed analyzer are no-ops
        let d = Analyzer::new(StandardAnalyzer::new())
            .with_ascii_folding()
            .with_stemming()
            .with_snowball_stemming()
            .with_synonyms(HashMap::new())
            .with_bidirectional_synonyms(HashMap::new());
        assert_eq!(d.analyze("Café")[0].term, "café");
        // keyword normalize is the identity
        assert_eq!(Analyzer::keyword().normalize("f", "AB").unwrap(), b"AB");
        let k = Analyzer::keyword();
        let mut ts = k.token_stream("f", "q").unwrap();
        assert!(ts.as_tokenizer().is_some());
        ts.close().unwrap();
    }
}
