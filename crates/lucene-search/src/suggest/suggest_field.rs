//! `org.apache.lucene.search.suggest.document.SuggestField`, the
//! `CompletionTokenStream` it indexes through, and `ContextSuggestField`:
//! the value, analyzed and concatenated into one token per path of its token
//! graph (`ConcatenateGraphFilter`), each token carrying the suggestion
//! payload (surface form, weight, type) that `CompletionFieldsConsumer`
//! builds the field's FST from. Route the field to
//! `per_field_postings::Lucene104PostingsFormat::completion104()` for the
//! payloads to become a suggester.

use std::borrow::Cow;

use lucene_analysis::miscellaneous::{
    ConcatenateGraphFilter, DEFAULT_MAX_GRAPH_EXPANSIONS, SEP_LABEL,
};
use lucene_analysis::token_stream::{TokenFilter, TokenStream};
use lucene_analysis::{AnalysisError, Analyzer};
use lucene_codecs::completion::END_BYTE;
use lucene_index::document::{Error, Result};
use lucene_index::document::{FieldToken, FieldTokens, FieldType, IndexableField, TextField};
use lucene_store::DataOutput;

/// `CompletionAnalyzer.HOLE_CHARACTER` (`TokenStreamToAutomaton.HOLE`).
pub const HOLE_CHARACTER: char = '\u{1E}';

/// The `CompletionTokenStream` options a `CompletionAnalyzer` indexes with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompletionOptions {
    /// `preserveSep`.
    pub preserve_sep: bool,
    /// `preservePositionIncrements`.
    pub preserve_position_increments: bool,
    /// `maxGraphExpansions`.
    pub max_graph_expansions: i32,
}

impl Default for CompletionOptions {
    /// `ConcatenateGraphFilter`'s defaults: separators and position
    /// increments preserved, no limit on graph expansions.
    fn default() -> Self {
        CompletionOptions {
            preserve_sep: true,
            preserve_position_increments: true,
            max_graph_expansions: DEFAULT_MAX_GRAPH_EXPANSIONS,
        }
    }
}

/// `SuggestField`: a value to suggest, with its weight.
#[derive(Debug, Clone, PartialEq)]
pub struct SuggestField {
    name: String,
    value: String,
    weight: i32,
    field_type: FieldType,
    options: CompletionOptions,
}

impl SuggestField {
    /// `SuggestField.TYPE`: the type byte of a plain suggestion.
    pub const TYPE: u8 = 0;

    /// `SuggestField.FIELD_TYPE`: tokenized, not stored, norms, docs, freqs
    /// and positions.
    pub fn field_type() -> FieldType {
        TextField::type_not_stored()
    }

    /// `new SuggestField(name, value, weight)`, with its checks: a weight of
    /// at least 0, a non-empty value, and none of the reserved characters
    /// (the token separator, the hole, the end byte) in it.
    pub fn new(name: impl Into<String>, value: impl Into<String>, weight: i32) -> Result<Self> {
        let value = value.into();
        if weight < 0 {
            return Err(Error::IllegalArgument("weight must be >= 0".to_string()));
        }
        if value.is_empty() {
            return Err(Error::IllegalArgument(
                "value must have a length > 0".to_string(),
            ));
        }
        let reserved = [
            u16::from(SEP_LABEL as u8),
            HOLE_CHARACTER as u16,
            u16::from(END_BYTE),
        ];
        if let Some((i, c)) = value
            .encode_utf16()
            .enumerate()
            .find(|(_, c)| reserved.contains(c))
        {
            return Err(Error::IllegalArgument(format!(
                "Illegal input [{value}] UTF-16 codepoint [0x{c:x}] at position {i} is a \
                 reserved character"
            )));
        }
        Ok(SuggestField {
            name: name.into(),
            value,
            weight,
            field_type: Self::field_type(),
            options: CompletionOptions::default(),
        })
    }

    /// Indexing through a `CompletionAnalyzer` with these options rather
    /// than a plain analyzer (whose stream `wrapTokenStream` wraps with the
    /// defaults).
    pub fn with_options(mut self, options: CompletionOptions) -> Self {
        self.options = options;
        self
    }

    /// The options the value is indexed with.
    pub fn options(&self) -> CompletionOptions {
        self.options
    }

    /// `buildSuggestPayload()`: vint surface length, the surface bytes,
    /// vint `weight + 1`, the type byte.
    pub fn payload(&self) -> Vec<u8> {
        self.payload_of_type(Self::TYPE)
    }

    fn payload_of_type(&self, type_byte: u8) -> Vec<u8> {
        let mut out = Vec::new();
        out.write_vint(i32::try_from(self.value.len()).unwrap_or(i32::MAX));
        out.extend_from_slice(self.value.as_bytes());
        out.write_vint(self.weight.wrapping_add(1));
        out.push(type_byte);
        out
    }

    /// `tokenStream(analyzer, reuse)` over `input` (the analyzer's stream,
    /// possibly wrapped): a `CompletionTokenStream`, every token given
    /// `payload`.
    fn tokens(&self, input: impl TokenStream, payload: Vec<u8>) -> Result<Option<FieldTokens>> {
        let analysis = |e: AnalysisError| Error::IllegalArgument(e.to_string());
        let o = self.options;
        let mut ts = ConcatenateGraphFilter::with_options(
            input,
            o.preserve_sep.then_some(SEP_LABEL),
            o.preserve_position_increments,
            o.max_graph_expansions,
        );
        ts.reset().map_err(analysis)?;
        let mut tokens = Vec::new();
        while ts.increment_token().map_err(analysis)? {
            let a = ts.attributes();
            tokens.push(FieldToken {
                term: a.term_bytes().to_vec(),
                start_offset: a.start_offset(),
                end_offset: a.end_offset(),
                position_increment: a.position_increment(),
                term_frequency: a.term_frequency(),
                payload: Some(payload.clone()),
            });
        }
        ts.end().map_err(analysis)?;
        let end = ts.attributes().clone();
        ts.close().map_err(analysis)?;
        Ok(Some(FieldTokens {
            tokens,
            final_position_increment: end.position_increment(),
            final_offset: end.end_offset(),
            end_attributes: Some(end),
        }))
    }
}

impl IndexableField for SuggestField {
    fn name(&self) -> &str {
        &self.name
    }

    fn field_type(&self) -> &FieldType {
        &self.field_type
    }

    fn string_value(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.value))
    }

    /// `tokenStream(analyzer, reuse)`: the analyzer's stream through a
    /// `CompletionTokenStream`, every token given the suggestion payload.
    fn token_stream(&self, analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
        let input = analyzer
            .token_stream(&self.name, &self.value)
            .map_err(|e| Error::IllegalArgument(e.to_string()))?;
        self.tokens(input, self.payload())
    }
}

/// `ContextSuggestField`: a [`SuggestField`] whose value is indexed behind
/// each of its contexts (and once behind no context), for `ContextQuery` to
/// filter and boost by.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextSuggestField {
    field: SuggestField,
    contexts: Vec<String>,
}

impl ContextSuggestField {
    /// `ContextSuggestField.CONTEXT_SEPARATOR`: between a context and the
    /// value.
    pub const CONTEXT_SEPARATOR: char = '\u{1D}';

    /// `ContextSuggestField.TYPE`.
    pub const TYPE: u8 = 1;

    /// `new ContextSuggestField(name, value, weight, contexts...)`: the
    /// [`SuggestField`] checks, and no context separator in the value or a
    /// context. Contexts are a set, as in Java.
    pub fn new(
        name: impl Into<String>,
        value: impl Into<String>,
        weight: i32,
        contexts: &[&str],
    ) -> Result<Self> {
        let field = SuggestField::new(name, value, weight)?;
        validate(&field.value)?;
        let mut set: Vec<String> = Vec::new();
        for &c in contexts {
            validate(c)?;
            if !set.iter().any(|s| s == c) {
                set.push(c.to_string());
            }
        }
        Ok(ContextSuggestField {
            field,
            contexts: set,
        })
    }

    /// [`SuggestField::with_options`].
    pub fn with_options(mut self, options: CompletionOptions) -> Self {
        self.field = self.field.with_options(options);
        self
    }

    /// `contexts()`.
    pub fn contexts(&self) -> &[String] {
        &self.contexts
    }

    /// `buildSuggestPayload()` with this field's type.
    pub fn payload(&self) -> Vec<u8> {
        self.field.payload_of_type(Self::TYPE)
    }
}

/// `ContextSuggestField.validate`.
fn validate(value: &str) -> Result<()> {
    let separator = ContextSuggestField::CONTEXT_SEPARATOR as u16;
    if let Some((i, c)) = value
        .encode_utf16()
        .enumerate()
        .find(|&(_, c)| c == separator)
    {
        return Err(Error::IllegalArgument(format!(
            "Illegal value [{value}] UTF-16 codepoint [0x{c:x}] at position {i} is a reserved \
             character"
        )));
    }
    Ok(())
}

impl IndexableField for ContextSuggestField {
    fn name(&self) -> &str {
        &self.field.name
    }

    fn field_type(&self) -> &FieldType {
        &self.field.field_type
    }

    fn string_value(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.field.value))
    }

    /// `wrapTokenStream`: a `PrefixTokenFilter` puts the contexts ahead of
    /// the analyzer's tokens before the `CompletionTokenStream`.
    fn token_stream(&self, analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
        let input = analyzer
            .token_stream(&self.field.name, &self.field.value)
            .map_err(|e| Error::IllegalArgument(e.to_string()))?;
        let prefixed = PrefixTokenFilter {
            input,
            prefixes: self.contexts.clone(),
            current: None,
        };
        self.field.tokens(prefixed, self.payload())
    }
}

/// `ContextSuggestField.PrefixTokenFilter`: each prefix followed by the
/// separator, as tokens at one position ahead of the input's (the first with
/// an increment of 1, the rest 0); with no prefix, a lone separator.
struct PrefixTokenFilter<I> {
    input: I,
    prefixes: Vec<String>,
    /// `currentPrefix`: the next prefix's index, `None` before the first.
    current: Option<usize>,
}

impl<I: TokenStream> TokenFilter for PrefixTokenFilter<I> {
    type Input = I;

    fn input(&self) -> &I {
        &self.input
    }

    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    fn increment(&mut self) -> std::result::Result<bool, AnalysisError> {
        let next = match self.current {
            Some(i) if i >= self.prefixes.len() => return self.input.increment_token(),
            Some(i) => {
                self.input.attributes_mut().set_position_increment(0)?;
                i
            }
            None => {
                self.input.attributes_mut().set_position_increment(1)?;
                0
            }
        };
        let mut term = String::new();
        if let Some(prefix) = self.prefixes.get(next) {
            term.push_str(prefix);
        }
        term.push(ContextSuggestField::CONTEXT_SEPARATOR);
        self.current = Some(next.saturating_add(1));
        self.input.attributes_mut().set_term(&term);
        Ok(true)
    }

    fn reset_filter(&mut self) -> std::result::Result<(), AnalysisError> {
        self.input.reset()?;
        self.current = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_payload_is_surface_weight_and_type() {
        let f = SuggestField::new("s", "café", 7).unwrap();
        assert_eq!(f.payload(), [5, b'c', b'a', b'f', 0xC3, 0xA9, 8, 0]);
        assert_eq!(f.name(), "s");
        assert_eq!(f.string_value().as_deref(), Some("café"));
        assert_eq!(f.field_type(), &SuggestField::field_type());
    }

    #[test]
    fn invalid_values_are_refused() {
        assert!(SuggestField::new("s", "a", -1).is_err());
        assert!(SuggestField::new("s", "", 1).is_err());
        let e = SuggestField::new("s", "ab\u{1F}", 1)
            .unwrap_err()
            .to_string();
        assert!(e.contains("[0x1f] at position 2"), "{e}");
        assert!(SuggestField::new("s", "\u{1E}", 1).is_err());
        assert!(SuggestField::new("s", "a\0", 1).is_err());
        // Positions count UTF-16 units, as `String.charAt` does.
        let e = SuggestField::new("s", "😀\u{1F}", 1)
            .unwrap_err()
            .to_string();
        assert!(e.contains("at position 2"), "{e}");
        assert!(SuggestField::new("s", "x", i32::MAX).is_ok());
    }

    #[test]
    fn one_token_per_path_each_with_the_payload() {
        let f = SuggestField::new("s", "new york", 3).unwrap();
        let tokens = f.token_stream(&Analyzer::keyword()).unwrap().unwrap();
        assert_eq!(tokens.tokens.len(), 1);
        assert_eq!(tokens.tokens[0].term, b"new york");
        assert_eq!(tokens.tokens[0].payload.as_deref(), Some(&f.payload()[..]));
    }

    #[test]
    fn options_change_how_the_value_is_concatenated() {
        let analyzer = Analyzer::standard(None);
        let f = SuggestField::new("s", "new york", 3).unwrap();
        assert_eq!(f.options(), CompletionOptions::default());
        let sep = f.token_stream(&analyzer).unwrap().unwrap();
        assert_eq!(sep.tokens[0].term, b"new\x1fyork");
        let f = f.with_options(CompletionOptions {
            preserve_sep: false,
            ..CompletionOptions::default()
        });
        let joined = f.token_stream(&analyzer).unwrap().unwrap();
        assert_eq!(joined.tokens[0].term, b"newyork");
    }

    #[test]
    fn a_context_field_puts_each_context_ahead_of_the_value() {
        let f =
            ContextSuggestField::new("s", "new york", 3, &["music", "movies", "music"]).unwrap();
        assert_eq!(f.contexts(), ["music", "movies"]);
        assert_eq!(f.name(), "s");
        assert_eq!(f.string_value().as_deref(), Some("new york"));
        assert_eq!(f.field_type(), &SuggestField::field_type());
        assert_eq!(*f.payload().last().unwrap(), ContextSuggestField::TYPE);
        let tokens = f.token_stream(&Analyzer::standard(None)).unwrap().unwrap();
        let mut terms: Vec<&[u8]> = tokens.tokens.iter().map(|t| &t.term[..]).collect();
        terms.sort();
        assert_eq!(
            terms,
            [
                &b"movies\x1d\x1fnew\x1fyork"[..],
                &b"music\x1d\x1fnew\x1fyork"[..]
            ]
        );
        for t in &tokens.tokens {
            assert_eq!(t.payload.as_deref(), Some(&f.payload()[..]));
        }
    }

    #[test]
    fn a_context_field_without_contexts_has_a_lone_separator() {
        let f = ContextSuggestField::new("s", "x", 1, &[]).unwrap();
        let tokens = f.token_stream(&Analyzer::keyword()).unwrap().unwrap();
        assert_eq!(tokens.tokens.len(), 1);
        assert_eq!(tokens.tokens[0].term, b"\x1d\x1fx");
        let f = f.with_options(CompletionOptions::default());
        assert!(f.contexts().is_empty());
    }

    #[test]
    fn context_separators_are_refused_in_values_and_contexts() {
        let e = ContextSuggestField::new("s", "a\u{1d}", 1, &[])
            .unwrap_err()
            .to_string();
        assert!(e.contains("[0x1d] at position 1"), "{e}");
        assert!(ContextSuggestField::new("s", "a", 1, &["b\u{1d}"]).is_err());
        assert!(ContextSuggestField::new("s", "a", -1, &[]).is_err());
    }
}
