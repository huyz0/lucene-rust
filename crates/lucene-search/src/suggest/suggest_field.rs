//! `org.apache.lucene.search.suggest.document.SuggestField` and the
//! `CompletionTokenStream` it indexes through: the value, analyzed and
//! concatenated into one token per path of its token graph
//! (`ConcatenateGraphFilter`), each token carrying the suggestion payload
//! (surface form, weight, type) that `CompletionFieldsConsumer` builds the
//! field's FST from. Route the field to
//! `per_field_postings::Lucene104PostingsFormat::completion104()` for the
//! payloads to become a suggester.

use std::borrow::Cow;

use lucene_analysis::miscellaneous::{ConcatenateGraphFilter, SEP_LABEL};
use lucene_analysis::token_stream::TokenStream;
use lucene_analysis::Analyzer;
use lucene_codecs::completion::END_BYTE;
use lucene_index::document::{Error, Result};
use lucene_index::document::{FieldToken, FieldTokens, FieldType, IndexableField, TextField};
use lucene_store::DataOutput;

/// `CompletionAnalyzer.HOLE_CHARACTER` (`TokenStreamToAutomaton.HOLE`).
pub const HOLE_CHARACTER: char = '\u{1E}';

/// `SuggestField`: a value to suggest, with its weight.
#[derive(Debug, Clone, PartialEq)]
pub struct SuggestField {
    name: String,
    value: String,
    weight: i32,
    field_type: FieldType,
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
        })
    }

    /// `buildSuggestPayload()`: vint surface length, the surface bytes,
    /// vint `weight + 1`, the type byte.
    pub fn payload(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.write_vint(i32::try_from(self.value.len()).unwrap_or(i32::MAX));
        out.extend_from_slice(self.value.as_bytes());
        out.write_vint(self.weight.wrapping_add(1));
        out.push(Self::TYPE);
        out
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
    /// `CompletionTokenStream` (`ConcatenateGraphFilter` with its defaults),
    /// every token given the suggestion payload.
    fn token_stream(&self, analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
        let analysis = |e: lucene_analysis::AnalysisError| Error::IllegalArgument(e.to_string());
        let input = analyzer
            .token_stream(&self.name, &self.value)
            .map_err(analysis)?;
        let mut ts = ConcatenateGraphFilter::new(input);
        let payload = self.payload();
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
}
