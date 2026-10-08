//! `org.apache.lucene.analysis.icu.{ICUCollationAttributeFactory,
//! ICUCollationKeyAnalyzer, ICUCollationDocValuesField}` and
//! `tokenattributes.ICUCollatedTermAttributeImpl`: terms indexed as ICU
//! collation sort keys, so that range queries and sorting over the field
//! follow the collator's order.
//!
//! Java's attribute factory swaps the term attribute's implementation for
//! one whose `getBytesRef()` is `collator.getRawCollationKey(term)`. The
//! Rust `AttributeSource` has no pluggable implementations, so
//! [`ICUCollationAttributeFactory::wrap`] wraps a stream in
//! [`ICUCollatedTermFilter`], which sets the binary term
//! (`AttributeSource::set_bytes_term`) to the key of each token's term --
//! the bytes a consumer of `TermToBytesRefAttribute` reads, as in Java.
//! It must be the outermost stream (a filter after it could change the
//! term without recomputing the key; Java's attribute recomputes on every
//! `getBytesRef`). Terms are UTF-8 here, so a lone surrogate (U+FFFD in
//! the term) is keyed as U+FFFD, where Java keys the surrogate.

use std::sync::Arc;

use lucene_analysis::{
    AnalysisError, AnalyzerDefinition, KeywordTokenizer, TokenFilter, TokenStream,
    TokenStreamComponents,
};

use crate::icu4j::coll::collator::Collator;

/// `ICUCollationAttributeFactory`.
#[derive(Debug, Clone)]
pub struct ICUCollationAttributeFactory {
    collator: Arc<Collator>,
}

impl ICUCollationAttributeFactory {
    /// `ICUCollationAttributeFactory(collator)`.
    pub fn new(collator: Collator) -> Self {
        ICUCollationAttributeFactory {
            collator: Arc::new(collator),
        }
    }

    /// The collator.
    pub fn collator(&self) -> &Collator {
        &self.collator
    }

    /// The stream with its terms' bytes the collation keys
    /// (`createInstance()`'s attribute, for the whole stream).
    pub fn wrap<I: TokenStream>(&self, input: I) -> ICUCollatedTermFilter<I> {
        ICUCollatedTermFilter {
            input,
            collator: self.collator.clone(),
        }
    }
}

/// `ICUCollatedTermAttributeImpl`: each token's binary term is the
/// collation key of its term.
pub struct ICUCollatedTermFilter<I> {
    input: I,
    collator: Arc<Collator>,
}

impl<I: TokenStream> TokenFilter for ICUCollatedTermFilter<I> {
    type Input = I;

    fn input(&self) -> &I {
        &self.input
    }

    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let atts = self.input.attributes_mut();
        let key = self.collator.raw_collation_key(atts.term());
        atts.set_bytes_term(Some(key));
        Ok(true)
    }
}

/// `ICUCollationKeyAnalyzer`: the whole text as one token
/// (`KeywordTokenizer`), indexed as its collation key.
#[derive(Debug, Clone)]
pub struct ICUCollationKeyAnalyzer {
    factory: ICUCollationAttributeFactory,
}

impl ICUCollationKeyAnalyzer {
    /// `ICUCollationKeyAnalyzer(collator)`.
    pub fn new(collator: Collator) -> Self {
        ICUCollationKeyAnalyzer {
            factory: ICUCollationAttributeFactory::new(collator),
        }
    }
}

impl AnalyzerDefinition for ICUCollationKeyAnalyzer {
    fn create_components(&self, _field_name: &str) -> Result<TokenStreamComponents, AnalysisError> {
        Ok(TokenStreamComponents::new(
            self.factory.wrap(KeywordTokenizer::new()),
        ))
    }
}

/// `ICUCollationDocValuesField`: a sorted doc values field whose value is
/// the collation key of a string. Lucene's `Field` belongs to the search
/// crate, above this one; this type holds the name and computes the bytes
/// (`setStringValue`, then `binaryValue()`), which a caller indexes as a
/// `SortedDocValuesField`.
#[derive(Debug, Clone)]
pub struct ICUCollationDocValuesField {
    name: String,
    collator: Collator,
    bytes: Vec<u8>,
}

impl ICUCollationDocValuesField {
    /// `ICUCollationDocValuesField(name, collator)`.
    pub fn new(name: &str, collator: Collator) -> Self {
        ICUCollationDocValuesField {
            name: name.to_string(),
            collator,
            bytes: Vec::new(),
        }
    }

    /// `name()`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `setStringValue(value)`.
    pub fn set_string_value(&mut self, value: &str) {
        self.bytes = self.collator.raw_collation_key(value);
    }

    /// `binaryValue()` (`fieldsData`, empty before a value is set).
    pub fn binary_value(&self) -> &[u8] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::Analyzer;

    #[test]
    fn keys_as_terms() {
        let root = Collator::root().unwrap();
        let a = Analyzer::new(ICUCollationKeyAnalyzer::new(root.clone()));
        let mut ts = a.token_stream("f", "Hello world").unwrap();
        ts.reset().unwrap();
        assert!(ts.increment_token().unwrap());
        let bytes = ts.attributes().term_bytes().to_vec();
        assert_eq!(bytes, root.raw_collation_key("Hello world"));
        assert!(!ts.increment_token().unwrap());
        ts.end().unwrap();
        let mut f = ICUCollationDocValuesField::new("f", root.clone());
        assert_eq!(f.name(), "f");
        assert!(f.binary_value().is_empty());
        f.set_string_value("Hello world");
        assert_eq!(f.binary_value(), &bytes[..]);
        let factory = ICUCollationAttributeFactory::new(root);
        assert_eq!(
            factory.collator().strength(),
            crate::icu4j::coll::collator::TERTIARY
        );
    }
}
