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

use std::borrow::Cow;
use std::sync::Arc;

use lucene_analysis::{
    AnalysisError, Analyzer, AnalyzerDefinition, KeywordTokenizer, TokenFilter, TokenStream,
    TokenStreamComponents,
};
use lucene_index::document::SortedDocValuesField;
use lucene_index::document::{FieldTokens, FieldType, IndexableField};

use crate::icu4j::coll::collator::{Collator, KeyScratch};

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
            scratch: KeyScratch::default(),
        }
    }
}

/// `ICUCollatedTermAttributeImpl`: each token's binary term is the
/// collation key of its term.
pub struct ICUCollatedTermFilter<I> {
    input: I,
    collator: Arc<Collator>,
    scratch: KeyScratch,
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
        // The key replaces the previous token's bytes in place.
        let (term, key) = self.input.attributes_mut().term_and_bytes_term_mut();
        self.collator
            .raw_collation_key_into(term, &mut self.scratch, key);
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
/// the collation key of a string (`setStringValue`). It is a
/// [`SortedDocValuesField`] whose bytes the collator sets, so a document
/// indexes it through [`IndexableField`] like any other field.
#[derive(Debug, Clone)]
pub struct ICUCollationDocValuesField {
    collator: Collator,
    field: SortedDocValuesField,
}

impl ICUCollationDocValuesField {
    /// `ICUCollationDocValuesField(name, collator)`: no value until one is
    /// set.
    pub fn new(name: &str, collator: Collator) -> Self {
        ICUCollationDocValuesField {
            collator,
            field: SortedDocValuesField::new(name, Vec::new()),
        }
    }

    /// `name()`.
    pub fn name(&self) -> &str {
        IndexableField::name(&self.field)
    }

    /// `setStringValue(value)`: the value's collation key.
    pub fn set_string_value(&mut self, value: &str) {
        self.field.set_value(self.collator.raw_collation_key(value));
    }

    /// `binaryValue()` (empty before a value is set).
    pub fn binary_value(&self) -> &[u8] {
        self.field.value()
    }
}

impl IndexableField for ICUCollationDocValuesField {
    fn name(&self) -> &str {
        IndexableField::name(&self.field)
    }
    fn field_type(&self) -> &FieldType {
        self.field.field_type()
    }
    fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
        self.field.binary_value()
    }
    fn token_stream(
        &self,
        analyzer: &Analyzer,
    ) -> lucene_index::document::Result<Option<FieldTokens>> {
        self.field.token_stream(analyzer)
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

    /// `ICUCollationDocValuesField` is a sorted doc-values field whose value
    /// is the collation key: what a document indexes, as `IndexableField`.
    #[test]
    fn the_field_is_a_sorted_doc_values_field_of_the_key() {
        use lucene_index::document::{DocValuesType, IndexableField};
        let root = Collator::root().unwrap();
        let key = root.raw_collation_key("abc");
        let mut f = ICUCollationDocValuesField::new("sort", root);
        assert_eq!(IndexableField::name(&f), "sort");
        assert_eq!(f.name(), "sort");
        assert_eq!(f.field_type().doc_values_type(), DocValuesType::Sorted);
        assert_eq!(IndexableField::binary_value(&f).unwrap().len(), 0);
        f.set_string_value("abc");
        assert_eq!(IndexableField::binary_value(&f).unwrap().as_ref(), &key[..]);
        assert_eq!(f.binary_value(), &key[..]);
    }
}
