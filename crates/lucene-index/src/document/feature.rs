//! `FeatureField`: a positive float feature of a document, indexed as the
//! term frequency of a term named after the feature.
//!
//! What reaches disk: the term `featureName` in field `fieldName`
//! (`DOCS_AND_FREQS`, no norms, untokenized), with a custom frequency of
//! `Float.floatToIntBits(featureValue) >>> 15` -- the float's top 17 bits, so
//! the frequency orders as the value does and `decodeFeatureValue` recovers
//! the value to 8 significant bits.
//!
//! The queries (`newLinearQuery`, `newLogQuery`, `newSaturationQuery`,
//! `newSigmoidQuery`), `newFeatureSort` and `newDoubleValues` live in
//! `lucene_search::document`, over [`FeatureField::decode_feature_value`].

use std::borrow::Cow;

use lucene_analysis::Analyzer;

use super::{illegal, FieldToken, FieldTokens, FieldType, IndexOptions, IndexableField, Result};

/// `FeatureField`.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureField {
    name: String,
    field_type: FieldType,
    feature_name: String,
    feature_value: f32,
}

impl FeatureField {
    /// `MAX_FREQ`: the frequency of `Float.MAX_VALUE`.
    pub const MAX_FREQ: i32 = (f32::MAX.to_bits() >> 15) as i32;

    /// `MAX_WEIGHT`: the largest query weight (`Long.SIZE`).
    pub const MAX_WEIGHT: f32 = 64.0;

    /// `FIELD_TYPE` (or `FIELD_TYPE_STORE_TERM_VECTORS`).
    pub fn field_type_of(store_term_vectors: bool) -> FieldType {
        let mut ft = FieldType::new();
        ft.set_tokenized(false).expect("unfrozen");
        ft.set_omit_norms(true).expect("unfrozen");
        ft.set_index_options(IndexOptions::DocsAndFreqs)
            .expect("unfrozen");
        if store_term_vectors {
            ft.set_store_term_vectors(true).expect("unfrozen");
        }
        ft.frozen()
    }

    /// `new FeatureField(fieldName, featureName, featureValue)`.
    pub fn new(
        field_name: impl Into<String>,
        feature_name: impl Into<String>,
        feature_value: f32,
    ) -> Result<Self> {
        Self::with_term_vectors(field_name, feature_name, feature_value, false)
    }

    /// `new FeatureField(fieldName, featureName, featureValue,
    /// storeTermVectors)`.
    pub fn with_term_vectors(
        field_name: impl Into<String>,
        feature_name: impl Into<String>,
        feature_value: f32,
        store_term_vectors: bool,
    ) -> Result<Self> {
        let mut f = FeatureField {
            name: field_name.into(),
            field_type: Self::field_type_of(store_term_vectors),
            feature_name: feature_name.into(),
            feature_value: 1.0,
        };
        f.set_feature_value(feature_value)?;
        Ok(f)
    }

    /// `setFeatureValue`: finite and at least `Float.MIN_NORMAL`.
    pub fn set_feature_value(&mut self, feature_value: f32) -> Result<()> {
        if !feature_value.is_finite() {
            return Err(illegal(format!(
                "featureValue must be finite, got: {feature_value} for feature {} on field {}",
                self.feature_name, self.name
            )));
        }
        if feature_value < f32::MIN_POSITIVE {
            return Err(illegal(format!(
                "featureValue must be a positive normal float, got: {feature_value} for feature \
                 {} on field {} which is less than the minimum positive normal float: {:e}",
                self.feature_name,
                self.name,
                f32::MIN_POSITIVE
            )));
        }
        self.feature_value = feature_value;
        Ok(())
    }

    /// `getFeatureValue()`.
    pub fn feature_value(&self) -> f32 {
        self.feature_value
    }

    /// The feature (the term).
    pub fn feature_name(&self) -> &str {
        &self.feature_name
    }

    /// The indexed term frequency: `floatToIntBits(featureValue) >>> 15`.
    pub fn term_frequency(&self) -> i32 {
        (self.feature_value.to_bits() >> 15) as i32
    }

    /// `decodeFeatureValue(freq)`: the value a frequency encodes, saturated
    /// at `Float.MAX_VALUE`.
    pub fn decode_feature_value(freq: f32) -> f32 {
        if freq > Self::MAX_FREQ as f32 {
            return f32::MAX;
        }
        // Lossless: `freq` is an integral value below 2^24 here.
        let tf = freq as i32;
        f32::from_bits((tf as u32).wrapping_shl(15))
    }
}

impl IndexableField for FeatureField {
    fn name(&self) -> &str {
        &self.name
    }
    fn field_type(&self) -> &FieldType {
        &self.field_type
    }
    /// `fieldsData`: the feature name.
    fn string_value(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.feature_name))
    }
    /// `FeatureTokenStream`: the feature name, once, at the encoded
    /// frequency. The stream sets no offsets.
    fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
        let mut token = FieldToken::new(self.feature_name.as_bytes().to_vec(), 0, 0);
        token.term_frequency = self.term_frequency();
        Ok(Some(FieldTokens {
            tokens: vec![token],
            final_position_increment: 0,
            final_offset: 0,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequency_encodes_the_top_bits() {
        let f = FeatureField::new("features", "pagerank", 10.0).unwrap();
        assert_eq!(f.term_frequency(), (10.0f32.to_bits() >> 15) as i32);
        assert_eq!(
            FeatureField::decode_feature_value(f.term_frequency() as f32),
            10.0
        );
        assert_eq!(
            FeatureField::decode_feature_value(FeatureField::MAX_FREQ as f32 + 1.0),
            f32::MAX
        );
        let t = f.token_stream(&Analyzer::keyword()).unwrap().unwrap();
        assert_eq!(t.tokens[0].term, b"pagerank");
        assert_eq!(t.tokens[0].term_frequency, f.term_frequency());
        assert_eq!(f.string_value().as_deref(), Some("pagerank"));
        assert_eq!(f.feature_name(), "pagerank");
        assert_eq!(f.feature_value(), 10.0);
        assert_eq!(f.name(), "features");
        assert!(f.field_type().omit_norms());
        assert!(FeatureField::with_term_vectors("f", "x", 1.0, true)
            .unwrap()
            .field_type()
            .store_term_vectors());
    }

    #[test]
    fn values_must_be_positive_normal_finite() {
        assert!(FeatureField::new("f", "x", f32::NAN).is_err());
        assert!(FeatureField::new("f", "x", f32::INFINITY).is_err());
        assert!(FeatureField::new("f", "x", 0.0).is_err());
        assert!(FeatureField::new("f", "x", -1.0).is_err());
        assert!(FeatureField::new("f", "x", f32::MIN_POSITIVE / 2.0).is_err());
        assert!(FeatureField::new("f", "x", f32::MIN_POSITIVE).is_ok());
        let mut f = FeatureField::new("f", "x", 1.0).unwrap();
        assert!(f.set_feature_value(-2.0).is_err());
        assert_eq!(f.feature_value(), 1.0);
    }
}
