//! The doc-values fields -- `NumericDocValuesField`,
//! `SortedNumericDocValuesField`, `SortedDocValuesField`,
//! `SortedSetDocValuesField`, `BinaryDocValuesField` -- and `KeywordField`,
//! which indexes one untokenized term and a `SORTED_SET` doc value of the same
//! bytes.
//!
//! Each `indexedField(...)` variant adds a doc-values skip index
//! (`DocValuesSkipIndexType.RANGE`); the value written is the same.
//!
//! The slow doc-values queries these classes create (`newSlowRangeQuery`,
//! `newSlowExactQuery`, `newSlowSetQuery`) and `KeywordField`'s queries and
//! sort field live in `lucene_search::document`.

use std::borrow::Cow;

use lucene_analysis::Analyzer;

use super::{
    DocValuesSkipIndexType, DocValuesType, FieldTokens, FieldType, IndexOptions, IndexableField,
    InvertableType, Number, Result, Store, StoredValue,
};

fn dv_type(dv: DocValuesType, indexed: bool) -> FieldType {
    let mut ft = FieldType::new();
    ft.set_doc_values_type(dv).expect("unfrozen");
    if indexed {
        ft.set_doc_values_skip_index_type(DocValuesSkipIndexType::Range)
            .expect("unfrozen");
    }
    ft.frozen()
}

macro_rules! long_dv_field {
    ($(#[$m:meta])* $ty:ident, $dv:expr) => {
        $(#[$m])*
        #[derive(Debug, Clone, PartialEq)]
        pub struct $ty {
            name: String,
            field_type: FieldType,
            value: i64,
        }

        impl $ty {
            /// `new XxxDocValuesField(name, value)`.
            pub fn new(name: impl Into<String>, value: i64) -> Self {
                $ty {
                    name: name.into(),
                    field_type: Self::field_type_of(false),
                    value,
                }
            }

            /// `indexedField(name, value)`: the same value, with a skip
            /// index.
            pub fn indexed_field(name: impl Into<String>, value: i64) -> Self {
                $ty {
                    name: name.into(),
                    field_type: Self::field_type_of(true),
                    value,
                }
            }

            /// `TYPE` (or, `indexed`, the skip-indexed type).
            pub fn field_type_of(indexed: bool) -> FieldType {
                dv_type($dv, indexed)
            }

            pub fn value(&self) -> i64 {
                self.value
            }

            /// `setLongValue`.
            pub fn set_value(&mut self, value: i64) {
                self.value = value;
            }
        }

        impl IndexableField for $ty {
            fn name(&self) -> &str {
                &self.name
            }
            fn field_type(&self) -> &FieldType {
                &self.field_type
            }
            fn numeric_value(&self) -> Option<Number> {
                Some(Number::Long(self.value))
            }
            fn string_value(&self) -> Option<Cow<'_, str>> {
                Some(Cow::Owned(self.value.to_string()))
            }
            fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
                Ok(None)
            }
        }
    };
}

long_dv_field!(
    /// `NumericDocValuesField`: one `long` per document.
    NumericDocValuesField,
    DocValuesType::Numeric
);
long_dv_field!(
    /// `SortedNumericDocValuesField`: any number of `long`s per document,
    /// read back sorted.
    SortedNumericDocValuesField,
    DocValuesType::SortedNumeric
);

macro_rules! bytes_dv_field {
    ($(#[$m:meta])* $ty:ident, $dv:expr) => {
        $(#[$m])*
        #[derive(Debug, Clone, PartialEq)]
        pub struct $ty {
            name: String,
            field_type: FieldType,
            value: Vec<u8>,
        }

        impl $ty {
            /// `new XxxDocValuesField(name, bytes)`.
            pub fn new(name: impl Into<String>, value: impl Into<Vec<u8>>) -> Self {
                $ty {
                    name: name.into(),
                    field_type: dv_type($dv, false),
                    value: value.into(),
                }
            }

            pub fn value(&self) -> &[u8] {
                &self.value
            }

            /// `setBytesValue`.
            pub fn set_value(&mut self, value: impl Into<Vec<u8>>) {
                self.value = value.into();
            }
        }

        impl IndexableField for $ty {
            fn name(&self) -> &str {
                &self.name
            }
            fn field_type(&self) -> &FieldType {
                &self.field_type
            }
            fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
                Some(Cow::Borrowed(&self.value))
            }
            fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
                Ok(None)
            }
        }
    };
}

bytes_dv_field!(
    /// `SortedDocValuesField`: one `BytesRef` per document, deduplicated
    /// into sorted ordinals.
    SortedDocValuesField,
    DocValuesType::Sorted
);
bytes_dv_field!(
    /// `SortedSetDocValuesField`: a set of `BytesRef`s per document.
    SortedSetDocValuesField,
    DocValuesType::SortedSet
);
bytes_dv_field!(
    /// `BinaryDocValuesField`: one opaque `BytesRef` per document.
    BinaryDocValuesField,
    DocValuesType::Binary
);

impl SortedDocValuesField {
    /// `indexedField(name, bytes)`: with a skip index.
    pub fn indexed_field(name: impl Into<String>, value: impl Into<Vec<u8>>) -> Self {
        SortedDocValuesField {
            name: name.into(),
            field_type: dv_type(DocValuesType::Sorted, true),
            value: value.into(),
        }
    }
}

impl SortedSetDocValuesField {
    /// `indexedField(name, bytes)`: with a skip index.
    pub fn indexed_field(name: impl Into<String>, value: impl Into<Vec<u8>>) -> Self {
        SortedSetDocValuesField {
            name: name.into(),
            field_type: dv_type(DocValuesType::SortedSet, true),
            value: value.into(),
        }
    }
}

impl BinaryDocValuesField {
    /// A binary doc-values field over the given type -- the
    /// `BinaryDocValuesField` subclasses' constructor.
    pub(crate) fn with_type(name: String, field_type: FieldType, value: Vec<u8>) -> Self {
        BinaryDocValuesField {
            name,
            field_type,
            value,
        }
    }
}

/// `KeywordField`: an untokenized, docs-only, norms-free term and a
/// `SORTED_SET` doc value of the same bytes; stored when asked.
#[derive(Debug, Clone, PartialEq)]
pub struct KeywordField {
    name: String,
    field_type: FieldType,
    /// `fieldsData`: a `String` or a `BytesRef`.
    string: Option<String>,
    binary: Vec<u8>,
    stored: Option<StoredValue>,
}

impl KeywordField {
    /// `KeywordField`'s `FIELD_TYPE` (or `FIELD_TYPE_STORED`).
    pub fn field_type_of(store: Store) -> FieldType {
        let mut ft = FieldType::new();
        ft.set_index_options(IndexOptions::Docs).expect("unfrozen");
        ft.set_omit_norms(true).expect("unfrozen");
        ft.set_tokenized(false).expect("unfrozen");
        ft.set_doc_values_type(DocValuesType::SortedSet)
            .expect("unfrozen");
        if store == Store::Yes {
            ft.set_stored(true).expect("unfrozen");
        }
        ft.frozen()
    }

    /// `new KeywordField(name, String value, store)`: stored as a string.
    pub fn new(name: impl Into<String>, value: impl Into<String>, store: Store) -> Self {
        let value = value.into();
        KeywordField {
            name: name.into(),
            field_type: Self::field_type_of(store),
            binary: value.as_bytes().to_vec(),
            stored: (store == Store::Yes).then(|| StoredValue::String(value.clone())),
            string: Some(value),
        }
    }

    /// `new KeywordField(name, BytesRef value, store)`: stored as bytes.
    pub fn from_bytes(name: impl Into<String>, value: impl Into<Vec<u8>>, store: Store) -> Self {
        let value = value.into();
        KeywordField {
            name: name.into(),
            field_type: Self::field_type_of(store),
            stored: (store == Store::Yes).then(|| StoredValue::Binary(value.clone())),
            binary: value,
            string: None,
        }
    }

    /// `setStringValue`.
    pub fn set_string_value(&mut self, value: impl Into<String>) -> Result<()> {
        if self.string.is_none() {
            return Err(super::illegal(
                "cannot change value type from BytesRef to String",
            ));
        }
        let value = value.into();
        self.binary = value.as_bytes().to_vec();
        if self.stored.is_some() {
            self.stored = Some(StoredValue::String(value.clone()));
        }
        self.string = Some(value);
        Ok(())
    }

    /// `setBytesValue`.
    pub fn set_bytes_value(&mut self, value: impl Into<Vec<u8>>) -> Result<()> {
        if self.string.is_some() {
            return Err(super::illegal(
                "cannot change value type from String to BytesRef",
            ));
        }
        self.binary = value.into();
        if self.stored.is_some() {
            self.stored = Some(StoredValue::Binary(self.binary.clone()));
        }
        Ok(())
    }
}

impl IndexableField for KeywordField {
    fn name(&self) -> &str {
        &self.name
    }
    fn field_type(&self) -> &FieldType {
        &self.field_type
    }
    fn string_value(&self) -> Option<Cow<'_, str>> {
        self.string.as_deref().map(Cow::Borrowed)
    }
    fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
        Some(Cow::Borrowed(&self.binary))
    }
    fn stored_value(&self) -> Option<StoredValue> {
        self.stored.clone()
    }
    /// `KeywordField.invertableType()`: the bytes are the term.
    fn invertable_type(&self) -> InvertableType {
        InvertableType::Binary
    }
    fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_doc_values_fields() {
        let mut f = NumericDocValuesField::new("n", 7);
        assert_eq!(f.numeric_value(), Some(Number::Long(7)));
        assert_eq!(f.string_value().as_deref(), Some("7"));
        f.set_value(-1);
        assert_eq!(f.value(), -1);
        assert_eq!(f.field_type().doc_values_type(), DocValuesType::Numeric);
        assert_eq!(
            f.field_type().doc_values_skip_index_type(),
            DocValuesSkipIndexType::None
        );
        let i = SortedNumericDocValuesField::indexed_field("s", 3);
        assert_eq!(
            i.field_type().doc_values_skip_index_type(),
            DocValuesSkipIndexType::Range
        );
        assert_eq!(i.name(), "s");
        assert!(i.token_stream(&Analyzer::keyword()).unwrap().is_none());
        assert!(NumericDocValuesField::indexed_field("n", 1)
            .field_type()
            .is_frozen());
    }

    #[test]
    fn bytes_doc_values_fields() {
        let mut s = SortedDocValuesField::new("s", "a");
        assert_eq!(s.binary_value().unwrap().as_ref(), b"a");
        s.set_value("b");
        assert_eq!(s.value(), b"b");
        assert_eq!(
            SortedDocValuesField::indexed_field("s", "x")
                .field_type()
                .doc_values_skip_index_type(),
            DocValuesSkipIndexType::Range
        );
        let ss = SortedSetDocValuesField::indexed_field("ss", "x");
        assert_eq!(ss.field_type().doc_values_type(), DocValuesType::SortedSet);
        let b = BinaryDocValuesField::new("b", vec![0, 1]);
        assert_eq!(b.field_type().doc_values_type(), DocValuesType::Binary);
        assert_eq!(b.name(), "b");
        assert!(b.token_stream(&Analyzer::keyword()).unwrap().is_none());
    }

    #[test]
    fn keyword_fields() {
        let mut k = KeywordField::new("k", "v", Store::Yes);
        assert_eq!(k.invertable_type(), InvertableType::Binary);
        assert_eq!(k.stored_value(), Some(StoredValue::String("v".into())));
        assert_eq!(k.string_value().as_deref(), Some("v"));
        k.set_string_value("w").unwrap();
        assert_eq!(k.binary_value().unwrap().as_ref(), b"w");
        assert_eq!(k.stored_value(), Some(StoredValue::String("w".into())));
        assert!(k.set_bytes_value(vec![1]).is_err());
        assert!(k.token_stream(&Analyzer::keyword()).unwrap().is_none());
        let mut b = KeywordField::from_bytes("k", vec![1, 2], Store::Yes);
        assert_eq!(b.string_value(), None);
        b.set_bytes_value(vec![3]).unwrap();
        assert_eq!(b.stored_value(), Some(StoredValue::Binary(vec![3])));
        assert!(b.set_string_value("x").is_err());
        let n = KeywordField::new("k", "v", Store::No);
        assert_eq!(n.stored_value(), None);
        assert!(n.field_type().omit_norms());
        assert_eq!(n.field_type().index_options(), IndexOptions::Docs);
        assert_eq!(n.name(), "k");
    }
}
