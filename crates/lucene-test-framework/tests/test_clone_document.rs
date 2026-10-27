//! `TestUtil.cloneDocument`: every kind of field it rebuilds keeps its
//! name, type and value; the kinds Java rejects panic.

use lucene_index::document::{
    BinaryDocValuesField, BinaryPoint, Document, Field, IndexableField, IntField, KeywordField,
    NumericDocValuesField, SortedDocValuesField, SortedNumericDocValuesField, Store, StringField,
    TextField,
};
use lucene_test_framework::util::test_util::clone_document;

fn same(a: &dyn IndexableField, b: &dyn IndexableField) {
    assert_eq!(a.name(), b.name());
    assert_eq!(a.field_type(), b.field_type(), "{}", a.name());
    assert_eq!(a.string_value(), b.string_value(), "{}", a.name());
    if a.string_value().is_none() {
        // A port `StringField` also answers its text as bytes; the clone is
        // Java's plain `Field` of the string.
        assert_eq!(a.binary_value(), b.binary_value(), "{}", a.name());
    }
    assert_eq!(a.numeric_value(), b.numeric_value(), "{}", a.name());
    assert_eq!(a.stored_value(), b.stored_value(), "{}", a.name());
}

#[test]
fn every_kind_round_trips() {
    let mut doc = Document::new();
    doc.add(KeywordField::new("kw", "key", Store::Yes));
    doc.add(KeywordField::new("kw2", "other", Store::No));
    doc.add(IntField::new("int", -7, Store::Yes));
    doc.add(IntField::new("int2", 9, Store::No));
    doc.add(NumericDocValuesField::new("ndv", 42));
    doc.add(BinaryDocValuesField::new("bdv", b"bin".to_vec()));
    doc.add(SortedDocValuesField::new("sdv", b"sorted".to_vec()));
    doc.add(BinaryPoint::new("pt", &[b"ab", b"cd"]).unwrap());
    doc.add(StringField::new("id", "17", Store::Yes));
    doc.add(Field::from_string("body", "some text", TextField::type_stored()).unwrap());
    let clone = clone_document(&doc);
    assert_eq!(clone.fields().len(), doc.fields().len());
    for (a, b) in doc.fields().iter().zip(clone.fields()) {
        same(a.as_ref(), b.as_ref());
    }
}

#[test]
#[should_panic(expected = "unknown Type")]
fn sorted_numeric_of_another_field_panics() {
    let mut doc = Document::new();
    doc.add(SortedNumericDocValuesField::new("snd", 3));
    clone_document(&doc);
}

#[test]
#[should_panic(expected = "has no value")]
fn plain_field_without_text_panics() {
    let mut doc = Document::new();
    doc.add(StringField::from_bytes("b", b"raw".to_vec(), Store::Yes));
    clone_document(&doc);
}
