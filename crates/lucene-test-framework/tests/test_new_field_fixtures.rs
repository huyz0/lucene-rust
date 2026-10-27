//! Differential test against `fixtures/src/testframework/GenTestFrameworkNewField.java`:
//! for 200 seeds, [`new_field`] and its string/text forms upgrade each base
//! type to the stored/term-vector flags Lucene 10.5.0's
//! `LuceneTestCase.newField` picks, draw exactly as much, and reuse a
//! name's first type.
//!
//! Regenerate with `scripts/gen-test-framework-fixtures.sh --only TestFrameworkNewField`.

use std::fs;
use std::path::PathBuf;

use lucene_index::document::{Field, FieldType, IndexOptions, IndexableField, Store, TextField};
use lucene_test_framework::util::lucene_test_case::{
    new_field, new_string_field, new_text_field, FieldTypes, FieldValue, LuceneTestCase,
};
use lucene_util::java_random::JavaRandom;

/// One fixture case: builds its field from the seed's random.
type Make<'a> = dyn Fn(&mut JavaRandom, &mut FieldTypes) -> Field + 'a;

fn bits(f: &Field) -> String {
    let t = f.field_type();
    [
        t.stored(),
        t.store_term_vectors(),
        t.store_term_vector_positions(),
        t.store_term_vector_payloads(),
        t.store_term_vector_offsets(),
    ]
    .iter()
    .map(|&b| if b { '1' } else { '0' })
    .collect()
}

#[test]
fn new_field_matches_lucene() {
    let mut custom_freqs = FieldType::copy_of(&TextField::type_not_stored());
    custom_freqs
        .set_index_options(IndexOptions::DocsAndCustomFreqs)
        .unwrap();
    let mut stored_only = FieldType::new();
    stored_only.set_stored(true).unwrap();
    let mut vectors = FieldType::copy_of(&TextField::type_not_stored());
    vectors.set_store_term_vectors(true).unwrap();

    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/testframework/data/new_field.txt");
    let fixture = fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let mut lines = fixture.lines();
    let mut types = FieldTypes::default();
    for seed in 0..200 {
        let mut r = JavaRandom::new(seed);
        let n = |s: &str| format!("s{seed}_{s}");
        let v = FieldValue::String("v");
        let cases: Vec<(&str, Box<Make>)> = vec![
            (
                "string",
                Box::new(|r, t| new_string_field(r, t, &n("a"), v, Store::No)),
            ),
            (
                "string_stored",
                Box::new(|r, t| new_string_field(r, t, &n("b"), v, Store::Yes)),
            ),
            (
                "string_bytes",
                Box::new(|r, t| {
                    new_string_field(r, t, &n("c"), FieldValue::Bytes(b"v"), Store::No)
                }),
            ),
            (
                "text",
                Box::new(|r, t| new_text_field(r, t, &n("d"), "v w", Store::No)),
            ),
            (
                "text_stored",
                Box::new(|r, t| new_text_field(r, t, &n("e"), "v w", Store::Yes)),
            ),
            (
                "custom_freqs",
                Box::new(|r, t| new_field(r, t, &n("f"), v, &custom_freqs)),
            ),
            (
                "stored_only",
                Box::new(|r, t| new_field(r, t, &n("g"), v, &stored_only)),
            ),
            (
                "vectors",
                Box::new(|r, t| new_field(r, t, &n("h"), v, &vectors)),
            ),
            (
                "reuse",
                Box::new(|r, t| new_text_field(r, t, &n("a"), "v w", Store::Yes)),
            ),
        ];
        for (case, make) in &cases {
            let line = lines.next().expect("a fixture line per case");
            let field = make(&mut r, &mut types);
            assert_eq!(
                format!("{seed} {case} {} {}", bits(&field), r.next_int()),
                line,
                "seed {seed} case {case}"
            );
            assert!(field.field_type().is_frozen());
        }
    }
    assert_eq!(lines.next(), None);
}

#[test]
fn the_test_case_keeps_its_own_field_types() {
    let mut t = LuceneTestCase::with_config("newField", 1, false, 1);
    let a = t.new_text_field("f", "x y", Store::No);
    let b = t.new_string_field("f", FieldValue::String("z"), Store::Yes);
    assert_eq!(a.field_type(), b.field_type(), "one type per name");
    let mut untokenized = FieldType::copy_of(&TextField::type_stored());
    untokenized.set_tokenized(false).unwrap();
    for name in ["g", "h", "i", "j", "k", "l"] {
        let c = t.new_field(name, FieldValue::Bytes(b"z"), &untokenized);
        assert!(
            !c.field_type().store_term_vector_offsets(),
            "binary: no offsets"
        );
    }

    let mut types = FieldTypes::default();
    new_text_field(&mut JavaRandom::new(0), &mut types, "f", "x", Store::No);
    assert!(types.get("f").is_some());
    types.clear();
    assert!(types.get("f").is_none());
}

#[test]
#[should_panic(expected = "newField(h)")]
fn a_type_that_cannot_carry_the_value_panics() {
    let mut types = FieldTypes::default();
    new_field(
        &mut JavaRandom::new(0),
        &mut types,
        "h",
        FieldValue::Bytes(b"v"),
        &TextField::type_not_stored(),
    );
}
