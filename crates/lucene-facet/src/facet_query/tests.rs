#![allow(clippy::arithmetic_side_effects)]

use super::*;

fn config() -> FacetsConfig {
    let mut c = FacetsConfig::new();
    c.set_index_field_name("Color", "$color");
    c
}

#[test]
fn a_facet_query_is_the_categorys_term() {
    let config = config();
    let q = FacetQuery::new(&config, "Author", &["a3"]).unwrap();
    assert_eq!(q.query().field, "$facets");
    assert_eq!(q.query().term, b"Author\x1fa3");
    assert_eq!(q.to_string(), "$facets:Author\u{1f}a3");
    assert_eq!(q.to_string_with_field("$facets"), "Author\u{1f}a3");
    assert_eq!(q.to_string_with_field("other"), "$facets:Author\u{1f}a3");
    let color = FacetQuery::new(&config, "Color", &["c1", "s0"]).unwrap();
    assert_eq!(color.to_string(), "$color:Color\u{1f}c1\u{1f}s0");
    // The default config ignores the dim's index field.
    let d = FacetQuery::with_default_config("Color", &["c1"]).unwrap();
    assert_eq!(d.query().field, "$facets");
    assert!(matches!(Clause::from(d), Clause::Term(t) if t.term == b"Color\x1fc1"));
    assert_eq!(
        FacetQuery::new(&config, "Tag", &[]).unwrap().query().term,
        b"Tag"
    );
}

#[test]
fn a_multi_facet_query_is_the_categories_terms() {
    let config = config();
    let q = MultiFacetQuery::new(&config, "Tag", &[&["t6"], &["t1"], &["t6"]]).unwrap();
    assert_eq!(
        q.query().terms,
        vec![b"Tag\x1ft1".to_vec(), b"Tag\x1ft6".to_vec()]
    );
    assert_eq!(q.to_string(), "$facets:(Tag\u{1f}t1 Tag\u{1f}t6)");
    assert_eq!(q.to_string_with_field("$facets"), q.to_string());
    let color = MultiFacetQuery::new(&config, "Color", &[&["c2", "s1"], &["c0"]]).unwrap();
    assert_eq!(
        color.to_string(),
        "$color:(Color\u{1f}c0 Color\u{1f}c2\u{1f}s1)"
    );
    let d = MultiFacetQuery::with_default_config("Color", &[&["c0"]]).unwrap();
    assert!(matches!(Clause::from(d), Clause::TermInSet(t) if t.field == "$facets"));
    let none = MultiFacetQuery::new(&config, "Tag", &[]).unwrap();
    assert_eq!(none.to_string(), "$facets:()");
}

#[test]
fn refuses_an_empty_path_component() {
    let config = config();
    let msg = "each path component must have length > 0 (got: \"\")";
    for err in [
        FacetQuery::new(&config, "Author", &[""]).unwrap_err(),
        FacetQuery::with_default_config("Author", &["a", ""]).unwrap_err(),
        MultiFacetQuery::new(&config, "Tag", &[&["t1"], &[""]]).unwrap_err(),
        MultiFacetQuery::with_default_config("Tag", &[&[""]]).unwrap_err(),
    ] {
        assert!(
            matches!(&err, Error::IllegalArgument(m) if m == msg),
            "{err:?}"
        );
    }
}
