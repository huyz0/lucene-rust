//! Unit tests, after `TestDrillDownQuery`.
#![allow(clippy::arithmetic_side_effects)]

use super::*;

fn config() -> FacetsConfig {
    let mut c = FacetsConfig::new();
    c.set_index_field_name("Color", "$color");
    c
}

#[test]
fn values_of_a_dim_are_ored_and_dims_anded() {
    let config = config();
    let mut q = DrillDownQuery::new(&config);
    q.add("a", &["1"]).unwrap();
    q.add("b", &["x", "y"]).unwrap();
    q.add("a", &["2"]).unwrap();
    q.add("Color", &["red"]).unwrap();
    assert_eq!(q.dims(), ["a", "b", "Color"]);
    assert!(q.base_query().is_none());
    assert_eq!(
        q.to_string(),
        "#($facets:a\u{1f}1 $facets:a\u{1f}2) #($facets:b\u{1f}x\u{1f}y) #($color:Color\u{1f}red)"
    );
    let dims = q.drill_down_queries();
    assert_eq!(dims.len(), 3);
    match &dims[0] {
        Clause::Boolean(b) => assert_eq!(b.should.len(), 2),
        other => panic!("{other:?}"),
    }
    assert!(std::ptr::eq(q.config(), &config));
    // No base: the rewrite is the filters alone.
    let r = q.rewrite();
    assert!(r.must.is_empty());
    assert_eq!(r.filter.len(), 3);
}

#[test]
fn an_empty_query_rewrites_to_match_all() {
    let config = config();
    let q = DrillDownQuery::new(&config);
    assert!(matches!(q.rewrite_clause(), Clause::MatchAllDocs(_)));
    let r = q.rewrite();
    assert!(matches!(r.must[..], [Clause::MatchAllDocs(_)]));
    assert_eq!(q.to_string(), "");
    // A base alone is the base, required.
    let base = Clause::Term(TermQuery::new("f", "v"));
    let q = DrillDownQuery::with_base(&config, base.clone());
    assert_eq!(q.base_query(), Some(&base));
    assert_eq!(q.rewrite().must, vec![base]);
    assert_eq!(q.to_string(), "+f:v");
}

#[test]
fn a_filter_wraps_the_base() {
    let config = config();
    let filter = Clause::Term(TermQuery::new("q", "even"));
    let mut q = DrillDownQuery::new(&config);
    q.add("a", &["1"]).unwrap();
    let f = DrillDownQuery::with_filter(&config, filter.clone(), &q);
    assert_eq!(f.to_string(), "+(+*:* #q:even) #($facets:a\u{1f}1)");
    let mut based = DrillDownQuery::with_base(&config, Clause::Term(TermQuery::new("f", "v")));
    based.add("a", &["1"]).unwrap();
    let f = DrillDownQuery::with_filter(&config, filter, &based);
    assert_eq!(f.to_string(), "+(+f:v #q:even) #($facets:a\u{1f}1)");
    assert_eq!(f.dims(), ["a"]);
}

#[test]
fn custom_sub_queries() {
    let config = config();
    let mut q = DrillDownQuery::new(&config);
    q.add_query("body", Clause::Term(TermQuery::new("body", "x")));
    q.add("body", &["v"]).unwrap();
    assert_eq!(q.to_string(), "#(body:x $facets:body\u{1f}v)");
}

#[test]
fn equality_is_the_base_and_the_dims() {
    let config = config();
    let other_config = FacetsConfig::new();
    let mut a = DrillDownQuery::new(&config);
    a.add("a", &["1"]).unwrap();
    let mut b = DrillDownQuery::new(&other_config);
    b.add("a", &["1"]).unwrap();
    // Structural: Java would call these unequal (its dims' builders compare
    // by identity), a documented difference.
    assert_eq!(a, b);
    b.add("a", &["2"]).unwrap();
    assert_ne!(a, b);
    let c = DrillDownQuery::with_base(&config, Clause::Term(TermQuery::new("f", "v")));
    assert_ne!(DrillDownQuery::new(&config), c);
    assert_eq!(a.clone(), a);
}

#[test]
fn the_drill_down_term() {
    let t = DrillDownQuery::term("$facets", "Publish Date", &["2001", "3"]).unwrap();
    assert_eq!(t.field, "$facets");
    assert_eq!(t.term, b"Publish Date\x1f2001\x1f3");
    let dim = DrillDownQuery::term("$facets", "Tag", &[]).unwrap();
    assert_eq!(dim.term, b"Tag");
}

#[test]
fn refuses_an_empty_path_component() {
    let config = config();
    let mut q = DrillDownQuery::new(&config);
    let err = q.add("a", &["x", ""]).unwrap_err();
    assert!(
        matches!(&err, Error::IllegalArgument(m) if m == "each path component must have length > 0 (got: \"\")"),
        "{err:?}"
    );
    // Nothing was added.
    assert!(q.dims().is_empty());
    assert!(DrillDownQuery::term("$facets", "", &[]).is_err());
}
