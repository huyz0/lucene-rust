use super::*;
use std::collections::hash_map::DefaultHasher;

fn label(c: &[&str]) -> FacetLabel {
    FacetLabel::new(c.iter().copied()).unwrap()
}

fn std_hash(l: &FacetLabel) -> u64 {
    let mut h = DefaultHasher::new();
    l.hash(&mut h);
    h.finish()
}

#[test]
fn basic_and_root() {
    assert_eq!(FacetLabel::root().len(), 0);
    assert!(FacetLabel::root().is_empty());
    assert_eq!(label(&["hello"]).len(), 1);
    let l = label(&["hello", "world"]);
    assert_eq!(l.len(), 2);
    assert!(!l.is_empty());
    assert_eq!(l.path(), ["hello", "world"]);
    assert_eq!(l.components(), ["hello", "world"]);
    let d = FacetLabel::from_dim_path("dim", &["a", "b"]).unwrap();
    assert_eq!(d, label(&["dim", "a", "b"]));
}

#[test]
fn to_string_and_debug() {
    assert_eq!(FacetLabel::root().to_string(), "FacetLabel: []");
    assert_eq!(label(&["a", "b"]).to_string(), "FacetLabel: [a, b]");
    assert_eq!(format!("{:?}", label(&["x"])), "FacetLabel: [x]");
}

#[test]
fn subpath_shares_components() {
    let l = label(&["a", "b", "c"]);
    assert_eq!(l.subpath(5), l);
    assert_eq!(l.subpath(3), l);
    let p = l.subpath(1);
    assert_eq!(p.path(), ["a"]);
    assert_eq!(p.components().len(), 3);
    // Java reads `components.length`: the parent's last component.
    assert_eq!(p.last_component().unwrap(), "c");
    assert_eq!(l.subpath(0), FacetLabel::root());
    assert!(matches!(
        FacetLabel::root().last_component(),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn equals_and_hash() {
    let a = label(&["a", "b"]);
    let b = label(&["a", "b", "c"]).subpath(2);
    assert_eq!(a, b);
    assert_eq!(a.hash_code(), b.hash_code());
    assert_eq!(a.long_hash_code(), b.long_hash_code());
    assert_eq!(std_hash(&a), std_hash(&b));
    assert_ne!(a, label(&["a"]));
    assert_eq!(FacetLabel::root().hash_code(), 0);
    assert_eq!(FacetLabel::root().long_hash_code(), 0);
    // Java: `1 * 31 + "a".hashCode()`.
    assert_eq!(label(&["a"]).hash_code(), 31 + 97);
    assert_eq!(label(&["a"]).long_hash_code(), 65599 + 97);
    // A clone and a subpath share the array; equal without the strings.
    assert_eq!(a.clone(), a);
    assert_ne!(label(&["a", "b", "c"]).subpath(1), label(&["a", "b", "c"]));
    // Same length, a different component (same and different byte lengths),
    // and the boundary between components counts.
    assert_ne!(a, label(&["a", "x"]));
    assert_ne!(a, label(&["a", "bb"]));
    assert_ne!(label(&["ab", "c"]), label(&["a", "bc"]));
    assert_ne!(
        std_hash(&label(&["ab", "c"])),
        std_hash(&label(&["a", "bc"]))
    );
    // Components longer than one eight-byte word hash and compare whole.
    let long = label(&["dimension", "a-long-component"]);
    assert_eq!(long, label(&["dimension", "a-long-component"]));
    assert_eq!(
        std_hash(&long),
        std_hash(&label(&["dimension", "a-long-component"]))
    );
    assert_ne!(long, label(&["dimension", "a-long-componenT"]));
}

#[test]
fn compare_to_is_java_string_order() {
    assert_eq!(label(&["a"]).cmp(&label(&["a"])), Ordering::Equal);
    assert_eq!(label(&["a"]).cmp(&label(&["a", "b"])), Ordering::Less);
    assert_eq!(label(&["b"]).cmp(&label(&["a", "b"])), Ordering::Greater);
    assert_eq!(
        label(&["a", "b"]).partial_cmp(&label(&["a", "c"])),
        Some(Ordering::Less)
    );
    // A supplementary character sorts before U+FFFF in UTF-16 order, after
    // it in code point order.
    let supp = label(&["\u{10000}"]);
    let bmp = label(&["\u{ffff}"]);
    assert_eq!(supp.compare_to(&bmp), Ordering::Less);
    assert_eq!(java_compare("é", "a"), Ordering::Greater);
}

#[test]
fn empty_components_refused() {
    for c in [&["", "b"][..], &["a", ""], &[""]] {
        let err = FacetLabel::new(c.iter().copied()).unwrap_err();
        assert!(
            matches!(&err, Error::IllegalArgument(m) if m.starts_with("empty or null components not allowed: [")),
            "{err:?}"
        );
    }
    let err = FacetLabel::new(["a", ""]).unwrap_err();
    assert!(matches!(&err, Error::IllegalArgument(m) if m.ends_with("[a, ]")));
}

#[test]
fn max_category_path_length() {
    let max = FacetLabel::MAX_CATEGORY_PATH_LENGTH;
    assert_eq!(max, 8191);
    // Exactly the limit, separators counted.
    let half = "x".repeat(max / 2);
    FacetLabel::new([half.clone(), half.clone()]).unwrap();
    let too_long = "x".repeat(max + 1);
    let err = FacetLabel::new([too_long]).unwrap_err();
    let Error::IllegalArgument(m) = err else {
        panic!("{err:?}")
    };
    assert_eq!(
        m,
        format!(
            "category path exceeds maximum allowed path length: max={max} len={} path=[{}...",
            max + 1,
            "x".repeat(29)
        )
    );
    // UTF-16 units: a supplementary character counts twice.
    let supp = "\u{10000}".repeat(max / 2 + 1);
    assert!(FacetLabel::new([supp]).is_err());
    assert_eq!(utf16_len("a\u{10000}"), 3);
    assert_eq!(utf16_len("ab"), 2);
}

#[test]
fn term_round_trip() {
    let l = label(&["a\u{1f}b", "c\u{1e}d", "e"]);
    let term = l.to_term();
    assert_eq!(FacetLabel::from_term(term.as_bytes()).unwrap(), l);
    assert_eq!(label(&["x", "y"]).to_term(), "x\u{1f}y");
    assert_eq!(java_array_to_string::<&str>(&[]), "[]");
    assert_eq!(java_string_hash("ab"), 97 * 31 + 98);
}
