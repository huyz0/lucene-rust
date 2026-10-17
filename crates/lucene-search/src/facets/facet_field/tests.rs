use super::*;
use crate::facets::taxonomy::DirectoryTaxonomyWriter;
use lucene_store::ByteBuffersDirectory;

fn err_message<T: std::fmt::Debug>(r: Result<T>) -> String {
    match r.unwrap_err() {
        Error::IllegalArgument(m) | Error::IllegalState(m) => m,
        e => format!("{e:?}"),
    }
}

/// `(name, value)` of every field, the value as text where it is one.
fn describe(doc: &Document) -> Vec<String> {
    doc.fields()
        .iter()
        .map(|f| format!("{}={:?}", f.name(), f.field_type().doc_values_type()))
        .collect()
}

#[test]
fn facet_field_checks_dim_then_path() {
    let f = FacetField::new("dim", &["a", "b"]).unwrap();
    assert_eq!(f.to_string(), "FacetField(dim=dim path=[a, b])");
    assert_eq!(
        err_message(FacetField::new("", &[""])),
        "empty or null components not allowed; got: "
    );
    assert_eq!(
        err_message(FacetField::new::<&str>("d", &[])),
        "path must have at least one element"
    );
    // The sorted-set field checks its path before its dim.
    let s = SortedSetDocValuesFacetField::new("d", &["x"]).unwrap();
    assert_eq!(s.path, ["x"]);
    assert_eq!(
        err_message(SortedSetDocValuesFacetField::new("", &["x"])),
        "empty or null components not allowed; got: "
    );
    assert_eq!(
        err_message(SortedSetDocValuesFacetField::new::<&str>("d", &[])),
        "path must have at least one element"
    );
    assert!(SortedSetDocValuesFacetField::new("d", &[""]).is_err());
    assert_eq!(FacetEntry::from(f.clone()), FacetEntry::Taxonomy(f));
    assert_eq!(FacetEntry::from(s.clone()), FacetEntry::SortedSet(s));
    verify_label("ok").unwrap();
}

/// `build`: the taxonomy fields, then the sorted-set ones, then the
/// document's own; a hierarchical multi-valued dim records every ancestor
/// once.
#[test]
fn build_orders_and_dedupes() {
    let d = ByteBuffersDirectory::new();
    let w = DirectoryTaxonomyWriter::open(&d).unwrap();
    let mut config = FacetsConfig::new();
    config.set_hierarchical("Cat", true);
    config.set_multi_valued("Cat", true);
    let facets = vec![
        FacetField::new("Cat", &["a", "b"]).unwrap().into(),
        FacetField::new("Cat", &["a", "c"]).unwrap().into(),
        SortedSetDocValuesFacetField::new("Brand", &["x"])
            .unwrap()
            .into(),
    ];
    let mut doc = Document::new();
    doc.add(StringField::new("title", "t", Store::Yes));
    let built = config.build(Some(&w), &facets, doc).unwrap();
    let names: Vec<String> = built
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    // Per taxonomy label three drill-down terms (`ALL`: the full path,
    // `Cat`, `Cat/a`); then the ordinals Cat/a(2), Cat/a/b(3), Cat/a/c(4),
    // each once (the dim's own is dropped); then the sorted-set value and
    // its two drill-down terms.
    let described = describe(&built);
    assert_eq!(
        names.iter().filter(|n| *n == "$facets").count(),
        6 + 3 + 1 + 2,
        "{described:?}"
    );
    assert_eq!(
        described
            .iter()
            .filter(|d| d.ends_with("SortedNumeric"))
            .count(),
        3,
        "{described:?}"
    );
    assert_eq!(names.last().unwrap(), "title");
}

#[test]
fn build_refusals() {
    let config = FacetsConfig::new();
    let doc = Document::new;
    let taxonomy = vec![FacetEntry::from(FacetField::new("A", &["x"]).unwrap())];
    assert_eq!(
        err_message(config.build(None, &taxonomy, doc())),
        "a non-null TaxonomyWriter must be provided when indexing FacetField or \
         AssociationFacetField"
    );
    let twice = vec![
        FacetEntry::from(FacetField::new("A", &["x"]).unwrap()),
        FacetEntry::from(FacetField::new("A", &["y"]).unwrap()),
    ];
    assert_eq!(
        err_message(config.build(None, &twice, doc())),
        "dimension \"A\" is not multiValued, but it appears more than once in this document"
    );
    let deep = vec![FacetEntry::from(FacetField::new("A", &["x", "y"]).unwrap())];
    assert_eq!(
        err_message(config.build(None, &deep, doc())),
        "dimension \"A\" is not hierarchical yet has 2 components"
    );
    let ssdv = vec![FacetEntry::from(
        SortedSetDocValuesFacetField::new("S", &["x", "y"]).unwrap(),
    )];
    assert_eq!(
        err_message(config.build(None, &ssdv, doc())),
        "dimension \"S\" is not hierarchical yet has 2 components"
    );
    // Sorted-set fields alone need no taxonomy writer.
    let ok = vec![FacetEntry::from(
        SortedSetDocValuesFacetField::new("S", &["x"]).unwrap(),
    )];
    assert!(!config.build(None, &ok, doc()).unwrap().fields().is_empty());
}
