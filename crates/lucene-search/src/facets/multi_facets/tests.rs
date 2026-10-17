//! Unit tests, after `TestMultipleIndexFields`.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::facets::taxonomy::FastTaxonomyFacetCounts;
use crate::facets::test_index::{build, collect, doc, ff, match_all};
use crate::facets::FacetsConfig;

/// `TestMultipleIndexFields`' layout: `Author` in the default field, `Band`
/// and `Composer` each in their own.
fn config() -> FacetsConfig {
    let mut c = FacetsConfig::new();
    c.set_index_field_name("Band", "$bands");
    c.set_hierarchical("Band", true);
    c.set_index_field_name("Composer", "$composers");
    c
}

#[test]
fn each_dim_is_answered_by_its_own_facets() {
    let config = config();
    let index = build(
        &config,
        vec![
            doc(
                "d0",
                vec![
                    ff("Author", &["Mark Twain"]),
                    ff("Band", &["Rock & Pop", "The Beatles"]),
                    ff("Composer", &["Bach"]),
                ],
            ),
            doc(
                "d1",
                vec![ff("Author", &["Mark Twain"]), ff("Composer", &["Bach"])],
            ),
            doc("d2", vec![ff("Author", &["Stephen King"])]),
        ],
        &[],
        &[],
    );
    index.with(|searcher, reader, taxo| {
        let fc = collect(searcher, &match_all(reader), false);
        let authors = FastTaxonomyFacetCounts::new(taxo, &config, &fc).unwrap();
        let composers =
            FastTaxonomyFacetCounts::with_field("$composers", taxo, &config, &fc).unwrap();
        let multi = MultiFacets::new(
            vec![("Composer".to_string(), &composers as &dyn Facets)],
            Some(&authors),
        );
        let r = multi
            .get_top_children(10, "Composer", &[])
            .unwrap()
            .unwrap();
        assert_eq!(r.value, FacetValue::Int(2));
        let r = multi.get_top_children(10, "Author", &[]).unwrap().unwrap();
        assert_eq!(r.label_values[0].label, "Mark Twain");
        assert_eq!(
            multi.get_specific_value("Composer", &["Bach"]).unwrap(),
            FacetValue::Int(2)
        );
        assert!(multi.get_all_children("Composer", &[]).unwrap().is_some());
        // The mapped dims first, then the default facets' other dims.
        let dims = multi.get_all_dims(10).unwrap();
        assert_eq!(
            dims.iter().map(|r| r.dim.as_str()).collect::<Vec<_>>(),
            vec!["Composer", "Author"]
        );
        assert_eq!(multi.get_top_dims(1, 10).unwrap().len(), 1);
        // `Band` lives in neither field; the default facets refuse it.
        assert!(multi.get_top_children(10, "Band", &[]).is_err());

        let no_default = MultiFacets::new(
            vec![("Composer".to_string(), &composers as &dyn Facets)],
            None,
        );
        match no_default.get_top_children(10, "Author", &[]) {
            Err(Error::IllegalArgument(m)) => assert_eq!(m, "invalid dim \"Author\""),
            other => panic!("{other:?}"),
        }
        assert!(no_default.get_specific_value("Author", &["x"]).is_err());
        assert!(no_default.get_all_children("Author", &[]).is_err());
        assert_eq!(no_default.get_all_dims(3).unwrap().len(), 1);
        assert!(no_default.get_top_children(0, "Composer", &[]).is_err());
        assert!(no_default.get_all_dims(0).is_err());

        // A mapped dim with nothing counted is left out of the list.
        let empty = MultiFacets::new(vec![("Nope".to_string(), &composers as &dyn Facets)], None);
        assert!(empty.get_all_dims(3).unwrap().is_empty());
    });
}
