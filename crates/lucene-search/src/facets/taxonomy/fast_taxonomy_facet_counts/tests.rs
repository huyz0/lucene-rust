//! Unit tests, after `TestTaxonomyFacetCounts`.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::facets::test_index::{
    basic_config, basic_docs, basic_index, build, collect, doc, ff, match_all, term,
};
use crate::facets::{FacetValue, Facets};

fn int(v: i32) -> FacetValue {
    FacetValue::Int(v)
}

fn ia_message<T: std::fmt::Debug>(r: Result<T>) -> String {
    match r {
        Err(Error::IllegalArgument(m)) => m,
        other => panic!("expected IllegalArgument, got {other:?}"),
    }
}

#[test]
fn basic_counts_match_lucenes_test() {
    let index = basic_index();
    let config = basic_config();
    index.with(|searcher, reader, taxo| {
        let fc = collect(searcher, &match_all(reader), false);
        let facets = FastTaxonomyFacetCounts::new(taxo, &config, &fc).unwrap();

        let author = facets.get_top_children(10, "Author", &[]).unwrap().unwrap();
        assert_eq!(
            author.to_string(),
            "dim=Author path=[] value=5 childCount=4\n  Lisa (2)\n  Bob (1)\n  Susan (1)\n  \
             Frank (1)\n"
        );
        let date = facets
            .get_top_children(10, "Publish Date", &[])
            .unwrap()
            .unwrap();
        assert_eq!(
            date.to_string(),
            "dim=Publish Date path=[] value=5 childCount=3\n  2010 (2)\n  2012 (2)\n  1999 (1)\n"
        );
        // Hierarchical dims roll up, so the dim and its inner nodes count.
        assert_eq!(
            facets.get_specific_value("Publish Date", &[]).unwrap(),
            int(5)
        );
        assert_eq!(
            facets
                .get_specific_value("Publish Date", &["2012"])
                .unwrap(),
            int(2)
        );
        assert_eq!(
            facets.get_specific_value("Author", &["Lisa"]).unwrap(),
            int(2)
        );
        assert_eq!(
            facets.get_specific_value("Author", &["Nobody"]).unwrap(),
            int(-1)
        );
        let under = facets
            .get_top_children(10, "Publish Date", &["2012", "1"])
            .unwrap()
            .unwrap();
        assert_eq!(under.path, vec!["2012", "1"]);
        assert_eq!(
            under
                .label_values
                .iter()
                .map(|lv| lv.label.as_str())
                .collect::<Vec<_>>(),
            vec!["1", "7"]
        );
        // `getAllChildren` walks the siblings (newest ordinal first).
        let all = facets.get_all_children("Author", &[]).unwrap().unwrap();
        assert_eq!(all.value, int(5));
        assert_eq!(all.child_count, 4);
        assert_eq!(all.label_values.len(), 4);
        assert!(facets.children_loaded() && facets.siblings_loaded());

        // Ties on the value go to the dim name.
        let dims = facets.get_all_dims(1).unwrap();
        assert_eq!(
            dims.iter().map(|r| r.dim.as_str()).collect::<Vec<_>>(),
            vec!["Author", "Publish Date"]
        );
        let top = facets.get_top_dims(1, 1).unwrap();
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].dim, "Author");
        assert_eq!(top[0].label_values.len(), 1);
        assert_eq!(facets.get_top_dims(5, 2).unwrap().len(), 2);
        assert!(format!("{facets:?}").contains("Count"));
        assert_eq!(facets.index_field_name(), "$facets");
    });
}

#[test]
fn unknown_paths_and_dims() {
    let index = basic_index();
    let config = basic_config();
    index.with(|searcher, reader, taxo| {
        let fc = collect(searcher, &match_all(reader), false);
        let facets = FastTaxonomyFacetCounts::new(taxo, &config, &fc).unwrap();
        assert!(facets.get_top_children(10, "Nope", &[]).unwrap().is_none());
        assert!(facets.get_all_children("Nope", &[]).unwrap().is_none());
        assert!(facets
            .get_top_children(10, "Author", &["Lisa"])
            .unwrap()
            .is_none());
        assert!(facets
            .get_all_children("Author", &["Lisa"])
            .unwrap()
            .is_none());
        assert_eq!(facets.get_specific_value("Nope", &["x"]).unwrap(), int(-1));
        // A flat single-valued dim has no dimension-level value.
        assert_eq!(
            ia_message(facets.get_specific_value("Author", &[])),
            "cannot return dimension-level value alone; use getTopChildren instead"
        );
        assert_eq!(
            ia_message(facets.get_top_children(0, "Author", &[])),
            "topN must be > 0 (got: 0)"
        );
        assert_eq!(
            ia_message(facets.get_all_dims(-1)),
            "topN must be > 0 (got: -1)"
        );
        assert_eq!(ia_message(facets.get_top_dims(0, 1)), "topN must be > 0");
        assert_eq!(ia_message(facets.get_top_dims(1, 0)), "topN must be > 0");
        // An empty path component is refused by the label.
        assert!(facets.get_top_children(1, "Author", &[""]).is_err());
    });
}

#[test]
fn nothing_matched_is_uninitialized() {
    let index = basic_index();
    let config = basic_config();
    index.with(|searcher, _, taxo| {
        let fc = collect(searcher, &term("id", "none"), false);
        let facets = FastTaxonomyFacetCounts::new(taxo, &config, &fc).unwrap();
        assert!(facets
            .get_top_children(10, "Author", &[])
            .unwrap()
            .is_none());
        assert!(facets.get_all_children("Author", &[]).unwrap().is_none());
        assert!(facets.get_all_dims(10).unwrap().is_empty());
        assert!(facets.get_top_dims(10, 10).unwrap().is_empty());
        assert_eq!(
            facets.get_specific_value("Author", &["Lisa"]).unwrap(),
            int(0),
            "a known path counted nothing"
        );
    });
}

#[test]
fn a_selective_query_counts_only_its_hits() {
    let index = basic_index();
    let config = basic_config();
    index.with(|searcher, _, taxo| {
        let fc = collect(searcher, &term("id", "d2"), false);
        let facets = FastTaxonomyFacetCounts::new(taxo, &config, &fc).unwrap();
        let r = facets.get_top_children(10, "Author", &[]).unwrap().unwrap();
        assert_eq!(r.value, int(1));
        assert_eq!(r.label_values[0].label, "Lisa");
        assert_eq!(
            facets
                .get_specific_value("Publish Date", &["2012", "1"])
                .unwrap(),
            int(1)
        );
    });
}

#[test]
fn wrong_index_field_name() {
    // `TestTaxonomyFacetCounts.testWrongIndexFieldName`.
    let mut config = FacetsConfig::new();
    config.set_index_field_name("a", "$facets2");
    let index = build(&config, vec![doc("d0", vec![ff("a", &["foo1"])])], &[], &[]);
    index.with(|searcher, reader, taxo| {
        let fc = collect(searcher, &match_all(reader), false);
        let facets = FastTaxonomyFacetCounts::new(taxo, &config, &fc).unwrap();
        // The default field holds nothing of `a`'s.
        assert!(facets.get_all_dims(10).unwrap().is_empty());
        let want = "dimension \"a\" cannot be found in field \"$facets\", since it was \
                    configured to be indexed into field \"$facets2\"";
        assert_eq!(ia_message(facets.get_top_children(10, "a", &[])), want);
        assert_eq!(ia_message(facets.get_all_children("a", &[])), want);
        assert_eq!(ia_message(facets.get_specific_value("a", &["foo1"])), want);
        let right = FastTaxonomyFacetCounts::with_field("$facets2", taxo, &config, &fc).unwrap();
        assert_eq!(right.get_specific_value("a", &["foo1"]).unwrap(), int(1));
    });
}

#[test]
fn require_dim_count_and_multi_valued_dims() {
    // `TestTaxonomyFacetCounts.testRequireDimCount`, with a multi-valued dim
    // that does not require it.
    let mut config = FacetsConfig::new();
    config.set_require_dim_count("dim", true);
    config.set_multi_valued("dim2", true);
    config.set_require_dim_count("dim2", true);
    config.set_multi_valued("dim3", true);
    let index = build(
        &config,
        vec![
            doc(
                "d0",
                vec![
                    ff("dim", &["a"]),
                    ff("dim2", &["a"]),
                    ff("dim2", &["b"]),
                    ff("dim3", &["a"]),
                    ff("dim3", &["b"]),
                ],
            ),
            doc("d1", vec![ff("dim", &["b"]), ff("dim3", &["a"])]),
        ],
        &[],
        &[],
    );
    index.with(|searcher, reader, taxo| {
        let fc = collect(searcher, &match_all(reader), false);
        let facets = FastTaxonomyFacetCounts::new(taxo, &config, &fc).unwrap();
        let dim = facets.get_top_children(10, "dim", &[]).unwrap().unwrap();
        assert_eq!(dim.value, int(2));
        // The documents with the dim, not the sum of its children.
        let dim2 = facets.get_top_children(10, "dim2", &[]).unwrap().unwrap();
        assert_eq!(dim2.value, int(1));
        assert_eq!(dim2.child_count, 2);
        assert_eq!(facets.get_specific_value("dim2", &[]).unwrap(), int(1));
        let all2 = facets.get_all_children("dim2", &[]).unwrap().unwrap();
        assert_eq!(all2.value, int(1));
        // Multi-valued without a dim count: -1.
        let dim3 = facets.get_top_children(10, "dim3", &[]).unwrap().unwrap();
        assert_eq!(dim3.value, int(-1));
        assert_eq!(
            facets.get_all_children("dim3", &[]).unwrap().unwrap().value,
            int(-1)
        );
        // `getTopDims` gives such a dim -1 too, which ranks it last.
        let top = facets.get_top_dims(3, 1).unwrap();
        assert_eq!(
            top.iter().map(|r| r.dim.as_str()).collect::<Vec<_>>(),
            vec!["dim", "dim2", "dim3"]
        );
        // One slot: `dim2` (1) loses to `dim` (2); `dim3` (-1) never gets in.
        let one = facets.get_top_dims(1, 1).unwrap();
        assert_eq!(one[0].dim, "dim");
    });
}

#[test]
fn count_all_skips_deleted_documents() {
    let index = build(&basic_config(), basic_docs(), &[1, 3], &["d1", "d4"]);
    let config = basic_config();
    index.with(|_, reader, taxo| {
        let facets = FastTaxonomyFacetCounts::count_all("$facets", reader, taxo, &config).unwrap();
        let r = facets.get_top_children(10, "Author", &[]).unwrap().unwrap();
        assert_eq!(
            r.to_string(),
            "dim=Author path=[] value=3 childCount=3\n  Bob (1)\n  Lisa (1)\n  Susan (1)\n"
        );
        assert_eq!(
            facets.get_specific_value("Publish Date", &[]).unwrap(),
            int(3)
        );
        // A field nothing was indexed into: no counting at all.
        let none = FastTaxonomyFacetCounts::count_all("$other", reader, taxo, &config).unwrap();
        assert!(none.get_all_dims(10).unwrap().is_empty());
    });
}

#[test]
fn a_field_without_doc_values_counts_nothing() {
    let index = basic_index();
    let config = basic_config();
    index.with(|searcher, reader, taxo| {
        let fc = collect(searcher, &match_all(reader), false);
        let facets = FastTaxonomyFacetCounts::with_field("$other", taxo, &config, &fc).unwrap();
        assert!(facets.get_all_dims(10).unwrap().is_empty());
    });
}

#[test]
fn an_ordinal_past_the_taxonomy_is_an_error() {
    let mut counts = vec![0; 3];
    struct One(i64);
    impl crate::reader::DocIdSetIterator for One {
        fn doc_id(&self) -> i32 {
            0
        }
        fn next_doc(&mut self) -> Result<i32> {
            Ok(NO_MORE_DOCS)
        }
        fn advance(&mut self, _: i32) -> Result<i32> {
            Ok(NO_MORE_DOCS)
        }
        fn cost(&self) -> i64 {
            1
        }
    }
    impl crate::reader::DocValuesIterator for One {
        fn advance_exact(&mut self, _: i32) -> Result<bool> {
            Ok(true)
        }
    }
    impl SortedNumericDocValues for One {
        fn next_value(&mut self) -> Result<i64> {
            Ok(self.0)
        }
        fn doc_value_count(&self) -> i32 {
            1
        }
    }
    count_values(&mut counts, &mut One(2)).unwrap();
    assert_eq!(counts, vec![0, 0, 1]);
    match count_values(&mut counts, &mut One(3)) {
        Err(Error::IllegalArgument(m)) => assert_eq!(m, "Index 3 out of bounds for length 3"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_selective_query_over_a_large_taxonomy_counts_sparsely() {
    // 1,200 authors: a taxonomy past 1024 ordinals, and one hit in 1,200
    // documents, so the counts go to the hash table.
    let config = FacetsConfig::new();
    let docs = (0..1200)
        .map(|i| {
            doc(
                &format!("d{i}"),
                vec![
                    ff("Author", &[&format!("a{i}")]),
                    ff("Tag", &[&format!("t{}", i % 3)]),
                ],
            )
        })
        .collect();
    let index = build(&config, docs, &[599], &[]);
    index.with(|searcher, _, taxo| {
        let fc = collect(searcher, &term("id", "d7"), false);
        let facets = FastTaxonomyFacetCounts::new(taxo, &config, &fc).unwrap();
        assert!(matches!(facets.counts, OrdValues::Sparse(_)));
        let r = facets.get_top_children(10, "Author", &[]).unwrap().unwrap();
        assert_eq!(
            r.to_string(),
            "dim=Author path=[] value=1 childCount=1\n  a7 (1)\n"
        );
        let r = facets.get_all_children("Tag", &[]).unwrap().unwrap();
        assert_eq!(
            r.to_string(),
            "dim=Tag path=[] value=1 childCount=1\n  t1 (1)\n"
        );
        assert_eq!(
            facets.get_specific_value("Author", &["a8"]).unwrap(),
            int(0)
        );
    });
}

#[test]
fn a_sparse_ordinal_past_the_taxonomy_is_an_error() {
    // The sparse twin of `an_ordinal_past_the_taxonomy_is_an_error`: the hash
    // table takes any ordinal, and walking it for a path's children reads
    // `parents[ord]`, which Java refuses.
    let config = FacetsConfig::new();
    let docs = (0..1200)
        .map(|i| doc(&format!("d{i}"), vec![ff("Author", &[&format!("a{i}")])]))
        .collect();
    let index = build(&config, docs, &[], &[]);
    index.with(|searcher, _, taxo| {
        let fc = collect(searcher, &term("id", "d7"), false);
        let mut facets = FastTaxonomyFacetCounts::new(taxo, &config, &fc).unwrap();
        let size = taxo.get_size().unwrap();
        let OrdValues::Sparse(map) = &mut facets.inner.counts else {
            panic!("expected sparse counts");
        };
        map.insert(size, 1);
        facets.inner.finish_counting();
        assert_eq!(
            ia_message(facets.get_top_children(10, "Author", &[])),
            format!("Index {size} out of bounds for length {size}")
        );
    });
}
