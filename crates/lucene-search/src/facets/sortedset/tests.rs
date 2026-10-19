//! Unit tests.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::facets::test_index::{build, collect, doc, id_doc, match_all, plain_index, term, Doc};
use crate::facets::SortedSetDocValuesFacetField;

fn ss(dim: &str, path: &[&str]) -> crate::facets::FacetEntry {
    SortedSetDocValuesFacetField::new(dim, path).unwrap().into()
}

fn config() -> FacetsConfig {
    let mut c = FacetsConfig::new();
    c.set_hierarchical("Date", true);
    c.set_multi_valued("Tag", true);
    c
}

/// 40 documents over three segments: `Author` (flat), `Tag` (flat,
/// multi-valued, 30 distinct values), `Date` (hierarchical); the third
/// segment's first document has no facets at all.
fn docs() -> Vec<Doc> {
    (0..40)
        .map(|i| {
            if i == 30 {
                return doc("plain", Vec::new());
            }
            let id = match i {
                5 => "five",
                _ if i % 2 == 0 => "even",
                _ => "odd",
            };
            doc(
                id,
                vec![
                    ss("Author", &[["a", "b", "c"][i % 3]]),
                    ss("Tag", &[&format!("t{:02}", i % 30)]),
                    ss("Tag", &[&format!("t{:02}", (i + 7) % 30)]),
                    ss("Date", &["2010", &(1 + i % 4).to_string()]),
                ],
            )
        })
        .collect()
}

fn top(f: &dyn Facets, dim: &str, path: &[&str]) -> Option<FacetResult> {
    f.get_top_children(100, dim, path).unwrap()
}

#[test]
fn value_counts_past_int_max_are_refused() {
    assert!(check_value_count(i64::from(i32::MAX)).is_ok());
    let e = check_value_count(i64::from(i32::MAX) + 1).unwrap_err();
    assert!(matches!(e, Error::IllegalArgument(m) if m.contains("2147483648")));
}

#[test]
fn state_errors_read_as_java_words_them() {
    let e = state_error(FacetsStateError::NotHierarchical {
        path: vec!["a".into(), "b".into()],
        term: "x".into(),
    });
    assert!(matches!(e, Error::IllegalArgument(m)
        if m == "dimension not configured to handle hierarchical field; got: [a, b] x"));
    let e = state_error(FacetsStateError::EmptyLabel { ord: 3 });
    assert!(matches!(e, Error::IllegalArgument(m) if m.contains("ordinal 3")));
    assert!(matches!(bad_ord(-1), Error::IllegalState(m) if m == "ordinal -1 is out of bounds"));
}

#[test]
fn sinks_refuse_out_of_range_ordinals() {
    let mut plain = vec![0i32; 2];
    plain.as_mut_slice().add(1, 3).unwrap();
    plain.as_mut_slice().add(1, i32::MAX).unwrap();
    assert_eq!(plain, vec![0, 3i32.wrapping_add(i32::MAX)]);
    assert!(plain.as_mut_slice().add(2, 1).is_err());
    assert!(plain.as_mut_slice().add(-1, 1).is_err());
    let atomics = [AtomicI32::new(0), AtomicI32::new(0)];
    let mut sink: &[AtomicI32] = &atomics;
    sink.add(0, 4).unwrap();
    assert!(sink.add(2, 1).is_err());
    assert!(sink.add(-5, 1).is_err());
    assert_eq!(atomics[0].load(Ordering::Relaxed), 4);
}

#[test]
fn a_field_never_indexed_is_refused() {
    let c = config();
    let message = |r: Result<DefaultSortedSetDocValuesReaderState<'_>>| match r {
        Err(Error::IllegalArgument(m)) => m,
        other => panic!("{:?}", other.map(|_| ())),
    };
    // No leaves at all.
    plain_index(Vec::new(), &[]).with(|_, reader, _| {
        assert_eq!(
            message(DefaultSortedSetDocValuesReaderState::new(reader, &c)),
            "field \"$facets\" was not indexed with SortedSetDocValues"
        );
    });
    // One leaf, several: none holds the field.
    for flush in [&[][..], &[0][..]] {
        plain_index(vec![id_doc("a"), id_doc("b")], flush).with(|_, reader, _| {
            assert!(message(DefaultSortedSetDocValuesReaderState::with_field(
                reader, "other", &c
            ))
            .contains("\"other\""));
        });
    }
}

#[test]
fn a_deep_path_under_a_flat_dim_is_refused() {
    let idx = build(&config(), docs(), &[], &[]);
    idx.with(|_, reader, _| {
        let flat = FacetsConfig::new();
        let e = DefaultSortedSetDocValuesReaderState::new(reader, &flat).unwrap_err();
        assert!(matches!(e, Error::IllegalArgument(m)
            if m.starts_with("dimension not configured to handle hierarchical field")));
    });
}

#[test]
fn the_state_answers_its_getters() {
    let c = config();
    let multi = build(&c, docs(), &[9, 29], &[]);
    multi.with(|_, reader, _| {
        let state = DefaultSortedSetDocValuesReaderState::new(reader, &c).unwrap();
        assert_eq!(state.field(), "$facets");
        assert_eq!(state.reader().leaves().len(), 3);
        assert!(state.ordinal_map().is_some());
        assert_eq!(
            format!("{state:?}"),
            "DefaultSortedSetDocValuesReaderState(field=$facets reader=3 leaves)"
        );
        // 3 authors, 30 tags, Date + 2010 + 4 months.
        assert_eq!(state.size(), 3 + 30 + 6);
        assert!(state.facets_config().dim_config("Date").hierarchical);
        let mut dims: Vec<&str> = state.dims().collect();
        dims.sort_unstable();
        assert_eq!(dims, ["Author", "Date", "Tag"]);
        let ranges: Vec<(&str, OrdRange)> = state.prefix_to_ord_range().collect();
        assert_eq!(ranges.len(), 2);
        let tag = state.ord_range("Tag").unwrap().unwrap();
        assert_eq!(tag.end - tag.start + 1, 30);
        assert!(state.ord_range("Missing").unwrap().is_none());
        assert!(matches!(
            state.ord_range("Date"),
            Err(Error::Unsupported(_))
        ));
        assert!(state.dim_tree("Date").unwrap().is_some());
        assert!(matches!(state.dim_tree("Tag"), Err(Error::Unsupported(_))));
        assert!(state.facets_state().size() > 0);
    });
    let single = build(&c, docs(), &[], &[]);
    single.with(|_, reader, _| {
        let state = DefaultSortedSetDocValuesReaderState::new(reader, &c).unwrap();
        assert!(state.ordinal_map().is_none());
    });
}

#[test]
fn plain_and_concurrent_counts_agree_on_every_path() {
    let c = config();
    let multi = build(&c, docs(), &[9, 29], &["odd"]);
    let single = build(&c, docs(), &[], &[]);
    for idx in [&multi, &single] {
        idx.with(|searcher, reader, _| {
            let state = DefaultSortedSetDocValuesReaderState::new(reader, &c).unwrap();
            let all = SortedSetDocValuesFacetCounts::new(&state).unwrap();
            let all_c = ConcurrentSortedSetDocValuesFacetCounts::new(&state).unwrap();
            // Few hits against 30 tags remap as they go; all hits migrate.
            for q in [match_all(reader), term("id", "even"), term("id", "none")] {
                let hits = collect(searcher, &q, false);
                let plain = SortedSetDocValuesFacetCounts::with_hits(&state, &hits).unwrap();
                let conc =
                    ConcurrentSortedSetDocValuesFacetCounts::with_hits(&state, &hits).unwrap();
                for (dim, path) in [("Author", &[][..]), ("Tag", &[]), ("Date", &["2010"])] {
                    assert_eq!(top(&plain, dim, path), top(&conc, dim, path));
                }
                assert_eq!(
                    plain.get_all_dims(10).unwrap(),
                    conc.get_all_dims(10).unwrap()
                );
                assert_eq!(
                    plain.get_top_dims(2, 3).unwrap(),
                    conc.get_top_dims(2, 3).unwrap()
                );
                // With nothing counted the concurrent counter still holds its
                // (zero) counts, as Java's `AtomicIntegerArray` does.
                let (p, cc) = (
                    plain.get_all_children("Author", &[]).unwrap(),
                    conc.get_all_children("Author", &[]).unwrap(),
                );
                if hits.matching_docs().iter().any(|m| m.total_hits > 0) {
                    assert_eq!(p, cc);
                } else {
                    assert_eq!(p, None);
                    assert_eq!(cc.map(|r| r.value.int_value()), Some(0));
                }
                assert_eq!(
                    plain.get_specific_value("Author", &["a"]).unwrap(),
                    conc.get_specific_value("Author", &["a"]).unwrap()
                );
            }
            let hits = collect(searcher, &match_all(reader), false);
            let plain = SortedSetDocValuesFacetCounts::with_hits(&state, &hits).unwrap();
            assert_eq!(top(&plain, "Tag", &[]), top(&all, "Tag", &[]));
            assert_eq!(top(&all_c, "Date", &["2010"]), top(&all, "Date", &["2010"]));
            let author = top(&all, "Author", &[]).unwrap();
            // Every live document but the one without facets.
            assert_eq!(author.value.int_value(), reader.num_docs() - 1);
        });
    }
}

#[test]
fn few_hits_remap_as_they_come() {
    let c = config();
    let idx = build(&c, docs(), &[9, 29], &[]);
    idx.with(|searcher, reader, _| {
        let state = DefaultSortedSetDocValuesReaderState::new(reader, &c).unwrap();
        // One hit in a segment of some 30 ordinals: 1 < 30 / 10, so each
        // ordinal is remapped as it comes.
        let hits = collect(searcher, &term("id", "five"), false);
        assert_eq!(
            hits.matching_docs()
                .iter()
                .map(|m| m.total_hits)
                .sum::<i32>(),
            1
        );
        for f in [
            &SortedSetDocValuesFacetCounts::with_hits(&state, &hits).unwrap() as &dyn Facets,
            &ConcurrentSortedSetDocValuesFacetCounts::with_hits(&state, &hits).unwrap(),
        ] {
            let tag = top(f, "Tag", &[]).unwrap();
            let labels: Vec<&str> = tag.label_values.iter().map(|l| l.label.as_str()).collect();
            assert_eq!(labels, ["t05", "t12"]);
            assert_eq!(top(f, "Author", &[]).unwrap().value.int_value(), 1);
        }
    });
}

#[test]
fn hits_from_another_reader_are_refused() {
    let c = config();
    let a = build(&c, docs(), &[9], &[]);
    let b = build(&c, docs(), &[9], &[]);
    a.with(|_, reader_a, _| {
        let state = DefaultSortedSetDocValuesReaderState::new(reader_a, &c).unwrap();
        b.with(|searcher_b, reader_b, _| {
            let hits = collect(searcher_b, &match_all(reader_b), false);
            for e in [
                SortedSetDocValuesFacetCounts::with_hits(&state, &hits).err(),
                ConcurrentSortedSetDocValuesFacetCounts::with_hits(&state, &hits).err(),
            ] {
                assert!(matches!(e, Some(Error::IllegalState(m)) if m == STATE_MISMATCH));
            }
        });
    });
}

#[test]
fn a_leaf_outside_the_ordinal_map_is_refused() {
    let c = config();
    let idx = build(&c, docs(), &[9], &[]);
    idx.with(|_, reader, _| {
        let map = OrdinalMap::build::<Vec<u8>>(&[Vec::new()]);
        let leaves = reader.leaves();
        let mut counts = vec![0i32; 64];
        let e = count_one_segment(
            "$facets",
            Some(&map),
            &leaves[1],
            None,
            counts.as_mut_slice(),
        )
        .unwrap_err();
        assert!(matches!(e, Error::IllegalState(m) if m == "leaf 1 is not in the ordinal map"));
        // A map too small for the leaf's ordinals: the remap refuses.
        let e = count_one_segment(
            "$facets",
            Some(&map),
            &leaves[0],
            None,
            counts.as_mut_slice(),
        )
        .unwrap_err();
        assert!(matches!(e, Error::IllegalState(_)));
        // Counts too small for the leaf's ordinals.
        let e =
            count_one_segment("$facets", None, &leaves[0], None, &mut [0i32; 1][..]).unwrap_err();
        assert!(matches!(e, Error::IllegalState(_)));
    });
}
