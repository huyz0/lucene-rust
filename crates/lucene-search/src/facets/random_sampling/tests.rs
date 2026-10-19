//! Unit tests.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::facets::taxonomy::FastTaxonomyFacetCounts;
use crate::facets::test_index::{
    basic_config, basic_index, build, collect, doc, ff, match_all, plain_index, term,
};
use crate::facets::{FacetValue, Facets, LabelAndValue};

fn set_bits(fc: &FacetsCollector<'_>) -> Vec<Vec<i32>> {
    fc.matching_docs()
        .iter()
        .map(|m| {
            let mut out = Vec::new();
            let mut it = m.bits.iterator();
            let mut d = it.next_doc().unwrap();
            while d != NO_MORE_DOCS {
                out.push(d);
                d = it.next_doc().unwrap();
            }
            out
        })
        .collect()
}

#[test]
fn xorshift_seeds_and_bounds() {
    // Seed 0 is Java's `0xdeadbeef` int literal, sign-extended.
    let mut zero = XorShift64::new(0);
    let mut beef = XorShift64::new(0xffff_ffff_dead_beef_u64 as i64);
    for _ in 0..8 {
        assert_eq!(zero.random_long(), beef.random_long());
    }
    let mut r = XorShift64::new(42);
    for n in [1, 2, 7, 1000, i32::MAX] {
        for _ in 0..50 {
            let v = r.next_int(n);
            assert!((0..n).contains(&v), "{v} not in [0, {n})");
        }
    }
    // A zero bound does not divide by zero; a negative one draws as
    // Java's `%` does.
    assert_eq!(r.next_int(0), 0);
    let (mut p, mut q) = (XorShift64::new(5), XorShift64::new(5));
    for _ in 0..20 {
        assert_eq!(p.next_int(-7), q.next_int(7));
    }
    let mut a = XorShift64::new(-7);
    let mut b = XorShift64::new(-7);
    assert_eq!(a.next_int(100), b.next_int(100));
}

#[test]
fn a_sample_size_below_minus_the_hits_is_refused() {
    let idx = basic_index();
    idx.with(|searcher, reader, _| {
        // Fewer than 10 hits, at least 2: -10 makes bins of `(int) -h/10 =
        // 0` documents, Java's `/ by zero`; -2 makes negative bins, which
        // Java samples without throwing.
        let mut c = RandomSamplingFacetsCollectorManager::new(-10, 1)
            .collect(searcher, &match_all(reader))
            .unwrap();
        match c.matching_docs() {
            Err(Error::IllegalArgument(m)) => assert!(m.starts_with("/ by zero"), "{m}"),
            other => panic!("{other:?}"),
        }
        let mut c = RandomSamplingFacetsCollectorManager::new(-2, 1)
            .collect(searcher, &match_all(reader))
            .unwrap();
        assert!(c.matching_docs().is_ok());
    });
}

#[test]
fn few_hits_are_not_sampled() {
    let idx = basic_index();
    idx.with(|searcher, reader, _| {
        let mut c = RandomSamplingFacetsCollectorManager::new(10, 1)
            .collect(searcher, &match_all(reader))
            .unwrap();
        assert!(format!("{c:?}").contains("sample_size: 10"));
        let original = set_bits(c.original_matching_docs());
        assert_eq!(set_bits(c.matching_docs().unwrap()), original);
        assert_eq!(c.sampling_rate(), 0.0);
        // Not sampled: the counts come back as they are.
        let r = FacetResult::new(
            "Author",
            Vec::new(),
            FacetValue::Int(5),
            vec![LabelAndValue::new("Lisa", FacetValue::Int(2))],
            4,
        );
        let same = c
            .amortize_facet_counts(Some(r.clone()), &basic_config(), reader)
            .unwrap();
        assert_eq!(same, Some(r));
    });
}

#[test]
fn amortizing_before_sampling_or_nothing_is_the_identity() {
    let idx = basic_index();
    idx.with(|_, reader, _| {
        let c = RandomSamplingFacetsCollector::new(1);
        assert_eq!(
            c.amortize_facet_counts(None, &basic_config(), reader)
                .unwrap(),
            None
        );
        let r = FacetResult::new("Author", Vec::new(), FacetValue::Int(1), Vec::new(), 0);
        assert_eq!(
            c.amortize_facet_counts(Some(r.clone()), &basic_config(), reader)
                .unwrap(),
            Some(r)
        );
    });
}

#[test]
fn a_sample_takes_one_document_per_bin_across_segments() {
    let docs: Vec<_> = (0..100)
        .map(|i| doc(&format!("d{i}"), Vec::new()).1)
        .collect();
    // Segments of 7, 30 and 63 documents: bins straddle the boundaries.
    let idx = plain_index(docs, &[6, 36]);
    idx.with(|searcher, reader, _| {
        assert_eq!(reader.leaves().len(), 3);
        for (size, seed) in [(10, 0), (10, 42), (3, -7), (30, 5), (99, 9)] {
            let mut c = RandomSamplingFacetsCollectorManager::new(size, seed)
                .collect(searcher, &match_all(reader))
                .unwrap();
            let sampled = set_bits(c.matching_docs().unwrap());
            let n: usize = sampled.iter().map(Vec::len).sum();
            let bin = (100 / size) as usize;
            assert!(
                n <= 100 / bin + 1 && n + 1 >= 100 / bin,
                "size {size}: {n} sampled"
            );
            assert_eq!(c.sampling_rate(), f64::from(size) / 100.0);
            // Each sampled segment keeps its total hits.
            let totals: Vec<i32> = c
                .matching_docs()
                .unwrap()
                .matching_docs()
                .iter()
                .map(|m| m.total_hits)
                .collect();
            assert_eq!(totals, vec![7, 30, 63]);
            // Drawn once: a second call returns the same sample.
            assert_eq!(set_bits(c.matching_docs().unwrap()), sampled);
            // Same seed, same sample.
            let mut again = RandomSamplingFacetsCollectorManager::new(size, seed)
                .collect(searcher, &match_all(reader))
                .unwrap();
            assert_eq!(set_bits(again.matching_docs().unwrap()), sampled);
        }
    });
}

#[test]
fn amortized_counts_scale_by_the_rate_capped_by_doc_freq() {
    let config = basic_config();
    let docs: Vec<_> = (0..60)
        .map(|i| {
            doc(
                &format!("d{i}"),
                vec![
                    ff("Author", &[["a", "b", "c"][i % 3]]),
                    ff("Publish Date", &["2010", &(1 + i % 2).to_string()]),
                ],
            )
        })
        .collect();
    let idx = build(&config, docs, &[19, 39], &[]);
    idx.with(|searcher, reader, taxo| {
        let mut c = RandomSamplingFacetsCollectorManager::new(6, 3)
            .collect(searcher, &match_all(reader))
            .unwrap();
        let sampled = c.matching_docs().unwrap();
        let counts = FastTaxonomyFacetCounts::new(taxo, &config, sampled).unwrap();
        let author = counts.get_top_children(10, "Author", &[]).unwrap();
        let date = counts
            .get_top_children(10, "Publish Date", &["2010"])
            .unwrap();
        let rate = c.sampling_rate();
        assert_eq!(rate, 0.1);
        let fixed = c
            .amortize_facet_counts(author.clone(), &config, reader)
            .unwrap()
            .unwrap();
        let author = author.unwrap();
        assert_eq!(
            fixed.value.int_value(),
            ((author.value.double_value() / rate) as i32).min(60)
        );
        for (lv, f) in author.label_values.iter().zip(&fixed.label_values) {
            assert_eq!(f.label, lv.label);
            assert_eq!(
                f.value.int_value(),
                ((lv.value.double_value() / rate) as i32).min(20)
            );
        }
        assert_eq!(fixed.child_count, author.child_count);
        // A path under the dim: the children's drill-down terms.
        let fixed = c
            .amortize_facet_counts(date.clone(), &config, reader)
            .unwrap()
            .unwrap();
        assert_eq!(fixed.path, vec!["2010".to_string()]);
        for f in &fixed.label_values {
            assert!(f.value.int_value() <= 30);
        }
        // A zero total stays zero; an empty label is refused.
        let zero = FacetResult::new(
            "Author",
            Vec::new(),
            FacetValue::Int(0),
            vec![LabelAndValue::new("", FacetValue::Int(1))],
            1,
        );
        assert!(matches!(
            c.amortize_facet_counts(Some(zero), &config, reader),
            Err(Error::IllegalArgument(_))
        ));
        let zero = FacetResult::new("Author", Vec::new(), FacetValue::Int(0), Vec::new(), 0);
        let same = c
            .amortize_facet_counts(Some(zero), &config, reader)
            .unwrap()
            .unwrap();
        assert_eq!(same.value.int_value(), 0);
    });
}

#[test]
fn the_manager_reduces_none_one_or_many() {
    let m = RandomSamplingFacetsCollectorManager::new(4, 2);
    let none = m.reduce(Vec::new()).unwrap();
    assert!(none.original_matching_docs().matching_docs().is_empty());
    let idx = basic_index();
    idx.with(|searcher, reader, _| {
        let a = collect(searcher, &term("id", "d0"), false);
        let b = collect(searcher, &term("id", "d4"), false);
        let mut ca = m.new_collector();
        ca.inner = a;
        let mut cb = m.new_collector();
        cb.inner = b;
        let one = m.reduce(vec![ca]).unwrap();
        let hits: i32 = one
            .original_matching_docs()
            .matching_docs()
            .iter()
            .map(|md| md.total_hits)
            .sum();
        assert_eq!(hits, 1);
        let mut ca = m.new_collector();
        ca.inner = collect(searcher, &term("id", "d0"), false);
        let mut many = m.reduce(vec![ca, cb]).unwrap();
        let total: i32 = many
            .matching_docs()
            .unwrap()
            .matching_docs()
            .iter()
            .map(|md| md.total_hits)
            .sum();
        assert_eq!(total, 2);
        assert!(reader.max_doc() >= 5);
    });
}
