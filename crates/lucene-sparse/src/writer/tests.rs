#![cfg(test)]
use super::*;

fn vector(pairs: &[(u32, u8)]) -> Option<SparseVector> {
    Some(SparseVector::from_items(pairs.to_vec()))
}

#[test]
fn feature_values_drop_low_mantissa_bits() {
    assert_eq!(feature_value(1.0), 1.0);
    let v = f32::from_bits(1.0f32.to_bits() | 0x7FFF);
    assert_eq!(feature_value(v), 1.0);
}

#[test]
fn postings_per_term_scales_when_unset() {
    let p = ClusteringParams {
        n_postings: -1,
        cluster_ratio: 0.1,
        summary_prune_ratio: 0.4,
    };
    assert_eq!(p.postings_per_term(1000), 160);
    assert_eq!(p.postings_per_term(1_000_000), 500);
    let fixed = ClusteringParams { n_postings: 7, ..p };
    assert_eq!(fixed.postings_per_term(1_000_000), 7);
    let negative = ClusteringParams {
        n_postings: -5,
        ..p
    };
    assert_eq!(negative.postings_per_term(10), 0);
}

#[test]
fn top_k_keeps_the_heaviest_in_doc_order() {
    let p = [(0, 5), (1, 9), (2, 5), (3, 1)];
    assert_eq!(top_k(&p, 2), vec![(0, 5), (1, 9)]);
    assert_eq!(top_k(&p, 9), p.to_vec());
    assert!(top_k(&p, 0).is_empty());
}

#[test]
fn summaries_keep_the_heaviest_prefix() {
    let forward = vec![
        vector(&[(1, 10), (2, 50)]),
        vector(&[(1, 30), (3, 5)]),
        None,
    ];
    // Max per token: 1 -> 30, 2 -> 50, 3 -> 5; total 85, threshold 34.
    let s = summarize(&[(0, 1), (1, 1), (2, 1)], &forward, 0.4).unwrap();
    assert_eq!(s.tokens(), &[2]);
    let all = summarize(&[(0, 1), (1, 1)], &forward, 1.0).unwrap();
    assert_eq!(all.tokens(), &[1, 2, 3]);
    assert_eq!(summarize(&[(2, 1)], &forward, 0.4), None);
}

#[test]
fn short_postings_are_one_unskippable_cluster() {
    let params = ClusteringParams {
        n_postings: -1,
        cluster_ratio: 0.5,
        summary_prune_ratio: 0.4,
    };
    let p: Vec<(i32, u8)> = (0..9).map(|d| (d, 1)).collect();
    let c = cluster(&p, &[], &params, 100, &mut SplitMix(1));
    assert_eq!(c.len(), 1);
    assert!(c[0].should_not_skip);
    assert_eq!(c[0].docs.len(), 9);
    assert!(cluster(&[], &[], &params, 100, &mut SplitMix(1)).is_empty());
    let zero = ClusteringParams {
        cluster_ratio: 0.0,
        ..params
    };
    let long: Vec<(i32, u8)> = (0..20).map(|d| (d, 1)).collect();
    assert_eq!(cluster(&long, &[], &zero, 100, &mut SplitMix(1)).len(), 1);
}

#[test]
fn clustering_assigns_every_doc_with_a_vector_once() {
    let params = ClusteringParams {
        n_postings: -1,
        cluster_ratio: 0.2,
        summary_prune_ratio: 0.4,
    };
    // Two groups: even docs share token 1, odd docs token 2. Doc 19 has no vector.
    let forward: Vec<Option<SparseVector>> = (0..20)
        .map(|d| {
            if d == 19 {
                None
            } else if d % 2 == 0 {
                vector(&[(1, 100), (u32::try_from(d).unwrap() + 10, 1)])
            } else {
                vector(&[(2, 100)])
            }
        })
        .collect();
    let p: Vec<(i32, u8)> = (0..20).map(|d| (d, 7)).collect();
    let c = cluster(&p, &forward, &params, 100, &mut SplitMix(42));
    assert!(!c.is_empty() && c.len() <= 4);
    let mut docs: Vec<i32> = c.iter().flat_map(|c| c.docs.clone()).collect();
    docs.sort_unstable();
    assert_eq!(docs, (0..19).collect::<Vec<_>>());
    for cl in &c {
        assert!(!cl.should_not_skip);
        assert!(cl.summary.is_some());
        assert!(cl.docs.windows(2).all(|w| w[0] < w[1]));
        assert!(cl.weights.iter().all(|&w| w == 7));
    }
    // Same seed, same clusters.
    assert_eq!(c, cluster(&p, &forward, &params, 100, &mut SplitMix(42)));
}

fn sparse_field(attrs: &[(&str, &str)]) -> FieldInfo {
    // The first of a repeated key wins.
    let mut all = attrs.to_vec();
    all.extend([
        ("sparse_vector_field", "true"),
        ("approximate_threshold", "0"),
    ]);
    FieldInfo::new("emb", 4).with_attributes(
        all.into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
    )
}

#[test]
fn bad_attributes_are_refused() {
    assert!(ClusteringParams::from_field_info(&sparse_field(&[("n_postings", "x")])).is_err());
    assert!(ClusteringParams::from_field_info(&sparse_field(&[("cluster_ratio", "x")])).is_err());
    assert!(
        ClusteringParams::from_field_info(&sparse_field(&[("summary_prune_ratio", "x")])).is_err()
    );
    let p = ClusteringParams::from_field_info(&sparse_field(&[])).unwrap();
    assert_eq!(p.n_postings, -1);
}

#[test]
fn a_field_without_doc_values_has_no_terms_and_errors_surface_as_io() {
    use lucene_store::Directory;
    let dir = lucene_store::ByteBuffersDirectory::new();
    let id = [3u8; 16];
    let fields = [sparse_field(&[])];
    let write = |fields: &[FieldInfo]| {
        ClusteredPostingsWriter.write(SegmentWrite {
            directory: &dir,
            segment: "_0",
            id: &id,
            max_doc: 5,
            field_infos: fields,
            files: &[],
        })
    };
    assert_eq!(write(&fields).unwrap(), ["_0.sit", "_0.sip"]);
    let sit = dir.open("_0.sit").unwrap();
    let sip = std::sync::Arc::new(dir.open("_0.sip").unwrap());
    let p = postings::ClusteredPostings::open(&sit, sip, &id).unwrap();
    assert!(p.has_field(4));
    assert_eq!(p.clusters(4, b"1").unwrap(), None);
    let bad = [sparse_field(&[("n_postings", "x")])];
    let dir2 = lucene_store::ByteBuffersDirectory::new();
    let mut info = bad[0].clone();
    info.attributes
        .push((PER_FIELD_FORMAT_KEY.into(), "Lucene90".into()));
    info.attributes
        .push((PER_FIELD_SUFFIX_KEY.into(), "0".into()));
    let missing = ClusteredPostingsWriter.write(SegmentWrite {
        directory: &dir2,
        segment: "_0",
        id: &id,
        max_doc: 5,
        field_infos: &[info],
        files: &[],
    });
    assert!(missing.is_err());
    let corrupt = [sparse_field(&[("approximate_threshold", "x")])];
    let err = write(&corrupt).unwrap_err();
    assert!(err.to_string().contains("approximate_threshold"), "{err}");
}

#[test]
fn the_generator_stays_below_its_bound() {
    let mut r = SplitMix(seed(&[1; 16], 3, b"12"));
    for bound in 1..50 {
        assert!(r.below(bound) < bound);
    }
    assert_ne!(seed(&[1; 16], 3, b"12"), seed(&[1; 16], 3, b"13"));
}
