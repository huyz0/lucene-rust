// Test arithmetic over small in-memory values; see docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::field_infos::VectorSimilarityFunction;
use crate::vectors::{write_flat_vectors, FieldVectorData, FlatVectorsField, FlatVectorsReader};

const ID: [u8; 16] = *b"kmeans-unit-test";

fn grid(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            (0..dim)
                .map(|d| ((i * 7 + d * 3) % 23) as f32 + (i % 5) as f32 * 10.0)
                .collect()
        })
        .collect()
}

fn opts(method: KmeansInitializationMethod) -> ClusterOptions {
    ClusterOptions {
        initialization_method: method,
        ..ClusterOptions::defaults(false)
    }
}

#[test]
fn off_heap_vectors_cluster_like_in_memory_ones() {
    let vs = grid(600, 5);
    let field = FlatVectorsField {
        field_number: 0,
        similarity: VectorSimilarityFunction::Euclidean,
        dimension: 5,
        docs: (0..600).collect(),
        values: FieldVectorData::Float32(vs.iter().flatten().copied().collect()),
    };
    let (vec, vemf) = write_flat_vectors(&[field], 600, &ID, "").unwrap();
    let reader = FlatVectorsReader::open(&vemf, &vec, &ID, "").unwrap();
    let off_heap = reader.float_vector_values(0).unwrap();
    assert_eq!(FloatVectorSource::size(&off_heap), 600);
    assert_eq!(FloatVectorSource::dimension(&off_heap), 5);
    let in_memory = InMemoryFloatVectors::new(&vs, 5);
    let o = opts(KmeansInitializationMethod::PlusPlus);
    assert_eq!(
        cluster_with(&off_heap, 4, &o).unwrap(),
        cluster_with(&in_memory, 4, &o).unwrap()
    );
    let mut scratch = vec![0.0; 5];
    assert!(FloatVectorSource::vector_value(&off_heap, usize::MAX, &mut scratch).is_err());
    let mut short = vec![0.0; 3];
    assert!(FloatVectorSource::vector_value(&off_heap, 0, &mut short).is_err());
}

#[test]
fn out_of_range_ordinals_are_errors() {
    let vs = grid(10, 3);
    let mem = InMemoryFloatVectors::new(&vs, 3);
    let mut scratch = vec![0.0; 3];
    assert_eq!(mem.vector_value(9, &mut scratch).unwrap(), vs[9].as_slice());
    assert!(mem.vector_value(10, &mut scratch).is_err());

    let sample = SampleReader::create(&mem, 4, 1).unwrap();
    assert_eq!(sample.size(), 4);
    assert_eq!(sample.dimension(), 3);
    let o = sample.origin_ord(2);
    assert_eq!(
        sample.vector_value(2, &mut scratch).unwrap(),
        vs[o].as_slice()
    );
    assert!(sample.vector_value(4, &mut scratch).is_err());
}

#[test]
fn reservoir_sampling_edges() {
    // k > n: Java fills `0..k` regardless.
    assert_eq!(reservoir_sample(2, 4, 0).unwrap(), vec![0, 1, 2, 3]);
    assert!(reservoir_sample(usize::MAX, 1, 0).is_err());
    assert!(reservoir_sample(1, usize::MAX, 0).is_err());
    assert_eq!(
        reservoir_sample_from_array(&[5, 6], 3, 0).unwrap(),
        vec![5, 6]
    );
    let picked = reservoir_sample_from_array(&[1, 2, 3, 4, 5, 6, 7, 8], 3, 7).unwrap();
    assert_eq!(picked.len(), 3);
    assert!(picked.iter().all(|v| (1..=8).contains(v)));
}

#[test]
fn argument_and_degenerate_errors() {
    let vs = grid(300, 4);
    let mem = InMemoryFloatVectors::new(&vs, 4);
    let msg = cluster(&mem, false, 0).unwrap_err().to_string();
    assert!(
        msg.contains("[numClusters] must be between [1] and [32767]"),
        "{msg}"
    );
    let zero_restarts = ClusterOptions {
        restarts: 0,
        ..ClusterOptions::defaults(false)
    };
    assert!(cluster_with(&mem, 2, &zero_restarts).is_err());
    // One centroid never runs a restart, in Java either.
    assert!(cluster_with(&mem, 1, &zero_restarts).unwrap().is_some());

    // k-means++ over vectors that are all one point: Java reads ordinal -1.
    let same = vec![vec![1.0f32, 2.0]; 200];
    let err = cluster(&InMemoryFloatVectors::new(&same, 2), false, 2).unwrap_err();
    assert!(err.to_string().contains("ordinal -1"), "{err}");

    // NaN vectors never win a distance comparison: every vector goes to
    // centroid 0 and the other is refilled from the queue, as in Java.
    let nan = vec![vec![f32::NAN, 0.0]; 200];
    let o = opts(KmeansInitializationMethod::Forgy);
    let res = cluster_with(&InMemoryFloatVectors::new(&nan, 2), 2, &o)
        .unwrap()
        .unwrap();
    assert_eq!(res.vector_centroids, Some(vec![0; 200]));
}

#[test]
fn defaults_are_javas() {
    let d = ClusterOptions::defaults(true);
    assert!(d.assign_centroids_to_vectors && d.normalize_centers);
    assert_eq!(
        (d.seed, d.restarts, d.iters, d.sample_size),
        (42, 5, 10, 100_000)
    );
    assert_eq!(
        d.initialization_method,
        KmeansInitializationMethod::PlusPlus
    );
    assert!(!ClusterOptions::defaults(false).normalize_centers);
}

#[test]
fn normalization_leaves_zero_and_unit_centroids_alone() {
    let normalize = ClusterOptions {
        normalize_centers: true,
        ..ClusterOptions::defaults(true)
    };
    // One cluster whose mean is the origin.
    let opposite: Vec<Vec<f32>> = (0..100)
        .map(|i| {
            if i % 2 == 0 {
                vec![3.0, 0.0]
            } else {
                vec![-3.0, 0.0]
            }
        })
        .collect();
    let res = cluster_with(&InMemoryFloatVectors::new(&opposite, 2), 1, &normalize)
        .unwrap()
        .unwrap();
    assert_eq!(res.centroids, vec![vec![0.0, 0.0]]);
    // One cluster of one unit vector: already unit, untouched.
    let unit = vec![vec![0.6f32, 0.8]; 50];
    let res = cluster_with(&InMemoryFloatVectors::new(&unit, 2), 1, &normalize)
        .unwrap()
        .unwrap();
    assert_eq!(res.centroids, vec![vec![0.6, 0.8]]);
    assert_eq!(res.vector_centroids, Some(vec![0; 50]));
    // Not unit: scaled.
    let long = vec![vec![3.0f32, 4.0]; 50];
    let res = cluster_with(&InMemoryFloatVectors::new(&long, 2), 1, &normalize)
        .unwrap()
        .unwrap();
    assert_eq!(res.centroids, vec![vec![0.6, 0.8]]);
}
