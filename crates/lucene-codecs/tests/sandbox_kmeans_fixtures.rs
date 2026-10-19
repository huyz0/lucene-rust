//! lucene-sandbox's `KMeans` and `SampleReader` against Java:
//! `GenSandboxKMeans` clusters three vector sets under every initialization
//! method and option shape, and records the centroids' float bits and every
//! vector's centroid; here the same calls must give the same bits.
//!
//! FORGY's centroid order follows a process-wide hash seed in Java, so both
//! sides write those cases canonically: centroids sorted by their bit
//! strings, assignments renumbered to match.

// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::fs;
use std::path::PathBuf;

use lucene_codecs::quantization::{
    cluster, cluster_with, reservoir_sample, reservoir_sample_from_array, ClusterOptions,
    FloatVectorSource, InMemoryFloatVectors, KmeansInitializationMethod, Results, SampleReader,
};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/sandbox_kmeans")
}

fn read(name: &str) -> String {
    fs::read_to_string(dir().join(name)).unwrap()
}

fn parse_bits(line: &str) -> Vec<f32> {
    line.split(',')
        .map(|h| f32::from_bits(u32::from_str_radix(h, 16).unwrap()))
        .collect()
}

fn bits(v: &[f32]) -> String {
    v.iter()
        .map(|x| format!("{:x}", x.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn vectors(set: &str) -> Vec<Vec<f32>> {
    read(&format!("vectors.{set}.txt"))
        .lines()
        .map(parse_bits)
        .collect()
}

fn render(res: &Results, canonical: bool) -> Vec<String> {
    let mut centroids = res.centroids.clone();
    let mut assign = res.vector_centroids.clone();
    if canonical {
        let mut order: Vec<usize> = (0..centroids.len()).collect();
        order.sort_by_key(|&i| bits(&centroids[i]));
        let mut rank = vec![0i16; order.len()];
        for (r, &i) in order.iter().enumerate() {
            rank[i] = r as i16;
        }
        centroids = order.iter().map(|&i| centroids[i].clone()).collect();
        assign = assign.map(|a| a.iter().map(|&c| rank[c as usize]).collect());
    }
    let mut lines = vec![format!(
        "centroids {} {}",
        centroids.len(),
        centroids[0].len()
    )];
    lines.extend(centroids.iter().map(|c| bits(c)));
    lines.push(match assign {
        None => "assign null".to_string(),
        Some(a) => format!(
            "assign {} {}",
            a.len(),
            a.iter().map(i16::to_string).collect::<Vec<_>>().join(",")
        ),
    });
    lines
}

fn check(name: &str, res: Option<Results>, canonical: bool) {
    let got = render(&res.unwrap(), canonical);
    let want: Vec<String> = read(&format!("case.{name}.txt"))
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(got.len(), want.len(), "{name}: line count");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "{name}: line {i}");
    }
}

fn opts(
    assign: bool,
    seed: i64,
    method: KmeansInitializationMethod,
    normalize: bool,
    restarts: usize,
    iters: usize,
    sample_size: usize,
) -> ClusterOptions {
    ClusterOptions {
        assign_centroids_to_vectors: assign,
        seed,
        initialization_method: method,
        normalize_centers: normalize,
        restarts,
        iters,
        sample_size,
    }
}

#[test]
fn clusterings_match_lucene_bit_for_bit() {
    use KmeansInitializationMethod::{Forgy, PlusPlus, ReservoirSampling};
    let (blobs, sphere, dupes) = (vectors("blobs"), vectors("sphere"), vectors("dupes"));
    let a = InMemoryFloatVectors::new(&blobs, 8);
    let b = InMemoryFloatVectors::new(&sphere, 16);
    let c = InMemoryFloatVectors::new(&dupes, 4);

    check("default_euclid", cluster(&a, false, 12).unwrap(), false);
    check("default_cosine", cluster(&b, true, 8).unwrap(), false);
    let cases: [(&str, &InMemoryFloatVectors<'_>, usize, ClusterOptions, bool); 9] = [
        (
            "sampled_pp",
            &a,
            6,
            opts(true, 7, PlusPlus, false, 3, 20, 600),
            false,
        ),
        (
            "reservoir",
            &b,
            10,
            opts(true, 11, ReservoirSampling, true, 2, 10, 100_000),
            false,
        ),
        (
            "forgy",
            &a,
            9,
            opts(true, 13, Forgy, false, 4, 10, 100_000),
            true,
        ),
        (
            "no_assign",
            &a,
            5,
            opts(false, 17, PlusPlus, false, 2, 5, 100_000),
            false,
        ),
        (
            "one_cluster",
            &a,
            1,
            opts(true, 5, PlusPlus, false, 2, 5, 100_000),
            false,
        ),
        (
            "dupes_reservoir",
            &c,
            8,
            opts(true, 19, ReservoirSampling, false, 3, 10, 100_000),
            false,
        ),
        (
            "shrunk",
            &c,
            50,
            opts(true, 29, PlusPlus, false, 2, 10, 100_000),
            false,
        ),
        (
            "zero_iters",
            &a,
            4,
            opts(true, 23, PlusPlus, false, 2, 0, 100_000),
            false,
        ),
        (
            "sampled_forgy_cosine",
            &b,
            3,
            opts(true, 31, Forgy, true, 2, 10, 300),
            true,
        ),
    ];
    for (name, v, k, o, canonical) in cases {
        check(name, cluster_with(v, k, &o).unwrap(), canonical);
    }
}

#[test]
fn reservoir_samples_match_lucene() {
    let blobs = vectors("blobs");
    let a = InMemoryFloatVectors::new(&blobs, 8);
    let arr: Vec<i32> = (0..150).map(|i| i * 3 + 1).collect();
    let join = |xs: Vec<String>| xs.join(",");
    let ints = |xs: Vec<usize>| join(xs.iter().map(usize::to_string).collect());
    let ints32 = |xs: Vec<i32>| join(xs.iter().map(i32::to_string).collect());
    let reader = SampleReader::create(&a, 5, 42).unwrap();
    let mut scratch = vec![0.0; 8];
    let mut line = format!("reader {} {}", reader.size(), reader.dimension());
    for i in 0..reader.size() {
        line.push(' ');
        line.push_str(&bits(reader.vector_value(i, &mut scratch).unwrap()));
    }
    let got = [
        format!(
            "sample 1000 10 5 {}",
            ints(reservoir_sample(1000, 10, 5).unwrap())
        ),
        format!(
            "sample 10 10 5 {}",
            ints(reservoir_sample(10, 10, 5).unwrap())
        ),
        format!(
            "sample 100000 7 42 {}",
            ints(reservoir_sample(100_000, 7, 42).unwrap())
        ),
        format!(
            "array 150 7 9 {}",
            ints32(reservoir_sample_from_array(&arr, 7, 9).unwrap())
        ),
        format!(
            "array 150 150 9 {}",
            ints32(reservoir_sample_from_array(&arr, 150, 9).unwrap())
        ),
        line,
    ];
    let want: Vec<String> = read("reservoir.txt").lines().map(str::to_string).collect();
    assert_eq!(got.to_vec(), want);
}

#[test]
fn argument_errors_match_lucene() {
    let blobs = vectors("blobs");
    let a = InMemoryFloatVectors::new(&blobs, 8);
    let mut got = Vec::new();
    for k in [0, 32_768] {
        let msg = match cluster(&a, false, k) {
            Ok(_) => "none".to_string(),
            Err(e) => e.to_string(),
        };
        got.push(format!("{k}\t{msg}"));
    }
    let empty: Vec<Vec<f32>> = Vec::new();
    let res = cluster(&InMemoryFloatVectors::new(&empty, 4), false, 3).unwrap();
    got.push(format!(
        "empty\t{}",
        if res.is_none() { "null" } else { "some" }
    ));
    let want: Vec<String> = read("errors.txt").lines().map(str::to_string).collect();
    for (g, w) in got.iter().zip(&want) {
        // The error's Display may carry a prefix of this port's own; the
        // message is Java's.
        assert!(g.ends_with(w.split('\t').nth(1).unwrap()), "{g} vs {w}");
        assert_eq!(g.split('\t').next(), w.split('\t').next());
    }
    assert_eq!(got.len(), want.len());
}
