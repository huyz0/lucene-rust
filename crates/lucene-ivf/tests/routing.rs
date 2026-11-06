//! Centroid routing through the IVF reader: the stored routing tree, its
//! format checks, and recall against exact routing.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_ivf::{
    merge_field, Error, FieldVectors, IvfFiles, IvfParams, IvfReader, IvfWriter, MergeSource,
    Routing, SearchParams, Similarity,
};
use lucene_vector_quant::osq::OsqFactory;
use lucene_vector_quant::QuantizerRegistry;

const ID: [u8; 16] = [4; 16];
const SIMS: [Similarity; 4] = [
    Similarity::Euclidean,
    Similarity::Cosine,
    Similarity::DotProduct,
    Similarity::MaximumInnerProduct,
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

fn dataset(n: usize, dim: usize, sim: Similarity, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    let mut out: Vec<f32> = (0..n * dim).map(|_| rng.next()).collect();
    if sim == Similarity::DotProduct {
        for v in out.chunks_exact_mut(dim) {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
        }
    }
    out
}

/// Small clusters and a low tree threshold, so a test-sized field stores a
/// routing tree of a few parents.
fn params() -> IvfParams {
    IvfParams {
        target_cluster_size: 16,
        routing_tree_min: 64,
        ..IvfParams::default()
    }
}

fn write(sim: Similarity, dim: usize, data: &[f32], params: &IvfParams) -> IvfFiles {
    let docs: Vec<i32> = (0..(data.len() / dim) as i32).collect();
    let mut w = IvfWriter::new(&ID, "");
    w.add_field(
        &FieldVectors {
            number: 1,
            similarity: sim,
            dimension: dim,
            docs: &docs,
            vectors: data,
        },
        &OsqFactory::bbq(),
        params,
    )
    .unwrap();
    w.finish()
}

fn open_meta(meta: &[u8], files: &IvfFiles) -> lucene_ivf::Result<IvfReader> {
    IvfReader::open(
        meta,
        Arc::new(files.postings.clone()),
        Arc::new(files.vectors.clone()),
        &ID,
        "",
        &QuantizerRegistry::with_builtins(),
    )
}

/// `file` under a fresh header at `version` and footer, its payload passed
/// through `edit`.
fn rewrite(file: &[u8], codec: &str, version: i32, edit: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let header = lucene_store::codec_util::index_header_length(codec, "");
    let mut payload = file[header..file.len() - 16].to_vec();
    edit(&mut payload);
    let mut out = Vec::new();
    lucene_store::codec_util::write_index_header(&mut out, codec, version, &ID, "");
    out.extend_from_slice(&payload);
    lucene_store::codec_util::write_footer(&mut out);
    out
}

fn ids(r: &IvfReader, q: &[f32], p: &SearchParams) -> Vec<i32> {
    r.search(1, q, p, None)
        .unwrap()
        .iter()
        .map(|h| h.doc)
        .collect()
}

#[test]
fn stored_tree_routes_close_to_exact() {
    let (dim, n) = (16, 3000);
    for sim in SIMS {
        let data = dataset(n, dim, sim, 11);
        let files = write(sim, dim, &data, &params());
        let mut r = open_meta(&files.meta, &files).unwrap();
        let f = r.field(1).unwrap();
        assert!(f.clusters() >= 64, "{}", f.clusters());
        assert!(f.has_routing_tree(), "{sim:?}");
        let p = SearchParams::new(10, 24, 100);
        let queries: Vec<&[f32]> = data.chunks_exact(dim).step_by(97).collect();
        let tree: Vec<Vec<i32>> = queries.iter().map(|q| ids(&r, q, &p)).collect();
        r.set_routing(Routing::Exact);
        let mut same = 0;
        let mut total = 0;
        for (q, t) in queries.iter().zip(&tree) {
            let exact = ids(&r, q, &p);
            same += t.iter().filter(|d| exact.contains(d)).count();
            total += exact.len();
        }
        assert!(same * 10 >= total * 9, "{sim:?}: {same} of {total}");
        // Quantized routing probes the same clusters as exact routing here.
        r.set_routing(Routing::Quantized);
        for q in &queries {
            let quantized = ids(&r, q, &p);
            r.set_routing(Routing::Exact);
            assert_eq!(quantized, ids(&r, q, &p), "{sim:?}");
            r.set_routing(Routing::Quantized);
        }
        // Every cluster is reachable through the tree.
        r.set_routing(Routing::Auto);
        let all = SearchParams::new(n, usize::MAX, 0);
        assert_eq!(r.search(1, queries[0], &all, None).unwrap().len(), n);
    }
}

#[test]
fn fields_below_the_threshold_store_no_tree() {
    let (dim, n) = (8, 800);
    let sim = Similarity::Euclidean;
    let data = dataset(n, dim, sim, 12);
    let files = write(
        sim,
        dim,
        &data,
        &IvfParams {
            routing_tree_min: usize::MAX,
            ..params()
        },
    );
    let mut r = open_meta(&files.meta, &files).unwrap();
    assert!(!r.field(1).unwrap().has_routing_tree());
    // An explicit tree is built at open instead.
    let p = SearchParams::new(5, 8, 50);
    r.set_routing(Routing::Tree { cover: 8 });
    assert_eq!(ids(&r, &data[..dim], &p)[0], 0);
}

#[test]
fn merged_segments_store_a_tree() {
    let (dim, n) = (12, 1500);
    let sim = Similarity::Cosine;
    let data = dataset(n * 2, dim, sim, 13);
    let a = write(sim, dim, &data[..n * dim], &params());
    let b = write(sim, dim, &data[n * dim..], &params());
    let (ra, rb) = (
        open_meta(&a.meta, &a).unwrap(),
        open_meta(&b.meta, &b).unwrap(),
    );
    let mut w = IvfWriter::new(&ID, "");
    let base = n as i32;
    let remap_b = move |d: i32| Some(d + base);
    merge_field(
        &mut w,
        1,
        &[
            MergeSource {
                reader: &ra,
                doc_map: &|d| Some(d),
            },
            MergeSource {
                reader: &rb,
                doc_map: &remap_b,
            },
        ],
        &OsqFactory::bbq(),
        &params(),
    )
    .unwrap();
    let merged = w.finish();
    let r = open_meta(&merged.meta, &merged).unwrap();
    assert!(r.field(1).unwrap().has_routing_tree());
    let hits = ids(
        &r,
        &data[(n + 3) * dim..(n + 4) * dim],
        &SearchParams::new(3, 16, 40),
    );
    assert_eq!(hits[0], base + 3);
}

/// The bytes of the field's stored tree at the end of a one-field meta
/// payload: the parent count, then one vint per cluster (all below 128).
fn tree_tail(r: &IvfReader) -> usize {
    1 + r.field(1).unwrap().clusters()
}

#[test]
fn version_three_files_have_no_tree() {
    let (dim, n) = (8, 1500);
    let sim = Similarity::Euclidean;
    let data = dataset(n, dim, sim, 14);
    let files = write(sim, dim, &data, &params());
    let current = open_meta(&files.meta, &files).unwrap();
    let tail = tree_tail(&current);
    let v3 = lucene_ivf::VERSION_CALIBRATION;
    let meta = rewrite(&files.meta, lucene_ivf::META_CODEC, v3, |p| {
        p.truncate(p.len() - tail)
    });
    let postings = rewrite(&files.postings, lucene_ivf::POSTINGS_CODEC, v3, |_| {});
    let vectors = rewrite(&files.vectors, lucene_ivf::VECTORS_CODEC, v3, |_| {});
    let old = IvfReader::open(
        &meta,
        Arc::new(postings),
        Arc::new(vectors),
        &ID,
        "",
        &QuantizerRegistry::with_builtins(),
    )
    .unwrap();
    assert!(!old.field(1).unwrap().has_routing_tree());
    let mut current = current;
    current.set_routing(Routing::Quantized);
    let p = SearchParams::new(10, 12, 60);
    for q in data.chunks_exact(dim).step_by(101) {
        assert_eq!(ids(&old, q, &p), ids(&current, q, &p));
    }
}

#[test]
fn corrupt_trees_are_rejected() {
    let (dim, n) = (8, 1500);
    let sim = Similarity::Cosine;
    let data = dataset(n, dim, sim, 15);
    let files = write(sim, dim, &data, &params());
    let r = open_meta(&files.meta, &files).unwrap();
    let tail = tree_tail(&r);
    let clusters = r.field(1).unwrap().clusters();
    let edit = |change: &dyn Fn(&mut Vec<u8>)| {
        let meta = rewrite(
            &files.meta,
            lucene_ivf::META_CODEC,
            lucene_ivf::VERSION_CURRENT,
            change,
        );
        open_meta(&meta, &files)
    };
    edit(&|_| {}).unwrap();
    let count_at = |p: &Vec<u8>| p.len() - tail;
    type Change<'a> = Box<dyn Fn(&mut Vec<u8>) + 'a>;
    let cases: Vec<(Change<'_>, &str)> = vec![
        (
            Box::new(|p| {
                let at = count_at(p) + 1;
                p[at] = 120;
            }),
            "out of range",
        ),
        (
            Box::new(|p| {
                // One more parent than any centroid names.
                let at = count_at(p);
                p[at] += 1;
            }),
            "empty routing parent",
        ),
        (
            Box::new(|p| {
                let at = count_at(p);
                p.truncate(at);
                // More parents than clusters (a two-byte vint).
                let big = clusters + 1;
                p.push((big as u8 & 0x7f) | 0x80);
                p.push((big >> 7) as u8);
                p.extend(std::iter::repeat_n(0u8, clusters));
            }),
            "routing parents",
        ),
        (
            Box::new(|p| {
                let at = count_at(p);
                p.truncate(at + 3);
            }),
            "3 routing parents for",
        ),
    ];
    for (change, message) in cases {
        match edit(&*change) {
            Err(Error::Corrupted(m)) => assert!(m.contains(message), "{m}"),
            Err(e) => panic!("{message}: {e}"),
            Ok(_) => panic!("{message} accepted"),
        }
    }
    // A zero parent count drops the tree.
    let r = edit(&|p| {
        let at = count_at(p);
        p.truncate(at);
        p.push(0);
    })
    .unwrap();
    assert!(!r.field(1).unwrap().has_routing_tree());
}
