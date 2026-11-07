#![allow(clippy::arithmetic_side_effects)]

use std::borrow::Cow;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use lucene_codecs::field_infos::{
    VectorEncoding as CodecEncoding, VectorSimilarityFunction as CodecSim,
};
use lucene_codecs::hnsw::{HnswGraphView, OnHeapHnswGraph};
use lucene_codecs::hnsw_vectors::{write_hnsw_vectors, HnswVectorsField, HnswVectorsReader};
use lucene_disk_hnsw::{
    build_graph, DiskHnswFiles, DiskHnswParams, DiskHnswReader, DiskHnswWriter, Error,
    FieldVectors, GraphParams, OpenOptions, RangeSource, SearchParams, Similarity, VectorEncoding,
};
use lucene_util::quantization::ScalarEncoding;
use lucene_vector_quant::osq::OsqFactory;
use lucene_vector_quant::rabitq::ExRabitqFactory;
use lucene_vector_quant::scalar::ScalarFactory;
use lucene_vector_quant::{QuantizerFactory, QuantizerRegistry};

const ID: [u8; 16] = *b"disk-hnsw-tests!";

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
}

fn dataset(n: usize, dim: usize, centers: usize, sim: Similarity, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    let cs: Vec<Vec<f32>> = (0..centers)
        .map(|_| (0..dim).map(|_| rng.next() * 4.0).collect())
        .collect();
    let mut out = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = &cs[i % centers];
        let mut v: Vec<f32> = c.iter().map(|x| x + rng.next()).collect();
        if matches!(sim, Similarity::DotProduct) {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
        }
        out.extend(v);
    }
    out
}

fn docs(n: usize) -> Vec<i32> {
    (0..n as i32).map(|i| i * 3 + 1).collect()
}

fn write(
    sim: Similarity,
    dim: usize,
    vectors: &[f32],
    factory: &dyn QuantizerFactory,
    params: &DiskHnswParams,
) -> DiskHnswFiles {
    let n = vectors.len() / dim;
    let docs = docs(n);
    let mut w = DiskHnswWriter::new(&ID, "");
    w.add_field(
        &FieldVectors {
            number: 7,
            similarity: sim,
            dimension: dim,
            docs: &docs,
            vectors,
        },
        factory,
        params,
    )
    .unwrap();
    w.finish()
}

fn open_with(
    files: &DiskHnswFiles,
    options: OpenOptions,
) -> lucene_disk_hnsw::Result<DiskHnswReader> {
    DiskHnswReader::open(
        &files.meta,
        Arc::new(files.codes.clone()),
        Arc::new(files.nodes.clone()),
        &ID,
        "",
        &QuantizerRegistry::with_builtins(),
        options,
    )
}

fn open(files: &DiskHnswFiles) -> DiskHnswReader {
    open_with(files, OpenOptions::default()).unwrap()
}

fn exact_top(sim: Similarity, dim: usize, vectors: &[f32], q: &[f32], k: usize) -> Vec<i32> {
    let docs = docs(vectors.len() / dim);
    let mut all: Vec<(f32, i32)> = vectors
        .chunks_exact(dim)
        .zip(&docs)
        .map(|(v, &d)| (sim.score(q, v), d))
        .collect();
    all.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    all.into_iter().take(k).map(|(_, d)| d).collect()
}

fn recall(
    reader: &DiskHnswReader,
    sim: Similarity,
    dim: usize,
    vectors: &[f32],
    queries: &[f32],
    params: &SearchParams,
) -> f64 {
    let mut found = 0;
    let mut total = 0;
    for q in queries.chunks_exact(dim) {
        let truth = exact_top(sim, dim, vectors, q, params.k);
        let out = reader.search(7, q, params, None).unwrap();
        assert!(out.hits.windows(2).all(|w| w[0].score >= w[1].score));
        for h in &out.hits {
            let i = docs(vectors.len() / dim).binary_search(&h.doc).unwrap();
            let exact = sim.score(q, &vectors[i * dim..][..dim]);
            assert!(
                (exact - h.score).abs() <= 1e-2 * exact.abs().max(1.0),
                "{exact} {}",
                h.score
            );
        }
        found += out.hits.iter().filter(|h| truth.contains(&h.doc)).count();
        total += truth.len();
    }
    found as f64 / total as f64
}

#[test]
fn recalls_over_similarities_quantizers_and_modes() {
    // 1-bit codes rank poorly in few dimensions; 48 is enough for BBQ.
    let dim = 48;
    let n = 1500;
    let factories: Vec<Box<dyn QuantizerFactory>> = vec![
        Box::new(OsqFactory::bbq()),
        Box::new(OsqFactory::new(ScalarEncoding::PackedNibble)),
        Box::new(ScalarFactory::int7()),
        Box::new(ExRabitqFactory::new(4)),
    ];
    for sim in [
        Similarity::Euclidean,
        Similarity::DotProduct,
        Similarity::Cosine,
        Similarity::MaximumInnerProduct,
    ] {
        let vectors = dataset(n, dim, 20, sim, 11);
        let queries = dataset(40, dim, 20, sim, 99);
        for (fi, factory) in factories.iter().enumerate() {
            for (inline, reorder) in [(false, true), (true, true), (false, false)] {
                let params = DiskHnswParams {
                    inline_codes: inline,
                    reorder,
                    ..DiskHnswParams::default()
                };
                let files = write(sim, dim, &vectors, factory.as_ref(), &params);
                let search = SearchParams {
                    k: 10,
                    list_len: 80,
                    beam_width: 4,
                };
                let r = recall(&open(&files), sim, dim, &vectors, &queries, &search);
                // 1-bit codes rank candidates coarsely; maximum inner
                // product graphs navigate worse whatever the codes.
                let floor = match (fi, sim) {
                    (_, Similarity::MaximumInnerProduct) => 0.75,
                    (0, _) => 0.8,
                    _ => 0.95,
                };
                assert!(
                    r >= floor,
                    "{sim:?} quantizer {fi} inline {inline}: recall {r}"
                );
                if inline {
                    let lean = open_with(
                        &files,
                        OpenOptions {
                            resident_codes: false,
                        },
                    )
                    .unwrap();
                    assert!(!lean.fields()[0].resident_codes());
                    assert!(lean.resident_bytes() < open(&files).resident_bytes());
                    let r2 = recall(&lean, sim, dim, &vectors, &queries, &search);
                    assert_eq!(r, r2, "inline and resident codes rank alike");
                }
            }
        }
    }
}

#[test]
fn every_vector_encoding_scores_its_decoded_vector() {
    let dim = 16;
    let sim = Similarity::Euclidean;
    let vectors = dataset(600, dim, 8, sim, 3);
    let queries = dataset(20, dim, 8, sim, 4);
    for encoding in VectorEncoding::ALL {
        let params = DiskHnswParams {
            encoding,
            ..DiskHnswParams::default()
        };
        let files = write(sim, dim, &vectors, &OsqFactory::bbq(), &params);
        let reader = open(&files);
        assert_eq!(reader.fields()[0].encoding(), encoding);
        let r = recall(
            &reader,
            sim,
            dim,
            &vectors,
            &queries,
            &SearchParams::default(),
        );
        assert!(r >= 0.9, "{encoding:?}: {r}");
    }
}

#[test]
fn fields_report_their_geometry() {
    let dim = 8;
    let sim = Similarity::Cosine;
    let vectors = dataset(300, dim, 4, sim, 5);
    let files = write(
        sim,
        dim,
        &vectors,
        &OsqFactory::bbq(),
        &DiskHnswParams::default(),
    );
    let reader = open(&files);
    reader.check_integrity().unwrap();
    let f = reader.field(7).unwrap();
    assert!(reader.field(8).is_none());
    assert_eq!(f.number(), 7);
    assert_eq!(f.similarity(), sim);
    assert_eq!(f.dimension(), dim);
    assert_eq!(f.count(), 300);
    assert_eq!(f.max_conn(), 16);
    assert!(f.num_levels() >= 1);
    assert_eq!(f.quantizer().name(), "osq");
    assert_eq!(f.nodes_offset() % 4096, 0);
    assert_eq!(f.layout().max_neighbors(), 32);
    assert!(f.resident_codes());
    assert!(format!("{reader:?}").contains("DiskHnswReader"));
}

#[test]
fn filters_keep_traversing() {
    let dim = 12;
    let sim = Similarity::Euclidean;
    let vectors = dataset(1200, dim, 10, sim, 8);
    let files = write(
        sim,
        dim,
        &vectors,
        &OsqFactory::bbq(),
        &DiskHnswParams::default(),
    );
    let reader = open(&files);
    let q = &dataset(1, dim, 10, sim, 77)[..];
    let even = |d: i32| d % 2 == 0;
    let out = reader
        .search(
            7,
            q,
            &SearchParams {
                list_len: 200,
                ..SearchParams::default()
            },
            Some(&even),
        )
        .unwrap();
    assert_eq!(out.hits.len(), 10);
    assert!(out.hits.iter().all(|h| h.doc % 2 == 0));
    let none = |_: i32| false;
    let out = reader
        .search(7, q, &SearchParams::default(), Some(&none))
        .unwrap();
    assert!(out.hits.is_empty());
    assert!(out.stats.expanded > 0);
}

#[test]
fn stats_count_rounds_and_reads() {
    let dim = 16;
    let sim = Similarity::Euclidean;
    let vectors = dataset(2000, dim, 10, sim, 21);
    let files = write(
        sim,
        dim,
        &vectors,
        &OsqFactory::bbq(),
        &DiskHnswParams::default(),
    );
    let reader = open(&files);
    let q = &dataset(1, dim, 10, sim, 22)[..];
    let one = reader
        .search(
            7,
            q,
            &SearchParams {
                beam_width: 1,
                ..SearchParams::default()
            },
            None,
        )
        .unwrap()
        .stats;
    let four = reader
        .search(7, q, &SearchParams::default(), None)
        .unwrap()
        .stats;
    assert_eq!(one.rounds, one.expanded);
    assert!(four.rounds * 3 < four.expanded, "{four:?}");
    assert!(one.expanded >= 100);
    assert_eq!(
        four.bytes_read,
        four.expanded * reader.fields()[0].layout().record_len()
    );
    assert!(four.estimated > four.expanded);
}

/// Counts reads and prefetches, and serves unaligned copies.
struct Counting {
    bytes: Vec<u8>,
    reads: AtomicUsize,
    prefetches: AtomicUsize,
}

impl RangeSource for Counting {
    fn len(&self) -> usize {
        self.bytes.len()
    }

    fn read(&self, offset: usize, len: usize) -> lucene_ivf::Result<Cow<'_, [u8]>> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        let mut v = Vec::with_capacity(len + 1);
        v.push(0);
        v.extend_from_slice(&self.bytes[offset..offset + len]);
        Ok(Cow::Owned(v[1..].to_vec()))
    }

    fn prefetch(&self, _: usize, _: usize) {
        self.prefetches.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn reads_one_range_per_expanded_node() {
    let dim = 20;
    let sim = Similarity::DotProduct;
    let vectors = dataset(1000, dim, 10, sim, 31);
    let files = write(
        sim,
        dim,
        &vectors,
        &OsqFactory::bbq(),
        &DiskHnswParams::default(),
    );
    let nodes = Arc::new(Counting {
        bytes: files.nodes.clone(),
        reads: AtomicUsize::new(0),
        prefetches: AtomicUsize::new(0),
    });
    let reader = DiskHnswReader::open(
        &files.meta,
        Arc::new(files.codes.clone()),
        nodes.clone(),
        &ID,
        "",
        &QuantizerRegistry::with_builtins(),
        OpenOptions::default(),
    )
    .unwrap();
    let opened = nodes.reads.load(Ordering::Relaxed);
    let q = &dataset(1, dim, 10, sim, 32)[..];
    let out = reader.search(7, q, &SearchParams::default(), None).unwrap();
    assert_eq!(
        nodes.reads.load(Ordering::Relaxed) - opened,
        out.stats.expanded
    );
    assert_eq!(nodes.prefetches.load(Ordering::Relaxed), out.stats.expanded);
    // Unaligned copies decode the same as the in-memory file.
    let plain = open(&files)
        .search(7, q, &SearchParams::default(), None)
        .unwrap();
    assert_eq!(plain.hits, out.hits);
}

#[test]
fn converts_a_stock_lucene99_graph() {
    let dim = 48;
    let sim = Similarity::Euclidean;
    let n = 1200;
    let vectors = dataset(n, dim, 12, sim, 41);
    let graph: OnHeapHnswGraph = build_graph(&vectors, dim, sim, &GraphParams::default()).unwrap();
    let (vex, vem) = write_hnsw_vectors(
        &[HnswVectorsField {
            field_number: 3,
            encoding: CodecEncoding::Float32,
            similarity: CodecSim::Euclidean,
            dimension: dim as i32,
            count: n as i32,
            graph: Some(&graph),
            m: 16,
        }],
        &ID,
        "",
    )
    .unwrap();
    let stock = HnswVectorsReader::open(&vem, &vex, &ID, "").unwrap();
    let off_heap = stock.graph(3).unwrap().unwrap();
    let docs = docs(n);
    let field = FieldVectors {
        number: 7,
        similarity: sim,
        dimension: dim,
        docs: &docs,
        vectors: &vectors,
    };
    let params = DiskHnswParams {
        reorder: false,
        ..DiskHnswParams::default()
    };
    let mut w = DiskHnswWriter::new(&ID, "");
    w.add_field_with_graph(&field, &off_heap, &OsqFactory::bbq(), &params)
        .unwrap();
    let from_stock = w.finish();
    let mut w = DiskHnswWriter::new(&ID, "");
    w.add_field_with_graph(&field, &graph, &OsqFactory::bbq(), &params)
        .unwrap();
    let from_heap = w.finish();
    assert!(from_stock.meta == from_heap.meta);
    assert!(from_stock.nodes == from_heap.nodes);
    assert!(from_stock.codes == from_heap.codes);

    // Without reordering, record i holds node i's stock neighbours.
    let layout = open(&from_stock).fields()[0].layout().to_owned();
    let start = open(&from_stock).fields()[0].nodes_offset();
    let mut nbs = Vec::new();
    for node in [0usize, 17, n - 1] {
        let rec =
            &from_stock.nodes[start + layout.record_offset(node).unwrap()..][..layout.record_len()];
        let count = u16::from_le_bytes([rec[4], rec[5]]) as usize;
        let stored: Vec<i32> = rec[layout.neighbors_start()..]
            .chunks_exact(4)
            .take(count)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        off_heap.neighbors_into(0, node as i32, &mut nbs).unwrap();
        nbs.sort_unstable();
        assert_eq!(stored, nbs);
        assert_eq!(
            i32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]),
            docs[node]
        );
    }
    let queries = dataset(30, dim, 12, sim, 42);
    let r = recall(
        &open(&from_stock),
        sim,
        dim,
        &vectors,
        &queries,
        &SearchParams::default(),
    );
    assert!(r >= 0.8, "{r}");
}

#[test]
fn several_fields_and_an_empty_one() {
    let dim = 8;
    let sim = Similarity::Euclidean;
    let a = dataset(400, dim, 5, sim, 51);
    let b = dataset(300, 4, 5, sim, 52);
    let da = docs(400);
    let db = docs(300);
    let mut w = DiskHnswWriter::new(&ID, "s");
    let params = DiskHnswParams::default();
    let bbq = OsqFactory::bbq();
    w.add_field(
        &FieldVectors {
            number: 1,
            similarity: sim,
            dimension: dim,
            docs: &da,
            vectors: &a,
        },
        &bbq,
        &params,
    )
    .unwrap();
    w.add_field(
        &FieldVectors {
            number: 2,
            similarity: sim,
            dimension: 4,
            docs: &[],
            vectors: &[],
        },
        &bbq,
        &params,
    )
    .unwrap();
    w.add_field(
        &FieldVectors {
            number: 3,
            similarity: sim,
            dimension: 4,
            docs: &db,
            vectors: &b,
        },
        &ScalarFactory::int7(),
        &DiskHnswParams {
            page_len: 1024,
            inline_codes: true,
            ..params
        },
    )
    .unwrap();
    let dup = w.add_field(
        &FieldVectors {
            number: 3,
            similarity: sim,
            dimension: 4,
            docs: &db,
            vectors: &b,
        },
        &bbq,
        &params,
    );
    assert!(matches!(dup, Err(Error::IllegalArgument(_))));
    let files = w.finish();
    let reader = DiskHnswReader::open(
        &files.meta,
        Arc::new(files.codes.clone()),
        Arc::new(files.nodes.clone()),
        &ID,
        "s",
        &QuantizerRegistry::with_builtins(),
        OpenOptions::default(),
    )
    .unwrap();
    assert_eq!(reader.fields().len(), 3);
    let empty = reader
        .search(2, &[0.0; 4], &SearchParams::default(), None)
        .unwrap();
    assert!(empty.hits.is_empty());
    for (number, vectors, d, docs) in [(1, &a, dim, &da), (3, &b, 4, &db)] {
        let q = &vectors[5 * d..][..d];
        let out = reader
            .search(number, q, &SearchParams::default(), None)
            .unwrap();
        assert_eq!(out.hits[0].doc, docs[5], "field {number}");
    }
    assert_eq!(reader.field(3).unwrap().nodes_offset() % 1024, 0);
    assert!(reader
        .search(9, &[0.0; 4], &SearchParams::default(), None)
        .is_err());
    assert!(reader
        .search(1, &[0.0; 3], &SearchParams::default(), None)
        .is_err());
}

#[test]
fn writer_rejects_bad_fields() {
    let bbq = OsqFactory::bbq();
    let p = DiskHnswParams::default();
    let sim = Similarity::Euclidean;
    let cases: Vec<(Vec<i32>, Vec<f32>, usize, DiskHnswParams)> = vec![
        (vec![0, 1], vec![0.0; 4], 0, p),
        (vec![0, 1], vec![0.0; 3], 2, p),
        (vec![1, 0], vec![0.0; 4], 2, p),
        (vec![-1, 0], vec![0.0; 4], 2, p),
        (vec![0, 1], vec![0.0, f32::NAN, 0.0, 0.0], 2, p),
        (
            vec![0, 1],
            vec![0.0; 4],
            2,
            DiskHnswParams { page_len: 6, ..p },
        ),
        (
            vec![0, 1],
            vec![0.0; 4],
            2,
            DiskHnswParams { page_len: 0, ..p },
        ),
        (
            vec![0, 1],
            vec![1e6, 0.0, 0.0, 0.0],
            2,
            DiskHnswParams {
                encoding: VectorEncoding::F16,
                ..p
            },
        ),
    ];
    for (i, (docs, vectors, dim, params)) in cases.iter().enumerate() {
        let mut w = DiskHnswWriter::new(&ID, "");
        let r = w.add_field(
            &FieldVectors {
                number: 0,
                similarity: sim,
                dimension: *dim,
                docs,
                vectors,
            },
            &bbq,
            params,
        );
        assert!(
            matches!(r, Err(Error::IllegalArgument(_))),
            "case {i}: {r:?}"
        );
    }
    // A graph of the wrong size.
    let v = dataset(10, 2, 2, sim, 1);
    let g = build_graph(&v, 2, sim, &GraphParams::default()).unwrap();
    let d = docs(5);
    let mut w = DiskHnswWriter::new(&ID, "");
    let r = w.add_field_with_graph(
        &FieldVectors {
            number: 0,
            similarity: sim,
            dimension: 2,
            docs: &d,
            vectors: &v[..10],
        },
        &g,
        &bbq,
        &p,
    );
    assert!(matches!(r, Err(Error::IllegalArgument(_))), "{r:?}");
}

#[test]
fn corrupt_files_are_refused() {
    let dim = 8;
    let sim = Similarity::Euclidean;
    let vectors = dataset(200, dim, 4, sim, 61);
    let files = write(
        sim,
        dim,
        &vectors,
        &OsqFactory::bbq(),
        &DiskHnswParams::default(),
    );
    let reg = QuantizerRegistry::with_builtins();
    let try_open = |meta: &[u8], codes: Vec<u8>, nodes: Vec<u8>, suffix: &str| {
        DiskHnswReader::open(
            meta,
            Arc::new(codes),
            Arc::new(nodes),
            &ID,
            suffix,
            &reg,
            OpenOptions::default(),
        )
    };
    assert!(try_open(&files.meta, files.codes.clone(), files.nodes.clone(), "x").is_err());
    assert!(try_open(&files.meta, files.codes.clone(), vec![0; 8], "").is_err());
    assert!(try_open(
        &files.meta,
        files.codes[..40].to_vec(),
        files.nodes.clone(),
        ""
    )
    .is_err());
    // Swapped files: wrong codecs.
    assert!(try_open(&files.meta, files.nodes.clone(), files.codes.clone(), "").is_err());
    // Every single-byte flip of the meta file fails its checksum.
    for i in 0..files.meta.len() {
        let mut m = files.meta.clone();
        m[i] ^= 0x40;
        assert!(
            try_open(&m, files.codes.clone(), files.nodes.clone(), "").is_err(),
            "byte {i}"
        );
    }
    // With the checksum rewritten, every flip and a few values per byte
    // either fail to open or open to a reader whose searches return
    // results or errors; none panics.
    let q = &vectors[..dim];
    let mut refused = 0;
    let payload = files.meta.len() - 16;
    for i in 0..payload {
        for v in [files.meta[i] ^ 0x01, files.meta[i] ^ 0x80, 0x00, 0xff] {
            let mut m = files.meta[..payload].to_vec();
            m[i] = v;
            lucene_store::codec_util::write_footer(&mut m);
            match try_open(&m, files.codes.clone(), files.nodes.clone(), "") {
                Err(_) => refused += 1,
                Ok(r) => {
                    for f in r.fields() {
                        let q = vec![0.5; f.dimension()];
                        let _ = r.search(f.number(), &q, &SearchParams::default(), None);
                    }
                    let _ = r.search(7, q, &SearchParams::default(), None);
                }
            }
        }
    }
    assert!(
        refused > payload,
        "{refused} of {} variants refused",
        payload * 4
    );
    // A truncated nodes file: the region no longer fits.
    let mut short = files.nodes[..files.nodes.len() - 4096 - 16].to_vec();
    lucene_store::codec_util::write_footer(&mut short);
    assert!(try_open(&files.meta, files.codes.clone(), short, "").is_err());
    // A flipped nodes byte passes open (only the footer shape is checked)
    // but fails the integrity check.
    let mut flipped = files.nodes.clone();
    let at = flipped.len() / 2;
    flipped[at] ^= 1;
    let r = try_open(&files.meta, files.codes.clone(), flipped, "").unwrap();
    assert!(r.check_integrity().is_err());
}

#[test]
fn corrupt_records_fail_the_search() {
    let dim = 8;
    let sim = Similarity::Euclidean;
    let vectors = dataset(300, dim, 4, sim, 71);
    let files = write(
        sim,
        dim,
        &vectors,
        &OsqFactory::bbq(),
        &DiskHnswParams::default(),
    );
    let reader = open(&files);
    let f = &reader.fields()[0];
    let (layout, start) = (*f.layout(), f.nodes_offset());
    let q = &vectors[..dim];
    for (what, patch) in [("degree", 4usize), ("neighbour", layout.neighbors_start())] {
        let mut nodes = files.nodes.clone();
        for node in 0..300 {
            let at = start + layout.record_offset(node).unwrap() + patch;
            nodes[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
            if what == "degree" {
                nodes[at + 2..at + 4].copy_from_slice(&[0, 0]);
            }
        }
        // Rewrite the footer so open's shape check passes.
        nodes.truncate(nodes.len() - 16);
        lucene_store::codec_util::write_footer(&mut nodes);
        let r = DiskHnswReader::open(
            &files.meta,
            Arc::new(files.codes.clone()),
            Arc::new(nodes),
            &ID,
            "",
            &QuantizerRegistry::with_builtins(),
            OpenOptions::default(),
        )
        .unwrap();
        let out = r.search(7, q, &SearchParams::default(), None);
        assert!(matches!(out, Err(Error::Corrupted(_))), "{what}: {out:?}");
    }
}

#[test]
fn a_single_vector_and_a_flat_graph() {
    let sim = Similarity::Euclidean;
    for inline in [false, true] {
        let params = DiskHnswParams {
            inline_codes: inline,
            ..DiskHnswParams::default()
        };
        let files = write(sim, 3, &[1.0, 2.0, 3.0], &OsqFactory::bbq(), &params);
        let reader = open_with(
            &files,
            OpenOptions {
                resident_codes: !inline,
            },
        )
        .unwrap();
        let out = reader
            .search(7, &[1.0, 2.0, 3.0], &SearchParams::default(), None)
            .unwrap();
        assert_eq!(out.hits.len(), 1);
        assert_eq!(out.hits[0].doc, 1);
        assert_eq!(reader.fields()[0].num_levels(), 1);
    }
}
