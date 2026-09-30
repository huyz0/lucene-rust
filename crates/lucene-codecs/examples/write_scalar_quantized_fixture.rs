//! Writes a `Lucene104HnswScalarQuantizedVectorsFormat` segment -- raw
//! `.vec`/`.vemf`, quantized `.veq`/`.vemq` and the `.vem`/`.vex` graph, all
//! produced by this port -- plus a `.fnm` and a manifest, into the directory
//! given as the first CLI argument. `fixtures/src/VerifyScalarQuantized.java`
//! opens it with real Lucene's own format and requires, for every query, the
//! top-10 this port's reader computed over the same bytes, scores bit for bit.
//!
//! One format instance (one file suffix) per `ScalarEncoding`, each with a
//! dense and a sparse field, so every encoding's writer and every similarity
//! is read back by Lucene: the codes, the corrective terms, the centroid and
//! its squared norm, and the sparse ord-to-doc map all feed the scores Lucene
//! computes. A 64-dimension EUCLIDEAN field checks a dimension past the
//! float kernels' short-vector cut-over (Euclidean scores use no centroid
//! norm, so the comparison stays exact there too).
//!
//! Run: `cargo run -p lucene-codecs --example write_scalar_quantized_fixture -- <dir>`
#![allow(clippy::arithmetic_side_effects)]

use std::fmt::Write as _;

use lucene_codecs::field_infos::{
    self, DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::hnsw::{self, HnswGraphBuilder, OnHeapHnswGraph};
use lucene_codecs::hnsw_vectors::{self, HnswVectorsField, HnswVectorsReader};
use lucene_codecs::scalar_quantized_vectors::{
    QuantizedQueryScorer, QuantizedVectorsField, ScalarQuantizedVectorsReader,
    ScalarQuantizedVectorsWriter, HNSW_NAME,
};
use lucene_codecs::vectors::{self, FieldVectorData, FlatVectorsField, FlatVectorsReader};
use lucene_util::quantization::ScalarEncoding;
use lucene_util::vector_util;

const SEGMENT_ID: [u8; 16] = *b"rustquantized001";
const SEGMENT: &str = "_0";
const MAX_DOC: i32 = 1500;
const M: i32 = 16;
const BEAM_WIDTH: i32 = 100;
const K: usize = 10;
const NUM_QUERIES: usize = 6;

fn lcg(state: i64) -> i64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

fn float_vector(dim: usize, seed: i64) -> Vec<f32> {
    let mut s = seed;
    (0..dim)
        .map(|_| {
            s = lcg(s);
            (((s as u64) >> 40) as f32 / (1u32 << 24) as f32) - 0.5
        })
        .collect()
}

struct Spec {
    name: String,
    number: i32,
    dim: usize,
    sim: VectorSimilarityFunction,
    encoding: ScalarEncoding,
    every: i32,
    seed: i64,
}

impl Spec {
    fn docs(&self) -> Vec<i32> {
        (0..MAX_DOC).filter(|d| d % self.every == 0).collect()
    }
    fn vector(&self, doc: i32) -> Vec<f32> {
        let mut v = float_vector(self.dim, self.seed + doc as i64);
        if self.sim == VectorSimilarityFunction::DotProduct {
            vector_util::l2normalize(&mut v, true).unwrap();
        }
        v
    }
    fn suffix(&self) -> String {
        format!("{HNSW_NAME}_{}", self.encoding.wire_number())
    }
}

fn specs() -> Vec<Spec> {
    let encodings = [
        ScalarEncoding::UnsignedByte,
        ScalarEncoding::PackedNibble,
        ScalarEncoding::SevenBit,
        ScalarEncoding::SingleBitQueryNibble,
        ScalarEncoding::DibitQueryNibble,
    ];
    let dense_sims = [
        VectorSimilarityFunction::Euclidean,
        VectorSimilarityFunction::DotProduct,
        VectorSimilarityFunction::MaximumInnerProduct,
        VectorSimilarityFunction::Cosine,
        VectorSimilarityFunction::Euclidean,
    ];
    let sparse_sims = [
        VectorSimilarityFunction::Cosine,
        VectorSimilarityFunction::Euclidean,
        VectorSimilarityFunction::DotProduct,
        VectorSimilarityFunction::MaximumInnerProduct,
        VectorSimilarityFunction::Cosine,
    ];
    let mut out = Vec::new();
    let mut number = 0;
    for (i, &encoding) in encodings.iter().enumerate() {
        let base = format!("{encoding:?}").to_lowercase();
        out.push(Spec {
            name: format!("{base}_dense"),
            number,
            // The last encoding's dense field is 64-dimensional (EUCLIDEAN).
            dim: if i == 4 { 64 } else { 16 - i },
            sim: dense_sims[i],
            encoding,
            every: 1,
            seed: 100_000 * (i as i64 + 1),
        });
        number += 1;
        out.push(Spec {
            name: format!("{base}_sparse"),
            number,
            dim: 8 + i,
            sim: sparse_sims[i],
            encoding,
            every: 4,
            seed: 7_000_000 * (i as i64 + 1),
        });
        number += 1;
    }
    out
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_scalar_quantized_fixture <dir>");
    std::fs::create_dir_all(&out_dir).unwrap();
    let specs = specs();
    let mut manifest = String::new();
    writeln!(manifest, "segment_name={SEGMENT}").unwrap();
    writeln!(manifest, "id_hex={}", hex(&SEGMENT_ID)).unwrap();
    writeln!(manifest, "max_doc={MAX_DOC}").unwrap();
    writeln!(manifest, "k={K}").unwrap();
    writeln!(manifest, "field_count={}", specs.len()).unwrap();

    let mut suffixes: Vec<String> = specs.iter().map(Spec::suffix).collect();
    suffixes.dedup();
    let mut total_queries = 0;
    for suffix in &suffixes {
        let fields: Vec<&Spec> = specs.iter().filter(|s| &s.suffix() == suffix).collect();
        let encoding = fields[0].encoding;
        // Raw vectors.
        let flat_fields: Vec<FlatVectorsField> = fields
            .iter()
            .map(|s| FlatVectorsField {
                field_number: s.number,
                similarity: s.sim,
                dimension: s.dim as i32,
                docs: s.docs(),
                values: FieldVectorData::Float32(
                    s.docs().iter().flat_map(|&d| s.vector(d)).collect(),
                ),
            })
            .collect();
        let (vec_bytes, vemf_bytes) =
            vectors::write_flat_vectors(&flat_fields, MAX_DOC, &SEGMENT_ID, suffix).unwrap();
        // Quantized vectors.
        let mut writer = ScalarQuantizedVectorsWriter::new(encoding, MAX_DOC, &SEGMENT_ID, suffix);
        for (s, f) in fields.iter().zip(&flat_fields) {
            let FieldVectorData::Float32(v) = &f.values else {
                unreachable!()
            };
            writer
                .write_field(&QuantizedVectorsField {
                    field_number: s.number,
                    similarity: s.sim,
                    dimension: s.dim as i32,
                    docs: &f.docs,
                    vectors: v,
                })
                .unwrap();
        }
        let (veq_bytes, vemq_bytes) = writer.finish();
        // The graph, as a flush builds it: over the raw float vectors.
        let flat = FlatVectorsReader::open(&vemf_bytes, &vec_bytes, &SEGMENT_ID, suffix).unwrap();
        let graphs: Vec<Option<OnHeapHnswGraph>> = fields
            .iter()
            .map(|s| {
                let values = flat.float_vector_values(s.number).unwrap();
                let n = values.size();
                hnsw::should_create_graph(hnsw::HNSW_GRAPH_THRESHOLD, n).then(|| {
                    HnswGraphBuilder::new(
                        values.ord_scorer(),
                        M,
                        BEAM_WIDTH,
                        hnsw::DEFAULT_RAND_SEED,
                    )
                    .unwrap()
                    .build(n)
                    .unwrap()
                })
            })
            .collect();
        let hnsw_fields: Vec<HnswVectorsField> = fields
            .iter()
            .zip(&graphs)
            .map(|(s, g)| HnswVectorsField {
                field_number: s.number,
                encoding: VectorEncoding::Float32,
                similarity: s.sim,
                dimension: s.dim as i32,
                count: s.docs().len() as i32,
                graph: g.as_ref(),
                m: M,
            })
            .collect();
        let (vex_bytes, vem_bytes) =
            hnsw_vectors::write_hnsw_vectors(&hnsw_fields, &SEGMENT_ID, suffix).unwrap();
        for (ext, bytes) in [
            ("vec", &vec_bytes),
            ("vemf", &vemf_bytes),
            ("veq", &veq_bytes),
            ("vemq", &vemq_bytes),
            ("vex", &vex_bytes),
            ("vem", &vem_bytes),
        ] {
            std::fs::write(format!("{out_dir}/{SEGMENT}_{suffix}.{ext}"), bytes).unwrap();
        }
        // This port's own answers, over the bytes just written.
        let reader =
            ScalarQuantizedVectorsReader::open(&vemq_bytes, &veq_bytes, &SEGMENT_ID, suffix)
                .unwrap();
        let hnsw_reader =
            HnswVectorsReader::open(&vem_bytes, &vex_bytes, &SEGMENT_ID, suffix).unwrap();
        for s in &fields {
            let key = format!("f{}", s.number);
            writeln!(manifest, "{key}.name={}", s.name).unwrap();
            writeln!(manifest, "{key}.suffix={suffix}").unwrap();
            writeln!(manifest, "{key}.dim={}", s.dim).unwrap();
            writeln!(manifest, "{key}.count={}", s.docs().len()).unwrap();
            writeln!(manifest, "{key}.encoding={}", encoding.wire_number()).unwrap();
            let values = reader.quantized_vector_values(s.number).unwrap();
            let graph = hnsw_reader.graph(s.number).unwrap();
            writeln!(manifest, "{key}.has_graph={}", graph.is_some()).unwrap();
            for q in 0..NUM_QUERIES {
                let mut query = float_vector(s.dim, 900_000 + 131 * q as i64 + s.seed);
                if s.sim == VectorSimilarityFunction::DotProduct {
                    vector_util::l2normalize(&mut query, true).unwrap();
                }
                let mut scorer = QuantizedQueryScorer::new(&values, s.sim, &query).unwrap();
                let (hits, _) = hnsw_vectors::search(
                    &mut scorer,
                    graph.as_ref(),
                    K,
                    u64::MAX,
                    hnsw_vectors::SearchOptions::default(),
                )
                .unwrap();
                let query_bits: Vec<String> = query
                    .iter()
                    .map(|x| (x.to_bits() as i32).to_string())
                    .collect();
                let hit_strs: Vec<String> = hits
                    .iter()
                    .map(|&(ord, score)| {
                        format!(
                            "{}:{}",
                            values.ord_to_doc(ord).unwrap(),
                            score.to_bits() as i32
                        )
                    })
                    .collect();
                writeln!(manifest, "{key}.q{q}.vec={}", query_bits.join(",")).unwrap();
                writeln!(manifest, "{key}.q{q}.hits={}", hit_strs.join(",")).unwrap();
                total_queries += 1;
            }
            writeln!(manifest, "{key}.queries={NUM_QUERIES}").unwrap();
        }
    }
    assert_eq!(total_queries, specs.len() * NUM_QUERIES);

    // The `.fnm`, with the per-field format attributes Lucene's
    // PerFieldKnnVectorsFormat would record.
    let infos: Vec<FieldInfo> = specs
        .iter()
        .map(|s| FieldInfo {
            name: s.name.clone(),
            number: s.number,
            store_term_vectors: false,
            omit_norms: true,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![
                ("PerFieldKnnVectorsFormat.format".into(), HNSW_NAME.into()),
                (
                    "PerFieldKnnVectorsFormat.suffix".into(),
                    s.encoding.wire_number().to_string(),
                ),
            ],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: s.dim as i32,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: s.sim,
        })
        .collect();
    std::fs::write(
        format!("{out_dir}/{SEGMENT}.fnm"),
        field_infos::write(&infos, &SEGMENT_ID, ""),
    )
    .unwrap();
    std::fs::write(format!("{out_dir}/manifest.properties"), manifest).unwrap();
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
