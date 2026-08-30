//! Writes a `Lucene99FlatVectorsFormat` `.vec`/`.vemf` pair and a
//! `Lucene99HnswVectorsFormat` `.vem`/`.vex` pair -- both produced by this
//! port -- plus a manifest, into the directory given as the first CLI
//! argument.
//!
//! This is the reverse of this repo's usual differential-testing direction
//! (Java writes, Rust reads): here Rust writes the vector files and
//! `fixtures/src/VerifyVectors.java` opens them with real Lucene's own
//! `Lucene99HnswVectorsFormat`, walks the graph, and runs a real
//! `TopKnnCollector` search over it. Same division of labour as
//! `write_points_fixture.rs`: a hand-built `SegmentInfo`/`FieldInfos` on the
//! Java side keeps this scoped to the vector formats themselves.
//!
//! Five fields, chosen to cover every branch a `Lucene99FlatVectorsReader`
//! and `Lucene99HnswVectorsReader` can take:
//!
//! - `dense_f32`: FLOAT32, EUCLIDEAN, present on **every** document, so the
//!   ord-to-doc configuration is the `docsWithFieldOffset == -1` dense case
//!   with no `IndexedDISI` and no `DirectMonotonicWriter` at all.
//! - `sparse_f32`: FLOAT32, COSINE, present on every third document -- the
//!   sparse case, which writes both structures into `.vec` after the vectors.
//! - `byte_dot`: BYTE, DOT_PRODUCT, present on the first 1500 documents.
//!   A byte field aligns to 4 bytes rather than 64, and Java's `compare` for
//!   byte DOT_PRODUCT is `dotProductScore`, not the float branch's
//!   `normalizeToUnitInterval`.
//! - `tiny_mip`: FLOAT32, MAXIMUM_INNER_PRODUCT, five documents -- below
//!   Lucene's `HNSW_GRAPH_THRESHOLD`, so **no graph** is written for it
//!   (`numLevels == 0`, a zero-length `.vex` region). Real Lucene must accept
//!   that and fall back to an exhaustive scan.
//! - `merged_f32`: FLOAT32, EUCLIDEAN, 1900 documents -- and unlike the other
//!   four it is not *flushed*, it is **merged**. Two sub-segments are written
//!   and reopened, then folded into this segment by
//!   `vectors::FlatVectorsWriter::merge_one_flat_vector_field` and
//!   `hnsw_vectors::merge_one_field`, which is the incremental path
//!   (`Lucene99FlatVectorsWriter.mergeOneFlatVectorField` +
//!   `IncrementalHnswGraphMerger`) that reuses the larger source graph instead
//!   of rebuilding. Everything Java checks for the other fields it checks for
//!   this one, so a merge that produces a structurally plausible but wrongly
//!   built graph fails on recall the same way a bad flush would.
//!
//! Run: `cargo run -p lucene-codecs --example write_vectors_fixture -- <dir>`
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::io::Write;

use lucene_codecs::field_infos::{
    self, DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::hnsw::{self, HnswGraphBuilder, HnswGraphView, OnHeapHnswGraph};
use lucene_codecs::hnsw_vectors::{self, HnswVectorsField, HnswVectorsReader};
use lucene_codecs::vectors::{
    self, FieldVectorData, FlatVectorMergeSource, FlatVectorsField, FlatVectorsReader,
    FlatVectorsWriter, FloatVectorValues, MergeSourceValues, MergedFlatVectorField,
};
use lucene_store::{DataOutput, Directory, FsDirectory};

const SEGMENT_ID: [u8; 16] = *b"rustwrittenvec01";
const SEGMENT: &str = "_0";
const MAX_DOC: i32 = 3000;
const M: i32 = 16;
const BEAM_WIDTH: i32 = 100;
const K: usize = 10;
const NUM_QUERIES: usize = 12;

/// The same 64-bit LCG `GenVectors.java` uses, so both sides of the sweep
/// look at data of the same shape.
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

fn byte_vector(dim: usize, seed: i64) -> Vec<u8> {
    let mut s = seed;
    (0..dim)
        .map(|_| {
            s = lcg(s);
            (((s as u64) >> 40) & 0xFF) as u8
        })
        .collect()
}

struct Spec {
    name: &'static str,
    number: i32,
    dim: usize,
    similarity: VectorSimilarityFunction,
    /// `None` for FLOAT32, `Some(())` marks BYTE.
    byte_encoded: bool,
    docs: Vec<i32>,
    seed_base: i64,
    /// `Some((a, b))` marks the merged field: its vectors reach this segment
    /// through the merge entry points, from two sub-segments of `a` and `b`
    /// documents, instead of through the flush entry point.
    merge_split: Option<(i32, i32)>,
}

impl Spec {
    fn encoding(&self) -> VectorEncoding {
        if self.byte_encoded {
            VectorEncoding::Byte
        } else {
            VectorEncoding::Float32
        }
    }

    fn field(&self) -> FlatVectorsField {
        let values = if self.byte_encoded {
            let mut flat = Vec::with_capacity(self.docs.len() * self.dim);
            for d in &self.docs {
                flat.extend_from_slice(&byte_vector(self.dim, self.seed_base + *d as i64));
            }
            FieldVectorData::Byte(flat)
        } else {
            let mut flat = Vec::with_capacity(self.docs.len() * self.dim);
            for d in &self.docs {
                flat.extend_from_slice(&float_vector(self.dim, self.seed_base + *d as i64));
            }
            FieldVectorData::Float32(flat)
        };
        FlatVectorsField {
            field_number: self.number,
            similarity: self.similarity,
            dimension: self.dim as i32,
            docs: self.docs.clone(),
            values,
        }
    }
}

fn specs() -> Vec<Spec> {
    vec![
        Spec {
            name: "dense_f32",
            number: 0,
            dim: 16,
            similarity: VectorSimilarityFunction::Euclidean,
            byte_encoded: false,
            docs: (0..MAX_DOC).collect(),
            seed_base: 1,
            merge_split: None,
        },
        Spec {
            name: "sparse_f32",
            number: 1,
            dim: 8,
            similarity: VectorSimilarityFunction::Cosine,
            byte_encoded: false,
            docs: (0..MAX_DOC).filter(|d| d % 3 == 0).collect(),
            seed_base: 1_000_003,
            merge_split: None,
        },
        Spec {
            name: "byte_dot",
            number: 2,
            dim: 8,
            similarity: VectorSimilarityFunction::DotProduct,
            byte_encoded: true,
            docs: (0..1500).collect(),
            seed_base: 7_000_019,
            merge_split: None,
        },
        Spec {
            name: "tiny_mip",
            number: 3,
            dim: 4,
            similarity: VectorSimilarityFunction::MaximumInnerProduct,
            byte_encoded: false,
            docs: (0..5).collect(),
            seed_base: 9_000_037,
            merge_split: None,
        },
        Spec {
            name: "merged_f32",
            number: 4,
            dim: 8,
            similarity: VectorSimilarityFunction::Euclidean,
            byte_encoded: false,
            docs: (0..1900).collect(),
            seed_base: 11_000_081,
            merge_split: Some((1000, 900)),
        },
    ]
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_vectors_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).unwrap();
    let dir = FsDirectory::open(&out_dir);

    let specs = specs();
    // The two sub-segments the merged field is folded in from. They are
    // written and reopened exactly like any other segment this port produces,
    // so the merge reads real `.vec`/`.vemf` bytes rather than an in-memory
    // shortcut.
    let merge_spec = specs
        .iter()
        .find(|s| s.merge_split.is_some())
        .expect("one merged field");
    let (source_a, source_b) = write_merge_sources(merge_spec);
    let a = FlatVectorsReader::open(&source_a.1, &source_a.0, &SEGMENT_ID, "").expect("source a");
    let b = FlatVectorsReader::open(&source_b.1, &source_b.0, &SEGMENT_ID, "").expect("source b");
    let (split_a, split_b) = merge_spec.merge_split.unwrap();
    let a_map: Vec<i32> = (0..split_a).collect();
    let b_map: Vec<i32> = (0..split_b).map(|d| d + split_a).collect();

    let mut writer = FlatVectorsWriter::new(MAX_DOC, &SEGMENT_ID, "");
    for spec in &specs {
        if spec.merge_split.is_some() {
            let sources = vec![
                FlatVectorMergeSource {
                    values: MergeSourceValues::Float32(a.float_vector_values(spec.number).unwrap()),
                    doc_map: &a_map,
                },
                FlatVectorMergeSource {
                    values: MergeSourceValues::Float32(b.float_vector_values(spec.number).unwrap()),
                    doc_map: &b_map,
                },
            ];
            writer
                .merge_one_flat_vector_field(&MergedFlatVectorField {
                    field_number: spec.number,
                    encoding: spec.encoding(),
                    similarity: spec.similarity,
                    dimension: spec.dim as i32,
                    sources: &sources,
                })
                .expect("merge flat field");
        } else {
            writer.write_field(&spec.field()).expect("flat write");
        }
    }
    let (vec_bytes, vemf_bytes) = writer.finish();

    // Build a graph per field, exactly where Lucene would: only when
    // `shouldCreateGraph` says the field is big enough to be worth one.
    let flat = FlatVectorsReader::open(&vemf_bytes, &vec_bytes, &SEGMENT_ID, "").expect("reopen");
    // The merged field's vectors must be exactly what a flush of the same
    // documents would have produced -- checked here rather than only in Java,
    // so a merge that loses a vector is caught before Lucene sees it.
    {
        let merged = flat.float_vector_values(merge_spec.number).unwrap();
        let expected = merge_spec.field();
        let FieldVectorData::Float32(components) = &expected.values else {
            unreachable!("the merged field is FLOAT32")
        };
        assert_eq!(merged.size() as usize, merge_spec.docs.len());
        for ord in 0..merged.size() {
            assert_eq!(
                merged.ord_to_doc(ord).unwrap(),
                merge_spec.docs[ord as usize]
            );
            let want =
                &components[ord as usize * merge_spec.dim..(ord as usize + 1) * merge_spec.dim];
            assert_eq!(merged.vector(ord).unwrap(), want, "merged ordinal {ord}");
        }
    }

    let mut graphs: Vec<Option<OnHeapHnswGraph>> = Vec::new();
    for spec in &specs {
        let count = spec.docs.len() as i32;
        if !hnsw::should_create_graph(hnsw::HNSW_GRAPH_THRESHOLD, count) {
            graphs.push(None);
            continue;
        }
        if spec.merge_split.is_some() {
            graphs.push(Some(merge_graph(spec, &flat, &a, &b, &a_map, &b_map)));
            continue;
        }
        let graph = if spec.byte_encoded {
            let values = flat.byte_vector_values(spec.number).unwrap();
            HnswGraphBuilder::new(values.ord_scorer(), M, BEAM_WIDTH, hnsw::DEFAULT_RAND_SEED)
                .unwrap()
                .build(count)
                .unwrap()
        } else {
            let values = flat.float_vector_values(spec.number).unwrap();
            HnswGraphBuilder::new(values.ord_scorer(), M, BEAM_WIDTH, hnsw::DEFAULT_RAND_SEED)
                .unwrap()
                .build(count)
                .unwrap()
        };
        graphs.push(Some(graph));
    }

    let hnsw_fields: Vec<HnswVectorsField> = specs
        .iter()
        .zip(&graphs)
        .map(|(spec, graph)| HnswVectorsField {
            field_number: spec.number,
            encoding: spec.encoding(),
            similarity: spec.similarity,
            dimension: spec.dim as i32,
            count: spec.docs.len() as i32,
            graph: graph.as_ref(),
            m: M,
        })
        .collect();
    let (vex_bytes, vem_bytes) =
        hnsw_vectors::write_hnsw_vectors(&hnsw_fields, &SEGMENT_ID, "").expect("graph write");

    // A real `.fnm` carrying the vector metadata, so the Java verifier can read
    // its `FieldInfos` back through Lucene's own `Lucene94FieldInfosFormat`
    // instead of hand-building one. That is what puts
    // `Lucene99FlatVectorsReader.FieldEntry`'s cross-checks -- the `.vemf`'s
    // similarity and dimension must equal the `FieldInfo`'s -- in front of
    // *our* bytes on both sides. A hand-built `FieldInfos` cannot see a
    // disagreement between the two files, which is exactly the class of defect
    // c4 found in a merged `.fnm` while thirteen hand-built verifiers passed.
    let fnm_bytes = field_infos::write(&field_infos_of(&specs), &SEGMENT_ID, "");

    for (name, bytes) in [
        (format!("{SEGMENT}.fnm"), &fnm_bytes),
        (format!("{SEGMENT}.vec"), &vec_bytes),
        (format!("{SEGMENT}.vemf"), &vemf_bytes),
        (format!("{SEGMENT}.vex"), &vex_bytes),
        (format!("{SEGMENT}.vem"), &vem_bytes),
    ] {
        let mut out = dir.create_output(&name).unwrap();
        out.write_bytes(bytes);
        out.close().unwrap();
    }
    dir.sync(&[
        format!("{SEGMENT}.fnm"),
        format!("{SEGMENT}.vec"),
        format!("{SEGMENT}.vemf"),
        format!("{SEGMENT}.vex"),
        format!("{SEGMENT}.vem"),
    ])
    .unwrap();

    write_manifest(&out_dir, &specs, &flat, &vem_bytes, &vex_bytes);
}

/// One segment's `(vec, vemf)` bytes.
type SegmentBytes = (Vec<u8>, Vec<u8>);

/// Writes the merged field's two source sub-segments: documents
/// `0..split_a` and, in the second segment, `0..split_b` whose vectors carry
/// the seeds the merged documents `split_a..` will have. Returns each
/// segment's `(vec, vemf)`.
fn write_merge_sources(spec: &Spec) -> (SegmentBytes, SegmentBytes) {
    let (split_a, split_b) = spec.merge_split.expect("a merged spec");
    let build = |first_doc: i32, count: i32| -> (Vec<u8>, Vec<u8>) {
        let mut flat = Vec::with_capacity(count as usize * spec.dim);
        for d in 0..count {
            flat.extend_from_slice(&float_vector(
                spec.dim,
                spec.seed_base + (first_doc + d) as i64,
            ));
        }
        vectors::write_flat_vectors(
            &[FlatVectorsField {
                field_number: spec.number,
                similarity: spec.similarity,
                dimension: spec.dim as i32,
                docs: (0..count).collect(),
                values: FieldVectorData::Float32(flat),
            }],
            count,
            &SEGMENT_ID,
            "",
        )
        .expect("source segment")
    };
    (build(0, split_a), build(split_a, split_b))
}

/// `Lucene99HnswVectorsWriter.mergeOneField`'s graph half over the two source
/// segments: each gets its own graph (as it would have had on disk), and the
/// merged graph is folded out of them rather than rebuilt.
fn merge_graph(
    spec: &Spec,
    merged: &FlatVectorsReader<'_>,
    a: &FlatVectorsReader<'_>,
    b: &FlatVectorsReader<'_>,
    a_map: &[i32],
    b_map: &[i32],
) -> OnHeapHnswGraph {
    let source_graph = |reader: &FlatVectorsReader<'_>| {
        let values = reader.float_vector_values(spec.number).unwrap();
        HnswGraphBuilder::new(values.ord_scorer(), M, BEAM_WIDTH, hnsw::DEFAULT_RAND_SEED)
            .unwrap()
            .build(values.size())
            .unwrap()
    };
    let a_graph = source_graph(a);
    let b_graph = source_graph(b);
    let a_docs: Vec<i32> = (0..a_map.len() as i32).collect();
    let b_docs: Vec<i32> = (0..b_map.len() as i32).collect();
    let merged_values = merged.float_vector_values(spec.number).unwrap();
    let merged_ord_to_doc: Vec<i32> = (0..merged_values.size())
        .map(|o| merged_values.ord_to_doc(o).unwrap())
        .collect();
    let sources = [
        hnsw_vectors::GraphMergeSource {
            graph: Some(&a_graph),
            ord_to_doc: &a_docs,
            doc_map: a_map,
        },
        hnsw_vectors::GraphMergeSource {
            graph: Some(&b_graph),
            ord_to_doc: &b_docs,
            doc_map: b_map,
        },
    ];
    hnsw_vectors::merge_one_field(
        merged_values.ord_scorer(),
        M,
        BEAM_WIDTH,
        hnsw::DEFAULT_RAND_SEED,
        &merged_ord_to_doc,
        &sources,
    )
    .expect("merge the graph")
    .expect("1900 vectors is past the graph threshold")
}

/// The `.fnm` entries for the five vector fields. Everything unrelated to
/// vectors is off, so the file describes exactly what this fixture is about.
fn field_infos_of(specs: &[Spec]) -> Vec<FieldInfo> {
    specs
        .iter()
        .map(|spec| FieldInfo {
            name: spec.name.to_string(),
            number: spec.number,
            store_term_vectors: false,
            omit_norms: true,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: Vec::new(),
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: spec.dim as i32,
            vector_encoding: spec.encoding(),
            vector_similarity_function: spec.similarity,
        })
        .collect()
}

fn write_manifest(
    out_dir: &str,
    specs: &[Spec],
    flat: &FlatVectorsReader<'_>,
    vem_bytes: &[u8],
    vex_bytes: &[u8],
) {
    let mut m = std::fs::File::create(format!("{out_dir}/manifest.properties")).unwrap();
    writeln!(m, "segment_name={SEGMENT}").unwrap();
    writeln!(m, "id_hex={}", hex(&SEGMENT_ID)).unwrap();
    writeln!(m, "max_doc={MAX_DOC}").unwrap();
    writeln!(m, "k={K}").unwrap();
    writeln!(m, "field_count={}", specs.len()).unwrap();

    // The graph rows are computed from the *written* bytes, read back through
    // this port's own `.vex` decoder -- so a Java mismatch means one of the two
    // decoders is wrong, not that the in-memory graph was described loosely.
    let hnsw_reader = HnswVectorsReader::open(vem_bytes, vex_bytes, &SEGMENT_ID, "").unwrap();

    for (i, spec) in specs.iter().enumerate() {
        let key = format!("f{i}");
        writeln!(m, "{key}.name={}", spec.name).unwrap();
        writeln!(m, "{key}.number={}", spec.number).unwrap();
        writeln!(m, "{key}.dim={}", spec.dim).unwrap();
        writeln!(
            m,
            "{key}.encoding={}",
            if spec.byte_encoded { "BYTE" } else { "FLOAT32" }
        )
        .unwrap();
        writeln!(m, "{key}.similarity={}", similarity_name(spec.similarity)).unwrap();
        writeln!(m, "{key}.count={}", spec.docs.len()).unwrap();
        let docs: Vec<String> = spec.docs.iter().map(|d| d.to_string()).collect();
        writeln!(m, "{key}.docs={}", docs.join(",")).unwrap();

        // An order-sensitive hash over every ordinal's raw component bits:
        // exact, cheap, and independent of float summation order.
        let mut value_hash = 0i64;
        if spec.byte_encoded {
            let values = flat.byte_vector_values(spec.number).unwrap();
            for ord in 0..values.size() {
                for b in values.vector(ord).unwrap() {
                    value_hash = value_hash.wrapping_mul(31).wrapping_add(*b as i8 as i64);
                }
            }
        } else {
            let values = flat.float_vector_values(spec.number).unwrap();
            for ord in 0..values.size() {
                for c in values.vector(ord).unwrap() {
                    value_hash = value_hash
                        .wrapping_mul(31)
                        .wrapping_add(c.to_bits() as i32 as i64);
                }
            }
        }
        writeln!(m, "{key}.value_hash={value_hash}").unwrap();

        let entry = hnsw_reader.field(spec.number).unwrap();
        writeln!(m, "{key}.num_levels={}", entry.num_levels).unwrap();
        if let Some(graph) = hnsw_reader.graph(spec.number).unwrap() {
            writeln!(m, "{key}.entry_node={}", graph.entry_node()).unwrap();
            writeln!(m, "{key}.max_conn={}", graph.max_conn()).unwrap();
            let mut neighbors = Vec::new();
            for level in 0..graph.num_levels() {
                let nodes = graph.sorted_nodes_on_level(level).unwrap();
                let mut arc_total = 0i64;
                let mut hash = 0i64;
                for node in &nodes {
                    graph.neighbors_into(level, *node, &mut neighbors).unwrap();
                    hash = hash.wrapping_mul(31).wrapping_add(*node as i64);
                    for n in &neighbors {
                        hash = hash.wrapping_mul(31).wrapping_add(*n as i64);
                        arc_total += 1;
                    }
                }
                writeln!(m, "{key}.level{level}.node_count={}", nodes.len()).unwrap();
                writeln!(m, "{key}.level{level}.arc_total={arc_total}").unwrap();
                writeln!(m, "{key}.level{level}.arc_hash={hash}").unwrap();
            }
        }

        if spec.byte_encoded {
            byte_queries(&mut m, &key, spec, flat);
        } else {
            float_queries(&mut m, &key, spec, flat);
        }
    }
}

fn float_queries(m: &mut std::fs::File, key: &str, spec: &Spec, flat: &FlatVectorsReader<'_>) {
    let values: FloatVectorValues = flat.float_vector_values(spec.number).unwrap();
    writeln!(m, "q.{key}.count={NUM_QUERIES}").unwrap();
    for q in 0..NUM_QUERIES {
        let target = float_vector(spec.dim, 900_000_007 + q as i64 * 31);
        let bits: Vec<String> = target
            .iter()
            .map(|c| (c.to_bits() as i32).to_string())
            .collect();
        writeln!(m, "q.{key}.{q}.vec={}", bits.join(",")).unwrap();
        let exact = values.exhaustive_search(&target, K).unwrap();
        let docs: Vec<String> = exact.iter().map(|(d, _)| d.to_string()).collect();
        writeln!(m, "q.{key}.{q}.exact={}", docs.join(",")).unwrap();
    }
}

fn byte_queries(m: &mut std::fs::File, key: &str, spec: &Spec, flat: &FlatVectorsReader<'_>) {
    let values = flat.byte_vector_values(spec.number).unwrap();
    writeln!(m, "q.{key}.count={NUM_QUERIES}").unwrap();
    for q in 0..NUM_QUERIES {
        let target = byte_vector(spec.dim, 800_000_011 + q as i64 * 37);
        let signed: Vec<String> = target.iter().map(|b| (*b as i8).to_string()).collect();
        writeln!(m, "q.{key}.{q}.vec={}", signed.join(",")).unwrap();
        let exact = values.exhaustive_search(&target, K).unwrap();
        let docs: Vec<String> = exact.iter().map(|(d, _)| d.to_string()).collect();
        writeln!(m, "q.{key}.{q}.exact={}", docs.join(",")).unwrap();
    }
}

fn similarity_name(s: VectorSimilarityFunction) -> &'static str {
    match s {
        VectorSimilarityFunction::Euclidean => "EUCLIDEAN",
        VectorSimilarityFunction::DotProduct => "DOT_PRODUCT",
        VectorSimilarityFunction::Cosine => "COSINE",
        VectorSimilarityFunction::MaximumInnerProduct => "MAXIMUM_INNER_PRODUCT",
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
