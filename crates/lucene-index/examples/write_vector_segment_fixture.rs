//! Writes a whole index *with vector fields* the way an application would --
//! one `IndexWriter`, `add_document_with_vectors`, `commit` -- and leaves it
//! for `VerifyVectorSegment` to open with a real `DirectoryReader`, search
//! with `KnnFloatVectorQuery`/`KnnByteVectorQuery`, and run `CheckIndex` over.
//!
//! This is deliberately *not* `write_vectors_fixture` (which hands Lucene the
//! four vector files with a hand-built `FieldInfos` and checks the formats
//! themselves). The gap that one leaves is everything that binds those files
//! into a segment: `PerFieldKnnVectorsFormat`'s suffixed file names, the
//! `.fnm` attributes that record them, `SegmentInfo.files` listing them, and
//! `.fnm` `vector_dimension` agreeing with the `.vemf` for **every** field --
//! including the ones this flush wrote no vectors for. A segment that gets any
//! of those wrong opens with the vector fields silently absent, which is the
//! same failure shape c4 found for postings and doc values.
//!
//! Four vector fields, chosen to cover every branch of the writer:
//!
//! - `dense_f32`: FLOAT32, EUCLIDEAN, on every document -- the dense
//!   ord-to-doc case (`docsWithFieldOffset == -1`, no `IndexedDISI`).
//! - `sparse_f32`: FLOAT32, COSINE, every third document -- the sparse case,
//!   which writes an `IndexedDISI` + `DirectMonotonicWriter` pair.
//! - `byte_dot`: BYTE, DOT_PRODUCT, the first 1500 documents -- 4-byte
//!   alignment instead of 64, and Java's *different* byte score transform.
//! - `tiny_mip`: FLOAT32, MAXIMUM_INNER_PRODUCT, five documents -- below
//!   `HNSW_GRAPH_THRESHOLD`, so no graph is built and real Lucene must fall
//!   back to an exhaustive scan.
//!
//! A fifth vector field, `never_written`, is declared in the `FieldInfo` list
//! and given a vector by no document at all. Nothing may be written for it,
//! and its `.fnm` `vector_dimension` must come back 0 -- otherwise
//! `FieldInfo.hasVectorValues()` is true (which is what a merge and
//! `CheckIndex` key off) while `PerFieldKnnVectorsFormat` registers no reader
//! for it, and the field reads back as vector-capable and yields nothing.
//! Real Lucene raises no error for that combination, which is precisely why
//! the fixture has to pin it.
//!
//! Postings, norms and stored fields are indexed alongside, so the fixture
//! also proves vectors coexist with the formats that were already wired up.
//!
//! Usage: `write_vector_segment_fixture <output-dir>`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::io::Write;

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::{DocumentVector, IndexWriter};
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

/// Enough documents to cross a stored-fields chunk boundary (1024) and to make
/// every vector field except `tiny_mip` worth a graph.
const NUM_DOCS: i32 = 3_000;
const K: usize = 10;
const NUM_QUERIES: usize = 12;

/// The same 64-bit LCG `GenVectors.java` and `write_vectors_fixture.rs` use.
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

struct VectorSpec {
    name: &'static str,
    number: i32,
    dim: usize,
    encoding: VectorEncoding,
    similarity: VectorSimilarityFunction,
    seed_base: i64,
}

impl VectorSpec {
    /// Which documents carry this field.
    fn has_doc(&self, doc: i32) -> bool {
        match self.name {
            "dense_f32" => true,
            "sparse_f32" => doc % 3 == 0,
            "byte_dot" => doc < 1500,
            "tiny_mip" => doc < 5,
            _ => false,
        }
    }

    fn docs(&self) -> Vec<i32> {
        (0..NUM_DOCS).filter(|d| self.has_doc(*d)).collect()
    }
}

fn vector_specs() -> Vec<VectorSpec> {
    vec![
        VectorSpec {
            name: "dense_f32",
            number: 3,
            dim: 16,
            encoding: VectorEncoding::Float32,
            similarity: VectorSimilarityFunction::Euclidean,
            seed_base: 1,
        },
        VectorSpec {
            name: "sparse_f32",
            number: 4,
            dim: 8,
            encoding: VectorEncoding::Float32,
            similarity: VectorSimilarityFunction::Cosine,
            seed_base: 1_000_003,
        },
        VectorSpec {
            name: "byte_dot",
            number: 5,
            dim: 8,
            encoding: VectorEncoding::Byte,
            similarity: VectorSimilarityFunction::DotProduct,
            seed_base: 7_000_019,
        },
        VectorSpec {
            name: "tiny_mip",
            number: 6,
            dim: 4,
            encoding: VectorEncoding::Float32,
            similarity: VectorSimilarityFunction::MaximumInnerProduct,
            seed_base: 9_000_037,
        },
    ]
}

fn plain_field(name: &str, number: i32, indexed: bool) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: !indexed,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: if indexed {
            IndexOptions::DocsAndFreqs
        } else {
            IndexOptions::None
        },
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Float32,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

fn vector_field(spec: &VectorSpec) -> FieldInfo {
    FieldInfo {
        vector_dimension: spec.dim as i32,
        vector_encoding: spec.encoding,
        vector_similarity_function: spec.similarity,
        ..plain_field(spec.name, spec.number, false)
    }
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_vector_segment_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).expect("create output dir");

    let specs = vector_specs();
    let mut fields = vec![plain_field("id", 0, false), plain_field("body", 1, true)];
    // Declared, opted in, and never given a value by any document.
    fields.push(FieldInfo {
        vector_dimension: 6,
        ..plain_field("never_written", 2, false)
    });
    fields.extend(specs.iter().map(vector_field));

    let dir = FsDirectory::open(&out_dir);
    let mut writer = IndexWriter::open(
        &dir,
        fields,
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .expect("open writer");
    writer
        .set_postings_field(Some("body"))
        .expect("set postings field");
    writer
        .set_norms_field(Some("body"))
        .expect("set norms field");
    writer
        .set_vector_field(Some("never_written"))
        .expect("set vector field");
    for spec in &specs {
        writer
            .add_vector_field(spec.name)
            .expect("add vector field");
    }

    let vocab: Vec<String> = (0..500)
        .map(|i| format!("{}{i:03}", (b'a' + (i % 26) as u8) as char))
        .collect();
    for doc in 0..NUM_DOCS {
        let i = doc as usize;
        let body = format!(
            "shared {} {}",
            vocab[i % vocab.len()],
            vocab[(i / 7) % vocab.len()]
        );
        let vectors: Vec<DocumentVector> = specs
            .iter()
            .filter(|s| s.has_doc(doc))
            .map(|s| match s.encoding {
                VectorEncoding::Float32 => {
                    DocumentVector::float32(s.name, float_vector(s.dim, s.seed_base + doc as i64))
                }
                VectorEncoding::Byte => {
                    DocumentVector::byte(s.name, byte_vector(s.dim, s.seed_base + doc as i64))
                }
            })
            .collect();
        writer
            .add_document_with_vectors(
                Document {
                    fields: vec![
                        StoredField {
                            field_number: 0,
                            value: FieldValue::String(format!("doc{doc}")),
                        },
                        StoredField {
                            field_number: 1,
                            value: FieldValue::String(body),
                        },
                    ],
                },
                vectors,
            )
            .expect("add document");
    }
    writer.commit().expect("commit");

    write_manifest(&out_dir, &specs);
    println!("wrote a {NUM_DOCS}-document vector index to {out_dir}");
}

/// The expectations `VerifyVectorSegment` checks, computed here from the same
/// generator the documents were built from -- deliberately *not* read back out
/// of the files this port just wrote, so a writer that dropped or reordered a
/// vector cannot make its own mistake the expectation.
fn write_manifest(out_dir: &str, specs: &[VectorSpec]) {
    let mut m = std::fs::File::create(format!("{out_dir}/manifest.properties")).unwrap();
    writeln!(m, "num_docs={NUM_DOCS}").unwrap();
    writeln!(m, "k={K}").unwrap();
    writeln!(m, "field_count={}", specs.len()).unwrap();
    for (i, spec) in specs.iter().enumerate() {
        let key = format!("f{i}");
        let docs = spec.docs();
        writeln!(m, "{key}.name={}", spec.name).unwrap();
        writeln!(m, "{key}.dim={}", spec.dim).unwrap();
        writeln!(
            m,
            "{key}.encoding={}",
            match spec.encoding {
                VectorEncoding::Float32 => "FLOAT32",
                VectorEncoding::Byte => "BYTE",
            }
        )
        .unwrap();
        writeln!(m, "{key}.similarity={}", similarity_name(spec.similarity)).unwrap();
        writeln!(m, "{key}.count={}", docs.len()).unwrap();
        writeln!(
            m,
            "{key}.docs={}",
            docs.iter()
                .map(i32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )
        .unwrap();

        // An order-sensitive hash over every ordinal's raw component bits, in
        // ordinal order -- exact, and independent of float summation order.
        let mut value_hash = 0i64;
        for d in &docs {
            match spec.encoding {
                VectorEncoding::Float32 => {
                    for c in float_vector(spec.dim, spec.seed_base + *d as i64) {
                        value_hash = value_hash
                            .wrapping_mul(31)
                            .wrapping_add(c.to_bits() as i32 as i64);
                    }
                }
                VectorEncoding::Byte => {
                    for b in byte_vector(spec.dim, spec.seed_base + *d as i64) {
                        value_hash = value_hash.wrapping_mul(31).wrapping_add(b as i8 as i64);
                    }
                }
            }
        }
        writeln!(m, "{key}.value_hash={value_hash}").unwrap();

        writeln!(m, "q.{key}.count={NUM_QUERIES}").unwrap();
        for q in 0..NUM_QUERIES {
            match spec.encoding {
                VectorEncoding::Float32 => {
                    let target = float_vector(spec.dim, 900_000_007 + q as i64 * 31);
                    writeln!(
                        m,
                        "q.{key}.{q}.vec={}",
                        target
                            .iter()
                            .map(|c| (c.to_bits() as i32).to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                    .unwrap();
                }
                VectorEncoding::Byte => {
                    let target = byte_vector(spec.dim, 800_000_011 + q as i64 * 37);
                    writeln!(
                        m,
                        "q.{key}.{q}.vec={}",
                        target
                            .iter()
                            .map(|b| (*b as i8).to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                    .unwrap();
                }
            }
        }
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
