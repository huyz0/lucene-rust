//! Writes two `HnswBitVectorsFormat` groups (lucene-codecs' `bitvectors`
//! package), plus their expected contents, to the directory given as the
//! first CLI argument.
//!
//! Reverse-direction fixture (Rust writes, Java reads): the counterpart
//! `fixtures/src/VerifyCodecsBitVectors.java` opens each group through real
//! Lucene's own `HnswBitVectorsFormat.fieldsReader`, constructing a matching
//! `SegmentInfo`/`FieldInfos` in Java (no `.si`/`.fnm` from Rust), the same
//! pattern as `write_bloom_fixture.rs`.
//!
//! `HnswBitVectorsFormat_0` (default 16/100) holds `b8` (8 bytes, every
//! other document, EUCLIDEAN) and `tiny` (4 bytes, a handful of documents:
//! no graph); `HnswBitVectorsFormat_1` (6/30) holds `b16` (16 bytes, every
//! document, DOT_PRODUCT). `fields.tsv` lists `name\tnumber\tdimension\t
//! similarity\tsuffix`; `vectors.<suffix>.tsv` lists `number\tord\tdoc\thex`;
//! `search.<suffix>.tsv` lists `number\tk\tquery-hex` then `doc:scoreBits`
//! per hit of this port's own search.
//!
//! Run: `cargo run -p lucene-codecs --example write_bitvectors_fixture -- <dir>`
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::fmt::Write as _;
use std::path::Path;

use lucene_codecs::bitvectors::{BitVectorsReader, HnswBitVectorsFormat};
use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_codecs::hnsw_vectors::SearchOptions;
use lucene_codecs::vectors::{FieldVectorData, FlatVectorsField};

const SEGMENT_ID: [u8; 16] = *b"rustbitvectorsid";
const MAX_DOC: i32 = 2_000;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

struct Spec {
    name: &'static str,
    number: i32,
    dimension: i32,
    similarity: VectorSimilarityFunction,
    every: i32,
}

fn field(rng: &mut Rng, spec: &Spec) -> FlatVectorsField {
    let docs: Vec<i32> = (0..MAX_DOC).filter(|d| d % spec.every == 0).collect();
    let values = rng.bytes(docs.len() * spec.dimension as usize);
    FlatVectorsField {
        field_number: spec.number,
        similarity: spec.similarity,
        dimension: spec.dimension,
        docs,
        values: FieldVectorData::Byte(values),
    }
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_bitvectors_fixture <dir>");
    let out = Path::new(&out);
    std::fs::create_dir_all(out).unwrap();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let groups = [
        (
            "HnswBitVectorsFormat_0",
            HnswBitVectorsFormat::default(),
            vec![
                Spec {
                    name: "b8",
                    number: 0,
                    dimension: 8,
                    similarity: VectorSimilarityFunction::Euclidean,
                    every: 2,
                },
                Spec {
                    name: "tiny",
                    number: 1,
                    dimension: 4,
                    similarity: VectorSimilarityFunction::MaximumInnerProduct,
                    every: 401,
                },
            ],
        ),
        (
            "HnswBitVectorsFormat_1",
            HnswBitVectorsFormat::new(6, 30).unwrap(),
            vec![Spec {
                name: "b16",
                number: 2,
                dimension: 16,
                similarity: VectorSimilarityFunction::DotProduct,
                every: 1,
            }],
        ),
    ];
    let mut fields_tsv = String::new();
    for (suffix, format, specs) in &groups {
        let fields: Vec<FlatVectorsField> = specs.iter().map(|s| field(&mut rng, s)).collect();
        let files = format.write(&fields, MAX_DOC, &SEGMENT_ID, suffix).unwrap();
        for (ext, bytes) in [
            ("vec", &files.vec),
            ("vemf", &files.vemf),
            ("vex", &files.vex),
            ("vem", &files.vem),
        ] {
            std::fs::write(out.join(format!("_0_{suffix}.{ext}")), bytes).unwrap();
        }
        let reader = BitVectorsReader::open(&files, &SEGMENT_ID, suffix).unwrap();
        let mut vectors_tsv = String::new();
        let mut search_tsv = String::new();
        for (spec, field) in specs.iter().zip(&fields) {
            let _ = writeln!(
                fields_tsv,
                "{}\t{}\t{}\t{:?}\t{suffix}",
                spec.name, spec.number, spec.dimension, spec.similarity
            );
            let dim = spec.dimension as usize;
            let FieldVectorData::Byte(values) = &field.values else {
                unreachable!()
            };
            for (ord, doc) in field.docs.iter().enumerate() {
                let v = &values[ord * dim..(ord + 1) * dim];
                let _ = writeln!(vectors_tsv, "{}\t{ord}\t{doc}\t{}", spec.number, hex(v));
            }
            for q in 0..30 {
                let query = if q % 2 == 0 {
                    rng.bytes(dim)
                } else {
                    let ord = (rng.next() % field.docs.len() as u64) as usize;
                    values[ord * dim..(ord + 1) * dim].to_vec()
                };
                for k in [1usize, 10, 40] {
                    let (hits, _) = reader
                        .search(
                            spec.number,
                            &query,
                            k,
                            i32::MAX as u64,
                            SearchOptions::default(),
                        )
                        .unwrap();
                    let _ = write!(search_tsv, "{}\t{k}\t{}", spec.number, hex(&query));
                    for (doc, score) in hits {
                        let _ = write!(search_tsv, "\t{doc}:{}", score.to_bits() as i32);
                    }
                    search_tsv.push('\n');
                }
            }
        }
        std::fs::write(out.join(format!("vectors.{suffix}.tsv")), vectors_tsv).unwrap();
        std::fs::write(out.join(format!("search.{suffix}.tsv")), search_tsv).unwrap();
    }
    std::fs::write(out.join("fields.tsv"), fields_tsv).unwrap();
    std::fs::write(
        out.join("manifest.properties"),
        format!("id_hex={}\nmax_doc={MAX_DOC}\n", hex(&SEGMENT_ID)),
    )
    .unwrap();
    println!("wrote bitvectors fixture to {}", out.display());
}
