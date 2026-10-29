//! lucene-codecs' `HnswBitVectorsFormat` against Java: `GenCodecsBitVectors`
//! writes a flushed segment and a force-merged one, routing `b8` and `tiny`
//! to a default `HnswBitVectorsFormat` (`HnswBitVectorsFormat_0`; `tiny` is
//! too small for a graph) and `b32` to `HnswBitVectorsFormat(8, 40)`
//! (`HnswBitVectorsFormat_1`), and records every vector and the hits of
//! `LeafReader.searchNearestVectors` for random and stored queries.
//!
//! Here the Rust reader opens Java's files and returns the same vectors and
//! the same hits with the same score bits, and the Rust writer rebuilds the
//! flushed segment's four files of each group byte for byte.

// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::fs;
use std::path::PathBuf;

use lucene_codecs::bitvectors::{BitVectorsFiles, BitVectorsReader, HnswBitVectorsFormat};
use lucene_codecs::field_infos::{self, FieldInfos};
use lucene_codecs::hnsw_vectors::SearchOptions;
use lucene_codecs::vectors::{FieldVectorData, FlatVectorsField};
use lucene_store::codec_util::ID_LENGTH;

/// `(suffix, format, fields)` of the two groups, fields in number order.
fn groups() -> [(&'static str, HnswBitVectorsFormat, &'static [&'static str]); 2] {
    [
        (
            "HnswBitVectorsFormat_0",
            HnswBitVectorsFormat::default(),
            &["b8", "tiny"],
        ),
        (
            "HnswBitVectorsFormat_1",
            HnswBitVectorsFormat::new(8, 40).unwrap(),
            &["b32"],
        ),
    ]
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/codecs_bitvectors")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn rows(path: PathBuf) -> Vec<Vec<String>> {
    fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{path:?}: {e} (run GenCodecsBitVectors)"))
        .lines()
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect()
}

struct Segment {
    dir: PathBuf,
    name: String,
    id: [u8; ID_LENGTH],
    infos: FieldInfos,
    max_doc: i32,
}

impl Segment {
    fn open(which: &str) -> Segment {
        let dir = root().join(which);
        let index = dir.join("index");
        let si = fs::read_dir(&index)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .find(|n| n.ends_with(".si"))
            .expect("run GenCodecsBitVectors");
        let buf = fs::read(index.join(&si)).unwrap();
        // magic, codec name (vInt length < 128 + bytes), version, id.
        let at = 4 + 1 + usize::from(buf[4]) + 4;
        let mut id = [0u8; ID_LENGTH];
        id.copy_from_slice(&buf[at..at + ID_LENGTH]);
        let name = si.trim_end_matches(".si").to_string();
        let fnm = fs::read(index.join(format!("{name}.fnm"))).unwrap();
        let infos = field_infos::parse(&fnm, &id, "").unwrap();
        let max_doc = rows(dir.join("meta.tsv"))[0][1].parse().unwrap();
        Segment {
            dir,
            name,
            id,
            infos,
            max_doc,
        }
    }

    fn files(&self, suffix: &str) -> BitVectorsFiles {
        let read = |ext: &str| {
            let path = self
                .dir
                .join("index")
                .join(format!("{}_{suffix}.{ext}", self.name));
            fs::read(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"))
        };
        BitVectorsFiles {
            vec: read("vec"),
            vemf: read("vemf"),
            vex: read("vex"),
            vem: read("vem"),
        }
    }

    fn number(&self, field: &str) -> i32 {
        self.infos.field_by_name(field).unwrap().number
    }

    /// `vectors.tsv`'s `(ord, doc, vector)` rows of `field`.
    fn vectors(&self, field: &str) -> Vec<(i32, i32, Vec<u8>)> {
        rows(self.dir.join("vectors.tsv"))
            .into_iter()
            .filter(|r| r[0] == field)
            .map(|r| (r[1].parse().unwrap(), r[2].parse().unwrap(), unhex(&r[3])))
            .collect()
    }
}

fn check_reader(which: &str) {
    let seg = Segment::open(which);
    let search = rows(seg.dir.join("search.tsv"));
    let mut searched = 0;
    for (suffix, _, fields) in groups() {
        let files = seg.files(suffix);
        let reader = BitVectorsReader::open(&files, &seg.id, suffix).unwrap();
        for &field in fields {
            let number = seg.number(field);
            let values = reader.byte_vector_values(number).unwrap();
            let expected = seg.vectors(field);
            assert_eq!(values.size() as usize, expected.len(), "{which} {field}");
            // `getVectorCount` (Lucene 10.6, #16647).
            assert_eq!(reader.vector_count(number).unwrap(), values.size());
            for (ord, doc, vector) in &expected {
                assert_eq!(values.ord_to_doc(*ord).unwrap(), *doc);
                assert_eq!(values.vector(*ord).unwrap(), &vector[..], "{field} {ord}");
            }
            for row in search.iter().filter(|r| r[0] == field) {
                let k: usize = row[1].parse().unwrap();
                let query = unhex(&row[2]);
                let (hits, early) = reader
                    .search(number, &query, k, i32::MAX as u64, SearchOptions::default())
                    .unwrap();
                assert!(!early);
                let got: Vec<String> = hits
                    .iter()
                    .map(|(doc, score)| format!("{doc}:{}", score.to_bits() as i32))
                    .collect();
                assert_eq!(
                    got,
                    row[3..].to_vec(),
                    "{which} {field} k={k} {}",
                    hex(&query)
                );
                searched += 1;
            }
        }
    }
    // Three fields, 40 queries, three values of k.
    assert_eq!(searched, 360, "{which}");
}

#[test]
fn reads_java_flushed_segment() {
    check_reader("flush");
}

#[test]
fn reads_java_merged_segment() {
    check_reader("merged");
}

#[test]
fn rebuilds_java_flushed_segment_byte_for_byte() {
    let seg = Segment::open("flush");
    for (suffix, format, fields) in groups() {
        let flat: Vec<FlatVectorsField> = fields
            .iter()
            .map(|&field| {
                let info = seg.infos.field_by_name(field).unwrap();
                let rows = seg.vectors(field);
                FlatVectorsField {
                    field_number: info.number,
                    similarity: info.vector_similarity_function,
                    dimension: info.vector_dimension,
                    docs: rows.iter().map(|r| r.1).collect(),
                    values: FieldVectorData::Byte(rows.into_iter().flat_map(|r| r.2).collect()),
                }
            })
            .collect();
        let written = format.write(&flat, seg.max_doc, &seg.id, suffix).unwrap();
        let java = seg.files(suffix);
        assert!(written.vec == java.vec, "{suffix}.vec");
        assert!(written.vemf == java.vemf, "{suffix}.vemf");
        assert!(written.vem == java.vem, "{suffix}.vem");
        assert!(written.vex == java.vex, "{suffix}.vex");
    }
}

#[test]
fn refuses_damaged_java_files() {
    let seg = Segment::open("flush");
    let (suffix, _, _) = &groups()[0];
    let files = seg.files(suffix);
    assert!(BitVectorsReader::open(&files, &seg.id, suffix).is_ok());
    // Another group's suffix: every header names this group's.
    assert!(BitVectorsReader::open(&files, &seg.id, "HnswBitVectorsFormat_1").is_err());
    // The segment id with its first byte flipped.
    let mut id = seg.id;
    id[0] ^= 0xff;
    assert!(BitVectorsReader::open(&files, &id, suffix).is_err());
    // Java's `.vem` with one data byte flipped: checksummed whole on open.
    let mut bad = seg.files(suffix);
    let mid = bad.vem.len() / 2;
    bad.vem[mid] ^= 0x01;
    assert!(BitVectorsReader::open(&bad, &seg.id, suffix).is_err());
    // Java's `.vemf` cut short inside its footer.
    let mut bad = seg.files(suffix);
    bad.vemf.truncate(bad.vemf.len() - 4);
    assert!(BitVectorsReader::open(&bad, &seg.id, suffix).is_err());
}
