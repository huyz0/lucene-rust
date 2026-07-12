//! Writes a `points::write`-produced `.kdm`/`.kdi`/`.kdd` triple plus a
//! manifest to the directory given as the first CLI argument.
//!
//! This is the reverse of this repo's usual differential-testing direction
//! (Java writes, Rust reads): here Rust writes a **single-leaf,
//! single-dimension** (`LongPoint`-style) BKD tree for one field, and
//! `fixtures/src/VerifyPoints.java` reads the result back through real
//! Lucene's own `Lucene90PointsFormat`/`PointValues.intersect`, using a
//! hand-built `SegmentInfo`/`FieldInfos` -- same division of labor as
//! `write_stored_fields_fixture.rs`/`write_field_infos_fixture.rs`, so this
//! slice doesn't also need a `.si`/`.fnm` writer.
//!
//! Kept strictly under the real default `maxPointsInLeafNode` (512, see
//! `BKDConfig.DEFAULT_MAX_POINTS_IN_LEAF_NODE`) so the whole field fits in
//! exactly one BKD leaf -- the only shape `points::write` supports (see its
//! module doc for why multi-leaf/multi-dimension are deferred).
//!
//! Run: `cargo run -p lucene-codecs --example write_points_fixture -- <dir>`

use lucene_codecs::points::{self, WritePointsField};
use lucene_store::{DataOutput, Directory, FsDirectory};
use std::io::Write;

const SEGMENT_ID: [u8; 16] = *b"rustwrittenkdt01";
const FIELD_NUMBER: i32 = 0;
const NUM_POINTS: usize = 200; // well under the 512-per-leaf default

/// `NumericUtils.longToSortableBytes`: flip the sign bit, then big-endian --
/// this is the byte encoding real `LongPoint`/`PointValues` readers expect,
/// unrelated to and simpler than this module's own vint/vlong wire helpers.
fn long_sortable_bytes(v: i64) -> Vec<u8> {
    ((v ^ i64::MIN) as u64).to_be_bytes().to_vec()
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_points_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).unwrap();

    // Every third doc skips the field entirely (like GenPoints.java's real
    // fixture) so doc ids aren't a trivial consecutive run and the write
    // side's BPV_32 doc-id path gets exercised, not just CONTINUOUS_IDS.
    let mut points: Vec<(i32, Vec<u8>)> = Vec::new();
    let mut doc_id = 0i32;
    for i in 0..NUM_POINTS {
        if i % 3 != 0 {
            let value = (i as i64) * 7919 - 1_000_000;
            points.push((doc_id, long_sortable_bytes(value)));
        }
        doc_id += 1;
    }
    let max_doc = doc_id;

    let field = WritePointsField {
        field_number: FIELD_NUMBER,
        bytes_per_dim: 8,
        points: points.clone(),
    };
    let (kdm, kdi, kdd) = points::write(
        &[field],
        points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
        &SEGMENT_ID,
        "",
    )
    .expect("single-leaf write");

    let dir = FsDirectory::open(&out_dir);
    for (name, bytes) in [("_0.kdm", &kdm), ("_0.kdi", &kdi), ("_0.kdd", &kdd)] {
        let mut out = dir.create_output(name).unwrap();
        out.write_bytes(bytes);
        out.close().unwrap();
    }
    dir.sync(&[
        "_0.kdm".to_string(),
        "_0.kdi".to_string(),
        "_0.kdd".to_string(),
    ])
    .unwrap();

    let mut manifest = std::fs::File::create(format!("{out_dir}/manifest.properties")).unwrap();
    writeln!(manifest, "id_hex={}", hex(&SEGMENT_ID)).unwrap();
    writeln!(manifest, "max_doc={max_doc}").unwrap();
    writeln!(manifest, "field_number={FIELD_NUMBER}").unwrap();
    writeln!(manifest, "bytes_per_dim=8").unwrap();
    writeln!(manifest, "point_count={}", points.len()).unwrap();
    let mut sorted = points.clone();
    sorted.sort_by_key(|(doc_id, _)| *doc_id);
    let rendered: Vec<String> = sorted
        .iter()
        .map(|(doc_id, packed)| {
            let value = i64::from_be_bytes(packed.as_slice().try_into().unwrap()) ^ i64::MIN;
            format!("{doc_id}:{value}")
        })
        .collect();
    writeln!(manifest, "points={}", rendered.join(";")).unwrap();

    println!("wrote points fixture to {out_dir}");
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
