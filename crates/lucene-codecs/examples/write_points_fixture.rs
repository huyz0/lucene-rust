//! Writes two `points::write`-produced `.kdm`/`.kdi`/`.kdd` triples plus a
//! manifest to the directory given as the first CLI argument.
//!
//! This is the reverse of this repo's usual differential-testing direction
//! (Java writes, Rust reads): here Rust writes **single-dimension**
//! (`LongPoint`-style) BKD trees, and `fixtures/src/VerifyPoints.java` reads
//! the result back through real Lucene's own
//! `Lucene90PointsFormat`/`PointValues.intersect`, using a hand-built
//! `SegmentInfo`/`FieldInfos` -- same division of labor as
//! `write_stored_fields_fixture.rs`/`write_field_infos_fixture.rs`, so this
//! slice doesn't also need a `.si`/`.fnm` writer.
//!
//! Two segments are written:
//! - `_0`: kept strictly under the real default `maxPointsInLeafNode` (512,
//!   see `BKDConfig.DEFAULT_MAX_POINTS_IN_LEAF_NODE`) so the whole field
//!   fits in exactly one BKD leaf -- the shape this write path originally
//!   shipped with.
//! - `_1`: a deliberately small `maxPointsInLeafNode` (8) with enough
//!   points (200, two-thirds of them present, like `_0`) to force **dozens
//!   of leaves and several levels of the packed-index tree** -- the
//!   multi-leaf shape this write path now also supports (see
//!   `points::write`'s module doc for the split algorithm/index format).
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

/// Every third doc skips the field entirely (like `GenPoints.java`'s real
/// fixture) so doc ids aren't a trivial consecutive run and the write
/// side's `BPV_32` doc-id path gets exercised, not just `CONTINUOUS_IDS`.
fn make_points(num_candidates: usize) -> (Vec<(i32, Vec<u8>)>, i32) {
    let mut points: Vec<(i32, Vec<u8>)> = Vec::new();
    let mut doc_id = 0i32;
    for i in 0..num_candidates {
        if i % 3 != 0 {
            let value = (i as i64) * 7919 - 1_000_000;
            points.push((doc_id, long_sortable_bytes(value)));
        }
        doc_id += 1;
    }
    (points, doc_id)
}

fn write_segment(
    dir: &FsDirectory,
    segment_name: &str,
    field: WritePointsField,
    max_points_in_leaf_node: i32,
) {
    let (kdm, kdi, kdd) =
        points::write(&[field], max_points_in_leaf_node, &SEGMENT_ID, "").expect("points write");

    let names = [
        format!("{segment_name}.kdm"),
        format!("{segment_name}.kdi"),
        format!("{segment_name}.kdd"),
    ];
    for (name, bytes) in names.iter().zip([&kdm, &kdi, &kdd]) {
        let mut out = dir.create_output(name).unwrap();
        out.write_bytes(bytes);
        out.close().unwrap();
    }
    dir.sync(&names).unwrap();
}

fn write_manifest_section(
    manifest: &mut std::fs::File,
    prefix: &str,
    max_doc: i32,
    bytes_per_dim: i32,
    points: &[(i32, Vec<u8>)],
) {
    writeln!(manifest, "{prefix}id_hex={}", hex(&SEGMENT_ID)).unwrap();
    writeln!(manifest, "{prefix}max_doc={max_doc}").unwrap();
    writeln!(manifest, "{prefix}field_number={FIELD_NUMBER}").unwrap();
    writeln!(manifest, "{prefix}bytes_per_dim={bytes_per_dim}").unwrap();
    writeln!(manifest, "{prefix}point_count={}", points.len()).unwrap();
    let mut sorted = points.to_vec();
    sorted.sort_by_key(|(doc_id, _)| *doc_id);
    let rendered: Vec<String> = sorted
        .iter()
        .map(|(doc_id, packed)| {
            let value = i64::from_be_bytes(packed.as_slice().try_into().unwrap()) ^ i64::MIN;
            format!("{doc_id}:{value}")
        })
        .collect();
    writeln!(manifest, "{prefix}points={}", rendered.join(";")).unwrap();
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_points_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).unwrap();
    let dir = FsDirectory::open(&out_dir);

    // -- _0: single-leaf (unchanged from this write path's original slice) --
    let (points0, max_doc0) = make_points(NUM_POINTS);
    let field0 = WritePointsField {
        field_number: FIELD_NUMBER,
        bytes_per_dim: 8,
        points: points0.clone(),
    };
    write_segment(&dir, "_0", field0, points::DEFAULT_MAX_POINTS_IN_LEAF_NODE);

    // -- _1: multi-leaf, small maxPointsInLeafNode to force many leaves and
    // several levels of the packed-index tree --
    const MULTI_LEAF_MAX_POINTS_IN_LEAF_NODE: i32 = 8;
    let (points1, max_doc1) = make_points(NUM_POINTS);
    let field1 = WritePointsField {
        field_number: FIELD_NUMBER,
        bytes_per_dim: 8,
        points: points1.clone(),
    };
    write_segment(&dir, "_1", field1, MULTI_LEAF_MAX_POINTS_IN_LEAF_NODE);

    let mut manifest = std::fs::File::create(format!("{out_dir}/manifest.properties")).unwrap();
    write_manifest_section(&mut manifest, "", max_doc0, 8, &points0);
    write_manifest_section(&mut manifest, "segment1_", max_doc1, 8, &points1);

    println!("wrote points fixtures (_0 single-leaf, _1 multi-leaf) to {out_dir}");
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
