//! Differential test against real `.kdm`/`.kdi`/`.kdd` files written by an
//! actual IndexWriter: 2000 docs, a single-dimension `LongPoint` field on
//! about two-thirds of them (every third doc skips it), forcing several
//! leaves (default maxPointsInLeafNode=512) and non-continuous doc ids
//! within a leaf. Regenerate with fixtures/src/GenPoints.java.
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::points;

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/points_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenPoints)");
        let kv = text
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Manifest { kv }
    }

    fn get(&self, key: &str) -> &str {
        self.kv
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
    }
}

fn id_from_hex(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for i in 0..16 {
        id[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

/// Undoes `NumericUtils.sortableBytesToLong`'s bias: the on-disk packed
/// value is `value XOR 0x8000000000000000` as big-endian bytes (sign bit
/// flipped so unsigned byte comparison matches signed numeric ordering).
fn sortable_bytes_to_long(bytes: &[u8]) -> i64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(bytes);
    let unsigned = u64::from_be_bytes(buf);
    (unsigned ^ 0x8000_0000_0000_0000) as i64
}

#[test]
fn parses_real_points_and_matches_lucene_values() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let kdm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("kdm_file_name"))).unwrap();
    let kdi = std::fs::read(format!("{}{}.raw", dir(), manifest.get("kdi_file_name"))).unwrap();
    let kdd = std::fs::read(format!("{}{}.raw", dir(), manifest.get("kdd_file_name"))).unwrap();

    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let field_number: i32 = manifest.get("field_number").parse().unwrap();
    let field = reader.field(field_number).unwrap();

    assert_eq!(
        field.num_dims,
        manifest.get("num_dims").parse::<i32>().unwrap()
    );
    assert_eq!(
        field.num_index_dims,
        manifest.get("num_index_dims").parse::<i32>().unwrap()
    );
    assert_eq!(
        field.bytes_per_dim,
        manifest.get("bytes_per_dim").parse::<i32>().unwrap()
    );
    assert_eq!(
        field.point_count,
        manifest.get("point_count").parse::<i64>().unwrap()
    );
    assert_eq!(
        field.doc_count,
        manifest.get("doc_count").parse::<i32>().unwrap()
    );

    let mut got: Vec<(i32, i64)> = reader
        .decode_all_points(field_number)
        .unwrap()
        .into_iter()
        .map(|p| (p.doc_id, sortable_bytes_to_long(&p.packed_value)))
        .collect();
    got.sort_by_key(|&(doc_id, _)| doc_id);

    let mut want: Vec<(i32, i64)> = manifest
        .get("points")
        .split(';')
        .map(|entry| {
            let (doc_id, value) = entry.split_once(':').unwrap();
            (doc_id.parse().unwrap(), value.parse().unwrap())
        })
        .collect();
    want.sort_by_key(|&(doc_id, _)| doc_id);

    assert_eq!(got.len(), want.len(), "point count");
    assert_eq!(got, want);
}

/// Undoes `NumericUtils.sortableBytesToInt`'s bias, mirroring
/// `sortable_bytes_to_long` above but for 4-byte dimensions.
fn sortable_bytes_to_int(bytes: &[u8]) -> i32 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(bytes);
    let unsigned = u32::from_be_bytes(buf);
    (unsigned ^ 0x8000_0000) as i32
}

/// The "multi" field (see `GenPoints.java`) is a 2-dimension `IntPoint`
/// where dim0 is a bijective hash of the doc id (spread across the full
/// 32-bit range) and dim1 only takes 4 distinct values (`i % 4`). This
/// shape forces real Lucene's `BKDWriter` to pick dim1 as `sortedDim`
/// (lowest in-leaf cardinality wins -- and dim0's hashed spread keeps its
/// own in-leaf cardinality far above 4, unlike a naive sequential dim0,
/// which BKDWriter's recursive range-narrowing squeezes down to 1-2 and
/// which therefore never loses the tie to dim1), and since every packed
/// tuple is still unique, the high-cardinality path -- so every leaf in
/// this fixture is written with `compressedDim == 1`, a real dimension
/// index greater than zero. This exercises `read_leaf_block`'s
/// `compressed_byte_offset = compressed_dim * bytes_per_dim + ...` math
/// with a nonzero `compressed_dim`, which the single-dimension `val` field
/// (compressedDim always 0 when present) can't reach.
///
/// `GenPoints.java` mechanically verifies this at generation time via
/// `CompressedDimSpy` (an independent, from-scratch reader of the raw
/// `.kdd`/`.kdi` bytes that does not go through this crate's decoder) and
/// records the observed per-leaf `compressedDim` byte in the
/// `multi_leaf_compressed_dims` manifest key; the assertion below re-checks
/// that recorded value directly, so this test fails if a future
/// regeneration ever stops exercising the `compressed_dim >= 1` branch,
/// rather than silently passing on a fixture that no longer covers it.
#[test]
fn parses_real_multi_dim_points_and_matches_lucene_values() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let kdm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("kdm_file_name"))).unwrap();
    let kdi = std::fs::read(format!("{}{}.raw", dir(), manifest.get("kdi_file_name"))).unwrap();
    let kdd = std::fs::read(format!("{}{}.raw", dir(), manifest.get("kdd_file_name"))).unwrap();

    let leaf_compressed_dims: Vec<i32> = manifest
        .get("multi_leaf_compressed_dims")
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    assert!(
        leaf_compressed_dims.iter().any(|&cd| cd >= 1),
        "fixture regenerated without ever exercising compressed_dim >= 1 \
         (GenPoints.java's own CompressedDimSpy check should have already \
         caught this at generation time): {leaf_compressed_dims:?}"
    );

    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let field_number: i32 = manifest.get("multi_field_number").parse().unwrap();
    let field = reader.field(field_number).unwrap();

    assert_eq!(
        field.num_dims,
        manifest.get("multi_num_dims").parse::<i32>().unwrap()
    );
    assert_eq!(
        field.num_index_dims,
        manifest.get("multi_num_index_dims").parse::<i32>().unwrap()
    );
    assert_eq!(
        field.bytes_per_dim,
        manifest.get("multi_bytes_per_dim").parse::<i32>().unwrap()
    );
    assert_eq!(
        field.point_count,
        manifest.get("multi_point_count").parse::<i64>().unwrap()
    );
    assert_eq!(
        field.doc_count,
        manifest.get("multi_doc_count").parse::<i32>().unwrap()
    );

    let bytes_per_dim = field.bytes_per_dim as usize;
    let mut got: Vec<(i32, i32, i32)> = reader
        .decode_all_points(field_number)
        .unwrap()
        .into_iter()
        .map(|p| {
            let dim0 = sortable_bytes_to_int(&p.packed_value[0..bytes_per_dim]);
            let dim1 = sortable_bytes_to_int(&p.packed_value[bytes_per_dim..2 * bytes_per_dim]);
            (p.doc_id, dim0, dim1)
        })
        .collect();
    got.sort_by_key(|&(doc_id, _, _)| doc_id);

    let mut want: Vec<(i32, i32, i32)> = manifest
        .get("multi_points")
        .split(';')
        .map(|entry| {
            let mut parts = entry.split(':');
            let doc_id = parts.next().unwrap().parse().unwrap();
            let dim0 = parts.next().unwrap().parse().unwrap();
            let dim1 = parts.next().unwrap().parse().unwrap();
            (doc_id, dim0, dim1)
        })
        .collect();
    want.sort_by_key(|&(doc_id, _, _)| doc_id);

    assert_eq!(got.len(), want.len(), "point count");
    assert_eq!(got, want);
}

/// Differential test for the `shape` field: `num_dims=4`/`num_index_dims=2`
/// (a `LatLonShape`-style bounding box with two trailing, non-indexed
/// data-only dimensions), written by a real `IndexWriter` via a custom
/// `FieldType::setDimensions(4, 2, Integer.BYTES)` (see `GenPoints.java`).
/// Proves this port's write-side support for `num_index_dims < num_dims`
/// against real Lucene bytes read back through this port's own reader --
/// every point's full 4-dimension packed value, including the two
/// non-indexed dims, must round-trip identically.
#[test]
fn parses_real_shape_points_and_matches_lucene_values() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let kdm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("kdm_file_name"))).unwrap();
    let kdi = std::fs::read(format!("{}{}.raw", dir(), manifest.get("kdi_file_name"))).unwrap();
    let kdd = std::fs::read(format!("{}{}.raw", dir(), manifest.get("kdd_file_name"))).unwrap();

    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let field_number: i32 = manifest.get("shape_field_number").parse().unwrap();
    let field = reader.field(field_number).unwrap();

    assert_eq!(
        field.num_dims,
        manifest.get("shape_num_dims").parse::<i32>().unwrap()
    );
    assert_eq!(
        field.num_index_dims,
        manifest.get("shape_num_index_dims").parse::<i32>().unwrap()
    );
    assert_eq!(field.num_index_dims, 2);
    assert_eq!(field.num_dims, 4);
    assert_eq!(
        field.bytes_per_dim,
        manifest.get("shape_bytes_per_dim").parse::<i32>().unwrap()
    );
    assert_eq!(
        field.point_count,
        manifest.get("shape_point_count").parse::<i64>().unwrap()
    );
    assert_eq!(
        field.doc_count,
        manifest.get("shape_doc_count").parse::<i32>().unwrap()
    );

    let bytes_per_dim = field.bytes_per_dim as usize;
    let mut got: Vec<(i32, i32, i32, i32, i32)> = reader
        .decode_all_points(field_number)
        .unwrap()
        .into_iter()
        .map(|p| {
            let d0 = sortable_bytes_to_int(&p.packed_value[0..bytes_per_dim]);
            let d1 = sortable_bytes_to_int(&p.packed_value[bytes_per_dim..2 * bytes_per_dim]);
            let d2 = sortable_bytes_to_int(&p.packed_value[2 * bytes_per_dim..3 * bytes_per_dim]);
            let d3 = sortable_bytes_to_int(&p.packed_value[3 * bytes_per_dim..4 * bytes_per_dim]);
            (p.doc_id, d0, d1, d2, d3)
        })
        .collect();
    got.sort_by_key(|&(doc_id, ..)| doc_id);

    let mut want: Vec<(i32, i32, i32, i32, i32)> = manifest
        .get("shape_points")
        .split(';')
        .map(|entry| {
            let mut parts = entry.split(':');
            let doc_id = parts.next().unwrap().parse().unwrap();
            let d0 = parts.next().unwrap().parse().unwrap();
            let d1 = parts.next().unwrap().parse().unwrap();
            let d2 = parts.next().unwrap().parse().unwrap();
            let d3 = parts.next().unwrap().parse().unwrap();
            (doc_id, d0, d1, d2, d3)
        })
        .collect();
    want.sort_by_key(|&(doc_id, ..)| doc_id);

    assert_eq!(got.len(), want.len(), "point count");
    assert_eq!(got, want);
}

/// Differential test for the **pruning** traversal
/// (`points::PointsReader::intersect` / `range_query`, the port of
/// `PointValues.intersect` over `BKDReader.BKDPointTree`) against the same
/// real Java-written `.kdi` bytes.
///
/// This is the only test that exercises the packed index's split *values*
/// rather than merely walking past them: `decode_all_points` reads each
/// node's split descriptor only to know how many suffix bytes to skip, so a
/// wrong prefix/first-diff-byte/negative-delta reconstruction would go
/// completely unnoticed there. Here every cell bound handed to `compare`
/// comes from those reconstructed values, so any error makes `intersect`
/// prune a subtree that actually contains matches -- and the assertion below
/// (equality with a brute-force filter over the manifest's own recorded
/// values, i.e. against Java's ground truth, not against this port's
/// decoder) fails.
#[test]
fn intersect_over_real_lucene_packed_index_matches_brute_force() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let kdm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("kdm_file_name"))).unwrap();
    let kdi = std::fs::read(format!("{}{}.raw", dir(), manifest.get("kdi_file_name"))).unwrap();
    let kdd = std::fs::read(format!("{}{}.raw", dir(), manifest.get("kdd_file_name"))).unwrap();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();

    // --- single-dimension LongPoint field ---
    let field_number: i32 = manifest.get("field_number").parse().unwrap();
    assert!(
        reader.field(field_number).unwrap().num_leaves > 1,
        "fixture must have a real multi-node tree for pruning to mean anything"
    );
    let want_points: Vec<(i32, i64)> = manifest
        .get("points")
        .split(';')
        .map(|entry| {
            let (doc_id, value) = entry.split_once(':').unwrap();
            (doc_id.parse().unwrap(), value.parse().unwrap())
        })
        .collect();
    let min_value = want_points.iter().map(|&(_, v)| v).min().unwrap();
    let max_value = want_points.iter().map(|&(_, v)| v).max().unwrap();
    let span = max_value - min_value;

    let long_bytes = |v: i64| ((v as u64) ^ 0x8000_0000_0000_0000).to_be_bytes().to_vec();
    for (lo, hi) in [
        (min_value, max_value),
        (min_value - 1000, min_value - 1),
        (max_value + 1, max_value + 1000),
        (min_value, min_value),
        (max_value, max_value),
        (min_value + span / 4, min_value + span / 2),
        (min_value + span / 3, min_value + span / 3),
        (min_value + 1, max_value - 1),
    ] {
        let mut want: Vec<i32> = want_points
            .iter()
            .filter(|&&(_, v)| v >= lo && v <= hi)
            .map(|&(d, _)| d)
            .collect();
        want.sort_unstable();
        let mut got = reader
            .range_query(field_number, &long_bytes(lo), &long_bytes(hi))
            .unwrap();
        got.sort_unstable();
        assert_eq!(got, want, "1-dim range [{lo}, {hi}]");
    }

    // --- 2-dimension IntPoint field (split dimension alternates, so the
    // per-dimension `lastSplitValues`/`negativeDeltas` state matters) ---
    let multi_field: i32 = manifest.get("multi_field_number").parse().unwrap();
    let multi_points: Vec<(i32, i32, i32)> = manifest
        .get("multi_points")
        .split(';')
        .map(|entry| {
            let mut parts = entry.split(':');
            (
                parts.next().unwrap().parse().unwrap(),
                parts.next().unwrap().parse().unwrap(),
                parts.next().unwrap().parse().unwrap(),
            )
        })
        .collect();
    let int_bytes = |v: i32| ((v as u32) ^ 0x8000_0000).to_be_bytes().to_vec();
    let pack = |x: i32, y: i32| {
        let mut out = int_bytes(x);
        out.extend_from_slice(&int_bytes(y));
        out
    };
    for (x0, x1, y0, y1) in [
        (i32::MIN, i32::MAX, i32::MIN, i32::MAX),
        (i32::MIN, i32::MAX, 1, 2),
        (0, i32::MAX, 0, 0),
        (i32::MIN, -1, 3, 3),
        (-100, 100, 0, 3),
        (i32::MIN, i32::MAX, 4, i32::MAX),
    ] {
        let mut want: Vec<i32> = multi_points
            .iter()
            .filter(|&&(_, x, y)| x >= x0 && x <= x1 && y >= y0 && y <= y1)
            .map(|&(d, _, _)| d)
            .collect();
        want.sort_unstable();
        let mut got = reader
            .range_query(multi_field, &pack(x0, y0), &pack(x1, y1))
            .unwrap();
        got.sort_unstable();
        assert_eq!(got, want, "2-dim box ({x0}..{x1}, {y0}..{y1})");
    }

    // --- 4-dimension / 2-index-dimension shape field: the two trailing
    // data-only dimensions must never take part in a cell bound. ---
    let shape_field: i32 = manifest.get("shape_field_number").parse().unwrap();
    let shape_points: Vec<(i32, i32, i32)> = manifest
        .get("shape_points")
        .split(';')
        .map(|entry| {
            let mut parts = entry.split(':');
            let doc_id = parts.next().unwrap().parse().unwrap();
            let d0 = parts.next().unwrap().parse().unwrap();
            let d1 = parts.next().unwrap().parse().unwrap();
            (doc_id, d0, d1)
        })
        .collect();
    let d0_min = shape_points.iter().map(|&(_, d, _)| d).min().unwrap();
    let d0_max = shape_points.iter().map(|&(_, d, _)| d).max().unwrap();
    for (a0, a1) in [
        (i32::MIN, i32::MAX),
        (d0_min, d0_min),
        (d0_max, d0_max),
        (d0_min, (d0_min / 2).saturating_add(d0_max / 2)),
        (d0_max, i32::MAX),
    ] {
        let mut want: Vec<i32> = shape_points
            .iter()
            .filter(|&&(_, d0, _)| d0 >= a0 && d0 <= a1)
            .map(|&(d, _, _)| d)
            .collect();
        want.sort_unstable();
        let mut got = reader
            .range_query(shape_field, &pack(a0, i32::MIN), &pack(a1, i32::MAX))
            .unwrap();
        got.sort_unstable();
        assert_eq!(got, want, "shape dim0 range [{a0}, {a1}]");
    }
}
