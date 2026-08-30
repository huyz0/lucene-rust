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

/// Every `point_estimate.*` case real Lucene recorded in the manifest
/// (`fixtures/src/AppendPointEstimateManifest.java`), checked against
/// `PointsReader::estimate_range_point_count`.
///
/// The estimate is deliberately *not* the match count -- `val.narrow` matches
/// 8 points and estimates 256, `multi.corner` matches 251 and estimates 744 --
/// so this pins the tree walk and `BKDPointTree.size()`'s arithmetic, which a
/// test written against the exact answer could not do. `.exact` is asserted
/// alongside from the same visitor, so a port that quietly answered the exact
/// count everywhere would fail on both halves rather than look plausible on
/// one.
#[test]
fn estimate_point_count_matches_lucene() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let kdm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("kdm_file_name"))).unwrap();
    let kdi = std::fs::read(format!("{}{}.raw", dir(), manifest.get("kdi_file_name"))).unwrap();
    let kdd = std::fs::read(format!("{}{}.raw", dir(), manifest.get("kdd_file_name"))).unwrap();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();

    let cases: Vec<(&str, i32)> = vec![
        ("val.all", manifest.get("field_number").parse().unwrap()),
        (
            "val.none_below",
            manifest.get("field_number").parse().unwrap(),
        ),
        (
            "val.none_above",
            manifest.get("field_number").parse().unwrap(),
        ),
        (
            "val.lower_half",
            manifest.get("field_number").parse().unwrap(),
        ),
        ("val.narrow", manifest.get("field_number").parse().unwrap()),
        ("val.single", manifest.get("field_number").parse().unwrap()),
        (
            "val.from_middle",
            manifest.get("field_number").parse().unwrap(),
        ),
        (
            "multi.all",
            manifest.get("multi_field_number").parse().unwrap(),
        ),
        (
            "multi.dim1_only",
            manifest.get("multi_field_number").parse().unwrap(),
        ),
        (
            "multi.corner",
            manifest.get("multi_field_number").parse().unwrap(),
        ),
        (
            "multi.empty",
            manifest.get("multi_field_number").parse().unwrap(),
        ),
        (
            "shape.all",
            manifest.get("shape_field_number").parse().unwrap(),
        ),
        (
            "shape.quadrant",
            manifest.get("shape_field_number").parse().unwrap(),
        ),
        (
            "shape.strip",
            manifest.get("shape_field_number").parse().unwrap(),
        ),
    ];
    assert!(!cases.is_empty());

    for (case, field_number) in cases {
        let lower = hex_bytes(manifest.get(&format!("point_estimate.{case}.lower_hex")));
        let upper = hex_bytes(manifest.get(&format!("point_estimate.{case}.upper_hex")));
        let want: i64 = manifest
            .get(&format!("point_estimate.{case}.points"))
            .parse()
            .unwrap();
        let got = reader
            .estimate_range_point_count(field_number, &lower, &upper)
            .unwrap();
        assert_eq!(got, want, "estimatePointCount for {case}");

        // The same box run exactly, so the two answers are checked against
        // each other as well as against Lucene's.
        let want_exact: i64 = manifest
            .get(&format!("point_estimate.{case}.exact"))
            .parse()
            .unwrap();
        let exact = reader
            .range_query(field_number, &lower, &upper)
            .unwrap()
            .len() as i64;
        assert_eq!(exact, want_exact, "exact match count for {case}");

        // `estimateDocCount` over the same visitor. The field name in the key
        // is the case's own prefix (`val`/`multi`/`shape`).
        let field_key = case.split('.').next().unwrap();
        let size: i64 = manifest
            .get(&format!("point_estimate.{field_key}.size"))
            .parse()
            .unwrap();
        let field_doc_count: i32 = manifest
            .get(&format!("point_estimate.{field_key}.doc_count"))
            .parse()
            .unwrap();
        let want_docs: i64 = manifest
            .get(&format!("point_estimate.{case}.docs"))
            .parse()
            .unwrap();
        assert_eq!(
            estimate_doc_count(got, size, field_doc_count),
            want_docs,
            "estimateDocCount for {case}"
        );
    }
}

/// Java's `PointValues.estimateDocCount`, duplicated here rather than reached
/// through `lucene-search`: `lucene-codecs` sits below it in the dependency
/// graph (see the `architecture` skill), so a `lucene-codecs` test cannot
/// depend on the crate that owns the function. Five lines, and pinned against
/// real Lucene's own answer by its only caller above -- if this copy and
/// `points_query::estimate_doc_count` ever disagree, one of the two fails.
fn estimate_doc_count(estimated_point_count: i64, size: i64, doc_count: i32) -> i64 {
    let size_f = size as f64;
    if estimated_point_count >= size {
        return i64::from(doc_count);
    }
    if size == i64::from(doc_count) || estimated_point_count == 0 {
        return estimated_point_count;
    }
    let doc_estimate = (f64::from(doc_count)
        * (1.0
            - ((size_f - estimated_point_count as f64) / size_f)
                .powf(size_f / f64::from(doc_count)))) as i64;
    if doc_estimate == 0 {
        1
    } else {
        doc_estimate
    }
}

/// `estimate_point_count_bounded` is `isEstimatedPointCountGreaterThanOrEqualTo`'s
/// engine: it stops descending once the running cost reaches the bound, so it
/// reports at least the bound exactly when the unbounded walk would.
#[test]
fn bounded_estimate_agrees_with_the_unbounded_one_about_reaching_the_bound() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let kdm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("kdm_file_name"))).unwrap();
    let kdi = std::fs::read(format!("{}{}.raw", dir(), manifest.get("kdi_file_name"))).unwrap();
    let kdd = std::fs::read(format!("{}{}.raw", dir(), manifest.get("kdd_file_name"))).unwrap();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let field: i32 = manifest.get("field_number").parse().unwrap();

    for case in ["val.all", "val.lower_half", "val.narrow", "val.none_below"] {
        let lower = hex_bytes(manifest.get(&format!("point_estimate.{case}.lower_hex")));
        let upper = hex_bytes(manifest.get(&format!("point_estimate.{case}.upper_hex")));
        let full = reader
            .estimate_range_point_count(field, &lower, &upper)
            .unwrap();
        for bound in [1i64, 100, 256, 512, 700, 1333, 5000] {
            let mut visitor = RangeBox::new(&lower, &upper, &reader, field);
            let bounded = reader
                .estimate_point_count_bounded(field, &mut visitor.visitor, bound)
                .unwrap();
            assert_eq!(
                bounded >= bound,
                full >= bound,
                "{case}: bounded={bounded} full={full} bound={bound}"
            );
        }
    }
}

/// A tiny adapter so the bounded test can reuse the crate's own range visitor
/// shape without `estimate_range_point_count`'s fixed `i64::MAX` bound.
struct RangeBox {
    visitor: BoxVisitor,
}

impl RangeBox {
    fn new(lower: &[u8], upper: &[u8], reader: &points::PointsReader<'_>, field: i32) -> Self {
        let f = reader.field(field).unwrap();
        RangeBox {
            visitor: BoxVisitor {
                lower: lower.to_vec(),
                upper: upper.to_vec(),
                num_index_dims: f.num_index_dims as usize,
                bytes_per_dim: f.bytes_per_dim as usize,
            },
        }
    }
}

struct BoxVisitor {
    lower: Vec<u8>,
    upper: Vec<u8>,
    num_index_dims: usize,
    bytes_per_dim: usize,
}

impl points::IntersectVisitor for BoxVisitor {
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> points::Relation {
        let mut crosses = false;
        for dim in 0..self.num_index_dims {
            let (lo, hi) = (dim * self.bytes_per_dim, (dim + 1) * self.bytes_per_dim);
            if max_packed[lo..hi] < self.lower[lo..hi] || min_packed[lo..hi] > self.upper[lo..hi] {
                return points::Relation::CellOutsideQuery;
            }
            crosses |=
                min_packed[lo..hi] < self.lower[lo..hi] || max_packed[lo..hi] > self.upper[lo..hi];
        }
        if crosses {
            points::Relation::CellCrossesQuery
        } else {
            points::Relation::CellInsideQuery
        }
    }

    fn visit(&mut self, _doc_id: i32) {
        panic!("estimate_point_count must never visit a document");
    }

    fn visit_with_value(&mut self, _doc_id: i32, _packed_value: &[u8]) {
        panic!("estimate_point_count must never decode a point");
    }
}

fn hex_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap())
        .collect()
}
