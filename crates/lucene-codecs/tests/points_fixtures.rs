//! Differential test against real `.kdm`/`.kdi`/`.kdd` files written by an
//! actual IndexWriter: 2000 docs, a single-dimension `LongPoint` field on
//! about two-thirds of them (every third doc skips it), forcing several
//! leaves (default maxPointsInLeafNode=512) and non-continuous doc ids
//! within a leaf. Regenerate with fixtures/src/GenPoints.java.

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
