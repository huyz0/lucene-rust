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
