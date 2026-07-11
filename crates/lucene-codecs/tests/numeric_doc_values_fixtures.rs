//! Differential test against real `.dvm`/`.dvd` files written by an actual
//! IndexWriter: a dense field with arbitrary signed values ("varying"), a
//! dense field whose values share a large GCD ("gcd"), and a sparse field
//! present on only some docs ("sparse"). Regenerate with
//! fixtures/src/GenNumericDocValues.java.

use lucene_codecs::{field_infos, numeric_doc_values as ndv};

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/numeric_dv_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenNumericDocValues)");
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

fn load_field_infos(manifest: &Manifest, id: &[u8; 16]) -> field_infos::FieldInfos {
    let buf = std::fs::read(format!("{}{}.raw", dir(), manifest.get("fnm_file_name"))).unwrap();
    field_infos::parse(&buf, id, "").unwrap()
}

/// `Lucene90DocValuesFormat` is wrapped in a `PerFieldDocValuesFormat`, which
/// gives each format instance its own segment-suffix ("Lucene90_0" here) on
/// top of the segment's own (empty) suffix -- derive it from the real
/// filename rather than hardcoding it, since the counter can vary.
fn dv_suffix(manifest: &Manifest) -> String {
    let segment_name = manifest.get("segment_name");
    let name = manifest.get("dvm_file_name");
    name.strip_prefix(&format!("{segment_name}_"))
        .and_then(|s| s.strip_suffix(".dvm"))
        .unwrap_or_else(|| panic!("unexpected dvm file name shape: {name}"))
        .to_string()
}

fn field_number(manifest: &Manifest, field: &str) -> i32 {
    manifest
        .get("field_numbers")
        .split(',')
        .find_map(|kv| {
            let (name, num) = kv.split_once(':').unwrap();
            (name == field).then(|| num.parse().unwrap())
        })
        .unwrap_or_else(|| panic!("field {field} missing from field_numbers"))
}

fn check_field(
    manifest: &Manifest,
    meta_buf: &[u8],
    data_buf: &[u8],
    fis: &field_infos::FieldInfos,
    field: &str,
) {
    let suffix = dv_suffix(manifest);
    let (_, parsed) =
        ndv::parse_meta(meta_buf, &id_from_hex(manifest.get("id_hex")), &suffix, fis).unwrap();
    let entry = parsed.entry(field_number(manifest, field)).unwrap();

    let expected: Vec<Option<i64>> = manifest
        .get(&format!("field.{field}.values"))
        .split(',')
        .map(|s| {
            if s == "NONE" {
                None
            } else {
                Some(s.parse().unwrap())
            }
        })
        .collect();

    for (doc, &want) in expected.iter().enumerate() {
        let got = ndv::numeric_value(data_buf, entry, doc as i32).unwrap();
        assert_eq!(got, want, "field {field} doc {doc}");
    }
}

#[test]
fn parses_real_varying_numeric_dv_and_matches_lucene_values() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let fis = load_field_infos(&manifest, &id);
    let meta_buf =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvm_file_name"))).unwrap();
    let data_buf =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvd_file_name"))).unwrap();

    let suffix = dv_suffix(&manifest);
    let (version, parsed) = ndv::parse_meta(&meta_buf, &id, &suffix, &fis).unwrap();
    assert_eq!(version, 2);
    let data_version = ndv::check_data_header_footer(&data_buf, &id, &suffix).unwrap();
    assert_eq!(data_version, version);

    let entry = parsed.entry(field_number(&manifest, "varying")).unwrap();
    assert!(entry.is_dense());

    check_field(&manifest, &meta_buf, &data_buf, &fis, "varying");
}

#[test]
fn parses_real_gcd_numeric_dv_and_matches_lucene_values() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let fis = load_field_infos(&manifest, &id);
    let meta_buf =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvm_file_name"))).unwrap();
    let data_buf =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvd_file_name"))).unwrap();

    check_field(&manifest, &meta_buf, &data_buf, &fis, "gcd");
}

#[test]
fn parses_real_sparse_numeric_dv_and_matches_lucene_values() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let fis = load_field_infos(&manifest, &id);
    let meta_buf =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvm_file_name"))).unwrap();
    let data_buf =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvd_file_name"))).unwrap();

    let suffix = dv_suffix(&manifest);
    let (_, parsed) = ndv::parse_meta(&meta_buf, &id, &suffix, &fis).unwrap();
    let entry = parsed.entry(field_number(&manifest, "sparse")).unwrap();
    assert!(!entry.is_dense());
    assert!(!entry.is_empty_field());

    check_field(&manifest, &meta_buf, &data_buf, &fis, "sparse");
}

#[test]
fn id_field_is_indexed_not_doc_values_and_is_absent_from_numeric_meta() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let fis = load_field_infos(&manifest, &id);
    let meta_buf =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvm_file_name"))).unwrap();

    let suffix = dv_suffix(&manifest);
    let (_, parsed) = ndv::parse_meta(&meta_buf, &id, &suffix, &fis).unwrap();
    assert!(parsed.entry(field_number(&manifest, "id")).is_none());
}
