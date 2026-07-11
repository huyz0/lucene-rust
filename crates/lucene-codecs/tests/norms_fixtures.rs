//! Differential test against real `.nvm`/`.nvd` files written by an actual
//! IndexWriter: a dense norms field ("body", every doc, varying token counts)
//! and a sparse one ("sparse_body", only docs 0/2/4 have it -- Lucene only
//! picks the IndexedDISI/sparse encoding when a field is missing from some
//! docs entirely). Regenerate with fixtures/src/GenNorms.java.

use lucene_codecs::norms;

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/norms_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenNorms)");
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

    fn get_i32(&self, key: &str) -> i32 {
        self.get(key).parse().unwrap()
    }
}

fn id_from_hex(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for i in 0..16 {
        id[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

fn check_field(manifest: &Manifest, meta_buf: &[u8], data_buf: &[u8], id: &[u8; 16], field: &str) {
    let (_, parsed) = norms::parse_meta(meta_buf, id, "").unwrap();
    let field_number = manifest.get_i32(&format!("field.{field}.number"));
    let entry = parsed.entry(field_number).unwrap();

    let expected: Vec<Option<i64>> = manifest
        .get(&format!("field.{field}.norm_values"))
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
        let got = norms::norm_value(data_buf, entry, doc as i32).unwrap();
        assert_eq!(got, want, "field {field} doc {doc}");
    }
}

#[test]
fn parses_real_dense_norms_and_matches_lucene_values() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let meta_buf =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("nvm_file_name"))).unwrap();
    let data_buf =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("nvd_file_name"))).unwrap();

    let (version, parsed) = norms::parse_meta(&meta_buf, &id, "").unwrap();
    assert_eq!(version, 0);
    let data_version = norms::check_data_header_footer(&data_buf, &id, "").unwrap();
    assert_eq!(data_version, version);

    let field_number = manifest.get_i32("field.body.number");
    let entry = parsed.entry(field_number).unwrap();
    assert!(entry.is_dense());
    assert_eq!(entry.num_docs_with_field, manifest.get_i32("max_doc"));

    check_field(&manifest, &meta_buf, &data_buf, &id, "body");
}

#[test]
fn parses_real_sparse_norms_and_matches_lucene_values() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let meta_buf =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("nvm_file_name"))).unwrap();
    let data_buf =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("nvd_file_name"))).unwrap();

    let (_, parsed) = norms::parse_meta(&meta_buf, &id, "").unwrap();
    let field_number = manifest.get_i32("field.sparse_body.number");
    let entry = parsed.entry(field_number).unwrap();
    assert!(!entry.is_dense());
    assert!(!entry.is_empty_field());
    assert_eq!(entry.num_docs_with_field, 3); // docs 0, 2, 4

    check_field(&manifest, &meta_buf, &data_buf, &id, "sparse_body");
}

#[test]
fn unknown_field_number_is_none() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let meta_buf =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("nvm_file_name"))).unwrap();
    let (_, parsed) = norms::parse_meta(&meta_buf, &id, "").unwrap();
    assert!(parsed.entry(999).is_none());
}

#[test]
fn doc_out_of_range_rejected_for_dense_field() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let field_number = manifest.get_i32("field.body.number");
    let max_doc = manifest.get_i32("max_doc");
    let meta_buf =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("nvm_file_name"))).unwrap();
    let (_, parsed) = norms::parse_meta(&meta_buf, &id, "").unwrap();
    let data_buf =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("nvd_file_name"))).unwrap();
    let entry = parsed.entry(field_number).unwrap();
    assert!(norms::norm_value(&data_buf, entry, max_doc).is_err());
    assert!(norms::norm_value(&data_buf, entry, -1).is_err());
}

#[test]
fn wrong_segment_id_rejected() {
    let manifest = Manifest::load();
    let meta_buf =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("nvm_file_name"))).unwrap();
    let wrong_id = [0u8; 16];
    assert!(norms::parse_meta(&meta_buf, &wrong_id, "").is_err());
}
