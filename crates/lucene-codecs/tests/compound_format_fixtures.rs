//! Differential test against a real `.cfs`/`.cfe` pair written by an actual
//! IndexWriter with `useCompoundFile=true`. Regenerate with
//! fixtures/src/GenCompoundFormat.java.

use lucene_codecs::compound_format;
use lucene_store::codec_util::ID_LENGTH;

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/compound_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenCompoundFormat)");
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

fn id_from_hex(hex: &str) -> [u8; ID_LENGTH] {
    let mut id = [0u8; ID_LENGTH];
    for i in 0..ID_LENGTH {
        id[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

fn expected_sub_files(manifest: &Manifest) -> Vec<(String, i64)> {
    manifest
        .get("sub_files")
        .split(',')
        .map(|kv| {
            let (name, len) = kv.rsplit_once(':').unwrap();
            (name.to_string(), len.parse().unwrap())
        })
        .collect()
}

#[test]
fn parses_real_entries_and_lists_every_real_sub_file() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let cfe_buf = std::fs::read(format!("{}{}.raw", dir(), manifest.get("cfe_file_name"))).unwrap();

    let entries = compound_format::parse_entries(&cfe_buf, &id).unwrap();

    let expected = expected_sub_files(&manifest);
    assert_eq!(entries.entries.len(), expected.len());
    for (name, len) in &expected {
        let entry = entries
            .get(name)
            .unwrap_or_else(|| panic!("missing entry for {name}"));
        assert_eq!(entry.length, *len, "wrong length for {name}");
    }
}

#[test]
fn check_data_header_footer_accepts_real_cfs() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let cfe_buf = std::fs::read(format!("{}{}.raw", dir(), manifest.get("cfe_file_name"))).unwrap();
    let cfs_buf = std::fs::read(format!("{}{}.raw", dir(), manifest.get("cfs_file_name"))).unwrap();

    let entries = compound_format::parse_entries(&cfe_buf, &id).unwrap();
    compound_format::check_data_header_footer(&cfs_buf, &id, &entries).unwrap();
}

#[test]
fn open_input_returns_correct_bytes_for_every_real_sub_file() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let cfe_buf = std::fs::read(format!("{}{}.raw", dir(), manifest.get("cfe_file_name"))).unwrap();
    let cfs_buf = std::fs::read(format!("{}{}.raw", dir(), manifest.get("cfs_file_name"))).unwrap();

    let entries = compound_format::parse_entries(&cfe_buf, &id).unwrap();
    for (name, len) in expected_sub_files(&manifest) {
        let bytes = compound_format::open_input(&cfs_buf, &entries, &name).unwrap();
        assert_eq!(bytes.len() as i64, len, "wrong slice length for {name}");
    }
}

#[test]
fn wrong_segment_id_rejected() {
    let manifest = Manifest::load();
    let cfe_buf = std::fs::read(format!("{}{}.raw", dir(), manifest.get("cfe_file_name"))).unwrap();
    let wrong_id = [9u8; ID_LENGTH];
    assert!(compound_format::parse_entries(&cfe_buf, &wrong_id).is_err());
}
