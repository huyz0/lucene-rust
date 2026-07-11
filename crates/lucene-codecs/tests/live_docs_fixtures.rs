//! Differential test against a real `.liv` file written by an actual IndexWriter
//! (5 docs, 2 deleted by term, NoMergePolicy so the segment isn't merged away).
//! Regenerate with fixtures/src/GenLiveDocs.java.

use lucene_codecs::live_docs;

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/live_docs_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenLiveDocs)");
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

    fn get_usize(&self, key: &str) -> usize {
        self.get(key).parse().unwrap()
    }

    fn get_i64(&self, key: &str) -> i64 {
        self.get(key).parse().unwrap()
    }

    fn get_ids(&self, key: &str) -> Vec<usize> {
        self.get(key)
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect()
    }
}

fn id_from_hex(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for i in 0..16 {
        id[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

#[test]
fn parses_real_liv_file_and_matches_deletions() {
    let manifest = Manifest::load();
    let liv_file_name = manifest.get("liv_file_name");
    let id = id_from_hex(manifest.get("id_hex"));
    let del_gen = manifest.get_i64("del_gen");
    let max_doc = manifest.get_usize("max_doc");
    let del_count = manifest.get_usize("del_count");

    let buf = std::fs::read(format!("{}{}.raw", dir(), liv_file_name)).unwrap();
    let live_docs = live_docs::parse(&buf, &id, del_gen, max_doc, del_count).unwrap();

    assert_eq!(live_docs.len(), max_doc);
    assert_eq!(max_doc - live_docs.cardinality(), del_count);

    for doc in manifest.get_ids("live_doc_ids") {
        assert!(live_docs.get(doc), "doc {doc} should be live");
    }
    for doc in manifest.get_ids("deleted_doc_ids") {
        assert!(!live_docs.get(doc), "doc {doc} should be deleted");
    }
}

#[test]
fn wrong_segment_id_rejected() {
    let manifest = Manifest::load();
    let liv_file_name = manifest.get("liv_file_name");
    let del_gen = manifest.get_i64("del_gen");
    let max_doc = manifest.get_usize("max_doc");
    let del_count = manifest.get_usize("del_count");

    let buf = std::fs::read(format!("{}{}.raw", dir(), liv_file_name)).unwrap();
    let wrong_id = [0u8; 16];
    assert!(live_docs::parse(&buf, &wrong_id, del_gen, max_doc, del_count).is_err());
}

#[test]
fn wrong_del_count_rejected() {
    let manifest = Manifest::load();
    let liv_file_name = manifest.get("liv_file_name");
    let id = id_from_hex(manifest.get("id_hex"));
    let del_gen = manifest.get_i64("del_gen");
    let max_doc = manifest.get_usize("max_doc");

    let buf = std::fs::read(format!("{}{}.raw", dir(), liv_file_name)).unwrap();
    assert!(live_docs::parse(&buf, &id, del_gen, max_doc, 999).is_err());
}
