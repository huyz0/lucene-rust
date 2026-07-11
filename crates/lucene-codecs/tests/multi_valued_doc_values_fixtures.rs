//! Differential test against real `.dvm`/`.dvd` files written by an actual
//! IndexWriter: a SORTED_NUMERIC field ("nums", 0-3 values per doc) and a
//! SORTED_SET field ("tags", 0-2 values per doc sharing a terms
//! dictionary), across 5 docs -- some with zero values (IndexedDISI-sparse
//! path) and others with more than one (DirectMonotonicReader address-range
//! path). Regenerate with fixtures/src/GenMultiValuedDocValues.java.

use lucene_codecs::doc_values::{self, SortedSetKind};
use lucene_codecs::{field_infos, terms_dict};

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/multi_valued_dv_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenMultiValuedDocValues)");
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

fn dv_suffix(manifest: &Manifest) -> String {
    let segment_name = manifest.get("segment_name");
    let name = manifest.get("dvm_file_name");
    name.strip_prefix(&format!("{segment_name}_"))
        .and_then(|s| s.strip_suffix(".dvm"))
        .unwrap_or_else(|| panic!("unexpected dvm file name shape: {name}"))
        .to_string()
}

fn parse_multi_value_list(s: &str) -> Option<Vec<i64>> {
    if s == "NONE" {
        None
    } else {
        Some(s.split(',').map(|v| v.parse().unwrap()).collect())
    }
}

#[test]
fn parses_real_sorted_numeric_and_matches_lucene_values() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let fnm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("fnm_file_name"))).unwrap();
    let fis = field_infos::parse(&fnm, &id, "").unwrap();
    let dvm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvm_file_name"))).unwrap();
    let dvd = std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvd_file_name"))).unwrap();
    let suffix = dv_suffix(&manifest);

    let (_, parsed) = doc_values::parse_meta(&dvm, &id, &suffix, &fis).unwrap();
    let entry = parsed
        .sorted_numeric_entry(field_number(&manifest, "nums"))
        .unwrap();

    let max_doc: i32 = manifest.get("max_doc").parse().unwrap();
    let expected: Vec<Option<Vec<i64>>> = manifest
        .get("field.nums.values")
        .split(';')
        .map(parse_multi_value_list)
        .collect();
    assert_eq!(expected.len(), max_doc as usize);

    for (doc, want) in expected.iter().enumerate() {
        let got = doc_values::sorted_numeric_values(&dvd, entry, doc as i32).unwrap();
        let got = if got.is_empty() { None } else { Some(got) };
        assert_eq!(got, *want, "doc {doc}");
    }
}

#[test]
fn parses_real_sorted_set_and_matches_lucene_ords_and_terms() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let fnm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("fnm_file_name"))).unwrap();
    let fis = field_infos::parse(&fnm, &id, "").unwrap();
    let dvm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvm_file_name"))).unwrap();
    let dvd = std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvd_file_name"))).unwrap();
    let suffix = dv_suffix(&manifest);

    let (_, parsed) = doc_values::parse_meta(&dvm, &id, &suffix, &fis).unwrap();
    let entry = parsed
        .sorted_set_entry(field_number(&manifest, "tags"))
        .unwrap();
    let (ords_entry, terms_entry) = match &entry.kind {
        SortedSetKind::Multi { ords, terms } => (ords, terms),
        SortedSetKind::Single(_) => panic!("expected a true multi-valued SORTED_SET field"),
    };

    let terms = terms_dict::decode_all_terms(&dvd, terms_entry).unwrap();
    let expected_terms: Vec<Vec<u8>> = manifest
        .get("field.tags.terms")
        .split(',')
        .map(|s| s.as_bytes().to_vec())
        .collect();
    assert_eq!(terms, expected_terms);

    let max_doc: i32 = manifest.get("max_doc").parse().unwrap();
    let expected: Vec<Option<Vec<i64>>> = manifest
        .get("field.tags.ords")
        .split(';')
        .map(parse_multi_value_list)
        .collect();
    assert_eq!(expected.len(), max_doc as usize);

    for (doc, want) in expected.iter().enumerate() {
        let got = doc_values::sorted_numeric_values(&dvd, ords_entry, doc as i32).unwrap();
        let got = if got.is_empty() { None } else { Some(got) };
        assert_eq!(got, *want, "doc {doc} ords");
    }
}
