//! Differential test against real `.dvm`/`.dvd` files written by an actual
//! IndexWriter: a single-valued SORTED field with repeated values ("banana",
//! "apple", "cherry", "apple", "banana") across 5 docs, so the terms
//! dictionary has 3 unique alphabetically-ordered terms and the ordinal
//! array has repeats. Regenerate with fixtures/src/GenSortedDocValues.java.

use lucene_codecs::{doc_values, field_infos, terms_dict};

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/sorted_dv_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenSortedDocValues)");
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

/// `Lucene90DocValuesFormat` is wrapped in a `PerFieldDocValuesFormat`,
/// which gives each format its own segment-suffix on top of the segment's
/// own (empty) suffix -- derive it from the real filename.
fn dv_suffix(manifest: &Manifest) -> String {
    let segment_name = manifest.get("segment_name");
    let name = manifest.get("dvm_file_name");
    name.strip_prefix(&format!("{segment_name}_"))
        .and_then(|s| s.strip_suffix(".dvm"))
        .unwrap_or_else(|| panic!("unexpected dvm file name shape: {name}"))
        .to_string()
}

#[test]
fn parses_real_sorted_dv_and_matches_lucene_ords_and_terms() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let fnm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("fnm_file_name"))).unwrap();
    let fis = field_infos::parse(&fnm, &id, "").unwrap();

    let dvm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvm_file_name"))).unwrap();
    let dvd = std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvd_file_name"))).unwrap();
    let suffix = dv_suffix(&manifest);

    let (_, parsed) = doc_values::parse_meta(&dvm, &id, &suffix, &fis).unwrap();
    let entry = parsed
        .sorted_entry(field_number(&manifest, "sorted"))
        .unwrap();

    let terms = terms_dict::decode_all_terms(&dvd, &entry.terms).unwrap();
    let expected_terms: Vec<Vec<u8>> = manifest
        .get("field.sorted.terms")
        .split(',')
        .map(|s| s.as_bytes().to_vec())
        .collect();
    assert_eq!(terms, expected_terms);

    let expected_ords: Vec<Option<i64>> = manifest
        .get("field.sorted.ords")
        .split(',')
        .map(|s| {
            if s == "NONE" {
                None
            } else {
                Some(s.parse().unwrap())
            }
        })
        .collect();
    for (doc, &want) in expected_ords.iter().enumerate() {
        let got = doc_values::sorted_ord(&dvd, entry, doc as i32).unwrap();
        assert_eq!(got, want, "doc {doc} ordinal");
        if let Some(ord) = got {
            let term = &terms[ord as usize];
            let want_term_text = manifest
                .get("field.sorted.terms")
                .split(',')
                .nth(ord as usize)
                .unwrap();
            assert_eq!(term, want_term_text.as_bytes(), "doc {doc} term");
        }
    }
}
