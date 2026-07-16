//! Differential test against a real `.dvm`/`.dvd` pair whose single NUMERIC
//! field trips `Lucene90DocValuesConsumer.writeValues`'s `doBlocks`
//! varying-bits-per-value split: two full 16384-value blocks with very
//! different value ranges (so per-block widths differ sharply from the
//! whole-field width) plus a trailing partial block.
//! Regenerate with `fixtures/src/GenDocValuesVaryingBpv.java`.

use lucene_codecs::{doc_values as ndv, field_infos};

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/doc_values_varying_bpv/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenDocValuesVaryingBpv)");
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
/// gives each format instance its own segment-suffix on top of the
/// segment's own (empty) suffix -- derive it from the real filename rather
/// than hardcoding it, since the counter can vary.
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

#[test]
fn parses_real_varying_bpv_numeric_dv_and_matches_lucene_values() {
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

    let entry = parsed
        .numeric_entry(field_number(&manifest, "varying_bpv"))
        .unwrap();
    assert!(entry.is_dense());
    // The whole point of this fixture: confirm the writer actually took the
    // `doBlocks` path rather than silently falling back to a single width.
    assert!(
        entry.block_shift.is_some(),
        "fixture's field didn't trip doBlocks -- adjust GenDocValuesVaryingBpv's value shape"
    );

    let expected: Vec<Option<i64>> = manifest
        .get("field.varying_bpv.values")
        .split(',')
        .map(|s| {
            if s == "NONE" {
                None
            } else {
                Some(s.parse().unwrap())
            }
        })
        .collect();

    let max_doc: usize = manifest.get("max_doc").parse().unwrap();
    assert_eq!(expected.len(), max_doc);

    for (doc, &want) in expected.iter().enumerate() {
        let got = ndv::numeric_value(&data_buf, entry, doc as i32).unwrap();
        assert_eq!(got, want, "doc {doc}");
    }
}
