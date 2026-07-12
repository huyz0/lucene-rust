//! Differential test against real `.fdt`/`.fdx`/`.fdm` files written with
//! `Lucene104Codec.Mode.BEST_COMPRESSION` (DEFLATE with a preset dictionary,
//! `Lucene90StoredFieldsHighData` data codec) -- same document shape as
//! `stored_fields_fixtures.rs`'s `Mode.BEST_SPEED` fixture, but with a long
//! repetitive string field so the DEFLATE dictionary + multi-sub-block
//! decode path is actually exercised, not just a trivial single unit.
//! Regenerate with fixtures/src/GenStoredFieldsBestCompression.java.

use lucene_codecs::stored_fields::{self, FieldValue};

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/stored_fields_best_compression_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenStoredFieldsBestCompression)");
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

/// Parses one `name:type:value` entry from the manifest's `;`-joined field
/// list. `value` for the `string` type may itself contain `:` (the repeated
/// sentence includes none, but keep this robust), so split greedily into 3
/// parts only.
fn expected_value(entry: &str) -> (String, FieldValue) {
    let mut parts = entry.splitn(3, ':');
    let name = parts.next().unwrap().to_string();
    let ty = parts.next().unwrap();
    let value = parts.next().unwrap();
    let field_value = match ty {
        "string" => FieldValue::String(value.to_string()),
        "binary" => FieldValue::Binary(
            (0..value.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&value[i..i + 2], 16).unwrap())
                .collect(),
        ),
        "int" => FieldValue::Int(value.parse().unwrap()),
        "long" => FieldValue::Long(value.parse().unwrap()),
        "float" => FieldValue::Float(value.parse().unwrap()),
        "double" => FieldValue::Double(value.parse().unwrap()),
        other => panic!("unknown manifest field type {other}"),
    };
    (name, field_value)
}

#[test]
fn parses_real_best_compression_stored_fields_and_matches_lucene_values() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let fdt = std::fs::read(format!("{}{}.raw", dir(), manifest.get("fdt_file_name"))).unwrap();
    let fdx = std::fs::read(format!("{}{}.raw", dir(), manifest.get("fdx_file_name"))).unwrap();
    let fdm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("fdm_file_name"))).unwrap();

    let reader = stored_fields::open(&fdt, &fdx, &fdm, &id, "").unwrap();
    let max_doc: i32 = manifest.get("max_doc").parse().unwrap();
    assert_eq!(reader.max_doc(), max_doc);

    for doc_id in 0..max_doc {
        let expected_line = manifest.get(&format!("doc.{doc_id}.fields"));
        let expected: Vec<(String, FieldValue)> =
            expected_line.split(';').map(expected_value).collect();

        let doc = reader.document(doc_id).unwrap();
        assert_eq!(doc.fields.len(), expected.len(), "doc {doc_id} field count");

        let mut got_values: Vec<FieldValue> = doc.fields.iter().map(|f| f.value.clone()).collect();
        let mut want_values: Vec<FieldValue> = expected.into_iter().map(|(_, v)| v).collect();
        got_values.sort_by_key(|v| format!("{v:?}"));
        want_values.sort_by_key(|v| format!("{v:?}"));
        assert_eq!(got_values, want_values, "doc {doc_id} values");
    }
}
