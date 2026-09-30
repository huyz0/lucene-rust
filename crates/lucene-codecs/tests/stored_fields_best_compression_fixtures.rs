//! Differential test against real `.fdt`/`.fdx`/`.fdm` files written with
//! `Lucene104Codec.Mode.BEST_COMPRESSION` (DEFLATE with a preset dictionary,
//! `Lucene90StoredFieldsHighData` data codec) -- same document shape as
//! `stored_fields_fixtures.rs`'s `Mode.BEST_SPEED` fixture, but with a long
//! repetitive string field so the DEFLATE dictionary + multi-sub-block
//! decode path is actually exercised, not just a trivial single unit.
//! Regenerate with fixtures/src/GenStoredFieldsBestCompression.java.
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

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

/// `GenStoredFieldsDeflate`'s documents, rebuilt from its LCG.
struct DeflateDocs {
    seed: u64,
}

impl DeflateDocs {
    const WORDS: [&'static str; 28] = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consectetur",
        "adipiscing",
        "elit",
        "sed",
        "do",
        "eiusmod",
        "tempor",
        "incididunt",
        "labore",
        "magna",
        "aliqua",
        "lucene",
        "rust",
        "segment",
        "merge",
    ];

    fn next(&mut self, bound: u64) -> u64 {
        self.seed = self
            .seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.seed >> 33) % bound
    }

    fn doc(&mut self, i: usize, number: &dyn Fn(&str) -> i32) -> stored_fields::Document {
        let mut fields = Vec::new();
        let kind = if i >= 497 { 0 } else { self.next(10) };
        let words = if kind == 0 {
            1 + self.next(3)
        } else if i.is_multiple_of(83) {
            5000 + self.next(2000)
        } else {
            20 + self.next(400)
        };
        let mut text = String::new();
        for _ in 0..words {
            text.push_str(Self::WORDS[self.next(Self::WORDS.len() as u64) as usize]);
            text.push_str(if self.next(5) == 0 { ". " } else { " " });
        }
        fields.push(stored_fields::StoredField {
            field_number: number("text"),
            value: FieldValue::String(text),
        });
        let blob_len = if i % 101 == 50 {
            20_000 + self.next(5000)
        } else {
            self.next(if kind == 1 { 1200 } else { 40 })
        };
        let blob: Vec<u8> = (0..blob_len).map(|_| self.next(256) as u8).collect();
        fields.push(stored_fields::StoredField {
            field_number: number("blob"),
            value: FieldValue::Binary(blob),
        });
        if kind == 2 {
            let len = self.next(2000) as usize;
            let byte = b'a' + self.next(3) as u8;
            fields.push(stored_fields::StoredField {
                field_number: number("run"),
                value: FieldValue::Binary(vec![byte; len]),
            });
        }
        fields.push(stored_fields::StoredField {
            field_number: number("num"),
            value: FieldValue::Int(self.next(1_000_000) as i32 - 500_000),
        });
        if i == 496 {
            let sentences: Vec<String> = (0..40)
                .map(|_| {
                    let mut s = String::new();
                    for _ in 0..12 {
                        s.push_str(Self::WORDS[self.next(Self::WORDS.len() as u64) as usize]);
                        s.push(' ');
                    }
                    s.push_str(". ");
                    s
                })
                .collect();
            let mut big = String::new();
            for _ in 0..9000 {
                big.push_str(&sentences[self.next(40) as usize]);
            }
            fields.push(stored_fields::StoredField {
                field_number: number("big"),
                value: FieldValue::String(big),
            });
        }
        stored_fields::Document { fields }
    }
}

/// The write side, byte for byte: `GenStoredFieldsDeflate`'s documents --
/// text, random bytes, runs, tiny and window-sliding documents over several
/// chunks -- written with `BEST_COMPRESSION` produce Lucene's `.fdt`, `.fdx`
/// and `.fdm`. Every DEFLATE unit is the zlib output Java's `Deflater`
/// produced, preset dictionary included.
#[test]
fn best_compression_is_written_byte_identical_to_lucene() {
    let base = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/stored_fields_deflate_index/"
    );
    let manifest = std::fs::read_to_string(format!("{base}manifest.properties"))
        .expect("run the fixtures generator first (GenStoredFieldsDeflate)");
    let get = |key: &str| -> String {
        manifest
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
            .to_string()
    };
    let hex = get("id_hex");
    let mut id = [0u8; 16];
    for (i, b) in id.iter_mut().enumerate() {
        *b = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    let number = |name: &str| -> i32 { get(&format!("field.{name}")).parse().unwrap() };
    let num_docs: usize = get("num_docs").parse().unwrap();
    let mut gen = DeflateDocs { seed: 20_260_930 };
    let docs: Vec<stored_fields::Document> = (0..num_docs).map(|i| gen.doc(i, &number)).collect();

    let (fdt, fdx, fdm) = stored_fields::write_best_compression(&docs, &id, "");
    let raw = |key: &str| std::fs::read(format!("{base}{}.raw", get(key))).unwrap();
    let want_fdt = raw("fdt_file_name");
    let at = fdt
        .iter()
        .zip(&want_fdt)
        .position(|(a, b)| a != b)
        .unwrap_or(fdt.len().min(want_fdt.len()));
    assert!(
        fdt == want_fdt,
        ".fdt differs at byte {at} (port {} bytes, Lucene {})",
        fdt.len(),
        want_fdt.len()
    );
    assert_eq!(fdx, raw("fdx_file_name"), ".fdx");
    assert_eq!(fdm, raw("fdm_file_name"), ".fdm");
}
