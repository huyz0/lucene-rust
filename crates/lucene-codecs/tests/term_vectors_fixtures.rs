//! Differential test against real `.tvd`/`.tvx`/`.tvm` files written by an
//! actual IndexWriter: three docs -- doc 0 has one field with repeated terms
//! (exercising same-term multi-occurrence position/offset delta chains) and
//! payloads on some occurrences but not others; doc 1 has two fields
//! (exercising the distinct-field-numbers array and multi-field-per-doc
//! offsets); doc 2 has no term-vector field at all. Regenerate with
//! fixtures/src/GenTermVectors.java.

use lucene_codecs::term_vectors;

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/term_vectors_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenTermVectors)");
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

    fn try_get(&self, key: &str) -> Option<&str> {
        self.kv
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

fn id_from_hex(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for i in 0..16 {
        id[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

/// One expected occurrence: (position, start_offset, end_offset, payload_hex_or_none).
struct ExpectedOccurrence {
    position: i32,
    start_offset: i32,
    end_offset: i32,
    payload: Option<Vec<u8>>,
}

struct ExpectedTerm {
    term: String,
    freq: i32,
    occurrences: Vec<ExpectedOccurrence>,
}

fn parse_expected_terms(spec: &str) -> Vec<ExpectedTerm> {
    spec.split(';')
        .map(|term_spec| {
            let mut parts = term_spec.split(':');
            let term = parts.next().unwrap().to_string();
            let freq: i32 = parts.next().unwrap().parse().unwrap();
            let occurrences = parts
                .map(|occ| {
                    let fields: Vec<&str> = occ.split(',').collect();
                    ExpectedOccurrence {
                        position: fields[0].parse().unwrap(),
                        start_offset: fields[1].parse().unwrap(),
                        end_offset: fields[2].parse().unwrap(),
                        payload: if fields[3] == "NONE" {
                            None
                        } else {
                            Some(
                                (0..fields[3].len())
                                    .step_by(2)
                                    .map(|i| u8::from_str_radix(&fields[3][i..i + 2], 16).unwrap())
                                    .collect(),
                            )
                        },
                    }
                })
                .collect();
            ExpectedTerm {
                term,
                freq,
                occurrences,
            }
        })
        .collect()
}

#[test]
fn parses_real_term_vectors_and_matches_lucene_positions_offsets_payloads() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let max_doc: i32 = manifest.get("max_doc").parse().unwrap();

    let tvd = std::fs::read(format!("{}{}.raw", dir(), manifest.get("tvd_file_name"))).unwrap();
    let tvx = std::fs::read(format!("{}{}.raw", dir(), manifest.get("tvx_file_name"))).unwrap();
    let tvm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("tvm_file_name"))).unwrap();

    let reader = term_vectors::open(&tvd, &tvx, &tvm, &id, "").unwrap();
    assert_eq!(reader.max_doc(), max_doc);

    for doc in 0..max_doc {
        let expected_fields_key = format!("doc.{doc}.fields");
        let expected_fields = manifest.get(&expected_fields_key);

        let got = reader.document(doc).unwrap();
        if expected_fields == "NONE" {
            assert!(got.is_none(), "doc {doc} expected no term vectors");
            continue;
        }

        let doc_vectors = got.unwrap_or_else(|| panic!("doc {doc} expected term vectors"));
        let expected_field_names: Vec<&str> = expected_fields.split(',').collect();
        assert_eq!(
            doc_vectors.fields.len(),
            expected_field_names.len(),
            "doc {doc} field count"
        );

        for (field_idx, field_name) in expected_field_names.iter().enumerate() {
            let field = &doc_vectors.fields[field_idx];
            let terms_key = format!("doc.{doc}.field.{field_name}.terms");
            let expected_terms = parse_expected_terms(manifest.get(&terms_key));

            assert_eq!(
                field.terms.len(),
                expected_terms.len(),
                "doc {doc} field {field_name} term count"
            );

            for (term_idx, expected) in expected_terms.iter().enumerate() {
                let got_term = &field.terms[term_idx];
                assert_eq!(
                    got_term.term,
                    expected.term.as_bytes(),
                    "doc {doc} field {field_name} term {term_idx} text"
                );
                assert_eq!(
                    got_term.freq, expected.freq,
                    "doc {doc} field {field_name} term {term_idx} freq"
                );

                for (occ_idx, occ) in expected.occurrences.iter().enumerate() {
                    if let Some(positions) = &got_term.positions {
                        assert_eq!(
                            positions[occ_idx], occ.position,
                            "doc {doc} field {field_name} term {term_idx} occ {occ_idx} position"
                        );
                    }
                    if let (Some(starts), Some(ends)) =
                        (&got_term.start_offsets, &got_term.end_offsets)
                    {
                        assert_eq!(
                            starts[occ_idx], occ.start_offset,
                            "doc {doc} field {field_name} term {term_idx} occ {occ_idx} start_offset"
                        );
                        assert_eq!(
                            ends[occ_idx], occ.end_offset,
                            "doc {doc} field {field_name} term {term_idx} occ {occ_idx} end_offset"
                        );
                    }
                    if let Some(payloads) = &got_term.payloads {
                        let want: Vec<u8> = occ.payload.clone().unwrap_or_default();
                        assert_eq!(
                            payloads[occ_idx], want,
                            "doc {doc} field {field_name} term {term_idx} occ {occ_idx} payload"
                        );
                    }
                }
            }
        }
    }

    // Sanity: manifest actually asserted the "no term vectors" path exists.
    assert!(manifest.try_get("doc.2.fields").is_some());
}
