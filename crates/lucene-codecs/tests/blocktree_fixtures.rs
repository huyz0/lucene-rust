//! Differential test against a real `.tim`/`.tip`/`.tmd` triple written by an
//! actual `IndexWriter` (`Lucene104PostingsFormat` -> `Lucene103BlockTreeTermsWriter`):
//! two fields, "body" (`IndexOptions.DOCS_AND_FREQS`, repeated terms across
//! five docs with known per-term frequencies, one doc missing the field
//! entirely) and "id" (`IndexOptions.DOCS`, one distinct token per doc,
//! exercising the DOCS-only sumDocFreq/sumTotalTermFreq aliasing path).
//! Both fields fit in a single non-floor leaf `.tim` block (well under the
//! default 25/48 min/maxItemsInBlock thresholds), which is this slice's
//! scope -- see `crates/lucene-codecs/src/blocktree.rs`'s module doc.
//! Regenerate with `fixtures/src/GenBlockTree.java`.

use lucene_codecs::blocktree;
use lucene_codecs::field_infos;
use lucene_codecs::postings;

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenBlockTree)");
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
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

fn read_raw(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}{}.raw", dir(), name)).unwrap_or_else(|_| panic!("missing {name}.raw"))
}

fn open_fixture() -> (blocktree::BlockTreeFields, Manifest) {
    let m = Manifest::load();
    let id = id_from_hex(m.get("id_hex"));
    let suffix = m.get("segment_suffix").to_string();
    let max_doc: i32 = m.get("max_doc").parse().unwrap();

    let fnm = read_raw(m.get("fnm_file_name"));
    let field_infos = field_infos::parse(&fnm, &id, "").expect("parse .fnm");

    let tim = read_raw(m.get("tim_file_name"));
    let tip = read_raw(m.get("tip_file_name"));
    let tmd = read_raw(m.get("tmd_file_name"));

    let fields = blocktree::open(&tim, &tip, &tmd, &field_infos, &id, &suffix, max_doc)
        .expect("open blocktree");
    (fields, m)
}

#[test]
fn field_level_stats_match_real_lucene() {
    let (fields, m) = open_fixture();

    for field_name in ["body", "id"] {
        let field = fields.field(field_name).unwrap_or_else(|| {
            panic!("expected field {field_name} to be present");
        });
        let num_terms: i64 = m
            .get(&format!("field.{field_name}.numTerms"))
            .parse()
            .unwrap();
        let sum_doc_freq: i64 = m
            .get(&format!("field.{field_name}.sumDocFreq"))
            .parse()
            .unwrap();
        let sum_total_term_freq: i64 = m
            .get(&format!("field.{field_name}.sumTotalTermFreq"))
            .parse()
            .unwrap();
        let doc_count: i32 = m
            .get(&format!("field.{field_name}.docCount"))
            .parse()
            .unwrap();
        let min_term = m.get(&format!("field.{field_name}.minTerm"));
        let max_term = m.get(&format!("field.{field_name}.maxTerm"));

        assert_eq!(field.num_terms, num_terms, "field={field_name}");
        assert_eq!(field.sum_doc_freq, sum_doc_freq, "field={field_name}");
        assert_eq!(
            field.sum_total_term_freq, sum_total_term_freq,
            "field={field_name}"
        );
        assert_eq!(field.doc_count, doc_count, "field={field_name}");
        assert_eq!(field.min_term, min_term.as_bytes(), "field={field_name}");
        assert_eq!(field.max_term, max_term.as_bytes(), "field={field_name}");
    }
}

#[test]
fn body_field_term_lookups_match_real_lucene() {
    let (fields, _m) = open_fixture();
    let body = fields.field("body").unwrap();

    let cases: &[(&str, Option<(i32, i64)>)] = &[
        ("cat", Some((2, 3))),
        ("dog", Some((2, 2))),
        ("bird", Some((2, 4))),
        ("zzz-missing", None),
        ("", None),
        ("ca", None),
    ];
    for (term, expected) in cases {
        let got = body.seek_exact(term.as_bytes());
        match expected {
            Some((doc_freq, total_term_freq)) => {
                let stats = got.unwrap_or_else(|| panic!("expected term {term:?} to be found"));
                assert_eq!(stats.doc_freq, *doc_freq, "term={term:?}");
                assert_eq!(stats.total_term_freq, *total_term_freq, "term={term:?}");
            }
            None => assert!(got.is_none(), "expected term {term:?} to be absent"),
        }
    }
}

#[test]
fn id_field_docs_only_term_lookups_match_real_lucene() {
    let (fields, _m) = open_fixture();
    let id_field = fields.field("id").unwrap();

    for i in 0..5 {
        let term = format!("id{i}");
        let stats = id_field
            .seek_exact(term.as_bytes())
            .unwrap_or_else(|| panic!("expected term {term:?} to be found"));
        assert_eq!(stats.doc_freq, 1);
        assert_eq!(stats.total_term_freq, 1);
    }
    assert!(id_field.seek_exact(b"id5-missing").is_none());
}

#[test]
fn missing_field_returns_none() {
    let (fields, _m) = open_fixture();
    assert!(fields.field("nonexistent").is_none());
}

fn open_doc_input(m: &Manifest) -> (Vec<u8>, [u8; 16], String) {
    let id = id_from_hex(m.get("id_hex"));
    let suffix = m.get("segment_suffix").to_string();
    let doc = read_raw(m.get("doc_file_name"));
    (doc, id, suffix)
}

/// `body` (`IndexOptions.DOCS_AND_FREQS`, `docFreq == 2` for every term):
/// exercises the multi-doc (`.doc` group-varint) postings decode path against
/// real `PostingsEnum.nextDoc()`/`freq()` output.
#[test]
fn body_field_postings_match_real_lucene_postings_enum() {
    let (fields, m) = open_fixture();
    let (doc, id, suffix) = open_doc_input(&m);
    let doc_in = postings::DocInput::open(&doc, &id, &suffix).expect("open .doc");
    let body = fields.field("body").unwrap();

    for term in ["cat", "dog", "bird"] {
        let expected_docs: Vec<i32> = m
            .get(&format!("field.body.term.{term}.postingsDocs"))
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let expected_freqs: Vec<i32> = m
            .get(&format!("field.body.term.{term}.postingsFreqs"))
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();

        let postings = body
            .postings(term.as_bytes(), Some(&doc_in))
            .unwrap_or_else(|e| panic!("postings({term:?}) failed: {e}"))
            .unwrap_or_else(|| panic!("expected term {term:?} to be found"));
        assert_eq!(postings.docs, expected_docs, "term={term:?}");
        assert_eq!(postings.freqs, expected_freqs, "term={term:?}");
    }
}

/// `id` (`IndexOptions.DOCS`, `docFreq == 1` for every term): exercises the
/// singleton path, which never touches the `.doc` file at all.
#[test]
fn id_field_postings_match_real_lucene_postings_enum() {
    let (fields, m) = open_fixture();
    let id_field = fields.field("id").unwrap();

    for i in 0..5 {
        let term = format!("id{i}");
        let expected_docs: Vec<i32> = m
            .get(&format!("field.id.term.{term}.postingsDocs"))
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let expected_freqs: Vec<i32> = m
            .get(&format!("field.id.term.{term}.postingsFreqs"))
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();

        // No .doc file needed: docFreq == 1 is reconstructed purely from
        // term-dictionary metadata (see postings::singleton_postings).
        let postings = id_field
            .postings(term.as_bytes(), None)
            .unwrap_or_else(|e| panic!("postings({term:?}) failed: {e}"))
            .unwrap_or_else(|| panic!("expected term {term:?} to be found"));
        assert_eq!(postings.docs, expected_docs, "term={term:?}");
        assert_eq!(postings.freqs, expected_freqs, "term={term:?}");
    }
}

/// `big` (`IndexOptions.DOCS_AND_FREQS`, `docFreq == 300`): exercises the
/// multi-block `.doc` decode path added on top of the single-block
/// group-varint path above -- one full 256-doc `ForUtil`/`PForUtil`-encoded
/// block followed by a 44-doc group-varint tail block, against real
/// `PostingsEnum.nextDoc()`/`freq()` output end to end.
#[test]
fn big_field_multi_block_postings_match_real_lucene_postings_enum() {
    let (fields, m) = open_fixture();
    let (doc, id, suffix) = open_doc_input(&m);
    let doc_in = postings::DocInput::open(&doc, &id, &suffix).expect("open .doc");
    let big = fields.field("big").unwrap();

    let expected_docs: Vec<i32> = m
        .get("field.big.term.everywhere.postingsDocs")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let expected_freqs: Vec<i32> = m
        .get("field.big.term.everywhere.postingsFreqs")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(
        expected_docs.len(),
        300,
        "fixture sanity: expected docFreq 300"
    );

    let postings = big
        .postings(b"everywhere", Some(&doc_in))
        .unwrap_or_else(|e| panic!("postings(\"everywhere\") failed: {e}"))
        .unwrap_or_else(|| panic!("expected term \"everywhere\" to be found"));
    assert_eq!(postings.docs, expected_docs);
    assert_eq!(postings.freqs, expected_freqs);
}

/// `pos` (`IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS`,
/// `hasPayloads` on some occurrences but not others): exercises
/// `postings::read_positions`/`FieldTerms::positions` against real
/// `PostingsEnum.nextPosition()`/`startOffset()`/`endOffset()`/`getPayload()`
/// output. Both terms fit entirely in the vint tail (`totalTermFreq <
/// BLOCK_SIZE`), so this doesn't exercise the full-`ForUtil`/`PForUtil`-block
/// path -- that's covered by `postings.rs`'s own hand-built unit test
/// (`read_positions_exactly_one_full_block_boundary`), since reaching it
/// with a real fixture would need 256+ real token occurrences.
#[test]
fn pos_field_positions_match_real_lucene_postings_enum() {
    let (fields, m) = open_fixture();
    let id = id_from_hex(m.get("id_hex"));
    let suffix = m.get("segment_suffix").to_string();
    let doc = read_raw(m.get("doc_file_name"));
    let pos = read_raw(m.get("pos_file_name"));
    let pay = read_raw(m.get("pay_file_name"));
    let doc_in = lucene_codecs::postings::DocInput::open(&doc, &id, &suffix).expect("open .doc");
    let pos_in = lucene_codecs::postings::PosInput::open(&pos, &id, &suffix).expect("open .pos");
    let pay_in = lucene_codecs::postings::PayInput::open(&pay, &id, &suffix).expect("open .pay");
    let field = fields.field("pos").unwrap();

    for term in ["alpha", "beta"] {
        let expected_docs: Vec<i32> = m
            .get(&format!("field.pos.term.{term}.postingsDocs"))
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let expected_occurrences: Vec<(i32, i32, i32, String)> = m
            .get(&format!("field.pos.term.{term}.occurrences"))
            .split(';')
            .map(|occ| {
                let parts: Vec<&str> = occ.split(',').collect();
                (
                    parts[0].parse().unwrap(),
                    parts[1].parse().unwrap(),
                    parts[2].parse().unwrap(),
                    parts[3].to_string(),
                )
            })
            .collect();

        let positions = field
            .positions(term.as_bytes(), Some(&doc_in), &pos_in, Some(&pay_in))
            .unwrap_or_else(|e| panic!("positions({term:?}) failed: {e}"))
            .unwrap_or_else(|| panic!("expected term {term:?} to be found"));
        assert_eq!(positions.len(), expected_docs.len(), "term={term:?}");

        let mut flat = Vec::new();
        for doc_positions in &positions {
            for p in doc_positions {
                let payload_hex = if p.payload.is_empty() {
                    "NONE".to_string()
                } else {
                    p.payload.iter().map(|b| format!("{b:02x}")).collect()
                };
                flat.push((p.position, p.start_offset, p.end_offset, payload_hex));
            }
        }
        assert_eq!(flat, expected_occurrences, "term={term:?}");
    }
}

#[test]
fn postings_missing_term_returns_none() {
    let (fields, m) = open_fixture();
    let (doc, id, suffix) = open_doc_input(&m);
    let doc_in = postings::DocInput::open(&doc, &id, &suffix).expect("open .doc");
    let body = fields.field("body").unwrap();
    assert!(body
        .postings(b"zzz-missing", Some(&doc_in))
        .unwrap()
        .is_none());
}
