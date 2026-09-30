//! The write side of one postings format of a `PerFieldPostingsFormat`
//! segment, byte for byte: `fixtures/data/per_field_formats_index/`
//! (`GenPerFieldFormats`) routes its `b_*` fields to
//! `Lucene104PostingsFormat(10, 20)` -- suffix `Lucene104_1`, smaller
//! blocks, and an empty `.pos` of its own because the *segment* has
//! positions (`a_text`). Its postings are read back through this port's
//! reader and handed to `postings_writer::write_fields_with_options` with
//! the same block sizes and `.pos`/`.pay` presence; `.tim`, `.tip`, `.tmd`,
//! `.doc` and `.psm` must be Lucene's.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::blocktree;
use lucene_codecs::field_infos::{self, IndexOptions};
use lucene_codecs::postings::DocInput;
use lucene_codecs::postings_writer::{self, FieldPostingsInput, TermPostings, WriteOptions};

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/per_field_formats_index/"
    )
    .to_string()
}

fn get(key: &str) -> String {
    std::fs::read_to_string(format!("{}manifest.properties", dir()))
        .expect("run the fixtures generator first (GenPerFieldFormats)")
        .lines()
        .find_map(|l| l.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("manifest key {key} missing"))
        .to_string()
}

fn read(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}{name}", dir())).unwrap_or_else(|_| panic!("missing {name}"))
}

#[test]
fn the_second_postings_format_is_written_byte_identical_to_lucene() {
    let hex = get("id_hex");
    let mut id = [0u8; 16];
    for (i, b) in id.iter_mut().enumerate() {
        *b = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    let seg = get("segment_name");
    let suffix = "Lucene104_1";
    let infos = field_infos::parse(&read(&format!("{seg}.fnm")), &id, "").unwrap();
    let file = |ext: &str| read(&format!("{seg}_{suffix}.{ext}"));
    let fields = blocktree::open(
        &file("tim"),
        &file("tip"),
        &file("tmd"),
        &infos,
        &id,
        suffix,
        get("max_doc").parse().unwrap(),
    )
    .unwrap();
    let doc_bytes = file("doc");
    let doc_in = DocInput::open(&doc_bytes, &id, suffix).unwrap();

    // Every field of this format, in `.tmd` order, with every posting.
    let mut per_field: Vec<(i32, i32, Vec<TermPostings>)> = Vec::new();
    for (name, terms) in fields.iter_fields() {
        let info = infos.fields.iter().find(|f| f.name == name).unwrap();
        assert_eq!(info.index_options, IndexOptions::Docs, "{name}");
        let mut out = Vec::new();
        let mut it = terms.iter();
        let mut all: Vec<Vec<u8>> = Vec::new();
        while let Some((term, _)) = it.try_next().unwrap() {
            all.push(term.to_vec());
        }
        for term in all {
            let p = terms.postings(&term, Some(&doc_in)).unwrap().unwrap();
            out.push(TermPostings {
                term,
                docs: p.docs.iter().zip(&p.freqs).map(|(&d, &f)| (d, f)).collect(),
                ..Default::default()
            });
        }
        per_field.push((info.number, terms.doc_count, out));
    }
    assert_eq!(per_field.len(), 2, "b_id and b_tag");
    let inputs: Vec<FieldPostingsInput<'_>> = per_field
        .iter()
        .map(|(number, doc_count, terms)| FieldPostingsInput {
            field_number: *number,
            index_options: IndexOptions::Docs,
            doc_count: *doc_count,
            has_payloads: false,
            terms,
        })
        .collect();
    let options = WriteOptions {
        min_items_in_block: 10,
        max_items_in_block: 20,
        has_prox: Some(true),
        has_payloads_or_offsets: Some(false),
    };
    let out =
        postings_writer::write_fields_with_options(&inputs, &[], &options, &id, suffix).unwrap();
    for (ext, got) in [
        ("doc", &out.doc),
        ("tim", &out.tim),
        ("tip", &out.tip),
        ("tmd", &out.tmd),
        ("psm", &out.psm),
        ("pos", &out.pos),
    ] {
        let want = file(ext);
        assert!(
            *got == want,
            "{ext}: {} bytes vs Lucene's {}",
            got.len(),
            want.len()
        );
    }

    // The default block sizes cut different blocks: the options matter.
    let default = postings_writer::write_fields_with_options(
        &inputs,
        &[],
        &WriteOptions {
            has_prox: Some(true),
            has_payloads_or_offsets: Some(false),
            ..WriteOptions::default()
        },
        &id,
        suffix,
    )
    .unwrap();
    assert_ne!(default.tim, file("tim"));
}

#[test]
fn block_sizes_are_validated_as_java_validates_them() {
    for (min, max, ok) in [
        (25, 48, true),
        (10, 20, true),
        (1, 48, false),
        (30, 20, false),
        (25, 47, false),
    ] {
        let options = WriteOptions {
            min_items_in_block: min,
            max_items_in_block: max,
            ..WriteOptions::default()
        };
        assert_eq!(options.validate().is_ok(), ok, "({min}, {max})");
    }
}
