//! Byte-identity differential for the term-dictionary writer
//! (`crate::blocktree_writer`, a port of `Lucene103BlockTreeTermsWriter` +
//! `TrieBuilder`): re-write four real Lucene 10.5.0 term dictionaries from
//! their own terms and require **the same `.tim`, `.tip` and `.tmd` bytes**.
//!
//! Why identity is the right bar for these four and not in general: every
//! field in them is `IndexOptions.DOCS` with one distinct term per document,
//! so every term is a singleton, `Lucene104PostingsWriter` writes no `.doc`
//! bytes at all, and each term's metadata depends only on its doc id and the
//! term before it. The only other inputs to the term dictionary are the term
//! list itself and the writer's own decisions -- block boundaries, floor
//! splits, non-leaf blocks, suffix compression (LZ4 and lowercase-ASCII),
//! trie node shapes and child-label strategies, the singleton-run stats
//! encoding and `encodeTerm`'s zigzag doc-id deltas. A port that makes any of
//! those decisions differently from Java produces different bytes here.
//!
//! The four fixtures were built for the *reader*, each to force a shape:
//! `blocktree_multilevel_index` (8000 random terms: non-leaf blocks, floor
//! splits, a `REVERSE_ARRAY` root), `blocktree_deep_nesting_index` (four-plus
//! trie levels), `blocktree_compressed_index` (LZ4-compressed suffixes) and
//! `blocktree_child_strategies_index` (one field per `ARRAY`/`BITS`/
//! `REVERSE_ARRAY` child strategy, several fields in one `.tmd`). The
//! deep-nesting one is written with Java's non-default block sizes (2/4),
//! which the port takes as [`BlockSizes`].
//!
//! The terms and doc ids are read back through this port's reader, which the
//! reader-side fixture tests already pin against the same bytes.
// Test code: see `docs/arithmetic-gate.md`'s "Test code" section.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::blocktree;
use lucene_codecs::field_infos::{self, IndexOptions};
use lucene_codecs::postings_writer::{self, BlockSizes, FieldPostingsInput, TermPostings};

fn dir(name: &str) -> String {
    format!("{}/../../fixtures/data/{name}/", env!("CARGO_MANIFEST_DIR"))
}

fn manifest(dir: &str) -> Vec<(String, String)> {
    std::fs::read_to_string(format!("{dir}manifest.properties"))
        .expect("fixture manifest")
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn get<'a>(m: &'a [(String, String)], key: &str) -> &'a str {
    m.iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .unwrap_or_else(|| panic!("manifest key {key} missing"))
}

fn raw(dir: &str, name: &str) -> Vec<u8> {
    std::fs::read(format!("{dir}{name}.raw")).unwrap_or_else(|_| panic!("missing {name}.raw"))
}

fn id_from_hex(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

/// Re-writes `fixture`'s term dictionary and compares it with Java's.
fn assert_rewrite_is_identical(fixture: &str, block_sizes: BlockSizes) {
    let dir = dir(fixture);
    let m = manifest(&dir);
    let id = id_from_hex(get(&m, "id_hex"));
    let suffix = get(&m, "segment_suffix");
    let max_doc: i32 = get(&m, "max_doc").parse().unwrap();
    let infos = field_infos::parse(&raw(&dir, get(&m, "fnm_file_name")), &id, "").unwrap();
    let java_tim = raw(&dir, get(&m, "tim_file_name"));
    let java_tip = raw(&dir, get(&m, "tip_file_name"));
    let java_tmd = raw(&dir, get(&m, "tmd_file_name"));
    let fields = blocktree::open(
        &java_tim, &java_tip, &java_tmd, &infos, &id, suffix, max_doc,
    )
    .unwrap();

    // `FieldsConsumer.write` visits fields in name order.
    let mut names: Vec<&str> = fields.iter_fields().map(|(name, _)| name).collect();
    names.sort_unstable();
    let mut per_field: Vec<(i32, i32, Vec<TermPostings>)> = Vec::new();
    for name in names {
        let info = infos.field_by_name(name).unwrap();
        assert_eq!(info.index_options, IndexOptions::Docs, "{fixture}/{name}");
        let terms = fields.field(name).unwrap();
        let mut out = Vec::new();
        let mut it = terms.iter();
        while let Some((term, stats)) = it.next() {
            assert_eq!(
                stats.doc_freq, 1,
                "{fixture}/{name}: identity needs singletons"
            );
            let term = term.to_vec();
            let meta = terms.term_metadata(&term).unwrap().unwrap();
            out.push(TermPostings {
                term,
                docs: vec![(meta.singleton_doc_id, 1)],
                ..TermPostings::default()
            });
        }
        per_field.push((info.number, terms.doc_count, out));
    }
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
    let out =
        postings_writer::write_fields_with_block_sizes(&inputs, &[], block_sizes, &id, suffix)
            .unwrap();

    assert_same(fixture, ".tip", &out.tip, &java_tip);
    assert_same(fixture, ".tim", &out.tim, &java_tim);
    assert_same(fixture, ".tmd", &out.tmd, &java_tmd);
}

fn assert_same(fixture: &str, ext: &str, ours: &[u8], java: &[u8]) {
    if ours != java {
        let at = ours
            .iter()
            .zip(java)
            .position(|(a, b)| a != b)
            .unwrap_or(ours.len().min(java.len()));
        panic!(
            "{fixture}{ext}: {} bytes written, Java wrote {}; first difference at byte {at}: \
             ours {:02x?} vs Java {:02x?}",
            ours.len(),
            java.len(),
            &ours[at..(at + 16).min(ours.len())],
            &java[at..(at + 16).min(java.len())],
        );
    }
}

#[test]
fn multilevel_term_dictionary_is_byte_identical() {
    assert_rewrite_is_identical("blocktree_multilevel_index", BlockSizes::default());
}

#[test]
fn deep_nesting_term_dictionary_is_byte_identical() {
    // `GenBlockTreeDeepNesting` builds with
    // `Lucene104PostingsFormat(2, 4)` to force its depth.
    assert_rewrite_is_identical(
        "blocktree_deep_nesting_index",
        BlockSizes {
            min_items_in_block: 2,
            max_items_in_block: 4,
        },
    );
}

#[test]
fn lz4_compressed_term_dictionary_is_byte_identical() {
    assert_rewrite_is_identical("blocktree_compressed_index", BlockSizes::default());
}

#[test]
fn child_strategy_term_dictionaries_are_byte_identical() {
    assert_rewrite_is_identical("blocktree_child_strategies_index", BlockSizes::default());
}
