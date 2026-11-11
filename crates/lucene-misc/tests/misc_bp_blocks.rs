//! `BPReorderingMergePolicy` over an index of document blocks with no parent
//! field, as OpenSearch writes nested documents without an index sort: the
//! blocks are found by their root documents' `_primary_term`
//! ([`BpIndexReorderer::set_block_root_field`]), the merge is reordered, and
//! every block stays whole, its root last.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_index::document::{Document, NumericDocValuesField, Store, StringField, TextField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::TieredMergePolicy;
use lucene_index::segment_info::LuceneVersion;
use lucene_misc::index::bp_merge_policy::REORDERED;
use lucene_misc::index::{BpIndexReorderer, BpReorderingMergePolicy};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::{LeafReader, StoredFieldVisitor, VisitStatus};
use lucene_store::ByteBuffersDirectory;

#[derive(Default)]
struct Block(Option<String>);

impl StoredFieldVisitor for Block {
    fn needs_field(&mut self, _n: i32) -> lucene_codecs::stored_fields::Result<VisitStatus> {
        Ok(VisitStatus::Yes)
    }
    fn string_field(&mut self, _n: i32, value: &str) -> lucene_codecs::stored_fields::Result<()> {
        self.0 = Some(value.to_string());
        Ok(())
    }
}

const WORDS: [&str; 6] = ["red", "green", "blue", "cyan", "magenta", "yellow"];

/// Block `b`: `b % 4` children, then its root; two kinds of text, so a
/// reordering moves blocks apart from their insertion order.
fn block(b: usize) -> Vec<Document> {
    let word = WORDS[(b * 7) % WORDS.len()];
    let mut docs = Vec::new();
    for c in 0..b % 4 {
        let mut d = Document::new();
        d.add(TextField::new(
            "body",
            format!("{word} child{c}"),
            Store::No,
        ));
        d.add(StringField::new("block", b.to_string(), Store::Yes));
        docs.push(d);
    }
    let mut root = Document::new();
    root.add(TextField::new("body", format!("{word} root"), Store::No));
    root.add(StringField::new("block", b.to_string(), Store::Yes));
    root.add(NumericDocValuesField::new("_primary_term", 1));
    docs.push(root);
    docs
}

fn merged(root_field: Option<&str>) -> DirectoryReader {
    merged_with(root_field, |_| {})
}

/// [`merged`], `end` adding documents to the end of each segment.
fn merged_with(root_field: Option<&str>, end: impl Fn(&mut IndexWriter<'_>)) -> DirectoryReader {
    let dir = ByteBuffersDirectory::new();
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
    let mut reorderer = BpIndexReorderer::new();
    reorderer.set_min_doc_freq(2).unwrap();
    reorderer.set_min_partition_size(2).unwrap();
    reorderer.set_fields(Some(vec!["body".to_string()]));
    reorderer.set_block_root_field(root_field.map(str::to_string));
    w.set_pluggable_merge_policy(Some(Arc::new(BpReorderingMergePolicy::new(
        Box::new(TieredMergePolicy::default()),
        reorderer,
    ))));
    for b in 0..120 {
        w.add_fields_documents(&block(b)).unwrap();
        if b % 40 == 39 {
            end(&mut w);
            w.commit().unwrap();
        }
    }
    w.force_merge(1).unwrap();
    w.close().unwrap();
    DirectoryReader::open(&dir).unwrap()
}

/// Each document's block, in segment order.
fn blocks(reader: &DirectoryReader) -> Vec<usize> {
    let seg = &reader.segment_readers()[0];
    (0..seg.max_doc)
        .map(|d| {
            let mut b = Block::default();
            seg.document(d, &mut b).unwrap();
            b.0.unwrap().parse().unwrap()
        })
        .collect()
}

#[test]
fn blocks_found_by_their_roots_are_reordered_whole() {
    let reader = merged(Some("_primary_term"));
    assert_eq!(reader.segment_readers().len(), 1);
    let seg = &reader.segment_readers()[0];
    assert!(
        seg.diagnostics()
            .iter()
            .any(|(k, v)| k == REORDERED && v == "true"),
        "{:?}",
        seg.diagnostics()
    );
    let order = blocks(&reader);
    // Whole blocks, each its own run of the right length.
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for b in &order {
        match runs.last_mut() {
            Some((last, n)) if last == b => *n += 1,
            _ => runs.push((*b, 1)),
        }
    }
    assert_eq!(runs.len(), 120, "a block split apart: {order:?}");
    for (b, n) in &runs {
        assert_eq!(*n, b % 4 + 1, "block {b}");
    }
    // Reordered: the blocks are no longer in insertion order.
    assert!(runs.windows(2).any(|w| w[0].0 > w[1].0), "{order:?}");
    // Each block's root (the one with `_primary_term`) is its last document.
    let mut values = seg.numeric_doc_values("_primary_term").unwrap().unwrap();
    let mut roots = Vec::new();
    loop {
        let d = values.next_doc().unwrap();
        if d == i32::MAX {
            break;
        }
        roots.push(d);
    }
    let mut end = 0;
    let want: Vec<i32> = runs
        .iter()
        .map(|(_, n)| {
            end += *n as i32;
            end - 1
        })
        .collect();
    assert_eq!(roots, want);
}

#[test]
fn blocks_without_a_root_field_are_left_in_order() {
    let reader = merged(None);
    let seg = &reader.segment_readers()[0];
    // Lucene's `hasBlocksButNoParentField`: the merge is not reordered.
    assert!(
        seg.diagnostics()
            .iter()
            .any(|(k, v)| k == REORDERED && v == "false"),
        "{:?}",
        seg.diagnostics()
    );
    // Each source segment's blocks stay in insertion order.
    let order = blocks(&reader);
    assert!(
        order.windows(2).all(|w| w[0] <= w[1] || w[1] % 40 == 0),
        "{order:?}"
    );
}

fn reordered(reader: &DirectoryReader) -> String {
    let seg = &reader.segment_readers()[0];
    seg.diagnostics()
        .iter()
        .find(|(k, _)| k == REORDERED)
        .map(|(_, v)| v.clone())
        .unwrap_or_default()
}

#[test]
fn a_block_without_its_root_is_left_in_order() {
    // Each segment ends with two children and no root after them: the merge's last block
    // has no root, which no renumbering can keep whole.
    let reader = merged_with(Some("_primary_term"), |w| {
        let orphans: Vec<Document> = (0..2)
            .map(|c| {
                let mut d = Document::new();
                d.add(TextField::new("body", format!("red orphan{c}"), Store::No));
                d.add(StringField::new("block", "999", Store::Yes));
                d
            })
            .collect();
        w.add_fields_documents(&orphans).unwrap();
    });
    assert_eq!(reordered(&reader), "false");
}

#[test]
fn documents_none_of_which_has_the_root_field_are_reordered_as_unblocked() {
    let dir = ByteBuffersDirectory::new();
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
    let mut reorderer = BpIndexReorderer::new();
    reorderer.set_min_doc_freq(2).unwrap();
    reorderer.set_min_partition_size(2).unwrap();
    reorderer.set_fields(Some(vec!["body".to_string()]));
    reorderer.set_block_root_field(Some("_primary_term".to_string()));
    w.set_pluggable_merge_policy(Some(Arc::new(BpReorderingMergePolicy::new(
        Box::new(TieredMergePolicy::default()),
        reorderer,
    ))));
    for i in 0..200 {
        let mut d = Document::new();
        d.add(TextField::new(
            "body",
            WORDS[(i * 7) % WORDS.len()].to_string(),
            Store::No,
        ));
        d.add(StringField::new("block", i.to_string(), Store::Yes));
        w.add_fields_document(&d).unwrap();
        if i % 50 == 49 {
            w.commit().unwrap();
        }
    }
    w.force_merge(1).unwrap();
    w.close().unwrap();
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reordered(&reader), "true");
    let order = blocks(&reader);
    assert!(order.windows(2).any(|w| w[0] > w[1]), "{order:?}");
}
