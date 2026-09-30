//! Writes `GenMergeReorder`'s index for `VerifyMergeReorder`: three
//! segments with deletes, force-merged through a merge policy whose every
//! `OneMerge` carries `wrapForMerge`/`reorder` hooks (hide every `rank`
//! multiple of 11, order by `rank` descending).
//!
//! Usage: `write_merge_reorder_fixture <output-dir>`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    Document, Field, IntPoint, NumericDocValuesField, SortedDocValuesField, Store, StoredValue,
    StringField, TextField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::filter::OneMergeWrappingMergePolicy;
use lucene_index::merge_policy::{MergePolicy, OneMerge, TieredMergePolicy};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::reader::filter::{FilterCodecReader, LeafFilter};
use lucene_search::reader::merge_readers::{MergeReaderHooks, SegmentMergeHooks};
use lucene_search::reader::sorting::DocMap;
use lucene_search::reader::{CodecReader, NO_MORE_DOCS};
use lucene_store::FsDirectory;
use lucene_util::fixed_bit_set::FixedBitSet;

const PER_SEGMENT: usize = 60;
const WORDS: [&str; 10] = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
];

fn value(i: usize, k: i64) -> i64 {
    let x = (i as i64 + 1) * 2_654_435_761 + k * 40_503;
    (x ^ ((x as u64) >> 13) as i64) % 100_000
}

fn word(i: usize, k: i64) -> &'static str {
    WORDS[(value(i, k) % WORDS.len() as i64) as usize]
}

/// `GenMergeReorder.doc`.
fn doc(i: usize) -> Document {
    let mut d = Document::new();
    d.add(StringField::new("id", format!("d{i}"), Store::Yes));
    d.add(TextField::new(
        "body",
        format!("{} {} {}", word(i, 1), word(i, 2), word(i, 3)),
        Store::No,
    ));
    d.add(NumericDocValuesField::new("rank", value(i, 0) % 1000));
    d.add(SortedDocValuesField::new("tag", word(i, 4)));
    d.add(IntPoint::new("pt", &[(value(i, 5) % 5000) as i32]).unwrap());
    d.add(Field::stored("n", StoredValue::Int(i as i32)));
    d
}

/// Every document's `rank`, by doc id.
fn ranks(reader: &dyn CodecReader) -> Vec<i64> {
    let mut out = vec![0i64; reader.max_doc() as usize];
    let mut dv = reader.numeric_doc_values("rank").unwrap().unwrap();
    loop {
        let doc = dv.next_doc().unwrap();
        if doc == NO_MORE_DOCS {
            break;
        }
        out[doc as usize] = dv.long_value();
    }
    out
}

/// A reader whose live documents are `live`.
struct Keep {
    live: FixedBitSet,
}

impl LeafFilter for Keep {
    fn live_docs<'a>(&'a self, _inner: Option<&'a FixedBitSet>) -> Option<&'a FixedBitSet> {
        Some(&self.live)
    }
}

/// `GenMergeReorder.RankMerge`.
struct RankHooks;

impl MergeReaderHooks for RankHooks {
    fn wrap_for_merge(
        &self,
        reader: Arc<dyn CodecReader>,
    ) -> lucene_search::Result<Arc<dyn CodecReader>> {
        let ranks = ranks(reader.as_ref());
        let mut live = FixedBitSet::new(ranks.len());
        for (doc, rank) in ranks.iter().enumerate() {
            let was_live = reader.live_docs().is_none_or(|l| l.get(doc));
            if was_live && rank % 11 != 0 {
                live.set(doc);
            }
        }
        Ok(Arc::new(FilterCodecReader::new(
            reader,
            Box::new(Keep { live }),
        )))
    }

    fn reorder(&self, reader: &dyn CodecReader) -> lucene_search::Result<Option<DocMap>> {
        let ranks = ranks(reader);
        let mut order: Vec<i32> = (0..reader.max_doc()).collect();
        order.sort_by(|&a, &b| ranks[b as usize].cmp(&ranks[a as usize]).then(a.cmp(&b)));
        DocMap::from_new_to_old(order).map(Some)
    }
}

fn write(dir: &FsDirectory) {
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", version).unwrap();
    w.set_max_full_flush_merge_wait_millis(0);
    let mut tmp = TieredMergePolicy::default();
    tmp.compound_file_settings_mut()
        .set_no_cfs_ratio(0.0)
        .unwrap();
    let hooks = Arc::new(SegmentMergeHooks::new(Arc::new(RankHooks)));
    let policy = OneMergeWrappingMergePolicy::new(
        Box::new(tmp),
        Arc::new(move |m: OneMerge| m.with_hooks(hooks.clone())),
    );
    w.set_pluggable_merge_policy(Some(Arc::new(policy)));
    for seg in 0..3 {
        for i in seg * PER_SEGMENT..(seg + 1) * PER_SEGMENT {
            w.add_fields_document(&doc(i)).unwrap();
        }
        w.commit().unwrap();
    }
    for i in (0..3 * PER_SEGMENT).step_by(13) {
        w.delete_documents_by_term(&[Term {
            field: "id".to_string(),
            bytes: format!("d{i}").into_bytes(),
        }])
        .unwrap();
    }
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    w.commit().unwrap();
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_merge_reorder_fixture <output-dir>");
    std::fs::create_dir_all(&out).unwrap();
    write(&FsDirectory::open(&out));
}
