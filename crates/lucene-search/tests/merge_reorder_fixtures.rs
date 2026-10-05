//! `OneMerge.wrapForMerge` and `OneMerge.reorder` through the writer, against
//! Java (`GenMergeReorder`): a merge policy whose every merge carries hooks
//! (`OneMerge::with_hooks` with a [`SegmentMergeHooks`] over a
//! `MergeReaderHooks`) that hide every document whose `rank` is a multiple of
//! 11 and reorder the merged view by `rank` descending. The same documents
//! and deletes are force-merged, and the merged segment must be Java's: the
//! same documents in the same order (`order.txt`) and every file but the
//! `.si` byte for byte, segment id normalised.

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
use lucene_index::segment_infos;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::filter::{FilterCodecReader, LeafFilter};
use lucene_search::reader::merge_readers::{MergeReaderHooks, SegmentMergeHooks};
use lucene_search::reader::sorting::DocMap;
use lucene_search::reader::{
    CodecReader, IndexReader, LeafReader, StoredFieldVisitor, VisitStatus, NO_MORE_DOCS,
};
use lucene_store::{Directory, FsDirectory};
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::test_support::TempDir;

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

/// `wrapForMerge` hiding every document whose `rank` is a multiple of 11.
fn drop_elevenths(reader: Arc<dyn CodecReader>) -> Arc<dyn CodecReader> {
    let ranks = ranks(reader.as_ref());
    let mut live = FixedBitSet::new(ranks.len());
    for (doc, rank) in ranks.iter().enumerate() {
        let was_live = reader.live_docs().is_none_or(|l| l.get(doc));
        if was_live && rank % 11 != 0 {
            live.set(doc);
        }
    }
    Arc::new(FilterCodecReader::new(reader, Box::new(Keep { live })))
}

/// The same wrapping, without a reorder.
struct WrapOnly;

impl MergeReaderHooks for WrapOnly {
    fn wrap_for_merge(
        &self,
        reader: Arc<dyn CodecReader>,
    ) -> lucene_search::Result<Arc<dyn CodecReader>> {
        Ok(drop_elevenths(reader))
    }
}

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
    write_with(dir, Arc::new(RankHooks));
}

fn write_with(dir: &FsDirectory, merge_hooks: Arc<dyn MergeReaderHooks>) {
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
    let hooks = Arc::new(SegmentMergeHooks::new(merge_hooks));
    assert_eq!(format!("{hooks:?}"), "SegmentMergeHooks");
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

fn fixture() -> String {
    format!(
        "{}/../../fixtures/data/merge_reorder",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// Reads a document's stored `id`.
#[derive(Default)]
struct Id(Option<String>);

impl StoredFieldVisitor for Id {
    fn needs_field(&mut self, _n: i32) -> lucene_codecs::stored_fields::Result<VisitStatus> {
        Ok(VisitStatus::Yes)
    }
    fn string_field(&mut self, _n: i32, value: &str) -> lucene_codecs::stored_fields::Result<()> {
        self.0 = Some(value.to_string());
        Ok(())
    }
}

/// `ours` with every copy of its segment id replaced and its footer
/// re-signed.
fn with_segment_id(ours: &[u8], our_id: &[u8; 16], id: &[u8; 16]) -> Vec<u8> {
    let mut bytes = ours.to_vec();
    let mut at = 0;
    while at + 16 <= bytes.len() {
        if &bytes[at..at + 16] == our_id {
            bytes[at..at + 16].copy_from_slice(id);
            at += 16;
        } else {
            at += 1;
        }
    }
    let n = bytes.len();
    let crc = u64::from(crc32fast::hash(&bytes[..n - 8]));
    bytes[n - 8..].copy_from_slice(&crc.to_be_bytes());
    bytes
}

#[test]
fn a_reordering_merge_writes_javas_segment() {
    let tmp = TempDir::new("merge-reorder");
    let dir = FsDirectory::open(&tmp);
    write(&dir);

    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), 1);
    let seg = &reader.segment_readers()[0];
    let ids: Vec<String> = (0..seg.max_doc())
        .map(|d| {
            let mut id = Id::default();
            seg.document(d, &mut id).unwrap();
            id.0.unwrap()
        })
        .collect();
    let expected: Vec<String> = std::fs::read_to_string(format!("{}/order.txt", fixture()))
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(ids, expected, "the merged documents and their order");

    let java = FsDirectory::open(fixture());
    let ours = segment_infos::read_latest(&dir).unwrap();
    let theirs = segment_infos::read_latest(&java).unwrap();
    let (a, b) = (&ours.segments[0], &theirs.segments[0]);
    let mut compared = 0;
    for name in java.list_all().unwrap() {
        let Some(rest) = name.strip_prefix(b.segment_name.as_str()) else {
            continue;
        };
        if rest == ".si" {
            continue;
        }
        let mine_name = format!("{}{rest}", a.segment_name);
        let mine = dir.open(&mine_name).unwrap();
        let java_bytes = java.open(&name).unwrap();
        assert!(
            with_segment_id(&mine, &a.segment_id, &b.segment_id) == java_bytes[..],
            "{mine_name} differs from Java's {name}"
        );
        compared += 1;
    }
    assert_eq!(compared, 18);
    for result in lucene_index::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

/// `wrapForMerge` without a reorder: the merged segment keeps the wrapped
/// readers' documents in their order -- every document `write` left live
/// whose `rank` is not a multiple of 11, each segment's in a run.
#[test]
fn a_wrapping_merge_without_a_reorder_keeps_the_wrapped_documents_in_order() {
    let tmp = TempDir::new("merge-wrap-only");
    let dir = FsDirectory::open(&tmp);
    write_with(&dir, Arc::new(WrapOnly));
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), 1);
    let seg = &reader.segment_readers()[0];
    let ids: Vec<String> = (0..seg.max_doc())
        .map(|d| {
            let mut id = Id::default();
            seg.document(d, &mut id).unwrap();
            id.0.unwrap()
        })
        .collect();
    let want: Vec<String> = (0..3 * PER_SEGMENT)
        .filter(|&i| i % 13 != 0 && (value(i, 0) % 1000) % 11 != 0)
        .map(|i| format!("d{i}"))
        .collect();
    assert!(want.len() > 100 && want.len() < 3 * PER_SEGMENT);
    // The merge takes its segments in the policy's order, each whole and in
    // its own order.
    let n = |id: &String| id[1..].parse::<usize>().unwrap();
    let mut sorted = ids.clone();
    sorted.sort_by_key(n);
    assert_eq!(sorted, want);
    let runs: Vec<usize> = ids.iter().map(|id| n(id) / PER_SEGMENT).collect();
    assert_eq!(
        runs.windows(2).filter(|w| w[0] != w[1]).count(),
        2,
        "{runs:?}"
    );
    assert!(ids
        .windows(2)
        .all(|w| n(&w[0]) / PER_SEGMENT != n(&w[1]) / PER_SEGMENT || n(&w[0]) < n(&w[1])));
    for result in lucene_index::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}
