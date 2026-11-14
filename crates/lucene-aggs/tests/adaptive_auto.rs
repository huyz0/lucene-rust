//! `index.codec.adaptive_doc_values: auto` end to end through the writer:
//! the segments before the sample is complete are stock, the choice is
//! recorded in the commit and kept by a reopened writer, merges follow it,
//! and aggregations over the mixed segments answer as over a stock copy.
#![allow(clippy::arithmetic_side_effects)]

use lucene_aggs::metric_aggs::{metric_states, MetricSpec, Source, ValueKind, NEED_ALL};
use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo};
use lucene_codecs::per_field_doc_values::Lucene90DocValuesFormat;
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::adaptive_auto::{COMMIT_KEY, SAMPLE_DOCS};
use lucene_index::index_writer::{ExplicitDocument, ExplicitFields, IndexWriter};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::MatchAllDocsQuery;
use lucene_search::{BooleanQuery, Clause};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};
const TS: i32 = 0;
const FARE: i32 = 1;
/// Documents per flushed segment.
const FLUSH: usize = 8_192;

/// A shuffle by a fixed odd multiplier: a column in no order.
fn scramble(i: usize) -> i64 {
    (i.wrapping_mul(2_654_435_761) % 1_000_003) as i64
}

fn doc(ts: i64, fare: i64) -> ExplicitDocument {
    let long = |field_number, v| StoredField {
        field_number,
        value: FieldValue::Long(v),
    };
    ExplicitDocument {
        stored: Vec::new(),
        fields: ExplicitFields {
            doc_values: vec![long(TS, ts), long(FARE, fare)],
            ..ExplicitFields::default()
        },
    }
}

fn writer(dir: &FsDirectory, auto: bool) -> IndexWriter<'_> {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    if auto {
        w.set_adaptive_doc_values_auto();
    }
    for (name, number) in [("ts", TS), ("fare", FARE)] {
        w.register_field(FieldInfo::new(name, number).with_doc_values(
            DocValuesType::SortedNumeric,
            DocValuesSkipIndexType::None,
            -1,
        ))
        .unwrap();
    }
    w
}

/// `flushes` segments of [`FLUSH`] documents from document `from` on, each
/// committed.
fn add(w: &mut IndexWriter<'_>, from: usize, flushes: usize, ts: impl Fn(usize) -> i64) {
    for f in 0..flushes {
        let start = from + f * FLUSH;
        let docs = (start..start + FLUSH)
            .map(|i| doc(ts(i), scramble(i)))
            .collect();
        w.add_explicit_documents(docs).unwrap();
        w.commit().unwrap();
    }
}

/// Per segment, in order: whether its `ts` column is in the adaptive
/// format.
fn adaptive_segments(dir: &FsDirectory) -> Vec<bool> {
    let reader = DirectoryReader::open(dir).unwrap();
    reader
        .segment_readers()
        .iter()
        .map(|r| {
            let (meta, _) = r.doc_values_for_field(TS).unwrap();
            meta.sorted_numeric_entry(TS)
                .unwrap()
                .numeric
                .adaptive
                .is_some()
        })
        .collect()
}

fn recorded(dir: &FsDirectory) -> Option<String> {
    lucene_index::segment_infos::read_latest(dir)
        .unwrap()
        .user_data
        .iter()
        .find(|(k, _)| k == COMMIT_KEY)
        .map(|(_, v)| v.clone())
}

fn stats(dir: &FsDirectory) -> Vec<lucene_aggs::metric_aggs::MetricState> {
    let reader = DirectoryReader::open(dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments: Vec<OpenSegment<'_>> = opened.as_open_segments();
    let mut q = BooleanQuery::new();
    q.filter = vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(i32::MAX))];
    let specs: Vec<MetricSpec> = ["ts", "fare"]
        .iter()
        .map(|f| MetricSpec {
            field: f.to_string(),
            kind: ValueKind::Long,
            source: Source::DocValues,
            needs: NEED_ALL,
        })
        .collect();
    metric_states(&segments, reader.segment_readers(), &q, &specs).unwrap()
}

#[test]
fn a_time_ordered_index_turns_adaptive_once_sampled_and_stays_so() {
    let tmp = [
        TempDir::new("auto-ordered"),
        TempDir::new("auto-ordered-stock"),
    ];
    let dirs = [
        FsDirectory::open(tmp[0].path()),
        FsDirectory::open(tmp[1].path()),
    ];
    let ts = |i: usize| 1_700_000_000_000 + i as i64 * 37;
    let sampled = SAMPLE_DOCS as usize / FLUSH;
    {
        let mut w = writer(&dirs[0], true);
        add(&mut w, 0, sampled - 1, ts);
        assert_eq!(w.adaptive_doc_values_choice(), None);
        assert_eq!(recorded(&dirs[0]), None);
        add(&mut w, (sampled - 1) * FLUSH, 2, ts);
        assert_eq!(w.adaptive_doc_values_choice(), Some(true));
    }
    let mut expected = vec![false; sampled - 1];
    expected.extend([true, true]);
    assert_eq!(adaptive_segments(&dirs[0]), expected);
    assert_eq!(recorded(&dirs[0]).as_deref(), Some("true"));

    // Reopened: the recorded choice holds, even over data in no order.
    {
        let mut w = writer(&dirs[0], true);
        assert_eq!(w.adaptive_doc_values_choice(), Some(true));
        add(&mut w, (sampled + 1) * FLUSH, 1, scramble);
        // Commit data the engine sets keeps the choice.
        w.set_live_commit_data(vec![("translog_uuid".into(), "u".into())]);
        w.commit().unwrap();
    }
    expected.push(true);
    assert_eq!(adaptive_segments(&dirs[0]), expected);
    assert_eq!(recorded(&dirs[0]).as_deref(), Some("true"));

    // The same documents in a stock copy answer the same; and so after a
    // merge, which writes the chosen format.
    {
        let mut w = writer(&dirs[1], false);
        add(&mut w, 0, sampled + 1, ts);
        add(&mut w, (sampled + 1) * FLUSH, 1, scramble);
    }
    assert_eq!(stats(&dirs[0]), stats(&dirs[1]));
    for (k, dir) in dirs.iter().enumerate() {
        let mut w = writer(dir, k == 0);
        w.force_merge(1).unwrap();
        w.commit().unwrap();
    }
    assert_eq!(adaptive_segments(&dirs[0]), vec![true]);
    assert_eq!(adaptive_segments(&dirs[1]), vec![false]);
    assert_eq!(stats(&dirs[0]), stats(&dirs[1]));
}

#[test]
fn an_index_in_no_order_stays_stock() {
    let tmp = TempDir::new("auto-shuffled");
    let dir = FsDirectory::open(tmp.path());
    {
        let mut w = writer(&dir, true);
        add(&mut w, 0, SAMPLE_DOCS as usize / FLUSH + 1, |i| {
            1_700_000_000_000 + scramble(i) * 37
        });
        assert_eq!(w.adaptive_doc_values_choice(), Some(false));
        w.force_merge(1).unwrap();
        w.commit().unwrap();
    }
    assert_eq!(adaptive_segments(&dir), vec![false]);
    assert_eq!(recorded(&dir).as_deref(), Some("false"));
}

#[test]
fn without_auto_nothing_is_sampled_or_recorded() {
    let tmp = TempDir::new("auto-off");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir, false);
    w.set_default_doc_values_format(Lucene90DocValuesFormat::rust_adaptive());
    add(&mut w, 0, SAMPLE_DOCS as usize / FLUSH + 1, scramble);
    assert_eq!(w.adaptive_doc_values_choice(), None);
    drop(w);
    assert_eq!(recorded(&dir), None);
}

#[test]
fn the_engines_own_fields_do_not_count() {
    // `_seq_no` rises with arrival in every index: in order here, with every
    // other column in no order, the index stays stock.
    const SEQ_NO: i32 = 2;
    let tmp = TempDir::new("auto-seq-no");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir, true);
    w.register_field(FieldInfo::new("_seq_no", SEQ_NO).with_doc_values(
        DocValuesType::Numeric,
        DocValuesSkipIndexType::None,
        -1,
    ))
    .unwrap();
    for f in 0..SAMPLE_DOCS as usize / FLUSH {
        let docs = (f * FLUSH..(f + 1) * FLUSH)
            .map(|i| {
                let mut d = doc(scramble(i), scramble(i + 7));
                d.fields.doc_values.push(StoredField {
                    field_number: SEQ_NO,
                    value: FieldValue::Long(i as i64),
                });
                d
            })
            .collect();
        w.add_explicit_documents(docs).unwrap();
        w.commit().unwrap();
    }
    assert_eq!(w.adaptive_doc_values_choice(), Some(false));
}
