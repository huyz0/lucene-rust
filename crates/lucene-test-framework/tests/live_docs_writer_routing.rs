//! Committed live-doc generations from real Java IndexWriter sessions.
//! Only the asserting live-doc writer is installed; reads keep the native graph.

use std::{
    io,
    sync::{Arc, Mutex},
};

use lucene_codecs::{
    codec::Lucene104Codec,
    live_docs_writer::{LiveDocsWriteState, LiveDocsWriter},
    stored_fields::FieldValue,
};
use lucene_index::{
    buffered_updates::{DeleteQuery, Term},
    document::{Document, NumericDocValuesField, Store, StringField},
    index_writer::IndexWriter,
    merge_policy::log::LogDocMergePolicy,
    segment_info::{IndexSortField, LuceneVersion},
    segment_infos,
};
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::{Directory, FsDirectory};
use lucene_test_framework::codecs::asserting_live_docs_writer::AssertingLiveDocsCodec;
use lucene_util::{fixed_bit_set::FixedBitSet, test_support::TempDir};

#[derive(Debug, Default)]
struct Probe(Mutex<Vec<String>>);

impl Probe {
    fn take(&self) -> Vec<String> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

#[derive(Debug)]
struct CountingCodec(Arc<Probe>);

impl Lucene104Codec for CountingCodec {
    fn live_docs_writer(&self) -> Option<&dyn LiveDocsWriter> {
        Some(self)
    }
}

impl LiveDocsWriter for CountingCodec {
    fn write_live_docs(
        &self,
        bits: &FixedBitSet,
        state: LiveDocsWriteState<'_>,
        new_del_count: usize,
    ) -> io::Result<()> {
        self.0 .0.lock().unwrap().push(format!(
            "{}:{}:{}:{}:{}:{}",
            state.max_doc,
            state.del_gen,
            state.next_del_gen,
            state.del_count,
            new_del_count,
            mask(Some(bits), state.max_doc),
        ));
        AssertingLiveDocsCodec.write_live_docs(bits, state, new_del_count)
    }
}

fn writer<'a>(dir: &'a dyn Directory, probe: Arc<Probe>, sorted: bool) -> IndexWriter<'a> {
    let mut writer = IndexWriter::open(
        dir,
        Vec::new(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    writer
        .set_codec(Some(Arc::new(CountingCodec(probe))))
        .unwrap();
    writer.set_max_full_flush_merge_wait_millis(0);
    writer.set_use_compound_file(false);
    let mut policy = LogDocMergePolicy::new();
    policy.set_merge_factor(1000).unwrap();
    writer.set_pluggable_merge_policy(Some(Arc::new(policy)));
    writer.enable_explicit_documents().unwrap();
    if sorted {
        writer
            .set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
            .unwrap();
    }
    writer
}

fn mask(bits: Option<&FixedBitSet>, length: usize) -> String {
    (0..length)
        .map(|doc| {
            if bits.is_none_or(|bits| bits.get(doc)) {
                '1'
            } else {
                '0'
            }
        })
        .collect()
}

fn document(row: &[&str]) -> Document {
    let mut doc = Document::new();
    doc.add(StringField::new("id", row[2], Store::Yes));
    doc.add(StringField::new("group", row[4], Store::No));
    doc.add(NumericDocValuesField::new("rank", row[3].parse().unwrap()));
    doc
}

fn snapshot(dir: &dyn Directory, case: &str, phase: &str) -> Vec<String> {
    let infos = segment_infos::read_latest(dir).unwrap();
    let reader = DirectoryReader::open(dir).unwrap();
    let mut rows = vec![format!(
        "CASE\t{case}\t{phase}\t{}\t{}\t{}",
        reader.max_doc(),
        reader.num_docs(),
        reader.segment_readers().len(),
    )];
    for (ordinal, segment) in reader.segment_readers().iter().enumerate() {
        let info = &infos.segments[ordinal];
        rows.push(format!(
            "SEG\t{case}\t{phase}\t{ordinal}\t{}\t{}\t{}\t{}",
            segment.max_doc,
            info.del_gen,
            info.del_count,
            mask(
                segment.live_docs(),
                usize::try_from(segment.max_doc).unwrap()
            ),
        ));
        for doc in 0..segment.max_doc {
            let stored = segment.stored_document(doc).unwrap().unwrap();
            let id = stored
                .fields
                .iter()
                .find_map(|field| match &field.value {
                    FieldValue::String(value) => Some(value.as_str()),
                    _ => None,
                })
                .unwrap();
            let live = i32::from(segment.live_docs().is_none_or(|bits| bits.get_doc(doc)));
            rows.push(format!(
                "DOC\t{case}\t{phase}\t{ordinal}\t{doc}\t{live}\t{id}"
            ));
        }
    }
    rows
}

fn assert_phase(fixture: &str, dir: &dyn Directory, probe: &Probe, case: &str, phase: &str) {
    let expected: Vec<String> = fixture
        .lines()
        .filter(|line| {
            ["CASE", "SEG", "DOC"]
                .iter()
                .any(|kind| line.starts_with(&format!("{kind}\t{case}\t{phase}\t")))
        })
        .map(str::to_owned)
        .collect();
    assert!(!expected.is_empty(), "missing Java snapshot {case} {phase}");
    assert_eq!(snapshot(dir, case, phase), expected, "{case} {phase}");
    // Segment identities and callback order are writer implementation details.
    // Every state/count/mask reaching the real native writer must still agree.
    let mut expected_events: Vec<String> = fixture
        .lines()
        .filter(|line| line.starts_with(&format!("TRACE\t{case}\t")))
        .map(|line| line.split('\t').nth(3).unwrap())
        .filter(|trace| trace.starts_with(&format!("{phase}:")))
        .map(|trace| trace.splitn(3, ':').nth(2).unwrap().to_owned())
        .collect();
    let mut actual_events = probe.take();
    expected_events.sort();
    actual_events.sort();
    assert_eq!(
        actual_events, expected_events,
        "writer callbacks {case} {phase}"
    );
}

#[test]
fn term_query_noop_append_and_sorted_merges_match_java_live_doc_generations() {
    let fixture = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/testframework/data/live_docs_writer_routing.tsv",
    ))
    .unwrap();
    let inputs: Vec<Vec<&str>> = fixture
        .lines()
        .filter(|line| line.starts_with("INPUT\t"))
        .map(|line| line.split('\t').collect())
        .collect();
    assert_eq!(inputs.len(), 9);
    for sorted in [false, true] {
        let case = if sorted { "sorted" } else { "unsorted" };
        let temp = TempDir::new("live-docs-routing-oracle");
        let dir = FsDirectory::open(&temp);
        let probe = Arc::new(Probe::default());
        {
            let mut writer = writer(&dir, probe.clone(), sorted);
            for batch in inputs[..8].chunks(4) {
                for row in batch {
                    writer.add_fields_document(&document(row)).unwrap();
                }
                writer.commit().unwrap();
            }
            assert_phase(&fixture, &dir, &probe, case, "initial");
            writer
                .delete_documents_by_term(&[Term::new("id", "b")])
                .unwrap();
            writer.commit().unwrap();
            assert_phase(&fixture, &dir, &probe, case, "term");
            writer
                .delete_documents_by_term(&[Term::new("id", "b"), Term::new("id", "missing")])
                .unwrap();
            writer
                .delete_documents_by_query(&[DeleteQuery::Term(Term::new("group", "never"))])
                .unwrap();
            writer.commit().unwrap();
            assert_phase(&fixture, &dir, &probe, case, "noop");
            writer
                .delete_documents_by_query(&[DeleteQuery::Term(Term::new("group", "drop"))])
                .unwrap();
            writer.commit().unwrap();
            assert_phase(&fixture, &dir, &probe, case, "query");
        }
        let mut writer = writer(&dir, probe.clone(), sorted);
        writer.add_fields_document(&document(&inputs[8])).unwrap();
        writer.commit().unwrap();
        assert_phase(&fixture, &dir, &probe, case, "append");
        writer
            .delete_documents_by_term(&[Term::new("id", "a"), Term::new("id", "h")])
            .unwrap();
        writer.commit().unwrap();
        assert_phase(&fixture, &dir, &probe, case, "append_delete");
        writer.force_merge(1).unwrap();
        writer.commit().unwrap();
        assert_phase(&fixture, &dir, &probe, case, "merged");
        writer
            .delete_documents_by_term(&[Term::new("id", "g")])
            .unwrap();
        writer.commit().unwrap();
        assert_phase(&fixture, &dir, &probe, case, "merged_delete");
    }
}
