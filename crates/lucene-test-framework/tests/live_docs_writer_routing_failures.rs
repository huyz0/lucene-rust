//! Real writer routing, failed-generation recovery and native cleanup ownership.

use std::{
    io,
    sync::{
        atomic::{AtomicU8, Ordering},
        Arc, Mutex,
    },
};

use lucene_codecs::{
    codec::Lucene104Codec,
    live_docs_writer::{LiveDocsWriteState, LiveDocsWriter},
    stored_fields::FieldValue,
};
use lucene_index::{
    buffered_updates::{DeleteQuery, Term},
    deletes::liv_file_name,
    document::{Document, Field, Store, StringField},
    index_writer::IndexWriter,
    merge_policy::log::LogDocMergePolicy,
    segment_info::LuceneVersion,
    segment_infos::{self, SegmentInfos},
};
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::{ByteBuffersDirectory, Directory};
use lucene_test_framework::{
    codecs::asserting_live_docs_writer::AssertingLiveDocsCodec,
    store::mock_directory_wrapper::{FailOn, FailureOp, MockDirectoryWrapper},
};
use lucene_util::{fixed_bit_set::FixedBitSet, java_random::JavaRandom};

#[derive(Debug, Clone, PartialEq, Eq)]
struct Call {
    segment: String,
    id: [u8; 16],
    max_doc: usize,
    del_gen: i64,
    next_del_gen: i64,
    old_count: usize,
    new_count: usize,
    live: Vec<usize>,
}

#[derive(Debug, Default)]
struct Probe {
    calls: Mutex<Vec<Call>>,
    // 0: success; 1: before creating an output; 2: after actual native close.
    failure: AtomicU8,
}

impl Probe {
    fn take(&self) -> Vec<Call> {
        std::mem::take(&mut *self.calls.lock().unwrap())
    }
}

impl Lucene104Codec for Probe {
    fn live_docs_writer(&self) -> Option<&dyn LiveDocsWriter> {
        Some(self)
    }
}

impl LiveDocsWriter for Probe {
    fn write_live_docs(
        &self,
        bits: &FixedBitSet,
        state: LiveDocsWriteState<'_>,
        new_del_count: usize,
    ) -> io::Result<()> {
        self.calls.lock().unwrap().push(Call {
            segment: state.segment.to_owned(),
            id: *state.id,
            max_doc: state.max_doc,
            del_gen: state.del_gen,
            next_del_gen: state.next_del_gen,
            old_count: state.del_count,
            new_count: new_del_count,
            // FBS: each index comes from 0..bits.len().
            live: (0..bits.len()).filter(|&doc| bits.get(doc)).collect(),
        });
        let failure = self.failure.load(Ordering::SeqCst);
        if failure == 1 {
            return Err(io::Error::other("before live-docs output"));
        }
        let directory = state.directory;
        let file = liv_file_name(state.segment, state.next_del_gen);
        let id = *state.id;
        let generation = state.next_del_gen;
        let max_doc = state.max_doc;
        let total_deletes = state.del_count + new_del_count;
        AssertingLiveDocsCodec.write_live_docs(bits, state, new_del_count)?;
        if failure == 2 {
            // Prove the competing error occurs after real native output close:
            // ByteBuffersDirectory cannot open an output still being written.
            let output = directory.open(&file).unwrap();
            let parsed =
                lucene_codecs::live_docs::parse(&output, &id, generation, max_doc, total_deletes)
                    .unwrap();
            assert_eq!(parsed.cardinality(), bits.cardinality());
            // FBS: every checked index is bounded by the supplied bitset length.
            assert!((0..bits.len()).all(|doc| parsed.get(doc) == bits.get(doc)));
            return Err(io::Error::other("after live-docs native close"));
        }
        Ok(())
    }
}

fn writer(directory: &dyn Directory, probe: Arc<Probe>) -> IndexWriter<'_> {
    let mut writer = IndexWriter::open(
        directory,
        Vec::new(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    writer.enable_explicit_documents().unwrap();
    writer.set_codec(Some(probe)).unwrap();
    writer.set_use_compound_file(false);
    writer.set_max_full_flush_merge_wait_millis(0);
    let mut policy = LogDocMergePolicy::new();
    policy.set_merge_factor(1000).unwrap();
    writer.set_pluggable_merge_policy(Some(Arc::new(policy)));
    writer
}

fn add_documents(writer: &mut IndexWriter<'_>) {
    for id in ["a", "b", "c", "d", "e", "f"] {
        let mut document = Document::new();
        document.add(StringField::new("id", id, Store::No));
        document.add(Field::stored("value", FieldValue::String(id.into())));
        writer.add_fields_document(&document).unwrap();
    }
    writer.commit().unwrap();
}

fn visible(directory: &dyn Directory) -> Vec<String> {
    let reader = DirectoryReader::open(directory).unwrap();
    let mut values = Vec::new();
    for segment in reader.segment_readers() {
        for doc in 0..segment.max_doc {
            if segment.live_docs().is_some_and(|live| !live.get_doc(doc)) {
                continue;
            }
            let stored = segment.stored_document(doc).unwrap().unwrap();
            let FieldValue::String(value) = &stored.fields[0].value else {
                panic!("stored id changed type")
            };
            values.push(value.clone());
        }
    }
    values
}

fn assert_committed_unchanged(directory: &dyn Directory, before: &SegmentInfos) {
    assert_eq!(
        segment_infos::read_latest(directory).unwrap().segments,
        before.segments
    );
    assert_eq!(visible(directory), ["b", "c", "d", "e", "f"]);
}

fn assert_transition(call: &Call, old: usize, new: usize, live: &[usize]) {
    assert_eq!(call.max_doc, 6);
    assert_eq!(call.old_count, old);
    assert_eq!(call.new_count, new);
    assert_eq!(call.live, live);
    assert_eq!(call.live.len() + old + new, call.max_doc);
}

#[test]
fn term_query_and_reader_deletes_supply_incremental_counts_and_existing_generation() {
    let directory = ByteBuffersDirectory::new();
    let probe = Arc::new(Probe::default());
    let mut writer = writer(&directory, probe.clone());
    add_documents(&mut writer);
    assert!(probe.take().is_empty());
    writer
        .delete_documents_by_term(&[Term::new("id", "a")])
        .unwrap();
    writer.commit().unwrap();
    let first = probe.take();
    assert_eq!(first.len(), 1);
    assert_transition(&first[0], 0, 1, &[1, 2, 3, 4, 5]);
    assert_eq!((first[0].del_gen, first[0].next_del_gen), (-1, 1));

    writer
        .delete_documents_by_query(&[DeleteQuery::Term(Term::new("id", "b"))])
        .unwrap();
    writer.commit().unwrap();
    let second = probe.take();
    assert_eq!(second.len(), 1);
    assert_transition(&second[0], 1, 1, &[2, 3, 4, 5]);
    assert_eq!((second[0].del_gen, second[0].next_del_gen), (1, 2));
    assert_eq!(second[0].id, first[0].id);
    assert_eq!(second[0].segment, first[0].segment);

    let snapshot = writer.nrt_snapshot(true, false).unwrap();
    assert!(writer
        .try_delete_document(&snapshot.segment_infos, 2)
        .unwrap()
        .is_some());
    let third = probe.take();
    assert_eq!(third.len(), 1);
    assert_transition(&third[0], 2, 1, &[3, 4, 5]);
    assert_eq!((third[0].del_gen, third[0].next_del_gen), (2, 3));
    // Until commit the reader opened from disk keeps the old live-docs file.
    assert_eq!(visible(&directory), ["c", "d", "e", "f"]);
    assert!(writer
        .try_delete_document(&snapshot.segment_infos, 2)
        .unwrap()
        .is_some());
    assert!(
        probe.take().is_empty(),
        "already-deleted reader id must not rewrite"
    );
    writer.commit().unwrap();
    assert_eq!(visible(&directory), ["d", "e", "f"]);

    writer.set_codec(None).unwrap();
    writer
        .delete_documents_by_term(&[Term::new("id", "d")])
        .unwrap();
    writer.commit().unwrap();
    assert!(probe.take().is_empty());
    assert_eq!(visible(&directory), ["e", "f"]);
}

#[test]
fn failures_before_write_and_after_native_close_keep_commit_and_advance_retry_generation() {
    for failure in [1, 2] {
        let directory = ByteBuffersDirectory::new();
        let probe = Arc::new(Probe::default());
        let mut writer = writer(&directory, probe.clone());
        add_documents(&mut writer);
        let snapshot = writer.nrt_snapshot(true, false).unwrap();
        writer
            .try_delete_document(&snapshot.segment_infos, 0)
            .unwrap()
            .unwrap();
        writer.commit().unwrap();
        let first = probe.take();
        assert_eq!(first.len(), 1);
        let committed = segment_infos::read_latest(&directory).unwrap();
        probe.failure.store(failure, Ordering::SeqCst);
        let error = writer
            .try_delete_document(&snapshot.segment_infos, 1)
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            if failure == 1 {
                "before live-docs output"
            } else {
                "after live-docs native close"
            }
        );
        let failed = probe.take();
        assert_eq!(failed.len(), 1);
        assert_transition(&failed[0], 1, 1, &[2, 3, 4, 5]);
        assert_eq!((failed[0].del_gen, failed[0].next_del_gen), (1, 2));
        assert_committed_unchanged(&directory, &committed);
        assert_eq!(writer.segment_infos().segments[0].del_gen, 1);
        assert_eq!(writer.segment_infos().segments[0].del_count, 1);
        assert_eq!(writer.segment_infos().segments[0].next_write_del_gen(), 3);
        let failed_file = liv_file_name(&failed[0].segment, 2);
        assert!(
            directory.open(&failed_file).is_err(),
            "failed candidate must be reclaimed"
        );

        probe.failure.store(0, Ordering::SeqCst);
        writer
            .try_delete_document(&snapshot.segment_infos, 1)
            .unwrap()
            .unwrap();
        let retried = probe.take();
        assert_eq!(retried.len(), 1);
        assert_transition(&retried[0], 1, 1, &[2, 3, 4, 5]);
        assert_eq!((retried[0].del_gen, retried[0].next_del_gen), (1, 3));
        assert_committed_unchanged(&directory, &committed);
        writer.commit().unwrap();
        assert_eq!(visible(&directory), ["c", "d", "e", "f"]);
        let final_info = segment_infos::read_latest(&directory).unwrap();
        assert_eq!(
            (
                final_info.segments[0].del_gen,
                final_info.segments[0].del_count
            ),
            (3, 2)
        );
    }
}

#[test]
fn preexisting_candidate_survives_precreate_and_already_exists_failures() {
    use lucene_store::data_output::DataOutput;

    for failure in [0, 1] {
        let directory = ByteBuffersDirectory::new();
        let probe = Arc::new(Probe::default());
        let mut writer = writer(&directory, probe.clone());
        add_documents(&mut writer);
        let snapshot = writer.nrt_snapshot(true, false).unwrap();
        writer
            .try_delete_document(&snapshot.segment_infos, 0)
            .unwrap()
            .unwrap();
        writer.commit().unwrap();
        probe.take();
        let committed = segment_infos::read_latest(&directory).unwrap();
        let candidate = liv_file_name(&committed.segments[0].segment_name, 2);
        let original = b"preexisting generation owned by another attempt";
        let mut output = directory.create_output(&candidate).unwrap();
        output.write_bytes(original);
        output.close().unwrap();
        probe.failure.store(failure, Ordering::SeqCst);
        let error = writer
            .try_delete_document(&snapshot.segment_infos, 1)
            .unwrap_err();
        if failure == 1 {
            assert_eq!(error.to_string(), "before live-docs output");
        } else {
            assert!(error.to_string().contains("exists"), "{error}");
        }
        assert_eq!(&*directory.open(&candidate).unwrap(), original);
        assert_committed_unchanged(&directory, &committed);
        assert_eq!(writer.segment_infos().segments[0].next_write_del_gen(), 3);
        let calls = probe.take();
        assert_eq!(calls.len(), 1);
        assert_transition(&calls[0], 1, 1, &[2, 3, 4, 5]);
        assert_eq!((calls[0].del_gen, calls[0].next_del_gen), (1, 2));

        probe.failure.store(0, Ordering::SeqCst);
        writer
            .try_delete_document(&snapshot.segment_infos, 1)
            .unwrap()
            .unwrap();
        let retry = probe.take();
        assert_eq!(retry.len(), 1);
        assert_eq!(retry[0].next_del_gen, 3);
        writer.commit().unwrap();
        assert_eq!(visible(&directory), ["c", "d", "e", "f"]);
    }
}

#[test]
fn sync_failure_and_cleanup_failure_preserve_primary_error_and_skip_abandoned_generation() {
    for fail_cleanup in [false, true] {
        let directory =
            MockDirectoryWrapper::new(&mut JavaRandom::new(19), ByteBuffersDirectory::new());
        let probe = Arc::new(Probe::default());
        let mut writer = writer(&directory, probe.clone());
        add_documents(&mut writer);
        let snapshot = writer.nrt_snapshot(true, false).unwrap();
        writer
            .try_delete_document(&snapshot.segment_infos, 0)
            .unwrap()
            .unwrap();
        writer.commit().unwrap();
        probe.take();
        let committed = segment_infos::read_latest(&directory).unwrap();
        let sync = FailOn::once(|context| {
            context.op == FailureOp::Sync && context.name.ends_with("_2.liv")
        });
        directory.fail_on(sync.clone());
        let cleanup = FailOn::new(move |context| {
            fail_cleanup && context.op == FailureOp::DeleteFile && context.name.ends_with("_2.liv")
        });
        directory.fail_on(cleanup.clone());
        let error = writer
            .try_delete_document(&snapshot.segment_infos, 1)
            .unwrap_err();
        assert!(error.to_string().contains("Sync"), "{error}");
        assert_eq!(sync.fired(), 1);
        assert_eq!(cleanup.fired(), usize::from(fail_cleanup));
        assert_committed_unchanged(&directory, &committed);
        let failed = probe.take();
        assert_eq!(failed.len(), 1);
        assert_transition(&failed[0], 1, 1, &[2, 3, 4, 5]);
        let failed_file = liv_file_name(&failed[0].segment, 2);
        assert_eq!(directory.open(&failed_file).is_ok(), fail_cleanup);
        assert_eq!(writer.segment_infos().segments[0].next_write_del_gen(), 3);

        cleanup.clear_do_fail();
        writer
            .try_delete_document(&snapshot.segment_infos, 1)
            .unwrap()
            .unwrap();
        let retried = probe.take();
        assert_eq!(retried.len(), 1);
        assert_eq!((retried[0].del_gen, retried[0].next_del_gen), (1, 3));
        writer.commit().unwrap();
        assert_eq!(visible(&directory), ["c", "d", "e", "f"]);
        assert_eq!(
            segment_infos::read_latest(&directory).unwrap().segments[0].del_gen,
            3
        );
    }
}

#[test]
fn filtered_import_installs_writer_for_hidden_source_documents() {
    let source = ByteBuffersDirectory::new();
    let mut source_writer = writer(&source, Arc::new(Probe::default()));
    add_documents(&mut source_writer);
    drop(source_writer);
    let directory = ByteBuffersDirectory::new();
    let probe = Arc::new(Probe::default());
    let mut writer = writer(&directory, probe.clone());
    writer
        .add_indexes_filtered(&[&source], &|_, _, doc| doc % 2 == 0)
        .unwrap();
    let calls = probe.take();
    assert_eq!(calls.len(), 1);
    assert_transition(&calls[0], 0, 3, &[0, 2, 4]);
    assert_eq!((calls[0].del_gen, calls[0].next_del_gen), (-1, 1));
    writer.commit().unwrap();
    assert_eq!(visible(&directory), ["a", "c", "e"]);
    let reader = DirectoryReader::open(&directory).unwrap();
    assert_eq!((reader.max_doc(), reader.num_docs()), (3, 3));
}

#[test]
fn private_delete_before_first_flush_routes_newly_built_segment_without_publishing_early() {
    let directory = ByteBuffersDirectory::new();
    let probe = Arc::new(Probe::default());
    let mut writer = writer(&directory, probe.clone());
    for id in ["a", "b", "c"] {
        let mut document = Document::new();
        document.add(StringField::new("id", id, Store::No));
        document.add(Field::stored("value", FieldValue::String(id.into())));
        writer.add_fields_document(&document).unwrap();
    }
    writer
        .delete_documents_by_term(&[Term::new("id", "a")])
        .unwrap();
    assert!(
        probe.take().is_empty(),
        "unflushed deletes cannot write .liv yet"
    );
    assert!(segment_infos::read_latest(&directory).is_err());
    writer.flush().unwrap();
    let calls = probe.take();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        (calls[0].max_doc, calls[0].old_count, calls[0].new_count),
        (3, 0, 1)
    );
    assert_eq!((calls[0].del_gen, calls[0].next_del_gen), (-1, 1));
    assert_eq!(calls[0].live, [1, 2]);
    assert!(segment_infos::read_latest(&directory).is_err());
    writer.commit().unwrap();
    assert!(probe.take().is_empty());
    assert_eq!(visible(&directory), ["b", "c"]);
}

#[test]
fn deletes_issued_while_merge_is_warmed_route_the_carried_generation_before_publication() {
    use std::sync::{atomic::AtomicBool, mpsc};
    use std::time::Duration;

    use lucene_codecs::{
        field_infos::{FieldInfo, IndexOptions},
        stored_fields::{self, StoredField},
    };
    use lucene_index::{
        concurrent_writer::ConcurrentIndexWriter,
        index_writer::{MergedSegmentWarmer, Result},
        merge_policy::MergePolicyConfig,
        segment_infos::SegmentCommitInfo,
    };

    struct PauseFirstMerge {
        first: AtomicBool,
        entered: mpsc::SyncSender<String>,
        release: Mutex<mpsc::Receiver<()>>,
    }
    impl MergedSegmentWarmer for PauseFirstMerge {
        fn warm(&self, directory: &dyn Directory, segment: &SegmentCommitInfo) -> Result<()> {
            if self.first.swap(false, Ordering::SeqCst) {
                directory.open(&format!("{}.si", segment.segment_name))?;
                self.entered.send(segment.segment_name.clone()).unwrap();
                self.release
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(30))
                    .unwrap();
            }
            Ok(())
        }
    }

    let directory = ByteBuffersDirectory::new();
    let probe = Arc::new(Probe::default());
    // The concurrent writer accepts ordinary fixed-schema documents. It owns
    // this IndexWriter's configuration for both reader-id and carried deletes.
    let mut field = FieldInfo::new("id", 0);
    field.index_options = IndexOptions::Docs;
    field.omit_norms = true;
    let mut single = IndexWriter::open(
        &directory,
        vec![field],
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    single.set_codec(Some(probe.clone())).unwrap();
    single.set_postings_field(Some("id")).unwrap();
    single.set_use_compound_file(false);
    single.set_max_full_flush_merge_wait_millis(0);
    single.set_merge_policy(Some(MergePolicyConfig {
        max_merge_at_once: 2,
        segments_per_tier: 2,
        floor_segment_size: 1 << 30,
        ..MergePolicyConfig::default()
    }));
    let (entered_send, entered_recv) = mpsc::sync_channel(1);
    let (release_send, release_recv) = mpsc::sync_channel(1);
    single.set_merged_segment_warmer(Some(Arc::new(PauseFirstMerge {
        first: AtomicBool::new(true),
        entered: entered_send,
        release: Mutex::new(release_recv),
    })));
    let writer = ConcurrentIndexWriter::new(single, 1).unwrap();
    for batch in [["a", "b"], ["c", "d"], ["e", "f"]] {
        for id in batch {
            writer
                .add_document(stored_fields::Document {
                    fields: vec![StoredField {
                        field_number: 0,
                        value: FieldValue::String(id.into()),
                    }],
                })
                .unwrap();
        }
        writer.commit().unwrap();
    }
    let committed = segment_infos::read_latest(&directory).unwrap();
    assert_eq!(committed.segments.len(), 3);
    assert!(probe.take().is_empty());
    std::thread::scope(|scope| {
        let merging = scope.spawn(|| writer.maybe_merge());
        let merged_name = entered_recv.recv_timeout(Duration::from_secs(30)).unwrap();
        // Every possible two-source merge must carry exactly two of these
        // deletes. A source retains one live document, so it is not dropped.
        for doc in [0, 2, 4] {
            writer
                .try_delete_document(&committed, doc)
                .unwrap()
                .unwrap();
        }
        let source_calls = probe.take();
        assert_eq!(source_calls.len(), 3);
        for call in source_calls {
            assert_eq!((call.max_doc, call.old_count, call.new_count), (2, 0, 1));
            assert_eq!(call.live, [1]);
        }
        assert_eq!(
            segment_infos::read_latest(&directory).unwrap().segments,
            committed.segments
        );
        assert_eq!(visible(&directory), ["a", "b", "c", "d", "e", "f"]);
        release_send.send(()).unwrap();
        assert!(merging.join().unwrap().unwrap() >= 1);
        let carried = probe.take();
        assert_eq!(carried.len(), 1);
        assert_eq!(carried[0].segment, merged_name);
        assert_eq!(
            (
                carried[0].max_doc,
                carried[0].old_count,
                carried[0].new_count
            ),
            (4, 0, 2)
        );
        assert_eq!((carried[0].del_gen, carried[0].next_del_gen), (-1, 1));
        assert_eq!(carried[0].live, [1, 3]);
        assert_eq!(
            segment_infos::read_latest(&directory).unwrap().segments,
            committed.segments
        );
        writer.commit().unwrap();
    });
    let mut values = visible(&directory);
    values.sort();
    assert_eq!(values, ["b", "d", "f"]);
}
