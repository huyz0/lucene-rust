//! Installed postings failures retain committed bytes and retry original inputs.
use lucene_codecs::{
    codec::Lucene104Codec,
    field_infos::IndexOptions,
    per_field_postings::Lucene104PostingsFormat,
    postings_consumer::{PostingsWriteSink, PostingsWriteState, PostingsWriterFactory},
    postings_writer::{self, FieldNorms, FieldPostingsInput, TermSource},
    stored_fields::FieldValue,
};
use lucene_index::{
    document::{Document, Field, FieldType, Store, StringField},
    index_writer::IndexWriter,
    merge_policy::log::LogDocMergePolicy,
    segment_info::LuceneVersion,
    segment_infos,
};
use lucene_search::{directory_reader::DirectoryReader, reader::IndexReader};
use lucene_store::{ByteBuffersDirectory, DataOutput, Directory};
use lucene_test_framework::{
    codecs::asserting_postings::AssertingPostingsCodec,
    store::mock_directory_wrapper::{FailOn, FailureOp, MockDirectoryWrapper},
};
use lucene_util::java_random::JavaRandom;
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc, Mutex,
};

const EXTS: [&str; 7] = ["psm", "doc", "pos", "pay", "tim", "tip", "tmd"];
#[derive(Debug, Default)]
struct Probe {
    mode: AtomicU8,
    events: Mutex<Vec<(String, &'static str)>>,
    attempts: Mutex<Vec<String>>,
}
#[derive(Debug)]
struct Codec(Arc<Probe>);
impl Lucene104Codec for Codec {
    fn postings_writer_factory(&self) -> Option<&dyn PostingsWriterFactory> {
        Some(self)
    }
    fn postings_format_for_field(&self, name: &str) -> Lucene104PostingsFormat {
        if name == "aaa_suggest" {
            Lucene104PostingsFormat::completion104()
        } else if name == "p" {
            Lucene104PostingsFormat::new(2, 3).unwrap()
        } else {
            Lucene104PostingsFormat::default()
        }
    }
}
fn failure(message: &'static str) -> postings_writer::Error {
    lucene_store::Error::Io(std::io::Error::other(message)).into()
}
impl PostingsWriterFactory for Codec {
    fn fields_consumer<'a>(
        &self,
        state: PostingsWriteState<'a>,
    ) -> postings_writer::Result<Box<dyn PostingsWriteSink + 'a>> {
        let suffix = state.suffix.to_owned();
        self.0.events.lock().unwrap().push((suffix.clone(), "open"));
        self.0
            .attempts
            .lock()
            .unwrap()
            .push(format!("{}_{}", state.segment, suffix));
        let mode = self.0.mode.load(Ordering::SeqCst);
        if mode == 1 || (mode == 2 && suffix.ends_with("_1")) {
            return Err(failure("postings open"));
        }
        Ok(Box::new(Sink {
            delegate: AssertingPostingsCodec.fields_consumer(state)?,
            probe: self.0.clone(),
            suffix,
            mode,
        }))
    }
}
struct Sink<'a> {
    delegate: Box<dyn PostingsWriteSink + 'a>,
    probe: Arc<Probe>,
    suffix: String,
    mode: u8,
}
impl Sink<'_> {
    fn record(&self, event: &'static str) {
        self.probe
            .events
            .lock()
            .unwrap()
            .push((self.suffix.clone(), event));
    }
    fn last(&self) -> bool {
        self.suffix.ends_with("_1")
    }
}
impl PostingsWriteSink for Sink<'_> {
    fn write(
        &mut self,
        inputs: &[FieldPostingsInput<'_, &dyn TermSource>],
        norms: &[FieldNorms<'_>],
    ) -> postings_writer::Result<()> {
        self.record("write");
        if self.last() && matches!(self.mode, 3 | 6) {
            return Err(failure("postings write"));
        }
        self.delegate.write(inputs, norms)?;
        if self.last() && self.mode == 4 {
            return Err(failure("postings after write"));
        }
        Ok(())
    }
    fn merge(
        &mut self,
        inputs: &[FieldPostingsInput<'_, &dyn TermSource>],
        norms: &[FieldNorms<'_>],
    ) -> postings_writer::Result<()> {
        self.record("merge");
        if self.last() && self.mode == 7 {
            return Err(failure("postings merge"));
        }
        self.delegate.merge(inputs, norms)?;
        if self.last() && self.mode == 8 {
            return Err(failure("postings after merge"));
        }
        Ok(())
    }
    fn close(&mut self) -> postings_writer::Result<()> {
        self.record("close");
        self.delegate.close()?;
        if (self.last() && self.mode == 5) || self.mode == 6 {
            return Err(failure("postings secondary close"));
        }
        Ok(())
    }
}
fn writer<'a>(dir: &'a dyn Directory, probe: Arc<Probe>) -> IndexWriter<'a> {
    let mut writer = IndexWriter::open(
        dir,
        vec![],
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    writer.enable_explicit_documents().unwrap();
    writer.set_codec(Some(Arc::new(Codec(probe)))).unwrap();
    writer.set_use_compound_file(false);
    writer.set_max_full_flush_merge_wait_millis(0);
    let mut policy = LogDocMergePolicy::new();
    policy.set_merge_factor(1000).unwrap();
    writer.set_pluggable_merge_policy(Some(Arc::new(policy)));
    writer
}
fn add(writer: &mut IndexWriter<'_>, id: &str, value: &str) {
    let mut document = Document::new();
    document.add(StringField::new("id", id, Store::Yes));
    let mut ty = FieldType::new();
    ty.set_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
        .unwrap();
    ty.set_stored(true).unwrap();
    document.add(Field::from_string("p", value, ty).unwrap());
    writer.add_fields_document(&document).unwrap();
}
fn visible(dir: &dyn Directory) -> Vec<String> {
    let reader = DirectoryReader::open(dir).unwrap();
    let mut values = Vec::new();
    for segment in reader.segment_readers() {
        segment.check_integrity().unwrap();
        let field = segment
            .field_infos()
            .fields
            .iter()
            .find(|field| field.name == "p")
            .unwrap()
            .number;
        for doc in 0..segment.max_doc() {
            let document = segment.stored_document(doc).unwrap().unwrap();
            let value = &document
                .fields
                .iter()
                .find(|stored| stored.field_number == field)
                .unwrap()
                .value;
            let FieldValue::String(value) = value else {
                panic!("stored text expected")
            };
            values.push(value.clone());
        }
    }
    values.sort();
    values
}
fn committed_bytes(dir: &dyn Directory) -> Vec<(String, Vec<u8>)> {
    let infos = segment_infos::read_latest(dir).unwrap();
    let mut names: Vec<_> = infos
        .segments
        .iter()
        .flat_map(|s| {
            let si = dir.open(&format!("{}.si", s.segment_name)).unwrap();
            lucene_index::segment_info::parse(&si, &s.segment_id)
                .unwrap()
                .files
        })
        .collect();
    names.sort();
    names.dedup();
    names
        .into_iter()
        .map(|name| {
            let bytes = dir.open(&name).unwrap().to_vec();
            (name, bytes)
        })
        .collect()
}
fn clear(probe: &Probe) {
    probe.events.lock().unwrap().clear();
    probe.attempts.lock().unwrap().clear();
}
fn assert_no_outputs(dir: &dyn Directory, probe: &Probe) {
    for stem in probe.attempts.lock().unwrap().iter() {
        for extension in EXTS {
            assert!(
                dir.open(&format!("{stem}.{extension}")).is_err(),
                "{stem}.{extension}"
            );
        }
    }
}
fn assert_held_until_all_callbacks(probe: &Probe, operation: &str) {
    let events = probe.events.lock().unwrap();
    let first_close = events
        .iter()
        .position(|(_, action)| *action == "close")
        .unwrap();
    assert_eq!(
        events[..first_close]
            .iter()
            .filter(|(_, action)| *action == operation)
            .count(),
        2
    );
    assert_eq!(
        events[first_close..]
            .iter()
            .map(|(suffix, action)| (suffix.as_str(), *action))
            .collect::<Vec<_>>(),
        [("Lucene104_0", "close"), ("Lucene104_1", "close")]
    );
}
#[test]
fn later_group_flush_errors_close_prior_groups_preserve_primary_and_retry_original_documents() {
    for (mode, message) in [
        (1, "postings open"),
        (2, "postings open"),
        (3, "postings write"),
        (4, "postings after write"),
        (5, "postings secondary close"),
        (6, "postings write"),
    ] {
        let dir = ByteBuffersDirectory::new();
        let probe = Arc::new(Probe::default());
        let mut writer = writer(&dir, probe.clone());
        add(&mut writer, "a", "before");
        writer.commit().unwrap();
        let before = committed_bytes(&dir);
        clear(&probe);
        add(&mut writer, "b", "after");
        probe.mode.store(mode, Ordering::SeqCst);
        let error = writer.commit().unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(visible(&dir), ["before"]);
        assert_eq!(committed_bytes(&dir), before);
        assert_no_outputs(&dir, &probe);
        let closes = probe
            .events
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, action)| *action == "close")
            .count();
        assert_eq!(
            closes,
            match mode {
                1 => 0,
                2 => 1,
                _ => 2,
            }
        );
        clear(&probe);
        probe.mode.store(0, Ordering::SeqCst);
        writer.commit().unwrap();
        assert_eq!(visible(&dir), ["after", "before"]);
        assert_held_until_all_callbacks(&probe, "write");
    }
}
#[test]
fn every_native_output_open_and_close_failure_cleans_both_acquired_groups() {
    for operation in [FailureOp::CreateOutput, FailureOp::CloseOutput] {
        for extension in EXTS {
            let dir =
                MockDirectoryWrapper::new(&mut JavaRandom::new(67), ByteBuffersDirectory::new());
            let probe = Arc::new(Probe::default());
            let mut writer = writer(&dir, probe.clone());
            add(&mut writer, "a", "before");
            writer.commit().unwrap();
            let before = committed_bytes(&dir);
            clear(&probe);
            add(&mut writer, "b", "after");
            let fault = FailOn::once(move |context| {
                context.op == operation
                    && context.name.contains("_Lucene104_1.")
                    && context.name.ends_with(&format!(".{extension}"))
            });
            dir.fail_on(fault.clone());
            assert!(writer.commit().is_err());
            assert_eq!(fault.fired(), 1);
            assert_eq!(committed_bytes(&dir), before);
            assert_eq!(visible(&dir), ["before"]);
            assert_no_outputs(&dir, &probe);
            writer.commit().unwrap();
            assert_eq!(visible(&dir), ["after", "before"]);
        }
    }
}
#[test]
fn later_group_merge_errors_keep_sources_and_close_every_acquired_consumer() {
    for (mode, message) in [
        (2, "postings open"),
        (7, "postings merge"),
        (8, "postings after merge"),
        (5, "postings secondary close"),
    ] {
        let dir = ByteBuffersDirectory::new();
        let probe = Arc::new(Probe::default());
        let mut writer = writer(&dir, probe.clone());
        add(&mut writer, "a", "before");
        writer.commit().unwrap();
        add(&mut writer, "b", "after");
        writer.commit().unwrap();
        let before = committed_bytes(&dir);
        clear(&probe);
        probe.mode.store(mode, Ordering::SeqCst);
        let error = writer.force_merge(1).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(committed_bytes(&dir), before);
        assert_eq!(visible(&dir), ["after", "before"]);
        assert_no_outputs(&dir, &probe);
        clear(&probe);
        probe.mode.store(0, Ordering::SeqCst);
        writer.force_merge(1).unwrap();
        writer.commit().unwrap();
        assert_eq!(visible(&dir), ["after", "before"]);
        assert_held_until_all_callbacks(&probe, "merge");
    }
}
#[test]
fn later_group_constructor_collisions_preserve_unowned_bytes_and_remove_owned_mates() {
    for extension in EXTS {
        let dir = ByteBuffersDirectory::new();
        let probe = Arc::new(Probe::default());
        let mut writer = writer(&dir, probe.clone());
        add(&mut writer, "a", "before");
        writer.commit().unwrap();
        add(&mut writer, "b", "after");
        writer.commit().unwrap();
        let before = committed_bytes(&dir);
        clear(&probe);
        let infos = segment_infos::read_latest(&dir).unwrap();
        let name = format!(
            "_{}_Lucene104_1.{extension}",
            lucene_util::base36::to_base36(infos.counter)
        );
        let mut out = dir.create_output(&name).unwrap();
        out.write_bytes(b"other owner");
        out.close().unwrap();
        assert!(writer.force_merge(1).is_err());
        assert_eq!(&*dir.open(&name).unwrap(), b"other owner");
        assert_eq!(committed_bytes(&dir), before);
        assert_eq!(visible(&dir), ["after", "before"]);
        for stem in probe.attempts.lock().unwrap().iter() {
            for ext in EXTS {
                let candidate = format!("{stem}.{ext}");
                if candidate != name {
                    assert!(dir.open(&candidate).is_err());
                }
            }
        }
    }
}

#[test]
fn integrity_checks_each_original_postings_file_for_native_and_installed_groups() {
    #[derive(Debug)]
    struct NativeCodec;
    impl Lucene104Codec for NativeCodec {
        fn postings_format_for_field(&self, name: &str) -> Lucene104PostingsFormat {
            if name == "p" {
                Lucene104PostingsFormat::new(2, 3).unwrap()
            } else {
                Lucene104PostingsFormat::default()
            }
        }
    }
    for installed in [false, true] {
        let dir = ByteBuffersDirectory::new();
        let probe = Arc::new(Probe::default());
        let mut writer = writer(&dir, probe);
        if !installed {
            writer.set_codec(Some(Arc::new(NativeCodec))).unwrap();
        }
        add(&mut writer, "a", "before");
        writer.commit().unwrap();
        drop(writer);
        assert_eq!(visible(&dir), ["before"]);
        let files: Vec<_> = committed_bytes(&dir)
            .into_iter()
            .filter(|(name, _)| {
                [".doc", ".pos", ".pay"]
                    .iter()
                    .any(|ext| name.ends_with(ext))
            })
            .collect();
        assert_eq!(files.len(), 6);
        for (name, bytes) in files {
            lucene_store::codec_util::check_whole_file_footer(
                &bytes,
                bytes.len() - lucene_store::codec_util::FOOTER_LENGTH,
            )
            .unwrap();
            let mut corrupted = bytes.clone();
            // Change only the stored checksum, leaving headers and postings readable.
            *corrupted.last_mut().unwrap() ^= 1;
            dir.delete_file(&name).unwrap();
            let mut output = dir.create_output(&name).unwrap();
            output.write_bytes(&corrupted);
            output.close().unwrap();
            let reader = DirectoryReader::open(&dir).unwrap();
            assert!(
                reader.segment_readers()[0].check_integrity().is_err(),
                "unchecked original file {name}, installed={installed}"
            );
            drop(reader);
            dir.delete_file(&name).unwrap();
            let mut output = dir.create_output(&name).unwrap();
            output.write_bytes(&bytes);
            output.close().unwrap();
            assert_eq!(visible(&dir), ["before"]);
        }
    }
}

#[test]
fn merge_checks_every_source_postings_file_before_acquiring_any_consumer() {
    let dir = ByteBuffersDirectory::new();
    let probe = Arc::new(Probe::default());
    let mut writer = writer(&dir, probe.clone());
    add(&mut writer, "a", "before");
    writer.commit().unwrap();
    add(&mut writer, "b", "after");
    writer.commit().unwrap();
    let infos = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(infos.segments.len(), 2);
    let prefix = format!("{}_Lucene104_", infos.segments[0].segment_name);
    let files: Vec<_> = committed_bytes(&dir)
        .into_iter()
        .filter(|(name, _)| {
            name.starts_with(&prefix)
                && EXTS
                    .iter()
                    .any(|extension| name.ends_with(&format!(".{extension}")))
        })
        .collect();
    assert_eq!(files.len(), 14);
    for (name, bytes) in files {
        let mut corrupted = bytes.clone();
        *corrupted.last_mut().unwrap() ^= 1;
        dir.delete_file(&name).unwrap();
        let mut output = dir.create_output(&name).unwrap();
        output.write_bytes(&corrupted);
        output.close().unwrap();
        let source_bytes = committed_bytes(&dir);
        let names = dir.list_all().unwrap();
        clear(&probe);
        assert!(
            writer.force_merge(1).is_err(),
            "unchecked merge source {name}"
        );
        assert!(
            probe.events.lock().unwrap().is_empty(),
            "consumer acquired before source integrity check: {name}"
        );
        assert_eq!(
            committed_bytes(&dir),
            source_bytes,
            "failed merge changed committed source {name}"
        );
        assert_eq!(
            dir.list_all().unwrap(),
            names,
            "failed merge left owned outputs for {name}"
        );
        dir.delete_file(&name).unwrap();
        let mut output = dir.create_output(&name).unwrap();
        output.write_bytes(&bytes);
        output.close().unwrap();
        assert_eq!(visible(&dir), ["after", "before"]);
    }
    writer.force_merge(1).unwrap();
    writer.commit().unwrap();
    assert_eq!(visible(&dir), ["after", "before"]);
    assert_held_until_all_callbacks(&probe, "merge");
}

#[test]
fn missing_routed_group_is_rejected_when_another_group_survives() {
    let dir = ByteBuffersDirectory::new();
    let probe = Arc::new(Probe::default());
    let mut writer = writer(&dir, probe.clone());
    add(&mut writer, "a", "before");
    writer.commit().unwrap();
    add(&mut writer, "b", "after");
    writer.commit().unwrap();
    assert_eq!(visible(&dir), ["after", "before"]);
    let infos = segment_infos::read_latest(&dir).unwrap();
    let commit = &infos.segments[0];
    let si_name = format!("{}.si", commit.segment_name);
    let si_bytes = dir.open(&si_name).unwrap().to_vec();
    let mut si = lucene_index::segment_info::parse(&si_bytes, &commit.segment_id).unwrap();
    let removed_stem = format!("{}_Lucene104_1.", commit.segment_name);
    let removed: Vec<_> = si
        .files
        .iter()
        .filter(|file| file.starts_with(&removed_stem))
        .cloned()
        .collect();
    assert_eq!(removed.len(), 7);
    for name in &removed {
        dir.delete_file(name).unwrap();
    }
    // Model a writer that omitted the entire group from its valid .si, rather
    // than merely relying on the existing listed-file existence check.
    si.files.retain(|file| !removed.contains(file));
    assert!(si
        .files
        .iter()
        .any(|file| file.ends_with("_Lucene104_0.tmd")));
    dir.delete_file(&si_name).unwrap();
    let mut output = dir.create_output(&si_name).unwrap();
    output.write_bytes(&lucene_index::segment_info::write(&si, ""));
    output.close().unwrap();

    let checks = lucene_index::check_index::check_directory(&dir).unwrap();
    assert!(checks
        .iter()
        .any(|result| result.failures().iter().any(|check| {
            check.name == "fnm.postings_vs_files"
                && check.message.contains("missing postings group Lucene104_1")
        })));
    let error = DirectoryReader::open(&dir).expect_err("missing group must fail open");
    assert!(error
        .to_string()
        .contains("missing postings group Lucene104_1"));
    let before = committed_bytes(&dir);
    let names = dir.list_all().unwrap();
    clear(&probe);
    let error = writer.force_merge(1).unwrap_err();
    assert!(error
        .to_string()
        .contains("missing postings group Lucene104_1"));
    assert!(probe.events.lock().unwrap().is_empty());
    assert_eq!(committed_bytes(&dir), before);
    assert_eq!(dir.list_all().unwrap(), names);
}

#[test]
fn mixed_completion_keeps_ordinary_outputs_and_only_installs_native_groups_on_flush_and_merge() {
    use lucene_search::suggest::document::{lookup, SegmentView, TopSuggestDocsCollector};
    use lucene_search::suggest::suggest_field::SuggestField;
    use lucene_util::automaton::automata::make_binary;

    fn add_mixed(writer: &mut IndexWriter<'_>, id: &str, surface: &str, weight: i32) {
        let mut document = Document::new();
        document.add(StringField::new("id", id, Store::Yes));
        let mut ty = FieldType::new();
        ty.set_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
            .unwrap();
        ty.set_stored(true).unwrap();
        document.add(Field::from_string("p", surface, ty).unwrap());
        // This completion group sorts before either installed native group.
        document.add_boxed(Box::new(
            SuggestField::new("aaa_suggest", surface, weight).unwrap(),
        ));
        writer.add_fields_document(&document).unwrap();
    }
    fn suggestions(dir: &dyn Directory) -> Vec<(String, i32)> {
        let reader = DirectoryReader::open(dir).unwrap();
        let mut actual = Vec::new();
        for segment in reader.segment_readers() {
            segment.check_integrity().unwrap();
            let suggester = segment.suggester("aaa_suggest").unwrap().unwrap();
            assert!(segment.suggester("p").unwrap().is_none());
            let accept = |_: i32| true;
            let view = SegmentView {
                num_docs: segment.max_doc(),
                max_doc: segment.max_doc(),
                filtered: false,
                accept: &accept,
            };
            let mut collector = TopSuggestDocsCollector::new(10, false).unwrap();
            lookup(&suggester, &make_binary(b"al"), &view, &mut collector).unwrap();
            collector.finish();
            actual.extend(
                collector
                    .get()
                    .iter()
                    .map(|hit| (hit.key.clone(), hit.score as i32)),
            );
        }
        actual.sort();
        actual
    }
    fn native_scope(probe: &Probe, operation: &str) {
        assert_held_until_all_callbacks(probe, operation);
        let attempts = probe.attempts.lock().unwrap();
        assert_eq!(attempts.len(), 2);
        assert!(attempts.iter().all(|name| name.contains("_Lucene104_")));
        assert!(probe
            .events
            .lock()
            .unwrap()
            .iter()
            .all(|(suffix, _)| suffix.starts_with("Lucene104_")));
    }
    let dir = MockDirectoryWrapper::new(&mut JavaRandom::new(83), ByteBuffersDirectory::new());
    let probe = Arc::new(Probe::default());
    let mut writer = writer(&dir, probe.clone());
    add_mixed(&mut writer, "a", "alpha", 7);
    writer.commit().unwrap();
    native_scope(&probe, "write");
    assert_eq!(suggestions(&dir), [("alpha".into(), 7)]);
    let before = committed_bytes(&dir);
    clear(&probe);
    add_mixed(&mut writer, "b", "alpine", 11);
    let fault = FailOn::once(|context| {
        context.op == FailureOp::CreateOutput && context.name.ends_with("_Completion104_0.cmp")
    });
    dir.fail_on(fault.clone());
    assert!(writer.commit().is_err());
    assert_eq!(fault.fired(), 1);
    native_scope(&probe, "write");
    assert_eq!(committed_bytes(&dir), before);
    assert_eq!(suggestions(&dir), [("alpha".into(), 7)]);
    assert_no_outputs(&dir, &probe);
    clear(&probe);
    writer.commit().unwrap();
    native_scope(&probe, "write");
    assert_eq!(visible(&dir), ["alpha", "alpine"]);
    assert_eq!(
        suggestions(&dir),
        [("alpha".into(), 7), ("alpine".into(), 11)]
    );
    let before = committed_bytes(&dir);
    clear(&probe);
    probe.mode.store(8, Ordering::SeqCst);
    assert!(writer
        .force_merge(1)
        .unwrap_err()
        .to_string()
        .contains("postings after merge"));
    assert_eq!(committed_bytes(&dir), before);
    assert_no_outputs(&dir, &probe);
    clear(&probe);
    probe.mode.store(0, Ordering::SeqCst);
    writer.force_merge(1).unwrap();
    writer.commit().unwrap();
    native_scope(&probe, "merge");
    assert_eq!(
        DirectoryReader::open(&dir).unwrap().segment_readers().len(),
        1
    );
    assert_eq!(visible(&dir), ["alpha", "alpine"]);
    assert_eq!(
        suggestions(&dir),
        [("alpha".into(), 7), ("alpine".into(), 11)]
    );
    let files = dir.list_all().unwrap();
    for extension in ["lkp", "cmp"] {
        assert_eq!(
            files
                .iter()
                .filter(|name| name.ends_with(&format!("_Completion104_0.{extension}")))
                .count(),
            1
        );
    }
}
