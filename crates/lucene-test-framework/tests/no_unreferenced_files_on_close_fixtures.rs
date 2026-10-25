//! Actual Java close outcomes, cleanup phases and surviving index files.
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use lucene_index::deletion_policy::NoDeletionPolicy;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::index_output::FsIndexOutput;
use lucene_store::lock::Lock;
use lucene_store::{ByteBuffersDirectory, DataOutput, Directory, Error, Input};
use lucene_test_framework::store::base_directory_wrapper::BaseDirectoryWrapper;
use lucene_test_framework::store::mock_directory_wrapper::{MockDirectoryWrapper, Throttling};
use lucene_util::java_random::JavaRandom;

const DATA: &str =
    include_str!("../../../fixtures/testframework/data/no_unreferenced_files_on_close.tsv");
type Files = BTreeMap<String, Vec<u8>>;

fn unhex(text: &str) -> Vec<u8> {
    assert_eq!(text.len() % 2, 0);
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}
fn text(encoded: &str) -> String {
    String::from_utf8(unhex(encoded)).unwrap()
}
fn names(encoded: &str) -> BTreeSet<String> {
    encoded
        .split(',')
        .filter(|name| !name.is_empty())
        .map(text)
        .collect()
}
fn source(kind: &str) -> Files {
    if kind == "none" {
        return Files::new();
    }
    let relative = if kind == "parent_binary" {
        "misc/bp_blocks".into()
    } else if kind == "soft_delete_binary" {
        "field_infos_index".into()
    } else if let Some(stage) = kind.strip_prefix("simpletext_") {
        format!("codecs_simpletext/codec/{stage}/index")
    } else {
        "merge_policies/commits_index".into()
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data")
        .join(relative);
    let mut files: Files = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (
                entry.file_name().into_string().unwrap(),
                std::fs::read(entry.path()).unwrap(),
            )
        })
        .filter(|(name, _)| {
            !matches!(kind, "parent_binary" | "soft_delete_binary")
                || (name != "manifest.properties" && !name.ends_with(".raw"))
        })
        .collect();
    if kind == "latest" {
        files.remove("segments_1");
        files.remove("segments_2");
    }
    if kind == "empty_latest" {
        let row = include_str!("../../../fixtures/testframework/data/corrupt_unknown_files.tsv")
            .lines()
            .find(|line| line.starts_with("snapshot\tsegments_4\t"))
            .unwrap();
        files.insert("segments_4".into(), unhex(row.split('\t').nth(2).unwrap()));
    }
    files
}
fn input_files(kind: &str, operation: &str) -> Files {
    let mut files = source(kind);
    match operation {
        "orphan_codec" | "fault_delete" => {
            files.insert("_z.orphan".into(), vec![1]);
        }
        "orphan_pending" => {
            files.insert("pending_segments_orphan".into(), vec![1]);
        }
        "decoys" => {
            files.insert("README".into(), vec![]);
            files.insert("_guardwrite.lock".into(), vec![]);
        }
        "name_edges" => {
            for name in [
                "_z.x\n",
                "_z_x\r.foo",
                "_z.x\u{85}",
                "_z.x\u{2028}",
                "_z.x\u{2029}",
                "_z.x\u{b}",
                "_z.x\u{c}",
                "pending_segments_tail\n",
            ] {
                files.insert(name.into(), vec![1]);
            }
        }
        "exact_segments" => {
            files.insert("segments".into(), vec![1, 2, 3]);
            files.insert("_z.orphan".into(), vec![1]);
        }
        "malformed_segments" => {
            files.insert("segments_?".into(), vec![1, 2, 3]);
        }
        "corrupt_latest" => {
            files.insert("segments_3".into(), vec![1, 2, 3]);
        }
        "corrupt_old" => {
            files.insert("segments_1".into(), vec![1, 2, 3]);
        }
        "missing_si" => {
            files.remove("_0.si");
        }
        "corrupt_si" => {
            files.insert("_0.si".into(), vec![1, 2, 3]);
        }
        "missing_fnm" => {
            files.remove("_0.fnm");
            files.insert("_z.orphan".into(), vec![1]);
        }
        "corrupt_fnm" => {
            files.insert("_0.fnm".into(), vec![1, 2, 3]);
            files.insert("_z.orphan".into(), vec![1]);
        }
        "missing_live" => {
            files.remove("_0_1.liv");
        }
        "corrupt_live" => {
            files.insert("_0_1.liv".into(), vec![1, 2, 3]);
        }
        "clean" | "rates" | "leak_input" | "wrapper_lock" | "delegate_lock" | "fault_listing"
        | "fault_lock" | "fault_read" | "added_on_rollback" => (),
        _ => panic!("unknown Java input operation {operation}"),
    }
    files
}
fn write(dir: &dyn Directory, name: &str, bytes: &[u8]) -> lucene_store::Result<()> {
    let mut out = dir.create_output(name)?;
    out.write_bytes(bytes);
    out.close().map(drop)
}
fn num_docs(dir: &dyn Directory) -> i32 {
    DirectoryReader::open(dir)
        .map(|reader| reader.num_docs())
        .unwrap_or(-1)
}

#[derive(Default, Debug)]
struct Events {
    phase: usize,
    faults: usize,
    removed: BTreeMap<usize, BTreeSet<String>>,
    added: BTreeMap<usize, BTreeSet<String>>,
}
struct Trace {
    inner: Arc<ByteBuffersDirectory>,
    events: Arc<Mutex<Events>>,
    armed: AtomicBool,
    operation: String,
}
impl Trace {
    fn fault(&self, location: &str) -> lucene_store::Result<()> {
        let mut events = self.events.lock().unwrap();
        if self.armed.load(Ordering::SeqCst)
            && events.faults == 0
            && self.operation == format!("fault_{location}")
        {
            events.faults += 1;
            return Err(Error::Io(std::io::Error::other(format!(
                "fixture fault: {location}"
            ))));
        }
        Ok(())
    }
}
#[derive(Debug)]
struct TraceLock {
    delegate: Box<dyn Lock>,
    inner: Arc<ByteBuffersDirectory>,
    events: Arc<Mutex<Events>>,
    added_on_close: bool,
    closed: AtomicBool,
}
impl Lock for TraceLock {
    fn ensure_valid(&self) -> lucene_store::Result<()> {
        self.delegate.ensure_valid()
    }
    fn close(&self) -> lucene_store::Result<()> {
        if self.closed.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        self.delegate.close()?;
        if self.added_on_close {
            write(self.inner.as_ref(), "_added.after", &[])?;
            self.events
                .lock()
                .unwrap()
                .added
                .entry(1)
                .or_default()
                .insert("_added.after".into());
        }
        Ok(())
    }
}
impl Drop for TraceLock {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
impl Directory for Trace {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.fault("listing")?;
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> lucene_store::Result<Input> {
        let phase = self.events.lock().unwrap().phase;
        if phase > 0 && name.ends_with(".si") {
            self.fault("read")?;
        }
        self.inner.open(name)
    }
    fn create_output(&self, name: &str) -> lucene_store::Result<FsIndexOutput> {
        self.inner.create_output(name)
    }
    fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
        self.inner.sync(names)
    }
    fn rename(&self, source: &str, dest: &str) -> lucene_store::Result<()> {
        self.inner.rename(source, dest)
    }
    fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
        if name == "_z.orphan" {
            self.fault("delete")?;
        }
        self.inner.delete_file(name)?;
        if self.armed.load(Ordering::SeqCst) {
            let mut events = self.events.lock().unwrap();
            let phase = events.phase;
            events.removed.entry(phase).or_default().insert(name.into());
        }
        Ok(())
    }
    fn sync_meta_data(&self) -> lucene_store::Result<()> {
        self.inner.sync_meta_data()
    }
    fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn Lock>> {
        let phase = if self.armed.load(Ordering::SeqCst) {
            let mut events = self.events.lock().unwrap();
            events.phase += 1;
            events.phase
        } else {
            0
        };
        self.fault("lock")?;
        let delegate = self.inner.obtain_lock(name)?;
        Ok(Box::new(TraceLock {
            delegate,
            inner: self.inner.clone(),
            events: self.events.clone(),
            added_on_close: phase == 1 && self.operation == "added_on_rollback",
            closed: AtomicBool::new(false),
        }))
    }
}
fn row<'a>(lines: &[&'a str], tag: &str, stage: Option<&str>) -> Vec<&'a str> {
    let mut found = lines
        .iter()
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .filter(|fields| fields[0] == tag && stage.is_none_or(|stage| fields[2] == stage));
    let result = found
        .next()
        .unwrap_or_else(|| panic!("missing {tag} {stage:?}"));
    assert!(found.next().is_none(), "duplicate {tag} {stage:?}");
    result
}
// Rust close returns String. Preserve Java's exception class/message in the
// oracle and compare success/failure plus the stable lock/leak/fault categories.
fn outcome(result: &Result<(), String>, expected: &[&str], label: &str) {
    assert_eq!(
        if result.is_ok() { "ok" } else { "failure" },
        expected[3],
        "{label}: {result:?}"
    );
    if let Err(message) = result {
        let java = text(expected[5]);
        if expected[4] == "LockObtainFailedException" {
            assert!(message.contains("lock obtain failed"), "{label}: {message}");
        }
        if java.contains("open files:") {
            assert!(message.contains("open files:"), "{label}: {message}");
        }
        for marker in [
            "unreferenced files",
            "fixture fault:",
            "cannot delete after crash",
            "parent document field",
            "soft-deletes",
            "cannot change field",
        ] {
            if java.contains(marker) {
                assert!(message.contains(marker), "{label}: {message}, Java: {java}");
            }
        }
    }
}
fn run_case(lines: &[&str]) {
    let config: Vec<_> = lines[0].split('\t').collect();
    assert_eq!(config.len(), 6);
    let label = config[1];
    let operation = config[5];
    let files = input_files(config[2], operation);
    let inner = Arc::new(ByteBuffersDirectory::new());
    for (name, bytes) in &files {
        write(inner.as_ref(), name, bytes).unwrap();
    }
    let trace = Arc::new(Trace {
        inner: inner.clone(),
        events: Arc::new(Mutex::new(Events::default())),
        armed: AtomicBool::new(false),
        operation: operation.into(),
    });
    let dir = MockDirectoryWrapper::new(&mut JavaRandom::new(31), trace.clone());
    dir.set_check_index_on_close(config[4].parse().unwrap());
    dir.set_use_slow_open_closers(false);
    dir.set_throttling(Throttling::Never);
    match config[3] {
        "on" => dir.set_assert_no_unreferenced_files_on_close(true),
        "off" => dir.set_assert_no_unreferenced_files_on_close(false),
        "default" => (),
        _ => panic!("bad assertion flag"),
    }
    let leaked = (operation == "leak_input").then(|| dir.open("_0.si").unwrap());
    let held = match operation {
        "wrapper_lock" => Some(dir.obtain_lock("write.lock").unwrap()),
        "delegate_lock" => Some(trace.obtain_lock("write.lock").unwrap()),
        _ => None,
    };
    let before = row(lines, "state", Some("before"));
    assert_eq!(
        files.keys().cloned().collect::<BTreeSet<_>>(),
        names(before[3]),
        "{label}"
    );
    assert_eq!(
        num_docs(inner.as_ref()),
        before[4].parse::<i32>().unwrap(),
        "{label}"
    );
    if operation == "rates" {
        dir.set_random_io_exception_rate(1.0);
        dir.set_random_io_exception_rate_on_open(1.0);
    }
    trace.armed.store(true, Ordering::SeqCst);
    let result = dir.close();
    outcome(&result, &row(lines, "outcome", Some("first")), label);
    let after = row(lines, "state", Some("after"));
    if let Err(message) = &result {
        if message.contains("unreferenced files") {
            let start = names(before[3]);
            let end = names(after[3]);
            for changed in start.symmetric_difference(&end) {
                assert!(
                    message.contains(changed) || message.contains(&format!("{changed:?}")),
                    "{label}: changed filename {changed:?} missing from {message}"
                );
            }
        }
    }
    assert_eq!(
        inner
            .list_all()
            .unwrap()
            .into_iter()
            .collect::<BTreeSet<_>>(),
        names(after[3]),
        "{label}"
    );
    assert_eq!(
        num_docs(inner.as_ref()),
        after[4].parse::<i32>().unwrap(),
        "{label}"
    );
    let observed = row(lines, "trace", None);
    {
        let events = trace.events.lock().unwrap();
        assert_eq!(
            events.phase,
            observed[2].parse::<usize>().unwrap(),
            "{label}"
        );
        assert_eq!(
            events.faults,
            observed[3].parse::<usize>().unwrap(),
            "{label}"
        );
        for phase in 0..=events.phase {
            let stage = phase.to_string();
            assert_eq!(
                events.removed.get(&phase).cloned().unwrap_or_default(),
                names(row(lines, "removed", Some(&stage))[3]),
                "{label}, phase {phase}"
            );
            assert_eq!(
                events.added.get(&phase).cloned().unwrap_or_default(),
                names(row(lines, "added", Some(&stage))[3]),
                "{label}, phase {phase}"
            );
        }
    }
    // Java closes its delegate in finally. Rust Directory closes by Drop, so
    // these counts document Java behavior without inventing a Rust close hook.
    assert_eq!(
        dir.is_open(),
        observed[6].parse::<bool>().unwrap(),
        "{label}"
    );
    assert_eq!(
        dir.crashed(),
        observed[4].parse::<usize>().unwrap() > 0,
        "{label}"
    );
    assert_eq!(
        dir.random_io_exception_rate(),
        observed[7].parse::<f64>().unwrap(),
        "{label}"
    );
    assert_eq!(
        dir.random_io_exception_rate_on_open(),
        observed[8].parse::<f64>().unwrap(),
        "{label}"
    );
    drop(leaked);
    drop(held);
    let second = dir.close();
    outcome(&second, &row(lines, "outcome", Some("second")), label);
    let repeated = row(lines, "repeat", None);
    let events = trace.events.lock().unwrap();
    assert_eq!(
        events.phase,
        repeated[2].parse::<usize>().unwrap(),
        "{label}"
    );
    assert_eq!(
        events.faults,
        repeated[3].parse::<usize>().unwrap(),
        "{label}"
    );
    assert_eq!(
        dir.crashed(),
        repeated[4].parse::<usize>().unwrap() > 0,
        "{label}"
    );
    assert_eq!(
        inner
            .list_all()
            .unwrap()
            .into_iter()
            .collect::<BTreeSet<_>>(),
        names(repeated[6]),
        "{label}"
    );
    assert_eq!(
        num_docs(inner.as_ref()),
        repeated[7].parse::<i32>().unwrap(),
        "{label}"
    );
}
#[test]
fn close_matches_actual_java_cleanup_and_failure_phases() {
    let mut group = Vec::new();
    let mut cases = 0;
    let mut seen = BTreeSet::new();
    for line in DATA.lines() {
        if line.starts_with("rollback_probe\t") {
            continue;
        }
        if line.starts_with("case\t") {
            if !group.is_empty() {
                run_case(&group);
                cases += 1;
                group.clear();
            }
            assert!(seen.insert(line.split('\t').nth(1).unwrap()));
        }
        group.push(line);
    }
    assert!(!group.is_empty());
    run_case(&group);
    cases += 1;
    assert_eq!(cases, 36);
}

fn run_rollback_probe(expected: &[&str]) {
    assert_eq!(expected.len(), 9);
    let operation = expected[1];
    assert_eq!(expected[2], "failure");
    assert_eq!(expected[7], "1");
    assert_eq!(
        expected[5], expected[6],
        "failed rollback must preserve committed files"
    );
    let inner = Arc::new(ByteBuffersDirectory::new());
    for (name, bytes) in source("latest") {
        write(inner.as_ref(), &name, &bytes).unwrap();
    }
    assert_eq!(
        inner
            .list_all()
            .unwrap()
            .into_iter()
            .collect::<BTreeSet<_>>(),
        names(expected[5])
    );
    let trace = Trace {
        inner: inner.clone(),
        events: Arc::new(Mutex::new(Events::default())),
        armed: AtomicBool::new(false),
        operation: "fault_listing".into(),
    };
    let latest = lucene_util::version::Version::LATEST;
    let version = LuceneVersion {
        major: latest.major(),
        minor: latest.minor(),
        bugfix: latest.bugfix(),
    };
    let result = {
        let mut writer = IndexWriter::open_with_deletion_policy(
            &trace,
            Vec::new(),
            "Lucene104",
            version,
            Arc::new(NoDeletionPolicy),
        )
        .unwrap();
        trace.armed.store(true, Ordering::SeqCst);
        match operation {
            "rollback" => {
                let result = writer.try_rollback().map_err(|error| error.to_string());
                // The compatibility writer keeps its lock until scope exit,
                // including after an error. The helper drops it before returning.
                assert!(inner.obtain_lock("write.lock").is_err());
                result
            }
            "close_without_commit" => {
                writer.set_commit_on_close(false);
                let result = writer.close().map_err(|error| error.to_string());
                // Consuming close releases the writer immediately on failure.
                let lock = inner.obtain_lock("write.lock");
                assert_eq!(lock.is_ok(), expected[8].parse::<bool>().unwrap());
                drop(lock);
                result
            }
            _ => panic!("unknown rollback probe {operation}"),
        }
    };
    outcome(
        &result,
        &[
            "outcome",
            "rollback_probe",
            "first",
            expected[2],
            expected[3],
            expected[4],
        ],
        "rollback_probe",
    );
    assert_eq!(
        trace.events.lock().unwrap().faults,
        expected[7].parse::<usize>().unwrap()
    );
    let lock = inner.obtain_lock("write.lock");
    assert_eq!(lock.is_ok(), expected[8].parse::<bool>().unwrap());
    drop(lock);
    assert_eq!(
        inner
            .list_all()
            .unwrap()
            .into_iter()
            .collect::<BTreeSet<_>>(),
        names(expected[6])
    );
}

#[test]
fn rollback_and_close_propagate_refresh_failure_and_release_the_writer_lock() {
    let probes: Vec<_> = DATA
        .lines()
        .filter(|line| line.starts_with("rollback_probe\t"))
        .collect();
    assert_eq!(probes.len(), 2);
    let mut seen = BTreeSet::new();
    for probe in probes {
        let expected: Vec<_> = probe.split('\t').collect();
        assert!(seen.insert(expected[1]));
        run_rollback_probe(&expected);
    }
    assert_eq!(seen, BTreeSet::from(["rollback", "close_without_commit"]));
}
