//! Actual mock-directory close: both writer passes, readers, lists and locks.
use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lucene_search::directory_reader::DirectoryReader;
use lucene_store::{ByteBuffersDirectory, DataOutput, Directory};
use lucene_test_framework::store::base_directory_wrapper::BaseDirectoryWrapper;
use lucene_test_framework::store::mock_directory_wrapper::{MockDirectoryWrapper, Throttling};
use lucene_util::java_random::JavaRandom;

type Files = BTreeMap<String, Vec<u8>>;
fn digest(files: &BTreeSet<String>, docs: i32) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for name in files {
        for byte in name.bytes().chain(std::iter::once(0)) {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
        }
    }
    (hash ^ docs as u64).wrapping_mul(0x100_0000_01b3)
}
fn source(path: &Path) -> Files {
    std::fs::read_dir(path)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            assert!(entry.file_type().unwrap().is_file());
            (
                entry.file_name().into_string().unwrap(),
                std::fs::read(entry.path()).unwrap(),
            )
        })
        .collect()
}
fn run(
    files: &Files,
    expected_files: &BTreeSet<String>,
    expected_docs: i32,
    expected_error: &str,
    expected_change: Option<bool>,
    millis: u64,
) -> (u128, u64) {
    let wall = Instant::now();
    let mut nanos = 0;
    let mut count = 0;
    loop {
        let memory = Arc::new(ByteBuffersDirectory::new());
        for (name, bytes) in files {
            let mut output = memory.create_output(name).unwrap();
            output.write_bytes(bytes);
            output.close().unwrap();
        }
        let dir = MockDirectoryWrapper::new(&mut JavaRandom::new(42), Arc::clone(&memory));
        dir.set_check_index_on_close(false);
        dir.set_assert_no_unreferenced_files_on_close(true);
        dir.set_use_slow_open_closers(false);
        dir.set_throttling(Throttling::Never);
        let start = Instant::now();
        let result = dir.close();
        nanos += start.elapsed().as_nanos();
        if expected_error.is_empty() {
            result.unwrap();
        } else {
            let error = result.expect_err("expected parent-field close failure");
            assert!(
                error.contains(expected_error),
                "wrong close failure: {error}"
            );
        }
        let after: BTreeSet<_> = memory.list_all().unwrap().into_iter().collect();
        assert_eq!(&after, expected_files, "final file set");
        if let Some(changed) = expected_change {
            let before: BTreeSet<_> = files.keys().cloned().collect();
            assert_ne!(changed, before == after, "retained commits must be pruned");
        }
        let docs = DirectoryReader::open(memory.as_ref()).unwrap().num_docs();
        assert_eq!(docs, expected_docs, "document count");
        black_box(digest(&after, docs));
        // Backing-store and wrapper destruction are outside timing in both
        // engines. All writer/reader/lock cleanup within close is timed.
        drop(dir);
        drop(memory);
        count += 1;
        if wall.elapsed() >= Duration::from_millis(millis) {
            break;
        }
    }
    (nanos, count)
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let root = Path::new(&args[1]);
    let warmup = args[2].parse().unwrap();
    let measure = args[3].parse().unwrap();
    assert!(measure > 0, "measurement duration must be positive");
    for stage in [
        "latest",
        "retained",
        "simpletext_flush",
        "simpletext_merged",
    ] {
        let expected_files =
            std::fs::read_to_string(root.join(format!("{stage}-expected-files.txt")))
                .unwrap()
                .lines()
                .map(str::to_owned)
                .collect();
        let expected_docs =
            std::fs::read_to_string(root.join(format!("{stage}-expected-docs.txt")))
                .unwrap()
                .trim()
                .parse()
                .unwrap();
        let files = source(&root.join(stage));
        let expected_error =
            std::fs::read_to_string(root.join(format!("{stage}-expected-error.txt"))).unwrap();
        let retained = stage == "retained";
        let expected_change = (!stage.starts_with("simpletext_")).then_some(retained);
        if retained {
            assert!(files.contains_key("segments_1") && files.contains_key("segments_2"));
        }
        run(
            &files,
            &expected_files,
            expected_docs,
            &expected_error,
            expected_change,
            warmup,
        );
        let (nanos, count) = run(
            &files,
            &expected_files,
            expected_docs,
            &expected_error,
            expected_change,
            measure,
        );
        let name = format!("no_unreferenced_close_{stage}");
        println!("{name}\t{:.3}\t{count}", nanos as f64 / count as f64);
        println!(
            "#check\t{name}\t{}\t{}\t{expected_docs}\t{}",
            digest(&expected_files, expected_docs),
            expected_files.len(),
            if expected_error.is_empty() {
                "ok"
            } else {
                "failure"
            }
        );
    }
}
