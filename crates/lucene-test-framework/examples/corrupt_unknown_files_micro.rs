//! Actual unknown-file selection over Java-produced retained commits; no random damage.
use lucene_store::{ByteBuffersDirectory, DataOutput, Directory};
use lucene_test_framework::store::base_directory_wrapper::BaseDirectoryWrapper;
use lucene_test_framework::store::mock_directory_wrapper::{MockDirectoryWrapper, Throttling};
use lucene_util::java_random::JavaRandom;
use std::collections::BTreeSet;
use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};

const BASIS: u64 = 0xcbf2_9ce4_8422_2325;
fn digest(files: &BTreeSet<String>) -> (u64, usize) {
    let mut hash = BASIS;
    for name in files {
        for byte in name.bytes().chain(std::iter::once(0)) {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
        }
    }
    (hash, files.len())
}
fn run(dir: &MockDirectoryWrapper, expected: (u64, usize), millis: u64) -> (u128, u64) {
    let deadline = Duration::from_millis(millis);
    let wall = Instant::now();
    let mut nanos = 0;
    let mut count = 0;
    loop {
        let mut result = None;
        let start = Instant::now();
        dir.corrupt_unknown_files_with(|files| {
            result = Some(digest(files));
            Ok(())
        })
        .unwrap();
        nanos += start.elapsed().as_nanos();
        let result = result.expect("corruption callback must run");
        assert_eq!(result, expected, "victim selection mismatch");
        black_box(result);
        count += 1;
        if wall.elapsed() >= deadline {
            break;
        }
    }
    (nanos, count)
}
fn prepare(snapshot: &Path, orphans: usize) -> (MockDirectoryWrapper, (u64, usize)) {
    let memory = ByteBuffersDirectory::new();
    for file in std::fs::read_dir(snapshot).unwrap() {
        let file = file.unwrap();
        assert!(file.file_type().unwrap().is_file());
        let name = file.file_name().into_string().unwrap();
        let bytes = std::fs::read(file.path()).unwrap();
        let mut output = memory.create_output(&name).unwrap();
        output.write_bytes(&bytes);
        output.close().unwrap();
    }
    let victims: BTreeSet<_> = (0..orphans)
        .map(|i| {
            if i % 2 == 0 {
                format!("_orphan{i:06}.si")
            } else {
                format!("pending_segments_orphan{i:06}")
            }
        })
        .collect();
    for name in victims
        .iter()
        .map(String::as_str)
        .chain(["README", "_guardwrite.lock"])
    {
        let mut output = memory.create_output(name).unwrap();
        output.write_bytes(&[1]);
        output.close().unwrap();
    }
    let dir = MockDirectoryWrapper::new(&mut JavaRandom::new(42), memory);
    dir.set_check_index_on_close(false);
    dir.set_use_slow_open_closers(false);
    dir.set_throttling(Throttling::Never);
    (dir, digest(&victims))
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let snapshot = Path::new(&args[1]);
    let warmup = args[2].parse().unwrap();
    let measure = args[3].parse().unwrap();
    assert!(measure > 0, "measurement duration must be positive");
    for orphans in [8, 2048] {
        let (dir, expected) = prepare(snapshot, orphans);
        run(&dir, expected, warmup);
        let (nanos, count) = run(&dir, expected, measure);
        let name = if orphans == 8 {
            "corrupt_unknown_small"
        } else {
            "corrupt_unknown_large"
        };
        println!("{name}\t{:.3}\t{count}", nanos as f64 / count as f64);
        println!("#check\t{name}\t{}\t{}", expected.0, expected.1);
        dir.close().unwrap();
    }
}
