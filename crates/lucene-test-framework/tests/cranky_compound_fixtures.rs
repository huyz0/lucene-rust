//! Replays the actual Java cranky wrapper over Lucene90 compound archives.
use std::cell::RefCell;
use std::rc::Rc;

use lucene_codecs::compound_format;
use lucene_index::compound_reader::CompoundReader;
use lucene_store::{ByteBuffersDirectory, DataOutput, Directory};
use lucene_test_framework::codecs::cranky_compound::{
    CompoundFormat, CompoundInfo, CrankyCompoundFormat, Lucene90CompoundFormat,
};
use lucene_util::java_random::JavaRandom;

const ID: [u8; 16] = *b"0123456789abcdef";
const IDS: [&str; 3] = [".a", ".bb", "_Codec_0.cc"];
const INPUTS: [&[u8]; 3] = [
    include_bytes!("../../../fixtures/testframework/data/cranky_compound_input_0.bin"),
    include_bytes!("../../../fixtures/testframework/data/cranky_compound_input_1.bin"),
    include_bytes!("../../../fixtures/testframework/data/cranky_compound_input_2.bin"),
];
const CFS: &[u8] = include_bytes!("../../../fixtures/testframework/data/cranky_compound.cfs");
const CFE: &[u8] = include_bytes!("../../../fixtures/testframework/data/cranky_compound.cfe");

fn put(dir: &dyn Directory, name: &str, bytes: &[u8]) {
    let mut output = dir.create_output(name).unwrap();
    output.write_bytes(bytes);
    output.close().unwrap();
}
fn inputs(dir: &dyn Directory, name: &str) -> Vec<String> {
    IDS.iter()
        .zip(INPUTS)
        .map(|(suffix, bytes)| {
            let name = format!("{name}{suffix}");
            put(dir, &name, bytes);
            name
        })
        .collect()
}
fn info<'a>(name: &'a str, files: &'a [String]) -> CompoundInfo<'a> {
    CompoundInfo {
        name,
        id: &ID,
        files,
    }
}
fn digest(reader: &CompoundReader<'_>) -> u64 {
    let mut digest = 0xcbf29ce484222325_u64;
    let mut hash = |value: u64| {
        digest = (digest ^ value).wrapping_mul(0x100000001b3);
    };
    let mut names = reader.member_files();
    names.sort();
    for name in names {
        for value in name.bytes() {
            hash(u64::from(value));
        }
        let bytes = reader.open(&name).unwrap();
        hash(bytes.len() as u64);
        for &value in bytes.iter() {
            hash(u64::from(value));
        }
    }
    reader.check_integrity().unwrap();
    digest
}
#[derive(Default)]
struct State {
    reads: usize,
    writes: usize,
    fail_read: bool,
    fail_write: bool,
}
struct Recording {
    state: Rc<RefCell<State>>,
}
impl CompoundFormat for Recording {
    fn get_compound_reader<'d>(
        &mut self,
        dir: &'d dyn Directory,
        info: &CompoundInfo<'_>,
    ) -> lucene_store::Result<CompoundReader<'d>> {
        let fail = {
            let mut state = self.state.borrow_mut();
            state.reads += 1;
            state.fail_read
        };
        if fail {
            return Err(lucene_store::Error::Io(std::io::Error::other(
                "delegate read failure",
            )));
        }
        Lucene90CompoundFormat.get_compound_reader(dir, info)
    }
    fn write(
        &mut self,
        dir: &dyn Directory,
        info: &CompoundInfo<'_>,
    ) -> compound_format::Result<()> {
        let fail = {
            let mut state = self.state.borrow_mut();
            state.writes += 1;
            state.fail_write
        };
        if fail {
            return Err(
                lucene_store::Error::Io(std::io::Error::other("delegate write failure")).into(),
            );
        }
        Lucene90CompoundFormat.write(dir, info)
    }
}

#[test]
fn java_shared_rng_and_actual_compound_delegation() {
    let mut lines =
        include_str!("../../../fixtures/testframework/data/cranky_compound.tsv").lines();
    let mut operations = 0;
    for seed in [0, 1, 42, -1, i64::MIN, i64::MAX] {
        for failures in [false, true] {
            let random = Rc::new(RefCell::new(JavaRandom::new(seed)));
            let state = Rc::new(RefCell::new(State::default()));
            let mut formats = [
                CrankyCompoundFormat::new(
                    Recording {
                        state: state.clone(),
                    },
                    random.clone(),
                ),
                CrankyCompoundFormat::new(
                    Recording {
                        state: state.clone(),
                    },
                    random.clone(),
                ),
            ];
            let base = ByteBuffersDirectory::new();
            let base_files = inputs(&base, "_base");
            let base_info = info("_base", &base_files);
            Lucene90CompoundFormat.write(&base, &base_info).unwrap();
            assert_eq!(&*base.open("_base.cfs").unwrap(), CFS);
            assert_eq!(&*base.open("_base.cfe").unwrap(), CFE);
            for op in 0..256 {
                {
                    let mut state = state.borrow_mut();
                    state.fail_read = failures && op % 21 == 0;
                    state.fail_write = failures && op % 19 == 0;
                }
                let target = ByteBuffersDirectory::new();
                let files = inputs(&target, "_write");
                let original = files.clone();
                let read = op % 3 == 0;
                let mut checksum = "-".to_string();
                let result = if read {
                    match formats[op & 1].get_compound_reader(&base, &base_info) {
                        Ok(reader) => {
                            checksum = digest(&reader).to_string();
                            "ok".to_string()
                        }
                        Err(error) => error.to_string(),
                    }
                } else {
                    formats[op & 1]
                        .write(&target, &info("_write", &files))
                        .map_or_else(|error| error.to_string(), |_| "ok".to_string())
                };
                let names = target.list_all().unwrap();
                let state = state.borrow();
                let line = format!(
                    "op\t{seed}\t{failures}\t{op}\t{}\t{result}\t{}\t{}\t{}\t{}\t{}\t{checksum}",
                    if read { "read" } else { "write" },
                    state.reads,
                    state.writes,
                    names.iter().any(|name| name == "_write.cfs"),
                    names.iter().any(|name| name == "_write.cfe"),
                    files == original
                );
                assert_eq!(
                    Some(line.as_str()),
                    lines.next(),
                    "seed={seed} failures={failures} op={op}"
                );
                operations += 1;
            }
            let line = format!(
                "rng\t{seed}\t{failures}\t{}",
                random.borrow_mut().next_int()
            );
            assert_eq!(Some(line.as_str()), lines.next());
        }
    }
    assert_eq!(operations, 3072);
    for kind in [
        "reader_bad_cfe",
        "reader_short_cfs",
        "writer_missing_input",
        "writer_bad_footer",
        "writer_bad_id",
    ] {
        let dir = ByteBuffersDirectory::new();
        let files = inputs(&dir, "_bad");
        let metadata = info("_bad", &files);
        if kind.starts_with("reader") {
            Lucene90CompoundFormat.write(&dir, &metadata).unwrap();
            let name = if kind == "reader_bad_cfe" {
                "_bad.cfe"
            } else {
                "_bad.cfs"
            };
            let mut bytes = dir.open(name).unwrap().to_vec();
            if kind == "reader_bad_cfe" {
                bytes[0] ^= 1;
            } else {
                bytes.pop();
            }
            dir.delete_file(name).unwrap();
            put(&dir, name, &bytes);
        } else if kind == "writer_missing_input" {
            dir.delete_file("_bad.a").unwrap();
        } else {
            let mut bytes = dir.open("_bad.a").unwrap().to_vec();
            let offset = if kind == "writer_bad_footer" {
                bytes.len() - 1
            } else {
                4 + 1 + "CrankyCompoundInput".len() + 4
            };
            bytes[offset] ^= 1;
            dir.delete_file("_bad.a").unwrap();
            put(&dir, "_bad.a", &bytes);
        }
        let random = Rc::new(RefCell::new(JavaRandom::new(42)));
        let mut format = CrankyCompoundFormat::new(Lucene90CompoundFormat, random.clone());
        if kind.starts_with("reader") {
            assert!(
                format.get_compound_reader(&dir, &metadata).is_err(),
                "{kind}"
            );
        } else {
            let error = format.write(&dir, &metadata).unwrap_err();
            if kind == "writer_missing_input" {
                assert!(
                    matches!(error, compound_format::Error::Store(lucene_store::Error::Io(ref e)) if e.kind() == std::io::ErrorKind::NotFound)
                );
            } else {
                assert!(!matches!(
                    error,
                    compound_format::Error::Store(lucene_store::Error::Io(_))
                ));
            }
        }
        let names = dir.list_all().unwrap();
        let line = format!(
            "negative\t{kind}\t{}\t{}\t{}",
            random.borrow_mut().next_int(),
            names.iter().any(|name| name == "_bad.cfs"),
            names.iter().any(|name| name == "_bad.cfe")
        );
        assert_eq!(Some(line.as_str()), lines.next());
    }
    assert!(lines.next().is_none());
}

#[test]
fn real_java_archives_match_writer_and_reader_bytes() {
    let dir = ByteBuffersDirectory::new();
    let files = inputs(&dir, "_base");
    let metadata = info("_base", &files);
    let random = Rc::new(RefCell::new(JavaRandom::new(42)));
    let mut format = CrankyCompoundFormat::new(Lucene90CompoundFormat, random);
    format.write(&dir, &metadata).unwrap();
    assert_eq!(&*dir.open("_base.cfs").unwrap(), CFS);
    assert_eq!(&*dir.open("_base.cfe").unwrap(), CFE);
    let reader = format.get_compound_reader(&dir, &metadata).unwrap();
    let mut expected: Vec<_> = files.clone();
    expected.sort();
    let mut names = reader.member_files();
    names.sort();
    assert_eq!(names, expected);
    for (name, bytes) in files.iter().zip(INPUTS) {
        assert_eq!(&*reader.open(name).unwrap(), bytes);
    }
    reader.check_integrity().unwrap();
    let empty = ByteBuffersDirectory::new();
    format.write(&empty, &info("_empty", &[])).unwrap();
    assert_eq!(
        &*empty.open("_empty.cfs").unwrap(),
        include_bytes!("../../../fixtures/testframework/data/cranky_compound_empty.cfs")
    );
    assert_eq!(
        &*empty.open("_empty.cfe").unwrap(),
        include_bytes!("../../../fixtures/testframework/data/cranky_compound_empty.cfe")
    );
    let reader = format
        .get_compound_reader(&empty, &info("_empty", &[]))
        .unwrap();
    assert!(reader.member_files().is_empty());
    reader.check_integrity().unwrap();
}

struct LaterOpenFailure {
    inner: ByteBuffersDirectory,
    opened: std::sync::Mutex<Vec<String>>,
    lengths: std::sync::atomic::AtomicUsize,
}
impl Directory for LaterOpenFailure {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.inner.list_all()
    }
    fn file_length(&self, name: &str) -> lucene_store::Result<u64> {
        self.lengths
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.file_length(name)
    }
    fn open(&self, name: &str) -> lucene_store::Result<lucene_store::Input> {
        self.opened.lock().unwrap().push(name.into());
        if name == "_bad.bb" {
            return Err(lucene_store::Error::Io(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "later open",
            )));
        }
        self.inner.open(name)
    }
    fn create_output(&self, name: &str) -> lucene_store::Result<lucene_store::FsIndexOutput> {
        self.inner.create_output(name)
    }
    fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
        self.inner.sync(names)
    }
    fn rename(&self, from: &str, to: &str) -> lucene_store::Result<()> {
        self.inner.rename(from, to)
    }
    fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
        self.inner.delete_file(name)
    }
    fn sync_meta_data(&self) -> lucene_store::Result<()> {
        self.inner.sync_meta_data()
    }
    fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn lucene_store::Lock>> {
        self.inner.obtain_lock(name)
    }
}

#[test]
fn java_validates_each_member_before_opening_the_next() {
    let fixture =
        include_str!("../../../fixtures/testframework/data/cranky_compound_failure_order.tsv");
    for line in fixture.lines() {
        let fields: Vec<_> = line.split('\t').collect();
        let kind = fields[0];
        let dir = LaterOpenFailure {
            inner: ByteBuffersDirectory::new(),
            opened: Default::default(),
            lengths: Default::default(),
        };
        let files = inputs(&dir.inner, "_bad");
        if kind != "valid" {
            let mut bytes = dir.inner.open("_bad.a").unwrap().to_vec();
            let offset = if kind == "checksum" {
                bytes.len() - 1
            } else {
                4 + 1 + "CrankyCompoundInput".len() + 4
            };
            bytes[offset] ^= 1;
            dir.inner.delete_file("_bad.a").unwrap();
            put(&dir.inner, "_bad.a", &bytes);
        }
        let random = Rc::new(RefCell::new(JavaRandom::new(42)));
        let mut format = CrankyCompoundFormat::new(Lucene90CompoundFormat, random.clone());
        let error = format.write(&dir, &info("_bad", &files)).unwrap_err();
        let result = match error {
            compound_format::Error::Store(lucene_store::Error::Io(error)) => {
                assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
                assert_eq!(error.to_string(), "later open");
                "later open"
            }
            compound_format::Error::Store(lucene_store::Error::Corrupted(_)) => "corruption",
            error => panic!("unexpected validation error: {error}"),
        };
        let actual = format!(
            "{kind}\t{result}\t{}\t{}\t{}",
            dir.lengths.load(std::sync::atomic::Ordering::Relaxed),
            dir.opened.lock().unwrap().join(","),
            random.borrow_mut().next_int()
        );
        assert_eq!(actual, line);
        // All failure exits must still close both memory outputs.
        assert!(!dir.inner.open("_bad.cfs").unwrap().is_empty());
        assert!(!dir.inner.open("_bad.cfe").unwrap().is_empty());
    }
    assert_eq!(fixture.lines().count(), 3);
}
