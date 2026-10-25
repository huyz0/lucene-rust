//! Actual Lucene cranky wrappers: callback traces, RNG tails and production bytes.
use std::cell::RefCell;
use std::rc::Rc;

use lucene_codecs::norms::{self, NormsField};
use lucene_store::{ByteBuffersDirectory, DataOutput, Directory};
use lucene_test_framework::codecs::cranky_norms::{
    CrankyNormsConsumer, CrankyNormsFormat, Lucene90NormsFormat, NormsConsumer, NormsFormat,
    NormsState,
};
use lucene_util::java_random::JavaRandom;

fn failure(message: &str) -> norms::Error {
    lucene_store::Error::Io(std::io::Error::other(message.to_string())).into()
}

#[derive(Default)]
struct Counts {
    opens: usize,
    reads: usize,
    adds: usize,
    closes: usize,
    fail: bool,
}
struct Recording(Rc<RefCell<Counts>>);
struct Consumer(Rc<RefCell<Counts>>);
impl NormsConsumer for Consumer {
    fn add_norms_field(&mut self, _: NormsField<'_>) -> norms::Result<()> {
        let mut c = self.0.borrow_mut();
        c.adds += 1;
        if c.fail {
            Err(failure("delegate add"))
        } else {
            Ok(())
        }
    }
    fn close(&mut self) -> norms::Result<()> {
        let mut c = self.0.borrow_mut();
        c.closes += 1;
        if c.fail {
            Err(failure("delegate close"))
        } else {
            Ok(())
        }
    }
}
impl NormsFormat for Recording {
    type Consumer = Consumer;
    type Producer = Rc<RefCell<Counts>>;
    fn norms_consumer(&mut self, _: &NormsState<'_>) -> norms::Result<Consumer> {
        let mut c = self.0.borrow_mut();
        c.opens += 1;
        if c.fail {
            Err(failure("delegate open"))
        } else {
            Ok(Consumer(self.0.clone()))
        }
    }
    fn norms_producer(&mut self, _: &NormsState<'_>) -> norms::Result<Self::Producer> {
        let mut c = self.0.borrow_mut();
        c.reads += 1;
        if c.fail {
            Err(failure("delegate read"))
        } else {
            Ok(self.0.clone())
        }
    }
}

#[test]
fn actual_java_callback_trace_matches_error_order_and_rng_tails() {
    let fixture = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/testframework/data/cranky_norms.tsv"
    ));
    let directory = ByteBuffersDirectory::new();
    let state = NormsState {
        directory: &directory,
        segment: "_s",
        id: &[0; 16],
        suffix: "",
        max_doc: 1,
    };
    let random = Rc::new(RefCell::new(JavaRandom::new(0)));
    let counts = Rc::new(RefCell::new(Counts::default()));
    let mut format = CrankyNormsFormat::new(Recording(counts.clone()), random.clone());
    let mut consumer = CrankyNormsConsumer::new(Consumer(counts.clone()), random.clone());
    let mut seed = None;
    let (mut rows, mut tails, mut failures) = (0, 0, 0);
    for line in fixture.lines() {
        let f: Vec<_> = line.split('\t').collect();
        let next: i64 = f[0].parse().unwrap();
        if seed != Some(next) {
            random.borrow_mut().set_seed(next);
            *counts.borrow_mut() = Counts::default();
            seed = Some(next);
        }
        if f[1] == "tail" {
            assert_eq!(random.borrow_mut().next_int().to_string(), f[2]);
            tails += 1;
            continue;
        }
        assert_eq!(f.len(), 9);
        format.delegate_mut().0.borrow_mut().fail = f[3].parse().unwrap();
        let result = match f[2] {
            "open" => format.norms_consumer(&state).map(|_| ()),
            "read" => format
                .norms_producer(&state)
                .map(|producer| assert!(Rc::ptr_eq(&producer, &counts))),
            "add" => consumer.add_norms_field(NormsField::Dense(0, &[3])),
            "close" => consumer.close(),
            _ => panic!("unexpected operation"),
        };
        let status = result.map_or_else(|e| e.to_string(), |_| "ok".into());
        assert_eq!(status, f[4], "seed={next}, step={}", f[1]);
        failures += usize::from(status.starts_with("Fake IOException"));
        let c = format.delegate().0.borrow();
        assert_eq!(
            [c.opens, c.reads, c.adds, c.closes],
            [f[5], f[6], f[7], f[8]].map(|s| s.parse::<usize>().unwrap())
        );
        rows += 1;
    }
    assert_eq!((rows, tails), (3584, 7));
    assert!(failures > 0);
}

const META: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../fixtures/testframework/data/cranky_norms.nvm"
));
const DATA: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../fixtures/testframework/data/cranky_norms.nvd"
));
const EMPTY_META: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../fixtures/testframework/data/cranky_norms_empty.nvm"
));
const EMPTY_DATA: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../fixtures/testframework/data/cranky_norms_empty.nvd"
));

fn publish(directory: &ByteBuffersDirectory, file: &str, bytes: &[u8]) {
    let mut out = directory.create_output(file).unwrap();
    out.write_bytes(bytes);
    out.close().unwrap();
}

#[test]
fn concrete_delegate_reproduces_java_dense_sparse_and_empty_bytes() {
    for empty in [false, true] {
        let directory = ByteBuffersDirectory::new();
        let id = std::array::from_fn(|i| i as u8);
        let state = NormsState {
            directory: &directory,
            segment: "_s",
            id: &id,
            suffix: "s",
            max_doc: 5,
        };
        let random = Rc::new(RefCell::new(JavaRandom::new(0)));
        let mut format = CrankyNormsFormat::new(Lucene90NormsFormat, random.clone());
        let mut consumer = format.norms_consumer(&state).unwrap();
        if !empty {
            consumer
                .add_norms_field(NormsField::Dense(0, &[-129, 0, 127, 128, 32768]))
                .unwrap();
            consumer
                .add_norms_field(NormsField::Sparse(1, &[(0, 7), (2, 8), (4, 9)]))
                .unwrap();
        }
        consumer.close().unwrap();
        let (meta, data) = if empty {
            (EMPTY_META, EMPTY_DATA)
        } else {
            (META, DATA)
        };
        assert_eq!(&*directory.open("_s_s.nvm").unwrap(), meta);
        assert_eq!(&*directory.open("_s_s.nvd").unwrap(), data);
        let producer = format.norms_producer(&state).unwrap();
        producer.check_integrity().unwrap();
        if empty {
            assert!(producer.metadata.entries.is_empty());
            consumer.close().unwrap();
            let status = consumer
                .add_norms_field(NormsField::Dense(0, &[-129, 0, 127, 128, 32768]))
                .unwrap_err()
                .to_string();
            let lifecycle = include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../fixtures/testframework/data/cranky_norms_lifecycle.tsv"
            ));
            assert_eq!(
                format!(
                    "close\tok\nclose\tok\nadd\t{status}\ntail\t{}\n",
                    random.borrow_mut().next_int()
                ),
                lifecycle
            );
        } else {
            for (doc, value) in [-129, 0, 127, 128, 32768].into_iter().enumerate() {
                assert_eq!(producer.value(0, doc as i32).unwrap(), Some(value));
                assert_eq!(
                    producer.value(1, doc as i32).unwrap(),
                    match doc {
                        0 => Some(7),
                        2 => Some(8),
                        4 => Some(9),
                        _ => None,
                    }
                );
            }
            assert!(producer.value(0, -1).is_err());
            assert!(producer.value(0, 5).is_err());
            assert!(producer.value(9, 0).is_err());
        }
    }
}

#[test]
fn actual_java_bytes_reject_wrong_id_corrupt_metadata_and_data_checksum() {
    let directory = ByteBuffersDirectory::new();
    let id = std::array::from_fn(|i| i as u8);
    publish(&directory, "_s_s.nvm", META);
    publish(&directory, "_s_s.nvd", DATA);
    let state = NormsState {
        directory: &directory,
        segment: "_s",
        id: &id,
        suffix: "s",
        max_doc: 5,
    };
    let mut format = Lucene90NormsFormat;
    assert!(format
        .norms_producer(&NormsState {
            id: &[99; 16],
            ..state
        })
        .is_err());
    assert!(format
        .norms_producer(&NormsState {
            suffix: "wrong",
            ..state
        })
        .is_err());
    let mut bad = META.to_vec();
    bad[0] ^= 1;
    publish(&directory, "_precedence_s.nvm", &bad);
    let expected = norms::parse_meta(&bad, &id, "s").unwrap_err().to_string();
    assert_eq!(
        format
            .norms_producer(&NormsState {
                segment: "_precedence",
                ..state
            })
            .err()
            .unwrap()
            .to_string(),
        expected
    );
    let mut checksum = META.to_vec();
    *checksum.last_mut().unwrap() ^= 1;
    publish(&directory, "_checksum_s.nvm", &checksum);
    let expected = norms::parse_meta(&checksum, &id, "s")
        .unwrap_err()
        .to_string();
    assert_eq!(
        format
            .norms_producer(&NormsState {
                segment: "_checksum",
                ..state
            })
            .err()
            .unwrap()
            .to_string(),
        expected
    );
    publish(&directory, "_bad_s.nvm", &bad);
    publish(&directory, "_bad_s.nvd", DATA);
    assert!(format
        .norms_producer(&NormsState {
            segment: "_bad",
            ..state
        })
        .is_err());
    publish(&directory, "_crc_s.nvm", META);
    let mut bad = DATA.to_vec();
    bad[60] ^= 1;
    publish(&directory, "_crc_s.nvd", &bad);
    let producer = format
        .norms_producer(&NormsState {
            segment: "_crc",
            ..state
        })
        .unwrap();
    assert!(producer.check_integrity().is_err());
    publish(&directory, "_short_s.nvm", META);
    publish(&directory, "_short_s.nvd", &DATA[..12]);
    assert!(format
        .norms_producer(&NormsState {
            segment: "_short",
            ..state
        })
        .is_err());
}

struct FailingDirectory {
    inner: ByteBuffersDirectory,
    extension: String,
    calls: std::sync::Mutex<Vec<String>>,
}

impl Directory for FailingDirectory {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> lucene_store::Result<lucene_store::Input> {
        self.inner.open(name)
    }
    fn create_output(&self, name: &str) -> lucene_store::Result<lucene_store::FsIndexOutput> {
        self.calls.lock().unwrap().push(name.into());
        if name.ends_with(&format!(".{}", self.extension)) {
            return Err(lucene_store::Error::Io(std::io::Error::other(format!(
                "create {}",
                self.extension
            ))));
        }
        self.inner.create_output(name)
    }
    fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
        self.inner.sync(names)
    }
    fn rename(&self, source: &str, destination: &str) -> lucene_store::Result<()> {
        self.inner.rename(source, destination)
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
fn actual_java_production_open_failure_order_matches() {
    let fixture = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/testframework/data/cranky_norms_open_failures.tsv"
    ));
    for line in fixture.lines() {
        let f: Vec<_> = line.split('\t').collect();
        let directory = FailingDirectory {
            inner: ByteBuffersDirectory::new(),
            extension: f[0].into(),
            calls: Default::default(),
        };
        let random = Rc::new(RefCell::new(JavaRandom::new(0)));
        let mut format = CrankyNormsFormat::new(Lucene90NormsFormat, random.clone());
        let state = NormsState {
            directory: &directory,
            segment: "_fail",
            id: &[0; 16],
            suffix: "",
            max_doc: 5,
        };
        assert_eq!(
            format.norms_consumer(&state).err().unwrap().to_string(),
            f[2]
        );
        assert_eq!(directory.calls.lock().unwrap().join(","), f[1]);
        assert_eq!(random.borrow_mut().next_int().to_string(), f[3]);
        if f[0] == "nvm" {
            assert_eq!(
                &*directory.open("_fail.nvd").unwrap(),
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../fixtures/testframework/data/cranky_norms_open_failure.nvd"
                ))
            );
        }
    }
    assert_eq!(fixture.lines().count(), 2);
}
