//! Real Lucene 10.5.0 cranky callback/error traces and `.fnm` bytes.

use std::cell::RefCell;
use std::rc::Rc;

use lucene_codecs::field_infos::{self, FieldInfo, FieldInfos, IndexOptions};
use lucene_store::{ByteBuffersDirectory, DataOutput, Directory};
use lucene_test_framework::codecs::cranky_field_infos::{
    CrankyFieldInfosFormat, FieldInfosFormat, Lucene94FieldInfosFormat,
};
use lucene_util::java_random::JavaRandom;

#[derive(Default)]
struct Recording {
    reads: RefCell<usize>,
    writes: usize,
    fail: bool,
}

impl FieldInfosFormat for Recording {
    fn read(
        &self,
        _: &dyn Directory,
        _: &str,
        _: &[u8; 16],
        suffix: &str,
    ) -> field_infos::Result<FieldInfos> {
        assert_eq!(suffix, "suffix");
        *self.reads.borrow_mut() += 1;
        if self.fail {
            Err(lucene_store::Error::Io(std::io::Error::other("delegate read")).into())
        } else {
            FieldInfos::new(vec![])
        }
    }

    fn write(
        &mut self,
        _: &dyn Directory,
        _: &str,
        _: &[u8; 16],
        suffix: &str,
        infos: &FieldInfos,
    ) -> field_infos::Result<()> {
        assert_eq!(suffix, "suffix");
        assert!(infos.fields.is_empty());
        self.writes += 1;
        if self.fail {
            Err(lucene_store::Error::Io(std::io::Error::other("delegate write")).into())
        } else {
            Ok(())
        }
    }
}

#[test]
fn cranky_field_infos_callback_trace_matches_actual_java_wrapper() {
    let fixture = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/testframework/data/cranky_field_infos.tsv"
    ));
    let directory = ByteBuffersDirectory::new();
    let infos = FieldInfos::new(vec![]).unwrap();
    let mut current_seed = None;
    let random = Rc::new(RefCell::new(JavaRandom::new(0)));
    let mut format = CrankyFieldInfosFormat::new(Recording::default(), random.clone());
    let mut rows = 0;
    let mut tails = 0;
    for line in fixture.lines() {
        let fields: Vec<_> = line.split('\t').collect();
        let seed: i64 = fields[0].parse().unwrap();
        if current_seed != Some(seed) {
            random.borrow_mut().set_seed(seed);
            format = CrankyFieldInfosFormat::new(Recording::default(), random.clone());
            current_seed = Some(seed);
        }
        if fields[1] == "tail" {
            assert_eq!(fields.len(), 3);
            assert_eq!(random.borrow_mut().next_int().to_string(), fields[2]);
            tails += 1;
            continue;
        }
        assert_eq!(fields.len(), 7);
        format.delegate_mut().fail = fields[3].parse().unwrap();
        let status = match fields[2] {
            "read" => format
                .read(&directory, "_cfi", &[0; 16], "suffix")
                .map(|read| {
                    assert!(read.fields.is_empty());
                }),
            "write" => format.write(&directory, "_cfi", &[0; 16], "suffix", &infos),
            _ => panic!("unknown operation"),
        }
        .map_or_else(|error| error.to_string(), |_| "ok".into());
        assert_eq!(status, fields[4], "seed={seed}, step={}", fields[1]);
        assert_eq!(format.delegate().reads.borrow().to_string(), fields[5]);
        assert_eq!(format.delegate().writes.to_string(), fields[6]);
        rows += 1;
    }
    assert_eq!((rows, tails), (7 * 256, 7));
}

#[test]
fn cranky_field_infos_concrete_delegate_reads_and_reproduces_java_bytes() {
    let bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/testframework/data/cranky_field_infos.fnm"
    ));
    let directory = ByteBuffersDirectory::new();
    let mut output = directory.create_output("_cfi_s.fnm").unwrap();
    output.write_bytes(bytes);
    output.close().unwrap();
    let id = std::array::from_fn(|i| i as u8);
    let mut format = CrankyFieldInfosFormat::new(
        Lucene94FieldInfosFormat,
        Rc::new(RefCell::new(JavaRandom::new(0))),
    );
    let read = format.read(&directory, "_cfi", &id, "s").unwrap();
    assert_eq!(read.fields.len(), 1);
    assert_eq!(read.fields[0].name, "field");
    assert_eq!(read.fields[0].index_options, IndexOptions::Docs);
    let infos = FieldInfos::new(vec![
        FieldInfo::new("field", 0).with_index_options(IndexOptions::Docs)
    ])
    .unwrap();
    format.write(&directory, "_rust", &id, "s", &infos).unwrap();
    assert_eq!(&*directory.open("_rust_s.fnm").unwrap(), bytes);
    let mut damaged = bytes.to_vec();
    damaged[0] ^= 1;
    let mut output = directory.create_output("_bad_s.fnm").unwrap();
    output.write_bytes(&damaged);
    output.close().unwrap();
    assert!(format.read(&directory, "_bad", &id, "s").is_err());
}
