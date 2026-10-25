//! Actual compound lifecycle paired with CrankyCompoundMicro.java.
use std::cell::RefCell;
use std::hint::black_box;
use std::rc::Rc;
use std::time::{Duration, Instant};

use lucene_store::{ByteBuffersDirectory, DataOutput, Directory};
use lucene_test_framework::codecs::cranky_compound::{
    CompoundFormat, CompoundInfo, CrankyCompoundFormat, Lucene90CompoundFormat, FAKE_WRITE_ERROR,
};
use lucene_util::java_random::JavaRandom;

const ID: [u8; 16] = *b"0123456789abcdef";
const IDS: [&str; 3] = [".a", ".bb", "_Codec_0.cc"];
const INPUTS: [&[u8]; 3] = [
    include_bytes!("../../../fixtures/testframework/data/cranky_compound_input_0.bin"),
    include_bytes!("../../../fixtures/testframework/data/cranky_compound_input_1.bin"),
    include_bytes!("../../../fixtures/testframework/data/cranky_compound_input_2.bin"),
];
fn hash(digest: u64, value: u64) -> u64 {
    (digest ^ value).wrapping_mul(0x100_0000_01b3)
}
fn put(dir: &dyn Directory, name: &str, bytes: &[u8]) {
    let mut output = dir.create_output(name).unwrap();
    output.write_bytes(bytes);
    output.close().unwrap();
}
fn consume(mut digest: u64, dir: &dyn Directory, name: &str) -> u64 {
    let bytes = dir.open(name).unwrap();
    digest = hash(digest, bytes.len() as u64);
    for &byte in bytes.iter() {
        digest = hash(digest, u64::from(byte));
    }
    digest
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup: u64 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(500);
    let measure: u64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let random = Rc::new(RefCell::new(JavaRandom::new(42)));
    let mut format = CrankyCompoundFormat::new(Lucene90CompoundFormat, random.clone());
    let owned_bytes = std::mem::size_of_val(&format)
        + 2 * std::mem::size_of::<usize>()
        + std::mem::size_of::<RefCell<JavaRandom>>();
    println!("cranky_compound_owned_bytes\t{owned_bytes}");
    let files: Vec<String> = IDS.iter().map(|suffix| format!("_base{suffix}")).collect();
    let metadata = CompoundInfo {
        name: "_base",
        id: &ID,
        files: &files,
    };
    let base = ByteBuffersDirectory::new();
    put(
        &base,
        "_base.cfs",
        include_bytes!("../../../fixtures/testframework/data/cranky_compound.cfs"),
    );
    put(
        &base,
        "_base.cfe",
        include_bytes!("../../../fixtures/testframework/data/cranky_compound.cfe"),
    );
    for write in [false, true] {
        let case = if write {
            "cranky_compound_write"
        } else {
            "cranky_compound_read"
        };
        let mut batch = || {
            *random.borrow_mut() = JavaRandom::new(42);
            let directory = ByteBuffersDirectory::new();
            if write {
                for (name, bytes) in files.iter().zip(INPUTS) {
                    put(&directory, name, bytes);
                }
            }
            let mut digest = 0xcbf2_9ce4_8422_2325;
            for _ in 0..256 {
                if write {
                    match format.write(&directory, &metadata) {
                        Ok(()) => {
                            digest = hash(digest, metadata.files.len() as u64);
                            for name in ["_base.cfs", "_base.cfe"] {
                                digest = consume(digest, &directory, name);
                                directory.delete_file(name).unwrap();
                            }
                        }
                        Err(error) => {
                            assert_eq!(error.to_string(), FAKE_WRITE_ERROR);
                            assert!(!directory
                                .list_all()
                                .unwrap()
                                .iter()
                                .any(|name| name.ends_with(".cfs") || name.ends_with(".cfe")));
                            digest = hash(digest, u64::MAX);
                        }
                    }
                } else {
                    let reader = format.get_compound_reader(&base, &metadata).unwrap();
                    let mut names = reader.member_files();
                    names.sort();
                    for name in names {
                        for byte in name.bytes() {
                            digest = hash(digest, u64::from(byte));
                        }
                        digest = consume(digest, &reader, &name);
                    }
                    reader.check_integrity().unwrap();
                }
            }
            black_box(digest)
        };
        let expected = batch();
        for (phase, budget) in [warmup, measure].into_iter().enumerate() {
            let start = Instant::now();
            let mut operations = 0_u64;
            loop {
                assert_eq!(batch(), expected);
                operations += 256;
                if start.elapsed() >= Duration::from_millis(budget) {
                    break;
                }
            }
            if phase == 1 {
                println!(
                    "{case}\t{:.3}\t{operations}",
                    start.elapsed().as_nanos() as f64 / operations as f64
                );
            }
        }
        println!("#check\t{case}\t{expected}\t256");
    }
}
