//! Real segment-info reads and writes, paired with CrankySegmentInfoMicro.java.
use std::cell::RefCell;
use std::hint::black_box;
use std::rc::Rc;
use std::time::{Duration, Instant};

use lucene_index::segment_info::{LuceneVersion, SegmentInfo};
use lucene_store::{ByteBuffersDirectory, Directory};
use lucene_test_framework::codecs::cranky_segment_info::{
    CrankySegmentInfoFormat, Lucene99SegmentInfoFormat, SegmentInfoFormatDelegate,
};
use lucene_util::java_random::JavaRandom;

const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
fn hash(digest: u64, value: u64) -> u64 {
    (digest ^ value).wrapping_mul(0x100_0000_01b3)
}
fn info() -> SegmentInfo {
    SegmentInfo {
        id: *b"0123456789abcdef",
        version: LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
        min_version: None,
        doc_count: 7,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: vec![],
        attributes: vec![],
        index_sort: None,
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup: u64 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(500);
    let measure: u64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let random = Rc::new(RefCell::new(JavaRandom::new(42)));
    let mut format = CrankySegmentInfoFormat::new(Lucene99SegmentInfoFormat, random.clone());
    let owned_bytes = std::mem::size_of_val(&format)
        + 2 * std::mem::size_of::<usize>()
        + std::mem::size_of::<RefCell<JavaRandom>>();
    println!("cranky_segment_info_owned_bytes\t{owned_bytes}");
    let base = ByteBuffersDirectory::new();
    Lucene99SegmentInfoFormat
        .write(&base, "_base", &mut info())
        .unwrap();
    for write in [false, true] {
        let case = if write {
            "cranky_segment_info_write"
        } else {
            "cranky_segment_info_read"
        };
        let mut batch = || {
            *random.borrow_mut() = JavaRandom::new(42);
            let directory = ByteBuffersDirectory::new();
            let mut digest = OFFSET;
            for op in 0..256 {
                if write {
                    let name = format!("_s{op}");
                    let mut info = info();
                    match format.write(&directory, &name, &mut info) {
                        Ok(()) => {
                            let bytes = directory.open(&format!("{name}.si")).unwrap();
                            digest = hash(digest, info.files.len() as u64);
                            for byte in bytes.iter() {
                                digest = hash(digest, u64::from(*byte));
                            }
                        }
                        Err(err) => {
                            assert_eq!(
                                err.to_string(),
                                "Fake IOException from SegmentInfoFormat.write()"
                            );
                            assert!(info.files.is_empty());
                            digest = hash(digest, u64::MAX);
                        }
                    }
                } else {
                    let read = format.read(&base, "_base", &info().id).unwrap();
                    digest = hash(
                        hash(digest, u64::try_from(read.doc_count).unwrap()),
                        read.files.len() as u64,
                    );
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
