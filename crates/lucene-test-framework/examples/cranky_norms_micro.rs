//! Real norms lifecycle/read benchmark; Java twin: `CrankyNormsMicro.java`.
use std::cell::RefCell;
use std::hint::black_box;
use std::rc::Rc;
use std::time::{Duration, Instant};

use lucene_codecs::norms::NormsField;
use lucene_store::{ByteBuffersDirectory, Directory};
use lucene_test_framework::codecs::cranky_norms::{
    CrankyNormsFormat, Lucene90NormsFormat, NormsConsumer, NormsFormat, NormsState, FAKE_ADD_ERROR,
    FAKE_CLOSE_ERROR, FAKE_OPEN_ERROR,
};
use lucene_util::java_random::JavaRandom;

const DENSE: [i64; 5] = [-129, 0, 127, 128, 32768];
const SPARSE: [(i32, i64); 3] = [(0, 7), (2, 8), (4, 9)];
const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

fn state(directory: &dyn Directory) -> NormsState<'_> {
    NormsState {
        directory,
        segment: "_cn",
        id: &ID,
        suffix: "s",
        max_doc: 5,
    }
}

fn digest_step(digest: u64, value: u64) -> u64 {
    digest.wrapping_mul(31).wrapping_add(value)
}

fn batch(saved: &dyn Directory, write: bool) -> u64 {
    let random = Rc::new(RefCell::new(JavaRandom::new(black_box(0))));
    let mut format = CrankyNormsFormat::new(Lucene90NormsFormat, random.clone());
    let mut digest = 0_u64;
    for _ in 0..256 {
        if write {
            let directory = ByteBuffersDirectory::new();
            let mut consumer = match format.norms_consumer(&state(&directory)) {
                Ok(consumer) => consumer,
                Err(error) => {
                    assert_eq!(error.to_string(), FAKE_OPEN_ERROR);
                    digest = digest_step(digest, 101);
                    assert!(directory.list_all().unwrap().is_empty());
                    continue;
                }
            };
            for field in [NormsField::Dense(0, &DENSE), NormsField::Sparse(1, &SPARSE)] {
                if let Err(error) = consumer.add_norms_field(black_box(field)) {
                    assert_eq!(error.to_string(), FAKE_ADD_ERROR);
                    digest = digest_step(digest, 103);
                }
            }
            if let Err(error) = consumer.close() {
                assert_eq!(error.to_string(), FAKE_CLOSE_ERROR);
                digest = digest_step(digest, 107);
            }
            for file in ["_cn_s.nvm", "_cn_s.nvd"] {
                for &byte in directory.open(file).unwrap().iter() {
                    digest = digest_step(digest, u64::from(byte));
                }
            }
        } else {
            let producer = format.norms_producer(&state(saved)).unwrap();
            for field in 0..2 {
                for doc in 0..5 {
                    let value = producer.value(field, doc).unwrap().unwrap_or(-777);
                    digest = digest_step(digest, value as u64);
                }
            }
        }
    }
    let tail = u64::from(random.borrow_mut().next_int() as u32);
    black_box(digest.wrapping_add(tail))
}

fn run(saved: &dyn Directory, write: bool, millis: u64) -> (Duration, u64) {
    let start = Instant::now();
    let mut count = 0;
    while start.elapsed() < Duration::from_millis(millis) {
        black_box(batch(saved, write));
        count += 256;
    }
    (start.elapsed(), count)
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(500);
    let measure = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let directory = ByteBuffersDirectory::new();
    let mut consumer = Lucene90NormsFormat
        .norms_consumer(&state(&directory))
        .unwrap();
    consumer
        .add_norms_field(NormsField::Dense(0, &DENSE))
        .unwrap();
    consumer
        .add_norms_field(NormsField::Sparse(1, &SPARSE))
        .unwrap();
    consumer.close().unwrap();
    let owned = std::mem::size_of::<CrankyNormsFormat<Lucene90NormsFormat>>()
        + 2 * std::mem::size_of::<usize>()
        + std::mem::size_of::<RefCell<JavaRandom>>();
    println!("cranky_norms_format_owned_bytes\t{owned}");
    for write in [false, true] {
        let case = if write {
            "cranky_norms_lifecycle"
        } else {
            "cranky_norms_read"
        };
        println!("#check\t{case}\t{}\t256", batch(&directory, write));
        run(&directory, write, warmup);
        let (elapsed, count) = run(&directory, write, measure);
        println!(
            "{case}\t{:.3}\t{count}",
            elapsed.as_nanos() as f64 / count as f64
        );
    }
}
