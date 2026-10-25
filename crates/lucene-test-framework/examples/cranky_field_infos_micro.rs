//! Java twin: `CrankyFieldInfosMicro.java`. Explicit wrapper calls, no codec I/O.

use lucene_codecs::field_infos::{self, FieldInfos};
use lucene_store::{ByteBuffersDirectory, Directory};
use lucene_test_framework::codecs::cranky_field_infos::{
    CrankyFieldInfosFormat, FieldInfosFormat, FAKE_WRITE_ERROR,
};
use lucene_util::java_random::JavaRandom;
use std::cell::{Cell, RefCell};
use std::hint::black_box;
use std::rc::Rc;
use std::time::{Duration, Instant};

struct Counting {
    reads: Cell<u64>,
    writes: u64,
}
impl FieldInfosFormat for Counting {
    fn read(
        &self,
        _: &dyn Directory,
        _: &str,
        _: &[u8; 16],
        suffix: &str,
    ) -> field_infos::Result<FieldInfos> {
        self.reads.set(self.reads.get() + suffix.len() as u64 + 1);
        FieldInfos::new(vec![])
    }
    fn write(
        &mut self,
        _: &dyn Directory,
        _: &str,
        _: &[u8; 16],
        suffix: &str,
        _: &FieldInfos,
    ) -> field_infos::Result<()> {
        self.writes += suffix.len() as u64 + 1;
        Ok(())
    }
}

fn batch(directory: &dyn Directory, write: bool) -> u64 {
    let random = Rc::new(RefCell::new(JavaRandom::new(black_box(0))));
    let mut format = CrankyFieldInfosFormat::new(
        Counting {
            reads: Cell::new(0),
            writes: 0,
        },
        random.clone(),
    );
    let infos = FieldInfos::new(vec![]).unwrap();
    let names = ["", "a", "abc", "abcdefg"];
    let mut failures = 0;
    for i in 0..256 {
        let suffix = black_box(names[i % names.len()]);
        if write {
            if let Err(error) = format.write(directory, "_bench", &[0; 16], suffix, &infos) {
                assert_eq!(error.to_string(), FAKE_WRITE_ERROR);
                failures += 1;
            }
        } else {
            assert!(format
                .read(directory, "_bench", &[0; 16], suffix)
                .unwrap()
                .fields
                .is_empty());
        }
    }
    let tail = u64::from(random.borrow_mut().next_int() as u32);
    black_box(
        format.delegate().reads.get() * 17 + format.delegate().writes * 31 + failures * 101 + tail,
    )
}

fn run(directory: &dyn Directory, write: bool, ms: u64) -> (Duration, u64) {
    let start = Instant::now();
    let mut count = 0;
    while start.elapsed() < Duration::from_millis(ms) {
        black_box(batch(directory, write));
        count += 256;
    }
    (start.elapsed(), count)
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(500);
    let measure = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1000);
    // One wrapper with its inline delegate and its one shared RNG allocation.
    // Rc has two usize reference counts; allocator metadata is excluded.
    let owned = std::mem::size_of::<CrankyFieldInfosFormat<Counting>>()
        + 2 * std::mem::size_of::<usize>()
        + std::mem::size_of::<RefCell<JavaRandom>>();
    println!("#memory\tcranky_field_infos\t{owned}");
    let directory = ByteBuffersDirectory::new();
    for write in [false, true] {
        let name = if write {
            "cranky_field_infos_write"
        } else {
            "cranky_field_infos_read"
        };
        println!("#check\t{name}\t{}\t256", batch(&directory, write));
        run(&directory, write, warmup);
        let (elapsed, count) = run(&directory, write, measure);
        println!(
            "{name}\t{:.3}\t{count}",
            elapsed.as_nanos() as f64 / count as f64
        );
    }
}
