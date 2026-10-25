//! Java twin: CrankyLiveDocsMicro; each batch replays the same 256 writes.
use std::cell::{Cell, RefCell};
use std::hint::black_box;
use std::io;
use std::rc::Rc;
use std::time::{Duration, Instant};

use lucene_store::directory::Directory;
use lucene_store::ByteBuffersDirectory;
use lucene_test_framework::codecs::cranky_live_docs::{
    CrankyLiveDocsFormat, LiveDocsFormat, LiveDocsInfo, Lucene90LiveDocsFormat,
};
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::java_random::JavaRandom;

struct Counter(Cell<u64>);
impl LiveDocsFormat for &Counter {
    fn read_live_docs(&self, _: &dyn Directory, _: &LiveDocsInfo<'_>) -> io::Result<FixedBitSet> {
        unreachable!()
    }
    fn write_live_docs(
        &self,
        _: &FixedBitSet,
        _: &dyn Directory,
        _: &LiveDocsInfo<'_>,
        _: usize,
    ) -> io::Result<()> {
        self.0.set(self.0.get() + 1);
        Ok(())
    }
    fn files(&self, _: &LiveDocsInfo<'_>, _: &mut Vec<String>) -> io::Result<()> {
        unreachable!()
    }
}

fn batch(real: bool) -> u64 {
    let random = Rc::new(RefCell::new(JavaRandom::new(42)));
    let counter = Counter(Cell::new(0));
    let guard = CrankyLiveDocsFormat::new(&counter, random.clone());
    let writer = CrankyLiveDocsFormat::new(Lucene90LiveDocsFormat, random);
    let info = LiveDocsInfo {
        name: "_0",
        segment_id: b"0123456789abcdef",
        max_doc: 65,
        del_gen: -1,
        next_del_gen: 1,
        del_count: 0,
    };
    let mut bits = FixedBitSet::new(65);
    for i in 0..64 {
        bits.set(i);
    }
    let empty_dir = ByteBuffersDirectory::new();
    let mut digest = 0;
    for _ in 0..256 {
        if real {
            let dir = ByteBuffersDirectory::new();
            if let Err(error) = writer.write_live_docs(black_box(&bits), &dir, black_box(&info), 1)
            {
                assert_eq!(
                    error.to_string(),
                    "Fake IOException from LiveDocsFormat.writeLiveDocs()"
                );
                digest += 7;
            } else {
                digest += dir
                    .open("_0_1.liv")
                    .unwrap()
                    .iter()
                    .map(|&b| u64::from(b))
                    .sum::<u64>();
            }
        } else if let Err(error) =
            guard.write_live_docs(black_box(&bits), &empty_dir, black_box(&info), 1)
        {
            assert_eq!(
                error.to_string(),
                "Fake IOException from LiveDocsFormat.writeLiveDocs()"
            );
            digest += 1;
        }
    }
    if !real {
        digest += counter.0.get() * 31;
    }
    black_box(digest)
}

fn run(real: bool, ms: u64) -> (u64, Duration) {
    let start = Instant::now();
    let mut batches = 0;
    while start.elapsed() < Duration::from_millis(ms) {
        black_box(batch(real));
        batches += 1;
    }
    (batches, start.elapsed())
}

fn main() {
    let random = Rc::new(RefCell::new(JavaRandom::new(42)));
    let format = CrankyLiveDocsFormat::new(Lucene90LiveDocsFormat, random);
    // Requested wrapper + shared RNG allocation, counted once. Delegate,
    // input bits, and transient directory/output allocations are excluded.
    let owned = std::mem::size_of_val(&format)
        + 2 * std::mem::size_of::<usize>()
        + std::mem::size_of::<RefCell<JavaRandom>>();
    println!("#memory\tcranky_live_docs_wrapper\t{owned}");
    for real in [false, true] {
        let name = if real {
            "cranky_live_docs_write"
        } else {
            "cranky_live_docs_guard"
        };
        println!("#check\t{name}\t{}\t256", batch(real));
        run(real, 1000);
        let (batches, elapsed) = run(real, 1500);
        let operations = batches * 256;
        println!(
            "{name}\t{:.3}\t{operations}",
            elapsed.as_nanos() as f64 / operations as f64
        );
    }
}
