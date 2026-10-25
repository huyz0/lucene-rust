//! Java twin: AssertingLiveDocsMicro. Checks stay enabled in release.
use std::cell::Cell;
use std::hint::black_box;
use std::io;
use std::time::{Duration, Instant};

use lucene_store::directory::Directory;
use lucene_store::ByteBuffersDirectory;
use lucene_test_framework::codecs::asserting_live_docs::AssertingLiveDocsFormat;
use lucene_test_framework::codecs::cranky_live_docs::{
    LiveDocsFormat, LiveDocsInfo, Lucene90LiveDocsFormat,
};
use lucene_util::fixed_bit_set::FixedBitSet;

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
fn batch(
    real: bool,
    counter: &Counter,
    guard: &AssertingLiveDocsFormat<&Counter>,
    writer: &AssertingLiveDocsFormat<Lucene90LiveDocsFormat>,
) -> u64 {
    counter.0.set(0);
    let length = if real { 65 } else { 4096 };
    let mut bits = FixedBitSet::new(length);
    let mut deleted = 0;
    for i in 0..bits.len() {
        if i % 3 == 0 {
            deleted += 1;
        } else {
            bits.set(i);
        }
    }
    let info = LiveDocsInfo {
        name: "_ald",
        segment_id: b"0123456789abcdef",
        max_doc: length,
        del_gen: -1,
        next_del_gen: 1,
        del_count: 0,
    };
    let empty = ByteBuffersDirectory::new();
    let mut digest = 0;
    for _ in 0..256 {
        if real {
            let dir = ByteBuffersDirectory::new();
            writer
                .write_live_docs(black_box(&bits), &dir, black_box(&info), deleted)
                .unwrap();
            digest += dir
                .open("_ald_1.liv")
                .unwrap()
                .iter()
                .map(|&b| u64::from(b))
                .sum::<u64>();
        } else {
            guard
                .write_live_docs(black_box(&bits), &empty, black_box(&info), deleted)
                .unwrap();
        }
    }
    if !real {
        digest += counter.0.get() * 31;
    }
    black_box(digest)
}
fn run(
    real: bool,
    ms: u64,
    counter: &Counter,
    guard: &AssertingLiveDocsFormat<&Counter>,
    writer: &AssertingLiveDocsFormat<Lucene90LiveDocsFormat>,
) -> (u64, Duration) {
    let start = Instant::now();
    let mut batches = 0;
    while start.elapsed() < Duration::from_millis(ms) {
        black_box(batch(real, counter, guard, writer));
        batches += 1;
    }
    (batches, start.elapsed())
}
fn main() {
    let counter = Counter(Cell::new(0));
    let guard = AssertingLiveDocsFormat::new(&counter);
    let writer = AssertingLiveDocsFormat::new(Lucene90LiveDocsFormat);
    // Requested format-wrapper storage excluding its stateless delegate and
    // input bitset. Rust monomorphization erases this wrapper's storage.
    println!(
        "#memory\tasserting_live_docs_wrapper\t{}",
        std::mem::size_of_val(&writer)
    );
    for real in [false, true] {
        let name = if real {
            "asserting_live_docs_write"
        } else {
            "asserting_live_docs_guard"
        };
        println!(
            "#check\t{name}\t{}\t256",
            batch(real, &counter, &guard, &writer)
        );
        run(real, 1000, &counter, &guard, &writer);
        let (batches, elapsed) = run(real, 1500, &counter, &guard, &writer);
        let operations = batches * 256;
        println!(
            "{name}\t{:.3}\t{operations}",
            elapsed.as_nanos() as f64 / operations as f64
        );
    }
}
