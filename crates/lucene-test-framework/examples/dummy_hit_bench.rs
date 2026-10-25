//! Paired with `DummyTotalHitCountMicro.java`; two counters, 4096 callbacks.
use std::hint::black_box;
use std::time::{Duration, Instant};

use lucene_search::collector::Collector;
use lucene_test_framework::search::dummy_total_hit_count::DummyTotalHitCountCollector;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup: u64 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(500);
    let measure: u64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let mut state = 0x12345678_u32;
    let docs: Vec<i32> = (0..4096)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state & 0x7fff_ffff) as i32
        })
        .collect();
    let run = |budget| {
        let start = Instant::now();
        let mut batches = 0_u64;
        let mut digest = 0_i32;
        while start.elapsed() < Duration::from_millis(budget) {
            let mut collectors = [
                DummyTotalHitCountCollector::new(),
                DummyTotalHitCountCollector::new(),
            ];
            for &doc in black_box(&docs) {
                Collector::collect(&mut collectors[(doc & 1) as usize], doc);
            }
            assert_eq!(
                collectors[0].total_hits() + collectors[1].total_hits(),
                4096
            );
            digest = black_box(collectors[0].total_hits() * 31 + collectors[1].total_hits());
            batches += 1;
        }
        (start.elapsed(), batches * 4096, digest)
    };
    run(warmup);
    let (elapsed, operations, digest) = run(measure);
    println!(
        "dummy_hit_callbacks\t{:.3}\t{operations}\t{digest}",
        elapsed.as_nanos() as f64 / operations as f64
    );
}
