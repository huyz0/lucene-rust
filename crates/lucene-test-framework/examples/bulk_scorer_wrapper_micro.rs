//! Compare actual buffered wrapper traversal over identical deterministic bulk sources.
use std::convert::Infallible;
use std::hint::black_box;
use std::time::Instant;

use lucene_search::collector::ScoringCollector;
use lucene_test_framework::search::bulk_scorer_wrapper_scorer::{
    BulkScorer, BulkScorerWrapperScorer, NO_MORE_DOCS,
};

struct Source<'a> {
    docs: &'a [i32],
    index: usize,
}
impl BulkScorer for Source<'_> {
    type Error = Infallible;
    fn cost(&self) -> u64 {
        self.docs.len() as u64
    }
    fn score(
        &mut self,
        collector: &mut dyn ScoringCollector,
        min: i32,
        max: i32,
    ) -> Result<i32, Infallible> {
        while self.index < self.docs.len() && self.docs[self.index] < min {
            self.index += 1;
        }
        while self.index < self.docs.len() && self.docs[self.index] < max {
            let doc = self.docs[self.index];
            collector.collect(doc, f32::from_bits(0x3f000000 + (doc as u32 & 1023)));
            self.index += 1;
        }
        Ok(if self.index == self.docs.len() {
            NO_MORE_DOCS
        } else {
            max
        })
    }
}
fn mix(digest: u64, doc: i32, bits: u32) -> u64 {
    digest
        .wrapping_mul(31)
        .wrapping_add(doc as u64)
        .wrapping_mul(31)
        .wrapping_add(u64::from(bits))
}
fn expected(docs: &[i32], advance: bool) -> (u64, usize) {
    let (mut digest, mut count, mut target) = (0, 0, 0);
    for &doc in docs {
        if !advance || doc >= target {
            digest = mix(digest, doc, 0x3f000000 + (doc as u32 & 1023));
            count += 1;
            target = doc + 17;
        }
    }
    (digest, count)
}
fn run(docs: &[i32], advance: bool, millis: u64) -> (f64, u64, usize) {
    let expected = expected(docs, advance);
    let (mut elapsed, mut batches) = (0_u128, 0);
    while elapsed < u128::from(millis) * 1_000_000 {
        let mut wrapper = BulkScorerWrapperScorer::new(Source { docs, index: 0 }, 64).unwrap();
        assert_eq!(wrapper.cost(), docs.len() as u64);
        let iterator = wrapper.iterator();
        let (mut digest, mut count, mut target) = (0, 0, 0);
        let start = Instant::now();
        loop {
            let doc = if advance {
                iterator.advance(target)
            } else {
                iterator.next_doc()
            }
            .unwrap();
            if doc == NO_MORE_DOCS {
                break;
            }
            digest = mix(digest, doc, iterator.score().to_bits());
            count += 1;
            target = doc + 17;
        }
        elapsed += start.elapsed().as_nanos();
        assert_eq!(black_box((digest, count)), expected);
        batches += 1;
        black_box(wrapper);
    }
    (elapsed as f64 / batches as f64, expected.0, expected.1)
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup = args.get(1).map_or(500, |s| s.parse().unwrap());
    let measure = args.get(2).map_or(750, |s| s.parse().unwrap());
    for (name, gap) in [("dense", 1), ("sparse", 17)] {
        let docs: Vec<_> = (0..8192).map(|i| i * gap).collect();
        for advance in [false, true] {
            let case = format!(
                "bulk_wrapper_{name}_{}",
                if advance { "advance" } else { "next" }
            );
            black_box(run(&docs, advance, warmup));
            let (ns, digest, count) = run(&docs, advance, measure);
            println!("{case}\t{ns:.3}");
            println!("#check\t{case}\t{digest}\t{count}");
        }
    }
    println!("bulk_wrapper_buffer_payload_bytes\t512");
}
