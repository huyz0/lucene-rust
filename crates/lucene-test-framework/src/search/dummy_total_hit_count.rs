//! Lucene's `DummyTotalHitCountCollector`: count every collection callback.
//!
//! The flat Rust collector combines Java's collector and leaf collector; the
//! leaf has no state beyond the shared count. `wrapping_add` preserves Java
//! `int` arithmetic, including manager reduction. Bulk count hooks retain
//! their defaults so searches must visit each matching document.

use lucene_search::collector::{Collector, ScoreMode, ScoringCollector};
use lucene_search::collectors::CollectorManager;
use lucene_search::Result;

/// `DummyTotalHitCountCollector`, without `Weight.count` shortcuts.
#[derive(Debug, Default)]
pub struct DummyTotalHitCountCollector {
    total_hits: i32,
}

impl DummyTotalHitCountCollector {
    /// Java: `DummyTotalHitCountCollector()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Java: `getTotalHits()`.
    pub fn total_hits(&self) -> i32 {
        self.total_hits
    }

    /// Java: `createManager()`.
    pub fn create_manager() -> DummyTotalHitCountCollectorManager {
        DummyTotalHitCountCollectorManager
    }
}

impl Collector for DummyTotalHitCountCollector {
    /// Java: `LeafCollector.collect(doc)`.
    fn collect(&mut self, _doc_id: i32) {
        self.total_hits = self.total_hits.wrapping_add(1);
    }
}

impl ScoringCollector for DummyTotalHitCountCollector {
    /// Java: `LeafCollector.collect(doc)`; `setScorer` ignores the scorer.
    fn collect(&mut self, doc_id: i32, _score: f32) {
        Collector::collect(self, doc_id);
    }

    /// Java: `scoreMode()`.
    fn score_mode(&self) -> ScoreMode {
        ScoreMode::CompleteNoScores
    }
}

/// The independent collectors and wrapping reduction from `createManager()`.
#[derive(Debug, Default, Clone, Copy)]
pub struct DummyTotalHitCountCollectorManager;

impl CollectorManager for DummyTotalHitCountCollectorManager {
    type Collector = DummyTotalHitCountCollector;
    type Output = i32;

    /// Java: `newCollector()`.
    fn new_collector(&self) -> Result<Self::Collector> {
        Ok(DummyTotalHitCountCollector::new())
    }

    /// Java: `reduce(collectors)`.
    fn reduce(&self, collectors: Vec<Self::Collector>) -> Result<i32> {
        let mut sum = 0_i32;
        for collector in collectors {
            sum = sum.wrapping_add(collector.total_hits);
        }
        Ok(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn java_count(case: &str) -> i32 {
        include_str!("../../../../fixtures/testframework/data/dummy_total_hit_count.txt")
            .lines()
            .find_map(|line| {
                let (key, value) = line.split_once('\t')?;
                (key == case).then(|| value.parse().unwrap())
            })
            .expect("Java fixture case")
    }

    #[test]
    fn every_callback_counts_even_with_repeated_docs_and_ignored_scores() {
        let mut collector = DummyTotalHitCountCollector::new();
        assert_eq!(collector.total_hits(), 0);
        Collector::collect(&mut collector, 7);
        Collector::collect(&mut collector, 7);
        ScoringCollector::collect(&mut collector, 0, f32::NAN);
        collector.collect_range(2, 5, f32::INFINITY);
        assert_eq!(collector.total_hits(), java_count("callbacks"));
        assert_eq!(collector.score_mode(), ScoreMode::CompleteNoScores);
        assert_eq!(collector.pruning_threshold(), None);
        assert!(!collector.add_hits(100));
        assert!(!collector.count_losing_hits(100, 1.0));
        assert_eq!(collector.total_hits(), java_count("callbacks"));
    }

    #[test]
    fn java_int_increment_and_manager_reduction_wrap() {
        let mut collector = DummyTotalHitCountCollector {
            total_hits: i32::MAX,
        };
        Collector::collect(&mut collector, 0);
        assert_eq!(collector.total_hits(), java_count("overflow"));
        let manager = DummyTotalHitCountCollector::create_manager();
        assert_eq!(manager.reduce(Vec::new()).unwrap(), java_count("empty"));
        assert_eq!(
            manager
                .reduce(vec![
                    collector,
                    DummyTotalHitCountCollector { total_hits: -1 }
                ])
                .unwrap(),
            java_count("reduce")
        );
        let mut first = manager.new_collector().unwrap();
        let second = manager.new_collector().unwrap();
        Collector::collect(&mut first, 1);
        assert_eq!(second.total_hits(), 0);
        assert_eq!(manager.reduce(vec![first, second]).unwrap(), 1);
    }
}
