#![cfg(test)]
use super::*;

fn hit(doc_id: i32) -> ScoreDoc {
    ScoreDoc { doc_id, score: 0.0 }
}

/// `n` searches; each sampled one records `recall` at its level.
fn run(t: &RecallTuner, n: usize, recall: f64) -> Vec<usize> {
    let mut sampled = Vec::new();
    for q in 0..n {
        let s = t.begin();
        t.searched(&s);
        if s.sampled {
            sampled.push(q);
            t.record(s.level, &[(s.level, recall)]);
        }
    }
    sampled
}

#[test]
fn effort_ladder() {
    assert_eq!(effort(BASE_LEVEL), 1.0);
    assert!((effort(0) - 0.25).abs() < 1e-6);
    assert!((effort(LEVELS - 1) - 64.0).abs() < 1e-3);
    assert_eq!(effort(LEVELS + 5), effort(LEVELS - 1));
    assert!((effort(BASE_LEVEL + 1) - std::f32::consts::SQRT_2).abs() < 1e-6);
}

#[test]
fn target_checked() {
    assert!(RecallTarget::new(0.95, 0.01).is_ok());
    assert!(RecallTarget::new(1.0, 1.0).is_ok());
    for (r, s) in [
        (0.0, 0.1),
        (1.1, 0.1),
        (0.9, 0.0),
        (0.9, 1.5),
        (f32::NAN, 0.1),
    ] {
        assert!(RecallTarget::new(r, s).is_err(), "{r} {s}");
    }
    let t = RecallTarget::new(0.9, 0.01).unwrap();
    assert_eq!(t.every(), 100);
    assert_eq!(t.budget, DEFAULT_BUDGET);
    assert_eq!(t.with_budget(0.5).unwrap().budget, 0.5);
    assert!(t.with_budget(-1.0).is_err());
    assert!(t.with_budget(f32::NAN).is_err());
    assert_eq!(RecallTarget::new(0.9, 0.3).unwrap().every(), 3);
}

#[test]
fn recall_counts_exact_top_k() {
    let exact = [hit(1), hit(2), hit(3), hit(4)];
    assert_eq!(recall_at(&[hit(2), hit(1), hit(9)], &exact, 3), 2.0 / 3.0);
    assert_eq!(recall_at(&[], &[], 3), 1.0);
    assert_eq!(recall_at(&[hit(4)], &exact, 3), 0.0);
    // Fewer exact hits than `k`.
    assert_eq!(recall_at(&[hit(1)], &[hit(1)], 10), 1.0);
}

#[test]
fn samples_one_in_every() {
    let t = RecallTuner::new(
        RecallTarget::new(0.9, 0.25)
            .unwrap()
            .with_budget(f32::MAX)
            .unwrap(),
    );
    // A rate of 1/4 is above warm-up's 1/10: every fourth query.
    assert_eq!(run(&t, 40, 0.95), vec![0, 4, 8, 12, 16, 20, 24, 28, 32, 36]);
    let stats = t.stats();
    assert_eq!((stats.queries, stats.samples), (40, 10));
}

#[test]
fn warmup_samples_one_in_ten_then_the_rate() {
    let t = RecallTuner::new(
        RecallTarget::new(0.9, 0.001)
            .unwrap()
            .with_budget(f32::MAX)
            .unwrap(),
    );
    let n = (WARMUP_SAMPLES * 10) as usize;
    assert_eq!(run(&t, n, 0.95).len() as u64, WARMUP_SAMPLES);
    assert_eq!(run(&t, 2000, 0.95).len(), 2);
}

#[test]
fn budget_limits_samples_after_warmup() {
    let t = RecallTuner::new(
        RecallTarget::new(0.9, 1.0)
            .unwrap()
            .with_budget(0.0)
            .unwrap(),
    );
    // Warm-up samples within its own budget, whatever the target's.
    let warmup = WARMUP_SAMPLES as usize;
    assert_eq!(run(&t, warmup, 0.95).len(), warmup);
    // Checks that took any time at all exceed a zero budget.
    t.check_nanos.store(1, Ordering::Relaxed);
    assert!(run(&t, 100, 0.95).is_empty());
    // Within budget again once the searches have taken long enough.
    let t = RecallTuner::new(RecallTarget::new(0.9, 1.0).unwrap());
    run(&t, warmup, 0.95);
    t.check_nanos.store(1_000_000, Ordering::Relaxed);
    t.search_nanos.store(1_000_000, Ordering::Relaxed);
    assert!(run(&t, 10, 0.95).is_empty());
    t.search_nanos.store(100_000_000, Ordering::Relaxed);
    assert_eq!(run(&t, 1, 0.95).len(), 1);
}

#[test]
fn warmup_checks_take_no_longer_than_the_searches() {
    let t = RecallTuner::new(RecallTarget::new(0.9, 1.0).unwrap());
    t.check_nanos.store(u64::MAX / 2, Ordering::Relaxed);
    assert!(run(&t, 10, 0.95).is_empty());
    t.search_nanos.store(u64::MAX / 2, Ordering::Relaxed);
    assert_eq!(run(&t, 1, 0.95).len(), 1);
}

#[test]
fn a_claimed_sample_released_is_due_again() {
    let t = RecallTuner::new(RecallTarget::new(0.9, 0.01).unwrap());
    assert!(t.begin().sampled);
    // Claimed: nothing else samples until it is recorded or released.
    assert!(!t.begin().sampled);
    t.release();
    assert!(t.begin().sampled);
}

/// A simulated field whose recall depends on the level.
fn simulate(t: &RecallTuner, recall: impl Fn(usize) -> f64, samples: usize) {
    for _ in 0..samples {
        let level = t.level();
        let lowest = level.saturating_sub(NEIGHBOURS);
        let highest = (level + NEIGHBOURS).min(LEVELS - 1);
        let rs: Vec<_> = (lowest..=highest).map(|l| (l, recall(l))).collect();
        t.record(level, &rs);
    }
}

#[test]
fn climbs_to_the_shallowest_level_reaching_the_target() {
    let curve = |l: usize| 0.80 + 0.02 * l as f64;
    let t = RecallTuner::new(RecallTarget::new(0.95, 1.0).unwrap());
    simulate(&t, curve, 1000);
    // 0.80 + 0.02 * 8 = 0.96 is the first mean at 0.95 or above, but
    // its error (a coin flip's at 0.96, about 0.014 over ~100 decayed
    // samples) leaves it short: 0.98 at level 9 clears it.
    assert_eq!(t.level(), 9);
    let stats = t.stats();
    assert!((stats.recall.unwrap() - 0.98).abs() < 1e-9);
    let bound = stats.recall_lower_bound.unwrap();
    assert!((0.95..0.98).contains(&bound), "{bound}");
    assert_eq!(stats.effort, effort(9));
}

#[test]
fn descends_when_shallower_levels_suffice() {
    let t = RecallTuner::new(RecallTarget::new(0.90, 1.0).unwrap());
    simulate(&t, |l| 0.97 + 0.001 * l as f64, 1000);
    assert_eq!(t.level(), 0);
}

#[test]
fn follows_a_change_in_the_data() {
    let t = RecallTuner::new(RecallTarget::new(0.95, 1.0).unwrap());
    simulate(&t, |l| 0.80 + 0.02 * l as f64, 1000);
    assert_eq!(t.level(), 9);
    // The queries get harder: every level loses 0.04.
    simulate(&t, |l| 0.76 + 0.02 * l as f64, 1000);
    assert_eq!(t.level(), 11);
    // And easier again.
    simulate(&t, |l| 0.80 + 0.02 * l as f64, 1000);
    assert_eq!(t.level(), 9);
}

#[test]
fn needs_min_samples_before_moving() {
    let t = RecallTuner::new(RecallTarget::new(0.99, 1.0).unwrap());
    simulate(&t, |_| 0.5, 30);
    assert_eq!(t.level(), BASE_LEVEL);
    simulate(&t, |_| 0.5, 10);
    assert!(t.level() > BASE_LEVEL);
}

#[test]
fn caps_at_the_top_level() {
    let t = RecallTuner::new(RecallTarget::new(1.0, 1.0).unwrap());
    simulate(&t, |_| 0.5, 3000);
    assert_eq!(t.level(), LEVELS - 1);
}

#[test]
fn a_stale_sample_does_not_move_the_level() {
    let t = RecallTuner::new(RecallTarget::new(0.99, 1.0).unwrap());
    for _ in 0..50 {
        t.record(BASE_LEVEL + 3, &[(BASE_LEVEL + 3, 0.1)]);
    }
    assert_eq!(t.level(), BASE_LEVEL);
}
