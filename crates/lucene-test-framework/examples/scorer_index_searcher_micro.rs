//! Real indexed scorer-only searches, same corpus/query/collector work as Java.
#[path = "../tests/scorer_index_searcher_cases/mod.rs"]
mod cases;
use lucene_search::collector::{ScoreMode, ScoringCollector};
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, ConstantScoreQuery};
use lucene_test_framework::search::{
    disabling_bulk_scorer_query, scorer_index_searcher::ScorerIndexSearcher,
};
use std::hint::black_box;
use std::time::{Duration, Instant};
const BASIS: u64 = 0xcbf2_9ce4_8422_2325;
fn hash(d: u64, v: u64) -> u64 {
    (d ^ v).wrapping_mul(0x100_0000_01b3)
}
struct Digest {
    mode: ScoreMode,
    competitive: bool,
    digest: u64,
    count: u64,
}
impl ScoringCollector for Digest {
    fn collect(&mut self, doc: i32, score: f32) {
        self.count += 1;
        self.digest = hash(self.digest, doc as u64);
        self.digest = hash(
            self.digest,
            if self.mode.needs_scores() {
                u64::from(score.to_bits())
            } else {
                0
            },
        );
    }
    fn min_competitive_score(&self) -> Option<f32> {
        (self.competitive && self.count > 0).then_some(f32::INFINITY)
    }
    fn score_mode(&self) -> ScoreMode {
        self.mode
    }
}
fn batch(
    search: impl Fn(&BooleanQuery, &mut Digest),
    queries: &[BooleanQuery],
    mode: ScoreMode,
    competitive: bool,
) -> u64 {
    let mut digest = BASIS;
    for (q, query) in queries.iter().enumerate() {
        let mut c = Digest {
            mode,
            competitive,
            digest: BASIS,
            count: 0,
        };
        search(query, &mut c);
        if competitive {
            assert_eq!(
                c.count, 1,
                "constant term must stop after the first live hit"
            );
        }
        digest = hash(digest, q as u64);
        digest = hash(digest, c.count);
        digest = hash(digest, c.digest);
    }
    black_box(digest)
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup = args.get(1).map(|v| v.parse().unwrap()).unwrap_or(1000);
    let measure = args.get(2).map(|v| v.parse().unwrap()).unwrap_or(1500);
    let index = cases::index(64);
    index.with_searcher(|base| {
        let norms: Vec<_> = (0..base.segments().len()).map(|i| base.norms(i)).collect();
        let forced = ScorerIndexSearcher::new(base.segments(), &norms).unwrap();
        let plain = IndexSearcher::new(base.segments(), &norms).unwrap();
        let raw: Vec<_> = cases::clauses()
            .into_iter()
            .take(9)
            .map(cases::top_level)
            .collect();
        let wrapped: Vec<_> = cases::clauses()
            .into_iter()
            .take(9)
            .map(|c| cases::top_level(disabling_bulk_scorer_query::wrap(c)))
            .collect();
        let constant = cases::top_level(ConstantScoreQuery::new(cases::term("a"), 1.0).into());
        let wrapped_constant = cases::top_level(disabling_bulk_scorer_query::wrap(
            ConstantScoreQuery::new(cases::term("a"), 1.0),
        ));
        println!(
            "scorer_searcher_wrapper_extra_bytes\t{}",
            std::mem::size_of_val(&forced) - std::mem::size_of_val(&plain)
        );
        for case in 0..6 {
            let competitive = case >= 4;
            let scorer_searcher = case < 2 || case == 4;
            let mode = if competitive {
                ScoreMode::TopScores
            } else if case % 2 == 0 {
                ScoreMode::Complete
            } else {
                ScoreMode::CompleteNoScores
            };
            let name = format!(
                "{}_{}",
                if scorer_searcher {
                    "scorer_searcher"
                } else {
                    "disabling_query"
                },
                if competitive {
                    "constant_term_top_scores"
                } else if case % 2 == 0 {
                    "complete"
                } else {
                    "no_scores"
                }
            );
            let queries: &[BooleanQuery] = if competitive {
                std::slice::from_ref(if scorer_searcher {
                    &constant
                } else {
                    &wrapped_constant
                })
            } else if scorer_searcher {
                &raw
            } else {
                &wrapped
            };
            let search = |q: &BooleanQuery, c: &mut Digest| {
                if scorer_searcher {
                    forced.search_collector(q, c).unwrap()
                } else {
                    plain.search_collector(q, c).unwrap()
                }
            };
            let expected = batch(search, queries, mode, competitive);
            for (phase, budget) in [warmup, measure].into_iter().enumerate() {
                let start = Instant::now();
                let mut operations = 0u64;
                loop {
                    assert_eq!(batch(search, queries, mode, competitive), expected);
                    operations += 1;
                    if start.elapsed() >= Duration::from_millis(budget) {
                        break;
                    }
                }
                if phase == 1 {
                    println!(
                        "{name}\t{:.3}\t{operations}",
                        start.elapsed().as_nanos() as f64 / operations as f64
                    )
                }
            }
            println!("#check\t{name}\t{expected}\t{}", queries.len());
        }
    });
}
