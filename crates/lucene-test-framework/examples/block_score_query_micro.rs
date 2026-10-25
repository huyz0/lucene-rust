//! BlockScoreQueryWrapper scorer construction/iteration and prebuilt block bounds.
use lucene_analysis::core_analysis::WhitespaceAnalyzer;
use lucene_analysis::Analyzer;
use lucene_index::document::{Document, Store, StringField};
use lucene_search::collector::ScoreMode;
use lucene_search::query::{BooleanQuery, BoostQuery, Clause, ConstantScoreQuery, TermQuery};
use lucene_search::test_support::{with_leaf, Leaf, NO_MORE_DOCS};
use lucene_test_framework::search::block_score_query_wrapper::wrap;
use lucene_test_framework::search::test_index::TestIndex;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};
const N: usize = 256;
fn feed(hash: u64, value: u64) -> u64 {
    (hash ^ value).wrapping_mul(1099511628211)
}
// Preserve the measured Java twin's fixed-count loop and per-step branch.
#[allow(clippy::needless_range_loop)]
fn batch(leaf: &Leaf<'_>, bounds: bool) -> (u64, u128) {
    let mut prepared = if bounds {
        (0..N)
            .map(|_| leaf.scorer(ScoreMode::Complete).unwrap().unwrap())
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let mut digest = 14695981039346656037;
    let start = Instant::now();
    for step in 0..N {
        if bounds {
            let s = &mut prepared[step];
            for target in [0, 1, 2, 4, 7, 10, 15, NO_MORE_DOCS] {
                let end = s.advance_shallow(target).unwrap();
                digest = feed(digest, end as u32 as u64);
                digest = feed(digest, s.max_score(end).unwrap().to_bits() as u64);
                digest = feed(digest, s.max_score(NO_MORE_DOCS).unwrap().to_bits() as u64);
                digest = feed(digest, s.doc_id() as u32 as u64);
            }
        } else {
            let mut s = leaf.scorer(ScoreMode::Complete).unwrap().unwrap();
            digest = feed(digest, s.cost() as u64);
            loop {
                let doc = s.next_doc().unwrap();
                digest = feed(digest, doc as u32 as u64);
                digest = feed(digest, s.score().unwrap().to_bits() as u64);
                if doc == NO_MORE_DOCS {
                    break;
                }
            }
        }
    }
    let elapsed = start.elapsed().as_nanos();
    (black_box(digest), elapsed)
}
fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let warmup = args
        .get(1)
        .map(|s| s.parse::<u64>().unwrap())
        .unwrap_or(500);
    let measure = args
        .get(2)
        .map(|s| s.parse::<u64>().unwrap())
        .unwrap_or(1000);
    let docs = (0..16)
        .map(|id| {
            let mut d = Document::new();
            d.add(StringField::new("id", id.to_string(), Store::No));
            d
        })
        .collect();
    let index = TestIndex::build(
        Arc::new(Analyzer::new(WhitespaceAnalyzer::default())),
        &[docs],
        &[],
        &[],
    );
    index.with_searcher(|searcher| {
        for length in [1, 2, 3, 7] {
            let clauses = [0, 2, 3, 6, 8, 9, 11, 14]
                .into_iter()
                .zip([0.25, 2.0, 1.0, 5.0, 0.5, 3.0, 7.0, 4.0])
                .map(|(doc, score)| {
                    let term: Clause = TermQuery::new("id", doc.to_string()).into();
                    Clause::Boost(Box::new(BoostQuery::new(
                        Clause::ConstantScore(Box::new(ConstantScoreQuery::new(term, 1.0))),
                        score,
                    )))
                });
            let query = BooleanQuery::new()
                .with_must([wrap(BooleanQuery::new().with_should(clauses), length)]);
            with_leaf(searcher, &query, 0, |leaf| {
                for bounds in [false, true] {
                    let name = format!(
                        "block_{}_{length}",
                        if bounds { "bounds" } else { "buffer" }
                    );
                    let expected = batch(leaf, bounds).0;
                    for (phase, budget) in [warmup, measure].into_iter().enumerate() {
                        let start = Instant::now();
                        let mut ns = 0;
                        let mut operations = 0u64;
                        loop {
                            let (digest, elapsed) = batch(leaf, bounds);
                            assert_eq!(digest, expected);
                            ns += elapsed;
                            operations += N as u64;
                            if start.elapsed() >= Duration::from_millis(budget) {
                                break;
                            }
                        }
                        if phase == 1 {
                            println!("{name}\t{:.3}\t{operations}", ns as f64 / operations as f64);
                        }
                    }
                    println!("#check\t{name}\t{expected}\t{N}");
                }
                Ok(())
            })
            .unwrap();
        }
    });
}
