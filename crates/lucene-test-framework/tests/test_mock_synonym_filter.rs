//! Lucene's `TestMockSynonymFilter`, ported.

use lucene_test_framework::analysis::{assert_analyzes_to, Expected, MockSynonymAnalyzer};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;

/// One `assertAnalyzesTo(analyzer, input, output, starts, ends, types,
/// posIncs, posLens, true)` case.
type Case<'a> = (
    &'a str,
    &'a [&'a str],
    &'a [i32],
    &'a [i32],
    &'a [&'a str],
    &'a [i32],
    &'a [i32],
);

const W: &str = "word";
const S: &str = "SYNONYM";

/// test the mock synonym filter
#[test]
fn test() {
    let mut t = LuceneTestCase::new("test");
    let r = t.random();
    let analyzer = MockSynonymAnalyzer::analyzer();
    let cases: &[Case] = &[
        (
            "dogs",
            &["dogs", "dog"],
            &[0, 0],
            &[4, 4],
            &[W, S],
            &[1, 0],
            &[1, 1],
        ),
        (
            "small dogs",
            &["small", "dogs", "dog"],
            &[0, 6, 6],
            &[5, 10, 10],
            &[W, W, S],
            &[1, 1, 0],
            &[1, 1, 1],
        ),
        (
            "dogs running",
            &["dogs", "dog", "running"],
            &[0, 0, 5],
            &[4, 4, 12],
            &[W, S, W],
            &[1, 0, 1],
            &[1, 1, 1],
        ),
        (
            "small dogs running",
            &["small", "dogs", "dog", "running"],
            &[0, 6, 6, 11],
            &[5, 10, 10, 18],
            &[W, W, S, W],
            &[1, 1, 0, 1],
            &[1, 1, 1, 1],
        ),
        ("guinea", &["guinea"], &[0], &[6], &[W], &[1], &[1]),
        ("pig", &["pig"], &[0], &[3], &[W], &[1], &[1]),
        (
            "guinea pig",
            &["guinea", "cavy", "pig"],
            &[0, 0, 7],
            &[6, 10, 10],
            &[W, S, W],
            &[1, 0, 1],
            &[1, 2, 1],
        ),
        (
            "guinea dogs",
            &["guinea", "dogs", "dog"],
            &[0, 7, 7],
            &[6, 11, 11],
            &[W, W, S],
            &[1, 1, 0],
            &[1, 1, 1],
        ),
        (
            "dogs guinea",
            &["dogs", "dog", "guinea"],
            &[0, 0, 5],
            &[4, 4, 11],
            &[W, S, W],
            &[1, 0, 1],
            &[1, 1, 1],
        ),
        (
            "dogs guinea pig",
            &["dogs", "dog", "guinea", "cavy", "pig"],
            &[0, 0, 5, 5, 12],
            &[4, 4, 11, 15, 15],
            &[W, S, W, S, W],
            &[1, 0, 1, 0, 1],
            &[1, 1, 1, 2, 1],
        ),
        (
            "guinea pig dogs",
            &["guinea", "cavy", "pig", "dogs", "dog"],
            &[0, 0, 7, 11, 11],
            &[6, 10, 10, 15, 15],
            &[W, S, W, W, S],
            &[1, 0, 1, 1, 0],
            &[1, 2, 1, 1, 1],
        ),
        (
            "small dogs and guinea pig running",
            &[
                "small", "dogs", "dog", "and", "guinea", "cavy", "pig", "running",
            ],
            &[0, 6, 6, 11, 15, 15, 22, 26],
            &[5, 10, 10, 14, 21, 25, 25, 33],
            &[W, W, S, W, W, S, W, W],
            &[1, 1, 0, 1, 1, 0, 1, 1],
            &[1, 1, 1, 1, 1, 2, 1, 1],
        ),
        (
            "small guinea pig and dogs running",
            &[
                "small", "guinea", "cavy", "pig", "and", "dogs", "dog", "running",
            ],
            &[0, 6, 6, 13, 17, 21, 21, 26],
            &[5, 12, 16, 16, 20, 25, 25, 33],
            &[W, W, S, W, W, W, S, W],
            &[1, 1, 0, 1, 1, 1, 0, 1],
            &[1, 1, 2, 1, 1, 1, 1, 1],
        ),
    ];
    for &(input, output, starts, ends, types, incs, lens) in cases {
        let expected = Expected::terms(output)
            .offsets(starts, ends)
            .types(types)
            .pos_incs(incs)
            .pos_lens(lens)
            .graph_offsets(true);
        assert_analyzes_to(r, &analyzer, input, &expected).unwrap();
    }
}
