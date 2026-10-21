//! The mock analysis chain against `GenTestFrameworkAnalysis`
//! (`fixtures/testframework/data/analysis.txt`): over 100 seeds, every
//! chain -- `MockAnalyzer` in three set-ups, `MockGraphTokenFilter`,
//! `MockRandomLookaheadTokenFilter`, `CrankyTokenFilter`,
//! `MockCharFilter`, `MockSynonymFilter`, `MockHoleInjectingTokenFilter` --
//! built from the same seeded `JavaRandom`s as its Java twin must dump the
//! same tokens (term, offsets, position increment and length, type,
//! payload), the same end state, and throw at the same point.

use lucene_analysis::{AnalysisError, Analyzer, StrReader, TokenStream, Tokenizer};
use lucene_test_framework::analysis::mock_filters::english_stopset;
use lucene_test_framework::analysis::mock_tokenizer::{self, MockTokenizer};
use lucene_test_framework::analysis::{
    CrankyTokenFilter, MockAnalyzer, MockCharFilter, MockGraphTokenFilter,
    MockHoleInjectingTokenFilter, MockRandomLookaheadTokenFilter, MockSynonymFilter,
};
use lucene_test_framework::util::test_util;
use lucene_util::java_random::JavaRandom;

const SEEDS: i64 = 100;
const SYN_WORDS: [&str; 5] = ["dogs", "guinea", "pig", "small", "running"];

fn fixture() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/testframework/data/analysis.txt");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn units(s: &str) -> String {
    if s.is_empty() {
        return "-".into();
    }
    s.encode_utf16().map(|u| format!("{u:04x}")).collect()
}

fn payload(p: Option<&[u8]>) -> String {
    match p {
        None => "-".into(),
        Some(b) => b.iter().map(|x| format!("{x:02x}")).collect(),
    }
}

fn exc(out: &mut Vec<String>, e: &AnalysisError) {
    let AnalysisError::Io(msg) = e else {
        panic!("expected a fake I/O error, got {e:?}");
    };
    out.push(format!("exc {}", msg.replace(' ', "_")));
}

/// `GenTestFrameworkAnalysis.dump`.
fn dump(out: &mut Vec<String>, ts: &mut dyn TokenStream) {
    let run = |out: &mut Vec<String>, ts: &mut dyn TokenStream| -> Result<(), AnalysisError> {
        ts.reset()?;
        while ts.increment_token()? {
            let a = ts.attributes();
            out.push(format!(
                "tok {} {} {} {} {} {} {}",
                units(a.term()),
                a.start_offset(),
                a.end_offset(),
                a.position_increment(),
                a.position_length(),
                a.token_type(),
                payload(a.payload())
            ));
        }
        ts.end()?;
        let a = ts.attributes();
        out.push(format!("end {} {}", a.end_offset(), a.position_increment()));
        Ok(())
    };
    if let Err(e) = run(out, ts) {
        exc(out, &e);
    }
    if let Err(e) = ts.close() {
        exc(out, &e);
    }
}

fn whitespace(text: &str) -> MockTokenizer {
    MockTokenizer::new(mock_tokenizer::whitespace(), false)
        .with_text(text)
        .unwrap()
}

fn text(seed: i64, k: i64) -> String {
    test_util::random_analysis_string(&mut JavaRandom::new(seed * 31 + 100 + k), 20, false)
}

fn analyzer(out: &mut Vec<String>, chain: &str, a: &Analyzer, text: &str) {
    out.push(format!("text {chain} {}", units(text)));
    dump(out, &mut a.token_stream("f", text).unwrap());
    a.close();
}

fn stream(out: &mut Vec<String>, chain: &str, text: &str, ts: &mut dyn TokenStream) {
    out.push(format!("text {chain} {}", units(text)));
    dump(out, ts);
}

fn seed_lines(seed: i64) -> Vec<String> {
    let mut out = vec![format!("seed {seed}")];
    let out = &mut out;

    let a = MockAnalyzer::new(&mut JavaRandom::new(seed * 31)).build();
    analyzer(out, "mock", &a, &text(seed, 0));

    let mut r1 = JavaRandom::new(seed * 31 + 1);
    let simple_stop =
        MockAnalyzer::with_automaton(&mut r1, mock_tokenizer::simple(), true, english_stopset());
    let simple_stop = simple_stop
        .max_token_length(test_util::next_int(&mut r1, 1, 10))
        .build();
    analyzer(out, "simpleStop", &simple_stop, &text(seed, 1));

    let keyword = MockAnalyzer::with_tokenizer(
        &mut JavaRandom::new(seed * 31 + 2),
        mock_tokenizer::keyword(),
        false,
    )
    .build();
    analyzer(out, "keyword", &keyword, &text(seed, 2));

    let t = text(seed, 3);
    let mut graph = MockGraphTokenFilter::new(&mut JavaRandom::new(seed * 31 + 3), whitespace(&t));
    stream(out, "graph", &t, &mut graph);

    let t = text(seed, 4);
    let mut lookahead =
        MockRandomLookaheadTokenFilter::new(&mut JavaRandom::new(seed * 31 + 4), whitespace(&t));
    stream(out, "lookahead", &t, &mut lookahead);

    let t = text(seed, 5);
    // a throw leaves the tokenizer mid-stream: close() there is the point
    let mut cranky_source = whitespace(&t);
    cranky_source.set_enable_checks(false);
    let mut cranky = CrankyTokenFilter::new(cranky_source, JavaRandom::new(seed * 31 + 5));
    stream(out, "cranky", &t, &mut cranky);

    let t = text(seed, 6);
    let remainder = JavaRandom::new(seed * 31 + 6).next_int_bounded(10);
    let mut char_filter = MockTokenizer::new(mock_tokenizer::whitespace(), false);
    char_filter
        .set_reader(Box::new(
            MockCharFilter::new(Box::new(StrReader::new(t.clone())), remainder).unwrap(),
        ))
        .unwrap();
    stream(out, "charFilter", &t, &mut char_filter);

    let mut r7 = JavaRandom::new(seed * 31 + 107);
    let words = test_util::next_int(&mut r7, 1, 8);
    let t = (0..words)
        .map(|_| SYN_WORDS[r7.next_int_bounded(SYN_WORDS.len() as i32) as usize])
        .collect::<Vec<_>>()
        .join(" ");
    let mut synonym = MockSynonymFilter::new(whitespace(&t));
    stream(out, "synonym", &t, &mut synonym);

    let t = text(seed, 8);
    let mut holes =
        MockHoleInjectingTokenFilter::new(&mut JavaRandom::new(seed * 31 + 8), whitespace(&t));
    stream(out, "holes", &t, &mut holes);

    std::mem::take(out)
}

#[test]
fn mock_chains_dump_what_lucene_dumps() {
    let fixture = fixture();
    let mut want = fixture.lines();
    for seed in 0..SEEDS {
        for (i, got) in seed_lines(seed).into_iter().enumerate() {
            let expected = want
                .next()
                .unwrap_or_else(|| panic!("seed {seed}: fixture ends early"));
            assert_eq!(got, expected, "seed {seed}, line {i} of the seed");
        }
    }
    assert_eq!(want.next(), None, "the fixture has lines no seed produced");
}
