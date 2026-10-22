//! `checkRandomData` over every ported analyzer, the M12 modules' included:
//! each Lucene analyzer test class has a `testRandomStrings` that does
//! exactly this (`checkRandomData(random(), a, 200 * RANDOM_MULTIPLIER)`).
//! and most add a `testRandomHugeStrings` (`checkRandomData(random(), a,
//! 10 * RANDOM_MULTIPLIER, 8192)`). Here every analyzer gets its own test
//! running both, so a failure names it.
//!
//! `ThaiAnalyzer` is left out: its tokenizer refuses by design (see
//! `lucene_analysis::lang::th`), so it has no stream to check.

use std::sync::Arc;

use lucene_analysis::lang::*;
use lucene_analysis::{Analyzer, AnalyzerDefinition, CharArraySet, ENGLISH_STOP_WORDS};
use lucene_test_framework::analysis::{check_random_data, check_random_data_with, MockAnalyzer};
use lucene_test_framework::util::lucene_test_case::{default_multiplier, LuceneTestCase};

/// Java's `200 * RANDOM_MULTIPLIER`, before the multiplier.
const ITERATIONS: i32 = 200;

fn check<D: AnalyzerDefinition + Send + Sync + 'static>(name: &str, def: D) {
    check_with(LuceneTestCase::new(name), def);
}

fn check_with<D: AnalyzerDefinition + Send + Sync + 'static>(mut t: LuceneTestCase, def: D) {
    let a = Analyzer::new(def);
    let m = t.multiplier();
    check_random_data(t.random(), &a, ITERATIONS * m).unwrap();
    check_random_data_with(t.random(), &a, 10 * m, 8192, false, true).unwrap();
}

macro_rules! random_strings {
    ($($test:ident => $def:expr;)*) => {
        $(
            #[test]
            fn $test() {
                check(stringify!($test), $def);
            }
        )*
    };
}

random_strings! {
    arabic => ar::ArabicAnalyzer::default();
    bulgarian => bg::BulgarianAnalyzer::default();
    bengali => bn::BengaliAnalyzer::default();
    brazilian => br::BrazilianAnalyzer::default();
    catalan => ca::CatalanAnalyzer::default();
    sorani => ckb::SoraniAnalyzer::default();
    czech => cz::CzechAnalyzer::default();
    danish => da::DanishAnalyzer::default();
    german => de::GermanAnalyzer::default();
    greek => el::GreekAnalyzer::default();
    spanish => es::SpanishAnalyzer::default();
    estonian => et::EstonianAnalyzer::default();
    basque => eu::BasqueAnalyzer::default();
    persian => fa::PersianAnalyzer::default();
    finnish => fi::FinnishAnalyzer::default();
    french => fr::FrenchAnalyzer::default();
    irish => ga::IrishAnalyzer::default();
    galician => gl::GalicianAnalyzer::default();
    hindi => hi::HindiAnalyzer::default();
    hungarian => hu::HungarianAnalyzer::default();
    armenian => hy::ArmenianAnalyzer::default();
    indonesian => id::IndonesianAnalyzer::default();
    italian => it::ItalianAnalyzer::default();
    lithuanian => lt::LithuanianAnalyzer::default();
    latvian => lv::LatvianAnalyzer::default();
    nepali => ne::NepaliAnalyzer::default();
    dutch => nl::DutchAnalyzer::default();
    norwegian => no::NorwegianAnalyzer::default();
    portuguese => pt::PortugueseAnalyzer::default();
    romanian => ro::RomanianAnalyzer::default();
    russian => ru::RussianAnalyzer::default();
    serbian => sr::SerbianAnalyzer::default();
    swedish => sv::SwedishAnalyzer::default();
    tamil => ta::TamilAnalyzer::default();
    telugu => te::TeluguAnalyzer::default();
    turkish => tr::TurkishAnalyzer::default();
    cjk => lucene_analysis::cjk::CJKAnalyzer::default();
    classic => lucene_analysis::classic::ClassicAnalyzer::default();
    uax29_url_email => lucene_analysis::email::UAX29URLEmailAnalyzer::default();
    english => lucene_analysis::en::EnglishAnalyzer::default();
    standard => lucene_analysis::StandardAnalyzer::default();
    whitespace => lucene_analysis::core_analysis::WhitespaceAnalyzer::default();
    unicode_whitespace => lucene_analysis::core_analysis::UnicodeWhitespaceAnalyzer;
    simple => lucene_analysis::core_analysis::SimpleAnalyzer;
    stop => lucene_analysis::core_analysis::StopAnalyzer::new(Arc::new(
        CharArraySet::from_words(ENGLISH_STOP_WORDS, false),
    ));
    japanese => lucene_analysis_kuromoji::JapaneseAnalyzer::default();
    japanese_completion_index => lucene_analysis_kuromoji::JapaneseCompletionAnalyzer::new(
        None,
        lucene_analysis_kuromoji::completion::CompletionMode::Index,
    );
    japanese_completion_query => lucene_analysis_kuromoji::JapaneseCompletionAnalyzer::new(
        None,
        lucene_analysis_kuromoji::completion::CompletionMode::Query,
    );
    morfologik => lucene_analysis_morfologik::MorfologikAnalyzer::default();
    ukrainian_morfologik => lucene_analysis_morfologik::UkrainianMorfologikAnalyzer::default();
    korean => lucene_analysis_nori::KoreanAnalyzer::default();
    smart_chinese => lucene_analysis_smartcn::SmartChineseAnalyzer::default();
    polish => lucene_analysis_stempel::PolishAnalyzer::default();
    icu_collation_key => lucene_analysis_icu::ICUCollationKeyAnalyzer::new(
        lucene_analysis_icu::Collator::root().unwrap(),
    );
}

/// `icu_collation_key` at seed C950197: its huge-strings run indexes the
/// collation key of a long text, and `CheckIndex`'s
/// `postings.intersect_agrees` scan used to match it with a regexp
/// backtracker a few frames deep per code point, overflowing the 2 MB
/// test-thread stack (seen before the fix: `TESTS_SEED=C950197` aborted at
/// `RUST_MIN_STACK` 2, 1 and 0.5 MB, and passed at 64 MB).
#[test]
fn icu_collation_key_seed_c950197() {
    let t = LuceneTestCase::with_config(
        "icu_collation_key",
        0xC950197,
        false,
        default_multiplier(false),
    );
    check_with(
        t,
        lucene_analysis_icu::ICUCollationKeyAnalyzer::new(
            lucene_analysis_icu::Collator::root().unwrap(),
        ),
    );
}

/// `MockAnalyzer` itself, with its checks on (Java's `TestMockAnalyzer.
/// testRandomStrings` covers the other settings).
#[test]
fn mock_analyzer() {
    let mut t = LuceneTestCase::new("mock_analyzer");
    let r = t.random();
    let a = MockAnalyzer::new(r).build();
    check_random_data(r, &a, ITERATIONS).unwrap();
}
