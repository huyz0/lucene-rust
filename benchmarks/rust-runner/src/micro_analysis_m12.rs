//! M12's pairs (`benchmarks/micro/java/AnalysisM12Micro.java` is the Java
//! twin): the language modules' filters over corpora both sides generate
//! from `SweepMicro.Rng`.

use std::hint::black_box;
use std::time::Duration;

use lucene_analysis::util::{SegmentingBase, SegmentingTokenizer, Segmenter, WhitespaceTokenizer};
use lucene_analysis::{AnalysisError, Analyzer, AnalyzerDefinition, TokenStream, TokenStreamComponents};
use lucene_analysis_phonetic::bm::{NameType, PhoneticEngine, RuleType};
use lucene_analysis_phonetic::encoder::{Encoder, LANGUAGE_PACKAGE};
use lucene_analysis_kuromoji::{JapaneseAnalyzer, JapaneseTokenizer, Mode};
use lucene_analysis_nori::{DecompoundMode, KoreanAnalyzer, KoreanTokenizer};
use lucene_analysis_morfologik::analyzer::polish_dictionary;
use lucene_analysis_morfologik::{MorfologikFilter, UkrainianMorfologikAnalyzer};
use lucene_analysis_phonetic::{BeiderMorseFilter, DaitchMokotoffSoundexFilter, DoubleMetaphoneFilter, PhoneticFilter};
use lucene_analysis_smartcn::tokenizer::hmm_chinese_tokenizer;
use lucene_analysis_smartcn::SmartChineseAnalyzer;
use lucene_analysis_stempel::stemmer::default_table;
use lucene_analysis_stempel::{PolishAnalyzer, StempelFilter, StempelStemmer};

use super::{consume_stream, measure, Rng};

type Sink = Result<TokenStreamComponents, AnalysisError>;

struct Chain(Box<dyn Fn() -> Sink + Send + Sync>);

impl AnalyzerDefinition for Chain {
    fn create_components(&self, _field: &str) -> Sink {
        (self.0)()
    }
}

fn chain(f: impl Fn() -> Sink + Send + Sync + 'static) -> Analyzer {
    Analyzer::new(Chain(Box::new(f)))
}

fn comps(sink: impl TokenStream + 'static) -> Sink {
    Ok(TokenStreamComponents::new(sink))
}

fn run(name: &str, a: &Analyzer, docs: &[String], w: Duration, m: Duration) {
    measure(name, w, m, || {
        let mut tokens = 0u64;
        for text in docs {
            let mut ts = a.token_stream("body", black_box(text)).unwrap();
            tokens += consume_stream(&mut ts);
        }
        tokens
    });
}

/// `AnalysisM12Micro.SYLLABLES`.
const SYLLABLES: [&str; 50] = [
    "an", "ber", "schm", "idt", "ko", "wal", "ski", "mc", "don", "ald", "ph", "ough", "tz", "sch", "ch", "cz", "rz",
    "ei", "ie", "ou", "th", "gh", "w", "y", "ss", "ll", "tt", "n", "m", "r", "l", "k", "s", "t", "d", "b", "g", "p",
    "f", "v", "z", "x", "q", "j", "h", "a", "e", "i", "o", "u",
];

/// `AnalysisM12Micro.names`.
fn names(seed: u64, docs: usize) -> Vec<String> {
    let mut r = Rng(seed);
    (0..docs)
        .map(|_| {
            let mut s = String::new();
            for w in 0..50 {
                if w > 0 {
                    s.push(' ');
                }
                let n = 2 + r.next() % 3;
                let mut name = String::new();
                for _ in 0..n {
                    name.push_str(SYLLABLES[(r.next() % SYLLABLES.len() as u64) as usize]);
                }
                let first = name.remove(0).to_ascii_uppercase();
                s.push(first);
                s.push_str(&name);
            }
            s
        })
        .collect()
}

fn encoder(simple: &str) -> Encoder {
    Encoder::for_class_name(&format!("{LANGUAGE_PACKAGE}{simple}")).unwrap()
}

fn phonetic(simple: &'static str, inject: bool) -> Analyzer {
    chain(move || comps(PhoneticFilter::new(WhitespaceTokenizer::new(), encoder(simple), inject)))
}

pub(super) fn bench_analysis_m12(w: Duration, m: Duration) {
    let names = names(0x9E37_79B9_7F4A_7C15, 100);
    let few = names_few();
    run("ph_soundex", &phonetic("Soundex", true), &names, w, m);
    run("ph_refined_soundex", &phonetic("RefinedSoundex", false), &names, w, m);
    run("ph_metaphone", &phonetic("Metaphone", true), &names, w, m);
    run("ph_double_metaphone", &phonetic("DoubleMetaphone", true), &names, w, m);
    run("ph_caverphone2", &phonetic("Caverphone2", false), &names, w, m);
    run("ph_cologne", &phonetic("ColognePhonetic", true), &names, w, m);
    run("ph_nysiis", &phonetic("Nysiis", true), &names, w, m);
    run("ph_mra", &phonetic("MatchRatingApproachEncoder", true), &names, w, m);
    run(
        "double_metaphone_filter",
        &chain(|| comps(DoubleMetaphoneFilter::new(WhitespaceTokenizer::new(), 4, true)?)),
        &names,
        w,
        m,
    );
    run(
        "daitch_mokotoff",
        &chain(|| comps(DaitchMokotoffSoundexFilter::new(WhitespaceTokenizer::new(), true))),
        &names,
        w,
        m,
    );
    let approx = PhoneticEngine::new(NameType::Generic, RuleType::Approx, true).unwrap();
    let exact = PhoneticEngine::new(NameType::Ashkenazi, RuleType::Exact, true).unwrap();
    run(
        "beider_morse_gen_approx",
        &chain(move || comps(BeiderMorseFilter::new(WhitespaceTokenizer::new(), approx.clone()))),
        &few,
        w,
        m,
    );
    let polish = fixture_docs("fixtures/data/analysis_stempel/stems.tsv");
    let table = default_table();
    run(
        "stempel_filter",
        &chain(move || {
            comps(StempelFilter::new(
                WhitespaceTokenizer::new(),
                StempelStemmer::new(std::sync::Arc::clone(&table)),
            ))
        }),
        &polish,
        w,
        m,
    );
    run("polish_analyzer", &Analyzer::new(PolishAnalyzer::default()), &polish, w, m);
    let morf = fixture_docs("fixtures/data/analysis_morfologik/lookups_polish.tsv");
    let dict = polish_dictionary();
    run(
        "morfologik_filter",
        &chain(move || comps(MorfologikFilter::new(WhitespaceTokenizer::new(), std::sync::Arc::clone(&dict)))),
        &morf,
        w,
        m,
    );
    run(
        "ukrainian_analyzer",
        &Analyzer::new(UkrainianMorfologikAnalyzer::default()),
        &fixture_docs("fixtures/data/analysis_morfologik/lookups_ukrainian.tsv"),
        w,
        m,
    );
    let ja = japanese_docs();
    let kuromoji = |mode: Mode, discard_compound: bool, nbest: i32| {
        chain(move || {
            let mut t = JapaneseTokenizer::with_options(None, true, discard_compound, mode);
            t.set_n_best_cost(nbest);
            comps(t)
        })
    };
    run("kuromoji_normal", &kuromoji(Mode::Normal, true, 0), &ja, w, m);
    run("kuromoji_search", &kuromoji(Mode::Search, false, 0), &ja, w, m);
    run("kuromoji_extended", &kuromoji(Mode::Extended, true, 0), &ja, w, m);
    run("kuromoji_nbest", &kuromoji(Mode::Normal, true, 2000), &ja, w, m);
    run("japanese_analyzer", &Analyzer::new(JapaneseAnalyzer::default()), &ja, w, m);
    let ko = line_docs(&["fixtures/corpus/analysis-korean.txt", "fixtures/data/analysis_nori/stress.txt"]);
    run("nori_discard", &chain(|| comps(KoreanTokenizer::default())), &ko, w, m);
    run(
        "nori_mixed",
        &chain(|| comps(KoreanTokenizer::new(None, DecompoundMode::Mixed, true, false))),
        &ko,
        w,
        m,
    );
    run("korean_analyzer", &Analyzer::new(KoreanAnalyzer::default()), &ko, w, m);
    let zh = line_docs(&["fixtures/corpus/analysis-chinese.txt"]);
    // texts.txt holds escapes (`\n`); both sides read the escaped lines.
    let sentences = line_docs(&["fixtures/data/analysis_segmenting/texts.txt"]);
    run(
        "sentence_segmenting",
        &chain(|| comps(SegmentingTokenizer::new(WholeSentence::default()))),
        &sentences,
        w,
        m,
    );
    run("smartcn_tokenizer", &chain(|| comps(hmm_chinese_tokenizer())), &zh, w, m);
    run("smartcn_analyzer", &Analyzer::new(SmartChineseAnalyzer::default()), &zh, w, m);
    run(
        "beider_morse_ash_exact",
        &chain(move || comps(BeiderMorseFilter::new(WhitespaceTokenizer::new(), exact.clone()))),
        &few,
        w,
        m,
    );
}

/// `AnalysisM12Micro.fixtureDocs`: 50-word documents from a fixture file's
/// first column (escaped lines skipped), read from the repository root.
fn fixture_docs(file: &str) -> Vec<String> {
    let text = std::fs::read_to_string(file).unwrap_or_else(|e| panic!("{file}: {e}"));
    let words: Vec<&str> = text
        .split('\n')
        .map(|l| l.split('\t').next().unwrap_or(""))
        .filter(|w| !w.is_empty() && !w.contains('\\') && !w.contains(' '))
        .collect();
    words.chunks_exact(50).map(|c| c.join(" ")).collect()
}

/// `AnalysisM12Micro.japaneseDocs`.
fn japanese_docs() -> Vec<String> {
    line_docs(&["fixtures/corpus/analysis-japanese.txt", "fixtures/data/analysis_kuromoji/stress.txt"])
}

/// `AnalysisM12Micro.lineDocs`: every non-empty line of the files.
fn line_docs(files: &[&str]) -> Vec<String> {
    files
        .iter()
        .flat_map(|f| {
            std::fs::read_to_string(f)
                .unwrap_or_else(|e| panic!("{f}: {e}"))
                .split('\n')
                .filter(|l| !l.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn names_few() -> Vec<String> {
    names(0x2545_F491_4F6C_DD1D, 10)
}

/// `AnalysisM12Micro.WholeSentenceTokenizer`.
#[derive(Default)]
struct WholeSentence {
    bounds: Option<(usize, usize)>,
}

impl Segmenter for WholeSentence {
    fn set_next_sentence(&mut self, _: &SegmentingBase, start: usize, end: usize) -> Result<(), AnalysisError> {
        self.bounds = Some((start, end));
        Ok(())
    }

    fn increment_word(&mut self, base: &mut SegmentingBase) -> Result<bool, AnalysisError> {
        let Some((start, end)) = self.bounds.take() else {
            return Ok(false);
        };
        let s = base.correct_offset(base.offset() + start as i32);
        let e = base.correct_offset(base.offset() + end as i32);
        let mut term = [0u16; 1024];
        let len = end - start;
        term[..len].copy_from_slice(&base.buffer()[start..end]);
        let a = base.attributes_mut();
        a.clear_attributes();
        a.set_term_utf16(&term[..len]);
        a.set_offset(s, e)?;
        Ok(true)
    }
}
