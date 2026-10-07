//! M11 part 3's remaining pairs (`benchmarks/micro/java/AnalysisMiscMicro.java`
//! is the Java twin): the decompounders, the deprecated `WordDelimiterFilter`,
//! `ClassicAnalyzer`, `WikipediaTokenizer`, `DateRecognizerFilter` and
//! `Word2VecSynonymFilter`, each over a corpus both sides generate from
//! `SweepMicro.Rng` (so no input file is shared but the bytes are equal).

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use lucene_analysis::classic::ClassicAnalyzer;
use lucene_analysis::compound::{
    DictionaryCompoundWordTokenFilter, HyphenationCompoundWordTokenFilter, HyphenationTree,
};
use lucene_analysis::miscellaneous::{
    self as m, DateRecognizerFilter, SimpleDateFormat, WordDelimiterFilter,
    WordDelimiterGraphFilter,
};
use lucene_analysis::util::WhitespaceTokenizer;
use lucene_analysis::wikipedia::WikipediaTokenizer;
use lucene_analysis::{
    AnalysisError, Analyzer, AnalyzerDefinition, CharArraySet, KeywordTokenizer, TokenStream,
    TokenStreamComponents,
};
use lucene_search::word2vec::{Word2VecModel, Word2VecSynonymFilter, Word2VecSynonymProvider};
use lucene_util::base36::to_base36;

use super::{analysis_docs, consume_stream, measure, Rng};

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

const ALPHA: &[u8] = b"abcdefgh";

fn word(r: &mut Rng, len: u64) -> String {
    (0..len)
        .map(|_| char::from(ALPHA[(r.next() % ALPHA.len() as u64) as usize]))
        .collect()
}

/// `AnalysisMiscMicro.compoundDict`.
fn compound_dict() -> Vec<String> {
    let mut r = Rng(0x5EED_C0DE_1234_5678);
    (0..300)
        .map(|_| {
            let n = 2 + r.next() % 5;
            word(&mut r, n)
        })
        .collect()
}

/// `AnalysisMiscMicro.compoundDocs`.
fn compound_docs() -> Vec<String> {
    let mut r = Rng(0x0DDC_0FFE_E0DD_BA11);
    (0..200)
        .map(|_| {
            let mut s = String::new();
            for w in 0..100 {
                if w > 0 {
                    s.push(' ');
                }
                let n = 6 + r.next() % 20;
                s.push_str(&word(&mut r, n));
                if w % 3 == 0 {
                    s.push_str("-X9y");
                }
            }
            s
        })
        .collect()
}

/// `AnalysisMiscMicro.hyphenationXml`.
fn hyphenation_xml() -> String {
    let mut r = Rng(0x4859_5048_4E41_5445);
    let mut x = String::from(
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<hyphenation-info>\n<classes>\n",
    );
    for &c in ALPHA {
        x.push(char::from(c));
        x.push(char::from(c.to_ascii_uppercase()));
        x.push(' ');
    }
    x.push_str("\n</classes>\n<patterns>\n");
    for i in 0..3000 {
        let mut p = String::new();
        if r.next() % 6 == 0 {
            p.push('.');
        }
        let n = 1 + r.next() % 7;
        for _ in 0..n {
            if r.next() % 3 == 0 {
                p.push(char::from(b'0' + (r.next() % 10) as u8));
            }
            p.push(char::from(ALPHA[(r.next() % ALPHA.len() as u64) as usize]));
        }
        if r.next() % 3 == 0 {
            p.push(char::from(b'0' + (r.next() % 10) as u8));
        }
        if r.next() % 6 == 0 {
            p.push('.');
        }
        x.push_str(&p);
        x.push(if i % 12 == 11 { '\n' } else { ' ' });
    }
    x.push_str("\n</patterns>\n</hyphenation-info>\n");
    x
}

const WIKI: [&str; 9] = [
    "[[%s]]",
    "[[Category:%s]]",
    "'''%s'''",
    "''%s''",
    "[http://example.com/%s %s]",
    "{{cite %s}}",
    "<ref>%s</ref>",
    "== %s ==",
    "%s",
];

/// `AnalysisMiscMicro.wikiDocs`.
fn wiki_docs() -> Vec<String> {
    let mut r = Rng(0x5749_4B49_5045_4449);
    (0..1000)
        .map(|_| {
            let mut s = String::new();
            for w in 0..60 {
                if w > 0 {
                    s.push(' ');
                }
                let x = r.next();
                let word = format!("w{}", to_base36((x % 5000) as i64));
                s.push_str(&WIKI[((x >> 32) % WIKI.len() as u64) as usize].replace("%s", &word));
            }
            s
        })
        .collect()
}

/// `AnalysisMiscMicro.isoDateDocs`.
fn iso_date_docs() -> Vec<String> {
    let mut r = Rng(0x0DA7_E150_0DA7_E150);
    (0..2000)
        .map(|_| {
            let mut s = String::new();
            for w in 0..50 {
                if w > 0 {
                    s.push(' ');
                }
                let x = r.next();
                let (year, month, day) =
                    (1900 + (x >> 8) % 200, 1 + (x >> 16) % 12, 1 + (x >> 24) % 28);
                match x & 3 {
                    0 => s.push_str(&format!("{year:04}-{month:02}-{day:02}")),
                    1 => s.push_str(&format!("{year:04}-{month:02}")),
                    _ => {
                        s.push('t');
                        s.push_str(&to_base36(((x >> 8) % 50000) as i64));
                    }
                }
            }
            s
        })
        .collect()
}

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// `AnalysisMiscMicro.englishDateDocs`.
fn english_date_docs() -> Vec<String> {
    let mut r = Rng(0x0DA7_E0E0_0DA7_E0E0);
    (0..20000)
        .map(|_| {
            let x = r.next();
            if x & 1 == 0 {
                format!(
                    "{} {}, {}",
                    MONTHS[((x >> 8) % 12) as usize],
                    1 + (x >> 16) % 28,
                    1900 + (x >> 24) % 200
                )
            } else {
                format!("t{}", to_base36(((x >> 8) % 50000) as i64))
            }
        })
        .collect()
}

const W2V_TERMS: usize = 2000;
const W2V_DIM: usize = 50;

/// `AnalysisMiscMicro.w2vModel`.
fn w2v_model() -> Word2VecModel {
    let mut r = Rng(0x5752_3256_4543_5452);
    let mut m = Word2VecModel::new(W2V_TERMS, W2V_DIM);
    for i in 0..W2V_TERMS {
        let v: Vec<f32> = (0..W2V_DIM)
            .map(|_| ((r.next() % 2001) as i64 - 1000) as f32 / 1000f32)
            .collect();
        m.add_term_and_vector(format!("v{}", to_base36(i as i64)).into_bytes(), v)
            .unwrap();
    }
    m
}

/// `AnalysisMiscMicro.w2vDocs`.
fn w2v_docs() -> Vec<String> {
    let mut r = Rng(0x5752_3244_4F43_5321);
    (0..500)
        .map(|_| {
            let mut s = String::new();
            for w in 0..40 {
                if w > 0 {
                    s.push(' ');
                }
                s.push('v');
                s.push_str(&to_base36((r.next() % W2V_TERMS as u64) as i64));
            }
            s
        })
        .collect()
}

pub(super) fn bench_analysis_misc(w: Duration, mt: Duration) {
    let docs = analysis_docs();
    let compound = compound_docs();
    let dict = Arc::new(CharArraySet::from_words(compound_dict(), true));
    let tree = Arc::new(HyphenationTree::from_xml(&hyphenation_xml()).unwrap());
    let wdf = m::GENERATE_WORD_PARTS
        | m::GENERATE_NUMBER_PARTS
        | m::SPLIT_ON_CASE_CHANGE
        | m::SPLIT_ON_NUMERICS
        | m::STEM_ENGLISH_POSSESSIVE;
    let all = 511;
    let provider = Arc::new(Word2VecSynonymProvider::new(w2v_model()).unwrap());
    let (d1, d2, t1, t2) = (dict.clone(), dict.clone(), tree.clone(), tree);
    let cases: Vec<(&str, Analyzer, Vec<String>)> = vec![
        (
            "dict_compound",
            chain(move || {
                comps(DictionaryCompoundWordTokenFilter::dictionary(
                    WhitespaceTokenizer::new(),
                    d1.clone(),
                ))
            }),
            compound.clone(),
        ),
        (
            "hyph_compound",
            chain(move || {
                comps(HyphenationCompoundWordTokenFilter::hyphenation(
                    WhitespaceTokenizer::new(),
                    t1.clone(),
                    Some(d2.clone()),
                    Default::default(),
                    false,
                    false,
                ))
            }),
            compound.clone(),
        ),
        (
            "hyph_compound_nodict",
            chain(move || {
                comps(HyphenationCompoundWordTokenFilter::hyphenation(
                    WhitespaceTokenizer::new(),
                    t2.clone(),
                    None,
                    Default::default(),
                    false,
                    false,
                ))
            }),
            compound.clone(),
        ),
        (
            "wdf",
            chain(move || comps(WordDelimiterFilter::new(WhitespaceTokenizer::new(), wdf, None))),
            docs.clone(),
        ),
        (
            "wdf_all",
            chain(move || comps(WordDelimiterFilter::new(WhitespaceTokenizer::new(), all, None))),
            compound.clone(),
        ),
        (
            "wdgf_all",
            chain(move || {
                comps(WordDelimiterGraphFilter::new(WhitespaceTokenizer::new(), all, None)?)
            }),
            compound,
        ),
        ("classic", Analyzer::new(ClassicAnalyzer::default()), docs),
        (
            "wikipedia",
            chain(|| comps(WikipediaTokenizer::default())),
            wiki_docs(),
        ),
        (
            "date_iso",
            chain(|| {
                comps(DateRecognizerFilter::with_format(
                    WhitespaceTokenizer::new(),
                    SimpleDateFormat::new("yyyy-MM-dd")?,
                ))
            }),
            iso_date_docs(),
        ),
        (
            "date_default",
            chain(|| comps(DateRecognizerFilter::new(KeywordTokenizer::new()))),
            english_date_docs(),
        ),
        (
            "word2vec_synonym",
            chain(move || {
                comps(Word2VecSynonymFilter::new(
                    WhitespaceTokenizer::new(),
                    provider.clone(),
                    5,
                    0.0,
                ))
            }),
            w2v_docs(),
        ),
    ];
    for (name, a, corpus) in &cases {
        run(name, a, corpus, w, mt);
    }
}
