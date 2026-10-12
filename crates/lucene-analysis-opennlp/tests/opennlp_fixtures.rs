//! M12 T12.5: analysis-opennlp against Lucene 10.5.0 over OpenNLP 2.5.3.
//!
//! `fixtures/src/GenAnalysisOpenNLP.java` trains small models with
//! OpenNLP itself (a seeded synthetic grammar: sentence detector,
//! tokenizer, maxent/perceptron/UD-tagged POS taggers, maxent/perceptron
//! chunkers, a lemmatizer and a lemma dictionary) and writes
//! `analysis_opennlp/`: the `NLP*Op` outputs over `opennlp_text.txt`
//! (`tools.tsv`) and Lucene's analysis chains over the same lines
//! (`o_*.tsv`, configured as `factory_specs.tsv` says). Every row must be
//! equal.
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

#[path = "../../lucene-analysis/tests/support/mod.rs"]
mod support;

use lucene_analysis::factory::CustomAnalyzer;
use lucene_analysis_opennlp::ops::LemmatizerOp;
use lucene_analysis_opennlp::tools::chunker::{Chunker, ChunkerModel};
use lucene_analysis_opennlp::tools::dictionary::DictionaryLemmatizer;
use lucene_analysis_opennlp::tools::featuregen;
use lucene_analysis_opennlp::tools::lemmatizer::{Lemmatizer, LemmatizerModel};
use lucene_analysis_opennlp::tools::postag::{PosModel, PosTagger};
use lucene_analysis_opennlp::tools::sentdetect::{SentenceDetector, SentenceModel, Span};
use lucene_analysis_opennlp::tools::text::{self, JString};
use lucene_analysis_opennlp::tools::tokenize::{Tokenizer, TokenizerModel};
use std::path::PathBuf;
use std::sync::Arc;
use support::{analyze_line, data_dir, normalise_expected};

const DIR: &str = "analysis_opennlp";

fn bytes(file: &str) -> Vec<u8> {
    std::fs::read(data_dir(DIR) + file).unwrap()
}

fn lines() -> Vec<String> {
    std::fs::read_to_string(data_dir(DIR) + "opennlp_text.txt")
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect()
}

fn spans(s: &[Span]) -> String {
    s.iter()
        .map(|s| format!("{}-{}", s.start, s.end))
        .collect::<Vec<_>>()
        .join(" ")
}

fn joined(v: &[JString]) -> String {
    v.iter()
        .map(|t| text::to_string(t))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn tools_match_opennlp() {
    let detector = |f: &str| SentenceDetector::new(SentenceModel::load(&bytes(f)).unwrap());
    let tokenizer = |f: &str| Tokenizer::new(TokenizerModel::load(&bytes(f)).unwrap()).unwrap();
    let (mut detector, mut detector1) = (detector("en-sent.bin"), detector("en-sent-default.bin"));
    let (mut tokenizer, mut tokenizer1) =
        (tokenizer("en-token.bin"), tokenizer("en-token-noopt.bin"));
    let mut taggers: Vec<PosTagger> = [
        "en-pos-maxent.bin",
        "en-pos-ud.bin",
        "en-pos-perceptron.bin",
        "en-pos-v15.bin",
        "en-pos-nofeat.bin",
    ]
    .iter()
    .map(|f| PosTagger::new(&PosModel::load(&bytes(f)).unwrap()).unwrap())
    .collect();
    let mut chunkers: Vec<Chunker> = ["en-chunker.bin", "en-chunker-perceptron.bin"]
        .iter()
        .map(|f| Chunker::new(&ChunkerModel::load(&bytes(f)).unwrap()).unwrap())
        .collect();
    let dict = Arc::new(DictionaryLemmatizer::read(&bytes("en-lemmas.dict")).unwrap());
    let lemma_model = LemmatizerModel::load(&bytes("en-lemmatizer.bin")).unwrap();
    let mut lemmatizers = [
        LemmatizerOp::new(Some(Arc::clone(&dict)), None),
        LemmatizerOp::new(None, Some(Lemmatizer::new(&lemma_model).unwrap())),
        LemmatizerOp::new(Some(dict), Some(Lemmatizer::new(&lemma_model).unwrap())),
    ];
    let mut actual = Vec::new();
    for (ln, line) in lines().iter().enumerate() {
        let units = text::units(line);
        let sents = detector.sent_pos_detect(&units);
        actual.push(format!("S\t{ln}\t{}", spans(&sents)));
        actual.push(format!(
            "S1\t{ln}\t{}",
            spans(&detector1.sent_pos_detect(&units))
        ));
        for s in &sents {
            let sentence = &units[s.start..s.end];
            let toks = tokenizer.tokenize_pos(sentence);
            actual.push(format!("T\t{ln}\t{}", spans(&toks)));
            let toks1 = tokenizer1.tokenize_pos(sentence);
            actual.push(format!("T1\t{ln}\t{}", spans(&toks1)));
            let words: Vec<JString> = toks
                .iter()
                .map(|t| sentence[t.start..t.end].to_vec())
                .collect();
            if words.is_empty() {
                continue;
            }
            for (m, t) in taggers.iter_mut().enumerate() {
                actual.push(format!("P{m}\t{ln}\t{}", joined(&t.tag(&words).unwrap())));
            }
            let tags = taggers[0].tag(&words).unwrap();
            for (m, c) in chunkers.iter_mut().enumerate() {
                actual.push(format!(
                    "C{m}\t{ln}\t{}",
                    joined(&c.chunk(&words, &tags).unwrap())
                ));
            }
            for (m, l) in lemmatizers.iter_mut().enumerate() {
                actual.push(format!(
                    "L{m}\t{ln}\t{}",
                    joined(&l.lemmatize(&words, &tags).unwrap())
                ));
            }
        }
    }
    let expected = std::fs::read_to_string(data_dir(DIR) + "tools.tsv").unwrap();
    let expected: Vec<&str> = expected.lines().collect();
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert_eq!(a, e, "tools.tsv row {i} differs");
    }
    assert_eq!(actual.len(), expected.len());
    assert!(actual.len() > 600, "{}", actual.len());
}

fn build(spec: &str) -> Result<lucene_analysis::Analyzer, String> {
    let dir = PathBuf::from(data_dir(DIR));
    let mut b = CustomAnalyzer::builder_with_dir(dir).map_err(|e| e.java_class().to_string())?;
    for part in spec.split(' ') {
        let (kind, rest) = part.split_once(':').unwrap();
        let mut name_args = rest.split(',');
        let name = name_args.next().unwrap();
        let mut params = Vec::new();
        for a in name_args {
            let (k, v) = a.split_once('=').unwrap();
            params.push(k.to_string());
            params.push(v.to_string());
        }
        let p: Vec<&str> = params.iter().map(String::as_str).collect();
        b = match kind {
            "t" => b.with_tokenizer(name, &p),
            _ => b.add_token_filter(name, &p),
        }
        .map_err(|e| e.java_class().to_string())?;
    }
    b.build()
        .map(|c| c.into_analyzer())
        .map_err(|e| e.java_class().to_string())
}

#[test]
fn chains_match_lucene() {
    lucene_analysis_opennlp::register_factories().unwrap();
    let lines = lines();
    let specs = std::fs::read_to_string(data_dir(DIR) + "factory_specs.tsv").unwrap();
    let mut checked = 0;
    for row in specs.lines() {
        let (name, spec) = row.split_once('\t').unwrap();
        let expected: Vec<String> = std::fs::read_to_string(format!("{}{name}.tsv", data_dir(DIR)))
            .unwrap()
            .lines()
            .map(normalise_expected)
            .collect();
        let mut actual = Vec::new();
        match build(spec) {
            Ok(a) => {
                for (ln, line) in lines.iter().enumerate() {
                    analyze_line(&a, ln, line, &mut actual);
                }
            }
            Err(class) => actual.push(format!("B\t{class}")),
        }
        for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
            assert_eq!(a, e, "{name} ({spec}): row {i} differs");
        }
        assert_eq!(actual.len(), expected.len(), "{name}: row count");
        checked += 1;
    }
    assert_eq!(checked, 23);
}

/// `AnalysisRows.esc`'s inverse over UTF-16 units (a `\uXXXX` may be a lone
/// surrogate).
fn unesc_units(s: &str) -> JString {
    let mut out = Vec::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        let mut b = [0u16; 2];
        if c != '\\' {
            out.extend_from_slice(c.encode_utf16(&mut b));
            continue;
        }
        match it.next() {
            Some('t') => out.push(9),
            Some('n') => out.push(10),
            Some('r') => out.push(13),
            Some('u') => {
                let h: String = it.by_ref().take(4).collect();
                out.push(u16::from_str_radix(&h, 16).unwrap());
            }
            Some(o) => out.extend_from_slice(o.encode_utf16(&mut b)),
            None => out.push(u16::from(b'\\')),
        }
    }
    out
}

#[test]
fn token_classes_match_opennlp() {
    let rows = std::fs::read_to_string(data_dir(DIR) + "token_classes.tsv").unwrap();
    let mut n = 0;
    for row in rows.lines() {
        let (token, class) = row.rsplit_once('\t').unwrap();
        let token = unesc_units(token);
        assert_eq!(featuregen::token_feature(&token), class, "{row:?}");
        n += 1;
    }
    assert_eq!(n, 54);
}

#[test]
fn feature_generators_match_opennlp() {
    let descriptors =
        std::fs::read_to_string(data_dir(DIR) + "featuregen_descriptors.txt").unwrap();
    let mut generators: Vec<featuregen::Generator> = descriptors
        .lines()
        .map(|d| featuregen::create(d.as_bytes()).unwrap())
        .collect();
    let sentences: [Vec<JString>; 2] = [
        ["The", "ÄÖÜ", "A.", "12", "ひらがな", "x"]
            .map(text::units)
            .to_vec(),
        vec![text::units("Solo")],
    ];
    let rows = std::fs::read_to_string(data_dir(DIR) + "featuregen.tsv").unwrap();
    let mut n = 0;
    for row in rows.lines() {
        let mut f = row.split('\t');
        let d: usize = f.next().unwrap()[1..].parse().unwrap();
        let si: usize = f.next().unwrap().parse().unwrap();
        let i: usize = f.next().unwrap().parse().unwrap();
        let expected: Vec<JString> = f.map(unesc_units).collect();
        let toks = &sentences[si];
        let tags: Vec<JString> = (0..toks.len())
            .map(|t| text::units(&format!("T{t}")))
            .collect();
        let mut actual = Vec::new();
        generators[d]
            .create_features(&mut actual, toks, i, &tags, si as u64)
            .unwrap();
        assert_eq!(actual, expected, "{row}");
        n += 1;
    }
    assert_eq!(n, 14);
}
