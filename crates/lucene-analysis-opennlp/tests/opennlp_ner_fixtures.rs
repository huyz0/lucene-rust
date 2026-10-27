//! M14: `NLPNERTaggerOp` against Lucene 10.5.0 over OpenNLP 2.5.3.
//!
//! `fixtures/src/GenAnalysisOpenNLPNer.java` trains name finder models
//! with OpenNLP itself (GIS and perceptron, BIO and BILOU, typed and
//! untyped names, the default descriptor and one naming the dictionary,
//! POS-model and n-gram generators) and writes `analysis_opennlp_ner/`:
//! every model's names and token probabilities over documents of
//! sentences (`ner.tsv`), and each feature generator's features over three
//! sentences (`ner_featuregen.tsv`). Every row must be equal, the
//! probabilities to the bit.
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

#[path = "../../lucene-analysis/tests/support/mod.rs"]
mod support;

use std::collections::HashMap;
use std::sync::Arc;

use lucene_analysis_opennlp::ops::NerTaggerOp;
use lucene_analysis_opennlp::tools::dictionary::{ClusterDictionary, Dictionary};
use lucene_analysis_opennlp::tools::featuregen::{self, Owner, Resources};
use lucene_analysis_opennlp::tools::model::Artifact;
use lucene_analysis_opennlp::tools::namefind::{NameFinder, NameFinderModel, NameSpan};
use lucene_analysis_opennlp::tools::postag::PosModel;
use lucene_analysis_opennlp::tools::text::{self, JString};
use support::data_dir;

const DIR: &str = "analysis_opennlp_ner";

fn bytes(file: &str) -> Vec<u8> {
    std::fs::read(data_dir(DIR) + file).unwrap()
}

fn rows(file: &str) -> Vec<String> {
    std::fs::read_to_string(data_dir(DIR) + file)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect()
}

/// `ner_text.txt`: documents of sentences.
fn documents() -> Vec<Vec<Vec<JString>>> {
    let mut docs: Vec<Vec<Vec<JString>>> = Vec::new();
    for line in rows("ner_text.txt") {
        if line == "#doc" {
            docs.push(Vec::new());
        } else {
            let s = line.strip_prefix("s:").unwrap();
            let words = if s.is_empty() {
                Vec::new()
            } else {
                s.split(' ').map(text::units).collect()
            };
            docs.last_mut().unwrap().push(words);
        }
    }
    docs
}

fn hex(d: f64) -> String {
    format!("{:x}", d.to_bits())
}

fn names(spans: &[NameSpan]) -> String {
    spans
        .iter()
        .map(|s| {
            let kind = s
                .kind
                .as_ref()
                .map_or("null".to_string(), |k| text::to_string(k));
            format!("{}-{}:{kind}:{}", s.start, s.end, hex(s.prob))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn names_match_opennlp() {
    let models = [
        "ner-maxent.bin",
        "ner-nofeat.bin",
        "ner-badcodec.bin",
        "ner-beam1.bin",
        "ner-untyped.bin",
        "ner-bilou.bin",
        "ner-seq.bin",
    ];
    let docs = documents();
    let mut actual = Vec::new();
    for (m, file) in models.iter().enumerate() {
        let model = NameFinderModel::load(&bytes(file)).unwrap();
        let mut op = NerTaggerOp::new(&model).unwrap();
        let mut me = NameFinder::new(&model).unwrap();
        for (d, doc) in docs.iter().enumerate() {
            for (si, sentence) in doc.iter().enumerate() {
                let found = op.get_names(sentence).unwrap();
                actual.push(format!("N{m}\t{d}\t{si}\t{}", names(&found)));
                me.find(sentence).unwrap();
                let probs: Vec<String> = me.probs().iter().map(|&p| hex(p)).collect();
                actual.push(format!("Q{m}\t{d}\t{si}\t{}", probs.join(" ")));
            }
            op.reset();
            me.clear_adaptive_data();
        }
    }
    let expected = rows("ner.tsv");
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert_eq!(a, e, "ner.tsv row {i} differs");
    }
    assert_eq!(actual.len(), expected.len());
    assert!(actual.len() > 500, "{}", actual.len());
}

/// The generator rows' resources, as the generator's map holds them.
struct MapResources(HashMap<&'static str, Artifact>);

impl Resources for MapResources {
    fn resource(&self, key: &str) -> Option<&Artifact> {
        self.0.get(key)
    }
}

fn resources() -> MapResources {
    let mut map = HashMap::new();
    map.insert(
        "names.dict",
        Artifact::Dictionary(Arc::new(Dictionary::read(&bytes("names.dict")).unwrap())),
    );
    map.insert(
        "word.clusters",
        Artifact::WordClusters(Arc::new(
            ClusterDictionary::read_word_clusters(&bytes("word.clusters")).unwrap(),
        )),
    );
    map.insert(
        "brown.clusters",
        Artifact::BrownCluster(Arc::new(
            ClusterDictionary::read_brown(&bytes("brown.clusters")).unwrap(),
        )),
    );
    map.insert(
        "pos.bin",
        Artifact::PosModel(Arc::new(PosModel::load(&bytes("en-pos-ner.bin")).unwrap())),
    );
    MapResources(map)
}

/// `AnalysisRows.esc`'s inverse over UTF-16 units.
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
fn feature_generators_match_opennlp() {
    let resources = resources();
    let mut generators: Vec<featuregen::Generator> = rows("ner_featuregen_descriptors.txt")
        .iter()
        .map(|d| featuregen::create_with(d.as_bytes(), &resources, Owner::NameFinder).unwrap())
        .collect();
    let sentences: [Vec<JString>; 3] = [
        ["John", "Smith", "visited", "New", "York", "in", "1999", "."]
            .map(text::units)
            .to_vec(),
        [
            "Jean-Luc",
            "O'Brien",
            "e-mail:",
            "x@y.com",
            "3.5kg",
            "ÉCOLE",
            "東京",
            "naïve",
            "Ⅻ",
            "a\u{301}b",
            "  ",
            "",
            "Red",
            "Cross",
            "Society",
        ]
        .map(text::units)
        .to_vec(),
        vec![text::units("London")],
    ];
    let order: [Option<usize>; 5] = [Some(0), Some(1), Some(2), None, Some(0)];
    let mut actual = Vec::new();
    let mut sentence_id = 0u64;
    for (d, g) in generators.iter_mut().enumerate() {
        for (pass, which) in order.iter().enumerate() {
            let Some(si) = which else {
                g.clear_adaptive_data();
                continue;
            };
            sentence_id += 1;
            let toks = &sentences[*si];
            let tags: Vec<JString> = (0..toks.len())
                .map(|i| text::units(&format!("T{i}")))
                .collect();
            let outcomes: Vec<JString> = (0..toks.len())
                .map(|i| text::units(&format!("O{pass}-{i}")))
                .collect();
            for i in 0..toks.len() {
                let mut feats = Vec::new();
                g.create_features(&mut feats, toks, i, &tags, sentence_id)
                    .unwrap();
                actual.push((format!("G{d}\t{pass}\t{i}"), feats));
            }
            g.update_adaptive_data(toks, &outcomes);
        }
    }
    let expected = rows("ner_featuregen.tsv");
    for (row, (key, feats)) in expected.iter().zip(&actual) {
        let mut f = row.split('\t');
        let k = format!(
            "{}\t{}\t{}",
            f.next().unwrap(),
            f.next().unwrap(),
            f.next().unwrap()
        );
        let e: Vec<JString> = f.map(unesc_units).collect();
        assert_eq!(&k, key);
        let shown: Vec<String> = feats.iter().map(|x| text::to_string(x)).collect();
        assert_eq!(feats, &e, "{row} -- got {shown:?}");
    }
    assert_eq!(actual.len(), expected.len());
}
