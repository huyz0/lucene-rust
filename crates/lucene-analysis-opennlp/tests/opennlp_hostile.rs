//! Hostile and hand-built inputs for the OpenNLP port: model packages with
//! a manifest, artifact or event model Java refuses (each must fail the way
//! `BaseModel` and `GenericModelReader` do, never panic), feature
//! generator descriptors `GeneratorFactory` rejects, factories used out of
//! order, and every truncation of the fixture models.
#![allow(clippy::arithmetic_side_effects)] // test code: values built here

#[path = "../../lucene-analysis/tests/support/mod.rs"]
mod support;

use std::sync::Arc;

use lucene_analysis::factory::args::JavaArgs;
use lucene_analysis::factory::{
    AnalysisFactory, FactoryClass, FactoryError, JavaException, MapResourceLoader,
    TokenFilterFactory, TokenizerFactory,
};
use lucene_analysis::{AnalysisError, KeywordTokenizer, TokenStream, Tokenizer as _};
use lucene_analysis_opennlp::ops::LemmatizerOp;
use lucene_analysis_opennlp::tools::chunker::{Chunker, ChunkerModel};
use lucene_analysis_opennlp::tools::featuregen;
use lucene_analysis_opennlp::tools::lemmatizer::{Lemmatizer, LemmatizerModel};
use lucene_analysis_opennlp::tools::ml::{
    java_double_compare, BeamSearch, MaxentModel, ModelType, SequenceContext,
};
use lucene_analysis_opennlp::tools::model::{BaseModel, Component, Version};
use lucene_analysis_opennlp::tools::postag::{PosModel, PosTagger};
use lucene_analysis_opennlp::tools::sentdetect::{SentenceDetector, SentenceModel};
use lucene_analysis_opennlp::tools::text::{self, JString};
use lucene_analysis_opennlp::tools::tokenize::{Tokenizer, TokenizerModel};
use lucene_analysis_opennlp::tools::zip::ZipReader;
use lucene_analysis_opennlp::{
    OpenNlpChunkerFilterFactory, OpenNlpLemmatizerFilterFactory, OpenNlpPosFilterFactory,
    OpenNlpTokenizerFactory,
};
use support::data_dir;

const DIR: &str = "analysis_opennlp";

fn fixture(file: &str) -> Vec<u8> {
    std::fs::read(data_dir(DIR) + file).unwrap()
}

/// The entries of a fixture model.
fn entries(file: &str) -> Vec<(String, Vec<u8>)> {
    let bytes = fixture(file);
    let mut r = ZipReader::new(&bytes);
    let mut out = Vec::new();
    while let Some(e) = r.next_entry().unwrap() {
        out.push((e.name, e.data));
    }
    out
}

/// A STORED zip of `entries`, as `ZipOutputStream` writes one.
fn zip(entries: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut v = Vec::new();
    for (name, data) in entries {
        v.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        v.extend_from_slice(&20u16.to_le_bytes());
        v.extend_from_slice(&0u16.to_le_bytes());
        v.extend_from_slice(&0u16.to_le_bytes());
        v.extend_from_slice(&[0; 4]);
        v.extend_from_slice(&crc32(data).to_le_bytes());
        v.extend_from_slice(&(data.len() as u32).to_le_bytes());
        v.extend_from_slice(&(data.len() as u32).to_le_bytes());
        v.extend_from_slice(&(name.len() as u16).to_le_bytes());
        v.extend_from_slice(&0u16.to_le_bytes());
        v.extend_from_slice(name.as_bytes());
        v.extend_from_slice(data);
    }
    v
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// A fixture model with its manifest edited by `edit` and its entries by
/// `entries_edit`.
fn variant(
    file: &str,
    edit: impl Fn(&str) -> String,
    entries_edit: impl Fn(&mut Vec<(String, Vec<u8>)>),
) -> Vec<u8> {
    let mut e = entries(file);
    for (name, data) in &mut e {
        if name == "manifest.properties" {
            *data = edit(&String::from_utf8_lossy(data)).into_bytes();
        }
    }
    entries_edit(&mut e);
    zip(&e)
}

fn manifest(file: &str, edit: impl Fn(&str) -> String) -> Vec<u8> {
    variant(file, edit, |_| {})
}

fn load_err(component: Component, bytes: &[u8]) -> FactoryError {
    BaseModel::load(component, bytes).unwrap_err()
}

#[test]
fn manifests_java_refuses() {
    use Component::*;
    let cases: Vec<(Component, Vec<u8>, JavaException, &str)> = vec![
        (
            Tokenizer,
            manifest("en-token.bin", |m| {
                m.replace("OpenNLP-Version=2.5.3", "OpenNLP-Version=3.0.0")
            }),
            JavaException::IllegalArgument,
            "Model version 3.0.0 is not supported",
        ),
        (
            Tokenizer,
            manifest("en-token.bin", |m| {
                m.replace("OpenNLP-Version=2.5.3", "OpenNLP-Version=2.5.3-SNAPSHOT")
            }),
            JavaException::IllegalArgument,
            "is a snapshot",
        ),
        (
            Tokenizer,
            manifest("en-token.bin", |m| {
                m.replace("OpenNLP-Version=2.5.3", "OpenNLP-Version=two")
            }),
            JavaException::IllegalArgument,
            "Unable to parse model version 'two'!",
        ),
        (
            Tokenizer,
            manifest("en-token.bin", |m| m.replace("OpenNLP-Version=2.5.3", "")),
            JavaException::IllegalArgument,
            "Missing OpenNLP-Version property",
        ),
        (
            Tokenizer,
            manifest("en-token.bin", |m| {
                m.replace("Component-Name=TokenizerME", "")
            }),
            JavaException::IllegalArgument,
            "Missing Component-Name property",
        ),
        (
            Tokenizer,
            manifest("en-token.bin", |m| m.replace("Language=eng", "")),
            JavaException::IllegalArgument,
            "Missing Language property",
        ),
        (
            Tokenizer,
            manifest("en-token.bin", |m| {
                m.replace("useAlphaNumericOptimization=true", "")
            }),
            JavaException::IllegalArgument,
            "useAlphaNumericOptimization is a mandatory property!",
        ),
        (
            SentenceDetector,
            manifest("en-sent.bin", |m| m.replace("useTokenEnd=true", "")),
            JavaException::IllegalArgument,
            "useTokenEnd is a mandatory property!",
        ),
        (
            SentenceDetector,
            manifest("en-sent.bin", |m| {
                m.replace(
                    "factory=opennlp.tools.sentdetect.SentenceDetectorFactory",
                    "factory=my.Factory",
                )
            }),
            JavaException::UnsupportedOperation,
            "the tool factory my.Factory",
        ),
        (
            SentenceDetector,
            manifest("en-sent.bin", |m| {
                m.replace(
                    "opennlp.tools.util.model.DictionarySerializer",
                    "opennlp.tools.util.model.ByteArraySerializer",
                )
            }),
            JavaException::IllegalArgument,
            "Abbreviations dictionary 'byte[]' has wrong type",
        ),
        (
            SentenceDetector,
            manifest("en-sent.bin", |m| {
                m.replace("DictionarySerializer", "MySerializer")
            }),
            JavaException::UnsupportedOperation,
            "the artifact serializer opennlp.tools.util.model.MySerializer",
        ),
        (
            PosTagger,
            manifest("en-pos-maxent.bin", |m| {
                m.replace(
                    "opennlp.tools.postag.POSTaggerFactory$POSDictionarySerializer",
                    "opennlp.tools.util.model.DictionarySerializer",
                )
            }),
            JavaException::IllegalArgument,
            "POSTag dictionary has wrong type!",
        ),
        (
            PosTagger,
            variant(
                "en-pos-maxent.bin",
                |m| {
                    format!("{m}serializer-class-ngram.dictionary=opennlp.tools.util.model.ByteArraySerializer\n")
                },
                |e| e.push(("ngram.dictionary".into(), b"x".to_vec())),
            ),
            JavaException::IllegalArgument,
            "NGram dictionary has wrong type!",
        ),
        (
            Tokenizer,
            manifest("en-token.bin", |m| {
                m.replace(
                    "factory=opennlp.tools.tokenize.TokenizerFactory",
                    "factory=opennlp.tools.chunker.ChunkerFactory",
                )
            }),
            JavaException::UnsupportedOperation,
            "a TokenizerME model with the tool factory opennlp.tools.chunker.ChunkerFactory",
        ),
        (
            Chunker,
            manifest("en-chunker.bin", |m| {
                format!("{m}serializer-class-manifest.properties=opennlp.tools.util.model.ByteArraySerializer\n")
            }),
            JavaException::IllegalArgument,
            "Missing the manifest.properties!",
        ),
    ];
    for (component, bytes, kind, message) in cases {
        let e = load_err(component, &bytes);
        assert_eq!(e.kind, kind, "{}", e.message);
        assert!(e.message.contains(message), "{} lacks {message}", e.message);
    }
}

#[test]
fn artifacts_java_refuses() {
    let without = |file: &str, entry: &'static str| {
        variant(
            file,
            |m| m.to_string(),
            move |e| e.retain(|(n, _)| n != entry),
        )
    };
    for (component, file, entry, message) in [
        (
            Component::SentenceDetector,
            "en-sent.bin",
            "sent.model",
            "Unable to find sent.model maxent model!",
        ),
        (
            Component::Tokenizer,
            "en-token.bin",
            "token.model",
            "Token model is incomplete!",
        ),
        (
            Component::PosTagger,
            "en-pos-ud.bin",
            "pos.model",
            "POS model is incomplete!",
        ),
        (
            Component::Chunker,
            "en-chunker.bin",
            "chunker.model",
            "Chunker model is incomplete!",
        ),
        (
            Component::Lemmatizer,
            "en-lemmatizer.bin",
            "lemmatizer.model",
            "Lemmatizer model is incomplete!",
        ),
    ] {
        let e = load_err(component, &without(file, entry));
        assert_eq!(e.kind, JavaException::IllegalArgument);
        assert!(e.message.ends_with(message), "{}", e.message);
    }
    // a tokenizer's event model in a sentence detector and back
    let token_model = entries("en-token.bin")
        .into_iter()
        .find(|(n, _)| n == "token.model")
        .unwrap()
        .1;
    let sent_model = entries("en-sent.bin")
        .into_iter()
        .find(|(n, _)| n == "sent.model")
        .unwrap()
        .1;
    let swap = |file: &str, entry: &'static str, data: Vec<u8>| {
        variant(
            file,
            |m| m.to_string(),
            move |e| {
                for (n, d) in e.iter_mut() {
                    if n == entry {
                        *d = data.clone();
                    }
                }
            },
        )
    };
    let e = load_err(
        Component::SentenceDetector,
        &swap("en-sent.bin", "sent.model", token_model),
    );
    assert!(
        e.message
            .ends_with("not compatible with the sentence detector!"),
        "{}",
        e.message
    );
    let e = load_err(
        Component::Tokenizer,
        &swap("en-token.bin", "token.model", sent_model),
    );
    assert!(
        e.message.ends_with("not compatible with the tokenizer!"),
        "{}",
        e.message
    );
    // unknown extensions, a name without one, a package without a manifest
    let add = |name: &'static str| {
        variant(
            "en-chunker.bin",
            |m| m.to_string(),
            move |e| e.push((name.into(), b"x".to_vec())),
        )
    };
    let e = load_err(Component::Chunker, &add("notes.xyz"));
    assert_eq!(e.kind, JavaException::Io);
    assert!(
        e.message.ends_with("Unknown artifact format: xyz"),
        "{}",
        e.message
    );
    let e = load_err(Component::Chunker, &add("notes."));
    assert!(
        e.message
            .ends_with("Entry name must have type extension: notes."),
        "{}",
        e.message
    );
    let e = load_err(Component::Chunker, &add("tags.tagdict"));
    assert!(
        e.message.ends_with("Unknown artifact format: tagdict"),
        "{}",
        e.message
    );
    // the other serializers by name, and the plain-byte extensions
    let ok = variant(
        "en-chunker.bin",
        |m| {
            format!(
                "{m}serializer-class-a.bin=opennlp.tools.util.model.GenericModelSerializer\n\
                 serializer-class-b.bin=opennlp.tools.util.model.PropertiesSerializer\n"
            )
        },
        |e| {
            let model = e
                .iter()
                .find(|(n, _)| n == "chunker.model")
                .unwrap()
                .1
                .clone();
            e.push(("a.bin".into(), model));
            e.push(("b.bin".into(), b"k=v".to_vec()));
            e.push(("c.txt".into(), b"x".to_vec()));
            e.push(("d.html".into(), b"x".to_vec()));
            e.push(("e.dictionary".into(), b"<dictionary/>".to_vec()));
        },
    );
    let m = BaseModel::load(Component::Chunker, &ok).unwrap();
    assert!(m.dictionary("e.dictionary").is_some() && m.dictionary("c.txt").is_none());
    assert!(m.bytes("c.txt").is_some() && m.bytes("e.dictionary").is_none());
    assert!(m.pos_dictionary("e.dictionary").is_none());
    assert_eq!(m.component(), Component::Chunker);
    let e = load_err(Component::Chunker, &zip(&[("x.txt".into(), b"x".to_vec())]));
    assert_eq!(e.kind, JavaException::NullPointer);
    // a corrupt manifest
    let e = load_err(
        Component::Chunker,
        &manifest("en-chunker.bin", |m| format!("{m}x=\\uZZ\n")),
    );
    assert_eq!(e.kind, JavaException::IllegalArgument);
}

#[test]
fn versions() {
    let v = Version::parse("1.5.0").unwrap();
    assert_eq!((v.major, v.minor, v.revision, v.snapshot), (1, 5, 0, false));
    assert_eq!(v.to_string(), "1.5.0");
    let s = Version::parse("2.0.1-SNAPSHOT").unwrap();
    assert!(s.snapshot);
    assert_eq!(s.to_string(), "2.0.1-SNAPSHOT");
    assert!(Version::parse("1.5")
        .unwrap_err()
        .contains("expected two dots"));
    assert!(Version::parse("1.x.0")
        .unwrap_err()
        .contains("For input string"));
    assert!(Version::parse("1-2.3.4").is_err());
}

#[test]
fn every_truncation_of_every_model_fails_cleanly() {
    for (file, component) in [
        ("en-sent.bin", Component::SentenceDetector),
        ("en-token.bin", Component::Tokenizer),
        ("en-pos-maxent.bin", Component::PosTagger),
        ("en-lemmatizer.bin", Component::Lemmatizer),
    ] {
        let bytes = fixture(file);
        let step = (bytes.len() / 300).max(1);
        for cut in (0..bytes.len()).step_by(step) {
            // a cut inside the package fails or (past the last entry) loads
            let _ = BaseModel::load(component, &bytes[..cut]);
        }
        // flipped bytes, too: an error or a model, never a panic
        for at in (0..bytes.len()).step_by(step) {
            let mut b = bytes.clone();
            b[at] ^= 0x5A;
            let _ = BaseModel::load(component, &b);
        }
    }
}

/// `DataOutputStream` for hand-built event models.
#[derive(Default)]
struct Out(Vec<u8>);

impl Out {
    fn int(&mut self, i: i32) -> &mut Self {
        self.0.extend_from_slice(&i.to_be_bytes());
        self
    }
    fn double(&mut self, d: f64) -> &mut Self {
        self.0.extend_from_slice(&d.to_be_bytes());
        self
    }
    fn utf(&mut self, s: &str) -> &mut Self {
        let mut b = Vec::new();
        for u in s.encode_utf16() {
            match u {
                1..=0x7F => b.push(u as u8),
                0 | 0x80..=0x7FF => {
                    b.push(0xC0 | (u >> 6) as u8);
                    b.push(0x80 | (u & 0x3F) as u8);
                }
                _ => {
                    b.push(0xE0 | (u >> 12) as u8);
                    b.push(0x80 | ((u >> 6) & 0x3F) as u8);
                    b.push(0x80 | (u & 0x3F) as u8);
                }
            }
        }
        self.0.extend_from_slice(&(b.len() as u16).to_be_bytes());
        self.0.extend_from_slice(&b);
        self
    }
    fn raw(&mut self, b: &[u8]) -> &mut Self {
        self.0.extend_from_slice(b);
        self
    }
}

/// A GIS model: outcomes `a`, `b`; predicate `x` weighs 2 on `b`, `y` 1 on
/// `a`.
fn gis() -> Out {
    let mut o = Out::default();
    o.utf("GIS").int(1).double(0.0);
    o.int(2).utf("a").utf("b");
    o.int(2).utf("1 1").utf("1 0");
    o.int(2).utf("x").utf("y");
    o.double(2.0).double(1.0);
    o
}

#[test]
fn event_models() {
    let m = MaxentModel::read(&gis().0).unwrap();
    assert_eq!(m.model_type(), ModelType::Gis);
    assert_eq!(
        (
            m.num_outcomes(),
            m.outcome_name(1),
            m.index("b"),
            m.index("z")
        ),
        (2, "b", Some(1), None)
    );
    let mut probs = [0.0; 2];
    m.eval(
        &[text::units("x"), text::units("x"), text::units("unknown")],
        &mut probs,
    );
    assert!(probs[1] > probs[0]);
    assert_eq!(MaxentModel::best_outcome(&probs), 1);
    assert_eq!(MaxentModel::best_outcome(&[f64::NAN, 1.0]), 0);
    assert_eq!(MaxentModel::best_outcome(&[0.5, 1.0]), 1);
    assert_eq!(MaxentModel::best_outcome(&[1.0, f64::NAN]), 0);
    // perceptron: scores scaled by the largest magnitude
    let mut p = Out::default();
    p.utf("Perceptron")
        .int(2)
        .utf("a")
        .utf("b")
        .int(1)
        .utf("1 0 1")
        .int(1)
        .utf("x")
        .double(-5.0)
        .double(3.0);
    let m = MaxentModel::read(&p.0).unwrap();
    assert_eq!(m.model_type(), ModelType::Perceptron);
    m.eval(&[text::units("x")], &mut probs);
    assert!(probs[1] > probs[0] && (probs[0] + probs[1] - 1.0).abs() < 1e-12);
    // chunked strings and a predicate declared twice (the later wins), one
    // with no parameters (as unknown)
    let mut c = Out::default();
    c.utf("CHUNKED-MODEL-PARAMS:2").utf("Perc").utf("eptron");
    c.int(1)
        .utf("o")
        .int(1)
        .utf("1 0")
        .int(3)
        .utf("p")
        .utf("p")
        .utf("q")
        .double(1.0);
    let m = MaxentModel::read(&c.0).unwrap();
    m.eval(&[text::units("p"), text::units("q")], &mut [0.0]);
    let refused = |bytes: &[u8]| MaxentModel::read(bytes).unwrap_err();
    for t in ["QN", "NaiveBayes"] {
        assert_eq!(
            refused(&Out::default().utf(t).0).kind,
            JavaException::UnsupportedOperation
        );
    }
    assert!(refused(&Out::default().utf("SVM").0)
        .message
        .contains("Unknown model format: SVM"));
    assert!(refused(&Out::default().utf("CHUNKED-MODEL-PARAMS:x").0)
        .message
        .contains("For input string"));
    let header = || {
        let mut o = Out::default();
        o.utf("Perceptron");
        o
    };
    assert!(refused(&header().int(-1).0)
        .message
        .contains("NegativeArraySizeException"));
    assert!(refused(&header().int(1_000_000).0)
        .message
        .contains("EOFException"));
    assert!(refused(&header().int(1).utf("a").int(1).utf("").0)
        .message
        .contains("out of bounds for length 0"));
    assert!(refused(&header().int(1).utf("a").int(1).utf("1 x").0)
        .message
        .contains("For input string"));
    assert!(
        refused(&header().int(1).utf("a").int(1).utf("1 7").int(1).utf("p").0)
            .message
            .contains("is not one of")
    );
    // ... but a pattern no predicate uses is never looked at, as in Java
    let mut unused = header();
    unused
        .int(1)
        .utf("a")
        .int(2)
        .utf("0 7")
        .utf("1 0")
        .int(1)
        .utf("p")
        .double(1.0);
    assert!(MaxentModel::read(&unused.0).is_ok());
    assert!(refused(
        &header()
            .int(1)
            .utf("a")
            .int(1)
            .utf("2 0")
            .int(1)
            .utf("p")
            .double(1.0)
            .double(1.0)
            .0
    )
    .message
    .contains("out of bounds for length 1"));
    assert!(
        refused(&header().int(1).utf("a").int(1).utf("1 0").int(1).utf("p").0)
            .message
            .contains("EOFException")
    );
    // malformed modified UTF-8
    for bad in [
        &[0, 1, 0x80][..],
        &[0, 1, 0xC3],
        &[0, 2, 0xC3, 0x41],
        &[0, 2, 0xE2, 0x82],
        &[0, 3, 0xE2, 0x41, 0x82],
        &[0, 1, 0xF0],
        &[0, 5],
    ] {
        let e = refused(bad);
        assert_eq!(e.kind, JavaException::Io, "{bad:?}");
        assert!(
            e.message.contains("UTFDataFormatException") || e.message.contains("EOF"),
            "{}",
            e.message
        );
    }
    // a two- and a three-byte character read back
    let m = MaxentModel::read(
        &Out::default()
            .utf("Perceptron")
            .int(1)
            .utf("é€")
            .int(0)
            .int(0)
            .0,
    )
    .unwrap();
    assert_eq!(m.outcome(0), text::units("é€").as_slice());
    assert_eq!(m.outcomes().len(), 1);
    let _ = Out::default().raw(&[]);
}

#[test]
fn double_ordering() {
    use std::cmp::Ordering::*;
    assert_eq!(java_double_compare(-0.0, 0.0), Less);
    assert_eq!(java_double_compare(f64::NAN, 1.0), Greater);
    assert_eq!(java_double_compare(f64::NAN, f64::NAN), Equal);
    assert_eq!(java_double_compare(1.0, 2.0), Less);
}

/// A context accepting only outcome `only`.
struct Only(u32);

impl SequenceContext for Only {
    fn context(&mut self, _: usize, _: &[u32], _: &[JString], f: &mut Vec<JString>) {
        f.push(text::units("x"));
    }
    fn valid(&self, _: usize, _: &[u32], outcome: u32, _: &[JString]) -> bool {
        outcome == self.0
    }
}

#[test]
fn beam_search_prunes_and_falls_back() {
    // `x` makes `a` impossible: `a` (probability 0) is pruned at -100000, so a
    // search that only admits `a` ends with nothing (Java's null)
    let mut o = Out::default();
    o.utf("GIS").int(1).double(0.0);
    o.int(2)
        .utf("a")
        .utf("b")
        .int(1)
        .utf("1 0")
        .int(1)
        .utf("x")
        .double(-1000.0);
    let m = MaxentModel::read(&o.0).unwrap();
    let e = BeamSearch::new(3)
        .best_sequence(&m, 2, &mut Only(0))
        .unwrap_err();
    assert!(
        matches!(e, AnalysisError::IllegalState(ref s) if s.starts_with("NullPointerException"))
    );
    assert_eq!(
        BeamSearch::new(3)
            .best_sequence(&m, 2, &mut Only(1))
            .unwrap(),
        vec![1, 1]
    );
    // a beam of one keeps only the best outcome; when it is invalid every
    // valid outcome advances
    let m = MaxentModel::read(&gis().0).unwrap();
    assert_eq!(
        BeamSearch::new(1)
            .best_sequence(&m, 3, &mut Only(0))
            .unwrap(),
        vec![0, 0, 0]
    );
    // a beam below 1 is `new PriorityQueue<>(size)`'s
    // IllegalArgumentException, even over no positions
    for (size, len) in [(0, 0), (0, 1), (-3, 2)] {
        assert!(matches!(
            BeamSearch::new(size).best_sequence(&m, len, &mut Only(0)),
            Err(AnalysisError::IllegalArgument(ref s)) if s.is_empty()
        ));
    }
}

#[test]
fn feature_descriptors_java_refuses() {
    let fg = "opennlp.tools.util.featuregen.";
    let g = |inner: &str| format!("<featureGenerators>{inner}</featureGenerators>");
    let gen = |class: &str, args: &str| {
        format!("<generator class=\"{fg}{class}FeatureGeneratorFactory\">{args}</generator>")
    };
    let cases: Vec<(String, &str)> = vec![
        ("<generators/>".into(), "Classic configuration format"),
        ("<featureGenerators".into(), "Descriptor is not valid XML"),
        (g("<other/>"), "Unexpected element: featureGenerators"),
        (g(""), "must have one or more generators"),
        (g("<generator/>"), "generator must have class attribute"),
        (
            g("<generator class=\"my.Gen\"/>"),
            "is not supported by this port",
        ),
        (
            g(&gen(
                "Window",
                "<int name=\"prevLength\">1</int><int name=\"nextLength\">1</int>",
            )),
            "must contain an aggregator element",
        ),
        (
            g(&gen("Window", &gen("Token", ""))),
            "parameter prevLength must be set!",
        ),
        (
            g(&gen("Suffix", "<bool name=\"length\">true</bool>")),
            "parameter length must be integer!",
        ),
        (
            g(&gen("Token", "<int name=\"lowercase\">1</int>")),
            "parameter lowercase must be boolean!",
        ),
        (
            g(&gen("Suffix", "<int name=\"length\">x</int>")),
            "NumberFormatException",
        ),
        (
            g(&gen("Suffix", "<int name=\"length\"/>")),
            "NullPointerException",
        ),
        (
            g(&gen("Suffix", "<int name=\"length\"><b/></int>")),
            "ClassCastException",
        ),
        (
            g(&gen("Suffix", "<long name=\"l\">x</long>")),
            "NumberFormatException",
        ),
        (
            g(&gen("Suffix", "<float name=\"l\">x</float>")),
            "NumberFormatException",
        ),
        (
            g(&gen("Suffix", "<map name=\"l\">x</map>")),
            "child element must be one of",
        ),
        (
            g(&gen(
                "Window",
                &format!(
                    "<int name=\"prevLength\">1</int><int name=\"nextLength\">1</int>{}",
                    gen("Suffix", "<map/>")
                ),
            )),
            "RuntimeException",
        ),
        (
            g(&gen(
                "Window",
                &format!(
                    "<int name=\"prevLength\">1</int><int name=\"nextLength\">1</int>{}",
                    gen("Nope", "")
                ),
            )),
            "is not supported",
        ),
    ];
    for (descriptor, message) in cases {
        let e = featuregen::create(descriptor.as_bytes()).unwrap_err();
        assert!(format!("{e:?}").contains(message), "{descriptor}: {e:?}");
    }
    // the typed arguments the factories ignore parse, and negative lengths
    // give no features
    let mut ok = featuregen::create(
        g(&format!(
            "{}{}",
            gen(
                "Suffix",
                "<int name=\"length\">-2</int><long name=\"a\">5</long><float name=\"b\">1.5f</float><double name=\"c\">2</double><str name=\"d\">s</str>"
            ),
            gen("Prefix", "<int name=\"length\">-1</int>")
        ))
        .as_bytes(),
    )
    .unwrap();
    let mut f = Vec::new();
    ok.create_features(&mut f, &[text::units("word")], 0, &[], 1)
        .unwrap();
    assert!(f.is_empty());
    // a tag generator looking ahead reads past the previous tags: Java's
    // ArrayIndexOutOfBoundsException
    let mut ahead = featuregen::create(
        g(&gen(
            "Window",
            &format!(
                "<int name=\"prevLength\">0</int><int name=\"nextLength\">2</int>{}",
                gen("PosTagger", "")
            ),
        ))
        .as_bytes(),
    )
    .unwrap();
    let toks: Vec<JString> = ["a", "b", "c"].map(text::units).to_vec();
    let e = ahead.create_features(&mut f, &toks, 0, &[], 1).unwrap_err();
    assert!(
        format!("{e:?}").contains("ArrayIndexOutOfBoundsException"),
        "{e:?}"
    );
    // a window as wide as an int: only the sentence's neighbours count
    let mut wide = featuregen::create(
        g(&gen(
            "Window",
            &format!(
                "<int name=\"prevLength\">2147483647</int><int name=\"nextLength\">2147483647</int>{}",
                gen("Token", "")
            ),
        ))
        .as_bytes(),
    )
    .unwrap();
    let mut f = Vec::new();
    wide.create_features(&mut f, &toks, 1, &[], 2).unwrap();
    assert_eq!(f.len(), 3);
    // the cache holds 100 positions: a longer sentence evicts the eldest
    let mut cached = featuregen::create(
        format!(
            "<featureGenerators cache=\"true\">{}</featureGenerators>",
            gen("Token", "")
        )
        .as_bytes(),
    )
    .unwrap();
    let long: Vec<JString> = (0..150).map(|i| text::units(&format!("w{i}"))).collect();
    for round in 0..2 {
        for i in 0..long.len() {
            let mut f = Vec::new();
            cached.create_features(&mut f, &long, i, &[], 7).unwrap();
            assert_eq!(f, vec![text::units(&format!("w=w{i}"))], "{round}");
        }
    }
}

fn pos_model() -> PosModel {
    PosModel::load(&fixture("en-pos-maxent.bin")).unwrap()
}

#[test]
fn taggers_with_odd_beams() {
    let bad = manifest("en-pos-maxent.bin", |m| {
        m.replace("BeamSize=3", "BeamSize=x")
    });
    let e = PosTagger::new(&PosModel::load(&bad).unwrap()).unwrap_err();
    assert!(
        matches!(e, AnalysisError::IllegalArgument(ref m) if m.starts_with("NumberFormatException"))
    );
    let zero = manifest("en-pos-maxent.bin", |m| {
        m.replace("BeamSize=3", "BeamSize=-4")
    });
    // `new PriorityQueue<>(-4)`: refused when tagging starts, even an
    // empty sentence
    let mut t = PosTagger::new(&PosModel::load(&zero).unwrap()).unwrap();
    for sentence in [&[][..], &[text::units("dog")][..]] {
        assert!(
            matches!(t.tag(sentence), Err(AnalysisError::IllegalArgument(ref m)) if m.is_empty())
        );
    }
    let chunker = ChunkerModel::load(&manifest("en-chunker.bin", |m| {
        m.replace("BeamSize=10", "BeamSize=1x")
    }))
    .unwrap();
    assert!(Chunker::new(&chunker).is_err());
    let lemma = LemmatizerModel::load(&manifest("en-lemmatizer.bin", |m| {
        m.replace("BeamSize=3", "BeamSize=")
    }))
    .unwrap();
    assert!(Lemmatizer::new(&lemma).is_err());
    // a bad alphanumeric pattern fails when the tokenizer is made
    let tok = TokenizerModel::load(&manifest("en-token.bin", |m| {
        m.replace(
            "alphaNumericPattern=^[A-Za-z0-9]+$",
            "alphaNumericPattern=[",
        )
    }))
    .unwrap();
    assert!(Tokenizer::new(tok).is_err());
    // a chunk model whose outcomes include one too short for an I- check
    let mut o = Out::default();
    o.utf("GIS").int(1).double(0.0);
    o.int(2)
        .utf("X")
        .utf("I-NP")
        .int(1)
        .utf("1 0")
        .int(1)
        .utf("w0=a")
        .double(50.0);
    let model = variant(
        "en-chunker.bin",
        |m| m.to_string(),
        |e| {
            for (n, d) in e.iter_mut() {
                if n == "chunker.model" {
                    *d = o.0.clone();
                }
            }
        },
    );
    let mut c = Chunker::new(&ChunkerModel::load(&model).unwrap()).unwrap();
    let words = [text::units("a"), text::units("b")];
    assert!(c
        .chunk(&words, &[text::units("DT"), text::units("NN")])
        .is_err());
    // a sentence detector over a text of end-of-sentence characters only
    let mut d = SentenceDetector::new(SentenceModel::load(&fixture("en-sent.bin")).unwrap());
    let _ = d.sent_pos_detect(&text::units(". . ?!. \n"));
    assert!(d.sent_pos_detect(&text::units("   ")).is_empty());
    assert!(d.sent_pos_detect(&[]).is_empty());
    let _ = pos_model();
}

#[test]
fn lemmatizer_op_without_lemmatizers() {
    let mut op = LemmatizerOp::new(None, None);
    assert!(op
        .lemmatize(&[text::units("x")], &[text::units("NN")])
        .is_err());
}

fn loader() -> MapResourceLoader {
    let mut l = MapResourceLoader::new();
    for f in [
        "en-sent.bin",
        "en-token.bin",
        "en-pos-maxent.bin",
        "en-chunker.bin",
        "en-lemmatizer.bin",
        "en-lemmas.dict",
    ] {
        l = l.with(f, fixture(f));
    }
    // a package whose manifest fails its CRC check
    let mut corrupt = zip(&[("manifest.properties".into(), b"k=v".to_vec())]);
    let last = corrupt.len() - 1;
    corrupt[last] ^= 1;
    l.with("corrupt.bin", corrupt).with("bad.dict", b"one\ttwo")
}

fn args(pairs: &[(&str, &str)]) -> JavaArgs {
    JavaArgs::from_pairs(pairs)
}

fn keyword() -> Box<dyn TokenStream> {
    let mut k = KeywordTokenizer::new();
    k.set_reader(Box::new(lucene_analysis::StrReader::new("x")))
        .unwrap();
    Box::new(k)
}

#[test]
fn factories_out_of_order_and_misconfigured() {
    // created before inform: Java dereferences a null model
    let t = OpenNlpTokenizerFactory::from_args(&mut args(&[
        ("sentenceModel", "en-sent.bin"),
        ("tokenizerModel", "en-token.bin"),
    ]))
    .unwrap();
    assert!(t.create().is_err());
    let p =
        OpenNlpPosFilterFactory::from_args(&mut args(&[("posTaggerModel", "en-pos-maxent.bin")]))
            .unwrap();
    assert!(p.create(keyword()).is_err());
    let mut c =
        OpenNlpChunkerFilterFactory::from_args(&mut args(&[("chunkerModel", "en-chunker.bin")]))
            .unwrap();
    assert!(c.create(keyword()).is_err());
    assert!(c.is_resource_loader_aware());
    let _ = c.base_mut();
    c.inform(&loader()).unwrap();
    assert!(c.create(keyword()).is_ok());
    let l =
        OpenNlpLemmatizerFilterFactory::from_args(&mut args(&[("dictionary", "en-lemmas.dict")]))
            .unwrap();
    let mut s = l.create(keyword()).unwrap();
    s.reset().unwrap();
    assert!(s.increment_token().is_err());
    // read errors: the POS and chunker factories make them illegal arguments
    for (mut f, name) in [
        (
            Box::new(
                OpenNlpPosFilterFactory::from_args(&mut args(&[("posTaggerModel", "corrupt.bin")]))
                    .unwrap(),
            ) as Box<dyn AnalysisFactory>,
            "pos",
        ),
        (
            Box::new(
                OpenNlpChunkerFilterFactory::from_args(&mut args(&[(
                    "chunkerModel",
                    "corrupt.bin",
                )]))
                .unwrap(),
            ),
            "chunker",
        ),
    ] {
        assert_eq!(
            f.inform(&loader()).unwrap_err().kind,
            JavaException::IllegalArgument,
            "{name}"
        );
    }
    let mut t = OpenNlpTokenizerFactory::from_args(&mut args(&[
        ("sentenceModel", "en-sent.bin"),
        ("tokenizerModel", "corrupt.bin"),
    ]))
    .unwrap();
    assert_eq!(t.inform(&loader()).unwrap_err().kind, JavaException::Io);
    let mut l = OpenNlpLemmatizerFilterFactory::from_args(&mut args(&[("dictionary", "bad.dict")]))
        .unwrap();
    assert_eq!(
        l.inform(&loader()).unwrap_err().kind,
        JavaException::ArrayIndexOutOfBounds
    );
    let mut l =
        OpenNlpLemmatizerFilterFactory::from_args(&mut args(&[("lemmatizerModel", "corrupt.bin")]))
            .unwrap();
    assert_eq!(l.inform(&loader()).unwrap_err().kind, JavaException::Io);
    // arguments
    let e = OpenNlpChunkerFilterFactory::from_args(&mut args(&[("x", "y")]))
        .err()
        .unwrap();
    assert!(e.message.starts_with("Unknown parameters"), "{}", e.message);
    let e = OpenNlpLemmatizerFilterFactory::from_args(&mut args(&[]))
        .err()
        .unwrap();
    assert!(e
        .message
        .contains("at least one of 'dictionary' and 'lemmatizerModel'"));
    let e =
        OpenNlpLemmatizerFilterFactory::from_args(&mut args(&[("dictionary", "d"), ("x", "y")]))
            .err()
            .unwrap();
    assert!(e.message.starts_with("Unknown parameters"));
    let _ = Arc::new(0);
}
