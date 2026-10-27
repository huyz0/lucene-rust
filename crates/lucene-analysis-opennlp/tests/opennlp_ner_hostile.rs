//! Hostile and hand-built inputs for the name finder: model packages and
//! feature generator descriptors Java refuses (each must fail the way
//! `TokenNameFinderModel`, `NameFinderME` and `GeneratorFactory` do, never
//! panic), and the edges of the generators the fixtures do not reach.
#![allow(clippy::arithmetic_side_effects)] // test code: values built here

#[path = "../../lucene-analysis/tests/support/mod.rs"]
mod support;

use std::collections::HashMap;
use std::sync::Arc;

use lucene_analysis::factory::{JavaException, MapResourceLoader};
use lucene_analysis::AnalysisError;
use lucene_analysis_opennlp::factory::ner_tagger_model;
use lucene_analysis_opennlp::ops::NerTaggerOp;
use lucene_analysis_opennlp::tools::dictionary::Dictionary;
use lucene_analysis_opennlp::tools::featuregen::{self, NoResources, Owner, Resources};
use lucene_analysis_opennlp::tools::model::Artifact;
use lucene_analysis_opennlp::tools::namefind::{NameFinder, NameFinderModel, SequenceCodec};
use lucene_analysis_opennlp::tools::postag::PosModel;
use lucene_analysis_opennlp::tools::text::{self, JString};
use lucene_analysis_opennlp::tools::zip::ZipReader;
use support::data_dir;

const DIR: &str = "analysis_opennlp_ner";
const FG: &str = "opennlp.tools.util.featuregen.";

fn fixture(file: &str) -> Vec<u8> {
    std::fs::read(data_dir(DIR) + file).unwrap()
}

/// A STORED zip of `entries`, as `ZipOutputStream` writes one.
fn zip(entries: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut v = Vec::new();
    for (name, data) in entries {
        v.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        v.extend_from_slice(&20u16.to_le_bytes());
        v.extend_from_slice(&[0; 8]);
        v.extend_from_slice(&crc32fast::hash(data).to_le_bytes());
        v.extend_from_slice(&(data.len() as u32).to_le_bytes());
        v.extend_from_slice(&(data.len() as u32).to_le_bytes());
        v.extend_from_slice(&(name.len() as u16).to_le_bytes());
        v.extend_from_slice(&0u16.to_le_bytes());
        v.extend_from_slice(name.as_bytes());
        v.extend_from_slice(data);
    }
    v
}

/// `ner-maxent.bin` with its manifest edited by `edit` and entries by
/// `entries_edit`.
fn variant(
    edit: impl Fn(&str) -> String,
    entries_edit: impl Fn(&mut Vec<(String, Vec<u8>)>),
) -> Vec<u8> {
    let bytes = fixture("ner-maxent.bin");
    let mut r = ZipReader::new(&bytes);
    let mut e = Vec::new();
    while let Some(entry) = r.next_entry().unwrap() {
        e.push((entry.name, entry.data));
    }
    for (name, data) in &mut e {
        if name == "manifest.properties" {
            *data = edit(&String::from_utf8_lossy(data)).into_bytes();
        }
    }
    entries_edit(&mut e);
    zip(&e)
}

/// `ner-maxent.bin` with entry `name` set to `data`, read through the
/// serializer class `serializer` (when given).
fn with_entry(name: &str, data: &[u8], serializer: Option<&str>) -> Vec<u8> {
    variant(
        |m| match serializer {
            Some(s) => format!("{m}serializer-class-{name}={s}\n"),
            None => m.to_string(),
        },
        |e| {
            e.retain(|(n, _)| n != name);
            e.push((name.to_string(), data.to_vec()));
        },
    )
}

fn descriptor(generators: &str) -> Vec<u8> {
    with_entry(
        "generator.featuregen",
        format!("<featureGenerators>{generators}</featureGenerators>").as_bytes(),
        None,
    )
}

fn finder_err(bytes: &[u8]) -> AnalysisError {
    NameFinder::new(&NameFinderModel::load(bytes).unwrap()).unwrap_err()
}

fn u(s: &str) -> JString {
    text::units(s)
}

#[test]
fn models_java_refuses() {
    let maxent = NameFinderModel::load(&fixture("ner-maxent.bin")).unwrap();
    assert_eq!(maxent.sequence_codec(), SequenceCodec::Bio);
    let bilou = NameFinderModel::load(&fixture("ner-bilou.bin")).unwrap();
    assert_eq!(bilou.sequence_codec(), SequenceCodec::Bilou);
    let bad = NameFinderModel::load(&fixture("ner-badcodec.bin")).unwrap();
    assert_eq!(bad.sequence_codec(), SequenceCodec::Bio);
    let no_codec = variant(
        |m| {
            m.lines()
                .filter(|l| !l.starts_with("sequenceCodecImplName"))
                .map(|l| format!("{l}\n"))
                .collect()
        },
        |_| {},
    );
    assert_eq!(
        NameFinderModel::load(&no_codec).unwrap().sequence_codec(),
        SequenceCodec::Bio
    );

    // no event model
    let e = NameFinderModel::load(&variant(
        |m| m.to_string(),
        |e| e.retain(|(n, _)| n != "nameFinder.model"),
    ))
    .unwrap_err();
    assert!(
        e.message.contains("Token Name Finder model is incomplete!"),
        "{}",
        e.message
    );
    // a POS model is not a name finder's
    let e = NameFinderModel::load(&fixture("en-pos-ner.bin")).unwrap_err();
    assert!(e
        .message
        .contains("cannot load a model for the POSTaggerME"));

    // the cluster serializers close the zip stream: no package loads
    let words =
        "opennlp.tools.util.featuregen.WordClusterDictionary$WordClusterDictionarySerializer";
    let brown = "opennlp.tools.util.featuregen.BrownCluster$BrownClusterSerializer";
    for (data, serializer) in [(&b"john c1\n"[..], words), (b"0110\tJohn\t10\n", brown)] {
        let e =
            NameFinderModel::load(&with_entry("x.clusters", data, Some(serializer))).unwrap_err();
        assert_eq!(e.kind, JavaException::Io);
        assert_eq!(e.message, "IOException: Stream closed");
    }
    // ... after reading it: a bad frequency is Integer.parseInt's
    let e =
        NameFinderModel::load(&with_entry("x.clusters", b"0\ta\tmany", Some(brown))).unwrap_err();
    assert_eq!(e.kind, JavaException::NumberFormat);
    // a nested POS model that is not one
    let e = NameFinderModel::load(&with_entry(
        "pos.bin",
        b"not a zip",
        Some("opennlp.tools.util.model.POSModelSerializer"),
    ))
    .unwrap_err();
    assert_eq!(e.kind, JavaException::NullPointer, "{}", e.message);
    // a dictionary is not an abbreviation dictionary's wrong type here, but
    // a featuregen entry read as one is not a descriptor
    let dict = b"<dictionary><entry><token>a</token></entry></dictionary>";
    let e = finder_err(&with_entry(
        "generator.featuregen",
        dict,
        Some("opennlp.tools.util.model.DictionarySerializer"),
    ));
    assert!(format!("{e:?}").contains("ClassCastException"), "{e:?}");
}

#[test]
fn finders_java_refuses() {
    // the beam size is read before the generators are made
    let e = finder_err(&variant(
        |m| m.replace("BeamSize=3", "BeamSize=x"),
        |e| {
            for (n, d) in e.iter_mut() {
                if n == "generator.featuregen" {
                    *d = b"<broken".to_vec();
                }
            }
        },
    ));
    assert!(
        matches!(e, AnalysisError::IllegalArgument(ref m) if m.starts_with("NumberFormatException")),
        "{e:?}"
    );
    // a beam below 1 fails when a sentence is searched, even an empty one
    let zero = variant(|m| m.replace("BeamSize=3", "BeamSize=0"), |_| {});
    let mut f = NameFinder::new(&NameFinderModel::load(&zero).unwrap()).unwrap();
    assert!(f.find(&[]).is_err());
    assert!(f.find(&[u("John")]).is_err());

    // the descriptor's InvalidFormatException is the factory's
    // FeatureGeneratorCreationError
    let e = finder_err(&with_entry(
        "generator.featuregen",
        b"<featureGenerators",
        None,
    ));
    assert!(
        format!("{e:?}").contains(
            "TokenNameFinderModel.FeatureGeneratorCreationError: opennlp.tools.util.InvalidFormatException: Descriptor is not valid XML"
        ),
        "{e:?}"
    );
    let e = finder_err(&descriptor(""));
    assert!(
        format!("{e:?}").contains("FeatureGeneratorCreationError")
            && format!("{e:?}").contains("one or more generators"),
        "{e:?}"
    );
    // resources come from the package: it has no clusters
    let e = finder_err(&descriptor(&format!(
        "<generator class=\"{FG}BrownClusterTokenFeatureGeneratorFactory\"><str name=\"dict\">pos.bin</str></generator>"
    )));
    assert!(
        format!("{e:?}").contains(
            "RuntimeException: opennlp.tools.util.InvalidFormatException: Not a BrownLexicon resource for key: pos.bin"
        ),
        "{e:?}"
    );
    // but its dictionaries and POS models
    let bilou = NameFinderModel::load(&fixture("ner-bilou.bin")).unwrap();
    assert!(matches!(
        bilou.base().artifact("names.dict"),
        Some(Artifact::Dictionary(_))
    ));
    assert!(matches!(
        bilou.base().artifact("pos.bin"),
        Some(Artifact::PosModel(_))
    ));

    // an empty sentence has no names; the adaptive data survives a reset
    let maxent = NameFinderModel::load(&fixture("ner-maxent.bin")).unwrap();
    let mut op = NerTaggerOp::new(&maxent).unwrap();
    assert!(op.get_names(&[]).unwrap().is_empty());
    let first = op
        .get_names(&[u("John"), u("visited"), u("London")])
        .unwrap();
    op.reset();
    let again = op
        .get_names(&[u("John"), u("visited"), u("London")])
        .unwrap();
    assert_eq!(first, again);
    assert!(!first.is_empty());
}

#[test]
fn ops_factory_reads_models() {
    let loader = MapResourceLoader::new().with("ner.bin", fixture("ner-maxent.bin"));
    let model = ner_tagger_model("ner.bin", &loader).unwrap();
    assert!(NerTaggerOp::new(&model).is_ok());
    let e = ner_tagger_model("missing.bin", &loader).unwrap_err();
    assert_eq!(e.kind, JavaException::Io);
}

struct MapResources(HashMap<&'static str, Artifact>);

impl Resources for MapResources {
    fn resource(&self, key: &str) -> Option<&Artifact> {
        self.0.get(key)
    }
}

fn gen(class: &str, args: &str) -> String {
    format!("<generator class=\"{FG}{class}FeatureGeneratorFactory\">{args}</generator>")
}

fn generators(inner: &str) -> Vec<u8> {
    format!("<featureGenerators>{inner}</featureGenerators>").into_bytes()
}

#[test]
fn feature_descriptors_java_refuses() {
    let pos = PosModel::load(&fixture("en-pos-ner.bin")).unwrap();
    let mut map = HashMap::new();
    map.insert(
        "d",
        Artifact::Dictionary(Arc::new(
            Dictionary::read(b"<dictionary><entry><token>a</token></entry></dictionary>").unwrap(),
        )),
    );
    map.insert("pos", Artifact::PosModel(Arc::new(pos)));
    let resources = MapResources(map);
    let cases: Vec<(String, &str)> = vec![
        // top-level errors are the owner's; inside a factory, wrapped
        (
            "<generator/>".into(),
            "IllegalStateException: opennlp.tools.util.InvalidFormatException: generator must have class attribute",
        ),
        (
            gen("Window", "<generator/>"),
            "RuntimeException: opennlp.tools.util.InvalidFormatException: generator must have class attribute",
        ),
        (
            gen("Cached", &gen("Token", "")),
            "CachedFeatureGeneratorFactory cannot be specified manually",
        ),
        (
            gen("Cached", "<int name=\"x\">1</int>"),
            "Could not find containing generator element!",
        ),
        (
            gen("Aggregated", "<int name=\"generator#3\">1</int>"),
            "ClassCastException",
        ),
        (
            gen("CharacterNgram", "<int name=\"max\">2</int>"),
            "parameter min must be set!",
        ),
        (
            gen("WordCluster", "<str name=\"dict\">d</str>"),
            "parameter lowerCase must be set!",
        ),
        (
            gen("WordCluster", "<int name=\"dict\">1</int>"),
            "parameter dict must be double!",
        ),
        (
            gen(
                "WordCluster",
                "<str name=\"dict\">d</str><bool name=\"lowerCase\">true</bool>",
            ),
            "Not a WordClusterDictionary resource for key: d",
        ),
        (
            gen("BrownClusterBigram", "<str name=\"dict\">none</str>"),
            "Not a BrownLexicon resource for key: none",
        ),
        (
            gen("BrownClusterTokenClass", ""),
            "parameter dict must be set!",
        ),
        (
            gen("Dictionary", "<str name=\"dict\">pos</str>"),
            "No dictionary resource for key: pos",
        ),
        (
            gen("POSTaggerName", "<str name=\"model\">none</str>"),
            "NullPointerException: POSModel must not be NULL.",
        ),
        (
            gen("POSTaggerName", "<str name=\"model\">d</str>"),
            "ClassCastException",
        ),
        (
            gen("my.Own", ""),
            "is not supported by this port",
        ),
    ];
    for (inner, message) in cases {
        let e =
            featuregen::create_with(&generators(&inner), &resources, Owner::PosTagger).unwrap_err();
        assert!(format!("{e:?}").contains(message), "{inner}: {e:?}");
    }
    // a POS model whose beam the tagger refuses: when it is made, or when
    // it tags
    let bytes = fixture("en-pos-ner.bin");
    let mut r = ZipReader::new(&bytes);
    let mut entries = Vec::new();
    while let Some(e) = r.next_entry().unwrap() {
        entries.push((e.name, e.data));
    }
    let with_beam = |beam: &str| {
        let mut e = entries.clone();
        for (n, d) in e.iter_mut() {
            if n == "manifest.properties" {
                *d = String::from_utf8_lossy(d)
                    .replace("BeamSize=3", beam)
                    .into_bytes();
            }
        }
        Artifact::PosModel(Arc::new(PosModel::load(&zip(&e)).unwrap()))
    };
    let pos_gen = generators(&gen("POSTaggerName", "<str name=\"model\">pos</str>"));
    let bad = MapResources(HashMap::from([("pos", with_beam("BeamSize=y"))]));
    let e = featuregen::create_with(&pos_gen, &bad, Owner::NameFinder).unwrap_err();
    assert!(
        matches!(e, AnalysisError::IllegalArgument(ref m) if m.starts_with("NumberFormatException")),
        "{e:?}"
    );
    let zero = MapResources(HashMap::from([("pos", with_beam("BeamSize=0"))]));
    let mut g = featuregen::create_with(&pos_gen, &zero, Owner::NameFinder).unwrap();
    let mut f = Vec::new();
    assert!(g.create_features(&mut f, &[u("a")], 0, &[], 1).is_err());

    // character n-grams: a negative length fails on the first token, a
    // range ending before it starts has none
    let mut neg = featuregen::create_with(
        &generators(&gen(
            "CharacterNgram",
            "<int name=\"min\">-1</int><int name=\"max\">1</int>",
        )),
        &NoResources,
        Owner::NameFinder,
    )
    .unwrap();
    let e = neg
        .create_features(&mut f, &[u("ab")], 0, &[], 1)
        .unwrap_err();
    assert!(format!("{e:?}").contains("IndexOutOfBoundsException"));
    let mut empty = featuregen::create_with(
        &generators(&gen(
            "CharacterNgram",
            "<int name=\"min\">3</int><int name=\"max\">1</int>",
        )),
        &NoResources,
        Owner::NameFinder,
    )
    .unwrap();
    f.clear();
    empty
        .create_features(&mut f, &[u("abcd")], 0, &[], 1)
        .unwrap();
    assert!(f.is_empty());

    // a generator without a resource is fine where it needs none
    let mut plain = featuregen::create(&generators(&format!(
        "{}{}",
        gen("TrigramName", ""),
        gen("DocumentBegin", "")
    )))
    .unwrap();
    let toks = [u("a"), u("b"), u("c")];
    for s in 1..=2 {
        f.clear();
        plain.create_features(&mut f, &toks, 0, &[], s).unwrap();
        assert_eq!(f.contains(&u("D=begin")), s == 1, "{s}");
    }
}

#[test]
fn simple_tokenizer_splits_like_opennlp() {
    let t = |s: &str| -> Vec<String> {
        featuregen::simple_tokenize(&u(s))
            .iter()
            .map(|t| text::to_string(t))
            .collect()
    };
    assert_eq!(t("e-mail:x"), ["e", "-", "mail", ":", "x"]);
    assert_eq!(t("3.5kg"), ["3", ".", "5", "kg"]);
    assert_eq!(t("  a  "), ["a"]);
    assert_eq!(t("--!!"), ["--", "!!"]);
    assert_eq!(t("Ⓐb"), ["Ⓐb"]);
    assert!(t("   ").is_empty() && t("").is_empty());
}
