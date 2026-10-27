//! M14 T14.3, differentially against Lucene 10.5.0:
//! `fixtures/src/GenClassification.java`'s `classification.tsv`.
//!
//! Every classifier of `lucene-classification` under seventeen
//! configurations over two indexes (`assignClass`, `getClasses` and the
//! confusion matrix over the index it learnt from), the two document
//! classifiers, `DocToDoubleVectorUtils` over each document's term vector and
//! what `DatasetSplitter` writes. The file is rebuilt here line for line and
//! compared with Lucene's; scores are compared as bits.

// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;
use std::sync::Arc;

use lucene_analysis::core_analysis::WhitespaceAnalyzer;
use lucene_analysis::Analyzer;
use lucene_codecs::stored_fields::FieldValue;
use lucene_codecs::term_vectors::TermVectorField;
use lucene_index::document::{Document, Field, FieldType, TextField};
use lucene_search::classification::document::{
    DocumentClassifier, KNearestNeighborDocumentClassifier, SimpleNaiveBayesDocumentClassifier,
};
use lucene_search::classification::utils::{
    ConfusionMatrixGenerator, DatasetSplitter, DocToDoubleVectorUtils,
};
use lucene_search::classification::{
    term_vector, Bm25NbClassifier, BooleanPerceptronClassifier, CachingNaiveBayesClassifier,
    ClassName, ClassificationResult, Classifier, KNearestFuzzyClassifier,
    KNearestNeighborClassifier, SimpleNaiveBayesClassifier, TrainingIndex,
};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::SegmentNorms;
use lucene_search::query::{Clause, TermQuery};
use lucene_search::reader::IndexReader;
use lucene_search::similarities::ClassicSimilarity;
use lucene_search::{Error, Result};
use lucene_store::{ByteBuffersDirectory, Directory, FsDirectory};

fn data() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/classification")
}

/// `"C1A55".hashCode()`: the generator's `tests.seed`, hence
/// `StringHelper.GOOD_FAST_HASH_SEED`.
fn hash_seed() -> i32 {
    "C1A55"
        .encode_utf16()
        .fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(i32::from(c)))
}

fn hex(d: f64) -> String {
    // A NaN's sign bit is the FPU's choice (x86 produces fff8..., arm64
    // 7ff8...), in Java as in Rust: print every NaN as the canonical one.
    if d.is_nan() {
        return format!("{:x}", f64::NAN.to_bits());
    }
    format!("{:x}", d.to_bits())
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
}

fn result<T: ClassName>(r: Option<&ClassificationResult<T>>) -> String {
    match r {
        None => "-".into(),
        Some(r) => format!("{}:{}", esc(&r.assigned_class.class_name()), hex(r.score)),
    }
}

fn results<T: ClassName>(rs: &Option<Vec<ClassificationResult<T>>>) -> String {
    match rs {
        None => "null".into(),
        Some(rs) => rs
            .iter()
            .map(|r| result(Some(r)))
            .collect::<Vec<_>>()
            .join(","),
    }
}

const SAMPLE_INPUTS: [&str; 6] = [
    "Here are some interesting questions and answers about Mitt Romney.. If you don't know the answer to the question about Mitt Romney, then simply click on the answer below the question section.",
    "Much is made of what the likes of Facebook, Google and Apple know about users. Truth is, Amazon may know more.",
    "Much is made of what the likes of Facebook, Google and Apple know about users. Truth is, Amazon may know more. This technology observation is extracted from the internet.",
    "More than 400 million people trust Google with their e-mail, and 50 million store files in the cloud using the Dropbox service. People manage their bank accounts, pay bills, trade stocks and generally transfer or store huge volumes of personal data online. traveling seeks raises some questions Republican presidential. ",
    "nothing here is in the index zzz",
    "",
];

/// `java.util.Random`, as far as the generator uses it.
struct JavaRandom(i64);

impl JavaRandom {
    fn new(seed: i64) -> Self {
        Self((seed ^ 0x5DEECE66D) & ((1 << 48) - 1))
    }

    fn next(&mut self, bits: u32) -> i32 {
        self.0 = (self.0.wrapping_mul(0x5DEECE66D).wrapping_add(0xB)) & ((1 << 48) - 1);
        (self.0 >> (48 - bits)) as i32
    }

    fn next_int(&mut self, bound: i32) -> i32 {
        if bound & -bound == bound {
            return ((i64::from(bound) * i64::from(self.next(31))) >> 31) as i32;
        }
        loop {
            let bits = self.next(31);
            let val = bits % bound;
            if bits - val + (bound - 1) >= 0 {
                return val;
            }
        }
    }
}

const CLASSES: i32 = 4;
const VOCABULARY: i32 = 40;

fn word(r: &mut JavaRandom, cls: i32) -> String {
    let w = if r.next_int(3) == 0 {
        r.next_int(VOCABULARY)
    } else {
        cls * (VOCABULARY / CLASSES) + r.next_int(VOCABULARY / CLASSES)
    };
    format!("{}{w}", if r.next_int(25) == 0 { "x" } else { "w" })
}

fn random_text(r: &mut JavaRandom, cls: i32, words: i32) -> String {
    (0..words)
        .map(|_| word(r, cls))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The generator's `randomInputs`: its `Random` replayed past the 240
/// documents it indexed.
fn random_inputs() -> Vec<String> {
    let mut r = JavaRandom::new(0xc1a55);
    for _ in 0..240 {
        let cls = r.next_int(CLASSES);
        let words = 3 + r.next_int(12);
        // `randomText(r, cls, 3 + r.nextInt(12))` draws the length first.
        random_text(&mut r, cls, words);
    }
    let mut out: Vec<String> = (0..24)
        .map(|i| {
            let words = 2 + r.next_int(10);
            random_text(&mut r, i % CLASSES, words)
        })
        .collect();
    out.push("w1 w1 w1 x39".into());
    out.push("unknown words only".into());
    out
}

const CLASSIFIERS: [&str; 17] = [
    "snb",
    "snb-q",
    "snb-bool",
    "cnb",
    "cnb-q",
    "bm25nb",
    "bm25nb-q",
    "knn1",
    "knn3",
    "knn3-classic",
    "knn5-df",
    "knn3-boost",
    "knn3-q",
    "fuzzy3",
    "fuzzy5-classic",
    "perceptron",
    "perceptron-b3",
];

enum AnyClassifier<'s, 'a> {
    Bytes(Box<dyn Classifier<Vec<u8>> + 's>),
    Bool(
        BooleanPerceptronClassifier,
        std::marker::PhantomData<&'a ()>,
    ),
}

fn classifier<'s, 'a>(
    name: &str,
    index: TrainingIndex<'s, 'a>,
    a: &Arc<Analyzer>,
) -> Result<AnyClassifier<'s, 'a>> {
    let q = || Some(Clause::Term(TermQuery::new("bool", b"true".to_vec())));
    let a = Arc::clone(a);
    let classic = || -> lucene_search::classification::SharedSimilarity {
        Some(Arc::new(ClassicSimilarity::default()))
    };
    let knn = |sim, k, df, tf, query, field: &str| {
        KNearestNeighborClassifier::new(
            index,
            sim,
            Some(Arc::clone(&a)),
            query,
            k,
            df,
            tf,
            "cat",
            &[field],
        )
    };
    let b: Box<dyn Classifier<Vec<u8>> + 's> = match name {
        "snb" => Box::new(SimpleNaiveBayesClassifier::new(
            index,
            Arc::clone(&a),
            None,
            "cat",
            &["text"],
        )),
        "snb-q" => Box::new(SimpleNaiveBayesClassifier::new(
            index,
            Arc::clone(&a),
            q(),
            "cat",
            &["text"],
        )),
        "snb-bool" => Box::new(SimpleNaiveBayesClassifier::new(
            index,
            Arc::clone(&a),
            None,
            "bool",
            &["text"],
        )),
        "cnb" => Box::new(CachingNaiveBayesClassifier::new(
            index,
            Arc::clone(&a),
            None,
            "cat",
            &["text"],
        )?),
        "cnb-q" => Box::new(CachingNaiveBayesClassifier::new(
            index,
            Arc::clone(&a),
            q(),
            "cat",
            &["text"],
        )?),
        "bm25nb" => Box::new(Bm25NbClassifier::new(
            index,
            Arc::clone(&a),
            None,
            "cat",
            &["text"],
        )),
        "bm25nb-q" => Box::new(Bm25NbClassifier::new(
            index,
            Arc::clone(&a),
            q(),
            "cat",
            &["text"],
        )),
        "knn1" => Box::new(knn(None, 1, 0, 0, None, "text")),
        "knn3" => Box::new(knn(None, 3, 0, 0, None, "text")),
        "knn3-classic" => Box::new(knn(classic(), 3, 0, 0, None, "text")),
        "knn5-df" => Box::new(knn(None, 5, 1, 1, None, "text")),
        "knn3-boost" => Box::new(knn(None, 3, 0, 0, None, "text^2.5")),
        "knn3-q" => Box::new(knn(None, 3, 0, 0, q(), "text")),
        "fuzzy3" => Box::new(KNearestFuzzyClassifier::new(
            index,
            None,
            Arc::clone(&a),
            None,
            3,
            "cat",
            &["text"],
        )),
        "fuzzy5-classic" => Box::new(KNearestFuzzyClassifier::new(
            index,
            classic(),
            Arc::clone(&a),
            None,
            5,
            "cat",
            &["text"],
        )),
        "perceptron" => {
            return Ok(AnyClassifier::Bool(
                BooleanPerceptronClassifier::new(index, a, None, 1, None, "bool", "text")?,
                std::marker::PhantomData,
            ))
        }
        "perceptron-b3" => {
            return Ok(AnyClassifier::Bool(
                BooleanPerceptronClassifier::new(index, a, None, 3, Some(5.0), "bool", "text")?,
                std::marker::PhantomData,
            ))
        }
        other => panic!("{other}"),
    };
    Ok(AnyClassifier::Bytes(b))
}

fn classify_rows<T: ClassName>(
    rows: &mut Vec<String>,
    index_name: &str,
    name: &str,
    index: &TrainingIndex<'_, '_>,
    c: &dyn Classifier<T>,
    inputs: &[String],
) {
    for (i, input) in inputs.iter().enumerate() {
        let a = c.assign_class(input).unwrap();
        rows.push(format!(
            "A\t{index_name}\t{name}\t{i}\t{}",
            result(a.as_ref())
        ));
        let l = c.get_classes(input).unwrap();
        rows.push(format!("L\t{index_name}\t{name}\t{i}\t{}", results(&l)));
    }
    let field = if name.starts_with("perceptron") || name == "snb-bool" {
        "bool"
    } else {
        "cat"
    };
    let m = ConfusionMatrixGenerator::get_confusion_matrix(index, c, field, "text", -1).unwrap();
    let mut sorted = m.linearized_matrix();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut mx = Vec::new();
    let mut per = Vec::new();
    for (k, mut row) in sorted {
        row.sort_by(|a, b| a.0.cmp(&b.0));
        for (v, n) in row {
            mx.push(format!("{k}>{v}={n}"));
        }
        per.push(format!(
            "{k}={}/{}/{}",
            hex(m.precision_of(&k)),
            hex(m.recall_of(&k)),
            hex(m.f1_measure_of(&k))
        ));
    }
    rows.push(format!(
        "M\t{index_name}\t{name}\t{field}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        m.number_of_evaluated_docs(),
        hex(m.accuracy()),
        hex(m.precision()),
        hex(m.recall()),
        hex(m.f1_measure()),
        mx.join(","),
        per.join(",")
    ));
}

fn text_type() -> FieldType {
    let mut ft = FieldType::copy_of(&TextField::type_stored());
    ft.set_store_term_vectors(true).unwrap();
    ft.set_store_term_vector_offsets(true).unwrap();
    ft.set_store_term_vector_positions(true).unwrap();
    ft
}

fn document_rows(
    rows: &mut Vec<String>,
    index_name: &str,
    index: TrainingIndex<'_, '_>,
    a: &Arc<Analyzer>,
    inputs: &[String],
) {
    let mut f2a = HashMap::new();
    f2a.insert("text".to_string(), Arc::clone(a));
    f2a.insert("title".to_string(), Arc::clone(a));
    let ft = text_type();
    for name in ["knn-doc", "knn-doc-boost", "snb-doc", "snb-doc-boost"] {
        let boost = name.ends_with("-boost");
        let c: Box<dyn DocumentClassifier<Vec<u8>> + '_> = if name.starts_with("knn") {
            let fields: &[&str] = if boost {
                &["text^2", "title"]
            } else {
                &["text"]
            };
            Box::new(KNearestNeighborDocumentClassifier::new(
                index,
                None,
                None,
                3,
                0,
                0,
                "cat",
                f2a.clone(),
                fields,
            ))
        } else {
            let fields: &[&str] = if boost { &["text^2"] } else { &["text"] };
            Box::new(SimpleNaiveBayesDocumentClassifier::new(
                index,
                None,
                "cat",
                f2a.clone(),
                fields,
            ))
        };
        for i in 0..inputs.len() - 1 {
            let mut d = Document::new();
            d.add(Field::from_string("text", inputs[i].as_str(), ft.clone()).unwrap());
            d.add(Field::from_string("text", inputs[i + 1].as_str(), ft.clone()).unwrap());
            d.add(
                Field::from_string("title", inputs[inputs.len() - 1 - i].as_str(), ft.clone())
                    .unwrap(),
            );
            let a = c.assign_class(&d).unwrap();
            let l = c.get_classes(&d).unwrap();
            rows.push(format!(
                "D\t{index_name}\t{name}\t{i}\t{} | {}",
                result(a.as_ref()),
                results(&Some(l))
            ));
        }
    }
}

fn vector(v: Option<Vec<f64>>) -> String {
    match v {
        None => "null".into(),
        Some(v) => v.iter().map(|d| hex(*d)).collect::<Vec<_>>().join(","),
    }
}

/// The generator's `SPLITS`: name, index, term vectors, class field, field
/// names.
const SPLITS: [(&str, &str, bool, &str, &[&str]); 4] = [
    ("sample", "sample", false, "cat", &["text", "cat"]),
    ("random", "random", false, "cat", &["text", "cat"]),
    ("random-tv", "random", true, "cat", &[]),
    ("random-bool", "random", false, "bool", &["text", "bool"]),
];

/// The generator's `vectorRow`: a `T` row.
fn vector_row(key: &str, field: &str, v: &TermVectorField) -> String {
    let flag = |on: bool, c: char| if on { c } else { '-' };
    let mut row = format!(
        "T\t{key}\t{field}\t{}{}{}",
        flag(v.has_positions, 'p'),
        flag(v.has_offsets, 'o'),
        flag(v.has_payloads, 'y')
    );
    for t in &v.terms {
        row.push_str(&format!(
            "\t{}:{}",
            esc(&String::from_utf8_lossy(&t.term)),
            t.freq
        ));
        for i in 0..t.freq as usize {
            row.push(':');
            if let Some(p) = &t.positions {
                row.push_str(&p[i].to_string());
            }
            if let (Some(s), Some(e)) = (&t.start_offsets, &t.end_offsets) {
                row.push_str(&format!("/{}-{}", s[i], e[i]));
            }
        }
    }
    row
}

fn split_rows(
    rows: &mut Vec<String>,
    index: &TrainingIndex<'_, '_>,
    a: Arc<Analyzer>,
    (name, _, term_vectors, class_field, field_names): (&str, &str, bool, &str, &[&str]),
) {
    let train = ByteBuffersDirectory::new();
    let test = ByteBuffersDirectory::new();
    let cv = ByteBuffersDirectory::new();
    let split = DatasetSplitter::new(0.2, 0.1).split(
        index,
        &train,
        &test,
        &cv,
        a,
        term_vectors,
        class_field,
        field_names,
    );
    if let Err(e) = split {
        // Java's `IllegalStateException` from `TermGroupSelector`.
        assert!(matches!(e, Error::IllegalState(_)), "{e:?}");
        rows.push(format!("S\t{name}\t!IllegalStateException"));
        return;
    }
    let dirs: [(&str, &dyn Directory); 3] = [("train", &train), ("test", &test), ("cv", &cv)];
    for (part, dir) in dirs {
        let r = DirectoryReader::open(dir).unwrap();
        let mut doc = 0;
        for seg in r.segment_readers() {
            for local in 0..seg.max_doc {
                let stored = seg.stored_document(local).unwrap().unwrap_or_default();
                let mut fields: Vec<&str> = seg
                    .field_infos()
                    .fields
                    .iter()
                    .map(|f| f.name.as_str())
                    .collect();
                fields.sort_unstable();
                let mut vectors = Vec::new();
                for field in fields {
                    if let Some(v) = term_vector(&r, doc, field).unwrap() {
                        vectors.push(vector_row(&format!("{name}\t{part}\t{doc}"), field, &v));
                    }
                }
                let tv = if vectors.is_empty() { "-" } else { "tv" };
                let mut line = format!("S\t{name}\t{part}\t{doc}\t{tv}");
                for f in &stored.fields {
                    let info = seg.field_infos().field_by_number(f.field_number).unwrap();
                    let FieldValue::String(v) = &f.value else {
                        panic!("{:?}", f.value)
                    };
                    line.push_str(&format!("\t{}={}", info.name, esc(v)));
                }
                rows.push(line);
                rows.extend(vectors);
                doc += 1;
            }
        }
    }
}

fn open(name: &str) -> DirectoryReader {
    DirectoryReader::open(&FsDirectory::open(data().join(name))).unwrap()
}

#[test]
fn classification_matches_lucene() {
    let fields = ["text".to_string(), "cat".to_string(), "bool".to_string()];
    let sample = open("sample");
    let random = open("random");
    let sample_opened = sample.open_segments().unwrap();
    let random_opened = random.open_segments().unwrap();
    let sample_segments = sample_opened.as_open_segments();
    let random_segments = random_opened.as_open_segments();
    let sample_owned = sample.field_norms_by_field(&fields).unwrap();
    let random_owned = random.field_norms_by_field(&fields).unwrap();
    let sample_norms: Vec<SegmentNorms<'_, '_>> = sample_owned.iter().map(Some).collect();
    let random_norms: Vec<SegmentNorms<'_, '_>> = random_owned.iter().map(Some).collect();
    let sample_index =
        TrainingIndex::new(&sample, &sample_segments, &sample_norms).with_hash_seed(hash_seed());
    let random_index =
        TrainingIndex::new(&random, &random_segments, &random_norms).with_hash_seed(hash_seed());
    let standard = Arc::new(Analyzer::standard(None));
    let whitespace = Arc::new(Analyzer::new(WhitespaceAnalyzer::default()));
    let sample_inputs: Vec<String> = SAMPLE_INPUTS.iter().map(|s| s.to_string()).collect();
    let random_inputs = random_inputs();

    let mut rows = Vec::new();
    for (index_name, index, a, inputs) in [
        ("sample", sample_index, &standard, &sample_inputs),
        ("random", random_index, &whitespace, &random_inputs),
    ] {
        for name in CLASSIFIERS {
            match classifier(name, index, a).unwrap() {
                AnyClassifier::Bytes(c) => {
                    classify_rows(&mut rows, index_name, name, &index, &*c, inputs)
                }
                AnyClassifier::Bool(c, _) => {
                    classify_rows(&mut rows, index_name, name, &index, &c, inputs)
                }
            }
        }
    }
    document_rows(&mut rows, "sample", sample_index, &standard, &sample_inputs);
    document_rows(
        &mut rows,
        "random",
        random_index,
        &whitespace,
        &random_inputs,
    );

    let leaves = random.leaves();
    let leaf = leaves[0].reader;
    let field_terms = leaf.terms("text").unwrap().unwrap();
    for doc in 0..random.max_doc() {
        let doc_terms = term_vector(&random, doc, "text").unwrap();
        rows.push(format!(
            "V\t{doc}\t{}\t{}",
            vector(DocToDoubleVectorUtils::to_dense_local_freq_double_array(
                doc_terms.as_ref()
            )),
            vector(
                DocToDoubleVectorUtils::to_sparse_local_freq_double_array(
                    doc_terms.as_ref(),
                    &*field_terms
                )
                .unwrap()
            )
        ));
    }

    for spec in SPLITS {
        let (index, a) = if spec.1 == "sample" {
            (&sample_index, &standard)
        } else {
            (&random_index, &whitespace)
        };
        split_rows(&mut rows, index, Arc::clone(a), spec);
    }

    // `fuzzy/`: a token with more variants than the variant queue holds.
    let fuzzy = open("fuzzy");
    let fuzzy_opened = fuzzy.open_segments().unwrap();
    let fuzzy_segments = fuzzy_opened.as_open_segments();
    let fuzzy_owned = fuzzy.field_norms_by_field(&fields).unwrap();
    let fuzzy_norms: Vec<SegmentNorms<'_, '_>> = fuzzy_owned.iter().map(Some).collect();
    let fuzzy_index =
        TrainingIndex::new(&fuzzy, &fuzzy_segments, &fuzzy_norms).with_hash_seed(hash_seed());
    let fuzzy_inputs = ["abcd", "abce abdd", "abzzd", "abcdz abcd"];
    for name in ["fuzzy3", "fuzzy5-classic"] {
        let AnyClassifier::Bytes(c) = classifier(name, fuzzy_index, &whitespace).unwrap() else {
            unreachable!()
        };
        for (i, input) in fuzzy_inputs.iter().enumerate() {
            let a = c.assign_class(input).unwrap();
            rows.push(format!("A\tfuzzy\t{name}\t{i}\t{}", result(a.as_ref())));
            let l = c.get_classes(input).unwrap();
            rows.push(format!("L\tfuzzy\t{name}\t{i}\t{}", results(&l)));
        }
    }

    let want = std::fs::read_to_string(data().join("classification.tsv")).unwrap();
    // The fixture was written on x86, whose NaNs carry the sign bit; see `hex`.
    let want = want.replace(
        &format!("{:x}", (-f64::NAN).to_bits()),
        &format!("{:x}", f64::NAN.to_bits()),
    );
    let want: Vec<&str> = want.lines().collect();
    let mut bad = 0;
    for (i, (w, g)) in want.iter().zip(&rows).enumerate() {
        if *w != g.as_str() {
            bad += 1;
            if bad <= 20 {
                eprintln!("line {}:\n  want {w}\n  got  {g}", i + 1);
            }
        }
    }
    assert_eq!(want.len(), rows.len(), "line counts");
    assert_eq!(bad, 0, "{bad} of {} lines differ", want.len());
}
