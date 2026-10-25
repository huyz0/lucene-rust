//! M14 T14.3's pairs (`benchmarks/micro/java/ClassificationMicro.java` is
//! the Java twin): each classifier's `assign_class` over the stored text of
//! the first 40 documents of `fixtures/data/classification/random`, and
//! `cls_perceptron_train`, building the perceptron.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use lucene_analysis::core_analysis::WhitespaceAnalyzer;
use lucene_analysis::Analyzer;
use lucene_search::classification::{
    Bm25NbClassifier, BooleanPerceptronClassifier, CachingNaiveBayesClassifier, Classifier,
    KNearestFuzzyClassifier, KNearestNeighborClassifier, SimpleNaiveBayesClassifier, TrainingIndex,
};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::SegmentNorms;
use lucene_store::FsDirectory;

use super::measure;

const INPUTS: i32 = 40;

fn run<T>(name: &str, w: Duration, m: Duration, c: &dyn Classifier<T>, inputs: &[String]) {
    measure(name, w, m, || {
        for input in inputs {
            black_box(c.assign_class(input).unwrap());
        }
        inputs.len() as u64
    });
}

pub fn bench_classification(w: Duration, m: Duration) {
    let dir = FsDirectory::open("fixtures/data/classification/random");
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let fields = ["text".to_string(), "cat".to_string(), "bool".to_string()];
    let owned = reader.field_norms_by_field(&fields).unwrap();
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let index = TrainingIndex::new(&reader, &segments, &norms);
    let a = Arc::new(Analyzer::new(WhitespaceAnalyzer::default()));
    let inputs: Vec<String> = (0..INPUTS)
        .map(|doc| {
            let (seg, local) = (&reader.segment_readers()[0], doc);
            let stored = seg.stored_document(local).unwrap().unwrap();
            let text = seg.field_infos().field_by_name("text").unwrap().number;
            stored
                .fields
                .iter()
                .find(|f| f.field_number == text)
                .and_then(|f| match &f.value {
                    lucene_codecs::stored_fields::FieldValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap()
        })
        .collect();
    let snb = SimpleNaiveBayesClassifier::new(index, Arc::clone(&a), None, "cat", &["text"]);
    run("cls_snb", w, m, &snb, &inputs);
    let cnb =
        CachingNaiveBayesClassifier::new(index, Arc::clone(&a), None, "cat", &["text"]).unwrap();
    run("cls_cnb", w, m, &cnb, &inputs);
    let bm25 = Bm25NbClassifier::new(index, Arc::clone(&a), None, "cat", &["text"]);
    run("cls_bm25nb", w, m, &bm25, &inputs);
    let knn = KNearestNeighborClassifier::new(
        index,
        None,
        Some(Arc::clone(&a)),
        None,
        3,
        0,
        0,
        "cat",
        &["text"],
    );
    run("cls_knn3", w, m, &knn, &inputs);
    let fuzzy =
        KNearestFuzzyClassifier::new(index, None, Arc::clone(&a), None, 3, "cat", &["text"]);
    run("cls_fuzzy3", w, m, &fuzzy, &inputs);
    let train = || {
        BooleanPerceptronClassifier::new(index, Arc::clone(&a), None, 1, None, "bool", "text")
            .unwrap()
    };
    let perceptron = train();
    run("cls_perceptron", w, m, &perceptron, &inputs);
    measure("cls_perceptron_train", w, m, || {
        black_box(train());
        1
    });
}
