//! M14 T14.3: the classification module's edges the Lucene fixture
//! (`classification_fixtures.rs`) does not reach -- missing fields, empty
//! classes, Java's exceptions (`subList`, `/ by zero`, a missing term
//! vector) and the confusion matrix over documents it skips -- over small
//! indexes written here.

// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;
use std::sync::Arc;

use lucene_analysis::core_analysis::WhitespaceAnalyzer;
use lucene_analysis::Analyzer;
use lucene_index::document::{
    Document, Field, FieldType, SortedDocValuesField, SortedSetDocValuesField, Store, StringField,
    TextField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::classification::document::{
    DocumentClassifier, KNearestNeighborDocumentClassifier, SimpleNaiveBayesDocumentClassifier,
};
use lucene_search::classification::utils::{
    ConfusionMatrixGenerator, DatasetSplitter, DocToDoubleVectorUtils,
};
use lucene_search::classification::{
    term_vector, Bm25NbClassifier, BooleanPerceptronClassifier, CachingNaiveBayesClassifier,
    Classifier, KNearestNeighborClassifier, SimpleNaiveBayesClassifier, TrainingIndex,
};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::SegmentNorms;
use lucene_search::multi_terms::MultiTerms;
use lucene_search::query::{Clause, TermQuery};
use lucene_search::Error;
use lucene_store::{ByteBuffersDirectory, Directory};

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn analyzer() -> Arc<Analyzer> {
    Arc::new(Analyzer::new(WhitespaceAnalyzer::default()))
}

/// A stored text field, with term vectors (positions and offsets) or not.
fn text_type(vectors: bool) -> FieldType {
    let mut ft = FieldType::copy_of(&TextField::type_stored());
    if vectors {
        ft.set_store_term_vectors(true).unwrap();
        ft.set_store_term_vector_positions(true).unwrap();
        ft.set_store_term_vector_offsets(true).unwrap();
    }
    ft
}

/// A document of `(field, value)`s: `text` as [`text_type`], `cat` as a
/// stored string with sorted doc values, `cat-hidden` as an unstored `cat`,
/// anything else as a stored string.
fn doc(fields: &[(&str, &str)], vectors: bool) -> Document {
    let mut d = Document::new();
    for &(name, value) in fields {
        match name {
            "text" => d.add(Field::from_string("text", value, text_type(vectors)).unwrap()),
            "cat" => {
                d.add(StringField::new("cat", value, Store::Yes));
                d.add(SortedDocValuesField::new("cat", value.as_bytes()));
            }
            "cat-hidden" => {
                d.add(StringField::new("cat", value, Store::No));
                d.add(SortedDocValuesField::new("cat", value.as_bytes()));
            }
            _ => d.add(StringField::new(name, value, Store::Yes)),
        }
    }
    d
}

/// `docs` written to a fresh directory, one commit per segment.
fn write(segments: &[Vec<Document>]) -> ByteBuffersDirectory {
    let dir = ByteBuffersDirectory::new();
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_analyzer(Some(analyzer()));
    for docs in segments {
        for d in docs {
            w.add_fields_document(d).unwrap();
        }
        w.commit().unwrap();
    }
    w.close().unwrap();
    dir
}

/// `f` over the index in `dir`, its text norms loaded.
fn with_index<R>(dir: &dyn Directory, f: impl FnOnce(TrainingIndex<'_, '_>) -> R) -> R {
    let reader = DirectoryReader::open(dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let fields = ["text".to_string(), "cat".to_string(), "bool".to_string()];
    let owned = reader.field_norms_by_field(&fields).unwrap();
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    f(TrainingIndex::new(&reader, &segments, &norms))
}

/// The common index: two classes, a document without text, one without a
/// class, one whose class is not stored.
fn common(vectors: bool) -> ByteBuffersDirectory {
    write(&[vec![
        doc(&[("text", "a b"), ("cat", "x"), ("bool", "false")], vectors),
        doc(&[("text", "a c"), ("cat", "y"), ("bool", "false")], vectors),
        doc(&[("cat", "x"), ("bool", "true")], vectors),
        doc(&[("text", "b")], vectors),
        doc(&[("text", "d"), ("cat-hidden", "y")], vectors),
    ]])
}

#[test]
fn naive_bayes_without_classes_or_with_empty_ones() {
    let dir = common(true);
    with_index(&dir, |index| {
        let snb = SimpleNaiveBayesClassifier::new(index, analyzer(), None, "cat", &["text"]);
        let all = snb.get_classes("a").unwrap().unwrap();
        assert_eq!(all.len(), 2);
        // `getClasses(text, max)`: `subList(0, max)`.
        assert_eq!(snb.get_classes_max("a", 1).unwrap().unwrap(), all[..1]);
        assert!(matches!(
            snb.get_classes_max("a", 3),
            Err(Error::IllegalArgument(m)) if m == "toIndex = 3"
        ));

        // No class field: no classes, Java's `null` best class.
        let none = SimpleNaiveBayesClassifier::new(index, analyzer(), None, "nope", &["text"]);
        assert_eq!(none.assign_class("a").unwrap(), None);
        assert_eq!(none.get_classes("a").unwrap(), Some(Vec::new()));
        // A text field without terms: Java's `NullPointerException`.
        let no_text = SimpleNaiveBayesClassifier::new(index, analyzer(), None, "cat", &["nope"]);
        assert!(matches!(
            no_text.assign_class("a"),
            Err(Error::IllegalState(_))
        ));

        assert!(matches!(
            CachingNaiveBayesClassifier::new(index, analyzer(), None, "nope", &["text"]),
            Err(Error::IllegalState(_))
        ));
        assert!(matches!(
            CachingNaiveBayesClassifier::new(index, analyzer(), None, "cat", &["nope"]),
            Err(Error::IllegalState(_))
        ));
        let mut cnb =
            CachingNaiveBayesClassifier::new(index, analyzer(), None, "cat", &["text"]).unwrap();
        // A word outside the cache is never searched...
        let cached_only = cnb.get_classes("zz").unwrap().unwrap();
        // ...unless the cache is rebuilt without `justCachedTerms`, when it
        // is (and finds nothing either), and a word under the minimum is
        // searched every time.
        cnb.re_init_cache(5, false).unwrap();
        assert_eq!(cnb.get_classes("zz").unwrap().unwrap(), cached_only);
        let a = cnb.assign_class("a b").unwrap().unwrap();
        assert_eq!(cnb.assign_class("a b").unwrap().unwrap(), a);

        assert!(matches!(
            Bm25NbClassifier::new(index, analyzer(), None, "nope", &["text"]).assign_class("a"),
            Err(Error::IllegalState(_))
        ));
    });

    // Only the empty class: skipped, so nothing is left to assign.
    let empty = write(&[vec![doc(&[("text", "a"), ("cat", "")], false)]]);
    with_index(&empty, |index| {
        let snb = SimpleNaiveBayesClassifier::new(index, analyzer(), None, "cat", &["text"]);
        assert_eq!(snb.assign_class("a").unwrap(), None);
        let bm25 = Bm25NbClassifier::new(index, analyzer(), None, "cat", &["text"]);
        assert_eq!(bm25.get_classes("a").unwrap(), Some(Vec::new()));
        assert!(matches!(
            bm25.assign_class("a"),
            Err(Error::IllegalArgument(m)) if m == "Index 0 out of bounds for length 0"
        ));
    });
}

#[test]
fn perceptron_errors_and_its_null_class_list() {
    let dir = common(true);
    with_index(&dir, |index| {
        let a = analyzer();
        assert!(matches!(
            BooleanPerceptronClassifier::new(index, a.clone(), None, 1, None, "bool", "nope"),
            Err(Error::IllegalState(m)) if m.contains("term vectors need to be available")
        ));
        // "a b" scores 3 against a bias of 5/3, wrongly: the update divides
        // by the batch size.
        assert!(matches!(
            BooleanPerceptronClassifier::new(index, a.clone(), None, 0, None, "bool", "text"),
            Err(Error::IllegalArgument(m)) if m == "/ by zero"
        ));
        let q = Some(Clause::Term(TermQuery::new("cat", b"x".to_vec())));
        let c =
            BooleanPerceptronClassifier::new(index, a.clone(), q, 1, None, "bool", "text").unwrap();
        assert!(c.assign_class("a b").unwrap().is_some());
        // `getClasses` is Java's `null`, with or without a maximum.
        assert_eq!(c.get_classes("a b").unwrap(), None);
        assert_eq!(c.get_classes_max("a b", 1).unwrap(), None);
    });

    // The same documents without term vectors cannot update the weights.
    let plain = common(false);
    with_index(&plain, |index| {
        assert!(matches!(
            BooleanPerceptronClassifier::new(index, analyzer(), None, 1, None, "bool", "text"),
            Err(Error::IllegalState(m)) if m == "term vectors must be stored for field text"
        ));
    });
}

#[test]
fn document_classifiers_edges() {
    let dir = common(true);
    with_index(&dir, |index| {
        let mut f2a = HashMap::new();
        f2a.insert("text".to_string(), analyzer());
        let mut d = Document::new();
        d.add(Field::from_string("text", "a b", text_type(true)).unwrap());

        let knn = KNearestNeighborDocumentClassifier::new(
            index,
            None,
            None,
            3,
            1,
            1,
            "cat",
            f2a.clone(),
            &["text"],
        );
        let all = knn.get_classes(&d).unwrap();
        assert!(!all.is_empty());
        // `getClasses(document, max)` truncates, past the end too.
        assert_eq!(knn.get_classes_max(&d, 1).unwrap(), all[..1]);
        assert_eq!(knn.get_classes_max(&d, 99).unwrap(), all);

        let snb =
            SimpleNaiveBayesDocumentClassifier::new(index, None, "cat", f2a.clone(), &["text"]);
        let all = snb.get_classes(&d).unwrap();
        assert_eq!(snb.get_classes_max(&d, 1).unwrap(), all[..1]);
        assert!(matches!(
            snb.get_classes_max(&d, 3),
            Err(Error::IllegalArgument(_))
        ));
        // No class field: nothing, without analysing the document.
        let none =
            SimpleNaiveBayesDocumentClassifier::new(index, None, "nope", f2a.clone(), &["text"]);
        assert_eq!(none.assign_class(&d).unwrap(), None);
        // A field without an analyzer: Java's `NullPointerException`.
        let mut t = Document::new();
        t.add(Field::from_string("title", "a", text_type(false)).unwrap());
        let untitled =
            SimpleNaiveBayesDocumentClassifier::new(index, None, "cat", f2a.clone(), &["title"]);
        assert!(matches!(
            untitled.assign_class(&t),
            Err(Error::IllegalArgument(m)) if m == "no analyzer for field title"
        ));
        // A boost that is not a float: Java's `NumberFormatException`.
        let bad = SimpleNaiveBayesDocumentClassifier::new(index, None, "cat", f2a, &["text^x"]);
        assert!(bad.assign_class(&d).is_err());
    });
}

#[test]
fn confusion_matrix_skips_what_it_cannot_evaluate() {
    let dir = common(true);
    with_index(&dir, |index| {
        let knn = KNearestNeighborClassifier::new(
            index,
            None,
            Some(analyzer()),
            None,
            1,
            1,
            1,
            "cat",
            &["text"],
        );
        // Of the four classed documents, one has no text and one no stored
        // class: two are evaluated.
        let m = ConfusionMatrixGenerator::get_confusion_matrix(&index, &knn, "cat", "text", -1)
            .unwrap();
        assert_eq!(m.number_of_evaluated_docs(), 2);
        assert!(m.avg_classification_time() >= 0.0);
        assert_eq!(m.precision_of("nope"), 0.0);
        assert_eq!(m.recall_of("nope"), 0.0);
        assert_eq!(m.f1_measure_of("nope"), 0.0);

        // A classifier that never answers evaluates nothing; neither does a
        // text field nobody stored.
        let none = SimpleNaiveBayesClassifier::new(index, analyzer(), None, "nope", &["text"]);
        let m = ConfusionMatrixGenerator::get_confusion_matrix(&index, &none, "cat", "text", 0)
            .unwrap();
        assert_eq!(m.number_of_evaluated_docs(), 0);
        assert!(m.avg_classification_time().is_nan());
        let m =
            ConfusionMatrixGenerator::get_confusion_matrix(&index, &knn, "cat", "nope", 0).unwrap();
        assert_eq!(m.number_of_evaluated_docs(), 0);
        assert!(m.linearized_matrix().is_empty());
    });
}

#[test]
fn vectors_of_missing_documents_and_unsized_terms() {
    assert_eq!(
        DocToDoubleVectorUtils::to_dense_local_freq_double_array(None),
        None
    );
    // Two segments: `MultiTerms` does not know its size.
    let dir = write(&[
        vec![doc(&[("text", "a b"), ("cat", "x")], true)],
        vec![doc(&[("text", "b c"), ("cat", "y")], true)],
    ]);
    let reader = DirectoryReader::open(&dir).unwrap();
    let terms = MultiTerms::get_terms(&reader, "text").unwrap().unwrap();
    let v = term_vector(&reader, 0, "text").unwrap();
    assert!(v.is_some());
    assert_eq!(
        DocToDoubleVectorUtils::to_sparse_local_freq_double_array(v.as_ref(), &*terms).unwrap(),
        None
    );
    assert_eq!(
        DocToDoubleVectorUtils::to_sparse_local_freq_double_array(None, &*terms).unwrap(),
        None
    );
}

#[test]
fn dataset_splitter_refuses_what_java_cannot_split() {
    let split = |dir: &dyn Directory, class: &str, names: &[&str]| {
        with_index(dir, |index| {
            let (train, test, cv) = (
                ByteBuffersDirectory::new(),
                ByteBuffersDirectory::new(),
                ByteBuffersDirectory::new(),
            );
            DatasetSplitter::new(0.2, 0.1).split(
                &index,
                &train,
                &test,
                &cv,
                analyzer(),
                false,
                class,
                names,
            )
        })
    };
    let dir = common(false);
    // A class field with neither sorted doc values nor terms.
    assert!(matches!(
        split(&dir, "nope", &["text"]),
        Err(Error::IllegalState(_))
    ));

    // A class field of sorted-set doc values, which grouping refuses.
    let mut d = Document::new();
    d.add(StringField::new("tag", "x", Store::Yes));
    d.add(SortedSetDocValuesField::new("tag", b"x".to_vec()));
    d.add(Field::from_string("text", "a", text_type(false)).unwrap());
    let set = write(&[vec![d]]);
    assert!(split(&set, "tag", &["text"]).is_err());

    // Every stored field copied, one of them binary: a text field cannot
    // hold it.
    let mut d = Document::new();
    d.add(StringField::from_bytes("bin", b"\x01".to_vec(), Store::Yes));
    d.add(StringField::new("cat", "x", Store::Yes));
    d.add(SortedDocValuesField::new("cat", b"x".to_vec()));
    let binary = write(&[vec![d]]);
    assert!(matches!(
        split(&binary, "cat", &[]),
        Err(Error::IllegalArgument(m)) if m.contains("binary value")
    ));
}
