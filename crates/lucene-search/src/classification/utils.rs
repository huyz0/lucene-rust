//! `org.apache.lucene.classification.utils`: evaluating a classifier
//! ([`ConfusionMatrixGenerator`]), splitting an index into training, test
//! and cross-validation sets ([`DatasetSplitter`]), and term vectors as
//! frequency vectors ([`DocToDoubleVectorUtils`]).

use std::sync::Arc;
use std::time::Instant;

use lucene_analysis::Analyzer;
use lucene_codecs::term_vectors::TermVectorField;
use lucene_index::document::{Document, Field, FieldType, TextField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::Directory;

use super::{java_string_cmp, ClassName, Classifier, TrainingIndex};
use crate::grouping::{GroupingSearch, Sort, TermGroupSelector};
use crate::mlt::java_hash_order;
use crate::query::{BooleanQuery, Clause, MatchAllDocsQuery, WildcardQuery};
use crate::reader::{IndexReader, Terms};
use crate::{Error, Result};

/// `ConfusionMatrixGenerator`.
pub struct ConfusionMatrixGenerator;

/// The longest one classification may take before Java gives up on it
/// (`Future.get(5, TimeUnit.SECONDS)`), in milliseconds.
const CLASSIFICATION_TIMEOUT_MS: f64 = 5000.0;

impl ConfusionMatrixGenerator {
    /// `getConfusionMatrix(reader, classifier, classFieldName,
    /// textFieldName, timeoutMilliseconds)`: every document with a class is
    /// classified by its stored text and counted under its true class (the
    /// assigned one when it is among the document's classes, else the first
    /// of them), until `timeout_ms` (when positive) has been spent.
    ///
    /// Java abandons a classification after five seconds and counts five
    /// seconds for it; this one cannot be interrupted, so it waits for the
    /// classification and then discards it the same way.
    ///
    /// # Errors
    /// What reading the index or the classifier reports.
    pub fn get_confusion_matrix<T: ClassName>(
        index: &TrainingIndex<'_, '_>,
        classifier: &dyn Classifier<T>,
        class_field: &str,
        text_field: &str,
        timeout_ms: i64,
    ) -> Result<ConfusionMatrix> {
        let q = BooleanQuery {
            must: vec![Clause::Wildcard(WildcardQuery::new(
                class_field,
                b"*".to_vec(),
            ))],
            ..Default::default()
        };
        let max_doc = usize::try_from(index.reader.max_doc()).unwrap_or(0);
        let top = index.searcher(None)?.search(&q, max_doc.max(1))?;
        let mut counts = ConfusionMatrix::default();
        let mut time = 0f64;
        let mut counter = 0i32;
        for sd in &top.score_docs {
            if timeout_ms > 0 && time >= timeout_ms as f64 {
                break;
            }
            let mut correct_answers = index.stored_values(sd.doc, class_field)?;
            if correct_answers.is_empty() {
                continue;
            }
            correct_answers.sort_by(|a, b| java_string_cmp(a, b));
            let Some(text) = index.stored_values(sd.doc, text_field)?.into_iter().next() else {
                continue;
            };
            let start = Instant::now();
            let result = classifier.assign_class(&text)?;
            let elapsed = start.elapsed().as_millis() as f64;
            if elapsed > CLASSIFICATION_TIMEOUT_MS {
                time += CLASSIFICATION_TIMEOUT_MS;
                continue;
            }
            time += elapsed;
            let Some(result) = result else {
                continue;
            };
            counter += 1;
            let classified = result.assigned_class.class_name();
            let correct = if correct_answers
                .binary_search_by(|a| java_string_cmp(a, &classified))
                .is_ok()
            {
                classified.clone()
            } else {
                correct_answers.swap_remove(0)
            };
            counts.add(correct, classified);
        }
        counts.avg_classification_time = time / f64::from(counter);
        counts.number_of_evaluated_docs = counter;
        Ok(counts)
    }
}

/// `ConfusionMatrixGenerator.ConfusionMatrix`: how often each true class was
/// given each class.
#[derive(Debug, Clone, Default)]
pub struct ConfusionMatrix {
    /// The rows, each with its counts, in insertion order (Java's
    /// `HashMap`s are iterated through [`java_hash_order`]).
    rows: Vec<(String, Vec<(String, i64)>)>,
    avg_classification_time: f64,
    number_of_evaluated_docs: i32,
}

/// A row's or a map's entries in `HashMap<String, _>` iteration order.
fn hash_ordered<V>(entries: &[(String, V)]) -> impl Iterator<Item = &(String, V)> {
    let keys: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
    java_hash_order(&keys).into_iter().map(move |i| &entries[i])
}

impl ConfusionMatrix {
    fn add(&mut self, correct: String, classified: String) {
        let row = match self.rows.iter().position(|(k, _)| *k == correct) {
            Some(i) => &mut self.rows[i].1,
            None => {
                self.rows.push((correct, Vec::new()));
                &mut self.rows.last_mut().expect("just pushed").1
            }
        };
        match row.iter_mut().find(|(k, _)| *k == classified) {
            Some((_, n)) => *n += 1,
            None => row.push((classified, 1)),
        }
    }

    fn row(&self, class: &str) -> Option<&[(String, i64)]> {
        self.rows
            .iter()
            .find(|(k, _)| k == class)
            .map(|(_, r)| r.as_slice())
    }

    fn count(row: &[(String, i64)], class: &str) -> Option<i64> {
        row.iter().find(|(k, _)| k == class).map(|&(_, n)| n)
    }

    /// `getLinearizedMatrix()`: each true class with the classes it was
    /// given and how often, in Java's `HashMap` iteration order.
    pub fn linearized_matrix(&self) -> Vec<(String, Vec<(String, i64)>)> {
        hash_ordered(&self.rows)
            .map(|(k, row)| (k.clone(), hash_ordered(row).cloned().collect()))
            .collect()
    }

    /// `getPrecision(klass)`: of the documents given `class`, the share that
    /// are `class`.
    pub fn precision_of(&self, class: &str) -> f64 {
        let mut tp = 0.0;
        let mut den = 0.0;
        if let Some(row) = self.row(class) {
            tp += Self::count(row, class).unwrap_or(0) as f64;
            for (_, values) in &self.rows {
                if let Some(n) = Self::count(values, class) {
                    den += n as f64;
                }
            }
        }
        if tp > 0.0 {
            tp / den
        } else {
            0.0
        }
    }

    /// `getRecall(klass)`: of the documents of `class`, the share given it.
    pub fn recall_of(&self, class: &str) -> f64 {
        let mut tp = 0.0;
        let mut fn_ = 0.0;
        if let Some(row) = self.row(class) {
            for (k, n) in row {
                if k == class {
                    tp += *n as f64;
                } else {
                    fn_ += *n as f64;
                }
            }
        }
        if tp + fn_ > 0.0 {
            tp / (tp + fn_)
        } else {
            0.0
        }
    }

    /// `getF1Measure(klass)`.
    pub fn f1_measure_of(&self, class: &str) -> f64 {
        f1(self.precision_of(class), self.recall_of(class))
    }

    /// `getF1Measure()`: from the macro averages.
    pub fn f1_measure(&self) -> f64 {
        f1(self.precision(), self.recall())
    }

    /// `getAccuracy()`: `(tp + tn) / (tp + fp + fn + tn)`, where a true
    /// negative is a (row, class) pair the row never gave.
    pub fn accuracy(&self) -> f64 {
        let (mut tp, mut tn, mut tfp, mut fn_) = (0.0, 0.0, 0.0, 0.0);
        for (class, row) in &self.rows {
            for (k, n) in row {
                if k == class {
                    tp += *n as f64;
                } else {
                    fn_ += *n as f64;
                }
            }
            for (_, values) in &self.rows {
                match Self::count(values, class) {
                    Some(n) => tfp += n as f64,
                    None => tn += 1.0,
                }
            }
        }
        (tp + tn) / (tfp + fn_ + tn)
    }

    /// `getPrecision()`: the macro average, summed in `HashMap` order.
    pub fn precision(&self) -> f64 {
        let mut p = 0.0;
        for (class, _) in hash_ordered(&self.rows) {
            p += self.precision_of(class);
        }
        p / self.rows.len() as f64
    }

    /// `getRecall()`: the macro average, summed in `HashMap` order.
    pub fn recall(&self) -> f64 {
        let mut r = 0.0;
        for (class, _) in hash_ordered(&self.rows) {
            r += self.recall_of(class);
        }
        r / self.rows.len() as f64
    }

    /// `getAvgClassificationTime()`, in milliseconds.
    pub fn avg_classification_time(&self) -> f64 {
        self.avg_classification_time
    }

    /// `getNumberOfEvaluatedDocs()`.
    pub fn number_of_evaluated_docs(&self) -> i32 {
        self.number_of_evaluated_docs
    }
}

fn f1(precision: f64, recall: f64) -> f64 {
    if precision > 0.0 && recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

/// `DocToDoubleVectorUtils`.
pub struct DocToDoubleVectorUtils;

impl DocToDoubleVectorUtils {
    /// `toSparseLocalFreqDoubleArray(docTerms, fieldTerms)`: for each term of
    /// the field, its frequency in the document (`0` where absent); `None`
    /// without a document vector or when the field's term count is unknown
    /// (`MultiTerms`).
    ///
    /// # Errors
    /// What reading the field's terms reports.
    pub fn to_sparse_local_freq_double_array(
        doc_terms: Option<&TermVectorField>,
        field_terms: &dyn Terms,
    ) -> Result<Option<Vec<f64>>> {
        let Some(doc_terms) = doc_terms else {
            return Ok(None);
        };
        if field_terms.size() == -1 {
            return Ok(None);
        }
        let mut out = Vec::with_capacity(usize::try_from(field_terms.size()).unwrap_or(0));
        let mut te = field_terms.iterator()?;
        while let Some(term) = te.next()? {
            let freq = doc_terms
                .terms
                .binary_search_by(|t| t.term.as_slice().cmp(term))
                .map_or(0.0, |i| f64::from(doc_terms.terms[i].freq));
            out.push(freq);
        }
        Ok(Some(out))
    }

    /// `toDenseLocalFreqDoubleArray(docTerms)`: the document's term
    /// frequencies, in term order.
    pub fn to_dense_local_freq_double_array(
        doc_terms: Option<&TermVectorField>,
    ) -> Option<Vec<f64>> {
        doc_terms.map(|d| d.terms.iter().map(|t| f64::from(t.freq)).collect())
    }
}

/// `DatasetSplitter`: an index's documents into training, test and
/// cross-validation indexes, class by class.
pub struct DatasetSplitter {
    test_ratio: f64,
    cross_validation_ratio: f64,
}

fn index_error(e: impl std::fmt::Display) -> Error {
    Error::IllegalArgument(e.to_string())
}

impl DatasetSplitter {
    /// `new DatasetSplitter(testRatio, crossValidationRatio)`.
    pub fn new(test_ratio: f64, cross_validation_ratio: f64) -> Self {
        Self {
            test_ratio,
            cross_validation_ratio,
        }
    }

    /// `split(originalIndex, trainingIndex, testIndex, crossValidationIndex,
    /// analyzer, termVectors, classFieldName, fieldNames...)`: the documents
    /// grouped by `class_field` (its sorted doc values), the groups and the
    /// documents within them in index order; of each group's documents,
    /// every other one goes to the test index until it holds
    /// `test_ratio` of the group, then the cross-validation index takes its
    /// `cross_validation_ratio`, then the training index the rest. Each copy
    /// holds `field_names`' first stored values (every stored field when
    /// none are named) as stored text fields, with term vectors when asked.
    ///
    /// Java asks for as many groups as the leaves have class values summed,
    /// so a group past that count (the one of documents without a class, in
    /// a single-leaf index) is left out, as here. Java also closes the
    /// original reader; this one is borrowed.
    ///
    /// # Errors
    /// What reading the original index or writing the new ones reports; a
    /// leaf without the class field's sorted doc values and without its
    /// terms (Java's `NullPointerException`).
    #[allow(clippy::too_many_arguments)]
    pub fn split(
        &self,
        original: &TrainingIndex<'_, '_>,
        training: &dyn Directory,
        test: &dyn Directory,
        cross_validation: &dyn Directory,
        analyzer: Arc<Analyzer>,
        term_vectors: bool,
        class_field: &str,
        field_names: &[&str],
    ) -> Result<()> {
        let version = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut test_writer = open_writer(test, version, &analyzer)?;
        let mut cv_writer = open_writer(cross_validation, version, &analyzer)?;
        let mut training_writer = open_writer(training, version, &analyzer)?;

        let mut no_of_classes = 0i64;
        for leaf in original.reader.leaves() {
            let mut value_count = 0i64;
            let sorted = leaf.reader.sorted_doc_values(class_field)?;
            match &sorted {
                Some(dv) => value_count = i64::from(dv.value_count()),
                None => {
                    if let Some(ss) = leaf.reader.sorted_set_doc_values(class_field)? {
                        value_count = ss.value_count();
                    }
                    let terms = leaf.reader.terms(class_field)?.ok_or_else(|| {
                        Error::IllegalState(format!("no terms for field {class_field}"))
                    })?;
                    no_of_classes += terms.size();
                }
            }
            no_of_classes += value_count;
        }

        let max_doc = original.reader.max_doc();
        let searcher = original.searcher(None)?;
        let grouping = GroupingSearch::new(|| TermGroupSelector::new(class_field))
            .set_group_sort(Sort::index_order())
            .set_sort_within_group(Sort::index_order())
            .set_all_groups(true)
            .set_group_docs_limit(usize::try_from(max_doc).unwrap_or(0));
        let all = BooleanQuery {
            must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(max_doc))],
            ..Default::default()
        };
        let groups = grouping
            .search(
                &searcher,
                &all,
                0,
                usize::try_from(no_of_classes).unwrap_or(0),
            )?
            .top_groups;

        let mut ft = FieldType::copy_of(&TextField::type_stored());
        if term_vectors {
            ft.set_store_term_vectors(true).map_err(index_error)?;
            ft.set_store_term_vector_offsets(true)
                .map_err(index_error)?;
            ft.set_store_term_vector_positions(true)
                .map_err(index_error)?;
        }
        let mut b = 0usize;
        for group in &groups.groups {
            let total = group.total_hits.value as f64;
            let test_size = total * self.test_ratio;
            let cv_size = total * self.cross_validation_ratio;
            let (mut tc, mut cvc) = (0usize, 0usize);
            for sd in &group.score_docs {
                let doc = new_doc(original, &ft, sd.doc, field_names)?;
                if b.is_multiple_of(2) && (tc as f64) < test_size {
                    test_writer.add_fields_document(&doc).map_err(index_error)?;
                    tc += 1;
                } else if (cvc as f64) < cv_size {
                    cv_writer.add_fields_document(&doc).map_err(index_error)?;
                    cvc += 1;
                } else {
                    training_writer
                        .add_fields_document(&doc)
                        .map_err(index_error)?;
                }
                b += 1;
            }
        }
        for w in [&mut test_writer, &mut cv_writer, &mut training_writer] {
            w.commit().map_err(index_error)?;
        }
        for w in [&mut test_writer, &mut cv_writer, &mut training_writer] {
            w.force_merge(3).map_err(index_error)?;
        }
        test_writer.close().map_err(index_error)?;
        cv_writer.close().map_err(index_error)?;
        training_writer.close().map_err(index_error)?;
        Ok(())
    }
}

fn open_writer<'d>(
    dir: &'d dyn Directory,
    version: LuceneVersion,
    analyzer: &Arc<Analyzer>,
) -> Result<IndexWriter<'d>> {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", version).map_err(index_error)?;
    w.set_analyzer(Some(Arc::clone(analyzer)));
    Ok(w)
}

/// `createNewDoc`: the named fields' first stored values, or every stored
/// field, as fields of type `ft`.
fn new_doc(
    index: &TrainingIndex<'_, '_>,
    ft: &FieldType,
    doc: i32,
    field_names: &[&str],
) -> Result<Document> {
    let mut out = Document::new();
    if field_names.is_empty() {
        let (seg, local) = crate::mlt::locate(index.reader, doc)?;
        let stored = seg.stored_document(local)?.unwrap_or_default();
        for f in &stored.fields {
            let Some(info) = seg.field_infos().field_by_number(f.field_number) else {
                continue;
            };
            let value = crate::mlt::stored_string_value(&f.value).ok_or_else(|| {
                Error::IllegalArgument(format!(
                    "field {} has a binary value a text field cannot index",
                    info.name
                ))
            })?;
            out.add(
                Field::from_string(info.name.as_str(), value, ft.clone()).map_err(index_error)?,
            );
        }
    } else {
        for name in field_names {
            if let Some(v) = index.stored_values(doc, name)?.into_iter().next() {
                out.add(Field::from_string(*name, v, ft.clone()).map_err(index_error)?);
            }
        }
    }
    Ok(out)
}
