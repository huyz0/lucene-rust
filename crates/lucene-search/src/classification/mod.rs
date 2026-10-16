//! `org.apache.lucene.classification`: classifiers that learn from an
//! index's documents -- naive Bayes ([`SimpleNaiveBayesClassifier`],
//! [`CachingNaiveBayesClassifier`], [`Bm25NbClassifier`]), k nearest
//! neighbours by `MoreLikeThis` or by fuzzy terms
//! ([`KNearestNeighborClassifier`], [`KNearestFuzzyClassifier`]), a boolean
//! perceptron ([`BooleanPerceptronClassifier`]) -- their document-input
//! flavours ([`document`]) and the evaluation tools ([`utils`]).
//!
//! The classifiers ask the index what Java's ask it, through the same
//! queries, so their scores are Java's bit for bit (`Math.log`/`Math.exp`
//! are `StrictMath`'s, as the fixtures are generated). Two things Java leaves
//! to the JVM are explicit here:
//!
//! - **Hash order.** Java collects some classes in a `HashMap<BytesRef, _>`
//!   and reports them in its iteration order, which follows
//!   `BytesRef.hashCode` -- murmur3 seeded by `StringHelper
//!   .GOOD_FAST_HASH_SEED`, the clock unless `tests.seed` is set. Equal
//!   scores come out in that order, and `NearestFuzzyQuery` sums its clauses
//!   in it. [`TrainingIndex::hash_seed`] is that seed.
//! - **The index.** Java's classifiers take an `IndexReader` and build their
//!   own `IndexSearcher`s; here a [`TrainingIndex`] carries the reader with
//!   its opened segments and norms, and each classifier builds its searchers
//!   from them with its own similarity.

mod knn;
mod naive_bayes;
mod nearest_fuzzy;
mod perceptron;

pub mod document;
pub mod utils;

pub use knn::{KNearestFuzzyClassifier, KNearestNeighborClassifier};
pub use naive_bayes::{Bm25NbClassifier, CachingNaiveBayesClassifier, SimpleNaiveBayesClassifier};
pub use nearest_fuzzy::NearestFuzzyQuery;
pub use perceptron::BooleanPerceptronClassifier;

use std::cmp::Ordering;
use std::sync::Arc;

use lucene_analysis::{Analyzer, TokenStream};

use crate::directory_reader::DirectoryReader;
use crate::function::java_double_compare;
use crate::index_searcher::{IndexSearcher, SegmentNorms};
use crate::multi_segment::OpenSegment;
use crate::similarities::Similarity;
use crate::{Error, Result};

/// `ClassificationResult<T>`: a class and its score.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassificationResult<T> {
    pub assigned_class: T,
    pub score: f64,
}

impl<T> ClassificationResult<T> {
    pub fn new(assigned_class: T, score: f64) -> Self {
        Self {
            assigned_class,
            score,
        }
    }

    /// `compareTo`: by score, highest first (`Double.compare(o.score,
    /// this.score)`).
    pub fn compare(&self, other: &Self) -> Ordering {
        java_double_compare(other.score, self.score)
    }
}

/// `Classifier<T>`.
pub trait Classifier<T> {
    /// `assignClass(text)`: the best class, `None` for Java's `null`.
    ///
    /// # Errors
    /// What reading the index or analysing `text` reports.
    fn assign_class(&self, text: &str) -> Result<Option<ClassificationResult<T>>>;

    /// `getClasses(text)`: every class, best first; `None` where Java returns
    /// `null` (the perceptron).
    ///
    /// # Errors
    /// As [`Self::assign_class`].
    fn get_classes(&self, text: &str) -> Result<Option<Vec<ClassificationResult<T>>>>;

    /// `getClasses(text, max)`: the first `max` of [`Self::get_classes`].
    ///
    /// # Errors
    /// As [`Self::assign_class`]; `max` past the classes there are (Java's
    /// `subList` throws).
    fn get_classes_max(
        &self,
        text: &str,
        max: usize,
    ) -> Result<Option<Vec<ClassificationResult<T>>>> {
        self.get_classes(text)?
            .map(|c| sub_list(c, max))
            .transpose()
    }
}

/// A class as `ConfusionMatrixGenerator` names it: `BytesRef.utf8ToString()`
/// or `toString()`.
pub trait ClassName {
    fn class_name(&self) -> String;
}

impl ClassName for Vec<u8> {
    fn class_name(&self) -> String {
        String::from_utf8_lossy(self).into_owned()
    }
}

impl ClassName for bool {
    fn class_name(&self) -> String {
        self.to_string()
    }
}

/// The index a classifier learns from: the reader, its opened segments and
/// the norms of every field a classifier scores (`IndexSearcher` only reads
/// the norms it is given), and the JVM's `BytesRef` hash seed.
#[derive(Clone, Copy)]
pub struct TrainingIndex<'s, 'a> {
    pub reader: &'a DirectoryReader,
    pub segments: &'s [OpenSegment<'a>],
    pub norms: &'s [SegmentNorms<'s, 'a>],
    /// `StringHelper.GOOD_FAST_HASH_SEED`: `tests.seed`'s `hashCode()` for a
    /// JVM run with one, else the clock's millis.
    pub hash_seed: i32,
}

impl<'s, 'a> TrainingIndex<'s, 'a> {
    /// The index, with hash seed `0`.
    pub fn new(
        reader: &'a DirectoryReader,
        segments: &'s [OpenSegment<'a>],
        norms: &'s [SegmentNorms<'s, 'a>],
    ) -> Self {
        Self {
            reader,
            segments,
            norms,
            hash_seed: 0,
        }
    }

    /// This index under another `GOOD_FAST_HASH_SEED`.
    pub fn with_hash_seed(mut self, hash_seed: i32) -> Self {
        self.hash_seed = hash_seed;
        self
    }

    /// `new IndexSearcher(reader)` with `setSimilarity(similarity)` (`None`
    /// is the default BM25).
    pub(crate) fn searcher<'x>(
        &'x self,
        similarity: Option<&'x dyn Similarity>,
    ) -> Result<IndexSearcher<'x, 'a>> {
        let segments: &'x [OpenSegment<'a>] = self.segments;
        let norms: &'x [SegmentNorms<'x, 'a>] = self.norms;
        let mut searcher = IndexSearcher::new(segments, norms)?;
        if let Some(s) = similarity {
            searcher.set_similarity(s);
        }
        Ok(searcher)
    }

    /// `BytesRef.hashCode()` under this index's seed.
    pub(crate) fn bytes_hash(&self, bytes: &[u8]) -> i32 {
        lucene_util::string_helper::murmurhash3_x86_32(bytes, self.hash_seed)
    }

    /// The stored values of `field` in global document `doc`, as
    /// `Document.getValues(field)` reads them.
    pub(crate) fn stored_values(&self, doc: i32, field: &str) -> Result<Vec<String>> {
        stored_values(self.reader, doc, field)
    }

    /// [`Self::stored_values`] for each of `fields`, one decoding of `doc`.
    pub(crate) fn stored_values_of(&self, doc: i32, fields: &[&str]) -> Result<Vec<Vec<String>>> {
        stored_values_of(self.reader, doc, fields)
    }
}

/// `Document.getValues(field)` of global document `doc`'s stored fields.
pub(crate) fn stored_values(
    reader: &DirectoryReader,
    doc: i32,
    field: &str,
) -> Result<Vec<String>> {
    Ok(stored_values_of(reader, doc, &[field])?.remove(0))
}

/// [`stored_values`] for each of `fields`, from one decoding of the document.
pub(crate) fn stored_values_of(
    reader: &DirectoryReader,
    doc: i32,
    fields: &[&str],
) -> Result<Vec<Vec<String>>> {
    let (seg, local) = crate::mlt::locate(reader, doc)?;
    let numbers: Vec<Option<i32>> = fields
        .iter()
        .map(|f| seg.field_infos().field_by_name(f).map(|i| i.number))
        .collect();
    let mut out = vec![Vec::new(); fields.len()];
    if numbers.iter().all(Option::is_none) {
        return Ok(out);
    }
    let document = seg.stored_document(local)?.unwrap_or_default();
    for f in &document.fields {
        for (slot, number) in out.iter_mut().zip(&numbers) {
            if *number == Some(f.field_number) {
                if let Some(v) = crate::mlt::stored_string_value(&f.value) {
                    slot.push(v);
                }
            }
        }
    }
    Ok(out)
}

/// An analyzer error, as this crate reports it.
pub(crate) fn analysis(e: lucene_analysis::AnalysisError) -> Error {
    Error::IllegalArgument(e.to_string())
}

/// The terms `analyzer` makes of `text` as `field`
/// (`CharTermAttribute.toString()` per token).
pub(crate) fn analyze(analyzer: &Analyzer, field: &str, text: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for_each_token(analyzer, field, text, |t| out.push(t.to_string()))?;
    Ok(out)
}

/// `tokenize(text)`: `text` analysed as each of `fields` in turn, the tokens
/// concatenated.
pub(crate) fn tokenize(analyzer: &Analyzer, fields: &[String], text: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for f in fields {
        out.extend(analyze(analyzer, f, text)?);
    }
    Ok(out)
}

/// `Collections.sort(results)`: stable, best score first.
pub(crate) fn sort_results<T>(results: &mut [ClassificationResult<T>]) {
    results.sort_by(|a, b| a.compare(b));
}

/// The first result whose score beats every earlier one (`maxscore` starting
/// at `-Double.MAX_VALUE`, compared with `>`): `assignClass`'s loop.
pub(crate) fn best<T: Clone>(
    results: &[ClassificationResult<T>],
) -> Option<ClassificationResult<T>> {
    let mut best: Option<&ClassificationResult<T>> = None;
    let mut max = -f64::MAX;
    for r in results {
        if r.score > max {
            best = Some(r);
            max = r.score;
        }
    }
    best.cloned()
}

/// `list.subList(0, max)`.
pub(crate) fn sub_list<T>(mut list: Vec<T>, max: usize) -> Result<Vec<T>> {
    if max > list.len() {
        return Err(Error::IllegalArgument(format!("toIndex = {max}")));
    }
    list.truncate(max);
    Ok(list)
}

/// `normClassificationResults`: log scores to probabilities summing to one,
/// `exp(x - log(sum(exp(x_n))))`, computed from the best score down to keep
/// the exponentials in range. Sorts `assigned` first, as Java does.
pub(crate) fn norm_classification_results(
    mut assigned: Vec<ClassificationResult<Vec<u8>>>,
) -> Vec<ClassificationResult<Vec<u8>>> {
    use lucene_util::strict_math::{exp, log};
    if assigned.is_empty() {
        return assigned;
    }
    sort_results(&mut assigned);
    let smax = assigned[0].score;
    let mut sum_log = 0.0;
    for cr in &assigned {
        sum_log += exp(cr.score - smax);
    }
    let loga = smax + log(sum_log);
    assigned
        .into_iter()
        .map(|cr| ClassificationResult::new(cr.assigned_class, exp(cr.score - loga)))
        .collect()
}

/// A text field name with an optional `^boost` suffix (`"title^10"`), as
/// `fieldName.split("\\^")` reads it: the name and, when present, the boost's
/// text.
pub(crate) fn split_boost(name: &str) -> (&str, Option<&str>) {
    match name.split_once('^') {
        Some((field, boost)) => (field, Some(boost.split('^').next().unwrap_or(boost))),
        None => (name, None),
    }
}

/// `String.compareTo`: by UTF-16 code unit.
pub(crate) fn java_string_cmp(a: &str, b: &str) -> Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

/// A `String` ordered as Java orders it ([`java_string_cmp`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct JavaString(pub(crate) String);

impl PartialOrd for JavaString {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for JavaString {
    fn cmp(&self, other: &Self) -> Ordering {
        java_string_cmp(&self.0, &other.0)
    }
}

/// `termVectors().get(doc, field)` of global document `doc`.
pub fn term_vector(
    reader: &DirectoryReader,
    doc: i32,
    field: &str,
) -> Result<Option<lucene_codecs::term_vectors::TermVectorField>> {
    let (seg, local) = crate::mlt::locate(reader, doc)?;
    match seg.term_vectors_reader()? {
        Some(tv) => {
            crate::term_vectors_query::term_vector_for_doc(&tv, seg.field_infos(), local, field)
        }
        None => Ok(None),
    }
}

/// `indexReader.termVectors()`: [`term_vector`] with each segment's reader
/// opened once.
pub(crate) struct TermVectors<'r> {
    reader: &'r DirectoryReader,
    segments: Vec<Option<lucene_codecs::term_vectors::TermVectorsReader<'r>>>,
}

impl<'r> TermVectors<'r> {
    pub(crate) fn new(reader: &'r DirectoryReader) -> Result<Self> {
        let segments = reader
            .segment_readers()
            .iter()
            .map(|s| s.term_vectors_reader())
            .collect::<Result<_>>()?;
        Ok(Self { reader, segments })
    }

    /// `get(doc, field)` of global document `doc`.
    pub(crate) fn get(
        &self,
        doc: i32,
        field: &str,
    ) -> Result<Option<lucene_codecs::term_vectors::TermVectorField>> {
        let mut base = 0i32;
        for (seg, tv) in self.reader.segment_readers().iter().zip(&self.segments) {
            if doc >= base && doc < base.saturating_add(seg.max_doc) {
                return match tv {
                    Some(tv) => crate::term_vectors_query::term_vector_for_doc(
                        tv,
                        seg.field_infos(),
                        doc - base,
                        field,
                    ),
                    None => Ok(None),
                };
            }
            base = base.saturating_add(seg.max_doc);
        }
        Err(Error::IllegalArgument(format!(
            "doc {doc} is in no segment"
        )))
    }
}

/// [`analyze`] without collecting: `f` sees each token's term in turn.
pub(crate) fn for_each_token(
    analyzer: &Analyzer,
    field: &str,
    text: &str,
    mut f: impl FnMut(&str),
) -> Result<()> {
    let mut ts = analyzer.token_stream(field, text).map_err(analysis)?;
    ts.reset().map_err(analysis)?;
    while ts.increment_token().map_err(analysis)? {
        f(ts.attributes().term());
    }
    ts.end().map_err(analysis)?;
    Ok(())
}

/// `Float.parseFloat(boost)`.
pub(crate) fn parse_boost(boost: &str) -> Result<f32> {
    boost
        .trim()
        .parse::<f32>()
        .map_err(|_| Error::IllegalArgument(format!("For input string: \"{boost}\"")))
}

/// The similarity a classifier was given, shared by its searchers.
pub type SharedSimilarity = Option<Arc<dyn Similarity>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_strings_order_by_utf16_code_unit() {
        // U+FF61 sorts after U+1F600 as UTF-16 (the emoji is the surrogate
        // pair 0xD83D 0xDE00), before it as UTF-8 bytes.
        let (bmp, astral) = (
            JavaString("\u{ff61}".into()),
            JavaString("\u{1f600}".into()),
        );
        assert!(bmp.0.as_bytes() < astral.0.as_bytes());
        assert_eq!(astral.partial_cmp(&bmp), Some(Ordering::Less));
        assert_eq!(parse_boost(" 2.5 ").unwrap(), 2.5);
    }
}
