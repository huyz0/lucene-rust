//! Lucene 10.5.0's `search/similarities` package: every similarity it ships,
//! behind its `Similarity`/`SimScorer` contract, plus Lucene 10.6's
//! [`StableTflSimilarity`] (`branch_10_6`, #16243).
//!
//! [`crate::similarity`] is BM25 as the scorer tree's fast path computes it
//! (a weight, a norm-inverse table, block bounds over impacts). This module
//! is the general contract that path specialises: a [`Similarity`] computes
//! a document's norm at index time ([`NormSimilarity::compute_norm`]) and, per
//! query term, a [`SimScorer`] from the collection and term statistics
//! ([`Similarity::scorer`]) that scores a `(freq, norm)` pair. Any scorer
//! Lucene accepts must not decrease as `freq` grows nor increase as the
//! unsigned `norm` grows; that is what lets impacts bound a block for every
//! similarity, not just BM25.
//!
//! Every formula is ported with Java's arithmetic, operand for operand:
//! `SimilarityBase` scores in `double` and narrows once; where Java computes
//! in `float` (`NormalizationH3`'s `(F + 1F) / (T + 1F)`,
//! `LMJelinekMercerSimilarity`'s `1 - lambda`, every `TFIDFSimilarity`
//! product) so does this. The components Lucene lets a user choose --
//! [`BasicModel`], [`AfterEffect`], [`Normalization`], [`Distribution`],
//! [`Lambda`], [`Independence`], [`CollectionModel`], [`AxiomaticVariant`] --
//! are the closed sets Lucene ships, as enums; a user-defined similarity is a
//! type implementing [`Similarity`], as it is a subclass in Java.
//!
//! [`SimScorer::explain`] is each similarity's own breakdown, string for
//! string: BM25's and `TFIDFSimilarity`'s reuse `crate::explain`'s nodes,
//! `SimilarityBase`'s families explain their components (`BasicModel`,
//! `AfterEffect`, `Normalization`, `Distribution`, `Lambda`, ...) as their
//! Java classes do, and a similarity without one (`RawTFSimilarity`) keeps
//! `SimScorer.explain`'s default layout.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

pub use lucene_index::similarity::{default_compute_norm, FieldInvertState, NormSimilarity};
use lucene_util::small_float;

use crate::explain::Explanation;

mod stable_tfl;
pub use stable_tfl::StableTflSimilarity;

/// `CollectionStatistics`: a field's reader-wide statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectionStatistics {
    pub max_doc: i64,
    /// Documents with at least one term for the field.
    pub doc_count: i64,
    /// Sum of every term's `totalTermFreq`: the field's tokens.
    pub sum_total_term_freq: i64,
    /// Sum of every term's `docFreq`.
    pub sum_doc_freq: i64,
}

impl CollectionStatistics {
    /// `new CollectionStatistics(...)`'s checks: `maxDoc > 0`,
    /// `0 < docCount <= maxDoc`, `sumDocFreq >= docCount`,
    /// `sumTotalTermFreq >= sumDocFreq`.
    pub fn new(
        max_doc: i64,
        doc_count: i64,
        sum_total_term_freq: i64,
        sum_doc_freq: i64,
    ) -> Result<Self, String> {
        if max_doc <= 0 {
            return Err(format!("maxDoc must be positive, maxDoc: {max_doc}"));
        }
        if doc_count <= 0 || doc_count > max_doc {
            return Err(format!(
                "docCount must be positive and at most maxDoc, docCount: {doc_count}, maxDoc: {max_doc}"
            ));
        }
        if sum_doc_freq < doc_count {
            return Err(format!(
                "sumDocFreq must be at least docCount, sumDocFreq: {sum_doc_freq}, docCount: {doc_count}"
            ));
        }
        if sum_total_term_freq < sum_doc_freq {
            return Err(format!(
                "sumTotalTermFreq must be at least sumDocFreq, sumTotalTermFreq: {sum_total_term_freq}, sumDocFreq: {sum_doc_freq}"
            ));
        }
        Ok(Self {
            max_doc,
            doc_count,
            sum_total_term_freq,
            sum_doc_freq,
        })
    }
}

/// `TermStatistics`: a term's reader-wide statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TermStatistics {
    pub doc_freq: i64,
    pub total_term_freq: i64,
    /// `term()`: the term's bytes, which a similarity that scores on the
    /// term itself ([`StableTflSimilarity`]) reads. `None` where this port
    /// builds statistics without one (a caller's `TermStatistics::new`, a
    /// synthetic statistic); Java's constructor requires the term.
    pub term: Option<Arc<[u8]>>,
}

impl TermStatistics {
    /// `new TermStatistics(...)`'s checks: `docFreq > 0`,
    /// `totalTermFreq >= docFreq`.
    pub fn new(doc_freq: i64, total_term_freq: i64) -> Result<Self, String> {
        if doc_freq <= 0 {
            return Err(format!("docFreq must be positive, docFreq: {doc_freq}"));
        }
        if total_term_freq < doc_freq {
            return Err(format!(
                "totalTermFreq must be at least docFreq, totalTermFreq: {total_term_freq}, docFreq: {doc_freq}"
            ));
        }
        Ok(Self {
            doc_freq,
            total_term_freq,
            term: None,
        })
    }

    /// These statistics with `term()` set to `term`'s bytes.
    pub fn with_term(mut self, term: &[u8]) -> Self {
        self.term = Some(Arc::from(term));
        self
    }
}

/// `Similarity.SimScorer`: a term's (or a phrase's) score for one document.
pub trait SimScorer: Send + Sync {
    /// `score(float freq, long norm)`: `norm` as the field's `NumericDocValues`
    /// returns it (a sign-extended byte for the default norms), `1` when the
    /// field has no norms.
    fn score(&self, freq: f32, norm: i64) -> f32;

    /// `BulkSimScorer.score`: `DefaultBulkSimScorer`, one call per document.
    fn score_bulk(&self, freqs: &[f32], norms: &[i64], scores: &mut Vec<f32>) {
        scores.clear();
        scores.extend(freqs.iter().zip(norms).map(|(&f, &n)| self.score(f, n)));
    }

    /// `BM25Scorer`'s `weight` and its `cache` of `normInverse` per norm byte,
    /// when this is one: [`Self::score`] is then exactly `weight - weight /
    /// (1 + freq * cache[norm as u8])`, which a scorer's per-document loop can
    /// compute without a dynamic call per document (and over a batch at a
    /// time). `None` for every other similarity.
    fn bm25_parts(&self) -> Option<(f32, &[f32; 256])> {
        None
    }

    /// `explain(freq, norm)`: how [`Self::score`] of `freq`'s value and
    /// `norm` was computed, `freq` being the frequency's own explanation.
    /// The default is `SimScorer.explain`'s: the score over `freq`.
    fn explain(&self, freq: Explanation, norm: i64) -> Explanation {
        Explanation::match_(
            self.score(freq.value, norm),
            format!("score(freq={}), with freq of:", freq.value_string()),
        )
        .with_details(vec![freq])
    }
}

/// `Similarity`: the index-time norm ([`NormSimilarity`], which lives in
/// `lucene-index` because the writer computes norms and that crate sits below
/// this one) and the query-time scorer. An `Arc<dyn Similarity>` upcasts to
/// the `Arc<dyn NormSimilarity>` `IndexWriter::set_similarity` takes.
pub trait Similarity: NormSimilarity {
    /// `Similarity.scorer(boost, collectionStats, termStats...)`. `field` is
    /// `collectionStats.field()`.
    fn scorer(
        &self,
        field: &str,
        boost: f32,
        collection: &CollectionStatistics,
        terms: &[TermStatistics],
    ) -> Arc<dyn SimScorer>;

    /// Whether this is BM25 at Lucene's default parameters with overlaps
    /// discounted: the similarity the scorer tree's fast path computes.
    fn is_default_bm25(&self) -> bool {
        false
    }

    /// `IDFValueSource.asTFIDF(sim, field)`: this similarity as a
    /// `TFIDFSimilarity` (a `PerFieldSimilarityWrapper` unwrapped to
    /// `field`'s), or `None` when it is not one.
    fn as_tfidf(&self, _field: &str) -> Option<&dyn TfIdfSimilarity> {
        None
    }

    /// This similarity as a `BM25Similarity` for `field` (a
    /// `PerFieldSimilarityWrapper` unwrapped to `field`'s), whose `k1` and
    /// `b` its scorer's explanation reports; `None` when it is not one.
    fn as_bm25(&self, _field: &str) -> Option<&Bm25Similarity> {
        None
    }

    /// An owned copy of this similarity, for a caller that must keep it past
    /// the borrow it was handed (an explanation's segment state); `None`
    /// for a similarity that cannot be copied.
    fn shared(&self) -> Option<Arc<dyn Similarity>> {
        None
    }

    /// `getClass().getSimpleName()`, which a weight's explanation prints
    /// (`weight(q in doc) [BM25Similarity], result of:`). A similarity this
    /// port does not know the Java class of is `Similarity`.
    fn simple_name(&self) -> &str {
        "Similarity"
    }
}

/// `TFIDFSimilarity`'s `tf` and `idf`, which the function queries' `tf()`
/// and `idf()` sources read.
pub trait TfIdfSimilarity: Similarity {
    /// `tf(freq)`.
    fn tf(&self, freq: f32) -> f32;
    /// `idf(docFreq, docCount)`.
    fn idf(&self, doc_freq: i64, doc_count: i64) -> f32;
    /// `TFIDFSimilarity.scorer`'s `normTable`: this similarity's
    /// `lengthNorm` of each norm byte's length, which its scorer and its
    /// explanation (`fieldNorm`) read.
    fn norm_table(&self) -> [f32; 256];
}

/// `SmallFloat.byte4ToInt((byte) i)` for every norm byte, as a float
/// (`SimilarityBase.LENGTH_TABLE`, `BM25Similarity.LENGTH_TABLE`).
fn length_table() -> [f32; 256] {
    std::array::from_fn(|i| small_float::byte4_to_int(i as u8) as f32)
}

/// `SimilarityBase.log2`.
fn log2(x: f64) -> f64 {
    x.ln() / std::f64::consts::LN_2
}

/// Java's `Math.max(double, double)`: `NaN` wins, and `0.0 > -0.0`.
fn java_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == b {
        // Equal: only the zeros differ, and `+0.0` is the larger.
        return if a.is_sign_negative() { b } else { a };
    }
    if a > b {
        a
    } else {
        b
    }
}

// ---------------------------------------------------------------------------
// BM25
// ---------------------------------------------------------------------------

/// `BM25Similarity`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bm25Similarity {
    k1: f32,
    b: f32,
    discount_overlaps: bool,
}

impl Default for Bm25Similarity {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            discount_overlaps: true,
        }
    }
}

impl Bm25Similarity {
    /// `BM25Similarity(k1, b, discountOverlaps)`.
    pub fn new(k1: f32, b: f32, discount_overlaps: bool) -> Result<Self, String> {
        if !k1.is_finite() || k1 < 0.0 {
            return Err(format!(
                "illegal k1 value: {k1}, must be a non-negative finite value"
            ));
        }
        if b.is_nan() || !(0.0..=1.0).contains(&b) {
            return Err(format!("illegal b value: {b}, must be between 0 and 1"));
        }
        Ok(Self {
            k1,
            b,
            discount_overlaps,
        })
    }

    pub fn k1(&self) -> f32 {
        self.k1
    }

    pub fn b(&self) -> f32 {
        self.b
    }

    /// `BM25Similarity.idf`.
    pub fn idf(doc_freq: i64, doc_count: i64) -> f32 {
        (1.0 + ((doc_count - doc_freq) as f64 + 0.5) / (doc_freq as f64 + 0.5)).ln() as f32
    }
}

struct Bm25Scorer {
    weight: f32,
    cache: [f32; 256],
    /// What [`SimScorer::explain`] reports: Java builds the idf node
    /// eagerly; this keeps the statistics and builds it when asked.
    sim: Bm25Similarity,
    boost: f32,
    collection: CollectionStatistics,
    terms: Vec<TermStatistics>,
}

impl SimScorer for Bm25Scorer {
    fn score(&self, freq: f32, norm: i64) -> f32 {
        let norm_inverse = self.cache[norm as u8 as usize];
        self.weight - self.weight / (1.0 + freq * norm_inverse)
    }
    fn bm25_parts(&self) -> Option<(f32, &[f32; 256])> {
        Some((self.weight, &self.cache))
    }
    /// `BM25Scorer.explain`.
    fn explain(&self, freq: Explanation, norm: i64) -> Explanation {
        let doc_count = self.collection.doc_count;
        let idfs = self
            .terms
            .iter()
            .map(|t| {
                let idf = Bm25Similarity::idf(t.doc_freq, doc_count);
                crate::explain::idf_explanation(idf, t.doc_freq, doc_count)
            })
            .collect();
        let idf = crate::explain::summed_idf_explanation(idfs, "idf, sum of:");
        let value = self.score(freq.value, norm);
        crate::explain::explain_bm25(
            &self.sim,
            self.boost,
            value,
            idf,
            freq,
            norm,
            &self.collection,
        )
    }
}

/// `idfExplain(collectionStats, termStats[])`'s value: one term's idf, or
/// the float idfs of several summed as a `double` and narrowed once.
pub(crate) fn summed_idf(idf: impl Fn(&TermStatistics) -> f32, terms: &[TermStatistics]) -> f32 {
    match terms {
        [one] => idf(one),
        many => many.iter().map(|t| f64::from(idf(t))).sum::<f64>() as f32,
    }
}

impl NormSimilarity for Bm25Similarity {
    fn discount_overlaps(&self) -> bool {
        self.discount_overlaps
    }
}

impl Similarity for Bm25Similarity {
    fn simple_name(&self) -> &str {
        "BM25Similarity"
    }

    fn scorer(
        &self,
        _field: &str,
        boost: f32,
        collection: &CollectionStatistics,
        terms: &[TermStatistics],
    ) -> Arc<dyn SimScorer> {
        let idf = summed_idf(|t| Self::idf(t.doc_freq, collection.doc_count), terms);
        let avgdl = (collection.sum_total_term_freq as f64 / collection.doc_count as f64) as f32;
        let lengths = length_table();
        let (k1, b) = (self.k1, self.b);
        let cache = std::array::from_fn(|i| 1.0 / (k1 * ((1.0 - b) + b * lengths[i] / avgdl)));
        Arc::new(Bm25Scorer {
            weight: boost * idf,
            cache,
            sim: *self,
            boost,
            collection: *collection,
            terms: terms.to_vec(),
        })
    }

    fn is_default_bm25(&self) -> bool {
        *self == Self::default()
    }

    fn as_bm25(&self, _field: &str) -> Option<&Bm25Similarity> {
        Some(self)
    }

    fn shared(&self) -> Option<Arc<dyn Similarity>> {
        Some(Arc::new(*self))
    }
}

// ---------------------------------------------------------------------------
// TF-IDF, Boolean, raw TF
// ---------------------------------------------------------------------------

/// `ClassicSimilarity`: `TFIDFSimilarity` with `sqrt(freq)`,
/// `log((docCount + 1) / (docFreq + 1)) + 1` and `1 / sqrt(length)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassicSimilarity {
    discount_overlaps: bool,
}

impl Default for ClassicSimilarity {
    fn default() -> Self {
        Self {
            discount_overlaps: true,
        }
    }
}

impl ClassicSimilarity {
    pub fn new(discount_overlaps: bool) -> Self {
        Self { discount_overlaps }
    }

    /// `ClassicSimilarity.tf`.
    pub fn tf(freq: f32) -> f32 {
        f64::from(freq).sqrt() as f32
    }

    /// `ClassicSimilarity.idf`.
    pub fn idf(doc_freq: i64, doc_count: i64) -> f32 {
        (((doc_count + 1) as f64 / (doc_freq + 1) as f64).ln() + 1.0) as f32
    }

    /// `ClassicSimilarity.lengthNorm`.
    pub fn length_norm(num_terms: i32) -> f32 {
        (1.0 / f64::from(num_terms).sqrt()) as f32
    }
}

struct TfIdfScorer {
    query_weight: f32,
    norm_table: [f32; 256],
    sim: ClassicSimilarity,
    explain: TfIdfExplain,
}

impl SimScorer for TfIdfScorer {
    fn score(&self, freq: f32, norm: i64) -> f32 {
        let raw = ClassicSimilarity::tf(freq) * self.query_weight;
        raw * self.norm_table[(norm & 0xFF) as usize]
    }
    fn explain(&self, freq: Explanation, norm: i64) -> Explanation {
        self.explain.explain(&self.sim, freq, norm)
    }
}

/// What `TFIDFScorer.explain` reports beyond its score: the boost and the
/// statistics its idf node is built from (when asked, where Java builds it
/// in `scorer`).
pub(crate) struct TfIdfExplain {
    boost: f32,
    doc_count: i64,
    terms: Vec<TermStatistics>,
}

impl TfIdfExplain {
    pub(crate) fn new(
        boost: f32,
        collection: &CollectionStatistics,
        terms: &[TermStatistics],
    ) -> Self {
        Self {
            boost,
            doc_count: collection.doc_count,
            terms: terms.to_vec(),
        }
    }

    /// `TFIDFScorer.explain(freq, norm)` under `sim`.
    pub(crate) fn explain(
        &self,
        sim: &dyn TfIdfSimilarity,
        freq: Explanation,
        norm: i64,
    ) -> Explanation {
        let idfs = self
            .terms
            .iter()
            .map(|t| crate::explain::tfidf_idf_explanation(sim, t.doc_freq, self.doc_count))
            .collect();
        let idf = crate::explain::summed_idf_explanation(idfs, "idf(), sum of:");
        crate::explain::explain_tfidf(sim, self.boost, idf, freq, norm)
    }
}

impl NormSimilarity for ClassicSimilarity {
    fn discount_overlaps(&self) -> bool {
        self.discount_overlaps
    }
}

impl Similarity for ClassicSimilarity {
    fn simple_name(&self) -> &str {
        "ClassicSimilarity"
    }

    fn scorer(
        &self,
        _field: &str,
        boost: f32,
        collection: &CollectionStatistics,
        terms: &[TermStatistics],
    ) -> Arc<dyn SimScorer> {
        let idf = summed_idf(|t| Self::idf(t.doc_freq, collection.doc_count), terms);
        Arc::new(TfIdfScorer {
            query_weight: boost * idf,
            norm_table: Self::norm_table(),
            sim: *self,
            explain: TfIdfExplain::new(boost, collection, terms),
        })
    }

    fn as_tfidf(&self, _field: &str) -> Option<&dyn TfIdfSimilarity> {
        Some(self)
    }

    fn shared(&self) -> Option<Arc<dyn Similarity>> {
        Some(Arc::new(*self))
    }
}

impl ClassicSimilarity {
    /// `TFIDFSimilarity.scorer`'s `normTable`: `lengthNorm` of each norm
    /// byte's length from byte 1 up, then byte 0 as the inverse of byte
    /// 255's.
    pub(crate) fn norm_table() -> [f32; 256] {
        let mut norm_table = [0.0f32; 256];
        for (i, slot) in norm_table.iter_mut().enumerate().skip(1) {
            *slot = Self::length_norm(small_float::byte4_to_int(i as u8) as i32);
        }
        norm_table[0] = 1.0 / norm_table[255];
        norm_table
    }
}

impl TfIdfSimilarity for ClassicSimilarity {
    fn tf(&self, freq: f32) -> f32 {
        Self::tf(freq)
    }
    fn idf(&self, doc_freq: i64, doc_count: i64) -> f32 {
        Self::idf(doc_freq, doc_count)
    }
    fn norm_table(&self) -> [f32; 256] {
        Self::norm_table()
    }
}

/// `BooleanSimilarity`: every match scores its query boost.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BooleanSimilarity;

struct ConstantSimScorer(f32);

impl SimScorer for ConstantSimScorer {
    fn score(&self, _freq: f32, _norm: i64) -> f32 {
        self.0
    }
    /// `BooleanSimilarity.BooleanWeight.explain`: the boost (the scorer's
    /// class names itself).
    fn explain(&self, _freq: Explanation, _norm: i64) -> Explanation {
        Explanation::match_(self.0, "score(BooleanWeight), computed from:")
            .with_details(vec![Explanation::match_(self.0, "boost, query boost")])
    }
}

impl NormSimilarity for BooleanSimilarity {}

impl Similarity for BooleanSimilarity {
    fn simple_name(&self) -> &str {
        "BooleanSimilarity"
    }

    fn scorer(
        &self,
        _field: &str,
        boost: f32,
        _collection: &CollectionStatistics,
        _terms: &[TermStatistics],
    ) -> Arc<dyn SimScorer> {
        Arc::new(ConstantSimScorer(boost))
    }

    fn shared(&self) -> Option<Arc<dyn Similarity>> {
        Some(Arc::new(*self))
    }
}

/// `RawTFSimilarity`: `boost * freq`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawTfSimilarity {
    discount_overlaps: bool,
}

impl Default for RawTfSimilarity {
    fn default() -> Self {
        Self {
            discount_overlaps: true,
        }
    }
}

impl RawTfSimilarity {
    pub fn new(discount_overlaps: bool) -> Self {
        Self { discount_overlaps }
    }
}

struct RawTfScorer(f32);

impl SimScorer for RawTfScorer {
    fn score(&self, freq: f32, _norm: i64) -> f32 {
        self.0 * freq
    }
}

impl NormSimilarity for RawTfSimilarity {
    fn discount_overlaps(&self) -> bool {
        self.discount_overlaps
    }
}

impl Similarity for RawTfSimilarity {
    fn simple_name(&self) -> &str {
        "RawTFSimilarity"
    }

    fn scorer(
        &self,
        _field: &str,
        boost: f32,
        _collection: &CollectionStatistics,
        _terms: &[TermStatistics],
    ) -> Arc<dyn SimScorer> {
        Arc::new(RawTfScorer(boost))
    }

    fn shared(&self) -> Option<Arc<dyn Similarity>> {
        Some(Arc::new(*self))
    }
}

// ---------------------------------------------------------------------------
// SimilarityBase and its families
// ---------------------------------------------------------------------------

/// `BasicStats` (with `LMStats`' collection probability).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasicStats {
    pub number_of_documents: i64,
    pub number_of_field_tokens: i64,
    pub avg_field_length: f64,
    pub doc_freq: i64,
    pub total_term_freq: i64,
    pub boost: f64,
    /// `LMStats.collectionProbability`; `0` outside the language models.
    pub collection_probability: f64,
}

/// `SimilarityBase`: a similarity scored in `double` from [`BasicStats`].
trait BaseModel: Send + Sync + fmt::Debug {
    /// `SimilarityBase.score(stats, freq, docLen)`.
    fn score(&self, stats: &BasicStats, freq: f64, doc_len: f64) -> f64;

    /// `LMSimilarity`'s collection model, if this is a language model.
    fn collection_model(&self) -> Option<CollectionModel> {
        None
    }

    /// `getClass().getSimpleName()`.
    fn simple_name(&self) -> &'static str;

    /// `SimilarityBase.explain(subExpls, stats, freq, docLen)`: the details
    /// [`Self::explain`]'s default reports. `LMSimilarity`'s (the
    /// collection probability) for a language model, none otherwise.
    fn explain_details(
        &self,
        subs: &mut Vec<Explanation>,
        stats: &BasicStats,
        _freq: f64,
        _doc_len: f64,
    ) {
        if let Some(model) = self.collection_model() {
            subs.push(lm_collection_probability(stats, model));
        }
    }

    /// `SimilarityBase.explain(stats, freq, docLen)`.
    fn explain(&self, stats: &BasicStats, freq: Explanation, doc_len: f64) -> Explanation {
        let f = f64::from(freq.value);
        let mut subs = Vec::new();
        self.explain_details(&mut subs, stats, f, doc_len);
        Explanation::match_(
            self.score(stats, f, doc_len) as f32,
            format!(
                "score({}, freq={}), computed from:",
                self.simple_name(),
                freq.value_string()
            ),
        )
        .with_details(subs)
    }
}

/// `SimilarityBase`'s subclasses' `score(<name>, freq=f), computed as ...`
/// top line over `subs`; `freq.getValue().doubleValue()` is what they score.
fn base_explanation(
    model: &impl BaseModel,
    stats: &BasicStats,
    freq: &Explanation,
    doc_len: f64,
    computed_as: &str,
    subs: Vec<Explanation>,
) -> Explanation {
    Explanation::match_(
        model.score(stats, freq.double_value(), doc_len) as f32,
        format!(
            "score({}, freq={}), computed as {computed_as} from:",
            model.simple_name(),
            freq.value_string()
        ),
    )
    .with_details(subs)
}

/// `subs.add(Explanation.match((float) stats.getBoost(), description))`
/// unless the boost is 1.
fn push_boost(subs: &mut Vec<Explanation>, stats: &BasicStats, description: &str) {
    if stats.boost != 1.0 {
        subs.push(Explanation::match_(stats.boost as f32, description));
    }
}

struct BasicSimScorer<M> {
    model: M,
    stats: BasicStats,
    lengths: [f32; 256],
}

impl<M: BaseModel> SimScorer for BasicSimScorer<M> {
    fn score(&self, freq: f32, norm: i64) -> f32 {
        let len = f64::from(self.lengths[norm as u8 as usize]);
        self.model.score(&self.stats, f64::from(freq), len) as f32
    }
    fn explain(&self, freq: Explanation, norm: i64) -> Explanation {
        let len = f64::from(self.lengths[norm as u8 as usize]);
        self.model.explain(&self.stats, freq, len)
    }
}

/// `MultiSimilarity.MultiSimScorer`: the float scores summed as a `double`.
struct MultiSimScorer(Vec<Arc<dyn SimScorer>>);

impl SimScorer for MultiSimScorer {
    fn score(&self, freq: f32, norm: i64) -> f32 {
        self.0
            .iter()
            .map(|s| f64::from(s.score(freq, norm)))
            .sum::<f64>() as f32
    }
    fn explain(&self, freq: Explanation, norm: i64) -> Explanation {
        let value = self.score(freq.value, norm);
        let subs = self
            .0
            .iter()
            .map(|s| s.explain(freq.clone(), norm))
            .collect();
        Explanation::match_(value, "sum of:").with_details(subs)
    }
}

/// `SimilarityBase.scorer`: one [`BasicSimScorer`] per term, summed by a
/// `MultiSimScorer` when there are several.
fn base_scorer<M: BaseModel + Clone + 'static>(
    model: &M,
    boost: f32,
    collection: &CollectionStatistics,
    terms: &[TermStatistics],
) -> Arc<dyn SimScorer> {
    let lengths = length_table();
    let mut scorers: Vec<Arc<dyn SimScorer>> = terms
        .iter()
        .map(|t| {
            let mut stats = BasicStats {
                number_of_documents: collection.doc_count,
                number_of_field_tokens: collection.sum_total_term_freq,
                avg_field_length: collection.sum_total_term_freq as f64
                    / collection.doc_count as f64,
                doc_freq: t.doc_freq,
                total_term_freq: t.total_term_freq,
                boost: f64::from(boost),
                collection_probability: 0.0,
            };
            if let Some(cm) = model.collection_model() {
                stats.collection_probability = cm.compute_probability(&stats);
            }
            Arc::new(BasicSimScorer {
                model: model.clone(),
                stats,
                lengths,
            }) as Arc<dyn SimScorer>
        })
        .collect();
    if scorers.len() == 1 {
        scorers.pop().unwrap()
    } else {
        Arc::new(MultiSimScorer(scorers))
    }
}

/// `BasicModel`: DFR's basic randomness models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasicModel {
    /// `BasicModelG`: geometric approximation of Bose-Einstein.
    G,
    /// `BasicModelIF`: inverse term frequency.
    IF,
    /// `BasicModelIn`: inverse document frequency.
    In,
    /// `BasicModelIne`: inverse expected document frequency.
    Ine,
}

impl BasicModel {
    /// `BasicModel.score(stats, tfn, aeTimes1pTfn)`.
    pub fn score(self, stats: &BasicStats, tfn: f64, ae_times_1p_tfn: f64) -> f64 {
        match self {
            BasicModel::G => {
                let f = (stats.total_term_freq + 1) as f64;
                let n = stats.number_of_documents as f64;
                let lambda = f / (n + f);
                let a = log2(lambda + 1.0);
                let b = log2((1.0 + lambda) / lambda);
                (b - (b - a) / (1.0 + tfn)) * ae_times_1p_tfn
            }
            BasicModel::IF => {
                let n = stats.number_of_documents;
                let f = stats.total_term_freq;
                let a = log2(1.0 + (n + 1) as f64 / (f as f64 + 0.5));
                a * ae_times_1p_tfn * (1.0 - 1.0 / (1.0 + tfn))
            }
            BasicModel::In => {
                let n = stats.number_of_documents;
                let df = stats.doc_freq;
                let a = log2((n + 1) as f64 / (df as f64 + 0.5));
                a * ae_times_1p_tfn * (1.0 - 1.0 / (1.0 + tfn))
            }
            BasicModel::Ine => {
                let n = stats.number_of_documents;
                let f = stats.total_term_freq;
                let ne = n as f64 * (1.0 - ((n - 1) as f64 / n as f64).powf(f as f64));
                let a = log2((n + 1) as f64 / (ne + 0.5));
                a * ae_times_1p_tfn * (1.0 - 1.0 / (1.0 + tfn))
            }
        }
    }

    /// `getClass().getSimpleName()`.
    pub fn simple_name(self) -> &'static str {
        match self {
            BasicModel::G => "BasicModelG",
            BasicModel::IF => "BasicModelIF",
            BasicModel::In => "BasicModelIn",
            BasicModel::Ine => "BasicModelIne",
        }
    }

    /// `BasicModel.explain(stats, tfn, aeTimes1pTfn)`.
    pub fn explain(self, stats: &BasicStats, tfn: f64, ae_times_1p_tfn: f64) -> Explanation {
        let value =
            (self.score(stats, tfn, ae_times_1p_tfn) * (1.0 + tfn) / ae_times_1p_tfn) as f32;
        let tfn_expl = Explanation::match_(tfn as f32, "tfn, normalized term frequency");
        let n_docs = "N, total number of documents with field";
        let (formula, details) = match self {
            BasicModel::G => {
                let f = (stats.total_term_freq + 1) as f64;
                let n = stats.number_of_documents as f64;
                let lambda = f / (n + f);
                let lambda =
                    Explanation::match_(lambda as f32, "lambda, computed as F / (N + F) from:")
                        .with_details(vec![
                            Explanation::match_(
                                f as f32,
                                "F, total number of occurrences of term across all docs + 1",
                            ),
                            Explanation::match_(n as f32, n_docs),
                        ]);
                (
                    "log2(lambda + 1) + tfn * log2((1 + lambda) / lambda)",
                    vec![tfn_expl, lambda],
                )
            }
            BasicModel::IF => (
                "tfn * log2(1 + (N + 1) / (F + 0.5))",
                vec![
                    tfn_expl,
                    Explanation::match_long(stats.number_of_documents, n_docs),
                    Explanation::match_long(
                        stats.total_term_freq,
                        "F, total number of occurrences of term across all documents",
                    ),
                ],
            ),
            BasicModel::In => (
                "tfn * log2((N + 1) / (n + 0.5))",
                vec![
                    tfn_expl,
                    Explanation::match_long(stats.number_of_documents, n_docs),
                    Explanation::match_long(
                        stats.doc_freq,
                        "n, number of documents containing term",
                    ),
                ],
            ),
            BasicModel::Ine => {
                let f = stats.total_term_freq as f64;
                let n = stats.number_of_documents as f64;
                let ne = n * (1.0 - ((n - 1.0) / n).powf(f));
                let ne = Explanation::match_(
                    ne as f32,
                    "ne, computed as N * (1 - Math.pow((N - 1) / N, F)) from:",
                )
                .with_details(vec![
                    Explanation::match_(
                        f as f32,
                        "F, total number of occurrences of term across all docs",
                    ),
                    Explanation::match_(n as f32, n_docs),
                ]);
                ("tfn * log2((N + 1) / (ne + 0.5))", vec![tfn_expl, ne])
            }
        };
        Explanation::match_(
            value,
            format!("{}, computed as {formula} from:", self.simple_name()),
        )
        .with_details(details)
    }
}

/// `AfterEffect`: DFR's first normalisation of information gain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AfterEffect {
    /// `AfterEffectB`: Bernoulli ratio.
    B,
    /// `AfterEffectL`: Laplace's law of succession.
    L,
}

impl AfterEffect {
    /// `AfterEffect.scoreTimes1pTfn(stats)`.
    pub fn score_times_1p_tfn(self, stats: &BasicStats) -> f64 {
        match self {
            AfterEffect::B => {
                let f = stats.total_term_freq + 1;
                let n = stats.doc_freq + 1;
                (f as f64 + 1.0) / n as f64
            }
            AfterEffect::L => 1.0,
        }
    }

    /// `AfterEffect.explain(stats, tfn)`.
    pub fn explain(self, stats: &BasicStats, tfn: f64) -> Explanation {
        let value = (self.score_times_1p_tfn(stats) / (1.0 + tfn)) as f32;
        let tfn_expl = || Explanation::match_(tfn as f32, "tfn, normalized term frequency");
        match self {
            AfterEffect::B => Explanation::match_(
                value,
                "AfterEffectB, computed as (F + 1) / (n * (tfn + 1)) from:",
            )
            .with_details(vec![
                tfn_expl(),
                Explanation::match_long(
                    stats.total_term_freq,
                    "F, total number of occurrences of term across all documents + 1",
                ),
                Explanation::match_long(
                    stats.doc_freq,
                    "n, number of documents containing term + 1",
                ),
                tfn_expl(),
            ]),
            AfterEffect::L => {
                Explanation::match_(value, "AfterEffectL, computed as 1 / (tfn + 1) from:")
                    .with_details(vec![tfn_expl()])
            }
        }
    }
}

/// `Normalization`: term-frequency normalisation by document length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Normalization {
    /// `Normalization.NoNormalization`.
    None,
    /// `NormalizationH1(c)`: uniform distribution of frequency.
    H1(f32),
    /// `NormalizationH2(c)`: frequency density decreasing with length.
    H2(f32),
    /// `NormalizationH3(mu)`: Dirichlet priors.
    H3(f32),
    /// `NormalizationZ(z)`: Pareto-Zipf.
    Z(f32),
}

impl Normalization {
    /// `NormalizationH1()`, `c = 1`.
    pub const H1_DEFAULT: Normalization = Normalization::H1(1.0);
    /// `NormalizationH2()`, `c = 1`.
    pub const H2_DEFAULT: Normalization = Normalization::H2(1.0);
    /// `NormalizationH3()`, `mu = 800`.
    pub const H3_DEFAULT: Normalization = Normalization::H3(800.0);
    /// `NormalizationZ()`, `z = 0.30`.
    pub const Z_DEFAULT: Normalization = Normalization::Z(0.30);

    /// The constructors' argument checks.
    pub fn checked(self) -> Result<Self, String> {
        match self {
            Normalization::H1(c) | Normalization::H2(c) if !c.is_finite() || c < 0.0 => Err(
                format!("illegal c value: {c}, must be a non-negative finite value"),
            ),
            Normalization::H3(mu) if !mu.is_finite() || mu < 0.0 => Err(format!(
                "illegal mu value: {mu}, must be a non-negative finite value"
            )),
            Normalization::Z(z) if z.is_nan() || z <= 0.0 || z >= 0.5 => Err(format!(
                "illegal z value: {z}, must be in the range (0 .. 0.5)"
            )),
            ok => Ok(ok),
        }
    }

    /// `Normalization.tfn(stats, tf, len)`.
    pub fn tfn(self, stats: &BasicStats, tf: f64, len: f64) -> f64 {
        match self {
            Normalization::None => tf,
            Normalization::H1(c) => tf * f64::from(c) * (stats.avg_field_length / len),
            Normalization::H2(c) => tf * log2(1.0 + f64::from(c) * stats.avg_field_length / len),
            Normalization::H3(mu) => {
                // `(F + 1F) / (T + 1F)` and its product with `mu` are float
                // arithmetic in Java; the sum with `tf` is where it widens.
                let ratio = (stats.total_term_freq as f32 + 1.0)
                    / (stats.number_of_field_tokens as f32 + 1.0);
                (tf + f64::from(mu * ratio)) / (len + f64::from(mu)) * f64::from(mu)
            }
            Normalization::Z(z) => tf * (stats.avg_field_length / len).powf(f64::from(z)),
        }
    }

    /// `Normalization.explain(stats, tf, len)`.
    pub fn explain(self, stats: &BasicStats, tf: f64, len: f64) -> Explanation {
        let value = self.tfn(stats, tf, len) as f32;
        let tf_expl = || {
            Explanation::match_(
                tf as f32,
                "tf, number of occurrences of term in the document",
            )
        };
        let avgfl = || {
            Explanation::match_(
                stats.avg_field_length as f32,
                "avgfl, average length of field across all documents",
            )
        };
        let fl = || Explanation::match_(len as f32, "fl, field length of the document");
        let (name, formula, details) = match self {
            // `Explanation.match(1, ...)`: an `Integer`.
            Normalization::None => return Explanation::match_long(1, "no normalization"),
            Normalization::H1(c) => (
                "NormalizationH1",
                "tf * c * (avgfl / fl)",
                vec![
                    tf_expl(),
                    Explanation::match_(c, "c, hyper-parameter"),
                    avgfl(),
                    fl(),
                ],
            ),
            Normalization::H2(c) => (
                "NormalizationH2",
                "tf * log2(1 + c * avgfl / fl)",
                vec![
                    tf_expl(),
                    Explanation::match_(c, "c, hyper-parameter"),
                    avgfl(),
                    fl(),
                ],
            ),
            Normalization::H3(mu) => (
                "NormalizationH3",
                "(tf + mu * ((F+1) / (T+1))) / (fl + mu) * mu",
                vec![
                    tf_expl(),
                    Explanation::match_(mu, "mu, smoothing parameter"),
                    Explanation::match_(
                        stats.total_term_freq as f32,
                        "F,  total number of occurrences of term across all documents",
                    ),
                    Explanation::match_(
                        stats.number_of_field_tokens as f32,
                        "T, total number of tokens of the field across all documents",
                    ),
                    fl(),
                ],
            ),
            Normalization::Z(z) => (
                "NormalizationZ",
                "tf * Math.pow(avgfl / fl, z)",
                vec![
                    tf_expl(),
                    avgfl(),
                    fl(),
                    Explanation::match_(z, "z, relates to specificity of the language"),
                ],
            ),
        };
        Explanation::match_(value, format!("{name}, computed as {formula} from:"))
            .with_details(details)
    }
}

/// `DFRSimilarity`: divergence from randomness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DfrSimilarity {
    pub basic_model: BasicModel,
    pub after_effect: AfterEffect,
    pub normalization: Normalization,
    discount_overlaps: bool,
}

impl DfrSimilarity {
    pub fn new(
        basic_model: BasicModel,
        after_effect: AfterEffect,
        normalization: Normalization,
    ) -> Result<Self, String> {
        Self::with_discount_overlaps(basic_model, after_effect, normalization, true)
    }

    pub fn with_discount_overlaps(
        basic_model: BasicModel,
        after_effect: AfterEffect,
        normalization: Normalization,
        discount_overlaps: bool,
    ) -> Result<Self, String> {
        Ok(Self {
            basic_model,
            after_effect,
            normalization: normalization.checked()?,
            discount_overlaps,
        })
    }
}

impl BaseModel for DfrSimilarity {
    fn score(&self, stats: &BasicStats, freq: f64, doc_len: f64) -> f64 {
        let tfn = self.normalization.tfn(stats, freq, doc_len);
        let ae_times_1p_tfn = self.after_effect.score_times_1p_tfn(stats);
        stats.boost * self.basic_model.score(stats, tfn, ae_times_1p_tfn)
    }

    fn simple_name(&self) -> &'static str {
        "DFRSimilarity"
    }

    fn explain(&self, stats: &BasicStats, freq: Explanation, doc_len: f64) -> Explanation {
        let f = freq.double_value();
        let mut subs = Vec::with_capacity(4);
        push_boost(&mut subs, stats, "boost, query boost");
        let tfn = self.normalization.tfn(stats, f, doc_len);
        let ae_times_1p_tfn = self.after_effect.score_times_1p_tfn(stats);
        subs.push(self.normalization.explain(stats, f, doc_len));
        subs.push(self.basic_model.explain(stats, tfn, ae_times_1p_tfn));
        subs.push(self.after_effect.explain(stats, tfn));
        base_explanation(
            self,
            stats,
            &freq,
            doc_len,
            "boost * basicModel.score(stats, tfn) * afterEffect.score(stats, tfn)",
            subs,
        )
    }
}

/// `Distribution`: IB's probabilistic distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distribution {
    /// `DistributionLL`: log-logistic.
    LL,
    /// `DistributionSPL`: smoothed power-law.
    SPL,
}

impl Distribution {
    /// `Distribution.score(stats, tfn, lambda)`.
    pub fn score(self, tfn: f64, lambda: f64) -> f64 {
        match self {
            Distribution::LL => -(lambda / (tfn + lambda)).ln(),
            Distribution::SPL => {
                let mut q = 1.0 - 1.0 / (tfn + 1.0);
                if q == 1.0 {
                    q = 1.0f64.next_down();
                }
                let mut pow = lambda.powf(q);
                if pow == lambda {
                    pow = if lambda < 1.0 {
                        lambda.next_up()
                    } else {
                        lambda.next_down()
                    };
                }
                -((pow - lambda) / (1.0 - lambda)).ln()
            }
        }
    }

    /// `Distribution.explain(stats, tfn, lambda)`: the score, named.
    pub fn explain(self, tfn: f64, lambda: f64) -> Explanation {
        let name = match self {
            Distribution::LL => "DistributionLL",
            Distribution::SPL => "DistributionSPL",
        };
        Explanation::match_(self.score(tfn, lambda) as f32, name)
    }
}

/// `Lambda`: IB's `w` parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lambda {
    /// `LambdaDF`: from document frequency.
    DF,
    /// `LambdaTTF`: from total term frequency.
    TTF,
}

impl Lambda {
    /// `Lambda.lambda(stats)` -- a `float`, as in Java.
    pub fn lambda(self, stats: &BasicStats) -> f32 {
        match self {
            Lambda::DF => {
                let lambda = ((stats.doc_freq as f64 + 1.0)
                    / (stats.number_of_documents as f64 + 1.0)) as f32;
                if lambda == 1.0 {
                    lambda.next_down()
                } else {
                    lambda
                }
            }
            Lambda::TTF => {
                let lambda = ((stats.total_term_freq as f64 + 1.0)
                    / (stats.number_of_documents as f64 + 1.0)) as f32;
                if lambda == 1.0 {
                    lambda.next_up()
                } else {
                    lambda
                }
            }
        }
    }

    /// `Lambda.explain(stats)`.
    pub fn explain(self, stats: &BasicStats) -> Explanation {
        let n_docs = Explanation::match_long(
            stats.number_of_documents,
            "N, total number of documents with field",
        );
        let (description, count) = match self {
            Lambda::DF => (
                "LambdaDF, computed as (n + 1) / (N + 1) from:",
                Explanation::match_long(stats.doc_freq, "n, number of documents containing term"),
            ),
            Lambda::TTF => (
                "LambdaTTF, computed as (F + 1) / (N + 1) from:",
                Explanation::match_long(
                    stats.total_term_freq,
                    "F, total number of occurrences of term across all documents",
                ),
            ),
        };
        Explanation::match_(self.lambda(stats), description).with_details(vec![count, n_docs])
    }
}

/// `IBSimilarity`: information-based models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IbSimilarity {
    pub distribution: Distribution,
    pub lambda: Lambda,
    pub normalization: Normalization,
    discount_overlaps: bool,
}

impl IbSimilarity {
    pub fn new(
        distribution: Distribution,
        lambda: Lambda,
        normalization: Normalization,
    ) -> Result<Self, String> {
        Self::with_discount_overlaps(distribution, lambda, normalization, true)
    }

    pub fn with_discount_overlaps(
        distribution: Distribution,
        lambda: Lambda,
        normalization: Normalization,
        discount_overlaps: bool,
    ) -> Result<Self, String> {
        Ok(Self {
            distribution,
            lambda,
            normalization: normalization.checked()?,
            discount_overlaps,
        })
    }
}

impl BaseModel for IbSimilarity {
    fn score(&self, stats: &BasicStats, freq: f64, doc_len: f64) -> f64 {
        stats.boost
            * self.distribution.score(
                self.normalization.tfn(stats, freq, doc_len),
                f64::from(self.lambda.lambda(stats)),
            )
    }

    fn simple_name(&self) -> &'static str {
        "IBSimilarity"
    }

    fn explain(&self, stats: &BasicStats, freq: Explanation, doc_len: f64) -> Explanation {
        let mut subs = Vec::with_capacity(4);
        push_boost(&mut subs, stats, "boost, query boost");
        let norm = self
            .normalization
            .explain(stats, freq.double_value(), doc_len);
        let lambda = self.lambda.explain(stats);
        // Java hands the distribution the two nodes' float values.
        let distribution = self
            .distribution
            .explain(f64::from(norm.value), f64::from(lambda.value));
        subs.extend([norm, lambda, distribution]);
        base_explanation(
            self,
            stats,
            &freq,
            doc_len,
            "boost * distribution.score(stats, normalization.tfn(stats, freq, docLen), \
             lambda.lambda(stats))",
            subs,
        )
    }
}

/// `Independence`: DFI's measure of divergence from independence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Independence {
    /// `IndependenceStandardized`.
    Standardized,
    /// `IndependenceSaturated`.
    Saturated,
    /// `IndependenceChiSquared`.
    ChiSquared,
}

impl Independence {
    /// `Independence.score(freq, expected)`.
    pub fn score(self, freq: f64, expected: f64) -> f64 {
        match self {
            Independence::Standardized => (freq - expected) / expected.sqrt(),
            Independence::Saturated => (freq - expected) / expected,
            Independence::ChiSquared => (freq - expected) * (freq - expected) / expected,
        }
    }
}

/// `DFISimilarity`: divergence from independence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DfiSimilarity {
    pub independence: Independence,
    discount_overlaps: bool,
}

impl DfiSimilarity {
    pub fn new(independence: Independence) -> Self {
        Self::with_discount_overlaps(independence, true)
    }

    pub fn with_discount_overlaps(independence: Independence, discount_overlaps: bool) -> Self {
        Self {
            independence,
            discount_overlaps,
        }
    }
}

impl BaseModel for DfiSimilarity {
    fn score(&self, stats: &BasicStats, freq: f64, doc_len: f64) -> f64 {
        let expected = (stats.total_term_freq + 1) as f64 * doc_len
            / (stats.number_of_field_tokens + 1) as f64;
        if freq <= expected {
            return 0.0;
        }
        let measure = self.independence.score(freq, expected);
        stats.boost * log2(measure + 1.0)
    }

    fn simple_name(&self) -> &'static str {
        "DFISimilarity"
    }

    fn explain(&self, stats: &BasicStats, freq: Explanation, doc_len: f64) -> Explanation {
        let f = freq.double_value();
        let expected = (stats.total_term_freq + 1) as f64 * doc_len
            / (stats.number_of_field_tokens + 1) as f64;
        if f <= expected {
            return Explanation::match_(
                0.0,
                format!(
                    "score(DFISimilarity, freq={}), equals to 0",
                    freq.value_string()
                ),
            );
        }
        let expected_expl = Explanation::match_(
            expected as f32,
            "expected, computed as (F + 1) * dl / (T + 1) from:",
        )
        .with_details(vec![
            Explanation::match_long(
                stats.total_term_freq,
                "F, total number of occurrences of term across all docs",
            ),
            Explanation::match_(doc_len as f32, "dl, length of field"),
            Explanation::match_long(
                stats.number_of_field_tokens,
                "T, total number of tokens in the field",
            ),
        ]);
        let measure = self.independence.score(f, expected);
        let measure = Explanation::match_(
            measure as f32,
            "measure, computed as independence.score(freq, expected) from:",
        )
        .with_details(vec![freq.clone(), expected_expl]);
        let subs = vec![
            Explanation::match_(stats.boost as f32, "boost, query boost"),
            measure,
        ];
        base_explanation(
            self,
            stats,
            &freq,
            doc_len,
            "boost * log2(measure + 1)",
            subs,
        )
    }
}

/// `LMSimilarity.CollectionModel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectionModel {
    /// `LMSimilarity.DefaultCollectionModel`: `(F + 1) / (T + 1)`.
    Default,
    /// `IndriDirichletSimilarity.IndriCollectionModel`: `F / T`.
    Indri,
}

impl CollectionModel {
    /// `CollectionModel.computeProbability(stats)`.
    pub fn compute_probability(self, stats: &BasicStats) -> f64 {
        match self {
            CollectionModel::Default => {
                (stats.total_term_freq as f64 + 1.0) / (stats.number_of_field_tokens as f64 + 1.0)
            }
            CollectionModel::Indri => {
                stats.total_term_freq as f64 / stats.number_of_field_tokens as f64
            }
        }
    }
}

/// `LMDirichletSimilarity`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LmDirichletSimilarity {
    pub collection_model: CollectionModel,
    pub mu: f32,
    discount_overlaps: bool,
}

impl Default for LmDirichletSimilarity {
    fn default() -> Self {
        Self {
            collection_model: CollectionModel::Default,
            mu: 2000.0,
            discount_overlaps: true,
        }
    }
}

impl LmDirichletSimilarity {
    /// `LMDirichletSimilarity(collectionModel, discountOverlaps, mu)`.
    pub fn new(
        collection_model: CollectionModel,
        discount_overlaps: bool,
        mu: f32,
    ) -> Result<Self, String> {
        if !mu.is_finite() || mu < 0.0 {
            return Err(format!(
                "illegal mu value: {mu}, must be a non-negative finite value"
            ));
        }
        Ok(Self {
            collection_model,
            mu,
            discount_overlaps,
        })
    }
}

impl BaseModel for LmDirichletSimilarity {
    fn score(&self, stats: &BasicStats, freq: f64, doc_len: f64) -> f64 {
        let mu = f64::from(self.mu);
        let score = stats.boost
            * ((1.0 + freq / (mu * stats.collection_probability)).ln()
                + (mu / (doc_len + mu)).ln());
        if score > 0.0 {
            score
        } else {
            0.0
        }
    }

    fn collection_model(&self) -> Option<CollectionModel> {
        Some(self.collection_model)
    }

    fn simple_name(&self) -> &'static str {
        "LMDirichletSimilarity"
    }

    fn explain(&self, stats: &BasicStats, freq: Explanation, doc_len: f64) -> Explanation {
        let f = freq.double_value();
        let mu = f64::from(self.mu);
        let p = stats.collection_probability;
        let mut subs = Vec::with_capacity(6);
        push_boost(&mut subs, stats, "query boost");
        subs.push(Explanation::match_(self.mu, "mu"));
        subs.push(
            Explanation::match_(
                (1.0 + f / (mu * p)).ln() as f32,
                "term weight, computed as log(1 + freq /(mu * P)) from:",
            )
            .with_details(vec![lm_freq(f), lm_probability(p)]),
        );
        subs.push(Explanation::match_(
            (mu / (doc_len + mu)).ln() as f32,
            "document norm, computed as log(mu / (dl + mu))",
        ));
        subs.push(Explanation::match_(doc_len as f32, "dl, length of field"));
        self.explain_details(&mut subs, stats, f, doc_len);
        base_explanation(
            self,
            stats,
            &freq,
            doc_len,
            "boost * (term weight + document norm)",
            subs,
        )
    }
}

/// `LMSimilarity.explain(subs, ...)`: the collection probability.
fn lm_collection_probability(stats: &BasicStats, model: CollectionModel) -> Explanation {
    Explanation::match_(
        model.compute_probability(stats) as f32,
        "collection probability",
    )
}

/// The language models' `freq` detail.
fn lm_freq(freq: f64) -> Explanation {
    Explanation::match_(
        freq as f32,
        "freq, number of occurrences of term in the document",
    )
}

/// The language models' `P` detail.
fn lm_probability(p: f64) -> Explanation {
    Explanation::match_(
        p as f32,
        "P, probability that the current term is generated by the collection",
    )
}

/// `LMJelinekMercerSimilarity`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LmJelinekMercerSimilarity {
    pub collection_model: CollectionModel,
    pub lambda: f32,
    discount_overlaps: bool,
}

impl LmJelinekMercerSimilarity {
    /// `LMJelinekMercerSimilarity(collectionModel, discountOverlaps, lambda)`.
    pub fn new(
        collection_model: CollectionModel,
        discount_overlaps: bool,
        lambda: f32,
    ) -> Result<Self, String> {
        if lambda.is_nan() || lambda <= 0.0 || lambda > 1.0 {
            return Err("lambda must be in the range (0 .. 1]".to_string());
        }
        Ok(Self {
            collection_model,
            lambda,
            discount_overlaps,
        })
    }
}

impl BaseModel for LmJelinekMercerSimilarity {
    fn score(&self, stats: &BasicStats, freq: f64, doc_len: f64) -> f64 {
        // `1 - lambda` is float arithmetic in Java.
        let one_minus = f64::from(1.0f32 - self.lambda);
        stats.boost
            * (1.0
                + (one_minus * freq / doc_len)
                    / (f64::from(self.lambda) * stats.collection_probability))
                .ln()
    }

    fn collection_model(&self) -> Option<CollectionModel> {
        Some(self.collection_model)
    }

    fn simple_name(&self) -> &'static str {
        "LMJelinekMercerSimilarity"
    }

    fn explain(&self, stats: &BasicStats, freq: Explanation, doc_len: f64) -> Explanation {
        let f = freq.double_value();
        let mut subs = Vec::with_capacity(6);
        push_boost(&mut subs, stats, "boost");
        subs.push(Explanation::match_(self.lambda, "lambda"));
        subs.push(lm_probability(stats.collection_probability));
        subs.push(lm_freq(f));
        subs.push(Explanation::match_(doc_len as f32, "dl, length of field"));
        self.explain_details(&mut subs, stats, f, doc_len);
        base_explanation(
            self,
            stats,
            &freq,
            doc_len,
            "boost * log(1 + ((1 - lambda) * freq / dl) /(lambda * P))",
            subs,
        )
    }
}

/// `IndriDirichletSimilarity`. Its score ignores the boost, as Java's does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndriDirichletSimilarity {
    pub collection_model: CollectionModel,
    pub mu: f32,
    discount_overlaps: bool,
}

impl Default for IndriDirichletSimilarity {
    /// `IndriDirichletSimilarity()`: the Indri collection model, `mu = 2000`.
    fn default() -> Self {
        Self {
            collection_model: CollectionModel::Indri,
            mu: 2000.0,
            discount_overlaps: true,
        }
    }
}

impl IndriDirichletSimilarity {
    /// `IndriDirichletSimilarity(collectionModel, discountOverlaps, mu)`.
    /// (`IndriDirichletSimilarity(float mu)` is the `Default` collection
    /// model, not Indri's: it inherits `LMSimilarity()`.)
    pub fn new(collection_model: CollectionModel, discount_overlaps: bool, mu: f32) -> Self {
        Self {
            collection_model,
            mu,
            discount_overlaps,
        }
    }
}

impl BaseModel for IndriDirichletSimilarity {
    fn score(&self, stats: &BasicStats, freq: f64, doc_len: f64) -> f64 {
        let mu = f64::from(self.mu);
        let score = (freq + mu * stats.collection_probability) / (doc_len + mu);
        score.ln()
    }

    fn collection_model(&self) -> Option<CollectionModel> {
        Some(self.collection_model)
    }

    fn simple_name(&self) -> &'static str {
        "IndriDirichletSimilarity"
    }

    /// `IndriDirichletSimilarity.explain(subs, ...)`, under
    /// `SimilarityBase.explain`'s top line.
    fn explain_details(
        &self,
        subs: &mut Vec<Explanation>,
        stats: &BasicStats,
        freq: f64,
        doc_len: f64,
    ) {
        if stats.boost != 1.0 {
            // `Explanation.match(stats.getBoost(), ...)`: a `Double`.
            subs.push(Explanation::match_double(stats.boost, "boost"));
        }
        let mu = f64::from(self.mu);
        subs.push(Explanation::match_(self.mu, "mu"));
        let p = stats.collection_probability;
        subs.push(Explanation::match_(
            ((freq + mu * p) / (doc_len + mu)).ln() as f32,
            "term weight",
        ));
        subs.push(Explanation::match_(
            (mu / (doc_len + mu)).ln() as f32,
            "document norm",
        ));
        subs.push(lm_collection_probability(stats, self.collection_model));
    }
}

/// The six axiomatic retrieval functions (`AxiomaticF1EXP` ... `F3LOG`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxiomaticVariant {
    F1Exp,
    F1Log,
    F2Exp,
    F2Log,
    F3Exp,
    F3Log,
}

impl AxiomaticVariant {
    /// The Java subclass's simple name (`AxiomaticF1EXP`, ...).
    pub fn simple_name(self) -> &'static str {
        match self {
            AxiomaticVariant::F1Exp => "AxiomaticF1EXP",
            AxiomaticVariant::F1Log => "AxiomaticF1LOG",
            AxiomaticVariant::F2Exp => "AxiomaticF2EXP",
            AxiomaticVariant::F2Log => "AxiomaticF2LOG",
            AxiomaticVariant::F3Exp => "AxiomaticF3EXP",
            AxiomaticVariant::F3Log => "AxiomaticF3LOG",
        }
    }
}

/// `Axiomatic` and its subclasses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxiomaticSimilarity {
    pub variant: AxiomaticVariant,
    pub s: f32,
    pub query_len: i32,
    pub k: f32,
    discount_overlaps: bool,
}

impl AxiomaticSimilarity {
    /// `Axiomatic(discountOverlaps, s, queryLen, k)`'s checks. The Java
    /// subclasses' constructors fix `queryLen` (1 for F1/F2) and default
    /// `k` to 0.35 and `s` to 0.25.
    pub fn new(
        variant: AxiomaticVariant,
        discount_overlaps: bool,
        s: f32,
        query_len: i32,
        k: f32,
    ) -> Result<Self, String> {
        if !s.is_finite() || !(0.0..=1.0).contains(&s) {
            return Err(format!("illegal s value: {s}, must be between 0 and 1"));
        }
        if !k.is_finite() || !(0.0..=1.0).contains(&k) {
            return Err(format!("illegal k value: {k}, must be between 0 and 1"));
        }
        if query_len < 0 {
            return Err(format!(
                "illegal query length value: {query_len}, must be larger 0"
            ));
        }
        Ok(Self {
            variant,
            s,
            query_len,
            k,
            discount_overlaps,
        })
    }

    fn log_tf(freq: f64) -> f64 {
        // "otherwise gives negative scores for freqs < 1"
        let freq = freq + 1.0;
        1.0 + (1.0 + freq.ln()).ln()
    }
}

impl AxiomaticSimilarity {
    /// `tf(stats, freq, docLen)`.
    fn tf(&self, freq: f64) -> f64 {
        use AxiomaticVariant::*;
        match self.variant {
            F1Exp | F1Log | F3Exp | F3Log => Self::log_tf(freq),
            F2Exp | F2Log => 1.0,
        }
    }

    /// `ln(stats, freq, docLen)`.
    fn ln(&self, stats: &BasicStats, doc_len: f64) -> f64 {
        let s = f64::from(self.s);
        let avgdl = stats.avg_field_length;
        match self.variant {
            AxiomaticVariant::F1Exp | AxiomaticVariant::F1Log => {
                (avgdl + s) / (avgdl + doc_len * s)
            }
            _ => 1.0,
        }
    }

    /// `tfln(stats, freq, docLen)`.
    fn tfln(&self, stats: &BasicStats, freq: f64, doc_len: f64) -> f64 {
        let s = f64::from(self.s);
        match self.variant {
            AxiomaticVariant::F2Exp | AxiomaticVariant::F2Log => {
                freq / (freq + s + s * doc_len / stats.avg_field_length)
            }
            _ => 1.0,
        }
    }

    /// Whether `idf` is `Math.pow((N + 1) / n, k)` (the `EXP` variants)
    /// rather than `log((N + 1) / n)`.
    fn exp_idf(&self) -> bool {
        use AxiomaticVariant::*;
        matches!(self.variant, F1Exp | F2Exp | F3Exp)
    }

    /// `idf(stats, freq, docLen)`.
    fn idf(&self, stats: &BasicStats) -> f64 {
        let ratio = (stats.number_of_documents as f64 + 1.0) / stats.doc_freq as f64;
        if self.exp_idf() {
            ratio.powf(f64::from(self.k))
        } else {
            ratio.ln()
        }
    }

    /// `gamma(stats, freq, docLen)`.
    fn gamma(&self, stats: &BasicStats, doc_len: f64) -> f64 {
        match self.variant {
            AxiomaticVariant::F3Exp | AxiomaticVariant::F3Log => {
                let q = f64::from(self.query_len);
                (doc_len - q) * f64::from(self.s) * q / stats.avg_field_length
            }
            _ => 0.0,
        }
    }

    /// `tf * ln * tfln * idf - gamma`, before the boost.
    fn unboosted(&self, stats: &BasicStats, freq: f64, doc_len: f64) -> f64 {
        self.tf(freq) * self.ln(stats, doc_len) * self.tfln(stats, freq, doc_len) * self.idf(stats)
            - self.gamma(stats, doc_len)
    }

    /// The subclasses' `tfExplain`, `lnExplain`, `tflnExplain` and
    /// `idfExplain`.
    fn component_explanations(
        &self,
        stats: &BasicStats,
        freq: f64,
        doc_len: f64,
    ) -> [Explanation; 4] {
        use AxiomaticVariant::*;
        let freq_expl = || {
            Explanation::match_(
                freq as f32,
                "freq, number of occurrences of term in the document",
            )
        };
        let avgdl = || {
            Explanation::match_(
                stats.avg_field_length as f32,
                "avgdl, average length of field across all documents",
            )
        };
        let dl = || Explanation::match_(doc_len as f32, "dl, length of field");
        let tf = self.tf(freq) as f32;
        let tf = match self.variant {
            F2Exp | F2Log => Explanation::match_(tf, "tf, term frequency, equals to 1"),
            _ => Explanation::match_(
                tf,
                "tf, term frequency computed as 1 + log(1 + log(freq)) from:",
            )
            .with_details(vec![freq_expl()]),
        };
        let ln = self.ln(stats, doc_len) as f32;
        let ln = match self.variant {
            F1Exp | F1Log => Explanation::match_(
                ln,
                "ln, document length computed as (avgdl + s) / (avgdl + dl * s) from:",
            )
            .with_details(vec![avgdl(), dl()]),
            _ => Explanation::match_(ln, "ln, document length, equals to 1"),
        };
        let tfln = self.tfln(stats, freq, doc_len) as f32;
        let tfln = match self.variant {
            F2Exp | F2Log => Explanation::match_(
                tfln,
                "tfln, mixed term frequency and document length, \
                 computed as freq / (freq + s + s * dl / avgdl) from:",
            )
            .with_details(vec![freq_expl(), dl(), avgdl()]),
            _ => Explanation::match_(
                tfln,
                "tfln, mixed term frequency and document length, equals to 1",
            ),
        };
        let idf = Explanation::match_(
            self.idf(stats) as f32,
            if self.exp_idf() {
                "idf, inverted document frequency computed as Math.pow((N + 1) / n, k) from:"
            } else {
                "idf, inverted document frequency computed as log((N + 1) / n) from:"
            },
        )
        .with_details(vec![
            Explanation::match_(
                stats.number_of_documents as f32,
                "N, total number of documents with field",
            ),
            Explanation::match_(
                stats.doc_freq as f32,
                "n, number of documents containing term",
            ),
        ]);
        [tf, ln, tfln, idf]
    }
}

impl BaseModel for AxiomaticSimilarity {
    fn score(&self, stats: &BasicStats, freq: f64, doc_len: f64) -> f64 {
        let mut score = self.unboosted(stats, freq, doc_len);
        score *= stats.boost;
        java_max(0.0, score)
    }

    fn simple_name(&self) -> &'static str {
        self.variant.simple_name()
    }

    /// `Axiomatic.explain(subs, stats, freq, docLen)`.
    fn explain_details(
        &self,
        subs: &mut Vec<Explanation>,
        stats: &BasicStats,
        freq: f64,
        doc_len: f64,
    ) {
        push_boost(subs, stats, "boost, query boost");
        subs.push(Explanation::match_(
            self.k,
            "k, hyperparam for the primitive weighting function",
        ));
        subs.push(Explanation::match_(
            self.s,
            "s, hyperparam for the growth function",
        ));
        subs.push(Explanation::match_long(
            i64::from(self.query_len),
            "queryLen, query length",
        ));
        subs.extend(self.component_explanations(stats, freq, doc_len));
        subs.push(Explanation::match_(
            self.gamma(stats, doc_len) as f32,
            "gamma",
        ));
    }

    /// `Axiomatic.explain(stats, freq, docLen)`: the unboosted score, then
    /// the boost, then the floor at 0.
    fn explain(&self, stats: &BasicStats, freq: Explanation, doc_len: f64) -> Explanation {
        let f = freq.double_value();
        let mut subs = Vec::with_capacity(10);
        self.explain_details(&mut subs, stats, f, doc_len);
        let score = self.unboosted(stats, f, doc_len);
        let mut explanation = Explanation::match_(
            score as f32,
            format!(
                "score({}, freq={}), computed from:",
                self.variant.simple_name(),
                freq.value_string()
            ),
        )
        .with_details(subs);
        if stats.boost != 1.0 {
            explanation = Explanation::match_(
                (score * stats.boost) as f32,
                "Boosted score, computed as (score * boost) from:",
            )
            .with_details(vec![
                explanation,
                Explanation::match_(stats.boost as f32, "Query boost"),
            ]);
        }
        if score < 0.0 {
            // `Explanation.match(0, ...)`: `Integer`s.
            explanation = Explanation::match_long(0, "max of:").with_details(vec![
                Explanation::match_long(0, "Minimum legal score"),
                explanation,
            ]);
        }
        explanation
    }
}

macro_rules! base_similarity {
    ($($t:ty),*) => {$(
        impl NormSimilarity for $t {
            fn discount_overlaps(&self) -> bool {
                self.discount_overlaps
            }
        }

        impl Similarity for $t {
            fn scorer(
                &self,
                _field: &str,
                boost: f32,
                collection: &CollectionStatistics,
                terms: &[TermStatistics],
            ) -> Arc<dyn SimScorer> {
                base_scorer(self, boost, collection, terms)
            }

            fn simple_name(&self) -> &str {
                BaseModel::simple_name(self)
            }

            fn shared(&self) -> Option<Arc<dyn Similarity>> {
                Some(Arc::new(*self))
            }
        }
    )*};
}

base_similarity!(
    DfrSimilarity,
    IbSimilarity,
    DfiSimilarity,
    LmDirichletSimilarity,
    LmJelinekMercerSimilarity,
    IndriDirichletSimilarity
);

impl NormSimilarity for AxiomaticSimilarity {
    fn discount_overlaps(&self) -> bool {
        self.discount_overlaps
    }
}

impl Similarity for AxiomaticSimilarity {
    fn scorer(
        &self,
        _field: &str,
        boost: f32,
        collection: &CollectionStatistics,
        terms: &[TermStatistics],
    ) -> Arc<dyn SimScorer> {
        base_scorer(self, boost, collection, terms)
    }

    fn simple_name(&self) -> &str {
        self.variant.simple_name()
    }

    fn shared(&self) -> Option<Arc<dyn Similarity>> {
        Some(Arc::new(*self))
    }
}

// ---------------------------------------------------------------------------
// Composition
// ---------------------------------------------------------------------------

/// `MultiSimilarity`: the sum of several similarities' scores; norms from
/// the first.
#[derive(Debug, Clone)]
pub struct MultiSimilarity {
    sims: Vec<Arc<dyn Similarity>>,
}

impl MultiSimilarity {
    /// At least one similarity: Java reads `sims[0]` for the norm.
    pub fn new(sims: Vec<Arc<dyn Similarity>>) -> Result<Self, String> {
        if sims.is_empty() {
            return Err("MultiSimilarity needs at least one similarity".to_string());
        }
        Ok(Self { sims })
    }
}

impl NormSimilarity for MultiSimilarity {
    fn compute_norm(&self, field: &str, state: &FieldInvertState) -> i64 {
        self.sims[0].compute_norm(field, state)
    }
}

impl Similarity for MultiSimilarity {
    fn simple_name(&self) -> &str {
        "MultiSimilarity"
    }

    fn scorer(
        &self,
        field: &str,
        boost: f32,
        collection: &CollectionStatistics,
        terms: &[TermStatistics],
    ) -> Arc<dyn SimScorer> {
        Arc::new(MultiSimScorer(
            self.sims
                .iter()
                .map(|s| s.scorer(field, boost, collection, terms))
                .collect(),
        ))
    }

    fn shared(&self) -> Option<Arc<dyn Similarity>> {
        Some(Arc::new(self.clone()))
    }
}

/// `PerFieldSimilarityWrapper`: a similarity chosen by field name. Java's is
/// abstract (`get(String)`); this is the common concrete form, a map with a
/// default.
#[derive(Debug, Clone)]
pub struct PerFieldSimilarity {
    default: Arc<dyn Similarity>,
    fields: HashMap<String, Arc<dyn Similarity>>,
}

impl PerFieldSimilarity {
    pub fn new(default: Arc<dyn Similarity>) -> Self {
        Self {
            default,
            fields: HashMap::new(),
        }
    }

    pub fn with_field(mut self, field: impl Into<String>, sim: Arc<dyn Similarity>) -> Self {
        self.fields.insert(field.into(), sim);
        self
    }

    /// `PerFieldSimilarityWrapper.get(name)`.
    pub fn get(&self, field: &str) -> &Arc<dyn Similarity> {
        self.fields.get(field).unwrap_or(&self.default)
    }
}

impl NormSimilarity for PerFieldSimilarity {
    fn compute_norm(&self, field: &str, state: &FieldInvertState) -> i64 {
        self.get(field).compute_norm(field, state)
    }
}

impl Similarity for PerFieldSimilarity {
    /// Java's `PerFieldSimilarityWrapper` is abstract, so the name printed is
    /// its subclass's; this port's concrete form names the wrapper.
    fn simple_name(&self) -> &str {
        "PerFieldSimilarityWrapper"
    }

    fn scorer(
        &self,
        field: &str,
        boost: f32,
        collection: &CollectionStatistics,
        terms: &[TermStatistics],
    ) -> Arc<dyn SimScorer> {
        self.get(field).scorer(field, boost, collection, terms)
    }

    fn is_default_bm25(&self) -> bool {
        self.default.is_default_bm25() && self.fields.values().all(|s| s.is_default_bm25())
    }

    fn as_tfidf(&self, field: &str) -> Option<&dyn TfIdfSimilarity> {
        self.get(field).as_tfidf(field)
    }

    fn as_bm25(&self, field: &str) -> Option<&Bm25Similarity> {
        self.get(field).as_bm25(field)
    }

    fn shared(&self) -> Option<Arc<dyn Similarity>> {
        Some(Arc::new(self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coll() -> CollectionStatistics {
        CollectionStatistics::new(100, 90, 5000, 900).unwrap()
    }

    fn term() -> TermStatistics {
        TermStatistics::new(10, 40).unwrap()
    }

    /// `MultiSimScorer.explain`: "sum of:" each sub-scorer's explanation,
    /// valued as the summed score -- for `MultiSimilarity` and for a
    /// `SimilarityBase` over several terms.
    #[test]
    fn multi_scorers_explain_as_a_sum() {
        let freq = || Explanation::match_(3.0, "freq, occurrences of term within document");
        let multi = MultiSimilarity::new(vec![
            Arc::new(Bm25Similarity::default()),
            Arc::new(BooleanSimilarity),
        ])
        .unwrap();
        let scorer = multi.scorer("f", 2.0, &coll(), &[term()]);
        let e = scorer.explain(freq(), 7);
        assert_eq!(e.description, "sum of:");
        assert_eq!(e.value, scorer.score(3.0, 7));
        assert_eq!(e.details.len(), 2);
        assert!(e.details[0]
            .description
            .starts_with("score(freq=3.0), computed as boost * idf * tf"));
        assert_eq!(
            e.details[1].description,
            "score(BooleanWeight), computed from:"
        );

        let dfi = DfiSimilarity::new(Independence::Saturated);
        let scorer = dfi.scorer("f", 1.0, &coll(), &[term(), term()]);
        let e = scorer.explain(freq(), 7);
        assert_eq!(e.description, "sum of:");
        assert_eq!(e.value, scorer.score(3.0, 7));
        assert!(e.details[0]
            .description
            .starts_with("score(DFISimilarity, freq=3.0)"));
    }

    /// Each similarity reports its Java class's simple name, which an
    /// explanation prints.
    #[test]
    fn simple_names_are_the_java_classes() {
        use AxiomaticVariant::*;
        let none = Normalization::None;
        let mut sims: Vec<(Arc<dyn Similarity>, &str)> = vec![
            (Arc::new(Bm25Similarity::default()), "BM25Similarity"),
            (Arc::new(ClassicSimilarity::default()), "ClassicSimilarity"),
            (Arc::new(BooleanSimilarity), "BooleanSimilarity"),
            (Arc::new(RawTfSimilarity::default()), "RawTFSimilarity"),
            (
                Arc::new(DfrSimilarity::new(BasicModel::G, AfterEffect::B, none).unwrap()),
                "DFRSimilarity",
            ),
            (
                Arc::new(IbSimilarity::new(Distribution::LL, Lambda::DF, none).unwrap()),
                "IBSimilarity",
            ),
            (
                Arc::new(DfiSimilarity::new(Independence::Saturated)),
                "DFISimilarity",
            ),
            (
                Arc::new(LmDirichletSimilarity::default()),
                "LMDirichletSimilarity",
            ),
            (
                Arc::new(
                    LmJelinekMercerSimilarity::new(CollectionModel::Default, true, 0.5).unwrap(),
                ),
                "LMJelinekMercerSimilarity",
            ),
            (
                Arc::new(IndriDirichletSimilarity::default()),
                "IndriDirichletSimilarity",
            ),
        ];
        for (v, name) in [
            (F1Exp, "AxiomaticF1EXP"),
            (F1Log, "AxiomaticF1LOG"),
            (F2Exp, "AxiomaticF2EXP"),
            (F2Log, "AxiomaticF2LOG"),
            (F3Exp, "AxiomaticF3EXP"),
            (F3Log, "AxiomaticF3LOG"),
        ] {
            sims.push((
                Arc::new(AxiomaticSimilarity::new(v, true, 0.25, 1, 0.35).unwrap()),
                name,
            ));
        }
        let bm25: Arc<dyn Similarity> = Arc::new(Bm25Similarity::default());
        sims.push((
            Arc::new(MultiSimilarity::new(vec![Arc::clone(&bm25)]).unwrap()),
            "MultiSimilarity",
        ));
        sims.push((
            Arc::new(PerFieldSimilarity::new(bm25)),
            "PerFieldSimilarityWrapper",
        ));
        for (sim, name) in &sims {
            assert_eq!(sim.simple_name(), *name);
        }
    }

    /// The constructors refuse what Java's refuse, with Java's messages.
    #[test]
    fn arguments_are_checked_as_java_checks_them() {
        assert!(CollectionStatistics::new(0, 0, 0, 0).is_err());
        assert!(CollectionStatistics::new(10, 11, 20, 20).is_err());
        assert!(CollectionStatistics::new(10, 5, 20, 4).is_err());
        assert!(CollectionStatistics::new(10, 5, 4, 5).is_err());
        assert!(TermStatistics::new(0, 1).is_err());
        assert!(TermStatistics::new(5, 4).is_err());
        assert!(Bm25Similarity::new(-1.0, 0.5, true).is_err());
        assert!(Bm25Similarity::new(1.2, 1.5, true).is_err());
        assert!(Bm25Similarity::new(f32::INFINITY, 0.5, true).is_err());
        for bad in [
            Normalization::H1(-1.0),
            Normalization::H2(f32::NAN),
            Normalization::H3(-0.5),
            Normalization::Z(0.5),
            Normalization::Z(0.0),
        ] {
            assert!(
                DfrSimilarity::new(BasicModel::G, AfterEffect::B, bad).is_err(),
                "{bad:?}"
            );
            assert!(IbSimilarity::new(Distribution::LL, Lambda::DF, bad).is_err());
        }
        assert!(LmDirichletSimilarity::new(CollectionModel::Default, true, -1.0).is_err());
        assert!(LmJelinekMercerSimilarity::new(CollectionModel::Default, true, 0.0).is_err());
        assert!(LmJelinekMercerSimilarity::new(CollectionModel::Default, true, 1.5).is_err());
        use AxiomaticVariant::F1Exp;
        assert!(AxiomaticSimilarity::new(F1Exp, true, 1.5, 1, 0.35).is_err());
        assert!(AxiomaticSimilarity::new(F1Exp, true, 0.25, 1, -0.1).is_err());
        assert!(AxiomaticSimilarity::new(F1Exp, true, 0.25, -1, 0.35).is_err());
        assert!(MultiSimilarity::new(vec![]).is_err());
    }

    /// `PerFieldSimilarityWrapper` scores and norms with the field's own
    /// similarity, and falls back to the default for any other field.
    #[test]
    fn per_field_dispatches_by_field_name() {
        let per_field = PerFieldSimilarity::new(Arc::new(Bm25Similarity::default()))
            .with_field("title", Arc::new(ClassicSimilarity::new(false)));
        let (c, t) = (coll(), [term()]);
        let classic = ClassicSimilarity::new(false).scorer("title", 1.0, &c, &t);
        let bm25 = Bm25Similarity::default().scorer("body", 1.0, &c, &t);
        assert_eq!(
            per_field
                .scorer("title", 1.0, &c, &t)
                .score(3.0, 7)
                .to_bits(),
            classic.score(3.0, 7).to_bits()
        );
        assert_eq!(
            per_field
                .scorer("body", 1.0, &c, &t)
                .score(3.0, 7)
                .to_bits(),
            bm25.score(3.0, 7).to_bits()
        );
        let state = FieldInvertState {
            docs_only: false,
            length: 10,
            num_overlap: 4,
            unique_term_count: 3,
            ..FieldInvertState::default()
        };
        // Classic without discounting counts the overlaps; BM25 does not.
        assert_ne!(
            per_field.compute_norm("title", &state),
            per_field.compute_norm("body", &state)
        );
        assert!(!per_field.is_default_bm25());
        assert!(PerFieldSimilarity::new(Arc::new(Bm25Similarity::default())).is_default_bm25());
    }

    #[test]
    fn default_bm25_is_recognised_and_bulk_scoring_is_per_document() {
        assert!(Bm25Similarity::default().is_default_bm25());
        assert!(!Bm25Similarity::new(2.0, 0.75, true)
            .unwrap()
            .is_default_bm25());
        assert!(!ClassicSimilarity::default().is_default_bm25());
        let s = DfiSimilarity::new(Independence::Saturated).scorer("f", 1.0, &coll(), &[term()]);
        let mut out = Vec::new();
        s.score_bulk(&[1.0, 5.0, 30.0], &[1, 40, -3], &mut out);
        let one: Vec<u32> = [(1.0, 1), (5.0, 40), (30.0, -3)]
            .iter()
            .map(|&(f, n)| s.score(f, n).to_bits())
            .collect();
        assert_eq!(out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), one);
        assert_eq!(java_max(0.0, f64::NAN).to_bits(), f64::NAN.to_bits());
        assert_eq!(java_max(-0.0, 0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(java_max(0.0, -0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(java_max(1.0, 2.0), 2.0);
        assert_eq!(java_max(3.0, 2.0), 3.0);
    }
}
