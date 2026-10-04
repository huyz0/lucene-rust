//! The term- and index-statistics sources: `DocFreqValueSource`,
//! `IDFValueSource`, `TermFreqValueSource`, `TFValueSource`,
//! `TotalTermFreqValueSource`, `SumTotalTermFreqValueSource`,
//! `NumDocsValueSource`, `MaxDocValueSource` and `NormValueSource`.
//!
//! The reader-wide numbers (a term's `docFreq` and `totalTermFreq`, a
//! field's `sumTotalTermFreq`, `numDocs`, `maxDoc`) are computed by
//! `createWeight` over every leaf (see the parent module's doc); `tf()`,
//! `idf()` and `norm()` read the searcher's similarity at `getValues`, as
//! Java does, and refuse anything but a `TFIDFSimilarity`.

use lucene_codecs::postings::{LazyDocsCursor, PostingsFlags};

use crate::function::docvalues::{
    Double, DoubleDocValues, Float, FloatDocValues, Int, IntDocValues, Long, LongDocValues,
};
use crate::function::{
    java_double, not_weighted, BoxValues, FunctionContext, TopLevel, ValueLeaf, ValueSource,
};
use crate::reader::{NumericDocValues, NO_MORE_DOCS};
use crate::similarities::{CollectionStatistics, SimScorer, TermStatistics, TfIdfSimilarity};
use crate::{Error, Result};

/// `UnsupportedOperationException("requires a TFIDFSimilarity (such as
/// ClassicSimilarity)")`.
fn needs_tfidf() -> Error {
    Error::Unsupported("requires a TFIDFSimilarity (such as ClassicSimilarity)".into())
}

/// `IDFValueSource.asTFIDF(searcher.getSimilarity(), field)`.
fn tfidf<'a>(leaf: &ValueLeaf<'a>, field: &str) -> Result<&'a dyn TfIdfSimilarity> {
    leaf.similarity()
        .and_then(|s| s.as_tfidf(field))
        .ok_or_else(needs_tfidf)
}

/// `DocFreqValueSource.ConstIntDocValues`: one `int` for every document,
/// described `parent=value`.
struct ConstIntValues {
    val: i32,
    description: String,
}

impl IntDocValues for ConstIntValues {
    fn description(&self) -> String {
        self.description.clone()
    }
    fn int_val(&mut self, _doc: i32) -> Result<i32> {
        Ok(self.val)
    }
    fn float_val(&mut self, _doc: i32) -> Result<f32> {
        Ok(self.val as f32)
    }
    fn long_val(&mut self, _doc: i32) -> Result<i64> {
        Ok(i64::from(self.val))
    }
    fn double_val(&mut self, _doc: i32) -> Result<f64> {
        Ok(f64::from(self.val))
    }
    fn str_val(&mut self, _doc: i32) -> Result<Option<String>> {
        Ok(Some(self.val.to_string()))
    }
    fn to_string_doc(&mut self, _doc: i32) -> Result<String> {
        Ok(format!("{}={}", self.description, self.val))
    }
}

fn const_int<'a>(val: i32, description: String) -> BoxValues<'a> {
    Box::new(Int(ConstIntValues { val, description }))
}

/// `DocFreqValueSource.ConstDoubleDocValues`.
struct ConstDoubleValues {
    val: f64,
    description: String,
}

impl DoubleDocValues for ConstDoubleValues {
    fn description(&self) -> String {
        self.description.clone()
    }
    fn double_val(&mut self, _doc: i32) -> Result<f64> {
        Ok(self.val)
    }
    fn float_val(&mut self, _doc: i32) -> Result<f32> {
        Ok(self.val as f32)
    }
    fn int_val(&mut self, _doc: i32) -> Result<i32> {
        Ok(self.val as i32)
    }
    fn long_val(&mut self, _doc: i32) -> Result<i64> {
        Ok(self.val as i64)
    }
    fn str_val(&mut self, _doc: i32) -> Result<Option<String>> {
        Ok(Some(java_double(self.val)))
    }
    fn to_string_doc(&mut self, _doc: i32) -> Result<String> {
        Ok(format!("{}={}", self.description, java_double(self.val)))
    }
}

/// The term a term-statistics source names: `field`/`val` for its
/// description, `indexedField`/`indexedBytes` for the lookup.
#[derive(Debug, Clone)]
struct TermSpec {
    field: String,
    val: String,
    indexed_field: String,
    indexed_bytes: Vec<u8>,
}

impl TermSpec {
    fn description(&self, name: &str) -> String {
        format!("{name}({},{})", self.field, self.val)
    }
}

/// What `createWeight` computes for a `docfreq`/`idf` source: the term's
/// top-level `docFreq` and the reader's `maxDoc`.
#[derive(Debug, Clone, Copy)]
struct DocFreqState {
    doc_freq: i32,
    max_doc: i32,
}

fn doc_freq_state(spec: &TermSpec, top: &TopLevel<'_>) -> Result<DocFreqState> {
    Ok(DocFreqState {
        doc_freq: top.doc_freq(&spec.indexed_field, &spec.indexed_bytes)?,
        max_doc: top.max_doc()?,
    })
}

/// `DocFreqValueSource`: the term's reader-wide `docFreq` (deleted
/// documents included), for every document.
#[derive(Debug, Clone)]
pub struct DocFreqValueSource {
    spec: TermSpec,
}

impl DocFreqValueSource {
    pub fn new(
        field: impl Into<String>,
        val: impl Into<String>,
        indexed_field: impl Into<String>,
        indexed_bytes: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            spec: TermSpec {
                field: field.into(),
                val: val.into(),
                indexed_field: indexed_field.into(),
                indexed_bytes: indexed_bytes.into(),
            },
        }
    }

    /// `name()`.
    pub fn name(&self) -> &'static str {
        "docfreq"
    }
}

impl ValueSource for DocFreqValueSource {
    fn get_values<'a>(
        &self,
        fcx: &FunctionContext,
        _leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        let state = fcx
            .get::<DocFreqState, _>(self)
            .ok_or_else(|| not_weighted(&self.description()))?;
        Ok(const_int(state.doc_freq, self.description()))
    }
    fn description(&self) -> String {
        self.spec.description(self.name())
    }
    fn create_weight(&self, fcx: &mut FunctionContext, top: &TopLevel<'_>) -> Result<()> {
        let state = doc_freq_state(&self.spec, top)?;
        fcx.put(self, state);
        Ok(())
    }
}

/// `IDFValueSource`: the similarity's `idf(docFreq, maxDoc)` of the term,
/// for every document (a `TFIDFSimilarity` only).
#[derive(Debug, Clone)]
pub struct IDFValueSource {
    spec: TermSpec,
}

impl IDFValueSource {
    pub fn new(
        field: impl Into<String>,
        val: impl Into<String>,
        indexed_field: impl Into<String>,
        indexed_bytes: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            spec: TermSpec {
                field: field.into(),
                val: val.into(),
                indexed_field: indexed_field.into(),
                indexed_bytes: indexed_bytes.into(),
            },
        }
    }

    /// `name()`.
    pub fn name(&self) -> &'static str {
        "idf"
    }
}

impl ValueSource for IDFValueSource {
    fn get_values<'a>(&self, fcx: &FunctionContext, leaf: &ValueLeaf<'a>) -> Result<BoxValues<'a>> {
        let sim = tfidf(leaf, &self.spec.field)?;
        let state = fcx
            .get::<DocFreqState, _>(self)
            .ok_or_else(|| not_weighted(&self.description()))?;
        let idf = sim.idf(i64::from(state.doc_freq), i64::from(state.max_doc));
        Ok(Box::new(Double(ConstDoubleValues {
            val: f64::from(idf),
            description: self.description(),
        })))
    }
    fn description(&self) -> String {
        self.spec.description(self.name())
    }
    fn create_weight(&self, fcx: &mut FunctionContext, top: &TopLevel<'_>) -> Result<()> {
        let state = doc_freq_state(&self.spec, top)?;
        fcx.put(self, state);
        Ok(())
    }
}

/// A term's postings as `TermFreqValueSource` walks them: re-opened (the
/// `reset()`) when a document before the last is asked for.
struct TermPostings<'a> {
    leaf: crate::exec::LeafContext<'a>,
    field: String,
    term: Vec<u8>,
    docs: Option<LazyDocsCursor<'a>>,
    at_doc: i32,
    last_doc: i32,
}

impl<'a> TermPostings<'a> {
    fn new(leaf: &ValueLeaf<'a>, field: &str, term: &[u8]) -> Result<Self> {
        let mut p = Self {
            leaf: leaf.ctx,
            field: field.to_string(),
            term: term.to_vec(),
            docs: None,
            at_doc: -1,
            last_doc: -1,
        };
        p.reset()?;
        Ok(p)
    }

    /// `reset()`: the term's postings from the start (an empty iterator
    /// when the field or the term is absent).
    fn reset(&mut self) -> Result<()> {
        self.docs = None;
        if let Some(ft) = self.leaf.fields.field(&self.field) {
            if let Some(seeked) = ft.seek_term_state(&self.term)? {
                let doc_in = self.leaf.doc_in.ok_or_else(|| {
                    Error::IllegalState(format!(
                        "termfreq({}): the segment's postings are not open",
                        self.field
                    ))
                })?;
                self.docs = Some(ft.lazy_postings_for(&seeked, doc_in, PostingsFlags::Freqs)?);
            }
        }
        self.at_doc = -1;
        Ok(())
    }

    /// The document's frequency of the term, `None` when it lacks it.
    fn freq(&mut self, doc: i32) -> Result<Option<i32>> {
        if doc < self.last_doc {
            self.reset()?;
        }
        self.last_doc = doc;
        if self.at_doc < doc {
            self.at_doc = match &mut self.docs {
                Some(d) => d.advance(doc)?,
                None => NO_MORE_DOCS,
            };
        }
        if self.at_doc > doc {
            return Ok(None);
        }
        Ok(Some(
            self.docs
                .as_ref()
                .and_then(LazyDocsCursor::freq)
                .unwrap_or(1),
        ))
    }
}

/// `TermFreqValueSource`: the document's frequency of the term (`0`
/// without it).
#[derive(Debug, Clone)]
pub struct TermFreqValueSource {
    spec: TermSpec,
}

impl TermFreqValueSource {
    pub fn new(
        field: impl Into<String>,
        val: impl Into<String>,
        indexed_field: impl Into<String>,
        indexed_bytes: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            spec: TermSpec {
                field: field.into(),
                val: val.into(),
                indexed_field: indexed_field.into(),
                indexed_bytes: indexed_bytes.into(),
            },
        }
    }

    /// `name()`.
    pub fn name(&self) -> &'static str {
        "termfreq"
    }
}

struct TermFreqValues<'a> {
    postings: TermPostings<'a>,
    description: String,
}

impl IntDocValues for TermFreqValues<'_> {
    fn description(&self) -> String {
        self.description.clone()
    }
    fn int_val(&mut self, doc: i32) -> Result<i32> {
        Ok(self.postings.freq(doc)?.unwrap_or(0))
    }
}

impl ValueSource for TermFreqValueSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        Ok(Box::new(Int(TermFreqValues {
            postings: TermPostings::new(leaf, &self.spec.indexed_field, &self.spec.indexed_bytes)?,
            description: self.description(),
        })))
    }
    fn description(&self) -> String {
        self.spec.description(self.name())
    }
}

/// `TFValueSource`: the similarity's `tf` of the document's frequency of
/// the term (a `TFIDFSimilarity` only).
#[derive(Debug, Clone)]
pub struct TFValueSource {
    spec: TermSpec,
}

impl TFValueSource {
    pub fn new(
        field: impl Into<String>,
        val: impl Into<String>,
        indexed_field: impl Into<String>,
        indexed_bytes: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            spec: TermSpec {
                field: field.into(),
                val: val.into(),
                indexed_field: indexed_field.into(),
                indexed_bytes: indexed_bytes.into(),
            },
        }
    }

    /// `name()`.
    pub fn name(&self) -> &'static str {
        "tf"
    }
}

struct TfValues<'a> {
    postings: TermPostings<'a>,
    sim: &'a dyn TfIdfSimilarity,
    description: String,
}

impl FloatDocValues for TfValues<'_> {
    fn description(&self) -> String {
        self.description.clone()
    }
    fn float_val(&mut self, doc: i32) -> Result<f32> {
        let freq = self.postings.freq(doc)?.unwrap_or(0);
        Ok(self.sim.tf(freq as f32))
    }
}

impl ValueSource for TFValueSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        let sim = tfidf(leaf, &self.spec.indexed_field)?;
        Ok(Box::new(Float(TfValues {
            postings: TermPostings::new(leaf, &self.spec.indexed_field, &self.spec.indexed_bytes)?,
            sim,
            description: self.description(),
        })))
    }
    fn description(&self) -> String {
        self.spec.description(self.name())
    }
}

/// A `long` computed by `createWeight`, for every document
/// (`TotalTermFreqValueSource`'s and `SumTotalTermFreqValueSource`'s
/// `LongDocValues`).
struct ConstLongValues {
    val: i64,
    description: String,
}

impl LongDocValues for ConstLongValues {
    fn description(&self) -> String {
        self.description.clone()
    }
    fn long_val(&mut self, _doc: i32) -> Result<i64> {
        Ok(self.val)
    }
}

/// What `createWeight` put in the context for a source whose `getValues`
/// returns it (Java's `(FunctionValues) context.get(this)`).
#[derive(Debug, Clone, Copy)]
struct WeightedLong(i64);

fn weighted_long<'a, S>(fcx: &FunctionContext, source: &S) -> Result<BoxValues<'a>>
where
    S: ValueSource,
{
    let state = fcx
        .get::<WeightedLong, _>(source)
        .filter(|_| !fcx.searcher_only)
        .ok_or_else(|| not_weighted(&source.description()))?;
    Ok(Box::new(Long(ConstLongValues {
        val: state.0,
        description: source.description(),
    })))
}

/// `TotalTermFreqValueSource`: the term's reader-wide `totalTermFreq`, for
/// every document.
#[derive(Debug, Clone)]
pub struct TotalTermFreqValueSource {
    spec: TermSpec,
}

impl TotalTermFreqValueSource {
    pub fn new(
        field: impl Into<String>,
        val: impl Into<String>,
        indexed_field: impl Into<String>,
        indexed_bytes: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            spec: TermSpec {
                field: field.into(),
                val: val.into(),
                indexed_field: indexed_field.into(),
                indexed_bytes: indexed_bytes.into(),
            },
        }
    }

    /// `name()`.
    pub fn name(&self) -> &'static str {
        "totaltermfreq"
    }
}

impl ValueSource for TotalTermFreqValueSource {
    fn get_values<'a>(
        &self,
        fcx: &FunctionContext,
        _leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        weighted_long(fcx, self)
    }
    fn description(&self) -> String {
        self.spec.description(self.name())
    }
    /// The term's `totalTermFreq` in each leaf, summed.
    fn create_weight(&self, fcx: &mut FunctionContext, top: &TopLevel<'_>) -> Result<()> {
        let mut ttf = 0i64;
        for leaf in &top.leaves {
            if let Some(t) = leaf.fields.field(&self.spec.indexed_field) {
                if let Some(s) = t.try_seek_exact(&self.spec.indexed_bytes)? {
                    ttf = ttf.wrapping_add(s.total_term_freq);
                }
            }
        }
        fcx.put(self, WeightedLong(ttf));
        Ok(())
    }
}

/// `SumTotalTermFreqValueSource`: the field's reader-wide
/// `sumTotalTermFreq`, for every document.
#[derive(Debug, Clone)]
pub struct SumTotalTermFreqValueSource {
    indexed_field: String,
}

impl SumTotalTermFreqValueSource {
    pub fn new(indexed_field: impl Into<String>) -> Self {
        Self {
            indexed_field: indexed_field.into(),
        }
    }

    /// `name()`.
    pub fn name(&self) -> &'static str {
        "sumtotaltermfreq"
    }
}

impl ValueSource for SumTotalTermFreqValueSource {
    fn get_values<'a>(
        &self,
        fcx: &FunctionContext,
        _leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        weighted_long(fcx, self)
    }
    fn description(&self) -> String {
        format!("{}({})", self.name(), self.indexed_field)
    }
    /// Each leaf's `Terms.getSumTotalTermFreq()` (`0` for a leaf without
    /// the field), summed.
    fn create_weight(&self, fcx: &mut FunctionContext, top: &TopLevel<'_>) -> Result<()> {
        let mut sum = 0i64;
        for leaf in &top.leaves {
            if let Some(t) = leaf.fields.field(&self.indexed_field) {
                sum = sum.wrapping_add(t.sum_total_term_freq);
            }
        }
        fcx.put(self, WeightedLong(sum));
        Ok(())
    }
}

/// `NumDocsValueSource`: the top-level reader's `numDocs()`, for every
/// document.
#[derive(Debug, Clone, Copy, Default)]
pub struct NumDocsValueSource;

/// The top-level `numDocs()`/`maxDoc()` `createWeight` read.
#[derive(Debug, Clone, Copy)]
struct ReaderCount(i32);

impl NumDocsValueSource {
    pub fn new() -> Self {
        Self
    }

    /// `name()`.
    pub fn name(&self) -> &'static str {
        "numdocs"
    }
}

impl ValueSource for NumDocsValueSource {
    fn get_values<'a>(
        &self,
        fcx: &FunctionContext,
        _leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        let n = fcx
            .get::<ReaderCount, _>(self)
            .ok_or_else(|| not_weighted(&self.description()))?;
        Ok(const_int(n.0, self.description()))
    }
    fn description(&self) -> String {
        format!("{}()", self.name())
    }
    fn create_weight(&self, fcx: &mut FunctionContext, top: &TopLevel<'_>) -> Result<()> {
        fcx.put(self, ReaderCount(top.num_docs()?));
        Ok(())
    }
}

/// `MaxDocValueSource`: the reader's `maxDoc()`, for every document.
#[derive(Debug, Clone, Copy, Default)]
pub struct MaxDocValueSource;

impl MaxDocValueSource {
    pub fn new() -> Self {
        Self
    }

    /// `name()`.
    pub fn name(&self) -> &'static str {
        "maxdoc"
    }
}

impl ValueSource for MaxDocValueSource {
    fn get_values<'a>(
        &self,
        fcx: &FunctionContext,
        _leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        let n = fcx
            .get::<ReaderCount, _>(self)
            .ok_or_else(|| not_weighted(&self.description()))?;
        Ok(const_int(n.0, self.description()))
    }
    fn description(&self) -> String {
        format!("{}()", self.name())
    }
    fn create_weight(&self, fcx: &mut FunctionContext, top: &TopLevel<'_>) -> Result<()> {
        fcx.put(self, ReaderCount(top.max_doc()?));
        Ok(())
    }
}

/// `NormValueSource`: the similarity's score of a single occurrence under
/// the document's norm (`1` without one) with every statistic `1` -- the
/// length normalization alone (a `TFIDFSimilarity` only).
#[derive(Debug, Clone)]
pub struct NormValueSource {
    field: String,
}

impl NormValueSource {
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
        }
    }

    /// `name()`.
    pub fn name(&self) -> &'static str {
        "norm"
    }
}

struct NormValues<'a> {
    norms: Option<Box<dyn NumericDocValues + 'a>>,
    scorer: std::sync::Arc<dyn SimScorer>,
    last_doc: i32,
    description: String,
}

impl FloatDocValues for NormValues<'_> {
    fn description(&self) -> String {
        self.description.clone()
    }
    fn float_val(&mut self, doc: i32) -> Result<f32> {
        if doc < self.last_doc {
            // Java's `AssertionError`.
            return Err(Error::IllegalState(format!(
                "docs out of order: lastDocID={} docID={doc}",
                self.last_doc
            )));
        }
        self.last_doc = doc;
        let mut norm = 1i64;
        if let Some(n) = &mut self.norms {
            if n.advance_exact(doc)? {
                norm = n.long_value();
            }
        }
        Ok(self.scorer.score(1.0, norm))
    }
}

impl ValueSource for NormValueSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        let sim = tfidf(leaf, &self.field)?;
        let stats = |e: String| Error::IllegalArgument(e);
        let scorer = sim.scorer(
            &self.field,
            1.0,
            &CollectionStatistics::new(1, 1, 1, 1).map_err(stats)?,
            &[TermStatistics::new(1, 1).map_err(stats)?],
        );
        let norms = crate::reader::LeafReader::norm_values(leaf.reader()?, &self.field)?;
        Ok(Box::new(Float(NormValues {
            norms,
            scorer,
            last_doc: -1,
            description: self.description(),
        })))
    }
    fn description(&self) -> String {
        format!("{}({})", self.name(), self.field)
    }
}
