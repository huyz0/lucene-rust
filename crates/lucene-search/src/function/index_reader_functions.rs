//! `IndexReaderFunctions`: values sources of the reader's statistics --
//! `docFreq`, `maxDoc`, `numDocs`, `numDeletedDocs`, `sumTotalTermFreq`,
//! `termFreq`, `totalTermFreq`, `sumDocFreq`, `docCount`.
//!
//! All but `termFreq` are one number for the whole reader: unusable until
//! rewritten against a searcher ([`DoubleValuesSource::rewrite`], which
//! the function queries run), when they become that number for every
//! document.

use std::sync::Arc;

use lucene_codecs::postings::{LazyDocsCursor, PostingsFlags};

use super::TopLevel;
use crate::values_source::{
    BoxDoubleValues, BoxLongValues, DoubleValues, DoubleValuesSource, EmptyDoubleValues,
    LongValues, LongValuesSource, ValuesContext,
};
use crate::{Error, Result};

/// `Term.toString()`.
fn term_string(field: &str, term: &[u8]) -> String {
    format!("{field}:{}", String::from_utf8_lossy(term))
}

/// `UnsupportedOperationException("IndexReaderFunction must be rewritten
/// before use")`.
fn not_rewritten() -> Error {
    Error::Unsupported("IndexReaderFunction must be rewritten before use".into())
}

/// A function of the top-level reader.
type ReaderFn = dyn Fn(&TopLevel<'_>) -> Result<f64> + Send + Sync;

/// `IndexReaderDoubleValuesSource`: a reader statistic, rewritten into a
/// constant.
struct IndexReaderDoubleValuesSource {
    func: Arc<ReaderFn>,
    description: String,
}

impl DoubleValuesSource for IndexReaderDoubleValuesSource {
    fn get_values<'c>(
        &self,
        _ctx: &ValuesContext<'c>,
        _leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        Err(not_rewritten())
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        false
    }
    fn describe(&self) -> String {
        self.description.clone()
    }
    fn rewrite(&self, top: &TopLevel<'_>) -> Result<Option<Arc<dyn DoubleValuesSource>>> {
        Ok(Some(Arc::new(NoCacheConstantDoubleValuesSource {
            value: (self.func)(top)?,
            description: self.description.clone(),
        })))
    }
}

/// `NoCacheConstantDoubleValuesSource`: the rewritten statistic.
struct NoCacheConstantDoubleValuesSource {
    value: f64,
    description: String,
}

impl DoubleValuesSource for NoCacheConstantDoubleValuesSource {
    fn get_values<'c>(
        &self,
        _ctx: &ValuesContext<'c>,
        _leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        Ok(Box::new(crate::values_source::ConstantDoubleValues(
            self.value,
        )))
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        false
    }
    fn describe(&self) -> String {
        self.description.clone()
    }
}

fn reader_source(func: Arc<ReaderFn>, description: String) -> Arc<dyn DoubleValuesSource> {
    Arc::new(IndexReaderDoubleValuesSource { func, description })
}

/// `IndexReaderFunctions.docFreq(term)`.
pub fn doc_freq(field: &str, term: &[u8]) -> Arc<dyn DoubleValuesSource> {
    let (f, t) = (field.to_string(), term.to_vec());
    reader_source(
        Arc::new(move |top: &TopLevel<'_>| Ok(f64::from(top.doc_freq(&f, &t)?))),
        format!("docFreq({})", term_string(field, term)),
    )
}

/// `IndexReaderFunctions.maxDoc()`.
pub fn max_doc() -> Arc<dyn DoubleValuesSource> {
    reader_source(
        Arc::new(|top: &TopLevel<'_>| Ok(f64::from(top.max_doc()?))),
        "maxDoc()".into(),
    )
}

/// `IndexReaderFunctions.numDocs()`.
pub fn num_docs() -> Arc<dyn DoubleValuesSource> {
    reader_source(
        Arc::new(|top: &TopLevel<'_>| Ok(f64::from(top.num_docs()?))),
        "numDocs()".into(),
    )
}

/// `IndexReaderFunctions.numDeletedDocs()`.
pub fn num_deleted_docs() -> Arc<dyn DoubleValuesSource> {
    reader_source(
        Arc::new(|top: &TopLevel<'_>| {
            Ok(f64::from(top.max_doc()?.saturating_sub(top.num_docs()?)))
        }),
        "numDeletedDocs()".into(),
    )
}

/// `IndexReaderFunctions.totalTermFreq(term)`: the leaves' summed.
pub fn total_term_freq(field: &str, term: &[u8]) -> Arc<dyn DoubleValuesSource> {
    let (f, t) = (field.to_string(), term.to_vec());
    reader_source(
        Arc::new(move |top: &TopLevel<'_>| {
            let mut sum = 0i64;
            for leaf in &top.leaves {
                if let Some(ft) = leaf.fields.field(&f) {
                    if let Some(s) = ft.try_seek_exact(&t)? {
                        sum = sum.wrapping_add(s.total_term_freq);
                    }
                }
            }
            Ok(sum as f64)
        }),
        format!("totalTermFreq({})", term_string(field, term)),
    )
}

/// `IndexReaderFunctions.sumDocFreq(field)`.
pub fn sum_doc_freq(field: &str) -> Arc<dyn DoubleValuesSource> {
    let f = field.to_string();
    reader_source(
        Arc::new(move |top: &TopLevel<'_>| {
            Ok(top
                .leaves
                .iter()
                .filter_map(|l| l.fields.field(&f))
                .fold(0i64, |s, t| s.wrapping_add(t.sum_doc_freq)) as f64)
        }),
        format!("sumDocFreq({field})"),
    )
}

/// `IndexReaderFunctions.docCount(field)`.
pub fn doc_count(field: &str) -> Arc<dyn DoubleValuesSource> {
    let f = field.to_string();
    reader_source(
        Arc::new(move |top: &TopLevel<'_>| {
            Ok(top
                .leaves
                .iter()
                .filter_map(|l| l.fields.field(&f))
                .fold(0i32, |s, t| s.wrapping_add(t.doc_count))
                .into())
        }),
        format!("docCount({field})"),
    )
}

/// `SumTotalTermFreqValuesSource`: the field's reader-wide
/// `sumTotalTermFreq`, rewritten into a constant.
struct SumTotalTermFreqValuesSource {
    field: String,
}

impl LongValuesSource for SumTotalTermFreqValuesSource {
    fn get_values<'c>(
        &self,
        _ctx: &ValuesContext<'c>,
        _leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxLongValues<'c>> {
        Err(not_rewritten())
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        false
    }
    fn describe(&self) -> String {
        format!("sumTotalTermFreq({})", self.field)
    }
    fn rewrite(&self, top: &TopLevel<'_>) -> Result<Option<Arc<dyn LongValuesSource>>> {
        let value = top
            .leaves
            .iter()
            .filter_map(|l| l.fields.field(&self.field))
            .fold(0i64, |s, t| s.wrapping_add(t.sum_total_term_freq));
        Ok(Some(Arc::new(NoCacheConstantLongValuesSource {
            value,
            description: self.describe(),
        })))
    }
}

/// `NoCacheConstantLongValuesSource`.
struct NoCacheConstantLongValuesSource {
    value: i64,
    description: String,
}

struct ConstLong(i64);

impl LongValues for ConstLong {
    fn advance_exact(&mut self, _doc: i32) -> Result<bool> {
        Ok(true)
    }
    fn long_value(&mut self) -> Result<i64> {
        Ok(self.0)
    }
}

impl LongValuesSource for NoCacheConstantLongValuesSource {
    fn get_values<'c>(
        &self,
        _ctx: &ValuesContext<'c>,
        _leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxLongValues<'c>> {
        Ok(Box::new(ConstLong(self.value)))
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        false
    }
    fn describe(&self) -> String {
        self.description.clone()
    }
}

/// `IndexReaderFunctions.sumTotalTermFreq(field)`.
pub fn sum_total_term_freq(field: &str) -> Arc<dyn LongValuesSource> {
    Arc::new(SumTotalTermFreqValuesSource {
        field: field.to_string(),
    })
}

/// `TermFreqDoubleValuesSource`: the document's frequency of the term.
struct TermFreqDoubleValuesSource {
    field: String,
    term: Vec<u8>,
}

struct TermFreqValues<'c> {
    pe: LazyDocsCursor<'c>,
}

impl DoubleValues for TermFreqValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        if self.pe.doc_id() > doc {
            return Ok(false);
        }
        Ok(self.pe.doc_id() == doc || self.pe.advance(doc)? == doc)
    }
    fn double_value(&mut self) -> Result<f64> {
        Ok(f64::from(self.pe.freq().unwrap_or(1)))
    }
}

impl DoubleValuesSource for TermFreqDoubleValuesSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        let (fields, doc_in) = match ctx.exec_leaf() {
            Some(lc) => (lc.fields, lc.doc_in),
            None => {
                let seg = ctx.searcher()?.segments().get(leaf).ok_or_else(|| {
                    Error::IllegalArgument(format!("leaf {leaf} is not a segment"))
                })?;
                (seg.fields, seg.doc_in)
            }
        };
        let Some(ft) = fields.field(&self.field) else {
            return Ok(Box::new(EmptyDoubleValues));
        };
        let Some(seeked) = ft.seek_term_state(&self.term)? else {
            return Ok(Box::new(EmptyDoubleValues));
        };
        let doc_in = doc_in.ok_or_else(|| {
            Error::IllegalState("termFreq: the segment's postings are not open".into())
        })?;
        Ok(Box::new(TermFreqValues {
            pe: ft.lazy_postings_for(&seeked, doc_in, PostingsFlags::Freqs)?,
        }))
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        true
    }
    fn describe(&self) -> String {
        format!("termFreq({})", term_string(&self.field, &self.term))
    }
}

/// `IndexReaderFunctions.termFreq(term)`.
pub fn term_freq(field: &str, term: &[u8]) -> Arc<dyn DoubleValuesSource> {
    Arc::new(TermFreqDoubleValuesSource {
        field: field.to_string(),
        term: term.to_vec(),
    })
}
