//! `LuceneDictionary` and `HighFrequencyDictionary` (M13 T13.3e): a field's
//! terms, read off an index, as a [`Dictionary`].
//!
//! Departure from Java: both read the field's terms when
//! [`get_entry_iterator`](Dictionary::get_entry_iterator) is called and
//! replay them from memory (Java walks a live `TermsEnum`), so a failed read
//! surfaces there.

use std::collections::BTreeSet;

use crate::multi_terms::MultiTerms;
use crate::reader::IndexReader;
use crate::spell::Dictionary;
use crate::suggest::input::{InputIterator, InputIteratorWrapper};
use crate::Result;

/// `LuceneDictionary`: every term of a field, in term order, weight 1.
pub struct LuceneDictionary<'r, R: IndexReader + ?Sized> {
    reader: &'r R,
    field: String,
}

impl<'r, R: IndexReader + ?Sized> LuceneDictionary<'r, R> {
    /// `new LuceneDictionary(reader, field)`.
    pub fn new(reader: &'r R, field: impl Into<String>) -> Self {
        Self {
            reader,
            field: field.into(),
        }
    }
}

/// The field's terms with their document frequencies, in term order; empty
/// when no segment has the field.
fn field_terms<R: IndexReader + ?Sized>(reader: &R, field: &str) -> Result<Vec<(Vec<u8>, i32)>> {
    let mut out = Vec::new();
    if let Some(terms) = MultiTerms::get_terms(reader, field)? {
        let mut te = terms.iterator()?;
        while let Some(term) = te.next()? {
            let term = term.to_vec();
            out.push((term, te.doc_freq()?));
        }
    }
    Ok(out)
}

impl<R: IndexReader + ?Sized> Dictionary for LuceneDictionary<'_, R> {
    fn get_entry_iterator(&self) -> Result<Box<dyn InputIterator>> {
        // `InputIterator.EMPTY` for a missing field: an empty wrapper too
        let terms: Vec<Vec<u8>> = field_terms(self.reader, &self.field)?
            .into_iter()
            .map(|(t, _)| t)
            .collect();
        Ok(Box::new(InputIteratorWrapper::new(terms.into_iter())))
    }
}

/// `HighFrequencyDictionary`: the terms of a field found in at least
/// `thresh` of the reader's live documents, each weighted by its document
/// frequency (deleted documents counted, as `docFreq` counts them).
pub struct HighFrequencyDictionary<'r, R: IndexReader + ?Sized> {
    reader: &'r R,
    field: String,
    thresh: f32,
}

impl<'r, R: IndexReader + ?Sized> HighFrequencyDictionary<'r, R> {
    /// `new HighFrequencyDictionary(reader, field, thresh)`.
    pub fn new(reader: &'r R, field: impl Into<String>, thresh: f32) -> Self {
        Self {
            reader,
            field: field.into(),
            thresh,
        }
    }
}

/// `HighFrequencyDictionary.HighFrequencyIterator`, replayed.
struct HighFrequencyIterator {
    entries: std::vec::IntoIter<(Vec<u8>, i32)>,
    freq: i64,
}

impl InputIterator for HighFrequencyIterator {
    fn next(&mut self) -> Option<Vec<u8>> {
        let (term, freq) = self.entries.next()?;
        self.freq = i64::from(freq);
        Some(term)
    }
    fn weight(&self) -> i64 {
        self.freq
    }
    fn payload(&self) -> Option<&[u8]> {
        None
    }
    fn has_payloads(&self) -> bool {
        false
    }
    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
        None
    }
    fn has_contexts(&self) -> bool {
        false
    }
}

impl<R: IndexReader + ?Sized> Dictionary for HighFrequencyDictionary<'_, R> {
    fn get_entry_iterator(&self) -> Result<Box<dyn InputIterator>> {
        // `(int) (thresh * (float) reader.numDocs())`
        let min_num_docs = (self.thresh * self.reader.num_docs() as f32) as i32;
        let entries: Vec<(Vec<u8>, i32)> = field_terms(self.reader, &self.field)?
            .into_iter()
            .filter(|(_, df)| *df >= min_num_docs)
            .collect();
        Ok(Box::new(HighFrequencyIterator {
            entries: entries.into_iter(),
            freq: 0,
        }))
    }
}
