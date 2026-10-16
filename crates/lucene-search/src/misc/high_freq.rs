//! `HighFreqTerms` and `TermStats`: the top `n` terms of an index by document
//! frequency, or by total term frequency with `-t`.

use std::cmp::Ordering;

use crate::classification::java_string_cmp;
use crate::java_priority_queue::JavaPriorityQueue;
use crate::multi_terms::{indexed_fields, MultiTerms};
use crate::reader::{IndexReader, TermsEnum};
use crate::{Error, Result};

/// `HighFreqTerms.DEFAULT_NUMTERMS`.
pub const DEFAULT_NUMTERMS: usize = 100;

/// `TermStats`: a term with its field and statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TermStats {
    /// `termtext`: the term's bytes (`BytesRef.deepCopyOf`).
    pub termtext: Vec<u8>,
    pub field: String,
    /// `docFreq`.
    pub doc_freq: i32,
    /// `totalTermFreq`.
    pub total_term_freq: i64,
}

impl TermStats {
    /// `new TermStats(field, termtext, df, tf)`.
    pub fn new(field: &str, termtext: &[u8], doc_freq: i32, total_term_freq: i64) -> Self {
        Self {
            termtext: termtext.to_vec(),
            field: field.to_string(),
            doc_freq,
            total_term_freq,
        }
    }

    /// `getTermText()`: the term's bytes as a Java `String`, lossily.
    pub fn term_text(&self) -> String {
        String::from_utf8_lossy(&self.termtext).into_owned()
    }
}

/// The `Comparator<TermStats>` `HighFreqTerms` ranks by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TermOrder {
    /// `DocFreqComparator`: by `docFreq`, the default.
    DocFreq,
    /// `TotalTermFreqComparator`: by `totalTermFreq`, with `-t`.
    TotalTermFreq,
}

impl TermOrder {
    /// `compare(a, b)`: the frequency, then the field (`String.compareTo`),
    /// then the term (`BytesRef.compareTo`, unsigned bytes).
    pub fn compare(self, a: &TermStats, b: &TermStats) -> Ordering {
        let by_freq = match self {
            TermOrder::DocFreq => i64::from(a.doc_freq).cmp(&i64::from(b.doc_freq)),
            TermOrder::TotalTermFreq => a.total_term_freq.cmp(&b.total_term_freq),
        };
        by_freq
            .then_with(|| java_string_cmp(&a.field, &b.field))
            .then_with(|| a.termtext.cmp(&b.termtext))
    }
}

/// `HighFreqTerms.getHighFreqTerms(reader, numTerms, field, comparator)`: the
/// `num_terms` greatest terms of `field`, or of every indexed field when
/// `field` is `None`, highest first.
///
/// # Errors
/// Java's `RuntimeException`s: the named field has no terms, or the index has
/// no indexed fields. And what reading the index reports.
pub fn get_high_freq_terms<R: IndexReader + ?Sized>(
    reader: &R,
    num_terms: usize,
    field: Option<&str>,
    order: TermOrder,
) -> Result<Vec<TermStats>> {
    let mut tiq: JavaPriorityQueue<TermStats, _> =
        JavaPriorityQueue::new(num_terms, move |a: &TermStats, b: &TermStats| {
            order.compare(a, b) == Ordering::Less
        });
    if let Some(field) = field {
        let terms = MultiTerms::get_terms(reader, field)?
            .ok_or_else(|| Error::IllegalState(format!("field {field} not found")))?;
        fill(&mut tiq, field, terms.iterator()?.as_mut())?;
    } else {
        let fields = indexed_fields(reader);
        if fields.is_empty() {
            return Err(Error::IllegalState("no fields found for this index".into()));
        }
        for name in &fields {
            if let Some(terms) = MultiTerms::get_terms(reader, name)? {
                fill(&mut tiq, name, terms.iterator()?.as_mut())?;
            }
        }
    }
    // Highest first: the queue pops least first, so fill from the back.
    let mut result: Vec<TermStats> = Vec::with_capacity(tiq.size());
    while let Some(t) = tiq.pop() {
        result.push(t);
    }
    result.reverse();
    Ok(result)
}

/// `TermStatsQueue.fill(field, termsEnum)`: every term of `terms`, each
/// inserted with its statistics, keeping the queue's `size` best.
fn fill(
    tiq: &mut JavaPriorityQueue<TermStats, impl Fn(&TermStats, &TermStats) -> bool>,
    field: &str,
    terms: &mut dyn TermsEnum,
) -> Result<()> {
    while let Some(term) = terms.next()? {
        let term = term.to_vec();
        let doc_freq = terms.doc_freq()?;
        let total_term_freq = terms.total_term_freq()?;
        tiq.insert_with_overflow(TermStats::new(field, &term, doc_freq, total_term_freq));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_is_frequency_then_field_then_unsigned_term() {
        let a = TermStats::new("f", b"x", 2, 9);
        let b = TermStats::new("f", b"y", 3, 1);
        assert_eq!(TermOrder::DocFreq.compare(&a, &b), Ordering::Less);
        assert_eq!(TermOrder::TotalTermFreq.compare(&a, &b), Ordering::Greater);
        // A tie on frequency: the field, then the term's unsigned bytes.
        let c = TermStats::new("f", &[0xff], 2, 9);
        assert_eq!(TermOrder::DocFreq.compare(&a, &c), Ordering::Less);
        let d = TermStats::new("g", b"a", 2, 9);
        assert_eq!(TermOrder::DocFreq.compare(&a, &d), Ordering::Less);
    }

    #[test]
    fn term_text_is_lossy_like_utf8_to_string() {
        let t = TermStats::new("f", &[b'a', 0xff], 1, 1);
        assert_eq!(t.term_text(), "a\u{fffd}");
    }
}
