//! `BooleanPerceptronClassifier`: a perceptron over term weights, trained on
//! the index's documents through their term vectors.

use std::collections::BTreeMap;
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_util::strict_math::exp;

use super::{analyze, term_vector, ClassificationResult, Classifier, JavaString, TrainingIndex};
use crate::multi_terms::MultiTerms;
use crate::query::{BooleanQuery, Clause};
use crate::{Error, Result};

/// `BooleanPerceptronClassifier`: a document is `true` when the summed
/// weights of its terms reach the bias. The weights start as each term's
/// `totalTermFreq` and are corrected by every training document the
/// classifier gets wrong, by its term frequencies; the weights it classifies
/// with are a snapshot taken every `batch_size`th training document.
///
/// Java keeps the snapshot in an `FST<Long>` (rebuilt from the weights, a
/// map); here it is the same map of terms to truncated weights.
pub struct BooleanPerceptronClassifier {
    bias: f64,
    analyzer: Arc<Analyzer>,
    text_field: String,
    /// `fst`: term bytes to `weight.longValue()`.
    weights_snapshot: BTreeMap<Vec<u8>, i64>,
}

impl BooleanPerceptronClassifier {
    /// `new BooleanPerceptronClassifier(reader, analyzer, query, batchSize,
    /// bias, classFieldName, textFieldName)`: trains on the documents with a
    /// class (and matching `query`), in score order. A `bias` of `None` or
    /// zero is the average `totalTermFreq` per document.
    ///
    /// # Errors
    /// Java's `IOException`s -- a text field without terms or a training
    /// document without its term vector; a zero `batch_size` (Java's
    /// `ArithmeticException`); weights whose terms sort differently as
    /// strings and as bytes (Java's FST rejects them); what reading the index
    /// reports.
    pub fn new(
        index: TrainingIndex<'_, '_>,
        analyzer: Arc<Analyzer>,
        query: Option<Clause>,
        batch_size: i32,
        bias: Option<f64>,
        class_field: &str,
        text_field: &str,
    ) -> Result<Self> {
        let text_terms = MultiTerms::get_terms(index.reader, text_field)?.ok_or_else(|| {
            Error::IllegalState(format!(
                "term vectors need to be available for field {text_field}"
            ))
        })?;
        let bias = match bias {
            Some(b) if b != 0.0 => b,
            _ => {
                use crate::reader::IndexReader;
                let t = index.reader.sum_total_term_freq(text_field)? as f64
                    / f64::from(IndexReader::doc_count(index.reader, text_field)?);
                if t == -1.0 {
                    return Err(Error::IllegalState(format!(
                        "bias cannot be assigned since term vectors for field {text_field} do not exist"
                    )));
                }
                t
            }
        };
        let mut weights: BTreeMap<JavaString, f64> = BTreeMap::new();
        let mut te = text_terms.iterator()?;
        while let Some(t) = te.next()? {
            let s = String::from_utf8_lossy(t).into_owned();
            let ttf = te.total_term_freq()?;
            weights.insert(JavaString(s), ttf as f64);
        }
        let mut c = Self {
            bias,
            analyzer,
            text_field: text_field.to_string(),
            weights_snapshot: BTreeMap::new(),
        };
        c.weights_snapshot = snapshot(&weights)?;

        let mut q = BooleanQuery {
            must: vec![super::knn::any_class(class_field)],
            ..Default::default()
        };
        if let Some(f) = query {
            q.must.push(f);
        }
        let max_doc = usize::try_from(index.reader.max_doc()).unwrap_or(0);
        let hits = index.searcher(None)?.search(&q, max_doc.max(1))?;
        let mut batch_count = 0i32;
        for sd in &hits.score_docs {
            let text = index.stored_values(sd.doc, text_field)?.into_iter().next();
            let class = index.stored_values(sd.doc, class_field)?.into_iter().next();
            let (Some(text), Some(class)) = (text, class) else {
                continue;
            };
            let assigned = c.classify(&text)?.assigned_class;
            let correct = class.eq_ignore_ascii_case("true");
            // `Math.signum(correct.compareTo(assigned))`.
            let modifier = f64::from(i32::from(correct) - i32::from(assigned));
            if modifier != 0.0 {
                if batch_size == 0 {
                    return Err(Error::IllegalArgument("/ by zero".into()));
                }
                let update = batch_count.wrapping_rem(batch_size) == 0;
                c.update_weights(index, sd.doc, &mut weights, modifier, update)?;
            }
            batch_count = batch_count.wrapping_add(1);
        }
        Ok(c)
    }

    /// `updateWeights`: every term of the document's vector moved by
    /// `modifier` times its frequency from its snapshot weight (to no less
    /// than zero; a term missing from the snapshot to zero).
    fn update_weights(
        &mut self,
        index: TrainingIndex<'_, '_>,
        doc: i32,
        weights: &mut BTreeMap<JavaString, f64>,
        modifier: f64,
        update_snapshot: bool,
    ) -> Result<()> {
        let vector = term_vector(index.reader, doc, &self.text_field)?.ok_or_else(|| {
            Error::IllegalState(format!(
                "term vectors must be stored for field {}",
                self.text_field
            ))
        })?;
        for t in &vector.terms {
            let previous = self.weights_snapshot.get(&t.term);
            let term = String::from_utf8_lossy(&t.term).into_owned();
            let freq = i64::from(t.freq);
            let w = match previous {
                None => 0.0,
                Some(&p) => f64::max(0.0, p as f64 + modifier * freq as f64),
            };
            weights.insert(JavaString(term), w);
        }
        if update_snapshot {
            self.weights_snapshot = snapshot(weights)?;
        }
        Ok(())
    }

    fn classify(&self, text: &str) -> Result<ClassificationResult<bool>> {
        let mut output = 0i64;
        for token in analyze(&self.analyzer, &self.text_field, text)? {
            if let Some(&w) = self.weights_snapshot.get(token.as_bytes()) {
                output = output.wrapping_add(w);
            }
        }
        let score = 1.0 - exp(-(self.bias - output as f64).abs() / self.bias);
        Ok(ClassificationResult::new(output as f64 >= self.bias, score))
    }
}

/// `updateFST(weights)`: the weights as the FST stores them, keyed by UTF-8
/// bytes and truncated to `long`. The FST takes its inputs in byte order, so
/// a string order that disagrees is its "inputs are added out of order".
fn snapshot(weights: &BTreeMap<JavaString, f64>) -> Result<BTreeMap<Vec<u8>, i64>> {
    let mut out = BTreeMap::new();
    let mut last: Option<&[u8]> = None;
    for (k, &v) in weights {
        let bytes = k.0.as_bytes();
        if last.is_some_and(|l| l >= bytes) {
            return Err(Error::IllegalArgument(format!(
                "inputs are added out of order lastInput={:?} vs input={:?}",
                last.unwrap_or_default(),
                bytes
            )));
        }
        last = Some(bytes);
        out.insert(bytes.to_vec(), v as i64);
    }
    Ok(out)
}

impl Classifier<bool> for BooleanPerceptronClassifier {
    fn assign_class(&self, text: &str) -> Result<Option<ClassificationResult<bool>>> {
        self.classify(text).map(Some)
    }

    /// `getClasses`: Java's `null`.
    fn get_classes(&self, _text: &str) -> Result<Option<Vec<ClassificationResult<bool>>>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_truncates_and_rejects_string_order_unlike_byte_order() {
        let mut w = BTreeMap::new();
        w.insert(JavaString("a".into()), 2.9);
        w.insert(JavaString("b".into()), 0.0);
        let s = snapshot(&w).unwrap();
        assert_eq!(s.get(b"a".as_slice()), Some(&2));
        assert_eq!(s.get(b"b".as_slice()), Some(&0));
        // U+10000 sorts before U+FFFD as UTF-16 units, after it as bytes.
        let mut w = BTreeMap::new();
        w.insert(JavaString("\u{fffd}".into()), 1.0);
        w.insert(JavaString("\u{10000}".into()), 1.0);
        assert!(snapshot(&w).is_err());
    }
}
