#![cfg(test)]
use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

fn expr(src: &str) -> Arc<Expression> {
    Arc::new(compile(src).unwrap())
}

fn message(r: lucene_search::Result<()>) -> String {
    r.unwrap_err().to_string()
}

#[test]
fn simple_bindings_resolve_sources_and_expressions() {
    let mut b = SimpleBindings::new();
    b.add_source("a", constant(2.0));
    b.add_expression("b", expr("a * 3"));
    assert_eq!(
        b.double_values_source("a").unwrap().describe(),
        "constant(2.0)"
    );
    assert_eq!(
        b.double_values_source("b").unwrap().describe(),
        "expr(a * 3)"
    );
    assert_eq!(
        b.double_values_source("zzz").err().unwrap().to_string(),
        "Invalid reference 'zzz'"
    );
    b.validate().unwrap();
}

#[test]
fn expression_over_an_unbound_variable_is_refused() {
    let b = SimpleBindings::new();
    let e = expr("missing + 1");
    assert_eq!(
        e.double_values_source(&b).err().unwrap().to_string(),
        "Invalid reference 'missing'"
    );
}

#[test]
fn validate_reports_cycles_and_dangling_references() {
    let mut b = SimpleBindings::new();
    b.add_expression("x", expr("y + 1"));
    b.add_expression("y", expr("x + 1"));
    assert_eq!(
        message(b.validate()),
        "Recursion error: Cycle detected [x, y]->x"
    );
    // never validated: the cycle is still an error, not a stack overflow
    assert_eq!(
        message(b.double_values_source("y").map(|_| ())),
        "Recursion error: Cycle detected [y, x]->y"
    );
    let source = ExpressionValueSource::new(&b, expr("x * 2"));
    assert!(message(source.map(|_| ())).starts_with("Recursion error"));

    let mut b = SimpleBindings::new();
    b.add_expression("x", expr("nowhere"));
    assert_eq!(message(b.validate()), "Invalid reference 'nowhere'");

    let mut b = SimpleBindings::new();
    b.add_source("leaf", constant(1.0));
    b.add_expression("x", expr("leaf + leaf"));
    b.add_expression("y", expr("x + leaf"));
    b.validate().unwrap();
}

#[test]
fn a_source_needs_scores_when_a_variable_does() {
    let mut b = SimpleBindings::new();
    b.add_source("s", lucene_search::values_source::scores());
    b.add_source("c", constant(1.0));
    let with = ExpressionValueSource::new(&b, expr("s + c")).unwrap();
    let without = ExpressionValueSource::new(&b, expr("c * 2")).unwrap();
    assert!(with.needs_scores());
    assert!(!without.needs_scores());
    assert_eq!(with.describe(), "expr(s + c)");
    let caching = ExpressionValueSource::caching(&b, expr("c")).unwrap();
    assert!(caching
        .as_any()
        .and_then(|a| a.downcast_ref::<ExpressionValueSource>())
        .is_some());
}

/// Even documents have the value `doc`, odd ones none; each read of a
/// value is counted, and the first read at `fail_at` fails.
struct Probe {
    reads: Arc<AtomicUsize>,
    fail_at: i32,
    failed: Arc<AtomicBool>,
}

struct ProbeValues {
    reads: Arc<AtomicUsize>,
    fail_at: i32,
    failed: Arc<AtomicBool>,
    doc: i32,
}

impl DoubleValues for ProbeValues {
    fn advance_exact(&mut self, doc: i32) -> lucene_search::Result<bool> {
        self.doc = doc;
        Ok(doc % 2 == 0)
    }
    fn double_value(&mut self) -> lucene_search::Result<f64> {
        if self.doc == self.fail_at && !self.failed.swap(true, Relaxed) {
            return Err(lucene_search::Error::IllegalState("probe".into()));
        }
        self.reads.fetch_add(1, Relaxed);
        Ok(f64::from(self.doc))
    }
    fn batch_capable(&self) -> bool {
        true
    }
}

/// The scores a batch is handed, or `-doc` read one document at a time.
struct BatchScores(i32);

impl DoubleValues for BatchScores {
    fn advance_exact(&mut self, doc: i32) -> lucene_search::Result<bool> {
        self.0 = doc;
        Ok(true)
    }
    fn double_value(&mut self) -> lucene_search::Result<f64> {
        Ok(f64::from(-self.0))
    }
    fn batch_capable(&self) -> bool {
        true
    }
    fn fill_batch(
        &mut self,
        docs: &[i32],
        scores: &[f32],
        out: &mut [f64],
        has: &mut [bool],
    ) -> lucene_search::Result<()> {
        for ((o, h), &s) in out.iter_mut().zip(has.iter_mut()).zip(scores) {
            *o = f64::from(s);
            *h = true;
        }
        self.0 = docs.last().copied().unwrap_or(self.0);
        Ok(())
    }
}

#[test]
fn a_batch_reads_each_variable_once_per_document_and_matches_per_document_values() {
    let reads = Arc::new(AtomicUsize::new(0));
    let failed = Arc::new(AtomicBool::new(false));
    let mut b = SimpleBindings::new();
    b.add_source(
        "p",
        Arc::new(Probe {
            reads: reads.clone(),
            fail_at: -5,
            failed: failed.clone(),
        }),
    );
    b.add_source("_score", lucene_search::values_source::scores());
    b.add_source("c", constant(3.0));
    // a caching nested source, whose `p` a caching outer one shares
    let twice = ExpressionValueSource::caching(&b, expr("p + p + _score")).unwrap();
    b.add_source("twice", Arc::new(twice));
    let searcher = lucene_search::index_searcher::IndexSearcher::new(&[], &[]).unwrap();
    let ctx = ValuesContext::new(&searcher);
    let scores = || Some(Box::new(BatchScores(-1)) as BoxDoubleValues<'_>);
    let docs: Vec<i32> = (0..40).collect();
    for caching in [false, true] {
        let outer = expr("p * p + twice + (p ? _score : c) + c");
        let source = if caching {
            ExpressionValueSource::caching(&b, outer).unwrap()
        } else {
            ExpressionValueSource::new(&b, outer).unwrap()
        };
        // per document, with the scores `-doc`
        let mut one = source.get_values(&ctx, 0, scores()).unwrap();
        let want: Vec<u64> = docs
            .iter()
            .map(|&d| {
                assert!(one.advance_exact(d).unwrap());
                one.double_value().unwrap().to_bits()
            })
            .collect();
        reads.store(0, Relaxed);
        let mut v = source.get_values(&ctx, 0, scores()).unwrap();
        assert!(v.batch_capable());
        let mut got = Vec::new();
        for chunk in docs.chunks(6) {
            let s: Vec<f32> = chunk.iter().map(|&d| -d as f32).collect();
            let (mut out, mut has) = (vec![f64::NAN; 6], vec![false; 6]);
            v.fill_batch(chunk, &s, &mut out, &mut has).unwrap();
            assert!(has[..chunk.len()].iter().all(|&h| h));
            got.extend(out[..chunk.len()].iter().map(|x| x.to_bits()));
        }
        assert_eq!(got, want, "caching={caching}");
        // the last document's value is current after the batch
        assert!(v.advance_exact(39).unwrap());
        assert_eq!(v.double_value().unwrap().to_bits(), want[39]);
        // `p` read once per even document by each expression naming
        // it, or once in all when a caching source shares it
        let per_doc = if caching { 1 } else { 2 };
        assert_eq!(reads.load(Relaxed), 20 * per_doc, "caching={caching}");
    }

    // An expression over no variable, and a failing read failing the batch.
    let none = ExpressionValueSource::new(&b, expr("1 + 2")).unwrap();
    let mut v = none.get_values(&ctx, 0, None).unwrap();
    let (mut out, mut has) = ([0.0; 3], [false; 3]);
    v.fill_batch(&[0, 1, 2], &[0.0; 3], &mut out, &mut has)
        .unwrap();
    assert_eq!((out, has), ([3.0; 3], [true; 3]));
    let mut f = SimpleBindings::new();
    f.add_source(
        "p",
        Arc::new(Probe {
            reads,
            fail_at: 4,
            failed,
        }),
    );
    let source = ExpressionValueSource::new(&f, expr("p + 1")).unwrap();
    let mut v = source.get_values(&ctx, 0, None).unwrap();
    let (mut out, mut has) = ([0.0; 8], [false; 8]);
    let docs: Vec<i32> = (0..8).collect();
    let e = v.fill_batch(&docs, &[0.0; 8], &mut out, &mut has);
    assert!(e.unwrap_err().to_string().contains("probe"));
}

impl DoubleValuesSource for Probe {
    fn get_values<'c>(
        &self,
        _ctx: &ValuesContext<'c>,
        _leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> lucene_search::Result<BoxDoubleValues<'c>> {
        Ok(Box::new(ProbeValues {
            reads: self.reads.clone(),
            fail_at: self.fail_at,
            failed: self.failed.clone(),
            doc: -1,
        }))
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        true
    }
    fn describe(&self) -> String {
        "probe".into()
    }
}

#[test]
fn variables_read_once_per_doc_zero_when_missing_and_retried_after_an_error() {
    let reads = Arc::new(AtomicUsize::new(0));
    let failed = Arc::new(AtomicBool::new(false));
    let mut b = SimpleBindings::new();
    b.add_source(
        "p",
        Arc::new(Probe {
            reads: reads.clone(),
            fail_at: 4,
            failed: failed.clone(),
        }),
    );
    b.add_expression("twice", expr("p + p"));
    let searcher = lucene_search::index_searcher::IndexSearcher::new(&[], &[]).unwrap();
    let ctx = ValuesContext::new(&searcher);
    for caching in [false, true] {
        reads.store(0, Relaxed);
        failed.store(false, Relaxed);
        let outer = expr("p * p + twice + p");
        let source = if caching {
            ExpressionValueSource::caching(&b, outer).unwrap()
        } else {
            ExpressionValueSource::new(&b, outer).unwrap()
        };
        let mut v = source.get_values(&ctx, 0, None).unwrap();
        let mut got = Vec::new();
        for doc in 0..6 {
            assert!(v.advance_exact(doc).unwrap());
            match v.double_value() {
                Ok(x) => got.push(x),
                Err(e) => {
                    assert_eq!(doc, 4);
                    assert!(e.to_string().contains("probe"), "{e}");
                    // the same document read again succeeds
                    got.push(v.double_value().unwrap());
                }
            }
            // a second read of the document is the evaluated value
            assert_eq!(v.double_value().unwrap(), got[got.len() - 1]);
        }
        let want: Vec<f64> = (0..6)
            .map(|d| {
                if d % 2 == 0 {
                    f64::from(d * d + 2 * d + d)
                } else {
                    0.0
                }
            })
            .collect();
        assert_eq!(got, want, "caching={caching}");
        // per even document, `p` is read once for the outer expression
        // and once for the nested one (whose values a caching source
        // shares, but each variable keeps its own per-document value)
        assert_eq!(reads.load(Relaxed), 6, "caching={caching}");
    }
}
