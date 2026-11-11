//! The pairs (`benchmarks/micro/java/ExpressionsMicro.java` is the
//! Java twin): `expr_compile`, every expression of
//! `fixtures/corpus/expressions.txt` that compiles with the default
//! functions; `expr_<shape>`, one expression evaluated over 1,024 documents'
//! values of `a`, `b` and `c`; `expr_<shape>_batch`, the same over batches
//! of 128 documents' columns (what `ExpressionValueSource::fill_batch` hands
//! the expression), against Java's only path, one document at a time.

use std::hint::black_box;
use std::time::Duration;

use lucene_expressions::{compile, set_default_backend, Backend};
use lucene_search::values_source::{BoxDoubleValues, DoubleValues};

use super::measure;

const EVAL: &[(&str, &str)] = &[
    ("expr_arith", "a * 2 + b - c / 3"),
    ("expr_math", "sqrt(abs(b)) + ln(1 + abs(a)) * cos(c)"),
    ("expr_cond", "a > 100 ? b : (c < 0 && b != 0 ? -a : c)"),
    ("expr_bits", "((a << 2) | 3) ^ (c >>> 1)"),
    ("expr_haversin", "haversin(a / 10, b, c / 10, b + 1)"),
    ("expr_funcs", "max(a, b) * min(c, 1) + pow(b, 2) + floor(c)"),
];
const DOCS: i32 = 1024;
const BATCH: usize = 128;

/// The value of variable `k` for `doc`, as the Java side computes it.
fn value(doc: i32, k: i64) -> f64 {
    ((i64::from(doc) * 7919 + k * 104_729) % 401 - 200) as f64 + f64::from(doc % 7) / 8.0
}

struct ArrayValues {
    values: Vec<f64>,
    doc: usize,
}

impl DoubleValues for ArrayValues {
    fn advance_exact(&mut self, doc: i32) -> lucene_search::Result<bool> {
        self.doc = doc as usize;
        Ok(true)
    }

    fn double_value(&mut self) -> lucene_search::Result<f64> {
        Ok(self.values[self.doc])
    }
}

pub fn bench_expressions(w: Duration, m: Duration) {
    let texts: Vec<String> = std::fs::read_to_string("fixtures/corpus/expressions.txt")
        .unwrap()
        .lines()
        .filter(|l| compile(l).is_ok())
        .map(str::to_string)
        .collect();
    measure("expr_compile", w, m, || {
        for t in &texts {
            black_box(compile(t).unwrap().variables.len());
        }
        texts.len() as u64
    });
    let arrays: Vec<Vec<f64>> = (0..3)
        .map(|k| (0..DOCS).map(|doc| value(doc, k)).collect())
        .collect();
    // BENCH_EXPR_BACKEND=closure times the closure backend (set after
    // `expr_compile`, which stays the default backend's compile time)
    if std::env::var("BENCH_EXPR_BACKEND").as_deref() == Ok("closure") {
        set_default_backend(Backend::Closure);
    }
    for (name, text) in EVAL {
        let e = compile(text).unwrap();
        let mut values: Vec<BoxDoubleValues<'static>> = e
            .variables
            .iter()
            .map(|v| {
                let k = i64::from(v.as_bytes()[0] - b'a');
                Box::new(ArrayValues {
                    values: (0..DOCS).map(|doc| value(doc, k)).collect(),
                    doc: 0,
                }) as BoxDoubleValues<'static>
            })
            .collect();
        measure(name, w, m, || {
            let mut sum = 0.0;
            for doc in 0..DOCS {
                for v in &mut values {
                    v.advance_exact(doc).unwrap();
                }
                sum += e.evaluate(&mut values).unwrap();
            }
            black_box(sum);
            DOCS as u64
        });
        let columns: Vec<&[f64]> = e
            .variables
            .iter()
            .map(|v| {
                let k = usize::from(v.as_bytes()[0] - b'a');
                &arrays[k][..]
            })
            .collect();
        let mut cols = vec![0.0; columns.len() * BATCH];
        let mut out = vec![0.0; BATCH];
        measure(&format!("{name}_batch"), w, m, || {
            let mut sum = 0.0;
            for start in (0..DOCS as usize).step_by(BATCH) {
                for (c, column) in cols.chunks_exact_mut(BATCH).zip(&columns) {
                    c.copy_from_slice(&column[start..start + BATCH]);
                }
                e.evaluate_columns(&cols, &mut out).unwrap();
                sum += out.iter().sum::<f64>();
            }
            black_box(sum);
            DOCS as u64
        });
    }
}
