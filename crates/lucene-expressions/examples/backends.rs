//! Times each [`Backend`] on the micro bench's expressions: compile latency,
//! one document at a time (as `ExpressionsMicro.java` evaluates), and a
//! batch at a time over columns. `cargo run --release --example backends
//! `.

use std::hint::black_box;
use std::time::{Duration, Instant};

use lucene_expressions::{compile, Backend};
use lucene_search::values_source::{BoxDoubleValues, DoubleValues};

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

/// Nanoseconds per unit, the median of 7 runs of `f` (which returns units).
fn time(mut f: impl FnMut() -> u64) -> f64 {
    let warm = Instant::now();
    while warm.elapsed() < Duration::from_millis(300) {
        f();
    }
    let mut runs: Vec<f64> = (0..7)
        .map(|_| {
            let (t, mut units) = (Instant::now(), 0);
            while t.elapsed() < Duration::from_millis(150) {
                units += f();
            }
            t.elapsed().as_nanos() as f64 / units as f64
        })
        .collect();
    runs.sort_by(f64::total_cmp);
    runs[3]
}

fn rss_kb() -> u64 {
    let s = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    s.split_whitespace()
        .nth(1)
        .and_then(|p| p.parse::<u64>().ok())
        .unwrap_or(0)
        * 4
}

fn main() {
    let backends = [("closure", Backend::Closure), ("vector", Backend::Vector)];
    let texts: Vec<String> = std::fs::read_to_string("../../fixtures/corpus/expressions.txt")
        .unwrap()
        .lines()
        .filter(|l| compile(l).is_ok())
        .map(str::to_string)
        .collect();
    let parsed: Vec<_> = texts.iter().map(|t| compile(t).unwrap()).collect();
    println!("compile, {} expressions (us per expression):", texts.len());
    for (name, b) in &backends {
        let ns = time(|| {
            for e in &parsed {
                black_box(e.with_backend(*b));
            }
            parsed.len() as u64
        });
        println!("  {name:8} {:10.2}", ns / 1000.0);
    }
    for (name, b) in &backends {
        let before = rss_kb();
        let kept: Vec<_> = (0..4)
            .flat_map(|_| parsed.iter().map(|e| e.with_backend(*b)))
            .collect();
        println!(
            "  {name:8} resident memory {:.1} KB per compiled expression",
            (rss_kb() - before) as f64 / kept.len() as f64
        );
    }
    println!(
        "\ncase              mode   {}",
        backends
            .iter()
            .map(|b| format!("{:>9}", b.0))
            .collect::<String>()
    );
    for (case, text) in EVAL {
        let base = compile(text).unwrap();
        let cols: Vec<Vec<f64>> = base
            .variables
            .iter()
            .map(|v| {
                let k = i64::from(v.as_bytes()[0] - b'a');
                (0..DOCS).map(|doc| value(doc, k)).collect()
            })
            .collect();
        let mut doc_row = String::new();
        let mut batch_row = String::new();
        let mut reference: Option<(f64, f64)> = None;
        for (_, b) in &backends {
            let e = base.with_backend(*b);
            let mut values: Vec<BoxDoubleValues<'static>> = cols
                .iter()
                .map(|c| {
                    Box::new(ArrayValues {
                        values: c.clone(),
                        doc: 0,
                    }) as BoxDoubleValues<'static>
                })
                .collect();
            let mut sum1 = 0.0;
            let ns = time(|| {
                let mut sum = 0.0;
                for doc in 0..DOCS {
                    for v in &mut values {
                        v.advance_exact(doc).unwrap();
                    }
                    sum += e.evaluate(&mut values).unwrap();
                }
                sum1 = black_box(sum);
                DOCS as u64
            });
            doc_row += &format!("{ns:9.2}");
            // columns of BATCH documents, as ExpressionFunctionValues fills them
            let nv = cols.len();
            let blocks: Vec<Vec<f64>> = (0..DOCS as usize / BATCH)
                .map(|blk| {
                    let mut c = Vec::with_capacity(nv * BATCH);
                    for col in &cols {
                        c.extend_from_slice(&col[blk * BATCH..][..BATCH]);
                    }
                    c
                })
                .collect();
            let mut out = vec![0.0; BATCH];
            let mut sum2 = 0.0;
            let ns = time(|| {
                let mut sum = 0.0;
                for c in &blocks {
                    e.evaluate_columns(c, &mut out).unwrap();
                    sum += out.iter().sum::<f64>();
                }
                sum2 = black_box(sum);
                DOCS as u64
            });
            batch_row += &format!("{ns:9.2}");
            match reference {
                None => reference = Some((sum1, sum2)),
                Some(r) => assert_eq!(r, (sum1, sum2), "{case}: backends disagree"),
            }
        }
        let mut values: Vec<BoxDoubleValues<'static>> = cols
            .iter()
            .map(|c| {
                Box::new(ArrayValues {
                    values: c.clone(),
                    doc: 0,
                }) as BoxDoubleValues<'static>
            })
            .collect();
        let floor = time(|| {
            let mut sum = 0.0;
            for doc in 0..DOCS {
                for v in &mut values {
                    v.advance_exact(doc).unwrap();
                }
                for v in &mut values {
                    sum += v.double_value().unwrap();
                }
            }
            black_box(sum);
            DOCS as u64
        });
        println!("{case:17} doc    {doc_row}   (positioning and reading alone {floor:.2})");
        println!("{:17} batch  {batch_row}", "");
    }
}
