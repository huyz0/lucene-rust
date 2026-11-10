//! A Painless `script_score` script made of one k-NN Painless function call
//! (`l2Squared`, `l1Norm`, `lInfNorm`, `innerProduct`, `cosineSimilarity`)
//! inside arithmetic on number literals, run natively as the formula of the
//! `knn_score` node ([`crate::jvm_knn_score`]).
//!
//! The plugin's encoder parses the script, folds its constant parts with
//! Painless's (Java's) numeric promotion, and sends a typed postfix program:
//! every operation already says whether it is `float` or `double`
//! arithmetic, so this side only replays it. The function returns a `float`,
//! as the stock functions do, and the program's value is cast to `float` as
//! `ScriptScoreQuery` casts the script's `double`.
//!
//! Anything Painless would throw on -- a document without a vector, vectors
//! of different dimensions, a zero vector's `NaN` cosine -- is an error here,
//! and the plugin then runs the query in Java, which throws as stock does.
//!
//! The formula, after `knn_score`'s target:
//!
//! | field | layout |
//! |---|---|
//! | function | `u8`: `0` l2Squared, `1` l1Norm, `2` lInfNorm, `3` innerProduct, `4` cosineSimilarity, `5` cosineSimilarity with the query's squared norm |
//! | norm | `f32`: that squared norm (0 for the other functions) |
//! | program | `i32` count, then that many operations |
//!
//! An operation is a `u8` code: `0` a `float` constant (`i32` bits), `1` a
//! `double` constant (`i64` bits), `2` the function's value (a `float`), `3`
//! widen a `float` to `double`, `16..=20` `float` add, subtract, multiply,
//! divide, negate, `32..=36` the same in `double`.

use lucene_codecs::vectors::{
    dot_product, dot_product_bytes, square_distance, square_distance_bytes,
};
use lucene_search::{Error, Result};
use lucene_util::simd::cosine_parts_f32;
use lucene_util::vector_util::cosine_sums_i8;

use crate::error::FfiStatus;
use crate::jvm_nodes::invalid;
use crate::jvm_reader::Cursor;

/// The most operations a program may have: far more than any script the
/// encoder accepts.
const MAX_OPS: usize = 256;

/// The deepest a program's stack may get, so evaluation runs on a small
/// fixed array.
const MAX_DEPTH: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Function {
    L2Squared,
    L1Norm,
    LInfNorm,
    InnerProduct,
    Cosine,
    CosineWithNorm,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Op {
    Float(f32),
    Double(f64),
    Value,
    Widen,
    Float2(Arith),
    FloatNeg,
    Double2(Arith),
    DoubleNeg,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Arith {
    Add,
    Sub,
    Mul,
    Div,
}

impl Arith {
    fn f32(self, a: f32, b: f32) -> f32 {
        match self {
            Arith::Add => a + b,
            Arith::Sub => a - b,
            Arith::Mul => a * b,
            Arith::Div => a / b,
        }
    }
    fn f64(self, a: f64, b: f64) -> f64 {
        match self {
            Arith::Add => a + b,
            Arith::Sub => a - b,
            Arith::Mul => a * b,
            Arith::Div => a / b,
        }
    }
}

/// The script: its function and the program around it.
#[derive(Debug, Clone)]
pub(crate) struct Painless {
    function: Function,
    norm: f32,
    ops: Vec<Op>,
}

#[derive(Clone, Copy, PartialEq)]
enum Ty {
    Float,
    Double,
}

pub(crate) fn decode(c: &mut Cursor<'_>) -> std::result::Result<Painless, FfiStatus> {
    let function = match c.u8()? {
        0 => Function::L2Squared,
        1 => Function::L1Norm,
        2 => Function::LInfNorm,
        3 => Function::InnerProduct,
        4 => Function::Cosine,
        5 => Function::CosineWithNorm,
        other => {
            return Err(invalid(format!(
                "query tree: painless k-NN function {other} (expected 0..=5)"
            )))
        }
    };
    let norm = f32::from_bits(c.i32()? as u32);
    let n = c.len()?;
    if n == 0 || n > MAX_OPS {
        return Err(invalid(format!(
            "query tree: painless program of {n} operations (expected 1..={MAX_OPS})"
        )));
    }
    let mut ops = Vec::with_capacity(n); // alloc-ok: at most MAX_OPS
                                         // The types on the stack as the program runs, so a bad program is
                                         // refused here rather than misread per document.
    let mut types: Vec<Ty> = Vec::with_capacity(n); // alloc-ok: at most MAX_OPS
    let mut depth = 0;
    let bad = || invalid("query tree: painless program does not type-check");
    for _ in 0..n {
        let op = match c.u8()? {
            0 => Op::Float(f32::from_bits(c.i32()? as u32)),
            1 => Op::Double(f64::from_bits(c.i64()? as u64)),
            2 => Op::Value,
            3 => Op::Widen,
            code @ 16..=19 => Op::Float2(arith(code - 16)),
            20 => Op::FloatNeg,
            code @ 32..=35 => Op::Double2(arith(code - 32)),
            36 => Op::DoubleNeg,
            other => return Err(invalid(format!("query tree: painless operation {other}"))),
        };
        match op {
            Op::Float(_) | Op::Value => types.push(Ty::Float),
            Op::Double(_) => types.push(Ty::Double),
            Op::Widen => match types.last_mut() {
                Some(t @ Ty::Float) => *t = Ty::Double,
                _ => return Err(bad()),
            },
            Op::FloatNeg | Op::DoubleNeg => {
                let want = if op == Op::FloatNeg {
                    Ty::Float
                } else {
                    Ty::Double
                };
                if types.last() != Some(&want) {
                    return Err(bad());
                }
            }
            Op::Float2(_) | Op::Double2(_) => {
                let want = if matches!(op, Op::Float2(_)) {
                    Ty::Float
                } else {
                    Ty::Double
                };
                if types.pop() != Some(want) || types.last() != Some(&want) {
                    return Err(bad());
                }
            }
        }
        depth = depth.max(types.len());
        ops.push(op);
    }
    if types.len() != 1 || !ops.contains(&Op::Value) || depth > MAX_DEPTH {
        return Err(bad());
    }
    Ok(Painless {
        function,
        norm,
        ops,
    })
}

fn arith(code: u8) -> Arith {
    match code {
        0 => Arith::Add,
        1 => Arith::Sub,
        2 => Arith::Mul,
        _ => Arith::Div,
    }
}

fn same_dimension(q: usize, v: usize) -> Result<()> {
    if q == v {
        Ok(())
    } else {
        Err(Error::IllegalArgument(format!(
            "query vector dimension mismatch. Expected: {v}, Given: {q}"
        )))
    }
}

fn nan_cosine() -> Error {
    Error::IllegalArgument("cosineSimilarity of a zero vector".into())
}

impl Painless {
    /// Whether the function reads the query as floats on a byte field, which
    /// the encoder never sends.
    pub(crate) fn needs_float_target(&self) -> bool {
        self.function == Function::CosineWithNorm
    }

    /// The function over a float (or half-float) document vector.
    fn value_floats(&self, q: &[f32], v: &[f32]) -> Result<f32> {
        same_dimension(q.len(), v.len())?;
        Ok(match self.function {
            Function::L2Squared => square_distance(q, v),
            Function::L1Norm => q.iter().zip(v).fold(0.0f32, |d, (a, b)| d + (a - b).abs()),
            Function::LInfNorm => q
                .iter()
                .zip(v)
                .fold(0.0f32, |d, (a, b)| (a - b).abs().max(d)),
            Function::InnerProduct => dot_product(q, v),
            Function::Cosine => {
                // `VectorUtil.cosine`: NaN for a zero vector.
                let (sum, n1, n2) = cosine_parts_f32(q, v);
                if n1 == 0.0 || n2 == 0.0 {
                    return Err(nan_cosine());
                }
                (f64::from(sum) / (f64::from(n1) * f64::from(n2)).sqrt()) as f32
            }
            Function::CosineWithNorm => {
                // `cosinesimilOptimized`: sequential float sums.
                let (mut dot, mut norm) = (0.0f32, 0.0f32);
                for (a, b) in q.iter().zip(v) {
                    dot += a * b;
                    norm += b * b;
                }
                let product = self.norm * norm;
                if product == 0.0 {
                    0.0
                } else {
                    (f64::from(dot) / f64::from(product).sqrt()) as f32
                }
            }
        })
    }

    /// The function over a byte document vector.
    fn value_bytes(&self, q: &[u8], v: &[u8]) -> Result<f32> {
        same_dimension(q.len(), v.len())?;
        let diffs = || {
            q.iter()
                .zip(v)
                .map(|(&a, &b)| (f32::from(a as i8) - f32::from(b as i8)).abs())
        };
        Ok(match self.function {
            Function::L2Squared => square_distance_bytes(q, v) as f32,
            Function::L1Norm => diffs().fold(0.0f32, |d, x| d + x),
            Function::LInfNorm => diffs().fold(0.0f32, |d, x| x.max(d)),
            Function::InnerProduct => dot_product_bytes(q, v) as f32,
            Function::Cosine => {
                let (sum, n1, n2) = cosine_sums_i8(q, v);
                if n1 == 0 || n2 == 0 {
                    return Err(nan_cosine());
                }
                (f64::from(sum) / (f64::from(n1) * f64::from(n2)).sqrt()) as f32
            }
            Function::CosineWithNorm => return Err(nan_cosine()),
        })
    }

    /// The program's value, as the `float` `ScriptScoreQuery` scores with.
    fn run(&self, value: f32) -> f32 {
        // Every value is held as an `f64`: exact for a `float`, and each
        // `float` operation rounds back to `f32` as Java's does.
        let mut stack = [0f64; MAX_DEPTH];
        let mut top = 0usize;
        for op in &self.ops {
            // The decode-time type check proved every pop has a value under
            // it and the stack never gets deeper than `MAX_DEPTH`.
            match *op {
                Op::Float(x) => {
                    stack[top] = f64::from(x);
                    top += 1;
                }
                Op::Double(x) => {
                    stack[top] = x;
                    top += 1;
                }
                Op::Value => {
                    stack[top] = f64::from(value);
                    top += 1;
                }
                Op::Widen => {}
                Op::FloatNeg => stack[top - 1] = f64::from(-(stack[top - 1] as f32)),
                Op::DoubleNeg => stack[top - 1] = -stack[top - 1],
                Op::Float2(a) => {
                    top -= 1;
                    stack[top - 1] = f64::from(a.f32(stack[top - 1] as f32, stack[top] as f32));
                }
                Op::Double2(a) => {
                    top -= 1;
                    stack[top - 1] = a.f64(stack[top - 1], stack[top]);
                }
            }
        }
        stack[0] as f32
    }

    pub(crate) fn score_floats(&self, q: &[f32], v: &[f32]) -> Result<f32> {
        Ok(self.run(self.value_floats(q, v)?))
    }

    pub(crate) fn score_bytes(&self, q: &[u8], v: &[u8]) -> Result<f32> {
        Ok(self.run(self.value_bytes(q, v)?))
    }

    pub(crate) fn describe(&self) -> String {
        format!("{:?}{:?}", self.function, self.ops)
    }
}

/// The error a document without a vector gets: Painless's `getValue` throws.
pub(crate) fn missing(field: &str, doc: i32) -> Error {
    Error::IllegalArgument(format!("document [{doc}] has no value for field '{field}'"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jvm_nodes::tests::Blob;

    fn program(function: u8, norm: f32, ops: &[Blob]) -> Blob {
        let mut b = Blob::default().u8(function).f32(norm).i32(ops.len() as i32);
        for op in ops {
            b = b.raw(op);
        }
        b
    }

    fn op(code: u8) -> Blob {
        Blob::default().u8(code)
    }

    fn float(x: f32) -> Blob {
        op(0).f32(x)
    }

    fn double(x: f64) -> Blob {
        op(1).f64(x)
    }

    fn decoded(b: &Blob) -> std::result::Result<Painless, FfiStatus> {
        decode(&mut Cursor::new(&b.0))
    }

    #[test]
    fn float_arithmetic_rounds_as_java_floats() {
        // `1 / (1 + l2Squared(q, v))`: the int literals folded to floats.
        let p = decoded(&program(
            0,
            0.0,
            &[float(1.0), float(1.0), op(2), op(16), op(19)],
        ))
        .unwrap();
        let q = [1.0f32, 2.0];
        let v = [0.5f32, 0.25];
        let d = (1.0f32 - 0.5).powi(2) + (2.0f32 - 0.25).powi(2);
        assert_eq!(p.score_floats(&q, &v).unwrap(), 1.0f32 / (1.0 + d));
        // `1.0 / (1.0 + l2Squared(q, v))`: double arithmetic, then the cast.
        let p = decoded(&program(
            0,
            0.0,
            &[double(1.0), double(1.0), op(2), op(3), op(32), op(35)],
        ))
        .unwrap();
        assert_eq!(
            p.score_floats(&q, &v).unwrap(),
            (1.0f64 / (1.0 + f64::from(d))) as f32
        );
        // Negation and the other operations.
        let p = decoded(&program(
            3,
            0.0,
            &[op(2), op(20), float(3.0), op(17), float(2.0), op(18)],
        ))
        .unwrap();
        let dot = 0.5f32 + 0.5;
        assert_eq!(p.score_floats(&q, &v).unwrap(), (-dot - 3.0) * 2.0);
        let p = decoded(&program(
            3,
            0.0,
            &[
                op(2),
                op(3),
                op(36),
                double(0.5),
                op(33),
                double(4.0),
                op(34),
            ],
        ))
        .unwrap();
        assert_eq!(
            p.score_floats(&q, &v).unwrap(),
            ((-f64::from(dot) - 0.5) * 4.0) as f32
        );
    }

    #[test]
    fn each_function_matches_the_painless_ones() {
        let q = [1.0f32, -2.0, 0.5];
        let v = [0.5f32, 1.0, -1.5];
        let value = |f: u8, norm: f32| {
            decoded(&program(f, norm, &[op(2)]))
                .unwrap()
                .score_floats(&q, &v)
                .unwrap()
        };
        assert_eq!(value(0, 0.0), square_distance(&q, &v));
        assert_eq!(value(1, 0.0), 0.5 + 3.0 + 2.0);
        assert_eq!(value(2, 0.0), 3.0);
        assert_eq!(value(3, 0.0), dot_product(&q, &v));
        let (sum, n1, n2) = cosine_parts_f32(&q, &v);
        assert_eq!(
            value(4, 0.0),
            (f64::from(sum) / (f64::from(n1) * f64::from(n2)).sqrt()) as f32
        );
        let qn: f32 = q.iter().map(|x| x * x).sum();
        let (dot, vn) = q
            .iter()
            .zip(&v)
            .fold((0f32, 0f32), |(d, n), (a, b)| (d + a * b, n + b * b));
        assert_eq!(
            value(5, qn),
            (f64::from(dot) / f64::from(qn * vn).sqrt()) as f32
        );
        assert_eq!(value(5, 0.0), 0.0);

        let qb = [1u8, 0xfe, 3];
        let vb = [2u8, 2, 0xff];
        let bytes = |f: u8| {
            decoded(&program(f, 0.0, &[op(2)]))
                .unwrap()
                .score_bytes(&qb, &vb)
                .unwrap()
        };
        assert_eq!(bytes(0), square_distance_bytes(&qb, &vb) as f32);
        assert_eq!(bytes(1), 1.0 + 4.0 + 4.0);
        assert_eq!(bytes(2), 4.0);
        assert_eq!(bytes(3), dot_product_bytes(&qb, &vb) as f32);
        let (sum, n1, n2) = cosine_sums_i8(&qb, &vb);
        assert_eq!(
            bytes(4),
            (f64::from(sum) / (f64::from(n1) * f64::from(n2)).sqrt()) as f32
        );
    }

    #[test]
    fn what_painless_throws_on_is_an_error() {
        let p = decoded(&program(0, 0.0, &[op(2)])).unwrap();
        assert!(p.score_floats(&[1.0], &[1.0, 2.0]).is_err(), "dimensions");
        assert!(p.score_bytes(&[1], &[1, 2]).is_err(), "dimensions");
        let cos = decoded(&program(4, 0.0, &[op(2)])).unwrap();
        assert!(
            cos.score_floats(&[1.0, 0.0], &[0.0, 0.0]).is_err(),
            "zero vector"
        );
        assert!(cos.score_bytes(&[1, 0], &[0, 0]).is_err(), "zero vector");
        let normed = decoded(&program(5, 1.0, &[op(2)])).unwrap();
        assert!(normed.needs_float_target());
        assert!(normed.score_bytes(&[1], &[1]).is_err());
        assert!(!cos.needs_float_target());
        assert!(missing("v", 3).to_string().contains("'v'"));
        assert!(!cos.describe().is_empty());
    }

    #[test]
    fn bad_programs_are_refused() {
        assert!(decoded(&program(6, 0.0, &[op(2)])).is_err(), "function");
        assert!(decoded(&program(0, 0.0, &[])).is_err(), "empty");
        assert!(
            decoded(&program(0, 0.0, &[float(1.0)])).is_err(),
            "no value"
        );
        assert!(
            decoded(&program(0, 0.0, &[op(2), op(2)])).is_err(),
            "two values left"
        );
        assert!(
            decoded(&program(0, 0.0, &[op(2), op(16)])).is_err(),
            "pops nothing"
        );
        assert!(
            decoded(&program(0, 0.0, &[op(2), double(1.0), op(16)])).is_err(),
            "float op on a double"
        );
        assert!(
            decoded(&program(0, 0.0, &[op(2), op(32)])).is_err(),
            "double op on floats"
        );
        assert!(
            decoded(&program(0, 0.0, &[op(2), op(3), op(3)])).is_err(),
            "widen a double"
        );
        assert!(
            decoded(&program(0, 0.0, &[op(2), op(36)])).is_err(),
            "double negate of a float"
        );
        assert!(
            decoded(&program(0, 0.0, &[op(2), op(3), op(20)])).is_err(),
            "float negate of a double"
        );
        assert!(
            decoded(&program(0, 0.0, &[op(2), op(99)])).is_err(),
            "operation"
        );
        let deep: Vec<Blob> = std::iter::once(op(2))
            .chain((0..MAX_DEPTH).map(|_| float(1.0)))
            .chain((0..MAX_DEPTH).map(|_| op(16)))
            .collect();
        assert!(decoded(&program(0, 0.0, &deep)).is_err(), "too deep");
        let long: Vec<Blob> = std::iter::once(op(2))
            .chain((0..MAX_OPS).map(|_| op(20)))
            .collect();
        assert!(decoded(&program(0, 0.0, &long)).is_err(), "too long");
        // Truncated.
        assert!(decode(&mut Cursor::new(&[0u8, 0, 0])).is_err());
    }
}
