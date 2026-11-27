//! `MovingFunctions`, the static functions a `moving_fn` script calls on its
//! window of `values`: `max`, `min`, `sum`, `stdDev`, `unweightedAvg`,
//! `linearWeightedAvg`, `ewma`, `holt` and `holtWinters`, ported from
//! OpenSearch 3.10.0 including its quirks (`linearWeightedAvg` divides by one
//! more than the weights' sum; the Holt-Winters forecast reads past the
//! values it dropped as NaN).

use crate::api::{array, bool_arg, double_arg, int_arg};
use crate::value::Value;
use crate::vm::{ApiFn, Throw};

pub const CLASS: &str = "org.opensearch.search.aggregations.pipeline.MovingFunctions";

/// `MovingFunctions`'s static methods.
pub fn static_lookup(name: &str, arity: usize) -> Option<ApiFn> {
    Some(match (name, arity) {
        ("max", 1) => |_, a| Ok(Value::Double(extreme(&values(&a[0])?, true))),
        ("min", 1) => |_, a| Ok(Value::Double(extreme(&values(&a[0])?, false))),
        ("sum", 1) => |_, a| Ok(Value::Double(sum(&values(&a[0])?))),
        ("unweightedAvg", 1) => |_, a| Ok(Value::Double(unweighted_avg(&values(&a[0])?))),
        ("linearWeightedAvg", 1) => |_, a| Ok(Value::Double(linear_weighted_avg(&values(&a[0])?))),
        ("stdDev", 2) => |_, a| Ok(Value::Double(std_dev(&values(&a[0])?, double_arg(&a[1])?))),
        ("ewma", 2) => |_, a| Ok(Value::Double(ewma(&values(&a[0])?, double_arg(&a[1])?))),
        ("holt", 3) => |_, a| {
            let v = values(&a[0])?;
            let (alpha, beta) = (double_arg(&a[1])?, double_arg(&a[2])?);
            Ok(Value::Double(if v.is_empty() {
                f64::NAN
            } else {
                holt_forecast(&v, alpha, beta, 1)[0]
            }))
        },
        ("holtWinters", 6) => |_, a| {
            let v = values(&a[0])?;
            let (alpha, beta, gamma) = (double_arg(&a[1])?, double_arg(&a[2])?, double_arg(&a[3])?);
            let (period, multiplicative) = (int_arg(&a[4])?, bool_arg(&a[5])?);
            if v.is_empty() {
                return Ok(Value::Double(f64::NAN));
            }
            let padding = if multiplicative { 1.0e-10 } else { 0.0 };
            let f = holt_winters_forecast(
                &v,
                [alpha, beta, gamma],
                period,
                padding,
                multiplicative,
                1,
            )?;
            Ok(Value::Double(f[0]))
        },
        _ => return None,
    })
}

/// The elements of a `double[]` argument.
fn values(v: &Value) -> Result<Vec<f64>, Throw> {
    array::items(v)?.iter().map(double_arg).collect()
}

/// `DoubleStream.max()` / `min()` over `Math.max` / `Math.min`, which let a
/// NaN win and order -0.0 below 0.0; NaN when empty.
fn extreme(v: &[f64], max: bool) -> f64 {
    let mut it = v.iter().copied();
    let Some(first) = it.next() else {
        return f64::NAN;
    };
    it.fold(first, |a, b| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else if max {
            if a == b {
                if a.is_sign_negative() {
                    b
                } else {
                    a
                }
            } else if a > b {
                a
            } else {
                b
            }
        } else if a == b {
            if a.is_sign_negative() {
                a
            } else {
                b
            }
        } else if a < b {
            a
        } else {
            b
        }
    })
}

/// `DoubleStream.sum()` over the values with NaN as 0: Kahan-compensated, with
/// the simple sum answering when the compensated one is NaN from infinities.
fn sum(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let (mut total, mut compensation, mut simple) = (0.0_f64, 0.0_f64, 0.0_f64);
    for &x in v {
        let d = if x.is_nan() { 0.0 } else { x };
        let tmp = d - compensation;
        let velvel = total + tmp;
        compensation = (velvel - total) - tmp;
        total = velvel;
        simple += d;
    }
    let result = total - compensation;
    if result.is_nan() && simple.is_infinite() {
        simple
    } else {
        result
    }
}

fn unweighted_avg(v: &[f64]) -> f64 {
    let (mut avg, mut count) = (0.0_f64, 0_i64);
    for &x in v.iter().filter(|x| !x.is_nan()) {
        avg += x;
        count += 1;
    }
    if count == 0 {
        f64::NAN
    } else {
        avg / count as f64
    }
}

/// Weights 1, 2, 3 ...; divided by one more than their sum, as Java does.
fn linear_weighted_avg(v: &[f64]) -> f64 {
    let (mut avg, mut total, mut current) = (0.0_f64, 1_i64, 1_i64);
    for &x in v.iter().filter(|x| !x.is_nan()) {
        avg += x * current as f64;
        total += current;
        current += 1;
    }
    if total == 1 {
        f64::NAN
    } else {
        avg / total as f64
    }
}

fn std_dev(v: &[f64], avg: f64) -> f64 {
    if avg.is_nan() {
        return f64::NAN;
    }
    let (mut count, mut variance) = (0_i64, 0.0_f64);
    for &x in v.iter().filter(|x| !x.is_nan()) {
        variance += (x - avg).powf(2.0);
        count += 1;
    }
    (variance / count as f64).sqrt()
}

fn ewma(v: &[f64], alpha: f64) -> f64 {
    let (mut avg, mut first) = (f64::NAN, true);
    for &x in v.iter().filter(|x| !x.is_nan()) {
        if first {
            avg = x;
            first = false;
        } else {
            avg = x * alpha + avg * (1.0 - alpha);
        }
    }
    avg
}

fn holt_forecast(v: &[f64], alpha: f64, beta: f64, forecasts: usize) -> Vec<f64> {
    let (mut s, mut last_s, mut b, mut last_b) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut counter = 0;
    for &x in v.iter().filter(|x| !x.is_nan()) {
        if counter == 0 {
            s = x;
            // `value - last` where `last` is the value itself: NaN for an infinity.
            b = if x.is_infinite() { f64::NAN } else { 0.0 };
        } else {
            s = alpha * x + (1.0 - alpha) * (last_s + last_b);
            b = beta * (s - last_s) + (1.0 - beta) * last_b;
        }
        counter += 1;
        last_s = s;
        last_b = b;
    }
    if counter == 0 {
        return vec![f64::NAN; forecasts];
    }
    (0..forecasts).map(|i| s + i as f64 * b).collect()
}

fn holt_winters_forecast(
    v: &[f64],
    [alpha, beta, gamma]: [f64; 3],
    period: i32,
    padding: f64,
    multiplicative: bool,
    forecasts: usize,
) -> Result<Vec<f64>, Throw> {
    let len = v.len();
    let period_len = usize::try_from(period).unwrap_or(0);
    if (len as i64) < i64::from(period) * 2 {
        return Err(Throw::new(
            "java.lang.IllegalArgumentException",
            format!(
                "Holt-Winters aggregation requires at least (2 * period == 2 * {period} == {}) data-points to function.  Only [{len}] were provided.",
                i64::from(period) * 2
            ),
        ));
    }
    let mut seasonal = vec![0.0_f64; len];
    let vs: Vec<f64> = v
        .iter()
        .filter(|x| !x.is_nan())
        .map(|x| x + padding)
        .collect();
    if vs.is_empty() {
        return Ok(vec![f64::NAN; forecasts]);
    }
    // Java sizes `vs` as `values`, and the entries past the non-NaN ones stay 0.
    let mut vs = vs;
    vs.resize(len, 0.0);
    let (mut s, mut b) = (0.0_f64, 0.0_f64);
    for i in 0..period_len {
        s += vs[i];
        b += (vs[i + period_len] - vs[i]) / f64::from(period);
    }
    s /= f64::from(period);
    b /= f64::from(period);
    let (mut last_s, mut last_b) = (s, 0.0_f64);
    if s == 0.0 {
        seasonal.fill(0.0);
    } else {
        for i in 0..period_len {
            seasonal[i] = vs[i] / s;
        }
    }
    for i in period_len..len {
        s = if multiplicative {
            alpha * (vs[i] / seasonal[i - period_len]) + (1.0 - alpha) * (last_s + last_b)
        } else {
            alpha * (vs[i] - seasonal[i - period_len]) + (1.0 - alpha) * (last_s + last_b)
        };
        b = beta * (s - last_s) + (1.0 - beta) * last_b;
        seasonal[i] = if multiplicative {
            gamma * (vs[i] / (last_s + last_b)) + (1.0 - gamma) * seasonal[i - period_len]
        } else {
            gamma * (vs[i] - (last_s - last_b)) + (1.0 - gamma) * seasonal[i - period_len]
        };
        last_s = s;
        last_b = b;
    }
    Ok((1..=forecasts)
        .map(|i| {
            let at = len - period_len + (i - 1) % period_len;
            if multiplicative {
                (s + i as f64 * b) * seasonal[at]
            } else {
                s + i as f64 * b + seasonal[at]
            }
        })
        .collect())
}
