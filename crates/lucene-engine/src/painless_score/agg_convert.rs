//! How an aggregation script's value is read as the node's long or double
//! (`ScriptLongValues` / `ScriptDoubleValues`).

use super::*;

/// `ScriptLongValues`: a number's `longValue`, a boolean as `1` or `0`.
pub fn agg_long(v: &Value) -> Result<i64> {
    Ok(match v {
        Value::Bool(b) => i64::from(*b),
        Value::Byte(n) => i64::from(*n),
        Value::Short(n) => i64::from(*n),
        Value::Int(n) => i64::from(*n),
        Value::Long(n) => *n,
        Value::Float(n) => *n as i64,
        Value::Double(n) => *n as i64,
        other => {
            return Err(Error::IllegalArgument(format!(
            "{AGGREGATION_ERROR}Unsupported script value [{}], expected a number, date, or boolean",
            other.java_to_string()
        )))
        }
    })
}

/// `ScriptDoubleValues.toDoubleValue`.
pub fn agg_double(v: &Value) -> Result<f64> {
    Ok(match v {
        Value::Bool(b) => f64::from(u8::from(*b)),
        Value::Byte(n) => f64::from(*n),
        Value::Short(n) => f64::from(*n),
        Value::Int(n) => f64::from(*n),
        Value::Long(n) => *n as f64,
        Value::Float(n) => f64::from(*n),
        Value::Double(n) => *n,
        other => {
            return Err(Error::IllegalArgument(format!(
            "{AGGREGATION_ERROR}Unsupported script value [{}], expected a number, date, or boolean",
            other.java_to_string()
        )))
        }
    })
}
