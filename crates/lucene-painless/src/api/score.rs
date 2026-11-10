//! The score context's functions (OpenSearch's `ScoreScriptUtils`):
//! `saturation`, `sigmoid`, the numeric decay functions and `randomScore`.

use lucene_util::strict_math as sm;
use lucene_util::string_helper::murmurhash3_x86_32;

use crate::api::{double_arg, int_arg, str_arg, ClassBinding};
use crate::object::JObject;
use crate::value::Value;
use crate::vm::{ApiFn, Throw, Vm};

pub const UTILS: &str = "org.opensearch.script.ScoreScriptUtils";

/// `ScoreScriptUtils`'s static methods.
pub fn static_lookup(name: &str, arity: usize) -> Option<ApiFn> {
    Some(match (name, arity) {
        ("saturation", 2) => |_, a| {
            let (v, k) = (double_arg(&a[0])?, double_arg(&a[1])?);
            Ok(Value::Double(v / (k + v)))
        },
        ("sigmoid", 3) => |_, a| {
            let (v, k, e) = (double_arg(&a[0])?, double_arg(&a[1])?, double_arg(&a[2])?);
            Ok(Value::Double(
                sm::pow(v, e) / (sm::pow(k, e) + sm::pow(v, e)),
            ))
        },
        _ => return None,
    })
}

/// A class binding's state, made when the script first calls it.
#[derive(Debug)]
enum State {
    /// A numeric decay: `origin`, `offset` and the precomputed `scaling`.
    Decay {
        origin: f64,
        offset: f64,
        scaling: f64,
    },
    /// `RandomScoreDoc`: the salted seed.
    RandomDoc(i32),
    /// `RandomScoreField`: the salted seed and the field's values view.
    RandomField(i32, Value),
}

impl JObject for State {
    fn class_name(&self) -> &str {
        UTILS
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn state(v: &Value) -> &State {
    crate::object::downcast::<State>(v).expect("a class binding's own state")
}

/// The four arguments of a numeric decay's constructor.
fn decay_args(a: &[Value]) -> Result<[f64; 4], Throw> {
    Ok([
        double_arg(&a[0])?,
        double_arg(&a[1])?,
        double_arg(&a[2])?,
        double_arg(&a[3])?,
    ])
}

fn distance(s: &State, doc_value: &Value) -> Result<f64, Throw> {
    let State::Decay { origin, offset, .. } = s else {
        unreachable!("a decay's state")
    };
    Ok(0.0f64.max((double_arg(doc_value)? - origin).abs() - offset))
}

fn scaling(s: &State) -> f64 {
    match s {
        State::Decay { scaling, .. } => *scaling,
        _ => unreachable!("a decay's state"),
    }
}

/// `(scoreScript._getIndex().hashCode() << 10) | scoreScript._getShardId()`,
/// mixed with the seed.
fn salted(vm: &mut Vm, seed: &Value) -> Result<i32, Throw> {
    let seed = int_arg(seed)?;
    let index = vm.env.internal("_getIndex", &[])?;
    let shard = int_arg(&vm.env.internal("_getShardId", &[])?)?;
    let hash = match &index {
        Value::Null => return Err(crate::api::npe()),
        v => v.java_hash(),
    };
    Ok(mix32(((hash << 10) | shard) ^ seed))
}

/// `BitMixer.mix32`: MurmurHash3's finaliser.
fn mix32(k: i32) -> i32 {
    let mut k = k as u32;
    k = (k ^ (k >> 16)).wrapping_mul(0x85eb_ca6b);
    k = (k ^ (k >> 13)).wrapping_mul(0xc2b2_ae35);
    (k ^ (k >> 16)) as i32
}

/// `(hash & 0x00FFFFFF) / (float) (1 << 24)`.
fn random(text: &str, seed: i32) -> Value {
    let hash = murmurhash3_x86_32(text.as_bytes(), seed);
    Value::Double(f64::from((hash & 0x00FF_FFFF) as f32 / (1u32 << 24) as f32))
}

/// The score context's class bindings.
pub fn class_binding(class: &str) -> Option<ClassBinding> {
    let name = class.strip_prefix(UTILS)?.strip_prefix('$')?;
    Some(match name {
        "DecayNumericLinear" => ClassBinding {
            construct: |_, a| {
                let [origin, scale, offset, decay] = decay_args(a)?;
                Ok(Value::object(State::Decay {
                    origin,
                    offset,
                    scaling: scale / (1.0 - decay),
                }))
            },
            call: |_, s, a| {
                let s = state(s);
                let (d, scaling) = (distance(s, &a[0])?, scaling(s));
                Ok(Value::Double(0.0f64.max((scaling - d) / scaling)))
            },
        },
        "DecayNumericExp" => ClassBinding {
            construct: |_, a| {
                let [origin, scale, offset, decay] = decay_args(a)?;
                Ok(Value::object(State::Decay {
                    origin,
                    offset,
                    scaling: sm::log(decay) / scale,
                }))
            },
            call: |_, s, a| {
                let s = state(s);
                Ok(Value::Double(sm::exp(scaling(s) * distance(s, &a[0])?)))
            },
        },
        "DecayNumericGauss" => ClassBinding {
            construct: |_, a| {
                let [origin, scale, offset, decay] = decay_args(a)?;
                Ok(Value::object(State::Decay {
                    origin,
                    offset,
                    scaling: 0.5 * sm::pow(scale, 2.0) / sm::log(decay),
                }))
            },
            call: |_, s, a| {
                let s = state(s);
                let d = distance(s, &a[0])?;
                Ok(Value::Double(sm::exp(0.5 * sm::pow(d, 2.0) / scaling(s))))
            },
        },
        "RandomScoreDoc" => ClassBinding {
            construct: |vm, a| Ok(Value::object(State::RandomDoc(salted(vm, &a[0])?))),
            call: |vm, s, _| {
                let State::RandomDoc(seed) = state(s) else {
                    unreachable!("RandomScoreDoc's state")
                };
                let doc = int_arg(&vm.env.internal("_getDocBaseId", &[])?)?;
                Ok(random(&doc.to_string(), *seed))
            },
        },
        "RandomScoreField" => ClassBinding {
            construct: |vm, a| {
                let seed = salted(vm, &a[0])?;
                let field = str_arg(&a[1])?;
                // The constructor takes the field's values view; reading
                // it again per document reads that document's values.
                vm.env.doc_values(&field)?;
                Ok(Value::object(State::RandomField(seed, Value::Str(field))))
            },
            call: |vm, s, _| {
                let State::RandomField(seed, field) = state(s) else {
                    unreachable!("RandomScoreField's state")
                };
                let Value::Str(field) = field else {
                    unreachable!("a field name")
                };
                let values = vm.env.doc_values(field)?;
                let first = crate::api::dynamic::call(vm, &values, "get", &[Value::Int(0)])?;
                Ok(random(&first.java_to_string(), *seed))
            },
        },
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix32_is_bitmixers() {
        // BitMixer.mix32(1), BitMixer.mix32(-1).
        assert_eq!(mix32(1), 1_364_076_727);
        assert_eq!(mix32(0), 0);
    }
}
