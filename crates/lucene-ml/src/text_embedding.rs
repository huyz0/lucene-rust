//! ml-commons' `TEXT_EMBEDDING` function for ONNX models: tokenize, run,
//! pool, normalize, the way `ONNXSentenceTransformerTextEmbeddingTranslator`
//! does on DJL arrays.
//!
//! Texts run one at a time, as `TextEmbeddingModel.predict` loops over them.
//! Padding a batch to its longest text costs more than it saves on CPU for
//! document-length texts, and running alone keeps each vector independent of
//! its neighbours.

use std::path::Path;

use tokenizers::Tokenizer;

pub use crate::onnx::Tensor;
use crate::onnx::{Feed, OnnxModel};
use crate::tokenizer;
use crate::{MlError, Result};

/// `TextEmbeddingModelConfig.PoolingMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// Masked mean of the token embeddings.
    Mean,
    /// Masked sum over the square root of the token count.
    MeanSqrtLen,
    /// Masked maximum per dimension, kept as one row.
    Max,
    /// Masked mean weighted by position (1, 2, ...).
    WeightedMean,
    /// The first token's embedding.
    Cls,
    /// The last unmasked token's embedding.
    LastToken,
    /// The model's second output when it has one (pre-pooled), else its first
    /// as it is.
    None,
}

impl Pooling {
    /// The mode by its ml-commons name (`MEAN`, `mean_sqrt_len`, ...).
    pub fn from_name(name: &str) -> Result<Pooling> {
        Ok(match name.to_ascii_uppercase().as_str() {
            "MEAN" => Pooling::Mean,
            "MEAN_SQRT_LEN" => Pooling::MeanSqrtLen,
            "MAX" => Pooling::Max,
            "WEIGHTED_MEAN" => Pooling::WeightedMean,
            "CLS" => Pooling::Cls,
            "LAST_TOKEN" => Pooling::LastToken,
            "NONE" => Pooling::None,
            other => {
                return Err(MlError::InvalidArgument(format!(
                    "unsupported pooling mode {other}"
                )))
            }
        })
    }
}

/// How a model's inputs and outputs are read.
#[derive(Debug, Clone)]
pub struct Config {
    /// `model_config.model_type`: `bert` and `albert` models also take
    /// `token_type_ids`.
    pub model_type: String,
    /// `model_config.pooling_mode`.
    pub pooling: Pooling,
    /// `model_config.normalize_result`: L2-normalize along the first axis.
    pub normalize: bool,
}

/// A deployed text-embedding model.
pub struct TextEmbeddingModel {
    tokenizer: Tokenizer,
    model: OnnxModel,
    token_types: bool,
    pooling: Pooling,
    normalize: bool,
}

impl TextEmbeddingModel {
    /// Loads the ONNX model at `model` with the tokenizer file `tokenizer`.
    /// [`crate::runtime::init`] must have run.
    pub fn load(model: &Path, tokenizer_json: &[u8], config: &Config) -> Result<Self> {
        let tokenizer = tokenizer::load(tokenizer_json)?;
        let model = OnnxModel::load(model)?;
        let model_type = config.model_type.to_ascii_lowercase();
        Ok(TextEmbeddingModel {
            tokenizer,
            model,
            token_types: model_type == "bert" || model_type == "albert",
            pooling: config.pooling,
            normalize: config.normalize,
        })
    }

    /// The embedding of each text, in order.
    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Tensor>> {
        texts.iter().map(|t| self.embed_one(t)).collect()
    }

    fn embed_one(&self, text: &str) -> Result<Tensor> {
        let encoded = tokenizer::encode(&self.tokenizer, text)?;
        let outputs = self.run(&encoded)?;
        let raw = if self.pooling == Pooling::None && outputs.len() > 1 {
            outputs.into_iter().nth(1)
        } else {
            outputs.into_iter().next()
        }
        .ok_or_else(|| MlError::Model("the model has no output".into()))?;
        let raw = drop_batch(raw);
        let pooled = pool(raw, &encoded.attention_mask, self.pooling)?;
        Ok(if self.normalize {
            normalize_first_axis(pooled)
        } else {
            pooled
        })
    }

    fn run(&self, encoded: &tokenizer::Encoded) -> Result<Vec<Tensor>> {
        let mut feeds = vec![
            ("input_ids", Feed::Ids),
            ("attention_mask", Feed::AttentionMask),
        ];
        if self.token_types {
            feeds.push(("token_type_ids", Feed::TypeIds));
        }
        let outputs = self.model.run(encoded, &feeds)?;
        Ok(outputs.into_iter().map(|o| o.tensor).collect())
    }
}

/// `embeddings.get(0)` when the output is `[batch, tokens, dim]`.
fn drop_batch(t: Tensor) -> Tensor {
    if t.shape.len() != 3 {
        return t;
    }
    let shape = t.shape[1..].to_vec();
    let n: usize = shape.iter().product();
    let mut data = t.data;
    data.truncate(n);
    Tensor { shape, data }
}

/// Pools `[tokens, dim]` token embeddings by `mode` with the attention mask.
pub fn pool(t: Tensor, mask: &[i64], mode: Pooling) -> Result<Tensor> {
    if mode == Pooling::None {
        return Ok(t);
    }
    let dim = match t.shape.as_slice() {
        // At least one token: CLS and LAST_TOKEN index the first row.
        [r, d]
            if *d > 0 && *r > 0 && *r == mask.len() && t.data.len().checked_div(*d) == Some(*r) =>
        {
            *d
        }
        other => {
            return Err(MlError::Model(format!(
                "token embeddings of shape {other:?} cannot be pooled over {} tokens",
                mask.len()
            )))
        }
    };
    let rows = || t.data.chunks_exact(dim).zip(mask.iter().copied());
    let vector = |data: Vec<f32>| Tensor {
        shape: vec![dim],
        data,
    };
    // Masks and positions are at most a few thousand, far from any overflow.
    let weighted_sum = |weight: &dyn Fn(usize, i64) -> i64| {
        let mut sum = vec![0f32; dim];
        let mut total = 0i64;
        for (position, (row, m)) in rows().enumerate() {
            let w = weight(position, m);
            total = total.saturating_add(w);
            if w != 0 {
                let f = w as f32;
                sum.iter_mut().zip(row).for_each(|(s, &v)| *s += v * f);
            }
        }
        (sum, total)
    };
    Ok(match mode {
        Pooling::Mean | Pooling::MeanSqrtLen => {
            let (sum, count) = weighted_sum(&|_, m| m);
            let count = count.max(1) as f32;
            let d = if mode == Pooling::Mean {
                count
            } else {
                count.sqrt()
            };
            vector(sum.into_iter().map(|s| s / d).collect())
        }
        Pooling::WeightedMean => {
            let (sum, total) = weighted_sum(&|position, m| {
                m.saturating_mul(
                    i64::try_from(position)
                        .unwrap_or(i64::MAX)
                        .saturating_add(1),
                )
            });
            let total = total as f32;
            vector(sum.into_iter().map(|s| s / total).collect())
        }
        Pooling::Max => {
            let mut max = vec![f32::NEG_INFINITY; dim];
            for (row, m) in rows() {
                for (x, &v) in max.iter_mut().zip(row) {
                    let v = if m == 0 { -1e9 } else { v };
                    if v > *x {
                        *x = v;
                    }
                }
            }
            Tensor {
                shape: vec![1, dim],
                data: max,
            }
        }
        Pooling::Cls => vector(t.data[..dim].to_vec()),
        Pooling::LastToken => {
            let count = mask.iter().filter(|&&m| m != 0).count();
            let last = count.saturating_sub(1);
            let row = t.data.chunks_exact(dim).nth(last).unwrap_or(&t.data[..dim]);
            vector(row.to_vec())
        }
        Pooling::None => unreachable!("returned above"),
    })
}

/// DJL's `normalize(2, 0)`: each slice along the first axis divided by its
/// L2 norm, the norm floored at 1e-12.
pub fn normalize_first_axis(mut t: Tensor) -> Tensor {
    let rows = t.shape.first().copied().unwrap_or(0);
    if rows == 0 || t.data.is_empty() {
        return t;
    }
    let stride = t.data.len().checked_div(rows).unwrap_or(0);
    if rows == 1 {
        t.data.iter_mut().for_each(|v| *v /= v.abs().max(1e-12));
        return t;
    }
    let mut norms = vec![0f32; stride];
    for row in t.data.chunks_exact(stride) {
        norms.iter_mut().zip(row).for_each(|(n, &v)| *n += v * v);
    }
    norms.iter_mut().for_each(|n| *n = n.sqrt().max(1e-12));
    for row in t.data.chunks_exact_mut(stride) {
        row.iter_mut().zip(&norms).for_each(|(v, &n)| *v /= n);
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens() -> Tensor {
        Tensor {
            shape: vec![3, 2],
            data: vec![1.0, 2.0, 3.0, -4.0, 100.0, 100.0],
        }
    }
    const MASK: [i64; 3] = [1, 1, 0];

    #[test]
    fn pooling_modes() {
        let p = |m| pool(tokens(), &MASK, m).unwrap();
        assert_eq!(p(Pooling::Mean).data, vec![2.0, -1.0]);
        let s = 2f32.sqrt();
        assert_eq!(p(Pooling::MeanSqrtLen).data, vec![4.0 / s, -2.0 / s]);
        let max = p(Pooling::Max);
        assert_eq!((max.shape, max.data), (vec![1, 2], vec![3.0, 2.0]));
        assert_eq!(p(Pooling::WeightedMean).data, vec![7.0 / 3.0, -6.0 / 3.0]);
        assert_eq!(p(Pooling::Cls).data, vec![1.0, 2.0]);
        assert_eq!(p(Pooling::LastToken).data, vec![3.0, -4.0]);
        assert_eq!(p(Pooling::None), tokens());
    }

    #[test]
    fn last_token_of_an_empty_mask_is_the_first() {
        let t = pool(tokens(), &[0, 0, 0], Pooling::LastToken).unwrap();
        assert_eq!(t.data, vec![1.0, 2.0]);
    }

    #[test]
    fn pooling_refuses_bad_shapes() {
        let flat = Tensor {
            shape: vec![6],
            data: vec![0.0; 6],
        };
        assert!(pool(flat, &MASK, Pooling::Mean).is_err());
        assert!(pool(tokens(), &[1, 1], Pooling::Mean).is_err());
        let none = Tensor {
            shape: vec![0, 2],
            data: vec![],
        };
        assert!(pool(none, &[], Pooling::Cls).is_err());
    }

    #[test]
    fn normalizes_along_the_first_axis() {
        let v = normalize_first_axis(Tensor {
            shape: vec![2],
            data: vec![3.0, 4.0],
        });
        assert_eq!(v.data, vec![0.6, 0.8]);
        // A [1, dim] row (MAX pooling): every column is its own slice.
        let row = normalize_first_axis(Tensor {
            shape: vec![1, 3],
            data: vec![-2.0, 0.5, 0.0],
        });
        assert_eq!(row.data, vec![-1.0, 1.0, 0.0]);
        let empty = normalize_first_axis(Tensor {
            shape: vec![],
            data: vec![],
        });
        assert!(empty.data.is_empty());
    }

    #[test]
    fn drops_the_batch_axis() {
        let t = drop_batch(Tensor {
            shape: vec![1, 2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        });
        assert_eq!(t.shape, vec![2, 2]);
        let flat = drop_batch(Tensor {
            shape: vec![4],
            data: vec![0.0; 4],
        });
        assert_eq!(flat.shape, vec![4]);
    }

    #[test]
    fn pooling_names() {
        assert_eq!(Pooling::from_name("mean").unwrap(), Pooling::Mean);
        assert_eq!(
            Pooling::from_name("MEAN_SQRT_LEN").unwrap(),
            Pooling::MeanSqrtLen
        );
        assert_eq!(Pooling::from_name("max").unwrap(), Pooling::Max);
        assert_eq!(
            Pooling::from_name("weighted_mean").unwrap(),
            Pooling::WeightedMean
        );
        assert_eq!(Pooling::from_name("cls").unwrap(), Pooling::Cls);
        assert_eq!(
            Pooling::from_name("last_token").unwrap(),
            Pooling::LastToken
        );
        assert_eq!(Pooling::from_name("none").unwrap(), Pooling::None);
        assert!(Pooling::from_name("median").is_err());
    }
}
