//! ml-commons' `SPARSE_ENCODING` function for ONNX models
//! (`TextEmbeddingSparseEncodingModel` with its `SparseEncodingTranslator`):
//! one text at a time, the model's per-vocabulary weights read back as the
//! non-zero entries of each output, keyed by the decoded token (special
//! tokens decode to nothing and are dropped) or by the token id.

use std::collections::HashMap;
use std::path::Path;

use tokenizers::Tokenizer;

use crate::onnx::{self, Feed, OnnxModel};
use crate::sparse_tokenize::Weight;
use crate::{tokenizer, MlError, Result};

/// One model output's sparse vector.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseOutput {
    /// The output's name in the model graph.
    pub name: String,
    /// Its non-zero weights in token id order; `token` is empty when the
    /// vector is keyed by id.
    pub weights: Vec<Weight>,
}

/// A deployed sparse encoding model.
pub struct SparseEncodingModel {
    tokenizer: Tokenizer,
    model: OnnxModel,
    feeds: Vec<(String, Feed)>,
}

impl SparseEncodingModel {
    /// Loads the ONNX model at `model` with the tokenizer file
    /// `tokenizer_json`. [`crate::runtime::init`] must have run.
    pub fn load(model: &Path, tokenizer_json: &[u8]) -> Result<SparseEncodingModel> {
        let tokenizer = tokenizer::load(tokenizer_json)?;
        let model = OnnxModel::load(model)?;
        let feeds = onnx::feeds_by_name(model.input_names())?;
        Ok(SparseEncodingModel {
            tokenizer,
            model,
            feeds,
        })
    }

    /// `text`'s sparse vector from each model output, keyed by token id when
    /// `by_id` (`sparse_embedding_format: TOKEN_ID`), else by token.
    pub fn encode(&self, text: &str, by_id: bool) -> Result<Vec<SparseOutput>> {
        let encoded = tokenizer::encode(&self.tokenizer, text)?;
        let outputs = self.model.run(&encoded, &self.feeds)?;
        outputs
            .into_iter()
            .map(|o| {
                let tensor = onnx::unbatch(o.tensor);
                if tensor.shape.len() > 1 {
                    return Err(MlError::Model(format!(
                        "sparse encoding output [{}] has shape {:?}, not one weight per token id",
                        o.name, tensor.shape
                    )));
                }
                let weights = self.weights(&tensor.data, by_id)?;
                Ok(SparseOutput {
                    name: o.name,
                    weights,
                })
            })
            .collect()
    }

    fn weights(&self, values: &[f32], by_id: bool) -> Result<Vec<Weight>> {
        let mut out: Vec<Weight> = Vec::new();
        // A later id that decodes to the same token replaces the earlier
        // weight, as the translator's map put does.
        let mut slot: HashMap<String, usize> = HashMap::new();
        for (index, &weight) in values.iter().enumerate() {
            if weight == 0.0 {
                continue;
            }
            let id = u32::try_from(index).map_err(onnx::model_error)?;
            if by_id {
                out.push(Weight {
                    id,
                    token: String::new(),
                    weight,
                });
                continue;
            }
            let token = self
                .tokenizer
                .decode(&[id], true)
                .map_err(|e| MlError::Tokenizer(e.to_string()))?;
            if token.is_empty() {
                continue;
            }
            match slot.get(&token) {
                Some(&at) => out[at] = Weight { id, token, weight },
                None => {
                    slot.insert(token.clone(), out.len());
                    out.push(Weight { id, token, weight });
                }
            }
        }
        Ok(out)
    }
}
