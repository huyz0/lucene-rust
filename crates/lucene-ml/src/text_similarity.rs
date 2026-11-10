//! ml-commons' `TEXT_SIMILARITY` function for ONNX cross-encoders
//! (`TextSimilarityCrossEncoderModel` with its `TextSimilarityTranslator`):
//! the query and one document are tokenized as a pair and run together; each
//! model output (the relevance logits) comes back without its batch axis.
//! ONNX models are fed by input name, TorchScript ones three positional
//! arrays.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::onnx::{self, OnnxModel, Tensor};
use crate::torchscript::{Engine, TorchScript, PAIR_FEEDS};
use crate::{tokenizer, Result};

/// A deployed cross-encoder.
pub struct TextSimilarityModel {
    tokenizer: Tokenizer,
    engine: Engine,
}

impl TextSimilarityModel {
    /// Loads the ONNX model at `model` with the tokenizer file
    /// `tokenizer_json`. [`crate::runtime::init`] must have run.
    pub fn load(model: &Path, tokenizer_json: &[u8]) -> Result<TextSimilarityModel> {
        let tokenizer = tokenizer::load(tokenizer_json)?;
        let model = OnnxModel::load(model)?;
        let feeds = onnx::feeds_by_name(model.input_names())?;
        Ok(TextSimilarityModel {
            tokenizer,
            engine: Engine::Onnx(model, feeds),
        })
    }

    /// A TorchScript cross-encoder, already loaded as `module`, with the
    /// tokenizer file `tokenizer_json`.
    pub fn torchscript(
        module: Box<dyn TorchScript>,
        tokenizer_json: &[u8],
    ) -> Result<TextSimilarityModel> {
        Ok(TextSimilarityModel {
            tokenizer: tokenizer::load(tokenizer_json)?,
            engine: Engine::Torch(module, PAIR_FEEDS),
        })
    }

    /// The outputs for `query` against `doc`, in graph order.
    pub fn score(&self, query: &str, doc: &str) -> Result<Vec<Tensor>> {
        let encoded = tokenizer::encode_pair(&self.tokenizer, query, doc)?;
        Ok(self
            .engine
            .unbatched(&encoded)?
            .into_iter()
            .map(|o| o.tensor)
            .collect())
    }
}
