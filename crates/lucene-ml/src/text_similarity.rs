//! ml-commons' `TEXT_SIMILARITY` function for ONNX cross-encoders
//! (`TextSimilarityCrossEncoderModel` with its `TextSimilarityTranslator`):
//! the query and one document are tokenized as a pair and run together; each
//! model output (the relevance logits) comes back without its batch axis.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::onnx::{self, Feed, OnnxModel, Tensor};
use crate::{tokenizer, Result};

/// A deployed cross-encoder.
pub struct TextSimilarityModel {
    tokenizer: Tokenizer,
    model: OnnxModel,
    feeds: Vec<(String, Feed)>,
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
            model,
            feeds,
        })
    }

    /// The outputs for `query` against `doc`, in graph order.
    pub fn score(&self, query: &str, doc: &str) -> Result<Vec<Tensor>> {
        let encoded = tokenizer::encode_pair(&self.tokenizer, query, doc)?;
        let outputs = self.model.run(&encoded, &self.feeds)?;
        outputs
            .into_iter()
            .map(|o| onnx::unbatch(o.tensor))
            .collect()
    }
}
