//! One ONNX model session, shared by the model functions: loads the file,
//! feeds one tokenized text (a batch of one) by input name and reads every
//! output back as a float [`Tensor`].

use std::path::Path;
use std::sync::Mutex;

use ort::session::Session;
use ort::value::Tensor as OrtTensor;

use crate::tokenizer::Encoded;
use crate::{runtime, MlError, Result};

/// A float tensor, row major.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// The dimensions.
    pub shape: Vec<usize>,
    /// The values.
    pub data: Vec<f32>,
}

/// One output of a run, by the model's name for it.
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    /// The output's name in the model graph.
    pub name: String,
    /// Its values.
    pub tensor: Tensor,
}

/// Which tokenizer array a model input takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feed {
    /// `input_ids`.
    Ids,
    /// `attention_mask`.
    AttentionMask,
    /// `token_type_ids`.
    TypeIds,
}

pub(crate) fn model_error(e: impl std::fmt::Display) -> MlError {
    MlError::Model(e.to_string())
}

/// A loaded ONNX model.
pub struct OnnxModel {
    session: Mutex<Session>,
    inputs: Vec<String>,
}

impl OnnxModel {
    /// Loads the ONNX model at `model`. [`runtime::init`] must have run.
    pub fn load(model: &Path) -> Result<OnnxModel> {
        let environment = runtime::environment()?;
        let session = Session::builder(environment)
            .map_err(model_error)?
            .commit_from_file(model)
            .map_err(model_error)?;
        let inputs = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        Ok(OnnxModel {
            session: Mutex::new(session),
            inputs,
        })
    }

    /// The graph's input names, in order.
    pub fn input_names(&self) -> &[String] {
        &self.inputs
    }

    /// Runs `encoded` as a batch of one, feeding input `name` the array
    /// `feed` picks; the outputs come back in graph order.
    pub fn run<S: AsRef<str>>(
        &self,
        encoded: &Encoded,
        feeds: &[(S, Feed)],
    ) -> Result<Vec<Output>> {
        if encoded.ids.is_empty() {
            // DJL's ONNX engine refuses an empty array before running.
            return Err(MlError::Model(
                "OnnxRuntime doesn't support 0 length tensor.".into(),
            ));
        }
        let shape = [1usize, encoded.ids.len()];
        let mut inputs = Vec::new();
        for (name, feed) in feeds {
            let values = match feed {
                Feed::Ids => &encoded.ids,
                Feed::AttentionMask => &encoded.attention_mask,
                Feed::TypeIds => &encoded.type_ids,
            };
            let tensor = OrtTensor::from_array((shape, values.clone())).map_err(model_error)?;
            inputs.push((name.as_ref(), tensor));
        }
        let mut session = self.session.lock().unwrap_or_else(|p| p.into_inner());
        let outputs = session.run(inputs).map_err(model_error)?;
        let mut result = Vec::new();
        for (name, value) in outputs.iter() {
            let (shape, data) = value.try_extract_tensor::<f32>().map_err(model_error)?;
            let shape = shape
                .iter()
                .map(|&d| usize::try_from(d).map_err(model_error))
                .collect::<Result<Vec<_>>>()?;
            result.push(Output {
                name: name.to_string(),
                tensor: Tensor {
                    shape,
                    data: data.to_vec(),
                },
            });
        }
        Ok(result)
    }
}

/// Feeds every graph input by its name: `input_ids`, `attention_mask` and
/// `token_type_ids`, each optionally prefixed `input1.` (the names
/// ml-commons' `SentenceTransformerTranslator` gives its arrays).
pub fn feeds_by_name(inputs: &[String]) -> Result<Vec<(String, Feed)>> {
    inputs
        .iter()
        .map(|name| {
            let feed = match name.strip_prefix("input1.").unwrap_or(name) {
                "input_ids" => Feed::Ids,
                "attention_mask" => Feed::AttentionMask,
                "token_type_ids" => Feed::TypeIds,
                _ => {
                    return Err(MlError::Model(format!(
                        "the model takes an input [{name}] that is not input_ids, attention_mask \
                         or token_type_ids"
                    )))
                }
            };
            Ok((name.clone(), feed))
        })
        .collect()
}

/// One item of a batch of one: DJL's `Batchifier.STACK` unbatchify, which
/// drops the leading axis. An output without a leading axis of 1 is refused,
/// as DJL fails to split it.
pub fn unbatch(t: Tensor) -> Result<Tensor> {
    match t.shape.split_first() {
        Some((1, rest)) => Ok(Tensor {
            shape: rest.to_vec(),
            data: t.data,
        }),
        _ => Err(MlError::Model(format!(
            "an output of shape {:?} is not a batch of one",
            t.shape
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unbatch_drops_the_leading_axis() {
        let t = Tensor {
            shape: vec![1, 3],
            data: vec![1.0, 2.0, 3.0],
        };
        assert_eq!(
            unbatch(t).unwrap(),
            Tensor {
                shape: vec![3],
                data: vec![1.0, 2.0, 3.0]
            }
        );
        let one = Tensor {
            shape: vec![1],
            data: vec![4.0],
        };
        assert_eq!(unbatch(one).unwrap().shape, Vec::<usize>::new());
        for shape in [vec![], vec![0, 2], vec![2, 1]] {
            let t = Tensor {
                shape,
                data: vec![5.0, 6.0],
            };
            assert!(matches!(unbatch(t), Err(MlError::Model(_))));
        }
    }

    #[test]
    fn feeds_inputs_by_name() {
        let names = ["input1.input_ids", "attention_mask", "token_type_ids"].map(String::from);
        let feeds = feeds_by_name(&names).unwrap();
        assert_eq!(
            feeds,
            vec![
                ("input1.input_ids".to_string(), Feed::Ids),
                ("attention_mask".to_string(), Feed::AttentionMask),
                ("token_type_ids".to_string(), Feed::TypeIds)
            ]
        );
        assert!(matches!(
            feeds_by_name(&["pixel_values".to_string()]),
            Err(MlError::Model(_))
        ));
    }
}
