//! TorchScript models, as ml-commons runs them on DJL's PyTorch engine.
//!
//! The runtime itself (libtorch) is reached through [`TorchScript`], which
//! the caller implements over the native library; this module holds what
//! DJL does around a forward call: which arrays go in under which names, and
//! how the outputs come back.
//!
//! DJL names each input array. One named `<group>.<key>` goes into a dict
//! argument with the group's other arrays, at the position of the group's
//! first array; any other name is a positional argument. Outputs are
//! flattened: a tensor, a list or tuple of them, or a dict whose keys name
//! its tensors. A batch of one comes back without its batch axis.

use crate::onnx::{self, Feed, Output, Tensor};
use crate::tokenizer::Encoded;
use crate::{MlError, Result};

/// One named input array: `values` as a `[1, len]` int64 tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Input<'a> {
    /// DJL's name for the array.
    pub name: &'a str,
    /// The token values.
    pub values: &'a [i64],
}

/// An output tensor's values, in its own data type.
#[derive(Debug, Clone, PartialEq)]
pub enum Data {
    /// `FLOAT32`.
    F32(Vec<f32>),
    /// `INT64`.
    I64(Vec<i64>),
}

impl Data {
    /// The number of values.
    pub fn len(&self) -> usize {
        match self {
            Data::F32(v) => v.len(),
            Data::I64(v) => v.len(),
        }
    }

    /// Whether there are no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The ml-commons data type name.
    pub fn type_name(&self) -> &'static str {
        match self {
            Data::F32(_) => "FLOAT32",
            Data::I64(_) => "INT64",
        }
    }
}

/// One output of a forward call.
#[derive(Debug, Clone, PartialEq)]
pub struct TorchOutput {
    /// The dict key it came under; empty for a tensor outside a dict.
    pub name: String,
    /// The dict it came from, numbered from 1 in output order; 0 outside a
    /// dict. DJL reads a dict back through a hash map, whose iteration order
    /// the caller applies per dict.
    pub group: u32,
    /// The dimensions.
    pub shape: Vec<usize>,
    /// The values.
    pub data: Data,
}

/// A loaded TorchScript module.
pub trait TorchScript: Send + Sync {
    /// Runs the module's `forward` on `inputs`, grouped and flattened as the
    /// module doc says.
    fn forward(&self, inputs: &[Input<'_>]) -> Result<Vec<TorchOutput>>;
}

/// The arrays ml-commons' cross-encoder translator feeds a TorchScript
/// model: three positional arguments. It asks the model which inputs it
/// takes, but a TorchScript model knows only after its first run, which is
/// fed all three.
pub const PAIR_FEEDS: &[(&str, Feed)] = &[
    ("input_ids", Feed::Ids),
    ("attention_mask", Feed::AttentionMask),
    ("token_type_ids", Feed::TypeIds),
];

/// The arrays a sentence-transformers translator feeds (ml-commons'
/// `SentenceTransformerTranslator`, also the sparse encoding one): one dict
/// argument.
pub const DICT_FEEDS: &[(&str, Feed)] = &[
    ("input1.input_ids", Feed::Ids),
    ("input1.attention_mask", Feed::AttentionMask),
];

/// Runs `module` on `encoded`, feeding the arrays `feeds` names, and drops
/// each output's batch axis.
pub fn run(
    module: &dyn TorchScript,
    encoded: &Encoded,
    feeds: &[(&str, Feed)],
) -> Result<Vec<TorchOutput>> {
    let inputs: Vec<Input<'_>> = feeds
        .iter()
        .map(|&(name, feed)| Input {
            name,
            values: match feed {
                Feed::Ids => &encoded.ids,
                Feed::AttentionMask => &encoded.attention_mask,
                Feed::TypeIds => &encoded.type_ids,
            },
        })
        .collect();
    module.forward(&inputs)?.into_iter().map(unbatch).collect()
}

/// [`onnx::unbatch`] for a typed output.
fn unbatch(o: TorchOutput) -> Result<TorchOutput> {
    match o.shape.split_first() {
        Some((1, rest)) => Ok(TorchOutput {
            shape: rest.to_vec(),
            ..o
        }),
        _ => Err(MlError::Model(format!(
            "an output of shape {:?} is not a batch of one",
            o.shape
        ))),
    }
}

/// `o` as a float output, for the translators that read floats; an output of
/// another type is refused.
pub fn float_output(o: TorchOutput) -> Result<Output> {
    match o.data {
        Data::F32(data) => Ok(Output {
            name: o.name,
            tensor: Tensor {
                shape: o.shape,
                data,
            },
        }),
        other => Err(MlError::Model(format!(
            "the model output [{}] is {}, not FLOAT32",
            o.name,
            other.type_name()
        ))),
    }
}

/// Runs `module` and reads every output as floats, batch axis dropped.
pub(crate) fn run_float(
    module: &dyn TorchScript,
    encoded: &Encoded,
    feeds: &[(&str, Feed)],
) -> Result<Vec<Output>> {
    run(module, encoded, feeds)?
        .into_iter()
        .map(float_output)
        .collect()
}

/// The engine a model function runs on.
pub(crate) enum Engine {
    /// An ONNX Runtime session, fed by input name.
    Onnx(onnx::OnnxModel, Vec<(String, Feed)>),
    /// A TorchScript module, fed as the translator names its arrays.
    Torch(Box<dyn TorchScript>, &'static [(&'static str, Feed)]),
}

impl Engine {
    /// Runs one encoded text; every output comes back as floats, without
    /// its batch axis.
    pub(crate) fn unbatched(&self, encoded: &Encoded) -> Result<Vec<Output>> {
        match self {
            Engine::Onnx(model, feeds) => model
                .run(encoded, feeds)?
                .into_iter()
                .map(|o| {
                    Ok(Output {
                        name: o.name,
                        tensor: onnx::unbatch(o.tensor)?,
                    })
                })
                .collect(),
            Engine::Torch(module, feeds) => run_float(module.as_ref(), encoded, feeds),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A module that records its inputs and answers with fixed outputs.
    pub(crate) struct Fake {
        pub(crate) seen: Mutex<Vec<(String, Vec<i64>)>>,
        pub(crate) outputs: Vec<TorchOutput>,
    }

    impl Fake {
        pub(crate) fn new(outputs: Vec<TorchOutput>) -> Fake {
            Fake {
                seen: Mutex::new(Vec::new()),
                outputs,
            }
        }
    }

    impl TorchScript for Fake {
        fn forward(&self, inputs: &[Input<'_>]) -> Result<Vec<TorchOutput>> {
            let mut seen = self.seen.lock().unwrap();
            seen.clear();
            seen.extend(
                inputs
                    .iter()
                    .map(|i| (i.name.to_string(), i.values.to_vec())),
            );
            Ok(self.outputs.clone())
        }
    }

    pub(crate) fn out(name: &str, group: u32, shape: &[usize], data: Data) -> TorchOutput {
        TorchOutput {
            name: name.into(),
            group,
            shape: shape.to_vec(),
            data,
        }
    }

    fn encoded() -> Encoded {
        Encoded {
            ids: vec![5, 6],
            attention_mask: vec![1, 1],
            type_ids: vec![0, 0],
        }
    }

    #[test]
    fn feeds_named_arrays_and_drops_the_batch_axis() {
        let fake = Fake::new(vec![
            out("a", 1, &[1, 2], Data::F32(vec![1.0, 2.0])),
            out("b", 1, &[1, 2], Data::I64(vec![3, 4])),
        ]);
        let outputs = run(&fake, &encoded(), DICT_FEEDS).unwrap();
        assert_eq!(
            *fake.seen.lock().unwrap(),
            vec![
                ("input1.input_ids".to_string(), vec![5, 6]),
                ("input1.attention_mask".to_string(), vec![1, 1])
            ]
        );
        assert_eq!(outputs[0].shape, vec![2]);
        assert_eq!(outputs[1].data, Data::I64(vec![3, 4]));
        assert_eq!(outputs[1].data.type_name(), "INT64");
        assert!(!outputs[1].data.is_empty());
    }

    #[test]
    fn refuses_unbatched_and_non_float_outputs() {
        let fake = Fake::new(vec![out("", 0, &[2], Data::F32(vec![1.0, 2.0]))]);
        assert!(matches!(
            run(&fake, &encoded(), DICT_FEEDS),
            Err(MlError::Model(_))
        ));
        let fake = Fake::new(vec![out("ids", 1, &[1, 2], Data::I64(vec![1, 2]))]);
        let e = run_float(&fake, &encoded(), DICT_FEEDS).unwrap_err();
        assert!(e.to_string().contains("[ids] is INT64"), "{e}");
        let fake = Fake::new(vec![out("x", 0, &[1], Data::F32(vec![0.5]))]);
        let engine = Engine::Torch(Box::new(fake), DICT_FEEDS);
        let outputs = engine.unbatched(&encoded()).unwrap();
        assert_eq!(outputs[0].tensor.shape, Vec::<usize>::new());
        assert_eq!(outputs[0].tensor.data, vec![0.5]);
    }
}
