#![cfg(test)]
use super::*;
use crate::torchscript::tests::{out, Fake};

/// A word-level tokenizer: `a` is 2, `b` 3, with `[CLS]` (1) first.
fn tokenizer_json() -> Vec<u8> {
    br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},
            "post_processor":{"type":"TemplateProcessing",
              "single":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],
              "pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],
              "special_tokens":{"[CLS]":{"id":"[CLS]","ids":[1],"tokens":["[CLS]"]}}},
            "decoder":null,
            "model":{"type":"WordLevel","vocab":{"[UNK]":0,"[CLS]":1,"a":2,"b":3},"unk_token":"[UNK]"}}"#
            .to_vec()
}

#[test]
fn sentence_transformers_outputs_pass_through() {
    let outputs = vec![
        out("input_ids", 1, &[1, 2], Data::I64(vec![1, 2])),
        out("sentence_embedding", 1, &[1, 2], Data::F32(vec![0.5, 1.5])),
    ];
    let fake = Fake::new(outputs);
    let model =
        TorchTextEmbeddingModel::sentence_transformers(Box::new(fake), &tokenizer_json()).unwrap();
    let got = model.embed("a").unwrap();
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].shape, vec![2]);
    assert_eq!(got[1].data, Data::F32(vec![0.5, 1.5]));
}

#[test]
fn hugging_face_pools_the_token_embeddings() {
    // Two tokens ([CLS] a), two dimensions, behind an unnamed first output.
    let tokens = out("", 0, &[1, 2, 2], Data::F32(vec![1.0, 2.0, 3.0, 4.0]));
    let pooled = out("pooler_output", 0, &[1, 2], Data::F32(vec![3.0, 4.0]));
    let fake = Fake::new(vec![tokens.clone(), pooled.clone()]);
    let mean = TorchTextEmbeddingModel::hugging_face(
        Box::new(fake),
        &tokenizer_json(),
        Pooling::Mean,
        false,
    )
    .unwrap();
    let got = mean.embed("a").unwrap();
    assert_eq!(got[0].name, "sentence_embedding");
    assert_eq!(got[0].shape, vec![1, 2]);
    assert_eq!(got[0].data, Data::F32(vec![2.0, 3.0]));

    let none = |outputs: Vec<TorchOutput>| {
        let model = TorchTextEmbeddingModel::hugging_face(
            Box::new(Fake::new(outputs)),
            &tokenizer_json(),
            Pooling::None,
            true,
        )
        .unwrap();
        model.embed("a").unwrap().remove(0).data
    };
    assert_eq!(
        none(vec![tokens.clone(), pooled]),
        Data::F32(vec![0.6, 0.8])
    );
    let second = out("", 0, &[1, 2], Data::F32(vec![4.0, 3.0]));
    assert_eq!(
        none(vec![tokens.clone(), second]),
        Data::F32(vec![0.8, 0.6])
    );

    let cls = TorchTextEmbeddingModel::hugging_face(
        Box::new(Fake::new(vec![out("", 0, &[1, 2], Data::I64(vec![1, 2]))])),
        &tokenizer_json(),
        Pooling::Cls,
        false,
    )
    .unwrap();
    assert!(matches!(cls.embed("a"), Err(MlError::Model(_))));
    let empty = TorchTextEmbeddingModel::hugging_face(
        Box::new(Fake::new(vec![])),
        &tokenizer_json(),
        Pooling::Cls,
        false,
    )
    .unwrap();
    assert!(matches!(empty.embed("a"), Err(MlError::Model(_))));
}

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
