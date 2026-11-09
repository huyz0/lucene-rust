//! The vectors stock ml-commons gives for fixtures/ml's models
//! (fixtures/ml/expected.json, from scripts/ml-golden.py), for each pooling
//! mode, reproduced here.
//!
//! Needs the ONNX Runtime library: `LUCENE_ML_ONNXRUNTIME` names it (CI extracts it
//! from the onnxruntime Maven jar). ONNX Runtime's own loader does not read that
//! variable, so the tests also check that `runtime::init` loads the library it is
//! given. Without it these tests say so and pass.

use std::path::{Path, PathBuf};

use lucene_ml::text_embedding::{Config, Pooling, TextEmbeddingModel};
use lucene_ml::{runtime, MlError};
use serde_json::Value;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/ml")
}

fn runtime_loaded() -> bool {
    match std::env::var_os("LUCENE_ML_ONNXRUNTIME") {
        Some(lib) => {
            runtime::init(Path::new(&lib), 2).expect("ONNX Runtime loads");
            true
        }
        None => {
            eprintln!("LUCENE_ML_ONNXRUNTIME is not set: skipping the ONNX model tests");
            false
        }
    }
}

fn expected() -> Value {
    let text = std::fs::read_to_string(fixtures().join("expected.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

#[test]
fn matches_stock_for_every_model_and_pooling_mode() {
    if !runtime_loaded() {
        return;
    }
    let expected = expected();
    let texts: Vec<&str> = expected["texts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    let tokenizer = std::fs::read(fixtures().join("tokenizer.json")).unwrap();
    let mut checked = 0;
    for case in expected["cases"].as_array().unwrap() {
        let name = case["model"].as_str().unwrap();
        let config = Config {
            model_type: case["model_type"].as_str().unwrap().to_string(),
            pooling: Pooling::from_name(case["pooling"].as_str().unwrap()).unwrap(),
            normalize: case["normalize"].as_bool().unwrap(),
        };
        let model = TextEmbeddingModel::load(
            &fixtures().join(format!("{name}.onnx")),
            &tokenizer,
            &config,
        )
        .unwrap();
        for (text, want) in texts.iter().zip(case["outputs"].as_array().unwrap()) {
            let got = model.embed(&[text]);
            if want.get("error").is_some() {
                assert!(got.is_err(), "{name} {config:?} {text:?}: stock refused it");
                continue;
            }
            let got = got.unwrap().remove(0);
            let shape: Vec<usize> = want["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(got.shape, shape, "{name} {config:?} {text:?}");
            let data = want["data"].as_array().unwrap();
            assert_eq!(got.data.len(), data.len());
            for (g, w) in got.data.iter().zip(data) {
                let w = w.as_f64().unwrap() as f32;
                assert!(
                    (g - w).abs() <= 2e-5 * w.abs().max(1.0),
                    "{name} {config:?} {text:?}: {g} vs {w}"
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 100, "only {checked} vectors compared");
}

#[test]
fn refuses_a_missing_model_and_a_wrong_input_set() {
    if !runtime_loaded() {
        return;
    }
    let tokenizer = std::fs::read(fixtures().join("tokenizer.json")).unwrap();
    let config = |model_type: &str| Config {
        model_type: model_type.into(),
        pooling: Pooling::Mean,
        normalize: true,
    };
    let missing =
        TextEmbeddingModel::load(&fixtures().join("absent.onnx"), &tokenizer, &config("bert"));
    assert!(matches!(missing, Err(MlError::Model(_))));
    // A BERT graph needs token_type_ids, which a `roberta` config does not send.
    let model = TextEmbeddingModel::load(
        &fixtures().join("tiny-bert.onnx"),
        &tokenizer,
        &config("roberta"),
    )
    .unwrap();
    assert!(matches!(model.embed(&["hello"]), Err(MlError::Model(_))));
}

#[test]
fn a_second_init_keeps_the_first_runtime() {
    if !runtime_loaded() {
        return;
    }
    assert!(runtime::is_loaded());
    runtime::init(Path::new("/nonexistent/libonnxruntime.so"), 1).unwrap();
}
