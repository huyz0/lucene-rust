#![cfg(test)]
use super::*;

#[test]
fn orders_libraries_as_djl_loads_them() {
    let dir = lucene_util::test_support::TempDir::new("ml-torch-order");
    for name in [
        "libtorch.so",
        "libtorch_cpu.so",
        "libc10.so",
        "libgomp-98b21ff3.so.1",
        "libtorch_python.so",
        "0.31.1-libdjl_torch.so",
        "libopenblasp-r0.so",
        "libarm_compute_core.so",
        "notes.txt",
        "libfoo.so.x",
    ] {
        std::fs::write(dir.path().join(name), b"").unwrap();
    }
    let names: Vec<String> = load_order(dir.path())
        .unwrap()
        .into_iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        [
            "libc10.so",
            "libgomp-98b21ff3.so.1",
            "libopenblasp-r0.so",
            "libarm_compute_core.so",
            "libtorch_cpu.so",
            "libtorch.so"
        ]
    );
    assert!(load_order(&dir.path().join("missing")).is_err());
}

#[test]
fn refuses_a_directory_without_libtorch() {
    let dir = lucene_util::test_support::TempDir::new("ml-torch-empty");
    assert!(matches!(check_dir(dir.path()), Err(MlError::Runtime(_))));
    std::fs::write(dir.path().join("libtorch.so"), b"").unwrap();
    assert!(check_dir(dir.path()).is_ok());
}

/// The TorchScript fixtures carry the weights of the ONNX ones, so they
/// give the vectors stock recorded for those (fixtures/ml/expected.json),
/// when libtorch and the shims are available (`LUCENE_ML_LIBTORCH`,
/// `LUCENE_ML_TORCH_SHIMS`).
#[test]
fn torchscript_models_give_stock_vectors() {
    use lucene_ml::sparse_encoding::SparseEncodingModel;
    use lucene_ml::text_embedding::{Pooling, TorchTextEmbeddingModel};
    use lucene_ml::text_similarity::TextSimilarityModel;
    use serde_json::Value;

    let (Some(dir), Some(shims)) = (
        std::env::var_os("LUCENE_ML_LIBTORCH"),
        std::env::var_os("LUCENE_ML_TORCH_SHIMS"),
    ) else {
        eprintln!("LUCENE_ML_LIBTORCH or LUCENE_ML_TORCH_SHIMS is not set: skipping");
        return;
    };
    let shims = shims.into_string().unwrap();
    let shims: Vec<&Path> = shims.split(':').map(Path::new).collect();
    init(Path::new(&dir), &shims, 2).unwrap();
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/ml");
    let tokenizer = std::fs::read(fixtures.join("tokenizer.json")).unwrap();
    let expected: Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures.join("expected.json")).unwrap())
            .unwrap();
    let texts: Vec<&str> = expected["texts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    let module = |name: &str| Box::new(Module::load(&fixtures.join(name)).unwrap());
    let close = |g: f32, w: &Value| {
        let w = w.as_f64().unwrap() as f32;
        (g - w).abs() <= 1e-4 * w.abs().max(1.0)
    };
    let mut checked = 0;
    for case in expected["cases"].as_array().unwrap() {
        if case["model"] != "tiny-bert" {
            continue;
        }
        let pooling = Pooling::from_name(case["pooling"].as_str().unwrap()).unwrap();
        let normalize = case["normalize"].as_bool().unwrap();
        let model = TorchTextEmbeddingModel::hugging_face(
            module("tiny-hf.pt"),
            &tokenizer,
            pooling,
            normalize,
        )
        .unwrap();
        for (text, want) in texts.iter().zip(case["outputs"].as_array().unwrap()) {
            let got = model.embed(text);
            if want.get("error").is_some() {
                assert!(got.is_err(), "{pooling:?} {text:?}: stock refused it");
                continue;
            }
            let Data::F32(got) = &got.unwrap()[0].data else {
                panic!("not FLOAT32");
            };
            let want = want["data"].as_array().unwrap();
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(want) {
                assert!(close(*g, w), "{pooling:?} {text:?}: {g} vs {w}");
            }
            checked += 1;
        }
    }
    // The sentence-transformers trace shares tiny-bert's weights: its
    // sentence_embedding is stock's unnormalized mean pooling.
    let st =
        TorchTextEmbeddingModel::sentence_transformers(module("tiny-st.pt"), &tokenizer).unwrap();
    let mean = expected["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["model"] == "tiny-bert" && c["pooling"] == "mean" && c["normalize"] == false)
        .unwrap();
    for (text, want) in texts.iter().zip(mean["outputs"].as_array().unwrap()) {
        let got = st.embed(text);
        if want.get("error").is_some() {
            assert!(got.is_err(), "{text:?}: stock refused it");
            continue;
        }
        let got = got.unwrap();
        let mut names: Vec<&str> = got.iter().map(|o| o.name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "attention_mask",
                "input_ids",
                "sentence_embedding",
                "token_embeddings"
            ]
        );
        assert!(got.iter().all(|o| o.group == 1));
        let ids = got.iter().find(|o| o.name == "input_ids").unwrap();
        assert_eq!(ids.data.type_name(), "INT64");
        let embedding = got.iter().find(|o| o.name == "sentence_embedding").unwrap();
        let Data::F32(values) = &embedding.data else {
            panic!("not FLOAT32");
        };
        let want = want["data"].as_array().unwrap();
        assert_eq!(values.len(), want.len());
        for (g, w) in values.iter().zip(want) {
            assert!(close(*g, w), "sentence_transformers {text:?}: {g} vs {w}");
        }
        checked += 1;
    }
    let sparse = SparseEncodingModel::torchscript(module("tiny-sparse-ts.pt"), &tokenizer).unwrap();
    for (text, want) in texts.iter().zip(
        expected["sparse_encoding"]["outputs"]["WORD"]
            .as_array()
            .unwrap(),
    ) {
        let got = sparse.encode(text, false);
        if want.get("error").is_some() {
            assert!(got.is_err(), "{text:?}: stock refused it");
            continue;
        }
        let got = got.unwrap();
        assert_eq!(got[0].name, "output");
        let want = want[0]["dataAsMap"]["response"][0].as_object().unwrap();
        assert_eq!(got[0].weights.len(), want.len(), "{text:?}");
        for w in &got[0].weights {
            assert!(close(w.weight, &want[&w.token]), "{text:?} {}", w.token);
        }
        checked += 1;
    }
    let cross = TextSimilarityModel::torchscript(module("tiny-cross-ts.pt"), &tokenizer).unwrap();
    let case = &expected["text_similarity"];
    let query = case["query"].as_str().unwrap();
    for (text, want) in texts.iter().zip(case["outputs"].as_array().unwrap()) {
        let got = cross.score(query, text).unwrap();
        assert_eq!(got[0].shape, vec![1]);
        assert!(close(got[0].data[0], &want[0]["data"][0]), "{text:?}");
        checked += 1;
    }
    assert!(checked >= 50, "only {checked} vectors compared");
}
