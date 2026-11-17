#![cfg(test)]
//! The built-in pre- and post-process functions, and the payload and
//! header checks, as ml-commons' connector helpers behave.
use super::*;

fn docs(items: &[&str]) -> Input {
    Input::TextDocs(items.iter().map(|d| Some((*d).to_string())).collect())
}

fn sim(query: &str, docs: &[&str]) -> Input {
    Input::TextSimilarity {
        query: query.into(),
        docs: docs.iter().map(|d| (*d).to_string()).collect(),
    }
}

fn pre(function: &str, input: &Input) -> Result<Vec<(String, Option<String>)>> {
    builtin_pre(&format!("connector.pre_process.{function}"), input)
}

fn message(r: Result<impl std::fmt::Debug>) -> String {
    r.expect_err("an error").to_string()
}

fn num(x: &str) -> Json {
    Json::Number(x.into())
}

fn floats(xs: &[&str]) -> Json {
    Json::Array(xs.iter().map(|x| num(x)).collect())
}

#[test]
fn embedding_pre_process_functions() {
    let input = docs(&["a", "b\"c"]);
    for (f, key) in [
        ("openai.embedding", "input"),
        ("default.embedding", "input"),
        ("cohere.embedding", "texts"),
    ] {
        let p = pre(f, &input).unwrap();
        assert_eq!(p.len(), 1, "{f}");
        assert_eq!(p[0].0, key, "{f}");
        let v = p[0].1.clone().unwrap();
        assert!(v.starts_with('[') && v.contains("\"a\""), "{f}: {v}");
    }
    for f in ["bedrock.embedding", "bedrock.nova.text_embedding"] {
        assert_eq!(
            pre(f, &input).unwrap(),
            vec![("inputText".to_string(), Some("a".to_string()))]
        );
        assert!(message(pre(f, &docs(&[]))).contains("out of bounds"));
    }
    assert_eq!(
        pre("bedrock.nova.image_embedding", &input).unwrap()[0].0,
        "inputImage"
    );
    assert!(message(pre("bedrock.nova.image_embedding", &docs(&[]))).contains("No input image"));
    // A remote model's own parameters pass through whatever the function.
    let remote = Input::Remote(vec![("k".into(), Some("v".into()))]);
    assert_eq!(
        pre("openai.embedding", &remote).unwrap(),
        vec![("k".to_string(), Some("v".to_string()))]
    );
    assert!(message(pre("openai.embedding", &sim("q", &["d"]))).contains("TextDocsInputDataSet"));
    assert!(message(builtin_pre("connector.pre_process.nope", &input)).contains("unknown"));
}

#[test]
fn multimodal_pre_process_functions() {
    let both = docs(&["text", "image"]);
    let p = pre("bedrock.multimodal_embedding", &both).unwrap();
    assert_eq!(p[0], ("inputText".to_string(), Some("text".to_string())));
    assert_eq!(p[1], ("inputImage".to_string(), Some("image".to_string())));
    assert_eq!(
        pre("bedrock.multimodal_embedding", &docs(&["only"]))
            .unwrap()
            .len(),
        1
    );
    let none = Input::TextDocs(vec![None]);
    assert!(message(pre("bedrock.multimodal_embedding", &none)).contains("No input text or image"));
    assert!(
        message(pre("bedrock.multimodal_embedding", &docs(&[]))).contains("No input text or image")
    );
    let p = pre("cohere.multimodal_embedding", &both).unwrap();
    assert_eq!(p[0].0, "images");
    assert!(message(pre("cohere.multimodal_embedding", &none)).contains("No image provided"));
    assert!(message(pre("cohere.multimodal_embedding", &docs(&[]))).contains("No image provided"));
}

#[test]
fn rerank_pre_process_functions() {
    let input = sim("what", &["d1", "d2"]);
    for f in ["cohere.rerank", "default.rerank"] {
        let p = pre(f, &input).unwrap();
        let keys: Vec<&str> = p.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["query", "documents", "top_n"]);
        assert_eq!(p[0].1.as_deref(), Some("what"));
        assert_eq!(p[2].1.as_deref(), Some("2"));
        assert!(p[1].1.as_deref().unwrap().contains("d2"));
    }
    let p = pre("bedrock.rerank", &input).unwrap();
    let keys: Vec<&str> = p.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["queries", "sources", "numberOfResults"]);
    let queries = Json::parse(p[0].1.as_deref().unwrap()).unwrap();
    assert_eq!(queries.to_compact(), p[0].1.clone().unwrap());
    assert!(p[0].1.as_deref().unwrap().contains("textQuery"));
    assert!(p[1].1.as_deref().unwrap().contains("inlineDocumentSource"));
    assert_eq!(p[2].1.as_deref(), Some("2"));
    for f in ["cohere.rerank", "bedrock.rerank"] {
        assert!(message(pre(f, &docs(&["x"]))).contains("TextSimilarityInputDataSet"));
    }
}

#[test]
fn embedding_post_process_functions() {
    let post = |f: &str, v: Json| builtin_post(&format!("connector.post_process.{f}"), v, None);
    for f in [
        "bedrock.embedding",
        "bedrock_v2.embedding.float",
        "bedrock_v2.embedding.binary",
    ] {
        let t = post(f, floats(&["1", "2", "3"])).unwrap();
        assert_eq!(t.len(), 1);
        match &t[0] {
            Tensor::Numbers {
                name,
                data_type,
                shape,
                data,
            } => {
                assert_eq!(name, "sentence_embedding");
                assert_eq!(data_type, "FLOAT32");
                assert_eq!(shape, &[3]);
                assert_eq!(data.len(), 3);
            }
            other => panic!("{other:?}"),
        }
        assert!(message(post(f, Json::Null)).contains("not a List"));
        assert!(
            message(post(f, Json::Array(vec![Json::String("x".into())])))
                .contains("containing Float values")
        );
    }
    let typed = builtin_post(
        "connector.post_process.bedrock.embedding",
        floats(&["1"]),
        Some("INT8"),
    )
    .unwrap();
    assert!(matches!(&typed[0], Tensor::Numbers { data_type, .. } if data_type == "INT8"));
    // The default (openai-style): a list of embeddings.
    let t = builtin_post(
        "connector.post_process.default.embedding",
        Json::Array(vec![floats(&["1", "2"]), floats(&["3", "4"])]),
        None,
    )
    .unwrap();
    assert_eq!(t.len(), 2);
    assert!(message(builtin_post("x", Json::Null, None)).contains("not a List"));
    assert!(message(builtin_post("x", floats(&["1"]), None)).contains("List of Float values"));
    assert!(
        message(builtin_post("x", Json::Array(vec![floats(&[])]), None))
            .contains("non-empty List containing Float values")
    );
    assert!(message(builtin_post(
        "x",
        Json::Array(vec![floats(&["1"]), Json::Null]),
        None
    ))
    .contains("List of Float values"));
    assert!(builtin_post("x", Json::Array(vec![]), None)
        .unwrap()
        .is_empty());
}

fn result(index: &str, key: &str, score: &str) -> Json {
    Json::Object(vec![("index".into(), num(index)), (key.into(), num(score))])
}

#[test]
fn rerank_post_process_functions() {
    let results = Json::Array(vec![
        result("1", "relevance_score", "0.9"),
        result("0", "relevance_score", "0.1"),
    ]);
    let t = builtin_post("connector.post_process.cohere.rerank", results, None).unwrap();
    assert_eq!(t.len(), 2);
    match &t[0] {
        Tensor::Numbers {
            name, data, shape, ..
        } => {
            assert_eq!(name, "similarity");
            assert_eq!(shape, &[1]);
            assert_eq!(data, &[num("0.1")]);
        }
        other => panic!("{other:?}"),
    }
    let b = builtin_post(
        "connector.post_process.bedrock.rerank",
        Json::Array(vec![result("0", "relevanceScore", "0.5")]),
        None,
    )
    .unwrap();
    assert_eq!(b.len(), 1);
    assert!(message(builtin_post(
        "connector.post_process.bedrock.rerank",
        Json::Array(vec![]),
        None
    ))
    .contains("input is empty"));
    assert!(message(builtin_post(
        "connector.post_process.bedrock.rerank",
        Json::Array(vec![result("0", "relevance_score", "1")]),
        None
    ))
    .contains("index and relevanceScore"));
    assert!(message(builtin_post(
        "connector.post_process.default.rerank",
        Json::Array(vec![result("0", "relevanceScore", "1")]),
        None
    ))
    .contains("index and relevance_score"));
    assert!(message(builtin_post(
        "connector.post_process.cohere.rerank",
        Json::Array(vec![result("7", "relevance_score", "1")]),
        None
    ))
    .contains("Index 7 out of bounds for length 1"));
    assert!(message(builtin_post(
        "connector.post_process.cohere.rerank",
        Json::Array(vec![result("x", "relevance_score", "1")]),
        None
    ))
    .contains("bad rerank index"));
    assert!(message(builtin_post(
        "connector.post_process.cohere.rerank",
        Json::Null,
        None
    ))
    .contains("not a List"));
}

#[test]
fn payload_and_header_checks() {
    assert!(validate_payload(None).is_ok());
    assert!(validate_payload(Some("{\"a\":1}")).is_ok());
    let e = message(validate_payload(Some(
        "{\"a\":\"${parameters.x}\",\"b\":\"${parameters.y}\",\"c\":\"${parameters.\"}",
    )));
    assert!(e.contains("not filled in payload: x, y"), "{e}");
    let h = |k: &str, v: &str| vec![(k.to_string(), v.to_string())];
    assert!(validate_headers(&h("a", "b\tc")).is_ok());
    assert!(message(validate_headers(&h("a", "b\nc"))).contains("control character"));
    assert!(message(validate_headers(&h("a", "b\u{7f}"))).contains("control character"));
    assert!(message(validate_headers(&h("a", &"x".repeat(9000)))).contains("exceeds 8KB"));
    let many: Vec<(String, String)> = (0..9)
        .map(|i| (format!("h{i}"), "x".repeat(8000)))
        .collect();
    assert!(message(validate_headers(&many)).contains("exceeds 64KB"));
}

#[test]
fn escape_helpers_and_null_defaults() {
    assert!(calls_escape("x = escape(a)"));
    assert!(!calls_escape("x = a"));
    assert!(defines_escape("String escape(def input) { return input; }"));
    assert!(!defines_escape("x = escape(a)"));
    let with = add_default_escape("return escape(params.a);");
    assert!(with.contains("String escape(def input)"));
    assert_eq!(add_default_escape("return 1;"), "return 1;");
    let own = "String escape(def i) { return i; } return escape(params.a);";
    assert_eq!(add_default_escape(own), own);
    let keys = null_default_keys("{\"a\":\"${parameters.a:-null}\",\"b\":\"${parameters.b}\"}");
    assert_eq!(keys, ["a"]);
}
