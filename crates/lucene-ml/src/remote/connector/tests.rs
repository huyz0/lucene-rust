#![cfg(test)]
use super::*;

fn connector(protocol: &str, action: &str, extra: &str) -> Connector {
    Connector::from_json(&format!(
            r#"{{"name":"c","protocol":"{protocol}","parameters":{{"model":"m","endpoint":"localhost:8791"{extra}}},
               "credential":{{"key":"sk-secret","access_key":"AKIDEXAMPLE","secret_key":"wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"}},
               "actions":[{action}]}}"#
        ))
        .unwrap()
}

fn call(
    c: &Connector,
    algorithm: &str,
    input: &Input,
    extra: &[(String, Option<String>)],
) -> Result<Call> {
    match c.prepare("predict", algorithm, input, extra, None, "20261111T022922Z")? {
        Prepared::Call(call) => Ok(*call),
        Prepared::Script(s) => panic!("script {s:?}"),
    }
}

fn body(c: &Call) -> String {
    String::from_utf8(c.request.body.clone()).unwrap()
}

fn header<'a>(c: &'a Call, k: &str) -> Option<&'a str> {
    c.request
        .headers
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(k))
        .map(|(_, v)| v.as_str())
}

fn docs(d: &[&str]) -> Input {
    Input::TextDocs(d.iter().map(|s| Some(s.to_string())).collect())
}

fn remote(p: &[(&str, &str)]) -> Input {
    Input::Remote(
        p.iter()
            .map(|(k, v)| (k.to_string(), Some(v.to_string())))
            .collect(),
    )
}

const OPENAI: &str = r#"{"action_type":"predict","method":"POST","url":"http://${parameters.endpoint}/v1/openai/embeddings",
        "headers":{"Authorization":"Bearer ${credential.key}"},
        "request_body":"{ \"input\": ${parameters.input}, \"model\": \"${parameters.model}\" }",
        "pre_process_function":"connector.pre_process.openai.embedding","post_process_function":"connector.post_process.openai.embedding"}"#;

#[test]
fn builds_requests_as_stock_sent_them() {
    let c = connector("http", OPENAI, "");
    let parts = c.split("predict", docs(&["hello world", "b"])).unwrap();
    assert_eq!(parts.len(), 1);
    let r = call(&c, "TEXT_EMBEDDING", &parts[0], &[]).unwrap();
    assert_eq!(
        body(&r),
        r#"{ "input": ["hello world","b"], "model": "m" }"#
    );
    assert_eq!(r.request.url, "http://localhost:8791/v1/openai/embeddings");
    assert_eq!(header(&r, "authorization"), Some("Bearer sk-secret"));
    assert_eq!(header(&r, "content-length"), Some("46"));
    assert_eq!(header(&r, "content-type"), Some("application/json"));
    let r = call(&c, "REMOTE", &remote(&[("input", "[\"hello\"]")]), &[]).unwrap();
    assert_eq!(body(&r), r#"{ "input": ["hello"], "model": "m" }"#);

    let bedrock = r#"{"action_type":"predict","method":"POST","url":"http://${parameters.endpoint}/bedrock/model/amazon.titan-embed-text-v1/invoke",
            "headers":{"content-type":"application/json","x-amz-content-sha256":"required"},
            "request_body":"{ \"inputText\": \"${parameters.inputText}\" }",
            "pre_process_function":"connector.pre_process.bedrock.embedding","post_process_function":"connector.post_process.bedrock.embedding"}"#;
    let c = connector(
        "aws_sigv4",
        bedrock,
        r#","region":"us-east-1","service_name":"bedrock""#,
    );
    let parts = c
        .split("predict", docs(&["hello \"q\" world", "b"]))
        .unwrap();
    assert_eq!(parts.len(), 2);
    let r = call(&c, "TEXT_EMBEDDING", &parts[0], &[]).unwrap();
    assert_eq!(body(&r), r#"{ "inputText": "hello \"q\" world" }"#);
    assert!(header(&r, "authorization")
        .unwrap()
        .ends_with("Signature=3436fa765d1e2479a91a756ad4c36a4665684a86d88a8ab4fda8603e556444df"));
}

const GENERIC: &str = r#"{"action_type":"predict","method":"POST","url":"http://${parameters.endpoint}/gen/${parameters.model}",
        "request_body":"{ \"q\": \"${parameters.q}\", \"n\": ${parameters.n:-null}, \"d\": \"${parameters.d:-dflt}\" }"}"#;

#[test]
fn fills_remote_parameters() {
    let c = connector("http", GENERIC, "");
    let r = call(&c, "REMOTE", &remote(&[("q", "a \"b\"\nc/é")]), &[]).unwrap();
    assert_eq!(
        body(&r),
        r#"{ "q": "a \"b\"\nc\/\u00E9", "n": null, "d": "dflt" }"#
    );
    assert_eq!(r.request.url, "http://localhost:8791/gen/m");
    let e = call(&c, "REMOTE", &remote(&[]), &[])
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("Some parameter placeholder not filled in payload: q"),
        "{e}"
    );
    let skip = [(
        "skip_validating_missing_parameters".to_string(),
        Some("true".to_string()),
    )];
    assert!(body(&call(&c, "REMOTE", &remote(&[]), &skip).unwrap()).contains("${parameters.q}"));
    let r = call(
        &c,
        "REMOTE",
        &remote(&[("q", "x\"y"), ("no_escape_params", "q")]),
        &[],
    )
    .unwrap_err();
    assert!(r.to_string().contains("Invalid payload"), "{r}");
    assert!(c
        .prepare("nope", "REMOTE", &remote(&[]), &[], None, "")
        .unwrap_err()
        .to_string()
        .contains("no nope action found"));
}

#[test]
fn reads_responses() {
    let c = connector("http", OPENAI, "");
    let r = call(&c, "REMOTE", &remote(&[("input", "[\"a\"]")]), &[]).unwrap();
    let body = r#"{"data":[{"embedding":[0.1,0.25,-5e-5]},{"embedding":[1,2]}]}"#;
    let Parsed::Tensors(t) = c.parse("predict", &r, body, None).unwrap() else {
        panic!()
    };
    assert_eq!(t.len(), 2);
    assert_eq!(
        t[0],
        Tensor::Numbers {
            name: "sentence_embedding".into(),
            data_type: "FLOAT32".into(),
            shape: vec![3],
            data: ["0.1", "0.25", "-5e-5"]
                .iter()
                .map(|n| Json::Number(n.to_string()))
                .collect()
        }
    );
    assert!(c
        .parse("predict", &r, r#"{"data":[{"embedding":["x"]}]}"#, None)
        .is_err());

    let c = connector("http", GENERIC, "");
    let r = call(&c, "REMOTE", &remote(&[("q", "x")]), &[]).unwrap();
    let body = r#"{"echo":"e","nested":{"vals":[1,2.5,"s"]}}"#;
    let Parsed::Tensors(t) = c.parse("predict", &r, body, None).unwrap() else {
        panic!()
    };
    assert_eq!(t, vec![Tensor::Map(Json::parse(body).unwrap())]);
    let filtered = [
        ("q".to_string(), Some("x".to_string())),
        (
            "response_filter".to_string(),
            Some("$.nested.vals".to_string()),
        ),
    ];
    let r = call(&c, "REMOTE", &Input::Remote(filtered.to_vec()), &[]).unwrap();
    let Parsed::Tensors(t) = c.parse("predict", &r, body, None).unwrap() else {
        panic!()
    };
    assert_eq!(
        t[0],
        Tensor::Map(Json::parse(r#"{"response":[1,2.5,"s"]}"#).unwrap())
    );
    let missing = [
        ("q".to_string(), Some("x".to_string())),
        ("response_filter".to_string(), Some("$.nope".to_string())),
    ];
    let r = call(&c, "REMOTE", &Input::Remote(missing.to_vec()), &[]).unwrap();
    let Parsed::Tensors(t) = c.parse("predict", &r, body, None).unwrap() else {
        panic!()
    };
    assert_eq!(t[0], Tensor::Map(Json::parse(body).unwrap()));
    let Parsed::Tensors(t) = c.parse("predict", &r, "[1]", None).unwrap() else {
        panic!()
    };
    assert_eq!(
        t[0],
        Tensor::Map(Json::parse(r#"{"response":[1]}"#).unwrap())
    );
}

#[test]
fn hands_scripts_to_the_caller() {
    let painless = r#"{"action_type":"predict","method":"POST","url":"http://h/pl","request_body":"{\"q\": ${parameters.q}}",
            "pre_process_function":"return params.text_docs[0];","post_process_function":"return escape(params.x);"}"#;
    let c = connector("http", painless, "");
    let input = docs(&["h\"i"]);
    let Prepared::Script(s) = c
        .prepare("predict", "TEXT_EMBEDDING", &input, &[], None, "")
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(
        s.params,
        r#"{"algorithm":"TEXT_EMBEDDING","text_docs":["h\\\"i"]}"#
    );
    assert_eq!(s.source, "return params.text_docs[0];");
    let quoted = r#"{"parameters":{"q":"\"hi\"","n":1}}"#;
    let e = c
        .prepare("predict", "TEXT_EMBEDDING", &input, &[], Some(quoted), "")
        .unwrap_err()
        .to_string();
    assert!(e.contains(r#"Invalid payload: {"q": \"hi\"}"#), "{e}");
    let out = r#"{"parameters":{"q":"hi","n":1}}"#;
    let Prepared::Call(r) = c
        .prepare("predict", "TEXT_EMBEDDING", &input, &[], Some(out), "")
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(body(&r), r#"{"q": hi}"#);
    assert_eq!(r.parameters.get("n"), Some(&Some("1.0".to_string())));
    let Parsed::Script(s) = c.parse("predict", &r, "[1]", None).unwrap() else {
        panic!()
    };
    assert!(s.source.starts_with("\n    String escape(def input)"));
    assert_eq!(s.params, r#"{"result":[1]}"#);
    let tensors = r#"[{"name":"sentence_embedding","data":[1]}]"#;
    let Parsed::Tensors(t) = c.parse("predict", &r, "[1]", Some(tensors)).unwrap() else {
        panic!()
    };
    assert!(matches!(&t[0], Tensor::Raw(_)));
    let Parsed::Tensors(t) = c.parse("predict", &r, "[1]", Some("plain")).unwrap() else {
        panic!()
    };
    assert_eq!(
        t[0],
        Tensor::Map(Json::parse(r#"{"response":"plain"}"#).unwrap())
    );
    assert!(c.parse("predict", &r, "x", None).is_err());
}

#[test]
fn splits_and_reranks() {
    let rerank = r#"{"action_type":"predict","method":"POST","url":"http://h/r",
            "request_body":"{\"query\":\"${parameters.query}\",\"documents\":${parameters.documents},\"top_n\":${parameters.top_n}}",
            "pre_process_function":"connector.pre_process.cohere.rerank","post_process_function":"connector.post_process.cohere.rerank"}"#;
    let c = connector("http", rerank, r#","input_docs_processed_step_size":"2""#);
    assert_eq!(c.split("predict", docs(&["a", "b", "c"])).unwrap().len(), 2);
    let input = Input::TextSimilarity {
        query: "q".into(),
        docs: vec!["a".into(), "b".into()],
    };
    let r = call(&c, "TEXT_SIMILARITY", &input, &[]).unwrap();
    assert_eq!(body(&r), r#"{"query":"q","documents":["a","b"],"top_n":2}"#);
    let resp =
        r#"{"results":[{"index":1,"relevance_score":0.5},{"index":0,"relevance_score":0.25}]}"#;
    let Parsed::Tensors(t) = c.parse("predict", &r, resp, None).unwrap() else {
        panic!()
    };
    assert!(
        matches!(&t[0], Tensor::Numbers { name, data, .. } if name == "similarity" && data[0] == Json::Number("0.25".into()))
    );
    let c = connector("http", rerank, r#","input_docs_processed_step_size":"0""#);
    assert!(c.split("predict", docs(&["a"])).is_err());
}

#[test]
fn checks_headers_and_escapes() {
    let bad = r#"{"action_type":"predict","method":"POST","url":"http://h/x","headers":{"Authorization":"${parameters.k}"},"request_body":"{}"}"#;
    let c = connector("http", bad, "");
    assert!(call(&c, "REMOTE", &remote(&[("k", "v")]), &[])
        .unwrap_err()
        .to_string()
        .contains("security reasons"));
    let unresolved = r#"{"action_type":"predict","method":"POST","url":"http://h/x","headers":{"X-T":"${parameters.k}"},"request_body":"{}"}"#;
    let c = connector("http", unresolved, "");
    assert!(call(&c, "REMOTE", &remote(&[]), &[])
        .unwrap_err()
        .to_string()
        .contains("unresolved placeholder"));
    assert_eq!(
        header(
            &call(&c, "REMOTE", &remote(&[("k", "v")]), &[]).unwrap(),
            "x-t"
        ),
        Some("v")
    );
    assert_eq!(
        header(
            &call(&c, "REMOTE", &remote(&[("k", "a\nb")]), &[]).unwrap(),
            "x-t"
        ),
        Some("a\\nb")
    );
    let raw = [("k".to_string(), Some("a\nb".to_string()))];
    assert!(call(&c, "REMOTE", &remote(&[]), &raw)
        .unwrap_err()
        .to_string()
        .contains("control character"));
    assert!(
        calls_escape("x = escape(params.a)")
            && !calls_escape("String escape(def a) {}")
            && !calls_escape("unescape(x)")
    );
    assert!(
        defines_escape("String escape(def input) { return input; }")
            && !defines_escape("escape(x)")
    );
    assert_eq!(
        add_default_escape("String escape(def i) {} escape(x)"),
        "String escape(def i) {} escape(x)"
    );
    assert_eq!(
        null_default_keys(r#"{"a":"${parameters.x:-null}","b":"${parameters.y}"}"#),
        vec!["x"]
    );
    let mut p: HashMap<String, Option<String>> = HashMap::new();
    p.insert("t".into(), Some("${parameters.v.toString()}".into()));
    p.insert("v".into(), Some("a\"b".into()));
    add_to_string_params(&mut p);
    assert_eq!(p["v.toString()"], Some("a\\\"b".into()));
    let mut fp = HashMap::new();
    fp.insert(
        "f".to_string(),
        Some("connector.post_process.openai.embedding".to_string()),
    );
    assert_eq!(
        fill_function(&fp, "${parameters.f}").unwrap(),
        "\"connector.post_process.openai.embedding\""
    );
    assert!(validate_headers(&[("k".into(), "v".repeat(9000))]).is_err());
}
