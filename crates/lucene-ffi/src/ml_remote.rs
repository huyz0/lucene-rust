//! Remote models for the plugin's ml-commons port: the connector work of a
//! predict call, over [`lucene_ml::remote`], and the credential encryption
//! ml-commons stores connectors with.
//!
//! The plugin opens a deployed remote model's connector behind a model
//! handle ([`ffi_ml_remote_open`], closed with
//! [`crate::ml::ffi_ml_model_close`]) and drives each call through
//! [`ffi_ml_remote_call_alloc`], a JSON request and answer:
//!
//! - `{"op":"split","action":A,"input":I}` answers `{"inputs":[I, ...]}`,
//!   the inputs one request each;
//! - `{"op":"prepare","action":A,"algorithm":F,"input":I,"parameters":{..},
//!   "script_output":S?,"amz_date":D}` answers
//!   `{"request":{"method","url","headers":[[k,v],..],"body"},"parameters":{..}}`,
//!   or `{"script":{"source","params"}}` when a pre-process script must run
//!   first (prepare again with its result as `script_output`);
//! - `{"op":"parse","action":A,"parameters":{..},"body":B,"script_output":S?}`
//!   reads a successful response, with the `parameters` its `prepare`
//!   answered, into `{"output":[tensor, ..]}` (as `ModelTensor` writes
//!   them), or `{"script":..}` for a post-process script.
//!
//! - `{"op":"google_token_request","now":T,"key":K?}`, for a `google_cloud`
//!   connector, answers `{"token_uri","body"}` (a service-account assertion
//!   to post, form-encoded) or, for application default credentials without
//!   a key file `K` (a service account's `private_key`, `client_email`,
//!   `token_uri`, or an `authorized_user`'s refresh token),
//!   `{"adc":true,"scope"}`.
//!
//! An input `I` is `{"text_docs":[..]}`, `{"query_text":q,"text_docs":[..]}`
//! or `{"parameters":{..}}`. Answers are UTF-8 JSON in a buffer to free with
//! [`crate::ffm_bridge::ffi_jvm_free_bytes`].

use std::sync::Arc;

use lucene_ml::remote::connector::{Connector, Input, Parsed, Prepared, Script};
use lucene_ml::remote::esdk;
use lucene_ml::remote::google;
use lucene_ml::remote::json::Json;
use lucene_ml::MlError;

use crate::error::{guard, set_last_error, FfiStatus};
use crate::ffm_bridge::{clear_out, give};
use crate::ml::{model, models, wrong_kind, Model};
use crate::raw::str_from_raw;
use crate::registry::lock_recovering;

/// A failure with ml-commons' own message, unprefixed (the plugin shows it
/// as stock does, unlike local models' errors): a caller's mistake as
/// [`FfiStatus::InvalidArgument`], anything else as [`FfiStatus::Search`].
fn ml_failure(e: MlError) -> FfiStatus {
    let (status, why) = match e {
        MlError::InvalidArgument(m) => (FfiStatus::InvalidArgument, m),
        MlError::Model(m) | MlError::Runtime(m) | MlError::Tokenizer(m) => (FfiStatus::Search, m),
    };
    set_last_error(why);
    status
}

fn invalid(why: impl Into<String>) -> FfiStatus {
    set_last_error(why.into());
    FfiStatus::InvalidArgument
}

/// Writes a new base64 master key to `*out_ptr`/`*out_len`.
///
/// # Safety
/// `out_ptr` and `out_len` must be valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_ml_master_key_alloc(
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len) }?;
        let key = esdk::generate_master_key().map_err(ml_failure)?;
        // SAFETY: caller contract.
        unsafe { give(key.into_bytes(), out_ptr, out_len) };
        Ok(())
    })
}

/// Encrypts the credential value `plain` under the base64 master key `key`;
/// the stored (base64) value goes to `*out_ptr`/`*out_len`.
///
/// # Safety
/// Each pointer must be valid for its length; `out_ptr` and `out_len` for
/// one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_ml_credential_encrypt_alloc(
    key: *const u8,
    key_len: usize,
    plain: *const u8,
    plain_len: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len) }?;
        // SAFETY: caller contract, for both.
        let (key, plain) = unsafe {
            (
                str_from_raw(key.cast(), key_len)?,
                str_from_raw(plain.cast(), plain_len)?,
            )
        };
        let stored = esdk::encrypt_credential(key, plain).map_err(ml_failure)?;
        // SAFETY: caller contract.
        unsafe { give(stored.into_bytes(), out_ptr, out_len) };
        Ok(())
    })
}

/// Decrypts the stored credential value `stored` under the base64 master key
/// `key`; the clear value goes to `*out_ptr`/`*out_len`.
///
/// # Safety
/// Each pointer must be valid for its length; `out_ptr` and `out_len` for
/// one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_ml_credential_decrypt_alloc(
    key: *const u8,
    key_len: usize,
    stored: *const u8,
    stored_len: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len) }?;
        // SAFETY: caller contract, for both.
        let (key, stored) = unsafe {
            (
                str_from_raw(key.cast(), key_len)?,
                str_from_raw(stored.cast(), stored_len)?,
            )
        };
        let plain = esdk::decrypt_credential(key, stored).map_err(ml_failure)?;
        // SAFETY: caller contract.
        unsafe { give(plain.into_bytes(), out_ptr, out_len) };
        Ok(())
    })
}

/// Opens the connector document `doc` (its `credential` values as stored,
/// decrypted here under the base64 master key `key`) as a remote model;
/// writes its handle to `*out_handle`.
///
/// # Safety
/// Each pointer must be valid for its length; `out_handle` for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_ml_remote_open(
    doc: *const u8,
    doc_len: usize,
    key: *const u8,
    key_len: usize,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract, for both.
        let (doc, key) = unsafe {
            (
                str_from_raw(doc.cast(), doc_len)?,
                str_from_raw(key.cast(), key_len)?,
            )
        };
        let mut connector = Connector::from_json(doc).map_err(ml_failure)?;
        for value in connector.credential.values_mut() {
            *value = esdk::decrypt_credential(key, value).map_err(ml_failure)?;
        }
        let handle =
            lock_recovering(models()).insert_checked(Arc::new(Model::Remote(connector)))?;
        // SAFETY: caller contract.
        unsafe { *out_handle = handle };
        Ok(())
    })
}

/// Runs one step of a remote call (see the module doc) on the remote model
/// `handle`; the answer goes to `*out_ptr`/`*out_len`.
///
/// # Safety
/// `request` must be valid for `len` bytes; `out_ptr` and `out_len` for one
/// write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_ml_remote_call_alloc(
    handle: u64,
    request: *const u8,
    len: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len) }?;
        // SAFETY: caller contract.
        let request = unsafe { str_from_raw(request.cast(), len) }?;
        let model = model(handle)?;
        let Model::Remote(connector) = &*model else {
            return Err(wrong_kind("remote"));
        };
        let answer = call(connector, request)?;
        // SAFETY: caller contract.
        unsafe { give(answer.into_bytes(), out_ptr, out_len) };
        Ok(())
    })
}

fn call(connector: &Connector, request: &str) -> Result<String, FfiStatus> {
    let req = Json::parse(request).ok_or_else(|| invalid("the remote call request is not JSON"))?;
    let text = |k: &str| match req.get(k) {
        Some(Json::String(s)) => Some(s.as_str()),
        _ => None,
    };
    let action = text("action").unwrap_or("predict");
    let answer = match text("op") {
        Some("split") => {
            let inputs = connector
                .split(action, input(req.get("input"))?)
                .map_err(ml_failure)?;
            Json::Object(vec![(
                "inputs".into(),
                Json::Array(inputs.iter().map(input_json).collect()),
            )])
        }
        Some("prepare") => {
            let extra = params(req.get("parameters"));
            let prepared = connector
                .prepare(
                    action,
                    text("algorithm").unwrap_or("REMOTE"),
                    &input(req.get("input"))?,
                    &extra,
                    text("script_output"),
                    text("amz_date").unwrap_or(""),
                )
                .map_err(ml_failure)?;
            match prepared {
                Prepared::Script(s) => script_json(&s),
                Prepared::Call(c) => {
                    let r = &c.request;
                    let body = String::from_utf8(r.body.clone())
                        .map_err(|_| invalid("the request body is not UTF-8"))?;
                    let headers = r
                        .headers
                        .iter()
                        .map(|(k, v)| {
                            Json::Array(vec![Json::String(k.clone()), Json::String(v.clone())])
                        })
                        .collect();
                    let mut parameters: Vec<(String, Json)> = c
                        .parameters
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone().map_or(Json::Null, Json::String)))
                        .collect();
                    parameters.sort_by(|a, b| a.0.cmp(&b.0));
                    Json::Object(vec![
                        (
                            "request".into(),
                            Json::Object(vec![
                                ("method".into(), Json::String(r.method.clone())),
                                ("url".into(), Json::String(r.url.clone())),
                                ("headers".into(), Json::Array(headers)),
                                ("body".into(), Json::String(body)),
                            ]),
                        ),
                        ("parameters".into(), Json::Object(parameters)),
                    ])
                }
            }
        }
        Some("parse") => {
            let parameters = params(req.get("parameters")).into_iter().collect();
            let call = lucene_ml::remote::connector::Call {
                request: lucene_ml::remote::connector::Request {
                    method: String::new(),
                    url: String::new(),
                    headers: Vec::new(),
                    body: Vec::new(),
                },
                parameters,
            };
            let body = text("body").unwrap_or("");
            match connector
                .parse(action, &call, body, text("script_output"))
                .map_err(ml_failure)?
            {
                Parsed::Script(s) => script_json(&s),
                Parsed::Tensors(t) => Json::Object(vec![(
                    "output".into(),
                    Json::Array(t.iter().map(|t| t.to_json()).collect()),
                )]),
            }
        }
        Some("google_token_request") => google_token_request(connector, &req)?,
        _ => return Err(invalid("unknown remote call op")),
    };
    Ok(answer.to_compact())
}

/// A `google_cloud` connector's sign-in: `{"adc":true,"scope":S}` when it
/// uses application default credentials, else the token endpoint and the
/// form body to post there, signed at `now` (seconds since the epoch).
fn google_token_request(connector: &Connector, req: &Json) -> Result<Json, FfiStatus> {
    let scope = connector
        .parameters
        .get("scopes")
        .map_or(google::DEFAULT_SCOPE, String::as_str);
    let adc = connector
        .parameters
        .get("auth_mode")
        .is_some_and(|m| m.eq_ignore_ascii_case("adc"));
    // An application default credentials file's service account key, when
    // the plugin found one.
    let file = match req.get("key") {
        Some(Json::Object(m)) => Some(
            m.iter()
                .filter_map(|(k, v)| match v {
                    Json::String(s) => Some((k.clone(), s.clone())),
                    _ => None,
                })
                .collect::<std::collections::HashMap<_, _>>(),
        ),
        _ => None,
    };
    if adc && file.is_none() {
        return Ok(Json::Object(vec![
            ("adc".into(), Json::Bool(true)),
            ("scope".into(), Json::String(scope.into())),
        ]));
    }
    let source = file.as_ref().unwrap_or(&connector.credential);
    let get = |k: &str| source.get(k).map_or("", String::as_str);
    let token_uri = source
        .get("token_uri")
        .map_or(google::TOKEN_URI, String::as_str);
    if file.is_some() && get("type") == "authorized_user" {
        // A user's credentials (`gcloud auth application-default login`).
        let body =
            google::refresh_request(get("client_id"), get("client_secret"), get("refresh_token"));
        return Ok(Json::Object(vec![
            ("token_uri".into(), Json::String(token_uri.into())),
            ("body".into(), Json::String(body)),
        ]));
    }
    let now = req.get("now").and_then(Json::as_f64).unwrap_or(0.0) as u64;
    let body = google::token_request(
        get("private_key"),
        get("client_email"),
        token_uri,
        scope,
        now,
    )
    .map_err(ml_failure)?;
    google::validate_token_uri(token_uri).map_err(ml_failure)?;
    Ok(Json::Object(vec![
        ("token_uri".into(), Json::String(token_uri.into())),
        ("body".into(), Json::String(body)),
        (
            "client_email".into(),
            Json::String(get("client_email").into()),
        ),
    ]))
}

/// Parameters as the plugin sends them: a value that is not a string as its
/// JSON, as ml-commons' `getParameterMap` keeps it.
fn params(v: Option<&Json>) -> Vec<(String, Option<String>)> {
    match v {
        Some(Json::Object(m)) => m
            .iter()
            .map(|(k, v)| {
                let v = match v {
                    Json::Null => None,
                    Json::String(s) => Some(s.clone()),
                    other => Some(other.to_compact()),
                };
                (k.clone(), v)
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn input(v: Option<&Json>) -> Result<Input, FfiStatus> {
    let v = v.ok_or_else(|| invalid("the remote call has no input"))?;
    if let Some(p) = v.get("parameters") {
        return Ok(Input::Remote(params(Some(p))));
    }
    let docs: Vec<Option<String>> = match v.get("text_docs") {
        Some(Json::Array(a)) => a
            .iter()
            .map(|d| match d {
                Json::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    match v.get("query_text") {
        Some(Json::String(q)) => Ok(Input::TextSimilarity {
            query: q.clone(),
            docs: docs.into_iter().map(Option::unwrap_or_default).collect(),
        }),
        _ => Ok(Input::TextDocs(docs)),
    }
}

fn input_json(i: &Input) -> Json {
    let docs = |d: Vec<Json>| ("text_docs".to_string(), Json::Array(d));
    match i {
        Input::TextDocs(d) => Json::Object(vec![docs(
            d.iter()
                .map(|s| s.clone().map_or(Json::Null, Json::String))
                .collect(),
        )]),
        Input::TextSimilarity { query, docs: d } => Json::Object(vec![
            ("query_text".into(), Json::String(query.clone())),
            docs(d.iter().cloned().map(Json::String).collect()),
        ]),
        Input::Remote(p) => Json::Object(vec![(
            "parameters".into(),
            Json::Object(
                p.iter()
                    .map(|(k, v)| (k.clone(), v.clone().map_or(Json::Null, Json::String)))
                    .collect(),
            ),
        )]),
    }
}

fn script_json(s: &Script) -> Json {
    Json::Object(vec![(
        "script".into(),
        Json::Object(vec![
            ("source".into(), Json::String(s.source.clone())),
            ("params".into(), Json::String(s.params.clone())),
        ]),
    )])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffm_bridge::ffi_jvm_free_bytes;

    const KEY: &str = "NX/bPD1UpNIkPWG6gGqmH/xTQWrBOlrQSZBrpX/vJTw=";

    fn alloc(f: impl FnOnce(*mut *mut u8, *mut usize) -> i32) -> Result<String, i32> {
        let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
        let status = f(&mut p, &mut n);
        if status != 0 {
            return Err(status);
        }
        // SAFETY: the call gave a buffer of `n` bytes, or null for none.
        let out = if p.is_null() {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
        };
        // SAFETY: freed once, as given.
        unsafe { ffi_jvm_free_bytes(p, n) };
        Ok(String::from_utf8(out).unwrap())
    }

    fn remote_call(handle: u64, req: &str) -> Result<String, i32> {
        // SAFETY: valid buffers.
        alloc(|p, n| unsafe { ffi_ml_remote_call_alloc(handle, req.as_ptr(), req.len(), p, n) })
    }

    #[test]
    fn a_credential_decrypts_to_what_was_encrypted() {
        // SAFETY: valid buffers.
        let stored = alloc(|p, n| unsafe {
            ffi_ml_credential_encrypt_alloc(KEY.as_ptr(), KEY.len(), b"sk-1".as_ptr(), 4, p, n)
        })
        .unwrap();
        // SAFETY: valid buffers.
        let plain = alloc(|p, n| unsafe {
            ffi_ml_credential_decrypt_alloc(
                KEY.as_ptr(),
                KEY.len(),
                stored.as_ptr(),
                stored.len(),
                p,
                n,
            )
        })
        .unwrap();
        assert_eq!(plain, "sk-1");
    }

    #[test]
    fn runs_a_call_through_the_handle() {
        // SAFETY: valid buffers.
        let stored = alloc(|p, n| unsafe {
            ffi_ml_credential_encrypt_alloc(KEY.as_ptr(), KEY.len(), b"sk-1".as_ptr(), 4, p, n)
        })
        .unwrap();
        let doc = format!(
            r#"{{"protocol":"http","parameters":{{"model":"m"}},"credential":{{"key":"{stored}"}},
            "actions":[{{"action_type":"predict","method":"POST","url":"http://h/e",
            "headers":{{"Authorization":"Bearer ${{credential.key}}"}},
            "request_body":"{{ \"input\": ${{parameters.input}}, \"model\": \"${{parameters.model}}\" }}",
            "pre_process_function":"connector.pre_process.openai.embedding",
            "post_process_function":"connector.post_process.openai.embedding"}}]}}"#
        );
        let mut handle = 0u64;
        // SAFETY: valid buffers.
        assert_eq!(
            unsafe {
                ffi_ml_remote_open(
                    doc.as_ptr(),
                    doc.len(),
                    KEY.as_ptr(),
                    KEY.len(),
                    &mut handle,
                )
            },
            0
        );
        let split =
            remote_call(handle, r#"{"op":"split","input":{"text_docs":["a","b"]}}"#).unwrap();
        assert_eq!(split, r#"{"inputs":[{"text_docs":["a","b"]}]}"#);
        let prepared = remote_call(
            handle,
            r#"{"op":"prepare","algorithm":"TEXT_EMBEDDING","input":{"text_docs":["a","b"]},"parameters":{}}"#,
        )
        .unwrap();
        let v = Json::parse(&prepared).unwrap();
        let r = v.get("request").unwrap();
        assert_eq!(
            r.get("body"),
            Some(&Json::String(
                r#"{ "input": ["a","b"], "model": "m" }"#.into()
            ))
        );
        assert!(r
            .get("headers")
            .unwrap()
            .to_compact()
            .contains(r#"["Authorization","Bearer sk-1"]"#));
        let parse = format!(
            r#"{{"op":"parse","parameters":{},"body":"{{\"data\":[{{\"embedding\":[0.5,1]}}]}}"}}"#,
            v.get("parameters").unwrap().to_compact()
        );
        assert_eq!(
            remote_call(handle, &parse).unwrap(),
            r#"{"output":[{"name":"sentence_embedding","data_type":"FLOAT32","shape":[2],"data":[0.5,1]}]}"#
        );
        let remote = remote_call(
            handle,
            r#"{"op":"prepare","input":{"parameters":{"input":"[1]"}}}"#,
        )
        .unwrap();
        assert!(remote.contains(r#"\"input\": [1]"#), "{remote}");
        assert_eq!(
            remote_call(handle, r#"{"op":"nope"}"#),
            Err(FfiStatus::InvalidArgument.code())
        );
        assert_eq!(
            remote_call(handle, "x"),
            Err(FfiStatus::InvalidArgument.code())
        );
        assert_eq!(
            remote_call(
                handle,
                r#"{"op":"prepare","input":{"parameters":{"input":"x y"}}}"#
            ),
            Err(FfiStatus::InvalidArgument.code())
        );
        assert_eq!(crate::ml::ffi_ml_model_close(handle), 0);
        assert_eq!(
            remote_call(handle, r#"{"op":"split"}"#),
            Err(FfiStatus::InvalidHandle.code())
        );
    }

    #[test]
    fn hands_scripts_and_keys_out() {
        let key = alloc(|p, n| unsafe { ffi_ml_master_key_alloc(p, n) }).unwrap();
        assert_eq!(esdk::base64_decode(&key).unwrap().len(), 32);
        let doc = r#"{"protocol":"http","actions":[{"action_type":"predict","method":"POST","url":"http://h/e",
            "request_body":"{\"q\": ${parameters.q}}","pre_process_function":"return params.text_docs[0];",
            "post_process_function":"return params.result;"}]}"#;
        let mut handle = 0u64;
        // SAFETY: valid buffers.
        assert_eq!(
            unsafe {
                ffi_ml_remote_open(
                    doc.as_ptr(),
                    doc.len(),
                    key.as_ptr(),
                    key.len(),
                    &mut handle,
                )
            },
            0
        );
        let s = remote_call(
            handle,
            r#"{"op":"prepare","algorithm":"TEXT_EMBEDDING","input":{"text_docs":["x"]}}"#,
        )
        .unwrap();
        assert_eq!(
            s,
            r#"{"script":{"source":"return params.text_docs[0];","params":"{\"algorithm\":\"TEXT_EMBEDDING\",\"text_docs\":[\"x\"]}"}}"#
        );
        let s = remote_call(
            handle,
            r#"{"op":"prepare","algorithm":"TEXT_EMBEDDING","input":{"text_docs":["x"]},"script_output":"{\"parameters\":{\"q\":\"[2]\"}}"}"#,
        )
        .unwrap();
        assert!(s.contains(r#""body":"{\"q\": [2]}""#), "{s}");
        let p = remote_call(handle, r#"{"op":"parse","parameters":{},"body":"[1]"}"#).unwrap();
        assert!(
            p.starts_with(
                r#"{"script":{"source":"return params.result;","params":"{\"result\":[1]}"#
            ),
            "{p}"
        );
        let p = remote_call(
            handle,
            r#"{"op":"parse","parameters":{},"body":"[1]","script_output":"[{\"name\":\"n\"}]"}"#,
        )
        .unwrap();
        assert_eq!(p, r#"{"output":[{"name":"n"}]}"#);
        let sim = remote_call(
            handle,
            r#"{"op":"split","input":{"query_text":"q","text_docs":["a"]}}"#,
        )
        .unwrap();
        assert_eq!(sim, r#"{"inputs":[{"query_text":"q","text_docs":["a"]}]}"#);
        assert_eq!(crate::ml::ffi_ml_model_close(handle), 0);
        let bad = r#"{"protocol":"http","credential":{"k":"not-base64!"}}"#;
        // SAFETY: valid buffers.
        assert_ne!(
            unsafe {
                ffi_ml_remote_open(
                    bad.as_ptr(),
                    bad.len(),
                    key.as_ptr(),
                    key.len(),
                    &mut handle,
                )
            },
            0
        );
    }

    #[test]
    fn signs_in_to_google() {
        let pem = include_str!("../../lucene-ml/src/remote/testdata/google-sa.pem");
        let open = |doc: &str| {
            let mut handle = 0u64;
            // SAFETY: valid buffers.
            let rc = unsafe {
                ffi_ml_remote_open(
                    doc.as_ptr(),
                    doc.len(),
                    KEY.as_ptr(),
                    KEY.len(),
                    &mut handle,
                )
            };
            assert_eq!(rc, 0);
            handle
        };
        let enc = |v: &str| {
            // SAFETY: valid buffers.
            alloc(|p, n| unsafe {
                ffi_ml_credential_encrypt_alloc(KEY.as_ptr(), KEY.len(), v.as_ptr(), v.len(), p, n)
            })
            .unwrap()
        };
        let action = r#""actions":[{"action_type":"predict","method":"POST","url":"https://x/v","request_body":"{}"}]"#;
        let doc = format!(
            r#"{{"protocol":"google_cloud","credential":{{"private_key":"{}","client_email":"{}"}},{action}}}"#,
            enc(pem),
            enc("a@p.iam.gserviceaccount.com")
        );
        let h = open(&doc);
        let r = remote_call(h, r#"{"op":"google_token_request","now":5}"#).unwrap();
        let v = Json::parse(&r).unwrap();
        assert_eq!(
            v.get("token_uri"),
            Some(&Json::String(google::TOKEN_URI.into()))
        );
        assert!(matches!(v.get("body"), Some(Json::String(b)) if b.starts_with("grant_type=")));
        assert_eq!(crate::ml::ffi_ml_model_close(h), 0);
        let doc = format!(
            r#"{{"protocol":"google_cloud","credential":{{"private_key":"{}","client_email":"{}","token_uri":"{}"}},{action}}}"#,
            enc(pem),
            enc("a"),
            enc("http://oauth2.googleapis.com/token")
        );
        let h = open(&doc);
        assert_eq!(
            remote_call(h, r#"{"op":"google_token_request"}"#),
            Err(FfiStatus::InvalidArgument.code())
        );
        assert_eq!(crate::ml::ffi_ml_model_close(h), 0);
        let doc = format!(
            r#"{{"protocol":"google_cloud","parameters":{{"auth_mode":"ADC","scopes":"s"}},{action}}}"#
        );
        let h = open(&doc);
        assert_eq!(
            remote_call(h, r#"{"op":"google_token_request"}"#).unwrap(),
            r#"{"adc":true,"scope":"s"}"#
        );
        let user = r#"{"op":"google_token_request","key":{"type":"authorized_user","client_id":"c","client_secret":"s","refresh_token":"r"}}"#;
        assert_eq!(
            remote_call(h, user).unwrap(),
            r#"{"token_uri":"https://oauth2.googleapis.com/token","body":"grant_type=refresh_token&client_id=c&client_secret=s&refresh_token=r"}"#
        );
        assert_eq!(crate::ml::ffi_ml_model_close(h), 0);
        let doc = format!(
            r#"{{"protocol":"google_cloud","credential":{{"private_key":"{}","client_email":"{}"}},{action}}}"#,
            enc("x"),
            enc("a")
        );
        let h = open(&doc);
        assert_eq!(
            remote_call(h, r#"{"op":"google_token_request"}"#),
            Err(FfiStatus::Search.code())
        );
        assert_eq!(crate::error::last_error(), "Invalid PKCS#8 data.");
        assert_eq!(crate::ml::ffi_ml_model_close(h), 0);
    }
}
