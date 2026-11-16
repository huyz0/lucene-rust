#![cfg(test)]
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
    let split = remote_call(handle, r#"{"op":"split","input":{"text_docs":["a","b"]}}"#).unwrap();
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
        p.starts_with(r#"{"script":{"source":"return params.result;","params":"{\"result\":[1]}"#),
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
    let pem = include_str!("../../../lucene-ml/src/remote/testdata/google-sa.pem");
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
