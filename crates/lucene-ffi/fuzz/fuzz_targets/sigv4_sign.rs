//! `sigv4::sign` over an arbitrary request: method, URL, headers, body and credentials. A request
//! must be signed or its URL refused, never a panic, and the secret key and the session token must
//! not appear in any header but the token's own.
#![no_main]

use libfuzzer_sys::fuzz_target;
use lucene_ml::remote::sigv4::{sign, url_parts::Url, Signing};

fuzz_target!(|data: &[u8]| {
    let mut parts = data.split(|b| *b == 0).map(|p| String::from_utf8_lossy(p).into_owned());
    let (method, url) = (parts.next().unwrap_or_default(), parts.next().unwrap_or_default());
    let (service, region) = (parts.next().unwrap_or_default(), parts.next().unwrap_or_default());
    let secret = "fuzz-secret-9f3a1c70d2".to_string();
    let _ = parts.next();
    let token = parts.next().map(|_| "fuzz-token-5b8e44a1c9".to_string());
    let body = parts.next().unwrap_or_default();
    let mut headers: Vec<(String, String)> = parts
        .collect::<Vec<_>>()
        .chunks(2)
        .map(|c| (c[0].clone(), c.get(1).cloned().unwrap_or_default()))
        .collect();
    // Inputs cannot contain the credentials (they are fixed strings the fuzzer would have to guess),
    // so any appearance in a header is a leak.
    if String::from_utf8_lossy(data).contains("fuzz-") {
        return;
    }
    let Some(url) = Url::parse(&url) else { return };
    let signing = Signing { access_key: "AKID", secret_key: &secret, session_token: token.as_deref(), service: &service, region: &region };
    sign(&method, &url, &mut headers, body.as_bytes(), &signing, "20260101T000000Z");
    for (k, v) in &headers {
        assert!(!v.contains(&secret), "the secret key reached header {k}");
        if let Some(t) = &token {
            assert!(k.eq_ignore_ascii_case("x-amz-security-token") || !v.contains(t.as_str()), "the session token reached header {k}");
        }
    }
});
