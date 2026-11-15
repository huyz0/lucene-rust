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
    let secret = format!("S3CR3T{}", parts.next().unwrap_or_default());
    let token = parts.next().map(|t| format!("T0K3N{t}"));
    let body = parts.next().unwrap_or_default();
    let mut headers: Vec<(String, String)> = parts
        .collect::<Vec<_>>()
        .chunks(2)
        .map(|c| (c[0].clone(), c.get(1).cloned().unwrap_or_default()))
        .collect();
    // The scope and the signed-header list echo the request, so only a secret that did not come in
    // with the request itself can be a leak.
    let echoed = |s: &str| s.contains(&secret) || token.as_deref().is_some_and(|t| s.contains(t));
    if [&method, &url, &service, &region, &body, &secret].into_iter().chain(token.as_ref()).any(|s| echoed(s) && s != &secret && Some(s) != token.as_ref() || token.as_deref().is_some_and(|t| t.contains(&secret) || secret.contains(t))) || headers.iter().any(|(k, v)| echoed(k) || echoed(v)) {
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
