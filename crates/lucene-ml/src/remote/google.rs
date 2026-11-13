//! Google Cloud service-account sign-in, as ml-commons' `google_cloud`
//! (Vertex AI) connector does it through google-auth-library: a JWT bearer
//! assertion (RS256 over the service account's PKCS#8 key) that the caller
//! trades at the token endpoint for an OAuth2 access token.

use aws_lc_rs::rand::SystemRandom;
use aws_lc_rs::signature::{RsaKeyPair, RSA_PKCS1_SHA256};

use super::json::gson_quote;
use super::sigv4::url_parts::Url;
use crate::{MlError, Result};

/// The token endpoint google-auth-library uses when none is given.
pub const TOKEN_URI: &str = "https://oauth2.googleapis.com/token";
/// The scope a connector asks for when it names none.
pub const DEFAULT_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";
const TOKEN_HOST: &str = "oauth2.googleapis.com";
/// How long an assertion is good for, as google-auth-library makes it.
const LIFETIME_SECONDS: u64 = 3600;

fn arg(why: impl Into<String>) -> MlError {
    MlError::InvalidArgument(why.into())
}

/// `GoogleCredentialProvider.validateTokenUri`: an https URL on Google's
/// token host and its default port.
pub fn validate_token_uri(uri: &str) -> Result<()> {
    let parsed = Url::parse(uri).ok_or_else(|| arg(format!("Invalid token_uri: {uri}")))?;
    if !parsed.scheme.eq_ignore_ascii_case("https") {
        return Err(arg(format!(
            "token_uri must be an https Google endpoint, got: {uri}"
        )));
    }
    if !parsed.host.eq_ignore_ascii_case(TOKEN_HOST) {
        return Err(arg(format!(
            "token_uri host must be {TOKEN_HOST}, got: {}",
            parsed.host
        )));
    }
    if let Some(port) = parsed.port.filter(|p| *p != 443) {
        return Err(arg(format!(
            "token_uri must use the default https port, got port: {port}"
        )));
    }
    Ok(())
}

/// The DER of the first `PRIVATE KEY` section of a PEM text, as
/// `OAuth2Utils.privateKeyFromPkcs8` reads it.
fn pkcs8_der(pem: &str) -> Option<Vec<u8>> {
    let begin = "-----BEGIN PRIVATE KEY-----";
    let (_, rest) = pem.split_once(begin)?;
    let end = rest.find("-----END PRIVATE KEY-----")?;
    let b64: String = rest[..end].chars().filter(|c| !c.is_whitespace()).collect();
    super::esdk::base64_decode(&b64)
}

fn base64url(b: &[u8]) -> String {
    super::esdk::base64_encode(b)
        .trim_end_matches('=')
        .replace('+', "-")
        .replace('/', "_")
}

/// The form body that trades a signed assertion for `client_email`'s access
/// token with `scope` at `token_uri`, signed at `now` (seconds since the
/// epoch). A key that is not PKCS#8 RSA fails as google-auth-library does.
pub fn token_request(
    private_key: &str,
    client_email: &str,
    token_uri: &str,
    scope: &str,
    now: u64,
) -> Result<String> {
    let der =
        pkcs8_der(private_key).ok_or_else(|| MlError::Model("Invalid PKCS#8 data.".into()))?;
    let key =
        RsaKeyPair::from_pkcs8(&der).map_err(|_| MlError::Model("Invalid PKCS#8 data.".into()))?;
    let quoted = |s: &str| {
        let mut out = String::new();
        gson_quote(s, &mut out);
        out
    };
    let header = r#"{"alg":"RS256","typ":"JWT"}"#;
    let claims = format!(
        r#"{{"iss":{},"aud":{},"iat":{now},"exp":{},"scope":{}}}"#,
        quoted(client_email),
        quoted(token_uri),
        now.saturating_add(LIFETIME_SECONDS),
        quoted(scope),
    );
    let signing_input = format!(
        "{}.{}",
        base64url(header.as_bytes()),
        base64url(claims.as_bytes())
    );
    let mut signature = vec![0u8; key.public_modulus_len()];
    key.sign(
        &RSA_PKCS1_SHA256,
        &SystemRandom::new(),
        signing_input.as_bytes(),
        &mut signature,
    )
    .map_err(|_| MlError::Model("cannot sign the service account assertion".into()))?;
    Ok(format!(
        "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer&assertion={signing_input}.{}",
        base64url(&signature)
    ))
}

/// The form body that trades an `authorized_user` application default
/// credentials file's refresh token for an access token, as
/// google-auth-library's `UserCredentials` posts it.
pub fn refresh_request(client_id: &str, client_secret: &str, refresh_token: &str) -> String {
    format!(
        "grant_type=refresh_token&client_id={}&client_secret={}&refresh_token={}",
        form(client_id),
        form(client_secret),
        form(refresh_token)
    )
}

/// `URLEncoder.encode(s, UTF_8)`: letters, digits and `.-*_` kept, space as
/// `+`, every other byte as `%XX`.
fn form(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'.' | b'-' | b'*' | b'_' => {
                out.push(char::from(b))
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::json::Json;
    use aws_lc_rs::signature::KeyPair;
    use aws_lc_rs::signature::{UnparsedPublicKey, RSA_PKCS1_2048_8192_SHA256};

    // A 2048-bit test key, PKCS#8 PEM, made for this test only.
    const KEY: &str = include_str!("testdata/google-sa.pem");

    fn b64url_decode(s: &str) -> Vec<u8> {
        let mut t = s.replace('-', "+").replace('_', "/");
        while !t.len().is_multiple_of(4) {
            t.push('=');
        }
        super::super::esdk::base64_decode(&t).unwrap()
    }

    #[test]
    fn signs_a_jwt_bearer_assertion() {
        let body = token_request(
            KEY,
            "a@p.iam.gserviceaccount.com",
            TOKEN_URI,
            DEFAULT_SCOPE,
            1_700_000_000,
        )
        .unwrap();
        let assertion = body
            .strip_prefix(
                "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer&assertion=",
            )
            .unwrap();
        let parts: Vec<&str> = assertion.split('.').collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(b64url_decode(parts[0]), br#"{"alg":"RS256","typ":"JWT"}"#);
        let claims = Json::parse(std::str::from_utf8(&b64url_decode(parts[1])).unwrap()).unwrap();
        assert_eq!(
            claims.get("iss"),
            Some(&Json::String("a@p.iam.gserviceaccount.com".into()))
        );
        assert_eq!(claims.get("aud"), Some(&Json::String(TOKEN_URI.into())));
        assert_eq!(claims.get("exp"), Some(&Json::Number("1700003600".into())));
        assert_eq!(
            claims.get("scope"),
            Some(&Json::String(DEFAULT_SCOPE.into()))
        );
        let der = pkcs8_der(KEY).unwrap();
        let pair = RsaKeyPair::from_pkcs8(&der).unwrap();
        let public =
            UnparsedPublicKey::new(&RSA_PKCS1_2048_8192_SHA256, pair.public_key().as_ref());
        let signed = format!("{}.{}", parts[0], parts[1]);
        public
            .verify(signed.as_bytes(), &b64url_decode(parts[2]))
            .unwrap();
        assert!(!assertion.contains(['+', '/', '=']));
    }

    #[test]
    fn refuses_as_stock() {
        let e = token_request("x", "a", TOKEN_URI, DEFAULT_SCOPE, 0)
            .unwrap_err()
            .to_string();
        assert!(e.contains("Invalid PKCS#8 data."), "{e}");
        let bad = "-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----";
        assert!(token_request(bad, "a", TOKEN_URI, DEFAULT_SCOPE, 0).is_err());
        let cases = [
            ("http://oauth2.googleapis.com/token", "token_uri must be an https Google endpoint, got: http://oauth2.googleapis.com/token"),
            ("https://localhost:8791/token", "token_uri host must be oauth2.googleapis.com, got: localhost"),
            ("https://oauth2.googleapis.com:8443/token", "token_uri must use the default https port, got port: 8443"),
            ("::", "Invalid token_uri: ::"),
        ];
        for (uri, want) in cases {
            assert_eq!(validate_token_uri(uri).unwrap_err().to_string(), want);
        }
        validate_token_uri(TOKEN_URI).unwrap();
        validate_token_uri("https://oauth2.googleapis.com:443/token").unwrap();
        assert_eq!(
            refresh_request("id.apps", "s/é", "1//a b"),
            "grant_type=refresh_token&client_id=id.apps&client_secret=s%2F%C3%A9&refresh_token=1%2F%2Fa+b"
        );
    }
}
