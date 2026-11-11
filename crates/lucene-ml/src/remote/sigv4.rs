//! AWS Signature Version 4, as the AWS SDK for Java v2's `AwsV4HttpSigner`
//! signs a connector's request: the path encoded twice (the SDK's default
//! for every service but S3), every header the request carries signed,
//! `host` added, `x-amz-content-sha256` filled with the payload hash when
//! the connector asks for it, and the session token sent as
//! `X-Amz-Security-Token`.

use aws_lc_rs::{digest, hmac};

/// Credentials and scope of a signature.
#[derive(Debug, Clone, Copy)]
pub struct Signing<'a> {
    /// The access key id.
    pub access_key: &'a str,
    /// The secret access key.
    pub secret_key: &'a str,
    /// A session token, for temporary credentials.
    pub session_token: Option<&'a str>,
    /// The service's signing name (`bedrock`, `sagemaker`, ...).
    pub service: &'a str,
    /// The region.
    pub region: &'a str,
}

const HASH_HEADER: &str = "x-amz-content-sha256";

/// Signs a request in place: `headers` gains `Host` (unless present),
/// `X-Amz-Date`, the security token and `Authorization`. `amz_date` is the
/// signing time as `yyyyMMddTHHmmssZ`.
pub fn sign(
    method: &str,
    url: &url_parts::Url<'_>,
    headers: &mut Vec<(String, String)>,
    body: &[u8],
    signing: &Signing<'_>,
    amz_date: &str,
) {
    let payload_hash = hex(digest::digest(&digest::SHA256, body).as_ref());
    for (k, v) in headers.iter_mut() {
        if k.eq_ignore_ascii_case(HASH_HEADER) {
            *v = payload_hash.clone();
        }
    }
    if !headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("host")) {
        headers.push(("Host".into(), url.host_header()));
    }
    headers.push(("X-Amz-Date".into(), amz_date.into()));
    if let Some(token) = signing.session_token {
        headers.push(("X-Amz-Security-Token".into(), token.into()));
    }
    let mut canonical: Vec<(String, String)> = headers
        .iter()
        .map(|(k, v)| (k.to_ascii_lowercase(), collapse(v)))
        .collect();
    canonical.sort_by(|a, b| a.0.cmp(&b.0));
    // A header sent more than once signs as one, its values comma-joined.
    let mut merged: Vec<(String, String)> = Vec::new();
    for (k, v) in canonical {
        match merged.last_mut() {
            Some(last) if last.0 == k => {
                last.1.push(',');
                last.1.push_str(&v);
            }
            _ => merged.push((k, v)),
        }
    }
    let signed_headers = merged
        .iter()
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>()
        .join(";");
    let mut request = String::new();
    request.push_str(&method.to_ascii_uppercase());
    request.push('\n');
    request.push_str(&canonical_path(url.path));
    request.push('\n');
    request.push_str(&canonical_query(url.query));
    request.push('\n');
    for (k, v) in &merged {
        request.push_str(k);
        request.push(':');
        request.push_str(v);
        request.push('\n');
    }
    request.push('\n');
    request.push_str(&signed_headers);
    request.push('\n');
    request.push_str(&payload_hash);

    let date = &amz_date[..8.min(amz_date.len())];
    let scope = format!("{date}/{}/{}/aws4_request", signing.region, signing.service);
    let to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        hex(digest::digest(&digest::SHA256, request.as_bytes()).as_ref())
    );
    let mut key = mac(
        format!("AWS4{}", signing.secret_key).as_bytes(),
        date.as_bytes(),
    );
    for part in [signing.region, signing.service, "aws4_request"] {
        key = mac(&key, part.as_bytes());
    }
    let signature = hex(&mac(&key, to_sign.as_bytes()));
    headers.push((
        "Authorization".into(),
        format!(
            "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
            signing.access_key
        ),
    ));
}

fn mac(key: &[u8], data: &[u8]) -> Vec<u8> {
    hmac::sign(&hmac::Key::new(hmac::HMAC_SHA256, key), data)
        .as_ref()
        .to_vec()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A header value trimmed, runs of spaces as one.
fn collapse(v: &str) -> String {
    v.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// URI-encodes `s` keeping the unreserved characters, and `/` when asked.
fn encode(s: &str, keep_slash: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric()
            || matches!(b, b'-' | b'_' | b'.' | b'~')
            || (keep_slash && b == b'/')
        {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

// ARITH: `i` indexes `s`, and `i + 2` is compared with its length before
// `i + 3` is used.
#[allow(clippy::arithmetic_side_effects)]
fn decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The raw path, normalized and then encoded again.
fn canonical_path(path: &str) -> String {
    let mut segments: Vec<&str> = Vec::new();
    for s in path.split('/') {
        match s {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            s => segments.push(s),
        }
    }
    let mut p = format!("/{}", segments.join("/"));
    if path.ends_with('/') && p.len() > 1 {
        p.push('/');
    }
    encode(&p, true)
}

fn canonical_query(query: Option<&str>) -> String {
    let Some(q) = query.filter(|q| !q.is_empty()) else {
        return String::new();
    };
    let mut pairs: Vec<(String, String)> = q
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (encode(&decode(k), false), encode(&decode(v), false))
        })
        .collect();
    pairs.sort();
    pairs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("&")
}

/// The parts of an absolute `http(s)` URL a request needs.
pub mod url_parts {
    /// A URL split as `java.net.URI` splits it (raw, still encoded).
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Url<'a> {
        /// `http` or `https` (as written).
        pub scheme: &'a str,
        /// The host; an IPv6 literal keeps its brackets.
        pub host: &'a str,
        /// The port, when written.
        pub port: Option<u16>,
        /// The raw path; empty when the URL has none.
        pub path: &'a str,
        /// The raw query, without `?`.
        pub query: Option<&'a str>,
    }

    impl<'a> Url<'a> {
        /// Splits `url`; `None` when it has no scheme or host.
        // ARITH: `close` is where `find` saw `]` after the leading `[`, so
        // `close + 2` is at most the authority's length.
        #[allow(clippy::arithmetic_side_effects)]
        pub fn parse(url: &'a str) -> Option<Url<'a>> {
            let (scheme, rest) = url.split_once("://")?;
            if scheme.is_empty()
                || !scheme
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"+-.".contains(&b))
            {
                return None;
            }
            let rest = rest.split('#').next().unwrap_or(rest);
            let end = rest.find(['/', '?']).unwrap_or(rest.len());
            let authority = &rest[..end];
            let tail = &rest[end..];
            let authority = authority.rsplit('@').next().unwrap_or(authority);
            let (host, port) = if let Some(stripped) = authority.strip_prefix('[') {
                let close = stripped.find(']')?;
                let host = &authority[..close + 2];
                let after = &authority[close + 2..];
                (host, after.strip_prefix(':'))
            } else {
                match authority.rsplit_once(':') {
                    Some((h, p)) => (h, Some(p)),
                    None => (authority, None),
                }
            };
            if host.is_empty() {
                return None;
            }
            let port = match port {
                Some("") | None => None,
                Some(p) => Some(p.parse().ok()?),
            };
            let (path, query) = match tail.split_once('?') {
                Some((p, q)) => (p, Some(q)),
                None => (tail, None),
            };
            Some(Url {
                scheme,
                host,
                port,
                path,
                query,
            })
        }

        /// The `Host` header: the host, and the port unless it is the
        /// scheme's default.
        pub fn host_header(&self) -> String {
            let default = if self.scheme.eq_ignore_ascii_case("https") {
                443
            } else {
                80
            };
            match self.port {
                Some(p) if p != default => format!("{}:{p}", self.host),
                _ => self.host.to_string(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::url_parts::Url;
    use super::*;

    fn header<'a>(h: &'a [(String, String)], k: &str) -> &'a str {
        &h.iter().find(|(n, _)| n.eq_ignore_ascii_case(k)).unwrap().1
    }

    #[test]
    fn signs_as_the_java_sdk_did_for_stock() {
        // A request stock ml-commons signed for a Bedrock connector, as the
        // remote end received it.
        let body = br#"{ "inputText": "hello \"q\" world" }"#;
        let url =
            Url::parse("http://localhost:8791/bedrock/model/amazon.titan-embed-text-v1/invoke")
                .unwrap();
        let mut headers = vec![
            ("content-type".to_string(), "application/json".to_string()),
            ("x-amz-content-sha256".to_string(), "required".to_string()),
            ("Content-Length".to_string(), body.len().to_string()),
        ];
        let signing = Signing {
            access_key: "AKIDEXAMPLE",
            secret_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            session_token: None,
            service: "bedrock",
            region: "us-east-1",
        };
        sign(
            "POST",
            &url,
            &mut headers,
            body,
            &signing,
            "20261111T022922Z",
        );
        assert_eq!(
            header(&headers, "x-amz-content-sha256"),
            "7de5a55a07523aec96cc332c71c7c9ac26a0ca966c6f947ebff788a9be31abb5"
        );
        assert_eq!(header(&headers, "host"), "localhost:8791");
        assert_eq!(
            header(&headers, "authorization"),
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20261111/us-east-1/bedrock/aws4_request, \
             SignedHeaders=content-length;content-type;host;x-amz-content-sha256;x-amz-date, \
             Signature=3436fa765d1e2479a91a756ad4c36a4665684a86d88a8ab4fda8603e556444df"
        );
    }

    #[test]
    fn matches_the_aws_test_suite() {
        // aws-sig-v4-test-suite get-vanilla-query-order-key-case.
        let url = Url::parse("https://example.amazonaws.com/?Param2=value2&Param1=value1").unwrap();
        let mut headers = vec![];
        let signing = Signing {
            access_key: "AKIDEXAMPLE",
            secret_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            session_token: None,
            service: "service",
            region: "us-east-1",
        };
        sign("GET", &url, &mut headers, b"", &signing, "20150830T123600Z");
        assert_eq!(header(&headers, "host"), "example.amazonaws.com");
        assert!(header(&headers, "authorization").ends_with(
            "SignedHeaders=host;x-amz-date, Signature=b97d918cfa904a5beff61c982a1b6f458b799221646efd99d3219ec94cdf2500"
        ));
        let mut headers = vec![];
        let with_token = Signing {
            session_token: Some("tok"),
            ..signing
        };
        sign(
            "GET",
            &url,
            &mut headers,
            b"",
            &with_token,
            "20150830T123600Z",
        );
        assert_eq!(header(&headers, "x-amz-security-token"), "tok");
        assert!(header(&headers, "authorization").contains("host;x-amz-date;x-amz-security-token"));
    }

    #[test]
    fn canonicalizes_paths_and_queries() {
        assert_eq!(canonical_path(""), "/");
        assert_eq!(canonical_path("/a/./b/../c/"), "/a/c/");
        assert_eq!(canonical_path("/m/a:b%20c"), "/m/a%3Ab%2520c");
        assert_eq!(canonical_query(Some("b=2&a=%41 x&c")), "a=A%20x&b=2&c=");
        assert_eq!(canonical_query(None), "");
        assert_eq!(collapse("  a   b "), "a b");
        assert_eq!(decode("%zz%4"), "%zz%4");
    }

    #[test]
    fn splits_urls() {
        let u = Url::parse("https://user@[::1]:9/p?q=1#f").unwrap();
        assert_eq!(
            (u.host, u.port, u.path, u.query),
            ("[::1]", Some(9), "/p", Some("q=1"))
        );
        assert_eq!(u.host_header(), "[::1]:9");
        let u = Url::parse("HTTPS://h:443").unwrap();
        assert_eq!((u.path, u.host_header()), ("", "h".to_string()));
        assert!(Url::parse("nohost").is_none());
        assert!(Url::parse("http://:80/").is_none());
        assert!(Url::parse("http://h:x/").is_none());
        assert!(Url::parse("ht tp://h/").is_none());
    }
}
