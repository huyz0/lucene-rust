#![allow(clippy::arithmetic_side_effects)]
use super::*;

fn blob(fields: &[(&str, u8, u8, i32, i32)]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&(fields.len() as i32).to_le_bytes());
    for &(name, kind, flags, number, ignore_above) in fields {
        b.extend_from_slice(&(name.len() as i32).to_le_bytes());
        b.extend_from_slice(name.as_bytes());
        b.push(kind);
        b.push(flags);
        b.extend_from_slice(&number.to_le_bytes());
        b.extend_from_slice(&ignore_above.to_le_bytes());
        b.extend_from_slice(&7i32.to_le_bytes());
        b.push(3);
        b.push(2);
    }
    b
}

fn logs() -> Arc<Plan> {
    let id = register(&blob(&[
        ("host", KEYWORD, 3, 10, i32::MAX),
        ("status", INTEGER, 3, 11, 0),
        ("bytes", LONG, 3, 12, 0),
        ("latency_ms", FLOAT, 3, 13, 0),
        ("@timestamp", DATE, 3, 14, 0),
        ("message", TEXT, 1, 15, 0),
    ]))
    .unwrap();
    plan(id).unwrap()
}

fn run(src: &str) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    parse(&logs(), src.as_bytes(), &mut out).then_some(out)
}

fn counts(out: &[u8]) -> [i32; 4] {
    let c = |i: usize| i32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
    [c(0), c(1), c(2), c(3)]
}

#[test]
fn a_logs_document_yields_every_entry() {
    let out = run(r#"{"@timestamp":"2026-11-01T00:00:00.037Z","host":"host-001","status":200,"bytes":1234,"latency_ms":40.12,"message":"a b\né"}"#).unwrap();
    assert_eq!(counts(&out), [0, 2, 5, 4]);
}

#[test]
fn the_plan_registers_once_per_blob() {
    assert_eq!(
        register(&blob(&[("a", KEYWORD, 1, 1, 5)])).unwrap(),
        register(&blob(&[("a", KEYWORD, 1, 1, 5)])).unwrap()
    );
}

#[test]
fn what_the_fast_path_does_not_reproduce_is_refused() {
    for src in [
        r#"{"unknown":1}"#,
        r#"{"host":"a","host":"b"}"#,
        r#"{"host":null}"#,
        r#"{"host":["a"]}"#,
        r#"{"host":1}"#,
        r#"{"status":1.5}"#,
        r#"{"status":3000000000}"#,
        r#"{"status":"200"}"#,
        r#"{"latency_ms":1e999}"#,
        r#"{"@timestamp":"2026-02-30T00:00:00Z"}"#,
        r#"{"@timestamp":"2026-11-01"}"#,
        r#"{"host":"a"} x"#,
        r#"{"host":"a",}"#,
        r#"{"host":"\ud800"}"#,
        r#"{"host":"a" /* c */}"#,
        "{\"host\":\"a\u{1}\"}",
        r#"[1]"#,
        "",
    ] {
        assert!(run(src).is_none(), "{src}");
    }
}

#[test]
fn ignore_above_counts_utf16_units() {
    let id = register(&blob(&[("k", KEYWORD, 3, 1, 2)])).unwrap();
    let p = plan(id).unwrap();
    let mut out = Vec::new();
    assert!(parse(&p, r#"{"k":"ab"}"#.as_bytes(), &mut out));
    assert!(parse(&p, r#"{"k":"é"}"#.as_bytes(), &mut out));
    assert!(!parse(&p, r#"{"k":"abc"}"#.as_bytes(), &mut out));
    assert!(!parse(&p, r#"{"k":"😀a"}"#.as_bytes(), &mut out));
}

#[test]
fn sortable_encodings_match_lucene() {
    assert_eq!(float_sortable(1.0), 0x3f80_0000);
    assert_eq!(float_sortable(-1.0), 0xbf80_0000u32 as i32 ^ 0x7fff_ffff);
    assert!(float_sortable(-1.0) < float_sortable(-0.5));
    assert!(float_sortable(-0.5) < float_sortable(0.0));
    assert!(double_sortable(-2.0) < double_sortable(-1.0));
    assert_eq!(date_millis(b"1970-01-01T00:00:00.5Z"), Some(500));
    assert_eq!(
        date_millis(b"2026-11-01T00:00:00.037Z"),
        Some(1_793_491_200_037)
    );
    assert_eq!(
        date_millis(b"2024-02-29T23:59:59Z"),
        Some(1_709_251_199_000)
    );
}

#[test]
fn a_bad_plan_is_refused() {
    assert!(register(&[]).is_err());
    assert!(register(&blob(&[("a", 9, 1, 1, 1)])).is_err());
    assert!(register(&blob(&[("a", KEYWORD, 1, 1, 1), ("a", KEYWORD, 1, 2, 1)])).is_err());
    let mut b = blob(&[("a", KEYWORD, 1, 1, 1)]);
    b.push(0);
    assert!(register(&b).is_err());
}

#[test]
fn plain_json_shapes_are_accepted() {
    for src in [
        "{}",
        " {\t\"host\" : \"a\" ,\n \"status\" : -12 } ",
        r#"{"host":"a\"b\\c\/d\b\f\n\r\té😀"}"#,
        r#"{"status":0,"bytes":-9223372036854775808}"#,
        r#"{"bytes":9223372036854775807}"#,
        r#"{"latency_ms":-0.0}"#,
        r#"{"latency_ms":1.5e3}"#,
        r#"{"latency_ms":12}"#,
        r#"{"message":"é 😀 text"}"#,
        r#"{"@timestamp":"2026-11-01T01:02:03Z"}"#,
        r#"{"@timestamp":"2026-11-01T01:02:03.4Z"}"#,
        r#"{"@timestamp":"1999-12-31T23:59:59.999Z"}"#,
        r#"{"@timestamp":1793491200037}"#,
    ] {
        let _ = run(src);
    }
    assert!(run("{}").is_some());
    assert!(run(r#"{"status":0,"bytes":-9223372036854775808}"#).is_some());
    assert!(run(r#"{"bytes":9223372036854775808}"#).is_none());
    assert!(run(r#"{"host":"a😀é"}"#).is_some());
    assert!(run(r#"{"host":"\ud83d"}"#).is_none());
    assert!(run(r#"{"status":01}"#).is_none());
    assert!(run(r#"{"status":+1}"#).is_none());
    assert!(run(r#"{"status":-}"#).is_none());
    assert!(run(r#"{"status":1e}"#).is_none());
}

#[test]
fn a_fragment_leads_with_its_header() {
    let mut out = Vec::new();
    assert!(parse(&logs(), br#"{"host":"abc"}"#, &mut out));
    assert!(out.len() > 32);
}
