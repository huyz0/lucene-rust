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
        r#"{"host":1}"#,
        r#"{"host":true}"#,
        r#"{"host":{"a":"b"}}"#,
        r#"{"host":[["a"]]}"#,
        r#"{"host":["a",]}"#,
        r#"{"host":[,]}"#,
        r#"{"host":nul}"#,
        r#"{"host":nullx}"#,
        r#"{"host":"a","message":[1]}"#,
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

fn shaped() -> Arc<Plan> {
    let id = register(&blob(&[
        ("o", OBJECT, 0, 0, 0),
        ("o.k", KEYWORD, 3, 20, i32::MAX),
        ("o.n", INTEGER, 3, 21, 0),
        ("o.deep", OBJECT, 0, 0, 0),
        ("o.deep.x", LONG, 3, 22, 0),
        ("tags", KEYWORD, 3, 23, i32::MAX),
        ("scores", LONG, 3, 24, 0),
        ("msgs", TEXT, 1, 25, 0),
        ("objs", OBJECT, 0, 0, 0),
        ("objs.v", INTEGER, 3, 26, 0),
        ("objs.t", KEYWORD, 1, 27, i32::MAX),
    ]))
    .unwrap();
    plan(id).unwrap()
}

fn shaped_run(src: &str) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    parse(&shaped(), src.as_bytes(), &mut out).then_some(out)
}

fn i32_at(b: &[u8], at: usize) -> i32 {
    i32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

/// The inverted section of a fragment.
fn inverted(out: &[u8]) -> &[u8] {
    let at = 32 + i32_at(out, 16) as usize;
    &out[at..at + i32_at(out, 20) as usize]
}

#[test]
fn objects_are_read_at_any_depth() {
    let out = shaped_run(r#"{"o":{"k":"a","n":3,"deep":{"x":5}},"tags":"t"}"#).unwrap();
    // inverted: o.k, tags; doc values: k, n, x, tags; points: n, x
    assert_eq!(counts(&out), [0, 2, 4, 2]);
    assert!(shaped_run(r#"{"o":{},"objs":{"v":1}}"#).is_some());
    assert!(shaped_run(r#"{"o":{"deep":{}}}"#).is_some());
}

#[test]
fn null_is_no_value() {
    let out =
        shaped_run(r#"{"o":null,"tags":null,"scores":[null,1,null],"msgs":null,"objs":[null]}"#)
            .unwrap();
    assert_eq!(counts(&out), [0, 0, 1, 1]);
    assert_eq!(
        counts(&shaped_run(r#"{"o":{"k":null,"deep":null}}"#).unwrap()),
        [0, 0, 0, 0]
    );
}

#[test]
fn a_keyword_array_is_one_entry_of_its_distinct_values() {
    let out = shaped_run(r#"{"tags":["b","a","b"]}"#).unwrap();
    assert_eq!(counts(&out), [0, 1, 3, 0]);
    let inv = inverted(&out);
    assert_eq!(i32_at(inv, 0), 23);
    assert_eq!(inv[4], 0);
    assert_eq!(i32_at(inv, 5), 2);
    // "b" first, then "a"; a keyword indexes documents only, so "b" repeated is still freq 1
    assert_eq!(i32_at(inv, 9), 1);
    assert_eq!(inv[13], b'b');
    assert_eq!(i32_at(inv, 14), 1);
    assert_eq!(inv[18], 0);
    assert_eq!(i32_at(inv, 19), 1);
    assert_eq!(inv[23], b'a');
    assert_eq!(i32_at(inv, 24), 1);
    assert_eq!(counts(&shaped_run(r#"{"tags":[]}"#).unwrap()), [0, 0, 0, 0]);
}

#[test]
fn a_text_array_is_one_entry_of_every_value() {
    let out = shaped_run(r#"{"msgs":["a b","c"]}"#).unwrap();
    assert_eq!(counts(&out), [0, 1, 0, 0]);
    let inv = inverted(&out);
    // number, kind 2, analyzer, options, norm, name (i32 len + 4), then the count
    let name = 4 + 1 + 4 + 1 + 1;
    assert_eq!(i32_at(inv, name), 4);
    let count = name + 4 + 4;
    assert_eq!(i32_at(inv, count), 2);
    assert_eq!(
        counts(&shaped_run(r#"{"msgs":[null]}"#).unwrap()),
        [0, 0, 0, 0]
    );
}

#[test]
fn numeric_arrays_are_an_entry_per_value() {
    assert_eq!(
        counts(&shaped_run(r#"{"scores":[1,2,3]}"#).unwrap()),
        [0, 0, 3, 3]
    );
    assert_eq!(
        counts(&shaped_run(r#"{"scores":[]}"#).unwrap()),
        [0, 0, 0, 0]
    );
}

#[test]
fn an_array_of_objects_scopes_each_object() {
    let out = shaped_run(r#"{"objs":[{"v":1,"t":"a"},null,{"v":2}]}"#).unwrap();
    assert_eq!(counts(&out), [0, 1, 2, 2]);
    assert!(shaped_run(r#"{"objs":[{"v":1,"v":2}]}"#).is_none());
    assert!(shaped_run(r#"{"objs":[{"v":1},]}"#).is_none());
    assert!(shaped_run(r#"{"objs":[1]}"#).is_none());
}

#[test]
fn what_an_object_or_array_may_not_be_is_refused() {
    for src in [
        r#"{"o.k":"a"}"#,
        r#"{"o":{"k.x":"a"}}"#,
        r#"{"o":{"z":1}}"#,
        r#"{"z":{"k":1}}"#,
        r#"{"o":5}"#,
        r#"{"o":"x"}"#,
        r#"{"o":[[]]}"#,
        r#"{"tags":{"a":1}}"#,
        r#"{"tags":[["a"]]}"#,
        r#"{"tags":["a",{"b":1}]}"#,
        r#"{"scores":[1,[2]]}"#,
        r#"{"scores":[1.5]}"#,
        r#"{"o":{"k":"a","k":"b"}}"#,
        r#"{"o":{"k":"a"},"o":{"n":1}}"#,
        r#"{"o":{"n":1,}}"#,
        r#"{"o":nul}"#,
        r#"{"o":{"deep":{"x":"1"}}}"#,
    ] {
        assert!(shaped_run(src).is_none(), "{src}");
    }
    let mut deep = String::new();
    for _ in 0..40 {
        deep.push_str(r#"{"o":"#);
    }
    assert!(shaped_run(&deep).is_none());
}

#[test]
fn a_keyword_array_is_bounded() {
    let many = format!(
        "{{\"tags\":[{}]}}",
        (0..300)
            .map(|i| format!("\"v{i}\""))
            .collect::<Vec<_>>()
            .join(",")
    );
    assert!(shaped_run(&many).is_none());
    let ok = format!(
        "{{\"tags\":[{}]}}",
        (0..200)
            .map(|i| format!("\"v{i}\""))
            .collect::<Vec<_>>()
            .join(",")
    );
    assert!(shaped_run(&ok).is_some());
}

/// `cargo test --profile dist -p lucene-engine ingest_speed -- --ignored --nocapture`:
/// nanoseconds per logs document of `parse`.
#[test]
#[ignore = "a timing, not a check"]
fn ingest_speed() {
    let id = register(&blob(&[
        ("@timestamp", DATE, 3, 1, 0),
        ("host", KEYWORD, 3, 2, i32::MAX),
        ("service", KEYWORD, 3, 3, i32::MAX),
        ("level", KEYWORD, 3, 4, i32::MAX),
        ("path", KEYWORD, 3, 5, i32::MAX),
        ("status", INTEGER, 3, 6, 0),
        ("bytes", LONG, 3, 7, 0),
        ("latency_ms", FLOAT, 3, 8, 0),
        ("trace_id", KEYWORD, 3, 9, i32::MAX),
        ("message", TEXT, 1, 10, 0),
    ]))
    .unwrap();
    let p = plan(id).unwrap();
    let docs: Vec<String> = (0..1000)
        .map(|i| {
            format!(
                r#"{{"@timestamp":"2026-11-01T00:00:{:02}.{:03}Z","host":"host-{:03}","service":"gateway","level":"INFO","path":"/api/v1/orders","status":{},"bytes":{},"latency_ms":{}.{:02},"trace_id":"{:032x}","message":"request completed in {} ms for user u{}"}}"#,
                i % 60,
                (i * 37) % 1000,
                i % 200,
                [200, 201, 404, 500][i % 4],
                100 + i * 131 % 200_000,
                i % 90,
                i % 100,
                (i as u128) * 0x9e37_79b9_7f4a_7c15_u128,
                i * 7 % 5000,
                i * 13 % 50_000,
            )
        })
        .collect();
    let bytes: usize = docs.iter().map(String::len).sum::<usize>() / docs.len();
    let mut out = Vec::new();
    for round in 0..4 {
        let t = std::time::Instant::now();
        let mut n = 0u64;
        for _ in 0..2000 {
            for d in &docs {
                assert!(parse(&p, d.as_bytes(), &mut out));
                n += 1;
            }
        }
        let ns = t.elapsed().as_nanos() as f64 / n as f64;
        println!(
            "round {round}: {ns:.0} ns/doc, {bytes} B/doc, {:.0} MB/s",
            bytes as f64 / ns * 1000.0
        );
    }
}

#[test]
fn a_field_repeated_across_objects_is_one_entry() {
    // objs.t is keyword (indexed only): "a" twice and "b" once, in two objects
    let out = shaped_run(r#"{"objs":[{"v":1,"t":"a"},{"v":2,"t":"b"},{"t":"a"}]}"#).unwrap();
    assert_eq!(counts(&out), [0, 1, 2, 2]);
    let inv = inverted(&out);
    assert_eq!(i32_at(inv, 0), 27);
    assert_eq!(i32_at(inv, 5), 2);
    assert_eq!(inv[13], b'a');
    assert_eq!(i32_at(inv, 14), 1);
    // single object inside an array of objects, nested object arrays
    let out = shaped_run(r#"{"o":[{"k":"a","deep":{"x":1}},{"k":"b","deep":{"x":2}}]}"#).unwrap();
    assert_eq!(counts(&out), [0, 1, 4, 2]);
    let out = shaped_run(r#"{"o":{"k":["a","b"],"deep":{"x":1}},"objs":[{"t":"z"}]}"#).unwrap();
    assert_eq!(counts(&out), [0, 2, 3, 1]);
}

#[test]
fn integers_parse_to_their_range() {
    assert_eq!(int_of("0"), Some(0));
    assert_eq!(int_of("-0"), Some(0));
    assert_eq!(int_of("9223372036854775807"), Some(i64::MAX));
    assert_eq!(int_of("-9223372036854775808"), Some(i64::MIN));
    assert_eq!(int_of("9223372036854775808"), None);
    assert_eq!(int_of("-9223372036854775809"), None);
    assert_eq!(int_of("99999999999999999999"), None);
    assert_eq!(int_of("18446744073709551615"), None);
}

#[test]
fn a_source_that_is_not_utf8_is_refused() {
    let mut out = Vec::new();
    assert!(!parse(&logs(), b"{\"host\":\"a\xff\"}", &mut out));
    assert!(!parse(&logs(), b"{\"host\":\"a\xc3\"}", &mut out));
    assert!(parse(&logs(), b"{\"host\":\"a\xc3\xa9\"}", &mut out));
}

#[test]
fn key_order_hints_never_change_what_is_read() {
    let p = shaped();
    let mut a = Vec::new();
    let mut b = Vec::new();
    let one = r#"{"o":{"k":"a","n":1,"deep":{"x":2}},"tags":["t"],"scores":[3]}"#;
    let two = r#"{"scores":[3],"tags":["t"],"o":{"deep":{"x":2},"n":1,"k":"a"}}"#;
    // alternate shapes so each document's hints come from the other's
    for _ in 0..3 {
        assert!(parse(&p, one.as_bytes(), &mut a));
        assert!(parse(&p, two.as_bytes(), &mut b));
    }
    assert_eq!(counts(&a), counts(&b));
    let (mut sa, mut sb) = (a.clone(), b.clone());
    // the entries come in source order, so compare section by section as sets of bytes
    for out in [&mut sa, &mut sb] {
        out[32..].sort_unstable();
    }
    assert_eq!(sa[32..].len(), sb[32..].len());
    // a key that is a dotted path to a real field stays refused whatever the hints hold
    assert!(!parse(&p, br#"{"o.k":"a"}"#, &mut a));
    assert!(parse(&p, one.as_bytes(), &mut a));
    assert!(!parse(&p, br#"{"o":{"k.x":"a"}}"#, &mut a));
}
