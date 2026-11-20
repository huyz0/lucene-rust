#![allow(clippy::arithmetic_side_effects)]
use super::*;

fn blob(fields: &[(&str, u8, u8, i32, i32)]) -> Vec<u8> {
    let fields: Vec<_> = fields
        .iter()
        .map(|&(n, k, f, num, ia)| (n, k, f, num, ia, -1, &[][..]))
        .collect();
    blob_with(&fields, 100)
}

type FieldRow<'a> = (&'a str, u8, u8, i32, i32, i32, &'a [i32]);

/// Fields with their multi-field parent index (-1 for none), `copy_to` targets and the nested limit.
fn blob_with(fields: &[FieldRow], nested_limit: i32) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&(fields.len() as i32).to_le_bytes());
    for &(name, kind, flags, number, ignore_above, parent, copy) in fields {
        b.extend_from_slice(&(name.len() as i32).to_le_bytes());
        b.extend_from_slice(name.as_bytes());
        b.push(kind);
        b.push(flags);
        b.extend_from_slice(&number.to_le_bytes());
        b.extend_from_slice(&ignore_above.to_le_bytes());
        b.extend_from_slice(&7i32.to_le_bytes());
        b.push(3);
        b.push(2);
        b.extend_from_slice(&parent.to_le_bytes());
        b.extend_from_slice(&(copy.len() as i32).to_le_bytes());
        for c in copy {
            b.extend_from_slice(&c.to_le_bytes());
        }
    }
    b.extend_from_slice(&nested_limit.to_le_bytes());
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

/// The root document's fragment of what [`parse`] wrote, after the child documents.
fn root(out: &[u8]) -> &[u8] {
    let mut at = 4;
    for _ in 0..i32_at(out, 0) {
        at += 8 + i32_at(out, at + 4) as usize;
    }
    &out[at..]
}

fn counts(out: &[u8]) -> [i32; 4] {
    let out = root(out);
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
    let indexed = |src: &str, out: &mut Vec<u8>| {
        assert!(parse(&p, src.as_bytes(), out), "{src}");
        counts(out)[1]
    };
    assert_eq!(indexed(r#"{"k":"ab"}"#, &mut out), 1);
    assert_eq!(indexed(r#"{"k":"é"}"#, &mut out), 1);
    assert_eq!(indexed(r#"{"k":"abc"}"#, &mut out), 0);
    assert_eq!(indexed(r#"{"k":"😀a"}"#, &mut out), 0);
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
    assert!(register(&blob(&[("a", 11, 1, 1, 1)])).is_err());
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
    assert!(out.len() > 36);
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
    let out = root(out);
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
fn null_on_a_name_the_plan_does_not_hold_is_no_value() {
    let with =
        shaped_run(r#"{"gone":null,"tags":"t","o":{"nope":null,"k":"a"},"also":null}"#).unwrap();
    let without = shaped_run(r#"{"tags":"t","o":{"k":"a"}}"#).unwrap();
    assert_eq!(with, without);
    for src in [
        r#"{"gone":1}"#,
        r#"{"gone":null,"gone2":"x"}"#,
        r#"{"gone":nul}"#,
        r#"{"gone":nullx}"#,
        r#"{"gone":[null]}"#,
        r#"{"gone":null"#,
        r#"{"gone" null}"#,
    ] {
        assert!(shaped_run(src).is_none(), "{src}");
    }
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
        out[36..].sort_unstable();
    }
    assert_eq!(sa[36..].len(), sb[36..].len());
    // a key that is a dotted path to a real field stays refused whatever the hints hold
    assert!(!parse(&p, br#"{"o.k":"a"}"#, &mut a));
    assert!(parse(&p, one.as_bytes(), &mut a));
    assert!(!parse(&p, br#"{"o":{"k.x":"a"}}"#, &mut a));
}

/// The children and root fragments of what [`parse`] wrote.
fn split(out: &[u8]) -> (Vec<(i32, Vec<u8>)>, Vec<u8>) {
    let mut kids = Vec::new();
    let mut at = 4;
    for _ in 0..i32_at(out, 0) {
        let len = i32_at(out, at + 4) as usize;
        kids.push((i32_at(out, at), out[at + 8..at + 8 + len].to_vec()));
        at += 8 + len;
    }
    (kids, out[at..].to_vec())
}

fn frag_counts(f: &[u8]) -> [i32; 4] {
    [i32_at(f, 0), i32_at(f, 4), i32_at(f, 8), i32_at(f, 12)]
}

fn run_with(p: &Plan, src: &str) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    parse(p, src.as_bytes(), &mut out).then_some(out)
}

fn geo_plan() -> Arc<Plan> {
    plan(
        register(&blob(&[
            ("loc", GEO_POINT, 3, 30, 0),
            ("ok", BOOLEAN, 3, 31, 0),
        ]))
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn a_geo_point_is_a_point_and_a_doc_value_in_every_form() {
    let p = geo_plan();
    for src in [
        r#"{"loc":{"lat":40.7,"lon":-74.0}}"#,
        r#"{"loc":{"lon":-74,"lat":40.7}}"#,
        r#"{"loc":[-74.0,40.7]}"#,
        r#"{"loc":"40.7,-74.0"}"#,
        r#"{"loc":" 40.7 , -74.0"}"#,
    ] {
        let out = run_with(&p, src).unwrap();
        assert_eq!(counts(&out), [0, 0, 1, 1], "{src}");
    }
    let (_, f) = split(&run_with(&p, r#"{"loc":[2.5,-90],"ok":null}"#).unwrap());
    let points =
        &f[32 + i32_at(&f, 16) as usize + i32_at(&f, 20) as usize + i32_at(&f, 24) as usize..];
    assert_eq!(i32_at(points, 0), 30);
    assert_eq!(i32_at(points, 4), 8);
    // latitude -90 encodes to i32::MIN, longitude 2.5 to 2.5 / (360 / 2^32)
    assert_eq!(&points[8..12], &[0, 0, 0, 0]);
    assert_eq!(
        counts(&run_with(&p, r#"{"loc":[[1,2],[3,4],{"lat":5,"lon":6},"7,8"]}"#).unwrap()),
        [0, 0, 4, 4]
    );
    assert_eq!(
        counts(&run_with(&p, r#"{"loc":[]}"#).unwrap()),
        [0, 0, 0, 0]
    );
    assert_eq!(
        counts(&run_with(&p, r#"{"loc":null}"#).unwrap()),
        [0, 0, 0, 0]
    );
}

#[test]
fn geo_points_the_mapper_alone_reads_are_refused() {
    let p = geo_plan();
    for src in [
        r#"{"loc":[1,2,3]}"#,
        r#"{"loc":[1]}"#,
        r#"{"loc":{"lat":91,"lon":0}}"#,
        r#"{"loc":{"lat":0,"lon":181}}"#,
        r#"{"loc":{"lat":0}}"#,
        r#"{"loc":{"lat":0,"lon":0,"x":1}}"#,
        r#"{"loc":{"lat":"1","lon":2}}"#,
        r#"{"loc":"drm3btev3e86"}"#,
        r#"{"loc":"POINT (1 2)"}"#,
        r#"{"loc":"1,2,3"}"#,
        r#"{"loc":"1e1,2"}"#,
        r#"{"loc":"NaN,2"}"#,
        r#"{"loc":"+1,2"}"#,
        r#"{"loc":[null]}"#,
        r#"{"loc":[[1,2],null]}"#,
        r#"{"loc":true}"#,
        r#"{"loc":5}"#,
    ] {
        assert!(run_with(&p, src).is_none(), "{src}");
    }
}

#[test]
fn a_boolean_is_a_term_and_a_doc_value() {
    let p = geo_plan();
    assert_eq!(
        counts(&run_with(&p, r#"{"ok":true}"#).unwrap()),
        [0, 1, 1, 0]
    );
    assert_eq!(
        counts(&run_with(&p, r#"{"ok":false}"#).unwrap()),
        [0, 1, 1, 0]
    );
    // one inverted entry for the array, a doc value for each value
    let out = run_with(&p, r#"{"ok":[true,false,true]}"#).unwrap();
    assert_eq!(counts(&out), [0, 1, 3, 0]);
    assert_eq!(i32_at(inverted(&out), 5), 2);
    for src in [
        r#"{"ok":"true"}"#,
        r#"{"ok":1}"#,
        r#"{"ok":tru}"#,
        r#"{"ok":truex}"#,
        r#"{"ok":[true,]}"#,
    ] {
        assert!(run_with(&p, src).is_none(), "{src}");
    }
}

fn nested_plan(limit: i32) -> Arc<Plan> {
    plan(
        register(&blob_with(
            &[
                ("title", TEXT, 1, 40, 0, -1, &[]),
                ("c", NESTED, 0, 0, 0, -1, &[]),
                ("c.user", KEYWORD, 3, 41, i32::MAX, -1, &[]),
                ("c.stars", INTEGER, 3, 42, 0, -1, &[]),
                ("c.d", NESTED, 0, 0, 0, -1, &[]),
                ("c.d.x", LONG, 3, 43, 0, -1, &[]),
                ("o", OBJECT, 0, 0, 0, -1, &[]),
                ("o.k", KEYWORD, 3, 44, i32::MAX, -1, &[]),
            ],
            limit,
        ))
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn nested_objects_are_documents_before_their_parent() {
    let p = nested_plan(10);
    let out = run_with(
        &p,
        r#"{"title":"t","c":[{"user":"a","stars":1,"d":[{"x":1},{"x":2}]},{"user":"b"}],"o":{"k":"z"}}"#,
    )
    .unwrap();
    let (kids, root) = split(&out);
    // the mapper's order: the grandchildren, their parent, then the next child
    let order: Vec<i32> = kids.iter().map(|k| k.0).collect();
    assert_eq!(order, [4, 4, 1, 1]);
    assert_eq!(frag_counts(&kids[0].1), [0, 0, 1, 1]);
    assert_eq!(frag_counts(&kids[2].1), [0, 1, 2, 1]);
    assert_eq!(frag_counts(&kids[3].1), [0, 1, 1, 0]);
    assert_eq!(frag_counts(&root), [0, 2, 1, 0]);
    // a single object, an empty one, nulls and an empty array
    let (kids, _) = split(&run_with(&p, r#"{"c":{}}"#).unwrap());
    assert_eq!(kids.len(), 1);
    assert_eq!(frag_counts(&kids[0].1), [0, 0, 0, 0]);
    for src in [r#"{"c":null}"#, r#"{"c":[]}"#, r#"{"c":[null]}"#] {
        assert_eq!(split(&run_with(&p, src).unwrap()).0.len(), 0, "{src}");
    }
}

#[test]
fn nested_limits_and_shapes_are_checked() {
    let p = nested_plan(3);
    // the limit is 3 nested documents for the whole document
    assert!(run_with(&p, r#"{"c":[{},{"d":[{}]}]}"#).is_some());
    assert!(run_with(&p, r#"{"c":[{},{"d":[{},{}]}]}"#).is_none());
    assert!(run_with(&p, r#"{"c":[{},{},{},{}]}"#).is_none());
    for src in [
        r#"{"c":5}"#,
        r#"{"c":[1]}"#,
        r#"{"c":[{"user":"a","user":"b"}]}"#,
        r#"{"c":{"z":1}}"#,
        r#"{"c":{"d":{"x":"1"}}}"#,
    ] {
        assert!(run_with(&p, src).is_none(), "{src}");
    }
    // a failed document leaves nothing behind for the next
    let ok = run_with(&p, r#"{"c":[{"user":"a"}]}"#).unwrap();
    assert!(run_with(&p, r#"{"c":[{"user":"a"},{"zzz":1}]}"#).is_none());
    assert_eq!(run_with(&p, r#"{"c":[{"user":"a"}]}"#).unwrap(), ok);
}

#[test]
fn arrays_of_objects_inside_a_nested_object_stay_in_its_document() {
    let id = register(&blob_with(
        &[
            ("c", NESTED, 0, 0, 0, -1, &[]),
            ("c.o", OBJECT, 0, 0, 0, -1, &[]),
            ("c.o.t", KEYWORD, 1, 50, i32::MAX, -1, &[]),
            ("t", KEYWORD, 1, 51, i32::MAX, -1, &[]),
        ],
        100,
    ))
    .unwrap();
    let p = plan(id).unwrap();
    let (kids, root) = split(
        &run_with(
            &p,
            r#"{"t":"r","c":[{"o":[{"t":"a"},{"t":"b"}]},{"o":[{"t":"a"}]}]}"#,
        )
        .unwrap(),
    );
    assert_eq!(kids.len(), 2);
    // one entry for c.o.t holding a and b; the second child holds a
    assert_eq!(frag_counts(&kids[0].1)[1], 1);
    assert_eq!(frag_counts(&kids[1].1)[1], 1);
    assert_eq!(frag_counts(&root)[1], 1);
}

fn merged_plan() -> Arc<Plan> {
    // 0 msg (text) with sub 3 (msg.keyword) and copy to 2; 1 tag (keyword) copies to 2; 2 all (text); 3 msg.keyword
    plan(
        register(&blob_with(
            &[
                ("msg", TEXT, 1, 60, 0, -1, &[2]),
                ("tag", KEYWORD, 3, 61, i32::MAX, -1, &[2]),
                ("all", TEXT, 1, 62, 0, -1, &[]),
                ("msg.keyword", KEYWORD, 3, 63, 4, 0, &[]),
            ],
            100,
        ))
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn multi_fields_and_copy_to_write_one_entry_per_field() {
    let p = merged_plan();
    let out = run_with(&p, r#"{"msg":"abc","tag":"x"}"#).unwrap();
    // inverted: msg, msg.keyword, all, tag; doc values: msg.keyword, tag
    assert_eq!(counts(&out), [0, 4, 2, 0]);
    // arrays: still one entry per field
    let out = run_with(&p, r#"{"msg":["a","b"],"tag":["x","y"]}"#).unwrap();
    assert_eq!(counts(&out), [0, 4, 4, 0]);
    assert_eq!(
        counts(&run_with(&p, r#"{"all":"direct","msg":"m"}"#).unwrap()),
        [0, 3, 1, 0]
    );
    // a value too long for the sub-field is skipped there only
    assert_eq!(
        counts(&run_with(&p, r#"{"msg":"abcde"}"#).unwrap()),
        [0, 2, 0, 0]
    );
    // a multi-field has no key of its own
    assert!(run_with(&p, r#"{"msg.keyword":"a"}"#).is_none());
    assert!(run_with(&p, r#"{"msg":"a","msg":"b"}"#).is_none());
}

#[test]
fn a_plan_with_a_bad_multi_field_or_copy_to_is_refused() {
    let bad = |fields: &[FieldRow]| register(&blob_with(fields, 100)).is_err();
    // a multi-field of a number; a copy to a number; a copy source that is a number
    assert!(bad(&[
        ("a", INTEGER, 3, 1, 0, -1, &[]),
        ("a.k", KEYWORD, 3, 2, 9, 0, &[])
    ]));
    assert!(bad(&[
        ("a", KEYWORD, 3, 1, 9, -1, &[1]),
        ("b", INTEGER, 3, 2, 0, -1, &[])
    ]));
    assert!(bad(&[
        ("a", INTEGER, 3, 1, 0, -1, &[1]),
        ("b", KEYWORD, 3, 2, 9, -1, &[])
    ]));
    // a chain of copies; a multi-field of a multi-field; a parent out of range
    assert!(bad(&[
        ("a", KEYWORD, 3, 1, 9, -1, &[1]),
        ("b", KEYWORD, 3, 2, 9, -1, &[0])
    ]));
    assert!(bad(&[
        ("a", KEYWORD, 3, 1, 9, -1, &[]),
        ("a.b", KEYWORD, 3, 2, 9, 0, &[]),
        ("a.b.c", KEYWORD, 3, 3, 9, 1, &[])
    ]));
    assert!(bad(&[("a", KEYWORD, 3, 1, 9, 5, &[])]));
}

#[test]
fn a_date_as_epoch_millis_needs_five_digits() {
    let p = logs();
    assert_eq!(
        counts(&run_with(&p, r#"{"@timestamp":1700000000000}"#).unwrap()),
        [0, 0, 1, 1]
    );
    assert_eq!(
        counts(&run_with(&p, r#"{"@timestamp":12345}"#).unwrap()),
        [0, 0, 1, 1]
    );
    for src in [
        r#"{"@timestamp":2026}"#,
        r#"{"@timestamp":0}"#,
        r#"{"@timestamp":1234}"#,
        r#"{"@timestamp":-1700000000000}"#,
        r#"{"@timestamp":1.7e12}"#,
        r#"{"@timestamp":1700000000000.5}"#,
        r#"{"@timestamp":1700000000000000000}"#,
        r#"{"@timestamp":"1700000000000"}"#,
    ] {
        assert!(run_with(&p, src).is_none(), "{src}");
    }
}

#[test]
fn a_keyword_over_ignore_above_is_skipped() {
    let id = register(&blob_with(
        &[
            ("t", TEXT, 1, 70, 0, -1, &[]),
            ("t.keyword", KEYWORD, 3, 71, 3, 0, &[]),
            ("k", KEYWORD, 3, 72, 3, -1, &[]),
        ],
        100,
    ))
    .unwrap();
    let p = plan(id).unwrap();
    assert_eq!(
        counts(&run_with(&p, r#"{"t":"abc"}"#).unwrap()),
        [0, 2, 1, 0]
    );
    // over 3: the text only
    assert_eq!(
        counts(&run_with(&p, r#"{"t":"abcd"}"#).unwrap()),
        [0, 1, 0, 0]
    );
    // a keyword array skips just the long values
    assert_eq!(
        counts(&run_with(&p, r#"{"k":["ab","abcd","c"]}"#).unwrap()),
        [0, 1, 2, 0]
    );
    assert_eq!(
        counts(&run_with(&p, r#"{"k":["abcd","abcde"]}"#).unwrap()),
        [0, 0, 0, 0]
    );
}

/// `nested_plan`'s blob with the metadata trailer: `_id` 1, `_seq_no` 2, `_primary_term` 3,
/// `_version` 4, `_source` 5, `_nested_path` 6, and a path per plan entry.
fn nested_plan_with_meta(seq_point: bool) -> Arc<Plan> {
    let mut b = blob_with(
        &[
            ("title", KEYWORD, 3, 40, i32::MAX, -1, &[]),
            ("c", NESTED, 0, 0, 0, -1, &[]),
            ("c.user", KEYWORD, 3, 41, i32::MAX, -1, &[]),
        ],
        100,
    );
    for n in [1i32, 2, 3, 4, 5, 6] {
        b.extend_from_slice(&n.to_le_bytes());
    }
    b.push(u8::from(seq_point));
    b.extend_from_slice(&3i32.to_le_bytes());
    for path in ["", "c", ""] {
        b.extend_from_slice(&(path.len() as i32).to_le_bytes());
        b.extend_from_slice(path.as_bytes());
    }
    plan(register(&b).unwrap()).unwrap()
}

#[test]
fn a_prepared_source_carries_the_metadata_the_mapper_adds() {
    let plan = nested_plan_with_meta(true);
    let src = br#"{"title":"hello world","c":[{"user":"a"},{"user":"b"}]}"#;
    let prepared = prepare(&plan, src, b"id1").unwrap().unwrap();
    assert_eq!(prepared.len(), 3);
    assert_eq!(prepared.id(), b"id1");
    assert_eq!(prepared.objects, [1, 1]);
    let docs = prepared.finish(&plan, 7, 2, 9);
    let number = |fields: &[lucene_codecs::stored_fields::StoredField]| {
        fields.iter().map(|f| f.field_number).collect::<Vec<_>>()
    };
    for child in &docs[..2] {
        // `_id` and `_nested_path` after the entries read from the source; no stored `_id`.
        let inverted = number_of(&child.fields.inverted);
        assert_eq!(&inverted[inverted.len() - 2..], &[1, 6]);
        assert!(child.stored.is_empty());
        assert_eq!(number(&child.fields.doc_values), [41, 2, 4]);
        assert_eq!(number(&child.fields.points), [2]);
    }
    let root = &docs[2];
    assert_eq!(number(&root.stored), [1, 5]);
    assert_eq!(number(&root.fields.doc_values), [40, 2, 3, 4]);
    assert_eq!(
        format!("{:?}", root.fields.doc_values[2].value),
        "Long(2)",
        "the primary term"
    );
    assert_eq!(
        format!("{:?}", root.fields.points[0].value),
        format!("Binary({:?})", vec![0x80u8, 0, 0, 0, 0, 0, 0, 7])
    );
}

fn number_of(fields: &[lucene_index::index_writer::InvertedField]) -> Vec<i32> {
    fields.iter().map(|f| f.field_number).collect()
}

#[test]
fn a_prepared_source_without_the_seq_no_point_has_none() {
    let plan = nested_plan_with_meta(false);
    let prepared = prepare(&plan, br#"{"title":"x"}"#, b"i").unwrap().unwrap();
    let docs = prepared.finish(&plan, 0, 1, 1);
    assert_eq!(docs.len(), 1);
    assert!(docs[0].fields.points.is_empty());
}

#[test]
fn a_prepare_is_refused_like_a_parse() {
    let plan = nested_plan_with_meta(true);
    assert!(prepare(&plan, br#"{"unmapped":1}"#, b"i")
        .unwrap()
        .is_none());
    // a plan registered without the metadata cannot prepare
    assert!(prepare(&nested_plan(100), br#"{"title":"x"}"#, b"i")
        .unwrap()
        .is_none());
}

#[test]
fn a_negative_seq_no_sorts_below_a_positive_one() {
    use super::prepared::long_point_for_test;
    assert!(long_point_for_test(-1) < long_point_for_test(0));
    assert!(long_point_for_test(0) < long_point_for_test(1));
}
