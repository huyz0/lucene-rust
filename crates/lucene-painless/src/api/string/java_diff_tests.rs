#![cfg(test)]
//! Differential checks against JDK 21: `String.format` of doubles and floats
//! and `Base64.getDecoder().decode`, from tables the JDK produced.
use super::*;
use crate::api::test_support::{s, text};

fn parse(v: &str) -> f64 {
    match v {
        "NaN" => f64::NAN,
        "Infinity" => f64::INFINITY,
        "-Infinity" => f64::NEG_INFINITY,
        _ => v.parse().unwrap(),
    }
}

#[test]
fn float_formats_match_the_jdk() {
    let (mut rows, mut bad) = (0, Vec::new());
    for line in include_str!("java_float_format.tsv").lines() {
        let mut p = line.split('\t');
        let (f, v, want) = (p.next().unwrap(), p.next().unwrap(), p.next().unwrap());
        let arg = match v.strip_suffix('f') {
            Some(x) => Value::Float(parse(x) as f32),
            None => Value::Double(parse(v)),
        };
        let got = text(format(f, &[arg]).map(Value::string));
        let want = want
            .strip_prefix('!')
            .map_or(want.to_string(), |c| format!("{c}: "));
        if !(got == want || (want.ends_with(": ") && got.starts_with(&want))) {
            bad.push(format!("{f} of {v}: {got} != {want}"));
        }
        rows += 1;
    }
    assert!(
        bad.is_empty(),
        "{} of {rows}:\n{}",
        bad.len(),
        bad.join("\n")
    );
    assert!(rows > 1000);
}

#[test]
fn base64_decoding_matches_the_jdk() {
    for line in include_str!("java_base64.tsv").lines() {
        let (input, want) = line.split_once('\t').unwrap();
        if input.contains('?') {
            continue; // the table lost the non-ASCII input to the console encoding
        }
        let input = input.replace("\\n", "\n");
        let got = match base64_decode(&input) {
            Ok(b) => format!(
                "ok:{}",
                b.iter().map(|x| format!("{x:02x}")).collect::<String>()
            ),
            Err(m) => format!("IAE:{m}"),
        };
        assert_eq!(got, want, "decode of {input:?}");
    }
    assert_eq!(
        base64_decode("\u{e9}AAA"),
        Err("Illegal base64 character -17".to_string())
    );
    assert_eq!(
        base64_decode("\u{4e2d}AAA"),
        Err("Illegal base64 character 3f".to_string())
    );
    assert_eq!(
        text(call_decode("A")),
        "java.lang.IllegalArgumentException: Input byte[] should at least have 2 bytes for base64 bytes"
    );
}

fn call_decode(x: &str) -> Result<Value, crate::vm::Throw> {
    crate::api::test_support::call(&s(x), "decodeBase64", &[])
}
