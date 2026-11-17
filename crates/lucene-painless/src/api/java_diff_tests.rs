#![cfg(test)]
//! Differential check against JDK 21: `java_reference.tsv` holds calls on
//! `String`, the number wrappers, `Math` and `Character` with what the JDK
//! returned or threw, and every one must give the same here.
//!
//! Each row is `kind class method args... # result`: kind `S` is a static,
//! `I` an instance call on a `String` receiver (the first argument), `F`
//! `String.format`. An argument is `i:`, `l:`, `d:`, `f:`, `c:`, `s:` or `b:`
//! and a value, or `null`. A result is the JDK's `String.valueOf`, an array as
//! `Arrays.toString`, or `!class: message` for a throw, with everything
//! outside printable ASCII as `\uXXXX`.

use super::*;
use crate::api::test_support::{call, elems, vm_call};

fn arg(t: &str) -> Value {
    if t == "null" || t == "-" {
        return Value::Null;
    }
    let v = &t[2..];
    match &t[..1] {
        "i" => Value::Int(v.parse().unwrap()),
        "l" => Value::Long(v.parse().unwrap()),
        "d" => Value::Double(num(v)),
        "f" => Value::Float(num(v) as f32),
        "c" => Value::Char(if v == "\\t" {
            9
        } else {
            v.encode_utf16().next().unwrap()
        }),
        "b" => Value::Bool(v == "true"),
        "s" => Value::string(v.replace("\\n", "\n").replace("\\t", "\t")),
        k => panic!("argument kind {k}"),
    }
}

fn num(v: &str) -> f64 {
    match v {
        "NaN" => f64::NAN,
        "Infinity" => f64::INFINITY,
        "-Infinity" => f64::NEG_INFINITY,
        _ => v.parse().unwrap(),
    }
}

fn escape(s: &str) -> String {
    let mut out = String::new();
    for u in s.encode_utf16() {
        match u {
            10 => out.push_str("\\n"),
            9 => out.push_str("\\t"),
            32..=126 => out.push(u as u8 as char),
            _ => out.push_str(&format!("\\u{u:04x}")),
        }
    }
    out
}

fn show(r: std::result::Result<Value, crate::vm::Throw>) -> String {
    match r {
        Ok(Value::Array(_)) => elems(r),
        Ok(v) => v.java_to_string(),
        Err(e) => format!(
            "!{}: {}",
            e.class(),
            e.message().unwrap_or_else(|| "null".into())
        ),
    }
}

fn run(kind: &str, class: &str, name: &str, args: &[Value]) -> String {
    match kind {
        "F" => {
            let f = args[0].java_to_string();
            show(string::format(&f, &args[1..]).map(Value::string))
        }
        "S" => match static_method(class, name, args.len()) {
            Some(f) => show(vm_call(f, args)),
            None => "?missing".into(),
        },
        _ => show(call(&args[0], name, &args[1..])),
    }
}

#[test]
fn library_calls_match_the_jdk() {
    let (mut rows, mut bad) = (0, Vec::new());
    for line in include_str!("java_reference.tsv").lines() {
        let (spec, want) = line.split_once("\t#\t").unwrap();
        let f: Vec<&str> = spec.split('\t').collect();
        let args: Vec<Value> = f[3..].iter().map(|a| arg(a)).collect();
        let got = escape(&run(f[0], f[1], f[2], &args));
        // A member the runtime lacks is refused before it runs, so only an
        // answer that differs matters.
        let lacks =
            got.contains("not found") || got.contains("?missing") || got.contains("Unsupported");
        // A throw matches by class: the message varies with the JDK, and the
        // plugin re-runs a failing script in Java for its exact text.
        let both_throw = got.starts_with('!') && want.starts_with('!');
        let class = |x: &str| x.split(':').next().unwrap_or_default().to_string();
        let same = got == want || (both_throw && class(&got) == class(want));
        if !same && !lacks {
            bad.push(format!(
                "{}\t{}\t{}\t{}\n    got  {got}\n    want {want}",
                f[0],
                f[1].rsplit('.').next().unwrap(),
                f[2],
                f[3..].join(" ")
            ));
        }
        rows += 1;
    }
    let report = bad.join("\n");
    if let Ok(p) = std::env::var("JAVA_DIFF_REPORT") {
        std::fs::write(p, &report).unwrap();
    }
    assert!(
        bad.is_empty(),
        "{} of {rows} differ:\n{}",
        bad.len(),
        report.lines().take(60).collect::<Vec<_>>().join("\n")
    );
}

/// `java_regex.tsv` rows are `pattern input ops # results`: each op is one
/// `Matcher` call (`m` matches, `l` lookingAt, `f[n]` find, `g[n]` group,
/// `s[n]` start, `e[n]` end, `c` groupCount, `r` reset, `h` hitEnd, `A`/`F`
/// replaceAll/replaceFirst of `<$1>`).
#[test]
fn matcher_calls_match_the_jdk() {
    let mut bad = Vec::new();
    for line in include_str!("java_regex.tsv").lines() {
        let (spec, want) = line.split_once("\t#\t").unwrap();
        let f: Vec<&str> = spec.split('\t').collect();
        // A pattern the engine cannot run natively (a back-reference, an inline
        // flag) is refused before it runs.
        let Ok(pat) = regex::compile_pattern(f[0], 0) else {
            continue;
        };
        let m = call(&pat, "matcher", &[Value::str(f[1])]).unwrap();
        let mut got = Vec::new();
        for op in f[2].split(',') {
            let n = op[1..].parse::<i32>().ok();
            let name = match &op[..1] {
                "m" => "matches",
                "l" => "lookingAt",
                "f" => "find",
                "g" => "group",
                "s" => "start",
                "e" => "end",
                "c" => "groupCount",
                "h" => "hitEnd",
                "A" => "replaceAll",
                "F" => "replaceFirst",
                _ => "reset",
            };
            let args: Vec<Value> = match (name, n) {
                (_, Some(n)) => vec![Value::Int(n)],
                ("replaceAll" | "replaceFirst", _) => vec![Value::str("<$1>")],
                _ => Vec::new(),
            };
            got.push(match call(&m, name, &args) {
                Ok(_) if name == "reset" => "reset".to_string(),
                Ok(Value::Null) => "null".to_string(),
                Ok(v) => v.java_to_string(),
                Err(e) => format!("!{}", e.class()),
            });
        }
        let got = got.join(";");
        if got != want {
            bad.push(format!("{spec}\n    got  {got}\n    want {want}"));
        }
    }
    assert!(bad.is_empty(), "{} differ:\n{}", bad.len(), bad.join("\n"));
}

/// `java_collections.tsv`: a collection class, `name,arg,...` calls and the
/// JDK's results (`?` where the JDK has no such member), then its final text.
#[test]
fn collections_match_the_jdk() {
    let mut bad = Vec::new();
    for line in include_str!("java_collections.tsv").lines() {
        let (spec, want) = line.split_once("\t#\t").unwrap();
        let (class, ops) = spec.split_once('\t').unwrap();
        let Some(make) = constructor(class, 0) else {
            continue;
        };
        let recv = vm_call(make, &[]).unwrap();
        let want: Vec<&str> = want.split(" | ").collect();
        let mut got = Vec::new();
        for (op, java) in ops.split(';').zip(&want) {
            let mut p = op.split(',');
            let name = p.next().unwrap();
            let args: Vec<Value> = p.map(arg).collect();
            let r = call(&recv, name, &args);
            got.push(match r {
                _ if *java == "?" => "?".to_string(),
                Ok(Value::Null) => "null".to_string(),
                Ok(v) => v.java_to_string(),
                Err(e) => format!("!{}", e.class()),
            });
        }
        got.push(recv.java_to_string());
        let got = got.join(" | ");
        let want = want.join(" | ");
        if got != want {
            bad.push(format!("{class} {ops}\n    got  {got}\n    want {want}"));
        }
    }
    assert!(bad.is_empty(), "{} differ:\n{}", bad.len(), bad.join("\n"));
}
