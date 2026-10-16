//! M14 T14.2 (`expressions`) against Lucene 10.5.0: what `JavascriptCompiler`
//! makes of 7,484 expressions -- Lucene's own tests' and random ones -- the
//! values it evaluates to bit for bit, the errors it reports, and
//! `VariableContext.parse`.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use lucene_search::expressions::variable_context::{self, Type};
use lucene_search::expressions::{compile_with, default_functions, Error, ExprFunction};
use lucene_search::values_source::{BoxDoubleValues, ConstantDoubleValues};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/expressions")
}

/// `AnalysisRows.esc` undone.
fn unesc(s: &str) -> String {
    let mut out: Vec<u16> = Vec::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            let mut buf = [0u16; 2];
            out.extend_from_slice(c.encode_utf16(&mut buf));
            continue;
        }
        match it.next() {
            Some('u') => {
                let h: String = it.by_ref().take(4).collect();
                out.push(u16::from_str_radix(&h, 16).unwrap());
            }
            Some('t') => out.push(9),
            Some('n') => out.push(10),
            Some('r') => out.push(13),
            Some(o) => out.push(o as u16),
            None => out.push(u16::from(b'\\')),
        }
    }
    String::from_utf16_lossy(&out)
}

const SETS: [[f64; 5]; 4] = [
    [0.0, 1.0, -1.0, 2.5, 3.0],
    [f64::NAN, f64::INFINITY, -0.0, 1e20, f64::NEG_INFINITY],
    [3.0, 7.9, -7.9, (1u64 << 40) as f64, 0.5],
    [
        -2.5e-300,
        i64::MAX as f64,
        -9.3e18,
        2147483648.0,
        -2147483649.5,
    ],
];

/// `Double.doubleToLongBits`.
fn bits(d: f64) -> u64 {
    if d.is_nan() {
        0x7ff8_0000_0000_0000
    } else {
        d.to_bits()
    }
}

fn custom() -> HashMap<String, ExprFunction> {
    let mut m = default_functions();
    m.insert("foo".into(), ExprFunction::new(0, |_| 5.0));
    m.insert("bar".into(), ExprFunction::new(1, |a| a[0] * 3.0 + 1.0));
    m.insert(
        "method.method".into(),
        ExprFunction::new(3, |a| a[0] * 100.0 + a[1] * 10.0 + a[2]),
    );
    m
}

#[test]
fn compiled_expressions_equal_lucenes() {
    let want = fs::read_to_string(dir().join("expressions.tsv")).unwrap();
    let (defaults, custom) = (default_functions(), custom());
    let (mut checked, mut ok_rows, mut bad) = (0usize, 0usize, Vec::new());
    for row in want.lines().filter(|r| r.starts_with("E\t")) {
        let c: Vec<&str> = row.split('\t').collect();
        let functions = if c[1] == "D" { &defaults } else { &custom };
        let text = unesc(c[2]);
        checked += 1;
        let got = compile_with(&text, functions);
        let mismatch = if c[3] == "OK" {
            ok_rows += 1;
            match got {
                Err(e) => Some(format!("compile failed: {e}")),
                Ok(e) => {
                    // joined with U+001F, which `esc` escapes
                    let joined = unesc(c[4]);
                    let vars: Vec<String> = joined
                        .split('\u{1f}')
                        .filter(|v| !v.is_empty())
                        .map(str::to_string)
                        .collect();
                    if e.variables != vars {
                        Some(format!("variables {:?} want {vars:?}", e.variables))
                    } else {
                        let expected: Vec<&str> = c[5].split(',').collect();
                        let mut problem = None;
                        for (k, set) in SETS.iter().enumerate() {
                            let mut vals: Vec<BoxDoubleValues<'_>> = (0..e.variables.len())
                                .map(|i| {
                                    Box::new(ConstantDoubleValues(set[(i + k) % set.len()]))
                                        as BoxDoubleValues<'_>
                                })
                                .collect();
                            let v = bits(e.evaluate(&mut vals).unwrap());
                            if format!("{v:x}") != expected[k] {
                                problem = Some(format!(
                                    "set {k}: got {v:x} ({}) want {}",
                                    f64::from_bits(v),
                                    expected[k]
                                ));
                                break;
                            }
                        }
                        problem
                    }
                }
            }
        } else {
            // ERR, class, offset, message
            match got {
                Ok(_) => Some("compiled, Java refused".to_string()),
                Err(Error::NumberFormat(_)) => {
                    (c[4] != "NumberFormatException").then(|| format!("class {}", c[4]))
                }
                Err(Error::Parse(p)) => {
                    let message = unesc(c.get(6).copied().unwrap_or(""));
                    // a mismatched token's message lists ANTLR's expected
                    // token set, which is not reproduced; every other
                    // message is compared whole
                    let compared = !message.contains(" was expecting ");
                    if c[4] != "ParseException" || p.error_offset.to_string() != c[5] {
                        Some(format!(
                            "{} at {} want {} at {}: {}",
                            "ParseException", p.error_offset, c[4], c[5], message
                        ))
                    } else if compared && p.message != message {
                        Some(format!("message {:?} want {message:?}", p.message))
                    } else {
                        None
                    }
                }
            }
        };
        if let Some(m) = mismatch {
            bad.push(format!("{} {text:?}: {m}", c[1]));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {checked} differ:\n{}",
        bad.len(),
        bad.iter().take(12).cloned().collect::<Vec<_>>().join("\n")
    );
    assert!(
        checked > 7000 && ok_rows > 2000,
        "{checked} rows, {ok_rows} ok"
    );
}

fn kind(t: Type) -> &'static str {
    match t {
        Type::Member => "MEMBER",
        Type::StrIndex => "STR_INDEX",
        Type::IntIndex => "INT_INDEX",
        Type::Method => "METHOD",
    }
}

#[test]
fn variable_contexts_equal_lucenes() {
    let want = fs::read_to_string(dir().join("expressions.tsv")).unwrap();
    let (mut checked, mut bad) = (0usize, Vec::new());
    for row in want.lines().filter(|r| r.starts_with("V\t")) {
        let c: Vec<&str> = row.split('\t').collect();
        let variable = unesc(c[1]);
        let expected = c.get(2).copied().unwrap_or("");
        let got = match variable_context::parse(&variable) {
            Err(_) => "ERR".to_string(),
            Ok(cs) => cs
                .iter()
                .map(|v| {
                    format!(
                        "{}:{}:{}",
                        kind(v.kind),
                        v.text.clone().map_or("null".to_string(), |t| t),
                        v.integer
                    )
                })
                .collect::<Vec<_>>()
                .join("\u{1f}"),
        };
        let expected = if expected.starts_with("ERR") {
            "ERR".to_string()
        } else {
            // escaped texts: compare after undoing `esc` on each part
            expected
                .split('\u{1f}')
                .map(|p| {
                    let (k, rest) = p.split_once(':').unwrap();
                    let (t, i) = rest.rsplit_once(':').unwrap();
                    format!("{k}:{}:{i}", unesc(t))
                })
                .collect::<Vec<_>>()
                .join("\u{1f}")
        };
        checked += 1;
        if got != expected {
            bad.push(format!("{variable:?}: got {got:?} want {expected:?}"));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {checked} differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
    assert!(checked > 20, "{checked} rows");
}
