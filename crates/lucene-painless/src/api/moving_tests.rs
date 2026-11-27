#![cfg(test)]
//! `MovingFunctions` against what OpenSearch 3.10.0 returned or threw for
//! 899 recorded calls (`fixtures/data/painless-moving`, written by
//! `fixtures/src/GenMovingFunctions.java`): every one must give the same
//! text here, doubles printed as Java prints them.

use super::moving::static_lookup;
use super::test_support::{stat, text};
use crate::value::Value;

const RECORDED: &str =
    include_str!("../../../../fixtures/data/painless-moving/java-moving-functions.txt");

fn double(t: &str) -> f64 {
    match t {
        "NaN" => f64::NAN,
        "Infinity" => f64::INFINITY,
        "-Infinity" => f64::NEG_INFINITY,
        _ => t.parse().expect("a double"),
    }
}

fn array(t: &str) -> Value {
    let inner = t.trim_start_matches('[').trim_end_matches(']');
    let items = if inner.is_empty() {
        Vec::new()
    } else {
        inner
            .split(", ")
            .map(|v| Value::Double(double(v)))
            .collect()
    };
    Value::array("double", items)
}

#[test]
fn every_recorded_call_gives_what_java_gave() {
    let mut checked = 0;
    for line in RECORDED.lines() {
        let (call, want) = line.split_once(" -> ").expect("a result");
        let (name, rest) = call.split_once(' ').expect("a name");
        let (arr, rest) = rest
            .split_once("] ")
            .map_or((rest.trim_end_matches(']'), ""), |(a, r)| (a, r));
        let mut args = vec![array(&format!("{arr}]"))];
        let rest: Vec<&str> = rest.split(' ').filter(|s| !s.is_empty()).collect();
        for (i, a) in rest.iter().enumerate() {
            args.push(match *a {
                "true" => Value::Bool(true),
                "false" => Value::Bool(false),
                n if name == "holtWinters" && i == 3 => Value::Int(n.parse().expect("an int")),
                n => Value::Double(double(n)),
            });
        }
        let got = text(stat(static_lookup_of, name, &args));
        let want = want.strip_prefix('!').unwrap_or(want);
        assert_eq!(got, want, "{line}");
        checked += 1;
    }
    assert_eq!(checked, 899);
}

fn static_lookup_of(name: &str, arity: usize) -> Option<crate::vm::ApiFn> {
    static_lookup(name, arity)
}

/// A script reaches the functions through its allowlist: the window as a
/// `double[]` handed to `MovingFunctions`.
#[test]
fn a_script_calls_the_functions_on_an_array() {
    use crate::ir::Constant;
    let none = Constant::Map(Vec::new());
    let run = |src: &str| crate::scripted_metric::reduce(src, &none, &[]).expect("runs");
    assert_eq!(
        run("double[] v = new double[] {1.0, 2.0, 6.0}; return MovingFunctions.unweightedAvg(v)"),
        Constant::Double(3.0)
    );
    assert_eq!(
        run("double[] v = new double[] {1.0, 2.0, 6.0}; return MovingFunctions.max(v)"),
        Constant::Double(6.0)
    );
}
