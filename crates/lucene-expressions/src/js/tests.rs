#![cfg(test)]
use super::*;
use lucene_search::values_source::ConstantDoubleValues;

fn run(src: &str, vars: &[f64]) -> f64 {
    let e = compile(src).unwrap();
    let mut vals: Vec<BoxDoubleValues<'_>> = vars
        .iter()
        .map(|v| Box::new(ConstantDoubleValues(*v)) as BoxDoubleValues<'_>)
        .collect();
    e.evaluate(&mut vals).unwrap()
}

#[test]
fn precedence_and_associativity() {
    assert_eq!(run("1 + 2 * 3", &[]), 7.0);
    assert_eq!(run("10 - 4 - 3", &[]), 3.0);
    assert_eq!(run("-2 * 3", &[]), -6.0);
    assert_eq!(run("1 ? 2 : 0 ? 3 : 4", &[]), 2.0);
    assert_eq!(run("0 ? 2 : 0 ? 3 : 4", &[]), 4.0);
    assert_eq!(run("1 < 2 == 1", &[]), 1.0);
    assert_eq!(run("1 | 6 & 3 ^ 1", &[]), 3.0);
}

#[test]
fn jvm_conversions_match() {
    // `!0.5`: d2i(0.5) is 0, so it is true
    assert_eq!(run("!0.5", &[]), 1.0);
    assert_eq!(run("!1", &[]), 0.0);
    assert_eq!(run("~5", &[]), -6.0);
    assert_eq!(run("1 << 40", &[]), (1i64 << 40) as f64);
    assert_eq!(run("-1 >>> 60", &[]), 15.0);
    assert_eq!(run("1.9 & 3", &[]), 1.0);
    assert_eq!(run("0x10 + 010", &[]), 24.0);
}

#[test]
fn variables_are_distinct_and_normalised() {
    let e = compile("a + b + a + doc[\"x\"].value").unwrap();
    assert_eq!(e.variables, vec!["a", "b", "doc['x'].value"]);
    assert_eq!(run("a * a", &[3.0]), 9.0);
}

#[test]
fn functions_and_their_arity() {
    assert_eq!(run("max(1, 2) + min(5, 4)", &[]), 6.0);
    assert_eq!(run("sqrt(16)", &[]), 4.0);
    let e = compile("tan(1, 1)").unwrap_err();
    assert!(matches!(e, ExprError::Parse(p) if p.error_offset == 0));
    let e = compile("1 + nosuch(2)").unwrap_err();
    assert!(matches!(e, ExprError::Parse(p) if p.error_offset == 4));
    // a custom function past four arguments, and one over a variable
    let mut functions = default_functions();
    functions.insert(
        "digits".into(),
        ExprFunction::new(6, |a| a.iter().fold(0.0, |n, d| n * 10.0 + d)),
    );
    let e = compile_with("digits(1, 2, 3, x, 5, 6) + abs(-x)", &functions).unwrap();
    let mut vals: Vec<BoxDoubleValues<'_>> = vec![Box::new(ConstantDoubleValues(4.0))];
    assert_eq!(e.evaluate(&mut vals).unwrap(), 123_460.0);
    assert!(format!("{e:?}").starts_with("Expression { source_text: \"digits("));
}

/// Values whose every read fails.
struct Failing;

impl lucene_search::values_source::DoubleValues for Failing {
    fn advance_exact(&mut self, _doc: i32) -> lucene_search::Result<bool> {
        Ok(true)
    }
    fn double_value(&mut self) -> lucene_search::Result<f64> {
        Err(lucene_search::Error::IllegalState("unreadable".into()))
    }
}

#[test]
fn a_failed_read_is_the_evaluations_error() {
    // the first failure is reported, whichever branch read it
    for src in ["x * 2 + 1", "x > 0 ? 1 : x", "(x << 1) | 1"] {
        let e = compile(src).unwrap();
        let mut vals: Vec<BoxDoubleValues<'_>> = vec![Box::new(Failing)];
        let err = e.evaluate(&mut vals).unwrap_err();
        assert!(err.to_string().contains("unreadable"), "{src}: {err}");
    }
    // fewer values than variables is refused, not an out-of-bounds read
    let e = compile("x + y").unwrap();
    let mut vals: Vec<BoxDoubleValues<'_>> = vec![Box::new(ConstantDoubleValues(1.0))];
    let err = e.evaluate(&mut vals).unwrap_err().to_string();
    assert!(err.contains("1 values for the 2 variables"), "{err}");
}

#[test]
fn syntax_errors_carry_the_offending_offset() {
    for (src, at) in [
        ("100 100", 4),
        ("7*/-8", 2),
        ("", 0),
        ("()", 1),
        ("0y1234", 1),
    ] {
        match compile(src) {
            Err(ExprError::Parse(p)) => assert_eq!(p.error_offset, at, "{src}"),
            other => panic!("{src}: {other:?}"),
        }
    }
    assert!(matches!(compile("a @ b"), Err(ExprError::Parse(p)) if p.error_offset == 2));
}

#[test]
fn oversized_literals_are_number_format_errors() {
    assert!(matches!(
        compile("0xffffffffffffffff"),
        Err(ExprError::NumberFormat(_))
    ));
}

/// Trees past [`MAX_DEPTH`] are refused, whichever way they nest; ones at
/// the limit compile and evaluate (on a test thread's 2 MiB stack, in a
/// debug build).
#[test]
fn nesting_is_bounded() {
    let deep = |n: usize| {
        [
            format!("{}1{}", "(".repeat(n), ")".repeat(n)),
            format!("{}1", "-".repeat(n)),
            format!("{}1{}", "1-(".repeat(n), ")".repeat(n)),
            format!("{}1{}", "abs(".repeat(n), ")".repeat(n)),
            format!("{}1{}", "1?".repeat(n), ":0".repeat(n)),
        ]
    };
    // a parenthesis or a conditional's branch opens two parser frames, a
    // parenthesised right operand three
    for text in deep(MAX_DEPTH as usize / 3 - 1) {
        run(&text, &[]);
    }
    for text in deep(MAX_DEPTH as usize + 1)
        .into_iter()
        .chain(deep(200_000))
    {
        match compile(&text) {
            Err(ExprError::Parse(p)) => {
                assert!(p.message.ends_with("Nesting level too deep"), "{p:?}")
            }
            other => panic!("{:?}", other.map(|e| e.source_text.len())),
        }
    }
    // The deepest tree of each shape compiles and evaluates on a quarter
    // of a JVM thread's default 1 MiB stack in an optimised build (it
    // needs between 128 and 256 KiB); an unoptimised one needs about
    // 1.5 MiB.
    let stack = if cfg!(debug_assertions) {
        4 << 20
    } else {
        256 << 10
    };
    std::thread::Builder::new()
        .stack_size(stack)
        .spawn(move || {
            for shape in 0..5 {
                let deepest = (1..=MAX_DEPTH as usize)
                    .rev()
                    .map(|n| deep(n)[shape].clone())
                    .find(|text| compile(text).is_ok())
                    .unwrap();
                run(&deepest, &[]);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn long_chains_are_flat_and_typed_as_nested_operators() {
    // each level's run, past `CHAIN`, equals the left-nested parenthesised
    // form (binary nodes) under several assignments
    let vars = ["a", "b", "c"];
    let values = [[3.0, -2.5, 7.0], [0.0, 1.0, -1.0], [f64::NAN, 1e300, 0.5]];
    for ops in [
        &["*", "/", "%"][..],
        &["+", "-"],
        &["<<", ">>", ">>>"],
        &["<", "<=", ">", ">="],
        &["==", "!="],
        &["&"],
        &["^"],
        &["|"],
        &["&&"],
        &["||"],
    ] {
        for n in [CHAIN, CHAIN + 1, 40] {
            let mut flat = String::from("a");
            let mut nested = String::from("a");
            for i in 0..n {
                let (op, v) = (ops[i % ops.len()], vars[(i + 1) % vars.len()]);
                flat = format!("{flat} {op} {v}");
                nested = format!("({nested} {op} {v})");
            }
            for vals in values {
                for wrap in ["{}", "-({})", "!({})", "~({})", "({}) ? 2 : 3"] {
                    let (f, g) = (wrap.replace("{}", &flat), wrap.replace("{}", &nested));
                    assert_eq!(
                        run(&f, &vals).to_bits(),
                        run(&g, &vals).to_bits(),
                        "{f} at {vals:?}"
                    );
                }
            }
        }
    }
    // a chain is one level, however long, so only real nesting is bounded
    let sum = format!("a{}", " + a".repeat(100_000));
    assert_eq!(run(&sum, &[1.0]), 100_001.0);
    let nested = format!("{}{}{}", "(".repeat(100), sum, ")".repeat(100));
    assert_eq!(run(&nested, &[0.5]), 50_000.5);
    // `&&` and `||` chains stop at their first deciding operand
    let mut counting: Vec<BoxDoubleValues<'_>> = vec![Box::new(Failing)];
    let and = compile(&format!("0{} && x", " && 1".repeat(CHAIN))).unwrap();
    assert_eq!(and.evaluate(&mut counting).unwrap(), 0.0);
    let or = compile(&format!("1{} || x", " || 0".repeat(CHAIN))).unwrap();
    assert_eq!(or.evaluate(&mut counting).unwrap(), 1.0);
}

#[test]
fn java_math_edge_cases() {
    assert!(run("pow(1, 1 / 0)", &[]).is_nan());
    assert!(run("max(0 / 0, 1)", &[]).is_nan());
    assert_eq!(run("max(-0, 0)", &[]).to_bits(), 0.0f64.to_bits());
    assert_eq!(run("min(0, -0)", &[]).to_bits(), (-0.0f64).to_bits());
}

#[test]
fn normalize_quotes_unifies_quoting() {
    assert_eq!(normalize_quotes("a[\"b\"]"), "a['b']");
    assert_eq!(normalize_quotes("a['it\\'s']"), "a['it\\'s']");
    assert_eq!(normalize_quotes("a[\"it's\"]"), "a['it\\'s']");
}
