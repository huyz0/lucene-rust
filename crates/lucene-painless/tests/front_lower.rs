//! The front end's analysis against Painless's own test suite
//! (`tests/data/corpus.bin`): each script it lowers runs natively and must
//! do what Painless did (the value, or the exception's class), and the
//! rest are counted by why they were refused.
//!
//! `PAINLESS_FRONT_VERBOSE=1` prints each refusal and mismatch.

use std::collections::BTreeMap;
use std::rc::Rc;

use lucene_painless::compile::compile;
use lucene_painless::front::{analyze, parse, Limits, Refusal};
use lucene_painless::jmap::{JMap, Order};
use lucene_painless::value::{MapKind, Value};
use lucene_painless::vm::{ParamsEnv, Throw, Vm};

mod common;
use common::{matches, read_value, show, Expected, Rd};

/// The scripts that lowered and matched when this floor was last raised;
/// it only goes up.
const MIN_LOWERED: usize = 4880;

type Outcome = Result<Expected, String>;

/// Every script Painless compiled, with what it did.
fn scripts() -> Vec<(String, Outcome)> {
    let data = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/corpus.bin"
    ))
    .unwrap();
    let mut r = Rd { b: &data, p: 4 };
    let mut out = Vec::new();
    while r.p < data.len() {
        let src = r.str();
        match r.u8() {
            0 => {
                let v = read_value(&mut r);
                r.bytes();
                out.push((src, Ok(v)));
            }
            1 => {
                let class = r.str();
                if r.u8() == 1 {
                    r.bytes();
                }
                r.i32();
                r.bytes();
                out.push((src, Err(class)));
            }
            2 => {}
            3 => {
                r.bytes();
            }
            s => panic!("status {s}"),
        }
    }
    out
}

fn run(src: &str, expected: &Outcome) -> Result<Result<(), String>, Refusal> {
    let ast = parse(src, &Limits::default()).expect("every corpus script parses");
    let script = analyze(&ast)?;
    let prog = Rc::new(compile(&script).map_err(|e| Refusal::Unsupported(e.to_string()))?);
    let mut env = ParamsEnv {
        params: Value::map(MapKind::HashMap, JMap::new(Order::Hash)),
    };
    let got: Result<Value, Throw> = Vm::new(prog, &mut env).execute(&[]);
    // A method the runtime has no code for yet is a script it cannot run,
    // counted as refused rather than as a script that did the wrong thing.
    if let Err(t) = &got {
        if t.class() == lucene_painless::api::UNSUPPORTED {
            return Err(Refusal::Unsupported(show(&got)));
        }
    }
    let ok = match (expected, &got) {
        (Ok(e), Ok(v)) => matches(e, v),
        (Err(class), Err(t)) => *class == t.class(),
        _ => false,
    };
    Ok(if ok {
        Ok(())
    } else {
        Err(format!("expected {expected:?}, got {}", show(&got)))
    })
}

#[test]
fn lowered_scripts_do_what_painless_did() {
    let verbose = std::env::var("PAINLESS_FRONT_VERBOSE").is_ok();
    let (mut total, mut lowered, mut mismatched, mut rejected) = (0, 0, 0, 0);
    let mut refusals: BTreeMap<String, usize> = BTreeMap::new();
    for (src, expected) in scripts() {
        total += 1;
        match run(&src, &expected) {
            Ok(Ok(())) => lowered += 1,
            Ok(Err(why)) => {
                mismatched += 1;
                println!("--- differs\n{src}\n  {why}");
            }
            Err(Refusal::Compile { offset, message }) => {
                rejected += 1;
                if verbose {
                    println!("--- rejected: {message} at {offset}\n{src}");
                }
            }
            Err(Refusal::Unsupported(what)) => {
                if verbose {
                    println!("--- unsupported: {what}\n{src}");
                }
                let key: String = what.chars().take(60).collect();
                *refusals.entry(key).or_default() += 1;
            }
        }
    }
    let mut rs: Vec<_> = refusals.into_iter().collect();
    rs.sort_by_key(|r| std::cmp::Reverse(r.1));
    for (k, n) in rs.iter().take(40) {
        println!("{n:5}  {k}");
    }
    println!(
        "front end: {lowered}/{total} lower and match Painless; {mismatched} differ, \
         {rejected} wrongly rejected"
    );
    assert!(
        lowered >= MIN_LOWERED,
        "{lowered} scripts lower and match Painless, expected at least {MIN_LOWERED}"
    );
    assert_eq!(mismatched, 0, "lowered scripts differ from Painless");
    assert_eq!(rejected, 0, "scripts Painless compiled were rejected");
}
