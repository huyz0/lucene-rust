//! The front end's site table against Painless's own: for each corpus
//! script the front end lowers, the members it resolves by name, and
//! Painless's answer for each native class, are the ones the Java front
//! end wrote.
//!
//! `PAINLESS_FRONT_VERBOSE=1` prints each script whose table differs.

use lucene_painless::check::{check, Served};
use lucene_painless::compile::compile;
use lucene_painless::front::{analyze, parse, Limits};
use lucene_painless::ir::{decode, Script, Site};
use lucene_painless::jmap::{JMap, Order};
use lucene_painless::specialize::{specialize, Facts};
use lucene_painless::value::{MapKind, Value};

mod common;
use common::Rd;

/// The scripts whose site tables matched when this floor was last raised;
/// it only goes up.
const MIN_EQUAL: usize = 4963;

/// Whether the check passes `script` as a script with no fields.
fn verdict(script: &Script) -> Option<bool> {
    let mut prog = compile(script).ok()?;
    let params = Value::map(MapKind::HashMap, JMap::new(Order::Hash));
    specialize(
        &mut prog,
        &Facts {
            params: Some(&params),
            fields: &[],
        },
    );
    Some(
        check(
            script,
            &prog,
            &Served {
                fields: &[],
                unmapped: &[],
                internals: &["getParams"],
            },
        )
        .is_ok(),
    )
}

fn key(s: &Site) -> String {
    let bits = |v: &[bool]| {
        v.iter()
            .map(|b| if *b { '1' } else { '0' })
            .collect::<String>()
    };
    format!(
        "{:?} {} /{} on {} admits {} resolves {}",
        s.kind,
        s.name,
        s.arity,
        s.receiver,
        bits(&s.admits),
        bits(&s.resolves)
    )
}

#[test]
fn site_tables_equal_painless() {
    let data = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/corpus.bin"
    ))
    .unwrap();
    let verbose = std::env::var_os("PAINLESS_FRONT_VERBOSE").is_some();
    let mut r = Rd { b: &data, p: 4 };
    let (mut equal, mut differ, mut compared, mut verdicts_differ) =
        (0usize, 0usize, 0usize, 0usize);
    while r.p < data.len() {
        let src = r.str();
        let ir_blob = match r.u8() {
            0 => {
                skip_value(&mut r);
                Some(r.bytes().to_vec())
            }
            1 => {
                r.str();
                if r.u8() == 1 {
                    r.bytes();
                }
                r.i32();
                Some(r.bytes().to_vec())
            }
            2 => None,
            3 => {
                r.bytes();
                None
            }
            s => panic!("status {s}"),
        };
        let Some(blob) = ir_blob else { continue };
        let Ok(java) = decode(&blob) else { continue };
        let Ok(ast) = parse(&src, &Limits::default()) else {
            continue;
        };
        let Ok(rust) = analyze(&ast) else { continue };
        compared += 1;
        // A script the runtime cannot compile has no verdict.
        if let (Some(j), Some(r)) = (verdict(&java), verdict(&rust)) {
            if j != r {
                verdicts_differ += 1;
                if verbose {
                    println!("--- check verdict differs: {src}");
                }
            }
        }
        let mut a: Vec<String> = java.sites.iter().map(key).collect();
        let mut b: Vec<String> = rust.sites.iter().map(key).collect();
        a.sort();
        b.sort();
        // A reference cast Painless adds between two static types the front
        // end leaves out has no member to resolve.
        a.retain(|x| !x.starts_with("InstanceOf") || b.contains(x));
        if a == b {
            equal += 1;
        } else {
            differ += 1;
            if verbose {
                let only_java: Vec<&String> = a.iter().filter(|x| !b.contains(x)).collect();
                let only_rust: Vec<&String> = b.iter().filter(|x| !a.contains(x)).collect();
                println!("--- {src}\n  java only: {only_java:?}\n  rust only: {only_rust:?}");
            }
        }
    }
    println!("site tables: {equal} of {compared} equal, {differ} differ");
    println!("check verdicts that differ from the Java table's: {verdicts_differ}");
    assert!(equal >= MIN_EQUAL);
    assert_eq!(
        verdicts_differ, 0,
        "the check judges a lowered script as it judges Painless's IR"
    );
}

fn skip_value(r: &mut Rd) {
    common::read_value(r);
}
