//! Differential test against Painless itself: every script of Painless's
//! own test suite (`tests/data/corpus.bin`, written by the plugin's
//! `tools/painless-corpus`), run natively and compared with what Painless
//! returned or threw.
//!
//! `PAINLESS_CORPUS_VERBOSE=1` prints each mismatch.

use std::collections::BTreeMap;
use std::rc::Rc;

use lucene_painless::check::{check, Served};
use lucene_painless::compile::compile;
use lucene_painless::ir;
use lucene_painless::jmap::{JMap, Order};
use lucene_painless::specialize::{specialize, Facts};
use lucene_painless::typed::{lower, BatchEnv, BatchRegs};
use lucene_painless::value::{MapKind, Value};
use lucene_painless::vm::{ParamsEnv, Throw, Vm};

mod common;
use common::{matches, read_value, same, show, Rd};

/// A batch with no fields and no score, as the corpus's scripts read.
struct NoDocs;

impl BatchEnv for NoDocs {
    fn column(&mut self, field: usize) -> Result<(&[u64], &[u64]), Throw> {
        unreachable!("no fields: {field}")
    }
    fn scores(&mut self, _out: &mut [u64]) -> Result<(), Throw> {
        unreachable!("no score")
    }
}

fn run_corpus() {
    let data = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/corpus.bin"
    ))
    .unwrap();
    assert_eq!(&data[..4], b"PCO2");
    let verbose = std::env::var("PAINLESS_CORPUS_VERBOSE").is_ok();
    let filter = std::env::var("PAINLESS_CORPUS_FILTER").ok();
    let mut r = Rd { b: &data, p: 4 };
    let (mut total, mut pass) = (0, 0);
    let (mut lowered, mut batched, mut tier_mismatches) = (0, 0, 0);
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    let mut refusals: BTreeMap<String, usize> = BTreeMap::new();
    let (mut checked, mut unsound) = (0, 0);
    while r.p < data.len() {
        let src = r.str();
        let status = r.u8();
        let expected = match status {
            0 => Ok(read_value(&mut r)),
            1 => {
                let class = r.str();
                let message = (r.u8() == 1).then(|| r.str());
                Err((class, message, r.i32()))
            }
            3 => {
                r.str();
                continue;
            }
            _ => continue,
        };
        let blob = r.bytes();
        if filter.as_deref().is_some_and(|f| !src.contains(f)) {
            continue;
        }
        total += 1;
        let script = match ir::decode(blob) {
            Ok(s) => s,
            Err(e) => {
                if verbose {
                    println!("--- decode: {e}\n{src}");
                }
                *reasons.entry(format!("decode: {e}")).or_default() += 1;
                continue;
            }
        };
        if std::env::var("PAINLESS_CORPUS_DUMP").is_ok() {
            println!("{:#?}", script.functions);
            if let Ok(p) = compile(&script) {
                for f in &p.funcs {
                    println!("{:#?}", f.body);
                }
            }
        }
        let prog = match compile(&script) {
            Ok(p) => Rc::new(p),
            Err(e) => {
                let key = format!("compile: {e}");
                let key: String = key.chars().take(90).collect();
                if verbose {
                    println!("--- {key}\n{src}");
                }
                *reasons.entry(key).or_default() += 1;
                continue;
            }
        };
        let mut env = ParamsEnv {
            params: Value::map(MapKind::HashMap, JMap::new(Order::Hash)),
        };
        let (got, at) = {
            let mut vm = Vm::new(prog, &mut env);
            let r = vm.execute(&[]);
            (r, vm.loc)
        };
        // The specialised program, and its typed form where it has one,
        // must do exactly what the program did.
        let mut special = compile(&script).unwrap();
        let params = Value::map(MapKind::HashMap, JMap::new(Order::Hash));
        specialize(
            &mut special,
            &Facts {
                params: Some(&params),
                fields: &[],
            },
        );
        let verdict = check(
            &script,
            &special,
            &Served {
                fields: &[],
                unmapped: &[],
                internals: &["getParams"],
            },
        );
        let typed = lower(&special, &[]);
        let mut env = ParamsEnv { params };
        let mut runs = vec![("specialised", {
            let mut vm = Vm::new(Rc::new(special), &mut env);
            vm.execute(&[])
        })];
        if let Some(t) = typed {
            lowered += 1;
            let mut regs = t.regs();
            runs.push(("typed", t.run(&mut env, &mut regs)));
            if t.batchable() {
                batched += 1;
                let mut out = [0.0; 3];
                let r = t
                    .run_batch(&mut NoDocs, &mut BatchRegs::default(), &mut out)
                    .and_then(|failed| match failed {
                        None if t.returns_bool() => Ok(Value::Bool(out[0] != 0.0)),
                        None => Ok(Value::Double(out[0])),
                        Some((_, t, _)) => Err(t),
                    });
                assert!(
                    out.iter().all(|v| v.to_bits() == out[0].to_bits()),
                    "{src}: lanes differ"
                );
                runs.push(("batch", r));
            }
        }
        for (tier, r) in runs {
            if !same(&got, &r) {
                println!(
                    "--- {tier} differs\n{src}\n  vm {}\n  {tier} {}",
                    show(&got),
                    show(&r)
                );
                tier_mismatches += 1;
            }
        }
        let ok = match (&expected, &got) {
            (Ok(e), Ok(v)) => matches(e, v),
            (Err((class, _, _)), Err(t)) => *class == t.class(),
            _ => false,
        };
        // What the check passes runs exactly as Painless does: the value,
        // or the exception with its message, at the same offset.
        match &verdict {
            Ok(()) => {
                checked += 1;
                let exact = match (&expected, &got) {
                    (Err((class, message, offset)), Err(t)) => {
                        *class == t.class() && *message == t.message() && *offset == at
                    }
                    _ => ok,
                };
                let known = KNOWN_GAPS.contains(&src.as_str())
                    && matches!((&expected, &got), (Err((class, ..)), Err(t)) if *class == t.class());
                if !exact && !known {
                    println!(
                        "--- passed the check, differs\n{src}\n  expected {expected:?}\n  got {} at {at}",
                        show(&got)
                    );
                    unsound += 1;
                }
            }
            Err(why) => {
                if verbose {
                    println!("--- check refused: {why}\n{src}");
                }
                let key: String = format!("check: {why}").chars().take(90).collect();
                *refusals.entry(key).or_default() += 1;
            }
        }
        if ok {
            pass += 1;
            continue;
        }
        let key = match (&expected, &got) {
            (_, Err(t)) if t.class() == lucene_painless::api::UNSUPPORTED => {
                format!("runtime unsupported: {}", t.message().unwrap_or_default())
            }
            (Ok(_), Ok(_)) => "wrong value".to_string(),
            (Ok(_), Err(t)) => format!("threw {}", t.class()),
            (Err((c, ..)), Ok(_)) => format!("returned, expected {c}"),
            (Err((c, ..)), Err(t)) => format!("threw {} expected {c}", t.class()),
        };
        let key: String = key.chars().take(90).collect();
        if verbose {
            let g = match &got {
                Ok(v) => format!("{} {}", v.class_name(), v.java_to_string()),
                Err(t) => format!("{} {:?}", t.class(), t.message()),
            };
            println!("--- {key}\n{src}\n  expected {expected:?}\n  got {g}");
        }
        *reasons.entry(key).or_default() += 1;
    }
    let mut rs: Vec<_> = reasons.into_iter().collect();
    rs.sort_by_key(|r| std::cmp::Reverse(r.1));
    for (k, n) in rs.iter().take(60) {
        println!("{n:5}  {k}");
    }
    let mut rs: Vec<_> = refusals.into_iter().collect();
    rs.sort_by_key(|r| std::cmp::Reverse(r.1));
    for (k, n) in rs.iter().take(40) {
        println!("{n:5}  {k}");
    }
    println!("corpus: {pass}/{total} match Painless; {lowered} run typed, {batched} in batches");
    println!("check: {checked}/{total} pass, {unsound} of them differ from Painless");
    assert_eq!(unsound, 0, "scripts the check passes differ from Painless");
    assert!(
        checked >= MIN_CHECKED,
        "{checked} scripts pass the check, expected at least {MIN_CHECKED}"
    );
    assert_eq!(
        tier_mismatches, 0,
        "specialised or typed runs differ from the VM"
    );
    let floor: usize = std::env::var("PAINLESS_CORPUS_FLOOR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MIN_PASS);
    assert!(
        pass >= floor,
        "{pass} scripts match Painless, expected at least {floor}"
    );
}

/// Scripts the check passes whose exception differs from Painless's in its
/// message and position alone: a `WrongMethodTypeException` for a string
/// index into a list or array, whose text is the JVM's method handle types
/// and whose line is the call site's bootstrap.
const KNOWN_GAPS: &[&str] = &[
    "def x = new ArrayList(); return x.get('bogus');",
    "def x = new long[1]; x[0]=1; return x['bogus'];",
    "def x = new ArrayList(); x.add('foo'); return x['bogus'];",
];

/// Scripts that must match; raised as the runtime grows.
const MIN_PASS: usize = 4920;

/// Scripts the check must pass; raised as the runtime grows.
const MIN_CHECKED: usize = 4905;

#[test]
fn corpus_matches_painless() {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(run_corpus)
        .unwrap()
        .join()
        .unwrap();
}
