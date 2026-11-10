//! Differential test against Painless itself: every script of Painless's
//! own test suite (`tests/data/corpus.bin`, written by the plugin's
//! `tools/painless-corpus`), run natively and compared with what Painless
//! returned or threw.
//!
//! `PAINLESS_CORPUS_VERBOSE=1` prints each mismatch.

use std::collections::BTreeMap;
use std::rc::Rc;

use lucene_painless::compile::compile;
use lucene_painless::ir;
use lucene_painless::jmap::{JMap, Order};
use lucene_painless::specialize::{specialize, Facts};
use lucene_painless::typed::{lower, BatchEnv, BatchRegs};
use lucene_painless::value::{MapKind, Value};
use lucene_painless::vm::{ParamsEnv, Throw, Vm};

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn u8(&mut self) -> u8 {
        let v = self.b[self.p];
        self.p += 1;
        v
    }
    fn i32(&mut self) -> i32 {
        let v = i32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn i64(&mut self) -> i64 {
        let lo = self.i32() as u32 as u64;
        let hi = self.i32() as u32 as u64;
        (lo | (hi << 32)) as i64
    }
    fn bytes(&mut self) -> &'a [u8] {
        let n = self.i32() as usize;
        let v = &self.b[self.p..self.p + n];
        self.p += n;
        v
    }
    fn str(&mut self) -> String {
        String::from_utf8(self.bytes().to_vec()).unwrap()
    }
}

/// An expected value: the corpus encoding. The class names only show in
/// the verbose mismatch output.
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum Expected {
    Null,
    Bool(bool),
    Byte(i32),
    Short(i32),
    Char(i32),
    Int(i32),
    Long(i64),
    Float(u32),
    Double(u64),
    Str(String),
    List(String, Vec<Expected>),
    Map(String, Vec<(Expected, Expected)>),
    Array(String, Vec<Expected>),
    Other(String, String),
    Collection(String, Vec<Expected>),
}

fn read_value(r: &mut Rd) -> Expected {
    match r.u8() {
        0 => Expected::Null,
        1 => Expected::Bool(r.u8() == 1),
        2 => Expected::Byte(r.i32()),
        3 => Expected::Short(r.i32()),
        4 => Expected::Char(r.i32()),
        5 => Expected::Int(r.i32()),
        6 => Expected::Long(r.i64()),
        7 => Expected::Float(r.i32() as u32),
        8 => Expected::Double(r.i64() as u64),
        9 => Expected::Str(r.str()),
        11 => {
            let c = r.str();
            let n = r.i32();
            Expected::List(c, (0..n).map(|_| read_value(r)).collect())
        }
        12 => {
            let c = r.str();
            let n = r.i32();
            Expected::Map(c, (0..n).map(|_| (read_value(r), read_value(r))).collect())
        }
        13 => {
            let c = r.str();
            let n = r.i32();
            Expected::Array(c, (0..n).map(|_| read_value(r)).collect())
        }
        14 => Expected::Other(r.str(), r.str()),
        15 => {
            let c = r.str();
            let n = r.i32();
            Expected::Collection(c, (0..n).map(|_| read_value(r)).collect())
        }
        t => panic!("value tag {t}"),
    }
}

fn matches(e: &Expected, v: &Value) -> bool {
    match (e, v) {
        (Expected::Null, Value::Null) => true,
        (Expected::Bool(a), Value::Bool(b)) => a == b,
        (Expected::Byte(a), Value::Byte(b)) => *a == i32::from(*b),
        (Expected::Short(a), Value::Short(b)) => *a == i32::from(*b),
        (Expected::Char(a), Value::Char(b)) => *a == i32::from(*b),
        (Expected::Int(a), Value::Int(b)) => a == b,
        (Expected::Long(a), Value::Long(b)) => a == b,
        (Expected::Float(a), Value::Float(b)) => {
            *a == b.to_bits() || (f32::from_bits(*a).is_nan() && b.is_nan())
        }
        (Expected::Double(a), Value::Double(b)) => {
            *a == b.to_bits() || (f64::from_bits(*a).is_nan() && b.is_nan())
        }
        (Expected::Str(a), Value::Str(b)) => **a == **b,
        (Expected::List(_, items), Value::List(l)) => {
            let l = l.borrow();
            items.len() == l.items.len() && items.iter().zip(&l.items).all(|(e, v)| matches(e, v))
        }
        (Expected::Map(_, entries), Value::Map(m)) => {
            let m = m.borrow();
            entries.len() == m.map.len()
                && entries
                    .iter()
                    .zip(m.map.iter())
                    .all(|((ek, ev), (k, v))| matches(ek, k) && matches(ev, v))
        }
        (Expected::Collection(_, items), Value::Set(s)) => {
            let s = s.borrow();
            items.len() == s.map.len()
                && items
                    .iter()
                    .zip(s.map.keys().iter())
                    .all(|(e, v)| matches(e, v))
        }
        (Expected::Collection(_, items), Value::List(l)) => {
            let l = l.borrow();
            items.len() == l.items.len() && items.iter().zip(&l.items).all(|(e, v)| matches(e, v))
        }
        (Expected::Array(c, items), Value::Array(a)) => {
            let a = a.borrow();
            *c == v.class_name()
                && items.len() == a.items.len()
                && items.iter().zip(&a.items).all(|(e, v)| matches(e, v))
        }
        (Expected::Other(c, s), v) => *c == v.class_name() && *s == v.java_to_string(),
        _ => false,
    }
}

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

fn show(r: &Result<Value, Throw>) -> String {
    match r {
        Ok(v) => format!("{} {}", v.class_name(), v.java_to_string()),
        Err(t) => format!("{} {:?}", t.class(), t.message()),
    }
}

/// Equal results; identity hashes (`[I@2b80`) may differ.
fn same(a: &Result<Value, Throw>, b: &Result<Value, Throw>) -> bool {
    let strip = |s: String| {
        let mut out = String::new();
        let mut skip = false;
        for c in s.chars() {
            if c == '@' {
                skip = true;
            } else if skip && c.is_ascii_hexdigit() {
                continue;
            } else {
                skip = false;
            }
            out.push(c);
        }
        out
    };
    strip(show(a)) == strip(show(b))
}

fn run_corpus() {
    let data = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/corpus.bin"
    ))
    .unwrap();
    assert_eq!(&data[..4], b"PCO1");
    let verbose = std::env::var("PAINLESS_CORPUS_VERBOSE").is_ok();
    let filter = std::env::var("PAINLESS_CORPUS_FILTER").ok();
    let mut r = Rd { b: &data, p: 4 };
    let (mut total, mut pass) = (0, 0);
    let (mut lowered, mut batched, mut tier_mismatches) = (0, 0, 0);
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    while r.p < data.len() {
        let src = r.str();
        let status = r.u8();
        let expected = match status {
            0 => Ok(read_value(&mut r)),
            1 => Err((r.str(), r.str())),
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
        let got = {
            let mut vm = Vm::new(prog, &mut env);
            vm.execute(&[])
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
                    .map(|()| Value::Double(out[0]));
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
            (Err((class, _)), Err(t)) => *class == t.class(),
            _ => false,
        };
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
            (Err((c, _)), Ok(_)) => format!("returned, expected {c}"),
            (Err((c, _)), Err(t)) => format!("threw {} expected {c}", t.class()),
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
    println!("corpus: {pass}/{total} match Painless; {lowered} run typed, {batched} in batches");
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

/// Scripts that must match; raised as the runtime grows.
const MIN_PASS: usize = 4910;

#[test]
fn corpus_matches_painless() {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(run_corpus)
        .unwrap()
        .join()
        .unwrap();
}
