//! What the corpus tests share: reading `corpus.bin` and comparing values.

#![allow(dead_code)]

use lucene_painless::value::Value;
use lucene_painless::vm::Throw;

pub struct Rd<'a> {
    pub b: &'a [u8],
    pub p: usize,
}

impl<'a> Rd<'a> {
    pub fn u8(&mut self) -> u8 {
        let v = self.b[self.p];
        self.p += 1;
        v
    }
    pub fn i32(&mut self) -> i32 {
        let v = i32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    pub fn i64(&mut self) -> i64 {
        let lo = self.i32() as u32 as u64;
        let hi = self.i32() as u32 as u64;
        (lo | (hi << 32)) as i64
    }
    pub fn bytes(&mut self) -> &'a [u8] {
        let n = self.i32() as usize;
        let v = &self.b[self.p..self.p + n];
        self.p += n;
        v
    }
    pub fn str(&mut self) -> String {
        String::from_utf8(self.bytes().to_vec()).unwrap()
    }
}

/// An expected value: the corpus encoding. The class names only show in
/// the verbose mismatch output.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Expected {
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

pub fn read_value(r: &mut Rd) -> Expected {
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

pub fn matches(e: &Expected, v: &Value) -> bool {
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

pub fn show(r: &Result<Value, Throw>) -> String {
    match r {
        Ok(v) => format!("{} {}", v.class_name(), v.java_to_string()),
        Err(t) => format!("{} {:?}", t.class(), t.message()),
    }
}

/// Equal results; identity hashes (`[I@2b80`) may differ.
pub fn same(a: &Result<Value, Throw>, b: &Result<Value, Throw>) -> bool {
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
