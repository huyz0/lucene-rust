//! Source ingest: one `_source` JSON object turned into the field entries
//! the plugin's `DocumentEncoder` writes for the fields of a mapping, without
//! the JVM's document mapper.
//!
//! A [`Plan`] is a mapping's fields, compiled by the plugin only for mappings
//! whose every field this module reproduces exactly (see [`register`]). [`parse`]
//! then reads a source against it and either writes the fragment (the stored,
//! inverted, doc-values and point entries in the layout `writer_blob` decodes)
//! or refuses, and a refusal always means "let the JVM's mapper handle this
//! document": an unmapped field, a value the fast path does not reproduce
//! digit for digit, a repeated field, an array of arrays, a number or boolean for
//! a keyword, anything that is not plain JSON. Plain objects (the mapping's
//! `object` fields, at any depth, as `a.b` paths), arrays of scalars or of
//! objects, and `null` (no value, as the mapper reads it) are read here. The
//! scanner is strict RFC 8259; whatever the JVM's parser would accept beyond
//! that is refused here.
//!
//! A nested object (`type: nested`) is a document of its own: each object of its value is
//! read into a child document, children before their parents as the mapper orders them.
//! `geo_point` (an object with `lat` and `lon`, a `[lon, lat]` pair, a `"lat,lon"` string),
//! `boolean`, multi-fields of text and keyword fields, and `copy_to` between text and keyword
//! fields of one document are read here too.
//!
//! What [`parse`] writes is `i32` child count, then per child `i32` plan index of its nested
//! object, `i32` length, and the child's fragment; then the root's fragment. A fragment is
//! `4 x i32` entry counts (stored, inverted, doc values, points), `4 x i32` byte lengths, then
//! the four sections.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, LazyLock, RwLock};

use crate::status::{set_last_error, Status};

/// Field kinds of a plan, as the plan blob carries them.
pub const KEYWORD: u8 = 0;
pub const TEXT: u8 = 1;
pub const INTEGER: u8 = 2;
pub const LONG: u8 = 3;
pub const FLOAT: u8 = 4;
pub const DOUBLE: u8 = 5;
pub const DATE: u8 = 6;
/// An `object` field: a path whose children are looked up as `path.child`.
pub const OBJECT: u8 = 7;
pub const GEO_POINT: u8 = 8;
pub const BOOLEAN: u8 = 9;
/// A `nested` field: like [`OBJECT`], and every object of its value is a document.
pub const NESTED: u8 = 10;

/// Plan field flags.
const INDEXED: u8 = 1;
const DOC_VALUES: u8 = 2;

/// `IndexWriter.MAX_TERM_LENGTH`.
const MAX_TERM_LENGTH: usize = 32766;
/// Fields a plan holds (the duplicate check is a 256-bit mask).
const MAX_FIELDS: usize = 256;

struct Field {
    /// Its index in the plan.
    idx: u16,
    /// Where the last `.` of its name is; [`NO_DOT`] for a name without one (a field of the root).
    last_dot: u32,
    name: Box<[u8]>,
    kind: u8,
    flags: u8,
    number: i32,
    /// Keyword only: `ignore_above`, in UTF-16 units.
    ignore_above: u32,
    /// Text only: the analysis registration and its entry's index options and norm mode.
    analyzer: i32,
    options: u8,
    norm: u8,
    /// The field this one is a multi-field of; such a field has no key of its own.
    parent: Option<u16>,
    /// Multi-fields of this field, and the fields it copies to.
    subs: Vec<u16>,
    copy_to: Vec<u16>,
    /// Values of this field meet other fields' (a `copy_to` target, or a field with multi-fields
    /// or targets): they are collected and written as one entry per field at the end of the document.
    merge: bool,
}

const NO_DOT: u32 = u32::MAX;

/// A mapping's fields, compiled.
pub struct Plan {
    fields: Vec<Field>,
    /// Open-addressed table of `field index + 1`, 0 for empty.
    slots: Vec<u16>,
    /// Per place in a document -- the start of an object, or a key already read -- the field
    /// that followed it last time, as `field index + 1` (0 for none): documents of one shape
    /// repeat their key order, so a key is checked against that field before it is hashed.
    /// Only a guess, always confirmed against the key's bytes.
    next: Vec<AtomicU16>,
    /// `index.mapping.nested_objects.limit`: nested documents one document may have.
    nested_limit: u32,
    /// The metadata fields' numbers, when the plan was registered with them ([`prepare`] needs them).
    meta: Option<prepared::Meta>,
}

/// A set of plan field indices.
#[derive(Clone, Copy, Default)]
struct Mask([u64; MAX_FIELDS / 64]);

impl Mask {
    /// Sets `i`; whether it was already set.
    #[inline]
    fn test_and_set(&mut self, i: usize) -> bool {
        let (w, b) = (i / 64, 1u64 << (i % 64));
        let was = self.0[w] & b != 0;
        self.0[w] |= b;
        was
    }
}

fn invalid(what: &str) -> Status {
    set_last_error(format!("ingest plan: {what}"));
    Status::InvalidArgument
}

fn hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    h ^ (h >> 29)
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Status> {
        let end = self.pos.checked_add(n).ok_or_else(|| invalid("length"))?;
        let s = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| invalid("truncated"))?;
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, Status> {
        Ok(self.take(1)?[0])
    }

    fn i32(&mut self) -> Result<i32, Status> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

impl Plan {
    /// `i32 count`, then per field `i32 len, name, u8 kind, u8 flags, i32 field
    /// number, i32 ignore_above, i32 analyzer, u8 index options, u8 norm mode, i32
    /// parent (the field it is a multi-field of, or -1), i32 n, n x i32 copy_to
    /// targets`, then `i32` nested document limit, and optionally the metadata: `6 x i32` field
    /// numbers (`_id`, `_seq_no`, `_primary_term`, `_version`, `_source`, `_nested_path`), `u8` whether
    /// `_seq_no` is a point too, `i32 n`, and `n x (i32 len, bytes)` the `_nested_path` term of each
    /// nested object.
    fn decode(blob: &[u8]) -> Result<Plan, Status> {
        let mut r = Reader { buf: blob, pos: 0 };
        let n = usize::try_from(r.i32()?).map_err(|_| invalid("count"))?;
        if n == 0 || n > MAX_FIELDS {
            return Err(invalid("field count"));
        }
        let mut fields = Vec::with_capacity(n);
        for _ in 0..n {
            let len = usize::try_from(r.i32()?).map_err(|_| invalid("name length"))?;
            let name = r.take(len)?.to_vec().into_boxed_slice();
            let kind = r.u8()?;
            if kind > NESTED {
                return Err(invalid("kind"));
            }
            let flags = r.u8()?;
            let number = r.i32()?;
            let ignore_above = u32::try_from(r.i32()?).map_err(|_| invalid("ignore_above"))?;
            let analyzer = r.i32()?;
            let options = r.u8()?;
            let norm = r.u8()?;
            let parent = match r.i32()? {
                -1 => None,
                p => Some(u16::try_from(p).map_err(|_| invalid("parent"))?),
            };
            let copies = usize::try_from(r.i32()?).map_err(|_| invalid("copy_to count"))?;
            let mut copy_to = Vec::with_capacity(copies.min(n));
            for _ in 0..copies {
                copy_to.push(u16::try_from(r.i32()?).map_err(|_| invalid("copy_to"))?);
            }
            fields.push(Field {
                idx: u16::try_from(fields.len()).map_err(|_| invalid("count"))?,
                last_dot: name
                    .iter()
                    .rposition(|&b| b == b'.')
                    .map_or(NO_DOT, |p| p as u32),
                name,
                kind,
                flags,
                number,
                ignore_above,
                analyzer,
                options,
                norm,
                parent,
                subs: Vec::new(),
                copy_to,
                merge: false,
            });
        }
        let nested_limit = u32::try_from(r.i32()?).map_err(|_| invalid("nested limit"))?;
        let meta = if r.pos == blob.len() {
            None
        } else {
            let (id, seq_no, primary_term) = (r.i32()?, r.i32()?, r.i32()?);
            let (version, source, nested_path) = (r.i32()?, r.i32()?, r.i32()?);
            let seq_point = r.u8()? != 0;
            let count = usize::try_from(r.i32()?).map_err(|_| invalid("nested path count"))?;
            let mut nested_paths = Vec::with_capacity(count.min(MAX_FIELDS));
            for _ in 0..count {
                let len = usize::try_from(r.i32()?).map_err(|_| invalid("nested path"))?;
                nested_paths.push(r.take(len)?.to_vec().into_boxed_slice());
            }
            Some(prepared::Meta {
                id,
                seq_no,
                primary_term,
                version,
                source,
                nested_path,
                seq_point,
                nested_paths,
            })
        };
        if r.pos != blob.len() {
            return Err(invalid("trailing bytes"));
        }
        for i in 0..n {
            if let Some(p) = fields[i].parent {
                let p = usize::from(p);
                if p >= n || !matches!(fields[p].kind, TEXT | KEYWORD) || fields[p].parent.is_some()
                {
                    return Err(invalid("parent"));
                }
                if !matches!(fields[i].kind, TEXT | KEYWORD) {
                    return Err(invalid("multi-field kind"));
                }
                fields[p].subs.push(i as u16);
                fields[p].merge = true;
            }
            for k in 0..fields[i].copy_to.len() {
                let t = usize::from(fields[i].copy_to[k]);
                if t >= n
                    || !matches!(fields[i].kind, TEXT | KEYWORD)
                    || !matches!(fields[t].kind, TEXT | KEYWORD)
                    || fields[t].parent.is_some()
                    || !fields[t].copy_to.is_empty()
                {
                    return Err(invalid("copy_to"));
                }
                fields[t].merge = true;
                fields[i].merge = true;
            }
        }
        let size = (n * 2).next_power_of_two().max(8);
        let mut slots = vec![0u16; size];
        for (i, f) in fields.iter().enumerate() {
            let mut s = (hash(&f.name) as usize) & (size - 1);
            while slots[s] != 0 {
                if fields[usize::from(slots[s]) - 1].name == f.name {
                    return Err(invalid("duplicate field"));
                }
                s = (s + 1) & (size - 1);
            }
            slots[s] = u16::try_from(i + 1).map_err(|_| invalid("count"))?;
        }
        let next = (0..=n * 2).map(|_| AtomicU16::new(0)).collect();
        Ok(Plan {
            fields,
            slots,
            next,
            nested_limit,
            meta,
        })
    }

    fn find(&self, name: &[u8]) -> Option<usize> {
        let mask = self.slots.len() - 1;
        let mut s = (hash(name) as usize) & mask;
        loop {
            let e = self.slots[s];
            if e == 0 {
                return None;
            }
            let i = usize::from(e) - 1;
            if *self.fields[i].name == *name {
                return Some(i);
            }
            s = (s + 1) & mask;
        }
    }
}

#[derive(Default)]
struct Registry {
    by_blob: HashMap<Vec<u8>, u32>,
    plans: Vec<Arc<Plan>>,
}

static REGISTRY: LazyLock<RwLock<Registry>> = LazyLock::new(Default::default);

/// Plans kept for the life of the process: one per distinct mapping seen.
const MAX_PLANS: usize = 1 << 14;

/// Registers a plan blob once, answering its id for [`parse`].
pub fn register(blob: &[u8]) -> Result<u32, Status> {
    if let Some(&id) = REGISTRY
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .by_blob
        .get(blob)
    {
        return Ok(id);
    }
    let plan = Arc::new(Plan::decode(blob)?);
    let mut r = REGISTRY.write().unwrap_or_else(|e| e.into_inner());
    if let Some(&id) = r.by_blob.get(blob) {
        return Ok(id);
    }
    if r.plans.len() >= MAX_PLANS {
        return Err(invalid("too many plans"));
    }
    let id = u32::try_from(r.plans.len()).map_err(|_| Status::InvalidArgument)?;
    r.plans.push(plan);
    r.by_blob.insert(blob.to_vec(), id);
    Ok(id)
}

/// The plan `id` names.
pub fn plan(id: u32) -> Option<Arc<Plan>> {
    REGISTRY
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .plans
        .get(usize::try_from(id).ok()?)
        .cloned()
}

/// One document being read: its entries in the four sections.
#[derive(Default)]
struct Doc {
    stored: Vec<u8>,
    inverted: Vec<u8>,
    doc_values: Vec<u8>,
    points: Vec<u8>,
    counts: [i32; 4],
    /// Arrays of objects being read (0 outside any): inside one a field can occur once per
    /// object, and its keyword and text values wait in `deferred` to become one entry.
    repeat: usize,
    /// `(field, from, to)` into `held`, in the order read.
    deferred: Vec<(u16, u32, u32)>,
    held: Vec<u8>,
}

impl Doc {
    fn clear(&mut self) {
        self.stored.clear();
        self.inverted.clear();
        self.doc_values.clear();
        self.points.clear();
        self.counts = [0; 4];
        self.repeat = 0;
        self.deferred.clear();
        self.held.clear();
    }
}

#[derive(Default)]
struct Sections {
    /// The document being read.
    d: Doc,
    /// The documents it is nested in, and emptied ones to reuse.
    parked: Vec<Doc>,
    free: Vec<Doc>,
    /// The finished child documents, as [`parse`] writes them, and how many there are.
    kids: Vec<u8>,
    nkids: u32,
    /// Child documents begun, finished or not.
    entered: u32,
    /// The values of the keyword array being read, and where each one is in `values`.
    values: Vec<u8>,
    spans: Vec<(u32, u32)>,
}

impl Sections {
    fn clear(&mut self) {
        self.d.clear();
        self.free.append(&mut self.parked);
        self.kids.clear();
        self.nkids = 0;
        self.entered = 0;
    }

    /// Starts a child document: what is read next goes there.
    fn enter(&mut self) {
        self.entered += 1;
        let mut fresh = self.free.pop().unwrap_or_default();
        fresh.clear();
        self.parked.push(std::mem::replace(&mut self.d, fresh));
    }

    /// Ends the child document begun by [`Self::enter`]: its fragment joins `kids` and reading
    /// goes on in its parent.
    fn leave(&mut self, plan: &Plan, object: usize) {
        flush_deferred(plan, self);
        put_i32(&mut self.kids, object as i32);
        let at = self.kids.len();
        put_i32(&mut self.kids, 0);
        fragment(&self.d, &mut self.kids);
        let len = (self.kids.len() - at - 4) as i32;
        self.kids[at..at + 4].copy_from_slice(&len.to_le_bytes());
        self.nkids += 1;
        if let Some(parent) = self.parked.pop() {
            self.free.push(std::mem::replace(&mut self.d, parent));
        }
    }
}

/// Bytes of a fragment's header: `4 x i32` counts and `4 x i32` section lengths.
pub(super) const FRAGMENT_HEADER: usize = 32;

/// A document's fragment: `4 x i32` counts, `4 x i32` section lengths, the sections.
fn fragment(d: &Doc, out: &mut Vec<u8>) {
    for c in d.counts {
        put_i32(out, c);
    }
    for len in [
        d.stored.len(),
        d.inverted.len(),
        d.doc_values.len(),
        d.points.len(),
    ] {
        put_i32(out, len as i32);
    }
    out.extend_from_slice(&d.stored);
    out.extend_from_slice(&d.inverted);
    out.extend_from_slice(&d.doc_values);
    out.extend_from_slice(&d.points);
}

/// Most values one keyword array may hold here (its terms are merged pairwise).
const MAX_ARRAY_TERMS: usize = 256;
/// Most keyword and text values the objects of one array may hold.
const MAX_DEFERRED: usize = 4096;
/// Deepest nesting of objects read here.
const MAX_DEPTH: usize = 32;

thread_local! {
    static SECTIONS: RefCell<Sections> = RefCell::new(Sections::default());
    static UNESCAPED: RefCell<(Vec<u8>, Vec<u8>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

fn put_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_i64(out: &mut Vec<u8>, v: i64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    put_i32(out, b.len() as i32);
    out.extend_from_slice(b);
}

const ONES: u64 = 0x0101_0101_0101_0101;
const HIGHS: u64 = 0x8080_8080_8080_8080;

/// The index of the first byte at or after `from` that ends a run of plain string bytes (`"`,
/// `\`, or a control byte), or `src.len()`; eight bytes at a time. The subtraction can flag
/// a byte above a true hit, never below one, so the lowest flag is exact.
#[inline]
fn plain_run(src: &[u8], mut from: usize) -> usize {
    while let Some(w) = src.get(from..from + 8) {
        let x = u64::from_le_bytes([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]);
        let zero = |v: u64| v.wrapping_sub(ONES) & !v;
        let hits = (zero(x ^ (ONES * u64::from(b'"')))
            | zero(x ^ (ONES * u64::from(b'\\')))
            | x.wrapping_sub(ONES * 0x20) & !x)
            & HIGHS;
        if hits != 0 {
            return from + (hits.trailing_zeros() / 8) as usize;
        }
        from += 8;
    }
    while let Some(&b) = src.get(from) {
        if b == b'"' || b == b'\\' || b < 0x20 {
            break;
        }
        from += 1;
    }
    from
}

/// Where a scanned string lives: in the source, or unescaped in the scratch.
enum Str {
    Source(usize, usize),
    Scratch,
}

struct Scanner<'s, 'u> {
    src: &'s [u8],
    pos: usize,
    scratch: &'u mut Vec<u8>,
    /// The dotted path of the object being read, up to its own length.
    path: &'u mut Vec<u8>,
}

impl<'s> Scanner<'s, '_> {
    #[inline]
    fn skip_ws(&mut self) {
        while let Some(&b) = self.src.get(self.pos) {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    #[inline]
    fn expect(&mut self, c: u8) -> Option<()> {
        self.skip_ws();
        if *self.src.get(self.pos)? == c {
            self.pos += 1;
            Some(())
        } else {
            None
        }
    }

    fn string(&mut self) -> Option<Str> {
        if *self.src.get(self.pos)? != b'"' {
            return None;
        }
        let start = self.pos + 1;
        let mut i = plain_run(self.src, start);
        match *self.src.get(i)? {
            b'"' => {
                self.pos = i + 1;
                return Some(Str::Source(start, i));
            }
            b'\\' => {}
            _ => return None,
        }
        self.scratch.clear();
        self.scratch.extend_from_slice(&self.src[start..i]);
        loop {
            let b = *self.src.get(i)?;
            match b {
                b'"' => {
                    self.pos = i + 1;
                    return Some(Str::Scratch);
                }
                b'\\' => {
                    let e = *self.src.get(i + 1)?;
                    i += 2;
                    let c = match e {
                        b'"' => b'"',
                        b'\\' => b'\\',
                        b'/' => b'/',
                        b'b' => 8,
                        b'f' => 12,
                        b'n' => b'\n',
                        b'r' => b'\r',
                        b't' => b'\t',
                        b'u' => {
                            let mut cp = self.hex4(i)?;
                            i += 4;
                            if (0xD800..0xDC00).contains(&cp) {
                                if self.src.get(i) != Some(&b'\\')
                                    || self.src.get(i + 1) != Some(&b'u')
                                {
                                    return None;
                                }
                                let lo = self.hex4(i + 2)?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return None;
                                }
                                i += 6;
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            } else if (0xDC00..0xE000).contains(&cp) {
                                return None;
                            }
                            let ch = char::from_u32(cp)?;
                            let mut buf = [0u8; 4];
                            self.scratch
                                .extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                            continue;
                        }
                        _ => return None,
                    };
                    self.scratch.push(c);
                }
                0..=0x1f => return None,
                _ => {
                    self.scratch.push(b);
                    i += 1;
                }
            }
        }
    }

    fn hex4(&self, at: usize) -> Option<u32> {
        let mut v = 0u32;
        for k in 0..4 {
            let d = (*self.src.get(at + k)? as char).to_digit(16)?;
            v = v * 16 + d;
        }
        Some(v)
    }

    /// A scanned string's bytes: UTF-8, as the whole source was checked once ([`read_document`])
    /// and an escape decodes to a scalar value.
    fn text(&self, s: &Str) -> &[u8] {
        match *s {
            Str::Source(a, b) => &self.src[a..b],
            Str::Scratch => &self.scratch[..],
        }
    }

    /// A number token: its text and whether it has a fraction or exponent.
    fn number(&mut self) -> Option<(&'s str, bool)> {
        let s = self.pos;
        let mut i = s;
        if self.src.get(i) == Some(&b'-') {
            i += 1;
        }
        match self.src.get(i)? {
            b'0' => i += 1,
            b'1'..=b'9' => {
                while matches!(self.src.get(i), Some(b'0'..=b'9')) {
                    i += 1;
                }
            }
            _ => return None,
        }
        let mut real = false;
        if self.src.get(i) == Some(&b'.') {
            real = true;
            i += 1;
            let d = i;
            while matches!(self.src.get(i), Some(b'0'..=b'9')) {
                i += 1;
            }
            if i == d {
                return None;
            }
        }
        if matches!(self.src.get(i), Some(b'e' | b'E')) {
            real = true;
            i += 1;
            if matches!(self.src.get(i), Some(b'+' | b'-')) {
                i += 1;
            }
            let d = i;
            while matches!(self.src.get(i), Some(b'0'..=b'9')) {
                i += 1;
            }
            if i == d {
                return None;
            }
        }
        self.pos = i;
        std::str::from_utf8(&self.src[s..i]).ok().map(|t| (t, real))
    }
}

/// An integer token (`-?(0|[1-9][0-9]*)`, as [`Scanner::number`] read it) as `i64`; `None` past its range.
fn int_of(t: &str) -> Option<i64> {
    let b = t.as_bytes();
    let (neg, digits) = match b.split_first()? {
        (b'-', rest) => (true, rest),
        _ => (false, b),
    };
    if digits.len() > 19 {
        return None;
    }
    let mut v: u64 = 0;
    for &c in digits {
        v = v * 10 + u64::from(c - b'0');
    }
    if neg {
        0i64.checked_sub_unsigned(v)
    } else {
        i64::try_from(v).ok()
    }
}

/// `NumericUtils.floatToSortableInt`.
fn float_sortable(v: f32) -> i32 {
    let bits = v.to_bits() as i32;
    bits ^ ((bits >> 31) & 0x7fff_ffff)
}

/// `NumericUtils.doubleToSortableLong`.
fn double_sortable(v: f64) -> i64 {
    let bits = v.to_bits() as i64;
    bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff)
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn digits(b: &[u8]) -> Option<i64> {
    let mut v = 0i64;
    for &c in b {
        if !c.is_ascii_digit() {
            return None;
        }
        v = v * 10 + i64::from(c - b'0');
    }
    Some(v)
}

/// `yyyy-MM-ddTHH:mm:ss[.S{1,3}]Z` in epoch milliseconds, the one shape of
/// `strict_date_optional_time` this path reproduces; anything else is `None`.
fn date_millis(t: &[u8]) -> Option<i64> {
    if t.len() < 20 || *t.last()? != b'Z' {
        return None;
    }
    if t[4] != b'-' || t[7] != b'-' || t[10] != b'T' || t[13] != b':' || t[16] != b':' {
        return None;
    }
    let (y, mo, d) = (digits(&t[0..4])?, digits(&t[5..7])?, digits(&t[8..10])?);
    let (h, mi, s) = (
        digits(&t[11..13])?,
        digits(&t[14..16])?,
        digits(&t[17..19])?,
    );
    let mut ms = 0;
    match t.len() {
        20 => {}
        22..=24 => {
            if t[19] != b'.' {
                return None;
            }
            let frac = &t[20..t.len() - 1];
            ms = digits(frac)?;
            for _ in frac.len()..3 {
                ms *= 10;
            }
        }
        _ => return None,
    }
    if y < 1000 || !(1..=12).contains(&mo) || d < 1 || h > 23 || mi > 59 || s > 59 {
        return None;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let dim = match mo {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ if leap => 29,
        _ => 28,
    };
    if d > dim {
        return None;
    }
    let days = days_from_civil(y, mo, d);
    Some(((days * 24 + h) * 60 + mi) * 60_000 + s * 1000 + ms)
}

fn utf16_len(s: &str) -> usize {
    s.chars().map(char::len_utf16).sum()
}

/// Reads `src` against `plan`; `true` with the fragment in `out`, `false` when
/// the JVM's mapper has to take the document.
pub fn parse(plan: &Plan, src: &[u8], out: &mut Vec<u8>) -> bool {
    SECTIONS.with(|s| {
        UNESCAPED.with(|u| {
            let mut sections = s.borrow_mut();
            let mut scratch = u.borrow_mut();
            let (text, path) = &mut *scratch;
            sections.clear();
            if read_document(plan, src, &mut sections, text, path).is_none() {
                return false;
            }
            out.clear();
            put_i32(out, sections.nkids as i32);
            out.extend_from_slice(&sections.kids);
            fragment(&sections.d, out);
            true
        })
    })
}

fn read_document(
    plan: &Plan,
    src: &[u8],
    sec: &mut Sections,
    scratch: &mut Vec<u8>,
    path: &mut Vec<u8>,
) -> Option<()> {
    let mut sc = Scanner {
        src,
        pos: 0,
        scratch,
        path,
    };
    sc.skip_ws();
    if *sc.src.get(sc.pos)? != b'{' || std::str::from_utf8(src).is_err() {
        return None;
    }
    read_object(plan, &mut sc, sec, 0, plan.fields.len(), 0)?;
    sc.skip_ws();
    if sc.pos != src.len() {
        return None;
    }
    flush_deferred(plan, sec);
    Some(())
}

impl Scanner<'_, '_> {
    /// The plan field a key names inside the object whose path is the first
    /// `prefix` bytes of `self.path` (the root when 0); a key with a `.` is
    /// the mapper's to expand.
    fn lookup(&mut self, plan: &Plan, key: &Str, prefix: usize, place: usize) -> Option<usize> {
        let kb = match *key {
            Str::Source(a, b) => &self.src[a..b],
            Str::Scratch => &self.scratch[..],
        };
        let guess = usize::from(plan.next[place].load(Ordering::Relaxed));
        if guess != 0 {
            let f = &plan.fields[guess - 1];
            let hit = if prefix == 0 {
                f.last_dot == NO_DOT && *f.name == *kb
            } else {
                f.last_dot as usize == prefix
                    && f.name[prefix + 1..] == *kb
                    && f.name[..prefix] == self.path[..prefix]
            };
            if hit {
                return Some(guess - 1);
            }
        }
        if kb.contains(&b'.') {
            return None;
        }
        let found = if prefix == 0 {
            plan.find(kb)
        } else {
            self.path.truncate(prefix);
            self.path.push(b'.');
            self.path.extend_from_slice(kb);
            plan.find(self.path)
        }?;
        if plan.fields[found].parent.is_some() {
            return None;
        }
        plan.next[place].store(found as u16 + 1, Ordering::Relaxed);
        Some(found)
    }

    /// Consumes the literal `null`.
    fn null(&mut self) -> bool {
        if self.src.get(self.pos..self.pos + 4) == Some(b"null") {
            self.pos += 4;
            true
        } else {
            false
        }
    }

    /// Consumes `true` or `false`.
    fn boolean(&mut self) -> Option<bool> {
        if self.src.get(self.pos..self.pos + 4) == Some(b"true") {
            self.pos += 4;
            Some(true)
        } else if self.src.get(self.pos..self.pos + 5) == Some(b"false") {
            self.pos += 5;
            Some(false)
        } else {
            None
        }
    }

    /// A number token as `f64`.
    fn double(&mut self) -> Option<f64> {
        let (t, _) = self.number()?;
        t.parse().ok()
    }

    /// `, ` between the members of an array or object, or its closing `close`; whether it closed.
    fn separator(&mut self, close: u8) -> Option<bool> {
        self.skip_ws();
        let b = *self.src.get(self.pos)?;
        if b == b',' {
            self.pos += 1;
            Some(false)
        } else if b == close {
            self.pos += 1;
            Some(true)
        } else {
            None
        }
    }
}

/// The object at the cursor, whose path is the first `prefix` bytes of the
/// scanner's path buffer (0 for the root).
fn read_object(
    plan: &Plan,
    sc: &mut Scanner<'_, '_>,
    sec: &mut Sections,
    prefix: usize,
    start: usize,
    depth: usize,
) -> Option<()> {
    if depth > MAX_DEPTH {
        return None;
    }
    sc.expect(b'{')?;
    let mut seen = Mask::default();
    let mut place = start;
    sc.skip_ws();
    if sc.src.get(sc.pos) == Some(&b'}') {
        sc.pos += 1;
        return Some(());
    }
    loop {
        sc.skip_ws();
        let key = sc.string()?;
        let Some(idx) = sc.lookup(plan, &key, prefix, place) else {
            // A name the plan does not hold with the value `null` is no value, as the mapper
            // reads it: it maps nothing and indexes nothing. Any other value is the
            // complete path's.
            sc.expect(b':')?;
            sc.skip_ws();
            sc.null().then_some(())?;
            if sc.separator(b'}')? {
                return Some(());
            }
            continue;
        };
        place = idx;
        if seen.test_and_set(idx) {
            return None;
        }
        sc.expect(b':')?;
        sc.skip_ws();
        let f = &plan.fields[idx];
        match f.kind {
            OBJECT | NESTED => {
                sc.path.clear();
                sc.path.extend_from_slice(&f.name);
                let at = sc.path.len();
                member_object(plan, f, sc, sec, at, plan.fields.len() + 1 + idx, depth)?;
            }
            _ => member(plan, f, sc, sec)?,
        }
        if sc.separator(b'}')? {
            return Some(());
        }
    }
}

/// An `object` or `nested` field's value: an object, an array of objects (each its own
/// scope, and for a nested field its own document), or `null`.
fn member_object(
    plan: &Plan,
    f: &Field,
    sc: &mut Scanner<'_, '_>,
    sec: &mut Sections,
    at: usize,
    start: usize,
    depth: usize,
) -> Option<()> {
    match *sc.src.get(sc.pos)? {
        b'{' => element(plan, f, sc, sec, at, start, depth),
        b'n' => sc.null().then_some(()),
        b'[' => {
            sc.pos += 1;
            sc.skip_ws();
            if sc.src.get(sc.pos) == Some(&b']') {
                sc.pos += 1;
                return Some(());
            }
            sec.d.repeat += 1;
            loop {
                sc.skip_ws();
                match *sc.src.get(sc.pos)? {
                    b'{' => element(plan, f, sc, sec, at, start, depth)?,
                    b'n' => sc.null().then_some(())?,
                    _ => return None,
                }
                if sc.separator(b']')? {
                    break;
                }
            }
            sec.d.repeat -= 1;
            Some(())
        }
        _ => None,
    }
}

/// One object of an object field's value: read into the document being read, or for a nested
/// field into a child document.
fn element(
    plan: &Plan,
    f: &Field,
    sc: &mut Scanner<'_, '_>,
    sec: &mut Sections,
    at: usize,
    start: usize,
    depth: usize,
) -> Option<()> {
    if f.kind != NESTED {
        return read_object(plan, sc, sec, at, start, depth + 1);
    }
    if sec.entered >= plan.nested_limit {
        return None;
    }
    sec.enter();
    read_object(plan, sc, sec, at, start, depth + 1)?;
    sec.leave(plan, usize::from(f.idx));
    Some(())
}

/// A leaf field's value: a scalar, `null` (nothing), or an array of them.
fn member(plan: &Plan, f: &Field, sc: &mut Scanner<'_, '_>, sec: &mut Sections) -> Option<()> {
    match *sc.src.get(sc.pos)? {
        b'n' => sc.null().then_some(()),
        b'[' if f.kind == GEO_POINT => geo_array(f, sc, sec),
        b'[' => array(plan, f, sc, sec),
        b'{' if f.kind == GEO_POINT => geo_object(f, sc, sec),
        b'{' => None,
        _ => value(plan, f, sc, sec),
    }
}

/// An array's values, in order; a `null` among them is no value.
fn array(plan: &Plan, f: &Field, sc: &mut Scanner<'_, '_>, sec: &mut Sections) -> Option<()> {
    sc.pos += 1;
    sc.skip_ws();
    if sc.src.get(sc.pos) == Some(&b']') {
        sc.pos += 1;
        return Some(());
    }
    // Outside an array of objects, a text array is one entry holding every value and a
    // keyword array one entry holding every distinct value; inside one, and for a field whose
    // values meet others', each value waits with the field's others ([`flush_deferred`]).
    let direct = sec.d.repeat == 0 && !f.merge && f.kind != BOOLEAN;
    let mut at = usize::MAX;
    sec.values.clear();
    sec.spans.clear();
    loop {
        sc.skip_ws();
        match *sc.src.get(sc.pos)? {
            b'n' => sc.null().then_some(())?,
            b'[' | b'{' => return None,
            _ => match f.kind {
                TEXT => {
                    let s = sc.string()?;
                    let t = sc.text(&s);
                    if !direct {
                        string_field(plan, f, t, sec)?;
                    } else {
                        if at == usize::MAX {
                            text_header(sec, f);
                            at = sec.d.inverted.len();
                            put_i32(&mut sec.d.inverted, 0);
                        }
                        put_bytes(&mut sec.d.inverted, t);
                        sec.spans.push((0, 0));
                    }
                }
                KEYWORD => {
                    let s = sc.string()?;
                    let t = sc.text(&s);
                    if !direct {
                        string_field(plan, f, t, sec)?;
                    } else if keyword_fit(f, t)? {
                        if sec.spans.len() >= MAX_ARRAY_TERMS {
                            return None;
                        }
                        let from = sec.values.len() as u32;
                        sec.values.extend_from_slice(t);
                        sec.spans.push((from, sec.values.len() as u32));
                    }
                }
                BOOLEAN => boolean_value(f, sc, sec, true)?,
                _ => value(plan, f, sc, sec)?,
            },
        }
        if sc.separator(b']')? {
            break;
        }
    }
    match f.kind {
        TEXT if at != usize::MAX => {
            let n = sec.spans.len() as i32;
            sec.d.inverted[at..at + 4].copy_from_slice(&n.to_le_bytes());
        }
        KEYWORD if direct && !sec.spans.is_empty() => {
            let values = std::mem::take(&mut sec.values);
            let spans = std::mem::take(&mut sec.spans);
            let each: Vec<&[u8]> = spans
                .iter()
                .map(|&(a, b)| &values[a as usize..b as usize])
                .collect();
            keyword_entries(f, &each, sec);
            sec.values = values;
            sec.spans = spans;
        }
        _ => {}
    }
    Some(())
}

/// A keyword, text or boolean value that waits for the document's others of its field.
fn defer(f: &Field, value: &[u8], sec: &mut Sections) -> Option<()> {
    if sec.d.deferred.len() >= MAX_DEFERRED {
        return None;
    }
    let from = sec.d.held.len() as u32;
    sec.d.held.extend_from_slice(value);
    sec.d.deferred.push((f.idx, from, sec.d.held.len() as u32));
    Some(())
}

/// The deferred values as the mapper's one entry per field: a keyword's distinct values with
/// their counts (and a doc value per value), a text's values in order.
fn flush_deferred(plan: &Plan, sec: &mut Sections) {
    if sec.d.deferred.is_empty() {
        return;
    }
    let items = std::mem::take(&mut sec.d.deferred);
    let held = std::mem::take(&mut sec.d.held);
    let mut done = Mask::default();
    for (k, &(idx, ..)) in items.iter().enumerate() {
        if done.test_and_set(usize::from(idx)) {
            continue;
        }
        let f = &plan.fields[usize::from(idx)];
        let mine: Vec<&[u8]> = items[k..]
            .iter()
            .filter(|i| i.0 == idx)
            .map(|&(_, a, b)| &held[a as usize..b as usize])
            .collect();
        match f.kind {
            TEXT => {
                text_header(sec, f);
                put_i32(&mut sec.d.inverted, mine.len() as i32);
                for t in &mine {
                    put_bytes(&mut sec.d.inverted, t);
                }
            }
            BOOLEAN => boolean_entries(f, &mine, sec),
            _ => keyword_entries(f, &mine, sec),
        }
    }
    sec.d.deferred = items;
    sec.d.deferred.clear();
    sec.d.held = held;
    sec.d.held.clear();
}

fn text_header(sec: &mut Sections, f: &Field) {
    put_i32(&mut sec.d.inverted, f.number);
    sec.d.inverted.push(2);
    put_i32(&mut sec.d.inverted, f.analyzer);
    sec.d.inverted.push(f.options);
    sec.d.inverted.push(f.norm);
    put_bytes(&mut sec.d.inverted, &f.name);
    sec.d.counts[1] += 1;
}

/// Whether a keyword's text is indexed: `Some(true)`; `Some(false)` when it is over the field's
/// `ignore_above`, which the mapper skips without a trace; `None` when the mapper would refuse
/// it (an immense term), which is the mapper's to say.
fn keyword_fit(f: &Field, t: &[u8]) -> Option<bool> {
    if f.ignore_above != u32::MAX
        && t.len() as u64 > u64::from(f.ignore_above)
        && utf16_len(std::str::from_utf8(t).ok()?) as u64 > u64::from(f.ignore_above)
    {
        return Some(false);
    }
    (t.len() <= MAX_TERM_LENGTH).then_some(true)
}

/// A string value of the text or keyword field `f`, written to it, to its multi-fields, and
/// to the fields it copies to (and their multi-fields).
fn string_field(plan: &Plan, f: &Field, t: &[u8], sec: &mut Sections) -> Option<()> {
    let wait = sec.d.repeat > 0 || f.merge;
    string_into(f, t, wait, sec)?;
    for &s in &f.subs {
        string_into(&plan.fields[usize::from(s)], t, wait, sec)?;
    }
    for &c in &f.copy_to {
        let g = &plan.fields[usize::from(c)];
        string_into(g, t, wait, sec)?;
        for &s in &g.subs {
            string_into(&plan.fields[usize::from(s)], t, wait, sec)?;
        }
    }
    Some(())
}

/// One string value of the field `g`: its entry now, or with the field's others when `wait`.
fn string_into(g: &Field, t: &[u8], wait: bool, sec: &mut Sections) -> Option<()> {
    if g.kind == KEYWORD && !keyword_fit(g, t)? {
        return Some(());
    }
    if wait {
        return defer(g, t, sec);
    }
    if g.kind == TEXT {
        text_header(sec, g);
        put_i32(&mut sec.d.inverted, 1);
        put_bytes(&mut sec.d.inverted, t);
    } else {
        keyword_entries(g, &[t], sec);
    }
    Some(())
}

/// The entries of a keyword field's values: one inverted entry whose terms are the
/// distinct values in first-seen order (a keyword field indexes documents only, so a
/// repeated value still counts once), and a doc value per value.
fn keyword_entries(f: &Field, values: &[&[u8]], sec: &mut Sections) {
    if f.flags & INDEXED != 0 {
        put_i32(&mut sec.d.inverted, f.number);
        sec.d.inverted.push(0);
        let distinct = (0..values.len())
            .filter(|&i| (0..i).all(|j| values[j] != values[i]))
            .count();
        put_i32(&mut sec.d.inverted, distinct as i32);
        for i in 0..values.len() {
            if (0..i).any(|j| values[j] == values[i]) {
                continue;
            }
            put_bytes(&mut sec.d.inverted, values[i]);
            put_i32(&mut sec.d.inverted, 1);
            sec.d.inverted.push(0);
        }
        sec.d.counts[1] += 1;
    }
    if f.flags & DOC_VALUES != 0 {
        for v in values {
            put_i32(&mut sec.d.doc_values, f.number);
            sec.d.doc_values.push(1);
            put_bytes(&mut sec.d.doc_values, v);
            sec.d.counts[2] += 1;
        }
    }
}

/// The entries of a boolean field's values: one inverted entry whose terms are the distinct
/// values (`T`, `F`) in first-seen order, and a doc value per value.
fn boolean_entries(f: &Field, values: &[&[u8]], sec: &mut Sections) {
    if f.flags & INDEXED != 0 {
        put_i32(&mut sec.d.inverted, f.number);
        sec.d.inverted.push(0);
        let distinct = (0..values.len())
            .filter(|&i| (0..i).all(|j| values[j] != values[i]))
            .count();
        put_i32(&mut sec.d.inverted, distinct as i32);
        for i in 0..values.len() {
            if (0..i).any(|j| values[j] == values[i]) {
                continue;
            }
            put_bytes(&mut sec.d.inverted, values[i]);
            put_i32(&mut sec.d.inverted, 1);
            sec.d.inverted.push(0);
        }
        sec.d.counts[1] += 1;
    }
    for v in values {
        numeric(sec, f, i64::from(v == b"T"));
    }
}

/// A boolean literal: its entries now, or with the field's others when `wait`.
fn boolean_value(
    f: &Field,
    sc: &mut Scanner<'_, '_>,
    sec: &mut Sections,
    wait: bool,
) -> Option<()> {
    let v = sc.boolean()?;
    let t: &[u8] = if v { b"T" } else { b"F" };
    if wait || sec.d.repeat > 0 {
        return defer(f, t, sec);
    }
    boolean_entries(f, &[t], sec);
    Some(())
}

/// A `geo_point` as an object: exactly `lat` and `lon`, both numbers.
fn geo_object(f: &Field, sc: &mut Scanner<'_, '_>, sec: &mut Sections) -> Option<()> {
    sc.expect(b'{')?;
    let (mut lat, mut lon) = (None, None);
    loop {
        sc.skip_ws();
        let key = sc.string()?;
        let slot = match sc.text(&key) {
            b"lat" => &mut lat,
            b"lon" => &mut lon,
            _ => return None,
        };
        if slot.is_some() {
            return None;
        }
        sc.expect(b':')?;
        sc.skip_ws();
        *slot = Some(sc.double()?);
        if sc.separator(b'}')? {
            break;
        }
    }
    geo_value(f, sec, lat?, lon?)
}

/// `[lon, lat]`: the cursor is on the `[`; exactly two numbers.
fn geo_pair(f: &Field, sc: &mut Scanner<'_, '_>, sec: &mut Sections) -> Option<()> {
    sc.expect(b'[')?;
    sc.skip_ws();
    let lon = sc.double()?;
    sc.expect(b',')?;
    sc.skip_ws();
    let lat = sc.double()?;
    sc.expect(b']')?;
    geo_value(f, sec, lat, lon)
}

/// A `geo_point` as an array: a `[lon, lat]` pair, or an array of points.
fn geo_array(f: &Field, sc: &mut Scanner<'_, '_>, sec: &mut Sections) -> Option<()> {
    let open = sc.pos;
    sc.pos += 1;
    sc.skip_ws();
    match *sc.src.get(sc.pos)? {
        b']' => {
            sc.pos += 1;
            Some(())
        }
        b'-' | b'0'..=b'9' => {
            sc.pos = open;
            geo_pair(f, sc, sec)
        }
        _ => loop {
            sc.skip_ws();
            match *sc.src.get(sc.pos)? {
                b'{' => geo_object(f, sc, sec)?,
                b'[' => geo_pair(f, sc, sec)?,
                b'"' => geo_string(f, sc, sec)?,
                _ => return None,
            }
            if sc.separator(b']')? {
                return Some(());
            }
        },
    }
}

/// `-?digits[.digits]`, as `Double.parseDouble` reads it.
fn plain_decimal(t: &[u8]) -> Option<f64> {
    let digits = t.strip_prefix(b"-").unwrap_or(t);
    let (whole, frac) = match digits.iter().position(|&b| b == b'.') {
        Some(p) => (&digits[..p], Some(&digits[p + 1..])),
        None => (digits, None),
    };
    let ok = |d: &[u8]| !d.is_empty() && d.iter().all(u8::is_ascii_digit);
    if !ok(whole) || frac.is_some_and(|d| !ok(d)) {
        return None;
    }
    std::str::from_utf8(t).ok()?.parse().ok()
}

/// A `geo_point` as `"lat,lon"` (blanks around either number are the mapper's to trim).
fn geo_string(f: &Field, sc: &mut Scanner<'_, '_>, sec: &mut Sections) -> Option<()> {
    let s = sc.string()?;
    let t = sc.text(&s);
    let comma = t.iter().position(|&b| b == b',')?;
    let trim = |b: &[u8]| {
        let from = b.iter().position(|&c| c != b' ').unwrap_or(b.len());
        let to = b.iter().rposition(|&c| c != b' ').map_or(from, |p| p + 1);
        b[from..to].to_vec()
    };
    let lat = plain_decimal(&trim(&t[..comma]))?;
    let lon = plain_decimal(&trim(&t[comma + 1..]))?;
    geo_value(f, sec, lat, lon)
}

/// One point's entries: `LatLonPoint`'s packed bytes and `LatLonDocValuesField`'s value.
fn geo_value(f: &Field, sec: &mut Sections, lat: f64, lon: f64) -> Option<()> {
    use lucene_util::geo::GeoEncodingUtils;
    let la = GeoEncodingUtils::encode_latitude(lat).ok()?;
    let lo = GeoEncodingUtils::encode_longitude(lon).ok()?;
    let mut packed = [0u8; 8];
    packed[..4].copy_from_slice(&((la as u32) ^ 0x8000_0000).to_be_bytes());
    packed[4..].copy_from_slice(&((lo as u32) ^ 0x8000_0000).to_be_bytes());
    point(sec, f, &packed);
    numeric(sec, f, (i64::from(la) << 32) | i64::from(lo as u32));
    Some(())
}

/// One scalar of a leaf field.
fn value(plan: &Plan, f: &Field, sc: &mut Scanner<'_, '_>, sec: &mut Sections) -> Option<()> {
    match f.kind {
        KEYWORD | TEXT => {
            let s = sc.string()?;
            let t = sc.text(&s);
            string_field(plan, f, t, sec)?;
        }
        BOOLEAN => boolean_value(f, sc, sec, false)?,
        GEO_POINT => match *sc.src.get(sc.pos)? {
            b'"' => geo_string(f, sc, sec)?,
            _ => return None,
        },
        INTEGER | LONG => {
            let (t, real) = sc.number()?;
            if real {
                return None;
            }
            let v = int_of(t)?;
            if f.kind == INTEGER {
                let v = i32::try_from(v).ok()?;
                point(sec, f, &((v as u32) ^ 0x8000_0000).to_be_bytes());
            } else {
                point(sec, f, &((v as u64) ^ (1 << 63)).to_be_bytes());
            }
            numeric(sec, f, v);
        }
        FLOAT => {
            let (t, real) = sc.number()?;
            let v: f32 = if real {
                t.parse().ok()?
            } else {
                int_of(t)? as f32
            };
            if !v.is_finite() {
                return None;
            }
            let s = float_sortable(v);
            point(sec, f, &((s as u32) ^ 0x8000_0000).to_be_bytes());
            numeric(sec, f, i64::from(s));
        }
        DOUBLE => {
            let (t, real) = sc.number()?;
            let v: f64 = if real {
                t.parse().ok()?
            } else {
                int_of(t)? as f64
            };
            if !v.is_finite() {
                return None;
            }
            let s = double_sortable(v);
            point(sec, f, &((s as u64) ^ (1 << 63)).to_be_bytes());
            numeric(sec, f, s);
        }
        DATE => {
            let ms = if matches!(sc.src.get(sc.pos), Some(b'0'..=b'9')) {
                // A JSON integer is read as its text: of five digits or more it is not a year of
                // `strict_date_optional_time`, so it is `epoch_millis`.
                let (t, real) = sc.number()?;
                if real || !(5..=15).contains(&t.len()) {
                    return None;
                }
                int_of(t)?
            } else {
                let s = sc.string()?;
                date_millis(match s {
                    Str::Source(a, b) => &sc.src[a..b],
                    Str::Scratch => &sc.scratch[..],
                })?
            };
            point(sec, f, &((ms as u64) ^ (1 << 63)).to_be_bytes());
            numeric(sec, f, ms);
        }
        _ => return None,
    }
    Some(())
}

fn point(sec: &mut Sections, f: &Field, packed: &[u8]) {
    if f.flags & INDEXED != 0 {
        put_i32(&mut sec.d.points, f.number);
        put_bytes(&mut sec.d.points, packed);
        sec.d.counts[3] += 1;
    }
}

fn numeric(sec: &mut Sections, f: &Field, v: i64) {
    if f.flags & DOC_VALUES != 0 {
        put_i32(&mut sec.d.doc_values, f.number);
        sec.d.doc_values.push(0);
        put_i64(&mut sec.d.doc_values, v);
        sec.d.counts[2] += 1;
    }
}

mod prepared;
mod verify;
pub use prepared::{prepare, Prepared};
pub use verify::{verify, verify_docs};

#[cfg(test)]
mod tests;
