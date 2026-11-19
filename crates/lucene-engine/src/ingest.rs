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
//! The fragment is `4 x i32` entry counts (stored, inverted, doc values,
//! points), `4 x i32` byte lengths, then the four sections.

// ARITH: offsets index a source slice already bounded by its length, counters
// are capped by MAX_FIELDS and entry sizes by the i32 section length checks.
#![allow(clippy::arithmetic_side_effects)]

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

/// Plan field flags.
const INDEXED: u8 = 1;
const DOC_VALUES: u8 = 2;

/// `IndexWriter.MAX_TERM_LENGTH`.
const MAX_TERM_LENGTH: usize = 32766;
/// Fields a plan holds (the duplicate check is one `u128`).
const MAX_FIELDS: usize = 128;

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
    /// number, i32 ignore_above, i32 analyzer, u8 index options, u8 norm mode`.
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
            if kind > OBJECT {
                return Err(invalid("kind"));
            }
            let flags = r.u8()?;
            let number = r.i32()?;
            let ignore_above = u32::try_from(r.i32()?).map_err(|_| invalid("ignore_above"))?;
            let analyzer = r.i32()?;
            let options = r.u8()?;
            let norm = r.u8()?;
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
            });
        }
        if r.pos != blob.len() {
            return Err(invalid("trailing bytes"));
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

#[derive(Default)]
struct Sections {
    stored: Vec<u8>,
    inverted: Vec<u8>,
    doc_values: Vec<u8>,
    points: Vec<u8>,
    counts: [i32; 4],
    /// The values of the keyword array being read, and where each one is in `values`.
    values: Vec<u8>,
    spans: Vec<(u32, u32)>,
    /// Arrays of objects being read (0 outside any): inside one a field can occur once per
    /// object, and its keyword and text values wait in `deferred` to become one entry.
    repeat: usize,
    /// `(field, from, to)` into `held`, in the order read.
    deferred: Vec<(u16, u32, u32)>,
    held: Vec<u8>,
}

impl Sections {
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
            for c in sections.counts {
                put_i32(out, c);
            }
            for len in [
                sections.stored.len(),
                sections.inverted.len(),
                sections.doc_values.len(),
                sections.points.len(),
            ] {
                put_i32(out, len as i32);
            }
            out.extend_from_slice(&sections.stored);
            out.extend_from_slice(&sections.inverted);
            out.extend_from_slice(&sections.doc_values);
            out.extend_from_slice(&sections.points);
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
    (sc.pos == src.len()).then_some(())
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
    let mut seen: u128 = 0;
    let mut place = start;
    sc.skip_ws();
    if sc.src.get(sc.pos) == Some(&b'}') {
        sc.pos += 1;
        return Some(());
    }
    loop {
        sc.skip_ws();
        let key = sc.string()?;
        let idx = sc.lookup(plan, &key, prefix, place)?;
        place = idx;
        if seen >> idx & 1 == 1 {
            return None;
        }
        seen |= 1u128 << idx;
        sc.expect(b':')?;
        sc.skip_ws();
        let f = &plan.fields[idx];
        if f.kind == OBJECT {
            sc.path.clear();
            sc.path.extend_from_slice(&f.name);
            let at = sc.path.len();
            member_object(plan, sc, sec, at, plan.fields.len() + 1 + idx, depth)?;
        } else {
            member(f, sc, sec)?;
        }
        sc.skip_ws();
        match *sc.src.get(sc.pos)? {
            b',' => sc.pos += 1,
            b'}' => {
                sc.pos += 1;
                return Some(());
            }
            _ => return None,
        }
    }
}

/// An `object` field's value: an object, an array of objects (each its own
/// scope), or `null`.
fn member_object(
    plan: &Plan,
    sc: &mut Scanner<'_, '_>,
    sec: &mut Sections,
    at: usize,
    start: usize,
    depth: usize,
) -> Option<()> {
    match *sc.src.get(sc.pos)? {
        b'{' => read_object(plan, sc, sec, at, start, depth + 1),
        b'n' => sc.null().then_some(()),
        b'[' => {
            sc.pos += 1;
            sc.skip_ws();
            if sc.src.get(sc.pos) == Some(&b']') {
                sc.pos += 1;
                return Some(());
            }
            sec.repeat += 1;
            loop {
                sc.skip_ws();
                match *sc.src.get(sc.pos)? {
                    b'{' => read_object(plan, sc, sec, at, start, depth + 1)?,
                    b'n' => sc.null().then_some(())?,
                    _ => return None,
                }
                sc.skip_ws();
                match *sc.src.get(sc.pos)? {
                    b',' => sc.pos += 1,
                    b']' => {
                        sc.pos += 1;
                        break;
                    }
                    _ => return None,
                }
            }
            sec.repeat -= 1;
            if sec.repeat == 0 {
                flush_deferred(plan, sec);
            }
            Some(())
        }
        _ => None,
    }
}

/// A leaf field's value: a scalar, `null` (nothing), or an array of them.
fn member(f: &Field, sc: &mut Scanner<'_, '_>, sec: &mut Sections) -> Option<()> {
    match *sc.src.get(sc.pos)? {
        b'n' => sc.null().then_some(()),
        b'[' => array(f, sc, sec),
        b'{' => None,
        _ => value(f, sc, sec),
    }
}

/// An array's values, in order; a `null` among them is no value.
fn array(f: &Field, sc: &mut Scanner<'_, '_>, sec: &mut Sections) -> Option<()> {
    sc.pos += 1;
    sc.skip_ws();
    if sc.src.get(sc.pos) == Some(&b']') {
        sc.pos += 1;
        return Some(());
    }
    // Outside an array of objects, a text array is one entry holding every value and a
    // keyword array one entry holding every distinct value; inside one each value waits
    // with the field's others ([`flush_deferred`]).
    let direct = sec.repeat == 0;
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
                        defer(f, t, sec)?;
                    } else {
                        if at == usize::MAX {
                            text_header(sec, f);
                            at = sec.inverted.len();
                            put_i32(&mut sec.inverted, 0);
                        }
                        put_bytes(&mut sec.inverted, t);
                        sec.spans.push((0, 0));
                    }
                }
                KEYWORD => {
                    let s = sc.string()?;
                    let t = keyword_text(f, sc, &s)?;
                    if !direct {
                        defer(f, t, sec)?;
                    } else {
                        if sec.spans.len() >= MAX_ARRAY_TERMS {
                            return None;
                        }
                        let from = sec.values.len() as u32;
                        sec.values.extend_from_slice(t);
                        sec.spans.push((from, sec.values.len() as u32));
                    }
                }
                _ => value(f, sc, sec)?,
            },
        }
        sc.skip_ws();
        match *sc.src.get(sc.pos)? {
            b',' => sc.pos += 1,
            b']' => {
                sc.pos += 1;
                break;
            }
            _ => return None,
        }
    }
    match f.kind {
        TEXT if at != usize::MAX => {
            let n = sec.spans.len() as i32;
            sec.inverted[at..at + 4].copy_from_slice(&n.to_le_bytes());
        }
        KEYWORD if !sec.spans.is_empty() => {
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

/// A keyword or text value read inside an array of objects.
fn defer(f: &Field, value: &[u8], sec: &mut Sections) -> Option<()> {
    if sec.deferred.len() >= MAX_DEFERRED {
        return None;
    }
    let from = sec.held.len() as u32;
    sec.held.extend_from_slice(value);
    sec.deferred.push((f.idx, from, sec.held.len() as u32));
    Some(())
}

/// The deferred values as the mapper's one entry per field: a keyword's distinct values with
/// their counts (and a doc value per value), a text's values in order.
fn flush_deferred(plan: &Plan, sec: &mut Sections) {
    let items = std::mem::take(&mut sec.deferred);
    let held = std::mem::take(&mut sec.held);
    let mut done: u128 = 0;
    for (k, &(idx, ..)) in items.iter().enumerate() {
        if done >> idx & 1 == 1 {
            continue;
        }
        done |= 1u128 << idx;
        let f = &plan.fields[usize::from(idx)];
        let mine: Vec<&[u8]> = items[k..]
            .iter()
            .filter(|i| i.0 == idx)
            .map(|&(_, a, b)| &held[a as usize..b as usize])
            .collect();
        if f.kind == TEXT {
            text_header(sec, f);
            put_i32(&mut sec.inverted, mine.len() as i32);
            for t in &mine {
                put_bytes(&mut sec.inverted, t);
            }
        } else {
            keyword_entries(f, &mine, sec);
        }
    }
    sec.deferred = items;
    sec.deferred.clear();
    sec.held = held;
    sec.held.clear();
}

fn text_header(sec: &mut Sections, f: &Field) {
    put_i32(&mut sec.inverted, f.number);
    sec.inverted.push(2);
    put_i32(&mut sec.inverted, f.analyzer);
    sec.inverted.push(f.options);
    sec.inverted.push(f.norm);
    put_bytes(&mut sec.inverted, &f.name);
    sec.counts[1] += 1;
}

/// A keyword's text, refused when the mapper would not index it as is.
fn keyword_text<'a>(f: &Field, sc: &'a Scanner<'_, '_>, s: &Str) -> Option<&'a [u8]> {
    let t = sc.text(s);
    if t.len() > MAX_TERM_LENGTH {
        return None;
    }
    if f.ignore_above != u32::MAX
        && t.len() as u64 > u64::from(f.ignore_above)
        && utf16_len(std::str::from_utf8(t).ok()?) as u64 > u64::from(f.ignore_above)
    {
        return None;
    }
    Some(t)
}

/// The entries of a keyword field's values: one inverted entry whose terms are the
/// distinct values in first-seen order (a keyword field indexes documents only, so a
/// repeated value still counts once), and a doc value per value.
fn keyword_entries(f: &Field, values: &[&[u8]], sec: &mut Sections) {
    if f.flags & INDEXED != 0 {
        put_i32(&mut sec.inverted, f.number);
        sec.inverted.push(0);
        let distinct = (0..values.len())
            .filter(|&i| (0..i).all(|j| values[j] != values[i]))
            .count();
        put_i32(&mut sec.inverted, distinct as i32);
        for i in 0..values.len() {
            if (0..i).any(|j| values[j] == values[i]) {
                continue;
            }
            put_bytes(&mut sec.inverted, values[i]);
            put_i32(&mut sec.inverted, 1);
            sec.inverted.push(0);
        }
        sec.counts[1] += 1;
    }
    if f.flags & DOC_VALUES != 0 {
        for v in values {
            put_i32(&mut sec.doc_values, f.number);
            sec.doc_values.push(1);
            put_bytes(&mut sec.doc_values, v);
            sec.counts[2] += 1;
        }
    }
}

/// One scalar of a leaf field.
fn value(f: &Field, sc: &mut Scanner<'_, '_>, sec: &mut Sections) -> Option<()> {
    match f.kind {
        KEYWORD => {
            let s = sc.string()?;
            let b = keyword_text(f, sc, &s)?;
            if sec.repeat > 0 {
                return defer(f, b, sec);
            }
            if f.flags & INDEXED != 0 {
                put_i32(&mut sec.inverted, f.number);
                sec.inverted.push(0);
                put_i32(&mut sec.inverted, 1);
                put_bytes(&mut sec.inverted, b);
                put_i32(&mut sec.inverted, 1);
                sec.inverted.push(0);
                sec.counts[1] += 1;
            }
            if f.flags & DOC_VALUES != 0 {
                put_i32(&mut sec.doc_values, f.number);
                sec.doc_values.push(1);
                put_bytes(&mut sec.doc_values, b);
                sec.counts[2] += 1;
            }
        }
        TEXT => {
            let s = sc.string()?;
            let t = sc.text(&s);
            if sec.repeat > 0 {
                return defer(f, t, sec);
            }
            text_header(sec, f);
            put_i32(&mut sec.inverted, 1);
            put_bytes(&mut sec.inverted, t);
        }
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
            let s = sc.string()?;
            let ms = date_millis(match s {
                Str::Source(a, b) => &sc.src[a..b],
                Str::Scratch => &sc.scratch[..],
            })?;
            point(sec, f, &((ms as u64) ^ (1 << 63)).to_be_bytes());
            numeric(sec, f, ms);
        }
        _ => return None,
    }
    Some(())
}

fn point(sec: &mut Sections, f: &Field, packed: &[u8]) {
    if f.flags & INDEXED != 0 {
        put_i32(&mut sec.points, f.number);
        put_bytes(&mut sec.points, packed);
        sec.counts[3] += 1;
    }
}

fn numeric(sec: &mut Sections, f: &Field, v: i64) {
    if f.flags & DOC_VALUES != 0 {
        put_i32(&mut sec.doc_values, f.number);
        sec.doc_values.push(0);
        put_i64(&mut sec.doc_values, v);
        sec.counts[2] += 1;
    }
}

mod verify;
pub use verify::verify;

#[cfg(test)]
mod tests;
