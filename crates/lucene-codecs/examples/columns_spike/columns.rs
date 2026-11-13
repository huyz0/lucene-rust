//! SPIKE, not production code: a prototype block-columnar value array
//! ("Columns100") to measure against Lucene90 and Adaptive90 doc values.
//!
//! Layout: values cut into vectors of 1024. Each vector is one of
//! CONST, FOR (frame of reference + bit-pack), LANE_DELTA (16 lanes, delta
//! between rows of the same lane, then FOR; row 0 kept raw) or ALP (doubles
//! as scaled integers `d / 10^e`, exceptions kept verbatim, then FOR). Bits are
//! packed in a 16-lane interleaved layout (FastLanes-style) so the unpack loop
//! over lanes auto-vectorises. A zone entry per vector keeps min/max (sortable
//! long domain), and for integer columns the exact sum.
#![allow(clippy::arithmetic_side_effects, clippy::needless_range_loop)]

pub const N: usize = 1024;
const LANES: usize = 16;
const ROWS: usize = N / LANES;

/// What a column holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// Longs (long, int, date, keyword ordinals).
    Int,
    /// Doubles stored as sortable longs (`NumericUtils.doubleToSortableLong`).
    Double,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Codec {
    Const,
    For,
    LaneDelta,
    Alp,
    DeltaT,
}

/// Per-vector header + zone entry.
#[derive(Clone, Debug)]
pub struct Zone {
    pub codec: Codec,
    pub width: u8,
    pub exp: u8,
    pub count: u16,
    pub base: i64,
    pub gcd: i64,
    pub word_off: u32,
    pub row0_off: u32,
    pub exc_off: u32,
    pub exc_n: u16,
    pub min: i64,
    pub max: i64,
    pub isum: i128,
}

pub struct Column {
    pub kind: Kind,
    pub len: usize,
    pub zones: Vec<Zone>,
    pub words: Vec<u64>,
    pub row0: Vec<i64>,
    pub exc_pos: Vec<u16>,
    pub exc_val: Vec<i64>,
}

/// Bytes on disk per vector besides its packed words: a 16-byte header
/// (codec, width, exponent, exception count, base) and a 32-byte zone entry
/// (first doc, count, page offset, min, max); integer columns add a 16-byte
/// exact sum.
pub const HEADER_BYTES: usize = 24;
pub const ZONE_BYTES: usize = 32;
pub const SUM_BYTES: usize = 16;
pub const PAGE: usize = 16 << 10;

pub fn sortable_to_f64(v: i64) -> f64 {
    f64::from_bits((v ^ ((v >> 63) & 0x7fff_ffff_ffff_ffff)) as u64)
}
pub fn f64_to_sortable(x: f64) -> i64 {
    let b = x.to_bits() as i64;
    b ^ ((b >> 63) & 0x7fff_ffff_ffff_ffff)
}

const IP10: [f64; 19] = [
    1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14,
    1e-15, 1e-16, 1e-17, 1e-18,
];

/// Decode by multiplying with `10^-e` (ALP's choice) instead of dividing by
/// `10^e`; the encoder verifies with the same formula.
pub const ALP_MUL: bool = option_env!("ALP_MUL").is_some();

#[inline(always)]
fn alp_decode(d: i64, e: usize) -> f64 {
    if ALP_MUL {
        (d as f64) * IP10[e]
    } else {
        (d as f64) / P10[e]
    }
}

const P10: [f64; 19] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18,
];

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn gcd_of(xs: impl Iterator<Item = u64>) -> u64 {
    let mut g = 0u64;
    for x in xs {
        g = gcd(g, x);
        if g == 1 {
            break;
        }
    }
    g.max(1)
}

fn width(range: u64) -> u32 {
    64 - range.leading_zeros()
}

fn pack(vals: &[u64; N], w: u32, out: &mut Vec<u64>) {
    if w == 0 {
        return;
    }
    let w = w as usize;
    let start = out.len();
    out.resize(start + w * LANES, 0);
    let o = &mut out[start..];
    for r in 0..ROWS {
        let bit = r * w;
        let k = bit >> 6;
        let s = bit & 63;
        for l in 0..LANES {
            let v = vals[r * LANES + l];
            o[k * LANES + l] |= v << s;
            if s + w > 64 {
                o[(k + 1) * LANES + l] |= v >> (64 - s);
            }
        }
    }
}

#[inline(always)]
fn unpack_w<const W: usize>(i: &[u64], o: &mut [u64; N]) {
    let mask = if W == 64 { u64::MAX } else { (1u64 << W) - 1 };
    let i = &i[..W * LANES];
    for r in 0..ROWS {
        let bit = r * W;
        let k = bit >> 6;
        let s = bit & 63;
        let dst: &mut [u64; LANES] = (&mut o[r * LANES..r * LANES + LANES]).try_into().unwrap();
        let a: &[u64; LANES] = i[k * LANES..k * LANES + LANES].try_into().unwrap();
        if s + W > 64 {
            let b: &[u64; LANES] = i[(k + 1) * LANES..(k + 1) * LANES + LANES]
                .try_into()
                .unwrap();
            for l in 0..LANES {
                dst[l] = ((a[l] >> s) | (b[l] << (64 - s))) & mask;
            }
        } else {
            for l in 0..LANES {
                dst[l] = (a[l] >> s) & mask;
            }
        }
    }
}

macro_rules! dispatch {
    ($w:expr, $i:expr, $o:expr; $($n:literal),*) => {
        match $w {
            0 => $o.fill(0),
            $($n => unpack_w::<$n>($i, $o),)*
            _ => unreachable!(),
        }
    };
}

fn unpack(w: u8, i: &[u64], o: &mut [u64; N]) {
    dispatch!(w, i, o; 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64)
}

/// Exact decimal encoding of `x` at exponent `e`, if `d / 10^e == x` bit for bit.
#[inline]
fn alp_encode(x: f64, e: usize) -> Option<i64> {
    if !x.is_finite() || (x == 0.0 && x.is_sign_negative()) {
        return None;
    }
    let s = x * P10[e];
    if s.abs() >= 9.0e15 {
        return None;
    }
    let d = s.round() as i64;
    if alp_decode(d, e) == x {
        Some(d)
    } else {
        None
    }
}

impl Column {
    pub fn encode(kind: Kind, values: &[i64]) -> Column {
        let mut c = Column {
            kind,
            len: values.len(),
            zones: Vec::with_capacity(values.len().div_ceil(N)),
            words: Vec::new(),
            row0: Vec::new(),
            exc_pos: Vec::new(),
            exc_val: Vec::new(),
        };
        let mut buf = [0i64; N];
        for chunk in values.chunks(N) {
            buf[..chunk.len()].copy_from_slice(chunk);
            let last = *chunk.last().unwrap();
            buf[chunk.len()..].fill(last);
            let z = match kind {
                Kind::Int => c.encode_int(&buf, chunk.len()),
                Kind::Double => c.encode_double(&buf, chunk.len()),
            };
            c.zones.push(z);
        }
        c
    }

    fn zone(&self, codec: Codec, count: usize, v: &[i64; N]) -> Zone {
        let n = count;
        let (mut min, mut max, mut sum) = (i64::MAX, i64::MIN, 0i128);
        for &x in &v[..n] {
            min = min.min(x);
            max = max.max(x);
            sum += x as i128;
        }
        Zone {
            codec,
            width: 0,
            exp: 0,
            count: n as u16,
            base: 0,
            gcd: 1,
            word_off: self.words.len() as u32,
            row0_off: self.row0.len() as u32,
            exc_off: self.exc_pos.len() as u32,
            exc_n: 0,
            min,
            max,
            isum: sum,
        }
    }

    fn encode_int(&mut self, v: &[i64; N], count: usize) -> Zone {
        let mut z = self.zone(Codec::For, count, v);
        let (vmin, vmax) = v
            .iter()
            .fold((i64::MAX, i64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        if vmin == vmax {
            z.codec = Codec::Const;
            z.base = vmin;
            return z;
        }
        let g_for = gcd_of(v.iter().map(|&x| x.wrapping_sub(vmin) as u64));
        let w_for = width(vmax.wrapping_sub(vmin) as u64 / g_for);
        // Lane-contiguous delta: lane l holds values l*64 .. l*64+63, so a
        // delta is between neighbours; row 0 (each lane's first value) raw.
        let mut d = [0i64; N];
        let (mut dmin, mut dmax) = (i64::MAX, i64::MIN);
        for l in 0..LANES {
            for r in 1..ROWS {
                let i = l * ROWS + r;
                let x = v[i].wrapping_sub(v[i - 1]);
                d[i] = x;
                dmin = dmin.min(x);
                dmax = dmax.max(x);
            }
        }
        let g_d = gcd_of(
            (0..LANES)
                .flat_map(|l| (1..ROWS).map(move |r| l * ROWS + r))
                .map(|i| d[i].wrapping_sub(dmin) as u64),
        );
        let w_delta = width(dmax.wrapping_sub(dmin) as u64 / g_d);
        let gcd_cost = |g: u64| if g > 1 { 1 } else { 0 };
        let mut tmp = [0u64; N];
        if (w_delta as usize) * LANES * 8 + LANES * 8 < (w_for as usize) * LANES * 8 {
            z.codec = Codec::DeltaT;
            z.width = w_delta as u8;
            z.base = dmin;
            z.gcd = g_d as i64;
            let _ = gcd_cost;
            for l in 0..LANES {
                self.row0.push(v[l * ROWS]);
                for r in 1..ROWS {
                    tmp[r * LANES + l] = d[l * ROWS + r].wrapping_sub(dmin) as u64 / g_d;
                }
            }
            pack(&tmp, w_delta, &mut self.words);
        } else {
            z.codec = Codec::For;
            z.width = w_for as u8;
            z.base = vmin;
            z.gcd = g_for as i64;
            for i in 0..N {
                tmp[i] = v[i].wrapping_sub(vmin) as u64 / g_for;
            }
            pack(&tmp, w_for, &mut self.words);
        }
        z
    }

    fn encode_double(&mut self, v: &[i64; N], count: usize) -> Zone {
        let mut z = self.zone(Codec::Alp, count, v);
        z.isum = 0;
        if v.iter().all(|&x| x == v[0]) {
            z.codec = Codec::Const;
            z.base = v[0];
            return z;
        }
        // Pick the exponent that encodes most of a sample exactly.
        let xs: Vec<f64> = v.iter().map(|&s| sortable_to_f64(s)).collect();
        let mut best = (0usize, 0usize);
        for e in 0..P10.len() {
            let hits = (0..N)
                .step_by(16)
                .filter(|&i| alp_encode(xs[i], e).is_some())
                .count();
            if hits > best.1 {
                best = (e, hits);
            }
            if hits == N / 16 {
                break;
            }
        }
        if best.1 * 2 < N / 16 {
            // Not decimal-like: FOR over the sortable longs.
            let mut z2 = self.encode_int(v, count);
            z2.isum = 0;
            return z2;
        }
        let e = best.0;
        let mut d = [0i64; N];
        let mut prev = None;
        let mut exc = Vec::new();
        for i in 0..N {
            match alp_encode(xs[i], e) {
                Some(x) => {
                    d[i] = x;
                    prev = Some(x);
                }
                None => {
                    exc.push(i);
                    d[i] = prev.unwrap_or(0);
                }
            }
        }
        // Leading exceptions: borrow the first encodable value.
        if let Some(first) = (0..N).find(|i| !exc.contains(i)) {
            for &i in &exc {
                if i < first {
                    d[i] = d[first];
                }
            }
        }
        let (dmin, dmax) = d
            .iter()
            .fold((i64::MAX, i64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        let w = width(dmax.wrapping_sub(dmin) as u64);
        // Fall back to FOR on sortable longs when that is smaller.
        let (smin, smax) = v
            .iter()
            .fold((i64::MAX, i64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        let w_raw = width(smax.wrapping_sub(smin) as u64);
        if (w_raw as usize) * LANES * 8 <= (w as usize) * LANES * 8 + exc.len() * 10 {
            let mut z2 = self.encode_int(v, count);
            z2.isum = 0;
            return z2;
        }
        z.width = w as u8;
        z.exp = e as u8;
        z.base = dmin;
        z.exc_n = exc.len() as u16;
        for &i in &exc {
            self.exc_pos.push(i as u16);
            self.exc_val.push(v[i]);
        }
        let mut tmp = [0u64; N];
        for i in 0..N {
            tmp[i] = d[i].wrapping_sub(dmin) as u64;
        }
        pack(&tmp, w, &mut self.words);
        z
    }

    fn words_of(&self, z: &Zone) -> &[u64] {
        let off = z.word_off as usize;
        &self.words[off..off + z.width as usize * LANES]
    }

    /// Vector `i` as longs (sortable longs for doubles: the doc-values contract).
    pub fn decode_i64(&self, i: usize, out: &mut [i64; N]) {
        let z = &self.zones[i];
        match z.codec {
            Codec::Const => out.fill(z.base),
            Codec::For => {
                let mut u = [0u64; N];
                unpack(z.width, self.words_of(z), &mut u);
                if z.gcd == 1 {
                    for (o, &x) in out.iter_mut().zip(u.iter()) {
                        *o = (x as i64).wrapping_add(z.base);
                    }
                } else {
                    for (o, &x) in out.iter_mut().zip(u.iter()) {
                        *o = (x as i64).wrapping_mul(z.gcd).wrapping_add(z.base);
                    }
                }
            }
            Codec::DeltaT => {
                let mut u = [0u64; N];
                unpack(z.width, self.words_of(z), &mut u);
                let r0 = z.row0_off as usize;
                let mut acc: [i64; LANES] = self.row0[r0..r0 + LANES].try_into().unwrap();
                for l in 0..LANES {
                    out[l * ROWS] = acc[l];
                }
                for r in 1..ROWS {
                    let d: &[u64; LANES] = u[r * LANES..r * LANES + LANES].try_into().unwrap();
                    for l in 0..LANES {
                        acc[l] = acc[l]
                            .wrapping_add((d[l] as i64).wrapping_mul(z.gcd).wrapping_add(z.base));
                    }
                    for l in 0..LANES {
                        out[l * ROWS + r] = acc[l];
                    }
                }
            }
            Codec::LaneDelta => {
                let mut u = [0u64; N];
                unpack(z.width, self.words_of(z), &mut u);
                let r0 = z.row0_off as usize;
                out[..LANES].copy_from_slice(&self.row0[r0..r0 + LANES]);
                for r in 1..ROWS {
                    let (prev, cur) = out.split_at_mut(r * LANES);
                    let prev: &[i64; LANES] = prev[(r - 1) * LANES..].try_into().unwrap();
                    let cur: &mut [i64; LANES] = (&mut cur[..LANES]).try_into().unwrap();
                    let d: &[u64; LANES] = u[r * LANES..r * LANES + LANES].try_into().unwrap();
                    for l in 0..LANES {
                        cur[l] = prev[l].wrapping_add((d[l] as i64).wrapping_add(z.base));
                    }
                }
            }
            Codec::Alp => {
                let mut f = [0f64; N];
                self.decode_f64(i, &mut f);
                for (o, x) in out.iter_mut().zip(f.iter()) {
                    *o = f64_to_sortable(*x);
                }
            }
        }
    }

    /// Vector `i` of a double column as doubles.
    pub fn decode_f64(&self, i: usize, out: &mut [f64; N]) {
        let z = &self.zones[i];
        match z.codec {
            Codec::Alp => {
                let mut u = [0u64; N];
                unpack(z.width, self.words_of(z), &mut u);
                let e = z.exp as usize;
                for (o, &x) in out.iter_mut().zip(u.iter()) {
                    *o = alp_decode((x as i64).wrapping_add(z.base), e);
                }
                let e0 = z.exc_off as usize;
                for k in e0..e0 + z.exc_n as usize {
                    out[self.exc_pos[k] as usize] = sortable_to_f64(self.exc_val[k]);
                }
            }
            _ => {
                let mut s = [0i64; N];
                self.decode_i64(i, &mut s);
                for (o, &x) in out.iter_mut().zip(s.iter()) {
                    *o = sortable_to_f64(x);
                }
            }
        }
    }

    /// Bytes on disk: (unaligned, 16 KiB pages with no vector crossing a page).
    pub fn size(&self) -> (usize, usize) {
        let per_vec =
            HEADER_BYTES + ZONE_BYTES + if self.kind == Kind::Int { SUM_BYTES } else { 0 };
        let meta = self.zones.len() * per_vec;
        let side = self.row0.len() * 8 + self.exc_pos.len() * 10;
        let mut flat = 0usize;
        let mut paged = 0usize;
        for z in &self.zones {
            let b = z.width as usize * LANES * 8;
            flat += b;
            let room = PAGE - paged % PAGE;
            if b > room {
                paged += room;
            }
            paged += b;
        }
        (meta + side + flat, meta + side + paged)
    }

    pub fn codec_mix(&self) -> String {
        let mut c = [0usize; 5];
        for z in &self.zones {
            c[z.codec as usize] += 1;
        }
        let n = self.zones.len().max(1) as f64;
        format!(
            "const {:.0}% for {:.0}% delta {:.0}% alp {:.0}% exc/vec {:.1}",
            100.0 * c[0] as f64 / n,
            100.0 * c[1] as f64 / n,
            100.0 * (c[2] + c[4]) as f64 / n,
            100.0 * c[3] as f64 / n,
            self.exc_pos.len() as f64 / n
        )
    }
}
