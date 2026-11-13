//! SPIKE: Columns100 prototype vs Lucene90 and Adaptive90 doc values.
//!
//!   cargo run --release -p lucene-codecs --example columns_spike -- COLS_DIR [reps]
//!
//! COLS_DIR holds `<dataset>_<column>.i64` files from
//! `/mnt/project-files/perf/columns-spike/prep.py`. Prints markdown tables.
#![allow(clippy::arithmetic_side_effects, clippy::needless_range_loop)]

mod columns;

use std::hint::black_box;
use std::time::Instant;

use columns::{sortable_to_f64, Column, Kind, N};
use lucene_codecs::doc_values::{
    self, DenseField, Lucene90DocValuesFormat, NumericEntry, NumericReader,
};
use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos};

const ID: [u8; 16] = [9; 16];
const SKIP: usize = 4096;

struct DvCol {
    dvm_len: usize,
    data: Vec<u8>,
    entry: NumericEntry,
}

fn write_dv(values: &[i64], format: Lucene90DocValuesFormat) -> DvCol {
    let fis = FieldInfos {
        fields: vec![FieldInfo::new("f", 0).with_doc_values(
            DocValuesType::Numeric,
            DocValuesSkipIndexType::None,
            -1,
        )],
    };
    let suffix = format!("{}_0", format.name());
    let (dvm, dvd, _) = doc_values::write_fields_with_format(
        &[DenseField::Numeric(0, values)],
        &[],
        values.len() as i32,
        &ID,
        &suffix,
        format,
    )
    .unwrap();
    let (_, meta) = doc_values::parse_meta(&dvm, &ID, &suffix, &fis).unwrap();
    let entry = meta.numeric_entry(0).unwrap().clone();
    DvCol {
        dvm_len: dvm.len(),
        data: dvd,
        entry,
    }
}

/// Per-4096-doc min/max: what `.dvs` level 0 holds.
fn skip_index(values: &[i64]) -> Vec<(i64, i64)> {
    values
        .chunks(SKIP)
        .map(|c| {
            c.iter()
                .fold((i64::MAX, i64::MIN), |(a, b), &x| (a.min(x), b.max(x)))
        })
        .collect()
}

fn load(dir: &str, name: &str) -> Vec<i64> {
    let b = std::fs::read(format!("{dir}/{name}.i64")).unwrap();
    b.chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn time<R>(reps: usize, mut f: impl FnMut() -> R) -> (f64, R) {
    let mut r = f();
    let mut ts = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        r = black_box(f());
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (ts[ts.len() / 2], r)
}

// ---------- kernels on Lucene90 / Adaptive90 through NumericReader::fill_window

fn dv_sum(c: &DvCol, n: usize, kind: Kind) -> (i64, f64) {
    let mut r = NumericReader::new(&c.data, &c.entry);
    let mut w = vec![0i64; N];
    let mut p = vec![0u64; N / 64];
    let (mut is, mut fs) = (0i64, 0f64);
    for start in (0..n).step_by(N) {
        let len = (n - start).min(N);
        r.fill_window(start as i32, &mut w[..len], &mut p).unwrap();
        match kind {
            Kind::Int => {
                for &x in &w[..len] {
                    is = is.wrapping_add(x);
                }
            }
            Kind::Double => {
                for &x in &w[..len] {
                    fs += sortable_to_f64(x);
                }
            }
        }
    }
    (is, fs)
}

fn dv_minmax(c: &DvCol, n: usize) -> (i64, i64) {
    let mut r = NumericReader::new(&c.data, &c.entry);
    let mut w = vec![0i64; N];
    let mut p = vec![0u64; N / 64];
    let (mut lo, mut hi) = (i64::MAX, i64::MIN);
    for start in (0..n).step_by(N) {
        let len = (n - start).min(N);
        r.fill_window(start as i32, &mut w[..len], &mut p).unwrap();
        for &x in &w[..len] {
            lo = lo.min(x);
            hi = hi.max(x);
        }
    }
    (lo, hi)
}

/// Accept words for `lo <= ts <= hi`, using a 4096-doc skip index when given.
fn dv_filter(
    ts: &DvCol,
    n: usize,
    lo: i64,
    hi: i64,
    skip: Option<&[(i64, i64)]>,
    words: &mut [u64],
) {
    words.fill(0);
    let mut r = NumericReader::new(&ts.data, &ts.entry);
    let mut w = vec![0i64; N];
    let mut p = vec![0u64; N / 64];
    for start in (0..n).step_by(N) {
        let len = (n - start).min(N);
        let wi = start / 64;
        if let Some(s) = skip {
            let (a, b) = s[start / SKIP];
            if b < lo || a > hi {
                continue;
            }
            if a >= lo && b <= hi {
                set_all(&mut words[wi..], len);
                continue;
            }
        }
        r.fill_window(start as i32, &mut w[..len], &mut p).unwrap();
        for (i, &x) in w[..len].iter().enumerate() {
            words[wi + i / 64] |= ((x >= lo && x <= hi) as u64) << (i % 64);
        }
    }
}

fn set_all(words: &mut [u64], len: usize) {
    for i in 0..len / 64 {
        words[i] = u64::MAX;
    }
    if len % 64 != 0 {
        words[len / 64] = (1u64 << (len % 64)) - 1;
    }
}

/// Filtered sum of `agg` over accepted docs (doubles summed in doc order).
fn dv_filtered_sum(agg: &DvCol, n: usize, kind: Kind, words: &[u64]) -> (u64, i64, f64) {
    let mut r = NumericReader::new(&agg.data, &agg.entry);
    let mut w = vec![0i64; N];
    let mut p = vec![0u64; N / 64];
    let (mut cnt, mut is, mut fs) = (0u64, 0i64, 0f64);
    for start in (0..n).step_by(N) {
        let len = (n - start).min(N);
        let ws = &words[start / 64..(start + len).div_ceil(64)];
        if ws.iter().all(|&x| x == 0) {
            continue;
        }
        r.fill_window(start as i32, &mut w[..len], &mut p).unwrap();
        for (k, &word) in ws.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let i = k * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                cnt += 1;
                match kind {
                    Kind::Int => is = is.wrapping_add(w[i]),
                    Kind::Double => fs += sortable_to_f64(w[i]),
                }
            }
        }
    }
    (cnt, is, fs)
}

/// Terms-style: count and double sum per ordinal over accepted docs.
fn dv_group(
    ord: &DvCol,
    val: &DvCol,
    n: usize,
    words: &[u64],
    card: usize,
) -> (Vec<u64>, Vec<f64>) {
    let mut ro = NumericReader::new(&ord.data, &ord.entry);
    let mut rv = NumericReader::new(&val.data, &val.entry);
    let (mut wo, mut wv) = (vec![0i64; N], vec![0i64; N]);
    let mut p = vec![0u64; N / 64];
    let (mut counts, mut sums) = (vec![0u64; card], vec![0f64; card]);
    for start in (0..n).step_by(N) {
        let len = (n - start).min(N);
        let ws = &words[start / 64..(start + len).div_ceil(64)];
        if ws.iter().all(|&x| x == 0) {
            continue;
        }
        ro.fill_window(start as i32, &mut wo[..len], &mut p)
            .unwrap();
        rv.fill_window(start as i32, &mut wv[..len], &mut p)
            .unwrap();
        for (k, &word) in ws.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let i = k * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let o = wo[i] as usize;
                counts[o] += 1;
                sums[o] += sortable_to_f64(wv[i]);
            }
        }
    }
    (counts, sums)
}

// ---------- kernels on Columns100

fn col_sum(c: &Column) -> (i64, f64) {
    let (mut is, mut fs) = (0i64, 0f64);
    match c.kind {
        Kind::Int => {
            // Exact integer sums from zones (wrapping to match the baseline).
            for z in &c.zones {
                is = is.wrapping_add(z.isum as i64);
            }
        }
        Kind::Double => {
            let mut f = [0f64; N];
            for (i, z) in c.zones.iter().enumerate() {
                c.decode_f64(i, &mut f);
                for &x in &f[..z.count as usize] {
                    fs += x;
                }
            }
        }
    }
    (is, fs)
}

/// Decode every vector (no zone shortcut): raw decode speed.
fn col_decode_sum(c: &Column) -> (i64, f64) {
    let (mut is, mut fs) = (0i64, 0f64);
    match c.kind {
        Kind::Int => {
            let mut v = [0i64; N];
            for (i, z) in c.zones.iter().enumerate() {
                c.decode_i64(i, &mut v);
                for &x in &v[..z.count as usize] {
                    is = is.wrapping_add(x);
                }
            }
        }
        Kind::Double => {
            let mut f = [0f64; N];
            for (i, z) in c.zones.iter().enumerate() {
                c.decode_f64(i, &mut f);
                for &x in &f[..z.count as usize] {
                    fs += x;
                }
            }
        }
    }
    (is, fs)
}

fn col_minmax(c: &Column) -> (i64, i64) {
    c.zones.iter().fold((i64::MAX, i64::MIN), |(a, b), z| {
        (a.min(z.min), b.max(z.max))
    })
}

fn col_filter(ts: &Column, lo: i64, hi: i64, words: &mut [u64]) {
    words.fill(0);
    let mut v = [0i64; N];
    for (i, z) in ts.zones.iter().enumerate() {
        if z.max < lo || z.min > hi {
            continue;
        }
        let wi = i * N / 64;
        let len = z.count as usize;
        if z.min >= lo && z.max <= hi {
            set_all(&mut words[wi..], len);
            continue;
        }
        ts.decode_i64(i, &mut v);
        let out = &mut words[wi..wi + len.div_ceil(64)];
        for k in 0..out.len() {
            let mut word = 0u64;
            let end = (len - k * 64).min(64);
            for b in 0..end {
                let x = v[k * 64 + b];
                word |= ((x >= lo && x <= hi) as u64) << b;
            }
            out[k] = word;
        }
    }
}

/// Lower and upper bound of the filter's count from zones alone.
fn col_estimate(ts: &Column, lo: i64, hi: i64) -> (u64, u64) {
    let (mut a, mut b) = (0u64, 0u64);
    for z in &ts.zones {
        if z.max < lo || z.min > hi {
            continue;
        }
        b += z.count as u64;
        if z.min >= lo && z.max <= hi {
            a += z.count as u64;
        }
    }
    (a, b)
}

fn col_filtered_sum(agg: &Column, words: &[u64]) -> (u64, i64, f64) {
    let (mut cnt, mut is, mut fs) = (0u64, 0i64, 0f64);
    let mut v = [0i64; N];
    let mut f = [0f64; N];
    for (i, z) in agg.zones.iter().enumerate() {
        let len = z.count as usize;
        let ws = &words[i * N / 64..i * N / 64 + len.div_ceil(64)];
        if ws.iter().all(|&x| x == 0) {
            continue;
        }
        let full = ws.iter().map(|w| w.count_ones() as usize).sum::<usize>() == len;
        match agg.kind {
            Kind::Int => {
                if full {
                    cnt += len as u64;
                    is = is.wrapping_add(z.isum as i64);
                    continue;
                }
                agg.decode_i64(i, &mut v);
                for (k, &word) in ws.iter().enumerate() {
                    let mut bits = word;
                    while bits != 0 {
                        let j = k * 64 + bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        cnt += 1;
                        is = is.wrapping_add(v[j]);
                    }
                }
            }
            Kind::Double => {
                agg.decode_f64(i, &mut f);
                if full {
                    cnt += len as u64;
                    for &x in &f[..len] {
                        fs += x;
                    }
                    continue;
                }
                for (k, &word) in ws.iter().enumerate() {
                    let mut bits = word;
                    while bits != 0 {
                        let j = k * 64 + bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        cnt += 1;
                        fs += f[j];
                    }
                }
            }
        }
    }
    (cnt, is, fs)
}

fn col_group(ord: &Column, val: &Column, words: &[u64], card: usize) -> (Vec<u64>, Vec<f64>) {
    let (mut counts, mut sums) = (vec![0u64; card], vec![0f64; card]);
    let mut o = [0i64; N];
    let mut f = [0f64; N];
    for (i, z) in ord.zones.iter().enumerate() {
        let len = z.count as usize;
        let ws = &words[i * N / 64..i * N / 64 + len.div_ceil(64)];
        if ws.iter().all(|&x| x == 0) {
            continue;
        }
        ord.decode_i64(i, &mut o);
        val.decode_f64(i, &mut f);
        for (k, &word) in ws.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let j = k * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let g = o[j] as usize;
                counts[g] += 1;
                sums[g] += f[j];
            }
        }
    }
    (counts, sums)
}

// ---------- driver

struct Fmt {
    l90: DvCol,
    a90: DvCol,
    col: Column,
    skip: Vec<(i64, i64)>,
}

fn build(values: &[i64], kind: Kind) -> Fmt {
    Fmt {
        l90: write_dv(values, Lucene90DocValuesFormat::default()),
        a90: write_dv(values, Lucene90DocValuesFormat::rust_adaptive()),
        col: Column::encode(kind, values),
        skip: skip_index(values),
    }
}

fn check_roundtrip(name: &str, values: &[i64], f: &Fmt) {
    let mut v = [0i64; N];
    for (i, chunk) in values.chunks(N).enumerate() {
        f.col.decode_i64(i, &mut v);
        assert_eq!(&v[..chunk.len()], chunk, "{name}: vector {i} differs");
    }
}

fn mb(b: usize) -> f64 {
    b as f64 / (1 << 20) as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let reps: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(7);

    let datasets: &[(&str, &[(&str, Kind)])] = &[
        (
            "taxi",
            &[
                ("ts", Kind::Int),
                ("fare", Kind::Double),
                ("tip", Kind::Double),
                ("distance", Kind::Double),
                ("total", Kind::Double),
                ("passengers", Kind::Int),
                ("pu_location_kw", Kind::Int),
            ],
        ),
        (
            "taxisorted",
            &[
                ("ts", Kind::Int),
                ("fare", Kind::Double),
                ("tip", Kind::Double),
                ("distance", Kind::Double),
                ("total", Kind::Double),
                ("passengers", Kind::Int),
                ("pu_location_kw", Kind::Int),
            ],
        ),
        (
            "bgl",
            &[
                ("ts", Kind::Int),
                ("host", Kind::Int),
                ("level", Kind::Int),
                ("component", Kind::Int),
                ("bytes", Kind::Int),
            ],
        ),
    ];

    println!("# Columns100 spike, reps={reps} (median ms, single thread)\n");
    for (ds, cols) in datasets {
        let mut built = Vec::new();
        for (c, kind) in cols.iter() {
            let values = load(dir, &format!("{ds}_{c}"));
            let f = build(&values, *kind);
            check_roundtrip(&format!("{ds}_{c}"), &values, &f);
            built.push((*c, *kind, values, f));
        }
        let n = built[0].2.len();
        println!("## {ds} ({n} docs)\n");
        println!("### Size (MiB; bytes/value) and full-column scan (ms)\n");
        println!("| column | Lucene90 | Adaptive90 | Columns100 | Columns100 16K-paged | codec mix | scan L90 | scan A90 | scan C100 decode | sum C100 zones | min/max L90 | min/max C100 zones |");
        println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
        for (c, kind, _values, f) in &built {
            let l = f.l90.data.len() + f.l90.dvm_len;
            let a = f.a90.data.len() + f.a90.dvm_len;
            let (cf, cp) = f.col.size();
            let bpv = |b: usize| b as f64 / n as f64;
            let (t_l, r_l) = time(reps, || dv_sum(&f.l90, n, *kind));
            let (t_a, r_a) = time(reps, || dv_sum(&f.a90, n, *kind));
            let (t_c, r_c) = time(reps, || col_decode_sum(&f.col));
            let (t_z, r_z) = time(reps, || col_sum(&f.col));
            assert_eq!(r_l.0, r_a.0);
            assert_eq!(r_l.1.to_bits(), r_a.1.to_bits(), "{c} a90 sum");
            assert_eq!(r_l.0, r_c.0);
            assert_eq!(r_l.1.to_bits(), r_c.1.to_bits(), "{c} col sum");
            assert_eq!(r_l.0, r_z.0);
            let (t_ml, m_l) = time(reps, || dv_minmax(&f.l90, n));
            let (t_mc, m_c) = time(reps, || col_minmax(&f.col));
            assert_eq!(m_l, m_c);
            println!(
                "| {c} | {:.1} ({:.2}) | {:.1} ({:.2}) | {:.1} ({:.2}) | {:.1} ({:.2}) | {} | {:.1} | {:.1} | {:.1} | {} | {:.1} | {:.3} |",
                mb(l), bpv(l), mb(a), bpv(a), mb(cf), bpv(cf), mb(cp), bpv(cp),
                f.col.codec_mix(), t_l, t_a, t_c,
                if *kind == Kind::Int { format!("{t_z:.3}") } else { "n/a (per value)".into() },
                t_ml, t_mc
            );
        }
        // Filter + aggregation on the timestamp column.
        let ts_idx = 0;
        let mut sorted_ts = built[ts_idx].2.clone();
        sorted_ts.sort_unstable();
        let agg_name = if *ds == "bgl" { "bytes" } else { "fare" };
        let agg = built.iter().find(|b| b.0 == agg_name).unwrap();
        let (ord_name, card) = if *ds == "bgl" {
            ("host", 69227)
        } else {
            ("pu_location_kw", 261)
        };
        let ord = built.iter().find(|b| b.0 == ord_name).unwrap();
        let ts = &built[ts_idx].3;
        println!("\n### Range filter on ts + sum({agg_name}), terms({ord_name}) count+sum, prefilter bitset (ms)\n");
        println!("| selectivity | matched | bitset L90 scan | bitset L90+skip | bitset A90+skip | bitset C100 | estimate C100 (bounds) | sum L90+skip | sum A90+skip | sum C100 | terms L90+skip | terms A90+skip | terms C100 |");
        println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
        let words_len = n.div_ceil(64) + 16;
        for sel in [0.001, 0.01, 0.1, 0.5] {
            let mid = n / 3;
            let half = ((n as f64 * sel) / 2.0) as usize;
            let lo = sorted_ts[mid.saturating_sub(half)];
            let hi = sorted_ts[(mid + half).min(n - 1)];
            let mut w1 = vec![0u64; words_len];
            let mut w2 = vec![0u64; words_len];
            let mut w3 = vec![0u64; words_len];
            let mut w4 = vec![0u64; words_len];
            let (tb0, _) = time(reps, || dv_filter(&ts.l90, n, lo, hi, None, &mut w1));
            let (tb1, _) = time(reps, || {
                dv_filter(&ts.l90, n, lo, hi, Some(&ts.skip), &mut w2)
            });
            let (tb2, _) = time(reps, || {
                dv_filter(&ts.a90, n, lo, hi, Some(&ts.skip), &mut w3)
            });
            let (tb3, _) = time(reps, || col_filter(&ts.col, lo, hi, &mut w4));
            assert_eq!(w1, w2);
            assert_eq!(w1, w3);
            assert_eq!(w1, w4);
            let matched: u64 = w1.iter().map(|w| w.count_ones() as u64).sum();
            let (te, (ea, eb)) = time(reps, || col_estimate(&ts.col, lo, hi));
            assert!(ea <= matched && matched <= eb);
            let k = agg.1;
            let (ts1, s1) = time(reps, || dv_filtered_sum(&agg.3.l90, n, k, &w1));
            let (ts2, s2) = time(reps, || dv_filtered_sum(&agg.3.a90, n, k, &w1));
            let (ts3, s3) = time(reps, || col_filtered_sum(&agg.3.col, &w1));
            assert_eq!((s1.0, s1.1, s1.2.to_bits()), (s2.0, s2.1, s2.2.to_bits()));
            assert_eq!((s1.0, s1.1, s1.2.to_bits()), (s3.0, s3.1, s3.2.to_bits()));
            let val = if *ds == "bgl" { agg } else { agg };
            let (tg1, g1) = time(reps, || dv_group(&ord.3.l90, &val.3.l90, n, &w1, card));
            let (tg2, g2) = time(reps, || dv_group(&ord.3.a90, &val.3.a90, n, &w1, card));
            let (tg3, g3) = time(reps, || col_group(&ord.3.col, &val.3.col, &w1, card));
            assert_eq!(g1.0, g2.0);
            assert_eq!(g1.0, g3.0);
            for i in 0..card {
                assert_eq!(g1.1[i].to_bits(), g3.1[i].to_bits());
            }
            println!(
                "| {:.1}% | {matched} | {tb0:.2} | {tb1:.2} | {tb2:.2} | {tb3:.2} | {te:.3} ({ea}..{eb}) | {ts1:.2} | {ts2:.2} | {ts3:.2} | {tg1:.2} | {tg2:.2} | {tg3:.2} |",
                sel * 100.0
            );
        }
        // No-filter terms over the whole column.
        let all = {
            let mut w = vec![0u64; words_len];
            set_all(&mut w, n);
            w
        };
        let (tg1, _) = time(reps, || dv_group(&ord.3.l90, &agg.3.l90, n, &all, card));
        let (tg2, _) = time(reps, || dv_group(&ord.3.a90, &agg.3.a90, n, &all, card));
        let (tg3, _) = time(reps, || col_group(&ord.3.col, &agg.3.col, &all, card));
        println!("\nUnfiltered terms({ord_name}) + sum({agg_name}): L90 {tg1:.1} ms, A90 {tg2:.1} ms, C100 {tg3:.1} ms\n");
    }
}
