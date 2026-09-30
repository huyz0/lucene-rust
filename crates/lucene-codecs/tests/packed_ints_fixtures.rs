//! Differential test for the packed-ints family against Lucene 10.5.0: replays
//! `fixtures/data/packed_ints/ops.txt` (written by
//! `fixtures/src/GenPackedInts.java`) through the Rust port and asserts every
//! result Lucene recorded -- the in-memory arrays' backing words, growth
//! widths and values, and every serialized form's exact bytes.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::block_packed::{encode_all_with_block_size, BlockPackedReaderIterator};
use lucene_codecs::monotonic_block_packed::{
    MonotonicBlockPackedReader, MonotonicBlockPackedWriter,
};
use lucene_codecs::packed_data::{PackedDataInput, PackedDataOutput};
use lucene_codecs::packed_ints::{
    DirectPacked64SingleBlockReader, PackedReaderIterator, PackedWriter,
};
use lucene_store::data_input::SliceInput;
use lucene_util::packed::packed_long_values::Kind;
use lucene_util::packed::{
    fastest_format_and_bits, get_mutable_with_format, Format, GrowableWriter, Mutable,
    PackedLongValues, PagedGrowableWriter, PagedMutable, Reader, VERSION_CURRENT,
};

fn ops() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/packed_ints/ops.txt"
    ))
    .expect("run scripts/gen-fixtures.sh --only GenPackedInts")
}

fn nums<T: std::str::FromStr>(parts: &[&str]) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    if parts == ["-"] {
        return Vec::new();
    }
    parts.iter().map(|p| p.parse().unwrap()).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    if s == "-" {
        return Vec::new();
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// Splits the script into records: a header line and the lines up to `end`.
fn records(text: &str) -> Vec<Vec<Vec<&str>>> {
    let mut out = Vec::new();
    let mut cur: Vec<Vec<&str>> = Vec::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split(' ').collect();
        if parts[0] == "fastest" {
            out.push(vec![parts]);
            continue;
        }
        if parts[0] == "end" {
            out.push(std::mem::take(&mut cur));
        } else {
            cur.push(parts);
        }
    }
    assert!(cur.is_empty());
    out
}

#[derive(Default)]
struct Counts {
    fastest: usize,
    mutable: usize,
    growable: usize,
    paged: usize,
    longvalues: usize,
    writer: usize,
    direct: usize,
    packeddata: usize,
    monotonic: usize,
    blockpacked: usize,
}

#[test]
fn packed_ints_replay_matches_lucene() {
    let text = ops();
    let mut c = Counts::default();
    for rec in records(&text) {
        let head = &rec[0];
        match head[0] {
            "fastest" => {
                let ratio = f32::from_bits(head[1].parse::<i32>().unwrap() as u32);
                let want: Vec<u32> = nums(&head[2..]);
                for bpv in 1..=64u32 {
                    assert_eq!(
                        fastest_format_and_bits(Some(100), bpv, ratio).bits_per_value,
                        want[bpv as usize - 1],
                        "fastest ratio={ratio} bpv={bpv}"
                    );
                }
                c.fastest += 1;
            }
            "mutable" => {
                replay_mutable(&rec);
                c.mutable += 1;
            }
            "growable" => {
                replay_growable(&rec);
                c.growable += 1;
            }
            "paged" | "pagedgrowable" => {
                replay_paged(&rec);
                c.paged += 1;
            }
            "longvalues" => {
                replay_long_values(&rec);
                c.longvalues += 1;
            }
            "writer" => {
                replay_writer(&rec);
                c.writer += 1;
            }
            "directsingleblock" => {
                let bpv: u32 = head[1].parse().unwrap();
                let n: usize = head[2].parse().unwrap();
                let bytes = unhex(rec[1][1]);
                let want: Vec<i64> = nums(&rec[2][1..]);
                let r = DirectPacked64SingleBlockReader::new(bpv, n, &bytes[1..]).unwrap();
                assert_eq!(r.size(), n);
                for (i, &w) in want.iter().enumerate() {
                    assert_eq!(r.get(i).unwrap(), w, "direct single block bpv={bpv} i={i}");
                }
                c.direct += 1;
            }
            "packeddata" => {
                replay_packed_data(&rec);
                c.packeddata += 1;
            }
            "monotonic" => {
                let block_size: usize = head[1].parse().unwrap();
                let vals: Vec<i64> = nums(&rec[1][1..]);
                let want = unhex(rec[2][1]);
                let mut buf = Vec::new();
                let mut w = MonotonicBlockPackedWriter::new(&mut buf, block_size);
                for &v in &vals {
                    w.add(v);
                }
                w.finish();
                assert_eq!(
                    buf,
                    want,
                    "monotonic bytes block_size={block_size} n={}",
                    vals.len()
                );
                let r = MonotonicBlockPackedReader::of(
                    &mut SliceInput::new(&want),
                    VERSION_CURRENT,
                    block_size,
                    vals.len() as u64,
                )
                .unwrap();
                for (i, &v) in vals.iter().enumerate() {
                    assert_eq!(r.get(i as u64), v);
                }
                c.monotonic += 1;
            }
            "blockpacked" => {
                let block_size: usize = head[1].parse().unwrap();
                let vals: Vec<i64> = nums(&rec[1][1..]);
                let want = unhex(rec[2][1]);
                assert_eq!(
                    encode_all_with_block_size(&vals, block_size),
                    want,
                    "block packed bytes block_size={block_size}"
                );
                let mut input = SliceInput::new(&want);
                let mut it = BlockPackedReaderIterator::new(
                    &mut input,
                    VERSION_CURRENT,
                    block_size,
                    vals.len() as u64,
                )
                .unwrap();
                for &v in &vals {
                    assert_eq!(it.next_value().unwrap(), v);
                }
                c.blockpacked += 1;
            }
            other => panic!("unknown record {other}"),
        }
    }
    // The generator's shape: every supported (format, width) pair once.
    assert_eq!(c.fastest, 9);
    assert_eq!(c.mutable, 64 + 14);
    assert_eq!(c.writer, 64 + 14);
    assert_eq!(c.growable, 12);
    assert_eq!(c.paged, 12);
    assert_eq!(c.longvalues, 18);
    assert_eq!(c.direct, 14);
    assert_eq!(c.packeddata, 6);
    assert_eq!(c.monotonic, 15);
    assert_eq!(c.blockpacked, 12);
}

fn replay_mutable(rec: &[Vec<&str>]) {
    let format = Format::by_id(rec[0][1].parse().unwrap()).unwrap();
    let bpv: u32 = rec[0][2].parse().unwrap();
    let n: usize = rec[0][3].parse().unwrap();
    let mut m = get_mutable_with_format(n, bpv, format);
    let ctx = format!("{format:?} bpv={bpv}");
    for op in &rec[1..] {
        match op[0] {
            "set" => m.set(op[1].parse().unwrap(), op[2].parse().unwrap()),
            "bset" => {
                let i: usize = op[1].parse().unwrap();
                let got: usize = op[2].parse().unwrap();
                let arr: Vec<i64> = nums(&op[3..]);
                assert_eq!(m.set_bulk(i, &arr), got, "{ctx} bset");
            }
            "fill" => m.fill(
                op[1].parse().unwrap(),
                op[2].parse().unwrap(),
                op[3].parse().unwrap(),
            ),
            "bget" => {
                let i: usize = op[1].parse().unwrap();
                let len: usize = op[2].parse().unwrap();
                let got: usize = op[3].parse().unwrap();
                let want: Vec<i64> = nums(&op[4..]);
                let mut arr = vec![0i64; len];
                assert_eq!(m.get_bulk(i, &mut arr), got, "{ctx} bget count");
                assert_eq!(&arr[..got], &want[..], "{ctx} bget values");
            }
            "blocks" => {
                let want: Vec<i64> = nums(&op[1..]);
                let have: Vec<i64> = m.blocks().iter().map(|&w| w as i64).collect();
                assert_eq!(have, want, "{ctx} backing words");
            }
            "values" => {
                let want: Vec<i64> = nums(&op[1..]);
                for (i, &w) in want.iter().enumerate() {
                    assert_eq!(m.get(i), w, "{ctx} value {i}");
                }
            }
            other => panic!("unknown op {other}"),
        }
    }
}

fn replay_growable(rec: &[Vec<&str>]) {
    let start: u32 = rec[0][1].parse().unwrap();
    let n: usize = rec[0][2].parse().unwrap();
    let ratio = f32::from_bits(rec[0][3].parse::<i32>().unwrap() as u32);
    let mut w = GrowableWriter::new(start, n, ratio);
    assert_eq!(w.bits_per_value(), rec[0][4].parse::<u32>().unwrap());
    for op in &rec[1..] {
        match op[0] {
            "set" => {
                w.set(op[1].parse().unwrap(), op[2].parse().unwrap());
                assert_eq!(
                    w.bits_per_value(),
                    op[3].parse::<u32>().unwrap(),
                    "growable set"
                );
            }
            "fill" => {
                w.fill(
                    op[1].parse().unwrap(),
                    op[2].parse().unwrap(),
                    op[3].parse().unwrap(),
                );
                assert_eq!(
                    w.bits_per_value(),
                    op[4].parse::<u32>().unwrap(),
                    "growable fill"
                );
            }
            "resize" => {
                w = w.resize(op[1].parse().unwrap());
                assert_eq!(
                    w.bits_per_value(),
                    op[2].parse::<u32>().unwrap(),
                    "growable resize"
                );
            }
            "values" => {
                let want: Vec<i64> = nums(&op[1..]);
                assert_eq!(w.size(), want.len());
                for (i, &v) in want.iter().enumerate() {
                    assert_eq!(w.get(i), v, "growable value {i}");
                }
            }
            other => panic!("unknown op {other}"),
        }
    }
}

enum Paged {
    Fixed(PagedMutable),
    Growable(PagedGrowableWriter),
}

fn replay_paged(rec: &[Vec<&str>]) {
    let size: u64 = rec[0][1].parse().unwrap();
    let page: usize = rec[0][2].parse().unwrap();
    let bpv: u32 = rec[0][3].parse().unwrap();
    let ratio = f32::from_bits(rec[0][4].parse::<i32>().unwrap() as u32);
    let mut p = if rec[0][0] == "paged" {
        Paged::Fixed(PagedMutable::new(size, page, bpv, ratio).unwrap())
    } else {
        Paged::Growable(PagedGrowableWriter::new(size, page, bpv, ratio).unwrap())
    };
    for op in &rec[1..] {
        match op[0] {
            "set" => {
                let (i, v) = (op[1].parse().unwrap(), op[2].parse().unwrap());
                match &mut p {
                    Paged::Fixed(m) => m.set(i, v),
                    Paged::Growable(m) => m.set(i, v),
                }
            }
            "resize" => {
                let n = op[1].parse().unwrap();
                p = match p {
                    Paged::Fixed(m) => Paged::Fixed(m.resize(n).unwrap()),
                    Paged::Growable(m) => Paged::Growable(m.resize(n).unwrap()),
                };
            }
            "grow" => {
                let min = op[1].parse().unwrap();
                let want: u64 = op[2].parse().unwrap();
                p = match p {
                    Paged::Fixed(m) => Paged::Fixed(m.grow(min).unwrap()),
                    Paged::Growable(m) => Paged::Growable(m.grow(min).unwrap()),
                };
                let size = match &p {
                    Paged::Fixed(m) => m.size(),
                    Paged::Growable(m) => m.size(),
                };
                assert_eq!(size, want, "grow size");
            }
            "pagebits" => {
                let want: Vec<u32> = nums(&op[1..]);
                let have: Vec<u32> = match &p {
                    Paged::Fixed(m) => m.pages().iter().map(|x| x.bits_per_value()).collect(),
                    Paged::Growable(m) => m.pages().iter().map(|x| x.bits_per_value()).collect(),
                };
                assert_eq!(have, want, "paged page widths");
            }
            "values" => {
                let want: Vec<i64> = nums(&op[1..]);
                for (i, &v) in want.iter().enumerate() {
                    let got = match &p {
                        Paged::Fixed(m) => m.get(i as u64),
                        Paged::Growable(m) => m.get(i as u64),
                    };
                    assert_eq!(got, v, "paged value {i}");
                }
            }
            other => panic!("unknown op {other}"),
        }
    }
}

fn replay_long_values(rec: &[Vec<&str>]) {
    let kind = rec[0][1];
    let page: usize = rec[0][2].parse().unwrap();
    let ratio = f32::from_bits(rec[0][3].parse::<i32>().unwrap() as u32);
    let mut b = match kind {
        "packed" => PackedLongValues::packed_builder(page, ratio),
        "delta" => PackedLongValues::delta_packed_builder(page, ratio),
        _ => PackedLongValues::monotonic_builder(page, ratio),
    }
    .unwrap();
    let vals: Vec<i64> = nums(&rec[1][1..]);
    b.add_all(&vals);
    let built = b.build();
    let ctx = format!("{kind} page={page} ratio={ratio}");
    for op in &rec[2..] {
        match op[0] {
            "pagebits" => {
                let want: Vec<u32> = nums(&op[1..]);
                let have: Vec<u32> = built.pages().iter().map(|p| p.bits_per_value()).collect();
                assert_eq!(have, want, "{ctx} page widths");
            }
            "mins" => assert_eq!(built.mins(), &nums::<i64>(&op[1..])[..], "{ctx} mins"),
            "avgbits" => {
                let want: Vec<u32> = nums::<i32>(&op[1..])
                    .into_iter()
                    .map(|b| b as u32)
                    .collect();
                let have: Vec<u32> = built.averages().iter().map(|a| a.to_bits()).collect();
                assert_eq!(have, want, "{ctx} averages");
            }
            other => panic!("unknown op {other}"),
        }
    }
    match kind {
        "packed" => assert_eq!(built.kind(), Kind::Packed),
        "delta" => assert_eq!(built.kind(), Kind::DeltaPacked),
        _ => assert_eq!(built.kind(), Kind::Monotonic),
    }
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(built.get(i as u64), v, "{ctx} get {i}");
    }
    assert_eq!(built.iter().collect::<Vec<_>>(), vals, "{ctx} iterator");
}

fn replay_writer(rec: &[Vec<&str>]) {
    let format = Format::by_id(rec[0][1].parse().unwrap()).unwrap();
    let bpv: u32 = rec[0][2].parse().unwrap();
    let n: usize = rec[0][3].parse().unwrap();
    let mem: usize = rec[0][4].parse().unwrap();
    let vals: Vec<i64> = nums(&rec[1][1..]);
    let want = unhex(rec[2][1]);
    let mut buf = Vec::new();
    {
        let mut w = PackedWriter::new(&mut buf, format, Some(n), bpv, mem);
        for &v in &vals {
            w.add(v).unwrap();
        }
        w.finish().unwrap();
    }
    assert_eq!(buf, want, "{format:?} bpv={bpv} n={n} mem={mem} bytes");
    // Read back with every buffer size.
    for mem in [0usize, 100, 4096] {
        let mut input = SliceInput::new(&want);
        let mut it =
            PackedReaderIterator::new(&mut input, format, VERSION_CURRENT, n, bpv, mem).unwrap();
        for i in 0..n {
            let expect = vals.get(i).copied().unwrap_or(0);
            assert_eq!(
                it.next_value().unwrap(),
                expect,
                "{format:?} bpv={bpv} iter {i}"
            );
        }
        assert!(n == 0 || it.next_value().is_err());
    }
}

fn replay_packed_data(rec: &[Vec<&str>]) {
    let items: Vec<(i64, u32, bool)> = rec[1][1..]
        .iter()
        .map(|t| {
            let mut f = t.split(':');
            (
                f.next().unwrap().parse().unwrap(),
                f.next().unwrap().parse().unwrap(),
                f.next().unwrap() == "1",
            )
        })
        .collect();
    let want = unhex(rec[2][1]);
    let mut buf = Vec::new();
    {
        let mut out = PackedDataOutput::new(&mut buf);
        for &(v, b, flush) in &items {
            out.write_long(v, b);
            if flush {
                out.flush();
            }
        }
        out.flush();
    }
    assert_eq!(buf, want, "packed data bytes");
    let mut input = SliceInput::new(&want);
    let mut pin = PackedDataInput::new(&mut input);
    for &(v, b, flush) in &items {
        assert_eq!(pin.read_long(b).unwrap(), v);
        if flush {
            pin.skip_to_next_byte();
        }
    }
}

/// `Packed64` vs `Packed64SingleBlock` copies through `PackedInts.copy`
/// produce the same values: a check the fixture's per-structure records do
/// not make on their own.
#[test]
fn copy_across_formats_preserves_fixture_values() {
    let text = ops();
    let rec = records(&text)
        .into_iter()
        .find(|r| r[0][0] == "mutable" && r[0][1] == "0" && r[0][2] == "21")
        .unwrap();
    let want: Vec<i64> = nums(&rec.last().unwrap()[1..]);
    let mut src = get_mutable_with_format(want.len(), 21, Format::Packed);
    for (i, &v) in want.iter().enumerate() {
        src.set(i, v);
    }
    let mut dst = get_mutable_with_format(want.len(), 21, Format::PackedSingleBlock);
    lucene_util::packed::copy(&src, 0, &mut dst, 0, want.len(), 64);
    for (i, &v) in want.iter().enumerate() {
        assert_eq!(dst.get(i), v);
    }
}
