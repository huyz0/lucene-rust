//! Differential test for `BkdWriter` against Lucene 10.5.0's `BKDWriter`:
//! every case in `fixtures/data/bkd_writer/cases.txt` (written by
//! `fixtures/src/GenBkdWriter.java`) regenerates the same points from the
//! shared LCG, drives the same entry point (the flush path over a
//! `MutablePointTree`, `add` + `finish` in heap and spilling to temp files,
//! or the one-dimensional `merge`), and must write meta, index and data
//! bytes of Lucene's length and CRC32 (and, for small cases, the data bytes
//! themselves).
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::bkd_writer::{BkdConfig, BkdWriter, MutablePointTree, VERSION_CURRENT};
use lucene_store::directory::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;

/// `GenBkdWriter.Points`.
struct Points {
    seed: u64,
    num_dims: usize,
    bpd: usize,
    card: u64,
    multi: u64,
    shuffled: bool,
    doc_stride: i64,
    doc: i64,
}

impl Points {
    fn next(&mut self) -> u64 {
        self.seed = self
            .seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.seed
    }

    fn next_int(&mut self, bound: u64) -> u64 {
        (self.next() >> 16) % bound
    }

    fn point(&mut self, value: &mut [u8]) -> i32 {
        for d in 0..self.num_dims {
            let v = self.next_int(self.card) as u32;
            for b in 0..self.bpd {
                let from_end = self.bpd - 1 - b;
                value[d * self.bpd + b] = if from_end < 4 {
                    (v >> (from_end * 8)) as u8
                } else {
                    (d * 17 + 1) as u8
                };
            }
        }
        let this_doc = self.doc;
        if self.multi == 0 || self.next_int(self.multi) != 0 {
            self.doc += 1;
        }
        if self.shuffled {
            ((this_doc * 7919) % 100_003 * self.doc_stride) as i32
        } else {
            (this_doc * self.doc_stride) as i32
        }
    }
}

fn describe(meta: &[u8], index: &[u8], data: &[u8], with_data: bool) -> String {
    let crc = |b: &[u8]| format!("{}:{:x}", b.len(), crc32fast::hash(b));
    let hex = if with_data && !data.is_empty() {
        data.iter().map(|b| format!("{b:02x}")).collect::<String>()
    } else {
        "-".to_string()
    };
    format!("{} {} {} {hex}", crc(meta), crc(index), crc(data))
}

/// Flushes `docs`/`values` through `writeField`, returning (meta, index, data).
fn flush(
    config: BkdConfig,
    max_doc: usize,
    max_mb: f64,
    docs: &[i32],
    values: &[u8],
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let stride = config.packed_bytes_length();
    let mut tree = MutablePointTree::new(stride);
    for (i, &d) in docs.iter().enumerate() {
        tree.push(&values[i * stride..(i + 1) * stride], d);
    }
    let mut w = BkdWriter::new(
        max_doc,
        None,
        "_0",
        config,
        max_mb,
        docs.len() as u64,
        VERSION_CURRENT,
    )
    .unwrap();
    let (mut meta, mut index, mut data) = (Vec::new(), Vec::new(), Vec::new());
    let plan = w.write_field(&mut data, &mut tree).unwrap().unwrap();
    w.write_index(&mut meta, &mut index, &plan);
    (meta, index, data)
}

#[test]
fn bkd_writer_matches_lucene() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/bkd_writer/cases.txt"
    ))
    .expect("run scripts/gen-fixtures.sh --only GenBkdWriter");
    let mut cases = 0;
    let mut failures = Vec::new();
    for line in text.lines() {
        let p: Vec<&str> = line.split(' ').collect();
        let (name, mode) = (p[0], p[1]);
        let num: Vec<usize> = p[2..7].iter().map(|s| s.parse().unwrap()).collect();
        let (num_dims, num_index_dims, bpd, max_leaf, n) = (num[0], num[1], num[2], num[3], num[4]);
        let seed: u64 = p[7].parse().unwrap();
        let card: u64 = p[8].parse().unwrap();
        let multi: u64 = p[9].parse().unwrap();
        let shuffled = p[10] == "true";
        let doc_stride: i64 = p[11].parse().unwrap();
        let max_mb: f64 = p[12].parse().unwrap();
        let expected_max_doc: usize = p[13].parse().unwrap();
        let expected = p[14..].join(" ");

        let config = BkdConfig::new(num_dims, num_index_dims, bpd, max_leaf).unwrap();
        let stride = config.packed_bytes_length();
        let mut gen = Points {
            seed,
            num_dims,
            bpd,
            card,
            multi,
            shuffled,
            doc_stride,
            doc: 0,
        };
        let mut docs = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n * stride);
        let mut v = vec![0u8; stride];
        let mut max_doc = 0usize;
        for _ in 0..n {
            let d = gen.point(&mut v);
            docs.push(d);
            values.extend_from_slice(&v);
            max_doc = max_doc.max(d as usize + 1);
        }
        assert_eq!(max_doc, expected_max_doc, "{name}: generator drift");

        let (meta, index, data) = match mode {
            "flush" => flush(config, max_doc, max_mb, &docs, &values),
            "add" => {
                let tmp = TempDir::new(&format!("bkd_writer_{name}"));
                let dir = FsDirectory::open(tmp.path());
                let (mut meta, mut index, mut data) = (Vec::new(), Vec::new(), Vec::new());
                {
                    let mut w = BkdWriter::new(
                        max_doc,
                        Some(&dir),
                        "_0",
                        config,
                        max_mb,
                        n as u64,
                        VERSION_CURRENT,
                    )
                    .unwrap();
                    for (i, &d) in docs.iter().enumerate() {
                        w.add(&values[i * stride..(i + 1) * stride], d).unwrap();
                    }
                    let plan = w.finish(&mut data).unwrap().unwrap();
                    w.write_index(&mut meta, &mut index, &plan);
                }
                assert!(
                    dir.list_all().unwrap().is_empty(),
                    "{name}: temp files left"
                );
                (meta, index, data)
            }
            "merge" => {
                // Two segments (each half flushed on its own, with
                // segment-relative docs), merged under doc maps that shift the
                // second segment and delete every 7th doc. A one-dimensional
                // segment's leaves hold its points sorted by (value, doc),
                // which is what `BKDReader`'s merge iteration yields.
                let half = n / 2;
                let mut seg_max = [0usize; 2];
                let mut segs: Vec<Vec<(Vec<u8>, i32)>> = Vec::new();
                for (s, seg_max_s) in seg_max.iter_mut().enumerate() {
                    let (from, to) = if s == 0 { (0, half) } else { (half, n) };
                    let sd: Vec<i32> = docs[from..to].iter().map(|d| d - docs[from]).collect();
                    *seg_max_s = sd.iter().map(|&d| d as usize + 1).max().unwrap();
                    // Flushing the segment exercises writeField as Lucene did;
                    // its leaf order is the sorted order below.
                    flush(
                        config,
                        *seg_max_s,
                        max_mb,
                        &sd,
                        &values[from * stride..to * stride],
                    );
                    let mut pts: Vec<(Vec<u8>, i32)> = sd
                        .iter()
                        .enumerate()
                        .map(|(i, &d)| {
                            (
                                values[(from + i) * stride..(from + i + 1) * stride].to_vec(),
                                d,
                            )
                        })
                        .collect();
                    pts.sort();
                    segs.push(pts);
                }
                let base = seg_max[0] as i32;
                let map0 = |d: i32| if d % 7 == 0 { -1 } else { d };
                let map1 = move |d: i32| if d % 7 == 3 { -1 } else { d + base };
                let s1 = segs.pop().unwrap();
                let s0 = segs.pop().unwrap();
                let sources: Vec<Box<dyn Iterator<Item = (Vec<u8>, i32)>>> = vec![
                    Box::new(
                        s0.into_iter()
                            .map(move |(v, d)| (v, map0(d)))
                            .filter(|(_, d)| *d != -1),
                    ),
                    Box::new(
                        s1.into_iter()
                            .map(move |(v, d)| (v, map1(d)))
                            .filter(|(_, d)| *d != -1),
                    ),
                ];
                let mut w = BkdWriter::new(
                    seg_max[0] + seg_max[1],
                    None,
                    "_m",
                    config,
                    max_mb,
                    n as u64,
                    VERSION_CURRENT,
                )
                .unwrap();
                let (mut meta, mut index, mut data) = (Vec::new(), Vec::new(), Vec::new());
                let plan = w.merge(&mut data, sources).unwrap().unwrap();
                w.write_index(&mut meta, &mut index, &plan);
                (meta, index, data)
            }
            other => panic!("unknown mode {other}"),
        };
        let got = describe(&meta, &index, &data, n <= 1200);
        if got != expected {
            failures.push(format!(
                "{name}:\n  got      {got:.120}\n  expected {expected:.120}"
            ));
        }
        cases += 1;
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(cases, 17);
}
