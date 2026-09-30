//! Differential test for `OfflineSorter` against Lucene 10.5.0: every case in
//! `fixtures/data/offline_sorter/cases.txt` (written by
//! `fixtures/src/GenOfflineSorter.java`) regenerates the same input from the
//! shared formula, sorts it in a fresh directory, and must produce Lucene's
//! result file name, temp-file / merge-round / line counts, and a result of
//! Lucene's length and footer checksum (so the same bytes).
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::offline_sorter::{write_byte_sequence, OfflineSorter};
use lucene_store::codec_util;
use lucene_store::directory::{Directory, FsDirectory};
use lucene_store::DataOutput;

fn variable(i: u64) -> Vec<u8> {
    let mut s = format!("{:x}", (i * 2_654_435_761) % 1_000_003);
    s.push_str(&"k".repeat((i % 37) as usize));
    s.into_bytes()
}

fn fixed(i: u64) -> Vec<u8> {
    (((i * 7919) % 100_003) as u32).to_be_bytes().to_vec()
}

#[test]
fn offline_sorter_matches_lucene() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/offline_sorter/cases.txt"
    ))
    .expect("run scripts/gen-fixtures.sh --only GenOfflineSorter");
    let mut cases = 0;
    for line in text.lines() {
        let p: Vec<&str> = line.split(' ').collect();
        let (name, n, is_fixed) = (p[0], p[1].parse::<u64>().unwrap(), p[2] == "true");
        let buffer: u64 = p[3].parse().unwrap();
        let max_temp: usize = p[4].parse().unwrap();
        let path = std::env::temp_dir().join(format!(
            "offline_sorter_fixture_{name}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        let dir = FsDirectory::open(&path);
        // The input, as `ByteSequencesWriter` + `CodecUtil.writeFooter`.
        let mut bytes = Vec::new();
        for i in 0..n {
            let item = if is_fixed { fixed(i) } else { variable(i) };
            write_byte_sequence(&mut bytes, &item).unwrap();
        }
        codec_util::write_footer(&mut bytes);
        let mut out = dir.create_output("in").unwrap();
        out.write_bytes(&bytes);
        out.close().unwrap();

        let mut sorter =
            OfflineSorter::new(&dir, "t", None, buffer, max_temp, is_fixed.then_some(4)).unwrap();
        let result = sorter.sort("in").unwrap();
        let info = sorter.sort_info().clone();
        let sorted = dir.open(&result).unwrap();
        let checksum = u64::from_be_bytes(sorted[sorted.len() - 8..].try_into().unwrap());
        let writes = if info.writes.is_empty() {
            "-".to_string()
        } else {
            info.writes
                .iter()
                .map(|(n, c)| format!("{n}:{c}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let got = format!(
            "{result} {} {} {} {} {checksum} {writes}",
            info.temp_merge_files,
            info.merge_rounds,
            info.line_count,
            sorted.len()
        );
        assert_eq!(got, p[5..].join(" "), "{name}");
        let mut left = dir.list_all().unwrap();
        left.sort();
        assert_eq!(
            left,
            vec!["in".to_string(), result],
            "{name}: temp files left over"
        );
        std::fs::remove_dir_all(&path).unwrap();
        cases += 1;
    }
    assert_eq!(cases, 8);
}
