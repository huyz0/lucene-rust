//! Where a scan's time goes: decoding the doc values alone, against the whole
//! single-thread scan, for each column of the `arrow_gen` index.
//!
//! `cargo run --release -p lucene-sql --example arrow_decode -- <dir>`

use std::sync::Arc;
use std::time::Instant;

use lucene_codecs::doc_values::NumericReader;
use lucene_search::directory_reader::DirectoryReader;
use lucene_sql::export::{Scan, ScanOptions};
use lucene_sql::{Snapshot, TableSpec};
use lucene_store::FsDirectory;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let spec = TableSpec::new("logs")
        .key("id")
        .long("ts")
        .long("status")
        .long("bytes")
        .long("latency")
        .keyword("host")
        .keyword("level")
        .keyword("region");
    let reader = DirectoryReader::open(&FsDirectory::open(path)).unwrap();
    let snap = Arc::new(Snapshot::new(reader, spec).unwrap());
    let rows = snap.max_doc() as f64;
    for col in ["ts", "status", "bytes", "latency", "host", "level"] {
        let mut best_decode = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            let mut total = 0i64;
            for (s, leaf) in snap.reader().segment_readers().iter().enumerate() {
                let n = leaf.field_infos().field_by_name(col).unwrap().number;
                let (meta, data) = leaf.doc_values_for_field(n).unwrap();
                let kind_numeric = meta.numeric_entry(n);
                let mut vals = vec![0i64; 65536];
                let mut present = vec![0u64; 1024];
                let max = leaf.max_doc as usize;
                let mut start = 0;
                if let Some(e) = kind_numeric {
                    let mut r = NumericReader::new(data, e);
                    while start < max {
                        let w = 65536.min(max - start);
                        r.fill_window(start as i32, &mut vals[..w], &mut present)
                            .unwrap();
                        total += vals[w - 1];
                        start += w;
                    }
                } else if let Some((lucene_search::global_ords::Ords::Single(mut r), _, _)) =
                    lucene_search::global_ords::keyword_column(leaf, col).unwrap()
                {
                    while start < max {
                        let w = 65536.min(max - start);
                        r.fill_window(start as i32, &mut vals[..w], &mut present)
                            .unwrap();
                        total += vals[w - 1];
                        start += w;
                    }
                }
                let _ = s;
            }
            std::hint::black_box(total);
            best_decode = best_decode.min(t.elapsed().as_secs_f64());
        }
        let mut best_scan = f64::MAX;
        let o = ScanOptions {
            columns: Some(vec![col.to_string()]),
            threads: 1,
            ..ScanOptions::default()
        };
        let scan = Scan::new(snap.clone(), &o).unwrap();
        for _ in 0..5 {
            let t = Instant::now();
            let n: usize = scan.open().map(|b| b.unwrap().num_rows()).sum();
            assert_eq!(n as f64, rows);
            best_scan = best_scan.min(t.elapsed().as_secs_f64());
        }
        println!(
            "{col:<8} decode {:6.1} ms ({:4.2} ns/row)   scan(1 thread) {:6.1} ms ({:4.2} ns/row)",
            best_decode * 1e3,
            best_decode * 1e9 / rows,
            best_scan * 1e3,
            best_scan * 1e9 / rows
        );
    }
    for cols in [
        vec!["bytes", "latency"],
        vec!["host", "level"],
        vec![
            "ts", "status", "bytes", "latency", "host", "level", "region",
        ],
    ] {
        for threads in [1, 2, 4, 8] {
            let o = ScanOptions {
                columns: Some(cols.iter().map(|c| c.to_string()).collect()),
                threads,
                ..ScanOptions::default()
            };
            let scan = Scan::new(snap.clone(), &o).unwrap();
            let mut best = f64::MAX;
            for _ in 0..7 {
                let t = Instant::now();
                let n: usize = scan.open().map(|b| b.unwrap().num_rows()).sum();
                assert_eq!(n as f64, rows);
                best = best.min(t.elapsed().as_secs_f64());
            }
            println!(
                "{} cols, {threads} threads: {:6.1} ms",
                cols.len(),
                best * 1e3
            );
        }
    }
}
