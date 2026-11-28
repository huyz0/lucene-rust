//! Builds the index the Arrow source benchmark scans.
//!
//! `cargo run --release -p lucene-sql --example arrow_gen -- <dir> [docs] [segments] [adaptive]`
//! writes `docs` log-like documents (default 10M) in `segments` commits:
//! `id` (stored key), `ts`, `status`, `bytes`, `latency` (long doc values) and
//! `host`, `level`, `region` (keyword doc values). A fourth argument `adaptive`
//! writes the doc values in the adaptive format, with zone statistics.

use std::time::Instant;

use lucene_index::document::{Document, KeywordField, NumericDocValuesField, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

fn next(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = &a[1];
    let docs: usize = a.get(2).map_or(10_000_000, |s| s.parse().unwrap());
    let segments: usize = a.get(3).map_or(8, |s| s.parse().unwrap());
    std::fs::create_dir_all(path).unwrap();
    let dir = FsDirectory::open(path.clone());
    let v = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", v).unwrap();
    w.set_ram_buffer_size_mb(1024.0).unwrap();
    if a.get(4).is_some_and(|m| m == "adaptive") {
        w.set_default_doc_values_format(
            lucene_codecs::per_field_doc_values::Lucene90DocValuesFormat::rust_adaptive(),
        );
    }
    let mut st = 0x9e37_79b9_7f4a_7c15u64;
    let per = docs.div_ceil(segments);
    let levels = ["debug", "info", "info", "info", "warn", "error"];
    let t = Instant::now();
    let mut ts = 1_700_000_000_000i64;
    for i in 0..docs {
        ts += (next(&mut st) % 40) as i64;
        let mut d = Document::new();
        d.add(StringField::new("id", format!("doc-{i:09}"), Store::Yes));
        d.add(NumericDocValuesField::new("ts", ts));
        let status = [200, 200, 200, 200, 201, 204, 301, 404, 500][(next(&mut st) % 9) as usize];
        d.add(NumericDocValuesField::new("status", status));
        d.add(NumericDocValuesField::new(
            "bytes",
            (next(&mut st) % 100_000) as i64,
        ));
        d.add(NumericDocValuesField::new(
            "latency",
            ((next(&mut st) % 2000) * (next(&mut st) % 5 + 1)) as i64,
        ));
        d.add(KeywordField::new(
            "host",
            format!("host-{:04}", next(&mut st) % 1000),
            Store::No,
        ));
        d.add(KeywordField::new(
            "level",
            levels[(next(&mut st) % 6) as usize].to_string(),
            Store::No,
        ));
        d.add(KeywordField::new(
            "region",
            format!("region-{:02}", next(&mut st) % 20),
            Store::No,
        ));
        w.add_fields_document(&d).unwrap();
        if (i + 1) % per == 0 {
            w.commit().unwrap();
        }
    }
    w.commit().unwrap();
    eprintln!("built {docs} docs in {:.0?}", t.elapsed());
}
