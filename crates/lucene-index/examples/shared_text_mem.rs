//! Peak resident memory of indexing one huge text field, owned versus shared:
//! `cargo run --release -p lucene-index --example shared_text_mem -- 200 shared`
//! (`owned` for the `String` path, `string` for a `String` shared as it is,
//! with no copy); the text is handed over after the baseline is read, so the
//! copy `shared` makes (`Arc<str>::from(String)`) is inside the measurement. A third argument `store` also stores the
//! field. Linux only: it reads `/proc/self`.
#![allow(missing_docs, clippy::arithmetic_side_effects)]
use lucene_index::document::{Document, Store, TextField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::directory::FsDirectory;
use std::sync::Arc;

fn status(key: &str) -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap();
    s.lines()
        .find(|l| l.starts_with(key))
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse::<u64>()
        .unwrap()
        / 1024
}
fn main() {
    let mb: usize = std::env::args().nth(1).unwrap().parse().unwrap();
    let mode = std::env::args().nth(2).unwrap();
    let store = if std::env::args().nth(3).as_deref() == Some("store") {
        Store::Yes
    } else {
        Store::No
    };
    let mut text = String::with_capacity(mb << 20);
    let mut i = 0u64;
    while text.len() < mb << 20 {
        text.push_str(&format!("w{} ", i % 5000));
        i += 1;
    }
    let tmp = std::env::temp_dir().join(format!("stm-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let dir = FsDirectory::open(&tmp);
    let mut w = IndexWriter::open(
        &dir,
        Vec::new(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    let t = std::time::Instant::now();
    // forget the transient peak of building the text; only handing it over
    // and indexing it are measured
    std::fs::write("/proc/self/clear_refs", "5").unwrap();
    let base = status("VmRSS:");
    let before = status("VmHWM:");
    let mut d = Document::new();
    match mode.as_str() {
        "shared" => {
            let a: Arc<str> = Arc::from(text); // copies the text
            d.add(TextField::new_shared("body", a, store));
        }
        "string" => d.add(TextField::new_shared("body", text, store)),
        _ => d.add(TextField::new("body", text, store)),
    }
    w.add_fields_document(&d).unwrap();
    let after_add = status("VmHWM:");
    w.commit().unwrap();
    println!("{mb} MiB {}: base_rss {base} MiB, hwm before add {before}, after add {after_add}, after commit {} ; {:?}", mode, status("VmHWM:"), t.elapsed());
    std::fs::remove_dir_all(&tmp).ok();
}
