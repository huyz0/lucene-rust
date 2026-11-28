#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::jvm_reader::{ffi_close_jvm_reader, ffi_open_jvm_reader};
use lucene_index::document::{
    Document, KeywordField, NumericDocValuesField, Store, StringField, TextField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const DOCS: usize = 60;

struct Shard {
    _dir: TempDir,
    handle: u64,
}

impl Drop for Shard {
    fn drop(&mut self) {
        ffi_close_jvm_reader(self.handle);
    }
}

fn put(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    put(out, u32::try_from(b.len()).unwrap());
    out.extend_from_slice(b);
}

fn analyzer() -> Vec<u8> {
    lucene_engine::analysis::test_support::spec(&[], ("standard", &[]), &[("lowercase", &[])], &[])
}

fn table_with(name: &str, version: u32) -> Vec<u8> {
    let mut b = Vec::new();
    put(&mut b, version);
    put_bytes(&mut b, name.as_bytes());
    put(&mut b, 4);
    for (kind, col, arg, analyzer) in [
        (KEY, "id", 0, Vec::new()),
        (LONG, "price", 0, Vec::new()),
        (KEYWORD, "cat", 0, Vec::new()),
        (TEXT, "body", 0, analyzer()),
    ] {
        b.push(kind);
        put_bytes(&mut b, col.as_bytes());
        put(&mut b, arg);
        put_bytes(&mut b, &analyzer);
    }
    b
}

fn table() -> Vec<u8> {
    table_with("t", TABLE_VERSION)
}

/// `DOCS` documents in three segments; `dead` are masked as the JVM's live
/// docs do (words per segment, in segment order).
fn shard(dead: &[usize]) -> Shard {
    shard_with(DOCS, 3, dead)
}

fn shard_of(docs: usize, segments: usize) -> Shard {
    shard_with(docs, segments, &[])
}

fn shard_with(docs: usize, segments: usize, dead: &[usize]) -> Shard {
    let dir = TempDir::new("ffi-sql");
    {
        let fs = FsDirectory::open(dir.path_str());
        let version = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).unwrap();
        for i in 0..docs {
            let mut doc = Document::new();
            doc.add(StringField::new("id", format!("doc-{i:03}"), Store::Yes));
            let body = if i % 2 == 0 { "Alpha beta" } else { "gamma" };
            doc.add(TextField::new("body", body.to_string(), Store::No));
            doc.add(NumericDocValuesField::new("price", i as i64));
            doc.add(KeywordField::new("cat", format!("c{}", i % 5), Store::No));
            w.add_fields_document(&doc).unwrap();
            if (i + 1) % (docs / segments) == 0 {
                w.commit().unwrap();
            }
        }
        w.commit().unwrap();
    }
    let path = dir.path_str().to_string();
    let reader =
        lucene_search::directory_reader::DirectoryReader::open(&FsDirectory::open(&path)).unwrap();
    let max_docs: Vec<i32> = reader.segment_readers().iter().map(|s| s.max_doc).collect();
    let generation = reader.segment_infos.generation;
    let infos = std::fs::read(format!("{path}/segments_{}", base36(generation))).unwrap();
    let mut counts = Vec::new();
    let mut words: Vec<u64> = Vec::new();
    let mut base = 0usize;
    for &m in &max_docs {
        let m = m as usize;
        let hit: Vec<usize> = dead
            .iter()
            .copied()
            .filter(|d| (base..base + m).contains(d))
            .map(|d| d - base)
            .collect();
        if hit.is_empty() {
            counts.push(0usize);
        } else {
            let mut w = vec![u64::MAX; m.div_ceil(64)];
            if !m.is_multiple_of(64) {
                *w.last_mut().unwrap() = (1u64 << (m % 64)) - 1;
            }
            for d in hit {
                w[d / 64] &= !(1u64 << (d % 64));
            }
            counts.push(w.len());
            words.extend(w);
        }
        base += m;
    }
    let mut handle = 0u64;
    // SAFETY: every pointer is to a live local buffer of the stated length.
    let rc = unsafe {
        ffi_open_jvm_reader(
            path.as_ptr().cast(),
            path.len(),
            infos.as_ptr(),
            infos.len(),
            generation,
            0,
            max_docs.as_ptr(),
            max_docs.len(),
            words.as_ptr(),
            counts.as_ptr(),
            &mut handle,
        )
    };
    assert_eq!(rc, 0, "{}", lucene_engine::status::last_error());
    Shard { _dir: dir, handle }
}

fn base36(mut n: i64) -> String {
    if n == 0 {
        return "0".into();
    }
    let mut digits = Vec::new();
    while n > 0 {
        digits.push(char::from_digit((n % 36) as u32, 36).unwrap());
        n /= 36;
    }
    digits.iter().rev().collect()
}

fn run(shard: &Shard, table: &[u8], sql: &str) -> Result<serde_json::Value, String> {
    let mut ptr = std::ptr::null_mut();
    let mut len = 0usize;
    // SAFETY: the buffers are live locals; the owned result is freed below.
    let rc = unsafe {
        ffi_jvm_sql_query(
            shard.handle,
            table.as_ptr(),
            table.len(),
            sql.as_ptr().cast(),
            sql.len(),
            &mut ptr,
            &mut len,
        )
    };
    if rc != 0 {
        assert!(ptr.is_null());
        return Err(lucene_engine::status::last_error());
    }
    // SAFETY: the library wrote `len` bytes at `ptr`.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
    // SAFETY: the pair is the one the call returned.
    unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
    Ok(serde_json::from_slice(&bytes).unwrap())
}

fn rows(v: &serde_json::Value) -> &Vec<serde_json::Value> {
    v["datarows"].as_array().unwrap()
}

#[test]
fn a_statement_reads_the_shard_through_the_table_blob() {
    let s = shard(&[]);
    let t = table();
    let v = run(&s, &t, "SELECT id, price FROM t WHERE id = 'doc-007'").unwrap();
    assert_eq!(v["schema"][0]["name"], "id");
    assert_eq!(v["schema"][1]["type"], "long");
    assert_eq!(rows(&v), &[serde_json::json!(["doc-007", 7])]);

    let v = run(
        &s,
        &t,
        "SELECT cat, count(*) AS n, sum(price) AS total FROM t GROUP BY cat ORDER BY cat",
    )
    .unwrap();
    assert_eq!(rows(&v).len(), 5);
    let n: i64 = rows(&v).iter().map(|r| r[1].as_i64().unwrap()).sum();
    assert_eq!(n, DOCS as i64);

    // The text column is analysed with the analyzer the blob carries.
    let v = run(
        &s,
        &t,
        "SELECT id, _score FROM t WHERE match_text(body, 'ALPHA') ORDER BY _score DESC LIMIT 5",
    )
    .unwrap();
    assert_eq!(rows(&v).len(), 5);
    assert!(rows(&v)
        .iter()
        .all(|r| r[0].as_str().unwrap()[4..].parse::<usize>().unwrap() % 2 == 0));
}

#[test]
fn the_jvm_s_live_docs_decide_which_documents_exist() {
    let dead = [3usize, 7, 42];
    let s = shard(&dead);
    let t = table();
    let v = run(&s, &t, "SELECT count(*) FROM t").unwrap();
    assert_eq!(rows(&v)[0][0], DOCS as i64 - 3);
    let v = run(&s, &t, "SELECT id FROM t WHERE id = 'doc-007'").unwrap();
    assert!(rows(&v).is_empty());
    assert_eq!(v["schema"][0]["name"], "id");
    let v = run(&s, &t, "SELECT id FROM t WHERE id = 'doc-008'").unwrap();
    assert_eq!(rows(&v).len(), 1);
}

#[test]
fn engines_are_kept_per_handle_and_dropped_when_it_closes() {
    let s = shard(&[]);
    let t = table();
    run(&s, &t, "SELECT count(*) FROM t").unwrap();
    assert!(lock(engines()).contains_key(&s.handle));
    let handle = s.handle;
    drop(s);
    assert!(!lock(engines()).contains_key(&handle));
}

#[test]
fn a_changed_table_replaces_the_engine() {
    let s = shard(&[]);
    run(&s, &table(), "SELECT count(*) FROM t").unwrap();
    let narrower = {
        let mut b = Vec::new();
        put(&mut b, TABLE_VERSION);
        put_bytes(&mut b, b"t");
        put(&mut b, 1);
        b.push(LONG);
        put_bytes(&mut b, b"price");
        put(&mut b, 0);
        put_bytes(&mut b, b"");
        b
    };
    let err = run(&s, &narrower, "SELECT cat FROM t").unwrap_err();
    assert!(err.contains("cat"), "{err}");
    let v = run(&s, &narrower, "SELECT max(price) FROM t").unwrap();
    assert_eq!(rows(&v)[0][0], DOCS as i64 - 1);
}

#[test]
fn bad_input_is_a_status_with_a_message_never_a_panic() {
    let s = shard(&[]);
    let err = run(&s, &table_with("t", 9), "SELECT 1 FROM t").unwrap_err();
    assert!(err.contains("version"), "{err}");
    assert!(run(&s, &table()[..10], "SELECT 1 FROM t").is_err());
    let err = run(&s, &table(), "DELETE FROM t").unwrap_err();
    assert!(
        err.to_lowercase().contains("delete") || err.contains("DDL") || err.contains("DML"),
        "{err}"
    );
    assert!(run(&s, &table(), "SELEC").is_err());
    let err = run(&s, &table(), "SELECT nope FROM t").unwrap_err();
    assert!(err.contains("nope"), "{err}");

    let mut ptr = std::ptr::null_mut();
    let mut len = 0usize;
    // SAFETY: live locals.
    let rc = unsafe {
        ffi_jvm_sql_query(
            0xdead_beef,
            table().as_ptr(),
            0,
            b"x".as_ptr().cast(),
            1,
            &mut ptr,
            &mut len,
        )
    };
    assert_ne!(rc, 0);
}

#[test]
fn the_table_of_a_statement_is_named_before_anything_opens() {
    let name = |sql: &str| -> Result<String, String> {
        let mut ptr = std::ptr::null_mut();
        let mut len = 0usize;
        // SAFETY: live locals; the owned result is freed below.
        let rc = unsafe { ffi_jvm_sql_table(sql.as_ptr().cast(), sql.len(), &mut ptr, &mut len) };
        if rc != 0 {
            return Err(lucene_engine::status::last_error());
        }
        // SAFETY: `len` bytes at `ptr`, returned by the call.
        let out = String::from_utf8(unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec());
        // SAFETY: the pair the call returned.
        unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
        Ok(out.unwrap())
    };
    assert_eq!(name("SELECT * FROM \"logs-1\" LIMIT 1").unwrap(), "logs-1");
    assert!(name("SELECT * FROM a JOIN b ON a.x = b.x").is_err());
}

#[test]
fn router_and_standard_statements_are_counted() {
    let s = shard(&[]);
    let t = table();
    let mut before = [0u64; 2];
    // SAFETY: room for two values.
    unsafe { ffi_jvm_sql_stats(before.as_mut_ptr()) };
    run(&s, &t, "SELECT id FROM t WHERE id = 'doc-001'").unwrap();
    run(&s, &t, "SELECT DISTINCT price FROM t").unwrap();
    let mut after = [0u64; 2];
    // SAFETY: room for two values.
    unsafe { ffi_jvm_sql_stats(after.as_mut_ptr()) };
    assert!(after[0] > before[0] && after[1] > before[1]);
}

/// What the bridge and the JSON writer add to a statement: the engine's own
/// batches against `query_json` and the whole exported call. Run with
/// `cargo test --release -p lucene-ffi --lib sql_overhead -- --ignored --nocapture`.
#[test]
#[ignore = "a measurement, not a check"]
fn sql_overhead() {
    use std::time::Instant;
    let s = shard_of(200_000, 4);
    let t = table();
    let engine = engine_for(s.handle, &t).unwrap();
    let statements = [
        ("point read", "SELECT id, price FROM t WHERE id = 'doc-123456'"),
        (
            "text top-10",
            "SELECT price, _score FROM t WHERE match_text(body, 'alpha') ORDER BY _score DESC LIMIT 10",
        ),
        (
            "group by",
            "SELECT cat, count(*), sum(price), min(price), max(price) FROM t GROUP BY cat",
        ),
        ("range count", "SELECT count(*) FROM t WHERE price > 150000 AND cat = 'c3'"),
    ];
    let p50 = |f: &mut dyn FnMut()| {
        for _ in 0..300 {
            f();
        }
        let mut us: Vec<f64> = (0..2000)
            .map(|_| {
                let t0 = Instant::now();
                f();
                t0.elapsed().as_secs_f64() * 1e6
            })
            .collect();
        us.sort_by(f64::total_cmp);
        us[us.len() / 2]
    };
    println!(
        "{:14} {:>10} {:>12} {:>12} {:>12}",
        "statement", "engine us", "json us", "ffi call us", "scoped us"
    );
    for (name, sql) in statements {
        let a = p50(&mut || {
            std::hint::black_box(engine.query_blocking(sql).unwrap());
        });
        let b = p50(&mut || {
            std::hint::black_box(query_json(s.handle, &t, sql).unwrap());
        });
        let c = p50(&mut || {
            run(&s, &t, sql).unwrap();
        });
        // The same call inside a request memory scope, as the plugin opens one.
        let d = p50(&mut || {
            // SAFETY: a label of 3 valid bytes; the end call has room for 4 values.
            unsafe {
                crate::memory::ffi_memory_scope_begin(1, 0, b"sql".as_ptr(), 3);
                run(&s, &t, sql).unwrap();
                let mut out = [0i64; 4];
                crate::memory::ffi_memory_scope_end(out.as_mut_ptr(), 4);
            }
        });
        println!("{name:14} {a:>10.1} {b:>12.1} {c:>12.1} {d:>12.1}");
    }
}
