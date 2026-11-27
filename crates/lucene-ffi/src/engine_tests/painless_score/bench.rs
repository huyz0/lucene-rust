#![cfg(test)]
use crate::engine_tests::nodes::tests::{open, Blob};
use crate::jvm_reader::tests::run_limit;

use lucene_engine::painless_score::*;

use lucene_painless::ir;

use lucene_search::values_source::{DoubleValues, DoubleValuesSource, ValuesContext};
use lucene_search::Result;

use lucene_util::test_support::TempDir;

use std::sync::Arc;

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// `PAINLESS_BENCH_INDEX=<dir> PAINLESS_BENCH_BLOBS=<tsv> cargo test
/// --release -p lucene-ffi --lib painless_score_bench -- --ignored
/// --nocapture`: each script's ns per document over a match-all query.
#[test]
#[ignore]
fn painless_score_bench() {
    let from = std::env::var("PAINLESS_BENCH_INDEX").unwrap();
    let tmp = TempDir::new("painless-bench");
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), tmp.path().join(e.file_name())).unwrap();
    }
    let h = open(&tmp);
    let time = |blob: &Blob| {
        let mut best = f64::MAX;
        let mut total = 0;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            total = run_limit(h, &blob.0, 10, i64::MAX)
                .unwrap_or_else(|_| panic!("{}", lucene_engine::status::last_error()))
                .1;
            best = best.min(t.elapsed().as_nanos() as f64);
        }
        best / total as f64
    };
    let base = time(&Blob::tree().u8(5));
    println!("match_all\t{base:.2} ns/doc");
    let blobs = std::fs::read_to_string(std::env::var("PAINLESS_BENCH_BLOBS").unwrap()).unwrap();
    for line in blobs.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        let mut b = Blob::tree()
            .u8(NODE_PAINLESS_SCORE)
            .u8(5)
            .u8(1)
            .bytes(&unhex(cols[1]))
            .bytes(&unhex(cols[2]))
            .str("idx")
            .i32(0)
            .i32(0)
            .i32(cols.len() as i32 - 3);
        for f in &cols[3..] {
            let (name, kind) = f.split_once(':').unwrap();
            b = b.str(name).u8(kind.parse().unwrap());
        }
        let t = time(&b);
        println!("{}\t{:.2} ns/doc (script {:.2})", cols[0], t, t - base);
    }
}

struct OneAndHalf;

impl DoubleValues for OneAndHalf {
    fn advance_exact(&mut self, _doc: i32) -> Result<bool> {
        Ok(true)
    }
    fn double_value(&mut self) -> Result<f64> {
        Ok(1.5)
    }
}

/// The script alone per document, as the Java harness times
/// `setDocument` + `execute`: `PAINLESS_BENCH_INDEX` (one segment) and
/// `PAINLESS_BENCH_BLOBS` as for [`painless_score_bench`].
#[test]
#[ignore]
fn painless_script_loop_bench() {
    let dir =
        lucene_store::directory::FsDirectory::open(std::env::var("PAINLESS_BENCH_INDEX").unwrap());
    let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
    let seg = &reader.segment_readers()[0];
    let ctx = ValuesContext::for_reader(seg);
    let blobs = std::fs::read_to_string(std::env::var("PAINLESS_BENCH_BLOBS").unwrap()).unwrap();
    for line in blobs.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        let ir_blob = unhex(cols[1]);
        let params_blob = unhex(cols[2]);
        let fields = cols[3..]
            .iter()
            .map(|f| {
                let (name, kind) = f.split_once(':').unwrap();
                let kind = match kind {
                    "0" => FieldKind::Long,
                    "1" => FieldKind::Double,
                    "5" => FieldKind::Keyword,
                    other => panic!("kind {other}"),
                };
                FieldSpec {
                    name: name.to_string(),
                    read: name.to_string(),
                    kind,
                }
            })
            .collect();
        let source = PainlessScoreSource(Arc::new(Spec {
            script: ir::decode(&ir_blob).unwrap(),
            params: ir::decode_constant(&params_blob).unwrap(),
            ir_blob,
            params_blob,
            needs_score: true,
            filter: false,
            sort: false,
            agg: false,
            metric_map: false,
            terms_set: false,
            interval: false,
            similarity: false,
            similarity_weight: false,
            index: "idx".into(),
            shard: 0,
            script_id: 0,
            script_id_at: 0,
            fields,
            js: None,
            expr: None,
        }));
        let n = seg.max_doc;
        let mut best = f64::MAX;
        let mut sum = 0.0;
        for _ in 0..8 {
            let mut v = source
                .get_values(&ctx, 0, Some(Box::new(OneAndHalf)))
                .unwrap();
            let t = std::time::Instant::now();
            sum = 0.0;
            if std::env::var("PAINLESS_BENCH_PER_DOC").is_ok() {
                for d in 0..n {
                    v.advance_exact(d).unwrap();
                    sum += v.double_value().unwrap();
                }
            } else {
                // The function score scorer's batches.
                let (mut docs, scores) = (Vec::new(), vec![1.5f32; 64]);
                let (mut out, mut has) = (vec![0.0; 64], vec![false; 64]);
                for start in (0..n).step_by(64) {
                    docs.clear();
                    docs.extend(start..(start + 64).min(n));
                    v.fill_batch(&docs, &scores, &mut out, &mut has).unwrap();
                    sum += out[..docs.len()].iter().sum::<f64>();
                }
            }
            best = best.min(t.elapsed().as_nanos() as f64 / f64::from(n));
        }
        println!("{}\t{best:.2} ns/doc\tsum={sum}", cols[0]);
    }
}
