//! Recall and latency of grouped KNN (the best `size` documents of each of
//! the best `k` groups of a field) through the real index path, against the
//! ways to get groups out of an ungrouped search: collapsing its top `k`
//! (stock's knn + collapse) or a top `k * over` grouped afterwards.
//!
//! `<prefix>.base.fbin` and `<prefix>.query.fbin` are the vectors (`u32`
//! count, `u32` dimension, then `f32`s, normalized here), `--groups FILE`
//! one little-endian `u32` group per base row, stored as the numeric doc
//! value `g`. The index is written once into `--dir`, in four segments.
//!
//! ```text
//! cargo run --release -p lucene-search --example knn_grouped -- PREFIX \
//!     --groups FILE --format hnsw|ivf --ef 100,200 --size 1,3 --queries 200
//! ```
#![allow(clippy::arithmetic_side_effects)]

use std::time::Instant;

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, VectorEncoding, VectorSimilarityFunction,
};
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{DocumentVector, ExplicitDocument, IndexWriter};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::knn_search::{knn_top_hits_grouped, knn_top_hits_with, KnnGroupBy, KnnTarget};
use lucene_search::top_field::CollapseKind;
use lucene_search::ScoreDoc;
use lucene_store::FsDirectory;

const K: usize = 10;

/// A search of query `q`: the grouped rows, and the groups its hits cover.
type Run<'a> = Box<dyn Fn(usize) -> (Vec<usize>, usize) + 'a>;
const SEGMENTS: usize = 4;

fn read_bin(path: &str) -> (usize, usize, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    (
        n,
        d,
        bytes[8..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    )
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The best `size` rows of each of the best `k` groups of `hits` (best
/// first), as rows; and the groups `hits` cover.
fn group(hits: &[(f32, usize)], groups: &[u32], k: usize, size: usize) -> (Vec<usize>, usize) {
    let mut kept: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for &(_, row) in hits {
        let g = groups[row];
        seen.insert(g);
        let n = kept.len();
        let e = kept.entry(g);
        let held = matches!(e, std::collections::hash_map::Entry::Occupied(_));
        if !held && n >= k {
            continue;
        }
        let c = e.or_insert(0);
        if *c < size {
            *c += 1;
            out.push(row);
        }
    }
    (out, seen.len())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prefix = &args[1];
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let list = |name: &str, default: &str| -> Vec<usize> {
        opt(name)
            .unwrap_or_else(|| default.into())
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect()
    };
    let nq: usize = opt("--queries").map_or(200, |s| s.parse().unwrap());
    let efs = list("--ef", "100");
    let sizes = list("--size", "1,3");
    let overs = list("--over", "10,100");
    let fmt = opt("--format").unwrap_or_else(|| "hnsw".into());
    let groups_file = opt("--groups").expect("--groups FILE");
    let (n, dim, mut base) = read_bin(&format!("{prefix}.base.fbin"));
    let (_, _, mut queries) = read_bin(&format!("{prefix}.query.fbin"));
    for v in base
        .chunks_exact_mut(dim)
        .chain(queries.chunks_exact_mut(dim))
    {
        let norm = dot(v, v).sqrt().max(1e-12);
        v.iter_mut().for_each(|x| *x /= norm);
    }
    let groups: Vec<u32> = std::fs::read(&groups_file)
        .unwrap()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(groups.len(), n, "one group per row");
    let distinct = groups
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
    let label = std::path::Path::new(&groups_file)
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let path = std::path::PathBuf::from(opt("--dir").unwrap_or_else(|| "/tmp/knn-grouped".into()))
        .join(format!("{label}-{n}-{fmt}"));
    let marker = path.join("built");
    let built = marker.exists();
    if !built {
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
    }
    let dir = FsDirectory::open(&path);
    if !built {
        let t = Instant::now();
        let version = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
        w.enable_explicit_documents().unwrap();
        w.register_field(FieldInfo::new("g", 0).with_doc_values(
            DocValuesType::Numeric,
            DocValuesSkipIndexType::None,
            -1,
        ))
        .unwrap();
        w.register_field(FieldInfo::new("v", 0).with_vectors(
            dim as i32,
            VectorEncoding::Float32,
            VectorSimilarityFunction::DotProduct,
        ))
        .unwrap();
        w.set_knn_vectors_format_for_field(
            "v",
            match fmt.as_str() {
                "ivf" => KnnVectorsFormat::ivf(384, 0).unwrap(),
                "hnsw" => KnnVectorsFormat::hnsw(16, 100).unwrap(),
                o => panic!("{o}"),
            },
        );
        w.set_ram_buffer_size_mb(8192.0).unwrap();
        w.set_merge_policy(None);
        let w = ConcurrentIndexWriter::new(w, 4).unwrap();
        let per = n.div_ceil(SEGMENTS);
        for s in 0..SEGMENTS {
            let (batch, vectors): (Vec<_>, Vec<_>) = (s * per..((s + 1) * per).min(n))
                .map(|row| {
                    let mut d = ExplicitDocument::default();
                    d.fields.doc_values.push(StoredField {
                        field_number: 0,
                        value: FieldValue::Long(i64::from(groups[row])),
                    });
                    (
                        d,
                        vec![DocumentVector::float32(
                            "v",
                            base[row * dim..(row + 1) * dim].to_vec(),
                        )],
                    )
                })
                .unzip();
            w.add_explicit_documents_with_vectors_in(None, batch, vectors)
                .unwrap();
            w.flush().unwrap();
        }
        w.commit().unwrap();
        w.into_writer().unwrap().close().unwrap();
        std::fs::write(&marker, "").unwrap();
        eprintln!("built {fmt} {n} in {:.2}s", t.elapsed().as_secs_f64());
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segs = opened.as_open_segments();
    // Each query's every score, best first, once: the exact grouped answers
    // come from it.
    let t = Instant::now();
    let ranked: Vec<Vec<(f32, usize)>> = (0..nq)
        .map(|q| {
            let qv = &queries[q * dim..(q + 1) * dim];
            let mut s: Vec<(f32, usize)> = (0..n)
                .map(|row| (dot(&base[row * dim..(row + 1) * dim], qv), row))
                .collect();
            // Enough rows for any answer asked for: the best groups' members
            // are near the top.
            let keep = 50_000.min(n - 1);
            s.select_nth_unstable_by(keep, |a, b| b.0.total_cmp(&a.0));
            s.truncate(keep);
            s.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            s
        })
        .collect();
    eprintln!("exact scores in {:.1}s", t.elapsed().as_secs_f64());
    println!(
        "groups,vectors,distinct_groups,format,size,method,setting,groups_found,recall,mean_us,p99_us"
    );
    let rows = |hits: &[ScoreDoc]| -> Vec<(f32, usize)> {
        hits.iter().map(|h| (h.score, h.doc_id as usize)).collect()
    };
    let target = |q: usize| KnnTarget::Float(queries[q * dim..(q + 1) * dim].to_vec());
    for &size in &sizes {
        let truth: Vec<Vec<usize>> = ranked
            .iter()
            .map(|r| group(r, &groups, K, size).0)
            .collect();
        let by = KnnGroupBy {
            field: "g".into(),
            kind: CollapseKind::Numeric,
            per_group: size,
        };
        // (method, setting, search returning the grouped rows).
        let mut runs: Vec<(String, String, Run<'_>)> = Vec::new();
        for &ef in &efs {
            let segs = &segs;
            let groups = &groups;
            let target = &target;
            runs.push((
                "collapse_top_k".into(),
                format!("k={K} ef={ef}"),
                Box::new(move |q| {
                    let hits = knn_top_hits_with(segs, "v", &target(q), K, ef, None, 0).unwrap();
                    group(&rows(&hits), groups, K, size)
                }),
            ));
            for &over in &overs {
                let k = K * size * over;
                runs.push((
                    "oversample_then_group".into(),
                    format!("k={k} ef={}", ef.max(k)),
                    Box::new(move |q| {
                        let hits = knn_top_hits_with(segs, "v", &target(q), k, ef.max(k), None, 0)
                            .unwrap();
                        group(&rows(&hits), groups, K, size)
                    }),
                ));
            }
            let by = by.clone();
            runs.push((
                "group_by".into(),
                format!("k={K} ef={ef}"),
                Box::new(move |q| {
                    let hits =
                        knn_top_hits_grouped(segs, "v", &target(q), K, ef, None, &by, 0).unwrap();
                    let r = rows(&hits);
                    let found = r
                        .iter()
                        .map(|&(_, row)| groups[row])
                        .collect::<std::collections::HashSet<_>>()
                        .len();
                    (r.into_iter().map(|(_, row)| row).collect(), found)
                }),
            ));
        }
        for (method, setting, run) in &runs {
            for q in 0..nq.min(30) {
                run(q);
            }
            let mut times = Vec::new();
            let (mut found, mut want, mut covered) = (0usize, 0usize, 0usize);
            for (q, truth) in truth.iter().enumerate() {
                let tm = Instant::now();
                let (got, groups_found) = run(q);
                times.push(tm.elapsed().as_secs_f64() * 1e6);
                covered += groups_found.min(K);
                want += truth.len();
                found += truth.iter().filter(|r| got.contains(r)).count();
            }
            times.sort_by(f64::total_cmp);
            let m = times.len();
            println!(
                "{label},{n},{distinct},{fmt},{size},{method},{setting},{:.2},{:.4},{:.0},{:.0}",
                covered as f64 / nq as f64,
                found as f64 / want as f64,
                times.iter().sum::<f64>() / m as f64,
                times[(m * 99 / 100).min(m - 1)]
            );
        }
    }
}
