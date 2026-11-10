//! Recall and latency of nested (block-join) KNN searches through the real
//! index path: the best child of each of the best `k` parents, over GloVe
//! vectors grouped into parents of 1 to `--children` children each.
//!
//! `--order local` reads `<prefix>.local.fbin`, whose consecutive rows lie
//! near each other, so a parent's children are alike (as a document's
//! chunks are); `--order random` reads `<prefix>.base.fbin`, whose order is
//! random. `--filter half` passes the even children only.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;
use std::time::Instant;

use lucene_codecs::field_infos::{
    FieldInfo, IndexOptions, VectorEncoding, VectorSimilarityFunction,
};
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{
    DocumentVector, ExplicitDocument, IndexWriter, InvertedField, InvertedTerms,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::join::BitSetProducer;
use lucene_search::knn_search::{knn_top_hits_nested_with, KnnTarget};
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::{Clause, TermQuery};
use lucene_store::FsDirectory;
use lucene_util::fixed_bit_set::FixedBitSet;

const K: usize = 10;

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

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^ (x >> 32)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[derive(Debug)]
struct Parents(Arc<FixedBitSet>);

impl BitSetProducer for Parents {
    fn bit_set(&self, _: &OpenSegment<'_>) -> lucene_search::Result<Option<Arc<FixedBitSet>>> {
        Ok(Some(self.0.clone()))
    }

    fn key(&self) -> String {
        "parents".into()
    }
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
    let nprobes = list("--nprobes", "0");
    let efs = list("--ef", "10");
    let max_children: usize = opt("--children").map_or(8, |s| s.parse().unwrap());
    let order = opt("--order").unwrap_or_else(|| "local".into());
    let filter = opt("--filter").unwrap_or_else(|| "none".into());
    let fmt = opt("--format").unwrap_or_else(|| "ivf-f16".into());
    let file = match order.as_str() {
        "local" => "local",
        "random" => "base",
        o => panic!("{o}"),
    };
    let (n_all, dim, mut base) = read_bin(&format!("{prefix}.{file}.fbin"));
    let n = opt("--limit").map_or(n_all, |s| s.parse::<usize>().unwrap().min(n_all));
    base.truncate(n * dim);
    let (_, _, mut queries) = read_bin(&format!("{prefix}.query.fbin"));
    for v in base
        .chunks_exact_mut(dim)
        .chain(queries.chunks_exact_mut(dim))
    {
        let norm = dot(v, v).sqrt().max(1e-12);
        v.iter_mut().for_each(|x| *x /= norm);
    }
    // Parents over consecutive rows: `parent_of[row]`, and each doc's row
    // (`None` for the parent closing a block).
    let mut parent_of = Vec::with_capacity(n);
    let mut docs: Vec<Option<usize>> = Vec::with_capacity(n + n / 2);
    let mut p = 0usize;
    while parent_of.len() < n {
        let size = 1 + (mix(p as u64) as usize) % max_children;
        for _ in 0..size.min(n - parent_of.len()) {
            docs.push(Some(parent_of.len()));
            parent_of.push(p);
        }
        docs.push(None);
        p += 1;
    }
    let parents_n = p;
    let mut bits = FixedBitSet::new(docs.len());
    for (d, r) in docs.iter().enumerate() {
        if r.is_none() {
            bits.set(d);
        }
    }
    let parents = Parents(Arc::new(bits));
    let path = std::path::PathBuf::from(opt("--dir").unwrap_or_else(|| "/tmp/knn-nested".into()))
        .join(format!(
            "{}-{n}-{order}-c{max_children}-{fmt}",
            prefix.replace('/', "_")
        ));
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
        w.register_field(
            FieldInfo::new("t", 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
        )
        .unwrap();
        w.register_field(FieldInfo::new("v", 0).with_vectors(
            dim as i32,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Cosine,
        ))
        .unwrap();
        w.set_knn_vectors_format_for_field(
            "v",
            match fmt.as_str() {
                "ivf-f16" => KnnVectorsFormat::ivf(384, 1).unwrap(),
                "ivf" => KnnVectorsFormat::ivf(384, 0).unwrap(),
                "hnsw" => KnnVectorsFormat::hnsw(16, 100).unwrap(),
                o => panic!("{o}"),
            },
        );
        w.set_ram_buffer_size_mb(8192.0).unwrap();
        w.set_merge_policy(None);
        let w = ConcurrentIndexWriter::new(w, 1).unwrap();
        let (batch, vectors): (Vec<_>, Vec<_>) = docs
            .iter()
            .map(|r| match *r {
                Some(row) => {
                    let mut d = ExplicitDocument::default();
                    let mut terms = InvertedTerms::default();
                    let term: &[u8] = if row.is_multiple_of(2) { b"even" } else { b"odd" };
                    terms.push::<Vec<u8>>(term, 1, &[], &[], &[]);
                    d.fields.inverted.push(InvertedField {
                        field_number: 0,
                        terms,
                        norm: None,
                    });
                    (
                        d,
                        vec![DocumentVector::float32(
                            "v",
                            base[row * dim..(row + 1) * dim].to_vec(),
                        )],
                    )
                }
                None => (ExplicitDocument::default(), Vec::new()),
            })
            .unzip();
        w.add_explicit_documents_with_vectors_in(None, batch, vectors)
            .unwrap();
        w.flush().unwrap();
        w.commit().unwrap();
        w.into_writer().unwrap().close().unwrap();
        std::fs::write(&marker, "").unwrap();
        eprintln!("built {fmt} {n} in {:.2}s", t.elapsed().as_secs_f64());
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), 1, "doc ids are layout ids");
    let opened = reader.open_segments().unwrap();
    let segs = opened.as_open_segments();
    let keep = |row: usize| filter == "none" || row.is_multiple_of(2);
    let clause = (filter != "none").then(|| {
        Clause::Term(TermQuery {
            field: "t".into(),
            term: b"even".to_vec(),
            doc_freq: None,
        })
    });
    // Each query's exact best `K` parents.
    let truth: Vec<Vec<usize>> = (0..nq)
        .map(|q| {
            let qv = &queries[q * dim..(q + 1) * dim];
            let mut best = vec![f32::NEG_INFINITY; parents_n];
            for row in (0..n).filter(|&r| keep(r)) {
                let s = dot(&base[row * dim..(row + 1) * dim], qv);
                let b = &mut best[parent_of[row]];
                *b = b.max(s);
            }
            let mut s: Vec<(f32, usize)> = best
                .into_iter()
                .enumerate()
                .filter(|(_, s)| s.is_finite())
                .map(|(p, s)| (s, p))
                .collect();
            s.select_nth_unstable_by(K, |a, b| b.0.total_cmp(&a.0));
            s.truncate(K);
            s.into_iter().map(|(_, p)| p).collect()
        })
        .collect();
    let settings: Vec<(usize, usize)> = if fmt.starts_with("ivf") {
        nprobes.iter().map(|&p| (p, K)).collect()
    } else {
        efs.iter().map(|&e| (0, e)).collect()
    };
    println!(
        "index,vectors,parents,order,format,filter,setting,recall@{K},min_q_recall,mean_us,p99_us"
    );
    for &(np, ef) in &settings {
        let search = |q: usize| {
            knn_top_hits_nested_with(
                &segs,
                "v",
                &KnnTarget::Float(queries[q * dim..(q + 1) * dim].to_vec()),
                K,
                ef,
                clause.as_ref(),
                &parents,
                false,
                np,
            )
            .unwrap()
        };
        for q in 0..nq.min(30) {
            search(q);
        }
        let mut times = Vec::new();
        let (mut found, mut worst) = (0usize, 1f64);
        for (q, want) in truth.iter().enumerate() {
            let tm = Instant::now();
            let hits = search(q);
            times.push(tm.elapsed().as_secs_f64() * 1e6);
            let got: Vec<usize> = hits
                .iter()
                .take(K)
                .map(|h| parent_of[docs[h.doc_id as usize].expect("a child")])
                .collect();
            let f = want.iter().filter(|p| got.contains(p)).count();
            found += f;
            worst = worst.min(f as f64 / K as f64);
        }
        times.sort_by(f64::total_cmp);
        let m = times.len();
        let setting = if fmt.starts_with("ivf") {
            format!("nprobe={np}")
        } else {
            format!("ef={ef}")
        };
        println!(
            "{},{n},{parents_n},{order},{fmt},{filter},{setting},{:.4},{worst:.2},{:.0},{:.0}",
            prefix.rsplit('/').next().unwrap(),
            found as f64 / (nq * K) as f64,
            times.iter().sum::<f64>() / m as f64,
            times[(m * 99 / 100).min(m - 1)]
        );
    }
}
