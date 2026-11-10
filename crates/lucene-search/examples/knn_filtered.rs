//! Recall and latency of tenant-filtered KNN searches through the real index
//! path, by tenant size, for tenants unrelated to the vector space (random
//! documents) and tenants that are one region of it (the documents nearest
//! an anchor), with queries from anywhere or from the tenant's own region.
#![allow(clippy::arithmetic_side_effects)]

use std::time::Instant;

use lucene_codecs::field_infos::{
    FieldInfo, IndexOptions, VectorEncoding, VectorSimilarityFunction,
};
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{
    DocumentVector, ExplicitDocument, IndexWriter, InvertedField, InvertedTerms,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::knn_search::{knn_top_hits_with, KnnTarget};
use lucene_search::query::{Clause, TermQuery};
use lucene_store::FsDirectory;

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

/// The `m` rows of `rows` nearest `q`.
fn nearest(
    rows: &[f32],
    dim: usize,
    q: &[f32],
    m: usize,
    skip: impl Fn(usize) -> bool,
) -> Vec<usize> {
    let mut s: Vec<(f32, usize)> = rows
        .chunks_exact(dim)
        .enumerate()
        .filter(|(i, _)| !skip(*i))
        .map(|(i, r)| (dot(r, q), i))
        .collect();
    let m = m.min(s.len());
    if m < s.len() {
        s.select_nth_unstable_by(m, |a, b| b.0.total_cmp(&a.0));
    }
    s.truncate(m);
    s.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    s.into_iter().map(|(_, i)| i).collect()
}

struct Tenant {
    label: String,
    term: String,
    members: Vec<usize>,
    /// Query indices into the query file.
    queries: Vec<usize>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prefix = &args[1];
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let nq: usize = opt("--queries").map_or(200, |s| s.parse().unwrap());
    let nprobes: Vec<usize> = opt("--nprobes")
        .unwrap_or_else(|| "0".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let efs: Vec<usize> = opt("--ef")
        .unwrap_or_else(|| "10".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let sizes: Vec<usize> = opt("--sizes")
        .unwrap_or_else(|| "30,300,1000,3000,10000,40000".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let only: Option<String> = opt("--filters");
    let fmt = opt("--format").unwrap_or_else(|| "ivf-f16".into());
    let (n_all, dim, mut base) = read_bin(&format!("{prefix}.base.fbin"));
    let n = opt("--limit").map_or(n_all, |s| s.parse::<usize>().unwrap().min(n_all));
    base.truncate(n * dim);
    let (nq_all, _, mut queries) = read_bin(&format!("{prefix}.query.fbin"));
    for v in base
        .chunks_exact_mut(dim)
        .chain(queries.chunks_exact_mut(dim))
    {
        let norm = dot(v, v).sqrt().max(1e-12);
        v.iter_mut().for_each(|x| *x /= norm);
    }
    // Tenants. Unclustered `u<size>`: random documents. Clustered `c<size>`:
    // the documents nearest an anchor (a query vector), queried with the
    // query vectors nearest that anchor (`c<size>-local`) or random ones.
    let anchors_from = nq_all - 64;
    let mut tenants: Vec<Tenant> = vec![Tenant {
        label: "none".into(),
        term: String::new(),
        members: (0..n).collect(),
        queries: (0..nq).collect(),
    }];
    for (j, &size) in sizes.iter().enumerate() {
        let members: Vec<usize> = (0..n)
            .filter(|&i| mix(i as u64 * 31 + j as u64) % (n as u64) < size as u64)
            .collect();
        tenants.push(Tenant {
            label: format!("u{size}"),
            term: format!("u{j}"),
            members,
            queries: (0..nq).collect(),
        });
    }
    for (j, &size) in sizes.iter().enumerate() {
        let a = anchors_from + j;
        let anchor = &queries[a * dim..(a + 1) * dim];
        let mut members = nearest(&base, dim, anchor, size, |_| false);
        members.sort_unstable();
        let local = nearest(&queries[..(anchors_from * dim)], dim, anchor, nq, |_| false);
        tenants.push(Tenant {
            label: format!("c{size}-local"),
            term: format!("c{j}"),
            members: members.clone(),
            queries: local,
        });
        tenants.push(Tenant {
            label: format!("c{size}-any"),
            term: format!("c{j}"),
            members,
            queries: (0..nq).collect(),
        });
    }
    // `--subset L`: an index of only tenant L's documents, searched without a
    // filter: a structure built over one tenant's block.
    let (base, n, tenants) = match opt("--subset") {
        Some(l) => {
            let t = tenants.into_iter().find(|t| t.label == l).expect("tenant");
            let sub: Vec<f32> = t
                .members
                .iter()
                .flat_map(|&i| base[i * dim..(i + 1) * dim].iter().copied())
                .collect();
            let m = t.members.len();
            (
                sub,
                m,
                vec![Tenant {
                    label: format!("sub-{}", t.label),
                    term: String::new(),
                    members: (0..m).collect(),
                    queries: t.queries,
                }],
            )
        }
        None => (base, n, tenants),
    };
    let tag = opt("--subset").map_or(String::new(), |l| format!("-sub-{l}"));
    let mut doc_terms: Vec<Vec<String>> = vec![Vec::new(); n];
    for t in tenants.iter().filter(|t| !t.term.is_empty()) {
        for &m in &t.members {
            if !doc_terms[m].contains(&t.term) {
                doc_terms[m].push(t.term.clone());
            }
        }
    }
    let path = std::path::PathBuf::from(opt("--dir").unwrap_or_else(|| "/tmp/knn-filtered".into()))
        .join(format!("{}-{n}-{fmt}{tag}", prefix.replace('/', "_")));
    let marker = path.join("built");
    let key = format!("{:?}", sizes);
    let built = std::fs::read_to_string(&marker).is_ok_and(|k| k == key);
    if !built {
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
    }
    let dir = FsDirectory::open(&path);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    if !built {
        let t = Instant::now();
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
        w.enable_explicit_documents().unwrap();
        w.register_field(FieldInfo::new("id", 0)).unwrap();
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
                "hnsw" => KnnVectorsFormat::hnsw(16, 100).unwrap(),
                o => panic!("{o}"),
            },
        );
        w.set_ram_buffer_size_mb(8192.0).unwrap();
        w.set_merge_policy(None);
        let w = ConcurrentIndexWriter::new(w, 1).unwrap();
        let (docs, vectors): (Vec<_>, Vec<_>) = (0..n)
            .map(|i| {
                let mut d = ExplicitDocument::default();
                d.stored.push(StoredField {
                    field_number: 0,
                    value: FieldValue::Int(i as i32),
                });
                if !doc_terms[i].is_empty() {
                    let mut terms = InvertedTerms::default();
                    for term in &doc_terms[i] {
                        terms.push::<Vec<u8>>(term.as_bytes(), 1, &[], &[], &[]);
                    }
                    d.fields.inverted.push(InvertedField {
                        field_number: 1,
                        terms,
                        norm: None,
                    });
                }
                (
                    d,
                    vec![DocumentVector::float32(
                        "v",
                        base[i * dim..(i + 1) * dim].to_vec(),
                    )],
                )
            })
            .unzip();
        w.add_explicit_documents_with_vectors_in(None, docs, vectors)
            .unwrap();
        w.flush().unwrap();
        w.commit().unwrap();
        w.into_writer().unwrap().close().unwrap();
        std::fs::write(&marker, &key).unwrap();
        let bytes: u64 = std::fs::read_dir(&path)
            .unwrap()
            .filter_map(|e| e.ok()?.metadata().ok())
            .map(|m| m.len())
            .sum();
        eprintln!(
            "built {fmt} {n}{tag} in {:.2}s, {:.2} MB",
            t.elapsed().as_secs_f64(),
            bytes as f64 / 1e6
        );
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segs = opened.as_open_segments();
    let mut ids: Vec<i32> = Vec::with_capacity(n);
    for seg in reader.segment_readers() {
        for doc in 0..seg.max_doc {
            match seg.stored_document(doc).unwrap().unwrap().fields[0].value {
                FieldValue::Int(id) => ids.push(id),
                ref o => panic!("{o:?}"),
            }
        }
    }
    let settings: Vec<(usize, usize)> = if fmt.starts_with("ivf") {
        nprobes.iter().map(|&p| (p, K)).collect()
    } else {
        efs.iter().map(|&e| (0, e)).collect()
    };
    println!("index,n,format,tenant,passing,setting,recall@{K},min_q_recall,mean_us,p99_us");
    for t in &tenants {
        if only
            .as_ref()
            .is_some_and(|o| !o.split(',').any(|x| t.label.starts_with(x)))
        {
            continue;
        }
        let truth: Vec<Vec<i32>> = t
            .queries
            .iter()
            .map(|&q| {
                let qv = &queries[q * dim..(q + 1) * dim];
                let mut s: Vec<(f32, usize)> = t
                    .members
                    .iter()
                    .map(|&i| (dot(&base[i * dim..(i + 1) * dim], qv), i))
                    .collect();
                s.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                s.truncate(K);
                s.into_iter().map(|(_, i)| i as i32).collect()
            })
            .collect();
        let clause = (!t.term.is_empty()).then(|| {
            Clause::Term(TermQuery {
                field: "t".into(),
                term: t.term.as_bytes().to_vec(),
                doc_freq: None,
            })
        });
        for &(np, ef) in &settings {
            let search = |q: usize| {
                knn_top_hits_with(
                    &segs,
                    "v",
                    &KnnTarget::Float(queries[q * dim..(q + 1) * dim].to_vec()),
                    K,
                    ef,
                    clause.as_ref(),
                    np,
                )
                .unwrap()
            };
            for &q in t.queries.iter().take(30) {
                search(q);
            }
            let mut times = Vec::new();
            let (mut found, mut total, mut worst) = (0usize, 0usize, 1f64);
            for (qi, &q) in t.queries.iter().enumerate() {
                let tm = Instant::now();
                let hits = search(q);
                times.push(tm.elapsed().as_secs_f64() * 1e6);
                let f = hits
                    .iter()
                    .take(K)
                    .filter(|h| truth[qi].contains(&ids[h.doc_id as usize]))
                    .count();
                found += f;
                total += truth[qi].len();
                worst = worst.min(f as f64 / truth[qi].len().max(1) as f64);
            }
            times.sort_by(f64::total_cmp);
            let m = times.len();
            let setting = if fmt.starts_with("ivf") {
                format!("nprobe={np}")
            } else {
                format!("ef={ef}")
            };
            println!(
                "{},{n},{fmt},{},{},{setting},{:.4},{worst:.2},{:.0},{:.0}",
                prefix.rsplit('/').next().unwrap(),
                t.label,
                t.members.len(),
                found as f64 / total.max(1) as f64,
                times.iter().sum::<f64>() / m as f64,
                times[(m * 99 / 100).min(m - 1)]
            );
        }
    }
}
