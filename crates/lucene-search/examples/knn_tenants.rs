//! Tenant-filtered KNN recall and latency at scale: a synthetic index of
//! millions of vectors (jittered copies of a base set) shared by tenants of
//! very different sizes, a few holding millions, a long tail holding
//! hundreds, laid out as an index sorted by tenant: each segment's
//! documents grouped by tenant. Some tenants are random documents, others
//! one region of the vector space; region tenants are queried both from
//! their own region (`local`) and from anywhere (`any`).
//!
//! ```text
//! cargo run --release -p lucene-search --example knn_tenants -- <prefix> <out-prefix> [options]
//!   --n N          documents (default 8000000 / scale)
//!   --scale D      tenant sizes divided by D (default 1)
//!   --segments S   segments (default 4)
//!   --format F     ivf-f16 | hnsw (default ivf-f16)
//!   --ef L         candidates (default 100; use 10, the hit count, for IVF)
//!   --partition-clusters  IVF clusters per tenant (`set_ivf_partition_clusters`)
//!   --queries Q    queries per tenant (default 100)
//!   --dir D        index directory (default /tmp/knn-tenants)
//!   --sort S       tenant | composite | time (default tenant): the index
//!                  sort, by tenant, by tenant then sub-partition, or by a
//!                  random time; each document is in one of 16
//!                  sub-partitions, and each tenant is also queried with a
//!                  filter on one of them (`<tenant>+sub`)
//! ```
#![allow(clippy::arithmetic_side_effects)]

use std::time::Instant;

use rayon::prelude::*;

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{
    DocumentVector, ExplicitDocument, IndexWriter, InvertedField, InvertedTerms,
};
use lucene_index::segment_info::{
    IndexSortField, IndexSortKind, LuceneVersion, StringMissingValue,
};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::knn_search::{knn_top_hits_with, KnnTarget};
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_store::FsDirectory;

const K: usize = 10;

/// Tenant sizes, how many tenants of each, and whether they are regions.
const CLASSES: [(usize, usize, bool); 9] = [
    (1_500_000, 1, false),
    (1_500_000, 1, true),
    (400_000, 2, false),
    (400_000, 2, true),
    (40_000, 15, false),
    (40_000, 15, true),
    (4_000, 300, false),
    (300, 3000, false),
    (30, 3000, false),
];

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

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut x = self.0;
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^ (x >> 33)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn gauss(&mut self) -> f32 {
        let (u, v) = (self.unit().max(1e-7), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn normalize(v: &mut [f32]) {
    let n = dot(v, v).sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= n);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (prefix, out) = (&args[1], &args[2]);
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let scale: usize = opt("--scale").map_or(1, |s| s.parse::<usize>().unwrap().max(1));
    let classes: Vec<(usize, usize, bool)> = CLASSES
        .iter()
        .map(|&(size, count, region)| ((size / scale).max(1), count, region))
        .collect();
    let n: usize = opt("--n").map_or(8_000_000 / scale, |s| s.parse().unwrap());
    let segments: usize = opt("--segments").map_or(4, |s| s.parse().unwrap());
    let fmt = opt("--format").unwrap_or_else(|| "ivf-f16".into());
    let ef: usize = opt("--ef").map_or(100, |s| s.parse().unwrap());
    let partition_clusters = args.iter().any(|a| a == "--partition-clusters");
    let nq: usize = opt("--queries").map_or(100, |s| s.parse().unwrap());
    let sort_mode = opt("--sort").unwrap_or_else(|| "tenant".into());
    const SUBS: u64 = 16;
    let sub_of = |i: usize| Rng(i as u64 ^ 0x5DEE_CE66).next() % SUBS;
    let (_, dim, mut src) = read_bin(&format!("{prefix}.base.fbin"));
    let (nq_all, _, mut queries) = read_bin(&format!("{prefix}.query.fbin"));
    src.chunks_exact_mut(dim)
        .chain(queries.chunks_exact_mut(dim))
        .for_each(normalize);
    let nsrc = src.len() / dim;

    // The data set, made once and kept: vectors, then each one's tenant.
    let base_path = format!("{out}.{n}.base.fbin");
    let tenant_path = format!("{out}.{n}.tenants.u32");
    let anchors_from = nq_all - 64;
    let mut tenants_meta: Vec<(usize, bool, usize)> = Vec::new(); // (size, region, anchor query)
    for &(size, count, region) in &classes {
        for _ in 0..count {
            let anchor = if region {
                anchors_from + tenants_meta.iter().filter(|t| t.1).count()
            } else {
                0
            };
            tenants_meta.push((size, region, anchor));
        }
    }
    let assigned: usize = tenants_meta.iter().map(|t| t.0).sum();
    assert!(assigned <= n, "tenants hold {assigned} > {n}");
    let (base, tenant): (Vec<f32>, Vec<u32>) = if std::path::Path::new(&tenant_path).exists() {
        let (_, _, b) = read_bin(&base_path);
        let t = std::fs::read(&tenant_path)
            .unwrap()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (b, t)
    } else {
        let t0 = Instant::now();
        let mut base = vec![0f32; n * dim];
        base.par_chunks_exact_mut(dim)
            .enumerate()
            .for_each(|(i, v)| {
                let mut r = Rng(i as u64 * 0x9E37_79B9 + 17);
                let s = (r.next() % nsrc as u64) as usize;
                for (x, &y) in v.iter_mut().zip(&src[s * dim..(s + 1) * dim]) {
                    *x = y + 0.06 * r.gauss();
                }
                normalize(v);
            });
        let mut tenant = vec![u32::MAX; n];
        // Regions first, largest first: the nearest unassigned documents.
        for (t, &(size, region, anchor)) in tenants_meta.iter().enumerate() {
            if !region {
                continue;
            }
            let a = &queries[anchor * dim..(anchor + 1) * dim];
            let mut s: Vec<(f32, u32)> = base
                .par_chunks_exact(dim)
                .enumerate()
                .filter(|(i, _)| tenant[*i] == u32::MAX)
                .map(|(i, v)| (dot(v, a), i as u32))
                .collect();
            s.select_nth_unstable_by(size, |x, y| y.0.total_cmp(&x.0));
            for &(_, i) in &s[..size] {
                tenant[i as usize] = t as u32;
            }
        }
        let mut free: Vec<u32> = (0..n as u32)
            .filter(|&i| tenant[i as usize] == u32::MAX)
            .collect();
        let mut r = Rng(99);
        for i in (1..free.len()).rev() {
            free.swap(i, (r.next() % (i as u64 + 1)) as usize);
        }
        let mut at = 0;
        for (t, &(size, region, _)) in tenants_meta.iter().enumerate() {
            if region {
                continue;
            }
            for &i in &free[at..at + size] {
                tenant[i as usize] = t as u32;
            }
            at += size;
        }
        // The rest: one more tenant.
        let rest = tenants_meta.len() as u32;
        for &i in &free[at..] {
            tenant[i as usize] = rest;
        }
        let mut f = Vec::with_capacity(8 + base.len() * 4);
        f.extend_from_slice(&(n as u32).to_le_bytes());
        f.extend_from_slice(&(dim as u32).to_le_bytes());
        for x in &base {
            f.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(&base_path, f).unwrap();
        std::fs::write(
            &tenant_path,
            tenant
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
        eprintln!("data set {n} x {dim} in {:.0}s", t0.elapsed().as_secs_f64());
        (base, tenant)
    };

    // Segments: a random split, each grouped by tenant.
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut r = Rng(7);
    for i in (1..order.len()).rev() {
        order.swap(i, (r.next() % (i as u64 + 1)) as usize);
    }
    let per = n.div_ceil(segments);
    for chunk in order.chunks_mut(per) {
        chunk.sort_by_key(|&i| (tenant[i as usize], i));
    }
    let block = format!("{}{sort_mode}", if partition_clusters { "p" } else { "" });
    let path = std::path::PathBuf::from(opt("--dir").unwrap_or_else(|| "/tmp/knn-tenants".into()))
        .join(format!("{n}-{segments}-{fmt}-b{block}"));
    let marker = path.join("built");
    let built = marker.exists();
    if !built {
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
    }
    let dir = FsDirectory::open(&path);
    if !built {
        let t0 = Instant::now();
        let version = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
        w.enable_explicit_documents().unwrap();
        let keyword = |f: &str| IndexSortField {
            field: f.to_string(),
            reverse: false,
            kind: IndexSortKind::String(StringMissingValue::None),
        };
        let sort = match sort_mode.as_str() {
            "tenant" => vec![keyword("t")],
            "composite" => vec![keyword("t"), keyword("s")],
            "time" => vec![IndexSortField::long("p", false, None)],
            o => panic!("{o}"),
        };
        w.set_index_sort(Some(&sort)).unwrap();
        w.set_ivf_partition_clusters(partition_clusters.then_some(20_000));
        w.register_field(FieldInfo::new("id", 0)).unwrap();
        let keyword_field = |name: &str| {
            FieldInfo::new(name, 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true)
                .with_doc_values(DocValuesType::Sorted, DocValuesSkipIndexType::None, -1)
        };
        // Numbered in registration order: id 0, t 1, v 2, p 3, s 4.
        w.register_field(keyword_field("t")).unwrap();
        w.register_field(FieldInfo::new("v", 0).with_vectors(
            dim as i32,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Cosine,
        ))
        .unwrap();
        w.register_field(FieldInfo::new("p", 3).with_doc_values(
            DocValuesType::Numeric,
            DocValuesSkipIndexType::None,
            -1,
        ))
        .unwrap();
        w.register_field(keyword_field("s")).unwrap();
        w.set_knn_vectors_format_for_field(
            "v",
            match fmt.as_str() {
                "ivf-f16" => KnnVectorsFormat::ivf(384, 1).unwrap(),
                "hnsw" => KnnVectorsFormat::hnsw(16, 100).unwrap(),
                o => panic!("{o}"),
            },
        );
        w.set_ram_buffer_size_mb(16_384.0).unwrap();
        w.set_merge_policy(None);
        let w = ConcurrentIndexWriter::new(w, 1).unwrap();
        for chunk in order.chunks(per) {
            // One at a time, built as added: a list of documents is a block,
            // which a sorted index takes only with a parent field.
            for (d, v) in chunk.iter().map(|&i| {
                let i = i as usize;
                let mut d = ExplicitDocument::default();
                d.stored.push(StoredField {
                    field_number: 0,
                    value: FieldValue::Int(i as i32),
                });
                let (t, sub) = (format!("t{}", tenant[i]), format!("s{}", sub_of(i)));
                for (number, value) in [(1, t), (4, sub)] {
                    let mut terms = InvertedTerms::default();
                    terms.push::<Vec<u8>>(value.as_bytes(), 1, &[], &[], &[]);
                    d.fields.doc_values.push(StoredField {
                        field_number: number,
                        value: FieldValue::Binary(value.into_bytes()),
                    });
                    d.fields.inverted.push(InvertedField {
                        field_number: number,
                        terms,
                        norm: None,
                    });
                }
                // A time: random, unrelated to the tenant.
                d.fields.doc_values.push(StoredField {
                    field_number: 3,
                    value: FieldValue::Long((Rng(i as u64 + 3).next() >> 20) as i64),
                });
                (
                    d,
                    vec![DocumentVector::float32(
                        "v",
                        base[i * dim..(i + 1) * dim].to_vec(),
                    )],
                )
            }) {
                w.add_explicit_documents_with_vectors_in(None, vec![d], vec![v])
                    .unwrap();
            }
            w.flush().unwrap();
            eprintln!("segment at {:.0}s", t0.elapsed().as_secs_f64());
        }
        w.commit().unwrap();
        w.into_writer().unwrap().close().unwrap();
        std::fs::write(&marker, "").unwrap();
        let bytes: u64 = std::fs::read_dir(&path)
            .unwrap()
            .filter_map(|e| e.ok()?.metadata().ok())
            .map(|m| m.len())
            .sum();
        eprintln!(
            "built {fmt} {n} block {block:?} in {:.0}s, {:.0} MB",
            t0.elapsed().as_secs_f64(),
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
    // One measured tenant per class and the unfiltered search.
    let mut measured: Vec<(String, Option<u32>, Vec<usize>)> =
        vec![("none".into(), None, (0..nq).collect())];
    let with_subs = |m: Vec<(String, Option<u32>, Vec<usize>)>| {
        m.into_iter()
            .flat_map(|(l, t, q)| {
                let sub = t.map(|_| (format!("{l}+sub"), t, q.clone()));
                std::iter::once((l, t, q)).chain(sub)
            })
            .collect::<Vec<_>>()
    };
    let mut first = 0;
    for &(size, count, region) in &classes {
        let t = first as u32;
        let kind = if region { "region" } else { "random" };
        measured.push((format!("{kind}{size}-any"), Some(t), (0..nq).collect()));
        if region {
            let a = tenants_meta[first].2;
            let anchor = &queries[a * dim..(a + 1) * dim];
            let mut s: Vec<(f32, usize)> = (0..anchors_from)
                .map(|q| (dot(&queries[q * dim..(q + 1) * dim], anchor), q))
                .collect();
            s.sort_unstable_by(|x, y| y.0.total_cmp(&x.0));
            measured.push((
                format!("{kind}{size}-local"),
                Some(t),
                s.iter().take(nq).map(|p| p.1).collect(),
            ));
        }
        first += count;
    }
    let measured = with_subs(measured);
    println!("n,format,block,tenant,docs,recall@{K},min_q,mean_us,p50_us,p99_us");
    let only = std::env::var("ONLY").ok();
    for (label, t, qs) in &measured {
        if only
            .as_ref()
            .is_some_and(|o| !o.split(',').any(|x| x == label))
        {
            continue;
        }
        let on_sub = label.ends_with("+sub");
        let members: Vec<usize> = match t {
            Some(t) => (0..n)
                .filter(|&i| tenant[i] == *t && (!on_sub || sub_of(i) == 3))
                .collect(),
            None => (0..n).collect(),
        };
        let truth: Vec<Vec<i32>> = qs
            .par_iter()
            .map(|&q| {
                let qv = &queries[q * dim..(q + 1) * dim];
                let mut s: Vec<(f32, usize)> = members
                    .iter()
                    .map(|&i| (dot(&base[i * dim..(i + 1) * dim], qv), i))
                    .collect();
                let k = K.min(s.len());
                if k < s.len() {
                    s.select_nth_unstable_by(k, |a, b| b.0.total_cmp(&a.0));
                }
                s[..k].iter().map(|p| p.1 as i32).collect()
            })
            .collect();
        let clause = t.map(|t| {
            let tenant = Clause::Term(TermQuery::new("t", format!("t{t}").as_bytes()));
            if on_sub {
                Clause::Boolean(Box::new(BooleanQuery {
                    filter: vec![tenant, Clause::Term(TermQuery::new("s", b"s3".as_slice()))],
                    ..BooleanQuery::default()
                }))
            } else {
                tenant
            }
        });
        let search = |q: usize| {
            knn_top_hits_with(
                &segs,
                "v",
                &KnnTarget::Float(queries[q * dim..(q + 1) * dim].to_vec()),
                K,
                ef,
                clause.as_ref(),
                0,
            )
            .unwrap()
        };
        for &q in qs.iter().take(20) {
            search(q);
        }
        let (mut found, mut total, mut worst) = (0usize, 0usize, 1f64);
        let mut times = Vec::new();
        for (qi, &q) in qs.iter().enumerate() {
            let t0 = Instant::now();
            let hits = search(q);
            times.push(t0.elapsed().as_secs_f64() * 1e6);
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
        println!(
            "{n},{fmt},{block},{label},{},{:.4},{worst:.2},{:.0},{:.0},{:.0}",
            members.len(),
            found as f64 / total.max(1) as f64,
            times.iter().sum::<f64>() / m as f64,
            times[m / 2],
            times[(m * 99 / 100).min(m - 1)]
        );
    }
}
