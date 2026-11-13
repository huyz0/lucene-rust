//! Single-thread latency of a Seismic (`neural_sparse` with
//! `method_parameters`) query on one clustered segment, and the hits' digest,
//! so two builds can be compared on the same index.
//!
//! ```text
//! cargo run --release -p lucene-sparse --example seismic_bench -- [docs] [queries] [dir]
//! ```
//!
//! Documents hold 64 distinct tokens drawn from a Zipf-like law over a 30 000
//! token vocabulary (SPLADE-shaped); queries hold 16. With `dir` the index is
//! written there on the first run and reused after.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::compound_reader::CompoundReader;
use lucene_index::index_writer::{
    ExplicitDocument, ExplicitFields, IndexWriter, InvertedField, InvertedTerm,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_sparse::search::{ForwardVectors, SegmentView};
use lucene_sparse::{postings, ClusteredPostings, SeismicField, SeismicQuery};
use lucene_store::directory::FsDirectory;
use lucene_store::Directory;
use std::sync::Arc;
use std::time::Instant;

const VOCAB: u32 = 30_000;
const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 2,
};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn unit(&mut self) -> f64 {
        f64::from(self.next()) / f64::from(1u32 << 31)
    }

    /// A token with probability falling as `1 / rank`.
    fn token(&mut self) -> u32 {
        ((f64::from(VOCAB)).powf(self.unit()) as u32).min(VOCAB - 1)
    }

    fn vector(&mut self, len: usize) -> Vec<(u32, f32)> {
        let mut v: Vec<(u32, f32)> = Vec::with_capacity(len);
        while v.len() < len {
            let t = self.token();
            if v.iter().all(|&(u, _)| u != t) {
                v.push((t, 0.05 + (self.next() % 1000) as f32 / 333.0));
            }
        }
        v
    }
}

fn attrs() -> Vec<(String, String)> {
    [
        ("sparse_vector_field", "true"),
        ("n_postings", "-1"),
        ("summary_prune_ratio", "0.4"),
        ("cluster_ratio", "0.1"),
        ("approximate_threshold", "1000"),
        ("quantization_ceiling_ingest", "3.0"),
        ("quantization_ceiling_search", "16.0"),
        ("engine", "lucene"),
        ("forward_index", "shared"),
        ("clustering_batch_size", "1"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

fn document(vector: &[(u32, f32)], emb: i32) -> ExplicitDocument {
    let mut forward = Vec::new();
    for &(t, w) in vector {
        forward.extend_from_slice(&(t as i32).to_be_bytes());
        forward.extend_from_slice(&w.to_be_bytes());
    }
    ExplicitDocument {
        stored: Vec::new(),
        fields: ExplicitFields {
            inverted: vec![InvertedField {
                field_number: emb,
                terms: vector
                    .iter()
                    .map(|&(t, w)| InvertedTerm {
                        term: t.to_string().into_bytes(),
                        freq: (w.to_bits() >> 15) as i32,
                        positions: Vec::new(),
                        offsets: Vec::new(),
                        payloads: Vec::new(),
                    })
                    .collect::<Vec<_>>()
                    .into(),
                norm: None,
            }],
            doc_values: vec![StoredField {
                field_number: emb,
                value: FieldValue::Binary(forward),
            }],
            ..ExplicitFields::default()
        },
    }
}

fn build(dir: &FsDirectory, docs: usize) {
    let mut w = IndexWriter::open(dir, Vec::new(), lucene_sparse::CODEC_NAME, VERSION).unwrap();
    w.set_codec(Some(Arc::new(lucene_sparse::SparseCodec)))
        .unwrap();
    w.set_use_compound_file(false);
    w.enable_explicit_documents().unwrap();
    let emb = w
        .register_field(
            FieldInfo::new("emb", 0)
                .with_index_options(IndexOptions::DocsAndFreqs)
                .with_omit_norms(true)
                .with_doc_values(DocValuesType::Binary, DocValuesSkipIndexType::None, -1)
                .with_attributes(attrs()),
        )
        .unwrap();
    let mut r = Lcg(7);
    let start = Instant::now();
    for chunk in 0..docs.div_ceil(10_000) {
        let n = (docs - chunk * 10_000).min(10_000);
        let batch = (0..n).map(|_| document(&r.vector(64), emb)).collect();
        w.add_explicit_documents(batch).unwrap();
    }
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    eprintln!(
        "built {docs} docs in {:.1} s",
        start.elapsed().as_secs_f64()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let docs: usize = args.get(1).map_or(200_000, |a| a.parse().unwrap());
    let queries: usize = args.get(2).map_or(2000, |a| a.parse().unwrap());
    let tmp;
    let path = match args.get(3) {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            tmp = lucene_util::test_support::TempDir::new("seismic-bench");
            tmp.path().to_path_buf()
        }
    };
    std::fs::create_dir_all(&path).unwrap();
    lucene_sparse::register_codec();
    let dir = FsDirectory::open(&path);
    if !dir
        .list_all()
        .unwrap()
        .iter()
        .any(|f| f.starts_with("segments_"))
    {
        build(&dir, docs);
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    let seg = &reader.segment_readers()[0];
    let info = seg
        .field_infos()
        .fields
        .iter()
        .find(|f| f.name == "emb")
        .unwrap();
    let field = SeismicField::from_field_info(info).unwrap().unwrap();
    let name = &seg.segment_name;
    let id = seg.segment_id();
    let compound = dir
        .list_all()
        .unwrap()
        .contains(&format!("{name}.cfs"))
        .then(|| CompoundReader::open(&dir, name, &id).unwrap());
    let files: &dyn Directory = match &compound {
        Some(c) => c,
        None => &dir,
    };
    let sit = files
        .open(&format!("{name}.{}", postings::TERMS_EXTENSION))
        .unwrap();
    let sip = Arc::new(
        files
            .open(&format!("{name}.{}", postings::POSTINGS_EXTENSION))
            .unwrap(),
    );
    let postings = ClusteredPostings::open(&sit, sip, &id).unwrap();
    let (meta, data) = seg.doc_values_for_field(field.number).unwrap();
    let entry = meta.binary_entry(field.number).unwrap();
    let forward = ForwardVectors::read(data, entry, seg.max_doc, field.ceiling_ingest).unwrap();
    let view = SegmentView {
        max_doc: seg.max_doc,
        live_docs: seg.live_docs(),
        forward: Some(&forward),
        postings: &postings,
    };
    let mut r = Lcg(99);
    let qs: Vec<SeismicQuery> = (0..queries + 200)
        .map(|_| SeismicQuery::new(&r.vector(16), 10, 10, 1.0, 1.0, &field).unwrap())
        .collect();
    let mut digest = 0u64;
    let mut times = Vec::with_capacity(queries);
    for (i, q) in qs.iter().enumerate() {
        let t = Instant::now();
        let hits = lucene_sparse::search(&view, &field, q, None).unwrap();
        let took = t.elapsed();
        if i >= 200 {
            times.push(took.as_secs_f64() * 1e6);
            for (d, s) in hits {
                digest = digest
                    .wrapping_mul(31)
                    .wrapping_add(d as u64 ^ u64::from(s.to_bits()));
            }
        }
    }
    times.sort_by(f64::total_cmp);
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "docs,{},queries,{},mean_us,{mean:.1},p50_us,{:.1},p99_us,{:.1},digest,{digest:016x}",
        seg.max_doc,
        times.len(),
        times[times.len() / 2],
        times[times.len() * 99 / 100]
    );
}
