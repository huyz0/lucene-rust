//! A small deterministic index and helpers shared by the tests.
#![allow(dead_code)]

use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::util::display::array_value_to_string;
use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    Document, KeywordField, KnnFloatVectorField, LongField, NumericDocValuesField, Store,
    StringField, TextField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_sql::{Snapshot, SqlConfig, SqlEngine, TableSpec};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

pub const DIM: usize = 8;
pub const CATS: u64 = 12;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

pub fn id_of(i: usize) -> String {
    format!("doc-{i:05}")
}

pub struct Doc {
    pub id: String,
    pub body: String,
    pub price: i64,
    pub cat: String,
    pub vec: Vec<f32>,
}

pub fn corpus(n: usize) -> Vec<Doc> {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    (0..n)
        .map(|i| {
            let words: Vec<String> = (0..12)
                .map(|_| format!("w{}", (400f64.powf(rng.unit())) as u64))
                .collect();
            Doc {
                id: id_of(i),
                body: words.join(" "),
                price: (rng.next() % 1000) as i64,
                cat: format!("c{:02}", rng.next() % CATS),
                vec: (0..DIM).map(|_| (rng.unit() * 2.0 - 1.0) as f32).collect(),
            }
        })
        .collect()
}

pub fn spec() -> TableSpec {
    TableSpec::new("t")
        .key("id")
        .long("price")
        .keyword("cat")
        .text("body")
        .vector("vec", DIM)
}

pub struct Fixture {
    pub _dir: TempDir,
    pub docs: Vec<Doc>,
    pub deleted: Vec<usize>,
    pub engine: SqlEngine,
}

impl Fixture {
    pub fn snapshot(&self) -> &Arc<Snapshot> {
        self.engine.snapshot()
    }

    pub fn live(&self) -> impl Iterator<Item = &Doc> {
        self.docs
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.deleted.contains(i))
            .map(|(_, d)| d)
    }
}

/// `n` documents in `segments` commits; `delete` are deleted by key after the
/// last commit and committed again.
pub fn fixture(n: usize, segments: usize, delete: &[usize]) -> Fixture {
    fixture_with(n, segments, delete, spec(), &SqlConfig::default())
}

pub fn fixture_with(
    n: usize,
    segments: usize,
    delete: &[usize],
    spec: TableSpec,
    config: &SqlConfig,
) -> Fixture {
    fixture_docs(corpus(n), segments, delete, spec, config)
}

/// As [`fixture_with`] over `docs`; a document with an empty `cat` has no
/// `cat` field.
pub fn fixture_docs(
    docs: Vec<Doc>,
    segments: usize,
    delete: &[usize],
    spec: TableSpec,
    config: &SqlConfig,
) -> Fixture {
    build(docs, segments, delete, spec, config, false)
}

/// As [`fixture`] with `price` indexed as OpenSearch does: a point and a
/// sorted-numeric doc value.
pub fn fixture_points(n: usize, segments: usize, delete: &[usize]) -> Fixture {
    build(
        corpus(n),
        segments,
        delete,
        spec(),
        &SqlConfig::default(),
        true,
    )
}

fn build(
    docs: Vec<Doc>,
    segments: usize,
    delete: &[usize],
    spec: TableSpec,
    config: &SqlConfig,
    points: bool,
) -> Fixture {
    let dir = TempDir::new("sql");
    let n = docs.len();
    {
        let fs = FsDirectory::open(dir.path_str());
        let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", VERSION).unwrap();
        let per = n.div_ceil(segments);
        for (i, d) in docs.iter().enumerate() {
            let mut doc = Document::new();
            doc.add(StringField::new("id", d.id.clone(), Store::Yes));
            doc.add(TextField::new("body", d.body.clone(), Store::No));
            if points {
                doc.add(LongField::new("price", d.price, Store::No));
            } else {
                doc.add(NumericDocValuesField::new("price", d.price));
            }
            if !d.cat.is_empty() {
                doc.add(KeywordField::new("cat", d.cat.clone(), Store::No));
            }
            doc.add(
                KnnFloatVectorField::new("vec", d.vec.clone(), VectorSimilarityFunction::Euclidean)
                    .unwrap(),
            );
            w.add_fields_document(&doc).unwrap();
            if (i + 1) % per == 0 {
                w.commit().unwrap();
            }
        }
        w.commit().unwrap();
        if !delete.is_empty() {
            let terms: Vec<Term> = delete.iter().map(|&i| Term::new("id", id_of(i))).collect();
            w.delete_documents_by_term(&terms).unwrap();
            w.commit().unwrap();
        }
    }
    let reader = DirectoryReader::open(&FsDirectory::open(dir.path_str())).unwrap();
    let snapshot = Arc::new(Snapshot::new(reader, spec).unwrap());
    Fixture {
        _dir: dir,
        docs,
        deleted: delete.to_vec(),
        engine: SqlEngine::new(snapshot, config).unwrap(),
    }
}

pub fn rows(batches: &[RecordBatch]) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            out.push(
                b.columns()
                    .iter()
                    .map(|c| array_value_to_string(c, r).unwrap())
                    .collect(),
            );
        }
    }
    out
}

pub fn sorted(mut r: Vec<Vec<String>>) -> Vec<Vec<String>> {
    r.sort();
    r
}

pub fn vec_text(v: &[f32]) -> String {
    v.iter().map(f32::to_string).collect::<Vec<_>>().join(",")
}

pub fn run(f: &Fixture, sql: &str) -> Vec<Vec<String>> {
    rows(&f.engine.query_blocking(sql).unwrap())
}

pub fn standard(f: &Fixture, sql: &str) -> Vec<Vec<String>> {
    rows(&f.engine.standard_blocking(sql).unwrap())
}
