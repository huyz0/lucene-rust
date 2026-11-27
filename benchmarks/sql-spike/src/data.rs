//! Synthetic corpus: `id` (unique key), `body` (Zipf text), `price` (long doc
//! value), `cat` (keyword, 100 values) and `vec` (32-d float vector).

use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_index::document::{
    Document, KeywordField, KnnFloatVectorField, NumericDocValuesField, Store, StringField,
    TextField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

use crate::VEC_DIM;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

pub fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

pub fn unit(state: &mut u64) -> f64 {
    (next(state) >> 11) as f64 / (1u64 << 53) as f64
}

pub fn id_of(i: usize) -> String {
    format!("doc-{i:08}")
}

pub fn random_vec(state: &mut u64) -> Vec<f32> {
    (0..VEC_DIM)
        .map(|_| (unit(state) + unit(state) + unit(state) - 1.5) as f32)
        .collect()
}

/// Writes `docs` documents in `segments` commits (no merge).
pub fn build(path: &str, docs: usize, segments: usize) {
    let dir = FsDirectory::open(path);
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_ram_buffer_size_mb(512.0).unwrap();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let per = docs.div_ceil(segments);
    let t = std::time::Instant::now();
    for i in 0..docs {
        let mut d = Document::new();
        d.add(StringField::new("id", id_of(i), Store::Yes));
        let body: Vec<String> = (0..40)
            .map(|_| format!("w{}", (50_000f64.powf(unit(&mut state))) as u64))
            .collect();
        d.add(TextField::new("body", body.join(" "), Store::No));
        d.add(NumericDocValuesField::new(
            "price",
            (next(&mut state) % 10_000) as i64,
        ));
        d.add(KeywordField::new(
            "cat",
            format!("c{:03}", next(&mut state) % 100),
            Store::No,
        ));
        d.add(
            KnnFloatVectorField::new(
                "vec",
                random_vec(&mut state),
                VectorSimilarityFunction::Euclidean,
            )
            .unwrap(),
        );
        w.add_fields_document(&d).unwrap();
        if (i + 1) % per == 0 {
            w.commit().unwrap();
        }
    }
    w.commit().unwrap();
    eprintln!(
        "indexed {docs} docs in {segments} commits in {:.1?}",
        t.elapsed()
    );
}
