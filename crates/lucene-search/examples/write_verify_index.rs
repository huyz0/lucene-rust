//! M3's end-to-end write-path proof (T3.4), Rust half: build a realistic
//! multi-segment index through the real `IndexWriter`, then record what this
//! port's searcher answers over it, so `fixtures/src/VerifyIndex.java` can
//! check real Lucene reads the same index and answers the same.
//!
//! Writes, under `<out>`:
//!
//! - `index/` -- 120 000 documents in five segments, written by
//!   `lucene_index::index_writer::IndexWriter` (the path through
//!   `indexing_chain.rs` and `segment_writer.rs`, not hand-assembled files).
//!   Fields: `id` (stored, and indexed as one unique term per document:
//!   every term a singleton), `body` (Zipf-distributed text over a 5 000-term
//!   vocabulary, plus a word unique to its document in one document of eight;
//!   positions, offsets, payloads and norms), `keyword` (one Zipf-drawn token
//!   per document, `IndexOptions::Docs`), `num` (numeric doc values). The
//!   commonest `body` terms occur in most documents, so their postings cross
//!   full 256-document blocks and level-1 skip spans in every segment; the
//!   unique words and the ids are singletons, which the Zipf vocabulary alone
//!   never produces at this size -- and singletons are what `encodeTerm`
//!   delta-codes (an off-by-one there went unnoticed until they were added).
//! - `postings.tsv` -- the **expected** postings of every `body`,
//!   `keyword` and `id` term, computed from the generated text itself, not read back
//!   through this port's writer or reader: `field, term, docFreq,
//!   totalTermFreq, hash`, where `hash` is FNV-1a over every document id,
//!   freq, position, start and end offset and payload (for `keyword` and
//!   `id`, the document ids only). Java walks every term of the index and must
//!   reproduce every line (T3.1).
//! - `queries.tsv` / `rust-results.tsv` -- 59 queries (term, boolean
//!   conjunction and disjunction, cross-field, phrase, doc-values range) and
//!   this port's top 50 for each: `id, doc:score,...` in rank order (for
//!   the range queries, `doc:value`, the sort key). Java runs the same queries with `IndexSearcher` and BM25 and
//!   requires the same documents in the same order and every score within
//!   1e-5.
//!
//! Usage: `cargo run -p lucene-search --example write_verify_index -- <out>`.
// Example code: see `docs/arithmetic-gate.md`'s "Test code" section.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;
use std::fmt::Write as _;

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::collector::{ScoreDoc, SortDirection};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::doc_value_query::MissingValue;
use lucene_search::multi_segment::{
    search_numeric_range_sorted_by_field_multi_segment, DocValueSegment,
};
use lucene_search::query::{BooleanQuery, Clause, PhraseQuery, TermQuery};
use lucene_search::{search_boolean_query_multi_segment, search_term_query_multi_segment};
use lucene_store::{FsDirectory, MmapDirectory};

/// Must match `VerifyIndex.java`.
const NUM_DOCS: usize = 120_000;
/// Five segments: 25 000 buffered documents per flush, and no merging.
const DOCS_PER_SEGMENT: i32 = 25_000;
const VOCABULARY: usize = 5_000;
const KEYWORDS: usize = 300;
const TOP_N: usize = 50;

const F_ID: i32 = 0;
const F_BODY: i32 = 1;
const F_KEYWORD: i32 = 2;
const F_NUM: i32 = 3;

fn field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: true,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: IndexOptions::None,
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Byte,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

/// xorshift64, fixed seed: the corpus is the same on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Draws ranks `0..n` with probability proportional to `1 / (rank + 1)`.
struct Zipf(Vec<f64>);

impl Zipf {
    fn new(n: usize) -> Self {
        let mut cdf = Vec::with_capacity(n);
        let mut sum = 0.0;
        for r in 0..n {
            sum += 1.0 / (r as f64 + 1.0);
            cdf.push(sum);
        }
        for c in &mut cdf {
            *c /= sum;
        }
        Zipf(cdf)
    }

    fn draw(&self, rng: &mut Rng) -> usize {
        let u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
        self.0.partition_point(|&c| c < u).min(self.0.len() - 1)
    }
}

/// Document `doc`'s `id`: stored, and indexed as its one term -- every
/// term of the field a singleton, the shape `encodeTerm` delta-codes.
fn id(doc: usize) -> String {
    format!("doc{doc}")
}

fn term(rank: usize) -> String {
    format!("t{rank}")
}

/// The payload of `term` at `position`: depends on nothing else, so it is
/// the same whichever segment the document lands in (the payload source
/// sees segment-local document ids). Every seventh position has none.
fn payload(term: &str, position: i32) -> Vec<u8> {
    let seed = term.bytes().fold(position as u32, |h, b| {
        h.wrapping_mul(31).wrapping_add(b as u32)
    });
    let len = (seed % 7) as usize;
    (0..len).map(|i| (seed >> (i * 3)) as u8).collect()
}

struct Doc {
    body: String,
    keyword: String,
    num: i64,
}

fn corpus() -> Vec<Doc> {
    let mut rng = Rng(0x5eed_1234_abcd_ef01);
    let words = Zipf::new(VOCABULARY);
    let keywords = Zipf::new(KEYWORDS);
    (0..NUM_DOCS)
        .map(|doc| {
            let len = 4 + (rng.next() % 37) as usize;
            let mut tokens: Vec<String> = (0..len).map(|_| term(words.draw(&mut rng))).collect();
            // One document in eight carries a word no other document has:
            // singleton terms *with* positions, offsets and payloads, which
            // the Zipf vocabulary alone never produces at this size.
            if rng.next().is_multiple_of(8) {
                let at = (rng.next() % (len as u64 + 1)) as usize;
                tokens.insert(at, format!("u{doc}"));
            }
            let body = tokens.join(" ");
            let keyword = format!("k{}", keywords.draw(&mut rng));
            let num = (rng.next() % 1_000_000) as i64;
            Doc { body, keyword, num }
        })
        .collect()
}

/// FNV-1a, 64-bit, over little-endian `i32`s and raw bytes. `VerifyIndex.java`
/// has the same function.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn int(&mut self, v: i32) {
        self.bytes(&v.to_le_bytes());
    }
}

/// One occurrence: position, start and end offset.
type Occurrence = (i32, i32, i32);

/// The expected inverted index, straight from the text: whitespace-separated
/// ASCII tokens, which the standard analyzer leaves as they are.
fn expected_postings(docs: &[Doc]) -> String {
    let mut body: BTreeMap<&str, Vec<(i32, Vec<Occurrence>)>> = BTreeMap::new();
    let mut keyword: BTreeMap<&str, Vec<i32>> = BTreeMap::new();
    for (doc_id, d) in docs.iter().enumerate() {
        let doc_id = doc_id as i32;
        let mut offset = 0i32;
        let mut per_doc: BTreeMap<&str, Vec<Occurrence>> = BTreeMap::new();
        for (position, token) in d.body.split(' ').enumerate() {
            let end = offset + token.len() as i32;
            per_doc
                .entry(token)
                .or_default()
                .push((position as i32, offset, end));
            offset = end + 1;
        }
        for (token, occurrences) in per_doc {
            body.entry(token).or_default().push((doc_id, occurrences));
        }
        keyword.entry(&d.keyword).or_default().push(doc_id);
    }
    let ids: Vec<String> = (0..docs.len()).map(id).collect();
    let mut id_postings: BTreeMap<&str, i32> = BTreeMap::new();
    for (doc_id, id) in ids.iter().enumerate() {
        id_postings.insert(id, doc_id as i32);
    }
    let mut out = String::new();
    for (token, postings) in &body {
        let mut h = Fnv::new();
        let mut ttf = 0i64;
        for (doc, occurrences) in postings {
            h.int(*doc);
            h.int(occurrences.len() as i32);
            ttf += occurrences.len() as i64;
            for &(position, start, end) in occurrences {
                h.int(position);
                h.int(start);
                h.int(end);
                let p = payload(token, position);
                h.int(p.len() as i32);
                h.bytes(&p);
            }
        }
        writeln!(
            out,
            "body\t{token}\t{}\t{ttf}\t{:016x}",
            postings.len(),
            h.0
        )
        .unwrap();
    }
    for (token, docs) in &keyword {
        let mut h = Fnv::new();
        for &doc in docs {
            h.int(doc);
        }
        writeln!(out, "keyword\t{token}\t{}\t-1\t{:016x}", docs.len(), h.0).unwrap();
    }
    for (token, doc) in &id_postings {
        let mut h = Fnv::new();
        h.int(*doc);
        writeln!(out, "id\t{token}\t1\t-1\t{:016x}", h.0).unwrap();
    }
    out
}

fn write_index(dir_path: &str, docs: &[Doc]) {
    std::fs::create_dir_all(dir_path).expect("create index dir");
    let dir = FsDirectory::open(dir_path);
    let fields = vec![
        FieldInfo {
            index_options: IndexOptions::Docs,
            ..field("id", F_ID)
        },
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            store_payloads: true,
            omit_norms: false,
            ..field("body", F_BODY)
        },
        FieldInfo {
            index_options: IndexOptions::Docs,
            ..field("keyword", F_KEYWORD)
        },
        FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..field("num", F_NUM)
        },
    ];
    let mut writer = IndexWriter::open(
        &dir,
        fields,
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .expect("open writer");
    writer
        .set_max_buffered_docs(DOCS_PER_SEGMENT)
        .expect("max buffered docs");
    writer.set_ram_buffer_size_mb(4096.0).expect("ram buffer");
    writer.set_postings_field(Some("body")).expect("body");
    writer.add_postings_field("keyword").expect("keyword");
    writer.add_postings_field("id").expect("id");
    writer.set_doc_values_field(Some("num")).expect("num");
    writer
        .set_payload_source(Some(Box::new(|ctx| {
            let p = payload(ctx.term, ctx.position);
            (!p.is_empty()).then_some(p)
        })))
        .expect("payload source");
    for (i, d) in docs.iter().enumerate() {
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: F_ID,
                        value: FieldValue::String(id(i)),
                    },
                    StoredField {
                        field_number: F_BODY,
                        value: FieldValue::String(d.body.clone()),
                    },
                    StoredField {
                        field_number: F_KEYWORD,
                        value: FieldValue::String(d.keyword.clone()),
                    },
                    StoredField {
                        field_number: F_NUM,
                        value: FieldValue::Long(d.num),
                    },
                ],
            })
            .expect("add document");
    }
    let sis = writer.commit().expect("commit");
    assert!(
        sis.segments.len() >= 2,
        "the proof is about a multi-segment index; got {} segment(s)",
        sis.segments.len()
    );
}

/// One query of the set, as `queries.tsv` spells it: `id, kind, field,
/// args...` -- the same schema `benchmarks/queries.tsv` uses.
struct Query {
    id: String,
    kind: &'static str,
    field: &'static str,
    args: Vec<String>,
}

fn queries() -> Vec<Query> {
    let mut qs = Vec::new();
    let mut add = |kind: &'static str, field: &'static str, args: Vec<String>| {
        let id = format!("q{:02}", qs.len() + 1);
        qs.push(Query {
            id,
            kind,
            field,
            args,
        });
    };
    // Terms across the frequency range: most documents down to singletons.
    for r in [0, 1, 2, 5, 10, 30, 100, 300, 1000, 2500, 4000, 4999] {
        add("term", "body", vec![term(r)]);
    }
    for k in [0, 1, 7, 50, 299] {
        add("term", "keyword", vec![format!("k{k}")]);
    }
    for d in [0, 77_777, 119_999] {
        add("term", "id", vec![id(d)]);
    }
    for d in [8, 40_000, 119_992] {
        add("or", "body", vec![format!("u{d}"), term(3)]);
    }
    for pair in [
        (0, 1),
        (0, 10),
        (1, 2),
        (3, 50),
        (10, 100),
        (0, 999),
        (5, 6),
        (200, 201),
    ] {
        add("and", "body", vec![term(pair.0), term(pair.1)]);
    }
    add("and", "body", vec![term(0), term(1), term(2)]);
    add("and", "body", vec![term(2), term(20), term(200)]);
    for pair in [
        (0, 1),
        (10, 20),
        (100, 1000),
        (7, 4000),
        (2, 3),
        (3000, 4000),
    ] {
        add("or", "body", vec![term(pair.0), term(pair.1)]);
    }
    add("or", "body", vec![term(0), term(1), term(2), term(3)]);
    add(
        "or",
        "body",
        vec![term(50), term(500), term(1500), term(4500)],
    );
    // Cross-field: a body term that must match, and a keyword that must.
    for (t, k) in [(0, 0), (1, 5), (10, 1), (100, 2), (3, 40)] {
        add("and_kw", "body", vec![term(t), format!("k{k}")]);
    }
    for pair in [
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 1),
        (5, 0),
        (0, 10),
        (10, 3),
        (1, 1),
    ] {
        add("phrase", "body", vec![term(pair.0), term(pair.1)]);
    }
    add("phrase", "body", vec![term(0), term(1), term(2)]);
    for (lo, hi) in [
        (0, 10_000),
        (500_000, 500_500),
        (999_000, 999_999),
        (123_456, 234_567),
    ] {
        add("dv_range", "num", vec![lo.to_string(), hi.to_string()]);
    }
    qs
}

/// A scored hit as `rust-results.tsv` spells it: nine significant digits,
/// enough to tell two `f32`s apart.
fn scored(h: &ScoreDoc) -> (i32, String) {
    (h.doc_id, format!("{:.9e}", h.score))
}

fn run_queries(dir_path: &str, qs: &[Query]) -> String {
    let dir = MmapDirectory::open(dir_path.to_string());
    let reader = DirectoryReader::open(&dir).expect("open index");
    let opened = reader.open_segments().expect("open segments");
    let segments = opened.as_open_segments();
    let fields = vec!["body".to_string(), "keyword".to_string(), "id".to_string()];
    let by_field = reader.field_norms_by_field(&fields);
    let bool_norms: Vec<Option<&_>> = by_field.iter().map(Some).collect();

    let mut out = String::new();
    for q in qs {
        let term_q = |field: &str, t: &str| TermQuery {
            field: field.to_string(),
            term: t.as_bytes().to_vec(),
        };
        let hits: Vec<(i32, String)> = match q.kind {
            "term" => {
                let norms = reader.field_norms(q.field);
                let norms: Vec<Option<&_>> = norms.iter().map(Option::as_ref).collect();
                search_term_query_multi_segment(
                    &segments,
                    &term_q(q.field, &q.args[0]),
                    &norms,
                    TOP_N,
                )
                .expect("term")
                .iter()
                .map(scored)
                .collect()
            }
            "and" | "or" | "and_kw" | "phrase" => {
                let bq = match q.kind {
                    "and" => BooleanQuery {
                        must: q
                            .args
                            .iter()
                            .map(|t| Clause::Term(term_q("body", t)))
                            .collect(),
                        ..Default::default()
                    },
                    "or" => BooleanQuery {
                        should: q
                            .args
                            .iter()
                            .map(|t| Clause::Term(term_q("body", t)))
                            .collect(),
                        ..Default::default()
                    },
                    "and_kw" => BooleanQuery {
                        must: vec![
                            Clause::Term(term_q("body", &q.args[0])),
                            Clause::Term(term_q("keyword", &q.args[1])),
                        ],
                        ..Default::default()
                    },
                    _ => BooleanQuery {
                        must: vec![Clause::Phrase(PhraseQuery {
                            field: "body".to_string(),
                            terms: q.args.iter().map(|t| t.as_bytes().to_vec()).collect(),
                            slop: 0,
                        })],
                        ..Default::default()
                    },
                };
                search_boolean_query_multi_segment(&segments, &bq, &bool_norms, TOP_N)
                    .expect("boolean")
                    .iter()
                    .map(scored)
                    .collect()
            }
            "dv_range" => {
                let lo: i64 = q.args[0].parse().unwrap();
                let hi: i64 = q.args[1].parse().unwrap();
                let dv: Vec<DocValueSegment<'_>> = reader
                    .segment_readers()
                    .iter()
                    .zip(segments.iter())
                    .filter_map(|(r, s)| {
                        let data = r.doc_values_data()?;
                        let meta = r.doc_values_meta()?;
                        let num = r.field_infos().fields.iter().find(|f| f.name == "num")?;
                        let entry = meta.numeric_entry(num.number)?;
                        Some(DocValueSegment {
                            range_data: data,
                            range_entry: entry,
                            sort_data: data,
                            sort_entry: entry,
                            live_docs: s.live_docs,
                            max_doc: r.max_doc,
                            doc_base: s.doc_base,
                        })
                    })
                    .collect();
                search_numeric_range_sorted_by_field_multi_segment(
                    &dv,
                    lo,
                    hi,
                    SortDirection::Ascending,
                    MissingValue::Exclude,
                    TOP_N,
                )
                .expect("dv range")
                .into_iter()
                // The sort value in the score column, exactly, as Java's
                // `FieldDoc.fields[0]` carries it.
                .map(|h| (h.doc_id, h.value.to_string()))
                .collect()
            }
            other => unreachable!("{other}"),
        };
        let ranked: Vec<String> = hits.iter().map(|(doc, v)| format!("{doc}:{v}")).collect();
        writeln!(out, "{}\t{}", q.id, ranked.join(",")).unwrap();
    }
    out
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_verify_index <output-dir>");
    let index = format!("{out}/index");
    let docs = corpus();
    std::fs::create_dir_all(&out).expect("create output dir");
    std::fs::write(format!("{out}/postings.tsv"), expected_postings(&docs)).expect("postings");
    write_index(&index, &docs);

    let qs = queries();
    let mut spec = String::new();
    for q in &qs {
        writeln!(
            spec,
            "{}\t{}\t{}\t{}",
            q.id,
            q.kind,
            q.field,
            q.args.join("\t")
        )
        .unwrap();
    }
    std::fs::write(format!("{out}/queries.tsv"), spec).expect("queries");
    std::fs::write(format!("{out}/rust-results.tsv"), run_queries(&index, &qs)).expect("results");
    println!(
        "wrote {NUM_DOCS}-document index and {} queries to {out}",
        qs.len()
    );
}
