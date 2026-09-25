//! Writes a complete, non-toy index through this port's `IndexWriter`, runs a
//! fixed query set over it with this port's own searcher, and records both, for
//! `VerifyIndex` to open with real Lucene 10.5.0 and compare hit for hit.
//!
//! This is M3's headline check (task T3.4, see
//! `docs/milestones/m3-write-path-proven.md`). Every other write-path verifier
//! checks one property of a small segment. This one checks the property all of
//! them exist for: that an index this port writes *searches the same* in real
//! Lucene as in this port -- same `CheckIndex` verdict, same top-50 documents
//! in the same order, same BM25 scores.
//!
//! # The index
//!
//! - **120 000 documents**, flushed by the writer's own `maxBufferedDocs`
//!   trigger into several segments, across two commits, with a batch of
//!   deletes between them so earlier segments carry `.liv` files. Multi-segment
//!   matters for scoring: Lucene scores every leaf with reader-wide statistics,
//!   and a port that used per-segment ones would rank differently.
//! - **`body`**: Zipfian text (`s = 1.07` over 20 000 words), indexed with
//!   positions, offsets **and payloads**, with norms. The head of the
//!   distribution puts terms in tens of thousands of documents -- far past
//!   `BLOCK_SIZE` (256) and `LEVEL1_NUM_DOCS` (8192) -- so the bit-packed
//!   blocks, both skip levels and impacts are all written; the tail keeps
//!   singleton terms, which are pulsed into the term dictionary.
//! - **`title`**: short text with positions, no offsets, with norms.
//! - **`tag`**: a keyword (one token, `DOCS`, norms omitted), 12 values.
//! - **`id`**: a unique keyword per document, the delete key.
//! - **`num`**: a sparse `NUMERIC` doc-values column (every eleventh document
//!   has none) with repeated values, so a sort on it has ties to break.
//!
//! A payload is a function of the term and the position only (see
//! [`payload_for`]), never of the document, so the verifier can check any
//! occurrence it reads without a manifest.
//!
//! # Outputs, beside the index
//!
//! - `verify-queries.tsv`: `id <TAB> kind <TAB> field <TAB> args...`, the query
//!   set, which `VerifyIndex` parses into real Lucene queries.
//! - `verify-rust-results.tsv`: `id <TAB> hits <TAB> total <TAB> doc:value,...`
//!   -- this port's top 50 per query and its count of every match, `value` being the BM25 score (an `f32` printed
//!   in Rust's shortest round-trip form, which Java's `Float.parseFloat`
//!   reads back to the same bits) or, for a doc-values sort, the sort key.
//!
//! Neither file is a Lucene index file, so neither disturbs `DirectoryReader`
//! or `CheckIndex`.
//!
//! Usage: `write_verify_index_fixture <output-dir>`.
// Test-support code opts out of the arithmetic gate at the file boundary: the
// gate exists for values read off disk in production decode paths, not for a
// fixture builder's own index arithmetic. See `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;
use std::fmt::Write as _;

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::collector::{ScoreDoc, SortDirection};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::doc_value_query::MissingValue;
use lucene_search::field_norms::FieldNorms;
use lucene_search::multi_segment::{
    search_boolean_query_multi_segment, search_numeric_range_sorted_by_field_multi_segment,
    search_term_query_multi_segment, DocValueSegment,
};
use lucene_search::query::{BooleanQuery, Clause, PhraseQuery, TermQuery};
use lucene_store::{FsDirectory, MmapDirectory};

const NUM_DOCS: usize = 120_000;
/// The writer flushes a segment every this many documents, so the index has
/// `NUM_DOCS / FLUSH_EVERY` segments or so without any explicit flush call.
const FLUSH_EVERY: i32 = 23_000;
/// The first commit lands here; the deletes are issued after it.
const FIRST_COMMIT_AT: usize = 70_000;
/// Every document whose number is a multiple of this is deleted.
const DELETE_EVERY: usize = 97;
const VOCAB: usize = 20_000;
const ZIPF_S: f64 = 1.07;
const TOP_N: usize = 50;
const TAGS: usize = 12;

const F_ID: i32 = 0;
const F_BODY: i32 = 1;
const F_TITLE: i32 = 2;
const F_TAG: i32 = 3;
const F_NUM: i32 = 4;

fn base_field(name: &str, number: i32) -> FieldInfo {
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

/// xorshift64*: deterministic, dependency-free, and good enough to shape a
/// corpus. The seed is fixed so the index -- and therefore the expected
/// results -- are the same on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Cumulative Zipf weights over `VOCAB` ranks; sampling is a binary search.
struct Zipf(Vec<f64>);

impl Zipf {
    fn new() -> Self {
        let mut acc = 0.0;
        let cdf = (1..=VOCAB)
            .map(|rank| {
                acc += 1.0 / (rank as f64).powf(ZIPF_S);
                acc
            })
            .collect::<Vec<_>>();
        Zipf(cdf)
    }

    fn sample(&self, rng: &mut Rng) -> usize {
        let target = rng.unit() * self.0[VOCAB - 1];
        self.0.partition_point(|&c| c < target).min(VOCAB - 1)
    }
}

fn word(rank: usize) -> String {
    format!("w{rank}")
}

/// Payload of one occurrence: a function of the term's text and the position
/// alone, so `VerifyIndex` can recompute it for any occurrence it reads.
/// Length cycles `0..=3`; length 0 is Lucene's "no payload".
fn payload_for(term: &str, position: i32) -> Option<Vec<u8>> {
    let len = (position as usize + term.len()) % 4;
    if len == 0 {
        return None;
    }
    let seed = term.bytes().fold(position as u32, |h, b| {
        h.wrapping_mul(31).wrapping_add(u32::from(b))
    });
    Some((0..len).map(|i| (seed >> (8 * i)) as u8).collect())
}

struct Corpus {
    rng: Rng,
    zipf: Zipf,
}

impl Corpus {
    fn body(&mut self, doc: usize) -> String {
        let len = 4 + self.rng.below(37);
        let mut words: Vec<String> = (0..len)
            .map(|_| word(self.zipf.sample(&mut self.rng)))
            .collect();
        // A planted phrase in a known fraction of documents, so the exact and
        // sloppy phrase queries below have a dense, non-accidental target.
        if doc.is_multiple_of(13) {
            let at = self.rng.below(words.len());
            words.insert(at, "quick".to_string());
            words.insert(at + 1, "brown".to_string());
            words.insert(at + 2, "fox".to_string());
        }
        words.join(" ")
    }

    fn title(&mut self) -> String {
        let len = 1 + self.rng.below(5);
        (0..len)
            .map(|_| word(self.zipf.sample(&mut self.rng) % 400))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn tag_for(doc: usize) -> String {
    // Skewed: tag0 on ~1/2 of documents, tag1 on ~1/4, and so on.
    let tz = (doc as u64 | (1 << (TAGS - 1))).trailing_zeros() as usize;
    format!("tag{tz}")
}

fn num_for(doc: usize) -> Option<i64> {
    (doc % 11 != 5).then(|| ((doc * 7919) % 5_000) as i64 - 2_500)
}

/// One query of the checked set; parsed identically by `VerifyIndex`.
struct Query {
    id: String,
    kind: &'static str,
    field: &'static str,
    args: Vec<String>,
}

fn query_set() -> Vec<Query> {
    let mut qs = Vec::new();
    let mut add = |kind: &'static str, field: &'static str, args: &[&str]| {
        qs.push(Query {
            id: format!("q{:02}", qs.len() + 1),
            kind,
            field,
            args: args.iter().map(|s| s.to_string()).collect(),
        });
    };
    // Term, across the whole frequency range: the head (bit-packed blocks,
    // level-1 skips, impacts pruning), the middle, the tail, a singleton, and
    // a term that does not exist.
    for t in [
        "w0",
        "w1",
        "w2",
        "w5",
        "w13",
        "w40",
        "w120",
        "w333",
        "w900",
        "w2500",
        "w7000",
        "w15000",
        "quick",
        "fox",
        "nosuchterm",
    ] {
        add("term", "body", &[t]);
    }
    for t in ["w0", "w7", "w150", "w399"] {
        add("term", "title", &[t]);
    }
    for t in ["tag0", "tag3", "tag11"] {
        add("term", "tag", &[t]);
    }
    // Conjunctions.
    add("and", "body", &["w0", "w1"]);
    add("and", "body", &["w0", "w1", "w2"]);
    add("and", "body", &["w3", "w50"]);
    add("and", "body", &["w1", "w700"]);
    add("and", "body", &["quick", "w0"]);
    add("and", "title", &["w0", "w1"]);
    // Disjunctions.
    add("or", "body", &["w0", "w1"]);
    add("or", "body", &["w4", "w90", "w1200"]);
    add("or", "body", &["w0", "w1", "w2", "w3"]);
    add("or", "body", &["w5000", "w6000", "w8000"]);
    add("or", "title", &["w2", "w30", "w300"]);
    // Must + must-not.
    add("not", "body", &["w1", "w0"]);
    add("not", "body", &["fox", "w2", "w3"]);
    // minimumShouldMatch.
    add("msm", "body", &["2", "w0", "w1", "w2", "w3"]);
    add("msm", "body", &["3", "w0", "w5", "w10", "w20", "w30"]);
    // Must + should: the should clauses only add score.
    add("mixed", "body", &["w2", "w0", "w1"]);
    add("mixed", "body", &["quick", "w0", "w9"]);
    // A scored term with a non-scoring keyword filter.
    add("filter", "body", &["w0", "tag", "tag2"]);
    add("filter", "body", &["w6", "tag", "tag0"]);
    // Phrases: exact and sloppy, planted and accidental.
    add("phrase", "body", &["0", "quick", "brown"]);
    add("phrase", "body", &["0", "quick", "brown", "fox"]);
    add("phrase", "body", &["0", "brown", "quick"]);
    add("phrase", "body", &["0", "w0", "w1"]);
    add("phrase", "body", &["0", "w1", "w0"]);
    add("phrase", "body", &["0", "w0", "w0"]);
    add("phrase", "body", &["1", "quick", "fox"]);
    add("phrase", "body", &["2", "fox", "quick"]);
    add("phrase", "body", &["2", "w0", "w2"]);
    add("phrase", "title", &["0", "w0", "w1"]);
    // Numeric doc-values ranges, sorted by the field: ties broken by doc id.
    add("dvrange", "num", &["-2500", "2500", "asc"]);
    add("dvrange", "num", &["-100", "100", "asc"]);
    add("dvrange", "num", &["0", "0", "asc"]);
    add("dvrange", "num", &["1000", "4000", "desc"]);
    add("dvrange", "num", &["-2500", "-2400", "desc"]);
    add("dvrange", "num", &["9999", "10000", "asc"]);
    qs
}

fn term(field: &str, t: &str) -> Clause {
    Clause::Term(TermQuery {
        field: field.to_string(),
        term: t.as_bytes().to_vec(),
    })
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_verify_index_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).expect("create output dir");
    write_index(&out_dir);
    run_queries(&out_dir);
}

fn write_index(out_dir: &str) {
    let dir = FsDirectory::open(out_dir);
    let fields = vec![
        FieldInfo {
            index_options: IndexOptions::Docs,
            ..base_field("id", F_ID)
        },
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            store_payloads: true,
            omit_norms: false,
            ..base_field("body", F_BODY)
        },
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            omit_norms: false,
            ..base_field("title", F_TITLE)
        },
        FieldInfo {
            index_options: IndexOptions::Docs,
            ..base_field("tag", F_TAG)
        },
        FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..base_field("num", F_NUM)
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
        .set_max_buffered_docs(FLUSH_EVERY)
        .expect("max buffered docs");
    writer
        .set_ram_buffer_size_mb(4096.0)
        .expect("ram buffer size");
    writer.set_postings_field(Some("body")).expect("body");
    for name in ["title", "tag", "id"] {
        writer.add_postings_field(name).expect("add postings field");
    }
    writer.set_doc_values_field(Some("num")).expect("num");
    writer
        .set_payload_source(Some(Box::new(|ctx| {
            assert_eq!(ctx.field, "body", "only body stores payloads");
            payload_for(ctx.term, ctx.position)
        })))
        .expect("payload source");

    let mut corpus = Corpus {
        rng: Rng(0x9E37_79B9_7F4A_7C15),
        zipf: Zipf::new(),
    };
    for doc in 0..NUM_DOCS {
        let mut fields = vec![
            StoredField {
                field_number: F_ID,
                value: FieldValue::String(format!("doc{doc}")),
            },
            StoredField {
                field_number: F_BODY,
                value: FieldValue::String(corpus.body(doc)),
            },
            StoredField {
                field_number: F_TITLE,
                value: FieldValue::String(corpus.title()),
            },
            StoredField {
                field_number: F_TAG,
                value: FieldValue::String(tag_for(doc)),
            },
        ];
        if let Some(n) = num_for(doc) {
            fields.push(StoredField {
                field_number: F_NUM,
                value: FieldValue::Long(n),
            });
        }
        writer
            .add_document(Document { fields })
            .expect("add document");
        if doc + 1 == FIRST_COMMIT_AT {
            writer.commit().expect("first commit");
            // Deletes reaching back into committed segments (a `.liv` file per
            // segment they touch) and into the buffered documents alike.
            let terms: Vec<Term> = (0..=doc)
                .step_by(DELETE_EVERY)
                .map(|d| Term::new("id", format!("doc{d}").into_bytes()))
                .collect();
            writer.delete_documents_by_term(&terms).expect("delete");
        }
    }
    let sis = writer.commit().expect("final commit");
    assert!(
        sis.segments.len() >= 3,
        "the fixture exists to test a multi-segment index, got {} segment(s)",
        sis.segments.len()
    );
    println!(
        "wrote {NUM_DOCS} documents in {} segments to {out_dir}",
        sis.segments.len()
    );
}

fn run_queries(out_dir: &str) {
    let dir = MmapDirectory::open(out_dir.to_string());
    let reader = DirectoryReader::open(&dir).expect("open index");
    let opened = reader.open_segments().expect("open segments");
    let segments = opened.as_open_segments();

    let text_fields = vec!["body".to_string(), "title".to_string(), "tag".to_string()];
    let norms_by_seg: Vec<HashMap<String, FieldNorms<'_>>> =
        reader.field_norms_by_field(&text_fields);
    let bool_norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> =
        norms_by_seg.iter().map(Some).collect();

    let queries = query_set();
    let mut qtsv = String::from("# written by write_verify_index_fixture; see VerifyIndex.java\n");
    let mut rtsv = String::new();
    for q in &queries {
        writeln!(
            qtsv,
            "{}\t{}\t{}\t{}",
            q.id,
            q.kind,
            q.field,
            q.args.join("\t")
        )
        .unwrap();
        let hits: Vec<(i32, String)> = match q.kind {
            "dvrange" => dv_range(&reader, &segments, q, TOP_N)
                .into_iter()
                .map(|(d, v)| (d, v.to_string()))
                .collect(),
            _ => scored(&segments, &reader, &bool_norms, q, TOP_N)
                .into_iter()
                .map(|h| (h.doc_id, format!("{}", h.score)))
                .collect(),
        };
        let joined = hits
            .iter()
            .map(|(d, v)| format!("{d}:{v}"))
            .collect::<Vec<_>>()
            .join(",");
        // Every match, not only the top 50: a defect that loses or invents a
        // document below rank 50 changes this and nothing else.
        let total = match q.kind {
            "dvrange" => dv_range(&reader, &segments, q, NUM_DOCS).len(),
            _ => scored(&segments, &reader, &bool_norms, q, NUM_DOCS).len(),
        };
        writeln!(rtsv, "{}\t{}\t{}\t{}", q.id, hits.len(), total, joined).unwrap();
    }
    let base = std::path::Path::new(out_dir);
    std::fs::write(base.join("verify-queries.tsv"), qtsv).expect("write queries");
    std::fs::write(base.join("verify-rust-results.tsv"), rtsv).expect("write results");
    println!("ran {} queries", queries.len());
}

fn scored(
    segments: &[lucene_search::multi_segment::OpenSegment<'_>],
    reader: &DirectoryReader,
    bool_norms: &[Option<&HashMap<String, FieldNorms<'_>>>],
    q: &Query,
    top_n: usize,
) -> Vec<ScoreDoc> {
    let f = q.field;
    let a = &q.args;
    let bq = match q.kind {
        "term" => {
            // The dedicated single-term path, not a one-clause boolean: it is
            // the one a plain `TermQuery` takes, impacts pruning included.
            let norms = reader.field_norms(f);
            let norms: Vec<Option<&FieldNorms<'_>>> = norms.iter().map(Option::as_ref).collect();
            let tq = TermQuery {
                field: f.to_string(),
                term: a[0].as_bytes().to_vec(),
            };
            return search_term_query_multi_segment(segments, &tq, &norms, top_n)
                .expect("term query");
        }
        "and" => BooleanQuery {
            must: a.iter().map(|t| term(f, t)).collect(),
            ..Default::default()
        },
        "or" => BooleanQuery {
            should: a.iter().map(|t| term(f, t)).collect(),
            ..Default::default()
        },
        "not" => BooleanQuery {
            must: vec![term(f, &a[0])],
            must_not: a[1..].iter().map(|t| term(f, t)).collect(),
            ..Default::default()
        },
        "msm" => BooleanQuery {
            should: a[1..].iter().map(|t| term(f, t)).collect(),
            minimum_should_match: a[0].parse().expect("msm count"),
            ..Default::default()
        },
        "mixed" => BooleanQuery {
            must: vec![term(f, &a[0])],
            should: a[1..].iter().map(|t| term(f, t)).collect(),
            ..Default::default()
        },
        "filter" => BooleanQuery {
            must: vec![term(f, &a[0])],
            filter: vec![term(&a[1], &a[2])],
            ..Default::default()
        },
        "phrase" => BooleanQuery {
            must: vec![Clause::Phrase(PhraseQuery {
                field: f.to_string(),
                terms: a[1..].iter().map(|t| t.as_bytes().to_vec()).collect(),
                slop: a[0].parse().expect("slop"),
            })],
            ..Default::default()
        },
        other => panic!("unknown query kind {other}"),
    };
    search_boolean_query_multi_segment(segments, &bq, bool_norms, top_n).expect("boolean query")
}

fn dv_range(
    reader: &DirectoryReader,
    segments: &[lucene_search::multi_segment::OpenSegment<'_>],
    q: &Query,
    top_n: usize,
) -> Vec<(i32, i64)> {
    let min: i64 = q.args[0].parse().expect("min");
    let max: i64 = q.args[1].parse().expect("max");
    let direction = match q.args[2].as_str() {
        "asc" => SortDirection::Ascending,
        "desc" => SortDirection::Descending,
        other => panic!("unknown direction {other}"),
    };
    let dv: Vec<DocValueSegment<'_>> = reader
        .segment_readers()
        .iter()
        .zip(segments)
        .filter_map(|(r, s)| {
            let number = r
                .field_infos()
                .fields
                .iter()
                .find(|f| f.name == q.field)?
                .number;
            let (meta, data) = r.doc_values_for_field(number)?;
            let entry = meta.numeric_entry(number)?;
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
        min,
        max,
        direction,
        MissingValue::Exclude,
        top_n,
    )
    .expect("doc-values range")
    .into_iter()
    .map(|h| (h.doc_id, h.value))
    .collect()
}
