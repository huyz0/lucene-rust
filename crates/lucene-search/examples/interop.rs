//! The Rust half of M4's interoperability matrix (T4.6): indexes written
//! partly by real Lucene and partly by this port, in either order, and read
//! back by both. `fixtures/src/InteropIndex.java` is the Java half, and
//! `scripts/verify-interop.sh` drives the eight directions.
//!
//! Document `i` is the same in both engines (`InteropIndex.java` has the same
//! table):
//!
//! | field | shape | value |
//! |---|---|---|
//! | `id` | stored | `"doc{i}"` |
//! | `body` | stored, positions and norms | [`body`] |
//! | `score` | stored + NUMERIC doc values | `3i - 1000` |
//! | `pt` | stored + one 8-byte point dimension | `7i - 500` |
//! | `cat` | stored + SORTED doc values | `"c{i % 10}"` |
//!
//! Usage:
//!
//! - `interop write <dir> <from> <to> <docs-per-segment>`: add documents
//!   `from..to` through `IndexWriter` -- appending to whatever index is
//!   already there, Java's or this port's -- and commit.
//! - `interop write-concurrent <dir> <num-docs> <docs-per-segment> <threads>`:
//!   add documents `0..num_docs` from `threads` threads at once through a
//!   `ConcurrentIndexWriter` (thread `t` adds every document `i` with
//!   `i % threads == t`), with a merge thread running and a commit every so
//!   often; then delete `body:w7` while that merge thread is still running,
//!   and commit. The resulting segment layout is timing-dependent; the
//!   document set is not.
//! - `interop merge <dir>`: open the index and merge every segment into one,
//!   however many engines wrote them.
//! - `interop delete <dir> <word>`: delete every document whose `body` has
//!   `word` (`deleteDocuments(Term)`), whichever engine wrote its segment.
//! - `interop verify <dir> <num-docs> [w<n>]`: open it with `DirectoryReader`
//!   and require exactly documents `0..num_docs` -- less those carrying
//!   `w<n>`, when given -- each one's doc values and points, term and phrase
//!   counts over `body`, and this port's own `CheckIndex`.
//!
//! This port's reader returns doc ids, not stored documents, so `verify`
//! identifies each document by its `score` doc value, which is unique.
// Example code: see `docs/arithmetic-gate.md`'s "Test code" section.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::points::{IntersectVisitor, Relation};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_codecs::{doc_values, points};
use lucene_index::buffered_updates::Term;
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::MergePolicyConfig;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::query::{BooleanQuery, Clause, PhraseQuery, TermQuery};
use lucene_search::{search_boolean_query_multi_segment, search_term_query_multi_segment};
use lucene_store::FsDirectory;

// The **reverse** of the numbers Java gives the same fields (`id` 0 ...
// `cat` 4, in the order `InteropIndex.document` adds them), on purpose: a
// segment's field numbers are its own, and every path that reads another
// engine's segment -- the merge, the delete, the reader -- has to go through
// that segment's `.fnm` rather than assume its writer's numbering. With equal
// numbers a reader that did the latter would pass every check here.
const F_ID: i32 = 4;
const F_BODY: i32 = 3;
const F_SCORE: i32 = 2;
const F_PT: i32 = 1;
const F_CAT: i32 = 0;

/// Must match `InteropIndex.body`: a word of 20, "shared" once to three
/// times, a word of 97.
fn body(i: i64) -> String {
    let mut b = format!("w{} ", i % 20);
    for _ in 0..=i % 3 {
        b.push_str("shared ");
    }
    b.push_str(&format!("v{}", i % 97));
    b
}

fn score(i: i64) -> i64 {
    3 * i - 1000
}

fn pt(i: i64) -> i64 {
    7 * i - 500
}

fn field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: false,
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
        vector_encoding: VectorEncoding::Float32,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

/// This port's schema for the table above (numbered in reverse of Java's --
/// see `F_ID`).
fn schema() -> Vec<FieldInfo> {
    vec![
        field("id", F_ID),
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            ..field("body", F_BODY)
        },
        FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..field("score", F_SCORE)
        },
        FieldInfo {
            point_dimension_count: 1,
            point_index_dimension_count: 1,
            point_num_bytes: 8,
            ..field("pt", F_PT)
        },
        FieldInfo {
            doc_values_type: DocValuesType::Sorted,
            ..field("cat", F_CAT)
        },
    ]
}

fn document(i: i64) -> Document {
    let stored = |field_number, value| StoredField {
        field_number,
        value,
    };
    Document {
        fields: vec![
            stored(F_ID, FieldValue::String(format!("doc{i}"))),
            stored(F_BODY, FieldValue::String(body(i))),
            stored(F_SCORE, FieldValue::Long(score(i))),
            stored(F_PT, FieldValue::Long(pt(i))),
            stored(F_CAT, FieldValue::String(format!("c{}", i % 10))),
        ],
    }
}

fn open_writer(dir: &FsDirectory) -> IndexWriter<'_> {
    let mut writer = IndexWriter::open(
        dir,
        schema(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .expect("open writer");
    writer.set_postings_field(Some("body")).expect("body");
    writer.set_doc_values_field(Some("score")).expect("score");
    writer.add_doc_values_field("cat").expect("cat");
    writer.add_points_field("pt").expect("pt");
    writer
}

fn write(path: &str, from: i64, to: i64, docs_per_segment: i32) {
    std::fs::create_dir_all(path).expect("create dir");
    let dir = FsDirectory::open(path);
    let mut writer = open_writer(&dir);
    writer
        .set_max_buffered_docs(docs_per_segment)
        .expect("max buffered docs");
    writer.set_ram_buffer_size_mb(4096.0).expect("ram buffer");
    for i in from..to {
        writer.add_document(document(i)).expect("add document");
    }
    writer.commit().expect("commit");
    println!("interop: this port wrote documents {from}..{to}");
}

fn write_concurrent(path: &str, num_docs: i64, docs_per_segment: i32, threads: i64) {
    use std::sync::atomic::{AtomicBool, Ordering};
    std::fs::create_dir_all(path).expect("create dir");
    let dir = FsDirectory::open(path);
    let mut writer = open_writer(&dir);
    writer
        .set_max_buffered_docs(docs_per_segment)
        .expect("max buffered docs");
    writer.set_ram_buffer_size_mb(4096.0).expect("ram buffer");
    writer.set_merge_policy(Some(MergePolicyConfig {
        max_merge_at_once: 4,
        segments_per_tier: 4,
        floor_segment_size: 1 << 40,
        ..MergePolicyConfig::default()
    }));
    let writer = ConcurrentIndexWriter::new(writer, threads as usize).expect("concurrent writer");
    let stop = AtomicBool::new(false);
    let merges = std::thread::scope(|scope| {
        let merger = scope.spawn(|| writer.run_merges(&stop).expect("merge thread"));
        let indexers: Vec<_> = (0..threads)
            .map(|t| {
                let writer = &writer;
                scope.spawn(move || {
                    for i in (t..num_docs).step_by(threads as usize) {
                        writer.add_document(document(i)).expect("add document");
                        if i % 1500 == t {
                            writer.commit().expect("commit");
                        }
                    }
                })
            })
            .collect();
        for indexer in indexers {
            indexer.join().expect("indexing thread");
        }
        writer.commit().expect("commit");
        // With the merge thread still running: a merge that started before
        // the delete has to carry it onto its merged segment.
        writer
            .delete_documents_by_term(&[Term {
                field: "body".to_string(),
                bytes: b"w7".to_vec(),
            }])
            .expect("delete");
        writer.commit().expect("commit");
        stop.store(true, Ordering::Release);
        merger.join().expect("merge thread")
    });
    println!(
        "interop: this port wrote documents 0..{num_docs} from {threads} threads \
         ({merges} concurrent merges) and deleted body:w7"
    );
}

fn merge(path: &str) {
    let dir = FsDirectory::open(path);
    let mut writer = open_writer(&dir);
    let before = writer.segment_infos().segments.len();
    writer.set_merge_policy(Some(MergePolicyConfig {
        max_merge_at_once: 100,
        segments_per_tier: 2,
        max_merged_segment_size: u64::MAX / 4,
        floor_segment_size: 1 << 40,
        ..MergePolicyConfig::default()
    }));
    writer.commit().expect("commit triggers the merge");
    let after = writer.segment_infos().segments.len();
    assert_eq!(after, 1, "merged {before} segments into {after}");
    println!("interop: this port merged {before} segments into 1");
}

/// Every point of a segment's `pt` field, by segment-local document.
struct Collect(Vec<(i32, i64)>);

impl IntersectVisitor for Collect {
    fn compare(&mut self, _min: &[u8], _max: &[u8]) -> Relation {
        Relation::CellCrossesQuery
    }

    fn visit(&mut self, _doc_id: i32) {
        unreachable!("every cell crosses the query");
    }

    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        let bits = u64::from_be_bytes(packed_value.try_into().expect("8-byte point"));
        self.0.push((doc_id, (bits ^ (1 << 63)) as i64));
    }
}

fn delete(path: &str, word: &str) {
    let dir = FsDirectory::open(path);
    let mut writer = open_writer(&dir);
    writer
        .delete_documents_by_term(&[Term {
            field: "body".to_string(),
            bytes: word.as_bytes().to_vec(),
        }])
        .expect("delete");
    writer.commit().expect("commit");
    println!("interop: this port deleted body:{word}");
}

/// `verify`'s view of a delete: `Some(w)` when every document whose body has
/// word `w{w}` was deleted.
fn is_live(deleted: Option<i64>, i: i64) -> bool {
    deleted.is_none_or(|w| i % 20 != w)
}

fn verify(path: &str, num_docs: i64, deleted: Option<i64>) -> usize {
    let live = |i: i64| is_live(deleted, i);
    let dir = FsDirectory::open(path);
    let reader = DirectoryReader::open(&dir).expect("open index");
    let mut failures = 0usize;
    let mut fail = |message: String| {
        failures += 1;
        if failures <= 30 {
            println!("MISMATCH {message}");
        }
    };
    let mut seen = vec![false; num_docs as usize];
    for (s, segment) in reader.segment_readers().iter().enumerate() {
        let number = |name: &str| {
            segment
                .field_infos()
                .fields
                .iter()
                .find(|f| f.name == name)
                .map(|f| f.number)
        };
        let (Some(score_number), Some(cat_number), Some(pt_number)) =
            (number("score"), number("cat"), number("pt"))
        else {
            fail(format!("segment {s}: a field is missing from its .fnm"));
            continue;
        };
        let (score_meta, score_data) = segment
            .doc_values_for_field(score_number)
            .expect("score doc values");
        let score_entry = score_meta.numeric_entry(score_number).expect("score entry");
        let (cat_meta, cat_data) = segment
            .doc_values_for_field(cat_number)
            .expect("cat doc values");
        let cat_entry = cat_meta.sorted_entry(cat_number).expect("cat entry");
        // A segment's ordinals are its own: a segment holding only some of the
        // ten categories numbers just those. Resolve each through its terms.
        let mut cat_terms = Vec::new();
        let mut cursor = lucene_codecs::terms_dict::TermsCursor::open(cat_data, &cat_entry.terms)
            .expect("cat terms");
        while let Some(term) = cursor.next_term().expect("cat term") {
            cat_terms.push(String::from_utf8(term.to_vec()).expect("utf-8 cat"));
        }
        let mut points_by_doc = Collect(Vec::new());
        if let Some((kdm, kdi, kdd)) = segment.points_files() {
            let points = points::open(kdm, kdi, kdd, &segment.segment_id(), "").expect("points");
            points
                .intersect(pt_number, &mut points_by_doc)
                .expect("walk points");
        }
        points_by_doc.0.sort_unstable();

        let max_doc = segment.max_doc;
        for doc in 0..max_doc {
            if segment
                .live_docs()
                .is_some_and(|live| !live.get(doc as usize))
            {
                continue;
            }
            let Some(value) =
                doc_values::numeric_value(score_data, score_entry, doc).expect("score")
            else {
                fail(format!("segment {s} doc {doc}: no score"));
                continue;
            };
            let i = (value + 1000) / 3;
            if score(i) != value || !(0..num_docs).contains(&i) || seen[i as usize] {
                fail(format!(
                    "segment {s} doc {doc}: score {value} names no fresh document"
                ));
                continue;
            }
            seen[i as usize] = true;
            let ord = doc_values::sorted_ord(cat_data, cat_entry, doc).expect("cat");
            let cat = ord.and_then(|o| cat_terms.get(o as usize));
            let want = format!("c{}", i % 10);
            if cat != Some(&want) {
                fail(format!("doc{i}: cat {cat:?} (ord {ord:?}), want {want}"));
            }
            let from = points_by_doc.0.partition_point(|&(d, _)| d < doc);
            let to = points_by_doc.0.partition_point(|&(d, _)| d <= doc);
            let values: Vec<i64> = points_by_doc.0[from..to].iter().map(|&(_, v)| v).collect();
            if values != [pt(i)] {
                fail(format!("doc{i}: points {values:?}, want [{}]", pt(i)));
            }
        }
    }
    for (i, &seen) in seen.iter().enumerate() {
        if seen != live(i as i64) {
            fail(format!(
                "doc{i} {}",
                if seen {
                    "was deleted but is still live"
                } else {
                    "is missing"
                }
            ));
        }
    }

    let opened = reader.open_segments().expect("open segments");
    let segments = opened.as_open_segments();
    let top_n = num_docs as usize + 1;
    let norms = reader.field_norms("body");
    let norms: Vec<Option<&_>> = norms.iter().map(Option::as_ref).collect();
    let fields = vec!["body".to_string()];
    let by_field = reader.field_norms_by_field(&fields);
    let bool_norms: Vec<Option<&_>> = by_field.iter().map(Some).collect();
    let mut check = |what: String, got: usize, predicate: &dyn Fn(i64) -> bool| {
        let want = (0..num_docs).filter(|&i| live(i) && predicate(i)).count();
        if got != want {
            fail(format!("{what}: count {got}, want {want}"));
        }
    };
    let term_count = |term: &str| {
        search_term_query_multi_segment(
            &segments,
            &TermQuery {
                field: "body".to_string(),
                term: term.as_bytes().to_vec(),
            },
            &norms,
            top_n,
        )
        .expect("term query")
        .len()
    };
    for w in 0..20 {
        check(format!("body:w{w}"), term_count(&format!("w{w}")), &|i| {
            i % 20 == w
        });
    }
    check("body:shared".to_string(), term_count("shared"), &|_| true);
    let phrase_count = |terms: [&str; 2]| {
        let query = BooleanQuery {
            must: vec![Clause::Phrase(PhraseQuery {
                field: "body".to_string(),
                terms: terms.iter().map(|t| t.as_bytes().to_vec()).collect(),
                slop: 0,
            })],
            ..Default::default()
        };
        search_boolean_query_multi_segment(&segments, &query, &bool_norms, top_n)
            .expect("phrase query")
            .len()
    };
    check(
        "body:\"shared shared\"".to_string(),
        phrase_count(["shared", "shared"]),
        &|i| i % 3 != 0,
    );
    check(
        "body:\"w3 shared\"".to_string(),
        phrase_count(["w3", "shared"]),
        &|i| i % 20 == 3,
    );

    for result in lucene_index::check_index::check_directory(&dir).expect("check index") {
        if !result.all_passed() {
            fail(format!("CheckIndex: {:?}", result.failures()));
        }
    }
    failures
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |n: usize| -> &str { args.get(n).map(String::as_str).expect("missing argument") };
    match arg(0) {
        "write" => write(
            arg(1),
            arg(2).parse().expect("from"),
            arg(3).parse().expect("to"),
            arg(4).parse().expect("docs per segment"),
        ),
        "write-concurrent" => write_concurrent(
            arg(1),
            arg(2).parse().expect("num docs"),
            arg(3).parse().expect("docs per segment"),
            arg(4).parse().expect("threads"),
        ),
        "merge" => merge(arg(1)),
        "delete" => delete(arg(1), arg(2)),
        "verify" => {
            let num_docs: i64 = arg(2).parse().expect("num docs");
            let deleted = args
                .get(3)
                .map(|w| w.trim_start_matches('w').parse().expect("deleted word"));
            let failures = verify(arg(1), num_docs, deleted);
            if failures > 0 {
                println!("{failures} check(s) failed");
                std::process::exit(1);
            }
            println!("interop: this port reads all {num_docs} documents. PASS");
        }
        other => panic!("unknown command {other}"),
    }
}
