//! The Rust half of M4's differential operation-stream fuzzer (T4.5):
//! `fixtures/src/OpStreamFuzz.java`'s seeded stream -- adds, updates,
//! deletes by id and by body word, numeric doc-values updates, flushes and
//! commits -- applied to this port's `IndexWriter`, then dumped in the same
//! **semantic** form, so `scripts/op-stream-fuzz.sh` can compare the two
//! engines line by line. Segment counts, merge timing and file layouts
//! legitimately differ and do not appear in the dump.
//!
//! The dump: every live document (`doc <id> ver <v> score <s> cat <c> pt
//! [<p>]`, by id); for every body term, the ids of the live documents it
//! matches; for a set of two-word phrases and one point range, the ids
//! matched. This port's reader returns doc ids, not stored documents, so ids
//! come from the `idn` doc values, as on the Java side.
//!
//! Usage: `op_stream_fuzz <out-dir> <first-seed> <end-seed> <ops>` writes
//! `<out-dir>/<seed>.rust.txt` for every seed in `first..end`.
// Example code: see `docs/arithmetic-gate.md`'s "Test code" section.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use lucene_codecs::field_infos::{DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::points::{IntersectVisitor, Relation};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_codecs::{doc_values, points, terms_dict};
use lucene_index::buffered_updates::Term;
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::MergePolicyConfig;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::query::{BooleanQuery, Clause, PhraseQuery, TermQuery};
use lucene_search::{search_boolean_query_multi_segment, search_term_query_multi_segment};
use lucene_store::FsDirectory;

const WORDS: u64 = 10;
const XS: u64 = 13;

/// Identical to `OpStreamFuzz.Rng`.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        Rng((z ^ (z >> 31)) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn body(id: u64, v: u64) -> String {
    format!(
        "w{} w{} x{}",
        (id * 7 + v) % WORDS,
        (id + 3 * v) % WORDS,
        id % XS
    )
}

fn cat(id: u64, v: u64) -> Option<String> {
    (!(id + v).is_multiple_of(6)).then(|| format!("c{}", (id + v) % 5))
}

fn pt(id: u64, v: u64) -> i64 {
    id as i64 * 3 - v as i64
}

const F_ID: i32 = 0;
const F_BODY: i32 = 1;
const F_IDN: i32 = 2;
const F_VER: i32 = 3;
const F_SCORE: i32 = 4;
const F_CAT: i32 = 5;
const F_PT: i32 = 6;

fn schema() -> Vec<FieldInfo> {
    vec![
        FieldInfo {
            index_options: IndexOptions::Docs,
            omit_norms: true,
            ..FieldInfo::new("id", F_ID)
        },
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            ..FieldInfo::new("body", F_BODY)
        },
        FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..FieldInfo::new("idn", F_IDN)
        },
        FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..FieldInfo::new("ver", F_VER)
        },
        FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..FieldInfo::new("score", F_SCORE)
        },
        FieldInfo {
            doc_values_type: DocValuesType::Sorted,
            ..FieldInfo::new("cat", F_CAT)
        },
        FieldInfo {
            point_dimension_count: 1,
            point_index_dimension_count: 1,
            point_num_bytes: 8,
            ..FieldInfo::new("pt", F_PT)
        },
    ]
}

fn document(id: u64, v: u64) -> Document {
    let field = |field_number, value| StoredField {
        field_number,
        value,
    };
    let mut fields = vec![
        field(F_ID, FieldValue::String(format!("i{id}"))),
        field(F_BODY, FieldValue::String(body(id, v))),
        field(F_IDN, FieldValue::Long(id as i64)),
        field(F_VER, FieldValue::Long(v as i64)),
        field(F_SCORE, FieldValue::Long((id * 10 + v) as i64)),
    ];
    if let Some(c) = cat(id, v) {
        fields.push(field(F_CAT, FieldValue::String(c)));
    }
    fields.push(field(F_PT, FieldValue::Long(pt(id, v))));
    Document { fields }
}

fn run(path: &Path, seed: u64, num_ops: usize) -> Result<(), String> {
    let mut rng = Rng::new(seed);
    let max_buffered_docs = 2 + rng.below(20) as i32;
    let mut live: BTreeMap<u64, u64> = BTreeMap::new();
    let mut next_id = 0u64;
    let dir = FsDirectory::open(path);
    let e = |e: lucene_index::index_writer::Error| e.to_string();
    let mut w = IndexWriter::open(
        &dir,
        schema(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .map_err(e)?;
    w.set_postings_field(Some("id")).map_err(e)?;
    w.add_postings_field("body").map_err(e)?;
    w.set_doc_values_field(Some("idn")).map_err(e)?;
    for f in ["ver", "score", "cat"] {
        w.add_doc_values_field(f).map_err(e)?;
    }
    w.add_points_field("pt").map_err(e)?;
    w.set_max_buffered_docs(max_buffered_docs).map_err(e)?;
    w.set_ram_buffer_size_mb(4096.0).map_err(e)?;
    // `TieredMergePolicy`'s shape, with a small floor so merges fire at these
    // sizes; Java's own schedule differs, which is the point of comparing
    // semantically.
    w.set_merge_policy(Some(MergePolicyConfig {
        floor_segment_size: 1 << 16,
        ..MergePolicyConfig::default()
    }));
    let id_term = |id: u64| Term::new("id", format!("i{id}").into_bytes());

    for _ in 0..num_ops {
        let roll = rng.below(100);
        let needs_pick = (48..=77).contains(&roll) && !(67..=70).contains(&roll);
        // One pick in five is any id ever issued -- possibly deleted, so an
        // update of it is a plain add and a delete or doc-values update of it
        // matches nothing. The rest are a live id.
        let pick = if !needs_pick {
            None
        } else if next_id > 0 && rng.below(5) == 0 {
            Some(rng.below(next_id))
        } else if !live.is_empty() {
            let k = rng.below(live.len() as u64) as usize;
            live.keys().nth(k).copied()
        } else {
            None
        };
        if roll <= 47 || (needs_pick && pick.is_none()) {
            let id = next_id;
            next_id += 1;
            w.add_document(document(id, 0)).map_err(e)?;
            live.insert(id, 0);
        } else if roll <= 59 {
            let id = pick.expect("picked");
            let v = live.get(&id).map_or(1, |v| v + 1);
            w.update_document(id_term(id), document(id, v)).map_err(e)?;
            live.insert(id, v);
        } else if roll <= 66 {
            let id = pick.expect("picked");
            w.delete_documents_by_term(&[id_term(id)]).map_err(e)?;
            live.remove(&id);
        } else if roll <= 70 {
            let word = rng.below(WORDS);
            let term = format!("w{word}");
            w.delete_documents_by_term(&[Term::new("body", term.clone().into_bytes())])
                .map_err(e)?;
            live.retain(|&id, &mut v| !body(id, v).split(' ').take(2).any(|t| t == term));
        } else if roll <= 77 {
            let value = rng.below(100_000);
            w.update_numeric_doc_value(id_term(pick.expect("picked")), "score", value as i64)
                .map_err(e)?;
        } else if roll <= 83 {
            w.flush().map_err(e)?;
        } else {
            w.commit().map_err(e)?;
        }
    }
    w.commit().map_err(e)?;
    Ok(())
}

struct AllPoints(Vec<(i32, i64)>);

impl IntersectVisitor for AllPoints {
    fn compare(&mut self, _min: &[u8], _max: &[u8]) -> Relation {
        Relation::CellCrossesQuery
    }
    fn visit(&mut self, _doc_id: i32) {
        unreachable!("every cell crosses the query");
    }
    fn visit_with_value(&mut self, doc_id: i32, packed: &[u8]) {
        let bits = u64::from_be_bytes(packed.try_into().expect("8-byte point"));
        self.0.push((doc_id, (bits ^ (1 << 63)) as i64));
    }
}

fn set(ids: &BTreeSet<i64>) -> String {
    let items: Vec<String> = ids.iter().map(i64::to_string).collect();
    format!("[{}]", items.join(", "))
}

fn dump(path: &Path) -> Result<String, String> {
    let dir = FsDirectory::open(path);
    let reader = DirectoryReader::open(&dir).map_err(|e| e.to_string())?;
    let mut out = String::new();
    let mut id_of: Vec<i64> = Vec::new();
    let mut docs: BTreeMap<i64, String> = BTreeMap::new();
    for segment in reader.segment_readers() {
        let fields = &segment.field_infos().fields;
        let number = |name: &str| fields.iter().find(|f| f.name == name).map(|f| f.number);
        let numeric = |name: &str, doc: i32| -> Result<Option<i64>, String> {
            let Some(n) = number(name) else {
                return Ok(None);
            };
            let Some((meta, data)) = segment.doc_values_for_field(n) else {
                return Ok(None);
            };
            let Some(entry) = meta.numeric_entry(n) else {
                return Ok(None);
            };
            doc_values::numeric_value(data, entry, doc).map_err(|e| e.to_string())
        };
        let cat_terms = match number("cat").and_then(|n| {
            let (meta, data) = segment.doc_values_for_field(n)?;
            Some((meta.sorted_entry(n)?, data))
        }) {
            Some((entry, data)) => Some((
                entry,
                data,
                terms_dict::decode_all_terms(data, &entry.terms).map_err(|e| e.to_string())?,
            )),
            None => None,
        };
        let mut points_by_doc = AllPoints(Vec::new());
        if let (Some((kdm, kdi, kdd)), Some(n)) = (segment.points_files(), number("pt")) {
            let reader = points::open(kdm, kdi, kdd, &segment.segment_id(), "")
                .map_err(|e| e.to_string())?;
            if reader.field(n).is_some() {
                reader
                    .intersect(n, &mut points_by_doc)
                    .map_err(|e| e.to_string())?;
            }
        }
        points_by_doc.0.sort_unstable();
        for doc in 0..segment.max_doc {
            let id = numeric("idn", doc)?.unwrap_or(-1);
            id_of.push(id);
            if segment.live_docs().is_some_and(|l| !l.get(doc as usize)) {
                continue;
            }
            let show = |v: Option<i64>| v.map_or("-".to_string(), |v| v.to_string());
            let c = match &cat_terms {
                Some((entry, data, terms)) => doc_values::sorted_ord(data, entry, doc)
                    .map_err(|e| e.to_string())?
                    .map_or("-".to_string(), |ord| {
                        String::from_utf8_lossy(&terms[ord as usize]).into_owned()
                    }),
                None => "-".to_string(),
            };
            let from = points_by_doc.0.partition_point(|&(d, _)| d < doc);
            let to = points_by_doc.0.partition_point(|&(d, _)| d <= doc);
            let p: Vec<String> = points_by_doc.0[from..to]
                .iter()
                .map(|&(_, v)| v.to_string())
                .collect();
            let line = format!(
                "doc {id} ver {} score {} cat {c} pt [{}]",
                show(numeric("ver", doc)?),
                show(numeric("score", doc)?),
                p.join(", ")
            );
            if docs.insert(id, line).is_some() {
                writeln!(out, "DUPLICATE id {id}").unwrap();
            }
        }
    }
    for line in docs.values() {
        writeln!(out, "{line}").unwrap();
    }
    std::fs::write(
        path.with_extension("meta"),
        format!("{} {}\n", reader.segment_readers().len(), id_of.len()),
    )
    .map_err(|e| e.to_string())?;

    let opened = reader.open_segments().map_err(|e| e.to_string())?;
    let segments = opened.as_open_segments();
    let top_n = id_of.len().max(1);
    let norms = reader.field_norms("body");
    let norms: Vec<Option<&_>> = norms.iter().map(Option::as_ref).collect();
    let fields = vec!["body".to_string()];
    let by_field = reader.field_norms_by_field(&fields);
    let bool_norms: Vec<Option<&_>> = by_field.iter().map(Some).collect();
    let ids = |hits: Vec<lucene_search::collector::ScoreDoc>| -> BTreeSet<i64> {
        hits.iter().map(|h| id_of[h.doc_id as usize]).collect()
    };
    let mut terms: Vec<String> = (0..WORDS).map(|i| format!("w{i}")).collect();
    terms.extend((0..XS).map(|i| format!("x{i}")));
    for t in &terms {
        let hits = search_term_query_multi_segment(
            &segments,
            &TermQuery {
                field: "body".to_string(),
                term: t.as_bytes().to_vec(),
            },
            &norms,
            top_n,
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "term {t} {}", set(&ids(hits))).unwrap();
    }
    let phrase = |a: &str, b: &str| -> Result<BTreeSet<i64>, String> {
        let query = BooleanQuery {
            must: vec![Clause::Phrase(PhraseQuery {
                field: "body".to_string(),
                terms: vec![a.as_bytes().to_vec(), b.as_bytes().to_vec()],
                slop: 0,
            })],
            ..Default::default()
        };
        search_boolean_query_multi_segment(&segments, &query, &bool_norms, top_n)
            .map(ids)
            .map_err(|e| e.to_string())
    };
    for a in 0..WORDS {
        for b in (0..WORDS).step_by(3) {
            let got = phrase(&format!("w{a}"), &format!("w{b}"))?;
            writeln!(out, "phrase w{a} w{b} {}", set(&got)).unwrap();
        }
        let got = phrase(&format!("w{a}"), &format!("x{a}"))?;
        writeln!(out, "phrase w{a} x{a} {}", set(&got)).unwrap();
    }
    // Through the port's own `PointValues.intersect` pruning walk
    // (`PointsReader::range_query`), not the per-document values above.
    let mut in_range: BTreeSet<i64> = BTreeSet::new();
    let mut doc_base = 0usize;
    for segment in reader.segment_readers() {
        let n = segment
            .field_infos()
            .fields
            .iter()
            .find(|f| f.name == "pt")
            .map(|f| f.number);
        if let (Some((kdm, kdi, kdd)), Some(n)) = (segment.points_files(), n) {
            let points = points::open(kdm, kdi, kdd, &segment.segment_id(), "")
                .map_err(|e| e.to_string())?;
            if points.field(n).is_some() {
                let lower = lucene_search::pack_i64(-20);
                let upper = lucene_search::pack_i64(150);
                for doc in points
                    .range_query(n, &lower, &upper)
                    .map_err(|e| e.to_string())?
                {
                    if segment.live_docs().is_none_or(|l| l.get(doc as usize)) {
                        in_range.insert(id_of[doc_base + doc as usize]);
                    }
                }
            }
        }
        doc_base += segment.max_doc as usize;
    }
    writeln!(out, "range pt -20..150 {}", set(&in_range)).unwrap();
    Ok(out)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = Path::new(&args[0]);
    let first: u64 = args[1].parse().expect("first seed");
    let end: u64 = args[2].parse().expect("end seed");
    let num_ops: usize = args[3].parse().expect("ops");
    std::fs::create_dir_all(out).expect("out dir");
    let work = std::env::temp_dir().join(format!("op-stream-rust-{}", std::process::id()));
    for seed in first..end {
        let idx = work.join(format!("s{seed}"));
        let _ = std::fs::remove_dir_all(&idx);
        std::fs::create_dir_all(&idx).expect("index dir");
        let text = run(&idx, seed, num_ops)
            .and_then(|()| dump(&idx))
            .unwrap_or_else(|why| format!("ERROR {why}\n"));
        std::fs::write(out.join(format!("{seed}.rust.txt")), text).expect("write dump");
        // Segment count and `maxDoc` at the end: evidence that merges ran
        // (a `maxDoc` below the documents ever added means a merge reclaimed
        // deletes).
        if let Ok(meta) = std::fs::read(idx.with_extension("meta")) {
            std::fs::write(out.join(format!("{seed}.rust.meta")), meta).expect("write meta");
        }
        let _ = std::fs::remove_dir_all(&idx);
    }
    let _ = std::fs::remove_dir_all(&work);
    println!("op_stream_fuzz: this port ran seeds {first}..{end}");
}
