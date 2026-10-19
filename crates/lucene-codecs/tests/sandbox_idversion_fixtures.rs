//! lucene-sandbox's `IDVersionPostingsFormat` against Java:
//! `GenSandboxIdVersion` writes a flushed segment (updates and deletes
//! inside it) and a force-merged one, routing `id` (block sizes 25/48) and
//! `id2` (4/9) to the format, and records each field's statistics, every
//! term with its document and version, and a random walk of
//! `seekExact(id, minVersion)`/`seekCeil`/`next` on one enum.
//!
//! Here the Rust reader replays that walk on Java's files, and the Rust
//! writer rebuilds Java's `.tiv`/`.tipv` byte for byte -- from the terms it
//! read, and (for the flush) from the documents added, deleted ones
//! included, so the writer's skip of documents dead at flush is exercised.

// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::field_infos::{self, IndexOptions};
use lucene_codecs::idversion::{
    write_fields, IdVersionFieldInput, IdVersionPostings, IdVersionPostingsFormat,
    VersionBlockTreeTermsReader,
};
use lucene_codecs::postings_writer::TermPostings;
use lucene_store::codec_util::ID_LENGTH;

fn dir(which: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data/sandbox_idversion")
        .join(which)
}

/// The one segment's name and id (from its `.si` header).
fn segment(which: &str) -> (String, [u8; ID_LENGTH]) {
    let index = dir(which).join("index");
    let si = fs::read_dir(&index)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .find(|n| n.ends_with(".si"))
        .expect("run fixtures generator first (GenSandboxIdVersion)");
    let buf = fs::read(index.join(&si)).unwrap();
    // magic, codec name (vInt length < 128 + bytes), version, id.
    let at = 4 + 1 + usize::from(buf[4]) + 4;
    let mut id = [0u8; ID_LENGTH];
    id.copy_from_slice(&buf[at..at + ID_LENGTH]);
    (si.trim_end_matches(".si").to_string(), id)
}

struct Fixture {
    name: String,
    id: [u8; ID_LENGTH],
    names: BTreeMap<i32, String>,
}

impl Fixture {
    fn open(which: &str) -> Fixture {
        let (name, id) = segment(which);
        let fnm = fs::read(dir(which).join("index").join(format!("{name}.fnm"))).unwrap();
        let fis = field_infos::parse(&fnm, &id, "").unwrap();
        let names = fis
            .fields
            .iter()
            .map(|f| (f.number, f.name.clone()))
            .collect();
        Fixture { name, id, names }
    }

    fn file(&self, which: &str, suffix: &str, ext: &str) -> Vec<u8> {
        fs::read(
            dir(which)
                .join("index")
                .join(format!("{}_{suffix}.{ext}", self.name)),
        )
        .unwrap()
    }

    fn number(&self, field: &str) -> i32 {
        *self.names.iter().find(|(_, n)| *n == field).unwrap().0
    }

    fn reader(&self, which: &str, suffix: &str, max_doc: i32) -> VersionBlockTreeTermsReader {
        VersionBlockTreeTermsReader::open(
            self.file(which, suffix, "tiv"),
            &self.file(which, suffix, "tipv"),
            &self.id,
            suffix,
            max_doc,
            &|n| self.names.get(&n).cloned(),
        )
        .unwrap()
    }
}

/// `(suffix, field, format)` of the two groups.
fn groups() -> [(&'static str, &'static str, IdVersionPostingsFormat); 2] {
    [
        ("IDVersion_0", "id", IdVersionPostingsFormat::default()),
        (
            "IDVersion_1",
            "id2",
            IdVersionPostingsFormat::new(4, 9).unwrap(),
        ),
    ]
}

/// `term doc version`, as `GenSandboxIdVersion.hit` prints it.
fn hit(te: &mut lucene_codecs::idversion::IdVersionSegmentTermsEnum<'_>) -> String {
    let term = String::from_utf8(te.term().to_vec()).unwrap();
    match te.postings(true).unwrap() {
        IdVersionPostings::Positions(mut p) => {
            let doc = p.next_doc();
            assert_eq!(p.next_position(), 0);
            let version = i64::from_be_bytes(p.payload());
            format!("{term} {doc} {version}")
        }
        IdVersionPostings::Docs(_) => unreachable!("positions were asked for"),
    }
}

/// The lines of `ops.tsv` for `field`.
fn ops_lines(which: &str, field: &str) -> Vec<String> {
    let text = fs::read_to_string(dir(which).join("ops.tsv")).unwrap();
    let mut out = Vec::new();
    let mut inside = false;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("field\t") {
            inside = rest.split('\t').next() == Some(field);
        }
        if inside {
            out.push(line.to_string());
        }
    }
    out
}

fn deleted(which: &str) -> BTreeSet<i32> {
    let text = fs::read_to_string(dir(which).join("ops.tsv")).unwrap();
    let first = text.lines().next().unwrap();
    first
        .split(' ')
        .skip(1)
        .map(|d| d.parse().unwrap())
        .collect()
}

/// `SegmentInfo.maxDoc()`: the `.si` header (id, empty suffix), the
/// version (three ints), the minimum version (a flag, three ints if set),
/// then the document count.
fn max_doc(which: &str) -> i32 {
    let (name, _) = segment(which);
    let buf = fs::read(dir(which).join("index").join(format!("{name}.si"))).unwrap();
    let at = 4 + 1 + usize::from(buf[4]) + 4 + ID_LENGTH;
    assert_eq!(buf[at], 0, "no suffix");
    let at = at + 13 + 1 + if buf[at + 13] == 1 { 12 } else { 0 };
    i32::from_le_bytes(buf[at..at + 4].try_into().unwrap())
}

/// Replays `ops.tsv` for one field: statistics, the full term walk, then
/// the random seeks.
fn replay(which: &str) {
    let fx = Fixture::open(which);
    for (suffix, field, _) in groups() {
        let reader = fx.reader(which, suffix, max_doc(which));
        assert_eq!(reader.size(), 1);
        assert_eq!(reader.field_names().collect::<Vec<_>>(), vec![field]);
        reader.check_integrity().unwrap();
        let fr = reader.terms(field).unwrap();
        assert_eq!(fr.field_number(), fx.number(field));
        let mut got = vec![format!(
            "field\t{field}\t{} {} {} {} {} {}",
            fr.size(),
            fr.doc_count(),
            fr.sum_doc_freq(),
            fr.sum_total_term_freq(),
            String::from_utf8_lossy(fr.min()),
            String::from_utf8_lossy(fr.max())
        )];
        let mut all = reader.iterator(field).unwrap();
        while all.next().unwrap().is_some() {
            assert_eq!((all.doc_freq(), all.total_term_freq()), (1, 1));
            got.push(format!("term\t{}", hit(&mut all)));
        }
        let expected = ops_lines(which, field);
        let mut te = reader.iterator(field).unwrap();
        for line in &expected[got.len()..] {
            let mut cols = line.split('\t');
            let (op, arg) = (cols.next().unwrap(), cols.next().unwrap());
            let out = match op {
                "exact" => {
                    let (t, min) = arg.split_once(' ').unwrap();
                    if te
                        .seek_exact_with_version(t.as_bytes(), min.parse().unwrap())
                        .unwrap()
                    {
                        let version = te.version();
                        format!("exact\t{arg}\ttrue {version} {}", hit(&mut te))
                    } else {
                        format!("exact\t{arg}\tfalse")
                    }
                }
                "ceil" => {
                    let status = te.try_seek_ceil(arg.as_bytes()).unwrap();
                    let name = match status {
                        SeekStatus::Found => "FOUND",
                        SeekStatus::NotFound => "NOT_FOUND",
                        SeekStatus::End => "END",
                    };
                    if status == SeekStatus::End {
                        format!("ceil\t{arg}\t{name}")
                    } else {
                        format!("ceil\t{arg}\t{name} {}", hit(&mut te))
                    }
                }
                "next" => {
                    if te.next().unwrap().is_some() {
                        format!("next\t\t{}", hit(&mut te))
                    } else {
                        "next\t\tnull".to_string()
                    }
                }
                other => panic!("unknown op {other}"),
            };
            got.push(out);
        }
        assert_eq!(got.len(), expected.len(), "{which}/{field}");
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert_eq!(g, e, "{which}/{field} line {i}");
        }
    }
}

#[test]
fn the_reader_replays_javas_seeks_on_a_flushed_segment() {
    replay("flush");
}

#[test]
fn the_reader_replays_javas_seeks_on_a_merged_segment() {
    replay("merged");
}

/// One term in one document at position 0, its version the payload.
fn posting(term: &[u8], docs: &[(i32, i64)]) -> TermPostings {
    TermPostings {
        term: term.to_vec(),
        docs: docs.iter().map(|&(d, _)| (d, 1)).collect(),
        positions: docs.iter().map(|_| vec![0]).collect(),
        offsets: vec![],
        payload_bytes: docs.iter().flat_map(|&(_, v)| v.to_be_bytes()).collect(),
        payload_lengths: docs.iter().map(|_| 8).collect(),
    }
}

fn write_one(
    fx: &Fixture,
    field: &str,
    suffix: &str,
    format: &IdVersionPostingsFormat,
    terms: &[TermPostings],
    live: Option<&dyn Fn(i32) -> bool>,
) -> lucene_codecs::idversion::IdVersionOutput {
    let input = IdVersionFieldInput {
        field_number: fx.number(field),
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        store_term_vectors: false,
        has_payloads: true,
        terms,
    };
    write_fields(&[input], live, format, &fx.id, suffix).unwrap()
}

/// The writer rebuilds Java's files from the terms the reader returns.
fn rewrite(which: &str) {
    let fx = Fixture::open(which);
    for (suffix, field, format) in groups() {
        let reader = fx.reader(which, suffix, max_doc(which));
        let mut te = reader.iterator(field).unwrap();
        let mut terms = Vec::new();
        while let Some(t) = te.next().unwrap() {
            let t = t.to_vec();
            let state = te.term_state().unwrap();
            terms.push(posting(&t, &[(state.doc_id, state.id_version)]));
        }
        let out = write_one(&fx, field, suffix, &format, &terms, None);
        assert!(
            out.tiv == fx.file(which, suffix, "tiv"),
            "{which}/{suffix}.tiv"
        );
        assert!(
            out.tipv == fx.file(which, suffix, "tipv"),
            "{which}/{suffix}.tipv"
        );
    }
}

#[test]
fn the_writer_rebuilds_a_flushed_segments_files_byte_for_byte() {
    rewrite("flush");
}

#[test]
fn the_writer_rebuilds_a_merged_segments_files_byte_for_byte() {
    rewrite("merged");
}

#[test]
fn the_writer_skips_documents_deleted_at_flush_as_java_does() {
    let fx = Fixture::open("flush");
    let dead = deleted("flush");
    let text = fs::read_to_string(dir("flush").join("docs.tsv")).unwrap();
    for (suffix, field, format) in groups() {
        let mut by_term: BTreeMap<Vec<u8>, Vec<(i32, i64)>> = BTreeMap::new();
        for line in text.lines() {
            let c: Vec<&str> = line.split('\t').collect();
            if c[1] == field {
                by_term
                    .entry(c[2].as_bytes().to_vec())
                    .or_default()
                    .push((c[0].parse().unwrap(), c[3].parse().unwrap()));
            }
        }
        let terms: Vec<TermPostings> = by_term.iter().map(|(t, d)| posting(t, d)).collect();
        // Some term must lose a document to the deletions, or this proves
        // nothing.
        assert!(
            by_term.values().flatten().any(|(d, _)| dead.contains(d)),
            "{field}"
        );
        let live = |d: i32| !dead.contains(&d);
        let out = write_one(&fx, field, suffix, &format, &terms, Some(&live));
        assert!(out.tiv == fx.file("flush", suffix, "tiv"), "{suffix}.tiv");
        assert!(
            out.tipv == fx.file("flush", suffix, "tipv"),
            "{suffix}.tipv"
        );
    }
}
