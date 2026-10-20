//! lucene-codecs' `org.apache.lucene.codecs.simpletext` postings against
//! Java: `GenCodecsSimpleText` writes a flushed segment and a force-merged
//! one whose six fields (DOCS with the empty term, binary terms holding `\n`
//! and `\`, freqs, positions, offsets with payloads) all go to
//! `SimpleTextPostingsFormat`, and records every field's statistics and
//! postings, `seekCeil`/`seekExact`/`next` on a reused enum, `nextDoc` /
//! `advance` walks over the skip lists, and `impacts(FREQS)` driven by
//! `advanceShallow`/`advance`.
//!
//! Here the Rust reader opens Java's `.pst` and answers every one of those
//! the same way, the Rust writer rebuilds the `.pst` byte for byte from the
//! postings and norms Java recorded, and damaged copies of Java's bytes are
//! refused.

// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use lucene_codecs::field_infos::{self, FieldInfos, IndexOptions};
use lucene_codecs::postings_writer::{FieldNorms, FieldPostingsInput, TermPostings};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_codecs::simpletext::fields_reader::{
    SimpleTextFieldsReader, SimpleTextImpactsEnum, SimpleTextPostings, SimpleTextTerms,
};
use lucene_codecs::simpletext::fields_writer::{postings_file_name, write};
use lucene_codecs::simpletext::skip::Impacts;
use lucene_codecs::simpletext::{self, SeekStatus, ALL, FREQS, NO_MORE_DOCS};
use lucene_store::codec_util::ID_LENGTH;

const SUFFIX: &str = "SimpleText_0";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/codecs_simpletext/postings")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn rows(path: PathBuf) -> Vec<Vec<String>> {
    fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{path:?}: {e} (run GenCodecsSimpleText)"))
        .lines()
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect()
}

struct Doc {
    doc: i32,
    freq: i32,
    positions: Vec<i32>,
    offsets: Vec<(i32, i32)>,
    payloads: Vec<Vec<u8>>,
}

fn parse_doc(s: &str) -> Doc {
    let parts: Vec<&str> = s.split(':').collect();
    let list = |i: usize| -> Vec<&str> {
        parts
            .get(i)
            .filter(|p| !p.is_empty())
            .map_or(Vec::new(), |p| p.split('.').collect())
    };
    Doc {
        doc: parts[0].parse().unwrap(),
        freq: parts[1].parse().unwrap(),
        positions: list(2).iter().map(|p| p.parse().unwrap()).collect(),
        offsets: list(3)
            .iter()
            .map(|o| {
                let (s, e) = o.split_once('-').unwrap();
                (s.parse().unwrap(), e.parse().unwrap())
            })
            .collect(),
        payloads: list(4)
            .iter()
            .map(|p| if *p == "-" { Vec::new() } else { unhex(p) })
            .collect(),
    }
}

struct TermRow {
    term: Vec<u8>,
    doc_freq: i32,
    total_term_freq: i64,
    docs: Vec<Doc>,
}

struct FieldRow {
    number: i32,
    index_options: IndexOptions,
    has_payloads: bool,
    size: i64,
    sum_total_term_freq: i64,
    sum_doc_freq: i64,
    doc_count: i32,
    suffix: String,
}

struct Segment {
    dir: PathBuf,
    name: String,
    id: [u8; ID_LENGTH],
    infos: FieldInfos,
    max_doc: i32,
}

impl Segment {
    fn open(which: &str) -> Segment {
        let dir = root().join(which);
        let index = dir.join("index");
        let si = fs::read_dir(&index)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .find(|n| n.ends_with(".si"))
            .expect("run GenCodecsSimpleText");
        let buf = fs::read(index.join(&si)).unwrap();
        // magic, codec name (vInt length < 128 + bytes), version, id.
        let at = 4 + 1 + usize::from(buf[4]) + 4;
        let mut id = [0u8; ID_LENGTH];
        id.copy_from_slice(&buf[at..at + ID_LENGTH]);
        let name = si.trim_end_matches(".si").to_string();
        let fnm = fs::read(index.join(format!("{name}.fnm"))).unwrap();
        let infos = field_infos::parse(&fnm, &id, "").unwrap();
        let max_doc = rows(dir.join("meta.tsv"))[0][1].parse().unwrap();
        Segment {
            dir,
            name,
            id,
            infos,
            max_doc,
        }
    }

    fn pst(&self) -> Vec<u8> {
        let path = self
            .dir
            .join("index")
            .join(postings_file_name(&self.name, SUFFIX));
        fs::read(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"))
    }

    fn read_state(&self) -> SegmentReadState<'_> {
        SegmentReadState {
            max_doc: self.max_doc,
            segment_id: &self.id,
            segment_suffix: SUFFIX,
            field_infos: &self.infos,
        }
    }

    fn reader(&self) -> SimpleTextFieldsReader {
        SimpleTextFieldsReader::open(self.pst(), &self.read_state()).unwrap()
    }

    fn fields(&self) -> BTreeMap<String, FieldRow> {
        rows(self.dir.join("fields.tsv"))
            .into_iter()
            .map(|r| {
                let index_options = match r[2].as_str() {
                    "DOCS" => IndexOptions::Docs,
                    "DOCS_AND_FREQS" => IndexOptions::DocsAndFreqs,
                    "DOCS_AND_FREQS_AND_POSITIONS" => IndexOptions::DocsAndFreqsAndPositions,
                    "DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS" => {
                        IndexOptions::DocsAndFreqsAndPositionsAndOffsets
                    }
                    other => panic!("{other}"),
                };
                (
                    r[0].clone(),
                    FieldRow {
                        number: r[1].parse().unwrap(),
                        index_options,
                        has_payloads: r[3] == "true",
                        size: r[4].parse().unwrap(),
                        sum_total_term_freq: r[5].parse().unwrap(),
                        sum_doc_freq: r[6].parse().unwrap(),
                        doc_count: r[7].parse().unwrap(),
                        suffix: r[8].clone(),
                    },
                )
            })
            .collect()
    }

    fn postings(&self, field: &str) -> Vec<TermRow> {
        rows(self.dir.join(format!("postings.{field}.tsv")))
            .into_iter()
            .map(|r| TermRow {
                term: unhex(&r[0]),
                doc_freq: r[1].parse().unwrap(),
                total_term_freq: r[2].parse().unwrap(),
                docs: r[3..].iter().map(|d| parse_doc(d)).collect(),
            })
            .collect()
    }

    fn norms(&self) -> BTreeMap<String, Vec<i64>> {
        rows(self.dir.join("norms.tsv"))
            .into_iter()
            .map(|r| {
                (
                    r[0].clone(),
                    r[1..].iter().map(|v| v.parse().unwrap()).collect(),
                )
            })
            .collect()
    }
}

/// The generator's `doc(terms, pe)` over the Rust enum.
fn doc_text(terms: &SimpleTextTerms, pe: &mut SimpleTextPostings<'_>) -> String {
    let freq = pe.freq();
    let mut s = format!("{}:{freq}", pe.doc_id());
    if terms.has_positions() {
        let (mut pos, mut offs, mut pays) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..freq {
            pos.push(pe.next_position().unwrap().to_string());
            if terms.has_offsets() {
                offs.push(format!("{}-{}", pe.start_offset(), pe.end_offset()));
            }
            if terms.has_payloads() {
                pays.push(match pe.payload() {
                    Some(p) if !p.is_empty() => hex(p),
                    _ => "-".to_string(),
                });
            }
        }
        s += &format!(":{}:{}:{}", pos.join("."), offs.join("."), pays.join("."));
    }
    s
}

fn doc_row_text(terms: &SimpleTextTerms, d: &Doc) -> String {
    let mut s = format!("{}:{}", d.doc, d.freq);
    if terms.has_positions() {
        let pos: Vec<String> = d.positions.iter().map(i32::to_string).collect();
        let offs: Vec<String> = d.offsets.iter().map(|(a, b)| format!("{a}-{b}")).collect();
        let pays: Vec<String> = d
            .payloads
            .iter()
            .map(|p| if p.is_empty() { "-".into() } else { hex(p) })
            .collect();
        s += &format!(":{}:{}:{}", pos.join("."), offs.join("."), pays.join("."));
    }
    s
}

fn check_reader(which: &str) {
    let seg = Segment::open(which);
    let reader = seg.reader();
    reader.check_integrity().unwrap();
    let fields = seg.fields();
    assert_eq!(
        reader.field_names().collect::<Vec<_>>(),
        fields.keys().map(String::as_str).collect::<Vec<_>>(),
        "{which}"
    );
    assert!(reader.terms("nope").unwrap().is_none());
    for (name, row) in &fields {
        assert_eq!(row.suffix, SUFFIX);
        let terms = reader.terms(name).unwrap().unwrap();
        let ctx = format!("{which} {name}");
        assert_eq!(terms.info().number, row.number, "{ctx}");
        assert_eq!(terms.size(), row.size, "{ctx}");
        assert_eq!(
            terms.sum_total_term_freq(),
            row.sum_total_term_freq,
            "{ctx}"
        );
        assert_eq!(terms.sum_doc_freq(), row.sum_doc_freq, "{ctx}");
        assert_eq!(terms.doc_count(), row.doc_count, "{ctx}");
        assert_eq!(terms.has_payloads(), row.has_payloads, "{ctx}");
        assert_eq!(
            terms.has_positions(),
            row.index_options.subsumes_positions(),
            "{ctx}"
        );
        assert_eq!(
            terms.has_offsets(),
            row.index_options.subsumes_offsets(),
            "{ctx}"
        );
        assert_eq!(
            terms.has_freqs(),
            row.index_options != IndexOptions::Docs,
            "{ctx}"
        );
        let mut te = terms.iterator();
        for t in seg.postings(name) {
            let got = te.next().unwrap().map(<[u8]>::to_vec);
            assert_eq!(got.as_deref(), Some(t.term.as_slice()), "{ctx}");
            assert_eq!(te.doc_freq(), t.doc_freq, "{ctx} {}", hex(&t.term));
            assert_eq!(te.total_term_freq(), t.total_term_freq, "{ctx}");
            let mut pe = te.postings(ALL).unwrap();
            assert_eq!(pe.cost(), i64::from(t.doc_freq));
            assert_eq!(pe.doc_id(), -1);
            for d in &t.docs {
                assert_eq!(pe.next_doc().unwrap(), d.doc, "{ctx} {}", hex(&t.term));
                assert_eq!(doc_text(terms, &mut pe), doc_row_text(terms, d), "{ctx}");
            }
            assert_eq!(pe.next_doc().unwrap(), NO_MORE_DOCS, "{ctx}");
        }
        assert!(te.next().unwrap().is_none(), "{ctx}");
    }
}

#[test]
fn reader_reads_java_flushed_segment() {
    check_reader("flush");
}

#[test]
fn reader_reads_java_merged_segment() {
    check_reader("merged");
}

fn status(s: SeekStatus) -> &'static str {
    match s {
        SeekStatus::Found => "FOUND",
        SeekStatus::NotFound => "NOT_FOUND",
        SeekStatus::End => "END",
    }
}

fn check_probes(which: &str) {
    let seg = Segment::open(which);
    let reader = seg.reader();
    let mut enums = BTreeMap::new();
    let mut n = 0;
    for r in rows(seg.dir.join("probes.tsv")) {
        let terms = reader.terms(&r[0]).unwrap().unwrap();
        let te = enums
            .entry(r[0].clone())
            .or_insert_with(|| terms.iterator());
        let target = unhex(&r[1]);
        let st = te.seek_ceil(&target).unwrap();
        let ceil = if st == SeekStatus::End {
            "-".to_string()
        } else {
            format!(
                "{}:{}:{}",
                hex(te.term()),
                te.doc_freq(),
                te.total_term_freq()
            )
        };
        let exact = terms.iterator().seek_exact(&target).unwrap();
        let mut next = "-".to_string();
        if st != SeekStatus::End {
            if let Some(t) = te.next().unwrap() {
                let t = hex(t);
                next = format!("{t}:{}", te.doc_freq());
            }
        }
        let got = [status(st).to_string(), ceil, exact.to_string(), next];
        assert_eq!(&got[..], &r[2..6], "{which} {} {}", r[0], r[1]);
        n += 1;
    }
    assert!(n > 2000, "{which}: {n} probes");
}

#[test]
fn seeks_match_java() {
    check_probes("flush");
    check_probes("merged");
}

fn check_advance(which: &str) {
    let seg = Segment::open(which);
    let reader = seg.reader();
    for r in rows(seg.dir.join("advance.tsv")) {
        let terms = reader.terms(&r[0]).unwrap().unwrap();
        let mut te = terms.iterator();
        assert!(te.seek_exact(&unhex(&r[1])).unwrap());
        let mut pe = te.postings(r[2].parse().unwrap()).unwrap();
        for step in &r[3..] {
            let (op, want) = step.split_once('=').unwrap();
            let doc = if op == "n" {
                pe.next_doc().unwrap()
            } else {
                pe.advance(op[1..].parse().unwrap()).unwrap()
            };
            let got = if doc == NO_MORE_DOCS {
                "-".to_string()
            } else {
                doc_text(terms, &mut pe)
            };
            assert_eq!(got, want, "{which} {} {} {step}", r[0], r[1]);
        }
    }
}

#[test]
fn advance_matches_java() {
    check_advance("flush");
    check_advance("merged");
}

fn impacts_text(impacts: Impacts<'_>) -> String {
    let mut s = impacts.num_levels().to_string();
    for level in 0..impacts.num_levels() {
        let list: Vec<String> = impacts
            .impacts(level)
            .unwrap()
            .iter()
            .map(|i| format!("{}-{}", i.freq, i.norm))
            .collect();
        s += &format!(
            "/{}:{}",
            impacts.doc_id_up_to(level).unwrap(),
            list.join(".")
        );
    }
    s
}

fn check_impacts(which: &str) {
    let seg = Segment::open(which);
    let reader = seg.reader();
    let mut skipped = 0;
    for r in rows(seg.dir.join("impacts.tsv")) {
        let terms = reader.terms(&r[0]).unwrap().unwrap();
        let mut te = terms.iterator();
        assert!(te.seek_exact(&unhex(&r[1])).unwrap());
        let mut ie = te.impacts(FREQS).unwrap();
        if matches!(ie, SimpleTextImpactsEnum::Skip(_)) {
            skipped += 1;
        }
        let ctx = format!("{which} {} {}", r[0], r[1]);
        assert_eq!(
            format!("i={}", impacts_text(ie.impacts().unwrap())),
            r[2],
            "{ctx}"
        );
        for step in &r[3..] {
            let (op, want) = step.split_once('=').unwrap();
            let target: i32 = op[1..].parse().unwrap();
            let got = if op.starts_with('s') {
                ie.advance_shallow(target).unwrap();
                impacts_text(ie.impacts().unwrap())
            } else {
                let doc = ie.postings().advance(target).unwrap();
                if doc == NO_MORE_DOCS {
                    "-".to_string()
                } else {
                    let freq = ie.postings().freq();
                    format!("{doc}:{freq}={}", impacts_text(ie.impacts().unwrap()))
                }
            };
            assert_eq!(got, want, "{ctx} {step}");
        }
    }
    assert!(skipped > 100, "{which}: {skipped} terms over a skip list");
}

#[test]
fn impacts_match_java() {
    check_impacts("flush");
    check_impacts("merged");
}

fn rebuild(which: &str) -> (Segment, Vec<u8>) {
    let seg = Segment::open(which);
    let norms = seg.norms();
    let fields = seg.fields();
    let built: Vec<(&FieldRow, Vec<TermPostings>)> = fields
        .iter()
        .map(|(name, row)| {
            let terms = seg
                .postings(name)
                .into_iter()
                .map(|t| {
                    let mut tp = TermPostings {
                        term: t.term,
                        docs: t.docs.iter().map(|d| (d.doc, d.freq)).collect(),
                        ..TermPostings::default()
                    };
                    if row.index_options.subsumes_positions() {
                        tp.positions = t.docs.iter().map(|d| d.positions.clone()).collect();
                    }
                    if row.index_options.subsumes_offsets() {
                        tp.offsets = t.docs.iter().map(|d| d.offsets.clone()).collect();
                    }
                    if row.has_payloads {
                        for p in t.docs.iter().flat_map(|d| &d.payloads) {
                            tp.payload_lengths.push(p.len() as u32);
                            tp.payload_bytes.extend_from_slice(p);
                        }
                    }
                    tp
                })
                .collect();
            (row, terms)
        })
        .collect();
    let inputs: Vec<FieldPostingsInput<'_>> = built
        .iter()
        .map(|(row, terms)| FieldPostingsInput {
            field_number: row.number,
            index_options: row.index_options,
            doc_count: row.doc_count,
            has_payloads: row.has_payloads,
            terms,
        })
        .collect();
    let field_norms: Vec<FieldNorms<'_>> = norms
        .iter()
        .map(|(name, values)| FieldNorms {
            field_number: seg.infos.field_by_name(name).unwrap().number,
            values,
        })
        .collect();
    let state = SegmentWriteState {
        max_doc: seg.max_doc,
        segment_id: &seg.id,
        segment_suffix: SUFFIX,
        field_infos: &seg.infos,
    };
    let out = write(&inputs, &field_norms, &state).unwrap();
    drop(inputs);
    (seg, out)
}

#[test]
fn writer_rebuilds_java_bytes() {
    for which in ["flush", "merged"] {
        let (seg, out) = rebuild(which);
        let java = seg.pst();
        assert!(
            out == java,
            "{which}: {} bytes, Java {} (first difference at {:?})",
            out.len(),
            java.len(),
            out.iter().zip(&java).position(|(a, b)| a != b)
        );
    }
}

fn open_err(seg: &Segment, bytes: Vec<u8>) -> String {
    match SimpleTextFieldsReader::open(bytes, &seg.read_state()) {
        Ok(_) => panic!("damaged .pst opened"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn damaged_java_bytes_are_refused() {
    let seg = Segment::open("flush");
    let java = seg.pst();
    let footer = java.len() - b"checksum 00000000000819200768\n".len();
    assert!(java[footer..].starts_with(b"checksum "));

    // A checksum line that is not Java's.
    let mut bad = java.clone();
    bad[java.len() - 2] ^= 1;
    assert!(open_err(&seg, bad).contains("checksum failure"));

    // A flipped byte in the body: the CRC no longer matches.
    let mut flipped = java.clone();
    flipped[java.len() / 2] ^= 0x20;
    assert!(open_err(&seg, flipped).contains("checksum"));

    // Truncations: inside the footer, at it, mid-body.
    for cut in [java.len() - 1, footer, footer - 1, java.len() / 3, 0] {
        open_err(&seg, java[..cut].to_vec());
    }

    // Bytes after the footer.
    let mut longer = java.clone();
    longer.push(b'\n');
    assert!(open_err(&seg, longer).contains("Unexpected stuff"));

    // No checksum line after END.
    let mut no_footer = java[..footer].to_vec();
    no_footer.extend_from_slice(b"chksum 1\n");
    assert!(open_err(&seg, no_footer).contains("expected checksum line"));

    // Damage past a valid footer: Java's first doc line moved beyond maxDoc,
    // re-checksummed. `open` passes; loading that field's terms must not.
    let at = find(&java, b"\n    doc ").unwrap() + b"\n    doc ".len();
    let end = at + java[at..].iter().position(|&b| b == b'\n').unwrap();
    let field_at = java[..at]
        .windows(b"field ".len())
        .rposition(|w| w == b"field ")
        .unwrap();
    let field_end = field_at + java[field_at..].iter().position(|&b| b == b'\n').unwrap();
    let field = std::str::from_utf8(&java[field_at + b"field ".len()..field_end]).unwrap();
    let mut beyond = java[..at].to_vec();
    beyond.extend_from_slice(b"99999");
    beyond.extend_from_slice(&java[end..footer]);
    simpletext::util::write_checksum(&mut beyond);
    let r = SimpleTextFieldsReader::open(beyond, &seg.read_state()).unwrap();
    let e = r.terms(field).map(|_| ()).unwrap_err().to_string();
    assert!(e.contains("doc 99999 out of bounds"), "{e}");
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}
