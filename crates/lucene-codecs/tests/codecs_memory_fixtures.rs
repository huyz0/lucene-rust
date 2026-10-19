//! lucene-codecs' `org.apache.lucene.codecs.memory` formats against Java:
//! `GenCodecsMemory` writes a flushed segment (deletes inside it) and a
//! force-merged one, routing `id`/`tag`/`body`/`pay` (DOCS, DOCS_AND_FREQS,
//! positions, offsets and payloads) to `FSTPostingsFormat` and
//! `did`/`dbody`/`dpay` to `DirectPostingsFormat`, and records every field's
//! statistics, every term's postings, `seekCeil`/`seekExact`/`next` on present
//! and absent targets, and `Terms.intersect` over a set of patterns and start
//! terms.
//!
//! Here the Rust readers open Java's files and answer every one of those the
//! same way, and the Rust writers rebuild each group's files byte for byte
//! from the postings Java recorded.

// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::field_infos::{self, FieldInfos, IndexOptions};
use lucene_codecs::memory::{
    DirectFields, DirectFiles, DirectPostingsEnum, DirectPostingsFormat, FstPostingsFormat,
    FstTermsReader,
};
use lucene_codecs::postings::{self, DocInput, PayInput, PosInput, PostingsFlags};
use lucene_codecs::postings_writer::{FieldNorms, FieldPostingsInput, TermPostings};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_store::codec_util::ID_LENGTH;
use lucene_util::automaton::{operations, CompiledAutomaton, RegExp};

const FST_SUFFIX: &str = "FST50_0";
const DIRECT_SUFFIX: &str = "Direct_0";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/codecs_memory")
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
        .unwrap_or_else(|e| panic!("{path:?}: {e} (run GenCodecsMemory)"))
        .lines()
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect()
}

/// One document of a term, as `postings.FIELD.tsv` records it.
#[derive(Debug, Clone, PartialEq)]
struct Doc {
    doc: i32,
    freq: i32,
    positions: Vec<i32>,
    offsets: Vec<(i32, i32)>,
    payloads: Vec<Vec<u8>>,
}

#[derive(Debug, Clone)]
struct TermRow {
    term: Vec<u8>,
    doc_freq: i32,
    total_term_freq: i64,
    docs: Vec<Doc>,
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

struct FieldRow {
    number: i32,
    index_options: IndexOptions,
    has_payloads: bool,
    size: i64,
    sum_total_term_freq: i64,
    sum_doc_freq: i64,
    doc_count: i32,
    to_string: String,
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
            .expect("run GenCodecsMemory");
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

    fn file(&self, suffix: &str, ext: &str) -> Vec<u8> {
        let path = self
            .dir
            .join("index")
            .join(format!("{}_{suffix}.{ext}", self.name));
        fs::read(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"))
    }

    fn try_file(&self, suffix: &str, ext: &str) -> Option<Vec<u8>> {
        fs::read(
            self.dir
                .join("index")
                .join(format!("{}_{suffix}.{ext}", self.name)),
        )
        .ok()
    }

    fn read_state<'a>(&'a self, suffix: &'a str) -> SegmentReadState<'a> {
        SegmentReadState {
            max_doc: self.max_doc,
            segment_id: &self.id,
            segment_suffix: suffix,
            field_infos: &self.infos,
        }
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
                        to_string: r[8].clone(),
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

/// The group's `.pos`/`.pay`, opened when present.
struct Group {
    doc: Vec<u8>,
    pos: Vec<u8>,
    pay: Vec<u8>,
}

impl Group {
    fn open(seg: &Segment, suffix: &str) -> Group {
        Group {
            doc: seg.file(suffix, "doc"),
            pos: seg.file(suffix, "pos"),
            pay: seg.file(suffix, "pay"),
        }
    }
}

const FST_FIELDS: [&str; 4] = ["id", "tag", "body", "pay"];

fn check_fst_reader(which: &str) {
    let seg = Segment::open(which);
    let fields = seg.fields();
    let tfp = seg.file(FST_SUFFIX, "tfp");
    let reader = FstPostingsFormat
        .open(&tfp, &seg.read_state(FST_SUFFIX))
        .unwrap();
    assert_eq!(
        reader.field_names().collect::<Vec<_>>(),
        ["body", "id", "pay", "tag"]
    );
    let group = Group::open(&seg, FST_SUFFIX);
    let doc_in = DocInput::open(&group.doc, &seg.id, FST_SUFFIX).unwrap();
    let pos_in = PosInput::open(&group.pos, &seg.id, FST_SUFFIX).unwrap();
    let pay_in = PayInput::open(&group.pay, &seg.id, FST_SUFFIX).unwrap();
    let mut occurrences = 0usize;
    for name in FST_FIELDS {
        let row = &fields[name];
        let terms = reader.terms(name).unwrap();
        assert_eq!(terms.size(), row.size, "{which} {name}");
        assert_eq!(terms.sum_total_term_freq(), row.sum_total_term_freq);
        assert_eq!(terms.sum_doc_freq(), row.sum_doc_freq);
        assert_eq!(terms.doc_count(), row.doc_count);
        assert_eq!(terms.to_string(), row.to_string, "{which} {name}");
        assert_eq!(terms.has_payloads(), row.has_payloads);

        let want = seg.postings(name);
        let mut walk = terms.iter();
        for t in &want {
            assert_eq!(
                walk.next().unwrap(),
                Some(t.term.as_slice()),
                "{which} {name}"
            );
            assert_eq!(
                walk.doc_freq(),
                t.doc_freq,
                "{which} {name} {}",
                hex(&t.term)
            );
            assert_eq!(walk.total_term_freq(), t.total_term_freq);
            let p = walk.postings(&doc_in, PostingsFlags::Freqs).unwrap();
            let docs: Vec<i32> = t.docs.iter().map(|d| d.doc).collect();
            assert_eq!(p.docs, docs, "{which} {name} {}", hex(&t.term));
            if terms.has_freqs() {
                let freqs: Vec<i32> = t.docs.iter().map(|d| d.freq).collect();
                assert_eq!(p.freqs, freqs, "{which} {name} {}", hex(&t.term));
            }
            if terms.has_positions() {
                let positions = postings::read_positions(
                    &pos_in,
                    Some(&pay_in),
                    walk.term_metadata().unwrap(),
                    &p.freqs,
                    walk.total_term_freq(),
                    row.index_options,
                    row.has_payloads,
                )
                .unwrap();
                for (d, got) in t.docs.iter().zip(&positions) {
                    let pos: Vec<i32> = got.iter().map(|p| p.position).collect();
                    assert_eq!(pos, d.positions, "{which} {name} {}", hex(&t.term));
                    if terms.has_offsets() {
                        let offs: Vec<(i32, i32)> =
                            got.iter().map(|p| (p.start_offset, p.end_offset)).collect();
                        assert_eq!(offs, d.offsets);
                    }
                    if terms.has_payloads() {
                        let pays: Vec<Vec<u8>> = got.iter().map(|p| p.payload.clone()).collect();
                        assert_eq!(pays, d.payloads);
                    }
                    occurrences += got.len();
                }
            }
        }
        assert_eq!(walk.next().unwrap(), None, "{which} {name}: extra terms");
        assert!(walk.ord().is_err());
    }
    assert!(occurrences > 10_000, "{which}: {occurrences}");
}

#[test]
fn fst_reader_reads_java_flushed_segment() {
    check_fst_reader("flush");
}

#[test]
fn fst_reader_reads_java_merged_segment() {
    check_fst_reader("merged");
}

fn check_probes(which: &str) {
    let seg = Segment::open(which);
    let tfp = seg.file(FST_SUFFIX, "tfp");
    let reader = FstTermsReader::open(&tfp, &seg.read_state(FST_SUFFIX)).unwrap();
    let mut checked = 0;
    for row in rows(seg.dir.join("probes.tsv")) {
        let Some(terms) = reader.terms(&row[0]) else {
            continue;
        };
        let target = unhex(&row[1]);
        let ctx = format!("{which} {} {}", row[0], row[1]);
        let mut e = terms.iter();
        let status = e.seek_ceil(&target).unwrap();
        let want_status = match row[2].as_str() {
            "FOUND" => SeekStatus::Found,
            "NOT_FOUND" => SeekStatus::NotFound,
            "END" => SeekStatus::End,
            other => panic!("{other}"),
        };
        assert_eq!(status, want_status, "{ctx}");
        let ceil = e
            .term()
            .map_or("-".to_string(), |t| format!("{}:{}", hex(t), e.doc_freq()));
        assert_eq!(ceil, row[3], "{ctx}");
        assert_eq!(
            terms.iter().seek_exact(&target).unwrap().to_string(),
            row[4],
            "{ctx}"
        );
        if status != SeekStatus::End {
            let next = e.next().unwrap().map_or("-".to_string(), hex);
            assert_eq!(next, row[5], "{ctx}");
        }
        checked += 1;
    }
    assert!(checked > 600, "{which}: {checked}");
}

#[test]
fn fst_seeks_match_java() {
    check_probes("flush");
    check_probes("merged");
}

fn compile(pattern: &str) -> CompiledAutomaton {
    let a = RegExp::with_flags(pattern, RegExp::ALL, 0)
        .unwrap()
        .to_automaton()
        .unwrap();
    let a = operations::determinize(&a, operations::DEFAULT_DETERMINIZE_WORK_LIMIT).unwrap();
    CompiledAutomaton::with_options(&a, false, true, false).unwrap()
}

fn check_intersect(which: &str) {
    let seg = Segment::open(which);
    let tfp = seg.file(FST_SUFFIX, "tfp");
    let reader = FstTermsReader::open(&tfp, &seg.read_state(FST_SUFFIX)).unwrap();
    let (mut checked, mut matched) = (0, 0);
    for row in rows(seg.dir.join("intersect.tsv")) {
        let Some(terms) = reader.terms(&row[0]) else {
            continue;
        };
        let compiled = compile(&row[1]);
        if row.len() == 4 && !row[3].contains(':') {
            // Not NORMAL: Java's intersect refuses it, and so does this port.
            assert_eq!(format!("{:?}", compiled.automaton_type), row[3]);
            assert!(terms.intersect(&compiled, None).is_err());
            continue;
        }
        let start = (row[2] != "-").then(|| row[2].as_bytes().to_vec());
        let mut e = terms.intersect(&compiled, start.as_deref()).unwrap();
        let mut got = Vec::new();
        while let Some(t) = e.next().unwrap() {
            let t = hex(t);
            got.push(format!("{t}:{}:{}", e.doc_freq(), e.total_term_freq()));
        }
        assert_eq!(got, row[3..], "{which} {} {} {}", row[0], row[1], row[2]);
        matched += got.len();
        checked += 1;
    }
    assert!(checked > 100, "{which}: {checked}");
    assert!(matched > 1000, "{which}: {matched}");
}

#[test]
fn fst_intersect_matches_java() {
    check_intersect("flush");
    check_intersect("merged");
}

/// The fields of `names`, in name order (Java's), as the postings writer
/// takes them -- every document Java recorded, deleted ones included.
fn inputs(seg: &Segment, names: &[&str]) -> Vec<(i32, IndexOptions, bool, i32, Vec<TermPostings>)> {
    let fields = seg.fields();
    let mut names = names.to_vec();
    names.sort_unstable();
    names
        .iter()
        .map(|name| {
            let row = &fields[*name];
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
                        for d in &t.docs {
                            for p in &d.payloads {
                                tp.payload_lengths.push(p.len() as u32);
                                tp.payload_bytes.extend_from_slice(p);
                            }
                        }
                    }
                    tp
                })
                .collect();
            (
                row.number,
                row.index_options,
                row.has_payloads,
                row.doc_count,
                terms,
            )
        })
        .collect()
}

fn check_fst_rebuild(which: &str) {
    let seg = Segment::open(which);
    let norms = seg.norms();
    let fields = inputs(&seg, &FST_FIELDS);
    let inputs: Vec<FieldPostingsInput<'_>> = fields
        .iter()
        .map(
            |(number, io, payloads, doc_count, terms)| FieldPostingsInput {
                field_number: *number,
                index_options: *io,
                doc_count: *doc_count,
                has_payloads: *payloads,
                terms,
            },
        )
        .collect();
    let field_norms: Vec<FieldNorms<'_>> = FST_FIELDS
        .iter()
        .filter_map(|name| {
            norms.get(*name).map(|values| FieldNorms {
                field_number: seg.infos.field_by_name(name).unwrap().number,
                values,
            })
        })
        .collect();
    let state = SegmentWriteState {
        max_doc: seg.max_doc,
        segment_id: &seg.id,
        segment_suffix: FST_SUFFIX,
        field_infos: &seg.infos,
    };
    let out = FstPostingsFormat
        .write(&inputs, &field_norms, &state)
        .unwrap();
    for (ext, bytes) in [
        ("tfp", &out.tfp),
        ("doc", &out.postings.doc),
        ("pos", &out.postings.pos),
        ("pay", &out.postings.pay),
        ("psm", &out.postings.psm),
    ] {
        let java = seg.file(FST_SUFFIX, ext);
        assert!(
            *bytes == java,
            "{which} .{ext}: {} bytes, Java {} (first difference at {:?})",
            bytes.len(),
            java.len(),
            bytes.iter().zip(&java).position(|(a, b)| a != b)
        );
    }
}

#[test]
fn fst_writer_rebuilds_java_flushed_segment() {
    check_fst_rebuild("flush");
}

#[test]
fn fst_writer_rebuilds_java_merged_segment() {
    check_fst_rebuild("merged");
}

#[test]
fn damaged_java_tfp_is_refused() {
    let seg = Segment::open("flush");
    let tfp = seg.file(FST_SUFFIX, "tfp");
    let state = seg.read_state(FST_SUFFIX);
    assert!(FstTermsReader::open(&tfp, &state).is_ok());
    let mut flipped = tfp.clone();
    let at = flipped.len() / 2;
    flipped[at] ^= 1;
    assert!(FstTermsReader::open(&flipped, &state).is_err(), "checksum");
    assert!(FstTermsReader::open(&tfp[..tfp.len() - 3], &state).is_err());
    let other = SegmentReadState {
        segment_suffix: "FST50_1",
        ..state
    };
    assert!(FstTermsReader::open(&tfp, &other).is_err(), "suffix");
}

// ---- DirectPostingsFormat ----

const DIRECT_FIELDS: [&str; 3] = ["did", "dbody", "dpay"];

/// The `Direct_0` group's files, read whole.
struct DirectGroup {
    tim: Vec<u8>,
    tip: Vec<u8>,
    tmd: Vec<u8>,
    doc: Vec<u8>,
    pos: Option<Vec<u8>>,
    pay: Option<Vec<u8>>,
}

impl DirectGroup {
    fn read(seg: &Segment) -> DirectGroup {
        DirectGroup {
            tim: seg.file(DIRECT_SUFFIX, "tim"),
            tip: seg.file(DIRECT_SUFFIX, "tip"),
            tmd: seg.file(DIRECT_SUFFIX, "tmd"),
            doc: seg.file(DIRECT_SUFFIX, "doc"),
            pos: seg.try_file(DIRECT_SUFFIX, "pos"),
            pay: seg.try_file(DIRECT_SUFFIX, "pay"),
        }
    }

    fn files(&self) -> DirectFiles<'_> {
        DirectFiles {
            tim: &self.tim,
            tip: &self.tip,
            tmd: &self.tmd,
            doc: &self.doc,
            pos: self.pos.as_deref(),
            pay: self.pay.as_deref(),
        }
    }

    fn open(&self, seg: &Segment) -> DirectFields {
        DirectPostingsFormat::default()
            .open(&self.files(), &seg.read_state(DIRECT_SUFFIX))
            .unwrap()
    }
}

/// Every document of `p`, as `postings.FIELD.tsv` records it.
fn drain(
    mut p: DirectPostingsEnum<'_>,
    positions: bool,
    offsets: bool,
    payloads: bool,
) -> Vec<Doc> {
    assert_eq!(p.doc_id(), -1);
    let mut docs = Vec::new();
    loop {
        let doc = p.next_doc();
        if doc == postings::NO_MORE_DOCS {
            break;
        }
        assert_eq!(p.doc_id(), doc);
        let freq = p.freq();
        let mut d = Doc {
            doc,
            freq,
            positions: Vec::new(),
            offsets: Vec::new(),
            payloads: Vec::new(),
        };
        if positions {
            for _ in 0..freq {
                d.positions.push(p.next_position());
                if offsets {
                    d.offsets.push((p.start_offset(), p.end_offset()));
                }
                if payloads {
                    d.payloads
                        .push(p.payload().map_or(Vec::new(), <[u8]>::to_vec));
                }
            }
        }
        docs.push(d);
    }
    assert_eq!(p.doc_id(), postings::NO_MORE_DOCS);
    docs
}

fn check_direct_reader(which: &str) {
    let seg = Segment::open(which);
    let fields = seg.fields();
    let group = DirectGroup::read(&seg);
    let direct = group.open(&seg);
    assert_eq!(direct.to_string(), "DirectFields(fields=3)");
    assert_eq!(direct.size(), 3);
    assert_eq!(
        direct.field_names().collect::<Vec<_>>(),
        ["dbody", "did", "dpay"]
    );
    let (mut occurrences, mut low, mut high) = (0usize, 0usize, 0usize);
    for name in DIRECT_FIELDS {
        let row = &fields[name];
        let terms = direct.terms(name).unwrap();
        assert_eq!(terms.size(), row.size, "{which} {name}");
        assert_eq!(terms.sum_total_term_freq(), row.sum_total_term_freq);
        assert_eq!(terms.sum_doc_freq(), row.sum_doc_freq);
        assert_eq!(terms.doc_count(), row.doc_count);
        assert_eq!(terms.to_string(), row.to_string, "{which} {name}");
        assert_eq!(terms.has_payloads(), row.has_payloads);
        assert_eq!(terms.has_freqs(), row.index_options != IndexOptions::Docs);
        let (positions, offsets, payloads) = (
            terms.has_positions(),
            terms.has_offsets(),
            terms.has_payloads(),
        );
        let want = seg.postings(name);
        let mut walk = terms.iter();
        assert_eq!(walk.term(), None);
        for (ord, t) in want.iter().enumerate() {
            let ctx = format!("{which} {name} {}", hex(&t.term));
            assert_eq!(walk.next(), Some(t.term.as_slice()), "{ctx}");
            assert_eq!(walk.ord(), ord as i64);
            assert_eq!(walk.doc_freq(), t.doc_freq, "{ctx}");
            assert_eq!(walk.total_term_freq(), t.total_term_freq, "{ctx}");
            if t.doc_freq <= 32 {
                low += 1;
            } else {
                high += 1;
            }
            let got = drain(walk.postings(true).unwrap(), positions, offsets, payloads);
            assert_eq!(got, t.docs, "{ctx}");
            occurrences += got.iter().map(|d| d.positions.len()).sum::<usize>();
            // Without positions asked for: the same documents and freqs.
            let docs = drain(walk.postings(false).unwrap(), false, false, false);
            let want: Vec<(i32, i32)> = t.docs.iter().map(|d| (d.doc, d.freq)).collect();
            let got: Vec<(i32, i32)> = docs.iter().map(|d| (d.doc, d.freq)).collect();
            assert_eq!(got, want, "{ctx}");
            // `advance` lands on every document in turn, then runs off the end.
            for positions in [false, true] {
                let mut p = walk.postings(positions).unwrap();
                assert!(p.cost() >= i64::from(t.doc_freq), "{ctx}");
                for d in &t.docs {
                    assert_eq!(p.advance(d.doc), d.doc, "{ctx}");
                }
                assert_eq!(p.advance(i32::MAX - 1), postings::NO_MORE_DOCS);
            }
            // `seekExact(ord)` comes back to this term.
            let mut by_ord = terms.iter();
            by_ord.seek_exact_ord(ord as i64).unwrap();
            assert_eq!(by_ord.term(), Some(t.term.as_slice()));
        }
        assert_eq!(walk.next(), None, "{which} {name}: extra terms");
        assert_eq!(walk.next(), None);
        assert_eq!(walk.doc_freq(), 0);
        assert_eq!(walk.total_term_freq(), 0);
        assert!(walk.postings(true).is_none());
        assert!(terms.iter().seek_exact_ord(row.size).is_err());
        assert!(terms.iter().seek_exact_ord(-1).is_err());
    }
    assert!(occurrences > 10_000, "{which}: {occurrences}");
    assert!(low > 1000 && high > 20, "{which}: {low} low, {high} high");
}

#[test]
fn direct_reader_reads_java_flushed_segment() {
    check_direct_reader("flush");
}

#[test]
fn direct_reader_reads_java_merged_segment() {
    check_direct_reader("merged");
}

fn check_direct_probes(which: &str) {
    let seg = Segment::open(which);
    let group = DirectGroup::read(&seg);
    let direct = group.open(&seg);
    let mut checked = 0;
    for row in rows(seg.dir.join("probes.tsv")) {
        let Some(terms) = direct.terms(&row[0]) else {
            continue;
        };
        let target = unhex(&row[1]);
        let ctx = format!("{which} {} {}", row[0], row[1]);
        let mut e = terms.iter();
        let status = e.seek_ceil(&target);
        let want_status = match row[2].as_str() {
            "FOUND" => SeekStatus::Found,
            "NOT_FOUND" => SeekStatus::NotFound,
            "END" => SeekStatus::End,
            other => panic!("{other}"),
        };
        assert_eq!(status, want_status, "{ctx}");
        let ceil = if status == SeekStatus::End {
            "-".to_string()
        } else {
            format!("{}:{}", hex(e.term().unwrap()), e.doc_freq())
        };
        assert_eq!(ceil, row[3], "{ctx}");
        assert_eq!(
            terms.iter().seek_exact(&target).to_string(),
            row[4],
            "{ctx}"
        );
        if status != SeekStatus::End {
            let next = e.next().map_or("-".to_string(), hex);
            assert_eq!(next, row[5], "{ctx}");
        }
        checked += 1;
    }
    assert!(checked > 400, "{which}: {checked}");
}

#[test]
fn direct_seeks_match_java() {
    check_direct_probes("flush");
    check_direct_probes("merged");
}

fn check_direct_intersect(which: &str) {
    let seg = Segment::open(which);
    let group = DirectGroup::read(&seg);
    let direct = group.open(&seg);
    let (mut checked, mut matched) = (0, 0);
    for row in rows(seg.dir.join("intersect.tsv")) {
        let Some(terms) = direct.terms(&row[0]) else {
            continue;
        };
        let compiled = compile(&row[1]);
        if row.len() == 4 && !row[3].contains(':') {
            assert_eq!(format!("{:?}", compiled.automaton_type), row[3]);
            assert!(terms.intersect(&compiled, None).is_err());
            continue;
        }
        let start = (row[2] != "-").then(|| row[2].as_bytes().to_vec());
        let mut e = terms.intersect(&compiled, start.as_deref()).unwrap();
        assert_eq!(e.term(), None);
        let mut got = Vec::new();
        while let Some(t) = e.next() {
            assert_eq!(e.term(), Some(t));
            assert!(e.ord() >= 0);
            got.push(format!(
                "{}:{}:{}",
                hex(t),
                e.doc_freq(),
                e.total_term_freq()
            ));
            let mut p = e.postings(true).unwrap();
            assert_ne!(p.next_doc(), postings::NO_MORE_DOCS);
        }
        assert_eq!(got, row[3..], "{which} {} {} {}", row[0], row[1], row[2]);
        assert!(e.postings(false).is_none());
        assert!(e.seek_ceil(b"a").is_err());
        matched += got.len();
        checked += 1;
    }
    assert!(checked > 50, "{which}: {checked}");
    assert!(matched > 1000, "{which}: {matched}");
}

#[test]
fn direct_intersect_matches_java() {
    check_direct_intersect("flush");
    check_direct_intersect("merged");
}

fn check_direct_rebuild(which: &str) {
    let seg = Segment::open(which);
    let norms = seg.norms();
    let fields = inputs(&seg, &DIRECT_FIELDS);
    let inputs: Vec<FieldPostingsInput<'_>> = fields
        .iter()
        .map(
            |(number, io, payloads, doc_count, terms)| FieldPostingsInput {
                field_number: *number,
                index_options: *io,
                doc_count: *doc_count,
                has_payloads: *payloads,
                terms,
            },
        )
        .collect();
    let field_norms: Vec<FieldNorms<'_>> = DIRECT_FIELDS
        .iter()
        .filter_map(|name| {
            norms.get(*name).map(|values| FieldNorms {
                field_number: seg.infos.field_by_name(name).unwrap().number,
                values,
            })
        })
        .collect();
    let state = SegmentWriteState {
        max_doc: seg.max_doc,
        segment_id: &seg.id,
        segment_suffix: DIRECT_SUFFIX,
        field_infos: &seg.infos,
    };
    let out = DirectPostingsFormat::default()
        .write(&inputs, &field_norms, &state)
        .unwrap();
    for (ext, bytes) in [
        ("tim", &out.tim),
        ("tip", &out.tip),
        ("tmd", &out.tmd),
        ("doc", &out.doc),
        ("pos", &out.pos),
        ("pay", &out.pay),
        ("psm", &out.psm),
    ] {
        let java = seg.file(DIRECT_SUFFIX, ext);
        assert!(
            *bytes == java,
            "{which} .{ext}: {} bytes, Java {} (first difference at {:?})",
            bytes.len(),
            java.len(),
            bytes.iter().zip(&java).position(|(a, b)| a != b)
        );
    }
}

#[test]
fn direct_writer_rebuilds_java_flushed_segment() {
    check_direct_rebuild("flush");
}

#[test]
fn direct_writer_rebuilds_java_merged_segment() {
    check_direct_rebuild("merged");
}

#[test]
fn damaged_java_direct_files_are_refused() {
    let seg = Segment::open("flush");
    let group = DirectGroup::read(&seg);
    let state = seg.read_state(DIRECT_SUFFIX);
    let format = DirectPostingsFormat::default();
    assert_eq!(format.name(), "Direct");
    assert!(format.open(&group.files(), &state).is_ok());
    for which in ["tim", "tip", "doc", "pos", "pay"] {
        let mut damaged = DirectGroup::read(&seg);
        let buf = match which {
            "tim" => &mut damaged.tim,
            "tip" => &mut damaged.tip,
            "doc" => &mut damaged.doc,
            "pos" => damaged.pos.as_mut().unwrap(),
            _ => damaged.pay.as_mut().unwrap(),
        };
        let at = buf.len() - 20;
        buf[at] ^= 1;
        assert!(format.open(&damaged.files(), &state).is_err(), "{which}");
    }
    let no_pos = DirectFiles {
        pos: None,
        ..group.files()
    };
    assert!(
        format.open(&no_pos, &state).is_err(),
        "positions without .pos"
    );
    let short = DirectFiles {
        doc: &group.doc[..10],
        ..group.files()
    };
    assert!(format.open(&short, &state).is_err());
}
