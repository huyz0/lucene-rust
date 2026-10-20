//! lucene-codecs' `org.apache.lucene.codecs.blocktreeords` against Java:
//! `GenCodecsBlockTreeOrds` writes a flushed segment and a force-merged one,
//! routing `id`/`body` (DOCS, positions) to `BlockTreeOrdsPostingsFormat()`
//! and `bin`/`tag`/`pay` (long binary terms, DOCS_AND_FREQS, offsets and
//! payloads) to `BlockTreeOrdsPostingsFormat(10, 20)`, and records every
//! field's statistics, every term's ord and postings, `seekCeil`/`seekExact`/
//! `next` on a reused enum, `seekExact(ord)` then `next`, and
//! `Terms.intersect` over a set of patterns and start terms.
//!
//! Here the Rust reader opens Java's files and answers every one of those
//! the same way, and the Rust writer rebuilds each group's files byte for
//! byte from the postings Java recorded.

// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::blocktreeords::{BlockTreeOrdsPostingsFormat, OrdsBlockTreeTermsReader};
use lucene_codecs::field_infos::{self, FieldInfos, IndexOptions};
use lucene_codecs::postings::{self, DocInput, PayInput, PosInput, PostingsFlags};
use lucene_codecs::postings_writer::{FieldNorms, FieldPostingsInput, TermPostings};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_store::codec_util::ID_LENGTH;
use lucene_util::automaton::{operations, CompiledAutomaton, RegExp};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/codecs_blocktreeords")
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
        .unwrap_or_else(|e| panic!("{path:?}: {e} (run GenCodecsBlockTreeOrds)"))
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
    ord: i64,
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
    suffix: String,
    min: Vec<u8>,
    max: Vec<u8>,
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
            .expect("run GenCodecsBlockTreeOrds");
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
                        suffix: r[9].clone(),
                        min: unhex(&r[10]),
                        max: unhex(&r[11]),
                    },
                )
            })
            .collect()
    }

    /// The groups: suffix -> its fields in name order.
    fn groups(&self) -> BTreeMap<String, Vec<String>> {
        let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (name, row) in self.fields() {
            groups.entry(row.suffix.clone()).or_default().push(name);
        }
        groups
    }

    fn postings(&self, field: &str) -> Vec<TermRow> {
        rows(self.dir.join(format!("postings.{field}.tsv")))
            .into_iter()
            .map(|r| TermRow {
                term: unhex(&r[0]),
                ord: r[1].parse().unwrap(),
                doc_freq: r[2].parse().unwrap(),
                total_term_freq: r[3].parse().unwrap(),
                docs: r[4..].iter().map(|d| parse_doc(d)).collect(),
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

    fn open_group(&self, suffix: &str) -> OrdsBlockTreeTermsReader {
        BlockTreeOrdsPostingsFormat::default()
            .open(
                self.file(suffix, "tio"),
                &self.file(suffix, "tipo"),
                &self.read_state(suffix),
            )
            .unwrap()
    }

    /// Every field's reader, by field name.
    fn readers(&self) -> BTreeMap<String, (String, OrdsBlockTreeTermsReader)> {
        let mut out = BTreeMap::new();
        for (suffix, names) in self.groups() {
            for name in names {
                out.insert(name, (suffix.clone(), self.open_group(&suffix)));
            }
        }
        out
    }
}

fn check_reader(which: &str) {
    let seg = Segment::open(which);
    let fields = seg.fields();
    let groups = seg.groups();
    assert_eq!(groups.len(), 2, "{which}: two groups");
    let mut occurrences = 0usize;
    for (suffix, names) in &groups {
        let reader = seg.open_group(suffix);
        reader.check_integrity().unwrap();
        assert_eq!(reader.field_names().collect::<Vec<_>>(), *names);
        assert_eq!(reader.size(), names.len());
        assert_eq!(
            reader.to_string(),
            format!(
                "OrdsBlockTreeTermsReader(fields={},delegate=Lucene104PostingsReader)",
                names.len()
            )
        );
        let doc = seg.file(suffix, "doc");
        let pos = fs::read(
            seg.dir
                .join("index")
                .join(format!("{}_{suffix}.pos", seg.name)),
        );
        let pay = fs::read(
            seg.dir
                .join("index")
                .join(format!("{}_{suffix}.pay", seg.name)),
        );
        let doc_in = DocInput::open(&doc, &seg.id, suffix).unwrap();
        let pos_in = pos
            .as_ref()
            .ok()
            .map(|p| PosInput::open(p, &seg.id, suffix).unwrap());
        let pay_in = pay
            .as_ref()
            .ok()
            .map(|p| PayInput::open(p, &seg.id, suffix).unwrap());
        for name in names {
            let row = &fields[name];
            let terms = reader.terms(name).unwrap();
            let ctx = format!("{which} {name}");
            assert_eq!(terms.size(), row.size, "{ctx}");
            assert_eq!(terms.sum_total_term_freq(), row.sum_total_term_freq);
            assert_eq!(terms.sum_doc_freq(), row.sum_doc_freq);
            assert_eq!(terms.doc_count(), row.doc_count);
            assert_eq!(terms.to_string(), row.to_string, "{ctx}");
            assert_eq!(terms.has_payloads(), row.has_payloads);
            assert_eq!(terms.min(), row.min.as_slice(), "{ctx}");
            assert_eq!(terms.max(), row.max.as_slice(), "{ctx}");
            assert_eq!(terms.field_info().number, row.number);

            let want = seg.postings(name);
            assert_eq!(want.len() as i64, row.size);
            let mut walk = terms.iter();
            assert_eq!(walk.ord(), -1);
            for t in &want {
                let tctx = format!("{ctx} {}", hex(&t.term));
                assert_eq!(walk.next().unwrap(), Some(t.term.as_slice()), "{tctx}");
                assert_eq!(walk.term(), t.term.as_slice());
                assert_eq!(walk.ord(), t.ord, "{tctx}");
                assert_eq!(walk.doc_freq().unwrap(), t.doc_freq, "{tctx}");
                assert_eq!(walk.total_term_freq().unwrap(), t.total_term_freq);
                let p = walk.postings(&doc_in, PostingsFlags::Freqs).unwrap();
                let docs: Vec<i32> = t.docs.iter().map(|d| d.doc).collect();
                assert_eq!(p.docs, docs, "{tctx}");
                if terms.has_freqs() {
                    let freqs: Vec<i32> = t.docs.iter().map(|d| d.freq).collect();
                    assert_eq!(p.freqs, freqs, "{tctx}");
                }
                if terms.has_positions() {
                    let positions = postings::read_positions(
                        pos_in.as_ref().unwrap(),
                        pay_in.as_ref(),
                        walk.term_metadata().unwrap(),
                        &p.freqs,
                        walk.total_term_freq().unwrap(),
                        row.index_options,
                        row.has_payloads,
                    )
                    .unwrap();
                    for (d, got) in t.docs.iter().zip(&positions) {
                        let pos: Vec<i32> = got.iter().map(|p| p.position).collect();
                        assert_eq!(pos, d.positions, "{tctx}");
                        if terms.has_offsets() {
                            let offs: Vec<(i32, i32)> =
                                got.iter().map(|p| (p.start_offset, p.end_offset)).collect();
                            assert_eq!(offs, d.offsets);
                        }
                        if terms.has_payloads() {
                            let pays: Vec<Vec<u8>> =
                                got.iter().map(|p| p.payload.clone()).collect();
                            assert_eq!(pays, d.payloads);
                        }
                        occurrences += got.len();
                    }
                }
            }
            assert_eq!(walk.next().unwrap(), None, "{ctx}: extra terms");
            // A term state taken from one enum positions another.
            let mut a = terms.iter();
            let t = &want[want.len() / 2];
            assert!(a.seek_exact(&t.term).unwrap());
            let state = a.term_state().unwrap();
            assert_eq!(state.ord, t.ord);
            let mut b = terms.iter();
            b.seek_exact_state(&t.term, state);
            assert_eq!(b.ord(), t.ord);
            assert_eq!(b.doc_freq().unwrap(), t.doc_freq);
            // Java's quirk: an enum that never loaded a block starts over
            // from the root on `next()`, whatever the state put it on.
            assert_eq!(b.next().unwrap(), Some(want[0].term.as_slice()), "{ctx}");
            // Once loaded, `next()` re-seeks the state's term and steps on.
            b.seek_exact_state(&t.term, state);
            let next = &want[want.len() / 2 + 1];
            assert_eq!(b.next().unwrap(), Some(next.term.as_slice()), "{ctx}");
            assert_eq!(b.ord(), next.ord);
            // The same term's state on its own enum changes nothing.
            a.seek_exact_state(&t.term, state);
            assert_eq!(a.ord(), t.ord);
            // Out-of-range ords are Java's IllegalArgumentException.
            let err = terms.iter().seek_exact_ord(row.size).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!(
                    "targetOrd out of bounds (got: {}, numTerms={})",
                    row.size, row.size
                )
            );
            assert!(terms.iter().seek_exact_ord(-1).is_err());
        }
    }
    assert!(occurrences > 10_000, "{which}: {occurrences}");
}

#[test]
fn reader_reads_java_flushed_segment() {
    check_reader("flush");
}

#[test]
fn reader_reads_java_merged_segment() {
    check_reader("merged");
}

fn status(s: &str) -> SeekStatus {
    match s {
        "FOUND" => SeekStatus::Found,
        "NOT_FOUND" => SeekStatus::NotFound,
        "END" => SeekStatus::End,
        other => panic!("{other}"),
    }
}

fn check_probes(which: &str) {
    let seg = Segment::open(which);
    let readers = seg.readers();
    let all = rows(seg.dir.join("probes.tsv"));
    let mut checked = 0;
    for (name, (_, reader)) in &readers {
        let terms = reader.terms(name).unwrap();
        // One enum, reused across the field's probes as Java's was.
        let mut e = terms.iter();
        for row in all.iter().filter(|r| r[0] == *name) {
            let target = unhex(&row[1]);
            let ctx = format!("{which} {name} {}", row[1]);
            let got = e.seek_ceil(&target).unwrap();
            assert_eq!(got, status(&row[2]), "{ctx}");
            let ceil = if got == SeekStatus::End {
                "-".to_string()
            } else {
                format!("{}:{}:{}", hex(e.term()), e.doc_freq().unwrap(), e.ord())
            };
            assert_eq!(ceil, row[3], "{ctx}");
            assert_eq!(
                terms.iter().seek_exact(&target).unwrap().to_string(),
                row[4],
                "{ctx}"
            );
            if got == SeekStatus::End {
                e = terms.iter();
            } else {
                let next = match e.next().unwrap() {
                    Some(n) => format!("{}:{}", hex(n), e.ord()),
                    None => "-".to_string(),
                };
                assert_eq!(next, row[5], "{ctx}");
            }
            checked += 1;
        }
    }
    assert_eq!(checked, all.len(), "{which}: every probe's field opened");
    assert!(checked > 600, "{which}: {checked}");
}

#[test]
fn seeks_match_java() {
    check_probes("flush");
    check_probes("merged");
}

fn check_ords(which: &str) {
    let seg = Segment::open(which);
    let readers = seg.readers();
    let all = rows(seg.dir.join("ords.tsv"));
    let mut checked = 0;
    for (name, (_, reader)) in &readers {
        let terms = reader.terms(name).unwrap();
        let mut e = terms.iter();
        for row in all.iter().filter(|r| r[0] == *name) {
            let ord: i64 = row[1].parse().unwrap();
            let ctx = format!("{which} {name} {ord}");
            e.seek_exact_ord(ord).unwrap();
            assert_eq!(hex(e.term()), row[2], "{ctx}");
            assert_eq!(e.doc_freq().unwrap().to_string(), row[3], "{ctx}");
            assert_eq!(e.ord().to_string(), row[4], "{ctx}");
            let next = match e.next().unwrap() {
                Some(n) => format!("{}:{}", hex(n), e.ord()),
                None => "-".to_string(),
            };
            assert_eq!(next, row[5], "{ctx}");
            if next == "-" {
                e = terms.iter();
            }
            checked += 1;
        }
    }
    assert_eq!(checked, all.len());
    assert!(checked > 1000, "{which}: {checked}");
}

#[test]
fn ord_seeks_match_java() {
    check_ords("flush");
    check_ords("merged");
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
    let readers = seg.readers();
    let (mut checked, mut matched, mut refused) = (0, 0, 0);
    for row in rows(seg.dir.join("intersect.tsv")) {
        let (_, reader) = &readers[&row[0]];
        let terms = reader.terms(&row[0]).unwrap();
        let compiled = compile(&row[1]);
        let ctx = format!("{which} {} {} {}", row[0], row[1], row[2]);
        if row.len() == 4 && !row[3].contains(':') {
            // Not NORMAL: Java's intersect refuses it, and so does this port.
            assert_eq!(format!("{:?}", compiled.automaton_type), row[3], "{ctx}");
            assert!(terms.intersect(&compiled, None).is_err());
            refused += 1;
            continue;
        }
        let start = (row[2] != "-").then(|| row[2].as_bytes().to_vec());
        let mut e = terms.intersect(&compiled, start.as_deref()).unwrap();
        let mut got = Vec::new();
        while let Some(t) = e.next().unwrap() {
            let t = hex(t);
            assert_eq!(hex(e.term()), t);
            got.push(format!(
                "{t}:{}:{}",
                e.doc_freq().unwrap(),
                e.total_term_freq().unwrap()
            ));
        }
        assert_eq!(got, row[3..], "{ctx}");
        assert!(e.ord().is_err());
        assert!(e.seek().is_err());
        matched += got.len();
        checked += 1;
    }
    eprintln!("{which}: {checked} intersections, {matched} terms");
    assert!(checked > 150, "{which}: {checked}");
    assert!(matched > 3000, "{which}: {matched}");
    assert!(refused > 0, "{which}: no non-NORMAL automaton");
}

#[test]
fn intersect_matches_java() {
    check_intersect("flush");
    check_intersect("merged");
}

/// The fields of `names` (in name order, Java's) as the postings writer
/// takes them.
fn inputs(
    seg: &Segment,
    names: &[String],
) -> Vec<(i32, IndexOptions, bool, i32, Vec<TermPostings>)> {
    let fields = seg.fields();
    names
        .iter()
        .map(|name| {
            let row = &fields[name];
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

/// The format of a group: the small-block one holds `bin`.
fn format_of(names: &[String]) -> BlockTreeOrdsPostingsFormat {
    if names.iter().any(|n| n == "bin") {
        BlockTreeOrdsPostingsFormat::new(10, 20).unwrap()
    } else {
        BlockTreeOrdsPostingsFormat::default()
    }
}

fn check_rebuild(which: &str) {
    let seg = Segment::open(which);
    let norms = seg.norms();
    for (suffix, names) in seg.groups() {
        let fields = inputs(&seg, &names);
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
        let field_norms: Vec<FieldNorms<'_>> = names
            .iter()
            .filter_map(|name| {
                norms.get(name).map(|values| FieldNorms {
                    field_number: seg.infos.field_by_name(name).unwrap().number,
                    values,
                })
            })
            .collect();
        let state = SegmentWriteState {
            max_doc: seg.max_doc,
            segment_id: &seg.id,
            segment_suffix: &suffix,
            field_infos: &seg.infos,
        };
        let out = format_of(&names)
            .write(&inputs, &field_norms, &state)
            .unwrap();
        let mut files = vec![
            ("tio", &out.tio),
            ("tipo", &out.tipo),
            ("doc", &out.postings.doc),
            ("psm", &out.postings.psm),
        ];
        if !out.postings.pos.is_empty() {
            files.push(("pos", &out.postings.pos));
        }
        if !out.postings.pay.is_empty() {
            files.push(("pay", &out.postings.pay));
        }
        for (ext, bytes) in files {
            let java = seg.file(&suffix, ext);
            assert!(
                *bytes == java,
                "{which} {suffix}.{ext}: {} bytes, Java {} (first difference at {:?})",
                bytes.len(),
                java.len(),
                bytes.iter().zip(&java).position(|(a, b)| a != b)
            );
        }
    }
}

#[test]
fn writer_rebuilds_java_flushed_segment() {
    check_rebuild("flush");
}

#[test]
fn writer_rebuilds_java_merged_segment() {
    check_rebuild("merged");
}

#[test]
fn damaged_java_files_are_refused() {
    let seg = Segment::open("flush");
    let suffix = "BlockTreeOrds_0";
    let tio = seg.file(suffix, "tio");
    let tipo = seg.file(suffix, "tipo");
    let state = seg.read_state(suffix);
    let format = BlockTreeOrdsPostingsFormat::default();
    assert_eq!(format.name(), "BlockTreeOrds(blocksize=128)");
    assert!(format.open(tio.clone(), &tipo, &state).is_ok());
    // A flipped .tipo byte fails its whole-file checksum.
    let mut flipped = tipo.clone();
    let at = flipped.len() / 2;
    flipped[at] ^= 1;
    assert!(format.open(tio.clone(), &flipped, &state).is_err(), "tipo");
    // A flipped .tio byte is caught by checkIntegrity (Java only reads the
    // footer at open).
    let mut flipped = tio.clone();
    let at = flipped.len() / 2;
    flipped[at] ^= 1;
    let damaged = format.open(flipped, &tipo, &state);
    assert!(
        damaged.map_or(true, |r| r.check_integrity().is_err()),
        "tio"
    );
    // Truncated files.
    assert!(format
        .open(tio[..tio.len() - 3].to_vec(), &tipo, &state)
        .is_err());
    assert!(format
        .open(tio.clone(), &tipo[..tipo.len() - 3], &state)
        .is_err());
    // The wrong suffix.
    let other = SegmentReadState {
        segment_suffix: "BlockTreeOrds_1",
        ..state
    };
    assert!(format.open(tio.clone(), &tipo, &other).is_err(), "suffix");
    // The other group's .tipo beside this .tio.
    let tipo1 = seg.file("BlockTreeOrds_1", "tipo");
    assert!(format.open(tio, &tipo1, &state).is_err(), "mismatched pair");
}
