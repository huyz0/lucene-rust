//! lucene-codecs' `org.apache.lucene.codecs.blockterms` against Java:
//! `GenCodecsBlockTerms` writes a flushed segment and a force-merged one,
//! routing `id`/`pay` (DOCS; offsets and payloads) to a `BlockTermsWriter`
//! over `FixedGapTermsIndexWriter(16)`, `bin`/`tag` (binary terms with the
//! empty one; freqs) over `VariableGapTermsIndexWriter(EveryNTermSelector(20))`
//! and `body` (positions, skewed docFreqs) over
//! `EveryNOrDocFreqTermSelector(40, 24)`, and records every field's
//! statistics and postings (with ords where the index has them),
//! `seekCeil`/`seekExact`/`next` on a reused enum, `seekExact(ord)` on the
//! fixed-gap fields, and `seekExact(term, state)` then `next`.
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

use lucene_codecs::blockterms::{
    self, BlockTermsReader, EveryNOrDocFreqTermSelector, EveryNTermSelector,
    FixedGapTermsIndexReader, FixedGapTermsIndexWriter, SeekStatus, TermsIndexReader,
    VariableGapTermsIndexReader, VariableGapTermsIndexWriter,
};
use lucene_codecs::field_infos::{self, FieldInfos, IndexOptions};
use lucene_codecs::postings::{self, DocInput, PayInput, PosInput, PostingsFlags};
use lucene_codecs::postings_writer::{FieldNorms, FieldPostingsInput, PostingsFiles, TermPostings};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_store::codec_util::ID_LENGTH;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/codecs_blockterms")
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
        .unwrap_or_else(|e| panic!("{path:?}: {e} (run GenCodecsBlockTerms)"))
        .lines()
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect()
}

/// The terms index of a group, by its codec suffix.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    /// `FixedGapTermsIndexWriter(16)`.
    FixedGap,
    /// `EveryNTermSelector(20)`.
    EveryN,
    /// `EveryNOrDocFreqTermSelector(40, 24)`.
    DocFreq,
}

fn kind_of(suffix: &str) -> Kind {
    if suffix.starts_with("BlockTermsFixedGap_") {
        Kind::FixedGap
    } else if suffix.starts_with("BlockTermsVarGapN_") {
        Kind::EveryN
    } else if suffix.starts_with("BlockTermsVarGapDF_") {
        Kind::DocFreq
    } else {
        panic!("{suffix}")
    }
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
    ord: Option<i64>,
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
            .expect("run GenCodecsBlockTerms");
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

    fn path(&self, suffix: &str, ext: &str) -> PathBuf {
        self.dir
            .join("index")
            .join(format!("{}_{suffix}.{ext}", self.name))
    }

    fn file(&self, suffix: &str, ext: &str) -> Vec<u8> {
        let path = self.path(suffix, ext);
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
                        suffix: r[8].clone(),
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
                ord: (r[1] != "-").then(|| r[1].parse().unwrap()),
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

    fn index(&self, suffix: &str) -> blockterms::Result<TermsIndexReader> {
        let state = self.read_state(suffix);
        Ok(match kind_of(suffix) {
            Kind::FixedGap => {
                FixedGapTermsIndexReader::open(&self.file(suffix, "tii"), &state)?.into()
            }
            _ => VariableGapTermsIndexReader::open(
                &self.file(suffix, "tmv"),
                &self.file(suffix, "tiv"),
                &state,
            )?
            .into(),
        })
    }

    fn open_group(&self, suffix: &str) -> BlockTermsReader {
        BlockTermsReader::open(
            self.file(suffix, "tib"),
            self.index(suffix).unwrap(),
            &self.read_state(suffix),
        )
        .unwrap()
    }

    /// Every field's reader, by field name.
    fn readers(&self) -> BTreeMap<String, BlockTermsReader> {
        let mut out = BTreeMap::new();
        for (suffix, names) in self.groups() {
            for name in names {
                out.insert(name, self.open_group(&suffix));
            }
        }
        out
    }
}

/// Every field of the group `suffix` answers `fields.tsv` and
/// `postings.FIELD.tsv`; returns the positions checked.
fn check_group(seg: &Segment, suffix: &str, names: &[String], ctx: &str) -> usize {
    let fields = seg.fields();
    let reader = seg.open_group(suffix);
    reader.check_integrity().unwrap();
    assert_eq!(reader.field_names().collect::<Vec<_>>(), names);
    assert_eq!(reader.size(), names.len());
    let doc = seg.file(suffix, "doc");
    let pos = fs::read(seg.path(suffix, "pos"));
    let pay = fs::read(seg.path(suffix, "pay"));
    let doc_in = DocInput::open(&doc, &seg.id, suffix).unwrap();
    let pos_in = pos
        .as_ref()
        .ok()
        .map(|p| PosInput::open(p, &seg.id, suffix).unwrap());
    let pay_in = pay
        .as_ref()
        .ok()
        .map(|p| PayInput::open(p, &seg.id, suffix).unwrap());
    let mut occurrences = 0;
    for name in names {
        let row = &fields[name];
        let terms = reader.terms(name).unwrap();
        let ctx = format!("{ctx} {suffix} {name}");
        assert_eq!(terms.size(), row.size, "{ctx}");
        assert_eq!(terms.sum_total_term_freq(), row.sum_total_term_freq);
        assert_eq!(terms.sum_doc_freq(), row.sum_doc_freq);
        assert_eq!(terms.doc_count(), row.doc_count);
        assert_eq!(terms.has_payloads(), row.has_payloads);
        assert_eq!(terms.info().number, row.number);

        let want = seg.postings(name);
        assert_eq!(want.len() as i64, row.size);
        let mut walk = terms.iterator();
        for t in &want {
            let tctx = format!("{ctx} {}", hex(&t.term));
            assert_eq!(walk.next().unwrap(), Some(t.term.as_slice()), "{tctx}");
            assert_eq!(walk.term(), t.term.as_slice());
            assert_eq!(walk.ord().ok(), t.ord, "{tctx}");
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
                    walk.term_state().unwrap().meta,
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
                        let pays: Vec<Vec<u8>> = got.iter().map(|p| p.payload.clone()).collect();
                        assert_eq!(pays, d.payloads);
                    }
                    occurrences += got.len();
                }
            }
        }
        assert_eq!(walk.next().unwrap(), None, "{ctx}: extra terms");
        assert_eq!(walk.next().unwrap(), None, "{ctx}: stays at the end");
    }
    occurrences
}

fn check_reader(which: &str) {
    let seg = Segment::open(which);
    let groups = seg.groups();
    assert_eq!(groups.len(), 3, "{which}: three groups");
    let mut occurrences = 0usize;
    for (suffix, names) in &groups {
        occurrences += check_group(&seg, suffix, names, which);
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

/// An ord as the generator prints it: `-` without ord support.
fn ord_text(ord: blockterms::Result<i64>) -> String {
    ord.map_or("-".to_string(), |o| o.to_string())
}

fn check_probes(which: &str) {
    let seg = Segment::open(which);
    let readers = seg.readers();
    let all = rows(seg.dir.join("probes.tsv"));
    let mut checked = 0;
    for (name, reader) in &readers {
        let terms = reader.terms(name).unwrap();
        // One enum, reused across the field's probes as Java's was.
        let mut e = terms.iterator();
        for row in all.iter().filter(|r| r[0] == *name) {
            let target = unhex(&row[1]);
            let ctx = format!("{which} {name} {}", row[1]);
            let got = e.seek_ceil(&target).unwrap();
            assert_eq!(got, status(&row[2]), "{ctx}");
            let ceil = if got == SeekStatus::End {
                "-".to_string()
            } else {
                format!(
                    "{}:{}:{}",
                    hex(e.term()),
                    e.doc_freq().unwrap(),
                    ord_text(e.ord())
                )
            };
            assert_eq!(ceil, row[3], "{ctx}");
            assert_eq!(
                terms.iterator().seek_exact(&target).unwrap().to_string(),
                row[4],
                "{ctx}"
            );
            if got == SeekStatus::End {
                e = terms.iterator();
                assert_eq!(row[5], "-");
            } else {
                let next = e.next().unwrap().map(hex);
                let next = next.map_or("-".to_string(), |n| format!("{n}:{}", ord_text(e.ord())));
                assert_eq!(next, row[5], "{ctx}");
            }
            checked += 1;
        }
    }
    assert_eq!(checked, all.len(), "{which}: every probe's field opened");
    assert!(checked > 1000, "{which}: {checked}");
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
    for (name, reader) in &readers {
        let terms = reader.terms(name).unwrap();
        let mut e = terms.iterator();
        for row in all.iter().filter(|r| r[0] == *name) {
            let ctx = format!("{which} {name} {}", row[1]);
            e.seek_exact_ord(row[1].parse().unwrap()).unwrap();
            assert_eq!(hex(e.term()), row[2], "{ctx}");
            assert_eq!(e.doc_freq().unwrap().to_string(), row[3], "{ctx}");
            assert_eq!(e.ord().unwrap().to_string(), row[4], "{ctx}");
            let next = e.next().unwrap().map(hex);
            let next = next.map_or("-".to_string(), |n| format!("{n}:{}", e.ord().unwrap()));
            assert_eq!(next, row[5], "{ctx}");
            if row[5] == "-" {
                e = terms.iterator();
            }
            checked += 1;
        }
    }
    assert_eq!(checked, all.len(), "{which}: every ord's field opened");
    assert!(checked > 400, "{which}: {checked}");
}

#[test]
fn ords_match_java() {
    check_ords("flush");
    check_ords("merged");
}

fn check_states(which: &str) {
    let seg = Segment::open(which);
    let readers = seg.readers();
    let all = rows(seg.dir.join("states.tsv"));
    let mut checked = 0;
    for row in &all {
        let terms = readers[&row[0]].terms(&row[0]).unwrap();
        let term = unhex(&row[1]);
        let ctx = format!("{which} {} {}", row[0], row[1]);
        let mut a = terms.iterator();
        assert!(a.seek_exact(&term).unwrap(), "{ctx}");
        let state = a.term_state().unwrap();
        let mut b = terms.iterator();
        b.seek_exact_state(&term, &state);
        assert_eq!(b.term(), term.as_slice());
        assert_eq!(b.doc_freq().unwrap().to_string(), row[2], "{ctx}");
        assert_eq!(b.total_term_freq().unwrap().to_string(), row[3], "{ctx}");
        assert_eq!(b.term_state().unwrap(), state);
        for want in &row[4..] {
            let got = b.next().unwrap().map_or("-".to_string(), hex);
            assert_eq!(&got, want, "{ctx}");
        }
        // The same state on the enum it came from, then `next()`.
        a.seek_exact_state(&term, &state);
        let got = a.next().unwrap().map_or("-".to_string(), hex);
        assert_eq!(got, row[4], "{ctx}");
        checked += 1;
    }
    assert!(checked > 400, "{which}: {checked}");
}

#[test]
fn term_states_match_java() {
    check_states("flush");
    check_states("merged");
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

/// `files` against Java's, byte for byte.
fn compare(seg: &Segment, suffix: &str, files: &[(&str, &Vec<u8>)], ctx: &str) {
    for (ext, bytes) in files {
        let java = seg.file(suffix, ext);
        assert!(
            **bytes == java,
            "{ctx} {suffix}.{ext}: {} bytes, Java {} (first difference at {:?})",
            bytes.len(),
            java.len(),
            bytes.iter().zip(&java).position(|(a, b)| a != b)
        );
    }
}

fn postings_files<'a>(postings: &'a PostingsFiles, files: &mut Vec<(&str, &'a Vec<u8>)>) {
    files.push(("doc", &postings.doc));
    files.push(("psm", &postings.psm));
    if !postings.pos.is_empty() {
        files.push(("pos", &postings.pos));
    }
    if !postings.pay.is_empty() {
        files.push(("pay", &postings.pay));
    }
}

fn rebuild_group(seg: &Segment, suffix: &str, names: &[String], ctx: &str) {
    let norms = seg.norms();
    let fields = inputs(seg, names);
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
        segment_suffix: suffix,
        field_infos: &seg.infos,
    };
    match kind_of(suffix) {
        Kind::FixedGap => {
            let w = FixedGapTermsIndexWriter::new(&state, 16).unwrap();
            let out = blockterms::write(w, &inputs, &field_norms, &state).unwrap();
            let mut files = vec![("tib", &out.tib), ("tii", &out.index)];
            postings_files(&out.postings, &mut files);
            compare(seg, suffix, &files, ctx);
        }
        Kind::EveryN => {
            let w = VariableGapTermsIndexWriter::new(&state, EveryNTermSelector::new(20));
            let out = blockterms::write(w, &inputs, &field_norms, &state).unwrap();
            let mut files = vec![
                ("tib", &out.tib),
                ("tiv", &out.index.tiv),
                ("tmv", &out.index.tmv),
            ];
            postings_files(&out.postings, &mut files);
            compare(seg, suffix, &files, ctx);
        }
        Kind::DocFreq => {
            let selector = EveryNOrDocFreqTermSelector::new(40, 24);
            let w = VariableGapTermsIndexWriter::new(&state, selector);
            let out = blockterms::write(w, &inputs, &field_norms, &state).unwrap();
            let mut files = vec![
                ("tib", &out.tib),
                ("tiv", &out.index.tiv),
                ("tmv", &out.index.tmv),
            ];
            postings_files(&out.postings, &mut files);
            compare(seg, suffix, &files, ctx);
        }
    }
}

fn check_rebuild(which: &str) {
    let seg = Segment::open(which);
    for (suffix, names) in seg.groups() {
        rebuild_group(&seg, &suffix, &names, which);
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
    for (suffix, _) in seg.groups() {
        let kind = kind_of(&suffix);
        let state = seg.read_state(&suffix);
        let tib = seg.file(&suffix, "tib");
        let open = |tib: Vec<u8>, state: &SegmentReadState<'_>| {
            seg.index(&suffix)
                .and_then(|index| BlockTermsReader::open(tib, index, state))
        };
        assert!(open(tib.clone(), &state).is_ok());
        // A flipped .tib byte is caught by checkIntegrity (Java only reads
        // the footer's shape at open).
        let mut flipped = tib.clone();
        let at = flipped.len() / 2;
        flipped[at] ^= 1;
        assert!(
            open(flipped, &state).map_or(true, |r| r.check_integrity().is_err()),
            "{suffix} tib"
        );
        // Truncated, the wrong suffix, the wrong segment id.
        assert!(open(tib[..tib.len() - 3].to_vec(), &state).is_err());
        let other = SegmentReadState {
            segment_suffix: "BlockTerms_9",
            ..state
        };
        assert!(BlockTermsReader::open(tib.clone(), seg.index(&suffix).unwrap(), &other).is_err());
        let mut id = seg.id;
        id[0] ^= 1;
        let wrong_id = SegmentReadState {
            segment_id: &id,
            ..state
        };
        assert!(
            BlockTermsReader::open(tib.clone(), seg.index(&suffix).unwrap(), &wrong_id).is_err()
        );

        // The terms index files: a flipped byte fails their whole-file
        // checksums; truncated, the wrong suffix or id, they do not open.
        let exts: &[&str] = if kind == Kind::FixedGap {
            &["tii"]
        } else {
            &["tmv", "tiv"]
        };
        let open_index = |files: &BTreeMap<&str, Vec<u8>>, state: &SegmentReadState<'_>| {
            if kind == Kind::FixedGap {
                FixedGapTermsIndexReader::open(&files["tii"], state).map(|_| ())
            } else {
                VariableGapTermsIndexReader::open(&files["tmv"], &files["tiv"], state).map(|_| ())
            }
        };
        let files: BTreeMap<&str, Vec<u8>> =
            exts.iter().map(|&e| (e, seg.file(&suffix, e))).collect();
        assert!(open_index(&files, &state).is_ok());
        assert!(open_index(&files, &other).is_err(), "{suffix} index suffix");
        assert!(open_index(&files, &wrong_id).is_err(), "{suffix} index id");
        for &ext in exts {
            let mut damaged = files.clone();
            let f = damaged.get_mut(ext).unwrap();
            let at = f.len() / 2;
            f[at] ^= 1;
            assert!(
                open_index(&damaged, &state).is_err(),
                "{suffix}.{ext} flipped"
            );
            let mut short = files.clone();
            let f = short.get_mut(ext).unwrap();
            f.truncate(f.len() - 3);
            assert!(
                open_index(&short, &state).is_err(),
                "{suffix}.{ext} truncated"
            );
        }
    }
}
