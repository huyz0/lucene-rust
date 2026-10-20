//! lucene-codecs' `org.apache.lucene.codecs.uniformsplit.sharedterms`
//! against Java: `GenCodecsSharedTerms` writes a flushed segment and a
//! force-merged one, routing `id`/`body`/`kw`/`title` to
//! `STUniformSplitPostingsFormat()` and `bin`/`note`/`pay`/`tag` (binary
//! terms with the empty one, payloads, offsets, frequencies) to
//! `STUniformSplitPostingsFormat(8, 2, ...)`, the fields of a group sharing
//! many terms, plus the flushed segment's postings again through an
//! XOR-encoding `STUniformSplitPostingsFormat(16, 3, ...)` over all eight
//! fields. It records every field's statistics and postings,
//! `seekCeil`/`seekExact`/`next` on a reused enum (the targets include the
//! other fields' terms), `seekExact(term, state)` then `next`, and
//! `Terms.intersect`.
//!
//! Here the Rust reader opens Java's files and answers every one of those
//! the same way, the merging block reader walks each group's lines with
//! every field's state, and the Rust writer rebuilds each group's files --
//! the merged segment's with the fields in number order, as Java's merge
//! writes them, and the encoded group's -- byte for byte.

// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use lucene_codecs::field_infos::{self, FieldInfos, IndexOptions};
use lucene_codecs::postings::{self, DocInput, PayInput, PosInput, PostingsFlags};
use lucene_codecs::postings_writer::{FieldNorms, FieldPostingsInput, TermPostings};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_codecs::uniformsplit::sharedterms::{
    self, StUniformSplitPostingsFormat, StUniformSplitTermsReader,
};
use lucene_codecs::uniformsplit::{self, BlockDecoder, BlockEncoder, SeekStatus};
use lucene_store::codec_util::ID_LENGTH;
use lucene_util::automaton::{operations, CompiledAutomaton, RegExp};

const ENCODED_SUFFIX: &str = "SharedTermsUniformSplit_Xor";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/codecs_sharedterms")
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
        .unwrap_or_else(|e| panic!("{path:?}: {e} (run GenCodecsSharedTerms)"))
        .lines()
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect()
}

/// The generator's `XorEncoder`: every byte XOR 0x5a, then the (byte) sum
/// of the plain bytes.
struct XorEncoder;

impl BlockEncoder for XorEncoder {
    fn encode(&self, bytes: &[u8]) -> uniformsplit::Result<Vec<u8>> {
        let mut out: Vec<u8> = bytes.iter().map(|b| b ^ 0x5a).collect();
        out.push(bytes.iter().fold(0u8, |s, b| s.wrapping_add(*b)));
        Ok(out)
    }
}

/// The generator's `XorDecoder`, refusing a block whose sum does not match.
struct XorDecoder;

impl BlockDecoder for XorDecoder {
    fn decode(&self, bytes: &[u8]) -> uniformsplit::Result<Vec<u8>> {
        let Some((&sum, body)) = bytes.split_last() else {
            return Err(uniformsplit::Error::IllegalArgument(
                "empty xor block".into(),
            ));
        };
        let out: Vec<u8> = body.iter().map(|b| b ^ 0x5a).collect();
        if out.iter().fold(0u8, |s, b| s.wrapping_add(*b)) != sum {
            return Err(uniformsplit::Error::IllegalArgument(
                "xor block sum mismatch".into(),
            ));
        }
        Ok(out)
    }
}

fn xor_format() -> StUniformSplitPostingsFormat {
    StUniformSplitPostingsFormat::new(
        16,
        3,
        Some(Arc::new(XorEncoder)),
        Some(Arc::new(XorDecoder)),
        false,
    )
    .unwrap()
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
            .expect("run GenCodecsSharedTerms");
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

    /// `_N_SUFFIX.ext` from `index/`, or from `encoded/` for the XOR group.
    fn path(&self, suffix: &str, ext: &str) -> PathBuf {
        let sub = if suffix == ENCODED_SUFFIX {
            "encoded"
        } else {
            "index"
        };
        self.dir
            .join(sub)
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
                        min: unhex(&r[9]),
                        max: unhex(&r[10]),
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

    fn open_group(&self, suffix: &str) -> StUniformSplitTermsReader {
        format_of(
            suffix,
            &self.groups().get(suffix).cloned().unwrap_or_default(),
        )
        .open(
            self.file(suffix, "stustb"),
            &self.file(suffix, "stustd"),
            &self.read_state(suffix),
        )
        .unwrap()
    }

    /// Every field's group reader, by field name.
    fn readers(&self) -> BTreeMap<String, StUniformSplitTermsReader> {
        let mut out = BTreeMap::new();
        for (suffix, names) in self.groups() {
            for name in names {
                out.insert(name, self.open_group(&suffix));
            }
        }
        out
    }
}

/// The format of a group: the small-block one holds `bin`; the XOR one
/// every field.
fn format_of(suffix: &str, names: &[String]) -> StUniformSplitPostingsFormat {
    if suffix == ENCODED_SUFFIX {
        xor_format()
    } else if names.iter().any(|n| n == "bin") {
        StUniformSplitPostingsFormat::new(8, 2, None, None, true).unwrap()
    } else {
        StUniformSplitPostingsFormat::default()
    }
}

/// Every field of the group `suffix` answers `fields.tsv` and
/// `postings.FIELD.tsv`; returns the positions checked.
fn check_group(seg: &Segment, suffix: &str, names: &[String], ctx: &str) -> usize {
    let fields = seg.fields();
    let reader = seg.open_group(suffix);
    reader.check_integrity().unwrap();
    assert_eq!(reader.version(), sharedterms::VERSION_CURRENT);
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
        assert_eq!(terms.min().unwrap().as_deref(), Some(row.min.as_slice()));
        assert_eq!(terms.max(), row.max.as_slice(), "{ctx}");
        assert_eq!(terms.field_info().number, row.number);
        assert_eq!(terms.metadata().field_info().number, row.number);
        assert_eq!(
            terms.metadata().dictionary_start_fp(),
            terms.union_metadata().dictionary_start_fp()
        );

        let want = seg.postings(name);
        assert_eq!(want.len() as i64, row.size);
        let mut walk = terms.iter();
        assert_eq!(walk.term(), None);
        assert!(walk.doc_freq().is_err(), "{ctx}: unpositioned");
        for t in &want {
            let tctx = format!("{ctx} {}", hex(&t.term));
            assert_eq!(walk.next().unwrap(), Some(t.term.as_slice()), "{tctx}");
            assert_eq!(walk.term(), Some(t.term.as_slice()));
            assert_eq!(walk.doc_freq().unwrap(), t.doc_freq, "{tctx}");
            assert_eq!(walk.total_term_freq().unwrap(), t.total_term_freq);
            let p = walk.postings(&doc_in, PostingsFlags::Freqs).unwrap();
            let docs: Vec<i32> = t.docs.iter().map(|d| d.doc).collect();
            assert_eq!(p.docs, docs, "{tctx}");
            let state = walk.term_state().unwrap();
            assert_eq!(
                terms
                    .postings(&state, &doc_in, PostingsFlags::Freqs)
                    .unwrap()
                    .docs,
                docs
            );
            if terms.has_freqs() {
                let freqs: Vec<i32> = t.docs.iter().map(|d| d.freq).collect();
                assert_eq!(p.freqs, freqs, "{tctx}");
            }
            if terms.has_positions() {
                let positions = postings::read_positions(
                    pos_in.as_ref().unwrap(),
                    pay_in.as_ref(),
                    state.meta,
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
    }
    occurrences
}

fn check_reader(which: &str) {
    let seg = Segment::open(which);
    let groups = seg.groups();
    assert_eq!(groups.len(), 2, "{which}: two groups");
    let mut occurrences = 0usize;
    for (suffix, names) in &groups {
        assert_eq!(names.len(), 4, "{which} {suffix}");
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

#[test]
fn reader_reads_java_encoded_group() {
    let seg = Segment::open("flush");
    let names: Vec<String> = seg.fields().into_keys().collect();
    assert_eq!(names.len(), 8);
    let occurrences = check_group(&seg, ENCODED_SUFFIX, &names, "encoded");
    assert!(occurrences > 10_000, "{occurrences}");
    // Without its decoder the group does not read.
    let unencoded = StUniformSplitPostingsFormat::default().open(
        seg.file(ENCODED_SUFFIX, "stustb"),
        &seg.file(ENCODED_SUFFIX, "stustd"),
        &seg.read_state(ENCODED_SUFFIX),
    );
    assert!(unencoded.is_err());
}

/// `STMergingBlockReader` over each group: one line per distinct term of
/// the group's fields, each with exactly the fields that have it, in
/// number order, whose states read Java's postings.
fn check_merging(seg: &Segment, suffix: &str, names: &[String], ctx: &str) -> usize {
    let fields = seg.fields();
    // term -> (field number, docs) of every field that has it.
    let mut lines: BTreeMap<Vec<u8>, Vec<(i32, Vec<i32>)>> = BTreeMap::new();
    for name in names {
        for t in seg.postings(name) {
            let docs = t.docs.iter().map(|d| d.doc).collect();
            lines
                .entry(t.term)
                .or_default()
                .push((fields[name].number, docs));
        }
    }
    let mut shared = 0;
    for states in lines.values_mut() {
        states.sort_by_key(|(number, _)| *number);
        shared += usize::from(states.len() > 1);
    }
    assert!(shared > 100, "{ctx} {suffix}: {shared} shared terms");
    let reader = seg.open_group(suffix);
    let doc = seg.file(suffix, "doc");
    let doc_in = DocInput::open(&doc, &seg.id, suffix).unwrap();
    let terms = reader.terms(&names[0]).unwrap();
    let mut merging = terms.merging_block_reader();
    assert!(merging.seek_ceil(b"a").is_err());
    assert!(merging.seek_exact(b"a").is_err());
    let mut out = Vec::new();
    merging.read_field_term_states(&mut out).unwrap();
    assert!(out.is_empty(), "{ctx}: unpositioned");
    for (term, want) in &lines {
        let lctx = format!("{ctx} {suffix} {}", hex(term));
        assert_eq!(merging.next().unwrap(), Some(term.as_slice()), "{lctx}");
        assert_eq!(merging.term(), Some(term.as_slice()));
        merging.read_field_term_states(&mut out).unwrap();
        let numbers: Vec<i32> = out.iter().map(|(n, _)| *n).collect();
        let want_numbers: Vec<i32> = want.iter().map(|(n, _)| *n).collect();
        assert_eq!(numbers, want_numbers, "{lctx}");
        for ((number, state), (_, docs)) in out.iter().zip(want) {
            let got = merging
                .postings(*number, state, &doc_in, PostingsFlags::DocsOnly)
                .unwrap();
            assert_eq!(&got.docs, docs, "{lctx} {number}");
        }
    }
    assert_eq!(merging.next().unwrap(), None, "{ctx} {suffix}: extra lines");
    assert!(merging
        .postings(i32::MAX, &out[0].1, &doc_in, PostingsFlags::DocsOnly)
        .is_err());
    lines.len()
}

#[test]
fn merging_block_reader_reads_every_line() {
    for which in ["flush", "merged"] {
        let seg = Segment::open(which);
        for (suffix, names) in seg.groups() {
            let n = check_merging(&seg, &suffix, &names, which);
            assert!(n > 1000, "{which} {suffix}: {n}");
        }
    }
    let seg = Segment::open("flush");
    let names: Vec<String> = seg.fields().into_keys().collect();
    check_merging(&seg, ENCODED_SUFFIX, &names, "encoded");
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
    for (name, reader) in &readers {
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
                format!("{}:{}", hex(e.term().unwrap()), e.doc_freq().unwrap())
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
                let next = e.next().unwrap().map_or("-".to_string(), hex);
                assert_eq!(next, row[5], "{ctx}");
            }
            checked += 1;
        }
    }
    assert_eq!(checked, all.len(), "{which}: every probe's field opened");
    assert!(checked > 6000, "{which}: {checked}");
}

#[test]
fn seeks_match_java() {
    check_probes("flush");
    check_probes("merged");
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
        let mut a = terms.iter();
        assert!(a.seek_exact(&term).unwrap(), "{ctx}");
        let state = a.term_state().unwrap();
        let mut b = terms.iter();
        b.seek_exact_state(&term, &state);
        assert_eq!(b.term(), Some(term.as_slice()));
        assert_eq!(b.doc_freq().unwrap().to_string(), row[2], "{ctx}");
        assert_eq!(b.total_term_freq().unwrap().to_string(), row[3], "{ctx}");
        assert_eq!(b.term_state().unwrap(), state);
        for want in &row[4..] {
            let got = b.next().unwrap().map_or("-".to_string(), hex);
            assert_eq!(&got, want, "{ctx}");
        }
        checked += 1;
    }
    assert!(checked > 600, "{which}: {checked}");
}

#[test]
fn term_states_match_java() {
    check_states("flush");
    check_states("merged");
}

fn compile(pattern: &str) -> CompiledAutomaton {
    let a = RegExp::with_flags(pattern, RegExp::ALL, 0)
        .unwrap()
        .to_automaton()
        .unwrap();
    let a = operations::determinize(&a, operations::DEFAULT_DETERMINIZE_WORK_LIMIT).unwrap();
    CompiledAutomaton::with_options(&a, false, true, false).unwrap()
}

/// `intersect.tsv` against `encoded` (`None`: each field's own group).
fn check_intersect(seg: &Segment, encoded: Option<&StUniformSplitTermsReader>, ctx: &str) {
    let readers = seg.readers();
    let (mut checked, mut matched, mut refused) = (0, 0, 0);
    for row in rows(seg.dir.join("intersect.tsv")) {
        let reader = encoded.unwrap_or(&readers[&row[0]]);
        let terms = reader.terms(&row[0]).unwrap();
        let compiled = compile(&row[1]);
        let rctx = format!("{ctx} {} {} {}", row[0], row[1], row[2]);
        if row.len() == 4 && !row[3].contains(':') {
            // Not NORMAL: Java's intersect refuses it, and so does this port.
            assert_eq!(format!("{:?}", compiled.automaton_type), row[3], "{rctx}");
            assert!(terms.intersect(&compiled, None).is_err());
            refused += 1;
            continue;
        }
        let start = (row[2] != "-").then(|| row[2].as_bytes().to_vec());
        let mut e = terms.intersect(&compiled, start.as_deref()).unwrap();
        let mut got = Vec::new();
        while let Some(t) = e.next().unwrap() {
            let t = hex(t);
            assert_eq!(e.term().map(hex), Some(t.clone()));
            let state = e.term_state().unwrap();
            assert_eq!(state.doc_freq, e.doc_freq().unwrap());
            got.push(format!(
                "{t}:{}:{}",
                e.doc_freq().unwrap(),
                e.total_term_freq().unwrap()
            ));
        }
        assert_eq!(got, row[3..], "{rctx}");
        assert!(e.seek_ceil(b"a").is_err());
        matched += got.len();
        checked += 1;
    }
    eprintln!("{ctx}: {checked} intersections, {matched} terms");
    assert!(checked > 150, "{ctx}: {checked}");
    assert!(matched > 3000, "{ctx}: {matched}");
    assert!(refused > 0, "{ctx}: no non-NORMAL automaton");
}

#[test]
fn intersect_matches_java() {
    for which in ["flush", "merged"] {
        check_intersect(&Segment::open(which), None, which);
    }
    let seg = Segment::open("flush");
    let encoded = seg.open_group(ENCODED_SUFFIX);
    check_intersect(&seg, Some(&encoded), "encoded");
}

/// The fields of `names`, in that order, as the postings writer takes them.
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

/// `names` in the order Java's writer took them: name order for a flush,
/// number order (`mergeFieldInfos`) for a merge.
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
    let out = format_of(suffix, names)
        .write(&inputs, &field_norms, &state)
        .unwrap();
    let mut files = vec![
        ("stustb", &out.ustb),
        ("stustd", &out.ustd),
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
        let java = seg.file(suffix, ext);
        assert!(
            *bytes == java,
            "{ctx} {suffix}.{ext}: {} bytes, Java {} (first difference at {:?})",
            bytes.len(),
            java.len(),
            bytes.iter().zip(&java).position(|(a, b)| a != b)
        );
    }
}

#[test]
fn writer_rebuilds_java_flushed_segment() {
    let seg = Segment::open("flush");
    for (suffix, names) in seg.groups() {
        rebuild_group(&seg, &suffix, &names, "flush");
    }
}

#[test]
fn writer_rebuilds_java_merged_segment() {
    let seg = Segment::open("merged");
    let fields = seg.fields();
    for (suffix, mut names) in seg.groups() {
        names.sort_by_key(|n| fields[n].number);
        rebuild_group(&seg, &suffix, &names, "merged");
    }
}

#[test]
fn writer_rebuilds_java_encoded_group() {
    let seg = Segment::open("flush");
    let names: Vec<String> = seg.fields().into_keys().collect();
    rebuild_group(&seg, ENCODED_SUFFIX, &names, "encoded");
}

#[test]
fn damaged_java_files_are_refused() {
    let seg = Segment::open("flush");
    let suffix = "SharedTermsUniformSplit_0";
    let stustb = seg.file(suffix, "stustb");
    let stustd = seg.file(suffix, "stustd");
    let state = seg.read_state(suffix);
    let format = StUniformSplitPostingsFormat::new(8, 2, None, None, true).unwrap();
    assert!(format.open(stustb.clone(), &stustd, &state).is_ok());
    // A flipped .stustd byte fails its whole-file checksum.
    let mut flipped = stustd.clone();
    let at = flipped.len() / 2;
    flipped[at] ^= 1;
    assert!(
        format.open(stustb.clone(), &flipped, &state).is_err(),
        "stustd"
    );
    // A flipped .stustb byte is caught by checkIntegrity.
    let mut flipped = stustb.clone();
    let at = flipped.len() / 2;
    flipped[at] ^= 1;
    let damaged = format.open(flipped, &stustd, &state);
    assert!(
        damaged.map_or(true, |r| r.check_integrity().is_err()),
        "stustb"
    );
    // Truncated files.
    assert!(format
        .open(stustb[..stustb.len() - 3].to_vec(), &stustd, &state)
        .is_err());
    assert!(format
        .open(stustb.clone(), &stustd[..stustd.len() - 3], &state)
        .is_err());
    // The wrong suffix, and the wrong segment id.
    let other = SegmentReadState {
        segment_suffix: "SharedTermsUniformSplit_1",
        ..state
    };
    assert!(
        format.open(stustb.clone(), &stustd, &other).is_err(),
        "suffix"
    );
    let mut id = seg.id;
    id[0] ^= 1;
    let wrong_id = SegmentReadState {
        segment_id: &id,
        ..state
    };
    assert!(
        format.open(stustb.clone(), &stustd, &wrong_id).is_err(),
        "id"
    );
    // The other group's .stustd beside this .stustb.
    let stustd1 = seg.file("SharedTermsUniformSplit_1", "stustd");
    assert!(
        format.open(stustb.clone(), &stustd1, &state).is_err(),
        "mismatched pair"
    );
    // UniformSplit's reader refuses the shared-terms codec name.
    assert!(
        uniformsplit::UniformSplitPostingsFormat::new(8, 2, None, None, true)
            .unwrap()
            .open(stustb, &stustd, &state)
            .is_err()
    );
    // An encoded block whose sum is off is the decoder's error, at the
    // first read of that block.
    let mut stustb = seg.file(ENCODED_SUFFIX, "stustb");
    let stustd = seg.file(ENCODED_SUFFIX, "stustd");
    let enc_state = seg.read_state(ENCODED_SUFFIX);
    let first = xor_format()
        .open(stustb.clone(), &stustd, &enc_state)
        .unwrap()
        .terms("bin")
        .unwrap()
        .metadata()
        .first_block_start_fp();
    stustb[first as usize + 3] ^= 0x10;
    let reader = xor_format().open(stustb, &stustd, &enc_state).unwrap();
    assert!(reader.check_integrity().is_err());
    let bin = reader.terms("bin").unwrap();
    assert!(bin.iter().next().is_err());
}
