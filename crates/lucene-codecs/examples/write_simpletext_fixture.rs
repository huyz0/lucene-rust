//! Writes a SimpleText postings file (lucene-codecs' `simpletext` package),
//! `_0_SimpleText_0.pst`, plus its expected contents, to the directory given
//! as the first CLI argument.
//!
//! Reverse-direction fixture (Rust writes, Java reads): the counterpart
//! `fixtures/src/VerifyCodecsSimpleText.java` opens the file through real
//! Lucene's own `SimpleTextCodec().postingsFormat()` over a hand-built
//! `SegmentInfo`/`FieldInfos` (no `.si`/`.fnm` from Rust), walks it against
//! `terms.tsv`, then has Lucene's own writer re-write it from its own reader
//! with the norms of `norms.tsv` and requires the two files to be equal byte
//! for byte -- skip lists and their impacts included.
//!
//! The fields: `id` (docs only, no norms, with the empty term), `bin`
//! (docs only, binary terms holding `\n`, `\` and `0xff`), `tag` (freqs) and
//! `body` (positions, offsets and payloads that hold `\n` and `\`), the last
//! two with norms; enough documents that `body`'s common terms skip over
//! three levels.
//!
//! `terms.tsv` lists `field<TAB>term-hex<TAB>docFreq<TAB>totalTermFreq` then
//! one `doc:freq[:pos.pos:start-end.start-end:payload-hex|-]` per document;
//! `norms.tsv` lists `field` then each document's norm.
//!
//! `codec/` holds a SimpleText index without a segments file: `_0`
//! (`.si` with diagnostics, attributes and a three-provider index sort;
//! `.inf`; `.fld` with every stored type; `_0_1.liv`; `.dat` with every doc
//! values type, sorted as the index sort says, skip indexes on three; `.len`
//! norms and `.pst` postings for two indexed fields; `.tvc` vectors of both,
//! with positions, offsets and payloads matching the postings) and the
//! compound `_1` (`.si` + `.scf` holding its `.inf`/`.fld`/`.dat`, the
//! last its parent field), plus `infos/_2.inf` with every kind of field.
//! Java reads each through SimpleTextCodec's own formats against the `*.tsv`
//! manifests here, re-writes them with its own writers, commits a
//! `segments_N` over `_0`/`_1`, opens it with `DirectoryReader` and runs
//! `CheckIndex` over it at its slowest level.
//!
//! Run: `cargo run -p lucene-codecs --example write_simpletext_fixture -- <dir>`
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use lucene_codecs::doc_values::{
    BinaryColumn, SortedNumericColumn, SortedOrdsColumn, SortedSetOrdsColumn,
};
use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::postings_writer::{FieldNorms, FieldPostingsInput, TermPostings};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_codecs::simpletext::compound::{self as st_compound, SimpleTextCompoundReader};
use lucene_codecs::simpletext::doc_values_reader::SimpleTextDocValuesReader;
use lucene_codecs::simpletext::doc_values_writer::{
    doc_values_file_name, SimpleTextDocValuesWriter, DOC_VALUES_EXTENSION,
};
use lucene_codecs::simpletext::field_infos as st_infos;
use lucene_codecs::simpletext::fields_reader::SimpleTextFieldsReader;
use lucene_codecs::simpletext::fields_writer::{postings_file_name, write as write_pst};
use lucene_codecs::simpletext::norms::{norms_file_name, SimpleTextNormsConsumer};
use lucene_codecs::simpletext::segment_info::{
    self as st_si, SimpleTextSegmentInfo, SortFieldEntry,
};
use lucene_codecs::simpletext::stored_fields_reader::SimpleTextStoredFieldsReader;
use lucene_codecs::simpletext::term_vectors_reader::SimpleTextTermVectorsReader;
use lucene_codecs::simpletext::term_vectors_writer::{self as st_tv, term_vectors_file_name};
use lucene_codecs::simpletext::{live_docs, stored_fields_writer, ALL, NO_MORE_DOCS};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_codecs::term_vectors::{TermVectorField, TermVectorTerm, TermVectorsDocument};
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::geo::{java_double_string, java_float_string};
use lucene_util::version::Version;

const SEGMENT_ID: [u8; 16] = *b"rustsimpletextid";
const MAX_DOC: i32 = 3_000;
const SUFFIX: &str = "SimpleText_0";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

struct Rng(u64);

impl Rng {
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

/// One occurrence: position, offsets, payload.
type Occ = (i32, i32, i32, Vec<u8>);
/// term -> doc -> occurrences (one placeholder per occurrence for `tag`,
/// none for docs-only fields).
type Postings = BTreeMap<Vec<u8>, BTreeMap<i32, Vec<Occ>>>;

struct Corpus {
    id: Postings,
    bin: Postings,
    tag: Postings,
    body: Postings,
    tag_norms: Vec<i64>,
    body_norms: Vec<i64>,
}

fn corpus() -> Corpus {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut c = Corpus {
        id: Postings::new(),
        bin: Postings::new(),
        tag: Postings::new(),
        body: Postings::new(),
        tag_norms: Vec::new(),
        body_norms: Vec::new(),
    };
    let alphabet = [b'a', b'\n', b'\\', 0xff, b'z'];
    for doc in 0..MAX_DOC {
        let key = format!("id{:x}", doc * 7919).into_bytes();
        c.id.entry(key).or_default().insert(doc, vec![]);
        if doc % 211 == 5 {
            c.id.entry(Vec::new()).or_default().insert(doc, vec![]);
        }
        let len = 1 + rng.below(3) as usize;
        let bin: Vec<u8> = (0..len)
            .map(|_| alphabet[rng.below(alphabet.len() as u64) as usize])
            .collect();
        c.bin.entry(bin).or_default().insert(doc, vec![]);
        let mut tag_len = 0;
        for _ in 0..1 + rng.below(4) {
            let bound = 1 + rng.below(60);
            let t = format!("t{}", rng.below(bound)).into_bytes();
            c.tag
                .entry(t)
                .or_default()
                .entry(doc)
                .or_default()
                .push((0, 0, 0, vec![]));
            tag_len += 1;
        }
        c.tag_norms.push(tag_len);
        let mut offset = 0;
        let words = 2 + rng.below(10) as i32;
        for pos in 0..words {
            let bound = 1 + rng.below(400);
            let t = format!("w{}", rng.below(bound)).into_bytes();
            let payload = match rng.below(4) {
                0 => vec![],
                1 => vec![b'\n', b'\\'],
                _ => (0..1 + rng.below(3))
                    .map(|_| rng.below(256) as u8)
                    .collect(),
            };
            let end = offset + t.len() as i32;
            c.body
                .entry(t)
                .or_default()
                .entry(doc)
                .or_default()
                .push((pos, offset, end, payload));
            offset = end + 1;
        }
        // A norm that varies with the length, and sometimes not, so that the
        // competitive impacts keep more than one pair.
        c.body_norms
            .push(i64::from(words) + if rng.below(5) == 0 { 40 } else { 0 });
    }
    c
}

fn term_postings(p: &Postings, io: IndexOptions) -> Vec<TermPostings> {
    p.iter()
        .map(|(term, docs)| {
            let mut tp = TermPostings {
                term: term.clone(),
                docs: docs
                    .iter()
                    .map(|(&d, occ)| (d, occ.len().max(1) as i32))
                    .collect(),
                ..TermPostings::default()
            };
            if io.subsumes_positions() {
                tp.positions = docs
                    .values()
                    .map(|o| o.iter().map(|x| x.0).collect())
                    .collect();
                tp.offsets = docs
                    .values()
                    .map(|o| o.iter().map(|x| (x.1, x.2)).collect())
                    .collect();
                for o in docs.values().flatten() {
                    tp.payload_lengths.push(o.3.len() as u32);
                    tp.payload_bytes.extend_from_slice(&o.3);
                }
            }
            tp
        })
        .collect()
}

fn field_infos() -> FieldInfos {
    FieldInfos::new(vec![
        FieldInfo::new("id", 0)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
        FieldInfo::new("bin", 1)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
        FieldInfo::new("tag", 2).with_index_options(IndexOptions::DocsAndFreqs),
        FieldInfo::new("body", 3)
            .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
            .with_store_payloads(true),
    ])
    .unwrap()
}

/// `VerifyCodecsSimpleText.termLine`'s line for one term.
fn term_line(name: &str, term: &[u8], docs: &BTreeMap<i32, Vec<Occ>>, io: IndexOptions) -> String {
    let df = docs.len();
    let ttf: usize = if io == IndexOptions::Docs {
        df
    } else {
        docs.values().map(|o| o.len().max(1)).sum()
    };
    let mut s = format!("{name}\t{}\t{df}\t{ttf}", hex(term));
    for (doc, occ) in docs {
        write!(s, "\t{doc}:{}", occ.len().max(1)).unwrap();
        if io.subsumes_positions() {
            let pos: Vec<String> = occ.iter().map(|o| o.0.to_string()).collect();
            let offs: Vec<String> = occ.iter().map(|o| format!("{}-{}", o.1, o.2)).collect();
            let pays: Vec<String> = occ
                .iter()
                .map(|o| {
                    if o.3.is_empty() {
                        "-".into()
                    } else {
                        hex(&o.3)
                    }
                })
                .collect();
            write!(
                s,
                ":{}:{}:{}",
                pos.join("."),
                offs.join("."),
                pays.join(".")
            )
            .unwrap();
        }
    }
    s
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_simpletext_fixture <output-dir>");
    let out = Path::new(&out_dir);
    std::fs::create_dir_all(out).unwrap();
    let c = corpus();
    // Name order, as a flush hands them to the writer.
    let fields: [(&str, i32, IndexOptions, bool, &Postings); 4] = [
        ("bin", 1, IndexOptions::Docs, false, &c.bin),
        (
            "body",
            3,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            true,
            &c.body,
        ),
        ("id", 0, IndexOptions::Docs, false, &c.id),
        ("tag", 2, IndexOptions::DocsAndFreqs, false, &c.tag),
    ];
    let fis = field_infos();
    let built: Vec<Vec<TermPostings>> = fields
        .iter()
        .map(|(_, _, io, _, p)| term_postings(p, *io))
        .collect();
    let inputs: Vec<FieldPostingsInput<'_>> = fields
        .iter()
        .zip(&built)
        .map(|((_, number, io, payloads, p), terms)| {
            let docs: BTreeSet<i32> = p.values().flat_map(|d| d.keys().copied()).collect();
            FieldPostingsInput {
                field_number: *number,
                index_options: *io,
                doc_count: docs.len() as i32,
                has_payloads: *payloads,
                terms,
            }
        })
        .collect();
    let norms = [
        FieldNorms {
            field_number: 2,
            values: &c.tag_norms,
        },
        FieldNorms {
            field_number: 3,
            values: &c.body_norms,
        },
    ];
    let write = SegmentWriteState {
        max_doc: MAX_DOC,
        segment_id: &SEGMENT_ID,
        segment_suffix: SUFFIX,
        field_infos: &fis,
    };
    let pst = write_pst(&inputs, &norms, &write).unwrap();

    // This port reads its own bytes back before Java is asked to.
    let read = SegmentReadState {
        max_doc: MAX_DOC,
        segment_id: &SEGMENT_ID,
        segment_suffix: SUFFIX,
        field_infos: &fis,
    };
    let reader = SimpleTextFieldsReader::open(pst.clone(), &read).unwrap();
    let mut terms_tsv = String::new();
    for (name, _, io, _, p) in &fields {
        let terms = reader.terms(name).unwrap().unwrap();
        let mut e = terms.iterator();
        for (term, docs) in p.iter() {
            assert_eq!(e.next().unwrap(), Some(term.as_slice()), "{name}");
            let mut pe = e.postings(ALL).unwrap();
            for doc in docs.keys() {
                assert_eq!(pe.next_doc().unwrap(), *doc);
            }
            assert_eq!(pe.next_doc().unwrap(), NO_MORE_DOCS);
            terms_tsv.push_str(&term_line(name, term, docs, *io));
            terms_tsv.push('\n');
        }
        assert_eq!(e.next().unwrap(), None);
    }
    let mut norms_tsv = String::new();
    for (name, values) in [("tag", &c.tag_norms), ("body", &c.body_norms)] {
        norms_tsv.push_str(name);
        for v in values.iter() {
            write!(norms_tsv, "\t{v}").unwrap();
        }
        norms_tsv.push('\n');
    }
    std::fs::write(out.join(postings_file_name("_0", SUFFIX)), pst).unwrap();
    std::fs::write(out.join("terms.tsv"), &terms_tsv).unwrap();
    std::fs::write(out.join("norms.tsv"), &norms_tsv).unwrap();
    std::fs::write(
        out.join("manifest.properties"),
        format!("id_hex={}\nmax_doc={MAX_DOC}\n", hex(&SEGMENT_ID)),
    )
    .unwrap();
    write_codec(&out.join("codec"));
    println!(
        "wrote a SimpleText postings file and a SimpleText index to {}",
        out.display()
    );
}

// ---- codec/: the per-segment formats ----------------------------------------------------

fn hexs(s: &str) -> String {
    hex(s.as_bytes())
}

/// The stored-only fields of `_0` and `_1` (`IndexOptions.NONE`, no doc
/// values: a segment of just `.si`/`.inf`/`.fld` opens), one with an
/// attribute holding a newline and backslashes.
fn stored_infos() -> FieldInfos {
    let names = ["id", "s", "b", "i", "l", "f", "d"];
    let fields = names
        .iter()
        .enumerate()
        .map(|(n, name)| {
            let fi = FieldInfo::new(*name, n as i32);
            if *name == "s" {
                fi.with_attributes(vec![("note\\key".into(), "line\nbreak \\".into())])
            } else {
                fi
            }
        })
        .collect();
    FieldInfos::new(fields).unwrap()
}

/// Every kind of field, at most one attribute each (Java's re-write
/// iterates a `HashMap`).
fn every_kind_infos() -> FieldInfos {
    use DocValuesSkipIndexType as Skip;
    use DocValuesType as Dv;
    let fields = vec![
        FieldInfo::new("body", 0)
            .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
            .with_store_term_vectors(true)
            .with_store_payloads(true)
            .with_attributes(vec![(
                "PerFieldPostingsFormat.format".into(),
                "SimpleText".into(),
            )]),
        FieldInfo::new("tag", 1)
            .with_index_options(IndexOptions::DocsAndFreqs)
            .with_omit_norms(true),
        FieldInfo::new("pos", 2).with_index_options(IndexOptions::DocsAndFreqsAndPositions),
        FieldInfo::new("num", 3).with_doc_values(Dv::Numeric, Skip::Range, 4),
        FieldInfo::new("bin", 4).with_doc_values(Dv::Binary, Skip::None, -1),
        FieldInfo::new("sorted", 5)
            .with_index_options(IndexOptions::Docs)
            .with_doc_values(Dv::Sorted, Skip::Range, -1),
        FieldInfo::new("sset", 6).with_doc_values(Dv::SortedSet, Skip::None, 2),
        FieldInfo::new("snum", 7).with_doc_values(Dv::SortedNumeric, Skip::Range, -1),
        FieldInfo::new("pt", 8).with_points(3, 2, 4),
        FieldInfo::new("vec", 9).with_vectors(
            8,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Cosine,
        ),
        FieldInfo::new("bvec", 10).with_vectors(
            16,
            VectorEncoding::Byte,
            VectorSimilarityFunction::MaximumInnerProduct,
        ),
        FieldInfo::new("soft", 11)
            .with_doc_values(Dv::Numeric, Skip::None, -1)
            .with_soft_deletes_field(true),
        FieldInfo::new("parent", 12)
            .with_doc_values(Dv::Numeric, Skip::None, -1)
            .with_parent_field(true),
        FieldInfo::new("caf\u{e9} \\ \u{1f600}", 20)
            .with_index_options(IndexOptions::Docs)
            .with_attributes(vec![("k".into(), String::new())]),
    ];
    FieldInfos::new(fields).unwrap()
}

/// `n` documents holding every stored type: strings with newlines,
/// backslashes and supplementary characters, binary with 0x0a/0x5c/0x00/0xff,
/// int/long extremes, the special floats and doubles and random bits.
fn stored_docs(n: usize, seed: u64) -> Vec<Document> {
    let mut rng = Rng(seed);
    let strings = [
        "",
        "plain",
        "line\nbreak",
        "back\\slash\\",
        "\\\n\\n",
        "caf\u{e9} \u{4e2d}\u{6587}",
        "\u{1f600} emoji",
        "end\n",
    ];
    let floats = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        f32::from_bits(1),
        f32::MAX,
        f32::MIN_POSITIVE,
        1e7,
        9_999_999.0,
        0.001,
        1e-4,
        -1.5,
    ];
    let doubles = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        f64::from_bits(1),
        f64::MAX,
        f64::MIN_POSITIVE,
        1e7,
        1e-3,
        1e23,
        0.1,
        4.35,
    ];
    (0..n)
        .map(|i| {
            let mut fields = Vec::new();
            let mut add = |field_number: i32, value: FieldValue| {
                fields.push(StoredField {
                    field_number,
                    value,
                })
            };
            add(0, FieldValue::String(format!("id{i}")));
            add(
                1,
                FieldValue::String(strings[i % strings.len()].to_string()),
            );
            if i % 3 == 0 {
                let len = rng.below(7) as usize;
                let b = (0..len)
                    .map(|_| match rng.below(5) {
                        0 => 0x0a,
                        1 => 0x5c,
                        2 => 0x00,
                        3 => 0xff,
                        _ => rng.next() as u8,
                    })
                    .collect();
                add(2, FieldValue::Binary(b));
            }
            let int = [i32::MIN, i32::MAX, 0, -1, rng.next() as i32][i % 5];
            add(3, FieldValue::Int(int));
            let long = [i64::MIN, i64::MAX, 0, -1, rng.next() as i64][i % 5];
            add(4, FieldValue::Long(long));
            let mut float = || loop {
                let f = f32::from_bits(rng.next() as u32);
                if !f.is_nan() {
                    break f;
                }
            };
            let f = floats.get(i).copied().unwrap_or_else(&mut float);
            add(5, FieldValue::Float(f));
            add(5, FieldValue::Float(float()));
            let mut double = || loop {
                let d = f64::from_bits(rng.next());
                if !d.is_nan() {
                    break d;
                }
            };
            let d = doubles.get(i).copied().unwrap_or_else(&mut double);
            add(6, FieldValue::Double(d));
            add(6, FieldValue::Double(double()));
            if i % 4 == 1 {
                add(1, FieldValue::String(format!("second {i}")));
            }
            Document { fields }
        })
        .collect()
}

/// `GenCodecsSimpleTextCodec`'s `stored.SEG.tsv` lines.
fn stored_lines(docs: &[Document]) -> String {
    let mut s = String::new();
    for (n, doc) in docs.iter().enumerate() {
        writeln!(s, "doc\t{n}").unwrap();
        for f in &doc.fields {
            let (t, v) = match &f.value {
                FieldValue::String(v) => ("string", hexs(v)),
                FieldValue::Binary(v) => ("binary", hex(v)),
                FieldValue::Int(v) => ("int", v.to_string()),
                FieldValue::Long(v) => ("long", v.to_string()),
                FieldValue::Float(v) => (
                    "float",
                    format!("{:08x}\t{}", v.to_bits(), java_float_string(*v)),
                ),
                FieldValue::Double(v) => (
                    "double",
                    format!("{:016x}\t{}", v.to_bits(), java_double_string(*v)),
                ),
            };
            writeln!(s, "{}\t{t}\t{v}", f.field_number).unwrap();
        }
    }
    s
}

/// `fields.FILE.tsv` lines.
fn field_lines(fis: &FieldInfos) -> String {
    let mut s = String::new();
    for fi in &fis.fields {
        let mut atts = fi.attributes.clone();
        atts.sort();
        let atts: Vec<String> = atts
            .iter()
            .map(|(k, v)| format!("{}={}", hexs(k), hexs(v)))
            .collect();
        let row = [
            hexs(&fi.name),
            fi.number.to_string(),
            st_infos::index_options_name(fi.index_options).to_string(),
            fi.store_term_vectors.to_string(),
            fi.store_payloads.to_string(),
            fi.omit_norms.to_string(),
            st_infos::doc_values_type_name(fi.doc_values_type).to_string(),
            st_infos::skip_index_type_name(fi.doc_values_skip_index_type).to_string(),
            fi.doc_values_gen.to_string(),
            atts.join(","),
            fi.point_dimension_count.to_string(),
            fi.point_index_dimension_count.to_string(),
            fi.point_num_bytes.to_string(),
            fi.vector_dimension.to_string(),
            st_infos::vector_encoding_name(fi.vector_encoding).to_string(),
            st_infos::similarity_name(fi.vector_similarity_function).to_string(),
            fi.soft_deletes_field.to_string(),
            fi.parent_field.to_string(),
        ];
        writeln!(s, "{}", row.join("\t")).unwrap();
    }
    s
}

/// `si.SEG.tsv` lines.
fn si_lines(si: &SimpleTextSegmentInfo) -> String {
    let mut s = String::new();
    writeln!(s, "version\t{}", si.version).unwrap();
    let min = si.min_version.map_or("null".to_string(), |v| v.to_string());
    writeln!(s, "minVersion\t{min}").unwrap();
    writeln!(s, "docCount\t{}", si.doc_count).unwrap();
    writeln!(s, "compound\t{}", si.is_compound_file).unwrap();
    writeln!(s, "hasBlocks\t{}", si.has_blocks).unwrap();
    for (tag, map) in [("diag", &si.diagnostics), ("att", &si.attributes)] {
        let mut map = map.clone();
        map.sort();
        for (k, v) in map {
            writeln!(s, "{tag}\t{}\t{}", hexs(&k), hexs(&v)).unwrap();
        }
    }
    let mut files = si.files.clone();
    files.sort();
    for f in files {
        writeln!(s, "file\t{f}").unwrap();
    }
    writeln!(s, "id\t{}", hex(&si.id)).unwrap();
    for e in &si.index_sort {
        writeln!(
            s,
            "sort\t{}\t{}\t{}",
            e.provider,
            hexs(&e.description),
            hex(&e.bytes)
        )
        .unwrap();
    }
    s
}

/// The three providers' entries as Java's `SortFieldProvider.write` and
/// `SortField.toString` produce them (taken from `GenCodecsSimpleTextCodec`'s
/// flushed segment: this port keeps them undecoded).
fn index_sort() -> Vec<SortFieldEntry> {
    let entry = |provider: &str, description: &str, bytes: &[u8]| SortFieldEntry {
        provider: provider.into(),
        description: description.into(),
        bytes: bytes.to_vec(),
    };
    vec![
        entry(
            "SortField",
            "<long: \"num\">! missingValue=-9223372036854775808",
            &[
                3, 0x6e, 0x75, 0x6d, 4, 0x4c, 0x4f, 0x4e, 0x47, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0x80,
            ],
        ),
        entry(
            "SortedNumericSortField",
            "<sortednumeric: \"snum\"> selector=MAX type=INT",
            &[
                4, 0x73, 0x6e, 0x75, 0x6d, 3, 0x49, 0x4e, 0x54, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            ],
        ),
        entry(
            "SortedSetSortField",
            "<sortedset: \"sset\">! selector=MIN",
            &[
                4, 0x73, 0x73, 0x65, 0x74, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        ),
    ]
}

/// A segment's fields: the stored ones, then `_0`'s doc values and indexed
/// fields (the index sort's `num`/`snum`/`sset` among them) or `_1`'s
/// parent field.
fn segment_infos(compound: bool) -> FieldInfos {
    use DocValuesSkipIndexType as Skip;
    use DocValuesType as Dv;
    let mut fields = stored_infos().fields;
    if compound {
        fields.push(
            FieldInfo::new("parent", 7)
                .with_doc_values(Dv::Numeric, Skip::None, -1)
                .with_parent_field(true),
        );
    } else {
        fields.extend([
            FieldInfo::new("num", 7).with_doc_values(Dv::Numeric, Skip::None, -1),
            FieldInfo::new("snum", 8).with_doc_values(Dv::SortedNumeric, Skip::Range, -1),
            FieldInfo::new("sset", 9).with_doc_values(Dv::SortedSet, Skip::None, -1),
            FieldInfo::new("bin", 10).with_doc_values(Dv::Binary, Skip::None, -1),
            FieldInfo::new("sorted", 11).with_doc_values(Dv::Sorted, Skip::Range, -1),
            FieldInfo::new("dvneg", 12).with_doc_values(Dv::Numeric, Skip::Range, -1),
            FieldInfo::new("body", 13)
                .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
                .with_store_payloads(true)
                .with_store_term_vectors(true),
            FieldInfo::new("tag", 14)
                .with_index_options(IndexOptions::DocsAndFreqs)
                .with_store_term_vectors(true),
        ]);
    }
    FieldInfos::new(fields).unwrap()
}

/// Sorted-set and sorted terms: the empty term, `\n`, `\`, `0xff`.
fn dv_term(n: u64) -> Vec<u8> {
    match n {
        0 => Vec::new(),
        1 => b"a\nb".to_vec(),
        2 => b"c\\".to_vec(),
        3 => vec![0xff, b'\n'],
        _ => format!("t{n:02}").into_bytes(),
    }
}

/// `GenCodecsSimpleTextCodec`'s `dv` lines for one field's values (the
/// subset `VerifyCodecsSimpleText` dumps: values, dictionary, skipper).
enum DvValues {
    Numeric(Vec<(i32, i64)>),
    Binary(BinaryColumn),
    Sorted(SortedOrdsColumn),
    SortedNumeric(SortedNumericColumn),
    SortedSet(SortedSetOrdsColumn),
}

/// `_0`'s doc values (or `_1`'s parent field): `num` strictly descending
/// over docs 0..37 and missing after, `snum`'s maximum then ascending, so
/// the documents are in index-sort order.
fn dv_columns(name: &str, max_doc: i32) -> Vec<(&'static str, DvValues)> {
    let mut rng = Rng(0xD0C5 ^ max_doc as u64);
    if name == "_1" {
        let parents = (0..max_doc).filter(|d| d % 3 == 2 || *d == max_doc - 1);
        return vec![(
            "parent",
            DvValues::Numeric(parents.map(|d| (d, -1)).collect()),
        )];
    }
    let num = (0..37).map(|d| (d, 100 - 3 * i64::from(d))).collect();
    let mut snum = SortedNumericColumn::default();
    for d in 0..max_doc {
        let mut vs: Vec<i64> = match d {
            37 => vec![1, 5],
            38 => vec![7],
            39 => vec![-2, 9],
            _ => (0..d % 3).map(|_| rng.below(101) as i64 - 50).collect(),
        };
        if vs.is_empty() {
            continue;
        }
        vs.sort();
        snum.docs.push(d);
        snum.counts.push(vs.len() as u32);
        snum.values.extend(vs);
    }
    let mut dict: Vec<Vec<u8>> = (0..24).map(dv_term).collect();
    dict.sort();
    let mut sset = SortedSetOrdsColumn {
        dict: dict.clone(),
        ..SortedSetOrdsColumn::default()
    };
    for d in (0..max_doc).filter(|d| d % 5 != 2) {
        let ords: BTreeSet<i64> = (0..rng.below(6)).map(|_| rng.below(24) as i64).collect();
        if ords.is_empty() {
            continue;
        }
        sset.docs.push(d);
        sset.counts.push(ords.len() as u32);
        sset.ords.extend(ords);
    }
    let blobs: [&[u8]; 6] = [
        b"",
        b"\n",
        b"\\",
        &[0xff],
        b"x\ny\\z",
        &[0, 0x0a, 0x5c, 0xff],
    ];
    let mut bin = BinaryColumn::default();
    for d in (0..max_doc).filter(|d| d % 4 != 1) {
        let b = blobs[d as usize % blobs.len()];
        bin.docs.push(d);
        bin.lengths.push(b.len() as u32);
        bin.bytes.extend_from_slice(b);
    }
    // `sorted`'s dictionary is the terms some document uses.
    let picks: Vec<(i32, Vec<u8>)> = (0..max_doc)
        .filter(|d| d % 6 != 4)
        .map(|d| (d, dv_term(rng.below(9))))
        .collect();
    let sdict: Vec<Vec<u8>> = picks
        .iter()
        .map(|p| p.1.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let sorted = SortedOrdsColumn {
        docs: picks.iter().map(|p| p.0).collect(),
        ords: picks
            .iter()
            .map(|p| sdict.binary_search(&p.1).unwrap() as i64)
            .collect(),
        dict: sdict,
    };
    let extremes = [i64::MIN, i64::MAX, 0, -1, 1 << 53];
    let dvneg = (0..max_doc)
        .filter(|d| d % 3 != 0)
        .map(|d| (d, extremes[d as usize % extremes.len()]))
        .collect();
    vec![
        ("num", DvValues::Numeric(num)),
        ("snum", DvValues::SortedNumeric(snum)),
        ("sset", DvValues::SortedSet(sset)),
        ("bin", DvValues::Binary(bin)),
        ("sorted", DvValues::Sorted(sorted)),
        ("dvneg", DvValues::Numeric(dvneg)),
    ]
}

fn csv<T: ToString>(v: &[T]) -> String {
    v.iter().map(T::to_string).collect::<Vec<_>>().join(",")
}

/// Writes the segment's `.dat` (fields in number order, as Java's re-write
/// adds them) and `dv.NAME.dat.tsv`; returns the file after reading it back.
fn write_doc_values(root: &Path, name: &str, fis: &FieldInfos, max_doc: i32) -> Vec<u8> {
    let columns = dv_columns(name, max_doc);
    let mut w = SimpleTextDocValuesWriter::new(max_doc);
    let mut blocks = Vec::new();
    for (field, values) in &columns {
        let fi = fis.field_by_name(field).unwrap();
        let ty = st_infos::doc_values_type_name(fi.doc_values_type);
        let mut tsv = String::new();
        writeln!(tsv, "field\t{}\t{ty}", hexs(field)).unwrap();
        match values {
            DvValues::Numeric(v) => {
                w.add_numeric_field(fi, v).unwrap();
                for (d, x) in v {
                    writeln!(tsv, "d\t{d}\t{x}").unwrap();
                }
            }
            DvValues::Binary(c) => {
                w.add_binary_field(fi, c).unwrap();
                let mut at = 0;
                for (d, len) in c.docs.iter().zip(&c.lengths) {
                    let end = at + *len as usize;
                    writeln!(tsv, "d\t{d}\t{}", hex(&c.bytes[at..end])).unwrap();
                    at = end;
                }
            }
            DvValues::Sorted(c) => {
                w.add_sorted_field(fi, c).unwrap();
                for (d, o) in c.docs.iter().zip(&c.ords) {
                    writeln!(tsv, "d\t{d}\t{o}\t{}", hex(&c.dict[*o as usize])).unwrap();
                }
                for (o, t) in c.dict.iter().enumerate() {
                    writeln!(tsv, "ord\t{o}\t{}", hex(t)).unwrap();
                }
            }
            DvValues::SortedNumeric(c) => {
                w.add_sorted_numeric_field(fi, c).unwrap();
                let mut at = 0;
                for (d, n) in c.docs.iter().zip(&c.counts) {
                    let end = at + *n as usize;
                    writeln!(tsv, "d\t{d}\t{}", csv(&c.values[at..end])).unwrap();
                    at = end;
                }
            }
            DvValues::SortedSet(c) => {
                w.add_sorted_set_field(fi, c).unwrap();
                let mut at = 0;
                for (d, n) in c.docs.iter().zip(&c.counts) {
                    let end = at + *n as usize;
                    writeln!(tsv, "d\t{d}\t{}", csv(&c.ords[at..end])).unwrap();
                    at = end;
                }
                for (o, t) in c.dict.iter().enumerate() {
                    writeln!(tsv, "ord\t{o}\t{}", hex(t)).unwrap();
                }
            }
        }
        blocks.push(tsv);
    }
    let dat = w.finish().unwrap();
    // Read back: every value as written, then the skipper as this port
    // reads it (checked against Java's own reader by the fixture tests).
    let r = SimpleTextDocValuesReader::open(&dat, max_doc).unwrap();
    r.check_integrity().unwrap();
    assert_eq!(r.num_fields(), columns.len());
    let mut tsv = String::new();
    for ((field, values), block) in columns.iter().zip(&blocks) {
        tsv.push_str(block);
        let fi = fis.field_by_name(field).unwrap();
        if fi.doc_values_skip_index_type != DocValuesSkipIndexType::None {
            let mut s = r.skipper(field, fi.doc_values_type).unwrap();
            writeln!(
                tsv,
                "skipall\t{}\t{}\t{}\t{}",
                s.min_value(),
                s.max_value(),
                s.doc_count(),
                s.max_value_count()
            )
            .unwrap();
            let mut t = 0;
            loop {
                s.advance(t);
                if s.min_doc_id(0) == NO_MORE_DOCS {
                    break;
                }
                writeln!(
                    tsv,
                    "skip\t{}\t{}\t{}\t{}\t{}",
                    s.min_doc_id(0),
                    s.max_doc_id(0),
                    s.min_value_at(0),
                    s.max_value_at(0),
                    s.doc_count_at(0)
                )
                .unwrap();
                t = s.max_doc_id(0) + 1;
            }
        }
        if let DvValues::Numeric(v) = values {
            let mut it = r.numeric(field).unwrap();
            for (d, x) in v {
                assert_eq!(it.next_doc().unwrap(), *d);
                assert_eq!(it.long_value().unwrap(), *x);
            }
            assert_eq!(it.next_doc().unwrap(), NO_MORE_DOCS);
        }
    }
    std::fs::write(root.join(format!("dv.{name}.dat.tsv")), tsv).unwrap();
    dat
}

/// `_0`'s indexed fields from one corpus, so postings, norms and vectors
/// agree as CheckIndex requires: `body` (positions, offsets, payloads
/// holding `\n`, `\`, `0xff` or none) and `tag` (freqs only). Writes the
/// `.pst`, `.len`, `.tvc` and the `norms`/`tv` manifests; returns the file
/// names.
fn write_indexed(
    root: &Path,
    index: &Path,
    name: &str,
    fis: &FieldInfos,
    id: &[u8; 16],
    max_doc: i32,
) -> Vec<String> {
    let mut rng = Rng(0x7E57_0B0D ^ max_doc as u64);
    let words: [&[u8]; 8] = [
        b"a",
        b"b\nc",
        b"d\\",
        "\u{e9}".as_bytes(),
        "\u{1f600}".as_bytes(),
        "\u{ff5a}".as_bytes(),
        b"zz",
        b"",
    ];
    let mut body = Postings::new();
    let mut tag = Postings::new();
    let mut body_norms = Vec::new();
    let mut tag_norms = Vec::new();
    let mut tv_docs = Vec::new();
    for doc in 0..max_doc {
        let mut offset = 0;
        let n = 1 + rng.below(7) as i32;
        let mut doc_body: BTreeMap<Vec<u8>, Vec<Occ>> = BTreeMap::new();
        for pos in 0..n {
            let w = words[rng.below(words.len() as u64) as usize];
            let payload = match rng.below(3) {
                0 => Vec::new(),
                1 => vec![b'\n', b'\\', 0xff],
                _ => vec![rng.below(256) as u8],
            };
            let end = offset + w.len() as i32;
            doc_body
                .entry(w.to_vec())
                .or_default()
                .push((pos, offset, end, payload));
            offset = end + 1;
        }
        let mut doc_tag: BTreeMap<Vec<u8>, Vec<Occ>> = BTreeMap::new();
        let m = 1 + rng.below(3);
        for _ in 0..m {
            let t = format!("t{}", rng.below(4)).into_bytes();
            doc_tag.entry(t).or_default().push((0, 0, 0, Vec::new()));
        }
        body_norms.push(i64::from(n));
        tag_norms.push(m as i64);
        let body_field = TermVectorField {
            field_number: 13,
            has_positions: true,
            has_offsets: true,
            has_payloads: true,
            terms: doc_body
                .iter()
                .map(|(t, occ)| TermVectorTerm {
                    term: t.clone(),
                    freq: occ.len() as i32,
                    positions: Some(occ.iter().map(|o| o.0).collect()),
                    start_offsets: Some(occ.iter().map(|o| o.1).collect()),
                    end_offsets: Some(occ.iter().map(|o| o.2).collect()),
                    payloads: Some(occ.iter().map(|o| o.3.clone()).collect()),
                })
                .collect(),
        };
        let tag_field = TermVectorField {
            field_number: 14,
            has_positions: false,
            has_offsets: false,
            has_payloads: false,
            terms: doc_tag
                .iter()
                .map(|(t, occ)| TermVectorTerm {
                    term: t.clone(),
                    freq: occ.len() as i32,
                    positions: None,
                    start_offsets: None,
                    end_offsets: None,
                    payloads: None,
                })
                .collect(),
        };
        tv_docs.push(TermVectorsDocument {
            fields: vec![body_field, tag_field],
        });
        for (t, occ) in doc_body {
            body.entry(t).or_default().insert(doc, occ);
        }
        for (t, occ) in doc_tag {
            tag.entry(t).or_default().insert(doc, occ);
        }
    }

    // Postings, in field-name order.
    let fields: [(i32, IndexOptions, bool, &Postings); 2] = [
        (
            13,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            true,
            &body,
        ),
        (14, IndexOptions::DocsAndFreqs, false, &tag),
    ];
    let built: Vec<Vec<TermPostings>> = fields
        .iter()
        .map(|(_, io, _, p)| term_postings(p, *io))
        .collect();
    let inputs: Vec<FieldPostingsInput<'_>> = fields
        .iter()
        .zip(&built)
        .map(|((number, io, payloads, _), terms)| FieldPostingsInput {
            field_number: *number,
            index_options: *io,
            doc_count: max_doc,
            has_payloads: *payloads,
            terms,
        })
        .collect();
    let norms = [
        FieldNorms {
            field_number: 13,
            values: &body_norms,
        },
        FieldNorms {
            field_number: 14,
            values: &tag_norms,
        },
    ];
    let state = SegmentWriteState {
        max_doc,
        segment_id: id,
        segment_suffix: "",
        field_infos: fis,
    };
    let pst = write_pst(&inputs, &norms, &state).unwrap();
    let pst_name = postings_file_name(name, "");

    // Norms, in field-number order.
    let mut consumer = SimpleTextNormsConsumer::new(max_doc);
    let mut norms_tsv = String::new();
    for (field, values) in [("body", &body_norms), ("tag", &tag_norms)] {
        let dense: Vec<(i32, i64)> = values
            .iter()
            .enumerate()
            .map(|(d, v)| (d as i32, *v))
            .collect();
        consumer
            .add_norms_field(fis.field_by_name(field).unwrap(), &dense)
            .unwrap();
        writeln!(norms_tsv, "field\t{}", hexs(field)).unwrap();
        for (d, v) in &dense {
            writeln!(norms_tsv, "d\t{d}\t{v}").unwrap();
        }
    }
    let len = consumer.finish().unwrap();
    let len_name = norms_file_name(name, "");

    // Term vectors, read back before Java is asked to.
    let tvc = st_tv::write(&tv_docs, fis).unwrap();
    let tv_reader = SimpleTextTermVectorsReader::open(&tvc, max_doc).unwrap();
    let mut tv_tsv = String::new();
    for (d, want) in tv_docs.iter().enumerate() {
        writeln!(tv_tsv, "doc\t{d}").unwrap();
        let got = tv_reader.get(d as i32).unwrap().unwrap();
        assert_eq!(got.len(), want.fields.len());
        for ((field, g), w) in got.iter().zip(&want.fields) {
            assert_eq!(g, w);
            writeln!(
                tv_tsv,
                "field\t{}\t{}\t{}\t{}\t{}",
                hexs(field),
                w.has_positions,
                w.has_offsets,
                w.has_payloads,
                w.terms.len()
            )
            .unwrap();
            for t in &w.terms {
                writeln!(tv_tsv, "term\t{}\t{}", hex(&t.term), t.freq).unwrap();
                if !(w.has_positions || w.has_offsets) {
                    continue;
                }
                for k in 0..t.freq as usize {
                    let at = |v: &Option<Vec<i32>>| v.as_ref().map_or(-1, |v| v[k]);
                    let pay = match t.payloads.as_ref().map(|p| &p[k]) {
                        Some(p) if !p.is_empty() => hex(p),
                        _ => "-".to_string(),
                    };
                    writeln!(
                        tv_tsv,
                        "pos\t{}\t{}\t{}\t{pay}",
                        at(&t.positions),
                        at(&t.start_offsets),
                        at(&t.end_offsets)
                    )
                    .unwrap();
                }
            }
        }
    }
    let tvc_name = term_vectors_file_name(name);

    for (file, bytes) in [(&pst_name, &pst), (&len_name, &len), (&tvc_name, &tvc)] {
        std::fs::write(index.join(file), bytes).unwrap();
    }
    std::fs::write(root.join(format!("norms.{name}.tsv")), norms_tsv).unwrap();
    std::fs::write(root.join(format!("tv.{name}.tsv")), tv_tsv).unwrap();
    vec![pst_name, len_name, tvc_name]
}

/// `codec/`: segment `_0` (non-compound, sorted, deletes), `_1` (compound)
/// and a standalone `_2.inf`, each read back by this port first.
fn write_codec(root: &Path) {
    let index = root.join("index");
    let infos_dir = root.join("infos");
    std::fs::create_dir_all(&index).unwrap();
    std::fs::create_dir_all(&infos_dir).unwrap();
    let version = Version::parse("10.5.0").unwrap();
    let mut segments = String::new();
    for (name, max_doc, compound) in [("_0", 40usize, false), ("_1", 11, true)] {
        let id: [u8; 16] = *format!("rustsimpletext{name}")
            .as_bytes()
            .first_chunk()
            .unwrap();
        let fis = segment_infos(compound);
        let inf = st_infos::write(&fis.fields);
        assert_eq!(st_infos::read(&inf).unwrap().fields.len(), fis.fields.len());
        let docs = stored_docs(max_doc, 0x5eed ^ max_doc as u64);
        let fld = stored_fields_writer::write(&docs, &fis).unwrap();
        let reader = SimpleTextStoredFieldsReader::open(&fld, max_doc as i32).unwrap();
        assert_eq!(reader.max_doc(), max_doc);
        let fld_name = stored_fields_writer::stored_fields_file_name(name);
        let inf_name = st_infos::field_infos_file_name(name, "");
        let dat_name = doc_values_file_name(name, "", DOC_VALUES_EXTENSION);
        let dat = write_doc_values(root, name, &fis, max_doc as i32);
        let files = if compound {
            let sub = vec![(fld_name, fld), (inf_name, inf.clone()), (dat_name, dat)];
            let scf = st_compound::write(&sub);
            let cfs = SimpleTextCompoundReader::open(&scf, name).unwrap();
            let mut cfs_tsv = String::new();
            for n in cfs.list_all() {
                writeln!(cfs_tsv, "{n}\t{}", cfs.file_length(n).unwrap()).unwrap();
            }
            std::fs::write(root.join(format!("cfs.{name}.tsv")), cfs_tsv).unwrap();
            let scf_name = st_compound::compound_file_name(name);
            std::fs::write(index.join(&scf_name), scf).unwrap();
            vec![scf_name]
        } else {
            std::fs::write(index.join(&fld_name), fld).unwrap();
            std::fs::write(index.join(&inf_name), &inf).unwrap();
            std::fs::write(index.join(&dat_name), dat).unwrap();
            let mut files = vec![inf_name, fld_name, dat_name];
            files.extend(write_indexed(root, &index, name, &fis, &id, max_doc as i32));
            files
        };
        let mut si = SimpleTextSegmentInfo {
            name: name.into(),
            version,
            min_version: Some(version),
            doc_count: max_doc as i32,
            is_compound_file: compound,
            has_blocks: compound,
            diagnostics: vec![
                ("source".into(), "flush".into()),
                ("os".into(), "rust".into()),
                ("lucene.version".into(), "10.5.0".into()),
            ],
            attributes: vec![("rust.note".into(), format!("segment {name} \\ \u{e9}"))],
            files,
            id,
            index_sort: if compound { Vec::new() } else { index_sort() },
        };
        let si_bytes = st_si::write(&mut si).unwrap();
        assert_eq!(st_si::read(&si_bytes, name, &id).unwrap(), si);
        std::fs::write(index.join(st_si::segment_info_file_name(name)), si_bytes).unwrap();
        std::fs::write(root.join(format!("si.{name}.tsv")), si_lines(&si)).unwrap();
        std::fs::write(
            root.join(format!("fields.{name}.inf.tsv")),
            field_lines(&fis),
        )
        .unwrap();
        std::fs::write(root.join(format!("stored.{name}.tsv")), stored_lines(&docs)).unwrap();
        // Deletes on the non-compound segment only.
        let (del_count, del_gen) = if compound {
            (0, -1)
        } else {
            let mut live = FixedBitSet::new(max_doc);
            let mut deleted = 0;
            for d in 0..max_doc {
                if d % 7 == 3 || d == max_doc - 1 {
                    deleted += 1;
                } else {
                    live.set(d);
                }
            }
            let liv = live_docs::write(&live);
            let back = live_docs::read(&liv, max_doc as i32).unwrap();
            let bits: String = (0..max_doc)
                .map(|d| if back.get(d) { '1' } else { '0' })
                .collect();
            std::fs::write(
                root.join(format!("live.{name}.tsv")),
                format!("{max_doc}\n{bits}\n"),
            )
            .unwrap();
            std::fs::write(
                index.join(live_docs::live_docs_file_name(name, 1).unwrap()),
                liv,
            )
            .unwrap();
            (deleted, 1)
        };
        writeln!(
            segments,
            "{name}\t{max_doc}\t{del_count}\t{del_gen}\t{}",
            hex(&id)
        )
        .unwrap();
    }
    std::fs::write(root.join("segments.tsv"), segments).unwrap();

    let every = every_kind_infos();
    let inf = st_infos::write(&every.fields);
    assert_eq!(st_infos::write(&st_infos::read(&inf).unwrap().fields), inf);
    std::fs::write(infos_dir.join("_2.inf"), inf).unwrap();
    std::fs::write(root.join("fields._2.inf.tsv"), field_lines(&every)).unwrap();
}
