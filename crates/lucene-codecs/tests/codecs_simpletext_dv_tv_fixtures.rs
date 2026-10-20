//! lucene-codecs' SimpleText doc values (`.dat`), norms (`.len`) and term
//! vectors (`.tvc`) against Java: `GenCodecsSimpleTextCodec`'s two indexes
//! (flushed, and force-merged into a compound file) record what Java's
//! readers make of every such file -- each iterator's `nextDoc`, fresh
//! `advance` and running `advanceExact` probes, `lookupOrd`, the terms
//! enum, `lookupTerm` and `seekCeil` probes, the skipper's intervals; each
//! document's vectors with every occurrence and `seekCeil` probes.
//!
//! Here the Rust readers open Java's files and agree with every manifest,
//! the Rust writers rebuild each file byte for byte from it, and damaged
//! copies of Java's bytes are refused.

// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::fs;
use std::path::{Path, PathBuf};

use lucene_codecs::doc_values::{
    BinaryColumn, SortedNumericColumn, SortedOrdsColumn, SortedSetOrdsColumn,
};
use lucene_codecs::field_infos::{DocValuesType, FieldInfos};
use lucene_codecs::simpletext::compound::{self, SimpleTextCompoundReader};
use lucene_codecs::simpletext::doc_values_reader::SimpleTextDocValuesReader;
use lucene_codecs::simpletext::doc_values_writer::{
    doc_values_file_name, DocValuesInput, SimpleTextDocValuesWriter, DOC_VALUES_EXTENSION,
};
use lucene_codecs::simpletext::field_infos::{self as st_infos, doc_values_type};
use lucene_codecs::simpletext::norms::{norms_file_name, SimpleTextNormsProducer};
use lucene_codecs::simpletext::term_vectors_reader::{
    SimpleTextTermVectorsReader, SimpleTextTvTermsEnum,
};
use lucene_codecs::simpletext::term_vectors_writer::{self, term_vectors_file_name};
use lucene_codecs::simpletext::{util, SeekStatus, NO_MORE_DOCS, PAYLOADS};
use lucene_codecs::term_vectors::{TermVectorField, TermVectorTerm, TermVectorsDocument};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/codecs_simpletext/codec")
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

fn unhex_str(s: &str) -> String {
    String::from_utf8(unhex(s)).unwrap()
}

fn rows(path: &Path) -> Vec<Vec<String>> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path:?}: {e} (run GenCodecsSimpleTextCodec)"))
        .lines()
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect()
}

/// Bytes as Latin-1 text, so files holding raw values can be edited as
/// strings and turned back losslessly.
fn latin1(data: &[u8]) -> String {
    data.iter().map(|&b| char::from(b)).collect()
}

fn unlatin1(text: &str) -> Vec<u8> {
    text.chars()
        .map(|c| u8::try_from(u32::from(c)).unwrap())
        .collect()
}

/// `text` with its checksum line replaced by a valid one: semantic damage.
fn reseal(text: &str) -> Vec<u8> {
    let mut out = unlatin1(text.rsplit_once("checksum ").unwrap().0);
    util::write_checksum(&mut out);
    out
}

/// One segment of `segments.tsv`: name, maxDoc, fieldInfosGen.
struct Seg {
    name: String,
    max_doc: i32,
    field_infos_gen: i64,
}

/// A segment's files: core ones from the directory or its compound file.
struct Files {
    index: PathBuf,
    entries: Option<Vec<(String, Vec<u8>)>>,
}

impl Files {
    fn core(&self, file: &str) -> Vec<u8> {
        match &self.entries {
            Some(e) => e.iter().find(|e| e.0 == file).unwrap().1.clone(),
            None => fs::read(self.index.join(file)).unwrap(),
        }
    }

    fn dir(&self, file: &str) -> Vec<u8> {
        fs::read(self.index.join(file)).unwrap()
    }
}

fn segments(case: &str) -> Vec<(Seg, Files, FieldInfos, FieldInfos)> {
    let dir = root().join(case);
    let index = dir.join("index");
    rows(&dir.join("segments.tsv"))
        .into_iter()
        .map(|r| {
            let seg = Seg {
                name: r[0].clone(),
                max_doc: r[1].parse().unwrap(),
                field_infos_gen: r[5].parse().unwrap(),
            };
            let scf = index.join(compound::compound_file_name(&seg.name));
            let entries = scf.exists().then(|| {
                let scf = fs::read(scf).unwrap();
                let r = SimpleTextCompoundReader::open(&scf, &seg.name).unwrap();
                r.list_all()
                    .iter()
                    .map(|n| {
                        let b = r.open_input(&scf, n).unwrap().to_vec();
                        (n.clone(), b)
                    })
                    .collect()
            });
            let files = Files {
                index: index.clone(),
                entries,
            };
            let base = st_infos::read(&files.core(&st_infos::field_infos_file_name(&seg.name, "")))
                .unwrap();
            let latest = if seg.field_infos_gen == -1 {
                base.clone()
            } else {
                let name = lucene_codecs::simpletext::file_name_from_generation(
                    &seg.name,
                    st_infos::FIELD_INFOS_EXTENSION,
                    seg.field_infos_gen,
                )
                .unwrap();
                st_infos::read(&files.dir(&name)).unwrap()
            };
            (seg, files, base, latest)
        })
        .collect()
}

/// The `field <name>` lines of a SimpleText file, in file order.
fn file_fields(data: &[u8]) -> Vec<String> {
    latin1(data)
        .lines()
        .filter_map(|l| l.strip_prefix("field "))
        .map(|n| String::from_utf8(unlatin1(n)).unwrap())
        .collect()
}

/// One field's block of a `dv.FILE.tsv` / `norms.SEG.tsv`.
struct Block {
    name: String,
    ty: String,
    rows: Vec<Vec<String>>,
}

fn blocks(path: &Path) -> Vec<Block> {
    let mut out: Vec<Block> = Vec::new();
    for r in rows(path) {
        if r[0] == "field" {
            out.push(Block {
                name: unhex_str(&r[1]),
                ty: r.get(2).cloned().unwrap_or_else(|| "NUMERIC".into()),
                rows: Vec::new(),
            });
        } else {
            out.last_mut().unwrap().rows.push(r);
        }
    }
    out
}

impl Block {
    fn of<'a>(&'a self, tag: &'a str) -> impl Iterator<Item = &'a Vec<String>> + 'a {
        self.rows.iter().filter(move |r| r[0] == tag)
    }
}

fn csv<T: std::str::FromStr>(s: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(',').map(|v| v.parse().unwrap()).collect()
    }
}

fn status(s: SeekStatus) -> &'static str {
    match s {
        SeekStatus::Found => "FOUND",
        SeekStatus::NotFound => "NOT_FOUND",
        SeekStatus::End => "END",
    }
}

/// Java's `lookupTerm`: the ord, or `-insertionPoint - 1`.
fn java_lookup<T: std::ops::Neg<Output = T> + std::ops::Sub<Output = T> + From<i8>>(
    r: Result<T, T>,
) -> T {
    match r {
        Ok(o) => o,
        Err(ip) => -ip - T::from(1),
    }
}

/// Runs `$it` (a fresh iterator expression) through a block's `adv` and
/// `ax` probes.
macro_rules! probes {
    ($block:expr, $fresh:expr) => {{
        for r in $block.of("adv") {
            let mut it = $fresh;
            let got = it.advance(r[1].parse().unwrap()).unwrap();
            assert_eq!(got.to_string(), r[2], "{} advance({})", $block.name, r[1]);
        }
        let mut it = $fresh;
        for r in $block.of("ax") {
            let got = it.advance_exact(r[1].parse().unwrap()).unwrap();
            assert_eq!(
                got.to_string(),
                r[2],
                "{} advanceExact({})",
                $block.name,
                r[1]
            );
        }
    }};
}

/// Checks one `dv` block against the reader and returns the writer's input.
fn check_dv_block(r: &SimpleTextDocValuesReader<'_>, b: &Block) -> Input {
    let name = b.name.as_str();
    let d: Vec<&Vec<String>> = b.of("d").collect();
    let input = match b.ty.as_str() {
        "NUMERIC" => {
            let mut it = r.numeric(name).unwrap();
            let mut got = Vec::new();
            while it.next_doc().unwrap() != NO_MORE_DOCS {
                got.push((it.doc_id(), it.long_value().unwrap()));
            }
            let want: Vec<(i32, i64)> = d
                .iter()
                .map(|w| (w[1].parse().unwrap(), w[2].parse().unwrap()))
                .collect();
            assert_eq!(got, want, "{name}");
            probes!(b, r.numeric(name).unwrap());
            Input::Numeric(want)
        }
        "BINARY" => {
            let mut it = r.binary(name).unwrap();
            let mut got = Vec::new();
            while it.next_doc().unwrap() != NO_MORE_DOCS {
                got.push(vec![
                    it.doc_id().to_string(),
                    hex(&it.binary_value().unwrap()),
                ]);
            }
            let want: Vec<Vec<String>> = d.iter().map(|w| w[1..].to_vec()).collect();
            assert_eq!(got, want, "{name}");
            probes!(b, r.binary(name).unwrap());
            let mut col = BinaryColumn::default();
            for w in &want {
                let v = unhex(&w[1]);
                col.docs.push(w[0].parse().unwrap());
                col.lengths.push(v.len() as u32);
                col.bytes.extend(v);
            }
            Input::Binary(col)
        }
        "SORTED" => {
            let mut it = r.sorted(name).unwrap();
            let mut got = Vec::new();
            while it.next_doc().unwrap() != NO_MORE_DOCS {
                let ord = it.ord_value();
                let term = hex(it.lookup_ord(ord).unwrap());
                got.push(vec![it.doc_id().to_string(), ord.to_string(), term]);
            }
            let want: Vec<Vec<String>> = d.iter().map(|w| w[1..].to_vec()).collect();
            assert_eq!(got, want, "{name}");
            probes!(b, r.sorted(name).unwrap());
            let mut s = r.sorted(name).unwrap();
            let dict = check_dictionary(b, i64::from(s.value_count()), |o| {
                s.lookup_ord(o as i32).unwrap().to_vec()
            });
            let s = r.sorted(name).unwrap();
            check_terms(b, s.terms_enum(), |k| {
                i64::from(java_lookup(s.lookup_term(k).unwrap()))
            });
            Input::Sorted(SortedOrdsColumn {
                dict,
                docs: want.iter().map(|w| w[0].parse().unwrap()).collect(),
                ords: want.iter().map(|w| w[1].parse().unwrap()).collect(),
            })
        }
        "SORTED_NUMERIC" => {
            let mut it = r.sorted_numeric(name).unwrap();
            let mut got = Vec::new();
            while it.next_doc().unwrap() != NO_MORE_DOCS {
                let vs: Vec<String> = (0..it.doc_value_count())
                    .map(|_| it.next_value().unwrap().to_string())
                    .collect();
                got.push(vec![it.doc_id().to_string(), vs.join(",")]);
            }
            let want: Vec<Vec<String>> = d.iter().map(|w| w[1..].to_vec()).collect();
            assert_eq!(got, want, "{name}");
            probes!(b, r.sorted_numeric(name).unwrap());
            let mut col = SortedNumericColumn::default();
            for w in &want {
                let vs: Vec<i64> = csv(&w[1]);
                col.docs.push(w[0].parse().unwrap());
                col.counts.push(vs.len() as u32);
                col.values.extend(vs);
            }
            Input::SortedNumeric(col)
        }
        "SORTED_SET" => {
            let mut it = r.sorted_set(name).unwrap();
            let mut got = Vec::new();
            while it.next_doc().unwrap() != NO_MORE_DOCS {
                let vs: Vec<String> = (0..it.doc_value_count())
                    .map(|_| it.next_ord().unwrap().to_string())
                    .collect();
                got.push(vec![it.doc_id().to_string(), vs.join(",")]);
            }
            let want: Vec<Vec<String>> = d.iter().map(|w| w[1..].to_vec()).collect();
            assert_eq!(got, want, "{name}");
            probes!(b, r.sorted_set(name).unwrap());
            let mut s = r.sorted_set(name).unwrap();
            let dict = check_dictionary(b, s.value_count(), |o| s.lookup_ord(o).unwrap().to_vec());
            let s = r.sorted_set(name).unwrap();
            check_terms(b, s.terms_enum(), |k| {
                java_lookup(s.lookup_term(k).unwrap())
            });
            let mut col = SortedSetOrdsColumn {
                dict,
                ..Default::default()
            };
            for w in &want {
                let os: Vec<i64> = csv(&w[1]);
                col.docs.push(w[0].parse().unwrap());
                col.counts.push(os.len() as u32);
                col.ords.extend(os);
            }
            Input::SortedSet(col)
        }
        other => panic!("{other}"),
    };
    check_skipper(r, b);
    input
}

fn check_dictionary(b: &Block, count: i64, mut lookup: impl FnMut(i64) -> Vec<u8>) -> Vec<Vec<u8>> {
    let want: Vec<&Vec<String>> = b.of("ord").collect();
    assert_eq!(want.len() as i64, count, "{} valueCount", b.name);
    want.iter()
        .map(|w| {
            let o: i64 = w[1].parse().unwrap();
            let got = lookup(o);
            assert_eq!(hex(&got), w[2], "{} lookupOrd({o})", b.name);
            got
        })
        .collect()
}

fn check_terms(
    b: &Block,
    mut te: lucene_codecs::simpletext::doc_values_reader::SimpleTextDocValuesTermsEnum<'_>,
    lookup: impl Fn(&[u8]) -> i64,
) {
    let fresh = te.clone();
    let mut walked = Vec::new();
    while let Some(t) = te.next().unwrap() {
        let t = hex(t);
        walked.push(vec![te.ord().to_string(), t]);
    }
    let want: Vec<Vec<String>> = b.of("te").map(|w| w[1..].to_vec()).collect();
    assert_eq!(walked, want, "{} termsEnum", b.name);
    for w in b.of("lt") {
        assert_eq!(
            lookup(&unhex(&w[1])).to_string(),
            w[2],
            "{} lookupTerm({})",
            b.name,
            w[1]
        );
    }
    for w in b.of("ceil") {
        let mut c = fresh.clone();
        let st = c.seek_ceil(&unhex(&w[1])).unwrap();
        let ord = if st == SeekStatus::End { -1 } else { c.ord() };
        assert_eq!(
            [status(st).to_string(), ord.to_string()],
            [w[2].clone(), w[3].clone()],
            "{} seekCeil({})",
            b.name,
            w[1]
        );
        if st != SeekStatus::End {
            let term = c.term().to_vec();
            let mut e = fresh.clone();
            assert!(e.seek_exact(&term).unwrap());
            assert_eq!(e.ord(), ord);
            let mut e = fresh.clone();
            e.seek_exact_ord(ord).unwrap();
            assert_eq!(e.term(), term.as_slice());
        }
    }
}

fn check_skipper(r: &SimpleTextDocValuesReader<'_>, b: &Block) {
    let Some(all) = b.of("skipall").next() else {
        return;
    };
    let ty = doc_values_type(&b.ty).unwrap();
    let mut s = r.skipper(&b.name, ty).unwrap();
    assert_eq!(s.num_levels(), 1);
    assert_eq!(
        [
            s.min_value(),
            s.max_value(),
            i64::from(s.doc_count()),
            i64::from(s.max_value_count())
        ]
        .map(|v| v.to_string())
        .to_vec(),
        all[1..].to_vec(),
        "{} skipper",
        b.name
    );
    let mut got = Vec::new();
    let mut t = 0;
    loop {
        s.advance(t);
        if s.min_doc_id(0) == NO_MORE_DOCS {
            break;
        }
        got.push(
            [
                i64::from(s.min_doc_id(0)),
                i64::from(s.max_doc_id(0)),
                s.min_value_at(0),
                s.max_value_at(0),
                i64::from(s.doc_count_at(0)),
            ]
            .map(|v| v.to_string())
            .to_vec(),
        );
        t = s.max_doc_id(0) + 1;
    }
    let want: Vec<Vec<String>> = b.of("skip").map(|w| w[1..].to_vec()).collect();
    assert_eq!(got, want, "{} skip intervals", b.name);
}

/// A field's values as the writer takes them.
enum Input {
    Numeric(Vec<(i32, i64)>),
    Binary(BinaryColumn),
    Sorted(SortedOrdsColumn),
    SortedNumeric(SortedNumericColumn),
    SortedSet(SortedSetOrdsColumn),
}

impl Input {
    fn as_input(&self) -> DocValuesInput<'_> {
        match self {
            Input::Numeric(v) => DocValuesInput::Numeric(v),
            Input::Binary(c) => DocValuesInput::Binary(c),
            Input::Sorted(c) => DocValuesInput::Sorted(c),
            Input::SortedNumeric(c) => DocValuesInput::SortedNumeric(c),
            Input::SortedSet(c) => DocValuesInput::SortedSet(c),
        }
    }
}

/// Reads one `.dat` against its manifest and rebuilds it.
fn check_dat(data: &[u8], manifest: &Path, max_doc: i32, fis: &FieldInfos) -> usize {
    let r = SimpleTextDocValuesReader::open(data, max_doc).unwrap();
    r.check_integrity().unwrap();
    let order = file_fields(data);
    assert_eq!(r.num_fields(), order.len());
    let blocks = blocks(manifest);
    let mut w = SimpleTextDocValuesWriter::new(max_doc);
    for name in &order {
        let b = blocks
            .iter()
            .find(|b| &b.name == name)
            .unwrap_or_else(|| panic!("{name}"));
        let fi = fis.field_by_name(name).unwrap();
        assert_eq!(st_infos::doc_values_type_name(fi.doc_values_type), b.ty);
        let input = check_dv_block(&r, b);
        w.add_field(fi, input.as_input()).unwrap();
    }
    assert_eq!(w.finish().unwrap(), data, "{manifest:?} rebuilt");
    blocks.len()
}

fn check_case(case: &str) -> Vec<(String, Vec<u8>)> {
    let dir = root().join(case);
    let mut seen = Vec::new();
    for (seg, files, base, latest) in segments(case) {
        // The flush's .dat, then each doc-values generation's.
        let name = doc_values_file_name(&seg.name, "", DOC_VALUES_EXTENSION);
        let data = files.core(&name);
        check_dat(
            &data,
            &dir.join(format!("dv.{name}.tsv")),
            seg.max_doc,
            &base,
        );
        seen.push((name, data));
        let mut gens: Vec<i64> = latest
            .fields
            .iter()
            .map(|f| f.doc_values_gen)
            .filter(|&g| g != -1)
            .collect();
        gens.sort();
        gens.dedup();
        for g in gens {
            let name = lucene_codecs::simpletext::file_name_from_generation(
                &seg.name,
                DOC_VALUES_EXTENSION,
                g,
            )
            .unwrap();
            let data = files.dir(&name);
            check_dat(
                &data,
                &dir.join(format!("dv.{name}.tsv")),
                seg.max_doc,
                &latest,
            );
            seen.push((name, data));
        }
        // Norms.
        let name = norms_file_name(&seg.name, "");
        let data = files.core(&name);
        let p = SimpleTextNormsProducer::open(&data, seg.max_doc).unwrap();
        p.check_integrity().unwrap();
        let mut consumer =
            lucene_codecs::simpletext::norms::SimpleTextNormsConsumer::new(seg.max_doc);
        let blocks = blocks(&dir.join(format!("norms.{}.tsv", seg.name)));
        assert_eq!(file_fields(&data).len(), blocks.len());
        for field in file_fields(&data) {
            let b = blocks.iter().find(|b| b.name == field).unwrap();
            let mut it = p.norms(&field).unwrap();
            let mut got = Vec::new();
            while it.next_doc().unwrap() != NO_MORE_DOCS {
                got.push((it.doc_id(), it.long_value().unwrap()));
            }
            let want: Vec<(i32, i64)> = b
                .of("d")
                .map(|w| (w[1].parse().unwrap(), w[2].parse().unwrap()))
                .collect();
            assert_eq!(got, want, "{name} {field}");
            probes!(b, p.norms(&field).unwrap());
            consumer
                .add_norms_field(base.field_by_name(&field).unwrap(), &want)
                .unwrap();
        }
        assert_eq!(consumer.finish().unwrap(), data, "{name} rebuilt");
        seen.push((name, data));
        // Term vectors.
        let name = term_vectors_file_name(&seg.name);
        let data = files.core(&name);
        let docs = check_term_vectors(&data, &dir.join(format!("tv.{}.tsv", seg.name)), &base);
        assert_eq!(
            term_vectors_writer::write(&docs, &base).unwrap(),
            data,
            "{name} rebuilt"
        );
        seen.push((name, data));
    }
    seen
}

/// A field of `tv.SEG.tsv`: its name, its vectors and its seek probes.
type ManifestField = (String, TermVectorField, Vec<Vec<String>>);

/// Reads a `.tvc` against `tv.SEG.tsv`; returns the documents as the
/// writer takes them.
fn check_term_vectors(data: &[u8], manifest: &Path, fis: &FieldInfos) -> Vec<TermVectorsDocument> {
    // The manifest as documents, fields by name.
    let mut docs: Vec<Vec<ManifestField>> = Vec::new();
    let mut none = Vec::new();
    for r in rows(manifest) {
        match r[0].as_str() {
            "doc" => {
                docs.push(Vec::new());
                none.push(false);
            }
            "none" => *none.last_mut().unwrap() = true,
            "field" => {
                let name = unhex_str(&r[1]);
                let field = TermVectorField {
                    field_number: fis.field_by_name(&name).unwrap().number,
                    has_positions: r[2] == "true",
                    has_offsets: r[3] == "true",
                    has_payloads: r[4] == "true",
                    terms: Vec::new(),
                };
                docs.last_mut().unwrap().push((name, field, Vec::new()));
            }
            "term" => {
                let (_, f, _) = docs.last_mut().unwrap().last_mut().unwrap();
                let (pos, offs, pays) = (f.has_positions, f.has_offsets, f.has_payloads);
                f.terms.push(TermVectorTerm {
                    term: unhex(&r[1]),
                    freq: r[2].parse().unwrap(),
                    positions: pos.then(Vec::new),
                    start_offsets: offs.then(Vec::new),
                    end_offsets: offs.then(Vec::new),
                    payloads: (pos && pays).then(Vec::new),
                });
            }
            "pos" => {
                let (_, f, _) = docs.last_mut().unwrap().last_mut().unwrap();
                let t = f.terms.last_mut().unwrap();
                if let Some(p) = &mut t.positions {
                    p.push(r[1].parse().unwrap());
                }
                if let Some(s) = &mut t.start_offsets {
                    s.push(r[2].parse().unwrap());
                }
                if let Some(e) = &mut t.end_offsets {
                    e.push(r[3].parse().unwrap());
                }
                if let Some(p) = &mut t.payloads {
                    p.push(if r[4] == "-" {
                        Vec::new()
                    } else {
                        unhex(&r[4])
                    });
                }
            }
            "ceil" => docs.last_mut().unwrap().last_mut().unwrap().2.push(r),
            other => panic!("{other}"),
        }
    }
    let reader = SimpleTextTermVectorsReader::open(data, docs.len() as i32).unwrap();
    assert_eq!(reader.max_doc(), docs.len());
    let mut out = Vec::new();
    for (d, want) in docs.into_iter().enumerate() {
        let got = reader.get(d as i32).unwrap();
        if none[d] {
            assert!(got.is_none(), "doc {d}");
            out.push(TermVectorsDocument { fields: Vec::new() });
            continue;
        }
        let got = got.unwrap();
        assert_eq!(
            got.iter().map(|g| g.0.clone()).collect::<Vec<_>>(),
            want.iter().map(|w| w.0.clone()).collect::<Vec<_>>(),
            "doc {d} fields"
        );
        for ((_, g), (name, w, ceils)) in got.iter().zip(&want) {
            assert_eq!(g, w, "doc {d} field {name}");
            // The enums, as the manifest walked them.
            let mut te = SimpleTextTvTermsEnum::new(g);
            for t in &w.terms {
                assert_eq!(te.next(), Some(t.term.as_slice()));
                assert_eq!(te.term(), Some(t.term.as_slice()));
                assert_eq!(te.total_term_freq().unwrap(), i64::from(t.freq));
                assert_eq!(te.doc_freq(), 1);
                let mut pe = te.postings(PAYLOADS).unwrap();
                assert_eq!(pe.next_doc(), 0);
                if t.positions.is_some() || t.start_offsets.is_some() {
                    assert_eq!(pe.freq(), t.freq);
                    for k in 0..t.freq as usize {
                        let p = pe.next_position().unwrap();
                        assert_eq!(p, t.positions.as_ref().map_or(-1, |v| v[k]));
                        assert_eq!(
                            pe.start_offset(),
                            t.start_offsets.as_ref().map_or(-1, |v| v[k])
                        );
                        assert_eq!(pe.end_offset(), t.end_offsets.as_ref().map_or(-1, |v| v[k]));
                        let pay = t
                            .payloads
                            .as_ref()
                            .map(|v| v[k].as_slice())
                            .filter(|p| !p.is_empty());
                        assert_eq!(pe.payload(), pay);
                    }
                    assert!(pe.next_position().is_err());
                }
                assert_eq!(pe.next_doc(), NO_MORE_DOCS);
            }
            assert_eq!(te.next(), None);
            for c in ceils {
                let mut te = SimpleTextTvTermsEnum::new(g);
                let st = te.seek_ceil(&unhex(&c[1]));
                let term = if st == SeekStatus::End {
                    "-".to_string()
                } else {
                    hex(te.term().unwrap())
                };
                assert_eq!(
                    [status(st).to_string(), term],
                    [c[2].clone(), c[3].clone()],
                    "doc {d} {name} seekCeil({})",
                    c[1]
                );
            }
        }
        out.push(TermVectorsDocument {
            fields: want.into_iter().map(|w| w.1).collect(),
        });
    }
    out
}

#[test]
fn flushed_doc_values_norms_and_vectors_read_and_rebuild() {
    let seen = check_case("flush");
    for f in [
        "_0.dat", "_0_1.dat", "_0_2.dat", "_1.dat", "_0.len", "_1.len", "_0.tvc", "_1.tvc",
    ] {
        assert!(seen.iter().any(|(n, _)| n == f), "{f}");
    }
}

#[test]
fn merged_doc_values_norms_and_vectors_read_and_rebuild() {
    let seen = check_case("merged");
    for f in ["_3.dat", "_3_1.dat", "_3.len", "_3.tvc"] {
        assert!(seen.iter().any(|(n, _)| n == f), "{f}");
    }
}

fn java(file: &str) -> Vec<u8> {
    fs::read(root().join("flush").join("index").join(file)).unwrap()
}

/// A flipped byte inside the body, and truncations, are refused by `read`.
fn refuses_raw_damage(data: &[u8], read: impl Fn(&[u8]) -> bool) {
    assert!(read(data), "the intact file reads");
    let mut flipped = data.to_vec();
    flipped[data.len() / 2] ^= 0x01;
    assert!(!read(&flipped), "a flipped byte");
    for cut in [1, 2, 10, 29, data.len() / 2] {
        assert!(!read(&data[..data.len() - cut]), "truncated by {cut}");
    }
    assert!(!read(&[]), "empty");
}

/// Opens a `.dat` and reads every field of it fully, checking integrity.
fn read_all_dv(data: &[u8], max_doc: i32) -> bool {
    let full = || -> Result<(), lucene_codecs::simpletext::Error> {
        let r = SimpleTextDocValuesReader::open(data, max_doc)?;
        r.check_integrity()?;
        for name in file_fields(data) {
            let ty = r
                .field(&name)
                .ok_or(lucene_codecs::simpletext::Error::IllegalState("gone"))?
                .doc_values_type;
            match ty {
                DocValuesType::Numeric => {
                    let mut it = r.numeric(&name)?;
                    while it.next_doc()? != NO_MORE_DOCS {
                        it.long_value()?;
                    }
                }
                DocValuesType::Binary => {
                    let mut it = r.binary(&name)?;
                    while it.next_doc()? != NO_MORE_DOCS {
                        it.binary_value()?;
                    }
                }
                DocValuesType::Sorted => {
                    let mut it = r.sorted(&name)?;
                    while it.next_doc()? != NO_MORE_DOCS {
                        it.lookup_ord(it.ord_value())?;
                    }
                }
                DocValuesType::SortedNumeric => {
                    let mut it = r.sorted_numeric(&name)?;
                    while it.next_doc()? != NO_MORE_DOCS {
                        for _ in 0..it.doc_value_count() {
                            it.next_value()?;
                        }
                    }
                }
                DocValuesType::SortedSet => {
                    let mut it = r.sorted_set(&name)?;
                    while it.next_doc()? != NO_MORE_DOCS {
                        for _ in 0..it.doc_value_count() {
                            let o = it.next_ord()?;
                            it.lookup_ord(o)?;
                        }
                    }
                }
                DocValuesType::None => {}
            }
        }
        Ok(())
    };
    full().is_ok()
}

#[test]
fn damaged_doc_values_are_refused() {
    let data = java("_0.dat");
    let max_doc = 150;
    refuses_raw_damage(&data, |d| read_all_dv(d, max_doc));
    let text = latin1(&data);
    // Semantic damage under a valid checksum: each is refused at open or
    // on first read, never a panic.
    let damage: Vec<(&str, String)> = vec![
        ("type", text.replacen("  type NUMERIC", "  type NUMBER", 1)),
        ("doccount", text.replacen("  doccount ", "  doccount x", 1)),
        ("minvalue", text.replacen("  minalue ", "  minalue x", 1)),
        ("origin", text.replacen("  origin ", "  origin 9", 1)),
        ("pattern", text.replacen("  pattern 0", "  pattern #", 1)),
        (
            "maxlength",
            text.replacen("  maxlength ", "  maxlength 99999999", 1),
        ),
        (
            "numvalues",
            text.replacen("  numvalues ", "  numvalues -", 1),
        ),
        (
            "field prefix",
            text.replacen("field num\n", "feld num\n", 1),
        ),
        ("skip", text.replacen("    interval", "    intreval", 1)),
    ];
    for (what, t) in damage {
        assert_ne!(t, text, "{what} applied");
        assert!(!read_all_dv(&reseal(&t), max_doc), "{what}");
    }
    // The wrong maxDoc seeks past the data.
    assert!(!read_all_dv(&data, 10_000));
    assert!(!read_all_dv(&data, -1));
}

#[test]
fn damaged_norms_are_refused() {
    let data = java("_0.len");
    let read = |d: &[u8]| -> bool {
        let Ok(p) = SimpleTextNormsProducer::open(d, 150) else {
            return false;
        };
        p.check_integrity().is_ok()
            && file_fields(d).iter().all(|f| {
                p.norms(f).is_ok_and(|mut it| loop {
                    match it.next_doc() {
                        Ok(NO_MORE_DOCS) => break true,
                        Ok(_) if it.long_value().is_ok() => {}
                        _ => break false,
                    }
                })
            })
    };
    refuses_raw_damage(&data, read);
}

#[test]
fn damaged_term_vectors_are_refused() {
    let data = java("_0.tvc");
    let read = |d: &[u8]| -> bool {
        let Ok(r) = SimpleTextTermVectorsReader::open(d, 150) else {
            return false;
        };
        (0..150).all(|n| r.get(n).is_ok())
    };
    refuses_raw_damage(&data, read);
    let text = latin1(&data);
    for (what, t) in [
        (
            "numfields",
            text.replacen("  numfields ", "  numfields x", 1),
        ),
        ("freq", text.replacen("      freq ", "      freq -", 1)),
        (
            "position",
            text.replacen("      position ", "      position x", 1),
        ),
        (
            "numterms",
            text.replacen("    numterms ", "    numterms 9", 1),
        ),
        ("doc", text.replacen("doc 3\n", "dcc 3\n", 1)),
    ] {
        assert_ne!(t, text, "{what} applied");
        assert!(!read(&reseal(&t)), "{what}");
    }
    let r = SimpleTextTermVectorsReader::open(&data, 150).unwrap();
    assert!(r.get(150).is_err());
    assert!(r.get(-1).is_err());
    assert!(SimpleTextTermVectorsReader::open(&data, 149).is_err());
    assert!(SimpleTextTermVectorsReader::open(&data, 151).is_err());
    assert!(SimpleTextTermVectorsReader::open(&data, -1).is_err());
}
