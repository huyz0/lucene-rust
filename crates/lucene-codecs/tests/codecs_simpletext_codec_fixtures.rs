//! lucene-codecs' `org.apache.lucene.codecs.simpletext` per-segment formats
//! against Java: `GenCodecsSimpleTextCodec` runs an `IndexWriter` on
//! `SimpleTextCodec` and records, for a flushed index (index sort, soft
//! deletes, parent field, doc-values update, hard deletes, every field kind)
//! and a force-merged compound one, what Java's readers make of each
//! segment's `.si`, `.inf`, `.liv`, `.scf` and `.fld`.
//!
//! Here the Rust readers open Java's files and agree with every manifest,
//! the Rust writers rebuild each file byte for byte (from the manifest where
//! it has every value, from the read model where Java's hash order is not
//! in it), and damaged copies of Java's bytes are refused.

// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::fs;
use std::path::{Path, PathBuf};

use lucene_codecs::field_infos::FieldInfos;
use lucene_codecs::simpletext::compound::{self, SimpleTextCompoundReader};
use lucene_codecs::simpletext::field_infos::{
    self as st_infos, doc_values_type_name, index_options_name, similarity_name,
    skip_index_type_name, vector_encoding_name,
};
use lucene_codecs::simpletext::stored_fields_reader::SimpleTextStoredFieldsReader;
use lucene_codecs::simpletext::{live_docs, segment_info, stored_fields_writer, util};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::geo::{java_double_string, java_float_string};

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

/// Bytes as Latin-1 text, so files holding raw binary values (`.fld`,
/// `.scf`) can be edited as strings and turned back losslessly.
fn latin1(data: &[u8]) -> String {
    data.iter().map(|&b| char::from(b)).collect()
}

/// The inverse of `latin1`.
fn unlatin1(text: &str) -> Vec<u8> {
    text.chars()
        .map(|c| u8::try_from(u32::from(c)).unwrap())
        .collect()
}

/// A checksummed file's bytes before its `checksum ` line, as text.
fn body(data: &[u8]) -> String {
    latin1(data).rsplit_once("checksum ").unwrap().0.to_string()
}

/// `body` with a fresh, valid checksum line: semantic damage only.
fn reseal(body: &str) -> Vec<u8> {
    let mut out = unlatin1(body);
    util::write_checksum(&mut out);
    out
}

/// One segment of `segments.tsv`.
struct Seg {
    name: String,
    max_doc: i32,
    del_count: i32,
    del_gen: i64,
    field_infos_gen: i64,
}

fn segments(dir: &Path) -> Vec<Seg> {
    rows(&dir.join("segments.tsv"))
        .into_iter()
        .map(|r| Seg {
            name: r[0].clone(),
            max_doc: r[1].parse().unwrap(),
            del_count: r[2].parse().unwrap(),
            del_gen: r[4].parse().unwrap(),
            field_infos_gen: r[5].parse().unwrap(),
        })
        .collect()
}

fn sorted(mut v: Vec<(String, String)>) -> Vec<(String, String)> {
    v.sort();
    v
}

/// Reads `name.si` and checks it against `si.NAME.tsv`; returns the bytes,
/// the id and whether the segment is compound.
fn check_segment_info(dir: &Path, seg: &Seg) -> (Vec<u8>, [u8; 16], bool) {
    let data = fs::read(dir.join("index").join(format!("{}.si", seg.name))).unwrap();
    let manifest = rows(&dir.join(format!("si.{}.tsv", seg.name)));
    let get = |k: &str| manifest.iter().find(|r| r[0] == k).unwrap()[1].clone();
    let id: [u8; 16] = unhex(&get("id")).try_into().unwrap();
    let si = segment_info::read(&data, &seg.name, &id).unwrap();
    assert_eq!(si.version.to_string(), get("version"));
    assert_eq!(
        si.min_version
            .map(|v| v.to_string())
            .unwrap_or("null".into()),
        get("minVersion")
    );
    assert_eq!(si.doc_count.to_string(), get("docCount"));
    assert_eq!(si.doc_count, seg.max_doc);
    assert_eq!(si.is_compound_file.to_string(), get("compound"));
    assert_eq!(si.has_blocks.to_string(), get("hasBlocks"));
    let pairs = |tag: &str| -> Vec<(String, String)> {
        manifest
            .iter()
            .filter(|r| r[0] == tag)
            .map(|r| (unhex_str(&r[1]), unhex_str(&r[2])))
            .collect()
    };
    assert_eq!(sorted(si.diagnostics.clone()), pairs("diag"));
    assert_eq!(sorted(si.attributes.clone()), pairs("att"));
    let mut files = si.files.clone();
    files.sort();
    let want: Vec<String> = manifest
        .iter()
        .filter(|r| r[0] == "file")
        .map(|r| r[1].clone())
        .collect();
    assert_eq!(files, want);
    let sorts: Vec<_> = manifest.iter().filter(|r| r[0] == "sort").collect();
    assert_eq!(si.index_sort.len(), sorts.len());
    for (e, r) in si.index_sort.iter().zip(&sorts) {
        assert_eq!(e.provider, r[1]);
        assert_eq!(e.description, unhex_str(&r[2]));
        assert_eq!(hex(&e.bytes), r[3]);
    }
    // Java's diagnostics/attributes/files are hash-ordered, so the rebuild
    // starts from what was read (file order kept) rather than the manifest.
    let mut copy = si.clone();
    assert_eq!(
        segment_info::write(&mut copy).unwrap(),
        data,
        "{}.si",
        seg.name
    );
    assert_eq!(copy, si);
    (data, id, si.is_compound_file)
}

/// Checks one `.inf` against its manifest and rebuilds it.
fn check_field_infos(data: &[u8], manifest: &Path) -> FieldInfos {
    let fis = st_infos::read(data).unwrap();
    let want = rows(manifest);
    assert_eq!(fis.fields.len(), want.len());
    for (fi, r) in fis.fields.iter().zip(&want) {
        let mut atts = fi.attributes.clone();
        atts.sort();
        let atts: Vec<String> = atts
            .iter()
            .map(|(k, v)| format!("{}={}", hex(k.as_bytes()), hex(v.as_bytes())))
            .collect();
        let got = [
            hex(fi.name.as_bytes()),
            fi.number.to_string(),
            index_options_name(fi.index_options).to_string(),
            fi.store_term_vectors.to_string(),
            fi.store_payloads.to_string(),
            fi.omit_norms.to_string(),
            doc_values_type_name(fi.doc_values_type).to_string(),
            skip_index_type_name(fi.doc_values_skip_index_type).to_string(),
            fi.doc_values_gen.to_string(),
            atts.join(","),
            fi.point_dimension_count.to_string(),
            fi.point_index_dimension_count.to_string(),
            fi.point_num_bytes.to_string(),
            fi.vector_dimension.to_string(),
            vector_encoding_name(fi.vector_encoding).to_string(),
            similarity_name(fi.vector_similarity_function).to_string(),
            fi.soft_deletes_field.to_string(),
            fi.parent_field.to_string(),
        ];
        assert_eq!(got.as_slice(), r.as_slice(), "{manifest:?}");
    }
    assert_eq!(st_infos::write(&fis.fields), data, "{manifest:?}");
    fis
}

/// The documents `stored.SEG.tsv` records.
fn stored_manifest(path: &Path) -> Vec<Document> {
    let mut docs: Vec<Document> = Vec::new();
    for r in rows(path) {
        if r[0] == "doc" {
            assert_eq!(r[1].parse::<usize>().unwrap(), docs.len());
            docs.push(Document::default());
            continue;
        }
        let value = match r[1].as_str() {
            "string" => FieldValue::String(unhex_str(&r[2])),
            "binary" => FieldValue::Binary(unhex(&r[2])),
            "int" => FieldValue::Int(r[2].parse().unwrap()),
            "long" => FieldValue::Long(r[2].parse().unwrap()),
            "float" => {
                let v = f32::from_bits(u32::from_str_radix(&r[2], 16).unwrap());
                assert_eq!(java_float_string(v), r[3]);
                FieldValue::Float(v)
            }
            "double" => {
                let v = f64::from_bits(u64::from_str_radix(&r[2], 16).unwrap());
                assert_eq!(java_double_string(v), r[3]);
                FieldValue::Double(v)
            }
            t => panic!("type {t}"),
        };
        docs.last_mut().unwrap().fields.push(StoredField {
            field_number: r[0].parse().unwrap(),
            value,
        });
    }
    docs
}

/// Bit-exact equality (NaN payloads and -0.0 included).
fn same(a: &Document, b: &Document) -> bool {
    a.fields.len() == b.fields.len()
        && a.fields.iter().zip(&b.fields).all(|(x, y)| {
            x.field_number == y.field_number
                && match (&x.value, &y.value) {
                    (FieldValue::Float(p), FieldValue::Float(q)) => p.to_bits() == q.to_bits(),
                    (FieldValue::Double(p), FieldValue::Double(q)) => p.to_bits() == q.to_bits(),
                    (p, q) => format!("{p:?}") == format!("{q:?}"),
                }
        })
}

/// Every file of every segment of one case; returns the Java bytes of each
/// format for the negative cases.
fn check_case(case: &str) -> Vec<(String, Vec<u8>)> {
    let dir = root().join(case);
    let index = dir.join("index");
    let mut seen = Vec::new();
    for seg in segments(&dir) {
        let (si_bytes, _id, compound) = check_segment_info(&dir, &seg);
        seen.push((format!("{}.si", seg.name), si_bytes));
        // The core files, from the directory or the compound file.
        let scf;
        let cfs = if compound {
            scf = fs::read(index.join(compound::compound_file_name(&seg.name))).unwrap();
            let r = SimpleTextCompoundReader::open(&scf, &seg.name).unwrap();
            let want = rows(&dir.join(format!("cfs.{}.tsv", seg.name)));
            assert_eq!(
                r.list_all(),
                want.iter().map(|w| w[0].clone()).collect::<Vec<_>>()
            );
            let mut entries = Vec::new();
            for w in &want {
                assert_eq!(r.file_length(&w[0]).unwrap().to_string(), w[1]);
                entries.push((w[0].clone(), r.open_input(&scf, &w[0]).unwrap().to_vec()));
            }
            assert_eq!(compound::write(&entries), scf, "{}.scf", seg.name);
            seen.push((format!("{}.scf", seg.name), scf.clone()));
            Some(entries)
        } else {
            None
        };
        let core = |file: String| -> Vec<u8> {
            match &cfs {
                Some(entries) => entries.iter().find(|e| e.0 == file).unwrap().1.clone(),
                None => fs::read(index.join(&file)).unwrap(),
            }
        };
        let inf_name = st_infos::field_infos_file_name(&seg.name, "");
        let inf = core(inf_name.clone());
        let fis = check_field_infos(&inf, &dir.join(format!("fields.{inf_name}.tsv")));
        seen.push((inf_name, inf));
        if seg.field_infos_gen != -1 {
            let name = lucene_codecs::simpletext::file_name_from_generation(
                &seg.name,
                st_infos::FIELD_INFOS_EXTENSION,
                seg.field_infos_gen,
            )
            .unwrap();
            let data = fs::read(index.join(&name)).unwrap();
            check_field_infos(&data, &dir.join(format!("fields.{name}.tsv")));
        }
        if seg.del_count > 0 {
            let name = live_docs::live_docs_file_name(&seg.name, seg.del_gen).unwrap();
            let data = fs::read(index.join(&name)).unwrap();
            let want = rows(&dir.join(format!("live.{}.tsv", seg.name)));
            let bits = live_docs::read(&data, seg.max_doc).unwrap();
            assert_eq!(bits.len().to_string(), want[0][0]);
            let got: String = (0..bits.len())
                .map(|d| if bits.get(d) { '1' } else { '0' })
                .collect();
            assert_eq!(got, want[1][0]);
            assert_eq!(got.matches('0').count(), seg.del_count as usize);
            let mut rebuilt = FixedBitSet::new(want[1][0].len());
            for (d, c) in want[1][0].chars().enumerate() {
                if c == '1' {
                    rebuilt.set(d);
                }
            }
            assert_eq!(live_docs::write(&rebuilt), data, "{name}");
            seen.push((name, data));
        }
        let fld_name = stored_fields_writer::stored_fields_file_name(&seg.name);
        let fld = core(fld_name.clone());
        let want = stored_manifest(&dir.join(format!("stored.{}.tsv", seg.name)));
        let reader = SimpleTextStoredFieldsReader::open(&fld, seg.max_doc).unwrap();
        assert_eq!(reader.max_doc(), want.len());
        for (n, w) in want.iter().enumerate() {
            let got = reader.document(n as i32).unwrap();
            assert!(same(&got, w), "{fld_name} doc {n}: {got:?} vs {w:?}");
        }
        assert_eq!(
            stored_fields_writer::write(&want, &fis).unwrap(),
            fld,
            "{fld_name}"
        );
        seen.push((fld_name, fld));
    }
    seen
}

#[test]
fn flushed_segments_read_and_rebuild_byte_for_byte() {
    let seen = check_case("flush");
    // _0 (sorted, blocks, deletes, a field-infos generation) and _1 (the soft update).
    assert!(seen.iter().any(|(n, _)| n == "_0_1.liv"));
    assert!(seen.iter().any(|(n, _)| n == "_1.fld"));
}

#[test]
fn merged_compound_segment_reads_and_rebuilds_byte_for_byte() {
    let seen = check_case("merged");
    assert!(seen.iter().any(|(n, _)| n == "_3.scf"));
}

fn java(case: &str, file: &str) -> Vec<u8> {
    fs::read(root().join(case).join("index").join(file)).unwrap()
}

/// A flipped byte inside the checksummed body, and every truncation point
/// of the trailer, are refused by `read`.
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

#[test]
fn damaged_segment_info_is_refused() {
    let data = java("flush", "_0.si");
    let manifest = rows(&root().join("flush/si._0.tsv"));
    let id = unhex(&manifest.iter().find(|r| r[0] == "id").unwrap()[1]);
    refuses_raw_damage(&data, |d| segment_info::read(d, "_0", &id).is_ok());
    // Another segment's id.
    let mut other = id.clone();
    other[0] ^= 1;
    let err = segment_info::read(&data, "_0", &other)
        .unwrap_err()
        .to_string();
    assert!(err.contains("file mismatch"), "{err}");
    let b = body(&data);
    for (from, to) in [
        (
            "      name SortedSetSortField\n",
            "      name NoSuchSortField\n",
        ),
        (
            "    number of documents 150\n",
            "    number of documents x\n",
        ),
        ("    version 10.5.0\n", "    version 10.x\n"),
        ("    files 11\n", "    files 12\n"),
        ("      file _0.fld\n", "      file 0.fld\n"),
        ("    sort 3\n", "    sort -1\n"),
    ] {
        assert!(b.contains(from), "{from}");
        assert!(
            segment_info::read(&reseal(&b.replacen(from, to, 1)), "_0", &id).is_err(),
            "{to}"
        );
    }
}

#[test]
fn damaged_field_infos_are_refused() {
    let data = java("flush", "_0.inf");
    refuses_raw_damage(&data, |d| st_infos::read(d).is_ok());
    let b = body(&data);
    for (from, to) in [
        ("  number 1\n", "  number 0\n"),
        ("  index options DOCS\n", "  index options DOCZ\n"),
        ("number of fields 58\n", "number of fields 59\n"),
        ("  vector similarity COSINE\n", "  vector similarity COS\n"),
    ] {
        assert!(b.contains(from), "{from}");
        assert!(
            st_infos::read(&reseal(&b.replacen(from, to, 1))).is_err(),
            "{to}"
        );
    }
}

#[test]
fn damaged_live_docs_are_refused() {
    let data = java("flush", "_0_1.liv");
    refuses_raw_damage(&data, |d| live_docs::read(d, 150).is_ok());
    // More docs than the segment, and a doc past the size.
    assert!(live_docs::read(&data, 149).is_err());
    let b = body(&data);
    let b2 = b.replacen("END\n", "  doc 150\nEND\n", 1);
    assert!(live_docs::read(&reseal(&b2), 150).is_err());
    let b3 = b.replacen("  doc 0\n", "  doc x\n", 1);
    assert!(live_docs::read(&reseal(&b3), 150).is_err());
}

#[test]
fn damaged_stored_fields_are_refused() {
    let data = java("flush", "_0.fld");
    refuses_raw_damage(&data, |d| {
        SimpleTextStoredFieldsReader::open(d, 150).is_ok()
    });
    let b = body(&data);
    // An unknown type passes `open` (the index walk) and fails the visit.
    let bad = reseal(&b.replacen("    type int\n", "    type short\n", 1));
    let r = SimpleTextStoredFieldsReader::open(&bad, 150).unwrap();
    assert!((0..150).any(|n| r.document(n).is_err()));
    // A value that does not parse.
    let bad = reseal(&b.replacen("    type long\n    value ", "    type long\n    value z", 1));
    let r = SimpleTextStoredFieldsReader::open(&bad, 150).unwrap();
    assert!((0..150).any(|n| r.document(n).is_err()));
    // Fewer documents than the segment says.
    assert!(SimpleTextStoredFieldsReader::open(&data, 151).is_err());
    // A document past the end.
    let r = SimpleTextStoredFieldsReader::open(&data, 150).unwrap();
    assert!(r.document(150).is_err());
    assert!(r.document(-1).is_err());
}

#[test]
fn damaged_compound_file_is_refused() {
    let data = java("merged", "_3.scf");
    let ok = |d: &[u8]| {
        SimpleTextCompoundReader::open(d, "_3")
            .and_then(|r| {
                for name in r.list_all().to_vec() {
                    r.open_input(d, &name)?;
                }
                Ok(())
            })
            .is_ok()
    };
    assert!(ok(&data));
    for cut in [1, 5, 40] {
        assert!(!ok(&data[..data.len() - cut]), "truncated by {cut}");
    }
    assert!(!ok(&[]));
    let text = latin1(&data);
    // Swap two entries' names: the table is then out of order.
    let swapped = text
        .replacen("  filename: _3.dat\n", "  filename: _3.tmp\n", 1)
        .replacen("  filename: _3.dii\n", "  filename: _3.dat\n", 1)
        .replacen("  filename: _3.tmp\n", "  filename: _3.dii\n", 1);
    assert_ne!(swapped, text);
    assert!(!ok(&unlatin1(&swapped)));
    // A sub-file ending past the table.
    let (head, tail) = text.split_at(text.find("table of contents, size").unwrap());
    let mut ends = tail.match_indices("    end: ");
    let (at, _) = ends.next().unwrap();
    let line_end = at + tail[at..].find('\n').unwrap();
    let far = format!(
        "{head}{}    end: 99999999{}",
        &tail[..at],
        &tail[line_end..]
    );
    assert!(!ok(&unlatin1(&far)));
    // A flipped byte in the table's keywords.
    let mut flipped = data.clone();
    // A byte offset: `text` is Latin-1 widened to UTF-8.
    let key = b"table of contents, size";
    let k = data.windows(key.len()).position(|w| w == key).unwrap();
    flipped[k] ^= 0x01;
    assert!(!ok(&flipped));
}
