//! Full flat occurrence decoding against Java-written bytes and Java-enumerated
//! values, independently of the nested Rust occurrence reader.
//!
//! `GenAssertingPostings` covers positions-only and rich fields, absent/changing
//! payloads, singletons and mapped merges. `GenPostingsSkip` covers many complete
//! 256-occurrence blocks plus a vint tail, sparse docs and document boundaries
//! around 255/256, 511/512 and 8191/8192. The Java manifests are the oracle;
//! no Lucene bytes or expected occurrences are synthesized here.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use lucene_codecs::blocktree;
use lucene_codecs::field_infos;
use lucene_codecs::postings::{self, DocInput, FlatOccurrences, PayInput, PosInput};

#[derive(Debug)]
struct Occurrence {
    position: i32,
    start: i32,
    end: i32,
    payload: Vec<u8>,
}

fn unhex(value: &str) -> Vec<u8> {
    if matches!(value, "-" | "NONE") {
        return Vec::new();
    }
    assert!(value.len().is_multiple_of(2));
    (0..value.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&value[at..at + 2], 16).unwrap())
        .collect()
}

fn properties(path: &Path) -> BTreeMap<String, String> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter_map(|line| line.split_once('='))
        .map(|(key, value)| (key.into(), value.into()))
        .collect()
}

struct Fixture {
    fields: blocktree::BlockTreeFields,
    doc: Vec<u8>,
    pos: Vec<u8>,
    pay: Vec<u8>,
    id: [u8; 16],
    suffix: String,
}

impl Fixture {
    fn open(
        read: impl Fn(&str) -> Vec<u8>,
        fnm: &[u8],
        id: [u8; 16],
        suffix: &str,
        max_doc: i32,
    ) -> Self {
        let infos = field_infos::parse(fnm, &id, "").unwrap();
        Self {
            fields: blocktree::open(
                &read("tim"),
                &read("tip"),
                &read("tmd"),
                &infos,
                &id,
                suffix,
                max_doc,
            )
            .unwrap(),
            doc: read("doc"),
            pos: read("pos"),
            pay: read("pay"),
            id,
            suffix: suffix.into(),
        }
    }

    fn inputs(&self) -> (DocInput<'_>, PosInput<'_>, PayInput<'_>) {
        (
            DocInput::open(&self.doc, &self.id, &self.suffix).unwrap(),
            PosInput::open(&self.pos, &self.id, &self.suffix).unwrap(),
            PayInput::open(&self.pay, &self.id, &self.suffix).unwrap(),
        )
    }
}

fn assert_document(
    flat: &FlatOccurrences,
    index: usize,
    expected: &[Occurrence],
    offsets: bool,
    payloads: bool,
) {
    let actual = flat.document(index).expect("expected document slice");
    assert_eq!(
        actual.positions,
        expected
            .iter()
            .map(|value| value.position)
            .collect::<Vec<_>>()
    );
    if offsets {
        assert_eq!(
            actual.start_offsets,
            expected.iter().map(|value| value.start).collect::<Vec<_>>()
        );
        assert_eq!(
            actual.end_offsets,
            expected.iter().map(|value| value.end).collect::<Vec<_>>()
        );
    } else {
        assert!(actual.start_offsets.is_empty());
        assert!(actual.end_offsets.is_empty());
    }
    if payloads {
        assert_eq!(
            actual.payload_lengths,
            expected
                .iter()
                .map(|value| u32::try_from(value.payload.len()).unwrap())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            actual.payload_bytes,
            expected
                .iter()
                .flat_map(|value| value.payload.iter().copied())
                .collect::<Vec<_>>()
        );
    } else {
        assert!(actual.payload_lengths.is_empty());
        assert!(actual.payload_bytes.is_empty());
    }
    // Repeated access preserves the borrowed storage and cannot consume a doc.
    assert_eq!(
        actual.positions.as_ptr(),
        flat.document(index).unwrap().positions.as_ptr()
    );
}

#[test]
fn flat_documents_match_java_positions_only_and_rich_flush_and_merge_snapshots() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/testframework/data");
    let id = std::array::from_fn(|index| index as u8);
    let fnm = std::fs::read(root.join("asserting_postings.fnm")).unwrap();
    for prefix in [
        "asserting_postings",
        "asserting_postings_merge_sequential",
        "asserting_postings_merge_sorted",
    ] {
        let manifest = std::fs::read_to_string(root.join(format!("{prefix}.tsv"))).unwrap();
        let rows: Vec<_> = manifest
            .lines()
            .map(|line| line.split('\t').collect::<Vec<_>>())
            .collect();
        let max_doc = rows.iter().find(|row| row[0] == "maxDoc").unwrap()[1]
            .parse()
            .unwrap();
        let fixture = Fixture::open(
            |extension| std::fs::read(root.join(format!("{prefix}.{extension}"))).unwrap(),
            &fnm,
            id,
            "s",
            max_doc,
        );
        let (doc, pos, pay) = fixture.inputs();
        let mut checked_fields = 0;
        for field_row in rows.iter().filter(|row| row[0] == "field") {
            if field_row[3].parse::<u8>().unwrap() < 3 {
                continue;
            }
            let name = field_row[1];
            let field = fixture.fields.field(name).unwrap();
            let mut by_term = BTreeMap::<Vec<u8>, Vec<&Vec<&str>>>::new();
            for row in rows
                .iter()
                .filter(|row| row[0] == "posting" && row[1] == name)
            {
                by_term.entry(unhex(row[2])).or_default().push(row);
            }
            for (term, expected_docs) in by_term {
                let postings = field.postings(&term, Some(&doc)).unwrap().unwrap();
                let stats = field.try_seek_exact(&term).unwrap().unwrap();
                let meta = field.term_metadata(&term).unwrap().unwrap();
                let total: i64 = expected_docs
                    .iter()
                    .map(|row| row[4].parse::<i64>().unwrap())
                    .sum();
                assert_eq!(stats.total_term_freq, total, "{prefix}/{name}");
                assert_eq!(postings.docs.len(), expected_docs.len());
                let flat = postings::read_occurrences_flat(
                    &pos,
                    (field.index_options().subsumes_offsets() || field.has_payloads())
                        .then_some(&pay),
                    meta,
                    &postings.freqs,
                    stats.total_term_freq,
                    field.index_options(),
                    field.has_payloads(),
                )
                .unwrap();
                for (index, row) in expected_docs.iter().enumerate() {
                    assert_eq!(postings.docs[index], row[3].parse::<i32>().unwrap());
                    assert_eq!(postings.freqs[index], row[4].parse::<i32>().unwrap());
                    let positions: Vec<i32> = row[5]
                        .split(',')
                        .map(|value| value.parse().unwrap())
                        .collect();
                    let offsets: Vec<_> = row[6]
                        .split(',')
                        .map(|value| value.split_once(':').unwrap())
                        .collect();
                    let payloads: Vec<_> = row[7].split(',').map(unhex).collect();
                    let expected: Vec<_> = positions
                        .iter()
                        .enumerate()
                        .map(|(at, &position)| Occurrence {
                            position,
                            start: offsets[at].0.parse().unwrap(),
                            end: offsets[at].1.parse().unwrap(),
                            payload: payloads[at].clone(),
                        })
                        .collect();
                    assert_eq!(expected.len(), postings.freqs[index] as usize);
                    assert_document(
                        &flat,
                        index,
                        &expected,
                        field.index_options().subsumes_offsets(),
                        field.has_payloads(),
                    );
                }
                assert!(flat.document(expected_docs.len()).is_none());
                assert!(flat.document(usize::MAX).is_none());
            }
            checked_fields += 1;
        }
        assert!(checked_fields >= 2, "positions-only and rich fields");
    }
}

#[test]
fn flat_full_blocks_and_vint_tail_match_java_dense_and_sparse_boundary_samples() {
    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/postings_skip_index");
    let manifest = properties(&root.join("manifest.properties"));
    let id: [u8; 16] = unhex(&manifest["id_hex"]).try_into().unwrap();
    let fixture = Fixture::open(
        |extension| {
            let name = &manifest[&format!("{extension}_file_name")];
            std::fs::read(root.join(format!("{name}.raw"))).unwrap()
        },
        &std::fs::read(root.join(format!("{}.raw", manifest["fnm_file_name"]))).unwrap(),
        id,
        &manifest["segment_suffix"],
        manifest["max_doc"].parse().unwrap(),
    );
    let (doc, pos, pay) = fixture.inputs();
    let field = fixture.fields.field("pskip").unwrap();
    for (term_key, prefix, df_key, ttf_key) in [
        ("term", "doc", "docFreq", "totalTermFreq"),
        (
            "sparse_term",
            "sparse",
            "sparse_docFreq",
            "sparse_totalTermFreq",
        ),
    ] {
        let term = manifest[term_key].as_bytes();
        let postings = field.postings(term, Some(&doc)).unwrap().unwrap();
        let stats = field.try_seek_exact(term).unwrap().unwrap();
        assert_eq!(stats.doc_freq, manifest[df_key].parse::<i32>().unwrap());
        assert_eq!(
            stats.total_term_freq,
            manifest[ttf_key].parse::<i64>().unwrap()
        );
        assert_eq!(postings.docs.len(), stats.doc_freq as usize);
        assert!(stats.total_term_freq > 256 && stats.total_term_freq % 256 != 0);
        let flat = postings::read_occurrences_flat(
            &pos,
            Some(&pay),
            field.term_metadata(term).unwrap().unwrap(),
            &postings.freqs,
            stats.total_term_freq,
            field.index_options(),
            field.has_payloads(),
        )
        .unwrap();
        for (index, &freq) in postings.freqs.iter().enumerate() {
            let actual = flat.document(index).unwrap();
            assert_eq!(actual.positions.len(), freq as usize);
            assert_eq!(actual.start_offsets.len(), freq as usize);
            assert_eq!(actual.end_offsets.len(), freq as usize);
            assert_eq!(actual.payload_lengths.len(), freq as usize);
            assert_eq!(
                actual
                    .payload_lengths
                    .iter()
                    .map(|&len| len as usize)
                    .sum::<usize>(),
                actual.payload_bytes.len()
            );
        }
        let mut checked = 0;
        for target in manifest["sampled_docs"].split(',') {
            let actual_doc: i32 = if prefix == "sparse" {
                manifest[&format!("sparse.{target}.advance")]
                    .parse()
                    .unwrap()
            } else {
                target.parse().unwrap()
            };
            if actual_doc == postings::NO_MORE_DOCS {
                continue;
            }
            let index = postings.docs.binary_search(&actual_doc).unwrap();
            let expected: Vec<_> = manifest[&format!("{prefix}.{target}.occurrences")]
                .split(';')
                .map(|value| {
                    let values: Vec<_> = value.split(',').collect();
                    Occurrence {
                        position: values[0].parse().unwrap(),
                        start: values[1].parse().unwrap(),
                        end: values[2].parse().unwrap(),
                        payload: unhex(values[3]),
                    }
                })
                .collect();
            assert_document(&flat, index, &expected, true, true);
            checked += 1;
        }
        assert!(
            checked > 40,
            "Java samples both sides of many block boundaries"
        );
        assert!(flat.document(postings.docs.len()).is_none());
        assert!(flat.document(usize::MAX).is_none());
    }
}
