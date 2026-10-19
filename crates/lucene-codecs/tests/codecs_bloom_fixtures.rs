//! lucene-codecs' `BloomFilteringPostingsFormat` against Java:
//! `GenCodecsBloom` writes a flushed segment (deletes inside it) and a
//! force-merged one, routing `id` to one bloom instance over the default
//! `Lucene104PostingsFormat` and `tag`/`body` to a second over
//! `Lucene104PostingsFormat(4, 9)` -- `body`'s filter saturates and is left
//! out -- and records each `.blm` entry, `FuzzySet.contains` and
//! `TermsEnum.seekExact` for present and absent terms, and `FuzzySet`'s
//! sizing math over a table of inputs.
//!
//! Here the Rust reader opens each group over Java's files and answers every
//! probe the same way, and the Rust writer rebuilds each group's `.blm`
//! byte for byte from the terms it read.

// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use lucene_codecs::bloom::{
    get_estimated_number_unique_values_allowing_for_collisions, get_nearest_set_size,
    get_nearest_set_size_for_saturation, BloomFilteredFieldsProducer, BloomFilteringPostingsFormat,
    ContainsResult, FuzzySet,
};
use lucene_codecs::field_infos::{self, FieldInfos, IndexOptions};
use lucene_codecs::per_field_postings::Lucene104PostingsFormat;
use lucene_codecs::postings_writer::{FieldPostingsInput, TermPostings};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_store::codec_util::ID_LENGTH;
use lucene_util::java_random::JavaRandom;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/codecs_bloom")
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
        .unwrap_or_else(|e| panic!("{path:?}: {e} (run GenCodecsBloom)"))
        .lines()
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect()
}

/// `GenCodecsBloom.randomTerm`.
fn random_term(r: &mut JavaRandom) -> Vec<u8> {
    let len = 1 + r.next_int_bounded(12);
    (0..len)
        .map(|_| b'a' + r.next_int_bounded(26) as u8)
        .collect()
}

#[test]
fn fuzzy_set_math_matches_java() {
    let mut r = JavaRandom::new(17);
    let mut checked = 0;
    for row in rows(root().join("fuzzyset.tsv")) {
        let int = |i: usize| row[i].parse::<i32>().unwrap();
        let float = |i: usize| f32::from_bits(row[i].parse::<i32>().unwrap() as u32);
        match row[0].as_str() {
            "nearest" => assert_eq!(get_nearest_set_size(int(1)), int(2), "{row:?}"),
            "nearest_sat" => assert_eq!(
                get_nearest_set_size_for_saturation(int(1), float(2)),
                int(3),
                "{row:?}"
            ),
            "estimate" => assert_eq!(
                get_estimated_number_unique_values_allowing_for_collisions(int(1), int(2), int(3)),
                int(4),
                "{row:?}"
            ),
            "optimal" => {
                let n = int(1);
                let mut set = FuzzySet::create_optimal_set(n, float(2)).unwrap();
                let adds = if n == 1 {
                    1
                } else {
                    n / (1 + r.next_int_bounded(4))
                };
                assert_eq!(adds, int(3), "{row:?}");
                let mut first = Vec::new();
                for i in 0..adds {
                    let t = random_term(&mut r);
                    set.add_value(&t);
                    if i < 3 {
                        first.push(hex(&t));
                    }
                }
                assert_eq!(first.join(","), row[9], "replayed terms of {row:?}");
                assert_eq!(set.to_string(), row[4], "{row:?}");
                assert_eq!(set.saturation().to_bits() as i32, int(5), "{row:?}");
                assert_eq!(set.get_estimated_unique_values(), int(6), "{row:?}");
                let down = set.downsize(set.target_max_saturation());
                assert_eq!(
                    down.as_ref()
                        .map_or("null".to_string(), ToString::to_string),
                    row[7],
                    "{row:?}"
                );
                let mut bytes = Vec::new();
                down.as_ref().unwrap_or(&set).serialize(&mut bytes);
                assert_eq!(hex(&bytes), row[8], "{row:?}");
                let back = FuzzySet::deserialize(&mut lucene_store::data_input::SliceInput::new(
                    &unhex(&row[8]),
                ))
                .unwrap();
                assert_eq!(&back, down.as_ref().unwrap_or(&set));
            }
            "max_memory" => {
                let n = int(1);
                let mut set = FuzzySet::create_set_based_on_max_memory(n).unwrap();
                for i in 0..n {
                    set.add_value(i.to_string().as_bytes());
                }
                assert_eq!(set.to_string(), row[2]);
                let mut bytes = Vec::new();
                set.serialize(&mut bytes);
                assert_eq!(hex(&bytes), row[3]);
                let contains: String = (n..n + 200)
                    .map(|i| match set.contains(i.to_string().as_bytes()) {
                        ContainsResult::No => '0',
                        ContainsResult::Maybe => '1',
                    })
                    .collect();
                assert_eq!(contains, row[4]);
            }
            other => panic!("unknown row {other}"),
        }
        checked += 1;
    }
    assert_eq!(checked, 70, "every fuzzyset.tsv row");
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
            .expect("run GenCodecsBloom");
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
        fs::read(
            self.dir
                .join("index")
                .join(format!("{}_{suffix}.{ext}", self.name)),
        )
        .unwrap()
    }

    fn producer_with(&self, suffix: &str, blm: &[u8]) -> BloomFilteredFieldsProducer {
        let state = SegmentReadState {
            max_doc: self.max_doc,
            segment_id: &self.id,
            segment_suffix: suffix,
            field_infos: &self.infos,
        };
        BloomFilteredFieldsProducer::open(
            blm,
            &self.file(suffix, "tim"),
            &self.file(suffix, "tip"),
            &self.file(suffix, "tmd"),
            &state,
        )
        .unwrap()
    }

    fn producer(&self, suffix: &str) -> BloomFilteredFieldsProducer {
        self.producer_with(suffix, &self.file(suffix, "blm"))
    }

    /// `blm.tsv`'s groups: suffix -> (delegate, [(field number, FuzzySet)]).
    fn groups(&self) -> BTreeMap<String, (String, Vec<(i32, String)>)> {
        let mut groups = BTreeMap::new();
        for row in rows(self.dir.join("blm.tsv")) {
            match row[0].as_str() {
                "group" => {
                    groups.insert(row[1].clone(), (row[2].clone(), Vec::new()));
                }
                "field" => groups
                    .get_mut(&row[1])
                    .unwrap()
                    .1
                    .push((row[2].parse().unwrap(), row[3].clone())),
                _ => unreachable!(),
            }
        }
        groups
    }
}

fn check_segment(which: &str) {
    let seg = Segment::open(which);
    let groups = seg.groups();
    assert_eq!(groups.len(), 2);
    let producers: BTreeMap<String, BloomFilteredFieldsProducer> = groups
        .keys()
        .map(|s| (s.clone(), seg.producer(s)))
        .collect();

    // Every `.blm` entry, as `FuzzySet.toString` prints it.
    for (suffix, (delegate, fields)) in &groups {
        let p = &producers[suffix];
        assert_eq!(p.delegate_name(), delegate);
        for (number, set) in fields {
            let name = &seg.infos.field_by_number(*number).unwrap().name;
            assert_eq!(&p.bloom(name).unwrap().to_string(), set, "{which} {name}");
        }
    }

    // Every probe: the filter's answer and the filtered seekExact's.
    let mut probes = 0;
    let mut refused = 0;
    let mut false_positives = 0;
    for row in rows(seg.dir.join("probes.tsv")) {
        let field = &row[1];
        let terms = producers
            .values()
            .find_map(|p| p.terms(field))
            .unwrap_or_else(|| panic!("{which}: no group has {field}"));
        match row[0].as_str() {
            "terms" => {
                assert_eq!(terms.size().to_string(), row[2], "{which} {field}");
                let filter = terms.filter().map_or("-".to_string(), ToString::to_string);
                assert_eq!(filter, row[3], "{which} {field}");
            }
            "probe" => {
                let term = unhex(&row[2]);
                let contains = terms.filter().map_or("-", |f| match f.contains(&term) {
                    ContainsResult::No => "NO",
                    ContainsResult::Maybe => "MAYBE",
                });
                assert_eq!(contains, row[3], "{which} {field} {}", row[2]);
                let found = terms.try_seek_exact(&term).unwrap();
                assert_eq!(
                    found.is_some().to_string(),
                    row[4],
                    "{which} {field} {}",
                    row[2]
                );
                let doc_freq = found.map_or(-1, |s| s.doc_freq);
                assert_eq!(doc_freq.to_string(), row[5], "{which} {field} {}", row[2]);
                probes += 1;
                refused += usize::from(contains == "NO");
                false_positives += usize::from(contains == "MAYBE" && found.is_none());
            }
            _ => unreachable!(),
        }
    }
    assert!(probes > 2000, "{which}: {probes}");
    assert!(refused > 500, "{which}: {refused}");
    assert!(false_positives > 0, "{which}: no false positive exercised");

    // The Rust writer rebuilds each group's `.blm` from the terms read back:
    // the filters depend on nothing else (the delegate's own files are proven
    // by the postings writer's fixtures).
    for suffix in groups.keys() {
        let p = &producers[suffix];
        let mut fields: Vec<(i32, Vec<TermPostings>)> = Vec::new();
        for name in p.field_names() {
            let mut walk = p.terms(name).unwrap().iter();
            let mut terms = Vec::new();
            while let Some(t) = walk.try_next_term().unwrap() {
                terms.push(TermPostings {
                    term: t.to_vec(),
                    docs: vec![(0, 1)],
                    ..TermPostings::default()
                });
            }
            fields.push((seg.infos.field_by_name(name).unwrap().number, terms));
        }
        let inputs: Vec<FieldPostingsInput<'_>> = fields
            .iter()
            .map(|(number, terms)| FieldPostingsInput {
                field_number: *number,
                index_options: IndexOptions::Docs,
                doc_count: 1,
                has_payloads: false,
                terms,
            })
            .collect();
        let state = SegmentWriteState {
            max_doc: seg.max_doc,
            segment_id: &seg.id,
            segment_suffix: suffix,
            field_infos: &seg.infos,
        };
        let delegate = if p.field_names().any(|n| n == "id") {
            Lucene104PostingsFormat::default()
        } else {
            Lucene104PostingsFormat::new(4, 9).unwrap()
        };
        let out = BloomFilteringPostingsFormat::new(delegate)
            .write(&inputs, &[], &state)
            .unwrap();
        assert_eq!(
            hex(&out.blm),
            hex(&seg.file(suffix, "blm")),
            "{which} {suffix}"
        );
    }
}

#[test]
fn flushed_segment_reads_and_rebuilds_like_java() {
    check_segment("flush");
}

#[test]
fn merged_segment_reads_and_rebuilds_like_java() {
    check_segment("merged");
}

#[test]
fn damaged_java_blm_is_refused() {
    let seg = Segment::open("flush");
    let suffix = "BloomFilter_0";
    let blm = seg.file(suffix, "blm");
    let state = SegmentReadState {
        max_doc: seg.max_doc,
        segment_id: &seg.id,
        segment_suffix: suffix,
        field_infos: &seg.infos,
    };
    let open = |blm: &[u8]| {
        BloomFilteredFieldsProducer::open(
            blm,
            &seg.file(suffix, "tim"),
            &seg.file(suffix, "tip"),
            &seg.file(suffix, "tmd"),
            &state,
        )
    };
    assert!(open(&blm).is_ok());
    let mut flipped = blm.clone();
    let at = flipped.len() / 2;
    flipped[at] ^= 1;
    assert!(
        open(&flipped).is_err(),
        "a flipped filter byte fails the checksum"
    );
    assert!(open(&blm[..blm.len() - 5]).is_err(), "a truncated footer");
    let other = SegmentReadState {
        segment_id: &[0; ID_LENGTH],
        ..state
    };
    assert!(BloomFilteredFieldsProducer::open(
        &blm,
        &seg.file(suffix, "tim"),
        &seg.file(suffix, "tip"),
        &seg.file(suffix, "tmd"),
        &other
    )
    .is_err());
    // The same group reopened with its own bytes still answers.
    let p = seg.producer_with(suffix, &blm);
    assert!(p.size() >= 2);
}
