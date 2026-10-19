//! Unit tests of the bloom port: the checks and messages the Java-fixture
//! tests (`tests/codecs_bloom_fixtures.rs`) never reach.

// Test-support arithmetic over small in-memory values; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::field_infos::IndexOptions;
use crate::postings_writer::TermPostings;
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::data_output::DataOutput;

const ID: [u8; ID_LENGTH] = [9; ID_LENGTH];
const SUFFIX: &str = "BloomFilter_0";

fn term(t: &str, docs: &[i32]) -> TermPostings {
    TermPostings {
        term: t.as_bytes().to_vec(),
        docs: docs.iter().map(|&d| (d, 1)).collect(),
        ..TermPostings::default()
    }
}

fn infos() -> FieldInfos {
    FieldInfos::new(vec![
        FieldInfo::new("id", 0).with_index_options(IndexOptions::Docs),
        FieldInfo::new("tag", 1).with_index_options(IndexOptions::Docs),
    ])
    .unwrap()
}

fn id_terms() -> Vec<TermPostings> {
    (0..200).map(|i| term(&format!("k{i:04}"), &[i])).collect()
}

fn tag_terms() -> Vec<TermPostings> {
    vec![term("a", &[0, 1, 2]), term("b", &[3])]
}

fn write_state<'a>(fis: &'a FieldInfos, max_doc: i32) -> SegmentWriteState<'a> {
    SegmentWriteState {
        max_doc,
        segment_id: &ID,
        segment_suffix: SUFFIX,
        field_infos: fis,
    }
}

fn read_state(fis: &FieldInfos) -> SegmentReadState<'_> {
    SegmentReadState {
        max_doc: 200,
        segment_id: &ID,
        segment_suffix: SUFFIX,
        field_infos: fis,
    }
}

fn inputs<'a>(id: &'a [TermPostings], tag: &'a [TermPostings]) -> Vec<FieldPostingsInput<'a>> {
    vec![
        FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::Docs,
            doc_count: 200,
            has_payloads: false,
            terms: id,
        },
        FieldPostingsInput {
            field_number: 1,
            index_options: IndexOptions::Docs,
            doc_count: 4,
            has_payloads: false,
            terms: tag,
        },
    ]
}

fn written() -> BloomOutput {
    let fis = infos();
    let (id, tag) = (id_terms(), tag_terms());
    BloomFilteringPostingsFormat::new(Lucene104PostingsFormat::default())
        .write(&inputs(&id, &tag), &[], &write_state(&fis, 200))
        .unwrap()
}

fn open(out: &BloomOutput, blm: &[u8]) -> Result<BloomFilteredFieldsProducer> {
    let fis = infos();
    let d = &out.delegate;
    BloomFilteredFieldsProducer::open(blm, &d.tim, &d.tip, &d.tmd, &read_state(&fis))
}

#[test]
fn written_groups_read_back_filtered() {
    let out = written();
    let p = open(&out, &out.blm).unwrap();
    assert_eq!(p.delegate_name(), "Lucene104");
    assert_eq!(p.size(), 2);
    assert_eq!(p.field_names().collect::<Vec<_>>(), ["id", "tag"]);
    assert_eq!(
        p.to_string(),
        "BloomFilteredFieldsProducer(fields=2,delegate=Lucene104)"
    );
    let id = p.terms("id").unwrap();
    assert!(id.filter().is_some());
    assert_eq!(id.size(), 200);
    assert_eq!(id.doc_count(), 200);
    assert_eq!(id.sum_doc_freq(), 200);
    assert_eq!(id.sum_total_term_freq(), 200);
    assert_eq!(id.min(), b"k0000");
    assert_eq!(id.max(), b"k0199");
    assert_eq!(id.delegate().num_terms, 200);
    for i in 0..200 {
        let t = format!("k{i:04}");
        assert_eq!(
            id.try_seek_exact(t.as_bytes()).unwrap().unwrap().doc_freq,
            1
        );
    }
    // Absent terms: some refused by the filter, every one missing.
    let mut refused = 0;
    for i in 200..2200 {
        let t = format!("k{i:04}");
        if id.filter().unwrap().contains(t.as_bytes()) == ContainsResult::No {
            refused += 1;
        }
        assert!(id.try_seek_exact(t.as_bytes()).unwrap().is_none());
    }
    assert!(refused > 1000, "{refused}");
    let mut walk = id.iter();
    assert_eq!(walk.try_next_term().unwrap(), Some(&b"k0000"[..]));
    assert!(p.terms("missing").is_none());
    assert!(p.bloom("tag").is_some());
    assert!(check_blm_footer(&out.blm).is_ok());
}

#[derive(Debug)]
struct OnlyTag;

impl BloomFilterFactory for OnlyTag {
    fn get_set_for_field(
        &self,
        state: &SegmentWriteState<'_>,
        info: &FieldInfo,
    ) -> lucene_store::Result<Option<FuzzySet>> {
        if info.name == "tag" {
            DefaultBloomFilterFactory.get_set_for_field(state, info)
        } else {
            Ok(None)
        }
    }

    fn downsize(&self, _info: &FieldInfo, _initial: &FuzzySet) -> Option<FuzzySet> {
        None
    }

    fn is_saturated(&self, _set: &FuzzySet, _info: &FieldInfo) -> bool {
        false
    }
}

#[test]
fn a_factory_can_leave_a_field_unfiltered_and_skip_downsizing() {
    let fis = infos();
    let (id, tag) = (id_terms(), tag_terms());
    let out =
        BloomFilteringPostingsFormat::with_factory(Lucene104PostingsFormat::default(), OnlyTag)
            .write(&inputs(&id, &tag), &[], &write_state(&fis, 200))
            .unwrap();
    let p = open(&out, &out.blm).unwrap();
    let id = p.terms("id").unwrap();
    assert!(id.filter().is_none());
    assert!(id.try_seek_exact(b"k0007").unwrap().is_some());
    assert!(id.try_seek_exact(b"zzz").unwrap().is_none());
    // Not downsized: the optimal size for 200 documents, as created.
    let created = FuzzySet::create_optimal_set(200, 0.1023).unwrap();
    assert_eq!(p.bloom("tag").unwrap().bloom_size(), created.bloom_size());
}

#[test]
fn a_saturated_filter_is_not_written() {
    let fis = infos();
    // maxDoc 1 sizes the filter for one value; 200 terms saturate it.
    let (id, tag) = (id_terms(), tag_terms());
    let out = BloomFilteringPostingsFormat::new(Lucene104PostingsFormat::default())
        .write(&inputs(&id, &tag), &[], &write_state(&fis, 1))
        .unwrap();
    let p = open(&out, &out.blm).unwrap();
    assert!(p.bloom("id").is_none());
    assert!(p.bloom("tag").is_some());
    assert!(p
        .terms("id")
        .unwrap()
        .try_seek_exact(b"k0001")
        .unwrap()
        .is_some());
}

#[test]
fn writing_needs_a_lucene104_delegate_and_known_fields() {
    let fis = infos();
    let (id, tag) = (id_terms(), tag_terms());
    let state = write_state(&fis, 200);
    let e = BloomFilteringPostingsFormat::default()
        .write(&inputs(&id, &tag), &[], &state)
        .unwrap_err();
    assert!(matches!(e, Error::NoDelegate));
    assert!(e
        .to_string()
        .contains("constructed without a choice of PostingsFormat"));
    let e = BloomFilteringPostingsFormat::new(Lucene104PostingsFormat::completion104())
        .write(&inputs(&id, &tag), &[], &state)
        .unwrap_err();
    assert!(matches!(e, Error::UnsupportedDelegate(n) if n == "Completion104"));
    let mut bad = inputs(&id, &tag);
    bad[1].field_number = 5;
    let e = BloomFilteringPostingsFormat::new(Lucene104PostingsFormat::default())
        .write(&bad, &[], &state)
        .unwrap_err();
    assert!(e.to_string().contains("field number 5"), "{e}");
    let empty: Vec<TermPostings> = Vec::new();
    let e = BloomFilteringPostingsFormat::new(Lucene104PostingsFormat::default())
        .write(&inputs(&id, &empty), &[], &state)
        .unwrap_err();
    assert!(matches!(e, Error::Write(_)), "{e}");
}

#[test]
fn format_names_and_display() {
    let f = BloomFilteringPostingsFormat::new(Lucene104PostingsFormat::new(4, 9).unwrap());
    assert_eq!(f.name(), "BloomFilter");
    assert_eq!(f.delegate().unwrap().min_term_block_size, 4);
    assert_eq!(
        f.to_string(),
        "BloomFilteringPostingsFormat(PostingsFormat(name=Lucene104))"
    );
    assert_eq!(
        BloomFilteringPostingsFormat::default().to_string(),
        "BloomFilteringPostingsFormat(null)"
    );
    assert!(BloomFilteringPostingsFormat::default().delegate().is_none());
}

/// The `.blm` rebuilt around a different body, with a fresh footer.
fn reframed(delegate: &str, body: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let mut blm = Vec::new();
    codec_util::write_index_header(&mut blm, BLOOM_CODEC_NAME, VERSION_CURRENT, &ID, SUFFIX);
    blm.write_string(delegate);
    body(&mut blm);
    codec_util::write_footer(&mut blm);
    blm
}

#[test]
fn corrupt_blm_files_are_refused() {
    let out = written();
    // A flipped byte fails the checksum.
    let mut flipped = out.blm.clone();
    let mid = flipped.len() / 2;
    flipped[mid] ^= 0x40;
    assert!(open(&out, &flipped).is_err());
    assert!(check_blm_footer(&flipped).is_err());
    assert!(check_blm_footer(&[0; 4]).is_err());
    // A suffix other than the group's.
    let fis = infos();
    let d = &out.delegate;
    let wrong = SegmentReadState {
        segment_suffix: "BloomFilter_1",
        ..read_state(&fis)
    };
    assert!(BloomFilteredFieldsProducer::open(&out.blm, &d.tim, &d.tip, &d.tmd, &wrong).is_err());
    // A delegate this port has no format for.
    let e = open(&out, &reframed("SimpleText", |b| b.write_i32(0))).unwrap_err();
    assert!(matches!(e, Error::UnsupportedDelegate(n) if n == "SimpleText"));
    // A field number FieldInfos does not have.
    let set = FuzzySet::create_set_based_on_max_memory(8).unwrap();
    let e = open(
        &out,
        &reframed("Lucene104", |b| {
            b.write_i32(1);
            b.write_i32(42);
            set.serialize(b);
        }),
    )
    .unwrap_err();
    assert!(e.to_string().contains("invalid field number: 42"), "{e}");
    // Trailing bytes before the footer.
    let e = open(
        &out,
        &reframed("Lucene104", |b| {
            b.write_i32(0);
            b.write_byte(1);
        }),
    )
    .unwrap_err();
    assert!(e.to_string().contains("did not read all bytes"), "{e}");
    // No blooms at all: every field unfiltered.
    let p = open(&out, &reframed("Lucene104", |b| b.write_i32(0))).unwrap();
    assert!(p.terms("id").unwrap().filter().is_none());
}
