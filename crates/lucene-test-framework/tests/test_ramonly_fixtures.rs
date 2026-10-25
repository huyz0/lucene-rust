//! Java's footerless token and its instance-local RAM state and enum quirks.
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::field_infos;
use lucene_codecs::postings_writer::{FieldPostingsInput, TermPostings};
use lucene_codecs::segment_state::SegmentWriteState;
use lucene_search::reader::{PostingsEnum, PostingsFlags, Terms, TermsEnum, NO_MORE_DOCS};
use lucene_search::Result;
use lucene_test_framework::codecs::ramonly::{RamField, RamOnlyPostingsFormat, RamPostings};

// Java's selected.tsv records what Lucene104 returned for these flags,
// independently of RAMOnly's output. This source also asserts the requested
// flags so using ALL/POSITIONS for the no-payload path cannot pass unnoticed.
struct SelectedSource {
    field: Arc<RamField>,
    flags: PostingsFlags,
}
impl Terms for SelectedSource {
    fn iterator(&self) -> Result<Box<dyn TermsEnum + '_>> {
        Ok(Box::new(SelectedEnum {
            inner: self.field.iterator()?,
            flags: self.flags,
        }))
    }
    fn size(&self) -> i64 {
        self.field.size()
    }
    fn sum_total_term_freq(&self) -> i64 {
        self.field.sum_total_term_freq()
    }
    fn sum_doc_freq(&self) -> i64 {
        self.field.sum_doc_freq()
    }
    fn doc_count(&self) -> i32 {
        self.field.doc_count()
    }
    fn has_freqs(&self) -> bool {
        self.field.has_freqs()
    }
    fn has_positions(&self) -> bool {
        self.field.has_positions()
    }
    fn has_offsets(&self) -> bool {
        self.field.has_offsets()
    }
    fn has_payloads(&self) -> bool {
        self.field.has_payloads()
    }
}
struct SelectedEnum<'a> {
    inner: Box<dyn TermsEnum + 'a>,
    flags: PostingsFlags,
}
impl TermsEnum for SelectedEnum<'_> {
    fn next(&mut self) -> Result<Option<&[u8]>> {
        self.inner.next()
    }
    fn term(&self) -> Option<&[u8]> {
        self.inner.term()
    }
    fn try_seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        self.inner.try_seek_ceil(target)
    }
    fn doc_freq(&mut self) -> Result<i32> {
        self.inner.doc_freq()
    }
    fn total_term_freq(&mut self) -> Result<i64> {
        self.inner.total_term_freq()
    }
    fn postings(&mut self, flags: PostingsFlags) -> Result<Box<dyn PostingsEnum>> {
        assert_eq!(flags, self.flags);
        self.inner.postings(flags)
    }
}

fn unhex(s: &str) -> Vec<u8> {
    s.as_bytes()
        .chunks_exact(2)
        .map(|h| u8::from_str_radix(std::str::from_utf8(h).unwrap(), 16).unwrap())
        .collect()
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn manifest(producer: &RamPostings, payloads: bool) -> String {
    producer.check_integrity().unwrap();
    assert_eq!(producer.size(), 1);
    assert_eq!(producer.field_names(), ["body"]);
    assert!(producer.terms("absent").is_none());
    let terms = producer.terms("body").unwrap();
    assert!(terms.has_freqs());
    assert!(terms.has_positions());
    assert_eq!(terms.has_payloads(), payloads);
    assert!(!terms.has_offsets());
    let mut walk = terms.iterator().unwrap();
    assert!(walk.term().is_none());
    assert!(walk.doc_freq().is_err());
    assert!(walk.total_term_freq().is_err());
    assert!(walk.postings(PostingsFlags::All).is_err());
    assert!(walk.ord().is_err());
    assert!(walk.seek_exact_ord(0).is_err());
    let mut out = String::new();
    while let Some(term) = walk.next().unwrap() {
        out.push_str(&hex(term));
        out.push_str(&format!(
            "\t{}\t{}",
            walk.doc_freq().unwrap(),
            walk.total_term_freq().unwrap()
        ));
        let mut p = walk.postings(PostingsFlags::All).unwrap();
        assert_eq!(p.doc_id(), -1);
        assert_eq!(p.freq(), 0);
        assert!(p.payload().is_none());
        assert!(p.next_position().is_err());
        let mut seen = 0;
        while p.next_doc().unwrap() != NO_MORE_DOCS {
            assert_eq!(p.start_offset(), -1);
            assert_eq!(p.end_offset(), -1);
            out.push_str(&format!("\t{}:{}", p.doc_id(), p.freq()));
            for _ in 0..p.freq() {
                let position = p.next_position().unwrap();
                out.push_str(&format!(
                    ":{position}/{}",
                    p.payload().map_or("-".to_string(), hex)
                ));
            }
            assert!(p.next_position().is_err());
            seen += 1;
        }
        assert_eq!(p.cost(), seen);
        let last = p.doc_id(); // Java retains the last RAMDoc after exhaustion.
        assert_eq!(p.next_doc().unwrap(), NO_MORE_DOCS);
        assert_eq!(p.doc_id(), last);
        out.push('\n');
        let mut advanced = walk.postings(PostingsFlags::None).unwrap();
        let doc = advanced.advance(3).unwrap();
        assert!(doc >= 3);
        assert_eq!(advanced.advance(i32::MAX).unwrap(), NO_MORE_DOCS);
        let mut impacts = walk.impacts(PostingsFlags::Freqs).unwrap();
        assert_eq!(impacts.advance_shallow(0).unwrap(), NO_MORE_DOCS);
        assert_eq!(impacts.impacts()[0].1[0].freq, i32::MAX);
        assert_ne!(impacts.next_doc().unwrap(), NO_MORE_DOCS);
    }
    assert!(walk.next().unwrap().is_none());
    out
}

#[test]
fn java_ramonly_tokens_postings_and_seek_quirks_match() {
    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/testframework/data/ramonly");
    check_case(root.clone(), true);
    check_case(root.join("no-payloads"), false);
}

fn check_case(root: PathBuf, payloads: bool) {
    let infos = field_infos::parse(&fs::read(root.join("_0.fnm")).unwrap(), &[7; 16], "").unwrap();
    let state = SegmentWriteState {
        max_doc: 7,
        segment_id: &[7; 16],
        segment_suffix: "",
        field_infos: &infos,
    };
    let mut inputs = Vec::new();
    let selected = fs::read_to_string(root.join("selected.tsv")).unwrap();
    let all = fs::read_to_string(root.join("input.tsv")).unwrap();
    assert_eq!(selected == all, payloads);
    for line in selected.lines() {
        let cells: Vec<_> = line.split('\t').collect();
        let mut input = TermPostings {
            term: unhex(cells[0]),
            ..TermPostings::default()
        };
        for cell in &cells[3..] {
            let values: Vec<_> = cell.split(':').collect();
            input
                .docs
                .push((values[0].parse().unwrap(), values[1].parse().unwrap()));
            let mut positions = Vec::new();
            for p in &values[2..] {
                let (pos, payload) = p.split_once('/').unwrap();
                positions.push(pos.parse().unwrap());
                let bytes = if payload == "-" {
                    Vec::new()
                } else {
                    unhex(payload)
                };
                input.payload_lengths.push(bytes.len() as u32);
                input.payload_bytes.extend(bytes);
            }
            input.positions.push(positions);
        }
        inputs.push(input);
    }
    let format = RamOnlyPostingsFormat::default();
    assert_eq!(format.name(), "RAMOnly");
    let seed_format = RamOnlyPostingsFormat::default();
    let mut seed = seed_format.fields_consumer();
    seed.write_materialized(
        &[FieldPostingsInput {
            field_number: 0,
            index_options: infos.fields[0].index_options,
            doc_count: 6,
            has_payloads: payloads,
            terms: &inputs,
        }],
        &state,
    )
    .unwrap();
    let source = SelectedSource {
        field: seed_format
            .fields_producer(seed.token())
            .unwrap()
            .unwrap()
            .terms("body")
            .unwrap(),
        flags: if payloads {
            PostingsFlags::Payloads
        } else {
            PostingsFlags::None
        },
    };
    for slot in 0..2 {
        let mut consumer = format.fields_consumer();
        assert_eq!(
            consumer.token(),
            fs::read(root.join(format!("_0_RAMOnly_{slot}.id"))).unwrap()
        );
        let producer = format.fields_producer(consumer.token()).unwrap().unwrap();
        assert_eq!(producer.size(), 0); // The slot exists before write.
        consumer.write(&[(0, &source)], &state).unwrap();
        let expected = fs::read_to_string(root.join(format!("output{slot}.tsv"))).unwrap();
        assert_eq!(manifest(&producer, payloads), expected);
        let terms = producer.terms("body").unwrap();
        assert_eq!(
            format!(
                "{}\t{}\t{}\t{}\n",
                terms.size(),
                terms.sum_doc_freq(),
                terms.sum_total_term_freq(),
                terms.doc_count()
            ),
            fs::read_to_string(root.join("stats.tsv")).unwrap()
        );
        for probe in fs::read_to_string(root.join("seeks.tsv")).unwrap().lines() {
            let cells: Vec<_> = probe.split('\t').collect();
            let mut e = terms.iterator().unwrap();
            let status = match cells[1] {
                "FOUND" => SeekStatus::Found,
                "NOT_FOUND" => SeekStatus::NotFound,
                "END" => SeekStatus::End,
                other => panic!("{other}"),
            };
            assert_eq!(e.try_seek_ceil(&unhex(cells[0])).unwrap(), status);
            assert_eq!(e.term().unwrap(), unhex(cells[2]));
            assert_eq!(
                e.next().unwrap().map(hex),
                (cells[3] != "-").then(|| cells[3].to_string())
            );
        }
        assert!(terms.iterator().unwrap().try_seek_ceil(&[255]).is_err());
        assert!(RamOnlyPostingsFormat::default()
            .fields_producer(consumer.token())
            .unwrap()
            .is_none());
        let mut trailing = consumer.token().to_vec();
        trailing.push(42);
        assert!(format.fields_producer(&trailing).unwrap().is_some());
        for end in 0..consumer.token().len() {
            assert!(format.fields_producer(&consumer.token()[..end]).is_err());
        }
        let mut bad = consumer.token().to_vec();
        bad[0] ^= 1;
        assert!(format.fields_producer(&bad).is_err());
        drop(consumer);
        assert_eq!(manifest(&producer, payloads), expected);
    }
}
