//! `StMergeMicro.java`'s twin: three real shared-terms sources, deletions,
//! field-number remapping, target norms, and six allocated output files.
//! Opening readers and parsing document maps/norms is outside measurement.

use std::hint::black_box;
use std::time::Duration;

use lucene_codecs::field_infos::{self, FieldInfos};
use lucene_codecs::postings::{DocInput, PayInput, PosInput};
use lucene_codecs::postings_writer::FieldNorms;
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_codecs::uniformsplit::sharedterms::{
    StMergeSegment, StUniformSplitPostingsFormat, StUniformSplitTermsReader,
};

const SUFFIX: &str = "SharedTermsUniformSplit_0";

struct Segment {
    id: [u8; 16],
    infos: FieldInfos,
    reader: StUniformSplitTermsReader,
    doc: Vec<u8>,
    pos: Vec<u8>,
    pay: Vec<u8>,
}

fn format() -> StUniformSplitPostingsFormat {
    StUniformSplitPostingsFormat::new(8, 2, None, None, true).unwrap()
}

impl Segment {
    fn open(dir: &str, max_doc: i32) -> Self {
        let name = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .find_map(|n| n.strip_suffix(".si").map(str::to_owned))
            .unwrap();
        let si = std::fs::read(format!("{dir}/{name}.si")).unwrap();
        let at = 4 + 1 + usize::from(si[4]) + 4;
        let id = si[at..at + 16].try_into().unwrap();
        let infos = field_infos::parse(
            &std::fs::read(format!("{dir}/{name}.fnm")).unwrap(),
            &id,
            "",
        )
        .unwrap();
        let file = |ext| std::fs::read(format!("{dir}/{name}_{SUFFIX}.{ext}")).unwrap();
        let reader = format()
            .open(
                file("stustb"),
                &file("stustd"),
                &SegmentReadState {
                    max_doc,
                    segment_id: &id,
                    segment_suffix: SUFFIX,
                    field_infos: &infos,
                },
            )
            .unwrap();
        Self {
            id,
            infos,
            reader,
            doc: file("doc"),
            pos: file("pos"),
            pay: file("pay"),
        }
    }
}

fn rows(path: &str) -> Vec<Vec<i64>> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| line.split('\t').map(|v| v.parse().unwrap()).collect())
        .collect()
}

pub fn bench(warmup: Duration, measure: Duration, root: &str) {
    let segments: Vec<_> = (0..3)
        .map(|i| Segment::open(&format!("{root}/source{i}"), 300))
        .collect();
    let max_doc = std::fs::read_to_string(format!("{root}/max-doc.tsv"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    let target = Segment::open(&format!("{root}/merged"), max_doc);
    let maps: Vec<Vec<i32>> = rows(&format!("{root}/docmaps.tsv"))
        .iter()
        .map(|row| row[1..].iter().map(|&v| v as i32).collect())
        .collect();
    let owned_norms = rows(&format!("{root}/norms.tsv"));
    let norms: Vec<_> = owned_norms
        .iter()
        .map(|row| FieldNorms {
            field_number: row[0] as i32,
            values: &row[1..],
        })
        .collect();
    let docs: Vec<_> = segments
        .iter()
        .map(|s| DocInput::open(&s.doc, &s.id, SUFFIX).unwrap())
        .collect();
    let positions: Vec<_> = segments
        .iter()
        .map(|s| PosInput::open(&s.pos, &s.id, SUFFIX).unwrap())
        .collect();
    let payloads: Vec<_> = segments
        .iter()
        .map(|s| PayInput::open(&s.pay, &s.id, SUFFIX).unwrap())
        .collect();
    let sources: Vec<_> = segments
        .iter()
        .enumerate()
        .map(|(i, s)| StMergeSegment {
            reader: Some(&s.reader),
            field_infos: &s.infos,
            doc_map: &maps[i],
            doc: &docs[i],
            pos: Some(&positions[i]),
            pay: Some(&payloads[i]),
        })
        .collect();
    let state = SegmentWriteState {
        max_doc,
        segment_id: &target.id,
        segment_suffix: SUFFIX,
        field_infos: &target.infos,
    };
    let format = format();
    let run = || {
        format
            .merge(&sources, &norms, &state, false, || {
                panic!("unexpected generic merge")
            })
            .unwrap()
    };
    let output = run();
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for bytes in [
        &output.ustb,
        &output.ustd,
        &output.postings.doc,
        &output.postings.pos,
        &output.postings.pay,
        &output.postings.psm,
    ] {
        hash = (hash ^ bytes.len() as u64).wrapping_mul(0x0000_0100_0000_01b3);
        for &b in bytes {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    println!("#check\tcx_stsplit_merge\t{hash:016x}\t1");
    super::measure("cx_stsplit_merge", warmup, measure, || {
        black_box(run());
        1
    });
}
