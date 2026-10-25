//! Complete asserting postings validation and native seven-file adapter workloads.
use lucene_codecs::field_infos::{self, FieldInfos};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_search::reader::{PostingsFlags, NO_MORE_DOCS};
use lucene_test_framework::codecs::{asserting_postings::*, cranky_postings as byte};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};
const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
struct Sink;
impl FieldsConsumer for Sink {
    fn write(&mut self, _: &dyn WriteFields, _: Option<&dyn NormsProducer>) -> Result<()> {
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
fn step(d: u64, v: u64) -> u64 {
    d.wrapping_mul(31).wrapping_add(v)
}
fn bytes(mut d: u64, value: &[u8]) -> u64 {
    d = step(d, value.len() as u64);
    for &v in value {
        d = step(d, u64::from(v))
    }
    d
}
fn read(mut d: u64, reader: &impl FieldsProducer) -> u64 {
    reader.check_integrity().unwrap();
    d = step(d, reader.size() as u64);
    for name in reader.field_names().unwrap() {
        d = bytes(d, name.as_bytes());
        let terms = reader.terms(&name).unwrap().unwrap();
        for value in [
            terms.size(),
            terms.sum_doc_freq(),
            terms.sum_total_term_freq(),
            i64::from(terms.doc_count()),
            i64::from(terms.has_freqs()),
            i64::from(terms.has_positions()),
            i64::from(terms.has_offsets()),
            i64::from(terms.has_payloads()),
        ] {
            d = step(d, value as u64)
        }
        let mut e = terms.iterator().unwrap();
        while let Some(term) = e.next().unwrap() {
            d = bytes(d, term);
            let mut p = e.postings(PostingsFlags::All).unwrap();
            loop {
                let doc = p.next_doc().unwrap();
                if doc == NO_MORE_DOCS {
                    break;
                }
                d = step(d, doc as u64);
                let freq = p.freq();
                d = step(d, freq as u64);
                if terms.has_positions() {
                    for _ in 0..freq {
                        d = step(d, p.next_position().unwrap() as u64);
                        d = step(d, p.start_offset() as u64);
                        d = step(d, p.end_offset() as u64);
                        d = bytes(d, p.payload().unwrap_or_default())
                    }
                }
            }
        }
    }
    d
}
fn state(infos: &FieldInfos, max_doc: i32) -> SegmentWriteState<'_> {
    SegmentWriteState {
        max_doc,
        segment_id: &ID,
        segment_suffix: "s",
        field_infos: infos,
    }
}
fn batch(
    files: &Files,
    infos: &FieldInfos,
    source: &Lucene104FieldsProducer,
    norms: &[OwnedNorms; 3],
    merges: &[MergeState<'_>; 2],
    consumer: &mut AssertingFieldsConsumer<'_, Sink>,
    case: usize,
) -> u64 {
    let mut d = 0;
    for _ in 0..16 {
        let mut format = AssertingPostingsFormat::new(Lucene104PostingsFormat);
        if case == 0 {
            let mut reader = format
                .fields_producer(ReadState {
                    files,
                    segment: SegmentReadState {
                        max_doc: 9,
                        segment_id: &ID,
                        segment_suffix: "s",
                        field_infos: infos,
                    },
                })
                .unwrap();
            d = read(d, &reader);
            reader.close().unwrap();
        } else if case == 4 {
            consumer
                .write(&NativeFieldsView::new(source, infos), Some(&norms[0]))
                .unwrap();
            consumer.close().unwrap();
            d = step(d, 1);
        } else {
            let mut writer = format
                .fields_consumer(state(infos, if case == 1 { 9 } else { 16 }))
                .unwrap();
            if case == 1 {
                writer
                    .write(&NativeFieldsView::new(source, infos), Some(&norms[0]))
                    .unwrap()
            } else {
                writer
                    .merge(&merges[case - 2], Some(&norms[case - 1]))
                    .unwrap()
            }
            writer.close().unwrap();
            let output = writer.delegate().output().unwrap();
            for b in [
                &output.doc,
                &output.psm,
                &output.pos,
                &output.pay,
                &output.tim,
                &output.tip,
                &output.tmd,
            ] {
                d = bytes(d, b)
            }
        }
    }
    d
}
fn run(mut f: impl FnMut() -> u64, ms: u64, expected: u64) -> (f64, u64) {
    let start = Instant::now();
    let mut count = 0;
    loop {
        let actual = f();
        assert_eq!(actual, expected);
        black_box(actual);
        count += 16;
        if start.elapsed() >= Duration::from_millis(ms) {
            break;
        }
    }
    (start.elapsed().as_nanos() as f64 / count as f64, count)
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup = args[1].parse().unwrap();
    let measure = args[2].parse().unwrap();
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/testframework/data");
    let files: Files = ["doc", "psm", "pos", "pay", "tim", "tip", "tmd"]
        .into_iter()
        .map(|ext| {
            (
                ext.into(),
                std::fs::read(root.join(format!("asserting_postings.{ext}"))).unwrap(),
            )
        })
        .collect();
    let infos = field_infos::parse(
        &std::fs::read(root.join("asserting_postings.fnm")).unwrap(),
        &ID,
        "",
    )
    .unwrap();
    let source = Lucene104PostingsFormat
        .fields_producer(ReadState {
            files: &files,
            segment: SegmentReadState {
                max_doc: 9,
                segment_id: &ID,
                segment_suffix: "s",
                field_infos: &infos,
            },
        })
        .unwrap();
    let norms = std::array::from_fn(|kind| {
        let max_doc = if kind == 0 { 9 } else { 16 };
        let manifest = std::fs::read_to_string(root.join(match kind {
            0 => "asserting_postings.tsv",
            1 => "asserting_postings_merge_sequential.tsv",
            _ => "asserting_postings_merge_sorted.tsv",
        }))
        .unwrap();
        let mut n = OwnedNorms::default();
        for line in manifest.lines() {
            let row: Vec<_> = line.split('\t').collect();
            if row[0] == "norm" {
                n.0.entry(row[1].into())
                    .or_insert_with(|| vec![None; max_doc])[row[2].parse::<usize>().unwrap()] =
                    Some(row[3].parse().unwrap())
            }
        }
        n
    });
    let maps: [[Vec<i32>; 2]; 2] = std::array::from_fn(|kind| {
        std::array::from_fn(|side| {
            (0..9)
                .map(|doc| {
                    if doc == 1 {
                        -1
                    } else {
                        let at = if doc == 0 { 0 } else { doc - 1 };
                        if kind == 1 {
                            2 * at + side as i32
                        } else {
                            8 * side as i32 + at
                        }
                    }
                })
                .collect()
        })
    });
    let sources: [[byte::MergeSource<'_>; 2]; 2] = std::array::from_fn(|kind| {
        std::array::from_fn(|side| byte::MergeSource {
            reader: Some(&source),
            max_doc: 9,
            doc_map: &maps[kind][side],
        })
    });
    let check = || Ok(());
    let merges = std::array::from_fn(|kind| MergeState {
        bytes: byte::MergeState {
            sources: &sources[kind],
            check_aborted: &check,
        },
        field_infos: &infos,
    });
    println!(
        "asserting_postings_validation_shallow_bytes\t{}",
        std::mem::size_of::<AssertingFieldsConsumer<'_, Sink>>() + std::mem::size_of::<Sink>()
    );
    fn rc_headers<T>() -> usize {
        let alignment = std::mem::align_of::<T>().max(std::mem::align_of::<usize>());
        (2 * std::mem::size_of::<usize>() + std::mem::size_of::<T>()).div_ceil(alignment)
            * alignment
    }
    let common = rc_headers::<FieldInfos>() + rc_headers::<std::cell::Cell<bool>>();
    let current = std::mem::size_of::<Lucene104FieldsProducer>()
        + rc_headers::<Vec<byte::FieldPostings>>()
        + rc_headers::<(FieldInfos, std::cell::RefCell<Files>, std::cell::Cell<bool>)>();
    let previous = 3 * std::mem::size_of::<usize>()
        + rc_headers::<std::cell::RefCell<byte::Lucene104FieldsProducer>>()
        + common;
    println!("#native-reader-headers\tcurrent={current}\tprevious={previous}\tretained_decoded_copies=1\tacquisition_transient_decoded_copies=2");
    println!("#native-reader-headers excludes referenced FieldInfos allocations, field/term/posting/position/payload allocations, file buffer allocations and allocator overhead; those retained payloads are unchanged.");
    let mut consumer = AssertingFieldsConsumer::new(Sink, &infos);
    for (case, name) in [
        "read",
        "write",
        "merge_sequential",
        "merge_sorted",
        "validation",
    ]
    .into_iter()
    .enumerate()
    {
        let expected = batch(
            &files,
            &infos,
            &source,
            &norms,
            &merges,
            &mut consumer,
            case,
        );
        println!("#check\tasserting_postings_{name}\t{expected}\t16");
        run(
            || {
                batch(
                    &files,
                    &infos,
                    &source,
                    &norms,
                    &merges,
                    &mut consumer,
                    case,
                )
            },
            warmup,
            expected,
        );
        let (ns, count) = run(
            || {
                batch(
                    &files,
                    &infos,
                    &source,
                    &norms,
                    &merges,
                    &mut consumer,
                    case,
                )
            },
            measure,
            expected,
        );
        println!("asserting_postings_{name}\t{ns:.3}\t{count}")
    }
}
