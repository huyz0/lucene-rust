//! Actual cranky wrapper around the buffer-backed Lucene104 postings codec.
use lucene_codecs::field_infos::{self, FieldInfos};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_test_framework::codecs::cranky_postings::*;
use lucene_util::java_random::JavaRandom;
use std::{
    cell::RefCell,
    hint::black_box,
    rc::Rc,
    time::{Duration, Instant},
};
const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
fn step(d: u64, v: u64) -> u64 {
    d.wrapping_mul(31).wrapping_add(v)
}
fn bytes(mut d: u64, b: &[u8]) -> u64 {
    d = step(d, b.len() as u64);
    for &v in b {
        d = step(d, u64::from(v));
    }
    d
}
fn read(mut d: u64, reader: &impl FieldsProducer) -> u64 {
    reader.check_integrity().unwrap();
    for field in reader.materialize().unwrap().iter() {
        d = bytes(d, field.name.as_bytes());
        for term in &field.terms {
            d = bytes(d, &term.bytes);
            for posting in &term.postings {
                d = step(d, posting.doc as u64);
                d = step(d, posting.freq as u64);
                for p in &posting.positions {
                    d = step(d, p.position as u64);
                    d = step(d, p.start_offset as u64);
                    d = step(d, p.end_offset as u64);
                    d = bytes(d, &p.payload);
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
    case: usize,
) -> u64 {
    let mut d = 0;
    for _ in 0..16 {
        let random = Rc::new(RefCell::new(JavaRandom::new(42)));
        let mut format = CrankyPostingsFormat::new(Lucene104PostingsFormat, random.clone());
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
        } else {
            let mut consumer = format
                .fields_consumer(state(infos, if case == 1 { 9 } else { 16 }))
                .unwrap();
            if case == 1 {
                consumer.write(source, Some(&norms[0])).unwrap();
            } else {
                consumer
                    .merge(&merges[case - 2], Some(&norms[case - 1]))
                    .unwrap();
            }

            consumer.close().unwrap();
            let output = consumer.delegate().output().unwrap();
            for b in [
                &output.doc,
                &output.psm,
                &output.pos,
                &output.pay,
                &output.tim,
                &output.tip,
                &output.tmd,
            ] {
                d = bytes(d, b);
            }
        }
        d = step(d, u64::from(random.borrow_mut().next_int() as u32));
    }
    d
}
fn run(mut f: impl FnMut() -> u64, millis: u64, expected: u64) -> (f64, u64) {
    let start = Instant::now();
    let mut count = 0;
    loop {
        let actual = f();
        assert_eq!(actual, expected);
        black_box(actual);
        count += 16;
        if start.elapsed() >= Duration::from_millis(millis) {
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
                std::fs::read(root.join(format!("cranky_postings.{ext}"))).unwrap(),
            )
        })
        .collect();
    let infos = field_infos::parse(
        &std::fs::read(root.join("cranky_postings.fnm")).unwrap(),
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
    let norms = std::array::from_fn(|case| {
        let mut norms = OwnedNorms::default();
        let mut values = vec![None; if case == 0 { 9 } else { 16 }];
        for side in 0..if case == 0 { 1 } else { 2 } {
            for doc in 0..9 {
                if case > 0 && doc == 1 {
                    continue;
                }
                let at = if doc == 0 { 0 } else { doc - 1 };
                let dest = if case == 0 {
                    doc
                } else if case == 1 {
                    8 * side + at
                } else {
                    2 * at + side
                };
                values[dest] = Some(if doc == 1 { 4 } else { 3 });
            }
        }
        norms.0.insert("rich".into(), values);
        norms
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
    let sources: [[MergeSource<'_>; 2]; 2] = std::array::from_fn(|kind| {
        std::array::from_fn(|side| MergeSource {
            reader: Some(&source),
            max_doc: 9,
            doc_map: &maps[kind][side],
        })
    });
    let check = || Ok(());
    let merges = std::array::from_fn(|kind| MergeState {
        sources: &sources[kind],
        check_aborted: &check,
    });
    let owned = std::mem::size_of::<CrankyPostingsFormat<Lucene104PostingsFormat>>()
        + std::mem::size_of::<RefCell<JavaRandom>>()
        + 2 * std::mem::size_of::<usize>();
    println!("cranky_postings_format_owned_bytes\t{owned}");
    for (case, name) in ["read", "write", "merge_sequential", "merge_sorted"]
        .iter()
        .enumerate()
    {
        let expected = batch(&files, &infos, &source, &norms, &merges, case);
        println!("#check\tcranky_postings_{name}\t{expected}\t16");
        run(
            || batch(&files, &infos, &source, &norms, &merges, case),
            warmup,
            expected,
        );
        let (ns, count) = run(
            || batch(&files, &infos, &source, &norms, &merges, case),
            measure,
            expected,
        );
        println!("cranky_postings_{name}\t{ns:.3}\t{count}");
    }
}
