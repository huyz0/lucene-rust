//! Full native read/value snapshot, write and two inherited merges on Java bytes.
use lucene_codecs::doc_values as dv;
use lucene_codecs::field_infos::{self, DocValuesType, FieldInfos};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_test_framework::codecs::cranky_doc_values::*;
use lucene_util::java_random::JavaRandom;
use std::{
    cell::RefCell,
    hint::black_box,
    rc::Rc,
    time::{Duration, Instant},
};
const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
fn data(name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/testframework/data")
            .join(name),
    )
    .unwrap()
}
fn files(prefix: &str) -> Files {
    ["dvm", "dvd", "dvs"]
        .into_iter()
        .map(|extension| (extension.to_owned(), data(&format!("{prefix}.{extension}"))))
        .collect()
}
fn write(infos: &FieldInfos, max_doc: i32) -> SegmentWriteState<'_> {
    SegmentWriteState {
        max_doc,
        segment_id: &ID,
        segment_suffix: "s",
        field_infos: infos,
    }
}
fn read<'a>(infos: &'a FieldInfos, max_doc: i32, files: &'a Files) -> ReadState<'a> {
    ReadState {
        files,
        segment: SegmentReadState {
            max_doc,
            segment_id: &ID,
            segment_suffix: "s",
            field_infos: infos,
        },
    }
}
fn step(d: u64, value: u64) -> u64 {
    d.wrapping_mul(31).wrapping_add(value)
}
fn bytes(mut d: u64, values: &[u8]) -> u64 {
    d = step(d, values.len() as u64);
    for &value in values {
        d = step(d, u64::from(value));
    }
    d
}
fn consume(mut digest: u64, producer: &OwnedDocValuesProducer, infos: &FieldInfos) -> u64 {
    for field in &infos.fields {
        digest = bytes(digest, field.name.as_bytes());
        match field.doc_values_type {
            DocValuesType::None => continue,
            DocValuesType::Numeric => {
                for &(doc, value) in producer.get_numeric(field).unwrap().iter() {
                    digest = step(step(digest, doc as u64), value as u64);
                }
            }
            DocValuesType::Binary => {
                let column = producer.get_binary(field).unwrap();
                let mut offset = 0;
                for (&doc, &len) in column.docs.iter().zip(&column.lengths) {
                    let end = offset + usize::try_from(len).unwrap();
                    digest = bytes(step(digest, doc as u64), &column.bytes[offset..end]);
                    offset = end;
                }
            }
            DocValuesType::Sorted => {
                let column = producer.get_sorted(field).unwrap();
                for term in &column.dict {
                    digest = bytes(digest, term);
                }
                for (&doc, &ord) in column.docs.iter().zip(&column.ords) {
                    digest = step(step(digest, doc as u64), ord as u64);
                }
            }
            DocValuesType::SortedNumeric => {
                let column = producer.get_sorted_numeric(field).unwrap();
                digest = consume_multi(digest, &column.docs, &column.counts, &column.values);
            }
            DocValuesType::SortedSet => {
                let column = producer.get_sorted_set(field).unwrap();
                for term in &column.dict {
                    digest = bytes(digest, term);
                }
                digest = consume_multi(digest, &column.docs, &column.counts, &column.ords);
            }
        }
        if let Some(skip) = producer.get_skipper(field).unwrap() {
            for value in [
                skip.min_value,
                skip.max_value,
                i64::from(skip.doc_count),
                i64::from(skip.max_value_count),
            ] {
                digest = step(digest, value as u64);
            }
            for interval in &skip.intervals {
                for entry in &interval.levels {
                    for value in [
                        i64::from(entry.min_doc_id),
                        i64::from(entry.max_doc_id),
                        entry.min_value,
                        entry.max_value,
                        i64::from(entry.doc_count),
                    ] {
                        digest = step(digest, value as u64);
                    }
                }
            }
        }
    }
    digest
}
fn consume_multi(mut digest: u64, docs: &[i32], counts: &[u32], values: &[i64]) -> u64 {
    let mut offset = 0;
    for (&doc, &count) in docs.iter().zip(counts) {
        digest = step(step(digest, doc as u64), u64::from(count));
        let end = offset + usize::try_from(count).unwrap();
        for &value in &values[offset..end] {
            digest = step(digest, value as u64);
        }
        offset = end;
    }
    digest
}
fn batch(
    case: usize,
    input: &Files,
    infos: &FieldInfos,
    empty: &FieldInfos,
    source: &OwnedDocValuesProducer,
    merges: &[MergeState<'_>; 2],
) -> u64 {
    let mut digest = 0;
    for _ in 0..16 {
        let random = Rc::new(RefCell::new(JavaRandom::new(42)));
        let mut format = CrankyDocValuesFormat::new(
            Lucene90DocValuesFormat(dv::Lucene90DocValuesFormat::new(4).unwrap()),
            random.clone(),
        );
        if case == 0 {
            let mut producer = format.fields_producer(read(infos, 12, input)).unwrap();
            producer.check_integrity().unwrap();
            digest = consume(digest, &producer, infos);
            producer.close().unwrap();
        } else {
            let mut consumer = format
                .fields_consumer(write(
                    if case == 4 { empty } else { infos },
                    if case == 1 {
                        12
                    } else if case == 4 {
                        0
                    } else {
                        20
                    },
                ))
                .unwrap();
            if case == 1 {
                for field in &infos.fields {
                    match field.doc_values_type {
                        DocValuesType::Numeric => consumer.add_numeric_field(field, source),
                        DocValuesType::Binary => consumer.add_binary_field(field, source),
                        DocValuesType::Sorted => consumer.add_sorted_field(field, source),
                        DocValuesType::SortedNumeric => {
                            consumer.add_sorted_numeric_field(field, source)
                        }
                        DocValuesType::SortedSet => consumer.add_sorted_set_field(field, source),
                        DocValuesType::None => panic!(),
                    }
                    .unwrap();
                }
            } else if case != 4 {
                consumer.merge(&merges[case - 2]).unwrap();
            }
            consumer.close().unwrap();
            for extension in ["dvm", "dvd", "dvs"] {
                digest = bytes(digest, &consumer.delegate().output().unwrap()[extension]);
            }
        }
        digest = step(digest, u64::from(random.borrow_mut().next_int() as u32));
    }
    digest
}
fn run(mut batch: impl FnMut() -> u64, millis: u64, expected: u64) -> (f64, u64) {
    let start = Instant::now();
    let mut count = 0;
    loop {
        let actual = batch();
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
    let infos = field_infos::parse(&data("cranky_doc_values.fnm"), &ID, "").unwrap();
    let empty = FieldInfos::new(Vec::new()).unwrap();
    let input = files("cranky_doc_values_left");
    let right_input = files("cranky_doc_values_right");
    let mut native = Lucene90DocValuesFormat(dv::Lucene90DocValuesFormat::new(4).unwrap());
    let left = native.fields_producer(read(&infos, 12, &input)).unwrap();
    let right = native
        .fields_producer(read(&infos, 12, &right_input))
        .unwrap();
    let live: Vec<_> = (0..12).map(|doc| doc != 3 && doc != 10).collect();
    let maps: Vec<Vec<Vec<i32>>> = (0..2)
        .map(|sorted| {
            (0..2)
                .map(|side| {
                    (0..12)
                        .map(|doc| {
                            if doc == 3 || doc == 10 {
                                -1
                            } else {
                                let at = doc - i32::from(doc > 3) - i32::from(doc > 10);
                                if sorted == 1 {
                                    2 * at + side
                                } else {
                                    10 * side + at
                                }
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    let sources: [Vec<_>; 2] = std::array::from_fn(|kind| {
        [&left, &right]
            .into_iter()
            .enumerate()
            .map(|(side, producer)| MergeSource {
                producer: Some(producer),
                field_infos: &infos,
                max_doc: 12,
                doc_map: &maps[kind][side],
                live_docs: Some(&live),
            })
            .collect()
    });
    let merges = std::array::from_fn(|kind| MergeState {
        sources: &sources[kind],
        field_infos: &infos,
        check_aborted: &|| Ok(()),
    });
    println!(
        "cranky_doc_values_format_owned_bytes\t{}",
        std::mem::size_of::<CrankyDocValuesFormat<Lucene90DocValuesFormat>>()
            + std::mem::size_of::<RefCell<JavaRandom>>()
            + 2 * std::mem::size_of::<usize>()
    );
    for (case, name) in [
        "cranky_doc_values_read",
        "cranky_doc_values_write",
        "cranky_doc_values_merge_sequential",
        "cranky_doc_values_merge_sorted",
        "cranky_doc_values_empty_close",
    ]
    .iter()
    .enumerate()
    {
        let expected = batch(case, &input, &infos, &empty, &left, &merges);
        println!("#check\t{name}\t{expected}\t16");
        run(
            || batch(case, &input, &infos, &empty, &left, &merges),
            warmup,
            expected,
        );
        let (nanos, count) = run(
            || batch(case, &input, &infos, &empty, &left, &merges),
            measure,
            expected,
        );
        println!("{name}\t{nanos:.3}\t{count}");
    }
}
