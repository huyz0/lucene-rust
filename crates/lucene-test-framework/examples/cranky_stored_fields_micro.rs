//! Real cranky stored-fields lifecycle, paired with CrankyStoredFieldsMicro.java.
use lucene_codecs::stored_fields::{self, FieldValue, Mode, StoredFieldVisitor, VisitStatus};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::{ByteBuffersDirectory, Directory};
use lucene_test_framework::codecs::cranky_stored_fields::*;
use lucene_util::java_random::JavaRandom;
use std::cell::RefCell;
use std::hint::black_box;
use std::rc::Rc;
use std::time::{Duration, Instant};
const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const BATCH: u64 = 32;
const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
fn hash(digest: u64, value: u64) -> u64 {
    (digest ^ value).wrapping_mul(0x100_0000_01b3)
}
fn bytes(mut digest: u64, value: &[u8]) -> u64 {
    digest = hash(digest, value.len() as u64);
    for &byte in value {
        digest = hash(digest, u64::from(byte));
    }
    digest
}
struct Visitor {
    digest: u64,
}
impl Visitor {
    fn field(&mut self, number: i32, kind: u64) {
        self.digest = hash(hash(self.digest, number as u64), kind);
    }
}
impl StoredFieldVisitor for Visitor {
    fn needs_field(&mut self, _: i32) -> stored_fields::Result<VisitStatus> {
        Ok(VisitStatus::Yes)
    }
    fn int_field(&mut self, number: i32, value: i32) -> stored_fields::Result<()> {
        self.field(number, 1);
        self.digest = hash(self.digest, value as u64);
        Ok(())
    }
    fn long_field(&mut self, number: i32, value: i64) -> stored_fields::Result<()> {
        self.field(number, 2);
        self.digest = hash(self.digest, value as u64);
        Ok(())
    }
    fn float_field(&mut self, number: i32, value: f32) -> stored_fields::Result<()> {
        self.field(number, 3);
        self.digest = hash(self.digest, u64::from(value.to_bits()));
        Ok(())
    }
    fn double_field(&mut self, number: i32, value: f64) -> stored_fields::Result<()> {
        self.field(number, 4);
        self.digest = hash(self.digest, value.to_bits());
        Ok(())
    }
    fn binary_field(&mut self, number: i32, value: &[u8]) -> stored_fields::Result<()> {
        self.field(number, 5);
        self.digest = bytes(self.digest, value);
        Ok(())
    }
    fn string_field(&mut self, number: i32, value: &str) -> stored_fields::Result<()> {
        self.field(number, 6);
        self.digest = bytes(self.digest, value.as_bytes());
        Ok(())
    }
}
fn state(directory: &dyn Directory, mode: Mode) -> StoredFieldsState<'_> {
    StoredFieldsState {
        directory,
        segment: "_real",
        id: &ID,
        suffix: "",
        mode,
    }
}
fn consume(mut digest: u64, directory: &dyn Directory) -> u64 {
    for name in ["_real.fdt", "_real.fdx", "_real.fdm"] {
        let data = directory.open(name).unwrap();
        digest = bytes(digest, &data);
        directory.delete_file(name).unwrap();
    }
    assert!(directory.list_all().unwrap().is_empty());
    digest
}
fn measure(name: &str, warmup: u64, duration: u64, mut batch: impl FnMut() -> u64) {
    let expected = batch();
    for (phase, budget) in [warmup, duration].into_iter().enumerate() {
        let start = Instant::now();
        let mut operations = 0;
        loop {
            assert_eq!(black_box(batch()), expected);
            operations += BATCH;
            if start.elapsed() >= Duration::from_millis(budget) {
                break;
            }
        }
        if phase == 1 {
            println!(
                "{name}\t{:.3}\t{operations}",
                start.elapsed().as_nanos() as f64 / operations as f64
            );
        }
    }
    println!("#check\t{name}\t{expected}\t{BATCH}");
}
fn first(bound: i32) -> i64 {
    (0..)
        .find(|&seed| JavaRandom::new(seed).next_int_bounded(bound) == 0)
        .unwrap()
}
struct CountWriter {
    closed: u64,
}
impl StoredFieldsWriter for CountWriter {
    fn start_document(&mut self) -> stored_fields::Result<()> {
        panic!("start guard bypassed")
    }
    fn finish_document(&mut self) -> stored_fields::Result<()> {
        panic!("finish-document guard bypassed")
    }
    fn write_field(&mut self, _: i32, _: &FieldValue) -> stored_fields::Result<()> {
        panic!("field guard bypassed")
    }
    fn write_field_data_input(
        &mut self,
        _: i32,
        _: &mut dyn DataInput,
        _: usize,
    ) -> stored_fields::Result<()> {
        panic!("input guard bypassed")
    }
    fn finish(&mut self, _: i32) -> stored_fields::Result<()> {
        panic!("finish guard bypassed")
    }
    fn close(&mut self) -> stored_fields::Result<()> {
        self.closed += 1;
        Ok(())
    }
    fn ram_bytes_used(&self) -> usize {
        0
    }
    fn child_resources(&self) -> &[ChildResource] {
        &[]
    }
    fn merge(&mut self, _: &MergeState<'_>) -> stored_fields::Result<i32> {
        panic!("delegate merge called")
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup = args.get(1).map_or(500, |value| value.parse().unwrap());
    let duration = args.get(2).map_or(1000, |value| value.parse().unwrap());
    let random = Rc::new(RefCell::new(JavaRandom::new(0)));
    let mut format = CrankyStoredFieldsFormat::new(Lucene90StoredFieldsFormat, random.clone());
    let owned = std::mem::size_of_val(&format)
        + 2 * std::mem::size_of::<usize>()
        + std::mem::size_of::<RefCell<JavaRandom>>();
    println!("cranky_stored_fields_owned_bytes\t{owned}");
    let directory = ByteBuffersDirectory::new();
    let values: Vec<Vec<FieldValue>> = (0..3)
        .map(|doc| {
            vec![
                FieldValue::Int(-3 + doc),
                FieldValue::Long(i64::MIN + i64::from(doc)),
                FieldValue::Float(-0.0),
                FieldValue::Double(f64::from_bits(0x7ff8000000000042)),
                FieldValue::Binary(vec![0, 1, 255]),
                FieldValue::Binary(vec![9, 8]),
                FieldValue::String(format!("hello λ {doc}")),
            ]
        })
        .collect();
    let inputs = [
        (
            "speed",
            Mode::BestSpeed,
            include_bytes!("../../../fixtures/testframework/data/cranky_stored_fields_speed.fdt")
                .as_slice(),
            include_bytes!("../../../fixtures/testframework/data/cranky_stored_fields_speed.fdx")
                .as_slice(),
            include_bytes!("../../../fixtures/testframework/data/cranky_stored_fields_speed.fdm")
                .as_slice(),
        ),
        (
            "compression",
            Mode::BestCompression,
            include_bytes!(
                "../../../fixtures/testframework/data/cranky_stored_fields_compression.fdt"
            )
            .as_slice(),
            include_bytes!(
                "../../../fixtures/testframework/data/cranky_stored_fields_compression.fdx"
            )
            .as_slice(),
            include_bytes!(
                "../../../fixtures/testframework/data/cranky_stored_fields_compression.fdm"
            )
            .as_slice(),
        ),
    ];
    for (label, mode, fdt, fdx, fdm) in inputs {
        measure(
            &format!("cranky_stored_fields_read_{label}"),
            warmup,
            duration,
            || {
                random.borrow_mut().set_seed(0);
                let mut visitor = Visitor { digest: OFFSET };
                for _ in 0..BATCH {
                    let reader = format
                        .fields_reader(StoredFieldsReadState {
                            fdt,
                            fdx,
                            fdm,
                            id: &ID,
                            suffix: "",
                        })
                        .unwrap();
                    reader.check_integrity().unwrap();
                    visitor.digest = hash(visitor.digest, reader.max_doc() as u64);
                    for doc in 0..reader.max_doc() {
                        reader.visit_document(doc, &mut visitor).unwrap();
                    }
                }
                hash(visitor.digest, random.borrow_mut().next_int() as u64)
            },
        );
        measure(
            &format!("cranky_stored_fields_write_{label}"),
            warmup,
            duration,
            || {
                let mut digest = OFFSET;
                for _ in 0..BATCH {
                    random.borrow_mut().set_seed(0);
                    let mut writer = format.fields_writer(state(&directory, mode)).unwrap();
                    for document in &values {
                        writer.start_document().unwrap();
                        for (number, value) in document.iter().enumerate() {
                            if number == 5 {
                                writer
                                    .write_field_data_input(5, &mut SliceInput::new(&[9, 8]), 2)
                                    .unwrap();
                            } else {
                                writer.write_field(number as i32, value).unwrap();
                            }
                        }
                        writer.finish_document().unwrap();
                    }
                    writer.finish(3).unwrap();
                    writer.close().unwrap();
                    digest = consume(digest, &directory);
                    digest = hash(digest, random.borrow_mut().next_int() as u64);
                }
                digest
            },
        );
    }
    let (_, _, fdt, fdx, fdm) = inputs[0];
    let reader = stored_fields::open(fdt, fdx, fdm, &ID, "").unwrap();
    let remap = [(0, 6), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (6, 0)];
    for sorted in [false, true] {
        let maps = if sorted {
            [[0, -1, 2], [1, 3, -1]]
        } else {
            [[0, -1, 1], [2, 3, -1]]
        };
        let sources = [
            MergeSource {
                reader: &reader,
                doc_map: &maps[0],
                field_map: &remap,
            },
            MergeSource {
                reader: &reader,
                doc_map: &maps[1],
                field_map: &remap,
            },
        ];
        let merge = MergeState {
            sources: &sources,
            needs_index_sort: sorted,
            check_aborted: &|| Ok(()),
        };
        measure(
            if sorted {
                "cranky_stored_fields_merge_sorted"
            } else {
                "cranky_stored_fields_merge_sequential"
            },
            warmup,
            duration,
            || {
                let mut digest = OFFSET;
                for _ in 0..BATCH {
                    random.borrow_mut().set_seed(0);
                    let mut writer = format
                        .fields_writer(state(&directory, Mode::BestSpeed))
                        .unwrap();
                    digest = hash(digest, writer.merge(&merge).unwrap() as u64);
                    writer.close().unwrap();
                    digest = consume(digest, &directory);
                    digest = hash(digest, random.borrow_mut().next_int() as u64);
                }
                digest
            },
        );
    }
    let seeds = [first(100), first(1000), first(10000)];
    let mut writer = CrankyStoredFieldsWriter::new(CountWriter { closed: 0 }, random.clone());
    let merge = MergeState {
        sources: &[],
        needs_index_sort: false,
        check_aborted: &|| Ok(()),
    };
    measure("cranky_stored_fields_faults", warmup, duration, || {
        writer.delegate_mut().closed = 0;
        let mut digest = OFFSET;
        for _ in 0..BATCH {
            for op in 0..6 {
                random.borrow_mut().set_seed(
                    seeds[match op {
                        1 => 1,
                        2 | 3 => 2,
                        _ => 0,
                    }],
                );
                let mut input = SliceInput::new(&[9, 8]);
                let (result, expected) = match op {
                    0 => (
                        format
                            .fields_writer(state(&directory, Mode::BestSpeed))
                            .map(|_| ()),
                        FAKE_OPEN_ERROR,
                    ),
                    1 => (writer.close(), FAKE_CLOSE_ERROR),
                    2 => (writer.start_document(), FAKE_START_ERROR),
                    3 => (
                        writer.write_field_data_input(0, &mut input, 2),
                        FAKE_FIELD_ERROR,
                    ),
                    4 => (writer.finish(0), FAKE_FINISH_ERROR),
                    _ => (writer.merge(&merge).map(|_| ()), FAKE_MERGE_ERROR),
                };
                let message = result.unwrap_err().to_string();
                assert_eq!(message, expected);
                assert_eq!(input.position(), 0);
                digest = bytes(digest, message.as_bytes());
                digest = hash(digest, random.borrow_mut().next_int() as u64);
            }
        }
        assert_eq!(writer.delegate().closed, BATCH);
        assert!(directory.list_all().unwrap().is_empty());
        hash(digest, writer.delegate().closed)
    });
}
