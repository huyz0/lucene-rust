//! Asserting stored-fields baseline paired with AssertingStoredFieldsMicro.java.
use lucene_codecs::stored_fields::{self, FieldValue, Mode, StoredFieldVisitor, VisitStatus};
use lucene_store::{ByteBuffersDirectory, DataInput, Directory, SliceInput};
use lucene_test_framework::codecs::asserting_stored_fields::*;
use lucene_test_framework::codecs::cranky_stored_fields::{MergeSource, MergeState};
use std::hint::black_box;
use std::time::{Duration, Instant};
const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const BATCH: u64 = 32;
const OFFSET: u64 = 0xcbf29ce484222325;
fn hash(d: u64, v: u64) -> u64 {
    (d ^ v).wrapping_mul(0x100000001b3)
}
fn bytes(mut d: u64, b: &[u8]) -> u64 {
    d = hash(d, b.len() as u64);
    for &v in b {
        d = hash(d, u64::from(v));
    }
    d
}
struct Digest {
    value: u64,
}
impl Digest {
    fn new() -> Self {
        Self { value: OFFSET }
    }
    fn field(&mut self, n: i32, kind: u64) {
        self.value = hash(hash(self.value, n as u64), kind);
    }
    fn mix(&mut self, n: u64) {
        self.value = hash(self.value, n);
    }
}
impl StoredFieldVisitor for Digest {
    fn needs_field(&mut self, _: i32) -> stored_fields::Result<VisitStatus> {
        Ok(VisitStatus::Yes)
    }
    fn int_field(&mut self, n: i32, v: i32) -> stored_fields::Result<()> {
        self.field(n, 1);
        self.mix(v as u64);
        Ok(())
    }
    fn long_field(&mut self, n: i32, v: i64) -> stored_fields::Result<()> {
        self.field(n, 2);
        self.mix(v as u64);
        Ok(())
    }
    fn float_field(&mut self, n: i32, v: f32) -> stored_fields::Result<()> {
        self.field(n, 3);
        self.mix(u64::from(v.to_bits()));
        Ok(())
    }
    fn double_field(&mut self, n: i32, v: f64) -> stored_fields::Result<()> {
        self.field(n, 4);
        self.mix(v.to_bits());
        Ok(())
    }
    fn binary_field(&mut self, n: i32, v: &[u8]) -> stored_fields::Result<()> {
        self.field(n, 5);
        self.value = bytes(self.value, v);
        Ok(())
    }
    fn string_field(&mut self, n: i32, v: &str) -> stored_fields::Result<()> {
        self.field(n, 6);
        self.value = bytes(self.value, v.as_bytes());
        Ok(())
    }
}
impl StoredFieldsWriter for Digest {
    fn start_document(&mut self) -> stored_fields::Result<()> {
        self.mix(10);
        Ok(())
    }
    fn finish_document(&mut self) -> stored_fields::Result<()> {
        self.mix(11);
        Ok(())
    }
    fn write_field(&mut self, n: i32, v: &FieldValue) -> stored_fields::Result<()> {
        match v {
            FieldValue::Int(v) => self.int_field(n, *v),
            FieldValue::Long(v) => self.long_field(n, *v),
            FieldValue::Float(v) => self.float_field(n, *v),
            FieldValue::Double(v) => self.double_field(n, *v),
            FieldValue::Binary(v) => self.binary_field(n, v),
            FieldValue::String(v) => self.string_field(n, v),
        }
    }
    fn write_field_data_input(
        &mut self,
        n: i32,
        input: &mut dyn DataInput,
        length: usize,
    ) -> stored_fields::Result<()> {
        assert_eq!(length, 2);
        let mut value = [0; 2];
        input.read_bytes(&mut value)?;
        self.binary_field(n, &value)
    }
    fn finish(&mut self, n: i32) -> stored_fields::Result<()> {
        self.mix(12);
        self.mix(n as u64);
        Ok(())
    }
    fn close(&mut self) -> stored_fields::Result<()> {
        self.mix(13);
        Ok(())
    }
    fn ram_bytes_used(&self) -> usize {
        0
    }
    fn child_resources(&self) -> &[ChildResource] {
        &[]
    }
}
struct CountingFormat;
impl StoredFieldsFormat for CountingFormat {
    type Writer<'a> = Digest;
    type Reader<'a> = Lucene90StoredFieldsReader<'a>;
    fn fields_writer<'a>(
        &mut self,
        _: StoredFieldsState<'a>,
    ) -> stored_fields::Result<Self::Writer<'a>> {
        Ok(Digest::new())
    }
    fn fields_reader<'a>(
        &mut self,
        _: StoredFieldsReadState<'a>,
    ) -> stored_fields::Result<Self::Reader<'a>> {
        panic!("validation format reader")
    }
}
fn write_documents(writer: &mut impl StoredFieldsWriter, values: &[FieldValue]) {
    writer.start_document().unwrap();
    for (index, value) in values.iter().enumerate() {
        if index == 5 {
            let mut input = SliceInput::new(&[9, 8]);
            writer.write_field_data_input(7, &mut input, 2).unwrap();
            assert_eq!(input.position(), 2);
        } else {
            writer.write_field(7, value).unwrap();
        }
    }
    writer.finish_document().unwrap();
    writer.start_document().unwrap();
    writer.finish_document().unwrap();
    writer.finish(2).unwrap();
    writer.close().unwrap();
}
fn consume(mut digest: u64, directory: &dyn Directory) -> u64 {
    for ext in ["fdt", "fdx", "fdm"] {
        let name = format!("_asf.{ext}");
        digest = bytes(digest, &directory.open(&name).unwrap());
        directory.delete_file(&name).unwrap();
    }
    assert!(directory.list_all().unwrap().is_empty());
    digest
}
fn state(directory: &dyn Directory, mode: Mode) -> StoredFieldsState<'_> {
    StoredFieldsState {
        directory,
        segment: "_asf",
        id: &ID,
        suffix: "",
        mode,
    }
}
fn measure(name: &str, warmup: u64, duration: u64, mut batch: impl FnMut() -> u64) {
    let expected = batch();
    for (phase, millis) in [warmup, duration].into_iter().enumerate() {
        let start = Instant::now();
        let mut operations = 0;
        loop {
            assert_eq!(black_box(batch()), expected, "timed batch digest changed");
            operations += BATCH;
            if start.elapsed() >= Duration::from_millis(millis) {
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
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup = args.get(1).map_or(500, |v| v.parse().unwrap());
    let duration = args.get(2).map_or(1000, |v| v.parse().unwrap());
    let values = vec![
        FieldValue::Int(-17),
        FieldValue::Long(i64::MIN),
        FieldValue::Float(-0.0),
        FieldValue::Double(f64::from_bits(0x7ff8000000000042)),
        FieldValue::Binary(vec![0, 1, 255]),
        FieldValue::Binary(vec![9, 8]),
        FieldValue::String("hello λ".into()),
    ];
    let writer = AssertingStoredFieldsWriter::new(Digest::new());
    println!(
        "asserting_stored_fields_writer_overhead_bytes\t{}",
        std::mem::size_of_val(&writer) - std::mem::size_of::<Digest>()
    );
    let mut format = AssertingStoredFieldsFormat::new(Lucene90StoredFieldsFormat);
    println!(
        "asserting_stored_fields_format_owned_bytes\t{}",
        std::mem::size_of_val(&format)
    );
    let validation_directory = ByteBuffersDirectory::new();
    let mut validation_format = AssertingStoredFieldsFormat::new(CountingFormat);
    measure(
        "asserting_stored_fields_validation",
        warmup,
        duration,
        || {
            let mut digest = OFFSET;
            for _ in 0..BATCH {
                let mut writer = validation_format
                    .fields_writer(state(&validation_directory, Mode::BestSpeed))
                    .unwrap();
                write_documents(&mut writer, &values);
                digest = hash(digest, writer.delegate().value);
            }
            digest
        },
    );
    let inputs = [
        (
            "speed",
            Mode::BestSpeed,
            include_bytes!(
                "../../../fixtures/testframework/data/asserting_stored_fields_speed.fdt"
            )
            .as_slice(),
            include_bytes!(
                "../../../fixtures/testframework/data/asserting_stored_fields_speed.fdx"
            )
            .as_slice(),
            include_bytes!(
                "../../../fixtures/testframework/data/asserting_stored_fields_speed.fdm"
            )
            .as_slice(),
        ),
        (
            "compression",
            Mode::BestCompression,
            include_bytes!(
                "../../../fixtures/testframework/data/asserting_stored_fields_compression.fdt"
            )
            .as_slice(),
            include_bytes!(
                "../../../fixtures/testframework/data/asserting_stored_fields_compression.fdx"
            )
            .as_slice(),
            include_bytes!(
                "../../../fixtures/testframework/data/asserting_stored_fields_compression.fdm"
            )
            .as_slice(),
        ),
    ];
    for (label, mode, fdt, fdx, fdm) in inputs {
        measure(
            &format!("asserting_stored_fields_read_{label}"),
            warmup,
            duration,
            || {
                let mut visitor = Digest::new();
                for _ in 0..BATCH {
                    let mut reader = format
                        .fields_reader(StoredFieldsReadState {
                            fdt,
                            fdx,
                            fdm,
                            id: &ID,
                            suffix: "",
                        })
                        .unwrap();
                    reader.check_integrity().unwrap();
                    visitor.mix(reader.max_doc() as u64);
                    for doc in 0..reader.max_doc() {
                        visitor.mix(doc as u64);
                        reader.visit_document(doc, &mut visitor).unwrap();
                    }
                    reader.close().unwrap();
                }
                visitor.value
            },
        );
        measure(
            &format!("asserting_stored_fields_write_{label}"),
            warmup,
            duration,
            || {
                let mut digest = OFFSET;
                for _ in 0..BATCH {
                    let directory = ByteBuffersDirectory::new();
                    let mut writer = format.fields_writer(state(&directory, mode)).unwrap();
                    write_documents(&mut writer, &values);
                    digest = consume(digest, &directory);
                }
                digest
            },
        );
    }
    let (_, _, fdt, fdx, fdm) = inputs[0];
    let reader = format
        .fields_reader(StoredFieldsReadState {
            fdt,
            fdx,
            fdm,
            id: &ID,
            suffix: "",
        })
        .unwrap();
    for sorted in [false, true] {
        let maps = if sorted {
            [[1, -1], [0, 2]]
        } else {
            [[0, -1], [1, 2]]
        };
        let sources = [
            MergeSource {
                reader: &reader,
                doc_map: &maps[0],
                field_map: &[],
            },
            MergeSource {
                reader: &reader,
                doc_map: &maps[1],
                field_map: &[],
            },
        ];
        let merge = MergeState {
            sources: &sources,
            needs_index_sort: sorted,
            check_aborted: &|| Ok(()),
        };
        measure(
            if sorted {
                "asserting_stored_fields_merge_sorted"
            } else {
                "asserting_stored_fields_merge_sequential"
            },
            warmup,
            duration,
            || {
                let mut digest = OFFSET;
                for _ in 0..BATCH {
                    let directory = ByteBuffersDirectory::new();
                    let mut writer = format
                        .fields_writer(state(&directory, Mode::BestSpeed))
                        .unwrap();
                    digest = hash(digest, writer.merge(&merge).unwrap() as u64);
                    writer.close().unwrap();
                    digest = consume(digest, &directory);
                }
                digest
            },
        );
    }
}
