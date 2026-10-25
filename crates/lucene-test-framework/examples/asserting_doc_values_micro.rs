//! Same actual native bytes, iterator assertions, values and output digests as Java.
use lucene_codecs::field_infos::{
    self, DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos,
};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_search::reader::NO_MORE_DOCS;
use lucene_search::Result;
use lucene_test_framework::codecs::asserting_doc_values::*;
use std::hint::black_box;
use std::time::{Duration, Instant};
const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
#[derive(Default)]
struct Digest(i64);
impl Digest {
    fn mix(&mut self, value: i64) {
        self.0 = self.0.wrapping_mul(31).wrapping_add(value);
    }
    fn bytes(&mut self, bytes: &[u8]) {
        self.mix(bytes.len() as i64);
        for byte in bytes {
            self.mix(i64::from(*byte));
        }
    }
}
fn read<'a>(fields: &'a FieldInfos, files: &'a Files) -> ReadState<'a> {
    ReadState {
        files,
        segment: SegmentReadState {
            max_doc: 12,
            segment_id: &ID,
            segment_suffix: "s",
            field_infos: fields,
        },
    }
}
fn write(fields: &FieldInfos) -> SegmentWriteState<'_> {
    SegmentWriteState {
        max_doc: 12,
        segment_id: &ID,
        segment_suffix: "s",
        field_infos: fields,
    }
}
fn add(
    consumer: &mut impl DocValuesConsumer,
    field: &FieldInfo,
    producer: &dyn DocValuesProducer,
) -> Result<()> {
    match field.doc_values_type {
        DocValuesType::Numeric => consumer.add_numeric_field(field, producer),
        DocValuesType::Binary => consumer.add_binary_field(field, producer),
        DocValuesType::Sorted => consumer.add_sorted_field(field, producer),
        DocValuesType::SortedNumeric => consumer.add_sorted_numeric_field(field, producer),
        DocValuesType::SortedSet => consumer.add_sorted_set_field(field, producer),
        DocValuesType::None => Ok(()),
    }
}
fn digest_values(producer: &dyn DocValuesProducer, fields: &FieldInfos, digest: &mut Digest) {
    for field in &fields.fields {
        digest.mix(i64::from(field.number));
        match field.doc_values_type {
            DocValuesType::Numeric => {
                let mut v = producer.get_numeric(field).unwrap();
                loop {
                    let doc = v.next_doc().unwrap();
                    if doc == NO_MORE_DOCS {
                        break;
                    }
                    digest.mix(i64::from(doc));
                    digest.mix(v.long_value());
                }
            }
            DocValuesType::Binary => {
                let mut v = producer.get_binary(field).unwrap();
                loop {
                    let doc = v.next_doc().unwrap();
                    if doc == NO_MORE_DOCS {
                        break;
                    }
                    digest.mix(i64::from(doc));
                    digest.bytes(v.binary_value());
                }
            }
            DocValuesType::Sorted => {
                let mut v = producer.get_sorted(field).unwrap();
                digest.mix(i64::from(v.value_count()));
                for ord in 0..v.value_count() {
                    digest.bytes(&v.lookup_ord(ord).unwrap());
                }
                loop {
                    let doc = v.next_doc().unwrap();
                    if doc == NO_MORE_DOCS {
                        break;
                    }
                    digest.mix(i64::from(doc));
                    digest.mix(i64::from(v.ord_value()));
                }
            }
            DocValuesType::SortedNumeric => {
                let mut v = producer.get_sorted_numeric(field).unwrap();
                loop {
                    let doc = v.next_doc().unwrap();
                    if doc == NO_MORE_DOCS {
                        break;
                    }
                    digest.mix(i64::from(doc));
                    let count = v.doc_value_count();
                    digest.mix(i64::from(count));
                    for _ in 0..count {
                        digest.mix(v.next_value().unwrap());
                    }
                }
            }
            DocValuesType::SortedSet => {
                let mut v = producer.get_sorted_set(field).unwrap();
                digest.mix(v.value_count());
                for ord in 0..v.value_count() {
                    digest.bytes(&v.lookup_ord(ord).unwrap());
                }
                loop {
                    let doc = v.next_doc().unwrap();
                    if doc == NO_MORE_DOCS {
                        break;
                    }
                    digest.mix(i64::from(doc));
                    let count = v.doc_value_count();
                    digest.mix(i64::from(count));
                    for _ in 0..count {
                        digest.mix(v.next_ord().unwrap());
                    }
                }
            }
            DocValuesType::None => (),
        }
        if field.doc_values_skip_index_type != DocValuesSkipIndexType::None {
            let mut s = producer.get_skipper(field).unwrap().unwrap();
            digest.mix(s.global_min_value());
            digest.mix(s.global_max_value());
            digest.mix(i64::from(s.global_doc_count()));
            digest.mix(i64::from(s.max_value_count()));
            let mut target = 0;
            while target < 12 {
                s.advance(target).unwrap();
                if s.min_doc_id(0) == NO_MORE_DOCS {
                    break;
                }
                for level in 0..s.num_levels() {
                    digest.mix(s.min_doc_id(level).into());
                    digest.mix(s.max_doc_id(level).into());
                    digest.mix(s.min_value(level));
                    digest.mix(s.max_value(level));
                    digest.mix(s.doc_count(level).into());
                }
                target = s.max_doc_id(0) + 1;
            }
        }
    }
}
#[derive(Default)]
struct NoopConsumer {
    calls: i64,
}
macro_rules! noop {
    ($method:ident) => {
        fn $method(&mut self, field: &FieldInfo, _: &dyn DocValuesProducer) -> Result<()> {
            self.calls = self
                .calls
                .wrapping_mul(31)
                .wrapping_add(i64::from(field.number));
            Ok(())
        }
    };
}
impl DocValuesConsumer for NoopConsumer {
    noop!(add_numeric_field);
    noop!(add_binary_field);
    noop!(add_sorted_field);
    noop!(add_sorted_numeric_field);
    noop!(add_sorted_set_field);
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
type Reader<'a> = AssertingDocValuesProducer<'a, Lucene90DocValuesProducer>;
fn batch(
    name: &str,
    fields: &FieldInfos,
    files: &Files,
    prepared: &Reader<'_>,
    source: &Lucene90DocValuesProducer,
) -> i64 {
    let format = AssertingDocValuesFormat::new(Lucene90DocValuesFormat);
    let mut digest = Digest::default();
    for _ in 0..16 {
        match name {
            "validate" => {
                let mut consumer = AssertingDocValuesConsumer::new(NoopConsumer::default(), 12);
                for field in &fields.fields {
                    add(&mut consumer, field, source).unwrap();
                }
                digest.mix(consumer.delegate().calls);
            }
            "read" => {
                let mut producer = format.fields_producer(read(fields, files)).unwrap();
                producer.check_integrity().unwrap();
                digest_values(&producer, fields, &mut digest);
                producer.close().unwrap();
            }
            "write" => {
                let mut consumer = format.fields_consumer(write(fields)).unwrap();
                for field in &fields.fields {
                    add(&mut consumer, field, source).unwrap();
                }
                consumer.close().unwrap();
                for ext in ["dvm", "dvd", "dvs"] {
                    digest.bytes(&consumer.delegate().output().unwrap()[ext]);
                }
            }
            "merge_lifecycle" => {
                let mut producer = prepared.get_merge_instance();
                digest_values(&producer, fields, &mut digest);
                producer.close().unwrap();
            }
            _ => panic!("unknown case"),
        }
    }
    black_box(digest.0)
}
fn run(
    name: &str,
    fields: &FieldInfos,
    files: &Files,
    prepared: &Reader<'_>,
    source: &Lucene90DocValuesProducer,
    ms: u64,
) -> (u64, Duration) {
    let expected = batch(name, fields, files, prepared, source);
    let start = Instant::now();
    let mut count = 0;
    loop {
        assert_eq!(batch(name, fields, files, prepared, source), expected);
        count += 16;
        if start.elapsed() >= Duration::from_millis(ms) {
            return (count, start.elapsed());
        }
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup = args.get(1).map_or(1000, |s| s.parse().unwrap());
    let measure = args.get(2).map_or(1500, |s| s.parse().unwrap());
    let root = "fixtures/testframework/data/asserting_doc_values";
    let fields =
        field_infos::parse(&std::fs::read(format!("{root}.fnm")).unwrap(), &ID, "").unwrap();
    let files: Files = ["dvm", "dvd", "dvs"]
        .into_iter()
        .map(|ext| {
            (
                ext.to_owned(),
                std::fs::read(format!("{root}.{ext}")).unwrap(),
            )
        })
        .collect();
    let native = Lucene90DocValuesFormat;
    let source = native.fields_producer(read(&fields, &files)).unwrap();
    let format = AssertingDocValuesFormat::new(native);
    let mut prepared = format.fields_producer(read(&fields, &files)).unwrap();
    println!(
        "#memory\tasserting_doc_values_producer_wrapper\t{}",
        std::mem::size_of::<Reader<'_>>() - std::mem::size_of::<Lucene90DocValuesProducer>()
    );
    println!(
        "#memory\tasserting_doc_values_consumer_wrapper\t{}",
        std::mem::size_of::<AssertingDocValuesConsumer<NoopConsumer>>()
            - std::mem::size_of::<NoopConsumer>()
    );
    println!(
        "#memory\tasserting_doc_values_skipper_wrapper\t{}",
        std::mem::size_of::<AssertingDocValuesSkipper<'_>>()
    );
    for name in ["validate", "read", "write", "merge_lifecycle"] {
        println!(
            "#check\tasserting_doc_values_{name}\t{}\t16",
            batch(name, &fields, &files, &prepared, &source)
        );
        run(name, &fields, &files, &prepared, &source, warmup);
        let (count, elapsed) = run(name, &fields, &files, &prepared, &source, measure);
        println!(
            "asserting_doc_values_{name}\t{:.3}\t{count}",
            elapsed.as_nanos() as f64 / count as f64
        );
    }
    prepared.close().unwrap();
}
