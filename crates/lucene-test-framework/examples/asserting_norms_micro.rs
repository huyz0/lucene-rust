//! Java twin: `AssertingNormsMicro.java`; all cases consume deterministic work.
use lucene_codecs::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use lucene_search::reader::{DocIdSetIterator, DocValuesIterator, NumericDocValues, NO_MORE_DOCS};
use lucene_search::Result;
use lucene_store::{ByteBuffersDirectory, Directory};
use lucene_test_framework::codecs::asserting_norms::{
    AssertingNormsConsumer, AssertingNormsFormat, Lucene90NormsFormat, NormsConsumer, NormsFormat,
    NormsProducer,
};
use lucene_test_framework::codecs::cranky_norms::NormsState;
use std::cell::Cell;
use std::hint::black_box;
use std::time::{Duration, Instant};

const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const BATCH: u64 = 256;
fn step(digest: u64, value: u64) -> u64 {
    digest.wrapping_mul(31).wrapping_add(value)
}
fn field(name: &str, number: i32) -> FieldInfo {
    let mut field = FieldInfo::new(name, number);
    field.index_options = IndexOptions::Docs;
    field
}
fn state(directory: &dyn Directory) -> NormsState<'_> {
    NormsState {
        directory,
        segment: "_an",
        id: &ID,
        suffix: "s",
        max_doc: 65,
    }
}
struct Source {
    max_doc: i32,
    count: bool,
    gets: Cell<u64>,
    nexts: Cell<u64>,
    values: Cell<u64>,
}
impl Source {
    fn new(max_doc: i32, count: bool) -> Self {
        Self {
            max_doc,
            count,
            gets: Cell::new(0),
            nexts: Cell::new(0),
            values: Cell::new(0),
        }
    }
    fn reset(&self) {
        self.gets.set(0);
        self.nexts.set(0);
        self.values.set(0);
    }
}
struct Values<'a> {
    source: &'a Source,
    sparse: bool,
    doc: i32,
}
impl DocIdSetIterator for Values<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        if self.source.count {
            self.source.nexts.set(self.source.nexts.get() + 1);
        }
        let stride = if self.sparse { 3 } else { 1 };
        let next = if self.doc < 0 {
            0
        } else {
            (self.doc / stride + 1) * stride
        };
        self.doc = if next >= self.source.max_doc {
            NO_MORE_DOCS
        } else {
            next
        };
        Ok(self.doc)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        while self.next_doc()? < target {}
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        i64::from(if self.sparse {
            (self.source.max_doc + 2) / 3
        } else {
            self.source.max_doc
        })
    }
}
impl DocValuesIterator for Values<'_> {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        self.doc = target;
        Ok(!self.sparse || target % 3 == 0)
    }
}
impl NumericDocValues for Values<'_> {
    fn long_value(&self) -> i64 {
        if self.source.count {
            self.source.values.set(self.source.values.get() + 1);
        }
        if self.sparse {
            i64::from(self.doc) * 17 - 33
        } else {
            i64::from(self.doc) * 37 - 500
        }
    }
}
impl NormsProducer for Source {
    fn get_norms(&self, field: &FieldInfo) -> Result<Box<dyn NumericDocValues + '_>> {
        if self.count {
            self.gets.set(self.gets.get() + 1);
        }
        Ok(Box::new(Values {
            source: self,
            sparse: field.number == 1,
            doc: -1,
        }))
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
    fn check_integrity(&self) -> Result<()> {
        Ok(())
    }
    fn get_merge_instance(&self) -> Self {
        Self::new(self.max_doc, self.count)
    }
}
struct Target {
    expected: usize,
    digest: u64,
    calls: u64,
}
impl NormsConsumer for Target {
    fn add_norms_field(&mut self, field: &FieldInfo, producer: &dyn NormsProducer) -> Result<()> {
        assert_eq!(
            producer as *const dyn NormsProducer as *const () as usize,
            self.expected
        );
        self.calls += 1;
        let mut values = producer.get_norms(field)?;
        while values.next_doc()? != NO_MORE_DOCS {
            self.digest = step(self.digest, values.long_value() as u64);
        }
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
struct Harness {
    fields: FieldInfos,
    large_field: FieldInfo,
    source: Source,
    large: Box<Source>,
    validator: AssertingNormsConsumer<Target>,
    format: AssertingNormsFormat<Lucene90NormsFormat>,
    saved: ByteBuffersDirectory,
}
impl Harness {
    fn new() -> Self {
        let fields = FieldInfos::new(vec![field("dense", 0), field("sparse", 1)]).unwrap();
        let large = Box::new(Source::new(4096, true));
        let validator = AssertingNormsConsumer::new(
            Target {
                expected: &*large as *const Source as usize,
                digest: 0,
                calls: 0,
            },
            4096,
        );
        let mut result = Self {
            fields,
            large_field: field("large", 0),
            source: Source::new(65, false),
            large,
            validator,
            format: AssertingNormsFormat::new(Lucene90NormsFormat),
            saved: ByteBuffersDirectory::new(),
        };
        let mut writer = result.format.norms_consumer(&state(&result.saved)).unwrap();
        for field in &result.fields.fields {
            writer.add_norms_field(field, &result.source).unwrap();
        }
        writer.close().unwrap();
        result
    }
    fn batch(&mut self, case: u8) -> u64 {
        let mut digest = 0;
        if case == 0 {
            self.large.reset();
            self.validator.delegate_mut().digest = 0;
            self.validator.delegate_mut().calls = 0;
        }
        for _ in 0..BATCH {
            match case {
                0 => {
                    self.validator
                        .add_norms_field(black_box(&self.large_field), &*self.large)
                        .unwrap();
                }
                1 => {
                    let mut producer = self
                        .format
                        .norms_producer(&state(&self.saved), &self.fields)
                        .unwrap();
                    for field in &self.fields.fields {
                        let mut values = producer.get_norms(field).unwrap();
                        digest = step(digest, values.cost() as u64);
                        while values.next_doc().unwrap() != NO_MORE_DOCS {
                            digest = step(digest, values.doc_id() as u64);
                            digest = step(digest, values.long_value() as u64);
                        }
                    }
                    producer.close().unwrap();
                }
                _ => {
                    let directory = ByteBuffersDirectory::new();
                    let mut writer = self.format.norms_consumer(&state(&directory)).unwrap();
                    for field in &self.fields.fields {
                        writer
                            .add_norms_field(black_box(field), &self.source)
                            .unwrap();
                    }
                    writer.close().unwrap();
                    for name in ["_an_s.nvm", "_an_s.nvd"] {
                        for &byte in directory.open(name).unwrap().iter() {
                            digest = step(digest, u64::from(byte));
                        }
                    }
                }
            }
        }
        if case == 0 {
            assert_eq!(self.large.gets.get(), BATCH * 2);
            assert_eq!(self.large.nexts.get(), BATCH * 2 * 4097);
            assert_eq!(self.large.values.get(), BATCH * 2 * 4096);
            assert_eq!(self.validator.delegate().calls, BATCH);
            digest = self.validator.delegate().digest;
            for value in [
                self.large.gets.get(),
                self.large.nexts.get(),
                self.large.values.get(),
                self.validator.delegate().calls,
            ] {
                digest = step(digest, value);
            }
        }
        black_box(digest)
    }
    fn run(&mut self, case: u8, millis: u64) -> (Duration, u64) {
        let expected = self.batch(case);
        let start = Instant::now();
        let mut count = 0;
        loop {
            let actual = self.batch(case);
            assert_eq!(actual, expected, "timed batch digest changed");
            black_box(actual);
            count += BATCH;
            if start.elapsed() >= Duration::from_millis(millis) {
                break;
            }
        }
        (start.elapsed(), count)
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup: u64 = args.get(1).map(|v| v.parse().unwrap()).unwrap_or(500);
    let measure: u64 = args.get(2).map(|v| v.parse().unwrap()).unwrap_or(1000);
    let mut harness = Harness::new();
    // Outside timing: native iterator state excludes shared producer bytes,
    // asserting wrappers and allocator metadata. Report separately from the
    // format-only footprint rather than hiding the sparse cursor's storage.
    let mut native = Lucene90NormsFormat
        .norms_producer(&state(&harness.saved), &harness.fields)
        .unwrap();
    for field in &harness.fields.fields {
        let values = native.get_norms(field).unwrap();
        eprintln!(
            "native norms iterator {} shallow_owned_bytes={}",
            field.name,
            std::mem::size_of_val(&*values)
        );
    }
    native.close().unwrap();
    println!(
        "asserting_norms_format_owned_bytes\t{}",
        std::mem::size_of::<AssertingNormsFormat<Lucene90NormsFormat>>()
    );
    for (case, name) in [
        "asserting_norms_validation",
        "asserting_norms_read",
        "asserting_norms_lifecycle",
    ]
    .into_iter()
    .enumerate()
    {
        println!("#check\t{name}\t{}\t{BATCH}", harness.batch(case as u8));
        harness.run(case as u8, warmup);
        let (elapsed, count) = harness.run(case as u8, measure);
        println!(
            "{name}\t{:.3}\t{count}",
            elapsed.as_nanos() as f64 / count as f64
        );
    }
}
