//! IndexWriter stored-fields routing pair. Setup and decoded semantic checks
//! are outside each timed operation; all six cases share the Java corpus.

use std::{hint::black_box, mem::size_of, sync::Arc, time::Instant};

use lucene_codecs::{stored_fields::FieldValue, stored_fields_writer::StoredFieldsWriteSink};
use lucene_index::{
    buffered_updates::Term,
    document::{Document, Field, NumericDocValuesField, Store, StringField},
    index_writer::IndexWriter,
    merge_policy::log::LogDocMergePolicy,
    segment_info::{IndexSortField, LuceneVersion},
};
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::{ByteBuffersDirectory, Directory, FsDirectory};
use lucene_test_framework::codecs::{
    asserting_stored_fields::AssertingStoredFieldsWriter,
    asserting_stored_fields_writer::AssertingStoredFieldsCodec,
    cranky_stored_fields::Lucene90StoredFieldsWriter,
};

const SOURCE_COUNT: usize = 1024;
const FIELD_NAMES: [&str; 6] = ["binary", "text", "int", "long", "float", "double"];

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    asserting: bool,
    merge: bool,
    sorted: bool,
}

const CASES: [Case; 6] = [
    Case {
        name: "sf_routing_default_flush_unsorted",
        asserting: false,
        merge: false,
        sorted: false,
    },
    Case {
        name: "sf_routing_default_merge_deleted_unsorted",
        asserting: false,
        merge: true,
        sorted: false,
    },
    Case {
        name: "sf_routing_default_merge_deleted_sorted",
        asserting: false,
        merge: true,
        sorted: true,
    },
    Case {
        name: "sf_routing_asserting_flush_unsorted",
        asserting: true,
        merge: false,
        sorted: false,
    },
    Case {
        name: "sf_routing_asserting_merge_deleted_unsorted",
        asserting: true,
        merge: true,
        sorted: false,
    },
    Case {
        name: "sf_routing_asserting_merge_deleted_sorted",
        asserting: true,
        merge: true,
        sorted: true,
    },
];

fn values(doc: usize) -> Vec<FieldValue> {
    if doc.is_multiple_of(8) {
        return Vec::new();
    }
    vec![
        FieldValue::Binary(
            (0..128)
                .map(|at| ((doc * 31 + at * 17) & 255) as u8)
                .collect(),
        ),
        FieldValue::String(format!("doc-{doc}")),
        FieldValue::Int(doc as i32 * 31 - 16000),
        FieldValue::Long(-9_007_199_254_740_993 + doc as i64 * 104729),
        FieldValue::Float(if doc % 16 == 1 {
            -0.0
        } else {
            doc as f32 * 0.25
        }),
        FieldValue::Double(if doc % 16 == 2 {
            f64::from_bits(0x7ff8_0000_0000_0000)
        } else {
            -(doc as f64) * 0.125
        }),
    ]
}

fn corpus() -> Vec<Document> {
    (0..SOURCE_COUNT)
        .map(|doc| {
            let mut document = Document::new();
            document.add(StringField::new("id", doc.to_string(), Store::No));
            document.add(NumericDocValuesField::new(
                "rank",
                (SOURCE_COUNT - 1 - doc) as i64,
            ));
            for (name, value) in FIELD_NAMES.iter().zip(values(doc)) {
                document.add(Field::stored(*name, value));
            }
            document
        })
        .collect()
}

struct Digest(u64);

impl Digest {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn mix(&mut self, value: u64) {
        self.0 = (self.0 ^ value).wrapping_mul(0x100_0000_01b3);
    }

    fn bytes(&mut self, value: &[u8]) {
        self.mix(value.len() as u64);
        for byte in value {
            self.mix(u64::from(*byte));
        }
    }

    fn field(&mut self, name: &str, value: &FieldValue) {
        self.mix(match value {
            FieldValue::Binary(_) => 1,
            FieldValue::String(_) => 2,
            FieldValue::Int(_) => 3,
            FieldValue::Long(_) => 4,
            FieldValue::Float(_) => 5,
            FieldValue::Double(_) => 6,
        });
        self.bytes(name.as_bytes());
        match value {
            FieldValue::Binary(value) => self.bytes(value),
            FieldValue::String(value) => self.bytes(value.as_bytes()),
            FieldValue::Int(value) => self.mix(i64::from(*value) as u64),
            FieldValue::Long(value) => self.mix(*value as u64),
            FieldValue::Float(value) => self.mix(u64::from(value.to_bits())),
            FieldValue::Double(value) => self.mix(value.to_bits()),
        }
    }
}

fn expected(case: Case) -> (u64, usize) {
    let mut order: Vec<_> = (0..SOURCE_COUNT)
        .filter(|doc| !case.merge || doc % 7 != 0)
        .collect();
    if case.sorted {
        order.reverse();
    }
    let mut digest = Digest::new();
    for doc in &order {
        let fields = values(*doc);
        digest.mix(10);
        digest.mix(fields.len() as u64);
        for (name, value) in FIELD_NAMES.iter().zip(&fields) {
            digest.field(name, value);
        }
        digest.mix(11);
    }
    digest.mix(order.len() as u64);
    (digest.0, order.len())
}

fn writer(dir: &dyn Directory, case: Case) -> IndexWriter<'_> {
    let mut writer = IndexWriter::open(
        dir,
        Vec::new(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    writer.enable_explicit_documents().unwrap();
    if case.asserting {
        writer
            .set_codec(Some(Arc::new(AssertingStoredFieldsCodec)))
            .unwrap();
    }
    writer.set_use_compound_file(false);
    writer.set_max_full_flush_merge_wait_millis(0);
    let mut policy = LogDocMergePolicy::new();
    policy.set_merge_factor(1000).unwrap();
    writer.set_pluggable_merge_policy(Some(Arc::new(policy)));
    if case.sorted {
        writer
            .set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
            .unwrap();
    }
    writer
}

fn prepare_merge(writer: &mut IndexWriter<'_>, documents: &[Document]) {
    for batch in documents.chunks(128) {
        for document in black_box(batch) {
            writer.add_fields_document(black_box(document)).unwrap();
        }
        writer.flush().unwrap();
    }
    let deleted: Vec<_> = (0..SOURCE_COUNT)
        .step_by(7)
        .map(|doc| Term::new("id", doc.to_string()))
        .collect();
    writer.delete_documents_by_term(&deleted).unwrap();
    writer.commit().unwrap();
    assert_eq!(writer.segment_infos().segments.len(), 8);
}

fn verify(dir: &dyn Directory, case: Case) -> (u64, u64) {
    let (expected_digest, expected_count) = expected(case);
    let reader = DirectoryReader::open(dir).unwrap();
    assert_eq!(reader.max_doc(), expected_count as i32, "{}", case.name);
    assert_eq!(reader.num_docs(), expected_count as i32, "{}", case.name);
    assert_eq!(reader.segment_readers().len(), 1, "{}", case.name);
    let mut digest = Digest::new();
    let mut count = 0u64;
    for segment in reader.segment_readers() {
        for doc in 0..segment.max_doc {
            assert!(segment.live_docs().is_none_or(|live| live.get_doc(doc)));
            let document = segment.stored_document(doc).unwrap().unwrap_or_default();
            digest.mix(10);
            digest.mix(document.fields.len() as u64);
            for field in &document.fields {
                let name = &segment
                    .field_infos()
                    .fields
                    .iter()
                    .find(|info| info.number == field.field_number)
                    .unwrap()
                    .name;
                digest.field(name, &field.value);
            }
            digest.mix(11);
            count += 1;
        }
    }
    digest.mix(count);
    assert_eq!(digest.0, expected_digest, "{}", case.name);
    let encoded_bytes = dir
        .list_all()
        .unwrap()
        .iter()
        .filter(|file| file.as_str() != "write.lock")
        .map(|file| dir.file_length(file).unwrap())
        .sum();
    (black_box(digest.0), encoded_bytes)
}

fn iteration(case: Case, documents: &[Document]) -> (u128, u64) {
    let directory = ByteBuffersDirectory::new();
    let mut writer = writer(&directory, case);
    if case.merge {
        prepare_merge(&mut writer, documents);
    }
    let started = Instant::now();
    if case.merge {
        writer.force_merge(1).unwrap();
    } else {
        for document in black_box(documents) {
            writer.add_fields_document(black_box(document)).unwrap();
        }
        writer.flush().unwrap();
    }
    writer.commit().unwrap();
    let elapsed = started.elapsed().as_nanos();
    let (_, encoded_bytes) = verify(&directory, case);
    (elapsed, encoded_bytes)
}

fn run(case: Case, documents: &[Document], millis: u64) -> (f64, u64, u64) {
    let budget = u128::from(millis) * 1_000_000;
    let mut elapsed = 0u128;
    let mut iterations = 0u64;
    let encoded_bytes = loop {
        let (operation_ns, bytes) = iteration(case, documents);
        elapsed += operation_ns;
        iterations += 1;
        if elapsed >= budget {
            break bytes;
        }
    };
    (
        elapsed as f64 / iterations as f64,
        iterations,
        encoded_bytes,
    )
}

fn source_payload() -> usize {
    (0..SOURCE_COUNT)
        .flat_map(values)
        .map(|value| match value {
            FieldValue::Binary(value) => value.len(),
            FieldValue::String(value) => value.len(),
            FieldValue::Int(_) | FieldValue::Float(_) => 4,
            FieldValue::Long(_) | FieldValue::Double(_) => 8,
        })
        .sum()
}

fn main() {
    let arguments: Vec<_> = std::env::args().collect();
    let documents = corpus();
    if arguments
        .get(1)
        .is_some_and(|argument| argument == "--write-verify")
    {
        std::fs::create_dir_all(&arguments[2]).unwrap();
        let directory = FsDirectory::open(&arguments[2]);
        assert!(
            directory.list_all().unwrap().is_empty(),
            "verification output directory must be empty"
        );
        let case = CASES[5];
        let mut writer = writer(&directory, case);
        prepare_merge(&mut writer, &documents);
        writer.force_merge(1).unwrap();
        writer.commit().unwrap();
        let (digest, bytes) = verify(&directory, case);
        println!(
            "#verify\t{}\t{digest}\t{}\t8\t{bytes}",
            case.name,
            expected(case).1
        );
        return;
    }
    let warmup: u64 = arguments[1].parse().unwrap();
    let measure: u64 = arguments[2].parse().unwrap();
    assert!(measure > 0);
    println!(
        "#memory_scope\tsource stored-value payload excludes id/rank/names/objects; encoded bytes are complete committed index files; whole-writer retained memory and RSS are unmeasured"
    );
    println!(
        "sf_routing_source_stored_value_payload_bytes\t{}",
        source_payload()
    );
    println!(
        "#memory_scope\tincremental writer state is asserting-wrapper size minus native delegate size plus boxed sink fat pointer; native buffers, directory, allocator metadata and peak memory excluded; config shallow size is the unit codec/factory"
    );
    println!(
        "#memory_component\tsf_routing_asserting_wrapper_bookkeeping_bytes\t{}",
        size_of::<AssertingStoredFieldsWriter<Lucene90StoredFieldsWriter<'_>>>()
            - size_of::<Lucene90StoredFieldsWriter<'_>>()
    );
    println!(
        "#memory_component\tsf_routing_boxed_sink_fat_pointer_bytes\t{}",
        size_of::<Box<dyn StoredFieldsWriteSink>>()
    );
    println!(
        "sf_routing_asserting_writer_incremental_state_bytes\t{}",
        size_of::<AssertingStoredFieldsWriter<Lucene90StoredFieldsWriter<'_>>>()
            - size_of::<Lucene90StoredFieldsWriter<'_>>()
            + size_of::<Box<dyn StoredFieldsWriteSink>>()
    );
    println!(
        "sf_routing_asserting_codec_format_config_shallow_bytes\t{}",
        size_of::<AssertingStoredFieldsCodec>()
    );
    for index in [0, 3, 1, 4, 2, 5] {
        let case = CASES[index];
        let (digest, count) = expected(case);
        println!(
            "#check\t{}\t{digest}\t{count}\t{}",
            case.name,
            if case.merge { 8 } else { 1 }
        );
        run(case, &documents, warmup);
        let (ns, iterations, encoded_bytes) = run(case, &documents, measure);
        println!("{}\t{ns:.3}\t{iterations}", case.name);
        println!("{}_index_encoded_bytes\t{encoded_bytes}", case.name);
    }
}
