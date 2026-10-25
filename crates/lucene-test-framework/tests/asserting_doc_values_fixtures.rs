use lucene_codecs::field_infos::{
    self, DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos,
};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_search::reader::NO_MORE_DOCS;
use lucene_test_framework::codecs::asserting_doc_values::*;
use lucene_test_framework::codecs::cranky_doc_values::Column;
use std::panic::{catch_unwind, AssertUnwindSafe};
const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
fn data(name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/testframework/data")
            .join(name),
    )
    .unwrap()
}
fn files() -> Files {
    ["dvm", "dvd", "dvs"]
        .into_iter()
        .map(|ext| (ext.into(), data(&format!("asserting_doc_values.{ext}"))))
        .collect()
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
) -> lucene_search::Result<()> {
    match field.doc_values_type {
        DocValuesType::Numeric => consumer.add_numeric_field(field, producer),
        DocValuesType::Binary => consumer.add_binary_field(field, producer),
        DocValuesType::Sorted => consumer.add_sorted_field(field, producer),
        DocValuesType::SortedNumeric => consumer.add_sorted_numeric_field(field, producer),
        DocValuesType::SortedSet => consumer.add_sorted_set_field(field, producer),
        DocValuesType::None => Ok(()),
    }
}
fn hex(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        "-".into()
    } else {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}
fn terms(out: &mut String, number: i32, dict: &[Vec<u8>]) {
    for (ord, term) in dict.iter().enumerate() {
        out.push_str(&format!("term\t{number}\t{ord}\t{}\n", hex(term)));
    }
}
fn multi(out: &mut String, kind: &str, n: i32, docs: &[i32], counts: &[u32], values: &[i64]) {
    let mut offset = 0;
    for (&doc, &count) in docs.iter().zip(counts) {
        out.push_str(&format!("{kind}\t{n}\t{doc}"));
        for value in &values[offset..offset + count as usize] {
            out.push_str(&format!("\t{value}"));
        }
        out.push('\n');
        offset += count as usize;
    }
}
fn manifest(producer: &dyn DocValuesProducer, fields: &FieldInfos) -> String {
    let mut out = String::from("maxDoc\t12\n");
    for field in &fields.fields {
        let kind = match field.doc_values_type {
            DocValuesType::Numeric => "NUMERIC",
            DocValuesType::Binary => "BINARY",
            DocValuesType::Sorted => "SORTED",
            DocValuesType::SortedNumeric => "SORTED_NUMERIC",
            DocValuesType::SortedSet => "SORTED_SET",
            DocValuesType::None => continue,
        };
        let n = field.number;
        out.push_str(&format!(
            "field\t{n}\t{}\t{kind}\t{}\n",
            field.name,
            if field.doc_values_skip_index_type == DocValuesSkipIndexType::None {
                "NONE"
            } else {
                "RANGE"
            }
        ));
        match snapshot(field, producer).unwrap() {
            Column::Numeric(c) => {
                for (doc, value) in c {
                    out.push_str(&format!("numeric\t{n}\t{doc}\t{value}\n"));
                }
            }
            Column::Binary(c) => {
                let mut offset = 0;
                for (&doc, &len) in c.docs.iter().zip(&c.lengths) {
                    let end = offset + len as usize;
                    out.push_str(&format!(
                        "binary\t{n}\t{doc}\t{}\n",
                        hex(&c.bytes[offset..end])
                    ));
                    offset = end;
                }
            }
            Column::Sorted(c) => {
                terms(&mut out, n, &c.dict);
                for (&doc, &ord) in c.docs.iter().zip(&c.ords) {
                    out.push_str(&format!("sorted\t{n}\t{doc}\t{ord}\n"));
                }
            }
            Column::SortedNumeric(c) => {
                multi(&mut out, "sortedNumeric", n, &c.docs, &c.counts, &c.values)
            }
            Column::SortedSet(c) => {
                terms(&mut out, n, &c.dict);
                multi(&mut out, "sortedSet", n, &c.docs, &c.counts, &c.ords);
            }
        }
        if field.doc_values_skip_index_type != DocValuesSkipIndexType::None {
            let mut skipper = producer.get_skipper(field).unwrap().unwrap();
            out.push_str(&format!(
                "skipGlobal\t{n}\t{}\t{}\t{}\t{}\n",
                skipper.global_min_value(),
                skipper.global_max_value(),
                skipper.global_doc_count(),
                skipper.max_value_count()
            ));
            let mut target = 0;
            while target < 12 {
                skipper.advance(target).unwrap();
                if skipper.min_doc_id(0) == NO_MORE_DOCS {
                    break;
                }
                for level in 0..skipper.num_levels() {
                    out.push_str(&format!(
                        "skip\t{n}\t{level}\t{}\t{}\t{}\t{}\t{}\n",
                        skipper.min_doc_id(level),
                        skipper.max_doc_id(level),
                        skipper.min_value(level),
                        skipper.max_value(level),
                        skipper.doc_count(level)
                    ));
                }
                target = skipper.max_doc_id(0) + 1;
            }
        }
    }
    out
}
#[test]
fn actual_java_all_five_types_dictionary_skips_and_output_files_match() {
    let fields = field_infos::parse(&data("asserting_doc_values.fnm"), &ID, "").unwrap();
    let files = files();
    let native = Lucene90DocValuesFormat;
    let source = native.fields_producer(read(&fields, &files)).unwrap();
    let mut format = AssertingDocValuesFormat::new(native);
    assert_eq!(format.name(), "Asserting");
    assert_eq!(format.delegate().name(), "Lucene90");
    assert_eq!(format.delegate_mut().name(), "Lucene90");
    let mut consumer = format.fields_consumer(write(&fields)).unwrap();
    for field in &fields.fields {
        add(&mut consumer, field, &source).unwrap();
    }
    consumer.close().unwrap();
    assert_eq!(consumer.delegate().output(), Some(&files));
    assert_eq!(consumer.delegate_mut().output(), Some(&files));
    let mut producer = format.fields_producer(read(&fields, &files)).unwrap();
    producer.check_integrity().unwrap();
    assert!(producer
        .to_string()
        .starts_with("AssertingDocValuesProducer("));
    assert_eq!(
        manifest(&producer, &fields),
        String::from_utf8(data("asserting_doc_values_values.tsv")).unwrap()
    );
    let mut merged = producer.get_merge_instance();
    assert_eq!(manifest(&merged, &fields), manifest(&producer, &fields));
    producer.close().unwrap();
    assert!(producer.check_integrity().is_err());
    assert!(producer
        .get_numeric(fields.field_by_name("numeric").unwrap())
        .is_err());
    assert_eq!(
        manifest(&merged, &fields),
        String::from_utf8(data("asserting_doc_values_values.tsv")).unwrap()
    );
    merged.close().unwrap();
    merged.close().unwrap();
    assert!(merged
        .get_binary(fields.field_by_name("binary").unwrap())
        .is_err());
}
#[test]
fn native_headers_corruption_type_and_lifecycle_errors_are_observable() {
    let fields = field_infos::parse(&data("asserting_doc_values.fnm"), &ID, "").unwrap();
    let files = files();
    let format = Lucene90DocValuesFormat;
    for ext in ["dvm", "dvd", "dvs"] {
        let mut damaged = files.clone();
        damaged.get_mut(ext).unwrap()[0] ^= 1;
        assert!(format.fields_producer(read(&fields, &damaged)).is_err());
        let mut damaged = files.clone();
        damaged.get_mut(ext).unwrap().truncate(8);
        assert!(format.fields_producer(read(&fields, &damaged)).is_err());
    }
    let producer = format.fields_producer(read(&fields, &files)).unwrap();
    let n = fields.field_by_name("numeric").unwrap();
    let b = fields.field_by_name("binary").unwrap();
    assert!(producer.get_binary(n).is_err());
    assert!(producer.get_numeric(b).is_err());
    assert!(producer.get_sorted(n).is_err());
    assert!(producer.get_sorted_numeric(n).is_err());
    assert!(producer.get_sorted_set(n).is_err());
    assert!(producer
        .get_numeric(&FieldInfo::new("absent", 999).with_doc_values(
            DocValuesType::Numeric,
            DocValuesSkipIndexType::None,
            -1
        ))
        .is_err());
    assert!(snapshot(&FieldInfo::new("plain", 999), &producer).is_err());
    assert!(producer.get_skipper(b).unwrap().is_none());
    let empty = FieldInfos::new(Vec::new()).unwrap();
    let asserted = AssertingDocValuesFormat::new(format);
    assert!(catch_unwind(AssertUnwindSafe(
        || asserted.fields_producer(read(&empty, &files))
    ))
    .is_err());
    let mut consumer = asserted.fields_consumer(write(&fields)).unwrap();
    consumer.close().unwrap();
    assert!(add(&mut consumer, n, &producer).is_err());
}
#[test]
fn native_iterators_support_advance_exact_lookups_and_empty_exhaustion() {
    let fields = field_infos::parse(&data("asserting_doc_values.fnm"), &ID, "").unwrap();
    let files = files();
    let format = AssertingDocValuesFormat::new(Lucene90DocValuesFormat);
    let producer = format.fields_producer(read(&fields, &files)).unwrap();
    let mut numeric = producer
        .get_numeric(fields.field_by_name("numeric").unwrap())
        .unwrap();
    assert_eq!(numeric.cost(), 8);
    assert_eq!(numeric.advance(3).unwrap(), 3);
    assert!(numeric.advance_exact(4).unwrap());
    assert_eq!(numeric.long_value(), -148);
    assert!(!numeric.advance_exact(5).unwrap());
    assert_eq!(numeric.next_doc().unwrap(), 6);
    let mut sorted = producer
        .get_sorted(fields.field_by_name("sorted").unwrap())
        .unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| sorted.lookup_ord(-1))).is_err());
    let mut raw = Lucene90DocValuesFormat
        .fields_producer(read(&fields, &files))
        .unwrap();
    let mut sorted = raw
        .get_sorted(fields.field_by_name("sorted").unwrap())
        .unwrap();
    assert!(sorted.lookup_ord(-1).is_err());
    assert!(sorted.lookup_ord(i32::MAX).is_err());
    assert_eq!(sorted.lookup_term(b"zz").unwrap(), -5);
    drop(sorted);
    let mut empty = raw
        .get_sorted_numeric(fields.field_by_name("emptySortedNumeric").unwrap())
        .unwrap();
    assert_eq!(empty.next_doc().unwrap(), 3);
    assert_eq!(empty.next_value().unwrap(), 1);
    assert_eq!(empty.next_doc().unwrap(), NO_MORE_DOCS);
    assert_eq!(empty.next_doc().unwrap(), NO_MORE_DOCS);
    drop(empty);
    raw.close().unwrap();
}
#[test]
fn native_ownership_outcomes_match_actual_java_bytebuffers_handles() {
    let fields = field_infos::parse(&data("asserting_doc_values.fnm"), &ID, "").unwrap();
    let files = files();
    let format = AssertingDocValuesFormat::new(Lucene90DocValuesFormat);
    let mut actual = String::new();
    for probe in [
        "original-close-merged-read",
        "merged-close-original-read",
        "closed-original-merge",
    ] {
        let mut original = format.fields_producer(read(&fields, &files)).unwrap();
        let mut merged = original.get_merge_instance();
        let expected = manifest(&original, &fields);
        let status = match probe {
            "original-close-merged-read" => {
                original.close().unwrap();
                assert_eq!(manifest(&merged, &fields), expected);
                merged.close().unwrap();
                "ok"
            }
            "merged-close-original-read" => {
                merged.close().unwrap();
                assert_eq!(manifest(&original, &fields), expected);
                original.close().unwrap();
                "ok"
            }
            "closed-original-merge" => {
                original.close().unwrap();
                let panic = catch_unwind(AssertUnwindSafe(|| original.get_merge_instance()))
                    .err()
                    .unwrap();
                let message = panic
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
                    .unwrap();
                assert!(message.contains("closed doc values producer"));
                merged.close().unwrap();
                "closed"
            }
            _ => unreachable!(),
        };
        actual.push_str(&format!("{probe}\t{status}\n"));
    }
    assert_eq!(
        actual,
        String::from_utf8(data("asserting_doc_values_lifecycle.tsv")).unwrap()
    );
}
