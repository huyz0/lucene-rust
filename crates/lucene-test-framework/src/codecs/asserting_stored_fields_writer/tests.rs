#![cfg(test)]
use std::panic::{catch_unwind, AssertUnwindSafe};

use lucene_codecs::stored_fields_writer::with_writer;
use lucene_store::data_input::SliceInput;
use lucene_store::{ByteBuffersDirectory, Directory};

use super::*;

const ID: [u8; 16] = [39; 16];

fn state(directory: &dyn Directory) -> StoredFieldsWriteState<'_> {
    StoredFieldsWriteState {
        directory,
        segment: "_test",
        id: &ID,
        suffix: "",
        mode: Mode::BestSpeed,
    }
}

#[test]
fn installed_factory_writes_all_overloads_and_empty_documents_as_native_bytes() {
    let directory = ByteBuffersDirectory::new();
    let codec: &dyn Lucene104Codec = &AssertingStoredFieldsCodec;
    assert_eq!(codec.name(), "Lucene104");
    let factory = codec.stored_fields_writer_factory().unwrap();
    let values = [
        FieldValue::String("stored café".into()),
        FieldValue::Binary(vec![0, 255, 1]),
        FieldValue::Int(i32::MIN),
        FieldValue::Long(i64::MAX),
        FieldValue::Float(-0.0),
        FieldValue::Double(3.25),
    ];
    let result: stored_fields::Result<()> = with_writer(factory, state(&directory), |writer| {
        writer.start_document()?;
        for (number, value) in (0_i32..).zip(&values) {
            writer.write_field(number, value)?;
        }
        let mut input = SliceInput::new(&[12, 13, 14, 15]);
        writer.write_field_data_input(6, &mut input, 3)?;
        assert_eq!(input.read_byte().unwrap(), 15);
        writer.finish_document()?;
        writer.start_document()?;
        writer.finish_document()?;
        writer.finish(2)
    });
    result.unwrap();
    let fdt = directory.open("_test.fdt").unwrap();
    let fdx = directory.open("_test.fdx").unwrap();
    let fdm = directory.open("_test.fdm").unwrap();
    let mut expected_fields: Vec<stored_fields::StoredField> = (0_i32..)
        .zip(values.iter().cloned())
        .map(|(field_number, value)| stored_fields::StoredField {
            field_number,
            value,
        })
        .collect();
    expected_fields.push(stored_fields::StoredField {
        field_number: 6,
        value: FieldValue::Binary(vec![12, 13, 14]),
    });
    let expected = stored_fields::write_best_speed(
        &[
            stored_fields::Document {
                fields: expected_fields,
            },
            stored_fields::Document::default(),
        ],
        &ID,
        "",
    );
    assert_eq!(&*fdt, expected.0.as_slice());
    assert_eq!(&*fdx, expected.1.as_slice());
    assert_eq!(&*fdm, expected.2.as_slice());
    let reader = stored_fields::open(&fdt, &fdx, &fdm, &ID, "").unwrap();
    assert_eq!(reader.max_doc(), 2);
    reader.check_integrity().unwrap();
    let document = reader.document(0).unwrap();
    assert_eq!(document.fields.len(), 7);
    for ((number, expected), actual) in (0_i32..).zip(&values).zip(&document.fields) {
        assert_eq!(actual.field_number, number);
        assert_eq!(&actual.value, expected);
    }
    assert_eq!(document.fields[6].field_number, 6);
    assert_eq!(
        document.fields[6].value,
        FieldValue::Binary(vec![12, 13, 14])
    );
    let FieldValue::Float(value) = document.fields[4].value else {
        panic!("float field changed type")
    };
    assert_eq!(value.to_bits(), (-0.0_f32).to_bits());
    assert!(reader.document(1).unwrap().fields.is_empty());
}

#[test]
fn zero_documents_finish_and_repeated_close_are_valid() {
    let directory = ByteBuffersDirectory::new();
    let mut writer = AssertingStoredFieldsCodec
        .fields_writer(state(&directory))
        .unwrap();
    writer.finish(0).unwrap();
    writer.close().unwrap();
    writer.close().unwrap();
    let fdt = directory.open("_test.fdt").unwrap();
    let fdx = directory.open("_test.fdx").unwrap();
    let fdm = directory.open("_test.fdm").unwrap();
    assert_eq!(
        stored_fields::open(&fdt, &fdx, &fdm, &ID, "")
            .unwrap()
            .max_doc(),
        0
    );
}

#[test]
fn installed_wrapper_rejects_invalid_document_lifecycles() {
    for scenario in 0..4 {
        let directory = ByteBuffersDirectory::new();
        let mut writer = AssertingStoredFieldsCodec
            .fields_writer(state(&directory))
            .unwrap();
        assert!(catch_unwind(AssertUnwindSafe(|| {
            match scenario {
                0 => writer.write_field(0, &FieldValue::Int(1)).unwrap(),
                1 => writer.finish_document().unwrap(),
                2 => {
                    writer.start_document().unwrap();
                    writer.start_document().unwrap();
                }
                _ => {
                    writer.start_document().unwrap();
                    writer.finish(1).unwrap();
                }
            }
        }))
        .is_err());
        writer.close().unwrap();
    }
}

#[test]
fn invalid_factory_state_cannot_change_native_format_selection() {
    let directory = ByteBuffersDirectory::new();
    for (mode, suffix) in [(Mode::BestCompression, ""), (Mode::BestSpeed, "other")] {
        let result = AssertingStoredFieldsCodec.fields_writer(StoredFieldsWriteState {
            mode,
            suffix,
            ..state(&directory)
        });
        let error = match result {
            Ok(_) => panic!("factory accepted a different native format"),
            Err(error) => error,
        };
        assert_eq!(
            error.to_string(),
            "asserting stored-fields routing requires BestSpeed and an empty suffix"
        );
    }
    assert!(directory.list_all().unwrap().is_empty());
}

#[test]
fn input_and_finish_errors_escape_the_actual_wrapped_writer() {
    let directory = ByteBuffersDirectory::new();
    let result: stored_fields::Result<()> =
        with_writer(&AssertingStoredFieldsCodec, state(&directory), |writer| {
            writer.start_document()?;
            writer.write_field_data_input(0, &mut SliceInput::new(&[1]), 2)
        });
    assert!(matches!(result, Err(stored_fields::Error::Store(_))));
    assert!(directory.open("_test.fdx").is_err());

    let directory = ByteBuffersDirectory::new();
    let result: stored_fields::Result<()> =
        with_writer(&AssertingStoredFieldsCodec, state(&directory), |writer| {
            writer.start_document()?;
            writer.finish_document()?;
            writer.finish(2)
        });
    assert_eq!(
        result.unwrap_err().to_string(),
        "Wrote 1 docs, finish called with numDocs=2"
    );
    assert!(directory.open("_test.fdx").is_err());
}
