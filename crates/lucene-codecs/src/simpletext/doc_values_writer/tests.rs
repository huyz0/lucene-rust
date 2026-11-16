#![cfg(test)]
use super::*;
use crate::field_infos::DocValuesSkipIndexType;

fn fi(name: &str, ty: DocValuesType) -> FieldInfo {
    let mut f = FieldInfo::new(name, 0);
    f.doc_values_type = ty;
    f.doc_values_skip_index_type = DocValuesSkipIndexType::None;
    f
}

fn refused(r: Result<()>) {
    assert!(matches!(r, Err(Error::IllegalArgument(_))), "{r:?}");
}

#[test]
fn add_field_dispatches_on_the_input() {
    let mut w = SimpleTextDocValuesWriter::new(3);
    let bin = BinaryColumn {
        docs: vec![0],
        lengths: vec![1],
        bytes: b"z".to_vec(),
    };
    let sorted = SortedOrdsColumn {
        dict: vec![b"a".to_vec()],
        docs: vec![1],
        ords: vec![0],
    };
    let snum = SortedNumericColumn {
        docs: vec![2],
        counts: vec![1],
        values: vec![4],
    };
    let sset = SortedSetOrdsColumn {
        dict: vec![b"a".to_vec()],
        docs: vec![0],
        counts: vec![1],
        ords: vec![0],
    };
    w.add_field(
        &fi("n", DocValuesType::Numeric),
        DocValuesInput::Numeric(&[(0, 1)]),
    )
    .unwrap();
    w.add_field(
        &fi("b", DocValuesType::Binary),
        DocValuesInput::Binary(&bin),
    )
    .unwrap();
    w.add_field(
        &fi("s", DocValuesType::Sorted),
        DocValuesInput::Sorted(&sorted),
    )
    .unwrap();
    w.add_field(
        &fi("sn", DocValuesType::SortedNumeric),
        DocValuesInput::SortedNumeric(&snum),
    )
    .unwrap();
    w.add_field(
        &fi("ss", DocValuesType::SortedSet),
        DocValuesInput::SortedSet(&sset),
    )
    .unwrap();
    let data = w.finish().unwrap();
    assert!(data.starts_with(FIELD) && data.windows(END.len()).any(|x| x == END));
    assert_eq!(
        doc_values_file_name("_0", "", DOC_VALUES_EXTENSION),
        "_0.dat"
    );
    assert_eq!(
        doc_values_file_name("_0", "x", DOC_VALUES_EXTENSION),
        "_0_x.dat"
    );
}

#[test]
fn bad_input_is_refused() {
    let mut w = SimpleTextDocValuesWriter::new(3);
    // Wrong type (and not a normed field).
    refused(w.add_numeric_field(&fi("n", DocValuesType::Binary), &[(0, 1)]));
    // Out of order, repeated, outside maxDoc.
    refused(w.add_numeric_field(&fi("n", DocValuesType::Numeric), &[(1, 1), (0, 1)]));
    refused(w.add_numeric_field(&fi("n", DocValuesType::Numeric), &[(1, 1), (1, 1)]));
    refused(w.add_numeric_field(&fi("n", DocValuesType::Numeric), &[(3, 1)]));
    refused(w.add_numeric_field(&fi("n", DocValuesType::Numeric), &[(-1, 1)]));
    w.add_numeric_field(&fi("n", DocValuesType::Numeric), &[])
        .unwrap();
    refused(w.add_numeric_field(&fi("n", DocValuesType::Numeric), &[]));
    let b = fi("b", DocValuesType::Binary);
    refused(w.add_binary_field(
        &b,
        &BinaryColumn {
            docs: vec![0],
            lengths: vec![],
            bytes: vec![],
        },
    ));
    refused(w.add_binary_field(
        &b,
        &BinaryColumn {
            docs: vec![0],
            lengths: vec![2],
            bytes: vec![1],
        },
    ));
    refused(w.add_binary_field(&fi("b", DocValuesType::Numeric), &BinaryColumn::default()));
    let s = fi("s", DocValuesType::Sorted);
    refused(w.add_sorted_field(
        &s,
        &SortedOrdsColumn {
            dict: vec![],
            docs: vec![0],
            ords: vec![],
        },
    ));
    refused(w.add_sorted_field(
        &s,
        &SortedOrdsColumn {
            dict: vec![b"a".to_vec()],
            docs: vec![0],
            ords: vec![1],
        },
    ));
    refused(w.add_sorted_field(
        &s,
        &SortedOrdsColumn {
            dict: vec![b"a".to_vec()],
            docs: vec![0],
            ords: vec![-1],
        },
    ));
    let sn = fi("sn", DocValuesType::SortedNumeric);
    refused(w.add_sorted_numeric_field(
        &sn,
        &SortedNumericColumn {
            docs: vec![0],
            counts: vec![],
            values: vec![],
        },
    ));
    refused(w.add_sorted_numeric_field(
        &sn,
        &SortedNumericColumn {
            docs: vec![0],
            counts: vec![2],
            values: vec![1],
        },
    ));
    refused(w.add_sorted_numeric_field(
        &sn,
        &SortedNumericColumn {
            docs: vec![0],
            counts: vec![1],
            values: vec![1, 2],
        },
    ));
    let ss = fi("ss", DocValuesType::SortedSet);
    refused(w.add_sorted_set_field(
        &ss,
        &SortedSetOrdsColumn {
            dict: vec![],
            docs: vec![0],
            counts: vec![1],
            ords: vec![0],
        },
    ));
    refused(w.add_sorted_set_field(
        &ss,
        &SortedSetOrdsColumn {
            dict: vec![],
            docs: vec![0],
            counts: vec![],
            ords: vec![],
        },
    ));
    assert!(
        SimpleTextDocValuesWriter::new(1).finish().is_err(),
        "no field"
    );
}

#[test]
fn a_normed_field_may_write_numerics() {
    let mut f = fi("body", DocValuesType::None);
    f.index_options = IndexOptions::DocsAndFreqs;
    let mut w = SimpleTextDocValuesWriter::new(2);
    w.add_numeric_field(&f, &[(0, 3), (1, 1)]).unwrap();
    f.omit_norms = true;
    f.name = "omitted".into();
    refused(w.add_numeric_field(&f, &[(0, 3)]));
    w.finish().unwrap();
}
