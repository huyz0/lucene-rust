#![cfg(test)]
use super::*;
fn allocation_baseline_validate_column(column: &Column, max_doc: i32) -> Result<()> {
    let rows = column_rows(column)?;
    let mut previous = None;
    let dictionary = match column {
        Column::Sorted(c) => Some(&c.dict),
        Column::SortedSet(c) => Some(&c.dict),
        _ => None,
    };
    if dictionary.is_some_and(|dict| dict.windows(2).any(|pair| pair[0] >= pair[1])) {
        return Err(failure("dictionary is not strictly sorted"));
    }
    for (doc, values) in rows {
        if doc < 0 || doc >= max_doc || previous.is_some_and(|previous| previous >= doc) {
            return Err(failure("invalid doc values document order"));
        }
        previous = Some(doc);
        let mut prior = None;
        for value in values {
            if let Value::Long(value) = value {
                if let Some(dict) = dictionary {
                    let index = usize::try_from(value).map_err(|_| failure("negative ordinal"))?;
                    if index >= dict.len() {
                        return Err(failure("ordinal out of bounds"));
                    }
                }
                if matches!(column, Column::SortedNumeric(_) | Column::SortedSet(_))
                    && prior.is_some_and(|prior| {
                        prior > value || (matches!(column, Column::SortedSet(_)) && prior == value)
                    })
                {
                    return Err(failure("invalid multivalue order"));
                }
                prior = Some(value);
            }
        }
    }
    Ok(())
}

#[test]
fn slice_validation_preserves_allocation_baseline_and_mixed_defect_priority() {
    fn dictionary() -> Vec<Vec<u8>> {
        vec![b"a".to_vec(), b"b".to_vec()]
    }
    fn binary(docs: Vec<i32>, lengths: Vec<u32>, bytes: Vec<u8>) -> Column {
        Column::Binary(BinaryColumn {
            docs,
            lengths,
            bytes,
        })
    }
    fn sorted(docs: Vec<i32>, ords: Vec<i64>, dict: Vec<Vec<u8>>) -> Column {
        Column::Sorted(SortedOrdsColumn { docs, ords, dict })
    }
    fn numbers(docs: Vec<i32>, counts: Vec<u32>, values: Vec<i64>) -> Column {
        Column::SortedNumeric(SortedNumericColumn {
            docs,
            counts,
            values,
        })
    }
    fn set(docs: Vec<i32>, counts: Vec<u32>, ords: Vec<i64>, dict: Vec<Vec<u8>>) -> Column {
        Column::SortedSet(SortedSetOrdsColumn {
            docs,
            counts,
            ords,
            dict,
        })
    }
    let mut cases = vec![
        Column::Numeric(vec![]),
        Column::Numeric(vec![(0, -1), (2, 9)]),
        Column::Numeric(vec![(-1, 0)]),
        Column::Numeric(vec![(0, 0), (0, 1)]),
        Column::Numeric(vec![(2, 0), (1, 1)]),
        Column::Numeric(vec![(3, 0)]),
        binary(vec![], vec![], vec![]),
        binary(vec![0, 2], vec![0, 2], vec![0, 255]),
        binary(vec![-1], vec![], vec![1]),
        binary(vec![-1], vec![2], vec![1]),
        binary(vec![-1], vec![0], vec![1]),
        binary(vec![0, 0], vec![0, 0], vec![]),
        binary(vec![0], vec![u32::MAX], vec![]),
        sorted(vec![], vec![], vec![]),
        sorted(vec![0, 2], vec![0, 1], dictionary()),
        sorted(vec![-1], vec![], vec![b"b".to_vec(), b"a".to_vec()]),
        sorted(vec![-1], vec![-1], vec![b"b".to_vec(), b"a".to_vec()]),
        sorted(vec![0, -1], vec![-1, 0], dictionary()),
        sorted(vec![0, -1], vec![2, 0], dictionary()),
        sorted(vec![0, 0], vec![0, -1], dictionary()),
        numbers(vec![], vec![], vec![]),
        numbers(vec![0, 2], vec![2, 1], vec![-1, -1, 5]),
        numbers(vec![-1], vec![], vec![]),
        numbers(vec![-1], vec![0], vec![1]),
        numbers(vec![-1], vec![2], vec![1]),
        numbers(vec![-1], vec![1], vec![1, 2]),
        numbers(vec![-1], vec![2], vec![2, 1]),
        numbers(vec![0], vec![2], vec![2, 1]),
        numbers(vec![0, -1], vec![2, 1], vec![2, 1, 0]),
        numbers(vec![0], vec![u32::MAX], vec![]),
        set(vec![], vec![], vec![], vec![]),
        set(vec![0, 2], vec![2, 1], vec![0, 1, 1], dictionary()),
        set(
            vec![-1],
            vec![],
            vec![-1],
            vec![b"b".to_vec(), b"a".to_vec()],
        ),
        set(
            vec![-1],
            vec![0],
            vec![-1],
            vec![b"b".to_vec(), b"a".to_vec()],
        ),
        set(
            vec![-1],
            vec![2],
            vec![-1],
            vec![b"b".to_vec(), b"a".to_vec()],
        ),
        set(
            vec![-1],
            vec![1],
            vec![-1],
            vec![b"b".to_vec(), b"a".to_vec()],
        ),
        set(vec![-1], vec![1], vec![-1], dictionary()),
        set(vec![0], vec![1], vec![-1], dictionary()),
        set(vec![0], vec![1], vec![2], dictionary()),
        set(vec![0], vec![2], vec![1, 1], dictionary()),
        set(vec![0], vec![2], vec![1, 0], dictionary()),
        set(vec![0], vec![2], vec![1, 2], dictionary()),
        set(vec![0, -1], vec![2, 1], vec![1, 0, -1], dictionary()),
        set(vec![0, 0], vec![1, 1], vec![0, -1], dictionary()),
    ];
    // Dictionary comparison checks equal adjacent entries as well as reversal.
    cases.push(sorted(vec![0], vec![0], vec![b"a".to_vec(), b"a".to_vec()]));
    for column in cases {
        for max_doc in [-1, 0, 3] {
            let normalize = |result: Result<()>| {
                result.map_err(|error| (std::mem::discriminant(&error), error.to_string()))
            };
            assert_eq!(
                normalize(validate_column(&column, max_doc)),
                normalize(allocation_baseline_validate_column(&column, max_doc)),
                "{column:?}, max_doc={max_doc}"
            );
        }
    }
}
fn field(kind: DocValuesType) -> FieldInfo {
    FieldInfo::new("f", 0).with_doc_values(kind, DocValuesSkipIndexType::None, -1)
}
fn infos(kind: DocValuesType) -> FieldInfos {
    FieldInfos::new(vec![field(kind)]).unwrap()
}
fn state<'a>(infos: &'a FieldInfos, max_doc: i32) -> SegmentWriteState<'a> {
    SegmentWriteState {
        max_doc,
        segment_id: &[0; 16],
        segment_suffix: "",
        field_infos: infos,
    }
}
fn error<T>(result: Result<T>, text: &str) {
    assert!(result.err().unwrap().to_string().contains(text));
}
#[test]
fn malformed_owned_columns_are_rejected_without_panics() {
    let bad = [
        Column::Numeric(vec![(-1, 1)]),
        Column::Numeric(vec![(1, 1), (1, 2)]),
        Column::Numeric(vec![(4, 1)]),
        Column::Binary(BinaryColumn {
            docs: vec![0],
            lengths: vec![],
            bytes: vec![],
        }),
        Column::Binary(BinaryColumn {
            docs: vec![0],
            lengths: vec![2],
            bytes: vec![1],
        }),
        Column::Binary(BinaryColumn {
            docs: vec![],
            lengths: vec![],
            bytes: vec![1],
        }),
        Column::Sorted(SortedOrdsColumn {
            dict: vec![b"b".to_vec(), b"a".to_vec()],
            docs: vec![],
            ords: vec![],
        }),
        Column::Sorted(SortedOrdsColumn {
            dict: vec![],
            docs: vec![0],
            ords: vec![0],
        }),
        Column::Sorted(SortedOrdsColumn {
            dict: vec![vec![]],
            docs: vec![0],
            ords: vec![-1],
        }),
        Column::Sorted(SortedOrdsColumn {
            dict: vec![],
            docs: vec![0],
            ords: vec![],
        }),
        Column::SortedNumeric(SortedNumericColumn {
            docs: vec![0],
            counts: vec![],
            values: vec![],
        }),
        Column::SortedNumeric(SortedNumericColumn {
            docs: vec![0],
            counts: vec![0],
            values: vec![],
        }),
        Column::SortedNumeric(SortedNumericColumn {
            docs: vec![0],
            counts: vec![2],
            values: vec![1],
        }),
        Column::SortedNumeric(SortedNumericColumn {
            docs: vec![],
            counts: vec![],
            values: vec![1],
        }),
        Column::SortedNumeric(SortedNumericColumn {
            docs: vec![0],
            counts: vec![2],
            values: vec![2, 1],
        }),
        Column::SortedSet(SortedSetOrdsColumn {
            dict: vec![vec![]],
            docs: vec![0],
            counts: vec![2],
            ords: vec![0, 0],
        }),
    ];
    for column in bad {
        assert!(validate_column(&column, 4).is_err(), "{column:?}");
    }
    validate_column(
        &Column::SortedNumeric(SortedNumericColumn {
            docs: vec![0],
            counts: vec![2],
            values: vec![1, 1],
        }),
        1,
    )
    .unwrap();
}
#[test]
fn native_consumer_lifecycle_shape_and_field_errors() {
    let infos = infos(DocValuesType::Numeric);
    let f = &infos.fields[0];
    let mut format = Lucene90DocValuesFormat::default();
    error(
        format.fields_consumer(state(&infos, -1)),
        "negative maximum",
    );
    let good = OwnedDocValuesProducer::new(BTreeMap::from([(0, Column::Numeric(vec![(0, 7)]))]));
    let mut consumer = format.fields_consumer(state(&infos, 1)).unwrap();
    assert!(consumer.output().is_none());
    error(consumer.add_binary_field(f, &good), "incompatible");
    error(
        consumer.add_numeric_field(
            &FieldInfo::new("wrong", 0).with_doc_values(
                DocValuesType::Numeric,
                DocValuesSkipIndexType::None,
                -1,
            ),
            &good,
        ),
        "incompatible",
    );
    let bad = OwnedDocValuesProducer::new(BTreeMap::from([(0, Column::Numeric(vec![(2, 7)]))]));
    error(consumer.add_numeric_field(f, &bad), "document order");
    consumer.add_numeric_field(f, &good).unwrap();
    error(consumer.add_numeric_field(f, &good), "duplicate");
    consumer.close().unwrap();
    let output = consumer.output().unwrap().clone();
    consumer.close().unwrap();
    assert_eq!(consumer.output().unwrap(), &output);
    error(consumer.add_numeric_field(f, &good), "closed");
    let mut producer = format
        .fields_producer(ReadState {
            files: &output,
            segment: SegmentReadState {
                max_doc: 1,
                segment_id: &[0; 16],
                segment_suffix: "",
                field_infos: &infos,
            },
        })
        .unwrap();
    assert_eq!(&*producer.get_numeric(f).unwrap(), &[(0, 7)]);
    assert!(producer.get_skipper(f).unwrap().is_none());
    error(producer.get_binary(f), "unsupported");
    error(producer.get_sorted(f), "unsupported");
    error(producer.get_sorted_numeric(f), "unsupported");
    error(producer.get_sorted_set(f), "unsupported");
    error(
        producer.get_numeric(&field(DocValuesType::Binary)),
        "incompatible",
    );
    producer.check_integrity().unwrap();
    producer.close().unwrap();
    producer.close().unwrap();
    error(producer.check_integrity(), "closed");
    error(producer.get_numeric(f), "closed");
    for extension in ["dvm", "dvd", "dvs"] {
        let mut damaged = output.clone();
        damaged.remove(extension);
        error(
            format.fields_producer(ReadState {
                files: &damaged,
                segment: SegmentReadState {
                    max_doc: 1,
                    segment_id: &[0; 16],
                    segment_suffix: "",
                    field_infos: &infos,
                },
            }),
            "missing",
        );
    }
    let mut damaged = output.clone();
    damaged.get_mut("dvm").unwrap()[8] ^= 1;
    assert!(format
        .fields_producer(ReadState {
            files: &damaged,
            segment: SegmentReadState {
                max_doc: 1,
                segment_id: &[0; 16],
                segment_suffix: "",
                field_infos: &infos
            }
        })
        .is_err());
    let mut producer = format
        .fields_producer(ReadState {
            files: &output,
            segment: SegmentReadState {
                max_doc: 1,
                segment_id: &[0; 16],
                segment_suffix: "",
                field_infos: &infos,
            },
        })
        .unwrap();
    producer.files.get_mut("dvd").unwrap()[8] ^= 1;
    assert!(producer.check_integrity().is_err());
    error(checksum(&[]), "missing");
}
#[test]
fn mapped_views_validate_identity_maps_masks_and_keep_unpruned_terms_without_live_docs() {
    let infos = infos(DocValuesType::Sorted);
    let f = &infos.fields[0];
    let producer = OwnedDocValuesProducer::new(BTreeMap::from([(
        0,
        Column::Sorted(SortedOrdsColumn {
            dict: vec![b"deleted".to_vec(), b"live".to_vec()],
            docs: vec![0, 1],
            ords: vec![0, 1],
        }),
    )]));
    for live in [None, Some(&[false, true][..])] {
        let sources = [MergeSource {
            producer: Some(&producer),
            field_infos: &infos,
            max_doc: 2,
            doc_map: &[-1, 0],
            live_docs: live,
        }];
        let merge = MergeState {
            sources: &sources,
            field_infos: &infos,
            check_aborted: &|| Ok(()),
        };
        let mut mapped = MappedProducer::new(f, &merge);
        mapped.ordinals = Some(ordinal_map(f, &merge).unwrap());
        let column = mapped.get_sorted(f).unwrap();
        assert_eq!(column.docs, vec![0]);
        assert_eq!(column.dict.len(), if live.is_some() { 1 } else { 2 });
        assert_eq!(column.ords, vec![if live.is_some() { 0 } else { 1 }]);
        error(mapped.get_sorted(&f.clone()), "identity");
        error(mapped.get_numeric(f), "unsupported");
        error(mapped.get_binary(f), "unsupported");
        error(mapped.get_sorted_numeric(f), "unsupported");
        error(mapped.get_sorted_set(f), "unsupported");
        error(mapped.check_integrity(), "unsupported");
        error(mapped.close(), "unsupported");
    }
    for map in [&[][..], &[-2, 0][..]] {
        let sources = [MergeSource {
            producer: Some(&producer),
            field_infos: &infos,
            max_doc: 2,
            doc_map: map,
            live_docs: None,
        }];
        let merge = MergeState {
            sources: &sources,
            field_infos: &infos,
            check_aborted: &|| Ok(()),
        };
        assert!(MappedProducer::new(f, &merge).get_sorted(f).is_err());
    }
    let sources = [MergeSource {
        producer: Some(&producer),
        field_infos: &infos,
        max_doc: 1,
        doc_map: &[0, 1],
        live_docs: Some(&[true]),
    }];
    let merge = MergeState {
        sources: &sources,
        field_infos: &infos,
        check_aborted: &|| Ok(()),
    };
    assert!(ordinal_map(f, &merge).is_err());
    assert!(MappedProducer::new(f, &merge).get_sorted(f).is_err());
}
#[test]
fn default_getters_and_no_value_fields_are_typed_errors() {
    struct Empty;
    impl DocValuesProducer for Empty {
        fn check_integrity(&self) -> Result<()> {
            Ok(())
        }
        fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }
    let empty = Empty;
    let f = field(DocValuesType::Numeric);
    error(empty.get_numeric(&f), "unsupported");
    error(empty.get_binary(&f), "unsupported");
    error(empty.get_sorted(&f), "unsupported");
    error(empty.get_sorted_numeric(&f), "unsupported");
    error(empty.get_sorted_set(&f), "unsupported");
    assert!(empty.get_skipper(&f).unwrap().is_none());
    error(
        read_column(&empty, &FieldInfo::new("f", 0)),
        "no doc values",
    );
    error(
        build_column(DocValuesType::None, Vec::new(), Vec::new()),
        "no doc values",
    );
}
#[test]
fn merge_skips_absent_incompatible_sources_and_plain_fields() {
    let infos = FieldInfos::new(vec![
        field(DocValuesType::Numeric),
        FieldInfo::new("plain", 1),
    ])
    .unwrap();
    let missing = FieldInfos::new(Vec::new()).unwrap();
    let incompatible = self::infos(DocValuesType::Binary);
    let producer =
        OwnedDocValuesProducer::new(BTreeMap::from([(0, Column::Numeric(vec![(0, 7)]))]));
    let sources = [
        MergeSource {
            producer: None,
            field_infos: &infos,
            max_doc: 1,
            doc_map: &[0],
            live_docs: None,
        },
        MergeSource {
            producer: Some(&producer),
            field_infos: &missing,
            max_doc: 1,
            doc_map: &[0],
            live_docs: None,
        },
        MergeSource {
            producer: Some(&producer),
            field_infos: &incompatible,
            max_doc: 1,
            doc_map: &[0],
            live_docs: None,
        },
        MergeSource {
            producer: Some(&producer),
            field_infos: &infos,
            max_doc: 1,
            doc_map: &[0],
            live_docs: None,
        },
    ];
    let mut format = CrankyDocValuesFormat::new(
        Lucene90DocValuesFormat::default(),
        Rc::new(RefCell::new(JavaRandom::new(42))),
    );
    assert_eq!(format.name(), "Lucene90");
    assert_eq!(format.delegate().name(), "Lucene90");
    assert_eq!(format.delegate_mut().name(), "Lucene90");
    let mut consumer = format.fields_consumer(state(&infos, 1)).unwrap();
    consumer
        .merge(&MergeState {
            sources: &sources,
            field_infos: &infos,
            check_aborted: &|| Ok(()),
        })
        .unwrap();
    consumer.close().unwrap();
    let output = consumer.delegate_mut().output().unwrap().clone();
    assert_eq!(consumer.delegate().output(), Some(&output));
    let restored = format
        .fields_producer(ReadState {
            files: &output,
            segment: SegmentReadState {
                max_doc: 1,
                segment_id: &[0; 16],
                segment_suffix: "",
                field_infos: &infos,
            },
        })
        .unwrap();
    assert_eq!(&*restored.get_numeric(&infos.fields[0]).unwrap(), &[(0, 7)]);
    assert_eq!(restored.columns.len(), 1);
    let binary = field(DocValuesType::Binary);
    let producer = OwnedDocValuesProducer::new(BTreeMap::from([(
        0,
        Column::Binary(BinaryColumn::default()),
    )]));
    error(producer.get_numeric(&binary), "unsupported numeric");
}

#[test]
fn native_read_rejects_negative_max_doc_and_valid_but_mismatched_versions() {
    let infos = FieldInfos::new(Vec::new()).unwrap();
    let mut format = Lucene90DocValuesFormat::default();
    let mut consumer = format.fields_consumer(state(&infos, 0)).unwrap();
    consumer.close().unwrap();
    let output = consumer.output().unwrap().clone();
    let segment = SegmentReadState {
        max_doc: 0,
        segment_id: &[0; 16],
        segment_suffix: "",
        field_infos: &infos,
    };
    error(
        format.fields_producer(ReadState {
            files: &output,
            segment: SegmentReadState {
                max_doc: -1,
                ..segment
            },
        }),
        "negative maximum",
    );
    for (extension, codec) in [
        ("dvd", "Lucene90DocValuesData"),
        ("dvs", "Lucene90DocValuesSkipIndex"),
    ] {
        let mut damaged = output.clone();
        let mut bytes = Vec::new();
        codec_util::write_index_header(&mut bytes, codec, 1, &[0; 16], "");
        codec_util::write_footer(&mut bytes);
        damaged.insert(extension.to_string(), bytes);
        error(
            format.fields_producer(ReadState {
                files: &damaged,
                segment,
            }),
            "version mismatch",
        );
    }
}
