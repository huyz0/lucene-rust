//! Actual Java ConfigurableMCodec bytes, graph threshold and direct callbacks.
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use lucene_codecs::codec::Lucene104Codec;
use lucene_codecs::field_infos::{FieldInfo, FieldInfos, VectorEncoding, VectorSimilarityFunction};
use lucene_codecs::hnsw::{HnswGraphView, KnnCollector};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_codecs::vectors::{FieldVectorData, FlatVectorsField};
use lucene_test_framework::codecs::asserting_knn_vectors::{
    BitsAcceptDocs, KnnVectorsReader, VectorValues,
};
use lucene_test_framework::codecs::configurable_m::{
    CodecDelegate, ComponentKind, ConfigurableMCodec, Files, ReadState,
};

const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const ACCESSORS: [(ComponentKind, &str); 10] = [
    (ComponentKind::DocValues, "docValuesFormat"),
    (ComponentKind::FieldInfos, "fieldInfosFormat"),
    (ComponentKind::LiveDocs, "liveDocsFormat"),
    (ComponentKind::Norms, "normsFormat"),
    (ComponentKind::Postings, "postingsFormat"),
    (ComponentKind::SegmentInfo, "segmentInfoFormat"),
    (ComponentKind::StoredFields, "storedFieldsFormat"),
    (ComponentKind::TermVectors, "termVectorsFormat"),
    (ComponentKind::Compound, "compoundFormat"),
    (ComponentKind::Points, "pointsFormat"),
];

struct Probe {
    stable: Rc<()>,
    calls: RefCell<Vec<&'static str>>,
    fail: Cell<bool>,
    error: RefCell<Rc<String>>,
}
impl Probe {
    fn new() -> Self {
        Self {
            stable: Rc::new(()),
            calls: RefCell::new(Vec::new()),
            fail: Cell::new(false),
            error: RefCell::new(Rc::new(String::from("initial"))),
        }
    }
}
impl CodecDelegate for Probe {
    type Component = Rc<()>;
    type Error = Rc<String>;
    fn component(&self, kind: ComponentKind) -> Result<Self::Component, Self::Error> {
        let name = ACCESSORS
            .iter()
            .find(|(candidate, _)| *candidate == kind)
            .unwrap()
            .1;
        self.calls.borrow_mut().push(name);
        if self.fail.get() {
            return Err(Rc::clone(&self.error.borrow()));
        }
        // Java's default Points accessor constructs a fresh format each time.
        Ok(if kind == ComponentKind::Points {
            Rc::new(())
        } else {
            Rc::clone(&self.stable)
        })
    }
}

#[test]
fn java_configuration_and_each_delegated_identity_and_first_error() {
    let mut actual = String::new();
    assert_eq!(
        ConfigurableMCodec::default()
            .knn_vectors_format()
            .max_conn(),
        128
    );
    for m in [128, 4, 1, 512] {
        let codec = if m == 128 {
            ConfigurableMCodec::new()
        } else {
            ConfigurableMCodec::with_max_conn(m).unwrap()
        };
        let format = codec.knn_vectors_format();
        actual.push_str(&format!(
            "codec\t{m}\t{}\t{}\t{}\t{}\n",
            codec.name(),
            codec.delegate().name(),
            format.name(),
            std::ptr::eq(format, codec.knn_vectors_format())
        ));
        for (name, value) in [
            ("maxConn", format.max_conn().to_string()),
            ("beamWidth", format.beam_width().to_string()),
            ("numMergeWorkers", format.num_merge_workers().to_string()),
            (
                "tinySegmentsThreshold",
                format.tiny_segments_threshold().to_string(),
            ),
            ("writeVersion", format.write_version().to_string()),
            (
                "mergeExec",
                if format.merge_executor().is_none() {
                    "null"
                } else {
                    "executor"
                }
                .to_string(),
            ),
        ] {
            actual.push_str(&format!("setting\t{m}\t{name}\t{value}\n"));
        }
        if m != 128 && m != 4 {
            continue;
        }
        // This probe covers generic callback forwarding, not native default
        // factories: DefaultLucene104Codec has only per-field routing methods.
        let delegated = ConfigurableMCodec::with_delegate(m, Probe::new()).unwrap();
        for (kind, name) in ACCESSORS {
            let returned = delegated.component(kind).unwrap();
            let direct = delegated.delegate().component(kind).unwrap();
            actual.push_str(&format!(
                "delegate-identity\t{m}\t{name}\t{}\n",
                Rc::ptr_eq(&returned, &direct)
            ));
        }
        for (kind, name) in ACCESSORS {
            delegated.delegate().calls.borrow_mut().clear();
            let first = delegated.component(kind).unwrap();
            let second = delegated.component(kind).unwrap();
            actual.push_str(&format!(
                "forward\t{m}\t{name}\t{}\t{}\n",
                Rc::ptr_eq(&first, &second),
                delegated.delegate().calls.borrow().join(",")
            ));
            delegated.delegate().calls.borrow_mut().clear();
            *delegated.delegate().error.borrow_mut() = Rc::new(format!("delegate failure:{name}"));
            delegated.delegate().fail.set(true);
            let error = delegated.component(kind).unwrap_err();
            assert!(Rc::ptr_eq(&error, &delegated.delegate().error.borrow()));
            actual.push_str(&format!(
                "forward-error\t{m}\t{name}\t{error}\t{}\n",
                delegated.delegate().calls.borrow().join(",")
            ));
            delegated.delegate().fail.set(false);
        }
        delegated.delegate().calls.borrow_mut().clear();
        assert!(std::ptr::eq(
            delegated.knn_vectors_format(),
            delegated.knn_vectors_format()
        ));
        actual.push_str(&format!(
            "override-events\t{m}\t{}\n",
            delegated.delegate().calls.borrow().join(",")
        ));
    }
    for m in [-1, 0, 513, i32::MAX] {
        let error = ConfigurableMCodec::with_max_conn(m).err().unwrap();
        actual.push_str(&format!("invalid\t{m}\t{error}\n"));
    }
    assert_eq!(
        actual,
        include_str!("../../../fixtures/testframework/data/configurable_m_config.tsv")
    );
}

struct Fixture {
    m: i32,
    count: i32,
    files: Files<'static>,
    snapshot: &'static str,
    file_manifest: &'static str,
    errors: &'static str,
}
macro_rules! fixture {
    ($prefix:literal,$m:literal,$count:literal) => {
        Fixture {
            m: $m,
            count: $count,
            files: Files {
                vec: include_bytes!(concat!(
                    "../../../fixtures/testframework/data/",
                    $prefix,
                    ".vec"
                )),
                vemf: include_bytes!(concat!(
                    "../../../fixtures/testframework/data/",
                    $prefix,
                    ".vemf"
                )),
                vex: include_bytes!(concat!(
                    "../../../fixtures/testframework/data/",
                    $prefix,
                    ".vex"
                )),
                vem: include_bytes!(concat!(
                    "../../../fixtures/testframework/data/",
                    $prefix,
                    ".vem"
                )),
            },
            snapshot: include_str!(concat!(
                "../../../fixtures/testframework/data/",
                $prefix,
                ".tsv"
            )),
            file_manifest: include_str!(concat!(
                "../../../fixtures/testframework/data/",
                $prefix,
                "_files.tsv"
            )),
            errors: include_str!(concat!(
                "../../../fixtures/testframework/data/",
                $prefix,
                "_errors.tsv"
            )),
        }
    };
}
const FIXTURES: [Fixture; 4] = [
    fixture!("configurable_m_default_647", 128, 647),
    fixture!("configurable_m_default_648", 128, 648),
    fixture!("configurable_m_4_647", 4, 647),
    fixture!("configurable_m_4_648", 4, 648),
];
fn fields() -> FieldInfos {
    FieldInfos::new(
        [
            "float_dense",
            "byte_dense",
            "float_sparse",
            "byte_sparse",
            "float_empty",
            "byte_empty",
        ]
        .into_iter()
        .enumerate()
        .map(|(number, name)| {
            FieldInfo::new(name, number as i32).with_vectors(
                4,
                if name.starts_with("float") {
                    VectorEncoding::Float32
                } else {
                    VectorEncoding::Byte
                },
                VectorSimilarityFunction::Euclidean,
            )
        })
        .collect(),
    )
    .unwrap()
}
fn floats(doc: i32) -> [f32; 4] {
    [
        doc as f32 * 0.25,
        (doc % 17) as f32 * 0.5,
        (doc % 31) as f32 * 0.125,
        1.0,
    ]
}
fn bytes(doc: i32) -> [u8; 4] {
    [
        (doc % 127 - 63) as i8 as u8,
        (doc / 127 - 5) as i8 as u8,
        (doc * 17 % 127 - 63) as i8 as u8,
        (doc * 29 % 127 - 63) as i8 as u8,
    ]
}
fn inputs(infos: &FieldInfos, max_doc: i32) -> Vec<FlatVectorsField> {
    infos
        .fields
        .iter()
        .map(|info| {
            let docs: Vec<_> = if info.name.ends_with("empty") {
                Vec::new()
            } else {
                (0..max_doc)
                    .step_by(if info.name.ends_with("sparse") { 2 } else { 1 })
                    .collect()
            };
            let values = if info.vector_encoding == VectorEncoding::Float32 {
                FieldVectorData::Float32(docs.iter().flat_map(|&doc| floats(doc)).collect())
            } else {
                FieldVectorData::Byte(docs.iter().flat_map(|&doc| bytes(doc)).collect())
            };
            FlatVectorsField {
                field_number: info.number,
                dimension: 4,
                similarity: VectorSimilarityFunction::Euclidean,
                docs,
                values,
            }
        })
        .collect()
}
fn write_state(infos: &FieldInfos, max_doc: i32) -> SegmentWriteState<'_> {
    SegmentWriteState {
        max_doc,
        segment_id: &ID,
        segment_suffix: "s",
        field_infos: infos,
    }
}
fn read_state<'a>(infos: &'a FieldInfos, max_doc: i32, files: Files<'a>) -> ReadState<'a> {
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

#[test]
fn all_four_native_files_equal_java_and_every_value_arc_search_bit_and_visit_matches() {
    let infos = fields();
    for fixture in &FIXTURES {
        let max_doc = fixture.count * 2;
        let codec = ConfigurableMCodec::with_max_conn(fixture.m).unwrap();
        let format = codec.knn_vectors_format();
        let output = format
            .flush(write_state(&infos, max_doc), &inputs(&infos, max_doc), 42)
            .unwrap();
        let actual = output.files();
        let mut manifest = format!(
            "segment\t_cm\nsuffix\ts\nmax-doc\t{max_doc}\nsparse-count\t{}\n",
            fixture.count
        );
        for (name, rust, java) in [
            ("vec", actual.vec, fixture.files.vec),
            ("vemf", actual.vemf, fixture.files.vemf),
            ("vex", actual.vex, fixture.files.vex),
            ("vem", actual.vem, fixture.files.vem),
        ] {
            assert_eq!(
                rust.len(),
                java.len(),
                "M={},count={},file={name}",
                fixture.m,
                fixture.count
            );
            assert!(
                rust == java,
                "M={},count={},file={name},first difference={:?}",
                fixture.m,
                fixture.count,
                rust.iter().zip(java).position(|(r, j)| r != j)
            );
            manifest.push_str(&format!("file\t{name}\t{}\n", rust.len()));
        }
        assert_eq!(manifest, fixture.file_manifest);
        // Read the newly written Rust bytes and compare every Java snapshot
        // row. Exact file equality also covers graph node membership/order.
        let reader = format
            .fields_reader(read_state(&infos, max_doc, actual))
            .unwrap();
        reader.check_integrity().unwrap();
        let mut rows = 0;
        let mut searches = 0;
        for row in fixture.snapshot.lines() {
            let p: Vec<_> = row.split('\t').collect();
            let name = p[1];
            let float = name.starts_with("float");
            match p[0] {
                "field" => {
                    assert_eq!(
                        infos.field_by_name(name).unwrap().number,
                        p[2].parse::<i32>().unwrap()
                    );
                    let count = if float {
                        reader.float_values(name).unwrap().unwrap().size()
                    } else {
                        reader.byte_values(name).unwrap().unwrap().size()
                    };
                    assert_eq!(count, p[3].parse::<i32>().unwrap());
                    assert_eq!(count, p[4].parse::<i32>().unwrap());
                }
                "value" => {
                    let doc = p[2].parse().unwrap();
                    let ord = p[3].parse().unwrap();
                    if float {
                        let values = reader.float_values(name).unwrap().unwrap();
                        assert_eq!(values.dimension(), 4);
                        assert_eq!(values.ord_to_doc(ord).unwrap(), doc);
                        assert_eq!(
                            values
                                .vector(ord)
                                .unwrap()
                                .iter()
                                .map(|v| v.to_bits() as i32)
                                .collect::<Vec<_>>(),
                            p[4..]
                                .iter()
                                .map(|v| v.parse::<i32>().unwrap())
                                .collect::<Vec<_>>()
                        );
                    } else {
                        let values = reader.byte_values(name).unwrap().unwrap();
                        assert_eq!(values.dimension(), 4);
                        assert_eq!(values.ord_to_doc(ord).unwrap(), doc);
                        assert_eq!(
                            values
                                .vector(ord)
                                .unwrap()
                                .iter()
                                .map(|&v| v as i8)
                                .collect::<Vec<_>>(),
                            p[4..]
                                .iter()
                                .map(|v| v.parse::<i8>().unwrap())
                                .collect::<Vec<_>>()
                        );
                    }
                }
                "graph" => {
                    let graph = reader.graph(name).unwrap();
                    let actual =
                        graph.map_or([0, 0, 0], |g| [g.size(), g.num_levels(), g.entry_node()]);
                    assert_eq!(
                        actual.as_slice(),
                        p[2..]
                            .iter()
                            .map(|v| v.parse::<i32>().unwrap())
                            .collect::<Vec<_>>(),
                        "{row}"
                    );
                }
                "neighbors" => {
                    let graph = reader.graph(name).unwrap().unwrap();
                    let mut actual = Vec::new();
                    graph
                        .neighbors_into(p[2].parse().unwrap(), p[3].parse().unwrap(), &mut actual)
                        .unwrap();
                    assert_eq!(
                        actual,
                        p[4..]
                            .iter()
                            .map(|v| v.parse::<i32>().unwrap())
                            .collect::<Vec<_>>(),
                        "{row}"
                    );
                }
                "search" => {
                    let doc = p[2].parse().unwrap();
                    let mask: Vec<_> = (0..max_doc).map(|doc| doc % 3 != 0).collect();
                    let accept =
                        BitsAcceptDocs::new((p[3] == "true").then_some(mask.as_slice()), max_doc);
                    let mut collector = KnnCollector::new(7, i32::MAX as u64);
                    if float {
                        reader
                            .search_float(name, &floats(doc), &mut collector, &accept)
                            .unwrap();
                    } else {
                        reader
                            .search_byte(name, &bytes(doc), &mut collector, &accept)
                            .unwrap();
                    }
                    assert_eq!(
                        collector.visited_count(),
                        p[4].parse::<u64>().unwrap(),
                        "M={},count={}: {row}",
                        fixture.m,
                        fixture.count
                    );
                    assert_eq!(
                        collector.early_terminated(),
                        p[5] == "GREATER_THAN_OR_EQUAL_TO"
                    );
                    let hits: Vec<_> = collector
                        .top_docs()
                        .iter()
                        .map(|(doc, score)| format!("{doc}:{}", score.to_bits() as i32))
                        .collect();
                    assert_eq!(
                        hits,
                        p[6..],
                        "M={},count={}: {row}",
                        fixture.m,
                        fixture.count
                    );
                    searches += 1;
                }
                _ => panic!("unknown Java snapshot row: {row}"),
            }
            rows += 1;
        }
        assert!(rows > 5000);
        assert_eq!(searches, 36);
    }
}

#[test]
fn invalid_metadata_docs_shapes_and_nonfinite_vectors_fail_before_publication() {
    let infos = fields();
    let codec = ConfigurableMCodec::new();
    let format = codec.knn_vectors_format();
    assert!(format.flush(write_state(&infos, -1), &[], 42).is_err());
    let baseline = inputs(&infos, 2);
    let mut invalid = baseline.clone();
    invalid[0].field_number = 99;
    assert!(format.flush(write_state(&infos, 2), &invalid, 42).is_err());
    let mut invalid = baseline.clone();
    invalid.push(invalid[0].clone());
    assert!(format.flush(write_state(&infos, 2), &invalid, 42).is_err());
    for change in 0..3 {
        let mut invalid = baseline.clone();
        match change {
            0 => invalid[0].dimension = 3,
            1 => invalid[0].similarity = VectorSimilarityFunction::DotProduct,
            _ => invalid[0].values = FieldVectorData::Byte(vec![0; 8]),
        }
        assert!(format.flush(write_state(&infos, 2), &invalid, 42).is_err());
    }
    for docs in [vec![0, 0], vec![1, 0], vec![-1, 0], vec![0, 2]] {
        let mut invalid = baseline.clone();
        invalid[0].docs = docs;
        assert!(format.flush(write_state(&infos, 2), &invalid, 42).is_err());
    }
    for values in [vec![0.; 7], vec![f32::NAN; 8], vec![f32::INFINITY; 8]] {
        let mut invalid = baseline.clone();
        invalid[0].values = FieldVectorData::Float32(values);
        assert!(format.flush(write_state(&infos, 2), &invalid, 42).is_err());
    }
    let empty = format.flush(write_state(&infos, 0), &[], 42).unwrap();
    format
        .fields_reader(read_state(&infos, 0, empty.files()))
        .unwrap()
        .check_integrity()
        .unwrap();
    let tiny = format.flush(write_state(&infos, 2), &baseline, 42).unwrap();
    let reader = format
        .fields_reader(read_state(&infos, 2, tiny.files()))
        .unwrap();
    for info in &infos.fields {
        assert!(reader.graph(&info.name).unwrap().is_none());
    }
}

#[test]
fn java_wrong_id_error_and_native_crc_truncation_suffix_checks() {
    let infos = fields();
    for fixture in &FIXTURES {
        let codec = ConfigurableMCodec::with_max_conn(fixture.m).unwrap();
        let format = codec.knn_vectors_format();
        let max_doc = fixture.count * 2;
        assert_eq!(fixture.errors, "wrong-id\tCorruptIndexException\n");
        let wrong = [99; 16];
        let mut state = read_state(&infos, max_doc, fixture.files);
        state.segment.segment_id = &wrong;
        assert!(format.fields_reader(state).is_err());
        let mut state = read_state(&infos, max_doc, fixture.files);
        state.segment.segment_suffix = "bad";
        assert!(format.fields_reader(state).is_err());
        for slot in 0..4 {
            for truncate in [false, true] {
                let mut buffers = [
                    fixture.files.vec.to_vec(),
                    fixture.files.vemf.to_vec(),
                    fixture.files.vex.to_vec(),
                    fixture.files.vem.to_vec(),
                ];
                if truncate {
                    buffers[slot].truncate(8);
                } else {
                    let last_data = buffers[slot].len() - 17;
                    buffers[slot][last_data] ^= 1;
                }
                let files = Files {
                    vec: &buffers[0],
                    vemf: &buffers[1],
                    vex: &buffers[2],
                    vem: &buffers[3],
                };
                // As Java, a flipped `.vec` data byte is caught by
                // checkIntegrity, not on open.
                let opened = format.fields_reader(read_state(&infos, max_doc, files));
                let failed = match opened {
                    Ok(reader) if slot == 0 && !truncate => reader.check_integrity().is_err(),
                    Ok(_) => false,
                    Err(_) => true,
                };
                assert!(failed, "slot={slot},truncate={truncate}");
            }
        }
    }
}
