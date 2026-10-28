//! Real Java Lucene99 float/byte vectors and HNSW search through asserting readers.
use lucene_codecs::field_infos::{FieldInfo, FieldInfos, VectorEncoding, VectorSimilarityFunction};
use lucene_codecs::hnsw::{HnswGraphView, KnnCollector};
use lucene_codecs::segment_state::SegmentReadState;
use lucene_search::reader::NO_MORE_DOCS;
use lucene_test_framework::codecs::asserting_knn_vectors::{
    AssertingKnnVectorsReader, BitsAcceptDocs, Files, KnnVectorsReader, Lucene99KnnVectorsReader,
    ReadState, VectorValues,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const VEC: &[u8] = include_bytes!("../../../fixtures/testframework/data/asserting_knn_vectors.vec");
const VEMF: &[u8] =
    include_bytes!("../../../fixtures/testframework/data/asserting_knn_vectors.vemf");
const VEX: &[u8] = include_bytes!("../../../fixtures/testframework/data/asserting_knn_vectors.vex");
const VEM: &[u8] = include_bytes!("../../../fixtures/testframework/data/asserting_knn_vectors.vem");
fn fields() -> FieldInfos {
    FieldInfos::new(vec![
        FieldInfo::new("float", 0).with_vectors(
            2,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        ),
        FieldInfo::new("byte", 1).with_vectors(
            2,
            VectorEncoding::Byte,
            VectorSimilarityFunction::Euclidean,
        ),
    ])
    .unwrap()
}
fn files() -> Files<'static> {
    Files {
        vec: VEC,
        vemf: VEMF,
        vex: VEX,
        vem: VEM,
    }
}
fn state<'a>(fields: &'a FieldInfos, files: Files<'a>) -> ReadState<'a> {
    ReadState {
        files,
        segment: SegmentReadState {
            max_doc: 16,
            segment_id: &ID,
            segment_suffix: "s",
            field_infos: fields,
        },
    }
}

#[test]
fn native_values_all_graph_neighbors_and_filtered_searches_match_actual_java() {
    let fields = fields();
    let raw = Lucene99KnnVectorsReader::open(state(&fields, files())).unwrap();
    let mut reader = AssertingKnnVectorsReader::new(raw, &fields);
    reader.check_integrity().unwrap();
    let manifest =
        include_str!("../../../fixtures/testframework/data/asserting_knn_vectors_values.tsv");
    let mut rows = 0;
    let mut searches = 0;
    for row in manifest.lines() {
        let p: Vec<_> = row.split('\t').collect();
        let name = p[1];
        match p[0] {
            "value" => {
                let doc = p[2].parse().unwrap();
                let ord = p[3].parse().unwrap();
                if name == "float" {
                    let v = reader.float_values(name).unwrap().unwrap();
                    assert_eq!(v.dimension(), 2);
                    assert_eq!(v.ord_to_doc(ord).unwrap(), doc);
                    assert_eq!(
                        v.vector(ord)
                            .unwrap()
                            .into_iter()
                            .map(|v| v.to_bits() as i32)
                            .collect::<Vec<_>>(),
                        p[4..]
                            .iter()
                            .map(|v| v.parse::<i32>().unwrap())
                            .collect::<Vec<_>>()
                    );
                } else {
                    let v = reader.byte_values(name).unwrap().unwrap();
                    assert_eq!(v.dimension(), 2);
                    assert_eq!(v.ord_to_doc(ord).unwrap(), doc);
                    assert_eq!(
                        v.vector(ord)
                            .unwrap()
                            .into_iter()
                            .map(|v| v as i8)
                            .collect::<Vec<_>>(),
                        p[4..]
                            .iter()
                            .map(|v| v.parse::<i8>().unwrap())
                            .collect::<Vec<_>>()
                    );
                }
            }
            "graph" => {
                let g = reader.graph(name).unwrap().unwrap();
                assert_eq!(
                    [g.size(), g.num_levels(), g.entry_node()],
                    p[2..]
                        .iter()
                        .map(|v| v.parse().unwrap())
                        .collect::<Vec<_>>()
                        .as_slice()
                );
            }
            "neighbors" => {
                let g = reader.graph(name).unwrap().unwrap();
                let mut neighbors = Vec::new();
                g.neighbors_into(p[2].parse().unwrap(), p[3].parse().unwrap(), &mut neighbors)
                    .unwrap();
                assert_eq!(
                    neighbors,
                    p[4..]
                        .iter()
                        .map(|v| v.parse::<i32>().unwrap())
                        .collect::<Vec<_>>()
                );
            }
            "search" => {
                let q = p[2].parse::<i32>().unwrap();
                let filtered = p[3] == "true";
                let mask: Vec<_> = (0..16).map(|doc| doc % 3 != 0).collect();
                let accept = BitsAcceptDocs::new(filtered.then_some(mask.as_slice()), 16);
                let mut collector = KnnCollector::new(4, 100);
                if name == "float" {
                    reader
                        .search_float(
                            name,
                            &[q as f32 * 0.25, 16.0 - q as f32],
                            &mut collector,
                            &accept,
                        )
                        .unwrap();
                } else {
                    reader
                        .search_byte(
                            name,
                            &[(q - 8) as i8 as u8, (q * 2) as u8],
                            &mut collector,
                            &accept,
                        )
                        .unwrap();
                }
                assert_eq!(
                    collector.visited_count(),
                    p[4].parse::<u64>().unwrap(),
                    "{row}"
                );
                assert_eq!(
                    collector.early_terminated(),
                    p[5] == "GREATER_THAN_OR_EQUAL_TO"
                );
                let hits: Vec<_> = collector
                    .top_docs()
                    .into_iter()
                    .map(|(doc, score)| format!("{doc}:{}", score.to_bits() as i32))
                    .collect();
                assert_eq!(hits, p[6..], "{row}");
                searches += 1;
            }
            _ => panic!("unknown manifest row {row}"),
        }
        rows += 1;
    }
    assert!(rows > 50);
    assert_eq!(searches, 12);
    let f = reader.float_values("float").unwrap().unwrap();
    let mut it = f.iterator().unwrap();
    assert_eq!(it.doc_id(), -1);
    assert_eq!(it.cost(), 11);
    assert_eq!(it.next_doc().unwrap(), 0);
    assert_eq!(it.advance(5).unwrap(), 5);
    while it.next_doc().unwrap() != NO_MORE_DOCS {}
    assert_eq!(it.next_doc().unwrap(), NO_MORE_DOCS);
    drop(it);
    drop(f);
    let b = reader.byte_values("byte").unwrap().unwrap();
    let mut it = b.iterator().unwrap();
    assert_eq!(it.advance(3).unwrap(), 3);
    assert_eq!(it.cost(), 12);
    drop(it);
    drop(b);
    for fi in &fields.fields {
        let sizes = reader.off_heap_byte_size(fi);
        assert_eq!(sizes.len(), 2);
        assert!(sizes["vec"] > 0);
        assert!(sizes["vex"] > 0);
    }
    let mut merged = reader.get_merge_instance().unwrap().unwrap();
    merged.float_values("float").unwrap();
    merged.byte_values("byte").unwrap();
    merged.finish_merge().unwrap();
    reader.close().unwrap();
    reader.close().unwrap();
}

#[test]
fn native_lifecycle_checks_metadata_before_delegate_and_merge_handles_close_independently() {
    let fields = fields();
    let mut raw = Lucene99KnnVectorsReader::open(state(&fields, files())).unwrap();
    let mut merged = raw.get_merge_instance().unwrap().unwrap();
    raw.close().unwrap();
    raw.close().unwrap();
    assert!(raw.check_integrity().is_err());
    assert!(raw.float_values("float").is_err());
    assert!(raw.byte_values("byte").is_err());
    assert!(raw.graph("float").is_err());
    assert!(raw.get_merge_instance().is_err());
    assert!(raw.finish_merge().is_err());
    let accept = BitsAcceptDocs::new(None, 16);
    let mut c = KnnCollector::new(1, 100);
    assert!(raw
        .search_float("float", &[0., 1.], &mut c, &accept)
        .is_err());
    assert!(raw.search_byte("byte", &[0, 1], &mut c, &accept).is_err());
    assert_eq!(merged.float_values("float").unwrap().unwrap().size(), 11);
    assert_eq!(merged.byte_values("byte").unwrap().unwrap().size(), 12);
    merged.finish_merge().unwrap();
    merged.check_integrity().unwrap();
    merged.close().unwrap();
    let raw = Lucene99KnnVectorsReader::open(state(&fields, files())).unwrap();
    let mut asserted = AssertingKnnVectorsReader::new(raw, &fields);
    assert!(asserted.delegate().graph("float").unwrap().is_some());
    assert!(asserted.delegate_mut().graph("float").unwrap().is_some());
    for field in ["missing", "byte"] {
        assert!(catch_unwind(AssertUnwindSafe(|| asserted.float_values(field))).is_err());
    }
    assert!(catch_unwind(AssertUnwindSafe(|| asserted.byte_values("float"))).is_err());
    let bad = BitsAcceptDocs::new(Some(&[true]), 1);
    assert!(asserted
        .search_float("float", &[0., 1.], &mut c, &bad)
        .unwrap_err()
        .to_string()
        .contains("too short"));
    assert!(asserted
        .search_float("float", &[0.], &mut c, &accept)
        .is_err());
    assert!(asserted.search_byte("byte", &[0], &mut c, &accept).is_err());
    asserted.close().unwrap();
}

#[test]
fn native_rejects_wrong_headers_lengths_metadata_and_integrity_corruption() {
    let fields = fields();
    for slot in 0..4 {
        let mut buffers = [VEC.to_vec(), VEMF.to_vec(), VEX.to_vec(), VEM.to_vec()];
        buffers[slot][0] ^= 1;
        let f = Files {
            vec: &buffers[0],
            vemf: &buffers[1],
            vex: &buffers[2],
            vem: &buffers[3],
        };
        assert!(Lucene99KnnVectorsReader::open(state(&fields, f)).is_err());
        let mut buffers = [VEC.to_vec(), VEMF.to_vec(), VEX.to_vec(), VEM.to_vec()];
        buffers[slot].truncate(8);
        let f = Files {
            vec: &buffers[0],
            vemf: &buffers[1],
            vex: &buffers[2],
            vem: &buffers[3],
        };
        assert!(Lucene99KnnVectorsReader::open(state(&fields, f)).is_err());
    }
    let mut id_state = state(&fields, files());
    let wrong = [99; 16];
    id_state.segment.segment_id = &wrong;
    assert!(Lucene99KnnVectorsReader::open(id_state).is_err());
    let mut suffix_state = state(&fields, files());
    suffix_state.segment.segment_suffix = "bad";
    assert!(Lucene99KnnVectorsReader::open(suffix_state).is_err());
    let missing = FieldInfos::new(vec![fields.fields[0].clone()]).unwrap();
    assert!(Lucene99KnnVectorsReader::open(state(&missing, files())).is_err());
    let mut changed = fields.fields.clone();
    changed[0].vector_dimension = 3;
    let bad = FieldInfos::new(changed).unwrap();
    assert!(Lucene99KnnVectorsReader::open(state(&bad, files())).is_err());
    let mut metadata = String::new();
    for index in 0..2 {
        let mut changed = fields.fields.clone();
        changed[index].vector_similarity_function = VectorSimilarityFunction::DotProduct;
        let bad = FieldInfos::new(changed).unwrap();
        assert!(Lucene99KnnVectorsReader::open(state(&bad, files()))
            .err()
            .unwrap()
            .to_string()
            .contains("metadata"));
        metadata.push_str(&format!(
            "{}\tsimilarity\tmetadata-error\n",
            fields.fields[index].name
        ));
    }
    assert_eq!(
        metadata,
        include_str!("../../../fixtures/testframework/data/asserting_knn_vectors_metadata.tsv")
    );
    // As Java, the flat `.vec` data's CRC is computed in checkIntegrity, not
    // on open (open checks its footer structure); the metadata and graph
    // files are still verified on open.
    for slot in 0..4 {
        let mut buffers = [VEC.to_vec(), VEMF.to_vec(), VEX.to_vec(), VEM.to_vec()];
        let position = buffers[slot].len() - 17;
        buffers[slot][position] ^= 1;
        let f = Files {
            vec: &buffers[0],
            vemf: &buffers[1],
            vex: &buffers[2],
            vem: &buffers[3],
        };
        let error = match Lucene99KnnVectorsReader::open(state(&fields, f)) {
            Ok(reader) if slot == 0 => reader.check_integrity().unwrap_err(),
            Ok(_) => panic!("slot {slot} opened despite a corrupt checksum"),
            Err(e) => e,
        };
        assert!(
            error.to_string().contains("checksum"),
            "slot {slot}: {error}"
        );
    }
}

#[test]
fn native_search_matches_java_accept_callback_and_query_error_order() {
    use lucene_search::reader::DocIdSetIterator;
    use lucene_test_framework::codecs::asserting_knn_vectors::AcceptDocs;
    use std::cell::RefCell;
    use std::io;
    struct RecordingAccept {
        calls: RefCell<Vec<&'static str>>,
        fail_cost: bool,
    }
    impl AcceptDocs for RecordingAccept {
        fn cost(&self) -> io::Result<i32> {
            self.calls.borrow_mut().push("cost");
            if self.fail_cost {
                Err(io::Error::other("cost failure"))
            } else {
                Ok(16)
            }
        }
        fn bits(&self) -> io::Result<Option<&[bool]>> {
            self.calls.borrow_mut().push("bits");
            Ok(None)
        }
        fn iterator(&self) -> io::Result<Box<dyn DocIdSetIterator + '_>> {
            panic!("native search must use bits, not iterator")
        }
    }
    let fields = fields();
    let reader = AssertingKnnVectorsReader::new(
        Lucene99KnnVectorsReader::open(state(&fields, files())).unwrap(),
        &fields,
    );
    let mut actual = String::new();
    for float in [true, false] {
        let name = if float { "float" } else { "byte" };
        let accept = RecordingAccept {
            calls: RefCell::new(Vec::new()),
            fail_cost: false,
        };
        let mut collector = KnnCollector::new(4, 100);
        if float {
            reader
                .search_float("float", &[0., 1.], &mut collector, &accept)
                .unwrap();
        } else {
            reader
                .search_byte("byte", &[0, 1], &mut collector, &accept)
                .unwrap();
        }
        assert_eq!(*accept.calls.borrow(), ["cost", "bits"]);
        actual.push_str(&format!(
            "{name}\tnormal\tok\t{}\n",
            accept.calls.borrow().join(",")
        ));
        accept.calls.borrow_mut().clear();
        let error = if float {
            reader.search_float("float", &[0.], &mut collector, &accept)
        } else {
            reader.search_byte("byte", &[0], &mut collector, &accept)
        };
        assert!(error.is_err());
        assert!(
            accept.calls.borrow().is_empty(),
            "query dimensions must fail before accept callbacks"
        );
        actual.push_str(&format!("{name}\tquery-error\tquery-error\t\n"));
        let mut zero = KnnCollector::new(0, 100);
        if float {
            reader
                .search_float("float", &[0.], &mut zero, &accept)
                .unwrap();
        } else {
            reader
                .search_byte("byte", &[0], &mut zero, &accept)
                .unwrap();
        }
        assert!(
            accept.calls.borrow().is_empty(),
            "k=0 skips query construction and accept callbacks"
        );
        actual.push_str(&format!("{name}\tzero-k\tok\t\n"));
        let accept = RecordingAccept {
            calls: RefCell::new(Vec::new()),
            fail_cost: true,
        };
        let error = if float {
            reader.search_float("float", &[0., 1.], &mut collector, &accept)
        } else {
            reader.search_byte("byte", &[0, 1], &mut collector, &accept)
        };
        assert_eq!(error.unwrap_err().to_string(), "cost failure");
        assert_eq!(*accept.calls.borrow(), ["cost"]);
        actual.push_str(&format!(
            "{name}\tcost-error\tio:cost failure\t{}\n",
            accept.calls.borrow().join(",")
        ));
    }
    assert_eq!(
        actual,
        include_str!("../../../fixtures/testframework/data/asserting_knn_vectors_native_order.tsv")
    );
}
