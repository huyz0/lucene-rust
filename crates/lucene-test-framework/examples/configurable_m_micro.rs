//! Same selected ConfigurableM components, bytes and inputs as the Java twin.
use std::hint::black_box;
use std::time::{Duration, Instant};

use lucene_codecs::field_infos::{FieldInfo, FieldInfos, VectorEncoding, VectorSimilarityFunction};
use lucene_codecs::hnsw::{HnswGraphView, KnnCollector};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_codecs::vectors::{FieldVectorData, FlatVectorsField};
use lucene_test_framework::codecs::asserting_knn_vectors::{
    BitsAcceptDocs, KnnVectorsReader, VectorValues,
};
use lucene_test_framework::codecs::configurable_m::{
    ConfigurableMCodec, Files, Lucene99KnnVectorsReader, ReadState,
};

const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const EXTENSIONS: [&str; 4] = ["vec", "vemf", "vex", "vem"];
#[derive(Default)]
struct Digest(i64);
impl Digest {
    fn mix(&mut self, value: i64) {
        self.0 = self.0.wrapping_mul(31).wrapping_add(value);
    }
    fn bytes(&mut self, values: &[u8]) {
        self.mix(values.len() as i64);
        for &value in values {
            self.mix(i64::from(value));
        }
    }
}
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
fn files(buffers: &[Vec<u8>; 4]) -> Files<'_> {
    Files {
        vec: &buffers[0],
        vemf: &buffers[1],
        vex: &buffers[2],
        vem: &buffers[3],
    }
}
fn open<'a>(
    codec: &ConfigurableMCodec,
    infos: &'a FieldInfos,
    max_doc: i32,
    files: Files<'a>,
) -> Lucene99KnnVectorsReader<'a> {
    codec
        .knn_vectors_format()
        .fields_reader(ReadState {
            files,
            segment: SegmentReadState {
                max_doc,
                segment_id: &ID,
                segment_suffix: "s",
                field_infos: infos,
            },
        })
        .unwrap()
}
fn construct() -> i64 {
    let mut digest = Digest::default();
    for _ in 0..32 {
        for m in [128, 4, 1, 512] {
            let codec = if m == 128 {
                ConfigurableMCodec::new()
            } else {
                ConfigurableMCodec::with_max_conn(m).unwrap()
            };
            let codec = black_box(codec);
            let format = codec.knn_vectors_format();
            digest.bytes(codec.name().as_bytes());
            digest.bytes(format.name().as_bytes());
            digest.mix(i64::from(std::ptr::eq(format, codec.knn_vectors_format())));
        }
    }
    digest.0
}
fn read(infos: &FieldInfos, reader: &Lucene99KnnVectorsReader<'_>) -> i64 {
    reader.check_integrity().unwrap();
    let mut digest = Digest::default();
    for info in &infos.fields {
        digest.bytes(info.name.as_bytes());
        digest.mix(i64::from(info.number));
        if info.vector_encoding == VectorEncoding::Float32 {
            let values = reader.float_values(&info.name).unwrap().unwrap();
            digest.mix(values.dimension() as i64);
            digest.mix(i64::from(values.size()));
            for ord in 0..values.size() {
                digest.mix(i64::from(values.ord_to_doc(ord).unwrap()));
                digest.mix(i64::from(ord));
                for value in values.vector(ord).unwrap() {
                    digest.mix(i64::from(value.to_bits() as i32));
                }
            }
        } else {
            let values = reader.byte_values(&info.name).unwrap().unwrap();
            digest.mix(values.dimension() as i64);
            digest.mix(i64::from(values.size()));
            for ord in 0..values.size() {
                digest.mix(i64::from(values.ord_to_doc(ord).unwrap()));
                digest.mix(i64::from(ord));
                for value in values.vector(ord).unwrap() {
                    digest.mix(i64::from(value as i8));
                }
            }
        }
        if let Some(graph) = reader.graph(&info.name).unwrap() {
            digest.mix(i64::from(graph.size()));
            digest.mix(i64::from(graph.num_levels()));
            digest.mix(i64::from(graph.entry_node()));
            for level in 0..graph.num_levels() {
                digest.mix(i64::from(level));
                for node in graph.sorted_nodes_on_level(level).unwrap() {
                    digest.mix(i64::from(node));
                    let mut neighbors = Vec::new();
                    graph.neighbors_into(level, node, &mut neighbors).unwrap();
                    for &neighbor in &neighbors {
                        digest.mix(i64::from(neighbor));
                    }
                    digest.mix(neighbors.len() as i64);
                }
            }
        } else {
            digest.mix(0);
            digest.mix(0);
            digest.mix(0);
        }
    }
    digest.0
}
fn search(
    infos: &FieldInfos,
    max_doc: i32,
    reader: &Lucene99KnnVectorsReader<'_>,
    filtered: bool,
    mask: &[bool],
) -> i64 {
    let mut digest = Digest::default();
    let accept = BitsAcceptDocs::new(filtered.then_some(mask), max_doc);
    for info in &infos.fields {
        for doc in [0, max_doc / 2, max_doc - 1] {
            let mut collector = KnnCollector::new(7, i32::MAX as u64);
            if info.vector_encoding == VectorEncoding::Float32 {
                reader
                    .search_float(&info.name, &floats(doc), &mut collector, &accept)
                    .unwrap();
            } else {
                reader
                    .search_byte(&info.name, &bytes(doc), &mut collector, &accept)
                    .unwrap();
            }
            digest.mix(collector.visited_count() as i64);
            digest.mix(i64::from(collector.early_terminated()));
            let hits = collector.top_docs();
            digest.mix(hits.len() as i64);
            for (doc, score) in hits {
                digest.mix(i64::from(doc));
                digest.mix(i64::from(score.to_bits() as i32));
            }
        }
    }
    digest.0
}
fn write(
    codec: &ConfigurableMCodec,
    infos: &FieldInfos,
    max_doc: i32,
    source: &[FlatVectorsField],
) -> lucene_test_framework::codecs::configurable_m::Output {
    // Match Java's per-operation addValue buffering from precomputed source
    // vectors. The batch adapter's contiguous copies remain an API difference.
    let buffered = source.to_vec();
    codec
        .knn_vectors_format()
        .flush(
            SegmentWriteState {
                max_doc,
                segment_id: &ID,
                segment_suffix: "s",
                field_infos: infos,
            },
            &buffered,
            42,
        )
        .unwrap()
}
fn write_digest(
    codec: &ConfigurableMCodec,
    infos: &FieldInfos,
    max_doc: i32,
    source: &[FlatVectorsField],
) -> i64 {
    let output = write(codec, infos, max_doc, source);
    let mut digest = Digest::default();
    for bytes in [&output.vec, &output.vemf, &output.vex, &output.vem] {
        digest.bytes(bytes);
    }
    digest.0
}
fn run(f: &mut impl FnMut() -> i64, millis: u64, expected: i64, units: u64) -> (f64, u64) {
    let start = Instant::now();
    let mut count = 0;
    loop {
        let value = f();
        assert_eq!(value, expected);
        black_box(value);
        count += units;
        if start.elapsed() >= Duration::from_millis(millis) {
            return (start.elapsed().as_nanos() as f64 / count as f64, count);
        }
    }
}
fn report(name: &str, warmup: u64, measure: u64, units: u64, mut f: impl FnMut() -> i64) {
    let expected = f();
    println!("#check\t{name}\t{expected}\t{units}");
    run(&mut f, warmup, expected, units);
    let (ns, count) = run(&mut f, measure, expected, units);
    println!("{name}\t{ns:.3}\t{count}");
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup = args.get(1).map_or(500, |s| s.parse().unwrap());
    let measure = args.get(2).map_or(750, |s| s.parse().unwrap());
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/testframework/data");
    report("configurable_m_construct", warmup, measure, 32, construct);
    let infos = fields();
    for (label, m, count) in [
        ("default_647", 128, 647),
        ("default_648", 128, 648),
        ("4_647", 4, 647),
        ("4_648", 4, 648),
    ] {
        let codec = if m == 128 {
            ConfigurableMCodec::new()
        } else {
            ConfigurableMCodec::with_max_conn(m).unwrap()
        };
        let max_doc = count * 2;
        let buffers = EXTENSIONS.map(|extension| {
            std::fs::read(root.join(format!("configurable_m_{label}.{extension}"))).unwrap()
        });
        let source = inputs(&infos, max_doc);
        let output = write(&codec, &infos, max_doc, &source);
        for (extension, (actual, expected)) in EXTENSIONS.iter().zip(
            [&output.vec, &output.vemf, &output.vex, &output.vem]
                .into_iter()
                .zip(&buffers),
        ) {
            assert!(
                actual == expected,
                "{label}.{extension}: native output differs from actual Java fixture"
            );
        }
        println!(
            "#memory\tconfigurable_m_{label}_serialized_payload_bytes\t{}",
            buffers.iter().map(Vec::len).sum::<usize>()
        );
        report(
            &format!("configurable_m_{label}_read"),
            warmup,
            measure,
            1,
            || {
                let mut reader = open(&codec, &infos, max_doc, files(&buffers));
                let digest = read(&infos, &reader);
                reader.close().unwrap();
                digest
            },
        );
        let mut reader = open(&codec, &infos, max_doc, files(&buffers));
        let mask: Vec<_> = (0..max_doc).map(|doc| doc % 3 != 0).collect();
        for filtered in [false, true] {
            let kind = if filtered {
                "search_filtered"
            } else {
                "search_unfiltered"
            };
            report(
                &format!("configurable_m_{label}_{kind}"),
                warmup,
                measure,
                1,
                || search(&infos, max_doc, &reader, filtered, &mask),
            );
        }
        reader.close().unwrap();
        report(
            &format!("configurable_m_{label}_write"),
            warmup,
            measure,
            1,
            || write_digest(&codec, &infos, max_doc, &source),
        );
    }
}
