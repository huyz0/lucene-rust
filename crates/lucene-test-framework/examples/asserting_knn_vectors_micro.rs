//! Paired asserting-vector value acquisition, real reads/searches and merge lifecycle.
use lucene_codecs::field_infos::{FieldInfo, FieldInfos, VectorEncoding, VectorSimilarityFunction};
use lucene_codecs::hnsw::{HnswGraphView, KnnCollector};
use lucene_codecs::segment_state::SegmentReadState;
use lucene_test_framework::codecs::asserting_knn_vectors::{
    AssertingAcceptDocs, AssertingKnnVectorsReader, BitsAcceptDocs, Files, KnnVectorsReader,
    Lucene99KnnVectorsReader, ReadState, VectorValues,
};
use std::hint::black_box;
use std::time::{Duration, Instant};
const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
type Reader<'a> = AssertingKnnVectorsReader<'a, Lucene99KnnVectorsReader<'a>>;
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
fn open<'a>(fields: &'a FieldInfos, files: Files<'a>) -> Reader<'a> {
    AssertingKnnVectorsReader::new(
        Lucene99KnnVectorsReader::open(ReadState {
            files,
            segment: SegmentReadState {
                max_doc: 16,
                segment_id: &ID,
                segment_suffix: "s",
                field_infos: fields,
            },
        })
        .unwrap(),
        fields,
    )
}
#[derive(Default)]
struct Digest(i64);
impl Digest {
    fn mix(&mut self, v: i64) {
        self.0 = self.0.wrapping_mul(31).wrapping_add(v);
    }
}
fn read_digest(reader: &Reader<'_>, digest: &mut Digest) {
    reader.check_integrity().unwrap();
    for field in ["float", "byte"] {
        if field == "float" {
            let v = reader.float_values(field).unwrap().unwrap();
            digest.mix(i64::from(v.size()));
            digest.mix(v.dimension() as i64);
            let mut it = v.iterator().unwrap();
            let mut ord = 0;
            loop {
                let doc = it.next_doc().unwrap();
                if doc == i32::MAX {
                    break;
                }
                digest.mix(i64::from(doc));
                for n in v.vector(ord).unwrap() {
                    digest.mix(i64::from(n.to_bits() as i32));
                }
                ord += 1;
            }
        } else {
            let v = reader.byte_values(field).unwrap().unwrap();
            digest.mix(i64::from(v.size()));
            digest.mix(v.dimension() as i64);
            let mut it = v.iterator().unwrap();
            let mut ord = 0;
            loop {
                let doc = it.next_doc().unwrap();
                if doc == i32::MAX {
                    break;
                }
                digest.mix(i64::from(doc));
                for n in v.vector(ord).unwrap() {
                    digest.mix(i64::from(n as i8));
                }
                ord += 1;
            }
        }
        let graph = reader.graph(field).unwrap().unwrap();
        digest.mix(i64::from(graph.size()));
        digest.mix(i64::from(graph.num_levels()));
        digest.mix(i64::from(graph.entry_node()));
        for level in 0..graph.num_levels() {
            for node in graph.sorted_nodes_on_level(level).unwrap() {
                digest.mix(i64::from(node));
                let mut neighbors = Vec::new();
                graph.neighbors_into(level, node, &mut neighbors).unwrap();
                digest.mix(neighbors.len() as i64);
                for n in neighbors {
                    digest.mix(i64::from(n));
                }
            }
        }
    }
}
fn batch(name: &str, fields: &FieldInfos, files: Files<'_>, prepared: &Reader<'_>) -> i64 {
    let mut digest = Digest::default();
    for _ in 0..16 {
        match name {
            "values" => {
                let f = prepared.float_values("float").unwrap().unwrap();
                let b = prepared.byte_values("byte").unwrap().unwrap();
                digest.mix(i64::from(f.size()));
                digest.mix(f.dimension() as i64);
                digest.mix(i64::from(b.size()));
                digest.mix(b.dimension() as i64);
            }
            "read" => {
                let mut reader = open(fields, files);
                read_digest(&reader, &mut digest);
                reader.close().unwrap();
            }
            "search" => {
                for field in ["float", "byte"] {
                    for q in [0i32, 5, 15] {
                        for filtered in [false, true] {
                            let mask: Vec<_> = (0..16).map(|doc| doc % 3 != 0).collect();
                            let accept =
                                BitsAcceptDocs::new(filtered.then_some(mask.as_slice()), 16);
                            let mut c = KnnCollector::new(4, 100);
                            if field == "float" {
                                prepared
                                    .search_float(
                                        field,
                                        &[q as f32 * 0.25, 16.0 - q as f32],
                                        &mut c,
                                        &accept,
                                    )
                                    .unwrap();
                            } else {
                                prepared
                                    .search_byte(
                                        field,
                                        &[(q - 8) as i8 as u8, (q * 2) as u8],
                                        &mut c,
                                        &accept,
                                    )
                                    .unwrap();
                            }
                            digest.mix(c.visited_count() as i64);
                            digest.mix(i64::from(c.early_terminated()));
                            for (doc, score) in c.top_docs() {
                                digest.mix(i64::from(doc));
                                digest.mix(i64::from(score.to_bits() as i32));
                            }
                        }
                    }
                }
            }
            "merge_lifecycle" => {
                let mut merge = prepared.get_merge_instance().unwrap().unwrap();
                let f = merge.float_values("float").unwrap().unwrap();
                digest.mix(i64::from(f.size()));
                let b = merge.byte_values("byte").unwrap().unwrap();
                digest.mix(i64::from(b.size()));
                drop(f);
                drop(b);
                merge.finish_merge().unwrap();
            }
            _ => panic!("unknown case"),
        }
    }
    black_box(digest.0)
}
fn run(
    name: &str,
    fields: &FieldInfos,
    files: Files<'_>,
    reader: &Reader<'_>,
    ms: u64,
) -> (u64, Duration) {
    let expected = batch(name, fields, files, reader);
    let start = Instant::now();
    let mut operations = 0;
    loop {
        assert_eq!(batch(name, fields, files, reader), expected);
        operations += 16;
        if start.elapsed() >= Duration::from_millis(ms) {
            return (operations, start.elapsed());
        }
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup = args.get(1).map_or(1000, |s| s.parse().unwrap());
    let measure = args.get(2).map_or(1500, |s| s.parse().unwrap());
    let root = "fixtures/testframework/data/asserting_knn_vectors";
    let buffers: [Vec<u8>; 4] =
        ["vec", "vemf", "vex", "vem"].map(|ext| std::fs::read(format!("{root}.{ext}")).unwrap());
    let files = Files {
        vec: &buffers[0],
        vemf: &buffers[1],
        vex: &buffers[2],
        vem: &buffers[3],
    };
    let fields = fields();
    let mut reader = open(&fields, files);
    let wrapper = std::mem::size_of::<Reader<'_>>()
        - std::mem::size_of::<Lucene99KnnVectorsReader<'_>>()
        + 2 * std::mem::size_of::<std::sync::atomic::AtomicI32>()
        + 2 * std::mem::size_of::<usize>();
    println!("#memory\tasserting_knn_vectors_reader_wrapper\t{wrapper}");
    println!(
        "#memory\tasserting_knn_vectors_accept_wrapper\t{}",
        std::mem::size_of::<AssertingAcceptDocs<'_>>()
    );
    for name in ["values", "read", "search", "merge_lifecycle"] {
        println!(
            "#check\tasserting_knn_vectors_{name}\t{}\t16",
            batch(name, &fields, files, &reader)
        );
        run(name, &fields, files, &reader, warmup);
        let (count, elapsed) = run(name, &fields, files, &reader, measure);
        println!(
            "asserting_knn_vectors_{name}\t{:.3}\t{count}",
            elapsed.as_nanos() as f64 / count as f64
        );
    }
    reader.close().unwrap();
}
