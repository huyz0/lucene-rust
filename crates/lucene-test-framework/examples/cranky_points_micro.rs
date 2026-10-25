//! Actual CrankyPoints wrapper and native Lucene90 read/write/merge benchmark.
use lucene_codecs::{
    field_infos::{FieldInfo, FieldInfos},
    points::Relation,
};
use lucene_store::{ByteBuffersDirectory, Directory};
use lucene_test_framework::codecs::cranky_points::*;
use lucene_util::java_random::JavaRandom;
use std::cell::RefCell;
use std::hint::black_box;
use std::rc::Rc;
use std::time::{Duration, Instant};

const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const META: &[u8] = include_bytes!("../../../fixtures/testframework/data/cranky_points.kdm");
const INDEX: &[u8] = include_bytes!("../../../fixtures/testframework/data/cranky_points.kdi");
const DATA: &[u8] = include_bytes!("../../../fixtures/testframework/data/cranky_points.kdd");
const MAPS: [[i32; 10]; 2] = [
    [0, -1, 1, 2, 3, 4, 5, 6, 7, 8],
    [9, -1, 10, 11, 12, 13, 14, 15, 16, 17],
];
fn fields() -> FieldInfos {
    FieldInfos::new(
        (0..2)
            .map(|number| {
                let mut field = FieldInfo::new(if number == 0 { "one" } else { "two" }, number);
                field.point_dimension_count = number + 1;
                field.point_index_dimension_count = number + 1;
                field.point_num_bytes = 2;
                field
            })
            .collect(),
    )
    .unwrap()
}
fn read_state(fields: &FieldInfos) -> ReadState<'_> {
    ReadState {
        meta: META,
        index: INDEX,
        data: DATA,
        fields,
        suffix: "s",
        id: &ID,
    }
}
fn write_state(directory: &dyn Directory, max_doc: i32) -> WriteState<'_> {
    WriteState {
        directory,
        segment: "_cp",
        suffix: "s",
        id: &ID,
        max_doc,
        max_points_in_leaf_node: 4,
    }
}
fn step(digest: u64, value: u64) -> u64 {
    digest.wrapping_mul(31).wrapping_add(value)
}
struct Visitor(u64);
impl IntersectVisitor for Visitor {
    fn visit(&mut self, doc: i32) -> Result<()> {
        self.0 = step(self.0, doc as u64);
        Ok(())
    }
    fn visit_value(&mut self, doc: i32, value: &[u8]) -> Result<()> {
        self.visit(doc)?;
        for &byte in value {
            self.0 = step(self.0, u64::from(byte));
        }
        Ok(())
    }
    fn compare(&mut self, _: &[u8], _: &[u8]) -> Relation {
        Relation::CellCrossesQuery
    }
}
fn read(reader: &impl PointsReader, digest: u64) -> u64 {
    reader.check_integrity().unwrap();
    let mut visitor = Visitor(digest);
    for name in ["one", "two"] {
        let values = reader.values(name).unwrap().unwrap();
        for &byte in values
            .min_packed_value()
            .unwrap()
            .iter()
            .chain(values.max_packed_value().unwrap())
        {
            visitor.0 = step(visitor.0, u64::from(byte));
        }
        for value in [
            i64::from(values.num_dimensions().unwrap()),
            i64::from(values.num_index_dimensions().unwrap()),
            i64::from(values.bytes_per_dimension().unwrap()),
            values.size(),
            i64::from(values.doc_count()),
        ] {
            visitor.0 = step(visitor.0, value as u64);
        }
        let mut tree = values.point_tree().unwrap();
        for &byte in tree
            .min_packed_value()
            .iter()
            .chain(tree.max_packed_value())
        {
            visitor.0 = step(visitor.0, u64::from(byte));
        }
        visitor.0 = step(visitor.0, tree.size() as u64);
        tree.visit_doc_ids(&mut visitor).unwrap();
        tree.visit_doc_values(&mut visitor).unwrap();
        tree.clone_tree().visit_doc_values(&mut visitor).unwrap();
    }
    visitor.0
}
struct Setup<'a> {
    fields: &'a FieldInfos,
    source: Lucene90PointsReader<'a>,
    wrapped: CrankyPointsReader<Lucene90PointsReader<'a>>,
    merge_random: Rc<RefCell<JavaRandom>>,
}
fn batch(setup: &Setup<'_>, case: usize, seed: i64) -> u64 {
    let fields = setup.fields;
    let mut digest = 0;
    let mut native = Lucene90PointsFormat;
    for _ in 0..16 {
        let random = if case == 3 {
            let random = setup.merge_random.clone();
            *random.borrow_mut() = JavaRandom::new(42);
            random
        } else {
            Rc::new(RefCell::new(JavaRandom::new(if case == 0 {
                seed
            } else {
                42
            })))
        };
        let mut format = CrankyPointsFormat::new(Lucene90PointsFormat, random.clone());
        if case == 0 {
            let mut reader = format.fields_reader(read_state(fields)).unwrap();
            digest = read(&reader, digest);
            reader.close().unwrap();
        } else {
            let directory = ByteBuffersDirectory::new();
            if case == 1 {
                let mut writer = format.fields_writer(write_state(&directory, 10)).unwrap();
                for field in &fields.fields {
                    writer.write_field(field, &setup.source).unwrap();
                }
                writer.finish().unwrap();
                writer.close().unwrap();
            } else if case == 2 {
                let sources = [
                    MergeSource {
                        reader: Some(&setup.source),
                        fields,
                        doc_map: &MAPS[0],
                    },
                    MergeSource {
                        reader: Some(&setup.source),
                        fields,
                        doc_map: &MAPS[1],
                    },
                ];
                let mut writer = format.fields_writer(write_state(&directory, 18)).unwrap();
                writer
                    .merge(&MergeState {
                        sources: &sources,
                        fields,
                        check_aborted: &|| Ok(()),
                    })
                    .unwrap();
                writer.close().unwrap();
            } else {
                let sources = [
                    MergeSource {
                        reader: Some(&setup.wrapped),
                        fields,
                        doc_map: &MAPS[0],
                    },
                    MergeSource {
                        reader: Some(&setup.wrapped),
                        fields,
                        doc_map: &MAPS[1],
                    },
                ];
                let mut writer = native.fields_writer(write_state(&directory, 18)).unwrap();
                writer
                    .merge(&MergeState {
                        sources: &sources,
                        fields,
                        check_aborted: &|| Ok(()),
                    })
                    .unwrap();
                writer.close().unwrap();
            }
            for extension in ["kdm", "kdi", "kdd"] {
                for &byte in directory
                    .open(&format!("_cp_s.{extension}"))
                    .unwrap()
                    .iter()
                {
                    digest = step(digest, u64::from(byte));
                }
            }
        }
        digest = step(digest, u64::from(random.borrow_mut().next_int() as u32));
    }
    black_box(digest)
}
fn run(setup: &Setup<'_>, case: usize, seed: i64, millis: u64, expected: u64) -> (Duration, u64) {
    let start = Instant::now();
    let mut count = 0;
    while start.elapsed() < Duration::from_millis(millis) {
        assert_eq!(black_box(batch(setup, case, seed)), expected);
        count += 16;
    }
    (start.elapsed(), count)
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(500);
    let measure = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let fields = fields();
    let merge_random = Rc::new(RefCell::new(JavaRandom::new(42)));
    let mut native = Lucene90PointsFormat;
    let setup = Setup {
        fields: &fields,
        source: native.fields_reader(read_state(&fields)).unwrap(),
        wrapped: CrankyPointsReader::new(
            native.fields_reader(read_state(&fields)).unwrap(),
            merge_random.clone(),
        ),
        merge_random,
    };
    let seed = (0..)
        .find(|seed| {
            let mut random = JavaRandom::new(*seed);
            (0..21).all(|_| random.next_int_bounded(100) != 0)
        })
        .unwrap();
    let owned = std::mem::size_of::<CrankyPointsFormat<Lucene90PointsFormat>>()
        + 2 * std::mem::size_of::<usize>()
        + std::mem::size_of::<RefCell<JavaRandom>>();
    println!("cranky_points_format_owned_bytes\t{owned}");
    for (case, name) in [
        "cranky_points_read",
        "cranky_points_write",
        "cranky_points_merge_native",
        "cranky_points_merge_wrapped",
    ]
    .iter()
    .enumerate()
    {
        let expected = batch(&setup, case, seed);
        println!("#check\t{name}\t{expected}\t16");
        run(&setup, case, seed, warmup, expected);
        let (elapsed, count) = run(&setup, case, seed, measure, expected);
        println!(
            "{name}\t{:.3}\t{count}",
            elapsed.as_nanos() as f64 / count as f64
        );
    }
}
