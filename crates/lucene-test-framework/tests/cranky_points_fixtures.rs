use lucene_codecs::field_infos::{FieldInfo, FieldInfos};
use lucene_codecs::points::Relation;
use lucene_store::ByteBuffersDirectory;
use lucene_test_framework::codecs::cranky_points::*;
use lucene_util::java_random::JavaRandom;
use std::cell::RefCell;
use std::rc::Rc;

const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const META: &[u8] = include_bytes!("../../../fixtures/testframework/data/cranky_points.kdm");
const INDEX: &[u8] = include_bytes!("../../../fixtures/testframework/data/cranky_points.kdi");
const DATA: &[u8] = include_bytes!("../../../fixtures/testframework/data/cranky_points.kdd");
fn field(name: &str, number: i32, dims: i32) -> FieldInfo {
    let mut field = FieldInfo::new(name, number);
    field.point_dimension_count = dims;
    field.point_index_dimension_count = dims;
    field.point_num_bytes = 2;
    field
}
fn fields() -> FieldInfos {
    FieldInfos::new(vec![field("one", 0, 1), field("two", 1, 2)]).unwrap()
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
fn write_state(directory: &dyn lucene_store::Directory) -> WriteState<'_> {
    WriteState {
        directory,
        segment: "_cp",
        suffix: "s",
        id: &ID,
        max_doc: 10,
        max_points_in_leaf_node: 4,
    }
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
#[derive(Default)]
struct State {
    events: Vec<String>,
    fail: bool,
    visitor_fail: bool,
    expected: usize,
}
type Shared = Rc<RefCell<State>>;
fn event(state: &Shared, name: impl Into<String>) -> Result<()> {
    let mut state = state.borrow_mut();
    state.events.push(name.into());
    if state.fail {
        Err(Error::Message("delegate failure".into()))
    } else {
        Ok(())
    }
}
struct RecordingTree {
    state: Shared,
    node: i32,
    root: i32,
}
impl PointTree for RecordingTree {
    type CloneTree = Self;
    fn clone_tree(&self) -> Self {
        self.state.borrow_mut().events.push("clone".into());
        Self {
            state: self.state.clone(),
            node: self.node,
            root: self.node,
        }
    }
    fn move_to_child(&mut self) -> Result<bool> {
        event(&self.state, "child")?;
        if self.node != 1 {
            return Ok(false);
        }
        self.node = 2;
        Ok(true)
    }
    fn move_to_sibling(&mut self) -> Result<bool> {
        event(&self.state, "sibling")?;
        if self.node == self.root || self.node != 2 {
            return Ok(false);
        }
        self.node = 3;
        Ok(true)
    }
    fn move_to_parent(&mut self) -> Result<bool> {
        event(&self.state, "parent")?;
        if self.node == self.root {
            return Ok(false);
        }
        self.node = 1;
        Ok(true)
    }
    fn min_packed_value(&self) -> &[u8] {
        self.state.borrow_mut().events.push("treeMin".into());
        &[1, 2]
    }
    fn max_packed_value(&self) -> &[u8] {
        self.state.borrow_mut().events.push("treeMax".into());
        &[3, 4]
    }
    fn size(&self) -> i64 {
        self.state.borrow_mut().events.push("treeSize".into());
        if self.node == 1 {
            3
        } else {
            1
        }
    }
    fn visit_doc_ids<V: IntersectVisitor>(&mut self, visitor: &mut V) -> Result<()> {
        event(&self.state, "ids")?;
        visitor.grow(1);
        visitor.visit(self.node)
    }
    fn visit_doc_values<V: IntersectVisitor>(&mut self, visitor: &mut V) -> Result<()> {
        event(&self.state, "values")?;
        visitor.grow(1);
        visitor.visit_value(self.node, &[1, 2])
    }
}
struct RecordingValues(Shared);
impl PointValues for RecordingValues {
    type Tree<'a> = RecordingTree;
    fn point_tree(&self) -> Result<Self::Tree<'_>> {
        event(&self.0, "tree")?;
        Ok(RecordingTree {
            state: self.0.clone(),
            node: 1,
            root: 1,
        })
    }
    fn min_packed_value(&self) -> Result<&[u8]> {
        event(&self.0, "min")?;
        Ok(&[1, 2])
    }
    fn max_packed_value(&self) -> Result<&[u8]> {
        event(&self.0, "max")?;
        Ok(&[3, 4])
    }
    fn num_dimensions(&self) -> Result<i32> {
        event(&self.0, "dims")?;
        Ok(1)
    }
    fn num_index_dimensions(&self) -> Result<i32> {
        event(&self.0, "indexDims")?;
        Ok(1)
    }
    fn bytes_per_dimension(&self) -> Result<i32> {
        event(&self.0, "bytes")?;
        Ok(2)
    }
    fn size(&self) -> i64 {
        self.0.borrow_mut().events.push("size".into());
        3
    }
    fn doc_count(&self) -> i32 {
        self.0.borrow_mut().events.push("docCount".into());
        2
    }
}
struct RecordingReader(Shared);
impl PointsReader for RecordingReader {
    type Values<'a> = RecordingValues;
    fn values(&self, field: &str) -> Result<Option<Self::Values<'_>>> {
        event(&self.0, format!("get:{field}"))?;
        Ok((field != "missing").then(|| RecordingValues(self.0.clone())))
    }
    fn check_integrity(&self) -> Result<()> {
        event(&self.0, "integrity")
    }
    fn close(&mut self) -> Result<()> {
        event(&self.0, "readerClose")
    }
}
struct RecordingWriter(Shared);
impl PointsWriter for RecordingWriter {
    fn write_field<R: PointsReader>(&mut self, _: &FieldInfo, reader: &R) -> Result<()> {
        let same = reader as *const R as usize == self.0.borrow().expected;
        event(&self.0, format!("write:{same}"))
    }
    fn finish(&mut self) -> Result<()> {
        event(&self.0, "finish")
    }
    fn merge<R: PointsReader>(&mut self, _: &MergeState<'_, R>) -> Result<()> {
        event(&self.0, "merge")
    }
    fn close(&mut self) -> Result<()> {
        event(&self.0, "writerClose")
    }
}
struct RecordingFormat(Shared);
impl PointsFormat for RecordingFormat {
    type Writer<'a> = RecordingWriter;
    type Reader<'a> = RecordingReader;
    fn fields_writer<'a>(&mut self, _: WriteState<'a>) -> Result<Self::Writer<'a>> {
        event(&self.0, "writer")?;
        Ok(RecordingWriter(self.0.clone()))
    }
    fn fields_reader<'a>(&mut self, _: ReadState<'a>) -> Result<Self::Reader<'a>> {
        event(&self.0, "reader")?;
        Ok(RecordingReader(self.0.clone()))
    }
}
struct RecordingVisitor(Shared);
impl RecordingVisitor {
    fn callback(&self, name: String) -> Result<()> {
        let mut state = self.0.borrow_mut();
        state.events.push(name);
        if state.visitor_fail {
            Err(Error::Message("visitor failure".into()))
        } else {
            Ok(())
        }
    }
}
impl IntersectVisitor for RecordingVisitor {
    fn visit(&mut self, doc: i32) -> Result<()> {
        self.callback(format!("doc:{doc}"))
    }
    fn visit_value(&mut self, doc: i32, value: &[u8]) -> Result<()> {
        self.callback(format!("value:{doc}:{}", hex(value)))
    }
    fn grow(&mut self, count: usize) {
        self.0.borrow_mut().events.push(format!("grow:{count}"));
    }
    fn compare(&mut self, _: &[u8], _: &[u8]) -> Relation {
        Relation::CellCrossesQuery
    }
}
#[test]
fn actual_java_guard_order_all_callbacks_errors_clone_escape_and_rng_continuation() {
    let mut expected =
        include_str!("../../../fixtures/testframework/data/cranky_points.tsv").lines();
    let fields = fields();
    let directory = ByteBuffersDirectory::new();
    for seed in [0, 1, 42, -1, i64::MIN, i64::MAX] {
        let shared = Rc::new(RefCell::new(State::default()));
        let random = Rc::new(RefCell::new(JavaRandom::new(seed)));
        let source = Box::new(RecordingReader(shared.clone()));
        shared.borrow_mut().expected = &*source as *const RecordingReader as usize;
        let mut format = CrankyPointsFormat::new(RecordingFormat(shared.clone()), random.clone());
        let mut writer = format.fields_writer(write_state(&directory)).unwrap();
        let mut reader = format.fields_reader(read_state(&fields)).unwrap();
        let values = reader.values("f").unwrap().unwrap();
        let mut tree = values.point_tree().unwrap();
        let mut visitor = RecordingVisitor(shared.clone());
        let merge = MergeState::<RecordingReader> {
            sources: &[],
            fields: &fields,
            check_aborted: &|| Ok(()),
        };
        for step in 0..2800 {
            let op = step % 28;
            {
                let mut state = shared.borrow_mut();
                state.events.clear();
                state.fail = step % 17 == 0;
                state.visitor_fail = step % 19 == 0;
            }
            let result: Result<String> = match op {
                0 => format
                    .fields_writer(write_state(&directory))
                    .map(|_| "ok".into()),
                1 => format
                    .fields_reader(read_state(&fields))
                    .map(|_| "ok".into()),
                2 => writer
                    .write_field(&field("f", 0, 1), &*source)
                    .map(|_| "ok".into()),
                3 => writer.finish().map(|_| "ok".into()),
                4 => writer.merge(&merge).map(|_| "ok".into()),
                5 => writer.close().map(|_| "ok".into()),
                6 => reader.check_integrity().map(|_| "ok".into()),
                7 => reader.close().map(|_| "ok".into()),
                8 => reader
                    .values("missing")
                    .map(|values| values.is_none().to_string()),
                9 => reader
                    .values("f")
                    .map(|values| values.is_some().to_string()),
                10 => values.min_packed_value().map(hex),
                11 => values.max_packed_value().map(hex),
                12 => values.num_dimensions().map(|v| v.to_string()),
                13 => values.num_index_dimensions().map(|v| v.to_string()),
                14 => values.bytes_per_dimension().map(|v| v.to_string()),
                15 => Ok(values.size().to_string()),
                16 => Ok(values.doc_count().to_string()),
                17 => values.point_tree().map(|_| "ok".into()),
                18 => tree.move_to_child().map(|v| v.to_string()),
                19 => tree.move_to_sibling().map(|v| v.to_string()),
                20 => tree.move_to_parent().map(|v| v.to_string()),
                21 => Ok(hex(tree.min_packed_value())),
                22 => Ok(hex(tree.max_packed_value())),
                23 => Ok(tree.size().to_string()),
                24 => tree.visit_doc_ids(&mut visitor).map(|_| "ok".into()),
                25 => tree.visit_doc_values(&mut visitor).map(|_| "ok".into()),
                26 => tree
                    .clone_tree()
                    .visit_doc_ids(&mut visitor)
                    .map(|_| "ok".into()),
                27 => tree
                    .clone_tree()
                    .visit_doc_values(&mut visitor)
                    .map(|_| "ok".into()),
                _ => unreachable!(),
            };
            let status = result.unwrap_or_else(|error| error.to_string());
            let events = shared.borrow().events.join(",");
            let actual = format!(
                "{seed}\t{step}\t{op}\t{status}\t{events}\t{}",
                random.borrow_mut().next_int()
            );
            assert_eq!(
                Some(actual.as_str()),
                expected.next(),
                "seed={seed} step={step}"
            );
        }
    }
    assert!(expected.next().is_none());
}

#[derive(Default)]
struct Collected {
    ids: Vec<i32>,
    values: Vec<(i32, Vec<u8>)>,
    grows: Vec<usize>,
    fail: bool,
}
impl IntersectVisitor for Collected {
    fn visit(&mut self, doc: i32) -> Result<()> {
        self.ids.push(doc);
        if self.fail {
            Err(Error::Message("visitor failure".into()))
        } else {
            Ok(())
        }
    }
    fn visit_value(&mut self, doc: i32, value: &[u8]) -> Result<()> {
        self.values.push((doc, value.to_vec()));
        if self.fail {
            Err(Error::Message("visitor failure".into()))
        } else {
            Ok(())
        }
    }
    fn grow(&mut self, count: usize) {
        self.grows.push(count);
    }
    fn compare(&mut self, _: &[u8], _: &[u8]) -> Relation {
        Relation::CellCrossesQuery
    }
}
#[test]
fn real_java_native_reads_navigation_clones_and_ordinary_writer_bytes() {
    use lucene_store::Directory;
    let fields = fields();
    let mut format = Lucene90PointsFormat;
    let reader = format.fields_reader(read_state(&fields)).unwrap();
    reader.check_integrity().unwrap();
    let mut manifest = Vec::new();
    for field in &fields.fields {
        let values = reader.values(&field.name).unwrap().unwrap();
        let mut tree = values.point_tree().unwrap();
        assert_eq!(
            (
                values.num_dimensions().unwrap(),
                values.num_index_dimensions().unwrap(),
                values.bytes_per_dimension().unwrap()
            ),
            (
                field.point_dimension_count,
                field.point_index_dimension_count,
                2
            )
        );
        manifest.push(format!(
            "field\t{}\t{}\t{}\t{}\t{}",
            field.number,
            values.size(),
            values.doc_count(),
            hex(values.min_packed_value().unwrap()),
            hex(values.max_packed_value().unwrap())
        ));
        assert_eq!(tree.size(), 13);
        assert_eq!(tree.min_packed_value(), values.min_packed_value().unwrap());
        assert_eq!(tree.max_packed_value(), values.max_packed_value().unwrap());
        assert!(!tree.move_to_parent().unwrap());
        assert!(!tree.move_to_sibling().unwrap());
        assert!(tree.move_to_child().unwrap());
        let left_size = tree.size();
        let mut clone = tree.clone_tree();
        assert!(!clone.move_to_parent().unwrap());
        assert!(!clone.move_to_sibling().unwrap());
        assert!(tree.move_to_sibling().unwrap());
        assert_eq!(left_size + tree.size(), 13);
        assert!(!tree.move_to_sibling().unwrap());
        assert!(tree.move_to_parent().unwrap());
        let mut visits = Collected::default();
        tree.visit_doc_values(&mut visits).unwrap();
        assert_eq!(visits.values.len(), 13);
        let mut points: Vec<_> = visits
            .values
            .iter()
            .map(|(doc, value)| format!("{doc}:{}", hex(value)))
            .collect();
        points.sort();
        manifest.extend(
            points
                .into_iter()
                .map(|point| format!("point\t{}\t{point}", field.number)),
        );
        tree.visit_doc_ids(&mut visits).unwrap();
        assert_eq!(visits.ids.len(), 13);
        assert_eq!(visits.grows.last(), Some(&13));
        let mut failed = Collected {
            fail: true,
            ..Collected::default()
        };
        assert_eq!(
            tree.visit_doc_values(&mut failed).unwrap_err().to_string(),
            "visitor failure"
        );
        assert_eq!(failed.values.len(), 1);
        let mut failed = Collected {
            fail: true,
            ..Collected::default()
        };
        assert!(tree.visit_doc_ids(&mut failed).is_err());
        assert_eq!(failed.ids.len(), 1);
        while tree.move_to_child().unwrap() {}
        assert!(!tree.move_to_child().unwrap());
    }
    let expected = include_str!("../../../fixtures/testframework/data/cranky_points_real.tsv");
    assert_eq!(
        manifest.join("\n"),
        expected
            .lines()
            .filter(|line| !line.starts_with("rng\t"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    // Reindex the real tree through the ordinary non-MutablePointTree path.
    let directory = ByteBuffersDirectory::new();
    let random = Rc::new(RefCell::new(JavaRandom::new(42)));
    let mut cranky = CrankyPointsFormat::new(Lucene90PointsFormat, random.clone());
    let mut writer = cranky.fields_writer(write_state(&directory)).unwrap();
    for field in &fields.fields {
        writer.write_field(field, &reader).unwrap();
    }
    writer.finish().unwrap();
    writer.close().unwrap();
    for (extension, bytes) in [("kdm", META), ("kdi", INDEX), ("kdd", DATA)] {
        assert_eq!(
            directory
                .open(&format!("_cp_s.{extension}"))
                .unwrap()
                .as_ref(),
            bytes
        );
    }
    assert_eq!(
        format!("rng\t{}", random.borrow_mut().next_int()),
        expected.lines().last().unwrap()
    );
}
