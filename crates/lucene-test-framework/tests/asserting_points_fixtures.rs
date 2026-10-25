use lucene_codecs::{
    field_infos::{FieldInfo, FieldInfos},
    points::Relation,
};
use lucene_store::{ByteBuffersDirectory, Directory};
use lucene_test_framework::codecs::asserting_points::*;
use lucene_test_framework::codecs::cranky_points::{self as byte, PointsReader as ByteReader};
use std::fmt;
use std::sync::{Arc, Mutex};
type Shared = Arc<Mutex<State>>;
#[derive(Default)]
struct State {
    events: Vec<String>,
    fail: String,
    size: i64,
    docs: i32,
    tree_size: i64,
    missing: bool,
}
fn state() -> Shared {
    Arc::new(Mutex::new(State {
        size: 3,
        docs: 2,
        tree_size: 3,
        ..State::default()
    }))
}
fn event(s: &Shared, name: impl Into<String>) -> Result<()> {
    let name = name.into();
    let mut s = s.lock().unwrap();
    s.events.push(name.clone());
    if s.fail == name {
        Err(Error::Message("delegate failure".into()))
    } else {
        Ok(())
    }
}
fn log(s: &Shared, name: impl Into<String>) {
    s.lock().unwrap().events.push(name.into());
}
fn outcome(f: impl FnOnce() -> Result<()>) -> String {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(())) => "ok".into(),
        Ok(Err(e)) => format!("io:{e}"),
        Err(_) => "assert".into(),
    }
}
fn hex(v: &[u8]) -> String {
    v.iter().map(|b| format!("{b:02x}")).collect()
}
fn unhex(v: &str) -> Vec<u8> {
    v.as_bytes()
        .chunks_exact(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect()
}
struct Visitor {
    events: Shared,
    relation: Relation,
    fail: bool,
}
impl Visitor {
    fn new(events: Shared) -> Self {
        Self {
            events,
            relation: Relation::CellCrossesQuery,
            fail: false,
        }
    }
}
impl IntersectVisitor for Visitor {
    fn visit(&mut self, doc: i32) -> Result<()> {
        log(&self.events, format!("doc:{doc}"));
        if self.fail {
            Err(Error::Message("visitor failure".into()))
        } else {
            Ok(())
        }
    }
    fn visit_value(&mut self, doc: i32, v: &[u8]) -> Result<()> {
        log(&self.events, format!("value:{doc}:{}", hex(v)));
        if self.fail {
            Err(Error::Message("visitor failure".into()))
        } else {
            Ok(())
        }
    }
    fn grow(&mut self, n: usize) {
        log(&self.events, format!("grow:{n}"));
    }
    fn compare(&mut self, min: &[u8], max: &[u8]) -> Relation {
        log(&self.events, format!("compare:{}:{}", hex(min), hex(max)));
        self.relation
    }
}
// Tokens that change the delegate are interpreted before borrowing the checked
// visitor for a call, preserving the same persistent validator between tokens.
struct ConfigVisitor {
    base: Visitor,
    config: Arc<Mutex<(Relation, bool)>>,
}
impl IntersectVisitor for ConfigVisitor {
    fn visit(&mut self, d: i32) -> Result<()> {
        self.base.fail = self.config.lock().unwrap().1;
        self.base.visit(d)
    }
    fn visit_value(&mut self, d: i32, v: &[u8]) -> Result<()> {
        self.base.fail = self.config.lock().unwrap().1;
        self.base.visit_value(d, v)
    }
    fn grow(&mut self, n: usize) {
        self.base.grow(n)
    }
    fn compare(&mut self, min: &[u8], max: &[u8]) -> Relation {
        self.base.relation = self.config.lock().unwrap().0;
        self.base.compare(min, max)
    }
}
fn visitor_trace(program: &str) -> (String, String) {
    let s = state();
    let config = Arc::new(Mutex::new((Relation::CellCrossesQuery, false)));
    let mut delegate = ConfigVisitor {
        base: Visitor::new(s.clone()),
        config: config.clone(),
    };
    let mut visitor = AssertingIntersectVisitor::new(1, 1, 1, &mut delegate);
    let mut results = Vec::new();
    for token in program.split(',') {
        let a: Vec<_> = token.split(':').collect();
        results.push(outcome(|| {
            match a[0] {
                "g" => visitor.grow(a[1].parse().unwrap()),
                "d" => visitor.visit(a[1].parse().unwrap())?,
                "v" => visitor.visit_value(a[1].parse().unwrap(), &unhex(a[2]))?,
                "c" => {
                    visitor.compare(&unhex(a[1]), &unhex(a[2]));
                }
                "in" => config.lock().unwrap().0 = Relation::CellInsideQuery,
                "out" => config.lock().unwrap().0 = Relation::CellOutsideQuery,
                "cross" => config.lock().unwrap().0 = Relation::CellCrossesQuery,
                "fail" => config.lock().unwrap().1 = true,
                "ok" => config.lock().unwrap().1 = false,
                _ => panic!("{token}"),
            };
            Ok(())
        }));
    }
    let events = s.lock().unwrap().events.join(",");
    (results.join(";"), events)
}
struct Values(Shared);
impl PointValues for Values {
    type Tree<'a> = Tree;
    fn point_tree(&self) -> Result<Tree> {
        event(&self.0, "tree")?;
        Ok(Tree(self.0.clone()))
    }
    fn min_packed_value(&self) -> Result<&[u8]> {
        event(&self.0, "min")?;
        Ok(&[0])
    }
    fn max_packed_value(&self) -> Result<&[u8]> {
        event(&self.0, "max")?;
        Ok(&[255])
    }
    fn num_dimensions(&self) -> Result<i32> {
        event(&self.0, "dims")?;
        Ok(1)
    }
    fn num_index_dimensions(&self) -> Result<i32> {
        event(&self.0, "index")?;
        Ok(1)
    }
    fn bytes_per_dimension(&self) -> Result<i32> {
        event(&self.0, "bytes")?;
        Ok(1)
    }
    fn size(&self) -> i64 {
        log(&self.0, "size");
        self.0.lock().unwrap().size
    }
    fn doc_count(&self) -> i32 {
        log(&self.0, "docs");
        self.0.lock().unwrap().docs
    }
}
struct Tree(Shared);
impl PointTree for Tree {
    type CloneTree = Self;
    fn clone_tree(&self) -> Self {
        log(&self.0, "clone");
        Self(self.0.clone())
    }
    fn move_to_child(&mut self) -> Result<bool> {
        event(&self.0, "child")?;
        Ok(false)
    }
    fn move_to_sibling(&mut self) -> Result<bool> {
        event(&self.0, "sibling")?;
        Ok(false)
    }
    fn move_to_parent(&mut self) -> Result<bool> {
        event(&self.0, "parent")?;
        Ok(false)
    }
    fn min_packed_value(&self) -> &[u8] {
        log(&self.0, "treeMin");
        &[0]
    }
    fn max_packed_value(&self) -> &[u8] {
        log(&self.0, "treeMax");
        &[255]
    }
    fn size(&self) -> i64 {
        log(&self.0, "treeSize");
        self.0.lock().unwrap().tree_size
    }
    fn visit_doc_ids<I: IntersectVisitor>(&mut self, v: &mut I) -> Result<()> {
        event(&self.0, "ids")?;
        v.grow(1);
        v.visit(1)
    }
    fn visit_doc_values<I: IntersectVisitor>(&mut self, v: &mut I) -> Result<()> {
        event(&self.0, "values")?;
        v.grow(2);
        v.visit_value(0, &[0])?;
        v.visit_value(1, &[255])
    }
}
struct Reader(Shared);
impl ByteReader for Reader {
    type Values<'a> = Values;
    fn values(&self, name: &str) -> Result<Option<Values>> {
        event(&self.0, format!("get:{name}"))?;
        Ok((!self.0.lock().unwrap().missing).then(|| Values(self.0.clone())))
    }
    fn check_integrity(&self) -> Result<()> {
        event(&self.0, "integrity")
    }
    fn close(&mut self) -> Result<()> {
        event(&self.0, "close")
    }
}
impl PointsReader for Reader {
    fn get_merge_instance(&self) -> Self {
        log(&self.0, "merge");
        Self(self.0.clone())
    }
}
impl fmt::Display for Reader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ProbeReader")
    }
}
fn fields() -> FieldInfos {
    FieldInfos::new(vec![FieldInfo::new("f", 0).with_points(1, 1, 2)]).unwrap()
}
fn reader_trace(name: &str) -> (String, String) {
    let s = state();
    let fis = fields();
    let mut r = AssertingPointsReader::new(Reader(s.clone()), 2, &fis, false);
    assert!(Arc::ptr_eq(&r.delegate().0, &s));
    assert!(Arc::ptr_eq(&r.delegate_mut().0, &s));
    let result = outcome(|| {
        match name {
            "missing-field" => {
                r.values("missing")?;
            }
            "null" => {
                s.lock().unwrap().missing = true;
                assert!(r.values("f")?.is_none());
            }
            "size0" | "docs0" | "docs-size" | "docs-max" => {
                match name {
                    "size0" => s.lock().unwrap().size = 0,
                    "docs0" => s.lock().unwrap().docs = 0,
                    "docs-size" => s.lock().unwrap().docs = 4,
                    _ => s.lock().unwrap().docs = 3,
                };
                r.values("f")?;
            }
            "all" => {
                let v = r.values("f")?.unwrap();
                v.min_packed_value()?;
                v.max_packed_value()?;
                v.num_dimensions()?;
                v.num_index_dimensions()?;
                v.bytes_per_dimension()?;
                v.size();
                v.doc_count();
                let mut t = v.point_tree()?;
                t.move_to_child()?;
                t.move_to_sibling()?;
                t.move_to_parent()?;
                t.min_packed_value();
                t.max_packed_value();
                t.size();
                let mut vis = Visitor::new(s.clone());
                t.visit_doc_ids(&mut vis)?;
                t.visit_doc_values(&mut vis)?;
                t.clone_tree().visit_doc_values(&mut vis)?;
                r.check_integrity()?;
                r.close()?;
            }
            "tree0" => {
                s.lock().unwrap().tree_size = 0;
                r.values("f")?.unwrap().point_tree()?.size();
            }
            "tree-error" => {
                s.lock().unwrap().fail = "tree".into();
                r.values("f")?.unwrap().point_tree()?;
            }
            "dims-error" => {
                s.lock().unwrap().fail = "dims".into();
                r.values("f")?
                    .unwrap()
                    .point_tree()?
                    .visit_doc_ids(&mut Visitor::new(s.clone()))?;
            }
            "close-error" => {
                s.lock().unwrap().fail = "close".into();
                r.close()?;
            }
            "integrity-error" => {
                s.lock().unwrap().fail = "integrity".into();
                r.check_integrity()?;
            }
            "merge" => {
                r.get_merge_instance().values("f")?;
                r.close()?;
            }
            "thread" | "merge-thread" | "values-thread" => {
                let values = if name == "values-thread" {
                    r.values("f")?
                } else {
                    None
                };
                let target = if name == "merge-thread" {
                    r.get_merge_instance()
                } else {
                    r
                };
                let result = std::thread::scope(|scope| {
                    scope
                        .spawn(move || {
                            outcome(|| {
                                if let Some(values) = values {
                                    values.size();
                                } else {
                                    target.values("f")?;
                                }
                                Ok(())
                            })
                        })
                        .join()
                        .unwrap()
                });
                if result == "assert" {
                    panic!("wrong-thread assertion");
                }
                assert_eq!(result, "ok");
            }
            "display" => assert_eq!(r.to_string(), "AssertingPointsReader(ProbeReader)"),
            _ => panic!("{name}"),
        };
        Ok(())
    });
    let events = s.lock().unwrap().events.join(",");
    (result, events)
}
struct Writer(Shared);
impl PointsWriter for Writer {
    fn write_field<R: ByteReader>(&mut self, _: &FieldInfo, _: &R) -> Result<()> {
        event(&self.0, "write")
    }
    fn merge<R: ByteReader>(&mut self, _: &MergeState<'_, R>) -> Result<()> {
        event(&self.0, "merge")
    }
    fn finish(&mut self) -> Result<()> {
        event(&self.0, "finish")
    }
    fn close(&mut self) -> Result<()> {
        event(&self.0, "close")
    }
}
fn writer_trace(name: &str) -> (String, String) {
    let s = state();
    let mut w = AssertingPointsWriter::new(Writer(s.clone()));
    assert!(Arc::ptr_eq(&w.delegate().0, &s));
    assert!(Arc::ptr_eq(&w.delegate_mut().0, &s));
    let f = fields();
    let r = Reader(s.clone());
    let result = outcome(|| {
        match name {
            "all" => {
                w.write_field(&f.fields[0], &r)?;
                w.merge::<Reader>(&MergeState {
                    sources: &[],
                    fields: &f,
                    check_aborted: &|| Ok(()),
                })?;
                w.finish()?;
                w.close()?;
            }
            "zero" => {
                let err = w.write_field(&FieldInfo::new("f", 0), &r).unwrap_err();
                return Err(Error::Message(format!("illegal:{err}")));
            }
            "write-error" => {
                s.lock().unwrap().fail = "write".into();
                w.write_field(&f.fields[0], &r)?;
            }
            "close-error" => {
                s.lock().unwrap().fail = "close".into();
                w.close()?;
            }
            _ => panic!("{name}"),
        };
        Ok(())
    });
    let result = result
        .strip_prefix("io:illegal:")
        .map_or(result.clone(), |s| format!("illegal:{s}"));
    let events = s.lock().unwrap().events.join(",");
    (result, events)
}
fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/testframework/data")
            .join(name),
    )
    .unwrap()
}
#[test]
fn java_assertion_and_callback_order_traces() {
    let data = String::from_utf8(fixture("asserting_points.tsv")).unwrap();
    let mut counts = [0; 3];
    for line in data.lines() {
        let c: Vec<_> = line.split('\t').collect();
        let (actual, index) = match c[0] {
            "visitor" => (visitor_trace(c[1]), 0),
            "reader" => (reader_trace(c[1]), 1),
            "writer" => (writer_trace(c[1]), 2),
            _ => panic!("{line}"),
        };
        assert_eq!(
            (actual.0.as_str(), actual.1.as_str()),
            (c[2], c[3]),
            "{}",
            c[1]
        );
        counts[index] += 1;
    }
    assert_eq!(counts, [18, 17, 4]);
}
const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
fn real_fields() -> FieldInfos {
    FieldInfos::new(vec![
        FieldInfo::new("one", 0).with_points(1, 1, 2),
        FieldInfo::new("two", 1).with_points(2, 2, 2),
    ])
    .unwrap()
}
fn read_state<'a>(
    meta: &'a [u8],
    index: &'a [u8],
    data: &'a [u8],
    fields: &'a FieldInfos,
    max_doc: i32,
) -> ReadState<'a> {
    ReadState {
        bytes: byte::ReadState {
            meta,
            index,
            data,
            fields,
            suffix: "s",
            id: &ID,
        },
        max_doc,
    }
}
fn write_state(directory: &dyn lucene_store::Directory, max_doc: i32) -> WriteState<'_> {
    WriteState {
        directory,
        segment: "_cp",
        suffix: "s",
        id: &ID,
        max_doc,
        max_points_in_leaf_node: 4,
    }
}
struct Collect(Vec<String>);
impl IntersectVisitor for Collect {
    fn visit(&mut self, d: i32) -> Result<()> {
        self.0.push(format!("doc:{d}"));
        Ok(())
    }
    fn visit_value(&mut self, d: i32, v: &[u8]) -> Result<()> {
        self.0.push(format!("{d}:{}", hex(v)));
        Ok(())
    }
    fn compare(&mut self, _: &[u8], _: &[u8]) -> Relation {
        Relation::CellCrossesQuery
    }
}
#[test]
fn java_native_points_merge_and_independent_lifecycles() {
    let fis = real_fields();
    let meta = fixture("asserting_points.kdm");
    let index = fixture("asserting_points.kdi");
    let data = fixture("asserting_points.kdd");
    let mut format = AssertingPointsFormat::new(Lucene90PointsFormat);
    let mut reader = format
        .fields_reader(read_state(&meta, &index, &data, &fis, 10))
        .unwrap();
    reader.check_integrity().unwrap();
    assert!(reader.native().is_none());
    let address = format.delegate() as *const _;
    assert_eq!(address, format.delegate_mut() as *mut _ as *const _);
    let expected = String::from_utf8(fixture("asserting_points_real.tsv")).unwrap();
    let mut actual = String::new();
    for f in &fis.fields {
        let v = reader.values(&f.name).unwrap().unwrap();
        assert_eq!(v.size(), 13);
        assert_eq!(v.wrapped().size(), 13);
        assert_eq!(v.doc_count(), 10);
        assert_eq!(v.num_dimensions().unwrap(), f.point_dimension_count);
        assert_eq!(
            v.num_index_dimensions().unwrap(),
            f.point_index_dimension_count
        );
        assert_eq!(v.bytes_per_dimension().unwrap(), 2);
        let minimum = if f.number == 0 {
            vec![0, 0]
        } else {
            vec![0, 0, 0, 101]
        };
        let maximum = if f.number == 0 {
            vec![1, 188]
        } else {
            vec![1, 188, 2, 33]
        };
        assert_eq!(v.min_packed_value().unwrap(), minimum);
        assert_eq!(v.max_packed_value().unwrap(), maximum);
        let mut cursor = v.point_tree().unwrap();
        assert_eq!(cursor.min_packed_value(), minimum);
        assert_eq!(cursor.max_packed_value(), maximum);
        assert_eq!(cursor.size(), 13);
        assert!(cursor.move_to_child().unwrap());
        assert!(cursor.size() < 13);
        assert!(cursor.move_to_sibling().unwrap());
        assert!(cursor.move_to_parent().unwrap());
        assert!(!cursor.move_to_parent().unwrap());
        assert_eq!(cursor.clone_tree().size(), 13);
        let mut c = Collect(Vec::new());
        v.point_tree().unwrap().visit_doc_values(&mut c).unwrap();
        c.0.sort();
        for p in c.0 {
            actual.push_str(&format!("{}\t{p}\n", f.number));
        }
    }
    assert_eq!(actual, expected);
    let copied_dir = ByteBuffersDirectory::new();
    let mut copied = format.fields_writer(write_state(&copied_dir, 10)).unwrap();
    for f in &fis.fields {
        copied.write_field(f, &reader).unwrap();
    }
    copied.finish().unwrap();
    copied.close().unwrap();
    for ext in ["kdm", "kdi", "kdd"] {
        assert_eq!(
            copied_dir.open(&format!("_cp_s.{ext}")).unwrap().as_ref(),
            fixture(&format!("asserting_points.{ext}"))
        );
    }
    let maps = [
        [0, -1, 1, 2, 3, 4, 5, 6, 7, 8],
        [9, -1, 10, 11, 12, 13, 14, 15, 16, 17],
    ];
    let sources = [
        MergeSource {
            reader: Some(&reader),
            fields: &fis,
            doc_map: &maps[0],
        },
        MergeSource {
            reader: Some(&reader),
            fields: &fis,
            doc_map: &maps[1],
        },
    ];
    let dir = ByteBuffersDirectory::new();
    let mut w = format.fields_writer(write_state(&dir, 18)).unwrap();
    w.merge(&MergeState {
        sources: &sources,
        fields: &fis,
        check_aborted: &|| Ok(()),
    })
    .unwrap();
    w.close().unwrap();
    for ext in ["kdm", "kdi", "kdd"] {
        assert_eq!(
            dir.open(&format!("_cp_s.{ext}")).unwrap().as_ref(),
            fixture(&format!("asserting_points_merge.{ext}"))
        );
    }
    let mut merged = reader.get_merge_instance();
    reader.close().unwrap();
    assert!(reader.check_integrity().is_err());
    assert!(merged.values("one").unwrap().is_none());
    assert!(merged.check_integrity().is_err());
    let mut another = merged.get_merge_instance();
    merged.close().unwrap();
    assert!(another.values("two").unwrap().is_none());
    another.close().unwrap();
    assert!(another.values("one").unwrap().is_none());
    assert!(another
        .get_merge_instance()
        .values("one")
        .unwrap()
        .is_none());
}
#[test]
fn native_corruptions_and_separate_max_doc_bound() {
    let fis = real_fields();
    let meta = fixture("asserting_points.kdm");
    let index = fixture("asserting_points.kdi");
    let mut data = fixture("asserting_points.kdd");
    let mut format = AssertingPointsFormat::new(Lucene90PointsFormat);
    let reader = format
        .fields_reader(read_state(&meta, &index, &data, &fis, 9))
        .unwrap();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| reader.values("one"))).is_err()
    );
    let last = data.len() - 1;
    data[last] ^= 1;
    let reader = format
        .fields_reader(read_state(&meta, &index, &data, &fis, 10))
        .unwrap();
    assert!(reader.check_integrity().is_err());
    assert!(format
        .fields_reader(read_state(&meta[..8], &index, &data, &fis, 10))
        .is_err());
}

#[test]
fn actual_java_native_merge_identity_and_acquired_tree_lifecycle() {
    let fis = real_fields();
    let meta = fixture("asserting_points.kdm");
    let index = fixture("asserting_points.kdi");
    let data = fixture("asserting_points.kdd");
    let mut actual = String::new();
    for close_merge in [false, true] {
        let mut format = AssertingPointsFormat::new(Lucene90PointsFormat);
        let mut r = format
            .fields_reader(read_state(&meta, &index, &data, &fis, 10))
            .unwrap();
        let mut m = r.get_merge_instance();
        fn verify<R: PointsReader>(source: &R, other: &mut R) {
            let values = source.values("one").unwrap().unwrap();
            let mut tree = values.point_tree().unwrap();
            other.close().unwrap();
            assert!(source.values("one").unwrap().is_none());
            assert!(other.values("one").unwrap().is_none());
            assert_eq!(values.size(), 13);
            assert_eq!(values.min_packed_value().unwrap(), [0, 0]);
            assert_eq!(values.num_dimensions().unwrap(), 1);
            let mut points = Collect(Vec::new());
            tree.visit_doc_values(&mut points).unwrap();
            assert_eq!(points.0.len(), 13);
            assert_eq!(
                values.point_tree().err().unwrap().to_string(),
                "closed points reader"
            );
            assert_eq!(
                source.check_integrity().unwrap_err().to_string(),
                "closed points reader"
            );
            assert!(source.get_merge_instance().values("one").unwrap().is_none());
        }
        if close_merge {
            verify(&r, &mut m);
        } else {
            verify(&m, &mut r);
        }
        assert!(r.delegate().native().is_none());
        actual.push_str(&format!(
            "{}\tnull\t13\t13\tclosed\tclosed\tnull\n",
            if close_merge { "merge" } else { "original" }
        ));
    }
    assert_eq!(
        actual,
        String::from_utf8(fixture("asserting_points_native_lifecycle.tsv")).unwrap()
    );
}

#[test]
fn absent_point_fields_and_closed_field_validation_match_native_reader_map() {
    let mut fis = real_fields().fields;
    fis.push(FieldInfo::new("absent", 2).with_points(1, 1, 2));
    fis.push(FieldInfo::new("no-points", 3));
    let fis = FieldInfos::new(fis).unwrap();
    let meta = fixture("asserting_points.kdm");
    let index = fixture("asserting_points.kdi");
    let data = fixture("asserting_points.kdd");
    let mut format = AssertingPointsFormat::new(Lucene90PointsFormat);
    let mut reader = format
        .fields_reader(read_state(&meta, &index, &data, &fis, 10))
        .unwrap();
    assert!(reader.delegate().native().is_some());
    assert!(reader.values("absent").unwrap().is_none());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| reader.values("no-points")))
            .is_err()
    );
    reader.close().unwrap();
    assert!(reader.values("absent").unwrap().is_none());
    assert_eq!(
        reader
            .delegate()
            .values("unknown")
            .err()
            .unwrap()
            .to_string(),
        "field=\"unknown\" is unrecognized"
    );
    assert_eq!(
        reader
            .delegate()
            .values("no-points")
            .err()
            .unwrap()
            .to_string(),
        "field=\"no-points\" did not index point values"
    );
}
