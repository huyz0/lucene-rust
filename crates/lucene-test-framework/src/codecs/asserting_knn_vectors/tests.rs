#![cfg(test)]
use super::*;
use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_codecs::hnsw::KnnCollector;
use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

#[derive(Default)]
struct State {
    events: Vec<String>,
    fail: String,
    size: i32,
    dimension: usize,
    initial: i32,
    closes: i32,
    missing: bool,
    accept_identity: Option<usize>,
}
type Shared = Rc<RefCell<State>>;
fn shared() -> Shared {
    Rc::new(RefCell::new(State {
        size: 3,
        dimension: 2,
        initial: -1,
        ..State::default()
    }))
}
fn event(s: &Shared, name: impl Into<String>) -> Result<()> {
    let name = name.into();
    let mut s = s.borrow_mut();
    s.events.push(name.clone());
    if name == "close" {
        s.closes += 1;
    }
    if s.fail == name || (s.fail == "close2" && name == "close" && s.closes == 2) {
        Err(failure("delegate failure"))
    } else {
        Ok(())
    }
}
fn infos() -> FieldInfos {
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
        FieldInfo::new("plain", 2),
    ])
    .unwrap()
}
fn outcome(action: impl FnOnce() -> Result<()>) -> String {
    match catch_unwind(AssertUnwindSafe(action)) {
        Ok(Ok(())) => "ok".into(),
        Ok(Err(e)) => format!("io:{e}"),
        Err(_) => "assert".into(),
    }
}
struct ProbeIterator {
    state: Option<Shared>,
    doc: i32,
}
impl DocIdSetIterator for ProbeIterator {
    fn doc_id(&self) -> i32 {
        if let Some(s) = &self.state {
            s.borrow_mut().events.push("docID".into());
        }
        self.doc
    }
    fn next_doc(&mut self) -> lucene_search::Result<i32> {
        self.doc = if self.doc < 2 {
            self.doc + 1
        } else {
            NO_MORE_DOCS
        };
        Ok(self.doc)
    }
    fn advance(&mut self, target: i32) -> lucene_search::Result<i32> {
        self.doc = target;
        Ok(target)
    }
    fn cost(&self) -> i64 {
        3
    }
}
struct ProbeValues<T> {
    state: Shared,
    identity: Rc<()>,
    marker: std::marker::PhantomData<T>,
}
impl<T> Clone for ProbeValues<T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            identity: self.identity.clone(),
            marker: std::marker::PhantomData,
        }
    }
}
impl<T> VectorValues<T> for ProbeValues<T> {
    fn size(&self) -> i32 {
        let mut s = self.state.borrow_mut();
        s.events.push("size".into());
        s.size
    }
    fn dimension(&self) -> usize {
        let mut s = self.state.borrow_mut();
        s.events.push("dimension".into());
        s.dimension
    }
    fn iterator(&self) -> Result<Box<dyn DocIdSetIterator + '_>> {
        let mut s = self.state.borrow_mut();
        s.events.push("iterator".into());
        Ok(Box::new(ProbeIterator {
            state: Some(self.state.clone()),
            doc: s.initial,
        }))
    }
    fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        Ok(ord)
    }
    fn vector(&self, _ord: i32) -> Result<Vec<T>> {
        Ok(Vec::new())
    }
}
#[derive(Clone, Copy)]
struct EmptyGraph;
impl HnswGraphView for EmptyGraph {
    fn size(&self) -> i32 {
        0
    }
    fn num_levels(&self) -> i32 {
        0
    }
    fn entry_node(&self) -> i32 {
        -1
    }
    fn max_conn(&self) -> i32 {
        0
    }
    fn neighbors_into(
        &self,
        _level: i32,
        _node: i32,
        out: &mut Vec<i32>,
    ) -> std::result::Result<(), lucene_codecs::vectors::Error> {
        out.clear();
        Ok(())
    }
    fn sorted_nodes_on_level(
        &self,
        _level: i32,
    ) -> std::result::Result<Vec<i32>, lucene_codecs::vectors::Error> {
        Ok(Vec::new())
    }
}
struct ProbeReader {
    state: Shared,
    float: ProbeValues<f32>,
    byte: ProbeValues<u8>,
}
impl ProbeReader {
    fn new(state: Shared) -> Self {
        Self {
            float: ProbeValues {
                state: state.clone(),
                identity: Rc::new(()),
                marker: std::marker::PhantomData,
            },
            byte: ProbeValues {
                state: state.clone(),
                identity: Rc::new(()),
                marker: std::marker::PhantomData,
            },
            state,
        }
    }
}
impl KnnVectorsReader for ProbeReader {
    type Floats<'a> = ProbeValues<f32>;
    type Bytes<'a> = ProbeValues<u8>;
    type Graph<'a> = EmptyGraph;
    fn float_values(&self, field: &str) -> Result<Option<Self::Floats<'_>>> {
        event(&self.state, format!("float:{field}"))?;
        Ok((!self.state.borrow().missing).then(|| self.float.clone()))
    }
    fn byte_values(&self, field: &str) -> Result<Option<Self::Bytes<'_>>> {
        event(&self.state, format!("byte:{field}"))?;
        Ok((!self.state.borrow().missing).then(|| self.byte.clone()))
    }
    fn search_float(
        &self,
        field: &str,
        _target: &[f32],
        _collector: &mut dyn KnnCollect,
        accept: &dyn AcceptDocs,
    ) -> Result<()> {
        self.search(field, accept)
    }
    fn search_byte(
        &self,
        field: &str,
        _target: &[u8],
        _collector: &mut dyn KnnCollect,
        accept: &dyn AcceptDocs,
    ) -> Result<()> {
        self.search(field, accept)
    }
    fn check_integrity(&self) -> Result<()> {
        event(&self.state, "integrity")
    }
    fn get_merge_instance(&self) -> Result<Option<Self>> {
        event(&self.state, "merge")?;
        Ok((!self.state.borrow().missing).then(|| Self::new(self.state.clone())))
    }
    fn finish_merge(&mut self) -> Result<()> {
        event(&self.state, "finishMerge")
    }
    fn close(&mut self) -> Result<()> {
        event(&self.state, "close")
    }
    fn off_heap_byte_size(&self, field: &FieldInfo) -> BTreeMap<String, i64> {
        self.state
            .borrow_mut()
            .events
            .push(format!("memory:{}", field.name));
        BTreeMap::from([("vec".into(), 17)])
    }
    fn graph(&self, field: &str) -> Result<Option<Self::Graph<'_>>> {
        event(&self.state, format!("graph:{field}"))?;
        Ok(Some(EmptyGraph))
    }
}
impl ProbeReader {
    fn search(&self, field: &str, accept: &dyn AcceptDocs) -> Result<()> {
        if let Some(expected) = self.state.borrow().accept_identity {
            assert_eq!(
                accept as *const dyn AcceptDocs as *const () as usize,
                expected
            );
        }
        event(&self.state, format!("search:{field}"))?;
        self.state.borrow_mut().events.push(format!(
            "acceptWrapped:{}",
            if accept.is_asserting() {
                "AssertingAcceptDocs"
            } else {
                "ProbeAccept"
            }
        ));
        accept.bits()?;
        accept.cost()?;
        accept.iterator()?;
        Ok(())
    }
}
struct ProbeAccept(Shared);
impl AcceptDocs for ProbeAccept {
    fn bits(&self) -> Result<Option<&[bool]>> {
        event(&self.0, "bits")?;
        Ok(None)
    }
    fn cost(&self) -> Result<i32> {
        event(&self.0, "cost")?;
        Ok(3)
    }
    fn iterator(&self) -> Result<Box<dyn DocIdSetIterator + '_>> {
        event(&self.0, "acceptIterator")?;
        Ok(Box::new(ProbeIterator {
            state: None,
            doc: self.0.borrow().initial,
        }))
    }
}
fn reader_trace(program: &str) -> (String, String) {
    let fields = infos();
    let state = shared();
    let raw = ProbeReader::new(state.clone());
    let float_identity = raw.float.identity.clone();
    let byte_identity = raw.byte.identity.clone();
    let mut base = AssertingKnnVectorsReader::new(raw, &fields);
    let mut merged = [None, None];
    let accept = ProbeAccept(state.clone());
    let mut results = Vec::new();
    for token in program.split(',') {
        results.push(outcome(|| {
            let p: Vec<_> = token.split(':').collect();
            let field = p.get(1).copied();
            let mut collector = KnnCollector::new(1, 100);
            match p[0] {
                "f" => assert!(Rc::ptr_eq(
                    &base
                        .float_values(field.unwrap_or("float"))?
                        .unwrap()
                        .identity,
                    &float_identity
                )),
                "b" => assert!(Rc::ptr_eq(
                    &base.byte_values(field.unwrap_or("byte"))?.unwrap().identity,
                    &byte_identity
                )),
                "sf" => {
                    base.search_float(field.unwrap_or("float"), &[0., 1.], &mut collector, &accept)?
                }
                "sb" => {
                    base.search_byte(field.unwrap_or("byte"), &[0, 1], &mut collector, &accept)?
                }
                "merge" => merged[0] = base.get_merge_instance()?,
                "merge2" => merged[1] = base.get_merge_instance()?,
                "mf" => {
                    merged[0].as_ref().unwrap().float_values("float")?;
                }
                "mb" => {
                    merged[0].as_ref().unwrap().byte_values("byte")?;
                }
                "msf" => merged[0].as_ref().unwrap().search_float(
                    "float",
                    &[0., 1.],
                    &mut collector,
                    &accept,
                )?,
                "msb" => merged[0].as_ref().unwrap().search_byte(
                    "byte",
                    &[0, 1],
                    &mut collector,
                    &accept,
                )?,
                "mm" => {
                    merged[0].as_ref().unwrap().get_merge_instance()?;
                }
                "mc" => merged[0].as_mut().unwrap().close()?,
                "finish" => base.finish_merge()?,
                "mfinish" => merged[0].as_mut().unwrap().finish_merge()?,
                "finish2" => merged[1].as_mut().unwrap().finish_merge()?,
                "close" => base.close()?,
                "integrity" => base.check_integrity()?,
                "memory" => assert_eq!(
                    base.off_heap_byte_size(&fields.fields[0]),
                    BTreeMap::from([("vec".into(), 17)])
                ),
                "graph" => assert!(base.graph("float")?.is_some()),
                "fail" => state.borrow_mut().fail = token[5..].into(),
                "ok" => state.borrow_mut().fail.clear(),
                "size" => state.borrow_mut().size = p[1].parse().unwrap(),
                "dim" => state.borrow_mut().dimension = p[1].parse().unwrap(),
                "initial" => state.borrow_mut().initial = p[1].parse().unwrap(),
                "missing" => state.borrow_mut().missing = true,
                _ => panic!("unknown command {token}"),
            }
            Ok(())
        }));
    }
    let events = state.borrow().events.join(",");
    (results.join(";"), events)
}
#[derive(Clone)]
struct Task(Shared);
impl DeferredWork for Task {
    fn run(&mut self) -> Result<()> {
        event(&self.0, "deferred")
    }
}
struct ProbeWriter {
    state: Shared,
    handle: Rc<()>,
    task: Task,
}
impl KnnVectorsWriter for ProbeWriter {
    type FieldWriter = Rc<()>;
    type Deferred = Task;
    fn add_field(&mut self, field: &FieldInfo) -> Result<Self::FieldWriter> {
        event(&self.state, format!("add:{}", field.name))?;
        Ok(self.handle.clone())
    }
    fn flush(&mut self, max_doc: i32, map: Option<&[i32]>) -> Result<()> {
        event(&self.state, format!("flush:{max_doc}:{}", map.is_none()))
    }
    fn merge_one_field(
        &mut self,
        field: Option<&FieldInfo>,
        _state: Option<&MergeState<'_>>,
    ) -> Result<Option<Self::Deferred>> {
        event(&self.state, format!("mergeOne:{}", field.unwrap().name))?;
        Ok((!self.state.borrow().missing).then(|| self.task.clone()))
    }
    fn finish(&mut self) -> Result<()> {
        event(&self.state, "finish")
    }
    fn close(&mut self) -> Result<()> {
        event(&self.state, "close")
    }
    fn ram_bytes_used(&self) -> usize {
        self.state.borrow_mut().events.push("ram".into());
        17
    }
}
fn writer_trace(name: &str) -> (String, String) {
    if name.starts_with("merge-reversed") {
        return reversed_merge_trace(name);
    }
    let state = shared();
    let fields = infos();
    let handle = Rc::new(());
    let task = Task(state.clone());
    let mut writer = AssertingKnnVectorsWriter::new(ProbeWriter {
        state: state.clone(),
        handle: handle.clone(),
        task: task.clone(),
    });
    let merge = MergeState {
        fields: &fields,
        integrity: &[],
        check_aborted: &|| Ok(()),
    };
    let status = outcome(|| {
        match name {
            "all" => {
                assert!(Rc::ptr_eq(&writer.add_field(&fields.fields[0])?, &handle));
                writer.flush(7, None)?;
                let mut deferred = writer
                    .merge_one_field(Some(&fields.fields[0]), Some(&merge))?
                    .unwrap();
                assert!(Rc::ptr_eq(&deferred.0, &task.0));
                deferred.run()?;
                writer.finish()?;
                assert_eq!(writer.ram_bytes_used(), 17);
                writer.close()?;
            }
            "null-field" => {
                writer.merge_one_field(None, Some(&merge))?;
            }
            "null-state" => {
                writer.merge_one_field(Some(&fields.fields[0]), None)?;
            }
            "no-task" => {
                state.borrow_mut().missing = true;
                assert!(writer
                    .merge_one_field(Some(&fields.fields[0]), Some(&merge))?
                    .is_none());
            }
            "deferred-error" => {
                state.borrow_mut().fail = "deferred".into();
                writer
                    .merge_one_field(Some(&fields.fields[0]), Some(&merge))?
                    .unwrap()
                    .run()?;
            }
            "merge-order" => writer.merge(&merge)?,
            _ => {
                state.borrow_mut().fail = name.into();
                match name {
                    "add:float" => {
                        writer.add_field(&fields.fields[0])?;
                    }
                    "flush:7:true" => writer.flush(7, None)?,
                    "mergeOne:float" => {
                        writer.merge_one_field(Some(&fields.fields[0]), Some(&merge))?;
                    }
                    "finish" => writer.finish()?,
                    "close" => writer.close()?,
                    _ => panic!("unknown case"),
                }
            }
        }
        Ok(())
    });
    let events = state.borrow().events.join(",");
    (status, events)
}
fn accept_trace(name: &str) -> (String, String) {
    let state = shared();
    let raw = ProbeAccept(state.clone());
    let wrapped = AssertingAcceptDocs::wrap(&raw);
    let status = outcome(|| {
        match name {
            "all" => {
                let twice = AssertingAcceptDocs::wrap(&wrapped);
                assert!(
                    matches!(twice,AcceptDocsGuard::Existing(v) if std::ptr::eq(v,&wrapped as &dyn AcceptDocs))
                );
                wrapped.bits()?;
                wrapped.cost()?;
                wrapped.iterator()?;
            }
            "positioned" => {
                state.borrow_mut().initial = 0;
                wrapped.iterator()?;
            }
            _ => {
                state.borrow_mut().fail = name.into();
                match name {
                    "bits" => {
                        wrapped.bits()?;
                    }
                    "cost" => {
                        wrapped.cost()?;
                    }
                    "acceptIterator" => {
                        wrapped.iterator()?;
                    }
                    _ => panic!("unknown case"),
                }
            }
        }
        Ok(())
    });
    let events = state.borrow().events.join(",");
    (status, events)
}
#[test]
fn actual_java_assertions_identity_callback_order_and_lifecycle() {
    let fixture =
        include_str!("../../../../../fixtures/testframework/data/asserting_knn_vectors.tsv");
    let mut checked = 0;
    for row in fixture.lines() {
        let p: Vec<_> = row.split('\t').collect();
        if p[0] == "accept" && p[1] == "thread" {
            continue;
        }
        let actual = match p[0] {
            "reader" => reader_trace(p[1]),
            "writer" => writer_trace(p[1]),
            "accept" => accept_trace(p[1]),
            _ => panic!("unknown fixture"),
        };
        assert_eq!(
            actual,
            (p[2].into(), p.get(3).unwrap_or(&"").to_string()),
            "{row}"
        );
        checked += 1;
    }
    assert_eq!(checked, 42);
}
#[test]
fn already_asserting_accept_docs_reach_both_delegate_searches_by_identity() {
    let state = shared();
    let fields = infos();
    let raw = ProbeAccept(state.clone());
    let accept = AssertingAcceptDocs::new(&raw);
    state.borrow_mut().accept_identity = Some(&accept as *const _ as usize);
    let reader = AssertingKnnVectorsReader::new(ProbeReader::new(state), &fields);
    let mut collector = KnnCollector::new(1, 100);
    reader
        .search_float("float", &[0., 1.], &mut collector, &accept)
        .unwrap();
    reader
        .search_byte("byte", &[0, 1], &mut collector, &accept)
        .unwrap();
}
struct OrderedTask(Shared, String);
impl DeferredWork for OrderedTask {
    fn run(&mut self) -> Result<()> {
        event(&self.0, format!("deferred:{}", self.1))
    }
}
struct OrderedWriter(Shared);
impl KnnVectorsWriter for OrderedWriter {
    type FieldWriter = ();
    type Deferred = OrderedTask;
    fn add_field(&mut self, _: &FieldInfo) -> Result<()> {
        unreachable!()
    }
    fn flush(&mut self, _: i32, _: Option<&[i32]>) -> Result<()> {
        unreachable!()
    }
    fn merge_one_field(
        &mut self,
        field: Option<&FieldInfo>,
        _: Option<&MergeState<'_>>,
    ) -> Result<Option<OrderedTask>> {
        let field = field.unwrap();
        event(&self.0, format!("mergeOne:{}", field.name))?;
        Ok(Some(OrderedTask(self.0.clone(), field.name.clone())))
    }
    fn finish(&mut self) -> Result<()> {
        event(&self.0, "finish")
    }
    fn close(&mut self) -> Result<()> {
        unreachable!()
    }
    fn ram_bytes_used(&self) -> usize {
        unreachable!()
    }
}
fn reversed_merge_trace(name: &str) -> (String, String) {
    let state = shared();
    state.borrow_mut().fail = match name {
        "merge-reversed-field-error" => "mergeOne:byte",
        "merge-reversed-deferred-error" => "deferred:float",
        _ => "",
    }
    .into();
    let mut fields = infos().fields;
    fields.reverse();
    let fields = FieldInfos::new(fields).unwrap();
    assert_eq!(fields.fields[0].name, "plain");
    let merge = MergeState {
        fields: &fields,
        integrity: &[],
        check_aborted: &|| Ok(()),
    };
    let mut writer = AssertingKnnVectorsWriter::new(OrderedWriter(state.clone()));
    let status = outcome(|| writer.merge(&merge));
    let events = state.borrow().events.join(",");
    (status, events)
}
#[test]
fn accept_docs_cross_thread_checks_precede_callbacks() {
    let raw = BitsAcceptDocs::new(None, 3);
    let wrapped = AssertingAcceptDocs::new(&raw);
    for op in 0..3 {
        thread::scope(|scope| {
            let wrapped = &wrapped;
            assert!(scope
                .spawn(move || catch_unwind(AssertUnwindSafe(|| match op {
                    0 => {
                        wrapped.bits().unwrap();
                    }
                    1 => {
                        wrapped.cost().unwrap();
                    }
                    _ => {
                        wrapped.iterator().unwrap();
                    }
                }))
                .is_err())
                .join()
                .unwrap());
        });
    }
}
#[test]
fn bits_iterators_filter_advance_and_exhaustion() {
    let bits = [false, true, false, true];
    let accept = BitsAcceptDocs::new(Some(&bits), 4);
    assert_eq!(accept.bits().unwrap(), Some(bits.as_slice()));
    assert_eq!(accept.cost().unwrap(), 4);
    let mut iterator = accept.iterator().unwrap();
    assert_eq!(iterator.doc_id(), -1);
    assert_eq!(iterator.cost(), 4);
    assert_eq!(iterator.next_doc().unwrap(), 1);
    assert_eq!(iterator.advance(3).unwrap(), 3);
    assert_eq!(iterator.next_doc().unwrap(), NO_MORE_DOCS);
    assert_eq!(iterator.next_doc().unwrap(), NO_MORE_DOCS);
    assert!(catch_unwind(|| BitsAcceptDocs::new(None, -1)).is_err());
    assert!(catch_unwind(|| BitsAcceptDocs::new(Some(&bits), 3)).is_err());
    let all = BitsAcceptDocs::new(None, 2);
    assert!(all.bits().unwrap().is_none());
    let mut it = all.iterator().unwrap();
    assert_eq!(it.advance(1).unwrap(), 1);
    assert_eq!(it.next_doc().unwrap(), NO_MORE_DOCS);
}
#[test]
fn inherited_merge_aborts_and_integrity_errors_before_any_field_callback() {
    let fields = infos();
    let state = shared();
    let mut w = AssertingKnnVectorsWriter::new(ProbeWriter {
        state: state.clone(),
        handle: Rc::new(()),
        task: Task(state.clone()),
    });
    let integrity = || event(&state, "integrity");
    let abort = || event(&state, "abort");
    let sources: [&dyn Fn() -> Result<()>; 1] = [&integrity];
    let merge = MergeState {
        fields: &fields,
        integrity: &sources,
        check_aborted: &abort,
    };
    state.borrow_mut().fail = "abort".into();
    assert!(w.merge(&merge).is_err());
    assert_eq!(state.borrow().events, ["abort"]);
    state.borrow_mut().events.clear();
    state.borrow_mut().fail = "integrity".into();
    assert!(w.merge(&merge).is_err());
    assert_eq!(state.borrow().events, ["abort", "integrity"]);
    state.borrow_mut().events.clear();
    state.borrow_mut().fail = "mergeOne:float".into();
    assert!(w.merge(&merge).is_err());
    assert_eq!(
        state.borrow().events,
        ["abort", "integrity", "mergeOne:float"]
    );
    assert!(Rc::ptr_eq(&w.delegate().state, &state));
    assert!(Rc::ptr_eq(&w.delegate_mut().state, &state));
}
struct ProbeFormat {
    state: Shared,
}
impl KnnVectorsFormat for ProbeFormat {
    type Reader<'a> = ProbeReader;
    type Writer = ProbeWriter;
    fn fields_reader<'a>(&'a self, _state: ReadState<'a>) -> Result<Self::Reader<'a>> {
        event(&self.state, "reader-open")?;
        Ok(ProbeReader::new(self.state.clone()))
    }
    fn fields_writer(&mut self) -> Result<Self::Writer> {
        event(&self.state, "writer-open")?;
        Ok(ProbeWriter {
            state: self.state.clone(),
            handle: Rc::new(()),
            task: Task(self.state.clone()),
        })
    }
}
#[test]
fn format_factory_forwarding_constants_and_open_failures() {
    let state = shared();
    let mut format = AssertingKnnVectorsFormat::new(ProbeFormat {
        state: state.clone(),
    });
    let fields = infos();
    let id = [0; 16];
    let read = ReadState {
        files: Files {
            vec: &[],
            vemf: &[],
            vex: &[],
            vem: &[],
        },
        segment: SegmentReadState {
            max_doc: 3,
            segment_id: &id,
            segment_suffix: "",
            field_infos: &fields,
        },
    };
    assert_eq!(AssertingKnnVectorsFormat::<ProbeFormat>::NAME, "Asserting");
    assert_eq!(format.max_dimensions("float"), 1024);
    assert!(Rc::ptr_eq(&format.delegate().state, &state));
    assert!(Rc::ptr_eq(&format.delegate_mut().state, &state));
    let reader = format.fields_reader(read).unwrap();
    assert!(reader.float_values("float").unwrap().is_some());
    assert!(Rc::ptr_eq(&reader.delegate().state, &state));
    drop(reader);
    let mut writer = format.fields_writer().unwrap();
    writer.flush(3, Some(&[2, 1, 0])).unwrap();
    assert!(Rc::ptr_eq(&writer.delegate_mut().state, &state));
    state.borrow_mut().fail = "reader-open".into();
    assert!(format.fields_reader(read).is_err());
    state.borrow_mut().fail = "writer-open".into();
    assert!(format.fields_writer().is_err());
    let float = ProbeValues::<f32> {
        state: state.clone(),
        identity: Rc::new(()),
        marker: std::marker::PhantomData,
    };
    assert_eq!(float.ord_to_doc(2).unwrap(), 2);
    assert!(float.vector(0).unwrap().is_empty());
    let mut it = float.iterator().unwrap();
    assert_eq!(it.next_doc().unwrap(), 0);
    assert_eq!(it.advance(2).unwrap(), 2);
    assert_eq!(it.cost(), 3);
    assert_eq!(EmptyGraph.size(), 0);
    assert_eq!(EmptyGraph.num_levels(), 0);
    assert_eq!(EmptyGraph.entry_node(), -1);
    assert_eq!(EmptyGraph.max_conn(), 0);
    let mut neighbors = vec![1];
    EmptyGraph.neighbors_into(0, 0, &mut neighbors).unwrap();
    assert!(neighbors.is_empty());
    assert!(EmptyGraph.sorted_nodes_on_level(0).unwrap().is_empty());
}
