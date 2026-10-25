//! Actual CrankyCodec factories, component identity and one shared RNG stream.
//! Only four probe components perform callbacks here. The other component
//! operation contracts belong to their existing format-specific fixtures.
use lucene_codecs::{
    compound_format,
    field_infos::{self, FieldInfos},
    norms,
    segment_state::SegmentWriteState,
    stored_fields,
};
use lucene_index::{
    compound_reader::CompoundReader,
    segment_info::{self, LuceneVersion, SegmentInfo},
};
use lucene_store::{ByteBuffersDirectory, Directory};
use lucene_test_framework::codecs::{
    cranky_codec::{Codec, CrankyCodec},
    cranky_compound::{CompoundFormat, CompoundInfo},
    cranky_doc_values as dv,
    cranky_field_infos::FieldInfosFormat,
    cranky_live_docs::{LiveDocsFormat, LiveDocsInfo},
    cranky_norms as norm, cranky_points as point, cranky_postings as post,
    cranky_segment_info::SegmentInfoFormatDelegate,
    cranky_stored_fields as stored, cranky_term_vectors as vector,
};
use lucene_util::{fixed_bit_set::FixedBitSet, java_random::JavaRandom};
use std::{
    cell::{Cell, RefCell},
    fmt, io,
    panic::{catch_unwind, AssertUnwindSafe},
    rc::Rc,
};

const KINDS: [&str; 11] = [
    "docValues",
    "fieldInfos",
    "liveDocs",
    "norms",
    "postings",
    "segmentInfo",
    "storedFields",
    "termVectors",
    "compound",
    "points",
    "knnVectors",
];
const OPS: [&str; 11] = [
    "held-fieldInfos-write",
    "liveDocs-write",
    "segmentInfo-write",
    "compound-write",
    "fieldInfos-read",
    "liveDocs-read",
    "liveDocs-files",
    "all-accessors",
    "display",
    "failed-factory",
    "external-draw",
];
const ID: [u8; 16] = [0; 16];

struct State {
    events: RefCell<Vec<String>>,
    fail_factory: Cell<Option<usize>>,
    fail_operation: Cell<bool>,
    fail_display: Cell<bool>,
    last: RefCell<[Option<Rc<()>>; 11]>,
}
impl Default for State {
    fn default() -> Self {
        Self {
            events: RefCell::new(vec![]),
            fail_factory: Cell::new(None),
            fail_operation: Cell::new(false),
            fail_display: Cell::new(false),
            last: RefCell::new(std::array::from_fn(|_| None)),
        }
    }
}
#[derive(Clone)]
struct Handle {
    identity: Rc<()>,
    state: Rc<State>,
}
impl Handle {
    fn call(&self, name: &str) -> io::Result<()> {
        self.state.events.borrow_mut().push(name.into());
        if self.state.fail_operation.get() {
            Err(io::Error::other(format!("delegate:{name}")))
        } else {
            Ok(())
        }
    }
}
struct RecordingCodec {
    state: Rc<State>,
    components: [Handle; 11],
}
impl RecordingCodec {
    fn new() -> Self {
        let state = Rc::new(State::default());
        Self {
            components: std::array::from_fn(|_| Handle {
                identity: Rc::new(()),
                state: state.clone(),
            }),
            state,
        }
    }
    fn component(&self, kind: usize) -> Result<Handle, String> {
        self.state
            .events
            .borrow_mut()
            .push(format!("factory:{}", KINDS[kind]));
        if self.state.fail_factory.get() == Some(kind) {
            return Err(format!("factory:{}", KINDS[kind]));
        }
        let mut value = self.components[kind].clone();
        if kind == 9 {
            value.identity = Rc::new(());
        }
        self.state.last.borrow_mut()[kind] = Some(value.identity.clone());
        Ok(value)
    }
}
impl fmt::Display for RecordingCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.state.events.borrow_mut().push("display".into());
        if self.state.fail_display.get() {
            std::panic::panic_any("display");
        }
        f.write_str("DelegateDisplay")
    }
}
impl Codec for RecordingCodec {
    type Error = String;
    type DocValues = Handle;
    type FieldInfos = Handle;
    type LiveDocs = Handle;
    type Norms = Handle;
    type Postings = Handle;
    type SegmentInfo = Handle;
    type StoredFields = Handle;
    type TermVectors = Handle;
    type Compound = Handle;
    type Points = Handle;
    type KnnVectors = Handle;
    fn name(&self) -> &str {
        "CodecProbe"
    }
    fn doc_values_format(&self) -> Result<Self::DocValues, Self::Error> {
        self.component(0)
    }
    fn field_infos_format(&self) -> Result<Self::FieldInfos, Self::Error> {
        self.component(1)
    }
    fn live_docs_format(&self) -> Result<Self::LiveDocs, Self::Error> {
        self.component(2)
    }
    fn norms_format(&self) -> Result<Self::Norms, Self::Error> {
        self.component(3)
    }
    fn postings_format(&self) -> Result<Self::Postings, Self::Error> {
        self.component(4)
    }
    fn segment_info_format(&self) -> Result<Self::SegmentInfo, Self::Error> {
        self.component(5)
    }
    fn stored_fields_format(&self) -> Result<Self::StoredFields, Self::Error> {
        self.component(6)
    }
    fn term_vectors_format(&self) -> Result<Self::TermVectors, Self::Error> {
        self.component(7)
    }
    fn compound_format(&self) -> Result<Self::Compound, Self::Error> {
        self.component(8)
    }
    fn points_format(&self) -> Result<Self::Points, Self::Error> {
        self.component(9)
    }
    fn knn_vectors_format(&self) -> Result<Self::KnnVectors, Self::Error> {
        self.component(10)
    }
}
impl FieldInfosFormat for Handle {
    fn read(
        &self,
        _: &dyn Directory,
        _: &str,
        _: &[u8; 16],
        _: &str,
    ) -> field_infos::Result<FieldInfos> {
        self.call("fieldInfos:read")
            .map_err(lucene_store::Error::Io)?;
        Ok(FieldInfos::new(vec![]).unwrap())
    }
    fn write(
        &mut self,
        _: &dyn Directory,
        _: &str,
        _: &[u8; 16],
        _: &str,
        _: &FieldInfos,
    ) -> field_infos::Result<()> {
        self.call("fieldInfos:write")
            .map_err(lucene_store::Error::Io)?;
        Ok(())
    }
}
impl LiveDocsFormat for Handle {
    fn read_live_docs(&self, _: &dyn Directory, _: &LiveDocsInfo<'_>) -> io::Result<FixedBitSet> {
        self.call("liveDocs:read")?;
        Ok(FixedBitSet::new(1))
    }
    fn write_live_docs(
        &self,
        _: &FixedBitSet,
        _: &dyn Directory,
        _: &LiveDocsInfo<'_>,
        _: usize,
    ) -> io::Result<()> {
        self.call("liveDocs:write")
    }
    fn files(&self, _: &LiveDocsInfo<'_>, files: &mut Vec<String>) -> io::Result<()> {
        self.call("liveDocs:files")?;
        files.push("delegate-file".into());
        Ok(())
    }
}
impl SegmentInfoFormatDelegate for Handle {
    fn read(
        &mut self,
        _: &dyn Directory,
        _: &str,
        _: &[u8; 16],
    ) -> segment_info::Result<SegmentInfo> {
        unreachable!("composition fixture never reads segment info")
    }
    fn write(
        &mut self,
        _: &dyn Directory,
        _: &str,
        _: &mut SegmentInfo,
    ) -> segment_info::Result<()> {
        self.call("segmentInfo:write")
            .map_err(lucene_store::Error::Io)?;
        Ok(())
    }
}
impl CompoundFormat for Handle {
    fn get_compound_reader<'d>(
        &mut self,
        _: &'d dyn Directory,
        _: &CompoundInfo<'_>,
    ) -> lucene_store::Result<CompoundReader<'d>> {
        unreachable!("composition fixture never opens compound files")
    }
    fn write(&mut self, _: &dyn Directory, _: &CompoundInfo<'_>) -> compound_format::Result<()> {
        self.call("compound:write")
            .map_err(lucene_store::Error::Io)?;
        Ok(())
    }
}
// These six typed probe interfaces establish factory composition. No native
// format is constructed or substituted for Java's per-field/default factories.
impl dv::DocValuesFormat for Handle {
    type Consumer<'a> = <dv::Lucene90DocValuesFormat as dv::DocValuesFormat>::Consumer<'a>;
    type Producer = <dv::Lucene90DocValuesFormat as dv::DocValuesFormat>::Producer;
    fn name(&self) -> &str {
        "PerFieldDV40"
    }
    fn fields_consumer<'a>(&mut self, _: SegmentWriteState<'a>) -> dv::Result<Self::Consumer<'a>> {
        unreachable!("composition-only component")
    }
    fn fields_producer(&mut self, _: dv::ReadState<'_>) -> dv::Result<Self::Producer> {
        unreachable!("composition-only component")
    }
}
impl post::PostingsFormat for Handle {
    type Consumer<'a> = <post::Lucene104PostingsFormat as post::PostingsFormat>::Consumer<'a>;
    type Producer = <post::Lucene104PostingsFormat as post::PostingsFormat>::Producer;
    fn name(&self) -> &str {
        "PerField40"
    }
    fn fields_consumer<'a>(
        &mut self,
        _: SegmentWriteState<'a>,
    ) -> post::Result<Self::Consumer<'a>> {
        unreachable!("composition-only component")
    }
    fn fields_producer(&mut self, _: post::ReadState<'_>) -> post::Result<Self::Producer> {
        unreachable!("composition-only component")
    }
}
impl norm::NormsFormat for Handle {
    type Consumer = <norm::Lucene90NormsFormat as norm::NormsFormat>::Consumer;
    type Producer = <norm::Lucene90NormsFormat as norm::NormsFormat>::Producer;
    fn norms_consumer(&mut self, _: &norm::NormsState<'_>) -> norms::Result<Self::Consumer> {
        unreachable!("composition-only component")
    }
    fn norms_producer(&mut self, _: &norm::NormsState<'_>) -> norms::Result<Self::Producer> {
        unreachable!("composition-only component")
    }
}
impl point::PointsFormat for Handle {
    type Writer<'a> = <point::Lucene90PointsFormat as point::PointsFormat>::Writer<'a>;
    type Reader<'a> = <point::Lucene90PointsFormat as point::PointsFormat>::Reader<'a>;
    fn fields_writer<'a>(&mut self, _: point::WriteState<'a>) -> point::Result<Self::Writer<'a>> {
        unreachable!("composition-only component")
    }
    fn fields_reader<'a>(&mut self, _: point::ReadState<'a>) -> point::Result<Self::Reader<'a>> {
        unreachable!("composition-only component")
    }
}
impl stored::StoredFieldsFormat for Handle {
    type Writer<'a> =
        <stored::Lucene90StoredFieldsFormat as stored::StoredFieldsFormat>::Writer<'a>;
    type Reader<'a> =
        <stored::Lucene90StoredFieldsFormat as stored::StoredFieldsFormat>::Reader<'a>;
    fn fields_writer<'a>(
        &mut self,
        _: stored::StoredFieldsState<'a>,
    ) -> stored_fields::Result<Self::Writer<'a>> {
        unreachable!("composition-only component")
    }
    fn fields_reader<'a>(
        &mut self,
        _: stored::StoredFieldsReadState<'a>,
    ) -> stored_fields::Result<Self::Reader<'a>> {
        unreachable!("composition-only component")
    }
}
impl vector::TermVectorsFormat for Handle {
    type Writer = <vector::SimpleTextVectorsFormat as vector::TermVectorsFormat>::Writer;
    type Reader = <vector::SimpleTextVectorsFormat as vector::TermVectorsFormat>::Reader;
    fn vectors_writer(&mut self, _: &vector::VectorState<'_>) -> io::Result<Self::Writer> {
        unreachable!("composition-only component")
    }
    fn vectors_reader(&mut self, _: &vector::VectorState<'_>) -> io::Result<Self::Reader> {
        unreachable!("composition-only component")
    }
}
fn row(out: &mut String, fields: &[String]) {
    out.push_str(&fields.join("\t"));
    out.push('\n');
}
fn events(codec: &CrankyCodec<RecordingCodec>) -> String {
    codec.delegate().state.events.borrow().join(",")
}
fn last(codec: &CrankyCodec<RecordingCodec>, kind: usize) -> Rc<()> {
    codec.delegate().state.last.borrow()[kind]
        .as_ref()
        .unwrap()
        .clone()
}
fn outcome(action: impl FnOnce() -> Result<(), String>) -> String {
    match catch_unwind(AssertUnwindSafe(action)) {
        Ok(Ok(())) => "ok".into(),
        Ok(Err(error)) => format!("runtime:{error}"),
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap();
            format!("runtime:{message}")
        }
    }
}
fn component(codec: &CrankyCodec<RecordingCodec>, kind: usize) -> Result<(), String> {
    match kind {
        0 => {
            codec.doc_values_format()?;
        }
        1 => {
            codec.field_infos_format()?;
        }
        2 => {
            codec.live_docs_format()?;
        }
        3 => {
            codec.norms_format()?;
        }
        4 => {
            codec.postings_format()?;
        }
        5 => {
            codec.segment_info_format()?;
        }
        6 => {
            codec.stored_fields_format()?;
        }
        7 => {
            codec.term_vectors_format()?;
        }
        8 => {
            codec.compound_format()?;
        }
        9 => {
            codec.points_format()?;
        }
        10 => {
            codec.knn_vectors_format()?;
        }
        _ => unreachable!(),
    }
    Ok(())
}
fn accessors(
    codec: &CrankyCodec<RecordingCodec>,
    random: &Rc<RefCell<JavaRandom>>,
    out: &mut String,
) {
    for (kind, name) in KINDS.iter().enumerate() {
        codec.delegate().state.events.borrow_mut().clear();
        let refs = Rc::strong_count(random);
        let (fresh, exact, shared, same, format_name, class) = match kind {
            0 => {
                let first = Box::new(codec.doc_values_format().unwrap());
                let first_id = last(codec, kind);
                let second = Box::new(codec.doc_values_format().unwrap());
                let second_id = last(codec, kind);
                let exact = Rc::ptr_eq(&first.delegate().identity, &first_id)
                    && Rc::ptr_eq(&second.delegate().identity, &second_id);
                (
                    !std::ptr::eq(&*first, &*second),
                    exact,
                    Rc::strong_count(random) == refs + 2,
                    Rc::ptr_eq(&first_id, &second_id),
                    dv::DocValuesFormat::name(&*first).to_owned(),
                    "CrankyDocValuesFormat",
                )
            }
            1 => {
                let first = Box::new(codec.field_infos_format().unwrap());
                let first_id = last(codec, kind);
                let second = Box::new(codec.field_infos_format().unwrap());
                let second_id = last(codec, kind);
                let exact = Rc::ptr_eq(&first.delegate().identity, &first_id)
                    && Rc::ptr_eq(&second.delegate().identity, &second_id);
                (
                    !std::ptr::eq(&*first, &*second),
                    exact,
                    Rc::strong_count(random) == refs + 2,
                    Rc::ptr_eq(&first_id, &second_id),
                    "-".to_owned(),
                    "CrankyFieldInfosFormat",
                )
            }
            2 => {
                let first = Box::new(codec.live_docs_format().unwrap());
                let first_id = last(codec, kind);
                let second = Box::new(codec.live_docs_format().unwrap());
                let second_id = last(codec, kind);
                let live = codec.delegate().state.last.borrow();
                let exact = Rc::strong_count(live[kind].as_ref().unwrap()) == 6;
                drop(live);
                (
                    !std::ptr::eq(&*first, &*second),
                    exact,
                    Rc::strong_count(random) == refs + 2,
                    Rc::ptr_eq(&first_id, &second_id),
                    "-".to_owned(),
                    "CrankyLiveDocsFormat",
                )
            }
            3 => {
                let first = Box::new(codec.norms_format().unwrap());
                let first_id = last(codec, kind);
                let second = Box::new(codec.norms_format().unwrap());
                let second_id = last(codec, kind);
                let exact = Rc::ptr_eq(&first.delegate().identity, &first_id)
                    && Rc::ptr_eq(&second.delegate().identity, &second_id);
                (
                    !std::ptr::eq(&*first, &*second),
                    exact,
                    Rc::strong_count(random) == refs + 2,
                    Rc::ptr_eq(&first_id, &second_id),
                    "-".to_owned(),
                    "CrankyNormsFormat",
                )
            }
            4 => {
                let first = Box::new(codec.postings_format().unwrap());
                let first_id = last(codec, kind);
                let second = Box::new(codec.postings_format().unwrap());
                let second_id = last(codec, kind);
                let exact = Rc::ptr_eq(&first.delegate().identity, &first_id)
                    && Rc::ptr_eq(&second.delegate().identity, &second_id);
                (
                    !std::ptr::eq(&*first, &*second),
                    exact,
                    Rc::strong_count(random) == refs + 2,
                    Rc::ptr_eq(&first_id, &second_id),
                    post::PostingsFormat::name(&*first).to_owned(),
                    "CrankyPostingsFormat",
                )
            }
            5 => {
                let first = Box::new(codec.segment_info_format().unwrap());
                let first_id = last(codec, kind);
                let second = Box::new(codec.segment_info_format().unwrap());
                let second_id = last(codec, kind);
                let exact = Rc::ptr_eq(&first.delegate().identity, &first_id)
                    && Rc::ptr_eq(&second.delegate().identity, &second_id);
                (
                    !std::ptr::eq(&*first, &*second),
                    exact,
                    Rc::strong_count(random) == refs + 2,
                    Rc::ptr_eq(&first_id, &second_id),
                    "-".to_owned(),
                    "CrankySegmentInfoFormat",
                )
            }
            6 => {
                let first = Box::new(codec.stored_fields_format().unwrap());
                let first_id = last(codec, kind);
                let second = Box::new(codec.stored_fields_format().unwrap());
                let second_id = last(codec, kind);
                let exact = Rc::ptr_eq(&first.delegate().identity, &first_id)
                    && Rc::ptr_eq(&second.delegate().identity, &second_id);
                (
                    !std::ptr::eq(&*first, &*second),
                    exact,
                    Rc::strong_count(random) == refs + 2,
                    Rc::ptr_eq(&first_id, &second_id),
                    "-".to_owned(),
                    "CrankyStoredFieldsFormat",
                )
            }
            7 => {
                let first = Box::new(codec.term_vectors_format().unwrap());
                let first_id = last(codec, kind);
                let second = Box::new(codec.term_vectors_format().unwrap());
                let second_id = last(codec, kind);
                let exact = Rc::ptr_eq(&first.delegate().identity, &first_id)
                    && Rc::ptr_eq(&second.delegate().identity, &second_id);
                (
                    !std::ptr::eq(&*first, &*second),
                    exact,
                    Rc::strong_count(random) == refs + 2,
                    Rc::ptr_eq(&first_id, &second_id),
                    "-".to_owned(),
                    "CrankyTermVectorsFormat",
                )
            }
            8 => {
                let first = Box::new(codec.compound_format().unwrap());
                let first_id = last(codec, kind);
                let second = Box::new(codec.compound_format().unwrap());
                let second_id = last(codec, kind);
                let exact = Rc::ptr_eq(&first.delegate().identity, &first_id)
                    && Rc::ptr_eq(&second.delegate().identity, &second_id);
                (
                    !std::ptr::eq(&*first, &*second),
                    exact,
                    Rc::strong_count(random) == refs + 2,
                    Rc::ptr_eq(&first_id, &second_id),
                    "-".to_owned(),
                    "CrankyCompoundFormat",
                )
            }
            9 => {
                let first = Box::new(codec.points_format().unwrap());
                let first_id = last(codec, kind);
                let second = Box::new(codec.points_format().unwrap());
                let second_id = last(codec, kind);
                let exact = Rc::ptr_eq(&first.delegate().identity, &first_id)
                    && Rc::ptr_eq(&second.delegate().identity, &second_id);
                (
                    !std::ptr::eq(&*first, &*second),
                    exact,
                    Rc::strong_count(random) == refs + 2,
                    Rc::ptr_eq(&first_id, &second_id),
                    "-".to_owned(),
                    "CrankyPointsFormat",
                )
            }
            10 => {
                let first = codec.knn_vectors_format().unwrap();
                let first_id = last(codec, kind);
                let second = codec.knn_vectors_format().unwrap();
                let second_id = last(codec, kind);
                (
                    !Rc::ptr_eq(&first.identity, &second.identity),
                    Rc::ptr_eq(&first.identity, &first_id)
                        && Rc::ptr_eq(&second.identity, &second_id),
                    Rc::strong_count(random) == refs,
                    Rc::ptr_eq(&first_id, &second_id),
                    "-".to_owned(),
                    "delegate",
                )
            }
            _ => unreachable!(),
        };
        assert!(exact && shared, "{name}: component or RNG identity changed");
        row(
            out,
            &[
                "accessor".into(),
                (*name).into(),
                class.into(),
                fresh.to_string(),
                exact.to_string(),
                if kind == 10 {
                    "-".into()
                } else {
                    shared.to_string()
                },
                same.to_string(),
                format_name,
                random.borrow_mut().next_int().to_string(),
                events(codec),
            ],
        );
    }
}
fn configuration(out: &mut String) {
    let random = Rc::new(RefCell::new(JavaRandom::new(42)));
    let codec = CrankyCodec::new(RecordingCodec::new(), random.clone());
    assert!(Rc::ptr_eq(codec.random(), &random));
    row(
        out,
        &["metadata".into(), "name".into(), codec.name().into()],
    );
    row(
        out,
        &["metadata".into(), "display".into(), format!("{codec}")],
    );
    assert_eq!(events(&codec), "display");
    accessors(&codec, &random, out);
    for (kind, name) in KINDS.iter().enumerate() {
        codec.delegate().state.events.borrow_mut().clear();
        codec.delegate().state.fail_factory.set(Some(kind));
        let status = outcome(|| component(&codec, kind));
        codec.delegate().state.fail_factory.set(None);
        row(
            out,
            &[
                "factory-failure".into(),
                (*name).into(),
                status,
                random.borrow_mut().next_int().to_string(),
                events(&codec),
            ],
        );
    }
    codec.delegate().state.events.borrow_mut().clear();
    codec.delegate().state.fail_display.set(true);
    let status = outcome(|| {
        let _ = format!("{codec}");
        Ok(())
    });
    row(
        out,
        &[
            "display-failure".into(),
            status,
            random.borrow_mut().next_int().to_string(),
            events(&codec),
        ],
    );
}
fn info() -> SegmentInfo {
    SegmentInfo {
        id: ID,
        version: LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
        min_version: None,
        doc_count: 1,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: vec![],
        attributes: vec![],
        index_sort: None,
    }
}
fn interleaved(out: &mut String) {
    for seed in [0, 1, 18, 42, -1, i64::MIN, i64::MAX] {
        let random = Rc::new(RefCell::new(JavaRandom::new(seed)));
        let codec = CrankyCodec::new(RecordingCodec::new(), random.clone());
        let mut held = [
            codec.field_infos_format().unwrap(),
            codec.field_infos_format().unwrap(),
        ];
        assert!(Rc::ptr_eq(
            &held[0].delegate().identity,
            &held[1].delegate().identity
        ));
        let directory = ByteBuffersDirectory::new();
        let fields = FieldInfos::new(vec![]).unwrap();
        let bits = FixedBitSet::new(1);
        let live = LiveDocsInfo {
            name: "_p",
            segment_id: &ID,
            max_doc: 1,
            del_gen: -1,
            next_del_gen: 1,
            del_count: 0,
        };
        let compound = CompoundInfo {
            name: "_p",
            id: &ID,
            files: &[],
        };
        let mut segment = info();
        for step in 0..384 {
            codec.delegate().state.events.borrow_mut().clear();
            let fail = step % 17 == 0;
            codec.delegate().state.fail_operation.set(fail);
            let operation = step % OPS.len();
            let side = (step / OPS.len()) & 1;
            // Factory errors and component I/O are intentionally distinct.
            let status = match operation {
                0 => held[side]
                    .write(&directory, "_p", &ID, "", &fields)
                    .map_err(|e| format!("io:{e}")),
                1 => codec
                    .live_docs_format()
                    .unwrap()
                    .write_live_docs(&bits, &directory, &live, 0)
                    .map_err(|e| format!("io:{e}")),
                2 => codec
                    .segment_info_format()
                    .unwrap()
                    .write(&directory, "_p", &mut segment)
                    .map_err(|e| format!("io:{e}")),
                3 => codec
                    .compound_format()
                    .unwrap()
                    .write(&directory, &compound)
                    .map_err(|e| format!("io:{e}")),
                4 => codec
                    .field_infos_format()
                    .unwrap()
                    .read(&directory, "_p", &ID, "")
                    .map(|f| assert!(f.fields.is_empty()))
                    .map_err(|e| format!("io:{e}")),
                5 => codec
                    .live_docs_format()
                    .unwrap()
                    .read_live_docs(&directory, &live)
                    .map(|b| assert_eq!(b.len(), 1))
                    .map_err(|e| format!("io:{e}")),
                6 => {
                    let mut files = vec!["existing".into()];
                    codec
                        .live_docs_format()
                        .unwrap()
                        .files(&live, &mut files)
                        .map(|()| assert_eq!(files, ["existing", "delegate-file"]))
                        .map_err(|e| format!("io:{e}"))
                }
                7 => {
                    for kind in 0..KINDS.len() {
                        component(&codec, kind).unwrap();
                    }
                    Ok(())
                }
                8 => {
                    assert_eq!(format!("{codec}"), "Cranky(DelegateDisplay)");
                    Ok(())
                }
                9 => {
                    codec.delegate().state.fail_factory.set(Some(1));
                    codec
                        .field_infos_format()
                        .map(|_| ())
                        .map_err(|e| format!("runtime:{e}"))
                }
                10 => {
                    codec
                        .delegate()
                        .state
                        .events
                        .borrow_mut()
                        .push(format!("external:{}", random.borrow_mut().next_boolean()));
                    Ok(())
                }
                _ => unreachable!(),
            }
            .err()
            .unwrap_or_else(|| "ok".into());
            codec.delegate().state.fail_factory.set(None);
            row(
                out,
                &[
                    "mixed".into(),
                    seed.to_string(),
                    step.to_string(),
                    OPS[operation].into(),
                    fail.to_string(),
                    side.to_string(),
                    status,
                    random.borrow_mut().next_int().to_string(),
                    events(&codec),
                ],
            );
        }
    }
}
#[test]
fn actual_java_component_factories_identity_and_interleaved_rng_match() {
    let expected = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/testframework/data/cranky_codec.tsv"),
    )
    .unwrap();
    let mut actual = String::new();
    configuration(&mut actual);
    interleaved(&mut actual);
    assert_eq!(actual.lines().count(), expected.lines().count());
    for (index, (actual, expected)) in actual.lines().zip(expected.lines()).enumerate() {
        assert_eq!(actual, expected, "actual Java fixture row {}", index + 1);
    }
}
