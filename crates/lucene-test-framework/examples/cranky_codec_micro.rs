//! Codec composition costs with inert probes, paired with CrankyCodecMicro.java.
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
    fmt,
    hint::black_box,
    io,
    rc::Rc,
    time::{Duration, Instant},
};
const ID: [u8; 16] = [0; 16];
#[derive(Default)]
struct State {
    factories: Cell<u64>,
    operations: Cell<u64>,
    displays: Cell<u64>,
    alternate: Cell<bool>,
    trace: Cell<u64>,
}
#[derive(Clone)]
struct Handle {
    identity: Rc<()>,
    state: Rc<State>,
}
impl Handle {
    fn call(&self, _: &str) -> io::Result<()> {
        self.state.operations.set(self.state.operations.get() + 1);
        Ok(())
    }
}
struct Probe {
    state: Rc<State>,
    handle: Handle,
}
impl Probe {
    fn new() -> Self {
        let state = Rc::new(State::default());
        Self {
            handle: Handle {
                identity: Rc::new(()),
                state: state.clone(),
            },
            state,
        }
    }
    fn component(&self, kind: u64) -> Result<Handle, ()> {
        self.state.trace.set(hash(self.state.trace.get(), kind));
        self.state.factories.set(self.state.factories.get() + 1);
        Ok(self.handle.clone())
    }
}
impl fmt::Display for Probe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.state.displays.set(self.state.displays.get() + 1);
        f.write_str(if self.state.alternate.get() {
            "DelegateNext"
        } else {
            "DelegateFirst"
        })
    }
}
impl Codec for Probe {
    type Error = ();
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

fn hash(digest: u64, value: u64) -> u64 {
    (digest ^ value).wrapping_mul(0x100_0000_01b3)
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
struct Inputs {
    dir: ByteBuffersDirectory,
    fields: FieldInfos,
    bits: FixedBitSet,
    live: LiveDocsInfo<'static>,
    compound: CompoundInfo<'static>,
    segment: SegmentInfo,
}
impl Inputs {
    fn new() -> Self {
        Self {
            dir: ByteBuffersDirectory::new(),
            fields: FieldInfos::new(vec![]).unwrap(),
            bits: FixedBitSet::new(1),
            live: LiveDocsInfo {
                name: "_p",
                segment_id: &ID,
                max_doc: 1,
                del_gen: -1,
                next_del_gen: 1,
                del_count: 0,
            },
            compound: CompoundInfo {
                name: "_p",
                id: &ID,
                files: &[],
            },
            segment: info(),
        }
    }
}
fn batch(
    case: usize,
    inputs: &mut Inputs,
    codec: &CrankyCodec<Probe>,
    random: &Rc<RefCell<JavaRandom>>,
) -> u64 {
    *random.borrow_mut() = JavaRandom::new(42);
    let state = &codec.delegate().state;
    state.factories.set(0);
    state.trace.set(0xcbf2_9ce4_8422_2325);
    state.operations.set(0);
    state.displays.set(0);
    let mut digest = 0xcbf2_9ce4_8422_2325;
    match case {
        0 => {
            macro_rules! pair {
                ($method:ident) => {{
                    let first = black_box(codec.$method().unwrap());
                    let second = black_box(codec.$method().unwrap());
                    assert!(!std::ptr::eq(black_box(&first), black_box(&second)));
                    digest = hash(digest, 1);
                    black_box((first, second));
                }};
            }
            for _ in 0..256 {
                pair!(doc_values_format);
                pair!(field_infos_format);
                pair!(live_docs_format);
                pair!(norms_format);
                pair!(postings_format);
                pair!(segment_info_format);
                pair!(stored_fields_format);
                pair!(term_vectors_format);
                pair!(compound_format);
                pair!(points_format);
            }
            assert_eq!(state.factories.get(), 5120);
            assert_eq!(state.operations.get(), 0);
        }
        1 => {
            for _ in 0..256 {
                let first = black_box(codec.knn_vectors_format().unwrap());
                let second = black_box(codec.knn_vectors_format().unwrap());
                assert!(Rc::ptr_eq(
                    &first.identity,
                    &codec.delegate().handle.identity
                ));
                assert!(Rc::ptr_eq(&first.identity, &second.identity));
                digest = hash(digest, 1);
                black_box((first, second));
            }
            assert_eq!(state.factories.get(), 512);
        }
        2 => {
            let mut held = [
                codec.field_infos_format().unwrap(),
                codec.field_infos_format().unwrap(),
            ];
            let live_format = codec.live_docs_format().unwrap();
            let mut segment_format = codec.segment_info_format().unwrap();
            let mut compound_format = codec.compound_format().unwrap();
            assert!(Rc::ptr_eq(
                &held[0].delegate().identity,
                &held[1].delegate().identity
            ));

            black_box((&held, &live_format, &segment_format, &compound_format));
            for step in 0..256 {
                let error = match step % 5 {
                    0 | 1 => held[step % 2]
                        .write(&inputs.dir, "_p", &ID, "", &inputs.fields)
                        .err()
                        .map(|e| e.to_string()),
                    2 => live_format
                        .write_live_docs(&inputs.bits, &inputs.dir, &inputs.live, 0)
                        .err()
                        .map(|e| e.to_string()),
                    3 => segment_format
                        .write(&inputs.dir, "_p", &mut inputs.segment)
                        .err()
                        .map(|e| e.to_string()),
                    _ => compound_format
                        .write(&inputs.dir, &inputs.compound)
                        .err()
                        .map(|e| e.to_string()),
                };
                if let Some(message) = error {
                    digest = hash(digest, 0);
                    for byte in message.bytes() {
                        digest = hash(digest, u64::from(byte));
                    }
                } else {
                    digest = hash(digest, 1);
                }
            }
            assert_eq!(state.factories.get(), 5);
        }
        3 => {
            for step in 0..256 {
                state.alternate.set(step % 2 == 1);
                let value = black_box(format!("{codec}"));
                for byte in value.bytes() {
                    digest = hash(digest, u64::from(byte));
                }
            }
            assert_eq!(state.factories.get(), 0);
            assert_eq!(state.displays.get(), 256);
        }
        _ => unreachable!(),
    }
    digest = hash(digest, state.factories.get());
    digest = hash(digest, state.operations.get());
    digest = hash(digest, state.displays.get());
    digest = hash(digest, state.trace.get());
    digest = hash(digest, random.borrow_mut().next_int() as u64);
    black_box(digest)
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(500);
    let measure = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let random = Rc::new(RefCell::new(JavaRandom::new(42)));
    let codec = CrankyCodec::new(Probe::new(), random.clone());
    let mut inputs = Inputs::new();
    println!(
        "cranky_codec_shallow_adapter_bytes\t{}",
        std::mem::size_of_val(&codec) - std::mem::size_of::<Probe>()
    );
    println!("cranky_codec_name_owned_bytes\t{}", codec.name_heap_bytes());
    println!(
        "cranky_codec_shared_rng_bytes\t{}",
        2 * std::mem::size_of::<usize>() + std::mem::size_of::<RefCell<JavaRandom>>()
    );
    for (case, name) in [
        "cranky_codec_components",
        "cranky_codec_knn",
        "cranky_codec_shared_guards",
        "cranky_codec_display",
    ]
    .into_iter()
    .enumerate()
    {
        let expected = batch(case, &mut inputs, &codec, &random);
        for (phase, budget) in [warmup, measure].into_iter().enumerate() {
            let start = Instant::now();
            let mut operations = 0u64;
            loop {
                assert_eq!(batch(case, &mut inputs, &codec, &random), expected);
                operations += 256;
                if start.elapsed() >= Duration::from_millis(budget) {
                    break;
                }
            }
            if phase == 1 {
                println!(
                    "{name}\t{:.3}\t{operations}",
                    start.elapsed().as_nanos() as f64 / operations as f64
                );
            }
        }
        println!("#check\t{name}\t{expected}\t256");
    }
}
