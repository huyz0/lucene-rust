//! Partial codec configuration/composition pair; no constructor or I/O timing.
use lucene_test_framework::codecs::compressing_codec::{
    CodecDelegate, CodecKind, ComponentKind, CompressingCodec, CompressionMode, ModeHandle,
};
use std::{
    alloc::Layout,
    cell::{Cell, RefCell},
    hint::black_box,
    rc::Rc,
    time::{Duration, Instant},
};

const BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const LABELS: [&str; 6] = [
    "fast",
    "fast_decompression",
    "high_compression",
    "dummy",
    "deflate_preset",
    "lz4_preset",
];
const COMPONENTS: [(ComponentKind, &str); 9] = [
    (ComponentKind::DocValues, "docValuesFormat"),
    (ComponentKind::FieldInfos, "fieldInfosFormat"),
    (ComponentKind::LiveDocs, "liveDocsFormat"),
    (ComponentKind::Norms, "normsFormat"),
    (ComponentKind::Postings, "postingsFormat"),
    (ComponentKind::SegmentInfo, "segmentInfoFormat"),
    (ComponentKind::Compound, "compoundFormat"),
    (ComponentKind::Points, "pointsFormat"),
    (ComponentKind::KnnVectors, "knnVectorsFormat"),
];
fn hash(digest: u64, value: u64) -> u64 {
    (digest ^ value).wrapping_mul(0x100_0000_01b3)
}
fn text(mut digest: u64, value: &str) -> u64 {
    digest = hash(digest, value.len() as u64);
    for byte in value.bytes() {
        digest = hash(digest, u64::from(byte));
    }
    digest
}
fn configurations() -> Vec<CompressingCodec> {
    let mut result = vec![];
    for kind in CodecKind::ALL {
        result.push(kind.defaults());
        result.push(kind.configured(37, 3, false, 2).unwrap());
        result.push(kind.configured(37, 3, true, 22).unwrap());
    }
    let fast = || Some(ModeHandle::new(CompressionMode::Fast));
    result.push(CompressingCodec::new("CustomCompressingData", fast(), 11, 7, 2).unwrap());
    result.push(
        CompressingCodec::with_suffix("CustomCompressingData", "privateS", fast(), 11, 7, 22)
            .unwrap(),
    );
    result.push(
        CompressingCodec::with_suffix(
            "CustomCompressingData",
            "privateS",
            fast(),
            i32::MAX,
            i32::MAX,
            22,
        )
        .unwrap(),
    );
    result.push(CompressingCodec::new("", fast(), 1, 1, 2).unwrap());
    result.push(CompressingCodec::new("CustomCompressingData", None, 1, 1, 2).unwrap());
    result
}
fn configuration_check(codecs: &[CompressingCodec]) -> u64 {
    let mut digest = BASIS;
    for codec in codecs {
        digest = text(
            digest,
            codec.kind().map_or("BaseCodec", CodecKind::class_name),
        );
        digest = text(digest, codec.name());
        digest = text(digest, &codec.to_string());
        let sf = codec.stored_fields_format();
        let tv = codec.term_vectors_format();
        assert!(std::ptr::eq(sf, codec.stored_fields_format()));
        assert!(std::ptr::eq(tv, codec.term_vectors_format()));
        let same_mode = match (&sf.0.compression_mode, &tv.0.compression_mode) {
            (Some(a), Some(b)) => a.same_instance(b),
            (None, None) => true,
            _ => false,
        };
        assert!(same_mode);
        digest = hash(digest, 1);
        for settings in [&sf.0, &tv.0] {
            digest = text(digest, settings.format_name());
            digest = text(digest, settings.segment_suffix());
            digest = text(digest, &settings.mode_display());
            digest = hash(digest, settings.chunk_size as u64);
            digest = hash(digest, settings.max_docs_per_chunk as u64);
            digest = hash(digest, settings.block_shift as u64);
        }
    }
    for kind in CodecKind::ALL {
        let first = kind.defaults();
        let second = kind.defaults();
        let shared = first
            .stored_fields_format()
            .0
            .compression_mode
            .as_ref()
            .unwrap()
            .same_instance(
                second
                    .stored_fields_format()
                    .0
                    .compression_mode
                    .as_ref()
                    .unwrap(),
            );
        assert_eq!(
            shared,
            !matches!(
                kind,
                CodecKind::DeflateWithPreset | CodecKind::Lz4WithPreset
            )
        );
        digest = hash(digest, u64::from(shared));
    }
    black_box(digest)
}
#[derive(Default)]
struct State {
    calls: Cell<u64>,
    trace: Cell<u64>,
    fail: Cell<Option<usize>>,
}
struct Probe {
    state: Rc<State>,
    handles: [Rc<()>; 9],
    errors: [Rc<String>; 9],
    last: RefCell<Option<Rc<()>>>,
}
impl Probe {
    fn new(state: Rc<State>) -> Self {
        Self {
            state,
            handles: std::array::from_fn(|_| Rc::new(())),
            errors: std::array::from_fn(|i| {
                Rc::new(format!("delegate failure:{}", COMPONENTS[i].1))
            }),
            last: RefCell::new(None),
        }
    }
}
impl CodecDelegate for Probe {
    type Component = Rc<()>;
    type Error = Rc<String>;
    fn component(&self, kind: ComponentKind) -> Result<Self::Component, Self::Error> {
        let index = COMPONENTS
            .iter()
            .position(|(candidate, _)| *candidate == kind)
            .unwrap();
        self.state.calls.set(self.state.calls.get() + 1);
        self.state
            .trace
            .set(hash(self.state.trace.get(), index as u64));
        if self.state.fail.get() == Some(index) {
            return Err(self.errors[index].clone());
        }
        let handle = if kind == ComponentKind::Points {
            Rc::new(())
        } else {
            self.handles[index].clone()
        };
        *self.last.borrow_mut() = Some(handle.clone());
        Ok(handle)
    }
}
fn batch(
    case: usize,
    codecs: &[CompressingCodec],
    delegates: &[CompressingCodec<Probe>],
    state: &State,
) -> u64 {
    state.calls.set(0);
    state.trace.set(BASIS);
    state.fail.set(None);
    let mut digest = BASIS;
    for step in 0..256 {
        let codec = &codecs[step % codecs.len()];
        match case {
            0 => {
                let sf = black_box(codec.stored_fields_format());
                let tv = black_box(codec.term_vectors_format());
                assert!(std::ptr::eq(sf, black_box(codec.stored_fields_format())));
                assert!(std::ptr::eq(tv, black_box(codec.term_vectors_format())));
                digest = hash(digest, 1);
                digest = text(digest, black_box(codec.name()));
            }
            1 => {
                let value = black_box(codec.stored_fields_format().to_string());
                digest = text(digest, &value);
            }
            2 => {
                let value = black_box(codec.term_vectors_format().to_string());
                digest = text(digest, &value);
            }
            3 => {
                let value = black_box(codec.to_string());
                digest = text(digest, &value);
            }
            4 => {
                let codec = &delegates[step % delegates.len()];
                for (kind, _) in COMPONENTS {
                    let first = black_box(codec.component(kind).unwrap());
                    assert!(Rc::ptr_eq(
                        &first,
                        codec.delegate().last.borrow().as_ref().unwrap()
                    ));
                    let second = black_box(codec.component(kind).unwrap());
                    assert!(Rc::ptr_eq(
                        &second,
                        codec.delegate().last.borrow().as_ref().unwrap()
                    ));
                    let same = Rc::ptr_eq(&first, &second);
                    assert_eq!(same, kind != ComponentKind::Points);
                    digest = hash(digest, u64::from(same));
                    black_box((first, second));
                }
            }
            5 => {
                let codec = &delegates[step % delegates.len()];
                let index = step % COMPONENTS.len();
                state.fail.set(Some(index));
                let error = black_box(codec.component(COMPONENTS[index].0).unwrap_err());
                assert!(Rc::ptr_eq(&error, &codec.delegate().errors[index]));
                digest = hash(digest, 1);
                digest = text(digest, &error);
                black_box(error);
            }
            _ => unreachable!(),
        }
    }
    assert_eq!(
        state.calls.get(),
        match case {
            4 => 256 * 18,
            5 => 256,
            _ => 0,
        }
    );
    digest = hash(digest, state.calls.get());
    digest = hash(digest, state.trace.get());
    black_box(digest)
}
fn memory_rows() {
    for (kind, label) in CodecKind::ALL.into_iter().zip(LABELS) {
        let codec = kind.defaults();
        let shallow = std::mem::size_of_val(&codec)
            + std::mem::size_of_val(codec.stored_fields_format())
            + std::mem::size_of_val(codec.term_vectors_format())
            + codec.shared_settings_bytes();
        let strings = codec.owned_string_bytes();
        let mode = if matches!(
            kind,
            CodecKind::DeflateWithPreset | CodecKind::Lz4WithPreset
        ) {
            Layout::new::<[usize; 2]>()
                .extend(Layout::new::<CompressionMode>())
                .unwrap()
                .0
                .pad_to_align()
                .size()
        } else {
            0
        };
        println!("compressing_{label}_selected_shallow_bytes\t{shallow}");
        println!("compressing_{label}_owned_string_bytes\t{strings}");
        println!("compressing_{label}_owned_mode_bytes\t{mode}");
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup: u64 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let measure: u64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1500);
    let codecs = configurations();
    let config = configuration_check(&codecs);
    let state = Rc::new(State::default());
    let delegates: Vec<_> = CodecKind::ALL
        .into_iter()
        .map(|kind| kind.defaults().map_delegate(|_| Probe::new(state.clone())))
        .collect();
    memory_rows();
    println!("#setup\tcompressing_configuration\t{config}\t23");
    for (case, name) in [
        "compressing_cached",
        "compressing_stored_display",
        "compressing_vectors_display",
        "compressing_codec_display",
        "compressing_delegation",
        "compressing_delegate_errors",
    ]
    .into_iter()
    .enumerate()
    {
        let expected = batch(case, &codecs, &delegates, &state);
        for (phase, budget) in [warmup, measure].into_iter().enumerate() {
            let start = Instant::now();
            let mut operations = 0u64;
            loop {
                assert_eq!(batch(case, &codecs, &delegates, &state), expected);
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
        println!("#check\t{name}\t{expected}\t256\t{config}");
    }
}
