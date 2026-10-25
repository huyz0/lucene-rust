//! Replay actual Java constructor/settings, factory, error and RNG observations.
//! Native compression or format I/O is intentionally outside this partial port.
use lucene_codecs::codec::Lucene104Codec;
use lucene_test_framework::codecs::compressing_codec::{
    CodecDelegate, CodecKind, ComponentKind, CompressingCodec, CompressionMode,
    InvalidConfiguration, ModeHandle, RandomSource,
};
use lucene_util::java_random::JavaRandom;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

const LABELS: [&str; 6] = [
    "fast",
    "fast_decompression",
    "high_compression",
    "dummy",
    "deflate_preset",
    "lz4_preset",
];
const ACCESSORS: [(ComponentKind, &str); 9] = [
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
const SEEDS: [i64; 7] = [0, 1, 18, 42, -1, i64::MIN, i64::MAX];
const INVALID: [[i32; 3]; 8] = [
    [0, 0, 0],
    [-1, 1, 2],
    [1, 0, 0],
    [1, -1, 2],
    [1, 1, 0],
    [1, 1, 1],
    [1, 1, 23],
    [1, 1, i32::MAX],
];
fn row(out: &mut String, values: &[String]) {
    out.push_str(&values.join("\t"));
    out.push('\n');
}
fn matches_fixture(name: &str, actual: String) {
    let expected = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
        format!("../../fixtures/testframework/data/compressing_codec_{name}.tsv"),
    ))
    .unwrap();
    assert_eq!(
        actual.lines().count(),
        expected.lines().count(),
        "{name} row count"
    );
    for (i, (a, b)) in actual.lines().zip(expected.lines()).enumerate() {
        assert_eq!(a, b, "{name} Java fixture row {}", i + 1);
    }
}
fn modes_same<D, E>(a: &CompressingCodec<D>, b: &CompressingCodec<E>) -> bool {
    let a = a.stored_fields_format().0.compression_mode.as_ref();
    let b = b.stored_fields_format().0.compression_mode.as_ref();
    match (a, b) {
        (Some(a), Some(b)) => a.same_instance(b),
        (None, None) => true,
        _ => false,
    }
}
fn snapshot(out: &mut String, label: &str, codec: &CompressingCodec) {
    let sf = codec.stored_fields_format();
    let tv = codec.term_vectors_format();
    let mode_shared = match (&sf.0.compression_mode, &tv.0.compression_mode) {
        (Some(a), Some(b)) => a.same_instance(b),
        (None, None) => true,
        _ => false,
    };
    row(
        out,
        &[
            "codec".into(),
            label.into(),
            codec
                .kind()
                .map_or("BaseCodec", CodecKind::class_name)
                .into(),
            codec.name().into(),
            codec.delegate().name().into(),
            codec.to_string(),
        ],
    );
    row(
        out,
        &[
            "identity".into(),
            label.into(),
            std::ptr::eq(sf, codec.stored_fields_format()).to_string(),
            std::ptr::eq(tv, codec.term_vectors_format()).to_string(),
            mode_shared.to_string(),
        ],
    );
    for (tag, class, s, display) in [
        (
            "stored",
            "Lucene90CompressingStoredFieldsFormat",
            &sf.0,
            sf.to_string(),
        ),
        (
            "vectors",
            "Lucene90CompressingTermVectorsFormat",
            &tv.0,
            tv.to_string(),
        ),
    ] {
        row(
            out,
            &[
                tag.into(),
                label.into(),
                class.into(),
                s.format_name().into(),
                s.segment_suffix().into(),
                s.mode_display(),
                s.chunk_size.to_string(),
                s.max_docs_per_chunk.to_string(),
                s.block_shift.to_string(),
                display,
            ],
        );
    }
}
#[test]
fn all_subclass_and_base_configuration_observations_match_java() {
    let mut out = String::new();
    for (kind, label) in CodecKind::ALL.into_iter().zip(LABELS) {
        let first = kind.defaults();
        let second = kind.defaults();
        snapshot(&mut out, &format!("{label}_default"), &first);
        snapshot(
            &mut out,
            &format!("{label}_custom"),
            &kind.configured(37, 3, false, 2).unwrap(),
        );
        snapshot(
            &mut out,
            &format!("{label}_suffix"),
            &kind.configured(37, 3, true, 22).unwrap(),
        );
        row(
            &mut out,
            &[
                "between-codecs".into(),
                label.into(),
                std::ptr::eq(first.stored_fields_format(), second.stored_fields_format())
                    .to_string(),
                std::ptr::eq(first.term_vectors_format(), second.term_vectors_format()).to_string(),
                modes_same(&first, &second).to_string(),
            ],
        );
    }
    let fast = || Some(ModeHandle::new(CompressionMode::Fast));
    snapshot(
        &mut out,
        "base_empty_suffix",
        &CompressingCodec::new("CustomCompressingData", fast(), 11, 7, 2).unwrap(),
    );
    snapshot(
        &mut out,
        "base_suffix",
        &CompressingCodec::with_suffix("CustomCompressingData", "privateS", fast(), 11, 7, 22)
            .unwrap(),
    );
    snapshot(
        &mut out,
        "base_maximum",
        &CompressingCodec::with_suffix(
            "CustomCompressingData",
            "privateS",
            fast(),
            i32::MAX,
            i32::MAX,
            22,
        )
        .unwrap(),
    );
    snapshot(
        &mut out,
        "base_empty_name",
        &CompressingCodec::new("", fast(), 1, 1, 2).unwrap(),
    );
    snapshot(
        &mut out,
        "base_null_mode",
        &CompressingCodec::new("CustomCompressingData", None, 1, 1, 2).unwrap(),
    );
    matches_fixture("config", out);
}

/// Explicit callback probes model the real Java fixture's stable components and
/// fresh points factory. They do not install native Lucene104 component I/O.
struct Probe {
    handles: [Rc<()>; 9],
    events: RefCell<Vec<&'static str>>,
    fail: Cell<Option<ComponentKind>>,
    failure: RefCell<Rc<String>>,
    last: RefCell<Option<Rc<()>>>,
}
impl Probe {
    fn new() -> Self {
        Self {
            handles: std::array::from_fn(|_| Rc::new(())),
            events: RefCell::new(vec![]),
            fail: Cell::new(None),
            failure: RefCell::new(Rc::new(String::new())),
            last: RefCell::new(None),
        }
    }
}
impl CodecDelegate for Probe {
    type Component = Rc<()>;
    type Error = Rc<String>;
    fn component(&self, kind: ComponentKind) -> Result<Rc<()>, Rc<String>> {
        let (index, (_, name)) = ACCESSORS
            .iter()
            .enumerate()
            .find(|(_, (candidate, _))| *candidate == kind)
            .unwrap();
        self.events.borrow_mut().push(name);
        if self.fail.get() == Some(kind) {
            return Err(self.failure.borrow().clone());
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
#[test]
fn inherited_callbacks_preserve_objects_errors_and_selected_overrides() {
    let mut out = String::new();
    for (kind, label) in CodecKind::ALL.into_iter().zip(LABELS) {
        let original = kind.defaults();
        let sf = original.stored_fields_format() as *const _;
        let tv = original.term_vectors_format() as *const _;
        let codec = original.map_delegate(|_| Probe::new());
        assert_eq!(sf, codec.stored_fields_format() as *const _);
        assert_eq!(tv, codec.term_vectors_format() as *const _);
        for (component, name) in ACCESSORS {
            let actual = codec.component(component).unwrap();
            let direct = codec.delegate().component(component).unwrap();
            row(
                &mut out,
                &[
                    "delegate-identity".into(),
                    label.into(),
                    name.into(),
                    Rc::ptr_eq(&actual, &direct).to_string(),
                ],
            );
        }
        for (component, name) in ACCESSORS {
            let probe = codec.delegate();
            probe.events.borrow_mut().clear();
            let first = codec.component(component).unwrap();
            let first_exact = Rc::ptr_eq(&first, probe.last.borrow().as_ref().unwrap());
            let second = codec.component(component).unwrap();
            let exact = first_exact && Rc::ptr_eq(&second, probe.last.borrow().as_ref().unwrap());
            row(
                &mut out,
                &[
                    "forward".into(),
                    label.into(),
                    name.into(),
                    exact.to_string(),
                    Rc::ptr_eq(&first, &second).to_string(),
                    probe.events.borrow().join(","),
                ],
            );
            probe.events.borrow_mut().clear();
            probe.fail.set(Some(component));
            *probe.failure.borrow_mut() = Rc::new(format!("delegate failure:{name}"));
            let error = codec.component(component).unwrap_err();
            row(
                &mut out,
                &[
                    "forward-error".into(),
                    label.into(),
                    name.into(),
                    "IllegalStateException".into(),
                    (*error).clone(),
                    Rc::ptr_eq(&error, &probe.failure.borrow()).to_string(),
                    probe.events.borrow().join(","),
                ],
            );
            probe.fail.set(None);
        }
        codec.delegate().events.borrow_mut().clear();
        assert_eq!(sf, codec.stored_fields_format() as *const _);
        assert_eq!(tv, codec.term_vectors_format() as *const _);
        assert!(codec
            .to_string()
            .starts_with(&format!("{}(storedFieldsFormat=", codec.name())));
        row(
            &mut out,
            &[
                "override-events".into(),
                label.into(),
                codec.delegate().events.borrow().join(","),
            ],
        );
    }
    matches_fixture("delegation", out);
}

struct TraceRandom {
    random: JavaRandom,
    events: Vec<String>,
}
impl TraceRandom {
    fn new(seed: i64) -> Self {
        Self {
            random: JavaRandom::new(seed),
            events: vec![],
        }
    }
}
impl RandomSource for TraceRandom {
    fn next_boolean(&mut self) -> bool {
        let value = self.random.next_boolean();
        self.events.push(format!("boolean={value}"));
        value
    }
    fn next_int_bounded(&mut self, bound: i32) -> i32 {
        let value = self.random.next_int_bounded(bound);
        self.events.push(format!("int({bound})={value}"));
        value
    }
}
fn result(result: Result<CompressingCodec, InvalidConfiguration>) -> String {
    match result {
        Ok(codec) => {
            let s = &codec.stored_fields_format().0;
            format!(
                "ok:{};{};{};{};{};{};{}",
                codec.kind().unwrap().class_name(),
                codec.name(),
                s.segment_suffix(),
                s.mode_display(),
                s.chunk_size,
                s.max_docs_per_chunk,
                s.block_shift
            )
        }
        Err(error) => format!("IllegalArgumentException:{error}"),
    }
}
#[test]
fn all_random_overloads_match_actual_java_draws_results_and_tail() {
    let mut out = String::new();
    let overloads = [
        "explicit",
        "random",
        "reasonable",
        "suffix_false",
        "suffix_true",
    ];
    for seed in SEEDS {
        let mut random = TraceRandom::new(seed);
        for step in 0..384 {
            random.events.clear();
            let overload = overloads[step % 5];
            let value = match step % 5 {
                0 => CompressingCodec::random_explicit(&mut random, 37, 3, step & 1 != 0, 10),
                1 => CompressingCodec::random_instance(&mut random),
                2 => CompressingCodec::reasonable_instance(&mut random),
                3 => CompressingCodec::random_with_suffix(&mut random, false),
                _ => CompressingCodec::random_with_suffix(&mut random, true),
            };
            row(
                &mut out,
                &[
                    "random".into(),
                    seed.to_string(),
                    step.to_string(),
                    overload.into(),
                    result(value),
                    random.events.join(","),
                    random.random.next_int().to_string(),
                ],
            );
        }
    }
    matches_fixture("random", out);
}
#[test]
fn invalid_constructors_preserve_name_geometry_precedence_and_selector_draw() {
    let mut out = String::new();
    for (kind, label) in CodecKind::ALL.into_iter().zip(LABELS) {
        for [chunk, docs, shift] in INVALID {
            let error = kind.configured(chunk, docs, true, shift).unwrap_err();
            row(
                &mut out,
                &[
                    "invalid".into(),
                    format!("{label}:{chunk}:{docs}:{shift}"),
                    "IllegalArgumentException".into(),
                    error.to_string(),
                ],
            );
        }
    }
    for name in ["bad-name".into(), "é".into(), "a".repeat(128)] {
        let error =
            CompressingCodec::new(&name, Some(ModeHandle::new(CompressionMode::Fast)), 0, 0, 0)
                .unwrap_err();
        row(
            &mut out,
            &[
                "invalid".into(),
                format!("base-name:{name}"),
                "IllegalArgumentException".into(),
                error.to_string(),
            ],
        );
    }
    for seed in SEEDS {
        for [chunk, docs, shift] in INVALID {
            let mut random = TraceRandom::new(seed);
            let error = CompressingCodec::random_explicit(&mut random, chunk, docs, true, shift)
                .unwrap_err();
            row(
                &mut out,
                &[
                    "random-invalid".into(),
                    seed.to_string(),
                    chunk.to_string(),
                    docs.to_string(),
                    shift.to_string(),
                    "IllegalArgumentException".into(),
                    error.to_string(),
                    random.events.join(","),
                    random.random.next_int().to_string(),
                ],
            );
        }
    }
    matches_fixture("errors", out);
}
