//! Actual asserting adapters over explicit stable handles; no native graph construction or I/O.
use lucene_test_framework::codecs::asserting_codec::*;
use lucene_test_framework::codecs::{
    asserting_doc_values::AssertingDocValuesFormat,
    asserting_knn_vectors::AssertingKnnVectorsFormat, asserting_postings::AssertingPostingsFormat,
    cranky_live_docs::Lucene90LiveDocsFormat,
};
use std::{
    alloc::Layout,
    any::Any,
    cell::Cell,
    hint::black_box,
    rc::Rc,
    time::{Duration, Instant},
};
const BASIS: u64 = 0xcbf29ce484222325;
const MASKS: [u8; 3] = [0, 21, 63];
const BITS: [i8; 11] = [-1, -1, 3, 2, -1, -1, 0, 1, -1, 4, 5];
const FIELDS: [Option<&str>; 4] = [Some(""), Some("field"), Some("alternate"), None];
fn hash(d: u64, v: u64) -> u64 {
    (d ^ v).wrapping_mul(0x100000001b3)
}
fn token(field: Option<&str>) -> u64 {
    match field {
        Some("") => 0,
        Some("field") => 1,
        Some("alternate") => 2,
        None => 3,
        _ => panic!("unexpected field"),
    }
}
#[derive(Debug)]
struct Token;
type Error = Rc<()>;
type Handle = Rc<dyn Any>;
struct Probe {
    counts: [Cell<u64>; 11],
    trace: Cell<u64>,
    displays: Cell<u64>,
    native_selects: Cell<u64>,
    native_trace: Cell<u64>,
    fail: Cell<bool>,
    next: Cell<bool>,
    error: Error,
    formats: [Rc<Token>; 11],
    selected: Rc<Token>,
}
impl Probe {
    fn new() -> Self {
        Self {
            counts: std::array::from_fn(|_| Cell::new(0)),
            trace: Cell::new(BASIS),
            displays: Cell::new(0),
            native_selects: Cell::new(0),
            native_trace: Cell::new(BASIS),
            fail: Cell::new(false),
            next: Cell::new(false),
            error: Rc::new(()),
            formats: std::array::from_fn(|_| Rc::new(Token)),
            selected: Rc::new(Token),
        }
    }
    fn reset(&self) {
        for c in &self.counts {
            c.set(0)
        }
        self.trace.set(BASIS);
        self.displays.set(0);
        self.native_selects.set(0);
        self.native_trace.set(BASIS);
        self.fail.set(false)
    }
    fn format(&self, k: usize) -> Result<Rc<Token>, Error> {
        self.counts[k].set(self.counts[k].get() + 1);
        self.trace.set(hash(self.trace.get(), k as u64));
        if self.fail.get() {
            Err(self.error.clone())
        } else {
            Ok(self.formats[k].clone())
        }
    }
}
struct SharedProbe(Rc<Probe>);
macro_rules! factory {
    ($name:ident,$kind:expr) => {
        fn $name(&self) -> Result<Rc<Token>, Error> {
            self.0.format($kind)
        }
    };
}
impl Delegate for SharedProbe {
    type Error = Error;
    type DocValues = Rc<Token>;
    type FieldInfos = Rc<Token>;
    type LiveDocs = Rc<Token>;
    type Norms = Rc<Token>;
    type Postings = Rc<Token>;
    type SegmentInfo = Rc<Token>;
    type StoredFields = Rc<Token>;
    type TermVectors = Rc<Token>;
    type Compound = Rc<Token>;
    type Points = Rc<Token>;
    type KnnVectors = Rc<Token>;
    type SelectedKnnVectors = Rc<Token>;
    fn name(&self) -> &str {
        "Probe"
    }
    fn display(&self) -> Result<String, Error> {
        self.0.displays.set(self.0.displays.get() + 1);
        Ok(if self.0.next.get() {
            "ProbeNext"
        } else {
            "ProbeFirst"
        }
        .into())
    }
    factory!(doc_values_format, 0);
    factory!(field_infos_format, 1);
    factory!(live_docs_format, 2);
    factory!(norms_format, 3);
    factory!(postings_format, 4);
    factory!(segment_info_format, 5);
    factory!(stored_fields_format, 6);
    factory!(term_vectors_format, 7);
    factory!(compound_format, 8);
    factory!(points_format, 9);
    factory!(knn_vectors_format, 10);
    fn select_knn_vectors(
        &self,
        router: &Rc<Token>,
        field: Option<&str>,
    ) -> Result<Rc<Token>, Error> {
        assert!(Rc::ptr_eq(router, &self.0.formats[10]));
        self.0.native_selects.set(self.0.native_selects.get() + 1);
        self.0
            .native_trace
            .set(hash(self.0.native_trace.get(), token(field)));
        Ok(self.0.selected.clone())
    }
}
struct Provider(Rc<Probe>);
macro_rules! default_token {
    ($name:ident) => {
        fn $name(&mut self) -> Result<Rc<Token>, Error> {
            Ok(Rc::new(Token))
        }
    };
}
impl DefaultProvider for Provider {
    type Error = Error;
    type Delegate = SharedProbe;
    type TermVectors = Rc<Token>;
    type StoredFields = Rc<Token>;
    type Norms = Rc<Token>;
    type LiveDocs = Lucene90LiveDocsFormat;
    type Postings = Rc<Token>;
    type DocValues = Rc<Token>;
    type Points = Rc<Token>;
    type KnnVectors = Rc<Token>;
    fn default_codec(&mut self) -> Result<SharedProbe, Error> {
        Ok(SharedProbe(self.0.clone()))
    }
    default_token!(term_vectors);
    default_token!(stored_fields);
    default_token!(norms);
    default_token!(postings);
    default_token!(doc_values);
    default_token!(points);
    default_token!(knn_vectors);
    fn live_docs(&mut self) -> Result<Lucene90LiveDocsFormat, Error> {
        Ok(Lucene90LiveDocsFormat)
    }
}
struct Hooks {
    counts: [Cell<u64>; 3],
    trace: Cell<u64>,
    postings: PostingsSelection<Provider>,
    doc_values: DocValuesSelection<Provider>,
    knn: KnnSelection<Provider>,
}
impl Hooks {
    fn new() -> Self {
        Self {
            counts: std::array::from_fn(|_| Cell::new(0)),
            trace: Cell::new(BASIS),
            postings: Rc::new(AssertingPostingsFormat::new(Rc::new(Token))),
            doc_values: Rc::new(AssertingDocValuesFormat::new(Rc::new(Token))),
            knn: Rc::new(AssertingKnnVectorsFormat::new(Rc::new(Token))),
        }
    }
    fn before(&self, k: usize, f: Option<&str>) {
        self.counts[k].set(self.counts[k].get() + 1);
        self.trace
            .set(hash(self.trace.get(), k as u64 * 4 + token(f)))
    }
    fn reset(&self) {
        for c in &self.counts {
            c.set(0)
        }
        self.trace.set(BASIS)
    }
}
impl Selectors<Provider> for Hooks {
    fn postings(
        &self,
        d: &Defaults<Provider>,
        f: Option<&str>,
    ) -> Result<PostingsSelection<Provider>, Error> {
        self.before(0, f);
        Ok(if f == Some("alternate") {
            self.postings.clone()
        } else {
            d.postings.clone()
        })
    }
    fn doc_values(
        &self,
        d: &Defaults<Provider>,
        f: Option<&str>,
    ) -> Result<DocValuesSelection<Provider>, Error> {
        self.before(1, f);
        Ok(if f == Some("alternate") {
            self.doc_values.clone()
        } else {
            d.doc_values.clone()
        })
    }
    fn knn_vectors(
        &self,
        d: &Defaults<Provider>,
        f: Option<&str>,
    ) -> Result<KnnSelection<Provider>, Error> {
        self.before(2, f);
        Ok(if f == Some("alternate") {
            self.knn.clone()
        } else {
            d.knn_vectors.clone()
        })
    }
}
type Codec = AssertingCodec<Provider, Hooks>;
fn suppression(mask: u8) -> Suppression {
    Suppression::new(
        Format::ALL
            .into_iter()
            .filter(|f| mask & (1 << *f as u8) != 0),
    )
}
fn routed(mask: u8, k: usize) -> bool {
    matches!(k, 1 | 5 | 8) || (BITS[k] >= 0 && mask & (1 << BITS[k]) != 0)
}
// Setup erasure retains expected formats; timed calls consume their actual type.
fn shared_choice<T: Any>(value: FormatChoice<T, Rc<Token>>) -> Handle {
    match value {
        FormatChoice::Cached(value) => value,
        FormatChoice::Delegate(value) => value,
    }
}
#[inline(always)]
fn observe<T: Any>(value: Rc<T>, expected: &Handle, d: u64, event: u64) -> u64 {
    assert!(std::ptr::addr_eq(Rc::as_ptr(&value), Rc::as_ptr(expected)));
    black_box(value);
    hash(d, event)
}
#[inline(always)]
fn observe_choice<T: Any>(
    value: FormatChoice<T, Rc<Token>>,
    expected: &Handle,
    d: u64,
    event: u64,
) -> u64 {
    match value {
        FormatChoice::Cached(value) => observe(value, expected, d, event),
        FormatChoice::Delegate(value) => observe(value, expected, d, event),
    }
}
#[inline(always)]
fn observe_postings(
    value: PerFieldPostings<Provider, Hooks>,
    expected: &PerFieldPostings<Provider, Hooks>,
    d: u64,
    event: u64,
) -> u64 {
    assert!(value.same_instance(expected));
    black_box(value);
    hash(d, event)
}
#[inline(always)]
fn observe_doc_values(
    value: PerFieldDocValues<Provider, Hooks>,
    expected: &PerFieldDocValues<Provider, Hooks>,
    d: u64,
    event: u64,
) -> u64 {
    assert!(value.same_instance(expected));
    black_box(value);
    hash(d, event)
}
#[inline(always)]
fn observe_knn(
    value: PerFieldKnnVectors<Provider, Hooks>,
    expected: &PerFieldKnnVectors<Provider, Hooks>,
    d: u64,
    event: u64,
) -> u64 {
    assert!(value.same_instance(expected));
    black_box(value);
    hash(d, event)
}
struct Bundle {
    codec: Codec,
    probe: Rc<Probe>,
    expected: [Handle; 11],
    defaults: [Handle; 3],
    alternates: [Handle; 3],
    postings: PerFieldPostings<Provider, Hooks>,
    doc_values: PerFieldDocValues<Provider, Hooks>,
    knn: PerFieldKnnVectors<Provider, Hooks>,
    selected: Handle,
    display: [String; 2],
}
impl Bundle {
    fn new(mask: u8) -> Self {
        let probe = Rc::new(Probe::new());
        let codec = Codec::new(
            &mut Provider(probe.clone()),
            || Ok(suppression(mask)),
            Hooks::new(),
        )
        .unwrap();
        // Routers have typed expectations below. Their unused shared slots reuse
        // an existing probe handle and allocate nothing. All format expectations
        // come from the actual accessor once, outside timing.
        let expected: [Handle; 11] = std::array::from_fn(|k| match k {
            1 => codec.field_infos_format().unwrap(),
            2 => shared_choice(codec.live_docs_format().unwrap()),
            3 => shared_choice(codec.norms_format().unwrap()),
            5 => codec.segment_info_format().unwrap(),
            6 => shared_choice(codec.stored_fields_format().unwrap()),
            7 => shared_choice(codec.term_vectors_format().unwrap()),
            8 => codec.compound_format().unwrap(),
            9 => shared_choice(codec.points_format().unwrap()),
            10 => match codec.knn_vectors_format().unwrap() {
                KnnRouterChoice::Delegate(value) => value,
                KnnRouterChoice::Cached(_) => probe.formats[k].clone(),
            },
            _ => probe.formats[k].clone(),
        });
        let defaults: [Handle; 3] = [
            codec.defaults().postings.clone(),
            codec.defaults().doc_values.clone(),
            codec.defaults().knn_vectors.clone(),
        ];
        let alternates: [Handle; 3] = [
            codec.hooks().postings.clone(),
            codec.hooks().doc_values.clone(),
            codec.hooks().knn.clone(),
        ];
        let postings = codec.postings_format();
        let doc_values = codec.doc_values_format();
        let knn = codec.cached_knn_vectors_format();
        let display = std::array::from_fn(|next| {
            probe.next.set(next != 0);
            codec.display().unwrap()
        });
        let selected: Handle = probe.selected.clone();
        Self {
            codec,
            probe,
            expected,
            defaults,
            alternates,
            postings,
            doc_values,
            knn,
            selected,
            display,
        }
    }
    #[inline(always)]
    fn observe_component(&self, k: usize, d: u64, event: u64) -> Result<u64, Error> {
        Ok(match k {
            0 => observe_doc_values(self.codec.doc_values_format(), &self.doc_values, d, event),
            1 => observe(
                self.codec.field_infos_format()?,
                &self.expected[k],
                d,
                event,
            ),
            2 => observe_choice(self.codec.live_docs_format()?, &self.expected[k], d, event),
            3 => observe_choice(self.codec.norms_format()?, &self.expected[k], d, event),
            4 => observe_postings(self.codec.postings_format(), &self.postings, d, event),
            5 => observe(
                self.codec.segment_info_format()?,
                &self.expected[k],
                d,
                event,
            ),
            6 => observe_choice(
                self.codec.stored_fields_format()?,
                &self.expected[k],
                d,
                event,
            ),
            7 => observe_choice(
                self.codec.term_vectors_format()?,
                &self.expected[k],
                d,
                event,
            ),
            8 => observe(self.codec.compound_format()?, &self.expected[k], d, event),
            9 => observe_choice(self.codec.points_format()?, &self.expected[k], d, event),
            10 => match self.codec.knn_vectors_format()? {
                KnnRouterChoice::Cached(value) => observe_knn(value, &self.knn, d, event),
                KnnRouterChoice::Delegate(value) => observe(value, &self.expected[k], d, event),
            },
            _ => unreachable!(),
        })
    }
    #[inline(always)]
    fn observe_selection(
        &self,
        k: usize,
        path: usize,
        f: Option<&str>,
        expected: &Handle,
        d: u64,
        event: u64,
    ) -> Result<u64, Error> {
        Ok(match (k, path) {
            (0, 0) => observe(
                self.codec.get_postings_format_for_field(f)?,
                expected,
                d,
                event,
            ),
            (0, 1) => observe(self.postings.select(f)?, expected, d, event),
            (0, 2) => observe(self.codec.postings_format().select(f)?, expected, d, event),
            (1, 0) => observe(
                self.codec.get_doc_values_format_for_field(f)?,
                expected,
                d,
                event,
            ),
            (1, 1) => observe(self.doc_values.select(f)?, expected, d, event),
            (1, 2) => observe(
                self.codec.doc_values_format().select(f)?,
                expected,
                d,
                event,
            ),
            (2, 0) => observe(
                self.codec.get_knn_vectors_format_for_field(f)?,
                expected,
                d,
                event,
            ),
            (2, 1) => observe(self.knn.select(f)?, expected, d, event),
            (2, 2) => observe_choice(self.codec.select_public_knn_vectors(f)?, expected, d, event),
            _ => unreachable!(),
        })
    }
}
fn batch(case: usize, bundles: &[Bundle; 3]) -> u64 {
    for b in bundles {
        b.probe.reset();
        b.codec.hooks().reset();
        b.probe.fail.set(case == 6)
    }
    let mut d = BASIS;
    match case {
        0..=2 => {
            let m = [0, 2, 1][case];
            let b = &bundles[m];
            for _ in 0..256 {
                for k in 0..11 {
                    for _ in 0..2 {
                        d = b
                            .observe_component(
                                k,
                                d,
                                (k * 3 + usize::from(routed(MASKS[m], k)) * 2) as u64,
                            )
                            .unwrap()
                    }
                }
            }
            assert_eq!(
                b.probe.counts.iter().map(Cell::get).sum::<u64>(),
                [1536, 4608, 3072][case]
            );
        }
        3 | 4 => {
            let m = if case == 3 { 0 } else { 2 };
            let b = &bundles[m];
            for _ in 0..256 {
                for k in 0..3 {
                    for (f, field) in FIELDS.into_iter().enumerate() {
                        for path in 0..3 {
                            let bypass = case == 4 && k == 2 && path == 2;
                            let category = if bypass { 2 } else { usize::from(f == 2) };
                            let expected = if bypass {
                                &b.selected
                            } else if f == 2 {
                                &b.alternates[k]
                            } else {
                                &b.defaults[k]
                            };
                            d = b
                                .observe_selection(
                                    k,
                                    path,
                                    field,
                                    expected,
                                    d,
                                    ((k * 4 + f) * 3 + path) as u64 * 3 + category as u64,
                                )
                                .unwrap()
                        }
                    }
                }
            }
            assert_eq!(
                b.codec.hooks().counts.iter().map(Cell::get).sum::<u64>(),
                if case == 3 { 9216 } else { 8192 }
            );
            assert_eq!(b.probe.counts[10].get(), if case == 3 { 0 } else { 1024 });
            assert_eq!(
                b.probe.native_selects.get(),
                if case == 3 { 0 } else { 1024 }
            );
        }
        5 => {
            for step in 0..256 {
                for b in bundles {
                    b.probe.next.set(step % 2 != 0);
                    let value = black_box(b.codec.display().unwrap());
                    assert_eq!(value, b.display[step % 2]);
                    assert_eq!(b.codec.name(), "Asserting");
                    for ch in value.bytes() {
                        d = hash(d, u64::from(ch))
                    }
                }
            }
            for b in bundles {
                assert_eq!(b.probe.displays.get(), 256)
            }
        }
        6 => {
            for _ in 0..256 {
                for (m, b) in bundles.iter().enumerate() {
                    for k in 0..11 {
                        let forwarded = routed(MASKS[m], k);
                        match b.observe_component(k, d, ((m * 11 + k) * 2) as u64) {
                            Ok(next_digest) => {
                                assert!(!forwarded);
                                d = next_digest
                            }
                            Err(error) => {
                                assert!(forwarded && Rc::ptr_eq(&error, &b.probe.error));
                                black_box(error);
                                d = hash(d, ((m * 11 + k) * 2 + 1) as u64)
                            }
                        }
                    }
                }
            }
            for (m, b) in bundles.iter().enumerate() {
                assert_eq!(
                    b.probe.counts.iter().map(Cell::get).sum::<u64>(),
                    [768, 1536, 2304][m]
                )
            }
        }
        _ => unreachable!(),
    }
    for b in bundles {
        let p = &b.probe;
        d = hash(d, p.counts.iter().map(Cell::get).sum());
        for c in &p.counts {
            d = hash(d, c.get())
        }
        d = hash(d, p.displays.get());
        d = hash(d, p.native_selects.get());
        d = hash(d, p.native_trace.get());
        d = hash(d, p.trace.get());
        for c in &b.codec.hooks().counts {
            d = hash(d, c.get())
        }
        d = hash(d, b.codec.hooks().trace.get())
    }
    black_box(d)
}
fn rc_bytes<T>(_: &Rc<T>) -> usize {
    let (layout, _) = Layout::new::<[usize; 2]>()
        .extend(Layout::new::<T>())
        .unwrap();
    layout.pad_to_align().size()
}
fn memory() {
    let c = AssertingCodec::new(
        &mut Provider(Rc::new(Probe::new())),
        || Ok(Suppression::default()),
        DefaultSelectors,
    )
    .unwrap();
    let d = c.defaults();
    let wrappers = rc_bytes(&d.term_vectors)
        + rc_bytes(&d.stored_fields)
        + rc_bytes(&d.norms)
        + rc_bytes(&d.live_docs)
        + rc_bytes(&d.postings)
        + rc_bytes(&d.doc_values)
        + rc_bytes(&d.points)
        + rc_bytes(&d.knn_vectors);
    // The three typed router handles are inline in codec shallow bytes.
    // Their shared state allocation is counted separately, exactly once.
    let routers = 0;
    let shallow = std::mem::size_of_val(&c);
    let state = c.selector_state_bytes();
    for (name, value) in [
        ("shallow_codec", shallow),
        ("cached_wrappers", wrappers),
        ("cached_routers", routers),
        ("shared_selector", state),
        ("suppression", 0),
        ("selected_sum", shallow + wrappers + routers + state),
    ] {
        println!("asserting_codec_{name}_bytes\t{value}")
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup = args.get(1).map(|x| x.parse().unwrap()).unwrap_or(1000);
    let measure = args.get(2).map(|x| x.parse().unwrap()).unwrap_or(1500);
    let bundles = MASKS.map(Bundle::new);
    memory();
    for (case, name) in [
        "asserting_codec_cached",
        "asserting_codec_suppressed",
        "asserting_codec_mixed",
        "asserting_codec_selectors",
        "asserting_codec_knn_bypass",
        "asserting_codec_display",
        "asserting_codec_errors",
    ]
    .into_iter()
    .enumerate()
    {
        let expected = batch(case, &bundles);
        for (phase, budget) in [warmup, measure].into_iter().enumerate() {
            let start = Instant::now();
            let mut operations = 0u64;
            loop {
                assert_eq!(batch(case, &bundles), expected);
                operations += 256;
                if start.elapsed() >= Duration::from_millis(budget) {
                    break;
                }
            }
            if phase == 1 {
                println!(
                    "{name}\t{:.3}\t{operations}",
                    start.elapsed().as_nanos() as f64 / operations as f64
                )
            }
        }
        println!("#check\t{name}\t{expected}\t256")
    }
}
