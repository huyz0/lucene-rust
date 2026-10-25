//! Actual Java AssertingCodec composition rows, replayed through explicit providers.
//! Native descriptors below are opaque injected handles: this fixture does not
//! claim a native codec graph or exercise component/per-field I/O.
use lucene_test_framework::codecs::asserting_codec::*;
use lucene_test_framework::codecs::{
    asserting_doc_values::AssertingDocValuesFormat,
    asserting_knn_vectors::AssertingKnnVectorsFormat, asserting_postings::AssertingPostingsFormat,
    cranky_live_docs::Lucene90LiveDocsFormat,
};
use lucene_util::java_random::JavaRandom;
use std::{
    any::{type_name, Any},
    cell::{Cell, RefCell},
    rc::Rc,
};

type Error = Rc<str>;
#[derive(Debug)]
struct Native {
    class: &'static str,
    name: &'static str,
    threshold: i32,
}
fn native(class: &'static str, name: &'static str, threshold: i32) -> Rc<Native> {
    Rc::new(Native {
        class,
        name,
        threshold,
    })
}
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
const NATIVE_CLASSES: [&str; 11] = [
    "PerFieldDocValuesFormat",
    "Lucene94FieldInfosFormat",
    "Lucene90LiveDocsFormat",
    "Lucene90NormsFormat",
    "PerFieldPostingsFormat",
    "Lucene99SegmentInfoFormat",
    "Lucene90StoredFieldsFormat",
    "Lucene90TermVectorsFormat",
    "Lucene90CompoundFormat",
    "Lucene90PointsFormat",
    "PerFieldKnnVectorsFormat",
];
const NATIVE_NAMES: [&str; 11] = [
    "PerFieldDV40",
    "-",
    "-",
    "-",
    "PerField40",
    "-",
    "-",
    "-",
    "-",
    "-",
    "PerFieldVectors90",
];
struct Probe {
    events: RefCell<Vec<String>>,
    fail: Cell<usize>,
    fail_display: Cell<bool>,
    label: Cell<&'static str>,
    error: Error,
    formats: [Rc<Native>; 11],
    last: RefCell<Option<Rc<Native>>>,
}
impl Probe {
    fn new() -> Self {
        Self {
            events: RefCell::new(vec![]),
            fail: Cell::new(usize::MAX),
            fail_display: Cell::new(false),
            label: Cell::new("Lucene104"),
            error: Rc::from("probe"),
            formats: std::array::from_fn(|i| native(NATIVE_CLASSES[i], NATIVE_NAMES[i], 100)),
            last: RefCell::new(None),
        }
    }
    fn format(&self, kind: usize) -> Result<Rc<Native>, Error> {
        self.events
            .borrow_mut()
            .push(format!("factory:{}", KINDS[kind]));
        if self.fail.get() == kind {
            return Err(self.error.clone());
        }
        let result = if kind == 9 {
            native(NATIVE_CLASSES[kind], NATIVE_NAMES[kind], 100)
        } else {
            self.formats[kind].clone()
        };
        *self.last.borrow_mut() = Some(result.clone());
        Ok(result)
    }
    fn take(&self) -> String {
        std::mem::take(&mut *self.events.borrow_mut()).join(",")
    }
}
impl Delegate for Probe {
    type Error = Error;
    type DocValues = Rc<Native>;
    type FieldInfos = Rc<Native>;
    type LiveDocs = Rc<Native>;
    type Norms = Rc<Native>;
    type Postings = Rc<Native>;
    type SegmentInfo = Rc<Native>;
    type StoredFields = Rc<Native>;
    type TermVectors = Rc<Native>;
    type Compound = Rc<Native>;
    type Points = Rc<Native>;
    type KnnVectors = Rc<Native>;
    type SelectedKnnVectors = Rc<Native>;
    fn name(&self) -> &str {
        "Lucene104"
    }
    fn display(&self) -> Result<String, Error> {
        self.events.borrow_mut().push("display".into());
        if self.fail_display.get() {
            Err(self.error.clone())
        } else {
            Ok(self.label.get().into())
        }
    }
    fn doc_values_format(&self) -> Result<Rc<Native>, Error> {
        self.format(0)
    }
    fn field_infos_format(&self) -> Result<Rc<Native>, Error> {
        self.format(1)
    }
    fn live_docs_format(&self) -> Result<Rc<Native>, Error> {
        self.format(2)
    }
    fn norms_format(&self) -> Result<Rc<Native>, Error> {
        self.format(3)
    }
    fn postings_format(&self) -> Result<Rc<Native>, Error> {
        self.format(4)
    }
    fn segment_info_format(&self) -> Result<Rc<Native>, Error> {
        self.format(5)
    }
    fn stored_fields_format(&self) -> Result<Rc<Native>, Error> {
        self.format(6)
    }
    fn term_vectors_format(&self) -> Result<Rc<Native>, Error> {
        self.format(7)
    }
    fn compound_format(&self) -> Result<Rc<Native>, Error> {
        self.format(8)
    }
    fn points_format(&self) -> Result<Rc<Native>, Error> {
        self.format(9)
    }
    fn knn_vectors_format(&self) -> Result<Rc<Native>, Error> {
        self.format(10)
    }
    fn select_knn_vectors(
        &self,
        router: &Rc<Native>,
        _field: Option<&str>,
    ) -> Result<Rc<Native>, Error> {
        assert!(Rc::ptr_eq(router, &self.formats[10]));
        Ok(native(
            "Lucene99HnswVectorsFormat",
            "Lucene99HnswVectorsFormat",
            100,
        ))
    }
}
const ORDER: [&str; 10] = [
    "codec",
    "termVectors",
    "storedFields",
    "norms",
    "liveDocs",
    "postings",
    "docValues",
    "points",
    "knnVectors",
    "resolve",
];
struct Provider {
    calls: Rc<RefCell<Vec<&'static str>>>,
    fail: usize,
    error: Error,
}
impl Default for Provider {
    fn default() -> Self {
        Self {
            calls: Rc::new(RefCell::new(vec![])),
            fail: usize::MAX,
            error: Rc::from("default"),
        }
    }
}
impl Provider {
    fn before(&mut self, kind: usize) -> Result<(), Error> {
        self.calls.borrow_mut().push(ORDER[kind]);
        if self.fail == kind {
            Err(self.error.clone())
        } else {
            Ok(())
        }
    }
}
impl DefaultProvider for Provider {
    type Error = Error;
    type Delegate = Probe;
    type TermVectors = Rc<Native>;
    type StoredFields = Rc<Native>;
    type Norms = Rc<Native>;
    type LiveDocs = Lucene90LiveDocsFormat;
    type Postings = Rc<Native>;
    type DocValues = Rc<Native>;
    type Points = Rc<Native>;
    type KnnVectors = Rc<Native>;
    fn default_codec(&mut self) -> Result<Probe, Error> {
        self.before(0)?;
        Ok(Probe::new())
    }
    fn term_vectors(&mut self) -> Result<Rc<Native>, Error> {
        self.before(1)?;
        Ok(native("Lucene90TermVectorsFormat", "-", 0))
    }
    fn stored_fields(&mut self) -> Result<Rc<Native>, Error> {
        self.before(2)?;
        Ok(native("Lucene90StoredFieldsFormat", "-", 0))
    }
    fn norms(&mut self) -> Result<Rc<Native>, Error> {
        self.before(3)?;
        Ok(native("Lucene90NormsFormat", "-", 0))
    }
    fn live_docs(&mut self) -> Result<Lucene90LiveDocsFormat, Error> {
        self.before(4)?;
        Ok(Lucene90LiveDocsFormat)
    }
    fn postings(&mut self) -> Result<Rc<Native>, Error> {
        self.before(5)?;
        Ok(native("Lucene104PostingsFormat", "Lucene104", 0))
    }
    fn doc_values(&mut self) -> Result<Rc<Native>, Error> {
        self.before(6)?;
        Ok(native("Lucene90DocValuesFormat", "Lucene90", 0))
    }
    fn points(&mut self) -> Result<Rc<Native>, Error> {
        self.before(7)?;
        Ok(native("Lucene90PointsFormat", "-", 0))
    }
    fn knn_vectors(&mut self) -> Result<Rc<Native>, Error> {
        self.before(8)?;
        Ok(native(
            "Lucene99HnswVectorsFormat",
            "Lucene99HnswVectorsFormat",
            0,
        ))
    }
}
struct Alternatives {
    postings: PostingsSelection<Provider>,
    doc_values: DocValuesSelection<Provider>,
    knn_vectors: KnnSelection<Provider>,
}
struct Hooks {
    enabled: Cell<bool>,
    events: RefCell<Vec<String>>,
    fail: Cell<Option<usize>>,
    error: Error,
    alternatives: std::cell::OnceCell<Alternatives>,
}
impl Default for Hooks {
    fn default() -> Self {
        Self {
            enabled: Cell::new(false),
            events: RefCell::new(vec![]),
            fail: Cell::new(None),
            error: Rc::from("selector"),
            alternatives: std::cell::OnceCell::new(),
        }
    }
}
impl Hooks {
    fn before(&self, kind: usize, field: Option<&str>) -> Result<bool, Error> {
        if self.enabled.get() {
            self.events
                .borrow_mut()
                .push(format!("{kind}:{}", field.unwrap_or("null")));
        }
        if self.fail.get() == Some(kind) {
            return Err(self.error.clone());
        }
        Ok(field == Some("alternate"))
    }
    fn take(&self) -> String {
        std::mem::take(&mut *self.events.borrow_mut()).join(",")
    }
}
impl Selectors<Provider> for Hooks {
    fn postings(
        &self,
        d: &Defaults<Provider>,
        field: Option<&str>,
    ) -> Result<PostingsSelection<Provider>, Error> {
        Ok(if self.before(0, field)? {
            self.alternatives.get().unwrap().postings.clone()
        } else {
            d.postings.clone()
        })
    }
    fn doc_values(
        &self,
        d: &Defaults<Provider>,
        field: Option<&str>,
    ) -> Result<DocValuesSelection<Provider>, Error> {
        Ok(if self.before(1, field)? {
            self.alternatives.get().unwrap().doc_values.clone()
        } else {
            d.doc_values.clone()
        })
    }
    fn knn_vectors(
        &self,
        d: &Defaults<Provider>,
        field: Option<&str>,
    ) -> Result<KnnSelection<Provider>, Error> {
        Ok(if self.before(2, field)? {
            self.alternatives.get().unwrap().knn_vectors.clone()
        } else {
            d.knn_vectors.clone()
        })
    }
}
type Codec = AssertingCodec<Provider, Hooks>;
fn codec(mask: u8) -> Codec {
    let mut provider = Provider::default();
    let calls = provider.calls.clone();
    let codec = AssertingCodec::new(
        &mut provider,
        || {
            calls.borrow_mut().push("resolve");
            Ok(Suppression::new(
                Format::ALL
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, f)| (mask & (1 << i) != 0).then_some(f)),
            ))
        },
        Hooks::default(),
    )
    .unwrap();
    assert_eq!(*provider.calls.borrow(), ORDER);
    assert!(codec.hooks().take().is_empty());
    let _ = codec.hooks().alternatives.set(Alternatives {
        postings: Rc::new(AssertingPostingsFormat::new(native(
            "Lucene104PostingsFormat",
            "Lucene104",
            0,
        ))),
        doc_values: Rc::new(AssertingDocValuesFormat::new(native(
            "Lucene90DocValuesFormat",
            "Lucene90",
            0,
        ))),
        knn_vectors: Rc::new(AssertingKnnVectorsFormat::new(native(
            "Lucene99HnswVectorsFormat",
            "Lucene99HnswVectorsFormat",
            0,
        ))),
    });
    codec
}
enum Identity {
    Shared(Rc<dyn Any>),
    Postings(PerFieldPostings<Provider, Hooks>),
    DocValues(PerFieldDocValues<Provider, Hooks>),
    Knn(PerFieldKnnVectors<Provider, Hooks>),
}
impl Identity {
    fn shared(&self) -> &Rc<dyn Any> {
        match self {
            Self::Shared(value) => value,
            _ => panic!("expected format handle"),
        }
    }
}
struct Handle {
    id: Identity,
    class: &'static str,
    name: &'static str,
    cached: bool,
}
fn cached<A: Any>(value: Rc<A>, name: &'static str) -> Handle {
    let rust = type_name::<A>()
        .split('<')
        .next()
        .unwrap()
        .rsplit("::")
        .next()
        .unwrap();
    let class = match rust {
        "PerFieldPostings" => "PerFieldPostingsFormat",
        "PerFieldDocValues" => "PerFieldDocValuesFormat",
        "PerFieldKnnVectors" => "PerFieldKnnVectorsFormat",
        other => other,
    };
    Handle {
        id: Identity::Shared(value),
        class,
        name,
        cached: true,
    }
}
fn postings_router(value: PerFieldPostings<Provider, Hooks>, name: &'static str) -> Handle {
    Handle {
        id: Identity::Postings(value),
        class: "PerFieldPostingsFormat",
        name,
        cached: true,
    }
}
fn doc_values_router(value: PerFieldDocValues<Provider, Hooks>, name: &'static str) -> Handle {
    Handle {
        id: Identity::DocValues(value),
        class: "PerFieldDocValuesFormat",
        name,
        cached: true,
    }
}
fn knn_router(value: PerFieldKnnVectors<Provider, Hooks>, name: &'static str) -> Handle {
    Handle {
        id: Identity::Knn(value),
        class: "PerFieldKnnVectorsFormat",
        name,
        cached: true,
    }
}
fn knn_choice(value: KnnRouterChoice<Provider, Hooks>, name: &'static str) -> Handle {
    match value {
        KnnRouterChoice::Cached(router) => knn_router(router, name),
        KnnRouterChoice::Delegate(value) => delegated(value),
    }
}
fn delegated(value: Rc<Native>) -> Handle {
    Handle {
        class: value.class,
        name: value.name,
        id: Identity::Shared(value),
        cached: false,
    }
}
fn choice<A: Any>(value: FormatChoice<A, Rc<Native>>, name: &'static str) -> Handle {
    match value {
        FormatChoice::Cached(v) => cached(v, name),
        FormatChoice::Delegate(v) => delegated(v),
    }
}
fn component(codec: &Codec, kind: usize) -> Result<Handle, Error> {
    Ok(match kind {
        0 => doc_values_router(codec.doc_values_format(), "PerFieldDV40"),
        1 => delegated(codec.field_infos_format()?),
        2 => choice(codec.live_docs_format()?, "-"),
        3 => choice(codec.norms_format()?, "-"),
        4 => postings_router(codec.postings_format(), "PerField40"),
        5 => delegated(codec.segment_info_format()?),
        6 => choice(codec.stored_fields_format()?, "-"),
        7 => choice(codec.term_vectors_format()?, "-"),
        8 => delegated(codec.compound_format()?),
        9 => choice(codec.points_format()?, "-"),
        10 => knn_choice(codec.knn_vectors_format()?, "PerFieldVectors90"),
        _ => unreachable!(),
    })
}
fn selector(codec: &Codec, kind: usize, field: Option<&str>, path: usize) -> Result<Handle, Error> {
    Ok(match (kind, path) {
        (0, 0) => cached(codec.get_postings_format_for_field(field)?, "Asserting"),
        (0, _) => cached(codec.postings_format().select(field)?, "Asserting"),
        (1, 0) => cached(codec.get_doc_values_format_for_field(field)?, "Asserting"),
        (1, _) => cached(codec.doc_values_format().select(field)?, "Asserting"),
        (2, 0) => cached(codec.get_knn_vectors_format_for_field(field)?, "Asserting"),
        (2, 1) => cached(
            codec.cached_knn_vectors_format().select(field)?,
            "Asserting",
        ),
        (2, 2) => choice(codec.select_public_knn_vectors(field)?, "Asserting"),
        _ => unreachable!(),
    })
}
fn row(out: &mut String, fields: &[&str]) {
    out.push_str(&fields.join("\t"));
    out.push('\n');
}
fn same(a: &Handle, b: &Handle) -> bool {
    match (&a.id, &b.id) {
        (Identity::Shared(a), Identity::Shared(b)) => Rc::ptr_eq(a, b),
        (Identity::Postings(a), Identity::Postings(b)) => a.same_instance(b),
        (Identity::DocValues(a), Identity::DocValues(b)) => a.same_instance(b),
        (Identity::Knn(a), Identity::Knn(b)) => a.same_instance(b),
        _ => false,
    }
}
fn outcome(result: Result<Handle, Error>) -> &'static str {
    match result {
        Ok(_) => "ok",
        Err(error) => {
            assert!(error.as_ref() == "probe" || error.as_ref() == "selector");
            if error.as_ref() == "probe" {
                "runtime:probe"
            } else {
                "runtime:selector"
            }
        }
    }
}

fn snapshot(out: &mut String, label: &str, mask: u8) {
    let c = codec(mask);
    let second = codec(mask);
    let display = c.display().unwrap();
    c.delegate().take();
    assert_eq!(c.name(), "Asserting");
    assert_eq!(c.delegate().name(), "Lucene104");
    row(
        out,
        &[
            "config",
            label,
            &mask.to_string(),
            c.name(),
            "Lucene104Codec",
            c.delegate().name(),
            &display,
        ],
    );
    let d = c.defaults();
    let e = second.defaults();
    let caches = [
        (
            "postings",
            postings_router(c.postings_format(), "PerField40"),
            postings_router(second.postings_format(), "PerField40"),
            None,
        ),
        (
            "docValues",
            doc_values_router(c.doc_values_format(), "PerFieldDV40"),
            doc_values_router(second.doc_values_format(), "PerFieldDV40"),
            None,
        ),
        (
            "knnVectorsFormat",
            knn_router(c.cached_knn_vectors_format(), "PerFieldVectors90"),
            knn_router(second.cached_knn_vectors_format(), "PerFieldVectors90"),
            None,
        ),
        (
            "vectors",
            cached(d.term_vectors.clone(), "-"),
            cached(e.term_vectors.clone(), "-"),
            Some(d.term_vectors.delegate().as_ref()),
        ),
        (
            "storedFields",
            cached(d.stored_fields.clone(), "-"),
            cached(e.stored_fields.clone(), "-"),
            Some(d.stored_fields.delegate().as_ref()),
        ),
        (
            "norms",
            cached(d.norms.clone(), "-"),
            cached(e.norms.clone(), "-"),
            Some(d.norms.delegate().as_ref()),
        ),
        (
            "liveDocs",
            cached(d.live_docs.clone(), "-"),
            cached(e.live_docs.clone(), "-"),
            None,
        ),
        (
            "defaultFormat",
            cached(d.postings.clone(), "Asserting"),
            cached(e.postings.clone(), "Asserting"),
            Some(d.postings.delegate().as_ref()),
        ),
        (
            "defaultDVFormat",
            cached(d.doc_values.clone(), "Asserting"),
            cached(e.doc_values.clone(), "Asserting"),
            Some(d.doc_values.delegate().as_ref()),
        ),
        (
            "pointsFormat",
            cached(d.points.clone(), "-"),
            cached(e.points.clone(), "-"),
            Some(d.points.delegate().as_ref()),
        ),
        (
            "defaultKnnVectorsFormat",
            cached(d.knn_vectors.clone(), "Asserting"),
            cached(e.knn_vectors.clone(), "Asserting"),
            Some(d.knn_vectors.delegate().as_ref()),
        ),
    ];
    for (field, value, other, inner) in caches {
        assert!(!same(&value, &other));
        if let Some(inner) = inner {
            row(
                out,
                &[
                    "cache",
                    label,
                    field,
                    value.class,
                    value.name,
                    "independent",
                    inner.class,
                    inner.name,
                ],
            );
        } else if field == "liveDocs" {
            row(
                out,
                &[
                    "cache",
                    label,
                    field,
                    value.class,
                    value.name,
                    "independent",
                    "Lucene90LiveDocsFormat",
                    "-",
                ],
            );
        } else {
            row(
                out,
                &[
                    "cache",
                    label,
                    field,
                    value.class,
                    value.name,
                    "independent",
                ],
            );
        }
    }
    assert_eq!(d.knn_vectors.delegate().threshold, 0);
    row(
        out,
        &[
            "knn-defaults",
            label,
            "16",
            "100",
            &d.knn_vectors.delegate().threshold.to_string(),
            "1",
            "100",
            &d.knn_vectors.max_dimensions("field").to_string(),
        ],
    );
    for kind in 0..3 {
        for field in [Some(""), Some("field"), None] {
            let first = selector(&c, kind, field, 0).unwrap();
            let again = selector(&c, kind, Some("other"), 0).unwrap();
            assert!(same(&first, &again));
            row(
                out,
                &[
                    "default-selector",
                    label,
                    &kind.to_string(),
                    field.unwrap_or("null"),
                    first.class,
                    first.name,
                    "cached",
                ],
            );
        }
    }
    for (kind, name) in KINDS.iter().enumerate() {
        let first = component(&c, kind).unwrap();
        let again = component(&c, kind).unwrap();
        row(
            out,
            &[
                "accessor",
                label,
                name,
                first.class,
                first.name,
                &same(&first, &again).to_string(),
                if first.cached { "cached" } else { "delegate" },
            ],
        );
    }
    c.delegate().take();
    c.delegate().label.set("ProbeFirst");
    for (kind, name) in KINDS.iter().enumerate() {
        let first = component(&c, kind).unwrap();
        let returned = c.delegate().last.borrow().clone();
        let again = component(&c, kind).unwrap();
        if !first.cached {
            let returned: Rc<dyn Any> = returned.unwrap();
            assert!(Rc::ptr_eq(first.id.shared(), &returned));
            let last: Rc<dyn Any> = c.delegate().last.borrow().clone().unwrap();
            assert!(Rc::ptr_eq(again.id.shared(), &last));
        }
        let trace = c.delegate().take();
        row(
            out,
            &[
                "probe",
                label,
                name,
                &same(&first, &again).to_string(),
                &trace,
            ],
        );
        c.delegate().fail.set(kind);
        let result = component(&c, kind);
        if let Err(error) = &result {
            assert!(Rc::ptr_eq(error, &c.delegate().error));
        }
        let status = outcome(result);
        row(
            out,
            &["probe-failure", label, name, status, &c.delegate().take()],
        );
        c.delegate().fail.set(usize::MAX);
    }
    for value in ["ProbeFirst", "ProbeNext"] {
        c.delegate().label.set(value);
        let actual = c.display().unwrap();
        let trace = c.delegate().take();
        row(out, &["display", label, &actual, &trace, c.name()]);
    }
    c.delegate().fail_display.set(true);
    let error = c.display().unwrap_err();
    assert!(Rc::ptr_eq(&error, &c.delegate().error));
    row(
        out,
        &[
            "display-failure",
            label,
            "runtime:probe",
            &c.delegate().take(),
        ],
    );
    let c = codec(mask);
    c.hooks().enabled.set(true);
    for kind in 0..3 {
        for field in [Some(""), Some("field"), Some("alternate"), None] {
            let direct = selector(&c, kind, field, 0).unwrap();
            let trace = c.hooks().take();
            assert_eq!(trace, format!("{kind}:{}", field.unwrap_or("null")));
            let internal = selector(&c, kind, field, 1).unwrap();
            assert!(same(&direct, &internal));
            assert_eq!(c.hooks().take(), trace);
            let public = selector(&c, kind, field, 2).unwrap();
            let bypass = kind == 2 && mask & 32 != 0;
            if bypass {
                assert!(c.hooks().take().is_empty());
            } else {
                assert!(same(&direct, &public));
                assert_eq!(c.hooks().take(), trace);
            }
            row(
                out,
                &[
                    "hook",
                    label,
                    &kind.to_string(),
                    field.unwrap_or("null"),
                    direct.class,
                    direct.name,
                    "identity",
                    if bypass { "delegate" } else { "hook" },
                    public.class,
                    public.name,
                ],
            );
        }
        c.hooks().fail.set(Some(kind));
        for path in 0..3 {
            let result = selector(&c, kind, Some("failure"), path);
            if let Err(error) = &result {
                assert!(Rc::ptr_eq(error, &c.hooks().error));
            }
            let status = outcome(result);
            row(
                out,
                &[
                    "hook-failure",
                    label,
                    &kind.to_string(),
                    &path.to_string(),
                    status,
                    &c.hooks().take(),
                ],
            );
        }
        c.hooks().fail.set(None);
    }
    row(
        out,
        &[
            "random-tail",
            label,
            &JavaRandom::new(42).next_int().to_string(),
        ],
    );
}
#[test]
fn actual_java_5445_composition_rows_match() {
    let expected = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/testframework/data/asserting_codec.tsv"),
    )
    .unwrap();
    let mut out = String::new();
    snapshot(&mut out, "NoContext", 0);
    for mask in 0..64 {
        snapshot(&mut out, &format!("Mask{mask:02}"), mask);
    }
    snapshot(&mut out, "Unannotated", 0);
    snapshot(&mut out, "DuplicatedReversed", 49);
    snapshot(&mut out, "Inherited", 6);
    assert_thread(Some("values"), Some(std::thread::current().id()));
    row(&mut out, &["thread", "same", "ok"]);
    let owner = std::thread::current().id();
    for object in [Some("values"), Some(""), None] {
        let (error,expected)=std::thread::spawn(move || {
            let current=std::thread::current().id();
            let expected=format!("{} are only supposed to be consumed in the thread in which they have been acquired. But was acquired in {owner:?} and consumed in {current:?}.",object.unwrap_or("null"));
            (std::panic::catch_unwind(|| assert_thread(object,Some(owner))).unwrap_err(),expected)
        }).join().unwrap();
        assert_eq!(error.downcast_ref::<String>().unwrap(), &expected);
        row(
            &mut out,
            &[
                "thread",
                "different",
                object.unwrap_or("null"),
                "assertion",
                "message-exact",
            ],
        );
    }
    assert!(std::panic::catch_unwind(|| assert_thread(Some("values"), None)).is_err());
    row(&mut out, &["thread", "null-owner", "assertion"]);
    assert_eq!(out.lines().count(), 5445);
    assert_eq!(out, expected);
}
#[test]
fn eager_constructor_errors_stop_in_java_order_and_keep_identity() {
    for fail in 0..10 {
        let mut provider = Provider {
            fail,
            ..Provider::default()
        };
        let calls = provider.calls.clone();
        let error = provider.error.clone();
        let result = AssertingCodec::new(
            &mut provider,
            || {
                calls.borrow_mut().push("resolve");
                if fail == 9 {
                    Err(error.clone())
                } else {
                    Ok(Suppression::default())
                }
            },
            DefaultSelectors,
        );
        let actual = match result {
            Err(e) => e,
            Ok(_) => panic!("expected construction failure"),
        };
        assert!(Rc::ptr_eq(&actual, &error));
        assert_eq!(*calls.borrow(), ORDER[..=fail]);
    }
}
#[test]
fn routers_outlive_codec_and_retain_same_hook_defaults() {
    let c = codec(63);
    assert!(
        c.selector_state_bytes()
            >= std::mem::size_of_val(c.defaults())
                + std::mem::size_of_val(c.hooks())
                + 2 * std::mem::size_of::<usize>()
    );
    let defaults = c.defaults().postings.clone();
    let router = c.postings_format();
    let dv = c.doc_values_format();
    let knn = c.cached_knn_vectors_format();
    let independent = codec(63);
    assert!(router.same_instance(&c.postings_format()));
    assert!(dv.same_instance(&c.doc_values_format()));
    assert!(knn.same_instance(&c.cached_knn_vectors_format()));
    assert!(!router.same_instance(&independent.postings_format()));
    assert!(!dv.same_instance(&independent.doc_values_format()));
    assert!(!knn.same_instance(&independent.cached_knn_vectors_format()));
    let router_clone = router.clone();
    let dv_clone = dv.clone();
    let knn_clone = knn.clone();
    drop(c);
    drop(independent);
    assert!(router.same_instance(&router_clone));
    assert!(dv.same_instance(&dv_clone));
    assert!(knn.same_instance(&knn_clone));
    assert!(Rc::ptr_eq(&router.select(None).unwrap(), &defaults));
    assert_eq!(dv.select(None).unwrap().delegate().name, "Lucene90");
    assert_eq!(knn.select(None).unwrap().delegate().threshold, 0);
    let mut provider = Provider::default();
    let c = AssertingCodec::new(
        &mut provider,
        || Ok(Suppression::default()),
        DefaultSelectors,
    )
    .unwrap();
    assert_eq!(PerFieldPostings::<Provider, Hooks>::NAME, "PerField40");
    assert_eq!(PerFieldDocValues::<Provider, Hooks>::NAME, "PerFieldDV40");
    assert_eq!(
        PerFieldKnnVectors::<Provider, Hooks>::NAME,
        "PerFieldVectors90"
    );
    assert_eq!(c.suppression(), Suppression::default());
    assert_eq!(c.to_string(), "Asserting(Lucene104)");
    assert!(Rc::ptr_eq(
        &c.postings_format().select(None).unwrap(),
        &c.defaults().postings
    ));
    assert!(Rc::ptr_eq(
        &c.doc_values_format().select(None).unwrap(),
        &c.defaults().doc_values
    ));
    assert!(Rc::ptr_eq(
        &c.cached_knn_vectors_format().select(None).unwrap(),
        &c.defaults().knn_vectors
    ));
    c.delegate().fail_display.set(true);
    use std::fmt::Write;
    assert!(write!(&mut String::new(), "{c}").is_err());
    // Exercise the retained delegate seam separately: the codec never calls its
    // overridden DV/postings factories for the cached per-field accessors.
    assert_eq!(
        c.delegate().doc_values_format().unwrap().name,
        "PerFieldDV40"
    );
    assert_eq!(c.delegate().postings_format().unwrap().name, "PerField40");
}
