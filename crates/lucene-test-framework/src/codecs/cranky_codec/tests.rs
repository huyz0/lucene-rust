#![cfg(test)]
use super::*;
use std::cell::Cell;

struct FailingCodec {
    name: Cell<&'static str>,
    calls: RefCell<Vec<&'static str>>,
    display: Cell<&'static str>,
}
impl fmt::Display for FailingCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.calls.borrow_mut().push("display");
        f.write_str(self.display.get())
    }
}
impl Codec for FailingCodec {
    type Error = &'static str;
    type DocValues = super::super::cranky_doc_values::Lucene90DocValuesFormat;
    type FieldInfos = super::super::cranky_field_infos::Lucene94FieldInfosFormat;
    type LiveDocs = super::super::cranky_live_docs::Lucene90LiveDocsFormat;
    type Norms = super::super::cranky_norms::Lucene90NormsFormat;
    type Postings = super::super::cranky_postings::Lucene104PostingsFormat;
    type SegmentInfo = super::super::cranky_segment_info::Lucene99SegmentInfoFormat;
    type StoredFields = super::super::cranky_stored_fields::Lucene90StoredFieldsFormat;
    type TermVectors = super::super::cranky_term_vectors::SimpleTextVectorsFormat;
    type Compound = super::super::cranky_compound::Lucene90CompoundFormat;
    type Points = super::super::cranky_points::Lucene90PointsFormat;
    type KnnVectors = ();

    fn name(&self) -> &str {
        self.calls.borrow_mut().push("name");
        self.name.get()
    }
    fn doc_values_format(&self) -> Result<Self::DocValues, Self::Error> {
        self.calls.borrow_mut().push("doc_values_format");
        Err("doc_values_format")
    }
    fn field_infos_format(&self) -> Result<Self::FieldInfos, Self::Error> {
        self.calls.borrow_mut().push("field_infos_format");
        Err("field_infos_format")
    }
    fn live_docs_format(&self) -> Result<Self::LiveDocs, Self::Error> {
        self.calls.borrow_mut().push("live_docs_format");
        Err("live_docs_format")
    }
    fn norms_format(&self) -> Result<Self::Norms, Self::Error> {
        self.calls.borrow_mut().push("norms_format");
        Err("norms_format")
    }
    fn postings_format(&self) -> Result<Self::Postings, Self::Error> {
        self.calls.borrow_mut().push("postings_format");
        Err("postings_format")
    }
    fn segment_info_format(&self) -> Result<Self::SegmentInfo, Self::Error> {
        self.calls.borrow_mut().push("segment_info_format");
        Err("segment_info_format")
    }
    fn stored_fields_format(&self) -> Result<Self::StoredFields, Self::Error> {
        self.calls.borrow_mut().push("stored_fields_format");
        Err("stored_fields_format")
    }
    fn term_vectors_format(&self) -> Result<Self::TermVectors, Self::Error> {
        self.calls.borrow_mut().push("term_vectors_format");
        Err("term_vectors_format")
    }
    fn compound_format(&self) -> Result<Self::Compound, Self::Error> {
        self.calls.borrow_mut().push("compound_format");
        Err("compound_format")
    }
    fn points_format(&self) -> Result<Self::Points, Self::Error> {
        self.calls.borrow_mut().push("points_format");
        Err("points_format")
    }
    fn knn_vectors_format(&self) -> Result<Self::KnnVectors, Self::Error> {
        self.calls.borrow_mut().push("knn_vectors_format");
        Err("knn_vectors_format")
    }
}
fn codec() -> CrankyCodec<FailingCodec> {
    CrankyCodec::new(
        FailingCodec {
            name: Cell::new("original"),
            calls: RefCell::new(vec![]),
            display: Cell::new("first"),
        },
        Rc::new(RefCell::new(JavaRandom::new(42))),
    )
}
#[test]
fn construction_snapshots_name_and_display_remains_live() {
    let codec = codec();
    assert_eq!(&*codec.delegate().calls.borrow(), &["name"]);
    // A deliberately contract-breaking delegate proves that the outer name
    // is a snapshot, rather than a second virtual name lookup.
    codec.delegate().name.set("changed");
    assert_eq!(codec.name(), "original");
    assert_eq!(Codec::name(&codec), "original");
    assert_eq!(format!("{codec}"), "Cranky(first)");
    codec.delegate().display.set("second");
    assert_eq!(format!("{codec}"), "Cranky(second)");
    assert_eq!(
        &*codec.delegate().calls.borrow(),
        &["name", "display", "display"]
    );
}
#[test]
fn name_storage_boundaries_utf8_and_long_snapshots() {
    for name in [
        "",
        "123456789012345",
        "1234567890123456",
        "éééééééa",
        "éééééééé",
        "🦀🦀🦀",
        "🦀🦀🦀🦀",
        "a deliberately long delegate name retained as an immutable snapshot",
    ] {
        let random = Rc::new(RefCell::new(JavaRandom::new(42)));
        let codec = CrankyCodec::new(
            FailingCodec {
                name: Cell::new(name),
                calls: RefCell::new(vec![]),
                display: Cell::new("first"),
            },
            random.clone(),
        );
        assert_eq!(&*codec.delegate().calls.borrow(), &["name"]);
        assert!(Rc::ptr_eq(codec.random(), &random));
        assert_eq!(
            random.borrow_mut().next_int(),
            JavaRandom::new(42).next_int()
        );
        let inline = name.len() <= INLINE_NAME_BYTES;
        assert_eq!(matches!(&codec.name, CodecName::Inline { .. }), inline);
        assert_eq!(codec.name_heap_bytes(), if inline { 0 } else { name.len() });
        assert_eq!(codec.name(), name);
        assert_eq!(Codec::name(&codec), name);
        codec.delegate().name.set("changed after construction");
        codec.delegate().display.set("second");
        assert_eq!(codec.name(), name);
        assert_eq!(Codec::name(&codec), name);
        assert_eq!(format!("{codec}"), "Cranky(second)");
        assert_eq!(&*codec.delegate().calls.borrow(), &["name", "display"]);
    }
}
#[test]
fn every_factory_failure_forwards_without_drawing_or_cloning_random() {
    let codec = codec();
    let mut control = JavaRandom::new(42);
    let random = codec.random().clone();
    let handles = Rc::strong_count(&random);
    assert_eq!(codec.doc_values_format().err(), Some("doc_values_format"));
    assert_eq!(Rc::strong_count(&random), handles);
    assert_eq!(random.borrow_mut().next_int(), control.next_int());
    assert_eq!(codec.field_infos_format().err(), Some("field_infos_format"));
    assert_eq!(Rc::strong_count(&random), handles);
    assert_eq!(random.borrow_mut().next_int(), control.next_int());
    assert_eq!(codec.live_docs_format().err(), Some("live_docs_format"));
    assert_eq!(Rc::strong_count(&random), handles);
    assert_eq!(random.borrow_mut().next_int(), control.next_int());
    assert_eq!(codec.norms_format().err(), Some("norms_format"));
    assert_eq!(Rc::strong_count(&random), handles);
    assert_eq!(random.borrow_mut().next_int(), control.next_int());
    assert_eq!(codec.postings_format().err(), Some("postings_format"));
    assert_eq!(Rc::strong_count(&random), handles);
    assert_eq!(random.borrow_mut().next_int(), control.next_int());
    assert_eq!(
        codec.segment_info_format().err(),
        Some("segment_info_format")
    );
    assert_eq!(Rc::strong_count(&random), handles);
    assert_eq!(random.borrow_mut().next_int(), control.next_int());
    assert_eq!(
        codec.stored_fields_format().err(),
        Some("stored_fields_format")
    );
    assert_eq!(Rc::strong_count(&random), handles);
    assert_eq!(random.borrow_mut().next_int(), control.next_int());
    assert_eq!(
        codec.term_vectors_format().err(),
        Some("term_vectors_format")
    );
    assert_eq!(Rc::strong_count(&random), handles);
    assert_eq!(random.borrow_mut().next_int(), control.next_int());
    assert_eq!(codec.compound_format().err(), Some("compound_format"));
    assert_eq!(Rc::strong_count(&random), handles);
    assert_eq!(random.borrow_mut().next_int(), control.next_int());
    assert_eq!(codec.points_format().err(), Some("points_format"));
    assert_eq!(Rc::strong_count(&random), handles);
    assert_eq!(random.borrow_mut().next_int(), control.next_int());
    assert_eq!(codec.knn_vectors_format().err(), Some("knn_vectors_format"));
    assert_eq!(Rc::strong_count(&random), handles);
    assert_eq!(random.borrow_mut().next_int(), control.next_int());
    assert_eq!(codec.delegate().calls.borrow().len(), 12);
}
