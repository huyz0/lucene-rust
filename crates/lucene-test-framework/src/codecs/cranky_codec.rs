//! Java's `CrankyCodec`: ten fresh format wrappers share one random stream;
//! the delegate's KNN format passes through unchanged.
//!
//! Each accessor first calls the corresponding delegate factory exactly once,
//! then constructs its cranky wrapper without drawing randomness. The codec
//! snapshots the delegate name at construction (inline through 15 UTF-8 bytes,
//! boxed otherwise) and formats the current delegate
//! display on every call, as `CrankyCodec` and `FilterCodec` do.
//!
//! Rust-forced differences: a typed factory trait replaces the JVM codec class
//! and SPI; owned format handles replace Java object references, factory failures
//! are `Result`, and `Rc<RefCell<JavaRandom>>` is single-threaded. Delegates that
//! share formats return cloneable handles implementing the relevant format
//! trait. An accessor preserves the exact handle returned by its factory.
//! No default native codec graph or `IndexWriter` routing is supplied. In
//! particular, the existing SimpleText term-vector seam does not stand in for
//! Java's default Lucene90 term vectors or its per-field DV/postings formats.

use std::{cell::RefCell, fmt, rc::Rc};

use super::cranky_compound::{CompoundFormat, CrankyCompoundFormat};
use super::cranky_doc_values::{CrankyDocValuesFormat, DocValuesFormat};
use super::cranky_field_infos::{CrankyFieldInfosFormat, FieldInfosFormat};
use super::cranky_live_docs::{CrankyLiveDocsFormat, LiveDocsFormat};
use super::cranky_norms::{CrankyNormsFormat, NormsFormat};
use super::cranky_points::{CrankyPointsFormat, PointsFormat};
use super::cranky_postings::{CrankyPostingsFormat, PostingsFormat};
use super::cranky_segment_info::{CrankySegmentInfoFormat, SegmentInfoFormatDelegate};
use super::cranky_stored_fields::{CrankyStoredFieldsFormat, StoredFieldsFormat};
use super::cranky_term_vectors::{CrankyTermVectorsFormat, TermVectorsFormat};
use lucene_util::java_random::JavaRandom;

/// The eleven component factories of Java's `Codec`, with their exact types.
///
/// Java's final codec name is immutable. Return owned component handles so two
/// simultaneously live wrappers can retain the same underlying component.
/// `KnnVectors` is opaque because this adapter forwards it without operations.
pub trait Codec: fmt::Display {
    type Error;
    type DocValues: DocValuesFormat;
    type FieldInfos: FieldInfosFormat;
    type LiveDocs: LiveDocsFormat;
    type Norms: NormsFormat;
    type Postings: PostingsFormat;
    type SegmentInfo: SegmentInfoFormatDelegate;
    type StoredFields: StoredFieldsFormat;
    type TermVectors: TermVectorsFormat;
    type Compound: CompoundFormat;
    type Points: PointsFormat;
    type KnnVectors;

    /// Java: `Codec.getName`; this name must stay valid for the codec's life.
    fn name(&self) -> &str;
    /// Java: `Codec.docValuesFormat`.
    fn doc_values_format(&self) -> Result<Self::DocValues, Self::Error>;
    /// Java: `Codec.fieldInfosFormat`.
    fn field_infos_format(&self) -> Result<Self::FieldInfos, Self::Error>;
    /// Java: `Codec.liveDocsFormat`.
    fn live_docs_format(&self) -> Result<Self::LiveDocs, Self::Error>;
    /// Java: `Codec.normsFormat`.
    fn norms_format(&self) -> Result<Self::Norms, Self::Error>;
    /// Java: `Codec.postingsFormat`.
    fn postings_format(&self) -> Result<Self::Postings, Self::Error>;
    /// Java: `Codec.segmentInfoFormat`.
    fn segment_info_format(&self) -> Result<Self::SegmentInfo, Self::Error>;
    /// Java: `Codec.storedFieldsFormat`.
    fn stored_fields_format(&self) -> Result<Self::StoredFields, Self::Error>;
    /// Java: `Codec.termVectorsFormat`.
    fn term_vectors_format(&self) -> Result<Self::TermVectors, Self::Error>;
    /// Java: `Codec.compoundFormat`.
    fn compound_format(&self) -> Result<Self::Compound, Self::Error>;
    /// Java: `Codec.pointsFormat`.
    fn points_format(&self) -> Result<Self::Points, Self::Error>;
    /// Java: `Codec.knnVectorsFormat`.
    fn knn_vectors_format(&self) -> Result<Self::KnnVectors, Self::Error>;
}

// The measured short-name payload allocation exceeded Java's shared-name state.
// Store complete short UTF-8 strings in the enum; retain an owned snapshot for
// longer names without adding a lifetime or a shared-name requirement to Codec.
const INLINE_NAME_BYTES: usize = 15;
enum CodecName {
    Inline {
        bytes: [u8; INLINE_NAME_BYTES],
        len: u8,
    },
    Heap(Box<str>),
}
impl CodecName {
    fn new(value: &str) -> Self {
        if value.len() <= INLINE_NAME_BYTES {
            let mut bytes = [0; INLINE_NAME_BYTES];
            bytes[..value.len()].copy_from_slice(value.as_bytes());
            Self::Inline {
                bytes,
                len: value.len() as u8,
            }
        } else {
            Self::Heap(Box::from(value))
        }
    }
    fn as_str(&self) -> &str {
        match self {
            Self::Inline { bytes, len } => std::str::from_utf8(&bytes[..usize::from(*len)])
                .expect("the complete bytes were copied from a valid str"),
            Self::Heap(value) => value,
        }
    }
    fn heap_bytes(&self) -> usize {
        match self {
            Self::Inline { .. } => 0,
            Self::Heap(value) => value.len(),
        }
    }
}

/// Java: `CrankyCodec`, retaining its delegate and the supplied random handle.
pub struct CrankyCodec<D> {
    name: CodecName,
    delegate: D,
    random: Rc<RefCell<JavaRandom>>,
}
impl<D: Codec> CrankyCodec<D> {
    /// Java: `CrankyCodec(delegate, random)`; no component factories or RNG draws.
    pub fn new(delegate: D, random: Rc<RefCell<JavaRandom>>) -> Self {
        let name = CodecName::new(delegate.name());
        Self {
            name,
            delegate,
            random,
        }
    }

    /// Java: `Codec.getName`, captured before the delegate is retained.
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Heap payload bytes owned by the codec's immutable name snapshot.
    ///
    /// Short names occupy inline enum storage and return zero. Longer names
    /// own exactly their UTF-8 byte length in a boxed str. This excludes the
    /// enum's inline state, allocator metadata and the delegate's name storage.
    pub fn name_heap_bytes(&self) -> usize {
        self.name.heap_bytes()
    }

    /// The retained delegate, including any shared identity carried by its type.
    pub fn delegate(&self) -> &D {
        &self.delegate
    }

    /// The exact random handle shared by all ten component wrappers.
    pub fn random(&self) -> &Rc<RefCell<JavaRandom>> {
        &self.random
    }
}
impl<D: Codec> Codec for CrankyCodec<D> {
    type Error = D::Error;
    type DocValues = CrankyDocValuesFormat<D::DocValues>;
    type FieldInfos = CrankyFieldInfosFormat<D::FieldInfos>;
    type LiveDocs = CrankyLiveDocsFormat<D::LiveDocs>;
    type Norms = CrankyNormsFormat<D::Norms>;
    type Postings = CrankyPostingsFormat<D::Postings>;
    type SegmentInfo = CrankySegmentInfoFormat<D::SegmentInfo>;
    type StoredFields = CrankyStoredFieldsFormat<D::StoredFields>;
    type TermVectors = CrankyTermVectorsFormat<D::TermVectors>;
    type Compound = CrankyCompoundFormat<D::Compound>;
    type Points = CrankyPointsFormat<D::Points>;
    type KnnVectors = D::KnnVectors;

    fn name(&self) -> &str {
        self.name.as_str()
    }
    /// Java: `CrankyCodec.docValuesFormat`.
    fn doc_values_format(&self) -> Result<Self::DocValues, Self::Error> {
        let delegate = self.delegate.doc_values_format()?;
        Ok(CrankyDocValuesFormat::new(delegate, self.random.clone()))
    }
    /// Java: `CrankyCodec.fieldInfosFormat`.
    fn field_infos_format(&self) -> Result<Self::FieldInfos, Self::Error> {
        let delegate = self.delegate.field_infos_format()?;
        Ok(CrankyFieldInfosFormat::new(delegate, self.random.clone()))
    }
    /// Java: `CrankyCodec.liveDocsFormat`.
    fn live_docs_format(&self) -> Result<Self::LiveDocs, Self::Error> {
        let delegate = self.delegate.live_docs_format()?;
        Ok(CrankyLiveDocsFormat::new(delegate, self.random.clone()))
    }
    /// Java: `CrankyCodec.normsFormat`.
    fn norms_format(&self) -> Result<Self::Norms, Self::Error> {
        let delegate = self.delegate.norms_format()?;
        Ok(CrankyNormsFormat::new(delegate, self.random.clone()))
    }
    /// Java: `CrankyCodec.postingsFormat`.
    fn postings_format(&self) -> Result<Self::Postings, Self::Error> {
        let delegate = self.delegate.postings_format()?;
        Ok(CrankyPostingsFormat::new(delegate, self.random.clone()))
    }
    /// Java: `CrankyCodec.segmentInfoFormat`.
    fn segment_info_format(&self) -> Result<Self::SegmentInfo, Self::Error> {
        let delegate = self.delegate.segment_info_format()?;
        Ok(CrankySegmentInfoFormat::new(delegate, self.random.clone()))
    }
    /// Java: `CrankyCodec.storedFieldsFormat`.
    fn stored_fields_format(&self) -> Result<Self::StoredFields, Self::Error> {
        let delegate = self.delegate.stored_fields_format()?;
        Ok(CrankyStoredFieldsFormat::new(delegate, self.random.clone()))
    }
    /// Java: `CrankyCodec.termVectorsFormat`.
    fn term_vectors_format(&self) -> Result<Self::TermVectors, Self::Error> {
        let delegate = self.delegate.term_vectors_format()?;
        Ok(CrankyTermVectorsFormat::new(delegate, self.random.clone()))
    }
    /// Java: `CrankyCodec.compoundFormat`.
    fn compound_format(&self) -> Result<Self::Compound, Self::Error> {
        let delegate = self.delegate.compound_format()?;
        Ok(CrankyCompoundFormat::new(delegate, self.random.clone()))
    }
    /// Java: `CrankyCodec.pointsFormat`.
    fn points_format(&self) -> Result<Self::Points, Self::Error> {
        let delegate = self.delegate.points_format()?;
        Ok(CrankyPointsFormat::new(delegate, self.random.clone()))
    }

    /// Java: inherited `FilterCodec.knnVectorsFormat`; no wrapping or RNG clone.
    fn knn_vectors_format(&self) -> Result<Self::KnnVectors, Self::Error> {
        self.delegate.knn_vectors_format()
    }
}
impl<D: fmt::Display> fmt::Display for CrankyCodec<D> {
    /// Java: `CrankyCodec.toString`; delegate display is evaluated on every call.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cranky({})", self.delegate)
    }
}

#[cfg(test)]
mod tests {
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
}
