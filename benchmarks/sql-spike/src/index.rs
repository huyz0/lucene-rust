//! One opened index, leaked for the life of the process so the DataFusion
//! provider can hold `'static` references (this is a spike, not a library).

use std::collections::HashMap;
use std::sync::Arc;

use lucene_search::directory_reader::{DirectoryReader, SegmentReader};
use lucene_search::global_ords::GlobalOrds;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::multi_segment::OpenSegment;
use lucene_store::MmapDirectory;

/// Field numbers of one segment (they are assigned per segment).
#[derive(Clone, Copy, Debug)]
pub struct SegFields {
    pub id: i32,
    pub price: i32,
    pub cat: i32,
}

pub struct SpikeIndex {
    pub reader: &'static DirectoryReader,
    pub segments: &'static [OpenSegment<'static>],
    pub searcher: IndexSearcher<'static, 'static>,
    pub fields: Vec<SegFields>,
    /// Every term of `cat`, by segment, in the segment's ordinal order.
    pub cat_terms: Vec<Vec<String>>,
    pub cat_global: Arc<GlobalOrds>,
}

// The spike shares one index between threads: every structure behind it is
// immutable after open.
unsafe impl Sync for SpikeIndex {}
unsafe impl Send for SpikeIndex {}

impl SpikeIndex {
    pub fn open(path: &str) -> &'static SpikeIndex {
        let dir: &'static MmapDirectory =
            Box::leak(Box::new(MmapDirectory::open(path.to_string())));
        let reader: &'static DirectoryReader =
            Box::leak(Box::new(DirectoryReader::open(dir).expect("open index")));
        let opened = Box::leak(Box::new(reader.open_segments().expect("segments")));
        let segments: &'static [OpenSegment<'static>] =
            Box::leak(opened.as_open_segments().into_boxed_slice());
        let owned = Box::leak(Box::new(
            reader
                .field_norms_by_field(&["body".to_string()])
                .expect("norms"),
        ));
        let norms: &'static [SegmentNorms<'static, 'static>] = Box::leak(
            owned
                .iter()
                .map(Some)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        let searcher = IndexSearcher::new(segments, norms).expect("searcher");
        let num = |r: &SegmentReader, name: &str| {
            r.field_infos()
                .field_by_name(name)
                .unwrap_or_else(|| panic!("field {name}"))
                .number
        };
        let fields = reader
            .segment_readers()
            .iter()
            .map(|r| SegFields {
                id: num(r, "id"),
                price: num(r, "price"),
                cat: num(r, "cat"),
            })
            .collect::<Vec<_>>();
        let cat_terms = reader
            .segment_readers()
            .iter()
            .map(|r| {
                let (_, data, entry) = lucene_search::global_ords::keyword_column(r, "cat")
                    .unwrap()
                    .expect("cat column");
                lucene_codecs::terms_dict::decode_all_terms(data, entry)
                    .unwrap()
                    .into_iter()
                    .map(|t| String::from_utf8(t).unwrap())
                    .collect()
            })
            .collect();
        let cat_global = reader.global_ords("cat").expect("global ords");
        Box::leak(Box::new(SpikeIndex {
            reader,
            segments,
            searcher,
            fields,
            cat_terms,
            cat_global,
        }))
    }

    pub fn leaves(&self) -> &[SegmentReader] {
        self.reader.segment_readers()
    }

    pub fn max_doc(&self) -> i32 {
        self.reader.max_doc()
    }

    /// `(segment, segment-local doc)` of a global doc id.
    pub fn locate(&self, doc: i32) -> (usize, i32) {
        let s = self.searcher.segment_of(doc).expect("doc in range");
        (s, doc - self.segments[s].doc_base)
    }

    /// The `id` stored field of a segment-local document.
    pub fn stored_id(&self, seg: usize, local: i32) -> String {
        let mut v =
            lucene_codecs::stored_fields::DocumentVisitor::for_fields(&[self.fields[seg].id]);
        let _ = self.leaves()[seg]
            .visit_stored_document(local, &mut v)
            .expect("stored");
        match &v.document().fields[0].value {
            lucene_codecs::stored_fields::FieldValue::String(s) => s.clone(),
            other => panic!("id is {other:?}"),
        }
    }
}
