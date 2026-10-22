//! Ports of `index/SegmentWriteState` and `index/SegmentReadState`, as far
//! as a postings format that is not `Lucene104PostingsFormat` needs them:
//! the segment's id, `maxDoc` and `FieldInfos`, and the codec suffix of the
//! format instance (`PerFieldPostingsFormat` hands each group its own,
//! `BloomFilter_0`). Files are passed in and returned as bytes, so neither
//! carries Java's `Directory`, segment name or `IOContext`.
//!
//! The carrier for the M14.g postings formats that wrap or replace the
//! default one; the default codec's writers predate it and still take the
//! id and suffix as loose arguments.

use lucene_store::codec_util::ID_LENGTH;

use crate::field_infos::{FieldInfo, FieldInfos};
use crate::postings_writer::WriteOptions;

/// `SegmentWriteState`: what a `FieldsConsumer` is constructed with.
#[derive(Debug, Clone, Copy)]
pub struct SegmentWriteState<'a> {
    /// `segmentInfo.maxDoc()`.
    pub max_doc: i32,
    /// `segmentInfo.getId()`.
    pub segment_id: &'a [u8; ID_LENGTH],
    /// `segmentSuffix`.
    pub segment_suffix: &'a str,
    /// `fieldInfos`: the whole segment's, not only the fields this format
    /// was handed.
    pub field_infos: &'a FieldInfos,
}

impl SegmentWriteState<'_> {
    /// `fieldInfos.fieldInfo(number)`.
    pub fn field_info(&self, number: i32) -> Option<&FieldInfo> {
        self.field_infos.field_by_number(number)
    }

    /// `fieldInfos.hasProx()`: some field of the segment indexes positions.
    pub fn has_prox(&self) -> bool {
        crate::field_infos::has_prox(&self.field_infos.fields)
    }

    /// `fieldInfos.hasPayloads() || fieldInfos.hasOffsets()`.
    pub fn has_payloads_or_offsets(&self) -> bool {
        crate::field_infos::has_payloads_or_offsets(&self.field_infos.fields)
    }

    /// `options` with `.pos`/`.pay` presence decided over the segment, as
    /// `Lucene104PostingsWriter`'s constructor decides it from
    /// `state.fieldInfos`.
    pub fn postings_options(&self, options: WriteOptions) -> WriteOptions {
        WriteOptions {
            has_prox: Some(self.has_prox()),
            has_payloads_or_offsets: Some(self.has_payloads_or_offsets()),
            ..options
        }
    }
}

/// `SegmentReadState`: what a `FieldsProducer` is constructed with.
#[derive(Debug, Clone, Copy)]
pub struct SegmentReadState<'a> {
    /// `segmentInfo.maxDoc()`.
    pub max_doc: i32,
    /// `segmentInfo.getId()`.
    pub segment_id: &'a [u8; ID_LENGTH],
    /// `segmentSuffix`.
    pub segment_suffix: &'a str,
    /// `fieldInfos`.
    pub field_infos: &'a FieldInfos,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_infos::IndexOptions;

    fn infos(fields: Vec<FieldInfo>) -> FieldInfos {
        FieldInfos::new(fields).unwrap()
    }

    #[test]
    fn prox_and_payload_flags_come_from_the_whole_segment() {
        let id = [7u8; ID_LENGTH];
        let docs_only = infos(vec![
            FieldInfo::new("a", 0).with_index_options(IndexOptions::Docs)
        ]);
        let state = SegmentWriteState {
            max_doc: 3,
            segment_id: &id,
            segment_suffix: "X_0",
            field_infos: &docs_only,
        };
        assert!(!state.has_prox());
        assert!(!state.has_payloads_or_offsets());
        let o = state.postings_options(WriteOptions::default());
        assert_eq!(o.has_prox, Some(false));
        assert_eq!(o.has_payloads_or_offsets, Some(false));
        assert_eq!(state.field_info(0).unwrap().name, "a");
        assert!(state.field_info(1).is_none());

        let payloads = infos(vec![
            FieldInfo::new("a", 0).with_index_options(IndexOptions::Docs),
            FieldInfo::new("b", 1)
                .with_index_options(IndexOptions::DocsAndFreqsAndPositions)
                .with_store_payloads(true),
        ]);
        let state = SegmentWriteState {
            field_infos: &payloads,
            ..state
        };
        assert!(state.has_prox());
        assert!(state.has_payloads_or_offsets());

        let offsets = infos(vec![FieldInfo::new("c", 0)
            .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)]);
        let state = SegmentWriteState {
            field_infos: &offsets,
            ..state
        };
        assert!(state.has_payloads_or_offsets());
        let read = SegmentReadState {
            max_doc: 3,
            segment_id: &id,
            segment_suffix: "X_0",
            field_infos: &offsets,
        };
        assert_eq!(read.field_infos.fields.len(), 1);
    }
}
