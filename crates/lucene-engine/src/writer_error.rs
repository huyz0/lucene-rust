//! Classification of [`index_writer::Error`] into a [`Status`].

use lucene_index::index_writer;

use crate::status::{set_last_error, Status};

/// Maps a writer error to a status. What Java raises as
/// `IllegalArgumentException` (caller misuse, a refused document, an add past
/// `maxDocs`, an inconsistent configuration) is [`Status::InvalidArgument`]
/// and fails only the operation; I/O and corrupt-index errors are
/// [`Status::Io`] and tragic. Every variant is listed, so a new one must be
/// classified here.
pub fn writer_error(e: index_writer::Error) -> Status {
    let status = match &e {
        index_writer::Error::NoPreparedCommit
        | index_writer::Error::PrepareCommitAlreadyCalled
        | index_writer::Error::PreparedCommitPending(_)
        | index_writer::Error::ConcurrentUnsupported(_)
        | index_writer::Error::Explicit(_)
        | index_writer::Error::LiveSearch(_)
        | index_writer::Error::Document(_)
        | index_writer::Error::ZeroNorm(_)
        | index_writer::Error::MergeHook(_)
        | index_writer::Error::InvalidMaxNumSegments(_)
        | index_writer::Error::UnknownPostingsField(_)
        | index_writer::Error::UnsupportedPostingsIndexOptions(_, _)
        | index_writer::Error::FieldInfos(_)
        | index_writer::Error::NoPayloadFields
        | index_writer::Error::UnknownTermVectorField(_)
        | index_writer::Error::UnsupportedTermVectorField(_)
        | index_writer::Error::UnknownDocValuesField(_)
        | index_writer::Error::UnsupportedDocValuesType(_, _)
        | index_writer::Error::MissingDenseDocValue(_, _)
        | index_writer::Error::NonNumericDocValue(_, _, _)
        | index_writer::Error::NonBinaryDocValue(_, _, _)
        | index_writer::Error::UnknownVectorField(_)
        | index_writer::Error::UnknownPointsField(_)
        | index_writer::Error::UnsupportedPointsField(_, _, _, _)
        | index_writer::Error::DuplicatePointsField(_)
        | index_writer::Error::PointValueShape(_, _, _, _, _)
        | index_writer::Error::UnsupportedVectorField(_, _)
        | index_writer::Error::DuplicateVectorField(_)
        | index_writer::Error::VectorDimensionMismatch(_, _, _, _)
        | index_writer::Error::VectorEncodingMismatch(_, _, _, _)
        | index_writer::Error::DuplicatePostingsField(_)
        | index_writer::Error::UnknownCustomFreqPostingsField(_)
        | index_writer::Error::UnsupportedCustomFreqPostingsIndexOptions(_, _)
        | index_writer::Error::PostingsAndCustomFreqPostingsMutuallyExclusive(_)
        | index_writer::Error::DuplicateTermVectorField(_)
        | index_writer::Error::UnknownNormsField(_)
        | index_writer::Error::UnsupportedNormsField(_)
        | index_writer::Error::InvalidRamBufferSize(_)
        | index_writer::Error::InvalidMaxBufferedDocs(_)
        | index_writer::Error::BothAutoFlushTriggersDisabled
        | index_writer::Error::NoSoftDeletesSupplied
        | index_writer::Error::TooManyDocs(_)
        | index_writer::Error::NoDocValuesUpdatesSupplied
        | index_writer::Error::UnknownDocValuesUpdateField(_)
        | index_writer::Error::WrongDocValuesUpdateType { .. }
        | index_writer::Error::EmptyIndexSort
        | index_writer::Error::UnknownIndexSortField(_)
        | index_writer::Error::UnsupportedIndexSortField(_, _)
        | index_writer::Error::IndexSortFieldWithoutDocValues(_)
        | index_writer::Error::IncongruentIndexSort { .. }
        | index_writer::Error::IndexSortChangedMidBuffer(_)
        | index_writer::Error::IndexSortWithBlocksAndNoParentField
        | index_writer::Error::BlocksWithIndexSortNeedParentField
        | index_writer::Error::DocValuesUpdateOnIndexSortField { .. }
        | index_writer::Error::DuplicateDocValuesField(_)
        | index_writer::Error::MergeSortDisagreement { .. }
        | index_writer::Error::InvalidMergeRate(_)
        | index_writer::Error::InvalidMergeScheduler(_)
        | index_writer::Error::AddIndexes(_) => Status::InvalidArgument,
        // I/O, decode and corrupt-index failures; `MergeAbortedException` is an
        // `IOException` in Java.
        index_writer::Error::Store(_)
        | index_writer::Error::MergeThreadPanicked(_)
        | index_writer::Error::SegmentWriter(_)
        | index_writer::Error::SegmentInfos(_)
        | index_writer::Error::UpdateDocument(_)
        | index_writer::Error::TermDelete(_)
        | index_writer::Error::Deletes(_)
        | index_writer::Error::FieldUpdates(_)
        | index_writer::Error::Deleter(_)
        | index_writer::Error::Merge(_)
        | index_writer::Error::SegmentInfo(_)
        | index_writer::Error::SimpleText(_)
        | index_writer::Error::StoredFields(_)
        | index_writer::Error::LiveDocs(_)
        | index_writer::Error::PostingsWriter(_)
        | index_writer::Error::Blocktree(_)
        | index_writer::Error::Postings(_)
        | index_writer::Error::TermVectors(_)
        | index_writer::Error::DocValues(_)
        | index_writer::Error::Norms(_)
        | index_writer::Error::Vectors(_)
        | index_writer::Error::Ivf(_)
        | index_writer::Error::Points(_)
        | index_writer::Error::DocValuesRead(_)
        | index_writer::Error::NormsRead(_)
        | index_writer::Error::MergeSortColumnMissing(_)
        | index_writer::Error::SegmentDocCountMismatch { .. }
        | index_writer::Error::MergeAborted
        | index_writer::Error::UnreadableSegmentPostings { .. } => Status::Io,
    };
    set_last_error(format!("engine writer: {e}"));
    status
}
