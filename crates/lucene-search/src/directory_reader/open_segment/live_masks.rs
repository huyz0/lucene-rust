//! A reader's segments under replacement live docs.

use std::sync::Arc;

use lucene_util::fixed_bit_set::FixedBitSet;

use crate::directory_reader::{DirectoryReader, Error, Result, SegmentReader};

impl DirectoryReader {
    /// This reader with segment `i`'s live docs replaced by `masks[i]` (`None` keeps the
    /// segment's own), under a new cache key: what a search sees when a newer copy of a
    /// document sits in a later segment. Segments and doc bases are the reader's own.
    ///
    /// # Errors
    /// A `masks` of another length than the reader's segment count, or a mask whose length
    /// is not its segment's `max_doc`.
    pub fn with_live_docs_masks(&self, masks: Vec<Option<FixedBitSet>>) -> Result<Self> {
        if masks.len() != self.segments.len() {
            return Err(Error::Store(lucene_store::Error::Corrupted(format!(
                "{} live-docs masks for {} segments",
                masks.len(),
                self.segments.len()
            ))));
        }
        let mut segments = Vec::with_capacity(self.segments.len());
        for (segment, mask) in self.segments.iter().zip(masks) {
            segments.push(match mask {
                None => segment.clone_reader(),
                Some(m) if usize::try_from(segment.max_doc).ok() == Some(m.len()) => {
                    segment.with_live_docs(m)
                }
                Some(m) => {
                    return Err(Error::Store(lucene_store::Error::Corrupted(format!(
                        "a live-docs mask of {} bits for a segment of {} documents",
                        m.len(),
                        segment.max_doc
                    ))));
                }
            });
        }
        Ok(self.with_replaced_segments(segments))
    }

    /// This reader with segment `i` showing `live[i]` (`None`: every document live, whatever the
    /// segment's files say), under a new cache key: what a search sees when the host holds the
    /// deletions in memory.
    ///
    /// # Errors
    /// A `live` of another length than the reader's segment count.
    pub fn with_host_live_docs(&self, live: &[Option<FixedBitSet>]) -> Result<Self> {
        if live.len() != self.segments.len() {
            return Err(Error::Store(lucene_store::Error::Corrupted(format!(
                "{} live-docs masks for {} segments",
                live.len(),
                self.segments.len()
            ))));
        }
        let segments = self
            .segments
            .iter()
            .zip(live)
            .map(|(segment, mask)| match mask {
                Some(mask) => segment.with_live_docs(mask.clone()),
                None => segment.without_live_docs(),
            })
            .collect();
        Ok(self.with_replaced_segments(segments))
    }

    fn with_replaced_segments(&self, segments: Vec<SegmentReader>) -> Self {
        DirectoryReader {
            segment_infos: self.segment_infos.clone(),
            segments,
            global_ords: std::sync::Mutex::default(),
            appended_ords: Arc::default(),
            ords_cache: None,
            norm_tables: std::sync::Mutex::default(),
            norms_plans: std::sync::Mutex::default(),
            nrt_hold: self.nrt_hold.clone(),
            reader_cache: crate::reader::CacheHelper::new(),
            soft_deletes: self.soft_deletes.clone(),
        }
    }
}

impl SegmentReader {
    /// This segment with every document live.
    pub(crate) fn without_live_docs(&self) -> Self {
        let mut reader = self.clone_reader();
        reader.num_docs = self.max_doc;
        reader.live_docs = None;
        reader.reader_cache = crate::reader::CacheHelper::new();
        reader
    }
}
