//! Readers whose live docs the host supplies (deletions held in memory).

use std::sync::Arc;

use lucene_util::fixed_bit_set::FixedBitSet;

use super::{DirectoryReader, SegmentReader};
use crate::{Error, Result};

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

impl DirectoryReader {
    /// This reader with segment `i` showing `live[i]` as its live docs (`None`:
    /// no deletions, whatever the segment's files say), under a new cache key:
    /// what a search sees when the host holds the deletions in memory.
    ///
    /// # Errors
    /// A `live` of another length than the reader's segment count.
    pub fn with_live_docs(&self, live: &[Option<FixedBitSet>]) -> Result<Self> {
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
        Ok(DirectoryReader {
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
        })
    }
}
