//! A reader's segments under replacement live docs.

use std::sync::Arc;

use lucene_util::fixed_bit_set::FixedBitSet;

use crate::directory_reader::{DirectoryReader, Error, Result};

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
