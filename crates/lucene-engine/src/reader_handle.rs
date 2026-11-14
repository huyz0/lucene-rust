//! The reader a JVM reader handle stands for: a `DirectoryReader` over the
//! JVM reader's segments, the JVM's live-docs masks, and the per-reader
//! caches searches share.

use std::sync::Arc;

use lucene_index::buffered_updates::Term;
use lucene_util::fixed_bit_set::FixedBitSet;

/// Each segment's delete mask: `None` for a segment with no deletions.
pub type Masks = Vec<Option<Arc<FixedBitSet>>>;

/// A reader's [`Masks`] after the first `applied` of the delete terms its
/// views list: every view of one commit lists the same terms first, so the
/// next search applies only the new ones. `last` is the last of them, to
/// tell a list that does not extend these.
pub struct CommitLive {
    pub applied: usize,
    pub last: Term,
    pub masks: Arc<Masks>,
}

/// A `DirectoryReader` opened over a JVM reader's segments, plus that JVM
/// reader's live docs: `live_docs[i]` is segment `i`'s
/// mask, `None` meaning no deletions; it **replaces** whatever the segment
/// reader decoded from `.liv`, because an NRT reader's deletions live in the
/// JVM's memory, not on disk.
pub struct JvmReaderHandle {
    pub reader: lucene_search::directory_reader::DirectoryReader,
    pub live_docs: Vec<Option<FixedBitSet>>,
    /// Deleted documents per segment (`maxDoc - live_docs.cardinality()`),
    /// kept alongside the mask so total-hits lower bounds need not recount it
    /// per query.
    pub deleted: Vec<i64>,
    /// `live_docs` with the deletes issued since the reader's commit
    /// applied, as far as the last live search read them
    /// ([`CommitLive`]).
    pub commit_live: std::sync::Mutex<Option<CommitLive>>,
    /// How many of the reader's segments, from the first, the aggregation
    /// segment cache may keep parts of: the committed ones. A realtime view's
    /// buffers, appended after them, change with every view.
    pub cacheable: usize,
    /// Each segment's identity in the aggregation segment cache, found on
    /// first use: `None` past [`Self::cacheable`].
    pub agg_identities:
        std::sync::OnceLock<Vec<Option<lucene_aggs::bucket_aggs::segment_cache::SegmentIdentity>>>,
    /// The shard path the reader's files are under, noted for each call
    /// that reads them .
    pub path: std::sync::Arc<str>,
}

impl JvmReaderHandle {
    /// [`Self::agg_identities`], found now when no search has yet.
    pub fn agg_identities(
        &self,
    ) -> &[Option<lucene_aggs::bucket_aggs::segment_cache::SegmentIdentity>] {
        self.agg_identities.get_or_init(|| {
            self.reader
                .segment_readers()
                .iter()
                .zip(&self.live_docs)
                .enumerate()
                .map(|(i, (r, live))| {
                    (i < self.cacheable).then(|| {
                        lucene_aggs::bucket_aggs::segment_cache::SegmentIdentity::of(
                            r,
                            live.as_ref(),
                        )
                    })
                })
                .collect()
        })
    }
}
