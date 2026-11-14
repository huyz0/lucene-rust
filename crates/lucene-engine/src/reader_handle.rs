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

/// A reader over the segments listed in `infos` (a whole `segments_N`
/// file's bytes at generation `generation`) under `dir`, with segment `i`
/// checked to have exactly `expected[i]` documents and masked with the
/// JVM's live docs: `counts[i]` is how many words of segment `i`'s live-docs
/// bitset follow in `words` (the segments' words concatenated, in order);
/// `0` means no deletions, any other count exactly `bits2words(maxDoc)`
/// with no bit set past `maxDoc`. `counts` is `None` when no segment has
/// deletions. They replace whatever the segment has on disk.
///
/// `previous` is the handle of an earlier call over the same directory:
/// its unchanged segments are shared instead of re-read.
#[allow(clippy::too_many_arguments)]
pub fn open_reader(
    path: &str,
    dir: &dyn lucene_store::Directory,
    infos: &[u8],
    generation: i64,
    previous: Option<&JvmReaderHandle>,
    expected: &[i32],
    counts: Option<&[usize]>,
    words: &[u64],
) -> Result<JvmReaderHandle, crate::status::Status> {
    use crate::alloc::try_with_capacity;
    use crate::status::{set_last_error, Status};
    let segment_infos = lucene_index::segment_infos::parse(infos, generation).map_err(|e| {
        set_last_error(format!("parsing the JVM reader's SegmentInfos: {e}"));
        Status::Decode
    })?;
    let reader = match previous {
        None => lucene_search::directory_reader::DirectoryReader::open_at(dir, segment_infos),
        Some(prev) => prev.reader.reopen_at(dir, segment_infos),
    }
    .map_err(|e| {
        set_last_error(format!("opening the JVM reader's segments: {e}"));
        Status::Decode
    })?;

    let opened: Vec<i32> = reader.segment_readers().iter().map(|s| s.max_doc).collect();
    if opened != expected {
        set_last_error(format!(
            "segment maxDocs {opened:?} do not match the JVM reader's {expected:?}"
        ));
        return Err(Status::InvalidArgument);
    }
    let mut live_docs = try_with_capacity(opened.len())?;
    let mut deleted = try_with_capacity(opened.len())?;
    let mut at = 0usize;
    for (segment, &max_doc) in opened.iter().enumerate() {
        let n = counts.map_or(0, |c| c[segment]);
        let live = live_from_words(segment, max_doc, &words[at..at + n])?;
        at += n;
        deleted.push(live.as_ref().map_or(0, |l| {
            (usize::try_from(max_doc).unwrap_or(0) - l.cardinality()) as i64
        }));
        live_docs.push(live);
    }
    let cacheable = live_docs.len();
    Ok(JvmReaderHandle {
        reader,
        live_docs,
        deleted,
        commit_live: std::sync::Mutex::new(None),
        cacheable,
        agg_identities: std::sync::OnceLock::new(),
        path: Arc::from(path),
    })
}

/// Segment `segment`'s live docs from its `words` ([`open_reader`]'s
/// contract): `None` for no words, else exactly `bits2words(max_doc)` words
/// with no bit past `max_doc`.
fn live_from_words(
    segment: usize,
    max_doc: i32,
    words: &[u64],
) -> Result<Option<FixedBitSet>, crate::status::Status> {
    use crate::status::{set_last_error, Status};
    use lucene_util::fixed_bit_set::bits2words;
    if words.is_empty() {
        return Ok(None);
    }
    let max_doc = usize::try_from(max_doc).unwrap_or(0);
    if words.len() != bits2words(max_doc) {
        set_last_error(format!(
            "segment {segment}: {} live-docs words for maxDoc {max_doc}, expected {}",
            words.len(),
            bits2words(max_doc)
        ));
        return Err(Status::InvalidArgument);
    }
    let tail_bits = max_doc % 64;
    if tail_bits != 0 && words[words.len() - 1] >> tail_bits != 0 {
        set_last_error(format!(
            "segment {segment}: live-docs bits set past maxDoc {max_doc}"
        ));
        return Err(Status::InvalidArgument);
    }
    let mut owned = crate::alloc::try_with_capacity::<u64>(words.len())?;
    owned.extend_from_slice(words);
    Ok(Some(FixedBitSet::from_words(owned, max_doc)))
}
