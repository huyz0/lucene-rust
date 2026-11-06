//! Segment partitions: documents are added under a partition (the value of
//! a partition field, as in a partitioned table), each segment holds the
//! documents of one partition, and merges never mix partitions.
//!
//! The on-disk shape is OpenSearch's context-aware segments
//! (`index.context_aware.enabled`), so stock OpenSearch reads and merges
//! these segments the same way: every flushed segment records its partition
//! in a `.si` attribute (`bucket`), the partition also rides on a few
//! fields' `.fnm` attributes (`_id`, `_seq_no`) where OpenSearch's merge
//! reads it back, and the merge policy is asked about each partition's
//! segments on their own (`CriteriaBasedMergePolicy`). OpenSearch gets there
//! with one child `IndexWriter` per partition whose segments are copied into
//! the shard's writer at every refresh (`CompositeIndexWriter`); here one
//! writer keeps an indexing buffer per partition
//! ([`crate::concurrent_writer::ConcurrentIndexWriter`]), so nothing is
//! written twice. The segments are ordinary Lucene segments: a reader that
//! does not know the attribute reads them as any other.

use std::collections::{BTreeMap, HashMap};

use lucene_codecs::field_infos::FieldInfo;

use super::{IndexWriter, IndexingConfig, Result};
use crate::merge_policy::api::{MergeSegment, MergeSpecification};
use crate::merge_policy::SegmentStat;
use crate::segment_infos::SegmentCommitInfo;

/// How a writer records its segments' partitions and keeps them apart. See
/// the module documentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentPartitioning {
    /// The `.si` attribute naming a segment's partition.
    pub attribute: String,
    /// The partition recorded on a segment flushed from documents added
    /// without one, if any.
    pub default_partition: Option<String>,
    /// Fields whose `.fnm` entry records the partition under the same
    /// attribute.
    pub field_attributes: Vec<String>,
    /// A partition with fewer documents than this (counted since the writer
    /// opened, plus its own committed segments) is written to the default
    /// partition, which small partitions share as an unpartitioned index
    /// would; once it has this many its documents get segments of their
    /// own. Its earlier documents stay where they were written, so a search
    /// for it reads its own segments and the default partition's. `0` gives
    /// every partition its own segments from the first document.
    pub pool_below_docs: usize,
    /// The field whose value is the partition, when the caller knows one
    /// with a single exact value per document: segments of a partition
    /// (not the default one) then record it under
    /// [`PARTITION_FIELD_ATTRIBUTE`], and the partition under
    /// [`PARTITION_VALUE_ATTRIBUTE`], which tell a search that every
    /// document in the segment has that field equal to that value and no
    /// other.
    pub field: Option<String>,
}

/// The `.si` attribute naming the field every document of a partition's
/// segment holds the partition in ([`SegmentPartitioning::field`]).
pub const PARTITION_FIELD_ATTRIBUTE: &str = "lucene_rust.partition_field";

/// The `.si` attribute holding the partition of a segment that records
/// [`PARTITION_FIELD_ATTRIBUTE`].
pub const PARTITION_VALUE_ATTRIBUTE: &str = "lucene_rust.partition_value";

impl SegmentPartitioning {
    /// OpenSearch's context-aware segments: the `bucket` attribute on the
    /// segment and on `_id` and `_seq_no` (`CriteriaBasedCodec`,
    /// `CriteriaBasedDocValuesWriter`), and `-2` for documents the shard's
    /// own writer takes rather than a partition's (no-ops, delete tombstones).
    pub fn opensearch() -> Self {
        SegmentPartitioning {
            attribute: "bucket".to_string(),
            default_partition: Some("-2".to_string()),
            field_attributes: vec!["_id".to_string(), "_seq_no".to_string()],
            pool_below_docs: 0,
            field: None,
        }
    }
}

impl IndexingConfig {
    /// Whether segments are kept apart by partition
    /// ([`IndexWriter::set_segment_partitioning`]).
    pub(crate) fn partitions_segments(&self) -> bool {
        self.segment_partitioning.is_some()
    }

    /// See [`IndexWriter::set_segment_partitioning`].
    pub(crate) fn partitioning(&self) -> Option<&SegmentPartitioning> {
        self.segment_partitioning.as_ref()
    }

    /// The partitioning, and the partition a buffer flushed for `partition`
    /// records.
    fn recorded_partition<'a>(
        &'a self,
        partition: Option<&'a str>,
    ) -> Option<(&'a SegmentPartitioning, &'a str)> {
        let partitioning = self.segment_partitioning.as_ref()?;
        let value = partition.or(partitioning.default_partition.as_deref())?;
        Some((partitioning, value))
    }

    /// Stamps a flushed segment's partition onto its `.fnm` fields.
    pub(crate) fn partition_fields(
        &self,
        mut fields: Vec<FieldInfo>,
        partition: Option<&str>,
    ) -> Vec<FieldInfo> {
        if let Some((partitioning, value)) = self.recorded_partition(partition) {
            for f in &mut fields {
                if partitioning.field_attributes.contains(&f.name) {
                    set_attribute(&mut f.attributes, &partitioning.attribute, value);
                }
            }
        }
        fields
    }

    /// Stamps a flushed segment's partition onto its `.si` attributes.
    pub(crate) fn partition_segment(
        &self,
        attributes: &mut Vec<(String, String)>,
        partition: Option<&str>,
    ) {
        if let Some((partitioning, value)) = self.recorded_partition(partition) {
            set_attribute(attributes, &partitioning.attribute, value);
            if let (Some(_), Some(field)) = (partition, &partitioning.field) {
                set_attribute(attributes, PARTITION_FIELD_ATTRIBUTE, field);
                set_attribute(attributes, PARTITION_VALUE_ATTRIBUTE, value);
            }
        }
    }
}

/// Sets `key` in `attributes`, replacing its value if present.
pub(crate) fn set_attribute(attributes: &mut Vec<(String, String)>, key: &str, value: &str) {
    match attributes.iter_mut().find(|(k, _)| k == key) {
        Some(slot) => value.clone_into(&mut slot.1),
        None => attributes.push((key.to_string(), value.to_string())),
    }
}

impl IndexWriter<'_> {
    /// Keeps partitions apart: each flushed segment records the partition its
    /// documents were added under, and a merge only ever takes segments of
    /// one partition. `None` (the default) partitions nothing. Segments
    /// already in the index keep whatever they record; the segments recording
    /// no partition are merged among themselves.
    pub fn set_segment_partitioning(&mut self, partitioning: Option<SegmentPartitioning>) {
        self.cfg_mut().segment_partitioning = partitioning;
    }

    /// See [`Self::set_segment_partitioning`].
    pub fn segment_partitioning(&self) -> Option<&SegmentPartitioning> {
        self.cfg.segment_partitioning.as_ref()
    }

    /// The partition committed segment `sci` records (`None` when it records
    /// none, or the writer partitions nothing).
    pub(crate) fn partition_of(&self, sci: &SegmentCommitInfo) -> Result<Option<String>> {
        let Some(partitioning) = &self.cfg.segment_partitioning else {
            return Ok(None);
        };
        let si = self.cfg.reader_pool.segment_info(self.dir, sci)?;
        Ok(si
            .attributes
            .iter()
            .find(|(k, _)| *k == partitioning.attribute)
            .map(|(_, v)| v.clone()))
    }

    /// The documents each partition's own committed segments hold (the
    /// default partition left out), for a reopened writer to know which
    /// partitions already left the pool.
    pub(crate) fn partition_doc_counts(&self) -> Result<HashMap<String, usize>> {
        let mut counts = HashMap::new();
        let Some(partitioning) = &self.cfg.segment_partitioning else {
            return Ok(counts);
        };
        for sci in &self.segment_infos.segments {
            let si = self.cfg.reader_pool.segment_info(self.dir, sci)?;
            let Some((_, p)) = si
                .attributes
                .iter()
                .find(|(k, _)| *k == partitioning.attribute)
            else {
                continue;
            };
            if partitioning.default_partition.as_deref() == Some(p.as_str()) {
                continue;
            }
            let n = counts.entry(p.clone()).or_insert(0usize);
            *n = n.saturating_add(usize::try_from(si.doc_count).unwrap_or(0));
        }
        Ok(counts)
    }

    /// Every committed segment's partition, by name; `None` when the writer
    /// partitions nothing.
    fn partitions_by_name(&self) -> Result<Option<HashMap<String, Option<String>>>> {
        if self.cfg.segment_partitioning.is_none() {
            return Ok(None);
        }
        let mut partitions = HashMap::with_capacity(self.segment_infos.segments.len());
        for sci in &self.segment_infos.segments {
            partitions.insert(sci.segment_name.clone(), self.partition_of(sci)?);
        }
        Ok(Some(partitions))
    }

    /// The `.si` attribute a merge of `sources` records: their partition,
    /// when they all record the same one.
    pub(crate) fn merged_partition_attributes(
        &self,
        sources: &[SegmentCommitInfo],
    ) -> Result<Vec<(String, String)>> {
        let Some(partitioning) = &self.cfg.segment_partitioning else {
            return Ok(Vec::new());
        };
        let mut common: Option<String> = None;
        // The partition field survives only when every source records it.
        let mut field: Option<Option<String>> = None;
        for (i, sci) in sources.iter().enumerate() {
            let si = self.cfg.reader_pool.segment_info(self.dir, sci)?;
            let attr = |key: &str| {
                si.attributes
                    .iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, v)| v.clone())
            };
            let partition = attr(&partitioning.attribute);
            match (i, partition) {
                (_, None) => return Ok(Vec::new()),
                (0, Some(g)) => common = Some(g),
                (_, Some(g)) if common.as_deref() == Some(g.as_str()) => {}
                _ => return Ok(Vec::new()),
            }
            let f = attr(PARTITION_FIELD_ATTRIBUTE);
            field = match field {
                None => Some(f),
                Some(prev) if prev == f => Some(prev),
                Some(_) => Some(None),
            };
        }
        let Some(common) = common else {
            return Ok(Vec::new());
        };
        let mut out = vec![(partitioning.attribute.clone(), common.clone())];
        if let Some(Some(f)) = field {
            out.push((PARTITION_FIELD_ATTRIBUTE.to_string(), f));
            out.push((PARTITION_VALUE_ATTRIBUTE.to_string(), common));
        }
        Ok(out)
    }

    /// `find` asked about each partition's segments on its own
    /// (`CriteriaBasedMergePolicy`), its answers concatenated; asked once
    /// about all of them when the writer partitions nothing. A partition of
    /// one segment is still asked: a single segment may be merged to reclaim
    /// its deletes.
    pub(crate) fn per_partition_stats<T>(
        &self,
        stats: &[SegmentStat],
        mut find: impl FnMut(&[SegmentStat]) -> Vec<T>,
    ) -> Result<Vec<T>> {
        let Some(partitions) = self.partitions_by_name()? else {
            return Ok(find(stats));
        };
        let mut out = Vec::new();
        for part in split_by(stats, |s| partitions.get(&s.name).cloned().flatten()) {
            out.extend(find(&part));
        }
        Ok(out)
    }

    /// [`Self::per_partition_stats`] for a pluggable policy's segments and
    /// specification.
    pub(crate) fn per_partition_spec<E>(
        &self,
        infos: &[MergeSegment],
        mut find: impl FnMut(&[MergeSegment]) -> std::result::Result<Option<MergeSpecification>, E>,
    ) -> Result<std::result::Result<Option<MergeSpecification>, E>> {
        let Some(partitions) = self.partitions_by_name()? else {
            return Ok(find(infos));
        };
        let mut out: Option<MergeSpecification> = None;
        for part in split_by(infos, |s| partitions.get(&s.name).cloned().flatten()) {
            match find(&part) {
                Ok(Some(spec)) => out
                    .get_or_insert_with(MergeSpecification::new)
                    .merges
                    .extend(spec.merges),
                Ok(None) => {}
                Err(e) => return Ok(Err(e)),
            }
        }
        Ok(Ok(out))
    }
}

/// `items` split by `key`, each part in its original order; parts ordered
/// by key, the segments recording none first.
fn split_by<T: Clone>(items: &[T], key: impl Fn(&T) -> Option<String>) -> Vec<Vec<T>> {
    let mut parts: BTreeMap<Option<String>, Vec<T>> = BTreeMap::new();
    for item in items {
        parts.entry(key(item)).or_default().push(item.clone());
    }
    parts.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_keeps_order_within_each_partition() {
        let items = ["a1", "b1", "-", "a2", "b2"];
        let parts = split_by(&items, |s| {
            (!s.starts_with('-')).then(|| s[..1].to_string())
        });
        assert_eq!(parts, vec![vec!["-"], vec!["a1", "a2"], vec!["b1", "b2"]]);
    }

    fn writer_cfg(
        dir: &lucene_store::FsDirectory,
        partitioning: Option<SegmentPartitioning>,
    ) -> std::sync::Arc<IndexingConfig> {
        let mut w = IndexWriter::open(
            dir,
            Vec::new(),
            "Lucene104",
            crate::segment_info::LuceneVersion {
                major: 10,
                minor: 5,
                bugfix: 0,
            },
        )
        .unwrap();
        w.set_segment_partitioning(partitioning);
        w.shared_config()
    }

    #[test]
    fn opensearch_partitioning_stamps_the_bucket_on_the_segment_and_its_id_fields() {
        let tmp = lucene_util::test_support::TempDir::new("segment-partitions-stamp");
        let dir = lucene_store::FsDirectory::open(&tmp);
        let cfg = writer_cfg(&dir, Some(SegmentPartitioning::opensearch()));
        let mut attributes = vec![("other".to_string(), "x".to_string())];
        cfg.partition_segment(&mut attributes, Some("tenant-a"));
        assert_eq!(
            attributes,
            vec![
                ("other".to_string(), "x".to_string()),
                ("bucket".to_string(), "tenant-a".to_string())
            ]
        );
        let mut none = Vec::new();
        cfg.partition_segment(&mut none, None);
        assert_eq!(none, vec![("bucket".to_string(), "-2".to_string())]);
        let fields = cfg.partition_fields(
            vec![
                FieldInfo::new("_id", 0),
                FieldInfo::new("body", 1),
                FieldInfo::new("_seq_no", 2),
            ],
            Some("t"),
        );
        let bucket = |f: &FieldInfo| {
            f.attributes
                .iter()
                .find(|(k, _)| k == "bucket")
                .map(|(_, v)| v.clone())
        };
        assert_eq!(
            fields.iter().map(bucket).collect::<Vec<_>>(),
            vec![Some("t".to_string()), None, Some("t".to_string())]
        );
    }

    #[test]
    fn nothing_is_stamped_without_partitioning() {
        let tmp = lucene_util::test_support::TempDir::new("segment-partitions-off");
        let dir = lucene_store::FsDirectory::open(&tmp);
        let cfg = writer_cfg(&dir, None);
        let mut attributes = Vec::new();
        cfg.partition_segment(&mut attributes, Some("t"));
        assert!(attributes.is_empty());
    }
}
