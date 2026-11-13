//! OpenSearch's segment upgrade (`POST /<index>/_upgrade`): every committed
//! segment an older Lucene wrote is rewritten on its own by the current
//! codec, as `OpenSearchMergePolicy.findForcedMerges` proposes while
//! `upgradeInProgress` is set. A plain force merge leaves such a segment
//! alone when it is already merged (one segment, no deletes), on stock as
//! here; the upgrade is what rewrites it.

use super::{IndexWriter, Result};
use crate::segment_info::LuceneVersion;

impl IndexWriter<'_> {
    /// Rewrites each committed segment `OpenSearchMergePolicy.shouldUpgrade`
    /// picks against `current` (OpenSearch's `Version.CURRENT.luceneVersion`):
    /// one written by an older major, and with `only_ancient` false also one
    /// of an older minor of the same major. Each is merged alone, so the
    /// segment count does not change and partitions stay apart. Returns how
    /// many segments were rewritten; writes no commit when none is.
    pub fn upgrade_segments(
        &mut self,
        current: LuceneVersion,
        only_ancient: bool,
    ) -> Result<usize> {
        self.write_pending_try_changes()?;
        let mut old = Vec::new();
        for sci in &self.segment_infos.segments {
            let si = self.cfg.reader_pool.segment_info(self.dir, sci)?;
            if should_upgrade(si.version, current, only_ancient) {
                old.push(sci.segment_name.clone());
            }
        }
        for name in &old {
            self.execute_merge(std::slice::from_ref(name))?;
        }
        Ok(old.len())
    }
}

/// `OpenSearchMergePolicy.shouldUpgrade`: an older major always, an older
/// minor of the same major unless only ancient segments are upgraded.
pub(crate) fn should_upgrade(
    segment: LuceneVersion,
    current: LuceneVersion,
    only_ancient: bool,
) -> bool {
    current.major > segment.major
        || (!only_ancient && current.major == segment.major && current.minor > segment.minor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(major: i32, minor: i32, bugfix: i32) -> LuceneVersion {
        LuceneVersion {
            major,
            minor,
            bugfix,
        }
    }

    #[test]
    fn picks_older_majors_always_and_older_minors_unless_ancient_only() {
        let cur = v(10, 5, 1);
        assert!(should_upgrade(v(9, 12, 3), cur, true));
        assert!(should_upgrade(v(9, 12, 3), cur, false));
        assert!(!should_upgrade(v(10, 4, 0), cur, true));
        assert!(should_upgrade(v(10, 4, 0), cur, false));
        assert!(!should_upgrade(v(10, 5, 0), cur, false));
        assert!(!should_upgrade(v(10, 5, 2), cur, false));
    }

    use lucene_store::directory::{Directory, FsDirectory};
    use lucene_util::test_support::TempDir;

    use crate::merge_policy::upgrade::LATEST;
    use crate::segment_infos;

    /// A copy of `fixtures/data/bwc/<version>` (index files only).
    fn copy_fixture(version: &str) -> TempDir {
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/data/bwc")
            .join(version);
        let tmp = TempDir::new(&format!("upgrade-segments-{version}"));
        for entry in std::fs::read_dir(src).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.ends_with(".txt") {
                std::fs::copy(entry.path(), tmp.path().join(&name)).unwrap();
            }
        }
        tmp
    }

    /// Each committed segment's (writing version, live documents).
    fn segments(dir: &FsDirectory) -> Vec<(LuceneVersion, i32)> {
        segment_infos::read_latest(dir)
            .unwrap()
            .segments
            .iter()
            .map(|s| {
                let si = crate::segment_info::parse_for_codec(
                    &dir.open(&format!("{}.si", s.segment_name)).unwrap(),
                    &s.segment_id,
                    &s.codec_name,
                )
                .unwrap();
                (si.version, si.doc_count.saturating_sub(s.del_count))
            })
            .collect()
    }

    #[test]
    fn rewrites_each_lucene9_segment_alone_and_then_nothing() {
        let tmp = copy_fixture("9.12.2");
        let dir = FsDirectory::open(tmp.path());
        let before = segments(&dir);
        assert_eq!(before.len(), 2);
        assert!(before.iter().all(|(v, _)| v.major == 9));

        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).unwrap();
        assert_eq!(w.upgrade_segments(LATEST, true).unwrap(), 2);
        let after = segments(&dir);
        assert_eq!(after.len(), 2, "an upgrade merges each segment alone");
        assert!(after.iter().all(|(v, _)| *v == LATEST), "{after:?}");
        let live = |s: &[(LuceneVersion, i32)]| s.iter().map(|(_, n)| n).sum::<i32>();
        assert_eq!(live(&after), live(&before));
        for r in crate::check_index::check_directory(&dir).unwrap() {
            assert!(r.failures().is_empty(), "{}", r.segment_name);
        }
        let generation = segment_infos::read_latest(&dir).unwrap().generation;
        assert_eq!(w.upgrade_segments(LATEST, false).unwrap(), 0);
        assert_eq!(
            segment_infos::read_latest(&dir).unwrap().generation,
            generation
        );
    }

    #[test]
    fn a_force_merge_to_one_segment_rewrites_lucene9_segments() {
        let tmp = copy_fixture("9.12.2");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).unwrap();
        w.set_merge_policy(Some(crate::merge_policy::MergePolicyConfig::default()));
        w.force_merge(1).unwrap();
        let after = segments(&dir);
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].0, LATEST);
    }
}
