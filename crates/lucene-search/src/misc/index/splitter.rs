//! `IndexSplitter`: lists an index's segments and their sizes, removes
//! segments from its latest commit, or copies some of them -- file by file,
//! with no merging -- into a new index. It can only split an index that
//! already has several segments.
//!
//! Removing writes a new `segments_N` that no longer lists the segments; their
//! files stay in the directory.

use lucene_index::segment_info;
use lucene_index::segment_infos::{self, LuceneVersion, SegmentCommitInfo, SegmentInfos};
use lucene_store::directory::Directory;

use super::index_error;
use crate::{Error, Result};

/// `IndexSplitter`.
pub struct IndexSplitter<'d> {
    dir: &'d dyn Directory,
    /// `infos`: the latest commit, as read (and, after [`Self::remove`], as
    /// last written).
    pub infos: SegmentInfos,
}

impl<'d> IndexSplitter<'d> {
    /// `IndexSplitter(dir)`: reads the latest commit of `dir`.
    pub fn new(dir: &'d dyn Directory) -> Result<Self> {
        let infos = segment_infos::read_latest(dir).map_err(index_error)?;
        Ok(Self { dir, infos })
    }

    /// `listSegments()`: one line per segment, `name size`, where the size is
    /// the length of every file the segment owns in `###,###.###` format --
    /// what Java prints.
    pub fn list_segments(&self) -> Result<Vec<String>> {
        self.infos
            .segments
            .iter()
            .map(|sci| {
                let size = self.size_in_bytes(sci)?;
                Ok(format!("{} {}", sci.segment_name, group_digits(size)))
            })
            .collect()
    }

    /// `SegmentCommitInfo.sizeInBytes()`.
    fn size_in_bytes(&self, sci: &SegmentCommitInfo) -> Result<u64> {
        let mut total = 0u64;
        for file in self.files(sci)? {
            total = total.saturating_add(self.dir.file_length(&file)?);
        }
        Ok(total)
    }

    /// `SegmentCommitInfo.files()`.
    fn files(&self, sci: &SegmentCommitInfo) -> Result<Vec<String>> {
        let si = segment_info::parse_for_codec(
            &self.dir.open(&format!("{}.si", sci.segment_name))?,
            &sci.segment_id,
            &sci.codec_name,
        )
        .map_err(index_error)?;
        Ok(sci.files(&si.files))
    }

    fn get_info(&self, name: &str) -> Option<&SegmentCommitInfo> {
        self.infos.segments.iter().find(|s| s.segment_name == name)
    }

    /// `remove(segs)`: writes a commit without the named segments; a name that
    /// is not a segment is ignored.
    pub fn remove(&mut self, segs: &[&str]) -> Result<()> {
        self.infos
            .segments
            .retain(|s| !segs.contains(&s.segment_name.as_str()));
        // `changed()` then `commit(dir)`.
        self.infos.version = self.infos.version.saturating_add(1);
        self.infos.generation = self.infos.generation.saturating_add(1);
        self.infos.lucene_version = LATEST;
        self.infos.min_segment_lucene_version = self.min_version(&self.infos.segments)?;
        segment_infos::write(&self.infos, self.dir).map_err(index_error)?;
        Ok(())
    }

    /// `split(destDir, segs)`: a new index at `dest` of the named segments,
    /// their files copied as they are, under the same names.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a name that is not a segment (Java
    /// throws a `NullPointerException`).
    pub fn split(&self, dest: &dyn Directory, segs: &[&str]) -> Result<()> {
        let mut segments = Vec::with_capacity(segs.len());
        for name in segs {
            let sci = self
                .get_info(name)
                .ok_or_else(|| Error::IllegalArgument(format!("no segment named {name}")))?;
            segments.push(sci.clone());
            for file in self.files(sci)? {
                dest.copy_from(self.dir, &file, &file)?;
            }
        }
        let dest_infos = SegmentInfos {
            id: self.infos.id,
            generation: 1,
            format_version: segment_infos::VERSION_86,
            lucene_version: LATEST,
            index_created_version_major: self.infos.index_created_version_major,
            version: 1,
            counter: self.infos.counter,
            min_segment_lucene_version: self.min_version(&segments)?,
            segments,
            user_data: Vec::new(),
        };
        segment_infos::write(&dest_infos, dest).map_err(index_error)?;
        Ok(())
    }

    /// The oldest Lucene version that wrote any of `segments`, which
    /// `SegmentInfos.write` records.
    fn min_version(&self, segments: &[SegmentCommitInfo]) -> Result<Option<LuceneVersion>> {
        let mut min: Option<LuceneVersion> = None;
        for sci in segments {
            let si = segment_info::parse_for_codec(
                &self.dir.open(&format!("{}.si", sci.segment_name))?,
                &sci.segment_id,
                &sci.codec_name,
            )
            .map_err(index_error)?;
            let v = LuceneVersion {
                major: si.version.major,
                minor: si.version.minor,
                bugfix: si.version.bugfix,
            };
            if min.is_none_or(|m| (v.major, v.minor, v.bugfix) < (m.major, m.minor, m.bugfix)) {
                min = Some(v);
            }
        }
        Ok(min)
    }
}

/// `Version.LATEST`, which every commit written by this tool records.
const LATEST: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// `DecimalFormat("###,###.###")` of a whole number: digits in groups of three.
fn group_digits(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::group_digits;

    #[test]
    fn digits_group_in_threes() {
        assert_eq!(group_digits(0), "0");
        assert_eq!(group_digits(999), "999");
        assert_eq!(group_digits(1000), "1,000");
        assert_eq!(group_digits(1234567), "1,234,567");
    }
}
