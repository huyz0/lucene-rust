//! The Lucene index format generation a writer produces.
//!
//! Lucene 10.6 changed two on-disk formats that Lucene 10.5 cannot read: a
//! sparse doc-values field's `IndexedDISI` moved from `.dvd` into a `.dvp`
//! file (doc-values format version 3), and `segments_N` gained the
//! per-segment doc-values overlay list (`SegmentInfos.VERSION_10_6`). Both
//! are read by 10.6 and later, and 10.6 reads the older forms too, so a writer
//! picks the newest format the Lucene that will open the index understands:
//! [`IndexFormat::Lucene105`] for hosts that ship Lucene 10.5.x (OpenSearch
//! 3.9) and [`IndexFormat::Lucene106`] for hosts that ship 10.6 or later.

/// Which Lucene release's index formats a writer produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum IndexFormat {
    /// Lucene 10.5.x: doc-values version 2, `segments_N` version 10. Every
    /// Lucene 10.x opens these.
    #[default]
    Lucene105,
    /// Lucene 10.6 and later: doc-values version 3 (`.dvp`), `segments_N`
    /// version 11. Lucene 10.5 cannot open these.
    Lucene106,
}

impl IndexFormat {
    /// The newest format a host running Lucene `major.minor` reads. A host
    /// older than 10.5 gets the oldest format this writer has.
    pub fn for_lucene_version(major: i32, minor: i32) -> Self {
        if (major, minor) >= (10, 6) {
            Self::Lucene106
        } else {
            Self::Lucene105
        }
    }

    /// [`Self::for_lucene_version`] from a version string such as `10.5.1`
    /// or `10.6.0-SNAPSHOT`; `None` when it does not start `major.minor`.
    pub fn for_lucene_version_string(version: &str) -> Option<Self> {
        let mut parts = version.split(['.', '-']);
        let major = parts.next()?.trim().parse().ok()?;
        let minor = parts.next()?.trim().parse().ok()?;
        Some(Self::for_lucene_version(major, minor))
    }

    /// The setting value: `10.5` or `10.6`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Lucene105 => "10.5",
            Self::Lucene106 => "10.6",
        }
    }

    /// Parses [`Self::name`] (also `lucene_10_5` and `10.5.x` spellings).
    pub fn parse(value: &str) -> Option<Self> {
        match value
            .trim()
            .trim_start_matches("lucene_")
            .replace('_', ".")
            .as_str()
        {
            "10.5" | "10.5.x" => Some(Self::Lucene105),
            "10.6" | "10.6.x" => Some(Self::Lucene106),
            _ => None,
        }
    }

    /// Whether doc values are written as format version 3, sparse fields'
    /// `IndexedDISI` in a `.dvp` file.
    pub fn doc_values_disi_file(self) -> bool {
        self == Self::Lucene106
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_host_version_picks_the_newest_format_it_reads() {
        assert_eq!(
            IndexFormat::for_lucene_version(10, 5),
            IndexFormat::Lucene105
        );
        assert_eq!(
            IndexFormat::for_lucene_version(10, 4),
            IndexFormat::Lucene105
        );
        assert_eq!(
            IndexFormat::for_lucene_version(9, 12),
            IndexFormat::Lucene105
        );
        assert_eq!(
            IndexFormat::for_lucene_version(10, 6),
            IndexFormat::Lucene106
        );
        assert_eq!(
            IndexFormat::for_lucene_version(11, 0),
            IndexFormat::Lucene106
        );
        assert_eq!(
            IndexFormat::for_lucene_version_string("10.5.1"),
            Some(IndexFormat::Lucene105)
        );
        assert_eq!(
            IndexFormat::for_lucene_version_string("10.6.0-SNAPSHOT"),
            Some(IndexFormat::Lucene106)
        );
        assert_eq!(IndexFormat::for_lucene_version_string("garbage"), None);
    }

    #[test]
    fn names_round_trip_and_the_default_is_the_older_format() {
        for f in [IndexFormat::Lucene105, IndexFormat::Lucene106] {
            assert_eq!(IndexFormat::parse(f.name()), Some(f));
        }
        assert_eq!(
            IndexFormat::parse("lucene_10_6"),
            Some(IndexFormat::Lucene106)
        );
        assert_eq!(IndexFormat::parse("10.7"), None);
        assert_eq!(IndexFormat::default(), IndexFormat::Lucene105);
        assert!(IndexFormat::Lucene106.doc_values_disi_file());
        assert!(!IndexFormat::Lucene105.doc_values_disi_file());
    }
}
