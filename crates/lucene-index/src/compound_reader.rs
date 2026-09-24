//! Port of `Lucene90CompoundReader`: a compound segment's `.cfs`/`.cfe` pair
//! seen as a read-only [`Directory`] of the files packed inside it.
//!
//! Real Lucene flushes compound segments by default (`IndexWriterConfig`'s
//! `useCompoundFile`), so a segment Java wrote holds its whole codec state --
//! stored fields, postings, doc values, points, norms -- inside one `.cfs`,
//! and its `.si` lists just `.cfs`, `.cfe` and `.si`. Anything in this port
//! that reads a segment by file name (the merge, most of all) has to see the
//! members through this, or it sees a segment with no formats at all.
//!
//! One deliberate widening of Java's class: a file that lives *beside* a
//! sealed segment's archive is read from the directory underneath rather
//! than refused. Java's `SegmentReader` keeps two directories for this -- the
//! compound one for the segment's core files, the real one for everything
//! written after the segment was sealed (a `.liv`, a generational `.fnm` or
//! doc-values file) and for the `.si`/`.cfs`/`.cfe` themselves -- and this
//! lets a caller hold one. Only those names fall through: a generational name
//! (`IndexFileNames.parseGeneration > 0`), or the three archive files. Any
//! other name that is not a member is `NotFound`, as in Java, rather than a
//! stale loose file that happens to share a core file's name.

use lucene_codecs::compound_format::{self, CompoundEntries};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::{Directory, FsIndexOutput, Input};
use lucene_store::{Error, Result};

/// A compound segment's members, over the directory that holds it.
pub struct CompoundReader<'d> {
    base: &'d dyn Directory,
    segment_name: String,
    data: Input,
    entries: CompoundEntries,
}

impl<'d> CompoundReader<'d> {
    /// `Lucene90CompoundReader(directory, segmentInfo)`: reads `.cfe`, checks
    /// `.cfs`'s header, footer and length against it, and keeps `.cfs` open.
    /// A malformed pair is corruption, as in Java.
    pub fn open(
        base: &'d dyn Directory,
        segment_name: &str,
        segment_id: &[u8; ID_LENGTH],
    ) -> Result<Self> {
        let corrupt = |e: compound_format::Error| {
            Error::Corrupted(format!("compound segment {segment_name}: {e}"))
        };
        let cfe = base.open(&format!("{segment_name}.cfe"))?;
        let entries = compound_format::parse_entries(&cfe, segment_id).map_err(corrupt)?;
        let data = base.open(&format!("{segment_name}.cfs"))?;
        compound_format::check_data_header_footer(&data, segment_id, &entries).map_err(corrupt)?;
        Ok(CompoundReader {
            base,
            segment_name: segment_name.to_string(),
            data,
            entries,
        })
    }

    /// Every member's full file name (`_0` + `.fnm`, `_0` + `_Lucene104_0.tim`,
    /// ...) -- what the segment's `.si` would list if it were not compound.
    pub fn member_files(&self) -> Vec<String> {
        self.entries
            .names()
            .map(|id| format!("{}{id}", self.segment_name))
            .collect()
    }

    /// The member id of `name` (`IndexFileNames.stripSegmentName`), when it is
    /// one of this segment's members.
    fn member_id<'n>(&self, name: &'n str) -> Option<&'n str> {
        let id = name.strip_prefix(self.segment_name.as_str())?;
        self.entries.get(id).map(|_| id)
    }
}

/// Port of `IndexFileNames.parseGeneration`: the base-36 generation of a
/// per-generation file name (`_0_1.liv`, `_0_2_Lucene90_0.dvd`), `0` for a
/// file written with its segment (`_0.fdt`, `_0_Lucene104_0.tim`). An
/// unparsable generation reads as `0`, i.e. "not generational".
fn parse_generation(name: &str) -> i64 {
    let stem = name.split_once('.').map_or(name, |(stem, _)| stem);
    let Some(rest) = stem.strip_prefix('_') else {
        return 0;
    };
    let parts: Vec<&str> = rest.split('_').collect();
    if parts.len() == 2 || parts.len() == 4 {
        i64::from_str_radix(parts[1], 36).unwrap_or(0)
    } else {
        0
    }
}

/// A file that sits beside a sealed segment's archive rather than in it.
fn lives_beside_archive(name: &str) -> bool {
    parse_generation(name) > 0
        || [".si", ".cfs", ".cfe"]
            .iter()
            .any(|ext| name.ends_with(ext))
}

/// `Lucene90CompoundReader`'s write side: `UnsupportedOperationException`.
fn read_only(what: &str) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        format!("{what} on a compound-file directory, which is read-only"),
    ))
}

impl Directory for CompoundReader<'_> {
    fn list_all(&self) -> Result<Vec<String>> {
        let mut names = self.base.list_all()?;
        names.extend(self.member_files());
        names.sort();
        names.dedup();
        Ok(names)
    }

    fn open(&self, name: &str) -> Result<Input> {
        match self.member_id(name) {
            // `openInput`'s slice, copied out: an `Input` owns its bytes.
            Some(id) => compound_format::open_input(&self.data, &self.entries, id)
                .map(|input| Input::Owned(input.as_slice().to_vec()))
                .map_err(|e| Error::Corrupted(format!("{name}: {e}"))),
            None if lives_beside_archive(name) => self.base.open(name),
            None => Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{name} is not in compound segment {}", self.segment_name),
            ))),
        }
    }

    fn create_output(&self, _name: &str) -> Result<FsIndexOutput> {
        Err(read_only("createOutput"))
    }

    fn sync(&self, _names: &[String]) -> Result<()> {
        Err(read_only("sync"))
    }

    fn rename(&self, _source: &str, _dest: &str) -> Result<()> {
        Err(read_only("rename"))
    }

    fn delete_file(&self, _name: &str) -> Result<()> {
        Err(read_only("deleteFile"))
    }

    fn sync_meta_data(&self) -> Result<()> {
        Err(read_only("syncMetaData"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::codec_util;
    use lucene_store::FsDirectory;

    const ID: [u8; ID_LENGTH] = [7; ID_LENGTH];

    /// A complete standalone codec file: header, `body`, footer.
    fn sub_file(codec: &str, body: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        codec_util::write_index_header(&mut out, codec, 0, &ID, "");
        out.extend_from_slice(body);
        codec_util::write_footer(&mut out);
        out
    }

    fn compound_dir(tag: &str) -> (std::path::PathBuf, Vec<u8>, Vec<u8>) {
        let root = std::env::temp_dir().join(format!(
            "lucene-rust-compound-reader-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let fnm = sub_file("Fnm", b"fields");
        let tim = sub_file("Tim", b"terms, longer");
        let (cfs, cfe) = compound_format::write(
            &ID,
            &[
                (".fnm".to_string(), fnm.clone()),
                ("_Lucene104_0.tim".to_string(), tim.clone()),
            ],
        )
        .unwrap();
        std::fs::write(root.join("_3.cfs"), cfs).unwrap();
        std::fs::write(root.join("_3.cfe"), cfe).unwrap();
        std::fs::write(root.join("_3_1.liv"), b"loose").unwrap();
        (root, fnm, tim)
    }

    #[test]
    fn members_read_as_files_and_loose_files_fall_through() {
        let (root, fnm, tim) = compound_dir("read");
        let base = FsDirectory::open(&root);
        let reader = CompoundReader::open(&base, "_3", &ID).unwrap();
        assert_eq!(&*reader.open("_3.fnm").unwrap(), fnm.as_slice());
        assert_eq!(&*reader.open("_3_Lucene104_0.tim").unwrap(), tim.as_slice());
        // Not a member: the generational file beside the archive.
        assert_eq!(&*reader.open("_3_1.liv").unwrap(), b"loose");
        assert!(reader.open("_3.doc").is_err());
        // A loose file shadowing a core name is never read in a member's place.
        std::fs::write(root.join("_3.fdt"), b"stale").unwrap();
        assert!(matches!(
            reader.open("_3.fdt"),
            Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound
        ));
        // The archive files themselves are beside it too.
        assert!(reader.open("_3.cfe").is_ok());
        let mut members = reader.member_files();
        members.sort();
        assert_eq!(members, ["_3.fnm", "_3_Lucene104_0.tim"]);
        let all = reader.list_all().unwrap();
        for name in [
            "_3.cfs",
            "_3.cfe",
            "_3.fnm",
            "_3_1.liv",
            "_3_Lucene104_0.tim",
        ] {
            assert!(all.iter().any(|n| n == name), "{name} in {all:?}");
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn generations_parse_as_index_file_names_does() {
        for (name, generation) in [
            ("_0_1.liv", 1),
            ("_0_z.fnm", 35),
            ("_0_10_Lucene90_0.dvd", 36),
            ("_0.fdt", 0),
            ("_0_Lucene104_0.tim", 0),
            ("_0_notbase36!.liv", 0),
            ("segments_2", 0),
        ] {
            assert_eq!(parse_generation(name), generation, "{name}");
        }
    }

    #[test]
    fn a_compound_directory_is_read_only() {
        let (root, _, _) = compound_dir("read-only");
        let base = FsDirectory::open(&root);
        let reader = CompoundReader::open(&base, "_3", &ID).unwrap();
        let unsupported = |r: Result<()>| matches!(r, Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::Unsupported);
        assert!(reader.create_output("_3.x").is_err());
        assert!(unsupported(reader.sync(&["_3.fnm".to_string()])));
        assert!(unsupported(reader.rename("_3.fnm", "_3.y")));
        assert!(unsupported(reader.delete_file("_3.fnm")));
        assert!(unsupported(reader.sync_meta_data()));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_damaged_archive_is_corruption() {
        let (root, _, _) = compound_dir("damaged");
        let base = FsDirectory::open(&root);
        // Another segment's id.
        assert!(matches!(
            CompoundReader::open(&base, "_3", &[8; ID_LENGTH]),
            Err(Error::Corrupted(_))
        ));
        // A truncated `.cfs`.
        let cfs = std::fs::read(root.join("_3.cfs")).unwrap();
        std::fs::write(root.join("_3.cfs"), &cfs[..cfs.len() - 3]).unwrap();
        assert!(matches!(
            CompoundReader::open(&base, "_3", &ID),
            Err(Error::Corrupted(_))
        ));
        std::fs::remove_dir_all(root).unwrap();
    }
}
