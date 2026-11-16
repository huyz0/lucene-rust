//! Java's `CrankyCompoundFormat`: readers delegate unchanged, while writes
//! draw once from a shared random and fail before delegation on `nextInt(100) == 0`.
//!
//! The concrete adapter uses the production Lucene90 compound writer and
//! [`CompoundReader`]. Segment name/id/files are explicit because Rust's
//! `SegmentInfo` has no name. There is no `Codec`/`IndexWriter` routing or
//! `IOContext`; `Rc<RefCell<JavaRandom>>` is single-threaded, unlike Java Random.
//!
//! The production codec buffers complete files rather than streaming them.
//! Both outputs are created before any source file is inspected and closed
//! in Java's reverse order, including on errors. A failed native encode leaves
//! only the headers/count here, rather than Java's partly copied sub-files.
//! Equal-size input packing inherits the production writer's stable sort,
//! whereas Java 10.5.0 uses a heap with unspecified equal-size order. Real
//! compound readers also inherit [`CompoundReader`]'s widening: `member_files`
//! corresponds to Java `listAll`; native `list_all` additionally lists loose
//! files, and `open` allows archive-side files. Member inputs are copied into
//! owned buffers. These adaptations do not change the cranky guard's behavior.

use std::cell::RefCell;
use std::rc::Rc;

use lucene_codecs::compound_format;
use lucene_codecs::simpletext::compound::strip_segment_name;
use lucene_index::compound_reader::CompoundReader;
use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::{DataOutput, Directory};
use lucene_util::java_random::JavaRandom;

/// The exact Java write-injection message.
pub const FAKE_WRITE_ERROR: &str = "Fake IOException from CompoundFormat.write()";

/// The `SegmentInfo` metadata consumed by the compound format.
#[derive(Debug, Clone, Copy)]
pub struct CompoundInfo<'a> {
    pub name: &'a str,
    pub id: &'a [u8; ID_LENGTH],
    pub files: &'a [String],
}

/// Java's `CompoundFormat` over actual compound readers and directory writes.
pub trait CompoundFormat {
    /// Java: `CompoundFormat.getCompoundReader`.
    fn get_compound_reader<'d>(
        &mut self,
        directory: &'d dyn Directory,
        info: &CompoundInfo<'_>,
    ) -> lucene_store::Result<CompoundReader<'d>>;

    /// Java: `CompoundFormat.write`.
    fn write(
        &mut self,
        directory: &dyn Directory,
        info: &CompoundInfo<'_>,
    ) -> compound_format::Result<()>;
}

/// Concrete Lucene90 `.cfs`/`.cfe` operations using the production codecs.
#[derive(Debug, Clone, Copy, Default)]
pub struct Lucene90CompoundFormat;

impl CompoundFormat for Lucene90CompoundFormat {
    fn get_compound_reader<'d>(
        &mut self,
        directory: &'d dyn Directory,
        info: &CompoundInfo<'_>,
    ) -> lucene_store::Result<CompoundReader<'d>> {
        CompoundReader::open(directory, info.name, info.id)
    }

    fn write(
        &mut self,
        directory: &dyn Directory,
        info: &CompoundInfo<'_>,
    ) -> compound_format::Result<()> {
        let mut data = directory.create_output(&format!("{}.cfs", info.name))?;
        let mut entries = match directory.create_output(&format!("{}.cfe", info.name)) {
            Ok(entries) => entries,
            Err(error) => {
                // Java's first resource still closes when creating the second fails.
                let _ = data.close();
                return Err(error.into());
            }
        };
        let mut data_prefix = Vec::new();
        codec_util::write_index_header(&mut data_prefix, "Lucene90CompoundData", 0, info.id, "");
        let mut entries_prefix = Vec::new();
        codec_util::write_index_header(
            &mut entries_prefix,
            "Lucene90CompoundEntries",
            0,
            info.id,
            "",
        );
        entries_prefix.write_vint(info.files.len() as i32);
        data.write_bytes(&data_prefix);
        entries.write_bytes(&entries_prefix);
        let outcome: compound_format::Result<()> = (|| {
            // Java obtains every length before opening any sub-file, then opens
            // smallest-first. The production encoder also orders by byte length.
            let mut sized = info
                .files
                .iter()
                .map(|file| directory.file_length(file).map(|length| (file, length)))
                .collect::<lucene_store::Result<Vec<_>>>()?;
            sized.sort_by_key(|(_, length)| *length);
            let mut files = Vec::with_capacity(sized.len());
            for (file, _) in sized {
                let bytes = directory.open(file)?;
                let name = strip_segment_name(file).to_string();
                // Java validates this member before it opens the next one.
                // Preserve that failure order even though the encoder buffers.
                compound_format::verify_sub_file(&name, &bytes, info.id)?;
                files.push((name, bytes.to_vec()));
            }
            let (cfs, cfe) = compound_format::write(info.id, &files)?;
            data.write_bytes(&cfs[data_prefix.len()..]);
            entries.write_bytes(&cfe[entries_prefix.len()..]);
            Ok(())
        })();
        // Java closes entries, then data, including when the body or a close
        // failed. Body errors win; otherwise the first close error wins.
        let entries_close = entries.close();
        let data_close = data.close();
        outcome?;
        entries_close?;
        data_close?;
        Ok(())
    }
}

/// Java's cranky compound guard, sharing one live RNG with sibling wrappers.
pub struct CrankyCompoundFormat<F> {
    delegate: F,
    random: Rc<RefCell<JavaRandom>>,
}

impl<F> CrankyCompoundFormat<F> {
    /// Java: `CrankyCompoundFormat(delegate, random)`.
    pub fn new(delegate: F, random: Rc<RefCell<JavaRandom>>) -> Self {
        Self { delegate, random }
    }

    /// The wrapped format, for observing callback state in tests.
    pub fn delegate(&self) -> &F {
        &self.delegate
    }

    /// The wrapped format, for configuring failures in tests.
    pub fn delegate_mut(&mut self) -> &mut F {
        &mut self.delegate
    }
}

impl<F: CompoundFormat> CompoundFormat for CrankyCompoundFormat<F> {
    /// Java: `getCompoundReader`, no RNG draw or exception translation.
    fn get_compound_reader<'d>(
        &mut self,
        directory: &'d dyn Directory,
        info: &CompoundInfo<'_>,
    ) -> lucene_store::Result<CompoundReader<'d>> {
        self.delegate.get_compound_reader(directory, info)
    }

    /// Java: `write`, drawing before entry into even a failing delegate.
    fn write(
        &mut self,
        directory: &dyn Directory,
        info: &CompoundInfo<'_>,
    ) -> compound_format::Result<()> {
        let fail = self.random.borrow_mut().next_int_bounded(100) == 0;
        if fail {
            return Err(lucene_store::Error::Io(std::io::Error::other(FAKE_WRITE_ERROR)).into());
        }
        self.delegate.write(directory, info)
    }
}

#[cfg(test)]
mod tests;
