//! `CrankyLiveDocsFormat`: unchanged read/file delegation and a one-percent
//! write failure before touching the delegate. Java's shared `Random` becomes
//! `Rc<RefCell<JavaRandom>>`; exceptions become `io::Result`. The concrete
//! Lucene90 adapter uses this port's whole-file codec API and dense bitsets.
//! This is an explicit format adapter, not `Codec`/`IndexWriter` routing.

use std::cell::RefCell;
use std::io;
use std::rc::Rc;

use lucene_codecs::live_docs;
use lucene_store::data_output::DataOutput;
use lucene_store::directory::Directory;
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::java_random::JavaRandom;

/// The `SegmentCommitInfo` fields used by a live-docs format. The caller owns
/// generation advancement, including advancement after a failed write.
#[derive(Debug, Clone, Copy)]
pub struct LiveDocsInfo<'a> {
    pub name: &'a str,
    pub segment_id: &'a [u8; 16],
    pub max_doc: usize,
    pub del_gen: i64,
    pub next_del_gen: i64,
    pub del_count: usize,
}

/// The three `LiveDocsFormat` operations, with this port's directory and bits
/// representations. `IOContext` has no independent codec effect here.
pub trait LiveDocsFormat {
    fn read_live_docs(
        &self,
        dir: &dyn Directory,
        info: &LiveDocsInfo<'_>,
    ) -> io::Result<FixedBitSet>;
    fn write_live_docs(
        &self,
        bits: &FixedBitSet,
        dir: &dyn Directory,
        info: &LiveDocsInfo<'_>,
        new_del_count: usize,
    ) -> io::Result<()>;
    fn files(&self, info: &LiveDocsInfo<'_>, files: &mut Vec<String>) -> io::Result<()>;
}

/// Actual `.liv` parsing, writing, and file naming through `lucene-codecs`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Lucene90LiveDocsFormat;

fn directory_error(error: lucene_store::Error) -> io::Error {
    match error {
        lucene_store::Error::Io(error) => error,
        other => io::Error::other(other),
    }
}

fn file_name(name: &str, generation: i64) -> String {
    if generation == 0 {
        format!("{name}.liv")
    } else {
        format!("{name}_{}.liv", lucene_util::base36::to_base36(generation))
    }
}

impl LiveDocsFormat for Lucene90LiveDocsFormat {
    fn read_live_docs(
        &self,
        dir: &dyn Directory,
        info: &LiveDocsInfo<'_>,
    ) -> io::Result<FixedBitSet> {
        let bytes = dir
            .open(&file_name(info.name, info.del_gen))
            .map_err(directory_error)?;
        live_docs::parse(
            &bytes,
            info.segment_id,
            info.del_gen,
            info.max_doc,
            info.del_count,
        )
        .map_err(io::Error::other)
    }

    fn write_live_docs(
        &self,
        bits: &FixedBitSet,
        dir: &dyn Directory,
        info: &LiveDocsInfo<'_>,
        new_del_count: usize,
    ) -> io::Result<()> {
        // Java checks the count after closing the complete output. Ask the
        // byte encoder for the actual count, publish it, then check the caller's
        // expected count. A mismatched count must still leave the file behind.
        let actual = bits.len().checked_sub(bits.cardinality()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "live docs contain ghost bits")
        })?;
        let bytes = live_docs::write(bits, info.segment_id, info.next_del_gen, actual)
            .map_err(io::Error::other)?;
        let name = file_name(info.name, info.next_del_gen);
        let mut output = dir.create_output(&name).map_err(directory_error)?;
        output.write_bytes(&bytes);
        output.close().map_err(directory_error)?;
        if info.del_count.checked_add(new_del_count) != Some(actual) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "bits.deleted={actual} info.delcount={} newdelcount={new_del_count}",
                    info.del_count
                ),
            ));
        }
        Ok(())
    }

    fn files(&self, info: &LiveDocsInfo<'_>, files: &mut Vec<String>) -> io::Result<()> {
        // Java: Lucene90LiveDocsFormat.files, hasDeletions is delGen != -1.
        if info.del_gen != -1 {
            files.push(file_name(info.name, info.del_gen));
        }
        Ok(())
    }
}

/// A concrete format delegate and the same random stream other cranky
/// wrappers may share. Reads and file enumeration never draw randomness.
pub struct CrankyLiveDocsFormat<D> {
    delegate: D,
    random: Rc<RefCell<JavaRandom>>,
}

impl<D> CrankyLiveDocsFormat<D> {
    pub fn new(delegate: D, random: Rc<RefCell<JavaRandom>>) -> Self {
        Self { delegate, random }
    }
}

impl<D: LiveDocsFormat> LiveDocsFormat for CrankyLiveDocsFormat<D> {
    /// Java: `CrankyLiveDocsFormat.readLiveDocs`.
    fn read_live_docs(
        &self,
        dir: &dyn Directory,
        info: &LiveDocsInfo<'_>,
    ) -> io::Result<FixedBitSet> {
        self.delegate.read_live_docs(dir, info)
    }

    /// Java: `CrankyLiveDocsFormat.writeLiveDocs`.
    fn write_live_docs(
        &self,
        bits: &FixedBitSet,
        dir: &dyn Directory,
        info: &LiveDocsInfo<'_>,
        new_del_count: usize,
    ) -> io::Result<()> {
        if self.random.borrow_mut().next_int_bounded(100) == 0 {
            return Err(io::Error::other(
                "Fake IOException from LiveDocsFormat.writeLiveDocs()",
            ));
        }
        self.delegate
            .write_live_docs(bits, dir, info, new_del_count)
    }

    /// Java: `CrankyLiveDocsFormat.files`.
    fn files(&self, info: &LiveDocsInfo<'_>, files: &mut Vec<String>) -> io::Result<()> {
        self.delegate.files(info, files)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::ByteBuffersDirectory;

    const ID: [u8; 16] = *b"0123456789abcdef";

    fn info() -> LiveDocsInfo<'static> {
        LiveDocsInfo {
            name: "_0",
            segment_id: &ID,
            max_doc: 1,
            del_gen: -1,
            next_del_gen: 1,
            del_count: 0,
        }
    }

    struct Failing;
    impl LiveDocsFormat for Failing {
        fn read_live_docs(
            &self,
            _: &dyn Directory,
            _: &LiveDocsInfo<'_>,
        ) -> io::Result<FixedBitSet> {
            Err(io::Error::other("read delegate"))
        }
        fn write_live_docs(
            &self,
            _: &FixedBitSet,
            _: &dyn Directory,
            _: &LiveDocsInfo<'_>,
            _: usize,
        ) -> io::Result<()> {
            Err(io::Error::other("write delegate"))
        }
        fn files(&self, _: &LiveDocsInfo<'_>, files: &mut Vec<String>) -> io::Result<()> {
            files.push("before-error".into());
            Err(io::Error::other("files delegate"))
        }
    }

    #[test]
    fn directory_io_errors_keep_their_kind_and_other_errors_keep_context() {
        let dir = ByteBuffersDirectory::new();
        let mut committed = info();
        committed.del_gen = 1;
        committed.del_count = 1;
        assert_eq!(
            Lucene90LiveDocsFormat
                .read_live_docs(&dir, &committed)
                .unwrap_err()
                .kind(),
            io::ErrorKind::NotFound
        );
        let bits = FixedBitSet::new(1);
        Lucene90LiveDocsFormat
            .write_live_docs(&bits, &dir, &info(), 1)
            .unwrap();
        assert_eq!(
            Lucene90LiveDocsFormat
                .write_live_docs(&bits, &dir, &info(), 1)
                .unwrap_err()
                .kind(),
            io::ErrorKind::AlreadyExists
        );
        let error = directory_error(lucene_store::Error::Corrupted("broken directory".into()));
        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert!(error.to_string().contains("broken directory"));
    }

    #[test]
    fn delegation_errors_keep_their_message_and_draw_order() {
        let random = Rc::new(RefCell::new(JavaRandom::new(42)));
        let format = CrankyLiveDocsFormat::new(Failing, random.clone());
        let dir = ByteBuffersDirectory::new();
        assert_eq!(
            format
                .read_live_docs(&dir, &info())
                .unwrap_err()
                .to_string(),
            "read delegate"
        );
        let mut files = vec![];
        assert_eq!(
            format.files(&info(), &mut files).unwrap_err().to_string(),
            "files delegate"
        );
        assert_eq!(files, ["before-error"]);
        assert_eq!(
            format
                .write_live_docs(&FixedBitSet::new(1), &dir, &info(), 1)
                .unwrap_err()
                .to_string(),
            "write delegate"
        );
        let mut expected = JavaRandom::new(42);
        expected.next_int_bounded(100);
        assert_eq!(random.borrow_mut().next_long(), expected.next_long());
    }

    #[test]
    fn injection_precedes_validation_and_does_not_create_a_file() {
        let seed = (0..10000)
            .find(|&s| JavaRandom::new(s).next_int_bounded(100) == 0)
            .unwrap();
        let format = CrankyLiveDocsFormat::new(
            Lucene90LiveDocsFormat,
            Rc::new(RefCell::new(JavaRandom::new(seed))),
        );
        let dir = ByteBuffersDirectory::new();
        let err = format
            .write_live_docs(&FixedBitSet::new(1), &dir, &info(), usize::MAX)
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "Fake IOException from LiveDocsFormat.writeLiveDocs()"
        );
        assert!(dir.list_all().unwrap().is_empty());
    }

    #[test]
    fn adapter_handles_generation_zero_absent_files_and_postwrite_count_errors() {
        let mut info = info();
        let mut files = vec!["existing".into()];
        Lucene90LiveDocsFormat.files(&info, &mut files).unwrap();
        assert_eq!(files, ["existing"]);
        info.next_del_gen = 0;
        let dir = ByteBuffersDirectory::new();
        let err = Lucene90LiveDocsFormat
            .write_live_docs(&FixedBitSet::new(1), &dir, &info, 0)
            .unwrap_err();
        assert!(err.to_string().contains("bits.deleted=1"));
        assert_eq!(dir.list_all().unwrap(), ["_0.liv"]);
        info.del_gen = 0;
        info.del_count = 1;
        assert_eq!(
            Lucene90LiveDocsFormat
                .read_live_docs(&dir, &info)
                .unwrap()
                .cardinality(),
            0
        );
        Lucene90LiveDocsFormat.files(&info, &mut files).unwrap();
        assert_eq!(files, ["existing", "_0.liv"]);
        info.del_gen = 99;
        assert!(Lucene90LiveDocsFormat.read_live_docs(&dir, &info).is_err());
        assert!(Lucene90LiveDocsFormat
            .write_live_docs(&FixedBitSet::new(1), &dir, &info, usize::MAX)
            .is_err());
    }
}
