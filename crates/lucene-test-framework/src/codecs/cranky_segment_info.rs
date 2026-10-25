//! `CrankySegmentInfoFormat`: reads delegate unchanged; writes draw once
//! from Java's shared `Random` and fail before delegation on `nextInt(100) == 0`.
//!
//! The concrete Lucene99 adapter reads and writes real `.si` files through
//! `Directory` and `lucene_index::segment_info`. Rust passes the segment name
//! separately because its `SegmentInfo` has no name, and omits Java's IOContext
//! because `Directory` has no context argument. There is no Codec/IndexWriter
//! routing. Shared randomness uses `Rc<RefCell<_>>` in these single-threaded
//! test wrappers; its borrow ends before any delegate call.
//!
//! The concrete writer inherits the production encoder's trusted-metadata
//! precondition: the version major must be at least 7, and every file must
//! belong to the supplied segment name. It does not reproduce Java's
//! validation errors for metadata that violates those conditions.

use std::cell::RefCell;
use std::rc::Rc;

use lucene_index::segment_info::{self, SegmentInfo};
use lucene_store::{DataOutput, Directory};
use lucene_util::java_random::JavaRandom;

/// The read/write surface a cranky segment-info format delegates to.
pub trait SegmentInfoFormatDelegate {
    /// Java: `SegmentInfoFormat.read`.
    fn read(
        &mut self,
        directory: &dyn Directory,
        name: &str,
        id: &[u8; 16],
    ) -> segment_info::Result<SegmentInfo>;
    /// Java: `SegmentInfoFormat.write`.
    fn write(
        &mut self,
        directory: &dyn Directory,
        name: &str,
        info: &mut SegmentInfo,
    ) -> segment_info::Result<()>;
}

/// Real `Lucene99SegmentInfoFormat` file operations over the existing codec.
#[derive(Debug, Default, Clone, Copy)]
pub struct Lucene99SegmentInfoFormat;

impl SegmentInfoFormatDelegate for Lucene99SegmentInfoFormat {
    fn read(
        &mut self,
        directory: &dyn Directory,
        name: &str,
        id: &[u8; 16],
    ) -> segment_info::Result<SegmentInfo> {
        let bytes = directory.open(&format!("{name}.si"))?;
        segment_info::parse(&bytes, id)
    }

    fn write(
        &mut self,
        directory: &dyn Directory,
        name: &str,
        info: &mut SegmentInfo,
    ) -> segment_info::Result<()> {
        let file = format!("{name}.si");
        let mut output = directory.create_output(&file)?;
        // Java adds the .si only after createOutput succeeds.
        if !info.files.contains(&file) {
            info.files.push(file);
        }
        output.write_bytes(&segment_info::write(info, ""));
        output.close()?;
        Ok(())
    }
}

/// Java's `CrankySegmentInfoFormat`, sharing its random source with peers.
pub struct CrankySegmentInfoFormat<D> {
    delegate: D,
    random: Rc<RefCell<JavaRandom>>,
}

impl<D: SegmentInfoFormatDelegate> CrankySegmentInfoFormat<D> {
    /// Java: `CrankySegmentInfoFormat(delegate, random)`.
    pub fn new(delegate: D, random: Rc<RefCell<JavaRandom>>) -> Self {
        Self { delegate, random }
    }

    /// The wrapped format, for inspecting delegate state in tests.
    pub fn delegate(&self) -> &D {
        &self.delegate
    }

    /// The wrapped format, for configuring delegate failures in tests.
    pub fn delegate_mut(&mut self) -> &mut D {
        &mut self.delegate
    }
}

impl<D: SegmentInfoFormatDelegate> SegmentInfoFormatDelegate for CrankySegmentInfoFormat<D> {
    /// Java: `read`, with no random draw or exception translation.
    fn read(
        &mut self,
        directory: &dyn Directory,
        name: &str,
        id: &[u8; 16],
    ) -> segment_info::Result<SegmentInfo> {
        self.delegate.read(directory, name, id)
    }

    /// Java: `write`, drawing before delegation even when the delegate fails.
    fn write(
        &mut self,
        directory: &dyn Directory,
        name: &str,
        info: &mut SegmentInfo,
    ) -> segment_info::Result<()> {
        let fail = self.random.borrow_mut().next_int_bounded(100) == 0;
        if fail {
            return Err(lucene_store::Error::Io(std::io::Error::other(
                "Fake IOException from SegmentInfoFormat.write()",
            ))
            .into());
        }
        self.delegate.write(directory, name, info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_index::segment_info::LuceneVersion;
    use lucene_store::ByteBuffersDirectory;

    fn info() -> SegmentInfo {
        SegmentInfo {
            id: *b"0123456789abcdef",
            version: LuceneVersion {
                major: 10,
                minor: 5,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 7,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec![],
            attributes: vec![],
            index_sort: None,
        }
    }

    struct Fails {
        calls: usize,
        random: Rc<RefCell<JavaRandom>>,
    }
    impl SegmentInfoFormatDelegate for Fails {
        fn read(
            &mut self,
            _: &dyn Directory,
            _: &str,
            _: &[u8; 16],
        ) -> segment_info::Result<SegmentInfo> {
            assert!(self.random.try_borrow_mut().is_ok());
            self.calls += 1;
            Err(lucene_store::Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "delegate read failure",
            ))
            .into())
        }
        fn write(
            &mut self,
            _: &dyn Directory,
            _: &str,
            _: &mut SegmentInfo,
        ) -> segment_info::Result<()> {
            assert!(self.random.try_borrow_mut().is_ok());
            self.calls += 1;
            Err(lucene_store::Error::Io(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "delegate write failure",
            ))
            .into())
        }
    }

    #[test]
    fn exact_draws_and_delegate_failures_preserve_io_errors() {
        let random = Rc::new(RefCell::new(JavaRandom::new(42)));
        let mut expected = JavaRandom::new(42);
        let delegate = Fails {
            calls: 0,
            random: random.clone(),
        };
        let mut cranky = CrankySegmentInfoFormat::new(delegate, random.clone());
        let dir = ByteBuffersDirectory::new();
        let mut attempts = 0;
        let mut injected = 0;
        for _ in 0..1000 {
            let read = cranky.read(&dir, "_base", &info().id).unwrap_err();
            assert!(
                matches!(read, segment_info::Error::Store(lucene_store::Error::Io(ref e)) if e.kind() == std::io::ErrorKind::NotFound && e.to_string() == "delegate read failure")
            );
            attempts += 1;
            let fail = expected.next_int_bounded(100) == 0;
            let err = cranky.write(&dir, "_base", &mut info()).unwrap_err();
            if fail {
                injected += 1;
                assert_eq!(
                    err.to_string(),
                    "Fake IOException from SegmentInfoFormat.write()"
                );
            } else {
                attempts += 1;
                assert!(
                    matches!(err, segment_info::Error::Store(lucene_store::Error::Io(ref e)) if e.kind() == std::io::ErrorKind::PermissionDenied && e.to_string() == "delegate write failure")
                );
            }
        }
        assert!(injected > 0);
        assert_eq!(cranky.delegate().calls, attempts);
        cranky.delegate_mut().calls = 123;
        assert_eq!(cranky.delegate().calls, 123);
        assert_eq!(random.borrow_mut().next_int(), expected.next_int());
        assert!(dir.list_all().unwrap().is_empty());
    }

    #[test]
    fn real_codec_round_trip_and_create_failure() {
        let dir = ByteBuffersDirectory::new();
        let mut codec = Lucene99SegmentInfoFormat;
        let mut written = info();
        codec.write(&dir, "_base", &mut written).unwrap();
        assert_eq!(written.files, ["_base.si"]);
        let bytes = dir.open("_base.si").unwrap();
        assert_eq!(
            &*bytes,
            include_bytes!("../../../../fixtures/testframework/data/cranky_segment_info.si")
        );
        let read = codec.read(&dir, "_base", &written.id).unwrap();
        assert_eq!(read.doc_count, 7);
        assert_eq!(read.id, written.id);
        assert_eq!(read.files, written.files);
        assert!(codec.read(&dir, "_base", &[0; 16]).is_err());
        assert!(codec.read(&dir, "_absent", &written.id).is_err());
        let mut invalid = info();
        assert!(codec.write(&dir, "_base", &mut invalid).is_err());
        assert!(invalid.files.is_empty());
    }

    #[test]
    fn wrappers_share_one_live_random_source() {
        let random = Rc::new(RefCell::new(JavaRandom::new(17)));
        let mut first = CrankySegmentInfoFormat::new(
            Fails {
                calls: 0,
                random: random.clone(),
            },
            random.clone(),
        );
        let mut second = CrankySegmentInfoFormat::new(
            Fails {
                calls: 0,
                random: random.clone(),
            },
            random.clone(),
        );
        let mut expected = JavaRandom::new(17);
        let dir = ByteBuffersDirectory::new();
        for _ in 0..100 {
            for format in [&mut first, &mut second] {
                let err = format.write(&dir, "_s", &mut info()).unwrap_err();
                let expected_fake = expected.next_int_bounded(100) == 0;
                assert_eq!(
                    err.to_string().starts_with("Fake IOException"),
                    expected_fake
                );
            }
        }
        assert_eq!(random.borrow_mut().next_long(), expected.next_long());
    }
}
