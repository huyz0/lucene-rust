//! `CrankyFieldInfosFormat`: delegate reads unchanged and fail one in 100
//! writes before entering the delegate, using Java's exact random sequence.
//!
//! Rust has byte-level field-info functions rather than a codec SPI. The
//! concrete [`Lucene94FieldInfosFormat`] adapter connects this wrapper to those
//! production functions and real directory I/O. Segment name/id replace the
//! parts of `SegmentInfo` the format uses; the directory owns I/O policy, so
//! there is no `IOContext` argument. This adapter is for explicit format tests;
//! it does not install a codec in `IndexWriter`. The shared random handle lets
//! multiple cranky format wrappers consume the same stream, as `CrankyCodec`
//! does. Rust's single-threaded handle replaces Java's thread-safe `Random`.
//!
//! The concrete adapter inherits [`field_infos::write`]'s trusted-metadata
//! precondition: callers must supply consistent `FieldInfos`. It does not
//! repeat Java's per-field `checkConsistency` after callers mutate public
//! metadata; rebuild through [`FieldInfos::new`] to validate those changes.

use std::cell::RefCell;
use std::rc::Rc;

use lucene_codecs::field_infos::{self, FieldInfos};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::{DataOutput, Directory};
use lucene_util::java_random::JavaRandom;

/// The exact message from Java's `write`, including its obsolete method name.
pub const FAKE_WRITE_ERROR: &str = "Fake IOException from FieldInfosFormat.getFieldInfosWriter()";

/// The `FieldInfosFormat` operations needed by the cranky wrapper.
pub trait FieldInfosFormat {
    /// Java: `FieldInfosFormat.read`.
    fn read(
        &self,
        directory: &dyn Directory,
        segment: &str,
        id: &[u8; ID_LENGTH],
        suffix: &str,
    ) -> field_infos::Result<FieldInfos>;

    /// Java: `FieldInfosFormat.write`.
    fn write(
        &mut self,
        directory: &dyn Directory,
        segment: &str,
        id: &[u8; ID_LENGTH],
        suffix: &str,
        infos: &FieldInfos,
    ) -> field_infos::Result<()>;
}

/// Production `.fnm` reader/writer, exposed through the test framework's seam.
pub struct Lucene94FieldInfosFormat;

fn file_name(segment: &str, suffix: &str) -> String {
    if suffix.is_empty() {
        format!("{segment}.fnm")
    } else {
        format!("{segment}_{suffix}.fnm")
    }
}

impl FieldInfosFormat for Lucene94FieldInfosFormat {
    fn read(
        &self,
        directory: &dyn Directory,
        segment: &str,
        id: &[u8; ID_LENGTH],
        suffix: &str,
    ) -> field_infos::Result<FieldInfos> {
        let bytes = directory.open(&file_name(segment, suffix))?;
        field_infos::parse(&bytes, id, suffix)
    }

    fn write(
        &mut self,
        directory: &dyn Directory,
        segment: &str,
        id: &[u8; ID_LENGTH],
        suffix: &str,
        infos: &FieldInfos,
    ) -> field_infos::Result<()> {
        let mut output = directory.create_output(&file_name(segment, suffix))?;
        output.write_bytes(&field_infos::write(&infos.fields, id, suffix));
        output.close()?;
        Ok(())
    }
}

/// Java's `CrankyFieldInfosFormat`, with a concrete or instrumented delegate.
pub struct CrankyFieldInfosFormat<F> {
    delegate: F,
    random: Rc<RefCell<JavaRandom>>,
}

impl<F> CrankyFieldInfosFormat<F> {
    /// Java: `CrankyFieldInfosFormat(delegate, random)`.
    pub fn new(delegate: F, random: Rc<RefCell<JavaRandom>>) -> Self {
        Self { delegate, random }
    }

    /// The unchanged underlying format, for a test's callback assertions.
    pub fn delegate(&self) -> &F {
        &self.delegate
    }

    /// The underlying format, for configuring delegate failures.
    pub fn delegate_mut(&mut self) -> &mut F {
        &mut self.delegate
    }
}

impl<F: FieldInfosFormat> FieldInfosFormat for CrankyFieldInfosFormat<F> {
    /// Java: `CrankyFieldInfosFormat.read`, with no random draw.
    fn read(
        &self,
        directory: &dyn Directory,
        segment: &str,
        id: &[u8; ID_LENGTH],
        suffix: &str,
    ) -> field_infos::Result<FieldInfos> {
        self.delegate.read(directory, segment, id, suffix)
    }

    /// Java: `CrankyFieldInfosFormat.write`, one draw even if the delegate fails.
    fn write(
        &mut self,
        directory: &dyn Directory,
        segment: &str,
        id: &[u8; ID_LENGTH],
        suffix: &str,
        infos: &FieldInfos,
    ) -> field_infos::Result<()> {
        if self.random.borrow_mut().next_int_bounded(100) == 0 {
            return Err(lucene_store::Error::Io(std::io::Error::other(FAKE_WRITE_ERROR)).into());
        }
        self.delegate.write(directory, segment, id, suffix, infos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
    use lucene_store::ByteBuffersDirectory;

    struct Recording {
        reads: RefCell<usize>,
        writes: usize,
        fail: bool,
    }

    impl FieldInfosFormat for Recording {
        fn read(
            &self,
            _: &dyn Directory,
            segment: &str,
            id: &[u8; 16],
            suffix: &str,
        ) -> field_infos::Result<FieldInfos> {
            assert_eq!((segment, *id, suffix), ("_test", [7; 16], "abc"));
            *self.reads.borrow_mut() += 1;
            Err(lucene_store::Error::Io(std::io::Error::other("delegate read")).into())
        }

        fn write(
            &mut self,
            _: &dyn Directory,
            segment: &str,
            id: &[u8; 16],
            suffix: &str,
            infos: &FieldInfos,
        ) -> field_infos::Result<()> {
            assert_eq!((segment, *id, suffix), ("_test", [7; 16], "abc"));
            assert_eq!(infos.fields[0].name, "field");
            self.writes += 1;
            if self.fail {
                Err(lucene_store::Error::Io(std::io::Error::other("delegate write")).into())
            } else {
                Ok(())
            }
        }
    }

    fn infos() -> FieldInfos {
        FieldInfos::new(vec![
            FieldInfo::new("field", 0).with_index_options(IndexOptions::Docs)
        ])
        .unwrap()
    }

    #[test]
    fn empty_infos_results_do_not_retain_unused_field_storage() {
        for fields in [Vec::new(), Vec::with_capacity(16)] {
            let empty = FieldInfos::new(fields).unwrap();
            assert!(empty.fields.is_empty());
            assert_eq!(empty.fields.capacity(), 0);
        }
    }

    #[test]
    fn rng_and_delegate_errors_preserve_java_order() {
        let directory = ByteBuffersDirectory::new();
        let random = Rc::new(RefCell::new(JavaRandom::new(42)));
        let mut expected = JavaRandom::new(42);
        let mut wrapper = CrankyFieldInfosFormat::new(
            Recording {
                reads: RefCell::new(0),
                writes: 0,
                fail: true,
            },
            random.clone(),
        );
        let mut injected = 0;
        let infos = infos();
        for _ in 0..512 {
            assert_eq!(
                wrapper
                    .read(&directory, "_test", &[7; 16], "abc")
                    .unwrap_err()
                    .to_string(),
                "delegate read"
            );
            let fail = expected.next_int_bounded(100) == 0;
            let error = wrapper
                .write(&directory, "_test", &[7; 16], "abc", &infos)
                .unwrap_err();
            assert_eq!(
                error.to_string(),
                if fail {
                    FAKE_WRITE_ERROR
                } else {
                    "delegate write"
                }
            );
            injected += usize::from(fail);
        }
        assert!(injected > 0);
        assert_eq!(*wrapper.delegate().reads.borrow(), 512);
        assert_eq!(wrapper.delegate().writes, 512 - injected);
        assert_eq!(random.borrow_mut().next_int(), expected.next_int());
        wrapper.delegate_mut().fail = false;
        while expected.next_int_bounded(100) == 0 {
            assert!(wrapper
                .write(&directory, "_test", &[7; 16], "abc", &infos)
                .is_err());
        }
        wrapper
            .write(&directory, "_test", &[7; 16], "abc", &infos)
            .unwrap();
    }

    #[test]
    fn shared_rng_and_real_directory_round_trip() {
        let directory = ByteBuffersDirectory::new();
        let random = Rc::new(RefCell::new(JavaRandom::new(0)));
        let mut expected = JavaRandom::new(0);
        let mut first = CrankyFieldInfosFormat::new(Lucene94FieldInfosFormat, random.clone());
        let mut second = CrankyFieldInfosFormat::new(Lucene94FieldInfosFormat, random.clone());
        let infos = infos();
        for (format, segment, suffix) in [(&mut first, "_a", ""), (&mut second, "_b", "c")] {
            assert_ne!(expected.next_int_bounded(100), 0);
            format
                .write(&directory, segment, &[7; 16], suffix, &infos)
                .unwrap();
            let decoded = format.read(&directory, segment, &[7; 16], suffix).unwrap();
            assert_eq!(decoded.fields[0].name, "field");
            assert_eq!(decoded.fields[0].index_options, IndexOptions::Docs);
            assert!(format.read(&directory, segment, &[8; 16], suffix).is_err());
        }
        assert_eq!(directory.list_all().unwrap(), ["_a.fnm", "_b_c.fnm"]);
        assert_eq!(random.borrow_mut().next_int(), expected.next_int());
        assert!(first.read(&directory, "_missing", &[7; 16], "").is_err());
        // The directory must see no new output when injection takes precedence.
        let seed = (0..10000)
            .find(|&seed| JavaRandom::new(seed).next_int_bounded(100) == 0)
            .unwrap();
        random.borrow_mut().set_seed(seed);
        assert_eq!(
            first
                .write(&directory, "_never", &[7; 16], "", &infos)
                .unwrap_err()
                .to_string(),
            FAKE_WRITE_ERROR
        );
        assert_eq!(directory.list_all().unwrap(), ["_a.fnm", "_b_c.fnm"]);
    }
}
