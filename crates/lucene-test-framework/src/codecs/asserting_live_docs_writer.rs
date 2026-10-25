//! Asserting live-docs writes installed in the native codec's deletion path.
//!
//! This writer adapts the existing asserting format and Lucene90 delegate.
//! Readers and the `Lucene104` segment codec name retain native routing.

use std::io;

use lucene_codecs::codec::Lucene104Codec;
use lucene_codecs::live_docs_writer::{LiveDocsWriteState, LiveDocsWriter};
use lucene_util::fixed_bit_set::FixedBitSet;

use super::asserting_live_docs::AssertingLiveDocsFormat;
use super::cranky_live_docs::{LiveDocsInfo, Lucene90LiveDocsFormat};

/// Native `Lucene104` routing with assertions before every live-docs write.
#[derive(Debug, Default, Clone, Copy)]
pub struct AssertingLiveDocsCodec;

impl Lucene104Codec for AssertingLiveDocsCodec {
    fn live_docs_writer(&self) -> Option<&dyn LiveDocsWriter> {
        Some(self)
    }
}

impl LiveDocsWriter for AssertingLiveDocsCodec {
    fn write_live_docs(
        &self,
        bits: &FixedBitSet,
        state: LiveDocsWriteState<'_>,
        new_del_count: usize,
    ) -> io::Result<()> {
        AssertingLiveDocsFormat::new(Lucene90LiveDocsFormat).write_live_docs(
            bits,
            state.directory,
            &LiveDocsInfo {
                name: state.segment,
                segment_id: state.id,
                max_doc: state.max_doc,
                del_gen: state.del_gen,
                next_del_gen: state.next_del_gen,
                del_count: state.del_count,
            },
            new_del_count,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use lucene_codecs::live_docs;
    use lucene_store::{ByteBuffersDirectory, Directory};

    use super::*;

    const ID: [u8; 16] = [43; 16];

    fn state(directory: &dyn Directory) -> LiveDocsWriteState<'_> {
        LiveDocsWriteState {
            directory,
            segment: "_test",
            id: &ID,
            max_doc: 70,
            del_gen: 1,
            next_del_gen: 36,
            del_count: 1,
        }
    }

    fn bits() -> FixedBitSet {
        let mut bits = FixedBitSet::new(70);
        bits.set_range(0, 70);
        bits.clear(0);
        bits.clear(69);
        bits
    }

    #[test]
    fn installed_writer_preserves_native_identity_and_bytes() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AssertingLiveDocsCodec>();
        let codec = AssertingLiveDocsCodec;
        assert_eq!(codec.name(), "Lucene104");
        assert!(codec.stored_fields_writer_factory().is_none());
        let directory = ByteBuffersDirectory::new();
        let bits = bits();
        codec
            .live_docs_writer()
            .unwrap()
            .write_live_docs(&bits, state(&directory), 1)
            .unwrap();
        assert_eq!(directory.list_all().unwrap(), ["_test_10.liv"]);
        let bytes = directory.open("_test_10.liv").unwrap();
        assert_eq!(&*bytes, live_docs::write(&bits, &ID, 36, 2).unwrap());
        let parsed = live_docs::parse(&bytes, &ID, 36, 70, 2).unwrap();
        assert_eq!(parsed.words(), bits.words());
    }

    #[test]
    fn assertions_run_before_creating_output() {
        let directory = ByteBuffersDirectory::new();
        for case in 0..3 {
            let mut state = state(&directory);
            let new_del_count = match case {
                0 => {
                    state.max_doc = 71;
                    1
                }
                1 => 2,
                _ => {
                    state.del_count = usize::MAX;
                    1
                }
            };
            assert!(catch_unwind(AssertUnwindSafe(|| {
                AssertingLiveDocsCodec.write_live_docs(&bits(), state, new_del_count)
            }))
            .is_err());
            assert!(directory.list_all().unwrap().is_empty());
        }
    }

    #[test]
    fn delegate_io_error_retains_its_kind() {
        let directory = ByteBuffersDirectory::new();
        AssertingLiveDocsCodec
            .write_live_docs(&bits(), state(&directory), 1)
            .unwrap();
        let error = AssertingLiveDocsCodec
            .write_live_docs(&bits(), state(&directory), 1)
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
    }
}
