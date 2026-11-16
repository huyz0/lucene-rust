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
mod tests;
