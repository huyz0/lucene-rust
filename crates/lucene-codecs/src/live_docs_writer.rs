//! Optional same-wire live-docs writers installed by the segment codec.
//!
//! The index layer retains generation advancement, synchronization and failed
//! output cleanup. Implementations write Lucene90 live-docs bytes readable by
//! the native `Lucene104` reader graph.

use std::fmt;
use std::io;

use lucene_store::directory::Directory;
use lucene_util::fixed_bit_set::FixedBitSet;

/// The segment and committed deletion state before a live-docs write.
#[derive(Clone, Copy)]
pub struct LiveDocsWriteState<'a> {
    pub directory: &'a dyn Directory,
    pub segment: &'a str,
    pub id: &'a [u8; 16],
    pub max_doc: usize,
    pub del_gen: i64,
    pub next_del_gen: i64,
    pub del_count: usize,
}

/// A stateless codec hook for a single live-docs output.
///
/// Recovery applies to returned I/O errors. Callback panics unwind normally;
/// they do not guarantee output cleanup or advancement of the retry generation.
pub trait LiveDocsWriter: Send + Sync + fmt::Debug {
    /// Writes `bits` at `state.next_del_gen`, including `new_del_count` deletes
    /// beyond `state.del_count`. The caller publishes the generation on success.
    fn write_live_docs(
        &self,
        bits: &FixedBitSet,
        state: LiveDocsWriteState<'_>,
        new_del_count: usize,
    ) -> io::Result<()>;
}
