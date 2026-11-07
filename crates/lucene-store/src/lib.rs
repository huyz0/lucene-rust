//! lucene-store: Directory/IndexInput abstractions. See /PLAN.md.

#[cfg(target_os = "linux")]
pub mod alignment;
pub mod blob_directory;
pub mod byte_buffers_directory;
pub mod cached_blob_directory;
pub mod codec_util;
#[cfg(any(test, feature = "test-util"))]
pub mod crashing_directory;
pub mod data_input;
pub mod data_output;
pub mod directory;
pub mod error;
pub mod file_switch_directory;
pub mod fs_lock_factory;
pub mod index_output;
pub mod lock;
pub mod lock_wrappers;
pub mod nrt_caching_directory;
pub mod page_cache;
mod page_disk;
pub mod paged_fs_directory;
pub mod paged_input;
pub mod rate_limiter;
pub mod store_registry;

pub use blob_directory::{BlobDirectory, BlobStore, InMemoryBlobStore};
pub use byte_buffers_directory::ByteBuffersDirectory;
pub use cached_blob_directory::CachedBlobDirectory;
pub use data_input::{DataInput, SliceInput};
pub use data_output::{DataOutput, VecDataOutput};
pub use directory::{
    BaseDirectory, Directory, EstimatedWrites, ExternalBytes, FsDirectory, Input, MmapDirectory,
};
pub use error::{Error, Result};
pub use file_switch_directory::FileSwitchDirectory;
pub use fs_lock_factory::{FsLockFactory, NativeFsLockFactory, SimpleFsLockFactory};
pub use index_output::{FsIndexOutput, IndexOutput, OutputSink};
pub use lock::{Lock, LockFactory, NoLockFactory, SingleInstanceLockFactory, VerifyingLockFactory};
pub use lock_wrappers::{LockValidatingDirectoryWrapper, SleepingLockWrapper};
pub use nrt_caching_directory::NrtCachingDirectory;
pub use page_cache::{
    PageCache, PageCacheConfig, PageCacheStats, PageCodec, PageDecoder, UnitLayout,
};
pub use paged_fs_directory::{PagedFsConfig, PagedFsDirectory};
pub use rate_limiter::{RateLimitedIndexOutput, RateLimiter, SimpleRateLimiter};
pub use store_registry::{
    register_store_type, register_store_wrapper, StoreProvider, StoreSettings, StoreSpec,
    StoreWrapper,
};

/// `IndexInput.prefetch` for readers that hold plain byte slices: announces
/// that each of `ranges` is about to be read, so the pages of a file read
/// by range ([`CachedBlobDirectory`]) are fetched together, in parallel,
/// instead of one round trip per page fault. Returns the first fetch error,
/// which a fault could only abort on. Slices of any other memory are left
/// alone, at the cost of one map lookup.
pub fn prefetch(ranges: &[&[u8]]) -> Result<()> {
    #[cfg(target_os = "linux")]
    return paged_input::prefetch(ranges);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = ranges;
        Ok(())
    }
}

/// Whether `bytes` lies in a file read by range, whose reads a [`prefetch`]
/// can batch. Readers ask before doing extra work to find what to prefetch.
pub fn is_prefetchable(bytes: &[u8]) -> bool {
    #[cfg(target_os = "linux")]
    return paged_input::is_paged(bytes);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = bytes;
        false
    }
}
