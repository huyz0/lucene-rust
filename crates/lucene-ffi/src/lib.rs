//! lucene-ffi: the native library the OpenSearch plugin calls through Java's
//! Foreign Function & Memory API.
//!
//! - [`jvm_reader`] opens a shard's commit and runs query trees, sorts,
//!   collapse and rescore; [`jvm_aggs`] collects aggregations; [`jvm_fetch`]
//!   reads stored documents; [`jvm_analysis`] builds analyzers from the
//!   plugin's specs.
//! - [`engine_writer`] is the shard's index writer, fed operation blobs.
//! - [`store`] binds encrypted shard directories to their key upcall.
//! - [`ffm_bridge`] holds the allocation and error helpers only FFM needs.
//!
//! Every exported function runs inside [`error::guard`], so a panic becomes
//! [`FfiStatus::Panic`] and never unwinds into the JVM. Native objects cross
//! the boundary as generation-tagged `u64` handles ([`handle::SlotMap`]),
//! validated on every call; failures return a status code and leave a
//! thread-local message.

/// mimalloc returns freed memory to the system; glibc's per-thread arenas
/// kept what the JVM's indexing threads freed. It also frees a buffer that
/// another thread allocated without taking that thread's arena lock, which
/// glibc does, and those locks stalled write threads under bulk load.
///
/// Every allocation and free is reported to [`lucene_util::memory`], which
/// bounds what the library holds per request and process-wide.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static ALLOCATOR: Accounting<mimalloc::MiMalloc> = Accounting(mimalloc::MiMalloc);

/// A global allocator reporting to [`lucene_util::memory`].
#[cfg(feature = "mimalloc")]
struct Accounting<A>(A);

// SAFETY: every method forwards to the inner allocator with the caller's
// arguments unchanged, so its guarantees are this allocator's; the reports
// neither allocate nor touch the memory.
#[cfg(feature = "mimalloc")]
unsafe impl<A: std::alloc::GlobalAlloc> std::alloc::GlobalAlloc for Accounting<A> {
    #[inline]
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        // SAFETY: the caller's contract, passed through.
        let p = unsafe { self.0.alloc(layout) };
        if !p.is_null() {
            lucene_util::memory::on_alloc(layout.size());
        }
        p
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        // SAFETY: the caller's contract, passed through.
        let p = unsafe { self.0.alloc_zeroed(layout) };
        if !p.is_null() {
            lucene_util::memory::on_alloc(layout.size());
        }
        p
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        // SAFETY: the caller's contract, passed through.
        unsafe { self.0.dealloc(ptr, layout) };
        lucene_util::memory::on_free(layout.size());
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller's contract, passed through.
        let p = unsafe { self.0.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            if new_size >= layout.size() {
                lucene_util::memory::on_alloc(new_size - layout.size());
            } else {
                lucene_util::memory::on_free(layout.size() - new_size);
            }
        }
        p
    }
}

mod abi;
mod analysis_modules;
mod clauses;
pub(crate) mod engine_writer;
mod error;
mod ffm_bridge;
mod handle;
mod jvm_aggs;
mod jvm_analysis;
mod jvm_fetch;
mod jvm_function_score;
mod jvm_highlight;
mod jvm_highlight_fvh;
mod jvm_highlight_plain;
mod jvm_invert;
mod jvm_join;
mod jvm_knn;
mod jvm_knn_painless;
mod jvm_knn_score;
mod jvm_max_sim;
mod jvm_nodes;
mod jvm_painless_score;
mod jvm_pattern_phrase;
mod jvm_queries;
mod jvm_reader;
mod jvm_similarity;
mod live;
mod memory;
mod ml;
mod ml_torch;
mod raw;
mod registry;
mod sparse_writer;
mod store;
mod zstd_dictionary;
mod zstd_terms;

pub use error::FfiStatus;

/// Makes the codecs beyond Lucene's own readable: segments of an
/// `index.sparse` index record [`lucene_sparse::CODEC_NAME`]. Every exported
/// function's [`error::guard`] runs it once.
fn register_codecs() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(lucene_sparse::register_codec);
}
