//! The data-pack directory entry point; the analysis modules themselves are
//! registered by [`lucene_engine::analysis_modules`].

/// `lucene_analysis::data_pack::add_pack_dir`: `path` (UTF-8) is searched
/// for data packs after the directories added before.
///
/// # Safety
/// `path` must be valid for `path_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ffi_add_data_pack_dir(path: *const u8, path_len: usize) -> i32 {
    crate::error::guard(|| {
        // SAFETY: caller contract.
        let path = unsafe { crate::raw::bytes_from_raw(path, path_len)? };
        let path = std::str::from_utf8(path).map_err(|_| crate::error::FfiStatus::InvalidUtf8)?;
        lucene_analysis::data_pack::add_pack_dir(path);
        Ok(())
    })
}
