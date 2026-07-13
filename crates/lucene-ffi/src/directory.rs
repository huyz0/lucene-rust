//! `ffi_open_directory`/`ffi_close_directory`: the first handle type in this
//! crate's surface — opens a real `FsDirectory` (`lucene-store`) at a
//! filesystem path so later calls have something to read segment files
//! from. See the module doc in `lib.rs` for the overall design.

use std::os::raw::c_char;

use lucene_store::directory::{Directory, FsDirectory};

use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::str_from_raw;
use crate::registry::{directories, lock_recovering};

/// Opens an `FsDirectory` rooted at the `path_len`-byte UTF-8 path at
/// `path`, writing the new handle to `*out_handle` on success.
///
/// # Safety
/// `path` must be valid for reads of `path_len` bytes; `out_handle` must be
/// valid for one `u64` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_open_directory(
    path: *const c_char,
    path_len: usize,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees `path` is valid for `path_len` bytes.
        let path_str = unsafe { str_from_raw(path as *const u8, path_len) }?;
        let dir = FsDirectory::open(path_str);
        let handle = lock_recovering(directories()).insert(dir);
        // SAFETY: caller contract guarantees `out_handle` is valid for one write.
        unsafe {
            *out_handle = handle;
        }
        Ok(())
    })
}

/// Closes a directory handle opened by [`ffi_open_directory`]. Returns
/// [`FfiStatus::InvalidHandle`] for an unknown/already-closed handle.
#[no_mangle]
pub extern "C" fn ffi_close_directory(handle: u64) -> i32 {
    guard(|| {
        lock_recovering(directories())
            .remove(handle)
            .map(|_| ())
            .ok_or_else(|| {
                set_last_error("ffi_close_directory: unknown or already-closed handle");
                FfiStatus::InvalidHandle
            })
    })
}

/// Reads a whole file named `name` from the directory identified by
/// `dir_handle`. Shared by every "open a segment file" call in `segment.rs`.
pub(crate) fn read_whole_file(dir_handle: u64, name: &str) -> Result<Vec<u8>, FfiStatus> {
    let directories = lock_recovering(directories());
    let dir = directories.get(dir_handle).ok_or_else(|| {
        set_last_error("unknown or already-closed directory handle");
        FfiStatus::InvalidHandle
    })?;
    let input = dir.open(name).map_err(|e| {
        set_last_error(format!("opening {name}: {e}"));
        FfiStatus::Io
    })?;
    Ok(input.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_dir() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_index/"
        )
        .to_string()
    }

    #[test]
    fn open_and_close_directory_roundtrip() {
        let path = fixture_dir();
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_directory(
                path.as_ptr() as *const c_char,
                path.len(),
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_ne!(handle, 0);

        let bytes = read_whole_file(handle, "_0.fnm").expect("read .fnm");
        assert!(!bytes.is_empty());

        let rc = ffi_close_directory(handle);
        assert_eq!(rc, FfiStatus::Ok.code());

        // Using it again must fail cleanly, not read freed state.
        let err = read_whole_file(handle, "_0.fnm");
        assert_eq!(err, Err(FfiStatus::InvalidHandle));
    }

    #[test]
    fn close_unknown_handle_is_invalid_handle() {
        let rc = ffi_close_directory(0xDEAD_BEEF);
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn double_close_is_invalid_handle_not_a_crash() {
        let path = fixture_dir();
        let mut handle: u64 = 0;
        unsafe {
            ffi_open_directory(
                path.as_ptr() as *const c_char,
                path.len(),
                &mut handle as *mut _,
            );
        }
        assert_eq!(ffi_close_directory(handle), FfiStatus::Ok.code());
        assert_eq!(ffi_close_directory(handle), FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn open_directory_null_out_handle_is_null_pointer_error() {
        let path = fixture_dir();
        let rc = unsafe {
            ffi_open_directory(
                path.as_ptr() as *const c_char,
                path.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
    }

    #[test]
    fn open_directory_invalid_utf8_path_is_invalid_utf8_error() {
        let bytes = [0xFFu8, 0xFE];
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_directory(
                bytes.as_ptr() as *const c_char,
                bytes.len(),
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidUtf8.code());
    }

    #[test]
    fn read_whole_file_missing_file_is_io_error() {
        let path = fixture_dir();
        let mut handle: u64 = 0;
        unsafe {
            ffi_open_directory(
                path.as_ptr() as *const c_char,
                path.len(),
                &mut handle as *mut _,
            );
        }
        let err = read_whole_file(handle, "does-not-exist.raw");
        assert_eq!(err, Err(FfiStatus::Io));
        ffi_close_directory(handle);
    }

    #[test]
    fn read_whole_file_unknown_directory_handle_is_invalid_handle() {
        let err = read_whole_file(0xFFFF_FFFF, "whatever");
        assert_eq!(err, Err(FfiStatus::InvalidHandle));
    }
}
