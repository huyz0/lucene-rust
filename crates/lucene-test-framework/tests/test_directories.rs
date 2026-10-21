//! `BaseDirectoryTestCase` over every `Directory` implementation, as
//! Lucene's `TestByteBuffersDirectory`, `TestNIOFSDirectory`,
//! `TestMMapDirectory`, `TestNRTCachingDirectory` and
//! `TestFileSwitchDirectory` run it.

use std::collections::HashSet;
use std::path::Path;

use lucene_store::{
    ByteBuffersDirectory, FileSwitchDirectory, FsDirectory, MmapDirectory, NrtCachingDirectory,
};
use lucene_test_framework::base_directory_tests;
use lucene_test_framework::store::base_directory::BaseDirectoryTestCase;

mod byte_buffers {
    use super::*;
    base_directory_tests!(BaseDirectoryTestCase::new(|_: &Path| {
        ByteBuffersDirectory::new()
    }));
}

mod niofs {
    use super::*;
    base_directory_tests!(BaseDirectoryTestCase::new(|p: &Path| FsDirectory::open(p))
        .with_fs_faults(|p, faults| FsDirectory::open(p).with_fs_faults(faults)));
}

mod mmap {
    use super::*;
    base_directory_tests!(
        BaseDirectoryTestCase::new(|p: &Path| MmapDirectory::open(p))
            .with_fs_faults(|p, faults| MmapDirectory::open(p).with_fs_faults(faults))
    );
}

/// `TestNRTCachingDirectory.getDirectory`: a cache over an `FSDirectory`.
mod nrt_caching {
    use super::*;
    base_directory_tests!(BaseDirectoryTestCase::new(|p: &Path| {
        NrtCachingDirectory::new(FsDirectory::open(p), 0.1, 0.3)
    }));
}

/// `TestFileSwitchDirectory.getDirectory`: two `FSDirectory`s, the
/// stored-fields extensions on the primary.
mod file_switch {
    use super::*;
    base_directory_tests!(BaseDirectoryTestCase::new(|p: &Path| {
        // Java's `FSDirectory` constructor creates its directory; this
        // port's first output does, which a listing may precede.
        for side in ["a", "b"] {
            let _ = std::fs::create_dir_all(p.join(side));
        }
        let extensions: HashSet<String> = ["fdt", "fdx", "fdm"].map(String::from).into();
        FileSwitchDirectory::new(
            extensions,
            MmapDirectory::open(p.join("a")),
            FsDirectory::open(p.join("b")),
        )
        .unwrap()
    }));
}
