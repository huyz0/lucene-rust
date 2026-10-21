//! `org.apache.lucene.tests.mockfile`: the mock filesystems Lucene's tests
//! run on -- Windows' refusal to delete an open file, a virus checker that
//! holds files, extra files the index must ignore, a handle limit, leak
//! detection and disabled fsync.
//!
//! Java stacks these as `java.nio` filesystem providers under every
//! `FSDirectory` a test creates. This port has no filesystem layer to wrap:
//! [`mock_fs::MockFs`] is one object holding every behaviour, installed
//! under an `FsDirectory`/`MmapDirectory` through lucene-store's `FsFaults`
//! seam (deletes and moves) and consulted by
//! [`crate::store::mock_directory_wrapper::MockDirectoryWrapper`] for the
//! rest (open handles, fsync).

pub mod mock_fs;
