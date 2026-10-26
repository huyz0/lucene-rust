//! Operation-local installed norms output and returned-error cleanup.

use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::norms::{self, NormsField};
use lucene_codecs::norms_writer::{self, NormsWriteState, NormsWriterFactory};
use lucene_store::Directory;

/// Writes borrowed columns through the installed consumer before publication.
/// Only files successfully created by this operation are removed on error.
pub(crate) fn write_installed(
    factory: &dyn NormsWriterFactory,
    state: NormsWriteState<'_>,
    field_infos: &[FieldInfo],
    fields: &[NormsField<'_>],
) -> norms::Result<Vec<String>> {
    let outputs = NormsOutputs::new(state.directory);
    let result = norms_writer::with_writer(
        factory,
        NormsWriteState {
            directory: &outputs,
            ..state
        },
        |writer| {
            for values in fields {
                let number = match values {
                    NormsField::Dense(number, _) | NormsField::Sparse(number, _) => *number,
                };
                let field = field_infos
                    .iter()
                    .find(|field| field.number == number)
                    .ok_or(norms::Error::UnknownFieldNumber(number))?;
                let values = match values {
                    NormsField::Dense(number, values) => NormsField::Dense(*number, values),
                    NormsField::Sparse(number, values) => NormsField::Sparse(*number, values),
                };
                writer.add_norms_field(field, values)?;
            }
            Ok::<_, norms::Error>(())
        },
    );
    if let Err(error) = result {
        for name in outputs.files().iter() {
            let _ = state.directory.delete_file(name);
        }
        return Err(error);
    }
    let files = outputs.files().iter().cloned().collect();
    Ok(files)
}

struct NormsOutputs<'a> {
    inner: &'a dyn Directory,
    created: std::sync::Mutex<std::collections::BTreeSet<String>>,
}

impl<'a> NormsOutputs<'a> {
    fn new(inner: &'a dyn Directory) -> Self {
        Self {
            inner,
            created: Default::default(),
        }
    }

    fn files(&self) -> std::sync::MutexGuard<'_, std::collections::BTreeSet<String>> {
        self.created.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn output(
        &self,
        output: lucene_store::Result<lucene_store::FsIndexOutput>,
    ) -> lucene_store::Result<lucene_store::FsIndexOutput> {
        use lucene_store::IndexOutput;
        let output = output?;
        self.files().insert(output.name().to_owned());
        Ok(output)
    }
}

impl Directory for NormsOutputs<'_> {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> lucene_store::Result<lucene_store::Input> {
        self.inner.open(name)
    }
    fn file_length(&self, name: &str) -> lucene_store::Result<u64> {
        self.inner.file_length(name)
    }
    fn create_output(&self, name: &str) -> lucene_store::Result<lucene_store::FsIndexOutput> {
        self.output(self.inner.create_output(name))
    }
    fn create_output_with_estimate(
        &self,
        name: &str,
        estimated_bytes: Option<u64>,
    ) -> lucene_store::Result<lucene_store::FsIndexOutput> {
        self.output(
            self.inner
                .create_output_with_estimate(name, estimated_bytes),
        )
    }
    fn create_temp_output(
        &self,
        prefix: &str,
        suffix: &str,
    ) -> lucene_store::Result<lucene_store::FsIndexOutput> {
        self.output(self.inner.create_temp_output(prefix, suffix))
    }
    fn create_temp_output_with_estimate(
        &self,
        prefix: &str,
        suffix: &str,
        estimated_bytes: Option<u64>,
    ) -> lucene_store::Result<lucene_store::FsIndexOutput> {
        self.output(
            self.inner
                .create_temp_output_with_estimate(prefix, suffix, estimated_bytes),
        )
    }
    fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
        self.inner.sync(names)
    }
    fn rename(&self, source: &str, dest: &str) -> lucene_store::Result<()> {
        self.inner.rename(source, dest)?;
        let mut created = self.files();
        if created.remove(source) {
            created.insert(dest.to_owned());
        }
        Ok(())
    }
    fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
        self.inner.delete_file(name)?;
        self.files().remove(name);
        Ok(())
    }
    fn sync_meta_data(&self) -> lucene_store::Result<()> {
        self.inner.sync_meta_data()
    }
    fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn lucene_store::Lock>> {
        self.inner.obtain_lock(name)
    }
    fn pending_deletions(&self) -> lucene_store::Result<std::collections::BTreeSet<String>> {
        self.inner.pending_deletions()
    }
    fn fs_directory_path(&self) -> Option<&std::path::Path> {
        self.inner.fs_directory_path()
    }
    #[cfg(test)]
    fn disable_virus_checker(&self) -> bool {
        self.inner.disable_virus_checker()
    }
    fn copy_from(&self, from: &dyn Directory, src: &str, dest: &str) -> lucene_store::Result<()> {
        use lucene_store::DataOutput;
        // Opening or creating can fail without owning dest. The Directory
        // default deletes dest even then, so copy through tracked creation.
        let bytes = from.open(src)?;
        let mut output = self.create_output(dest)?;
        output.write_bytes(&bytes);
        if let Err(error) = output.close() {
            // A failed deletion keeps this creation tracked for the enclosing
            // operation's cleanup retry. The copy error remains primary.
            let _ = self.delete_file(dest);
            return Err(error);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::norms_writer::NormsWriteSink;
    use lucene_store::{ByteBuffersDirectory, DataOutput};
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Default)]
    struct Factory {
        fail: Option<&'static str>,
        events: Arc<Mutex<Vec<&'static str>>>,
    }
    struct Sink<'a> {
        dir: &'a dyn Directory,
        fail: Option<&'static str>,
        events: Arc<Mutex<Vec<&'static str>>>,
    }
    fn failure(stage: &'static str) -> norms::Error {
        lucene_store::Error::Io(std::io::Error::other(stage)).into()
    }
    impl NormsWriterFactory for Factory {
        fn norms_consumer<'a>(
            &self,
            state: NormsWriteState<'a>,
        ) -> norms::Result<Box<dyn NormsWriteSink + 'a>> {
            self.events.lock().unwrap().push("open");
            if self.fail == Some("rename") {
                state.directory.rename("preexisting", "renamed")?;
                return Err(failure("rename"));
            }
            let mut output = state
                .directory
                .create_output_with_estimate("_test.nvm", Some(8))?;
            output.write_bytes(&[3, 1]);
            output.close()?;
            if self.fail == Some("open") {
                return Err(failure("open"));
            }
            Ok(Box::new(Sink {
                dir: state.directory,
                fail: self.fail,
                events: self.events.clone(),
            }))
        }
    }
    impl NormsWriteSink for Sink<'_> {
        fn add_norms_field(
            &mut self,
            field: &FieldInfo,
            values: NormsField<'_>,
        ) -> norms::Result<()> {
            self.events.lock().unwrap().push("field");
            assert_eq!(field.name, "body");
            assert_eq!(field.attributes, [("identity".into(), "preserved".into())]);
            assert!(matches!(values, NormsField::Sparse(7, [(1, 0)])));
            self.dir.create_output("_test.nvd")?.close()?;
            if self.fail == Some("field") {
                return Err(failure("field"));
            }
            Ok(())
        }
        fn close(&mut self) -> norms::Result<()> {
            self.events.lock().unwrap().push("close");
            if self.fail == Some("close") {
                return Err(failure("close"));
            }
            Ok(())
        }
    }
    fn state(dir: &dyn Directory) -> NormsWriteState<'_> {
        NormsWriteState {
            directory: dir,
            segment: "_test",
            id: &[1; 16],
            suffix: "",
            max_doc: 2,
        }
    }
    fn field() -> FieldInfo {
        let mut field = FieldInfo::new("body", 7);
        field
            .attributes
            .push(("identity".into(), "preserved".into()));
        field
    }
    #[test]
    fn success_publishes_actual_files_and_preserves_metadata_and_sparse_zero() {
        let dir = ByteBuffersDirectory::new();
        let factory = Factory::default();
        let files = write_installed(
            &factory,
            state(&dir),
            &[field()],
            &[NormsField::Sparse(7, &[(1, 0)])],
        )
        .unwrap();
        assert_eq!(files, ["_test.nvd", "_test.nvm"]);
        assert_eq!(*factory.events.lock().unwrap(), ["open", "field", "close"]);
        assert_eq!(dir.list_all().unwrap(), files);
    }
    #[test]
    fn returned_errors_clean_only_the_operation_outputs() {
        for stage in ["open", "field", "close"] {
            let dir = ByteBuffersDirectory::new();
            dir.create_output("unrelated").unwrap().close().unwrap();
            let factory = Factory {
                fail: Some(stage),
                ..Factory::default()
            };
            let error = write_installed(
                &factory,
                state(&dir),
                &[field()],
                &[NormsField::Sparse(7, &[(1, 0)])],
            )
            .unwrap_err();
            assert_eq!(error.to_string(), stage);
            assert_eq!(dir.list_all().unwrap(), ["unrelated"]);
            if stage != "open" {
                assert_eq!(factory.events.lock().unwrap().last(), Some(&"close"));
            }
        }
    }
    #[test]
    fn failed_factory_does_not_clean_a_renamed_preexisting_file() {
        let dir = ByteBuffersDirectory::new();
        let mut output = dir.create_output("preexisting").unwrap();
        output.write_bytes(&[8, 5]);
        output.close().unwrap();
        let factory = Factory {
            fail: Some("rename"),
            ..Factory::default()
        };
        assert_eq!(
            write_installed(&factory, state(&dir), &[], &[])
                .unwrap_err()
                .to_string(),
            "rename"
        );
        assert_eq!(dir.list_all().unwrap(), ["renamed"]);
        assert_eq!(&*dir.open("renamed").unwrap(), &[8, 5]);
    }
    #[test]
    fn existing_output_is_preserved_and_missing_field_closes_and_cleans() {
        let dir = ByteBuffersDirectory::new();
        let mut output = dir.create_output("_test.nvm").unwrap();
        output.write_bytes(&[9]);
        output.close().unwrap();
        assert!(write_installed(&Factory::default(), state(&dir), &[], &[]).is_err());
        assert_eq!(&*dir.open("_test.nvm").unwrap(), &[9]);
        dir.delete_file("_test.nvm").unwrap();
        let factory = Factory {
            fail: Some("close"),
            ..Factory::default()
        };
        let error = write_installed(&factory, state(&dir), &[], &[NormsField::Dense(7, &[1, 2])])
            .unwrap_err();
        assert!(matches!(error, norms::Error::UnknownFieldNumber(7)));
        assert!(dir.list_all().unwrap().is_empty());
        assert_eq!(*factory.events.lock().unwrap(), ["open", "close"]);
    }
    #[test]
    fn copy_failures_before_creation_preserve_unowned_destination() {
        let dir = ByteBuffersDirectory::new();
        for (name, bytes) in [("source", &[1, 2][..]), ("occupied", &[8, 5][..])] {
            let mut output = dir.create_output(name).unwrap();
            output.write_bytes(bytes);
            output.close().unwrap();
        }
        let tracked = NormsOutputs::new(&dir);
        for source in ["source", "missing"] {
            assert!(tracked.copy_from(&dir, source, "occupied").is_err());
            assert_eq!(&*dir.open("occupied").unwrap(), &[8, 5]);
            assert_eq!(&*dir.open("source").unwrap(), &[1, 2]);
            assert!(tracked.files().is_empty());
        }
    }

    struct CopyFault {
        fail_write: bool,
    }
    impl lucene_store::index_output::OutputFaults for CopyFault {
        fn before_write(&mut self, _: &str, _: usize) -> lucene_store::index_output::WriteVerdict {
            use lucene_store::index_output::WriteVerdict;
            if self.fail_write {
                WriteVerdict::Fail {
                    keep: 1,
                    error: std::io::Error::other("copy write"),
                }
            } else {
                WriteVerdict::Write
            }
        }
        fn on_close(&mut self, _: &str) -> lucene_store::index_output::CloseVerdict {
            use lucene_store::index_output::CloseVerdict;
            if self.fail_write {
                CloseVerdict::Close
            } else {
                CloseVerdict::FailAfterFlush(std::io::Error::other("copy close"))
            }
        }
    }
    struct CopyFaultDirectory {
        inner: ByteBuffersDirectory,
        fail_write: bool,
        fail_delete: std::sync::atomic::AtomicBool,
    }
    impl Directory for CopyFaultDirectory {
        fn list_all(&self) -> lucene_store::Result<Vec<String>> {
            self.inner.list_all()
        }
        fn open(&self, name: &str) -> lucene_store::Result<lucene_store::Input> {
            self.inner.open(name)
        }
        fn file_length(&self, name: &str) -> lucene_store::Result<u64> {
            self.inner.file_length(name)
        }
        fn create_output(&self, name: &str) -> lucene_store::Result<lucene_store::FsIndexOutput> {
            let mut output = self.inner.create_output(name)?;
            output.set_faults(Box::new(CopyFault {
                fail_write: self.fail_write,
            }));
            Ok(output)
        }
        fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
            self.inner.sync(names)
        }
        fn rename(&self, source: &str, dest: &str) -> lucene_store::Result<()> {
            self.inner.rename(source, dest)
        }
        fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
            if self.fail_delete.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(lucene_store::Error::Io(std::io::Error::other(
                    "cleanup delete",
                )));
            }
            self.inner.delete_file(name)
        }
        fn sync_meta_data(&self) -> lucene_store::Result<()> {
            self.inner.sync_meta_data()
        }
        fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn lucene_store::Lock>> {
            self.inner.obtain_lock(name)
        }
    }
    #[test]
    fn failed_copy_cleans_its_partial_output_and_retains_failed_cleanup_for_retry() {
        let source = ByteBuffersDirectory::new();
        let mut output = source.create_output("source").unwrap();
        output.write_bytes(&[1, 2, 3]);
        output.close().unwrap();
        for fail_write in [false, true] {
            for fail_delete in [false, true] {
                let dir = CopyFaultDirectory {
                    inner: ByteBuffersDirectory::new(),
                    fail_write,
                    fail_delete: std::sync::atomic::AtomicBool::new(fail_delete),
                };
                let tracked = NormsOutputs::new(&dir);
                let error = tracked.copy_from(&source, "source", "dest").unwrap_err();
                assert_eq!(
                    error.to_string(),
                    if fail_write {
                        "copy write"
                    } else {
                        "copy close"
                    }
                );
                if fail_delete {
                    assert_eq!(
                        *tracked.files(),
                        std::collections::BTreeSet::from(["dest".into()])
                    );
                    assert_eq!(
                        &*dir.open("dest").unwrap(),
                        if fail_write { &[1][..] } else { &[1, 2, 3][..] }
                    );
                    dir.fail_delete
                        .store(false, std::sync::atomic::Ordering::SeqCst);
                    tracked.delete_file("dest").unwrap();
                }
                assert!(tracked.files().is_empty());
                assert!(dir.list_all().unwrap().is_empty());
                assert_eq!(&*source.open("source").unwrap(), &[1, 2, 3]);
            }
        }
    }

    #[test]
    fn tracked_outputs_forward_directory_policy_and_track_successful_changes() {
        use lucene_store::{DataOutput, Directory, IndexOutput};
        let tmp = lucene_util::test_support::TempDir::new("norms-output-tracking");
        let dir = lucene_store::FsDirectory::open(&tmp);
        let tracked = NormsOutputs::new(&dir);
        let mut output = tracked.create_output("original").unwrap();
        output.write_bytes(&[1, 2, 3]);
        output.close().unwrap();
        assert_eq!(tracked.file_length("original").unwrap(), 3);
        assert_eq!(&*tracked.open("original").unwrap(), &[1, 2, 3]);
        assert_eq!(tracked.list_all().unwrap(), ["original"]);
        tracked.sync(&["original".into()]).unwrap();
        tracked.sync_meta_data().unwrap();
        assert_eq!(tracked.fs_directory_path(), dir.fs_directory_path());
        assert!(tracked.pending_deletions().unwrap().is_empty());
        assert!(!tracked.disable_virus_checker());
        drop(tracked.obtain_lock("write.lock").unwrap());
        tracked.copy_from(&dir, "original", "copy").unwrap();
        tracked.rename("copy", "renamed").unwrap();
        tracked.delete_file("original").unwrap();
        assert_eq!(
            *tracked.files(),
            std::collections::BTreeSet::from(["renamed".into()])
        );
        assert!(tracked.rename("missing", "never").is_err());
        assert!(tracked.delete_file("missing").is_err());
        assert!(tracked.copy_from(&dir, "missing", "never").is_err());
        let estimated = tracked
            .create_output_with_estimate("estimated", Some(12))
            .unwrap();
        estimated.close().unwrap();
        let temp = tracked.create_temp_output("norms", "data").unwrap();
        let temp_name = temp.name().to_owned();
        temp.close().unwrap();
        let estimated_temp = tracked
            .create_temp_output_with_estimate("norms", "data", Some(10))
            .unwrap();
        let estimated_temp_name = estimated_temp.name().to_owned();
        estimated_temp.close().unwrap();
        assert_eq!(
            *tracked.files(),
            std::collections::BTreeSet::from([
                "renamed".into(),
                "estimated".into(),
                temp_name,
                estimated_temp_name,
            ])
        );

        let memory = lucene_store::ByteBuffersDirectory::new();
        let tracked = NormsOutputs::new(&memory);
        memory.create_output("occupied").unwrap().close().unwrap();
        assert!(tracked.create_output("occupied").is_err());
        assert!(tracked.files().is_empty());
    }
}
