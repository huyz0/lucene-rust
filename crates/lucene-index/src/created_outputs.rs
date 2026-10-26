//! Creation ownership for operation-local codec output cleanup.

use lucene_store::Directory;

pub(crate) struct CreatedOutputs<'a> {
    inner: &'a dyn Directory,
    created: std::sync::Mutex<std::collections::BTreeSet<String>>,
}

impl<'a> CreatedOutputs<'a> {
    pub(crate) fn new(inner: &'a dyn Directory) -> Self {
        Self {
            inner,
            created: Default::default(),
        }
    }

    pub(crate) fn files(&self) -> std::sync::MutexGuard<'_, std::collections::BTreeSet<String>> {
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

impl Directory for CreatedOutputs<'_> {
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
