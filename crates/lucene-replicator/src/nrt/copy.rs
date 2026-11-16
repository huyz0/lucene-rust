//! `CopyOneFile` and `CopyJob`: copying the files of an NRT point to a
//! replica.
//!
//! Java's `CopyJob` is an abstract class whose subclass owns the connection
//! to the primary; here the shared half is [`CopyJobCore`] and the
//! connection is a [`CopyStream`] a concrete job passes to
//! [`CopyOneFile::visit`] (see [`super::local`] for the in-process one).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use lucene_store::directory::Directory;
use lucene_store::{DataOutput, FsIndexOutput, IndexOutput};

use super::node::{bytes_to_string, NodeCore};
use super::replica_file_deleter::ReplicaFileDeleter;
use super::types::{CopyState, FileMetaData};
use lucene_search::{Error, Result};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

fn io_error(message: String) -> Error {
    Error::Store(lucene_store::Error::Io(std::io::Error::other(message)))
}

/// The bytes a primary sends: what `Connection.in` is to Java's copy job.
pub trait CopyStream: Send {
    /// `readBytes`: fills `buf` or fails.
    ///
    /// # Errors
    /// The stream ended early or broke.
    fn read_bytes(&mut self, buf: &mut [u8]) -> Result<()>;

    /// `readVLong`: at most nine bytes, 63 bits; a ninth byte with its
    /// continuation bit set is rejected, as Java's negative-free `readVLong`.
    ///
    /// # Errors
    /// The stream ended early or broke, or the value overflowed.
    fn read_vlong(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..=56).step_by(7) {
            let mut byte = [0u8; 1];
            self.read_bytes(&mut byte)?;
            value |= u64::from(byte[0] & 0x7f) << shift;
            if byte[0] & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(io_error("invalid vLong".to_string()))
    }

    /// `CodecUtil.readBELong`.
    ///
    /// # Errors
    /// The stream ended early or broke.
    fn read_be_long(&mut self) -> Result<u64> {
        let mut bytes = [0u8; 8];
        self.read_bytes(&mut bytes)?;
        Ok(u64::from_be_bytes(bytes))
    }

    /// Closes the connection; reading afterwards fails.
    fn close(&mut self) {}
}

/// What the files of a replica need besides its [`NodeCore`]: the deleter,
/// the set of files being copied and the temporary outputs.
pub struct ReplicaCore {
    pub node: Arc<NodeCore>,
    pub deleter: ReplicaFileDeleter,
    copying: Mutex<BTreeSet<String>>,
}

impl ReplicaCore {
    pub(crate) fn new(node: Arc<NodeCore>, deleter: ReplicaFileDeleter) -> Self {
        Self {
            node,
            deleter,
            copying: Mutex::new(BTreeSet::new()),
        }
    }

    /// The replica's directory.
    pub fn dir(&self) -> &Arc<dyn Directory> {
        self.node.directory()
    }

    /// `message(text)`.
    pub fn message(&self, text: &str) {
        self.node.message(text);
    }

    /// `isVerboseFiles()`.
    pub fn is_verbose_files(&self) -> bool {
        self.node.is_verbose_files()
    }

    /// `createTempOutput(prefix, suffix, ioContext)`.
    ///
    /// # Errors
    /// The directory could not create the file.
    pub fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
        Ok(self.dir().create_temp_output(prefix, suffix)?)
    }

    /// `getFilesToCopy(files)`: the entries the replica does not already
    /// hold byte for byte (same header and footer).
    ///
    /// # Errors
    /// Reading a local file failed.
    pub fn files_to_copy(
        &self,
        files: &BTreeMap<String, FileMetaData>,
    ) -> Result<Vec<(String, FileMetaData)>> {
        let mut to_copy = Vec::new();
        for (name, meta) in files {
            if !self.file_is_identical(name, meta)? {
                to_copy.push((name.clone(), meta.clone()));
            }
        }
        Ok(to_copy)
    }

    fn file_is_identical(&self, file_name: &str, src: &FileMetaData) -> Result<bool> {
        let Some(dest) = self.node.read_local_file_metadata(file_name)? else {
            return Ok(false);
        };
        if dest.header != src.header || dest.footer != src.footer {
            if self.is_verbose_files() {
                self.message(&format!(
                    "file {file_name}: will copy [header/footer is different]"
                ));
            }
            return Ok(false);
        }
        Ok(true)
    }

    /// `startCopyFile(name)`.
    ///
    /// # Errors
    /// The file is already being copied.
    pub fn start_copy_file(&self, name: &str) -> Result<()> {
        if lock(&self.copying).insert(name.to_string()) {
            Ok(())
        } else {
            Err(Error::IllegalState(format!(
                "file {name} is being copied in two places!"
            )))
        }
    }

    /// `finishCopyFile(name)`.
    ///
    /// # Errors
    /// The file was not being copied.
    pub fn finish_copy_file(&self, name: &str) -> Result<()> {
        if lock(&self.copying).remove(name) {
            Ok(())
        } else {
            Err(Error::IllegalState(format!(
                "file {name} was not actually being copied?"
            )))
        }
    }
}

/// `CopyOneFile`: one file being copied into a temporary file, verified
/// against the primary's checksum as it completes.
pub struct CopyOneFile {
    out: Option<FsIndexOutput>,
    dest: Arc<ReplicaCore>,
    pub name: String,
    pub tmp_name: String,
    pub meta_data: FileMetaData,
    /// The file's length without the 8-byte checksum that ends it.
    pub bytes_to_copy: u64,
    copy_start: Instant,
    buffer: Vec<u8>,
    bytes_copied: u64,
}

impl CopyOneFile {
    /// `new CopyOneFile(in, dest, name, metaData, buffer)`.
    ///
    /// # Errors
    /// The temporary file could not be created, or the file is already being
    /// copied.
    pub fn new(
        dest: Arc<ReplicaCore>,
        name: &str,
        meta_data: FileMetaData,
        buffer_size: usize,
    ) -> Result<Self> {
        // Checked before the temporary file exists, so a short length leaves
        // nothing behind.
        let bytes_to_copy = meta_data.length.checked_sub(8).ok_or_else(|| {
            Error::IllegalState(format!("file {name}: length {} < 8", meta_data.length))
        })?;
        let out = dest.create_temp_output(name, "copy")?;
        let tmp_name = out.name().to_string();
        if dest.is_verbose_files() {
            dest.message(&format!(
                "file {name}: start copying to tmp file {tmp_name} length={}",
                bytes_to_copy.saturating_add(8)
            ));
        }
        dest.start_copy_file(name)?;
        Ok(Self {
            out: Some(out),
            dest,
            name: name.to_string(),
            tmp_name,
            meta_data,
            bytes_to_copy,
            copy_start: Instant::now(),
            buffer: vec![0u8; buffer_size.max(1)],
            bytes_copied: 0,
        })
    }

    /// `close()`: closes the temporary file and ends the copy.
    ///
    /// # Errors
    /// Closing the output failed, or the copy was not registered.
    pub fn close(&mut self) -> Result<()> {
        if let Some(out) = self.out.take() {
            out.close()?;
            self.dest.finish_copy_file(&self.name)?;
        }
        Ok(())
    }

    /// `getBytesCopied()`.
    pub fn bytes_copied(&self) -> u64 {
        self.bytes_copied
    }

    /// `visit()`: copies up to ten buffers; `true` when the file is done
    /// and its checksum verified.
    ///
    /// # Errors
    /// The stream failed or the checksum disagrees.
    pub fn visit(&mut self, input: &mut dyn CopyStream) -> Result<bool> {
        for _ in 0..10 {
            let bytes_left = self.bytes_to_copy.saturating_sub(self.bytes_copied);
            let out = self
                .out
                .as_mut()
                .ok_or_else(|| Error::AlreadyClosed(format!("file {} is closed", self.name)))?;
            if bytes_left == 0 {
                let checksum = out.checksum();
                if checksum != self.meta_data.checksum {
                    self.dest.message(&format!(
                        "file {}: checksum mismatch after copy (bits flipped during network copy?) after-copy checksum={} vs expected={}; cancel job",
                        self.tmp_name, checksum, self.meta_data.checksum
                    ));
                    return Err(io_error(format!(
                        "file {}: checksum mismatch after file copy",
                        self.name
                    )));
                }
                let claimed = input.read_be_long()?;
                if claimed != checksum {
                    self.dest.message(&format!(
                        "file {}: checksum claimed by primary disagrees with the file's footer: claimed checksum={} vs actual={}",
                        self.tmp_name, checksum, claimed
                    ));
                    return Err(io_error(format!(
                        "file {}: checksum mismatch after file copy",
                        self.name
                    )));
                }
                out.write_bytes(&checksum.to_be_bytes());
                self.bytes_copied = self.bytes_copied.saturating_add(8);
                self.close()?;
                if self.dest.is_verbose_files() {
                    self.dest.message(&format!(
                        "file {}: done copying [{}, {:.3}ms]",
                        self.name,
                        bytes_to_string(self.meta_data.length),
                        self.copy_start.elapsed().as_secs_f64() * 1000.0
                    ));
                }
                return Ok(true);
            }
            let to_copy = usize::try_from(bytes_left)
                .unwrap_or(usize::MAX)
                .min(self.buffer.len());
            input.read_bytes(&mut self.buffer[..to_copy])?;
            out.write_bytes(&self.buffer[..to_copy]);
            self.bytes_copied = self.bytes_copied.saturating_add(to_copy as u64);
        }
        Ok(false)
    }
}

/// `CopyJob.OnceDone`.
pub type OnceDone = Arc<dyn Fn(&dyn CopyJob) -> Result<()> + Send + Sync>;

static JOB_COUNTER: AtomicU64 = AtomicU64::new(0);

/// What a job changes while it runs, guarded by one lock.
pub struct JobState {
    /// `toCopy`.
    pub to_copy: Vec<(String, FileMetaData)>,
    /// `totBytes`.
    pub tot_bytes: u64,
    /// `totBytesCopied`.
    pub tot_bytes_copied: u64,
    /// `current`.
    pub current: Option<CopyOneFile>,
    /// `exc`: why the job failed or was cancelled.
    pub exc: Option<String>,
    /// `cancelReason`.
    pub cancel_reason: Option<String>,
    /// `copiedFiles`: file name to its finished temporary file.
    pub copied_files: BTreeMap<String, String>,
}

/// `CopyJob`'s fields and the methods that need no connection.
pub struct CopyJobCore {
    /// `ord`: the job's number, which orders equal priorities.
    pub ord: u64,
    pub reason: String,
    pub dest: Arc<ReplicaCore>,
    /// `files`: every file of the point, `None` for a job that gets them
    /// with its copy state.
    pub files: Option<Arc<BTreeMap<String, FileMetaData>>>,
    pub high_priority: bool,
    pub once_done: Option<OnceDone>,
    pub start: Instant,
    pub state: Mutex<JobState>,
}

impl CopyJobCore {
    /// `CopyJob(reason, files, dest, highPriority, onceDone)`: works out
    /// which of `files` the replica lacks.
    ///
    /// # Errors
    /// Reading the local files failed (the job is cancelled).
    pub fn new(
        reason: &str,
        files: Option<Arc<BTreeMap<String, FileMetaData>>>,
        dest: Arc<ReplicaCore>,
        high_priority: bool,
        once_done: Option<OnceDone>,
    ) -> Result<Self> {
        let to_copy = match &files {
            Some(files) => match dest.files_to_copy(files) {
                Ok(to_copy) => to_copy,
                Err(e) => {
                    dest.message(&format!("top: cancel after copying 0 b; exc={e}"));
                    return Err(e);
                }
            },
            None => Vec::new(),
        };
        Ok(Self {
            ord: JOB_COUNTER
                .fetch_add(1, Ordering::Relaxed)
                .saturating_add(1),
            reason: reason.to_string(),
            dest,
            files,
            high_priority,
            once_done,
            start: Instant::now(),
            state: Mutex::new(JobState {
                to_copy,
                tot_bytes: 0,
                tot_bytes_copied: 0,
                current: None,
                exc: None,
                cancel_reason: None,
                copied_files: BTreeMap::new(),
            }),
        })
    }

    /// The job's state, locked.
    pub fn state(&self) -> MutexGuard<'_, JobState> {
        lock(&self.state)
    }

    /// `cancel(reason, exc)`: records the failure, deletes the finished
    /// temporary files and the partial one. A second cancel does nothing.
    ///
    /// # Errors
    /// Deleting the partial file failed.
    pub fn cancel(&self, reason: &str, exc: Option<&str>) -> Result<()> {
        let mut st = self.state();
        if st.exc.is_some() {
            return Ok(());
        }
        self.dest.message(&format!(
            "top: cancel after copying {}; exc={}:\n  files={:?}\n  copiedFiles={:?}",
            bytes_to_string(st.tot_bytes_copied),
            exc.unwrap_or("null"),
            self.files.as_ref().map(|f| f.keys().collect::<Vec<_>>()),
            st.copied_files.keys().collect::<Vec<_>>()
        ));
        st.exc = Some(exc.unwrap_or("cancelled").to_string());
        st.cancel_reason = Some(reason.to_string());
        for tmp in st.copied_files.values() {
            let _ = self.dest.dir().delete_file(tmp);
        }
        if let Some(mut current) = st.current.take() {
            let _ = current.close();
            if self.dest.is_verbose_files() {
                self.dest
                    .message(&format!("remove partial file {}", current.tmp_name));
            }
            self.dest.deleter.force_delete_file(&current.tmp_name)?;
        }
        Ok(())
    }

    /// `transferAndCancel(prevJob)`: takes over the files `prev` already
    /// copied, and its file in progress, then cancels it.
    ///
    /// # Errors
    /// Deleting `prev`'s leftovers failed (this job is cancelled).
    pub fn transfer_and_cancel(&self, prev: &CopyJobCore) -> Result<()> {
        let result = self.transfer(prev);
        if let Err(e) = &result {
            self.dest.message("xfer: exc during transferAndCancel");
            let _ = self.cancel("exc during transferAndCancel", Some(&e.to_string()));
        }
        result
    }

    fn transfer(&self, prev: &CopyJobCore) -> Result<()> {
        self.dest
            .message(&format!("CopyJob: now transfer prevJob ord={}", prev.ord));
        let mut mine = self.state();
        let mut theirs = prev.state();
        if theirs.exc.is_some() {
            self.dest
                .message("xfer: prevJob was already cancelled; skip transfer");
            return Ok(());
        }
        theirs.exc = Some("transferred".to_string());
        let mut already_copied = 0u64;
        let mut keep = Vec::new();
        for (file_name, meta) in std::mem::take(&mut mine.to_copy) {
            if let Some(prev_tmp) = theirs.copied_files.remove(&file_name) {
                already_copied = already_copied.saturating_add(meta.length);
                self.dest.message(&format!(
                    "xfer: carry over already-copied file {file_name} ({prev_tmp}, {} bytes)",
                    meta.length
                ));
                mine.copied_files.insert(file_name, prev_tmp);
            } else if theirs.current.as_ref().is_some_and(|c| c.name == file_name) {
                let current = theirs.current.take().expect("checked above");
                self.dest.message(&format!(
                    "xfer: carry over in-progress file {file_name} ({}) bytesCopied={} of {}",
                    current.tmp_name,
                    current.bytes_copied(),
                    current.bytes_to_copy
                ));
                already_copied = already_copied.saturating_add(current.bytes_copied());
                mine.tot_bytes = mine.tot_bytes.saturating_add(current.meta_data.length);
                mine.current = Some(current);
            } else {
                self.dest
                    .message(&format!("xfer: file {file_name} will be fully copied"));
                keep.push((file_name, meta));
            }
        }
        mine.to_copy = keep;
        self.dest.message(&format!(
            "xfer: {already_copied} bytes already copied of {}",
            mine.tot_bytes
        ));
        self.dest.message(&format!(
            "xfer: now delete old temp files: {:?}",
            theirs.copied_files.values().collect::<Vec<_>>()
        ));
        for tmp in theirs.copied_files.values() {
            let _ = self.dest.dir().delete_file(tmp);
        }
        theirs.copied_files.clear();
        if let Some(mut current) = theirs.current.take() {
            let _ = current.close();
            if self.dest.is_verbose_files() {
                self.dest
                    .message(&format!("remove partial file {}", current.tmp_name));
            }
            self.dest.deleter.force_delete_file(&current.tmp_name)?;
        }
        Ok(())
    }
}

/// `CopyJob`: copies the files of one NRT point (or of a merged segment
/// being warmed) from the primary.
pub trait CopyJob: Send + Sync {
    /// The shared fields.
    fn core(&self) -> &CopyJobCore;

    /// `start()`: asks the primary for the files.
    ///
    /// # Errors
    /// The primary could not be reached.
    fn start(&self) -> Result<()>;

    /// `runBlocking()`: copies everything on this thread.
    ///
    /// # Errors
    /// The copy failed or was cancelled.
    fn run_blocking(&self) -> Result<()>;

    /// One step of the copy (`SimpleCopyJob.visit()`): `true` once the job
    /// is done or failed.
    ///
    /// # Errors
    /// The stream failed or a checksum disagrees.
    fn visit(&self) -> Result<bool>;

    /// Closes the connection to the primary, if the job has one.
    fn close(&self) {}

    /// The job as its log lines name it (`toString()`).
    fn description(&self) -> String {
        format!("CopyJob(ord={} {})", self.core().ord, self.core().reason)
    }

    /// `finish()`: renames the copied temporary files into place.
    ///
    /// # Errors
    /// A rename failed.
    fn finish(&self) -> Result<()>;

    /// `conflicts(other)`: whether both copy a file of the same name.
    fn conflicts(&self, other: &dyn CopyJob) -> bool {
        let mine: BTreeSet<String> = self
            .core()
            .state()
            .to_copy
            .iter()
            .map(|(n, _)| n.clone())
            .collect();
        let theirs = other.core().state();
        theirs.to_copy.iter().any(|(n, _)| mine.contains(n))
    }

    /// `getCopyState()`.
    fn copy_state(&self) -> Option<&CopyState>;

    /// `cancel(reason, exc)`; a job holding a connection closes it here.
    ///
    /// # Errors
    /// Deleting the partial file failed.
    fn cancel(&self, reason: &str, exc: Option<&str>) -> Result<()> {
        self.core().cancel(reason, exc)
    }

    /// `transferAndCancel(prevJob)`.
    ///
    /// # Errors
    /// Deleting `prev`'s leftovers failed.
    fn transfer_and_cancel(&self, prev: &dyn CopyJob) -> Result<()> {
        self.core().transfer_and_cancel(prev.core())
    }

    /// `getFailed()`.
    fn failed(&self) -> bool {
        self.core().state().exc.is_some()
    }

    /// `getTotalBytesCopied()`.
    fn total_bytes_copied(&self) -> u64 {
        self.core().state().tot_bytes_copied
    }

    /// `getFileNamesToCopy()`.
    fn file_names_to_copy(&self) -> BTreeSet<String> {
        self.core()
            .state()
            .to_copy
            .iter()
            .map(|(n, _)| n.clone())
            .collect()
    }

    /// `getFileNames()`: every file of the point.
    fn file_names(&self) -> BTreeSet<String> {
        let core = self.core();
        match (&core.files, self.copy_state()) {
            (Some(files), _) => files.keys().cloned().collect(),
            (None, Some(cs)) => cs.files.keys().cloned().collect(),
            (None, None) => BTreeSet::new(),
        }
    }
}

/// `CopyJob.compareTo`: high priority first, then the older job.
pub fn compare_jobs(a: &CopyJobCore, b: &CopyJobCore) -> std::cmp::Ordering {
    b.high_priority
        .cmp(&a.high_priority)
        .then(a.ord.cmp(&b.ord))
}

#[cfg(test)]
mod tests;
