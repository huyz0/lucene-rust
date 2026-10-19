//! `Node`: what the primary and the replica share -- the directory, the
//! state string, the log and the metadata of the local files.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use lucene_store::codec_util;
use lucene_store::directory::Directory;

use super::types::FileMetaData;

/// `Node.PRIMARY_GEN_KEY`: the commit-data key naming the primary's
/// generation.
pub const PRIMARY_GEN_KEY: &str = "__primaryGen";
/// `Node.VERSION_KEY`: the commit-data key naming the NRT version.
pub const VERSION_KEY: &str = "__version";

/// Where a node's log lines go: `PrintStream`.
pub type Printer = Arc<dyn Fn(&str) + Send + Sync>;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

static LOCAL_START: OnceLock<Instant> = OnceLock::new();
static GLOBAL_START: Mutex<Option<Instant>> = Mutex::new(None);

/// `Node.globalStartNS = System.nanoTime()`: the instant a group of nodes
/// started, which their log lines count from.
pub fn set_global_start(start: Instant) {
    *lock(&GLOBAL_START) = Some(start);
}

fn secs_since(start: Instant) -> f64 {
    Instant::now()
        .saturating_duration_since(start)
        .as_secs_f64()
}

fn local_start() -> Instant {
    *LOCAL_START.get_or_init(Instant::now)
}

fn global_start() -> Instant {
    lock(&GLOBAL_START).unwrap_or_else(local_start)
}

fn thread_name() -> String {
    std::thread::current()
        .name()
        .unwrap_or("unnamed")
        .to_string()
}

/// `Node.nodeMessage(printStream, message)`.
pub fn node_message(printer: Option<&Printer>, message: &str) {
    if let Some(p) = printer {
        p(&format!(
            // SPACES-OK: Java's column layout.
            "{:5.3}s {:5.1}s:           [{:>11}] {}",
            secs_since(global_start()),
            secs_since(local_start()),
            thread_name(),
            message
        ));
    }
}

/// `Node.nodeMessage(printStream, id, message)`.
pub fn node_message_id(printer: Option<&Printer>, id: i32, message: &str) {
    if let Some(p) = printer {
        p(&format!(
            // SPACES-OK: Java's column layout.
            "{:5.3}s {:5.1}s:         N{} [{:>11}] {}",
            secs_since(global_start()),
            secs_since(local_start()),
            id,
            thread_name(),
            message
        ));
    }
}

/// `Node.bytesToString`.
pub fn bytes_to_string(bytes: u64) -> String {
    let b = bytes as f64;
    if bytes < 1024 {
        format!("{bytes} b")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", b / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", b / 1024.0 / 1024.0)
    } else {
        format!("{:.1} GB", b / 1024.0 / 1024.0 / 1024.0)
    }
}

/// The state `Node` keeps for both kinds of node.
pub struct NodeCore {
    pub(crate) id: i32,
    primary: bool,
    pub(crate) dir: Arc<dyn Directory>,
    printer: Option<Printer>,
    verbose_files: AtomicBool,
    state: Mutex<String>,
    last_file_meta_data: Mutex<Option<Arc<BTreeMap<String, FileMetaData>>>>,
}

impl NodeCore {
    /// `new Node(id, dir, ...)`; `primary` stands for `this instanceof
    /// PrimaryNode` in `name()`.
    pub fn new(id: i32, primary: bool, dir: Arc<dyn Directory>, printer: Option<Printer>) -> Self {
        Self {
            id,
            primary,
            dir,
            printer,
            verbose_files: AtomicBool::new(true),
            state: Mutex::new("idle".to_string()),
            last_file_meta_data: Mutex::new(None),
        }
    }

    /// `getDirectory()`.
    pub fn directory(&self) -> &Arc<dyn Directory> {
        &self.dir
    }

    /// `name()`: `P<id>` or `R<id>`.
    pub fn name(&self) -> String {
        format!("{}{}", if self.primary { 'P' } else { 'R' }, self.id)
    }

    /// The node's state string (`idle`, `syncing`, `closing`, ...).
    pub fn state(&self) -> String {
        lock(&self.state).clone()
    }

    pub(crate) fn set_state(&self, state: &str) {
        *lock(&self.state) = state.to_string();
    }

    /// `message(text)`.
    pub fn message(&self, message: &str) {
        if let Some(p) = &self.printer {
            p(&format!(
                "{:5.3}s {:5.1}s: {:>7} {:>2} [{:>11}] {}",
                secs_since(global_start()),
                secs_since(local_start()),
                self.state(),
                self.name(),
                thread_name(),
                message
            ));
        }
    }

    /// `isVerboseFiles()`.
    pub fn is_verbose_files(&self) -> bool {
        self.verbose_files.load(Ordering::Relaxed)
    }

    /// `setVerboseFiles(verbose)`.
    pub fn set_verbose_files(&self, verbose: bool) {
        self.verbose_files.store(verbose, Ordering::Relaxed);
    }

    /// `lastFileMetaData`: the metadata of the files of the last NRT point.
    pub fn last_file_meta_data(&self) -> Option<Arc<BTreeMap<String, FileMetaData>>> {
        lock(&self.last_file_meta_data).clone()
    }

    pub(crate) fn set_last_file_meta_data(
        &self,
        files: Option<Arc<BTreeMap<String, FileMetaData>>>,
    ) {
        *lock(&self.last_file_meta_data) = files;
    }

    /// `readLocalFileMetaData(fileName)`: from the last NRT point when the
    /// file is in it, else read off the file; `None` when the file is
    /// missing or its header or footer is damaged (the caller then copies
    /// it).
    ///
    /// # Errors
    /// An I/O failure other than a missing file.
    pub fn read_local_file_metadata(&self, file_name: &str) -> crate::Result<Option<FileMetaData>> {
        if let Some(cache) = self.last_file_meta_data() {
            if let Some(found) = cache.get(file_name) {
                return Ok(Some(found.clone()));
            }
        }
        let input = match self.dir.open(file_name) {
            Ok(input) => input,
            Err(lucene_store::Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                if self.is_verbose_files() {
                    self.message(&format!(
                        "file {file_name}: will copy [file does not exist]"
                    ));
                }
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        };
        let buf: &[u8] = &input;
        let length = buf.len() as u64;
        let parsed = index_header(buf).and_then(|header| {
            let checksum = codec_util::retrieve_checksum(buf).ok()?;
            let footer = buf.get(buf.len().checked_sub(codec_util::FOOTER_LENGTH)?..)?;
            Some((header.to_vec(), footer.to_vec(), checksum))
        });
        let Some((header, footer, checksum)) = parsed else {
            if self.is_verbose_files() {
                self.message(&format!(
                    "file {file_name}: will copy [existing file is corrupt]"
                ));
            }
            return Ok(None);
        };
        if self.is_verbose_files() {
            self.message(&format!(
                "file {file_name} has length={}",
                bytes_to_string(length)
            ));
        }
        Ok(Some(FileMetaData {
            header,
            footer,
            length,
            checksum,
        }))
    }
}

/// The bytes `CodecUtil.readIndexHeader` reads off the start of `buf`: the
/// magic, the codec name, the version, the 16-byte id and the suffix; `None`
/// when the file is too short or does not start with the magic.
fn index_header(buf: &[u8]) -> Option<&[u8]> {
    let magic = u32::from_be_bytes(buf.get(..4)?.try_into().ok()?);
    if magic != codec_util::CODEC_MAGIC {
        return None;
    }
    // Codec names are ASCII under 128 bytes, so the vint length is one byte.
    let name_len = usize::from(*buf.get(4)?);
    let after_name = 5usize.checked_add(name_len)?;
    let after_version = after_name.checked_add(4)?;
    let after_id = after_version.checked_add(codec_util::ID_LENGTH)?;
    let suffix_len = usize::from(*buf.get(after_id)?);
    let end = after_id.checked_add(1)?.checked_add(suffix_len)?;
    buf.get(..end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::{ByteBuffersDirectory, DataOutput};

    fn core(dir: Arc<ByteBuffersDirectory>, log: &Arc<Mutex<Vec<String>>>) -> NodeCore {
        let log = Arc::clone(log);
        NodeCore::new(
            3,
            false,
            dir,
            Some(Arc::new(move |s: &str| lock(&log).push(s.to_string()))),
        )
    }

    #[test]
    fn bytes_to_string_picks_the_unit() {
        assert_eq!(bytes_to_string(1023), "1023 b");
        assert_eq!(bytes_to_string(1536), "1.5 KB");
        assert_eq!(bytes_to_string(3 * 1024 * 1024), "3.0 MB");
        assert_eq!(bytes_to_string(5 * 1024 * 1024 * 1024), "5.0 GB");
    }

    #[test]
    fn index_header_ends_after_the_suffix() {
        let mut out = Vec::new();
        codec_util::write_index_header(&mut out, "Codec", 7, &[9u8; 16], "sfx");
        out.extend_from_slice(b"payload");
        assert_eq!(index_header(&out).unwrap().len(), out.len() - 7);
        assert!(index_header(&out[..10]).is_none());
        assert!(index_header(b"not a codec file at all").is_none());
    }

    #[test]
    fn local_metadata_reads_header_footer_and_checksum() {
        let dir = Arc::new(ByteBuffersDirectory::new());
        let mut buf = Vec::new();
        codec_util::write_index_header(&mut buf, "Codec", 1, &[1u8; 16], "");
        buf.extend_from_slice(b"hello");
        codec_util::write_footer(&mut buf);
        let mut out = dir.create_output("_0.dat").unwrap();
        out.write_bytes(&buf);
        out.close().unwrap();
        let mut junk = dir.create_output("junk").unwrap();
        junk.write_bytes(b"short");
        junk.close().unwrap();

        let log = Arc::new(Mutex::new(Vec::new()));
        let node = core(dir, &log);
        let meta = node.read_local_file_metadata("_0.dat").unwrap().unwrap();
        assert_eq!(meta.length, buf.len() as u64);
        assert_eq!(meta.header, buf[..buf.len() - 5 - 16]);
        assert_eq!(meta.footer, buf[buf.len() - 16..]);
        assert_eq!(meta.checksum, codec_util::retrieve_checksum(&buf).unwrap());
        assert_eq!(
            format!("{meta}"),
            format!(
                "FileMetaData(length={} checksum={})",
                meta.length, meta.checksum
            )
        );

        assert!(node.read_local_file_metadata("junk").unwrap().is_none());
        assert!(node.read_local_file_metadata("absent").unwrap().is_none());
        let lines = lock(&log).clone();
        assert!(lines
            .iter()
            .any(|l| l.contains("file junk: will copy [existing file is corrupt]")));
        assert!(lines
            .iter()
            .any(|l| l.contains("file absent: will copy [file does not exist]")));
        assert!(lines
            .iter()
            .any(|l| l.contains(" R3 ") && l.contains("file _0.dat has length=")));

        // The last NRT point's metadata wins over the file.
        let mut cache = BTreeMap::new();
        cache.insert("absent".to_string(), meta.clone());
        node.set_last_file_meta_data(Some(Arc::new(cache)));
        assert_eq!(node.read_local_file_metadata("absent").unwrap(), Some(meta));
    }

    #[test]
    fn names_states_and_messages() {
        let dir = Arc::new(ByteBuffersDirectory::new());
        let log = Arc::new(Mutex::new(Vec::new()));
        let node = core(dir, &log);
        assert_eq!(node.name(), "R3");
        assert_eq!(node.state(), "idle");
        node.set_state("syncing");
        node.set_verbose_files(false);
        assert!(!node.is_verbose_files());
        node.message("hello");
        let printer: Printer = {
            let log = Arc::clone(&log);
            Arc::new(move |s: &str| lock(&log).push(s.to_string()))
        };
        node_message(Some(&printer), "free");
        node_message_id(Some(&printer), 4, "with id");
        node_message(None, "dropped");
        let lines = lock(&log).clone();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("s: syncing R3 ["));
        assert!(lines[0].ends_with("] hello"));
        assert!(lines[1].ends_with("] free"));
        assert!(lines[2].contains("N4 [") && lines[2].ends_with("] with id"));
        assert_eq!(NodeCore::new(1, true, node.dir.clone(), None).name(), "P1");
    }
}
