//! `IndexWriter` with `iwc.setCodec(new SimpleTextCodec())`: a
//! [`Directory`] that an [`IndexWriter`](lucene_index::index_writer::IndexWriter)
//! opened with codec `Lucene104` writes into, and that publishes every
//! commit to the directory it wraps as a SimpleText index.
//!
//! The writer's own files (segments, live docs, updates, `segments_N`) stay
//! in memory, where the writer reads them back for merges, deletes and
//! updates as it reads any `Lucene104` segment. When it publishes a commit
//! (renames `pending_segments_N` to `segments_N`), each segment the commit
//! names is transcoded once into its SimpleText base files
//! ([`to_simpletext`] over [`base_commit`]) and each new generation into its
//! `.liv`/`.inf`/`.dat` ([`generation_files`]); then the commit itself is
//! written to the wrapped directory with every segment's codec
//! `SimpleText`. Deleting a `segments_N` (the writer's deletion policy)
//! deletes the published one and every SimpleText file no remaining commit
//! names, as Java's `IndexFileDeleter` would.
//!
//! An index the wrapped directory already holds is opened as Java's writer
//! opens it: its latest commit is mirrored into memory first, each
//! SimpleText segment re-encoded natively ([`to_native`]) at its own
//! generations, and its files counted as published.
//!
//! What lands in the wrapped directory is what Java's `IndexWriter` with
//! `SimpleTextCodec` writes for the same segments; the in-memory mirror is
//! this port's structural difference. Java's writer, opening an index,
//! also deletes the files no commit it keeps references (older commits, a
//! crashed session's leftovers); this leaves those in the wrapped directory.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Mutex;

use lucene_index::segment_info;
use lucene_index::segment_infos::{self, SegmentInfos};
use lucene_store::directory::{generation_from_segments_file_name, Input};
use lucene_store::DataOutput;
use lucene_store::{ByteBuffersDirectory, Directory, FsIndexOutput, Lock};

use crate::simpletext_native::to_native;
use crate::simpletext_transcode::{base_commit, generation_files, to_simpletext};

/// The codec name the published commit gives every segment.
pub const CODEC_NAME: &str = "SimpleText";

type StoreResult<T> = lucene_store::Result<T>;

fn store_error(e: impl std::fmt::Display) -> lucene_store::Error {
    lucene_store::Error::Io(std::io::Error::other(e.to_string()))
}

/// What the wrapped directory holds that this directory wrote.
#[derive(Debug, Default)]
struct Published {
    /// Every file written, by name.
    files: BTreeSet<String>,
    /// Each segment's base files (its `.si` included), by segment name.
    bases: BTreeMap<String, Vec<String>>,
    /// Each published commit's files (its `segments_N` included), by
    /// generation.
    commits: BTreeMap<i64, BTreeSet<String>>,
}

/// See the [module docs](self).
pub struct SimpleTextDirectory<'d> {
    real: &'d dyn Directory,
    native: ByteBuffersDirectory,
    published: Mutex<Published>,
}

impl std::fmt::Debug for SimpleTextDirectory<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimpleTextDirectory")
            .field("native", &self.native)
            .field("published", &self.published)
            .finish_non_exhaustive()
    }
}

impl<'d> SimpleTextDirectory<'d> {
    /// Publishes into `real`. An index `real` already holds (its latest
    /// commit, every segment SimpleText) is mirrored first, so the writer
    /// opens it and appends: each segment re-encoded natively
    /// ([`to_native`]) and its files counted as published, so the next
    /// commit names them as they are and the writer's deletion policy
    /// retires them as Java's would.
    ///
    /// # Errors
    /// Listing or reading `real` fails, a segment of its latest commit is
    /// not SimpleText, or one does not re-encode.
    pub fn new(real: &'d dyn Directory) -> crate::Result<Self> {
        let dir = SimpleTextDirectory {
            real,
            native: ByteBuffersDirectory::new(),
            published: Mutex::new(Published::default()),
        };
        if real.list_all()?.iter().any(|f| is_commit(f)) {
            dir.mirror()?;
        }
        Ok(dir)
    }

    /// Mirrors `real`'s latest commit into the writer's directory.
    fn mirror(&self) -> crate::Result<()> {
        let mut infos =
            segment_infos::read_latest(self.real).map_err(crate::directory_reader::Error::from)?;
        let mut published = self.lock();
        let mut referenced = BTreeSet::new();
        for sci in &mut infos.segments {
            if sci.codec_name != CODEC_NAME {
                return Err(crate::Error::IllegalArgument(format!(
                    "segment {} is {}, not {CODEC_NAME}",
                    sci.segment_name, sci.codec_name
                )));
            }
            let si_name = format!("{}.si", sci.segment_name);
            let info = segment_info::parse_simpletext(
                &self
                    .real
                    .open(&si_name)
                    .map_err(crate::directory_reader::Error::from)?,
                &sci.segment_name,
                &sci.segment_id,
            )
            .map_err(crate::directory_reader::Error::from)?;
            let mut base = info.files.clone();
            if !base.contains(&si_name) {
                base.push(si_name);
            }
            referenced.extend(sci.files(&base));
            published.bases.insert(sci.segment_name.clone(), base);
            *sci = to_native(self.real, sci, &self.native)?;
        }
        segment_infos::write(&infos, &self.native).map_err(crate::directory_reader::Error::from)?;
        referenced.extend(lucene_store::directory::segments_file_name(
            infos.generation,
        ));
        published.files.extend(referenced.iter().cloned());
        published.commits.insert(infos.generation, referenced);
        Ok(())
    }

    /// The directory commits are published to.
    pub fn published(&self) -> &'d dyn Directory {
        self.real
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Published> {
        self.published
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn write_real(&self, name: &str, bytes: &[u8]) -> StoreResult<()> {
        let mut out = self.real.create_output(name)?;
        out.write_bytes(bytes);
        out.close()?;
        Ok(())
    }

    /// Writes the commit the writer just published as `segments_name` to
    /// the wrapped directory.
    fn publish(&self, segments_name: &str) -> StoreResult<()> {
        let generation = generation_from_segments_file_name(segments_name)?;
        let mut infos: SegmentInfos =
            segment_infos::parse(&self.native.open(segments_name)?, generation)
                .map_err(store_error)?;
        let mut published = self.lock();
        let mut referenced = BTreeSet::new();
        let mut fresh = Vec::new();
        for sci in &mut infos.segments {
            let si_name = format!("{}.si", sci.segment_name);
            if !published.bases.contains_key(&sci.segment_name) {
                let info = segment_info::parse_for_codec(
                    &self.native.open(&si_name)?,
                    &sci.segment_id,
                    &sci.codec_name,
                )
                .map_err(store_error)?;
                let segment =
                    to_simpletext(&self.native, &base_commit(sci), &info).map_err(store_error)?;
                for (file, bytes) in &segment.files {
                    self.write_real(file, bytes)?;
                    fresh.push(file.clone());
                }
                self.write_real(&si_name, &segment.si)?;
                fresh.push(si_name.clone());
                let mut files: Vec<String> = segment.files.into_iter().map(|(f, _)| f).collect();
                files.push(si_name);
                published.files.extend(files.iter().cloned());
                published.bases.insert(sci.segment_name.clone(), files);
            }
            if let Some(files) = published.bases.get(&sci.segment_name) {
                referenced.extend(files.iter().cloned());
            }

            let known = |f: &str| published.files.contains(f);
            let generations = generation_files(&self.native, sci, &known).map_err(store_error)?;
            for (file, bytes) in &generations.files {
                if let Some(bytes) = bytes {
                    self.write_real(file, bytes)?;
                    fresh.push(file.clone());
                }
                referenced.insert(file.clone());
            }
            published
                .files
                .extend(generations.files.into_iter().map(|(f, _)| f));
            sci.field_infos_files = generations.field_infos_files;
            sci.dv_update_files = generations.dv_update_files;
            sci.codec_name = CODEC_NAME.to_string();
        }
        self.real.sync(&fresh)?;
        let written = segment_infos::write(&infos, self.real).map_err(store_error)?;
        referenced.insert(written.clone());
        published.files.insert(written);
        published.commits.insert(generation, referenced);
        Ok(())
    }

    /// The writer deleted `segments_name`: delete the published commit and
    /// every file no other published commit names.
    fn retire(&self, segments_name: &str) -> StoreResult<()> {
        let generation = generation_from_segments_file_name(segments_name)?;
        let mut published = self.lock();
        if published.commits.remove(&generation).is_none() {
            return Ok(());
        }
        let live: BTreeSet<&String> = published.commits.values().flatten().collect();
        let dead: Vec<String> = published
            .files
            .iter()
            .filter(|f| !live.contains(f))
            .cloned()
            .collect();
        for file in &dead {
            self.real.delete_file(file)?;
            published.files.remove(file);
        }
        let files = std::mem::take(&mut published.files);
        published
            .bases
            .retain(|segment, _| files.contains(&format!("{segment}.si")));
        published.files = files;
        Ok(())
    }
}

fn is_commit(name: &str) -> bool {
    name.starts_with("segments_")
}

impl Directory for SimpleTextDirectory<'_> {
    fn list_all(&self) -> StoreResult<Vec<String>> {
        self.native.list_all()
    }

    fn open(&self, name: &str) -> StoreResult<Input> {
        self.native.open(name)
    }

    fn file_length(&self, name: &str) -> StoreResult<u64> {
        self.native.file_length(name)
    }

    fn create_output(&self, name: &str) -> StoreResult<FsIndexOutput> {
        self.native.create_output(name)
    }

    fn create_temp_output(&self, prefix: &str, suffix: &str) -> StoreResult<FsIndexOutput> {
        self.native.create_temp_output(prefix, suffix)
    }

    fn sync(&self, names: &[String]) -> StoreResult<()> {
        self.native.sync(names)
    }

    fn rename(&self, source: &str, dest: &str) -> StoreResult<()> {
        self.native.rename(source, dest)?;
        if is_commit(dest) {
            self.publish(dest)?;
        }
        Ok(())
    }

    fn delete_file(&self, name: &str) -> StoreResult<()> {
        self.native.delete_file(name)?;
        if is_commit(name) {
            self.retire(name)?;
        }
        Ok(())
    }

    fn sync_meta_data(&self) -> StoreResult<()> {
        self.native.sync_meta_data()?;
        self.real.sync_meta_data()
    }

    fn obtain_lock(&self, name: &str) -> StoreResult<Box<dyn Lock>> {
        self.real.obtain_lock(name)
    }

    fn pending_deletions(&self) -> StoreResult<BTreeSet<String>> {
        self.native.pending_deletions()
    }

    fn fs_directory_path(&self) -> Option<&Path> {
        None
    }
}
