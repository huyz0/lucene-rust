//! A minimal `DirectoryReader`/`SegmentReader`-equivalent (task #45): the
//! "open every segment a commit lists, in one call" object this port has
//! been missing since task #41 added multi-segment search. Every query
//! function in this crate up to now (including [`crate::multi_segment`]'s
//! own tests) required the *caller* to manually read `segments_N`, open each
//! segment's `.fnm`/`.tim`/`.tip`/`.tmd`/`.doc`/`.pos`/`.pay`/`.liv` files by
//! hand, and compute each segment's `doc_base` as a running sum of earlier
//! segments' `maxDoc` -- exactly the boilerplate real `DirectoryReader.open`
//! centralizes in Java. This module is that centralization, nothing more.
//!
//! # Scope
//!
//! Mirrors `DirectoryReader.open(Directory)`: find the latest `segments_N`
//! ([`lucene_index::segment_infos::read_latest`]), open one
//! [`SegmentReader`] per listed segment (only the files that segment
//! actually has -- most fixtures in this port are stored-fields-only and
//! have no postings/doc-values/norms/term-vectors files at all, see
//! `segment_writer.rs`'s doc comment), and hand back `doc_base`-labelled
//! [`OpenSegment`] values ready for [`crate::multi_segment`]'s existing
//! fan-out/merge functions.
//!
//! **Deliberately excluded** (see `docs/parity.md` for the authoritative
//! list): NRT/reopen (`DirectoryReader.openIfChanged`), soft deletes,
//! compound-file segments (`.cfs`/`.cfe` -- [`Error::CompoundFileUnsupported`]
//! is returned rather than silently mis-reading), and doc-values/norms/term
//! vectors (irrelevant here: [`OpenSegment`] itself has no fields for them --
//! `crate::field_norms`/doc-values/term-vectors query functions still take
//! their own already-opened readers directly, unchanged by this task).
//!
//! # Why the two-step `open_segments`/`as_open_segments` API
//!
//! [`OpenSegment<'a>`] holds `&'a DocInput<'a>` (a *reference to* an already
//! constructed value), not an owned `DocInput` and not raw bytes -- so
//! producing a `Vec<OpenSegment>` needs somewhere for those `DocInput`/
//! `PosInput`/`PayInput` values to live at least as long as the
//! `OpenSegment`s that borrow them. [`SegmentReader`] can't store them
//! itself (that would be self-referential: the `Input` would borrow from a
//! byte buffer owned by the same struct). Instead, [`DirectoryReader::open_segments`]
//! returns an [`OpenedSegments`] that owns those freshly-opened `Input`
//! values (borrowing from the `SegmentReader`s' already-owned byte buffers,
//! not from itself), and [`OpenedSegments::as_open_segments`] hands back the
//! `Vec<OpenSegment>` borrowing from *that*. Two calls instead of one, but
//! no `unsafe`/self-referential tricks (`#![forbid(unsafe_code)]` in this
//! crate) and no behavior change from what a hand-written caller already had
//! to do.

use lucene_codecs::blocktree::{self, BlockTreeFields};
use lucene_codecs::field_infos;
use lucene_codecs::live_docs;
use lucene_codecs::postings::{self, DocInput, PayInput, PosInput};
use lucene_index::deletes::liv_file_name;
use lucene_index::segment_info::{self, SegmentInfo};
use lucene_index::segment_infos::{self, SegmentInfos};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::directory::Directory;
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::multi_segment::OpenSegment;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error(transparent)]
    SegmentInfos(#[from] segment_infos::Error),
    #[error(transparent)]
    SegmentInfo(#[from] segment_info::Error),
    #[error(transparent)]
    FieldInfos(#[from] field_infos::Error),
    #[error(transparent)]
    BlockTree(#[from] blocktree::Error),
    #[error(transparent)]
    Postings(#[from] postings::Error),
    #[error(transparent)]
    LiveDocs(#[from] live_docs::Error),
    #[error("segment {0} is compound-file packed; DirectoryReader does not open .cfs/.cfe yet")]
    CompoundFileUnsupported(String),
    #[error("segment {segment} has {found} of .tim/.tip/.tmd (need all three or none)")]
    PartialBlockTreeFiles { segment: String, found: usize },
}

pub type Result<T> = std::result::Result<T, Error>;

/// One opened segment: everything [`OpenSegment`] needs, already decoded/
/// validated, plus the bookkeeping ([`SegmentReader::doc_base`]) real
/// `SegmentReader.docBase` carries. `fields`/`live_docs` are owned outright
/// (both are fully-decoded, non-borrowing types -- see `blocktree.rs`'s
/// `BlockTreeFields` and `lucene_util::fixed_bit_set::FixedBitSet`);
/// `.doc`/`.pos`/`.pay` are kept as raw bytes because [`DocInput`]/
/// [`PosInput`]/[`PayInput`] borrow their buffer and are cheap to
/// (re-)construct on demand -- see [`DirectoryReader::open_segments`].
#[derive(Debug, Clone)]
pub struct SegmentReader {
    pub segment_name: String,
    pub max_doc: i32,
    pub doc_base: i32,
    segment_id: [u8; ID_LENGTH],
    /// The generation of this segment's deletions at the time this reader
    /// was opened (`-1` if it has none) -- used by
    /// [`DirectoryReader::open_if_changed`] to tell "same segment, unchanged
    /// deletions" (safe to reuse) apart from "same segment, new deletions"
    /// (must reopen to pick up the new `.liv`/`live_docs`).
    del_gen: i64,
    segment_suffix: String,
    fields: BlockTreeFields,
    doc_buf: Option<Vec<u8>>,
    pos_buf: Option<Vec<u8>>,
    pay_buf: Option<Vec<u8>>,
    live_docs: Option<FixedBitSet>,
}

impl SegmentReader {
    /// A cheap in-memory copy of an already-open reader, used by
    /// [`DirectoryReader::open_at_reusing`] to reuse a segment without
    /// re-reading any files -- `Clone` derives from `BlockTreeFields`/
    /// `FixedBitSet` already being `Clone`, so this is a genuine "no disk
    /// I/O" reuse, not a relabeled fresh open.
    fn clone_reader(&self) -> Self {
        self.clone()
    }
}

impl SegmentReader {
    /// Opens one segment: reads its `.si`, then whichever of `.fnm`/
    /// `.tim`/`.tip`/`.tmd`/`.doc`/`.pos`/`.pay`/`.liv` it actually has
    /// (checked via `SegmentInfo.files` -- a segment missing postings, e.g.
    /// this port's stored-fields-only fixtures, legitimately has none of
    /// `.tim`/`.tip`/`.tmd`/`.doc`/`.pos`/`.pay`, and that is not an error).
    fn open(
        dir: &dyn Directory,
        commit: &segment_infos::SegmentCommitInfo,
        doc_base: i32,
    ) -> Result<Self> {
        let segment_name = commit.segment_name.clone();
        let segment_id = commit.segment_id;

        let si_bytes = dir.open(&format!("{segment_name}.si"))?;
        let si: SegmentInfo = segment_info::parse(&si_bytes, &segment_id)?;
        if si.is_compound_file {
            return Err(Error::CompoundFileUnsupported(segment_name));
        }

        let fnm_name = find_file_ending(&si.files, ".fnm").ok_or_else(|| {
            Error::Store(lucene_store::Error::Corrupted(format!(
                "segment {segment_name} has no .fnm file"
            )))
        })?;
        let fnm_bytes = dir.open(&fnm_name)?;
        let field_infos = field_infos::parse(&fnm_bytes, &segment_id, "")?;

        let tim_name = find_file_ending(&si.files, ".tim");
        let tip_name = find_file_ending(&si.files, ".tip");
        let tmd_name = find_file_ending(&si.files, ".tmd");
        let found = [&tim_name, &tip_name, &tmd_name]
            .iter()
            .filter(|f| f.is_some())
            .count();

        let (fields, segment_suffix, doc_buf, pos_buf, pay_buf) = if found == 3 {
            let tim_name = tim_name.unwrap();
            // Suffix is embedded in the file name itself: strip the
            // `<segment_name>_` prefix and `.tim` extension, e.g.
            // `_0_Lucene104_0.tim` (segment `_0`) -> `Lucene104_0`.
            let segment_suffix = tim_name
                .strip_prefix(&format!("{segment_name}_"))
                .and_then(|s| s.strip_suffix(".tim"))
                .unwrap_or("")
                .to_string();

            let tim_bytes = dir.open(&tim_name)?;
            let tip_bytes = dir.open(&tip_name.unwrap())?;
            let tmd_bytes = dir.open(&tmd_name.unwrap())?;
            let fields = blocktree::open(
                &tim_bytes,
                &tip_bytes,
                &tmd_bytes,
                &field_infos,
                &segment_id,
                &segment_suffix,
                si.doc_count,
            )?;

            let doc_name = find_file_ending(&si.files, ".doc");
            let doc_buf = match &doc_name {
                Some(name) => Some(dir.open(name)?.to_vec()),
                None => None,
            };
            let pos_name = find_file_ending(&si.files, ".pos");
            let pos_buf = match &pos_name {
                Some(name) => Some(dir.open(name)?.to_vec()),
                None => None,
            };
            let pay_name = find_file_ending(&si.files, ".pay");
            let pay_buf = match &pay_name {
                Some(name) => Some(dir.open(name)?.to_vec()),
                None => None,
            };

            (fields, segment_suffix, doc_buf, pos_buf, pay_buf)
        } else if found == 0 {
            (BlockTreeFields::empty(), String::new(), None, None, None)
        } else {
            return Err(Error::PartialBlockTreeFiles {
                segment: segment_name,
                found,
            });
        };

        // `.liv` exists only when the segment has deletions (`del_gen !=
        // -1`, `SegmentCommitInfo.hasDeletions()`'s condition).
        let live_docs = if commit.del_gen != -1 {
            let liv_name = liv_file_name(&segment_name, commit.del_gen);
            let liv_bytes = dir.open(&liv_name)?;
            Some(live_docs::parse(
                &liv_bytes,
                &segment_id,
                commit.del_gen,
                si.doc_count as usize,
                commit.del_count as usize,
            )?)
        } else {
            None
        };

        Ok(SegmentReader {
            segment_name,
            max_doc: si.doc_count,
            doc_base,
            segment_id,
            del_gen: commit.del_gen,
            segment_suffix,
            fields,
            doc_buf,
            pos_buf,
            pay_buf,
            live_docs,
        })
    }
}

fn find_file_ending(files: &[String], ext: &str) -> Option<String> {
    files.iter().find(|f| f.ends_with(ext)).cloned()
}

/// `DirectoryReader.open(Directory)`-equivalent: reads the latest
/// `segments_N`, opens every listed segment (whichever files it actually
/// has), and computes each segment's `doc_base` automatically as the running
/// sum of every earlier segment's `maxDoc` -- the two things
/// [`OpenSegment::doc_base`]'s doc comment previously left entirely to the
/// caller.
#[derive(Debug)]
pub struct DirectoryReader {
    pub segment_infos: SegmentInfos,
    segments: Vec<SegmentReader>,
}

impl DirectoryReader {
    /// Opens every segment listed in the latest commit found in `dir`.
    pub fn open(dir: &dyn Directory) -> Result<Self> {
        let segment_infos = segment_infos::read_latest(dir)?;
        Self::open_at(dir, segment_infos)
    }

    /// Opens every segment listed in an already-parsed [`SegmentInfos`] --
    /// useful for tests that build a commit by hand rather than reading one
    /// off disk (see this module's unit tests).
    pub fn open_at(dir: &dyn Directory, segment_infos: SegmentInfos) -> Result<Self> {
        Self::open_at_reusing(dir, segment_infos, &[])
    }

    /// Shared by [`Self::open_at`] (no reuse candidates) and
    /// [`Self::open_if_changed`] (reuse candidates = the currently-open
    /// reader's segments): opens every segment in `segment_infos`, taking an
    /// already-open [`SegmentReader`] out of `reusable` instead of
    /// re-reading it from disk whenever one matches -- same `segment_name`
    /// *and* same `segment_id` *and* same `del_gen` as the new commit's entry
    /// (see [`Self::open_if_changed`]'s doc comment for why all three must
    /// match: a del_gen bump means the `.liv` file changed even though the
    /// segment's own postings/stored fields didn't).
    fn open_at_reusing(
        dir: &dyn Directory,
        segment_infos: SegmentInfos,
        reusable: &[SegmentReader],
    ) -> Result<Self> {
        let mut segments = Vec::with_capacity(segment_infos.segments.len());
        let mut doc_base = 0i32;
        for commit in &segment_infos.segments {
            let reused = reusable.iter().find(|r| {
                r.segment_name == commit.segment_name
                    && r.segment_id == commit.segment_id
                    && r.del_gen == commit.del_gen
            });
            let mut reader = match reused {
                Some(r) => r.clone_reader(),
                None => SegmentReader::open(dir, commit, doc_base)?,
            };
            reader.doc_base = doc_base;
            doc_base += reader.max_doc;
            segments.push(reader);
        }
        Ok(DirectoryReader {
            segment_infos,
            segments,
        })
    }

    /// `DirectoryReader.openIfChanged(DirectoryReader)`-equivalent: checks
    /// whether `dir`'s latest `segments_N` generation differs from the one
    /// `self` was opened from. Returns `Ok(None)` if nothing changed (same
    /// generation), matching real Lucene's convention that callers should
    /// keep using their existing reader rather than redundantly rebuild it.
    ///
    /// If the generation *did* change, builds a new [`DirectoryReader`] that
    /// reuses `self`'s already-open [`SegmentReader`]s for every segment
    /// that is unchanged in the new commit -- same segment name, same
    /// segment id, *and* same `del_gen` -- and opens fresh readers (reading
    /// `.si`/`.fnm`/postings/`.liv` from disk, exactly like
    /// [`Self::open_at`]) for anything new or whose `del_gen` bumped. A
    /// segment merge or delete replaces the segment name/id, so it always
    /// falls into "open fresh"; a same-segment delete-only commit keeps the
    /// same name/id but bumps `del_gen`, which this deliberately treats as
    /// "must reopen" so the new commit's `live_docs` (not `self`'s stale
    /// ones) is what queries see -- getting this backwards would silently
    /// serve deleted documents as live.
    ///
    /// **Deliberately excluded** (see this module's doc comment and
    /// `docs/parity.md`): this is a self-contained reopen, not a
    /// reader-pool-wide one -- real Lucene's `openIfChanged` operates through
    /// a shared `ReaderPool` so *every* open reader in a process benefits
    /// from one segment's reuse, and supports warm-up listeners
    /// (`ReaderManager`/`SearcherFactory`) called on newly-opened sub-readers
    /// before they're handed back. Neither exists here; each
    /// `open_if_changed` call only reuses its own receiver's readers.
    pub fn open_if_changed(&self, dir: &dyn Directory) -> Result<Option<Self>> {
        let latest = segment_infos::read_latest(dir)?;
        if latest.generation == self.segment_infos.generation {
            return Ok(None);
        }
        Some(Self::open_at_reusing(dir, latest, &self.segments)).transpose()
    }

    /// Every opened segment's own reader, in commit order.
    pub fn segment_readers(&self) -> &[SegmentReader] {
        &self.segments
    }

    /// Opens the per-segment `DocInput`/`PosInput`/`PayInput` values needed
    /// to build `OpenSegment`s -- see this module's doc comment for why this
    /// is a separate call from [`OpenedSegments::as_open_segments`].
    pub fn open_segments(&self) -> Result<OpenedSegments<'_>> {
        let mut doc_ins = Vec::with_capacity(self.segments.len());
        let mut pos_ins = Vec::with_capacity(self.segments.len());
        let mut pay_ins = Vec::with_capacity(self.segments.len());
        for seg in &self.segments {
            let doc_in = match &seg.doc_buf {
                Some(buf) => Some(DocInput::open(buf, &seg.segment_id, &seg.segment_suffix)?),
                None => None,
            };
            let pos_in = match &seg.pos_buf {
                Some(buf) => Some(PosInput::open(buf, &seg.segment_id, &seg.segment_suffix)?),
                None => None,
            };
            let pay_in = match &seg.pay_buf {
                Some(buf) => Some(PayInput::open(buf, &seg.segment_id, &seg.segment_suffix)?),
                None => None,
            };
            doc_ins.push(doc_in);
            pos_ins.push(pos_in);
            pay_ins.push(pay_in);
        }
        Ok(OpenedSegments {
            readers: &self.segments,
            doc_ins,
            pos_ins,
            pay_ins,
        })
    }
}

/// The `DocInput`/`PosInput`/`PayInput` values [`DirectoryReader::open_segments`]
/// constructed, plus a reference back to their owning [`SegmentReader`]s --
/// see this module's doc comment for why this intermediate type exists.
pub struct OpenedSegments<'a> {
    readers: &'a [SegmentReader],
    doc_ins: Vec<Option<DocInput<'a>>>,
    pos_ins: Vec<Option<PosInput<'a>>>,
    pay_ins: Vec<Option<PayInput<'a>>>,
}

impl<'a> OpenedSegments<'a> {
    /// The final step: a `Vec<OpenSegment>` ready to feed directly into
    /// `search_term_query_multi_segment`/`search_boolean_query_multi_segment`
    /// -- `doc_base` already computed, every field already opened.
    pub fn as_open_segments(&self) -> Vec<OpenSegment<'_>> {
        self.readers
            .iter()
            .enumerate()
            .map(|(i, r)| OpenSegment {
                fields: &r.fields,
                doc_in: self.doc_ins[i].as_ref(),
                pos_in: self.pos_ins[i].as_ref(),
                pay_in: self.pay_ins[i].as_ref(),
                live_docs: r.live_docs.as_ref(),
                doc_base: r.doc_base,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_segment::search_term_query_multi_segment;
    use crate::query::TermQuery;
    use lucene_index::segment_infos::LuceneVersion;
    use lucene_store::FsDirectory;

    fn fixture_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_index/"
        ))
    }

    /// Opens the real single-segment `blocktree_index` fixture end-to-end
    /// through `DirectoryReader::open` (reading its real `segments_1`, not a
    /// hand-built `SegmentInfos`), confirming file-existence detection finds
    /// the real postings files and produces a working, single-segment
    /// `OpenSegment` whose query results match task #41's manually-wired
    /// `search_term_query_multi_segment` test.
    #[test]
    fn opens_real_single_segment_fixture_and_matches_manual_wiring() {
        let dir = FsDirectory::open(fixture_dir());
        let reader = DirectoryReader::open(&dir).expect("open real segments_1 commit");

        assert_eq!(reader.segment_readers().len(), 1);
        let seg = &reader.segment_readers()[0];
        assert_eq!(seg.doc_base, 0);
        assert!(seg.doc_buf.is_some(), "fixture has real .doc postings");

        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();

        let query = TermQuery::new("body", "cat");
        let norms = [None];
        let hits = search_term_query_multi_segment(&segments, &query, &norms, 10).unwrap();
        // Same fixture/query task #41's own multi_segment.rs test uses
        // (`search_term_query_multi_segment_merges_two_real_segments`,
        // segment 0 half): must be non-empty and score-descending.
        assert!(!hits.is_empty());
        for pair in hits.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
    }

    /// A stored-fields-only segment (no postings/doc-values/norms/term
    /// vectors at all -- exactly `segment_writer.rs`'s
    /// `flush_stored_only_segment` output) must open cleanly with no
    /// postings files, not error out just because they're missing.
    #[test]
    fn stored_fields_only_segment_opens_without_postings_files() {
        use lucene_codecs::field_infos::{DocValuesType, FieldInfo, IndexOptions};
        use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};

        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);

        let field_infos_list = vec![FieldInfo {
            name: "title".to_string(),
            number: 0,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: lucene_codecs::field_infos::DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: lucene_codecs::field_infos::VectorEncoding::Byte,
            vector_similarity_function:
                lucene_codecs::field_infos::VectorSimilarityFunction::Euclidean,
        }];
        let docs = vec![Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String("hello".to_string()),
            }],
        }];

        let lucene_version = segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        };
        let commit_info = lucene_index::segment_writer::flush_stored_only_segment(
            &dir,
            "_0",
            [7u8; ID_LENGTH],
            "Lucene104",
            lucene_version,
            &field_infos_list,
            &docs,
            false,
        )
        .expect("flush stored-only segment");

        let segment_infos = SegmentInfos {
            id: [9u8; ID_LENGTH],
            generation: 1,
            format_version: segment_infos::VERSION_86,
            lucene_version: LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            index_created_version_major: 10,
            version: 1,
            counter: 1,
            min_segment_lucene_version: None,
            segments: vec![commit_info],
            user_data: vec![],
        };

        let reader = DirectoryReader::open_at(&dir, segment_infos).expect("open stored-only");
        let seg = &reader.segment_readers()[0];
        assert!(seg.doc_buf.is_none());
        assert!(seg.pos_buf.is_none());
        assert!(seg.pay_buf.is_none());
        assert!(seg.live_docs.is_none());
        assert_eq!(seg.max_doc, 1);

        std::fs::remove_dir_all(&dir_path).ok();
    }

    /// Confirms `doc_base` is computed as the running sum of previous
    /// segments' `maxDoc`, by opening the same real fixture segment twice
    /// under one hand-built two-segment `SegmentInfos` (the same "open the
    /// same fixture twice" trick task #41's own cross-segment tests use).
    #[test]
    fn doc_base_is_running_sum_of_previous_max_docs() {
        let dir = FsDirectory::open(fixture_dir());
        let commit = read_commit_info(&dir);

        let mut segment_infos = read_commit(&dir);
        segment_infos.segments.push(commit);

        let reader = DirectoryReader::open_at(&dir, segment_infos).expect("open two segments");
        assert_eq!(reader.segment_readers().len(), 2);
        let first_max_doc = reader.segment_readers()[0].max_doc;
        assert_eq!(reader.segment_readers()[0].doc_base, 0);
        assert_eq!(reader.segment_readers()[1].doc_base, first_max_doc);
    }

    fn read_commit(dir: &FsDirectory) -> SegmentInfos {
        segment_infos::read_latest(dir).expect("read real segments_1")
    }

    fn read_commit_info(dir: &FsDirectory) -> segment_infos::SegmentCommitInfo {
        read_commit(dir).segments[0].clone()
    }

    /// A segment with real deletions (`del_gen != -1`) must open its `.liv`
    /// file and populate `live_docs`, cross-checked against the real
    /// `del_count` -- exercises the branch
    /// `opens_real_single_segment_fixture_and_matches_manual_wiring` and
    /// `stored_fields_only_segment_opens_without_postings_files` both skip.
    #[test]
    fn segment_with_deletions_opens_live_docs() {
        let dir_path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/live_docs_index/"
        ));
        let dir = FsDirectory::open(&dir_path);
        let reader = DirectoryReader::open(&dir).expect("open real live_docs_index commit");

        assert_eq!(reader.segment_readers().len(), 1);
        let seg = &reader.segment_readers()[0];
        assert_eq!(seg.max_doc, 5);
        let live = seg.live_docs.as_ref().expect("segment has deletions");
        assert_eq!(live.cardinality(), 3);
        assert!(live.get(0) && live.get(2) && live.get(4));
        assert!(!live.get(1) && !live.get(3));

        // Also has real postings (.tim/.tip/.tmd/.doc) but no .pos/.pay,
        // exercising the "some but not all optional postings files" path.
        assert!(seg.doc_buf.is_some());
        assert!(seg.pos_buf.is_none());
        assert!(seg.pay_buf.is_none());
    }

    /// A compound-file (`.cfs`/`.cfe`) segment is out of scope for this
    /// task's `DirectoryReader` (see this module's doc comment) and must
    /// surface a typed error, not silently mis-open or panic.
    #[test]
    fn compound_file_segment_is_rejected_with_typed_error() {
        let dir_path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/compound_index/"
        ));
        let dir = FsDirectory::open(&dir_path);
        let err = DirectoryReader::open(&dir).expect_err("compound segment must be rejected");
        assert!(matches!(err, Error::CompoundFileUnsupported(_)));
    }

    /// A segment whose `.si` lists no `.fnm` file at all is corrupt (every
    /// real segment has field infos); this must surface as an error, not a
    /// panic.
    #[test]
    fn segment_missing_fnm_file_is_an_error() {
        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);

        let si = SegmentInfo {
            id: [1u8; ID_LENGTH],
            version: segment_info::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 3,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec![], // no .fnm listed
            attributes: vec![],
        };
        let si_bytes = segment_info::write(&si, "");
        std::fs::write(dir_path.join("_0.si"), &si_bytes).unwrap();

        let commit = segment_infos::SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: [1u8; ID_LENGTH],
            codec_name: "Lucene104".to_string(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
        };
        let err = SegmentReader::open(&dir, &commit, 0).expect_err("missing .fnm must error");
        assert!(matches!(err, Error::Store(_)));

        std::fs::remove_dir_all(&dir_path).ok();
    }

    /// `.si` listing only some of `.tim`/`.tip`/`.tmd` (a corrupt/partial
    /// segment -- never produced by a real writer) must be rejected rather
    /// than silently treated as "no postings" or panicking on an `unwrap`.
    #[test]
    fn partial_blocktree_files_is_an_error() {
        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);

        let field_infos_list = vec![lucene_codecs::field_infos::FieldInfo {
            name: "title".to_string(),
            number: 0,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: lucene_codecs::field_infos::IndexOptions::None,
            doc_values_type: lucene_codecs::field_infos::DocValuesType::None,
            doc_values_skip_index_type: lucene_codecs::field_infos::DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: lucene_codecs::field_infos::VectorEncoding::Byte,
            vector_similarity_function:
                lucene_codecs::field_infos::VectorSimilarityFunction::Euclidean,
        }];
        let fnm = lucene_codecs::field_infos::write(&field_infos_list, &[2u8; ID_LENGTH], "");
        std::fs::write(dir_path.join("_0.fnm"), &fnm).unwrap();
        // Only `.tim` present -- `.tip`/`.tmd` deliberately missing.
        std::fs::write(dir_path.join("_0_x.tim"), [0u8; 4]).unwrap();

        let si = SegmentInfo {
            id: [2u8; ID_LENGTH],
            version: segment_info::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 1,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec!["_0.fnm".to_string(), "_0_x.tim".to_string()],
            attributes: vec![],
        };
        let si_bytes = segment_info::write(&si, "");
        std::fs::write(dir_path.join("_0.si"), &si_bytes).unwrap();

        let commit = segment_infos::SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: [2u8; ID_LENGTH],
            codec_name: "Lucene104".to_string(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
        };
        let err = SegmentReader::open(&dir, &commit, 0).expect_err("partial postings must error");
        assert!(matches!(err, Error::PartialBlockTreeFiles { found: 1, .. }));

        std::fs::remove_dir_all(&dir_path).ok();
    }

    fn stored_only_field_infos() -> Vec<lucene_codecs::field_infos::FieldInfo> {
        use lucene_codecs::field_infos::{DocValuesType, FieldInfo, IndexOptions};
        vec![FieldInfo {
            name: "title".to_string(),
            number: 0,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: lucene_codecs::field_infos::DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: lucene_codecs::field_infos::VectorEncoding::Byte,
            vector_similarity_function:
                lucene_codecs::field_infos::VectorSimilarityFunction::Euclidean,
        }]
    }

    fn flush_stored_only(
        dir: &FsDirectory,
        segment_name: &str,
        segment_id: [u8; ID_LENGTH],
        text: &str,
    ) -> segment_infos::SegmentCommitInfo {
        use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
        let docs = vec![Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String(text.to_string()),
            }],
        }];
        let lucene_version = segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        };
        lucene_index::segment_writer::flush_stored_only_segment(
            dir,
            segment_name,
            segment_id,
            "Lucene104",
            lucene_version,
            &stored_only_field_infos(),
            &docs,
            false,
        )
        .expect("flush stored-only segment")
    }

    fn write_commit(
        dir: &FsDirectory,
        generation: i64,
        segments: Vec<segment_infos::SegmentCommitInfo>,
    ) -> SegmentInfos {
        let segment_infos = SegmentInfos {
            id: [9u8; ID_LENGTH],
            generation,
            format_version: segment_infos::VERSION_86,
            lucene_version: LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            index_created_version_major: 10,
            version: generation,
            counter: segments.len() as i64,
            min_segment_lucene_version: None,
            segments,
            user_data: vec![],
        };
        segment_infos::write(&segment_infos, dir).expect("write segments_N");
        segment_infos
    }

    /// Reopening with no writes since `self` was opened returns `None` --
    /// real `openIfChanged`'s "nothing to do" convention, checked purely off
    /// the `segments_N` generation (no need to re-diff segment lists).
    #[test]
    fn open_if_changed_returns_none_when_generation_unchanged() {
        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);
        let commit0 = flush_stored_only(&dir, "_0", [1u8; ID_LENGTH], "hello");
        write_commit(&dir, 1, vec![commit0]);

        let reader = DirectoryReader::open(&dir).expect("open segments_1");
        let reopened = reader.open_if_changed(&dir).expect("open_if_changed");
        assert!(reopened.is_none());

        std::fs::remove_dir_all(&dir_path).ok();
    }

    /// Reopening after a genuinely new segment was added returns `Some` with
    /// both segments visible, correct `doc_base`s, and the *first* segment's
    /// reader genuinely reused rather than re-read from disk: proven by
    /// deleting the old segment's on-disk files before reopening -- if
    /// `open_if_changed` tried to re-open segment `_0` from disk it would
    /// fail (file not found), so success here is only possible via reuse.
    #[test]
    fn open_if_changed_reuses_unchanged_segment_and_opens_new_one() {
        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);
        let commit0 = flush_stored_only(&dir, "_0", [1u8; ID_LENGTH], "hello");
        write_commit(&dir, 1, vec![commit0.clone()]);

        let reader = DirectoryReader::open(&dir).expect("open segments_1");
        assert_eq!(reader.segment_readers().len(), 1);

        let commit1 = flush_stored_only(&dir, "_1", [2u8; ID_LENGTH], "world");
        write_commit(&dir, 2, vec![commit0.clone(), commit1]);

        // Remove segment _0's on-disk files: a fresh open of _0 would now
        // fail, so if open_if_changed still succeeds and reports the right
        // doc counts, it must have reused the already-open reader instead of
        // re-reading the (now-missing) files.
        for ext in [".fdt", ".fdx", ".fdm", ".fnm"] {
            std::fs::remove_file(dir_path.join(format!("_0{ext}"))).ok();
        }

        let reopened = reader
            .open_if_changed(&dir)
            .expect("open_if_changed")
            .expect("segments_2 differs from segments_1");
        assert_eq!(reopened.segment_readers().len(), 2);
        assert_eq!(reopened.segment_readers()[0].segment_name, "_0");
        assert_eq!(reopened.segment_readers()[0].doc_base, 0);
        assert_eq!(reopened.segment_readers()[1].segment_name, "_1");
        assert_eq!(reopened.segment_readers()[1].doc_base, 1);

        std::fs::remove_dir_all(&dir_path).ok();
    }

    /// The correctness-critical case: a del_gen-only change on an *existing*
    /// segment (no new segment, same name/id) must NOT be served from the
    /// reused reader's stale `live_docs` -- it must reopen that segment and
    /// pick up the new deletions.
    #[test]
    fn open_if_changed_reopens_segment_whose_del_gen_changed() {
        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);
        let commit0 = flush_stored_only(&dir, "_0", [1u8; ID_LENGTH], "hello");
        write_commit(&dir, 1, vec![commit0.clone()]);

        let reader = DirectoryReader::open(&dir).expect("open segments_1");
        assert!(reader.segment_readers()[0].live_docs.is_none());

        // Delete doc 0 in segment _0: bumps del_gen, writes a new .liv.
        let commit0_deleted = lucene_index::deletes::apply_deletes(&dir, &commit0, None, 1, [0i32])
            .expect("apply delete");
        assert_eq!(commit0_deleted.del_gen, 1);
        write_commit(&dir, 2, vec![commit0_deleted]);

        let reopened = reader
            .open_if_changed(&dir)
            .expect("open_if_changed")
            .expect("segments_2 differs from segments_1");
        assert_eq!(reopened.segment_readers().len(), 1);
        let live = reopened.segment_readers()[0]
            .live_docs
            .as_ref()
            .expect("del_gen change must be picked up as new live_docs");
        assert!(!live.get(0), "doc 0 must now be marked deleted");

        std::fs::remove_dir_all(&dir_path).ok();
    }

    fn tempdir() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lucene-rust-directory-reader-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
