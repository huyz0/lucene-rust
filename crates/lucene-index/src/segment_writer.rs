//! A minimal, honest slice of what real Lucene's `DocumentsWriterPerThread`
//! (buffer documents) + `IndexWriter.commit()` (describe N segments in one
//! `segments_N`) do together -- scoped down to exactly what this port's
//! existing write-side primitives support today.
//!
//! # What this is
//!
//! [`flush_stored_only_segment`] takes an in-memory batch of already-built
//! [`Document`]s and [`FieldInfo`]s, "flushes" them to a brand-new segment
//! (`.fdt`/`.fdx`/`.fdm` stored fields + `.fnm` field infos + `.si` segment
//! info, all written and synced through a real [`Directory`]), and returns
//! the [`SegmentCommitInfo`] that describes it -- ready to push into a
//! [`SegmentInfos::segments`] list. Call it more than once against the same
//! `Directory` with distinct segment names, collect the resulting
//! [`SegmentCommitInfo`]s, and pass all of them to one [`segment_infos::write`]
//! call: that produces a single commit (`segments_N`) that lists multiple,
//! independently-flushed segments -- exactly what `IndexWriter.commit()`
//! does after several `DocumentsWriterPerThread.flush()` calls, minus
//! everything this port hasn't earned yet (see "What this deliberately is
//! not" below).
//!
//! [`segment_infos::write`] itself already generalizes to any number of
//! segments (`SegmentInfos::segments: Vec<SegmentCommitInfo>`, with a plain
//! loop over them in both `parse` and `write`) -- that part of a
//! multi-segment commit was *already* mechanical, not new work. What was
//! missing, and what this module adds, is the reusable "flush one batch of
//! documents to one new segment" building block, so a caller doesn't have to
//! hand-copy the `.fdt`/`.fnm`/`.si`-writing boilerplate (previously
//! duplicated across `write_segment_info_fixture.rs` and
//! `write_segment_infos_fixture.rs`) once per segment.
//!
//! # What this deliberately is not
//!
//! This is **not** an `IndexWriter`. In particular, on purpose, it has:
//! - no RAM accounting or automatic flush-triggering (the caller decides
//!   when to call [`flush_stored_only_segment`], there's no
//!   `ramBufferSizeMB`-style threshold),
//! - no merging (`TieredMergePolicy`/`ConcurrentMergeScheduler` equivalents),
//! - no deletes/updates during indexing (`BufferedUpdates`),
//! - no NRT reopen,
//! - no concurrency (`DocumentsWriterPerThread`-per-thread pooling) -- one
//!   caller, one directory, sequential calls,
//! - and no indexed fields at all yet: like the single-segment fixture it
//!   generalizes, every field is stored-only (`IndexOptions::None`, no doc
//!   values/points/vectors/term vectors), because this port has no write-side
//!   postings/doc-values/points/vectors format built into a reusable form
//!   yet. `SegmentCoreReaders` only opens those producers when
//!   `FieldInfos.hasPostings()`/`hasDocValues()`/etc. are true (see
//!   `org.apache.lucene.index.SegmentCoreReaders`), so a segment with zero
//!   indexed fields needs none of those files -- a real constraint, not a
//!   shortcut in this module.
//!
//! See `docs/parity.md` and `PLAN.md`'s Phase 5 section for the exact,
//! currently-true scope line.
//!
//! # Why a plain function, not a stateful writer/builder object
//!
//! Two shapes were weighed for this slice: (a) exactly what's here -- a
//! free function taking an already-built batch of documents and producing
//! one segment -- versus (b) a stateful `IndexWriter`-shaped builder with an
//! `add_document`/`commit()` API that internally buffers documents across
//! calls. (b) was rejected for now: it would still cap out at one segment
//! per `commit()` (this port has no RAM-threshold/flush-triggering logic to
//! decide *when* to start a second segment), so the extra stateful API
//! surface wouldn't unlock anything this module's callers can't already do
//! by calling [`flush_stored_only_segment`] more than once themselves (see
//! `write_multi_segment_commit_fixture.rs`). Revisit (b) once a real
//! flush-trigger policy (even a trivial "every N documents" one) gives a
//! builder object something genuine to own as internal state --
//! introducing it earlier would be state management with no real caller
//! yet.

use crate::segment_info::{self, LuceneVersion, SegmentInfo};
use crate::segment_infos::SegmentCommitInfo;
use lucene_codecs::compound_format;
use lucene_codecs::field_infos::{self, FieldInfo};
use lucene_codecs::stored_fields::{self, Document};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::data_output::DataOutput;
use lucene_store::directory::Directory;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error(transparent)]
    CompoundFormat(#[from] compound_format::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Flushes `docs` (already fully built -- no analysis chain here, see
/// `PLAN.md` Phase 5 item 1) to a brand-new, stored-fields-only segment named
/// `segment_name` inside `dir`: writes and syncs `<name>.fdt`/`.fdx`/`.fdm`
/// (stored fields), `<name>.fnm` (field infos), and `<name>.si` (segment
/// info), then returns the [`SegmentCommitInfo`] a caller can push into a
/// [`crate::segment_infos::SegmentInfos::segments`] list.
///
/// `segment_id` must be unique per segment in a commit (mirrors real
/// Lucene's `StringHelper.randomId()` per flushed segment -- the caller
/// picks it since this module has no random-id policy of its own yet).
/// `codec_name` is recorded in `segments_N` as-is; this port only ever
/// writes fresh segments so it's the caller's job to pass the name of
/// whatever codec produced the referenced files (matches
/// `crate::segment_info::write`'s existing stance of never round-tripping
/// an old format).
///
/// `use_compound_file` chooses the on-disk layout: when `false` (the
/// original behavior, unchanged), the segment's `.fdt`/`.fdx`/`.fdm`/`.fnm`
/// are written as independent loose files. When `true`, those same four
/// already-complete codec files are packed into one `<segment_name>.cfs`
/// (data) + `<segment_name>.cfe` (entries) pair via
/// [`compound_format::write`] instead -- mirroring real Lucene's
/// `SegmentInfo.setUseCompoundFile(true)` /
/// `Lucene90CompoundFormat.write(...)`, called from `IndexWriter` once a
/// flushed segment's size falls under `TieredMergePolicy`'s
/// `noCFSRatio`/`maxCFSSegmentSizeMB` threshold. This port has no merge
/// policy or segment-size accounting yet (see `PLAN.md` Phase 5), so rather
/// than fake a size heuristic that has nothing real to compare against, the
/// caller decides directly with this boolean -- simpler, and just as
/// correct for every caller this port has today (both `update_document.rs`
/// and this module's own tests pass a literal `true`/`false`).
#[allow(clippy::too_many_arguments)]
pub fn flush_stored_only_segment(
    dir: &dyn Directory,
    segment_name: &str,
    segment_id: [u8; ID_LENGTH],
    codec_name: &str,
    lucene_version: LuceneVersion,
    fields: &[FieldInfo],
    docs: &[Document],
    use_compound_file: bool,
) -> Result<SegmentCommitInfo> {
    let doc_count = docs.len() as i32;

    let (fdt, fdx, fdm) = stored_fields::write_best_speed(docs, &segment_id, "");
    let fnm = field_infos::write(fields, &segment_id, "");

    let fdt_name = format!("{segment_name}.fdt");
    let fdx_name = format!("{segment_name}.fdx");
    let fdm_name = format!("{segment_name}.fdm");
    let fnm_name = format!("{segment_name}.fnm");

    let (files, written_names) = if use_compound_file {
        let sub_files = vec![
            (".fdt".to_string(), fdt),
            (".fdx".to_string(), fdx),
            (".fdm".to_string(), fdm),
            (".fnm".to_string(), fnm),
        ];
        let (cfs, cfe) = compound_format::write(&segment_id, &sub_files)?;
        let cfs_name = format!("{segment_name}.cfs");
        let cfe_name = format!("{segment_name}.cfe");
        write_file(dir, &cfs_name, &cfs)?;
        write_file(dir, &cfe_name, &cfe)?;
        (
            vec![cfs_name.clone(), cfe_name.clone()],
            vec![cfs_name, cfe_name],
        )
    } else {
        for (name, bytes) in [(&fdt_name, &fdt), (&fdx_name, &fdx), (&fdm_name, &fdm)] {
            write_file(dir, name, bytes)?;
        }
        write_file(dir, &fnm_name, &fnm)?;
        (
            vec![
                fdt_name.clone(),
                fdx_name.clone(),
                fdm_name.clone(),
                fnm_name.clone(),
            ],
            vec![fdt_name, fdx_name, fdm_name, fnm_name],
        )
    };

    let si = SegmentInfo {
        id: segment_id,
        version: lucene_version,
        min_version: Some(lucene_version),
        doc_count,
        is_compound_file: use_compound_file,
        has_blocks: false,
        diagnostics: vec![
            ("source".to_string(), "flush".to_string()),
            (
                "lucene.version".to_string(),
                format!(
                    "{}.{}.{}",
                    lucene_version.major, lucene_version.minor, lucene_version.bugfix
                ),
            ),
        ],
        files: files.clone(),
        attributes: vec![(
            "Lucene90StoredFieldsFormat.mode".to_string(),
            "BEST_SPEED".to_string(),
        )],
    };
    let si_name = format!("{segment_name}.si");
    let si_bytes = segment_info::write(&si, "");
    write_file(dir, &si_name, &si_bytes)?;

    let mut synced = written_names;
    synced.push(si_name);
    dir.sync(&synced)?;

    Ok(SegmentCommitInfo {
        segment_name: segment_name.to_string(),
        segment_id,
        codec_name: codec_name.to_string(),
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
    })
}

fn write_file(dir: &dyn Directory, name: &str, bytes: &[u8]) -> Result<()> {
    let mut out = dir.create_output(name)?;
    out.write_bytes(bytes);
    out.close()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::field_infos::{
        DocValuesSkipIndexType, DocValuesType, IndexOptions, VectorEncoding,
        VectorSimilarityFunction,
    };
    use lucene_codecs::stored_fields::{FieldValue, StoredField};
    use lucene_store::directory::FsDirectory;

    fn version() -> LuceneVersion {
        LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        }
    }

    fn stored_only_field(name: &str, number: i32) -> FieldInfo {
        FieldInfo {
            name: name.to_string(),
            number,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        }
    }

    fn doc(id: &str) -> Document {
        Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            }],
        }
    }

    #[test]
    fn flushes_a_segment_with_the_expected_files_and_doc_count() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let docs = vec![doc("1"), doc("2")];

        let sci = flush_stored_only_segment(
            &dir,
            "_0",
            [7u8; ID_LENGTH],
            "Lucene104",
            version(),
            &fields,
            &docs,
            false,
        )
        .unwrap();

        assert_eq!(sci.segment_name, "_0");
        assert_eq!(sci.segment_id, [7u8; ID_LENGTH]);
        assert_eq!(sci.codec_name, "Lucene104");
        assert_eq!(sci.del_count, 0);
        for ext in ["fdt", "fdx", "fdm", "fnm", "si"] {
            assert!(
                std::path::Path::new(&tmp)
                    .join(format!("_0.{ext}"))
                    .exists(),
                "missing _0.{ext}"
            );
        }

        // The .si file must claim the same doc count we flushed -- cross-check
        // against segment_info::parse the same way the real fixture does.
        let si_bytes = std::fs::read(std::path::Path::new(&tmp).join("_0.si")).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        assert_eq!(si.doc_count, docs.len() as i32);
    }

    #[test]
    fn two_flushes_produce_two_independent_segments() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];

        let sci0 = flush_stored_only_segment(
            &dir,
            "_0",
            [1u8; ID_LENGTH],
            "Lucene104",
            version(),
            &fields,
            &[doc("1")],
            false,
        )
        .unwrap();
        let sci1 = flush_stored_only_segment(
            &dir,
            "_1",
            [2u8; ID_LENGTH],
            "Lucene104",
            version(),
            &fields,
            &[doc("2"), doc("3")],
            false,
        )
        .unwrap();

        assert_ne!(sci0.segment_name, sci1.segment_name);
        assert_ne!(sci0.segment_id, sci1.segment_id);
        for ext in ["fdt", "fdx", "fdm", "fnm", "si"] {
            assert!(std::path::Path::new(&tmp)
                .join(format!("_0.{ext}"))
                .exists());
            assert!(std::path::Path::new(&tmp)
                .join(format!("_1.{ext}"))
                .exists());
        }
    }

    #[test]
    fn flush_surfaces_directory_io_error_rather_than_panicking() {
        // A directory that doesn't exist makes the very first create_output
        // (the .fdt file) fail -- confirms Error::Store's #[from] wrapping
        // actually propagates a real Directory I/O failure as an Err rather
        // than panicking or silently losing the error, the one path this
        // module's own error type exists to cover.
        let dir = FsDirectory::open("/nonexistent-lucene-rust-segment-writer-test-dir");
        let fields = vec![stored_only_field("id", 0)];
        let docs = vec![doc("1")];

        let result = flush_stored_only_segment(
            &dir,
            "_0",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
            &fields,
            &docs,
            false,
        );

        assert!(matches!(result, Err(Error::Store(_))));
    }

    /// `use_compound_file: true` must produce a `.cfs`/`.cfe` pair instead of
    /// loose `.fdt`/`.fdx`/`.fdm`/`.fnm`, and the `.si` must record
    /// `is_compound_file: true` -- the two facts a reader relies on to know
    /// which layout to open.
    #[test]
    fn compound_flush_writes_cfs_cfe_pair_and_marks_si_compound() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![stored_only_field("id", 0)];
        let docs = vec![doc("1"), doc("2"), doc("3")];

        let sci = flush_stored_only_segment(
            &dir,
            "_0",
            [3u8; ID_LENGTH],
            "Lucene104",
            version(),
            &fields,
            &docs,
            true,
        )
        .unwrap();

        for ext in ["cfs", "cfe", "si"] {
            assert!(
                std::path::Path::new(&tmp)
                    .join(format!("_0.{ext}"))
                    .exists(),
                "missing _0.{ext}"
            );
        }
        for ext in ["fdt", "fdx", "fdm", "fnm"] {
            assert!(
                !std::path::Path::new(&tmp)
                    .join(format!("_0.{ext}"))
                    .exists(),
                "loose _0.{ext} should not exist in compound mode"
            );
        }

        let si_bytes = std::fs::read(std::path::Path::new(&tmp).join("_0.si")).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        assert!(si.is_compound_file);
        assert_eq!(si.files, vec!["_0.cfs".to_string(), "_0.cfe".to_string()]);
    }

    /// The meaningful end-to-end check: flush with `use_compound_file: true`,
    /// then recover the original `.fdt`/`.fdx`/`.fdm`/`.fnm` sub-files
    /// byte-for-byte via the already-verified `compound_format` reader
    /// (`compound_format::parse_entries` + `open_input`), and confirm
    /// `stored_fields::open` can read documents back out *through* those
    /// recovered slices -- not by re-deriving from the original in-memory
    /// buffers, so a byte-offset bug in the new wiring would show up here.
    #[test]
    fn compound_flush_round_trips_through_compound_reader_and_stored_fields() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let segment_id = [5u8; ID_LENGTH];
        let fields = vec![stored_only_field("id", 0)];
        let docs = vec![doc("alpha"), doc("beta"), doc("gamma")];

        flush_stored_only_segment(
            &dir,
            "_0",
            segment_id,
            "Lucene104",
            version(),
            &fields,
            &docs,
            true,
        )
        .unwrap();

        let cfs = std::fs::read(std::path::Path::new(&tmp).join("_0.cfs")).unwrap();
        let cfe = std::fs::read(std::path::Path::new(&tmp).join("_0.cfe")).unwrap();

        let entries = compound_format::parse_entries(&cfe, &segment_id).unwrap();
        compound_format::check_data_header_footer(&cfs, &segment_id, &entries).unwrap();

        let fdt = compound_format::open_input(&cfs, &entries, ".fdt")
            .unwrap()
            .as_slice();
        let fdx = compound_format::open_input(&cfs, &entries, ".fdx")
            .unwrap()
            .as_slice();
        let fdm = compound_format::open_input(&cfs, &entries, ".fdm")
            .unwrap()
            .as_slice();
        let fnm = compound_format::open_input(&cfs, &entries, ".fnm")
            .unwrap()
            .as_slice();

        // Field infos recovered through the compound reader must still parse
        // and describe the one stored-only field we flushed.
        let parsed_fields = field_infos::parse(fnm, &segment_id, "").unwrap();
        assert_eq!(parsed_fields.fields.len(), 1);
        assert_eq!(parsed_fields.fields[0].name, "id");

        // Stored fields recovered through the compound reader must still
        // open and yield the exact documents flushed, in order.
        let reader = stored_fields::open(fdt, fdx, fdm, &segment_id, "").unwrap();
        for (i, expected) in docs.iter().enumerate() {
            let got = reader.document(i as i32).unwrap();
            assert_eq!(got.fields.len(), expected.fields.len());
            let expected_value = match &expected.fields[0].value {
                FieldValue::String(s) => s.clone(),
                other => panic!("unexpected fixture field value shape: {other:?}"),
            };
            match &got.fields[0].value {
                FieldValue::String(s) => assert_eq!(*s, expected_value),
                other => panic!("unexpected recovered field value shape: {other:?}"),
            }
        }
    }

    fn tempdir() -> String {
        let dir = std::env::temp_dir().join(format!(
            "lucene-rust-segment-writer-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.to_str().unwrap().to_string()
    }
}
