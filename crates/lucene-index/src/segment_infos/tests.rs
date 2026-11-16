#![cfg(test)]
use super::*;

/// Test-only `segments_N` byte builder, independent of the real
/// `IndexWriter`-generated fixture under `tests/segment_infos_fixtures.rs`:
/// that exercises real bytes end-to-end, this covers error paths (negative
/// counts, bad markers, multiple segments/DV fields) that a real writer
/// would never produce.
struct SegBuilder {
    name: String,
    id: [u8; ID_LENGTH],
    codec: String,
    del_gen: i64,
    del_count: i32,
    field_infos_gen: i64,
    doc_values_gen: i64,
    soft_del_count: i32,
    sci_marker: Option<u8>, // None => omit entirely (format <= VERSION_74)
    dv_fields: Vec<(i32, Vec<String>)>,
    num_dv_fields_override: Option<i32>,
}

impl SegBuilder {
    fn valid(name: &str) -> Self {
        Self {
            name: name.to_string(),
            id: [2u8; ID_LENGTH],
            codec: "Lucene104".to_string(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_marker: Some(0),
            dv_fields: vec![],
            num_dv_fields_override: None,
        }
    }
}

struct SisBuilder {
    generation: i64,
    format_version: i32,
    id: [u8; ID_LENGTH],
    segments: Vec<SegBuilder>,
    num_segments_override: Option<i32>,
    commit_version: i64,
    counter: i64,
    user_data: Vec<(String, String)>,
    lucene_version_major: i32,
    lucene_version_minor: i32,
    index_created_version_major: i32,
}

impl SisBuilder {
    fn valid(generation: i64) -> Self {
        Self {
            generation,
            format_version: VERSION_86,
            id: [3u8; ID_LENGTH],
            segments: vec![],
            num_segments_override: None,
            commit_version: 1,
            counter: 1,
            user_data: vec![],
            lucene_version_major: 10,
            lucene_version_minor: 0,
            index_created_version_major: 10,
        }
    }

    fn build(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut out, CODEC_NAME);
        out.extend_from_slice(&(self.format_version as u32).to_be_bytes());
        out.extend_from_slice(&self.id);
        let suffix = lucene_util::base36::to_base36(self.generation);
        out.push(suffix.len() as u8);
        out.extend_from_slice(suffix.as_bytes());

        write_vint(&mut out, self.lucene_version_major);
        write_vint(&mut out, self.lucene_version_minor);
        write_vint(&mut out, 0); // bugfix
        write_vint(&mut out, self.index_created_version_major);

        out.extend_from_slice(&(self.commit_version as u64).to_be_bytes());
        write_vlong(&mut out, self.counter);

        let num_segments = self
            .num_segments_override
            .unwrap_or(self.segments.len() as i32);
        out.extend_from_slice(&(num_segments as u32).to_be_bytes());

        if num_segments > 0 {
            write_vint(&mut out, 10); // minSegmentLuceneVersion major
            write_vint(&mut out, 0);
            write_vint(&mut out, 0);
        }

        for seg in &self.segments {
            write_string(&mut out, &seg.name);
            out.extend_from_slice(&seg.id);
            write_string(&mut out, &seg.codec);
            out.extend_from_slice(&(seg.del_gen as u64).to_be_bytes());
            out.extend_from_slice(&(seg.del_count as u32).to_be_bytes());
            out.extend_from_slice(&(seg.field_infos_gen as u64).to_be_bytes());
            out.extend_from_slice(&(seg.doc_values_gen as u64).to_be_bytes());
            out.extend_from_slice(&(seg.soft_del_count as u32).to_be_bytes());
            if self.format_version > VERSION_74 {
                if let Some(marker) = seg.sci_marker {
                    out.push(marker);
                    if marker == 1 {
                        out.extend_from_slice(&seg.id); // reuse id as a dummy sciId
                    }
                }
            }
            write_vint(&mut out, 0); // fieldInfosFiles: empty set
            let num_dv_fields = seg
                .num_dv_fields_override
                .unwrap_or(seg.dv_fields.len() as i32);
            out.extend_from_slice(&(num_dv_fields as u32).to_be_bytes());
            for (field_number, files) in &seg.dv_fields {
                out.extend_from_slice(&(*field_number as u32).to_be_bytes());
                write_vint(&mut out, files.len() as i32);
                for f in files {
                    write_string(&mut out, f);
                }
            }
        }

        write_vint(&mut out, self.user_data.len() as i32);
        for (k, v) in &self.user_data {
            write_string(&mut out, k);
            write_string(&mut out, v);
        }

        out.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&out) as u64;
        out.extend_from_slice(&checksum.to_be_bytes());
        out
    }
}

fn write_vint(out: &mut Vec<u8>, mut v: i32) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v = ((v as u32) >> 7) as i32;
        if v != 0 {
            b |= 0x80;
            out.push(b);
        } else {
            out.push(b);
            break;
        }
    }
}

fn write_vlong(out: &mut Vec<u8>, mut v: i64) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v = ((v as u64) >> 7) as i64;
        if v != 0 {
            b |= 0x80;
            out.push(b);
        } else {
            out.push(b);
            break;
        }
    }
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    write_vint(out, s.len() as i32);
    out.extend_from_slice(s.as_bytes());
}

#[test]
fn empty_commit_no_segments() {
    let b = SisBuilder::valid(1);
    let sis = parse(&b.build(), 1).unwrap();
    assert_eq!(sis.segments.len(), 0);
    assert!(sis.min_segment_lucene_version.is_none());
}

#[test]
fn single_segment_no_sci_id_and_no_dv_fields() {
    let mut b = SisBuilder::valid(2);
    b.segments.push(SegBuilder::valid("_0"));
    let sis = parse(&b.build(), 2).unwrap();
    assert_eq!(sis.segments.len(), 1);
    assert!(sis.segments[0].sci_id.is_none());
    assert!(sis.min_segment_lucene_version.is_some());
}

#[test]
fn segment_with_sci_id_present() {
    let mut b = SisBuilder::valid(1);
    let mut seg = SegBuilder::valid("_0");
    seg.sci_marker = Some(1);
    b.segments.push(seg);
    let sis = parse(&b.build(), 1).unwrap();
    assert_eq!(sis.segments[0].sci_id, Some([2u8; ID_LENGTH]));
}

#[test]
fn format_at_version_74_omits_sci_marker_entirely() {
    let mut b = SisBuilder::valid(1);
    b.format_version = VERSION_74;
    let mut seg = SegBuilder::valid("_0");
    seg.sci_marker = None; // omitted at this format version, per real Lucene
    b.segments.push(seg);
    let sis = parse(&b.build(), 1).unwrap();
    assert!(sis.segments[0].sci_id.is_none());
}

/// `SegmentCommitInfo.generationAdvanced()` -> `id = randomId()`: every
/// generation-advancing mutator must give the segment-commit a new id,
/// because the id is the only change token a replication/NRT consumer
/// has.
#[test]
fn every_generation_advance_gives_the_segment_commit_a_new_id() {
    type Advance = fn(&mut SegmentCommitInfo);
    let advancers: Vec<(&str, Advance)> = vec![
        ("advance_del_gen", |s| s.advance_del_gen()),
        ("advance_doc_values_gen", |s| s.advance_doc_values_gen()),
        ("advance_field_infos_gen", |s| s.advance_field_infos_gen()),
        ("set_buffered_deletes_gen", |s| {
            s.set_buffered_deletes_gen(7)
        }),
    ];
    for (name, advance) in advancers {
        let mut sci = SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: [3u8; ID_LENGTH],
            codec_name: "Lucene104".to_string(),
            sci_id: Some([1u8; ID_LENGTH]),
            ..Default::default()
        };
        let before = sci.sci_id;
        advance(&mut sci);
        assert_ne!(sci.sci_id, before, "{name} left the id unchanged");
        assert!(sci.sci_id.is_some(), "{name} cleared the id");
    }
}

/// The *counters* that do not advance a generation (Java's
/// `advanceNextWrite*Gen`, which exist for the failed-write retry path)
/// deliberately do **not** call `generationAdvanced()`.
#[test]
fn stepping_only_the_next_write_counter_keeps_the_id() {
    let mut sci = SegmentCommitInfo {
        sci_id: Some([1u8; ID_LENGTH]),
        ..Default::default()
    };
    sci.advance_next_write_field_infos_gen();
    sci.advance_next_write_doc_values_gen();
    assert_eq!(sci.sci_id, Some([1u8; ID_LENGTH]));
}

/// Two commits that differ have different ids -- the property the whole
/// field exists for. A frozen id passes every read, so this is the only
/// shape of test that can see the defect.
#[test]
fn successive_generations_of_one_segment_get_distinct_ids() {
    let base = SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id: [9u8; ID_LENGTH],
        sci_id: Some([0u8; ID_LENGTH]),
        ..Default::default()
    };
    let mut sci = base.clone();
    let mut seen = vec![sci.sci_id];
    for _ in 0..8 {
        sci.advance_del_gen();
        assert!(!seen.contains(&sci.sci_id), "id repeated across a delete");
        seen.push(sci.sci_id);
        sci.advance_doc_values_gen();
        assert!(!seen.contains(&sci.sci_id), "id repeated across a DV round");
        seen.push(sci.sci_id);
    }

    // Two *different segments* at the same generations differ too: the
    // segment's own id is mixed in.
    let mut other = base.clone();
    other.segment_id = [10u8; ID_LENGTH];
    other.advance_del_gen();
    let mut same = base;
    same.advance_del_gen();
    assert_ne!(other.sci_id, same.sci_id);
}

#[test]
fn doc_values_update_fields_are_parsed() {
    let mut b = SisBuilder::valid(1);
    let mut seg = SegBuilder::valid("_0");
    seg.dv_fields = vec![
        (0, vec!["_0_1.dvd".to_string()]),
        (2, vec!["_0_2.dvd".to_string(), "_0_2b.dvd".to_string()]),
    ];
    b.segments.push(seg);
    let sis = parse(&b.build(), 1).unwrap();
    assert_eq!(sis.segments[0].dv_update_files, seg_dv_fields());
}

fn seg_dv_fields() -> Vec<(i32, Vec<String>)> {
    vec![
        (0, vec!["_0_1.dvd".to_string()]),
        (2, vec!["_0_2.dvd".to_string(), "_0_2b.dvd".to_string()]),
    ]
}

#[test]
fn multiple_segments_and_user_data() {
    let mut b = SisBuilder::valid(1);
    b.segments.push(SegBuilder::valid("_0"));
    b.segments.push(SegBuilder::valid("_1"));
    b.user_data.push(("k".to_string(), "v".to_string()));
    let sis = parse(&b.build(), 1).unwrap();
    assert_eq!(sis.segments.len(), 2);
    assert_eq!(sis.user_data, vec![("k".to_string(), "v".to_string())]);
}

#[test]
fn negative_num_segments_rejected() {
    let mut b = SisBuilder::valid(1);
    b.num_segments_override = Some(-1);
    assert!(matches!(
        parse(&b.build(), 1),
        Err(Error::InvalidSegmentCount(-1))
    ));
}

/// `SegmentInfos.readCommit`: a commit that claims to have been created
/// by a newer major than the one that wrote it is corrupt -- Java throws,
/// and so must we, since every later `indexCreatedVersionMajor` gate
/// trusts this value.
#[test]
fn created_version_newer_than_writer_version_rejected() {
    let mut b = SisBuilder::valid(1);
    b.lucene_version_major = 9;
    b.index_created_version_major = 10;
    assert!(matches!(
        parse(&b.build(), 1),
        Err(Error::CreatedVersionAheadOfWriter {
            created: 10,
            major: 9,
            ..
        })
    ));
}

/// Equal majors are legal (an index created by 10.x and written by 10.x).
#[test]
fn created_version_equal_to_writer_version_accepted() {
    let mut b = SisBuilder::valid(1);
    b.lucene_version_major = 10;
    b.index_created_version_major = 10;
    assert!(parse(&b.build(), 1).is_ok());
}

/// `Version.fromBits` packs each component into one byte.
#[test]
fn out_of_range_lucene_version_component_rejected() {
    let mut b = SisBuilder::valid(1);
    b.lucene_version_minor = 300;
    assert!(matches!(
        parse(&b.build(), 1),
        Err(Error::IllegalVersion {
            which: "minor",
            value: 300
        })
    ));
}

#[test]
fn negative_del_count_rejected() {
    let mut b = SisBuilder::valid(1);
    let mut seg = SegBuilder::valid("_0");
    seg.del_count = -1;
    b.segments.push(seg);
    assert!(matches!(
        parse(&b.build(), 1),
        Err(Error::InvalidDeletionCount(-1, name)) if name == "_0"
    ));
}

#[test]
fn negative_soft_del_count_rejected() {
    let mut b = SisBuilder::valid(1);
    let mut seg = SegBuilder::valid("_0");
    seg.soft_del_count = -1;
    b.segments.push(seg);
    assert!(matches!(
        parse(&b.build(), 1),
        Err(Error::InvalidDeletionCount(-1, name)) if name == "_0"
    ));
}

#[test]
fn invalid_sci_marker_rejected() {
    let mut b = SisBuilder::valid(1);
    let mut seg = SegBuilder::valid("_0");
    seg.sci_marker = Some(7); // neither 0 nor 1
    b.segments.push(seg);
    assert!(matches!(
        parse(&b.build(), 1),
        Err(Error::InvalidSciIdMarker(7))
    ));
}

#[test]
fn wrong_generation_suffix_rejected() {
    let b = SisBuilder::valid(1);
    assert!(matches!(parse(&b.build(), 2), Err(Error::Store(_))));
}

// --- write() round-trips through parse(), via a real on-disk Directory ---

use lucene_util::test_support::TempDir;

/// A scratch directory that removes itself when the test ends -- unless
/// the test is panicking, in which case its bytes stay for inspection.
fn tempdir() -> TempDir {
    TempDir::new("segment-infos-write")
}

fn sample_sis(generation: i64) -> SegmentInfos {
    SegmentInfos {
        id: [5u8; ID_LENGTH],
        generation,
        format_version: VERSION_CURRENT,
        lucene_version: LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        index_created_version_major: 10,
        version: 7,
        counter: 3,
        min_segment_lucene_version: None,
        segments: vec![],
        user_data: vec![],
    }
}

fn sample_segment(name: &str) -> SegmentCommitInfo {
    SegmentCommitInfo {
        segment_name: name.to_string(),
        segment_id: [6u8; ID_LENGTH],
        codec_name: "Lucene104".to_string(),
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    }
}

#[test]
fn write_empty_commit_round_trips() {
    let dir_path = tempdir();
    let dir = lucene_store::FsDirectory::open(&dir_path);
    let sis = sample_sis(1);

    let file_name = write(&sis, &dir).unwrap();
    assert_eq!(file_name, "segments_1");

    let bytes = std::fs::read(dir_path.join(&file_name)).unwrap();
    let parsed = parse(&bytes, 1).unwrap();
    assert_eq!(parsed.id, sis.id);
    assert_eq!(parsed.version, sis.version);
    assert_eq!(parsed.counter, sis.counter);
    assert_eq!(
        parsed.index_created_version_major,
        sis.index_created_version_major
    );
    assert!(parsed.segments.is_empty());
    assert!(parsed.min_segment_lucene_version.is_none());

    std::fs::remove_dir_all(&dir_path).ok();
}

#[test]
fn write_single_segment_round_trips() {
    let dir_path = tempdir();
    let dir = lucene_store::FsDirectory::open(&dir_path);
    let mut sis = sample_sis(2);
    sis.min_segment_lucene_version = Some(sis.lucene_version);
    sis.segments.push(sample_segment("_0"));

    let file_name = write(&sis, &dir).unwrap();
    assert_eq!(file_name, "segments_2");

    let bytes = std::fs::read(dir_path.join(&file_name)).unwrap();
    let parsed = parse(&bytes, 2).unwrap();
    assert_eq!(parsed.segments.len(), 1);
    assert_eq!(parsed.segments[0].segment_name, "_0");
    assert_eq!(parsed.segments[0].segment_id, [6u8; ID_LENGTH]);
    assert_eq!(parsed.segments[0].codec_name, "Lucene104");
    assert!(parsed.segments[0].sci_id.is_none());
    assert_eq!(parsed.min_segment_lucene_version, Some(sis.lucene_version));

    std::fs::remove_dir_all(&dir_path).ok();
}

#[test]
fn write_multi_segment_with_sci_id_del_and_dv_updates_round_trips() {
    let dir_path = tempdir();
    let dir = lucene_store::FsDirectory::open(&dir_path);
    let mut sis = sample_sis(3);

    let mut seg0 = sample_segment("_0");
    seg0.del_gen = 1;
    seg0.del_count = 2;
    seg0.field_infos_gen = 4;
    seg0.doc_values_gen = 5;
    seg0.soft_del_count = 1;
    seg0.sci_id = Some([9u8; ID_LENGTH]);
    seg0.field_infos_files = vec!["_0_4.fnm".to_string()];
    seg0.dv_update_files = vec![
        (0, vec!["_0_1.dvd".to_string()]),
        (2, vec!["_0_2.dvd".to_string(), "_0_2b.dvd".to_string()]),
    ];

    let seg1 = sample_segment("_1");

    sis.segments.push(seg0);
    sis.segments.push(seg1);
    sis.user_data.push(("k".to_string(), "v".to_string()));

    let file_name = write(&sis, &dir).unwrap();
    let bytes = std::fs::read(dir_path.join(&file_name)).unwrap();
    let parsed = parse(&bytes, 3).unwrap();

    assert_eq!(parsed.segments.len(), 2);
    let s0 = &parsed.segments[0];
    assert_eq!(s0.del_gen, 1);
    assert_eq!(s0.del_count, 2);
    assert_eq!(s0.field_infos_gen, 4);
    assert_eq!(s0.doc_values_gen, 5);
    assert_eq!(s0.soft_del_count, 1);
    assert_eq!(s0.sci_id, Some([9u8; ID_LENGTH]));
    assert_eq!(s0.field_infos_files, vec!["_0_4.fnm".to_string()]);
    assert_eq!(
        s0.dv_update_files,
        vec![
            (0, vec!["_0_1.dvd".to_string()]),
            (2, vec!["_0_2.dvd".to_string(), "_0_2b.dvd".to_string()]),
        ]
    );
    assert_eq!(parsed.user_data, vec![("k".to_string(), "v".to_string())]);

    std::fs::remove_dir_all(&dir_path).ok();
}

#[test]
fn write_uses_lucene_version_as_min_segment_version_when_unset() {
    let dir_path = tempdir();
    let dir = lucene_store::FsDirectory::open(&dir_path);
    let mut sis = sample_sis(1);
    sis.min_segment_lucene_version = None; // deliberately unset
    sis.segments.push(sample_segment("_0"));

    let file_name = write(&sis, &dir).unwrap();
    let bytes = std::fs::read(dir_path.join(&file_name)).unwrap();
    let parsed = parse(&bytes, 1).unwrap();
    assert_eq!(parsed.min_segment_lucene_version, Some(sis.lucene_version));

    std::fs::remove_dir_all(&dir_path).ok();
}

#[test]
fn write_generation_zero_uses_plain_segments_file_name() {
    let dir_path = tempdir();
    let dir = lucene_store::FsDirectory::open(&dir_path);
    let sis = sample_sis(0);

    let file_name = write(&sis, &dir).unwrap();
    assert_eq!(file_name, "segments");

    let bytes = std::fs::read(dir_path.join(&file_name)).unwrap();
    let parsed = parse(&bytes, 0).unwrap();
    assert_eq!(parsed.generation, 0);

    std::fs::remove_dir_all(&dir_path).ok();
}

#[test]
fn write_negative_generation_is_rejected() {
    let dir_path = tempdir();
    let dir = lucene_store::FsDirectory::open(&dir_path);
    let sis = sample_sis(-1);
    assert!(matches!(write(&sis, &dir), Err(Error::Store(_))));
    assert!(matches!(write_pending(&sis, &dir), Err(Error::Store(_))));
    assert!(matches!(finish_pending(&sis, &dir), Err(Error::Store(_))));
    // `rollback_pending` is infallible by contract and must simply do
    // nothing for a generation that has no pending file name at all.
    rollback_pending(&sis, &dir);
    std::fs::remove_dir_all(&dir_path).ok();
}

/// A `Directory` that delegates everything to a real `FsDirectory` but can
/// be told to fail `sync` or `sync_meta_data`, so the two "clean up the
/// file we just created" paths in `write_pending`/`finish_pending` can be
/// exercised. Both mirror Java (`IOUtils.deleteFilesSuppressingExceptions`
/// in `SegmentInfos.write`/`finishCommit`), and both are unreachable
/// through `FsDirectory` alone -- its `sync_meta_data` is best-effort and
/// never reports failure.
struct FailingDir {
    inner: lucene_store::FsDirectory,
    fail_sync: bool,
    fail_sync_meta_data: bool,
}

impl Directory for FailingDir {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> lucene_store::Result<lucene_store::directory::Input> {
        self.inner.open(name)
    }
    fn create_output(
        &self,
        name: &str,
    ) -> lucene_store::Result<lucene_store::index_output::FsIndexOutput> {
        self.inner.create_output(name)
    }
    fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
        if self.fail_sync {
            return Err(lucene_store::Error::Corrupted("sync failed".to_string()));
        }
        self.inner.sync(names)
    }
    fn rename(&self, source: &str, dest: &str) -> lucene_store::Result<()> {
        self.inner.rename(source, dest)
    }
    fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
        self.inner.delete_file(name)
    }
    fn sync_meta_data(&self) -> lucene_store::Result<()> {
        if self.fail_sync_meta_data {
            return Err(lucene_store::Error::Corrupted(
                "syncMetaData failed".to_string(),
            ));
        }
        self.inner.sync_meta_data()
    }
    fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn lucene_store::Lock>> {
        self.inner.obtain_lock(name)
    }
}

#[test]
fn a_pending_commit_file_that_cannot_be_synced_is_deleted_not_left_behind() {
    let dir_path = tempdir();
    let dir = FailingDir {
        inner: lucene_store::FsDirectory::open(&dir_path),
        fail_sync: true,
        fail_sync_meta_data: false,
    };
    let sis = sample_sis(3);
    assert!(write_pending(&sis, &dir).is_err());
    assert!(
        dir.list_all().unwrap().is_empty(),
        "a pending commit file that could not be fsynced must not survive: {:?}",
        dir.list_all().unwrap()
    );
    std::fs::remove_dir_all(&dir_path).ok();
}

#[test]
fn a_renamed_commit_file_whose_directory_cannot_be_synced_goes_back_to_pending() {
    let dir_path = tempdir();
    let write_dir = FailingDir {
        inner: lucene_store::FsDirectory::open(&dir_path),
        fail_sync: false,
        fail_sync_meta_data: false,
    };
    let sis = sample_sis(3);
    write_pending(&sis, &write_dir).unwrap();

    let finish_dir = FailingDir {
        inner: lucene_store::FsDirectory::open(&dir_path),
        fail_sync: false,
        fail_sync_meta_data: true,
    };
    assert!(finish_pending(&sis, &finish_dir).is_err());
    let listed = finish_dir.list_all().unwrap();
    assert!(
        !listed.iter().any(|f| f == "segments_3"),
        "a commit whose directory entry is not durable must not stay visible: {listed:?}"
    );
    // ...but it stays pending, so the writer's retry can publish it.
    assert!(listed.iter().any(|f| f == "pending_segments_3"));
    finish_pending(&sis, &write_dir).unwrap();
    assert_eq!(read_latest(&write_dir).unwrap().generation, 3);
    std::fs::remove_dir_all(&dir_path).ok();
}

/// `write_pending` leaves the previous commit current: the pending file is
/// not a name `read_latest` can find.
#[test]
fn write_pending_alone_does_not_publish_a_commit() {
    let dir_path = tempdir();
    let dir = lucene_store::FsDirectory::open(&dir_path);
    write(&sample_sis(1), &dir).unwrap();
    write_pending(&sample_sis(2), &dir).unwrap();

    assert_eq!(read_latest(&dir).unwrap().generation, 1);
    assert!(dir
        .list_all()
        .unwrap()
        .iter()
        .any(|f| f == "pending_segments_2"));

    finish_pending(&sample_sis(2), &dir).unwrap();
    assert_eq!(read_latest(&dir).unwrap().generation, 2);

    // ...and `rollback_pending` on a generation with no pending file is a
    // no-op rather than an error.
    rollback_pending(&sample_sis(9), &dir);
    std::fs::remove_dir_all(&dir_path).ok();
}

// --- the arithmetic gate ---

/// Eight flipped bytes in `delGen` are enough to make every *later* commit
/// panic: `getNextDelGen()` is `delGen + 1`, which Java wraps and Rust
/// does not. Before the cap, `parse` accepted `i64::MAX` here and
/// `next_write_del_gen()` overflowed on the first call.
#[test]
fn absurd_generations_are_decode_errors_not_overflowing_next_gens() {
    for (label, mutate) in [
        (
            "delGen",
            Box::new(|s: &mut SegBuilder| s.del_gen = i64::MAX) as Box<dyn Fn(&mut SegBuilder)>,
        ),
        (
            "fieldInfosGen",
            Box::new(|s: &mut SegBuilder| s.field_infos_gen = MAX_GENERATION + 1),
        ),
        (
            "docValuesGen",
            Box::new(|s: &mut SegBuilder| s.doc_values_gen = i64::MIN),
        ),
        // Below `-1`: `IndexFileNames.fileNameFromGeneration` asserts
        // `gen > 0` before it emits a suffix, so a `-5` would name a
        // `_0_-5.liv` no Lucene can read.
        (
            "negative delGen",
            Box::new(|s: &mut SegBuilder| s.del_gen = -5),
        ),
    ] {
        let mut b = SisBuilder::valid(1);
        let mut seg = SegBuilder::valid("_0");
        mutate(&mut seg);
        b.segments.push(seg);
        assert!(
            matches!(parse(&b.build(), 1), Err(Error::InvalidGeneration { .. })),
            "{label} should be rejected"
        );
    }
}

/// A generation exactly at the cap still parses, and the `+ 1` derivation
/// that follows it stays representable — the cap is not off by one.
///
/// Note what this does *not* bless: `MAX_GENERATION + 1` is a legal
/// in-memory next-write generation but **not** a legal thing to serialize,
/// which is what `a_commit_this_port_writes_is_always_one_it_can_read_back`
/// pins down.
#[test]
fn generation_at_the_cap_parses_and_still_derives_a_next_gen() {
    let mut b = SisBuilder::valid(1);
    let mut seg = SegBuilder::valid("_0");
    seg.del_gen = MAX_GENERATION;
    b.segments.push(seg);
    let sis = parse(&b.build(), 1).unwrap();
    assert_eq!(sis.segments[0].next_write_del_gen(), MAX_GENERATION + 1);
}

/// The round-trip property the cap has to satisfy: **anything this crate
/// can write is something `parse` accepts back**. Without the write-side
/// gate, a commit read at exactly `MAX_GENERATION` derives
/// `MAX_GENERATION + 1`, serializes it, and produces a `segments_N` this
/// port then refuses — an index it wrote and can no longer open. Refusing
/// the *commit* is the honest failure, and it leaves the previous
/// `segments_N` current.
#[test]
fn a_commit_this_port_writes_is_always_one_it_can_read_back() {
    let dir_path = tempdir();
    let dir = lucene_store::FsDirectory::open(&dir_path);

    let at_cap = |generation: i64| {
        let mut sis = sample_sis(generation);
        sis.min_segment_lucene_version = Some(sis.lucene_version);
        sis.segments.push(sample_segment("_0"));
        sis
    };

    // The boundary itself round-trips.
    let mut sis = at_cap(1);
    sis.version = MAX_GENERATION;
    sis.counter = MAX_GENERATION;
    sis.segments[0].del_gen = MAX_GENERATION;
    let name = write(&sis, &dir).unwrap();
    let bytes = std::fs::read(dir_path.join(&name)).unwrap();
    let back = parse(&bytes, 1).unwrap();
    assert_eq!(back.version, MAX_GENERATION);
    assert_eq!(back.counter, MAX_GENERATION);
    assert_eq!(back.segments[0].del_gen, MAX_GENERATION);

    // One past it is refused at write time, by every counter in turn,
    // rather than written and discovered unreadable on the next open.
    for (which, mutate) in [
        (
            "version",
            Box::new(|s: &mut SegmentInfos| s.version = MAX_GENERATION + 1)
                as Box<dyn Fn(&mut SegmentInfos)>,
        ),
        (
            "counter",
            Box::new(|s: &mut SegmentInfos| s.counter = MAX_GENERATION + 1),
        ),
        (
            "delGen",
            Box::new(|s: &mut SegmentInfos| s.segments[0].del_gen = MAX_GENERATION + 1),
        ),
        (
            "docValuesGen",
            Box::new(|s: &mut SegmentInfos| s.segments[0].doc_values_gen = MAX_GENERATION + 1),
        ),
        (
            "fieldInfosGen",
            Box::new(|s: &mut SegmentInfos| s.segments[0].field_infos_gen = MAX_GENERATION + 1),
        ),
    ] {
        let mut sis = at_cap(2);
        mutate(&mut sis);
        let err = write(&sis, &dir).unwrap_err();
        assert!(
            matches!(&err, Error::InvalidGeneration { which: w, .. } if *w == which),
            "{which}: unexpected error {err}"
        );
        // Nothing was published, and no pending file was left behind.
        assert_eq!(read_latest(&dir).unwrap().generation, 1);
        assert!(
            !dir.list_all().unwrap().iter().any(|f| f == "segments_2"),
            "{which}: an unreadable commit was published"
        );
    }
    std::fs::remove_dir_all(&dir_path).ok();
}

/// `version` and `counter` are stepped by `+ 1` on every commit
/// (`update_document`), so they carry the same cap.
#[test]
fn absurd_commit_version_and_counter_are_decode_errors() {
    let mut b = SisBuilder::valid(1);
    b.commit_version = i64::MAX;
    assert!(matches!(
        parse(&b.build(), 1),
        Err(Error::InvalidGeneration {
            which: "version",
            ..
        })
    ));

    let mut b = SisBuilder::valid(1);
    b.counter = MAX_GENERATION + 1;
    assert!(matches!(
        parse(&b.build(), 1),
        Err(Error::InvalidGeneration {
            which: "counter",
            ..
        })
    ));
}

/// The `N` in `segments_N` is base 36, so a directory entry named
/// `segments_1y2p0ij32e8e7` hands `parse` a well-formed `i64::MAX` — and
/// `update_document` then does `generation += 1` on it.
#[test]
fn absurd_file_name_generation_is_a_decode_error() {
    let b = SisBuilder::valid(i64::MAX);
    assert!(matches!(
        parse(&b.build(), i64::MAX),
        Err(Error::InvalidGeneration {
            which: "generation",
            ..
        })
    ));
}

/// `numSegments` sized a `Vec<SegmentCommitInfo>` — ~150 bytes of `String`s
/// and `Vec`s each — straight off a 4-byte header field. `i32::MAX` of them
/// is a ~300 GB reservation, and an allocation failure is an **abort**,
/// which `catch_unwind` cannot keep out of the JVM. Only the reservation
/// was unbounded: the loop itself would have hit EOF immediately.
#[test]
fn absurd_segment_count_errors_instead_of_reserving_for_it() {
    let mut b = SisBuilder::valid(1);
    b.num_segments_override = Some(i32::MAX);
    assert!(matches!(
        parse(&b.build(), 1),
        Err(Error::InvalidSegmentCount(i32::MAX))
    ));
}

/// The same shape one level down: `numDVFields` sizes a `Vec<(i32,
/// Vec<String>)>`. Java sizes a `HashMap` from it with no bound either.
#[test]
fn absurd_doc_values_field_count_errors_instead_of_reserving_for_it() {
    for count in [i32::MAX, -1] {
        let mut b = SisBuilder::valid(1);
        let mut seg = SegBuilder::valid("_0");
        seg.num_dv_fields_override = Some(count);
        b.segments.push(seg);
        assert!(
            matches!(
                parse(&b.build(), 1),
                Err(Error::InvalidDocValuesFieldCount(..))
            ),
            "numDVFields={count}"
        );
    }
}
