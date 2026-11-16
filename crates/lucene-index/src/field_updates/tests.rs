#![cfg(test)]
use super::*;
use lucene_codecs::field_infos::{IndexOptions, VectorEncoding, VectorSimilarityFunction};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::directory::FsDirectory;

use crate::segment_info::{LuceneVersion, SegmentInfo};

const SEG_ID: [u8; ID_LENGTH] = [42u8; ID_LENGTH];
const SUFFIX: &str = "Lucene90_0";

use lucene_util::test_support::TempDir;

/// A scratch directory that removes itself when the test ends -- unless
/// the test is panicking, in which case its bytes stay for inspection.
fn tempdir(tag: &str) -> TempDir {
    TempDir::new(&format!("field-updates-{tag}"))
}

fn field(name: &str, number: i32, ty: DocValuesType) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: IndexOptions::None,
        doc_values_type: ty,
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

/// A minimal segment on disk: a `.si` listing a `.fnm`, and optionally a
/// base NUMERIC doc-values column for field 0.
fn build_segment(
    dir: &FsDirectory,
    fields: &[FieldInfo],
    base_column: Option<&[i64]>,
    compound: bool,
    max_doc: i32,
) -> SegmentCommitInfo {
    // A real flush stamps the `PerFieldDocValuesFormat` attributes on the
    // field it actually wrote a column for (`IndexWriter::
    // fields_with_per_field_attributes`), and that is what tells a reader
    // -- Lucene's or this one -- which files are the field's. A field
    // without them has no column, whatever `.dvm` files the segment
    // carries for other fields.
    let mut fields = fields.to_vec();
    if base_column.is_some() {
        put_attribute(&mut fields[0], PER_FIELD_FORMAT_KEY, "Lucene90");
        put_attribute(&mut fields[0], PER_FIELD_SUFFIX_KEY, "0");
    }
    let fnm = lucene_codecs::field_infos::write(&fields, &SEG_ID, "");
    write_file(dir, "_0.fnm", &fnm).unwrap();
    let mut files = vec!["_0.fnm".to_string(), "_0.si".to_string()];

    if let Some(values) = base_column {
        let (dvm, dvd, dvs) =
            doc_values::write_single_dense_numeric_field(0, values, max_doc, &SEG_ID, SUFFIX)
                .unwrap();
        for (ext, bytes) in [("dvm", &dvm), ("dvd", &dvd), ("dvs", &dvs)] {
            let name = format!("_0_{SUFFIX}.{ext}");
            write_file(dir, &name, bytes).unwrap();
            files.push(name);
        }
    }

    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let si = SegmentInfo {
        id: SEG_ID,
        version,
        min_version: Some(version),
        doc_count: max_doc,
        is_compound_file: compound,
        has_blocks: false,
        diagnostics: vec![],
        files,
        attributes: vec![],
        index_sort: None,
    };
    write_file(dir, "_0.si", &segment_info::write(&si, "")).unwrap();

    SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id: SEG_ID,
        codec_name: "Lucene104".to_string(),
        ..Default::default()
    }
}

/// The generation's column, read back the way `SegmentDocValuesProducer`
/// resolves it.
fn read_generation(dir: &FsDirectory, sci: &SegmentCommitInfo, number: i32) -> Vec<Option<i64>> {
    let fnm = dir.open(&sci.field_infos_files[0]).unwrap();
    let infos = lucene_codecs::field_infos::parse(
        &fnm,
        &sci.segment_id,
        &lucene_util::base36::to_base36(sci.field_infos_gen),
    )
    .unwrap();
    let f = infos.fields.iter().find(|f| f.number == number).unwrap();
    let gen = f.doc_values_gen;
    // The field's own component, as `PerFieldDocValuesFormat.FieldsReader`
    // reads it -- not this test's default.
    let per_field = field_per_field_component(f).unwrap();
    let suffix = generation_segment_suffix(gen, &per_field);
    let dvm = dir
        .open(&generation_file_name(
            &sci.segment_name,
            gen,
            &per_field,
            "dvm",
        ))
        .unwrap();
    let dvd = dir
        .open(&generation_file_name(
            &sci.segment_name,
            gen,
            &per_field,
            "dvd",
        ))
        .unwrap();
    let only = FieldInfos {
        fields: vec![f.clone()],
    };
    let (_, meta) = doc_values::parse_meta(&dvm, &sci.segment_id, &suffix, &only).unwrap();
    let entry = meta.numeric_entry(number).unwrap();
    let si = segment_info::parse(&dir.open("_0.si").unwrap(), &sci.segment_id).unwrap();
    (0..si.doc_count)
        .map(|d| doc_values::numeric_value(&dvd, entry, d).unwrap())
        .collect()
}

#[test]
fn generational_names_and_suffixes_are_javas() {
    // `IndexFileNames.fileNameFromGeneration` writes the generation in
    // base 36; decimal and base 36 agree only below 10.
    assert_eq!(
        generation_file_name("_0", 1, "Lucene90_0", "dvd"),
        "_0_1_Lucene90_0.dvd"
    );
    assert_eq!(
        generation_file_name("_7", 36, "Lucene90_0", "dvm"),
        "_7_10_Lucene90_0.dvm"
    );
    assert_eq!(generation_segment_suffix(36, "Lucene90_0"), "10_Lucene90_0");
    assert_eq!(field_infos_gen_file_name("_0", 1), "_0_1.fnm");
    assert_eq!(field_infos_gen_file_name("_0", 100), "_0_2s.fnm");
}

#[test]
fn a_numeric_update_rewrites_the_whole_column_at_a_new_generation() {
    let tmp = tempdir("numeric");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("val", 0, DocValuesType::Numeric)];
    let mut sci = build_segment(&dir, &fields, Some(&[10, 20, 30]), false, 3);

    write_field_updates(&dir, &mut sci, &[(0, vec![(1, Some(99))])], &[], SUFFIX).unwrap();

    assert_eq!(sci.doc_values_gen, 1);
    assert_eq!(sci.field_infos_gen, 1);
    assert_eq!(sci.field_infos_files, vec!["_0_1.fnm".to_string()]);
    assert_eq!(
        sci.dv_update_files,
        vec![(
            0,
            vec![
                "_0_1_Lucene90_0.dvm".to_string(),
                "_0_1_Lucene90_0.dvd".to_string(),
                "_0_1_Lucene90_0.dvs".to_string(),
            ]
        )]
    );
    // The untouched documents keep their base values -- the thing a
    // full-column rewrite loses silently.
    assert_eq!(
        read_generation(&dir, &sci, 0),
        vec![Some(10), Some(99), Some(30)]
    );
    // The base column is untouched.
    assert!(tmp.join("_0_Lucene90_0.dvd").exists());
}

#[test]
fn a_second_round_reads_the_previous_generation_as_its_base() {
    let tmp = tempdir("second-round");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("val", 0, DocValuesType::Numeric)];
    let mut sci = build_segment(&dir, &fields, Some(&[10, 20, 30]), false, 3);

    write_field_updates(&dir, &mut sci, &[(0, vec![(1, Some(99))])], &[], SUFFIX).unwrap();
    write_field_updates(&dir, &mut sci, &[(0, vec![(2, Some(77))])], &[], SUFFIX).unwrap();

    assert_eq!(sci.doc_values_gen, 2);
    assert_eq!(sci.field_infos_gen, 2);
    // Generation 1's write survives into generation 2, which is only true
    // if generation 1 was read back as the base.
    assert_eq!(
        read_generation(&dir, &sci, 0),
        vec![Some(10), Some(99), Some(77)]
    );
    // ...and generation 1 is no longer referenced: a generation is the
    // field's complete column, so keeping it would leak files forever.
    assert_eq!(
        sci.dv_update_files[0].1,
        vec![
            "_0_2_Lucene90_0.dvm".to_string(),
            "_0_2_Lucene90_0.dvd".to_string(),
            "_0_2_Lucene90_0.dvs".to_string(),
        ]
    );
}

#[test]
fn a_reset_of_every_value_yields_an_empty_column_not_a_column_of_zeroes() {
    let tmp = tempdir("all-reset");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("val", 0, DocValuesType::Numeric)];
    let mut sci = build_segment(&dir, &fields, Some(&[10, 20]), false, 2);

    write_field_updates(
        &dir,
        &mut sci,
        &[(0, vec![(0, None), (1, None)])],
        &[],
        SUFFIX,
    )
    .unwrap();
    assert_eq!(read_generation(&dir, &sci, 0), vec![None, None]);
}

#[test]
fn a_field_with_no_base_column_gains_the_per_field_format_attributes() {
    // Java's `constructFieldInfo` case. Without the attributes no reader
    // registers a producer for the field at all, so the generation is on
    // disk, referenced, and never read.
    let tmp = tempdir("no-base");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("val", 0, DocValuesType::Numeric)];
    let mut sci = build_segment(&dir, &fields, None, false, 2);

    write_field_updates(&dir, &mut sci, &[(0, vec![(1, Some(5))])], &[], SUFFIX).unwrap();

    let fnm = dir.open(&sci.field_infos_files[0]).unwrap();
    let infos = lucene_codecs::field_infos::parse(&fnm, &SEG_ID, "1").unwrap();
    let f = &infos.fields[0];
    assert_eq!(f.doc_values_gen, 1);
    assert!(f
        .attributes
        .iter()
        .any(|(k, v)| k == PER_FIELD_FORMAT_KEY && v == "Lucene90"));
    assert!(f
        .attributes
        .iter()
        .any(|(k, v)| k == PER_FIELD_SUFFIX_KEY && v == "0"));
    assert_eq!(read_generation(&dir, &sci, 0), vec![None, Some(5)]);
}

#[test]
fn each_updated_field_takes_its_own_doc_values_generation() {
    // `handleDVUpdates` calls `advanceDocValuesGen()` inside its per-field
    // loop, not once around it, so two fields updated in one round never
    // share a generation -- which they must not, since each generation's
    // `.dvm` describes exactly one field.
    let tmp = tempdir("two-fields");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        field("val", 0, DocValuesType::Numeric),
        field("tag", 1, DocValuesType::Binary),
    ];
    let mut sci = build_segment(&dir, &fields, Some(&[1, 2]), false, 2);

    write_field_updates(
        &dir,
        &mut sci,
        &[(0, vec![(0, Some(9))])],
        &[(1, vec![(1, Some(b"x".to_vec()))])],
        SUFFIX,
    )
    .unwrap();

    assert_eq!(sci.doc_values_gen, 2, "two fields, two generations");
    assert_eq!(sci.field_infos_gen, 1, "one FieldInfos generation");
    let fnm = dir.open(&sci.field_infos_files[0]).unwrap();
    let infos = lucene_codecs::field_infos::parse(&fnm, &SEG_ID, "1").unwrap();
    let val = infos.fields.iter().find(|f| f.number == 0).unwrap();
    let tag = infos.fields.iter().find(|f| f.number == 1).unwrap();
    assert_eq!(val.doc_values_gen, 1);
    assert_eq!(tag.doc_values_gen, 2);
}

#[test]
fn a_failure_part_way_through_rolls_the_whole_round_back() {
    // Java's `finally { if (success == false) ... }`: the partially
    // written files are deleted, `sci` is left describing only files that
    // exist, and both next-write counters are advanced so a retry cannot
    // collide with what the failed attempt may have left behind.
    let tmp = tempdir("rollback");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        field("val", 0, DocValuesType::Numeric),
        field("also_numeric", 1, DocValuesType::Numeric),
    ];
    let mut sci = build_segment(&dir, &fields, Some(&[1, 2]), false, 2);

    // Field 1 is NUMERIC, so routing a BINARY update at it fails *after*
    // field 0's generation has already been written.
    let err = write_field_updates(
        &dir,
        &mut sci,
        &[(0, vec![(0, Some(9))])],
        &[(1, vec![(0, Some(b"x".to_vec()))])],
        SUFFIX,
    )
    .unwrap_err();
    assert!(matches!(err, Error::UnsupportedType { .. }), "{err}");

    assert_eq!(sci.doc_values_gen, -1, "no generation was installed");
    assert_eq!(sci.field_infos_gen, -1);
    assert!(sci.dv_update_files.is_empty());
    assert!(sci.field_infos_files.is_empty());
    assert!(
        !tmp.join("_0_1_Lucene90_0.dvd").exists(),
        "the partially written generation must be deleted"
    );
    // A retry goes *past* everything the failed attempt consumed. Field
    // 0's generation 1 was written before the failure, so the next
    // doc-values generation is 3, not 2: rewinding to 2 would reuse a name
    // the failed attempt already created, and the cleanup above is
    // best-effort. Java never rewinds either.
    assert_eq!(sci.next_write_doc_values_gen(), 3);
    assert_eq!(sci.next_write_field_infos_gen(), 2);
    write_field_updates(&dir, &mut sci, &[(0, vec![(0, Some(9))])], &[], SUFFIX).unwrap();
    assert_eq!(sci.doc_values_gen, 3);
    assert_eq!(sci.field_infos_gen, 2);
}

#[test]
fn an_unknown_field_number_is_refused() {
    let tmp = tempdir("unknown-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("val", 0, DocValuesType::Numeric)];
    let mut sci = build_segment(&dir, &fields, Some(&[1]), false, 1);
    let err =
        write_field_updates(&dir, &mut sci, &[(9, vec![(0, Some(1))])], &[], SUFFIX).unwrap_err();
    assert!(
        matches!(
            err,
            Error::UnknownField {
                field_number: 9,
                ..
            }
        ),
        "{err}"
    );
}

#[test]
fn a_field_with_a_doc_values_skip_index_gets_one_in_its_update_generation() {
    let tmp = tempdir("skipper");
    let dir = FsDirectory::open(&tmp);
    let mut f = field("val", 0, DocValuesType::Numeric);
    f.doc_values_skip_index_type = DocValuesSkipIndexType::Range;
    let fields = vec![f];
    let mut sci = build_segment(&dir, &fields, None, false, 1);
    write_field_updates(&dir, &mut sci, &[(0, vec![(0, Some(1))])], &[], SUFFIX).unwrap();
    let dvm = dir
        .open(&format!("_0_1_{SUFFIX}.dvm"))
        .expect("the update generation's .dvm");
    let (_, meta) = doc_values::parse_meta(
        &dvm,
        &SEG_ID,
        &format!("1_{SUFFIX}"),
        &FieldInfos { fields },
    )
    .unwrap();
    let skipper = meta
        .skipper_meta(0)
        .expect("the generation carries the skip index");
    assert_eq!((skipper.min_value, skipper.max_value), (1, 1));
}

#[test]
fn a_segment_whose_si_lists_no_fnm_is_an_error() {
    let tmp = tempdir("no-fnm");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("val", 0, DocValuesType::Numeric)];
    let mut sci = build_segment(&dir, &fields, None, false, 1);
    // Rewrite the `.si` without its `.fnm` entry.
    let mut si = segment_info::parse(&dir.open("_0.si").unwrap(), &SEG_ID).unwrap();
    si.files.retain(|f| !f.ends_with(".fnm"));
    dir.delete_file("_0.si").unwrap();
    write_file(&dir, "_0.si", &segment_info::write(&si, "")).unwrap();
    let err =
        write_field_updates(&dir, &mut sci, &[(0, vec![(0, Some(1))])], &[], SUFFIX).unwrap_err();
    assert!(
        matches!(err, Error::MissingFieldInfos(ref s) if s == "_0"),
        "{err}"
    );
}

#[test]
fn a_recorded_generation_whose_file_is_missing_names_the_generation() {
    let tmp = tempdir("missing-gen");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("val", 0, DocValuesType::Numeric)];
    let mut sci = build_segment(&dir, &fields, Some(&[1, 2]), false, 2);
    write_field_updates(&dir, &mut sci, &[(0, vec![(0, Some(9))])], &[], SUFFIX).unwrap();
    std::fs::remove_file(tmp.join("_0_1_Lucene90_0.dvm")).unwrap();

    let err =
        write_field_updates(&dir, &mut sci, &[(0, vec![(1, Some(8))])], &[], SUFFIX).unwrap_err();
    assert!(
        matches!(
            err,
            Error::MissingGenerationFile {
                gen: 1,
                ext: "dvm",
                ..
            }
        ),
        "{err}"
    );
}

#[test]
fn a_field_is_resolved_through_its_own_codec_suffix_not_the_first_dvm_in_the_si() {
    // A segment can carry more than one doc-values format instance.
    // Picking the first `.dvm` in the `.si` instead of the one the field's
    // own `PerFieldDocValuesFormat.suffix` names yields a meta with no
    // entry for the field, and a rewrite that starts from an all-absent
    // column -- silently dropping every untouched document's value.
    let tmp = tempdir("second-instance");
    let dir = FsDirectory::open(&tmp);
    let mut fields = vec![field("val", 0, DocValuesType::Numeric)];
    put_attribute(&mut fields[0], PER_FIELD_FORMAT_KEY, "Lucene90");
    put_attribute(&mut fields[0], PER_FIELD_SUFFIX_KEY, "1");
    let mut sci = build_segment(&dir, &fields, None, false, 3);

    // The field's real column, under `Lucene90_1`...
    let (dvm, dvd, dvs) =
        doc_values::write_single_dense_numeric_field(0, &[10, 20, 30], 3, &SEG_ID, "Lucene90_1")
            .unwrap();
    // ...and a decoy for some other field under `Lucene90_0`, which sorts
    // first in the `.si` file list.
    let (dvm0, dvd0, dvs0) =
        doc_values::write_single_dense_numeric_field(7, &[1, 2, 3], 3, &SEG_ID, "Lucene90_0")
            .unwrap();
    let mut si = segment_info::parse(&dir.open("_0.si").unwrap(), &SEG_ID).unwrap();
    for (suffix, m, d, sk) in [
        ("Lucene90_0", &dvm0, &dvd0, &dvs0),
        ("Lucene90_1", &dvm, &dvd, &dvs),
    ] {
        for (ext, bytes) in [("dvm", m), ("dvd", d), ("dvs", sk)] {
            let name = format!("_0_{suffix}.{ext}");
            write_file(&dir, &name, bytes).unwrap();
            si.files.push(name);
        }
    }
    dir.delete_file("_0.si").unwrap();
    write_file(&dir, "_0.si", &segment_info::write(&si, "")).unwrap();

    write_field_updates(&dir, &mut sci, &[(0, vec![(1, Some(99))])], &[], SUFFIX).unwrap();
    assert_eq!(
        read_generation(&dir, &sci, 0),
        vec![Some(10), Some(99), Some(30)],
        "the base must be read through `Lucene90_1`, the field's own suffix"
    );
    // ...and the new generation is named with that suffix too.
    assert!(tmp.join("_0_1_Lucene90_1.dvd").exists());
}

#[test]
fn a_field_whose_declared_column_is_missing_from_the_si_is_an_error() {
    let tmp = tempdir("missing-base");
    let dir = FsDirectory::open(&tmp);
    let mut fields = vec![field("val", 0, DocValuesType::Numeric)];
    put_attribute(&mut fields[0], PER_FIELD_FORMAT_KEY, "Lucene90");
    put_attribute(&mut fields[0], PER_FIELD_SUFFIX_KEY, "0");
    // Attributes say the field has a column; the `.si` lists none.
    let mut sci = build_segment(&dir, &fields, None, false, 2);
    let err =
        write_field_updates(&dir, &mut sci, &[(0, vec![(0, Some(1))])], &[], SUFFIX).unwrap_err();
    assert!(matches!(err, Error::MissingBaseColumn { .. }), "{err}");
}

#[test]
fn a_base_column_that_does_not_describe_the_field_is_an_error_not_an_empty_start() {
    let tmp = tempdir("wrong-entry");
    let dir = FsDirectory::open(&tmp);
    let mut fields = vec![
        field("val", 0, DocValuesType::Numeric),
        field("other", 1, DocValuesType::Numeric),
    ];
    for f in fields.iter_mut() {
        put_attribute(f, PER_FIELD_FORMAT_KEY, "Lucene90");
        put_attribute(f, PER_FIELD_SUFFIX_KEY, "0");
    }
    let mut sci = build_segment(&dir, &fields, None, false, 2);
    // A `Lucene90_0` column that describes field 1, not field 0 -- so the
    // `.dvm` parses cleanly and simply has no entry for the field being
    // updated. Starting from an all-absent column here would drop every
    // untouched document's value with no error at all.
    let mut si = segment_info::parse(&dir.open("_0.si").unwrap(), &SEG_ID).unwrap();
    let (dvm, dvd, dvs) =
        doc_values::write_single_dense_numeric_field(1, &[1, 2], 2, &SEG_ID, SUFFIX).unwrap();
    for (ext, bytes) in [("dvm", &dvm), ("dvd", &dvd), ("dvs", &dvs)] {
        let name = format!("_0_{SUFFIX}.{ext}");
        write_file(&dir, &name, bytes).unwrap();
        si.files.push(name);
    }
    dir.delete_file("_0.si").unwrap();
    write_file(&dir, "_0.si", &segment_info::write(&si, "")).unwrap();

    let err =
        write_field_updates(&dir, &mut sci, &[(0, vec![(0, Some(1))])], &[], SUFFIX).unwrap_err();
    assert!(matches!(err, Error::MissingBaseEntry { .. }), "{err}");
}

#[test]
fn the_per_field_component_splits_into_a_format_and_a_suffix() {
    assert_eq!(format_name("Lucene90_0"), "Lucene90");
    assert_eq!(suffix_component("Lucene90_0"), "0");
    // A component with no underscore is a format name with the default
    // suffix, not a suffix with no format.
    assert_eq!(format_name("Lucene90"), "Lucene90");
    assert_eq!(suffix_component("Lucene90"), "0");
}

#[test]
fn putting_an_attribute_twice_replaces_rather_than_duplicates() {
    let mut f = field("val", 0, DocValuesType::Numeric);
    put_attribute(&mut f, PER_FIELD_FORMAT_KEY, "Lucene90");
    put_attribute(&mut f, PER_FIELD_FORMAT_KEY, "Lucene90");
    assert_eq!(
        f.attributes
            .iter()
            .filter(|(k, _)| k == PER_FIELD_FORMAT_KEY)
            .count(),
        1
    );
}

#[test]
fn a_base_doc_values_file_name_with_no_codec_component_yields_an_empty_suffix() {
    assert_eq!(
        base_codec_suffix("_0_Lucene90_0.dvm", "_0", ".dvm"),
        "Lucene90_0"
    );
    assert_eq!(base_codec_suffix("_0.dvm", "_0", ".dvm"), "");
}
