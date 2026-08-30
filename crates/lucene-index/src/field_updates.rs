//! `ReadersAndUpdates.writeFieldUpdates`: turning resolved doc-values updates
//! into the files real Lucene reads them back from.
//!
//! # What a doc-values update looks like on disk
//!
//! Not a delta file. Lucene rewrites the updated field's **whole column** into
//! a *new generation* of ordinary `Lucene90DocValuesFormat` files and leaves
//! the base ones alone-but-superseded:
//!
//! ```text
//! _0.si                      the segment, unchanged
//! _0_Lucene90_0.dvm/.dvd/.dvs   the base column, still there, still read for
//!                               every field this update did not touch
//! _0_1_Lucene90_0.dvm/.dvd/.dvs generation 1 of the updated field's column
//! _0_1.fnm                      FieldInfos generation 1: the updated field's
//!                               FieldInfo.docValuesGen is now 1
//! ```
//!
//! Three separate pieces of bookkeeping have to agree, and each is silent when
//! it does not:
//!
//! 1. **The generation suffix.** `IndexFileNames.fileNameFromGeneration` writes
//!    the generation in **base 36**, and `PerFieldDocValuesFormat` then appends
//!    its own `<format>_<suffix>` component, so the segment suffix is
//!    `"<base36 gen>_Lucene90_0"` -- used *both* in the file name and inside
//!    each file's index header. A file whose name and header disagree exists
//!    and fails `checkIndexHeader`.
//! 2. **`FieldInfo.docValuesGen`.** A reader finds a field's newest column by
//!    reading `docValuesGen` out of the *newest* `FieldInfos*, which is why an
//!    update also writes a `FieldInfos` generation
//!    (`ReadersAndUpdates.writeFieldInfosGen`). Without it the new `.dvd` is
//!    on disk, referenced, checksummed -- and never read.
//! 3. **`SegmentCommitInfo.getDocValuesUpdatesFiles()`**, keyed by field
//!    number. This is what makes the deleter, `CheckIndex` and
//!    `checksum_verify` see the files. Java **replaces** a field's entry rather
//!    than appending to it, because a generation is the field's complete
//!    column: the previous generation is dead the moment this one lands
//!    ([`SegmentCommitInfo::set_doc_values_updates_files`]).
//!
//! # Scope
//!
//! NUMERIC and BINARY only -- the two types `IndexWriter.updateDocValues`
//! accepts (`DocValuesUpdate` has exactly those two subclasses, and
//! `handleDVUpdates` asserts on it).
//!
//! No compound-file segments: Java writes generational files outside the CFS,
//! but it still reads the *base* column and the base `FieldInfos` through the
//! compound reader. This port's whole buffered-update path already requires
//! loose files (see `index_writer::open_segment_for_deletes`), so a compound
//! segment is rejected here by name rather than silently mis-resolved.
//!
//! No doc-values skip index on an updated field: writing a generation means
//! running the field back through the doc-values consumer, and this port's
//! consumer has no `writeSkipIndex` (see `doc_values.rs`'s own scope note). A
//! field whose `.fnm` claims a skipper would come back out of a generation
//! without one, which its own reader then rejects -- so it is refused up front.

use lucene_codecs::doc_values::{self, DocValuesMeta};
use lucene_codecs::doc_values_updates;
use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos};
use lucene_store::data_output::DataOutput;
use lucene_store::directory::Directory;

use crate::segment_info;
use crate::segment_infos::SegmentCommitInfo;

/// `PerFieldDocValuesFormat.PER_FIELD_FORMAT_KEY`.
pub const PER_FIELD_FORMAT_KEY: &str = "PerFieldDocValuesFormat.format";
/// `PerFieldDocValuesFormat.PER_FIELD_SUFFIX_KEY`.
pub const PER_FIELD_SUFFIX_KEY: &str = "PerFieldDocValuesFormat.suffix";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error(transparent)]
    SegmentInfo(#[from] segment_info::Error),
    #[error(transparent)]
    FieldInfos(#[from] lucene_codecs::field_infos::Error),
    #[error(transparent)]
    DocValues(#[from] doc_values::Error),
    #[error(transparent)]
    DocValuesWrite(#[from] doc_values::WriteError),
    #[error("segment {0} has no .fnm file")]
    MissingFieldInfos(String),
    #[error("segment {segment} has no field number {field_number} in its .fnm")]
    UnknownField { segment: String, field_number: i32 },
    #[error(
        "field {field} in segment {segment} has doc_values_type {ty:?}; only NUMERIC and BINARY \
         can carry doc-values updates"
    )]
    UnsupportedType {
        segment: String,
        field: String,
        ty: DocValuesType,
    },
    #[error(
        "field {field} in segment {segment} declares a doc-values skip index; this port cannot \
         rewrite it into an update generation"
    )]
    SkipIndexUnsupported { segment: String, field: String },
    #[error(
        "segment {0} is a compound-file segment; doc-values updates against one are not supported"
    )]
    CompoundSegment(String),
    #[error(
        "segment {segment} records doc-values generation {gen} for field {field_number} but \
             no {ext} file for it"
    )]
    MissingGenerationFile {
        segment: String,
        field_number: i32,
        gen: i64,
        ext: &'static str,
    },
    #[error(
        "segment {segment} declares a doc-values column for field {field_number} under codec \
         suffix {suffix} but lists no such file"
    )]
    MissingBaseColumn {
        segment: String,
        field_number: i32,
        suffix: String,
    },
    #[error(
        "segment {segment}'s doc-values column carries no entry for field {field_number}, which \
         its .fnm says has one"
    )]
    MissingBaseEntry { segment: String, field_number: i32 },
}

pub type Result<T> = std::result::Result<T, Error>;

/// `IndexFileNames.fileNameFromGeneration(base, ext, gen)` with a per-field
/// codec component spliced in, exactly as `PerFieldDocValuesFormat` +
/// `Lucene90DocValuesConsumer` produce it: `_0` + gen 1 + `Lucene90_0` +
/// `dvd` -> `_0_1_Lucene90_0.dvd`.
pub fn generation_file_name(
    segment_name: &str,
    gen: i64,
    per_field_suffix: &str,
    ext: &str,
) -> String {
    format!(
        "{segment_name}_{}.{ext}",
        generation_segment_suffix(gen, per_field_suffix)
    )
}

/// The `SegmentWriteState.segmentSuffix` a doc-values update generation is
/// written with: `PerFieldDocValuesFormat.getFullSegmentSuffix(base36(gen),
/// "Lucene90_0")`. This string goes into the file name *and* into each file's
/// index header, and the two must be the same string.
pub fn generation_segment_suffix(gen: i64, per_field_suffix: &str) -> String {
    format!("{}_{per_field_suffix}", lucene_util::base36::to_base36(gen))
}

/// `IndexFileNames.fileNameFromGeneration(segment, "fnm", gen)`: the
/// `FieldInfos` generation a doc-values update writes alongside its columns.
/// No per-field component -- `FieldInfosFormat` is not per-field.
pub fn field_infos_gen_file_name(segment_name: &str, gen: i64) -> String {
    format!("{segment_name}_{}.fnm", lucene_util::base36::to_base36(gen))
}

/// One field's resolved NUMERIC updates: `(doc, Some(value))` sets, `(doc,
/// None)` is `DocValuesFieldUpdates.reset(doc)`. Entries are applied in order,
/// so a later one for the same doc wins.
pub type NumericFieldUpdates = (i32, Vec<(i32, Option<i64>)>);
/// The BINARY counterpart of [`NumericFieldUpdates`].
pub type BinaryFieldUpdates = (i32, Vec<(i32, Option<Vec<u8>>)>);

/// `ReadersAndUpdates.writeFieldUpdates`: writes one doc-values update round
/// for `sci` and updates its generation bookkeeping in place.
///
/// Java's per-field loop is reproduced exactly, including that **each updated
/// field takes its own `docValuesGen`** (`handleDVUpdates` calls
/// `info.advanceDocValuesGen()` inside the loop, not once around it) while the
/// whole round shares **one** `fieldInfosGen`.
///
/// `per_field_suffix` is `PerFieldDocValuesFormat`'s `<format>_<suffix>`
/// component, `"Lucene90_0"` for everything this port writes -- passed in
/// rather than hardcoded because `index_writer` owns that constant.
///
/// On failure the files written so far are deleted and `sci` is left
/// describing only files that exist -- but the two next-write counters are
/// **advanced past whatever the failed attempt consumed**, never rewound
/// (`advanceNextWriteFieldInfosGen`/`advanceNextWriteDocValuesGen`), so a retry
/// cannot reuse a name the failed attempt may already have created. That
/// matters precisely for a multi-field round: rewinding to the pre-round value
/// and adding one would land back inside the range the failed attempt already
/// wrote, and the file deletion below is best-effort -- an I/O failure is
/// exactly the case where it also fails.
pub fn write_field_updates(
    dir: &dyn Directory,
    sci: &mut SegmentCommitInfo,
    numeric: &[NumericFieldUpdates],
    binary: &[BinaryFieldUpdates],
    per_field_suffix: &str,
) -> Result<()> {
    let mut created: Vec<String> = Vec::new();
    let snapshot = sci.clone();
    match write_field_updates_inner(dir, sci, numeric, binary, per_field_suffix, &mut created) {
        Ok(()) => Ok(()),
        Err(e) => {
            // Java's `finally { if (success == false) { ... } }`. Java has no
            // snapshot to restore -- it only ever advances -- so the two
            // counters are carried across the restore at whichever value is
            // higher, which is the failed attempt's.
            let next_dv = sci
                .next_write_doc_values_gen()
                .max(snapshot.next_write_doc_values_gen());
            let next_fi = sci
                .next_write_field_infos_gen()
                .max(snapshot.next_write_field_infos_gen());
            *sci = snapshot;
            sci.set_next_write_doc_values_gen(next_dv);
            sci.set_next_write_field_infos_gen(next_fi);
            sci.advance_next_write_field_infos_gen();
            sci.advance_next_write_doc_values_gen();
            for name in &created {
                let _ = dir.delete_file(name);
            }
            Err(e)
        }
    }
}

fn write_field_updates_inner(
    dir: &dyn Directory,
    sci: &mut SegmentCommitInfo,
    numeric: &[NumericFieldUpdates],
    binary: &[BinaryFieldUpdates],
    per_field_suffix: &str,
    created: &mut Vec<String>,
) -> Result<()> {
    let segment = sci.segment_name.clone();
    let si_bytes = dir.open(&format!("{segment}.si"))?;
    let si = segment_info::parse(&si_bytes, &sci.segment_id)?;
    if si.is_compound_file {
        return Err(Error::CompoundSegment(segment));
    }
    let max_doc = si.doc_count;

    // Java clones `reader.getFieldInfos()` -- the segment's *current* infos,
    // which are the generational ones when a previous update round wrote some.
    let mut infos = read_current_field_infos(dir, sci, &si.files)?;

    // Deterministic order: Java iterates a `HashMap`, so the generation a
    // field lands at is arbitrary there. Field-number order makes this port's
    // output reproducible without changing what any reader sees (each field
    // records its own generation).
    let mut numeric: Vec<&NumericFieldUpdates> = numeric.iter().collect();
    numeric.sort_by_key(|(n, _)| *n);
    let mut binary: Vec<&BinaryFieldUpdates> = binary.iter().collect();
    binary.sort_by_key(|(n, _)| *n);

    for (field_number, updates) in numeric {
        let index = field_index(&infos, &segment, *field_number)?;
        check_updatable(&infos.fields[index], &segment, DocValuesType::Numeric)?;
        // `verifyOrCreateDvOnlyField`'s create half: the generational `.fnm`
        // is where a doc-values-only field first declares its type.
        infos.fields[index].doc_values_type = DocValuesType::Numeric;
        // The field's *own* `PerFieldDocValuesFormat` component, not this
        // writer's default: the suffix is per format instance, and a segment
        // Lucene wrote may have given this field a different one. Only a field
        // that has never had a column falls back to the caller's.
        let per_field = per_field_component(&infos.fields[index], per_field_suffix);
        let base = read_base_numeric(dir, sci, &si.files, &infos, index, &per_field)?;
        let column = doc_values_updates::merge_numeric_column(
            base.as_ref().map(|(e, d)| (e, d.as_slice())),
            updates,
            max_doc,
        )?;
        let gen = sci.next_write_doc_values_gen();
        let suffix = generation_segment_suffix(gen, &per_field);
        let (dvm, dvd, dvs) = doc_values_updates::write_numeric_generation(
            *field_number,
            &column,
            &sci.segment_id,
            &suffix,
        )?;
        finish_generation(
            dir, sci, &mut infos, index, gen, &per_field, dvm, dvd, dvs, created,
        )?;
    }

    for (field_number, updates) in binary {
        let index = field_index(&infos, &segment, *field_number)?;
        check_updatable(&infos.fields[index], &segment, DocValuesType::Binary)?;
        // `verifyOrCreateDvOnlyField`'s create half: the generational `.fnm`
        // is where a doc-values-only field first declares its type.
        infos.fields[index].doc_values_type = DocValuesType::Binary;
        let per_field = per_field_component(&infos.fields[index], per_field_suffix);
        let base = read_base_binary(dir, sci, &si.files, &infos, index, &per_field)?;
        let column = doc_values_updates::merge_binary_column(
            base.as_ref().map(|(e, d)| (e, d.as_slice())),
            updates,
            max_doc,
        )?;
        let gen = sci.next_write_doc_values_gen();
        let suffix = generation_segment_suffix(gen, &per_field);
        let (dvm, dvd, dvs) = doc_values_updates::write_binary_generation(
            *field_number,
            &column,
            &sci.segment_id,
            &suffix,
        )?;
        finish_generation(
            dir, sci, &mut infos, index, gen, &per_field, dvm, dvd, dvs, created,
        )?;
    }

    // `ReadersAndUpdates.writeFieldInfosGen`, once for the whole round.
    let fi_gen = sci.next_write_field_infos_gen();
    let fnm_name = field_infos_gen_file_name(&segment, fi_gen);
    let fnm = lucene_codecs::field_infos::write(
        &infos.fields,
        &sci.segment_id,
        &lucene_util::base36::to_base36(fi_gen),
    );
    write_file(dir, &fnm_name, &fnm)?;
    created.push(fnm_name.clone());
    sci.advance_field_infos_gen();
    sci.field_infos_files = vec![fnm_name];

    dir.sync(created)?;
    Ok(())
}

/// The tail of `handleDVUpdates`' per-field body: write the three files, stamp
/// the field's new `docValuesGen` into the `FieldInfos` being built, advance
/// the segment's doc-values generation, and record the files against the field.
#[allow(clippy::too_many_arguments)]
fn finish_generation(
    dir: &dyn Directory,
    sci: &mut SegmentCommitInfo,
    infos: &mut FieldInfos,
    index: usize,
    gen: i64,
    per_field_suffix: &str,
    dvm: Vec<u8>,
    dvd: Vec<u8>,
    dvs: Vec<u8>,
    created: &mut Vec<String>,
) -> Result<()> {
    let segment = sci.segment_name.clone();
    let mut files: Vec<String> = Vec::with_capacity(3);
    for (ext, bytes) in [("dvm", &dvm), ("dvd", &dvd), ("dvs", &dvs)] {
        let name = generation_file_name(&segment, gen, per_field_suffix, ext);
        write_file(dir, &name, bytes)?;
        created.push(name.clone());
        files.push(name);
    }

    let field = &mut infos.fields[index];
    field.doc_values_gen = gen;
    // `PerFieldDocValuesFormat.getInstance` stamps these on the cloned
    // `FieldInfo` it writes into the generational `.fnm`. A field that already
    // had a column in this segment carries them already -- and
    // `per_field_component` derived `per_field_suffix` *from* them, so these
    // two calls rewrite the same values. A field declared in the `.fnm` with a
    // doc-values type but never given a column in this segment has neither,
    // and without them no reader registers a producer for it at all. (Java
    // reaches an adjacent case, `FieldInfos.FieldNumbers.constructFieldInfo`,
    // where the field is absent from the segment's `FieldInfos` entirely;
    // this port cannot -- `field_index` refuses an unknown field, because the
    // writer's field list is fixed at `IndexWriter::open`.)
    put_attribute(field, PER_FIELD_FORMAT_KEY, format_name(per_field_suffix));
    put_attribute(
        field,
        PER_FIELD_SUFFIX_KEY,
        suffix_component(per_field_suffix),
    );

    sci.advance_doc_values_gen();
    let field_number = infos.fields[index].number;
    sci.set_doc_values_updates_files(field_number, files);
    Ok(())
}

/// `"Lucene90_0"` -> `"Lucene90"`: the format half of
/// `PerFieldDocValuesFormat.getSuffix(formatName, suffix)`.
fn format_name(per_field_suffix: &str) -> &str {
    per_field_suffix
        .rsplit_once('_')
        .map(|(name, _)| name)
        .unwrap_or(per_field_suffix)
}

/// `"Lucene90_0"` -> `"0"`.
fn suffix_component(per_field_suffix: &str) -> &str {
    per_field_suffix
        .rsplit_once('_')
        .map(|(_, suffix)| suffix)
        .unwrap_or("0")
}

fn put_attribute(field: &mut FieldInfo, key: &str, value: &str) {
    match field.attributes.iter_mut().find(|(k, _)| k == key) {
        Some(slot) => slot.1 = value.to_string(),
        None => field.attributes.push((key.to_string(), value.to_string())),
    }
}

fn field_index(infos: &FieldInfos, segment: &str, field_number: i32) -> Result<usize> {
    infos
        .fields
        .iter()
        .position(|f| f.number == field_number)
        .ok_or_else(|| Error::UnknownField {
            segment: segment.to_string(),
            field_number,
        })
}

/// `IndexWriter.updateDocValues` -> `globalFieldNumberMap.verifyOrCreateDvOnlyField`:
/// the field must either already carry `expected` doc values, or carry
/// **none** -- in which case Java *creates* it as a doc-values-only field and
/// the update's own generation is its first column. Rejecting `None` here
/// would make "the segment happens not to have written a column for this
/// field yet" an error, which is exactly the state a `.fnm` that only claims
/// doc values it actually holds produces (see
/// `IndexWriter::fields_with_per_field_attributes`).
fn check_updatable(field: &FieldInfo, segment: &str, expected: DocValuesType) -> Result<()> {
    if field.doc_values_type != expected && field.doc_values_type != DocValuesType::None {
        return Err(Error::UnsupportedType {
            segment: segment.to_string(),
            field: field.name.clone(),
            ty: field.doc_values_type,
        });
    }
    if field.doc_values_skip_index_type != DocValuesSkipIndexType::None {
        return Err(Error::SkipIndexUnsupported {
            segment: segment.to_string(),
            field: field.name.clone(),
        });
    }
    Ok(())
}

/// `IndexWriter.readFieldInfos(SegmentCommitInfo)`: the generational `.fnm`
/// when the segment has field updates, the base one otherwise. The segment
/// suffix is part of each file's index header, so reading a generational file
/// with the base suffix (or vice versa) fails the header check rather than
/// decoding wrong.
pub(crate) fn read_current_field_infos(
    dir: &dyn Directory,
    sci: &SegmentCommitInfo,
    si_files: &[String],
) -> Result<FieldInfos> {
    if sci.field_infos_gen != -1 {
        let name = field_infos_gen_file_name(&sci.segment_name, sci.field_infos_gen);
        let bytes = dir.open(&name)?;
        let suffix = lucene_util::base36::to_base36(sci.field_infos_gen);
        return Ok(lucene_codecs::field_infos::parse(
            &bytes,
            &sci.segment_id,
            &suffix,
        )?);
    }
    let name = si_files
        .iter()
        .find(|f| f.ends_with(".fnm"))
        .ok_or_else(|| Error::MissingFieldInfos(sci.segment_name.clone()))?;
    let bytes = dir.open(name)?;
    Ok(lucene_codecs::field_infos::parse(
        &bytes,
        &sci.segment_id,
        "",
    )?)
}

/// `PerFieldDocValuesFormat.getSuffix(formatName, suffix)` as recorded on the
/// field itself, `("Lucene90", "0") -> "Lucene90_0"`. `None` when the field
/// carries no format attributes, which is `PerFieldDocValuesFormat
/// .FieldsReader`'s "the field is in fieldInfos, but has no docvalues" case.
pub(crate) fn field_per_field_component(field: &FieldInfo) -> Option<String> {
    let attr = |key: &str| {
        field
            .attributes
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    };
    Some(format!(
        "{}_{}",
        attr(PER_FIELD_FORMAT_KEY)?,
        attr(PER_FIELD_SUFFIX_KEY)?
    ))
}

/// The component this field's doc-values files are named with: the field's own
/// when it has one, the caller's default when the field has never had a column
/// in this segment (the only case where nothing on disk fixes the answer).
pub(crate) fn per_field_component(field: &FieldInfo, fallback: &str) -> String {
    field_per_field_component(field).unwrap_or_else(|| fallback.to_string())
}

/// The `(meta, data)` pair holding this field's **current** column: the
/// generation `FieldInfo.docValuesGen` names when the field has one, the base
/// `.dvm`/`.dvd` otherwise. `Ok(None)` means the segment carries no
/// doc-values for this field at all -- legitimate, and the case a field
/// declared in the `.fnm` with a doc-values type but never written a column
/// for is in.
pub(crate) fn read_current_column(
    dir: &dyn Directory,
    sci: &SegmentCommitInfo,
    si_files: &[String],
    infos: &FieldInfos,
    index: usize,
    per_field: &str,
) -> Result<Option<(DocValuesMeta, Vec<u8>)>> {
    let field = &infos.fields[index];
    let field_number = field.number;

    if field.doc_values_gen != -1 {
        let gen = field.doc_values_gen;
        let suffix = generation_segment_suffix(gen, per_field);
        let meta_name = generation_file_name(&sci.segment_name, gen, per_field, "dvm");
        let data_name = generation_file_name(&sci.segment_name, gen, per_field, "dvd");
        // The generation's `.dvm` describes exactly one field, so Java hands
        // the producer a one-field `FieldInfos` (`SegmentDocValuesProducer`).
        // Passing all of them would silently accept a `.dvm` naming a field
        // this generation is not for.
        let only = FieldInfos {
            fields: vec![field.clone()],
        };
        let meta_bytes = dir
            .open(&meta_name)
            .map_err(|_| Error::MissingGenerationFile {
                segment: sci.segment_name.clone(),
                field_number,
                gen,
                ext: "dvm",
            })?;
        let data_bytes = dir
            .open(&data_name)
            .map_err(|_| Error::MissingGenerationFile {
                segment: sci.segment_name.clone(),
                field_number,
                gen,
                ext: "dvd",
            })?;
        let (_, meta) = doc_values::parse_meta(&meta_bytes, &sci.segment_id, &suffix, &only)?;
        return Ok(Some((meta, data_bytes.to_vec())));
    }

    // No format attributes means the segment never wrote a column for this
    // field, whatever `.dvm` files it does carry for other fields.
    if field_per_field_component(field).is_none() {
        return Ok(None);
    }

    // A segment can carry more than one doc-values format instance
    // (`_0_Lucene90_0.dvm` *and* `_0_Lucene90_1.dvm`); pick the one this
    // field's own attributes name, not the first `.dvm` in the list. Choosing
    // wrong yields a meta with no entry for the field, and a merge that starts
    // from an all-absent column -- silently dropping every untouched
    // document's value.
    let meta_name = si_files
        .iter()
        .find(|f| {
            f.ends_with(".dvm") && base_codec_suffix(f, &sci.segment_name, ".dvm") == per_field
        })
        .ok_or_else(|| Error::MissingBaseColumn {
            segment: sci.segment_name.clone(),
            field_number,
            suffix: per_field.to_string(),
        })?;
    let data_name = si_files
        .iter()
        .find(|f| {
            f.ends_with(".dvd") && base_codec_suffix(f, &sci.segment_name, ".dvd") == per_field
        })
        .ok_or_else(|| Error::MissingBaseColumn {
            segment: sci.segment_name.clone(),
            field_number,
            suffix: per_field.to_string(),
        })?;
    let meta_bytes = dir.open(meta_name)?;
    let data_bytes = dir.open(data_name)?;
    let (_, meta) = doc_values::parse_meta(&meta_bytes, &sci.segment_id, per_field, infos)?;
    Ok(Some((meta, data_bytes.to_vec())))
}

fn read_base_numeric(
    dir: &dyn Directory,
    sci: &SegmentCommitInfo,
    si_files: &[String],
    infos: &FieldInfos,
    index: usize,
    per_field: &str,
) -> Result<Option<(doc_values::NumericEntry, Vec<u8>)>> {
    let field_number = infos.fields[index].number;
    let Some((meta, data)) = read_current_column(dir, sci, si_files, infos, index, per_field)?
    else {
        return Ok(None);
    };
    // The column exists, so it must describe this field. Falling through to
    // `None` here would start the rewrite from an all-absent column and drop
    // every untouched document's value with no error.
    let entry =
        meta.numeric_entry(field_number)
            .cloned()
            .ok_or_else(|| Error::MissingBaseEntry {
                segment: sci.segment_name.clone(),
                field_number,
            })?;
    Ok(Some((entry, data)))
}

fn read_base_binary(
    dir: &dyn Directory,
    sci: &SegmentCommitInfo,
    si_files: &[String],
    infos: &FieldInfos,
    index: usize,
    per_field: &str,
) -> Result<Option<(doc_values::BinaryEntry, Vec<u8>)>> {
    let field_number = infos.fields[index].number;
    let Some((meta, data)) = read_current_column(dir, sci, si_files, infos, index, per_field)?
    else {
        return Ok(None);
    };
    let entry =
        meta.binary_entry(field_number)
            .cloned()
            .ok_or_else(|| Error::MissingBaseEntry {
                segment: sci.segment_name.clone(),
                field_number,
            })?;
    Ok(Some((entry, data)))
}

/// The `PerFieldDocValuesFormat` suffix embedded in a **base** doc-values file
/// name: `_0_Lucene90_0.dvm` -> `Lucene90_0`. A file with no such component
/// (`_0.dvm`, which this port's own older tests write) yields `""`.
fn base_codec_suffix(file_name: &str, segment_name: &str, ext: &str) -> String {
    file_name
        .strip_prefix(&format!("{segment_name}_"))
        .and_then(|s| s.strip_suffix(ext))
        .unwrap_or("")
        .to_string()
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
    fn read_generation(
        dir: &FsDirectory,
        sci: &SegmentCommitInfo,
        number: i32,
    ) -> Vec<Option<i64>> {
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
        let err = write_field_updates(&dir, &mut sci, &[(9, vec![(0, Some(1))])], &[], SUFFIX)
            .unwrap_err();
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
    fn a_field_with_a_doc_values_skip_index_is_refused_rather_than_silently_dropped() {
        let tmp = tempdir("skipper");
        let dir = FsDirectory::open(&tmp);
        let mut f = field("val", 0, DocValuesType::Numeric);
        f.doc_values_skip_index_type = DocValuesSkipIndexType::Range;
        let fields = vec![f];
        let mut sci = build_segment(&dir, &fields, None, false, 1);
        let err = write_field_updates(&dir, &mut sci, &[(0, vec![(0, Some(1))])], &[], SUFFIX)
            .unwrap_err();
        assert!(matches!(err, Error::SkipIndexUnsupported { .. }), "{err}");
    }

    #[test]
    fn a_compound_segment_is_refused_by_name() {
        let tmp = tempdir("compound");
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("val", 0, DocValuesType::Numeric)];
        let mut sci = build_segment(&dir, &fields, None, true, 1);
        let err = write_field_updates(&dir, &mut sci, &[(0, vec![(0, Some(1))])], &[], SUFFIX)
            .unwrap_err();
        assert!(
            matches!(err, Error::CompoundSegment(ref s) if s == "_0"),
            "{err}"
        );
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
        write_file(&dir, "_0.si", &segment_info::write(&si, "")).unwrap();
        let err = write_field_updates(&dir, &mut sci, &[(0, vec![(0, Some(1))])], &[], SUFFIX)
            .unwrap_err();
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

        let err = write_field_updates(&dir, &mut sci, &[(0, vec![(1, Some(8))])], &[], SUFFIX)
            .unwrap_err();
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
        let (dvm, dvd, dvs) = doc_values::write_single_dense_numeric_field(
            0,
            &[10, 20, 30],
            3,
            &SEG_ID,
            "Lucene90_1",
        )
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
        let err = write_field_updates(&dir, &mut sci, &[(0, vec![(0, Some(1))])], &[], SUFFIX)
            .unwrap_err();
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
        write_file(&dir, "_0.si", &segment_info::write(&si, "")).unwrap();

        let err = write_field_updates(&dir, &mut sci, &[(0, vec![(0, Some(1))])], &[], SUFFIX)
            .unwrap_err();
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
}
