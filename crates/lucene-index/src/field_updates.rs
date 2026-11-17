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
//! Compound-file segments -- what Java flushes by default, so what a shard
//! written by Java's engine holds -- are updated as Java updates them: the
//! base column and base `FieldInfos` are read through the compound reader,
//! and every generation is written loose beside the `.cfs`.
//!
//! A field with a doc-values skip index gets one in every generation too
//! (`doc_values_updates::write_numeric_generation_with_skip_index`), as
//! `Lucene90DocValuesConsumer.addNumericField` writes one for every field
//! whose `FieldInfo` asks for it.

use lucene_codecs::doc_values::{self, DocValuesMeta};
use lucene_codecs::doc_values_updates;
use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos};
use lucene_store::data_output::DataOutput;
use lucene_store::directory::{Directory, Input};

use crate::compound_reader::CompoundReader;
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
    write_field_updates_with_schema(dir, sci, numeric, binary, per_field_suffix, &[])
}

/// [`write_field_updates`] for a writer whose schema can name fields a
/// segment does not carry: an update to a field absent from the segment's
/// `FieldInfos` adds it -- as a doc-values-only field of the update's type,
/// keeping the schema's number and soft-deletes/parent flags -- to the
/// generation's `.fnm`. That is Java's `FieldInfos.FieldNumbers
/// .constructFieldInfo`, reached when `softUpdateDocument` marks a document in
/// a segment none of whose documents had the soft-deletes field. With an
/// empty `schema` an absent field stays an error ([`Error::UnknownField`]).
pub fn write_field_updates_with_schema(
    dir: &dyn Directory,
    sci: &mut SegmentCommitInfo,
    numeric: &[NumericFieldUpdates],
    binary: &[BinaryFieldUpdates],
    per_field_suffix: &str,
    schema: &[FieldInfo],
) -> Result<()> {
    write_field_updates_with_codec(dir, sci, numeric, binary, per_field_suffix, schema, None)
}

/// [`write_field_updates_with_schema`] with the writer's installed codec,
/// whose [`lucene_codecs::codec::Lucene104Codec::before_write`] can fail each
/// generation's doc-values write and the round's `.fnm`.
pub(crate) fn write_field_updates_with_codec(
    dir: &dyn Directory,
    sci: &mut SegmentCommitInfo,
    numeric: &[NumericFieldUpdates],
    binary: &[BinaryFieldUpdates],
    per_field_suffix: &str,
    schema: &[FieldInfo],
    codec: Option<&dyn lucene_codecs::codec::Lucene104Codec>,
) -> Result<()> {
    let mut created: Vec<String> = Vec::new();
    let snapshot = sci.clone();
    match write_field_updates_inner(
        dir,
        sci,
        numeric,
        binary,
        per_field_suffix,
        schema,
        codec,
        &mut created,
    ) {
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

#[allow(clippy::too_many_arguments)]
fn write_field_updates_inner(
    dir: &dyn Directory,
    sci: &mut SegmentCommitInfo,
    numeric: &[NumericFieldUpdates],
    binary: &[BinaryFieldUpdates],
    per_field_suffix: &str,
    schema: &[FieldInfo],
    codec: Option<&dyn lucene_codecs::codec::Lucene104Codec>,
    created: &mut Vec<String>,
) -> Result<()> {
    let segment = sci.segment_name.clone();
    let si_bytes = dir.open(&format!("{segment}.si"))?;
    let si = segment_info::parse_for_codec(&si_bytes, &sci.segment_id, &sci.codec_name)?;
    let max_doc = si.doc_count;

    // A compound segment keeps its base `.fnm` and base columns inside the
    // `.cfs`; a generation is always written beside it, loose, as Java does.
    // `CompoundReader` resolves both: members from the archive, everything
    // else (earlier generations) from `dir`.
    let compound = if si.is_compound_file {
        Some(CompoundReader::open(dir, &segment, &sci.segment_id)?)
    } else {
        None
    };
    let read_dir: &dyn Directory = match &compound {
        Some(c) => c,
        None => dir,
    };
    let files: Vec<String> = match &compound {
        Some(c) => si
            .files
            .iter()
            .filter(|f| !f.ends_with(".cfs") && !f.ends_with(".cfe"))
            .cloned()
            .chain(c.member_files())
            .collect(),
        None => si.files.clone(),
    };

    // Java clones `reader.getFieldInfos()` -- the segment's *current* infos,
    // which are the generational ones when a previous update round wrote some.
    let mut infos = read_current_field_infos(read_dir, sci, &files)?;

    // Each field takes the next generation in the order Java's
    // `handleDVUpdates` walks `pendingDVUpdates`, a `HashMap` by field name
    // ([`crate::indexing_chain::java_hash_map_order`]); fields enter it in
    // field-number order here, which only decides between names that share a
    // bucket.
    let mut numeric: Vec<&NumericFieldUpdates> = numeric.iter().collect();
    numeric.sort_by_key(|(n, _)| *n);
    let mut binary: Vec<&BinaryFieldUpdates> = binary.iter().collect();
    binary.sort_by_key(|(n, _)| *n);
    let mut rounds: Vec<(i32, bool, usize)> = numeric
        .iter()
        .enumerate()
        .map(|(k, (n, _))| (*n, false, k))
        .chain(binary.iter().enumerate().map(|(k, (n, _))| (*n, true, k)))
        .collect();
    rounds.sort_by_key(|(n, binary, _)| (*n, *binary));
    let names: Vec<String> = rounds
        .iter()
        .map(|(n, _, _)| {
            infos
                .fields
                .iter()
                .chain(schema)
                .find(|f| f.number == *n)
                .map_or_else(|| n.to_string(), |f| f.name.clone())
        })
        .collect();
    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let order = crate::indexing_chain::java_hash_map_order(&name_refs);

    for (is_binary, k) in order.into_iter().map(|i| (rounds[i].1, rounds[i].2)) {
        if is_binary {
            let (field_number, updates) = binary[k];
            write_binary_round(
                dir,
                sci,
                &mut infos,
                read_dir,
                &files,
                *field_number,
                updates,
                per_field_suffix,
                schema,
                codec,
                &segment,
                max_doc,
                created,
            )?;
            continue;
        }
        let (field_number, updates) = numeric[k];
        let index = field_index_or_construct(
            &mut infos,
            &segment,
            *field_number,
            schema,
            DocValuesType::Numeric,
        )?;
        check_updatable(&infos.fields[index], &segment, DocValuesType::Numeric)?;
        // `verifyOrCreateDvOnlyField`'s create half: the generational `.fnm`
        // is where a doc-values-only field first declares its type.
        infos.fields[index].doc_values_type = DocValuesType::Numeric;
        // The field's *own* `PerFieldDocValuesFormat` component, not this
        // writer's default: the suffix is per format instance, and a segment
        // Lucene wrote may have given this field a different one. Only a field
        // that has never had a column falls back to the caller's.
        let per_field = per_field_component(&infos.fields[index], per_field_suffix);
        let base = read_base_numeric(read_dir, sci, &files, &infos, index, &per_field)?;
        let column = doc_values_updates::merge_numeric_column(
            base.as_ref().map(|(e, d)| (e, &d[..])),
            updates,
            max_doc,
        )?;
        let gen = sci.next_write_doc_values_gen();
        let suffix = generation_segment_suffix(gen, &per_field);
        // The segment's own number for the field, which its readers look up.
        let write = if infos.fields[index].doc_values_skip_index_type
            == lucene_codecs::field_infos::DocValuesSkipIndexType::None
        {
            doc_values_updates::write_numeric_generation
        } else {
            doc_values_updates::write_numeric_generation_with_skip_index
        };
        let (dvm, dvd, dvs) = crate::index_writer::with_doc_values_faults(codec, 1, || {
            write(
                infos.fields[index].number,
                &column,
                &sci.segment_id,
                &suffix,
            )
            .map_err(Error::from)
        })?;
        check_generation(
            codec,
            sci,
            &infos,
            index,
            max_doc,
            &suffix,
            (&dvm, &dvd, &dvs),
        )?;
        finish_generation(
            dir, sci, &mut infos, index, gen, &per_field, dvm, dvd, dvs, created,
        )?;
    }

    // `ReadersAndUpdates.writeFieldInfosGen`, once for the whole round.
    let fi_gen = sci.next_write_field_infos_gen();
    let fnm_name = field_infos_gen_file_name(&segment, fi_gen);
    crate::index_writer::before_write(codec, lucene_codecs::codec::WritePoint::FieldInfos)?;
    let fnm = lucene_codecs::field_infos::write(
        &infos.fields,
        &sci.segment_id,
        &lucene_util::base36::to_base36(fi_gen),
    );
    write_file(dir, &fnm_name, &fnm)?;
    created.push(fnm_name.clone());
    sci.advance_field_infos_gen();
    sci.field_infos_files = vec![fnm_name];

    // Not synced: the commit that names these generation files syncs them
    // (`IndexWriter.startCommit`).
    Ok(())
}

/// The codec's look ([`crate::index_writer::check_doc_values`]) at one
/// field's update generation before it is written.
fn check_generation(
    codec: Option<&dyn lucene_codecs::codec::Lucene104Codec>,
    sci: &SegmentCommitInfo,
    infos: &FieldInfos,
    index: usize,
    max_doc: i32,
    suffix: &str,
    (dvm, dvd, dvs): (&[u8], &[u8], &[u8]),
) -> Result<()> {
    let field = &infos.fields[index];
    // The field's per-field format, or the one the codec routes it to.
    let format_name = field
        .attributes
        .iter()
        .find(|(k, _)| k == lucene_codecs::per_field_doc_values::PER_FIELD_FORMAT_KEY)
        .map(|(_, v)| v.clone())
        .or_else(|| {
            codec.map(|c| {
                c.doc_values_format_for_field(&field.name)
                    .name()
                    .to_string()
            })
        })
        .unwrap_or_else(|| "Lucene90".to_string());
    let field_infos = FieldInfos {
        fields: vec![field.clone()],
    };
    crate::index_writer::check_doc_values(
        codec,
        &lucene_codecs::codec::DocValuesWrite {
            max_doc,
            segment_id: &sci.segment_id,
            segment_suffix: suffix,
            format_name: &format_name,
            field_infos: &field_infos,
            dvm,
            dvd,
            dvs,
        },
    )?;
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

/// One BINARY field's generation of a round: [`write_field_updates_inner`]'s
/// BINARY arm.
#[allow(clippy::too_many_arguments)]
fn write_binary_round(
    dir: &dyn Directory,
    sci: &mut SegmentCommitInfo,
    infos: &mut FieldInfos,
    read_dir: &dyn Directory,
    files: &[String],
    field_number: i32,
    updates: &[(i32, Option<Vec<u8>>)],
    per_field_suffix: &str,
    schema: &[FieldInfo],
    codec: Option<&dyn lucene_codecs::codec::Lucene104Codec>,
    segment: &str,
    max_doc: i32,
    created: &mut Vec<String>,
) -> Result<()> {
    let index =
        field_index_or_construct(infos, segment, field_number, schema, DocValuesType::Binary)?;
    check_updatable(&infos.fields[index], segment, DocValuesType::Binary)?;
    // `verifyOrCreateDvOnlyField`'s create half: the generational `.fnm`
    // is where a doc-values-only field first declares its type.
    infos.fields[index].doc_values_type = DocValuesType::Binary;
    let per_field = per_field_component(&infos.fields[index], per_field_suffix);
    let base = read_base_binary(read_dir, sci, files, infos, index, &per_field)?;
    let column = doc_values_updates::merge_binary_column(
        base.as_ref().map(|(e, d)| (e, &d[..])),
        updates,
        max_doc,
    )?;
    let gen = sci.next_write_doc_values_gen();
    let suffix = generation_segment_suffix(gen, &per_field);
    let (dvm, dvd, dvs) = crate::index_writer::with_doc_values_faults(codec, 1, || {
        doc_values_updates::write_binary_generation(
            infos.fields[index].number,
            column,
            &sci.segment_id,
            &suffix,
        )
        .map_err(Error::from)
    })?;
    check_generation(
        codec,
        sci,
        infos,
        index,
        max_doc,
        &suffix,
        (&dvm, &dvd, &dvs),
    )?;
    finish_generation(
        dir, sci, infos, index, gen, &per_field, dvm, dvd, dvs, created,
    )?;
    Ok(())
}

/// The segment's own entry for the updated field, found **by name** as
/// `ReadersAndUpdates.writeFieldUpdates` does (`fieldInfos.fieldInfo(field)`):
/// the caller names fields by this writer's global numbers, and a segment
/// another writer numbered -- Java's, typically -- gives the same field
/// another number. When `schema` does not know `field_number`, the numbers
/// are taken to be the segment's own.
///
/// A field the segment lacks gets a doc-values-only `FieldInfo`
/// (`FieldNumbers.constructFieldInfo`: no indexing, no norms, the update's
/// doc-values type, no skip index, `dvGen == -1`, the schema's
/// soft-deletes/parent flags), numbered with the writer's number when the
/// segment has not used it and with the segment's next free number
/// otherwise -- the number only has to be unique within this segment's
/// `FieldInfos`.
fn field_index_or_construct(
    infos: &mut FieldInfos,
    segment: &str,
    field_number: i32,
    schema: &[FieldInfo],
    doc_values_type: DocValuesType,
) -> Result<usize> {
    let Some(known) = schema.iter().find(|f| f.number == field_number) else {
        return field_index(infos, segment, field_number);
    };
    if let Some(index) = infos.fields.iter().position(|f| f.name == known.name) {
        return Ok(index);
    }
    let number = if infos.fields.iter().any(|f| f.number == field_number) {
        infos
            .fields
            .iter()
            .map(|f| f.number)
            .max()
            .unwrap_or(-1)
            .saturating_add(1)
    } else {
        field_number
    };
    infos.fields.push(
        FieldInfo::new(known.name.clone(), number)
            .with_doc_values(doc_values_type, DocValuesSkipIndexType::None, -1)
            .with_soft_deletes_field(known.soft_deletes_field)
            .with_parent_field(known.parent_field),
    );
    Ok(infos.fields.len().saturating_sub(1))
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
) -> Result<Option<(DocValuesMeta, Input)>> {
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
        let (version, mut meta) =
            doc_values::parse_meta(&meta_bytes, &sci.segment_id, &suffix, &only)?;
        if doc_values::has_disi_file(version) {
            let disi_name = generation_file_name(&sci.segment_name, gen, per_field, "dvp");
            let disi_bytes = dir
                .open(&disi_name)
                .map_err(|_| Error::MissingGenerationFile {
                    segment: sci.segment_name.clone(),
                    field_number,
                    gen,
                    ext: "dvp",
                })?;
            meta.attach_disi(doc_values::DisiBytes::new(std::sync::Arc::new(disi_bytes)));
        }
        return Ok(Some((meta, data_bytes)));
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
    let (version, mut meta) =
        doc_values::parse_meta(&meta_bytes, &sci.segment_id, per_field, infos)?;
    if doc_values::has_disi_file(version) {
        let disi_name = si_files
            .iter()
            .find(|f| {
                f.ends_with(".dvp") && base_codec_suffix(f, &sci.segment_name, ".dvp") == per_field
            })
            .ok_or_else(|| Error::MissingBaseColumn {
                segment: sci.segment_name.clone(),
                field_number,
                suffix: per_field.to_string(),
            })?;
        let disi_bytes = dir.open(disi_name)?;
        meta.attach_disi(doc_values::DisiBytes::new(std::sync::Arc::new(disi_bytes)));
    }
    // The `Input` itself, not a copy: the base `.dvd` holds every
    // doc-values field of the segment, and under `MmapDirectory` it is a
    // mapping a copy would read in full once per updated field.
    Ok(Some((meta, data_bytes)))
}

fn read_base_numeric(
    dir: &dyn Directory,
    sci: &SegmentCommitInfo,
    si_files: &[String],
    infos: &FieldInfos,
    index: usize,
    per_field: &str,
) -> Result<Option<(doc_values::NumericEntry, Input)>> {
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
) -> Result<Option<(doc_values::BinaryEntry, Input)>> {
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
mod tests;
