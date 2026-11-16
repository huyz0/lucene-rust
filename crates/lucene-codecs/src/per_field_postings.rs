//! Port of `org.apache.lucene.codecs.perfield.PerFieldPostingsFormat` for
//! the postings format this port has -- `Lucene104PostingsFormat`, with any
//! block-tree block sizes (`Lucene104PostingsFormat(minTermBlockSize,
//! maxTermBlockSize)`), which is enough to route a segment's fields to
//! several format instances.
//!
//! **Write** ([`write()`], `FieldsWriter.write`/`merge` through
//! `buildFieldsGroupMapping`): the fields, in name order, are grouped by the
//! format their caller routes them to; each format *name* numbers its
//! instances from 0 in order of first appearance, and each group is written
//! as one ordinary set of postings files under the codec suffix
//! `<formatName>_<n>` ([`codec_suffix`]), whose number every field of the
//! group records in its `PerFieldPostingsFormat.suffix` attribute. `.pos`
//! and `.pay` presence follows the whole segment's `FieldInfos`, as
//! `Lucene104PostingsWriter` decides it from `state.fieldInfos` -- so every
//! group of a segment with positions anywhere has a `.pos`, empty or not.
//!
//! **Read** ([`open_groups`], `FieldsReader`): one dictionary per group,
//! combined into one [`BlockTreeFields`], with the groups' `.doc`/`.pos`/
//! `.pay` concatenated so a single postings input serves every field (each
//! field decodes its file pointers at its group's offset,
//! [`BlockTreeFields::combine`]). Only a segment with more than one group
//! pays for the copy.
//!
//! A field may also be routed to `Completion104PostingsFormat`
//! ([`Lucene104PostingsFormat::completion104`]): its group is written as
//! the default `Lucene104` files under the suffix `Completion104_<n>`, plus
//! the `.lkp`/`.cmp` that `CompletionFieldsConsumer` builds from the same
//! terms and payloads ([`CompletionOutput`]).
//!
//! Java groups by format *instance* identity, so two distinct but equal
//! instances get two suffixes; here a format is its block sizes and kind,
//! and equal ones share a group unless [`Lucene104PostingsFormat::as_instance`]
//! numbers them apart (Differs: identity is an explicit number).

use lucene_store::codec_util::ID_LENGTH;

use crate::blocktree::{self, BlockTreeFields, SharedBytes};
use crate::completion::{CompletionFieldWriter, CompletionFilesWriter, TermDoc};
use crate::field_infos::{FieldInfo, FieldInfos};
use crate::postings::{DocInput, PayInput, PosInput};
use crate::postings_writer::{
    self, FieldNorms, FieldPostingsInput, Output, StreamingFieldsWriter, TermSource, WriteOptions,
};
use crate::terms_dictionary::TermsDictionaryFormat;
use std::borrow::Cow;
use std::sync::Arc;

/// `PerFieldPostingsFormat.PER_FIELD_FORMAT_KEY`.
pub const PER_FIELD_FORMAT_KEY: &str = "PerFieldPostingsFormat.format";
/// `PerFieldPostingsFormat.PER_FIELD_SUFFIX_KEY`.
pub const PER_FIELD_SUFFIX_KEY: &str = "PerFieldPostingsFormat.suffix";

/// Verify every indexed field's routing pair names an existing dictionary group.
/// Java's `FieldsReader` opens groups from `FieldInfos`, including fields whose
/// last term was removed by an earlier merge. Looking only at surviving files
/// would silently omit a group missing from the segment's file list.
/// Unrouted zero-token fields and legacy direct formats require no group here.
/// File names may be loose segment names or compound archive member names.
pub fn validate_routed_groups(fields: &[FieldInfo], files: &[String]) -> Result<(), String> {
    for field in fields {
        if field.index_options == crate::field_infos::IndexOptions::None {
            continue;
        }
        let attribute = |key| {
            field
                .attributes
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        let (format, suffix) = match (
            attribute(PER_FIELD_FORMAT_KEY),
            attribute(PER_FIELD_SUFFIX_KEY),
        ) {
            (None, None) => continue,
            (Some(format), Some(suffix)) if !format.is_empty() && !suffix.is_empty() => {
                (format, suffix)
            }
            _ => {
                return Err(format!(
                    "field {:?} has incomplete postings routing",
                    field.name
                ));
            }
        };
        // SimpleTextPostingsFormat.getPostingsFileName uses the same group
        // suffix but stores the dictionary and postings together in `.pst`.
        let extension = if format == crate::simpletext::fields_writer::FORMAT_NAME {
            crate::simpletext::fields_writer::POSTINGS_EXTENSION
        } else {
            marker_extension(format)
        };
        let dictionary = format!("_{format}_{suffix}.{extension}");
        if !files.iter().any(|file| file.ends_with(&dictionary)) {
            return Err(format!(
                "field {:?} routes to missing postings group {format}_{suffix}",
                field.name
            ));
        }
    }
    Ok(())
}

/// `Lucene104PostingsFormat.getName()`.
pub const FORMAT_NAME: &str = "Lucene104";
/// `Completion104PostingsFormat.getName()`, also the codec name of its
/// `.lkp`/`.cmp` headers.
pub const COMPLETION_FORMAT_NAME: &str = "Completion104";
/// Every `CompletionPostingsFormat` subclass a Lucene 10.5.0 index can hold,
/// oldest first: `Completion90PostingsFormat` (Lucene 9.0-9.8),
/// `Completion99` (9.9-9.11), `Completion912` (9.12-10.0), `Completion101`
/// (10.1-10.3) and `Completion104`. They differ only in their name (the
/// `.cmp`/`.lkp` codec name) and their delegate, which the `.cmp` names
/// itself. `Completion50` (`"completion"`) and `Completion84` are Lucene
/// 8.x formats: neither Lucene 10 nor OpenSearch 3.x opens an index created
/// by Lucene 8 (`IndexFormatTooOldException`), and 9.x could not write them.
pub const COMPLETION_FORMAT_NAMES: [&str; 5] = [
    "Completion90",
    "Completion99",
    "Completion912",
    "Completion101",
    COMPLETION_FORMAT_NAME,
];

/// The completion format a group's codec suffix names (`Completion912_0` ->
/// `Completion912`), or `None` for a suffix of any other format
/// (`PostingsFormat.forName`'s lookup of the per-field format attribute).
pub fn completion_format_of(suffix: &str) -> Option<&'static str> {
    let (name, _) = suffix.rsplit_once('_')?;
    COMPLETION_FORMAT_NAMES.into_iter().find(|n| *n == name)
}

/// Whether `name`, a field's `PerFieldPostingsFormat.format` attribute, is a
/// completion format.
pub fn is_completion_format(name: &str) -> bool {
    COMPLETION_FORMAT_NAMES.contains(&name)
}

/// Which postings format a field is routed to.
#[derive(Debug, Clone, Copy, Default)]
pub enum PostingsFormatKind {
    /// `Lucene104PostingsFormat`.
    #[default]
    Lucene104,
    /// `Completion104PostingsFormat`: a default `Lucene104PostingsFormat`
    /// delegate plus the suggester files.
    Completion104,
    /// A format that is another terms dictionary over
    /// `Lucene104PostingsWriter`'s postings (`BlockTreeOrds`, ...). Equal
    /// when it is the same instance, as Java's `buildFieldsGroupMapping`
    /// groups by instance identity.
    Dictionary(&'static dyn TermsDictionaryFormat),
}

impl PartialEq for PostingsFormatKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Lucene104, Self::Lucene104) | (Self::Completion104, Self::Completion104) => true,
            (Self::Dictionary(a), Self::Dictionary(b)) => std::ptr::addr_eq(*a, *b),
            _ => false,
        }
    }
}

impl Eq for PostingsFormatKind {}

/// `PerFieldPostingsFormat.getSuffix(formatName, suffix)`: the codec suffix
/// a group's files and headers carry.
pub fn codec_suffix(format_name: &str, suffix: u32) -> String {
    format!("{format_name}_{suffix}")
}

/// `Lucene104PostingsFormat(minTermBlockSize, maxTermBlockSize)`: the
/// format a field is routed to. [`Default`] is `Lucene104PostingsFormat()`
/// (25, 48).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lucene104PostingsFormat {
    pub min_term_block_size: usize,
    pub max_term_block_size: usize,
    pub kind: PostingsFormatKind,
    /// The codec's `postingsFormat()` itself rather than a
    /// `PerFieldPostingsFormat` delegate (`CheapBastardCodec`): its files
    /// carry no codec suffix and its fields no `PerFieldPostingsFormat`
    /// attributes. See [`Lucene104PostingsFormat::direct`].
    pub direct: bool,
    /// The name of a format that wraps this one without changing its files
    /// (`AssertingPostingsFormat`'s `Asserting`), recorded in place of
    /// [`FORMAT_NAME`]. See [`Lucene104PostingsFormat::named`].
    pub wrapper_name: Option<&'static str>,
    /// Distinguishes equal formats Java holds as separate instances; see
    /// [`Lucene104PostingsFormat::as_instance`].
    pub instance: u32,
    /// [`WriteOptions::zstd_term_blocks`]; see
    /// [`Lucene104PostingsFormat::rust_zstd`].
    pub zstd_term_blocks: Option<i32>,
}

/// The name of [`Lucene104PostingsFormat::rust_zstd`]'s format.
pub const RUST_ZSTD_FORMAT_NAME: &str = "ZstdTerms104";

impl Default for Lucene104PostingsFormat {
    fn default() -> Self {
        let options = WriteOptions::default();
        Lucene104PostingsFormat {
            min_term_block_size: options.min_items_in_block,
            max_term_block_size: options.max_items_in_block,
            kind: PostingsFormatKind::Lucene104,
            direct: false,
            wrapper_name: None,
            instance: 0,
            zstd_term_blocks: None,
        }
    }
}

impl Lucene104PostingsFormat {
    /// The constructor, with `Lucene103BlockTreeTermsWriter.validateSettings`'s
    /// checks.
    pub fn new(
        min_term_block_size: usize,
        max_term_block_size: usize,
    ) -> postings_writer::Result<Self> {
        let format = Lucene104PostingsFormat {
            min_term_block_size,
            max_term_block_size,
            kind: PostingsFormatKind::Lucene104,
            direct: false,
            wrapper_name: None,
            instance: 0,
            zstd_term_blocks: None,
        };
        format.write_options().validate()?;
        Ok(format)
    }

    /// `new Completion104PostingsFormat()`: its delegate is
    /// `new Lucene104PostingsFormat()`, the default block sizes.
    pub fn completion104() -> Self {
        Lucene104PostingsFormat {
            kind: PostingsFormatKind::Completion104,
            ..Self::default()
        }
    }

    /// A field routed to `format`, a terms dictionary over the default
    /// postings. The block sizes are unused: the format has its own.
    pub fn dictionary(format: &'static dyn TermsDictionaryFormat) -> Self {
        Lucene104PostingsFormat {
            kind: PostingsFormatKind::Dictionary(format),
            ..Self::default()
        }
    }

    /// `new BlockTreeOrdsPostingsFormat(minTermsInBlock, maxTermsInBlock)`,
    /// with `validateSettings`' checks. The default sizes are the shared
    /// [`crate::blocktreeords::DEFAULT_FORMAT`]; other sizes make a new
    /// instance that lives as long as the process, as a codec's format
    /// instances do in Java.
    pub fn block_tree_ords(
        min_terms_in_block: usize,
        max_terms_in_block: usize,
    ) -> postings_writer::Result<Self> {
        use crate::blocktreeords::{self, BlockTreeOrdsPostingsFormat};
        let format: &'static BlockTreeOrdsPostingsFormat =
            if (min_terms_in_block, max_terms_in_block)
                == (
                    blocktreeords::DEFAULT_MIN_BLOCK_SIZE,
                    blocktreeords::DEFAULT_MAX_BLOCK_SIZE,
                )
            {
                &blocktreeords::DEFAULT_FORMAT
            } else {
                Box::leak(Box::new(
                    BlockTreeOrdsPostingsFormat::new(min_terms_in_block, max_terms_in_block)
                        .map_err(|e| postings_writer::Error::InvalidBlockSizes(e.to_string()))?,
                ))
            };
        Ok(Self::dictionary(format))
    }

    /// This format as a codec's whole `postingsFormat()`, not one
    /// `PerFieldPostingsFormat` routes fields to: every field is written in
    /// one group whose files have no codec suffix (`_0.tim`) and no field
    /// records the per-field attributes. Every field of a segment must get
    /// the same direct format.
    pub fn direct(self) -> Self {
        Lucene104PostingsFormat {
            direct: true,
            ..self
        }
    }

    /// This format's files under a wrapping format's name, as
    /// `AssertingPostingsFormat` writes its `Lucene104PostingsFormat`
    /// delegate's: fields record `name` and its group's files carry it
    /// (`_0_Asserting_0.tim`). Readers find the files by their extensions,
    /// whatever the name.
    pub fn named(self, name: &'static str) -> Self {
        Lucene104PostingsFormat {
            wrapper_name: Some(name),
            ..self
        }
    }

    /// `ZstdTerms104`: `Lucene104PostingsFormat()`'s postings under a terms
    /// dictionary whose blocks are zstd-compressed at `level` where that
    /// pays, a large field's against a dictionary trained on its own blocks
    /// ([`WriteOptions::zstd_term_blocks`]). Stock Lucene cannot read it;
    /// the plugin's Java side reads it as `ZstdTerms104`.
    ///
    /// # Errors
    /// `level` is not a zstd level.
    pub fn rust_zstd(level: i32) -> postings_writer::Result<Self> {
        let format = Lucene104PostingsFormat {
            wrapper_name: Some(RUST_ZSTD_FORMAT_NAME),
            zstd_term_blocks: Some(level),
            ..Self::default()
        };
        format.write_options().validate()?;
        Ok(format)
    }

    /// This format as Java's `instance`th separate object with equal
    /// settings (`RandomCodec`'s two `Lucene104` candidates):
    /// `PerFieldPostingsFormat` groups by identity, so its fields get their
    /// own group under the next suffix of its name.
    pub fn as_instance(self, instance: u32) -> Self {
        Lucene104PostingsFormat { instance, ..self }
    }

    /// `getName()`: the name recorded in a field's
    /// [`PER_FIELD_FORMAT_KEY`] and the stem of its codec suffix.
    pub fn name(&self) -> &'static str {
        if let Some(name) = self.wrapper_name {
            return name;
        }
        match self.kind {
            PostingsFormatKind::Lucene104 => FORMAT_NAME,
            PostingsFormatKind::Completion104 => COMPLETION_FORMAT_NAME,
            PostingsFormatKind::Dictionary(format) => format.name(),
        }
    }

    /// The writer options this format's files are written with.
    pub fn write_options(&self) -> WriteOptions {
        WriteOptions {
            min_items_in_block: self.min_term_block_size,
            max_items_in_block: self.max_term_block_size,
            zstd_term_blocks: self.zstd_term_blocks,
            ..WriteOptions::default()
        }
    }
}

/// One written group: its format's name, the suffix number its fields
/// record, their numbers, and its files.
#[derive(Debug)]
pub struct GroupOutput {
    /// Files already published by an installed consumer. The ordinary path
    /// leaves this empty and hands buffered bytes to the index layer instead.
    pub written_files: Vec<String>,
    /// `.doc`/`.pos`/`.pay` already written by a [`write_streaming`] sink:
    /// their buffers are empty and [`Self::files`] leaves them out.
    pub streamed_files: Vec<String>,
    pub format_name: &'static str,
    pub suffix: u32,
    pub field_numbers: Vec<i32>,
    pub output: Output,
    /// A `Completion104` group's suggester files.
    pub completion: Option<CompletionOutput>,
    /// A group written with another terms dictionary: its files by
    /// extension, in place of `output`'s `.tim`/`.tip`/`.tmd`.
    pub dictionary: Option<DictionaryFiles>,
    /// Written by a [`Lucene104PostingsFormat::direct`] format: no codec
    /// suffix, no per-field attributes.
    pub direct: bool,
}

impl GroupOutput {
    /// The group's codec suffix (`Completion104_0`), empty for a
    /// [`direct`](Self::direct) group.
    pub fn codec_suffix(&self) -> String {
        if self.direct {
            String::new()
        } else {
            codec_suffix(self.format_name, self.suffix)
        }
    }

    /// The group's file name for extension `ext` in segment `segment_name`:
    /// `_0_Lucene104_0.tim`, or `_0.tim` for a direct group.
    pub fn file_name(&self, segment_name: &str, ext: &str) -> String {
        group_file_name(segment_name, &self.codec_suffix(), ext)
    }

    /// Every file of the group by extension, for a caller that writes the
    /// buffered bytes itself: `.doc`, `.psm`, the dictionary, then `.pos`
    /// and `.pay` when the segment has them and the suggester files of a
    /// completion group. Empty when `written_files` were published already.
    pub fn files(&self) -> Vec<(&'static str, &[u8])> {
        if !self.written_files.is_empty() {
            return Vec::new();
        }
        let output = &self.output;
        // A dictionary that keeps its own postings writes neither.
        let streamed = !self.streamed_files.is_empty();
        let mut files: Vec<(&'static str, &[u8])> = [("doc", &output.doc), ("psm", &output.psm)]
            .into_iter()
            .filter(|(ext, _)| !(streamed && *ext == "doc"))
            .filter(|(_, bytes)| self.dictionary.is_none() || !bytes.is_empty())
            .map(|(ext, bytes)| (ext, bytes.as_slice()))
            .collect();
        match &self.dictionary {
            // A dictionary file the format left unused is not written.
            Some(dictionary) => files.extend(
                dictionary
                    .iter()
                    .filter(|(_, bytes)| !bytes.is_empty())
                    .map(|(ext, bytes)| (*ext, bytes.as_slice())),
            ),
            None => files.extend([
                ("tim", output.tim.as_slice()),
                ("tip", output.tip.as_slice()),
                ("tmd", output.tmd.as_slice()),
            ]),
        }
        if !output.pos.is_empty() {
            files.push(("pos", &output.pos));
        }
        if !output.pay.is_empty() {
            files.push(("pay", &output.pay));
        }
        if let Some(completion) = &self.completion {
            files.push(("lkp", &completion.lkp));
            files.push(("cmp", &completion.cmp));
        }
        files
    }
}

/// `CompletionFieldsConsumer`'s own files.
#[derive(Debug, Clone)]
pub struct CompletionOutput {
    /// `.lkp`: every completion field's FST.
    pub lkp: Vec<u8>,
    /// `.cmp`: the delegate's name and the field table.
    pub cmp: Vec<u8>,
}

/// `FieldsWriter.write`: groups `inputs` by `format_for(field name)` in
/// name order and writes each group. `name_of` names a field number.
/// `.pos`/`.pay` presence (and so the `.psm` lengths) is decided over the
/// whole segment, as `Lucene104PostingsWriter`'s constructor decides it from
/// `state.fieldInfos`: `segment_fields` is the segment's `.fnm`, which may
/// index positions on a field that got no terms, and must describe every
/// input as the `.fnm` does (checked in debug builds). An empty list (unit
/// tests with no `.fnm`) decides from `inputs` instead.
pub fn write<T: TermSource>(
    inputs: &[FieldPostingsInput<'_, T>],
    segment_fields: &[FieldInfo],
    norms: &[FieldNorms<'_>],
    name_of: &dyn Fn(i32) -> Option<String>,
    format_for: &dyn Fn(&str) -> Lucene104PostingsFormat,
    segment_id: &[u8; ID_LENGTH],
) -> postings_writer::Result<Vec<GroupOutput>> {
    debug_assert!(
        segment_fields.is_empty()
            || inputs.iter().all(|i| segment_fields.iter().any(|f| {
                f.number == i.field_number
                    && f.index_options == i.index_options
                    && (f.store_payloads && f.index_options.subsumes_positions()) == i.has_payloads
            })),
        "every postings input must be described by the segment's FieldInfos"
    );
    let (has_prox, has_payloads_or_offsets) = if segment_fields.is_empty() {
        (
            inputs.iter().any(|i| i.index_options.subsumes_positions()),
            inputs
                .iter()
                .any(|i| i.index_options.subsumes_offsets() || i.has_payloads),
        )
    } else {
        (
            crate::field_infos::has_prox(segment_fields),
            crate::field_infos::has_payloads_or_offsets(segment_fields),
        )
    };
    let named: Vec<String> = inputs
        .iter()
        .map(|input| name_of(input.field_number).unwrap_or_else(|| input.field_number.to_string()))
        .collect();
    let groups = group(&named, format_for);
    let segment_infos = FieldInfos {
        fields: segment_fields.to_vec(),
    };
    let mut out = Vec::with_capacity(groups.len());
    check_direct(&groups)?;
    for (format, suffix, members) in groups {
        let codec_suffix = group_codec_suffix(&format, suffix);
        // Each group keeps its fields in name order: `FreqProxFields` (the
        // flush, `FreqProxTermsWriter.flush` sorts its fields by name) and
        // `MultiFields` (a merge) both iterate that way, and the blocktree
        // writes fields in iteration order.
        let group_inputs: Vec<FieldPostingsInput<'_, T>> =
            members.iter().map(|&i| inputs[i]).collect();
        let options = WriteOptions {
            has_prox: Some(has_prox),
            has_payloads_or_offsets: Some(has_payloads_or_offsets),
            ..format.write_options()
        };
        // `Lucene103BlockTreeTermsWriter.TermsWriter.finish`: a field with
        // no terms (a merge's, whose every source term was deleted) gets no
        // `.tmd` record, but stays routed to the group, whose files are
        // written even when every one of its fields is empty.
        let with_terms: Vec<FieldPostingsInput<'_, T>> = group_inputs
            .iter()
            .copied()
            .filter(|input| !input.terms.is_empty())
            .collect();
        let (output, dictionary) = write_group_files(
            &format,
            &segment_infos,
            &with_terms,
            norms,
            &options,
            segment_id,
            &codec_suffix,
        )?;
        let completion = match format.kind {
            PostingsFormatKind::Lucene104 | PostingsFormatKind::Dictionary(_) => None,
            PostingsFormatKind::Completion104 => Some(
                write_completion(&group_inputs, &members, &named, segment_id, &codec_suffix)
                    .map_err(postings_writer::Error::Completion)?,
            ),
        };
        out.push(GroupOutput {
            streamed_files: Vec::new(),
            written_files: Vec::new(),
            format_name: format.name(),
            suffix,
            field_numbers: group_inputs.iter().map(|i| i.field_number).collect(),
            output,
            completion,
            dictionary,
            direct: format.direct,
        });
    }
    Ok(out)
}

/// One field [`write_streaming`] writes: what [`FieldPostingsInput`] says
/// about it besides its terms, which the caller supplies when asked.
#[derive(Debug, Clone, Copy)]
pub struct StreamedFieldSpec {
    pub field_number: i32,
    pub index_options: crate::field_infos::IndexOptions,
    pub has_payloads: bool,
}

/// [`write()`], with each field's terms supplied as the writer reaches it
/// instead of up front -- `PerFieldPostingsFormat.FieldsWriter.merge` over a
/// `MultiFields` view. `fill(i, writer)` adds field `fields[i]`'s terms in
/// order ([`StreamingFieldsWriter::add_terms`], any number of batches) and
/// returns its `docsSeen` count; fields are asked for group by group, each
/// group in name order, as [`write()`] writes them.
///
/// `Ok(None)`, before anything is asked, when some field's format is not
/// the block-tree `Lucene104` one (another terms dictionary, or
/// `Completion104`): those encoders need the whole field, so the caller
/// gathers it and calls [`write()`].
///
/// # Errors
/// What [`write()`] rejects, and what `fill` fails with.
pub fn write_streaming<E: From<postings_writer::Error>>(
    sink: Option<(&dyn lucene_store::Directory, &str)>,
    fields: &[StreamedFieldSpec],
    segment_fields: &[FieldInfo],
    name_of: &dyn Fn(i32) -> Option<String>,
    format_for: &dyn Fn(&str) -> Lucene104PostingsFormat,
    segment_id: &[u8; ID_LENGTH],
    fill: &mut dyn FnMut(usize, &mut StreamingFieldsWriter<'_>) -> Result<i32, E>,
) -> Result<Option<Vec<GroupOutput>>, E> {
    let (has_prox, has_payloads_or_offsets) = if segment_fields.is_empty() {
        (
            fields.iter().any(|f| f.index_options.subsumes_positions()),
            fields
                .iter()
                .any(|f| f.index_options.subsumes_offsets() || f.has_payloads),
        )
    } else {
        (
            crate::field_infos::has_prox(segment_fields),
            crate::field_infos::has_payloads_or_offsets(segment_fields),
        )
    };
    let named: Vec<String> = fields
        .iter()
        .map(|f| name_of(f.field_number).unwrap_or_else(|| f.field_number.to_string()))
        .collect();
    let groups = group(&named, format_for);
    if groups
        .iter()
        .any(|(format, _, _)| !matches!(format.kind, PostingsFormatKind::Lucene104))
    {
        return Ok(None);
    }
    check_direct(&groups)?;
    let mut out = Vec::with_capacity(groups.len());
    for (format, suffix, members) in groups {
        let codec_suffix = group_codec_suffix(&format, suffix);
        let options = WriteOptions {
            has_prox: Some(has_prox),
            has_payloads_or_offsets: Some(has_payloads_or_offsets),
            ..format.write_options()
        };
        let mut writer = StreamingFieldsWriter::new(&options, segment_id, &codec_suffix)?;
        if let Some((dir, segment)) = sink {
            let create = |ext: &str| -> Result<lucene_store::FsIndexOutput, E> {
                dir.create_output(&group_file_name(segment, &codec_suffix, ext))
                    .map_err(|e| E::from(postings_writer::Error::Store(e)))
            };
            let pos = if has_prox { Some(create("pos")?) } else { None };
            let pay = if has_payloads_or_offsets {
                Some(create("pay")?)
            } else {
                None
            };
            writer.stream_to(create("doc")?, pos, pay);
        }
        for &i in &members {
            let field = fields[i];
            writer.start_field(field.field_number, field.index_options, field.has_payloads)?;
            let doc_count = fill(i, &mut writer)?;
            writer.finish_field(doc_count)?;
        }
        let output = writer.finish()?;
        // `.doc`/`.pos`/`.pay` went out as they were written; the dictionary
        // and `.psm` stay buffered for the caller.
        let streamed_files = sink.map_or_else(Vec::new, |(_, segment)| {
            let mut names = vec![group_file_name(segment, &codec_suffix, "doc")];
            if has_prox {
                names.push(group_file_name(segment, &codec_suffix, "pos"));
            }
            if has_payloads_or_offsets {
                names.push(group_file_name(segment, &codec_suffix, "pay"));
            }
            names
        });
        out.push(GroupOutput {
            written_files: Vec::new(),
            streamed_files,
            format_name: format.name(),
            suffix,
            field_numbers: members.iter().map(|&i| fields[i].field_number).collect(),
            output,
            completion: None,
            dictionary: None,
            direct: format.direct,
        });
    }
    Ok(Some(out))
}

/// A group's file for extension `ext` in `segment`: `_0_Lucene104_0.tim`,
/// or `_0.tim` for a group with no codec suffix.
fn group_file_name(segment: &str, codec_suffix: &str, ext: &str) -> String {
    if codec_suffix.is_empty() {
        format!("{segment}.{ext}")
    } else {
        format!("{segment}_{codec_suffix}.{ext}")
    }
}

/// The codec suffix a group of `format` numbered `suffix` writes its files
/// and headers with: none for a [`Lucene104PostingsFormat::direct`] one.
fn group_codec_suffix(format: &Lucene104PostingsFormat, suffix: u32) -> String {
    if format.direct {
        String::new()
    } else {
        codec_suffix(format.name(), suffix)
    }
}

/// A direct format is the codec's only postings format: a segment cannot
/// mix it with another, or two groups would share the unsuffixed files.
fn check_direct(
    groups: &[(Lucene104PostingsFormat, u32, Vec<usize>)],
) -> postings_writer::Result<()> {
    if groups.len() > 1 && groups.iter().any(|(f, _, _)| f.direct) {
        return Err(postings_writer::Error::Dictionary(
            "a direct postings format must be every field's format".to_string(),
        ));
    }
    Ok(())
}

/// A non-block-tree dictionary's files by extension ([`GroupOutput::dictionary`]).
pub type DictionaryFiles = Vec<(&'static str, Vec<u8>)>;

/// One group's postings and dictionary with this port's own encoders:
/// `inputs` are the group's fields that have terms (none writes the
/// format's files with no field in them). A block-tree group's dictionary
/// is in the returned [`Output`]; another dictionary's files come back
/// beside it, `Output`'s `.tim`/`.tip`/`.tmd` left empty.
fn write_group_files<T: TermSource>(
    format: &Lucene104PostingsFormat,
    segment_field_infos: &FieldInfos,
    inputs: &[FieldPostingsInput<'_, T>],
    norms: &[FieldNorms<'_>],
    options: &WriteOptions,
    segment_id: &[u8; ID_LENGTH],
    codec_suffix: &str,
) -> postings_writer::Result<(Output, Option<DictionaryFiles>)> {
    match format.kind {
        PostingsFormatKind::Lucene104 | PostingsFormatKind::Completion104 => {
            let output = if inputs.is_empty() {
                postings_writer::write_no_fields(options, segment_id, codec_suffix)?
            } else {
                postings_writer::write_fields_with_options(
                    inputs,
                    norms,
                    options,
                    segment_id,
                    codec_suffix,
                )?
            };
            Ok((output, None))
        }
        PostingsFormatKind::Dictionary(dictionary) => {
            options.validate()?;
            let terms: Vec<Vec<&dyn TermSource>> = inputs
                .iter()
                .map(|i| i.terms.iter().map(|t| t as &dyn TermSource).collect())
                .collect();
            let erased: Vec<FieldPostingsInput<'_, &dyn TermSource>> = inputs
                .iter()
                .zip(&terms)
                .map(|(input, terms)| FieldPostingsInput {
                    field_number: input.field_number,
                    index_options: input.index_options,
                    doc_count: input.doc_count,
                    has_payloads: input.has_payloads,
                    terms: terms.as_slice(),
                })
                .collect();
            let state = crate::segment_state::SegmentWriteState {
                max_doc: 0,
                segment_id,
                segment_suffix: codec_suffix,
                field_infos: segment_field_infos,
            };
            let (postings, files) = dictionary.write(&erased, norms, options, &state)?;
            if files.len() != dictionary.extensions().len() {
                return Err(postings_writer::Error::Dictionary(format!(
                    "{} wrote {} files for {} extensions",
                    dictionary.name(),
                    files.len(),
                    dictionary.extensions().len()
                )));
            }
            let output = Output {
                doc: postings.doc,
                psm: postings.psm,
                pos: postings.pos,
                pay: postings.pay,
                ..Output::default()
            };
            Ok((
                output,
                Some(dictionary.extensions().iter().copied().zip(files).collect()),
            ))
        }
    }
}

/// Installed native consumers over the same grouping and borrowed term runs.
/// Java: `PerFieldPostingsFormat.FieldsWriter.write/merge/close`. Every acquired
/// native consumer stays open until all groups finish, including when a later
/// group fails. Completion groups keep their ordinary encoder and publication.
/// Group visitation uses first-field order; Java's HashMap order is unspecified.
pub fn write_installed<T: TermSource>(
    inputs: &[FieldPostingsInput<'_, T>],
    norms: &[FieldNorms<'_>],
    name_of: &dyn Fn(i32) -> Option<String>,
    format_for: &dyn Fn(&str) -> Lucene104PostingsFormat,
    factory: &dyn crate::postings_consumer::PostingsWriterFactory,
    state: crate::postings_consumer::PostingsWriteState<'_>,
    merge: bool,
) -> postings_writer::Result<Vec<GroupOutput>> {
    use crate::postings_consumer::PostingsWriteSink;
    let names: Vec<_> = inputs
        .iter()
        .map(|i| name_of(i.field_number).unwrap_or_else(|| i.field_number.to_string()))
        .collect();
    let segment_infos = FieldInfos {
        fields: state.field_infos.to_vec(),
    };
    let grouped = group(&names, format_for);
    check_direct(&grouped)?;
    let plans: Vec<_> = grouped
        .into_iter()
        .map(|(format, suffix, members)| {
            (format, suffix, members, group_codec_suffix(&format, suffix))
        })
        .collect();
    let mut consumers: Vec<Box<dyn PostingsWriteSink + '_>> = Vec::new();
    let result = (|| {
        let mut groups = Vec::with_capacity(plans.len());
        for (format, suffix, members, codec_suffix) in &plans {
            let selected: Vec<_> = members.iter().map(|&i| inputs[i]).collect();
            if format.kind != PostingsFormatKind::Lucene104 {
                // The installed consumer wraps the block-tree group writer;
                // a completion or other-dictionary group keeps its own.
                let options = WriteOptions {
                    has_prox: Some(crate::field_infos::has_prox(state.field_infos)),
                    has_payloads_or_offsets: Some(crate::field_infos::has_payloads_or_offsets(
                        state.field_infos,
                    )),
                    ..format.write_options()
                };
                let populated: Vec<_> = selected
                    .iter()
                    .copied()
                    .filter(|i| !i.terms.is_empty())
                    .collect();
                let (output, dictionary) = write_group_files(
                    format,
                    &segment_infos,
                    &populated,
                    norms,
                    &options,
                    state.id,
                    codec_suffix,
                )?;
                let completion = if format.kind == PostingsFormatKind::Completion104 {
                    Some(
                        write_completion(&selected, members, &names, state.id, codec_suffix)
                            .map_err(postings_writer::Error::Completion)?,
                    )
                } else {
                    None
                };
                groups.push(GroupOutput {
                    streamed_files: Vec::new(),
                    written_files: Vec::new(),
                    format_name: format.name(),
                    suffix: *suffix,
                    field_numbers: selected.iter().map(|i| i.field_number).collect(),
                    output,
                    completion,
                    dictionary,
                    direct: format.direct,
                });
                continue;
            }
            let group_state = crate::postings_consumer::PostingsWriteState {
                suffix: codec_suffix,
                format: *format,
                ..state
            };
            consumers.push(factory.fields_consumer(group_state)?);
            // One pointer per term; all document, position, offset and payload
            // runs retain their original allocation and native source type.
            let terms: Vec<Vec<&dyn TermSource>> = selected
                .iter()
                .map(|i| i.terms.iter().map(|term| term as &dyn TermSource).collect())
                .collect();
            let erased: Vec<_> = selected
                .iter()
                .zip(&terms)
                .map(|(input, terms)| FieldPostingsInput {
                    field_number: input.field_number,
                    index_options: input.index_options,
                    doc_count: input.doc_count,
                    has_payloads: input.has_payloads,
                    terms: terms.as_slice(),
                })
                .collect();
            let consumer = consumers.last_mut().expect("acquired consumer");
            if merge {
                consumer.merge(&erased, norms)?;
            } else {
                consumer.write(&erased, norms)?;
            }
            groups.push(GroupOutput {
                streamed_files: Vec::new(),
                written_files: crate::postings_consumer::output_files(group_state),
                format_name: format.name(),
                suffix: *suffix,
                field_numbers: selected.iter().map(|i| i.field_number).collect(),
                output: Output::default(),
                completion: None,
                dictionary: None,
                direct: format.direct,
            });
        }
        Ok(groups)
    })();
    let mut first_close = None;
    for consumer in &mut consumers {
        if let Err(error) = consumer.close() {
            first_close.get_or_insert(error);
        }
    }
    match result {
        Err(error) => Err(error),
        Ok(groups) => first_close.map_or(Ok(groups), Err),
    }
}

/// `CompletionFieldsConsumer.write` after its delegate: each field of the
/// group, in name order, through a [`CompletionFieldWriter`], every
/// occurrence's payload a `SuggestField` payload.
fn write_completion<T: TermSource>(
    group_inputs: &[FieldPostingsInput<'_, T>],
    members: &[usize],
    names: &[String],
    segment_id: &[u8; ID_LENGTH],
    codec_suffix: &str,
) -> lucene_store::Result<CompletionOutput> {
    let mut files = CompletionFilesWriter::new(
        COMPLETION_FORMAT_NAME,
        FORMAT_NAME,
        *segment_id,
        codec_suffix,
    );
    for (input, &member) in group_inputs.iter().zip(members) {
        let mut writer = CompletionFieldWriter::new();
        for term in input.terms {
            writer.write_term(term.term(), &term_docs(term)?)?;
        }
        files.add_field(&names[member], input.field_number, writer)?;
    }
    let (lkp, cmp) = files.finish();
    Ok(CompletionOutput { lkp, cmp })
}

/// A term's documents with the payload of each occurrence, split out of the
/// flat doc-major run ([`postings_writer::TermPostings::payload_lengths`] holds one length
/// per occurrence).
fn term_docs(term: &impl TermSource) -> lucene_store::Result<Vec<TermDoc>> {
    let inconsistent = || {
        lucene_store::Error::Corrupted(
            "completion term's payload lengths do not match its occurrences".to_string(),
        )
    };
    let mut rest: &[u8] = term.payload_bytes();
    let mut lengths = term.payload_lengths().iter();
    let mut docs = Vec::with_capacity(term.docs().len());
    for &(doc_id, freq) in term.docs() {
        let mut payloads = Vec::new();
        for _ in 0..freq {
            let length = *lengths.next().ok_or_else(inconsistent)? as usize;
            // `assert postingsEnum.getPayload() != null`: an occurrence
            // without one is not a suggestion.
            if length == 0 {
                return Err(lucene_store::Error::Corrupted(
                    "completion field occurrence has no payload".to_string(),
                ));
            }
            let (payload, tail) = rest.split_at_checked(length).ok_or_else(inconsistent)?;
            payloads.push(payload.to_vec());
            rest = tail;
        }
        docs.push(TermDoc { doc_id, payloads });
    }
    if !rest.is_empty() || lengths.next().is_some() {
        return Err(inconsistent());
    }
    Ok(docs)
}

/// `formatToGroupBuilders`: the indexes of `names` grouped by format,
/// groups in the order their first field appears in name order
/// (`FreqProxFields`' and `MultiFields`' iteration order), each with its
/// suffix number: `buildFieldsGroupMapping` numbers a format *name*'s
/// instances from 0, so `Lucene104_0` and `Completion104_0` coexist.
fn group(
    names: &[String],
    format_for: &dyn Fn(&str) -> Lucene104PostingsFormat,
) -> Vec<(Lucene104PostingsFormat, u32, Vec<usize>)> {
    let mut order: Vec<usize> = (0..names.len()).collect();
    order.sort_by(|&a, &b| names[a].encode_utf16().cmp(names[b].encode_utf16()));
    let mut groups: Vec<(Lucene104PostingsFormat, u32, Vec<usize>)> = Vec::new();
    for i in order {
        let format = format_for(&names[i]);
        match groups.iter_mut().find(|(f, _, _)| *f == format) {
            Some((_, _, members)) => members.push(i),
            None => {
                let suffix = groups
                    .iter()
                    .filter(|(f, _, _)| f.name() == format.name())
                    .count();
                groups.push((format, u32::try_from(suffix).unwrap_or(u32::MAX), vec![i]));
            }
        }
    }
    groups
}

/// The format name and suffix number [`write()`] gives each of `fields`
/// (`(number, name)`), for a caller that records the `.fnm` attributes
/// before writing the files (`SegmentMerger` via `PerFieldMergeState`).
/// Fields of a [`Lucene104PostingsFormat::direct`] format are left out:
/// they record none.
pub fn field_suffixes(
    fields: &[(i32, String)],
    format_for: &dyn Fn(&str) -> Lucene104PostingsFormat,
) -> Vec<(i32, &'static str, u32)> {
    let names: Vec<String> = fields.iter().map(|(_, name)| name.clone()).collect();
    let mut out = Vec::with_capacity(fields.len());
    // A direct format's fields record no per-field attributes.
    for (format, suffix, members) in group(&names, format_for)
        .into_iter()
        .filter(|(format, _, _)| !format.direct)
    {
        out.extend(
            members
                .into_iter()
                .map(|i| (fields[i].0, format.name(), suffix)),
        );
    }
    out.sort_unstable();
    out
}

/// The block tree's dictionary extensions, in [`GroupFiles::dictionary`]
/// order.
const BLOCK_TREE_EXTENSIONS: &[&str] = &["tim", "tip", "tmd"];

/// The format name a group's codec suffix starts with (`BlockTreeOrds_0` ->
/// `BlockTreeOrds`); the whole suffix when it has no instance number.
pub fn format_of_suffix(suffix: &str) -> &str {
    suffix.rsplit_once('_').map_or(suffix, |(name, _)| name)
}

/// Whether a group of `format` is written with a terms dictionary other than
/// the block tree ([`crate::terms_dictionary`]), which [`open_groups`]
/// rewrites as one on open.
pub fn is_foreign_format(format: &str) -> bool {
    crate::terms_dictionary::lookup(format).is_some()
}

/// The dictionary files a group of `format` has, in the order
/// [`GroupFiles::dictionary`] holds them: `.tim`, `.tip`, `.tmd` for the
/// block tree (every format not in this port's foreign list, including the
/// completion formats' delegates and older Lucene's own block trees).
pub fn dictionary_extensions(format: &str) -> &'static [&'static str] {
    crate::terms_dictionary::lookup(format).map_or(BLOCK_TREE_EXTENSIONS, |f| f.extensions())
}

/// The dictionary file whose presence marks a group of `format`.
pub fn marker_extension(format: &str) -> &'static str {
    match crate::terms_dictionary::lookup(format) {
        Some(f) => f.extensions()[0],
        None => "tmd",
    }
}

/// Whether a group of `format` must have its dictionary file `.ext`: every
/// block tree file, and a foreign format's marker. A foreign format writes
/// only the dictionary files it uses (MockRandom's seed picks its
/// dictionary), and a missing one reads as empty.
pub fn dictionary_file_required(format: &str, ext: &str) -> bool {
    !is_foreign_format(format) || marker_extension(format) == ext
}

/// Whether `file` (a loose name or a compound entry) is a dictionary file
/// of a group whose format writes no footer
/// ([`TermsDictionaryFormat::checksummed`]), which no integrity check reads.
pub fn is_unchecksummed_dictionary_file(file: &str) -> bool {
    let Some((stem, ext)) = file.rsplit_once('.') else {
        return false;
    };
    let Some((head, n)) = stem.rsplit_once('_') else {
        return false;
    };
    if !n.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    crate::terms_dictionary::all().into_iter().any(|d| {
        !d.checksummed()
            && d.extensions().contains(&ext)
            && head
                .strip_suffix(d.name())
                .is_some_and(|rest| rest.is_empty() || rest.ends_with('_'))
    })
}

/// Whether `files` hold any postings group's terms dictionary.
pub fn has_postings_files(files: &[String]) -> bool {
    files.iter().any(|f| {
        f.ends_with(".tim")
            || crate::terms_dictionary::all()
                .iter()
                .any(|d| f.ends_with(&format!(".{}", d.extensions()[0])))
    })
}

/// One group's files, as `FieldsReader` opens them.
#[derive(Debug, Clone)]
pub struct GroupFiles<'a> {
    /// The group's codec suffix (`Lucene104_1`).
    pub suffix: &'a str,
    /// The terms dictionary's files, in [`dictionary_extensions`] order for
    /// the suffix's format.
    pub dictionary: Vec<&'a [u8]>,
    pub doc: Option<&'a [u8]>,
    pub pos: Option<&'a [u8]>,
    pub pay: Option<&'a [u8]>,
}

/// Every group's dictionary as one, and the groups' `.doc`/`.pos`/`.pay`
/// concatenated (empty when no group has the file). Wrap them with
/// `DocInput::validated` and friends: every group was checked against its
/// own header and footer here.
#[derive(Debug)]
pub struct CombinedPostings {
    pub fields: BlockTreeFields,
    pub doc: Vec<u8>,
    pub pos: Vec<u8>,
    pub pay: Vec<u8>,
}

/// `FieldsReader`: opens and validates every group, then combines them.
///
/// Copies each group's `.tim`/`.tip` into the dictionary it returns; use
/// [`open_groups_shared`] when the caller already holds them shared.
pub fn open_groups(
    groups: &[GroupFiles<'_>],
    field_infos: &FieldInfos,
    segment_id: &[u8; ID_LENGTH],
    max_doc: i32,
) -> blocktree::Result<CombinedPostings> {
    open_groups_shared(groups, &[], field_infos, segment_id, max_doc)
}

/// [`open_groups`] over `.tim`/`.tip` the caller holds shared:
/// `shared[i]`, when present, is `groups[i]`'s first two dictionary files,
/// which the combined dictionary then navigates in place
/// ([`blocktree::open_shared`]) instead of copying. A group read through a
/// rewrite ignores it: its dictionary is the rewrite's own.
pub fn open_groups_shared(
    groups: &[GroupFiles<'_>],
    shared: &[Option<[SharedBytes; 2]>],
    field_infos: &FieldInfos,
    segment_id: &[u8; ID_LENGTH],
    max_doc: i32,
) -> blocktree::Result<CombinedPostings> {
    let mut combined = Vec::with_capacity(groups.len());
    let (mut doc, mut pos, mut pay) = (Vec::new(), Vec::new(), Vec::new());
    for (i, g) in groups.iter().enumerate() {
        let format = format_of_suffix(g.suffix);
        let expected = dictionary_extensions(format).len();
        if g.dictionary.len() != expected {
            return Err(blocktree::Error::Store(lucene_store::Error::Corrupted(
                format!(
                    "postings group {} has {} dictionary files, its format needs {expected}",
                    g.suffix,
                    g.dictionary.len()
                ),
            )));
        }
        // Each branch hands `open_shared` its `.tim`/`.tip` without a copy
        // where it can: a rewrite's own buffers move in, a caller's shared
        // ones are cloned by reference count.
        let (tim, tip, tmd): (SharedBytes, SharedBytes, Cow<'_, [u8]>);
        let mut reencoded = reencode_group(g, field_infos, segment_id, max_doc).transpose()?;
        if let Some(out) = &mut reencoded {
            tim = Arc::new(std::mem::take(&mut out.tim));
            tip = Arc::new(std::mem::take(&mut out.tip));
            tmd = Cow::Owned(std::mem::take(&mut out.tmd));
        } else if is_foreign_format(format) {
            let rewritten = rewrite_group_as_block_tree(g, field_infos, segment_id, max_doc)?;
            tim = Arc::new(rewritten.0);
            tip = Arc::new(rewritten.1);
            tmd = Cow::Owned(rewritten.2);
        } else if let Some(Some([shared_tim, shared_tip])) = shared.get(i) {
            tim = Arc::clone(shared_tim);
            tip = Arc::clone(shared_tip);
            tmd = Cow::Borrowed(g.dictionary[2]);
        } else {
            tim = Arc::new(g.dictionary[0].to_vec());
            tip = Arc::new(g.dictionary[1].to_vec());
            tmd = Cow::Borrowed(g.dictionary[2]);
        }
        let fields =
            blocktree::open_shared(tim, tip, &tmd, field_infos, segment_id, g.suffix, max_doc)?;
        let base = [doc.len() as u64, pos.len() as u64, pay.len() as u64];
        if let Some(out) = &reencoded {
            // Its own encoding, already consistent: no header to re-check.
            doc.extend_from_slice(&out.doc);
            pos.extend_from_slice(&out.pos);
            pay.extend_from_slice(&out.pay);
            combined.push((fields, base));
            continue;
        }
        if let Some(bytes) = g.doc {
            DocInput::open(bytes, segment_id, g.suffix)?;
            doc.extend_from_slice(bytes);
        }
        if let Some(bytes) = g.pos {
            PosInput::open(bytes, segment_id, g.suffix)?;
            pos.extend_from_slice(bytes);
        }
        if let Some(bytes) = g.pay {
            PayInput::open(bytes, segment_id, g.suffix)?;
            pay.extend_from_slice(bytes);
        }
        combined.push((fields, base));
    }
    Ok(CombinedPostings {
        fields: BlockTreeFields::combine(combined)?,
        doc,
        pos,
        pay,
    })
}

/// Reads a foreign group's dictionary with its own reader and writes it
/// back as a block tree under the same suffix, every term keeping its
/// `IntBlockTermState`: the postings files are the group's own, untouched.
///
/// Differs from Java: Java reads the group with that format's own
/// `FieldsProducer`. The terms, statistics and postings every query sees are
/// the same; the format's extras (`BlockTreeOrds`' term ords) are not served.
pub fn rewrite_group_as_block_tree(
    group: &GroupFiles<'_>,
    field_infos: &FieldInfos,
    segment_id: &[u8; ID_LENGTH],
    max_doc: i32,
) -> blocktree::Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let state = crate::segment_state::SegmentReadState {
        max_doc,
        segment_id,
        segment_suffix: group.suffix,
        field_infos,
    };
    let Some(format) = crate::terms_dictionary::lookup(format_of_suffix(group.suffix)) else {
        return Err(blocktree::Error::Unsupported(
            "a foreign postings group with no dictionary reader",
        ));
    };
    let fields = format.read(&group.dictionary, &state)?;
    postings_writer::write_block_tree_dictionary(
        &fields,
        &WriteOptions::default(),
        segment_id,
        group.suffix,
    )
    .map_err(|e| blocktree::Error::ForeignDictionary {
        format: "block tree rewrite",
        message: e.to_string(),
    })
}

/// A group of a format that keeps its own postings
/// ([`TermsDictionaryFormat::read_postings`]: `RAMOnly`), encoded as a
/// block tree over `Lucene104` postings under the group's suffix: its
/// `.tim`/`.tip`/`.tmd` and `.doc`/`.pos`/`.pay` in the returned output.
/// `None` for any other group.
///
/// Differs from Java: Java serves such a group through the format's own
/// `FieldsProducer`. The terms, statistics and postings are the same; terms
/// enumerate in byte order, and impacts carry no norms (every bound is the
/// loosest), since the format keeps none.
pub fn reencode_group(
    group: &GroupFiles<'_>,
    field_infos: &FieldInfos,
    segment_id: &[u8; ID_LENGTH],
    max_doc: i32,
) -> Option<blocktree::Result<Output>> {
    let format = crate::terms_dictionary::lookup(format_of_suffix(group.suffix))?;
    let state = crate::segment_state::SegmentReadState {
        max_doc,
        segment_id,
        segment_suffix: group.suffix,
        field_infos,
    };
    let fields = match format.read_postings(&group.dictionary, &state)? {
        Ok(fields) => fields,
        Err(e) => return Some(Err(e)),
    };
    let inputs: Vec<FieldPostingsInput<'_, postings_writer::TermPostings>> = fields
        .iter()
        .filter(|f| !f.terms.is_empty())
        .map(|f| FieldPostingsInput {
            field_number: f.field_number,
            index_options: f.index_options,
            doc_count: f.doc_count,
            has_payloads: f.has_payloads,
            terms: &f.terms,
        })
        .collect();
    let written = if inputs.is_empty() {
        postings_writer::write_no_fields(&WriteOptions::default(), segment_id, group.suffix)
    } else {
        postings_writer::write_fields_with_options(
            &inputs,
            &[],
            &WriteOptions::default(),
            segment_id,
            group.suffix,
        )
    };
    Some(written.map_err(|e| blocktree::Error::ForeignDictionary {
        format: "postings re-encode",
        message: e.to_string(),
    }))
}

/// The codec suffixes of a segment's postings groups -- one per `.tmd`
/// among `files` (loose names or compound entry names), or per marker file
/// of a foreign dictionary ([`marker_extension`]) -- sorted.
pub fn group_suffixes(files: &[String], segment_name: &str) -> Vec<String> {
    let loose = format!("{segment_name}_");
    let mut suffixes: Vec<String> = files
        .iter()
        .filter_map(|f| {
            f.strip_suffix(".tmd").or_else(|| {
                crate::terms_dictionary::all().into_iter().find_map(|d| {
                    let stem = f.strip_suffix(&format!(".{}", d.extensions()[0]))?;
                    // A marker extension only counts under its own format:
                    // the stem ends `<name>_<n>`, not merely contains it
                    // (`SharedTermsUniformSplit_0` is no `UniformSplit_0`).
                    let (head, n) = stem.rsplit_once('_')?;
                    let own = head
                        .strip_suffix(d.name())
                        .is_some_and(|rest| rest.is_empty() || rest.ends_with('_'));
                    (own && n.bytes().all(|b| b.is_ascii_digit())).then_some(stem)
                })
            })
        })
        .map(|stem| {
            if stem.is_empty() || stem == segment_name {
                // No suffix at all (`_0.tmd`, or `.tmd` inside a compound
                // file).
                ""
            } else {
                stem.strip_prefix(&loose)
                    .or_else(|| stem.strip_prefix('_'))
                    .unwrap_or(stem)
            }
        })
        .map(str::to_string)
        .collect();
    suffixes.sort();
    suffixes.dedup();
    suffixes
}

#[cfg(test)]
mod tests;
