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
//! What this does not model: Java groups by format *instance* identity, so
//! two distinct but equal instances get two suffixes; here a format is its
//! block sizes and kind, and equal ones share a group.

use lucene_store::codec_util::ID_LENGTH;

use crate::blocktree::{self, BlockTreeFields};
use crate::completion::{CompletionFieldWriter, CompletionFilesWriter, TermDoc};
use crate::field_infos::{FieldInfo, FieldInfos};
use crate::postings::{DocInput, PayInput, PosInput};
use crate::postings_writer::{
    self, FieldNorms, FieldPostingsInput, Output, TermSource, WriteOptions,
};
use crate::terms_dictionary::TermsDictionaryFormat;

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
}

impl Default for Lucene104PostingsFormat {
    fn default() -> Self {
        let options = WriteOptions::default();
        Lucene104PostingsFormat {
            min_term_block_size: options.min_items_in_block,
            max_term_block_size: options.max_items_in_block,
            kind: PostingsFormatKind::Lucene104,
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

    /// `getName()`: the name recorded in a field's
    /// [`PER_FIELD_FORMAT_KEY`] and the stem of its codec suffix.
    pub fn name(&self) -> &'static str {
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
    pub format_name: &'static str,
    pub suffix: u32,
    pub field_numbers: Vec<i32>,
    pub output: Output,
    /// A `Completion104` group's suggester files.
    pub completion: Option<CompletionOutput>,
    /// A group written with another terms dictionary: its files by
    /// extension, in place of `output`'s `.tim`/`.tip`/`.tmd`.
    pub dictionary: Option<DictionaryFiles>,
}

impl GroupOutput {
    /// The group's codec suffix (`Completion104_0`).
    pub fn codec_suffix(&self) -> String {
        codec_suffix(self.format_name, self.suffix)
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
        let mut files: Vec<(&'static str, &[u8])> =
            vec![("doc", &output.doc), ("psm", &output.psm)];
        match &self.dictionary {
            Some(dictionary) => files.extend(
                dictionary
                    .iter()
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
    for (format, suffix, members) in groups {
        let codec_suffix = codec_suffix(format.name(), suffix);
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
            written_files: Vec::new(),
            format_name: format.name(),
            suffix,
            field_numbers: group_inputs.iter().map(|i| i.field_number).collect(),
            output,
            completion,
            dictionary,
        });
    }
    Ok(out)
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
    let plans: Vec<_> = group(&names, format_for)
        .into_iter()
        .map(|(format, suffix, members)| {
            (format, suffix, members, codec_suffix(format.name(), suffix))
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
                    written_files: Vec::new(),
                    format_name: format.name(),
                    suffix: *suffix,
                    field_numbers: selected.iter().map(|i| i.field_number).collect(),
                    output,
                    completion,
                    dictionary,
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
                written_files: crate::postings_consumer::output_files(group_state),
                format_name: format.name(),
                suffix: *suffix,
                field_numbers: selected.iter().map(|i| i.field_number).collect(),
                output: Output::default(),
                completion: None,
                dictionary: None,
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
pub fn field_suffixes(
    fields: &[(i32, String)],
    format_for: &dyn Fn(&str) -> Lucene104PostingsFormat,
) -> Vec<(i32, &'static str, u32)> {
    let names: Vec<String> = fields.iter().map(|(_, name)| name.clone()).collect();
    let mut out = Vec::with_capacity(fields.len());
    for (format, suffix, members) in group(&names, format_for) {
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
pub fn open_groups(
    groups: &[GroupFiles<'_>],
    field_infos: &FieldInfos,
    segment_id: &[u8; ID_LENGTH],
    max_doc: i32,
) -> blocktree::Result<CombinedPostings> {
    let mut combined = Vec::with_capacity(groups.len());
    let (mut doc, mut pos, mut pay) = (Vec::new(), Vec::new(), Vec::new());
    for g in groups {
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
        let rewritten;
        let (tim, tip, tmd) = if is_foreign_format(format) {
            rewritten = rewrite_group_as_block_tree(g, field_infos, segment_id, max_doc)?;
            (&rewritten.0[..], &rewritten.1[..], &rewritten.2[..])
        } else {
            (g.dictionary[0], g.dictionary[1], g.dictionary[2])
        };
        let fields = blocktree::open(tim, tip, tmd, field_infos, segment_id, g.suffix, max_doc)?;
        let base = [doc.len() as u64, pos.len() as u64, pay.len() as u64];
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
mod tests {
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use crate::field_infos::IndexOptions;
    use crate::postings_writer::TermPostings;

    #[test]
    fn routed_groups_require_their_own_dictionary_even_without_terms() {
        let routed = |name: &str, suffix: &str| {
            let mut field = FieldInfo::new(name, 0);
            field.index_options = crate::field_infos::IndexOptions::Docs;
            field.attributes = vec![
                (PER_FIELD_FORMAT_KEY.into(), FORMAT_NAME.into()),
                (PER_FIELD_SUFFIX_KEY.into(), suffix.into()),
            ];
            field
        };
        let fields = [routed("populated", "0"), routed("termless", "1")];
        for prefix in ["_0_", "_"] {
            let files = vec![
                format!("{prefix}Lucene104_0.tmd"),
                format!("{prefix}Lucene104_1.tmd"),
            ];
            assert!(validate_routed_groups(&fields, &files).is_ok());
            let error = validate_routed_groups(&fields, &files[..1]).unwrap_err();
            assert!(error.contains("termless") && error.contains("Lucene104_1"));
        }
        let mut field = fields[0].clone();
        for attributes in [
            vec![field.attributes[0].clone()],
            vec![field.attributes[1].clone()],
            vec![
                (PER_FIELD_FORMAT_KEY.into(), String::new()),
                (PER_FIELD_SUFFIX_KEY.into(), "0".into()),
            ],
        ] {
            field.attributes = attributes;
            assert!(validate_routed_groups(&[field.clone()], &[])
                .unwrap_err()
                .contains("incomplete"));
        }
        field.attributes.clear();
        assert!(validate_routed_groups(&[field.clone()], &[]).is_ok());
        assert!(validate_routed_groups(&[field.clone()], &["_0.tmd".into()]).is_ok());
        field.index_options = crate::field_infos::IndexOptions::None;
        field.attributes = fields[0].attributes.clone();
        assert!(validate_routed_groups(&[field], &[]).is_ok());
    }

    #[test]
    fn routed_simpletext_groups_require_their_own_postings_text_file() {
        let mut text = FieldInfo::new("text", 0);
        text.index_options = IndexOptions::Docs;
        text.attributes = vec![
            (PER_FIELD_FORMAT_KEY.into(), "SimpleText".into()),
            (PER_FIELD_SUFFIX_KEY.into(), "0".into()),
        ];
        let mut binary = text.clone();
        binary.name = "binary".into();
        binary.number = 1;
        binary.attributes[0].1 = FORMAT_NAME.into();
        let fields = [text, binary];
        for prefix in ["_0_", "_"] {
            let files = vec![
                format!("{prefix}SimpleText_0.pst"),
                format!("{prefix}Lucene104_0.tmd"),
            ];
            assert!(validate_routed_groups(&fields, &files).is_ok());
            assert!(validate_routed_groups(&fields, &files[1..])
                .unwrap_err()
                .contains("missing postings group SimpleText_0"));
            assert!(validate_routed_groups(&fields, &files[..1])
                .unwrap_err()
                .contains("missing postings group Lucene104_0"));
            let wrong_extension = vec![format!("{prefix}SimpleText_0.tmd"), files[1].clone()];
            assert!(validate_routed_groups(&fields, &wrong_extension).is_err());
            let wrong_suffix = vec![format!("{prefix}SimpleText_1.pst"), files[1].clone()];
            assert!(validate_routed_groups(&fields, &wrong_suffix).is_err());
        }
    }

    #[test]
    fn suffix_assignment_and_group_iteration_follow_java_utf16_order() {
        // UTF-8 puts U+E000 first; Java compares the U+D800 surrogate first.
        let names = vec![
            "\u{e000}".to_string(),
            "\u{10000}".to_string(),
            "a".to_string(),
        ];
        let route = |name: &str| {
            if name == "\u{e000}" {
                Lucene104PostingsFormat::new(2, 3).unwrap()
            } else {
                Lucene104PostingsFormat::default()
            }
        };
        let groups = group(&names, &route);
        assert_eq!(
            groups
                .iter()
                .map(|(_, suffix, members)| (*suffix, members.clone()))
                .collect::<Vec<_>>(),
            [(0, vec![2, 1]), (1, vec![0])]
        );
        let fields = vec![
            (0, names[0].clone()),
            (1, names[1].clone()),
            (2, names[2].clone()),
        ];
        assert_eq!(
            field_suffixes(&fields, &route),
            [
                (0, "Lucene104", 1),
                (1, "Lucene104", 0),
                (2, "Lucene104", 0)
            ]
        );
    }

    fn terms(n: usize, doc: i32) -> Vec<TermPostings> {
        (0..n)
            .map(|i| TermPostings {
                term: format!("t{i:04}").into_bytes(),
                docs: vec![(doc, 1), (doc + 1, 1)],
                ..Default::default()
            })
            .collect()
    }

    /// A suffix names its completion format by everything before its last
    /// `_`; Lucene 8's formats and every other format are none.
    #[test]
    fn completion_suffixes_name_their_format() {
        assert_eq!(completion_format_of("Completion90_0"), Some("Completion90"));
        assert_eq!(
            completion_format_of("Completion912_3"),
            Some("Completion912")
        );
        assert_eq!(
            completion_format_of("Completion104_0"),
            Some("Completion104")
        );
        for other in [
            "Completion84_0",
            "completion_0",
            "Lucene104_0",
            "",
            "Completion90",
        ] {
            assert_eq!(completion_format_of(other), None, "{other}");
        }
        assert!(is_completion_format("Completion101"));
        assert!(!is_completion_format("Lucene101"));
        let delegates: Vec<&str> = COMPLETION_FORMAT_NAMES
            .iter()
            .map(|n| {
                crate::postings::PostingsFormat::completion_delegate(n)
                    .unwrap()
                    .name()
            })
            .collect();
        assert_eq!(
            delegates,
            [
                "Lucene90",
                "Lucene99",
                "Lucene912",
                "Lucene101",
                "Lucene104"
            ]
        );
        assert_eq!(
            crate::postings::PostingsFormat::completion_delegate("Completion84"),
            None
        );
    }

    /// A field whose `.fnm` indexes positions and offsets but which got no
    /// terms still frames `.pos`/`.pay` in `.psm`, as Java's
    /// `state.fieldInfos.hasProx()` does: `Lucene104PostingsReader` reads
    /// those lengths off the FieldInfos, and without them it found 8 bytes
    /// where the footer belonged (`checksum status indeterminate`).
    #[test]
    fn psm_frames_pos_and_pay_from_the_segment_fields_not_the_inputs() {
        let docs_only = terms(5, 0);
        let inputs = [FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::Docs,
            doc_count: 2,
            has_payloads: false,
            terms: &docs_only,
        }];
        let segment_fields = [
            FieldInfo {
                index_options: IndexOptions::Docs,
                ..FieldInfo::new("id", 0)
            },
            FieldInfo {
                index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
                ..FieldInfo::new("body", 1)
            },
        ];
        let run = |fields: &[FieldInfo]| {
            write(
                &inputs,
                fields,
                &[],
                &|n| Some(["id", "body"][n as usize].to_string()),
                &|_| Lucene104PostingsFormat::default(),
                &[3; ID_LENGTH],
            )
            .unwrap()
            .remove(0)
            .output
        };
        let (bare, framed) = (run(&segment_fields[..1]), run(&segment_fields));
        assert!(bare.pos.is_empty() && bare.pay.is_empty());
        assert!(!framed.pos.is_empty() && !framed.pay.is_empty());
        // Two more longs: the `.pos` and the `.pay` lengths.
        assert_eq!(framed.psm.len(), bare.psm.len() + 16);
    }

    /// Two fields to the small format, one to the default: two groups,
    /// numbered in the name order of their first field, each field listed
    /// once; the second group has `.pos` because the segment has positions.
    #[test]
    fn fields_are_grouped_by_format_in_name_order() {
        let mut a = terms(60, 0);
        for t in &mut a {
            t.positions = vec![vec![3], vec![5]];
        }
        let (b, c) = (terms(60, 2), terms(5, 4));
        let inputs = [
            FieldPostingsInput {
                field_number: 0,
                index_options: IndexOptions::DocsAndFreqsAndPositions,
                doc_count: 2,
                has_payloads: false,
                terms: &a,
            },
            FieldPostingsInput {
                field_number: 1,
                index_options: IndexOptions::Docs,
                doc_count: 2,
                has_payloads: false,
                terms: &b,
            },
            FieldPostingsInput {
                field_number: 2,
                index_options: IndexOptions::Docs,
                doc_count: 2,
                has_payloads: false,
                terms: &c,
            },
        ];
        let names = ["z_pos", "b_small", "a_small"];
        let small = Lucene104PostingsFormat::new(10, 20).unwrap();
        let groups = write(
            &inputs,
            &[],
            &[],
            &|n| names.get(n as usize).map(|s| s.to_string()),
            &|name| {
                if name.ends_with("small") {
                    small
                } else {
                    Lucene104PostingsFormat::default()
                }
            },
            &[1; ID_LENGTH],
        )
        .unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].suffix, 0);
        // Name order within a group too: `a_small` (2) before `b_small` (1).
        assert_eq!(groups[0].field_numbers, [2, 1]);
        assert_eq!(groups[1].field_numbers, [0]);
        assert!(!groups[0].output.pos.is_empty(), "segment-level .pos");
        assert!(groups[0].output.pay.is_empty());
        assert_eq!(codec_suffix(FORMAT_NAME, 1), "Lucene104_1");
        assert!(Lucene104PostingsFormat::new(1, 20).is_err());
        let fields: Vec<(i32, String)> =
            (0..3).map(|n| (n, names[n as usize].to_string())).collect();
        assert_eq!(
            field_suffixes(&fields, &|name| if name.ends_with("small") {
                small
            } else {
                Lucene104PostingsFormat::default()
            }),
            [
                (0, FORMAT_NAME, 1),
                (1, FORMAT_NAME, 0),
                (2, FORMAT_NAME, 0)
            ]
        );

        // Read back: every field from its own group, at its own offsets.
        let infos = FieldInfos {
            fields: (0..3)
                .map(|n| crate::field_infos::FieldInfo {
                    index_options: inputs[n as usize].index_options,
                    ..crate::field_infos::FieldInfo::new(names[n as usize], n)
                })
                .collect(),
        };
        let suffixes: Vec<String> = groups
            .iter()
            .map(|g| codec_suffix(FORMAT_NAME, g.suffix))
            .collect();
        let files: Vec<GroupFiles<'_>> = groups
            .iter()
            .zip(&suffixes)
            .map(|(g, suffix)| GroupFiles {
                suffix,
                dictionary: vec![&g.output.tim, &g.output.tip, &g.output.tmd],
                doc: Some(&g.output.doc),
                pos: (!g.output.pos.is_empty()).then_some(&g.output.pos[..]),
                pay: None,
            })
            .collect();
        let combined = open_groups(&files, &infos, &[1; ID_LENGTH], 6).unwrap();
        let doc_in = DocInput::validated(&combined.doc);
        for (n, name) in names.iter().enumerate() {
            let field = combined.fields.field(name).unwrap();
            let p = field.postings(b"t0003", Some(&doc_in)).unwrap().unwrap();
            let doc = [0, 2, 4][n];
            assert_eq!(p.docs, [doc, doc + 1], "{name}");
        }
        let mut dup = files.clone();
        dup.push(files[0].clone());
        assert!(open_groups(&dup, &infos, &[1; ID_LENGTH], 6).is_err());
        assert_eq!(
            group_suffixes(
                &[
                    "_0_Lucene104_1.tmd".into(),
                    "_0_Lucene104_0.tmd".into(),
                    "_0_Lucene104_0.tim".into(),
                    "_0.tmd".into(),
                    "_Lucene104_0.tmd".into(),
                ],
                "_0"
            ),
            ["", "Lucene104_0", "Lucene104_1"]
        );
        // A foreign marker counts only under its own format's name.
        assert_eq!(
            group_suffixes(
                &[
                    "_0_SharedTermsUniformSplit_0.stustd".into(),
                    "_0_SharedTermsUniformSplit_0.ustd".into(),
                    "_0_UniformSplit_1.ustd".into(),
                    "_0_BlockTreeOrds_0.ustd".into(),
                ],
                "_0"
            ),
            ["SharedTermsUniformSplit_0", "UniformSplit_1"]
        );
    }

    /// A suggestion payload: vint surface length, surface, vint weight + 1,
    /// type 0 (`SuggestField.buildSuggestPayload`).
    fn suggestion(surface: &str, weight: u8) -> Vec<u8> {
        let mut p = vec![surface.len() as u8];
        p.extend_from_slice(surface.as_bytes());
        p.extend_from_slice(&[weight + 1, 0]);
        p
    }

    fn suggest_terms(words: &[&str]) -> Vec<TermPostings> {
        words
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let payloads = [suggestion(w, i as u8), suggestion(w, 9)];
                TermPostings {
                    term: w.as_bytes().to_vec(),
                    docs: vec![(i as i32, 1), (i as i32 + 10, 1)],
                    positions: vec![vec![0], vec![0]],
                    payload_lengths: payloads.iter().map(|p| p.len() as u32).collect(),
                    payload_bytes: payloads.concat(),
                    ..Default::default()
                }
            })
            .collect()
    }

    /// Completion fields get their own `Completion104_0` group, numbered
    /// apart from `Lucene104_0`, with `.lkp`/`.cmp` that open as Lucene's
    /// reader opens them.
    #[test]
    fn completion_fields_are_their_own_format() {
        let completion = Lucene104PostingsFormat::completion104();
        assert_eq!(completion.name(), COMPLETION_FORMAT_NAME);
        assert_eq!(Lucene104PostingsFormat::default().name(), FORMAT_NAME);
        assert_eq!(
            completion.write_options().min_items_in_block,
            Lucene104PostingsFormat::default().min_term_block_size
        );
        let (s, t) = (suggest_terms(&["car", "cat"]), suggest_terms(&["dog"]));
        let text = terms(3, 0);
        let input = |field_number, index_options, has_payloads, terms| FieldPostingsInput {
            field_number,
            index_options,
            doc_count: 2,
            has_payloads,
            terms,
        };
        let inputs = [
            input(0, IndexOptions::DocsAndFreqsAndPositions, true, &s[..]),
            input(1, IndexOptions::Docs, false, &text[..]),
            input(2, IndexOptions::DocsAndFreqsAndPositions, true, &t[..]),
        ];
        let names = ["suggest", "body", "title"];
        let route = |name: &str| {
            if name == "body" {
                Lucene104PostingsFormat::default()
            } else {
                completion
            }
        };
        let id = [7; ID_LENGTH];
        let groups = write(
            &inputs,
            &[],
            &[],
            &|n| names.get(n as usize).map(|s| s.to_string()),
            &route,
            &id,
        )
        .unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].codec_suffix(), "Lucene104_0");
        assert!(groups[0].completion.is_none());
        assert_eq!(groups[1].codec_suffix(), "Completion104_0");
        assert_eq!(groups[1].field_numbers, [0, 2]);
        let fields: Vec<(i32, String)> =
            (0..3).map(|n| (n, names[n as usize].to_string())).collect();
        assert_eq!(
            field_suffixes(&fields, &route),
            [
                (0, COMPLETION_FORMAT_NAME, 0),
                (1, FORMAT_NAME, 0),
                (2, COMPLETION_FORMAT_NAME, 0)
            ]
        );
        let files = groups[1].completion.clone().unwrap();
        let opened = crate::completion::CompletionFields::open(
            COMPLETION_FORMAT_NAME,
            &id,
            "Completion104_0",
            &files.cmp,
            files.lkp,
        )
        .unwrap();
        assert_eq!(opened.delegate_format, FORMAT_NAME);
        for number in [0, 2] {
            assert!(opened.suggester(number).unwrap().is_some(), "{number}");
        }
        assert!(opened.suggester(1).unwrap().is_none());
    }

    /// `assert postingsEnum.getPayload() != null`: an occurrence of a
    /// completion field with no payload is refused rather than indexed as no
    /// suggestion, and so are mixed suggestion types in one field.
    #[test]
    fn a_completion_occurrence_without_a_payload_is_refused() {
        let mut empty = suggest_terms(&["car"]);
        let second = empty[0].payload_lengths[1] as usize;
        empty[0].payload_lengths[1] = 0;
        let keep = empty[0].payload_bytes.len() - second;
        empty[0].payload_bytes.truncate(keep);
        let mut mixed = suggest_terms(&["car"]);
        // the second occurrence's type byte: 1, beside the first's 0
        *mixed[0].payload_bytes.last_mut().unwrap() = 1;
        for terms in [empty, mixed] {
            let inputs = [FieldPostingsInput {
                field_number: 0,
                index_options: IndexOptions::DocsAndFreqsAndPositions,
                doc_count: 2,
                has_payloads: true,
                terms: &terms,
            }];
            let err = write(
                &inputs,
                &[],
                &[],
                &|_| Some("suggest".to_string()),
                &|_| Lucene104PostingsFormat::completion104(),
                &[7; ID_LENGTH],
            );
            assert!(
                matches!(err, Err(postings_writer::Error::Completion(_))),
                "{err:?}"
            );
        }
    }

    /// A term whose payload lengths disagree with its occurrences or its
    /// payload bytes is an internal inconsistency, not a missing payload.
    #[test]
    fn inconsistent_payload_runs_are_refused() {
        let good = suggest_terms(&["car"]).remove(0);
        assert!(term_docs(&good).is_ok());
        let mut short = good.clone();
        short.payload_lengths.pop();
        let mut extra_length = good.clone();
        extra_length.payload_lengths.push(1);
        let mut overlong = good.clone();
        *overlong.payload_lengths.last_mut().unwrap() += 1;
        let mut trailing = good.clone();
        trailing.payload_bytes.push(0);
        for term in [short, extra_length, overlong, trailing] {
            let err = term_docs(&term).unwrap_err().to_string();
            assert!(err.contains("do not match"), "{err}");
        }
    }
}
