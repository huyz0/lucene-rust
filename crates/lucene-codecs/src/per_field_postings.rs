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

/// `PerFieldPostingsFormat.PER_FIELD_FORMAT_KEY`.
pub const PER_FIELD_FORMAT_KEY: &str = "PerFieldPostingsFormat.format";
/// `PerFieldPostingsFormat.PER_FIELD_SUFFIX_KEY`.
pub const PER_FIELD_SUFFIX_KEY: &str = "PerFieldPostingsFormat.suffix";
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PostingsFormatKind {
    /// `Lucene104PostingsFormat`.
    #[default]
    Lucene104,
    /// `Completion104PostingsFormat`: a default `Lucene104PostingsFormat`
    /// delegate plus the suggester files.
    Completion104,
}

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

    /// `getName()`: the name recorded in a field's
    /// [`PER_FIELD_FORMAT_KEY`] and the stem of its codec suffix.
    pub fn name(&self) -> &'static str {
        match self.kind {
            PostingsFormatKind::Lucene104 => FORMAT_NAME,
            PostingsFormatKind::Completion104 => COMPLETION_FORMAT_NAME,
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
    pub format_name: &'static str,
    pub suffix: u32,
    pub field_numbers: Vec<i32>,
    pub output: Output,
    /// A `Completion104` group's suggester files.
    pub completion: Option<CompletionOutput>,
}

impl GroupOutput {
    /// The group's codec suffix (`Completion104_0`).
    pub fn codec_suffix(&self) -> String {
        codec_suffix(self.format_name, self.suffix)
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
        let output = if with_terms.is_empty() {
            postings_writer::write_no_fields(&options, segment_id, &codec_suffix)?
        } else {
            postings_writer::write_fields_with_options(
                &with_terms,
                norms,
                &options,
                segment_id,
                &codec_suffix,
            )?
        };
        let completion = match format.kind {
            PostingsFormatKind::Lucene104 => None,
            PostingsFormatKind::Completion104 => Some(
                write_completion(&group_inputs, &members, &named, segment_id, &codec_suffix)
                    .map_err(postings_writer::Error::Completion)?,
            ),
        };
        out.push(GroupOutput {
            format_name: format.name(),
            suffix,
            field_numbers: group_inputs.iter().map(|i| i.field_number).collect(),
            output,
            completion,
        });
    }
    Ok(out)
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
    order.sort_by(|&a, &b| names[a].cmp(&names[b]));
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

/// One group's files, as `FieldsReader` opens them.
#[derive(Debug, Clone, Copy)]
pub struct GroupFiles<'a> {
    /// The group's codec suffix (`Lucene104_1`).
    pub suffix: &'a str,
    pub tim: &'a [u8],
    pub tip: &'a [u8],
    pub tmd: &'a [u8],
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
        let fields = blocktree::open(
            g.tim,
            g.tip,
            g.tmd,
            field_infos,
            segment_id,
            g.suffix,
            max_doc,
        )?;
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

/// The codec suffixes of a segment's postings groups -- one per `.tmd` among
/// `files` (loose names or compound entry names) -- sorted.
pub fn group_suffixes(files: &[String], segment_name: &str) -> Vec<String> {
    let loose = format!("{segment_name}_");
    let mut suffixes: Vec<String> = files
        .iter()
        .filter_map(|f| f.strip_suffix(".tmd"))
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
                tim: &g.output.tim,
                tip: &g.output.tip,
                tmd: &g.output.tmd,
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
        dup.push(files[0]);
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
