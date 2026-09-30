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
//! What this does not model: Java groups by format *instance* identity, so
//! two distinct but equal instances get two suffixes; here a format is its
//! block sizes, and equal sizes share a group.

use lucene_store::codec_util::ID_LENGTH;

use crate::blocktree::{self, BlockTreeFields};
use crate::field_infos::FieldInfos;
use crate::postings::{DocInput, PayInput, PosInput};
use crate::postings_writer::{self, FieldNorms, FieldPostingsInput, Output, WriteOptions};

/// `PerFieldPostingsFormat.PER_FIELD_FORMAT_KEY`.
pub const PER_FIELD_FORMAT_KEY: &str = "PerFieldPostingsFormat.format";
/// `PerFieldPostingsFormat.PER_FIELD_SUFFIX_KEY`.
pub const PER_FIELD_SUFFIX_KEY: &str = "PerFieldPostingsFormat.suffix";
/// `Lucene104PostingsFormat.getName()`.
pub const FORMAT_NAME: &str = "Lucene104";

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
}

impl Default for Lucene104PostingsFormat {
    fn default() -> Self {
        let options = WriteOptions::default();
        Lucene104PostingsFormat {
            min_term_block_size: options.min_items_in_block,
            max_term_block_size: options.max_items_in_block,
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
        };
        format.write_options().validate()?;
        Ok(format)
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

/// One written group: the suffix number its fields record, their numbers,
/// and its files.
#[derive(Debug)]
pub struct GroupOutput {
    pub suffix: u32,
    pub field_numbers: Vec<i32>,
    pub output: Output,
}

/// `FieldsWriter.write`: groups `inputs` by `format_for(field name)` in
/// name order and writes each group. `name_of` names a field number.
/// `.pos`/`.pay` presence is decided over *all* `inputs`, the segment's
/// postings fields.
pub fn write(
    inputs: &[FieldPostingsInput<'_>],
    norms: &[FieldNorms<'_>],
    name_of: &dyn Fn(i32) -> Option<String>,
    format_for: &dyn Fn(&str) -> Lucene104PostingsFormat,
    segment_id: &[u8; ID_LENGTH],
) -> postings_writer::Result<Vec<GroupOutput>> {
    let has_prox = inputs.iter().any(|i| i.index_options.subsumes_positions());
    let has_payloads_or_offsets = inputs
        .iter()
        .any(|i| i.index_options.subsumes_offsets() || i.has_payloads);
    let named: Vec<String> = inputs
        .iter()
        .map(|input| name_of(input.field_number).unwrap_or_else(|| input.field_number.to_string()))
        .collect();
    let groups = group(&named, format_for);
    let mut out = Vec::with_capacity(groups.len());
    for (suffix, (format, members)) in (0u32..).zip(groups) {
        // Each group keeps its fields in the order the caller gave them.
        let mut members = members;
        members.sort_unstable();
        let group_inputs: Vec<FieldPostingsInput<'_>> =
            members.iter().map(|&i| inputs[i]).collect();
        let options = WriteOptions {
            has_prox: Some(has_prox),
            has_payloads_or_offsets: Some(has_payloads_or_offsets),
            ..format.write_options()
        };
        let output = postings_writer::write_fields_with_options(
            &group_inputs,
            norms,
            &options,
            segment_id,
            &codec_suffix(FORMAT_NAME, suffix),
        )?;
        out.push(GroupOutput {
            suffix,
            field_numbers: group_inputs.iter().map(|i| i.field_number).collect(),
            output,
        });
    }
    Ok(out)
}

/// `formatToGroupBuilders`: the indexes of `names` grouped by format,
/// groups in the order their first field appears in name order
/// (`FreqProxFields`' and `MultiFields`' iteration order).
fn group(
    names: &[String],
    format_for: &dyn Fn(&str) -> Lucene104PostingsFormat,
) -> Vec<(Lucene104PostingsFormat, Vec<usize>)> {
    let mut order: Vec<usize> = (0..names.len()).collect();
    order.sort_by(|&a, &b| names[a].cmp(&names[b]));
    let mut groups: Vec<(Lucene104PostingsFormat, Vec<usize>)> = Vec::new();
    for i in order {
        let format = format_for(&names[i]);
        match groups.iter_mut().find(|(f, _)| *f == format) {
            Some((_, members)) => members.push(i),
            None => groups.push((format, vec![i])),
        }
    }
    groups
}

/// The suffix number [`write()`] gives each of `fields` (`(number, name)`),
/// for a caller that records the `.fnm` attributes before writing the files
/// (`SegmentMerger` via `PerFieldMergeState`).
pub fn field_suffixes(
    fields: &[(i32, String)],
    format_for: &dyn Fn(&str) -> Lucene104PostingsFormat,
) -> Vec<(i32, u32)> {
    let names: Vec<String> = fields.iter().map(|(_, name)| name.clone()).collect();
    let mut out = Vec::with_capacity(fields.len());
    for (suffix, (_, members)) in (0u32..).zip(group(&names, format_for)) {
        out.extend(members.into_iter().map(|i| (fields[i].0, suffix)));
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
        assert_eq!(groups[0].field_numbers, [1, 2]);
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
            [(0, 1), (1, 0), (2, 0)]
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
}
