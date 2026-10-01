//! The write half of `org.apache.lucene.codecs.perfield.PerFieldDocValuesFormat`
//! for the doc-values format this port has, `Lucene90DocValuesFormat` with
//! any `skipIndexIntervalSize` ([`Lucene90DocValuesFormat`]).
//!
//! `FieldsWriter.getInstance`: each field goes to the format its caller
//! routes it to (`getDocValuesFormatForField`); the first field of a format
//! instance opens that instance's consumer under the next suffix number of
//! the format's *name*, so the instances of `Lucene90` in a segment are
//! `Lucene90_0`, `Lucene90_1`, ... in the order the consumer first sees a
//! field of each ([`group_fields`]). Each field records its instance in its
//! `PerFieldDocValuesFormat.format`/`.suffix` attributes; each instance's
//! files carry the codec suffix `<name>_<n>` ([`codec_suffix`]) in their
//! names and headers.
//!
//! The order a consumer sees fields in is the caller's: `IndexingChain`'s
//! field hash at flush, the merged `FieldInfos` (field-number order) at a
//! merge. An update generation does not come here: Java writes it with the
//! format its `.fnm` attribute names, looked up *by name* -- the default
//! instance -- under the field's own suffix.
//!
//! What this does not model: Java keys instances by identity, so two
//! distinct but equal instances get two suffixes; a format here is its
//! setting, and equal settings share one group. The read side is
//! `lucene-index`'s `field_updates::read_current_column` (each field opens the
//! instance its attributes name).

pub use crate::doc_values::Lucene90DocValuesFormat;

/// `PerFieldDocValuesFormat.PER_FIELD_FORMAT_KEY`.
pub const PER_FIELD_FORMAT_KEY: &str = "PerFieldDocValuesFormat.format";
/// `PerFieldDocValuesFormat.PER_FIELD_SUFFIX_KEY`.
pub const PER_FIELD_SUFFIX_KEY: &str = "PerFieldDocValuesFormat.suffix";

/// `PerFieldDocValuesFormat.getSuffix(formatName, suffix)` for
/// `Lucene90`: the codec suffix an instance's files and headers carry.
pub fn codec_suffix(suffix: u32) -> String {
    format!("{}_{suffix}", Lucene90DocValuesFormat::NAME)
}

/// One format instance's share of a segment: its suffix number, its format,
/// and its fields' numbers in the order the consumer received them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    pub suffix: u32,
    pub format: Lucene90DocValuesFormat,
    pub field_numbers: Vec<i32>,
}

/// `FieldsWriter.getInstance` over `fields` (`(number, name)`) in the order
/// the consumer receives them: each field joins the group of
/// `format_for(name)`, a new format opening the next group, numbered from 0
/// in order of first appearance.
pub fn group_fields(
    fields: &[(i32, &str)],
    format_for: &dyn Fn(&str) -> Lucene90DocValuesFormat,
) -> Vec<Group> {
    let mut groups: Vec<Group> = Vec::new();
    for &(number, name) in fields {
        let format = format_for(name);
        match groups.iter_mut().find(|g| g.format == format) {
            Some(group) => group.field_numbers.push(number),
            None => {
                let suffix = u32::try_from(groups.len()).unwrap_or(u32::MAX);
                groups.push(Group {
                    suffix,
                    format,
                    field_numbers: vec![number],
                });
            }
        }
    }
    groups
}

/// The suffix number [`group_fields`] gives field `number`, if any.
pub fn suffix_of(groups: &[Group], number: i32) -> Option<u32> {
    groups
        .iter()
        .find(|g| g.field_numbers.contains(&number))
        .map(|g| g.suffix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instances_are_numbered_in_order_of_first_appearance() {
        let small = Lucene90DocValuesFormat::new(1024).unwrap();
        let route = |name: &str| {
            if name.starts_with("dvb") {
                small
            } else {
                Lucene90DocValuesFormat::default()
            }
        };
        // The consumer sees `dvb_num` first: the small format is `_0`.
        let groups = group_fields(&[(5, "dvb_num"), (4, "dva_num"), (6, "dvb_sorted")], &route);
        assert_eq!(
            groups,
            vec![
                Group {
                    suffix: 0,
                    format: small,
                    field_numbers: vec![5, 6]
                },
                Group {
                    suffix: 1,
                    format: Lucene90DocValuesFormat::default(),
                    field_numbers: vec![4]
                },
            ]
        );
        assert_eq!(suffix_of(&groups, 6), Some(0));
        assert_eq!(suffix_of(&groups, 4), Some(1));
        assert_eq!(suffix_of(&groups, 9), None);
        assert_eq!(codec_suffix(1), "Lucene90_1");
        assert!(group_fields(&[], &route).is_empty());
    }

    #[test]
    fn the_interval_is_validated_as_java_does() {
        assert!(Lucene90DocValuesFormat::new(1).is_err());
        assert_eq!(
            Lucene90DocValuesFormat::new(2)
                .unwrap()
                .skip_index_interval_size(),
            2
        );
        assert_eq!(
            Lucene90DocValuesFormat::default().skip_index_interval_size(),
            Lucene90DocValuesFormat::DEFAULT_SKIP_INDEX_INTERVAL_SIZE
        );
    }
}
