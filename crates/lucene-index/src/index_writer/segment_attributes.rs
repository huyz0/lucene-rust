//! `.si` attributes that list names, comma separated, as OpenSearch k-NN's derived-source stored
//! fields format records its fields (`DerivedSourceSegmentAttributeParser`): a flushed segment
//! records the names added under a key ([`IndexWriter::add_segment_list_value`]), and a merge
//! records, under each key [`IndexWriter::set_merge_union_attributes`] names, every name its
//! sources record.

use super::segment_partitions::set_attribute;
use super::{IndexWriter, IndexingConfig, Result};
use crate::segment_infos::SegmentCommitInfo;

impl IndexingConfig {
    /// The list attributes a flushed segment records.
    pub(crate) fn record_list_attributes(&self, attributes: &mut Vec<(String, String)>) {
        for (key, values) in &self.segment_list_attributes {
            if !values.is_empty() {
                set_attribute(attributes, key, &values.join(","));
            }
        }
    }
}

impl IndexWriter<'_> {
    /// Adds `value` to the list attribute `key` every segment flushed from now on records.
    pub fn add_segment_list_value(&mut self, key: &str, value: &str) {
        let lists = &mut self.cfg_mut().segment_list_attributes;
        let slot = match lists.iter().position(|(k, _)| k == key) {
            Some(i) => &mut lists[i].1,
            None => {
                lists.push((key.to_string(), Vec::new()));
                &mut lists.last_mut().expect("just pushed").1
            }
        };
        if !slot.iter().any(|v| v == value) {
            slot.push(value.to_string());
        }
    }

    /// The list attributes a merge carries over: under each of `keys`, the names any source
    /// records.
    pub fn set_merge_union_attributes(&mut self, keys: Vec<String>) {
        self.cfg_mut().merge_union_attributes = keys;
    }

    /// The union attributes a merge of `sources` records.
    pub(crate) fn merged_union_attributes(
        &self,
        sources: &[SegmentCommitInfo],
    ) -> Result<Vec<(String, String)>> {
        let keys = &self.cfg.merge_union_attributes;
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let mut lists: Vec<Vec<String>> = vec![Vec::new(); keys.len()];
        for sci in sources {
            let si = self.cfg.reader_pool.segment_info(self.dir, sci)?;
            for (key, list) in keys.iter().zip(lists.iter_mut()) {
                let Some((_, value)) = si.attributes.iter().find(|(k, _)| k == key) else {
                    continue;
                };
                for name in value.split(',').filter(|s| !s.is_empty()) {
                    if !list.iter().any(|v| v == name) {
                        list.push(name.to_string());
                    }
                }
            }
        }
        Ok(keys
            .iter()
            .zip(lists)
            .filter(|(_, list)| !list.is_empty())
            .map(|(key, list)| (key.clone(), list.join(",")))
            .collect())
    }
}
