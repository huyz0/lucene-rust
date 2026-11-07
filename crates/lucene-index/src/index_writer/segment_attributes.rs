//! `.si` attributes the writer adds to its segments, as OpenSearch k-NN's derived-source stored
//! fields format records them:
//!
//! - lists of names, comma separated (`DerivedSourceSegmentAttributeParser`): a flushed segment
//!   records the names added under a key ([`IndexWriter::add_segment_list_value`]); a merge
//!   records, under each key [`IndexWriter::set_merge_union_attributes`] names, every name its
//!   sources record and every name the writer adds;
//! - single values ([`IndexWriter::set_segment_attribute`]), such as the stored-fields codec a
//!   segment is written with: every flushed and merged segment records the writer's value.

use super::segment_partitions::set_attribute;
use super::{IndexWriter, IndexingConfig, Result};
use crate::segment_infos::SegmentCommitInfo;

impl IndexingConfig {
    /// The attributes a flushed segment records.
    pub(crate) fn record_list_attributes(&self, attributes: &mut Vec<(String, String)>) {
        for (key, values) in &self.segment_list_attributes {
            if !values.is_empty() {
                set_attribute(attributes, key, &values.join(","));
            }
        }
        for (key, value) in &self.segment_attributes {
            set_attribute(attributes, key, value);
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

    /// Records `key` as `value` on every segment flushed or merged from now on.
    pub fn set_segment_attribute(&mut self, key: &str, value: &str) {
        let fixed = &mut self.cfg_mut().segment_attributes;
        match fixed.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => value.clone_into(&mut slot.1),
            None => fixed.push((key.to_string(), value.to_string())),
        }
    }

    /// The value [`Self::set_segment_attribute`] set for `key`.
    pub fn segment_attribute(&self, key: &str) -> Option<&str> {
        self.cfg
            .segment_attributes
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// The list attributes a merge carries over: under each of `keys`, the names any source
    /// records.
    pub fn set_merge_union_attributes(&mut self, keys: Vec<String>) {
        self.cfg_mut().merge_union_attributes = keys;
    }

    /// Adds what a merge of `sources` records to `attributes`: the union lists, then the
    /// single values.
    pub(crate) fn merged_segment_attributes(
        &self,
        sources: &[SegmentCommitInfo],
        attributes: &mut Vec<(String, String)>,
    ) -> Result<()> {
        let keys = &self.cfg.merge_union_attributes;
        let mut lists: Vec<Vec<String>> = vec![Vec::new(); keys.len()];
        if !keys.is_empty() {
            for sci in sources {
                let si = self.cfg.reader_pool.segment_info(self.dir, sci)?;
                for (key, list) in keys.iter().zip(lists.iter_mut()) {
                    if let Some((_, value)) = si.attributes.iter().find(|(k, _)| k == key) {
                        union(list, value.split(','));
                    }
                }
            }
        }
        for (key, list) in keys.iter().zip(lists.iter_mut()) {
            if let Some((_, own)) = self
                .cfg
                .segment_list_attributes
                .iter()
                .find(|(k, _)| k == key)
            {
                union(list, own.iter().map(String::as_str));
            }
            if !list.is_empty() {
                set_attribute(attributes, key, &list.join(","));
            }
        }
        for (key, value) in &self.cfg.segment_attributes {
            set_attribute(attributes, key, value);
        }
        Ok(())
    }
}

fn union<'a>(list: &mut Vec<String>, names: impl Iterator<Item = &'a str>) {
    for name in names.filter(|s| !s.is_empty()) {
        if !list.iter().any(|v| v == name) {
            list.push(name.to_string());
        }
    }
}
