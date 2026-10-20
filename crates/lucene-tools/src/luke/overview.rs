//! `org.apache.lucene.luke.models.overview`: the index at a glance -- its
//! fields, documents and terms, the commit it was opened at, and a field's
//! most frequent terms.

use std::collections::HashMap;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::misc::high_freq::{get_high_freq_terms, TermOrder};
use lucene_search::multi_terms::MultiTerms;
use lucene_search::reader::{merged_field_infos, IndexReader};

use super::{decode, read_map_string, LukeError, Result};

/// `TermCountsOrder`: how [`Overview::sorted_term_counts`] orders the fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TermCountsOrder {
    NameAsc,
    NameDesc,
    CountAsc,
    CountDesc,
}

impl TermCountsOrder {
    /// `TermCountsOrder.values()`.
    pub const ALL: [TermCountsOrder; 4] = [
        TermCountsOrder::NameAsc,
        TermCountsOrder::NameDesc,
        TermCountsOrder::CountAsc,
        TermCountsOrder::CountDesc,
    ];

    /// `name()`.
    pub fn name(self) -> &'static str {
        match self {
            TermCountsOrder::NameAsc => "NAME_ASC",
            TermCountsOrder::NameDesc => "NAME_DESC",
            TermCountsOrder::CountAsc => "COUNT_ASC",
            TermCountsOrder::CountDesc => "COUNT_DESC",
        }
    }
}

/// `overview.TermStats`: a term, decoded, with its field and document
/// frequency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TermStats {
    pub decoded_term_text: String,
    pub field: String,
    pub doc_freq: i32,
}

impl std::fmt::Display for TermStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TermStats{{decodedTermText='{}', field='{}', docFreq={}}}",
            self.decoded_term_text, self.field, self.doc_freq
        )
    }
}

/// The field names of a reader, in field-number order
/// (`IndexUtils.getFieldNames` over `FieldInfos.getMergedFieldInfos`).
pub fn field_names(reader: &DirectoryReader) -> Vec<String> {
    merged_field_infos(&reader.leaves())
        .fields
        .into_iter()
        .map(|f| f.name)
        .collect()
}

/// `OverviewImpl`.
pub struct Overview<'r> {
    reader: &'r DirectoryReader,
    /// `TermCounts`: every field and its number of terms, in the order a
    /// `java.util.HashMap` filled in field order iterates them.
    term_counts: Vec<(String, i64)>,
    /// `TopTerms`' cache.
    top_terms: HashMap<String, Vec<TermStats>>,
}

impl<'r> Overview<'r> {
    /// `new OverviewImpl(reader, indexPath)`: counts every field's terms.
    ///
    /// # Errors
    /// `LukeException` when the terms cannot be read.
    pub fn new(reader: &'r DirectoryReader) -> Result<Self> {
        let names = field_names(reader);
        let mut counts = Vec::with_capacity(names.len());
        for name in &names {
            let count = count_terms(reader, name).map_err(|_| {
                LukeError::luke("An error occurred when collecting term statistics.")
            })?;
            counts.push(count);
        }
        let keys: Vec<&str> = names.iter().map(String::as_str).collect();
        let term_counts = lucene_search::mlt::java_hash_order(&keys)
            .into_iter()
            .map(|i| (names[i].clone(), counts[i]))
            .collect();
        Ok(Overview {
            reader,
            term_counts,
            top_terms: HashMap::new(),
        })
    }

    /// `getNumFields()`.
    pub fn num_fields(&self) -> usize {
        merged_field_infos(&self.reader.leaves()).fields.len()
    }

    /// `getNumDocuments()`.
    pub fn num_documents(&self) -> i32 {
        self.reader.num_docs()
    }

    /// `getNumTerms()`: the terms of every field.
    pub fn num_terms(&self) -> i64 {
        self.term_counts.iter().map(|(_, c)| c).sum()
    }

    /// `hasDeletions()`.
    pub fn has_deletions(&self) -> bool {
        IndexReader::has_deletions(self.reader)
    }

    /// `getNumDeletedDocs()`.
    pub fn num_deleted_docs(&self) -> i32 {
        IndexReader::num_deleted_docs(self.reader)
    }

    /// `isOptimized()`: whether the commit has one segment.
    pub fn is_optimized(&self) -> Option<bool> {
        Some(self.reader.segment_infos.segments.len() == 1)
    }

    /// `getIndexVersion()`: `DirectoryReader.getVersion()`.
    pub fn index_version(&self) -> Option<i64> {
        Some(self.reader.segment_infos.version)
    }

    /// `getIndexFormat()`: `IndexUtils.getIndexFormat`, named from the
    /// `segments_N` header's version.
    pub fn index_format(&self) -> Option<String> {
        Some(index_format(self.reader.segment_infos.format_version).to_string())
    }

    /// `getCommitDescription()`: `segments_N (generation=N, segs=K)`.
    pub fn commit_description(&self) -> Option<String> {
        let infos = &self.reader.segment_infos;
        Some(format!(
            "{} (generation={}, segs={})",
            segments_file_name(infos.generation),
            infos.generation,
            infos.segments.len()
        ))
    }

    /// `getCommitUserData()`: the user data map's `toString()`.
    pub fn commit_user_data(&self) -> Option<String> {
        Some(read_map_string(&self.reader.segment_infos.user_data))
    }

    /// `getFieldNames()`.
    pub fn field_names(&self) -> Vec<String> {
        field_names(self.reader)
    }

    /// `getSortedTermCounts(order)`: each field's term count, sorted (a stable
    /// sort over the `HashMap`'s order, as Java's stream sorts).
    pub fn sorted_term_counts(&self, order: TermCountsOrder) -> Vec<(String, i64)> {
        let mut counts = self.term_counts.clone();
        match order {
            TermCountsOrder::NameAsc => counts.sort_by(|a, b| a.0.cmp(&b.0)),
            TermCountsOrder::NameDesc => counts.sort_by(|a, b| b.0.cmp(&a.0)),
            TermCountsOrder::CountAsc => counts.sort_by_key(|c| c.1),
            TermCountsOrder::CountDesc => counts.sort_by_key(|c| std::cmp::Reverse(c.1)),
        }
        counts
    }

    /// `getTopTerms(field, numTerms)`: `HighFreqTerms` by document frequency,
    /// cached per field; a cached list at least `num_terms` long is returned
    /// whole.
    ///
    /// # Errors
    /// `IllegalArgumentException` for a negative `num_terms`;
    /// `LukeException` when the field has no terms.
    pub fn top_terms(&mut self, field: &str, num_terms: i32) -> Result<Vec<TermStats>> {
        let Ok(n) = usize::try_from(num_terms) else {
            return Err(LukeError::illegal_argument(format!(
                "'numTerms' must be a positive integer: {num_terms} is not accepted."
            )));
        };
        let cached = self.top_terms.get(field).is_some_and(|t| t.len() >= n);
        if !cached {
            let stats = get_high_freq_terms(self.reader, n, Some(field), TermOrder::DocFreq)
                .map_err(|_| {
                    LukeError::luke(format!("Top terms for field {field} not available."))
                })?;
            let top = stats
                .iter()
                .map(|s| TermStats {
                    decoded_term_text: decode(&s.termtext),
                    field: s.field.clone(),
                    doc_freq: s.doc_freq,
                })
                .collect();
            self.top_terms.insert(field.to_string(), top);
        }
        Ok(self.top_terms.get(field).cloned().unwrap_or_default())
    }
}

/// `IndexUtils.countTerms` for one field: its terms across the index.
fn count_terms(reader: &DirectoryReader, field: &str) -> lucene_search::Result<i64> {
    let Some(terms) = MultiTerms::get_terms(reader, field)? else {
        return Ok(0);
    };
    let mut te = terms.iterator()?;
    let mut count = 0i64;
    while te.next()?.is_some() {
        count = count.saturating_add(1);
    }
    Ok(count)
}

/// `IndexUtils.getIndexFormat`'s name for a `segments_N` format version.
pub fn index_format(version: i32) -> &'static str {
    // SegmentInfos.VERSION_74 / VERSION_86
    match version {
        9 => "Lucene 7.4 or later",
        10 => "Lucene 8.6 or later",
        v if v > 10 => "Lucene 8.6 or later (UNSUPPORTED)",
        _ => "unknown",
    }
}

/// `IndexFileNames.fileNameFromGeneration("segments", "", generation)`.
pub fn segments_file_name(generation: i64) -> String {
    lucene_store::directory::segments_file_name(generation)
        .unwrap_or_else(|| "segments".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_are_named_by_header_version() {
        assert_eq!(index_format(9), "Lucene 7.4 or later");
        assert_eq!(index_format(10), "Lucene 8.6 or later");
        assert_eq!(index_format(11), "Lucene 8.6 or later (UNSUPPORTED)");
        assert_eq!(index_format(3), "unknown");
        assert_eq!(segments_file_name(36), "segments_10");
        let names: Vec<&str> = TermCountsOrder::ALL.iter().map(|o| o.name()).collect();
        assert_eq!(names, ["NAME_ASC", "NAME_DESC", "COUNT_ASC", "COUNT_DESC"]);
    }
}
