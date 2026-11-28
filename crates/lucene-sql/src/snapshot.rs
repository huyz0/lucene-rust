//! A point-in-time view of an index and the table declared over it.
//!
//! A [`Snapshot`] owns the reader (so deletions, segments and doc values are
//! those of the commit it was opened at) and what is cheap to keep per
//! snapshot: each segment's field numbers and each keyword column's terms. A
//! search borrows the segments through [`Snapshot::with_searcher`], the way
//! the engine opens them for every search.

use std::collections::HashMap;
use std::sync::Arc;

use lucene_codecs::doc_values::SortedSetKind;
use lucene_codecs::stored_fields::{DocumentVisitor, FieldValue};
use lucene_codecs::terms_dict::TermsDict;
use lucene_search::directory_reader::{DirectoryReader, SegmentReader};
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::multi_segment::OpenSegment;

use crate::error::{Error, Result};
use crate::spec::{ColumnKind, TableSpec};

/// An index at one commit, with the table declared over it.
pub struct Snapshot {
    reader: DirectoryReader,
    spec: Arc<TableSpec>,
    /// `field_numbers[segment][column]`: the field's number in that segment
    /// (numbers are assigned per segment), `None` when the segment lacks it.
    field_numbers: Vec<Vec<Option<i32>>>,
    /// `keyword_terms[segment][column]`: the terms of a keyword column in
    /// ordinal order; empty for other columns and segments without values.
    keyword_terms: Vec<Vec<Vec<String>>>,
    /// `keyword_dense[column]`: every document of every segment has a value
    /// (so a `GROUP BY` has no NULL group). False for other columns.
    keyword_dense: Vec<bool>,
}

/// The segments of a snapshot opened for one search.
pub struct Searcher<'s, 'a> {
    /// Searches the whole index with the cross-segment pruning bound.
    pub searcher: &'s IndexSearcher<'s, 'a>,
    /// The segments, with their doc bases and live docs.
    pub segments: &'s [OpenSegment<'a>],
    /// Each segment's reader.
    pub leaves: &'s [SegmentReader],
}

impl Snapshot {
    /// A snapshot of `reader` read as `spec`.
    ///
    /// # Errors
    /// An invalid `spec`, or an unreadable keyword column.
    pub fn new(reader: DirectoryReader, spec: TableSpec) -> Result<Self> {
        spec.validate()?;
        let mut field_numbers = Vec::new();
        let mut keyword_terms = Vec::new();
        for leaf in reader.segment_readers() {
            let mut numbers = Vec::with_capacity(spec.columns().len());
            let mut terms = Vec::with_capacity(spec.columns().len());
            for c in spec.columns() {
                numbers.push(leaf.field_infos().field_by_name(&c.name).map(|f| f.number));
                terms.push(match c.kind {
                    ColumnKind::Keyword => keyword_terms_of(leaf, &c.name)?,
                    _ => Vec::new(),
                });
            }
            field_numbers.push(numbers);
            keyword_terms.push(terms);
        }
        let keyword_dense = spec
            .columns()
            .iter()
            .enumerate()
            .map(|(c, col)| {
                col.kind == ColumnKind::Keyword
                    && reader
                        .segment_readers()
                        .iter()
                        .enumerate()
                        .all(|(s, leaf)| keyword_is_dense(leaf, field_numbers[s][c]))
            })
            .collect();
        Ok(Self {
            reader,
            spec: Arc::new(spec),
            field_numbers,
            keyword_terms,
            keyword_dense,
        })
    }

    /// The table declaration.
    pub fn spec(&self) -> &Arc<TableSpec> {
        &self.spec
    }

    /// The reader.
    pub fn reader(&self) -> &DirectoryReader {
        &self.reader
    }

    /// Segments in the snapshot.
    pub fn segment_count(&self) -> usize {
        self.reader.segment_readers().len()
    }

    /// Documents including deleted ones.
    pub fn max_doc(&self) -> i32 {
        self.reader.max_doc()
    }

    /// Live documents.
    pub fn num_docs(&self) -> i32 {
        self.reader.num_docs()
    }

    /// Whether every document has a value for the keyword column `column`.
    pub fn keyword_is_dense(&self, column: usize) -> bool {
        self.keyword_dense.get(column).copied().unwrap_or(false)
    }

    pub(crate) fn leaf_max_doc(&self, segment: usize) -> i32 {
        self.leaf(segment).max_doc
    }

    pub(crate) fn leaf(&self, segment: usize) -> &SegmentReader {
        &self.reader.segment_readers()[segment]
    }

    pub(crate) fn field_number(&self, segment: usize, column: usize) -> Option<i32> {
        self.field_numbers
            .get(segment)
            .and_then(|s| s.get(column))
            .copied()
            .flatten()
    }

    pub(crate) fn terms(&self, segment: usize, column: usize) -> &[String] {
        &self.keyword_terms[segment][column]
    }

    /// Opens the segments and runs `f` over them.
    ///
    /// # Errors
    /// Unreadable segment files, or `f`'s own error.
    pub fn with_searcher<R>(&self, f: impl FnOnce(&Searcher<'_, '_>) -> Result<R>) -> Result<R> {
        let text_fields: Vec<String> = self
            .spec
            .columns()
            .iter()
            .filter(|c| c.kind == ColumnKind::Text)
            .map(|c| c.name.clone())
            .collect();
        let opened = self.reader.open_segments()?;
        let segments = opened.as_open_segments();
        let norms = self.reader.field_norms_by_field(&text_fields)?;
        let norm_refs: Vec<SegmentNorms<'_, '_>> = norms.iter().map(Some).collect();
        let searcher = IndexSearcher::new(&segments, &norm_refs)?;
        f(&Searcher {
            searcher: &searcher,
            segments: &segments,
            leaves: self.reader.segment_readers(),
        })
    }

    /// `(segment, segment-local doc)` of a global doc id.
    pub(crate) fn locate(&self, doc: i32) -> Result<(usize, i32)> {
        let leaves = self.reader.segment_readers();
        let segment = leaves
            .iter()
            .rposition(|l| l.doc_base <= doc)
            .ok_or_else(|| Error::unsupported(format!("doc {doc} is outside the index")))?;
        Ok((segment, doc.saturating_sub(leaves[segment].doc_base)))
    }

    /// The stored string values of `columns` for one segment-local document.
    pub(crate) fn stored_strings(
        &self,
        segment: usize,
        local: i32,
        columns: &[usize],
    ) -> Result<Vec<Option<String>>> {
        let numbers: Vec<Option<i32>> = columns
            .iter()
            .map(|&c| self.field_number(segment, c))
            .collect();
        let wanted: Vec<i32> = numbers.iter().flatten().copied().collect();
        let mut visitor = DocumentVisitor::for_fields(&wanted);
        let _ = self
            .leaf(segment)
            .visit_stored_document(local, &mut visitor)?;
        let mut by_number: HashMap<i32, String> = HashMap::new();
        for f in &visitor.document().fields {
            if let FieldValue::String(s) = &f.value {
                by_number.entry(f.field_number).or_insert_with(|| s.clone());
            }
        }
        Ok(numbers
            .iter()
            .map(|n| n.and_then(|n| by_number.remove(&n)))
            .collect())
    }
}

fn keyword_terms_of(leaf: &SegmentReader, field: &str) -> Result<Vec<String>> {
    let Some((_, data, entry)) = lucene_search::global_ords::keyword_column(leaf, field)? else {
        return Ok(Vec::new());
    };
    let bad =
        |e: &dyn std::fmt::Display| Error::unsupported(format!("keyword column `{field}`: {e}"));
    // Through the dictionary's ordinal lookup, which shows a masked field's
    // values as masked.
    let mut dict = TermsDict::open(data, entry).map_err(|e| bad(&e))?;
    (0..entry.terms_dict_size)
        .map(|ord| {
            let term = dict.lookup_ord(ord).map_err(|e| bad(&e))?;
            String::from_utf8(term.to_vec()).map_err(|_| bad(&"not UTF-8"))
        })
        .collect()
}

fn keyword_is_dense(leaf: &SegmentReader, number: Option<i32>) -> bool {
    if leaf.max_doc == 0 {
        return true;
    }
    let Some(n) = number else { return false };
    let Some((meta, _)) = leaf.doc_values_for_field(n) else {
        return false;
    };
    if let Some(e) = meta.sorted_entry(n) {
        return e.ords.is_dense();
    }
    matches!(
        meta.sorted_set_entry(n).map(|e| &e.kind),
        Some(SortedSetKind::Single(se)) if se.ords.is_dense()
    )
}
