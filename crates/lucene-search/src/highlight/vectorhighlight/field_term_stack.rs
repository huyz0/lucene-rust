//! `FieldTermStack`: the occurrences of a query's terms in one document's
//! term vector, in position order.

use std::collections::VecDeque;

use super::super::{Error, Result};
use super::field_query::FieldQuery;
use crate::directory_reader::DirectoryReader;
use crate::reader::LeafReader;

/// `FieldTermStack.TermInfo`.
#[derive(Debug, Clone)]
pub struct TermInfo {
    text: String,
    start_offset: i32,
    end_offset: i32,
    position: i32,
    weight: f32,
    /// The occurrence at the same position, the ring closing on the first.
    next: usize,
}

impl TermInfo {
    /// `getText()`.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// `getStartOffset()`.
    pub fn start_offset(&self) -> i32 {
        self.start_offset
    }

    /// `getEndOffset()`.
    pub fn end_offset(&self) -> i32 {
        self.end_offset
    }

    /// `getPosition()`.
    pub fn position(&self) -> i32 {
        self.position
    }

    /// `getWeight()`.
    pub fn weight(&self) -> f32 {
        self.weight
    }
}

impl std::fmt::Display for TermInfo {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}({},{},{})",
            self.text, self.start_offset, self.end_offset, self.position
        )
    }
}

/// `FieldTermStack`. The occurrences live in an arena; `list` holds the
/// first of each position, the others are reached through
/// [`FieldTermStack::next_at_position`].
#[derive(Debug)]
pub struct FieldTermStack {
    field: String,
    arena: Vec<TermInfo>,
    list: VecDeque<usize>,
}

impl FieldTermStack {
    /// `FieldTermStack(reader, docId, fieldName, fieldQuery)`.
    ///
    /// A document without a term vector of positions for the field, or
    /// without offsets in it, leaves the stack empty.
    ///
    /// # Errors
    /// A document outside the reader, or a term vector that cannot be read.
    pub fn new(
        reader: &DirectoryReader,
        doc: i32,
        field: &str,
        field_query: &FieldQuery,
    ) -> Result<Self> {
        let mut this = Self {
            field: field.to_string(),
            arena: Vec::new(),
            list: VecDeque::new(),
        };
        let Some(term_set) = field_query.term_set(field) else {
            return Ok(this);
        };
        let segment = reader
            .segment_readers()
            .iter()
            .find(|s| s.doc_base <= doc && doc - s.doc_base < s.max_doc)
            .ok_or_else(|| Error::IllegalArgument(format!("doc {doc} is in no segment")))?;
        let local = doc - segment.doc_base;
        let Some(number) = segment.field_infos().field_by_name(field).map(|f| f.number) else {
            return Ok(this);
        };
        let Some(vectors) = segment
            .term_vectors(local)
            .map_err(|e| Error::Io(e.to_string()))?
        else {
            return Ok(this);
        };
        let Some(vector) = vectors.fields.iter().find(|f| f.field_number == number) else {
            return Ok(this);
        };
        if !vector.has_positions {
            return Ok(this);
        }
        let num_docs = f64::from(reader.max_doc());
        for t in &vector.terms {
            let text = String::from_utf8_lossy(&t.term).into_owned();
            if !term_set.contains(&text) {
                continue;
            }
            let doc_freq = reader
                .doc_freq(field, &t.term)
                .map_err(|e| Error::Io(e.to_string()))?;
            // `(float) (Math.log(numDocs / (double) (docFreq + 1)) + 1.0)`
            let weight = ((num_docs / (doc_freq + 1) as f64).ln() + 1.0) as f32;
            let positions = t.positions.as_deref().unwrap_or(&[]);
            for (i, &position) in positions.iter().enumerate() {
                let (Some(start), Some(end)) = (
                    t.start_offsets.as_ref().and_then(|o| o.get(i)),
                    t.end_offsets.as_ref().and_then(|o| o.get(i)),
                ) else {
                    return Ok(this); // no offsets, null snippet
                };
                if *start < 0 {
                    return Ok(this);
                }
                let id = this.arena.len();
                this.arena.push(TermInfo {
                    text: text.clone(),
                    start_offset: *start,
                    end_offset: *end,
                    position,
                    weight,
                    next: id,
                });
                this.list.push_back(id);
            }
        }
        // `Collections.sort`: stable, by position
        let mut sorted: Vec<usize> = this.list.drain(..).collect();
        sorted.sort_by_key(|&i| this.arena[i].position);
        let mut current_pos = -1;
        let mut previous: Option<usize> = None;
        let mut first: Option<usize> = None;
        for id in sorted {
            let position = this.arena[id].position;
            if position == current_pos {
                if let Some(p) = previous {
                    this.arena[p].next = id;
                }
                previous = Some(id);
            } else {
                if let (Some(p), Some(f)) = (previous, first) {
                    this.arena[p].next = f;
                }
                previous = Some(id);
                first = Some(id);
                current_pos = position;
                this.list.push_back(id);
            }
        }
        if let (Some(p), Some(f)) = (previous, first) {
            this.arena[p].next = f;
        }
        Ok(this)
    }

    /// `getFieldName()`.
    pub fn field_name(&self) -> &str {
        &self.field
    }

    /// `pop()`: the first occurrence, `None` for an empty stack.
    pub fn pop(&mut self) -> Option<usize> {
        self.list.pop_front()
    }

    /// `push(termInfo)`.
    pub fn push(&mut self, id: usize) {
        self.list.push_front(id);
    }

    /// `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// The occurrence `id` stands for.
    pub fn info(&self, id: usize) -> &TermInfo {
        &self.arena[id]
    }

    /// `getNext()`: the next occurrence at the same position (the first
    /// again after the last).
    pub fn next_at_position(&self, id: usize) -> usize {
        self.arena[id].next
    }
}
