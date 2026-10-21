//! A one-document index read straight off a stored term vector: what
//! `MemoryIndex.addField(field, new TokenStreamFromTermVector(vector, -1))`
//! (offsets kept, payloads not) holds, without replaying the vector as a
//! token stream and hashing it back into terms. A term vector is already
//! that index's shape -- each distinct term once, in term order, with its
//! occurrences -- so the reader takes the terms as they are and counts the
//! stream's statistics (tokens, overlaps, last position) for the norm.

use std::sync::Arc;

use lucene_codecs::postings_writer::TermPostings;
use lucene_codecs::term_vectors::{TermVectorField, TermVectorTerm};

use super::reader::{Document, FieldPostings, MemoryIndexReader};
use super::MemoryIndex;
use crate::Result;

/// `Short.MAX_VALUE`: `TokenStreamFromTermVector` cuts a longer term or
/// offset span.
const SHORT_MAX: i32 = i16::MAX as i32;

/// A term's positions and its occurrences' `(start, end)` offsets.
type Occurrences = (Vec<i32>, Vec<(i32, i32)>);

/// One term's occurrences, as the stream would index them: `None` when
/// they are not in the plain case [`MemoryIndex::vector_reader`] covers.
fn occurrences(t: &TermVectorTerm) -> Option<Occurrences> {
    let freq = usize::try_from(t.freq).ok().filter(|&f| f > 0)?;
    let positions = t.positions.as_deref()?.get(..freq)?;
    let starts = t.start_offsets.as_deref()?.get(..freq)?;
    let ends = t.end_offsets.as_deref()?.get(..freq)?;
    let mut offsets = Vec::with_capacity(freq);
    let mut last = (-1, -1, -1);
    for ((&pos, &start), &end) in positions.iter().zip(starts).zip(ends) {
        let span = end.checked_sub(start)?.min(SHORT_MAX);
        // the stream refuses a negative position, offset or span, and
        // orders a term's occurrences by (position, start, span)
        let key = (pos, start, span);
        if pos < 0 || start < 0 || span < 0 || key < last {
            return None;
        }
        last = key;
        offsets.push((start, start.checked_add(span)?));
    }
    Some((positions.to_vec(), offsets))
}

impl MemoryIndex {
    /// The reader of `MemoryIndex::with_options(true, false)` given field
    /// `name` as `TokenStreamFromTermVector::new(vector, -1)` replays it, built
    /// off the vector. `None` when the vector is not the plain case this
    /// covers -- positions and offsets for every occurrence, none of them
    /// negative, each term's occurrences in the stream's order, and terms of
    /// valid UTF-8 shorter than `Short.MAX_VALUE` -- and the stream must be
    /// replayed instead.
    pub(crate) fn vector_reader(
        name: &str,
        vector: &TermVectorField,
    ) -> Result<Option<MemoryIndexReader>> {
        if !vector.has_positions || !vector.has_offsets || vector.terms.is_empty() {
            return Ok(None);
        }
        let mut terms = Vec::with_capacity(vector.terms.len());
        let mut all_positions = Vec::new();
        let mut max_freq = 0;
        for t in &vector.terms {
            let plain = t.term.len() < SHORT_MAX as usize && std::str::from_utf8(&t.term).is_ok();
            let Some((positions, offsets)) = plain.then(|| occurrences(t)).flatten() else {
                return Ok(None);
            };
            all_positions.extend_from_slice(&positions);
            max_freq = max_freq.max(t.freq);
            terms.push(TermPostings {
                term: t.term.clone(),
                docs: vec![(0, t.freq)],
                positions: vec![positions],
                offsets: vec![offsets],
                payload_bytes: Vec::new(),
                payload_lengths: Vec::new(),
            });
        }
        // the stream's statistics: one token per occurrence, each after the
        // first at its position an overlap (increment 0)
        all_positions.sort_unstable();
        let num_tokens = all_positions.len();
        let last_position = all_positions.last().copied().unwrap_or(0);
        all_positions.dedup();
        let num_overlap = num_tokens - all_positions.len();

        let mut index = MemoryIndex::with_options(true, false);
        let ft = index.default_field_type();
        index.get_info(name, &ft)?;
        let similarity = Arc::clone(&index.similarity);
        let info = index.info_mut(name);
        info.num_tokens = i32::try_from(num_tokens).unwrap_or(i32::MAX);
        info.num_overlap_tokens = i32::try_from(num_overlap).unwrap_or(i32::MAX);
        info.sum_total_term_freq = i64::try_from(num_tokens).unwrap_or(i64::MAX);
        info.max_term_frequency = max_freq;
        info.last_position = last_position;
        let norm =
            (!info.field_info.omit_norms).then(|| info.norm_of(terms.len(), similarity.as_ref()));
        let mut fi = info.field_info.clone();
        fi.store_term_vectors = false;
        fi.store_payloads = false;
        let doc = Document {
            fields: vec![FieldPostings {
                info: fi,
                has_payloads: false,
                terms,
                norm,
            }],
            ..Document::default()
        };
        MemoryIndexReader::new(doc, similarity).map(Some)
    }
}
