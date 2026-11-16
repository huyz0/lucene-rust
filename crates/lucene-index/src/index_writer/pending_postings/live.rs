//! Reading the buffered postings for a search, without a flush: one term's
//! documents (and, when asked, its occurrences) decoded from its streams,
//! and the field statistics BM25 needs.

use lucene_codecs::postings_writer::TermPostings;

use super::{Columns, Cursor, FieldRuns, PendingPostings, Result};

/// Terms with fewer documents are read from their start each search.
const LIVE_CURSOR_DOCS: u32 = if cfg!(test) { 2 } else { 1024 };

/// How many reads' ends a term keeps at most.
const LIVE_CURSORS: usize = 16;

/// A buffered field's statistics, as a flushed segment would report them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct LiveFieldStats {
    pub(crate) num_terms: i64,
    pub(crate) sum_doc_freq: i64,
    pub(crate) sum_total_term_freq: i64,
    pub(crate) doc_count: i32,
}

thread_local! {
    static DECODER: std::cell::RefCell<lucene_codecs::postings::DocBlockDecoder> =
        std::cell::RefCell::default();
}

impl FieldRuns {
    /// The field's statistics over every buffered document.
    pub(crate) fn live_stats(&self) -> LiveFieldStats {
        LiveFieldStats {
            num_terms: i64::try_from(self.terms.count()).unwrap_or(i64::MAX),
            sum_doc_freq: self.sum_doc_freq,
            sum_total_term_freq: self.sum_total_term_freq,
            doc_count: self.doc_count,
        }
    }

    /// The buffered terms `matches` takes, in no order.
    pub(crate) fn live_matching(&self, matches: impl Fn(&[u8]) -> bool) -> Vec<&[u8]> {
        let n = u32::try_from(self.terms.count()).unwrap_or(u32::MAX);
        (0..n)
            .map(|id| self.terms.term(id))
            .filter(|t| matches(t))
            .collect()
    }

    /// Whether some occurrence of the field carried a payload.
    pub(crate) fn has_payloads(&self) -> bool {
        self.payloads
    }

    /// `term`'s `(doc, freq)` pairs from document `from` on, doc ascending,
    /// onto `out`; nothing when the buffer does not hold the term. A term
    /// of [`LIVE_CURSOR_DOCS`] documents or more is read from where an
    /// earlier read ended before `from`, when one did.
    pub(crate) fn live_docs_of(&self, term: &[u8], from: i32, out: &mut Vec<(i32, i32)>) {
        let Some(id) = self.terms.find(term) else {
            return;
        };
        let run = &self.runs[id as usize];
        let blocks = self.sealed_blocks(id);
        if let Some(last) = blocks.last() {
            // The sealed blocks from the first reaching `from`, then the
            // documents written after them, then the newest. A block that
            // does not decode leaves the term to its stream.
            let at = out.len();
            let first = blocks.partition_point(|b| b.last_doc < from);
            let decoded = DECODER.with(|d| {
                let mut d = d.borrow_mut();
                blocks
                    .iter()
                    .enumerate()
                    .skip(first)
                    .try_for_each(|(i, b)| {
                        let prev = i
                            .checked_sub(1)
                            .and_then(|p| blocks.get(p))
                            .map_or(-1, |p| p.last_doc);
                        d.decode(&b.bytes, prev, self.freqs, from, out)
                    })
            });
            if decoded.is_ok() {
                let mut r = self.pool.resume(last.end, run.docs);
                let mut doc = last.last_doc;
                while !r.eof() {
                    let freq = r.next_doc(&mut doc);
                    if doc >= from {
                        out.push((doc, if self.freqs { freq } else { 1 }));
                    }
                }
                if run.last_doc >= from {
                    out.push((run.last_doc, if self.freqs { run.freq } else { 1 }));
                }
                return;
            }
            debug_assert!(false, "a sealed block decodes");
            out.truncate(at);
        }
        let [docs_start, _] = self.starts[id as usize];
        let cursored = run.doc_count >= LIVE_CURSOR_DOCS;
        let lock = || {
            self.live_cursors
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        };
        let start = (cursored && from > 0)
            .then(|| {
                lock().get(&id).and_then(|cs| {
                    cs.iter()
                        .filter(|c| c.doc < from)
                        .max_by_key(|c| c.doc)
                        .copied()
                })
            })
            .flatten();
        let mut r = match start {
            Some(c) => self.pool.resume(c.docs, run.docs),
            None => self.pool.reader(docs_start, run.docs),
        };
        // The document the next code is a delta from: every document
        // before where the read starts is at most it.
        let mut doc = start.map_or(0, |c| c.doc);
        while !r.eof() {
            let freq = r.next_doc(&mut doc);
            if doc >= from {
                out.push((doc, if self.freqs { freq } else { 1 }));
            }
        }
        if cursored {
            let end = Cursor {
                docs: r.pos(),
                prox: None,
                doc,
            };
            let mut cursors = lock();
            let cs = cursors.entry(id).or_default();
            if !cs.iter().any(|c| c.doc == doc) {
                if cs.len() >= LIVE_CURSORS {
                    cs.remove(0);
                }
                cs.push(end);
            }
        }
        if run.doc_count > 0 && run.last_doc >= from {
            out.push((run.last_doc, if self.freqs { run.freq } else { 1 }));
        }
    }

    /// `term`'s postings as a flush would write them, documents in buffered
    /// order; its occurrences only when `positions` is set and the field
    /// indexes them. `None` when the buffer does not hold the term.
    pub(crate) fn live_term(
        &self,
        term: &[u8],
        positions: bool,
        from: i32,
    ) -> Result<Option<TermPostings>> {
        let Some(id) = self.terms.find(term) else {
            return Ok(None);
        };
        let mut out = TermPostings {
            term: term.to_vec(),
            ..TermPostings::default()
        };
        if !(positions && self.positions) {
            self.live_docs_of(term, from, &mut out.docs);
            return Ok(Some(out));
        }
        let mut cols = Columns::default();
        self.decode(id, &mut cols)?;
        let mut from = 0usize;
        out.positions.reserve(cols.ends.len());
        for &end in &cols.ends {
            let end = end as usize;
            out.positions.push(cols.positions[from..end].to_vec());
            if self.offsets {
                out.offsets.push(cols.offsets[from..end].to_vec());
            }
            from = end;
        }
        out.docs = cols.docs;
        out.payload_bytes = cols.payload_bytes;
        out.payload_lengths = cols.payload_lengths;
        Ok(Some(out))
    }
}

impl PendingPostings {
    /// Field `number`'s buffered postings, if any document gave it a term.
    pub(crate) fn live_field(&self, number: i32) -> Option<&FieldRuns> {
        self.fields.get(&number)
    }

    /// Every field's [`FieldRuns::live_stats`], by field number.
    pub(crate) fn live_stats_all(&self) -> std::collections::HashMap<i32, LiveFieldStats> {
        self.fields
            .iter()
            .map(|(&n, f)| (n, f.live_stats()))
            .collect()
    }
}

#[cfg(test)]
mod tests;
