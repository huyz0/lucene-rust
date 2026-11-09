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
mod tests {
    // Test arithmetic on small constants; see docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

    use lucene_codecs::field_infos::IndexOptions;
    use lucene_codecs::postings_writer::TermSource;

    use super::super::super::explicit::{InvertedField, InvertedTerm};
    use super::super::{Checkpoint, SEALED_DOCS};
    use super::*;

    const FULL: IndexOptions = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;

    fn term(t: &str, positions: &[i32], payload: &[u8]) -> InvertedTerm {
        InvertedTerm {
            term: t.as_bytes().to_vec(),
            freq: i32::try_from(positions.len()).unwrap(),
            positions: positions.to_vec(),
            offsets: positions.iter().map(|&p| (p * 4, p * 4 + 3)).collect(),
            payloads: positions.iter().map(|_| payload.to_vec()).collect(),
        }
    }

    fn add(p: &mut PendingPostings, doc: i32, terms: Vec<InvertedTerm>) {
        let field = InvertedField {
            field_number: 0,
            terms: terms.into(),
            norm: None,
        };
        p.add_field(doc, &field, FULL).unwrap();
    }

    /// Documents `0..n`: "a" in each (doc % 3 + 1 times), "b" in the even
    /// ones, a payload on "b" in every fourth.
    fn buffer(n: i32) -> PendingPostings {
        let mut p = PendingPostings::default();
        p.seal_blocks();
        for doc in 0..n {
            let a: Vec<i32> = (0..=doc % 3).map(|k| k * 2).collect();
            let mut terms = vec![term("a", &a, b"")];
            if doc % 2 == 0 {
                let payload: &[u8] = if doc % 4 == 0 { b"pp" } else { b"" };
                terms.push(term("b", &[7], payload));
            }
            add(&mut p, doc, terms);
        }
        p
    }

    /// A buffered term reads back as the flush lays it out, with and
    /// without its occurrences; a term the buffer lacks is `None`.
    #[test]
    fn a_buffered_term_reads_as_the_flush_lays_it_out() {
        let p = buffer(300);
        let flushed = p.flushed(None).unwrap();
        let lists = flushed.term_lists();
        let runs = p.live_field(0).unwrap();
        for t in ["a", "b"] {
            let want = lists[0]
                .2
                .iter()
                .find(|v| v.term() == t.as_bytes())
                .unwrap();
            let full = runs.live_term(t.as_bytes(), true, 0).unwrap().unwrap();
            assert_eq!(full.docs, want.docs(), "{t}");
            for (i, positions) in full.positions.iter().enumerate() {
                assert_eq!(positions, want.positions(i), "{t} doc {i}");
                assert_eq!(full.offsets[i], want.offsets(i), "{t} doc {i}");
            }
            assert_eq!(full.payload_lengths, want.payload_lengths(), "{t}");
            assert_eq!(full.payload_bytes, want.payload_bytes(), "{t}");
            let docs = runs.live_term(t.as_bytes(), false, 0).unwrap().unwrap();
            assert_eq!(docs.docs, want.docs(), "{t}");
            assert!(docs.positions.is_empty() && docs.payload_bytes.is_empty());
        }
        assert!(runs.live_term(b"c", true, 0).unwrap().is_none());
        assert!(runs.has_payloads());
    }

    /// The statistics count every buffered document, and a rollback takes
    /// back what the undone documents added.
    #[test]
    fn statistics_follow_adds_and_rollbacks() {
        let mut p = buffer(300);
        let want = LiveFieldStats {
            num_terms: 2,
            // "a" in all 300, "b" in 150.
            sum_doc_freq: 450,
            // "a" 1+2+3 per three documents, "b" once each.
            sum_total_term_freq: 600 + 150,
            doc_count: 300,
        };
        assert_eq!(p.live_field(0).unwrap().live_stats(), want);
        let cp = p.checkpoint(300);
        add(
            &mut p,
            300,
            vec![term("a", &[0], b""), term("c", &[1, 2], b"")],
        );
        let grown = p.live_field(0).unwrap().live_stats();
        assert_eq!(
            (
                grown.num_terms,
                grown.sum_doc_freq,
                grown.sum_total_term_freq,
                grown.doc_count
            ),
            (3, 452, 753, 301)
        );
        p.rollback(cp);
        assert_eq!(p.live_field(0).unwrap().live_stats(), want);
    }

    /// A term with sealed blocks reads the same documents from any
    /// document on, and after a rollback cut its last block.
    #[test]
    fn sealed_blocks_read_as_the_streams() {
        let n = 3 * SEALED_DOCS as i32 + 17;
        let mut p = buffer(n);
        let flushed = p.flushed(None).unwrap();
        let lists = flushed.term_lists();
        let want = |t: &str| {
            lists[0]
                .2
                .iter()
                .find(|v| v.term() == t.as_bytes())
                .unwrap()
                .docs()
                .to_vec()
        };
        let (a, b) = (want("a"), want("b"));
        drop(flushed);
        let runs = p.live_field(0).unwrap();
        let id = runs.terms.find(b"a").unwrap();
        assert_eq!(runs.sealed_blocks(id).len(), 3);
        for from in [0, 1, 255, 256, 257, 600, n - 1, n] {
            let mut got = Vec::new();
            runs.live_docs_of(b"a", from, &mut got);
            let expect: Vec<_> = a.iter().copied().filter(|d| d.0 >= from).collect();
            assert_eq!(got, expect, "a from {from}");
            got.clear();
            runs.live_docs_of(b"b", from, &mut got);
            let expect: Vec<_> = b.iter().copied().filter(|d| d.0 >= from).collect();
            assert_eq!(got, expect, "b from {from}");
        }
        // Cut back into the third block: it goes, the first two stay.
        let cut = 2 * SEALED_DOCS as i32 + 5;
        let cp = p.checkpoint(n);
        p.rollback(Checkpoint { doc: cut, ..cp });
        let runs = p.live_field(0).unwrap();
        assert_eq!(runs.sealed_blocks(id).len(), 2);
        let mut got = Vec::new();
        runs.live_docs_of(b"a", 0, &mut got);
        let expect: Vec<_> = a.iter().copied().filter(|d| d.0 < cut).collect();
        assert_eq!(got, expect);
    }
}
