//! I/O hints for a query over files read by range (`IndexInput.prefetch`,
//! as Lucene's `TermQuery` and `PhraseQuery` weights issue them): every
//! term's terms-dictionary block is announced before any is sought, then
//! every term's first postings bytes before any is read, so a cold query
//! waits for two round trips however many terms and segments it has. Over
//! files in memory or mapped nothing is done beyond a few map lookups.

use lucene_store::{is_prefetchable, is_resident, prefetch};

use crate::multi_segment::OpenSegment;
use crate::query::{BooleanQuery, Clause};
use crate::Result;

/// A term the query reads postings for, and whether positions too.
struct QueryTerm<'q> {
    field: &'q str,
    term: &'q [u8],
    positions: bool,
}

fn collect<'q>(query: &'q BooleanQuery, out: &mut Vec<QueryTerm<'q>>) {
    let clauses = query
        .must
        .iter()
        .chain(&query.filter)
        .chain(&query.should)
        .chain(&query.must_not);
    for clause in clauses {
        match clause {
            Clause::Term(t) => out.push(QueryTerm {
                field: &t.field,
                term: &t.term,
                positions: false,
            }),
            Clause::Phrase(p) => out.extend(p.terms.iter().map(|term| QueryTerm {
                field: &p.field,
                term,
                positions: true,
            })),
            Clause::Boolean(b) => collect(b, out),
            _ => {}
        }
    }
}

/// Prefetches what `query`'s term and phrase clauses (nested booleans
/// included) read first in each of `segments`, where those files are read by
/// range: the terms-dictionary blocks, then the start of each posting list
/// (and position list for a phrase).
///
/// # Errors
/// A page that could not be fetched, or a corrupt terms index.
pub fn prefetch_query(segments: &[OpenSegment<'_>], query: &BooleanQuery) -> Result<()> {
    let mut terms = Vec::new();
    collect(query, &mut terms);
    if terms.is_empty() {
        return Ok(());
    }

    // Which (segment, term) pairs to look up postings for: those whose
    // terms-dictionary block was not resident. A term whose block is in
    // memory was read recently, and so most likely were its postings;
    // looking its postings up here would cost a second seek per term for
    // nothing. Where the terms file is not read by page there is no such
    // signal, and every term is looked up.
    let mut cold = vec![true; segments.len().saturating_mul(terms.len())];
    let mut ranges: Vec<&[u8]> = Vec::new();
    for (s, segment) in segments.iter().enumerate() {
        for (i, t) in terms.iter().enumerate() {
            if let Some(field) = segment.fields.field(t.field) {
                if is_prefetchable(field.terms_file()) {
                    match field.term_block(t.term)? {
                        Some(block) if is_resident(block) => {
                            cold[s.saturating_mul(terms.len()).saturating_add(i)] = false;
                        }
                        Some(block) => ranges.push(block),
                        None => cold[s.saturating_mul(terms.len()).saturating_add(i)] = false,
                    }
                }
            }
        }
    }
    prefetch(&ranges)?;

    ranges.clear();
    for (s, segment) in segments.iter().enumerate() {
        let doc_in = segment.doc_in.filter(|d| is_prefetchable(d.bytes()));
        let pos_in = segment.pos_in.filter(|p| is_prefetchable(p.bytes()));
        if doc_in.is_none() && pos_in.is_none() {
            continue;
        }
        for (i, t) in terms.iter().enumerate() {
            if !cold[s.saturating_mul(terms.len()).saturating_add(i)] {
                continue;
            }
            let Some(field) = segment.fields.field(t.field) else {
                continue;
            };
            let Some(meta) = field.term_metadata(t.term)? else {
                continue;
            };
            let first = |bytes: &'_ [u8], fp: u64| -> Option<std::ops::Range<usize>> {
                let at = usize::try_from(fp).ok()?;
                (at < bytes.len()).then(|| at..at.saturating_add(1))
            };
            if meta.singleton_doc_id == -1 {
                if let Some(doc_in) = doc_in {
                    let bytes = doc_in.bytes();
                    if let Some(r) = first(bytes, meta.doc_start_fp) {
                        ranges.push(&bytes[r]);
                    }
                }
            }
            if t.positions {
                if let Some(pos_in) = pos_in {
                    let bytes = pos_in.bytes();
                    if let Some(r) = first(bytes, meta.pos_start_fp) {
                        ranges.push(&bytes[r]);
                    }
                }
            }
        }
    }
    prefetch(&ranges)?;
    Ok(())
}
