//! From a query to what it reads from the writer's buffers, and from what
//! the buffers gave back to segments the query runs on.

use std::collections::{BTreeMap, BTreeSet};

use lucene_index::concurrent_writer::{
    LiveBufferField, LiveFieldRequest, LiveRequest, LiveSnapshot,
};

use super::{LiveField, LiveSegment};
use crate::query::Clause;
use crate::Result;

/// The terms `clause` reads, by field, with whether it reads their
/// positions; `None` when it has a clause the buffers cannot answer yet
/// (multi-term, points, doc values, spans and other extended queries).
pub fn live_request(clause: &Clause) -> Option<LiveRequest> {
    let mut fields: BTreeMap<&str, (BTreeSet<&[u8]>, bool)> = BTreeMap::new();
    collect(clause, &mut fields)?;
    Some(LiveRequest {
        fields: fields
            .into_iter()
            .map(|(field, (terms, positions))| LiveFieldRequest {
                field: field.to_string(),
                terms: terms.into_iter().map(<[u8]>::to_vec).collect(),
                positions,
            })
            .collect(),
    })
}

type Fields<'a> = BTreeMap<&'a str, (BTreeSet<&'a [u8]>, bool)>;

fn add<'a>(fields: &mut Fields<'a>, field: &'a str, term: &'a [u8], positions: bool) {
    let e = fields.entry(field).or_default();
    e.0.insert(term);
    e.1 |= positions;
}

fn collect<'a>(clause: &'a Clause, fields: &mut Fields<'a>) -> Option<()> {
    match clause {
        Clause::Term(t) => add(fields, &t.field, &t.term, false),
        Clause::TermInSet(q) => {
            for t in &q.terms {
                add(fields, &q.field, t, false);
            }
        }
        Clause::Phrase(p) => {
            for t in &p.terms {
                add(fields, &p.field, t, true);
            }
        }
        Clause::MultiPhrase(p) => {
            for t in p.term_arrays.iter().flatten() {
                add(fields, &p.field, t, true);
            }
        }
        Clause::Boolean(b) => {
            for c in b
                .must
                .iter()
                .chain(&b.filter)
                .chain(&b.should)
                .chain(&b.must_not)
            {
                collect(c, fields)?;
            }
        }
        Clause::DisjunctionMax(d) => {
            for c in &d.disjuncts {
                collect(c, fields)?;
            }
        }
        Clause::ConstantScore(c) => collect(&c.inner, fields)?,
        Clause::Boost(b) => collect(&b.inner, fields)?,
        Clause::MatchAllDocs(_) | Clause::MatchNoDocs(_) => {}
        _ => return None,
    }
    Some(())
}

impl From<LiveBufferField> for LiveField {
    fn from(f: LiveBufferField) -> Self {
        LiveField {
            info: f.info,
            terms: f.terms,
            norms: f.norms,
            num_terms: f.num_terms,
            sum_total_term_freq: f.sum_total_term_freq,
            sum_doc_freq: f.sum_doc_freq,
            doc_count: f.doc_count,
        }
    }
}

/// One [`LiveSegment`] per buffer of `snapshot`, in its order.
pub fn live_segments(snapshot: LiveSnapshot) -> Result<Vec<LiveSegment>> {
    snapshot
        .buffers
        .into_iter()
        .map(|b| {
            let fields: Vec<LiveField> = b.fields.into_iter().map(LiveField::from).collect();
            LiveSegment::build(&fields, b.max_doc, &b.deleted)
        })
        .collect()
}
