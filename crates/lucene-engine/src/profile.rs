//! The `profile` option of a search: the request's query run once more with a timer around
//! every clause of its scorer tree ([`lucene_search::profile_search`]), the timers written
//! for the host to lay out as OpenSearch's profile response.

use lucene_search::query::{BooleanQuery, Clause};
use lucene_search::similarities::Similarity;

use crate::clauses::map_search_error;
use crate::query_blob::{decode_request, query_uses_points, similarity_checked, JvmQuery};
use crate::reader_handle::JvmReaderHandle;
use crate::status::{set_last_error, Status};

/// How a profiled run collects ([`lucene_search::profile_search::Collect`]).
pub const COLLECT_MATCHES: u8 = 0;
/// The `n` best by score, counting under `track_total_hits`.
pub const COLLECT_TOP_SCORES: u8 = 1;

/// Runs the query of request blob `blob` over `h`'s segments with a timer around each clause.
///
/// The answer, little-endian: `clock: u64` (the timers' clock when the run began), `total: u64`
/// (the documents it counted), `nodes: u32`, then per clause and segment it was built for:
/// `segment: u32`, `path_len: u32`, the path as `u32`s, `kind_len: u8` and the kind's bytes, and
/// eight timers (build_scorer, next_doc, advance, match, score, shallow_advance,
/// compute_max_score, set_min_competitive_score), each `nanos: u64`, `count: u64`, `start: u64`.
///
/// # Errors
/// A malformed blob, or what opening the segments or scoring a clause reports.
pub fn profile_blob(
    h: &JvmReaderHandle,
    blob: &[u8],
    collect: u8,
    top_n: usize,
    count_limit: i64,
) -> Result<Vec<u8>, Status> {
    let (query, _min_score, sim) = decode_request(blob)?;
    let query = similarity_checked(sim.as_ref(), crate::join::resolve(h, query, sim.as_ref()))?;
    let mut opened = h.reader.open_segments().map_err(|e| {
        set_last_error(format!("opening segment postings: {e}"));
        Status::Decode
    })?;
    if query_uses_points(&query) {
        opened.open_points().map_err(|e| {
            set_last_error(format!("opening segment points: {e}"));
            Status::Decode
        })?;
    }
    let segments: Vec<_> = opened
        .as_open_segments()
        .into_iter()
        .zip(&h.live_docs)
        .map(|(mut s, live)| {
            s.live_docs = live.as_ref();
            s.similarity = sim.as_ref().map(|p| p as &dyn Similarity);
            s
        })
        .collect();
    let (root, as_boolean) = match &query {
        JvmQuery::Term(t) => (
            Clause::Term(t.clone()),
            BooleanQuery {
                must: vec![Clause::Term(t.clone())],
                ..Default::default()
            },
        ),
        JvmQuery::Boolean(b) => (Clause::Boolean(Box::new(b.clone())), b.clone()),
    };
    let fields: Vec<String> = crate::clauses::clause_field_names(&as_boolean)
        .into_iter()
        .map(str::to_string)
        .collect();
    let owned = h
        .reader
        .field_norms_by_field(&fields)
        .map_err(|error| map_search_error(error.into()))?;
    let norms: Vec<Option<&std::collections::HashMap<String, _>>> =
        owned.iter().map(|m| (!m.is_empty()).then_some(m)).collect();
    let collecting = if collect == COLLECT_TOP_SCORES {
        lucene_search::profile_search::Collect::TopScores {
            top_n,
            total_hits_threshold: u64::try_from(count_limit).unwrap_or(0),
        }
    } else {
        lucene_search::profile_search::Collect::Matches
    };
    let clock = lucene_search::profile_search::clock();
    let run = similarity_checked(
        sim.as_ref(),
        lucene_search::profile_search::profile_query(&segments, &root, &norms, collecting)
            .map_err(map_search_error),
    )?;
    let mut out = Vec::new();
    out.extend_from_slice(&clock.to_le_bytes());
    out.extend_from_slice(&run.total.to_le_bytes());
    let count = u32::try_from(run.nodes.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&count.to_le_bytes());
    for node in &run.nodes {
        out.extend_from_slice(
            &u32::try_from(node.segment)
                .unwrap_or(u32::MAX)
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &u32::try_from(node.path.len())
                .unwrap_or(u32::MAX)
                .to_le_bytes(),
        );
        for step in &node.path {
            out.extend_from_slice(&step.to_le_bytes());
        }
        let kind = node.kind.as_bytes();
        out.push(u8::try_from(kind.len()).unwrap_or(u8::MAX));
        out.extend_from_slice(&kind[..kind.len().min(255)]);
        for timer in &node.timers {
            out.extend_from_slice(&timer.nanos().to_le_bytes());
            out.extend_from_slice(&timer.count().to_le_bytes());
            out.extend_from_slice(&timer.earliest_start().to_le_bytes());
        }
    }
    Ok(out)
}
