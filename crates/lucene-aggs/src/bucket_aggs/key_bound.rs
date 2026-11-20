//! The bound on a nested bucket aggregation's key tables: each is created
//! with a cap of the request's bucket limit, and a pass that refused a key for
//! it fails with `TooManyBuckets`.

use super::*;

/// `nodes`' fresh states, each nested keyed aggregation's key table bound to
/// `limit` keys (a top-level one holds at most the field's values, which
/// the caller's reservation covers).
pub(super) fn new_states(nodes: &[AggNode], limit: usize) -> Vec<State> {
    let mut states: Vec<State> = nodes.iter().map(State::new).collect();
    for state in &mut states {
        bound_nested(state, limit);
    }
    states
}

/// Binds the key tables of `state`'s sub-aggregations, at every depth, to
/// `limit` keys.
fn bound_nested(state: &mut State, limit: usize) {
    let (State::Keyed { subs, .. } | State::Fixed { subs, .. }) = state else {
        return;
    };
    for sub in subs {
        if let State::Keyed { ords, .. } = sub {
            ords.cap = if limit == usize::MAX { 0 } else { limit.max(1) };
        }
        bound_nested(sub, limit);
    }
}

/// Whether a nested key table refused a key for its bound.
fn refused_for_bound(states: &[State]) -> bool {
    states.iter().any(|state| match state {
        State::Keyed { ords, subs, .. } => ords.over || refused_for_bound(subs),
        State::Fixed { subs, .. } => refused_for_bound(subs),
        _ => false,
    })
}

/// Fails with `TooManyBuckets` when `states` hold more buckets than `limit`
/// (counted from below by [`buckets_made`]) or a nested key table refused a
/// key for its cap.
pub(super) fn check_limit(nodes: &[AggNode], states: &[State], limit: usize) -> Result<()> {
    let made = buckets_made(nodes, states);
    if made > limit || refused_for_bound(states) {
        return Err(lucene_search::Error::TooManyBuckets(
            made.max(limit.saturating_add(1)),
            limit,
        ));
    }
    Ok(())
}
