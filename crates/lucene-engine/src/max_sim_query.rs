//! The plugin's `max_sim` query over a `multi_vector` field, as node `44` of
//! the query tree ([`crate::queries`]).
//!
//! The sub-query's documents are scored by the `MaxSim` of the query's token
//! vectors against the field's stored multi-vector
//! ([`lucene_tensor::MaxSimValuesSource`], any of its encodings), through the
//! ported Lucene [`FunctionScoreQuery`]: a document without a value, or with a
//! negative `MaxSim`, scores 0. The plugin's Java `MaxSimQuery` scores the
//! same way.
//!
//! The node, after its kind byte:
//!
//! | field | layout |
//! |---|---|
//! | sub-query | a node |
//! | `field` | string |
//! | `dim` | `i32`, at least 1 |
//! | tokens | `i32` count, at least 1, then `count * dim` `f32` |

use std::sync::Arc;

use lucene_search::extended_query::ExtendedQuery;
use lucene_search::function::FunctionScoreQuery;
use lucene_search::query::Clause;
use lucene_tensor::{MaxSimQuery, MaxSimValuesSource};

use crate::nodes::invalid;
use crate::query_blob::Cursor;
use crate::status::Status;

pub const NODE_MAX_SIM: u8 = 44;

pub fn decode(
    c: &mut Cursor<'_>,
    depth: usize,
    nodes: &mut usize,
) -> std::result::Result<Clause, Status> {
    let sub = crate::query_blob::decode_node(c, depth + 1, nodes)?;
    let field = std::str::from_utf8(c.bytes()?)
        .map_err(|_| Status::InvalidUtf8)?
        .to_string();
    let dim = c.length()?;
    let count = c.length()?;
    if dim == 0 || count == 0 {
        return Err(invalid("query tree: max_sim without tokens"));
    }
    let floats = dim
        .checked_mul(count)
        .filter(|&n| n <= c.remaining() / 4)
        .ok_or_else(|| invalid("query tree: max_sim tokens past the blob"))?;
    let mut tokens = Vec::with_capacity(count); // alloc-ok: bounded by the blob
    let mut token = Vec::with_capacity(dim); // alloc-ok: bounded by the blob
    for _ in 0..floats {
        token.push(f32::from_bits(c.i32()? as u32));
        if token.len() == dim {
            tokens.push(std::mem::take(&mut token));
        }
    }
    let query = MaxSimQuery::new(&tokens).map_err(|e| invalid(format!("query tree: {e}")))?;
    let source = MaxSimValuesSource::new(&field, query);
    Ok(Clause::Extended(Box::new(ExtendedQuery::FunctionScore(
        FunctionScoreQuery::new(sub, Arc::new(source)),
    ))))
}
