//! A whole-reader KNN search, run before the rest of a request: the
//! `lucene` engine's `knn` query is `KnnFloatVectorQuery`, whose
//! `rewrite(IndexSearcher)` finds the reader's top `k` and becomes a
//! `DocAndScoreQuery`. The plugin's query does that rewrite through
//! [`ffi_jvm_reader_knn`] and sends the hits back in the request's tree as a
//! doc-and-score node (`NODE_DOC_AND_SCORE`), so the request itself runs
//! native like any other.
//!
//! Request blob: `field` (`i32 len`, UTF-8), the target
//! ([`crate::jvm_queries::decode_knn_target`]), `k: i32`, `candidates: i32`
//! (the beam the search keeps before cutting to `k`, `max(k, ef_search)`),
//! `has_filter: u8` and the filter (a query-tree node) when `1`, then
//! `nested: u8`: when `1`, the field is nested and the parent filter (a
//! query-tree node, whose bit sets are cached as a nested query's are) and
//! `expand: u8` (`expand_nested_docs`) follow, and last `nprobes: i32`: the
//! clusters probed in each segment whose field has an IVF index
//! (`LuceneRustIvfVectorsFormat`; `method_parameters.nprobes`), `0` for the
//! depth each segment calibrated.
//!
//! A nested search keeps the best child of each of the best `k` parents; an
//! expanded one returns every scored child of those parents, which can be
//! more than `k`: when they do not fit the buffers, the call is
//! [`FfiStatus::BufferTooSmall`] with the count it needs in `*out_count`.

use std::sync::Arc;

use lucene_search::join::BitSetProducer;
use lucene_search::knn_search::{knn_top_hits_nested, knn_top_hits_with};
use lucene_search::multi_segment::OpenSegment;

use crate::clauses::map_search_error;
use crate::error::{guard, set_last_error, FfiStatus};
use crate::jvm_reader::{lookup, Cursor};
use crate::raw::bytes_from_raw;

/// A decoded request.
struct Request {
    field: String,
    target: lucene_search::knn_search::KnnTarget,
    k: usize,
    candidates: usize,
    filter: Option<lucene_search::query::Clause>,
    /// A nested field's parent filter and `expand_nested_docs`.
    nested: Option<(Arc<dyn BitSetProducer>, bool)>,
    nprobes: usize,
}

fn invalid(message: &str) -> FfiStatus {
    set_last_error(format!("KNN request: {message}"));
    FfiStatus::InvalidArgument
}

fn decode(blob: &[u8]) -> Result<Request, FfiStatus> {
    let mut c = Cursor::new(blob);
    let field = std::str::from_utf8(c.bytes()?)
        .map_err(|_| FfiStatus::InvalidUtf8)?
        .to_string();
    let target = crate::jvm_queries::decode_knn_target(&mut c)?;
    let positive = |v: i32, what: &str| {
        usize::try_from(v)
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| invalid(what))
    };
    let k = positive(c.i32()?, "k must be at least 1")?;
    let candidates = positive(c.i32()?, "candidates must be at least 1")?;
    let mut nodes = 0usize;
    let filter = match c.u8()? {
        0 => None,
        1 => Some(crate::jvm_reader::decode_node(&mut c, 0, &mut nodes)?),
        _ => return Err(invalid("filter flag (expected 0..=1)")),
    };
    let nested = match c.u8()? {
        0 => None,
        1 => {
            let parents = crate::jvm_nodes::decode_parents(&mut c, 0, &mut nodes)?;
            let expand = match c.u8()? {
                0 => false,
                1 => true,
                _ => return Err(invalid("expand flag (expected 0..=1)")),
            };
            Some((parents, expand))
        }
        _ => return Err(invalid("nested flag (expected 0..=1)")),
    };
    let nprobes = usize::try_from(c.i32()?).map_err(|_| invalid("nprobes must not be negative"))?;
    if c.remaining() != 0 {
        return Err(invalid("trailing bytes"));
    }
    Ok(Request {
        field,
        target,
        k,
        candidates,
        filter,
        nested,
        nprobes,
    })
}

/// The reader's top `k` for the request in `request` ([module doc](self)):
/// global doc ids into `out_docs`, scores into `out_scores`, best first,
/// `*out_count` of them.
///
/// # Safety
/// `request` valid for `request_len` bytes; `out_docs`/`out_scores` for
/// `buf_len` elements each, with `buf_len >= k`; `out_count` for one write.
/// An expanded nested search may need more than `k` (see the
/// [module doc](self)).
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_reader_knn(
    handle: u64,
    request: *const u8,
    request_len: usize,
    out_docs: *mut i32,
    out_scores: *mut f32,
    buf_len: usize,
    out_count: *mut usize,
) -> i32 {
    guard(|| {
        if out_docs.is_null() || out_scores.is_null() || out_count.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(request, request_len)? };
        let Request {
            field,
            target,
            k,
            candidates,
            filter,
            nested,
            nprobes,
        } = decode(blob)?;
        if buf_len < k {
            return Err(FfiStatus::BufferTooSmall);
        }
        let h = lookup(
            handle,
            "ffi_jvm_reader_knn: unknown or already-closed handle",
        )?;
        let mut opened = h.reader.open_segments().map_err(|e| {
            set_last_error(format!("opening segment postings: {e}"));
            FfiStatus::Decode
        })?;
        if let Some(f) = &filter {
            let probe = lucene_search::query::BooleanQuery::new().with_filter([f.clone()]);
            if crate::jvm_reader::boolean_uses_points(&probe) {
                opened.open_points().map_err(|e| {
                    set_last_error(format!("opening segment points: {e}"));
                    FfiStatus::Decode
                })?;
            }
        }
        let segments: Vec<OpenSegment<'_>> = opened
            .as_open_segments()
            .into_iter()
            .zip(&h.live_docs)
            .map(|(mut s, live)| {
                s.live_docs = live.as_ref();
                s
            })
            .collect();
        let hits = match &nested {
            None => knn_top_hits_with(
                &segments,
                &field,
                &target,
                k,
                candidates,
                filter.as_ref(),
                nprobes,
            ),
            Some((parents, expand)) => knn_top_hits_nested(
                &segments,
                &field,
                &target,
                k,
                candidates,
                filter.as_ref(),
                &**parents,
                *expand,
            ),
        }
        .map_err(map_search_error)?;
        if hits.len() > buf_len {
            // SAFETY: caller contract.
            unsafe { *out_count = hits.len() };
            set_last_error(format!(
                "KNN request: {} hits do not fit {buf_len}",
                hits.len()
            ));
            return Err(FfiStatus::BufferTooSmall);
        }
        // SAFETY: caller contract; `hits.len() <= buf_len`.
        unsafe {
            for (i, hit) in hits.iter().enumerate() {
                *out_docs.add(i) = hit.doc_id;
                *out_scores.add(i) = hit.score;
            }
            *out_count = hits.len();
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(k: i32, candidates: i32, filter: Option<&[u8]>) -> Vec<u8> {
        let mut b = request_without_nested(k, candidates, filter);
        b.push(0);
        b.extend_from_slice(&0i32.to_le_bytes());
        b
    }

    fn request_without_nested(k: i32, candidates: i32, filter: Option<&[u8]>) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&1i32.to_le_bytes());
        b.push(b'v');
        b.push(crate::jvm_queries::KNN_TARGET_FLOAT);
        b.extend_from_slice(&2i32.to_le_bytes());
        b.extend_from_slice(&1.0f32.to_bits().to_le_bytes());
        b.extend_from_slice(&0.5f32.to_bits().to_le_bytes());
        b.extend_from_slice(&k.to_le_bytes());
        b.extend_from_slice(&candidates.to_le_bytes());
        match filter {
            None => b.push(0),
            Some(node) => {
                b.push(1);
                b.extend_from_slice(node);
            }
        }
        b
    }

    #[test]
    fn decodes_a_request_and_refuses_bad_ones() {
        let r = decode(&request(3, 10, None)).unwrap();
        assert_eq!(r.field, "v");
        assert_eq!(
            r.target,
            lucene_search::knn_search::KnnTarget::Float(vec![1.0, 0.5])
        );
        assert_eq!((r.k, r.candidates), (3, 10));
        assert!(r.filter.is_none() && r.nested.is_none());
        assert_eq!(r.nprobes, 0);
        let mut probed = request(3, 10, None);
        let n = probed.len();
        probed[n - 4..].copy_from_slice(&32i32.to_le_bytes());
        assert_eq!(decode(&probed).unwrap().nprobes, 32);
        probed[n - 4..].copy_from_slice(&(-1i32).to_le_bytes());
        assert!(decode(&probed).is_err());
        assert!(decode(&request(0, 10, None)).is_err());
        assert!(decode(&request(3, 0, None)).is_err());
        let mut trailing = request(3, 10, None);
        trailing.push(0);
        assert!(decode(&trailing).is_err());
        assert!(decode(&request(3, 10, None)[..9]).is_err());
        assert!(decode(&request_without_nested(3, 10, None)).is_err());
    }

    #[test]
    fn decodes_a_nested_request() {
        let mut b = request_without_nested(3, 10, None);
        b.push(1);
        b.push(5); // the parent filter: match_all
        b.push(1);
        b.extend_from_slice(&0i32.to_le_bytes());
        let r = decode(&b).unwrap();
        let (parents, expand) = r.nested.unwrap();
        assert!(expand);
        assert!(parents.key().starts_with("parents("));
        b.truncate(b.len() - 5);
        b.push(2);
        b.extend_from_slice(&0i32.to_le_bytes());
        assert!(decode(&b).is_err());
    }
}
