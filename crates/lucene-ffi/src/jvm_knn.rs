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
//! `has_filter: u8` and the filter (a query-tree node) when `1`.

use lucene_search::knn_search::knn_top_hits;
use lucene_search::multi_segment::OpenSegment;

use crate::clauses::map_search_error;
use crate::error::{guard, set_last_error, FfiStatus};
use crate::jvm_reader::{lookup, Cursor};
use crate::raw::bytes_from_raw;

/// A decoded request: field, target, `k`, candidates, filter.
type Request = (
    String,
    lucene_search::knn_search::KnnTarget,
    usize,
    usize,
    Option<lucene_search::query::Clause>,
);

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
    if c.remaining() != 0 {
        return Err(invalid("trailing bytes"));
    }
    Ok((field, target, k, candidates, filter))
}

/// The reader's top `k` for the request in `request` ([module doc](self)):
/// global doc ids into `out_docs`, scores into `out_scores`, best first,
/// `*out_count` of them.
///
/// # Safety
/// `request` valid for `request_len` bytes; `out_docs`/`out_scores` for
/// `buf_len` elements each, with `buf_len >= k`; `out_count` for one write.
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
        let (field, target, k, candidates, filter) = decode(blob)?;
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
        let hits = knn_top_hits(&segments, &field, &target, k, candidates, filter.as_ref())
            .map_err(map_search_error)?;
        // SAFETY: caller contract; `hits.len() <= k <= buf_len`.
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
        let (field, target, k, candidates, filter) = decode(&request(3, 10, None)).unwrap();
        assert_eq!(field, "v");
        assert_eq!(
            target,
            lucene_search::knn_search::KnnTarget::Float(vec![1.0, 0.5])
        );
        assert_eq!((k, candidates), (3, 10));
        assert!(filter.is_none());
        assert!(decode(&request(0, 10, None)).is_err());
        assert!(decode(&request(3, 0, None)).is_err());
        let mut trailing = request(3, 10, None);
        trailing.push(0);
        assert!(decode(&trailing).is_err());
        assert!(decode(&request(3, 10, None)[..9]).is_err());
    }
}
