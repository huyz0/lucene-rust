//! A whole-reader KNN search, run before the rest of a request: the
//! `lucene` engine's `knn` query is `KnnFloatVectorQuery`, whose
//! `rewrite(IndexSearcher)` finds the reader's top `k` and becomes a
//! `DocAndScoreQuery`. The plugin's query does that rewrite through
//! `ffi_jvm_reader_knn` and sends the hits back in the request's tree as a
//! doc-and-score node (`NODE_DOC_AND_SCORE`), so the request itself runs
//! native like any other.
//!
//! Request blob: `field` (`i32 len`, UTF-8), the target
//! ([`crate::queries::decode_knn_target`]), `k: i32`, `candidates: i32`
//! (the beam the search keeps before cutting to `k`, `max(k, ef_search)`),
//! `has_filter: u8` and the filter (a query-tree node) when `1`, then
//! `nested: u8`: when `1`, the field is nested and the parent filter (a
//! query-tree node, whose bit sets are cached as a nested query's are) and
//! `expand: u8` (`expand_nested_docs`) follow, and last `nprobes: i32`: the
//! clusters probed in each segment whose field has an IVF index
//! (`IvfVectorsFormat`; `method_parameters.nprobes`), `0` for the
//! depth each segment calibrated, for a nested search too (more when fewer
//! than `k` parents are in reach), then `has_slack: u8` and, when `1`,
//! `exploration_slack: f32`: the graph walks' opt-in adaptive beam stop
//! (`method_parameters.exploration_slack`, `KnnFloatVectorQuery::
//! exploration_slack` over the best `k`; a nested search ignores it). Then
//! `grouped: u8`: when `1`, the search is grouped (the plugin's `group_by`):
//! the group field (`i32 len`, UTF-8), its kind (`0` numeric, `1` keyword,
//! as the collapse blob's) and `group_size: i32` follow, and `k` counts
//! groups; a grouped search is never nested, and ignores the slack and the
//! recall target.
//!
//! An optional tail asks for a recall-targeted search
//! ([`lucene_search::knn_recall`], `index.knn.target_recall`):
//! `recall: f64`, `sample_rate: f64`, `budget: f64` and the tuner's key
//! (`i32 len`, UTF-8; the plugin names the index, shard and field). The
//! search then runs at the depth the key's tuner has tuned, and a sampled
//! one is checked on a thread of its own, after the hits are returned, so
//! the check (an exact scan of the field) adds nothing to the query's
//! latency. That thread holds the reader until the check is done. Neither a
//! search over a realtime view's buffers nor an expanded nested one is
//! checked ([`lucene_search::knn_recall::check_sample`]).
//!
//! A nested search keeps the best child of each of the best `k` parents; an
//! expanded one returns every scored child of those parents, which can be
//! more than `k`, as can a grouped one (`k` times `group_size`): when they do
//! not fit the buffers, the call is [`Status::BufferTooSmall`] with the
//! count it needs in `*out_count`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use lucene_search::join::BitSetProducer;
use lucene_search::knn_recall::{RecallTarget, RecallTuner};
use lucene_search::knn_search::{knn_top_hits_grouped, KnnGroupBy};
use lucene_search::multi_segment::OpenSegment;

use crate::clauses::map_search_error;
use crate::query_blob::Cursor;
use crate::reader_handle::JvmReaderHandle;
use crate::status::{set_last_error, Status};

/// A decoded request.
pub struct Request {
    pub field: String,
    target: lucene_search::knn_search::KnnTarget,
    pub k: usize,
    candidates: usize,
    pub filter: Option<lucene_search::query::Clause>,
    /// Where the filter's encoding sits in the request (empty without one):
    /// two requests with the same bytes there filter alike.
    pub filter_bytes: std::ops::Range<usize>,
    /// A nested field's parent filter and `expand_nested_docs`.
    pub nested: Option<(Arc<dyn BitSetProducer>, bool)>,
    nprobes: usize,
    exploration_slack: Option<f32>,
    /// A grouped search's field, kind and documents per group.
    pub group_by: Option<KnnGroupBy>,
    /// The recall-targeted tail ([module doc](self)).
    recall: Option<RecallRequest>,
}

impl Request {
    /// Whether the search reads whole segments only: a nested or grouped
    /// one, which searches no realtime buffer.
    pub fn whole_segments(&self) -> bool {
        self.nested.is_some() || self.group_by.is_some()
    }
}

/// A request's recall target and its tuner's key, and the request's bytes
/// for the check to decode again on its own thread.
pub struct RecallRequest {
    target: RecallTarget,
    key: String,
    blob: Vec<u8>,
}

fn invalid(message: &str) -> Status {
    set_last_error(format!("KNN request: {message}"));
    Status::InvalidArgument
}

pub fn decode(blob: &[u8]) -> Result<Request, Status> {
    let mut c = Cursor::new(blob);
    let field = std::str::from_utf8(c.bytes()?)
        .map_err(|_| Status::InvalidUtf8)?
        .to_string();
    let target = crate::queries::decode_knn_target(&mut c)?;
    let positive = |v: i32, what: &str| {
        usize::try_from(v)
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| invalid(what))
    };
    let k = positive(c.i32()?, "k must be at least 1")?;
    let candidates = positive(c.i32()?, "candidates must be at least 1")?;
    let mut nodes = 0usize;
    let (filter, filter_bytes) = match c.u8()? {
        0 => (None, 0..0),
        1 => {
            let start = blob.len() - c.remaining();
            let node = crate::query_blob::decode_node(&mut c, 0, &mut nodes)?;
            (Some(node), start..blob.len() - c.remaining())
        }
        _ => return Err(invalid("filter flag (expected 0..=1)")),
    };
    let nested = match c.u8()? {
        0 => None,
        1 => {
            let parents = crate::nodes::decode_parents(&mut c, 0, &mut nodes)?;
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
    let exploration_slack = match c.u8()? {
        0 => None,
        1 => {
            let slack = f32::from_bits(c.i32()? as u32);
            if !slack.is_finite() || slack < 0.0 {
                return Err(invalid("exploration slack must be a finite number >= 0"));
            }
            Some(slack)
        }
        _ => return Err(invalid("slack flag (expected 0..=1)")),
    };
    let group_by = match c.u8()? {
        0 => None,
        1 => {
            let field = std::str::from_utf8(c.bytes()?)
                .map_err(|_| Status::InvalidUtf8)?
                .to_string();
            let kind = match c.u8()? {
                0 => lucene_search::top_field::CollapseKind::Numeric,
                1 => lucene_search::top_field::CollapseKind::Keyword,
                _ => return Err(invalid("group kind (expected 0..=1)")),
            };
            let per_group = positive(c.i32()?, "group_size must be at least 1")?;
            if nested.is_some() {
                return Err(invalid("a grouped search cannot be nested"));
            }
            Some(KnnGroupBy {
                field,
                kind,
                per_group,
            })
        }
        _ => return Err(invalid("grouped flag (expected 0..=1)")),
    };
    let recall = if c.remaining() == 0 {
        None
    } else {
        // An `f64` outside `f32`'s range becomes infinite, which the target
        // refuses.
        let recall = c.f64()? as f32;
        let sample_rate = c.f64()? as f32;
        let budget = c.f64()? as f32;
        let key = std::str::from_utf8(c.bytes()?)
            .map_err(|_| Status::InvalidUtf8)?
            .to_string();
        let target = RecallTarget::new(recall, sample_rate)
            .and_then(|t| t.with_budget(budget))
            .map_err(|e| invalid(&e.to_string()))?;
        Some(RecallRequest {
            target,
            key,
            blob: blob.to_vec(),
        })
    };
    if c.remaining() != 0 {
        return Err(invalid("trailing bytes"));
    }
    Ok(Request {
        field,
        target,
        k,
        candidates,
        filter,
        filter_bytes,
        nested,
        nprobes,
        exploration_slack,
        group_by,
        recall,
    })
}

/// The most tuners kept: one per index, shard and field searched
/// recall-targeted. Past it, the registry starts again empty.
const MAX_TUNERS: usize = 4096;

fn tuners() -> &'static Mutex<HashMap<String, Arc<RecallTuner>>> {
    static TUNERS: OnceLock<Mutex<HashMap<String, Arc<RecallTuner>>>> = OnceLock::new();
    TUNERS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The tuner for `r`'s key, a new one when there is none or the target
/// changed (a setting updated).
fn tuner(r: &RecallRequest) -> Arc<RecallTuner> {
    let mut all = tuners()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(t) = all.get(&r.key).filter(|t| t.target() == r.target) {
        return t.clone();
    }
    if all.len() >= MAX_TUNERS && !all.contains_key(&r.key) {
        all.clear();
    }
    let t = Arc::new(RecallTuner::new(r.target));
    all.insert(r.key.clone(), t.clone());
    t
}

/// Every tuner's [`lucene_search::knn_recall::RecallStats`], as a JSON
/// object keyed by tuner key.
pub fn recall_stats_json() -> String {
    let all = tuners()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut keys: Vec<&String> = all.keys().collect();
    keys.sort();
    let number = |v: Option<f64>| {
        v.filter(|v| v.is_finite())
            .map_or("null".to_string(), |v| v.to_string())
    };
    let mut out = String::from("{");
    for (i, key) in keys.into_iter().enumerate() {
        let s = all[key].stats();
        if i > 0 {
            out.push(',');
        }
        json_string(&mut out, key);
        out.push_str(&format!(
            concat!(
                ":{{\"target_recall\":{},\"sample_rate\":{},\"check_budget\":{},",
                "\"level\":{},\"effort\":{},\"recall\":{},\"recall_lower_bound\":{},",
                "\"queries\":{},\"samples\":{},\"search_nanos\":{},\"check_nanos\":{}}}"
            ),
            s.target.recall,
            s.target.sample_rate,
            s.target.budget,
            s.level,
            s.effort,
            number(s.recall),
            number(s.recall_lower_bound),
            s.queries,
            s.samples,
            s.search_nanos,
            s.check_nanos,
        ));
    }
    out.push('}');
    out
}

/// `s` as a JSON string.
fn json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if u32::from(c) < 0x20 => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `request`'s hits over the reader of `h`, its deletes skipped: global doc
/// ids, best first.
pub fn search(
    h: &Arc<JvmReaderHandle>,
    request: &Request,
    buffers: &[lucene_search::knn_search::BufferedVectors<'_>],
) -> Result<Vec<lucene_search::ScoreDoc>, Status> {
    if request.group_by.is_some() {
        return search_at(h, request, buffers, 1.0);
    }
    let Some(recall) = &request.recall else {
        return search_at(h, request, buffers, 1.0);
    };
    let tuner = tuner(recall);
    let search = tuner.begin();
    let hits = search_at(h, request, buffers, search.effort);
    tuner.searched(&search);
    if search.sampled {
        match &hits {
            Ok(hits) if buffers.is_empty() => check_later(h, recall, tuner, search, hits),
            _ => tuner.release(),
        }
    }
    hits
}

/// Checks a sampled search on a thread of its own ([module doc](self)).
fn check_later(
    h: &Arc<JvmReaderHandle>,
    recall: &RecallRequest,
    tuner: Arc<RecallTuner>,
    search: lucene_search::knn_recall::TunedSearch,
    hits: &[lucene_search::ScoreDoc],
) {
    let h = h.clone();
    let blob = recall.blob.clone();
    let hits = hits.to_vec();
    let releasing = tuner.clone();
    let spawned = std::thread::Builder::new()
        .name("knn-recall-check".into())
        .spawn(move || {
            let checked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                check(&h, &blob, &tuner, &search, &hits)
            }));
            if !matches!(checked, Ok(Ok(()))) {
                tuner.release();
            }
        });
    if spawned.is_err() {
        releasing.release();
    }
}

/// [`lucene_search::knn_recall::check_sample`] of the request in `blob`
/// over the reader of `h`.
fn check(
    h: &JvmReaderHandle,
    blob: &[u8],
    tuner: &RecallTuner,
    search: &lucene_search::knn_recall::TunedSearch,
    hits: &[lucene_search::ScoreDoc],
) -> Result<(), Status> {
    let request = decode(blob)?;
    with_segments(h, &request, |segments| {
        lucene_search::knn_recall::check_sample(
            segments,
            &request.field,
            &request.target,
            request.k,
            request.candidates,
            request.filter.as_ref(),
            request
                .nested
                .as_ref()
                .map(|(p, expand)| (&**p as &dyn BitSetProducer, *expand)),
            request.nprobes,
            &[],
            request.exploration_slack,
            tuner,
            search,
            hits,
        )
        .map_err(map_search_error)
    })
}

/// `f` over the reader of `h`'s segments, their deletes and (for the
/// request's filter) points opened.
fn with_segments<T>(
    h: &JvmReaderHandle,
    request: &Request,
    f: impl FnOnce(&[OpenSegment<'_>]) -> Result<T, Status>,
) -> Result<T, Status> {
    let mut opened = h.reader.open_segments().map_err(|e| {
        set_last_error(format!("opening segment postings: {e}"));
        Status::Decode
    })?;
    if let Some(filter) = &request.filter {
        let probe = lucene_search::query::BooleanQuery::new().with_filter([filter.clone()]);
        if crate::query_blob::boolean_uses_points(&probe) {
            opened.open_points().map_err(|e| {
                set_last_error(format!("opening segment points: {e}"));
                Status::Decode
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
    f(&segments)
}

/// `request`'s hits at search `effort` (`1.0`: as asked).
fn search_at(
    h: &JvmReaderHandle,
    request: &Request,
    buffers: &[lucene_search::knn_search::BufferedVectors<'_>],
    effort: f32,
) -> Result<Vec<lucene_search::ScoreDoc>, Status> {
    with_segments(h, request, |segments| {
        if let Some(group_by) = &request.group_by {
            return knn_top_hits_grouped(
                segments,
                &request.field,
                &request.target,
                request.k,
                request.candidates,
                request.filter.as_ref(),
                group_by,
                request.nprobes,
            )
            .map_err(map_search_error);
        }
        lucene_search::knn_recall::knn_top_hits_at_effort(
            segments,
            &request.field,
            &request.target,
            request.k,
            request.candidates,
            request.filter.as_ref(),
            request
                .nested
                .as_ref()
                .map(|(p, expand)| (&**p as &dyn BitSetProducer, *expand)),
            request.nprobes,
            buffers,
            request.exploration_slack,
            effort,
        )
        .map_err(map_search_error)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(k: i32, candidates: i32, filter: Option<&[u8]>) -> Vec<u8> {
        let mut b = request_without_nested(k, candidates, filter);
        b.push(0);
        b.extend_from_slice(&0i32.to_le_bytes());
        // No exploration slack, not grouped.
        b.push(0);
        b.push(0);
        b
    }

    fn request_without_nested(k: i32, candidates: i32, filter: Option<&[u8]>) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&1i32.to_le_bytes());
        b.push(b'v');
        b.push(crate::queries::KNN_TARGET_FLOAT);
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
        probed[n - 6..n - 2].copy_from_slice(&32i32.to_le_bytes());
        assert_eq!(decode(&probed).unwrap().nprobes, 32);
        probed[n - 6..n - 2].copy_from_slice(&(-1i32).to_le_bytes());
        assert!(decode(&probed).is_err());
        assert!(decode(&request(0, 10, None)).is_err());
        assert!(decode(&request(3, 0, None)).is_err());
        let mut trailing = request(3, 10, None);
        trailing.push(0);
        assert!(decode(&trailing).is_err());
        assert!(decode(&request(3, 10, None)[..9]).is_err());
        assert!(decode(&request_without_nested(3, 10, None)).is_err());
    }

    fn with_recall(mut b: Vec<u8>, recall: f64, rate: f64, budget: f64, key: &str) -> Vec<u8> {
        for v in [recall, rate, budget] {
            b.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        b.extend_from_slice(&(key.len() as i32).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b
    }

    #[test]
    fn decodes_a_recall_target_and_refuses_bad_ones() {
        let b = with_recall(request(3, 10, None), 0.95, 0.01, 0.05, "i/0/v");
        let r = decode(&b).unwrap().recall.unwrap();
        assert_eq!(r.target, RecallTarget::new(0.95, 0.01).unwrap());
        assert_eq!(r.key, "i/0/v");
        assert_eq!(r.blob, b);
        assert!(decode(&request(3, 10, None)).unwrap().recall.is_none());
        for (recall, rate, budget) in [
            (0.0, 0.01, 0.05),
            (1.5, 0.01, 0.05),
            (0.9, 0.0, 0.05),
            (0.9, 0.5, -1.0),
            (1e300, 0.5, 0.05),
        ] {
            let b = with_recall(request(3, 10, None), recall, rate, budget, "k");
            assert!(decode(&b).is_err(), "{recall} {rate} {budget}");
        }
        let mut bad_key = with_recall(request(3, 10, None), 0.9, 0.5, 0.05, "k");
        let n = bad_key.len();
        bad_key[n - 1] = 0xff;
        assert!(decode(&bad_key).is_err());
        let truncated = with_recall(request(3, 10, None), 0.9, 0.5, 0.05, "k");
        assert!(decode(&truncated[..truncated.len() - 2]).is_err());
    }

    #[test]
    fn a_tuner_is_kept_per_key_until_its_target_changes() {
        let req = |recall: f64, key: &str| {
            decode(&with_recall(request(3, 10, None), recall, 0.5, 0.05, key))
                .unwrap()
                .recall
                .unwrap()
        };
        let key = format!("tuner-test-{}", std::process::id());
        let a = tuner(&req(0.9, &key));
        assert!(Arc::ptr_eq(&a, &tuner(&req(0.9, &key))));
        let b = tuner(&req(0.8, &key));
        assert!(!Arc::ptr_eq(&a, &b));
        assert_eq!(b.target().recall, 0.8);
        let json = recall_stats_json();
        assert!(json.starts_with('{') && json.ends_with('}'), "{json}");
        assert!(
            json.contains(&format!("\"{key}\":{{\"target_recall\":0.8,")),
            "{json}"
        );
        assert!(json.contains("\"recall\":null"), "{json}");
    }

    #[test]
    fn json_strings_are_escaped() {
        let mut out = String::new();
        json_string(&mut out, "a\"b\\c\nd");
        assert_eq!(out, "\"a\\\"b\\\\c\\u000ad\"");
    }

    #[test]
    fn decodes_the_exploration_slack() {
        assert_eq!(
            decode(&request(3, 10, None)).unwrap().exploration_slack,
            None
        );
        let with = |bits: u32| {
            let mut b = request(3, 10, None);
            b.truncate(b.len() - 2);
            b.push(1);
            b.extend_from_slice(&bits.to_le_bytes());
            b.push(0);
            b
        };
        assert_eq!(
            decode(&with(0.25f32.to_bits())).unwrap().exploration_slack,
            Some(0.25)
        );
        assert_eq!(
            decode(&with(0.0f32.to_bits())).unwrap().exploration_slack,
            Some(0.0)
        );
        for bad in [-0.5f32, f32::NAN, f32::INFINITY] {
            assert!(decode(&with(bad.to_bits())).is_err(), "{bad}");
        }
        let mut flag = request(3, 10, None);
        let n = flag.len();
        flag[n - 2] = 2;
        assert!(decode(&flag).is_err());
        // The flag set without its value.
        let mut short = request(3, 10, None);
        short[n - 2] = 1;
        assert!(decode(&short).is_err());
    }

    #[test]
    fn decodes_a_nested_request() {
        let mut b = request_without_nested(3, 10, None);
        b.push(1);
        b.push(5); // the parent filter: match_all
        b.push(1);
        b.extend_from_slice(&0i32.to_le_bytes());
        b.push(0);
        b.push(0);
        let r = decode(&b).unwrap();
        assert!(r.whole_segments() && r.group_by.is_none());
        let (parents, expand) = r.nested.unwrap();
        assert!(expand);
        assert!(parents.key().starts_with("parents("));
        // Nested and grouped at once: refused.
        let mut both = b.clone();
        both.pop();
        both.extend_from_slice(&grouping("cat", 1, 2));
        assert!(decode(&both).is_err());
        b.truncate(b.len() - 7);
        b.push(2);
        b.extend_from_slice(&0i32.to_le_bytes());
        b.push(0);
        b.push(0);
        assert!(decode(&b).is_err());
    }

    /// The grouped tail: flag, field, kind, group size.
    fn grouping(field: &str, kind: u8, size: i32) -> Vec<u8> {
        let mut b = vec![1u8];
        b.extend_from_slice(&(field.len() as i32).to_le_bytes());
        b.extend_from_slice(field.as_bytes());
        b.push(kind);
        b.extend_from_slice(&size.to_le_bytes());
        b
    }

    #[test]
    fn decodes_a_grouped_request() {
        let plain = request(3, 10, None);
        assert!(!decode(&plain).unwrap().whole_segments());
        let with = |tail: Vec<u8>| {
            let mut b = plain[..plain.len() - 1].to_vec();
            b.extend_from_slice(&tail);
            b
        };
        let r = decode(&with(grouping("cat", 1, 2))).unwrap();
        assert!(r.whole_segments());
        assert_eq!(
            r.group_by,
            Some(KnnGroupBy {
                field: "cat".into(),
                kind: lucene_search::top_field::CollapseKind::Keyword,
                per_group: 2,
            })
        );
        let r = decode(&with(grouping("n", 0, 1))).unwrap();
        assert_eq!(
            r.group_by.unwrap().kind,
            lucene_search::top_field::CollapseKind::Numeric
        );
        assert!(decode(&with(grouping("n", 2, 1))).is_err());
        assert!(decode(&with(grouping("n", 0, 0))).is_err());
        assert!(decode(&with(vec![2])).is_err());
        let mut bad_utf8 = grouping("n", 0, 1);
        bad_utf8[5] = 0xFF;
        assert!(decode(&with(bad_utf8)).is_err());
        // A recall target after a grouped tail decodes too.
        let recall = with_recall(with(grouping("n", 0, 1)), 0.9, 0.5, 0.05, "k");
        let r = decode(&recall).unwrap();
        assert!(r.group_by.is_some() && r.recall.is_some());
    }
}
