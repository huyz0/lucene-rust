#![cfg(test)]
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
