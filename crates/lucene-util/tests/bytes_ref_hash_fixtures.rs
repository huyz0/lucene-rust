//! Differential test for `BytesRefHash` against Lucene 10.5.0: replays
//! `fixtures/data/util_primitives/bytes_ref_hash.txt` (written by
//! `fixtures/src/GenUtilPrimitives.java`) -- six hashes of initial capacity
//! 1..32, each through three rounds of adds (duplicates, 1- and 2-byte
//! length prefixes, terms at the 32766-byte limit that force a new block),
//! finds, `compact`/`clear(resetPool)`/`reinit`, and a final `sort` -- and
//! requires Lucene's ids, byte starts, find results, table sizes and sort
//! order.

use lucene_util::bytes_ref_hash::BytesRefHash;

fn decode(s: &str) -> Vec<u8> {
    if s == "-" {
        return Vec::new();
    }
    if let Some(rest) = s.strip_prefix('*') {
        let (b, len) = rest.split_once('x').unwrap();
        return vec![u8::from_str_radix(b, 16).unwrap(); len.parse().unwrap()];
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn bytes_ref_hash_matches_lucene() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/util_primitives/bytes_ref_hash.txt"
    ))
    .expect("run scripts/gen-fixtures.sh --only GenUtilPrimitives");
    let mut h: Option<BytesRefHash> = None;
    let (mut adds, mut dups, mut sorts) = (0, 0, 0);
    for (ln, line) in text.lines().enumerate() {
        let p: Vec<&str> = line.split(' ').collect();
        let ctx = || format!("line {}", ln + 1);
        match p[0] {
            "new" => {
                h = Some(BytesRefHash::with_capacity(p[1].parse().unwrap(), 12345).unwrap());
            }
            "add" => {
                let h = h.as_mut().unwrap();
                let id = h.add(&decode(p[1])).unwrap();
                assert_eq!(id.to_string(), p[2], "{}", ctx());
                let abs = if id >= 0 { id } else { -id - 1 } as usize;
                assert_eq!(h.byte_start(abs).unwrap().to_string(), p[3], "{}", ctx());
                adds += 1;
                if id < 0 {
                    dups += 1;
                }
            }
            "find" => {
                let got = h.as_ref().unwrap().find(&decode(p[1])).unwrap();
                assert_eq!(got.to_string(), p[2], "{}", ctx());
            }
            "compact" => {
                let c = h.as_mut().unwrap().compact();
                assert_eq!(c.len().to_string(), p[1], "{}", ctx());
            }
            "clear" => {
                let h = h.as_mut().unwrap();
                h.clear(p[1] == "true");
                h.reinit();
            }
            "sort" => {
                let h = h.as_mut().unwrap();
                let n = h.size();
                let sorted = h.sort();
                assert_eq!(sorted.len().to_string(), p[1], "{}", ctx());
                let got: Vec<String> = sorted[..n].iter().map(|v| v.to_string()).collect();
                assert_eq!(got.join(","), p[2], "{}", ctx());
                sorts += 1;
            }
            other => panic!("unknown op {other} at {}", ctx()),
        }
    }
    assert!(
        adds > 5000 && dups > 500 && sorts == 6,
        "{adds} adds, {dups} dups, {sorts} sorts"
    );
}
