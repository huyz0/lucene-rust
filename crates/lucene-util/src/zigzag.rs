//! ZigZag encoding, matching `org.apache.lucene.util.BitUtil.zigZag{Encode,Decode}`.

#[inline]
pub fn encode(v: i64) -> u64 {
    ((v >> 63) ^ (v << 1)) as u64
}

#[inline]
pub fn decode(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_edges() {
        for v in [0, 1, -1, i64::MAX, i64::MIN, 123456789, -987654321] {
            assert_eq!(decode(encode(v)), v);
        }
    }

    /// Cross-checked against Java BitUtil via fixtures/data/zigzag_pairs.expected.
    #[test]
    fn matches_java_reference() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/zigzag_pairs.expected"
        );
        let text = std::fs::read_to_string(path).expect("run fixtures/ generator first");
        for line in text.lines() {
            let mut it = line.split_whitespace();
            let v: i64 = it.next().unwrap().parse().unwrap();
            let enc: i64 = it.next().unwrap().parse().unwrap(); // Java prints as signed
            assert_eq!(encode(v), enc as u64, "encode({v})");
            assert_eq!(decode(enc as u64), v, "decode({enc})");
        }
    }
}
