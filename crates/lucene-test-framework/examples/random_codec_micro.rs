//! Actual Java RandomCodec's cached and first-lookup routing, without constructor timing.
use std::hint::black_box;
use std::rc::Rc;
use std::time::Instant;

use lucene_test_framework::index::random_codec::{
    java_string_hash, DocValuesFormat, PostingsFormat, RandomCodec,
};
use lucene_util::java_random::JavaRandom;

const KEYS: usize = 4096;
#[derive(Default)]
struct Digest(i64);
impl Digest {
    fn mix(&mut self, value: i64) {
        self.0 = self.0.wrapping_mul(31).wrapping_add(value);
    }
    fn text(&mut self, text: &str) {
        self.mix(text.encode_utf16().count() as i64);
        for unit in text.encode_utf16() {
            self.mix(i64::from(unit));
        }
    }
    fn setting(&mut self, name: &str, value: i64) {
        self.text(name);
        self.mix(value);
    }
}
fn postings_token(format: &PostingsFormat) -> i64 {
    let mut digest = Digest::default();
    digest.text(format.name());
    match format {
        PostingsFormat::Lucene104 {
            min_items,
            max_items,
        }
        | PostingsFormat::BlockTreeOrds {
            min_items,
            max_items,
        } => {
            digest.setting("minTermBlockSize", i64::from(*min_items));
            digest.setting("maxTermBlockSize", i64::from(*max_items));
        }
        PostingsFormat::Direct {
            min_skip_count,
            low_freq_cutoff,
        } => {
            digest.setting("minSkipCount", i64::from(*min_skip_count));
            digest.setting("lowFreqCutoff", i64::from(*low_freq_cutoff));
        }
        PostingsFormat::MockRandom { seed } => digest.setting("seed", *seed),
        PostingsFormat::FixedGap {
            term_index_interval,
        }
        | PostingsFormat::VarGapFixedInterval {
            term_index_interval,
        } => {
            digest.setting("termIndexInterval", i64::from(*term_index_interval));
        }
        PostingsFormat::VarGapDocFreqInterval {
            term_index_interval,
            doc_freq_threshold,
        } => {
            digest.setting("termIndexInterval", i64::from(*term_index_interval));
            digest.setting("docFreqThreshold", i64::from(*doc_freq_threshold));
        }
        PostingsFormat::Fst | PostingsFormat::Bloom | PostingsFormat::Asserting => {}
    }
    digest.0
}
fn doc_values_token(format: &DocValuesFormat) -> i64 {
    let mut digest = Digest::default();
    digest.text(format.name());
    if let DocValuesFormat::Lucene90 {
        skip_index_interval_size,
    } = format
    {
        digest.setting(
            "skipIndexIntervalSize",
            i64::from(*skip_index_interval_size),
        );
    }
    digest.0
}
struct Prepared {
    codec: RandomCodec,
    postings: Vec<Rc<PostingsFormat>>,
    doc_values: Vec<Rc<DocValuesFormat>>,
    postings_tokens: Vec<i64>,
    doc_values_tokens: Vec<i64>,
    constructor_tail: i32,
}
impl Prepared {
    fn new() -> Self {
        let mut random = JavaRandom::new(42);
        let codec = RandomCodec::new(&mut random, false, 1);
        let postings = codec.postings_formats().to_vec();
        let doc_values = codec.doc_values_formats().to_vec();
        let postings_tokens = postings
            .iter()
            .map(|format| postings_token(format))
            .collect();
        let doc_values_tokens = doc_values
            .iter()
            .map(|format| doc_values_token(format))
            .collect();
        Self {
            codec,
            postings,
            doc_values,
            postings_tokens,
            doc_values_tokens,
            constructor_tail: random.next_int(),
        }
    }
    fn setup_digest(&self, fields: &[Vec<u16>]) -> i64 {
        let mut digest = Digest::default();
        digest.mix(i64::from(self.codec.per_field_seed()));
        digest.mix(i64::from(self.constructor_tail));
        let points = self.codec.points_format();
        digest.mix(i64::from(points.max_points_in_leaf_node));
        digest.mix(points.max_mb_sort_in_heap.to_bits() as i64);
        digest.mix(i64::from(points.bkd_split_random_seed));
        for names in [
            self.codec.format_names(),
            self.codec.doc_values_format_names(),
        ] {
            digest.mix(names.len() as i64);
            for name in names {
                digest.text(name);
            }
        }
        for tokens in [&self.postings_tokens, &self.doc_values_tokens] {
            digest.mix(tokens.len() as i64);
            for &token in tokens {
                digest.mix(token);
            }
        }
        for field in fields {
            digest.mix(field.len() as i64);
            for &unit in field {
                digest.mix(i64::from(unit));
            }
            digest.mix(i64::from(java_string_hash(field)));
        }
        digest.0
    }
    fn batch(&mut self, fields: &[Vec<u16>], dv: bool, cached: bool) -> i64 {
        let mut digest = Digest::default();
        let rounds = if cached { 4 } else { 1 };
        for _ in 0..rounds {
            for (index, field) in fields.iter().enumerate() {
                let (ordinal, token) = if dv {
                    let selected = self.codec.doc_values_format_for_field(Some(field)).unwrap();
                    let ordinal = self
                        .doc_values
                        .iter()
                        .position(|format| Rc::ptr_eq(format, &selected))
                        .unwrap();
                    (ordinal, self.doc_values_tokens[ordinal])
                } else {
                    let selected = self.codec.postings_format_for_field(Some(field)).unwrap();
                    let ordinal = self
                        .postings
                        .iter()
                        .position(|format| Rc::ptr_eq(format, &selected))
                        .unwrap();
                    (ordinal, self.postings_tokens[ordinal])
                };
                digest.mix(index as i64);
                digest.mix(ordinal as i64);
                digest.mix(token);
            }
        }
        digest.0
    }
    fn count(&self, dv: bool) -> usize {
        if dv {
            self.codec.doc_values_mapping_count()
        } else {
            self.codec.postings_mapping_count()
        }
    }
}
fn fields() -> Vec<Vec<u16>> {
    (0..KEYS)
        .map(|index| {
            let mut field = match index % 8 {
                0 => vec![0x41, 0x61],
                1 => vec![0x42, 0x42],
                2 => vec![0xd83d, 0xde00],
                3 => vec![0xd800],
                4 => vec![0xdc00],
                5 => vec![0x4e2d, 0xe9],
                6 => Vec::new(),
                _ => vec![0],
            };
            field.push(u16::from(b':'));
            field.extend((index / 8).to_string().encode_utf16());
            field
        })
        .collect()
}
fn run_cached(
    prepared: &mut Prepared,
    fields: &[Vec<u16>],
    dv: bool,
    millis: u64,
    expected: i64,
) -> (u128, u64) {
    let mut elapsed = 0;
    let mut calls = 0;
    loop {
        let start = Instant::now();
        let actual = prepared.batch(fields, dv, true);
        let nanos = start.elapsed().as_nanos();
        assert_eq!(actual, expected);
        black_box(actual);
        assert_eq!(prepared.count(dv), KEYS);
        elapsed += nanos;
        calls += (KEYS * 4) as u64;
        if elapsed >= u128::from(millis) * 1_000_000 {
            return (elapsed, calls);
        }
    }
}
fn run_first(fields: &[Vec<u16>], dv: bool, millis: u64, expected: i64) -> (u128, u64) {
    let mut elapsed = 0;
    let mut calls = 0;
    loop {
        // Both engines construct their unequal component graphs outside timing.
        let mut prepared = Prepared::new();
        assert_eq!(prepared.count(dv), 0);
        let start = Instant::now();
        let actual = prepared.batch(fields, dv, false);
        let nanos = start.elapsed().as_nanos();
        assert_eq!(actual, expected);
        black_box(actual);
        assert_eq!(prepared.count(dv), KEYS);
        elapsed += nanos;
        calls += KEYS as u64;
        drop(prepared); // Destruction is also outside the measured interval.
        if elapsed >= u128::from(millis) * 1_000_000 {
            return (elapsed, calls);
        }
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup: u64 = args.get(1).map_or(500, |value| value.parse().unwrap());
    let measure: u64 = args.get(2).map_or(750, |value| value.parse().unwrap());
    let fields = fields();
    let prepared = Prepared::new();
    let input_digest = prepared.setup_digest(&fields);
    println!("#setup\trandom_codec_inputs\t{input_digest}\t{KEYS}");
    let units: usize = fields.iter().map(Vec::len).sum();
    println!("#memory\trandom_codec_external_key_count\t{KEYS}");
    println!("#memory\trandom_codec_external_logical_utf16_units\t{units}");
    for kind in ["postings", "doc_values"] {
        println!(
            "#memory\trandom_codec_{kind}_cache_owned_utf16_payload_bytes\t{}",
            units * 2
        );
        println!(
            "#memory\trandom_codec_{kind}_cache_inline_key_box_bytes\t{}",
            KEYS * std::mem::size_of::<Box<[u16]>>()
        );
    }
    drop(prepared);
    for dv in [false, true] {
        let kind = if dv { "doc_values" } else { "postings" };
        for cached in [true, false] {
            let mode = if cached { "cached" } else { "first_lookup" };
            let name = format!("random_codec_{kind}_{mode}");
            let mut prepared = Prepared::new();
            let first_expected = prepared.batch(&fields, dv, false);
            assert_eq!(prepared.count(dv), KEYS);
            let expected = if cached {
                prepared.batch(&fields, dv, true)
            } else {
                first_expected
            };
            let units = if cached { KEYS * 4 } else { KEYS };
            println!("#check\t{name}\t{expected}\t{units}\t{input_digest}");
            let (nanos, calls) = if cached {
                run_cached(&mut prepared, &fields, dv, warmup, expected);
                run_cached(&mut prepared, &fields, dv, measure, expected)
            } else {
                drop(prepared);
                run_first(&fields, dv, warmup, expected);
                run_first(&fields, dv, measure, expected)
            };
            println!("{name}\t{:.3}\t{calls}", nanos as f64 / calls as f64);
        }
    }
}
