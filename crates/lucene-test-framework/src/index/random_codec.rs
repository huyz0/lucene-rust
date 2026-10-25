//! Java `tests.index.RandomCodec` constructor and per-field selection.
//!
//! Candidates are constructed in Java order before avoidance filtering, shuffled
//! using the same random stream and truncated to four. The complete allowed
//! name sets precede truncation. Distinct candidates retain identity even when
//! their names and settings are equal. Field keys are Java UTF-16 code units,
//! including unpaired surrogates; hash arithmetic wraps as `String.hashCode`.
//! Cached mappings retain candidate identity, and the 10,000-entry assertion is
//! checked after insertion, so a failed first lookup succeeds when retried.
//!
//! Differs: candidates and fresh point formats are typed settings descriptors,
//! not native format factories. `Rc`/`RefCell` and exclusive routing access replace
//! Java's shared set and synchronized debug maps; this adapter is single-threaded.
//! Native caches own boxed UTF-16 slices, rather than retaining Java String
//! objects; standard map hashing and full slice equality preserve borrowed lookup.
//! Java assertion/exception paths become [`Error`] with assertions enabled.
//! Point configuration records the seed, leaf size and heap limit without
//! creating a BKD writer. Gap: heterogeneous postings/doc-values dispatch,
//! identity-based per-field suffix groups, randomized BKD splitting,
//! Codec/SPI/IndexWriter installation, inherited AssertingCodec composition and
//! Java's display (including HashMap iteration order).

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::rc::Rc;

use lucene_util::java_random::JavaRandom;

use crate::util::lucene_test_case::rarely;
use crate::util::random_numbers::random_int_between;

/// Java's original mutable `Set<String>` reference, shared with the caller.
pub type AvoidCodecs = Rc<RefCell<BTreeSet<String>>>;

/// A distinct constructor candidate. `Rc` identity, not this value's equality,
/// determines which retained candidate a field selects.
#[derive(Debug, PartialEq, Eq)]
pub enum PostingsFormat {
    Lucene104 {
        min_items: i32,
        max_items: i32,
    },
    Fst,
    Direct {
        min_skip_count: i32,
        low_freq_cutoff: i32,
    },
    Bloom,
    MockRandom {
        seed: i64,
    },
    BlockTreeOrds {
        min_items: i32,
        max_items: i32,
    },
    FixedGap {
        term_index_interval: i32,
    },
    VarGapFixedInterval {
        term_index_interval: i32,
    },
    VarGapDocFreqInterval {
        term_index_interval: i32,
        doc_freq_threshold: i32,
    },
    Asserting,
}

impl PostingsFormat {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Lucene104 { .. } => "Lucene104",
            Self::Fst => "FST50",
            Self::Direct { .. } => "Direct",
            Self::Bloom => "TestBloomFilteredLucenePostings",
            Self::MockRandom { .. } => "MockRandom",
            Self::BlockTreeOrds { .. } => "BlockTreeOrds",
            Self::FixedGap { .. } => "LuceneFixedGap",
            Self::VarGapFixedInterval { .. } => "LuceneVarGapFixedInterval",
            Self::VarGapDocFreqInterval { .. } => "LuceneVarGapDocFreqInterval",
            Self::Asserting => "Asserting",
        }
    }
}

/// Java's two separate default Lucene90 candidates have equal settings and
/// distinct `Rc` identities. The asserting candidate wraps a default Lucene90.
#[derive(Debug, PartialEq, Eq)]
pub enum DocValuesFormat {
    Lucene90 { skip_index_interval_size: i32 },
    Asserting,
}
impl DocValuesFormat {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Lucene90 { .. } => "Lucene90",
            Self::Asserting => "Asserting",
        }
    }
}

/// Settings of the fresh asserting/custom points format Java returns per call.
#[derive(Debug, PartialEq)]
pub struct PointsFormat {
    pub max_points_in_leaf_node: i32,
    pub max_mb_sort_in_heap: f64,
    pub bkd_split_random_seed: i32,
}
impl PointsFormat {
    /// Seed passed to `RandomlySplittingBKDWriter` for a Java field name.
    pub fn field_split_seed(&self, field: &[u16]) -> i32 {
        self.bkd_split_random_seed ^ java_string_hash(field)
    }
}

/// Assertion-enabled Java failure classes, represented without Rust panics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    NullAvoidCodecs,
    NullField,
    EmptyFormats,
    NegativeFormatIndex,
    TooManyFields,
}
impl Error {
    pub fn java_class(self) -> &'static str {
        match self {
            Self::NullAvoidCodecs | Self::NullField => "NullPointerException",
            Self::EmptyFormats => "ArithmeticException",
            Self::NegativeFormatIndex => "IndexOutOfBoundsException",
            Self::TooManyFields => "AssertionError",
        }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NullAvoidCodecs => "null avoidance set",
            Self::NullField => "null field name",
            Self::EmptyFormats => "/ by zero",
            Self::NegativeFormatIndex => "negative format index after Math.abs(Integer.MIN_VALUE)",
            Self::TooManyFields => "test went insane",
        })
    }
}
impl std::error::Error for Error {}

type FieldCache<F> = HashMap<Box<[u16]>, Rc<F>>;

/// `java.lang.String.hashCode`, over code units rather than Unicode scalars.
pub fn java_string_hash(field: &[u16]) -> i32 {
    field.iter().fold(0i32, |hash, &unit| {
        hash.wrapping_mul(31).wrapping_add(i32::from(unit))
    })
}

/// Constructor state and identity-preserving, memoized per-field selection.
#[derive(Debug)]
pub struct RandomCodec {
    formats: Vec<Rc<PostingsFormat>>,
    dv_formats: Vec<Rc<DocValuesFormat>>,
    format_names: BTreeSet<String>,
    dv_format_names: BTreeSet<String>,
    avoid_codecs: AvoidCodecs,
    previous_mappings: FieldCache<PostingsFormat>,
    previous_dv_mappings: FieldCache<DocValuesFormat>,
    per_field_seed: i32,
    max_points_in_leaf_node: i32,
    max_mb_sort_in_heap: f64,
    bkd_split_random_seed: i32,
}

fn shuffle<T>(items: &mut [T], random: &mut JavaRandom) {
    // Java: Collections.shuffle(List, Random), RandomAccess branch.
    for size in (2..=items.len()).rev() {
        let selected = random.next_int_bounded(size as i32) as usize;
        items.swap(size - 1, selected);
    }
}

fn select<F>(
    formats: &[Rc<F>],
    cache: &mut FieldCache<F>,
    per_field_seed: i32,
    field: Option<&[u16]>,
) -> Result<Rc<F>, Error> {
    let field = field.ok_or(Error::NullField)?;
    if let Some(format) = cache.get(field) {
        return Ok(Rc::clone(format));
    }
    // Java evaluates the hash before the modulo, including on an empty list.
    let hash = (per_field_seed ^ java_string_hash(field)).wrapping_abs();
    if formats.is_empty() {
        return Err(Error::EmptyFormats);
    }
    let index = hash % formats.len() as i32;
    let index = usize::try_from(index).map_err(|_| Error::NegativeFormatIndex)?;
    let format = Rc::clone(&formats[index]);
    cache.insert(field.to_vec().into_boxed_slice(), Rc::clone(&format));
    if cache.len() >= 10_000 {
        return Err(Error::TooManyFields);
    }
    Ok(format)
}

impl RandomCodec {
    /// Java: `RandomCodec(Random)`, with an empty avoidance set.
    pub fn new(random: &mut JavaRandom, nightly: bool, multiplier: i32) -> Self {
        Self::with_avoid(
            random,
            nightly,
            multiplier,
            Some(Rc::new(RefCell::new(BTreeSet::new()))),
        )
        .expect("non-null empty avoidance set")
    }

    /// Java: `RandomCodec(Random, Set<String>)`. All postings candidates are
    /// constructed before the first `avoidCodecs.contains` can reject null.
    pub fn with_avoid(
        random: &mut JavaRandom,
        nightly: bool,
        multiplier: i32,
        avoid_codecs: Option<AvoidCodecs>,
    ) -> Result<Self, Error> {
        let per_field_seed = random.next_int();
        let min_items = random_int_between(random, 2, 100);
        let max_items = 2 * 2.max(min_items - 1) + random.next_int_bounded(100);
        let low_freq_cutoff = random_int_between(random, 2, 100);
        let max_points_in_leaf_node = random_int_between(random, 16, 2048);
        let max_mb_sort_in_heap = 5.0 + 3.0 * random.next_double();
        let bkd_split_random_seed = random.next_int();
        let min_skip_count = if rarely(random, nightly, multiplier) {
            1
        } else if rarely(random, nightly, multiplier) {
            i32::MAX
        } else {
            max_items
        };
        let direct_low_freq_cutoff = if rarely(random, nightly, multiplier) {
            1
        } else if rarely(random, nightly, multiplier) {
            i32::MAX
        } else {
            low_freq_cutoff
        };
        let candidates = vec![
            PostingsFormat::Lucene104 {
                min_items,
                max_items,
            },
            PostingsFormat::Fst,
            PostingsFormat::Direct {
                min_skip_count,
                low_freq_cutoff: direct_low_freq_cutoff,
            },
            PostingsFormat::Bloom,
            PostingsFormat::MockRandom {
                seed: random.next_long(),
            },
            PostingsFormat::BlockTreeOrds {
                min_items,
                max_items,
            },
            PostingsFormat::FixedGap {
                term_index_interval: random_int_between(random, 1, 1000),
            },
            PostingsFormat::VarGapFixedInterval {
                term_index_interval: random_int_between(random, 1, 1000),
            },
            // Java constructor arguments: threshold first, interval second.
            PostingsFormat::VarGapDocFreqInterval {
                doc_freq_threshold: random_int_between(random, 1, 100),
                term_index_interval: random_int_between(random, 1, 1000),
            },
            PostingsFormat::Lucene104 {
                min_items: 25,
                max_items: 48,
            },
            PostingsFormat::Asserting,
        ];
        let avoid_codecs = avoid_codecs.ok_or(Error::NullAvoidCodecs)?;
        let mut formats = Vec::new();
        let mut format_names = BTreeSet::new();
        for candidate in candidates {
            if !avoid_codecs.borrow().contains(candidate.name()) {
                format_names.insert(candidate.name().to_string());
                formats.push(Rc::new(candidate));
            }
        }
        let mut dv_formats = Vec::new();
        let mut dv_format_names = BTreeSet::new();
        for candidate in [
            DocValuesFormat::Lucene90 {
                skip_index_interval_size: 4096,
            },
            DocValuesFormat::Lucene90 {
                skip_index_interval_size: 4096,
            },
            DocValuesFormat::Asserting,
        ] {
            if !avoid_codecs.borrow().contains(candidate.name()) {
                dv_format_names.insert(candidate.name().to_string());
                dv_formats.push(Rc::new(candidate));
            }
        }
        shuffle(&mut formats, random);
        shuffle(&mut dv_formats, random);
        formats.truncate(4);
        dv_formats.truncate(4);
        Ok(Self {
            formats,
            dv_formats,
            format_names,
            dv_format_names,
            avoid_codecs,
            previous_mappings: HashMap::new(),
            previous_dv_mappings: HashMap::new(),
            per_field_seed,
            max_points_in_leaf_node,
            max_mb_sort_in_heap,
            bkd_split_random_seed,
        })
    }

    pub fn postings_formats(&self) -> &[Rc<PostingsFormat>] {
        &self.formats
    }
    pub fn doc_values_formats(&self) -> &[Rc<DocValuesFormat>] {
        &self.dv_formats
    }
    pub fn format_names(&self) -> &BTreeSet<String> {
        &self.format_names
    }
    pub fn doc_values_format_names(&self) -> &BTreeSet<String> {
        &self.dv_format_names
    }
    pub fn avoid_codecs(&self) -> &AvoidCodecs {
        &self.avoid_codecs
    }
    pub fn per_field_seed(&self) -> i32 {
        self.per_field_seed
    }
    pub fn postings_mapping_count(&self) -> usize {
        self.previous_mappings.len()
    }
    pub fn doc_values_mapping_count(&self) -> usize {
        self.previous_dv_mappings.len()
    }

    /// Java: `getPostingsFormatForField`, including post-insert assertion.
    pub fn postings_format_for_field(
        &mut self,
        field: Option<&[u16]>,
    ) -> Result<Rc<PostingsFormat>, Error> {
        select(
            &self.formats,
            &mut self.previous_mappings,
            self.per_field_seed,
            field,
        )
    }
    /// Java: `getDocValuesFormatForField`, including post-insert assertion.
    pub fn doc_values_format_for_field(
        &mut self,
        field: Option<&[u16]>,
    ) -> Result<Rc<DocValuesFormat>, Error> {
        select(
            &self.dv_formats,
            &mut self.previous_dv_mappings,
            self.per_field_seed,
            field,
        )
    }
    /// Java: `pointsFormat`. A new descriptor preserves per-call identity.
    pub fn points_format(&self) -> Box<PointsFormat> {
        Box::new(PointsFormat {
            max_points_in_leaf_node: self.max_points_in_leaf_node,
            max_mb_sort_in_heap: self.max_mb_sort_in_heap,
            bkd_split_random_seed: self.bkd_split_random_seed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_keeps_java_units_collisions_and_overflow() {
        assert_eq!(java_string_hash(&[]), 0);
        assert_eq!(
            java_string_hash(&[0x41, 0x61]),
            java_string_hash(&[0x42, 0x42])
        );
        assert_eq!(java_string_hash(&[0xd83d, 0xde00]), 1_772_899);
        assert_eq!(java_string_hash(&[0xd800]), 0xd800);
        assert_eq!(java_string_hash(&[0xdc00]), 0xdc00);
        assert_eq!(
            java_string_hash(&"field2".encode_utf16().collect::<Vec<_>>()),
            -1_274_708_360
        );
    }

    #[test]
    fn boxed_cache_keeps_java_hash_collisions_distinct() {
        let first_key = [0x41, 0x61];
        let second_key = [0x42, 0x42];
        assert_eq!(java_string_hash(&first_key), java_string_hash(&second_key));

        let formats = vec![Rc::new(1), Rc::new(2)];
        let mut cache: FieldCache<i32> = HashMap::new();
        let first = select(&formats, &mut cache, 0, Some(&first_key)).unwrap();
        let second = select(&formats, &mut cache, 0, Some(&second_key)).unwrap();
        assert!(Rc::ptr_eq(&first, &second));
        assert_eq!(cache.len(), 2);
        assert!(Rc::ptr_eq(&cache[first_key.as_slice()], &first));
        assert!(Rc::ptr_eq(&cache[second_key.as_slice()], &second));
    }

    #[test]
    fn signed_minimum_selection_and_insert_before_assert_are_java() {
        let formats = vec![Rc::new(1), Rc::new(2), Rc::new(3)];
        let mut cache: FieldCache<i32> = HashMap::new();
        assert_eq!(
            select(&formats, &mut cache, i32::MIN, Some(&[])),
            Err(Error::NegativeFormatIndex)
        );
        assert_eq!(
            select(&formats[..2], &mut cache, i32::MIN, Some(&[])),
            Ok(Rc::clone(&formats[0]))
        );
        assert_eq!(select(&[], &mut cache, 0, None), Err(Error::NullField));
        assert_eq!(
            select(&[], &mut cache, 0, Some(&[1])),
            Err(Error::EmptyFormats)
        );
        cache.clear();
        for i in 0..9999u16 {
            assert!(select(&formats, &mut cache, 0, Some(&[i])).is_ok());
        }
        assert_eq!(
            select(&formats, &mut cache, 0, Some(&[9999])),
            Err(Error::TooManyFields)
        );
        let retry = select(&formats, &mut cache, 0, Some(&[9999])).unwrap();
        assert!(Rc::ptr_eq(&retry, &formats[0]));
        assert_eq!(cache.len(), 10000);
    }

    #[test]
    fn shared_avoid_mutation_does_not_rebuild_candidates_or_merge_equal_instances() {
        let avoid = Rc::new(RefCell::new(BTreeSet::new()));
        let mut random = JavaRandom::new(42);
        let mut codec =
            RandomCodec::with_avoid(&mut random, false, 1, Some(Rc::clone(&avoid))).unwrap();
        assert!(Rc::ptr_eq(codec.avoid_codecs(), &avoid));
        let before = codec
            .postings_formats()
            .iter()
            .map(Rc::clone)
            .collect::<Vec<_>>();
        avoid
            .borrow_mut()
            .extend(codec.format_names().iter().cloned());
        assert!(before
            .iter()
            .zip(codec.postings_formats())
            .all(|(a, b)| Rc::ptr_eq(a, b)));
        assert_eq!(codec.format_names().len(), 10);
        let equal: Vec<_> = codec
            .doc_values_formats()
            .iter()
            .filter(|f| matches!(***f, DocValuesFormat::Lucene90 { .. }))
            .collect();
        assert_eq!(equal[0], equal[1]);
        assert!(!Rc::ptr_eq(equal[0], equal[1]));
        assert_eq!(codec.postings_format_for_field(None), Err(Error::NullField));
        assert_eq!(
            codec.doc_values_format_for_field(None),
            Err(Error::NullField)
        );
        let points = codec.points_format();
        let another = codec.points_format();
        assert_eq!(points, another);
        assert!(!std::ptr::eq(points.as_ref(), another.as_ref()));
        assert_eq!(
            points.field_split_seed(&[0xd800]),
            points.bkd_split_random_seed ^ 0xd800
        );
        assert_eq!(codec.postings_mapping_count(), 0);
        assert_eq!(codec.doc_values_mapping_count(), 0);
        assert_eq!(codec.per_field_seed(), -1_170_105_035);
    }

    #[test]
    fn errors_have_explicit_assertion_enabled_classes_and_messages() {
        for (error, class, message) in [
            (
                Error::NullAvoidCodecs,
                "NullPointerException",
                "null avoidance set",
            ),
            (Error::NullField, "NullPointerException", "null field name"),
            (Error::EmptyFormats, "ArithmeticException", "/ by zero"),
            (
                Error::NegativeFormatIndex,
                "IndexOutOfBoundsException",
                "negative format index after Math.abs(Integer.MIN_VALUE)",
            ),
            (Error::TooManyFields, "AssertionError", "test went insane"),
        ] {
            assert_eq!(error.java_class(), class);
            assert_eq!(error.to_string(), message);
        }
    }

    fn outcome(codec: &mut RandomCodec, dv: bool, field: Option<&[u16]>) -> String {
        let result = if dv {
            codec
                .doc_values_format_for_field(field)
                .map(|format| format.name())
        } else {
            codec
                .postings_format_for_field(field)
                .map(|format| format.name())
        };
        match result {
            Ok(name) => format!("ok:{name}"),
            Err(error) => error.java_class().to_string(),
        }
    }

    #[test]
    fn actual_java_injected_minimum_cache_limit_and_null_constructor_rows_match() {
        for (nightly, multiplier, fixture) in [
            (
                false,
                1,
                include_str!("../../../../fixtures/testframework/data/random_codec_false_1.tsv"),
            ),
            (
                false,
                3,
                include_str!("../../../../fixtures/testframework/data/random_codec_false_3.tsv"),
            ),
            (
                true,
                1,
                include_str!("../../../../fixtures/testframework/data/random_codec_true_1.tsv"),
            ),
            (
                true,
                3,
                include_str!("../../../../fixtures/testframework/data/random_codec_true_3.tsv"),
            ),
        ] {
            let mut actual = String::new();
            for dv in [false, true] {
                let mut codec = RandomCodec::new(&mut JavaRandom::new(42), nightly, multiplier);
                if !dv {
                    codec.formats.truncate(3);
                }
                codec.per_field_seed = i32::MIN;
                actual.push_str(&format!(
                    "min-abs\t{dv}\t{}\n",
                    outcome(&mut codec, dv, Some(&[]))
                ));
                actual.push_str(&format!(
                    "min-abs-retry\t{dv}\t{}\n",
                    outcome(&mut codec, dv, Some(&[]))
                ));
                if dv {
                    codec.dv_formats.truncate(2);
                } else {
                    codec.formats.truncate(2);
                }
                actual.push_str(&format!(
                    "min-abs-zero-remainder\t{dv}\t{}\n",
                    outcome(&mut codec, dv, Some(&[]))
                ));

                let mut codec = RandomCodec::new(&mut JavaRandom::new(42), nightly, multiplier);
                for i in 0..9999 {
                    let key: Vec<_> = format!("cache{i}").encode_utf16().collect();
                    assert!(outcome(&mut codec, dv, Some(&key)).starts_with("ok:"));
                }
                let key: Vec<_> = "cache9999".encode_utf16().collect();
                actual.push_str(&format!(
                    "cache-limit\t{dv}\t{}\n",
                    outcome(&mut codec, dv, Some(&key))
                ));
                actual.push_str(&format!(
                    "cache-retry\t{dv}\t{}\n",
                    outcome(&mut codec, dv, Some(&key))
                ));
                let next: Vec<_> = "cache10000".encode_utf16().collect();
                actual.push_str(&format!(
                    "cache-next\t{dv}\t{}\n",
                    outcome(&mut codec, dv, Some(&next))
                ));
                let size = if dv {
                    codec.doc_values_mapping_count()
                } else {
                    codec.postings_mapping_count()
                };
                actual.push_str(&format!("cache-size\t{dv}\t{size}\n"));
            }
            let avoid = Rc::new(RefCell::new(BTreeSet::new()));
            let codec = RandomCodec::with_avoid(
                &mut JavaRandom::new(42),
                nightly,
                multiplier,
                Some(Rc::clone(&avoid)),
            )
            .unwrap();
            let before: Vec<_> = codec.postings_formats().iter().map(Rc::clone).collect();
            avoid
                .borrow_mut()
                .extend(codec.format_names().iter().cloned());
            actual.push_str(&format!(
                "avoid-mutation\t{}\t{}\t{}\n",
                Rc::ptr_eq(codec.avoid_codecs(), &avoid),
                before
                    .iter()
                    .zip(codec.postings_formats())
                    .all(|(a, b)| Rc::ptr_eq(a, b)),
                avoid.borrow().len()
            ));
            let mut random = JavaRandom::new(42);
            let error =
                RandomCodec::with_avoid(&mut random, nightly, multiplier, None).unwrap_err();
            actual.push_str(&format!(
                "null-avoid\t{}\t{}\n",
                error.java_class(),
                random.next_int()
            ));
            let expected = fixture
                .lines()
                .skip_while(|row| !row.starts_with("min-abs\t"))
                .map(|row| format!("{row}\n"))
                .collect::<String>();
            assert_eq!(
                actual, expected,
                "nightly={nightly},multiplier={multiplier}"
            );
        }
    }
}
