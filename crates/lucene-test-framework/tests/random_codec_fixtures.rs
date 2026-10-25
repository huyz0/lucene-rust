//! Real RandomCodec constructor state, candidate identities and UTF-16 routes.
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

use lucene_test_framework::index::random_codec::{
    java_string_hash, AvoidCodecs, DocValuesFormat, PostingsFormat, RandomCodec,
};
use lucene_util::java_random::JavaRandom;

const POSTINGS_NAMES: [&str; 10] = [
    "Lucene104",
    "FST50",
    "Direct",
    "TestBloomFilteredLucenePostings",
    "MockRandom",
    "BlockTreeOrds",
    "LuceneFixedGap",
    "LuceneVarGapFixedInterval",
    "LuceneVarGapDocFreqInterval",
    "Asserting",
];
const FIXTURES: [(bool, i32, &str); 4] = [
    (
        false,
        1,
        include_str!("../../../fixtures/testframework/data/random_codec_false_1.tsv"),
    ),
    (
        false,
        3,
        include_str!("../../../fixtures/testframework/data/random_codec_false_3.tsv"),
    ),
    (
        true,
        1,
        include_str!("../../../fixtures/testframework/data/random_codec_true_1.tsv"),
    ),
    (
        true,
        3,
        include_str!("../../../fixtures/testframework/data/random_codec_true_3.tsv"),
    ),
];

fn avoidance(label: &str) -> AvoidCodecs {
    let names: Vec<_> = match label {
        "none" => Vec::new(),
        "unknown" => vec!["unknown"],
        "all-postings" => POSTINGS_NAMES.to_vec(),
        "all-dv" => vec!["Lucene90", "Asserting"],
        "only-lucene104" => POSTINGS_NAMES
            .into_iter()
            .filter(|name| *name != "Lucene104")
            .collect(),
        label => vec![label.strip_prefix("omit-").unwrap()],
    };
    Rc::new(RefCell::new(names.into_iter().map(str::to_owned).collect()))
}

fn units(text: &str) -> Vec<u16> {
    if text == "-" {
        Vec::new()
    } else {
        text.split(',')
            .map(|unit| u16::from_str_radix(unit, 16).unwrap())
            .collect()
    }
}

fn name_set(names: &BTreeSet<String>) -> String {
    format!("[{}]", names.iter().cloned().collect::<Vec<_>>().join(", "))
}

fn postings_settings(format: &PostingsFormat) -> (&'static str, String) {
    match format {
        PostingsFormat::Lucene104 {
            min_items,
            max_items,
        } => (
            "Lucene104PostingsFormat",
            format!("minTermBlockSize={min_items},maxTermBlockSize={max_items}"),
        ),
        PostingsFormat::Fst => ("FSTPostingsFormat", "-".into()),
        PostingsFormat::Direct {
            min_skip_count,
            low_freq_cutoff,
        } => (
            "DirectPostingsFormat",
            format!("minSkipCount={min_skip_count},lowFreqCutoff={low_freq_cutoff}"),
        ),
        PostingsFormat::Bloom => ("TestBloomFilteredLucenePostings", "-".into()),
        PostingsFormat::MockRandom { .. } => ("MockRandomPostingsFormat", "-".into()),
        PostingsFormat::BlockTreeOrds {
            min_items,
            max_items,
        } => (
            "BlockTreeOrdsPostingsFormat",
            format!("minTermBlockSize={min_items},maxTermBlockSize={max_items}"),
        ),
        PostingsFormat::FixedGap {
            term_index_interval,
        } => (
            "LuceneFixedGap",
            format!("termIndexInterval={term_index_interval}"),
        ),
        PostingsFormat::VarGapFixedInterval {
            term_index_interval,
        } => (
            "LuceneVarGapFixedInterval",
            format!("termIndexInterval={term_index_interval}"),
        ),
        PostingsFormat::VarGapDocFreqInterval {
            term_index_interval,
            doc_freq_threshold,
        } => (
            "LuceneVarGapDocFreqInterval",
            format!(
                "termIndexInterval={term_index_interval},docFreqThreshold={doc_freq_threshold}"
            ),
        ),
        PostingsFormat::Asserting => ("AssertingPostingsFormat", "-".into()),
    }
}

#[test]
fn actual_java_constructor_and_every_retained_instance_route_match() {
    for (nightly, multiplier, fixture) in FIXTURES {
        let mut codec = None::<RandomCodec>;
        let mut random = JavaRandom::new(0);
        let mut mock_seed = 0;
        let mut cases = 0;
        let mut routes = 0;
        let mut checked_mock_seeds = 0;
        let mut format_rows = 0;
        let mut points_rows = 0;
        let mut identity_pairs = 0;
        for row in fixture.lines() {
            let cells: Vec<_> = row.split('\t').collect();
            if cells[0] == "min-abs" {
                break;
            } // Controlled injection is replayed in module units.
            match cells[0] {
                "state" => {
                    if let Some(previous) = codec.as_ref() {
                        assert!(previous.postings_mapping_count() <= 16);
                        assert!(previous.doc_values_mapping_count() <= 16);
                    }
                    let (seed, label) = cells[1].split_once(':').unwrap();
                    random = JavaRandom::new(seed.parse().unwrap());
                    let avoid = avoidance(label);
                    let actual = RandomCodec::with_avoid(
                        &mut random,
                        nightly,
                        multiplier,
                        Some(Rc::clone(&avoid)),
                    )
                    .unwrap();
                    assert_eq!(
                        actual.per_field_seed(),
                        cells[2].parse::<i32>().unwrap(),
                        "{row}"
                    );
                    let points = actual.points_format();
                    assert_eq!(
                        points.max_points_in_leaf_node,
                        cells[3].parse::<i32>().unwrap(),
                        "{row}"
                    );
                    assert_eq!(
                        points.max_mb_sort_in_heap.to_bits(),
                        cells[4].parse::<u64>().unwrap(),
                        "{row}"
                    );
                    assert_eq!(
                        points.bkd_split_random_seed,
                        cells[5].parse::<i32>().unwrap(),
                        "{row}"
                    );
                    assert_eq!(random.next_int(), cells[6].parse::<i32>().unwrap(), "{row}");
                    assert_eq!(
                        Rc::ptr_eq(actual.avoid_codecs(), &avoid).to_string(),
                        cells[7]
                    );
                    for pair in
                        actual
                            .doc_values_formats()
                            .iter()
                            .enumerate()
                            .flat_map(|(at, first)| {
                                actual.doc_values_formats()[at + 1..]
                                    .iter()
                                    .map(move |second| (first, second))
                            })
                    {
                        assert!(!Rc::ptr_eq(pair.0, pair.1));
                        if pair.0 == pair.1 {
                            identity_pairs += 1;
                        }
                    }
                    codec = Some(actual);
                    cases += 1;
                }
                "mock-seed" => {
                    mock_seed = cells[2].parse::<i64>().unwrap();
                }
                "names" => {
                    let actual = codec.as_ref().unwrap();
                    assert_eq!(name_set(actual.format_names()), cells[2], "{row}");
                    assert_eq!(
                        name_set(actual.doc_values_format_names()),
                        cells[3],
                        "{row}"
                    );
                }
                "format" => {
                    let actual = codec.as_ref().unwrap();
                    let ordinal = cells[3].parse::<usize>().unwrap();
                    if cells[2] == "true" {
                        let format = &actual.doc_values_formats()[ordinal];
                        assert_eq!(format.name(), cells[4], "{row}");
                        let (class, settings) = match format.as_ref() {
                            DocValuesFormat::Lucene90 {
                                skip_index_interval_size,
                            } => (
                                "Lucene90DocValuesFormat",
                                format!("skipIndexIntervalSize={skip_index_interval_size}"),
                            ),
                            DocValuesFormat::Asserting => ("AssertingDocValuesFormat", "-".into()),
                        };
                        assert_eq!(class, cells[5], "{row}");
                        assert_eq!(settings, cells[6], "{row}");
                    } else {
                        let format = &actual.postings_formats()[ordinal];
                        assert_eq!(format.name(), cells[4], "{row}");
                        let (class, settings) = postings_settings(format);
                        assert_eq!(class, cells[5], "{row}");
                        assert_eq!(settings, cells[6], "{row}");
                        if let PostingsFormat::MockRandom { seed } = format.as_ref() {
                            assert_eq!(*seed, mock_seed, "{row}");
                            checked_mock_seeds += 1;
                        }
                    }
                    format_rows += 1;
                }
                "route" | "route-error" => {
                    let actual = codec.as_mut().unwrap();
                    let field = units(cells[3]);
                    if cells[0] == "route" {
                        assert_eq!(
                            java_string_hash(&field),
                            cells[4].parse::<i32>().unwrap(),
                            "{row}"
                        );
                    }
                    if cells[2] == "true" {
                        let selected = actual.doc_values_format_for_field(Some(&field));
                        if cells[0] == "route-error" {
                            assert_eq!(selected.unwrap_err().java_class(), cells[4], "{row}");
                        } else {
                            let selected = selected.unwrap();
                            let ordinal = cells[5].parse::<usize>().unwrap();
                            assert!(
                                Rc::ptr_eq(&selected, &actual.doc_values_formats()[ordinal]),
                                "{row}"
                            );
                            assert!(Rc::ptr_eq(
                                &selected,
                                &actual.doc_values_format_for_field(Some(&field)).unwrap()
                            ));
                        }
                    } else {
                        let selected = actual.postings_format_for_field(Some(&field));
                        if cells[0] == "route-error" {
                            assert_eq!(selected.unwrap_err().java_class(), cells[4], "{row}");
                        } else {
                            let selected = selected.unwrap();
                            let ordinal = cells[5].parse::<usize>().unwrap();
                            assert!(
                                Rc::ptr_eq(&selected, &actual.postings_formats()[ordinal]),
                                "{row}"
                            );
                            assert!(Rc::ptr_eq(
                                &selected,
                                &actual.postings_format_for_field(Some(&field)).unwrap()
                            ));
                        }
                    }
                    routes += 1;
                }
                "null" => {
                    let actual = codec.as_mut().unwrap();
                    let error = if cells[2] == "true" {
                        actual.doc_values_format_for_field(None).unwrap_err()
                    } else {
                        actual.postings_format_for_field(None).unwrap_err()
                    };
                    assert_eq!(error.java_class(), cells[3], "{row}");
                }
                "tail" => {
                    assert_eq!(random.next_int(), cells[2].parse::<i32>().unwrap(), "{row}");
                    assert_eq!(
                        random.next_long(),
                        cells[3].parse::<i64>().unwrap(),
                        "{row}"
                    );
                }
                "points-fresh" => {
                    let actual = codec.as_ref().unwrap();
                    let first = actual.points_format();
                    let second = actual.points_format();
                    assert_eq!(first, second);
                    assert_eq!(
                        (!std::ptr::eq(first.as_ref(), second.as_ref())).to_string(),
                        cells[2]
                    );
                    points_rows += 1;
                }
                _ => panic!("unknown actual Java row: {row}"),
            }
        }
        assert_eq!(cases, 315);
        assert_eq!(routes, cases * 32);
        assert_eq!(points_rows, cases);
        assert!(checked_mock_seeds > 50);
        assert!(format_rows > 1800);
        assert!(identity_pairs > 250);
    }
}
