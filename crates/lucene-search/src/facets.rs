//! Basic faceting over a SORTED_SET doc-values field — a simplified port of
//! real Lucene's `SortedSetDocValuesFacetCounts` (`lucene-facet` module):
//! for every matching doc, increment a per-ordinal counter for each of that
//! doc's SortedSet ordinals, then resolve ordinals back to their string
//! labels via the field's terms dictionary (`lookupOrd`-equivalent, see
//! [`lucene_codecs::terms_dict::decode_all_terms`]).
//!
//! ## Scope decisions
//!
//! **Single-segment only.** Real Lucene's faceting (`FacetsCollector` +
//! `SortedSetDocValuesFacetCounts`) is index-wide: it fans out over every
//! segment of a `DirectoryReader` and sums per-ordinal counts across all of
//! them (ordinals are already globally consistent for a single field's
//! terms dictionary in real Lucene's `SortedSetDocValuesReaderState`, which
//! builds one merged ordinal map up front). This port has no such merged
//! ordinal map — each segment's SORTED_SET terms dictionary assigns its own
//! ordinals independently (see `doc_values.rs`/`terms_dict.rs`), so summing
//! raw ordinal counts across segments would silently conflate unrelated
//! terms that happen to share an ordinal number in different segments. This
//! module therefore counts one already-opened segment's doc values only;
//! callers doing multi-segment faceting must count each segment separately
//! and merge counts **by resolved string label**, not by raw ordinal,
//! before combining. This is a straightforward extension once per-segment
//! counting works (group [`facet_counts`]'s output across segments by the
//! label each ordinal resolves to via [`resolve_labels`], summing counts
//! for matching labels) but isn't implemented here, matching this task's
//! honestly-scoped brief.
//!
//! **Query-scoped counting is primary; "count everything" is the caller's
//! trivial special case.** Real Lucene's `FacetsCollector` always counts
//! over a matched-query doc set (there's no separate "count the whole
//! index" API distinct from running `MatchAllDocsQuery`). [`facet_counts`]
//! takes an explicit matching-doc-ID slice; a caller wanting "count every
//! doc in the segment" passes `0..max_doc` (every live doc ID) as that
//! slice — no separate code path is needed or added.

use lucene_codecs::doc_values::{self, NumericEntry, SortedNumericEntry};
use lucene_codecs::terms_dict::{self, TermsDictEntry};

use crate::Result;

/// A single facet's ordinal, resolved label, and count — [`top_n_facets`]'s
/// element type, and also a convenient return shape for [`facet_counts`]
/// once resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FacetCount {
    pub ord: i64,
    pub label: String,
    pub count: u64,
}

/// Counts, for every ordinal in a SORTED_SET field's terms dictionary, how
/// many of `matching_docs` have that ordinal among their values. Multi-valued
/// docs increment a counter for *every* one of their ordinals (real Lucene's
/// `SortedSetDocValuesFacetCounts.count` semantics), not just a "primary"
/// one. A doc not present in `matching_docs` contributes nothing. An empty
/// `matching_docs` produces a result with every ordinal present at count 0
/// (not an empty `Vec`) — this is the cleaner of the two options the task
/// brief allows: it keeps the result's ordinal set always equal to "every
/// term in the dictionary" regardless of how many docs matched, so callers
/// can always resolve labels for the full term set rather than special-
/// casing an empty match set.
///
/// `doc_values_data` is the segment's whole `.dvd` file, exactly as every
/// other function in [`crate::doc_value_query`] takes it. Returns counts
/// indexed by ordinal (`result[ord] == count for ordinal ord`).
pub fn facet_counts(
    doc_values_data: &[u8],
    entry: &SortedNumericEntry,
    terms: &TermsDictEntry,
    matching_docs: &[i32],
) -> Result<Vec<u64>> {
    let mut counts = vec![0u64; terms.terms_dict_size as usize];
    for &doc_id in matching_docs {
        let ords = doc_values::sorted_numeric_values(doc_values_data, entry, doc_id)?;
        for ord in ords {
            if let Some(slot) = counts.get_mut(ord as usize) {
                *slot += 1;
            }
        }
    }
    Ok(counts)
}

/// Resolves every ordinal's count (as returned by [`facet_counts`]) to its
/// string label via the field's terms dictionary, in ordinal order.
pub fn resolve_labels(
    doc_values_data: &[u8],
    terms: &TermsDictEntry,
    counts: &[u64],
) -> Result<Vec<FacetCount>> {
    let labels = terms_dict::decode_all_terms(doc_values_data, terms)
        .map_err(lucene_codecs::doc_values::Error::from)?;
    Ok(labels
        .into_iter()
        .zip(counts.iter().copied())
        .enumerate()
        .map(|(ord, (label_bytes, count))| FacetCount {
            ord: ord as i64,
            label: String::from_utf8_lossy(&label_bytes).into_owned(),
            count,
        })
        .collect())
}

/// Sorts `facets` descending by count, ties broken by ascending ordinal (this
/// crate's existing tie-break convention for deterministic output — see
/// [`crate::collector::TopDocsCollector`]'s "lower doc ID wins" rule, applied
/// here to "lower ordinal wins"), then truncates to at most `n` — real
/// Lucene's `Facets.getTopChildren` convention (`FacetResult.labelValues`,
/// descending count).
pub fn top_n_facets(mut facets: Vec<FacetCount>, n: usize) -> Vec<FacetCount> {
    facets.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.ord.cmp(&b.ord)));
    facets.truncate(n);
    facets
}

/// A single caller-defined numeric bucket for [`range_facet_counts`] — a
/// simplified port of real Lucene's `LongRange`/`DoubleRange`
/// (`lucene-facet` module): a `[min, max]` interval with each end
/// independently inclusive or exclusive, plus the label the bucket is
/// reported under. Values are `i64` here (the NUMERIC doc-values field's raw
/// stored representation); a `DoubleRange`-equivalent caller converts its
/// `f64` bounds to the field's `NumericUtils.doubleToSortableLong`-equivalent
/// `i64` encoding before constructing one of these, same as
/// [`crate::doc_value_query`]'s numeric functions already assume for any
/// non-integer numeric field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumericRange {
    pub label: String,
    pub min: i64,
    pub min_inclusive: bool,
    pub max: i64,
    pub max_inclusive: bool,
}

impl NumericRange {
    /// Whether `value` falls inside this range, honoring each bound's own
    /// inclusive/exclusive flag independently.
    fn contains(&self, value: i64) -> bool {
        let above_min = if self.min_inclusive {
            value >= self.min
        } else {
            value > self.min
        };
        let below_max = if self.max_inclusive {
            value <= self.max
        } else {
            value < self.max
        };
        above_min && below_max
    }
}

/// Counts, for every range in `ranges`, how many of `matching_docs` have a
/// NUMERIC doc-value falling inside it — a simplified port of real Lucene's
/// `LongRangeFacetCounts`/`DoubleRangeFacetCounts.count`.
///
/// **Ranges are caller-defined and may overlap** (real Lucene doesn't
/// require ranges to partition the value space): a doc whose value matches
/// two or more ranges is counted in *each* one, independently — this
/// function makes one pass per range per doc's already-decoded value, with
/// no notion of "the" bucket a doc belongs to.
///
/// **A doc with no value for the field** (`numeric_value` returns `None` —
/// legitimate for a sparse NUMERIC field, see [`crate::doc_value_query`]'s
/// module doc) contributes to **no** range, including an unbounded-looking
/// one like `i64::MIN..=i64::MAX` — the same missing-value rule
/// [`crate::doc_value_query::search_numeric_range`] already documents,
/// applied here per-range instead of to a single range.
///
/// A doc not present in `matching_docs` contributes nothing. An empty
/// `matching_docs` produces every range present at count 0 (not an empty
/// `Vec`) — the same convention [`facet_counts`] documents for an empty
/// match set, kept consistent here.
///
/// Returns `(label, count)` pairs in the **same order** as `ranges` — real
/// Lucene's `FacetResult.labelValues` preserves caller-specified range
/// order rather than sorting by count (unlike [`top_n_facets`], which is a
/// distinct, opt-in sort for the string-facet case).
pub fn range_facet_counts(
    doc_values_data: &[u8],
    entry: &NumericEntry,
    ranges: &[NumericRange],
    matching_docs: &[i32],
) -> Result<Vec<(String, u64)>> {
    let mut counts = vec![0u64; ranges.len()];
    for &doc_id in matching_docs {
        if let Some(value) = doc_values::numeric_value(doc_values_data, entry, doc_id)? {
            for (range, count) in ranges.iter().zip(counts.iter_mut()) {
                if range.contains(value) {
                    *count += 1;
                }
            }
        }
    }
    Ok(ranges
        .iter()
        .zip(counts)
        .map(|(range, count)| (range.label.clone(), count))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::doc_values::{DocValuesMeta, SortedSetKind};

    fn multi_dv_dir() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/multi_valued_dv_index/"
        )
        .to_string()
    }

    struct Manifest {
        kv: Vec<(String, String)>,
    }

    impl Manifest {
        fn load(dir: &str) -> Self {
            let text = std::fs::read_to_string(format!("{dir}manifest.properties"))
                .expect("run fixtures generator first");
            let kv = text
                .lines()
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            Manifest { kv }
        }

        fn get(&self, key: &str) -> &str {
            self.kv
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_str())
                .unwrap_or_else(|| panic!("manifest key {key} missing"))
        }
    }

    fn id_from_hex(hex: &str) -> [u8; 16] {
        let mut id = [0u8; 16];
        for i in 0..16 {
            id[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        id
    }

    fn dv_suffix(manifest: &Manifest) -> String {
        let segment_name = manifest.get("segment_name");
        let name = manifest.get("dvm_file_name");
        name.strip_prefix(&format!("{segment_name}_"))
            .and_then(|s| s.strip_suffix(".dvm"))
            .unwrap()
            .to_string()
    }

    fn field_number(manifest: &Manifest, field: &str) -> i32 {
        manifest
            .get("field_numbers")
            .split(',')
            .find_map(|kv| {
                let (name, num) = kv.split_once(':').unwrap();
                (name == field).then(|| num.parse().unwrap())
            })
            .unwrap_or_else(|| panic!("field {field} missing from field_numbers"))
    }

    fn load_dv_meta(dir: &str) -> (Manifest, Vec<u8>, DocValuesMeta) {
        let manifest = Manifest::load(dir);
        let id = id_from_hex(manifest.get("id_hex"));
        let fnm = std::fs::read(format!("{dir}{}.raw", manifest.get("fnm_file_name"))).unwrap();
        let fis = lucene_codecs::field_infos::parse(&fnm, &id, "").unwrap();
        let meta_buf =
            std::fs::read(format!("{dir}{}.raw", manifest.get("dvm_file_name"))).unwrap();
        let data_buf =
            std::fs::read(format!("{dir}{}.raw", manifest.get("dvd_file_name"))).unwrap();
        let suffix = dv_suffix(&manifest);
        let (_, parsed) = doc_values::parse_meta(&meta_buf, &id, &suffix, &fis).unwrap();
        (manifest, data_buf, parsed)
    }

    /// Ground truth taken directly from the manifest, which real Lucene wrote
    /// via a straightforward per-doc `SortedSetDocValues.nextOrd()` iteration
    /// (see `GenMultiValuedDocValues.java`'s `field.tags.ords`/`.terms`
    /// output) -- an honest differential check without depending on the
    /// `lucene-facet` module (not a project dependency), per this task's
    /// brief: "a straightforward manual per-doc iteration ... is a
    /// reasonable, real ground truth even without the facet module
    /// specifically."
    fn expected_counts_from_manifest(manifest: &Manifest, num_terms: usize) -> Vec<u64> {
        let mut counts = vec![0u64; num_terms];
        for doc_ords in manifest.get("field.tags.ords").split(';') {
            if doc_ords == "NONE" {
                continue;
            }
            for ord in doc_ords.split(',') {
                counts[ord.parse::<usize>().unwrap()] += 1;
            }
        }
        counts
    }

    fn tags_entry(
        meta: &DocValuesMeta,
        field_number: i32,
    ) -> (&SortedNumericEntry, &TermsDictEntry) {
        let entry = meta.sorted_set_entry(field_number).unwrap();
        match &entry.kind {
            SortedSetKind::Multi { ords, terms } => (ords, terms),
            SortedSetKind::Single(_) => panic!("expected a multi-valued SORTED_SET"),
        }
    }

    #[test]
    fn facet_counts_match_real_lucene_ground_truth() {
        let (manifest, data, meta) = load_dv_meta(&multi_dv_dir());
        let field_num = field_number(&manifest, "tags");
        let (ords, terms) = tags_entry(&meta, field_num);

        // All 5 docs matching -- doc0=[red,blue], doc1=NONE, doc2=[green],
        // doc3=[blue], doc4=[red,green].
        let matching: Vec<i32> = (0..5).collect();
        let counts = facet_counts(&data, ords, terms, &matching).unwrap();

        let expected = expected_counts_from_manifest(&manifest, terms.terms_dict_size as usize);
        assert_eq!(counts, expected);

        // Cross-check against the manifest's resolved term strings too, not
        // just raw ordinal counts, so a label/ordinal mismatch would also
        // fail this test.
        let resolved = resolve_labels(&data, terms, &counts).unwrap();
        let expected_terms: Vec<&str> = manifest.get("field.tags.terms").split(',').collect();
        assert_eq!(resolved.len(), expected_terms.len());
        for (fc, expected_label) in resolved.iter().zip(expected_terms.iter()) {
            assert_eq!(&fc.label, expected_label);
        }
    }

    #[test]
    fn multi_valued_doc_increments_every_ordinal_not_just_first() {
        let (manifest, data, meta) = load_dv_meta(&multi_dv_dir());
        let field_num = field_number(&manifest, "tags");
        let (ords, terms) = tags_entry(&meta, field_num);

        // doc0 alone has ords [red, blue] (two ordinals) -- both must be
        // incremented, not just the first ("primary") one.
        let counts = facet_counts(&data, ords, terms, &[0]).unwrap();
        assert_eq!(counts.iter().sum::<u64>(), 2);

        let resolved = resolve_labels(&data, terms, &counts).unwrap();
        let red = resolved.iter().find(|f| f.label == "red").unwrap();
        let blue = resolved.iter().find(|f| f.label == "blue").unwrap();
        assert_eq!(red.count, 1);
        assert_eq!(blue.count, 1);
        assert!(resolved
            .iter()
            .all(|f| f.label == "red" || f.label == "blue" || f.count == 0));
    }

    #[test]
    fn doc_not_in_matching_set_contributes_nothing() {
        let (manifest, data, meta) = load_dv_meta(&multi_dv_dir());
        let field_num = field_number(&manifest, "tags");
        let (ords, terms) = tags_entry(&meta, field_num);

        // Every doc except doc4 ([red, green]) -- doc4's ordinals must not
        // show up.
        let matching: Vec<i32> = vec![0, 1, 2, 3];
        let counts = facet_counts(&data, ords, terms, &matching).unwrap();
        let expected = expected_counts_from_manifest(&manifest, terms.terms_dict_size as usize);
        // green appears in doc2 and doc4 in the full set; excluding doc4
        // should drop green's count by exactly one relative to full-set
        // expected.
        let resolved = resolve_labels(&data, terms, &counts).unwrap();
        let resolved_full = resolve_labels(&data, terms, &expected).unwrap();
        let green = resolved.iter().find(|f| f.label == "green").unwrap();
        let green_full = resolved_full.iter().find(|f| f.label == "green").unwrap();
        assert_eq!(green.count, green_full.count - 1);
    }

    #[test]
    fn empty_matching_set_yields_all_zero_counts() {
        let (manifest, data, meta) = load_dv_meta(&multi_dv_dir());
        let field_num = field_number(&manifest, "tags");
        let (ords, terms) = tags_entry(&meta, field_num);

        let counts = facet_counts(&data, ords, terms, &[]).unwrap();
        assert_eq!(counts.len(), terms.terms_dict_size as usize);
        assert!(counts.iter().all(|&c| c == 0));
    }

    #[test]
    fn top_n_facets_sorts_descending_and_truncates() {
        let facets = vec![
            FacetCount {
                ord: 0,
                label: "a".into(),
                count: 3,
            },
            FacetCount {
                ord: 1,
                label: "b".into(),
                count: 7,
            },
            FacetCount {
                ord: 2,
                label: "c".into(),
                count: 5,
            },
        ];
        let top2 = top_n_facets(facets, 2);
        assert_eq!(
            top2.iter().map(|f| f.label.as_str()).collect::<Vec<_>>(),
            vec!["b", "c"]
        );
    }

    #[test]
    fn top_n_facets_ties_broken_by_ascending_ordinal() {
        let facets = vec![
            FacetCount {
                ord: 2,
                label: "z".into(),
                count: 4,
            },
            FacetCount {
                ord: 0,
                label: "x".into(),
                count: 4,
            },
            FacetCount {
                ord: 1,
                label: "y".into(),
                count: 4,
            },
        ];
        let top = top_n_facets(facets, 3);
        assert_eq!(top.iter().map(|f| f.ord).collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn top_n_facets_n_larger_than_available_returns_all() {
        let facets = vec![FacetCount {
            ord: 0,
            label: "a".into(),
            count: 1,
        }];
        let top = top_n_facets(facets, 100);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn top_n_facets_empty_input_yields_empty_output() {
        let top = top_n_facets(Vec::new(), 5);
        assert!(top.is_empty());
    }

    #[test]
    fn facet_counts_propagates_decode_errors() {
        let mut numeric = doc_values::NumericEntry {
            field_number: 0,
            docs_with_field_offset: -1,
            docs_with_field_length: 0,
            jump_table_entry_count: -1,
            dense_rank_power: 0xFF,
            num_values: 1,
            table: None,
            bits_per_value: 8,
            min_value: 0,
            gcd: 1,
            values_offset: 0,
            values_length: 1,
        };
        numeric.bits_per_value = 8;
        let entry = SortedNumericEntry {
            field_number: 0,
            numeric,
            num_docs_with_field: 1,
            addresses: None,
        };
        let terms = TermsDictEntry {
            terms_dict_size: 1,
            max_term_length: 8,
            terms_data_offset: 0,
            terms_data_length: 0,
        };
        let err = facet_counts(&[], &entry, &terms, &[0]).unwrap_err();
        assert!(matches!(err, crate::Error::DocValues(_)));
    }

    // --- range_facet_counts ---
    //
    // Reuses `doc_values_index`'s `varying` field (task #21/#31's own
    // fixture -- see `doc_value_query.rs`'s tests), whose 5 docs' real-Lucene-
    // recorded values are already differentially verified there: -100, 7, 42,
    // 1000, -3 for docs 0..4. Bucket assignment is hand-verified against
    // those recorded values rather than re-deriving decode correctness.

    fn dv_dir() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/doc_values_index/"
        )
        .to_string()
    }

    fn dv_meta_and_data(dir: &str) -> (Manifest, Vec<u8>, DocValuesMeta) {
        let (manifest, data, meta) = load_dv_meta(dir);
        (manifest, data, meta)
    }

    fn field_num(manifest: &Manifest, field: &str) -> i32 {
        manifest
            .get("field_numbers")
            .split(',')
            .find_map(|kv| {
                let (name, num) = kv.split_once(':').unwrap();
                (name == field).then(|| num.parse().unwrap())
            })
            .unwrap_or_else(|| panic!("field {field} missing from field_numbers"))
    }

    #[test]
    fn range_facet_counts_partitions_non_overlapping_ranges() {
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "varying")).unwrap();
        // values: -100, 7, 42, 1000, -3 for docs 0..4.
        let ranges = vec![
            NumericRange {
                label: "negative".into(),
                min: i64::MIN,
                min_inclusive: true,
                max: 0,
                max_inclusive: false,
            },
            NumericRange {
                label: "small_positive".into(),
                min: 0,
                min_inclusive: true,
                max: 100,
                max_inclusive: true,
            },
            NumericRange {
                label: "large".into(),
                min: 100,
                min_inclusive: false,
                max: i64::MAX,
                max_inclusive: true,
            },
        ];
        let matching: Vec<i32> = (0..5).collect();
        let counts = range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        assert_eq!(
            counts,
            vec![
                ("negative".to_string(), 2),       // -100, -3
                ("small_positive".to_string(), 2), // 7, 42
                ("large".to_string(), 1),          // 1000
            ]
        );
    }

    #[test]
    fn range_facet_counts_overlapping_ranges_count_doc_in_both() {
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "varying")).unwrap();
        // doc2's value (42) falls in both overlapping ranges below.
        let ranges = vec![
            NumericRange {
                label: "0-50".into(),
                min: 0,
                min_inclusive: true,
                max: 50,
                max_inclusive: true,
            },
            NumericRange {
                label: "10-1000".into(),
                min: 10,
                min_inclusive: true,
                max: 1000,
                max_inclusive: true,
            },
        ];
        let matching: Vec<i32> = (0..5).collect();
        let counts = range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        // "0-50": 7, 42 -> 2. "10-1000": 42, 1000 -> 2. doc2 (42) counted in both.
        assert_eq!(
            counts,
            vec![("0-50".to_string(), 2), ("10-1000".to_string(), 2)]
        );
    }

    #[test]
    fn range_facet_counts_boundary_inclusive_inclusive() {
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "varying")).unwrap();
        // [42, 42] inclusive-inclusive: only doc2 (value 42) matches.
        let ranges = vec![NumericRange {
            label: "exact".into(),
            min: 42,
            min_inclusive: true,
            max: 42,
            max_inclusive: true,
        }];
        let matching: Vec<i32> = (0..5).collect();
        let counts = range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        assert_eq!(counts, vec![("exact".to_string(), 1)]);
    }

    #[test]
    fn range_facet_counts_boundary_inclusive_exclusive() {
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "varying")).unwrap();
        // [42, 42) inclusive-exclusive: the max bound equal to the value
        // itself must exclude it.
        let ranges = vec![NumericRange {
            label: "r".into(),
            min: 42,
            min_inclusive: true,
            max: 42,
            max_inclusive: false,
        }];
        let matching: Vec<i32> = (0..5).collect();
        let counts = range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        assert_eq!(counts, vec![("r".to_string(), 0)]);
    }

    #[test]
    fn range_facet_counts_boundary_exclusive_inclusive() {
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "varying")).unwrap();
        // (42, 42] exclusive-inclusive: the min bound equal to the value
        // itself must exclude it.
        let ranges = vec![NumericRange {
            label: "r".into(),
            min: 42,
            min_inclusive: false,
            max: 42,
            max_inclusive: true,
        }];
        let matching: Vec<i32> = (0..5).collect();
        let counts = range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        assert_eq!(counts, vec![("r".to_string(), 0)]);
    }

    #[test]
    fn range_facet_counts_boundary_exclusive_exclusive() {
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "varying")).unwrap();
        // (7, 1000) exclusive-exclusive: doc2 (42) still matches, but the
        // endpoints 7 and 1000 themselves (docs 1 and 3) must not.
        let ranges = vec![NumericRange {
            label: "r".into(),
            min: 7,
            min_inclusive: false,
            max: 1000,
            max_inclusive: false,
        }];
        let matching: Vec<i32> = (0..5).collect();
        let counts = range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        assert_eq!(counts, vec![("r".to_string(), 1)]);
    }

    #[test]
    fn range_facet_counts_missing_value_never_counted_even_unbounded() {
        // `sparse` field: 5, NONE, 15, NONE, 25 -- docs 1 and 3 have no value.
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "sparse")).unwrap();
        // An unbounded-above range that would catch every value if a missing
        // doc wrongly counted.
        let ranges = vec![NumericRange {
            label: "everything".into(),
            min: i64::MIN,
            min_inclusive: true,
            max: i64::MAX,
            max_inclusive: true,
        }];
        let matching: Vec<i32> = (0..5).collect();
        let counts = range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        // Only docs 0, 2, 4 (5, 15, 25) have a value.
        assert_eq!(counts, vec![("everything".to_string(), 3)]);
    }

    #[test]
    fn range_facet_counts_doc_not_in_matching_set_contributes_nothing() {
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "varying")).unwrap();
        // Excludes doc3 (value 1000); an unbounded range would otherwise
        // count it.
        let ranges = vec![NumericRange {
            label: "everything".into(),
            min: i64::MIN,
            min_inclusive: true,
            max: i64::MAX,
            max_inclusive: true,
        }];
        let matching: Vec<i32> = vec![0, 1, 2, 4];
        let counts = range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        assert_eq!(counts, vec![("everything".to_string(), 4)]);
    }

    #[test]
    fn range_facet_counts_empty_matching_set_yields_all_zero_counts() {
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "varying")).unwrap();
        let ranges = vec![
            NumericRange {
                label: "a".into(),
                min: 0,
                min_inclusive: true,
                max: 10,
                max_inclusive: true,
            },
            NumericRange {
                label: "b".into(),
                min: 10,
                min_inclusive: false,
                max: 100,
                max_inclusive: true,
            },
        ];
        let counts = range_facet_counts(&data, entry, &ranges, &[]).unwrap();
        assert_eq!(counts, vec![("a".to_string(), 0), ("b".to_string(), 0)]);
    }

    #[test]
    fn range_facet_counts_preserves_caller_specified_range_order() {
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "varying")).unwrap();
        // Deliberately out of value order -- output must mirror input order,
        // not sort by count.
        let ranges = vec![
            NumericRange {
                label: "large".into(),
                min: 100,
                min_inclusive: false,
                max: i64::MAX,
                max_inclusive: true,
            },
            NumericRange {
                label: "negative".into(),
                min: i64::MIN,
                min_inclusive: true,
                max: 0,
                max_inclusive: false,
            },
        ];
        let matching: Vec<i32> = (0..5).collect();
        let counts = range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        assert_eq!(
            counts.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>(),
            vec!["large", "negative"]
        );
        assert_eq!(counts[0].1, 1); // large: 1000
        assert_eq!(counts[1].1, 2); // negative: -100, -3
    }

    #[test]
    fn range_facet_counts_propagates_decode_errors() {
        let mut entry = doc_values::NumericEntry {
            field_number: 0,
            docs_with_field_offset: -1,
            docs_with_field_length: 0,
            jump_table_entry_count: -1,
            dense_rank_power: 0xFF,
            num_values: 1,
            table: None,
            bits_per_value: 8,
            min_value: 0,
            gcd: 1,
            values_offset: 0,
            values_length: 1,
        };
        entry.bits_per_value = 8;
        let ranges = vec![NumericRange {
            label: "r".into(),
            min: 0,
            min_inclusive: true,
            max: 100,
            max_inclusive: true,
        }];
        let err = range_facet_counts(&[], &entry, &ranges, &[0]).unwrap_err();
        assert!(matches!(err, crate::Error::DocValues(_)));
    }
}
