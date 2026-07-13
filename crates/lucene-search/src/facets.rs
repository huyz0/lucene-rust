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

use lucene_codecs::doc_values::{self, SortedNumericEntry};
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
}
