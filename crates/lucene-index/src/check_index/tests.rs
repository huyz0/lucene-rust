#![cfg(test)]
// Test code opts out of the arithmetic gate at the module boundary: the
// gate exists for values read off disk, not for a test's own index
// arithmetic. See `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::FsDirectory;

fn fixture_dir(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(format!(
        "{}/../../fixtures/data/{name}/",
        env!("CARGO_MANIFEST_DIR")
    ))
}

/// A scratch directory that actually goes away: `/tmp` can be a tmpfs, i.e.
/// RAM, so leaked directories from repeated runs would eat memory. See
/// [`lucene_util::test_support`].
fn tempdir() -> lucene_util::test_support::TempDir {
    lucene_util::test_support::TempDir::new("check-index")
}

/// A genuinely valid, real-Lucene-written fixture must pass every check
/// cleanly -- the baseline "no false positives" test.
#[test]
fn valid_blocktree_fixture_passes_every_check() {
    let dir = FsDirectory::open(fixture_dir("blocktree_index"));
    let results = check_directory(&dir).expect("read segments_N");
    // results[0] is the commit-level result; results[1..] are the segments.
    assert_eq!(results.len(), 2);
    let result = &results[1];
    assert!(
        result.all_passed(),
        "unexpected failures: {:?}",
        result.failures()
    );
    // Sanity: real checks actually ran, this isn't a vacuous pass.
    assert!(result.checks.len() >= 5);
}

/// A fixture with real deletions must pass the `.liv`-specific checks
/// (max_doc match, cardinality vs del_count) on genuinely valid data.
#[test]
fn valid_live_docs_fixture_passes_every_check() {
    let dir = FsDirectory::open(fixture_dir("live_docs_index"));
    let results = check_directory(&dir).expect("read segments_N");
    assert_eq!(results.len(), 2);
    assert!(
        results[1].all_passed(),
        "unexpected failures: {:?}",
        results[1].failures()
    );
    assert!(results[1]
        .checks
        .iter()
        .any(|c| c.name == "liv.cardinality_matches_del_count"));
}

/// Every real-Lucene index fixture in the tree must pass every check
/// cleanly. This is the no-false-positives baseline for the checks added
/// beyond the original scope (doc values, term vectors, postings field
/// summaries, commit-level invariants) -- each of these fixtures was
/// written by a real `IndexWriter`, so any failure is ours.
#[test]
fn every_real_lucene_index_fixture_passes_every_check() {
    for name in [
        "blocktree_index",
        "live_docs_index",
        "doc_values_index",
        "sorted_dv_index",
        "multi_valued_dv_index",
        "term_vectors_index",
        "norms_index",
        "points_index",
        "doc_values_skip_index",
        "doc_values_varying_bpv",
        "vectors_index",
        // A real Java-written index that declares an **index sort**, so
        // `check_index_sort`'s success path is proved against Lucene's
        // own sort-on-flush output and not only against this port's.
        "sorted_index",
    ] {
        let dir = FsDirectory::open(fixture_dir(name));
        let results = check_directory(&dir).unwrap_or_else(|e| panic!("{name}: {e}"));
        for result in &results {
            assert!(
                result.all_passed(),
                "{name} segment {}: unexpected failures: {:?}",
                result.segment_name,
                result.failures()
            );
        }
    }
}

/// The doc-values checks must actually run (and be non-vacuous) on a
/// fixture that has doc values -- a check that silently never fires is
/// worse than no check.
#[test]
fn doc_values_checks_actually_run_on_a_doc_values_fixture() {
    let dir = FsDirectory::open(fixture_dir("doc_values_index"));
    let results = check_directory(&dir).unwrap();
    let names: Vec<&str> = results[1].checks.iter().map(|c| c.name.as_str()).collect();
    assert!(
        names
            .iter()
            .any(|n| n.starts_with("doc_values.values_decode:")),
        "expected doc_values checks, got {names:?}"
    );
}

/// Copies a fixture directory into a fresh temp dir, dropping the
/// `.raw`/manifest side-files the fixture ships for other tests.
fn copy_fixture(name: &str) -> lucene_util::test_support::TempDir {
    let src = fixture_dir(name);
    let dst = tempdir();
    for entry in std::fs::read_dir(&src).unwrap() {
        let entry = entry.unwrap();
        let file_name = entry.file_name();
        let s = file_name.to_string_lossy().to_string();
        if s.ends_with(".raw") || s.contains("manifest") || s.ends_with(".tsv") {
            continue;
        }
        if entry.file_type().unwrap().is_file() {
            std::fs::copy(entry.path(), dst.join(&s)).unwrap();
        }
    }
    dst
}

fn find_file(dir: &std::path::Path, ext: &str) -> std::path::PathBuf {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.to_string_lossy().ends_with(ext))
        .unwrap_or_else(|| panic!("no {ext} file in {dir:?}"))
}

fn failed_names(results: &[CheckResult]) -> Vec<String> {
    results
        .iter()
        .flat_map(|r| r.failures())
        .map(|c| c.name.clone())
        .collect()
}

/// Real Java `CheckIndex -level 3` was run on each of these fixtures and
/// its printed counts recorded here verbatim. `test: terms, freq,
/// prox...OK [N terms; M terms/docs pairs; K tokens]` is
/// `term_count`/`term_doc_pairs`/`token_count`; `test: field
/// norms.........OK [N fields]` is `norm_fields`; `test: term
/// vectors........OK [N total term vector count]` is
/// `term_vector_fields`; `test: vectors.............OK [N fields, M
/// vectors]` is `vector_fields`/`vector_values`.
///
/// A disagreement is itself a finding: it means one side enumerated
/// something the other did not.
#[test]
fn counts_match_real_java_check_index_output() {
    // (fixture, terms, terms/docs pairs, tokens, norm fields,
    //  term vector count, vector fields, vector values)
    struct Expected {
        fixture: &'static str,
        terms: i64,
        term_doc_pairs: i64,
        tokens: i64,
        norm_fields: i64,
        term_vectors: i64,
        vector_fields: i64,
        vector_values: i64,
    }
    let e = |fixture,
             terms,
             term_doc_pairs,
             tokens,
             norm_fields,
             term_vectors,
             vector_fields,
             vector_values| Expected {
        fixture,
        terms,
        term_doc_pairs,
        tokens,
        norm_fields,
        term_vectors,
        vector_fields,
        vector_values,
    };
    let expected = [
        e("blocktree_index", 414, 8968, 13547, 6, 0, 0, 0),
        e("term_vectors_index", 8, 8, 9, 2, 3, 0, 0),
        e("norms_index", 24, 43, 43, 2, 0, 0, 0),
        e("points_index", 2000, 2000, 2000, 0, 0, 0, 0),
        e("vectors_index", 0, 0, 0, 0, 0, 5, 7911),
    ];
    for x in expected {
        let name = x.fixture;
        let dir = FsDirectory::open(fixture_dir(name));
        let results = check_directory(&dir).unwrap_or_else(|e| panic!("{name}: {e}"));
        let s = &results[1].stats;
        let got = [
            s.term_count,
            s.term_doc_pairs,
            s.token_count,
            s.norm_fields,
            s.term_vector_fields,
            s.vector_fields,
            s.vector_values,
        ];
        let want = [
            x.terms,
            x.term_doc_pairs,
            x.tokens,
            x.norm_fields,
            x.term_vectors,
            x.vector_fields,
            x.vector_values,
        ];
        assert_eq!(
            got, want,
            "{name}: counts disagree with real Java CheckIndex"
        );
    }
}

/// The norms values check must actually run on a fixture with norms, and
/// so must the terms-vs-norms cross-check.
#[test]
fn norms_checks_actually_run_on_a_norms_fixture() {
    let dir = FsDirectory::open(fixture_dir("norms_index"));
    let results = check_directory(&dir).unwrap();
    let names: Vec<&str> = results[1].checks.iter().map(|c| c.name.as_str()).collect();
    assert!(
        names.iter().any(|n| n.starts_with("norms.values_decode:")),
        "expected norms value checks, got {names:?}"
    );
    assert!(
        names
            .iter()
            .any(|n| n.starts_with("norms.agree_with_postings:")),
        "expected the terms-vs-norms cross-check, got {names:?}"
    );
}

/// The positional checks must actually run on a fixture whose fields
/// index positions, offsets and payloads.
#[test]
fn positional_checks_actually_run_on_the_blocktree_fixture() {
    let dir = FsDirectory::open(fixture_dir("blocktree_index"));
    let results = check_directory(&dir).unwrap();
    let names: Vec<&str> = results[1].checks.iter().map(|c| c.name.as_str()).collect();
    for prefix in [
        "postings.positions_valid:",
        "postings.seek_agrees:",
        "postings.intersect_agrees:",
        "postings.advance_agrees:",
        "postings.term_stats:",
        "postings.term_dict_shape:",
    ] {
        assert!(
            names.iter().any(|n| n.starts_with(prefix)),
            "expected a {prefix}* check, got {names:?}"
        );
    }
}

/// The vector and HNSW-graph checks must actually run on the real
/// `Lucene99HnswVectorsFormat` fixture.
#[test]
fn vector_checks_actually_run_on_the_vectors_fixture() {
    let dir = FsDirectory::open(fixture_dir("vectors_index"));
    let results = check_directory(&dir).unwrap();
    let names: Vec<&str> = results[1].checks.iter().map(|c| c.name.as_str()).collect();
    for prefix in [
        "vectors.values_decode:",
        "vectors.ord_to_doc:",
        "vectors.field_entry_matches_fnm:",
        "hnsw.neighbors_on_level:",
        "hnsw.neighbors_sorted:",
        "hnsw.entry_point_reachable:",
    ] {
        assert!(
            names.iter().any(|n| n.starts_with(prefix)),
            "expected a {prefix}* check, got {names:?}"
        );
    }
    assert!(
        results[1].all_passed(),
        "unexpected failures: {:?}",
        results[1].failures()
    );
}

/// The term-vectors-vs-postings cross-check must actually run.
#[test]
fn term_vector_postings_cross_check_actually_runs() {
    let dir = FsDirectory::open(fixture_dir("term_vectors_index"));
    let results = check_directory(&dir).unwrap();
    let names: Vec<&str> = results[1].checks.iter().map(|c| c.name.as_str()).collect();
    assert!(names.contains(&"term_vectors.match_postings"));
    assert!(names.contains(&"term_vectors.self_consistent"));
    assert!(
        results[1].all_passed(),
        "unexpected failures: {:?}",
        results[1].failures()
    );
}

/// A hand-built one-field segment whose single term spans `num_docs`
/// documents -- enough for the `.doc` file to carry several full
/// 128-document blocks and therefore real level-0/level-1 skip data,
/// which is what `postings.advance_agrees` cross-checks against the
/// decoded blocks. Small on disk, so a byte-by-byte corruption scan over
/// it is cheap.
fn write_many_doc_postings_fixture(
    dst_dir: &std::path::Path,
    num_docs: i32,
) -> segment_infos::SegmentCommitInfo {
    use lucene_codecs::postings_writer::TermPostings;
    let terms = vec![TermPostings {
        term: b"alpha".to_vec(),
        docs: (0..num_docs).map(|d| (d, 1 + (d % 3))).collect(),
        positions: vec![],
        offsets: vec![],
        payload_bytes: vec![],
        payload_lengths: vec![],
    }];
    write_postings_fixture(dst_dir, &terms, num_docs, num_docs, None)
}

/// Negative control for `postings.advance_agrees`: the `.doc` file
/// carries both the packed doc-id blocks and the level-0/level-1 skip
/// data that indexes them, and this check is the only place the two are
/// compared -- the skip-driven `advance` is required to land on exactly
/// the document a linear scan of the fully decoded block would.
#[test]
fn corrupting_the_doc_skip_data_is_caught_by_the_advance_check() {
    let dst = tempdir();
    let commit = write_many_doc_postings_fixture(&dst, 1000);
    let doc_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.doc"));
    let original = std::fs::read(&doc_path).unwrap();
    let dir = FsDirectory::open(&dst);
    assert!(
        check_segment(&dir, &commit).all_passed(),
        "the fixture must start clean: {:?}",
        check_segment(&dir, &commit).failures()
    );

    // Every corruption is re-signed, so `file:*`'s CRC cannot "catch" it
    // and only the semantic checks can. The previous shape of this test
    // left the footer stale, which made `!failed.is_empty()` trivially
    // true for every flip and reduced the real claim to
    // `caught_by_advance > 0` -- an assertion that survives a regression
    // to one caught byte in three thousand.
    //
    // No `catch_unwind`: a corrupt `.doc` yields a typed `Corrupted` error,
    // never a panic, so every byte flip is checked.
    let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
    let mut caught_by_advance = 0usize;
    let mut caught_by_other = 0usize;
    let mut accepted = 0usize;
    let mut isolated: Option<(usize, u8)> = None;
    for off in 48..body_end {
        for mask in [0x01u8, 0x80, 0xff] {
            let mut bytes = original.clone();
            bytes[off] ^= mask;
            repair_checksum(&mut bytes);
            std::fs::write(&doc_path, &bytes).unwrap();
            let dir = FsDirectory::open(&dst);
            let result = check_segment(&dir, &commit);
            let failed: Vec<&str> = result.failures().iter().map(|c| c.name.as_str()).collect();
            if failed.contains(&"postings.advance_agrees:body") {
                caught_by_advance += 1;
                if failed.len() == 1 {
                    isolated.get_or_insert((off, mask));
                }
            } else if failed.is_empty() {
                accepted += 1;
            } else {
                caught_by_other += 1;
            }
        }
    }
    std::fs::write(&doc_path, &original).unwrap();
    let total = caught_by_advance + caught_by_other + accepted;
    // Measured when this was written, over 2 034 re-signed single-byte
    // `.doc` corruptions of a 1 000-document term: **5** were rejected by
    // `postings.advance_agrees` specifically, 2 026 by another postings
    // check, and **3** were accepted by every check in the module.
    //
    // The second number is the one worth asserting, and it is the one the
    // old `caught_by_advance > 0` shape could not state: with the CRC out
    // of the picture, all but 3 of 2 034 corruptions of a `.doc` are
    // still rejected on their semantics alone.
    assert!(
        accepted <= 3,
        "{accepted} of {total} re-signed .doc corruptions were accepted by every check; \
             3 were when this test was written"
    );
    assert!(
        caught_by_advance >= 4,
        "postings.advance_agrees rejected only {caught_by_advance} of {total} re-signed \
             .doc corruptions ({caught_by_other} caught by another check, {accepted} accepted); \
             5 were rejected when this test was written"
    );
    // No isolation assertion, deliberately, and this is a real property
    // of the check rather than a gap in the test: **none** of the 2 034
    // corruptions tripped `postings.advance_agrees` alone. A `.doc` byte
    // that makes the skip list disagree with the decoded blocks almost
    // always also makes a doc ID non-increasing or out of range, which
    // `postings.doc_ids_valid` sees first. The check is therefore a
    // *second* witness rather than a sole one -- which is still worth
    // having (it is the only place the skip data and the blocks it
    // indexes are compared) but is not falsifiable in isolation by byte
    // corruption. Recorded here so the next reader does not add an
    // isolation assertion that cannot hold.
    assert!(
        isolated.is_none(),
        "a .doc corruption now trips postings.advance_agrees alone (byte {isolated:?}); \
             none did when this test was written -- if that is a real improvement, assert the \
             isolation here instead of this line"
    );

    let dir = FsDirectory::open(&dst);
    assert!(check_segment(&dir, &commit).all_passed());
}

fn impact(freq: i32, norm: i64) -> postings::Impact {
    postings::Impact { freq, norm }
}

/// `checkImpacts`' rules, one failing input each, and inputs that pass.
#[test]
fn check_impacts_rejects_each_malformed_shape() {
    let ok = vec![
        (300, vec![impact(1, 5), impact(3, 7)]),
        (9000, vec![impact(2, 5), impact(4, 9)]),
    ];
    assert_eq!(check_impacts(&ok, 10), Ok(()));
    assert_eq!(check_impacts(&ok[..1], 300), Ok(()));
    let err = |levels: &[(i32, Vec<postings::Impact>)], target| {
        check_impacts(levels, target).unwrap_err()
    };
    assert!(err(&[], 0).contains("must be >= 1"));
    assert!(err(&ok, 301).contains("less than the target"));
    assert!(err(&[(300, vec![impact(1, 1)]), (200, vec![impact(1, 1)])], 0).contains("Decreasing"));
    assert!(err(&[(300, vec![])], 0).contains("empty list"));
    assert!(err(&[(300, vec![impact(0, 1)])], 0).contains("freq <= 0"));
    assert!(err(&[(300, vec![impact(1, 0)])], 0).contains("norm == 0"));
    assert!(err(&[(300, vec![impact(2, 5), impact(2, 7)])], 0).contains("not ordered"));
    assert!(err(&[(300, vec![impact(2, 5), impact(3, 5)])], 0).contains("not ordered"));
    // Norms compare unsigned: a negative norm is the largest.
    assert_eq!(
        check_impacts(&[(300, vec![impact(1, 5), impact(2, -1)])], 0),
        Ok(())
    );
    // Level 1 must dominate level 0: (6, 2) on level 0 scores better
    // than anything level 1 has left.
    assert!(err(
        &[
            (300, vec![impact(5, 1), impact(6, 2)]),
            (9000, vec![impact(1, 1), impact(2, 3)])
        ],
        0
    )
    .contains("triggers a better score"));
}

#[test]
fn shared_impacts_validation_keeps_java_quirks_and_signed_diagnostics() {
    // Java accepts equal level bounds and compares later pairs only to
    // the first one, even if the last two pairs regress.
    assert_eq!(
        check_impacts(
            &[
                (10, vec![impact(1, 1), impact(5, 5), impact(3, 3)]),
                (10, vec![impact(10, 1)]),
            ],
            10,
        ),
        Ok(())
    );
    // Advancing the upper pair consumes the lower pair without checking
    // it again. Both upper pairs here score worse than the lower one.
    assert_eq!(
        check_impacts(
            &[
                (10, vec![impact(10, 1)]),
                (20, vec![impact(1, 1), impact(2, 2)]),
            ],
            0,
        ),
        Ok(())
    );
    assert_eq!(
            check_impacts(
                &[(10, vec![impact(1, 1)]), (9, vec![impact(1, 1)])],
                0,
            )
            .unwrap_err(),
            "Decreasing return for getDocIdUpTo: level 0 returned 10 but level 1 returned 9 for target 0"
        );
    // Norm comparisons are unsigned, but Java renders signed longs in
    // error messages, including the first pair's norm.
    assert_eq!(
        check_impacts(&[(10, vec![impact(1, -1), impact(2, i64::MIN)])], 0).unwrap_err(),
        "Impacts are not ordered or contain dups, got (1,-1) then (2,-9223372036854775808)"
    );
    assert_eq!(
            check_impacts(
                &[
                    (10, vec![impact(10, 1), impact(11, 2)]),
                    (20, vec![impact(1, 1), impact(2, 2)]),
                ],
                0,
            )
            .unwrap_err(),
            "Found impact (11,2) on level 0 but no impact on level 1 triggers a better score: freqs=[1, 2] norms=[1, 2]"
        );
}

/// `postings.impacts` and `postings.doc_id_runs` run on every term of a
/// real `.doc` (1000 documents: three full blocks with level-0 impacts and
/// a tail) and pass; a sweep of re-signed single-bit `.doc` flips shows
/// the impacts check catching corruptions of its own.
#[test]
fn impacts_and_doc_id_run_checks_run_and_catch_impact_corruption() {
    let dst = tempdir();
    let commit = write_many_doc_postings_fixture(&dst, 1000);
    let dir = FsDirectory::open(&dst);
    let clean = check_segment(&dir, &commit);
    for name in ["postings.impacts:body", "postings.doc_id_runs:body"] {
        let check = clean.checks.iter().find(|c| c.name == name).expect(name);
        assert!(check.passed(), "{name}: {}", check.message);
    }
    let doc_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.doc"));
    let original = std::fs::read(&doc_path).unwrap();
    let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
    let mut caught_by_impacts = 0usize;
    for off in 48..body_end {
        let mut bytes = original.clone();
        bytes[off] ^= 0x01;
        repair_checksum(&mut bytes);
        std::fs::write(&doc_path, &bytes).unwrap();
        let result = check_segment(&FsDirectory::open(&dst), &commit);
        if result
            .failures()
            .iter()
            .any(|c| c.name == "postings.impacts:body")
        {
            caught_by_impacts += 1;
        }
    }
    std::fs::write(&doc_path, &original).unwrap();
    assert!(
        caught_by_impacts >= 1,
        "no re-signed .doc bit flip was caught by postings.impacts"
    );
}

/// `postings.intersect_agrees` scans a term of tens of thousands of code
/// points in constant stack: a backtracker for `.*[a-e].*` would use a few
/// frames per code point and overflow a 2 MB test-thread stack on a long
/// collation key. A binary
/// term takes part in Java's `anyBinary [a-e] anyBinary` round too.
#[test]
fn intersect_agrees_scans_a_long_term_in_constant_stack() {
    use crate::document::{Document, Store, StringField};
    use crate::index_writer::IndexWriter;
    let dst = tempdir();
    let dir = FsDirectory::open(&dst);
    let version = segment_info::LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
    let terms = [
        "z".repeat(30_000).into_bytes(),
        // Not UTF-8, so only the binary automaton matches it.
        vec![0xff, b'b', 0xff],
        b"cat".to_vec(),
    ];
    let (compiled, run) = any_binary_a_to_e_any_binary().unwrap();
    assert_eq!(
        compiled.automaton_type,
        lucene_util::automaton::AutomatonType::NORMAL
    );
    let regexp = lucene_codecs::regexp::RegexpPattern::new(b".*[a-e].*")
        .unwrap()
        .to_dfa()
        .unwrap();
    use lucene_util::automaton::ByteRunnable;
    let verdicts: Vec<(bool, bool)> = terms.iter().map(|t| (run.run(t), regexp.run(t))).collect();
    assert_eq!(verdicts, [(false, false), (true, false), (true, true)]);
    for term in terms {
        let mut d = Document::new();
        d.add(StringField::from_bytes("k", term, Store::No));
        w.add_fields_document(&d).unwrap();
    }
    w.commit().unwrap();
    drop(w);
    let results = check_directory(&dir).unwrap();
    let check = results[1]
        .checks
        .iter()
        .find(|c| c.name == "postings.intersect_agrees:k")
        .expect("the intersect check ran on the field");
    assert!(check.passed(), "{check:?}");
    assert!(results.iter().all(CheckResult::all_passed));
}

/// A segment whose only indexed field has term vectors enabled but no
/// tokens has neither postings nor any vectors: the cross-check has
/// nothing to compare and passes, as in Java, rather than reporting
/// `term_vectors.match_postings` as skipped ("the postings side").
#[test]
fn term_vectors_without_any_terms_pass_the_postings_cross_check() {
    use crate::document::{Document, Field, FieldType, IndexOptions};
    use crate::index_writer::IndexWriter;
    let dst = tempdir();
    let dir = FsDirectory::open(&dst);
    let version = segment_info::LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
    let mut ft = FieldType::new();
    ft.set_tokenized(true).unwrap();
    ft.set_index_options(IndexOptions::DocsAndFreqsAndPositions)
        .unwrap();
    ft.set_store_term_vectors(true).unwrap();
    let ft = ft.frozen();
    let mut d = Document::new();
    d.add(Field::from_string("body", "", ft).unwrap());
    w.add_fields_document(&d).unwrap();
    w.commit().unwrap();
    drop(w);
    let results = check_directory(&dir).unwrap();
    assert!(
        results.iter().all(CheckResult::all_passed),
        "{:?}",
        results
            .iter()
            .flat_map(|r| r.failures())
            .collect::<Vec<_>>()
    );
}

/// `-exorcise`: a segment whose `.fdt` fails its checksum is dropped
/// from a new commit, the other segment's documents survive, and the
/// result checks clean. Results of another commit are refused, and a
/// clean index writes nothing.
#[test]
fn exorcise_drops_the_broken_segment_in_a_new_commit() {
    use crate::index_writer::IndexWriter;
    let dst = tempdir();
    let dir = FsDirectory::open(&dst);
    let version = segment_info::LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(
        &dir,
        vec![field_infos::FieldInfo::new("id", 0)],
        "Lucene104",
        version,
    )
    .unwrap();
    let doc = |id: &str| stored_fields::Document {
        fields: vec![stored_fields::StoredField {
            field_number: 0,
            value: stored_fields::FieldValue::String(id.to_string()),
        }],
    };
    w.add_document(doc("a")).unwrap();
    w.commit().unwrap();
    w.add_document(doc("b")).unwrap();
    w.add_document(doc("c")).unwrap();
    w.commit().unwrap();
    drop(w);
    let results = check_directory(&dir).unwrap();
    assert_eq!(exorcise(&dir, &results).unwrap(), None, "a clean index");

    let before = segment_infos::read_latest(&dir).unwrap();
    let broken = &before.segments[1];
    let fdt = dir
        .list_all()
        .unwrap()
        .into_iter()
        .find(|f| *f == format!("{}.fdt", broken.segment_name))
        .expect("the second segment's .fdt");
    let path = dst.join(&fdt);
    let mut bytes = std::fs::read(&path).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x40;
    std::fs::write(&path, &bytes).unwrap();
    let results = check_directory(&dir).unwrap();
    assert!(!results[2].all_passed());
    assert!(results[1].all_passed());
    assert!(
        exorcise(&dir, &results[..2]).is_err(),
        "a partial check is refused"
    );

    let written = exorcise(&dir, &results).unwrap().expect("a new commit");
    let after = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(
        Some(written),
        lucene_store::directory::segments_file_name(after.generation)
    );
    assert_eq!(after.generation, before.generation + 1);
    assert!(after.version > before.version);
    assert_eq!(after.segments.len(), 1);
    assert_eq!(
        after.segments[0].segment_name,
        before.segments[0].segment_name
    );
    assert!(check_directory(&dir)
        .unwrap()
        .iter()
        .all(CheckResult::all_passed));
    // The old results no longer describe the latest commit.
    assert!(exorcise(&dir, &results).is_err());
}

/// Negative control for `postings.seek_agrees`: the `.tip` trie is the
/// one structure a `seekExact`/`seekCeil` consults and a forward scan
/// does not, so corrupting it is exactly the case where the two
/// disagree -- a term the scan enumerated that a seek can no longer
/// find, or a `seekCeil` that lands on the wrong term. `try_seek_exact`
/// / `try_seek_ceil` (the error-carrying forms) are what let that
/// surface as a failure instead of a silent "not found".
#[test]
fn corrupting_the_term_index_is_caught_by_the_seek_check() {
    let dst = copy_fixture("blocktree_index");
    let tip_path = find_file(&dst, ".tip");
    let original = std::fs::read(&tip_path).unwrap();
    let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
    let mut caught_by_seek = 0usize;
    let mut caught_by_other = 0usize;
    let mut accepted = 0usize;
    let mut isolated: Option<(usize, u8)> = None;
    // Sampled rather than exhaustive: a full check of this 8 959-document
    // fixture costs ~20 ms, and an every-byte x every-mask sweep would
    // dominate the test suite. Every fourth byte still covers every
    // structural region of the trie.
    //
    // Re-signed, like every other negative control in this module: with a
    // stale footer the `file:*` CRC catches every flip, so
    // `!failed.is_empty()` is trivially true and `caught_by_seek > 0` is
    // the only real claim left -- one that survives a regression to a
    // single caught byte.
    for off in (48..body_end).step_by(4) {
        for mask in [0x01u8, 0x80, 0xff] {
            let mut bytes = original.clone();
            bytes[off] ^= mask;
            repair_checksum(&mut bytes);
            std::fs::write(&tip_path, &bytes).unwrap();
            let dir = FsDirectory::open(&dst);
            // A corruption that makes the *commit* unreadable is still a
            // corruption this module rejected -- counting it as
            // "caught by another check" keeps `total` equal to the sweep's
            // own iteration count, which is what makes the floors below
            // mean what they say.
            let failed = match check_directory(&dir) {
                Ok(r) => failed_names(&r),
                Err(e) => vec![format!("commit.unreadable: {e}")],
            };
            let is_seek = |n: &String| n.starts_with("postings.seek_agrees:");
            if failed.iter().any(is_seek) {
                caught_by_seek += 1;
                if failed.iter().all(is_seek) {
                    isolated.get_or_insert((off, mask));
                }
            } else if failed.is_empty() {
                accepted += 1;
            } else {
                caught_by_other += 1;
            }
        }
    }
    std::fs::write(&tip_path, &original).unwrap();
    let total = caught_by_seek + caught_by_other + accepted;
    // Measured when this was written, over 99 re-signed single-byte
    // `.tip` corruptions: **44** were rejected by `postings.seek_agrees`
    // specifically, 12 by another check, and 43 were accepted. The 43 are
    // expected and are a property of the format, not a hole: the `.tip`
    // trie is an *index* into the `.tim`, so many of its bytes only
    // change which block a seek starts scanning from, and a scan that
    // starts in the right block still finds the right term.
    assert!(
        caught_by_seek >= 38,
        "postings.seek_agrees rejected only {caught_by_seek} of {total} re-signed .tip \
             corruptions ({caught_by_other} caught by another check, {accepted} accepted); \
             44 were rejected when this test was written"
    );
    // The isolation: at least one corruption trips the seek check and
    // nothing else, which is what proves it is a witness in its own right
    // rather than riding on a neighbouring check's failure.
    let (off, mask) = isolated.expect(
        "no .tip corruption tripped postings.seek_agrees alone, so the check cannot be \
             distinguished from the checks around it",
    );
    let mut bytes = original.clone();
    bytes[off] ^= mask;
    repair_checksum(&mut bytes);
    std::fs::write(&tip_path, &bytes).unwrap();
    let dir = FsDirectory::open(&dst);
    let failed = failed_names(&check_directory(&dir).unwrap());
    std::fs::write(&tip_path, &original).unwrap();
    assert!(
        !failed.is_empty()
            && failed
                .iter()
                .all(|n| n.starts_with("postings.seek_agrees:")),
        "byte {off} ^ {mask:#x} was expected to trip only the seek check: {failed:?}"
    );
    let dir = FsDirectory::open(&dst);
    assert!(check_directory(&dir)
        .unwrap()
        .iter()
        .all(|r| r.all_passed()));
}

/// Negative control on real Java-written positional bytes: no
/// single-byte corruption of `blocktree_index`'s `.pos` may pass every
/// check. Without a position decode a `.pos` would be validated only by its
/// footer.
#[test]
fn no_corruption_of_a_real_pos_file_is_silently_accepted() {
    let dst = copy_fixture("blocktree_index");
    let pos_path = find_file(&dst, ".pos");
    let original = std::fs::read(&pos_path).unwrap();
    let payload = 48..(original.len() - lucene_store::codec_util::FOOTER_LENGTH);
    for off in payload {
        for mask in [0x01u8, 0x80, 0xff] {
            let mut bytes = original.clone();
            bytes[off] ^= mask;
            std::fs::write(&pos_path, &bytes).unwrap();
            let dir = FsDirectory::open(&dst);
            let results = check_directory(&dir).unwrap();
            assert!(
                !failed_names(&results).is_empty(),
                "flipping .pos byte {off} with {mask:#x} was silently accepted"
            );
        }
    }
    std::fs::write(&pos_path, &original).unwrap();
    let dir = FsDirectory::open(&dst);
    assert!(
        check_directory(&dir)
            .unwrap()
            .iter()
            .all(|r| r.all_passed()),
        "restoring the original bytes must restore a clean run"
    );
}

/// Negative control for `doc_values.terms_sorted` and
/// `doc_values.ords_dense`: no single-byte corruption of a SORTED
/// field's `.dvd` may be silently accepted, and at least one must be
/// caught by the ordinal-space checks specifically -- an ordinal nothing
/// uses, or an ordinal-to-term dictionary that is no longer sorted (in
/// which case ordinal comparison stops being term comparison, and every
/// `SortedDocValues` range query and index sort built on it is wrong
/// while every other check still passes).
#[test]
fn corrupted_sorted_dv_ordinal_space_is_caught() {
    let dst = copy_fixture("sorted_dv_index");
    let dvd_path = find_file(&dst, ".dvd");
    let original = std::fs::read(&dvd_path).unwrap();
    let dir = FsDirectory::open(&dst);
    assert!(
        check_directory(&dir)
            .unwrap()
            .iter()
            .all(|r| r.all_passed()),
        "the fixture must start clean"
    );

    // The footer is recomputed over every corruption, so the file is
    // perfectly well-formed and only the *semantic* checks can reject it.
    // Without that, `file:*`'s CRC catches every flip on its own and a
    // `caught_by_ords > 0` assertion says nothing about the ordinal-space
    // checks.
    let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
    let mut caught_by_ords = 0usize;
    let mut caught_by_other = 0usize;
    let mut accepted = 0usize;
    let mut isolated: Option<(usize, u8)> = None;
    for off in 48..body_end {
        for mask in [0x01u8, 0x80, 0xff] {
            let mut bytes = original.clone();
            bytes[off] ^= mask;
            repair_checksum(&mut bytes);
            std::fs::write(&dvd_path, &bytes).unwrap();
            let dir = FsDirectory::open(&dst);
            // A corruption that makes the *commit* unreadable is still a
            // corruption this module rejected -- counting it as
            // "caught by another check" keeps `total` equal to the sweep's
            // own iteration count, which is what makes the floors below
            // mean what they say.
            let failed = match check_directory(&dir) {
                Ok(r) => failed_names(&r),
                Err(e) => vec![format!("commit.unreadable: {e}")],
            };
            let is_ord = |n: &String| {
                n.starts_with("doc_values.terms_sorted:") || n.starts_with("doc_values.ords_dense:")
            };
            if failed.iter().any(is_ord) {
                caught_by_ords += 1;
                if failed.iter().all(is_ord) {
                    isolated.get_or_insert((off, mask));
                }
            } else if failed.is_empty() {
                accepted += 1;
            } else {
                caught_by_other += 1;
            }
        }
    }
    std::fs::write(&dvd_path, &original).unwrap();
    // A floor with the measured number in it, not `> 0`: `> 0` survives a
    // regression that leaves a single corruption in a hundred caught.
    // Measured when this was written: of 99 re-signed single-byte `.dvd`
    // corruptions, **18** were rejected by the ordinal-space checks
    // specifically, 27 by another doc-values check, and 54 were accepted.
    // The 54 are real and expected: `sorted_dv_index`'s `.dvd` is 97
    // bytes of which most are the term dictionary's own *bytes*, and
    // changing a byte inside a term leaves the ordinal space -- which is
    // about ordinal density and term *order* -- perfectly valid. That is
    // precisely why re-signing matters: unsigned, all 99 would have been
    // "caught", by the CRC, and the number would have said nothing.
    assert!(
        caught_by_ords >= 15,
        "the ordinal-space checks rejected only {caught_by_ords} of {} re-signed .dvd \
             corruptions ({caught_by_other} caught by another check, {accepted} accepted); \
             18 were rejected when this test was written",
        caught_by_ords + caught_by_other + accepted
    );

    // The wiring, and the isolation: at least one corruption trips the
    // ordinal-space checks *and nothing else*, which is what proves the
    // checks are not riding on a neighbour's failure.
    let (off, mask) = isolated.expect(
        "no .dvd corruption tripped the ordinal-space checks alone, so they cannot be \
             distinguished from the checks around them",
    );
    let mut bytes = original.clone();
    bytes[off] ^= mask;
    repair_checksum(&mut bytes);
    std::fs::write(&dvd_path, &bytes).unwrap();
    let dir = FsDirectory::open(&dst);
    let failed = failed_names(&check_directory(&dir).unwrap());
    std::fs::write(&dvd_path, &original).unwrap();
    assert!(
        failed
            .iter()
            .all(|n| n.starts_with("doc_values.terms_sorted:")
                || n.starts_with("doc_values.ords_dense:")),
        "byte {off} ^ {mask:#x} was expected to trip only the ordinal-space checks: \
             {failed:?}"
    );

    let dir = FsDirectory::open(&dst);
    assert!(check_directory(&dir)
        .unwrap()
        .iter()
        .all(|r| r.all_passed()));
}

/// The doc-values skipper check must actually run, and be non-vacuous, on
/// the one fixture that has a `.dvs` -- a check that silently never fires
/// is worse than no check.
#[test]
fn skipper_checks_actually_run_on_the_skip_index_fixture() {
    let dir = FsDirectory::open(fixture_dir("doc_values_skip_index"));
    let results = check_directory(&dir).unwrap();
    let skipper_checks: Vec<&Check> = results
        .iter()
        .flat_map(|r| r.checks.iter())
        .filter(|c| c.name.starts_with("doc_values.skipper:"))
        .collect();
    assert!(
        !skipper_checks.is_empty(),
        "expected a doc_values.skipper check on the skip-index fixture, got {:?}",
        results
            .iter()
            .flat_map(|r| r.checks.iter().map(|c| c.name.clone()))
            .collect::<Vec<_>>()
    );
    assert!(
        skipper_checks.iter().all(|c| c.passed()),
        "the skip-index fixture must pass its own skipper check: {skipper_checks:?}"
    );
    // That the walk is non-vacuous -- that it really visits intervals and
    // really tests their bounds -- is what
    // `corrupting_the_doc_values_skipper_is_caught_by_the_skipper_check`
    // proves; a passing check on a good file cannot prove it alone.
}

/// Negative control for `doc_values.skipper`: a `.dvs` whose *contents*
/// lie but whose CRC is honest.
///
/// Every other negative control in this module flips a byte and leaves the
/// footer stale, which the `file:*` checksum check catches on its own --
/// useless for proving a *semantic* check works. Here the footer is
/// recomputed over the corrupted bytes, so the file is perfectly
/// well-formed and only the skip index's own invariants (each level's
/// bounds nested inside the global ones, doc ranges not inverted, the
/// per-interval doc counts summing to the declared `docCount`) can catch
/// it. A `.dvs` that was only CRC-verified and never interpreted would
/// pass every one of these.
///
/// The sweep runs against the check function directly, because a
/// `check_directory` call over this fixture costs ~0.9 s and there are
/// hundreds of corruptions; one of the corruptions it finds is then put
/// through the whole of `check_directory` to prove the wiring.
#[test]
fn corrupting_the_doc_values_skipper_is_caught_by_the_skipper_check() {
    let dst = copy_fixture("doc_values_skip_index");
    let dvs_path = find_file(&dst, ".dvs");
    let original = std::fs::read(&dvs_path).unwrap();
    let dir = FsDirectory::open(&dst);
    let before = check_directory(&dir).unwrap();
    assert!(
        before.iter().all(|r| r.all_passed()),
        "the fixture must start clean: {:?}",
        failed_names(&before)
    );

    // Everything the sweep needs to parse a `.dvs` on its own, read once.
    let infos = segment_infos::read_latest(&dir).unwrap();
    let commit = &infos.segments[0];
    let si = open_si(&dir, commit).unwrap();
    let fnm = open_fnm(&dir, commit, &si).unwrap();
    let dvm_name = si.files.iter().find(|f| f.ends_with(".dvm")).unwrap();
    let suffix = dvm_name
        .strip_prefix(&format!("{}_", commit.segment_name))
        .and_then(|s| s.strip_suffix(".dvm"))
        .unwrap_or_default()
        .to_string();
    let dvm = dir.open(dvm_name).unwrap();
    let (_version, meta) = doc_values::parse_meta(&dvm, &commit.segment_id, &suffix, &fnm).unwrap();
    let field = fnm
        .fields
        .iter()
        .find(|f| f.doc_values_skip_index_type != field_infos::DocValuesSkipIndexType::None)
        .expect("the fixture has a skip-index field");
    let skipper_meta = meta
        .skipper_meta(field.number)
        .expect("the .dvm carries its skipper entry");

    // Past the index header (codec name, version, segment id, suffix),
    // whose own validation would catch a flip without needing any of the
    // semantic checks.
    let body_start = 64usize;
    let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
    let resign = |off: usize, mask: u8| {
        let mut bytes = original[..body_end].to_vec();
        bytes[off] ^= mask;
        lucene_store::codec_util::write_footer(&mut bytes);
        bytes
    };

    let mut caught = 0usize;
    let mut accepted = 0usize;
    let mut first_caught: Option<(usize, u8)> = None;
    for off in body_start..body_end {
        for mask in [0x01u8, 0xff] {
            let bytes = resign(off, mask);
            let rejected = match doc_values::parse_skip_index(
                &bytes,
                &commit.segment_id,
                &suffix,
                skipper_meta,
            ) {
                // A structurally impossible skip index (a zero or
                // over-large level count, a truncated level body) is the
                // decoder's own rejection, not this check's.
                Err(_) => false,
                Ok(index) => !check_doc_value_skipper(&index).is_empty(),
            };
            if rejected {
                caught += 1;
                first_caught.get_or_insert((off, mask));
            } else {
                accepted += 1;
            }
        }
    }
    // A floor with the measured number in it, not `> 0`: `> 0` survives a
    // regression that leaves one check firing out of nine. 428 of 574
    // single-byte corruptions were rejected when this was written; the
    // other 146 are genuinely harmless (a bound moved but stayed nested
    // inside the global range).
    assert!(
        caught >= 400,
        "the skipper checks rejected only {caught} of {} re-signed .dvs corruptions \
             ({accepted} accepted); 428 were rejected when this test was written",
        caught + accepted
    );

    // The wiring: one corruption the checks reject must come back out of
    // `check_directory` as a `doc_values.skipper` failure -- and as
    // *only* that, since the file's CRC is honest and every other check
    // reads a different file.
    let (off, mask) = first_caught.expect("at least one corruption was caught");
    std::fs::write(&dvs_path, resign(off, mask)).unwrap();
    let dir = FsDirectory::open(&dst);
    let failed = failed_names(&check_directory(&dir).unwrap());
    std::fs::write(&dvs_path, &original).unwrap();
    assert!(
        failed.iter().any(|n| n.starts_with("doc_values.skipper:")),
        "a .dvs corruption with an honest CRC was not reported by the skipper check \
             (byte {off} ^ {mask:#x}); failures: {failed:?}"
    );
    assert!(
        failed.iter().all(|n| n.starts_with("doc_values.skipper:")),
        "the corruption was expected to trip only the skipper check: {failed:?}"
    );

    let dir = FsDirectory::open(&dst);
    assert!(check_directory(&dir)
        .unwrap()
        .iter()
        .all(|r| r.all_passed()));
}

/// Negative control for `norms.entries_name_real_norms_fields`: a `.nvm`
/// whose entry names a field number the `.fnm` does not have.
///
/// Java's `Lucene90NormsProducer.readFields` rejects the segment outright
/// for this; this port accepted it, and the consequence is silent rather
/// than loud -- the entry becomes unreachable, so every norm lookup for
/// the *real* field falls back to "this field has no norms" and every
/// score is computed against a default norm. Nothing else in this module
/// notices, because `norms.entry_present` only looks the other way (a
/// field with no entry), and the file's CRC is re-signed here so the
/// integrity checks stay clean.
#[test]
fn a_nvm_entry_for_a_nonexistent_field_is_caught() {
    let dst = copy_fixture("norms_index");
    let nvm_path = find_file(&dst, ".nvm");
    let original = std::fs::read(&nvm_path).unwrap();
    let dir = FsDirectory::open(&dst);
    let before = check_directory(&dir).unwrap();
    assert!(
        before.iter().all(|r| r.all_passed()),
        "the fixture must start clean: {:?}",
        failed_names(&before)
    );

    // The first entry's field number sits immediately after the index
    // header (magic, codec name, version, segment id, empty suffix).
    let infos = segment_infos::read_latest(&dir).unwrap();
    let commit = &infos.segments[0];
    let (_version, norms) = norms::parse_meta(&original, &commit.segment_id, "").unwrap();
    let first = norms.entries[0].field_number;
    const FIELD_NUMBER_OFFSET: usize = 47;
    assert_eq!(
        i32::from_le_bytes(
            original[FIELD_NUMBER_OFFSET..FIELD_NUMBER_OFFSET + 4]
                .try_into()
                .unwrap()
        ),
        first,
        "the .nvm's first field number is not where this test patches it"
    );

    let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
    let mut bytes = original[..body_end].to_vec();
    // 4242: a field number no `.fnm` in this fixture has.
    bytes[FIELD_NUMBER_OFFSET..FIELD_NUMBER_OFFSET + 4].copy_from_slice(&4242i32.to_le_bytes());
    lucene_store::codec_util::write_footer(&mut bytes);
    std::fs::write(&nvm_path, &bytes).unwrap();

    let dir = FsDirectory::open(&dst);
    let failed = failed_names(&check_directory(&dir).unwrap());
    std::fs::write(&nvm_path, &original).unwrap();
    assert!(
        failed
            .iter()
            .any(|n| n == "norms.entries_name_real_norms_fields"),
        "a .nvm naming a nonexistent field was not caught: {failed:?}"
    );
}

/// Negative control for `norms.agree_with_postings`: zeroing the norm of
/// a document that does have terms is exactly the "norm value is 0,
/// which may only be used on documents that have no terms" case Java
/// rejects, and nothing else in this module would notice it.
#[test]
fn a_zeroed_norm_for_a_doc_with_terms_is_caught() {
    let dst = copy_fixture("norms_index");
    let nvd_path = find_file(&dst, ".nvd");
    let original = std::fs::read(&nvd_path).unwrap();
    let dir = FsDirectory::open(&dst);
    let before = check_directory(&dir).unwrap();
    assert!(
        before.iter().all(|r| r.all_passed()),
        "the fixture must start clean: {:?}",
        failed_names(&before)
    );

    // The norms payload is a flat `numDocsWithField * bytesPerNorm`
    // array; zeroing any byte of it zeroes some document's norm.
    let mut zeroed = 0usize;
    for off in 0..(original.len() - lucene_store::codec_util::FOOTER_LENGTH) {
        if original[off] == 0 {
            continue;
        }
        let mut bytes = original.clone();
        bytes[off] = 0;
        std::fs::write(&nvd_path, &bytes).unwrap();
        let dir = FsDirectory::open(&dst);
        let failed = failed_names(&check_directory(&dir).unwrap());
        if failed
            .iter()
            .any(|n| n.starts_with("norms.agree_with_postings:"))
        {
            zeroed += 1;
            break;
        }
    }
    std::fs::write(&nvd_path, &original).unwrap();
    assert!(
        zeroed > 0,
        "zeroing a norm byte was never caught by norms.agree_with_postings"
    );
}

/// Negative control for the norms *value* pass: truncating the `.nvd`
/// payload region so a norm read runs off the end must be reported by
/// `norms.values_decode`, not silently skipped.
#[test]
fn a_norms_file_that_cannot_be_read_is_caught_by_values_decode() {
    let dst = copy_fixture("norms_index");
    let nvm_path = find_file(&dst, ".nvm");
    let original = std::fs::read(&nvm_path).unwrap();
    let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
    let mut caught_by_norms = 0usize;
    let mut caught_by_other = 0usize;
    let mut accepted = 0usize;
    let mut isolated: Option<usize> = None;
    for off in 24..body_end {
        let mut bytes = original.clone();
        bytes[off] ^= 0x11;
        repair_checksum(&mut bytes);
        std::fs::write(&nvm_path, &bytes).unwrap();
        let dir = FsDirectory::open(&dst);
        // See the note in the `.tip` sweep: an unreadable commit counts
        // as caught by another check, not as a skipped iteration.
        let failed = match check_directory(&dir) {
            Ok(r) => failed_names(&r),
            Err(e) => vec![format!("commit.unreadable: {e}")],
        };
        let is_norms = |n: &String| n.starts_with("norms.");
        if failed.iter().any(is_norms) {
            caught_by_norms += 1;
            if failed.iter().all(is_norms) {
                isolated.get_or_insert(off);
            }
        } else if failed.is_empty() {
            accepted += 1;
        } else {
            caught_by_other += 1;
        }
    }
    std::fs::write(&nvm_path, &original).unwrap();
    let total = caught_by_norms + caught_by_other + accepted;
    // Measured when this was written, over 99 re-signed single-byte
    // `.nvm` corruptions: **85** were rejected by a `norms.*` check, 0 by
    // any other check, and 14 accepted. The zero is the interesting
    // number: the `.nvm` is read by nothing else in this module, so if
    // the norms checks do not catch a corruption of it, nothing does --
    // which is exactly why a `> 0` floor here was worth replacing.
    assert!(
        caught_by_norms >= 75,
        "the norms checks rejected only {caught_by_norms} of {total} re-signed .nvm \
             corruptions ({caught_by_other} caught by another check, {accepted} accepted); \
             85 were rejected when this test was written"
    );
    assert_eq!(
        caught_by_other, 0,
        "a .nvm corruption was reported by something other than a norms.* check; the \
             norms checks are meant to be the only reader of this file"
    );
    let off = isolated.expect("at least one .nvm corruption is caught by norms.* alone");
    let mut bytes = original.clone();
    bytes[off] ^= 0x11;
    repair_checksum(&mut bytes);
    std::fs::write(&nvm_path, &bytes).unwrap();
    let dir = FsDirectory::open(&dst);
    let failed = failed_names(&check_directory(&dir).unwrap());
    std::fs::write(&nvm_path, &original).unwrap();
    assert!(
        !failed.is_empty() && failed.iter().all(|n| n.starts_with("norms.")),
        "byte {off} was expected to trip only the norms checks: {failed:?}"
    );
}

/// Overwrites the codec footer's CRC-32 with the one the (possibly
/// modified) payload actually hashes to, so a deliberately corrupted
/// file still passes `CodecUtil.checksumEntireFile`.
///
/// Without this a corruption test of `.vec`/`.vex` proves nothing about
/// the structural checks: `FlatVectorsReader::open` and
/// `HnswVectorsReader::open` both verify the *whole file's* checksum
/// before parsing a single field entry (as
/// `Lucene99FlatVectorsReader`/`Lucene99HnswVectorsReader` do), so every
/// byte flip surfaces as `vectors.open`/`hnsw.open` and never reaches
/// the graph. Repairing the checksum isolates the check under test from
/// the checksum, which is what "corrupt exactly the thing this check is
/// meant to catch" requires here.
fn repair_checksum(bytes: &mut [u8]) {
    let n = bytes.len();
    let crc = crc32fast::hash(&bytes[..n - 8]) as u64;
    bytes[n - 8..].copy_from_slice(&crc.to_be_bytes());
}

/// Negative control for the HNSW graph checks. The `.vex` graph region
/// is corrupted and the file's checksum repaired (see
/// [`repair_checksum`]), so the corruption reaches the graph walk rather
/// than being stopped by the reader's whole-file CRC. At least one
/// corruption must be reported by `hnsw.neighbors_on_level` or
/// `hnsw.neighbors_sorted` -- a neighbour that is not on its own level,
/// out of the ordinal range, repeated, or out of order -- which is
/// exactly what Java's `testHnswGraph` rejects and what nothing else in
/// this module would notice.
#[test]
fn corrupted_hnsw_neighbours_are_caught_by_the_graph_checks() {
    let dst = copy_fixture("vectors_index");
    let vex_path = find_file(&dst, ".vex");
    let original = std::fs::read(&vex_path).unwrap();
    let body_end = original.len() - 16;
    let mut caught_by_graph = 0usize;
    let mut caught_by_other = 0usize;
    let mut accepted = 0usize;
    let mut reached_the_graph = 0usize;
    let mut isolated: Option<(usize, u8)> = None;
    // A fixed, uniform sample of the 223 kB `.vex` rather than an
    // early-exit-at-three walk: the old shape stopped as soon as it had
    // three hits, so the count it asserted (`> 0`) could never say how
    // far the checks actually reach. Every 2 111th byte x three masks is
    // ~105 x 3 whole-directory checks over a 4 000-document vectors
    // index, which is the affordable budget here.
    for off in (48..body_end).step_by(2111) {
        for mask in [0x01u8, 0x80, 0xff] {
            let mut bytes = original.clone();
            bytes[off] ^= mask;
            repair_checksum(&mut bytes);
            std::fs::write(&vex_path, &bytes).unwrap();
            let dir = FsDirectory::open(&dst);
            // A corruption that makes the *commit* unreadable is still a
            // corruption this module rejected -- counting it as
            // "caught by another check" keeps `total` equal to the sweep's
            // own iteration count, which is what makes the floors below
            // mean what they say.
            let failed = match check_directory(&dir) {
                Ok(r) => failed_names(&r),
                Err(e) => vec![format!("commit.unreadable: {e}")],
            };
            if !failed.iter().any(|n| n == "hnsw.open") {
                reached_the_graph += 1;
            }
            let is_graph = |n: &String| {
                n.starts_with("hnsw.neighbors_on_level:") || n.starts_with("hnsw.neighbors_sorted:")
            };
            if failed.iter().any(is_graph) {
                caught_by_graph += 1;
                if failed.iter().all(is_graph) {
                    isolated.get_or_insert((off, mask));
                }
            } else if failed.is_empty() {
                accepted += 1;
            } else {
                caught_by_other += 1;
            }
        }
    }
    std::fs::write(&vex_path, &original).unwrap();
    let total = caught_by_graph + caught_by_other + accepted;
    // Measured when this was written, over 318 re-signed single-byte
    // `.vex` corruptions (106 offsets x 3 masks): **315** got past
    // `hnsw.open` into the graph itself, **138** were rejected by the
    // neighbour checks specifically, 3 by another check, and 177 were
    // accepted -- the `.vex` is mostly packed neighbour ordinals, and a
    // flip that keeps an ordinal in range, distinct and in order is a
    // different but still well-formed graph, which no structural check
    // can or should reject.
    assert!(
        reached_the_graph >= 300,
        "only {reached_the_graph} of {total} corruptions got past hnsw.open, so the graph \
             checks were barely exercised; 315 did when this test was written"
    );
    assert!(
        caught_by_graph >= 120,
        "the neighbour checks rejected only {caught_by_graph} of {total} re-signed .vex \
             corruptions ({caught_by_other} caught by another check, {accepted} accepted); \
             138 were rejected when this test was written"
    );
    let (off, mask) =
        isolated.expect("at least one .vex corruption is caught by the neighbour checks alone");
    let mut bytes = original.clone();
    bytes[off] ^= mask;
    repair_checksum(&mut bytes);
    std::fs::write(&vex_path, &bytes).unwrap();
    let dir = FsDirectory::open(&dst);
    let failed = failed_names(&check_directory(&dir).unwrap());
    std::fs::write(&vex_path, &original).unwrap();
    assert!(
        !failed.is_empty()
            && failed
                .iter()
                .all(|n| n.starts_with("hnsw.neighbors_on_level:")
                    || n.starts_with("hnsw.neighbors_sorted:")),
        "byte {off} ^ {mask:#x} was expected to trip only the neighbour checks: {failed:?}"
    );
    let dir = FsDirectory::open(&dst);
    assert!(check_directory(&dir)
        .unwrap()
        .iter()
        .all(|r| r.all_passed()));
}

/// The complement: an *unrepaired* `.vex` byte flip must still be caught,
/// by the reader's own whole-file checksum and by the `file:*` integrity
/// check. Nothing gets through either way.
#[test]
fn corrupted_hnsw_graph_bytes_are_never_silently_accepted() {
    let dst = copy_fixture("vectors_index");
    let vex_path = find_file(&dst, ".vex");
    let original = std::fs::read(&vex_path).unwrap();
    for off in (48..(original.len() - 16)).step_by(997) {
        let mut bytes = original.clone();
        bytes[off] ^= 0x37;
        std::fs::write(&vex_path, &bytes).unwrap();
        let dir = FsDirectory::open(&dst);
        let failed = failed_names(&check_directory(&dir).unwrap());
        assert!(
            failed.iter().any(|n| n == "hnsw.open"),
            "flipping .vex byte {off} was not reported by hnsw.open: {failed:?}"
        );
    }
    std::fs::write(&vex_path, &original).unwrap();
}

/// Negative control for `vectors.ord_to_doc`: the ordinal-to-document
/// mapping lives in `.vemf` (a `DirectMonotonic` block plus, for a
/// sparse field, an `IndexedDISI`), so that is what gets corrupted --
/// with the checksum repaired (see [`repair_checksum`]) so the
/// corruption reaches the mapping rather than being stopped by
/// `FlatVectorsReader::open`'s whole-file CRC. Nothing may be silently
/// accepted, and at least one corruption must be reported by a
/// `vectors.*` check.
#[test]
fn corrupted_vector_ord_to_doc_mapping_is_caught() {
    let dst = copy_fixture("vectors_index");
    let vemf_path = find_file(&dst, ".vemf");
    let original = std::fs::read(&vemf_path).unwrap();
    let mut caught_by_vectors = 0;
    let mut reached_the_fields = 0;
    'sweep: for off in 48..(original.len() - 16) {
        for mask in [0x01u8, 0x80, 0xff] {
            if caught_by_vectors >= 3 && reached_the_fields >= 3 {
                break 'sweep;
            }
            let mut bytes = original.clone();
            bytes[off] ^= mask;
            repair_checksum(&mut bytes);
            std::fs::write(&vemf_path, &bytes).unwrap();
            let dir = FsDirectory::open(&dst);
            // A corruption that makes the *commit* unreadable is still a
            // corruption this module rejected -- counting it as
            // "caught by another check" keeps `total` equal to the sweep's
            // own iteration count, which is what makes the floors below
            // mean what they say.
            let failed = match check_directory(&dir) {
                Ok(r) => failed_names(&r),
                Err(e) => vec![format!("commit.unreadable: {e}")],
            };
            // Note the *absence* of a "nothing is silently accepted"
            // assertion here, unlike the `.pos`/`.tim`/`.doc` sweeps:
            // `.vemf` is pure metadata with no redundancy, so some of
            // its bytes (a `DirectMonotonic` block shift that still
            // decodes to a monotone in-range doc list, say) have no
            // second copy to disagree with. Their guard is the
            // checksum, which is exactly why
            // `Lucene99FlatVectorsReader` verifies the whole file at
            // open and why `file:*` now does a full CRC too --
            // `corrupted_vector_data_is_caught` pins that direction.
            // Repairing the checksum removes that guard on purpose, to
            // reach the per-field checks at all.
            if !failed.iter().any(|n| n == "vectors.open") {
                reached_the_fields += 1;
            }
            if failed.iter().any(|n| {
                n.starts_with("vectors.ord_to_doc:")
                    || n.starts_with("vectors.field_entry_matches_fnm:")
                    || n.starts_with("vectors.values_decode:")
            }) {
                caught_by_vectors += 1;
            }
        }
    }
    std::fs::write(&vemf_path, &original).unwrap();
    assert!(
        reached_the_fields > 0,
        "no corruption got past vectors.open, so the field checks were never exercised"
    );
    assert!(
        caught_by_vectors > 0,
        "no .vemf corruption was caught by a per-field vectors check"
    );
    let dir = FsDirectory::open(&dst);
    assert!(check_directory(&dir)
        .unwrap()
        .iter()
        .all(|r| r.all_passed()));
}

/// Negative control for the vector-values pass: corrupting the `.vec`
/// payload must be reported, and the vectors reader's own whole-file
/// checksum is what reports it (`vectors.open`) -- the point of the test
/// is that the vectors are opened and read at all.
#[test]
fn corrupted_vector_data_is_caught() {
    let dst = copy_fixture("vectors_index");
    let vec_path = find_file(&dst, ".vec");
    let original = std::fs::read(&vec_path).unwrap();
    let mut bytes = original.clone();
    let off = original.len() / 2;
    bytes[off] ^= 0xff;
    std::fs::write(&vec_path, &bytes).unwrap();
    let dir = FsDirectory::open(&dst);
    let failed = failed_names(&check_directory(&dir).unwrap());
    assert!(
        failed.iter().any(|n| n == "vectors.open"),
        "expected vectors.open to fail, got {failed:?}"
    );
    std::fs::write(&vec_path, &original).unwrap();
}

/// Negative control for `term_vectors.match_postings`: rewriting the
/// segment's postings so a term vector's term is no longer in the
/// inverted index must be reported. Corrupting the `.tvd` alone would
/// be caught by the term-vectors decoder; corrupting the postings side
/// is what exercises the cross-check.
#[test]
fn a_term_vector_term_missing_from_the_postings_is_caught() {
    let dst = copy_fixture("term_vectors_index");
    let tim_path = find_file(&dst, ".tim");
    let original = std::fs::read(&tim_path).unwrap();
    let mut caught = 0;
    for off in 40..(original.len() - lucene_store::codec_util::FOOTER_LENGTH) {
        let mut bytes = original.clone();
        bytes[off] ^= 0x21;
        std::fs::write(&tim_path, &bytes).unwrap();
        let dir = FsDirectory::open(&dst);
        // See the note in the `.tip` sweep: an unreadable commit counts
        // as caught by another check, not as a skipped iteration.
        let failed = match check_directory(&dir) {
            Ok(r) => failed_names(&r),
            Err(e) => vec![format!("commit.unreadable: {e}")],
        };
        assert!(
            !failed.is_empty(),
            "flipping .tim byte {off} was silently accepted"
        );
        if failed.iter().any(|n| n == "term_vectors.match_postings") {
            caught += 1;
        }
    }
    std::fs::write(&tim_path, &original).unwrap();
    assert!(
        caught > 0,
        "no .tim corruption was caught by term_vectors.match_postings"
    );
}

/// Negative control for `CodecUtil.checksumEntireFile` (as opposed to
/// `retrieve_checksum`, footer *shape* only) in the `file:*` checks: a byte
/// flipped in the middle of a file, leaving every footer field intact,
/// must fail.
#[test]
fn a_payload_byte_flip_fails_the_file_integrity_check() {
    let dst = copy_fixture("points_index");
    let kdd_path = find_file(&dst, ".kdd");
    let original = std::fs::read(&kdd_path).unwrap();
    let mut bytes = original.clone();
    let off = original.len() / 2;
    bytes[off] ^= 0x01;
    std::fs::write(&kdd_path, &bytes).unwrap();
    let dir = FsDirectory::open(&dst);
    let failed = failed_names(&check_directory(&dir).unwrap());
    assert!(
        failed
            .iter()
            .any(|n| n.starts_with("file:") && n.ends_with(".kdd")),
        "expected the .kdd file integrity check to fail, got {failed:?}"
    );
    // The old shape-only check would not have noticed: prove that by
    // asserting the footer itself is still well-formed.
    assert!(lucene_store::codec_util::retrieve_checksum(&bytes).is_ok());
    std::fs::write(&kdd_path, &original).unwrap();
}

/// Same for term vectors.
#[test]
fn term_vector_checks_actually_run_on_a_term_vectors_fixture() {
    let dir = FsDirectory::open(fixture_dir("term_vectors_index"));
    let results = check_directory(&dir).unwrap();
    let names: Vec<&str> = results[1].checks.iter().map(|c| c.name.as_str()).collect();
    assert!(
        names.contains(&"term_vectors.every_doc_decodes"),
        "expected term_vectors checks, got {names:?}"
    );
    assert!(names.contains(&"term_vectors.fields_marked_in_fnm"));
}

/// Corrupting a byte inside the `.dvd` payload must be caught by the
/// value-decode pass; without a value read this exact corruption would pass
/// clean.
#[test]
fn corrupted_doc_values_payload_is_caught() {
    let src_dir = fixture_dir("sorted_dv_index");
    let dst_dir = tempdir();
    for entry in std::fs::read_dir(&src_dir).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), dst_dir.join(entry.file_name())).unwrap();
    }
    let dvd = dst_dir.join("_0_Lucene90_0.dvd");
    let mut bytes = std::fs::read(&dvd).unwrap();
    // The ordinals live right after the index header; flipping the high
    // bits of an early payload byte pushes an ordinal out of the terms
    // dictionary's range.
    let at = 40;
    bytes[at] ^= 0xFF;
    std::fs::write(&dvd, &bytes).unwrap();

    let dir = FsDirectory::open(&dst_dir);
    let results = check_directory(&dir).unwrap();
    let failures: Vec<&str> = results[1]
        .failures()
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    assert!(
        failures.iter().any(|n| n.starts_with("doc_values.")),
        "expected a doc_values failure, got {failures:?}"
    );

    std::fs::remove_dir_all(&dst_dir).ok();
}

/// A commit whose `counter` has fallen behind a live segment's name is
/// how `IndexWriter` ends up reusing a name and clobbering a segment --
/// real `CheckIndex`'s `validCounter`/`maxSegmentName` check.
#[test]
fn commit_counter_behind_segment_names_is_flagged() {
    let dir = FsDirectory::open(fixture_dir("blocktree_index"));
    let mut infos = segment_infos::read_latest(&dir).unwrap();
    infos.counter = 0; // segment `_0` parses to 0, so counter must be > 0
    let result = check_commit(&infos);
    let failure = result
        .checks
        .iter()
        .find(|c| c.name == "commit.counter_ahead_of_segment_names")
        .unwrap();
    assert!(!failure.passed(), "{failure:?}");
}

#[test]
fn duplicate_segment_names_in_a_commit_are_flagged() {
    let dir = FsDirectory::open(fixture_dir("blocktree_index"));
    let mut infos = segment_infos::read_latest(&dir).unwrap();
    let dupe = infos.segments[0].clone();
    infos.segments.push(dupe);
    infos.counter = 99;
    let result = check_commit(&infos);
    assert!(!result
        .checks
        .iter()
        .find(|c| c.name == "commit.segment_names_unique")
        .unwrap()
        .passed());
}

/// `SegmentInfos.readCommit` refuses a `delCount` larger than the
/// segment's own `maxDoc`. This port's `segment_infos::parse` cannot see
/// `maxDoc`, so the check lives here -- and without it nothing validated
/// it at all.
#[test]
fn del_count_larger_than_max_doc_is_flagged() {
    let dir = FsDirectory::open(fixture_dir("live_docs_index"));
    let mut commit = read_commit(&dir);
    commit.del_count = 1_000_000;
    let result = check_segment(&dir, &commit);
    assert!(!result
        .checks
        .iter()
        .find(|c| c.name == "commit.del_count_within_max_doc")
        .unwrap()
        .passed());
}

/// A commit that records deletions without a delete generation has no
/// `.liv` file to hold them -- `SegmentCommitInfo.hasDeletions()` is
/// `delGen != -1`.
#[test]
fn del_count_without_del_gen_is_flagged() {
    let dir = FsDirectory::open(fixture_dir("blocktree_index"));
    let mut commit = read_commit(&dir);
    assert_eq!(commit.del_gen, -1);
    commit.del_count = 3;
    let result = check_segment(&dir, &commit);
    assert!(!result
        .checks
        .iter()
        .find(|c| c.name == "commit.del_count_zero_without_del_gen")
        .unwrap()
        .passed());
}

// -- index sort / soft deletes --

const SORT_SEG_ID: [u8; ID_LENGTH] = [21u8; ID_LENGTH];

/// Writes a one-field segment (NUMERIC doc values, no postings/stored
/// fields) declaring `index_sort` over that field, with `values` as the
/// per-doc sort keys in doc-id order. `soft_deletes` marks the field as
/// the soft-deletes field so the same fixture drives both new checks.
fn write_sorted_dv_fixture(
    dst_dir: &std::path::Path,
    values: &[i64],
    sort: Option<Vec<segment_info::IndexSortField>>,
    soft_deletes: bool,
    soft_del_count: i32,
) -> segment_infos::SegmentCommitInfo {
    let max_doc = values.len() as i32;
    let fi = field_infos::FieldInfo {
        name: "ts".to_string(),
        number: 0,
        store_term_vectors: false,
        // A non-indexed field must not omit norms (`Lucene94FieldInfosFormat`
        // rejects that combination outright).
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: soft_deletes,
        parent_field: false,
        index_options: field_infos::IndexOptions::None,
        doc_values_type: field_infos::DocValuesType::Numeric,
        doc_values_skip_index_type: field_infos::DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: field_infos::VectorEncoding::Float32,
        vector_similarity_function: field_infos::VectorSimilarityFunction::Euclidean,
    };
    let fnm = field_infos::write(std::slice::from_ref(&fi), &SORT_SEG_ID, "");
    std::fs::write(dst_dir.join("_0.fnm"), &fnm).unwrap();
    let (dvm, dvd, dvs) = lucene_codecs::doc_values::write_single_dense_numeric_field(
        0,
        values,
        max_doc,
        &SORT_SEG_ID,
        "",
    )
    .unwrap();
    std::fs::write(dst_dir.join("_0.dvm"), &dvm).unwrap();
    std::fs::write(dst_dir.join("_0.dvd"), &dvd).unwrap();
    std::fs::write(dst_dir.join("_0.dvs"), &dvs).unwrap();

    let si = SegmentInfo {
        id: SORT_SEG_ID,
        version: segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        min_version: None,
        doc_count: max_doc,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: vec![
            "_0.si".to_string(),
            "_0.fnm".to_string(),
            "_0.dvm".to_string(),
            "_0.dvd".to_string(),
            "_0.dvs".to_string(),
        ],
        attributes: vec![],
        index_sort: sort,
    };
    std::fs::write(dst_dir.join("_0.si"), segment_info::write(&si, "")).unwrap();

    segment_infos::SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id: SORT_SEG_ID,
        codec_name: "Lucene104".to_string(),
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    }
}

fn sort_asc() -> Option<Vec<segment_info::IndexSortField>> {
    Some(vec![segment_info::IndexSortField::long(
        "ts",
        false,
        Some(i64::MAX),
    )])
}

/// The same ascending sort over a **multi-valued** column: a
/// `SortedNumericSortField` with the `MIN` selector, which is what Java's
/// `SortedNumericSortField` produces and the only kind whose
/// `getIndexSorter` reads a SORTED_NUMERIC column. A plain
/// `SortField(ts, LONG)` over the same field is what `DocValues.getNumeric`
/// throws on in Java, and what `sort_key_values` reports as a kind/type
/// mismatch here.
fn sorted_numeric_sort_asc() -> Option<Vec<segment_info::IndexSortField>> {
    Some(vec![segment_info::IndexSortField {
        field: "ts".to_string(),
        reverse: false,
        kind: segment_info::IndexSortKind::SortedNumeric {
            key: segment_info::NumericSortKey::Long(Some(i64::MAX)),
            selector: segment_info::SortedNumericSelector::Min,
        },
    }])
}

/// A segment whose doc values really are in the declared sort order must
/// pass; one whose are not must fail. A `.si`'s `indexSort` claim is
/// exactly the claim a sorted-index merge and an early-terminating query
/// both silently rely on.
#[test]
fn index_sort_order_is_verified_against_the_actual_doc_values() {
    let ok_dir = tempdir();
    let commit = write_sorted_dv_fixture(&ok_dir, &[10, 20, 20, 30], sort_asc(), false, 0);
    let dir = FsDirectory::open(&ok_dir);
    let result = check_segment(&dir, &commit);
    assert!(
        result.all_passed(),
        "unexpected failures: {:?}",
        result.failures()
    );
    assert!(result
        .checks
        .iter()
        .any(|c| c.name == "sort.docs_in_index_sort_order"));
    std::fs::remove_dir_all(&ok_dir).ok();

    let bad_dir = tempdir();
    let commit = write_sorted_dv_fixture(&bad_dir, &[10, 30, 20, 40], sort_asc(), false, 0);
    let dir = FsDirectory::open(&bad_dir);
    let result = check_segment(&dir, &commit);
    let failure = result
        .checks
        .iter()
        .find(|c| c.name == "sort.docs_in_index_sort_order")
        .unwrap();
    assert!(!failure.passed(), "{failure:?}");
    assert!(failure.message.contains("sorts after"));
    std::fs::remove_dir_all(&bad_dir).ok();
}

/// An unsorted segment must not get a sort check at all (nothing to
/// verify), rather than a vacuous pass.
#[test]
fn unsorted_segment_gets_no_sort_check() {
    let dst_dir = tempdir();
    let commit = write_sorted_dv_fixture(&dst_dir, &[3, 1, 2], None, false, 0);
    let dir = FsDirectory::open(&dst_dir);
    let result = check_segment(&dir, &commit);
    assert!(!result
        .checks
        .iter()
        .any(|c| c.name == "sort.docs_in_index_sort_order"));
    assert!(result.all_passed(), "{:?}", result.failures());
    std::fs::remove_dir_all(&dst_dir).ok();
}

/// `CheckIndex.checkSoftDeletes`: `softDelCount` must equal the number of
/// live docs carrying a value for the soft-deletes field.
/// **The soft-deletes check runs on a segment whose soft-deletes field
/// exists only in a field-infos generation**, and agrees with Java. In
/// `GenFieldInfos` the field arrives through `updateDocValues` on a
/// segment that never had it: it is in `_0_1.fnm`, not `_0.fnm`. Looked
/// up in the base `.fnm`, the check was never run -- no pass, no skip --
/// and `softDelCount` went unvalidated on exactly the segments OpenSearch
/// soft-deletes into after they were written.
/// `testDocValues` reads an updated field through its update
/// generation's producer: `GenDocValuesUpdates`' `val` lives in
/// generation 4 (`_0_4_Lucene90_0.*`, a sparse column),
/// `tag` in generation 2 and `keep` in the base column, and every one of
/// them is decoded -- Java's `CheckIndex` passes the index, and so must
/// this.
#[test]
fn doc_values_update_generations_are_decoded_against_java() {
    let dir = FsDirectory::open(fixture_dir("doc_values_updates_index"));
    let results = check_directory(&dir).unwrap();
    assert!(failed_names(&results).is_empty(), "{results:?}");
    let segment = results.iter().find(|r| r.segment_name == "_0").unwrap();
    for name in [
        "doc_values.values_decode:val",
        "doc_values.values_decode:tag",
        "doc_values.values_decode:keep",
    ] {
        let check = segment.checks.iter().find(|c| c.name == name);
        assert!(
            check.is_some_and(|c| c.passed()),
            "{name} must run and pass: {check:?}"
        );
    }
}

/// Negative control for the above: corrupting an update generation's
/// `.dvd` payload (checksum repaired, so only a decode can notice) is
/// caught by the doc-values checks. Before they read generations, every
/// one of these flips passed: the generation files were only
/// checksummed.
#[test]
fn a_corrupt_update_generation_is_caught_by_the_doc_values_checks() {
    let dst = copy_fixture("doc_values_updates_index");
    let dvd_path = dst.join("_0_2_Lucene90_0.dvd");
    let original = std::fs::read(&dvd_path).unwrap();
    let mut caught = 0usize;
    for off in 40..original.len() - 16 {
        let mut bytes = original.clone();
        bytes[off] ^= 0xff;
        repair_checksum(&mut bytes);
        std::fs::write(&dvd_path, &bytes).unwrap();
        let dir = FsDirectory::open(&dst);
        if let Ok(results) = check_directory(&dir) {
            if failed_names(&results)
                .iter()
                .any(|n| n.starts_with("doc_values."))
            {
                caught += 1;
            }
        }
    }
    std::fs::write(&dvd_path, &original).unwrap();
    assert!(
        caught > 0,
        "no corruption of the generation-2 column was caught"
    );
}

#[test]
fn a_soft_deletes_field_added_by_an_update_is_checked_against_java() {
    let dir = FsDirectory::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/field_infos_index"
    ));
    let results = check_directory(&dir).unwrap();
    let segment = results.iter().find(|r| r.segment_name == "_0").unwrap();
    let soft = segment
        .checks
        .iter()
        .find(|c| c.name == "soft_deletes.count_matches")
        .expect("the soft-deletes check must run");
    assert!(soft.passed(), "{soft:?}");
}

#[test]
fn soft_delete_count_is_verified_against_the_soft_deletes_field() {
    let dst_dir = tempdir();
    // Dense NUMERIC doc values: all 3 docs have a value -> 3 soft deletes.
    let commit = write_sorted_dv_fixture(&dst_dir, &[1, 1, 1], None, true, 3);
    let dir = FsDirectory::open(&dst_dir);
    let result = check_segment(&dir, &commit);
    assert!(
        result.all_passed(),
        "unexpected failures: {:?}",
        result.failures()
    );
    std::fs::remove_dir_all(&dst_dir).ok();

    let bad_dir = tempdir();
    let commit = write_sorted_dv_fixture(&bad_dir, &[1, 1, 1], None, true, 1);
    let dir = FsDirectory::open(&bad_dir);
    let result = check_segment(&dir, &commit);
    let failure = result
        .checks
        .iter()
        .find(|c| c.name == "soft_deletes.count_matches")
        .unwrap();
    assert!(!failure.passed());
    assert!(failure.message.contains("actual soft deletes: 3"));
    std::fs::remove_dir_all(&bad_dir).ok();
}

fn read_commit(dir: &FsDirectory) -> segment_infos::SegmentCommitInfo {
    segment_infos::read_latest(dir)
        .expect("read real segments_N")
        .segments[0]
        .clone()
}

/// Hand-construct a `SegmentCommitInfo` with a wrong `del_count` (copied
/// from the real fixture but mutated) and confirm the cardinality check
/// reports a clear failure, not a panic or a false pass -- the fixture's
/// `.liv` bytes on disk are untouched, only the in-memory commit info
/// this module cross-checks against is wrong.
#[test]
fn wrong_del_count_fails_cardinality_check_with_clear_message() {
    let dir = FsDirectory::open(fixture_dir("live_docs_index"));
    let mut commit = read_commit(&dir);
    assert_eq!(commit.del_count, 2, "fixture's real recorded del_count");
    commit.del_count = 4; // wrong on purpose

    let result = check_segment(&dir, &commit);
    assert!(!result.all_passed());
    let failure = result
        .checks
        .iter()
        .find(|c| c.name == "liv.cardinality_matches_del_count")
        .expect("cardinality check must have run");
    assert!(!failure.passed());
    assert!(failure.message.contains("del_count"));

    // Every other check on this segment (files, .fnm, stored fields)
    // must still have run and passed -- one wrong field must not
    // suppress unrelated checks.
    assert!(result
        .checks
        .iter()
        .filter(|c| c.name != "liv.cardinality_matches_del_count")
        .all(|c| c.passed()));
}

/// Truncating the `.liv` file's bytes (fewer bits than `si.doc_count`
/// implies) must fail the max_doc-consistency check, not panic.
#[test]
fn truncated_liv_file_fails_max_doc_check() {
    let src_dir = fixture_dir("live_docs_index");
    let dst_dir = tempdir();
    for entry in std::fs::read_dir(&src_dir).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), dst_dir.join(entry.file_name())).unwrap();
    }

    let dir = FsDirectory::open(&dst_dir);
    let commit = read_commit(&dir);
    let liv_name = liv_file_name(&commit.segment_name, commit.del_gen);
    let liv_path = dst_dir.join(&liv_name);
    let mut bytes = std::fs::read(&liv_path).unwrap();
    // Truncate well below the footer length so the .liv fails to parse
    // at all (a byte-count too short to even hold a header/footer) --
    // this must surface as a `liv.open` failure, not a panic.
    bytes.truncate(4);
    std::fs::write(&liv_path, &bytes).unwrap();

    let result = check_segment(&dir, &commit);
    assert!(!result.all_passed());
    let failure = result
        .checks
        .iter()
        .find(|c| c.name == "liv.open")
        .expect("liv.open check must have run");
    assert!(!failure.passed());

    std::fs::remove_dir_all(&dst_dir).ok();
}

/// A segment whose `.si` lists a file that does not actually exist on
/// disk must be flagged by the "every listed file exists" check, not
/// error out the whole directory-level walk.
#[test]
fn missing_listed_file_fails_files_exist_check() {
    use crate::segment_info;

    let dst_dir = tempdir();
    let dir = FsDirectory::open(&dst_dir);

    let si = SegmentInfo {
        id: [7u8; ID_LENGTH],
        version: segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        min_version: None,
        doc_count: 1,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: vec!["_0.fnm".to_string(), "_0.missing".to_string()],
        attributes: vec![],
        index_sort: None,
    };
    let si_bytes = segment_info::write(&si, "");
    std::fs::write(dst_dir.join("_0.si"), &si_bytes).unwrap();
    // No .fnm nor .missing actually written.

    let commit = segment_infos::SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id: [7u8; ID_LENGTH],
        codec_name: "Lucene104".to_string(),
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    };

    let result = check_segment(&dir, &commit);
    assert!(!result.all_passed());
    let file_failures: Vec<_> = result
        .checks
        .iter()
        .filter(|c| c.name.starts_with("file:") && !c.passed())
        .collect();
    assert_eq!(file_failures.len(), 2);

    std::fs::remove_dir_all(&dst_dir).ok();
}

/// `.si` itself failing to parse must short-circuit every other check
/// (nothing else can be trusted without a valid file list) -- and must
/// **say so**, family by family, rather than returning one failure and
/// sixteen silences. `all_passed()` is false either way here, but a
/// caller counting failures would otherwise see one problem where
/// sixteen classes of invariant went unchecked.
#[test]
fn corrupt_si_short_circuits_remaining_checks() {
    let dst_dir = tempdir();
    let dir = FsDirectory::open(&dst_dir);
    std::fs::write(dst_dir.join("_0.si"), b"not a real .si file").unwrap();

    let commit = segment_infos::SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id: [9u8; ID_LENGTH],
        codec_name: "Lucene104".to_string(),
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    };

    let result = check_segment(&dir, &commit);
    assert_eq!(result.checks[0].name, "si.open");
    assert_eq!(result.checks[0].outcome, Outcome::Failed);
    // Exactly one failure; everything else is named as *not run*.
    assert_eq!(
        result
            .checks
            .iter()
            .filter(|c| c.outcome == Outcome::Failed)
            .count(),
        1
    );
    assert_eq!(result.skipped().len(), FAMILIES_BELOW_SI.len());
    assert!(result
        .skipped()
        .iter()
        .all(|c| c.message.contains("not run: si.open failed")));
    assert!(!result.all_passed());
    // The families a caller most needs to know went unchecked.
    let skipped: Vec<&str> = result.skipped().iter().map(|c| c.name.as_str()).collect();
    for family in ["postings.*", "doc_values.*", "norms.*", "term_vectors.*"] {
        assert!(skipped.contains(&family), "{skipped:?}");
    }

    std::fs::remove_dir_all(&dst_dir).ok();
}

/// A `.fnm` claiming doc-values with no matching `.dvd`/`.dvm` file
/// present must be flagged as an orphaned claim, not silently ignored --
/// exercises the "claims but no files" branch of the field-flags
/// cross-check without needing a whole hand-built doc-values fixture.
#[test]
fn field_claiming_doc_values_without_files_is_flagged() {
    use lucene_codecs::field_infos::{DocValuesType, FieldInfo, IndexOptions};

    let field = FieldInfo {
        name: "f".to_string(),
        number: 0,
        store_term_vectors: false,
        omit_norms: true,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: IndexOptions::None,
        doc_values_type: DocValuesType::Numeric,
        doc_values_skip_index_type: field_infos::DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: field_infos::VectorEncoding::Float32,
        vector_similarity_function: field_infos::VectorSimilarityFunction::Euclidean,
    };
    let fields = FieldInfos {
        fields: vec![field],
    };
    let mut checks = Vec::new();
    // No .dvd/.dvm in this file list.
    check_field_flags_vs_files(&fields, &["_0.fnm".to_string()], &mut checks);

    let failure = checks
        .iter()
        .find(|c| c.name == "fnm.doc_values_vs_files")
        .expect("doc values check must have run");
    assert!(!failure.passed());
    assert!(failure.message.contains("claims"));
}

/// The reverse orphan direction: `.dvd`/`.dvm` files present but no
/// field in `.fnm` claims doc-values at all.
#[test]
fn doc_values_files_without_any_claiming_field_is_flagged() {
    let fields = FieldInfos { fields: vec![] };
    let mut checks = Vec::new();
    check_field_flags_vs_files(
        &fields,
        &["_0.dvd".to_string(), "_0.dvm".to_string()],
        &mut checks,
    );

    let failure = checks
        .iter()
        .find(|c| c.name == "fnm.doc_values_vs_files")
        .expect("doc values check must have run");
    assert!(!failure.passed());
    assert!(failure.message.contains("no field claims"));
}

/// `CheckResult::failures()` must return exactly the failed checks, in
/// order, for a mixed pass/fail result.
#[test]
fn check_result_failures_filters_correctly() {
    let result = CheckResult {
        segment_name: "_0".to_string(),
        max_doc: Some(3),
        checks: vec![Check::pass("a"), Check::fail("b", "bad"), Check::pass("c")],
        stats: CheckStats::default(),
    };
    assert!(!result.all_passed());
    let failures = result.failures();
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].name, "b");
}

/// A file listed in `.si` that opens but has a corrupted/missing codec
/// footer must fail the files-exist-and-validate check with a clear
/// message, not panic -- exercises the `retrieve_checksum` failure
/// branch (as opposed to the file simply not existing at all).
#[test]
fn file_with_corrupted_footer_fails_files_check() {
    let dst_dir = tempdir();
    let dir = FsDirectory::open(&dst_dir);
    // Long enough to have a plausible footer position, but the footer
    // bytes themselves are garbage.
    std::fs::write(dst_dir.join("_0.junk"), vec![0u8; 32]).unwrap();

    let si = SegmentInfo {
        id: [3u8; ID_LENGTH],
        version: segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        min_version: None,
        doc_count: 1,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: vec!["_0.junk".to_string()],
        attributes: vec![],
        index_sort: None,
    };
    let commit = SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id: si.id,
        codec_name: "Lucene104".to_string(),
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    };
    let mut checks = Vec::new();
    check_files_exist_and_validate(&dir, &commit, &si, &mut checks);
    // [0] is the `.si` self-reference check (this hand-built `.si` does
    // not list itself), [1] is the `_0.junk` footer failure.
    assert_eq!(checks.len(), 2);
    assert_eq!(checks[0].name, "si.files_lists_itself");
    assert!(!checks[0].passed());
    assert_eq!(checks[1].name, "file:_0.junk");
    assert!(!checks[1].passed());

    std::fs::remove_dir_all(&dst_dir).ok();
}

/// A segment with `del_gen != -1` (deletions expected) but no `.liv`
/// file actually on disk must fail `liv.open` with the underlying I/O
/// error, not panic.
#[test]
fn missing_liv_file_fails_liv_open() {
    let dst_dir = tempdir();
    let dir = FsDirectory::open(&dst_dir);

    let si = SegmentInfo {
        id: [4u8; ID_LENGTH],
        version: segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        min_version: None,
        doc_count: 4,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: vec![],
        attributes: vec![],
        index_sort: None,
    };
    let commit = segment_infos::SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id: [4u8; ID_LENGTH],
        codec_name: "Lucene104".to_string(),
        del_gen: 1,
        del_count: 1,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    };

    let mut checks = Vec::new();
    check_live_docs(&dir, &commit, &si, &mut checks);
    assert_eq!(checks[0].name, "liv.open");
    assert_eq!(checks[0].outcome, Outcome::Failed);
    // ... and the two checks that needed the file are reported as not
    // run, rather than vanishing from the result.
    assert_eq!(
        checks[1..]
            .iter()
            .map(|c| (c.name.as_str(), c.outcome))
            .collect::<Vec<_>>(),
        [
            ("liv.max_doc_matches_si", Outcome::Skipped),
            ("liv.cardinality_matches_del_count", Outcome::Skipped),
        ]
    );

    std::fs::remove_dir_all(&dst_dir).ok();
}

/// A `.liv` file whose byte size implies fewer words than `.si`'s
/// `doc_count` requires must fail the independent `liv.max_doc_matches_si`
/// size check, and then also fail `liv.open` when `live_docs::parse`
/// itself runs out of bytes trying to read the (wrongly) larger bit
/// array -- two related but distinct failures from one root cause, not
/// a panic.
#[test]
fn liv_size_mismatch_fails_max_doc_check_and_parse() {
    use lucene_util::fixed_bit_set::FixedBitSet;

    let mut bits = FixedBitSet::new(4);
    bits.set(0);
    bits.set(1);
    bits.set(2);
    bits.set(3);
    let segment_id = [5u8; ID_LENGTH];
    let liv_bytes = live_docs::write(&bits, &segment_id, 1, 0).unwrap();

    let dst_dir = tempdir();
    let dir = FsDirectory::open(&dst_dir);
    std::fs::write(dst_dir.join(liv_file_name("_0", 1)), &liv_bytes).unwrap();

    let si = SegmentInfo {
        id: segment_id,
        version: segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        min_version: None,
        // 100 bits needs 2 words; the real .liv above only has 1 word
        // (built for max_doc=4).
        doc_count: 100,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: vec![],
        attributes: vec![],
        index_sort: None,
    };
    let commit = segment_infos::SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id,
        codec_name: "Lucene104".to_string(),
        del_gen: 1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    };

    let mut checks = Vec::new();
    check_live_docs(&dir, &commit, &si, &mut checks);
    let size_check = checks
        .iter()
        .find(|c| c.name == "liv.max_doc_matches_si")
        .expect("size check must have run");
    assert!(!size_check.passed());
    let open_check = checks
        .iter()
        .find(|c| c.name == "liv.open")
        .expect("liv.open must have run");
    assert!(!open_check.passed());

    std::fs::remove_dir_all(&dst_dir).ok();
}

/// A segment listing only some of `.fdt`/`.fdx`/`.fdm` (not all three,
/// not none) must be flagged as an inconsistent file set, not silently
/// skipped or panic.
#[test]
fn partial_stored_fields_file_set_is_flagged() {
    let dst_dir = tempdir();
    let dir = FsDirectory::open(&dst_dir);
    let si = SegmentInfo {
        id: [6u8; ID_LENGTH],
        version: segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        min_version: None,
        doc_count: 1,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: vec!["_0.fdt".to_string()], // missing .fdx/.fdm
        attributes: vec![],
        index_sort: None,
    };
    let commit = segment_infos::SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id: [6u8; ID_LENGTH],
        codec_name: "Lucene104".to_string(),
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    };

    let mut checks = Vec::new();
    check_stored_fields_doc_count(&dir, &commit, &si, &mut checks);
    assert_eq!(checks.len(), 1);
    assert!(!checks[0].passed());
    assert!(checks[0].message.contains("some but not all"));

    std::fs::remove_dir_all(&dst_dir).ok();
}

/// Real `.fdt`/`.fdx`/`.fdm` bytes (copied from the `blocktree_index`
/// fixture) but a deliberately wrong `.si` `doc_count` must fail the
/// doc-count cross-check with a clear message, not panic -- the
/// stored-fields half of the same "wrong recorded count" family of
/// tests as `wrong_del_count_fails_cardinality_check_with_clear_message`.
#[test]
fn stored_fields_doc_count_mismatch_is_flagged() {
    let src_dir = fixture_dir("blocktree_index");
    let dir = FsDirectory::open(&src_dir);
    let commit = read_commit(&dir);

    let mut si = open_si(&dir, &commit).expect("real .si parses");
    assert_ne!(si.doc_count, 999);
    si.doc_count = 999; // wrong on purpose; real .fdt/.fdx/.fdm untouched

    let mut checks = Vec::new();
    check_stored_fields_doc_count(&dir, &commit, &si, &mut checks);
    // Two checks: the doc-count cross-check (which must fail) and the
    // decode-every-document pass (which must still succeed -- the .fdt
    // bytes are genuinely fine, only `.si` lies).
    assert_eq!(checks.len(), 2);
    assert_eq!(checks[0].name, "stored_fields.doc_count_matches_si");
    assert!(!checks[0].passed());
    assert!(checks[0].message.contains("999"));
    assert_eq!(checks[1].name, "stored_fields.every_doc_decodes");
    assert!(checks[1].passed());
}

// -- postings term-by-term re-derivation (task: "revisit scope") --

const POSTINGS_SEG_ID: [u8; ID_LENGTH] = [11u8; ID_LENGTH];
const POSTINGS_SUFFIX: &str = "Lucene104_0";

fn postings_field_info(index_options: field_infos::IndexOptions) -> field_infos::FieldInfo {
    field_infos::FieldInfo {
        name: "body".to_string(),
        number: 0,
        store_term_vectors: false,
        omit_norms: true,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options,
        doc_values_type: field_infos::DocValuesType::None,
        doc_values_skip_index_type: field_infos::DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: field_infos::VectorEncoding::Float32,
        vector_similarity_function: field_infos::VectorSimilarityFunction::Euclidean,
    }
}

/// Writes a minimal, self-contained, non-compound one-field segment
/// (`.si`/`.fnm`/`.tim`/`.tip`/`.tmd`/`.doc`) into `dst_dir` from
/// `postings_writer::write_single_field`'s output, and returns the
/// `SegmentCommitInfo` to open it with. `doc_bytes_override` lets a test
/// substitute a *different* `.doc` buffer than the one that naturally
/// matches `terms` -- the mechanism the corruption test below uses to
/// build a term dictionary that claims one `totalTermFreq` while the
/// actual `.doc` bytes sum to a different one, without any raw byte
/// surgery.
fn write_postings_fixture(
    dst_dir: &std::path::Path,
    terms: &[lucene_codecs::postings_writer::TermPostings],
    field_doc_count: i32,
    max_doc: i32,
    doc_bytes_override: Option<&[u8]>,
) -> segment_infos::SegmentCommitInfo {
    use lucene_codecs::field_infos::IndexOptions;
    use lucene_codecs::postings_writer::{write_single_field, FieldPostingsInput};

    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: field_doc_count,
        has_payloads: false,
        terms,
    };
    let output = write_single_field(&input, &POSTINGS_SEG_ID, POSTINGS_SUFFIX)
        .expect("hand-built postings must write cleanly");
    let doc_bytes = doc_bytes_override.unwrap_or(&output.doc);

    let fields = field_infos::write(
        &[postings_field_info(IndexOptions::DocsAndFreqs)],
        &POSTINGS_SEG_ID,
        "",
    );
    std::fs::write(dst_dir.join("_0.fnm"), &fields).unwrap();
    std::fs::write(
        dst_dir.join(format!("_0_{POSTINGS_SUFFIX}.tim")),
        &output.tim,
    )
    .unwrap();
    std::fs::write(
        dst_dir.join(format!("_0_{POSTINGS_SUFFIX}.tip")),
        &output.tip,
    )
    .unwrap();
    std::fs::write(
        dst_dir.join(format!("_0_{POSTINGS_SUFFIX}.tmd")),
        &output.tmd,
    )
    .unwrap();
    std::fs::write(dst_dir.join(format!("_0_{POSTINGS_SUFFIX}.doc")), doc_bytes).unwrap();

    let si = SegmentInfo {
        id: POSTINGS_SEG_ID,
        version: segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        min_version: None,
        doc_count: max_doc,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: vec![
            // Real Lucene's `.si` always lists itself (the format's
            // `write` does `si.addFile(fileName)`), so a hand-built
            // fixture must too or `si.files_lists_itself` fails.
            "_0.si".to_string(),
            "_0.fnm".to_string(),
            format!("_0_{POSTINGS_SUFFIX}.tim"),
            format!("_0_{POSTINGS_SUFFIX}.tip"),
            format!("_0_{POSTINGS_SUFFIX}.tmd"),
            format!("_0_{POSTINGS_SUFFIX}.doc"),
        ],
        attributes: vec![],
        index_sort: None,
    };
    std::fs::write(dst_dir.join("_0.si"), segment_info::write(&si, "")).unwrap();

    segment_infos::SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id: POSTINGS_SEG_ID,
        codec_name: "Lucene104".to_string(),
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    }
}

/// A hand-built one-field segment with **positions, offsets and
/// payloads**, big enough that at least one term's position stream fills
/// a whole 128-value `for_util` block (the packed path, as opposed to
/// the vint tail). The positional negative controls need that: a
/// corrupted vint tail can only ever make positions *larger*, because
/// vint deltas are non-negative and positions accumulate, whereas a
/// corrupted packed block's bits-per-value header makes the decoder read
/// arbitrary 32-bit values -- which is exactly the "position out of
/// bounds / before the previous one / past MAX_POSITION" family Java's
/// `checkFields` rejects.
fn write_positional_postings_fixture(
    dst_dir: &std::path::Path,
    occurrences_in_first_doc: usize,
) -> segment_infos::SegmentCommitInfo {
    use lucene_codecs::field_infos::IndexOptions;
    use lucene_codecs::postings_writer::{write_single_field, FieldPostingsInput, TermPostings};

    const OPTS: IndexOptions = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
    let big: Vec<i32> = (0..occurrences_in_first_doc as i32)
        .map(|i| i * 3)
        .collect();
    let big_offsets: Vec<(i32, i32)> = big.iter().map(|&p| (p * 4, p * 4 + 3)).collect();
    let big_payloads: Vec<Vec<u8>> = big.iter().map(|&p| vec![(p % 251) as u8; 2]).collect();
    let terms = vec![
        TermPostings {
            term: b"alpha".to_vec(),
            docs: vec![(0, occurrences_in_first_doc as i32), (2, 2)],
            positions: vec![big, vec![1, 7]],
            offsets: vec![big_offsets, vec![(4, 7), (28, 31)]],
            payload_bytes: big_payloads
                .iter()
                .flatten()
                .copied()
                .chain([9u8, 9])
                .collect(),
            payload_lengths: big_payloads
                .iter()
                .map(|p| p.len() as u32)
                .chain([1, 1])
                .collect(),
        },
        TermPostings {
            term: b"beta".to_vec(),
            docs: vec![(1, 3)],
            positions: vec![vec![0, 5, 9]],
            offsets: vec![vec![(0, 4), (20, 24), (36, 40)]],
            payload_bytes: vec![1, 2, 3],
            payload_lengths: vec![1, 1, 1],
        },
    ];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: OPTS,
        doc_count: 3,
        has_payloads: true,
        terms: &terms,
    };
    let output = write_single_field(&input, &POSTINGS_SEG_ID, POSTINGS_SUFFIX)
        .expect("hand-built positional postings must write cleanly");

    let mut fi = postings_field_info(OPTS);
    fi.store_payloads = true;
    let fields = field_infos::write(&[fi], &POSTINGS_SEG_ID, "");
    std::fs::write(dst_dir.join("_0.fnm"), &fields).unwrap();
    for (ext, bytes) in [
        ("tim", &output.tim),
        ("tip", &output.tip),
        ("tmd", &output.tmd),
        ("doc", &output.doc),
        ("pos", &output.pos),
        ("pay", &output.pay),
    ] {
        std::fs::write(
            dst_dir.join(format!("_0_{POSTINGS_SUFFIX}.{ext}")),
            bytes.as_slice(),
        )
        .unwrap();
    }

    let si = SegmentInfo {
        id: POSTINGS_SEG_ID,
        version: segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        min_version: None,
        doc_count: 3,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: ["_0.si".to_string(), "_0.fnm".to_string()]
            .into_iter()
            .chain(
                ["tim", "tip", "tmd", "doc", "pos", "pay"]
                    .iter()
                    .map(|e| format!("_0_{POSTINGS_SUFFIX}.{e}")),
            )
            .collect(),
        attributes: vec![],
        index_sort: None,
    };
    std::fs::write(dst_dir.join("_0.si"), segment_info::write(&si, "")).unwrap();

    segment_infos::SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id: POSTINGS_SEG_ID,
        codec_name: "Lucene104".to_string(),
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    }
}

/// The hand-built positional fixture must pass every check cleanly --
/// the "no false positives" side of the positional negative controls
/// below.
#[test]
fn hand_built_positional_postings_pass_every_check() {
    let dst = tempdir();
    let commit = write_positional_postings_fixture(&dst, 300);
    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    assert!(
        result.all_passed(),
        "unexpected failures: {:?}",
        result.failures()
    );
    let names: Vec<&str> = result.checks.iter().map(|c| c.name.as_str()).collect();
    assert!(names.contains(&"postings.positions_valid:body"));
    assert!(names.contains(&"postings.offsets_valid:body"));
    // 305 occurrences across 3 docs, all live.
    assert_eq!(result.stats.token_count, 305);
    assert_eq!(result.stats.term_doc_pairs, 3);
    assert_eq!(result.stats.term_count, 2);
}

/// Negative control for `postings.positions_valid` and
/// `postings.offsets_valid`, driven directly at the predicate: a writer
/// that emits a negative position, a decreasing position, a position
/// past `IndexWriter.MAX_POSITION`, a negative or decreasing
/// `startOffset`, or an `endOffset` before its `startOffset` must be
/// reported -- and clean occurrences must not be. See
/// [`check_occurrences`]' own doc comment for why this is driven at the
/// predicate rather than by corrupting bytes.
#[test]
fn bad_positions_and_offsets_are_reported_by_the_predicate() {
    use lucene_codecs::postings::Position;
    let occ = |position, start_offset, end_offset| Position {
        position,
        start_offset,
        end_offset,
        payload: Vec::new(),
    };

    // Clean input: no complaint from either family.
    let mut pos = Vec::new();
    let mut off = Vec::new();
    check_occurrences(
        "body",
        b"t",
        0,
        &[occ(0, 0, 4), occ(5, 10, 14), occ(5, 10, 12)],
        true,
        &mut pos,
        &mut off,
    );
    assert!(pos.is_empty() && off.is_empty(), "{pos:?} {off:?}");

    // Each of Java's six rejected shapes, checked against the family
    // that is supposed to name it. (A bad position also perturbs the
    // offset sequence and vice versa, so the assertion is "the right
    // family complains", not "only that family does".)
    for (bad, wants_position, wants_offset) in [
        (occ(-1, 9, 11), true, false),
        (occ(MAX_POSITION + 1, 9, 11), true, false),
        (occ(3, 9, 11), true, false), // < the 7 planted before it
        (occ(9, -1, 1), false, true),
        (occ(9, 9, -1), false, true),
        (occ(9, 9, 3), false, true),
        (occ(9, 1, 5), false, true), // startOffset before the previous one
    ] {
        let mut pos = Vec::new();
        let mut off = Vec::new();
        check_occurrences(
            "body",
            b"t",
            7,
            &[occ(7, 8, 12), bad.clone()],
            true,
            &mut pos,
            &mut off,
        );
        if wants_position {
            assert!(!pos.is_empty(), "position family missed {bad:?}");
        }
        if wants_offset {
            assert!(!off.is_empty(), "offset family missed {bad:?}");
        }
    }
}

/// The complement of the predicate test: no single-byte corruption of a
/// real packed `.pos`/`.pay` may be silently accepted. It is caught one
/// layer lower than `postings.positions_valid` -- the `for_util`/
/// `SliceInput` decoders reject a bad bits-per-value or block length
/// before a position is ever produced, so the failure surfaces as
/// `postings.terms_decode` or as the file's
/// own CRC. That is the property worth pinning: nothing gets through.
#[test]
fn no_single_byte_corruption_of_pos_or_pay_is_silently_accepted() {
    let dst = tempdir();
    let commit = write_positional_postings_fixture(&dst, 300);
    let pos_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.pos"));
    let pay_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.pay"));
    let pos_original = std::fs::read(&pos_path).unwrap();
    let pay_original = std::fs::read(&pay_path).unwrap();

    for (path, original) in [(&pos_path, &pos_original), (&pay_path, &pay_original)] {
        for off in 48..(original.len() - lucene_store::codec_util::FOOTER_LENGTH) {
            for mask in [0x01u8, 0x40, 0x80, 0xff] {
                let mut bytes = original.clone();
                bytes[off] ^= mask;
                std::fs::write(path, &bytes).unwrap();
                let dir = FsDirectory::open(&dst);
                let result = check_segment(&dir, &commit);
                assert!(
                    !result.all_passed(),
                    "flipping {path:?} byte {off} with {mask:#x} was silently accepted"
                );
            }
            std::fs::write(path, original).unwrap();
        }
    }
    std::fs::write(&pos_path, &pos_original).unwrap();
    std::fs::write(&pay_path, &pay_original).unwrap();
    let dir = FsDirectory::open(&dst);
    assert!(check_segment(&dir, &commit).all_passed());
}

/// The real `blocktree_index` fixture (genuine Java-written postings, a
/// mix of singleton and multi-doc terms) must pass the new re-derivation
/// checks cleanly -- the "no false positives on real data" baseline,
/// same role `valid_blocktree_fixture_passes_every_check` plays for the
/// rest of this module.
#[test]
fn valid_blocktree_fixture_passes_postings_re_derivation() {
    let dir = FsDirectory::open(fixture_dir("blocktree_index"));
    let results = check_directory(&dir).expect("read segments_N");
    let result = &results[1];
    assert!(
        result.all_passed(),
        "unexpected failures: {:?}",
        result.failures()
    );
    assert!(
        result
            .checks
            .iter()
            .any(|c| c.name.starts_with("postings.total_term_freq:") && c.passed()),
        "expected a passing postings.total_term_freq:<field> check, got: {:?}",
        result.checks
    );
    assert!(
        result
            .checks
            .iter()
            .any(|c| c.name.starts_with("postings.doc_ids_valid:") && c.passed()),
        "expected a passing postings.doc_ids_valid:<field> check, got: {:?}",
        result.checks
    );
}

/// A hand-built, genuinely self-consistent segment (real writer output,
/// not raw-byte surgery) must pass the re-derivation checks -- proves
/// the machinery works on this port's own writer output, not just the
/// one real-Lucene fixture above.
#[test]
fn hand_built_consistent_postings_pass_re_derivation() {
    use lucene_codecs::postings_writer::TermPostings;

    let terms = vec![
        TermPostings {
            term: b"apple".to_vec(),
            docs: vec![(0, 2), (2, 1), (5, 3)],
            ..Default::default()
        },
        TermPostings {
            term: b"kiwi".to_vec(),
            docs: vec![(1, 1)], // singleton
            ..Default::default()
        },
    ];

    // Distinct docs across both terms: {0, 1, 2, 5} -> field doc_count 4;
    // max_doc must exceed the highest doc ID (5) -> 6.
    let dst_dir = tempdir();
    let commit = write_postings_fixture(&dst_dir, &terms, 4, 6, None);
    let dir = FsDirectory::open(&dst_dir);

    let result = check_segment(&dir, &commit);
    assert!(
        result.all_passed(),
        "unexpected failures: {:?}",
        result.failures()
    );

    std::fs::remove_dir_all(&dst_dir).ok();
}

/// The actual proof this check does something real: the term
/// dictionary (`.tim`/`.tip`/`.tmd`) is built from `claimed_terms` (which
/// says term `"apple"`'s `totalTermFreq` is 60), but the `.doc` bytes it
/// points at are swapped for `actual_terms`' real postings (whose three
/// per-doc freqs actually sum to 6) -- same doc IDs and doc count in
/// both (so decoding itself succeeds cleanly; this is a metadata/data
/// disagreement, not a corrupt/truncated file), yet
/// `postings.total_term_freq:body` must fail and name the exact
/// mismatch, while `postings.doc_ids_valid:body` (an unrelated
/// dimension) must still pass -- proving the new check independently
/// recomputes from the actual postings rather than trusting the
/// dictionary's own claim.
#[test]
fn corrupted_total_term_freq_is_caught_by_re_derivation() {
    use lucene_codecs::field_infos::IndexOptions;
    use lucene_codecs::postings_writer::{write_single_field, FieldPostingsInput, TermPostings};

    let actual_terms = vec![TermPostings {
        term: b"apple".to_vec(),
        docs: vec![(0, 2), (2, 1), (5, 3)], // real per-doc freqs, sum = 6
        ..Default::default()
    }];
    let claimed_terms = vec![TermPostings {
        term: b"apple".to_vec(),
        // Same doc IDs/doc count (docFreq stays consistent -- this test
        // isolates totalTermFreq disagreement), different per-doc freqs
        // so the dictionary's recorded totalTermFreq (60) disagrees with
        // what the swapped-in real `.doc` bytes below actually contain.
        docs: vec![(0, 20), (2, 10), (5, 30)],
        ..Default::default()
    }];

    // Distinct docs {0, 2, 5} -> field doc_count 3; max_doc must exceed
    // the highest doc ID (5) -> 6.
    let actual_output = write_single_field(
        &FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 3,
            has_payloads: false,
            terms: &actual_terms,
        },
        &POSTINGS_SEG_ID,
        POSTINGS_SUFFIX,
    )
    .unwrap();
    assert!(!actual_output.doc.is_empty());

    let dst_dir = tempdir();
    // `write_postings_fixture` builds .tim/.tip/.tmd from
    // `claimed_terms` (dictionary says totalTermFreq=60) but the `.doc`
    // file on disk is overridden to `actual_output.doc` (real bytes
    // summing to 6) -- both used the same doc IDs/doc_count, so
    // `meta.doc_start_fp` still points at the right offset and decoding
    // succeeds; only the recorded stat disagrees with the real data.
    let commit = write_postings_fixture(&dst_dir, &claimed_terms, 3, 6, Some(&actual_output.doc));
    let dir = FsDirectory::open(&dst_dir);

    let result = check_segment(&dir, &commit);
    assert!(!result.all_passed());

    let freq_check = result
        .checks
        .iter()
        .find(|c| c.name == "postings.total_term_freq:body")
        .expect("total_term_freq check must have run");
    assert!(!freq_check.passed());
    assert!(freq_check.message.contains("totalTermFreq=60"));
    assert!(freq_check.message.contains("sum to 6"));

    // An unrelated dimension (doc ID validity) must still pass -- one
    // wrong stat must not suppress or corrupt an unrelated check.
    let doc_ids_check = result
        .checks
        .iter()
        .find(|c| c.name == "postings.doc_ids_valid:body")
        .expect("doc_ids_valid check must have run");
    assert!(doc_ids_check.passed());

    std::fs::remove_dir_all(&dst_dir).ok();
}

/// The `postings.doc_ids_valid` proxy's own actual proof: unlike
/// `total_term_freq` above, this swaps in `.doc` bytes whose per-doc
/// freqs still sum correctly (so `total_term_freq` passes) but whose
/// decoded doc IDs include one at/past `si.doc_count` -- exactly the
/// "wrong claimed docFreq made the reader wander into unrelated bytes"
/// symptom the doc comment above describes as this check's real
/// purpose. Without this test, the check added specifically to catch
/// docFreq corruption had never actually been exercised on its failure
/// path.
#[test]
fn corrupted_doc_id_is_caught_by_doc_ids_valid_check() {
    use lucene_codecs::field_infos::IndexOptions;
    use lucene_codecs::postings_writer::{write_single_field, FieldPostingsInput, TermPostings};

    let claimed_terms = vec![TermPostings {
        term: b"apple".to_vec(),
        docs: vec![(0, 2), (2, 1), (5, 3)], // sum = 6, doc IDs all < max_doc (6)
        ..Default::default()
    }];
    let actual_terms = vec![TermPostings {
        term: b"apple".to_vec(),
        // Same per-doc freqs in the same order (sum still 6, so
        // total_term_freq must still agree) but the third doc ID is
        // 9, past this segment's max_doc of 6 -- doc_ids_valid must
        // catch it even though total_term_freq does not.
        docs: vec![(0, 2), (2, 1), (9, 3)],
        ..Default::default()
    }];

    let actual_output = write_single_field(
        &FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 3,
            has_payloads: false,
            terms: &actual_terms,
        },
        &POSTINGS_SEG_ID,
        POSTINGS_SUFFIX,
    )
    .unwrap();

    let dst_dir = tempdir();
    let commit = write_postings_fixture(&dst_dir, &claimed_terms, 3, 6, Some(&actual_output.doc));
    let dir = FsDirectory::open(&dst_dir);

    let result = check_segment(&dir, &commit);
    assert!(!result.all_passed());

    let doc_ids_check = result
        .checks
        .iter()
        .find(|c| c.name == "postings.doc_ids_valid:body")
        .expect("doc_ids_valid check must have run");
    assert!(!doc_ids_check.passed());
    assert!(doc_ids_check.message.contains("doc ID 9"));

    // total_term_freq is an unrelated dimension here (both sides sum
    // to 6) -- must still pass, proving the two checks are
    // independent.
    let freq_check = result
        .checks
        .iter()
        .find(|c| c.name == "postings.total_term_freq:body")
        .expect("total_term_freq check must have run");
    assert!(freq_check.passed());

    std::fs::remove_dir_all(&dst_dir).ok();
}

/// A field claiming postings whose segment is missing the `.doc` file
/// (needed for any term with `docFreq > 1`) must be flagged as
/// `postings.doc_open`, not panic -- exercises the "term needs `.doc`
/// bytes but none were found" branch distinctly from a plain I/O error.
#[test]
fn missing_doc_file_for_multi_doc_term_fails_doc_open_check() {
    use lucene_codecs::postings_writer::TermPostings;

    let terms = vec![TermPostings {
        term: b"apple".to_vec(),
        docs: vec![(0, 2), (2, 1), (5, 3)],
        ..Default::default()
    }];

    // Distinct docs {0, 2, 5} -> field doc_count 3; max_doc must exceed
    // the highest doc ID (5) -> 6.
    let dst_dir = tempdir();
    let commit = write_postings_fixture(&dst_dir, &terms, 3, 6, None);

    // Make the .doc file genuinely absent, not just unlisted: delete it
    // from disk *and* drop it from `.si`'s file list, then rewrite
    // `.si` -- so this is "the segment legitimately has no .doc file"
    // from this function's point of view, not an I/O error on an
    // expected file (that's a different, already-covered failure mode).
    let dir_ro = FsDirectory::open(&dst_dir);
    let mut si = open_si(&dir_ro, &commit).expect("hand-built .si parses");
    si.files.retain(|f| !f.ends_with(".doc"));
    std::fs::write(dst_dir.join("_0.si"), segment_info::write(&si, "")).unwrap();
    std::fs::remove_file(dst_dir.join(format!("_0_{POSTINGS_SUFFIX}.doc"))).unwrap();

    let dir = FsDirectory::open(&dst_dir);
    let result = check_segment(&dir, &commit);
    let doc_open_check = result
        .checks
        .iter()
        .find(|c| c.name == "postings.doc_open")
        .expect("postings.doc_open check must have run");
    assert!(!doc_open_check.passed());

    std::fs::remove_dir_all(&dst_dir).ok();
}

// -- points-tree structural invariants --

/// The real `points_index` fixture (genuine Java-written BKD data with
/// a single-dimension field, a 2-dimension field, and a 4-dim/2-index-
/// dim "shape"-style field -- see `GenPoints.java`) must pass every
/// structural-invariant check cleanly, exercising both the plain
/// field-bounds check (all three fields) and the leaf-bounds-subset
/// check (the two multi-index-dim fields only) on real data -- the
/// "no false positives" baseline for this check, mirroring
/// `valid_blocktree_fixture_passes_postings_re_derivation`'s role for
/// postings.
#[test]
fn valid_points_fixture_passes_structural_invariants() {
    let dir = FsDirectory::open(fixture_dir("points_index"));
    let results = check_directory(&dir).expect("read segments_N");
    assert_eq!(results.len(), 2);
    let result = &results[1];
    assert!(
        result.all_passed(),
        "unexpected failures: {:?}",
        result.failures()
    );
    assert!(
        result
            .checks
            .iter()
            .any(|c| c.name.starts_with("points.value_within_field_bounds:") && c.passed()),
        "expected a passing points.value_within_field_bounds:<field> check, got: {:?}",
        result.checks
    );
    assert!(
        result
            .checks
            .iter()
            .any(|c| c.name.starts_with("points.leaf_bounds_subset_of_field:") && c.passed()),
        "expected a passing points.leaf_bounds_subset_of_field:<field> check \
             (the fixture has multi-index-dim fields), got: {:?}",
        result.checks
    );
    assert!(
        result
            .checks
            .iter()
            .any(|c| c.name.starts_with("points.point_count_matches:") && c.passed()),
        "expected a passing points.point_count_matches:<field> check, got: {:?}",
        result.checks
    );
}

const POINTS_SEG_ID: [u8; ID_LENGTH] = [13u8; ID_LENGTH];

fn points_field_info() -> field_infos::FieldInfo {
    field_infos::FieldInfo {
        name: "loc".to_string(),
        number: 5,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: field_infos::IndexOptions::None,
        doc_values_type: field_infos::DocValuesType::None,
        doc_values_skip_index_type: field_infos::DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 1,
        point_index_dimension_count: 1,
        point_num_bytes: 8,
        vector_dimension: 0,
        vector_encoding: field_infos::VectorEncoding::Float32,
        vector_similarity_function: field_infos::VectorSimilarityFunction::Euclidean,
    }
}

/// Writes a minimal, self-contained, non-compound one-field segment
/// (`.si`/`.fnm`/`.kdm`/`.kdi`/`.kdd`) from
/// [`lucene_codecs::points::write`]'s output for a single-dimension,
/// 8-byte-per-value field with the given `(docID, packedValue)` pairs,
/// and returns the `SegmentCommitInfo` to open it with. `kdd_override`
/// lets a test substitute a *different* `.kdd` buffer than the one that
/// naturally matches `points` -- the same "swap the on-disk bytes for a
/// legitimately-decodable but wrong one" mechanism
/// `write_postings_fixture` uses for its own corruption tests, letting
/// this test hand-corrupt a decoded point value without any raw-offset
/// byte surgery.
fn write_points_fixture(
    dst_dir: &std::path::Path,
    points: &[(i32, Vec<u8>)],
    max_doc: i32,
    kdd_override: Option<&[u8]>,
) -> segment_infos::SegmentCommitInfo {
    write_points_fixture_dims(dst_dir, points, max_doc, 1, 8, kdd_override, &|_| {})
}

/// [`write_points_fixture`] with the dimension count and a `.kdm` editor
/// exposed.
///
/// `points.leaf_bounds_subset_of_field` is skipped for a single-dimension
/// field, and every other hand-built points fixture here is one.
/// `kdm_edit` is the same idea as `kdd_override` applied to the
/// metadata: the field summary a `.kdm` records (`pointCount`,
/// `docCount`) is computed by the writer from the points it is given, so
/// it cannot be *asked* for a summary that lies -- which is precisely
/// what `points.doc_count_matches` exists to reject.
fn write_points_fixture_dims(
    dst_dir: &std::path::Path,
    points: &[(i32, Vec<u8>)],
    max_doc: i32,
    num_index_dims: i32,
    bytes_per_dim: i32,
    kdd_override: Option<&[u8]>,
    kdm_edit: &dyn Fn(&mut Vec<u8>),
) -> segment_infos::SegmentCommitInfo {
    use lucene_codecs::points::WritePointsField;

    let field = WritePointsField {
        field_number: 5,
        num_dims: num_index_dims,
        num_index_dims,
        bytes_per_dim,
        points: points.to_vec(),
    };
    let (mut kdm, kdi, kdd) = points::write(
        &[field],
        points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
        &POINTS_SEG_ID,
        "",
    )
    .expect("hand-built points must write cleanly");
    kdm_edit(&mut kdm);
    let kdd = kdd_override.unwrap_or(&kdd);

    let mut info = points_field_info();
    info.point_dimension_count = num_index_dims;
    info.point_index_dimension_count = num_index_dims;
    info.point_num_bytes = bytes_per_dim;
    let fields = field_infos::write(&[info], &POINTS_SEG_ID, "");
    std::fs::write(dst_dir.join("_0.fnm"), &fields).unwrap();
    std::fs::write(dst_dir.join("_0.kdm"), &kdm).unwrap();
    std::fs::write(dst_dir.join("_0.kdi"), &kdi).unwrap();
    std::fs::write(dst_dir.join("_0.kdd"), kdd).unwrap();

    let si = SegmentInfo {
        id: POINTS_SEG_ID,
        version: segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        min_version: None,
        doc_count: max_doc,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: vec![
            "_0.si".to_string(),
            "_0.fnm".to_string(),
            "_0.kdm".to_string(),
            "_0.kdi".to_string(),
            "_0.kdd".to_string(),
        ],
        attributes: vec![],
        index_sort: None,
    };
    std::fs::write(dst_dir.join("_0.si"), segment_info::write(&si, "")).unwrap();

    segment_infos::SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id: POINTS_SEG_ID,
        codec_name: "Lucene104".to_string(),
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    }
}

/// A hand-built, genuinely self-consistent points segment (real writer
/// output, not raw-byte surgery) must pass the structural-invariant
/// checks -- proves the machinery works on this port's own writer
/// output too, not just the one real-Lucene fixture above.
#[test]
fn hand_built_consistent_points_pass_structural_invariants() {
    let points = vec![
        (0, vec![0, 0, 0, 0, 0, 0, 0, 1]),
        (1, vec![0, 0, 0, 0, 0, 0, 0, 2]),
        (2, vec![0, 0, 0, 0, 0, 0, 0, 3]),
    ];
    let dst_dir = tempdir();
    let commit = write_points_fixture(&dst_dir, &points, 3, None);
    let dir = FsDirectory::open(&dst_dir);

    let result = check_segment(&dir, &commit);
    assert!(
        result.all_passed(),
        "unexpected failures: {:?}",
        result.failures()
    );

    std::fs::remove_dir_all(&dst_dir).ok();
}

/// The actual proof this check does something real: build a segment
/// whose `.kdm` field-level bounds are derived from the *real* points
/// `(0,...,1)`/`(1,...,2)`/`(2,...,3)` (so `min_packed_value`/
/// `max_packed_value` = `...01`/`...03`), but swap the `.kdd` bytes for
/// ones decoded from a *different* point set where the middle point's
/// value is `...FF` -- decoding still succeeds cleanly (same doc IDs,
/// same doc count, same leaf layout), so this is a genuine "declared
/// bound vs. actual decoded value" disagreement, not a truncated/corrupt
/// file. `points.value_within_field_bounds:loc` must fail and name the
/// offending doc, while `points.point_count_matches:loc` (an unrelated
/// dimension: the leaf still decodes exactly 3 points either way) must
/// still pass -- proving the two checks are independent, mirroring
/// `corrupted_total_term_freq_is_caught_by_re_derivation`'s structure
/// for postings.
#[test]
fn corrupted_point_value_is_caught_by_bounds_check() {
    let real_points = vec![
        (0, vec![0, 0, 0, 0, 0, 0, 0, 1]),
        (1, vec![0, 0, 0, 0, 0, 0, 0, 2]),
        (2, vec![0, 0, 0, 0, 0, 0, 0, 3]),
    ];
    let corrupted_points = vec![
        (0, vec![0, 0, 0, 0, 0, 0, 0, 1]),
        (1, vec![0xFF, 0, 0, 0, 0, 0, 0, 0xFF]), // wildly out of [..01, ..03]
        (2, vec![0, 0, 0, 0, 0, 0, 0, 3]),
    ];
    let corrupted_field = lucene_codecs::points::WritePointsField {
        field_number: 5,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points: corrupted_points,
    };
    let (_, _, corrupted_kdd) = points::write(
        &[corrupted_field],
        points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
        &POINTS_SEG_ID,
        "",
    )
    .unwrap();

    let dst_dir = tempdir();
    let commit = write_points_fixture(&dst_dir, &real_points, 3, Some(&corrupted_kdd));
    let dir = FsDirectory::open(&dst_dir);

    let result = check_segment(&dir, &commit);
    assert!(!result.all_passed());

    let bounds_check = result
        .checks
        .iter()
        .find(|c| c.name == "points.value_within_field_bounds:loc")
        .expect("value_within_field_bounds check must have run");
    assert!(!bounds_check.passed());
    assert!(bounds_check.message.contains("doc 1"));

    // Unrelated dimension: the corrupted leaf still decodes exactly 3
    // points, same as the declared point_count, so this must still
    // pass -- one wrong value must not suppress or corrupt an unrelated
    // check.
    let count_check = result
        .checks
        .iter()
        .find(|c| c.name == "points.point_count_matches:loc")
        .expect("point_count_matches check must have run");
    assert!(count_check.passed());

    std::fs::remove_dir_all(&dst_dir).ok();
}

/// A field claiming points in `.fnm` whose segment is missing one of
/// `.kdm`/`.kdi`/`.kdd` must be flagged as `points.open`, not panic --
/// the points analogue of
/// `missing_doc_file_for_multi_doc_term_fails_doc_open_check`.
#[test]
fn missing_points_file_fails_points_open_check() {
    let points = vec![(0, vec![0, 0, 0, 0, 0, 0, 0, 1])];
    let dst_dir = tempdir();
    let commit = write_points_fixture(&dst_dir, &points, 1, None);

    // Make the .kdd file genuinely absent, not just unlisted.
    let dir_ro = FsDirectory::open(&dst_dir);
    let mut si = open_si(&dir_ro, &commit).expect("hand-built .si parses");
    si.files.retain(|f| !f.ends_with(".kdd"));
    std::fs::write(dst_dir.join("_0.si"), segment_info::write(&si, "")).unwrap();
    std::fs::remove_file(dst_dir.join("_0.kdd")).unwrap();

    let dir = FsDirectory::open(&dst_dir);
    let result = check_segment(&dir, &commit);
    let points_open_check = result
        .checks
        .iter()
        .find(|c| c.name == "points.open")
        .expect("points.open check must have run");
    assert!(!points_open_check.passed());

    std::fs::remove_dir_all(&dst_dir).ok();
}

/// A segment where no field claims points at all must skip the points
/// checks entirely (not fail, not run vacuously) -- exercises the
/// early-return branch for a segment this check has nothing to verify
/// about, using the real `blocktree_index` fixture (postings only, no
/// points).
#[test]
fn segment_without_points_fields_skips_points_checks() {
    let dir = FsDirectory::open(fixture_dir("blocktree_index"));
    let results = check_directory(&dir).expect("read segments_N");
    assert!(!results[1]
        .checks
        .iter()
        .any(|c| c.name.starts_with("points.")));
}

/// FieldInfo retains dimensions after a merge deletes every point in
/// the field. The valid points triple then contains no field entry.
#[test]
fn declared_points_field_with_empty_triple_passes() {
    let points = vec![(0, vec![0, 0, 0, 0, 0, 0, 0, 1])];
    let dst_dir = tempdir();
    let commit = write_points_fixture(&dst_dir, &points, 1, None);
    let (kdm, kdi, kdd) = points::write_packed(
        Vec::new(),
        points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
        1,
        &POINTS_SEG_ID,
        "",
    )
    .unwrap();
    std::fs::write(dst_dir.join("_0.kdm"), &kdm).unwrap();
    std::fs::write(dst_dir.join("_0.kdi"), &kdi).unwrap();
    std::fs::write(dst_dir.join("_0.kdd"), &kdd).unwrap();

    let dir = FsDirectory::open(&dst_dir);
    let result = check_segment(&dir, &commit);
    assert!(result.all_passed(), "{:?}", result.failures());
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "points.field_empty:loc")
        .expect("empty field must be recorded");
    assert!(check.passed());
    assert!(!result.checks.iter().any(|c| POINTS_PER_FIELD
        .iter()
        .any(|family| c.name == format!("{family}:loc"))));

    // Null-field handling cannot hide a damaged or absent triple.
    // Corrupt only the checksum, leaving the empty metadata readable.
    let mut corrupt = kdm.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    std::fs::write(dst_dir.join("_0.kdm"), corrupt).unwrap();
    let result = check_segment(&dir, &commit);
    assert!(!result.all_passed());
    assert!(result.failures().iter().any(|c| c.name == "file:_0.kdm"));

    std::fs::write(dst_dir.join("_0.kdm"), kdm).unwrap();
    std::fs::remove_file(dst_dir.join("_0.kdd")).unwrap();
    let result = check_segment(&dir, &commit);
    assert!(result.failures().iter().any(|c| c.name == "points.open"));

    std::fs::remove_dir_all(&dst_dir).ok();
}

/// The leaf's own decoded point count must sum to `.kdm`'s declared
/// `point_count` for the field: build a segment whose `.kdm`/`.kdi`
/// claim 3 points (from `real_points`) but swap in `.kdd` bytes
/// re-encoded from only 2 points -- decoding still succeeds (same
/// field shape, one leaf), but the leaf actually yields 2 points, not
/// 3, so `points.point_count_matches:loc` must fail while
/// `points.value_within_field_bounds:loc` (checked against the
/// looser, real-points-derived field bounds) still passes.
/// Negative control for `points.doc_count_matches`: a `.kdd` whose
/// points carry fewer *distinct* doc IDs than `.kdm`'s declared
/// `docCount`, with the point count and every packed value left intact
/// -- so only Java's `getDocCountSeen() != docCount` check can catch it.
#[test]
fn fewer_distinct_docs_than_declared_fails_doc_count_check() {
    let real_points = vec![
        (0, vec![0, 0, 0, 0, 0, 0, 0, 1]),
        (1, vec![0, 0, 0, 0, 0, 0, 0, 2]),
        (2, vec![0, 0, 0, 0, 0, 0, 0, 3]),
    ];
    // Same three packed values, all on doc 0: point_count is unchanged
    // and every value is still inside the field's declared bounds.
    let same_doc_points = vec![
        (0, vec![0, 0, 0, 0, 0, 0, 0, 1]),
        (0, vec![0, 0, 0, 0, 0, 0, 0, 2]),
        (0, vec![0, 0, 0, 0, 0, 0, 0, 3]),
    ];
    let same_doc_field = lucene_codecs::points::WritePointsField {
        field_number: 5,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points: same_doc_points,
    };
    let (_, _, same_doc_kdd) = points::write(
        &[same_doc_field],
        points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
        &POINTS_SEG_ID,
        "",
    )
    .unwrap();

    let dst_dir = tempdir();
    let commit = write_points_fixture(&dst_dir, &real_points, 3, Some(&same_doc_kdd));
    let dir = FsDirectory::open(&dst_dir);
    let result = check_segment(&dir, &commit);
    let failed: Vec<&str> = result.failures().iter().map(|c| c.name.as_str()).collect();
    assert!(
        failed.contains(&"points.doc_count_matches:loc"),
        "expected the doc-count check to fail, got {failed:?}"
    );
    assert!(
        !failed.contains(&"points.point_count_matches:loc"),
        "the point count is unchanged, so only the doc count may fail: {failed:?}"
    );
}

#[test]
fn fewer_actual_points_than_declared_fails_point_count_check() {
    let real_points = vec![
        (0, vec![0, 0, 0, 0, 0, 0, 0, 1]),
        (1, vec![0, 0, 0, 0, 0, 0, 0, 2]),
        (2, vec![0, 0, 0, 0, 0, 0, 0, 3]),
    ];
    let fewer_points = vec![
        (0, vec![0, 0, 0, 0, 0, 0, 0, 1]),
        (1, vec![0, 0, 0, 0, 0, 0, 0, 2]),
    ];
    let fewer_field = lucene_codecs::points::WritePointsField {
        field_number: 5,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points: fewer_points,
    };
    let (_, _, fewer_kdd) = points::write(
        &[fewer_field],
        points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
        &POINTS_SEG_ID,
        "",
    )
    .unwrap();

    let dst_dir = tempdir();
    let commit = write_points_fixture(&dst_dir, &real_points, 3, Some(&fewer_kdd));
    let dir = FsDirectory::open(&dst_dir);

    let result = check_segment(&dir, &commit);
    assert!(!result.all_passed());

    let count_check = result
        .checks
        .iter()
        .find(|c| c.name == "points.point_count_matches:loc")
        .expect("point_count_matches check must have run");
    assert!(!count_check.passed());
    assert!(count_check.message.contains("point_count=3"));
    assert!(count_check.message.contains("decoded 2"));

    std::fs::remove_dir_all(&dst_dir).ok();
}

// ----------------------------------------------------------------------
// Hand-built structures for the checks whose *failure* arms a real
// segment never reaches.
//
// Each failure arm here must be seen to fire at least once. A check
// that is never seen to fire is the failure mode this module exists to prevent -- a check
// that silently does nothing looks exactly like a check that passes.
//
// The two functions below take plain data structures rather than files,
// so a hand-built input reaches every arm directly; that is the same
// "test-only builder" pattern the `test-coverage` skill sanctions, and it
// is far cheaper than synthesizing a `.tvd`/`.dvs` per arm.
// ----------------------------------------------------------------------

fn vector_term(term: &[u8], freq: i32) -> term_vectors::TermVectorTerm {
    term_vectors::TermVectorTerm {
        term: term.to_vec(),
        freq,
        positions: None,
        start_offsets: None,
        end_offsets: None,
        payloads: None,
    }
}

/// Every self-consistency arm of `check_one_vector_field` -- Java's
/// `checkFields(..., isVectors = true)` block -- fires on a hand-built
/// term vector that breaks exactly that invariant, and *only* that one.
#[test]
fn every_term_vector_self_consistency_arm_reports_its_own_invariant() {
    // `postings = None`, so only the self-consistency half runs.
    let check = |field: &term_vectors::TermVectorField| -> Vec<String> {
        let mut memo: TermPostingsMemo = std::collections::HashMap::new();
        let mut memo_elements = 0usize;
        let mut self_problems = Vec::new();
        let mut postings_problems = Vec::new();
        check_one_vector_field(
            7,
            field,
            None,
            None,
            &mut memo,
            &mut memo_elements,
            &mut self_problems,
            &mut postings_problems,
        );
        assert!(
            postings_problems.is_empty(),
            "no postings were supplied, so no cross-check may fire: {postings_problems:?}"
        );
        self_problems
    };
    let field = |terms: Vec<term_vectors::TermVectorTerm>,
                 positions: bool,
                 offsets: bool,
                 payloads: bool| {
        term_vectors::TermVectorField {
            field_number: 0,
            has_positions: positions,
            has_offsets: offsets,
            has_payloads: payloads,
            terms,
        }
    };

    // A clean vector: nothing fires, and the field name falls back to
    // `<unknown>` only when it does (`info = None` here).
    let mut clean = vector_term(b"aaa", 2);
    clean.positions = Some(vec![0, 4]);
    clean.start_offsets = Some(vec![0, 8]);
    clean.end_offsets = Some(vec![3, 11]);
    clean.payloads = Some(vec![vec![1], vec![2]]);
    assert!(check(&field(vec![clean.clone()], true, true, true)).is_empty());

    // 1. terms out of order (`bbb` then `aaa`), and 2. a duplicate term
    //    (`<=`, not `<`).
    let out_of_order = check(&field(
        vec![vector_term(b"bbb", 1), vector_term(b"aaa", 1)],
        false,
        false,
        false,
    ));
    assert_eq!(out_of_order.len(), 1, "{out_of_order:?}");
    assert!(out_of_order[0].contains("vector terms out of order"));
    let duplicate = check(&field(
        vec![vector_term(b"aaa", 1), vector_term(b"aaa", 1)],
        false,
        false,
        false,
    ));
    assert_eq!(duplicate.len(), 1, "{duplicate:?}");
    assert!(duplicate[0].contains("vector terms out of order"));

    // 3. freq <= 0.
    let zero_freq = check(&field(vec![vector_term(b"aaa", 0)], false, false, false));
    assert_eq!(zero_freq.len(), 1, "{zero_freq:?}");
    assert!(zero_freq[0].contains("freq 0 is out of bounds"));
    assert!(zero_freq[0].contains("<unknown>"), "{zero_freq:?}");

    // 4. positions present but their count disagrees with `freq`.
    let mut t = vector_term(b"aaa", 3);
    t.positions = Some(vec![0, 4]);
    let wrong_count = check(&field(vec![t], true, false, false));
    assert_eq!(wrong_count.len(), 1, "{wrong_count:?}");
    assert!(wrong_count[0].contains("freq=3 but 2 positions"));

    // 5. a negative position, and 6. a position before the previous one.
    let mut t = vector_term(b"aaa", 2);
    t.positions = Some(vec![-1, 4]);
    let negative = check(&field(vec![t], true, false, false));
    assert_eq!(negative.len(), 1, "{negative:?}");
    assert!(negative[0].contains("position -1 is out of bounds"));
    let mut t = vector_term(b"aaa", 3);
    t.positions = Some(vec![0, 9, 4]);
    let backwards = check(&field(vec![t], true, false, false));
    assert_eq!(backwards.len(), 1, "{backwards:?}");
    assert!(backwards[0].contains("before the previous position 9"));

    // 7. the field claims positions and the term carries none.
    let no_positions = check(&field(vec![vector_term(b"aaa", 1)], true, false, false));
    assert_eq!(no_positions.len(), 1, "{no_positions:?}");
    assert!(no_positions[0].contains("claims positions but"));

    // 8. offset counts disagreeing with `freq` and with each other.
    let mut t = vector_term(b"aaa", 2);
    t.start_offsets = Some(vec![0]);
    t.end_offsets = Some(vec![3, 9]);
    let offset_counts = check(&field(vec![t], false, true, false));
    assert_eq!(offset_counts.len(), 1, "{offset_counts:?}");
    assert!(offset_counts[0].contains("freq=2 but 1 start / 2 end offsets"));

    // 9. a negative start offset, and 10. an end before its start.
    let mut t = vector_term(b"aaa", 1);
    t.start_offsets = Some(vec![-4]);
    t.end_offsets = Some(vec![3]);
    let negative_offset = check(&field(vec![t], false, true, false));
    assert_eq!(negative_offset.len(), 1, "{negative_offset:?}");
    assert!(negative_offset[0].contains("offsets [-4, 3] are out of bounds"));
    let mut t = vector_term(b"aaa", 1);
    t.start_offsets = Some(vec![9]);
    t.end_offsets = Some(vec![3]);
    let inverted = check(&field(vec![t], false, true, false));
    assert_eq!(inverted.len(), 1, "{inverted:?}");
    assert!(inverted[0].contains("offsets [9, 3] are out of bounds"));

    // 11. the field claims offsets and the term carries none.
    let no_offsets = check(&field(vec![vector_term(b"aaa", 1)], false, true, false));
    assert_eq!(no_offsets.len(), 1, "{no_offsets:?}");
    assert!(no_offsets[0].contains("claims offsets but"));

    // 12. start offsets present, end offsets absent (and the mirror).
    let mut t = vector_term(b"aaa", 1);
    t.start_offsets = Some(vec![0]);
    let half = check(&field(vec![t], false, true, false));
    assert_eq!(half.len(), 1, "{half:?}");
    assert!(half[0].contains("start and end offsets disagree"));
    let mut t = vector_term(b"aaa", 1);
    t.end_offsets = Some(vec![3]);
    let other_half = check(&field(vec![t], false, true, false));
    assert_eq!(other_half.len(), 1, "{other_half:?}");
    assert!(other_half[0].contains("start and end offsets disagree"));

    // 13. payloads on a field whose header says it has none.
    let mut t = vector_term(b"aaa", 1);
    t.payloads = Some(vec![vec![7]]);
    let stray_payloads = check(&field(vec![t], false, false, false));
    assert_eq!(stray_payloads.len(), 1, "{stray_payloads:?}");
    assert!(stray_payloads[0].contains("hasPayloads=false"));
}

/// Every cross-check arm of `check_one_vector_field` -- Java's
/// `testTermVectors` slow level, the block that requires the stored
/// vector to agree with the inverted index term for term -- fires on a
/// hand-built vector that disagrees with real, hand-written postings in
/// exactly one way.
///
/// Driven at the function rather than by corrupting a `.tvd`: this port
/// has no term-vector *writer* that can be told to emit a wrong freq, and
/// a byte flip in a real `.tvd` overwhelmingly fails the decode long
/// before it produces a well-formed vector that merely disagrees.
#[test]
fn every_term_vector_postings_cross_check_arm_reports_its_own_disagreement() {
    let dst = tempdir();
    let commit = write_positional_postings_fixture(&dst, 4);
    let dir = FsDirectory::open(&dst);
    let si = open_si(&dir, &commit).unwrap();
    let fnm = open_fnm(&dir, &commit, &si).unwrap();
    let bytes = open_postings_bytes(&dir, &si, &commit.segment_name, None)
        .unwrap()
        .unwrap();
    let handles = bytes.handles(&commit, &fnm, &si).unwrap();
    let info = fnm.fields.iter().find(|f| f.number == 0).unwrap().clone();

    // The fixture: term `alpha` occurs 4x in doc 0 at positions 0/3/6/9
    // with offsets (p*4, p*4+3) and 2-byte payloads, and 2x in doc 2.
    // Term `beta` occurs 3x in doc 1.
    let check = |field: &term_vectors::TermVectorField,
                 doc_id: i32,
                 info: Option<&field_infos::FieldInfo>| {
        let mut memo: TermPostingsMemo = std::collections::HashMap::new();
        let mut memo_elements = 0usize;
        let mut self_problems = Vec::new();
        let mut postings_problems = Vec::new();
        check_one_vector_field(
            doc_id,
            field,
            info,
            Some(&handles),
            &mut memo,
            &mut memo_elements,
            &mut self_problems,
            &mut postings_problems,
        );
        (self_problems, postings_problems)
    };
    let field = |terms: Vec<term_vectors::TermVectorTerm>| term_vectors::TermVectorField {
        field_number: 0,
        has_positions: true,
        has_offsets: true,
        has_payloads: true,
        terms,
    };
    let alpha_in_doc0 = || {
        let mut t = vector_term(b"alpha", 4);
        t.positions = Some(vec![0, 3, 6, 9]);
        t.start_offsets = Some(vec![0, 12, 24, 36]);
        t.end_offsets = Some(vec![3, 15, 27, 39]);
        t.payloads = Some(vec![vec![0, 0], vec![3, 3], vec![6, 6], vec![9, 9]]);
        t
    };

    // Baseline: a vector that matches the postings exactly is silent.
    let (self_p, cross) = check(&field(vec![alpha_in_doc0()]), 0, Some(&info));
    assert!(
        self_p.is_empty() && cross.is_empty(),
        "{self_p:?} {cross:?}"
    );

    // 1. a field the postings do not have at all, on an indexed field.
    let mut missing_field_info = info.clone();
    missing_field_info.name = "no_such_field".to_string();
    let (_, cross) = check(&field(vec![alpha_in_doc0()]), 0, Some(&missing_field_info));
    assert_eq!(cross.len(), 1, "{cross:?}");
    assert!(cross[0].contains("does not exist in the postings"));

    // ... and the same field with `IndexOptions::None`, where Java does
    // not complain either, because the field is not indexed.
    let mut unindexed = missing_field_info.clone();
    unindexed.index_options = field_infos::IndexOptions::None;
    let (_, cross) = check(&field(vec![alpha_in_doc0()]), 0, Some(&unindexed));
    assert!(cross.is_empty(), "{cross:?}");

    // 2. a term the postings do not carry.
    let mut t = vector_term(b"zzz", 1);
    t.positions = Some(vec![0]);
    t.start_offsets = Some(vec![0]);
    t.end_offsets = Some(vec![1]);
    t.payloads = Some(vec![vec![1, 1]]);
    let (_, cross) = check(&field(vec![t]), 0, Some(&info));
    assert_eq!(cross.len(), 1, "{cross:?}");
    assert!(cross[0].contains("does not exist in the postings"));

    // 3. a term the postings carry, but not for this document.
    let (_, cross) = check(&field(vec![alpha_in_doc0()]), 1, Some(&info));
    assert_eq!(cross.len(), 1, "{cross:?}");
    assert!(cross[0].contains("was not found in the postings for this doc"));

    // 4. a freq the postings disagree with.
    let mut t = alpha_in_doc0();
    t.freq = 3;
    t.positions = Some(vec![0, 3, 6]);
    t.start_offsets = Some(vec![0, 12, 24]);
    t.end_offsets = Some(vec![3, 15, 27]);
    t.payloads = Some(vec![vec![0, 0], vec![3, 3], vec![6, 6]]);
    let (_, cross) = check(&field(vec![t]), 0, Some(&info));
    assert_eq!(cross.len(), 4, "{cross:?}");
    assert!(cross[0].contains("vector freq=3 differs from postings freq=4"));
    assert!(cross[1].contains("vector positions"));
    assert!(cross[2].contains("vector offsets"));
    assert!(cross[3].contains("vector payloads differ"));

    // 5. positions only.
    let mut t = alpha_in_doc0();
    t.positions = Some(vec![0, 3, 6, 11]);
    let (_, cross) = check(&field(vec![t]), 0, Some(&info));
    assert_eq!(cross.len(), 1, "{cross:?}");
    assert!(cross[0].contains("vector positions"));

    // 6. offsets only.
    let mut t = alpha_in_doc0();
    t.end_offsets = Some(vec![3, 15, 27, 99]);
    let (_, cross) = check(&field(vec![t]), 0, Some(&info));
    assert_eq!(cross.len(), 1, "{cross:?}");
    assert!(cross[0].contains("vector offsets"));

    // 7. payloads only.
    let mut t = alpha_in_doc0();
    t.payloads = Some(vec![vec![0, 0], vec![3, 3], vec![6, 6], vec![9, 8]]);
    let (_, cross) = check(&field(vec![t]), 0, Some(&info));
    assert_eq!(cross.len(), 1, "{cross:?}");
    assert!(cross[0].contains("vector payloads differ"));

    // 8. the memo really is a memo: two documents' worth of the same term
    //    decode the postings once, and the cap resets it rather than
    //    growing without bound.
    let mut memo: TermPostingsMemo = std::collections::HashMap::new();
    let mut memo_elements = TERM_VECTOR_POSTINGS_MEMO_ELEMENTS;
    memo.insert((99, b"stale".to_vec()), (Default::default(), None));
    let mut self_problems = Vec::new();
    let mut postings_problems = Vec::new();
    check_one_vector_field(
        0,
        &field(vec![alpha_in_doc0()]),
        Some(&info),
        Some(&handles),
        &mut memo,
        &mut memo_elements,
        &mut self_problems,
        &mut postings_problems,
    );
    assert!(postings_problems.is_empty(), "{postings_problems:?}");
    assert!(
        !memo.contains_key(&(99, b"stale".to_vec())),
        "past the element cap the memo must be dropped, not grown"
    );
    assert!(memo_elements > 0 && memo_elements < TERM_VECTOR_POSTINGS_MEMO_ELEMENTS);

    std::fs::remove_dir_all(&dst).ok();
}

/// Every *global* arm of `check_doc_value_skipper` --
/// `CheckIndex.checkDocValueSkipper`'s four pre-walk guards -- fires on a
/// hand-built skip index. The per-level arms are already covered by
/// `corrupting_the_doc_values_skipper_is_caught_by_the_skipper_check`'s
/// sweep over real `.dvs` bytes; these four are not reachable that way,
/// because they read `.dvm`-derived summary fields the sweep does not
/// corrupt.
#[test]
fn the_doc_values_skipper_global_guards_each_fire_on_their_own_input() {
    let level = |min_doc_id, max_doc_id, min_value, max_value, doc_count| {
        doc_values::SkipIndexLevelInterval {
            min_doc_id,
            max_doc_id,
            min_value,
            max_value,
            doc_count,
        }
    };
    let one_interval = || doc_values::SkipIndexInterval {
        levels: vec![level(0, 3, -5, 5, 4)],
    };
    let clean = doc_values::DocValuesSkipIndex {
        min_value: -5,
        max_value: 5,
        doc_count: 4,
        max_doc_id: 3,
        max_value_count: 1,
        intervals: vec![one_interval()],
    };
    assert!(
        check_doc_value_skipper(&clean).is_empty(),
        "{:?}",
        check_doc_value_skipper(&clean)
    );

    // 1. an inverted global value range.
    let mut bad = clean.clone();
    bad.min_value = 6;
    let problems = check_doc_value_skipper(&bad);
    assert!(
        problems
            .iter()
            .any(|p| p.contains("inverted global value range: 6 > 5")),
        "{problems:?}"
    );

    // 2. `maxValueCount < -1`.
    let mut bad = clean.clone();
    bad.max_value_count = -2;
    let problems = check_doc_value_skipper(&bad);
    assert!(
        problems
            .iter()
            .any(|p| p.contains("invalid maxValueCount -2")),
        "{problems:?}"
    );

    // 3. a non-zero `maxValueCount` on a field with no documents.
    let empty = doc_values::DocValuesSkipIndex {
        min_value: 0,
        max_value: 0,
        doc_count: 0,
        max_doc_id: -1,
        max_value_count: 3,
        intervals: Vec::new(),
    };
    let problems = check_doc_value_skipper(&empty);
    assert!(
        problems
            .iter()
            .any(|p| p.contains("maxValueCount is 3 for a field with no documents")),
        "{problems:?}"
    );

    // 4. intervals that do not add up to the declared `docCount`.
    let mut bad = clean.clone();
    bad.doc_count = 9;
    let problems = check_doc_value_skipper(&bad);
    assert!(
        problems
            .iter()
            .any(|p| p.contains("declares docCount=9 but the intervals cover 4")),
        "{problems:?}"
    );

    // 5. `maxDocID(0) == i32::MAX` on a fresh skipper: the walk's
    //    `max_doc_id(0) + 1` must not be a plain `+`, or a `.dvs` whose
    //    first interval already sits at the sentinel would panic the
    //    verifier in a debug build instead of reporting the file.
    let sentinel = doc_values::DocValuesSkipIndex {
        min_value: 0,
        max_value: 0,
        doc_count: 1,
        max_doc_id: doc_values::NO_MORE_DOCS,
        max_value_count: 1,
        intervals: vec![doc_values::SkipIndexInterval {
            levels: vec![level(
                doc_values::NO_MORE_DOCS,
                doc_values::NO_MORE_DOCS,
                0,
                0,
                1,
            )],
        }],
    };
    let problems = check_doc_value_skipper(&sentinel);
    assert!(
        !problems.is_empty(),
        "a sentinel-only skip index is corrupt"
    );
}

/// `doc_values.entry_present:<f>`, all five arms: a `.fnm` that claims a
/// doc-values type the `.dvm` has no entry for.
///
/// This is not a contrived shape. `Lucene90DocValuesProducer` routes each
/// `.dvm` entry by the **type byte in the `.dvm` itself**, while every
/// reader asks for a field's values by the type in the **`.fnm`**. When
/// the two disagree the entry becomes unreachable and the field silently
/// reads as "no doc values at all" -- every range query, sort and facet on
/// it quietly returns nothing, and nothing else in this module notices,
/// because the `.dvd` still decodes perfectly. Each of the five types has
/// its own arm, so each needs its own case.
#[test]
fn a_fnm_claiming_a_doc_values_type_the_dvm_lacks_is_caught_for_every_type() {
    use field_infos::DocValuesType;
    let dst = copy_fixture("doc_values_index");
    let fnm_path = dst.join("_0.fnm");
    let original = std::fs::read(&fnm_path).unwrap();
    let dir = FsDirectory::open(&dst);
    let infos = segment_infos::read_latest(&dir).unwrap();
    let commit = infos.segments[0].clone();
    assert!(
        check_directory(&dir)
            .unwrap()
            .iter()
            .all(|r| r.all_passed()),
        "the fixture must start clean"
    );
    let parsed = field_infos::parse(&original, &commit.segment_id, "").unwrap();

    // `varying` is NUMERIC and `bin_var` is BINARY in the fixture, so
    // between them every one of the five claimed types has a case where
    // the `.dvm` entry is filed under a different one.
    for (field, claimed) in [
        ("varying", DocValuesType::Binary),
        ("varying", DocValuesType::Sorted),
        ("varying", DocValuesType::SortedNumeric),
        ("varying", DocValuesType::SortedSet),
        ("bin_var", DocValuesType::Numeric),
    ] {
        let mut fields = parsed.fields.clone();
        let fi = fields
            .iter_mut()
            .find(|f| f.name == field)
            .expect("fixture field");
        assert_ne!(fi.doc_values_type, claimed);
        fi.doc_values_type = claimed;
        std::fs::write(
            &fnm_path,
            field_infos::write(&fields, &commit.segment_id, ""),
        )
        .unwrap();

        let dir = FsDirectory::open(&dst);
        let failed = failed_names(&check_directory(&dir).unwrap());
        assert!(
            failed.contains(&format!("doc_values.entry_present:{field}")),
            "a .fnm claiming {claimed:?} for {field:?} with no matching .dvm entry was \
                 not caught: {failed:?}"
        );
    }

    std::fs::write(&fnm_path, &original).unwrap();
    let dir = FsDirectory::open(&dst);
    assert!(check_directory(&dir)
        .unwrap()
        .iter()
        .all(|r| r.all_passed()));
    std::fs::remove_dir_all(&dst).ok();
}

/// Every arm of `vectors.field_entry_matches_fnm`, driven by rewriting the
/// `.fnm` of a real Java-written vectors segment.
///
/// The `.vemf` records a vector field's dimension, encoding and
/// similarity, and so does the `.fnm`; nothing on the wire ties the two
/// together. A disagreement is silent and consequential -- the reader
/// sizes its buffers from one and interprets distances with the other --
/// and until this test none of the four arms that catch it had ever run.
#[test]
fn a_fnm_disagreeing_with_the_vemf_about_a_vector_field_is_caught() {
    let dst = copy_fixture("vectors_index");
    let fnm_path = dst.join("_0.fnm");
    let original = std::fs::read(&fnm_path).unwrap();
    let dir = FsDirectory::open(&dst);
    let infos = segment_infos::read_latest(&dir).unwrap();
    let commit = infos.segments[0].clone();
    assert!(
        check_directory(&dir)
            .unwrap()
            .iter()
            .all(|r| r.all_passed()),
        "the fixture must start clean"
    );
    let parsed = field_infos::parse(&original, &commit.segment_id, "").unwrap();
    let vector_field = parsed
        .fields
        .iter()
        .find(|f| f.vector_dimension > 0)
        .expect("the fixture has a vector field")
        .clone();

    let rewrite = |mutate: &dyn Fn(&mut field_infos::FieldInfo)| -> Vec<String> {
        let mut fields = parsed.fields.clone();
        let fi = fields
            .iter_mut()
            .find(|f| f.number == vector_field.number)
            .unwrap();
        mutate(fi);
        std::fs::write(
            &fnm_path,
            field_infos::write(&fields, &commit.segment_id, ""),
        )
        .unwrap();
        let dir = FsDirectory::open(&dst);
        failed_names(&check_directory(&dir).unwrap())
    };
    let name = &vector_field.name;

    // 1. a dimension the `.vemf` disagrees with.
    let failed = rewrite(&|fi| fi.vector_dimension += 1);
    assert!(
        failed.contains(&format!("vectors.field_entry_matches_fnm:{name}")),
        "{failed:?}"
    );

    // 2. an encoding the `.vemf` disagrees with.
    let flipped_encoding = match vector_field.vector_encoding {
        field_infos::VectorEncoding::Float32 => field_infos::VectorEncoding::Byte,
        field_infos::VectorEncoding::Byte => field_infos::VectorEncoding::Float32,
    };
    let failed = rewrite(&|fi| fi.vector_encoding = flipped_encoding);
    assert!(
        failed.contains(&format!("vectors.field_entry_matches_fnm:{name}")),
        "{failed:?}"
    );

    // 3. a similarity function the `.vemf` disagrees with. Nothing else
    //    in the segment can catch this one: the distances still compute,
    //    they are just the wrong distances.
    let flipped_similarity = match vector_field.vector_similarity_function {
        field_infos::VectorSimilarityFunction::Euclidean => {
            field_infos::VectorSimilarityFunction::DotProduct
        }
        _ => field_infos::VectorSimilarityFunction::Euclidean,
    };
    let failed = rewrite(&|fi| fi.vector_similarity_function = flipped_similarity);
    assert!(
        failed.contains(&format!("vectors.field_entry_matches_fnm:{name}")),
        "{failed:?}"
    );

    std::fs::write(&fnm_path, &original).unwrap();
    let dir = FsDirectory::open(&dst);
    assert!(check_directory(&dir)
        .unwrap()
        .iter()
        .all(|r| r.all_passed()));
    std::fs::remove_dir_all(&dst).ok();
}

/// `norms.entry_present:<f>`: a `.fnm` field that claims norms the `.nvm`
/// has no entry for.
///
/// Silent in normal use -- the field scores every document against a
/// default norm, and nothing else notices, because
/// `norms.entries_name_real_norms_fields` only looks the other way (an
/// entry naming a field that does not exist). `.fnm` and `.nvm` are
/// written by different consumers with nothing on the wire tying them
/// together, which is why `CheckIndex` cross-checks them at all -- and
/// why this arm needs a `.fnm` rewrite rather than a byte flip to reach.
#[test]
fn a_fnm_claiming_norms_the_nvm_does_not_have_is_caught() {
    let dst = copy_fixture("blocktree_index");
    let fnm_path = dst.join("_0.fnm");
    let original = std::fs::read(&fnm_path).unwrap();
    let dir = FsDirectory::open(&dst);
    let infos = segment_infos::read_latest(&dir).unwrap();
    let commit = infos.segments[0].clone();
    assert!(
        check_directory(&dir)
            .unwrap()
            .iter()
            .all(|r| r.all_passed()),
        "the fixture must start clean"
    );
    let parsed = field_infos::parse(&original, &commit.segment_id, "").unwrap();

    let rewrite = |mutate: &dyn Fn(&mut Vec<field_infos::FieldInfo>)| -> Vec<String> {
        let mut fields = parsed.fields.clone();
        mutate(&mut fields);
        std::fs::write(
            &fnm_path,
            field_infos::write(&fields, &commit.segment_id, ""),
        )
        .unwrap();
        let dir = FsDirectory::open(&dst);
        failed_names(&check_directory(&dir).unwrap())
    };

    // 1. a new indexed field that claims norms. The `.nvm` has no entry
    //    for it, so every score for the field would silently be computed
    //    against a default norm. (Adding a field rather than flipping
    //    `omitNorms` on an existing one: every indexed field in this
    //    fixture already has norms, and `FieldInfo::check_consistency`
    //    refuses norms on a *non*-indexed field, so there is no existing
    //    field to flip.)
    let template = parsed
        .fields
        .iter()
        .find(|f| f.index_options != field_infos::IndexOptions::None)
        .expect("the fixture has an indexed field")
        .clone();
    let ghost_number = parsed.fields.iter().map(|f| f.number).max().unwrap() + 1;
    let failed = rewrite(&|fields| {
        let mut ghost = template.clone();
        ghost.name = "ghost".to_string();
        ghost.number = ghost_number;
        ghost.omit_norms = false;
        ghost.doc_values_type = field_infos::DocValuesType::None;
        ghost.doc_values_skip_index_type = field_infos::DocValuesSkipIndexType::None;
        ghost.store_term_vectors = false;
        ghost.point_dimension_count = 0;
        ghost.point_index_dimension_count = 0;
        ghost.point_num_bytes = 0;
        ghost.vector_dimension = 0;
        fields.push(ghost);
    });
    assert!(
        failed.contains(&"norms.entry_present:ghost".to_string()),
        "a field claiming norms the .nvm lacks was not caught: {failed:?}"
    );

    std::fs::write(&fnm_path, &original).unwrap();
    let dir = FsDirectory::open(&dst);
    assert!(check_directory(&dir)
        .unwrap()
        .iter()
        .all(|r| r.all_passed()));
    std::fs::remove_dir_all(&dst).ok();
}

/// A SORTED_SET field where every document has exactly one value.
///
/// `Lucene90DocValuesConsumer.addSortedSetField` collapses that shape to a
/// plain `SortedEntry` (`multiValued = 0`), and the reader's
/// `SortedSetKind::Single` branch decodes it -- so `check_doc_values` has
/// a whole second SORTED_SET code path, with its own ordinal bookkeeping,
/// that no fixture in this repo exercised. Every Java-written SORTED_SET
/// fixture here happens to be genuinely multi-valued.
fn write_single_valued_sorted_set_fixture(
    dst_dir: &std::path::Path,
    values: &[&[u8]],
) -> segment_infos::SegmentCommitInfo {
    let max_doc = values.len() as i32;
    let fi = field_infos::FieldInfo {
        name: "tags".to_string(),
        number: 0,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: field_infos::IndexOptions::None,
        doc_values_type: field_infos::DocValuesType::SortedSet,
        doc_values_skip_index_type: field_infos::DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: field_infos::VectorEncoding::Float32,
        vector_similarity_function: field_infos::VectorSimilarityFunction::Euclidean,
    };
    let fnm = field_infos::write(std::slice::from_ref(&fi), &SORT_SEG_ID, "");
    std::fs::write(dst_dir.join("_0.fnm"), &fnm).unwrap();
    let per_doc: Vec<Vec<Vec<u8>>> = values.iter().map(|v| vec![v.to_vec()]).collect();
    let (dvm, dvd, dvs) = lucene_codecs::doc_values::write_single_dense_sorted_set_field(
        0,
        &per_doc,
        max_doc,
        &SORT_SEG_ID,
        "",
    )
    .unwrap();
    std::fs::write(dst_dir.join("_0.dvm"), &dvm).unwrap();
    std::fs::write(dst_dir.join("_0.dvd"), &dvd).unwrap();
    std::fs::write(dst_dir.join("_0.dvs"), &dvs).unwrap();

    let si = SegmentInfo {
        id: SORT_SEG_ID,
        version: segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        min_version: None,
        doc_count: max_doc,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: vec![
            "_0.si".to_string(),
            "_0.fnm".to_string(),
            "_0.dvm".to_string(),
            "_0.dvd".to_string(),
            "_0.dvs".to_string(),
        ],
        attributes: vec![],
        index_sort: None,
    };
    std::fs::write(dst_dir.join("_0.si"), segment_info::write(&si, "")).unwrap();

    segment_infos::SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id: SORT_SEG_ID,
        codec_name: "Lucene104".to_string(),
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    }
}

/// The `SortedSetKind::Single` half of `check_doc_values`: a clean
/// single-valued SORTED_SET passes every check, and a corrupted one is
/// rejected by the ordinal-space checks that branch owns.
#[test]
fn a_single_valued_sorted_set_field_is_checked_through_the_single_branch() {
    let dst = tempdir();
    let commit = write_single_valued_sorted_set_fixture(
        &dst,
        &[b"alpha", b"beta", b"gamma", b"alpha", b"delta"],
    );
    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    assert!(
        result.all_passed(),
        "a single-valued SORTED_SET must pass cleanly: {:?}",
        result.failures()
    );
    let names: Vec<&str> = result.checks.iter().map(|c| c.name.as_str()).collect();
    for expected in [
        "doc_values.values_decode:tags",
        "doc_values.ords_dense:tags",
        "doc_values.terms_sorted:tags",
    ] {
        assert!(
            names.contains(&expected),
            "{expected} did not run: {names:?}"
        );
    }

    // The branch's own ordinal bookkeeping really is live: re-signed
    // single-byte corruptions of the `.dvd` must be rejected by it.
    let dvd_path = dst.join("_0.dvd");
    let original = std::fs::read(&dvd_path).unwrap();
    let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
    let mut caught = 0usize;
    let mut total = 0usize;
    for off in 48..body_end {
        for mask in [0x01u8, 0xff] {
            let mut bytes = original.clone();
            bytes[off] ^= mask;
            repair_checksum(&mut bytes);
            std::fs::write(&dvd_path, &bytes).unwrap();
            let dir = FsDirectory::open(&dst);
            total += 1;
            // Attributed to the ordinal-space checks specifically, not to
            // "some check failed": `doc_values.values_decode` alone would
            // satisfy a bare `all_passed()` counter, and the assertion
            // below would then say nothing about the `Single` branch.
            if check_segment(&dir, &commit).failures().iter().any(|c| {
                c.name == "doc_values.ords_dense:tags" || c.name == "doc_values.terms_sorted:tags"
            }) {
                caught += 1;
            }
        }
    }
    std::fs::write(&dvd_path, &original).unwrap();
    // 17 of 54 when this was written. The dictionary is five short
    // terms, so most byte flips land inside a term's bytes and leave a
    // different but perfectly valid single-valued set behind -- the same
    // reason the `.dvd` control on `sorted_dv_index` accepts 54 of 99.
    assert!(total >= 50, "the sweep must actually run: {total}");
    assert!(
        caught >= 14,
        "only {caught} of {total} re-signed .dvd corruptions were rejected by the \
             SortedSetKind::Single branch's checks; 17 were when this test was written"
    );

    std::fs::remove_dir_all(&dst).ok();
}

/// `check_term_vectors`' three "the segment disagrees with itself" arms,
/// on a real Java-written term-vectors segment: a `.si` that lists only
/// some of `.tvd`/`.tvx`/`.tvm`, a `.si` whose `docCount` disagrees with
/// the term-vectors reader's own `maxDoc`, and a `.fnm` that says a field
/// carrying vectors does not store them.
///
/// All three are cross-file: `.si`, `.fnm` and the `.tv*` triple are
/// written by different consumers, and nothing on the wire ties them
/// together -- which is why `CheckIndex` compares them and why none of
/// these arms is reachable by corrupting bytes inside a single file.
#[test]
fn a_segment_that_disagrees_with_its_own_term_vectors_is_caught() {
    let dst = copy_fixture("term_vectors_index");
    let dir = FsDirectory::open(&dst);
    let infos = segment_infos::read_latest(&dir).unwrap();
    let commit = infos.segments[0].clone();
    let si_path = dst.join(format!("{}.si", commit.segment_name));
    let original_si = std::fs::read(&si_path).unwrap();
    let fnm_path = dst.join(format!("{}.fnm", commit.segment_name));
    let original_fnm = std::fs::read(&fnm_path).unwrap();
    let si = segment_info::parse(&original_si, &commit.segment_id).unwrap();
    assert!(
        check_segment(&dir, &commit).all_passed(),
        "the fixture must start clean: {:?}",
        check_segment(&dir, &commit).failures()
    );

    let write_si = |mutate: &dyn Fn(&mut SegmentInfo)| -> Vec<String> {
        let mut mutated = si.clone();
        mutate(&mut mutated);
        std::fs::write(&si_path, segment_info::write(&mutated, "")).unwrap();
        let dir = FsDirectory::open(&dst);
        check_segment(&dir, &commit)
            .failures()
            .iter()
            .map(|c| c.name.clone())
            .collect()
    };

    // 1. the `.si` lists `.tvd`/`.tvx` but not `.tvm`.
    let failed = write_si(&|s| s.files.retain(|f| !f.ends_with(".tvm")));
    assert!(
        failed.contains(&"term_vectors.open".to_string()),
        "a partial .tv* file set was not caught: {failed:?}"
    );

    // 2. the `.si` claims more documents than the term-vectors reader
    //    has. `doc_count_matches_si` is the only check that compares the
    //    two, and a reader that stops early would otherwise look clean.
    let failed = write_si(&|s| s.doc_count += 3);
    assert!(
        failed.contains(&"term_vectors.doc_count_matches_si".to_string()),
        "a .si docCount above the term-vectors reader's maxDoc was not caught: {failed:?}"
    );
    std::fs::write(&si_path, &original_si).unwrap();

    // 3. the `.fnm` says a field that carries vectors does not store
    //    them. Java reports this per document, and so does this port.
    let parsed = field_infos::parse(&original_fnm, &commit.segment_id, "").unwrap();
    let vectored = parsed
        .fields
        .iter()
        .find(|f| f.store_term_vectors)
        .expect("the fixture has a term-vector field")
        .name
        .clone();
    let mut fields = parsed.fields.clone();
    fields
        .iter_mut()
        .find(|f| f.name == vectored)
        .unwrap()
        .store_term_vectors = false;
    std::fs::write(
        &fnm_path,
        field_infos::write(&fields, &commit.segment_id, ""),
    )
    .unwrap();
    let dir = FsDirectory::open(&dst);
    let failed: Vec<String> = check_segment(&dir, &commit)
        .failures()
        .iter()
        .map(|c| c.name.clone())
        .collect();
    assert!(
        failed.contains(&"term_vectors.fields_marked_in_fnm".to_string()),
        "a .fnm denying term vectors a document actually carries was not caught: {failed:?}"
    );
    std::fs::write(&fnm_path, &original_fnm).unwrap();

    let dir = FsDirectory::open(&dst);
    assert!(check_segment(&dir, &commit).all_passed());
    std::fs::remove_dir_all(&dst).ok();
}

/// The NUMERIC/BINARY halves of `check_doc_values`' per-document decode,
/// driven by re-signed single-byte corruptions of a real Java-written
/// `.dvd` and `.dvp` (the sparse fields' `IndexedDISI`, in the `.dvp` since
/// format version 3) carrying dense numeric, GCD-compressed numeric, sparse
/// numeric, fixed-length binary, variable-length binary and sparse binary
/// fields.
///
/// `corrupted_doc_values_payload_is_caught` covers the SORTED fixture;
/// this one covers the five other entry shapes, whose decode-failure arms
/// are separate code. Re-signed, so the `file:*` CRC cannot stand in for
/// the value checks.
#[test]
fn no_re_signed_corruption_of_a_numeric_or_binary_dvd_goes_unnoticed() {
    let dst = copy_fixture("doc_values_index");
    let files = [find_file(&dst, ".dvd"), find_file(&dst, ".dvp")];
    let originals: Vec<Vec<u8>> = files.iter().map(|p| std::fs::read(p).unwrap()).collect();
    let dir = FsDirectory::open(&dst);
    assert!(
        check_directory(&dir)
            .unwrap()
            .iter()
            .all(|r| r.all_passed()),
        "the fixture must start clean"
    );

    let mut caught_by_dv = 0usize;
    let mut caught_by_other = 0usize;
    let mut accepted = 0usize;
    let mut isolated: Option<(usize, usize, u8)> = None;
    for (file, original) in originals.iter().enumerate() {
        let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
        for off in 48..body_end {
            for mask in [0x01u8, 0x80, 0xff] {
                let mut bytes = original.clone();
                bytes[off] ^= mask;
                repair_checksum(&mut bytes);
                std::fs::write(&files[file], &bytes).unwrap();
                let dir = FsDirectory::open(&dst);
                // A corruption that makes the *commit* unreadable is still a
                // corruption this module rejected -- counting it as
                // "caught by another check" keeps `total` equal to the
                // sweep's own iteration count, which is what makes the
                // floors below mean what they say.
                let failed = match check_directory(&dir) {
                    Ok(r) => failed_names(&r),
                    Err(e) => vec![format!("commit.unreadable: {e}")],
                };
                let is_dv = |n: &String| n.starts_with("doc_values.");
                if failed.iter().any(is_dv) {
                    caught_by_dv += 1;
                    if failed.iter().all(is_dv) {
                        isolated.get_or_insert((file, off, mask));
                    }
                } else if failed.is_empty() {
                    accepted += 1;
                } else {
                    caught_by_other += 1;
                }
            }
        }
        std::fs::write(&files[file], original).unwrap();
    }
    let total = caught_by_dv + caught_by_other + accepted;
    // Of the 288 re-signed corruptions, 96 were rejected by a `doc_values.*`
    // check, none by anything else, and 192 accepted; the floor leaves a
    // margin below the 96. Nothing but these checks reads the `.dvd`
    // and `.dvp`, so a corruption they miss is a corruption nothing
    // catches; most of the `.dvd` is raw values, and a different value is
    // still a valid one.
    assert!(
        caught_by_dv >= 85,
        "the doc-values checks rejected only {caught_by_dv} of {total} re-signed .dvd/.dvp \
             corruptions ({caught_by_other} caught by another check, {accepted} accepted)"
    );
    assert_eq!(
        caught_by_other, 0,
        "a .dvd or .dvp corruption was reported by something other than a doc_values.* check"
    );
    let (file, off, mask) =
        isolated.expect("at least one corruption is caught by doc_values alone");
    let mut bytes = originals[file].clone();
    bytes[off] ^= mask;
    repair_checksum(&mut bytes);
    std::fs::write(&files[file], &bytes).unwrap();
    let dir = FsDirectory::open(&dst);
    let failed = failed_names(&check_directory(&dir).unwrap());
    std::fs::write(&files[file], &originals[file]).unwrap();
    assert!(
        !failed.is_empty() && failed.iter().all(|n| n.starts_with("doc_values.")),
        "byte {off} ^ {mask:#x} was expected to trip only the doc-values checks: {failed:?}"
    );

    let dir = FsDirectory::open(&dst);
    assert!(check_directory(&dir)
        .unwrap()
        .iter()
        .all(|r| r.all_passed()));
    std::fs::remove_dir_all(&dst).ok();
}

/// `checkSoftDeletes` on a segment that has *both* soft deletes and hard
/// deletes.
///
/// Java's `PendingSoftDeletes.countSoftDeletes` counts only documents that
/// are still live: a document that is both soft-deleted and hard-deleted
/// must not be counted twice. That is the `del_gen != -1` branch, which
/// reads the `.liv` and intersects it with the soft-deletes field --
/// untested until now, because every soft-deletes fixture here had
/// `del_gen == -1`, so the check only ever ran with `live = None`.
#[test]
fn soft_deletes_are_counted_only_among_live_documents() {
    use lucene_util::fixed_bit_set::FixedBitSet;

    let dst = tempdir();
    // All four docs carry the soft-deletes field, but doc 3 is also hard
    // deleted -- so the live soft-delete count is 3, not 4.
    let mut commit = write_sorted_dv_fixture(&dst, &[1, 1, 1, 1], None, true, 3);
    let mut live = FixedBitSet::new(4);
    for doc in 0..3 {
        live.set(doc);
    }
    let liv = live_docs::write(&live, &commit.segment_id, 1, 1).unwrap();
    std::fs::write(dst.join(liv_file_name("_0", 1)), &liv).unwrap();
    commit.del_gen = 1;
    commit.del_count = 1;

    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "soft_deletes.count_matches")
        .expect("the soft-deletes check must run");
    assert!(
        check.passed(),
        "a hard-deleted doc must not count towards softDelCount: {}",
        check.message
    );

    // And the count really is live-aware: claiming 4 (every doc, ignoring
    // the hard delete) must now fail.
    commit.soft_del_count = 4;
    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "soft_deletes.count_matches")
        .unwrap();
    assert!(!check.passed());
    assert!(
        check.message.contains("actual soft deletes: 3"),
        "{}",
        check.message
    );

    std::fs::remove_dir_all(&dst).ok();
}

/// Every subsystem's `*.open` arm: a file the `.si` lists, replaced by a
/// well-formed file of a *different* kind from the same segment.
///
/// This is the failure `check_index` most needs to survive gracefully.
/// Each decoder validates a codec header naming its own format, so the
/// wrong file is rejected at open -- and the module's contract is that it
/// then *reports* the failure under that subsystem's own name and keeps
/// checking the rest of the segment, rather than panicking or abandoning
/// the segment. Every one of these `Err(e) => checks.push(Check::fail(
/// "<x>.open", ..))` arms existed unexercised; a decoder change that
/// turned one of them into a panic would have gone unnoticed.
#[test]
fn a_file_replaced_by_another_kind_is_reported_by_its_own_open_check() {
    // (fixture, victim extension, donor extension, expected check prefix)
    let cases: &[(&str, &str, &str, &str)] = &[
        ("vectors_index", ".vemf", ".vec", "vectors.open"),
        ("vectors_index", ".vex", ".vec", "hnsw.open"),
        ("points_index", ".kdm", ".fdt", "points.open"),
        ("points_index", ".kdi", ".fdt", "points."),
        ("term_vectors_index", ".tvm", ".fdt", "term_vectors.open"),
        ("blocktree_index", ".fdt", ".nvd", "stored_fields."),
        ("blocktree_index", ".nvm", ".fdt", "norms.open"),
        ("blocktree_index", ".tim", ".nvd", "postings.open"),
        ("doc_values_index", ".dvm", ".fdt", "doc_values."),
        ("doc_values_skip_index", ".dvs", ".dvd", "doc_values."),
        ("blocktree_index", ".tip", ".nvd", "postings."),
        ("blocktree_index", ".tmd", ".nvd", "postings."),
        ("blocktree_index", ".doc", ".nvd", "postings."),
        ("vectors_index", ".vec", ".vemf", "vectors."),
        ("term_vectors_index", ".tvd", ".fdt", "term_vectors."),
        ("points_index", ".kdd", ".fdt", "points."),
    ];
    for (fixture, victim_ext, donor_ext, expected) in cases {
        let dst = copy_fixture(fixture);
        let victim = find_file(&dst, victim_ext);
        let donor = std::fs::read(find_file(&dst, donor_ext)).unwrap();
        let original = std::fs::read(&victim).unwrap();
        std::fs::write(&victim, &donor).unwrap();

        let dir = FsDirectory::open(&dst);
        // `check_directory` must not panic, and must not give up on the
        // segment: the commit-level result is always produced.
        let results = check_directory(&dir).expect("the commit itself is untouched");
        let failed = failed_names(&results);
        assert!(
            failed.iter().any(|n| n.starts_with(expected)),
            "{fixture}: {victim_ext} replaced by {donor_ext} was not reported by \
                 {expected}*; failures: {failed:?}"
        );

        std::fs::write(&victim, &original).unwrap();
        let dir = FsDirectory::open(&dst);
        assert!(
            check_directory(&dir)
                .unwrap()
                .iter()
                .all(|r| r.all_passed()),
            "{fixture} must be clean again after restoring {victim_ext}"
        );
        std::fs::remove_dir_all(&dst).ok();
    }
}

/// `testSort` over a **SORTED_NUMERIC** sort field.
///
/// `sort_key_values` has a separate arm per doc-values type, and Lucene
/// sorts on the *minimum* value of a multi-valued SORTED_NUMERIC field
/// (`SortedNumericSortField`'s `MIN` selector, which is what
/// `IndexWriterConfig.setIndexSort` produces). Every index-sort fixture
/// here used a plain NUMERIC field, so that arm -- and the `min()` that
/// makes multi-valued sorting well-defined at all -- had never run.
#[test]
fn an_index_sort_on_a_multi_valued_sorted_numeric_field_is_verified() {
    let write = |dst: &std::path::Path, values: &[Vec<i64>]| {
        let fi = field_infos::FieldInfo {
            name: "ts".to_string(),
            number: 0,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: field_infos::IndexOptions::None,
            doc_values_type: field_infos::DocValuesType::SortedNumeric,
            doc_values_skip_index_type: field_infos::DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: field_infos::VectorEncoding::Float32,
            vector_similarity_function: field_infos::VectorSimilarityFunction::Euclidean,
        };
        std::fs::write(
            dst.join("_0.fnm"),
            field_infos::write(std::slice::from_ref(&fi), &SORT_SEG_ID, ""),
        )
        .unwrap();
        let (dvm, dvd, dvs) = lucene_codecs::doc_values::write_single_dense_sorted_numeric_field(
            0,
            values,
            &SORT_SEG_ID,
            "",
        )
        .unwrap();
        std::fs::write(dst.join("_0.dvm"), &dvm).unwrap();
        std::fs::write(dst.join("_0.dvd"), &dvd).unwrap();
        std::fs::write(dst.join("_0.dvs"), &dvs).unwrap();
        let si = SegmentInfo {
            id: SORT_SEG_ID,
            version: segment_info::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: values.len() as i32,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec![
                "_0.si".to_string(),
                "_0.fnm".to_string(),
                "_0.dvm".to_string(),
                "_0.dvd".to_string(),
                "_0.dvs".to_string(),
            ],
            attributes: vec![],
            index_sort: sorted_numeric_sort_asc(),
        };
        std::fs::write(dst.join("_0.si"), segment_info::write(&si, "")).unwrap();
        segment_infos::SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: SORT_SEG_ID,
            codec_name: "Lucene104".to_string(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
            ..Default::default()
        }
    };

    // Sorted by each document's *minimum* value: 1, 3, 3, 9.
    let good = tempdir();
    let commit = write(&good, &[vec![1, 50], vec![3, 4], vec![3], vec![9, 9]]);
    let dir = FsDirectory::open(&good);
    let result = check_segment(&dir, &commit);
    assert!(
        result.all_passed(),
        "a correctly sorted SORTED_NUMERIC segment must pass: {:?}",
        result.failures()
    );
    assert!(result
        .checks
        .iter()
        .any(|c| c.name == "sort.docs_in_index_sort_order"));
    std::fs::remove_dir_all(&good).ok();

    // Sorted by each document's *maximum* instead: 50, 4, 3, 9 -- which
    // is out of order under `MIN`, and is exactly the mistake a writer
    // that picked the wrong selector would make.
    let bad = tempdir();
    let commit = write(&bad, &[vec![1, 50], vec![3, 4], vec![3], vec![0, 9]]);
    let dir = FsDirectory::open(&bad);
    let result = check_segment(&dir, &commit);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "sort.docs_in_index_sort_order")
        .expect("the sort check must run");
    assert!(!check.passed(), "{}", check.message);
    std::fs::remove_dir_all(&bad).ok();

    // A document whose values are stored *descending*.
    // `SortedNumericDocValuesWriter.finishCurrentDoc` sorts them, so real
    // Lucene never writes this and real
    // `CheckIndex.checkSortedNumericDocValues` throws `"values out of
    // order"` on it; the column is also what makes
    // `SortedNumericSelector.MIN`/`MAX` -- the first and the last stored
    // value -- mean what they say.
    let unsorted = tempdir();
    let commit = write(&unsorted, &[vec![1, 50], vec![4, 3], vec![9], vec![9, 9]]);
    let dir = FsDirectory::open(&unsorted);
    let result = check_segment(&dir, &commit);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "doc_values.values_decode:ts")
        .expect("the doc-values check must run");
    assert!(!check.passed(), "{}", check.message);
    assert!(
        check.message.contains("values out of order: 3 < 4"),
        "{}",
        check.message
    );
    std::fs::remove_dir_all(&unsorted).ok();
}

// ---------------------------------------------------------------------
// Failure arms no real segment reaches.
//
// A `Check::fail`/`problems.push` never seen to fire is
// indistinguishable, from outside, from a check that passes. Each test drives one
// family and asserts the *named* check fails, not merely that something
// did.
// ---------------------------------------------------------------------

/// A `FieldInfo` that claims **every** per-format feature at once, so one
/// segment can drive all seven format checks' guards in a single pass.
fn all_claiming_field_info() -> field_infos::FieldInfo {
    field_infos::FieldInfo {
        name: "everything".to_string(),
        number: 0,
        store_term_vectors: true,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: true,
        parent_field: false,
        index_options: field_infos::IndexOptions::DocsAndFreqs,
        doc_values_type: field_infos::DocValuesType::Numeric,
        doc_values_skip_index_type: field_infos::DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![
            (
                lucene_codecs::per_field_postings::PER_FIELD_FORMAT_KEY.to_string(),
                "Lucene104".to_string(),
            ),
            (
                lucene_codecs::per_field_postings::PER_FIELD_SUFFIX_KEY.to_string(),
                "0".to_string(),
            ),
        ],
        point_dimension_count: 1,
        point_index_dimension_count: 1,
        point_num_bytes: 8,
        vector_dimension: 4,
        vector_encoding: field_infos::VectorEncoding::Float32,
        vector_similarity_function: field_infos::VectorSimilarityFunction::Euclidean,
    }
}

#[test]
fn zero_token_fields_and_legacy_direct_postings_have_distinct_file_claims() {
    let mut field = field_infos::FieldInfo::new("empty", 0);
    field.index_options = field_infos::IndexOptions::DocsAndFreqsAndPositions;
    field.omit_norms = false;
    let mut fields = FieldInfos {
        fields: vec![field],
    };
    let norm_files = vec!["_0.nvm".to_string(), "_0.nvd".to_string()];
    let mut checks = Vec::new();
    check_field_flags_vs_files(&fields, &norm_files, &mut checks);
    assert!(checks.iter().all(Check::passed), "{checks:?}");

    // A per-field reader retains a routed field after its last term is
    // deleted. Even without terms it still requires its native group.
    let routing = all_claiming_field_info().attributes;
    // A partial routing pair also cannot excuse absent postings files.
    for attributes in [
        routing.clone(),
        vec![routing[0].clone()],
        vec![routing[1].clone()],
    ] {
        fields.fields[0].attributes = attributes;
        checks.clear();
        check_field_flags_vs_files(&fields, &norm_files, &mut checks);
        assert_eq!(
            checks
                .iter()
                .filter(|check| !check.passed())
                .map(|check| check.name.as_str())
                .collect::<Vec<_>>(),
            ["fnm.postings_vs_files"]
        );
    }

    // Older direct/no-suffix writers have indexed fields without routing
    // attributes. Their existing files must agree with the indexed field.
    fields.fields[0].attributes.clear();
    let mut direct_files = norm_files;
    direct_files.extend(["_0.tim", "_0.tip", "_0.tmd"].map(str::to_string));
    checks.clear();
    check_field_flags_vs_files(&fields, &direct_files, &mut checks);
    assert!(checks.iter().all(Check::passed), "{checks:?}");

    // The reverse check still rejects postings files with no indexed
    // field, regardless of whether a legacy writer used routing metadata.
    fields.fields[0].index_options = field_infos::IndexOptions::None;
    checks.clear();
    check_field_flags_vs_files(&fields, &["_0.tim".to_string()], &mut checks);
    assert_eq!(
        checks
            .iter()
            .filter(|check| !check.passed())
            .map(|check| check.name.as_str())
            .collect::<Vec<_>>(),
        ["fnm.postings_vs_files"]
    );
}

const CLAIMS_SEG_ID: [u8; ID_LENGTH] = [31u8; ID_LENGTH];

fn claiming_si(is_compound_file: bool, files: Vec<String>) -> SegmentInfo {
    SegmentInfo {
        id: CLAIMS_SEG_ID,
        version: segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        min_version: None,
        doc_count: 2,
        is_compound_file,
        has_blocks: false,
        diagnostics: vec![],
        files,
        attributes: vec![],
        index_sort: sort_asc(),
    }
}

/// Every codec file the claiming field's formats would open. The files
/// do not exist on disk: the point is that each check gets *past* its
/// "this segment has no such files" early return, so the only thing that
/// can still keep it quiet is the guard under test.
fn claiming_files() -> Vec<String> {
    [
        "_0.si",
        "_0.fnm",
        "_0.dvm",
        "_0.dvd",
        "_0.nvm",
        "_0.nvd",
        "_0.tvd",
        "_0.tvx",
        "_0.tvm",
        "_0.kdm",
        "_0.kdi",
        "_0.kdd",
        "_0_Lucene99HnswVectorsFormat_0.vec",
        "_0_Lucene99HnswVectorsFormat_0.vemf",
        "_0_Lucene104_0.tim",
        "_0_Lucene104_0.tip",
        "_0_Lucene104_0.tmd",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn claiming_commit() -> segment_infos::SegmentCommitInfo {
    segment_infos::SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id: CLAIMS_SEG_ID,
        codec_name: "Lucene104".to_string(),
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    }
}

/// A real compound segment, flushed by Lucene 10.5.0
/// (`fixtures/data/compound_index`: postings on `id`, NUMERIC doc values
/// on `num`), gets every check a loose one does -- the postings and
/// doc-values walks included -- and passes them.
#[test]
fn a_real_lucene_compound_segment_is_checked_in_full() {
    let src =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/compound_index");
    let dst = tempdir();
    for name in ["_0.cfs", "_0.cfe", "_0.si", "segments_1"] {
        std::fs::copy(src.join(name), dst.join(name)).unwrap();
    }
    let dir = FsDirectory::open(&dst);
    let results = check_directory(&dir).unwrap();
    let segment = &results[1];
    assert!(segment.all_passed(), "{:?}", segment.failures());
    let names: Vec<&str> = segment.checks.iter().map(|c| c.name.as_str()).collect();
    for family in ["cfs.open", "fnm.open", "postings.", "doc_values."] {
        assert!(
            names.iter().any(|n| n.starts_with(family)),
            "no {family} check: {names:?}"
        );
    }

    // A damaged archive is reported, and names what it took down.
    let cfs = std::fs::read(dst.join("_0.cfs")).unwrap();
    std::fs::write(dst.join("_0.cfs"), &cfs[..cfs.len() - 1]).unwrap();
    let results = check_directory(&dir).unwrap();
    let segment = &results[1];
    assert!(segment
        .checks
        .iter()
        .any(|c| c.name == "cfs.open" && c.outcome == Outcome::Failed));
    assert!(segment.checks.iter().any(|c| c.was_skipped()));
    std::fs::remove_dir_all(&dst).ok();
}

/// A compound segment must not make the per-format checks return
/// silently. Compound segments reach the checks only through
/// [`CompoundReader`] (`check_segment_in_commit`), as a loose segment
/// whose files are the archive's members, and no check looks at the flag
/// any more. This pins that: the flag changes nothing below
/// `check_segment_in_commit`, so a guard that crept back in would fail
/// here.
#[test]
fn no_format_check_skips_a_segment_for_being_compound() {
    let fields = FieldInfos {
        fields: vec![all_claiming_field_info()],
    };
    let commit = claiming_commit();
    let dst = tempdir();
    let dir = FsDirectory::open(&dst);

    // Run every per-format check over one `.si`, twice: once compound and
    // once not. The file list names every codec file the claiming field's
    // formats look for and **none of them exists on disk**, so the
    // non-compound run must report -- which is what makes the compound
    // run's silence mean "the guard fired" rather than "there was nothing
    // to do anyway". Without this pairing, five of the seven assertions
    // below would also hold with the guards deleted, because each check's
    // "this segment has no such files" early return would take over. That
    // is the weakness this test guards against.
    let run = |is_compound_file: bool| -> Vec<Check> {
        let si = claiming_si(is_compound_file, claiming_files());
        let mut stats = CheckStats::default();
        let mut checks = Vec::new();
        check_term_vectors(
            &dir,
            &commit,
            &si,
            Some(&fields),
            None,
            None,
            &mut stats,
            &mut checks,
        );
        check_doc_values(&dir, &commit, &si, &fields, &mut checks);
        check_index_sort(&dir, &commit, &si, &fields, &mut checks);
        check_soft_deletes(&dir, &commit, &si, &fields, &mut checks);
        check_field_norms(
            &dir,
            &commit,
            &si,
            &fields,
            None,
            &DocsWithTerms::new(),
            &mut stats,
            &mut checks,
        );
        check_vectors(&dir, &commit, &si, &fields, &mut stats, &mut checks);
        check_points_structural_invariants(&dir, &commit, &si, &fields, &mut checks);
        checks
    };

    // The control: with the compound flag off, every one of the seven
    // reports the files it cannot open, under its own subsystem's name.
    let reported = run(false);
    let mut names: Vec<&str> = reported
        .iter()
        .filter(|c| c.outcome == Outcome::Failed)
        .map(|c| {
            c.name
                .split_once(':')
                .map_or(c.name.as_str(), |(family, _)| family)
        })
        .collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(
        names,
        [
            "doc_values.open",
            "norms.open",
            "points.open",
            "soft_deletes.count_matches",
            "sort.docs_in_index_sort_order",
            "term_vectors.open",
            "vectors.open",
        ],
        "the non-compound control must reach all seven walks: {reported:?}"
    );
    // Each of those seven takes its own family down with it, and says so.
    assert!(
        reported.iter().filter(|c| c.was_skipped()).count() >= 20,
        "a failed *.open must name what it took down: {reported:?}"
    );

    // The subject: the same segment marked compound reports exactly the
    // same.
    let flagged = run(true);
    assert_eq!(
        flagged
            .iter()
            .map(|c| (&c.name, c.outcome))
            .collect::<Vec<_>>(),
        reported
            .iter()
            .map(|c| (&c.name, c.outcome))
            .collect::<Vec<_>>(),
    );
    assert!(
        matches!(
            open_postings_bytes(&dir, &claiming_si(true, claiming_files()), "_0", None),
            Some(Err(_))
        ),
        "the postings walk must not skip a compound-flagged segment either"
    );
    std::fs::remove_dir_all(&dst).ok();
}

/// A field that claims a format whose files the segment does not have is
/// reported **once**, by `fnm.*_vs_files`, and the format's own walk is
/// then skipped rather than reporting the same absence a second time
/// under a different name. Vectors are the one deliberate exception: a
/// `.vec`/`.vemf`-less vector field has no `fnm.vectors_vs_files`
/// counterpart (there is no such check), so `vectors.open` is the only
/// place it can be reported and it *must* fire.
///
/// Five of these six early returns had never run, which means the
/// no-duplicate-reporting decision was entirely unverified.
#[test]
fn a_field_claiming_a_format_with_no_files_is_not_walked_twice() {
    let fields = FieldInfos {
        fields: vec![all_claiming_field_info()],
    };
    let si = claiming_si(false, vec!["_0.si".to_string(), "_0.fnm".to_string()]);
    let commit = claiming_commit();
    let dst = tempdir();
    let dir = FsDirectory::open(&dst);

    // First prove the claims are *live*, so the silence asserted below is
    // a decision and not an accident of a `FieldInfo` that forgot to
    // claim anything. This is the `fnm.*_vs_files` half of "reported
    // once": all four families report here, and nowhere else.
    let mut claims = Vec::new();
    check_field_flags_vs_files(&fields, &si.files, &mut claims);
    let claimed: Vec<&str> = claims
        .iter()
        .filter(|c| !c.passed())
        .map(|c| c.name.as_str())
        .collect();
    assert_eq!(
        claimed,
        [
            "fnm.doc_values_vs_files",
            "fnm.norms_vs_files",
            "fnm.term_vectors_vs_files",
            "fnm.postings_vs_files",
        ],
        "the field must actually claim all four file groups"
    );
    assert!(
        si.index_sort.is_some(),
        "the .si must declare an index sort"
    );
    assert!(
        fields.fields[0].soft_deletes_field,
        "the field must be the soft-deletes field"
    );

    let mut checks = Vec::new();
    check_doc_values(&dir, &commit, &si, &fields, &mut checks);
    let mut stats = CheckStats::default();
    check_field_norms(
        &dir,
        &commit,
        &si,
        &fields,
        None,
        &DocsWithTerms::new(),
        &mut stats,
        &mut checks,
    );
    assert!(
        checks.is_empty(),
        "doc values and norms must stay quiet when the segment has no \
             .dvm/.dvd/.nvm/.nvd -- `fnm.doc_values_vs_files` and \
             `fnm.norms_vs_files` above already reported it: {:?}",
        checks.iter().map(|c| &c.name).collect::<Vec<_>>()
    );

    // The index sort and the soft-deletes count are different: their
    // *claims* live in the `.si` and `.fnm`, not in the doc-values files,
    // so `fnm.doc_values_vs_files` does not cover them and staying quiet
    // would leave a declared sort order verified by nothing at all. They
    // report a skip -- which is not a pass.
    check_index_sort(&dir, &commit, &si, &fields, &mut checks);
    check_soft_deletes(&dir, &commit, &si, &fields, &mut checks);
    assert_eq!(
        checks
            .iter()
            .map(|c| (c.name.as_str(), c.outcome))
            .collect::<Vec<_>>(),
        [
            ("sort.docs_in_index_sort_order", Outcome::Skipped),
            ("soft_deletes.count_matches", Outcome::Skipped),
        ],
        "{checks:?}"
    );
    assert!(!checks.iter().any(|c| c.passed()));
    checks.clear();
    // Vectors and points are the two deliberate exceptions: neither has
    // an `fnm.*_vs_files` counterpart above, so their own `open` check is
    // the only place a missing file group can be reported -- and it must
    // fire, or the absence goes unreported entirely.
    check_vectors(&dir, &commit, &si, &fields, &mut stats, &mut checks);
    check_points_structural_invariants(&dir, &commit, &si, &fields, &mut checks);
    let failed: Vec<&str> = checks
        .iter()
        .filter(|c| c.outcome == Outcome::Failed)
        .map(|c| c.name.as_str())
        .collect();
    assert_eq!(failed, ["vectors.open", "points.open"], "{checks:?}");
    // ... and each names the families it took down, rather than leaving
    // them absent from the result.
    assert_eq!(
        checks.iter().filter(|c| c.was_skipped()).count(),
        VECTOR_FAMILIES.len() + POINTS_FAMILIES.len(),
        "{checks:?}"
    );
    assert!(!checks.iter().any(|c| c.passed()));
    std::fs::remove_dir_all(&dst).ok();
}

/// `SegmentInfos.readCommit`'s three cross-`.si` header validations, none
/// of which had ever failed in a test: a segment older than the commit's
/// own recorded `minSegmentLuceneVersion`, a segment older than
/// `indexCreatedVersionMajor`, and a segment with no `minVersion` at all
/// once `indexCreatedVersionMajor >= 7`.
///
/// This port's `segment_infos::parse` never opens a `.si`, so these are
/// the only place the two headers are ever compared -- a commit that
/// claims to have been written by a version newer than the segments it
/// contains would otherwise parse, open and query perfectly while lying
/// about what wrote it (which is what every back-compat gate reads).
#[test]
fn a_commit_header_ahead_of_its_own_segment_is_caught() {
    use lucene_codecs::postings_writer::TermPostings;
    let dst = tempdir();
    let commit = write_postings_fixture(
        &dst,
        &[TermPostings {
            term: b"alpha".to_vec(),
            docs: vec![(0, 1), (1, 2)],
            positions: vec![],
            offsets: vec![],
            payload_bytes: vec![],
            payload_lengths: vec![],
        }],
        2,
        2,
        None,
    );
    let dir = FsDirectory::open(&dst);

    // The same segment, checked against a commit header that agrees with
    // it, must not report any of the three.
    let agreeing = SegmentInfos {
        id: CLAIMS_SEG_ID,
        generation: 1,
        format_version: 10,
        lucene_version: segment_infos::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        index_created_version_major: 6,
        version: 1,
        counter: 99,
        min_segment_lucene_version: Some(segment_infos::LuceneVersion {
            major: 9,
            minor: 0,
            bugfix: 0,
        }),
        segments: vec![],
        user_data: vec![],
    };
    let result = check_segment_in_commit(&dir, Some(&agreeing), &commit);
    assert!(
        result.all_passed(),
        "an agreeing commit header must pass: {:?}",
        result.failures()
    );

    // `indexCreatedVersionMajor = 6` above is deliberate: it is below
    // Java's `>= 7` gate, so the second and third checks do not even run.
    assert!(!result
        .checks
        .iter()
        .any(|c| c.name == "commit.segment_records_min_version"));

    let mut disagreeing = agreeing;
    disagreeing.min_segment_lucene_version = Some(segment_infos::LuceneVersion {
        major: 11,
        minor: 0,
        bugfix: 0,
    });
    disagreeing.index_created_version_major = 11;
    let result = check_segment_in_commit(&dir, Some(&disagreeing), &commit);
    for name in [
        "commit.segment_version_at_or_after_min",
        "commit.segment_version_at_or_after_created",
        "commit.segment_records_min_version",
    ] {
        let check = result
            .checks
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("{name} must have run"));
        assert!(!check.passed(), "{name} should have failed: {check:?}");
    }
    std::fs::remove_dir_all(&dst).ok();
}

/// `SegmentInfos.readCommit`'s `softDelCount` bound. `del_count`'s twin
/// was tested; this one -- and the combined
/// `delCount + softDelCount <= maxDoc` bound -- were not, so a commit
/// claiming more soft deletes than the segment has documents was
/// validated by nothing at all.
#[test]
fn soft_del_count_larger_than_max_doc_is_flagged() {
    let dir = FsDirectory::open(fixture_dir("live_docs_index"));
    let mut commit = read_commit(&dir);
    commit.soft_del_count = 1_000_000;
    let result = check_segment(&dir, &commit);
    for name in [
        "commit.soft_del_count_within_max_doc",
        "commit.del_plus_soft_del_within_max_doc",
    ] {
        let check = result.checks.iter().find(|c| c.name == name).unwrap();
        assert!(!check.passed(), "{check:?}");
    }
}

/// `CheckIndex.updateMaxSegmentName` parses every segment name as
/// `_<base36>`; a name it cannot parse leaves `maxSegmentName` at its
/// initial value, which silently turns the counter check into a
/// tautology. This port reports the unparsable name instead -- an arm
/// that had never fired, because every fixture in the repo is written by
/// a real `IndexWriter` and so is always well-formed.
#[test]
fn a_segment_name_that_is_not_base36_is_flagged() {
    let dir = FsDirectory::open(fixture_dir("blocktree_index"));
    let mut infos = segment_infos::read_latest(&dir).unwrap();
    infos.segments[0].segment_name = "not-a-segment".to_string();
    let result = check_commit(&infos);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "commit.segment_names_well_formed")
        .unwrap();
    assert!(!check.passed(), "{check:?}");
    assert!(check.message.contains("not-a-segment"), "{check:?}");
}

/// `SegmentInfos.readCommit`'s LUCENE-6299 bound: the readers' total
/// `maxDoc` cannot exceed `IndexWriter.MAX_DOCS`. It is the one
/// commit-level check that needs every segment's `.si` to have been
/// opened, which is why it is appended by `check_directory` rather than
/// computed in `check_commit` -- and it had never been observed to fire,
/// so the append was unverified too.
#[test]
fn a_commit_whose_segments_exceed_max_docs_is_flagged() {
    let dst = tempdir();
    let si = SegmentInfo {
        id: CLAIMS_SEG_ID,
        version: segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        min_version: None,
        // One document past `MAX_DOCS`, still inside `i32`.
        doc_count: (MAX_DOCS + 1) as i32,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: vec!["_0.si".to_string()],
        attributes: vec![],
        index_sort: None,
    };
    std::fs::write(dst.join("_0.si"), segment_info::write(&si, "")).unwrap();
    let infos = SegmentInfos {
        id: CLAIMS_SEG_ID,
        generation: 1,
        format_version: 10,
        lucene_version: segment_infos::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        index_created_version_major: 10,
        version: 1,
        counter: 99,
        min_segment_lucene_version: None,
        segments: vec![claiming_commit()],
        user_data: vec![],
    };
    let dir = FsDirectory::open(&dst);
    segment_infos::write(&infos, &dir).expect("write segments_1");

    let dir = FsDirectory::open(&dst);
    let results = check_directory(&dir).expect("read segments_N");
    let check = results[0]
        .checks
        .iter()
        .find(|c| c.name == "commit.total_max_doc_within_bounds")
        .expect("the total-maxDoc check must run");
    assert!(!check.passed(), "{check:?}");
    assert_eq!(results[1].max_doc, Some((MAX_DOCS + 1) as i32));
    std::fs::remove_dir_all(&dst).ok();
}

// -- the term dictionary's own claims, rewritten -------------------------
//
// `postings_writer` computes every term statistic from the postings it is
// given, so it cannot be *asked* for a dictionary that lies -- which is
// precisely the file `check_postings` exists to reject. These two helpers
// decode the written `.tim` term-statistics region and the `.tmd`
// field-summary record, let a test change one claim, and re-encode. That
// is a negative control: the bytes are real writer output
// apart from the one semantic claim under test, and every checksum is
// re-signed so `file:*`'s CRC cannot "catch" the corruption first.

const TIM_CODEC: &str = "BlockTreeTermsDict";
const TMD_CODEC: &str = "BlockTreeTermsMeta";
const TMD_POSTINGS_CODEC: &str = "Lucene104PostingsWriterTerms";

/// A minimal varint writer over the same encoding
/// `lucene_store::data_output` uses, spelled out here because the
/// negative deltas below are exactly what a `write_vlong` refuses to
/// emit: `.tim` stores `totalTermFreq - docFreq` as an *unsigned* vlong,
/// so a `totalTermFreq < docFreq` can only be spelled as the ten-byte
/// encoding of a negative `i64` -- and `DataInput::read_vlong` accepts
/// it, which is what makes that check falsifiable at all.
fn push_raw_vlong(out: &mut Vec<u8>, value: i64) {
    let mut v = value as u64;
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn push_vint(out: &mut Vec<u8>, value: i32) {
    push_raw_vlong(out, i64::from(value) & 0xffff_ffff);
}

/// Rewrites the per-term `(docFreq, totalTermFreq)` pairs in a hand-built
/// fixture's single `.tim` leaf block, returning the new `.tim` bytes and
/// the new whole-file length the `.tmd` must record for it.
fn patch_tim_stats(
    tim: &[u8],
    suffix: &str,
    mut edit: impl FnMut(usize, &mut i32, &mut i64),
) -> Vec<u8> {
    use lucene_store::data_input::{DataInput, SliceInput};
    let start = codec_util::index_header_length(TIM_CODEC, suffix);
    let mut input = SliceInput::new(&tim[..tim.len() - codec_util::FOOTER_LENGTH]);
    input.seek(start).unwrap();
    let code = input.read_vint().unwrap();
    let ent_count = (code as u32 >> 1) as usize;
    let code_l = input.read_vlong().unwrap();
    // The whole field must be one leaf block at the start of the file --
    // true of the writer's output for a field of at most
    // `maxItemsInBlock` (48) terms, which is what these fixtures are. A
    // larger fixture would otherwise have only its first block patched.
    assert!(
        code & 1 == 1 && code_l & 4 == 4,
        "patch_tim_stats handles a field that is one leaf block"
    );
    let suffix_len = (code_l as u64 >> 3) as usize;
    // This walks exactly one block. The writer splits a field past 48
    // terms, and a patch applied to the first of several blocks would
    // leave the rest untouched while the test still passed -- so require
    // the one-block shape rather than assume it.
    assert_eq!(code_l & 0x04, 0x04, "patch_tim_stats needs a leaf block");
    assert_eq!(
        code_l & 0x03,
        0,
        "patch_tim_stats needs uncompressed suffixes"
    );
    let suffix_at = input.position();
    input.seek(suffix_at + suffix_len).unwrap();
    let suffix_bytes = &tim[suffix_at..suffix_at + suffix_len];
    let sl_code = input.read_vint().unwrap();
    // `sl_code & 1`: every suffix length is the same one byte, stored
    // once (`writeBlock`'s all-equal case).
    let sl_len = if sl_code & 1 == 1 {
        1
    } else {
        (sl_code as u32 >> 1) as usize
    };
    let sl_at = input.position();
    input.seek(sl_at + sl_len).unwrap();
    let stats_len = input.read_vint().unwrap() as usize;
    let stats_at = input.position();
    input.seek(stats_at + stats_len).unwrap();
    let meta_len = input.read_vint().unwrap() as usize;
    let meta_at = input.position();
    assert_eq!(
        meta_at + meta_len,
        tim.len() - codec_util::FOOTER_LENGTH,
        "patch_tim_stats needs a .tim holding exactly one block"
    );

    // Decode the stats region. The fixtures this patches have no
    // singleton term (`docFreq == 1` and `totalTermFreq == 1`), which is
    // the only kind the writer run-length encodes, so every entry is
    // `vint(docFreq << 1)` plus, for a field with freqs,
    // `vlong(totalTermFreq - docFreq)` -- asserted below.
    //
    // **Only valid for a field that indexes freqs.** A DOCS-only field
    // writes no delta at all (`StatsWriter`'s `hasFreqs` guard in
    // `blocktree_writer`, and `blocktree` reads it back the
    // same way), so the "read a delta if bytes are left" rule below would
    // consume the *next* term's `docFreq` token. Re-encoding is canonical,
    // so the identity round trip would still be byte-identical and would
    // not catch it -- hence the assertion rather than a comment.
    let mut stats_in = SliceInput::new(&tim[stats_at..stats_at + stats_len]);
    let mut pairs: Vec<(i32, i64)> = Vec::new();
    while pairs.len() < ent_count {
        let token = stats_in.read_vint().unwrap();
        assert_eq!(
            token & 1,
            0,
            "a singleton run: patch_tim_stats needs a fixture without singleton terms"
        );
        let doc_freq = (token as u32 >> 1) as i32;
        assert!(
            stats_in.position() < stats_len,
            "patch_tim_stats only handles a field that indexes freqs"
        );
        let delta = stats_in.read_vlong().unwrap();
        pairs.push((doc_freq, i64::from(doc_freq) + delta));
    }
    assert_eq!(pairs.len(), ent_count);

    let mut stats = Vec::new();
    for (i, (doc_freq, total_term_freq)) in pairs.iter_mut().enumerate() {
        edit(i, doc_freq, total_term_freq);
        push_vint(&mut stats, (*doc_freq as u32 as i32) << 1);
        push_raw_vlong(
            &mut stats,
            total_term_freq.wrapping_sub(i64::from(*doc_freq)),
        );
    }

    let mut out = tim[..start].to_vec();
    push_vint(&mut out, code);
    push_raw_vlong(&mut out, code_l);
    out.extend_from_slice(suffix_bytes);
    push_vint(&mut out, sl_code);
    out.extend_from_slice(&tim[sl_at..sl_at + sl_len]);
    push_vint(&mut out, stats.len() as i32);
    out.extend_from_slice(&stats);
    push_vint(&mut out, meta_len as i32);
    out.extend_from_slice(&tim[meta_at..meta_at + meta_len]);
    out.extend_from_slice(&[0u8; codec_util::FOOTER_LENGTH]);
    let n = out.len();
    out[n - codec_util::FOOTER_LENGTH..n - 8]
        .copy_from_slice(&tim[tim.len() - codec_util::FOOTER_LENGTH..tim.len() - 8]);
    repair_checksum(&mut out);
    out
}

/// The `.tmd`'s one field-summary record, decoded so a test can change a
/// single claim.
struct TmdRecord {
    num_terms: i64,
    sum_total_term_freq: i64,
    sum_doc_freq: i64,
    doc_count: i32,
    min_term: Vec<u8>,
    max_term: Vec<u8>,
    /// The whole `.tim` file's length, which the `.tmd` records so the
    /// reader can checksum it -- so a test that rewrites the `.tim` has
    /// to keep this in step.
    terms_length: i64,
}

fn patch_tmd(tmd: &[u8], suffix: &str, edit: impl FnOnce(&mut TmdRecord)) -> Vec<u8> {
    use lucene_store::data_input::{DataInput, SliceInput};
    let start = codec_util::index_header_length(TMD_CODEC, suffix)
        + codec_util::index_header_length(TMD_POSTINGS_CODEC, suffix);
    let body_end = tmd.len() - codec_util::FOOTER_LENGTH;
    let mut input = SliceInput::new(&tmd[..body_end]);
    input.seek(start).unwrap();
    let block_size = input.read_vint().unwrap();
    let num_fields = input.read_vint().unwrap();
    assert_eq!(num_fields, 1, "this helper handles one-field fixtures only");
    let field_number = input.read_vint().unwrap();
    let num_terms = input.read_vlong().unwrap();
    // `read_freq_pair` reads *one* vlong for a DOCS-only field and
    // returns it as both sums, so reading two here would consume
    // `docCount` as `sumDocFreq` and shift the whole record. The
    // re-encoding is canonical, so the identity round trip would still be
    // byte-identical and would not catch it -- assert instead.
    let sum_total_term_freq = input.read_vlong().unwrap();
    let sum_doc_freq = input.read_vlong().unwrap();
    let doc_count = input.read_vint().unwrap();
    let read_bytes = |input: &mut SliceInput<'_>| {
        let len = input.read_vint().unwrap() as usize;
        let at = input.position();
        input.seek(at + len).unwrap();
        tmd[at..at + len].to_vec()
    };
    let min_term = read_bytes(&mut input);
    let max_term = read_bytes(&mut input);
    let index_start = input.read_vlong().unwrap();
    let root_fp = input.read_vlong().unwrap();
    let index_end = input.read_vlong().unwrap();
    // `write_i64` is little-endian in Lucene 9+ (`DataOutput.writeLong`),
    // which is only observable once a test changes the value: a
    // big-endian round trip is byte-identical and silently wrong.
    let index_length = i64::from_le_bytes(tmd[body_end - 16..body_end - 8].try_into().unwrap());
    let terms_length = i64::from_le_bytes(tmd[body_end - 8..body_end].try_into().unwrap());

    let mut record = TmdRecord {
        num_terms,
        sum_total_term_freq,
        sum_doc_freq,
        doc_count,
        min_term,
        max_term,
        terms_length,
    };
    edit(&mut record);

    let mut out = tmd[..start].to_vec();
    push_vint(&mut out, block_size);
    push_vint(&mut out, num_fields);
    push_vint(&mut out, field_number);
    push_raw_vlong(&mut out, record.num_terms);
    push_raw_vlong(&mut out, record.sum_total_term_freq);
    push_raw_vlong(&mut out, record.sum_doc_freq);
    push_vint(&mut out, record.doc_count);
    push_vint(&mut out, record.min_term.len() as i32);
    out.extend_from_slice(&record.min_term);
    push_vint(&mut out, record.max_term.len() as i32);
    out.extend_from_slice(&record.max_term);
    push_raw_vlong(&mut out, index_start);
    push_raw_vlong(&mut out, root_fp);
    push_raw_vlong(&mut out, index_end);
    out.extend_from_slice(&index_length.to_le_bytes());
    out.extend_from_slice(&record.terms_length.to_le_bytes());
    out.extend_from_slice(&[0u8; codec_util::FOOTER_LENGTH]);
    let n = out.len();
    out[n - codec_util::FOOTER_LENGTH..n - 8].copy_from_slice(&tmd[body_end..tmd.len() - 8]);
    repair_checksum(&mut out);
    out
}

/// Two terms with freqs, the shape both patchers work on.
fn two_term_postings() -> Vec<lucene_codecs::postings_writer::TermPostings> {
    use lucene_codecs::postings_writer::TermPostings;
    vec![
        TermPostings {
            term: b"alpha".to_vec(),
            docs: vec![(0, 2), (1, 3)],
            positions: vec![],
            offsets: vec![],
            payload_bytes: vec![],
            payload_lengths: vec![],
        },
        TermPostings {
            term: b"beta".to_vec(),
            docs: vec![(1, 1), (2, 4)],
            positions: vec![],
            offsets: vec![],
            payload_bytes: vec![],
            payload_lengths: vec![],
        },
    ]
}

/// The two patchers must be *identity* transforms when nothing is
/// changed. Without this the negative controls below would be worthless:
/// a test that fails because the re-encoding is broken proves nothing
/// about the check it names.
#[test]
fn rewriting_the_tim_and_tmd_unchanged_leaves_the_segment_clean() {
    let dst = tempdir();
    let commit = write_postings_fixture(&dst, &two_term_postings(), 3, 3, None);
    let tim_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.tim"));
    let tmd_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.tmd"));
    let tim = std::fs::read(&tim_path).unwrap();
    let tmd = std::fs::read(&tmd_path).unwrap();
    let round_tripped_tim = patch_tim_stats(&tim, POSTINGS_SUFFIX, |_, _, _| {});
    let round_tripped_tmd = patch_tmd(&tmd, POSTINGS_SUFFIX, |_| {});
    assert_eq!(round_tripped_tim, tim, ".tim round-trip changed the bytes");
    assert_eq!(round_tripped_tmd, tmd, ".tmd round-trip changed the bytes");
    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    assert!(result.all_passed(), "{:?}", result.failures());
    std::fs::remove_dir_all(&dst).ok();
}

/// `checkFields`' per-term statistic bounds, one arm per case. Every one
/// of these had never fired: the only way to reach them is a `.tim`
/// stats region that disagrees with the postings it indexes, and no
/// writer in this port can be asked to produce one.
///
/// The second case is the interesting one. `.tim` stores
/// `totalTermFreq - docFreq` as an unsigned vlong, so `totalTermFreq <
/// docFreq` looks unreachable -- and it would be, except that this port's
/// `read_vlong` accepts the ten-byte encoding whose top bit lands in bit
/// 63 (a deliberate divergence from Java's nine-byte cap, recorded in
/// `data_input.rs`). So the check *is* falsifiable, and this pins the
/// reasoning: if `read_vlong` ever gains Java's cap, this case becomes
/// unreachable and the arm should go the way of the five deleted in c25.
#[test]
fn every_per_term_statistic_bound_reports_its_own_violation() {
    // (label, edit, the check that must fail, the checks that must not)
    struct Case {
        label: &'static str,
        doc_freq: i32,
        total_term_freq: i64,
        expected_message: &'static str,
    }
    let cases = [
        Case {
            label: "docFreq and totalTermFreq both zero",
            doc_freq: 0,
            total_term_freq: 0,
            expected_message: "totalTermFreq=0 is not > 0",
        },
        Case {
            label: "totalTermFreq below docFreq",
            doc_freq: 2,
            total_term_freq: 1,
            expected_message: "totalTermFreq=1 < docFreq=2",
        },
    ];
    for case in cases {
        let dst = tempdir();
        let commit = write_postings_fixture(&dst, &two_term_postings(), 3, 3, None);
        let tim_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.tim"));
        let tim = std::fs::read(&tim_path).unwrap();
        let patched = patch_tim_stats(&tim, POSTINGS_SUFFIX, |i, df, ttf| {
            if i == 0 {
                *df = case.doc_freq;
                *ttf = case.total_term_freq;
            }
        });
        std::fs::write(&tim_path, &patched).unwrap();
        let tmd_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.tmd"));
        let tmd = std::fs::read(&tmd_path).unwrap();
        let new_len = patched.len() as i64;
        std::fs::write(
            &tmd_path,
            patch_tmd(&tmd, POSTINGS_SUFFIX, |r| r.terms_length = new_len),
        )
        .unwrap();

        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let check = result
            .checks
            .iter()
            .find(|c| c.name == "postings.term_stats:body")
            .unwrap_or_else(|| panic!("{}: {:?}", case.label, result.failures()));
        assert!(!check.passed(), "{}: {check:?}", case.label);
        assert!(
            check.message.contains(case.expected_message),
            "{}: expected {:?} in {:?}",
            case.label,
            case.expected_message,
            check.message
        );
        std::fs::remove_dir_all(&dst).ok();
    }

    // `docFreq <= 0` is a separate named check, and a term with
    // `docFreq == 0` is the only input that reaches it.
    let dst = tempdir();
    let commit = write_postings_fixture(&dst, &two_term_postings(), 3, 3, None);
    let tim_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.tim"));
    let tim = std::fs::read(&tim_path).unwrap();
    let patched = patch_tim_stats(&tim, POSTINGS_SUFFIX, |i, df, ttf| {
        if i == 0 {
            *df = 0;
            *ttf = 0;
        }
    });
    std::fs::write(&tim_path, &patched).unwrap();
    let tmd_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.tmd"));
    let tmd = std::fs::read(&tmd_path).unwrap();
    let new_len = patched.len() as i64;
    std::fs::write(
        &tmd_path,
        patch_tmd(&tmd, POSTINGS_SUFFIX, |r| r.terms_length = new_len),
    )
    .unwrap();
    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "postings.doc_freq_positive:body")
        .expect("the docFreq check must run");
    assert!(!check.passed(), "{check:?}");
    assert!(check.message.contains("docFreq=0 is not > 0"), "{check:?}");
    std::fs::remove_dir_all(&dst).ok();
}

/// The `totalTermFreq` accumulator's overflow arm (this sum is
/// `checked_add` precisely so a `.tmd` claiming `i64::MAX` cannot be made
/// to agree with itself). The arm reports and stops the walk.
#[test]
fn per_term_total_term_freqs_that_overflow_i64_are_reported_not_summed() {
    let dst = tempdir();
    let commit = write_postings_fixture(&dst, &two_term_postings(), 3, 3, None);
    let tim_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.tim"));
    let tim = std::fs::read(&tim_path).unwrap();
    let patched = patch_tim_stats(&tim, POSTINGS_SUFFIX, |_, _, ttf| *ttf = i64::MAX);
    std::fs::write(&tim_path, &patched).unwrap();
    let tmd_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.tmd"));
    let tmd = std::fs::read(&tmd_path).unwrap();
    let new_len = patched.len() as i64;
    std::fs::write(
        &tmd_path,
        patch_tmd(&tmd, POSTINGS_SUFFIX, |r| r.terms_length = new_len),
    )
    .unwrap();

    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "postings.term_stats:body")
        .expect("the term-statistic check must run");
    assert!(!check.passed(), "{check:?}");
    assert!(
        check
            .message
            .contains("the per-term totalTermFreq values overflow i64"),
        "{check:?}"
    );
    std::fs::remove_dir_all(&dst).ok();
}

/// A `.tmd` whose `maxTerm` is smaller than a term the dictionary
/// actually holds. `minTerm`/`maxTerm` are what a `TermRangeQuery`'s
/// pruning and `Terms.getMax()` read without ever looking at the
/// dictionary, so a wrong one silently drops matches -- and the
/// `maxTerm` half of the pair had never been observed to fire.
#[test]
fn a_tmd_max_term_below_the_dictionary_is_caught() {
    let dst = tempdir();
    let commit = write_postings_fixture(&dst, &two_term_postings(), 3, 3, None);
    let tmd_path = dst.join(format!("_0_{POSTINGS_SUFFIX}.tmd"));
    let tmd = std::fs::read(&tmd_path).unwrap();
    std::fs::write(
        &tmd_path,
        patch_tmd(&tmd, POSTINGS_SUFFIX, |r| r.max_term = b"aaa".to_vec()),
    )
    .unwrap();

    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let stats = result
        .checks
        .iter()
        .find(|c| c.name == "postings.term_stats:body")
        .unwrap();
    assert!(!stats.passed(), "{stats:?}");
    assert!(
        stats.message.contains("sorts after .tmd's maxTerm"),
        "{stats:?}"
    );
    let summary = result
        .checks
        .iter()
        .find(|c| c.name == "postings.field_summary:body")
        .unwrap();
    assert!(!summary.passed(), "{summary:?}");
    assert!(summary.message.contains("maxTerm"), "{summary:?}");
    std::fs::remove_dir_all(&dst).ok();
}

/// A field whose `.fnm` says it indexes positions (and offsets, and
/// payloads) in a segment whose file list has no `.pos`/`.pay`.
///
/// `Lucene104PostingsWriter` writes a `.pay` exactly when some field
/// indexes offsets or stores payloads, and a `.pos` exactly when some
/// field indexes positions -- so those two files are the only
/// *independent on-disk witness* the `.fnm`'s positional flags have
/// (`hasFreqs`/`hasPositions` have none, so they are not checked). Both
/// arms, and the two decode paths
/// that then decline to read positions rather than panicking, had never
/// run.
#[test]
fn a_positional_field_whose_segment_lists_no_pos_or_pay_is_caught() {
    let dst = tempdir();
    let mut commit = write_positional_postings_fixture(&dst, 300);
    // Re-write the `.si` with `.pos`/`.pay` dropped from its file list,
    // which is what an interrupted flush (or a lost file) looks like.
    let si_bytes = std::fs::read(dst.join("_0.si")).unwrap();
    let mut si = segment_info::parse(&si_bytes, &commit.segment_id).unwrap();
    si.files
        .retain(|f| !f.ends_with(".pos") && !f.ends_with(".pay"));
    std::fs::write(dst.join("_0.si"), segment_info::write(&si, "")).unwrap();
    commit.segment_name = "_0".to_string();

    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let shape = result
        .checks
        .iter()
        .find(|c| c.name == "postings.term_dict_shape:body")
        .expect("the term-dictionary shape check must run");
    assert!(!shape.passed(), "{shape:?}");
    assert!(shape.message.contains("no .pay file"), "{shape:?}");
    assert!(shape.message.contains("no .pos file"), "{shape:?}");
    // The walk still finishes: no positional check is reported, because
    // there are no positions to check, and nothing panics on the way.
    assert!(!result
        .checks
        .iter()
        .any(|c| c.name == "postings.positions_valid:body" && !c.passed()));
    std::fs::remove_dir_all(&dst).ok();
}

/// A `.tim` that is listed by the `.si` but absent from the directory:
/// the postings *file set* fails to open, as opposed to the postings
/// *readers* failing to parse. Those are two different arms and only the
/// second had ever run.
#[test]
fn a_listed_but_missing_tim_is_reported_by_postings_open() {
    let dst = tempdir();
    let commit = write_postings_fixture(&dst, &two_term_postings(), 3, 3, None);
    std::fs::remove_file(dst.join(format!("_0_{POSTINGS_SUFFIX}.tim"))).unwrap();
    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "postings.open")
        .expect("postings.open must be reported");
    assert!(!check.passed(), "{check:?}");
    // ... and the rest of the segment is still checked.
    assert!(result
        .checks
        .iter()
        .any(|c| c.name == "fnm.open" && c.passed()));
    std::fs::remove_dir_all(&dst).ok();
}

/// A points field whose leaves reference documents the segment does not
/// have. Java's `VerifyPointsVisitor.visit` rejects both the per-point
/// doc id and the field-level `docCount`; here the two arms sit next to
/// each other and neither had fired -- so a `.kdd` pointing at documents
/// past `maxDoc` (every `PointRangeQuery` on it then collecting garbage
/// doc ids) would have been reported only by the weaker distinct-count
/// mismatch.
#[test]
fn points_leaves_referencing_documents_past_max_doc_are_caught() {
    let points: Vec<(i32, Vec<u8>)> = (0..5)
        .map(|d| (d, vec![0, 0, 0, 0, 0, 0, 0, d as u8]))
        .collect();
    let dst = tempdir();
    // The tree is written for five documents; the `.si` claims two.
    let commit = write_points_fixture(&dst, &points, 2, None);
    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "points.doc_count_matches:loc")
        .expect("the points doc-count check must run");
    assert!(!check.passed(), "{check:?}");
    assert!(check.message.contains("is outside 0..2"), "{check:?}");
    assert!(
        check.message.contains("but the segment has maxDoc=2"),
        "{check:?}"
    );
    std::fs::remove_dir_all(&dst).ok();
}

/// A vector segment whose `.si` lists `.vec`/`.vemf`/`.vem`/`.vex` files
/// that are not in the directory. Every one of the four is opened by a
/// separate `dir.open(..)` whose failure arm reports under its own
/// subsystem name -- and those arms are *different* from the ones the
/// file-replacement test drives, which reach the readers' own parse
/// failures. A file that is simply gone is the more common accident of
/// the two (an interrupted copy, a partially-restored snapshot).
#[test]
fn vector_files_listed_but_absent_are_reported_by_their_own_open_checks() {
    for (victim, expected) in [(".vec", "vectors.open"), (".vex", "hnsw.open")] {
        let dst = copy_fixture("vectors_index");
        std::fs::remove_file(find_file(&dst, victim)).unwrap();
        let dir = FsDirectory::open(&dst);
        let results = check_directory(&dir).expect("the commit itself is untouched");
        let failed = failed_names(&results);
        assert!(
            failed.iter().any(|n| n == expected),
            "a missing {victim} was not reported by {expected}: {failed:?}"
        );
        std::fs::remove_dir_all(&dst).ok();
    }
}

/// A field that declares a doc-values skip index in `.fnm` while the
/// segment's file list has no `.dvs`. Java's `checkDocValueSkipper` is
/// only ever handed a skipper that exists; this port has to decide what
/// to do when the `.fnm` promises one and the segment does not carry it,
/// and it reports rather than skipping -- a decision that had never been
/// executed.
#[test]
fn a_field_declaring_a_skip_index_with_no_dvs_file_is_caught() {
    let dst = copy_fixture("doc_values_skip_index");
    let si_path = find_file(&dst, ".si");
    let commit = {
        let dir = FsDirectory::open(&dst);
        read_commit(&dir)
    };
    let si_bytes = std::fs::read(&si_path).unwrap();
    let mut si = segment_info::parse(&si_bytes, &commit.segment_id).unwrap();
    assert!(si.files.iter().any(|f| f.ends_with(".dvs")));
    si.files.retain(|f| !f.ends_with(".dvs"));
    std::fs::write(&si_path, segment_info::write(&si, "")).unwrap();

    let dir = FsDirectory::open(&dst);
    let results = check_directory(&dir).expect("the commit itself is untouched");
    let failed: Vec<String> = failed_names(&results)
        .into_iter()
        .filter(|n| n.starts_with("doc_values.skipper:"))
        .collect();
    assert!(!failed.is_empty(), "{:?}", failed_names(&results));
    let message = results
        .iter()
        .flat_map(|r| &r.checks)
        .find(|c| c.name == failed[0])
        .map(|c| c.message.clone())
        .unwrap();
    assert!(message.contains(".dvs file"), "{message}");
    std::fs::remove_dir_all(&dst).ok();
}

/// A `.si` that declares an index sort over a field its own `.fnm` does
/// not have. `check_index_sort`'s error arm reports it as a single failed
/// `sort.docs_in_index_sort_order` rather than panicking on the missing
/// field or -- worse -- skipping the sort check for a segment that
/// claims to be sorted. It had never run.
#[test]
fn an_index_sort_naming_a_field_the_fnm_lacks_is_caught() {
    let dst = tempdir();
    let commit = write_sorted_dv_fixture(&dst, &[1, 2, 3], sort_asc(), false, 0);
    let si_path = dst.join("_0.si");
    let si_bytes = std::fs::read(&si_path).unwrap();
    let mut si = segment_info::parse(&si_bytes, &commit.segment_id).unwrap();
    for sf in si.index_sort.iter_mut().flatten() {
        sf.field = "no-such-field".to_string();
    }
    std::fs::write(&si_path, segment_info::write(&si, "")).unwrap();

    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "sort.docs_in_index_sort_order")
        .expect("the sort check must run");
    assert!(!check.passed(), "{check:?}");
    assert!(check.message.contains("no-such-field"), "{check:?}");
    std::fs::remove_dir_all(&dst).ok();
}

/// The soft-deletes count check's error arm: a segment whose `.fnm`
/// marks a field as the soft-deletes field but whose `.dvm` carries no
/// entry for it. Java reaches the same state through
/// `DocValues.getNumeric` returning an empty iterator, and counts zero;
/// this port reports it, because a soft-deletes field with no column is
/// a `softDelCount` nothing can be verified against.
#[test]
fn a_soft_deletes_field_with_no_doc_values_entry_is_caught() {
    let dst = tempdir();
    // The fixture writes one NUMERIC column for field number 0 and marks
    // it as the soft-deletes field; renumbering the `.fnm` entry leaves
    // the `.dvm` entry unreachable.
    let commit = write_sorted_dv_fixture(&dst, &[1, 2, 3], None, true, 3);
    let fnm_path = dst.join("_0.fnm");
    let fnm = std::fs::read(&fnm_path).unwrap();
    let mut fields = field_infos::parse(&fnm, &commit.segment_id, "").unwrap();
    fields.fields[0].number = 7;
    std::fs::write(
        &fnm_path,
        field_infos::write(&fields.fields, &commit.segment_id, ""),
    )
    .unwrap();

    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "soft_deletes.count_matches")
        .expect("the soft-deletes check must run");
    assert!(!check.passed(), "{check:?}");
    std::fs::remove_dir_all(&dst).ok();
}

/// `hnsw.entry_point_reachable`'s one *failing* arm: an entry point that
/// reaches nothing but itself on a level that has more than one node.
///
/// Java reports connectedness without ever failing on it (it tolerates
/// historically-disconnected graphs), and so does this port -- except for
/// this degenerate case, which is not a quality issue but a graph whose
/// search can never return more than one document however large `k` is.
/// That distinction is the whole point of the check and it had never been
/// executed: the `.vex` corruption sweep cannot reach it, because
/// emptying the *entry node's* neighbour list specifically is not
/// something a byte flip does often.
///
/// Driven through this port's own `.vem`/`.vex` writer rather than by
/// corrupting bytes, for the same reason the term-vector arms are
/// driven at the function: a graph is a shape, not a byte, and this is the only
/// way to ask for a *specific* one.
#[test]
fn an_hnsw_entry_point_that_reaches_only_itself_is_caught() {
    use lucene_codecs::hnsw::{NeighborArray, OnHeapHnswGraph};
    use lucene_codecs::hnsw_vectors::{write_hnsw_vectors, HnswVectorsField};

    const GRAPH_SEG_ID: [u8; ID_LENGTH] = [41u8; ID_LENGTH];
    const GRAPH_SUFFIX: &str = "Lucene99HnswVectorsFormat_0";

    // `connect_entry` decides whether node 0 -- the entry point of a
    // single-level graph -- has any neighbours at all. Everything else
    // about the two graphs is identical, so the only difference the check
    // can be reacting to is the one under test.
    let build = |connect_entry: bool| {
        let mut graph = OnHeapHnswGraph::with_size(16, 3);
        for node in 0..3 {
            graph.add_node(0, node);
        }
        // One level, entry point node 0 -- the shape every small graph
        // this port builds has (`entry_level` defaults to 1, which would
        // describe a second, empty level).
        assert!(graph.try_set_new_entry_node(0, 0));
        let link = |g: &mut OnHeapHnswGraph, from: i32, to: &[i32]| {
            let mut arr = NeighborArray::new(33, true);
            for &n in to {
                arr.add_in_order(n, 1.0);
            }
            *g.neighbors_mut(0, from) = arr;
        };
        if connect_entry {
            link(&mut graph, 0, &[1, 2]);
        } else {
            link(&mut graph, 0, &[]);
        }
        link(&mut graph, 1, &[2]);
        link(&mut graph, 2, &[1]);
        graph
    };

    let field = field_infos::FieldInfo {
        name: "vec".to_string(),
        number: 0,
        store_term_vectors: false,
        omit_norms: true,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: field_infos::IndexOptions::None,
        doc_values_type: field_infos::DocValuesType::None,
        doc_values_skip_index_type: field_infos::DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 4,
        vector_encoding: field_infos::VectorEncoding::Float32,
        vector_similarity_function: field_infos::VectorSimilarityFunction::Euclidean,
    };

    let run = |connect_entry: bool| -> Vec<Check> {
        let graph = build(connect_entry);
        let (vex, vem) = write_hnsw_vectors(
            &[HnswVectorsField {
                field_number: 0,
                encoding: field_infos::VectorEncoding::Float32,
                similarity: field_infos::VectorSimilarityFunction::Euclidean,
                dimension: 4,
                count: 3,
                graph: Some(&graph),
                m: 16,
            }],
            &GRAPH_SEG_ID,
            GRAPH_SUFFIX,
        )
        .expect("hand-built HNSW graph must write cleanly");
        let dst = tempdir();
        std::fs::write(dst.join(format!("_0_{GRAPH_SUFFIX}.vex")), &vex).unwrap();
        std::fs::write(dst.join(format!("_0_{GRAPH_SUFFIX}.vem")), &vem).unwrap();
        let si = SegmentInfo {
            id: GRAPH_SEG_ID,
            version: segment_info::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 3,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec![
                format!("_0_{GRAPH_SUFFIX}.vex"),
                format!("_0_{GRAPH_SUFFIX}.vem"),
            ],
            attributes: vec![],
            index_sort: None,
        };
        let commit = segment_infos::SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: GRAPH_SEG_ID,
            codec_name: "Lucene104".to_string(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
            ..Default::default()
        };
        let dir = FsDirectory::open(&dst);
        let mut checks = Vec::new();
        check_hnsw_graphs(&dir, &commit, &si, &[&field], GRAPH_SUFFIX, &mut checks);
        std::fs::remove_dir_all(&dst).ok();
        checks
    };

    // Positive control: the same three nodes, with the entry point wired
    // into them, pass and report the connectedness ratio in the message.
    let good = run(true);
    let reachable = good
        .iter()
        .find(|c| c.name == "hnsw.entry_point_reachable:vec")
        .unwrap_or_else(|| panic!("the connectedness check must run: {good:?}"));
    assert!(reachable.passed(), "{reachable:?}");
    assert_eq!(reachable.message, "3/3 nodes reachable on level 0");
    assert!(good.iter().all(|c| c.passed()), "{good:?}");

    let bad = run(false);
    let reachable = bad
        .iter()
        .find(|c| c.name == "hnsw.entry_point_reachable:vec")
        .expect("the connectedness check must run");
    assert!(!reachable.passed(), "{reachable:?}");
    assert!(
        reachable.message.contains("reaches 1 of 3"),
        "{reachable:?}"
    );
    // ... and nothing else: an isolated entry point is not an
    // out-of-level or repeated neighbour, and must not be reported as one.
    assert!(
        bad.iter()
            .filter(|c| !c.passed())
            .all(|c| c.name == "hnsw.entry_point_reachable:vec"),
        "{bad:?}"
    );
}

/// Re-signed body corruption of `.fdt` and `.tvd`: the two "decode every
/// document, deleted ones included" walks Java's `testStoredFields` and
/// `testTermVectors` perform. Both walks existed and both had only ever
/// been observed *succeeding* -- the arm that reports a document that
/// will not decode had never run for either, which is the half of the
/// check that does the work.
///
/// The file-replacement control reaches these subsystems by swapping a
/// whole file for another kind, which fails at `open`. A flipped byte in
/// the middle of a compressed chunk is the failure that gets past `open`
/// -- and past `file:*`, because the footer is re-signed -- and lands in
/// the per-document decode. It found two real defects in
/// `lucene-codecs/src/term_vectors.rs` on its first run: a
/// `prefixLength` off disk slicing the *previous* term (a panic) and a
/// chunk's claimed decompressed length sizing a `vec![0u8; n]` (an
/// **abort**, which no `catch_unwind` at the FFI boundary can intercept).
/// Both are fixed; this test is what keeps them fixed.
///
/// Measured floors: the number that was
/// measured when this was written is in the failure message, the floor is
/// ~85% of it, and every corruption is accounted for as caught-here,
/// caught-elsewhere or accepted so the denominator equals the sweep's own
/// iteration count.
#[test]
fn a_re_signed_body_corruption_is_reported_by_the_per_document_decode() {
    // (fixture, extension, the check that must fail, measured floor)
    let cases: &[(&str, &str, &str, usize)] = &[
        (
            "stored_fields_index",
            ".fdt",
            "stored_fields.every_doc_decodes",
            28,
        ),
        (
            "term_vectors_index",
            ".tvd",
            "term_vectors.every_doc_decodes",
            12,
        ),
    ];
    for (fixture, ext, expected, floor) in cases {
        let dst = copy_fixture(fixture);
        let path = find_file(&dst, ext);
        let original = std::fs::read(&path).unwrap();
        let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
        let mut caught_here = 0usize;
        let mut caught_by_other = 0usize;
        let mut accepted = 0usize;
        // A fixed uniform sample rather than every byte: the point is the
        // rate, and the sample has to be reproducible run to run.
        let stride = ((body_end - 64) / 40).max(1);
        let mut off = 64;
        while off < body_end {
            let mut bytes = original.clone();
            bytes[off] ^= 0xff;
            repair_checksum(&mut bytes);
            std::fs::write(&path, &bytes).unwrap();
            let dir = FsDirectory::open(&dst);
            let failed = match check_directory(&dir) {
                Ok(r) => failed_names(&r),
                Err(e) => vec![format!("commit.unreadable: {e}")],
            };
            if failed.iter().any(|n| n == expected) {
                caught_here += 1;
            } else if failed.is_empty() {
                accepted += 1;
            } else {
                caught_by_other += 1;
            }
            off += stride;
        }
        std::fs::write(&path, &original).unwrap();
        let total = caught_here + caught_by_other + accepted;
        assert!(
            total >= 8,
            "{fixture}: the {ext} sample was only {total} corruptions"
        );
        assert!(
            caught_here >= *floor,
            "{fixture}: only {caught_here} of {total} re-signed {ext} corruptions were \
                 reported by {expected} ({caught_by_other} caught by another check, \
                 {accepted} accepted); the floor is the measured number when this was written"
        );
        if *ext == ".fdt" {
            // As with the `.nvm` and `.dvd` rows: nothing else in the
            // module reads `.fdt`, so a corruption this walk misses is a
            // corruption nothing catches. The 14 it does miss are
            // stored-field *values*, which have no second copy anywhere
            // on disk -- a different value is still a valid one, the same
            // phenomenon as the `.tip` row.
            assert_eq!(
                caught_by_other, 0,
                "{fixture}: a .fdt corruption was reported by something other than \
                     {expected}"
            );
        }
        let dir = FsDirectory::open(&dst);
        assert!(
            check_directory(&dir)
                .unwrap()
                .iter()
                .all(|r| r.all_passed()),
            "{fixture} must be clean again"
        );
        std::fs::remove_dir_all(&dst).ok();
    }
}

/// A document that carries term vectors for a field number the `.fnm`
/// does not have at all.
///
/// `term_vectors.fields_marked_in_fnm` has two arms -- the field exists
/// but says `storeTermVectors=false`, and the field is not in `.fnm` --
/// and the file-replacement control drives only the first. The second is what a `.fnm`
/// rewritten by a field-infos *update* generation (or a lost `.fnm`
/// generation) actually looks like, and it is worse than the first: the
/// vectors are unaddressable rather than merely unexpected.
#[test]
fn term_vectors_for_a_field_number_the_fnm_lacks_are_caught() {
    let dst = copy_fixture("term_vectors_index");
    let fnm_path = find_file(&dst, ".fnm");
    let commit = {
        let dir = FsDirectory::open(&dst);
        read_commit(&dir)
    };
    let fnm = std::fs::read(&fnm_path).unwrap();
    let mut fields = field_infos::parse(&fnm, &commit.segment_id, "").unwrap();
    for fi in &mut fields.fields {
        fi.number += 50;
    }
    std::fs::write(
        &fnm_path,
        field_infos::write(&fields.fields, &commit.segment_id, ""),
    )
    .unwrap();

    let dir = FsDirectory::open(&dst);
    let results = check_directory(&dir).expect("the commit itself is untouched");
    let check = results
        .iter()
        .flat_map(|r| &r.checks)
        .find(|c| c.name == "term_vectors.fields_marked_in_fnm")
        .expect("the term-vector field-flag check must run");
    assert!(!check.passed(), "{check:?}");
    assert!(check.message.contains("not in .fnm at all"), "{check:?}");
    std::fs::remove_dir_all(&dst).ok();
}

/// A segment whose `.fnm` says a field is not indexed at all while the
/// segment still carries that field's term dictionary.
///
/// This is the survivable half of Java's "fieldsEnum inconsistent with
/// fieldInfos" pair (the other half is `x == x` here -- see
/// `check_postings`' comment). It is what a `.fnm` rewritten by a
/// field-infos update generation looks like when the update drops a
/// field's `IndexOptions`: every term in the dictionary becomes
/// unreachable, because every reader asks whether the field is indexed
/// before it opens the terms at all.
#[test]
fn a_term_dictionary_for_a_field_the_fnm_says_is_not_indexed_is_caught() {
    let dst = tempdir();
    let commit = write_postings_fixture(&dst, &two_term_postings(), 3, 3, None);
    let fnm_path = dst.join("_0.fnm");
    let fnm = std::fs::read(&fnm_path).unwrap();
    let mut fields = field_infos::parse(&fnm, &commit.segment_id, "").unwrap();
    fields.fields[0].index_options = field_infos::IndexOptions::None;
    std::fs::write(
        &fnm_path,
        field_infos::write(&fields.fields, &commit.segment_id, ""),
    )
    .unwrap();

    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "postings.field_in_fnm:body")
        .expect("the field-shape check must run");
    assert!(!check.passed(), "{check:?}");
    assert!(check.message.contains("indexOptions=None"), "{check:?}");
    std::fs::remove_dir_all(&dst).ok();
}

/// A `.fnm` that does not parse. `fnm.open` fails, and the report must still
/// name every postings, doc-values, norms, points, vector and field-flag
/// check that did not run, and must not show term-vector families as
/// passes because nothing looked.
///
/// A verifier that reports safety over a region it did not examine is
/// worse than one that omits the check, because the omission is at least
/// visible in the check list. This pins the fix: every family a failed
/// prerequisite takes down is named, and `all_passed()` is false.
#[test]
fn a_prerequisite_failure_names_every_family_it_takes_down() {
    let dst = copy_fixture("term_vectors_index");
    let fnm_path = find_file(&dst, ".fnm");
    // Truncating the `.fnm` body is the cheapest stand-in for a `.fnm` the
    // parser rejects: any `.fnm` that will not parse
    // reproduces the state exactly.
    std::fs::write(&fnm_path, b"not a field-infos file").unwrap();

    let dir = FsDirectory::open(&dst);
    let results = check_directory(&dir).expect("the commit itself is untouched");
    let segment = &results[1];

    assert!(!segment.all_passed());
    let failed: Vec<&str> = segment
        .checks
        .iter()
        .filter(|c| c.outcome == Outcome::Failed)
        .map(|c| c.name.as_str())
        .collect();
    assert!(failed.contains(&"fnm.open"), "{failed:?}");

    // The heart of it: the families that could not run are *present* in
    // the result, named, and not passes.
    let skipped: Vec<&str> = segment.skipped().iter().map(|c| c.name.as_str()).collect();
    for family in FAMILIES_BELOW_FNM {
        assert!(
            skipped.contains(family),
            "{family} must be reported as not run: {skipped:?}"
        );
    }
    // And specifically the three that must not be reported as *passing*.
    for family in [
        "term_vectors.fields_marked_in_fnm",
        "term_vectors.self_consistent",
        "term_vectors.match_postings",
    ] {
        let check = segment
            .checks
            .iter()
            .find(|c| c.name == family)
            .unwrap_or_else(|| panic!("{family} must appear in the result at all"));
        assert_eq!(
            check.outcome,
            Outcome::Skipped,
            "{family} must not report a pass over vectors nothing cross-checked"
        );
    }
    // Nothing that did run is claimed to have run against the `.fnm`:
    // `term_vectors.every_doc_decodes` still works (it needs no `.fnm`)
    // and must still be a real pass, so the skip is targeted, not a
    // blanket "give up".
    assert!(segment
        .checks
        .iter()
        .any(|c| c.name == "term_vectors.every_doc_decodes" && c.passed()));

    std::fs::remove_dir_all(&dst).ok();
}

/// A segment that **declares an index sort** but carries no doc-values
/// files: the one skip case where `all_passed()` must still return `false`.
///
/// `fnm.doc_values_vs_files` does not cover it -- a sort field that is
/// not in `.fnm` at all leaves no field claiming doc values, so that
/// check passes too. Without a dedicated check the segment's declared sort order
/// (which every merge, every early-terminating query and Lucene's own
/// `testSort` trust) would be verified by nothing whatsoever yet reported
/// as entirely healthy.
#[test]
fn a_declared_index_sort_with_no_doc_values_is_reported_as_unverified() {
    let dst = tempdir();
    let commit = write_sorted_dv_fixture(&dst, &[1, 2, 3], sort_asc(), false, 0);

    // The healthy control: with its doc values, the segment passes and
    // the sort check really ran.
    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    assert!(result.all_passed(), "{:?}", result.failures());
    assert!(result
        .checks
        .iter()
        .any(|c| c.name == "sort.docs_in_index_sort_order" && c.passed()));

    // Now take the doc-values files out of the `.si`'s file list *and*
    // the field out of `.fnm`, which is what makes `fnm.doc_values_vs_files`
    // pass and leaves nothing else to notice.
    let si_bytes = std::fs::read(dst.join("_0.si")).unwrap();
    let mut si = segment_info::parse(&si_bytes, &commit.segment_id).unwrap();
    si.files
        .retain(|f| !f.ends_with(".dvm") && !f.ends_with(".dvd"));
    std::fs::write(dst.join("_0.si"), segment_info::write(&si, "")).unwrap();
    std::fs::write(
        dst.join("_0.fnm"),
        field_infos::write(&[], &commit.segment_id, ""),
    )
    .unwrap();

    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "sort.docs_in_index_sort_order")
        .expect("the sort check must appear in the result even when it cannot run");
    assert_eq!(check.outcome, Outcome::Skipped, "{check:?}");
    assert!(
        !result.all_passed(),
        "a segment whose declared sort order was verified by nothing must not report \
             as healthy: {:?}",
        result.checks
    );
    // Nothing else noticed -- which is exactly why the skip has to exist.
    assert_eq!(
        result
            .checks
            .iter()
            .filter(|c| c.outcome == Outcome::Failed)
            .count(),
        0,
        "{:?}",
        result.failures()
    );
    std::fs::remove_dir_all(&dst).ok();
}

/// The suppression inventory, pinned so it cannot drift silently.
///
/// Prerequisites in this module can take a family of checks down
/// with them. The danger is not that they do -- there is genuinely
/// nothing to check once the file will not open -- it is doing it
/// *invisibly*, so a caller could not tell a region that was examined and
/// found healthy from one nothing looked at. Every one of them names its
/// casualties.
///
/// This test exists so that adding a new `*.open` arm without its
/// `skip_families` call fails the suite rather than quietly reintroducing
/// the hole. The lists are also what the batch report's
/// "what each leaves unguarded" table is derived from.
#[test]
fn every_prerequisite_names_the_families_it_takes_down() {
    // (prerequisite, the families it suppresses)
    let inventory: &[(&str, &[&str])] = &[
        ("si.open", FAMILIES_BELOW_SI),
        ("fnm.open", FAMILIES_BELOW_FNM),
        ("postings.open", FAMILIES_BELOW_POSTINGS),
        ("liv.open", LIV_FAMILIES),
        ("points.open", POINTS_FAMILIES),
        ("points.decode:<f>", POINTS_PER_FIELD),
        ("vectors.open", VECTOR_FAMILIES),
        ("hnsw.open", HNSW_FAMILIES),
        ("hnsw.open:<f>", HNSW_FAMILIES),
        // These four have their lists inline at the site, because each is
        // a single arm rather than a shared constant.
        (
            "term_vectors.open",
            &[
                "term_vectors.doc_count_matches_si",
                "term_vectors.every_doc_decodes",
                "term_vectors.fields_marked_in_fnm",
                "term_vectors.self_consistent",
                "term_vectors.match_postings",
            ],
        ),
        (
            "stored_fields.doc_count_matches_si",
            &["stored_fields.every_doc_decodes"],
        ),
        (
            "doc_values.open",
            &[
                "doc_values.values_decode",
                "doc_values.skipper",
                "doc_values.ords_dense",
                "doc_values.terms_sorted",
            ],
        ),
        (
            "norms.open",
            &[
                "norms.entries_name_real_norms_fields",
                "norms.entry_present",
                "norms.values_decode",
                "norms.agree_with_postings",
            ],
        ),
        // Three per-field prerequisites: a `.dvm`/`.nvm`/`.vemf` that
        // parses but has no entry for a field the `.fnm` says has the
        // format. The field's whole column then goes unread while every
        // other field's checks still run.
        (
            "doc_values.entry_present:<f>",
            &[
                "doc_values.values_decode",
                "doc_values.ords_dense",
                "doc_values.terms_sorted",
            ],
        ),
        (
            "norms.entry_present:<f>",
            &["norms.values_decode", "norms.agree_with_postings"],
        ),
        (
            "vectors.field_entry_matches_fnm:<f>",
            &["vectors.values_decode", "vectors.ord_to_doc"],
        ),
    ];

    assert_eq!(
        inventory.len(),
        16,
        "a prerequisite was added or removed without updating the audit"
    );
    let mut families: Vec<&str> = inventory.iter().flat_map(|(_, f)| *f).copied().collect();
    families.sort_unstable();
    families.dedup();
    // 43 names, of which 9 are the `<subsystem>.*` roll-ups the `.si`
    // and `.fnm` failures use: once the file that *names the fields* is
    // gone, the per-field expansion is unknowable, so the roll-up is the
    // honest granularity. The other 34 are concrete check families.
    assert_eq!(
        families.len(),
        43,
        "the set of suppressible check families changed: {families:?}"
    );
    assert_eq!(families.iter().filter(|f| f.ends_with(".*")).count(), 9);
    // No prerequisite may suppress itself -- that would be a check
    // reporting that it did not run because it did not run.
    for (because, suppressed) in inventory {
        assert!(
            !suppressed.contains(because),
            "{because} lists itself among what it takes down"
        );
    }
    // `si.open` is the widest: it is the only one that can leave the
    // whole segment unexamined, which is why it enumerates every family.
    assert_eq!(FAMILIES_BELOW_SI.len(), 16);
    // The three cross-checks are the sharpest, because a cross-check with
    // one side missing is the shape that must not report a pass.
    for cross in [
        "term_vectors.match_postings",
        "norms.agree_with_postings",
        "postings.*",
    ] {
        assert!(FAMILIES_BELOW_POSTINGS.contains(&cross));
    }
}

/// Every file the `.si` lists that is *gone* from the directory must be
/// reported by the check that opens it, by name.
///
/// Every one of these arms sits *inside* a
/// `(|| -> Result<..>{ ... })()` block whose failure arm is the only
/// thing standing between a missing file and a silently empty walk.
/// A partially-restored snapshot or an interrupted copy is the common
/// way to produce one, and it must not read as agreement.
///
/// Each row also asserts the isolation property: the *named* check fails.
/// `file:<name>` fails too (the file is genuinely absent, and
/// `check_files_exist_and_validate` says so first), which is correct and is
/// not what this test is about.
#[test]
fn every_listed_file_that_goes_missing_is_reported_by_its_own_open_check() {
    // (fixture, extension to delete, the check that must fail)
    let cases: &[(&str, &str, &str)] = &[
        (
            "blocktree_index",
            ".fdt",
            "stored_fields.doc_count_matches_si",
        ),
        (
            "blocktree_index",
            ".fdx",
            "stored_fields.doc_count_matches_si",
        ),
        (
            "blocktree_index",
            ".fdm",
            "stored_fields.doc_count_matches_si",
        ),
        ("term_vectors_index", ".tvd", "term_vectors.open"),
        ("term_vectors_index", ".tvx", "term_vectors.open"),
        ("term_vectors_index", ".tvm", "term_vectors.open"),
        ("doc_values_index", ".dvm", "doc_values.open"),
        ("doc_values_index", ".dvd", "doc_values.open"),
        ("doc_values_skip_index", ".dvs", "doc_values.open"),
        ("norms_index", ".nvd", "norms.open"),
        ("norms_index", ".nvm", "norms.open"),
        ("points_index", ".kdm", "points.open"),
        ("points_index", ".kdi", "points.open"),
        ("points_index", ".kdd", "points.open"),
        ("blocktree_index", ".tip", "postings.open"),
        ("blocktree_index", ".tmd", "postings.open"),
        ("blocktree_index", ".doc", "postings.open"),
        ("blocktree_index", ".pos", "postings.open"),
        ("blocktree_index", ".pay", "postings.open"),
        ("vectors_index", ".vemf", "vectors.open"),
        ("vectors_index", ".vem", "hnsw.open"),
    ];
    for (fixture, victim, expected) in cases {
        let dst = copy_fixture(fixture);
        std::fs::remove_file(find_file(&dst, victim)).unwrap();
        let dir = FsDirectory::open(&dst);
        let results = check_directory(&dir).expect("the commit itself is untouched");
        let failed = failed_names(&results);
        assert!(
            failed.iter().any(|n| n == expected),
            "{fixture}: a missing {victim} was not reported by {expected}: {failed:?}"
        );
        std::fs::remove_dir_all(&dst).ok();
    }
}

/// The three commit-level degradations that are *not* a corrupt segment:
/// a directory with no `segments_N` at all, a `.si` file that is listed
/// but absent (as opposed to present and unparsable, which
/// `corrupt_si_short_circuits_remaining_checks` covers), and a
/// `SegmentInfos` whose generation cannot be turned back into a file
/// name.
///
/// The last is the one worth having: `check_commit`'s result is labelled
/// with the `segments_N` name, and a negative generation makes
/// `segments_file_name` return `None`. Without the fallback the label
/// would be a panic inside the verifier.
#[test]
fn commit_level_degradations_are_reported_rather_than_panicking() {
    // (1) No commit at all.
    let empty = tempdir();
    let dir = FsDirectory::open(&empty);
    assert!(
        check_directory(&dir).is_err(),
        "a directory with no segments_N must not report a clean index"
    );
    std::fs::remove_dir_all(&empty).ok();

    // (2) A `.si` that is not there at all.
    let missing = tempdir();
    let dir = FsDirectory::open(&missing);
    let commit = segment_infos::SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id: [7u8; ID_LENGTH],
        codec_name: "Lucene104".to_string(),
        del_gen: -1,
        ..Default::default()
    };
    let result = check_segment(&dir, &commit);
    assert_eq!(result.checks[0].name, "si.open");
    assert!(!result.checks[0].passed(), "{:?}", result.checks[0]);
    std::fs::remove_dir_all(&missing).ok();

    // (3) A commit whose generation has no file name.
    let infos = SegmentInfos {
        id: [3u8; ID_LENGTH],
        generation: -1,
        format_version: 10,
        lucene_version: segment_infos::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        index_created_version_major: 10,
        version: 1,
        counter: 1,
        min_segment_lucene_version: None,
        segments: vec![],
        user_data: vec![],
    };
    let result = check_commit(&infos);
    assert_eq!(result.segment_name, "<invalid generation -1>");
}

const BYTE_VEC_SEG_ID: [u8; ID_LENGTH] = [23u8; ID_LENGTH];
const BYTE_VEC_SUFFIX: &str = "Lucene99HnswVectorsFormat_0";

/// A **byte**-encoded flat vector segment, written through this port's
/// own `vectors::write_flat_vectors`.
///
/// Every `.vec` fixture in this repo is `Float32`, so the whole
/// `VectorEncoding::Byte` arm of `check_vectors` -- its decode loop, its
/// per-vector length validation and its `ordToDoc` walk -- had never run
/// against anything at all.
///
/// `writer_max_doc` and `si_max_doc` are separate on purpose: the
/// `ordToDoc` check compares the `.vec`'s doc mapping against the `.si`'s
/// `maxDoc`, and those are two different files that can disagree.
/// `fnm_dim` is likewise separate from `dim`, because `.fnm` and `.vemf`
/// each record the dimension independently.
#[allow(clippy::too_many_arguments)]
fn write_byte_vectors_fixture(
    dst: &std::path::Path,
    docs: &[i32],
    dim: i32,
    writer_max_doc: i32,
    si_max_doc: i32,
    fnm_dim: i32,
    extra_files: &[String],
    vemf_edit: impl FnOnce(&mut Vec<u8>),
) -> segment_infos::SegmentCommitInfo {
    use lucene_codecs::vectors::{FieldVectorData, FlatVectorsField};

    let values: Vec<u8> = (0..docs.len() as i32)
        .flat_map(|ord| (0..dim).map(move |c| (ord * 16 + c + 1) as u8))
        .collect();
    let (vec, mut vemf) = lucene_codecs::vectors::write_flat_vectors(
        &[FlatVectorsField {
            field_number: 0,
            similarity: field_infos::VectorSimilarityFunction::Euclidean,
            dimension: dim,
            docs: docs.to_vec(),
            values: FieldVectorData::Byte(values),
        }],
        writer_max_doc,
        &BYTE_VEC_SEG_ID,
        BYTE_VEC_SUFFIX,
    )
    .expect("hand-built byte vectors must write cleanly");
    vemf_edit(&mut vemf);

    let field = field_infos::FieldInfo {
        name: "bvec".to_string(),
        number: 0,
        store_term_vectors: false,
        omit_norms: true,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: field_infos::IndexOptions::None,
        doc_values_type: field_infos::DocValuesType::None,
        doc_values_skip_index_type: field_infos::DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: fnm_dim,
        vector_encoding: field_infos::VectorEncoding::Byte,
        vector_similarity_function: field_infos::VectorSimilarityFunction::Euclidean,
    };
    std::fs::write(
        dst.join("_0.fnm"),
        field_infos::write(&[field], &BYTE_VEC_SEG_ID, ""),
    )
    .unwrap();
    std::fs::write(dst.join(format!("_0_{BYTE_VEC_SUFFIX}.vemf")), &vemf).unwrap();
    std::fs::write(dst.join(format!("_0_{BYTE_VEC_SUFFIX}.vec")), &vec).unwrap();

    let si = SegmentInfo {
        id: BYTE_VEC_SEG_ID,
        version: segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        min_version: None,
        doc_count: si_max_doc,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: [
            "_0.si".to_string(),
            "_0.fnm".to_string(),
            format!("_0_{BYTE_VEC_SUFFIX}.vemf"),
            format!("_0_{BYTE_VEC_SUFFIX}.vec"),
        ]
        .into_iter()
        .chain(extra_files.iter().cloned())
        .collect(),
        attributes: vec![],
        index_sort: None,
    };
    std::fs::write(dst.join("_0.si"), segment_info::write(&si, "")).unwrap();

    segment_infos::SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id: BYTE_VEC_SEG_ID,
        codec_name: "Lucene104".to_string(),
        del_gen: -1,
        ..Default::default()
    }
}

/// The `VectorEncoding::Byte` half of `check_vectors`, and the flat
/// (graph-less) shape of `check_hnsw_graphs`.
///
/// Three cases against one writer, differing in exactly one claim each:
///
/// 1. the healthy control -- every check passes, and `check_hnsw_graphs`
///    returns silently because a segment with no `.vem`/`.vex` is the
///    flat exhaustive-search format, not a defect;
/// 2. a `.fnm` whose dimension disagrees with the `.vemf`'s -- the byte
///    branch reports it through a *length* comparison where the float
///    branch reports it as a decode error, an asymmetry that had never
///    been exercised on the byte side;
/// 3. a `.vec` whose ord->doc map points past the `.si`'s `maxDoc` --
///    the flat vector store is addressable as a `DocIdSetIterator` only
///    if that map stays inside `0..maxDoc` and strictly increases, and
///    the two numbers live in two different files.
#[test]
fn a_byte_encoded_vector_field_is_checked_through_the_byte_branch() {
    // (1) Healthy.
    let dst = tempdir();
    let commit = write_byte_vectors_fixture(&dst, &[0, 1, 2], 4, 3, 3, 4, &[], |_| {});
    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    assert!(
        result.all_passed(),
        "a byte-encoded vector segment must pass: {:?}",
        result.failures()
    );
    let decode = result
        .checks
        .iter()
        .find(|c| c.name == "vectors.values_decode:bvec")
        .expect("the byte decode walk must run");
    assert_eq!(decode.message, "ok");
    assert!(result
        .checks
        .iter()
        .any(|c| c.name == "vectors.ord_to_doc:bvec" && c.passed()));
    // A flat, graph-less vector segment produces no `hnsw.*` check at
    // all -- the absence is the format, not a failure.
    assert!(
        !result.checks.iter().any(|c| c.name.starts_with("hnsw.")),
        "{:?}",
        result.checks
    );
    std::fs::remove_dir_all(&dst).ok();

    // (2) `.fnm` says dimension 5, `.vemf` says 4.
    let dst = tempdir();
    let commit = write_byte_vectors_fixture(&dst, &[0, 1, 2], 4, 3, 3, 5, &[], |_| {});
    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let decode = result
        .checks
        .iter()
        .find(|c| c.name == "vectors.values_decode:bvec")
        .expect("the byte decode walk must run");
    assert!(!decode.passed(), "{decode:?}");
    assert!(
        decode
            .message
            .contains("vector has 4 bytes, not the field's dimension 5"),
        "{decode:?}"
    );
    // The disagreement itself is reported separately, and the ord->doc
    // map -- which the dimension does not touch -- still passes.
    assert!(result
        .checks
        .iter()
        .any(|c| c.name == "vectors.field_entry_matches_fnm:bvec" && !c.passed()));
    assert!(result
        .checks
        .iter()
        .any(|c| c.name == "vectors.ord_to_doc:bvec" && c.passed()));
    std::fs::remove_dir_all(&dst).ok();

    // (3) The `.vec` maps ordinal 2 to doc 4; the `.si` says maxDoc=3.
    let dst = tempdir();
    let commit = write_byte_vectors_fixture(&dst, &[0, 1, 4], 4, 5, 3, 4, &[], |_| {});
    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let ord = result
        .checks
        .iter()
        .find(|c| c.name == "vectors.ord_to_doc:bvec")
        .expect("the ord->doc walk must run");
    assert!(!ord.passed(), "{ord:?}");
    assert!(ord.message.contains("ord=2 maps to docID=4"), "{ord:?}");
    // ... and nothing else: an out-of-range doc mapping is not a decode
    // failure and must not be reported as one.
    assert!(result
        .checks
        .iter()
        .any(|c| c.name == "vectors.values_decode:bvec" && c.passed()));
    std::fs::remove_dir_all(&dst).ok();
}

/// A re-signed sweep over the **sparse** byte-vector segment's `.vemf`,
/// the metadata file that says where in the `.vec` every region lives.
///
/// The negative-control shape: overwrite one 8-byte field with a
/// hostile value, re-sign the footer so `file:*`'s CRC cannot claim the
/// catch, and require that the verifier **reports** rather than panics.
/// The `ordToDoc` addresses region is the interesting one -- its offset
/// and length are read off `.vemf` and used to slice the `.vec` with
/// nothing on the wire relating them to it.
#[test]
fn no_re_signed_vemf_field_overwrite_crashes_the_vector_checks() {
    let dst = tempdir();
    let commit = write_byte_vectors_fixture(&dst, &[0, 1, 4], 4, 5, 5, 4, &[], |_| {});
    let vemf_path = dst.join(format!("_0_{BYTE_VEC_SUFFIX}.vemf"));
    let original = std::fs::read(&vemf_path).unwrap();
    let body_end = original.len() - 16;
    let mut total = 0usize;
    let mut caught_by_vectors = 0usize;
    let mut caught_by_other = 0usize;
    let mut accepted = 0usize;
    let mut ord_to_doc_errors = 0usize;
    for off in 0..body_end.saturating_sub(8) {
        for value in [i64::MIN, -1i64, i64::MAX, 1 << 40] {
            total += 1;
            let mut bytes = original.clone();
            bytes[off..off + 8].copy_from_slice(&value.to_be_bytes());
            repair_checksum(&mut bytes);
            std::fs::write(&vemf_path, &bytes).unwrap();
            let dir = FsDirectory::open(&dst);
            let result = check_segment(&dir, &commit);
            let failed: Vec<&str> = result.failures().iter().map(|c| c.name.as_str()).collect();
            let by_vectors = failed
                .iter()
                .any(|n| n.starts_with("vectors.") || n.starts_with("hnsw."));
            if by_vectors {
                caught_by_vectors += 1;
            } else if !failed.is_empty() {
                caught_by_other += 1;
            } else {
                accepted += 1;
            }
            if failed.contains(&"vectors.ord_to_doc:bvec") {
                ord_to_doc_errors += 1;
            }
        }
    }
    std::fs::write(&vemf_path, &original).unwrap();
    std::fs::remove_dir_all(&dst).ok();
    assert_eq!(caught_by_vectors + caught_by_other + accepted, total);
    // Measured when this was written: 566 of 616 caught by the vector
    // checks, **0 by any other check** -- nothing else in the module
    // reads the `.vemf`, so a corruption these miss is one nothing
    // catches. The 50 it accepts are fields whose overwritten value
    // still describes a well-formed region (a similarity ordinal, a
    // block shift that still decodes) -- the `.tip` phenomenon, and the
    // reason `Lucene99FlatVectorsReader` verifies the whole file at
    // open. The floor is ~85% of the measured rate.
    assert!(
        caught_by_vectors >= 481,
        "only {caught_by_vectors} of {total} hostile .vemf field overwrites were rejected \
             by a vector check (was 566); {caught_by_other} elsewhere, {accepted} accepted"
    );
    assert_eq!(
        caught_by_other, 0,
        "the `.vemf` now has a second reader; re-measure the table in c30"
    );
    assert!(
        ord_to_doc_errors > 0,
        "no .vemf overwrite reached the ordToDoc walk's error arm"
    );
}

/// A `.vem` that parses cleanly but whose *per-field* graph does not
/// open: `hnsw.open:<field>`, and the three families it takes down.
///
/// The whole-file `hnsw.open` arm has its own test; this is the per-field
/// one, the one with a `skip_families` call attached, proving that a field whose graph is
/// unreadable *names* the checks that did not run on it. A segment can
/// carry several vector fields and lose the graph of exactly one.
///
/// The corruption is a single byte and it is the honest minimum: a field
/// written with **no** graph records `vectorIndexLength = 0` (which the
/// reader correctly reads as "flat field, no graph"), and flipping that
/// vlong to `1` makes the entry claim graph bytes while still recording
/// `numLevels = 0`, i.e. no node offsets. `OffHeapHnswGraph::new` rejects
/// exactly that pair. Everything else in the file, the `.vec`/`.vemf`
/// included, is real writer output.
#[test]
fn a_vem_whose_per_field_graph_cannot_open_names_what_it_takes_down() {
    use lucene_codecs::hnsw::OnHeapHnswGraph;
    use lucene_codecs::hnsw_vectors::{write_hnsw_vectors, HnswVectorsField};

    let dst = tempdir();
    let (vex, mut vem) = write_hnsw_vectors(
        &[HnswVectorsField {
            field_number: 0,
            encoding: field_infos::VectorEncoding::Byte,
            similarity: field_infos::VectorSimilarityFunction::Euclidean,
            dimension: 4,
            count: 3,
            graph: None::<&OnHeapHnswGraph>,
            m: 16,
        }],
        &BYTE_VEC_SEG_ID,
        BYTE_VEC_SUFFIX,
    )
    .expect("a graph-less .vem must write cleanly");

    // The entry's tail is `[vlong vectorIndexLength][vint dimension]
    // [i32 size][vint M][vint numLevels]`, then the `-1` field
    // terminator and the footer. With dimension=4 and M=16 every vint
    // here is one byte, so the length's vlong sits eight bytes before
    // the terminator -- asserted, not assumed.
    let terminator = vem.len() - lucene_store::codec_util::FOOTER_LENGTH - 4;
    let len_pos = terminator - 8;
    assert_eq!(
        vem[len_pos], 0,
        "expected a zero vectorIndexLength vlong at {len_pos}: {vem:?}"
    );
    vem[len_pos] = 1;
    repair_checksum(&mut vem);

    std::fs::write(dst.join(format!("_0_{BYTE_VEC_SUFFIX}.vex")), &vex).unwrap();
    std::fs::write(dst.join(format!("_0_{BYTE_VEC_SUFFIX}.vem")), &vem).unwrap();
    let commit = write_byte_vectors_fixture(
        &dst,
        &[0, 1, 2],
        4,
        3,
        3,
        4,
        &[
            format!("_0_{BYTE_VEC_SUFFIX}.vem"),
            format!("_0_{BYTE_VEC_SUFFIX}.vex"),
        ],
        |_| {},
    );

    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let open = result
        .checks
        .iter()
        .find(|c| c.name == "hnsw.open:bvec")
        .unwrap_or_else(|| panic!("hnsw.open:bvec must be reported: {:?}", result.checks));
    assert!(!open.passed(), "{open:?}");
    assert!(
        open.message.contains("graph has data but no node offsets"),
        "{open:?}"
    );
    // The three graph families are named as *not run*, per field.
    let skipped: Vec<&str> = result.skipped().iter().map(|c| c.name.as_str()).collect();
    for family in HNSW_FAMILIES {
        assert!(
            skipped.contains(&format!("{family}:bvec").as_str()),
            "{family}:bvec was not named as skipped: {skipped:?}"
        );
    }
    // The *flat* vector store is untouched and still fully checked --
    // the skip is targeted at the graph, not a blanket give-up.
    assert!(result
        .checks
        .iter()
        .any(|c| c.name == "vectors.values_decode:bvec" && c.passed()));
    assert!(result
        .checks
        .iter()
        .any(|c| c.name == "vectors.ord_to_doc:bvec" && c.passed()));
    std::fs::remove_dir_all(&dst).ok();
}

/// The `.liv`-aware arms of the term-vector and norms walks, plus the
/// `FixedBitSet` bound rule they now go through.
///
/// Three separate places consult `live_docs` while walking a doc id that
/// came from a *different* file -- `check_term_vectors`' per-document
/// loop, `check_postings`' per-posting loop and `check_field_norms`'
/// terms-vs-norms cross-check -- and only the postings one had ever been
/// reached, because no fixture in the tree carried both deletions and
/// term vectors or norms. That is exactly the combination a real index
/// has after its first delete.
#[test]
fn a_segment_with_deletions_still_checks_its_vectors_and_norms() {
    use lucene_util::fixed_bit_set::FixedBitSet;

    let dst = copy_fixture("term_vectors_index");
    let mut commit = {
        let dir = FsDirectory::open(&dst);
        read_commit(&dir)
    };
    let max_doc = {
        let dir = FsDirectory::open(&dst);
        let si = open_si(&dir, &commit).expect("the fixture's .si parses");
        si.doc_count
    };
    assert!(max_doc > 1, "the fixture needs more than one document");
    let mut live = FixedBitSet::new(max_doc as usize);
    for doc in 1..max_doc as usize {
        live.set(doc);
    }
    let liv = live_docs::write(&live, &commit.segment_id, 1, 1).unwrap();
    std::fs::write(dst.join(liv_file_name("_0", 1)), &liv).unwrap();
    commit.del_gen = 1;
    commit.del_count = 1;

    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    assert!(
        result.all_passed(),
        "deleting document 0 must not make a healthy segment fail: {:?}",
        result.failures()
    );
    // The three walks really ran, and really saw the deletion: the
    // term-vector and postings statistics are live-only, so they must
    // now be *smaller* than on the undeleted control.
    let deleted_stats = result.stats;
    let mut undeleted = commit.clone();
    undeleted.del_gen = -1;
    undeleted.del_count = 0;
    let control = check_segment(&dir, &undeleted);
    assert!(control.all_passed(), "{:?}", control.failures());
    assert!(
        deleted_stats.term_doc_pairs < control.stats.term_doc_pairs,
        "the postings walk did not skip the deleted document: {deleted_stats:?} vs {:?}",
        control.stats
    );
    assert!(
        deleted_stats.term_vector_fields < control.stats.term_vector_fields,
        "the term-vector walk did not skip the deleted document: {deleted_stats:?} vs {:?}",
        control.stats
    );
    std::fs::remove_dir_all(&dst).ok();
}

/// The bound rule itself, at the two helpers that implement it: a doc id
/// past the end of the `.liv` bitset must read as **live**, not index the
/// bitset.
///
/// `FixedBitSet::get` does `words[index >> 6]` behind a bare
/// `debug_assert`, so an unbounded id 64 or more past the end panics in a
/// release build too (the crate rule in `docs/arithmetic-gate.md`). Both
/// modes are covered: an *empty* bitset (`words` is empty, so any index is
/// a real panic) and a short one (`words` has room, so the read would
/// silently return a ghost bit).
#[test]
fn a_doc_id_past_the_end_of_the_live_docs_bitset_is_treated_as_live() {
    use lucene_util::fixed_bit_set::FixedBitSet;

    let empty = FixedBitSet::new(0);
    assert!(is_live(Some(&empty), 0));
    assert!(is_live(Some(&empty), 1_000_000));
    assert!(is_live_at(Some(&empty), usize::MAX));

    // Two bits, one live and one deleted, then the ghost-bit range.
    let mut short = FixedBitSet::new(2);
    short.set(1);
    assert!(!is_live(Some(&short), 0));
    assert!(is_live(Some(&short), 1));
    assert!(is_live(Some(&short), 2), "a ghost bit must not be read");
    assert!(is_live(Some(&short), 63), "a ghost bit must not be read");
    assert!(
        is_live(Some(&short), 64),
        "an index past `words` must not panic"
    );

    // A negative doc id -- `as usize` would sign-extend it to
    // `usize::MAX`.
    assert!(is_live(Some(&short), -1));
    // And no bitset at all means nothing is deleted.
    assert!(is_live(None, i32::MAX));
}

/// A re-signed `.kdd` corruption that gets past `points::open` and lands
/// in `decode_leaves`: `points.decode:<field>`, and the four per-field
/// families it takes down.
///
/// The tree's *metadata* is fine and its **leaf blocks** are not.
/// A missing field entry alone is legitimate after all its points were
/// deleted; this mutation instead exercises an existing tree. Every check below
/// it -- the value/bounds check, the doc-count check, the point-count
/// check -- reads the decoded leaves, so all four have to be named as
/// not run rather than silently omitted.
///
/// Reported as a rate: the footer is re-signed over each
/// corruption so `file:*`'s CRC cannot claim the catch.
#[test]
fn a_re_signed_kdd_corruption_is_reported_by_the_points_decode() {
    let points: Vec<(i32, Vec<u8>)> = (0..64)
        .map(|d| (d, (d as u64).to_be_bytes().to_vec()))
        .collect();
    let dst = tempdir();
    let clean = write_points_fixture(&dst, &points, 64, None);
    let original = std::fs::read(dst.join("_0.kdd")).unwrap();
    {
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &clean);
        assert!(result.all_passed(), "{:?}", result.failures());
    }

    let body_start = 32;
    let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
    let mut total = 0usize;
    let mut caught_by_decode = 0usize;
    let mut caught_by_other = 0usize;
    let mut accepted = 0usize;
    let mut named_skips = 0usize;
    for off in (body_start..body_end).step_by(3) {
        total += 1;
        let mut bytes = original.clone();
        bytes[off] ^= 0xff;
        repair_checksum(&mut bytes);
        let commit = write_points_fixture(&dst, &points, 64, Some(&bytes));
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        let failed: Vec<&str> = result.failures().iter().map(|c| c.name.as_str()).collect();
        if failed.contains(&"points.decode:loc") {
            caught_by_decode += 1;
            // Every per-field family below the decode must be named.
            if POINTS_PER_FIELD
                .iter()
                .all(|f| failed.contains(&format!("{f}:loc").as_str()))
            {
                named_skips += 1;
            }
        } else if !failed.is_empty() {
            caught_by_other += 1;
        } else {
            accepted += 1;
        }
    }
    std::fs::remove_dir_all(&dst).ok();
    assert_eq!(caught_by_decode + caught_by_other + accepted, total);
    // Measured when this was written: **200 of 200 rejected**, of which
    // only **2** by the decode walk and 198 by another points check.
    // That is the opposite of the `.fdt`/`.dvd`/`.nvm` rows, and it is
    // the finding: a BKD tree carries three independent redundancies
    // over the same bytes (the field's declared min/max packed value,
    // its `docCount` and its `pointCount`), so almost every flipped byte
    // is caught by a *cross-check* rather than by failing to decode. The
    // decode arm is still the only one that can report a leaf that does
    // not parse at all, which is why it needs its own driver.
    assert!(
        caught_by_decode >= 2,
        "only {caught_by_decode} of {total} re-signed .kdd corruptions reached \
             points.decode (was 2); {caught_by_other} elsewhere, {accepted} accepted"
    );
    assert_eq!(
        accepted, 0,
        "a re-signed .kdd corruption was silently accepted"
    );
    assert_eq!(
        named_skips, caught_by_decode,
        "a failed points.decode did not name every family it takes down"
    );
}

/// The two points arms no other fixture in the tree reaches.
///
/// 1. **`points.leaf_bounds_subset_of_field`** is skipped outright for a
///    single-index-dimension field (Lucene's `VerifyPointsVisitor` gets
///    the same bound from the packed values in that case), and every
///    points fixture here was one-dimensional. It matters for a
///    multi-dimensional field because a `PointRangeQuery` prunes whole
///    subtrees on the *leaf's own* bounding box without ever reading a
///    point: a leaf box outside the field box silently drops matches.
/// 2. **`docCount > pointCount`** -- a field claiming more documents than
///    it has points at all. `docCount` is what `IndexSearcher`'s cost
///    estimate reads, and the writer computes it from the points, so only
///    a hand-edited `.kdm` can say otherwise.
#[test]
fn a_multi_dimension_points_field_checks_its_leaf_boxes_and_its_doc_count() {
    // Two index dimensions, one byte each, and enough points to force
    // more than one leaf (so the leaves carry their own bounding boxes).
    let real: Vec<(i32, Vec<u8>)> = (0..1024i32)
        .map(|d| (d, vec![(d / 32) as u8, (d % 32) as u8]))
        .collect();
    // The same doc ids and the same leaf layout, but every value shifted
    // into a range the `.kdm`'s declared box does not contain.
    let wider: Vec<(i32, Vec<u8>)> = real
        .iter()
        .map(|(d, v)| (*d, vec![v[0], v[1].wrapping_add(200)]))
        .collect();

    // Control: a genuinely consistent two-dimensional field passes, and
    // the leaf-bounds check really runs on it (it does not exist at all
    // for a one-dimensional field).
    let dst = tempdir();
    let commit = write_points_fixture_dims(&dst, &real, 1024, 2, 1, None, &|_| {});
    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    assert!(result.all_passed(), "{:?}", result.failures());
    assert!(
        result
            .checks
            .iter()
            .any(|c| c.name == "points.leaf_bounds_subset_of_field:loc"),
        "the leaf-bounds check must run for a 2-dimension field: {:?}",
        result.checks
    );
    std::fs::remove_dir_all(&dst).ok();

    // (1) `.kdm`/`.kdi` from `real`, `.kdd` from `wider`.
    let dst = tempdir();
    let (_, _, wider_kdd) = points::write(
        &[lucene_codecs::points::WritePointsField {
            field_number: 5,
            num_dims: 2,
            num_index_dims: 2,
            bytes_per_dim: 1,
            points: wider,
        }],
        points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
        &POINTS_SEG_ID,
        "",
    )
    .expect("the wider point set must write cleanly");
    let commit = write_points_fixture_dims(&dst, &real, 1024, 2, 1, Some(&wider_kdd), &|_| {});
    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let bounds = result
        .checks
        .iter()
        .find(|c| c.name == "points.leaf_bounds_subset_of_field:loc")
        .expect("the leaf-bounds check must run");
    assert!(!bounds.passed(), "{bounds:?}");
    assert!(
        bounds.message.contains("bounding box for dim 1"),
        "{bounds:?}"
    );
    std::fs::remove_dir_all(&dst).ok();

    // (2) A `.kdm` claiming one more document than it has points.
    let few: Vec<(i32, Vec<u8>)> = (0..5i32).map(|d| (d, vec![d as u8, 0])).collect();
    let dst = tempdir();
    let commit = write_points_fixture_dims(&dst, &few, 8, 2, 1, None, &|kdm| {
        patch_kdm_doc_count(kdm, 6)
    });
    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let docs = result
        .checks
        .iter()
        .find(|c| c.name == "points.doc_count_matches:loc")
        .expect("the doc-count check must run");
    assert!(!docs.passed(), "{docs:?}");
    assert!(
        docs.message
            .contains("claims docCount=6 with only point_count=5"),
        "{docs:?}"
    );
    std::fs::remove_dir_all(&dst).ok();
}

/// Rewrites the per-field `docCount` vint in a `.kdm` and re-signs the
/// footer, so only the semantic invariant can fire.
///
/// The per-field record ends
/// `[vlong pointCount][vint docCount][vint numIndexBytes][i64
/// minLeafBlockFP][i64 indexStartPointer]`, and the file then ends with
/// the `-1` field terminator, the `.kdi`/`.kdd` total lengths and the
/// codec footer -- so the two vints are found by walking *backwards* from
/// a fixed offset, a vint's last byte being the one with its high bit
/// clear.
fn patch_kdm_doc_count(kdm: &mut [u8], doc_count: u8) {
    assert!(doc_count < 0x80, "single-byte vints only");
    // footer, `.kdd` length, `.kdi` length, `-1`, indexStartPointer,
    // minLeafBlockFP.
    let after_vints = kdm.len() - lucene_store::codec_util::FOOTER_LENGTH - 8 - 8 - 4 - 8 - 8;
    let mut num_index_bytes_start = after_vints - 1;
    assert_eq!(kdm[num_index_bytes_start] & 0x80, 0);
    while num_index_bytes_start > 0 && kdm[num_index_bytes_start - 1] & 0x80 != 0 {
        num_index_bytes_start -= 1;
    }
    let doc_count_pos = num_index_bytes_start - 1;
    assert_eq!(
        kdm[doc_count_pos] & 0x80,
        0,
        "docCount must be a single-byte vint"
    );
    kdm[doc_count_pos] = doc_count;
    repair_checksum(kdm);
}

/// A re-signed `.dvm` sweep: the metadata a doc-values column is decoded
/// *through*, corrupted one byte at a time with the footer re-signed so
/// only a semantic check can fire.
///
/// A sweep rather than a typed `.dvm` editor. A sweep is the right shape here is that the
/// arms it has to reach are not one invariant but a family: a per-document
/// decode that fails, an ordinal outside the terms dictionary, a
/// non-increasing SORTED_SET ordinal run, a dictionary whose size
/// disagrees with `valueCount`. Every one of them needs a `.dvm` that
/// *parses* and then disagrees with its `.dvd`, and a byte flip in the
/// entry region produces exactly that -- it moves an offset, a count or a
/// bit width without touching the payload those describe.
///
/// No check other than `doc_values.*` catches a `.dvd` corruption; the same
/// holds for the `.dvm` and is asserted, so
/// a skipped `doc_values` family really is the column's only reader.
#[test]
fn no_re_signed_dvm_corruption_of_a_doc_values_index_goes_unnoticed() {
    // (fixture, the rejection floor -- ~85% of what was measured when
    // this was written: 315/520, 225/380, 91/218, with **0** caught by
    // any non-`doc_values` check in all three.)
    for (fixture, floor) in [
        ("doc_values_index", 267usize),
        ("multi_valued_dv_index", 191),
        ("sorted_dv_index", 77),
    ] {
        let dst = copy_fixture(fixture);
        let dvm_path = find_file(&dst, ".dvm");
        let original = std::fs::read(&dvm_path).unwrap();
        let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
        let mut total = 0usize;
        let mut caught_by_doc_values = 0usize;
        let mut caught_by_other = 0usize;
        let mut accepted = 0usize;
        for off in (48..body_end).step_by(2) {
            for mask in [0x01u8, 0x80] {
                total += 1;
                let mut bytes = original.clone();
                bytes[off] ^= mask;
                repair_checksum(&mut bytes);
                std::fs::write(&dvm_path, &bytes).unwrap();
                let dir = FsDirectory::open(&dst);
                // A corruption the *commit* cannot survive still counts
                // as rejected, so `total` stays equal to the sweep's own
                // iteration count.
                let failed: Vec<String> = match check_directory(&dir) {
                    Ok(results) => failed_names(&results),
                    Err(e) => vec![format!("commit.unreadable: {e}")],
                };
                if failed.iter().any(|n| n.starts_with("doc_values.")) {
                    caught_by_doc_values += 1;
                } else if !failed.is_empty() {
                    caught_by_other += 1;
                } else {
                    accepted += 1;
                }
            }
        }
        std::fs::write(&dvm_path, &original).unwrap();
        let dir = FsDirectory::open(&dst);
        assert!(check_directory(&dir)
            .unwrap()
            .iter()
            .all(|r| r.all_passed()));
        std::fs::remove_dir_all(&dst).ok();
        assert_eq!(caught_by_doc_values + caught_by_other + accepted, total);
        assert!(
            caught_by_doc_values >= floor,
            "{fixture}: only {caught_by_doc_values} of {total} re-signed .dvm corruptions \
                 were caught by a doc_values check; {caught_by_other} elsewhere, \
                 {accepted} accepted"
        );
        assert_eq!(
            caught_by_other, 0,
            "{fixture}: the .dvm now has a second reader; re-measure the table in c30"
        );
    }
}

/// The error and skip paths of the two checks that read *per-document
/// sort keys* -- `sort.docs_in_index_sort_order` and
/// `soft_deletes.count_matches`. Every failure arm of both was unfired:
/// the only segments that reached them were healthy ones.
///
/// 1. A `SortedSetSortField` sort is verified, not skipped: the column
///    is reduced by its `SortedSetSelector` and compared by ordinal, so
///    a segment in order passes and a declared descending order over the
///    same ascending column fails.
/// 2. A sort whose *kind* disagrees with the field's doc-values type --
///    a numeric `SortField` over a SORTED_SET column, which is what
///    `DocValues.getNumeric` throws on in Java. A `.si` and a `.fnm`
///    disagreeing, and the consequence is the sharp one: the declared
///    order that every merge and every early-terminating query trusts
///    would be verified by nothing.
/// 3. A `.dvd` the `.si` lists and the directory does not have -- the
///    `dir.open` arm inside `open_base_column`, distinct from the
///    `doc_values.open` one tested elsewhere because this is the
///    *second*, independent open of the same pair.
/// 4. The same for the soft-deletes field, whose `softDelCount` is then a
///    claim about data that cannot be read.
#[test]
fn a_sort_or_soft_deletes_field_whose_values_cannot_be_read_is_reported() {
    let sorted_set_sort_on_tags = || {
        Some(vec![segment_info::IndexSortField {
            field: "tags".to_string(),
            reverse: false,
            kind: segment_info::IndexSortKind::SortedSet {
                selector: segment_info::SortedSetSelector::Min,
                missing: segment_info::StringMissingValue::None,
            },
        }])
    };
    let numeric_sort_on_tags = || {
        Some(vec![segment_info::IndexSortField::long(
            "tags",
            false,
            Some(i64::MAX),
        )])
    };
    let sort_check = |dir: &FsDirectory, commit: &segment_infos::SegmentCommitInfo| {
        check_segment(dir, commit)
            .checks
            .into_iter()
            .find(|c| c.name == "sort.docs_in_index_sort_order")
            .expect("the sort check must run")
    };
    let restamp_sort = |dst: &std::path::Path,
                        commit: &segment_infos::SegmentCommitInfo,
                        sort: Option<Vec<segment_info::IndexSortField>>| {
        let si_path = dst.join("_0.si");
        let mut si = segment_info::parse(&std::fs::read(&si_path).unwrap(), &commit.segment_id)
            .expect("the fixture's .si parses");
        si.index_sort = sort;
        std::fs::write(&si_path, segment_info::write(&si, "")).unwrap();
    };

    // (1) A SORTED_SET sort is verified: in order passes, and the same
    // column declared descending fails.
    let dst = tempdir();
    let commit = write_single_valued_sorted_set_fixture(&dst, &[b"a", b"b"]);
    restamp_sort(&dst, &commit, sorted_set_sort_on_tags());
    let dir = FsDirectory::open(&dst);
    let sort = sort_check(&dir, &commit);
    assert!(sort.passed(), "{sort:?}");
    let mut descending = sorted_set_sort_on_tags();
    descending.as_mut().unwrap()[0].reverse = true;
    restamp_sort(&dst, &commit, descending);
    let dir = FsDirectory::open(&dst);
    let sort = sort_check(&dir, &commit);
    assert!(!sort.passed() && !sort.was_skipped(), "{sort:?}");
    assert!(sort.message.contains("sorts after"), "{sort:?}");

    // (2) A numeric sort over the same SORTED_SET column: a real
    // `.si`/`.fnm` disagreement, and a failure.
    restamp_sort(&dst, &commit, numeric_sort_on_tags());
    let dir = FsDirectory::open(&dst);
    let sort = sort_check(&dir, &commit);
    assert!(!sort.passed() && !sort.was_skipped(), "{sort:?}");
    assert!(
        sort.message.contains("doc-values type SortedSet"),
        "{sort:?}"
    );

    // (3) ... and with the `.dvd` gone, the same check reports the open
    // failure instead. (Same directory, one file removed.)
    std::fs::remove_file(dst.join("_0.dvd")).unwrap();
    let dir = FsDirectory::open(&dst);
    let sort = sort_check(&dir, &commit);
    assert!(!sort.passed(), "{sort:?}");
    assert!(
        !sort.message.contains("doc-values type"),
        "the missing file must be reported, not the field type: {sort:?}"
    );
    std::fs::remove_dir_all(&dst).ok();

    // (2) A SORTED_SET soft-deletes field.
    let dst = tempdir();
    let mut commit = write_single_valued_sorted_set_fixture(&dst, &[b"a", b"b"]);
    commit.soft_del_count = 2;
    let fi = field_infos::FieldInfo {
        name: "tags".to_string(),
        number: 0,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: true,
        parent_field: false,
        index_options: field_infos::IndexOptions::None,
        doc_values_type: field_infos::DocValuesType::SortedSet,
        doc_values_skip_index_type: field_infos::DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: field_infos::VectorEncoding::Float32,
        vector_similarity_function: field_infos::VectorSimilarityFunction::Euclidean,
    };
    std::fs::write(
        dst.join("_0.fnm"),
        field_infos::write(std::slice::from_ref(&fi), &commit.segment_id, ""),
    )
    .unwrap();
    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let soft = result
        .checks
        .iter()
        .find(|c| c.name == "soft_deletes.count_matches")
        .expect("the soft-deletes check must run");
    assert!(!soft.passed(), "{soft:?}");
    assert!(
        soft.message
            .contains("soft-deletes field \"tags\": field has doc-values type SortedSet"),
        "{soft:?}"
    );
    std::fs::remove_dir_all(&dst).ok();
}

/// A re-signed `.dvm` sweep over the **hand-built single-valued
/// SORTED_SET** fixture -- the second SORTED_SET code path
/// (`Lucene90DocValuesConsumer.addSortedSetField` collapses a
/// one-value-per-doc set to a plain `SortedEntry`), whose ordinal
/// bookkeeping is separate code from the multi-valued branch and which
/// no Java-written fixture in this repo produces.
///
/// The arms this reaches are the ordinal-space ones: an ordinal outside
/// the terms dictionary, a document whose ordinal will not decode, and a
/// dictionary whose decoded size disagrees with `valueCount`. Those are
/// the checks that catch a dictionary nothing can ever match -- a
/// `SortedDocValues` range query or an index sort reads the ordinal
/// space without ever looking at the terms.
#[test]
fn no_re_signed_dvm_corruption_of_a_single_valued_sorted_set_goes_unnoticed() {
    let dst = tempdir();
    let commit = write_single_valued_sorted_set_fixture(
        &dst,
        &[b"alpha", b"beta", b"delta", b"gamma", b"omega"],
    );
    let dvm_path = dst.join("_0.dvm");
    let original = std::fs::read(&dvm_path).unwrap();
    {
        let dir = FsDirectory::open(&dst);
        let result = check_segment(&dir, &commit);
        assert!(result.all_passed(), "{:?}", result.failures());
    }

    let body_end = original.len() - lucene_store::codec_util::FOOTER_LENGTH;
    let mut total = 0usize;
    let mut caught_by_doc_values = 0usize;
    let mut caught_by_other = 0usize;
    let mut accepted = 0usize;
    let mut families: std::collections::BTreeSet<String> = Default::default();
    for off in 48..body_end {
        for mask in [0x01u8, 0x80, 0xff] {
            total += 1;
            let mut bytes = original.clone();
            bytes[off] ^= mask;
            repair_checksum(&mut bytes);
            std::fs::write(&dvm_path, &bytes).unwrap();
            let dir = FsDirectory::open(&dst);
            let result = check_segment(&dir, &commit);
            let failed: Vec<&str> = result.failures().iter().map(|c| c.name.as_str()).collect();
            if failed.iter().any(|n| n.starts_with("doc_values.")) {
                caught_by_doc_values += 1;
                families.extend(
                    failed
                        .iter()
                        .filter(|n| n.starts_with("doc_values."))
                        .map(|n| (*n).to_string()),
                );
            } else if !failed.is_empty() {
                caught_by_other += 1;
            } else {
                accepted += 1;
            }
        }
    }
    std::fs::write(&dvm_path, &original).unwrap();
    std::fs::remove_dir_all(&dst).ok();
    assert_eq!(caught_by_doc_values + caught_by_other + accepted, total);
    // Measured when this was written: 256 of 624 caught by a doc-values
    // check, **0** by anything else. The 368 it accepts are the `.dvm`'s
    // padding and its many bit-width/offset fields whose flipped value
    // still describes a self-consistent column -- pure metadata with no
    // second copy, which is why `Lucene90DocValuesProducer` verifies the
    // whole file at open and why `file:*` does a full CRC. Floor at
    // ~85% of the measured rate.
    assert!(
        caught_by_doc_values >= 217,
        "only {caught_by_doc_values} of {total} re-signed .dvm corruptions were caught by \
             a doc_values check (was 256); {caught_by_other} elsewhere, {accepted} accepted"
    );
    assert_eq!(caught_by_other, 0);
    // The sweep must reach the *ordinal-space* checks, not just the
    // per-document decode -- those are the ones with no second reader.
    for family in [
        "doc_values.values_decode:tags",
        "doc_values.ords_dense:tags",
        "doc_values.terms_sorted:tags",
    ] {
        assert!(
            families.contains(family),
            "{family} was never reached: {families:?}"
        );
    }
}

/// The remaining failure arms of the two sort-key readers: a `.liv` the
/// soft-deletes count needs and cannot open or cannot parse, a `.dvd` the
/// `.si` does not list at all, and a `.dvd` whose per-document numeric
/// values will not decode.
///
/// `check_soft_deletes` opens the `.liv` a **second** time, independently
/// of `check_live_docs`, because the count it computes is over *live*
/// documents -- so its own open and parse failures are separate arms, and
/// a soft-delete count computed as if nothing were deleted is exactly the
/// wrong answer (it is the number `IndexWriter` uses to decide when a
/// segment is fully deleted and can be dropped).
#[test]
fn the_sort_key_readers_report_every_way_their_input_can_fail() {
    // (1) A soft-deletes segment whose `.liv` is not there.
    let dst = tempdir();
    let mut commit = write_sorted_dv_fixture(&dst, &[1, 2, 3, 4], sort_asc(), true, 3);
    commit.del_gen = 1;
    commit.del_count = 1;
    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let soft = result
        .checks
        .iter()
        .find(|c| c.name == "soft_deletes.count_matches")
        .expect("the soft-deletes check must run");
    assert!(!soft.passed(), "{soft:?}");

    // (2) ... and with a `.liv` that is present but will not parse.
    std::fs::write(dst.join(liv_file_name("_0", 1)), b"not a .liv file").unwrap();
    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let soft = result
        .checks
        .iter()
        .find(|c| c.name == "soft_deletes.count_matches")
        .expect("the soft-deletes check must run");
    assert!(!soft.passed(), "{soft:?}");
    std::fs::remove_dir_all(&dst).ok();

    // (3) A `.si` that declares an index sort and lists a `.dvm` but no
    // `.dvd`: nothing can be read, so the sort is *unverified*, here
    // through a half-listed file set rather than no doc-values files.
    let dst = tempdir();
    let commit = write_sorted_dv_fixture(&dst, &[1, 2, 3, 4], sort_asc(), false, 0);
    let si_path = dst.join("_0.si");
    let mut si = segment_info::parse(&std::fs::read(&si_path).unwrap(), &commit.segment_id)
        .expect("the fixture's .si parses");
    si.files.retain(|f| !f.ends_with(".dvd"));
    std::fs::write(&si_path, segment_info::write(&si, "")).unwrap();
    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let sort = result
        .checks
        .iter()
        .find(|c| c.name == "sort.docs_in_index_sort_order")
        .expect("the sort check must be reported, not omitted");
    assert!(sort.was_skipped(), "{sort:?}");
    std::fs::remove_dir_all(&dst).ok();

    // (4) A `.dvd` that is *shorter* than the `.dvm` says its column is.
    // `open_base_column` parses only the `.dvm`, so the pair opens and the
    // failure lands in `sort_key_values`' per-document read -- the arm
    // that turns "the sort keys cannot be read" into a reported check
    // rather than a silent pass over an unverified order.
    let dst = tempdir();
    let commit = write_sorted_dv_fixture(&dst, &[1, 2, 3, 4], sort_asc(), false, 0);
    let dvd_path = dst.join("_0.dvd");
    let original = std::fs::read(&dvd_path).unwrap();
    std::fs::write(&dvd_path, &original[..original.len() / 2]).unwrap();
    let dir = FsDirectory::open(&dst);
    let result = check_segment(&dir, &commit);
    let sort = result
        .checks
        .iter()
        .find(|c| c.name == "sort.docs_in_index_sort_order")
        .expect("the sort check must run");
    assert!(!sort.passed(), "{sort:?}");
    assert!(
        sort.message.contains("sort field \"ts\""),
        "the failure must name the sort field: {sort:?}"
    );
    std::fs::remove_dir_all(&dst).ok();
}

fn writer_version() -> segment_info::LuceneVersion {
    segment_info::LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    }
}

/// NUMERIC doc-values fields registered up front, as an index sort
/// needs its fields to be.
fn numeric_dv_fields(names: &[&str]) -> Vec<field_infos::FieldInfo> {
    names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            field_infos::FieldInfo::new(*name, i as i32)
                .with_omit_norms(true)
                .with_doc_values(
                    field_infos::DocValuesType::Numeric,
                    field_infos::DocValuesSkipIndexType::None,
                    -1,
                )
        })
        .collect()
}

/// Two NUMERIC fields in one `Lucene90` doc-values instance, only one of
/// them with a skip index: the column `check_doc_values` opens for the
/// first field is shared by the second, so whether `.dvs` is opened must
/// not depend on which of the two happens to be read first.
#[test]
fn a_skip_index_field_sharing_a_column_with_a_plain_one_reads_its_dvs() {
    use crate::document::{Document, NumericDocValuesField};
    use crate::index_writer::IndexWriter;
    for plain_first in [true, false] {
        let dst = tempdir();
        let dir = FsDirectory::open(&dst);
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", writer_version()).unwrap();
        for i in 0..100i64 {
            let mut d = Document::new();
            if plain_first {
                d.add(NumericDocValuesField::new("a_plain", i));
                d.add(NumericDocValuesField::indexed_field("b_skip", i * 3));
            } else {
                d.add(NumericDocValuesField::indexed_field("a_skip", i * 3));
                d.add(NumericDocValuesField::new("b_plain", i));
            }
            w.add_fields_document(&d).unwrap();
        }
        w.commit().unwrap();
        drop(w);
        let results = check_directory(&dir).unwrap();
        let skip = if plain_first { "b_skip" } else { "a_skip" };
        let check = results[1]
            .checks
            .iter()
            .find(|c| c.name == format!("doc_values.skipper:{skip}"))
            .expect("the skipper check ran");
        assert!(check.passed(), "plain_first={plain_first}: {check:?}");
        assert!(
            results.iter().all(CheckResult::all_passed),
            "plain_first={plain_first}"
        );
    }
}

/// An index sort whose field lives in the *second*
/// `PerFieldDocValuesFormat` instance (`Lucene90_1`) is read from that
/// instance's column, not from the first `.dvm` the `.si` lists.
#[test]
fn an_index_sort_field_in_a_second_doc_values_instance_is_verified() {
    use crate::document::{Document, NumericDocValuesField};
    use crate::index_writer::IndexWriter;
    let dst = tempdir();
    let dir = FsDirectory::open(&dst);
    let mut w = IndexWriter::open(
        &dir,
        numeric_dv_fields(&["other", "ts"]),
        "Lucene104",
        writer_version(),
    )
    .unwrap();
    for f in ["other", "ts"] {
        w.add_doc_values_field(f).unwrap();
    }
    w.set_doc_values_format_for_field(
        "ts",
        lucene_codecs::per_field_doc_values::Lucene90DocValuesFormat::new(16).unwrap(),
    );
    w.set_index_sort(Some(&[segment_info::IndexSortField::long(
        "ts", false, None,
    )]))
    .unwrap();
    for i in 0..50i64 {
        let mut d = Document::new();
        d.add(NumericDocValuesField::new("other", i));
        d.add(NumericDocValuesField::new("ts", (i * 7919) % 101));
        w.add_fields_document(&d).unwrap();
    }
    w.commit().unwrap();
    drop(w);
    let results = check_directory(&dir).unwrap();
    let files = dir.list_all().unwrap();
    assert!(
        files.iter().filter(|f| f.ends_with(".dvm")).count() >= 2,
        "two doc-values instances: {files:?}"
    );
    let sort = results[1]
        .checks
        .iter()
        .find(|c| c.name == "sort.docs_in_index_sort_order")
        .expect("the sort check ran");
    assert!(sort.passed(), "{sort:?}");
    assert!(results.iter().all(CheckResult::all_passed));
}

/// `VerifyPointsVisitor`: a real multi-leaf 1-D and 2-D index passes the
/// cell and ordering checks, and the visitor itself rejects an inverted
/// cell, a point outside its cell and an out-of-order 1-D sweep.
#[test]
fn verify_points_visitor_passes_real_trees_and_rejects_bad_cells_and_order() {
    use crate::document::{Document, IntPoint};
    use crate::index_writer::IndexWriter;
    use points::IntersectVisitor;
    let dst = tempdir();
    let dir = FsDirectory::open(&dst);
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", writer_version()).unwrap();
    for i in 0..3000i32 {
        let mut d = Document::new();
        d.add(IntPoint::new("one", &[(i * 7919) % 1000]).unwrap());
        d.add(IntPoint::new("two", &[i % 37, (i * 13) % 101]).unwrap());
        w.add_fields_document(&d).unwrap();
    }
    w.commit().unwrap();
    drop(w);
    let results = check_directory(&dir).unwrap();
    for field in ["one", "two"] {
        for family in [
            "points.value_within_field_bounds",
            "points.point_count_matches",
        ] {
            let name = format!("{family}:{field}");
            let c = results[1]
                .checks
                .iter()
                .find(|c| c.name == name)
                .unwrap_or_else(|| panic!("{name} ran"));
            assert!(c.passed(), "{c:?}");
        }
    }
    assert!(results.iter().all(CheckResult::all_passed));

    let infos = segment_infos::read_latest(&dir).unwrap();
    let commit = &infos.segments[0];
    let open = |ext: &str| dir.open(&format!("{}.{ext}", commit.segment_name)).unwrap();
    let (kdm, kdi, kdd) = (open("kdm"), open("kdi"), open("kdd"));
    let reader = points::open(&kdm, &kdi, &kdd, &commit.segment_id, "").unwrap();
    let fis = crate::index_writer::segment_field_infos(&dir, commit).unwrap();
    let number = |name: &str| fis.fields.iter().find(|f| f.name == name).unwrap().number;
    let one = reader.field(number("one")).unwrap();
    assert!(one.num_leaves > 1, "a multi-leaf tree");

    let v = |x: u8| [0x80, 0, 0, x];
    let mut cells = Problems::new();
    let mut visitor = VerifyPointsVisitor::new("one", one, &mut cells);
    // In order, ties by docID: fine.
    visitor.compare(&v(0), &v(0xff));
    visitor.visit_with_value(3, &v(5));
    visitor.visit_with_value(4, &v(5));
    visitor.visit_with_value(1, &v(6));
    assert_eq!((visitor.in_cell.total, visitor.order.total), (0, 0));
    // A tie with a smaller docID, then a smaller value.
    visitor.visit_with_value(0, &v(6));
    visitor.visit_with_value(9, &v(2));
    assert_eq!(visitor.order.total, 2, "{:?}", visitor.order.shown);
    // A point outside the last cell.
    visitor.compare(&v(10), &v(20));
    visitor.visit_with_value(10, &v(30));
    assert_eq!(visitor.in_cell.total, 1, "{:?}", visitor.in_cell.shown);
    // An inverted cell.
    visitor.compare(&v(20), &v(10));
    drop(visitor);
    assert!(cells.total >= 1, "{:?}", cells.shown);
    assert!(cells.shown[0].contains("out-of-bounds of the cell's maxPackedValue"));
}

/// Not a correctness test: times `check_directory` over a 200k-document
/// segment with every doc-values type, sparse and dense, and an index
/// sort. `cargo test -p lucene-index --release --lib
/// check_index::tests::bench_doc_values_check -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_doc_values_check() {
    use crate::document::{
        BinaryDocValuesField, Document, NumericDocValuesField, SortedDocValuesField,
        SortedNumericDocValuesField, SortedSetDocValuesField,
    };
    use crate::index_writer::IndexWriter;
    let dst = tempdir();
    let dir = FsDirectory::open(&dst);
    let mut w = IndexWriter::open(
        &dir,
        numeric_dv_fields(&["dense"]),
        "Lucene104",
        writer_version(),
    )
    .unwrap();
    w.add_doc_values_field("dense").unwrap();
    w.set_index_sort(Some(&[segment_info::IndexSortField::long(
        "dense", false, None,
    )]))
    .unwrap();
    let n = 200_000i64;
    for i in 0..n {
        let mut d = Document::new();
        d.add(NumericDocValuesField::new("dense", i));
        if i % 3 == 0 {
            d.add(NumericDocValuesField::new("sparse", i * 31 % 1000));
            d.add(BinaryDocValuesField::new("bin", format!("v{}", i % 977)));
        }
        if i % 2 == 0 {
            d.add(SortedNumericDocValuesField::new("sn", i % 50));
            d.add(SortedNumericDocValuesField::new("sn", i % 70));
            d.add(SortedSetDocValuesField::new("set", format!("s{}", i % 300)));
            d.add(SortedSetDocValuesField::new("set", format!("t{}", i % 200)));
        }
        if i % 5 != 0 {
            d.add(SortedDocValuesField::new(
                "sorted",
                format!("k{}", i % 1000),
            ));
        }
        w.add_fields_document(&d).unwrap();
    }
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    drop(w);
    let _ = check_directory(&dir).unwrap();
    let mut best = std::time::Duration::MAX;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let results = check_directory(&dir).unwrap();
        best = best.min(t.elapsed());
        assert!(results.iter().all(CheckResult::all_passed));
    }
    eprintln!("check_directory over {n} docs: best of 3 = {best:?}");
}

/// Two one-document segments, as Lucene 10.6's `TestCheckIndex` builds
/// them; returns the second segment's name.
fn two_segments(dir: &FsDirectory) -> String {
    use crate::document::{Document, IntPoint};
    use crate::index_writer::IndexWriter;
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", writer_version()).unwrap();
    for i in 0..2 {
        let mut d = Document::new();
        d.add(IntPoint::new("id", &[i]).unwrap());
        w.add_fields_document(&d).unwrap();
        w.commit().unwrap();
    }
    drop(w);
    let infos = segment_infos::read_latest(dir).unwrap();
    assert_eq!(infos.segments.len(), 2);
    infos.segments[1].segment_name.clone()
}

/// Lucene 10.6's `TestCheckIndex.testCorruptSegmentInfoNamesTheSegment`
/// (#16476): a deleted or truncated `.si` is reported against the
/// segment it belongs to, with a cause naming the file.
#[test]
fn corrupt_segment_info_names_the_segment() {
    for corruption in ["delete-si", "truncate-si"] {
        let dst = tempdir();
        let dir = FsDirectory::open(&dst);
        let seg = two_segments(&dir);
        let si = dst.join(format!("{seg}.si"));
        if corruption == "delete-si" {
            std::fs::remove_file(&si).unwrap();
        } else {
            let bytes = std::fs::read(&si).unwrap();
            std::fs::write(&si, &bytes[..bytes.len() * 7 / 10]).unwrap();
        }
        let results = check_directory(&dir).unwrap();
        assert!(results[1].all_passed(), "{corruption}");
        let broken = &results[2];
        assert_eq!(broken.segment_name, seg);
        let c = &broken.checks[0];
        assert_eq!((c.name.as_str(), c.passed()), ("si.open", false));
        assert!(
            format!("{c:?}").contains(&format!(
                "segment info file: {seg}.si cannot be read - it may be missing or corrupt"
            )),
            "{corruption}: {c:?}"
        );
    }
}

/// Lucene 10.6's
/// `TestCheckIndex.testSegmentInfoIOExceptionDoesNotThrowCorruptionException`
/// (#16788): an I/O failure opening a `.si` is not blamed on the segment;
/// the check fails as a whole with the I/O error. (Java's "fail-read"
/// case has no twin: [`Directory::open`] hands back the whole file.)
#[test]
fn segment_info_io_error_is_not_reported_as_corruption() {
    struct FailingSi<'a>(&'a FsDirectory, String);
    impl Directory for FailingSi<'_> {
        fn list_all(&self) -> lucene_store::Result<Vec<String>> {
            self.0.list_all()
        }
        fn open(&self, name: &str) -> lucene_store::Result<lucene_store::Input> {
            if name == self.1 {
                return Err(lucene_store::Error::Io(std::io::Error::other(format!(
                    "simulated I/O failure opening {name}"
                ))));
            }
            self.0.open(name)
        }
        fn create_output(&self, name: &str) -> lucene_store::Result<lucene_store::FsIndexOutput> {
            self.0.create_output(name)
        }
        fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
            self.0.sync(names)
        }
        fn rename(&self, source: &str, dest: &str) -> lucene_store::Result<()> {
            self.0.rename(source, dest)
        }
        fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
            self.0.delete_file(name)
        }
        fn sync_meta_data(&self) -> lucene_store::Result<()> {
            self.0.sync_meta_data()
        }
        fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn lucene_store::Lock>> {
            self.0.obtain_lock(name)
        }
    }
    let dst = tempdir();
    let dir = FsDirectory::open(&dst);
    let seg = two_segments(&dir);
    let failing = FailingSi(&dir, format!("{seg}.si"));
    let err = check_directory(&failing).unwrap_err();
    assert!(
        matches!(&err, segment_infos::Error::Store(lucene_store::Error::Io(e))
                if e.kind() == std::io::ErrorKind::Other),
        "{err:?}"
    );
    assert!(
        err.to_string().starts_with("simulated I/O failure"),
        "{err}"
    );
}
