//! `CheckIndex.testPostings`: [`check_postings`] walks every field's term
//! dictionary and cross-checks it against the postings it decodes. It
//! reports one named check per field and kind of problem; the pieces below
//! each own one of those checks.

use super::*;

/// The problems one field's term walk collects, one list per check kind.
#[derive(Default)]
struct TermProblems {
    freq_mismatches: Vec<String>,
    doc_id_problems: Vec<String>,
    order_problems: Vec<String>,
    doc_freq_problems: Vec<String>,
    term_stat_problems: Vec<String>,
    position_problems: Vec<String>,
    offset_problems: Vec<String>,
    advance_problems: Vec<String>,
    run_problems: Vec<String>,
    impact_problems: Vec<String>,
}

/// What the per-term checks of one field share.
#[derive(Clone, Copy)]
struct FieldContext<'a> {
    si: &'a SegmentInfo,
    live_docs: Option<&'a lucene_util::fixed_bit_set::FixedBitSet>,
    field_name: &'a str,
    has_offsets: bool,
}

/// One term's dictionary statistics and decoded postings.
#[derive(Clone, Copy)]
struct DecodedTerm<'a> {
    claimed: &'a blocktree::TermStats,
    postings_of_term: &'a postings::Postings,
    positions: Option<&'a [Vec<postings::Position>]>,
}

/// What walking one field's terms leaves for the checks that follow it.
struct TermWalkOutcome {
    problems: TermProblems,
    needs_doc_file: bool,
    /// The last term the dictionary enumerated.
    prev_term: Option<Vec<u8>>,
    first_term: Option<Vec<u8>>,
    counted_terms: i64,
    summed_doc_freq: i64,
    summed_total_term_freq: i64,
    visited_docs: lucene_util::fixed_bit_set::FixedBitSet,
    /// Terms sampled for the seek/reseek round.
    seek_samples: Vec<Vec<u8>>,
    decode_error: Option<String>,
}

/// `CheckIndex.testPostings` -> `checkFields(fields, liveDocs, maxDoc,
/// fieldInfos, normsProducer, true, false, ...)`, the deepest single check
/// real `CheckIndex` performs.
///
/// Per field it re-derives, from the postings themselves, everything the
/// term dictionary and `.tmd` *claim*, and walks the positional streams:
///
/// | check | Java |
/// |---|---|
/// | `postings.field_in_fnm:<f>` | `isIndexed == false` (the `fieldsEnum inconsistent with fieldInfos` half is unfirable here -- see below) |
/// | `postings.term_dict_shape:<f>` | `hasPayloads`/`hasOffsets` vs `.fnm` (`docCount > maxDoc` and `minTerm`/`maxTerm` both-or-neither are unfirable -- see below) |
/// | `postings.terms_sorted:<f>` | `terms out of order` |
/// | `postings.doc_freq_positive:<f>` | `docfreq: N is out of bounds` |
/// | `postings.term_stats:<f>` | `totalTermFreq <= 0`, `totalTermFreq < docFreq`, term outside `[minTerm, maxTerm]`, and the accumulator overflowing `i64` |
/// | `postings.total_term_freq:<f>` | `totalTermFreq != recomputed totalTermFreq` |
/// | `postings.doc_ids_valid:<f>` | `doc <= lastDoc`, `doc >= maxDoc`, `freq <= 0` |
/// | `postings.positions_valid:<f>` | `pos < 0`, `pos > MAX_POSITION`, `pos < lastPos` |
/// | `postings.offsets_valid:<f>` | `startOffset < 0`, `startOffset < lastOffset`, `endOffset < 0`, `endOffset < startOffset` |
/// | `postings.advance_agrees:<f>` | the `Test skipping` block's seven `advance(maxDoc*i/8)` probes |
/// | `postings.doc_id_runs:<f>` | `checkDocIDRuns`, over docs-only and freqs cursors |
/// | `postings.impacts:<f>` | the slow level's impacts walk and the hashed `advance`/`advanceShallow` walk, each step through `checkImpacts` |
/// | `postings.seek_agrees:<f>` | `Test seeking by ord` / `seek to last term` / `seek to existing term ... failed` |
/// | `postings.intersect_agrees:<f>` | `checkTermsIntersect` |
/// | `postings.field_summary:<f>` | `sumDocFreq`/`sumTotalTermFreq`/`docCount`/`termCount`/`minTerm`/`maxTerm` vs recomputed |
///
/// Returns the counts real `CheckIndex` prints for `test: terms, freq,
/// prox...OK [N terms; M terms/docs pairs; K tokens]` plus, per field, the
/// set of documents that carry at least one term (`checkFields`'
/// `visitedDocs`), which [`check_field_norms`] consumes.
///
/// **Deliberately not ported, with reasons** (rather than silently omitted):
///
/// - *`docCount != docFreq` per term.* This port's decode API is
///   parameterized by the claimed `docFreq` (like Java's own
///   `BlockDocsEnum`/`TermState.docFreq`), so a recount can never disagree
///   -- see this module's top comment. The observable symptom of a wrong
///   `docFreq` is an out-of-range or non-increasing doc ID, which
///   `postings.doc_ids_valid` checks directly.
/// - *Every check that reads back the `.fnm`'s **frequency** flag* --
///   `Terms.hasFreqs()` vs `IndexOptions`, `!hasFreqs => totalTermFreq ==
///   docFreq`, `!hasFreqs => freq == 1`, `!hasFreqs => sumTotalTermFreq ==
///   sumDocFreq`. In Lucene these getters are *derived from*
///   `FieldInfo.getIndexOptions()`, and so is this port's decode path:
///   `blocktree` sets `totalTermFreq = docFreq` for an `IndexOptions::Docs`
///   field, `read_freq_pair` returns `(sumDocFreq, sumDocFreq)` for one, and
///   `read_tail_block`/`refill_full_block` do `freqs.fill(1)`. All four are
///   `x == x`, so nothing in this function reads
///   `has_freqs` at all -- **the `.fnm`'s frequency flag has no independent
///   on-disk witness in this port.** `hasPayloads`/`hasOffsets` *are*
///   checked, because those do have one: whether the segment carries a `.pay`
///   file.
/// - *A field in the dictionary with no `.fnm` entry, and `.tmd`'s `docCount
///   > maxDoc` / `sumTotalTermFreq < sumDocFreq`.* `blocktree::open` rejects
///   all three while parsing (`InvalidFieldNumber`, `InvalidDocCount`,
///   `InvalidSumTotalTermFreq`), the first because it takes each field's
///   *name* from the very `FieldInfos` a check here would compare against.
///   A file that would trip them is reported as `postings.open`.
/// - *`payload.length < 1`, and any payload check at all.* Java guards
///   against a codec handing back a zero-length `BytesRef`; this port
///   represents "no payload at this position" as an empty `Vec<u8>`, so a
///   zero-length payload and no payload are the same value. Nor can a
///   payload appear on a field whose `.fnm` says `storePayloads = false`:
///   `read_positions` is *told* whether to read payloads by the very same
///   `FieldInfo` flag, so the comparison is `x == x`. Payloads *are*
///   cross-checked, in the one place two independent copies of them exist
///   on disk -- `term_vectors.match_postings`.
/// - *`freq` vs. the number of decoded positions.* `read_positions` chops
///   the flat position stream into per-document groups *using* `freqs`, so
///   the group's length is `freq` by construction or the decode fails
///   outright. Same `x == x` shape.
/// - *`impacts`/`checkImpacts`/`checkDocIDRuns`.* Java's slow level
///   cross-checks the `ImpactsEnum` against the `PostingsEnum` and the
///   `docIDRunEnd` API. Here the lazy cursor (`advance_shallow`, the level-0
///   and level-1 impacts, `doc_id_run_end`) is walked the same two ways and
///   compared against the term's fully decoded doc list, which is the
///   independent copy (`postings.impacts`, `postings.doc_id_runs`, see
///   [`check_term_impacts`]). Java runs them at the slow level or on long or
///   every 1024th term; this module always runs its slow checks.
pub(super) fn check_postings(
    si: &SegmentInfo,
    field_infos: &FieldInfos,
    live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>,
    p: &PostingsHandles<'_>,
    checks: &mut Vec<Check>,
) -> (CheckStats, DocsWithTerms) {
    let mut stats = CheckStats::default();
    let mut docs_with_terms: DocsWithTerms = std::collections::HashMap::new();
    // One name -> `FieldInfo` map for the whole walk rather than a linear
    // `.fnm` scan per field. The first entry wins, as `find` did.
    let mut by_name: std::collections::HashMap<&str, &field_infos::FieldInfo> =
        std::collections::HashMap::with_capacity(field_infos.fields.len());
    for f in &field_infos.fields {
        by_name.entry(f.name.as_str()).or_insert(f);
    }
    // `checkTermsIntersect`'s two automata, compiled once per segment
    // rather than once per field.
    let intersect_automata = IntersectAutomata::build();
    // Reported once for the segment, but tracked *per field*: a field that
    // needs a missing `.doc` must not suppress the next field's `docCount`
    // re-derivation and `visitedDocs` (which would make that field's
    // `norms.agree_with_postings` silently vanish rather than fail).
    let mut any_field_needs_doc_file = false;
    let has_pay_file = p.pay_in.is_some();

    for (field_name, field_terms) in p.fields.iter_fields() {
        // Java's "fieldsEnum inconsistent with fieldInfos" -- a field in the
        // dictionary that `.fnm` has no entry for -- is **not** ported: it
        // cannot fire. `blocktree::open` resolves each `.tmd` record's field
        // *number* through `FieldInfos::field_by_number` (rejecting an
        // unknown one as `Error::InvalidFieldNumber`) and names the field
        // from what it found, so every name `iter_fields` yields came out of
        // this very `FieldInfos`. The lookup below is `x == x`; only the
        // `indexOptions=None` half has an independent claim to compare.
        let Some(&fi) = by_name.get(field_name) else {
            debug_assert!(
                false,
                "blocktree::open names every field from this FieldInfos, and rejects \
                 duplicate names, so the lookup cannot miss"
            );
            continue;
        };
        let index_options = fi.index_options;
        // No `has_freqs` here: every one of `checkFields`' three
        // freq-flag checks is `x == x` in this port and is documented as such
        // (see `postings.term_stats`
        // above, `postings.doc_ids_valid` below and `postings.field_summary`
        // at the end of this loop). `blocktree`'s decoder derives
        // `totalTermFreq`, `freq` and `sumTotalTermFreq` from the *same*
        // `.fnm` `indexOptions` this function would compare them against.
        //
        // The flag is not *unwitnessed* -- `index_has_freq` selects the
        // `.doc` block layout, so a `.fnm` that disagrees with the file that
        // was written misparses the stream -- but the witness is a decode
        // failure (`postings.terms_decode`, `postings.doc_ids_valid`), not a
        // value this function can compare against anything.
        let has_positions = matches!(
            index_options,
            field_infos::IndexOptions::DocsAndFreqsAndPositions
                | field_infos::IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        );
        let has_offsets = matches!(
            index_options,
            field_infos::IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        );
        let has_freqs = !matches!(
            index_options,
            field_infos::IndexOptions::None | field_infos::IndexOptions::Docs
        );
        check_field_shape(
            checks,
            field_name,
            field_terms,
            fi,
            (has_positions, has_offsets),
            p,
            has_pay_file,
        );

        let mut outcome = walk_field_terms(
            si,
            live_docs,
            p,
            field_name,
            field_terms,
            (has_positions, has_offsets, has_freqs),
            &mut stats,
        );
        any_field_needs_doc_file |= outcome.needs_doc_file;
        if let Some(e) = outcome.decode_error.take() {
            checks.push(Check::fail(
                format!("postings.terms_decode:{field_name}"),
                e,
            ));
        }
        check_seek_round(checks, field_name, field_terms, &outcome);
        check_intersect_round(checks, field_name, field_terms, &intersect_automata);
        push_term_problem_checks(
            checks,
            field_name,
            field_terms,
            (has_positions, has_offsets),
            &outcome.problems,
        );
        check_field_summary(
            checks,
            field_name,
            field_terms,
            fi,
            outcome,
            &mut docs_with_terms,
        );
    }
    if any_field_needs_doc_file {
        checks.push(Check::fail(
            "postings.doc_open",
            "a term with docFreq > 1 needs the segment's .doc file, but none was found",
        ));
    }
    (stats, docs_with_terms)
}

/// `checkFields`' shape checks: the field has a `.fnm` entry that indexes it,
/// and the segment has the files that entry needs.
fn check_field_shape(
    checks: &mut Vec<Check>,
    field_name: &str,
    field_terms: &blocktree::FieldTerms,
    fi: &field_infos::FieldInfo,
    (has_positions, has_offsets): (bool, bool),
    p: &PostingsHandles<'_>,
    has_pay_file: bool,
) {
    let mut shape_problems: Vec<String> = Vec::new();
    if fi.index_options == field_infos::IndexOptions::None {
        shape_problems.push(format!(
            "field {field_name:?} has a term dictionary but .fnm says indexOptions=None"
        ));
    }
    checks.push(named_field_check(
        &format!("postings.field_in_fnm:{field_name}"),
        &shape_problems,
        1,
        "field",
    ));

    let mut dict_problems: Vec<String> = Vec::new();
    // `checkFields`' "docCount > maxDoc for field" is **not** ported: it
    // cannot fire here. `blocktree::open` is handed `si.doc_count` as its
    // `max_doc` and rejects a `.tmd` whose `docCount` is outside
    // `0..=max_doc` (`Error::InvalidDocCount`) while parsing, so a
    // dictionary that would trip this check never reaches
    // `check_postings` at all -- it is reported as `postings.open`. This
    // module ships no check that can never fire.
    //
    // Java's `(minTerm == null) != (maxTerm == null)` has no
    // counterpart here: this port stores both as (possibly empty) byte
    // strings rather than nullables, so "absent" and "the empty term"
    // are the same value and the asymmetry cannot be observed. The
    // stronger property is checked instead, by
    // `postings.field_summary`: `.tmd`'s minTerm/maxTerm must equal the
    // first and last terms the dictionary actually enumerates.
    // `.pay` exists exactly when the field indexes offsets or payloads
    // (`Lucene104PostingsWriter`), so `.fnm`'s flags have an independent
    // on-disk witness here even though `hasFreqs`/`hasPositions` do not.
    if (has_offsets || fi.store_payloads) && has_positions && !has_pay_file {
        dict_problems.push(format!(
            "field {field_name:?}: .fnm says hasOffsets={has_offsets} \
                 storePayloads={} but the segment has no .pay file",
            fi.store_payloads
        ));
    }
    if has_positions && p.pos_in.is_none() {
        dict_problems.push(format!(
            "field {field_name:?}: .fnm says the field indexes positions but the segment \
                 has no .pos file"
        ));
    }
    checks.push(named_field_check(
        &format!("postings.term_dict_shape:{field_name}"),
        &dict_problems,
        field_terms.num_terms,
        "terms",
    ));
}

/// Walks one field's terms in order, decoding each term's postings and
/// running the per-term checks on them.
fn walk_field_terms(
    si: &SegmentInfo,
    live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>,
    p: &PostingsHandles<'_>,
    field_name: &str,
    field_terms: &blocktree::FieldTerms,
    (has_positions, has_offsets, has_freqs): (bool, bool, bool),
    stats: &mut CheckStats,
) -> TermWalkOutcome {
    let mut problems = TermProblems::default();
    let mut needs_doc_file = false;
    let mut prev_term: Option<Vec<u8>> = None;
    let mut first_term: Option<Vec<u8>> = None;
    let mut counted_terms = 0i64;
    let mut summed_doc_freq = 0i64;
    let mut summed_total_term_freq = 0i64;
    let mut visited_docs =
        lucene_util::fixed_bit_set::FixedBitSet::new(si.doc_count.max(0) as usize);
    // Terms sampled for the seek/reseek round, at most 10 000 of them --
    // Java's own `seekCount = min(10000, termCount)` cap.
    let mut seek_samples: Vec<Vec<u8>> = Vec::new();
    let seek_stride = (field_terms.num_terms / 10_000).max(1);

    let mut terms = field_terms.iter();
    let mut term_buf: Vec<u8> = Vec::new();
    let mut decode_error: Option<String> = None;
    loop {
        let stats_now = match terms.try_next() {
            Ok(Some((term, s))) => {
                term_buf.clear();
                term_buf.extend_from_slice(term);
                Some(s)
            }
            Ok(None) => None,
            Err(e) => {
                decode_error = Some(format!("field {field_name:?}: {e}"));
                None
            }
        };
        let Some(claimed) = stats_now else { break };
        let term = term_buf.as_slice();

        counted_terms = counted_terms.saturating_add(1);
        // `doc_freq`/`total_term_freq` come straight off `.tim`/`.tmd`, so
        // a corrupt dictionary can hand this loop `i64::MAX` repeatedly.
        //
        // `summed_doc_freq` saturates: `doc_freq` is an `i32`, so reaching
        // `i64::MAX` would take 2^32 terms, and a saturated value cannot
        // accidentally equal the `.tmd`'s own claim in any real file.
        //
        // `summed_total_term_freq` must **not** saturate. Its operand is
        // itself an `i64` off disk and it is compared against
        // `sum_total_term_freq`, another `i64` off disk -- so a `.tmd`
        // claiming `i64::MAX` paired with per-term values that overflow
        // would compare *equal*, and the check would pass on exactly the
        // file it exists to reject. Report the overflow instead.
        summed_doc_freq = summed_doc_freq.saturating_add(i64::from(claimed.doc_freq));
        match summed_total_term_freq.checked_add(claimed.total_term_freq) {
            Some(sum) => summed_total_term_freq = sum,
            None => {
                problems.term_stat_problems.push(format!(
                    "the per-term totalTermFreq values overflow i64 at term {:?}",
                    String::from_utf8_lossy(term)
                ));
                break;
            }
        }
        if first_term.is_none() {
            first_term = Some(term.to_vec());
        }
        // ARITH: `seek_stride` is `(num_terms / 10_000).max(1)`, so it is
        // at least 1 -- neither a divide-by-zero nor the `i64::MIN % -1`
        // overflow is reachable.
        #[allow(clippy::arithmetic_side_effects)]
        let sample = counted_terms % seek_stride == 0;
        if sample && seek_samples.len() < 10_000 {
            seek_samples.push(term.to_vec());
        }
        check_term_entry(
            &mut problems,
            field_name,
            field_terms,
            term,
            &claimed,
            prev_term.as_deref(),
        );
        prev_term = Some(term.to_vec());

        if claimed.doc_freq > 1 && p.doc_in.is_none() {
            needs_doc_file = true;
            continue;
        }

        // One metadata decode for both the docs/freqs and, when the
        // field indexes them, the positions/offsets/payloads.
        let decoded = if has_positions {
            match p.pos_in.as_ref() {
                Some(pos_in) => terms
                    .try_current_postings_and_positions(
                        p.doc_in.as_ref(),
                        pos_in,
                        p.pay_in.as_ref(),
                    )
                    .map(|o| o.map(|(d, pos)| (d, Some(pos)))),
                // Already reported by `postings.term_dict_shape`.
                None => Ok(None),
            }
        } else {
            terms
                .try_current_postings(p.doc_in.as_ref())
                .map(|o| o.map(|d| (d, None)))
        };
        let (postings_of_term, positions) = match decoded {
            Ok(Some(v)) => v,
            Ok(None) => continue,
            Err(e) => {
                decode_error = Some(format!("field {field_name:?} term {term:?}: {e}"));
                break;
            }
        };

        check_term_docs(
            &FieldContext {
                si,
                live_docs,
                field_name,
                has_offsets,
            },
            term,
            &DecodedTerm {
                claimed: &claimed,
                postings_of_term: &postings_of_term,
                positions: positions.as_deref(),
            },
            &mut problems,
            &mut visited_docs,
            stats,
        );

        if let Some(doc_in) = p.doc_in.as_ref() {
            if claimed.doc_freq > 1 {
                check_term_advance(
                    si,
                    field_name,
                    field_terms,
                    term,
                    doc_in,
                    &postings_of_term,
                    &mut problems.advance_problems,
                );
            }
            if claimed.doc_freq > 1
                && (problems.run_problems.is_empty() || problems.impact_problems.is_empty())
            {
                check_term_runs_and_impacts(
                    field_name,
                    field_terms,
                    term,
                    doc_in,
                    has_freqs,
                    &postings_of_term,
                    &mut problems,
                );
            }
        }
    }
    TermWalkOutcome {
        problems,
        needs_doc_file,
        prev_term,
        first_term,
        counted_terms,
        summed_doc_freq,
        summed_total_term_freq,
        visited_docs,
        seek_samples,
        decode_error,
    }
}

/// The dictionary-entry checks on one term: sorted order, positive
/// statistics, and inside `.tmd`'s min/max term.
fn check_term_entry(
    problems: &mut TermProblems,
    field_name: &str,
    field_terms: &blocktree::FieldTerms,
    term: &[u8],
    claimed: &blocktree::TermStats,
    prev_term: Option<&[u8]>,
) {
    if let Some(prev) = prev_term {
        if term <= prev {
            problems.order_problems.push(format!(
                "field {field_name:?}: term {term:?} does not come after previous \
                 term {prev:?}"
            ));
        }
    }
    if claimed.doc_freq <= 0 {
        problems.doc_freq_problems.push(format!(
            "field {field_name:?} term {term:?}: docFreq={} is not > 0",
            claimed.doc_freq
        ));
    }
    // `checkFields`' term-statistic bounds.
    if claimed.total_term_freq <= 0 {
        problems.term_stat_problems.push(format!(
            "field {field_name:?} term {term:?}: totalTermFreq={} is not > 0",
            claimed.total_term_freq
        ));
    }
    if claimed.total_term_freq < claimed.doc_freq as i64 {
        problems.term_stat_problems.push(format!(
            "field {field_name:?} term {term:?}: totalTermFreq={} < docFreq={}",
            claimed.total_term_freq, claimed.doc_freq
        ));
    }
    if !field_terms.min_term.is_empty() && term < field_terms.min_term.as_slice() {
        problems.term_stat_problems.push(format!(
            "field {field_name:?}: term {term:?} sorts before .tmd's minTerm {:?}",
            field_terms.min_term
        ));
    }
    if !field_terms.max_term.is_empty() && term > field_terms.max_term.as_slice() {
        problems.term_stat_problems.push(format!(
            "field {field_name:?}: term {term:?} sorts after .tmd's maxTerm {:?}",
            field_terms.max_term
        ));
    }
}

/// One term's decoded postings: `totalTermFreq` re-derived from the freqs,
/// every doc id in range and increasing, positions and offsets.
fn check_term_docs(
    ctx: &FieldContext<'_>,
    term: &[u8],
    decoded: &DecodedTerm<'_>,
    problems: &mut TermProblems,
    visited_docs: &mut lucene_util::fixed_bit_set::FixedBitSet,
    stats: &mut CheckStats,
) {
    let FieldContext {
        si,
        live_docs,
        field_name,
        has_offsets,
    } = *ctx;
    let DecodedTerm {
        claimed,
        postings_of_term,
        positions,
    } = *decoded;
    // `totalTermFreq` re-derivation: each doc's freq comes straight
    // off the wire, so this is genuinely independent of the claim.
    let actual_total_term_freq: i64 = postings_of_term.freqs.iter().map(|&f| f as i64).sum();
    if actual_total_term_freq != claimed.total_term_freq {
        problems.freq_mismatches.push(format!(
            "field {field_name:?} term {term:?}: dictionary claims \
                     totalTermFreq={}, but postings actually sum to {actual_total_term_freq}",
            claimed.total_term_freq
        ));
    }

    let mut prev_doc_id = -1i32;
    let mut term_is_live = false;
    for (i, &doc_id) in postings_of_term.docs.iter().enumerate() {
        // FBS: `visited_docs` is `FixedBitSet::new(si.doc_count.max(0))`
        // in this same function, so `si.doc_count` *is* its `len()`
        // whenever it is positive, and the guard is vacuous when it is
        // not.
        if doc_id >= 0 && doc_id < si.doc_count {
            visited_docs.set(doc_id as usize);
        }
        if doc_id <= prev_doc_id || doc_id >= si.doc_count {
            problems.doc_id_problems.push(format!(
                "field {field_name:?} term {term:?}: decoded doc ID {doc_id} is not \
                         in the valid strictly-increasing 0..{} range (previous was \
                         {prev_doc_id})",
                si.doc_count
            ));
            break;
        }
        prev_doc_id = doc_id;
        let freq = postings_of_term.freqs.get(i).copied().unwrap_or(0);
        if freq <= 0 {
            problems.doc_id_problems.push(format!(
                "field {field_name:?} term {term:?}: doc {doc_id} has freq {freq}, \
                         which is out of bounds"
            ));
        }
        let live = is_live(live_docs, doc_id);
        if live {
            term_is_live = true;
            stats.term_doc_pairs = stats.term_doc_pairs.saturating_add(1);
            stats.token_count = stats.token_count.saturating_add(i64::from(freq.max(0)));
        }

        // Positions/offsets/payloads for this doc.
        if let Some(per_doc) = positions.and_then(|p| p.get(i)) {
            check_occurrences(
                field_name,
                term,
                doc_id,
                per_doc,
                has_offsets,
                &mut problems.position_problems,
                &mut problems.offset_problems,
            );
        }
    }
    if term_is_live {
        stats.term_count = stats.term_count.saturating_add(1);
    } else {
        stats.del_term_count = stats.del_term_count.saturating_add(1);
    }
}

// `checkFields`' "Test skipping" block: seven `advance` probes
// spread across the doc-id space. Java only checks the returned
// doc is `>= target` and that `nextDoc` moves forward; this port
// already holds the term's whole decoded doc list, so it checks
// the stronger property -- the skip-list-driven `advance` lands
// on exactly the doc a linear scan of that list would. That is a
// real cross-check of the `.doc` level-0/level-1 skip data
// against the block payload it indexes.
fn check_term_advance(
    si: &SegmentInfo,
    field_name: &str,
    field_terms: &blocktree::FieldTerms,
    term: &[u8],
    doc_in: &DocInput<'_>,
    postings_of_term: &postings::Postings,
    advance_problems: &mut Vec<String>,
) {
    match field_terms.lazy_postings(term, doc_in) {
        Ok(Some(mut cursor)) => {
            // ARITH: `idx` is 0..7 and `doc_count` is an
            // `i32`, so the product is at most 8 * 2^31 --
            // three orders of magnitude inside `i64`.
            #[allow(clippy::arithmetic_side_effects)]
            for idx in 0..7i64 {
                let target =
                    ((idx + 1) * i64::from(si.doc_count) / 8).clamp(0, i32::MAX as i64) as i32;
                let expected = postings_of_term
                    .docs
                    .iter()
                    .copied()
                    .find(|&d| d >= target)
                    .unwrap_or(postings::NO_MORE_DOCS);
                match cursor.advance(target) {
                    Ok(got) if got == expected => {}
                    Ok(got) => {
                        advance_problems.push(format!(
                            "field {field_name:?} term {term:?}: \
                                             advance({target}) returned {got}, but the decoded \
                                             doc list's first doc >= {target} is {expected}"
                        ));
                        break;
                    }
                    Err(e) => {
                        advance_problems.push(format!(
                            "field {field_name:?} term {term:?}: \
                                             advance({target}) failed: {e}"
                        ));
                        break;
                    }
                }
                if expected == postings::NO_MORE_DOCS {
                    break;
                }
            }
        }
        Ok(None) => advance_problems.push(format!(
            "field {field_name:?}: term {term:?} was enumerated but a re-seek \
                             for its postings found nothing"
        )),
        Err(e) => advance_problems.push(format!(
            "field {field_name:?} term {term:?}: re-seek failed: {e}"
        )),
    }
}

/// A single-document term has no `.doc` postings to walk (it is inlined in
/// the term dictionary), as for the probes above. Checks the doc-id runs and
/// the impacts of one term against its decoded postings.
fn check_term_runs_and_impacts(
    field_name: &str,
    field_terms: &blocktree::FieldTerms,
    term: &[u8],
    doc_in: &DocInput<'_>,
    has_freqs: bool,
    postings_of_term: &postings::Postings,
    problems: &mut TermProblems,
) {
    let freqs: &[i32] = if has_freqs {
        &postings_of_term.freqs
    } else {
        &[]
    };
    let reference = TermDocs {
        docs: &postings_of_term.docs,
        freqs,
    };
    let flags = [
        postings::PostingsFlags::DocsOnly,
        postings::PostingsFlags::FreqsNoImpacts,
    ];
    for flags in flags {
        if !problems.run_problems.is_empty() {
            break;
        }
        let cursor = field_terms.lazy_postings_with_flags(term, doc_in, flags);
        if let Err(e) = cursor
            .map_err(|e| e.to_string())
            .and_then(|c| c.ok_or_else(|| "the re-seek found nothing".to_string()))
            .and_then(|mut c| check_doc_id_runs(&mut c))
        {
            problems.run_problems.push(format!(
                "field {field_name:?} term {term:?} ({flags:?}): {e}"
            ));
        }
    }
    if problems.impact_problems.is_empty() {
        if let Err(e) = check_term_impacts(field_name, has_freqs, &reference, || match field_terms
            .lazy_postings(term, doc_in)
        {
            Ok(Some(c)) => Ok(c),
            Ok(None) => Err("the re-seek found nothing".to_string()),
            Err(e) => Err(e.to_string()),
        }) {
            problems
                .impact_problems
                .push(format!("field {field_name:?} term {term:?}: {e}"));
        }
    }
}

// `checkFields`' seek round: every sampled term must be findable
// again by an independent `seekExact`/`seekCeil` from the trie root,
// returning the same statistics the forward scan reported. `try_*`
// is deliberate: a corrupt block must surface as an error
// here, not as "term not found".
fn check_seek_round(
    checks: &mut Vec<Check>,
    field_name: &str,
    field_terms: &blocktree::FieldTerms,
    outcome: &TermWalkOutcome,
) {
    let seek_samples = &outcome.seek_samples;
    let prev_term = &outcome.prev_term;
    let mut seek_problems: Vec<String> = Vec::new();
    {
        let mut seek_enum = field_terms.iter();
        for sample in seek_samples {
            match field_terms.try_seek_exact(sample) {
                Ok(Some(_)) => {}
                Ok(None) => seek_problems.push(format!(
                    "field {field_name:?}: term {sample:?} was enumerated but \
                         seekExact cannot find it"
                )),
                Err(e) => seek_problems.push(format!(
                    "field {field_name:?}: seekExact({sample:?}) failed: {e}"
                )),
            }
            match seek_enum.try_seek_ceil(sample) {
                Ok(blocktree::SeekStatus::Found) => match seek_enum.try_current() {
                    Ok(Some((landed, _))) if landed == sample.as_slice() => {}
                    Ok(Some((landed, _))) => seek_problems.push(format!(
                        "field {field_name:?}: seekCeil({sample:?}) reported FOUND but \
                             landed on {landed:?}"
                    )),
                    Ok(None) => seek_problems.push(format!(
                        "field {field_name:?}: seekCeil({sample:?}) reported FOUND but the \
                             cursor is on no term"
                    )),
                    Err(e) => seek_problems.push(format!(
                        "field {field_name:?}: seekCeil({sample:?}) landed on an \
                             undecodable term: {e}"
                    )),
                },
                Ok(other) => seek_problems.push(format!(
                    "field {field_name:?}: seekCeil({sample:?}) returned {other:?}, not FOUND"
                )),
                Err(e) => seek_problems.push(format!(
                    "field {field_name:?}: seekCeil({sample:?}) failed: {e}"
                )),
            }
        }
        // Java's dedicated "seek to last term" case. Its accompanying
        // `docFreq` recount is *not* ported: `postings()` is
        // parameterized by the claimed `docFreq` (see this module's top
        // comment), so `docs.len() == doc_freq` whenever the decode
        // succeeds at all and the comparison could never disagree.
        if let Some(last) = &prev_term {
            match field_terms.try_seek_exact(last) {
                Ok(Some(_)) => {}
                Ok(None) => seek_problems.push(format!(
                    "field {field_name:?}: seek to last term {last:?} failed"
                )),
                Err(e) => seek_problems.push(format!(
                    "field {field_name:?}: seek to last term {last:?} failed: {e}"
                )),
            }
        }
    }
    checks.push(named_field_check(
        &format!("postings.seek_agrees:{field_name}"),
        &seek_problems,
        seek_samples.len() as i64,
        "sampled terms",
    ));
}

// `checkTermsIntersect`: the pruning term walker must return exactly
// the terms a linear scan filtered by the same matcher returns, in
// the same order. Compared in lockstep rather than by collecting
// both sides: `.*[a-e].*` matches most natural-language terms, so
// materializing them would be close to a second copy of the whole
// dictionary.
//
// The automaton rounds' scans step a DFA a byte at a time, as Java's
// `ByteRunAutomaton.run` does -- never `RegexpPattern::matches`,
// whose backtracker recurses once per code point consumed and
// overflowed a 2 MB thread stack on a binary term of a few thousand
// valid UTF-8 code points (an ICU collation key). The glob's
// `WildcardPattern::matches` recurses per pattern token, not per
// term byte.
fn check_intersect_round(
    checks: &mut Vec<Check>,
    field_name: &str,
    field_terms: &blocktree::FieldTerms,
    intersect_automata: &IntersectAutomata,
) {
    let mut intersect_problems: Vec<String> = Vec::new();
    // Java's own round (the first of its four automaton/start-term
    // combinations): `makeAnyBinary() [a-e] makeAnyBinary()` over raw
    // bytes, so binary terms take part too.
    if let Some((compiled, run)) = &intersect_automata.any_binary {
        compare_intersect_with_scan(
            field_name,
            field_terms,
            "automaton `anyBinary [a-e] anyBinary`",
            |t| lucene_util::automaton::ByteRunnable::run(run, t),
            field_terms
                .compiled_terms(compiled)
                .map(|r| r.map(|(t, s)| (t, s.stats))),
            &mut intersect_problems,
        );
    } else {
        // Java throws here; a round that cannot run must not pass.
        intersect_problems.push(format!(
            "field {field_name:?}: `anyBinary [a-e] anyBinary` did not compile"
        ));
    }
    // The regexp walker this port adds (`RegexpQuery`'s), over the same
    // DFA it compiles to.
    if let Some((pattern, dfa)) = &intersect_automata.regexp {
        compare_intersect_with_scan(
            field_name,
            field_terms,
            "regexp `.*[a-e].*`",
            |t| dfa.run(t),
            field_terms.regexp_intersect(pattern),
            &mut intersect_problems,
        );
    } else {
        intersect_problems.push(format!(
            "field {field_name:?}: regexp `.*[a-e].*` did not compile to a DFA"
        ));
    }
    {
        let pattern = lucene_codecs::wildcard::WildcardPattern::new(b"*e*");
        compare_intersect_with_scan(
            field_name,
            field_terms,
            "wildcard `*e*`",
            |t| pattern.matches(t),
            field_terms.intersect(&pattern),
            &mut intersect_problems,
        );
    }
    checks.push(named_field_check(
        &format!("postings.intersect_agrees:{field_name}"),
        &intersect_problems,
        field_terms.num_terms,
        "terms",
    ));
}

/// One named check per kind of problem the term walk collected.
fn push_term_problem_checks(
    checks: &mut Vec<Check>,
    field_name: &str,
    field_terms: &blocktree::FieldTerms,
    (has_positions, has_offsets): (bool, bool),
    problems: &TermProblems,
) {
    let TermProblems {
        freq_mismatches,
        doc_id_problems,
        order_problems,
        doc_freq_problems,
        term_stat_problems,
        position_problems,
        offset_problems,
        advance_problems,
        run_problems,
        impact_problems,
    } = problems;
    for (name, problems) in [
        ("postings.total_term_freq", freq_mismatches),
        ("postings.doc_ids_valid", doc_id_problems),
        ("postings.terms_sorted", order_problems),
        ("postings.doc_freq_positive", doc_freq_problems),
        ("postings.term_stats", term_stat_problems),
        ("postings.advance_agrees", advance_problems),
        ("postings.doc_id_runs", run_problems),
        ("postings.impacts", impact_problems),
    ] {
        checks.push(named_field_check(
            &format!("{name}:{field_name}"),
            problems,
            field_terms.num_terms,
            "terms",
        ));
    }
    if has_positions {
        checks.push(named_field_check(
            &format!("postings.positions_valid:{field_name}"),
            position_problems,
            field_terms.num_terms,
            "terms",
        ));
    }
    if has_offsets {
        checks.push(named_field_check(
            &format!("postings.offsets_valid:{field_name}"),
            offset_problems,
            field_terms.num_terms,
            "terms",
        ));
    }
}

// Field-level totals. These come from `.tmd` and are what a
// scorer's IDF/normalisation reads.
fn check_field_summary(
    checks: &mut Vec<Check>,
    field_name: &str,
    field_terms: &blocktree::FieldTerms,
    fi: &field_infos::FieldInfo,
    outcome: TermWalkOutcome,
    docs_with_terms: &mut DocsWithTerms,
) {
    let TermWalkOutcome {
        needs_doc_file,
        prev_term,
        first_term,
        counted_terms,
        summed_doc_freq,
        summed_total_term_freq,
        visited_docs,
        ..
    } = outcome;
    let mut summary_problems: Vec<String> = Vec::new();
    if counted_terms != field_terms.num_terms {
        summary_problems.push(format!(
            "field {field_name:?}: .tmd claims numTerms={} but the dictionary enumerates \
                 {counted_terms}",
            field_terms.num_terms
        ));
    }
    if summed_doc_freq != field_terms.sum_doc_freq {
        summary_problems.push(format!(
            "field {field_name:?}: .tmd claims sumDocFreq={} but the terms' docFreqs sum \
                 to {summed_doc_freq}",
            field_terms.sum_doc_freq
        ));
    }
    if summed_total_term_freq != field_terms.sum_total_term_freq {
        summary_problems.push(format!(
            "field {field_name:?}: .tmd claims sumTotalTermFreq={} but the terms' \
                 totalTermFreqs sum to {summed_total_term_freq}",
            field_terms.sum_total_term_freq
        ));
    }
    if let Some(first) = &first_term {
        if first.as_slice() != field_terms.min_term.as_slice() {
            summary_problems.push(format!(
                "field {field_name:?}: .tmd claims minTerm={:?} but the first enumerated \
                     term is {first:?}",
                field_terms.min_term
            ));
        }
    }
    if let Some(last) = &prev_term {
        if last.as_slice() != field_terms.max_term.as_slice() {
            summary_problems.push(format!(
                "field {field_name:?}: .tmd claims maxTerm={:?} but the last enumerated \
                     term is {last:?}",
                field_terms.max_term
            ));
        }
    }
    if !needs_doc_file {
        // `visited_docs` is only complete when every term's postings
        // were actually decoded.
        let visited = visited_docs.cardinality() as i32;
        if visited != field_terms.doc_count {
            summary_problems.push(format!(
                "field {field_name:?}: .tmd claims docCount={} but {visited} distinct doc \
                     IDs appear in its postings",
                field_terms.doc_count
            ));
        }
        // Only the norms cross-check reads these, and only for a field
        // with norms: keeping a `maxDoc`-bit set for every other indexed
        // field until then was dead weight.
        if !fi.omit_norms && fi.index_options != field_infos::IndexOptions::None {
            docs_with_terms.insert(field_name.to_string(), visited_docs);
        }
    }
    checks.push(named_field_check(
        &format!("postings.field_summary:{field_name}"),
        &summary_problems,
        field_terms.num_terms,
        "terms",
    ));
}
