#![cfg(test)]
#[test]
fn utf16_map_agrees_with_the_scans_at_every_offset() {
    for text in [
        "",
        "plain ascii",
        "a\u{e9}\u{1F600}b \u{4e2d}\u{6587}",
        "\u{1F600}\u{1F600}",
    ] {
        let map = Utf16Map::new(text);
        assert_eq!(map.total_utf16, utf16_len(text));
        for u in 0..=utf16_len(text) + 2 {
            assert_eq!(
                map.to_byte(u),
                utf16_offset_to_byte(text, u),
                "{text:?} unit {u}"
            );
        }
        for (b, _) in text.char_indices().chain([(text.len(), ' ')]) {
            assert_eq!(
                map.to_utf16(b),
                byte_offset_to_utf16(text, b),
                "{text:?} byte {b}"
            );
        }
    }
}

#[test]
fn whitespace_snaps_with_memory_agree_with_the_scans() {
    let text = "\u{4e2d}\u{6587}\u{6587} ab  cd\u{6587}\u{6587} e";
    let offsets: Vec<usize> = text
        .char_indices()
        .map(|(b, _)| b)
        .chain([text.len()])
        .collect();
    let mut start = SnapStart::default();
    let mut end = SnapEnd::default();
    // Ascending, then a step back, as cluster windows can do.
    for &b in offsets.iter().chain(offsets.iter().rev().take(3)) {
        assert_eq!(
            start.snap(text, b),
            snap_start_to_whitespace(text, b),
            "start {b}"
        );
        assert_eq!(
            end.snap(text, b),
            snap_end_to_whitespace(text, b),
            "end {b}"
        );
    }
}

use super::*;

fn span(term: &str, start: i32, end: i32) -> TermOffsetSpan {
    TermOffsetSpan {
        term: term.to_string(),
        start_offset: start,
        end_offset: end,
    }
}

fn cfg(window_chars: usize, max_fragments: usize) -> FragmentConfig {
    FragmentConfig {
        window_chars,
        max_fragments,
        ..FragmentConfig::default()
    }
}

fn sentence_cfg(window_chars: usize, max_fragments: usize) -> FragmentConfig {
    FragmentConfig {
        snap_to_sentence: true,
        ..cfg(window_chars, max_fragments)
    }
}

// --- `PassageScorer` (real Lucene's exact formulas) ---

#[test]
fn passage_scorer_defaults_are_javas() {
    let s = PassageScorer::default();
    assert_eq!(s.k1, 1.2);
    assert_eq!(s.b, 0.75);
    assert_eq!(s.pivot, 87.0);
}

#[test]
fn passage_scorer_weight_tf_and_norm_match_the_java_formulas() {
    let s = PassageScorer::default();

    // weight(contentLength, totalTermFreq)
    //   = (k1 + 1) * ln(1 + (1 + contentLength/pivot + 0.5) / (ttf + 0.5))
    let content_length = 174usize; // exactly 2 pivots
    let ttf = 3usize;
    let num_docs = 1.0 + content_length as f32 / 87.0;
    let expected_weight =
        (1.2f32 + 1.0) * ((1.0 + (num_docs as f64 + 0.5) / (ttf as f64 + 0.5)).ln()) as f32;
    assert_eq!(s.weight(content_length, ttf), expected_weight);
    // Sanity: 174/87 == 2, so numDocs == 3.
    assert_eq!(num_docs, 3.0);

    // tf(freq, passageLen) = freq / (freq + k1 * ((1 - b) + b * len/pivot))
    let expected_tf = 2.0f32 / (2.0 + 1.2 * ((1.0 - 0.75) + 0.75 * (87.0 / 87.0)));
    assert_eq!(s.tf(2, 87), expected_tf);
    // At exactly one pivot the length factor collapses to k1 itself.
    assert_eq!(expected_tf, 2.0 / (2.0 + 1.2));

    // tf saturates: doubling freq must not double tf.
    assert!(s.tf(4, 87) < 2.0 * s.tf(2, 87));
    assert!(s.tf(4, 87) > s.tf(2, 87));
    // A longer passage is penalized.
    assert!(s.tf(2, 200) < s.tf(2, 20));

    // norm(passageStart) = 1 + 1/ln(pivot + start): earlier is better.
    assert_eq!(s.norm(0), 1.0 + 1.0 / (87.0f64.ln()) as f32);
    assert!(s.norm(0) > s.norm(500));
}

#[test]
fn passage_scorer_score_is_norm_times_the_per_term_sum() {
    let s = PassageScorer::default();
    let term_freqs = [(2usize, 5usize), (1, 9)];
    let (start, len, content) = (10usize, 60usize, 400usize);
    let expected = ((s.tf(2, len) * s.weight(content, 5)) as f64
        + (s.tf(1, len) * s.weight(content, 9)) as f64)
        * s.norm(start) as f64;
    assert_eq!(s.score(&term_freqs, start, len, content), expected as f32);
}

#[test]
fn max_fragments_keeps_the_highest_scoring_passages_not_the_first_ones() {
    // Three well-separated clusters: the first has one match, the last
    // has three of the same term packed together. Real Lucene keeps the
    // best `maxPassages` by `PassageScorer` and only then re-orders them
    // by offset -- so capping at 1 must keep the *dense* late cluster,
    // not the sparse early one.
    let pad = " pad".repeat(40);
    let mut text = String::from("cat");
    let mut spans = vec![span("cat", 0, 3)];
    text.push_str(&pad);
    text.push_str(&pad);
    let dense_start = text.chars().count();
    text.push_str("cat cat cat");
    for i in 0..3 {
        let s = (dense_start + i * 4) as i32;
        spans.push(span("cat", s, s + 3));
    }

    let all = assemble_fragments(&text, &spans, &cfg(5, 10));
    assert_eq!(all.len(), 2, "the two clusters must not merge");
    assert!(
        all[1].score > all[0].score,
        "the dense cluster must score higher: {} vs {}",
        all[1].score,
        all[0].score
    );

    let capped = assemble_fragments(&text, &spans, &cfg(5, 1));
    assert_eq!(capped.len(), 1);
    assert_eq!(
        capped[0].start_offset, all[1].start_offset,
        "capping must keep the best-scoring passage, not the first one"
    );
}

#[test]
fn fragments_are_returned_in_document_order_after_score_selection() {
    // Even when selection is by score, the surviving fragments come back
    // in document order (`Arrays.sort(passages, passageSortComparator)`).
    let pad = " pad".repeat(40);
    let mut text = String::new();
    let mut spans = Vec::new();
    for _ in 0..4 {
        let start = text.chars().count() as i32;
        text.push_str("cat");
        spans.push(span("cat", start, start + 3));
        text.push_str(&pad);
    }
    let out = assemble_fragments(&text, &spans, &cfg(5, 3));
    assert_eq!(out.len(), 3);
    assert!(out
        .windows(2)
        .all(|w| w[0].start_offset < w[1].start_offset));
}

#[test]
fn fragment_offsets_are_utf16_offsets_into_the_original_text() {
    // 'é' is two UTF-8 bytes but one UTF-16 code unit; the reported
    // offsets must be in the same unit `TermOffsetSpan` uses, not bytes.
    let text = "ééé cat ééé";
    let out = assemble_fragments(text, &[span("cat", 4, 7)], &cfg(100, 5));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].start_offset, 0);
    assert_eq!(out[0].end_offset, utf16_len(text));
}

// --- `DefaultPassageFormatter` ---

#[test]
fn format_fragments_joins_non_contiguous_passages_with_an_ellipsis() {
    let pad = " pad".repeat(30);
    let mut text = String::from("cat");
    let mut spans = vec![span("cat", 0, 3)];
    text.push_str(&pad);
    let second = text.chars().count() as i32;
    text.push_str("dog");
    spans.push(span("dog", second, second + 3));

    let config = cfg(5, 5);
    let fragments = assemble_fragments(&text, &spans, &config);
    assert_eq!(fragments.len(), 2);
    let formatted = format_fragments(&fragments, &config);
    assert!(
        formatted.contains("... "),
        "non-contiguous passages join with the default ellipsis: {formatted}"
    );
    assert!(formatted.contains("<b>cat</b>") && formatted.contains("<b>dog</b>"));
}

#[test]
fn format_fragments_of_a_single_passage_has_no_ellipsis() {
    let config = cfg(20, 5);
    let fragments = assemble_fragments("cat car cat", &[span("cat", 0, 3)], &config);
    let formatted = format_fragments(&fragments, &config);
    assert!(!formatted.contains("..."));
    assert_eq!(formatted, fragments[0].text);
}

#[test]
fn format_fragments_of_nothing_is_the_empty_string() {
    assert_eq!(format_fragments(&[], &FragmentConfig::default()), "");
}

#[test]
fn escape_html_escapes_content_but_never_the_markers() {
    let text = "a <b>&\"'/ cat";
    let config = FragmentConfig {
        escape: true,
        ..cfg(100, 5)
    };
    let start = text.chars().count() as i32 - 3;
    let out = assemble_fragments(text, &[span("cat", start, start + 3)], &config);
    assert_eq!(out.len(), 1);
    // Java's `DefaultPassageFormatter.append` escapes exactly these six.
    assert!(out[0]
        .text
        .starts_with("a &lt;b&gt;&amp;&quot;&#x27;&#x2F; "));
    // The markers themselves are emitted raw.
    assert!(out[0].text.ends_with("<b>cat</b>"));
}

#[test]
fn overlapping_matches_coalesce_into_one_marked_span() {
    // "cat" and "cats" both start at 0; Java's formatter merges any run
    // of overlapping matches into a single pre/post-wrapped span rather
    // than nesting the markers.
    let text = "cats sleep";
    let spans = [span("cat", 0, 3), span("cats", 0, 4)];
    let out = assemble_fragments(text, &spans, &cfg(100, 5));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].text, "<b>cats</b> sleep");
    assert_eq!(
        out[0].matched_terms,
        vec!["cat".to_string(), "cats".to_string()]
    );
}

#[test]
fn partially_overlapping_matches_extend_to_the_furthest_end() {
    let text = "abcdef ghi";
    // 0..4 and 2..6 overlap; the merged span must cover 0..6.
    let spans = [span("abcd", 0, 4), span("cdef", 2, 6)];
    let out = assemble_fragments(text, &spans, &cfg(100, 5));
    assert_eq!(out[0].text, "<b>abcdef</b> ghi");
}

#[test]
fn empty_spans_yields_no_fragments() {
    let out = assemble_fragments("the quick brown fox", &[], &cfg(10, 5));
    assert!(out.is_empty());
}

#[test]
fn empty_text_yields_no_fragments() {
    let out = assemble_fragments("", &[span("fox", 0, 3)], &cfg(10, 5));
    assert!(out.is_empty());
}

#[test]
fn single_match_produces_one_windowed_highlighted_fragment() {
    let text = "the quick brown fox jumps over the lazy dog near the river bank today";
    // "fox" is at char offset 16..19.
    let spans = [span("fox", 16, 19)];
    let out = assemble_fragments(text, &spans, &cfg(10, 5));
    assert_eq!(out.len(), 1);
    assert!(out[0].text.contains("<b>fox</b>"));
    assert_eq!(out[0].matched_terms, vec!["fox".to_string()]);
    // Window-snapping keeps whole words: no marker artifacts split
    // mid-word, and the fragment is shorter than the full text.
    assert!(out[0].text.len() < text.len());
}

#[test]
fn two_nearby_matches_merge_into_one_fragment_with_both_highlighted() {
    // Two matches 8 chars apart with a 10-char window each side --
    // their raw windows overlap, so they must merge into one fragment,
    // and marker insertion for the second (rightward) match must not
    // corrupt the first (leftward) match's already-inserted markers.
    let text = "alpha cat runs beta car stops gamma delta epsilon zeta";
    //           0     6   10          15  20
    // "cat" at 6..9, "car" at 20..23 (14 chars apart).
    let cat_start = text.find("cat").unwrap() as i32;
    let car_start = text.find("car").unwrap() as i32;
    let spans = [
        span("cat", cat_start, cat_start + 3),
        span("car", car_start, car_start + 3),
    ];
    let out = assemble_fragments(text, &spans, &cfg(20, 5));
    assert_eq!(out.len(), 1, "nearby matches must merge into one fragment");
    assert!(out[0].text.contains("<b>cat</b>"));
    assert!(out[0].text.contains("<b>car</b>"));
    assert_eq!(
        out[0].matched_terms,
        vec!["cat".to_string(), "car".to_string()]
    );
    // Both original words must still be intact (no off-by-one marker
    // corruption from inserting the second match's markers first).
    let unmarked = out[0].text.replace("<b>", "").replace("</b>", "");
    assert!(unmarked.contains("cat"));
    assert!(unmarked.contains("car"));
}

#[test]
fn two_far_apart_matches_produce_two_separate_fragments() {
    let mut text = String::from("cat ");
    text.push_str(&"filler ".repeat(30));
    text.push_str("car");
    let cat_start = 0i32;
    let car_start = text.rfind("car").unwrap() as i32;
    let spans = [
        span("cat", cat_start, cat_start + 3),
        span("car", car_start, car_start + 3),
    ];
    let out = assemble_fragments(&text, &spans, &cfg(10, 5));
    assert_eq!(out.len(), 2, "far-apart matches must not merge");
    assert!(out[0].text.contains("<b>cat</b>"));
    assert!(out[1].text.contains("<b>car</b>"));
}

#[test]
fn window_clamps_at_text_start_and_end_without_panicking() {
    let text = "cat";
    let spans = [span("cat", 0, 3)];
    let out = assemble_fragments(text, &spans, &cfg(50, 5));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].text, "<b>cat</b>");
}

#[test]
fn match_at_very_start_and_very_end_of_longer_text_does_not_panic() {
    let text = "cat is here and the last word is car";
    let cat_span = span("cat", 0, 3);
    let car_start = text.rfind("car").unwrap() as i32;
    let car_span = span("car", car_start, car_start + 3);
    let out = assemble_fragments(text, &[cat_span, car_span], &cfg(3, 5));
    // Small window keeps them from merging; neither slice should panic
    // nor run off either end of `text`.
    assert!(!out.is_empty());
    for f in &out {
        assert!(!f.text.is_empty());
    }
}

#[test]
fn max_fragments_truncates_extra_clusters() {
    // Five widely-separated matches, each far enough apart not to
    // merge, but capped to 2 fragments.
    let mut text = String::new();
    let mut spans = Vec::new();
    for _ in 0..5 {
        text.push_str("cat ");
        let start = text.len() as i32 - 4;
        spans.push(span("cat", start, start + 3));
        text.push_str(&"pad ".repeat(20));
    }
    let out = assemble_fragments(&text, &spans, &cfg(5, 2));
    assert_eq!(out.len(), 2);
}

/// Window arithmetic must add
/// `window_chars` (a CHAR count) to a CHAR offset, not a byte offset --
/// mixing the two units can push the raw window boundary to land in the
/// middle of a multi-byte character whenever one falls within
/// `window_chars` of a match, which panics on slicing. This text places
/// a 2-byte character ('é') exactly one char before the match, with a
/// window small enough that byte/char-count-mixed arithmetic would land
/// the window boundary inside that character's second byte.
#[test]
fn window_arithmetic_near_a_multi_byte_char_does_not_panic() {
    let text = "é match here";
    // "match" is at char offset 2..7 (é=0, space=1, m=2).
    let spans = [span("match", 2, 7)];
    let out = assemble_fragments(text, &spans, &cfg(1, 5));
    assert_eq!(out.len(), 1);
    assert!(out[0].text.contains("<b>match</b>"));
}

#[test]
fn multi_byte_utf8_match_does_not_panic_and_highlights_correctly() {
    // "café" -- 'é' is a 2-byte UTF-8 char but one Unicode scalar, at
    // char offset 3 (c-a-f-é), covering char offsets 0..4.
    let text = "café bar café bar café shop is nearby in the city center today for sure";
    let spans = [span("café", 0, 4)];
    let out = assemble_fragments(text, &spans, &cfg(5, 5));
    assert_eq!(out.len(), 1);
    assert!(out[0].text.contains("<b>café</b>"));
}

#[test]
fn out_of_range_and_invalid_spans_are_dropped_not_panicking() {
    let text = "cat dog";
    let spans = [
        span("bad", 100, 200), // entirely out of range
        span("bad2", 5, 2),    // end before start
        span("cat", 0, 3),     // valid
    ];
    let out = assemble_fragments(text, &spans, &cfg(10, 5));
    assert_eq!(out.len(), 1);
    assert!(out[0].text.contains("<b>cat</b>"));
}

// Real-fixture-composed test: genuine offsets from the
// checked-in Java-written fixture (`fixtures/data/term_vectors_index/`,
// generated by `fixtures/src/GenTermVectors.java`), composed with the
// REAL text those offsets describe. `GenTermVectors.java`'s doc 0
// "text" field is built from a `CannedTokenStream` of three tokens --
// "cat" at char offsets 0..3, "car" at 4..7, "cat" at 8..11 -- which
// describes the literal text "cat car cat" (space at offset 3, space
// at offset 7). This is not a made-up string: it is exactly what those
// real, differentially-verified offsets denote.
#[test]
fn real_fixture_offsets_composed_with_their_real_field_text() {
    let full_text = "cat car cat";
    let spans = [span("cat", 0, 3), span("car", 4, 7), span("cat", 8, 11)];
    let out = assemble_fragments(full_text, &spans, &cfg(20, 5));
    // All three matches are within a 20-char window of each other, so
    // they merge into a single fragment spanning the whole text.
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].text, "<b>cat</b> <b>car</b> <b>cat</b>");
    assert_eq!(
        out[0].matched_terms,
        vec!["cat".to_string(), "car".to_string()]
    );
}

// --- Sentence-boundary snapping (`snap_to_sentence: true`) ---

#[test]
fn sentence_snap_changes_output_vs_naive_fixed_window() {
    // A sentence boundary sits inside what a fixed 15-char window would
    // otherwise include: the naive window spills into the next
    // sentence's leading words, while sentence-snap mode must stop at
    // the sentence containing the match instead.
    let text = "Cats are great pets. Dogs are loyal companions too.";
    let start = text.find("great").unwrap() as i32;
    let end = start + "great".len() as i32;
    let spans = [span("great", start, end)];

    let naive = assemble_fragments(text, &spans, &cfg(15, 5));
    let snapped = assemble_fragments(text, &spans, &sentence_cfg(15, 5));

    assert_eq!(naive.len(), 1);
    assert_eq!(snapped.len(), 1);
    assert_ne!(
        naive[0].text, snapped[0].text,
        "sentence snapping must actually change the fragment boundaries"
    );
    // Sentence-snap keeps the whole first sentence, not a fragment of
    // the second one.
    assert!(snapped[0].text.starts_with("Cats are"));
    assert!(snapped[0].text.ends_with('.'));
    assert!(!snapped[0].text.contains("Dogs"));
}

#[test]
fn sentence_snap_with_no_terminators_still_produces_whole_text_fragment() {
    let text = "no sentence terminators here just plain running text forever";
    let start = text.find("running").unwrap() as i32;
    let end = start + "running".len() as i32;
    let spans = [span("running", start, end)];

    let out = assemble_fragments(text, &spans, &sentence_cfg(5, 5));
    assert_eq!(out.len(), 1);
    assert!(!out[0].text.is_empty());
    assert!(out[0].text.contains("<b>running</b>"));
    // No terminators at all -- the whole text is one "sentence".
    assert_eq!(out[0].text.replace("<b>", "").replace("</b>", ""), text);
}

// --- `FieldOffsetStrategy` selection -----------------------------------

#[test]
fn offset_source_follows_java_field_info_order() {
    use lucene_codecs::field_infos::IndexOptions;
    // Offsets in the postings win, and term vectors upgrade the answer.
    assert_eq!(
        offset_source_for_field(
            Some(IndexOptions::DocsAndFreqsAndPositionsAndOffsets),
            false
        ),
        OffsetSource::Postings
    );
    assert_eq!(
        offset_source_for_field(Some(IndexOptions::DocsAndFreqsAndPositionsAndOffsets), true),
        OffsetSource::PostingsWithTermVectors
    );
    // No offsets, but term vectors: Java takes them without checking they
    // carry offsets.
    assert_eq!(
        offset_source_for_field(Some(IndexOptions::DocsAndFreqsAndPositions), true),
        OffsetSource::TermVectors
    );
    assert_eq!(
        offset_source_for_field(Some(IndexOptions::Docs), true),
        OffsetSource::TermVectors
    );
    // Neither: re-analyze.
    assert_eq!(
        offset_source_for_field(Some(IndexOptions::DocsAndFreqs), false),
        OffsetSource::Analysis
    );
    // No FieldInfo at all is Java's `fieldInfo == null`: it never reaches
    // the `hasTermVectors()` test, so the "no FieldInfo but term vectors"
    // pair -- which cannot arise from a real reader anyway -- is ANALYSIS
    // too, not TERM_VECTORS.
    assert_eq!(offset_source_for_field(None, false), OffsetSource::Analysis);
    assert_eq!(offset_source_for_field(None, true), OffsetSource::Analysis);
}

#[test]
fn optimized_offset_source_follows_javas_switch() {
    // Nothing to highlight at all.
    assert_eq!(
        optimize_offset_source(OffsetSource::Postings, false, false),
        OffsetSource::NoneNeeded
    );
    // A multi-term part makes the postings source useless (it would mean
    // scanning the whole term dictionary), so Java downgrades to ANALYSIS.
    assert_eq!(
        optimize_offset_source(OffsetSource::Postings, true, true),
        OffsetSource::Analysis
    );
    assert_eq!(
        optimize_offset_source(OffsetSource::Postings, false, true),
        OffsetSource::Postings
    );
    // Without a multi-term part the term vectors are not needed.
    assert_eq!(
        optimize_offset_source(OffsetSource::PostingsWithTermVectors, false, true),
        OffsetSource::Postings
    );
    assert_eq!(
        optimize_offset_source(OffsetSource::PostingsWithTermVectors, true, true),
        OffsetSource::PostingsWithTermVectors
    );
    // Everything else passes through, including a multi-term query with
    // no plain terms (Java's `mtqOrRewrite` alone keeps a source alive).
    assert_eq!(
        optimize_offset_source(OffsetSource::TermVectors, true, false),
        OffsetSource::TermVectors
    );
    assert_eq!(
        optimize_offset_source(OffsetSource::Analysis, false, true),
        OffsetSource::Analysis
    );
    assert_eq!(
        optimize_offset_source(OffsetSource::NoneNeeded, true, true),
        OffsetSource::NoneNeeded
    );
}

#[test]
fn analysis_offset_strategy_keeps_only_matching_tokens_with_their_own_offsets() {
    let analyzer = lucene_analysis::Analyzer::standard(None);
    let content = "The quick brown fox jumps over the lazy dog";
    let spans = offsets_from_analysis(&analyzer, content, &["quick", "dog"]);
    let rendered: Vec<(&str, i32, i32)> = spans
        .iter()
        .map(|s| (s.term.as_str(), s.start_offset, s.end_offset))
        .collect();
    assert_eq!(
        rendered,
        vec![
            ("quick", 4, 9),
            ("dog", content.len() as i32 - 3, content.len() as i32),
        ]
    );
    // The offsets really do address the original text.
    for s in &spans {
        assert_eq!(
            &content[s.start_offset as usize..s.end_offset as usize],
            s.term
        );
    }
    // A term the analyzer never produces contributes nothing.
    assert!(offsets_from_analysis(&analyzer, content, &["unicorn"]).is_empty());
    assert!(offsets_from_analysis(&analyzer, "", &["quick"]).is_empty());
}

/// [`offsets_from_analysis`] passes the analyzer's offsets through
/// **unconverted**, because `lucene_analysis::Token` reports the same
/// UTF-16 code units this module works in.
///
/// A conversion here would shift every non-ASCII highlight, so this asserts the two sides are
/// identical -- not merely that the result happens to be right.
#[test]
fn analysis_offsets_pass_through_unconverted_in_java_char_units() {
    let analyzer = lucene_analysis::Analyzer::standard(None);
    // "café naïve" is 12 UTF-8 bytes but 10 UTF-16 code units, so a
    // byte-offset producer and a converting consumer are both visible.
    let content = "café naïve dog";
    let raw: Vec<(i32, i32)> = analyzer
        .analyze(content)
        .into_iter()
        .filter(|t| t.term == "dog")
        .map(|t| (t.start_offset, t.end_offset))
        .collect();
    assert_eq!(
        raw,
        vec![(11, 14)],
        "the analyzer reports Java `char` offsets, which is what \
             OffsetAttribute would have reported for the same token"
    );

    let spans = offsets_from_analysis(&analyzer, content, &["dog"]);
    assert_eq!(
        spans
            .iter()
            .map(|s| (s.start_offset, s.end_offset))
            .collect::<Vec<_>>(),
        raw,
        "offsets_from_analysis must not convert anything"
    );

    // ...and they highlight the right word.
    let fragments = assemble_fragments(
        content,
        &spans,
        &FragmentConfig {
            window_chars: 100,
            ..FragmentConfig::default()
        },
    );
    assert_eq!(fragments[0].text, "café naïve <b>dog</b>");

    // An astral character separates Java `char`s from Unicode scalars as
    // well as from bytes: "beta" is at char 9, scalar 8, byte 11.
    let astral = "alpha \u{1F600} beta";
    let beta = offsets_from_analysis(&analyzer, astral, &["beta"]);
    assert_eq!(
        beta.iter()
            .map(|s| (s.start_offset, s.end_offset))
            .collect::<Vec<_>>(),
        vec![(9, 13)]
    );
    assert_eq!(
        assemble_fragments(
            astral,
            &beta,
            &FragmentConfig {
                window_chars: 100,
                ..FragmentConfig::default()
            },
        )[0]
        .text,
        "alpha \u{1F600} <b>beta</b>"
    );
}

#[test]
fn utf16_offset_conversions_round_trip_and_clamp() {
    // "😀" is one Unicode scalar, 4 UTF-8 bytes, 2 UTF-16 code units.
    let text = "a😀b";
    assert_eq!(utf16_len(text), 4);
    assert_eq!(utf16_offset_to_byte(text, 0), 0);
    assert_eq!(utf16_offset_to_byte(text, 1), 1);
    // Inside the surrogate pair: rounds down to the emoji's own start.
    assert_eq!(utf16_offset_to_byte(text, 2), 1);
    assert_eq!(utf16_offset_to_byte(text, 3), 5);
    assert_eq!(utf16_offset_to_byte(text, 4), 6);
    // Past the end clamps rather than panicking.
    assert_eq!(utf16_offset_to_byte(text, 99), text.len());
    assert_eq!(byte_offset_to_utf16(text, 0), 0);
    assert_eq!(byte_offset_to_utf16(text, 1), 1);
    assert_eq!(byte_offset_to_utf16(text, 5), 3);
    assert_eq!(byte_offset_to_utf16(text, 6), 4);
    assert_eq!(byte_offset_to_utf16(text, 99), 4);
    assert_eq!(utf16_len(""), 0);
    assert_eq!(utf16_offset_to_byte("", 3), 0);
}

#[test]
fn analysis_offsets_come_back_in_offsets_enum_order() {
    let analyzer = lucene_analysis::Analyzer::standard(None);
    let content = "dog cat dog";
    let spans = offsets_from_analysis(&analyzer, content, &["dog", "cat"]);
    let starts: Vec<i32> = spans.iter().map(|s| s.start_offset).collect();
    assert_eq!(starts, vec![0, 4, 8]);
}

#[test]
fn sentence_snap_breaks_after_an_abbreviation_exactly_as_the_jdk_does() {
    // `BreakIterator.getSentenceInstance(Locale.ROOT)` --
    // and `Locale.ENGLISH` -- splits this text as
    // `["Mr. ", "Smith arrived early. ", "He left before noon."]`, applying
    // no abbreviation suppression at all (verified by running the JDK; see
    // `sentence_boundaries_match_jdk_break_iterator`).
    let text = "Mr. Smith arrived early. He left before noon.";
    let start = text.find("Smith").unwrap() as i32;
    let end = start + "Smith".len() as i32;
    let spans = [span("Smith", start, end)];

    let out = assemble_fragments(text, &spans, &sentence_cfg(5, 5));
    assert_eq!(out.len(), 1);
    assert!(out[0].text.starts_with("<b>Smith</b>"));
    assert!(out[0].text.ends_with("early."));
    assert!(!out[0].text.contains("He left"));
}

/// The whole point of replacing the heuristic: the JDK's own answers,
/// read from `fixtures/data/break_iterator/manifest.properties` --
/// `BreakIterator.getSentenceInstance` run over each text by
/// `fixtures/src/GenBreakIterator.java` and regenerable through
/// `scripts/gen-fixtures.sh`, so a JDK or CLDR data change shows up as a
/// failing check rather than as passages silently drifting away from
/// Lucene's.
///
/// Compares the sliced *substrings*, not offsets: `BreakIterator` reports
/// UTF-16 offsets and this reports UTF-8 byte offsets, and every fixture
/// text is ASCII precisely so the comparison never has to bridge the two.
#[test]
fn sentence_boundaries_match_the_jdks_break_iterator() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/break_iterator/manifest.properties"
    ))
    .expect("run scripts/gen-fixtures.sh first (GenBreakIterator)");
    let get = |key: &str| -> String {
        text.lines()
            .find_map(|l| l.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
            .to_string()
    };
    // Inverse of `GenBreakIterator.escape`.
    let unescape = |s: &str| -> String {
        let mut out = String::new();
        let mut it = s.chars();
        while let Some(c) = it.next() {
            if c != '\\' {
                out.push(c);
                continue;
            }
            match it.next() {
                Some('\\') => out.push('\\'),
                Some('n') => out.push('\n'),
                Some('u') => {
                    let hex: String = (0..4).filter_map(|_| it.next()).collect();
                    out.push(char::from_u32(u32::from_str_radix(&hex, 16).unwrap()).unwrap());
                }
                other => panic!("unknown escape \\{other:?}"),
            }
        }
        out
    };
    let split = |s: &str| -> Vec<String> {
        if s.is_empty() {
            Vec::new()
        } else {
            s.split('\u{1}').map(&unescape).collect()
        }
    };

    let count: usize = get("count").parse().unwrap();
    assert!(count >= 10, "the fixture must actually carry texts");
    let mut checked_a_multi_sentence_case = false;
    for i in 0..count {
        let input = unescape(&get(&format!("text.{i}")));
        let want = split(&get(&format!("root.{i}")));
        // `UnifiedHighlighter`'s default is `Locale.ROOT`; the fixture
        // records `Locale.ENGLISH` too, and this port is untailored, so the
        // two must agree for the port to be right for either.
        assert_eq!(
            want,
            split(&get(&format!("english.{i}"))),
            "text {i}: ROOT and ENGLISH disagree, so an untailored port \
                 cannot match both"
        );

        let b = sentence_boundaries(&input);
        let got: Vec<String> = b
            .windows(2)
            .map(|w| input[w[0]..w[1]].to_string())
            .collect();
        assert_eq!(got, want, "text {i} ({input:?})");
        if want.len() > 1 {
            checked_a_multi_sentence_case = true;
        }
    }
    assert!(
        checked_a_multi_sentence_case,
        "a fixture where every text is one sentence proves nothing"
    );
}

/// Degenerate inputs the fixture cannot express as a boundary list.
#[test]
fn sentence_boundaries_of_degenerate_inputs() {
    assert_eq!(sentence_boundaries(""), vec![0]);
    // A text that is nothing but whitespace is one "sentence", not many.
    assert_eq!(sentence_boundaries("\n\n\n"), vec![0, 3]);
    assert_eq!(sentence_boundaries("   "), vec![0, 3]);
}

/// `SplittingBreakIterator(sentenceIterator, MULTIVAL_SEP_CHAR)`: the
/// splitting character is always a boundary on both sides, and the
/// enclosed iterator never sees it -- so a passage can never straddle two
/// values of a multi-valued field.
#[test]
fn split_sentence_boundaries_reports_every_slice_char_as_a_boundary() {
    let text = "One. Two.\u{0}Three. Four.";
    let b = split_sentence_boundaries(text, '\u{0}');
    let sep = text.find('\u{0}').unwrap();
    assert!(b.contains(&sep), "the separator itself is a boundary");
    assert!(b.contains(&(sep + 1)), "and so is the position after it");
    for w in b.windows(2) {
        assert!(
            !(w[0] < sep && w[1] > sep + 1),
            "a slice must not straddle the separator"
        );
    }
    // Adjacent separators, and separators at either end, still produce
    // boundaries rather than empty confusion.
    let edges = split_sentence_boundaries("\u{0}\u{0}a\u{0}", '\u{0}');
    assert_eq!(edges, vec![0, 1, 2, 3, 4]);
}

#[test]
fn sentence_snap_ordinal_number_is_not_mistaken_for_st_abbreviation() {
    // "st" is on ABBREVIATIONS (for "St." as in a street name), which
    // would incorrectly also suppress a sentence break after any
    // ordinal number ending in "st" (21st, 1st, 91st, ...) unless the
    // abbreviation check excludes an alphabetic run directly preceded
    // by a digit. Found in review: without that guard, this exact text
    // incorrectly merged into one fragment instead of breaking after
    // "21st.".
    let text = "He finished 21st. She started next.";
    let start = text.find("She").unwrap() as i32;
    let end = start + "She".len() as i32;
    let spans = [span("She", start, end)];

    let out = assemble_fragments(text, &spans, &sentence_cfg(5, 5));
    assert_eq!(out.len(), 1);
    assert!(out[0].text.starts_with("<b>She</b>"));
    assert!(!out[0].text.contains("21st"));
}

#[test]
fn sentence_snap_closing_quote_after_terminator_is_recognized() {
    // A terminator immediately followed by a closing quote, then
    // whitespace and an uppercase letter, must still be recognized as a
    // sentence break -- without quote-skipping, the char right after
    // the period is `"` (not whitespace), so the old heuristic would
    // fail to find the break here and spill into the quoted sentence.
    let text = "He said \"Stop.\" Then he left the room for good today.";
    let start = text.find("Then").unwrap() as i32;
    let end = start + "Then".len() as i32;
    let spans = [span("Then", start, end)];

    let out = assemble_fragments(text, &spans, &sentence_cfg(3, 5));
    assert_eq!(out.len(), 1);
    assert!(out[0].text.starts_with("<b>Then</b>"));
    assert!(!out[0].text.contains("Stop"));
}

#[test]
fn sentence_snap_closing_paren_after_terminator_is_recognized() {
    let text = "(See note.) Next sentence begins here and continues on.";
    let start = text.find("Next").unwrap() as i32;
    let end = start + "Next".len() as i32;
    let spans = [span("Next", start, end)];

    let out = assemble_fragments(text, &spans, &sentence_cfg(3, 5));
    assert_eq!(out.len(), 1);
    assert!(out[0].text.starts_with("<b>Next</b>"));
    assert!(!out[0].text.contains("See note"));
}

#[test]
fn sentence_snap_match_at_very_start_and_very_end_does_not_panic() {
    let text = "First sentence here. Middle sentence stands alone. Last sentence ends.";
    let first_word_end = "First".len() as i32;
    let last_word_start = text.rfind("Last").unwrap() as i32;
    let last_word_end = last_word_start + "Last".len() as i32;
    let spans = [
        span("First", 0, first_word_end),
        span("Last", last_word_start, last_word_end),
    ];

    let out = assemble_fragments(text, &spans, &sentence_cfg(3, 5));
    assert!(!out.is_empty());
    for f in &out {
        assert!(!f.text.is_empty());
    }
    // The very-first match's fragment must start right at the text's
    // start (byte offset 0), not run off the front.
    let first_fragment = out
        .iter()
        .find(|f| f.text.contains("<b>First</b>"))
        .expect("a fragment containing the first match");
    assert!(first_fragment.text.starts_with("<b>First</b>"));
    // The very-last match's fragment must end at (or before) the text's
    // end, not run off the back.
    let last_fragment = out
        .iter()
        .find(|f| f.text.contains("<b>Last</b>"))
        .expect("a fragment containing the last match");
    assert!(last_fragment.text.ends_with('.'));
}

#[test]
fn sentence_snap_re_merges_clusters_that_expand_into_the_same_sentence() {
    // A small window_chars keeps these two matches in separate clusters
    // under the fixed-window sweep (they're far apart), but they both
    // fall inside the same long sentence -- sentence-snapping expands
    // both clusters' windows to that whole sentence. Without a re-merge
    // pass after snapping, this produced two separate, nearly-identical,
    // overlapping fragments instead of one fragment with both matches
    // highlighted.
    let text = "One two three four five six seven eight nine ten eleven twelve \
                     thirteen fourteen fifteen sixteen. Next sentence word.";
    let one_start = text.find("One").unwrap() as i32;
    let one_end = one_start + "One".len() as i32;
    let sixteen_start = text.find("sixteen").unwrap() as i32;
    let sixteen_end = sixteen_start + "sixteen".len() as i32;
    let spans = [
        span("One", one_start, one_end),
        span("sixteen", sixteen_start, sixteen_end),
    ];

    let out = assemble_fragments(text, &spans, &sentence_cfg(3, 5));
    assert_eq!(
        out.len(),
        1,
        "expected the two same-sentence clusters to merge into one fragment, got: {out:?}"
    );
    assert!(out[0].text.contains("<b>One</b>"));
    assert!(out[0].text.contains("<b>sixteen</b>"));
}

#[test]
fn sentence_snap_lowercase_after_period_is_not_a_terminator() {
    // "3.5" -- a period followed by a lowercase/digit character is not
    // this heuristic's terminator (it requires whitespace then an
    // uppercase letter, or end-of-text), so the fragment must not break
    // there.
    let text = "The price is 3.5 and rising steadily today.";
    let start = text.find("3.5").unwrap() as i32;
    let end = start + "3.5".len() as i32;
    let spans = [span("3.5", start, end)];

    let out = assemble_fragments(text, &spans, &sentence_cfg(3, 5));
    assert_eq!(out.len(), 1);
    assert!(out[0].text.starts_with("The price is"));
    assert!(out[0].text.ends_with("today."));
}

#[test]
fn sentence_snap_consecutive_terminators_do_not_panic() {
    // "Really?!" and "Wow..." both have runs of terminator characters;
    // the heuristic must not double-count them or panic walking past
    // the run.
    let text = "Really?! Wow... That is surprising indeed today.";
    let start = text.find("Wow").unwrap() as i32;
    let end = start + "Wow".len() as i32;
    let spans = [span("Wow", start, end)];

    let out = assemble_fragments(text, &spans, &sentence_cfg(3, 5));
    assert_eq!(out.len(), 1);
    assert!(out[0].text.contains("<b>Wow</b>"));
}

#[test]
fn sentence_snap_single_word_no_terminator_no_whitespace() {
    let text = "Hello";
    let spans = [span("Hello", 0, 5)];
    let out = assemble_fragments(text, &spans, &sentence_cfg(3, 5));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].text, "<b>Hello</b>");
}
