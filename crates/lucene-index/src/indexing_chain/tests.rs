#![cfg(test)]

/// Checked against Java: the first `GenPerFieldFormats` segment's
/// doc-values fields reach `PerFieldDocValuesFormat` as `dva_num` (4),
/// then the two `dvb_` fields, which is why `dva_num` is `Lucene90_0`.
/// `new HashMap<String, Object>()` with `bin`, `num` and `sorted` put in
/// any order iterates `sorted` (bucket 2), `bin` (6), `num` (7).
#[test]
fn java_hash_map_order_walks_buckets() {
    assert_eq!(super::java_hash_map_order(&["num", "bin"]), vec![1, 0]);
    assert_eq!(
        super::java_hash_map_order(&["sorted", "num", "bin"]),
        vec![0, 2, 1]
    );
    let many: Vec<String> = (0..13).map(|i| format!("f{i}")).collect();
    let refs: Vec<&str> = many.iter().map(String::as_str).collect();
    assert_eq!(super::java_hash_map_order(&refs).len(), 13);
}

#[test]
fn field_hash_order_walks_javas_table() {
    assert_eq!(super::java_string_hash(""), 0);
    assert_eq!(super::java_string_hash("a"), 97);
    // "hello".hashCode() == 99162322, "polygenelubricants" wraps negative.
    assert_eq!(super::java_string_hash("hello"), 99_162_322);
    assert_eq!(super::java_string_hash("polygenelubricants"), i32::MIN);
    let names = [
        "a_id",
        "a_text",
        "b_id",
        "b_tag",
        "dva_num",
        "dvb_num",
        "dvb_sorted",
    ];
    let order = super::field_hash_order(&names);
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..names.len()).collect::<Vec<_>>());
    let dv: Vec<&str> = order
        .iter()
        .map(|&i| names[i])
        .filter(|n| n.starts_with("dv"))
        .collect();
    assert_eq!(dv[0], "dva_num", "{order:?}");
    assert!(super::field_hash_order(&[]).is_empty());
}
use super::*;
use std::collections::HashSet;

fn occ(position: i32, start: i32, end: i32) -> Occurrence {
    Occurrence {
        position,
        start_offset: start,
        end_offset: end,
    }
}

/// The offsets this module forwards into `.pos`/`.pay`/`.tvd` are
/// Java `char` indices, so a JVM reader that does
/// `text.substring(startOffset, endOffset)` gets the term back.
///
/// `invert_documents` does not convert anything -- it passes
/// [`lucene_analysis::Token`]'s offsets straight through -- so this test
/// is what makes that pass-through a checked claim rather than an assumed
/// one, over text where UTF-8 bytes, Unicode scalars and UTF-16 code units
/// all disagree.
/// Java's `PerField` owns its `FieldInvertState` for the whole document,
/// so two values of one field are one field however many *other* fields'
/// values sit between them (`processField` resets only when
/// `pf.fieldGen != fieldGen`). A grouping that only joined *consecutive*
/// tuples would restart the second value at position 0 and offset 0 --
/// the very defect the grouping exists to remove -- for an input this
/// function's own contract accepts.
#[test]
fn a_fields_values_are_one_field_even_with_another_fields_value_between_them() {
    let analyzer = Analyzer::standard(None).with_position_increment_gap(7);
    let interleaved = invert_documents(
        &[(0, "f", "alpha"), (0, "g", "zulu"), (0, "f", "beta")],
        &analyzer,
    );
    let adjacent = invert_documents(
        &[(0, "f", "alpha"), (0, "f", "beta"), (0, "g", "zulu")],
        &analyzer,
    );
    assert_eq!(interleaved.terms, adjacent.terms);

    // ... and the second value really is offset, not restarted: "alpha" is
    // 5 characters, the offset gap is 1, and the position gap is 7.
    let beta = interleaved.posting_list("f", "beta").expect("beta indexed");
    assert_eq!(beta.entries.len(), 1);
    assert_eq!(
        beta.entries[0].occurrences,
        vec![Occurrence {
            position: 8,
            start_offset: 6,
            end_offset: 10,
        }]
    );
    // The interposed field is unaffected by either gap.
    let zulu = interleaved.posting_list("g", "zulu").expect("zulu indexed");
    assert_eq!(
        zulu.entries[0].occurrences,
        vec![Occurrence {
            position: 0,
            start_offset: 0,
            end_offset: 4,
        }]
    );
}

#[test]
fn offsets_forwarded_to_the_codec_slice_the_source_text_as_java_chars() {
    let analyzer = Analyzer::standard(None);
    // "caf\u{E9}" is 4 chars / 5 bytes; "\u{4E16}" is 1 char / 3 bytes;
    // "\u{1D306}" is 2 chars / 1 scalar / 4 bytes.
    let text = "alpha caf\u{E9} \u{4E16} \u{1D306} omega";
    let docs = vec![(0, "body", text)];
    let index = invert_documents(&docs, &analyzer);

    // What a JVM caller does with an OffsetAttribute: index the text by
    // UTF-16 code unit.
    let units: Vec<u16> = text.encode_utf16().collect();
    let mut seen = 0;
    for ((_, term), list) in &index.terms {
        for entry in &list.entries {
            for occurrence in &entry.occurrences {
                let (start, end) = (occurrence.start_offset, occurrence.end_offset);
                assert!(start >= 0 && end >= start, "offsets out of order");
                let slice = String::from_utf16(&units[start as usize..end as usize])
                    .expect("a Java `char` span of the source text");
                assert_eq!(&slice, term, "offsets {start}..{end} do not name the term");
                seen += 1;
            }
        }
    }
    assert_eq!(seen, 4, "expected alpha/caf\u{E9}/\u{4E16}/omega");
    // A byte-offset producer would have put "omega" at 21, not 16.
    assert_eq!(text.len(), 26);
    // Position 3, not 4: `U+1D306` is a symbol, so neither real Lucene's
    // `StandardTokenizer` (see the `utf16_astral_symbol` fixture case,
    // whose two tokens are one position apart across it) nor this port
    // emits a token for it, and a skipped run of text consumes no
    // position. The unit fix does not touch positions -- which is exactly
    // why one is asserted here alongside the offsets.
    assert_eq!(
        index.postings("body", "omega").expect("omega")[0].occurrences[0],
        occ(3, 16, 21)
    );
}

#[test]
fn single_doc_single_field_inverts_correctly() {
    let analyzer = Analyzer::standard(None);
    let docs = vec![(0, "body", "the quick fox")];
    let index = invert_documents(&docs, &analyzer);

    assert_eq!(
        index.postings("body", "the"),
        Some(
            &[PostingEntry {
                doc_id: 0,
                occurrences: vec![occ(0, 0, 3)],
            }][..]
        )
    );
    assert_eq!(
        index.postings("body", "quick"),
        Some(
            &[PostingEntry {
                doc_id: 0,
                occurrences: vec![occ(1, 4, 9)],
            }][..]
        )
    );
    assert_eq!(
        index.postings("body", "fox"),
        Some(
            &[PostingEntry {
                doc_id: 0,
                occurrences: vec![occ(2, 10, 13)],
            }][..]
        )
    );
    assert_eq!(index.terms.len(), 3);
}

#[test]
fn multiple_docs_sharing_a_term_are_doc_id_sorted() {
    let analyzer = Analyzer::standard(None);
    let docs = vec![
        (0, "body", "fox"),
        (1, "body", "fox jumps"),
        (2, "body", "the fox runs"),
    ];
    let index = invert_documents(&docs, &analyzer);

    let postings = index.postings("body", "fox").unwrap();
    assert_eq!(postings.len(), 3);
    assert_eq!(postings[0].doc_id, 0);
    assert_eq!(postings[1].doc_id, 1);
    assert_eq!(postings[2].doc_id, 2);
}

/// The doc-ID-sorted invariant must hold even when `docs` is supplied
/// out of order -- `invert_documents` sorts each term's posting list
/// itself rather than trusting the caller to pre-sort. Passing input in
/// REVERSE doc-ID order is the strongest possible proof this isn't just
/// an artifact of already-ascending test fixtures.
#[test]
fn out_of_order_input_docs_still_produce_doc_id_sorted_postings() {
    let analyzer = Analyzer::standard(None);
    let docs = vec![
        (2, "body", "the fox runs"),
        (0, "body", "fox"),
        (1, "body", "fox jumps"),
    ];
    let index = invert_documents(&docs, &analyzer);

    let postings = index.postings("body", "fox").unwrap();
    assert_eq!(postings.len(), 3);
    assert_eq!(
        postings.iter().map(|e| e.doc_id).collect::<Vec<_>>(),
        vec![0, 1, 2],
        "postings must be doc-ID-sorted regardless of input order: {postings:?}"
    );
}

#[test]
fn repeated_term_in_one_doc_has_correct_freq_and_all_positions() {
    let analyzer = Analyzer::standard(None);
    // "fox" occurs at positions 0 and 3 (0-indexed: fox=0, saw=1,
    // another=2, fox=3).
    let docs = vec![(0, "body", "fox saw another fox")];
    let index = invert_documents(&docs, &analyzer);

    let postings = index.postings("body", "fox").unwrap();
    assert_eq!(postings.len(), 1);
    let entry = &postings[0];
    assert_eq!(entry.doc_id, 0);
    assert_eq!(entry.term_freq(), 2);
    assert_eq!(entry.positions(), vec![0, 3]);
    assert_eq!(entry.offsets(), vec![(0, 3), (16, 19)]);
}

#[test]
fn multiple_fields_on_same_doc_are_independent() {
    let analyzer = Analyzer::standard(None);
    let docs = vec![(0, "title", "fox"), (0, "body", "fox and hound")];
    let index = invert_documents(&docs, &analyzer);

    // Same term "fox" in two different fields must be two distinct
    // entries, not merged into one.
    assert_eq!(index.terms.len(), 4); // title/fox, body/fox, body/and, body/hound
    let title_fox = index.postings("title", "fox").unwrap();
    let body_fox = index.postings("body", "fox").unwrap();
    assert_eq!(title_fox.len(), 1);
    assert_eq!(body_fox.len(), 1);
    assert_eq!(title_fox[0].occurrences, vec![occ(0, 0, 3)]);
    assert_eq!(body_fox[0].occurrences, vec![occ(0, 0, 3)]);
    assert!(index.postings("title", "and").is_none());
}

#[test]
fn stopword_filtered_text_excludes_stopword_preserves_positions() {
    let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
    let analyzer = Analyzer::standard(Some(&stopwords));
    // "the quick fox": "the" removed, "quick" absorbs the skipped
    // position (position_increment 2), so "quick" lands at absolute
    // position 1 and "fox" at position 2 -- not 0/1, which would happen
    // if the stopword's position gap were silently dropped instead of
    // preserved.
    let docs = vec![(0, "body", "the quick fox")];
    let index = invert_documents(&docs, &analyzer);

    assert!(index.postings("body", "the").is_none());
    let quick = index.postings("body", "quick").unwrap();
    assert_eq!(quick[0].occurrences, vec![occ(1, 4, 9)]);
    let fox = index.postings("body", "fox").unwrap();
    assert_eq!(fox[0].occurrences, vec![occ(2, 10, 13)]);
}

/// The byte count has to be a *count*, not a constant: adding a document
/// with new terms must move it, and every component (keys, posting vectors,
/// occurrence vectors) must be represented. Asserted as a lower bound
/// derived from the content rather than an exact figure, since `Vec`
/// capacity growth is an allocator/`RawVec` detail.
#[test]
fn ram_bytes_used_counts_keys_postings_and_occurrences() {
    let analyzer = Analyzer::standard(None);
    let empty = invert_documents(&[], &analyzer);
    assert_eq!(
        empty.ram_bytes_used(),
        std::mem::size_of::<InMemoryInvertedIndex>()
    );

    let one = invert_documents(&[(0, "body", "alpha beta")], &analyzer);
    let two = invert_documents(
        &[(0, "body", "alpha beta"), (1, "body", "gamma delta")],
        &analyzer,
    );
    assert!(one.ram_bytes_used() > empty.ram_bytes_used());
    assert!(
        two.ram_bytes_used() > one.ram_bytes_used(),
        "four terms must cost more than two: {} vs {}",
        two.ram_bytes_used(),
        one.ram_bytes_used()
    );

    // A lower bound built purely from the content: two terms, each with a
    // "body" key (4 bytes) plus the term bytes, one posting entry and one
    // occurrence.
    let floor = 2
        * (std::mem::size_of::<(TermKey, Vec<PostingEntry>)>()
            + 4
            + 5
            + std::mem::size_of::<PostingEntry>()
            + std::mem::size_of::<Occurrence>());
    assert!(
        one.ram_bytes_used() >= floor,
        "{} < content floor {floor}",
        one.ram_bytes_used()
    );

    // Repeating a term costs another `Occurrence` but no new key, so the
    // figure never *drops* -- and does not necessarily rise, because a
    // `Vec`'s second element usually fits in the capacity its first
    // allocation already reserved. Counting capacity rather than length is
    // the point: the memory is occupied either way.
    let repeated = invert_documents(&[(0, "body", "alpha beta alpha")], &analyzer);
    assert!(repeated.ram_bytes_used() >= one.ram_bytes_used());
}

/// A `store_payloads` field gets one payload slot per occurrence, in
/// occurrence order, with a `None` from the source recorded as a
/// zero-length entry -- Lucene's `null`-`PayloadAttribute` equivalent.
/// The control is the second field in the same batch: a field not named
/// in `payload_fields` gets no slots at all, so "payloads are present"
/// cannot be something this function does unconditionally.
#[test]
fn payload_slots_are_filled_only_for_the_fields_that_declare_payloads() {
    let analyzer = Analyzer::standard(None);
    let docs = vec![(0, "body", "fox saw a fox"), (0, "title", "fox")];
    let index = invert_documents_with_payloads(&docs, &analyzer, &["body"], &|ctx| {
        // Every second position carries no payload, so the zero-length
        // case is exercised alongside the non-empty one.
        if ctx.position % 2 == 0 {
            Some(format!("{}:{}", ctx.term, ctx.position).into_bytes())
        } else {
            None
        }
    });

    let body = index.posting_list("body", "fox").unwrap();
    assert_eq!(body.entries.len(), 1);
    assert!(body.has_payloads());
    assert_eq!(body.entries[0].term_freq(), 2);
    assert_eq!(body.entries[0].positions(), vec![0, 3]);
    // Position 0 is even (payload), position 3 is odd (none).
    assert_eq!(body.payload_lengths, vec![5, 0]);
    assert_eq!(body.payload_bytes, b"fox:0".to_vec());
    assert_eq!(body.payload(0, 0), Some(&b"fox:0"[..]));
    assert_eq!(body.payload(0, 1), Some(&b""[..]));

    let title = index.posting_list("title", "fox").unwrap();
    assert!(!title.has_payloads());
    assert!(title.payload_lengths.is_empty());
    assert!(title.payload_bytes.is_empty());
    assert_eq!(title.payload(0, 0), None);
}

/// The context handed to a source is the whole token, not just its text:
/// a source keyed on the document or the offsets has to be able to see
/// them, and `field` is what lets one source serve several fields.
#[test]
fn the_payload_source_sees_the_field_document_position_and_offsets() {
    let analyzer = Analyzer::standard(None);
    let docs = vec![(7, "body", "the quick fox")];
    let seen = std::cell::RefCell::new(Vec::new());
    let index = invert_documents_with_payloads(&docs, &analyzer, &["body"], &|ctx| {
        seen.borrow_mut().push((
            ctx.field.to_string(),
            ctx.term.to_string(),
            ctx.doc_id,
            ctx.position,
            ctx.start_offset,
            ctx.end_offset,
        ));
        None
    });
    assert_eq!(
        seen.into_inner(),
        vec![
            ("body".into(), "the".into(), 7, 0, 0, 3),
            ("body".into(), "quick".into(), 7, 1, 4, 9),
            ("body".into(), "fox".into(), 7, 2, 10, 13),
        ]
    );
    // And the run is still there, one zero length per occurrence.
    let list = index.posting_list("body", "fox").unwrap();
    assert_eq!(list.payload_lengths, vec![0]);
    assert!(list.payload_bytes.is_empty());
}

/// `invert_documents` is the no-payloads shorthand, and must stay exactly
/// that: no slots, and a source that would panic if it were ever called.
#[test]
fn invert_documents_records_no_payload_slots_at_all() {
    let analyzer = Analyzer::standard(None);
    let index = invert_documents(&[(0, "body", "fox")], &analyzer);
    assert!(!index.posting_list("body", "fox").unwrap().has_payloads());
    // Same batch through the payload entry point with an empty field
    // list: identical output, which is what makes the delegation safe.
    let same = invert_documents_with_payloads(&[(0, "body", "fox")], &analyzer, &[], &|_| {
        panic!("a field not in payload_fields must never reach the source")
    });
    assert_eq!(index, same);
}

/// The allocation-shape assertion: a
/// [`PostingEntry`] carries **no payload slot at all**, so a payload
/// costs nothing per posting entry for the overwhelmingly common field
/// that has none. Against the shape this replaced -- a `Vec<Vec<u8>>` on
/// every entry -- this is 24 bytes smaller per entry on a 64-bit target,
/// which on `index-bench`'s corpus is 19 MB of the 27 MB the change
/// removed from `InMemoryInvertedIndex`.
///
/// Written against `size_of` rather than a byte count so it is a
/// statement about the type, and so it holds on a 32-bit target too.
#[test]
fn a_posting_entry_carries_no_payload_slot() {
    assert_eq!(
        std::mem::size_of::<PostingEntry>(),
        std::mem::size_of::<Vec<Occurrence>>() + std::mem::size_of::<usize>(),
        "a PostingEntry is a doc id and its occurrences, nothing else"
    );
}

/// The cost of payloads must
/// be the **bytes**, not a per-occurrence slot. An all-empty-payload field
/// therefore costs one `u32` length per occurrence and nothing else --
/// where the nested shape cost a `Vec` header per posting entry (24 bytes)
/// plus one allocation per non-empty payload.
///
/// The bound is expressed as "cheaper than one empty `Vec` header per
/// occurrence", which is exactly what the nested shape charged before a
/// single payload byte was stored, so `Vec`'s growth slack (the run runs
/// about 8.7 bytes per occurrence here against the 4 it holds) cannot
/// make it flaky while it still fails outright against the old shape.
#[test]
fn an_all_empty_payload_field_costs_only_a_length_per_occurrence() {
    let analyzer = Analyzer::standard(None);
    let texts: Vec<String> = (0..200).map(|d| format!("alpha beta doc{d}")).collect();
    let docs: Vec<(i32, &str, &str)> = texts
        .iter()
        .enumerate()
        .map(|(d, t)| (d as i32, "body", t.as_str()))
        .collect();
    let without = invert_documents(&docs, &analyzer);
    let with_empty = invert_documents_with_payloads(&docs, &analyzer, &["body"], &|_| None);

    let occurrences: usize = with_empty
        .terms
        .values()
        .flat_map(|l| l.entries.iter())
        .map(|e| e.occurrences.len())
        .sum();
    assert_eq!(occurrences, 600, "200 docs x 3 tokens");
    let extra = with_empty.ram_bytes_used() - without.ram_bytes_used();
    assert!(
        extra < occurrences * std::mem::size_of::<Vec<u8>>(),
        "an all-empty-payload field cost {extra} bytes for {occurrences} occurrences, \
             which is not cheaper than a vector header each"
    );
    assert!(
        extra >= occurrences * 4,
        "but it must still cost the lengths it stores: {extra}"
    );
}

/// Payload bytes are heap this structure occupies, so `ram_bytes_used`
/// has to count the flat run -- both its lengths and its bytes.
#[test]
fn ram_bytes_used_counts_the_payload_run_and_its_bytes() {
    let analyzer = Analyzer::standard(None);
    let docs = [(0, "body", "alpha beta")];
    let without = invert_documents(&docs, &analyzer);
    let with_empty = invert_documents_with_payloads(&docs, &analyzer, &["body"], &|_| None);
    let with_bytes =
        invert_documents_with_payloads(&docs, &analyzer, &["body"], &|_| Some(vec![0u8; 64]));

    assert!(
        with_empty.ram_bytes_used() > without.ram_bytes_used(),
        "the length run itself is real memory: {} vs {}",
        with_empty.ram_bytes_used(),
        without.ram_bytes_used()
    );
    assert!(
        with_bytes.ram_bytes_used() >= with_empty.ram_bytes_used() + 2 * 64,
        "64 payload bytes on each of two terms must show up: {} vs {}",
        with_bytes.ram_bytes_used(),
        with_empty.ram_bytes_used()
    );
}

/// The flat run is laid out in `entries` order, so the doc-ID sort has to
/// carry it. Documents are handed in **descending** doc-ID order with a
/// distinct, distinctly-*sized* payload per document, so a sort that
/// permuted the entries and left the run alone would hand every document
/// somebody else's payload -- and a sort that permuted fixed-width slices
/// would too.
#[test]
fn sorting_entries_by_doc_id_carries_the_payload_run_with_them() {
    let analyzer = Analyzer::standard(None);
    // doc 2 has one occurrence, doc 1 has two, doc 0 has three, so the
    // per-document runs have three different lengths as well as three
    // different byte counts.
    let docs = vec![
        (2, "body", "fox"),
        (1, "body", "fox fox"),
        (0, "body", "fox fox fox"),
    ];
    let index = invert_documents_with_payloads(&docs, &analyzer, &["body"], &|ctx| {
        Some(vec![ctx.doc_id as u8; ctx.doc_id as usize + 1])
    });
    let list = index.posting_list("body", "fox").expect("fox");
    assert_eq!(
        list.entries.iter().map(|e| e.doc_id).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(list.payload_lengths, vec![1, 1, 1, 2, 2, 3]);
    assert_eq!(list.payload_bytes, vec![0, 0, 0, 1, 1, 1, 1, 2, 2, 2]);
    for (entry_index, doc_id) in [0u8, 1, 2].into_iter().enumerate() {
        let expected = vec![doc_id; doc_id as usize + 1];
        for occurrence in 0..list.entries[entry_index].occurrences.len() {
            assert_eq!(
                list.payload(entry_index, occurrence),
                Some(&expected[..]),
                "entry {entry_index}, occurrence {occurrence}"
            );
        }
    }
    // Out of range in either dimension is `None`, not a panic.
    assert_eq!(list.payload(3, 0), None);
    assert_eq!(list.payload(0, 3), None);
}

/// The same sort with no payloads at all must still order the entries --
/// the early return for a payload-free list is a separate branch.
#[test]
fn sorting_entries_by_doc_id_works_without_payloads() {
    let analyzer = Analyzer::standard(None);
    let docs = vec![(2, "body", "fox"), (0, "body", "fox"), (1, "body", "fox")];
    let index = invert_documents(&docs, &analyzer);
    let list = index.posting_list("body", "fox").expect("fox");
    assert_eq!(
        list.entries.iter().map(|e| e.doc_id).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert!(!list.has_payloads());
    assert_eq!(list.payload(0, 0), None);
}

#[test]
fn empty_docs_batch_yields_empty_index() {
    let analyzer = Analyzer::standard(None);
    let index = invert_documents(&[], &analyzer);
    assert!(index.terms.is_empty());
}

#[test]
fn postings_lookup_returns_none_for_unknown_term() {
    let analyzer = Analyzer::standard(None);
    let docs = vec![(0, "body", "fox")];
    let index = invert_documents(&docs, &analyzer);
    assert!(index.postings("body", "nonexistent").is_none());
    assert!(index.postings("nonexistent-field", "fox").is_none());
}

/// The position accumulator was a bare `position += posIncr` on an `i32`.
/// Java guards it twice — it detects the wrap after the fact (`position <
/// lastPosition`) and rejects anything past `IndexWriter.MAX_POSITION` —
/// and neither guard was ported. In Rust the bare `+=` is not a wrap but a
/// **panic** in a debug build, so one field value carrying 2^31 token
/// positions takes the process down instead of the document.
#[test]
fn position_accumulator_clamps_instead_of_overflowing() {
    // The ordinary case is untouched: -1 is the pre-first-token seed, and
    // a standard increment of 1 lands the first token at position 0.
    assert_eq!(advance_position(-1, 1), 0);
    assert_eq!(advance_position(0, 1), 1);
    // A synonym's zero increment stays put.
    assert_eq!(advance_position(7, 0), 7);

    // Past the ceiling: clamped, never wrapped, and never negative.
    assert_eq!(advance_position(MAX_POSITION, 1), MAX_POSITION);
    assert_eq!(advance_position(MAX_POSITION - 1, 5), MAX_POSITION);
    assert_eq!(advance_position(i32::MAX - 1, 1), MAX_POSITION);
    assert_eq!(advance_position(i32::MAX, i32::MAX), MAX_POSITION);
    assert!(advance_position(MAX_POSITION, i32::MAX) >= 0);
}
