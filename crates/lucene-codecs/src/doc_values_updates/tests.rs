#![cfg(test)]
use super::*;
use lucene_store::codec_util::ID_LENGTH;

const SEG_ID: [u8; ID_LENGTH] = [7u8; ID_LENGTH];

#[test]
fn overlay_round_trip() {
    let updates = [(0, Some(100i64)), (5, Some(200)), (3, Some(300))];
    let bytes = write_numeric_updates(&updates, &SEG_ID, "");
    let map = read_numeric_updates(&bytes, &SEG_ID, "").unwrap();
    assert_eq!(map.len(), 3);
    assert_eq!(map.get(&0), Some(&Some(100)));
    assert_eq!(map.get(&5), Some(&Some(200)));
    assert_eq!(map.get(&3), Some(&Some(300)));
}

#[test]
fn overlay_round_trip_unsorted_input_and_duplicate_doc_keeps_last() {
    // Doc 2 appears twice; the later entry (value 99) should win, matching
    // last-write-wins semantics for a single buffered update batch.
    let updates = [(5, Some(1i64)), (2, Some(42)), (2, Some(99))];
    let bytes = write_numeric_updates(&updates, &SEG_ID, "");
    let map = read_numeric_updates(&bytes, &SEG_ID, "").unwrap();
    assert_eq!(map.len(), 2);
    assert_eq!(map.get(&2), Some(&Some(99)));
    assert_eq!(map.get(&5), Some(&Some(1)));
}

#[test]
fn empty_overlay_round_trips_to_empty_map() {
    let bytes = write_numeric_updates(&[], &SEG_ID, "");
    let map = read_numeric_updates(&bytes, &SEG_ID, "").unwrap();
    assert!(map.is_empty());
}

#[test]
fn wrong_segment_id_rejected() {
    let bytes = write_numeric_updates(&[(0, Some(1))], &SEG_ID, "");
    let other_id = [9u8; ID_LENGTH];
    assert!(read_numeric_updates(&bytes, &other_id, "").is_err());
}

#[test]
fn truncated_file_rejected() {
    let bytes = write_numeric_updates(&[(0, Some(1)), (1, Some(2))], &SEG_ID, "");
    let truncated = &bytes[..bytes.len() - 4];
    assert!(read_numeric_updates(truncated, &SEG_ID, "").is_err());
}

#[test]
fn hand_built_out_of_order_doc_ids_rejected() {
    // Hand-build a file with doc ids [5, 3] (not ascending) to exercise
    // the OutOfOrder error path -- write_numeric_updates itself always
    // sorts, so this scenario can only be reached via a malformed file.
    let mut out: Vec<u8> = Vec::new();
    codec_util::write_index_header(
        &mut out,
        NUMERIC_UPDATES_CODEC,
        VERSION_CURRENT,
        &SEG_ID,
        "",
    );
    out.write_vint(2);
    out.write_i32(5);
    out.push(1);
    out.write_i64(1);
    out.write_i32(3);
    out.push(1);
    out.write_i64(2);
    codec_util::write_footer(&mut out);
    assert!(matches!(
        read_numeric_updates(&out, &SEG_ID, ""),
        Err(Error::OutOfOrder { prev: 5, next: 3 })
    ));
}

// --- numeric_value_with_updates ---

fn dense_entry_and_data() -> (NumericEntry, Vec<u8>) {
    // A trivially simple dense field: 4 docs, values [10, 20, 30, 40],
    // plain (no table, gcd=1, min=0), built via the real writer so this
    // stays honest about the base format rather than hand-rolling one.
    let (meta, data, _skip) =
        doc_values::write_single_dense_numeric_field(0, &[10, 20, 30, 40], 4, &SEG_ID, "").unwrap();
    let field_infos = crate::field_infos::FieldInfos {
        fields: vec![crate::field_infos::FieldInfo {
            name: "f".to_string(),
            number: 0,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: crate::field_infos::IndexOptions::None,
            doc_values_type: crate::field_infos::DocValuesType::Numeric,
            doc_values_skip_index_type: crate::field_infos::DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: Vec::new(),
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: crate::field_infos::VectorEncoding::Float32,
            vector_similarity_function: crate::field_infos::VectorSimilarityFunction::Euclidean,
        }],
    };
    let (_, parsed) = doc_values::parse_meta(&meta, &SEG_ID, "", &field_infos).unwrap();
    let entry = parsed.numeric_entry(0).unwrap().clone();
    (entry, data)
}

#[test]
fn overlay_value_overrides_base_value_for_doc_present_in_both() {
    let (entry, data) = dense_entry_and_data();
    assert_eq!(
        doc_values::numeric_value(&data, &entry, 1).unwrap(),
        Some(20)
    );
    let mut updates = HashMap::new();
    updates.insert(1, Some(999i64));
    let result = numeric_value_with_updates(&entry, &data, &updates, 1).unwrap();
    assert_eq!(result, Some(999));
}

#[test]
fn doc_absent_from_overlay_falls_back_to_base_value() {
    let (entry, data) = dense_entry_and_data();
    let mut updates = HashMap::new();
    updates.insert(1, Some(999i64));
    // Doc 2 isn't in the overlay -> falls back to its base value (30).
    let result = numeric_value_with_updates(&entry, &data, &updates, 2).unwrap();
    assert_eq!(result, Some(30));
}

#[test]
fn empty_overlay_is_a_no_op_fallback_to_base_for_every_doc() {
    let (entry, data) = dense_entry_and_data();
    let updates = HashMap::new();
    for (doc, expected) in [(0, 10), (1, 20), (2, 30), (3, 40)] {
        let result = numeric_value_with_updates(&entry, &data, &updates, doc).unwrap();
        assert_eq!(result, Some(expected));
    }
}

// --- numeric_value_with_generations (multi-generation overlay chain) ---

#[test]
fn three_generations_newest_wins_for_a_doc_touched_by_all_three() {
    let (entry, data) = dense_entry_and_data();
    // Doc 1 (base value 20) gets updated at generation 1, then again at
    // generation 2, then again at generation 3 -- generation 3's value
    // must win, matching newest-generation-wins semantics.
    let gen1 = HashMap::from([(1, Some(1_001i64))]);
    let gen2 = HashMap::from([(1, Some(1_002i64))]);
    let gen3 = HashMap::from([(1, Some(1_003i64))]);
    let generations = [gen1, gen2, gen3];
    let result = numeric_value_with_generations(&entry, &data, &generations, 1).unwrap();
    assert_eq!(result, Some(1_003));
}

#[test]
fn overlapping_doc_sets_across_generations_each_doc_takes_its_own_newest_write() {
    let (entry, data) = dense_entry_and_data();
    // gen1 touches docs 0 and 1; gen2 touches docs 1 and 2 (overlapping
    // on doc 1, where gen2 must win); gen3 touches only doc 2 again.
    let gen1 = HashMap::from([(0, Some(100i64)), (1, Some(101))]);
    let gen2 = HashMap::from([(1, Some(201i64)), (2, Some(202))]);
    let gen3 = HashMap::from([(2, Some(302i64))]);
    let generations = [gen1, gen2, gen3];

    // Doc 0: only gen1 touched it -> gen1's value.
    assert_eq!(
        numeric_value_with_generations(&entry, &data, &generations, 0).unwrap(),
        Some(100)
    );
    // Doc 1: gen1 then gen2 touched it -> gen2 (newer) wins.
    assert_eq!(
        numeric_value_with_generations(&entry, &data, &generations, 1).unwrap(),
        Some(201)
    );
    // Doc 2: gen2 then gen3 touched it -> gen3 (newest) wins.
    assert_eq!(
        numeric_value_with_generations(&entry, &data, &generations, 2).unwrap(),
        Some(302)
    );
}

#[test]
fn doc_untouched_by_any_generation_falls_back_to_base() {
    let (entry, data) = dense_entry_and_data();
    let gen1 = HashMap::from([(0, Some(900i64))]);
    let gen2 = HashMap::from([(1, Some(901i64))]);
    let generations = [gen1, gen2];
    // Doc 3 (base value 40) isn't in either generation.
    let result = numeric_value_with_generations(&entry, &data, &generations, 3).unwrap();
    assert_eq!(result, Some(40));
}

#[test]
fn empty_generation_chain_degenerates_to_plain_base_decode() {
    let (entry, data) = dense_entry_and_data();
    let generations: [HashMap<i32, Option<i64>>; 0] = [];
    for (doc, expected) in [(0, 10), (1, 20), (2, 30), (3, 40)] {
        let result = numeric_value_with_generations(&entry, &data, &generations, doc).unwrap();
        assert_eq!(result, Some(expected));
    }
}

#[test]
fn a_generation_that_reverts_to_an_earlier_generations_untouched_state_still_falls_through() {
    let (entry, data) = dense_entry_and_data();
    // gen1 touches doc 2; gen2 touches a disjoint doc (0) only, so for
    // doc 2 the chain must fall through past gen2 to gen1's write.
    let gen1 = HashMap::from([(2, Some(555i64))]);
    let gen2 = HashMap::from([(0, Some(777i64))]);
    let generations = [gen1, gen2];
    let result = numeric_value_with_generations(&entry, &data, &generations, 2).unwrap();
    assert_eq!(result, Some(555));
}

#[test]
fn generations_can_be_written_and_read_back_via_existing_single_generation_io_then_chained() {
    // Proves the chain composes with the *unmodified* per-generation
    // write_numeric_updates/read_numeric_updates I/O -- each generation
    // really is just a standalone file, as the module doc comment says.
    let (entry, data) = dense_entry_and_data();
    let gen1_bytes = write_numeric_updates(&[(1, Some(111i64))], &SEG_ID, "");
    let gen2_bytes = write_numeric_updates(&[(1, Some(222i64)), (2, Some(322))], &SEG_ID, "");
    let gen1 = read_numeric_updates(&gen1_bytes, &SEG_ID, "").unwrap();
    let gen2 = read_numeric_updates(&gen2_bytes, &SEG_ID, "").unwrap();
    let generations = [gen1, gen2];

    // Doc 1: both generations touched it -> gen2 (newer) wins.
    assert_eq!(
        numeric_value_with_generations(&entry, &data, &generations, 1).unwrap(),
        Some(222)
    );
    // Doc 2: only gen2 touched it.
    assert_eq!(
        numeric_value_with_generations(&entry, &data, &generations, 2).unwrap(),
        Some(322)
    );
    // Doc 0: untouched by either generation -> base value 10.
    assert_eq!(
        numeric_value_with_generations(&entry, &data, &generations, 0).unwrap(),
        Some(10)
    );
}

// --- reset(doc): a generation that removes a doc's value ---

#[test]
fn reset_entry_round_trips_as_a_none_value() {
    let bytes = write_numeric_updates(&[(1, None), (2, Some(5i64))], &SEG_ID, "");
    let map = read_numeric_updates(&bytes, &SEG_ID, "").unwrap();
    assert_eq!(map.get(&1), Some(&None));
    assert_eq!(map.get(&2), Some(&Some(5)));
    // "reset" is not the same as "absent from the overlay".
    assert!(map.contains_key(&1));
    assert_eq!(map.get(&3), None);
}

/// `DocValuesFieldUpdates.reset(doc)` makes the doc read back as having no
/// value at all, shadowing whatever the base `.dvd` holds -- unlike a doc
/// the overlay never touched, which falls through to the base.
#[test]
fn reset_shadows_the_base_value() {
    let (entry, data) = dense_entry_and_data();
    let updates = HashMap::from([(1, None)]);
    assert_eq!(
        doc_values::numeric_value(&data, &entry, 1).unwrap(),
        Some(20)
    );
    assert_eq!(
        numeric_value_with_updates(&entry, &data, &updates, 1).unwrap(),
        None
    );
    // Doc 2 is untouched: still the base value.
    assert_eq!(
        numeric_value_with_updates(&entry, &data, &updates, 2).unwrap(),
        Some(30)
    );
}

/// Across generations the newest write wins whether it is a value or a
/// reset, in both directions.
#[test]
fn newest_generation_wins_whether_it_sets_or_resets() {
    let (entry, data) = dense_entry_and_data();
    // Doc 0: set at gen1, reset at gen2 -> no value.
    // Doc 1: reset at gen1, set at gen2 -> gen2's value.
    let gen1 = HashMap::from([(0, Some(11i64)), (1, None)]);
    let gen2 = HashMap::from([(0, None), (1, Some(22i64))]);
    let generations = [gen1, gen2];
    assert_eq!(
        numeric_value_with_generations(&entry, &data, &generations, 0).unwrap(),
        None
    );
    assert_eq!(
        numeric_value_with_generations(&entry, &data, &generations, 1).unwrap(),
        Some(22)
    );
}

/// A reset in an older generation still shadows the base when no newer
/// generation touches the doc.
#[test]
fn an_older_generations_reset_survives_a_disjoint_newer_generation() {
    let (entry, data) = dense_entry_and_data();
    let gen1 = HashMap::from([(2, None)]);
    let gen2 = HashMap::from([(0, Some(7i64))]);
    let generations = [gen1, gen2];
    assert_eq!(
        numeric_value_with_generations(&entry, &data, &generations, 2).unwrap(),
        None
    );
}

/// A `VERSION_START` file predates the per-entry `has_value` byte; every
/// entry in one is a plain value assignment and must still decode.
#[test]
fn version_start_file_without_has_value_bytes_still_reads() {
    let mut out: Vec<u8> = Vec::new();
    codec_util::write_index_header(&mut out, NUMERIC_UPDATES_CODEC, VERSION_START, &SEG_ID, "");
    out.write_vint(2);
    out.write_i32(1);
    out.write_i64(111);
    out.write_i32(4);
    out.write_i64(444);
    codec_util::write_footer(&mut out);

    let map = read_numeric_updates(&out, &SEG_ID, "").unwrap();
    assert_eq!(map.get(&1), Some(&Some(111)));
    assert_eq!(map.get(&4), Some(&Some(444)));
}

// --- BINARY updates (`BinaryDocValuesFieldUpdates`) ---

fn bin(bytes: &[u8]) -> Option<Vec<u8>> {
    Some(bytes.to_vec())
}

#[test]
fn binary_overlay_round_trips_variable_length_values() {
    let updates = [
        (0, bin(b"alpha")),
        (5, bin(b"")),
        (3, bin(b"a much longer value than the others")),
    ];
    let bytes = write_binary_updates(&updates, &SEG_ID, "");
    let map = read_binary_updates(&bytes, &SEG_ID, "").unwrap();
    assert_eq!(map.len(), 3);
    assert_eq!(map[&0].as_deref(), Some(&b"alpha"[..]));
    // An empty value is a *value*, not an absence.
    assert_eq!(map[&5].as_deref(), Some(&b""[..]));
    assert_eq!(
        map[&3].as_deref(),
        Some(&b"a much longer value than the others"[..])
    );
}

#[test]
fn binary_reset_round_trips_as_a_none_value_distinct_from_an_empty_one() {
    let updates = [(0, None), (1, bin(b""))];
    let bytes = write_binary_updates(&updates, &SEG_ID, "");
    let map = read_binary_updates(&bytes, &SEG_ID, "").unwrap();
    assert_eq!(map[&0], None);
    assert_eq!(map[&1], Some(Vec::new()));
}

#[test]
fn binary_overlay_unsorted_input_and_duplicate_doc_keeps_last() {
    let updates = [(2, bin(b"first")), (0, bin(b"zero")), (2, bin(b"second"))];
    let bytes = write_binary_updates(&updates, &SEG_ID, "");
    let map = read_binary_updates(&bytes, &SEG_ID, "").unwrap();
    assert_eq!(map.len(), 2);
    assert_eq!(map[&2].as_deref(), Some(&b"second"[..]));
}

#[test]
fn binary_overlay_rejects_a_numeric_overlay_file() {
    // Distinct codec names: handing a numeric overlay to the binary reader
    // must fail the header check rather than decode nonsense.
    let numeric = write_numeric_updates(&[(0, Some(1))], &SEG_ID, "");
    assert!(read_binary_updates(&numeric, &SEG_ID, "").is_err());
    let binary = write_binary_updates(&[(0, bin(b"x"))], &SEG_ID, "");
    assert!(read_numeric_updates(&binary, &SEG_ID, "").is_err());
}

#[test]
fn binary_overlay_rejects_a_wrong_segment_id_and_a_corrupt_footer() {
    let bytes = write_binary_updates(&[(0, bin(b"x"))], &SEG_ID, "");
    assert!(read_binary_updates(&bytes, &[9u8; ID_LENGTH], "").is_err());
    let mut corrupt = bytes.clone();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 0xFF;
    assert!(read_binary_updates(&corrupt, &SEG_ID, "").is_err());
}

#[test]
fn binary_overlay_rejects_out_of_order_doc_ids_on_read() {
    let mut out: Vec<u8> = Vec::new();
    codec_util::write_index_header(&mut out, BINARY_UPDATES_CODEC, VERSION_CURRENT, &SEG_ID, "");
    out.write_vint(2);
    out.write_i32(5);
    out.push(0);
    out.write_i32(2);
    out.push(0);
    codec_util::write_footer(&mut out);
    assert!(matches!(
        read_binary_updates(&out, &SEG_ID, ""),
        Err(Error::OutOfOrder { prev: 5, next: 2 })
    ));
}

#[test]
fn binary_overlay_rejects_a_negative_value_length() {
    let mut out: Vec<u8> = Vec::new();
    codec_util::write_index_header(&mut out, BINARY_UPDATES_CODEC, VERSION_CURRENT, &SEG_ID, "");
    out.write_vint(1);
    out.write_i32(0);
    out.push(1);
    out.write_vint(-1);
    codec_util::write_footer(&mut out);
    assert!(matches!(
        read_binary_updates(&out, &SEG_ID, ""),
        Err(Error::NegativeLength(-1))
    ));
}

#[test]
fn binary_overlay_rejects_counts_and_lengths_past_the_file() {
    let mut out: Vec<u8> = Vec::new();
    codec_util::write_index_header(&mut out, BINARY_UPDATES_CODEC, VERSION_CURRENT, &SEG_ID, "");
    out.write_vint(i32::MAX);
    out.write_i32(0);
    out.push(1);
    out.write_vint(i32::MAX);
    codec_util::write_footer(&mut out);
    assert!(matches!(
        read_binary_updates(&out, &SEG_ID, ""),
        Err(Error::Store(lucene_store::Error::Eof { .. }))
    ));
}

/// The overlay-read helpers against an *empty* base entry: a binary field
/// with no values at all, so every answer comes from the overlay chain.
/// `BinaryEntry::is_empty_field()` short-circuits the base decode, which
/// is what lets this test exercise resolution without a real `.dvd`.
fn empty_binary_entry() -> doc_values::BinaryEntry {
    doc_values::BinaryEntry {
        disi: Default::default(),
        mask: None,
        field_number: 0,
        // `DOCS_WITH_FIELD_EMPTY` (-2): `is_empty_field()` is true, so
        // `binary_value` returns `None` without touching `base_data`.
        docs_with_field_offset: -2,
        docs_with_field_length: 0,
        jump_table_entry_count: 0,
        dense_rank_power: 0,
        num_docs_with_field: 0,
        min_length: 0,
        max_length: 0,
        data_offset: 0,
        data_length: 0,
        addresses: None,
        substring: None,
    }
}

#[test]
fn binary_overlay_read_prefers_the_overlay_then_falls_back_to_the_base() {
    let entry = empty_binary_entry();
    let mut updates = HashMap::new();
    updates.insert(0, bin(b"set"));
    updates.insert(1, None);
    assert_eq!(
        binary_value_with_updates(&entry, &[], &updates, 0).unwrap(),
        Some(&b"set"[..])
    );
    // A `reset` shadows the base rather than falling through to it.
    assert_eq!(
        binary_value_with_updates(&entry, &[], &updates, 1).unwrap(),
        None
    );
    // Untouched doc: the base decode (an empty field) answers.
    assert_eq!(
        binary_value_with_updates(&entry, &[], &updates, 2).unwrap(),
        None
    );
}

#[test]
fn binary_newest_generation_wins_whether_it_sets_or_resets() {
    let entry = empty_binary_entry();
    let mut older = HashMap::new();
    older.insert(0, bin(b"old"));
    older.insert(1, bin(b"kept"));
    let mut newer = HashMap::new();
    newer.insert(0, bin(b"new"));
    newer.insert(2, None);
    let generations = vec![older, newer];

    assert_eq!(
        binary_value_with_generations(&entry, &[], &generations, 0).unwrap(),
        Some(&b"new"[..])
    );
    // A doc the newer generation did not touch falls through to the older.
    assert_eq!(
        binary_value_with_generations(&entry, &[], &generations, 1).unwrap(),
        Some(&b"kept"[..])
    );
    assert_eq!(
        binary_value_with_generations(&entry, &[], &generations, 2).unwrap(),
        None
    );
}

#[test]
fn binary_an_older_generations_reset_survives_a_disjoint_newer_generation() {
    let entry = empty_binary_entry();
    let mut older = HashMap::new();
    older.insert(0, None);
    let mut newer = HashMap::new();
    newer.insert(1, bin(b"other"));
    let generations = vec![older, newer];
    assert_eq!(
        binary_value_with_generations(&entry, &[], &generations, 0).unwrap(),
        None
    );
}

#[test]
fn binary_no_generations_degenerates_to_the_base_decode() {
    let entry = empty_binary_entry();
    assert_eq!(
        binary_value_with_generations(&entry, &[], &[], 0).unwrap(),
        None
    );
}

// -----------------------------------------------------------------
// The generational (real-Lucene) representation
// -----------------------------------------------------------------

/// Writes a base NUMERIC column and reads its entry back, so the merge
/// tests below start from a genuine `.dvd`/`.dvm` rather than a
/// hand-built entry.
fn base_numeric(values: &[Option<i64>]) -> (NumericEntry, Vec<u8>) {
    let (meta, data, _) =
        write_numeric_generation(0, values, &SEG_ID, "").expect("write base column");
    let fis = numeric_field_infos();
    let (_, parsed) = doc_values::parse_meta(&meta, &SEG_ID, "", &fis).expect("parse base");
    (
        parsed.numeric_entry(0).expect("numeric entry").clone(),
        data,
    )
}

fn numeric_field_infos() -> crate::field_infos::FieldInfos {
    crate::field_infos::FieldInfos {
        fields: vec![dv_field(crate::field_infos::DocValuesType::Numeric)],
    }
}

fn binary_field_infos() -> crate::field_infos::FieldInfos {
    crate::field_infos::FieldInfos {
        fields: vec![dv_field(crate::field_infos::DocValuesType::Binary)],
    }
}

fn dv_field(ty: crate::field_infos::DocValuesType) -> crate::field_infos::FieldInfo {
    crate::field_infos::FieldInfo {
        name: "f".to_string(),
        number: 0,
        store_term_vectors: false,
        omit_norms: true,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: crate::field_infos::IndexOptions::None,
        doc_values_type: ty,
        doc_values_skip_index_type: crate::field_infos::DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: crate::field_infos::VectorEncoding::Float32,
        vector_similarity_function: crate::field_infos::VectorSimilarityFunction::Euclidean,
    }
}

fn read_numeric_column(meta: &[u8], data: &[u8], max_doc: i32) -> Vec<Option<i64>> {
    let fis = numeric_field_infos();
    let (_, parsed) = doc_values::parse_meta(meta, &SEG_ID, "", &fis).unwrap();
    let entry = parsed.numeric_entry(0).unwrap();
    (0..max_doc)
        .map(|d| doc_values::numeric_value(data, entry, d).unwrap())
        .collect()
}

fn read_binary_column(meta: &[u8], data: &[u8], max_doc: i32) -> Vec<Option<Vec<u8>>> {
    let fis = binary_field_infos();
    let (_, parsed) = doc_values::parse_meta(meta, &SEG_ID, "", &fis).unwrap();
    let entry = parsed.binary_entry(0).unwrap();
    (0..max_doc)
        .map(|d| {
            doc_values::binary_value(data, entry, d)
                .unwrap()
                .map(<[u8]>::to_vec)
        })
        .collect()
}

#[test]
fn a_merged_numeric_column_keeps_every_document_the_update_did_not_touch() {
    // The single easiest thing to lose in a full-column rewrite, and the
    // one nothing structural catches: a generation that only holds the
    // updated documents reads back as "no value" for all the rest.
    let (entry, data) = base_numeric(&[Some(10), Some(20), Some(30), Some(40)]);
    let column =
        merge_numeric_column(Some((&entry, &data)), &[(1, Some(99)), (3, None)], 4).unwrap();
    assert_eq!(column, vec![Some(10), Some(99), Some(30), None]);
}

#[test]
fn a_later_update_for_the_same_doc_wins() {
    let (entry, data) = base_numeric(&[Some(1), Some(2)]);
    let column = merge_numeric_column(
        Some((&entry, &data)),
        &[(0, Some(5)), (0, None), (0, Some(7))],
        2,
    )
    .unwrap();
    assert_eq!(column[0], Some(7));
}

#[test]
fn a_field_with_no_base_column_starts_from_all_absent() {
    // Java's `FieldInfos.FieldNumbers.constructFieldInfo` case: the field
    // exists globally but never had a value in this segment.
    let column = merge_numeric_column(None, &[(2, Some(4))], 4).unwrap();
    assert_eq!(column, vec![None, None, Some(4), None]);
}

#[test]
fn an_update_outside_the_segments_doc_range_is_an_error_not_a_silent_drop() {
    // A resolver that produced an out-of-range doc id has a bug, and
    // "that update silently didn't happen" is the worst way to learn it.
    // `doc_values::write_single_sparse_numeric_field` already errors on
    // the same condition; these two must agree.
    assert!(merge_numeric_column(None, &[(9, Some(2))], 3).is_err());
    assert!(merge_numeric_column(None, &[(-1, Some(1))], 3).is_err());
    assert!(merge_binary_column(None, &[(3, Some(vec![1]))], 3).is_err());
}

#[test]
fn a_fully_dense_merged_column_round_trips_through_the_dense_writer() {
    let (meta, data, _) =
        write_numeric_generation(0, &[Some(3), Some(4), Some(5)], &SEG_ID, "").unwrap();
    assert_eq!(
        read_numeric_column(&meta, &data, 3),
        vec![Some(3), Some(4), Some(5)]
    );
}

#[test]
fn a_partly_reset_merged_column_round_trips_through_the_sparse_writer() {
    let (meta, data, _) =
        write_numeric_generation(0, &[Some(3), None, Some(5)], &SEG_ID, "").unwrap();
    assert_eq!(
        read_numeric_column(&meta, &data, 3),
        vec![Some(3), None, Some(5)]
    );
}

#[test]
fn a_generation_that_reset_every_value_round_trips_as_an_empty_column() {
    // `updateDocValues(term, field with a null value)` matching every
    // document. Java writes the `meta[-2, 0]` "no documents with values"
    // shape here, not a zero-length DISI.
    let (meta, data, _) = write_numeric_generation(0, &[None, None], &SEG_ID, "").unwrap();
    assert_eq!(read_numeric_column(&meta, &data, 2), vec![None, None]);
}

#[test]
fn a_merged_binary_column_distinguishes_an_empty_value_from_a_removed_one() {
    let (meta, data, _) = write_binary_generation(
        0,
        vec![Some(Vec::new()), None, Some(b"x".to_vec())],
        &SEG_ID,
        "",
    )
    .unwrap();
    assert_eq!(
        read_binary_column(&meta, &data, 3),
        vec![Some(Vec::new()), None, Some(b"x".to_vec())]
    );
}

#[test]
fn a_binary_update_reads_the_base_column_for_the_docs_it_does_not_touch() {
    let base_values = vec![
        Some(b"aa".to_vec()),
        Some(b"bb".to_vec()),
        Some(b"cc".to_vec()),
    ];
    let (meta, data, _) = write_binary_generation(0, base_values.clone(), &SEG_ID, "").unwrap();
    let fis = binary_field_infos();
    let (_, parsed) = doc_values::parse_meta(&meta, &SEG_ID, "", &fis).unwrap();
    let entry = parsed.binary_entry(0).unwrap();

    let column = merge_binary_column(
        Some((entry, &data)),
        &[(1, Some(b"updated".to_vec())), (2, None)],
        3,
    )
    .unwrap();
    assert_eq!(
        column,
        vec![Some(b"aa".to_vec()), Some(b"updated".to_vec()), None]
    );
}

#[test]
fn a_generation_written_with_a_generation_suffix_only_opens_with_that_suffix() {
    // The suffix is in the file *name* and in the file's index header, and
    // the two must be the same string -- a mismatch is a file that exists
    // and fails `checkIndexHeader`.
    let (meta, _data, _) =
        write_numeric_generation(0, &[Some(1)], &SEG_ID, "1_Lucene90_0").unwrap();
    let fis = numeric_field_infos();
    assert!(doc_values::parse_meta(&meta, &SEG_ID, "1_Lucene90_0", &fis).is_ok());
    assert!(doc_values::parse_meta(&meta, &SEG_ID, "", &fis).is_err());
    assert!(doc_values::parse_meta(&meta, &SEG_ID, "2_Lucene90_0", &fis).is_err());
}
