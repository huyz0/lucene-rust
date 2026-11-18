#![cfg(test)]

#[test]
fn simpletext_norms_keep_long_values_missing_docs_and_independent_cursors() {
    use lucene_codecs::field_infos::{DocValuesType, FieldInfo};
    use lucene_codecs::simpletext::doc_values_reader::SimpleTextDocValuesReader;
    use lucene_codecs::simpletext::doc_values_writer::SimpleTextDocValuesWriter;

    let mut field = FieldInfo::new("norm", 0);
    field.doc_values_type = DocValuesType::Numeric;
    let mut writer = SimpleTextDocValuesWriter::new(5);
    let expected = [(0, 1), (1, 128), (3, -129), (4, i64::MAX)];
    writer.add_numeric_field(&field, &expected).unwrap();
    let bytes = writer.finish().unwrap();
    let reader = SimpleTextDocValuesReader::open(&bytes, 5).unwrap();
    let table = super::inverse_table(7.5);
    let norms = FieldNorms::from_simpletext(reader.numeric("norm").unwrap(), 7.5, table.clone());
    assert!(norms.cursor().dense_bytes().is_none());
    let mut first = norms.cursor();
    let mut second = norms.cursor();
    for doc in [4, 0, 1, 2, 3, 0] {
        let value = expected
            .iter()
            .find(|(d, _)| *d == doc)
            .map(|(_, value)| *value);
        assert_eq!(first.norm_long(doc).unwrap(), value);
        assert_eq!(first.has_norm(doc).unwrap(), value.is_some());
        assert_eq!(second.norm_long(doc).unwrap(), value);
        if let Some(value) = value {
            assert_eq!(
                first.norm_inverse(doc).unwrap().to_bits(),
                table[value as u8 as usize].to_bits()
            );
            assert_eq!(
                first.field_length(doc).unwrap(),
                crate::similarity::decode_norm(i64::from(value as u8))
            );
        } else {
            assert_eq!(
                first.field_length(doc).unwrap(),
                crate::similarity::UNNORMED_FIELD_LENGTH
            );
        }
    }
    for doc in [-1, 5] {
        assert!(matches!(
            first.norm_long(doc),
            Err(norms::Error::DocOutOfRange(_, 5))
        ));
        assert!(first.norm_inverse(doc).is_err());
    }
    let docs = [0, 1, 2, 3, 4];
    let mut longs = Vec::new();
    first.norm_long_batch(&docs, &mut longs).unwrap();
    assert_eq!(longs, [1, 128, 1, -129, i64::MAX]);
    let mut inverses = Vec::new();
    first.norm_inverse_batch(&docs, &mut inverses).unwrap();
    assert_eq!(inverses, docs.map(|doc| norms.norm_inverse(doc).unwrap()));
}

#[test]
fn simpletext_scoring_norms_report_lazy_value_corruption() {
    use lucene_codecs::field_infos::{DocValuesType, FieldInfo};
    use lucene_codecs::simpletext::doc_values_reader::SimpleTextDocValuesReader;
    use lucene_codecs::simpletext::doc_values_writer::SimpleTextDocValuesWriter;

    let mut field = FieldInfo::new("norm", 0);
    field.doc_values_type = DocValuesType::Numeric;
    let mut writer = SimpleTextDocValuesWriter::new(1);
    writer.add_numeric_field(&field, &[(0, 7)]).unwrap();
    let mut bytes = writer.finish().unwrap();
    let reader = SimpleTextDocValuesReader::open(&bytes, 1)
        .unwrap()
        .rebind(&[]);
    bytes[reader.field("norm").unwrap().data_start] = b'x';
    let norms = FieldNorms::from_simpletext(
        reader.numeric_from_data("norm", &bytes).unwrap(),
        7.0,
        super::inverse_table(7.0),
    );
    let error = norms.cursor().norm_long(0).unwrap_err().to_string();
    assert!(error.contains("SimpleText norms:"), "{error}");
    assert!(norms.norm_inverse(0).is_err());
}

/// `avgdl = sumTotalTermFreq / docCount`, real Lucene's formula.
#[test]
fn from_field_stats_divides_sum_by_doc_count() {
    let n = FieldNorms::from_field_stats(&[], dense_entry(1, 1, 0), 1000, 40);
    assert_eq!(n.avg_field_length, 25.0);
}

/// An empty field would divide by zero in Lucene; this port falls back to
/// the no-op constant, matching `open`'s treatment of the same edge case.
#[test]
fn from_field_stats_empty_field_falls_back_to_unnormed() {
    let n = FieldNorms::from_field_stats(&[], dense_entry(1, 1, 0), 0, 0);
    assert_eq!(n.avg_field_length, crate::similarity::UNNORMED_FIELD_LENGTH);
    let n = FieldNorms::from_field_stats(&[], dense_entry(1, 1, 0), 500, -1);
    assert_eq!(n.avg_field_length, crate::similarity::UNNORMED_FIELD_LENGTH);
}

/// Why averaging quantized norms is wrong, pinned as a test rather than left
/// as prose.
///
/// Norms are `SmallFloat`-quantized to one byte. Below 25 the encoding is
/// exact, so averaging decoded norms and dividing exact counters agree --
/// which is precisely why every existing fixture (whose documents are 1-3
/// terms long) passed with the wrong formula. Above that range the encoding
/// is lossy and the two diverge, which is what the 5M-document corpus,
/// with 40-160 term documents, exposed.
#[test]
fn small_float_encoding_is_exact_below_25_and_lossy_above() {
    for len in 1u32..25 {
        let decoded =
            crate::similarity::decode_norm(lucene_util::small_float::int_to_byte4(len) as i64);
        assert_eq!(
            decoded, len as f32,
            "length {len} should round-trip exactly"
        );
    }
    // A realistic document length: the round trip no longer returns the
    // input, so an average of decoded norms is not the average of lengths.
    let lossy: Vec<u32> = (100..=160)
        .filter(|&len| {
            crate::similarity::decode_norm(lucene_util::small_float::int_to_byte4(len) as i64)
                != len as f32
        })
        .collect();
    assert!(
        !lossy.is_empty(),
        "expected SmallFloat to be lossy for realistic document lengths"
    );
}

use super::*;

/// A field without norms scores every document at norm 1 -- field length
/// one -- against the real `avgdl`, as `LeafSimScorer` + `BM25Similarity`
/// do: `cache[1] = 1 / (k1 * ((1 - b) + b * 1 / avgdl))`. Forcing `avgdl` to
/// one as well would score a freqs-only field about 12% low.
#[test]
fn unnormed_field_scores_at_length_one_against_the_real_avgdl() {
    let avgdl = 3.5f32;
    let norms = FieldNorms::unnormed(10, avgdl);
    assert_eq!(norms.avg_field_length, avgdl);
    let want = 1.0
        / (crate::similarity::DEFAULT_K1
            * ((1.0 - crate::similarity::DEFAULT_B) + crate::similarity::DEFAULT_B * 1.0 / avgdl));
    let mut cursor = norms.cursor();
    for doc in [0, 5, 9] {
        assert_eq!(cursor.field_length(doc).unwrap(), 1.0);
        assert_eq!(cursor.norm_inverse(doc).unwrap(), want);
        assert_eq!(norms.field_length(doc).unwrap(), 1.0);
    }
    // Past the segment is an error, as for any dense norms.
    assert!(norms.field_length(10).is_err());
    // At avgdl 1 it is exactly the unnormed constant.
    let one = FieldNorms::unnormed(1, 1.0);
    assert_eq!(
        one.cursor().norm_inverse(0).unwrap(),
        crate::similarity::UNNORMED_NORM_INVERSE
    );
}
use lucene_codecs::norms::NormsEntry;

/// `IndexedDISI`'s "this entry carries no rank table", written as the byte
/// `0xFF` (Java's `denseRankPower == -1`). Every sparse entry this port
/// writes uses it, and it is the only value besides 7..=15 that
/// `lucene_codecs::indexed_disi` accepts.
///
/// A literal `0` would *not* be a legal `denseRankPower`:
/// `dense_rank_bytes(0)` rejects it, so any DENSE `IndexedDISI` block
/// reached with it fails to decode. Dense-norms entries never look at the
/// field and a 4-document sparse block is SPARSE, so a `0` would go
/// unnoticed -- but the fixtures would describe metadata no writer can
/// produce, an almost-right test input that hides real decode bugs.
const NO_RANK: u8 = 0xFF;

fn dense_entry(bytes_per_norm: u8, num_docs: i32, norms_offset: i64) -> NormsEntry {
    NormsEntry {
        field_number: 0,
        docs_with_field_offset: -1, // dense
        docs_with_field_length: 0,
        jump_table_entry_count: -1,
        dense_rank_power: NO_RANK,
        num_docs_with_field: num_docs,
        bytes_per_norm,
        norms_offset,
    }
}

/// The dense one-byte fast path and the general `norms::norm_value` path
/// must agree for every document and every possible norm byte, including
/// the ones that decode as negative `i8`. They are two implementations of
/// the same lookup, and the fast one exists only because it is faster --
/// so the only thing worth testing about it is that it is not also
/// different.
#[test]
fn dense_fast_path_agrees_with_the_general_norm_lookup_for_every_byte() {
    let data: Vec<u8> = (0..=255u8).collect();
    let entry = dense_entry(1, data.len() as i32, 0);
    let n = FieldNorms::from_field_stats(&data, entry, 4000, 256);
    assert!(
        n.dense_norm_bytes.is_some(),
        "this entry is exactly the shape the fast path is for"
    );
    for doc in 0..data.len() as i32 {
        let general = match norms::norm_value(n.data, &n.entry, doc).unwrap() {
            Some(norm) => n.norm_inverse[(norm as u8) as usize],
            None => unreachable!("dense field, every doc has a norm"),
        };
        assert_eq!(
            n.norm_inverse(doc).unwrap(),
            general,
            "fast and general paths disagree at doc {doc} (norm byte {})",
            data[doc as usize]
        );
    }
}

/// The sparse fast path and the general `norms::norm_value` path must agree
/// for every document, present and absent alike -- the same argument as the
/// dense test above, for the branch that decodes the `IndexedDISI` list
/// once instead of per lookup.
#[test]
fn sparse_fast_path_agrees_with_the_general_norm_lookup() {
    // Docs 0, 3, 6, 9 have a norm; 1, 2, 4, 5, 7, 8 do not.
    let present: Vec<i32> = (0..4).map(|i| i * 3).collect();
    let (disi, disi_jumps) = lucene_codecs::indexed_disi::write(&present);
    let norms_bytes = [7u8, 19, 31, 43];
    let mut data = disi.clone();
    let norms_offset = data.len() as i64;
    data.extend_from_slice(&norms_bytes);

    let entry = NormsEntry {
        field_number: 0,
        docs_with_field_offset: 0,
        docs_with_field_length: disi.len() as i64,
        jump_table_entry_count: disi_jumps,
        dense_rank_power: NO_RANK,
        num_docs_with_field: present.len() as i32,
        bytes_per_norm: 1,
        norms_offset,
    };
    let n = FieldNorms::from_field_stats(&data, entry, 400, 10);
    assert!(
        n.sparse_region.is_some(),
        "this entry is exactly the shape the sparse path is for"
    );
    assert!(
        n.dense_norm_bytes.is_none(),
        "a sparse field must not take the dense path"
    );

    for doc in 0..10i32 {
        let general = match norms::norm_value(n.data, &n.entry, doc).unwrap() {
            Some(norm) => n.norm_inverse[(norm as u8) as usize],
            None => n.norm_inverse(doc).unwrap(), // absent: both fall back identically
        };
        assert_eq!(
            n.norm_inverse(doc).unwrap(),
            general,
            "fast and general paths disagree at doc {doc}"
        );
    }
    // And the values are the ones actually written, not a coincidence of
    // both paths being wrong the same way.
    assert_eq!(n.norm_inverse(0).unwrap(), n.norm_inverse[7]);
    assert_eq!(n.norm_inverse(9).unwrap(), n.norm_inverse[43]);
    assert_eq!(n.field_length(3).unwrap(), n.norm_length[19]);
}

/// The previous test's sparse field has 4 documents, so its `IndexedDISI`
/// region is a single SPARSE block -- which never reads the rank table, and
/// so never notices an illegal `dense_rank_power`. This one crosses the
/// 4096-document threshold into a DENSE block, where the rank-power byte
/// *is* consulted (`dense_rank_bytes`), so it pins that the entry's
/// metadata is a value a real writer produces.
///
/// This is also the shape `Lucene90NormsProducer` reads for a large, mostly
/// -absent field: the norms values stay a flat ordinal-indexed array while
/// the doc-id list goes dense.
#[test]
fn sparse_norms_over_a_dense_indexed_disi_block_agree_with_the_general_path() {
    // 5000 of the first 10000 docs have a norm: > 4095 in block 0, so
    // `indexed_disi::write` emits a DENSE block, not a SPARSE one.
    let present: Vec<i32> = (0..5000).map(|i| i * 2).collect();
    let (disi, disi_jumps) = lucene_codecs::indexed_disi::write(&present);
    let norms_bytes: Vec<u8> = (0..present.len()).map(|i| (i % 251 + 1) as u8).collect();
    let mut data = disi.clone();
    let norms_offset = data.len() as i64;
    data.extend_from_slice(&norms_bytes);

    let entry = NormsEntry {
        field_number: 0,
        docs_with_field_offset: 0,
        docs_with_field_length: disi.len() as i64,
        jump_table_entry_count: disi_jumps,
        dense_rank_power: NO_RANK,
        num_docs_with_field: present.len() as i32,
        bytes_per_norm: 1,
        norms_offset,
    };
    let n = FieldNorms::from_field_stats(&data, entry, 250_000, 5000);
    assert!(
        n.sparse_region.is_some(),
        "a legal rank power must let the DENSE block decode"
    );
    assert_eq!(
        lucene_codecs::indexed_disi::decode_doc_ids(n.sparse_region.unwrap(), NO_RANK)
            .unwrap()
            .len(),
        present.len()
    );

    for doc in [0i32, 1, 2, 4321, 9998, 9999] {
        let general = norms::norm_value(n.data, &n.entry, doc).unwrap();
        match general {
            Some(norm) => {
                assert_eq!(
                    n.norm_inverse(doc).unwrap(),
                    n.norm_inverse[norm as u8 as usize]
                );
                assert_eq!(
                    n.field_length(doc).unwrap(),
                    n.norm_length[norm as u8 as usize]
                );
            }
            None => assert_eq!(
                n.field_length(doc).unwrap(),
                crate::similarity::UNNORMED_FIELD_LENGTH
            ),
        }
    }
}

/// The incremental cursor and the one-shot `&self` conveniences must agree
/// for every document of a sparse field, present and absent alike, in
/// ascending order (the cursor's fast path) *and* in descending order (its
/// rewind path). Two implementations of one lookup; the only thing worth
/// testing about the fast one is that it is not also different.
///
/// Guards the forward-only cursor:
/// a forward-only cursor that did *not* rewind would answer "absent" for
/// every doc after the first backwards step.
#[test]
fn the_cursor_agrees_with_the_one_shot_lookup_forwards_and_backwards() {
    // 5000 of the first 10000 docs: a DENSE `IndexedDISI` block, so the
    // rank table and the word walk are both exercised.
    let present: Vec<i32> = (0..5000).map(|i| i * 2).collect();
    let (disi, disi_jumps) = lucene_codecs::indexed_disi::write(&present);
    let norms_bytes: Vec<u8> = (0..present.len()).map(|i| (i % 251 + 1) as u8).collect();
    let mut data = disi.clone();
    let norms_offset = data.len() as i64;
    data.extend_from_slice(&norms_bytes);

    let entry = NormsEntry {
        field_number: 0,
        docs_with_field_offset: 0,
        docs_with_field_length: disi.len() as i64,
        jump_table_entry_count: disi_jumps,
        dense_rank_power: NO_RANK,
        num_docs_with_field: present.len() as i32,
        bytes_per_norm: 1,
        norms_offset,
    };
    let n = FieldNorms::from_field_stats(&data, entry, 250_000, 5000);

    let mut ascending = n.cursor();
    for doc in 0..10_000i32 {
        assert_eq!(
            ascending.norm_inverse(doc).unwrap(),
            n.norm_inverse(doc).unwrap(),
            "ascending cursor disagrees at doc {doc}"
        );
    }
    let mut descending = n.cursor();
    for doc in (0..10_000i32).rev() {
        assert_eq!(
            descending.field_length(doc).unwrap(),
            n.field_length(doc).unwrap(),
            "rewinding cursor disagrees at doc {doc}"
        );
    }
    // And an arbitrary order, with repeats -- a caller such as the
    // fuzzy-expansion loop restarts per expanded term, and asking the same
    // document twice must not consume it (the `Vec` this replaced was
    // trivially idempotent; a cursor is not obviously so).
    let mut jumping = n.cursor();
    for doc in [9998i32, 9998, 3, 3, 4321, 0, 9999, 9999, 2, 7777, 7777, 1] {
        assert_eq!(
            jumping.field_length(doc).unwrap(),
            n.field_length(doc).unwrap(),
            "jumping cursor disagrees at doc {doc}"
        );
    }
}

/// A dense field needs no `IndexedDISI` cursor at all, and a negative doc
/// must be the same `DocOutOfRange` error on every path -- the sparse
/// branch's `advance_exact` would otherwise assert (panic) on it.
#[test]
fn a_negative_doc_is_an_error_not_a_panic_on_either_shape() {
    let dense = FieldNorms::from_field_stats(&[5u8, 15, 25], dense_entry(1, 3, 0), 45, 3);
    assert!(dense.cursor().norm_inverse(-1).is_err());
    assert!(dense.cursor().field_length(-1).is_err());

    let present: Vec<i32> = (0..4).map(|i| i * 3).collect();
    let (disi, disi_jumps) = lucene_codecs::indexed_disi::write(&present);
    let mut data = disi.clone();
    let norms_offset = data.len() as i64;
    data.extend_from_slice(&[7u8, 19, 31, 43]);
    let sparse = FieldNorms::from_field_stats(
        &data,
        NormsEntry {
            field_number: 0,
            docs_with_field_offset: 0,
            docs_with_field_length: disi.len() as i64,
            jump_table_entry_count: disi_jumps,
            dense_rank_power: NO_RANK,
            num_docs_with_field: present.len() as i32,
            bytes_per_norm: 1,
            norms_offset,
        },
        400,
        10,
    );
    assert!(sparse.sparse_region.is_some());
    assert!(sparse.cursor().norm_inverse(-1).is_err());
    assert!(sparse.cursor().field_length(-1).is_err());
}

/// `avg_field_length()` on the cursor is the field's, so a scoring loop
/// holding only a cursor needs nothing else.
#[test]
fn the_cursor_carries_its_fields_avg_field_length() {
    let n = FieldNorms::from_field_stats(&[], dense_entry(1, 1, 0), 1000, 40);
    assert_eq!(n.cursor().avg_field_length(), 25.0);
}

/// The point of the cursor split: `FieldNorms` stays `Sync` shared state so
/// `multi_segment.rs` can hand the same `&FieldNorms` to every `rayon`
/// leaf, and the mutable `IndexedDISI` position lives in a per-task cursor.
///
/// A `RefCell`-ed cursor inside `FieldNorms` -- the obvious smaller diff --
/// would have made the type non-`Sync` and broken that fan-out, so this
/// test proves the fan-out is still *actually concurrent*, not merely still
/// compiling: two rayon tasks each take their own cursor over one shared
/// `&FieldNorms` and neither may finish until both have started. If the
/// work had been serialized, the first task would spin until the timeout
/// and the assert would fire.
#[test]
fn a_shared_field_norms_still_fans_out_concurrently() {
    fn assert_sync<T: Sync>() {}
    fn assert_send<T: Send>() {}
    assert_sync::<FieldNorms<'_>>();
    assert_send::<FieldNormsCursor<'_, '_>>();

    if rayon::current_num_threads() < 2 {
        return; // single-threaded pool: nothing to prove
    }

    let present: Vec<i32> = (0..5000).map(|i| i * 2).collect();
    let (disi, disi_jumps) = lucene_codecs::indexed_disi::write(&present);
    let mut data = disi.clone();
    let norms_offset = data.len() as i64;
    data.extend((0..present.len()).map(|i| (i % 251 + 1) as u8));
    let entry = NormsEntry {
        field_number: 0,
        docs_with_field_offset: 0,
        docs_with_field_length: disi.len() as i64,
        jump_table_entry_count: disi_jumps,
        dense_rank_power: NO_RANK,
        num_docs_with_field: present.len() as i32,
        bytes_per_norm: 1,
        norms_offset,
    };
    let norms = FieldNorms::from_field_stats(&data, entry, 250_000, 5000);

    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let started = AtomicUsize::new(0);
    let shared: &FieldNorms<'_> = &norms;

    let both_ran: Vec<bool> = (0..2)
        .into_par_iter()
        .map(|_| {
            // Each task takes its own cursor over the shared norms.
            let mut cursor = shared.cursor();
            let mut acc = 0.0f32;
            for &doc in &present {
                acc += cursor.field_length(doc).unwrap();
            }
            assert!(acc > 0.0);

            started.fetch_add(1, Ordering::SeqCst);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while started.load(Ordering::SeqCst) < 2 {
                if std::time::Instant::now() > deadline {
                    return false;
                }
                std::hint::spin_loop();
            }
            true
        })
        .collect();

    assert!(
        both_ran.iter().all(|&ok| ok),
        "the two leaves did not run concurrently -- sharing `&FieldNorms` \
             across rayon has been serialized"
    );
}

/// An illegal `dense_rank_power` (anything outside 7..=15 and `0xFF`) is
/// metadata no Lucene writer emits, and must not be silently tolerated:
/// the DISI decode rejects it, so the once-decoded fast path declines and
/// the general path surfaces the error per lookup rather than inventing a
/// norm.
#[test]
fn an_illegal_dense_rank_power_is_rejected_rather_than_guessed() {
    let present: Vec<i32> = (0..5000).map(|i| i * 2).collect();
    let (disi, disi_jumps) = lucene_codecs::indexed_disi::write(&present);
    let mut data = disi.clone();
    let norms_offset = data.len() as i64;
    data.extend_from_slice(&vec![7u8; present.len()]);

    let entry = NormsEntry {
        field_number: 0,
        docs_with_field_offset: 0,
        docs_with_field_length: disi.len() as i64,
        jump_table_entry_count: disi_jumps,
        dense_rank_power: 0, // not 7..=15, not 0xFF
        num_docs_with_field: present.len() as i32,
        bytes_per_norm: 1,
        norms_offset,
    };
    let n = FieldNorms::from_field_stats(&data, entry, 250_000, 5000);
    assert!(
        n.sparse_region.is_some(),
        "the region's bounds are fine; it is the rank power that is not"
    );
    assert!(
        n.norm_inverse(0).is_err(),
        "an illegal rank power must surface as an error, not a fallback norm"
    );
    assert!(
        n.field_length(0).is_err(),
        "and on the length path too, not just the reciprocal one"
    );
}

/// A doc past the end of a dense field must still be the error the general
/// path raises, not a silent read of whatever byte follows the array.
#[test]
fn dense_fast_path_defers_out_of_range_docs_to_the_general_path() {
    let data = vec![5u8, 15u8, 25u8];
    let entry = dense_entry(1, 3, 0);
    let n = FieldNorms::from_field_stats(&data, entry, 45, 3);
    assert!(n.norm_inverse(3).is_err());
    assert!(n.norm_inverse(-1).is_err());
}

/// Shapes the fast path must decline: a wider norm, and a constant-valued
/// field (`bytes_per_norm == 0`, the value carried in `norms_offset`).
/// Declining is what keeps them correct, so assert it rather than assume.
#[test]
fn fast_path_declines_shapes_it_cannot_serve() {
    let data = vec![0u8; 16];
    assert!(
        FieldNorms::from_field_stats(&data, dense_entry(2, 4, 0), 40, 4)
            .dense_norm_bytes
            .is_none()
    );
    let constant = FieldNorms::from_field_stats(&data, dense_entry(0, 4, 7), 40, 4);
    assert!(constant.dense_norm_bytes.is_none());
    // ...and still scores, through the general path.
    assert_eq!(constant.norm_inverse(0).unwrap(), constant.norm_inverse[7]);
}

#[test]
fn avg_field_length_averages_decoded_norms() {
    // Three docs, raw norm bytes 5, 15, 25 -- bytes < 24 decode exactly
    // (subnormal range, see `lucene_util::small_float`), byte 25 decodes
    // to something a bit larger than 25.
    let data = vec![5u8, 15u8, 25u8];
    let entry = dense_entry(1, 3, 0);
    let fn_ = FieldNorms::open(&data, entry, 3, None).unwrap();
    let expected = (5.0 + 15.0 + lucene_util::small_float::byte4_to_int(25) as f32) / 3.0;
    assert!((fn_.avg_field_length - expected).abs() < 1e-4);
}

#[test]
fn avg_field_length_skips_dead_docs() {
    let data = vec![5u8, 15u8, 25u8];
    let entry = dense_entry(1, 3, 0);
    let mut live = FixedBitSet::new(3);
    live.set(0);
    live.set(2);
    // doc 1 (norm byte 15) is dead and must not affect the average.
    let fn_ = FieldNorms::open(&data, entry, 3, Some(&live)).unwrap();
    let expected = (5.0 + lucene_util::small_float::byte4_to_int(25) as f32) / 2.0;
    assert!((fn_.avg_field_length - expected).abs() < 1e-4);
}

#[test]
fn avg_field_length_falls_back_to_unnormed_when_no_live_docs_have_a_norm() {
    let data = vec![5u8, 15u8];
    let entry = dense_entry(1, 2, 0);
    let live = FixedBitSet::new(2); // nothing set -- all dead
    let fn_ = FieldNorms::open(&data, entry, 2, Some(&live)).unwrap();
    assert_eq!(
        fn_.avg_field_length,
        crate::similarity::UNNORMED_FIELD_LENGTH
    );
}

#[test]
fn field_length_decodes_one_doc() {
    let data = vec![5u8, 15u8, 25u8];
    let entry = dense_entry(1, 3, 0);
    let fn_ = FieldNorms::open(&data, entry, 3, None).unwrap();
    assert_eq!(fn_.field_length(0).unwrap(), 5.0);
    assert_eq!(fn_.field_length(1).unwrap(), 15.0);
    assert_eq!(
        fn_.field_length(2).unwrap(),
        lucene_util::small_float::byte4_to_int(25) as f32
    );
}

#[test]
fn field_length_falls_back_for_a_doc_with_no_norm() {
    // An empty field: no doc has a norm value at all.
    let mut entry = dense_entry(1, 0, 0);
    entry.docs_with_field_offset = -2; // empty
    let fn_ = FieldNorms::open(&[], entry, 3, None).unwrap();
    assert_eq!(
        fn_.field_length(0).unwrap(),
        crate::similarity::UNNORMED_FIELD_LENGTH
    );
}
