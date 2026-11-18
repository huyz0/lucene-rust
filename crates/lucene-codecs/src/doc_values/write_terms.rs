use super::*;

/// Port of `Lucene90DocValuesConsumer.addTermsDict` (block-LZ4-compressed
/// terms + block-address array) followed by `writeTermsIndex` (a coarser
/// sampled reverse index, sampled every
/// [`TERMS_DICT_REVERSE_INDEX_MASK`]`+1`th ordinal) -- the machinery
/// `SORTED`'s single-valued dictionary and `SORTED_SET`'s multi-valued one
/// share verbatim in real Lucene (`addSortedField`/`addSortedSetField` both
/// end by calling this same method). `terms` must already be sorted
/// ascending with no duplicates (the caller's job -- this port always builds
/// it from a `BTreeSet`/`sort_unstable+dedup` over the field's raw values).
///
/// Writes directly into `meta`/`data`, mirroring
/// [`write_dense_numeric_entry_body`]'s "body only, no leading
/// field-number/type byte" shape -- appends exactly what
/// [`terms_dict::read_term_dict_entry`] expects to read right after a
/// SORTED/SORTED_SET field's ords entry.
// ARITH: `terms` is sorted ascending with no duplicates -- the doc comment
// above states it and every caller builds it from a `BTreeSet` or a
// `sort_unstable` + `dedup`. That is what makes `later.len() - prefix_len`
// non-negative (a common prefix never exceeds either operand's length),
// `suffix_len - 1` safe (distinct sorted terms differ, so a later term has at
// least one byte past the common prefix), and `&term[..sort_key_len]` in
// range (`sampled_previous < term` and neither is a prefix of the other in
// the wrong direction). The two `debug_assert!`s below make `cargo test`
// exercise both claims rather than leaving them as prose. Everything else is
// a `Vec` length or an offset recorded from an earlier `data.len()`.
#[allow(clippy::arithmetic_side_effects)]
pub(super) fn write_terms_dict<T: AsRef<[u8]>>(
    meta: &mut Vec<u8>,
    data: &mut Vec<u8>,
    terms: &[T],
) {
    let terms: Vec<&[u8]> = terms.iter().map(AsRef::as_ref).collect();
    let size = terms.len() as i64;
    meta.write_vlong(size);
    meta.write_i32(TERMS_DICT_DIRECT_MONOTONIC_BLOCK_SHIFT as i32);

    let start = data.len() as i64;
    let mut max_length = 0i32;
    let mut max_block_length = 0i32;
    let mut block_addresses: Vec<i64> = Vec::new();
    // `termsDictBuffer`: the block's first term (the preset dictionary)
    // followed by the rest of the block, prefix-compressed. One hash table
    // for the whole field, as Java's `addTermsDict` keeps.
    let mut block_buf: Vec<u8> = Vec::new();
    let mut ht = lz4::FastCompressionHashTable::new();

    let mut ord = 0usize;
    while ord < terms.len() {
        let term = &terms[ord];
        block_addresses.push(data.len() as i64 - start);
        data.write_vint(term.len() as i32);
        data.extend_from_slice(term);
        max_length = max_length.max(term.len() as i32);

        // The rest of this 64-term block, prefix-compressed against its
        // immediate predecessor.
        block_buf.clear();
        block_buf.extend_from_slice(term);
        let dict_len = term.len();
        let block_end = terms
            .len()
            .min(ord + (TERMS_DICT_BLOCK_LZ4_MASK as usize) + 1);
        let mut prev: &[u8] = term;
        for later in &terms[ord + 1..block_end] {
            let prefix_len = common_prefix_len(prev, later);
            debug_assert!(
                prefix_len < later.len(),
                "write_terms_dict requires strictly ascending, deduplicated terms"
            );
            let suffix_len = later.len() - prefix_len;
            block_buf.push((prefix_len.min(15) as u8) | (((suffix_len - 1).min(15) as u8) << 4));
            if prefix_len >= 15 {
                block_buf.write_vint((prefix_len - 15) as i32);
            }
            if suffix_len >= 16 {
                block_buf.write_vint((suffix_len - 16) as i32);
            }
            block_buf.extend_from_slice(&later[prefix_len..]);
            max_length = max_length.max(later.len() as i32);
            prev = later;
        }

        let body_len = block_buf.len() - dict_len;
        if body_len > 0 {
            // `compressAndGetTermsDictBlockLength`: the first term is LZ4's
            // preset dictionary, so a suffix repeating bytes of it compresses
            // to a back-reference exactly where Java's does.
            data.write_vint(body_len as i32);
            lz4::compress_with_dictionary(&block_buf, 0, dict_len, body_len, data, &mut ht);
        }
        max_block_length = max_block_length.max(body_len as i32);
        ord = block_end;
    }

    // `DirectMonotonicWriter` flushes each completed block's meta entry as
    // soon as it fills (`DirectMonotonicWriter.flush`), i.e. interleaved
    // with the term loop above and always finished before `maxLength` is
    // written -- see `read_term_dict_entry`'s read order, which expects this
    // meta array right after `block_shift`, before `max_term_length`.
    let (addr_meta, addr_data) =
        direct_monotonic::write(&block_addresses, TERMS_DICT_DIRECT_MONOTONIC_BLOCK_SHIFT);
    meta.extend_from_slice(&addr_meta);

    meta.write_i32(max_length);
    meta.write_i32(max_block_length);
    meta.write_i64(start);
    meta.write_i64(data.len() as i64 - start);

    let addresses_start = data.len() as i64;
    data.extend_from_slice(&addr_data);
    meta.write_i64(addresses_start);
    meta.write_i64(data.len() as i64 - addresses_start);

    // `writeTermsIndex`: a coarser reverse index, one sampled address + a
    // short disambiguating prefix every `TERMS_DICT_REVERSE_INDEX_MASK + 1`
    // ordinals.
    meta.write_i32(TERMS_DICT_REVERSE_INDEX_SHIFT as i32);
    let index_start = data.len() as i64;
    let mut offset = 0i64;
    let mut index_addresses: Vec<i64> = Vec::new();
    let mut sampled_previous: Vec<u8> = Vec::new();
    for (ord, term) in terms.iter().enumerate() {
        let ord = ord as i64;
        if ord & TERMS_DICT_REVERSE_INDEX_MASK == 0 {
            index_addresses.push(offset);
            let sort_key_len = if ord == 0 {
                0
            } else {
                common_prefix_len(&sampled_previous, term) + 1
            };
            debug_assert!(
                sort_key_len <= term.len(),
                "write_terms_dict requires strictly ascending, deduplicated terms"
            );
            offset += sort_key_len as i64;
            data.extend_from_slice(&term[..sort_key_len]);
        } else if ord & TERMS_DICT_REVERSE_INDEX_MASK == TERMS_DICT_REVERSE_INDEX_MASK {
            sampled_previous.clear();
            sampled_previous.extend_from_slice(term);
        }
    }
    index_addresses.push(offset);

    // Same interleaving note as the block-address writer above: its meta
    // entries are already fully flushed by this point, so they can be
    // written now, before `indexOffset`/`indexLength`.
    let (index_addr_meta, index_addr_data) =
        direct_monotonic::write(&index_addresses, TERMS_DICT_DIRECT_MONOTONIC_BLOCK_SHIFT);
    meta.extend_from_slice(&index_addr_meta);

    meta.write_i64(index_start);
    meta.write_i64(data.len() as i64 - index_start);

    let index_addresses_start = data.len() as i64;
    data.extend_from_slice(&index_addr_data);
    meta.write_i64(index_addresses_start);
    meta.write_i64(data.len() as i64 - index_addresses_start);
}
