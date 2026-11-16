#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)]

use super::*;

#[test]
fn msb_vlong_reads_most_significant_group_first() {
    // 300 = 0b10_0101100: groups [0b10, 0b0101100] -> 0x82, 0x2c.
    let mut r = SliceInput::new(&[0x82, 0x2c]);
    assert_eq!(read_msb_vlong(&mut r).unwrap(), 300);
    let mut r = SliceInput::new(&[0xff; 11]);
    assert!(read_msb_vlong(&mut r).is_err());
}

#[test]
fn frame_data_decodes_in_both_encodings() {
    // fp 5, has terms, floor, then two floor bytes.
    let code = (5 << 2) | 0x3;
    let v0 = decode_frame_data(&[code as u8, 0x01, 0x61], VERSION_START).unwrap();
    assert_eq!(
        v0,
        FstBlock {
            fp: 5,
            has_terms: true,
            floor_start: Some(1),
        }
    );
    let v1 = decode_frame_data(&[code as u8], VERSION_MSB_VLONG_OUTPUT).unwrap();
    assert_eq!(v1.floor_start, Some(1));
    let plain = decode_frame_data(&[5 << 2], VERSION_CURRENT).unwrap();
    assert!(!plain.has_terms);
    assert!(plain.floor_start.is_none());
    // The two encodings of a two-group code differ: 300 is LSB-first
    // [0xac, 0x02] and MSB-first [0x82, 0x2c].
    assert_eq!(
        decode_frame_data(&[0xac, 0x02], VERSION_START).unwrap().fp,
        75
    );
    assert_eq!(
        decode_frame_data(&[0x82, 0x2c], VERSION_CURRENT)
            .unwrap()
            .fp,
        75
    );
    // A vlong that decodes negative is not a block pointer.
    let neg = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01];
    assert!(decode_frame_data(&neg, VERSION_START).is_err());
}

/// The 9.0.0 fixture's first segment (`Lucene90` postings, block tree
/// version 0): `(tim, tip, tmd, fnm, id, suffix)`.
fn fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>, FieldInfos, [u8; 16], String) {
    let dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/bwc/9.0.0");
    let read = |n: &str| std::fs::read(dir.join(n)).unwrap();
    let tim = read("_0_Lucene90_0.tim");
    // The id every index header carries, after magic, name and version.
    let name_len = tim[4] as usize;
    let mut id = [0u8; 16];
    id.copy_from_slice(&tim[9 + name_len..25 + name_len]);
    let fnm = crate::field_infos::parse(&read("_0.fnm"), &id, "").unwrap();
    (
        tim,
        read("_0_Lucene90_0.tip"),
        read("_0_Lucene90_0.tmd"),
        fnm,
        id,
        "Lucene90_0".to_string(),
    )
}

fn open_fixture(
    tim: &[u8],
    tip: &[u8],
    tmd: &[u8],
    fnm: &FieldInfos,
    id: &[u8; 16],
    suffix: &str,
) -> Result<BlockTreeFields> {
    blocktree::open(tim, tip, tmd, fnm, id, suffix, 3000)
}

#[test]
fn a_real_lucene90_dictionary_opens_through_the_common_entry_point() {
    let (tim, tip, tmd, fnm, id, suffix) = fixture();
    let fields = open_fixture(&tim, &tip, &tmd, &fnm, &id, &suffix).unwrap();
    let body = fields.field("body").unwrap();
    assert_eq!(body.postings_format(), PostingsFormat::Lucene90);
    assert_eq!(body.num_terms, 20);
    assert!(body.try_seek_exact(b"zeta").unwrap().is_some());
    assert!(body.try_seek_exact(b"zzz").unwrap().is_none());
}

/// The index is read in place: every field keeps its FST in `.tip`,
/// and a clone shares it.
#[test]
fn the_fst_index_is_read_in_place() {
    let (tim, tip, tmd, fnm, id, suffix) = fixture();
    let fields = open_fixture(&tim, &tip, &tmd, &fnm, &id, &suffix).unwrap();
    for (name, field) in fields.iter_fields() {
        let debug = format!("{field:?}");
        assert!(debug.contains("index: Fst(field "), "{name}: {debug}");
    }
    let body = fields.field("body").unwrap();
    let copy = body.clone();
    assert!(copy.try_seek_exact(b"zeta").unwrap().is_some());
    assert!(copy.try_seek_exact(b"zzz").unwrap().is_none());
}

/// Every term of every field is found by `seek_exact` and by
/// `seek_ceil` on itself, and a key just past each one ceils to the
/// next term -- the FST walk sends every seek to the block the full
/// enumeration found it in.
#[test]
fn every_term_is_found_by_walking_the_fst() {
    let (tim, tip, tmd, fnm, id, suffix) = fixture();
    let fields = open_fixture(&tim, &tip, &tmd, &fnm, &id, &suffix).unwrap();
    let mut seen = 0;
    for (name, field) in fields.iter_fields() {
        let mut all = Vec::new();
        let mut it = field.iter();
        while let Some(t) = it.try_next_term().unwrap() {
            all.push(t.to_vec());
        }
        assert_eq!(all.len() as i64, field.num_terms, "{name}");
        for (i, t) in all.iter().enumerate() {
            assert!(field.try_seek_exact(t).unwrap().is_some(), "{name} {t:?}");
            let mut e = field.iter();
            assert_eq!(
                e.try_seek_ceil(t).unwrap(),
                blocktree::SeekStatus::Found,
                "{name}"
            );
            let mut past = t.clone();
            past.push(0);
            let mut e = field.iter();
            let status = e.try_seek_ceil(&past).unwrap();
            match all.get(i + 1) {
                Some(next) => {
                    assert_eq!(status, blocktree::SeekStatus::NotFound, "{name}");
                    assert_eq!(e.term(), Some(next.as_slice()), "{name}");
                }
                None => assert_eq!(status, blocktree::SeekStatus::End, "{name}"),
            }
            seen += 1;
        }
    }
    assert!(seen > 100, "{seen}");
}

/// A corrupt FST body is met by the lookups that reach it, as
/// Lucene's off-heap FST meets it at a seek (the checksum is
/// `CheckIndex`'s to verify): each one either fails -- an FST error
/// naming the field's terms index -- or misses, and none panics.
#[test]
fn a_corrupt_fst_body_is_met_by_the_lookups_that_reach_it() {
    let (tim, clean_tip, tmd, fnm, id, suffix) = fixture();
    let clean = open_fixture(&tim, &clean_tip, &tmd, &fnm, &id, &suffix).unwrap();
    let mut terms: Vec<(String, Vec<u8>)> = Vec::new();
    for (name, field) in clean.iter_fields() {
        let mut it = field.iter();
        while let Some(t) = it.try_next_term().unwrap() {
            terms.push((name.to_string(), t.to_vec()));
        }
    }
    let header = codec_util::index_header_length(TERMS_INDEX_CODEC_NAME, &suffix);
    let footer = clean_tip.len() - 16;
    for fill in [0x00u8, 0xff, 0x5a] {
        let mut tip = clean_tip.clone();
        for (i, b) in tip[header..footer].iter_mut().enumerate() {
            *b = if fill == 0x5a {
                (i * 31 % 251) as u8
            } else {
                fill
            };
        }
        let fields = open_fixture(&tim, &tip, &tmd, &fnm, &id, &suffix).unwrap();
        let (mut failed, mut missed) = (0, 0);
        for (name, term) in &terms {
            let field = fields.field(name).unwrap();
            match field.try_seek_exact(term) {
                Ok(Some(_)) => {}
                Ok(None) => missed += 1,
                Err(e) => {
                    let msg = e.to_string();
                    if msg.contains("FST") {
                        assert!(msg.contains("terms index"), "{name}: {msg}");
                    }
                    failed += 1;
                }
            }
        }
        assert!(
            failed + missed > 0,
            "fill {fill:#x}: no lookup met the corrupt bytes"
        );
    }
}

/// A node's frame data, pushed where no block is, is an error rather
/// than a frame at a made-up fp; the root's is `rootCode`.
#[test]
fn a_prefix_without_a_block_has_no_frame_data() {
    let index = FstTermsIndex {
        field: "f".to_string(),
        tip: Arc::new(vec![0u8; 4]),
        body: 0..0,
        metadata: FstMetadata {
            input_type: fst::InputType::Byte1,
            empty_output: None,
            start_node: 0,
            version: 8,
            num_bytes: 0,
        },
        root_code: vec![5 << 2],
        version: VERSION_CURRENT,
    };
    let root = index.root();
    assert!(root.has_block());
    assert_eq!(index.block_fp(&root).unwrap(), Some(5));
    // The start node has no arcs.
    assert!(index.child(&root, b'a').unwrap().is_none());
    let inner = FstNode {
        arc: fst::Arc::default(),
        output: Vec::new(),
        root: false,
    };
    assert!(!inner.has_block());
    assert_eq!(index.block_fp(&inner).unwrap(), None);
    let err = index.block(&inner, &mut Vec::new()).unwrap_err();
    assert!(err.to_string().contains("names no block"), "{err}");
    assert!(format!("{index:?}").contains("field \"f\""));
}

#[test]
fn a_field_naming_a_trie_format_is_rejected() {
    let (tim, tip, tmd, mut fnm, id, suffix) = fixture();
    for f in &mut fnm.fields {
        for (k, v) in &mut f.attributes {
            if k == "PerFieldPostingsFormat.format" {
                *v = "Lucene104".to_string();
            }
        }
    }
    let err = open_fixture(&tim, &tip, &tmd, &fnm, &id, &suffix).unwrap_err();
    assert!(
        err.to_string().contains("names no postings format"),
        "{err}"
    );
}

#[test]
fn truncated_or_mismatched_files_are_errors() {
    let (tim, tip, tmd, fnm, id, suffix) = fixture();
    // `.tmd` cut short: the field records run out.
    assert!(open_fixture(&tim, &tip, &tmd[..tmd.len() / 2], &fnm, &id, &suffix).is_err());
    // `.tip` of the wrong length fails `retrieveChecksum`'s length check.
    let mut long_tip = tip.clone();
    long_tip.insert(40, 0);
    assert!(open_fixture(&tim, &long_tip, &tmd, &fnm, &id, &suffix).is_err());
    // A different segment's id fails every header.
    let other = [7u8; 16];
    assert!(open_fixture(&tim, &tip, &tmd, &fnm, &other, &suffix).is_err());
}

/// `.tmd`'s field records checked as `Lucene90BlockTreeTermsReader`
/// checks them, each value re-encoded in place: an index block size
/// other than 128, a field count the file cannot hold, a field with no
/// terms, a doc count past `maxDoc`, sums smaller than what they sum.
#[test]
fn field_records_out_of_range_are_corrupt() {
    let (tim, tip, tmd, fnm, id, suffix) = fixture();
    let vint = |out: &mut Vec<u8>, v: i64| {
        let mut v = v as u64;
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    };
    // Where each value of the first field record starts and ends.
    let mut input = SliceInput::new(&tmd);
    let version = codec_util::check_index_header(
        &mut input,
        TERMS_META_CODEC_NAME,
        VERSION_START,
        VERSION_CURRENT,
        &id,
        &suffix,
    )
    .unwrap()
    .version;
    assert!(version >= VERSION_START);
    codec_util::check_index_header(
        &mut input,
        POSTINGS_TERMS_CODEC,
        0,
        POSTINGS_TERMS_VERSION_CURRENT,
        &id,
        &suffix,
    )
    .unwrap();
    type Spans = Vec<(usize, usize, i64)>;
    fn span(
        spans: &mut Spans,
        input: &mut SliceInput<'_>,
        read: &dyn Fn(&mut SliceInput<'_>) -> i64,
    ) {
        let start = input.position();
        let v = read(input);
        spans.push((start, input.position(), v));
    }
    let mut spans = Spans::new();
    let vi = |i: &mut SliceInput<'_>| i64::from(i.read_vint().unwrap());
    let vl = |i: &mut SliceInput<'_>| i.read_vlong().unwrap();
    span(&mut spans, &mut input, &vi); // 0: block size
    span(&mut spans, &mut input, &vi); // 1: field count
    span(&mut spans, &mut input, &vi); // 2: field number
    span(&mut spans, &mut input, &vl); // 3: term count
    blocktree::read_bytes_ref(&mut input).unwrap();
    let field_number = spans[2].2 as i32;
    let info = fnm.field_by_number(field_number).unwrap();
    let has_freqs = info.index_options != crate::field_infos::IndexOptions::Docs;
    if has_freqs {
        span(&mut spans, &mut input, &vl); // 4: sumTotalTermFreq
    }
    span(&mut spans, &mut input, &vl); // sumDocFreq
    span(&mut spans, &mut input, &vi); // docCount
    let n = spans.len();
    let (sum_df, doc_count) = (n - 2, n - 1);
    let patched = |at: usize, v: i64| {
        let (start, end, _) = spans[at];
        let mut out = tmd[..start].to_vec();
        vint(&mut out, v);
        out.extend_from_slice(&tmd[end..]);
        out
    };
    let err = |bytes: &[u8]| {
        open_fixture(&tim, &tip, bytes, &fnm, &id, &suffix)
            .unwrap_err()
            .to_string()
    };
    assert!(err(&patched(0, 64)).contains("64"));
    let too_many = err(&patched(1, 1 << 20));
    assert!(too_many.contains(&(1 << 20).to_string()), "{too_many}");
    let no_terms = err(&patched(3, 0));
    assert!(no_terms.contains(&field_number.to_string()), "{no_terms}");
    assert!(err(&patched(doc_count, 3001)).contains("3001"));
    let docs = spans[doc_count].2;
    assert!(err(&patched(sum_df, docs - 1)).contains(&(docs - 1).to_string()));
    if has_freqs {
        let sdf = spans[sum_df].2;
        assert!(err(&patched(4, sdf - 1)).contains(&(sdf - 1).to_string()));
    }
}
