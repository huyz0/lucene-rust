#![cfg(test)]
// The arithmetic gate is about values read off disk; a fixture builder's
// `i + 1` is not one. See `docs/arithmetic-gate.md`, "Test code".
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::field_infos::FieldInfo;
use lucene_store::data_output::DataOutput;

/// Test-only: every child of `node`, found by probing
/// [`lookup_child`] with all 256 labels.
///
/// Production never enumerates a trie node's children -- a seek walks one
/// label at a time down the target term's own bytes -- but the structural
/// tests below want the whole subtree, and probing every label exercises
/// each `ChildSaveStrategy`'s hit *and* miss paths rather than a
/// separate "list all" decoder that production would not use.
fn trie_children(slice: &[u8], node: &TrieNode) -> Result<Vec<(u8, TrieNode)>> {
    let mut out = Vec::new();
    for label in 0..=u8::MAX {
        if let Some(child) = lookup_child(slice, node, label)? {
            out.push((label, child));
        }
    }
    Ok(out)
}

/// Test-only: every physical `.tim` block one trie node's output
/// addresses -- the base block plus each follow-on floor sub-block --
/// driven through the production [`Frame::scan_to_floor_frame`] one
/// boundary label at a time. Production only ever asks for the *one*
/// block a target term falls in.
fn expand_floor(
    index: &[u8],
    base_fp: u64,
    base_has_terms: bool,
    floor_data_fp: Option<usize>,
) -> Result<Vec<(u64, bool)>> {
    let Some(fdp) = floor_data_fp else {
        return Ok(vec![(base_fp, base_has_terms)]);
    };
    let mut f = Frame {
        fp: base_fp as usize,
        fp_orig: base_fp as usize,
        has_terms: base_has_terms,
        has_terms_orig: base_has_terms,
        is_floor: true,
        next_ent: -1,
        ..Frame::default()
    };
    f.set_floor_data(index, fdp)?;
    let mut out = Vec::new();
    loop {
        out.push((f.fp as u64, f.has_terms));
        if f.next_floor_label > u8::MAX as u32 {
            return Ok(out);
        }
        let label = f.next_floor_label as u8;
        f.scan_to_floor_frame(index, &[label])?;
    }
}

/// Test-only: every physical `.tim` block reachable from `node`, paired
/// with the trie label path that led to it. Lets the fixture tests assert
/// on a field's whole block structure; `open` loads blocks lazily instead.
fn collect_leaf_blocks(
    slice: &[u8],
    node: &TrieNode,
    depth: u32,
    prefix: &mut Vec<u8>,
    out: &mut Vec<(u64, Vec<u8>)>,
) -> Result<()> {
    if depth > 10_000 {
        return Err(Error::Unsupported("trie nesting too deep (possible cycle)"));
    }
    if let Some(fp) = node.output_fp {
        for (block_fp, has_terms) in expand_floor(slice, fp, node.has_terms, node.floor_data_fp)? {
            // `hasTerms == false` means the block holds nothing but
            // pointers to further-nested sub-blocks.
            if has_terms {
                out.push((block_fp, prefix.clone()));
            }
        }
    }
    for (label, child) in trie_children(slice, node)? {
        prefix.push(label);
        let r = collect_leaf_blocks(slice, &child, depth + 1, prefix, out);
        prefix.pop();
        r?;
    }
    Ok(())
}

/// `DataOutput::write_vlong` without its non-negative assertion: the
/// ten-group encoding a corrupt file can legally contain.
fn write_vlong_allowing_negative(out: &mut Vec<u8>, v: i64) {
    let mut v = v as u64;
    while v & !0x7F != 0 {
        out.push(((v & 0x7F) as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn field_info(number: i32, name: &str, index_options: IndexOptions) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options,
        doc_values_type: crate::field_infos::DocValuesType::None,
        doc_values_skip_index_type: crate::field_infos::DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: Vec::new(),
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: crate::field_infos::VectorEncoding::Float32,
        vector_similarity_function: crate::field_infos::VectorSimilarityFunction::Euclidean,
    }
}

/// Hand-builds a single-field, single-block `.tim`/`.tip`/`.tmd` triple
/// (terms `["a", "ab", "b"]`, docFreq/totalTermFreq = 1/1, 2/3, 1/1) —
/// this port's own encoder, test-only, to exercise error/boundary paths
/// a real (small) fixture never reaches. Mirrors the pattern used by
/// `codec_util.rs`/`segment_info.rs`'s own test-only encoders.
struct Builder {
    id: [u8; ID_LENGTH],
    suffix: String,
    ov: Overrides,
}

/// Deliberate corruptions [`Builder`] can bake into an otherwise
/// well-formed, correctly-footered triple, so a test can aim one header
/// field at a decoder path without hand-assembling the whole file.
#[derive(Default, Clone)]
struct Overrides {
    /// `.tmd` `numFields`.
    num_fields: Option<i32>,
    /// `.tmd` `minTerm`'s length prefix (the bytes written stay the
    /// real ones).
    min_term_len: Option<i32>,
    /// `.tmd` `rootFP`.
    root_fp: Option<i64>,
    /// The per-term `totalTermFreq - docFreq` vlong in the `.tim` stats
    /// region, for every term.
    ttf_delta: Option<i64>,
    /// The `.tim` block header's `entCount`.
    ent_count: Option<u32>,
    /// `(declared region length, Some(byte) => write it as `allEqual`)`
    /// for the `.tim` suffix-lengths region.
    suffix_lengths_region: Option<(usize, Option<u8>)>,
}

impl Builder {
    fn new() -> Self {
        Builder {
            id: [7u8; ID_LENGTH],
            suffix: String::new(),
            ov: Overrides::default(),
        }
    }

    fn build(
        &self,
        index_options: IndexOptions,
        terms: &[(&str, u32, u64)],
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        // .tim
        let mut tim = Vec::new();
        codec_util::write_index_header(
            &mut tim,
            TERMS_CODEC_NAME,
            VERSION_CURRENT,
            &self.id,
            &self.suffix,
        );
        let block_fp = tim.len();

        let ent_count = self.ov.ent_count.unwrap_or(terms.len() as u32);
        let code = (ent_count << 1) | 1; // isLastInFloor
        tim.write_vint(code as i32);

        let mut suffix_bytes = Vec::new();
        let mut suffix_lengths = Vec::new();
        let mut stats = Vec::new();
        for (term, doc_freq, total_term_freq) in terms {
            suffix_bytes.extend_from_slice(term.as_bytes());
            suffix_lengths.write_vint(term.len() as i32);
            let token = (*doc_freq as i32) << 1; // never singleton-run-encoded, for test simplicity
            stats.write_vint(token);
            if index_options != IndexOptions::Docs {
                let delta = self
                    .ov
                    .ttf_delta
                    .unwrap_or((*total_term_freq as i64) - (*doc_freq as i64));
                stats.write_vlong(delta);
            }
        }

        let code_l = ((suffix_bytes.len() as u64) << 3) | 0x04; // isLeafBlock, NO_COMPRESSION
        tim.write_vlong(code_l as i64);
        tim.write_bytes(&suffix_bytes);

        match self.ov.suffix_lengths_region {
            None => {
                tim.write_vint((suffix_lengths.len() as i32) << 1); // not allEqual
                tim.write_bytes(&suffix_lengths);
            }
            Some((declared, None)) => {
                tim.write_vint((declared as i32) << 1);
                tim.write_bytes(&suffix_lengths);
            }
            Some((declared, Some(byte))) => {
                tim.write_vint(((declared as i32) << 1) | 1);
                tim.push(byte);
            }
        }

        tim.write_vint(stats.len() as i32);
        tim.write_bytes(&stats);

        // Postings metadata: one entry per term via the bit=0
        // (docStartFP-delta) branch, legal regardless of `absolute` --
        // these seek_exact-only tests don't exercise postings decode, so
        // the fake docStartFP/singletonDocID values are never read back.
        let mut meta = Vec::new();
        for (_, doc_freq, _) in terms {
            meta.write_vlong(10 << 1);
            if *doc_freq == 1 {
                meta.write_vint(0);
            }
        }
        tim.write_vint(meta.len() as i32);
        tim.write_bytes(&meta);

        codec_util::write_footer(&mut tim);

        // .tip: root node (SIGN_NO_CHILDREN), hasTerms, no floor.
        let mut tip = Vec::new();
        codec_util::write_index_header(
            &mut tip,
            TERMS_INDEX_CODEC_NAME,
            VERSION_CURRENT,
            &self.id,
            &self.suffix,
        );
        let index_start = tip.len();
        let root_fp = 0usize;
        let output_fp_bytes = 8usize; // keep it simple: always 8 bytes
        let header = (SIGN_NO_CHILDREN as u8)
            | ((output_fp_bytes as u8 - 1) << 2)
            | (LEAF_NODE_HAS_TERMS as u8);
        tip.push(header);
        tip.extend_from_slice(&(block_fp as u64).to_le_bytes());
        tip.extend_from_slice(&0u64.to_le_bytes()); // 8-byte over-read pad
        let index_end = tip.len();
        codec_util::write_footer(&mut tip);

        // .tmd
        let mut tmd = Vec::new();
        codec_util::write_index_header(
            &mut tmd,
            TERMS_META_CODEC_NAME,
            VERSION_CURRENT,
            &self.id,
            &self.suffix,
        );
        codec_util::write_index_header(
            &mut tmd,
            POSTINGS_TERMS_CODEC,
            VERSION_CURRENT,
            &self.id,
            &self.suffix,
        );
        tmd.write_vint(POSTINGS_BLOCK_SIZE);

        tmd.write_vint(self.ov.num_fields.unwrap_or(1)); // numFields
        tmd.write_vint(0); // field number
        let num_terms = terms.len() as i64;
        tmd.write_vlong(num_terms);
        let sum_doc_freq: i64 = terms.iter().map(|(_, d, _)| *d as i64).sum();
        let sum_total_term_freq: i64 = if index_options == IndexOptions::Docs {
            sum_doc_freq
        } else {
            terms.iter().map(|(_, _, t)| *t as i64).sum()
        };
        if index_options != IndexOptions::Docs {
            tmd.write_vlong(sum_total_term_freq);
        }
        tmd.write_vlong(sum_doc_freq);
        tmd.write_vint(1); // docCount
        let min_term = terms[0].0.as_bytes();
        let max_term = terms[terms.len() - 1].0.as_bytes();
        tmd.write_vint(self.ov.min_term_len.unwrap_or(min_term.len() as i32));
        tmd.write_bytes(min_term);
        tmd.write_vint(max_term.len() as i32);
        tmd.write_bytes(max_term);
        tmd.write_vlong(index_start as i64);
        match self.ov.root_fp {
            // `DataOutput::write_vlong` refuses a negative value (Java's
            // `writeVLong` asserts the same), but the *wire* format can
            // carry one in its tenth group -- and that is exactly what a
            // corrupt `.tmd` looks like to `as usize`.
            Some(v) => write_vlong_allowing_negative(&mut tmd, v),
            None => tmd.write_vlong(root_fp as i64),
        }
        tmd.write_vlong(index_end as i64);

        tmd.write_i64(tip.len() as i64); // indexLength
        tmd.write_i64((tim.len()) as i64); // termsLength
        codec_util::write_footer(&mut tmd);

        (tim, tip, tmd)
    }
}

#[test]
fn seek_exact_found_and_not_found() {
    let b = Builder::new();
    let (tim, tip, tmd) = b.build(
        IndexOptions::DocsAndFreqs,
        &[("a", 1, 1), ("ab", 2, 3), ("b", 1, 1)],
    );
    let fis = FieldInfos {
        fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
    };
    let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
    let field = fields.field("text").unwrap();
    assert_eq!(field.num_terms, 3);
    assert_eq!(field.sum_doc_freq, 4);
    assert_eq!(field.sum_total_term_freq, 5);
    assert_eq!(field.min_term, b"a");
    assert_eq!(field.max_term, b"b");

    assert_eq!(
        field.seek_exact(b"ab"),
        Some(TermStats {
            doc_freq: 2,
            total_term_freq: 3
        })
    );
    assert_eq!(
        field.seek_exact(b"a"),
        Some(TermStats {
            doc_freq: 1,
            total_term_freq: 1
        })
    );
    assert_eq!(field.seek_exact(b"missing"), None);
    assert_eq!(field.seek_exact(b""), None);
    assert!(fields.field("nope").is_none());
    // The prefetch hint names the one block, from the index alone.
    let block = field.term_block(b"ab").unwrap().unwrap();
    assert_eq!(block.len(), 1);
    assert_eq!(field.term_block(b"a").unwrap(), Some(block));
    assert_eq!(field.term_block(b"zzz").unwrap(), None);
}

#[test]
fn intersect_prefix_and_wildcard_over_materialized_field() {
    let b = Builder::new();
    let (tim, tip, tmd) = b.build(
        IndexOptions::DocsAndFreqs,
        &[
            ("apple", 1, 1),
            ("application", 1, 1),
            ("apply", 1, 1),
            ("banana", 1, 1),
            ("band", 1, 1),
        ],
    );
    let fis = FieldInfos {
        fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
    };
    let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
    let field = fields.field("text").unwrap();

    // Prefix: "app*" -> apple, application, apply (in sorted order).
    let pattern = WildcardPattern::new(b"app*");
    let got: Vec<Vec<u8>> = field.intersect(&pattern).map(|r| r.unwrap().0).collect();
    assert_eq!(
        got,
        vec![
            b"apple".to_vec(),
            b"application".to_vec(),
            b"apply".to_vec()
        ]
    );

    // "?" wildcard: "ban?" matches nothing here ("band" is 4 bytes so
    // "ban?" matches "band" exactly -- exercise it precisely).
    let pattern = WildcardPattern::new(b"ban?");
    let got: Vec<Vec<u8>> = field.intersect(&pattern).map(|r| r.unwrap().0).collect();
    assert_eq!(got, vec![b"band".to_vec()]);

    // No literal prefix ("*" in the middle only): "*ana*" -> banana.
    let pattern = WildcardPattern::new(b"*ana*");
    let got: Vec<Vec<u8>> = field.intersect(&pattern).map(|r| r.unwrap().0).collect();
    assert_eq!(got, vec![b"banana".to_vec()]);

    // Matches everything.
    let pattern = WildcardPattern::new(b"*");
    assert_eq!(field.intersect(&pattern).count(), 5);

    // Matches nothing: valid prefix range, no candidate satisfies the
    // rest of the pattern.
    let pattern = WildcardPattern::new(b"app??????");
    assert_eq!(field.intersect(&pattern).count(), 0);

    // Matches nothing: prefix outside the field's term range entirely.
    let pattern = WildcardPattern::new(b"zzz*");
    assert_eq!(field.intersect(&pattern).count(), 0);

    // Exact-match pattern (no wildcard bytes at all) behaves like
    // seek_exact.
    let pattern = WildcardPattern::new(b"banana");
    let got: Vec<Vec<u8>> = field.intersect(&pattern).map(|r| r.unwrap().0).collect();
    assert_eq!(got, vec![b"banana".to_vec()]);

    // PrefixQuery-shaped constructor.
    let pattern = WildcardPattern::prefix(b"ban");
    let got: Vec<Vec<u8>> = field.intersect(&pattern).map(|r| r.unwrap().0).collect();
    assert_eq!(got, vec![b"banana".to_vec(), b"band".to_vec()]);
}

/// Backs `FuzzyQuery::max_expansions`'s (`lucene-search`) selection-policy
/// claim: `fuzzy_intersect` returns a lazy iterator over this field's
/// already-fully-decoded, in-memory sorted entries (the whole term
/// dictionary is materialized once at segment open, not decoded on demand
/// here -- see this module's own doc comment), filtering one candidate at a
/// time as it's pulled. `.take(n)` stops pulling once `n` matches are
/// found, so it skips the fuzzy-match predicate/allocation for every entry
/// past the cap, but it does not skip any term-dictionary decode work (that
/// already happened). This test builds a term dictionary with more than 50
/// terms that all match the same fuzzy pattern (all single-byte
/// substitutions of a fixed-length target, so every one is within
/// `max_edits` of it) and checks that `.take(50)` yields exactly the first
/// 50 in sorted order, while the untruncated iterator yields every one of
/// them.
#[test]
fn fuzzy_intersect_take_truncates_the_lazy_walk() {
    let b = Builder::new();
    // 60 five-byte terms "aaaaa".."aaaaz"-ish, each one substitution away
    // from "aaaaa" in its last byte -- all within max_edits=1 of it, and
    // already in sorted order since they differ only in their last byte.
    // Offsets 0..60 land in ASCII 33..93 ('!'..']'), all single-byte
    // printable characters, so the resulting `String`s stay valid UTF-8
    // and byte-sort exactly like their trailing byte value.
    let mut owned_terms: Vec<String> = (0u8..60)
        .map(|i| format!("aaaa{}", (33 + i) as char))
        .collect();
    owned_terms.sort();
    let terms: Vec<(&str, u32, u64)> = owned_terms.iter().map(|t| (t.as_str(), 1, 1)).collect();
    let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, &terms);
    let fis = FieldInfos {
        fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
    };
    let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
    let field = fields.field("text").unwrap();

    let pattern = FuzzyMatch::new(b"aaaaa", 1, 0, true);

    // Untruncated: every one of the 60 terms matches.
    assert_eq!(field.fuzzy_intersect(&pattern).count(), 60);

    // Truncated to `max_expansions`-shaped 50: exactly 50 matches, and
    // they are the first 50 in sorted term-dictionary order (not an
    // arbitrary/unstable 50).
    let capped: Vec<Vec<u8>> = field
        .fuzzy_intersect(&pattern)
        .take(50)
        .map(|r| r.unwrap().0)
        .collect();
    assert_eq!(capped.len(), 50);
    let expected: Vec<Vec<u8>> = owned_terms[..50]
        .iter()
        .map(|t| t.as_bytes().to_vec())
        .collect();
    assert_eq!(capped, expected);

    // A cap that never binds (more than the total match count) is a
    // no-op, same "fewer matches than the limit" regression-safety shape
    // the task requires.
    assert_eq!(field.fuzzy_intersect(&pattern).take(1000).count(), 60);
}

#[test]
fn regexp_intersect_over_materialized_field() {
    let b = Builder::new();
    let (tim, tip, tmd) = b.build(
        IndexOptions::DocsAndFreqs,
        &[
            ("apple", 1, 1),
            ("application", 1, 1),
            ("apply", 1, 1),
            ("banana", 1, 1),
            ("band", 1, 1),
        ],
    );
    let fis = FieldInfos {
        fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
    };
    let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
    let field = fields.field("text").unwrap();

    // Literal-prefix-narrowed range: "appl.*" -> apple, application,
    // apply (in sorted order).
    let pattern = RegexpPattern::new(b"appl.*").unwrap();
    let got: Vec<Vec<u8>> = field
        .regexp_intersect(&pattern)
        .map(|r| r.unwrap().0)
        .collect();
    assert_eq!(
        got,
        vec![
            b"apple".to_vec(),
            b"application".to_vec(),
            b"apply".to_vec()
        ]
    );

    // Alternation has no useful literal prefix (falls back to a full
    // scan) but still matches correctly.
    let pattern = RegexpPattern::new(b"banana|band").unwrap();
    let got: Vec<Vec<u8>> = field
        .regexp_intersect(&pattern)
        .map(|r| r.unwrap().0)
        .collect();
    assert_eq!(got, vec![b"banana".to_vec(), b"band".to_vec()]);

    // Whole-term-match: "ban" alone matches neither "banana" nor "band".
    let pattern = RegexpPattern::new(b"ban").unwrap();
    assert_eq!(field.regexp_intersect(&pattern).count(), 0);

    // Matches nothing: prefix outside the field's term range entirely.
    let pattern = RegexpPattern::new(b"zzz.*").unwrap();
    assert_eq!(field.regexp_intersect(&pattern).count(), 0);
}

/// The dead-prefix skip must never change *which* terms come back, only
/// how many are tested. Checked against a brute-force filter over a
/// dictionary shaped like the search benchmark's (`t0`..`t2999`), on the
/// interior-constrained pattern family the skip exists for.
#[test]
fn regexp_intersect_skip_agrees_with_a_brute_force_scan() {
    let terms: Vec<String> = (0..3000).map(|i| format!("t{i}")).collect();
    let mut sorted: Vec<&str> = terms.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    let entries: Vec<(&str, u32, u64)> = sorted.iter().map(|t| (*t, 1u32, 1u64)).collect();

    let b = Builder::new();
    let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, &entries);
    let fis = FieldInfos {
        fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
    };
    let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
    let field = fields.field("text").unwrap();

    for src in [
        "t1[0-9]",
        "t1.",
        "t[0-9]{3}",
        "t.*9",
        "t1|t22|t333",
        "t9*",
        "t(1|2)[0-9][0-9]",
        "zzz.*",
        "t1[0-9]&t.*",
        "t<1-30>",
    ] {
        let pattern = RegexpPattern::new(src.as_bytes()).unwrap();
        let expected: Vec<&str> = sorted
            .iter()
            .copied()
            .filter(|t| pattern.matches(t.as_bytes()))
            .collect();
        let got: Vec<String> = field
            .regexp_intersect(&pattern)
            .map(|r| String::from_utf8(r.unwrap().0).unwrap())
            .collect();
        let expected: Vec<String> = expected.iter().map(|t| t.to_string()).collect();
        assert_eq!(got, expected, "pattern {src}");
    }
}

#[test]
fn prefix_upper_bound_handles_ff_bytes_and_empty() {
    assert_eq!(prefix_upper_bound(b""), None);
    assert_eq!(prefix_upper_bound(&[0xFF]), None);
    assert_eq!(prefix_upper_bound(&[0xFF, 0xFF]), None);
    assert_eq!(prefix_upper_bound(b"a"), Some(b"b".to_vec()));
    assert_eq!(prefix_upper_bound(&[b'a', 0xFF]), Some(vec![b'b']));
    assert_eq!(prefix_upper_bound(b"app"), Some(b"apq".to_vec()));
}

#[test]
fn single_term_field() {
    let b = Builder::new();
    let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("only", 1, 1)]);
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::Docs)],
    };
    let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
    let field = fields.field("f").unwrap();
    assert_eq!(field.min_term, field.max_term);
    assert_eq!(
        field.seek_exact(b"only"),
        Some(TermStats {
            doc_freq: 1,
            total_term_freq: 1
        })
    );
    assert_eq!(field.seek_exact(b"other"), None);
}

#[test]
fn terms_enum_next_walks_all_terms_in_order() {
    let b = Builder::new();
    let (tim, tip, tmd) = b.build(
        IndexOptions::DocsAndFreqs,
        &[("a", 1, 1), ("ab", 2, 3), ("b", 1, 1)],
    );
    let fis = FieldInfos {
        fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
    };
    let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
    let field = fields.field("text").unwrap();
    let mut it = field.iter();
    assert_eq!(it.current(), None);
    assert_eq!(
        it.next(),
        Some((
            b"a".as_slice(),
            TermStats {
                doc_freq: 1,
                total_term_freq: 1
            }
        ))
    );
    assert_eq!(
        it.current(),
        Some((
            b"a".as_slice(),
            TermStats {
                doc_freq: 1,
                total_term_freq: 1
            }
        ))
    );
    assert_eq!(
        it.next(),
        Some((
            b"ab".as_slice(),
            TermStats {
                doc_freq: 2,
                total_term_freq: 3
            }
        ))
    );
    assert_eq!(
        it.next(),
        Some((
            b"b".as_slice(),
            TermStats {
                doc_freq: 1,
                total_term_freq: 1
            }
        ))
    );
    assert_eq!(it.next(), None);
    // Idempotent past the end.
    assert_eq!(it.next(), None);
    assert_eq!(it.current(), None);
}

#[test]
fn terms_enum_seek_ceil_found_notfound_end_and_continues() {
    let b = Builder::new();
    let (tim, tip, tmd) = b.build(
        IndexOptions::DocsAndFreqs,
        &[("a", 1, 1), ("ab", 2, 3), ("b", 1, 1)],
    );
    let fis = FieldInfos {
        fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
    };
    let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
    let field = fields.field("text").unwrap();

    // Exact match.
    let mut it = field.iter();
    assert_eq!(it.seek_ceil(b"ab"), SeekStatus::Found);
    assert_eq!(
        it.current(),
        Some((
            b"ab".as_slice(),
            TermStats {
                doc_freq: 2,
                total_term_freq: 3
            }
        ))
    );
    // next() after seekCeil continues past the found term.
    assert_eq!(
        it.next(),
        Some((
            b"b".as_slice(),
            TermStats {
                doc_freq: 1,
                total_term_freq: 1
            }
        ))
    );

    // Ceiling match: falls strictly between "a" and "ab".
    let mut it = field.iter();
    assert_eq!(it.seek_ceil(b"aa"), SeekStatus::NotFound);
    assert_eq!(
        it.current(),
        Some((
            b"ab".as_slice(),
            TermStats {
                doc_freq: 2,
                total_term_freq: 3
            }
        ))
    );

    // Before the first term: ceiling is the first term.
    let mut it = field.iter();
    assert_eq!(it.seek_ceil(b""), SeekStatus::NotFound);
    assert_eq!(
        it.current(),
        Some((
            b"a".as_slice(),
            TermStats {
                doc_freq: 1,
                total_term_freq: 1
            }
        ))
    );

    // After the last term: no ceiling exists.
    let mut it = field.iter();
    assert_eq!(it.seek_ceil(b"z"), SeekStatus::End);
    assert_eq!(it.current(), None);
    assert_eq!(it.next(), None);
}

/// `TermsEnum`'s past-the-end cursor states. A real writer never emits a
/// zero-term field (`open()` itself rejects `numTerms <= 0`), so the
/// smallest dictionary that can stand in for one is a single-term field
/// walked past its only term: `next()` must stay idempotent at the end
/// (Java's `eof`), `current()` must report nothing, and a `seek_ceil`
/// past the last term must report `End` and leave the cursor unpositioned.
#[test]
fn terms_enum_past_the_end_is_idempotent() {
    let b = Builder::new();
    let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("only", 1, 1)]);
    let fis = FieldInfos {
        fields: vec![field_info(0, "text", IndexOptions::Docs)],
    };
    let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 1).unwrap();
    let field = fields.field("text").unwrap();

    let mut it = field.iter();
    assert!(it.next().is_some());
    assert_eq!(it.next(), None);
    assert_eq!(it.next(), None);
    assert_eq!(it.current(), None);

    let mut it = field.iter();
    assert_eq!(it.seek_ceil(b"zzz"), SeekStatus::End);
    assert_eq!(it.current(), None);
    assert_eq!(it.next(), None);

    // A seek *after* end-of-terms must still answer for its target, not
    // fall out of `next()`'s `eof` guard: `SegmentTermsEnum::reset` has to
    // clear `eof`, and only a walk-then-seek order can prove it does.
    let mut it = field.iter();
    while it.next().is_some() {}
    assert_eq!(it.seek_ceil(b"only"), SeekStatus::Found);
    assert_eq!(it.current().unwrap().0, b"only");
    assert_eq!(it.seek_ceil(b"a"), SeekStatus::NotFound);
    assert_eq!(it.current().unwrap().0, b"only");
    assert_eq!(it.seek_ceil(b"zzz"), SeekStatus::End);

    // A field with no `.tim`/`.tip`/`.tmd` at all has no fields to
    // iterate in the first place.
    assert!(BlockTreeFields::empty().field("text").is_none());
    assert_eq!(BlockTreeFields::empty().iter_fields().count(), 0);
}

#[test]
fn terms_enum_single_term_field() {
    let b = Builder::new();
    let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("only", 1, 1)]);
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::Docs)],
    };
    let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
    let field = fields.field("f").unwrap();
    let mut it = field.iter();
    assert_eq!(
        it.next(),
        Some((
            b"only".as_slice(),
            TermStats {
                doc_freq: 1,
                total_term_freq: 1
            }
        ))
    );
    assert_eq!(it.next(), None);

    let mut it2 = field.iter();
    assert_eq!(it2.seek_ceil(b"only"), SeekStatus::Found);
    assert_eq!(it2.next(), None);
}

#[test]
fn docs_only_index_options_omits_total_term_freq_field() {
    // IndexOptions::Docs never writes a distinct sumTotalTermFreq, and
    // per-term stats never write the extra totalTermFreq vlong either.
    let b = Builder::new();
    let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("x", 3, 3), ("y", 1, 1)]);
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::Docs)],
    };
    let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
    let field = fields.field("f").unwrap();
    assert_eq!(field.sum_total_term_freq, field.sum_doc_freq);
    assert_eq!(
        field.seek_exact(b"x"),
        Some(TermStats {
            doc_freq: 3,
            total_term_freq: 3
        })
    );
}

/// `CodecUtil.retrieveChecksum(IndexInput, long expectedLength)` rejects
/// "file too long" as well as "truncated file": the recorded length must
/// equal the file's own. A `.tmd` claiming a shorter `.tip`/`.tim` than the
/// one it describes would otherwise be accepted and the footer read from
/// the wrong offset.
#[test]
fn recorded_tip_or_tim_length_shorter_than_the_file_is_rejected() {
    let b = Builder::new();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::Docs)],
    };
    let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("only", 1, 1)]);
    // Sanity: unmodified, it opens.
    assert!(open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).is_ok());

    for which in ["tip", "tim"] {
        let mut tim = tim.clone();
        let mut tip = tip.clone();
        // One trailing byte past the recorded length -- Lucene's "file
        // too long" case.
        if which == "tip" {
            tip.push(0);
        } else {
            tim.push(0);
        }
        let err = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap_err();
        assert!(
            matches!(err, Error::Store(lucene_store::Error::Corrupted(_))),
            "{which}: {err:?}"
        );
    }
}

#[test]
fn invalid_num_fields_rejected() {
    let mut tmd = Vec::new();
    let id = [1u8; ID_LENGTH];
    codec_util::write_index_header(&mut tmd, TERMS_META_CODEC_NAME, VERSION_CURRENT, &id, "");
    codec_util::write_index_header(&mut tmd, POSTINGS_TERMS_CODEC, VERSION_CURRENT, &id, "");
    tmd.write_vint(POSTINGS_BLOCK_SIZE);
    tmd.write_vint(-1); // invalid numFields
    codec_util::write_footer(&mut tmd);

    let mut tim = Vec::new();
    codec_util::write_index_header(&mut tim, TERMS_CODEC_NAME, VERSION_CURRENT, &id, "");
    codec_util::write_footer(&mut tim);
    let mut tip = Vec::new();
    codec_util::write_index_header(&mut tip, TERMS_INDEX_CODEC_NAME, VERSION_CURRENT, &id, "");
    codec_util::write_footer(&mut tip);

    let fis = FieldInfos { fields: vec![] };
    let err = open(&tim, &tip, &tmd, &fis, &id, "", 5).unwrap_err();
    assert!(matches!(err, Error::InvalidNumFields(-1)));
}

#[test]
fn unexpected_postings_block_size_rejected() {
    let mut tmd = Vec::new();
    let id = [1u8; ID_LENGTH];
    codec_util::write_index_header(&mut tmd, TERMS_META_CODEC_NAME, VERSION_CURRENT, &id, "");
    codec_util::write_index_header(&mut tmd, POSTINGS_TERMS_CODEC, VERSION_CURRENT, &id, "");
    tmd.write_vint(128); // wrong block size
    codec_util::write_footer(&mut tmd);

    let mut tim = Vec::new();
    codec_util::write_index_header(&mut tim, TERMS_CODEC_NAME, VERSION_CURRENT, &id, "");
    codec_util::write_footer(&mut tim);
    let mut tip = Vec::new();
    codec_util::write_index_header(&mut tip, TERMS_INDEX_CODEC_NAME, VERSION_CURRENT, &id, "");
    codec_util::write_footer(&mut tip);

    let fis = FieldInfos { fields: vec![] };
    let err = open(&tim, &tip, &tmd, &fis, &id, "", 5).unwrap_err();
    assert!(matches!(err, Error::UnexpectedBlockSize { found: 128 }));
}

#[test]
fn multi_children_node_with_invalid_strategy_code_rejected() {
    // childSaveStrategy code 3 doesn't exist (only 0/1/2 are defined) ->
    // a structural Corrupted error, not silently misdecoded.
    let mut slice = vec![0u8; 24];
    let term: u32 = SIGN_MULTI_CHILDREN | (3 << 9); // invalid strategy code
    slice[0..3].copy_from_slice(&term.to_le_bytes()[0..3]);
    let err = load_node(&slice, 0)
        .and_then(|node| lookup_child(&slice, &node, 1))
        .unwrap_err();
    assert!(matches!(err, Error::Store(_)));
}

#[test]
fn single_child_trie_node_with_output_and_floor_round_trips() {
    // Build a leaf child at fp=0, then a SIGN_SINGLE_CHILD_WITH_OUTPUT
    // parent at fp=16 with its own output+floor data, pointing at the
    // child via a 1-byte delta -- exercises loadSingleChildNode's
    // "has output" branch end to end (TrieReader.loadSingleChildNode).
    let mut slice = vec![0u8; 40];
    // Child: SIGN_NO_CHILDREN, 1-byte output fp = 42, hasTerms.
    slice[0] = LEAF_NODE_HAS_TERMS as u8;
    slice[1] = 42;

    let parent_fp = 16usize;
    let child_delta_fp: u8 = parent_fp as u8; // 16
    let label: u8 = b'x';
    // encodeFP: (floor?1:0) | (hasTerms?2:0) | (fp << 2); output fp = 20.
    let encoded_fp: u64 = NON_LEAF_NODE_HAS_FLOOR | NON_LEAF_NODE_HAS_TERMS | (20 << 2);
    assert!(encoded_fp <= 0xFF, "fits in 1 byte for this test");

    // childDeltaFpBytesMinus1 = 0 (1 byte), encodedOutputFpBytesMinus1 = 0 (1 byte)
    let term: u32 = SIGN_SINGLE_CHILD_WITH_OUTPUT;
    slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());
    slice[parent_fp + 1] = label;
    slice[parent_fp + 2] = child_delta_fp;
    slice[parent_fp + 3] = encoded_fp as u8;
    // Floor data right after the 1-byte encoded output fp: one follow
    // block, floorLeadByte='y', code = (5 << 1) | 1 (hasTerms).
    let floor_fp = parent_fp + 4;
    slice[floor_fp] = 1; // numFollowFloorBlocks vint
    slice[floor_fp + 1] = b'y';
    slice[floor_fp + 2] = (5 << 1) | 1; // code vlong

    let node = load_node(&slice, parent_fp).unwrap();
    assert_eq!(node.sign, SIGN_SINGLE_CHILD_WITH_OUTPUT);
    assert_eq!(node.output_fp, Some(20));
    assert!(node.has_terms);
    assert_eq!(node.floor_data_fp, Some(floor_fp));
    assert_eq!(node.min_children_label, label);

    let blocks = expand_floor(&slice, 20, true, node.floor_data_fp).unwrap();
    assert_eq!(blocks, vec![(20, true), (25, true)]);

    let child_fp = parent_fp - node.child_delta_fp as usize;
    assert_eq!(child_fp, 0);
    let child = load_node(&slice, child_fp).unwrap();
    assert_eq!(child.output_fp, Some(42));
    assert!(child.has_terms);
}

#[test]
fn single_child_without_output_has_no_own_block() {
    let mut slice = vec![0u8; 24];
    let parent_fp = 8usize;
    // childDeltaFpBytesMinus1 = 0 (1 byte)
    let term: u32 = SIGN_SINGLE_CHILD_WITHOUT_OUTPUT;
    slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());
    slice[parent_fp + 1] = b'q';
    slice[parent_fp + 2] = 8; // child delta fp -> child at fp 0

    let node = load_node(&slice, parent_fp).unwrap();
    assert_eq!(node.output_fp, None);
    assert!(!node.has_terms);
    assert_eq!(node.min_children_label, b'q');
}

/// Builds a `SIGN_MULTI_CHILDREN` node (no output of its own) with two
/// children under the given `strategy`, and asserts `multi_children_fps`
/// recovers exactly the two child fps regardless of which
/// `ChildSaveStrategy` encoded them -- `TrieReader.lookupChild`'s three
/// strategies (`BITS`/`ARRAY`/`REVERSE_ARRAY`), generalized to "list all"
/// (see [`multi_children_labels_and_fps`]'s doc comment for why).
fn build_and_check_multi_children(strategy: u32, strategy_bytes_region: &[u8]) {
    let mut slice = vec![0u8; 32];
    // Child A: leaf, output fp = 10, hasTerms, at fp 0.
    slice[0] = LEAF_NODE_HAS_TERMS as u8;
    slice[1] = 10;
    // Child B: leaf, output fp = 20, hasTerms, at fp 2.
    slice[2] = LEAF_NODE_HAS_TERMS as u8;
    slice[3] = 20;

    let parent_fp = 8usize;
    let min_label = b'a';
    let strategy_bytes = strategy_bytes_region.len();
    // childrenDeltaFpBytesMinus1 = 0 (1 byte), no output
    let term: u32 = SIGN_MULTI_CHILDREN
        | (strategy << 9)
        | (((strategy_bytes - 1) as u32) << 11)
        | ((min_label as u32) << 16);
    slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());

    let strategy_fp = parent_fp + 3;
    slice[strategy_fp..strategy_fp + strategy_bytes].copy_from_slice(strategy_bytes_region);
    let fps_fp = strategy_fp + strategy_bytes;
    slice[fps_fp] = parent_fp as u8; // delta to child A
    slice[fps_fp + 1] = (parent_fp - 2) as u8; // delta to child B

    let node = load_node(&slice, parent_fp).unwrap();
    assert_eq!(node.output_fp, None);
    assert_eq!(node.child_save_strategy, strategy);
    assert_eq!(node.strategy_bytes, strategy_bytes);

    let mut child_fps: Vec<usize> = trie_children(&slice, &node)
        .unwrap()
        .into_iter()
        .map(|(_, child)| child.fp)
        .collect();
    child_fps.sort_unstable();
    assert_eq!(child_fps, vec![0, 2]);

    let mut collected = Vec::new();
    let mut prefix = Vec::new();
    collect_leaf_blocks(&slice, &node, 0, &mut prefix, &mut collected).unwrap();
    let mut fps: Vec<u64> = collected.iter().map(|(fp, _)| *fp).collect();
    fps.sort_unstable();
    assert_eq!(fps, vec![10, 20]);
}

#[test]
fn multi_children_array_strategy() {
    // ARRAY: labels[1..] stored explicitly ('b' = 0x62), minLabel='a'
    // implicit.
    build_and_check_multi_children(CHILD_STRATEGY_ARRAY, b"b");
}

#[test]
fn multi_children_bits_strategy() {
    // BITS: byteDistance = 'b'-'a'+1 = 2 -> 1 byte; bit0 (label 'a')
    // and bit1 (label 'b') both set -> 0b011 = 3.
    build_and_check_multi_children(CHILD_STRATEGY_BITS, &[0b011]);
}

#[test]
fn multi_children_reverse_array_strategy() {
    // REVERSE_ARRAY: byte0 = maxLabel ('b'), no missing labels between
    // 'a' and 'b' (they're consecutive) -> exactly 1 byte.
    build_and_check_multi_children(CHILD_STRATEGY_REVERSE_ARRAY, b"b");
}

#[test]
fn multi_children_reverse_array_strategy_with_gap() {
    // Labels 'a' and 'd' (a gap of 'b','c' in between): byteDistance=4,
    // labelCnt=2 -> strategyBytes = 4-2+1 = 3: [maxLabel='d', 'b', 'c'].
    let mut slice = vec![0u8; 32];
    slice[0] = LEAF_NODE_HAS_TERMS as u8;
    slice[1] = 10;
    slice[2] = LEAF_NODE_HAS_TERMS as u8;
    slice[3] = 20;

    let parent_fp = 8usize;
    let min_label = b'a';
    let strategy_bytes_region = *b"dbc";
    let strategy_bytes = strategy_bytes_region.len();
    let term: u32 = SIGN_MULTI_CHILDREN
        | (CHILD_STRATEGY_REVERSE_ARRAY << 9)
        | (((strategy_bytes - 1) as u32) << 11)
        | ((min_label as u32) << 16);
    slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());
    let strategy_fp = parent_fp + 3;
    slice[strategy_fp..strategy_fp + strategy_bytes].copy_from_slice(&strategy_bytes_region);
    let fps_fp = strategy_fp + strategy_bytes;
    slice[fps_fp] = parent_fp as u8; // delta to child A (fp 0)
    slice[fps_fp + 1] = (parent_fp - 2) as u8; // delta to child B (fp 2)

    let node = load_node(&slice, parent_fp).unwrap();
    let mut child_fps: Vec<usize> = trie_children(&slice, &node)
        .unwrap()
        .into_iter()
        .map(|(_, child)| child.fp)
        .collect();
    child_fps.sort_unstable();
    assert_eq!(child_fps, vec![0, 2]);
    // The two labels the encoding lists as *absent* between 'a' and 'd'
    // must miss, not resolve to a neighbour.
    assert!(lookup_child(&slice, &node, b'b').unwrap().is_none());
    assert!(lookup_child(&slice, &node, b'c').unwrap().is_none());
}

/// `REVERSE_ARRAY` with *two* gaps, so `lookup`'s bisection over the
/// absent-label list takes its `midLabel > target` branch and returns
/// through the `target - minLabel - low` arithmetic rather than hitting an
/// exact miss. Labels 'a', 'c', 'e'; absent 'b', 'd'; strategy region
/// `[maxLabel='e', 'b', 'd']` (`needBytes` = 5 - 3 + 1 = 3).
#[test]
fn multi_children_reverse_array_strategy_with_two_gaps() {
    let mut slice = vec![0u8; 40];
    // Three leaf children at fp 0, 2, 4 with distinct output fps.
    for (i, out) in [(0usize, 10u8), (2, 20), (4, 30)] {
        slice[i] = LEAF_NODE_HAS_TERMS as u8;
        slice[i + 1] = out;
    }

    let parent_fp = 12usize;
    let min_label = b'a';
    let strategy_bytes_region = *b"ebd";
    let strategy_bytes = strategy_bytes_region.len();
    let term: u32 = SIGN_MULTI_CHILDREN
        | (CHILD_STRATEGY_REVERSE_ARRAY << 9)
        | (((strategy_bytes - 1) as u32) << 11)
        | ((min_label as u32) << 16);
    slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());
    let strategy_fp = parent_fp + 3;
    slice[strategy_fp..strategy_fp + strategy_bytes].copy_from_slice(&strategy_bytes_region);
    let fps_fp = strategy_fp + strategy_bytes;
    slice[fps_fp] = parent_fp as u8; // 'a' -> fp 0
    slice[fps_fp + 1] = (parent_fp - 2) as u8; // 'c' -> fp 2
    slice[fps_fp + 2] = (parent_fp - 4) as u8; // 'e' -> fp 4

    let node = load_node(&slice, parent_fp).unwrap();
    assert_eq!(node.child_save_strategy, CHILD_STRATEGY_REVERSE_ARRAY);
    for (label, fp, output) in [(b'a', 0usize, 10u64), (b'c', 2, 20), (b'e', 4, 30)] {
        let child = lookup_child(&slice, &node, label)
            .unwrap()
            .unwrap_or_else(|| panic!("label {} should resolve", label as char));
        assert_eq!(child.fp, fp, "label {}", label as char);
        assert_eq!(child.output_fp, Some(output), "label {}", label as char);
    }
    for label in *b"`bdf" {
        assert!(
            lookup_child(&slice, &node, label).unwrap().is_none(),
            "label {} should miss",
            label as char
        );
    }
}

#[test]
fn collect_leaf_blocks_skips_output_with_no_terms() {
    // A node whose own output has hasTerms=false (a pointer-only block,
    // e.g. a coarser prefix the writer recursed past rather than
    // floor-split) contributes no block of its own -- it's skipped, not
    // an error, since any real terms under that prefix are reachable
    // through this node's own children instead (see
    // `collect_leaf_blocks`'s doc comment). Here the node has
    // `SIGN_NO_CHILDREN`, so skipping it means zero blocks collected.
    let mut slice = vec![0u8; 16];
    slice[0] = 0; // sign=0, fpBytesMinus1=0, no LEAF_NODE_HAS_TERMS bit
    slice[1] = 5;
    let node = load_node(&slice, 0).unwrap();
    let mut out = Vec::new();
    let mut prefix = Vec::new();
    collect_leaf_blocks(&slice, &node, 0, &mut prefix, &mut out).unwrap();
    assert!(out.is_empty());
}

#[test]
fn expand_floor_rejects_negative_num_follow() {
    let mut buf = Vec::new();
    buf.write_vint(-1);
    let err = expand_floor(&buf, 0, true, Some(0)).unwrap_err();
    assert!(matches!(err, Error::Store(_)));
}

#[test]
fn expand_floor_no_floor_data_returns_just_the_base_block() {
    let blocks = expand_floor(&[], 7, true, None).unwrap();
    assert_eq!(blocks, vec![(7, true)]);
}

/// End-to-end: a field whose terms span two floor sub-blocks under one
/// leaf trie node (`LEAF_NODE_HAS_FLOOR`), with a hand-built
/// `.tim`/`.tip`/`.tmd` triple no real (small) fixture reaches.
///
/// Exercises the two things floor blocks demand of the lazy reader:
/// `scanToFloorFrame` must pick the *one* sub-block a target's lead byte
/// falls in (so a term in the second block is not looked for in the
/// first), and `next()` must chain from one floor block to the next off
/// the `isLastInFloor` bit rather than stopping at the first block's end.
#[test]
fn open_floor_field_walks_both_blocks() {
    let id = [9u8; ID_LENGTH];
    let suffix = String::new();

    fn write_leaf_block(
        tim: &mut Vec<u8>,
        terms: &[(&str, u32, u64)],
        is_last_in_floor: bool,
    ) -> usize {
        let block_fp = tim.len();
        let ent_count = terms.len() as u32;
        tim.write_vint(((ent_count << 1) | u32::from(is_last_in_floor)) as i32);

        let mut suffix_bytes = Vec::new();
        let mut suffix_lengths = Vec::new();
        let mut stats = Vec::new();
        for (term, doc_freq, total_term_freq) in terms {
            suffix_bytes.extend_from_slice(term.as_bytes());
            suffix_lengths.write_vint(term.len() as i32);
            stats.write_vint((*doc_freq as i32) << 1);
            stats.write_vlong((*total_term_freq as i64) - (*doc_freq as i64));
        }
        let code_l = ((suffix_bytes.len() as u64) << 3) | 0x04;
        tim.write_vlong(code_l as i64);
        tim.write_bytes(&suffix_bytes);
        tim.write_vint((suffix_lengths.len() as i32) << 1);
        tim.write_bytes(&suffix_lengths);
        tim.write_vint(stats.len() as i32);
        tim.write_bytes(&stats);

        let mut meta = Vec::new();
        for (_, doc_freq, _) in terms {
            meta.write_vlong(10 << 1);
            if *doc_freq == 1 {
                meta.write_vint(0);
            }
        }
        tim.write_vint(meta.len() as i32);
        tim.write_bytes(&meta);
        block_fp
    }

    let mut tim = Vec::new();
    codec_util::write_index_header(&mut tim, TERMS_CODEC_NAME, VERSION_CURRENT, &id, &suffix);
    // Terms are sorted within each block, and split across the floor
    // boundary at 'm' exactly as `Lucene103BlockTreeTermsWriter` would.
    let block0_fp = write_leaf_block(&mut tim, &[("a", 1, 1), ("b", 1, 1)], false);
    let block1_fp = write_leaf_block(&mut tim, &[("m", 1, 1), ("z", 2, 5)], true);
    codec_util::write_footer(&mut tim);

    let mut tip = Vec::new();
    codec_util::write_index_header(
        &mut tip,
        TERMS_INDEX_CODEC_NAME,
        VERSION_CURRENT,
        &id,
        &suffix,
    );
    let index_start = tip.len();
    // Leaf root, floor: header | outputFp(block0_fp, 8 bytes to keep it
    // simple) | floor data.
    let header = LEAF_NODE_HAS_TERMS as u8 | LEAF_NODE_HAS_FLOOR as u8 | (7 << 2);
    tip.push(header);
    tip.extend_from_slice(&(block0_fp as u64).to_le_bytes());
    tip.write_vint(1); // numFollowFloorBlocks
    tip.write_byte(b'm'); // floorLeadByte for block1
    tip.write_vlong((((block1_fp - block0_fp) as i64) << 1) | 1); // code, hasTerms
    tip.extend_from_slice(&0u64.to_le_bytes()); // over-read pad
    let index_end = tip.len();
    codec_util::write_footer(&mut tip);

    let mut tmd = Vec::new();
    codec_util::write_index_header(
        &mut tmd,
        TERMS_META_CODEC_NAME,
        VERSION_CURRENT,
        &id,
        &suffix,
    );
    codec_util::write_index_header(
        &mut tmd,
        POSTINGS_TERMS_CODEC,
        VERSION_CURRENT,
        &id,
        &suffix,
    );
    tmd.write_vint(POSTINGS_BLOCK_SIZE);
    tmd.write_vint(1); // numFields
    tmd.write_vint(0); // field number
    tmd.write_vlong(4); // numTerms
    tmd.write_vlong(8); // sumTotalTermFreq = 1+1+1+5
    tmd.write_vlong(5); // sumDocFreq = 1+1+1+2
    tmd.write_vint(1); // docCount
    tmd.write_vint(1);
    tmd.write_bytes(b"a");
    tmd.write_vint(1);
    tmd.write_bytes(b"z");
    tmd.write_vlong(index_start as i64);
    tmd.write_vlong(0); // root fp within index slice
    tmd.write_vlong(index_end as i64);
    tmd.write_i64(tip.len() as i64); // indexLength: the whole file, footer included
    tmd.write_i64(tim.len() as i64);
    codec_util::write_footer(&mut tmd);

    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
    };
    let fields = open(&tim, &tip, &tmd, &fis, &id, &suffix, 5).unwrap();
    let field = fields.field("f").unwrap();
    assert_eq!(field.num_terms, 4);

    // Every term is reachable, whichever floor sub-block holds it.
    for (term, expected) in [("a", (1, 1)), ("b", (1, 1)), ("m", (1, 1)), ("z", (2, 5))] {
        let stats = field.seek_exact(term.as_bytes()).unwrap();
        assert_eq!(stats.doc_freq, expected.0, "term={term}");
        assert_eq!(stats.total_term_freq, expected.1, "term={term}");
    }
    // A miss whose lead byte falls in the *second* floor block, and one
    // that falls in the first: both must scan only their own block.
    assert!(field.seek_exact(b"missing").is_none());
    assert!(field.seek_exact(b"c").is_none());

    // `next()` must chain across the floor boundary.
    let mut it = field.iter();
    let mut walked = Vec::new();
    while let Some((term, stats)) = it.next() {
        walked.push((String::from_utf8(term.to_vec()).unwrap(), stats.doc_freq));
    }
    assert_eq!(
        walked,
        vec![
            ("a".to_string(), 1),
            ("b".to_string(), 1),
            ("m".to_string(), 1),
            ("z".to_string(), 2),
        ]
    );

    // And so must `seek_ceil` into the second block from a target in the
    // first block's range.
    let mut it = field.iter();
    assert_eq!(it.seek_ceil(b"c"), SeekStatus::NotFound);
    assert_eq!(it.current().unwrap().0, b"m");
}

#[test]
fn empty_terms_block_rejected() {
    let id = [2u8; ID_LENGTH];
    let mut tim = Vec::new();
    codec_util::write_index_header(&mut tim, TERMS_CODEC_NAME, VERSION_CURRENT, &id, "");
    let block_fp = tim.len();
    tim.write_vint(1); // entCount=0, isLastInFloor=true -> code = 0<<1|1 = 1
    codec_util::write_footer(&mut tim);

    let mut tip = Vec::new();
    codec_util::write_index_header(&mut tip, TERMS_INDEX_CODEC_NAME, VERSION_CURRENT, &id, "");
    let index_start = tip.len();
    let header = LEAF_NODE_HAS_TERMS as u8; // SIGN_NO_CHILDREN, 1-byte fp
    tip.push(header);
    tip.extend_from_slice(&(block_fp as u64).to_le_bytes());
    tip.extend_from_slice(&0u64.to_le_bytes());
    let index_end = tip.len();
    codec_util::write_footer(&mut tip);

    let mut tmd = Vec::new();
    codec_util::write_index_header(&mut tmd, TERMS_META_CODEC_NAME, VERSION_CURRENT, &id, "");
    codec_util::write_index_header(&mut tmd, POSTINGS_TERMS_CODEC, VERSION_CURRENT, &id, "");
    tmd.write_vint(POSTINGS_BLOCK_SIZE);
    tmd.write_vint(1);
    tmd.write_vint(0);
    tmd.write_vlong(1); // numTerms must be >0 to pass that check; block itself will be empty
    tmd.write_vlong(0); // sumDocFreq (Docs aliasing)
    tmd.write_vint(0); // docCount
    tmd.write_vint(0);
    tmd.write_bytes(&[]);
    tmd.write_vint(0);
    tmd.write_bytes(&[]);
    tmd.write_vlong(index_start as i64);
    tmd.write_vlong(0);
    tmd.write_vlong(index_end as i64);
    tmd.write_i64(tip.len() as i64); // indexLength: the whole file, footer included
    tmd.write_i64(tim.len() as i64);
    codec_util::write_footer(&mut tmd);

    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::Docs)],
    };
    // `open` reads no `.tim` block (see the module doc), so a
    // block this corrupt is rejected where real Lucene rejects it too:
    // from inside the lookup that loads it.
    let fields = open(&tim, &tip, &tmd, &fis, &id, "", 5).unwrap();
    let field = fields.field("f").unwrap();
    let err = field.try_seek_exact(b"").unwrap_err();
    assert!(matches!(err, Error::Store(_)));
    // The infallible spelling degrades it to "no such term".
    assert!(field.seek_exact(b"").is_none());
}

/// A corrupt block is now discovered where real Lucene discovers it --
/// inside the lookup that loads it, not at `open` -- and every lookup's
/// `Result`-returning form surfaces it. The infallible spellings degrade
/// to "no such term"/end-of-terms instead (see the module doc).
#[test]
fn a_corrupt_block_errors_from_the_lookup_not_from_open() {
    let b = Builder::new();
    let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("alpha", 1, 1), ("beta", 1, 1)]);
    let fis = FieldInfos {
        fields: vec![field_info(0, "text", IndexOptions::Docs)],
    };

    // Find the block's suffix-lengths region and inflate the first
    // entry's length past the end of the suffix bytes, which is what
    // `scanToTermLeaf`'s cursor has to reject.
    let header = codec_util::index_header_length(TERMS_CODEC_NAME, &b.suffix);
    let mut r = SliceInput::new(&tim);
    r.seek(header).unwrap();
    let _code = r.read_vint().unwrap();
    let code_l = r.read_vlong().unwrap() as u64;
    let num_suffix_bytes = (code_l >> 3) as usize;
    r.seek(r.position() + num_suffix_bytes).unwrap();
    let _num_suffix_length_bytes = r.read_vint().unwrap();
    let first_length_at = r.position();
    assert_eq!(tim[first_length_at], b"alpha".len() as u8);

    let mut corrupt = tim.clone();
    corrupt[first_length_at] = 100;
    let fields = open(&corrupt, &tip, &tmd, &fis, &b.id, &b.suffix, 5)
        .expect("open reads no .tim block, so it cannot see this");
    let field = fields.field("text").unwrap();

    let err = field.try_seek_exact(b"alpha").unwrap_err();
    assert!(matches!(err, Error::Store(_)), "{err:?}");
    assert!(field.seek_exact(b"alpha").is_none());

    let mut it = field.iter();
    assert!(it.try_next().is_err());
    let mut it = field.iter();
    assert_eq!(it.next(), None);
    let mut it = field.iter();
    assert!(it.try_seek_ceil(b"alpha").is_err());
    let mut it = field.iter();
    assert_eq!(it.seek_ceil(b"alpha"), SeekStatus::End);

    // The intersect iterators run over the same enum and now surface the
    // same error rather than ending quietly: a truncated term expansion
    // is a wrong hit set, not a smaller one.
    let pattern = WildcardPattern::new(b"a*");
    let items: Vec<_> = field.intersect(&pattern).collect();
    assert_eq!(items.len(), 1, "one error, then the walk is over");
    assert!(matches!(items[0], Err(Error::Store(_))), "{:?}", items[0]);
}

/// `SegmentTermsEnum.seekExact`'s "index exhausted and this frame has no
/// terms" fast path: the trie routed us to a node whose own `.tim` block
/// holds nothing but sub-block pointers, so the target cannot exist and
/// no block is loaded at all.
///
/// Uses `blocktree_child_strategies_index`, whose "arraystrat" field has
/// exactly that shape at the root (see
/// `open_field_with_no_terms_in_root_block_still_finds_every_term`), and
/// picks a lead byte inside the field's min/max range that is *not* one
/// of the root's five child labels, so the very first `lookupChild`
/// misses while the current frame is the no-terms root.
#[test]
fn seek_exact_stops_at_a_trie_node_with_no_terms() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_child_strategies_index/"
    );
    let text = std::fs::read_to_string(format!("{dir}manifest.properties"))
        .expect("run fixtures generator first (GenBlockTreeChildStrategies)");
    let kv: Vec<(String, String)> = text
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let get = |k: &str| {
        kv.iter()
            .find(|(kk, _)| kk == k)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("manifest key {k} missing"))
    };
    let mut id = [0u8; ID_LENGTH];
    let hex = get("id_hex");
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    let suffix = get("segment_suffix").to_string();
    let max_doc: i32 = get("max_doc").parse().unwrap();
    let read = |n: &str| std::fs::read(format!("{dir}{n}.raw")).unwrap();
    let field_infos = crate::field_infos::parse(&read(get("fnm_file_name")), &id, "").unwrap();
    let fields = open(
        &read(get("tim_file_name")),
        &read(get("tip_file_name")),
        &read(get("tmd_file_name")),
        &field_infos,
        &id,
        &suffix,
        max_doc,
    )
    .unwrap();
    let field = fields.field("arraystrat").unwrap();

    // Every lead byte strictly between the field's min and max term that
    // has no root child: `seek_exact` must miss without loading a block.
    let lo = field.min_term[0];
    let hi = field.max_term[0];
    let mut checked = 0;
    for lead in (lo + 1)..hi {
        let target = [lead, b'x'];
        if field.seek_exact(&target).is_some() {
            continue;
        }
        assert_eq!(field.try_seek_exact(&target).unwrap(), None);
        checked += 1;
    }
    assert!(
        checked > 0,
        "expected at least one absent lead byte inside [{lo}, {hi}]"
    );

    // Over this many-block field, the prefetch hint names exactly the
    // block a full seek loads, for every term.
    let tim: &[u8] = field.tim.as_ref().as_ref();
    let mut terms = Vec::new();
    let mut it = field.iter();
    while let Some((term, _)) = it.next() {
        terms.push(term.to_vec());
    }
    assert!(terms.len() > 100);
    for term in &terms {
        let loaded = field
            .with_scratch(|ste| {
                assert!(ste.seek_exact(term)?);
                Ok(ste.cur_ref().fp)
            })
            .unwrap();
        let hint = field.term_block(term).unwrap().expect("a block");
        assert_eq!(hint.as_ptr(), tim[loaded..].as_ptr(), "{term:?}");
    }
}

/// The pooled lookup scratch must never make concurrent searchers wait on
/// each other: `BlockTreeFields` is shared by every search thread through
/// one `Arc`, so a blocking lock here would serialize the whole term
/// dictionary. A thread that loses the race runs on its own state and
/// must get identical answers.
#[test]
fn concurrent_lookups_do_not_serialize_and_agree() {
    let b = Builder::new();
    let owned: Vec<String> = (0..200).map(|i| format!("term{i:04}")).collect();
    let mut sorted: Vec<&str> = owned.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    let entries: Vec<(&str, u32, u64)> = sorted
        .iter()
        .enumerate()
        .map(|(i, t)| (*t, i as u32 + 1, i as u64 + 1))
        .collect();
    let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, &entries);
    let fis = FieldInfos {
        fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
    };
    let fields = std::sync::Arc::new(open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 500).unwrap());

    let mut handles = Vec::new();
    for _ in 0..8 {
        let fields = std::sync::Arc::clone(&fields);
        let expected: Vec<(String, u32)> = entries
            .iter()
            .map(|(t, df, _)| ((*t).to_string(), *df))
            .collect();
        handles.push(std::thread::spawn(move || {
            let field = fields.field("text").unwrap();
            for _ in 0..20 {
                for (term, df) in &expected {
                    let stats = field
                        .try_seek_exact(term.as_bytes())
                        .expect("intact bytes")
                        .unwrap_or_else(|| panic!("term {term} not found"));
                    assert_eq!(stats.doc_freq, *df as i32);
                }
                assert!(field.seek_exact(b"nope").is_none());
            }
        }));
    }
    for h in handles {
        h.join().expect("no thread panicked");
    }
}

/// A cloned `FieldTerms` shares the segment's `.tim`/`.tip` buffers and
/// gets its own lookup scratch, so both copies keep working
/// independently -- `BlockTreeFields` is `Clone` and callers hold it
/// behind an `Arc`, so this is a real path.
#[test]
fn cloning_a_field_keeps_both_copies_usable() {
    let b = Builder::new();
    let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("alpha", 3, 3), ("beta", 4, 4)]);
    let fis = FieldInfos {
        fields: vec![field_info(0, "text", IndexOptions::Docs)],
    };
    let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
    let copy = fields.clone();
    for f in [&fields, &copy] {
        let field = f.field("text").unwrap();
        assert_eq!(field.seek_exact(b"alpha").unwrap().doc_freq, 3);
        assert_eq!(field.seek_exact(b"beta").unwrap().doc_freq, 4);
        assert!(field.seek_exact(b"gamma").is_none());
    }
    // The current format's trie is read in place, never deferred.
    let debug = format!("{:?}", fields.field("text").unwrap());
    assert!(debug.contains("index: InPlace("), "{debug}");
}

/// `open_shared` is `open` without the `.tim`/`.tip` copy: same answers,
/// same errors, buffers the caller already owns.
#[test]
fn open_shared_reads_the_same_dictionary_without_copying() {
    let b = Builder::new();
    let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("alpha", 3, 3), ("beta", 4, 4)]);
    let fis = FieldInfos {
        fields: vec![field_info(0, "text", IndexOptions::Docs)],
    };
    let shared_tim: SharedBytes = Arc::new(tim.clone());
    let shared_tip: SharedBytes = Arc::new(tip.clone());
    let fields = open_shared(
        SharedBytes::clone(&shared_tim),
        SharedBytes::clone(&shared_tip),
        &tmd,
        &fis,
        &b.id,
        &b.suffix,
        5,
    )
    .unwrap();
    let field = fields.field("text").unwrap();
    assert_eq!(field.seek_exact(b"beta").unwrap().doc_freq, 4);
    // The caller's buffers are shared, not copied: the reader holds a
    // reference to the very same allocation.
    assert!(Arc::strong_count(&shared_tim) > 1);
}

#[test]
fn read_bytes_ref_rejects_negative_length() {
    let mut buf = Vec::new();
    buf.write_vint(-1);
    let mut input = SliceInput::new(&buf);
    let err = read_bytes_ref(&mut input).unwrap_err();
    assert!(matches!(err, Error::Store(_)));
}

#[test]
fn load_node_leaf_eight_byte_output_fp() {
    // fpBytesMinus1 == 7 forces a fresh 8-byte read at fp+1.
    let mut slice = Vec::new();
    let header: u8 = LEAF_NODE_HAS_TERMS as u8 | (7 << 2); // sign=0, fpBytesMinus1=7
    slice.push(header);
    let big_fp: u64 = 0x0102_0304_0506_0708;
    slice.extend_from_slice(&big_fp.to_le_bytes()); // read fresh at fp+1
    slice.extend_from_slice(&0u64.to_le_bytes()); // over-read padding

    let node = load_node(&slice, 0).unwrap();
    assert_eq!(node.output_fp, Some(big_fp));
    assert!(node.has_terms);
}

#[test]
fn load_node_rejects_truncated_slice() {
    let slice = [0u8; 4];
    let err = load_node(&slice, 0).unwrap_err();
    assert!(matches!(err, Error::Store(_)));
}

/// Hand-builds a two-level `.tim` byte sequence: a leaf child block
/// (term `"zz"`, docFreq/totalTermFreq 1/1) followed by a non-leaf parent
/// block whose two entries are a real term (`"aa"`) and a sub-block
/// pointer (key byte `b`) resolving back to the child block via
/// `parent_fp - subCode` — the genuine "multi-level blocktree" case
/// (`SegmentTermsEnumFrame.nextNonLeaf`'s `code & 1` sub-block bit),
/// distinct from the `.tip` trie's own multi-level nesting (already
/// covered by [`collect_leaf_blocks`]'s tests) and from floor blocks.
/// Confirms `decode_block` recurses into the sub-block and reattaches its
/// key byte as a prefix, producing all three terms in the same block's
/// entry list.
#[test]
fn decode_block_recurses_into_sub_block() {
    let mut tim = Vec::new();

    // --- child (leaf) block: one term "zz", docFreq=1/totalTermFreq=1 ---
    let child_fp = tim.len();
    tim.write_vint((1 << 1) | 1); // entCount=1, isLastInFloor
    let child_suffix = b"zz";
    let child_code_l = ((child_suffix.len() as u64) << 3) | 0x04; // leaf, no compression
    tim.write_vlong(child_code_l as i64);
    tim.write_bytes(child_suffix);
    tim.write_vint((1i32 << 1) | 1); // allEqual, logical len 1
    tim.write_byte(2); // suffix length 2
    let mut child_stats = Vec::new();
    child_stats.write_vint(1 << 1); // token&1==0, docFreq=1
    tim.write_vint(child_stats.len() as i32);
    tim.write_bytes(&child_stats);
    let mut child_meta = Vec::new();
    child_meta.write_vlong(10 << 1); // docStartFP delta=10, absolute
    child_meta.write_vint(0); // singleton_doc_id (docFreq==1)
    tim.write_vint(child_meta.len() as i32);
    tim.write_bytes(&child_meta);

    // --- parent (non-leaf) block: term "aa" + sub-block "b" -> child ---
    let parent_fp = tim.len();
    tim.write_vint((2 << 1) | 1); // entCount=2, isLastInFloor
    let parent_suffix_bytes = b"ab"; // "a" (term "aa"'s suffix) then "b" (sub-block key)
    let parent_code_l = (parent_suffix_bytes.len() as u64) << 3; // non-leaf, no compression
    tim.write_vlong(parent_code_l as i64);
    tim.write_bytes(parent_suffix_bytes);

    let mut suffix_lengths = Vec::new();
    suffix_lengths.write_vint(1 << 1); // entry 0: suffix len 1, not a sub-block
    suffix_lengths.write_vint((1 << 1) | 1); // entry 1: suffix len 1, IS a sub-block
    let sub_code = (parent_fp - child_fp) as i64;
    suffix_lengths.write_vlong(sub_code); // entry 1's subCode
    tim.write_vint((suffix_lengths.len() as i32) << 1); // not allEqual
    tim.write_bytes(&suffix_lengths);

    let mut parent_stats = Vec::new();
    parent_stats.write_vint(1 << 1); // entry 0 ("aa"): docFreq=1
    tim.write_vint(parent_stats.len() as i32);
    tim.write_bytes(&parent_stats);

    let mut parent_meta = Vec::new();
    parent_meta.write_vlong(5 << 1); // entry 0's docStartFP delta=5, absolute
    parent_meta.write_vint(0); // singleton_doc_id
    tim.write_vint(parent_meta.len() as i32);
    tim.write_bytes(&parent_meta);

    let entries = decode_block(&tim, parent_fp, IndexOptions::Docs, false).unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].0, b"a");
    assert_eq!(entries[0].1.doc_freq, 1);
    assert_eq!(entries[1].0, b"bzz");
    assert_eq!(entries[1].1.doc_freq, 1);
    assert_eq!(entries[1].1.total_term_freq, 1);
}

#[test]
fn decode_block_rejects_sub_block_delta_fp_past_parent() {
    let mut tim = Vec::new();
    tim.write_vint((1 << 1) | 1); // entCount=1, isLastInFloor
    let suffix_bytes = b"x";
    let code_l = (suffix_bytes.len() as u64) << 3; // non-leaf
    tim.write_vlong(code_l as i64);
    tim.write_bytes(suffix_bytes);
    let mut suffix_lengths = Vec::new();
    suffix_lengths.write_vint((1 << 1) | 1); // suffix len 1, is a sub-block
    suffix_lengths.write_vlong(1_000_000); // subCode far exceeding this block's own fp
    tim.write_vint((suffix_lengths.len() as i32) << 1);
    tim.write_bytes(&suffix_lengths);
    tim.write_vint(0); // no stat bytes
    tim.write_vint(0); // no meta bytes

    let err = decode_block(&tim, 0, IndexOptions::Docs, false).unwrap_err();
    assert!(matches!(err, Error::Store(_)));
}

/// A `vint` byte count for one of a block's uncompressed regions that
/// exceeds the file must be an error, not an allocation. Java's
/// `new byte[n]` throws; `vec![0u8; n]` would abort the process, and an
/// abort cannot be caught at the FFI boundary.
#[test]
fn decode_block_rejects_region_lengths_larger_than_the_file() {
    // Base block: one leaf entry, then a stats length that claims far
    // more bytes than the file holds.
    fn block_with_stats_len(stats_len: i32) -> Vec<u8> {
        let mut tim = Vec::new();
        tim.write_vint((1 << 1) | 1); // entCount=1, isLastInFloor
        let suffix_bytes = b"x";
        tim.write_vlong((((suffix_bytes.len() as u64) << 3) | 0x04) as i64); // leaf, no compression
        tim.write_bytes(suffix_bytes);
        let mut suffix_lengths = Vec::new();
        suffix_lengths.write_vint(1);
        tim.write_vint((suffix_lengths.len() as i32) << 1);
        tim.write_bytes(&suffix_lengths);
        tim.write_vint(stats_len);
        tim
    }

    for stats_len in [i32::MAX, -1] {
        let tim = block_with_stats_len(stats_len);
        let err = decode_block(&tim, 0, IndexOptions::Docs, false).unwrap_err();
        assert!(
            matches!(err, Error::Store(lucene_store::Error::Corrupted(_))),
            "stats_len={stats_len} gave {err:?}"
        );
    }
}

/// The uncompressed suffix region gets the same treatment, but against
/// `remaining()` rather than a plain sign check: `numSuffixBytes` is a
/// `vlong`-derived field with 61 usable bits.
#[test]
fn decode_block_rejects_uncompressed_suffix_length_past_the_file() {
    let mut tim = Vec::new();
    tim.write_vint((1 << 1) | 1); // entCount=1, isLastInFloor
    tim.write_vlong(((1u64 << 40) << 3 | 0x04) as i64); // leaf, no compression, absurd length
    let err = decode_block(&tim, 0, IndexOptions::Docs, false).unwrap_err();
    assert!(matches!(
        err,
        Error::Store(lucene_store::Error::Corrupted(_))
    ));
}

#[test]
fn decode_block_rejects_illegal_compression_code() {
    // `code_l & 0x03 == 3` never corresponds to a `CompressionAlgorithm`
    // enum constant (only 0/NO_COMPRESSION, 1/LOWERCASE_ASCII,
    // 2/LZ4 are assigned) -- real Lucene's `CompressionAlgorithm.byCode`
    // throws for it too.
    let mut tim = Vec::new();
    tim.write_vint((1 << 1) | 1); // entCount=1, isLastInFloor
    tim.write_vlong(0x04 | 0x03); // isLeafBlock, illegal compressionAlg=3
    let err = decode_block(&tim, 0, IndexOptions::Docs, false).unwrap_err();
    assert!(matches!(
        err,
        Error::Store(lucene_store::Error::Corrupted(_))
    ));
}

#[test]
fn load_node_multi_children_with_output_and_floor() {
    // SIGN_MULTI_CHILDREN with its own output+floor data (the
    // `hasOutput`/`NON_LEAF_NODE_HAS_FLOOR` branch of
    // `loadMultiChildrenNode`, not yet exercised by the no-output
    // multi-children tests above).
    let mut slice = vec![0u8; 48];
    let parent_fp = 16usize;
    let min_label = b'a';
    let strategy_bytes_region = *b"b"; // ARRAY: one extra label 'b'.
    let strategy_bytes = strategy_bytes_region.len();
    let encoded_bytes_minus1 = 0u32; // 1-byte encoded output fp.
                                     // childrenDeltaFpBytesMinus1 = 0 (1 byte)
    let term: u32 = SIGN_MULTI_CHILDREN
            | (1 << 5) // has output
            | (encoded_bytes_minus1 << 6)
            | (CHILD_STRATEGY_ARRAY << 9)
            | (((strategy_bytes - 1) as u32) << 11)
            | ((min_label as u32) << 16);
    slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());

    // encodeFP: (floor?1:0) | (hasTerms?2:0) | (fp << 2); output fp = 9.
    let encoded_fp: u64 = NON_LEAF_NODE_HAS_FLOOR | NON_LEAF_NODE_HAS_TERMS | (9 << 2);
    assert!(encoded_fp <= 0xFF);
    // The 3-byte header only fills the low 24 bits of `term`, so byte
    // offset +3 (the word's 4th byte) is already the start of the
    // encoded-output-fp region, not part of the header -- matches
    // `loadMultiChildrenNode`'s `termLong >>> 24` inline read.
    let encoded_fp_off = parent_fp + 3;
    slice[encoded_fp_off] = encoded_fp as u8;

    // "has floor" branch: one byte childrenNum-1, then strategy bytes,
    // then children fps, then floor data.
    let children_num_off = encoded_fp_off + 1;
    slice[children_num_off] = 1; // childrenNum - 1 = 1 -> 2 children
    let strategy_fp = children_num_off + 1;
    slice[strategy_fp..strategy_fp + strategy_bytes].copy_from_slice(&strategy_bytes_region);
    let fps_fp = strategy_fp + strategy_bytes;
    // Two children, both leaf nodes at fp 0 and fp 2.
    slice[0] = LEAF_NODE_HAS_TERMS as u8;
    slice[1] = 30;
    slice[2] = LEAF_NODE_HAS_TERMS as u8;
    slice[3] = 40;
    slice[fps_fp] = parent_fp as u8; // delta to child A (fp 0)
    slice[fps_fp + 1] = (parent_fp - 2) as u8; // delta to child B (fp 2)
    let floor_fp = fps_fp + 2;
    slice[floor_fp] = 1; // numFollowFloorBlocks
    slice[floor_fp + 1] = b'z';
    slice[floor_fp + 2] = (3 << 1) | 1; // code

    let node = load_node(&slice, parent_fp).unwrap();
    assert_eq!(node.output_fp, Some(9));
    assert!(node.has_terms);
    assert_eq!(node.floor_data_fp, Some(floor_fp));
    assert_eq!(node.strategy_fp, strategy_fp);

    let mut out = Vec::new();
    let mut prefix = Vec::new();
    collect_leaf_blocks(&slice, &node, 0, &mut prefix, &mut out).unwrap();
    let mut fps: Vec<u64> = out.iter().map(|(fp, _)| *fp).collect();
    fps.sort_unstable();
    // Own output expands to blocks at fp 9 and fp 9+3=12, plus children
    // at fp 30 and fp 40.
    assert_eq!(fps, vec![9, 12, 30, 40]);
}

#[test]
fn collect_leaf_blocks_rejects_trie_nesting_too_deep() {
    let mut slice = vec![0u8; 16];
    slice[0] = LEAF_NODE_HAS_TERMS as u8;
    slice[1] = 5;
    let node = load_node(&slice, 0).unwrap();
    let mut out = Vec::new();
    let mut prefix = Vec::new();
    let err = collect_leaf_blocks(&slice, &node, 10_001, &mut prefix, &mut out).unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
}

#[test]
fn collect_leaf_blocks_rejects_single_child_delta_exceeding_parent_fp() {
    let mut slice = vec![0u8; 16];
    let term: u32 = SIGN_SINGLE_CHILD_WITHOUT_OUTPUT;
    slice[0..4].copy_from_slice(&term.to_le_bytes());
    slice[1] = b'x';
    slice[2] = 100; // child delta fp (100) > parent fp (0)
    let node = load_node(&slice, 0).unwrap();
    let mut out = Vec::new();
    let mut prefix = Vec::new();
    let err = collect_leaf_blocks(&slice, &node, 0, &mut prefix, &mut out).unwrap_err();
    assert!(matches!(err, Error::Store(_)));
}

#[test]
fn multi_children_fps_rejects_delta_exceeding_parent_fp() {
    let mut slice = vec![0u8; 24];
    let parent_fp = 8usize;
    let term: u32 = SIGN_MULTI_CHILDREN | (CHILD_STRATEGY_ARRAY << 9);
    slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());
    let strategy_fp = parent_fp + 3;
    slice[strategy_fp] = b'b'; // ARRAY strategy, one extra label
    slice[strategy_fp + 1] = 100; // delta (100) > parent fp (8)
    let node = load_node(&slice, parent_fp).unwrap();
    // Label 0 is the node's own `minChildrenLabel`, i.e. child position
    // 0, whose delta is the corrupt one.
    let err = lookup_child(&slice, &node, 0).unwrap_err();
    assert!(matches!(err, Error::Store(_)));
}

#[test]
fn decode_block_singleton_run_length_and_all_equal_suffixes() {
    // Hand-build a block with allEqual suffix lengths and a singleton
    // run (three consecutive docFreq=1/totalTermFreq=1 terms encoded via
    // the run-length token) to exercise both branches `Builder` (which
    // always emits per-entry non-run tokens and variable suffix
    // lengths) never reaches.
    let mut tim = Vec::new();
    let terms = ["aa", "bb", "cc"];
    let ent_count = terms.len() as u32;
    tim.write_vint(((ent_count << 1) | 1) as i32); // isLastInFloor

    let suffix_bytes: Vec<u8> = terms.iter().flat_map(|t| t.bytes()).collect();
    let code_l = ((suffix_bytes.len() as u64) << 3) | 0x04; // leaf, no compression
    tim.write_vlong(code_l as i64);
    tim.write_bytes(&suffix_bytes);

    // allEqual suffix lengths: all terms are 2 bytes. The logical array
    // size is still entCount (one vint-encoded length per entry) even
    // though only a single physical byte is written on disk.
    tim.write_vint(((ent_count as i32) << 1) | 1);
    tim.write_byte(2);

    // stats: one run-length token covering all three (docFreq=1 each).
    let mut stats = Vec::new();
    stats.write_vint((3 << 1) | 1); // token&1==1 -> singleton run of length 3
    tim.write_vint(stats.len() as i32);
    tim.write_bytes(&stats);

    // Postings metadata: three singleton entries, each via the bit=0
    // (docStartFP-delta) branch of `decode_term_metadata` -- legal
    // whether or not `absolute` is set, unlike the zigzag-delta branch.
    let mut meta = Vec::new();
    for singleton_doc_id in [0i32, 1, 2] {
        meta.write_vlong(0); // docStartFP delta = 0
        meta.write_vint(singleton_doc_id);
    }
    tim.write_vint(meta.len() as i32);
    tim.write_bytes(&meta);

    let entries = decode_block(&tim, 0, IndexOptions::DocsAndFreqs, false).unwrap();
    assert_eq!(entries.len(), 3);
    for (term, stats, _meta) in &entries {
        assert_eq!(term.len(), 2);
        assert_eq!(stats.doc_freq, 1);
        assert_eq!(stats.total_term_freq, 1);
    }
    assert_eq!(entries[0].0, b"aa");
    assert_eq!(entries[2].0, b"cc");
}

#[test]
fn decode_block_lz4_compressed_suffixes() {
    // Hand-built block using `code_l & 0x03 == 2` (LZ4) with the suffix
    // bytes actually run through this port's own `crate::lz4::compress`
    // (a real, general-purpose LZ4 compressor, not a fake/no-op one --
    // see `lz4.rs`'s module doc), then decoded back via `decode_block`'s
    // new LZ4 dispatch arm. This is a hand-built *test vector* (compress
    // + decompress round-trip through this port's own LZ4, cross-checked
    // separately against real Lucene bytes by
    // `tests/blocktree_compressed_fixture.rs`, which decodes an actual
    // `Lucene103BlockTreeTermsWriter`-produced LZ4 block).
    let mut tim = Vec::new();
    let terms = ["aaaaaaaa", "aaaaaaab", "aaaaaaac", "aaaaaaad"];
    let ent_count = terms.len() as u32;
    tim.write_vint(((ent_count << 1) | 1) as i32);

    let suffix_bytes: Vec<u8> = terms.iter().flat_map(|t| t.bytes()).collect();
    let compressed = crate::lz4::compress(&suffix_bytes);
    // Sanity: this input is repetitive enough that LZ4 actually shrinks
    // it -- otherwise this test wouldn't be exercising anything real.
    assert!(compressed.len() < suffix_bytes.len());

    let code_l = ((suffix_bytes.len() as u64) << 3) | 0x04 | 0x02; // leaf, LZ4
    tim.write_vlong(code_l as i64);
    tim.write_bytes(&compressed);

    tim.write_vint(((ent_count as i32) << 1) | 1); // allEqual suffix lengths
    tim.write_byte(8);

    let mut stats = Vec::new();
    stats.write_vint((ent_count << 1 | 1) as i32); // singleton run of length 4
    tim.write_vint(stats.len() as i32);
    tim.write_bytes(&stats);

    let mut meta = Vec::new();
    for singleton_doc_id in 0..ent_count as i32 {
        meta.write_vlong(0);
        meta.write_vint(singleton_doc_id);
    }
    tim.write_vint(meta.len() as i32);
    tim.write_bytes(&meta);

    let entries = decode_block(&tim, 0, IndexOptions::DocsAndFreqs, false).unwrap();
    assert_eq!(entries.len(), 4);
    for (i, (term, stats, _meta)) in entries.iter().enumerate() {
        assert_eq!(term, terms[i].as_bytes());
        assert_eq!(stats.doc_freq, 1);
        assert_eq!(stats.total_term_freq, 1);
    }
}

#[test]
fn decompress_lowercase_ascii_matches_real_lucene_compress_output() {
    // Real Lucene bytes: generated by directly invoking
    // `org.apache.lucene.util.compress.LowercaseAsciiCompression.compress`
    // (from the pinned lucene-core-10.5.0.jar) on the ASCII string below,
    // which mixes lowercase letters, digits, `.`/`-`/`_` (all
    // compressible) with two exceptions (`Z`, `!`, both outside the
    // compressible ranges) to exercise the exception-list decode branch
    // too. Not embedded in an actual on-disk `.tim` block -- see
    // `tests/blocktree_compressed_fixture.rs`'s module doc for why
    // forcing a real `IndexWriter` to choose `LOWERCASE_ASCII` (as
    // opposed to `LZ4` or `NO_COMPRESSION`) for this port's own fixtures
    // wasn't achieved in reasonable effort, and why this vector is the
    // honest fallback for that one mode.
    let original = b"the-quick_brown.fox.jumps_over-42.lazy_dogs.1234567890Z!abcdefghij";
    let compressed_hex = "7569664ef236aaa4aca0a3b3b0b8af8fa7b0b90fab362e3174607077a6b38e95134fad62fbbaa0e53068b4cf125394d5161701365a";
    let compressed: Vec<u8> = (0..compressed_hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&compressed_hex[i..i + 2], 16).unwrap())
        .collect();

    let mut r = SliceInput::new(&compressed);
    let mut out = vec![0u8; original.len()];
    decompress_lowercase_ascii(&mut r, &mut out).unwrap();
    assert_eq!(out, original);
}

#[test]
fn decompress_lowercase_ascii_rejects_out_of_range_exception_index() {
    // Hand-built, not real-Lucene-generated: 4-byte output (saved=1,
    // compressed_len=3), 3 arbitrary packed bytes, then a single
    // exception whose delta (10) pushes the cumulative index to 10,
    // past `out.len()` (4) -- must error before even reading the
    // exception's replacement value byte.
    let compressed: Vec<u8> = vec![0x61, 0x62, 0x63, 0x01, 0x0A];
    let mut r = SliceInput::new(&compressed);
    let mut out = vec![0u8; 4];
    let err = decompress_lowercase_ascii(&mut r, &mut out).unwrap_err();
    assert!(matches!(
        err,
        Error::Store(lucene_store::Error::Corrupted(_))
    ));
}

#[test]
fn decode_block_lowercase_ascii_compressed_suffixes() {
    // Same real-Lucene-generated compressed bytes as the standalone
    // decompress test above, this time threaded through the full
    // `decode_block` dispatch (`code_l & 0x03 == 1`) with a single
    // whole-block suffix rather than per-term suffix lengths (the term
    // boundary doesn't line up with the compression -- LowercaseAscii
    // compresses the concatenated suffix blob as one unit, same as
    // LZ4 -- so this test uses one giant "term" spanning the whole
    // decompressed suffix, which is enough to prove the dispatch wires
    // the compression-alg byte, the decompressed length, and the
    // decoded bytes together correctly).
    let mut tim = Vec::new();
    tim.write_vint((1 << 1) | 1); // entCount=1, isLastInFloor

    let original = b"the-quick_brown.fox.jumps_over-42.lazy_dogs.1234567890Z!abcdefghij";
    let compressed_hex = "7569664ef236aaa4aca0a3b3b0b8af8fa7b0b90fab362e3174607077a6b38e95134fad62fbbaa0e53068b4cf125394d5161701365a";
    let compressed: Vec<u8> = (0..compressed_hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&compressed_hex[i..i + 2], 16).unwrap())
        .collect();

    let code_l = ((original.len() as u64) << 3) | 0x04 | 0x01; // leaf, LOWERCASE_ASCII
    tim.write_vlong(code_l as i64);
    tim.write_bytes(&compressed);

    tim.write_vint((1 << 1) | 1); // allEqual, single entry -> irrelevant, but still 1 length byte
    tim.write_byte(original.len() as u8);

    let mut stats = Vec::new();
    stats.write_vint(1 << 1 | 1); // singleton run of length 1
    tim.write_vint(stats.len() as i32);
    tim.write_bytes(&stats);

    let mut meta = Vec::new();
    meta.write_vlong(0);
    meta.write_vint(0);
    tim.write_vint(meta.len() as i32);
    tim.write_bytes(&meta);

    let entries = decode_block(&tim, 0, IndexOptions::DocsAndFreqs, false).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, original);
    assert_eq!(entries[0].1.doc_freq, 1);
    assert_eq!(entries[0].1.total_term_freq, 1);
}

/// `.tmd` with caller-chosen per-field statistics, so each of `open`'s
/// record-level validations can be driven on its own. The `.tim`/`.tip`
/// come from [`Builder::build`] unchanged -- only the metadata varies.
#[allow(clippy::too_many_arguments)]
fn tmd_with(
    id: &[u8; ID_LENGTH],
    suffix: &str,
    num_terms: i64,
    sum_total_term_freq: i64,
    sum_doc_freq: i64,
    doc_count: i32,
    index_length: i64,
    terms_length: i64,
) -> Vec<u8> {
    let index_start = codec_util::index_header_length(TERMS_INDEX_CODEC_NAME, suffix);
    // `Builder::build`'s root node: header byte + an 8-byte output fp +
    // 8 bytes of over-read pad.
    let index_end = index_start + 17;

    let mut tmd = Vec::new();
    codec_util::write_index_header(&mut tmd, TERMS_META_CODEC_NAME, VERSION_CURRENT, id, suffix);
    codec_util::write_index_header(&mut tmd, POSTINGS_TERMS_CODEC, VERSION_CURRENT, id, suffix);
    tmd.write_vint(POSTINGS_BLOCK_SIZE);
    tmd.write_vint(1); // numFields
    tmd.write_vint(0); // field number
    tmd.write_vlong(num_terms);
    tmd.write_vlong(sum_total_term_freq);
    tmd.write_vlong(sum_doc_freq);
    tmd.write_vint(doc_count);
    tmd.write_vint(1);
    tmd.write_bytes(b"a");
    tmd.write_vint(1);
    tmd.write_bytes(b"a");
    tmd.write_vlong(index_start as i64);
    tmd.write_vlong(0); // root fp
    tmd.write_vlong(index_end as i64);
    tmd.write_i64(index_length);
    tmd.write_i64(terms_length);
    codec_util::write_footer(&mut tmd);
    tmd
}

/// The four `.tmd` record checks `Lucene103BlockTreeTermsReader`'s
/// constructor makes before it will trust a field: `numTerms > 0`,
/// `sumDocFreq >= docCount`, `sumTotalTermFreq >= sumDocFreq`, and
/// non-negative recorded `.tip`/`.tim` lengths. These are the only
/// validations left at open now that no block is read there, so each one
/// is worth its own case.
#[test]
fn tmd_record_level_validations_rejected() {
    let b = Builder::new();
    let (tim, tip, _) = b.build(IndexOptions::DocsAndFreqs, &[("a", 1, 1)]);
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
    };
    let ok_lengths = (tip.len() as i64, tim.len() as i64);
    let open_with = |tmd: Vec<u8>| open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5);

    // Sanity: the same helper with legal values opens cleanly, so each
    // rejection below is attributable to the one value it changed.
    assert!(open_with(tmd_with(
        &b.id,
        &b.suffix,
        1,
        1,
        1,
        1,
        ok_lengths.0,
        ok_lengths.1
    ))
    .is_ok());

    let err = open_with(tmd_with(
        &b.id,
        &b.suffix,
        0,
        1,
        1,
        1,
        ok_lengths.0,
        ok_lengths.1,
    ))
    .unwrap_err();
    assert!(matches!(err, Error::IllegalNumTerms(0)), "{err:?}");

    // sumDocFreq (1) < docCount (2).
    let err = open_with(tmd_with(
        &b.id,
        &b.suffix,
        1,
        5,
        1,
        2,
        ok_lengths.0,
        ok_lengths.1,
    ))
    .unwrap_err();
    assert!(matches!(err, Error::InvalidSumDocFreq { .. }), "{err:?}");

    // sumTotalTermFreq (1) < sumDocFreq (3).
    let err = open_with(tmd_with(
        &b.id,
        &b.suffix,
        1,
        1,
        3,
        1,
        ok_lengths.0,
        ok_lengths.1,
    ))
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidSumTotalTermFreq { .. }),
        "{err:?}"
    );

    for (index_length, terms_length) in [(-1, ok_lengths.1), (ok_lengths.0, -1)] {
        let err = open_with(tmd_with(
            &b.id,
            &b.suffix,
            1,
            1,
            1,
            1,
            index_length,
            terms_length,
        ))
        .unwrap_err();
        assert!(matches!(err, Error::Store(_)), "{err:?}");
    }
}

/// `TermMatcher`'s default `dead_prefix_len` -- the "this matcher can
/// never prove a prefix dead" answer that `CAN_SKIP == false` folds the
/// call site away for. Instantiated here so the default body is still
/// compiled and checked even though no production matcher reaches it.
#[test]
fn a_matcher_that_cannot_skip_reports_no_dead_prefix() {
    struct NeverSkips;
    impl TermMatcher for NeverSkips {
        fn matches(&mut self, term: &[u8]) -> bool {
            term == b"yes"
        }
    }
    const { assert!(!NeverSkips::CAN_SKIP) };
    assert!(NeverSkips.matches(b"yes"));
    assert_eq!(NeverSkips.dead_prefix_len(b"anything"), None);
}

/// Every postings-facing entry point must report a term that is not in
/// the dictionary as "nothing here" rather than erroring or reaching for
/// a `.doc`/`.pos` file -- the miss path each of them takes off the same
/// lazy `seekExact`.
#[test]
fn postings_entry_points_report_a_missing_term_as_none() {
    let b = Builder::new();
    let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, &[("alpha", 1, 1)]);
    let fis = FieldInfos {
        fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
    };
    let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
    let field = fields.field("text").unwrap();

    // A `.pos` file with nothing but its framing: none of these calls
    // gets far enough to read a byte of it.
    let mut pos_bytes = Vec::new();
    codec_util::write_index_header(
        &mut pos_bytes,
        postings::POS_CODEC,
        postings::VERSION_CURRENT,
        &b.id,
        &b.suffix,
    );
    codec_util::write_footer(&mut pos_bytes);
    let pos_in = postings::PosInput::open(&pos_bytes, &b.id, &b.suffix).unwrap();

    assert!(field.postings(b"beta", None).unwrap().is_none());
    assert!(field
        .positions(b"beta", None, &pos_in, None)
        .unwrap()
        .is_none());
    assert!(field
        .positions_flat(b"beta", None, &pos_in, None)
        .unwrap()
        .is_none());
    let (positions, starts) = field
        .positions_for_docs(b"beta", None, &pos_in, None, &[], 0, &[])
        .unwrap();
    assert!(positions.is_empty());
    assert_eq!(starts, vec![0]);
}

/// The trie region's two raw readers must reject a read that runs off the
/// end of the field's own `[indexStart, indexEnd)` slice rather than
/// panicking on the slice index -- a corrupt `.tmd` can point `rootFP`
/// anywhere inside it.
#[test]
fn trie_reads_past_the_index_slice_are_errors() {
    let slice = [0u8; 4];
    assert!(matches!(read_u64_at(&slice, 0), Err(Error::Store(_))));
    assert!(matches!(read_u8_at(&slice, 9), Err(Error::Store(_))));
    assert!(matches!(
        read_u64_n_bytes(&slice, 2, 8),
        Err(Error::Store(_))
    ));
    // In range: the low bytes of a little-endian run.
    assert_eq!(read_u64_n_bytes(&[1, 2, 3, 4], 1, 2).unwrap(), 0x0302);
}

/// A dead-prefix skip that jumps past the *last* term of the field must
/// end the walk, not fall off the end of the frame stack -- the
/// `SeekStatus::End` arm of `Intersect`'s skip.
#[test]
fn regexp_intersect_skipping_past_the_last_term_ends_the_walk() {
    let b = Builder::new();
    let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("za", 1, 1), ("zb", 1, 1)]);
    let fis = FieldInfos {
        fields: vec![field_info(0, "text", IndexOptions::Docs)],
    };
    let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
    let field = fields.field("text").unwrap();

    // `z[0-9]` shares the literal prefix `z` with both terms and matches
    // neither, and each of them is a dead prefix (nothing extending "za"
    // or "zb" can match a two-byte pattern), so the walk skips past "zb"
    // and off the end of the dictionary.
    let pattern = RegexpPattern::new(b"z[0-9]").unwrap();
    assert_eq!(field.regexp_intersect(&pattern).count(), 0);

    // The same pattern with a match in range still finds it.
    let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("z1", 1, 1), ("za", 1, 1)]);
    let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
    let field = fields.field("text").unwrap();
    let got: Vec<Vec<u8>> = field
        .regexp_intersect(&pattern)
        .map(|r| r.unwrap().0)
        .collect();
    assert_eq!(got, vec![b"z1".to_vec()]);
}

/// `TrieReader.load`'s two out-of-word fallbacks: when a node's packed fp
/// needs more bytes than fit alongside the header in the first 8-byte
/// word, Java re-reads a whole `long` at `fp + 2` (single child) or
/// `fp + 3` (multi children) instead of shifting the word it already has.
/// Neither is reachable from any fixture, because both need a `.tim`
/// large enough to push a block fp past ~2^40.
#[test]
fn load_node_reads_a_second_word_for_wide_file_pointers() {
    // Single child, 7-byte child delta fp -> `read_u64_at(fp + 2)`.
    let mut slice = vec![0u8; 32];
    slice[0] = LEAF_NODE_HAS_TERMS as u8; // leaf child at fp 0, output fp 42
    slice[1] = 42;
    let parent_fp = 16usize;
    let child_delta_bytes_minus1 = 6u32;
    slice[parent_fp] = (SIGN_SINGLE_CHILD_WITHOUT_OUTPUT | (child_delta_bytes_minus1 << 2)) as u8;
    slice[parent_fp + 1] = b'q';
    slice[parent_fp + 2] = parent_fp as u8; // delta 16 -> child at fp 0
    let node = load_node(&slice, parent_fp).unwrap();
    assert_eq!(node.sign, SIGN_SINGLE_CHILD_WITHOUT_OUTPUT);
    assert_eq!(node.child_delta_fp, parent_fp as u64);
    let child = lookup_child(&slice, &node, b'q').unwrap().expect("child q");
    assert_eq!(child.output_fp, Some(42));
    assert!(lookup_child(&slice, &node, b'r').unwrap().is_none());

    // Multi children with an output, 6-byte encoded output fp ->
    // `read_u64_at(fp + 3)`.
    let mut slice = vec![0u8; 48];
    slice[0] = LEAF_NODE_HAS_TERMS as u8;
    slice[1] = 30;
    slice[2] = LEAF_NODE_HAS_TERMS as u8;
    slice[3] = 40;
    let parent_fp = 16usize;
    let encoded_bytes_minus1 = 5u32;
    let min_label = b'a';
    let strategy_bytes = 1usize; // ARRAY: one extra label
    let term: u32 = SIGN_MULTI_CHILDREN
        | (1 << 5)
        | (encoded_bytes_minus1 << 6)
        | (CHILD_STRATEGY_ARRAY << 9)
        | (((strategy_bytes - 1) as u32) << 11)
        | ((min_label as u32) << 16);
    slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());
    // encodeFP with hasTerms, no floor, output fp 9; written at fp + 3 so
    // the 6-byte read starting there sees exactly it.
    slice[parent_fp + 3] = ((9u64 << 2) | NON_LEAF_NODE_HAS_TERMS) as u8;
    let strategy_fp = parent_fp + 4 + encoded_bytes_minus1 as usize;
    slice[strategy_fp] = b'b';
    slice[strategy_fp + 1] = parent_fp as u8; // 'a' -> fp 0
    slice[strategy_fp + 2] = (parent_fp - 2) as u8; // 'b' -> fp 2

    let node = load_node(&slice, parent_fp).unwrap();
    assert_eq!(node.output_fp, Some(9));
    assert!(node.has_terms);
    assert_eq!(node.floor_data_fp, None);
    assert_eq!(node.strategy_fp, strategy_fp);
    for (label, output) in [(b'a', 30u64), (b'b', 40)] {
        let child = lookup_child(&slice, &node, label)
            .unwrap()
            .unwrap_or_else(|| panic!("label {} should resolve", label as char));
        assert_eq!(child.output_fp, Some(output));
    }
}

#[test]
fn invalid_field_number_rejected() {
    let b = Builder::new();
    let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("a", 1, 1)]);
    // FieldInfos has no field numbered 0.
    let fis = FieldInfos {
        fields: vec![field_info(9, "other", IndexOptions::Docs)],
    };
    let err = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap_err();
    assert!(matches!(err, Error::InvalidFieldNumber(0)));
}

#[test]
fn invalid_doc_count_rejected() {
    let b = Builder::new();
    let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("a", 1, 1)]);
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::Docs)],
    };
    // docCount (1, baked into Builder::build) exceeds maxDoc=0.
    let err = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 0).unwrap_err();
    assert!(matches!(err, Error::InvalidDocCount { .. }));
}

#[test]
fn duplicate_field_rejected() {
    let id = [3u8; ID_LENGTH];
    let mut tmd = Vec::new();
    codec_util::write_index_header(&mut tmd, TERMS_META_CODEC_NAME, VERSION_CURRENT, &id, "");
    codec_util::write_index_header(&mut tmd, POSTINGS_TERMS_CODEC, VERSION_CURRENT, &id, "");
    tmd.write_vint(POSTINGS_BLOCK_SIZE);
    tmd.write_vint(2); // numFields

    // Build a single shared .tim block (one term "a") and .tip root node
    // that both field records point at, so the same field *name* is
    // reachable twice (two field numbers mapping to fields named "f").
    let mut tim = Vec::new();
    codec_util::write_index_header(&mut tim, TERMS_CODEC_NAME, VERSION_CURRENT, &id, "");
    let block_fp = tim.len();
    tim.write_vint((1 << 1) | 1);
    tim.write_vlong(((1u64 << 3) | 0x04) as i64);
    tim.write_bytes(b"a");
    tim.write_vint(1 << 1);
    tim.write_bytes(&[1]);
    let mut stats = Vec::new();
    stats.write_vint(1 << 1); // docFreq=1, non-singleton token
    tim.write_vint(stats.len() as i32);
    tim.write_bytes(&stats);
    let mut meta = Vec::new();
    meta.write_vlong(0); // docStartFP delta = 0
    meta.write_vint(0); // singletonDocID (docFreq == 1)
    tim.write_vint(meta.len() as i32);
    tim.write_bytes(&meta);
    codec_util::write_footer(&mut tim);

    let mut tip = Vec::new();
    codec_util::write_index_header(&mut tip, TERMS_INDEX_CODEC_NAME, VERSION_CURRENT, &id, "");
    let index_start = tip.len();
    let header = LEAF_NODE_HAS_TERMS as u8;
    tip.push(header);
    tip.extend_from_slice(&(block_fp as u64).to_le_bytes());
    tip.extend_from_slice(&0u64.to_le_bytes());
    let index_end = tip.len();
    codec_util::write_footer(&mut tip);

    for field_number in [0i32, 1i32] {
        tmd.write_vint(field_number);
        tmd.write_vlong(1); // numTerms
        tmd.write_vlong(1); // sumDocFreq (Docs aliasing)
        tmd.write_vint(1); // docCount
        tmd.write_vint(1);
        tmd.write_bytes(b"a");
        tmd.write_vint(1);
        tmd.write_bytes(b"a");
        tmd.write_vlong(index_start as i64);
        tmd.write_vlong(0);
        tmd.write_vlong(index_end as i64);
    }
    tmd.write_i64(tip.len() as i64); // indexLength: the whole file, footer included
    tmd.write_i64(tim.len() as i64);
    codec_util::write_footer(&mut tmd);

    let fis = FieldInfos {
        fields: vec![
            field_info(0, "f", IndexOptions::Docs),
            field_info(1, "f", IndexOptions::Docs),
        ],
    };
    let err = open(&tim, &tip, &tmd, &fis, &id, "", 5).unwrap_err();
    assert!(matches!(err, Error::DuplicateField(_)));
}

#[test]
fn index_region_out_of_bounds_rejected() {
    let b = Builder::new();
    let (tim, tip, _tmd) = b.build(IndexOptions::Docs, &[("a", 1, 1)]);
    let id = b.id;
    let suffix = b.suffix.clone();

    // Hand-build a .tmd whose indexEnd points past the end of .tip.
    let mut tmd = Vec::new();
    codec_util::write_index_header(
        &mut tmd,
        TERMS_META_CODEC_NAME,
        VERSION_CURRENT,
        &id,
        &suffix,
    );
    codec_util::write_index_header(
        &mut tmd,
        POSTINGS_TERMS_CODEC,
        VERSION_CURRENT,
        &id,
        &suffix,
    );
    tmd.write_vint(POSTINGS_BLOCK_SIZE);
    tmd.write_vint(1);
    tmd.write_vint(0);
    tmd.write_vlong(1);
    tmd.write_vlong(1);
    tmd.write_vint(1);
    tmd.write_vint(1);
    tmd.write_bytes(b"a");
    tmd.write_vint(1);
    tmd.write_bytes(b"a");
    tmd.write_vlong(0);
    tmd.write_vlong(0);
    tmd.write_vlong((tip.len() + 100) as i64); // out of bounds indexEnd
    tmd.write_i64(tip.len() as i64);
    tmd.write_i64(tim.len() as i64);
    codec_util::write_footer(&mut tmd);

    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::Docs)],
    };
    let err = open(&tim, &tip, &tmd, &fis, &id, &suffix, 5).unwrap_err();
    assert!(matches!(err, Error::Store(_)));
}

/// Structural proof (not just "lookups still work") that
/// `fixtures/data/blocktree_multilevel_index/` -- 8000 pseudo-random
/// terms, regenerated via `fixtures/src/GenBlockTreeMultilevel.java` --
/// actually forces real Lucene to write a genuine **non-leaf** `.tim`
/// block (some of its entries are in-block pointers to further-nested
/// sub-blocks, not raw term suffixes) reachable from this field's `.tip`
/// trie, i.e. the "root block -> internal block -> leaf block" case this
/// module's `decode_block`/`decode_block_at_depth` now decode. Walks the
/// same trie [`collect_leaf_blocks`] would, independently re-deriving
/// which physical `.tim` blocks are leaf vs. non-leaf by peeking each
/// one's own `isLeafBlock` bit -- this test would fail (assert
/// `saw_non_leaf_block`) if a future regen of this fixture, or a change
/// to real Lucene's own writer heuristics, stopped producing one, which
/// is exactly the failure mode a purely behavioral "every term still
/// findable" test could miss (it'd stay green even if this fixture
/// degenerated to an all-leaf-blocks shape). The full differential
/// (every term findable via the public API, matching real Lucene's own
/// ground truth) lives in `crates/lucene-codecs/tests/blocktree_multilevel_fixture.rs`,
/// same split as every other real-bytes fixture test in this crate:
/// external test = public-API differential, in-crate test = structural
/// invariant only reachable with this module's private internals.
/// The multilevel fixture's "many" field (8000 random lowercase terms):
/// opened through `open()` for the seek-state reuse tests below.
fn multilevel_many() -> BlockTreeFields {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_multilevel_index/"
    );
    let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
        .expect("run fixtures generator first (GenBlockTreeMultilevel)");
    let kv: std::collections::HashMap<String, String> = manifest
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let read_raw = |name: &str| std::fs::read(format!("{dir}{name}.raw")).unwrap();
    let mut id = [0u8; ID_LENGTH];
    let id_hex = &kv["id_hex"];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    let fnm = read_raw(&kv["fnm_file_name"]);
    let field_infos = crate::field_infos::parse(&fnm, &id, "").unwrap();
    open(
        &read_raw(&kv["tim_file_name"]),
        &read_raw(&kv["tip_file_name"]),
        &read_raw(&kv["tmd_file_name"]),
        &field_infos,
        &id,
        &kv["segment_suffix"],
        kv["max_doc"].parse().unwrap(),
    )
    .unwrap()
}

/// `iter()` borrows the field's pooled state and `drop` returns it: a
/// new enum starts unpositioned however the last one was left, and a
/// pooled lookup made while an enum holds the state still answers.
#[test]
fn terms_enums_share_the_pooled_state_and_start_unpositioned() {
    let fields = multilevel_many();
    let many = fields.field("many").unwrap();
    let first = many.iter().next().map(|(t, _)| t.to_vec()).unwrap();
    let mut it = many.iter();
    assert_eq!(it.seek_ceil(b"m"), SeekStatus::NotFound);
    let landed = it.term().unwrap().to_vec();
    // the enum holds the pooled state: a lookup runs on a fresh one
    assert!(many.seek_exact(&landed).is_some());
    let frames = it.st.stack.len();
    drop(it);
    assert!(many.scratch.slot().lock().unwrap().stack.len() >= frames);
    let mut again = many.iter();
    assert_eq!(again.term(), None, "unpositioned");
    assert_eq!(again.st.current, -1);
    assert!(again.st.stack.len() >= frames, "the frames came back");
    assert_eq!(again.next().map(|(t, _)| t.to_vec()), Some(first));
    // two enums at once: the second gets a state of its own
    let mut other = many.iter();
    assert!(other.st.stack.is_empty());
    assert_eq!(other.seek_ceil(&landed), SeekStatus::Found);
    drop(again);
    drop(other);
    assert!(many.seek_exact(&landed).is_some());
}

/// `seekCeil` to the term the enum stands on answers from `termExists`
/// without reading a block; a seek state whose stored path claims more
/// frames than the stack holds (which no sequence of seeks and `next()`s
/// produces) restarts from the root rather than trusting it.
#[test]
fn seek_state_reuse_answers_the_current_term_and_distrusts_a_broken_state() {
    let fields = multilevel_many();
    let many = fields.field("many").unwrap();
    let mut it = many.iter();
    let (first, _) = it.next().expect("a first term");
    let first = first.to_vec();
    let (second, _) = it.next().expect("a second term");
    let second = second.to_vec();
    assert_eq!(it.seek_ceil(&second), SeekStatus::Found);
    let loads = it.st.stack.iter().map(|f| f.next_ent).collect::<Vec<_>>();
    assert_eq!(it.seek_ceil(&second), SeekStatus::Found);
    assert_eq!(it.term(), Some(second.as_slice()));
    assert_eq!(
        it.st.stack.iter().map(|f| f.next_ent).collect::<Vec<_>>(),
        loads,
        "the repeated seek moved no cursor"
    );

    // A deep term, then the state claims the current frame is the root
    // while the stored prefix still carries the deep path's frames.
    let terms: Vec<Vec<u8>> = {
        let mut all = many.iter();
        let mut v = Vec::new();
        while let Some((t, _)) = all.next() {
            v.push(t.to_vec());
        }
        v
    };
    let deep = terms.iter().max_by_key(|t| t.len()).unwrap().clone();
    assert_eq!(it.seek_ceil(&deep), SeekStatus::Found);
    assert!(it.st.current > 0, "a deep term sits below the root frame");
    assert!(it.ste().shared_prefix(&deep).is_some());
    it.st.current = 0;
    assert!(it.ste().shared_prefix(&deep).is_none());
    let at = terms.partition_point(|t| t.as_slice() < deep.as_slice());
    let mut after = deep.clone();
    after.push(b'a');
    let want = terms.get(terms.partition_point(|t| t.as_slice() < after.as_slice()));
    let status = it.seek_ceil(&after);
    assert_eq!(it.term().map(<[u8]>::to_vec).as_ref(), want, "{status:?}");
    assert_eq!(terms[at], deep);
    // And the pooled exact seeks agree with the enum's.
    assert!(many.seek_exact(&first).is_some());
    assert!(many.seek_exact(&after).is_none());
    assert!(many.seek_exact(&deep).is_some());
}

#[test]
fn multilevel_fixture_reaches_a_genuine_non_leaf_block() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_multilevel_index/"
    );
    let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
        .expect("run fixtures generator first (GenBlockTreeMultilevel)");
    let kv: std::collections::HashMap<String, String> = manifest
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let read_raw = |name: &str| {
        std::fs::read(format!("{dir}{name}.raw")).unwrap_or_else(|_| panic!("missing {name}.raw"))
    };

    let tmd = read_raw(kv.get("tmd_file_name").unwrap());
    let tip = read_raw(kv.get("tip_file_name").unwrap());
    let tim = read_raw(kv.get("tim_file_name").unwrap());
    let fnm = read_raw(kv.get("fnm_file_name").unwrap());
    let id_hex = kv.get("id_hex").unwrap();
    let mut id = [0u8; ID_LENGTH];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    let suffix = kv.get("segment_suffix").unwrap();
    let field_infos = crate::field_infos::parse(&fnm, &id, "").unwrap();

    // Re-derive "many"'s index_start/root_fp/index_end the same way
    // `open()` does, so this test doesn't need any new `pub` surface on
    // this module just to expose them.
    let mut tmd_input = SliceInput::new(&tmd);
    codec_util::check_index_header(
        &mut tmd_input,
        TERMS_META_CODEC_NAME,
        VERSION_START,
        VERSION_CURRENT,
        &id,
        suffix,
    )
    .unwrap();
    codec_util::check_index_header(
        &mut tmd_input,
        POSTINGS_TERMS_CODEC,
        POSTINGS_VERSION_START,
        POSTINGS_VERSION_CURRENT,
        &id,
        suffix,
    )
    .unwrap();
    let _index_block_size = tmd_input.read_vint().unwrap();
    let num_fields = tmd_input.read_vint().unwrap();
    let mut field_index = None;
    for _ in 0..num_fields {
        let field_number = tmd_input.read_vint().unwrap();
        let _num_terms = tmd_input.read_vlong().unwrap();
        let fi = field_infos.field_by_number(field_number).unwrap();
        read_freq_pair(&mut tmd_input, fi.index_options).unwrap();
        let _doc_count = tmd_input.read_vint().unwrap();
        let _min_term = read_bytes_ref(&mut tmd_input).unwrap();
        let _max_term = read_bytes_ref(&mut tmd_input).unwrap();
        let index_start = tmd_input.read_vlong().unwrap() as usize;
        let root_fp = tmd_input.read_vlong().unwrap() as usize;
        let index_end = tmd_input.read_vlong().unwrap() as usize;
        if fi.name == "many" {
            field_index = Some((index_start, root_fp, index_end));
        }
    }
    let (index_start, root_fp, index_end) = field_index.expect("field \"many\" in .tmd");

    let index_slice = &tip[index_start..index_end];
    let root = load_node(index_slice, root_fp).unwrap();
    // This field's 8000 pseudo-random lowercase terms cover all 26
    // letters at depth 0, so the root's `(minLabel, maxLabel, labelCnt)`
    // is `('a', 'z', 26)` -- a fully dense range, for which
    // `TrieBuilder.ChildSaveStrategy.choose`'s cost formula picks
    // `REVERSE_ARRAY` (`needBytes` = `26 - 26 + 1` = 1, beating both
    // `ARRAY` = 25 and `BITS` = `ceil(26/8)` = 4). This is this fixture's
    // real-Lucene-forced multi-children strategy; `ARRAY` and `BITS` are
    // covered by the dedicated `blocktree_child_strategies_index`
    // fixture instead (see `child_strategies_fixture_forces_array_and_bits_strategies`).
    assert_eq!(root.child_save_strategy, CHILD_STRATEGY_REVERSE_ARRAY);
    assert_eq!(root.strategy_bytes, 1);
    assert_eq!(root.min_children_label, b'a');
    let mut blocks = Vec::new();
    let mut prefix = Vec::new();
    collect_leaf_blocks(index_slice, &root, 0, &mut prefix, &mut blocks).unwrap();
    assert!(
        blocks.len() > 1,
        "expected the trie to reach more than one physical block"
    );

    // Peek each reached block's own isLeafBlock bit directly (the same
    // two reads `decode_block_at_depth` starts with) without doing a
    // full decode -- purely structural.
    let mut saw_non_leaf_block = false;
    for (block_fp, _prefix) in &blocks {
        let mut r = SliceInput::new(&tim);
        r.seek(*block_fp as usize).unwrap();
        let _code = r.read_vint().unwrap();
        let code_l = r.read_vlong().unwrap() as u64;
        if (code_l & 0x04) == 0 {
            saw_non_leaf_block = true;
        }
    }
    assert!(
        saw_non_leaf_block,
        "expected at least one physical .tim block reachable from the \"many\" \
             field's trie to be non-leaf (isLeafBlock == false) -- this fixture is \
             supposed to force real Lucene into a genuine multi-level blocktree \
             (root block -> internal block -> leaf block), not just multiple \
             sibling leaf blocks/floor blocks under one trie node"
    );

    // And the full round trip through the *unmodified* public API must
    // still recover every term correctly despite that non-leaf block
    // (this is the behavioral half; the fuller differential -- matching
    // real Lucene's own sorted term list -- lives in
    // `tests/blocktree_multilevel_fixture.rs`).
    let max_doc: i32 = kv.get("max_doc").unwrap().parse().unwrap();
    let fields = open(&tim, &tip, &tmd, &field_infos, &id, suffix, max_doc).unwrap();
    let field = fields.field("many").unwrap();
    let num_terms: i64 = kv.get("field.many.numTerms").unwrap().parse().unwrap();
    assert_eq!(field.num_terms, num_terms);
    let mut it = field.iter();
    while let Some((term, stats)) = it.next() {
        assert_eq!(field.seek_exact(term).unwrap(), stats);
    }
}

/// Real-Lucene-fixture differential test proving `ChildSaveStrategy::ARRAY`
/// and `ChildSaveStrategy::BITS` (the two of the three real
/// `TrieBuilder.ChildSaveStrategy` label-encodings that
/// `multilevel_fixture_reaches_a_genuine_non_leaf_block`'s "many" field
/// does *not* land on -- that root happens to pick `REVERSE_ARRAY`, see
/// that test) are each forced onto a real `.tip` trie root node and
/// decode correctly.
///
/// `fixtures/src/GenBlockTreeChildStrategies.java` builds two fields
/// whose terms' leading bytes were hand-picked so
/// `TrieBuilder.ChildSaveStrategy.choose`'s own `needBytes` cost formula
/// -- BITS: `ceil((maxLabel-minLabel+1)/8)`, ARRAY: `labelCnt-1`,
/// REVERSE_ARRAY: `(maxLabel-minLabel+1)-labelCnt+1` -- picks a distinct
/// winner for each field (see that file's module doc for the exact
/// arithmetic): "arraystrat" (5 labels spanning printable-ASCII, distance
/// 94: BITS=12, ARRAY=4, REVERSE_ARRAY=90 -> ARRAY wins) and "bitsstrat"
/// (9 labels spaced 5 apart, distance 41: BITS=6, ARRAY=8,
/// REVERSE_ARRAY=33 -> BITS wins). This test decodes each field's root
/// trie node and asserts the exact `child_save_strategy` code real
/// Lucene's writer chose, then round-trips every term through the
/// public `open`/`seek_exact` API to prove the decode is not just
/// structurally plausible but actually correct.
#[test]
fn child_strategies_fixture_forces_array_and_bits_strategies() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_child_strategies_index/"
    );
    let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
        .expect("run fixtures generator first (GenBlockTreeChildStrategies)");
    let kv: std::collections::HashMap<String, String> = manifest
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let read_raw = |name: &str| {
        std::fs::read(format!("{dir}{name}.raw")).unwrap_or_else(|_| panic!("missing {name}.raw"))
    };
    let tmd = read_raw(kv.get("tmd_file_name").unwrap());
    let tip = read_raw(kv.get("tip_file_name").unwrap());
    let tim = read_raw(kv.get("tim_file_name").unwrap());
    let fnm = read_raw(kv.get("fnm_file_name").unwrap());
    let id_hex = kv.get("id_hex").unwrap();
    let mut id = [0u8; ID_LENGTH];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    let suffix = kv.get("segment_suffix").unwrap();
    let field_infos = crate::field_infos::parse(&fnm, &id, "").unwrap();

    let mut tmd_input = SliceInput::new(&tmd);
    codec_util::check_index_header(
        &mut tmd_input,
        TERMS_META_CODEC_NAME,
        VERSION_START,
        VERSION_CURRENT,
        &id,
        suffix,
    )
    .unwrap();
    codec_util::check_index_header(
        &mut tmd_input,
        POSTINGS_TERMS_CODEC,
        POSTINGS_VERSION_START,
        POSTINGS_VERSION_CURRENT,
        &id,
        suffix,
    )
    .unwrap();
    let _index_block_size = tmd_input.read_vint().unwrap();
    let num_fields = tmd_input.read_vint().unwrap();
    let mut field_index: std::collections::HashMap<String, (usize, usize, usize)> =
        std::collections::HashMap::new();
    for _ in 0..num_fields {
        let field_number = tmd_input.read_vint().unwrap();
        let _num_terms = tmd_input.read_vlong().unwrap();
        let fi = field_infos.field_by_number(field_number).unwrap();
        read_freq_pair(&mut tmd_input, fi.index_options).unwrap();
        let _doc_count = tmd_input.read_vint().unwrap();
        let _min_term = read_bytes_ref(&mut tmd_input).unwrap();
        let _max_term = read_bytes_ref(&mut tmd_input).unwrap();
        let index_start = tmd_input.read_vlong().unwrap() as usize;
        let root_fp = tmd_input.read_vlong().unwrap() as usize;
        let index_end = tmd_input.read_vlong().unwrap() as usize;
        field_index.insert(fi.name.clone(), (index_start, root_fp, index_end));
    }

    // "arraystrat": 5 labels, distance 94 (0x21..=0x7e) -> ARRAY (code 1).
    let (index_start, root_fp, index_end) = field_index["arraystrat"];
    let root = load_node(&tip[index_start..index_end], root_fp).unwrap();
    assert_eq!(
        root.child_save_strategy, CHILD_STRATEGY_ARRAY,
        "expected real Lucene to pick ChildSaveStrategy.ARRAY for \"arraystrat\"'s \
             root node (needBytes: BITS=12, ARRAY=4, REVERSE_ARRAY=90)"
    );
    assert_eq!(root.strategy_bytes, 4); // labelCnt - 1 = 5 - 1
    assert_eq!(root.min_children_label, 0x21);

    // "bitsstrat": 9 labels, distance 41 (0x21..=0x49) -> BITS (code 2).
    let (index_start, root_fp, index_end) = field_index["bitsstrat"];
    let root = load_node(&tip[index_start..index_end], root_fp).unwrap();
    assert_eq!(
        root.child_save_strategy, CHILD_STRATEGY_BITS,
        "expected real Lucene to pick ChildSaveStrategy.BITS for \"bitsstrat\"'s \
             root node (needBytes: BITS=6, ARRAY=8, REVERSE_ARRAY=33)"
    );
    assert_eq!(root.strategy_bytes, 6); // ceil(41 / 8)
    assert_eq!(root.min_children_label, 0x21);

    // Full round trip through the unmodified public API: every term in
    // both fields must be findable via seek_exact, proving the ARRAY and
    // BITS label decodes (not just the strategy *code*) are correct.
    let max_doc: i32 = kv.get("max_doc").unwrap().parse().unwrap();
    let fields = open(&tim, &tip, &tmd, &field_infos, &id, suffix, max_doc).unwrap();
    for name in ["arraystrat", "bitsstrat"] {
        let field = fields.field(name).unwrap();
        let expected_count: i64 = kv
            .get(&format!("field.{name}.count"))
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(field.num_terms, expected_count);
        let mut counted = 0i64;
        let mut count_it = field.iter();
        while count_it.next().is_some() {
            counted += 1;
        }
        assert_eq!(counted, expected_count);
        let terms_tsv = std::fs::read_to_string(format!("{dir}{name}.terms.tsv")).unwrap();
        let expected_terms: Vec<&str> = terms_tsv.lines().collect();
        assert_eq!(expected_terms.len() as i64, expected_count);
        for term in &expected_terms {
            let stats = field
                .seek_exact(term.as_bytes())
                .unwrap_or_else(|| panic!("term {term:?} not found in field {name}"));
            assert_eq!(stats.doc_freq, 1);
            assert_eq!(stats.total_term_freq, 1);
        }
        let mut it = field.iter();
        while let Some((term, stats)) = it.next() {
            assert_eq!(field.seek_exact(term).unwrap(), stats);
        }
    }
}

/// Real-Lucene-fixture differential test for the "root block with no
/// terms of its own (all sub-blocks)" shape: a field whose first terms
/// already diverge enough on their leading byte that real Lucene's
/// writer gives every top-level leading-byte group its own `.tim` block
/// before the root ever accumulates `minItemsInBlock` raw terms of its
/// own -- `writeBlocks(0, count)` (`Lucene103BlockTreeTermsWriter.java`)
/// then sees only `PendingBlock` entries at the root, never a loose
/// `PendingTerm`, so the root `PendingBlock`'s own `hasTerms` is `false`.
///
/// Reuses `fixtures/src/GenBlockTreeChildStrategies.java`'s existing
/// "arraystrat"/"bitsstrat" fields (5 and 9 distinct leading bytes, 30
/// terms each, comfortably above the default `minItemsInBlock=25`) rather
/// than adding a new generator -- confirmed structurally here (`assert!
/// (!root.has_terms)`) to be exactly this shape, which the
/// `child_strategies_fixture_forces_array_and_bits_strategies` test above
/// already opens successfully but never asserted the root's own
/// `has_terms` bit against.
///
/// This is the fixture that caught `open()`'s previous
/// `Error::Unsupported("root block with no terms (all sub-blocks) ...")`
/// rejection being unreachable-by-correct-data dead code: real Lucene's
/// `PendingBlock.compileIndex` always merges every sub-block's own
/// compiled trie into its parent's, so the `.tip` trie structurally
/// mirrors the `.tim` block hierarchy at *every* level including the
/// root -- `collect_leaf_blocks` already walks past a no-terms output
/// into that node's trie children unconditionally (see its own doc
/// comment), so `blocks` is never empty for this shape and needs no
/// special handling.
#[test]
fn open_field_with_no_terms_in_root_block_still_finds_every_term() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_child_strategies_index/"
    );
    let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
        .expect("run fixtures generator first (GenBlockTreeChildStrategies)");
    let kv: std::collections::HashMap<String, String> = manifest
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let read_raw = |name: &str| {
        std::fs::read(format!("{dir}{name}.raw")).unwrap_or_else(|_| panic!("missing {name}.raw"))
    };
    let tmd = read_raw(kv.get("tmd_file_name").unwrap());
    let tip = read_raw(kv.get("tip_file_name").unwrap());
    let tim = read_raw(kv.get("tim_file_name").unwrap());
    let fnm = read_raw(kv.get("fnm_file_name").unwrap());
    let id_hex = kv.get("id_hex").unwrap();
    let mut id = [0u8; ID_LENGTH];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    let suffix = kv.get("segment_suffix").unwrap();
    let field_infos = crate::field_infos::parse(&fnm, &id, "").unwrap();

    let mut tmd_input = SliceInput::new(&tmd);
    codec_util::check_index_header(
        &mut tmd_input,
        TERMS_META_CODEC_NAME,
        VERSION_START,
        VERSION_CURRENT,
        &id,
        suffix,
    )
    .unwrap();
    codec_util::check_index_header(
        &mut tmd_input,
        POSTINGS_TERMS_CODEC,
        POSTINGS_VERSION_START,
        POSTINGS_VERSION_CURRENT,
        &id,
        suffix,
    )
    .unwrap();
    let _index_block_size = tmd_input.read_vint().unwrap();
    let num_fields = tmd_input.read_vint().unwrap();
    let mut field_index: std::collections::HashMap<String, (usize, usize, usize)> =
        std::collections::HashMap::new();
    for _ in 0..num_fields {
        let field_number = tmd_input.read_vint().unwrap();
        let _num_terms = tmd_input.read_vlong().unwrap();
        let fi = field_infos.field_by_number(field_number).unwrap();
        read_freq_pair(&mut tmd_input, fi.index_options).unwrap();
        let _doc_count = tmd_input.read_vint().unwrap();
        let _min_term = read_bytes_ref(&mut tmd_input).unwrap();
        let _max_term = read_bytes_ref(&mut tmd_input).unwrap();
        let index_start = tmd_input.read_vlong().unwrap() as usize;
        let root_fp = tmd_input.read_vlong().unwrap() as usize;
        let index_end = tmd_input.read_vlong().unwrap() as usize;
        field_index.insert(fi.name.clone(), (index_start, root_fp, index_end));
    }

    // Structural confirmation, for both fields, that the root trie
    // node's own output really is `hasTerms == false` -- this is the
    // exact shape this test exists to cover, not an incidental fact.
    for name in ["arraystrat", "bitsstrat"] {
        let (index_start, root_fp, index_end) = field_index[name];
        let index_slice = &tip[index_start..index_end];
        let root = load_node(index_slice, root_fp).unwrap();
        assert!(
            !root.has_terms,
            "expected field {name:?}'s root .tip trie node to have \
                 hasTerms == false (every leading-byte group forming its \
                 own sub-block before the root ever keeps a loose term) -- \
                 if this fails, GenBlockTreeChildStrategies.java's label \
                 counts no longer force this shape and this test needs a \
                 different fixture"
        );
        assert_ne!(
            root.sign, SIGN_NO_CHILDREN,
            "a no-terms root with zero trie children would mean every \
                 term is unreachable -- this test's whole point is a \
                 no-terms root that still has children"
        );

        let mut blocks = Vec::new();
        let mut prefix = Vec::new();
        collect_leaf_blocks(index_slice, &root, 0, &mut prefix, &mut blocks).unwrap();
        assert!(
            !blocks.is_empty(),
            "field {name:?}: the previous (removed) `blocks.is_empty()` \
                 check would have wrongly rejected this fixture as \
                 unsupported"
        );
    }

    // Behavioral proof: seek_exact for every known term, plus full
    // ordered TermsEnum::next() enumeration, both still work correctly
    // through a root block that itself holds no terms.
    let max_doc: i32 = kv.get("max_doc").unwrap().parse().unwrap();
    let fields = open(&tim, &tip, &tmd, &field_infos, &id, suffix, max_doc).unwrap();
    for name in ["arraystrat", "bitsstrat"] {
        let field = fields.field(name).unwrap();
        let terms_tsv = std::fs::read_to_string(format!("{dir}{name}.terms.tsv")).unwrap();
        let mut expected_terms: Vec<&str> = terms_tsv.lines().collect();
        expected_terms.sort_unstable();

        for term in &expected_terms {
            field
                .seek_exact(term.as_bytes())
                .unwrap_or_else(|| panic!("term {term:?} not found in field {name}"));
        }

        let mut enumerated: Vec<Vec<u8>> = Vec::new();
        let mut cursor = field.iter();
        while let Some((term, _stats)) = cursor.next() {
            enumerated.push(term.to_vec());
        }
        let expected_bytes: Vec<Vec<u8>> = expected_terms
            .iter()
            .map(|t| t.as_bytes().to_vec())
            .collect();
        assert_eq!(
            enumerated, expected_bytes,
            "field {name:?}: TermsEnum::next() must yield every term in \
                 sorted order even though the root block contributes none \
                 of its own"
        );
    }
}

/// Structural proof that `fixtures/data/blocktree_deep_nesting_index/`
/// (2000 real-Lucene-written terms over a deliberately narrow `{a,b}`
/// alphabet, `minItemsInBlock=2`/`maxItemsInBlock=4` -- see
/// `fixtures/src/GenBlockTreeDeepNesting.java`'s module doc for why a
/// narrow alphabet plus small block-size thresholds is what actually
/// forces this, where `blocktree_multilevel_index`'s wide-alphabet/
/// default-thresholds fixture plateaus at a single non-leaf layer no
/// matter how many terms are added) forces real Lucene to write a
/// **chain of 4 or more nested non-leaf `.tim` blocks** -- i.e. the
/// `decode_block`/`decode_block_at_depth` sub-block recursion (see the
/// module doc's "Multi-level blocktree tries" section) is exercised at
/// real depth, not just depth 1 (root block -> one internal block ->
/// leaf, `multilevel_fixture_reaches_a_genuine_non_leaf_block`'s shape).
///
/// This independently re-derives nesting depth via its own from-scratch
/// walk of each reachable physical block's own bytes (peeking
/// `isLeafBlock`/`subFP` exactly like [`decode_block_at_depth`] does, but
/// without calling it, so this test cannot pass merely because
/// `decode_block_at_depth`'s own bookkeeping is self-consistent) and
/// asserts a *minimum* depth was reached, not just "some non-leaf block
/// exists" -- the failure mode a weaker assertion would miss is exactly
/// a future fixture regen or upstream writer change collapsing back to
/// shallow nesting while still leaving *some* non-leaf block around.
#[test]
fn deep_nesting_fixture_reaches_at_least_four_levels() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_deep_nesting_index/"
    );
    let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
        .expect("run fixtures generator first (GenBlockTreeDeepNesting)");
    let kv: std::collections::HashMap<String, String> = manifest
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let read_raw = |name: &str| {
        std::fs::read(format!("{dir}{name}.raw")).unwrap_or_else(|_| panic!("missing {name}.raw"))
    };

    let tmd = read_raw(kv.get("tmd_file_name").unwrap());
    let tip = read_raw(kv.get("tip_file_name").unwrap());
    let tim = read_raw(kv.get("tim_file_name").unwrap());
    let fnm = read_raw(kv.get("fnm_file_name").unwrap());
    let id_hex = kv.get("id_hex").unwrap();
    let mut id = [0u8; ID_LENGTH];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    let suffix = kv.get("segment_suffix").unwrap();
    let field_infos = crate::field_infos::parse(&fnm, &id, "").unwrap();

    // Re-derive "many"'s index_start/root_fp/index_end the same way
    // `open()` does, mirroring `multilevel_fixture_reaches_a_genuine_non_leaf_block`.
    let mut tmd_input = SliceInput::new(&tmd);
    codec_util::check_index_header(
        &mut tmd_input,
        TERMS_META_CODEC_NAME,
        VERSION_START,
        VERSION_CURRENT,
        &id,
        suffix,
    )
    .unwrap();
    codec_util::check_index_header(
        &mut tmd_input,
        POSTINGS_TERMS_CODEC,
        POSTINGS_VERSION_START,
        POSTINGS_VERSION_CURRENT,
        &id,
        suffix,
    )
    .unwrap();
    let _index_block_size = tmd_input.read_vint().unwrap();
    let num_fields = tmd_input.read_vint().unwrap();
    let mut field_index = None;
    for _ in 0..num_fields {
        let field_number = tmd_input.read_vint().unwrap();
        let _num_terms = tmd_input.read_vlong().unwrap();
        let fi = field_infos.field_by_number(field_number).unwrap();
        read_freq_pair(&mut tmd_input, fi.index_options).unwrap();
        let _doc_count = tmd_input.read_vint().unwrap();
        let _min_term = read_bytes_ref(&mut tmd_input).unwrap();
        let _max_term = read_bytes_ref(&mut tmd_input).unwrap();
        let index_start = tmd_input.read_vlong().unwrap() as usize;
        let root_fp = tmd_input.read_vlong().unwrap() as usize;
        let index_end = tmd_input.read_vlong().unwrap() as usize;
        if fi.name == "many" {
            field_index = Some((index_start, root_fp, index_end));
        }
    }
    let (index_start, root_fp, index_end) = field_index.expect("field \"many\" in .tmd");

    let index_slice = &tip[index_start..index_end];
    let root = load_node(index_slice, root_fp).unwrap();
    let mut blocks = Vec::new();
    let mut prefix = Vec::new();
    collect_leaf_blocks(index_slice, &root, 0, &mut prefix, &mut blocks).unwrap();
    assert!(
        blocks.len() > 1,
        "expected the trie to reach more than one physical block"
    );

    // Independently walks a chain of sub-block pointers starting at a
    // trie-reachable block, returning the number of blocks chained
    // (1 for a leaf block with no sub-block entries at all). Reads only
    // the header + suffix-lengths stream (skipping stats/meta bytes
    // wholesale, since sub-block-vs-term entry order and the subFP
    // delta live entirely in the suffix-lengths stream) -- deliberately
    // not sharing any code with `decode_block_at_depth`.
    fn block_chain_depth(tim: &[u8], fp: usize) -> usize {
        let mut r = SliceInput::new(tim);
        r.seek(fp).unwrap();
        let code = r.read_vint().unwrap();
        let ent_count = (code as u32) >> 1;
        let code_l = r.read_vlong().unwrap() as u64;
        let is_leaf_block = (code_l & 0x04) != 0;
        let num_suffix_bytes = (code_l >> 3) as usize;
        let compression_alg = code_l & 0x03;
        let mut suffix_bytes = vec![0u8; num_suffix_bytes];
        match compression_alg {
            0 => r.read_bytes(&mut suffix_bytes).unwrap(),
            1 => decompress_lowercase_ascii(&mut r, &mut suffix_bytes).unwrap(),
            2 => {
                crate::lz4::decompress(&mut r, num_suffix_bytes, &mut suffix_bytes, 0).unwrap();
            }
            _ => panic!("illegal compression code"),
        }
        let num_suffix_length_bytes_raw = r.read_vint().unwrap() as u32;
        let all_equal = (num_suffix_length_bytes_raw & 1) != 0;
        let num_suffix_length_bytes = (num_suffix_length_bytes_raw >> 1) as usize;
        let mut suffix_length_bytes = vec![0u8; num_suffix_length_bytes];
        if all_equal {
            let b = r.read_byte().unwrap();
            suffix_length_bytes.fill(b);
        } else {
            r.read_bytes(&mut suffix_length_bytes).unwrap();
        }

        if is_leaf_block {
            return 1;
        }
        let mut suffix_lengths_reader = SliceInput::new(&suffix_length_bytes);
        let mut max_child_depth = 0usize;
        for _ in 0..ent_count {
            let entry_code = suffix_lengths_reader.read_vint().unwrap() as u32;
            let is_sub_block = (entry_code & 1) != 0;
            if is_sub_block {
                let sub_code = suffix_lengths_reader.read_vlong().unwrap() as u64;
                let sub_fp = fp - sub_code as usize;
                max_child_depth = max_child_depth.max(block_chain_depth(tim, sub_fp));
            }
        }
        1 + max_child_depth
    }

    let mut max_depth = 0usize;
    for (block_fp, _prefix) in &blocks {
        max_depth = max_depth.max(block_chain_depth(&tim, *block_fp as usize));
    }
    assert!(
        max_depth >= 4,
        "expected at least a 4-block-deep chain of nested non-leaf .tim \
             blocks reachable from the \"many\" field's trie (root -> \
             internal -> internal -> ... -> leaf), got max_depth={max_depth} \
             -- if this fails, GenBlockTreeDeepNesting.java's term \
             shape/block-size thresholds no longer force this, and either \
             the fixture needs retuning or real Lucene's writer heuristics \
             changed"
    );

    // And the full round trip through the *unmodified* public API must
    // still recover every term correctly despite this much deeper
    // nesting (this is the behavioral half; the fuller differential --
    // matching real Lucene's own sorted term list -- lives in
    // `tests/blocktree_deep_nesting_fixture.rs`).
    let max_doc: i32 = kv.get("max_doc").unwrap().parse().unwrap();
    let fields = open(&tim, &tip, &tmd, &field_infos, &id, suffix, max_doc).unwrap();
    let field = fields.field("many").unwrap();
    let num_terms: i64 = kv.get("field.many.numTerms").unwrap().parse().unwrap();
    assert_eq!(field.num_terms, num_terms);
    let mut it = field.iter();
    while let Some((term, stats)) = it.next() {
        assert_eq!(field.seek_exact(term).unwrap(), stats);
    }
}

// -----------------------------------------------------------------
// The arithmetic gate: one regression test per bounded operation.
// -----------------------------------------------------------------

fn default_field_infos() -> FieldInfos {
    FieldInfos {
        fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
    }
}

const SAMPLE_TERMS: &[(&str, u32, u64)] = &[("a", 1, 1), ("ab", 2, 3), ("b", 1, 1)];

/// `rootFP` is a `.tmd` vlong that `open` casts with `as usize`, so a
/// negative one arrives at `load_node` as `usize::MAX`. A bound written as
/// `fp + 8 > slice.len()` panics on that in a debug build and wraps to `7`
/// -- passing the bound -- in a release one, so `read_u64_at` must not
/// form that sum.
#[test]
fn absurd_root_fp_is_a_decode_error_not_an_overflow() {
    let mut b = Builder::new();
    b.ov.root_fp = Some(-1);
    let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, SAMPLE_TERMS);
    let err = open(
        &tim,
        &tip,
        &tmd,
        &default_field_infos(),
        &b.id,
        &b.suffix,
        5,
    )
    .unwrap_err();
    assert!(
        format!("{err}").contains("trie node read past end of index slice"),
        "{err}"
    );
}

/// `numFields` sized a `Vec<(String, FieldTerms)>` reservation directly.
/// A `FieldTerms` is a couple of hundred bytes, so a corrupt vint bought
/// a several-hundred-gigabyte reservation -- and an allocation failure
/// *aborts*, which `catch_unwind` at the FFI boundary cannot intercept.
#[test]
fn absurd_num_fields_errors_instead_of_reserving_for_it() {
    let mut b = Builder::new();
    b.ov.num_fields = Some(i32::MAX);
    let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, SAMPLE_TERMS);
    let err = open(
        &tim,
        &tip,
        &tmd,
        &default_field_infos(),
        &b.id,
        &b.suffix,
        5,
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidNumFields(n) if n == i32::MAX),
        "{err}"
    );
}

/// `minTerm`/`maxTerm` are vint-length-prefixed and sized `vec![0u8; n]`
/// straight off that vint. Java's `new BytesRef(numBytes)` raises a
/// catchable `OutOfMemoryError`; the Rust equivalent aborts.
#[test]
fn absurd_min_term_length_errors_instead_of_allocating_for_it() {
    let mut b = Builder::new();
    b.ov.min_term_len = Some(i32::MAX);
    let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, SAMPLE_TERMS);
    let err = open(
        &tim,
        &tip,
        &tmd,
        &default_field_infos(),
        &b.id,
        &b.suffix,
        5,
    )
    .unwrap_err();
    assert!(
        format!("{err}").contains("exceeds") && format!("{err}").contains("remaining bytes"),
        "{err}"
    );
}

/// `totalTermFreq = docFreq + <vlong>` is a `long` add in Java, which
/// wraps silently; in Rust it panics in a debug build and, in a release
/// one, produces a *negative* frequency every scorer downstream would
/// treat as real.
#[test]
fn total_term_freq_overflow_is_a_decode_error() {
    let mut b = Builder::new();
    b.ov.ttf_delta = Some(i64::MAX);
    let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, SAMPLE_TERMS);
    let fields = open(
        &tim,
        &tip,
        &tmd,
        &default_field_infos(),
        &b.id,
        &b.suffix,
        5,
    )
    .unwrap();
    let field = fields.field("text").unwrap();
    let err = field.try_seek_exact(b"a").unwrap_err();
    assert!(
        format!("{err}").contains("totalTermFreq overflows"),
        "{err}"
    );
}

/// A block's `entCount` drives every scan cursor and
/// `binary_search_term_leaf`'s bisection bounds. The writer emits at
/// least one vint per entry into the suffix-lengths region, so
/// `entCount > numSuffixLengthBytes` is corruption -- and checking it
/// once per block load is what bounds `ent_count` for every per-entry
/// proof in this module.
#[test]
fn ent_count_past_the_suffix_lengths_region_is_a_decode_error() {
    let mut b = Builder::new();
    b.ov.ent_count = Some(50);
    let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, SAMPLE_TERMS);
    let fields = open(
        &tim,
        &tip,
        &tmd,
        &default_field_infos(),
        &b.id,
        &b.suffix,
        5,
    )
    .unwrap();
    let field = fields.field("text").unwrap();
    let err = field.try_seek_exact(b"a").unwrap_err();
    assert!(
        format!("{err}").contains("exceeds its 3-byte suffix-lengths region"),
        "{err}"
    );
}

/// `binarySearchTermLeaf`'s midpoint is `(start + end) >>> 1` in Java --
/// an *unsigned* shift, which is what keeps the midpoint correct once the
/// sum passes `Integer.MAX_VALUE`. This port had `>>`, so the same sum
/// panicked in a debug build and produced a negative `mid` in a release
/// one. Driven at the `Frame` level because reaching it through
/// `load_block` would need a gigabyte-scale suffix-lengths region.
#[test]
fn binary_search_midpoint_does_not_overflow_for_a_huge_ent_count() {
    let mut frame = Frame {
        ent_count: i32::MAX as u32,
        next_ent: 0,
        is_leaf_block: true,
        all_equal: true,
        // One vint, `0`: every suffix in the block is zero bytes long,
        // so the `suffix_length * ent_count <= suffix_bytes_len` guard
        // passes with an empty suffix region and the bisection runs its
        // full ~31 steps against `start + end` near `2 * i32::MAX`.
        suffix_length_bytes: vec![0u8],
        suffix_length_bytes_len: 1,
        ..Frame::default()
    };
    let mut term = TermBuf::default();
    let mut term_exists = false;
    let status = frame
        .binary_search_term_leaf(&mut term, b"zzz", true, &mut term_exists)
        .unwrap();
    // Every suffix is empty, so every one of them sorts before "zzz":
    // the bisection walks to the very last entry and reports End.
    assert_eq!(status, SeekStatus::End);
    assert_eq!(frame.next_ent, i32::MAX);
}

/// Recomputes the trailing 8-byte CRC32 of a codec-footer-terminated
/// buffer in place, so a byte flip in the body is not simply "caught" by
/// the checksum -- c15/c19/c25's shape.
fn resign_footer(buf: &mut [u8]) {
    let len = buf.len();
    let checksum = crc32fast::hash(&buf[..len - 8]) as u64;
    buf[len - 8..].copy_from_slice(&checksum.to_be_bytes());
}

/// Re-signed single-byte corruption sweep over the whole `.tim` block
/// body, the whole `.tip` trie region and the whole `.tmd` record region:
/// flip one bit, re-sign the footer so the CRC cannot "catch" the
/// corruption, and require a typed error or a clean decode --
/// never a panic, and never an allocation abort `catch_unwind` cannot
/// intercept.
///
/// The assertion is deliberately *not* "every corruption is rejected":
/// plenty of single-bit flips produce a self-consistent dictionary that
/// answers differently but well-formedly, which is exactly what the
/// checksum such re-signing defeats exists for.
#[test]
fn every_resigned_single_byte_terms_dict_corruption_is_an_error_or_a_clean_decode() {
    let b = Builder::new();
    let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, SAMPLE_TERMS);
    let fis = default_field_infos();
    // Bodies, past the index header and short of the footer. The `.tmd`
    // also records both other files' lengths in its last 16 body bytes; a
    // flip there is caught by `retrieve_checksum_with_expected_length`
    // rather than by anything semantic, so it is excluded as
    // uninteresting.
    let tim_body = 8..tim.len() - codec_util::FOOTER_LENGTH;
    let tip_body = 8..tip.len() - codec_util::FOOTER_LENGTH;
    let tmd_body = 8..tmd.len() - codec_util::FOOTER_LENGTH - 16;

    let mut total = 0usize;
    let mut rejected = 0usize;

    let exercise = |tim: &[u8], tip: &[u8], tmd: &[u8]| -> bool {
        match open(tim, tip, tmd, &fis, &b.id, &b.suffix, 5) {
            Err(_) => true,
            Ok(fields) => {
                let Some(field) = fields.field("text") else {
                    return true;
                };
                let mut bad = false;
                for (term, _, _) in SAMPLE_TERMS {
                    if field.try_seek_exact(term.as_bytes()).is_err() {
                        bad = true;
                    }
                }
                let mut it = field.iter();
                if it.try_seek_ceil(b"aa").is_err() {
                    bad = true;
                }
                let mut it = field.iter();
                loop {
                    match it.try_next() {
                        Ok(Some(_)) => {}
                        Ok(None) => break,
                        Err(_) => {
                            bad = true;
                            break;
                        }
                    }
                }
                bad
            }
        }
    };

    for off in tim_body {
        for mask in [0x01u8, 0x80] {
            let mut corrupt = tim.clone();
            corrupt[off] ^= mask;
            resign_footer(&mut corrupt);
            total += 1;
            if exercise(&corrupt, &tip, &tmd) {
                rejected += 1;
            }
        }
    }
    for off in tip_body {
        for mask in [0x01u8, 0x80] {
            let mut corrupt = tip.clone();
            corrupt[off] ^= mask;
            resign_footer(&mut corrupt);
            total += 1;
            if exercise(&tim, &corrupt, &tmd) {
                rejected += 1;
            }
        }
    }
    for off in tmd_body {
        for mask in [0x01u8, 0x80] {
            let mut corrupt = tmd.clone();
            corrupt[off] ^= mask;
            resign_footer(&mut corrupt);
            total += 1;
            if exercise(&tim, &tip, &corrupt) {
                rejected += 1;
            }
        }
    }

    // Measured when this was written: 391 of 436. The rest decode to a
    // different but self-consistent dictionary -- so a rate below
    // 100% is the norm, not a gap. What this pins is that nothing *panics or
    // aborts*, plus a floor so a future change that stops bounding
    // something fails loudly.
    assert_eq!(total, 436);
    assert!(
        rejected >= 385,
        "only {rejected} of {total} re-signed .tim/.tip/.tmd corruptions were rejected"
    );
}
