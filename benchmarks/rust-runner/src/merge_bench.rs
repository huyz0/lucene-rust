//! Merge throughput: the three stored-fields merge strategies, the postings
//! merge, and the BKD point merge, each measured against the implementation
//! this port had before batch `c4-merge-fastpath`.
//!
//! Every "before" number here is produced by code in this file that reproduces
//! the old algorithm exactly, over the same inputs, in the same process --
//! not by a remembered measurement.
//!
//! Usage: `merge-bench [docs-per-segment] [segments]`.

use std::collections::BTreeSet;
use std::time::Instant;

use lucene_codecs::blocktree;
use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::points::{self, WritePointsField};
use lucene_codecs::postings::{DocInput, PostingsFlags};
use lucene_codecs::postings_writer::TermPostings;
use lucene_codecs::stored_fields::{self, Document, FieldValue, StoredField};
use lucene_codecs::term_vectors::{
    self, TermVectorField, TermVectorTerm, TermVectorsDocument, TermVectorsWriter,
};
use lucene_index::index_writer::{per_field_codec_suffix, IndexWriter, POSTINGS_FORMAT_NAME};
use lucene_index::merge::{
    merge_segments, merge_stored_only_segments, MergeOptions, MergeSortKeySpec, MergeSource,
    SourcePostings,
};
use lucene_index::segment_info::{IndexSortField, LuceneVersion};
use lucene_store::{Directory, FsDirectory};
use lucene_util::fixed_bit_set::FixedBitSet;

fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    }
}

fn field(name: &str, number: i32, indexed: bool) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: true,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: if indexed {
            IndexOptions::DocsAndFreqs
        } else {
            IndexOptions::None
        },
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Byte,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

/// A document with a stored id and a stored body, so the payload per document
/// is realistic (~120 bytes) rather than a handful of bytes.
fn doc(seg: usize, n: usize, vocab: &[String]) -> Document {
    let body = format!(
        "{} {} {} {} {} {}",
        vocab[n % vocab.len()],
        vocab[(n / 3) % vocab.len()],
        vocab[(n / 7) % vocab.len()],
        vocab[(n / 11) % vocab.len()],
        vocab[(n / 13) % vocab.len()],
        vocab[(n / 17) % vocab.len()],
    );
    Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(format!("doc{seg}-{n}")),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::String(body),
            },
        ],
    }
}

struct Flushed {
    fdt: Vec<u8>,
    fdx: Vec<u8>,
    fdm: Vec<u8>,
    fields: Vec<FieldInfo>,
    segment_id: [u8; 16],
}

fn tempdir(tag: &str) -> String {
    let dir = std::env::temp_dir().join(format!(
        "lucene-rust-merge-bench-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_str().unwrap().to_string()
}

fn flush(tmp: &str, name: &str, id: u8, fields: &[FieldInfo], docs: &[Document]) -> Flushed {
    let dir = FsDirectory::open(tmp);
    let segment_id = [id; 16];
    lucene_index::segment_writer::flush_stored_only_segment(
        &dir,
        name,
        segment_id,
        "Lucene104",
        version(),
        fields,
        docs,
        false,
    )
    .unwrap();
    let read =
        |ext: &str| std::fs::read(std::path::Path::new(tmp).join(format!("{name}.{ext}"))).unwrap();
    Flushed {
        fdt: read("fdt"),
        fdx: read("fdx"),
        fdm: read("fdm"),
        fields: fields.to_vec(),
        segment_id,
    }
}

/// The stored-fields merge as this port did it before `c4`: every document
/// materialised into an owned `Document`, every field remapped, then the whole
/// list recompressed.
fn legacy_stored_fields_merge(
    sources: &[MergeSource],
    per_source_maps: &[std::collections::HashMap<i32, i32>],
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut merged_docs: Vec<Document> = Vec::new();
    for (source, map) in sources.iter().zip(per_source_maps) {
        for doc_id in 0..source.reader.max_doc() {
            let live = source
                .live_docs
                .map(|b| b.get(doc_id as usize))
                .unwrap_or(true);
            if !live {
                continue;
            }
            let mut d = source.reader.document(doc_id).unwrap();
            for f in &mut d.fields {
                f.field_number = *map.get(&f.field_number).unwrap();
            }
            merged_docs.push(d);
        }
    }
    stored_fields::write_best_speed(&merged_docs, &[9u8; 16], "")
}

fn report(label: &str, docs: usize, bytes: usize, before: f64, after: f64) {
    let mb = bytes as f64 / (1024.0 * 1024.0);
    println!(
        "  {label:<34} before {:>8.1} ms ({:>9.0} docs/s, {:>7.1} MB/s)   after {:>8.1} ms ({:>9.0} docs/s, {:>7.1} MB/s)   {:>5.2}x",
        before * 1e3,
        docs as f64 / before,
        mb / before,
        after * 1e3,
        docs as f64 / after,
        mb / after,
        before / after
    );
}

fn time<T>(reps: usize, mut f: impl FnMut() -> T) -> f64 {
    // One warm-up, then the best of `reps` -- the merge allocates heavily, so
    // the minimum is the least allocator-noise-contaminated sample.
    f();
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        let out = f();
        let e = t.elapsed().as_secs_f64();
        std::hint::black_box(out);
        best = best.min(e);
    }
    best
}

fn stored_fields_scenarios(docs_per_segment: usize, num_segments: usize, vocab: &[String]) {
    let tmp = tempdir("sf");
    let fields = vec![field("id", 0, false), field("body", 1, false)];
    // Non-contiguous field numbers: `reconcile_field_numbers` renumbers them
    // to 0/1, so *every* source is a non-matching reader and the whole merge
    // takes the VISITOR path (a permutation would leave source 0 matching,
    // since the merged numbering is seeded from the first source).
    let permuted = vec![field("id", 5, false), field("body", 9, false)];

    let mut flushed = Vec::new();
    let mut permuted_flushed = Vec::new();
    for seg in 0..num_segments {
        let docs: Vec<Document> = (0..docs_per_segment).map(|n| doc(seg, n, vocab)).collect();
        flushed.push(flush(
            &tmp,
            &format!("_{seg}"),
            seg as u8 + 1,
            &fields,
            &docs,
        ));
        let swapped: Vec<Document> = docs
            .iter()
            .map(|d| Document {
                fields: vec![
                    StoredField {
                        field_number: 5,
                        value: d.fields[0].value.clone(),
                    },
                    StoredField {
                        field_number: 9,
                        value: d.fields[1].value.clone(),
                    },
                ],
            })
            .collect();
        permuted_flushed.push(flush(
            &tmp,
            &format!("_p{seg}"),
            seg as u8 + 100,
            &permuted,
            &swapped,
        ));
    }
    let total_bytes: usize = flushed.iter().map(|f| f.fdt.len()).sum();
    let total_docs = docs_per_segment * num_segments;

    let readers: Vec<_> = flushed
        .iter()
        .map(|f| stored_fields::open(&f.fdt, &f.fdx, &f.fdm, &f.segment_id, "").unwrap())
        .collect();
    let permuted_readers: Vec<_> = permuted_flushed
        .iter()
        .map(|f| stored_fields::open(&f.fdt, &f.fdx, &f.fdm, &f.segment_id, "").unwrap())
        .collect();

    // -- BULK: no deletions, identical field numbering.
    let clean: Vec<MergeSource> = flushed
        .iter()
        .zip(&readers)
        .map(|(f, r)| MergeSource::stored_only(&f.fields, r, None, Some(version())))
        .collect();
    run_case("BULK (no deletions)", &clean, total_docs, total_bytes, &tmp);

    // -- DOC: every third document deleted.
    let live: Vec<FixedBitSet> = (0..num_segments)
        .map(|_| {
            let mut b = FixedBitSet::new(docs_per_segment);
            for i in 0..docs_per_segment {
                if !i.is_multiple_of(3) {
                    b.set(i);
                }
            }
            b
        })
        .collect();
    let deleted: Vec<MergeSource> = flushed
        .iter()
        .zip(&readers)
        .zip(&live)
        .map(|((f, r), l)| MergeSource::stored_only(&f.fields, r, Some(l), Some(version())))
        .collect();
    let surviving = total_docs - total_docs.div_ceil(3);
    run_case("DOC (1/3 deleted)", &deleted, surviving, total_bytes, &tmp);

    // -- VISITOR: every source's fields are renumbered by the merge.
    let renumbered: Vec<MergeSource> = permuted_flushed
        .iter()
        .zip(&permuted_readers)
        .map(|(f, r)| MergeSource::stored_only(&f.fields, r, None, Some(version())))
        .collect();
    run_case(
        "VISITOR (renumbered fields)",
        &renumbered,
        total_docs,
        total_bytes,
        &tmp,
    );

    std::fs::remove_dir_all(&tmp).ok();
}

/// What an **index sort** costs a merge, measured against the identical merge
/// of the identical sources with the sort switched off.
///
/// The sort is not free the way it is at flush time (c17 measured ~1%), and
/// the reason is worth naming: the stored-fields and term-vector merges copy
/// whole *compressed chunks* only for a run of consecutive documents from one
/// source (c4 measured that path at 520x). An index sort interleaves the
/// sources, so those runs collapse to roughly one document each and the merge
/// falls back to copying serialized documents. That is not a regression to
/// fix -- byte-copying a chunk is illegal when the document order changes --
/// but it is a cost that must be visible rather than silently traded away.
fn sorted_merge_scenario(docs_per_segment: usize, num_segments: usize, vocab: &[String]) {
    let tmp = tempdir("sorted");
    let fields = vec![field("id", 0, false), field("body", 1, false)];
    let mut flushed = Vec::new();
    for seg in 0..num_segments {
        let docs: Vec<Document> = (0..docs_per_segment).map(|n| doc(seg, n, vocab)).collect();
        flushed.push(flush(
            &tmp,
            &format!("_s{seg}"),
            seg as u8 + 1,
            &fields,
            &docs,
        ));
    }
    let total_bytes: usize = flushed.iter().map(|f| f.fdt.len()).sum();
    let total_docs = docs_per_segment * num_segments;
    let readers: Vec<_> = flushed
        .iter()
        .map(|f| stored_fields::open(&f.fdt, &f.fdx, &f.fdm, &f.segment_id, "").unwrap())
        .collect();
    let sources: Vec<MergeSource> = flushed
        .iter()
        .zip(&readers)
        .map(|(f, r)| MergeSource::stored_only(&f.fields, r, None, Some(version())))
        .collect();

    // Each source is internally sorted by its own key, and the sources'
    // key ranges fully overlap -- the worst case for run detection, and the
    // normal case for an index-sorted index (every flush covers the whole key
    // range).
    let keys: Vec<Vec<Option<i64>>> = (0..num_segments)
        .map(|_| (0..docs_per_segment).map(|n| Some(n as i64)).collect())
        .collect();
    let key_slices: Vec<&[Option<i64>]> = keys.iter().map(|k| k.as_slice()).collect();
    let sort = IndexSortField::long("rank", false, Some(i64::MAX));
    let specs = vec![MergeSortKeySpec {
        sort: &sort,
        per_source_keys: &key_slices,
    }];

    let dir = FsDirectory::open(&tmp);
    let unsorted = time(3, || {
        merge_segments(
            &dir,
            &sources,
            None,
            &MergeOptions::default(),
            "_mu",
            [8u8; 16],
            "Lucene104",
            version(),
        )
        .unwrap()
    });
    let sorted = time(3, || {
        merge_segments(
            &dir,
            &sources,
            Some(&specs),
            &MergeOptions::default(),
            "_ms",
            [9u8; 16],
            "Lucene104",
            version(),
        )
        .unwrap()
    });
    // `report`'s "before/after" columns read here as "unsorted/sorted", and
    // the ratio is therefore how much *cheaper* the unsorted merge is.
    report(
        "concatenated vs index-sorted",
        total_docs,
        total_bytes,
        unsorted,
        sorted,
    );
    std::fs::remove_dir_all(&tmp).ok();
}

fn run_case(label: &str, sources: &[MergeSource], docs: usize, bytes: usize, tmp: &str) {
    let sources_fields: Vec<&[FieldInfo]> = sources.iter().map(|s| s.field_infos).collect();
    let (_, maps) = lucene_index::merge::reconcile_field_numbers(&sources_fields).unwrap();
    let before = time(3, || legacy_stored_fields_merge(sources, &maps));
    let dir = FsDirectory::open(tmp);
    let after = time(3, || {
        merge_stored_only_segments(&dir, sources, "_m", [9u8; 16], "Lucene104", version()).unwrap()
    });
    report(label, docs, bytes, before, after);
}

/// The postings merge as this port did it before `c4`: materialise every
/// distinct term of the field into a `BTreeSet`, then seek each source's
/// dictionary once per term.
fn legacy_postings_merge(
    per_source: &[(&blocktree::FieldTerms, &DocInput<'_>)],
    doc_id_maps: &[Vec<i32>],
) -> Vec<TermPostings> {
    let mut all_terms: BTreeSet<Vec<u8>> = BTreeSet::new();
    for (terms, _) in per_source {
        let mut it = terms.iter();
        while let Some((term, _)) = it.next() {
            all_terms.insert(term.to_vec());
        }
    }
    let mut out = Vec::with_capacity(all_terms.len());
    for term in all_terms {
        let mut docs: Vec<(i32, i32)> = Vec::new();
        for (src_idx, (terms, doc_in)) in per_source.iter().enumerate() {
            let Some(p) = terms.postings(&term, Some(doc_in)).unwrap() else {
                continue;
            };
            for (&d, &f) in p.docs.iter().zip(p.freqs.iter()) {
                let map = &doc_id_maps[src_idx];
                if d >= 0 && (d as usize) < map.len() && map[d as usize] >= 0 {
                    docs.push((map[d as usize], f));
                }
            }
        }
        if !docs.is_empty() {
            out.push(TermPostings {
                term,
                docs,
                positions: Vec::new(),
                offsets: Vec::new(),
                payload_bytes: Vec::new(),
                payload_lengths: Vec::new(),
            });
        }
    }
    out
}

fn postings_scenario(docs_per_segment: usize, num_segments: usize, vocab: &[String]) {
    let tmp = tempdir("post");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("id", 0, false), field("body", 1, true)];
    let mut writer = IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    for seg in 0..num_segments {
        for n in 0..docs_per_segment {
            writer.add_document(doc(seg, n, vocab)).unwrap();
        }
        writer.commit().unwrap();
    }

    let suffix = per_field_codec_suffix(POSTINGS_FORMAT_NAME);
    let postings_infos = FieldInfos {
        fields: vec![field("body", 1, true)],
    };
    struct Opened {
        fdt: Vec<u8>,
        fdx: Vec<u8>,
        fdm: Vec<u8>,
        tim: Vec<u8>,
        tip: Vec<u8>,
        tmd: Vec<u8>,
        doc: Vec<u8>,
        id: [u8; 16],
    }
    let opened: Vec<Opened> = writer
        .segment_infos()
        .segments
        .iter()
        .map(|sci| {
            let n = &sci.segment_name;
            let seg = lucene_index::index_writer::per_field_segment(n, POSTINGS_FORMAT_NAME);
            Opened {
                fdt: dir.open(&format!("{n}.fdt")).unwrap().to_vec(),
                fdx: dir.open(&format!("{n}.fdx")).unwrap().to_vec(),
                fdm: dir.open(&format!("{n}.fdm")).unwrap().to_vec(),
                tim: dir.open(&format!("{seg}.tim")).unwrap().to_vec(),
                tip: dir.open(&format!("{seg}.tip")).unwrap().to_vec(),
                tmd: dir.open(&format!("{seg}.tmd")).unwrap().to_vec(),
                doc: dir.open(&format!("{seg}.doc")).unwrap().to_vec(),
                id: sci.segment_id,
            }
        })
        .collect();

    let readers: Vec<_> = opened
        .iter()
        .map(|o| stored_fields::open(&o.fdt, &o.fdx, &o.fdm, &o.id, "").unwrap())
        .collect();
    let bts: Vec<_> = opened
        .iter()
        .map(|o| {
            blocktree::open(
                &o.tim,
                &o.tip,
                &o.tmd,
                &postings_infos,
                &o.id,
                &suffix,
                docs_per_segment as i32,
            )
            .unwrap()
        })
        .collect();
    let doc_ins: Vec<_> = opened
        .iter()
        .map(|o| DocInput::open(&o.doc, &o.id, &suffix).unwrap())
        .collect();

    let per_source: Vec<(&blocktree::FieldTerms, &DocInput<'_>)> = bts
        .iter()
        .zip(&doc_ins)
        .map(|(bt, d)| (bt.field("body").unwrap(), d))
        .collect();
    let doc_id_maps: Vec<Vec<i32>> = (0..num_segments)
        .map(|s| {
            (0..docs_per_segment)
                .map(|i| (s * docs_per_segment + i) as i32)
                .collect()
        })
        .collect();
    let term_count = per_source[0].0.num_terms;

    let source_postings: Vec<Vec<SourcePostings>> = bts
        .iter()
        .zip(&doc_ins)
        .map(|(bt, d)| {
            vec![SourcePostings {
                field_number: 1,
                field_terms: bt.field("body").unwrap(),
                doc_in: Some(d),
                pos_in: None,
                pay_in: None,
            }]
        })
        .collect();
    let sources: Vec<MergeSource> = opened
        .iter()
        .zip(&readers)
        .zip(&source_postings)
        .map(|((_, r), sp)| MergeSource {
            field_infos: &fields,
            reader: r,
            live_docs: None,
            numeric_doc_values: &[],
            binary_doc_values: &[],
            sorted_doc_values: &[],
            sorted_numeric_doc_values: &[],
            sorted_set_doc_values: &[],
            norms: &[],
            term_vectors: None,
            postings: sp,
            points: &[],
            vectors: None,
            min_version: None,
            has_blocks: false,
        })
        .collect();

    let before = time(3, || legacy_postings_merge(&per_source, &doc_id_maps));
    let mdir = FsDirectory::open(&tmp);
    let after_full = time(3, || {
        merge_stored_only_segments(&mdir, &sources, "_pm", [9u8; 16], "Lucene104", version())
            .unwrap()
    });
    // The stored-fields half of the same merge, so the postings half can be
    // isolated: same sources, no postings supplied.
    let sources_no_postings: Vec<MergeSource> = opened
        .iter()
        .zip(&readers)
        .map(|(_, r)| MergeSource::stored_only(&fields, r, None, Some(version())))
        .collect();
    let after_stored_only = time(3, || {
        merge_stored_only_segments(
            &mdir,
            &sources_no_postings,
            "_pm2",
            [8u8; 16],
            "Lucene104",
            version(),
        )
        .unwrap()
    });
    let after = (after_full - after_stored_only).max(1e-6);
    println!(
        "  {:<34} before {:>8.1} ms   after {:>8.1} ms   {:>5.2}x   ({} terms x {} sources)",
        "postings k-way merge",
        before * 1e3,
        after * 1e3,
        before / after,
        term_count,
        num_segments
    );

    std::fs::remove_dir_all(&tmp).ok();
}

fn points_scenario(total_points: usize) {
    // The BKD side: the same merged 1-D point stream, handed to the writer
    // already sorted (what `merge_point_streams` now produces) versus in
    // per-source-concatenated order (what it produced before).
    let per_source = 4usize;
    let each = total_points / per_source;
    let mut streams: Vec<Vec<(i32, Vec<u8>)>> = Vec::new();
    for s in 0..per_source {
        let mut v: Vec<(i32, Vec<u8>)> = (0..each)
            .map(|i| {
                let value = ((i * per_source + s) as u32).wrapping_mul(2_654_435_761);
                ((s * each + i) as i32, value.to_be_bytes().to_vec())
            })
            .collect();
        v.sort_by(|a, b| a.1.cmp(&b.1));
        streams.push(v);
    }
    let concatenated: Vec<(i32, Vec<u8>)> = streams.iter().flatten().cloned().collect();
    let mut merged = concatenated.clone();
    merged.sort_by(|a, b| (a.1.as_slice(), a.0).cmp(&(b.1.as_slice(), b.0)));

    let build = |points: &Vec<(i32, Vec<u8>)>| WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 4,
        points: points.clone(),
    };
    let before = time(3, || {
        points::write(
            &[build(&concatenated)],
            points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
            &[1u8; 16],
            "",
        )
        .unwrap()
    });
    let after = time(3, || {
        points::write(
            &[build(&merged)],
            points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
            &[1u8; 16],
            "",
        )
        .unwrap()
    });
    let bytes = total_points * 4;
    report("BKD 1-D points::write", total_points, bytes, before, after);
}

// ---------------------------------------------------------------------------
// Term vectors (`c8-tv-chunking`).
//
// Every "before" figure here is the pre-`c8` writer: one chunk per segment.
// That is reproduced exactly -- not approximated -- by the same writer given
// `Lucene90CompressingTermVectorsWriter`'s own `chunkSize`/`maxDocsPerChunk`
// constructor parameters set past any segment this benchmark builds, which is
// what `write_best_speed(chunk_docs = docs.len())` did.
// ---------------------------------------------------------------------------

/// The pre-`c8` geometry: one chunk for the whole segment.
const SINGLE_CHUNK: usize = i32::MAX as usize;

/// A term-vector document with two fields, ~60 bytes of term text, positions
/// on both and offsets+payloads on one -- a realistic flush shape, and enough
/// text that the 4 096-byte trigger closes chunks before the 128-document one.
fn tv_doc(n: usize, vocab: &[String]) -> TermVectorsDocument {
    let terms: Vec<TermVectorTerm> = {
        let mut words: Vec<String> = (0..4)
            .map(|k| vocab[(n * 7 + k * 31) % vocab.len()].clone())
            .collect();
        words.sort();
        words.dedup();
        words
            .into_iter()
            .enumerate()
            .map(|(i, w)| {
                let pos = i as i32 * 3;
                TermVectorTerm {
                    term: w.into_bytes(),
                    freq: 2,
                    positions: Some(vec![pos, pos + 1]),
                    start_offsets: Some(vec![pos * 6, pos * 6 + 6]),
                    end_offsets: Some(vec![pos * 6 + 5, pos * 6 + 11]),
                    payloads: Some(vec![vec![(n % 251) as u8], vec![]]),
                }
            })
            .collect()
    };
    let tail: Vec<TermVectorTerm> = vec![TermVectorTerm {
        term: format!("id{n:07}").into_bytes(),
        freq: 1,
        positions: Some(vec![0]),
        start_offsets: None,
        end_offsets: None,
        payloads: None,
    }];
    TermVectorsDocument {
        fields: vec![
            TermVectorField {
                field_number: 0,
                has_positions: true,
                has_offsets: true,
                has_payloads: true,
                terms,
            },
            TermVectorField {
                field_number: 1,
                has_positions: true,
                has_offsets: false,
                has_payloads: false,
                terms: tail,
            },
        ],
    }
}

fn tv_write(docs: &[TermVectorsDocument], geometry: Option<usize>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut w = match geometry {
        Some(chunk) => TermVectorsWriter::with_geometry(&[7u8; 16], "", chunk, chunk),
        None => TermVectorsWriter::new(&[7u8; 16], ""),
    };
    for d in docs {
        w.add_document(d);
    }
    w.finish()
}

fn term_vectors_scenarios(docs_per_segment: usize, num_segments: usize, vocab: &[String]) {
    let docs: Vec<TermVectorsDocument> = (0..docs_per_segment).map(|n| tv_doc(n, vocab)).collect();

    // --- flush write throughput.
    let before = time(3, || tv_write(&docs, Some(SINGLE_CHUNK)));
    let after = time(3, || tv_write(&docs, None));
    let (single_tvd, single_tvx, single_tvm) = tv_write(&docs, Some(SINGLE_CHUNK));
    let (chunk_tvd, chunk_tvx, chunk_tvm) = tv_write(&docs, None);
    report(
        "flush write (1 segment)",
        docs_per_segment,
        single_tvd.len(),
        before,
        after,
    );
    println!(
        "  {:<34} before {:>8} B (1 chunk)          after {:>8} B ({} chunks)",
        ".tvd size",
        single_tvd.len(),
        chunk_tvd.len(),
        term_vectors::open(&chunk_tvd, &chunk_tvx, &chunk_tvm, &[7u8; 16], "")
            .unwrap()
            .num_chunks(),
    );

    // --- random-access read: one document's vectors.
    let single = term_vectors::open(&single_tvd, &single_tvx, &single_tvm, &[7u8; 16], "").unwrap();
    let chunked = term_vectors::open(&chunk_tvd, &chunk_tvx, &chunk_tvm, &[7u8; 16], "").unwrap();
    // A fixed pseudo-random sample, so both arms fetch the same documents.
    let sample: Vec<i32> = (0..200)
        .map(|i| ((i * 7919) % docs_per_segment) as i32)
        .collect();
    let before = time(3, || {
        for &d in &sample {
            std::hint::black_box(single.document(d).unwrap());
        }
    });
    let after = time(3, || {
        for &d in &sample {
            std::hint::black_box(chunked.document(d).unwrap());
        }
    });
    println!(
        "  {:<34} before {:>8.1} ms   after {:>8.1} ms   {:>5.2}x   ({} random single-doc reads)",
        "random-access document()",
        before * 1e3,
        after * 1e3,
        before / after,
        sample.len()
    );

    // --- merge.
    // "before": the pre-`c8` merge -- every source document materialised
    // through `reader.document(doc)` (which on a one-chunk segment decodes the
    // whole segment, every time) and the whole list re-encoded into one chunk.
    // "after": `Lucene90CompressingTermVectorsWriter.merge`'s BULK path,
    // `checkIntegrity` included, exactly as `write_merged_term_vectors` runs it
    // for a matching, deletion-free source.
    let singles: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..num_segments)
        .map(|_| tv_write(&docs, Some(SINGLE_CHUNK)))
        .collect();
    let chunkeds: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> =
        (0..num_segments).map(|_| tv_write(&docs, None)).collect();
    let single_readers: Vec<_> = singles
        .iter()
        .map(|(d, x, m)| term_vectors::open(d, x, m, &[7u8; 16], "").unwrap())
        .collect();
    let chunk_readers: Vec<_> = chunkeds
        .iter()
        .map(|(d, x, m)| term_vectors::open(d, x, m, &[7u8; 16], "").unwrap())
        .collect();

    // No warm-up rep for this one: it is quadratic in the segment's document
    // count (each `document(doc)` decodes the whole one-chunk segment), so a
    // second pass would double a multi-minute measurement for nothing.
    let before = {
        let t = Instant::now();
        let out = {
            let mut merged: Vec<TermVectorsDocument> = Vec::new();
            for r in &single_readers {
                for doc in 0..r.max_doc() {
                    merged.push(r.document(doc).unwrap().unwrap_or_default());
                }
            }
            let mut w =
                TermVectorsWriter::with_geometry(&[9u8; 16], "", SINGLE_CHUNK, SINGLE_CHUNK);
            for d in &merged {
                w.add_document(d);
            }
            w.finish()
        };
        let e = t.elapsed().as_secs_f64();
        std::hint::black_box(out);
        e
    };
    let after_bulk = time(3, || {
        let mut w = TermVectorsWriter::new(&[9u8; 16], "");
        for r in &chunk_readers {
            r.check_integrity().unwrap();
            w.copy_chunks(r, 0, r.max_doc()).unwrap();
        }
        w.finish()
    });
    // The path a source with deletions or renumbered fields still takes: decode
    // each document through the merge's `ChunkCursor` and re-encode it.
    let after_per_doc = time(3, || {
        let mut w = TermVectorsWriter::new(&[9u8; 16], "");
        for r in &chunk_readers {
            let mut cursor = term_vectors::ChunkCursor::new();
            for doc in 0..r.max_doc() {
                w.add_document(&cursor.document(r, doc).unwrap().unwrap_or_default());
            }
        }
        w.finish()
    });
    let total_docs = docs_per_segment * num_segments;
    let bytes: usize = singles.iter().map(|(d, ..)| d.len()).sum();
    report("merge, BULK", total_docs, bytes, before, after_bulk);
    report("merge, PER-DOC", total_docs, bytes, before, after_per_doc);

    // Sanity: the two merged segments hold the same documents.
    let (bd, bx, bm) = {
        let mut w = TermVectorsWriter::new(&[9u8; 16], "");
        for r in &chunk_readers {
            w.copy_chunks(r, 0, r.max_doc()).unwrap();
        }
        w.finish()
    };
    let merged = term_vectors::open(&bd, &bx, &bm, &[9u8; 16], "").unwrap();
    assert_eq!(merged.max_doc() as usize, total_docs);
    for &d in &sample {
        assert_eq!(
            merged.document(d).unwrap(),
            chunked.document(d).unwrap(),
            "bulk-merged doc {d} differs"
        );
    }
}

// ---------------------------------------------------------------------------
// Postings: `PostingsEnum` flags (b5's F6).
// ---------------------------------------------------------------------------

/// A docs-only walk over one long term, with and without
/// `PostingsEnum.FREQS`. Java's `Lucene104PostingsReader` calls
/// `PForUtil.skip` on the frequency block of every 256-document block when the
/// consumer did not ask for frequencies; before `c8` this port always unpacked
/// it.
fn postings_flags_scenario(doc_freq: i32, max_freq: usize, label: &str) {
    let tmp = tempdir("flags");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("id", 0, false), field("body", 1, true)];
    let mut writer = IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    for n in 0..doc_freq as usize {
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String(format!("d{n}")),
                    },
                    StoredField {
                        field_number: 1,
                        // "hot" is in every document, so its postings span
                        // `doc_freq / 256` full blocks plus a tail; the filler
                        // gives the block encoder realistic freqs.
                        // "hot" is in every document; how *often* is what
                        // sets the frequency block's bit width, and therefore
                        // how much work `PForUtil.skip` saves over a full
                        // 256-value unpack.
                        value: FieldValue::String(format!(
                            "{}{}",
                            "hot ".repeat(1 + n % max_freq),
                            "pad ".repeat(1 + n % 4)
                        )),
                    },
                ],
            })
            .unwrap();
    }
    writer.commit().unwrap();

    let suffix = per_field_codec_suffix(POSTINGS_FORMAT_NAME);
    let sci = writer.segment_infos().segments[0].clone();
    let seg =
        lucene_index::index_writer::per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME);
    let tim = dir.open(&format!("{seg}.tim")).unwrap().to_vec();
    let tip = dir.open(&format!("{seg}.tip")).unwrap().to_vec();
    let tmd = dir.open(&format!("{seg}.tmd")).unwrap().to_vec();
    let doc_bytes = dir.open(&format!("{seg}.doc")).unwrap().to_vec();
    let infos = FieldInfos {
        fields: vec![field("body", 1, true)],
    };
    let bt = blocktree::open(&tim, &tip, &tmd, &infos, &sci.segment_id, &suffix, doc_freq).unwrap();
    let doc_in = DocInput::open(&doc_bytes, &sci.segment_id, &suffix).unwrap();
    let terms = bt.field("body").unwrap();
    let df = terms.seek_exact(b"hot").unwrap().doc_freq;

    let walk = |flags: PostingsFlags| {
        let mut cursor = terms
            .lazy_postings_with_flags(b"hot", &doc_in, flags)
            .unwrap()
            .unwrap();
        let mut sum = 0i64;
        loop {
            let d = cursor.next_doc().unwrap();
            if d == lucene_codecs::postings::NO_MORE_DOCS {
                break;
            }
            sum += d as i64;
        }
        sum
    };
    let before_sum = walk(PostingsFlags::Freqs);
    let after_sum = walk(PostingsFlags::DocsOnly);
    assert_eq!(
        before_sum, after_sum,
        "docs-only must walk the same doc ids"
    );

    let before = time(20, || walk(PostingsFlags::Freqs));
    let after = time(20, || walk(PostingsFlags::DocsOnly));
    println!(
        "  {:<34} before {:>8.1} us   after {:>8.1} us   {:>5.2}x   (docFreq={df}, lazy cursor)",
        format!("docs-only walk, {label}"),
        before * 1e6,
        after * 1e6,
        before / after,
    );

    let before = time(20, || {
        terms
            .postings_with_flags(b"hot", Some(&doc_in), PostingsFlags::Freqs)
            .unwrap()
            .unwrap()
    });
    let after = time(20, || {
        terms
            .postings_with_flags(b"hot", Some(&doc_in), PostingsFlags::DocsOnly)
            .unwrap()
            .unwrap()
    });
    println!(
        "  {:<34} before {:>8.1} us   after {:>8.1} us   {:>5.2}x   (docFreq={df}, eager)",
        format!("docs-only read_postings, {label}"),
        before * 1e6,
        after * 1e6,
        before / after,
    );

    std::fs::remove_dir_all(&tmp).ok();
}

fn main() {
    let mut args = std::env::args().skip(1);
    let docs_per_segment: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(20_000);
    let num_segments: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(4);

    let vocab: Vec<String> = (0..2_000)
        .map(|i| format!("{}{i:04}", (b'a' + (i % 26) as u8) as char))
        .collect();

    println!(
        "merge-bench: {num_segments} segments x {docs_per_segment} documents\n\
         (each \"before\" figure is the pre-c4 algorithm, re-run here over the same inputs)\n"
    );
    println!("stored fields:");
    stored_fields_scenarios(docs_per_segment, num_segments, &vocab);
    println!("\npostings:");
    postings_scenario(docs_per_segment.min(20_000), num_segments, &vocab);
    println!("\npoints:");
    points_scenario(1_000_000);
    println!("\nterm vectors (before = the pre-c8 one-chunk-per-segment writer):");
    term_vectors_scenarios(docs_per_segment, num_segments, &vocab);
    println!("\nindex sort (before = the same merge, unsorted; after = sort-preserving):");
    sorted_merge_scenario(docs_per_segment, num_segments, &vocab);
    println!("\npostings flags (before = always decoding frequencies):");
    // The saving scales with the frequency block's bit width, so measure both
    // a near-degenerate corpus (freqs 1-3, ~2 bits) and a realistic one
    // (freqs 1-48, ~6 bits).
    postings_flags_scenario(200_000, 3, "freq 1-3");
    postings_flags_scenario(200_000, 48, "freq 1-48");
}
