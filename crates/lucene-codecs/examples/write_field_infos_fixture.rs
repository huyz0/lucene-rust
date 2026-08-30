//! Writes a `field_infos::write`-produced `.fnm` file plus a manifest to the
//! directory given as the first CLI argument.
//!
//! This is the reverse of this repo's usual differential-testing direction
//! (Java writes, Rust reads): here Rust writes, and
//! `fixtures/src/VerifyFieldInfos.java` reads the result back through real
//! Lucene's own `Lucene94FieldInfosFormat.read`, using a hand-built
//! `SegmentInfo` -- this keeps the write-path slice scoped to exactly the
//! field-infos format itself, the same way `write_stored_fields_fixture.rs`
//! doesn't also require a `.si` writer.
//!
//! Run: `cargo run -p lucene-codecs --example write_field_infos_fixture -- <dir>`
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    self, DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_store::{DataOutput, Directory, FsDirectory};
use std::io::Write;

const SEGMENT_ID: [u8; 16] = *b"rustwrittenfnm01";

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_field_infos_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).unwrap();

    let fields = vec![
        FieldInfo {
            name: "id".to_string(),
            number: 0,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::Docs,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        },
        FieldInfo {
            name: "with_tv".to_string(),
            number: 1,
            store_term_vectors: true,
            omit_norms: false,
            store_payloads: true,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![("attr_key".to_string(), "attr_value".to_string())],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        },
        FieldInfo {
            name: "num_dv".to_string(),
            number: 2,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::Numeric,
            doc_values_skip_index_type: DocValuesSkipIndexType::Range,
            doc_values_gen: 7,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        },
        FieldInfo {
            name: "sorted_dv".to_string(),
            number: 3,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::Sorted,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        },
        FieldInfo {
            name: "point_field".to_string(),
            number: 4,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 1,
            point_index_dimension_count: 1,
            point_num_bytes: 8,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        },
        FieldInfo {
            name: "vector_field".to_string(),
            number: 5,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 3,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Cosine,
        },
        FieldInfo {
            name: "__soft_deletes".to_string(),
            number: 6,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: true,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::Numeric,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        },
        // c40: the combination Java's `FieldInfo` constructor makes
        // unrepresentable and its *reader* silently clears. Recorded here with
        // the flags already coerced off, because that is what real Lucene
        // reports for the bytes below -- `force_non_indexed` patches this
        // field's IndexOptions byte to NONE *after* `write` has emitted the
        // three indexed-only bits, which no writer path can otherwise produce
        // (`field_infos::write` coerces them away, exactly as Java's writer
        // never sees them).
        FieldInfo {
            name: "noindex_flags".to_string(),
            number: 8,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        },
        FieldInfo {
            name: "__parent".to_string(),
            number: 7,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: true,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::Numeric,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        },
    ];

    // Write the wire bytes from a copy in which `noindex_flags` *is* indexed
    // and carries all three flags, so `write` emits bits 0x7 for it, then
    // patch its IndexOptions byte back to NONE. The result is a `.fnm` that no
    // writer -- this port's or Lucene's -- can produce, but that Lucene's
    // reader accepts, clearing the three bits in `FieldInfo`'s constructor
    // before `checkConsistency()` looks. `fields` above already records the
    // coerced expectation, so the manifest is what both engines must report.
    let mut wire = fields.clone();
    for f in wire.iter_mut() {
        if f.name == "noindex_flags" {
            f.index_options = IndexOptions::DocsAndFreqsAndPositions;
            f.store_term_vectors = true;
            f.omit_norms = true;
            f.store_payloads = true;
        }
    }
    let mut bytes = field_infos::write(&wire, &SEGMENT_ID, "");
    force_non_indexed(&mut bytes, "noindex_flags");

    let dir = FsDirectory::open(&out_dir);
    let mut out = dir.create_output("_0.fnm").unwrap();
    out.write_bytes(&bytes);
    out.close().unwrap();
    dir.sync(&["_0.fnm".to_string()]).unwrap();

    let mut manifest = std::fs::File::create(format!("{out_dir}/manifest.properties")).unwrap();
    writeln!(manifest, "id_hex={}", hex(&SEGMENT_ID)).unwrap();
    writeln!(manifest, "field_count={}", fields.len()).unwrap();
    let field_order: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    writeln!(manifest, "field_order={}", field_order.join(",")).unwrap();
    for f in &fields {
        let prefix = format!("field.{}.", f.name);
        writeln!(manifest, "{prefix}number={}", f.number).unwrap();
        writeln!(manifest, "{prefix}index_options={:?}", f.index_options).unwrap();
        writeln!(manifest, "{prefix}doc_values_type={:?}", f.doc_values_type).unwrap();
        writeln!(
            manifest,
            "{prefix}doc_values_skip_index_type={:?}",
            f.doc_values_skip_index_type
        )
        .unwrap();
        writeln!(manifest, "{prefix}doc_values_gen={}", f.doc_values_gen).unwrap();
        writeln!(
            manifest,
            "{prefix}has_term_vectors={}",
            f.store_term_vectors
        )
        .unwrap();
        writeln!(manifest, "{prefix}omit_norms={}", f.omit_norms).unwrap();
        writeln!(manifest, "{prefix}store_payloads={}", f.store_payloads).unwrap();
        writeln!(manifest, "{prefix}is_soft_deletes={}", f.soft_deletes_field).unwrap();
        writeln!(manifest, "{prefix}is_parent_field={}", f.parent_field).unwrap();
        writeln!(
            manifest,
            "{prefix}point_dimension_count={}",
            f.point_dimension_count
        )
        .unwrap();
        writeln!(
            manifest,
            "{prefix}point_index_dimension_count={}",
            f.point_index_dimension_count
        )
        .unwrap();
        writeln!(manifest, "{prefix}point_num_bytes={}", f.point_num_bytes).unwrap();
        writeln!(manifest, "{prefix}vector_dimension={}", f.vector_dimension).unwrap();
        writeln!(
            manifest,
            "{prefix}vector_similarity={:?}",
            f.vector_similarity_function
        )
        .unwrap();
        let attrs: Vec<String> = f
            .attributes
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        writeln!(manifest, "{prefix}attributes={}", attrs.join(";")).unwrap();
    }

    println!("wrote field-infos fixture to {out_dir}");
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Patches `field`'s `IndexOptions` byte in an already-written `.fnm` to
/// `NONE`, leaving its FieldBits untouched.
///
/// The per-field record starts `FieldName(String) FieldNumber(vint)
/// FieldBits(byte) IndexOptions(byte)`, so finding the encoded name locates
/// the two bytes that follow the number. Only a hand-patched file can carry
/// this combination: real Lucene's writer takes its bits from a `FieldInfo`
/// whose constructor already cleared them, and so does this port's `write`.
fn force_non_indexed(bytes: &mut [u8], field: &str) {
    // `write_string` is a vint length then the UTF-8 bytes; every field name
    // here is far under 128 bytes, so the length is a single byte.
    assert!(field.len() < 128, "single-byte vint length assumed");
    let mut needle = vec![field.len() as u8];
    needle.extend_from_slice(field.as_bytes());
    let at = bytes
        .windows(needle.len())
        .position(|w| w == needle.as_slice())
        .expect("field name not found in the written .fnm");
    let mut cursor = at + needle.len();
    // FieldNumber, a vint: skip its continuation bytes.
    while bytes[cursor] & 0x80 != 0 {
        cursor += 1;
    }
    cursor += 1;
    // FieldBits stays as written; IndexOptions becomes NONE (0).
    assert_ne!(bytes[cursor], 0, "expected non-zero FieldBits to preserve");
    bytes[cursor + 1] = 0;

    // The footer's CRC covers every byte before it, so re-stamp it -- a
    // hand-patched file has to stay a *valid* file, or the only thing the
    // verifier proves is that Lucene checks checksums.
    let checksum_at = bytes.len() - 8;
    let checksum = crc32fast::hash(&bytes[..checksum_at]) as u64;
    bytes[checksum_at..].copy_from_slice(&checksum.to_be_bytes());
}
