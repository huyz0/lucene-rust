//! Writes a `compound_format::write`-produced `_0.cfs`/`_0.cfe` pair to the
//! directory given as the first CLI argument, packing sub-files produced by
//! two already-committed write-side pieces of this port: `field_infos::write`
//! (a `.fnm`) and `stored_fields::write_best_speed` (a `.fdt`/`.fdx`/`.fdm`
//! triple) -- four distinct sub-files total, so the entries table's
//! offset/length bookkeeping and the ascending-size packing order both get
//! genuinely exercised, not just a single-file passthrough.
//!
//! Reverse-direction differential test (Rust writes, Java reads):
//! `fixtures/src/VerifyCompoundFormat.java` opens the pair through real
//! Lucene's `Lucene90CompoundFormat.getCompoundReader` with a hand-built
//! `SegmentInfo`, confirms the sub-file list/lengths, then re-decodes the
//! `.fnm` through real `Lucene94FieldInfosFormat` and the
//! `.fdt`/`.fdx`/`.fdm` triple through real `Lucene90StoredFieldsFormat` --
//! both *through* the compound reader, not the raw sub-file bytes directly --
//! confirming the packed byte offsets are correct, not just that the entries
//! table "looks right".
//!
//! Run: `cargo run -p lucene-codecs --example write_compound_format_fixture -- <dir>`

use lucene_codecs::compound_format;
use lucene_codecs::field_infos::{
    self, DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{self, Document, FieldValue, StoredField};
use lucene_store::{DataOutput, Directory, FsDirectory};
use std::io::Write;

const SEGMENT_ID: [u8; 16] = *b"rustwrittencfs01";
const SEGMENT_NAME: &str = "_0";

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_compound_format_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).unwrap();

    let fields = vec![
        plain_field("field0", 0),
        plain_field("field1", 1),
        plain_field("field2", 2),
    ];
    let fnm = field_infos::write(&fields, &SEGMENT_ID, "");

    let docs = vec![
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("hello compound world".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Int(-7),
                },
                StoredField {
                    field_number: 2,
                    value: FieldValue::Long(9_000_000_001),
                },
            ],
        },
        Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String("second doc".to_string()),
            }],
        },
    ];
    let (fdt, fdx, fdm) = stored_fields::write_best_speed(&docs, &SEGMENT_ID, "");

    // Record each sub-file's standalone length up front (used for the
    // manifest cross-check); packing must preserve these exactly.
    let sub_files: Vec<(String, Vec<u8>)> = vec![
        (".fnm".to_string(), fnm),
        (".fdt".to_string(), fdt),
        (".fdx".to_string(), fdx),
        (".fdm".to_string(), fdm),
    ];
    let lengths: Vec<(String, usize)> = sub_files
        .iter()
        .map(|(name, bytes)| (name.clone(), bytes.len()))
        .collect();

    let (cfs, cfe) = compound_format::write(&SEGMENT_ID, &sub_files).expect("compound write");

    let dir = FsDirectory::open(&out_dir);
    let cfs_file_name = format!("{SEGMENT_NAME}.cfs");
    let cfe_file_name = format!("{SEGMENT_NAME}.cfe");
    for (file_name, bytes) in [(&cfs_file_name, &cfs), (&cfe_file_name, &cfe)] {
        let mut out = dir.create_output(file_name).unwrap();
        out.write_bytes(bytes);
        out.close().unwrap();
    }
    dir.sync(&[cfs_file_name.clone(), cfe_file_name.clone()])
        .unwrap();

    let mut manifest = std::fs::File::create(format!("{out_dir}/manifest.properties")).unwrap();
    writeln!(manifest, "id_hex={}", hex(&SEGMENT_ID)).unwrap();
    writeln!(manifest, "segment_name={SEGMENT_NAME}").unwrap();
    writeln!(manifest, "cfs_file_name={cfs_file_name}").unwrap();
    writeln!(manifest, "cfe_file_name={cfe_file_name}").unwrap();
    let rendered_sub_files: Vec<String> = lengths
        .iter()
        .map(|(name, len)| format!("{name}:{len}"))
        .collect();
    writeln!(manifest, "sub_files={}", rendered_sub_files.join(",")).unwrap();
    writeln!(manifest, "num_fields={}", fields.len()).unwrap();
    writeln!(manifest, "max_doc={}", docs.len()).unwrap();
    for (i, doc) in docs.iter().enumerate() {
        let rendered: Vec<String> = doc
            .fields
            .iter()
            .map(|f| render_field(f.field_number, &f.value))
            .collect();
        writeln!(manifest, "doc.{i}.fields={}", rendered.join(";")).unwrap();
    }

    println!("wrote compound format fixture to {out_dir}");
}

fn plain_field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
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
    }
}

fn render_field(number: i32, value: &FieldValue) -> String {
    match value {
        FieldValue::String(s) => format!("{number}:string:{s}"),
        FieldValue::Binary(b) => format!("{number}:binary:{}", hex(b)),
        FieldValue::Int(v) => format!("{number}:int:{v}"),
        FieldValue::Long(v) => format!("{number}:long:{v}"),
        FieldValue::Float(v) => format!("{number}:float:{v}"),
        FieldValue::Double(v) => format!("{number}:double:{v}"),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
