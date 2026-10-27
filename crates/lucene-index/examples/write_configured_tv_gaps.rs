//! Export configured term-vector schema gaps before any Rust merge.
use lucene_codecs::{
    field_infos::{FieldInfo, IndexOptions},
    stored_fields::{Document, FieldValue, StoredField},
};
use lucene_index::{index_writer::IndexWriter, segment_info::LuceneVersion};
use lucene_store::FsDirectory;
use std::{path::Path, sync::Arc};
#[derive(Debug, Default)]
struct Installed {
    factory: lucene_codecs::term_vectors_writer::NativeTermVectorsWriterFactory,
}
impl lucene_codecs::codec::Lucene104Codec for Installed {
    fn term_vectors_writer_factory(
        &self,
    ) -> Option<&dyn lucene_codecs::term_vectors_writer::TermVectorsWriterFactory> {
        Some(&self.factory)
    }
}
fn document(id: &str, body: Option<FieldValue>) -> Document {
    let mut fields = vec![StoredField {
        field_number: 0,
        value: FieldValue::String(id.into()),
    }];
    if let Some(value) = body {
        fields.push(StoredField {
            field_number: 1,
            value,
        });
    }
    Document { fields }
}
fn snapshot(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(destination)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            std::fs::copy(entry.path(), destination.join(entry.file_name()))?;
        }
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::PathBuf::from(std::env::args().nth(1).expect("output directory"));
    std::fs::create_dir_all(&root)?;
    for installed in [false, true] {
        for reopen in [false, true] {
            for kind in ["missing", "non_string", "empty"] {
                let case = root.join(format!("{installed}-{reopen}-{kind}"));
                let live = case.join("live");
                std::fs::create_dir_all(&live)?;
                let directory = FsDirectory::open(&live);
                let fields = vec![
                    FieldInfo::new("id", 0).with_omit_norms(true),
                    FieldInfo::new("body", 1)
                        .with_index_options(IndexOptions::DocsAndFreqs)
                        .with_omit_norms(true)
                        .with_store_term_vectors(true),
                ];
                let open = || -> Result<IndexWriter<'_>, lucene_index::index_writer::Error> {
                    let mut writer = IndexWriter::open(
                        &directory,
                        fields.clone(),
                        "Lucene104",
                        LuceneVersion {
                            major: 10,
                            minor: 5,
                            bugfix: 0,
                        },
                    )?;
                    writer.set_postings_field(Some("body"))?;
                    writer.set_term_vector_field(Some("body"))?;
                    if installed {
                        writer.set_codec(Some(Arc::new(Installed::default())))?;
                    }
                    Ok(writer)
                };
                let mut writer = open()?;
                let body = match kind {
                    "missing" => None,
                    "non_string" => Some(FieldValue::Int(42)),
                    _ => Some(FieldValue::String(String::new())),
                };
                writer.add_document(document("a", body))?;
                writer.commit()?;
                snapshot(&live, &case.join("first"))?;
                if reopen {
                    drop(writer);
                    writer = open()?;
                }
                writer.add_document(document("b", Some(FieldValue::String("x x".into()))))?;
                let committed = writer.commit()?;
                assert_eq!(committed.segments.len(), 2, "must test unmerged schema");
                snapshot(&live, &case.join("two"))?;
                writer.rollback();
            }
        }
    }
    Ok(())
}
