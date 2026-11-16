//! `IndexFiles`: the demo's indexer. Every file under a directory becomes a
//! document -- its path (a `KeywordField`, stored), its modification time
//! (a `LongField`) and its contents (a `TextField`, through the standard
//! analyzer) -- and, given a word-vector dictionary, its contents' embedding
//! (a `KnnFloatVectorField` under `DOT_PRODUCT`).
//!
//! ```text
//! lucene-tools IndexFiles [-index INDEX_PATH] [-docs DOCS_PATH] [-update] [-knn_dict DICT_PATH]
//! ```

use std::path::{Path, PathBuf};
use std::time::{Instant, UNIX_EPOCH};

use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    Document, KeywordField, KnnFloatVectorField, LongField, Store, TextField,
    VectorSimilarityFunction,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::MergePolicyConfig;
use lucene_index::segment_info::LuceneVersion;
use lucene_index::segment_infos;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::directory::{Directory, FsDirectory};

use super::knn::{DemoEmbeddings, FileAlreadyExists, KnnVectorDict};
use crate::cli::{arg, exception, Error, Io, ToolResult};

/// `IndexFiles.KNN_DICT`: the dictionary's name in the index directory.
pub const KNN_DICT: &str = "knn-dict";

/// The codec and version the writer records, as Lucene 10.5.0's defaults.
pub(super) const CODEC: &str = "Lucene104";
pub(super) const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// `main`'s usage text, printed after `Usage: ` to `System.err`.
pub const USAGE: &str = "java org.apache.lucene.demo.IndexFiles [-index INDEX_PATH] [-docs DOCS_PATH] [-update] [-knn_dict DICT_PATH]\n\nThis indexes the documents in DOCS_PATH, creating a Lucene index in INDEX_PATH that can be searched with SearchFiles\nIF DICT_PATH contains a KnnVector dictionary, the index will also support KnnVector search";

/// `IndexFiles.main(args)`. `-create` (the default) replaces any index at
/// `INDEX_PATH` (`index` by default), `-update` adds to it, replacing a
/// document with the same path. No `-docs` prints the usage and exits 1, a
/// missing documents path prints so and exits 1. Prints `adding <path>` (or
/// `updating <path>`) per file, then `Indexed N documents in T ms`.
///
/// # Errors
/// Java's `IllegalArgumentException` for an unknown parameter and
/// `RuntimeException` for a dictionary too small for more than 100
/// documents; a failure to read or write is printed (` caught a ...`), not
/// thrown, as Java catches its `IOException`.
pub fn main(args: &[String], io: &mut Io<'_>) -> ToolResult {
    let mut index_path = "index".to_string();
    let mut docs_path = None;
    let mut vector_dict_source = None;
    let mut create = true;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-index" => {
                i = i.saturating_add(1);
                index_path = arg(args, i)?.to_string();
            }
            "-docs" => {
                i = i.saturating_add(1);
                docs_path = Some(arg(args, i)?.to_string());
            }
            "-knn_dict" => {
                i = i.saturating_add(1);
                vector_dict_source = Some(arg(args, i)?.to_string());
            }
            "-update" => create = false,
            "-create" => create = true,
            other => {
                return Err(exception(format!(
                    "java.lang.IllegalArgumentException: unknown parameter {other}"
                )))
            }
        }
        i = i.saturating_add(1);
    }
    let Some(docs_path) = docs_path else {
        writeln!(io.err, "Usage: {USAGE}")?;
        return Ok(1);
    };
    let doc_dir = java_path(&docs_path);
    if std::fs::metadata(&doc_dir).is_err() {
        let absolute = std::path::absolute(&doc_dir)?;
        let line = format!(
            "Document directory '{}' does not exist or is not readable, please check the path",
            absolute.display()
        );
        writeln!(io.out, "{line}")?;
        return Ok(1);
    }

    let start = Instant::now();
    writeln!(io.out, "Indexing to directory '{index_path}'...")?;
    match index(
        &index_path,
        &doc_dir,
        create,
        vector_dict_source.as_deref(),
        io,
    ) {
        Ok(num_docs) => {
            let ms = start.elapsed().as_millis();
            writeln!(io.out, "Indexed {num_docs} documents in {ms} ms")?;
            Ok(0)
        }
        Err(Failure::Io(e)) => {
            let (class, message) = match e.downcast_ref::<FileAlreadyExists>() {
                // `FSDirectory` resolves the name against its real path
                Some(f) => (
                    "java.nio.file.FileAlreadyExistsException",
                    real_path(&index_path).join(&f.0).display().to_string(),
                ),
                None => ("java.io.IOException", e.to_string()),
            };
            let line = format!(" caught a class {class}\n with message: {message}");
            writeln!(io.out, "{line}")?;
            Ok(0)
        }
        Err(Failure::Thrown(e)) => Err(e),
    }
}

/// How the indexing inside `main`'s `try` ended badly: an `IOException`
/// Java catches, or an exception it lets through.
#[derive(Debug)]
enum Failure {
    Io(Error),
    Thrown(Error),
}

/// An error Java's `catch (IOException e)` takes.
fn caught(e: impl Into<Error>) -> Failure {
    Failure::Io(e.into())
}

/// The dictionary's error: an I/O error is caught, its own checks (Java's
/// runtime exceptions) are not.
fn io_or_thrown(e: Error) -> Failure {
    if e.is::<std::io::Error>() || e.is::<lucene_store::Error>() || e.is::<FileAlreadyExists>() {
        Failure::Io(e)
    } else {
        Failure::Thrown(e)
    }
}

/// The body of `main`'s `try`: builds the dictionary when given one, adds
/// every document, and returns the index's `numDocs`.
fn index(
    index_path: &str,
    doc_dir: &Path,
    create: bool,
    vector_dict_source: Option<&str>,
    io: &mut Io<'_>,
) -> Result<i32, Failure> {
    std::fs::create_dir_all(index_path).map_err(caught)?;
    let dir = FsDirectory::open(index_path);
    let mut vector_dict = None;
    if let Some(source) = vector_dict_source {
        KnnVectorDict::build(&java_path(source), &dir, KNN_DICT).map_err(io_or_thrown)?;
        vector_dict = Some(KnnVectorDict::open(&dir, KNN_DICT).map_err(io_or_thrown)?);
    }
    let vector_dict_size = vector_dict
        .as_ref()
        .map_or(0, KnnVectorDict::ram_bytes_used);
    let embeddings = vector_dict.as_ref().map(DemoEmbeddings::new);

    let existing = segment_infos::read_latest(&dir).is_ok();
    let mut writer = IndexWriter::open(&dir, Vec::new(), CODEC, VERSION).map_err(caught)?;
    // `IndexWriterConfig`'s default `TieredMergePolicy`: Java merges the
    // demo's one-document segments as they pile up, which changes the doc
    // ids and the statistics a search scores with.
    writer.set_merge_policy(Some(MergePolicyConfig::default()));
    // `IndexWriterConfig`'s default compound files: a flushed segment is a
    // `.cfs` whose sub-files are 64-byte aligned. The policy sorts segments by
    // byte size, ties by name, so the padding decides which one-document
    // segments tie and so the order a merge takes their documents in.
    writer.set_use_compound_file(true);
    if create && existing {
        // `OpenMode.CREATE` over an index: the new commit holds none of it.
        writer.delete_all().map_err(caught)?;
    }
    let mut indexer = Indexer {
        writer: &mut writer,
        embeddings: embeddings.as_ref(),
        create,
        io,
    };
    indexer.index_docs(doc_dir)?;
    writer.close().map_err(caught)?;

    let reader = DirectoryReader::open(&dir as &dyn Directory).map_err(caught)?;
    let num_docs = reader.num_docs();
    if vector_dict_source.is_some() && num_docs > 100 && vector_dict_size < 1_000_000 {
        return Err(Failure::Thrown(exception(
            "java.lang.RuntimeException: Are you (ab)using the toy vector dictionary? See the \
             package javadocs to understand why you got this exception.",
        )));
    }
    Ok(num_docs)
}

/// `Path.toRealPath()`, or the path as given when it cannot be resolved.
fn real_path(path: &str) -> PathBuf {
    match std::fs::canonicalize(path) {
        Ok(p) => {
            let s = p.to_string_lossy();
            // Windows' verbatim prefix, which Java's real path does not carry
            PathBuf::from(s.strip_prefix(r"\\?\").unwrap_or(&s).to_string())
        }
        Err(_) => PathBuf::from(path),
    }
}

/// `Paths.get(s)`: the path with repeated and trailing separators dropped.
fn java_path(s: &str) -> PathBuf {
    Path::new(s).components().collect()
}

/// The writer and what each document needs.
struct Indexer<'w, 'd, 'e, 'o, 'i> {
    writer: &'w mut IndexWriter<'d>,
    embeddings: Option<&'e DemoEmbeddings<'e>>,
    create: bool,
    io: &'o mut Io<'i>,
}

impl Indexer<'_, '_, '_, '_, '_> {
    /// `indexDocs(writer, path)`: every file under `path`, depth first in
    /// directory order (`Files.walkFileTree`), or `path` itself.
    fn index_docs(&mut self, path: &Path) -> Result<(), Failure> {
        if path.is_dir() {
            self.walk(path)
        } else {
            self.index_doc(path)
        }
    }

    fn walk(&mut self, dir: &Path) -> Result<(), Failure> {
        for entry in std::fs::read_dir(dir).map_err(caught)? {
            let entry = entry.map_err(caught)?;
            let path = entry.path();
            // `walkFileTree` without `FOLLOW_LINKS`: a link, even to a
            // directory, is visited as a file, never descended into
            if entry.file_type().map_err(caught)?.is_dir() {
                self.walk(&path)?;
            } else {
                match self.index_doc(&path) {
                    // `visitFile` prints the `IOException` and goes on.
                    Err(Failure::Io(e)) => {
                        let line = format!("java.io.IOException: {e}");
                        writeln!(self.io.err, "{line}").map_err(caught)?;
                    }
                    other => other?,
                }
            }
        }
        Ok(())
    }

    /// `indexDoc(writer, file, lastModified)`: a failure to read or write is
    /// an `IOException`; the vector field's `IllegalArgumentException` (an
    /// empty or non-finite vector, which a dictionary `build` accepted cannot
    /// produce) is thrown.
    fn index_doc(&mut self, file: &Path) -> Result<(), Failure> {
        let bytes = std::fs::read(file).map_err(caught)?;
        let last_modified = std::fs::metadata(file)
            .and_then(|m| m.modified())
            .map_err(caught)?
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));
        // `InputStreamReader(stream, UTF_8)` replaces malformed input.
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let path = file.to_string_lossy().into_owned();
        let mut doc = Document::new();
        doc.add(KeywordField::new("path", path.clone(), Store::Yes));
        doc.add(LongField::new("modified", last_modified, Store::No));
        let vector = match self.embeddings {
            Some(embeddings) => Some(embeddings.compute_embedding(&text).map_err(io_or_thrown)?),
            None => None,
        };
        doc.add(TextField::from_reader("contents", text));
        if let Some(vector) = vector {
            let field = KnnFloatVectorField::new(
                "contents-vector",
                vector,
                VectorSimilarityFunction::DotProduct,
            );
            doc.add(field.map_err(|e| {
                Failure::Thrown(exception(format!(
                    "java.lang.IllegalArgumentException: {e}"
                )))
            })?);
        }
        if self.create {
            writeln!(self.io.out, "adding {path}").map_err(caught)?;
            self.writer.add_fields_document(&doc).map_err(caught)?;
        } else {
            writeln!(self.io.out, "updating {path}").map_err(caught)?;
            self.writer
                .update_fields_documents(Term::new("path", path.into_bytes()), &[doc])
                .map_err(caught)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
