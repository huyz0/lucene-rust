//! `DocHelper`: a fixed document of fourteen fields covering stored,
//! unstored, unindexed, norm-less, frequency-less, term-vector, UTF-8 and
//! large values, the field-name sets they fall into, and `writeDoc`.
//!
//! Java's static `Field`s and maps become functions building them afresh:
//! a `Field` here is a value, not a shared object. `lazyFieldBinary` is
//! `new StoredField(name, bytes)`, which has no type of its own here: a
//! [`Field`] from bytes on a stored-only [`FieldType`].

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, LazyLock};

use lucene_analysis::Analyzer;
use lucene_codecs::field_infos::IndexOptions;
use lucene_index::document::{Document, Field, FieldType, IndexableField, StringField, TextField};
use lucene_index::index_writer::{IndexWriter, Result};
use lucene_index::segment_info::LuceneVersion;
use lucene_index::segment_infos::SegmentCommitInfo;
use lucene_index::similarity::NormSimilarity;
use lucene_store::Directory;
use lucene_util::java_random::JavaRandom;

use crate::analysis::mock_analyzer::MockAnalyzer;
use crate::analysis::mock_tokenizer;

pub const FIELD_1_TEXT: &str = "field one text";
pub const TEXT_FIELD_1_KEY: &str = "textField1";
pub const FIELD_2_TEXT: &str = "field field field two text";
/// Fields will be lexicographically sorted. So, the order is: field, text, two.
pub const FIELD_2_FREQS: [i32; 3] = [3, 1, 1];
pub const TEXT_FIELD_2_KEY: &str = "textField2";
pub const FIELD_3_TEXT: &str = "aaaNoNorms aaaNoNorms bbbNoNorms";
pub const TEXT_FIELD_3_KEY: &str = "textField3";
pub const KEYWORD_TEXT: &str = "Keyword";
pub const KEYWORD_FIELD_KEY: &str = "keyField";
pub const NO_NORMS_TEXT: &str = "omitNormsText";
pub const NO_NORMS_KEY: &str = "omitNorms";
pub const NO_TF_TEXT: &str = "analyzed with no tf and positions";
pub const NO_TF_KEY: &str = "omitTermFreqAndPositions";
pub const UNINDEXED_FIELD_TEXT: &str = "unindexed field text";
pub const UNINDEXED_FIELD_KEY: &str = "unIndField";
pub const UNSTORED_1_FIELD_TEXT: &str = "unstored field text";
pub const UNSTORED_FIELD_1_KEY: &str = "unStoredField1";
pub const UNSTORED_2_FIELD_TEXT: &str = "unstored field text";
pub const UNSTORED_FIELD_2_KEY: &str = "unStoredField2";
pub const LAZY_FIELD_BINARY_KEY: &str = "lazyFieldBinary";
pub const LAZY_FIELD_BINARY_BYTES: &[u8] = b"These are some binary field bytes";
pub const LAZY_FIELD_KEY: &str = "lazyField";
pub const LAZY_FIELD_TEXT: &str = "These are some field bytes";
pub const LARGE_LAZY_FIELD_KEY: &str = "largeLazyField";
/// From Issue 509.
pub const FIELD_UTF1_TEXT: &str = "field one \u{4e00}text";
pub const TEXT_FIELD_UTF1_KEY: &str = "textField1Utf8";
pub const FIELD_UTF2_TEXT: &str = "field field field \u{4e00}two text";
/// Fields will be lexicographically sorted. So, the order is: field, text, two.
pub const FIELD_UTF2_FREQS: [i32; 3] = [3, 1, 1];
pub const TEXT_FIELD_UTF2_KEY: &str = "textField2Utf8";

/// `LARGE_LAZY_FIELD_TEXT`: a sentence repeated 10,000 times.
pub static LARGE_LAZY_FIELD_TEXT: LazyLock<String> =
    LazyLock::new(|| "Lazily loading lengths of language in lieu of laughing ".repeat(10000));

fn set(r: lucene_index::document::Result<()>) {
    r.expect("an unfrozen FieldType");
}

/// `customType`: `TextField.TYPE_STORED`.
pub fn custom_type() -> FieldType {
    FieldType::copy_of(&TextField::type_stored())
}

/// `TEXT_TYPE_STORED_WITH_TVS`: `TextField.TYPE_STORED` with term vectors,
/// their positions and offsets, frozen.
pub fn text_type_stored_with_tvs() -> FieldType {
    with_tvs(&TextField::type_stored())
}

/// `STRING_TYPE_STORED_WITH_TVS`: `StringField.TYPE_STORED` with term
/// vectors, their positions and offsets, frozen.
pub fn string_type_stored_with_tvs() -> FieldType {
    with_tvs(&StringField::type_stored())
}

fn with_tvs(base: &FieldType) -> FieldType {
    let mut t = FieldType::copy_of(base);
    set(t.set_store_term_vectors(true));
    set(t.set_store_term_vector_positions(true));
    set(t.set_store_term_vector_offsets(true));
    t.frozen()
}

/// `customType3`: stored text without norms.
pub fn custom_type3() -> FieldType {
    let mut t = custom_type();
    set(t.set_omit_norms(true));
    t
}

/// `customType5`: stored, untokenized, without norms.
pub fn custom_type5() -> FieldType {
    let mut t = custom_type3();
    set(t.set_tokenized(false));
    t
}

/// `customType6`: stored text indexed with `DOCS` only.
pub fn custom_type6() -> FieldType {
    let mut t = custom_type();
    set(t.set_index_options(IndexOptions::Docs));
    t
}

/// `customType7`: stored, not indexed.
pub fn custom_type7() -> FieldType {
    let mut t = FieldType::new();
    set(t.set_stored(true));
    t
}

/// `customType8`: `TextField.TYPE_NOT_STORED` with term vectors.
pub fn custom_type8() -> FieldType {
    let mut t = FieldType::copy_of(&TextField::type_not_stored());
    set(t.set_store_term_vectors(true));
    t
}

fn field(name: &str, value: &str, t: FieldType) -> Field {
    Field::from_string(name, value, t).expect("a valid DocHelper field")
}

/// `fields`: the fourteen fields, in Java's order (the binary field second
/// to last, the large one last).
pub fn fields() -> Vec<Field> {
    vec![
        field(TEXT_FIELD_1_KEY, FIELD_1_TEXT, custom_type()),
        field(TEXT_FIELD_2_KEY, FIELD_2_TEXT, text_type_stored_with_tvs()),
        field(TEXT_FIELD_3_KEY, FIELD_3_TEXT, custom_type3()),
        field(KEYWORD_FIELD_KEY, KEYWORD_TEXT, StringField::type_stored()),
        field(NO_NORMS_KEY, NO_NORMS_TEXT, custom_type5()),
        field(NO_TF_KEY, NO_TF_TEXT, custom_type6()),
        field(UNINDEXED_FIELD_KEY, UNINDEXED_FIELD_TEXT, custom_type7()),
        field(
            UNSTORED_FIELD_1_KEY,
            UNSTORED_1_FIELD_TEXT,
            TextField::type_not_stored(),
        ),
        field(UNSTORED_FIELD_2_KEY, UNSTORED_2_FIELD_TEXT, custom_type8()),
        field(TEXT_FIELD_UTF1_KEY, FIELD_UTF1_TEXT, custom_type()),
        field(
            TEXT_FIELD_UTF2_KEY,
            FIELD_UTF2_TEXT,
            text_type_stored_with_tvs(),
        ),
        field(LAZY_FIELD_KEY, LAZY_FIELD_TEXT, custom_type()),
        Field::from_bytes(
            LAZY_FIELD_BINARY_KEY,
            LAZY_FIELD_BINARY_BYTES.to_vec(),
            custom_type7(),
        )
        .expect("a stored binary field"),
        field(LARGE_LAZY_FIELD_KEY, &LARGE_LAZY_FIELD_TEXT, custom_type()),
    ]
}

/// The field-name sets of Java's maps `all`, `indexed`, `stored`, ...
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FieldSets {
    pub all: BTreeSet<String>,
    pub indexed: BTreeSet<String>,
    pub unindexed: BTreeSet<String>,
    pub termvector: BTreeSet<String>,
    pub notermvector: BTreeSet<String>,
    pub stored: BTreeSet<String>,
    pub unstored: BTreeSet<String>,
    pub no_tf: BTreeSet<String>,
    pub no_norms: BTreeSet<String>,
    /// Always empty, as Java's (`isLazy` is commented out there).
    pub lazy: BTreeSet<String>,
}

/// Java's static initializer filling `all`, `indexed`, ... from `fields`.
pub fn field_sets() -> FieldSets {
    let mut s = FieldSets::default();
    for f in fields() {
        let name = f.name().to_string();
        let t = f.field_type();
        let indexed = t.index_options() != IndexOptions::None;
        s.all.insert(name.clone());
        if indexed {
            s.indexed.insert(name.clone());
        } else {
            s.unindexed.insert(name.clone());
        }
        if t.store_term_vectors() {
            s.termvector.insert(name.clone());
        }
        if indexed && !t.store_term_vectors() {
            s.notermvector.insert(name.clone());
        }
        if t.stored() {
            s.stored.insert(name.clone());
        } else {
            s.unstored.insert(name.clone());
        }
        if t.index_options() == IndexOptions::Docs {
            s.no_tf.insert(name.clone());
        }
        if t.omit_norms() {
            s.no_norms.insert(name);
        }
    }
    s
}

/// A `nameValues` value: text, or `LAZY_FIELD_BINARY_BYTES`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NameValue {
    Text(String),
    Bytes(Vec<u8>),
}

/// `nameValues`: each field's value by name.
pub fn name_values() -> BTreeMap<&'static str, NameValue> {
    let text = |s: &str| NameValue::Text(s.to_string());
    BTreeMap::from([
        (TEXT_FIELD_1_KEY, text(FIELD_1_TEXT)),
        (TEXT_FIELD_2_KEY, text(FIELD_2_TEXT)),
        (TEXT_FIELD_3_KEY, text(FIELD_3_TEXT)),
        (KEYWORD_FIELD_KEY, text(KEYWORD_TEXT)),
        (NO_NORMS_KEY, text(NO_NORMS_TEXT)),
        (NO_TF_KEY, text(NO_TF_TEXT)),
        (UNINDEXED_FIELD_KEY, text(UNINDEXED_FIELD_TEXT)),
        (UNSTORED_FIELD_1_KEY, text(UNSTORED_1_FIELD_TEXT)),
        (UNSTORED_FIELD_2_KEY, text(UNSTORED_2_FIELD_TEXT)),
        (LAZY_FIELD_KEY, text(LAZY_FIELD_TEXT)),
        (
            LAZY_FIELD_BINARY_KEY,
            NameValue::Bytes(LAZY_FIELD_BINARY_BYTES.to_vec()),
        ),
        (LARGE_LAZY_FIELD_KEY, text(&LARGE_LAZY_FIELD_TEXT)),
        (TEXT_FIELD_UTF1_KEY, text(FIELD_UTF1_TEXT)),
        (TEXT_FIELD_UTF2_KEY, text(FIELD_UTF2_TEXT)),
    ])
}

/// `setupDoc(doc)`: adds [`fields`] to `doc`.
pub fn setup_doc(doc: &mut Document) {
    for f in fields() {
        doc.add(f);
    }
}

/// `writeDoc(random, dir, doc)`: [`write_doc_with`] under
/// `new MockAnalyzer(random, MockTokenizer.WHITESPACE, false)` and the
/// default similarity.
///
/// # Errors
/// What indexing or committing fails with.
pub fn write_doc(
    r: &mut JavaRandom,
    dir: &dyn Directory,
    doc: &Document,
) -> Result<SegmentCommitInfo> {
    let analyzer = MockAnalyzer::with_tokenizer(r, mock_tokenizer::whitespace(), false).build();
    write_doc_with(dir, Arc::new(analyzer), None, doc)
}

/// `writeDoc(dir, analyzer, similarity, doc)`: indexes `doc` alone,
/// commits, and returns the newest segment.
///
/// # Errors
/// What indexing or committing fails with.
pub fn write_doc_with(
    dir: &dyn Directory,
    analyzer: Arc<Analyzer>,
    similarity: Option<Arc<dyn NormSimilarity>>,
    doc: &Document,
) -> Result<SegmentCommitInfo> {
    let mut w = IndexWriter::open(
        dir,
        Vec::new(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )?;
    w.set_analyzer(Some(analyzer));
    w.set_similarity(similarity);
    w.add_fields_document(doc)?;
    let info = w
        .commit()?
        .segments
        .last()
        .cloned()
        .expect("a commit after adding a document has a segment");
    w.close()?;
    Ok(info)
}

/// `numFields(doc)`.
pub fn num_fields(doc: &Document) -> usize {
    doc.fields().len()
}

/// `createDocument(n, indexName, numFields)`.
pub fn create_document(n: i32, index_name: &str, num_fields: i32) -> Document {
    let mut doc = Document::new();
    doc.add(field("id", &n.to_string(), string_type_stored_with_tvs()));
    doc.add(field(
        "indexname",
        index_name,
        string_type_stored_with_tvs(),
    ));
    let mut sb = format!("a{n}");
    doc.add(field("field1", &sb, text_type_stored_with_tvs()));
    sb.push_str(&format!(" b{n}"));
    for i in 1..num_fields {
        doc.add(field(
            &format!("field{}", i + 1),
            &sb,
            text_type_stored_with_tvs(),
        ));
    }
    doc
}
