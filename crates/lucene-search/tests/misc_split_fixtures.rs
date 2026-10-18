//! M14 T14.3 `misc`, differentially against Lucene 10.5.0:
//! `fixtures/src/GenMiscSplit.java`'s `misc/split.tsv`.
//!
//! `PKIndexSplitter` and `MultiPassIndexSplitter` run over the same source
//! indexes (`misc/split_a`, with deletions, and `misc/split_b`); every output
//! must hold the same live documents in the same order, in as many segments.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::{
    ExtendedQuery, MultiTermQuery, MultiTermSource, RewriteMethod, TermRangeQuery,
};
use lucene_search::misc::index::{MultiPassIndexSplitter, PkIndexSplitter};
use lucene_search::query::{BooleanQuery, Clause};
use lucene_store::directory::Directory;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn data() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/misc")
}

/// A private copy of the committed fixture index `name` under `tmp`:
/// `addIndexes` write-locks its sources, and a test must never write into
/// `fixtures/`.
fn source(name: &str, tmp: &TempDir) -> FsDirectory {
    let to = tmp.path().join(name);
    std::fs::create_dir_all(&to).unwrap();
    for entry in std::fs::read_dir(data().join(name)).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name() != "write.lock" {
            std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
        }
    }
    FsDirectory::open(to)
}

/// `(segments, ids)` per `(case, output)`.
fn expected() -> BTreeMap<(String, String), (usize, Vec<String>)> {
    let tsv = std::fs::read_to_string(data().join("split.tsv")).unwrap();
    let mut out = BTreeMap::new();
    for line in tsv.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        assert_eq!(c[0], "R");
        let ids = c.get(4).copied().unwrap_or("");
        let ids = if ids.is_empty() {
            Vec::new()
        } else {
            ids.split(',').map(String::from).collect()
        };
        out.insert(
            (c[1].to_string(), c[2].to_string()),
            (c[3].parse().unwrap(), ids),
        );
    }
    out
}

/// The live documents' `id`s in order, and the segment count.
fn read(dir: &dyn Directory) -> (usize, Vec<String>) {
    let reader = DirectoryReader::open(dir).unwrap();
    let mut ids = Vec::new();
    for s in reader.segment_readers() {
        let field = s.field_infos().field_by_name("id").unwrap().number;
        for d in 0..s.max_doc {
            if s.live_docs()
                .is_some_and(|l| !l.get(usize::try_from(d).unwrap()))
            {
                continue;
            }
            let doc = s.stored_document(d).unwrap().unwrap();
            let f = doc.fields.iter().find(|f| f.field_number == field).unwrap();
            match &f.value {
                lucene_codecs::stored_fields::FieldValue::String(s) => ids.push(s.clone()),
                other => panic!("id is not a string: {other:?}"),
            }
        }
    }
    (reader.segment_readers().len(), ids)
}

fn range(
    lower: Option<&str>,
    upper: Option<&str>,
    inc_lower: bool,
    inc_upper: bool,
) -> BooleanQuery {
    let q = TermRangeQuery::new(
        "id",
        lower.map(|s| s.as_bytes().to_vec()),
        upper.map(|s| s.as_bytes().to_vec()),
        inc_lower,
        inc_upper,
    );
    BooleanQuery {
        must: vec![Clause::Extended(Box::new(ExtendedQuery::MultiTerm(
            MultiTermQuery::new(MultiTermSource::TermRange(q), RewriteMethod::default()),
        )))],
        ..Default::default()
    }
}

fn dirs(n: usize, tmp: &TempDir) -> Vec<FsDirectory> {
    (0..n)
        .map(|i| FsDirectory::open(tmp.path().join(format!("o{i}"))))
        .collect()
}

#[test]
fn pk_index_splitter_matches_lucene() {
    let want = expected();
    let sources = TempDir::new("split-src");
    let src = source("split_a", &sources);
    for case in ["pk_mid", "pk_mid_low", "pk_mid_high", "pk_range"] {
        let tmp = TempDir::new("split-pk");
        let o = dirs(2, &tmp);
        let splitter = match case {
            "pk_mid" => PkIndexSplitter::with_mid_term(&src, &o[0], &o[1], "id", b"k010"),
            "pk_mid_low" => PkIndexSplitter::with_mid_term(&src, &o[0], &o[1], "id", b"k000"),
            "pk_mid_high" => PkIndexSplitter::with_mid_term(&src, &o[0], &o[1], "id", b"k999"),
            _ => PkIndexSplitter::new(
                &src,
                &o[0],
                &o[1],
                range(Some("k005"), Some("k015"), true, true),
            ),
        };
        splitter.split().unwrap();
        for (i, name) in ["first", "second"].into_iter().enumerate() {
            let key = (case.to_string(), name.to_string());
            assert_eq!(read(&o[i]), want[&key], "{case} {name}");
        }
    }
}

#[test]
fn multi_pass_index_splitter_matches_lucene() {
    let want = expected();
    let sources = TempDir::new("split-src");
    let a = source("split_a", &sources);
    let b = source("split_b", &sources);
    let mut checked = 0;
    for seq in [true, false] {
        for parts in [2usize, 3, 5] {
            for (base, inputs) in [
                ("multi_a", vec![&a as &dyn Directory]),
                ("multi_b", vec![&b as &dyn Directory]),
                ("multi_ab", vec![&a as &dyn Directory, &b]),
            ] {
                let case = format!("{base}{}{parts}", if seq { "_seq_" } else { "_rr_" });
                let tmp = TempDir::new("split-multi");
                let o = dirs(parts, &tmp);
                let outs: Vec<&dyn Directory> = o.iter().map(|d| d as &dyn Directory).collect();
                MultiPassIndexSplitter.split(&inputs, &outs, seq).unwrap();
                for (i, dir) in o.iter().enumerate() {
                    let key = (case.clone(), format!("part{i}"));
                    assert_eq!(read(dir), want[&key], "{case} part{i}");
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 2 * (2 + 3 + 5) * 3);
}

#[test]
fn multi_pass_index_splitter_refuses_what_java_refuses() {
    let sources = TempDir::new("split-src");
    let a = source("split_a", &sources);
    let tmp = TempDir::new("split-refuse");
    let o = dirs(2, &tmp);
    let one: Vec<&dyn Directory> = vec![&o[0]];
    assert!(MultiPassIndexSplitter.split(&[&a], &one, true).is_err());
    let two: Vec<&dyn Directory> = vec![&o[0], &o[1]];
    // A one-document index has too few documents to split.
    let src = TempDir::new("split-one");
    let one_doc = FsDirectory::open(src.path());
    let mut w = lucene_index::index_writer::IndexWriter::open(
        &one_doc,
        Vec::new(),
        "Lucene104",
        lucene_index::segment_info::LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    w.add_document(lucene_codecs::stored_fields::Document::default())
        .unwrap();
    w.close().unwrap();
    assert!(MultiPassIndexSplitter
        .split(&[&one_doc], &two, false)
        .is_err());
}
