#![cfg(test)]
use super::*;

fn run(args: &[&str]) -> (ToolResult, String, String) {
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let r = main(
        &args,
        &mut Io {
            out: &mut out,
            err: &mut err,
        },
    );
    (
        r,
        String::from_utf8(out).unwrap(),
        String::from_utf8(err).unwrap(),
    )
}

#[test]
fn the_argument_checks_are_javas() {
    let (r, out, err) = run(&["-update"]);
    assert_eq!(r.unwrap(), 1);
    assert!(out.is_empty());
    assert_eq!(err, format!("Usage: {USAGE}\n"));
    let (r, _, _) = run(&["-x"]);
    assert_eq!(
        r.unwrap_err().to_string(),
        "java.lang.IllegalArgumentException: unknown parameter -x"
    );
    let (r, _, _) = run(&["-docs"]);
    assert!(r
        .unwrap_err()
        .to_string()
        .starts_with("java.lang.ArrayIndexOutOfBoundsException"));
}

#[test]
fn a_write_failure_is_caught_and_printed() {
    let tmp = lucene_util::test_support::TempDir::new("tools-index-files");
    let docs = tmp.path().join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(docs.join("a.txt"), "hello").unwrap();
    // the index path is a file
    let index = tmp.path().join("index");
    std::fs::write(&index, "").unwrap();
    let (r, out, _) = run(&[
        "-create",
        "-index",
        index.to_str().unwrap(),
        "-docs",
        docs.to_str().unwrap(),
    ]);
    assert_eq!(r.unwrap(), 0);
    assert!(out.contains(" caught a class java.io.IOException\n with message: "));
    // a dictionary that is not there is an `IOException`, caught
    let (r, out, _) = run(&[
        "-index",
        tmp.path().join("i2").to_str().unwrap(),
        "-docs",
        docs.to_str().unwrap(),
        "-knn_dict",
        tmp.path().join("missing.txt").to_str().unwrap(),
    ]);
    assert_eq!(r.unwrap(), 0);
    assert!(out.contains(" caught a class java.io.IOException"));
    // one that does not parse is thrown
    let bad = tmp.path().join("bad.txt");
    std::fs::write(&bad, "first 1 0\nx 1\n").unwrap();
    let (r, _, _) = run(&[
        "-index",
        tmp.path().join("i3").to_str().unwrap(),
        "-docs",
        docs.to_str().unwrap(),
        "-knn_dict",
        bad.to_str().unwrap(),
    ]);
    assert!(r
        .unwrap_err()
        .to_string()
        .starts_with("java.lang.IllegalStateException"));
}

#[test]
fn more_than_a_hundred_documents_over_the_toy_dictionary_is_thrown() {
    let tmp = lucene_util::test_support::TempDir::new("tools-index-toy");
    let docs = tmp.path().join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    for i in 0..101 {
        std::fs::write(docs.join(format!("{i}.txt")), "hello").unwrap();
    }
    let dict = tmp.path().join("dict.txt");
    std::fs::write(&dict, "first 1 0\nhello 0 1\n").unwrap();
    let (r, out, _) = run(&[
        "-index",
        tmp.path().join("index").to_str().unwrap(),
        "-docs",
        docs.to_str().unwrap(),
        "-knn_dict",
        dict.to_str().unwrap(),
    ]);
    assert_eq!(
        r.unwrap_err().to_string(),
        "java.lang.RuntimeException: Are you (ab)using the toy vector dictionary? See the \
             package javadocs to understand why you got this exception."
    );
    assert_eq!(out.matches("adding ").count(), 101);
}

#[test]
fn a_dictionary_of_no_dimensions_is_thrown_as_its_build_throws_it() {
    let tmp = lucene_util::test_support::TempDir::new("tools-index-dim0");
    let docs = tmp.path().join("a.txt");
    std::fs::write(&docs, "hello").unwrap();
    let dict = tmp.path().join("dict.txt");
    std::fs::write(&dict, "first\nhello\n").unwrap();
    let (r, _, _) = run(&[
        "-index",
        tmp.path().join("index").to_str().unwrap(),
        "-docs",
        docs.to_str().unwrap(),
        "-knn_dict",
        dict.to_str().unwrap(),
    ]);
    assert_eq!(
        r.unwrap_err().to_string(),
        "java.lang.IllegalArgumentException: Cannot normalize a zero-length vector"
    );
}

#[cfg(unix)]
#[test]
fn a_dangling_link_in_a_walk_is_printed_and_skipped() {
    let tmp = lucene_util::test_support::TempDir::new("tools-index-link");
    let docs = tmp.path().join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(docs.join("a.txt"), "hello").unwrap();
    std::os::unix::fs::symlink(tmp.path().join("gone"), docs.join("b.txt")).unwrap();
    let (r, out, err) = run(&[
        "-index",
        tmp.path().join("index").to_str().unwrap(),
        "-docs",
        docs.to_str().unwrap(),
    ]);
    assert_eq!(r.unwrap(), 0);
    assert!(err.starts_with("java.io.IOException: "), "{err}");
    assert!(out.contains("Indexed 1 documents in "), "{out}");
}

#[test]
fn a_walk_indexes_nested_files_and_a_vanished_file_fails_its_read() {
    let tmp = lucene_util::test_support::TempDir::new("tools-index-walk");
    let docs = tmp.path().join("docs");
    std::fs::create_dir_all(docs.join("sub")).unwrap();
    std::fs::write(docs.join("sub").join("a.txt"), "hello").unwrap();
    let index = tmp.path().join("index");
    let dict = tmp.path().join("dict.txt");
    std::fs::write(&dict, "first 1 0\nhello 0 1\n").unwrap();
    let mut out = Vec::new();
    let mut err = Vec::new();
    let mut io = Io {
        out: &mut out,
        err: &mut err,
    };
    let dir = FsDirectory::open(&index);
    std::fs::create_dir_all(&index).unwrap();
    KnnVectorDict::build(&dict, &dir, KNN_DICT).unwrap();
    let d = KnnVectorDict::open(&dir, KNN_DICT).unwrap();
    let embeddings = DemoEmbeddings::new(&d);
    let mut writer = IndexWriter::open(&dir, Vec::new(), CODEC, VERSION).unwrap();
    let mut indexer = Indexer {
        writer: &mut writer,
        embeddings: Some(&embeddings),
        create: false,
        io: &mut io,
    };
    indexer.walk(&docs).unwrap();
    // a file that vanished between the listing and the read
    assert!(matches!(
        indexer.index_doc(&docs.join("gone.txt")),
        Err(Failure::Io(_))
    ));
    writer.close().unwrap();
    let out = String::from_utf8(out).unwrap();
    assert_eq!(out.lines().count(), 1, "{out}");
    assert!(out.starts_with("updating "), "{out}");
    assert!(out.trim_end().ends_with("a.txt"), "{out}");
}

#[test]
fn a_failing_file_in_a_walk_is_reported_and_skipped() {
    let tmp = lucene_util::test_support::TempDir::new("tools-index-walk-fail");
    let docs = tmp.path().join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(docs.join("a.txt"), "hello").unwrap();
    // a dangling link: listed, but its read fails
    #[cfg(unix)]
    std::os::unix::fs::symlink(docs.join("none"), docs.join("dangling")).unwrap();
    // a link to the directory itself: visited as a file, not descended
    // into (Java's walk does not follow links) -- its read fails too
    #[cfg(unix)]
    std::os::unix::fs::symlink(&docs, docs.join("loop")).unwrap();
    let index = tmp.path().join("index");
    let (r, out, err) = run(&[
        "-index",
        index.to_str().unwrap(),
        "-docs",
        docs.to_str().unwrap(),
    ]);
    assert_eq!(r.unwrap(), 0);
    assert!(out.contains("Indexed 1 documents"), "{out}");
    #[cfg(unix)]
    assert_eq!(err.matches("java.io.IOException: ").count(), 2, "{err}");
    #[cfg(not(unix))]
    assert!(err.is_empty(), "{err}");
}

#[test]
fn a_dictionary_over_an_existing_one_is_file_already_exists() {
    let tmp = lucene_util::test_support::TempDir::new("tools-index-dict-again");
    let docs = tmp.path().join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(docs.join("a.txt"), "hello").unwrap();
    let dict = tmp.path().join("dict.txt");
    std::fs::write(&dict, "first 1 0\nhello 0 1\n").unwrap();
    let index = tmp.path().join("index");
    let args = [
        "-index",
        index.to_str().unwrap(),
        "-docs",
        docs.to_str().unwrap(),
        "-knn_dict",
        dict.to_str().unwrap(),
    ];
    let (r, out, _) = run(&args);
    assert_eq!(r.unwrap(), 0);
    assert!(out.contains("Indexed 1 documents"), "{out}");
    let (r, out, _) = run(&args);
    assert_eq!(r.unwrap(), 0);
    let real = real_path(index.to_str().unwrap()).join("knn-dict.bin");
    assert!(
        out.contains(&format!(
            " caught a class java.nio.file.FileAlreadyExistsException\n with message: {}",
            real.display()
        )),
        "{out}"
    );
    assert!(!out.contains("Indexed"), "{out}");
    // a path that cannot be resolved is printed as given
    assert_eq!(real_path("/no/such/dir"), PathBuf::from("/no/such/dir"));
}
