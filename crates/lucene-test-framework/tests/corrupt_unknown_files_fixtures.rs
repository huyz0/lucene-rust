//! Java MockDirectoryWrapper's exact unknown-file selection over retained
//! commits, malformed commit names, and Java's codec-name regular expression.
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use lucene_store::{ByteBuffersDirectory, DataOutput, Directory};
use lucene_test_framework::store::base_directory_wrapper::BaseDirectoryWrapper;
use lucene_test_framework::store::mock_directory_wrapper::{MockDirectoryWrapper, Throttling};
use lucene_util::java_random::JavaRandom;

const DATA: &str = include_str!("../../../fixtures/testframework/data/corrupt_unknown_files.tsv");
type Files = BTreeMap<String, Vec<u8>>;

fn unhex(text: &str) -> Vec<u8> {
    assert_eq!(text.len() % 2, 0);
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn filename(text: &str) -> String {
    String::from_utf8(unhex(text)).unwrap()
}

fn names(text: &str) -> BTreeSet<String> {
    text.split(',')
        .filter(|name| !name.is_empty())
        .map(filename)
        .collect()
}

fn source() -> Files {
    read_source("merge_policies/commits_index")
}

fn read_source(path: &str) -> Files {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data")
        .join(path);
    std::fs::read_dir(root)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (
                entry.file_name().into_string().unwrap(),
                std::fs::read(entry.path()).unwrap(),
            )
        })
        .collect()
}

fn extras() -> Files {
    [
        "_a.doc",
        "_abc_suffix.ext",
        "_0.",
        "_0___.x",
        "_a..",
        "_a_.",
        "_A.doc",
        "_a-b.doc",
        "a.doc",
        "_afoo",
        "_0_suffix",
        "readme",
        "write.lock",
        "otherwrite.lock",
        "_a.write.lock",
        "_a.notwrite.lock",
        "_0_x\nz.foo",
        "_0.x\nz",
        "_0.x\r",
        "_0.x\u{85}",
        "_0.x\u{2028}",
        "_0.x\u{2029}",
        "pending_segments",
        "pending_segments_1",
        "pending_segmentsAnything",
        "pending_segments_tail\n",
        "pending_segmentswrite.lock",
    ]
    .into_iter()
    .map(|name| (name.to_owned(), Vec::new()))
    .collect()
}

fn simpletext_files(label: &str) -> Files {
    let (mut files, operation) = if let Some(operation) = label.strip_prefix("simpletext_retained_")
    {
        let mut retained = read_source("codecs_simpletext/codec/flush/index");
        retained.extend(read_source("codecs_simpletext/codec/merged/index"));
        (retained, operation)
    } else {
        let rest = label.strip_prefix("simpletext_").unwrap();
        let (stage, operation) = rest.split_once('_').unwrap();
        assert!(matches!(stage, "flush" | "merged"));
        (
            read_source(&format!("codecs_simpletext/codec/{stage}/index")),
            operation,
        )
    };
    if operation == "clean" {
        return files;
    }
    let si = files
        .keys()
        .find(|name| name.ends_with(".si"))
        .unwrap()
        .clone();
    let commit = files
        .keys()
        .find(|name| name.starts_with("segments_"))
        .unwrap()
        .clone();
    let live = files
        .keys()
        .find(|name| name.ends_with(".liv"))
        .unwrap()
        .clone();
    files.extend(extras());
    match operation {
        "extras" => (),
        "missing_si" => {
            files.remove(&si);
        }
        "corrupt_si" => {
            files.insert(si, vec![1, 2, 3]);
        }
        "corrupt_commit" => {
            files.insert(commit, vec![1, 2, 3]);
        }
        "corrupt_live" => {
            files.insert(live, vec![1, 2, 3]);
        }
        "missing_live" => {
            files.remove(&live);
        }
        "latest_only" => {
            files.remove("segments_2");
        }
        "oldest_only" => {
            files.remove("segments_5");
        }
        _ => panic!("unknown SimpleText case {label}"),
    }
    files
}

fn files_for(label: &str, source: &Files, snapshot: &[u8]) -> Files {
    if label.starts_with("simpletext_") {
        return simpletext_files(label);
    }
    if label == "empty" {
        return Files::new();
    }
    if label == "name_edges" {
        return extras();
    }
    let mut files = source.clone();
    if label == "oldest_only" {
        files.remove("segments_2");
        files.remove("segments_3");
    } else {
        files.insert("segments_4".into(), snapshot.to_vec());
    }
    if label == "retained_clean" {
        return files;
    }
    files.extend(extras());
    match label {
        "retained_extras" | "oldest_only" => (),
        name if name.starts_with("cross_") => {
            files.remove("segments_1");
            files.remove("segments_2");
        }
        "latest_only" => {
            files.retain(|name, _| !name.starts_with("segments_") || name == "segments_4")
        }
        "missing_si" => {
            files.remove("_0.si");
        }
        "corrupt_si" => {
            files.insert("_0.si".into(), vec![1, 2, 3]);
        }
        _ => {
            if let Some(encoded) = label.strip_prefix("alias_") {
                let snapshot = files.remove("segments_4").unwrap();
                files.insert(filename(encoded), snapshot);
            } else if let Some(encoded) = label.strip_prefix("bad_") {
                files.insert(filename(encoded), vec![1, 2, 3]);
            } else {
                panic!("unknown Java fixture case {label}");
            }
        }
    }
    files
}

fn directory(files: &Files) -> MockDirectoryWrapper {
    let inner = ByteBuffersDirectory::new();
    for (name, bytes) in files {
        let mut out = inner.create_output(name).unwrap();
        out.write_bytes(bytes);
        out.close().unwrap();
    }
    let dir = MockDirectoryWrapper::new(&mut JavaRandom::new(31), inner);
    dir.set_check_index_on_close(false);
    dir.set_throttling(Throttling::Never);
    dir
}

fn contents(dir: &dyn Directory) -> Files {
    dir.list_all()
        .unwrap()
        .into_iter()
        .map(|name| {
            let bytes = dir.open(&name).unwrap().to_vec();
            (name, bytes)
        })
        .collect()
}

#[test]
fn unknown_file_selection_matches_actual_java() {
    let mut lines = DATA.lines();
    let snapshot: Vec<_> = lines.next().unwrap().split('\t').collect();
    assert_eq!(&snapshot[..2], &["snapshot", "segments_4"]);
    let snapshot = unhex(snapshot[2]);
    let proof: Vec<_> = lines.next().unwrap().split('\t').collect();
    assert_eq!(&proof[..2], &["proof", "old_only"]);
    let old_only = names(proof[2]);
    assert!(!old_only.is_empty());
    let source = source();
    assert!(old_only.iter().all(|name| source.contains_key(name)));
    let mut cases = 0;
    let mut seen = BTreeSet::new();
    let mut replacements: BTreeMap<String, Files> = BTreeMap::new();
    for line in lines {
        let row: Vec<_> = line.split('\t').collect();
        if row[0] == "replace" {
            assert_eq!(row.len(), 4, "{line}");
            assert!(replacements
                .entry(row[1].to_owned())
                .or_default()
                .insert(row[2].to_owned(), unhex(row[3]))
                .is_none());
            continue;
        }
        assert_eq!(row.len(), 5, "{line}");
        assert_eq!(row[0], "case");
        let label = row[1];
        assert!(seen.insert(label));
        let mut files = files_for(label, &source, &snapshot);
        if label.starts_with("cross_") {
            files.extend(
                replacements
                    .remove(label)
                    .expect("cross-check bytes from Java"),
            );
        }
        let dir = directory(&files);
        let mut calls = 0;
        let mut victims = BTreeSet::new();
        let result = dir.corrupt_unknown_files_with(|selected| {
            calls += 1;
            victims = selected.clone();
            Ok(())
        });
        assert_eq!(
            if result.is_ok() { "ok" } else { "failure" },
            row[2],
            "{label}: {result:?}"
        );
        assert_eq!(calls, row[3].parse::<usize>().unwrap(), "{label}");
        assert_eq!(victims, names(row[4]), "{label}");
        assert_eq!(contents(&dir), files, "selection changed bytes: {label}");
        if label == "retained_clean" {
            assert!(victims.is_disjoint(&old_only));
        } else if label == "latest_only" {
            assert!(old_only.is_subset(&victims));
        }
        dir.close().unwrap();
        cases += 1;
    }
    assert_eq!(cases, 56);
    assert!(replacements.is_empty(), "unreplayed Java mutations");
    assert!(seen.contains("retained_clean") && seen.contains("latest_only"));
}
