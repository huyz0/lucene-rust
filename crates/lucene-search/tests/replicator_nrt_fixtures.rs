//! `replicator.nrt` against Lucene 10.5.0's own `PrimaryNode`
//! (`fixtures/src/GenReplicator.java`, `fixtures/data/replicator/`).
//!
//! Java recorded, for an index of three segments with a deletion (`plain`)
//! and the same index force-merged with the primary's merged-segment warmer
//! running (`merged`), the `CopyState` its primary hands a replica: every
//! file's `FileMetaData`, the `segments_N` bytes, the completed merge files
//! and what the warmer asked to pre-copy. This suite checks that
//!
//! * the Rust node reads the same `FileMetaData` off the same files,
//! * the Rust `SegmentInfos` reads Java's `infosBytes` and writes them back
//!   byte for byte, and names the same files,
//! * a Rust replica copies a point a Java primary published -- the bytes
//!   arrive identical, and the reader over them sees Java's documents.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lucene_index::segment_infos::{self, SegmentInfos};
use lucene_search::replicator::nrt::local::{CopySource, FileStream, LocalReplicaHooks};
use lucene_search::replicator::nrt::{
    CopyOneFile, CopyState, CopyStream, FileMetaData, NodeCore, ReplicaNode, PRIMARY_GEN_KEY,
    VERSION_KEY,
};
use lucene_store::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;

fn fixture_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/replicator")
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// One tag's rows of `nrt.tsv`.
#[derive(Default)]
struct Point {
    version: i64,
    gen: i64,
    primary_gen: i64,
    files: BTreeMap<String, FileMetaData>,
    merged: BTreeSet<String>,
    warmed: BTreeMap<String, FileMetaData>,
    infos_bytes: Vec<u8>,
}

fn meta(cols: &[&str]) -> (String, FileMetaData) {
    (
        cols[2].to_string(),
        FileMetaData {
            length: cols[3].parse().unwrap(),
            checksum: cols[4].parse().unwrap(),
            header: unhex(cols[5]),
            footer: unhex(cols[6]),
        },
    )
}

fn points() -> BTreeMap<String, Point> {
    let text = std::fs::read_to_string(fixture_root().join("nrt.tsv")).unwrap();
    let mut out: BTreeMap<String, Point> = BTreeMap::new();
    for line in text.lines().filter(|l| !l.is_empty()) {
        let cols: Vec<&str> = line.split('\t').collect();
        let p = out.entry(cols[1].to_string()).or_default();
        match cols[0] {
            "P" => {
                p.version = cols[2].parse().unwrap();
                p.gen = cols[3].parse().unwrap();
                p.primary_gen = cols[4].parse().unwrap();
            }
            "F" => {
                let (name, m) = meta(&cols);
                p.files.insert(name, m);
            }
            "W" => {
                let (name, m) = meta(&cols);
                p.warmed.insert(name, m);
            }
            "M" => {
                p.merged.insert(cols[2].to_string());
            }
            "I" => p.infos_bytes = unhex(cols[2]),
            other => panic!("unknown row kind {other}"),
        }
    }
    assert_eq!(
        out.keys().map(String::as_str).collect::<Vec<_>>(),
        ["merged", "plain"]
    );
    out
}

#[test]
fn file_metadata_matches_lucene() {
    for (tag, point) in points() {
        let dir: Arc<dyn Directory> = Arc::new(FsDirectory::open(fixture_root().join(&tag)));
        let node = NodeCore::new(0, true, dir, None);
        assert!(!point.files.is_empty(), "{tag}");
        for (name, java) in point.files.iter().chain(point.warmed.iter()) {
            let rust = node
                .read_local_file_metadata(name)
                .unwrap()
                .unwrap_or_else(|| panic!("{tag}/{name}: no metadata"));
            assert_eq!(&rust, java, "{tag}/{name}");
            assert_eq!(rust.header, java.header, "{tag}/{name} header");
            assert_eq!(rust.footer, java.footer, "{tag}/{name} footer");
        }
    }
}

#[test]
fn infos_bytes_round_trip_and_name_the_point_files() {
    for (tag, point) in points() {
        let infos = segment_infos::parse(&point.infos_bytes, point.gen).unwrap();
        assert_eq!(infos.version, point.version, "{tag}");
        assert_eq!(infos.generation, point.gen, "{tag}");
        assert_eq!(
            segment_infos::to_bytes(&infos),
            point.infos_bytes,
            "{tag}: SegmentInfos.write of Java's infos"
        );
        let dir = FsDirectory::open(fixture_root().join(&tag));
        let named: BTreeSet<String> =
            lucene_index::index_file_deleter::files_of_infos(&dir, &infos, false)
                .unwrap()
                .into_iter()
                .collect();
        let java: BTreeSet<String> = point.files.keys().cloned().collect();
        assert_eq!(named, java, "{tag}: SegmentInfos.files(false)");
    }
}

/// Java's own `PreCopyMergedSegmentWarmer` output, cross-checked against
/// the rest of Java's rows: the fixture's merged point is exactly the files
/// Java's warmer pre-copied. The Rust warmer is not run over these bytes;
/// `replicator_nrt_merge.rs` covers it.
#[test]
fn javas_warmer_pre_copied_exactly_the_merged_segment() {
    let all = points();
    let plain = &all["plain"];
    assert!(plain.merged.is_empty() && plain.warmed.is_empty());
    let merged = &all["merged"];
    let warmed: BTreeSet<String> = merged.warmed.keys().cloned().collect();
    assert_eq!(warmed, merged.merged, "completedMergeFiles == pre-copied");
    // The merged point is that one segment: every file was pre-copied.
    let files: BTreeSet<String> = merged.files.keys().cloned().collect();
    assert_eq!(files, warmed);
    for (name, m) in &merged.warmed {
        assert_eq!(&merged.files[name], m, "{name}");
    }
}

/// A Java primary's published point, served from its directory.
struct JavaPrimary {
    dir: FsDirectory,
    state: CopyState,
    released: Mutex<usize>,
}

impl CopySource for JavaPrimary {
    fn copy_state(&self) -> lucene_search::Result<CopyState> {
        Ok(self.state.clone())
    }
    fn release_copy_state(&self, _copy_state: CopyState) {
        *self.released.lock().unwrap() += 1;
    }
    fn open_files(&self, requests: &[(String, u64)]) -> lucene_search::Result<Box<dyn CopyStream>> {
        Ok(Box::new(FileStream::open(&self.dir, requests)?))
    }
}

fn java_primary(tag: &str, point: &Point) -> Arc<JavaPrimary> {
    let infos: SegmentInfos = segment_infos::parse(&point.infos_bytes, point.gen).unwrap();
    Arc::new(JavaPrimary {
        dir: FsDirectory::open(fixture_root().join(tag)),
        state: CopyState {
            files: Arc::new(point.files.clone()),
            version: point.version,
            gen: point.gen,
            infos_bytes: Arc::new(point.infos_bytes.clone()),
            completed_merge_files: Arc::new(point.merged.clone()),
            primary_gen: point.primary_gen,
            infos: Arc::new(infos),
            hold: None,
        },
        released: Mutex::new(0),
    })
}

fn wait_for(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn a_rust_replica_copies_a_point_a_java_primary_published() {
    for (tag, point) in points() {
        let source = java_primary(&tag, &point);
        let tmp = TempDir::new(&format!("replicator-java-{tag}"));
        let transport = LocalReplicaHooks::new(source.clone(), None);
        let replica = ReplicaNode::new(
            1,
            Arc::new(FsDirectory::open(tmp.path())),
            None,
            transport.clone(),
        )
        .unwrap();
        replica.start(point.primary_gen).unwrap();
        assert_eq!(replica.acquire().unwrap().num_docs(), 0, "{tag}: empty");

        let job = replica
            .new_nrt_point(point.primary_gen, point.version)
            .unwrap()
            .expect("a new point starts a copy");
        assert_eq!(job.file_names_to_copy().len(), point.files.len(), "{tag}");
        wait_for(&tag, || {
            !replica.is_copying() && replica.current_searching_version().unwrap() == point.version
        });

        // 15 documents in three segments, `k003` deleted.
        let reader = replica.acquire().unwrap();
        assert_eq!(reader.num_docs(), 14, "{tag}");
        assert_eq!(reader.max_doc(), if tag == "merged" { 14 } else { 15 });
        assert_eq!(reader.doc_freq("id", b"k007").unwrap(), 1, "{tag}");
        assert_eq!(
            reader.doc_freq("body", b"alpha").unwrap(),
            i64::from(reader.max_doc())
        );
        drop(reader);

        // Every file arrived byte for byte.
        let java_dir = FsDirectory::open(fixture_root().join(&tag));
        for name in point.files.keys() {
            assert_eq!(
                replica.core().dir().open(name).unwrap().to_vec(),
                java_dir.open(name).unwrap().to_vec(),
                "{tag}/{name}"
            );
        }

        // A commit names the point's version and keeps exactly its files.
        replica.commit().unwrap();
        let committed = segment_infos::read_latest(&**replica.core().dir()).unwrap();
        let data: BTreeMap<_, _> = committed.user_data.iter().cloned().collect();
        assert_eq!(data[VERSION_KEY], point.version.to_string());
        assert_eq!(committed.version, point.version);
        let on_dir: BTreeSet<String> = replica
            .core()
            .dir()
            .list_all()
            .unwrap()
            .into_iter()
            .filter(|f| f != "write.lock" && !f.starts_with("segments_"))
            .collect();
        let java: BTreeSet<String> = point.files.keys().cloned().collect();
        assert_eq!(on_dir, java, "{tag}");

        replica.close().unwrap();
        transport.close();
        assert!(
            *source.released.lock().unwrap() >= 1,
            "{tag}: point released"
        );
    }
}

/// A copy of `plain`'s `name` in a scratch directory, `damage` applied.
fn damaged_copy(tmp: &TempDir, name: &str, damage: impl FnOnce(&mut Vec<u8>)) -> FsDirectory {
    let mut bytes = std::fs::read(fixture_root().join("plain").join(name)).unwrap();
    damage(&mut bytes);
    std::fs::write(tmp.path().join(name), bytes).unwrap();
    FsDirectory::open(tmp.path())
}

#[test]
fn damaged_java_files_do_not_pass_for_javas_metadata() {
    let java = &points()["plain"].files;
    let name = "_1.fdt";
    let java_meta = &java[name];

    // A broken footer magic: the file is corrupt, so the replica copies it.
    let tmp = TempDir::new("replicator-java-footer");
    let dir = damaged_copy(&tmp, name, |b| {
        let at = b.len() - 16;
        b[at] ^= 0xff;
    });
    let node = NodeCore::new(0, true, Arc::new(dir), None);
    assert_eq!(node.read_local_file_metadata(name).unwrap(), None);

    // A different stored checksum still parses, but no longer matches Java's.
    let tmp = TempDir::new("replicator-java-checksum");
    let dir = damaged_copy(&tmp, name, |b| {
        let last = b.len() - 1;
        b[last] ^= 0x01;
    });
    let node = NodeCore::new(0, true, Arc::new(dir), None);
    let rust = node.read_local_file_metadata(name).unwrap().unwrap();
    assert_eq!(rust.header, java_meta.header);
    assert_ne!(rust.checksum, java_meta.checksum);
    assert_ne!(&rust, java_meta);
}

/// Copies `name` out of `src` into a fresh replica against `meta`.
fn copy_one(src: &FsDirectory, name: &str, meta: FileMetaData) -> lucene_search::Result<()> {
    let tmp = TempDir::new("replicator-java-dest");
    let source = java_primary("plain", &points()["plain"]);
    let replica = ReplicaNode::new(
        1,
        Arc::new(FsDirectory::open(tmp.path())),
        None,
        LocalReplicaHooks::new(source, None),
    )
    .unwrap();
    let mut stream = FileStream::open(src, &[(name.to_string(), 0)]).unwrap();
    stream.read_vlong().unwrap();
    let mut copy = CopyOneFile::new(Arc::clone(replica.core()), name, meta, 4096)?;
    let result = loop {
        match copy.visit(&mut stream) {
            Ok(true) => break Ok(()),
            Ok(false) => {}
            Err(e) => break Err(e),
        }
    };
    replica.close().unwrap();
    result
}

#[test]
fn a_replica_rejects_java_bytes_that_do_not_match_javas_checksum() {
    let java = &points()["plain"].files;
    let name = "_1.fdt";
    let plain = FsDirectory::open(fixture_root().join("plain"));
    copy_one(&plain, name, java[name].clone()).expect("Java's own bytes copy");

    // The primary announces a checksum the bytes do not have.
    let mut wrong = java[name].clone();
    wrong.checksum ^= 1;
    let err = copy_one(&plain, name, wrong).unwrap_err();
    assert!(err.to_string().contains("checksum mismatch"), "{err}");

    // One body byte flipped in transit: the running checksum catches it.
    let tmp = TempDir::new("replicator-java-body");
    let damaged = damaged_copy(&tmp, name, |b| {
        let at = b.len() - 20;
        b[at] ^= 0x01;
    });
    let err = copy_one(&damaged, name, java[name].clone()).unwrap_err();
    assert!(err.to_string().contains("checksum mismatch"), "{err}");
}

/// How a [`Faulty`] primary misbehaves.
#[derive(Clone, Copy)]
enum Fault {
    /// `CMD_FETCH_FILES` is refused.
    RefuseOpen,
    /// The stream breaks after this many bytes.
    BreakAfter(usize),
    /// Every read waits a little: a copy stays in flight.
    Slow,
}

/// A Java primary's point served badly.
struct Faulty {
    inner: Arc<JavaPrimary>,
    fault: Fault,
}

struct FaultyStream {
    inner: Box<dyn CopyStream>,
    fault: Fault,
    sent: usize,
}

impl CopyStream for FaultyStream {
    fn read_bytes(&mut self, buf: &mut [u8]) -> lucene_search::Result<()> {
        match self.fault {
            Fault::BreakAfter(n) if self.sent.saturating_add(buf.len()) > n => {
                return Err(lucene_search::Error::IllegalState(
                    "connection reset".to_string(),
                ));
            }
            Fault::Slow => std::thread::sleep(Duration::from_millis(5)),
            _ => {}
        }
        self.sent = self.sent.saturating_add(buf.len());
        self.inner.read_bytes(buf)
    }
}

impl CopySource for Faulty {
    fn copy_state(&self) -> lucene_search::Result<CopyState> {
        self.inner.copy_state()
    }
    fn release_copy_state(&self, copy_state: CopyState) {
        self.inner.release_copy_state(copy_state);
    }
    fn open_files(&self, requests: &[(String, u64)]) -> lucene_search::Result<Box<dyn CopyStream>> {
        if let Fault::RefuseOpen = self.fault {
            return Err(lucene_search::Error::IllegalState(
                "connection refused".to_string(),
            ));
        }
        Ok(Box::new(FaultyStream {
            inner: self.inner.open_files(requests)?,
            fault: self.fault,
            sent: 0,
        }))
    }
}

fn faulty(tag: &str, fault: Fault) -> Arc<Faulty> {
    Arc::new(Faulty {
        inner: java_primary(tag, &points()[tag]),
        fault,
    })
}

/// A replica in `dir` that copied and committed Java's `plain` point;
/// returns the point's primary generation.
fn committed_plain_replica(dir: &std::path::Path) -> i64 {
    let all = points();
    let point = &all["plain"];
    let transport = LocalReplicaHooks::new(java_primary("plain", point), None);
    let replica =
        ReplicaNode::new(1, Arc::new(FsDirectory::open(dir)), None, transport.clone()).unwrap();
    replica.start(point.primary_gen).unwrap();
    replica
        .new_nrt_point(point.primary_gen, point.version)
        .unwrap()
        .unwrap();
    wait_for("plain", || {
        !replica.is_copying() && replica.current_searching_version().unwrap() == point.version
    });
    replica.commit().unwrap();
    replica.close().unwrap();
    transport.close();
    point.primary_gen
}

#[test]
fn a_point_the_primary_cannot_serve_is_skipped() {
    let all = points();
    let point = &all["plain"];
    let source = faulty("plain", Fault::RefuseOpen);
    let tmp = TempDir::new("replicator-java-refused");
    let transport = LocalReplicaHooks::new(source.clone(), None);
    let replica = ReplicaNode::new(
        1,
        Arc::new(FsDirectory::open(tmp.path())),
        None,
        transport.clone(),
    )
    .unwrap();
    replica.start(point.primary_gen).unwrap();
    let before = replica.current_searching_version().unwrap();
    assert!(replica
        .new_nrt_point(point.primary_gen, point.version)
        .unwrap()
        .is_none());
    assert_eq!(replica.current_searching_version().unwrap(), before);
    assert!(
        *source.inner.released.lock().unwrap() >= 1,
        "the point is given back"
    );
    replica.close().unwrap();
    transport.close();
}

#[test]
fn a_local_file_the_replica_cannot_read_fails_the_new_point() {
    let all = points();
    let point = &all["plain"];
    let tmp = TempDir::new("replicator-java-unreadable");
    let transport = LocalReplicaHooks::new(java_primary("plain", point), None);
    let replica = ReplicaNode::new(
        1,
        Arc::new(FsDirectory::open(tmp.path())),
        None,
        transport.clone(),
    )
    .unwrap();
    replica.start(point.primary_gen).unwrap();
    // A directory where one of the point's files would be.
    std::fs::create_dir(tmp.path().join("_1.fdt")).unwrap();
    assert!(replica
        .new_nrt_point(point.primary_gen, point.version)
        .is_err());
    let _ = replica.close();
    transport.close();
}

#[test]
fn a_broken_sync_on_startup_fails_the_start() {
    let tmp = TempDir::new("replicator-java-startup");
    let primary_gen = committed_plain_replica(tmp.path());
    // The primary changed while the replica was down, and its stream breaks.
    let transport = LocalReplicaHooks::new(faulty("merged", Fault::BreakAfter(64)), None);
    let replica = ReplicaNode::new(
        1,
        Arc::new(FsDirectory::open(tmp.path())),
        None,
        transport.clone(),
    )
    .unwrap();
    let err = replica.start(primary_gen + 1).unwrap_err();
    assert!(err.to_string().contains("connection reset"), "{err}");
    transport.close();
}

#[test]
fn a_commit_with_an_unparsable_primary_gen_is_refused() {
    let tmp = TempDir::new("replicator-java-bad-gen");
    committed_plain_replica(tmp.path());
    let dir = FsDirectory::open(tmp.path());
    let mut infos = segment_infos::read_latest(&dir).unwrap();
    infos.user_data = vec![(PRIMARY_GEN_KEY.to_string(), "x".to_string())];
    infos.generation += 1;
    segment_infos::write(&infos, &dir).unwrap();

    let transport = LocalReplicaHooks::new(java_primary("plain", &points()["plain"]), None);
    let replica = ReplicaNode::new(
        1,
        Arc::new(FsDirectory::open(tmp.path())),
        None,
        transport.clone(),
    )
    .unwrap();
    let err = replica.start(1).unwrap_err();
    assert!(
        err.to_string().contains(&format!("{PRIMARY_GEN_KEY}=x")),
        "{err}"
    );
    transport.close();
}

#[test]
fn closing_a_replica_cancels_the_copy_in_flight() {
    let all = points();
    let point = &all["plain"];
    let tmp = TempDir::new("replicator-java-close");
    let transport = LocalReplicaHooks::new(faulty("plain", Fault::Slow), None);
    let replica = ReplicaNode::new(
        1,
        Arc::new(FsDirectory::open(tmp.path())),
        None,
        transport.clone(),
    )
    .unwrap();
    replica.start(point.primary_gen).unwrap();
    let job = replica
        .new_nrt_point(point.primary_gen, point.version)
        .unwrap()
        .unwrap();
    assert!(replica.is_copying());
    replica.close().unwrap();
    transport.close();
    assert!(job.failed(), "the copy was cancelled");
    assert_ne!(
        replica.current_searching_version().ok(),
        Some(point.version)
    );
}
