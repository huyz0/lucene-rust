//! Cross-engine write-lock test: Lucene 10.5.0's `NativeFSLockFactory` (and
//! a real Java `IndexWriter`) and this port's `NativeFsLockFactory` exclude
//! each other on one directory, in both directions.
//!
//! The Java half is `fixtures/src/VerifyNativeLock.java`, run through the
//! source launcher. The test needs `java` on the PATH and the
//! `lucene-core-10.5.0.jar`, found through `LUCENE_CORE_JAR`, a `$JARS` /
//! `$LUCENE_JARS` directory (the container), `fixtures/.jars`, or the local
//! Gradle cache; without them it prints why and passes vacuously.
//! `scripts/verify-write-path.sh` runs it with the jar it resolved, so the
//! CI job that has Java runs it for real.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use lucene_store::{Directory, Error, FsDirectory};
use lucene_util::test_support::TempDir;

const JAR: &str = "lucene-core-10.5.0.jar";

fn find_jar() -> Option<PathBuf> {
    if let Some(jar) = std::env::var_os("LUCENE_CORE_JAR") {
        return Some(PathBuf::from(jar)).filter(|p| p.is_file());
    }
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut dirs: Vec<PathBuf> = ["JARS", "LUCENE_JARS"]
        .iter()
        .filter_map(std::env::var_os)
        .map(PathBuf::from)
        .collect();
    dirs.push(repo.join("fixtures/.jars"));
    if let Some(jar) = dirs.iter().map(|d| d.join(JAR)).find(|p| p.is_file()) {
        return Some(jar);
    }
    let gradle = PathBuf::from(std::env::var_os("HOME")?)
        .join(".gradle/caches/modules-2/files-2.1/org.apache.lucene/lucene-core/10.5.0");
    std::fs::read_dir(gradle)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path().join(JAR))
        .find(|p| p.is_file())
}

/// Lucene 10.5.0's classes are Java 21 bytecode (class file version 65).
const MIN_JAVA: u32 = 21;

/// The feature release of a `java -version` banner: `openjdk version
/// "17.0.12" ...` is 17, `"21"` is 21, and a pre-9 `"1.8.0_402"` is 1.
fn java_major(banner: &str) -> Option<u32> {
    let quoted = banner.split('"').nth(1)?;
    quoted.split(['.', '-', '+', '_']).next()?.parse().ok()
}

/// The major version of the `java` on the PATH, if there is one.
fn java_version() -> Option<u32> {
    let out = Command::new("java").arg("-version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    java_major(&String::from_utf8_lossy(&out.stderr))
}

#[test]
fn java_major_reads_the_version_banner() {
    let banner = |v: &str| format!("openjdk version \"{v}\" 2024-07-16\nOpenJDK Runtime\n");
    assert_eq!(java_major(&banner("17.0.12")), Some(17));
    assert_eq!(java_major(&banner("21")), Some(21));
    assert_eq!(java_major(&banner("25.0.4.1")), Some(25));
    assert_eq!(java_major(&banner("26-ea")), Some(26));
    assert_eq!(java_major(&banner("1.8.0_402")), Some(1));
    // A JAVA_TOOL_OPTIONS notice may come first; it carries no quotes.
    let noisy = format!("Picked up JAVA_TOOL_OPTIONS: -Dx=y\n{}", banner("21.0.8"));
    assert_eq!(java_major(&noisy), Some(21));
    assert_eq!(java_major("no version here"), None);
}

fn program() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/src/VerifyNativeLock.java")
}

fn java(jar: &Path, mode: &str, dir: &Path) -> Command {
    let mut cmd = Command::new("java");
    cmd.arg("-cp").arg(jar).arg(program()).arg(mode).arg(dir);
    cmd
}

/// Runs the Java `try` mode: whether Java's lock factory and Java's
/// `IndexWriter` could take `write.lock`.
fn java_try(jar: &Path, dir: &Path) -> String {
    let out = java(jar, "try", dir).output().expect("run java");
    assert!(
        out.status.success(),
        "java try failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn rust_and_java_native_locks_exclude_each_other() {
    let Some(jar) = find_jar() else {
        eprintln!("native_lock_interop: skipped -- no {JAR} (set LUCENE_CORE_JAR)");
        return;
    };
    // A runner's default `java` may predate Lucene 10 (ubuntu-24.04's is 17)
    // and cannot compile the program against the jar. The CI job that must
    // run this for real (scripts/verify-write-path.sh, on JDK 21) fails on
    // the word "skipped", so skipping here cannot hide it there.
    match java_version() {
        None => {
            eprintln!("native_lock_interop: skipped -- no `java` on the PATH");
            return;
        }
        Some(v) if v < MIN_JAVA => {
            eprintln!(
                "native_lock_interop: skipped -- `java` is {v}, Lucene 10.5.0 needs {MIN_JAVA}"
            );
            return;
        }
        Some(_) => {}
    }
    let root = TempDir::new("native-lock-interop");
    let dir = FsDirectory::open(&root);

    // Rust holds the lock: Java can take neither the lock nor a writer.
    let lock = dir.obtain_lock("write.lock").unwrap();
    assert_eq!(java_try(&jar, &root), "lock HELD\nwriter HELD\n");

    // Released: Java takes both.
    lock.close().unwrap();
    assert_eq!(java_try(&jar, &root), "lock OBTAINED\nwriter OBTAINED\n");

    // Java holds the lock: Rust cannot take it until Java lets go.
    let mut child = java(&jar, "hold", &root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn java");
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    stdout.read_line(&mut line).unwrap();
    assert_eq!(line, "LOCKED\n");
    let err = dir.obtain_lock("write.lock").unwrap_err();
    assert!(
        matches!(&err, Error::LockObtainFailed(m) if m.contains("another program")),
        "{err}"
    );
    // A second directory instance fares no better.
    assert!(matches!(
        FsDirectory::open(&root).obtain_lock("write.lock"),
        Err(Error::LockObtainFailed(_))
    ));

    child.stdin.take().unwrap().write_all(b"release\n").unwrap();
    line.clear();
    stdout.read_line(&mut line).unwrap();
    assert_eq!(line, "RELEASED\n");
    assert!(child.wait().unwrap().success());

    let lock = dir.obtain_lock("write.lock").unwrap();
    lock.ensure_valid().unwrap();
}
