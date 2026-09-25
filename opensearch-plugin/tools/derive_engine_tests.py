#!/usr/bin/env python3
"""Derives RustEngineTests -- OpenSearch's own InternalEngineTests, run on RustEngine.

The engine tests are OpenSearch's specification of an engine's contracts: versioning and
optimistic concurrency, sequence numbers and checkpoints, translog recovery, history, commits.
RustEngine is InternalEngine with the writer swapped (derive_engine.py), so the same tests apply
to it, rewritten to construct it: EngineTestCase and InternalTestEngine (from the test framework's
sources) and InternalEngineTests (from the server's test sources, pinned by tag) are copied into
the plugin's package with InternalEngine replaced by RustEngine.

Tests that exercise what the Rust engine does not have -- Lucene's IndexWriter itself (a custom
IndexWriter or merge policy injected to fail, IndexWriter infoStream, merge scheduling), or index
configurations it refuses (index sorting, compound files) -- are marked @Ignore with the reason,
listed in SKIPPED below: that list is the unsupported matrix, and nothing else is skipped.

Usage: derive_engine_tests.py <dir holding InternalEngineTests.java> <test framework sources> <out dir>
  curl -O https://raw.githubusercontent.com/opensearch-project/OpenSearch/3.8.0/server/src/test/java/org/opensearch/index/engine/InternalEngineTests.java
  curl -O https://repo1.maven.org/maven2/org/opensearch/test/framework/3.8.0/framework-3.8.0-sources.jar
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from derive_engine import PACKAGE, Patcher, relocate  # noqa: E402

NOTICE = """/*
 * Derived from OpenSearch 3.8.0 by opensearch-plugin/tools/derive_engine_tests.py -- do not edit;
 * change the script and re-run it. The original notice follows.
 */
"""

# Test name -> why it cannot apply to the Rust engine. Keep each reason specific.
SKIPPED: dict[str, str] = {
    "testLookupVersionWithPrunedAwayIds": "builds a Lucene IndexWriter with OpenSearch's PrunePostingsMergePolicy by hand",
}


def filesystem_stores(text: str) -> str:
    """The Rust writer works on files: every test store is a filesystem directory (still
    LuceneTestCase's checking wrapper), never an in-memory one."""
    return text.replace("newDirectory()", "newFSDirectory(createTempDir())")


def rename(text: str) -> str:
    text = re.sub(r"\bInternalEngineTests\b", "RustEngineTests", text)
    text = re.sub(r"\bInternalTestEngine\b", "RustTestEngine", text)
    text = re.sub(r"\bEngineTestCase\b", "RustEngineTestCase", text)
    text = re.sub(r"\bInternalEngine\b", "RustEngine", text)
    return text


def drop_create_writer_overrides(p: Patcher) -> None:
    """The anonymous engines' `createWriter` overrides: the Rust engine has no IndexWriter."""
    pattern = re.compile(r"\n( *)@Override\n\s*IndexWriter createWriter\(Directory directory, IndexWriterConfig iwc\) throws IOException \{")
    while True:
        m = pattern.search(p.text)
        if not m:
            return
        start, i = m.start() + 1, m.end() - 1
        depth, j = 0, p.text.index("{", i)
        while True:
            c = p.text[j]
            depth += (c == "{") - (c == "}")
            if depth == 0:
                break
            j += 1
        end = p.text.index("\n", j) + 1
        p.text = p.text[:start] + p.text[end:]


def derive_test_case(src: str) -> str:
    p = Patcher(filesystem_stores(reach_package_private(rename(relocate(src)))))
    drop_create_writer_overrides(p)
    # The Rust engine writes the default codec only (RustEngineSupport.checkSupported); the
    # framework would otherwise pick Lucene's randomized test codec.
    p.regex(
        r"        String name = Codec\.getDefault\(\)\.getName\(\);\n.*?            codecName = \"default\";\n        \}\n",
        "        codecName = \"default\";\n",
    )
    # The plugin loads liblucene_ffi in its constructor; the tests have no plugin.
    p.sub(
        "public abstract class RustEngineTestCase extends OpenSearchTestCase {\n",
        "public abstract class RustEngineTestCase extends OpenSearchTestCase {\n"
        "    static {\n"
        "        org.lucenerust.opensearch.NativeLibrary.load(java.nio.file.Path.of(\".\"));\n"
        "    }\n\n",
    )
    return NOTICE + p.text


def derive_test_engine(src: str) -> str:
    return NOTICE + rename(relocate(src))


# Receivers of the protected/package-private members EngineTestAccess exposes: an identifier, a
# call, or a parenthesized cast.
RECEIVER = r"(\(\([A-Za-z.]+\) [A-Za-z_]\w*\)|[A-Za-z_][\w.]*(?:\(\))?)"


def reach_package_private(text: str) -> str:
    """Rewrites what InternalEngineTests reached from inside org.opensearch.index.engine --
    on code lines only, and never a package qualifier (`org.apache.lucene.store.Directory`)."""
    lines = text.split("\n")
    for n, line in enumerate(lines):
        if not line.startswith("import "):
            lines[n] = reach_line(line)
    return "\n".join(lines)


def reach_line(text: str) -> str:
    for field in ("isClosed", "failedEngine", "engineConfig", "store", "writeLock"):
        text = re.sub(RECEIVER + r"\." + field + r"\b(?!\()(?!\.[A-Z])", rf"EngineTestAccess.{field}(\1)", text)
    text = re.sub(RECEIVER + r"\.getMaxSeqNoFromSearcher\(", r"EngineTestAccess.getMaxSeqNoFromSearcher(\1, ", text)
    text = re.sub(RECEIVER + r"\.id\(\)", r"EngineTestAccess.id(\1)", text)
    text = re.sub(RECEIVER + r"\.appliedOperations\(\)", r"EngineTestAccess.appliedOperations(\1)", text)
    text = text.replace("new PrunePostingsMergePolicy(", "EngineTestAccess.newPrunePostingsMergePolicy(")
    # The package-private four-argument Engine.Index constructor.
    text = re.sub(
        r"new Engine\.Index\((newUid\([^()]*\)), (primaryTerm\.get\(\)), (\w+), (Versions\.\w+|\d+)\)",
        r"EngineTestAccess.newIndex(\1, \2, \3, \4)",
        text,
    )
    return text


def derive_tests(src: str) -> str:
    p = Patcher(filesystem_stores(reach_package_private(rename(relocate(src)))))
    for name, reason in SKIPPED.items():
        p.sub(f"    public void {name}(", f'    @org.junit.Ignore("{reason}")\n    public void {name}(')
    return NOTICE + p.text


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    tests, framework, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    engine = framework / "org/opensearch/index/engine"
    out.mkdir(parents=True, exist_ok=True)
    (out / "RustEngineTestCase.java").write_text(derive_test_case((engine / "EngineTestCase.java").read_text()))
    (out / "RustTestEngine.java").write_text(derive_test_engine((engine / "InternalTestEngine.java").read_text()))
    (out / "RustEngineTests.java").write_text(derive_tests((tests / "InternalEngineTests.java").read_text()))
    # Server test helpers, verbatim in their own packages (test code shares one class loader).
    root = out
    for _ in PACKAGE.split("."):
        root = root.parent
    for name, package in (
        ("EngineSearcherTotalHitsMatcher", "org/opensearch/index/engine"),
        ("TestTranslog", "org/opensearch/index/translog"),
        ("SnapshotMatchers", "org/opensearch/index/translog"),
    ):
        (root / package).mkdir(parents=True, exist_ok=True)
        (root / package / f"{name}.java").write_text((tests / f"{name}.java").read_text())


if __name__ == "__main__":
    main()
