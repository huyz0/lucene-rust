#!/usr/bin/env python3
"""Exercise fixture selection with the production script and real Java tools.

The tiny generators need no Lucene jars. An invalid unrelated generator must
not block --only, while --check must still compile and execute every generator.
This checks dispatch and Java source dependency discovery, not Lucene fixture
determinism; the real generators' --check remains responsible for that.
"""
import pathlib
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("gen-test-framework-fixtures.sh")
JAR_HELPER = """#!/usr/bin/env bash
TEST_FRAMEWORK_DEPS=()
LUCENE_FIXTURE_JVM_OPTS=()
lucene_classpath() { printf ''; }
thirdparty_classpath() { printf ''; }
"""
DEPENDENCY = """import java.nio.file.*;
public class GenDependency {
  static String value() { return "selected via dependency\\n"; }
  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]); Files.createDirectories(out);
    Files.writeString(out.resolve("dependency.txt"), "dependency ran\\n");
  }
}
"""
SELECTED = """import java.nio.file.*;
public class GenSelected {
  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]); Files.createDirectories(out);
    Files.writeString(out.resolve("selected.txt"), GenDependency.value());
  }
}
"""
OTHER = """import java.nio.file.*;
public class GenOther {
  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]); Files.createDirectories(out);
    Files.writeString(out.resolve("other.txt"), "other ran\\n");
  }
}
"""
EXPECTED = {
    "dependency.txt": "dependency ran\n",
    "selected.txt": "selected via dependency\n",
    "other.txt": "other ran\n",
}


class FixtureSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for tool in ("bash", "javac", "java"):
            if shutil.which(tool) is None:
                raise RuntimeError(f"fixture selection tests require {tool} on PATH")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lucene fixture selection ")
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        self.script = scripts / SCRIPT.name
        shutil.copyfile(SCRIPT, self.script)
        (scripts / "lib-lucene-jars.sh").write_text(JAR_HELPER, encoding="utf-8")
        self.source = self.root / "fixtures/src/testframework"
        self.source.mkdir(parents=True)
        for name, source in {
            "GenDependency.java": DEPENDENCY,
            "GenSelected.java": SELECTED,
            "GenOther.java": OTHER,
            "GenBroken.java": "public class GenBroken { invalid Java here }\n",
        }.items():
            (self.source / name).write_text(source, encoding="utf-8")
        self.data = self.root / "fixtures/testframework/data"
        self.data.mkdir(parents=True)

    def run_script(self, *arguments):
        return subprocess.run(
            ["bash", str(self.script), *arguments],
            cwd=self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )

    @staticmethod
    def diagnostics(result):
        return f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"

    def fixtures(self):
        return {
            path.relative_to(self.data).as_posix(): path.read_text(encoding="utf-8")
            for path in self.data.rglob("*") if path.is_file()
        }

    def prepare_check(self):
        (self.source / "GenBroken.java").unlink()
        for name, value in EXPECTED.items():
            (self.data / name).write_text(value, encoding="utf-8")

    def test_only_resolves_source_dependency_and_ignores_broken_generator(self):
        result = self.run_script("--only", "GenSelected")
        self.assertEqual(result.returncode, 0, self.diagnostics(result))
        self.assertEqual(self.fixtures(), {"selected.txt": EXPECTED["selected.txt"]})
        # A fresh javac output directory is created for every run. Repeating
        # selection must produce the same fixture without executing the helper.
        first = self.fixtures()
        result = self.run_script("--only", "Selected")
        self.assertEqual(result.returncode, 0, self.diagnostics(result))
        self.assertEqual(self.fixtures(), first)

    def test_repeated_only_executes_both_selected_generators(self):
        result = self.run_script("--only", "Selected", "--only", "GenOther")
        self.assertEqual(result.returncode, 0, self.diagnostics(result))
        self.assertEqual(self.fixtures(), {
            "selected.txt": EXPECTED["selected.txt"],
            "other.txt": EXPECTED["other.txt"],
        })

    def test_term_vectors_routing_resolves_non_generator_helper_with_assertions(self):
        # A selected routing generator depends on a plain shared Java source,
        # also compiled by its benchmark. It must not become a generator entry.
        (self.source / "TermVectorsRoutingSupport.java").write_text(
            'public class TermVectorsRoutingSupport { '
            'static String value() { return "native helper"; } }\n', encoding="utf-8")
        (self.source / "GenTermVectorsWriterRouting.java").write_text(
            'import java.nio.file.*; public class GenTermVectorsWriterRouting { '
            'public static void main(String[] args) throws Exception { '
            'if (!GenTermVectorsWriterRouting.class.desiredAssertionStatus()) '
            'throw new AssertionError("missing -ea"); '
            'Path out = Path.of(args[0]); Files.createDirectories(out); '
            'Files.writeString(out.resolve("term-vectors.txt"), '
            'TermVectorsRoutingSupport.value()); } }\n', encoding="utf-8")
        result = self.run_script("--only", "GenTermVectorsWriterRouting")
        self.assertEqual(result.returncode, 0, self.diagnostics(result))
        self.assertEqual(self.fixtures(), {"term-vectors.txt": "native helper"})
        listed = self.run_script("--list")
        self.assertEqual(listed.returncode, 0, self.diagnostics(listed))
        self.assertNotIn("GenTermVectorsRoutingSupport", listed.stdout.splitlines())

    def test_check_compiles_and_executes_every_generator(self):
        self.prepare_check()
        result = self.run_script("--check")
        self.assertEqual(result.returncode, 0, self.diagnostics(result))
        self.assertIn("ok (3 generators)", result.stdout)
        self.assertEqual(self.fixtures(), EXPECTED)

    def test_check_rejects_any_broken_generator(self):
        result = self.run_script("--check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GenBroken.java", result.stderr, self.diagnostics(result))
        self.assertEqual(self.fixtures(), {})

    def test_check_rejects_fixture_drift_without_rewriting_it(self):
        self.prepare_check()
        (self.data / "other.txt").write_text("stale\n", encoding="utf-8")
        before = self.fixtures()
        result = self.run_script("--check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("differs from what Lucene", result.stdout, self.diagnostics(result))
        self.assertEqual(self.fixtures(), before)

    def test_list_and_invalid_arguments_do_not_reach_broken_compilation(self):
        result = self.run_script("--list")
        self.assertEqual(result.returncode, 0, self.diagnostics(result))
        self.assertEqual(result.stdout.splitlines(), [
            "GenBroken", "GenDependency", "GenOther", "GenSelected",
        ])
        for arguments, message in [
            (("--unknown",), "unknown argument"),
            (("--only", "Missing"), "no generator GenMissing"),
            (("--check", "--only", "Selected"), "cannot be combined with --only"),
        ]:
            with self.subTest(arguments=arguments):
                result = self.run_script(*arguments)
                self.assertEqual(result.returncode, 2, self.diagnostics(result))
                self.assertIn(message, result.stderr)
                self.assertNotIn("GenBroken.java:", result.stderr)
        self.assertEqual(self.fixtures(), {})


if __name__ == "__main__":
    unittest.main(verbosity=0)
