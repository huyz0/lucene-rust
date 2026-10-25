#!/usr/bin/env python3
"""Exercise the report's rejection paths using complete paired output files."""
import pathlib
import subprocess
import sys
import tempfile
import unittest


REPORT = pathlib.Path(__file__).with_name("bench-micro-report.py")
VALID = "case\tns\tops\nprobe\t12.5\t100\n#check\tprobe\tabcd\t100\n"


class ReportTests(unittest.TestCase):
    def run_report(self, replacements=None, required=True):
        with tempfile.TemporaryDirectory() as work:
            root = pathlib.Path(work)
            for engine in ("rust", "java"):
                for rep in (1, 2):
                    name = f"{engine}.{rep}.tsv"
                    (root / name).write_text((replacements or {}).get(name, VALID))
            args = [sys.executable, str(REPORT), str(root), "2"]
            if required:
                args += ["--require-check", "probe"]
            return subprocess.run(args, capture_output=True, text=True, check=False)

    def test_valid_pair(self):
        result = self.run_report()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 cases", result.stdout)

    def test_legacy_unchecked_pair(self):
        unchecked = "probe\t12.5\t100\n"
        replacements = {f"{engine}.{rep}.tsv": unchecked
                        for engine in ("rust", "java") for rep in (1, 2)}
        self.assertEqual(self.run_report(replacements, required=False).returncode, 0)
        result = self.run_report(replacements)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing result checks", result.stderr)

    def test_bad_repetitions_are_rejected(self):
        for label, value, message in [
            ("duplicate timing", VALID + "probe\t15\t100\n", "duplicate case"),
            ("duplicate check", VALID + "#check\tprobe\tabcd\t100\n", "duplicate result check"),
            ("missing check", "probe\t12.5\t100\n", "missing result checks"),
            ("changed digest", VALID.replace("abcd", "ffff"), "RESULT MISMATCH"),
            ("changed units", VALID.replace("abcd\t100", "abcd\t99"), "RESULT MISMATCH"),
            ("malformed check", VALID.replace("#check\tprobe", "#check probe"), "malformed result check"),
            ("empty check", VALID.replace("abcd", ""), "empty result check"),
            ("missing timing", "#check\tprobe\tabcd\t100\n", "timing cases differ"),
            ("extra timing", VALID + "other\t1\t100\n", "timing cases differ"),
            ("nonnumeric", VALID.replace("12.5", "bad"), "invalid numeric value"),
            ("zero", VALID.replace("12.5", "0"), "finite positive value"),
            ("negative", VALID.replace("12.5", "-1"), "finite positive value"),
            ("nan", VALID.replace("12.5", "NaN"), "finite positive value"),
            ("infinite", VALID.replace("12.5", "inf"), "finite positive value"),
        ]:
            with self.subTest(label=label):
                result = self.run_report({"java.2.tsv": value})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_checks_observed_once_are_required_everywhere(self):
        result = self.run_report({"rust.2.tsv": "probe\t12.5\t100\n"}, required=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing result checks", result.stderr)

    def test_zero_memory_is_valid(self):
        rows = VALID + "retained_bytes\t0\t1\n"
        replacements = {f"{engine}.{rep}.tsv": rows
                        for engine in ("rust", "java") for rep in (1, 2)}
        result = self.run_report(replacements)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("retained_bytes", result.stdout)
        self.assertIn("1 cases   median 1.00x   mean 1.00x", result.stdout)
        self.assertIn("noise floor 1.00x", result.stdout)
        self.assertNotIn("inf", result.stdout)

    def test_memory_only_report(self):
        replacements = {f"{engine}.{rep}.tsv": "retained_bytes\t0\t1\n"
                        for engine in ("rust", "java") for rep in (1, 2)}
        result = self.run_report(replacements, required=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("0 timing cases", result.stdout)
        self.assertNotIn("noise floor", result.stdout)


if __name__ == "__main__":
    unittest.main()
