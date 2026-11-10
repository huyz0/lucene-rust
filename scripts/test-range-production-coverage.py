#!/usr/bin/env python3
"""Archive admission/refusal controls; no compiler or coverage percentage mock.

Metadata header shape comes from native probe42828317246/artifact13771522050,
JSON SHA256 b79c1e8fafec19ae707091dbaeb376dbb765f776cb4900a9108c5a863c20c893.
"""
import runpy
from pathlib import Path
import unittest

classify = runpy.run_path(str(Path(__file__).with_name("range-production-coverage.py")))["classify_member"]

# Actual native metadata wrapper layout, with the large metadata size left
# intact. Refusal mutations below test sections, never source coverage hits.
HEADER = """
/scratch/production-maps/lucene-codecs-nonobject-0: file format elf64-x86-64
architecture: x86_64
start address: 0x0000000000000000

Sections:
Idx Name               Size     VMA              Type
  0                    00000000 0000000000000000
  1 .note.gnu.property 00000020 0000000000000000
  2 .rmeta             0090952a 0000000000000000
  3 .symtab            00000018 0000000000000000
  4 .strtab            00000001 0000000000000000
  5 .shstrtab          00000035 0000000000000000
"""


def extra(section, kind="DATA"):
    return HEADER + f"  6 {section} 00000020 0000000000000000 {kind}\n"


class ArchiveAdmission(unittest.TestCase):
    def test_observed_metadata_wrappers_without_maps(self):
        for member, headers in (("lib.rmeta", HEADER),
                                ("lib.rmeta-link", HEADER.replace(".rmeta ", ".rmeta-link "))):
            with self.subTest(member=member):
                self.assertEqual(classify(member, headers), (False, "metadata"))

    def test_metadata_filename_cannot_hide_native_maps(self):
        for member in ("lib.rmeta", "lib.rmeta-link"):
            with self.subTest(member=member):
                self.assertEqual(classify(member, extra("__llvm_covmap")), (True, "metadata"))

    def test_regular_object_map_is_included(self):
        self.assertEqual(classify("crate.rcgu.o", extra("__llvm_covfun")), (True, "code-object"))

    def test_filename_token_is_not_a_native_map(self):
        misleading = HEADER.replace("lucene-codecs-nonobject-0", "__llvm_covmap.rcgu.o")
        self.assertEqual(classify("__llvm_covmap.rcgu.o", misleading), (False, "code-object"))

    def test_unknown_member_refuses_even_with_native_maps(self):
        with self.assertRaisesRegex(ValueError, "unexpected nonobject"):
            classify("unknown.metadata", extra("__llvm_covmap"))

    def test_unmapped_metadata_with_code_or_unknown_section_refuses(self):
        for section, kind in ((".text", "TEXT"), (".unknown", "DATA")):
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, "metadata object sections"):
                classify("lib.rmeta", extra(section, kind))

    def test_missing_or_executable_metadata_payload_refuses(self):
        for headers in (HEADER.replace(".rmeta ", ".rmeta-link "),
                        HEADER.replace("0090952a 0000000000000000", "0090952a 0000000000000000 TEXT")):
            with self.assertRaisesRegex(ValueError, "metadata object sections"):
                classify("lib.rmeta", headers)

    def test_unknown_coverage_section_refuses(self):
        with self.assertRaisesRegex(ValueError, "unknown native coverage section"):
            classify("crate.o", extra("__llvm_covfuture"))

    def test_malformed_or_unexpected_native_object_refuses(self):
        for headers in (HEADER.replace("elf64-x86-64", "not-an-object"),
                        HEADER.replace("  5 .shstrtab", "  7 .shstrtab"),
                        HEADER + "not a section row\n"):
            with self.assertRaises(ValueError):
                classify("lib.rmeta", headers)

    def test_member_path_and_header_allocation_bounds_refuse(self):
        with self.assertRaisesRegex(ValueError, "invalid production archive member"):
            classify("../lib.rmeta", HEADER)
        with self.assertRaisesRegex(ValueError, "byte limit"):
            classify("lib.rmeta", HEADER + " " * 1024**2)


if __name__ == "__main__":
    unittest.main()
