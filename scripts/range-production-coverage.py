#!/usr/bin/env python3
"""Export native production maps for the three changed range-input libraries.

Producer only: percentages and source/floor policy belong to the review gate.
No local execution, source filtering, test-map denominator or new parser.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SCRATCH = Path("/scratch")
PACKAGES = ("lucene-store", "lucene-codecs", "lucene-search")


def output(argv, destination, limit, *, env=None):
    """Retain bounded stdout from one owned child; outer cgroup owns deadline."""
    total = 0
    with destination.open("xb") as target:
        child = subprocess.Popen(argv, stdout=subprocess.PIPE, env=env)
        try:
            while True:
                chunk = child.stdout.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise ValueError("native producer output limit")
                target.write(chunk)
            if child.wait() != 0:
                raise ValueError("native producer command failed: " + argv[0])
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            child.stdout.close()
    return total


def record(path):
    length = path.stat().st_size
    if length > 512 * 1024**2:
        raise ValueError("production record byte limit")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(65536):
            digest.update(chunk)
    return {"path": str(path), "bytes": length, "sha256": digest.hexdigest()}


def main():
    source = os.environ.get("EXPECTED_SOURCE", "")
    if sys.platform != "linux" or Path.cwd() != Path("/src") or not re.fullmatch(r"[0-9a-f]{40}", source):
        raise SystemExit("requires the fixed hosted Linux range cgroup")
    root = SCRATCH / "production-maps"
    root.mkdir()
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(SCRATCH / "production-target")
    env["RUSTFLAGS"] = "-C instrument-coverage"
    env["LLVM_PROFILE_FILE"] = str(root / "unused-%m-%p.profraw")
    artifacts = root / "compiler.jsonl"
    command = ["cargo", "build", "--locked", "--lib", "--message-format=json-render-diagnostics"]
    for package in PACKAGES:
        command.extend(["-p", package])
    output(command, artifacts, 16 * 1024**2, env=env)
    sysroot = subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip()
    host = subprocess.check_output(["rustc", "-vV"], text=True).split("host: ", 1)[1].splitlines()[0]
    llvm = Path(sysroot) / "lib" / "rustlib" / host / "bin"
    libraries = {}
    for line in artifacts.read_text().splitlines():
        event = json.loads(line)
        if event.get("reason") != "compiler-artifact":
            continue
        target = event["target"]
        name = target["name"].replace("_", "-")
        if name not in PACKAGES or target["src_path"] != f"/src/crates/{name}/src/lib.rs":
            continue
        if event["profile"]["test"] or target["kind"] != ["lib"]:
            raise ValueError("test/nonlibrary artifact cannot define production maps")
        files = [Path(p) for p in event["filenames"] if p.endswith(".rlib")]
        if len(files) != 1 or name in libraries:
            raise ValueError("missing/duplicate production library")
        artifact = files[0]
        if artifact.is_symlink() or not artifact.resolve().is_relative_to(SCRATCH / "production-target"):
            raise ValueError("production artifact escaped owned target")
        libraries[name] = artifact
    if set(libraries) != set(PACKAGES):
        raise ValueError("missing selected production library")
    objects, mapped = [], []
    for package, artifact in sorted(libraries.items()):
        inventory = root / (package + "-archive.txt")
        output([str(llvm / "llvm-ar"), "t", str(artifact)], inventory, 1024**2)
        members = inventory.read_text().splitlines()
        if not 0 < len(members) <= 10000 or len(set(members)) != len(members):
            raise ValueError("invalid production archive inventory")
        for member in members:
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,256}", member) or member.startswith("-"):
                raise ValueError("invalid production archive member")
            if not member.endswith(".o"):
                if member != "lib.rmeta":
                    raise ValueError("unexpected nonobject archive member")
                continue
            if len(objects) >= 10000:
                raise ValueError("production object count exhausted")
            obj = root / (str(len(objects)) + ".o")
            output([str(llvm / "llvm-ar"), "p", str(artifact), member], obj, 512 * 1024**2)
            sections = root / (str(len(objects)) + "-sections.txt")
            output([str(llvm / "llvm-objdump"), "--section-headers", str(obj)], sections, 1024**2)
            has_map = any(token in sections.read_text() for token in ("__llvm_covmap", "__llvm_covfun"))
            objects.append({**record(obj), "package": package, "member": member, "has_map": has_map})
            if has_map:
                mapped.append(str(obj))
    if not mapped or any(not any(o["package"] == p and o["has_map"] for o in objects) for p in PACKAGES):
        raise ValueError("selected production library has no native coverage map")
    destination = SCRATCH / "tmp" / "production-coverage.json"
    command = [str(llvm / "llvm-cov"), "export", "--empty-profile", "--num-threads=2", "--format=text", "--skip-functions", mapped[0]]
    for obj in mapped[1:]:
        command.extend(["--object", obj])
    output(command, destination, 64 * 1024**2)
    receipt = {"source": source, "kind": "native-empty-profile-production-library-maps",
               "packages": list(PACKAGES), "rustc": subprocess.check_output(["rustc", "-vV"], text=True),
               "libraries": {p: record(a) for p, a in libraries.items()}, "objects": objects,
               "compiler_inventory": record(artifacts), "native_export": record(destination),
               "source_filters": [], "floor_result": "not evaluated by producer"}
    (SCRATCH / "tmp" / "production-coverage-inventory.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
    print(json.dumps({"production_map_objects": len(mapped), "production_objects": len(objects),
                      "native_export": receipt["native_export"]}))


if __name__ == "__main__":
    main()
