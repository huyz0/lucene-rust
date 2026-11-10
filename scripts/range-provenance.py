#!/usr/bin/env python3
"""Fixed hosted provenance/gate self-controls, under range-feedback's cgroup."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import sys
import tarfile
import urllib.request

SCRATCH = Path("/scratch")


def download(url, path, algorithm, expected, limit):
    digest = hashlib.new(algorithm)
    total = 0
    with urllib.request.urlopen(url, timeout=30) as response, path.open("xb") as output:
        while chunk := response.read(65536):
            total += len(chunk)
            if total > limit:
                raise ValueError("provenance download envelope")
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("provenance artifact checksum mismatch")
    return total


def main():
    source = os.environ.get("EXPECTED_SOURCE", "")
    if sys.platform != "linux" or Path.cwd() != Path("/src") or not re.fullmatch(r"[0-9a-f]{40}", source):
        raise SystemExit("requires the fixed hosted range cgroup")
    pins_path = Path("scripts/range-provenance-pins.json")
    pins = json.loads(pins_path.read_text())
    assert pins["schema_version"] == 1 and pins["lucene_version"] == "10.5.2"
    jars = SCRATCH / "lucene-jars"
    jars.mkdir()
    receipts = []
    total = 0
    for pin in pins["jars"]:
        artifact = pin["artifact"]
        if not re.fullmatch(r"lucene-[a-z0-9-]+", artifact):
            raise ValueError("invalid pinned inventory artifact")
        destination = jars / (artifact + "-10.5.2.jar")
        size = download(pin["url"], destination, "sha512", pin["sha512"], min(64 * 1024**2, 512 * 1024**2 - total))
        total += size
        receipts.append({"module": pin["module"], "bytes": size, "sha512": pin["sha512"]})
    java_source = SCRATCH / "lucene-10.5.2"
    subprocess.run(["git", "init", "--quiet", str(java_source)], check=True)
    subprocess.run(["git", "-C", str(java_source), "remote", "add", "origin", pins["lucene_source"]["repository"]], check=True)
    subprocess.run(["git", "-C", str(java_source), "-c", "protocol.version=2", "fetch", "--quiet", "--depth=1", "--no-tags", "origin", pins["lucene_source"]["commit"]], check=True)
    subprocess.run(["git", "-C", str(java_source), "checkout", "--quiet", "--detach", "FETCH_HEAD"], check=True)
    commit = subprocess.check_output(["git", "-C", str(java_source), "rev-parse", "HEAD"], text=True).strip()
    if commit != pins["lucene_source"]["commit"] or subprocess.check_output(["git", "-C", str(java_source), "status", "--porcelain"]):
        raise ValueError("Java source is not the exact clean pin")
    env = os.environ.copy()
    env["JARS"] = str(jars)
    env["LUCENE_VERSION"] = "10.5.2"
    env["LUCENE_JAVA_SRC"] = str(java_source)
    subprocess.run([sys.executable, "-B", "scripts/check-java-refs.py", "--java", str(java_source)], env=env, check=True)
    for pin in pins["jars"]:
        subprocess.run([sys.executable, "-B", "scripts/check-port-inventory.py", "--module", pin["module"], "--require-jar"], env=env, check=True)
    subprocess.run([sys.executable, "-B", "scripts/test-bench-micro-report.py"], env=env, check=True)
    jdk_pin = pins["jdk"]
    archive = SCRATCH / "jdk.tar.gz"
    size = download(jdk_pin["url"], archive, "sha256", jdk_pin["sha256"], jdk_pin["bytes"])
    if size != jdk_pin["bytes"]:
        raise ValueError("JDK length mismatch")
    # The checksum authenticates a fixed official archive. Still refuse path,
    # entry/count/expanded-byte surprises before handing it to the native tar.
    count = expanded = 0
    prefix = None
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            count += 1
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError("JDK archive path")
            prefix = prefix or path.parts[0]
            if path.parts[0] != prefix or count > 50000:
                raise ValueError("JDK archive identity/count")
            if member.isfile():
                expanded += member.size
                if expanded > 1024**3:
                    raise ValueError("JDK expansion envelope")
            elif member.issym() or member.islnk():
                target = member.linkname if member.islnk() else posixpath.join(posixpath.dirname(member.name), member.linkname)
                target = PurePosixPath(posixpath.normpath(target))
                if target.is_absolute() or not target.parts or target.parts[0] != prefix or ".." in target.parts:
                    raise ValueError("JDK archive link escape")
            elif not member.isdir():
                raise ValueError("JDK archive entry kind")
    jdk = SCRATCH / "jdk"
    jdk.mkdir()
    subprocess.run(["tar", "--extract", "--gzip", "--file", str(archive), "--directory", str(jdk), "--strip-components=1", "--no-same-owner", "--no-same-permissions"], check=True)
    env["JAVA_HOME"] = str(jdk)
    env["JAVA_TOOL_OPTIONS"] = "-Djava.io.tmpdir=/scratch/tmp"
    env["PATH"] = str(jdk / "bin") + os.pathsep + env["PATH"]
    subprocess.run([str(jdk / "bin" / "java"), "-version"], env=env, check=True)
    subprocess.run([str(jdk / "bin" / "javac"), "-version"], env=env, check=True)
    subprocess.run([sys.executable, "-B", "scripts/test-test-framework-fixture-selection.py"], env=env, check=True)
    receipt = {"source": source, "lucene_source": commit, "jar_receipts": receipts,
               "pin_manifest_sha256": hashlib.sha256(pins_path.read_bytes()).hexdigest(),
               "jdk": jdk_pin, "jdk_expanded_bytes": expanded, "jdk_entries": count,
               "checks": ["check-java-refs default audited scope (real pinned tree)", "check-port-inventory all 32 modules --require-jar", "test-bench-micro-report", "test-test-framework-fixture-selection (real Java tools)"],
               "fixture_regeneration": "not run; committed fixture bytes unchanged"}
    (SCRATCH / "tmp" / "provenance-receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
    print(json.dumps({"provenance": "passed", "lucene_source": commit, "inventory_modules": len(receipts), "jar_bytes": total, "jdk_bytes": size}))


if __name__ == "__main__":
    main()
