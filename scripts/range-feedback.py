#!/usr/bin/env python3
"""Fixed hosted compiler feedback for the partial range-input branch.

Not a release gate or a replacement for scripts/gate.sh. No local execution,
shared cache, generic command argument, image build, or global Docker cleanup.
"""
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import time

IMAGE = "rust:1.97.1-bookworm@sha256:0e2bcaef56d041a486784e54104a81aebe0da44bd03019bd70bc0401e42e4a97"
COMMAND = r'''
set -euo pipefail
mkdir -p /scratch/cargo /scratch/target /scratch/tmp
[ "$(cat /sys/fs/cgroup/memory.max)" = 4294967296 ]
[ "$(cat /sys/fs/cgroup/memory.swap.max)" = 0 ]
[ "$(cat /sys/fs/cgroup/pids.max)" = 256 ]
read -r quota period </sys/fs/cgroup/cpu.max
[ "$quota" = "$((2 * period))" ]
[ "$(git -c safe.directory=/src rev-parse HEAD)" = "$EXPECTED_SOURCE" ]
printf 'RANGE_FEEDBACK source=%s cpu=%s/%s memory=4294967296 swap=0 pids=256\n' "$EXPECTED_SOURCE" "$quota" "$period"
rustc --version --verbose
cargo --version
if [ ! -f /src/Cargo.lock ]; then
  cp -a /src /scratch/source
  cd /scratch/source
  cargo generate-lockfile
  test "$(stat -c %s Cargo.lock)" -le 1048576
  cp Cargo.lock /scratch/resolved-Cargo.lock
  sha256sum /scratch/resolved-Cargo.lock
  echo 'Resolved graph is not yet committed source; compilation remains unrun.'
  exit 78
fi
cargo check --locked -p lucene-store -p lucene-codecs --all-targets
cargo test --locked -p lucene-store --lib range_input
cargo test --locked -p lucene-codecs --test blocktree_fixtures range_terms_postings_and_filter_match_java_with_eviction_and_faults
cargo test --locked -p lucene-codecs --test doc_values_fixtures numeric_range_values_match_java_and_transport_errors_propagate
cargo test --locked -p lucene-codecs --test doc_values_varying_bpv_fixtures varying_width_range_block_boundaries_match_java
cargo test --locked -p lucene-codecs --test stored_fields_fixtures range_stored_chunks_match_java_without_resident_fdt_and_refuse_faults
cargo fmt --all -- --check
'''


def main():
    if os.environ.get("GITHUB_ACTIONS") != "true" or sys.platform != "linux":
        raise SystemExit("requires a separate GitHub Linux runner")
    if len(sys.argv) != 2:
        raise SystemExit("expected the job-owned mounted scratch directory")
    source = Path(__file__).resolve().parents[1]
    if source != Path(os.environ["GITHUB_WORKSPACE"]).resolve():
        raise SystemExit("wrong source workspace")
    expected = os.environ["GITHUB_SHA"]
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise SystemExit("invalid source identity")
    observed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if observed != expected or subprocess.check_output(["git", "status", "--porcelain"], cwd=source):
        raise SystemExit("source is not the exact clean push commit")
    scratch = Path(sys.argv[1]).resolve(strict=True)
    runner_temp = Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    if scratch.parent != runner_temp or not scratch.name.startswith("lucene-range-"):
        raise SystemExit("scratch is not job-local")
    mount = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE,TARGET", "--target", str(scratch)], text=True).split()
    if mount != ["ext4", str(scratch)]:
        raise SystemExit("scratch lacks its own ext4 filesystem")
    stat = os.statvfs(scratch)
    if stat.f_blocks * stat.f_frsize > 8 * 1024**3:
        raise SystemExit("scratch exceeds 8 GiB")
    identity = os.environ["GITHUB_RUN_ID"] + "-" + os.environ["GITHUB_RUN_ATTEMPT"]
    if not re.fullmatch(r"[0-9]+-[0-9]+", identity):
        raise SystemExit("invalid job identity")
    name = "lucene-range-" + identity
    print(json.dumps({"kind": "partial-compiler-feedback", "source": expected, "image": IMAGE,
                      "cpu": 2, "memory_bytes": 4 * 1024**3, "swap_bytes": 0, "pids": 256,
                      "scratch_filesystem_bytes": stat.f_blocks * stat.f_frsize,
                      "deadline_seconds": 1800, "stdio_limit_bytes": 2 * 1024**2}), flush=True)
    argv = ["docker", "run", "--rm", "--name", name, "--label", "lucene.range.job=" + identity,
            "--label", "lucene.range.source=" + expected, "--cpus=2", "--memory=4g", "--memory-swap=4g",
            "--pids-limit=256", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--user", f"{os.getuid()}:{os.getgid()}", "--workdir=/src",
            "--mount", f"type=bind,src={source},dst=/src,readonly",
            "--mount", f"type=bind,src={scratch},dst=/scratch",
            "--env", "EXPECTED_SOURCE=" + expected, "--env", "RUSTUP_TOOLCHAIN=1.97.1",
            "--env", "CARGO_HOME=/scratch/cargo", "--env", "CARGO_TARGET_DIR=/scratch/target",
            "--env", "TMPDIR=/scratch/tmp", "--env", "CARGO_BUILD_JOBS=2",
            "--env", "CARGO_INCREMENTAL=0", "--env", "CARGO_PROFILE_DEV_DEBUG=0",
            "--env", "CARGO_PROFILE_TEST_DEBUG=0", IMAGE, "bash", "-c", COMMAND]
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    started = time.monotonic()
    total = 0
    failure = None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        while selector.get_map():
            if time.monotonic() - started > 1800:
                failure = "deadline"
                break
            for key, _ in selector.select(timeout=1):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                remaining = 2 * 1024**2 - 4096 - total
                sys.stdout.buffer.write(chunk[:remaining])
                sys.stdout.buffer.flush()
                total += len(chunk)
                if len(chunk) > remaining:
                    failure = "stdio-limit"
                    break
            if failure:
                break
        if not failure:
            process.wait(timeout=max(1, 1800 - (time.monotonic() - started)))
    except subprocess.TimeoutExpired:
        failure = "deadline"
    finally:
        selector.close()
        # Only remove the uniquely named container after verifying both labels.
        result = subprocess.run(["docker", "inspect", "--format", "{{json .Config.Labels}}", name],
                                capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            labels = json.loads(result.stdout)
            if labels.get("lucene.range.job") != identity or labels.get("lucene.range.source") != expected:
                raise SystemExit("refusing cleanup of foreign container")
            subprocess.run(["docker", "rm", "--force", name], check=True, timeout=30)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        process.stdout.close()
    resolved = scratch / "resolved-Cargo.lock"
    if resolved.is_file():
        import hashlib
        if resolved.stat().st_size > 1024**2:
            raise SystemExit("resolved lock exceeds artifact bound")
        data = resolved.read_bytes()
        destination = runner_temp / ("range-feedback-" + expected + ".lock")
        with destination.open("xb") as output:
            output.write(data)
        print(json.dumps({"resolved_lock_sha256": hashlib.sha256(data).hexdigest(),
                          "resolved_lock_bytes": len(data), "compiled": False}))
    print(json.dumps({"source": expected, "exit_code": process.returncode, "refusal": failure,
                      "observed_stdio_bytes": total, "elapsed_seconds": round(time.monotonic() - started, 3)}))
    return 1 if failure else process.returncode


if __name__ == "__main__":
    sys.exit(main())
