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

SCRATCH_BYTES = 20 * 1024**3
HEAVY_PROFILES = {"tests", "release", "coverage"}
IMAGE = "rust:1.97.1-bookworm@sha256:0e2bcaef56d041a486784e54104a81aebe0da44bd03019bd70bc0401e42e4a97"
SETUP = r'''
set -euo pipefail
trap 'result=$?; set +e; printf "RANGE_EXIT status=%s memory_events\n" "$result"; cat /sys/fs/cgroup/memory.events; printf "RANGE_EXIT cgroup_memory_peak_bytes\n"; cat /sys/fs/cgroup/memory.peak; printf "RANGE_EXIT filesystem_bytes\n"; df -B1 /scratch; exit "$result"' EXIT
mkdir -p /scratch/cargo /scratch/target /scratch/tmp
cp -a /usr/local/rustup /scratch/rustup
rustup component add rustfmt clippy --toolchain 1.97.1
[ "$(cat /sys/fs/cgroup/memory.max)" = "$EXPECTED_MEMORY_BYTES" ]
[ "$(cat /sys/fs/cgroup/memory.swap.max)" = 0 ]
[ "$(cat /sys/fs/cgroup/pids.max)" = 256 ]
read -r quota period </sys/fs/cgroup/cpu.max
[ "$quota" = "$((2 * period))" ]
[ "$(git -c safe.directory=/src rev-parse HEAD)" = "$EXPECTED_SOURCE" ]
printf 'RANGE_FEEDBACK source=%s cpu=%s/%s memory=%s swap=0 pids=256\n' "$EXPECTED_SOURCE" "$quota" "$period" "$EXPECTED_MEMORY_BYTES"
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
'''

PROFILES = {
    "component": r'''rustup component add llvm-tools-preview --toolchain 1.97.1
python3 -B scripts/range-production-coverage.py --archive-probe
echo 'Diagnostic archive probe complete; full component qualification intentionally not run.'
exit 78
cargo check --locked -p lucene-store -p lucene-codecs --all-targets
cargo check --locked -p lucene-search --lib
cargo test --locked -p lucene-store --lib range_input
cargo test --locked -p lucene-codecs --test blocktree_fixtures range_terms_postings_and_filter_match_java_with_eviction_and_faults
cargo test --locked -p lucene-codecs --test doc_values_fixtures numeric_range_values_match_java_and_transport_errors_propagate
cargo test --locked -p lucene-codecs --test doc_values_varying_bpv_fixtures varying_width_range_block_boundaries_match_java
cargo test --locked -p lucene-codecs --test stored_fields_fixtures range_stored_chunks_match_java_without_resident_fdt_and_refuse_faults
cargo test --locked -p lucene-search --test range_directory_queries -- --nocapture
cargo clippy --locked -p lucene-store -p lucene-codecs --all-targets -- -D warnings
cargo clippy --locked -p lucene-search --lib -- -D warnings
cargo fmt --all -- --check
if [ ! -f /src/benchmarks/rust-runner/Cargo.lock ]; then
  cp -a /src /scratch/benchmark-source
  cargo generate-lockfile --manifest-path /scratch/benchmark-source/benchmarks/rust-runner/Cargo.toml
  cp /scratch/benchmark-source/benchmarks/rust-runner/Cargo.lock /scratch/resolved-benchmark-Cargo.lock
  sha256sum /scratch/resolved-benchmark-Cargo.lock
  echo 'Benchmark graph is not yet committed source; benchmark compilation remains unrun.'
  exit 78
fi
cargo check --locked --manifest-path benchmarks/rust-runner/Cargo.toml --all-targets
''',
    "checks": r"""
cargo fmt --all -- --check
cargo fmt --manifest-path benchmarks/rust-runner/Cargo.toml -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo clippy --locked -p lucene-store --lib -- -D warnings
python3 -B scripts/check-arith-allows.py
python3 -B scripts/check-port-invariants.py
python3 -B scripts/check-parity.py
cargo fetch --locked
python3 -B scripts/check-licences.py
python3 -B scripts/check-vendored-licences.py
RUSTDOCFLAGS="-D warnings -A rustdoc::private_intra_doc_links" cargo doc --locked --workspace --no-deps --document-private-items
""",
    "tests": r"""
cargo build --locked -p lucene-ffi
cargo test --locked --quiet --workspace --lib --bins --tests -- --test-threads=2
cargo test --locked --quiet --workspace --doc
""",
    "release": r"""
cargo build --locked --release -p lucene-ffi
cargo test --locked --release --quiet -p lucene-search --lib --bins --tests -- --test-threads=2
cargo test --locked --release --quiet -p lucene-search --doc
""",
    "provenance": r"""
python3 -B scripts/range-provenance.py
""",
    "coverage": r"""
rustup component add llvm-tools-preview --toolchain 1.97.1
cargo install --locked cargo-llvm-cov --version 0.8.7 --root /scratch/cov-tools
export PATH="/scratch/cov-tools/bin:$PATH"
python3 -B scripts/range-production-coverage.py
set +e
scripts/coverage.sh --locked --workspace --quiet -- --summary-only --fail-under-lines 95 > /scratch/tmp/coverage-summary.txt
coverage_status=$?
set -e
cat /scratch/tmp/coverage-summary.txt
cargo llvm-cov report --json --skip-functions --output-path /scratch/tmp/coverage.json
exit "$coverage_status"
""",
}



def prestop_cgroup(name):
    """Read the verified owned live container's host cgroup before disposal."""
    try:
        identity = subprocess.check_output(
            ["docker", "inspect", "--format", "{{.Id}} {{.State.Pid}}", name],
            text=True, timeout=5).split()
        if len(identity) != 2 or not re.fullmatch(r"[0-9a-f]{64}", identity[0]) or not identity[1].isdecimal():
            raise ValueError("invalid live container identity")
        container_id, pid = identity
        if int(pid) <= 0:
            raise ValueError("container has no live host process")
        with Path(f"/proc/{pid}/cgroup").open() as source:
            membership = source.read(4097)
        if len(membership) > 4096:
            raise ValueError("host cgroup membership limit")
        unified = [line[3:] for line in membership.splitlines() if line.startswith("0::/")]
        if len(unified) != 1:
            raise ValueError("missing unified host cgroup")
        root = Path("/sys/fs/cgroup").resolve()
        group = (root / unified[0].lstrip("/")).resolve()
        if not group.is_relative_to(root) or not any(
                part in {container_id, "docker-" + container_id + ".scope"} for part in group.parts):
            raise ValueError("host cgroup does not identify owned container")
        counters = {}
        for filename in ("memory.events", "memory.peak"):
            with (group / filename).open() as source:
                content = source.read(1025)
            if len(content) > 1024:
                raise ValueError("host cgroup counter limit")
            counters[filename] = content
        print(json.dumps({"phase": "prestop-host-cgroup", "available": True,
                          "counters": counters}), flush=True)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"phase": "prestop-host-cgroup", "available": False,
                          "reason": str(error)[:256]}), flush=True)


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
    if stat.f_blocks * stat.f_frsize > SCRATCH_BYTES:
        raise SystemExit("scratch exceeds declared 20 GiB")
    profile = os.environ.get("RANGE_PROFILE", "component")
    if profile not in PROFILES:
        raise SystemExit("unknown fixed qualification profile")
    # Ordinary/release cells exceeded the original 30-minute bound. Keep
    # their commands; coverage shares the prospective longer timing limit.
    deadline = 5400 if profile in HEAVY_PROFILES else 1800
    memory_bytes = (8 if profile in HEAVY_PROFILES else 4) * 1024**3
    command = SETUP + PROFILES[profile]
    identity = os.environ["GITHUB_RUN_ID"] + "-" + os.environ["GITHUB_RUN_ATTEMPT"]
    if not re.fullmatch(r"[0-9]+-[0-9]+", identity):
        raise SystemExit("invalid job identity")
    name = "lucene-range-" + identity
    print(json.dumps({"kind": "bounded-native-qualification", "profile": profile, "source": expected, "image": IMAGE,
                      "cpu": 2, "memory_bytes": memory_bytes, "swap_bytes": 0, "pids": 256,
                      "scratch_filesystem_bytes": stat.f_blocks * stat.f_frsize, "scratch_limit_bytes": SCRATCH_BYTES,
                      "deadline_seconds": deadline, "stdio_limit_bytes": 2 * 1024**2}), flush=True)
    argv = ["docker", "run", "--rm", "--name", name, "--label", "lucene.range.job=" + identity,
            "--label", "lucene.range.source=" + expected, "--cpus=2", "--memory=" + str(memory_bytes), "--memory-swap=" + str(memory_bytes),
            "--pids-limit=256", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--user", f"{os.getuid()}:{os.getgid()}", "--workdir=/src",
            "--mount", f"type=bind,src={source},dst=/src,readonly",
            "--mount", f"type=bind,src={scratch},dst=/scratch",
            "--env", "EXPECTED_SOURCE=" + expected, "--env", "EXPECTED_MEMORY_BYTES=" + str(memory_bytes),
            "--env", "RUSTUP_TOOLCHAIN=1.97.1",
            "--env", "CARGO_HOME=/scratch/cargo", "--env", "RUSTUP_HOME=/scratch/rustup", "--env", "CARGO_TARGET_DIR=/scratch/target",
            "--env", "TMPDIR=/scratch/tmp", "--env", "CARGO_BUILD_JOBS=2",
            "--env", "CARGO_INCREMENTAL=0", "--env", "CARGO_PROFILE_DEV_DEBUG=0",
            "--env", "CARGO_PROFILE_TEST_DEBUG=0", IMAGE, "bash", "-c", command]
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    started = time.monotonic()
    total = 0
    failure = None
    minimum_scratch_available = stat.f_bavail * stat.f_frsize
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        while selector.get_map():
            sample = os.statvfs(scratch)
            minimum_scratch_available = min(minimum_scratch_available, sample.f_bavail * sample.f_frsize)
            if time.monotonic() - started > deadline:
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
            process.wait(timeout=max(1, deadline - (time.monotonic() - started)))
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
            # Deadline removal bypasses the guest EXIT trap. Capture the
            # owned live cgroup first; missing receipts remain unknown.
            prestop_cgroup(name)
            subprocess.run(["docker", "rm", "--force", name], check=True, timeout=30)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        process.stdout.close()
    ending = os.statvfs(scratch)
    print(json.dumps({"source": expected, "profile": profile,
                      "minimum_sampled_scratch_available_bytes": minimum_scratch_available,
                      "ending_scratch_total_bytes": ending.f_blocks * ending.f_frsize,
                      "ending_scratch_available_bytes": ending.f_bavail * ending.f_frsize,
                      "ending_scratch_allocated_bytes": (ending.f_blocks - ending.f_bfree) * ending.f_frsize}))
    trace = scratch / "tmp" / "range-query-physical-trace.tsv"
    if trace.is_file():
        import hashlib
        if trace.stat().st_size > 1024**2 or trace.is_symlink():
            raise SystemExit("range trace exceeds artifact envelope")
        data = trace.read_bytes()
        destination = runner_temp / ("range-feedback-" + expected + "-trace.tsv")
        with destination.open("xb") as output:
            output.write(data)
        print(json.dumps({"physical_trace_sha256": hashlib.sha256(data).hexdigest(), "physical_trace_bytes": len(data)}))
    benchmark_lock = scratch / "resolved-benchmark-Cargo.lock"
    if benchmark_lock.is_file():
        if benchmark_lock.is_symlink() or benchmark_lock.stat().st_size > 1024**2:
            raise SystemExit("benchmark lock exceeds artifact envelope")
        destination = runner_temp / ("range-feedback-" + expected + "-benchmark.lock")
        with destination.open("xb") as output:
            output.write(benchmark_lock.read_bytes())
    coverage = scratch / "tmp" / "coverage.json"
    if coverage.is_file():
        if coverage.is_symlink() or coverage.stat().st_size > 64 * 1024**2:
            raise SystemExit("coverage receipt exceeds artifact envelope")
        destination = runner_temp / ("range-feedback-" + expected + "-coverage.json")
        with destination.open("xb") as output:
            output.write(coverage.read_bytes())
    for suffix, limit in (("production-coverage", 64 * 1024**2), ("production-coverage-inventory", 4 * 1024**2)):
        artifact = scratch / "tmp" / (suffix + ".json")
        if artifact.is_file():
            if artifact.is_symlink() or artifact.stat().st_size > limit:
                raise SystemExit("production coverage artifact exceeds envelope")
            destination = runner_temp / ("range-feedback-" + expected + "-" + suffix + ".json")
            with destination.open("xb") as output:
                output.write(artifact.read_bytes())
    archive = scratch / "tmp" / "production-archive-inventory.json"
    if archive.is_file():
        if archive.is_symlink() or archive.stat().st_size > 4 * 1024**2:
            raise SystemExit("production archive receipt exceeds envelope")
        destination = runner_temp / ("range-feedback-" + expected + "-production-archive.json")
        with destination.open("xb") as output:
            output.write(archive.read_bytes())
    provenance = scratch / "tmp" / "provenance-receipt.json"
    if provenance.is_file():
        if provenance.is_symlink() or provenance.stat().st_size > 1024**2:
            raise SystemExit("provenance receipt exceeds envelope")
        destination = runner_temp / ("range-feedback-" + expected + "-provenance.json")
        with destination.open("xb") as output:
            output.write(provenance.read_bytes())
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
