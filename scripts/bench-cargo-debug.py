#!/usr/bin/env python3
"""Bounded, opt-in Cargo debug-info comparison; invoke inside docker-test.sh.

Run from the repository root:
  scripts/docker-test.sh python3 scripts/bench-cargo-debug.py \
    --output /work/target/cargo-debug-results

No adoption is performed. Only a disposable copy of lucene-util is compiled.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib

TIMING_HELPER = r'''
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

started = time.perf_counter()
result = subprocess.run(sys.argv[2:], check=False)
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
Path(sys.argv[1]).write_text(json.dumps({
    "wall_seconds": time.perf_counter() - started,
    "user_seconds": usage.ru_utime,
    "system_seconds": usage.ru_stime,
    "max_rss_kib": usage.ru_maxrss,
    "minor_page_faults": usage.ru_minflt,
    "major_page_faults": usage.ru_majflt,
    "returncode": result.returncode,
    "scope": "fresh helper RUSAGE_CHILDREN; maximum process RSS, not simultaneous tree RSS",
}, indent=2) + "\n", encoding="utf-8")
if result.returncode < 0:
    signum = -result.returncode
    if signum not in (signal.SIGKILL, signal.SIGSTOP):
        signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)
sys.exit(result.returncode)
'''


PROBES = r'''
#[cfg(test)]
mod cargo_debug_info_probe {
    #[test]
    fn assertions_and_overflow_remain_enabled() {
        assert!(cfg!(debug_assertions));
        let result = std::panic::catch_unwind(|| {
            let max = std::hint::black_box(u64::MAX);
            let one = std::hint::black_box(1_u64);
            std::hint::black_box(max + one);
        });
        assert!(result.is_err(), "test profile must retain overflow checks");
    }
    #[inline(never)]
    fn backtrace_leaf() { panic!("cargo-debug-info-probe"); }
    #[test]
    #[should_panic(expected = "cargo-debug-info-probe")]
    fn backtrace_has_source() { backtrace_leaf(); }
}
'''


def wait_for_command(process, timeout):
    """Reap every owned command before its scratch directory can be removed."""
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        # The new session does not inherit an interrupt sent to the parent.
        # Kill the complete owned group on interruption as well as timeout.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


def replace_list(text, key):
    pattern = rf"(?ms)^{re.escape(key)} = \[.*?^\]"
    text, count = re.subn(pattern, f'{key} = ["crates/lucene-util"]', text)
    if count != 1:
        raise RuntimeError(f"expected exactly one root workspace {key} list")
    return text


def directory_bytes(path):
    logical = allocated = 0
    seen_inodes = set()
    if path.exists():
        for root, _, files in os.walk(path):
            for filename in files:
                item = Path(root, filename)
                if item.is_symlink():
                    continue
                stat = item.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity in seen_inodes:
                    continue
                seen_inodes.add(identity)
                logical += stat.st_size
                allocated += getattr(stat, "st_blocks", 0) * 512
    return {"logical": logical, "allocated": allocated}


def compiler_commands(log):
    commands = []
    for line in log.splitlines():
        if "Running `" not in line or "--crate-name lucene_util " not in line:
            continue
        command = line.split("Running `", 1)[1].rsplit("`", 1)[0]
        words = shlex.split(command)
        codegen = []
        for index, word in enumerate(words):
            if word == "-C" and index + 1 < len(words):
                codegen.append(words[index + 1])
            elif word.startswith("-C"):
                codegen.append(word[2:])
        commands.append({"test": "--test" in words, "codegen": codegen,
                         "command": command})
    return commands



def prune_lockfile(original):
    """Retain the original util dependency closure; never choose new versions."""
    starts = list(re.finditer(r"(?m)^\[\[package\]\]\s*$", original))
    if not starts or re.search(r"(?m)^\[(?!\[)", original[starts[0].start():]):
        raise RuntimeError("unsupported lockfile structure; refusing lock rewriting")
    chunks = [original[m.start():starts[i + 1].start() if i + 1 < len(starts) else len(original)]
              for i, m in enumerate(starts)]
    packages = [tomllib.loads(chunk)["package"][0] for chunk in chunks]
    by_key = {(p["name"], p["version"], p.get("source")): p for p in packages}
    if len(by_key) != len(packages):
        raise RuntimeError("duplicate original lockfile package identity")

    def resolve(dependency):
        match = re.fullmatch(r"([^ ]+)(?: ([^ ()]+))?(?: \((.+)\))?", dependency)
        if match is None:
            raise RuntimeError(f"unsupported locked dependency: {dependency}")
        name, version, source = match.groups()
        candidates = [key for key in by_key if key[0] == name
                      and (version is None or key[1] == version)
                      and (source is None or key[2] == source)]
        if len(candidates) != 1:
            raise RuntimeError(f"ambiguous locked dependency: {dependency}")
        return candidates[0]

    roots = [key for key in by_key if key[0] == "lucene-util" and key[2] is None]
    if len(roots) != 1:
        raise RuntimeError("expected one locked lucene-util workspace package")
    reachable = set()
    pending = roots[:]
    while pending:
        key = pending.pop()
        if key in reachable:
            continue
        reachable.add(key)
        pending.extend(resolve(d) for d in by_key[key].get("dependencies", []))
    pruned = original[:starts[0].start()] + "".join(
        chunk for chunk, package in zip(chunks, packages)
        if (package["name"], package["version"], package.get("source")) in reachable)
    parsed = tomllib.loads(pruned)["package"]
    # Entire retained package records, including versions, checksums and edges,
    # remain byte-derived from the original lock. No registry resolution occurs.
    if len(parsed) != len(reachable) or any(
            p != by_key[(p["name"], p["version"], p.get("source"))] for p in parsed):
        raise RuntimeError("lockfile closure verification failed")
    return pruned, {"original_packages": len(packages), "retained_packages": len(parsed),
                    "resolution": "exact original reachable package records; no re-resolution"}


def effective_codegen(record):
    commands = [c for c in record["compiler_commands"] if c["test"]]
    if len(commands) != 1:
        raise RuntimeError("expected exactly one lucene-util test rustc invocation")
    options = {}
    for flag in commands[0]["codegen"]:
        key, _, value = flag.partition("=")
        options[key] = value
    return options


def validate_pair(left, right, coverage=False):
    options = [effective_codegen(record) for record in (left, right)]
    actual = [o.get("debuginfo", "0") for o in options]
    if not coverage and actual != ["2", "1"]:
        raise RuntimeError(f"test profile debuginfo was overridden: {actual}")
    if coverage and actual not in (["2", "1"], ["2", "2"], ["1", "1"]):
        raise RuntimeError(f"unexpected effective coverage debuginfo: {actual}")
    ignore = {"debuginfo", "metadata", "extra-filename", "incremental"}
    semantic = [{k: v for k, v in o.items() if k not in ignore} for o in options]
    if semantic[0] != semantic[1]:
        raise RuntimeError(f"codegen options other than debug information differ: {semantic}")
    return {"effective_debuginfo": actual, "other_codegen_options": semantic[0],
            "coverage_override": coverage and actual[0] == actual[1]}


def coverage_signature(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for datum in report["data"]:
        for file in datum["files"]:
            # Counts need not match; actual covered lines/regions must.
            if "/crates/lucene-util/src/" not in file["filename"]:
                continue
            key = file["filename"].split("/crates/lucene-util/src/", 1)[1]
            result[key] = {
                "lines": file["summary"]["lines"],
                "regions": file["summary"]["regions"],
                "segments": [[s[0], s[1], bool(s[2]), *s[3:]]
                             for s in file["segments"]],
            }
    if "fixed_bit_set.rs" not in result:
        raise RuntimeError("coverage JSON omitted the exercised FixedBitSet file")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True,
                        help="new, empty results directory outside scratch")
    parser.add_argument("--max-seconds", type=int, default=1200,
                        help="whole experiment deadline; default 20 minutes")
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    if not Path("/opt/lucene-jars").is_dir() or not os.environ.get("DOCKER_TARGET_VOLUME"):
        parser.error("run through scripts/docker-test.sh inside the capped container")
    for executable in ("cargo", "rustc"):
        if shutil.which(executable) is None:
            parser.error(f"missing {executable}")
    if not 1 <= args.max_seconds <= 1800:
        parser.error("--max-seconds must be between 1 and 1800")
    output = args.output.resolve()
    if output.exists():
        parser.error("--output must not already exist")
    output.mkdir(parents=True)
    target_parent = Path(os.environ["CARGO_TARGET_DIR"]).resolve(strict=True)
    scratch = Path(tempfile.mkdtemp(prefix="cargo-debug-info-probe-", dir=target_parent))
    timing_helper = scratch / "timing-helper.py"
    deadline = time.monotonic() + args.max_seconds
    results = {"scope": "lucene-util library tests only; no release builds",
               "runs": [], "status": "running", "scratch": str(scratch),
               "limitations": ["One truly cold build per setting is exploratory, not a stable estimate",
                               "Cold order is debug2 then debug1; filesystem caches are not flushed",
                               "Coverage compares library tests only, not integration-test coverage or a full gate",
                               "Two changed-test-literal repetitions do not model all production edits",
                               "No workspace-wide build-time, memory or cache-retention claim",
                               "debug=1 intentionally loses debugger local/type detail"]}

    def run(name, command, env, timeout=300, cwd=None):
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            raise RuntimeError("whole experiment deadline exhausted")
        log_path = output / f"{name}.log"
        timing_path = output / f"{name}.time.json"
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, str(timing_helper), str(timing_path),
                                        *command], cwd=cwd or source, env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                status = wait_for_command(process, timeout=min(timeout, remaining))
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"{name}: command timeout; see {log_path}")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        record = {"name": name, "command": command, "exit": status,
                  "wall_seconds": time.perf_counter() - started,
                  "timing": json.loads(timing_path.read_text(encoding="utf-8")),
                  "compiler_commands": compiler_commands(log_text)}
        results["runs"].append(record)
        print(json.dumps({key: value for key, value in record.items()
                          if key != "compiler_commands"}), flush=True)
        if status:
            raise RuntimeError(f"{name}: exit {status}; see {log_path}")
        return log_text, record

    try:
        timing_helper.write_text(TIMING_HELPER, encoding="utf-8", newline="\n")
        source = scratch / "source"
        source.mkdir()
        manifest = (repo / "Cargo.toml").read_text(encoding="utf-8")
        manifest = replace_list(replace_list(manifest, "members"), "default-members")
        (source / "Cargo.toml").write_text(manifest, encoding="utf-8", newline="\n")
        shutil.copyfile(repo / "rust-toolchain.toml", source / "rust-toolchain.toml")
        lock, results["lockfile"] = prune_lockfile((repo / "Cargo.lock").read_text(encoding="utf-8"))
        (source / "Cargo.lock").write_text(lock, encoding="utf-8", newline="\n")
        shutil.copytree(repo / ".cargo", source / ".cargo")
        shutil.copytree(repo / "crates/lucene-util", source / "crates/lucene-util")
        (source / "fixtures").symlink_to(repo / "fixtures", target_is_directory=True)
        (source / "scripts").mkdir()
        shutil.copyfile(repo / "scripts/coverage.sh", source / "scripts/coverage.sh")
        lib = source / "crates/lucene-util/src/lib.rs"
        lib.write_text(lib.read_text(encoding="utf-8") + PROBES,
                       encoding="utf-8", newline="\n")
        changed_file = source / "crates/lucene-util/src/zigzag.rs"
        original = changed_file.read_text(encoding="utf-8")
        if original.count("123456789") != 1:
            raise RuntimeError("expected the known zigzag test literal exactly once")
        alternate = original.replace("123456789", "123456788")
        base_env = os.environ.copy()
        # Preserve rustflags, assertions, overflow checks, optimisation, panic,
        # incremental settings and target CPU. Change only debug info.
        settings = {}
        for setting in (2, 1):
            env = base_env.copy()
            env["CARGO_PROFILE_DEV_DEBUG"] = str(setting)
            env["CARGO_PROFILE_TEST_DEBUG"] = str(setting)
            env["CARGO_TARGET_DIR"] = str(scratch / f"target-debug-{setting}")
            env["CARGO_TERM_COLOR"] = "never"
            settings[setting] = env
        run("rustc-version", ["rustc", "-vV"], base_env, timeout=30)
        run("cargo-version", ["cargo", "-V"], base_env, timeout=30)
        run("llvm-cov-version", ["cargo", "llvm-cov", "--version"], base_env, timeout=30)
        command = ["cargo", "test", "-p", "lucene-util", "--lib", "--no-run",
                   "--locked", "--offline", "-vv"]
        cold_records = []
        for setting in (2, 1):
            _, record = run(f"cold-debug-{setting}", command, settings[setting])
            if not record["compiler_commands"]:
                raise RuntimeError("cold build did not expose actual lucene-util rustc flags")
            target = Path(settings[setting]["CARGO_TARGET_DIR"])
            record["artifact_bytes"] = directory_bytes(target)
            record["incremental_bytes"] = directory_bytes(target / "debug/incremental")
            record["deps_bytes"] = directory_bytes(target / "debug/deps")
            cold_records.append(record)
            run(f"noop-debug-{setting}", command, settings[setting], timeout=60)
        results["test_codegen_validation"] = validate_pair(*cold_records)
        # Alternate order and change only an input of an existing unit test.
        # Public functions and all executable production semantics stay identical.
        for repetition, order in enumerate(((1, 2), (2, 1)), start=1):
            for setting in order:
                changed_file.write_text(alternate if repetition == 1 else original,
                                        encoding="utf-8", newline="\n")
                _, record = run(f"changed-{repetition}-debug-{setting}", command,
                                settings[setting], timeout=180)
                if not record["compiler_commands"]:
                    raise RuntimeError("changed source failed to trigger lucene-util compilation")
        changed_file.write_text(original, encoding="utf-8", newline="\n")
        coverage_records = []
        for setting in (2, 1):
            env = settings[setting].copy()
            env["RUST_BACKTRACE"] = "full"
            text, _ = run(f"semantics-backtrace-debug-{setting}",
                          ["cargo", "test", "-p", "lucene-util", "--lib", "--locked",
                           "--offline", "cargo_debug_info_probe::", "--", "--nocapture"], env)
            if not re.search(r"(?m)^\s*\d+: [^\n]*backtrace_leaf[^\n]*\n"
                             r"(?:[^\n]*\n){0,1}\s+at [^\n]*crates/lucene-util/src/lib\.rs:\d+", text):
                raise RuntimeError(f"debug={setting} backtrace lost function/source line")
            run(f"cov-show-env-debug-{setting}", ["cargo", "llvm-cov", "show-env"], env,
                timeout=30)
            report = output / f"coverage-debug-{setting}.json"
            _, record = run(f"coverage-debug-{setting}",
                          ["bash", "scripts/coverage.sh", "-p", "lucene-util", "--lib",
                           "--locked", "--offline", "-vv", "--",
                           "--json", "--output-path", str(report)], env)
            if not record["compiler_commands"]:
                raise RuntimeError("coverage build did not expose actual rustc flags")
            target = Path(env["CARGO_TARGET_DIR"]) / "llvm-cov-target"
            record["coverage_artifact_bytes"] = directory_bytes(target)
            coverage_records.append(record)
        results["coverage_codegen_validation"] = validate_pair(*coverage_records, coverage=True)
        results["coverage_scope"] = "all lucene-util library tests; compare source segment hit sets and per-file line/region totals"
        left = coverage_signature(output / "coverage-debug-2.json")
        right = coverage_signature(output / "coverage-debug-1.json")
        results["coverage_equal"] = left == right
        if left != right:
            raise RuntimeError("covered source segments or per-file totals differ")
        results["status"] = "complete; measurements only, no adoption"
    except KeyboardInterrupt:
        results["status"] = "interrupted"
        results["error"] = "KeyboardInterrupt; owned command group reaped"
    except Exception as error:
        results["status"] = "failed or bounded stop"
        results["error"] = str(error)
    finally:
        # Delete exactly the directory returned by mkdtemp, never a Cargo cache.
        resolved = scratch.resolve(strict=True)
        if resolved.parent != target_parent or not resolved.name.startswith("cargo-debug-info-probe-"):
            raise RuntimeError(f"refusing unsafe scratch cleanup: {resolved}")
        shutil.rmtree(resolved)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n",
                                              encoding="utf-8", newline="\n")
        print(json.dumps({"status": results["status"], "results": str(output)}), flush=True)
    if results["status"] == "interrupted":
        return 130
    return 0 if results["status"].startswith("complete") else 1


if __name__ == "__main__":
    sys.exit(main())
