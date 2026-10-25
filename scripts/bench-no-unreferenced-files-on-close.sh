#!/usr/bin/env bash
# Run via scripts/docker-test.sh bash scripts/bench-no-unreferenced-files-on-close.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
WARMUP="${1:-1000}"
MEASURE="${2:-1500}"
check_load() {
  local load
  load=$(cut -d' ' -f1 /proc/loadavg)
  if awk "BEGIN{exit !($load > 1.5)}"; then
    echo "no-unreferenced-files-on-close benchmark: load average $load exceeds 1.5" >&2
    exit 3
  fi
}
check_load
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
python3 - "$OUT/snapshot" <<'PY'
from pathlib import Path
import shutil
import sys
root = Path(sys.argv[1])
source = Path('fixtures/data/merge_policies/commits_index')
properties = dict(line.split('=', 1) for line in Path(
    'fixtures/data/merge_policies/commits.properties').read_text().splitlines()
    if line and not line.startswith('#'))
expected = sorted(properties['commit.2.files'].split(','))
assert properties['commit.2.segments_file'] == 'segments_3'
assert properties['commit.2.num_docs'] == '3'
shutil.copytree(source, root / 'retained')
(root / 'latest').mkdir()
for name in expected:
    shutil.copyfile(source / name, root / 'latest' / name)
assert sorted(p.name for p in (root / 'retained').glob('segments_*')) == [
    'segments_1', 'segments_2', 'segments_3']
assert set(p.name for p in (root / 'retained').iterdir()) - set(expected) == {
    'segments_1', 'segments_2'}
for stage in ['latest', 'retained']:
    (root / f'{stage}-expected-files.txt').write_text('\n'.join(expected) + '\n')
    (root / f'{stage}-expected-docs.txt').write_text(properties['commit.2.num_docs'] + '\n')
    (root / f'{stage}-expected-error.txt').write_text('')
# These are the actual close oracle's final listings and reader counts, rather
# than assuming that every source file survives writer cleanup.
oracle = [line.split('\t') for line in Path(
    'fixtures/testframework/data/no_unreferenced_files_on_close.tsv').read_text().splitlines()]
for stage in ['flush', 'merged']:
    label = f'simpletext_{stage}'
    shutil.copytree(Path('fixtures/data/codecs_simpletext/codec') / stage / 'index', root / label)
    outcomes = [row for row in oracle if row[:3] == ['outcome', label, 'first']]
    if len(outcomes) != 1:
        raise SystemExit(f'{label}: missing unique Java close outcome')
    expected_error = ''
    if stage == 'flush':
        if outcomes[0][3:5] != ['failure', 'IllegalArgumentException']:
            raise SystemExit(f'{label}: expected Java parent-field configuration failure')
        expected_error = bytes.fromhex(outcomes[0][5]).decode('utf-8')
        if 'parent document field' not in expected_error:
            raise SystemExit(f'{label}: wrong Java failure message')
    elif outcomes[0][3] != 'ok':
        raise SystemExit(f'{label}: Java merged close oracle must succeed')
    states = [row for row in oracle if row[:3] == ['state', label, 'after']]
    if len(states) != 1 or len(states[0]) != 5:
        raise SystemExit(f'{label}: missing unique Java final-state oracle')
    names = sorted(bytes.fromhex(name).decode('utf-8') for name in states[0][3].split(',') if name)
    docs = int(states[0][4])
    if docs <= 0:
        raise SystemExit(f'{label}: expected a readable nonempty Java index')
    (root / f'{label}-expected-files.txt').write_text('\n'.join(names) + '\n')
    (root / f'{label}-expected-docs.txt').write_text(str(docs) + '\n')
    (root / f'{label}-expected-error.txt').write_text(expected_error)
PY
cargo build --release -p lucene-test-framework --example no_unreferenced_files_on_close_micro --quiet
javac -nowarn -proc:none -cp "$CP" -d "$OUT" benchmarks/micro/java/NoUnreferencedFilesOnCloseMicro.java
RUST="${CARGO_TARGET_DIR:-target}/release/examples/no_unreferenced_files_on_close_micro"
run_rust() {
  check_load
  taskset -c "${BENCH_PIN:-0,1}" "$RUST" "$OUT/snapshot" "$WARMUP" "$MEASURE" > "$OUT/rust.$1.tsv"
}
run_java() {
  check_load
  taskset -c "${BENCH_PIN:-0,1}" java -ea --add-modules jdk.incubator.vector -Xms128m -Xmx128m -cp "$CP:$OUT" NoUnreferencedFilesOnCloseMicro "$OUT/snapshot" "$WARMUP" "$MEASURE" > "$OUT/java.$1.tsv"
}
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then run_rust "$rep"; run_java "$rep";
  else run_java "$rep"; run_rust "$rep"; fi
done
python3 scripts/bench-micro-report.py "$OUT" 3 \
  --require-check no_unreferenced_close_latest --require-check no_unreferenced_close_retained \
  --require-check no_unreferenced_close_simpletext_flush --require-check no_unreferenced_close_simpletext_merged
for engine in rust java; do for rep in 1 2 3; do sed "s/^/$engine rep=$rep /" "$OUT/$engine.$rep.tsv"; done; done
echo 'Actual MockDirectoryWrapper.close with assert-no-unreferenced enabled and CheckIndex disabled: binary latest-only, all three retained binary commits, SimpleText merged, and SimpleText flushed. The first three succeed; flushed fails during the first writer construction because its parent field is not configured, matching the exact Java oracle error. Every iteration verifies status, exact final filenames and live document counts; retained binary input must lose segments_1 and segments_2 in the second writer pass.'
echo 'Timed: actual close success or expected failure and internal cleanup. The three successful cases include both writers, internal readers/doc-count comparisons, file lists and locks; SimpleText flush stops at the first writer configuration failure. Fresh memory directories, wrapper creation, external status/filename/doc-count validation and backing-store cleanup are outside timing in both engines. Java records delegate close without destroying the memory store; Rust Directory has no close.'
echo 'Memory: the option adds one retained boolean flag. Transient writer, reader and scan allocations are unmeasured; this does not establish whole-wrapper owned-memory, peak-allocation, GC or RSS parity.'
