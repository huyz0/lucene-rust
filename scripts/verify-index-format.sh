#!/usr/bin/env bash
# The index-format matrix: this port writes an index in the format of one Lucene
# release (the writer's `IndexFormat`, 10.5 or 10.6), and real Lucene jars of
# 10.5.1 and 10.6.0 try to open it. 10.6 files must open on 10.6 and not on
# 10.5 (that is the point of the switch); 10.5 files must open on both.
#
# Usage: scripts/verify-index-format.sh [--jars DIR] [--keep]
# The jars come from DIR (default fixtures/.jars/bwc, the bwc fixtures' cache)
# or Maven Central.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
JARS="$PWD/fixtures/.jars/bwc"
KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --jars) JARS="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    *) echo "verify-index-format: unknown argument: $1" >&2; exit 2 ;;
  esac
done
MAVEN_BASE="https://repo1.maven.org/maven2/org/apache/lucene"
jar_for() {
  local v="$1" jar="$JARS/lucene-core-$1.jar"
  if [ ! -s "$jar" ]; then
    mkdir -p "$JARS"
    curl -fsSL --retry 5 -o "$jar" "$MAVEN_BASE/lucene-core/$v/lucene-core-$v.jar"
  fi
  echo "$jar"
}
WORK=$(mktemp -d)
trap '[ "$KEEP" = 1 ] && echo "kept $WORK" || rm -rf "$WORK"' EXIT
mkdir -p "$WORK/cls"
J105=$(jar_for 10.5.1)
J106=$(jar_for 10.6.0)
javac -nowarn -proc:none -cp "$J106" -d "$WORK/cls" fixtures/src/VerifyIndexFormat.java
cargo build -q -p lucene-search --example write_index_format
EX=target/debug/examples/write_index_format
status=0
for fmt in 10.5 10.6; do
  for merge in "" --merge; do
    out="$WORK/index-$fmt${merge:+-merged}"
    "$EX" "$out" "$fmt" $merge >/dev/null
    for v in 10.5.1 10.6.0; do
      jar=$([ "$v" = 10.5.1 ] && echo "$J105" || echo "$J106")
      want=any
      [ "$fmt" = 10.6 ] && want=with-dvp
      [ "$fmt" = 10.5 ] && want=no-dvp
      if java -cp "$WORK/cls:$jar" VerifyIndexFormat "$out" "$want" >"$WORK/log" 2>&1; then
        result=opens
      else
        result=REFUSED
      fi
      expected=opens
      [ "$fmt" = 10.6 ] && [ "$v" = 10.5.1 ] && expected=REFUSED
      mark=ok
      [ "$result" = "$expected" ] || { mark=FAIL; status=1; sed 's/^/    /' "$WORK/log" | grep -v "Picked up" | head -8; }
      echo "written as $fmt${merge:+ (+force merge)}, opened on Lucene $v: $result [$mark]"
    done
  done
done
[ "$status" = 0 ] && echo "verify-index-format: ok" || echo "verify-index-format: FAILED"
exit "$status"
