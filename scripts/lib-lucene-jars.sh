#!/usr/bin/env bash
# Shared Lucene jar resolution, sourced by the fixture and benchmark scripts.
#
# Resolution order: an explicit --jars directory, then the local Gradle cache
# (fast, and what a developer machine already has), then Maven Central (what a
# CI runner needs, since it has no ~/.gradle).
#
# Callers set JARS to the cache directory before sourcing, then call
# `lucene_classpath <module>...` to get a ready `:`-joined classpath.

LUCENE_VERSION="${LUCENE_VERSION:-10.5.0}"
MAVEN_BASE="https://repo1.maven.org/maven2/org/apache/lucene"

lucene_resolve_jar() {
  local module="$1"
  local jar="$module-$LUCENE_VERSION.jar"
  local found=""
  if [ -f "$JARS/$jar" ]; then echo "$JARS/$jar"; return; fi
  found=$(find "$HOME/.gradle/caches" -name "$jar" ! -name '*sources*' ! -name '*javadoc*' 2>/dev/null | head -1 || true)
  if [ -n "$found" ]; then echo "$found"; return; fi
  mkdir -p "$JARS"
  echo "lucene-jars: downloading $jar from Maven Central" >&2
  curl -fsSL -o "$JARS/$jar" "$MAVEN_BASE/$module/$LUCENE_VERSION/$jar"
  echo "$JARS/$jar"
}

lucene_classpath() {
  local cp="" m
  for m in "$@"; do cp="$cp${cp:+:}$(lucene_resolve_jar "$m")"; done
  echo "$cp"
}
