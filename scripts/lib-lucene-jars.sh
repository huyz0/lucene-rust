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

# JVM flags every fixture-writing `java` passes. Lucene's scalar float kernels
# (DefaultVectorUtilSupport: dotProduct, squareDistance, cosine) fuse
# multiply-add only where Constants.HAS_FAST_SCALAR_FMA guesses it is fast --
# on for Intel and arm64 Linux, off for an AMD CPU below AVX-512 -- so the
# same generator writes different vector score bits on different machines.
# That is how m7_*_index/searches.tsv failed `gen-fixtures.sh --check` on a CI
# runner while regenerating byte for byte locally. The committed fixtures are
# the FMA kernel's (crates/lucene-codecs/tests/vectors_fixtures.rs says so), so
# pin it on. It takes effect wherever the CPU has FMA at all (HotSpot UseFMA),
# which every x86-64-v3 and arm64 machine this project runs on does.
LUCENE_FIXTURE_JVM_OPTS=(-Dlucene.useScalarFMA=true -Dlucene.useVectorFMA=true)

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

# Third-party jars a Lucene module depends on, as `group:artifact:version`
# (Maven coordinates, so the Gradle cache and Maven Central lookups match
# the ones above). lucene-spatial-extras 10.5.0's pom names these two; both
# are Apache-2.0 (docs/licences.md).
SPATIAL_EXTRAS_DEPS=(org.locationtech.spatial4j:spatial4j:0.8 io.sgr:s2-geometry-library-java:1.0.0)

thirdparty_resolve_jar() {
  local coord="$1" group artifact version jar found=""
  IFS=: read -r group artifact version <<< "$coord"
  jar="$artifact-$version.jar"
  if [ -f "$JARS/$jar" ]; then echo "$JARS/$jar"; return; fi
  found=$(find "$HOME/.gradle/caches" -name "$jar" 2>/dev/null | head -1 || true)
  if [ -n "$found" ]; then echo "$found"; return; fi
  mkdir -p "$JARS"
  echo "lucene-jars: downloading $jar from Maven Central" >&2
  curl -fsSL -o "$JARS/$jar" \
    "https://repo1.maven.org/maven2/${group//.//}/$artifact/$version/$jar"
  echo "$JARS/$jar"
}

thirdparty_classpath() {
  local cp="" c
  for c in "$@"; do cp="$cp${cp:+:}$(thirdparty_resolve_jar "$c")"; done
  echo "$cp"
}
