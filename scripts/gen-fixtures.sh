#!/usr/bin/env bash
# Regenerate the Java-produced differential-testing fixtures under fixtures/data.
#
# Modes:
#   --only NAME   regenerate exactly one generator's fixtures (repeatable)
#   --all         regenerate EVERY fixture in place -- destructive, see below
#   --check       regenerate into temp dirs and verify the committed fixtures are
#                 still what Lucene 10.5.0 actually produces (writes nothing)
#
# Usage:
#   scripts/gen-fixtures.sh --only GenFst [--only GenNorms]
#   scripts/gen-fixtures.sh --check
#   scripts/gen-fixtures.sh --all           # requires the explicit flag
#   scripts/gen-fixtures.sh --out DIR       # scratch dir: no flag needed
#
#   --only NAME  run just this generator (with or without the `Gen` prefix),
#                then the manifest appenders. Repeatable. --list names them.
#   --all        regenerate everything into fixtures/data. Required to be
#                explicit: see "Why a full run needs a flag".
#   --list       print the generator names --only accepts, and exit
#   --out DIR    write fixtures here (default: fixtures/data). A non-default
#                --out cannot clobber the evidence base, so it needs no --all.
#   --check      verify the fixtures at --out (default fixtures/data) against
#                two fresh generations into temp dirs; never writes to --out
#   --jars DIR   look for the Lucene jars here before the Gradle cache;
#                also where they are downloaded to (default: fixtures/.jars)
#
# ---------------------------------------------------------------------------
# Why a full run needs a flag
#
# Every correctness claim in this port rests on fixtures produced by real
# Lucene 10.5.0. A full regeneration does not "refresh" them -- it REPLACES
# them with different-but-equally-plausible bytes: Lucene stamps a fresh
# random segment id (StringHelper.randomId()) into every index it writes, so
# 629 of the 675 generated files change on every run by design. The suite then
# still goes green, over evidence that is no longer the evidence the findings
# were written against, and the diff is hundreds of opaque binary files that no
# reviewer can read. Batch c29 triggered exactly this and had to revert the
# whole tree by hand.
#
# So: an in-place full run requires --all, and a single fixture is regenerated
# with --only, which touches nothing else.
# ---------------------------------------------------------------------------
# What --check can and cannot prove
#
# Fixtures written through a real Lucene IndexWriter are NOT byte-reproducible
# (see above), so --check does not diff everything against the committed tree.
# It:
#   1. generates twice, and calls a file deterministic iff the two runs agree;
#   2. asserts every deterministic file matches the committed copy byte for
#      byte -- this is what catches a hand-edit;
#   3. asserts the generated file tree matches the committed tree, so a
#      generator that silently stops emitting a file is caught even where the
#      bytes cannot be compared;
#   4. asserts every committed manifest carries exactly the keys a full
#      generate-then-append run produces -- this is what catches the
#      Append*Manifest keys being dropped, which byte-comparison cannot,
#      because the manifests of IndexWriter-written fixtures are in the
#      non-deterministic set;
#   5. asserts every committed index still carries the segment id recorded in
#      fixtures/segment-ids.txt -- this is what catches an index having been
#      regenerated, which nothing else here can see, since fresh bytes are
#      indistinguishable from correct bytes.
#
# Deriving the deterministic set by generating twice, rather than maintaining a
# hand-written list, means the check stays correct as fixtures are added.
#
# Not yet checkable: running the Rust suite against freshly generated fixtures.
# crates/lucene-ffi/src/segment.rs hardcodes the committed blocktree fixture's
# segment ID, so fresh bytes fail those tests -- which is the same coupling
# fixtures/segment-ids.txt now makes explicit. See docs/milestones/
# m0-ci-and-green-tree.md ("Findings") for the follow-up.
# ---------------------------------------------------------------------------
set -euo pipefail

# Modules the generators actually need. lucene-queries is required by
# GenBlockTree (org.apache.lucene.queries.spans); the fixtures README used to
# document only lucene-core + lucene-analysis-common, which no longer compiles.
# lucene-facet is required by GenFacets (org.apache.lucene.facet.*).
LUCENE_MODULES=(lucene-core lucene-analysis-common lucene-queries lucene-facet)

cd "$(git rev-parse --show-toplevel)"
FIXTURES="$PWD/fixtures"
OUT="$FIXTURES/data"
JARS="$FIXTURES/.jars"
IDS_BASELINE="$FIXTURES/segment-ids.txt"
IDS_SCRIPT="$PWD/scripts/fixture-segment-ids.py"
CHECK=0
ALL=0
LIST=0
OUT_EXPLICIT=0
ONLY=()

while [ $# -gt 0 ]; do
  case "$1" in
    --out)   OUT="$2"; OUT_EXPLICIT=1; shift 2 ;;
    --jars)  JARS="$2"; shift 2 ;;
    --only)  ONLY+=("$2"); shift 2 ;;
    --all)   ALL=1; shift ;;
    --list)  LIST=1; shift ;;
    --check) CHECK=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "gen-fixtures: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Generators, then manifest appenders. Both lists come from the filesystem so a
# newly added program is picked up without editing this script.
#
# Order matters: Append*Manifest programs open an already-generated index
# read-only and append cross-engine ground truth to its manifest.properties
# (they deliberately do not regenerate the index). Running only the Gen*
# programs yields an incomplete manifest -- 239 keys instead of 468 for
# blocktree_index. That is why --only runs the appenders too: they are
# idempotent (each strips its own prefix before re-appending), so re-running
# them over an index this invocation did not touch rewrites the same bytes.
mapfile -t GENERATORS < <(cd "$FIXTURES/src" && ls Gen*.java | sed 's/\.java$//' | sort)
mapfile -t APPENDERS  < <(cd "$FIXTURES/src" && ls Append*.java | sed 's/\.java$//' | sort)

if [ "$LIST" -eq 1 ]; then
  printf '%s\n' "${GENERATORS[@]}"
  exit 0
fi

# --- resolve --only ----------------------------------------------------------
SELECTED=()
if [ "${#ONLY[@]}" -gt 0 ]; then
  for want in "${ONLY[@]}"; do
    # Accept "GenFst" and "Fst" alike; the Gen prefix is noise at the CLI.
    case "$want" in Gen*) cls="$want" ;; *) cls="Gen$want" ;; esac
    found=0
    for g in "${GENERATORS[@]}"; do
      if [ "$g" = "$cls" ]; then found=1; SELECTED+=("$cls"); break; fi
    done
    if [ "$found" -eq 0 ]; then
      echo "gen-fixtures: no such generator: $want" >&2
      echo "gen-fixtures: run 'scripts/gen-fixtures.sh --list' for the names" >&2
      exit 2
    fi
  done
fi

# --- reject combinations that would quietly do the wrong thing ----------------
# --check always generates the full set (that is what makes the deterministic
# subset derivable), so honouring --only there would verify less than the flag
# implies. Say so rather than silently widening it.
if [ "$CHECK" -eq 1 ] && [ "${#SELECTED[@]}" -gt 0 ]; then
  echo "gen-fixtures: --check verifies the whole tree; it cannot be scoped with --only" >&2
  exit 2
fi
if [ "$ALL" -eq 1 ] && [ "${#SELECTED[@]}" -gt 0 ]; then
  echo "gen-fixtures: --all and --only contradict each other; pick one" >&2
  exit 2
fi

# --- refuse a full in-place run without --all --------------------------------
if [ "$CHECK" -eq 0 ] && [ "${#SELECTED[@]}" -eq 0 ] && [ "$ALL" -eq 0 ] && [ "$OUT_EXPLICIT" -eq 0 ]; then
  cat >&2 <<'REFUSAL'
gen-fixtures: refusing to regenerate every fixture in place.

A full run REPLACES every committed index with different bytes -- Lucene stamps
a fresh random segment id into each one -- so the suite stays green over
evidence that is no longer the evidence the sweep's findings were written
against, and the diff is hundreds of unreadable binary files.

What you probably want:
  scripts/gen-fixtures.sh --only <Generator>   regenerate one fixture
  scripts/gen-fixtures.sh --list               the generator names
  scripts/gen-fixtures.sh --check              verify the committed tree
  scripts/gen-fixtures.sh --out /tmp/scratch   generate somewhere harmless

If you really do mean all of them (a Lucene version bump is the usual reason),
pass --all. Expect fixtures/segment-ids.txt to change; that diff is the
human-readable record that every index was replaced, and it belongs in the
commit message.
REFUSAL
  exit 2
fi

# --- resolve the Lucene jars -------------------------------------------------
# Prefer --jars, then the Gradle cache (fast, local), then Maven Central (CI).
# shellcheck source=scripts/lib-lucene-jars.sh
source "$(dirname "$0")/lib-lucene-jars.sh"
CP=$(lucene_classpath "${LUCENE_MODULES[@]}")

# --- compile -----------------------------------------------------------------
CLASSES=$(mktemp -d)
trap 'rm -rf "$CLASSES" ${TMP_A:-} ${TMP_B:-}' EXIT
javac -nowarn -cp "$CP" -d "$CLASSES" "$FIXTURES"/src/*.java

generate_into() {
  local dest="$1"; shift
  local -a classes=("$@")
  mkdir -p "$dest"
  for cls in "${classes[@]}"; do
    java --enable-native-access=ALL-UNNAMED -cp "$CLASSES:$CP" "$cls" "$dest" >/dev/null
  done
  # IndexWriter leaves a zero-byte write.lock behind in every index it creates.
  # It is a lock artifact, not a fixture, and nothing in crates/ reads it --
  # drop it so an in-place regeneration leaves a git-clean tree.
  find "$dest" -name 'write.lock' -type f -delete
}

if [ "$CHECK" -eq 0 ]; then
  if [ "${#SELECTED[@]}" -gt 0 ]; then
    echo "gen-fixtures: generating ${SELECTED[*]} + ${#APPENDERS[@]} appenders into $OUT"
    generate_into "$OUT" "${SELECTED[@]}" "${APPENDERS[@]}"
  else
    echo "gen-fixtures: generating ${#GENERATORS[@]} generators + ${#APPENDERS[@]} appenders into $OUT"
    generate_into "$OUT" "${GENERATORS[@]}" "${APPENDERS[@]}"
  fi
  # Keep the segment-id baseline in step with the tree it describes, but only
  # for the real fixture directory -- a scratch --out has no baseline.
  if [ "$OUT" = "$FIXTURES/data" ]; then
    python3 "$IDS_SCRIPT" "$OUT" > "$IDS_BASELINE"
    echo "gen-fixtures: refreshed $(basename "$IDS_BASELINE") ($(wc -l < "$IDS_BASELINE") indexes)"
  fi
  echo "gen-fixtures: ok"
  exit 0
fi

# --- check mode --------------------------------------------------------------
TMP_A=$(mktemp -d); TMP_B=$(mktemp -d)
echo "gen-fixtures: generating twice to derive the deterministic subset"
generate_into "$TMP_A" "${GENERATORS[@]}" "${APPENDERS[@]}"
generate_into "$TMP_B" "${GENERATORS[@]}" "${APPENDERS[@]}"

status=0
deterministic=0; nondeterministic=0; missing=0; mismatched=0; extra=0
manifest_bad=0; id_bad=0

# write.lock is an IndexWriter lock artifact, not a fixture: generators leave it
# behind and it is deliberately not committed.
GENERATED_NOISE='(^|/)write\.lock$'

while IFS= read -r rel; do
  if [[ "$rel" =~ $GENERATED_NOISE ]]; then continue; fi
  a="$TMP_A/$rel"; b="$TMP_B/$rel"; c="$OUT/$rel"
  if [ ! -e "$c" ]; then
    echo "  MISSING from committed fixtures: $rel"; missing=$((missing+1)); status=1; continue
  fi
  if cmp -s "$a" "$b"; then
    deterministic=$((deterministic+1))
    if ! cmp -s "$a" "$c"; then
      echo "  MISMATCH (deterministic file differs from committed): $rel"
      mismatched=$((mismatched+1)); status=1
    fi
  else
    nondeterministic=$((nondeterministic+1))
  fi
done < <(cd "$TMP_A" && find . -type f | sed 's|^\./||' | sort)

# Files committed under fixtures/data that no Java generator produces. These are
# written by Rust examples (see scripts/verify-write-path.sh) and are expected.
RUST_WRITTEN='^sparse_numeric_doc_values/'
while IFS= read -r rel; do
  [ -e "$TMP_A/$rel" ] && continue
  if [[ "$rel" =~ $GENERATED_NOISE ]]; then continue; fi
  if [[ "$rel" =~ $RUST_WRITTEN ]]; then continue; fi
  echo "  EXTRA in committed fixtures, produced by no generator: $rel"
  extra=$((extra+1)); status=1
done < <(cd "$OUT" && find . -type f | sed 's|^\./||' | sort)

# --- manifest keys -----------------------------------------------------------
# The manifests of IndexWriter-written fixtures land in the non-deterministic
# set (they carry segment-dependent values), so the byte comparison above says
# nothing about them. Compare their KEY SETS instead: that is stable across
# runs, and it is precisely what an Append*Manifest program contributes. A tree
# regenerated with the Gen* programs alone loses 229 keys from
# blocktree_index/manifest.properties and every byte check still passes.
manifest_keys() { sed -n 's/^\([A-Za-z0-9_.-]*\)=.*/\1/p' "$1" | sort -u; }

while IFS= read -r rel; do
  c="$OUT/$rel"
  [ -e "$c" ] || continue   # already reported as MISSING above
  fresh=$(manifest_keys "$TMP_A/$rel")
  live=$(manifest_keys "$c")
  [ "$fresh" = "$live" ] && continue
  dropped=$(comm -23 <(printf '%s\n' "$fresh") <(printf '%s\n' "$live"))
  added=$(comm -13 <(printf '%s\n' "$fresh") <(printf '%s\n' "$live"))
  if [ -n "$dropped" ]; then
    echo "  MANIFEST KEYS DROPPED: $rel -- $(printf '%s\n' "$dropped" | wc -l) key(s) the generators produce are not in the committed file"
    printf '%s\n' "$dropped" | head -5 | sed 's/^/      missing: /'
    printf '%s\n' "$dropped" | sed -n '6p' | grep -q . && echo "      ... (run gen-fixtures.sh --check for the rest)"
  fi
  if [ -n "$added" ]; then
    echo "  MANIFEST KEYS UNEXPLAINED: $rel -- $(printf '%s\n' "$added" | wc -l) committed key(s) no generator produces"
    printf '%s\n' "$added" | head -5 | sed 's/^/      extra: /'
  fi
  manifest_bad=$((manifest_bad+1)); status=1
done < <(cd "$TMP_A" && find . -type f \( -name 'manifest.properties' -o -name '*.manifest.properties' \) | sed 's|^\./||' | sort)

# --- segment ids -------------------------------------------------------------
# The one kind of damage nothing above can see: an index regenerated in place.
# Fresh bytes are indistinguishable from correct bytes -- same generator, same
# Lucene, only a new StringHelper.randomId(). fixtures/segment-ids.txt is the
# committed record of which id each index is supposed to carry; a diff against
# it names the regenerated indexes, one readable line each.
if [ ! -e "$IDS_BASELINE" ]; then
  echo "  SEGMENT ID BASELINE MISSING: $IDS_BASELINE (regenerate with scripts/fixture-segment-ids.py fixtures/data > $IDS_BASELINE)"
  id_bad=1; status=1
else
  live_ids=$(python3 "$IDS_SCRIPT" "$OUT")
  # `|| true`: diff exits 1 on a difference, which is the case we handle.
  id_diff=$(diff -u "$IDS_BASELINE" <(printf '%s\n' "$live_ids") | tail -n +3 | grep '^[-+]' || true)
  if [ -n "$id_diff" ]; then
    echo "  SEGMENT IDS CHANGED: the committed fixtures are not the indexes segment-ids.txt records."
    echo "    An index was regenerated (a fresh StringHelper.randomId()), or one was added"
    echo "    or removed without refreshing the baseline. Every hardcoded segment id in the"
    echo "    Rust suite -- crates/lucene-ffi/src/segment.rs among them -- is pinned to these."
    printf '%s\n' "$id_diff" | head -10 | sed 's/^/      /'
    id_bad=$(printf '%s\n' "$id_diff" | wc -l)
    status=1
  fi
fi

echo
echo "gen-fixtures --check summary:"
echo "  deterministic files verified byte-identical : $((deterministic - mismatched))"
echo "  non-deterministic files (random segment id) : $nondeterministic"
echo "  deterministic mismatches                    : $mismatched"
echo "  missing from committed tree                 : $missing"
echo "  unexplained extras in committed tree        : $extra"
echo "  manifests with a wrong key set              : $manifest_bad"
echo "  segment-id baseline lines that disagree     : $id_bad"
[ "$status" -eq 0 ] && echo "gen-fixtures: ok" || echo "gen-fixtures: FAILED"
exit "$status"
