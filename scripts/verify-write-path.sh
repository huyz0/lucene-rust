#!/usr/bin/env bash
# Run the reverse-direction differential tests: Rust writes real Lucene bytes,
# and a Java program confirms unmodified Lucene 10.5.0 can read them back.
#
# Every fixture under fixtures/data is Java-writes/Rust-reads. These verifiers
# are the opposite direction, and they are the only automated evidence that this
# port's write path emits bytes real Lucene accepts. Round-tripping through this
# port's own reader cannot catch a misreading of the spec that the reader and
# writer share; these can.
#
# Usage: scripts/verify-write-path.sh [--jars DIR] [--keep]
#   --jars DIR  look for the Lucene jars here (default: fixtures/.jars)
#   --keep      keep the generated fixtures instead of deleting them on exit
#
# The postings/term-dictionary write path (.doc/.pos/.pay/.tim/.tip/.tmd) is
# covered by the full-segment, merged-segment, sorted-segment and
# positions-segment cases below -- the last of which (c23) is the one that
# reads back positions, offsets and payloads occurrence by occurrence. Task
# T3.1, see docs/milestones/m3-write-path-proven.md.
set -euo pipefail

LUCENE_MODULES=(lucene-core lucene-analysis-common lucene-queries)

cd "$(git rev-parse --show-toplevel)"
FIXTURES="$PWD/fixtures"
JARS="$FIXTURES/.jars"
KEEP=0

while [ $# -gt 0 ]; do
  case "$1" in
    --jars) JARS="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "verify-write-path: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# shellcheck source=scripts/lib-lucene-jars.sh
source "$(dirname "$0")/lib-lucene-jars.sh"
CP=$(lucene_classpath "${LUCENE_MODULES[@]}")

WORK=$(mktemp -d)
CLASSES=$(mktemp -d)
cleanup() { [ "$KEEP" -eq 1 ] && echo "verify-write-path: fixtures kept in $WORK" || rm -rf "$WORK"; rm -rf "$CLASSES"; }
trap cleanup EXIT

# crate | example | output subdir | verifier class | optional verifier args
CASES=(
  "lucene-codecs|write_stored_fields_fixture|stored-fields|VerifyStoredFields"
  "lucene-codecs|write_field_infos_fixture|field-infos|VerifyFieldInfos"
  "lucene-index|write_segment_info_fixture|segment-info|VerifySegmentInfo"
  "lucene-index|write_segment_infos_fixture|segment-infos|VerifySegmentInfos"
  "lucene-index|write_multi_segment_commit_fixture|multi-segment|VerifySegmentInfos"
  "lucene-codecs|write_points_fixture|points|VerifyPoints"
  "lucene-codecs|write_term_vectors_fixture|term-vectors|VerifyTermVectors"
  "lucene-codecs|write_doc_values_fixture|doc-values|VerifyDocValues"
  "lucene-codecs|write_sparse_numeric_doc_values_fixture|sparse-numeric-dv|VerifySparseNumericDocValues"
  "lucene-codecs|write_norms_fixture|norms|VerifyNorms"
  "lucene-codecs|write_live_docs_fixture|live-docs|VerifyLiveDocs"
  "lucene-codecs|write_compound_format_fixture|compound-format|VerifyCompoundFormat"
  "lucene-codecs|write_fst_fixture|fst|VerifyFst"
  # Vectors: the flat .vec/.vemf store AND the .vem/.vex HNSW graph, both
  # written by this port. Java checks the vectors (an order-sensitive hash over
  # every ordinal's raw bits), the ordinal->doc mapping, the graph arc by arc,
  # and then runs a real TopKnnCollector search over the Rust-built graph and
  # measures its recall against the exact top-k. A graph that decodes cleanly
  # but is built wrong passes every structural check and fails the recall one.
  "lucene-codecs|write_vectors_fixture|vectors|VerifyVectors"
  # Last, and unlike every case above: a whole index written by the real
  # IndexWriter, opened by DirectoryReader and run through CheckIndex. The
  # cases above each hand Lucene one codec file with a hand-built
  # SegmentInfo/FieldInfos, which cannot see anything that binds those files
  # into a segment -- four such defects were live at once while all thirteen
  # passed.
  "lucene-index|write_full_segment_fixture|full-segment|VerifyFullSegment"
  # And the same again for a *merged* segment. The merge's two fast
  # stored-fields paths copy already-compressed chunk bytes (BULK) and
  # already-serialized document bytes (DOC) rather than re-encoding them, so a
  # boundary error there produces a segment that reads back plausible but wrong
  # documents -- something only a reader that is not this port can catch.
  "lucene-index|write_merged_segment_fixture|merged-segment|VerifyMergedSegment"
  # And a segment whose documents carry *vector* fields, written through
  # IndexWriter.add_document_with_vectors. VerifyVectors above hands Lucene the
  # four vector files with a hand-built FieldInfos; what that cannot see is what
  # binds them into a segment -- PerFieldKnnVectorsFormat's suffixed file names
  # and .fnm attributes, SegmentInfo.files, and .fnm vector_dimension agreeing
  # with the .vemf for every field including the ones the flush wrote no vectors
  # for. Each of those fails silently (the field reads back absent, or the
  # segment refuses to open), so the verifier runs real KnnFloatVectorQuery /
  # KnnByteVectorQuery against Lucene's own brute-force top-k, and CheckIndex.
  "lucene-index|write_vector_segment_fixture|vector-segment|VerifyVectorSegment"
  # And a segment built out of document *blocks* (`addDocuments`). `hasBlocks`
  # is one byte in the `.si` that nothing else in the write path sets; a
  # segment carrying blocks but reporting `hasBlocks=false` reads back
  # perfectly while silently invalidating every parent/child join query
  # against it, so the verifier asserts on `LeafMetaData.hasBlocks()` and on
  # the blocks' contiguity, not only on CheckIndex being clean.
  "lucene-index|write_block_segment_fixture|block-segment|VerifyBlockSegment"
  # And a segment whose doc-values have been *updated in place*
  # (`updateNumericDocValue`/`updateBinaryDocValue`). This port used to write
  # a delta file of its own invention here, so an index carrying a doc-values
  # update was one real Lucene could not open at all. The format is now
  # Lucene's -- the updated field's whole column rewritten into a
  # generation-suffixed `.dvm`/`.dvd`/`.dvs` plus a `FieldInfos` generation --
  # and four separate pieces of bookkeeping have to agree for a reader to find
  # it. Each way of getting them wrong reads back fine through this port's own
  # reader, so the verifier reads every document's value through a real
  # DirectoryReader, asserts the superseded generations were reclaimed, and
  # runs CheckIndex.
  "lucene-index|write_doc_values_updates_fixture|dv-updates|VerifyDocValuesUpdates"
  # And an index-sorted segment. An index sort is the one property here whose
  # violation leaves every file valid: correct checksums, in-range doc ids,
  # decodable term dictionary -- only the association between the files is
  # wrong. So the verifier re-derives the expected permutation itself and
  # checks every doc id's stored value, both doc-values columns, its unique
  # postings term, its norm and its vector against the document that is
  # supposed to be there, then runs CheckIndex, whose testSort rebuilds the
  # sort's comparators from the `.si` and walks adjacent doc ids. Measured:
  # a flush that permutes the documents but not the vectors is CheckIndex-
  # clean and silently attaches every vector to the wrong document, and a
  # missing-value comparator that disagrees with the `.si` it wrote fails
  # testSort at docID=1944.
  "lucene-index|write_sorted_segment_fixture|sorted-segment|VerifySortedSegment"
  # And the same index-sorted corpus again, this time produced by a **merge**
  # of eight internally-sorted flushes with one document in fifty-three
  # deleted -- verified by the *same* class, because the whole claim is that a
  # merged sorted segment is indistinguishable from a flushed one. A merge is
  # strictly harder than a flush here: the sources' key ranges overlap
  # completely (so a concatenation cannot come out ordered by accident), the
  # deletions rule out the stored-fields and term-vector byte-copy paths, one
  # HNSW graph has to be rebuilt over a brand-new merged ordinal space, and
  # each of postings, doc values, norms, term vectors and vectors is mapped
  # through its own doc map. Every one of those can go wrong while leaving a
  # segment that decodes cleanly and passes every checksum.
  "lucene-index|write_sorted_merged_segment_fixture|sorted-merged-segment|VerifySortedSegment|53"
  # And a segment whose postings carry positions, offsets and payloads. Until
  # c23 the only whole-index case above (`write_full_segment_fixture`) indexed
  # DOCS_AND_FREQS, so no `.pos` or `.pay` file this port wrote had ever been
  # read by anything but this port itself -- and c20 had just added a whole new
  # wire region to them (the level-0/level-1 `.pos`/`.pay` skip records) whose
  # only evidence was two of our own readers agreeing with each other. That is
  # the same evidence shape that let b4's FST framing bug and b11's invented
  # `.si` sort encoding round-trip perfectly while being wrong.
  #
  # The fixture is sized to the format rather than to convenience: 20 000
  # documents with a term in every one of them, so two whole LEVEL1_NUM_DOCS
  # spans close and 78 level-0 blocks plus a group-varint tail all carry skip
  # records; per-document frequencies cycling with a period coprime with 256 so
  # a level-1 `posBufferUpto` of zero is distinguishable from a hardcoded zero
  # (c20's Tier-2 review found exactly that degeneracy in its own fixture); and
  # six fields covering every IndexOptions rung plus offsets-without-payloads
  # and payloads-without-offsets, since Lucene frames `.pay` differently for
  # each. Java walks 51 documents chosen around those boundaries with a fresh
  # PostingsEnum per sample -- so each is reached through `advance` and the skip
  # records -- comparing every occurrence's position, both offsets and its
  # payload, then runs a PhraseQuery (positions that decode but sit at wrong
  # absolute values match every term query and no phrase) and CheckIndex.
  # Measured: writing the level-1 `posBufferUpto` byte as a constant 0 fails at
  # doc 8192, the level-0 one at doc 256, and dropping the payload-length
  # stream while the `.fnm` still claims payloads mis-frames every offset.
  "lucene-index|write_positions_segment_fixture|positions-segment|VerifyPositionsSegment"
)

echo "verify-write-path: compiling verifiers"
javac -nowarn -cp "$CP" -d "$CLASSES" "$FIXTURES"/src/Verify*.java

echo "verify-write-path: writing fixtures from Rust and verifying with Lucene $LUCENE_VERSION"
failed=0
for case in "${CASES[@]}"; do
  IFS='|' read -r crate example subdir verifier extra <<<"$case"
  out="$WORK/$subdir"
  if ! cargo run --quiet -p "$crate" --example "$example" -- "$out" >/dev/null; then
    echo "  FAIL (rust write)   $example"; failed=$((failed+1)); continue
  fi
  # shellcheck disable=SC2086 -- $extra is a deliberate word split
  if java --enable-native-access=ALL-UNNAMED -cp "$CLASSES:$CP" "$verifier" "$out" $extra >/dev/null 2>&1; then
    echo "  ok                  $verifier <- $example"
  else
    echo "  FAIL (java verify)  $verifier <- $example"
    # shellcheck disable=SC2086
    java --enable-native-access=ALL-UNNAMED -cp "$CLASSES:$CP" "$verifier" "$out" $extra 2>&1 | sed 's/^/      /' | tail -15
    failed=$((failed+1))
  fi
done

echo
if [ "$failed" -eq 0 ]; then
  echo "verify-write-path: ok (${#CASES[@]}/${#CASES[@]} passed)"
else
  echo "verify-write-path: FAILED ($failed of ${#CASES[@]} cases)"
fi
exit "$failed"
