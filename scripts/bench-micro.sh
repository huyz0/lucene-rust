#!/usr/bin/env bash
# Component microbenchmarks: run the Rust and Java harnesses over the same
# generated inputs, join on case name, and report the per-case ratio.
#
# Where scripts/bench-compare.sh answers "is a query faster", this answers
# "is this decode kernel faster". M1-e2e's profile came out flat -- largest
# single item 14.78% -- and a flat profile is precisely what an end-to-end
# benchmark cannot diagnose: it cannot tell you a kernel sitting at 9% of the
# profile is 3x off Lucene's. That is what this measures.
#
# Java runs with --add-modules jdk.incubator.vector so Lucene's Panama
# vectorized decode is live. Without it Lucene silently falls back to the
# scalar DefaultVectorizationProvider, and every ratio here flatters Rust.
#
# Usage: scripts/bench-micro.sh [--bench NAME] [--warmup-ms N] [--measure-ms N] [--pin CPUS]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

BENCH=for_decode
INDEX_ARG=""
# Repetitions of the whole A/B pair. Three is the minimum that gives a median
# and a spread; the spread is what decides whether a difference is reportable.
REPS=3
# Empty until --warmup-ms or the bench's own BENCH_WARMUP sets it; 1500 else.
WARMUP=""
# Set by a bench below, never taken from the environment.
BENCH_WARMUP=""
CASES=""
REPORT_ARGS=()
MEASURE=2000
JARS="$PWD/fixtures/.jars"
# Same pinning rationale as bench-compare.sh: this is a hybrid P/E-core part,
# and a mid-measurement migration onto an E-core dominates the variance.
PIN="0,1"

while [ $# -gt 0 ]; do
  case "$1" in
    --bench)      BENCH="$2";   shift 2 ;;
    --index)      INDEX_ARG="$2"; shift 2 ;;
    --reps)       REPS="$2";    shift 2 ;;
    --warmup-ms)  WARMUP="$2";  shift 2 ;;
    --measure-ms) MEASURE="$2"; shift 2 ;;
    --pin)        PIN="$2";     shift 2 ;;
    --jars)       JARS="$2";    shift 2 ;;
    -h|--help)    sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "bench-micro: unknown argument: $1" >&2; exit 2 ;;
  esac
done

INDEX="${INDEX_ARG:-$PWD/benchmarks/.corpus/merged}"
NEEDS_INDEX=""
case "$BENCH" in
  for_decode)
    MAIN=org.apache.lucene.codecs.lucene104.ForUtilMicro
    SRC=benchmarks/micro/java/org/apache/lucene/codecs/lucene104/ForUtilMicro.java ;;
  postings_iter)
    MAIN=PostingsIterMicro
    SRC=benchmarks/micro/java/PostingsIterMicro.java
    NEEDS_INDEX=1 ;;
  direct_reader)
    MAIN=DirectReaderMicro
    SRC=benchmarks/micro/java/DirectReaderMicro.java ;;
  stored_fields)
    MAIN=StoredFieldsMicro
    SRC=benchmarks/micro/java/StoredFieldsMicro.java
    NEEDS_INDEX=1 ;;
  reader_open)
    MAIN=ReaderOpenMicro
    SRC=benchmarks/micro/java/ReaderOpenMicro.java
    NEEDS_INDEX=1 ;;
  index)
    # The one write-path case: both sides build their own index from the same
    # generated corpus, so there is no shared input index to point at.
    MAIN=IndexMicro
    SRC=benchmarks/micro/java/IndexMicro.java
    JAR_MODULES="lucene-core lucene-analysis-common"
    RUST_BIN=index-bench ;;
  # M11: one analyzer per analysis-common package, over SweepMicro's documents.
  analysis_common)
    MAIN=AnalysisCommonMicro
    SRC=benchmarks/micro/java/AnalysisCommonMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-analysis-common" ;;
  # M11 part 3: language analyzers and synonym filters over their corpora.
  analysis_lang)
    MAIN=AnalysisLangMicro
    SRC=benchmarks/micro/java/AnalysisLangMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-analysis-common" ;;
  # M11 part 3: decompounders, WordDelimiterFilter, classic, wikipedia,
  # DateRecognizerFilter and word2vec synonyms over generated corpora.
  analysis_misc)
    MAIN=AnalysisMiscMicro
    SRC=benchmarks/micro/java/AnalysisMiscMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-analysis-common" ;;
  # M12: the language modules' filters over generated corpora; each module's
  # third-party jars join through M12_THIRDPARTY.
  analysis_m12)
    MAIN=AnalysisM12Micro
    SRC=benchmarks/micro/java/AnalysisM12Micro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-analysis-common lucene-analysis-phonetic lucene-analysis-stempel lucene-analysis-morfologik lucene-analysis-kuromoji lucene-analysis-nori lucene-analysis-smartcn"
    M12_THIRDPARTY=1 ;;
  # M12 T12.4: analysis-icu over ICU4J (its jar joins through ICU_THIRDPARTY).
  analysis_icu)
    MAIN=AnalysisIcuMicro
    SRC=benchmarks/micro/java/AnalysisIcuMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-analysis-common lucene-analysis-icu"
    ICU_THIRDPARTY=1 ;;
  # analysis_icu taken apart: the tokenizers alone and each ICU kernel alone.
  analysis_icu_parts)
    MAIN=AnalysisIcuPartsMicro
    SRC=benchmarks/micro/java/AnalysisIcuPartsMicro.java
    EXTRA_SRC="benchmarks/micro/java/AnalysisIcuMicro.java benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-analysis-common lucene-analysis-icu"
    ICU_THIRDPARTY=1 ;;
  # M12 T12.5: analysis-opennlp over OpenNLP (its jars join through
  # OPENNLP_THIRDPARTY).
  analysis_opennlp)
    MAIN=AnalysisOpenNlpMicro
    SRC=benchmarks/micro/java/AnalysisOpenNlpMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-analysis-common lucene-analysis-opennlp"
    OPENNLP_THIRDPARTY=1 ;;
  # M13 T13.1: the classic and multi-field query parsers over the fixture corpus.
  queryparser)
    MAIN=QueryParserMicro
    SRC=benchmarks/micro/java/QueryParserMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-analysis-common lucene-queries lucene-queryparser" ;;
  # M13 T13.5: MemoryIndex build, build-then-search and frozen search.
  memory_index)
    MAIN=MemoryIndexMicro
    SRC=benchmarks/micro/java/MemoryIndexMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-analysis-common lucene-queries lucene-queryparser lucene-memory" ;;
  # M13 T13.2: the unified and fast vector highlighters.
  highlight)
    MAIN=HighlightMicro
    SRC=benchmarks/micro/java/HighlightMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-analysis-common lucene-queries lucene-queryparser lucene-memory lucene-highlighter"
    # One process per case: in one JVM, fvh and uh_tv inherited the JIT's
    # profile of the cases run before them (the term-vector and stored-field
    # reads they share) and their Java times wandered 1.3-2.9x across reps.
    CASES="uh_postings uh_analysis uh_tv uh_phrases fvh"
    # Lucene's fvh path was still compiling 1.5 s in: 80-90 us a unit after
    # that warmup, 71-74 us after 4 s.
    BENCH_WARMUP=4000 ;;
  # M13 T13.3c: AnalyzingSuggester and FuzzySuggester, lookups and builds.
  analyzing)
    MAIN=AnalyzingSuggestMicro
    SRC=benchmarks/micro/java/AnalyzingSuggestMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-analysis-common lucene-suggest" ;;
  # M13 T13.3c: FreeTextSuggester, lookups and builds.
  freetext)
    MAIN=FreeTextSuggestMicro
    SRC=benchmarks/micro/java/FreeTextSuggestMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-analysis-common lucene-suggest" ;;
  # M13 T13.3c: AnalyzingInfixSuggester, lookups and builds.
  infix)
    MAIN=InfixSuggestMicro
    SRC=benchmarks/micro/java/InfixSuggestMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-analysis-common lucene-suggest" ;;
  # M13 T13.3e: DirectSpellChecker, SpellChecker and WordBreakSpellChecker.
  spell)
    MAIN=SpellMicro
    SRC=benchmarks/micro/java/SpellMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-suggest" ;;
  # M13 T13.4a: the facet taxonomy's writer (addCategory) and reader lookups.
  taxonomy)
    MAIN=TaxonomyMicro
    SRC=benchmarks/micro/java/TaxonomyMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-facet" ;;
  facet_counts)
    # M13 T13.4b's pair (FacetCountsMicro.java / micro_facet_counts.rs):
    # FacetsCollector, FastTaxonomyFacetCounts (dense and sparse),
    # getTopChildren and the association sums over the 200 000-document index
    # and taxonomy FacetCountsMicro builds (once) under
    # benchmarks/.corpus/facet-counts.
    MAIN=FacetCountsMicro
    SRC=benchmarks/micro/java/FacetCountsMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-facet"
    INDEX="${INDEX_ARG:-$PWD/benchmarks/.corpus/facet-counts}"
    PREP_ARGS=(build "$INDEX")
    JAVA_ARGS=(run)
    NEEDS_INDEX=1 ;;
  drill_sideways)
    # M13 T13.4c's pair (DrillSidewaysMicro.java / micro_drill_sideways.rs):
    # a sequential DrillSideways.search per DrillSidewaysScorer strategy, over
    # facet_counts' corpus (built by FacetCountsMicro, which this prep calls).
    MAIN=DrillSidewaysMicro
    SRC=benchmarks/micro/java/DrillSidewaysMicro.java
    EXTRA_SRC="benchmarks/micro/java/FacetCountsMicro.java benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-facet"
    INDEX="${INDEX_ARG:-$PWD/benchmarks/.corpus/facet-counts}"
    PREP_ARGS=(build "$INDEX")
    JAVA_ARGS=(run)
    NEEDS_INDEX=1 ;;
  value_facets)
    # M13 T13.4d1's pair (ValueFacetsMicro.java / micro_value_facets.rs):
    # LongValueFacetCounts, StringValueFacetCounts and the long/double range
    # counters over the 200 000-document index ValueFacetsMicro builds (once)
    # under benchmarks/.corpus/value-facets.
    MAIN=ValueFacetsMicro
    SRC=benchmarks/micro/java/ValueFacetsMicro.java
    EXTRA_SRC="benchmarks/micro/java/FacetCountsMicro.java benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-facet"
    INDEX="${INDEX_ARG:-$PWD/benchmarks/.corpus/value-facets}"
    PREP_ARGS=(build "$INDEX")
    JAVA_ARGS=(run)
    NEEDS_INDEX=1 ;;
  ssdv_facets)
    # M13 T13.4d2's pair (SsdvFacetsMicro.java / micro_ssdv_facets.rs): the
    # sorted-set reader state, SortedSetDocValuesFacetCounts (plain,
    # concurrent, over a reader) and RandomSamplingFacetsCollector over the
    # 200 000-document index SsdvFacetsMicro builds (once) under
    # benchmarks/.corpus/ssdv-facets.
    MAIN=SsdvFacetsMicro
    SRC=benchmarks/micro/java/SsdvFacetsMicro.java
    EXTRA_SRC="benchmarks/micro/java/FacetCountsMicro.java benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-facet"
    INDEX="${INDEX_ARG:-$PWD/benchmarks/.corpus/ssdv-facets}"
    PREP_ARGS=(build "$INDEX")
    JAVA_ARGS=(run)
    NEEDS_INDEX=1 ;;
  # The document API's indexing: IndexWriter.addDocument over three document shapes.
  doc_index)
    MAIN=DocIndexMicro
    SRC=benchmarks/micro/java/DocIndexMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java benchmarks/micro/java/SpellMicro.java"
    JAR_MODULES="lucene-core lucene-analysis-common lucene-suggest" ;;
  # M13 T13.3b: WFSTCompletionLookup and FSTCompletionLookup, lookups and builds.
  fst_lookup)
    MAIN=FstLookupMicro
    SRC=benchmarks/micro/java/FstLookupMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-suggest" ;;
  # M13 T13.3d: completion suggestions through SuggestIndexSearcher.
  completion)
    MAIN=CompletionMicro
    SRC=benchmarks/micro/java/CompletionMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-analysis-common lucene-suggest" ;;
  # M13 T13.5c: the Monitor -- register the corpus, match four documents.
  monitor)
    MAIN=MonitorMicro
    SRC=benchmarks/micro/java/MonitorMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-analysis-common lucene-queries lucene-queryparser lucene-memory lucene-monitor" ;;
  # M14 T14.2: expressions compiled from the corpus and evaluated per document.
  expressions)
    MAIN=ExpressionsMicro
    SRC=benchmarks/micro/java/ExpressionsMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-expressions"
    EXPRESSIONS_THIRDPARTY=1 ;;
  # M14 T14.3: each classifier's assignClass over forty stored texts.
  classification)
    MAIN=ClassificationMicro
    SRC=benchmarks/micro/java/ClassificationMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-analysis-common lucene-queries lucene-grouping lucene-classification" ;;
  # M14 T14.3: the replicator's per-file work -- reading a file's metadata and
  # copying it into a replica -- over fixtures/data/replicator/plain.
  replicator)
    MAIN=ReplicatorMicro
    SRC=benchmarks/micro/java/ReplicatorMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-replicator" ;;
  # M11 T11.3: SnowballFilter per language over the fixture vocabularies.
  snowball)
    MAIN=SnowballMicro
    SRC=benchmarks/micro/java/SnowballMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-analysis-common" ;;
  # M11 T11.4: HunspellStemFilter per dictionary over its fixture words.
  hunspell)
    MAIN=HunspellMicro
    SRC=benchmarks/micro/java/HunspellMicro.java
    EXTRA_SRC="benchmarks/micro/java/SweepMicro.java"
    JAR_MODULES="lucene-core lucene-analysis-common" ;;
  # The per-area sweep: one Java class, the bench name as its first argument.
  vint|bitset|lz4|direct_monotonic|checksum|analysis|vectors|automaton|quantized|fst_build|bytes_ref_hash|bkd_build)
    MAIN=SweepMicro
    SRC=benchmarks/micro/java/SweepMicro.java
    JAVA_ARGS=("$BENCH") ;;
  postings_adv|postings_freq|positions|term_seek|doc_values|norms|points|memory)
    MAIN=SweepMicro
    SRC=benchmarks/micro/java/SweepMicro.java
    JAVA_ARGS=("$BENCH")
    NEEDS_INDEX=1 ;;
  term_dict_write)
    # Write side: both engines generate the same terms and write one field's
    # term dictionary in memory, so there is no index to point at.
    MAIN=TermDictWriteMicro
    SRC=benchmarks/micro/java/TermDictWriteMicro.java ;;
  dv_merge)
    # Write side: both engines build the same four segments and time merging
    # them, each from a fresh copy.
    MAIN=DvMergeMicro
    SRC=benchmarks/micro/java/DvMergeMicro.java ;;
  points_write)
    # Write side: points at flush and at merge, both engines building their
    # own segments from the same documents.
    MAIN=PointsWriteMicro
    SRC=benchmarks/micro/java/PointsWriteMicro.java ;;
  concurrent_index)
    # Write side: one IndexWriter shared by one and by four threads, both
    # engines indexing the same documents. Needs --pin with four CPUs for the
    # four-thread cases to mean anything.
    MAIN=ConcurrentIndexMicro
    SRC=benchmarks/micro/java/ConcurrentIndexMicro.java
    JAR_MODULES="lucene-core lucene-analysis-common" ;;
  # M7's pairs (M7Micro.java): the queries parse with the fixture generators'
  # own grammars, so those sources compile alongside; spans need lucene-queries.
  m7_fixture|query_builder|stored_fields_write|m7_corpus|similarity|sort_pruning)
    MAIN=M7Micro
    SRC=benchmarks/micro/java/M7Micro.java
    EXTRA_SRC="fixtures/src/GenM7Queries.java fixtures/src/GenSimilaritySearch.java fixtures/src/GenStoredFieldsDeflate.java benchmarks/java-runner/src/BenchRunner.java"
    JAR_MODULES="lucene-core lucene-queries"
    JAVA_ARGS=("$BENCH")
    case "$BENCH" in m7_corpus|similarity|sort_pruning) NEEDS_INDEX=1 ;; esac ;;
  # M9's geo pair (GeoMicro.java / micro_geo.rs): the Tessellator, Component2D
  # build/relate/contains/intersectsTriangle and haversin, over the inputs of
  # the fixtures under fixtures/data/geo/ (read from the repository root).
  geo)
    MAIN=GeoMicro
    SRC=benchmarks/micro/java/GeoMicro.java ;;
  geo3d)
    # M9 T9.4's pair (Geo3dMicro.java / micro_geo3d.rs): geo3d shape
    # construction, x/y/z cell relations, membership and distances, over
    # inputs both sides draw from the same SplitMix64 stream.
    MAIN=Geo3dMicro
    SRC=benchmarks/micro/java/Geo3dMicro.java
    JAR_MODULES="lucene-core lucene-spatial3d" ;;
  spatial_extras)
    # M9 T9.5's pair (SpatialExtrasMicro.java / micro_spatial_extras.rs): RPT
    # indexing a Geo3D polygon, RPT intersects, BBox with the overlap-ratio
    # similarity, heatmaps and date ranges over the 100 000-document index
    # SpatialExtrasMicro builds (once) under benchmarks/.corpus/spatial-extras.
    MAIN=SpatialExtrasMicro
    SRC=benchmarks/micro/java/SpatialExtrasMicro.java
    JAR_MODULES="lucene-core lucene-analysis-common lucene-spatial3d lucene-spatial-extras"
    SPATIAL_JARS=1
    INDEX="${INDEX_ARG:-$PWD/benchmarks/.corpus/spatial-extras}"
    PREP_ARGS=(build "$INDEX")
    JAVA_ARGS=(run)
    NEEDS_INDEX=1 ;;
  geo3d_points)
    # M9 T9.4's query pair (Geo3dPointsMicro.java / micro_geo3d_points.rs):
    # PointInGeo3DShapeQuery and the distance sorts over the 300 000-point
    # index Geo3dPointsMicro builds (once) under benchmarks/.corpus/geo3d-points.
    MAIN=Geo3dPointsMicro
    SRC=benchmarks/micro/java/Geo3dPointsMicro.java
    JAR_MODULES="lucene-core lucene-analysis-common lucene-spatial3d"
    INDEX="${INDEX_ARG:-$PWD/benchmarks/.corpus/geo3d-points}"
    PREP_ARGS=(build "$INDEX")
    JAVA_ARGS=(run)
    NEEDS_INDEX=1 ;;
  geo_points)
    # Both engines read the million-point index GeoPointsMicro builds (once)
    # under benchmarks/.corpus/geo-points, with its query set beside it.
    MAIN=GeoPointsMicro
    SRC=benchmarks/micro/java/GeoPointsMicro.java
    JAR_MODULES="lucene-core lucene-analysis-common"
    INDEX="${INDEX_ARG:-$PWD/benchmarks/.corpus/geo-points}"
    PREP_ARGS=(build "$INDEX")
    JAVA_ARGS=(run)
    NEEDS_INDEX=1 ;;
  geo_shapes)
    # Both engines read the 200 000-shape index GeoShapesMicro builds (once)
    # under benchmarks/.corpus/geo-shapes, with its polygons and query set.
    MAIN=GeoShapesMicro
    SRC=benchmarks/micro/java/GeoShapesMicro.java
    JAR_MODULES="lucene-core lucene-analysis-common"
    INDEX="${INDEX_ARG:-$PWD/benchmarks/.corpus/geo-shapes}"
    PREP_ARGS=(build "$INDEX")
    JAVA_ARGS=(run)
    NEEDS_INDEX=1 ;;
  join)
    # M10 T10.2's pair (JoinMicro.java / micro_join.rs): the block-join
    # queries and the ToParentBlockJoinSortField sort over the 60 000-block
    # index JoinMicro builds (once) under benchmarks/.corpus/join.
    MAIN=JoinMicro
    SRC=benchmarks/micro/java/JoinMicro.java
    JAR_MODULES="lucene-core lucene-analysis-common lucene-join"
    INDEX="${INDEX_ARG:-$PWD/benchmarks/.corpus/join}"
    PREP_ARGS=(build "$INDEX")
    JAVA_ARGS=(run)
    NEEDS_INDEX=1 ;;
  query_join)
    # M10 T10.3's pair (QueryJoinMicro.java / micro_query_join.rs): JoinUtil's
    # query-time joins over the 200 000-document index QueryJoinMicro builds
    # (once) under benchmarks/.corpus/query-join.
    MAIN=QueryJoinMicro
    SRC=benchmarks/micro/java/QueryJoinMicro.java
    JAR_MODULES="lucene-core lucene-analysis-common lucene-join"
    INDEX="${INDEX_ARG:-$PWD/benchmarks/.corpus/query-join}"
    PREP_ARGS=(build "$INDEX")
    JAVA_ARGS=(run)
    NEEDS_INDEX=1 ;;
  grouping)
    # M10 T10.4's pair (GroupingMicro.java / micro_grouping.rs): grouping
    # searches, block grouping, distinct values and grouped facets over the
    # 200 000-document index GroupingMicro builds (once) under
    # benchmarks/.corpus/grouping.
    MAIN=GroupingMicro
    SRC=benchmarks/micro/java/GroupingMicro.java
    JAR_MODULES="lucene-core lucene-analysis-common lucene-grouping"
    INDEX="${INDEX_ARG:-$PWD/benchmarks/.corpus/grouping}"
    PREP_ARGS=(build "$INDEX")
    JAVA_ARGS=(run)
    NEEDS_INDEX=1 ;;
  function)
    # M10 T10.5's pair (FunctionMicro.java / micro_function.rs): function,
    # function-score, range and match queries over the 200 000-document index
    # FunctionMicro builds (once) under benchmarks/.corpus/function.
    MAIN=FunctionMicro
    SRC=benchmarks/micro/java/FunctionMicro.java
    JAR_MODULES="lucene-core lucene-analysis-common lucene-queries"
    INDEX="${INDEX_ARG:-$PWD/benchmarks/.corpus/function}"
    PREP_ARGS=(build "$INDEX")
    JAVA_ARGS=(run)
    NEEDS_INDEX=1 ;;
  ints_ref_fst_enum)
    MAIN=IntsRefFstEnumMicro
    SRC=benchmarks/micro/java/IntsRefFstEnumMicro.java
    EXTRA_SRC=benchmarks/micro/java/SweepMicro.java
    JAR_MODULES="lucene-core"
    for check in ints_enum_ceil ints_enum_exact ints_enum_floor ints_enum_next; do
      REPORT_ARGS+=(--require-check "$check")
    done ;;
  stphrase)
    MAIN=SimpleTextPhraseMicro
    SRC=benchmarks/micro/java/SimpleTextPhraseMicro.java
    JAR_MODULES="lucene-core lucene-codecs"
    INDEX="${INDEX_ARG:-$PWD/fixtures/data/simpletext_phrase_scorer/deep/merged/index}"
    CASES="cx_st_phrase_exact_all cx_st_phrase_exact_top cx_st_phrase_sloppy_all cx_st_phrase_sloppy_top cx_st_phrase_repeat_all cx_st_phrase_repeat_top"
    NEEDS_INDEX=1 ;;
  stsi)
    MAIN=SimpleTextSiMicro
    SRC=benchmarks/micro/java/SimpleTextSiMicro.java
    JAR_MODULES="lucene-core lucene-codecs"
    INDEX="${INDEX_ARG:-$PWD/fixtures/data/simpletext_index_sort}"
    NEEDS_INDEX=1 ;;
  stmerge)
    # Direct specialized FieldsConsumer.merge over the exact fixture bytes
    # used by Rust; excludes IndexWriter's other formats and reader opening.
    MAIN=StMergeMicro
    SRC=benchmarks/micro/java/StMergeMicro.java
    JAR_MODULES="lucene-core lucene-codecs"
    INDEX="${INDEX_ARG:-$PWD/fixtures/data/codecs_sharedterms_merge/unsorted}"
    NEEDS_INDEX=1 ;;
  codecs)
    # M14.g's pair (CodecsMicro.java / micro_codecs.rs): lucene-codecs'
    # non-default formats, each over the index CodecsMicro builds (once) in
    # its own subdirectory of benchmarks/.corpus/codecs -- so far
    # BloomFilteringPostingsFormat's filtered seekExact (bloom/) and
    # FSTPostingsFormat's and DirectPostingsFormat's seekExact (memory/) and
    # HnswBitVectorsFormat's top-10 graph search (bitvectors/) and
    # BlockTreeOrdsPostingsFormat's seekExact by term and by ord (ords/) and
    # UniformSplitPostingsFormat's seekExact (usplit/) and
    # STUniformSplitPostingsFormat's seekExact on a shared field (stsplit/).
    MAIN=CodecsMicro
    SRC=benchmarks/micro/java/CodecsMicro.java
    JAR_MODULES="lucene-core lucene-codecs"
    INDEX="${INDEX_ARG:-$PWD/benchmarks/.corpus/codecs}"
    PREP_ARGS=(build "$INDEX")
    JAVA_ARGS=(run)
    NEEDS_INDEX=1 ;;
  sandbox)
    # M14.f's pair (SandboxMicro.java / micro_sandbox.rs): lucene-sandbox's
    # multi-range queries, FloatPointNearestNeighbor and (over its text/
    # index) the scoring queries and the large-hits collector, and (over
    # its idv/ index) IDVersionSegmentTermsEnum.seekExact(id, minVersion),
    # and KMeans.cluster over kmeans/vectors.bin, and the facet package's
    # range counts and histograms, over the 200 000-document index
    # SandboxMicro builds (once) under benchmarks/.corpus/sandbox.
    MAIN=SandboxMicro
    SRC=benchmarks/micro/java/SandboxMicro.java
    JAR_MODULES="lucene-core lucene-sandbox lucene-facet lucene-queries"
    INDEX="${INDEX_ARG:-$PWD/benchmarks/.corpus/sandbox}"
    PREP_ARGS=(build "$INDEX")
    JAVA_ARGS=(run)
    NEEDS_INDEX=1 ;;
  queries)
    # M10 T10.6's pair (QueriesMicro.java / micro_queries.rs): interval
    # queries over the 200 000-document index QueriesMicro builds (once)
    # under benchmarks/.corpus/queries.
    MAIN=QueriesMicro
    SRC=benchmarks/micro/java/QueriesMicro.java
    JAR_MODULES="lucene-core lucene-analysis-common lucene-queries"
    INDEX="${INDEX_ARG:-$PWD/benchmarks/.corpus/queries}"
    PREP_ARGS=(build "$INDEX")
    JAVA_ARGS=(run)
    NEEDS_INDEX=1 ;;
  aggs)
    # M10 stage 3's pair (AggsMicro.java / micro_aggs.rs): the plugin's
    # `terms` aggregation behind a dense filter against OpenSearch's
    # collector for it, over the benchmark corpus (benchmarks/.corpus/merged).
    MAIN=AggsMicro
    SRC=benchmarks/micro/java/AggsMicro.java
    NEEDS_INDEX=1 ;;
  pfor_decode)
    MAIN=org.apache.lucene.codecs.lucene104.PForUtilMicro
    SRC=benchmarks/micro/java/org/apache/lucene/codecs/lucene104/PForUtilMicro.java ;;
  *) echo "bench-micro: no Java counterpart for $BENCH" >&2; exit 2 ;;
esac

WARMUP="${WARMUP:-${BENCH_WARMUP:-1500}}"

# shellcheck source=scripts/lib-lucene-jars.sh
source "$(dirname "$0")/lib-lucene-jars.sh"
# shellcheck disable=SC2086  # JAR_MODULES is a deliberate word list
CP=$(lucene_classpath ${JAR_MODULES:-lucene-core})
# lucene-spatial-extras' Spatial4j and S2 jars.
if [ -n "${SPATIAL_JARS:-}" ]; then CP="$CP:$(thirdparty_classpath "${SPATIAL_EXTRAS_DEPS[@]}")"; fi
# The M12 modules' third-party jars (Commons Codec, Morfologik).
if [ -n "${ICU_THIRDPARTY:-}" ]; then CP="$CP:$(thirdparty_classpath "${ICU_DEPS[@]}")"; fi
if [ -n "${OPENNLP_THIRDPARTY:-}" ]; then CP="$CP:$(thirdparty_classpath "${OPENNLP_DEPS[@]}")"; fi
# lucene-expressions' ANTLR runtime and ASM.
if [ -n "${EXPRESSIONS_THIRDPARTY:-}" ]; then CP="$CP:$(thirdparty_classpath "${EXPRESSIONS_DEPS[@]}")"; fi
if [ -n "${M12_THIRDPARTY:-}" ]; then
  CP="$CP:$(thirdparty_classpath "${PHONETIC_DEPS[@]}" "${MORFOLOGIK_DEPS[@]}")"
fi

OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT

# Refuse to measure on a busy machine -- see bench-compare.sh's own note; two
# M1 measurement rounds were thrown away to background load before that guard.
LOAD=$(cut -d' ' -f1 /proc/loadavg)
MAXLOAD="${BENCH_MAX_LOAD:-1.5}"
if awk "BEGIN{exit !($LOAD > $MAXLOAD)}"; then
  echo "bench-micro: refusing to measure -- 1-minute load average is $LOAD (limit $MAXLOAD)." >&2
  ps -eo pcpu,comm --sort=-pcpu | head -4 | sed 's/^/    /' >&2
  echo "  Wait for the machine to settle, or override with BENCH_MAX_LOAD=<n>." >&2
  exit 3
fi

echo "bench-micro: building" >&2
( cd benchmarks/rust-runner && cargo build --release --quiet --bin "${RUST_BIN:-micro}" )
# Where cargo actually put it. `scripts/docker-test.sh` exports
# CARGO_TARGET_DIR, so inside the container the build lands there and NOT in
# `benchmarks/rust-runner/target` -- which, being a bind mount of the host
# repo, may still hold a months-old host build. This script used to run that
# stale binary and report its timings as the current engine's; `c42-readpath-perf`
# found it doing exactly that, reporting a figure identical to three digits
# across a change that moved the operation by 20%.
RUST_TARGET_DIR="${CARGO_TARGET_DIR:-benchmarks/rust-runner/target}"
# shellcheck disable=SC2086  # EXTRA_SRC is a deliberate word list
javac -nowarn -cp "$CP" -d "$OUT/classes" "$SRC" ${EXTRA_SRC:-}

# A bench whose Java side builds the shared input first (once).
if [ -n "${PREP_ARGS+x}" ]; then
  echo "bench-micro: preparing ${PREP_ARGS[*]}" >&2
  java -cp "$CP:$OUT/classes" "$MAIN" "${PREP_ARGS[@]}"
fi

# Every case but `index` measures a read path out of the shared `micro` binary,
# selected by name and pointed at a prebuilt index; `index` has its own binary
# because it builds an index rather than reading one, so it takes neither.
RUST_ARGS=("$BENCH" ${NEEDS_INDEX:+"$INDEX"})
if [ -n "${RUST_BIN:-}" ]; then RUST_ARGS=(); else RUST_BIN=micro; fi

PINCMD=(taskset -c "$PIN")
if ! command -v taskset >/dev/null; then
  # Git Bash on Windows has no taskset: unpinned runs migrate between P- and
  # E-cores and read ~25% slow on one engine, in runs that look clean.
  echo "bench-micro: WARNING -- no taskset, running UNPINNED; run it through scripts/docker-test.sh" >&2
  PINCMD=()
fi

# Interleave the two engines rather than running all of one then all of the
# other. A run takes minutes and this machine drifts over that: whatever the
# drift is, alternating makes it fall on both sides equally instead of biasing
# whichever went second.
#
# A bench that sets CASES runs each of those cases in a process of its own
# (`MICRO_CASE` / `-DmicroCase`), still alternating the engines, so no case is
# measured in a JVM the cases before it have JIT-compiled for their own paths.
for rep in $(seq 1 "$REPS"); do
  : > "$OUT/rust.$rep.tsv"
  : > "$OUT/java.$rep.tsv"
  for case in ${CASES:-""}; do
    echo "bench-micro: rep $rep/$REPS rust ($BENCH${case:+ $case})" >&2
    env ${case:+MICRO_CASE="$case"} MICRO_WARMUP_MS="$WARMUP" MICRO_MEASURE_MS="$MEASURE" \
      "${PINCMD[@]}" "$RUST_TARGET_DIR/release/$RUST_BIN" "${RUST_ARGS[@]}" \
      >> "$OUT/rust.$rep.tsv"

    echo "bench-micro: rep $rep/$REPS java ($BENCH${case:+ $case})" >&2
    "${PINCMD[@]}" java --add-modules jdk.incubator.vector \
      -DwarmupMs="$WARMUP" -DmeasureMs="$MEASURE" ${case:+"-DmicroCase=$case"} \
      -cp "$CP:$OUT/classes" "$MAIN" ${JAVA_ARGS[@]+"${JAVA_ARGS[@]}"} ${NEEDS_INDEX:+"$INDEX"} >> "$OUT/java.$rep.tsv"
  done
done

python3 scripts/bench-micro-report.py "$OUT" "$REPS" "${REPORT_ARGS[@]}"
