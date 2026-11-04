# lucene-sparse -- neural-search's Seismic sparse index

[Index](../parity.md). Not a Lucene module: OpenSearch's neural-search plugin
(3.9.0.0) writes these files for `sparse_vector` fields of an `index.sparse`
index (codec `Sparse10010Codec`, `lucene` engine). Byte layout and algorithm:
`/mnt/project-files/plan/seismic-format.md` in the project notes; source of
truth `org/opensearch/neuralsearch/sparse/` in neural-search.

| Java | Rust | Status |
|---|---|---|
| -- (neural-search `codec/SparseCodec`, `SparseTermsLuceneReader`, `data/DocumentCluster`) | `lucene-sparse/src/postings.rs::{ClusteredPostings, Cluster, CODEC_NAME, TERMS_EXTENSION, POSTINGS_EXTENSION}`, `lucene-sparse/src/lib.rs::{register_codec, CODEC_NAME}` | **rust-only** (port of neural-search, read side). `.sit` term dictionary and `.sip` clusters with summaries; headers checked, checksums not (as neural-search). Gap: no writer (clustering), no `native` engine (`.nsparse`). Tests: `scripts/gen-seismic-fixture.py` -> `tests/seismic_fixtures.rs`; unit tests for truncated and out-of-range files. |
| -- (neural-search `query/SparseQueryWeight`, `SeismicBaseScorer`, `OrderedPostingWithClustersScorer`, `ExactMatchScorer`, `SparseAnnQueryBuilder`) | `lucene-sparse/src/search.rs::{search, SeismicQuery, SeismicField, SegmentView}` | **rust-only** (port of neural-search). Same hits and bit-equal scores as OpenSearch 3.9.0 on 120 query/index cases (flushed compound and merged segments; `top_n`, `k`, `heap_factor`, post-filter and exact filtered paths). Skip heap is `java.util.PriorityQueue`'s order. Differs: tokens tied in weight at the `top_n` cut are taken by token, not `HashMap` order. Gap: segments under `approximate_threshold` (plain `FeatureField` query) are not handled here. Tests: `seismic_hits_match_neural_search`. |
| -- (neural-search `data/SparseVector`, `quantization/ByteQuantizer`, `common/BinaryVectorUtils`) | `lucene-sparse/src/vector.rs::{SparseVector, quantize, decode_forward, TOKEN_MODULUS}` | **rust-only** (port of neural-search). Folding mod 32768 with max merge, `Math.round` half-up quantization in `f32`, big-endian forward-index pairs. Tests: unit tests; the fixture test above. |
