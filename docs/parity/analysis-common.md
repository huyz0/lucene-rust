# lucene-analysis-common

[Index](../parity.md). M11, one row per package; per-class status is `docs/inventory/lucene-analysis-common.tsv`. Tests: `GenAnalysisCommon` -> `tests/analysis_common_fixtures.rs`.

| Java | Rust | Status |
|---|---|---|
| `analysis/core/{UpperCaseFilter, DecimalDigitFilter, TypeTokenFilter, WhitespaceAnalyzer, UnicodeWhitespaceAnalyzer, SimpleAnalyzer, StopAnalyzer}` | `lucene-analysis/src/core_analysis/mod.rs`, `lucene-analysis/src/core_analysis/filters.rs::{UpperCaseFilter, DecimalDigitFilter, TypeTokenFilter}`, `lucene-analysis/src/core_analysis/analyzers.rs::{WhitespaceAnalyzer, SimpleAnalyzer, StopAnalyzer}` | **ported** M11 T11.6. Differs: case and digit folding map code points of the `String` (no simple mapping crosses the BMP). |
| `analysis/core/FlattenGraphFilter` | `lucene-analysis/src/core_analysis/flatten_graph_filter.rs::FlattenGraphFilter` | **ported** M11 T11.6, state for state. Tests: seven canned graphs vs Lucene (unit). |
| `analysis/util/{CharTokenizer, CharacterUtils, UnicodeProps}`, `analysis/core/{WhitespaceTokenizer, LetterTokenizer, UnicodeWhitespaceTokenizer}`, `util/RollingBuffer` | `lucene-analysis/src/util/mod.rs`, `lucene-analysis/src/util/char_tokenizer.rs::CharTokenizer`, `lucene-analysis/src/util/rolling_buffer.rs::RollingBuffer` | **ported** M11 T11.6: `CharacterUtils.fill`'s 4096-unit buffer; a tokenizer is `CharTokenizer` over a predicate. |
| *(JDK)* `java.lang.Character` categories | `lucene-analysis/src/java_character.rs`, `lucene-analysis/src/java_character_tables.rs` | **generated** from UCD 16.0.0 (`tools/gen_java_character_tables.py`; identical to JDK 25's `getType`). |
