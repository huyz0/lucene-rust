# lucene-painless -- Painless scripts run natively

[Index](../parity.md). OpenSearch's `lang-painless` module: the plugin runs Painless's own front
end (parse, analysis, `ir`) and sends the IR (`PainlessIr.java`, `PIR1`); Rust compiles it to a
register program and runs it with Painless's runtime semantics (`def` dispatch, `DefMath`,
casts, the allowlisted classes).

**Checked before it runs.** A `script_score` script runs natively from its first document to its
last, or not at all. Before the query is sent, `lucene-painless/src/check.rs::check` (through
`ffi_jvm_painless_check`) decides statically, on the compiled and specialised program and without
running any of it, that the native runtime serves every operation exactly as Painless does: every
call, `def` field access and cast by the receiver classes its static type admits (each site
carries Painless's own answer per class, `PainlessIr.siteTable`), every `doc` read by a constant
mapped field with a native reading, every `params` read folded, and only the script class's own
methods the context serves. A refused script runs in Java (fallback `script_score_painless`); the
answer is cached per script, params and field kinds (`PainlessScore.CHECKED`), so a refused
script costs one check, not one per request. Regular expressions run natively only where Rust's
`regex` matches `java.util.regex` exactly (`api/regex.rs::translate`); others (`$`, `\b`,
back-references, lookaround, inline flags, possessive quantifiers, other flags) are refused. The check
also refuses a change to a collection view (`keySet`, `values`, `entrySet` and its entries,
`subList`, sorted sub-views: live views in Java, copies natively), a cast of one other than to
its interface, and a method reference the runtime has no target for. Past a document's values,
`doc[...]` reads `ScriptDocValues`'s reused array as Java does (stale values, empty keyword
slots, `ArrayIndexOutOfBoundsException` past its grown length).

**Never re-run.** A script that passed the check and then throws natively fails the search as it
fails in Java: the same `ScriptException` (class, message, `script_stack`, position) or, for a
negative or NaN score, `ScriptScoreQuery`'s `IllegalArgumentException`. The search is not run
again in Java (`RustQueryPhaseSearcher.scriptFailure`); the stats count it as `script_errors`,
not `native_errors`. The exception itself is Painless's: the native error names the segment,
document and score the script threw at, and the plugin runs the script once more in Java on that
one document (`PainlessScore.failure`), so the message (a helpful `NullPointerException`'s local
names included) and the Java frames in `script_stack` (`LeafDocLookup.get`,
`ScriptDocValues$Longs.get`, ...) are the JVM's own. That run has no effect beyond its result:
the check passes only scripts whose sole effect is their score. A gap the check let through (the runtime's own `Unsupported`) is the
plugin's failure, not the script's: like any other internal native failure it falls back to
Lucene, which runs the script again (score scripts are pure, so only time is lost). Tested by `jvm_painless_score.rs::scripts_score_as_opensearch_does` (each
fixture outcome, offsets included), `tests/corpus.rs` (every corpus script the check passes
throws exactly what Painless throws) and `e2e/verify_opensearch.py::script_errors` (stock and
native error responses equal, no fallback). Known gap: three corpus scripts whose Painless error
is a JVM `WrongMethodTypeException` (a `def` brace access with a string key on a list or array)
throw a different message natively; in the plugin the search still fails with Painless's own.

| Java | Rust | Status |
|---|---|---|
| `painless/ir/*` (the IR tree), `painless/phase/DefaultIRTreeToASMBytesPhase` | `lucene-painless/src/ir.rs::{decode, decode_constant, Script, Node, Constant}`, `lucene-painless/src/compile.rs::{compile, constant}`, `lucene-painless/src/vm.rs::{Program, Op, Vm, Env}` | **partial**: every IR node of Painless's own test suite but the ones listed as unsupported (static `Optional`, streams, `java.time`); the VM evaluates in Java's order with Java's exceptions. Equal to Painless over 4,897 of 4,964 scripts of its test suite, results, exception class, message and offset (`lucene-painless/tests/corpus.rs`, `tests/data/corpus.bin` from `opensearch-plugin/tools/painless-corpus`); the static check passes 4,894 of them, all of which match. |
| `painless/Def`, `DefMath`, `DefBootstrap`, `AnalyzerCaster` | `lucene-painless/src/def.rs::{arith, def_arith, compare, unary, cast, widens}`, `lucene-painless/src/types.rs::{prim, unboxed, instance_of, convert}`, `lucene-painless/src/number.rs`, `lucene-painless/src/compare.rs`, `lucene-painless/src/value.rs::Value`, `lucene-painless/src/object.rs::JObject`, `lucene-painless/src/jmap.rs::JMap`, `lucene-painless/src/lambda.rs` | **ported** for the corpus: `def` operators promote by runtime class, compound and explicit results convert as `DefMath` does, `HashMap` iteration order matches Java's hashing. |
| the allowlisted JDK classes (`java.lang`, `java.util`, `java.util.regex`, `java.util.function`) | `lucene-painless/src/api/mod.rs`, `lucene-painless/src/api/array.rs`, `lucene-painless/src/api/collection.rs`, `lucene-painless/src/api/dynamic.rs`, `lucene-painless/src/api/functional.rs`, `lucene-painless/src/api/iter.rs`, `lucene-painless/src/api/list.rs`, `lucene-painless/src/api/map.rs`, `lucene-painless/src/api/math.rs`, `lucene-painless/src/api/number.rs`, `lucene-painless/src/api/objects.rs`, `lucene-painless/src/api/regex.rs`, `lucene-painless/src/api/string.rs` | **partial**: what Painless's test suite calls; `Math` on fdlibm (`StrictMath`), so a HotSpot `Math.log` intrinsic may differ in the last bit of a `double` (scores are `float`). No streams, `java.time`, `Optional`. |
| `ScoreScript`, `LeafDocLookup`, `ScriptDocValues.{Longs, Doubles, Booleans, Strings}`, `ScoreScriptUtils` | `lucene-painless/src/api/doc.rs::{DocLookup, DocValues, DocKind}`, `lucene-painless/src/api/score.rs::{static_lookup, class_binding}` | **partial**: `doc[...]` views, `saturation`, `sigmoid`, numeric decays, `randomScore` (doc and field). Gap: date and geo decays, `termFreq`, `_source`, date fields. |
| (none: Painless compiles to bytecode the JIT specialises) | `lucene-painless/src/specialize.rs::{specialize, Facts}`, `lucene-painless/src/typed.rs::{lower, Typed, BatchEnv, BatchRegs}` | **no Java counterpart**: a per-query pass folds `params` (when the script never changes or hands it on), turns `doc['f']` reads into direct reads and types `def` arithmetic by the known classes; a script whose live operations are all on numbers and strings then runs on unboxed registers, and one with forward branches only runs a batch of documents per instruction. Checked against the VM over the corpus (2,364 typed, 381 batched, no difference). |
| OpenSearch `ScriptScoreQuery`, `ScriptScoreFunction` scorer | `lucene-ffi/src/jvm_painless_score.rs::{decode, PainlessScoreSource, NODE_PAINLESS_SCORE, ScriptFunction}`, `jvm_function_score.rs` (function tag 5) | **partial**: `script_score` with `lang: painless` as node 43, the score checked and boosted as `ScriptScorer` does, through the typed and batch machines when the script lowers. Equal to OpenSearch on 43 of 50 fixture scripts, scores bit for bit or the same exception at the same offset (`fixtures/data/painless-score`, `opensearch-plugin/tools/painless-corpus/ScoreCorpus.java`); the check refuses the other 7 (date fields, `_source`, streams, `params.doc`, a changed `params` collection or alias, a changed key set), which run in Java. Bench (Java harness `setDocument` + `execute` vs `painless_script_loop_bench`, ns/doc): arith 24.9 vs 9.1, log 33.4 vs 15.2, missing 62 vs 33, loop 327 vs 173, keyword 167 vs 68, def 37.8 vs 32. `function_score`'s `script_score` functions run on the same runtime (`ScriptScoreFunction`: the sub-query's score as `_score`, the script's `double` refused when negative, `NaN` passed on); in a `function_score` holding a script the query's own errors (a missing field value, a negative field value score, an invalid final score) fail the search as OpenSearch's do, without running it again in Java (`jvm_painless_score.rs::script_functions_score_as_script_score_does`, `e2e/verify_opensearch.py::script_errors`). Gap: the other script contexts run in Java. |
| OpenSearch `ScriptQuery` (`FilterScript`) | `lucene-ffi/src/jvm_painless_score.rs::{decode_filter, NODE_PAINLESS_FILTER}` | **partial**: `script` with `lang: painless` as node 45 on the same runtime and check: the script's own `params`, a `FunctionMatchQuery` at `ScriptQuery`'s match cost 1000, `boolean` results batched as `1.0`/`0.0`. Equal to OpenSearch on the fixture's 17 filter scripts (hits, or the same exception at the same offset); a filter's exception re-runs on its one document in Java for Painless's exact `ScriptException` (`e2e/verify_opensearch.py::script_errors`). |
| OpenSearch `ScriptSortBuilder` `number` (`NumberSortScript`) | `lucene-ffi/src/jvm_painless_score.rs::PainlessSort`, `jvm_reader.rs::SORT_SCRIPT` | **partial**: a `_script` sort key of type `number` on the same runtime and check: a custom comparator over the script's `double` (never missing, `Double.compare` order), hit values as sortable longs. Equal to OpenSearch's per-document values on the fixture's 10 sort scripts (order, values, or the same exception at the same offset); errors name the segment by doc base. Gap: scripts reading `_score`, nested keys and `string` sorts run in Java. |
