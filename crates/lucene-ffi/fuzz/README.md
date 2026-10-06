# lucene-ffi fuzz targets

libFuzzer targets over `lucene-ffi`'s **C ABI** — the exported `ffi_*`
functions, declared here with `extern "C"` exactly as a C or JNI caller sees
them — under AddressSanitizer (cargo-fuzz's default). M2 task T2.5.

| target | drives |
|---|---|
| `jvm_search` | `ffi_jvm_reader_search` with an arbitrary query blob, `top_n` and count limit, over a real two-segment Java-written index |
| `jvm_search_sorted` | `ffi_jvm_reader_search_sorted` with an arbitrary query blob and sort blob (split by the input's third byte): every sort-key type -- the geo-distance keys' comparators, origins and units included -- `search_after`, the options, slices and index-sort flags, over the same index |
| `jvm_open_reader` | `ffi_open_jvm_reader` with arbitrary `SegmentInfos` bytes, generation and expected segment sizes; whatever opens is searched and closed |
| `jvm_live_docs` | `ffi_open_jvm_reader` with arbitrary live-docs words and per-segment word counts, then searches and counts under whatever was accepted |
| `boolean_clause_arrays` | the occur-tagged clause-array format through `ffi_search_boolean_query_multi_segment`: arbitrary occurs, kinds, parents and params |
| `read_planet_object` | geo3d's `readPlanetObject` (the bytes a `Geo3dBinaryCodec` shape doc value holds) straight through `lucene-util`, not the C ABI: any input must read or fail with an `Err` -- never a panic, a stack overflow (nesting stops at 64 levels) or a count-sized allocation -- and a shape that reads must write, re-read and re-write to the same bytes. Seeds are one Java-written shape per geo3d class, from `fixtures/data/geo3d/shapes.tsv`. Run it with `-a` too: debug assertions catch a geo3d exception raised outside a `catch` |

**A caught panic is a finding.** The boundary would survive it — every entry
point is `catch_unwind`-guarded and reports `FfiStatus::Panic` — but every
input must be answered with a status a caller can act on, so each target
asserts the status is never `Panic` (9). The search targets also assert the
results are plausible: no more hits than asked for, doc IDs inside the index,
a total no smaller than the hits returned.

```
rustup toolchain install nightly --profile minimal
cargo +nightly install cargo-fuzz --locked
cd crates/lucene-ffi/fuzz
cargo +nightly fuzz run jvm_search corpus/jvm_search seeds/jvm_search -- -max_total_time=300
```

`seeds/` holds the checked-in starting inputs (a real `segments_2`, term,
boolean and nested-wrapper blobs, the geo nodes: a distance, a shape query
over every geometry, a points box in a boolean, and the M10 nodes `m10_*`: a
block join, spans, intervals with a multi-term automaton, combined fields and
a function score with every function kind); `corpus/` is what a run grows and is not
committed. CI (`fuzz` job) runs each target for 60 seconds per push.

Outside the workspace because libFuzzer needs nightly and the workspace pins
a stable toolchain.
