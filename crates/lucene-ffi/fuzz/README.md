# lucene-ffi fuzz targets

libFuzzer targets over `lucene-ffi`'s **C ABI** — the exported `ffi_*`
functions, declared here with `extern "C"` exactly as a C or JNI caller sees
them — under AddressSanitizer (cargo-fuzz's default). M2 task T2.5.

| target | drives |
|---|---|
| `jvm_search` | `ffi_jvm_reader_search` with an arbitrary query blob, `top_n` and count limit, over a real two-segment Java-written index |
| `jvm_open_reader` | `ffi_open_jvm_reader` with arbitrary `SegmentInfos` bytes, generation and expected segment sizes; whatever opens is searched and closed |
| `jvm_live_docs` | `ffi_jvm_reader_set_live_docs` with arbitrary words, then searches and counts under whatever was accepted |
| `boolean_clause_arrays` | the occur-tagged clause-array format through `ffi_search_boolean_query_multi_segment`: arbitrary occurs, kinds, parents and params |

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
boolean and nested-wrapper blobs); `corpus/` is what a run grows and is not
committed. CI (`fuzz` job) runs each target for 60 seconds per push.

Outside the workspace because libFuzzer needs nightly and the workspace pins
a stable toolchain.
