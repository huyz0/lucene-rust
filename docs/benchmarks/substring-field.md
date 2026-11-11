# Substring field (`RustSubstring` doc values)

`scripts/bench-substring-field.sh` (2026-11-11, 4 vCPU Xeon 2.3 GHz cloud
VM, one thread, query cache off). Corpus: 6,454,808 log lines (856 MB),
the first 250 MB of each of Loghub's BGL, HDFS, OpenSSH and Thunderbird,
interleaved by `Random(7)`, one line per document. Stock Lucene 10.5.1
writes two indexes. `text` is a `TextField` with StandardAnalyzer, searched
by token wildcard, regexp, term or AND. `wildcard` is OpenSearch's
`wildcard` field layout: anchored 3-grams (docs only) plus sorted-set doc
values, searched by `WildcardMatchingQuery` (gram approximation, then
every candidate's value through the pattern). This port writes `msg` as
BINARY doc values in the `RustSubstring` format and searches it with
`SubstringQuery`. Every index flushes 256 MB buffers without merging, then
force-merges to one segment.

## The format

It is opt-in per field (`Lucene90DocValuesFormat::rust_substring(shift)`).
Stock Lucene's files are kept, and the format name stops stock Lucene from
reading the field. A BINARY entry (minLength `-1001`) holds:

- Documents in blocks of `2^shift` doc ids. Each block is LZ4-compressed
  (Lucene's LZ4, fast table) and holds its doc offsets, lengths and values
  end to end. Block addresses are DirectMonotonic.
- A gram table: one 16-byte row per gram (key, block count, offset),
  binary-searched in place. A gram is three symbols over ASCII-folded
  bytes, with a boundary symbol at each end of the value.
- Gram postings as block ids, either 128-delta FOR chunks or a bitset,
  whichever is smaller.

A query intersects its grams' block lists, rarest first. It skips a gram
whose postings cost more to decode than the blocks it would rule out. Then
it decompresses each candidate block once and tests its values. A
substring is searched over the whole block with `memchr::memmem`. A
wildcard or regexp first looks for its longest required literal, then runs
the test only on the values that hold it. Wildcards match their
`*`-separated segments leftmost first, without backtracking. Regexps run
the cached DFA and stop at a dead or all-accepting state.

Case-insensitive search follows OpenSearch. A wildcard lower-cases both
the pattern and the value; a regexp uses `ASCII_CASE_INSENSITIVE`.
`BinaryDocValues` still returns every value, and merges, CheckIndex and
deletes go through the usual doc values paths.

## Size and write cost

| index | MiB | index s | docs/s | force merge s |
|---|---:|---:|---:|---:|
| `text` (stock) | 192.2 | 26.6 | 242k | 7.1 |
| `wildcard` (stock, OpenSearch layout) | 784.9 | 74.4 | 87k | 15.7 |
| plain BINARY doc values | 837.7 | 8.9 | 727k | 4.0 |
| `RustSubstring` shift 3 (8 docs/block) | 681.6 | 27.1 | 238k | 22.7 |
| `RustSubstring` shift 4 (16) | 503.9 | 25.4 | 254k | 20.4 |
| **`RustSubstring` shift 5 (32, default)** | **382.6** | 22.7 | 285k | 18.2 |
| `RustSubstring` shift 6 (64) | 303.4 | 21.4 | 302k | 15.7 |

LZ4 high compression saved only 5% (365.2 MiB at shift 5, 283.0 at
shift 6) for 20% more write time, so blocks use the fast table. The
earlier prototype's 872 MiB came from document-level postings. Block-level
postings are 4-7x smaller.

## Latency, median ms

Hits are the same as the `wildcard` field's on every query, both in count
and in a hash of the hits' line numbers. `text` answers a related token
query, not the same one, so its hit counts differ. It is a reference for
what a text field costs.

| query | pattern | text | wildcard | shift 4 | **shift 5** | hits |
|---|---|---:|---:|---:|---:|---:|
| rare-timeout | contains `timeout`, ci | 50.5 | 4.79 | 2.22 | **3.07** | 1,279 |
| rare-SocketTimeout | contains `SocketTimeout` | 54.4 | 0.069 | 0.051 | **0.058** | 6 |
| rare-blk | contains `blk_-16089` | 58.7 | 1.09 | 1.12 | **0.93** | 268 |
| rare-ip | contains `10.250.19.102` | 0.002 | 1.65 | 4.72 | **12.4** | 116 |
| rare-postdrop | contains `postdrop` | 54.1 | 2.70 | 1.38 | **1.38** | 2,128 |
| ci-sockettimeoutexception | contains, ci | 54.0 | 0.220 | 0.023 | **0.026** | 6 |
| common-error | contains `error`, ci | 55.2 | 228 | 87.5 | **89.3** | 258,149 |
| common-failed-password | contains `failed password`, ci | 0.76 | 147 | 163 | **167** | 197,587 |
| common-parity | contains `parity error corrected` | 0.65 | 46.9 | 18.5 | **15.6** | 64,933 |
| common-auth-failure | contains `authentication failure` | 0.95 | 183 | 142 | **145** | 214,972 |
| suffix-corrected | `*corrected` | 0.002 | 60.9 | 22.0 | **21.2** | 74,927 |
| prefix-dec10 | `Dec 10 07:2*` | - | 0.110 | 1.74 | **0.74** | 82 |
| wild-user-r??t | `*user=r??t*` | 3.78 | 136 | 138 | **143** | 176,720 |
| wild-sshd | `*sshd[2420?]*` | 0.023 | 0.627 | 1.49 | **2.75** | 187 |
| ci-kernel-instruction | `*kernel info*instruction*`, ci | 1.54 | 87.7 | 50.6 | **43.2** | 79,592 |
| re-terminating | `.*blk_-?[0-9]+ terminating` | 0.001 | 294 | 289 | **301** | 316,513 |
| re-rhost | `.*rhost=[0-9]+\.95\.[0-9.]+ .*` | 8.90 | 0.294 | 2.74 | **3.01** | 24 |
| re-timeout | `.*[Tt]imeout.*` | 52.9 | 4.19 | 1.97 | **3.33** | 1,278 |
| re-ci-password | `.*failed PASSWORD.*`, ci | 0.75 | 148 | 187 | **194** | 197,587 |

Shift 3 and 6 were run too (`/mnt/project-files/perf/substring-field/`).
Shift 3 is 10-35% faster on common substrings and needs 78% more disk.
Shift 6 is 3x slower on number-heavy needles (`rare-ip` 35 ms).

## Recommendation

The default is shift 5, opt-in per field. It takes half the disk of the
`wildcard` field (383 vs 785 MiB, 2x a text field) and indexes 3.3x
faster. Substrings with distinctive grams take 0.03-3 ms, against 50-58 ms
for a leading wildcard on a text field, and match or beat the `wildcard`
field. Common substrings and wildcards are 1.3-3x faster than it, or
within 15% (`failed password`, `*user=r??t*`, the `terminating` regexp).
The cost is a single scan of the candidate blocks: decompression is about
1.7 µs per 32-line block.

Where it loses:

- Rare matches whose grams are common: digits, dots and `?`
  (`10.250.19.102` 12 ms vs 1.6, `*sshd[2420?]*` 2.8 vs 0.6, the `rhost`
  regexp 3.0 vs 0.3, `Dec 10 07:2*` 0.7 vs 0.1). Blocks hold all of the
  grams, in different lines, so they are read for nothing (`10.250.19.102`:
  5,747 candidate blocks for 116 hits). Shift 4 roughly halves these
  (4.7 ms for the IP) for 32% more disk.
- Case-insensitive regexps over common needles (194 vs 148 ms). The DFA
  runs on each of the 200k values that hold the needle.
