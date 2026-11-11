# Logs preset

The plugin registers two component templates for log indices. They change
nothing until an index template names them in `composed_of`:

| Template | What it sets |
|---|---|
| `lucene_rust_logs@settings` | the Rust engine (`index.lucene_rust.engine`), index sort `@timestamp` descending, paged zstd stored fields (`index.lucene_rust.stored_fields.paged_zstd`), zstd term blocks (`index.lucene_rust.zstd_terms`), adaptive doc values (`index.lucene_rust.adaptive_doc_values`) |
| `lucene_rust_logs@mappings` | `@timestamp` as `date` with a doc-values skip index (`skip_list`) beside its points; `host.name`, `service.name`, `log.level` as `keyword` with doc values only (`index: false`); `http.response.status_code`, `event.duration` and every dynamically mapped number as doc values with a skip index only; other dynamic strings as `keyword` (`ignore_above: 1024`); `message` as `text` without norms (positions kept, so phrases stay fast); `trace.id` and `span.id` as `hex_id` (lowercase hex ids indexed and doc-valued as their bytes) |

Opt in per index template:

```
PUT _index_template/logs
{
  "index_patterns": ["logs-*"],
  "data_stream": {},
  "composed_of": ["lucene_rust_logs@settings", "lucene_rust_logs@mappings"],
  "template": {
    "settings": { "index.derived_source.enabled": true },
    "mappings": { "properties": { "message": { "type": "pattern_text" } } }
  }
}
```

The index template's own `settings` and `mappings` win over the preset's, so
a field the preset maps differently (a `host.name` you filter on without a
time range, say, or a `trace.id` you search with `prefix` or `wildcard`, which
`hex_id` refuses) is overridden there.

`index.derived_source.enabled` is left to the index template because it is a
trade: it drops the stored `_source` copy (the largest saving below) and adds
1-3 ms to a search returning 100 hits, since each hit's `_source` is rebuilt
from doc values and stored fields. With it, map `message` as `pattern_text`
too, as above: the message is then kept as a template plus its variables,
42 MiB less on the corpus below, for about 1 ms more per phrase and 3 ms more
per newest-100 tail. Without derived source `pattern_text` costs space (the
variables are stored beside `_source`: 35 MiB more), so the mappings template
keeps `message` as `text`.

The elected cluster manager puts each template when it is missing or has a
lower `version` than this plugin's, so upgrading the plugin updates them. Copy
one under your own name to change it. `lucene_rust.logs_preset.templates.enabled:
false` (node setting) stops the registration.

## Compatibility

The settings template's `lucene_rust` settings need this plugin, and three of
them write formats only this plugin reads: `RustZstd104` terms, `RustAdaptive`
doc values and `LuceneRustZstdPaged104` stored fields. To make an index
stock-readable again, close it, set those three to `false`, reopen it and
force-merge to rewrite every segment. The mappings template uses stock mapping
options except `hex_id`, a type of this plugin whose segments are plain Lucene
terms and `SORTED_SET` doc values; on a stock node map those two fields as
`keyword` instead.

## Measured

5M synthetic log events over 7 days (1426 MiB NDJSON, a 32-character hex
`trace.id` on each; the corpus and scripts are the logs study's), one shard,
force-merged to one segment, then searched. Median ms of 15 runs, concurrent
segment search off. `MiB` is the sum of the shard's files. Every query runs
natively in all three (577 native, the same 36 `approximate` fallbacks).

| | Plugin default | Preset | Preset + derived source |
|---|---|---|---|
| MiB on disk (x NDJSON) | 1193 (0.84) | 715 (0.50) | 532 (0.37) |
| stored fields `.fdt` | 636 | 315 | 131 |
| terms `.tim` / postings `.doc` | 209 / 76 | 158 / 63 | 159 / 63 |
| points `.kdd` / doc values `.dvd` / norms | 37 / 198 / 5 | 34 / 112 / 0 | 34 / 112 / 0 |
| indexing, CPU µs per event | 55.0 | 61.7 | 55.8 |
| needle trace.id, 7 d | 4.1 | 4.8 | 4.3 |
| term, last 1 h, newest 100 | 8.6 | 5.4 | 4.3 |
| term, 7 d, newest 100 | 10.9 | 8.7 | 11.1 |
| phrase, 24 h, newest 100 | 7.4 | 8.5 | 9.0 |
| rare term, 7 d | 5.7 | 7.0 | 8.5 |
| host AND level:ERROR, 24 h | 5.1 | 6.2 | 8.8 |
| tail: newest 100 | 12.1 | 5.5 | 8.7 |
| stats count by service, ERROR, 24 h | 7.6 | 9.3 | 8.5 |
| timechart 1 m x level, last 1 h | 7.8 | 7.4 | 7.0 |
| timechart 1 h, 7 d | 5.2 | 6.4 | 5.9 |
| timechart status>=500, 7 d | 8.0 | 13.2 | 10.0 |
| p50/p99 duration by service, 15 m | 6.8 | 7.7 | 8.2 |
| count status>=500, 7 d | 3.3 | 3.3 | 3.3 |
| count service:checkout, 7 d | 2.6 | 3.0 | 3.0 |
| wildcard `*timeoutexception*`, 24 h | 9.4 | 11.0 | 11.0 |

The preset stores the events in 60% of the default's space, and with derived
source in 45%. Searches stay within a few ms of the default: newest-first
searches are faster (the index sort), filters and aggregations on the
doc-values-only fields 1-5 ms slower, and derived source adds 1-3 ms to a
search returning 100 hits. Of the saving, paged zstd stored fields are most
(636 to 315 MiB), then `hex_id` trace ids (81: doc values 179 to 112, terms
171 to 158), zstd term blocks (38), the mapping's doc-values-only fields and
no norms (27) and adaptive doc values (19). Indexing takes about 12% more CPU
per event than the default without derived source, mostly for the index sort
and zstd, and about the default's with it.

Measured and left out: `@timestamp` with doc values only saves another 19 MiB
but makes a timechart over all 7 days 5x slower (32 vs 6 ms), since the
date histogram can no longer count from the points. Indexing `host.name`,
`service.name` and `log.level` gave no measured search gain.

`message` as `pattern_text` (same corpus, same queries, same hits):

| | Preset | Preset + `pattern_text` | Preset + derived source | Preset + derived source + `pattern_text` |
|---|---|---|---|---|
| MiB on disk | 715 | 750 | 532 | 490 |
| stored fields `.fdt` / positions `.pos` | 315 / 30 | 381 / 0 | 131 / 30 | 121 / 0 |
| phrase, 24 h, newest 100 | 8.5 | 7.3 | 10.6 | 11.6 |
| term, 7 d, newest 100 | 8.7 | 7.5 | 10.0 | 11.5 |
| tail: newest 100 | 5.5 | 5.2 | 7.6 | 10.7 |
| wildcard `*timeoutexception*`, 24 h | 11.0 | 7.9 | 10.3 | 11.3 |

## Rollover and force merge

Logs arrive in time order, so each segment already covers a narrow time slice
and a time-range filter skips the others. What keeps searches and storage small
as data grows is many small time-bound indices, each force-merged once it stops
taking writes: one segment per index (fewer per-segment costs, best
compression), and retention by deleting whole indices.

With the index-management plugin, an ISM policy does both:

```
PUT _plugins/_ism/policies/logs
{
  "policy": {
    "description": "Roll over daily or at 50 GB, force-merge, delete after 30 days.",
    "default_state": "hot",
    "states": [
      { "name": "hot",
        "actions": [ { "rollover": { "min_index_age": "1d", "min_primary_shard_size": "50gb" } } ],
        "transitions": [ { "state_name": "warm" } ] },
      { "name": "warm",
        "actions": [ { "force_merge": { "max_num_segments": 1 } } ],
        "transitions": [ { "state_name": "delete", "conditions": { "min_index_age": "30d" } } ] },
      { "name": "delete", "actions": [ { "delete": {} } ] }
    ],
    "ism_template": { "index_patterns": ["logs-*"], "priority": 100 }
  }
}
```

Without it, the same from a scheduler:

```
POST logs-app/_rollover
{ "conditions": { "max_age": "1d", "max_primary_shard_size": "50gb" } }
POST .ds-logs-app-000041/_forcemerge?max_num_segments=1     # the backing index just rolled
DELETE .ds-logs-app-000011                                  # past retention
```
