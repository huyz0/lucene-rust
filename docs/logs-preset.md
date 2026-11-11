# Logs preset

The plugin registers two component templates for log indices. They change
nothing until an index template names them in `composed_of`:

| Template | What it sets |
|---|---|
| `lucene_rust_logs@settings` | the Rust engine (`index.lucene_rust.engine`), index sort `@timestamp` descending, paged zstd stored fields (`index.lucene_rust.stored_fields.paged_zstd`), zstd term blocks (`index.lucene_rust.zstd_terms`), adaptive doc values (`index.lucene_rust.adaptive_doc_values`) |
| `lucene_rust_logs@mappings` | `@timestamp` as `date` with a doc-values skip index (`skip_list`) beside its points; `host.name`, `service.name`, `log.level` as `keyword` with doc values only (`index: false`); `http.response.status_code`, `event.duration` and every dynamically mapped number as doc values with a skip index only; other dynamic strings as `keyword` (`ignore_above: 1024`); `message` as `text` without norms (positions kept, so phrases stay fast) |

Opt in per index template:

```
PUT _index_template/logs
{
  "index_patterns": ["logs-*"],
  "data_stream": {},
  "composed_of": ["lucene_rust_logs@settings", "lucene_rust_logs@mappings"],
  "template": {
    "settings": { "index.derived_source.enabled": true },
    "mappings": { "properties": { "trace": { "properties": { "id": { "type": "keyword" } } } } }
  }
}
```

The index template's own `settings` and `mappings` win over the preset's, so
a field the preset maps differently (a `host.name` you filter on without a
time range, say) is overridden there.

`index.derived_source.enabled` is left to the index template because it is a
trade: it drops the stored `_source` copy (the largest saving below) and makes
every search that returns 100 hits about 2x slower, since each hit's `_source`
is rebuilt from doc values and stored fields.

The elected cluster manager puts each template when it is missing or has a
lower `version` than this plugin's, so upgrading the plugin updates them. Copy
one under your own name to change it. `lucene_rust.logs_preset.templates.enabled:
false` (node setting) stops the registration.

## Compatibility

The mappings template uses stock OpenSearch mapping options only, and works
on a stock node too. The settings template's `lucene_rust` settings need this
plugin, and three of them write formats only this plugin reads:
`RustZstd104` terms, `RustAdaptive` doc values and `LuceneRustZstdPaged104`
stored fields. To make an index stock-readable again, close it, set those
three to `false`, reopen it and force-merge to rewrite every segment.

## Measured

5M synthetic log events over 7 days (1426 MiB NDJSON; the corpus and scripts
are the logs study's), one shard, force-merged to one segment, then searched.
Median ms of 15 runs, concurrent segment search off. `MiB` is the sum of the
shard's files.

| | Plugin default | Preset | Preset + derived source | Plugin "max" (codec zstd, derived source, zstd terms, adaptive DV, stock mapping) |
|---|---|---|---|---|
| MiB on disk (x NDJSON) | 1193 (0.84) | 797 (0.56) | 611 (0.43) | 638 (0.45) |
| stored fields `.fdt` | 636 | 317 | 131 | 133 |
| terms `.tim` / postings `.doc` | 209 / 76 | 171 / 63 | 171 / 63 | 171 / 76 |
| points `.kdd` / doc values `.dvd` / norms | 36 / 198 / 5 | 34 / 179 / 0 | 34 / 179 / 0 | 41 / 179 / 5 |
| indexing, CPU µs per event | 52.5 | 58.6 | 56.0 | 59.9 |
| needle trace.id, 7 d | 3.7 | 5.0 | 5.6 | 7.1 |
| term, last 1 h, newest 100 | 5.4 | 4.6 | 4.2 | 4.5 |
| term, 7 d, newest 100 | 11.0 | 9.9 | 17.8 | 36.2 |
| phrase, 24 h, newest 100 | 7.3 | 8.8 | 15.6 | 36.2 |
| host AND level:ERROR, 24 h | 6.0 | 6.0 | 10.1 | 23.5 |
| tail: newest 100 | 9.7 | 4.4 | 10.2 | 15.1 |
| stats count by service, ERROR, 24 h | 7.2 | 5.2 | 4.7 | 5.3 |
| timechart 1 m x level, last 1 h | 7.1 | 9.2 | 7.8 | 9.7 |
| timechart 1 h, 7 d | 5.2 | 8.0 | 9.0 | 8.5 |
| timechart status>=500, 7 d | 7.2 | 6.9 | 8.6 | 7.7 |
| p50/p99 duration by service, 15 m | 8.2 | 8.0 | 8.5 | 7.8 |
| count status>=500, 7 d | 3.4 | 2.7 | 2.8 | 3.3 |
| count service:checkout, 7 d | 2.9 | 2.8 | 3.2 | 2.9 |
| wildcard `*timeoutexception*`, 24 h | 8.7 | 9.6 | 16.1 | 38.4 |

The preset stores the events in two thirds of the default's space at the same
search speed; with derived source, in half, and the searches returning 100 hits
take about twice as long. Of the preset's saving, the stored fields' paged zstd
is most (636 to 317 MiB), then zstd term blocks (38) and adaptive doc values
(19); the mappings template saves 27 MiB more (no postings for the small
keywords, no points for the numbers, no norms on `message`) at no measured
search cost. Indexing takes about 12% more CPU per event than the default,
mostly for the index sort and zstd.

Measured and left out: `@timestamp` with doc values only saves another 19 MiB
but makes a timechart over all 7 days 5x slower (32 vs 6 ms), since the
date histogram can no longer count from the points.

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
