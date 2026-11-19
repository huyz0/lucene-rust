# Get started

## Requirements

| | |
|---|---|
| OpenSearch | **3.9.0** exactly (a plugin must match its node's version) |
| JDK | the one OpenSearch 3.9.0 bundles (25) |
| Platform | Linux x86_64 or aarch64, glibc 2.34 or newer. No Windows or macOS |
| Storage | a local file system (remote-backed storage and searchable snapshots are not supported) |
| Other plugins | none that registers a `QueryPhaseSearcher`; `neural-search` does, and this plugin serves its APIs itself |

## 1. Get the plugin zip

There is no published release yet; [build the zip from source](building.md). It is named
`accelerated-search-<version>.zip` and carries the native library for the platforms it was built on.

## 2. Install it on every node

```sh
bin/opensearch-plugin install file:///path/to/accelerated-search-0.1.0.zip
echo '--enable-native-access=ALL-UNNAMED' > config/jvm.options.d/accelerated-search.options
```

Restart the node (a rolling restart for a running cluster). At startup the node loads the native
library and checks its version; a missing library or a platform mismatch stops the node with one
line naming the file.

Installing changes nothing about existing indices' data. Native search starts serving eligible
searches on every index at once, and every index that does not opt out is written by the accelerated
engine from its shard's next start. It reads and writes stock Lucene segments.

## 3. Check that it works

```
GET /_plugins/accelerated/stats
```

Every search is counted as either served natively (`accelerated_queries`) or sent back to
OpenSearch under a named reason. Run your real traffic for an hour and take two readings; the
difference is your workload's mix. `scripts/fallback-report.py` turns every node's counters into
shares by reason.

## Switching things off

| To… | Set |
|---|---|
| stop native search on one index | `index.search.accelerated: false` |
| keep OpenSearch's engine for new indices on a node | `indices.engine.accelerated.default: false` |
| roll a node back to OpenSearch's engine entirely | `indices.engine.accelerated.node_enabled: false` |

Full install, upgrade and rollback procedures, failure signatures and memory sizing are in
[Operating it](reference/operations.md).
