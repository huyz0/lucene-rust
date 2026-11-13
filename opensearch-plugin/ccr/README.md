# Cross-cluster replication with followers on the accelerated engine

OpenSearch takes one engine factory per index. The stock replication plugin offers its
`ReplicationEngine` for every follower index, so this plugin steps aside there: a follower runs on
OpenSearch's Java engine, and an `extended` k-NN field cannot be followed.

`0001-configurable-follower-engine.patch` (against cross-cluster-replication 3.9.0.0; also on the `3.9` branch of huyz0/cross-cluster-replication) adds a
static node setting to the replication plugin:

| `plugins.replication.follower.engine` | follower index engine |
|---|---|
| `replication` (default, stock behaviour) | replication's `ReplicationEngine` |
| `delegate` | replication offers none; this plugin serves the follower |

With `delegate`, this plugin gives a follower `FollowerEngines.Accelerated` (the accelerated
engine, replaying at the leader's sequence numbers as `ReplicationEngine` does) or, for an index
the accelerated engine does not serve, `FollowerEngines.Standard`. `extended` fields, derived
source and native reads then work on followers as on any accelerated index. Set it on every
follower-cluster node; the leader needs nothing.

The patch also fails a replay whose engine applied an operation at a sequence number other than
the leader's (a `delegate` node with no plugin serving followers).

Build: `opensearch-plugin/ccr/build.sh`. Check: `e2e/verify_ccr_follower.py` against a
leader on :9200/:9300 and a follower on :9201.
