/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.rest;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Supplier;

/**
 * k-NN's {@code KNNStats}: every stat the stock plugin's {@code _plugins/_knn/stats} reports, by its
 * {@code StatNames} name. This node's {@code knn_vector} fields are lucene-engine ones, which the
 * stock plugin keeps out of its native-memory cache, graph build and script counters; those stats
 * read as they do for a node that only has lucene-engine fields. The query counters count each
 * shard's {@code knn} query as the stock plugin's {@code KNNQueryBuilder.doToQuery} does.
 */
public final class KnnStats {
    private static final LongAdder[] QUERIES = new LongAdder[6];
    static {
        for (int i = 0; i < QUERIES.length; i++) {
            QUERIES[i] = new LongAdder();
        }
    }

    /** Stats the response gives once for the cluster, outside {@code nodes}. */
    static final Set<String> CLUSTER = Set.of("circuit_breaker_triggered", "model_index_status");

    private static final Map<String, Supplier<Object>> STATS = stats();

    /**
     * The names a request may ask for: the stats, and the names of the groups inside {@code graph_stats}
     * and {@code remote_vector_index_build_stats}, which {@code StatNames} lists too but no node
     * reports on their own.
     */
    private static final Set<String> NAMES = names(STATS.keySet());

    private KnnStats() {}

    /**
     * Counts a shard's {@code knn} query: {@code kind} 0 for {@code k}, 1 for {@code min_score},
     * 2 for {@code max_distance} ({@code VectorQueryType}).
     */
    public static void query(int kind, boolean filtered) {
        QUERIES[kind * 2].increment();
        if (filtered) {
            QUERIES[kind * 2 + 1].increment();
        }
    }

    private static Set<String> names(Set<String> stats) {
        java.util.HashSet<String> all = new java.util.HashSet<>(stats);
        all.addAll(List.of("refresh", "merge", "client_stats", "repository_stats", "build_stats"));
        return Set.copyOf(all);
    }

    /** Every name a request may ask for, as {@code StatNames.getNames}. */
    static Set<String> names() {
        return NAMES;
    }

    /** The stats this node reports. */
    static Set<String> reported() {
        return STATS.keySet();
    }

    /** The value of stat {@code name}. */
    static Object value(String name) {
        return STATS.get(name).get();
    }

    private static Map<String, Supplier<Object>> stats() {
        Map<String, Supplier<Object>> m = new LinkedHashMap<>();
        String[] queries = {
            "knn_query_requests",
            "knn_query_with_filter_requests",
            "min_score_query_requests",
            "min_score_query_with_filter_requests",
            "max_distance_query_requests",
            "max_distance_query_with_filter_requests" };
        for (int i = 0; i < queries.length; i++) {
            LongAdder counter = QUERIES[i];
            m.put(queries[i], counter::sum);
        }
        for (String zero : List.of(
            "hit_count",
            "miss_count",
            "load_success_count",
            "load_exception_count",
            "total_load_time",
            "eviction_count",
            "graph_memory_usage",
            "graph_query_errors",
            "graph_query_requests",
            "graph_index_errors",
            "graph_index_requests",
            "script_compilations",
            "script_compilation_errors",
            "script_query_requests",
            "script_query_errors",
            "training_requests",
            "training_errors",
            "training_memory_usage"
        )) {
            m.put(zero, () -> 0L);
        }
        m.put("graph_memory_usage_percentage", () -> 0.0);
        m.put("training_memory_usage_percentage", () -> 0.0);
        m.put("cache_capacity_reached", () -> false);
        m.put("indices_in_cache", Map::of);
        m.put("circuit_breaker_triggered", () -> false);
        m.put("faiss_initialized", () -> false);
        m.put("nmslib_initialized", () -> false);
        m.put("lucene_initialized", () -> true);
        m.put("indexing_from_model_degraded", () -> false);
        m.put("model_index_status", () -> null);
        m.put("graph_stats", KnnStats::graphStats);
        m.put("remote_vector_index_build_stats", KnnStats::remoteBuildStats);
        return Map.copyOf(m);
    }

    private static Map<String, Object> zeros(String... names) {
        Map<String, Object> m = new LinkedHashMap<>();
        for (String n : names) {
            m.put(n, 0L);
        }
        return m;
    }

    private static Object graphStats() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put(
            "merge",
            zeros("current", "total", "total_time_in_millis", "current_docs", "total_docs", "total_size_in_bytes", "current_size_in_bytes")
        );
        m.put("refresh", zeros("total", "total_time_in_millis"));
        return m;
    }

    private static Object remoteBuildStats() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put(
            "repository_stats",
            zeros(
                "read_success_count",
                "read_failure_count",
                "successful_read_time_in_millis",
                "write_success_count",
                "write_failure_count",
                "successful_write_time_in_millis"
            )
        );
        m.put(
            "client_stats",
            zeros(
                "status_request_failure_count",
                "index_build_terminal_exception",
                "status_request_success_count",
                "index_build_success_count",
                "index_build_failure_count",
                "build_request_failure_count",
                "index_build_merge_abort_exception",
                "build_request_success_count",
                "waiting_time_in_ms"
            )
        );
        m.put(
            "build_stats",
            zeros(
                "remote_index_build_flush_time_in_millis",
                "remote_index_build_current_merge_operations",
                "remote_index_build_current_flush_operations",
                "remote_index_build_current_merge_size",
                "remote_index_build_current_flush_size",
                "remote_index_build_merge_time_in_millis"
            )
        );
        return m;
    }
}
