/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.ml;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The counters {@code _plugins/_ml/stats} reports for this node: requests and failures in all, by
 * function and action, and by model and action; an action's counters appear once they are first
 * touched, as stock's do.
 */
final class MlStats {
    static final String REQUESTS = "ml_action_request_count";
    static final String FAILURES = "ml_action_failure_count";

    private long requests;
    private long failures;
    private final Map<String, Map<String, long[]>> algorithms = new TreeMap<>();
    private final Map<String, Map<String, long[]>> models = new TreeMap<>();
    /** The register and deploy runs in flight. */
    final AtomicInteger executing = new AtomicInteger();
    /** Times a request was refused for lack of memory. */
    final AtomicInteger breakerTrips = new AtomicInteger();

    synchronized void request(String function, String modelId, String action) {
        requests++;
        bump(algorithms, function.toLowerCase(java.util.Locale.ROOT), action, 0);
        if (modelId != null) {
            bump(models, modelId, action, 0);
        }
    }

    synchronized void failure(String function, String modelId, String action) {
        failures++;
        bump(algorithms, function.toLowerCase(java.util.Locale.ROOT), action, 1);
        if (modelId != null) {
            bump(models, modelId, action, 1);
        }
    }

    private static void bump(Map<String, Map<String, long[]>> by, String key, String action, int slot) {
        by.computeIfAbsent(key, k -> new TreeMap<>()).computeIfAbsent(action, a -> new long[] { -1, -1 });
        long[] c = by.get(key).get(action);
        c[slot] = c[slot] < 0 ? 1 : c[slot] + 1;
    }

    /** Prediction timings of a model, in milliseconds: every call's count, extremes and mean, and percentiles over the last {@value #RING}. */
    static final class Latency {
        private static final int RING = 1000;
        private final double[] ring = new double[RING];
        private long count;
        private double sum;
        private double min = Double.MAX_VALUE;
        private double max;

        synchronized void add(double ms) {
            ring[(int) (count % RING)] = ms;
            count++;
            sum += ms;
            min = Math.min(min, ms);
            max = Math.max(max, ms);
        }

        synchronized Map<String, Object> render() {
            double[] sorted = java.util.Arrays.copyOf(ring, (int) Math.min(count, RING));
            java.util.Arrays.sort(sorted);
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("average", sum / count);
            out.put("count", count);
            out.put("max", max);
            out.put("min", min);
            out.put("p50", percentile(sorted, 0.5));
            out.put("p90", percentile(sorted, 0.9));
            out.put("p99", percentile(sorted, 0.99));
            return out;
        }

        private static double percentile(double[] sorted, double q) {
            double position = (sorted.length - 1) * q;
            int below = (int) Math.floor(position);
            int above = (int) Math.ceil(position);
            return sorted[below] + (sorted[above] - sorted[below]) * (position - below);
        }
    }

    private final Map<String, Latency[]> latencies = new java.util.concurrent.ConcurrentHashMap<>();

    /** Records a successful prediction of {@code modelId}: its whole request and the model's own run. */
    void latency(String modelId, double requestMs, double inferenceMs) {
        Latency[] l = latencies.computeIfAbsent(modelId, k -> new Latency[] { new Latency(), new Latency() });
        l[0].add(requestMs);
        l[1].add(inferenceMs);
    }

    /** {@code [request, inference]} timings of {@code modelId}, or {@code null} before its first prediction. */
    java.util.List<Map<String, Object>> latencies(String modelId) {
        Latency[] l = latencies.get(modelId);
        return l == null ? null : java.util.List.of(l[0].render(), l[1].render());
    }

    void forget(String modelId) {
        latencies.remove(modelId);
    }

    /** This node's stats; {@code deployed} is how many models it holds. */
    synchronized Map<String, Object> snapshot(int deployed) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("ml_executing_task_count", executing.get());
        out.put("ml_jvm_heap_usage", heapPercent());
        out.put("ml_circuit_breaker_trigger_count", breakerTrips.get());
        out.put("ml_request_count", requests);
        out.put("ml_deployed_model_count", deployed);
        out.put("ml_failure_count", failures);
        out.put("algorithms", render(algorithms));
        out.put("models", render(models));
        return out;
    }

    private static int heapPercent() {
        java.lang.management.MemoryUsage heap = java.lang.management.ManagementFactory.getMemoryMXBean().getHeapMemoryUsage();
        long max = heap.getMax() > 0 ? heap.getMax() : heap.getCommitted();
        return (int) (heap.getUsed() * 100 / Math.max(1, max));
    }

    private static Map<String, Object> render(Map<String, Map<String, long[]>> by) {
        Map<String, Object> out = new LinkedHashMap<>();
        by.forEach((key, actions) -> {
            Map<String, Object> rendered = new LinkedHashMap<>();
            actions.forEach((action, c) -> {
                Map<String, Object> counts = new LinkedHashMap<>();
                if (c[1] >= 0) {
                    counts.put(FAILURES, c[1]);
                }
                if (c[0] >= 0) {
                    counts.put(REQUESTS, c[0]);
                }
                rendered.put(action, counts);
            });
            out.put(key, rendered);
        });
        return out;
    }
}
