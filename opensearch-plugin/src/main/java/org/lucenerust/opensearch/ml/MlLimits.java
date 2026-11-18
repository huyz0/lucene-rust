/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.ml;

import org.opensearch.OpenSearchStatusException;
import org.opensearch.common.settings.Setting;
import org.opensearch.core.rest.RestStatus;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.IntSupplier;

/**
 * What a node refuses to start for lack of memory: ml-commons' memory circuit breakers (the JVM heap
 * above {@code jvm_heap_memory_threshold} percent, native memory above {@code native_memory_threshold}
 * percent of the container or machine) and the {@code max_model_on_node} count. A refused register,
 * deploy or predict fails with a 429 and stock's text instead of growing the process until the
 * container kills it.
 */
final class MlLimits {
    static final String HEAP_OPEN = "Memory Circuit Breaker is open, please check your resources!";
    static final String NATIVE_OPEN = "Native Memory Circuit Breaker is open, please check your resources!";
    /** Bytes of heap one float of an answer costs at its peak: a boxed list element, the JSON builder, and the byte copies the response goes through. */
    static final long HEAP_PER_FLOAT = 100;
    private static final long COLLECT_EVERY_MILLIS = 5_000;

    private final IntSupplier heapThreshold;
    private final IntSupplier nativeThreshold;
    private final IntSupplier maxModels;
    private final AtomicInteger trips;
    private final AtomicLong lastCollection = new AtomicLong(Long.MIN_VALUE / 2);

    MlLimits(IntSupplier heapThreshold, IntSupplier nativeThreshold, IntSupplier maxModels, AtomicInteger trips) {
        this.heapThreshold = heapThreshold;
        this.nativeThreshold = nativeThreshold;
        this.maxModels = maxModels;
        this.trips = trips;
    }

    /** The limits of a node, read from its settings. */
    static MlLimits of(java.util.function.Function<Setting<Integer>, Integer> setting, AtomicInteger trips) {
        return new MlLimits(
            () -> setting.apply(MlSettings.ML_COMMONS_JVM_HEAP_MEM_THRESHOLD),
            () -> setting.apply(MlSettings.ML_COMMONS_NATIVE_MEM_THRESHOLD),
            () -> setting.apply(MlSettings.ML_COMMONS_MAX_MODELS_PER_NODE),
            trips
        );
    }

    /** Fails with a 429 when the heap or native memory is over its threshold. */
    void checkBreakers() {
        checkBreakers(0);
    }

    /** As {@link #checkBreakers()}, counting {@code heapNeeded} more bytes of heap as already used. */
    void checkBreakers(long heapNeeded) {
        String open = open(heapNeeded);
        if (HEAP_OPEN.equals(open) && collectIfDue()) {
            // The heap counts garbage: a refusal is for live data, so collect once and look again.
            open = open(heapNeeded);
        }
        if (open != null) {
            trips.incrementAndGet();
            throw new OpenSearchStatusException(open, RestStatus.TOO_MANY_REQUESTS);
        }
    }

    private String open(long heapNeeded) {
        return open(heapUsed() + heapNeeded, heapMax(), heapThreshold.getAsInt(), nativeAnonymous(), nativeTotal(), nativeThreshold.getAsInt());
    }

    /** Runs a collection unless one was asked for in the last {@link #COLLECT_EVERY_MILLIS} milliseconds. */
    private boolean collectIfDue() {
        long now = System.nanoTime() / 1_000_000;
        long last = lastCollection.get();
        if (now - last < COLLECT_EVERY_MILLIS || lastCollection.compareAndSet(last, now) == false) {
            return false;
        }
        System.gc();
        return true;
    }

    /** Fails with a 429 when {@code deployed} models of a kind are already on the node and one more would pass {@code max_model_on_node}. */
    void checkModelCount(boolean remote, long deployed) {
        if (deployed >= maxModels.getAsInt()) {
            trips.incrementAndGet();
            throw new OpenSearchStatusException("Exceed max " + (remote ? "remote" : "local") + " model per node limit", RestStatus.TOO_MANY_REQUESTS);
        }
    }

    /** The text of the breaker that is open for these readings, or null. A threshold of 100 never opens; unknown native readings (negative) never open. */
    static String open(long heapUsed, long heapMax, int heapThreshold, long nativeAnonymous, long nativeTotal, int nativeThreshold) {
        if (heapThreshold < 100 && heapMax > 0 && heapUsed * 100 >= heapMax * heapThreshold) {
            return HEAP_OPEN;
        }
        if (nativeThreshold < 100 && nativeAnonymous >= 0 && nativeTotal > 0 && nativeAnonymous * 100 >= nativeTotal * nativeThreshold) {
            return NATIVE_OPEN;
        }
        return null;
    }

    /** Heap that predicting {@code texts} embeddings of {@code dimension} floats takes before the answer is written. */
    static long answerHeap(long texts, long dimension) {
        return Math.multiplyExact(Math.max(0, texts), Math.max(0, dimension)) * HEAP_PER_FLOAT;
    }

    /** The heap a predict of {@code body}'s {@code text_docs} takes to hold its answers: one embedding of the {@code model}'s {@code embedding_dimension} per text. */
    static long answerHeap(java.util.Map<String, Object> body, java.util.Map<String, Object> model) {
        long texts = body.get("text_docs") instanceof java.util.List<?> docs ? docs.size() : 0;
        long dimension = model != null && model.get("model_config") instanceof java.util.Map<?, ?> c && c.get("embedding_dimension") instanceof Number n ? n.longValue() : 0;
        return answerHeap(texts, dimension);
    }

    private static long heapUsed() {
        return ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed();
    }

    private static long heapMax() {
        java.lang.management.MemoryUsage heap = ManagementFactory.getMemoryMXBean().getHeapMemoryUsage();
        return heap.getMax() > 0 ? heap.getMax() : heap.getCommitted();
    }

    private static long nativeTotal() {
        if (ManagementFactory.getOperatingSystemMXBean() instanceof com.sun.management.OperatingSystemMXBean os) {
            return os.getTotalMemorySize();
        }
        return -1;
    }

    /**
     * The process's anonymous memory (the heap, the models, the libraries' buffers; not the page cache, which a
     * container counts but frees on demand), or -1 where it cannot be read.
     */
    private static long nativeAnonymous() {
        try {
            for (String line : Files.readAllLines(Path.of("/proc/self/status"))) {
                if (line.startsWith("RssAnon:")) {
                    return Long.parseLong(line.replaceAll("\\D+", "")) * 1024;
                }
            }
        } catch (IOException | NumberFormatException e) {
            return -1;
        }
        return -1;
    }
}
