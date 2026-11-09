/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.stats.metrics;


import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.util.Locale;

/**
 * Memory stat information class which gets memory stat data
 */
public class MemoryStat implements MetricStat {

    public static final int BYTES_PER_KILOBYTES = 1024;
    private final MetricStatName statName;

    /**
     * Constructor
     * @param statName the associate stat name identifier
     */
    public MemoryStat(MetricStatName statName) {
        this.statName = statName;
    }

    /**
     * @return the current memory stat value: 0, as the sparse_vector search reads its segments natively
     *     and keeps no JVM forward-index or clustered-posting caches
     */
    private long getByteSize() {
        return 0L;
    }

    public Double getValue() {
        long byteSize = getByteSize();
        if (statName == MetricStatName.MEMORY_SPARSE_MEMORY_USAGE_PERCENTAGE) {
            MemoryMXBean memoryMXBean = ManagementFactory.getMemoryMXBean();
            // memoryMXBean.getHeapMemoryUsage().getMax() might return -1 in special case
            long heapMaxBytes = Math.max(0, memoryMXBean.getHeapMemoryUsage().getMax());
            double percentage = (double) byteSize / heapMaxBytes * 100;
            return Math.round(percentage * 100.0) / 100.0;
        }
        double kbSize = (double) byteSize / BYTES_PER_KILOBYTES;
        return Math.round(kbSize * 100.0) / 100.0;
    }

    @Override
    public MemoryStatSnapshot getStatSnapshot() {
        // For sparse memory usage percentage, there is no use to perform
        // aggregation metric memory usage percentage is a node-specific metric
        return MemoryStatSnapshot.builder()
            .statName(statName)
            .value(getValue())
            .isAggregationMetric(statName != MetricStatName.MEMORY_SPARSE_MEMORY_USAGE_PERCENTAGE)
            .build();
    }
}
