/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.stats;

import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.atomic.LongAdder;

/** This node's neural-search event counters. */
public final class EventStatsManager {
    private static final Map<EventStatName, LongAdder> COUNTS = new EnumMap<>(EventStatName.class);

    static {
        for (EventStatName name : EventStatName.values()) {
            COUNTS.put(name, new LongAdder());
        }
    }

    private EventStatsManager() {}

    public static void increment(EventStatName name) {
        COUNTS.get(name).increment();
    }

    public static long count(EventStatName name) {
        return COUNTS.get(name).sum();
    }
}
