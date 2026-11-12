/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.store;

import org.lucenerust.opensearch.NativeBridge;
import org.lucenerust.opensearch.engine.RustEngineSupport;

import org.opensearch.common.unit.TimeValue;
import org.opensearch.core.common.breaker.CircuitBreaker;
import org.opensearch.threadpool.ThreadPool;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Charges the page cache's resident bytes to the plugin's native-memory breaker ({@code
 * lucene_rust_writer}), so they show in {@code _nodes/stats/breaker} and count toward the parent
 * breaker where it sums its children ({@code indices.breaker.total.use_real_memory: false}). The
 * page cache holds at most its budget, and the breaker's default limit is raised by the budget, so
 * the charge never refuses an index operation the writer's own share would allow. It is refreshed
 * every second.
 */
public final class PageCacheAccounting {
    private static final AtomicLong CHARGED = new AtomicLong();

    private PageCacheAccounting() {}

    /** Starts refreshing the charge every second on {@code threadPool}. */
    public static void start(ThreadPool threadPool) {
        threadPool.scheduleWithFixedDelay(PageCacheAccounting::refresh, TimeValue.timeValueSeconds(1), ThreadPool.Names.GENERIC);
    }

    /** Moves the breaker's charge to the bytes resident now. */
    static void refresh() {
        CircuitBreaker breaker = RustEngineSupport.breaker();
        if (breaker == null) {
            return;
        }
        long resident = Math.max(0, NativeBridge.storePageCacheStats()[1]);
        long previous = CHARGED.getAndSet(resident);
        if (resident != previous) {
            breaker.addWithoutBreaking(resident - previous);
        }
    }

    /** The bytes charged now. */
    static long charged() {
        return CHARGED.get();
    }
}
