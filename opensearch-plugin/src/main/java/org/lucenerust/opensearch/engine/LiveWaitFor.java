/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.search.ReferenceManager;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.index.engine.EngineConfig;
import org.opensearch.index.shard.RefreshListeners;
import org.opensearch.threadpool.Scheduler;
import org.opensearch.threadpool.ThreadPool;

import java.io.Closeable;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * {@code refresh=wait_for} on an index whose searches read the writer's buffered documents: a write
 * is searchable once acknowledged, so the shard's refresh listeners waiting on it are released
 * without a refresh. Every {@link #INTERVAL} the shard's {@link RefreshListeners} are checked; when
 * one waits, they are told a refresh up to the translog's last write location happened, which
 * releases every listener at or before it. A real refresh between the two calls only moves that
 * location on, since every write before it is searchable too.
 */
final class LiveWaitFor implements Closeable {
    private static final Logger logger = LogManager.getLogger(LiveWaitFor.class);
    /** How often waiting listeners are looked for: the most a {@code wait_for} write waits. */
    static final TimeValue INTERVAL = TimeValue.timeValueMillis(5);

    private final RefreshListeners listeners;
    private final ThreadPool threadPool;
    private final AtomicBoolean releasing = new AtomicBoolean();
    private final Scheduler.Cancellable task;

    private LiveWaitFor(RefreshListeners listeners, ThreadPool threadPool) {
        this.listeners = listeners;
        this.threadPool = threadPool;
        // The check on the scheduler thread: a volatile read; the release, which answers the
        // waiting requests, on the refresh pool.
        this.task = threadPool.scheduleWithFixedDelay(this::check, INTERVAL, ThreadPool.Names.SAME);
    }

    /** Releases {@code config}'s waiting refresh listeners as above, or null when the index has none to release. */
    static LiveWaitFor start(EngineConfig config, Object writer) {
        if (!(writer instanceof RustIndexWriter w) || w.liveSearch() == false || config.getThreadPool() == null) {
            return null;
        }
        for (ReferenceManager.RefreshListener l : config.getExternalRefreshListener()) {
            if (l instanceof RefreshListeners listeners) {
                return new LiveWaitFor(listeners, config.getThreadPool());
            }
        }
        return null;
    }

    private void check() {
        if (listeners.refreshNeeded() && releasing.compareAndSet(false, true)) {
            try {
                threadPool.executor(ThreadPool.Names.REFRESH).execute(this::release);
            } catch (RuntimeException e) {
                releasing.set(false);
                logger.debug("lucene-rust: could not release wait_for listeners", e);
            }
        }
    }

    private void release() {
        try {
            listeners.beforeRefresh();
            listeners.afterRefresh(true);
        } catch (IOException | RuntimeException e) {
            logger.warn("lucene-rust: releasing wait_for listeners failed", e);
        } finally {
            releasing.set(false);
        }
    }

    @Override
    public void close() {
        task.cancel();
    }
}
