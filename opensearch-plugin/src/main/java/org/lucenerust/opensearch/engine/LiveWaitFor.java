/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.search.ReferenceManager;
import org.apache.lucene.store.AlreadyClosedException;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.index.engine.EngineConfig;
import org.opensearch.index.shard.RefreshListeners;
import org.opensearch.threadpool.Scheduler;
import org.opensearch.threadpool.ThreadPool;

import java.io.Closeable;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * {@code refresh=wait_for} on an index whose searches read the writer's buffered documents: a write
 * is searchable once acknowledged, so the shard's refresh listeners waiting on it are released
 * without a refresh. One task per node looks at every such shard's {@link RefreshListeners} every
 * {@link #INTERVAL}; when one waits, they are told a refresh up to the translog's last write
 * location happened, which releases every listener at or before it. A real refresh running at the
 * same time may set the location it read instead: every write before either is searchable, so a
 * listener is never released early, at worst on the next look.
 */
final class LiveWaitFor implements Closeable {
    private static final Logger logger = LogManager.getLogger(LiveWaitFor.class);
    /** How often waiting listeners are looked for. */
    static final TimeValue INTERVAL = TimeValue.timeValueMillis(5);

    /** Per node (thread pool): the shards looked at, and the task looking while there are any. */
    private static final Map<ThreadPool, Node> NODES = new HashMap<>();

    private static final class Node {
        final Set<LiveWaitFor> shards = ConcurrentHashMap.newKeySet();
        Scheduler.Cancellable task;
    }

    private final RefreshListeners listeners;
    private final ThreadPool threadPool;
    private final RustIndexWriter writer;
    private final AtomicBoolean releasing = new AtomicBoolean();
    private volatile boolean closed;

    private LiveWaitFor(RefreshListeners listeners, ThreadPool threadPool, RustIndexWriter writer) {
        this.listeners = listeners;
        this.threadPool = threadPool;
        this.writer = writer;
    }

    /** Releases {@code config}'s waiting refresh listeners as above, or null when the index has none to release. */
    static LiveWaitFor start(EngineConfig config, Object writer) {
        if (!(writer instanceof RustIndexWriter w) || w.liveSearch() == false || config.getThreadPool() == null) {
            return null;
        }
        for (ReferenceManager.RefreshListener l : config.getExternalRefreshListener()) {
            if (l instanceof RefreshListeners listeners) {
                ThreadPool pool = config.getThreadPool();
                LiveWaitFor shard = new LiveWaitFor(listeners, pool, w);
                synchronized (NODES) {
                    Node node = NODES.computeIfAbsent(pool, p -> new Node());
                    node.shards.add(shard);
                    if (node.task == null) {
                        // The look on the scheduler thread, a volatile read per shard; the release,
                        // which answers the waiting requests, on the generic pool.
                        node.task = pool.scheduleWithFixedDelay(() -> checkAll(node), INTERVAL, ThreadPool.Names.SAME);
                    }
                }
                return shard;
            }
        }
        return null;
    }

    private static void checkAll(Node node) {
        for (LiveWaitFor shard : node.shards) {
            shard.check();
        }
    }

    private void check() {
        if (closed == false && listeners.refreshNeeded() && releasing.compareAndSet(false, true)) {
            try {
                threadPool.generic().execute(this::release);
            } catch (RuntimeException e) {
                releasing.set(false);
                logger.debug("lucene-rust: could not release wait_for listeners", e);
            }
        }
    }

    /**
     * Releases the waiting listeners. Under a staleness bound ({@link
     * RustIndexWriter#liveStalenessNanos}) the translog's last write location is read first, then
     * every later search is made to read a view opened after it ({@link
     * RustIndexWriter#refreshLiveShared}), which holds the writes up to that location; when that cannot
     * be done now, the listeners wait for the next look.
     */
    private void release() {
        try {
            if (closed == false) {
                listeners.beforeRefresh();
                // Not refreshed, the listeners wait for the next look.
                if (writer.liveStalenessNanos() == 0 || writer.refreshLiveShared()) {
                    listeners.afterRefresh(true);
                }
            }
        } catch (AlreadyClosedException e) {
            logger.debug("lucene-rust: the shard closed while releasing wait_for listeners", e);
        } catch (IOException | RuntimeException e) {
            logger.warn("lucene-rust: releasing wait_for listeners failed", e);
        } finally {
            releasing.set(false);
        }
    }

    @Override
    public void close() {
        closed = true;
        synchronized (NODES) {
            Node node = NODES.get(threadPool);
            if (node != null && node.shards.remove(this) && node.shards.isEmpty()) {
                node.task.cancel();
                NODES.remove(threadPool);
            }
        }
    }
}
