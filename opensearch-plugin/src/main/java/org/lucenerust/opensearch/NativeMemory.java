/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.common.breaker.CircuitBreaker;
import org.opensearch.core.common.breaker.CircuitBreakingException;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.core.indices.breaker.CircuitBreakerService;
import org.opensearch.core.tasks.TaskCancelledException;
import org.opensearch.search.internal.SearchContext;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * The memory a native search, aggregation or rescore allocates, charged to OpenSearch's own {@code
 * request} circuit breaker while the call runs, and the node's bound on everything the library holds.
 *
 * <p>Each native query phase runs in a {@link Scope}. The library counts what the call's threads
 * allocate and, as the request grows, asks for more through an upcall, which this class answers by
 * charging the request breaker ({@code addEstimateBytesAndMaybeBreak}), as OpenSearch's aggregators
 * charge it for their {@code BigArrays}. Native and Java memory of every concurrent request so share
 * one limit, {@code indices.breaker.request.limit}, and the parent breaker. A refused charge stops
 * the native call at its next checkpoint; the search then fails with the breaker's own {@link
 * CircuitBreakingException} (a 429), and is not run again on Lucene. When the scope closes, every
 * byte it was granted is released.
 *
 * <p>{@link #OFF_HEAP_LIMIT} bounds what the library holds across the node -- indexing buffers,
 * caches and every running request -- so that memory outside the heap cannot grow until the
 * container is killed: past it, native requests fail with a {@code CircuitBreakingException} until
 * enough is freed, and indexing goes on under its own breaker. A task cancelled while its native
 * call runs (by the tasks API or search backpressure) stops at the call's next checkpoint.
 */
public final class NativeMemory {
    private static final Logger logger = LogManager.getLogger(NativeMemory.class);

    /**
     * The most the library may hold on this node, outside the JVM heap: {@code auto} (the default)
     * is half of what the machine or container leaves after the heap and the page cache budget, at
     * least 256mb; {@code 0} or {@code -1} is no bound; otherwise a byte size.
     */
    public static final Setting<String> OFF_HEAP_LIMIT = new Setting<>(
        "indices.memory.off_heap.limit",
        "auto",
        v -> {
            parseLimit(v, 1L << 30, 0, 0);
            return v;
        },
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    /** The smallest automatic bound. */
    static final long MIN_AUTO_LIMIT = 256L << 20;

    /** How often running scopes are checked for a cancelled task. */
    static final long CANCEL_POLL_MILLIS = 100;

    private static final ConcurrentHashMap<Long, Scope> OPEN = new ConcurrentHashMap<>();
    private static final AtomicLong TOKENS = new AtomicLong();
    /** The innermost scope open on this thread. */
    private static final ThreadLocal<Scope> CURRENT = new ThreadLocal<>();
    private static volatile boolean installed;
    /** The node's breakers, from the first request (for the global ordinals' fielddata charge). */
    private static volatile CircuitBreakerService breakers;
    /** What the global ordinals' bytes last charged the fielddata breaker; the watch thread only. */
    private static long fielddataCharged;
    private static volatile long limit;

    private NativeMemory() {}

    /** Registers the grant upcall, applies {@link #OFF_HEAP_LIMIT} and starts the cancel watch. */
    @SuppressWarnings("restricted") // the stub has MemoryUpcall's signature in memory.rs
    public static synchronized void install(Settings settings, long pageCacheBudget) {
        if (installed == false) {
            try {
                MethodHandle target = MethodHandles.lookup()
                    .findStatic(NativeMemory.class, "grant", MethodType.methodType(int.class, long.class, long.class));
                MemorySegment stub = Linker.nativeLinker()
                    .upcallStub(target, FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG), Arena.global());
                if (NativeBridge.memorySetUpcall(stub) != NativeBridge.OK) {
                    throw new IllegalStateException("lucene-rust: registering the memory upcall failed: " + NativeBridge.lastError());
                }
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException(e);
            }
            Thread watch = new Thread(NativeMemory::watchCancels, "lucene-rust-memory-cancel");
            watch.setDaemon(true);
            watch.start();
            installed = true;
        }
        applyLimit(OFF_HEAP_LIMIT.get(settings), pageCacheBudget);
        // The native query caches take OpenSearch's own query cache budget: the sets they hold are
        // the ones IndicesQueryCache would hold on the heap for the same searches.
        long queryCache = org.opensearch.indices.IndicesQueryCache.INDICES_CACHE_QUERY_SIZE_SETTING.get(settings).getBytes();
        if (NativeBridge.queryCacheSetBudget(queryCache) != NativeBridge.OK) {
            throw new IllegalStateException("lucene-rust: cannot set the query cache budget: " + NativeBridge.lastError());
        }
    }

    /** Applies a value of {@link #OFF_HEAP_LIMIT}. */
    static void applyLimit(String value, long pageCacheBudget) {
        long total = totalMemory();
        long heap = Runtime.getRuntime().maxMemory();
        long bytes = parseLimit(value, total, heap, pageCacheBudget);
        limit = bytes;
        int rc = NativeBridge.memorySetProcessLimit(bytes);
        if (rc != NativeBridge.OK) {
            throw new IllegalStateException("lucene-rust: cannot set the off-heap limit (status " + rc + ")");
        }
        logger.info(
            "lucene-rust: off-heap limit {} (machine or container {}, heap {})",
            bytes == 0 ? "none" : new ByteSizeValue(bytes),
            new ByteSizeValue(total),
            new ByteSizeValue(heap)
        );
    }

    /** {@code value} of {@link #OFF_HEAP_LIMIT} in bytes, 0 for no bound. */
    static long parseLimit(String value, long total, long heap, long pageCacheBudget) {
        String v = value.trim();
        if (v.equals("auto")) {
            long left = total - heap - Math.max(0, pageCacheBudget);
            return Math.max(MIN_AUTO_LIMIT, left / 2);
        }
        if (v.equals("0") || v.equals("-1")) {
            return 0;
        }
        long bytes = ByteSizeValue.parseBytesSizeValue(v, OFF_HEAP_LIMIT.getKey()).getBytes();
        if (bytes < 0) {
            throw new IllegalArgumentException("[" + OFF_HEAP_LIMIT.getKey() + "] must be auto, 0 or a byte size, got [" + value + "]");
        }
        return bytes;
    }

    /** The machine's memory, or the container's limit when there is one (the JDK reads the cgroup). */
    static long totalMemory() {
        if (ManagementFactory.getOperatingSystemMXBean() instanceof com.sun.management.OperatingSystemMXBean os) {
            return os.getTotalMemorySize();
        }
        return Runtime.getRuntime().maxMemory() * 2;
    }

    /** The node's bound in bytes, 0 when there is none. */
    public static long limit() {
        return limit;
    }

    /**
     * The library's counters: bytes held, the most held, the bound, requests stopped by the request
     * breaker, requests stopped by the bound, requests cancelled mid-call, the query caches' bytes,
     * budget and evictions, decoded sort columns' bytes, and the global ordinals' bytes. Null when
     * unavailable.
     */
    public static long[] stats() {
        long[] out = new long[11];
        return NativeBridge.memoryStats(out) == NativeBridge.OK ? out : null;
    }

    /** Opens a scope on this thread for {@code ctx}'s native calls; close it after them. */
    public static Scope open(SearchContext ctx, String label) {
        CircuitBreaker breaker = requestBreaker(ctx);
        long token = TOKENS.incrementAndGet();
        Scope s = new Scope(token, ctx, breaker, label);
        s.outer = CURRENT.get();
        CURRENT.set(s);
        OPEN.put(token, s);
        int rc = NativeBridge.memoryScopeBegin(token, 0, label.getBytes(StandardCharsets.UTF_8));
        if (rc != NativeBridge.OK) {
            OPEN.remove(token);
            logger.warn("lucene-rust: cannot open a memory scope ({}): {}", rc, NativeBridge.lastError());
            s.opened = false;
        }
        return s;
    }

    /**
     * Throws what a native call's status {@code rc} means when the request's memory bound or a cancel
     * stopped it (see {@link Scope#stopped}); returns for any other status. Call it right after the
     * native call, on its thread, before anything else reads the last error.
     */
    public static void check(int rc) {
        if (rc != NativeBridge.MEMORY_LIMIT && rc != NativeBridge.CANCELLED) {
            return;
        }
        String message = NativeBridge.lastError();
        Scope s = CURRENT.get();
        RuntimeException e = s != null ? s.stopped(rc, message)
            : rc == NativeBridge.CANCELLED ? new TaskCancelledException("cancelled while a native call ran")
            : new CircuitBreakingException("[" + OFF_HEAP_LIMIT.getKey() + "] " + message, CircuitBreaker.Durability.TRANSIENT);
        throw e;
    }

    private static CircuitBreaker requestBreaker(SearchContext ctx) {
        try {
            CircuitBreakerService service = ctx.bigArrays().breakerService();
            if (service != null && breakers == null) {
                breakers = service;
            }
            return service == null ? null : service.getBreaker(CircuitBreaker.REQUEST);
        } catch (RuntimeException e) {
            return null;
        }
    }

    /**
     * The library asking for {@code bytes} more for the request {@code token}: charged to its
     * breaker, 0 when granted. Never throws, since an exception crossing an upcall ends the JVM.
     */
    private static int grant(long token, long bytes) {
        try {
            Scope s = OPEN.get(token);
            if (s == null || s.breaker == null) {
                return 0;
            }
            s.breaker.addEstimateBytesAndMaybeBreak(bytes, s.label);
            s.charged.addAndGet(bytes);
            return 0;
        } catch (CircuitBreakingException e) {
            Scope s = OPEN.get(token);
            if (s != null) {
                s.tripped = e;
            }
            return 1;
        } catch (Throwable t) {
            return 1;
        }
    }

    private static void watchCancels() {
        while (true) {
            try {
                Thread.sleep(CANCEL_POLL_MILLIS);
                syncFielddata();
                for (Scope s : OPEN.values()) {
                    if (s.cancelSent == false && s.ctx.isCancelled()) {
                        s.cancelSent = true;
                        NativeBridge.memoryScopeCancel(s.token);
                    }
                }
            } catch (InterruptedException e) {
                return;
            } catch (Throwable t) {
                logger.debug("lucene-rust: memory cancel watch", t);
            }
        }
    }

    /**
     * Brings the {@code fielddata} breaker to what the library's global ordinals hold now, as
     * OpenSearch charges it for the global ordinals it builds on the heap: they outlive the request
     * that built them, so the parent breaker counts them until the reader holding them closes. The
     * charge follows the library's count; it never refuses, since the ordinals are already built.
     */
    private static void syncFielddata() {
        CircuitBreakerService service = breakers;
        if (service == null) {
            return;
        }
        long[] s = stats();
        if (s == null) {
            return;
        }
        long now = Math.max(0, s[10]);
        long delta = now - fielddataCharged;
        if (delta != 0) {
            service.getBreaker(CircuitBreaker.FIELDDATA).addWithoutBreaking(delta);
            fielddataCharged = now;
        }
    }

    /** One request's native calls on one thread. */
    public static final class Scope implements AutoCloseable {
        final long token;
        final SearchContext ctx;
        final CircuitBreaker breaker;
        final String label;
        final AtomicLong charged = new AtomicLong();
        volatile CircuitBreakingException tripped;
        volatile boolean cancelSent;
        boolean opened = true;
        Scope outer;
        private boolean closed;
        private long peak;

        Scope(long token, SearchContext ctx, CircuitBreaker breaker, String label) {
            this.token = token;
            this.ctx = ctx;
            this.breaker = breaker;
            this.label = label;
        }

        /** The most the request's native calls held, once closed. */
        public long peak() {
            return peak;
        }

        /**
         * The exception to throw for a native call that returned {@code rc}: the request breaker's own
         * when it refused the request, one for the node bound, a cancellation; null for any other
         * status, which the caller handles as before.
         */
        public RuntimeException stopped(int rc, String message) {
            if (rc == NativeBridge.CANCELLED) {
                return new TaskCancelledException("cancelled while the " + label + " ran");
            }
            if (rc != NativeBridge.MEMORY_LIMIT) {
                return null;
            }
            CircuitBreakingException e = tripped;
            if (e != null) {
                return e;
            }
            return new CircuitBreakingException(
                "[" + OFF_HEAP_LIMIT.getKey() + "] " + message,
                0,
                limit,
                CircuitBreaker.Durability.TRANSIENT
            );
        }

        @Override
        public void close() {
            if (closed) {
                return;
            }
            closed = true;
            if (CURRENT.get() == this) {
                CURRENT.set(outer);
            }
            if (opened) {
                long[] out = new long[3];
                if (NativeBridge.memoryScopeEnd(out) == NativeBridge.OK) {
                    peak = out[1];
                }
                OPEN.remove(token);
            }
            long c = charged.getAndSet(0);
            if (c != 0 && breaker != null) {
                breaker.addWithoutBreaking(-c);
            }
        }
    }
}
