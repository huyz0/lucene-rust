/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.index.SegmentInfos;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.engine.Segment;
import org.opensearch.index.engine.SegmentsStats;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;

/**
 * The package-private members of OpenSearch's engine classes that {@code InternalEngine} uses and
 * {@link RustEngine}, living in another package and class loader, cannot call directly. Reached
 * reflectively, once, at class initialization -- a missing member fails the plugin at startup
 * rather than at the first call.
 */
final class EngineAccess {
    private static final Method SEGMENT_INFO;
    private static final Method UPDATE_MAX_UNSAFE_AUTO_ID_TIMESTAMP;

    static {
        try {
            SEGMENT_INFO = Engine.class.getDeclaredMethod("getSegmentInfo", SegmentInfos.class, boolean.class);
            SEGMENT_INFO.setAccessible(true);
            UPDATE_MAX_UNSAFE_AUTO_ID_TIMESTAMP = SegmentsStats.class.getDeclaredMethod("updateMaxUnsafeAutoIdTimestamp", long.class);
            UPDATE_MAX_UNSAFE_AUTO_ID_TIMESTAMP.setAccessible(true);
        } catch (NoSuchMethodException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private EngineAccess() {}

    /** {@code Engine.Operation.id()}: only indexes and deletes have one. */
    static String id(Engine.Operation op) {
        if (op instanceof Engine.Index index) {
            return index.id();
        }
        if (op instanceof Engine.Delete delete) {
            return delete.id();
        }
        return null;
    }

    /** {@code Engine.getSegmentInfo(SegmentInfos, boolean)}. */
    static Segment[] segmentInfo(Engine engine, SegmentInfos infos, boolean verbose) {
        return (Segment[]) invoke(SEGMENT_INFO, engine, infos, verbose);
    }

    /** {@code SegmentsStats.updateMaxUnsafeAutoIdTimestamp(long)}. */
    static void updateMaxUnsafeAutoIdTimestamp(SegmentsStats stats, long timestamp) {
        invoke(UPDATE_MAX_UNSAFE_AUTO_ID_TIMESTAMP, stats, timestamp);
    }

    private static Object invoke(Method m, Object target, Object... args) {
        try {
            return m.invoke(target, args);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        } catch (InvocationTargetException e) {
            if (e.getCause() instanceof RuntimeException r) {
                throw r;
            }
            throw new IllegalStateException(e.getCause());
        }
    }

    /** {@code Engine.NoOpLock}, which is protected. */
    static final class NoOpLock implements Lock {
        @Override
        public void lock() {}

        @Override
        public void lockInterruptibly() {}

        @Override
        public boolean tryLock() {
            return true;
        }

        @Override
        public boolean tryLock(long time, TimeUnit unit) {
            return true;
        }

        @Override
        public void unlock() {}

        @Override
        public Condition newCondition() {
            throw new UnsupportedOperationException("NoOpLock can't provide a condition");
        }
    }
}
