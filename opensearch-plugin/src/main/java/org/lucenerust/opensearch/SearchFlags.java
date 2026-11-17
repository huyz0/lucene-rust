/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.opensearch.search.internal.SearchContext;

/** The plugin's per-index settings, read once per settings version. */
final class SearchFlags {
    /** The plugin's index settings as last read, and the settings version they were read at. */
    record Flags(long version, boolean enabled, boolean allShapes, boolean nativeApproximate, boolean aggCache) {}

    private static final java.util.concurrent.ConcurrentHashMap<String, Flags> FLAGS = new java.util.concurrent.ConcurrentHashMap<>();

    /**
     * The plugin's index settings for {@code ctx}'s index, re-read only when the index's settings
     * version moves: {@code IndexSettings.getValue} parses the setting's string on every call, a
     * cost every search paid twice. One small entry per index ever searched on this node.
     */
    static Flags of(SearchContext ctx) {
        org.opensearch.index.IndexSettings settings = ctx.indexShard().indexSettings();
        long version = settings.getIndexMetadata().getSettingsVersion();
        String uuid = settings.getUUID();
        Flags f = FLAGS.get(uuid);
        if (f == null || f.version() != version) {
            f = new Flags(
                version,
                settings.getValue(AcceleratedQueryPhaseSearcher.ENABLED),
                "all".equals(settings.getValue(AcceleratedQueryPhaseSearcher.NATIVE_SHAPES)),
                settings.getValue(AcceleratedQueryPhaseSearcher.NATIVE_APPROXIMATE),
                settings.getValue(AcceleratedQueryPhaseSearcher.AGG_SEGMENT_CACHE)
            );
            FLAGS.put(uuid, f);
        }
        return f;
    }

    /** Whether {@code ctx}'s index runs every shape native ({@code accelerated_shapes: all}). */
    static boolean allShapes(SearchContext ctx) {
        return of(ctx).allShapes();
    }
}
