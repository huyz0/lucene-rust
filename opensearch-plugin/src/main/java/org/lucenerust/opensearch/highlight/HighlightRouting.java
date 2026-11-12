/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.highlight;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.search.fetch.subphase.highlight.SearchHighlightContext;
import org.opensearch.search.internal.SearchContext;

import java.lang.reflect.Field;

/**
 * Points a search's {@code unified} highlighting -- named, or the default when a field names no
 * {@code type} -- at {@link NativeUnifiedHighlighter} before its fetch phase starts.
 *
 * <p>OpenSearch registers its highlighters by name and refuses a plugin that registers a built-in
 * name again, so the native one is registered as {@value NativeUnifiedHighlighter#NAME} and each
 * search's field options are retargeted to it here ({@code SearchOperationListener.onPreFetchPhase},
 * which runs before {@code HighlightPhase} picks a highlighter per field). {@code plain} and {@code
 * fvh} fields keep OpenSearch's highlighters.
 */
public final class HighlightRouting {
    private static final Logger logger = LogManager.getLogger(HighlightRouting.class);

    /** {@code SearchHighlightContext.FieldOptions.highlighterType}, or null when it cannot be set. */
    private static final Field HIGHLIGHTER_TYPE = find();

    private HighlightRouting() {}

    private static Field find() {
        try {
            Field f = SearchHighlightContext.FieldOptions.class.getDeclaredField("highlighterType");
            f.setAccessible(true);
            return f;
        } catch (ReflectiveOperationException | RuntimeException e) {
            logger.warn("lucene-rust: highlighting stays on OpenSearch's highlighters: {}", e.toString());
            return null;
        }
    }

    /** Whether the native highlighter can be routed to at all. */
    public static boolean available() {
        return HIGHLIGHTER_TYPE != null;
    }

    /** Retargets {@code context}'s {@code unified} fields to the native highlighter. */
    public static void route(SearchContext context) {
        SearchHighlightContext highlight = context.highlight();
        if (highlight == null || HIGHLIGHTER_TYPE == null) {
            return;
        }
        for (SearchHighlightContext.Field field : highlight.fields()) {
            String type = field.fieldOptions().highlighterType();
            if (type == null || type.equals("unified")) {
                try {
                    HIGHLIGHTER_TYPE.set(field.fieldOptions(), NativeUnifiedHighlighter.NAME);
                } catch (IllegalAccessException e) {
                    return;
                }
            }
        }
    }
}
