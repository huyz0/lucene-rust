/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.util.function.Function;
import org.apache.lucene.index.IndexReaderContext;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.search.internal.SearchContext;

/**
 * What an encoder knows about the request it encodes besides the query itself, passed down the
 * encoders explicitly: the reader being searched, the index's field types and script setting for
 * scripts that name a field, and the highlighter the query is read for, if it is one's.
 *
 * <p>{@link #NONE} is a plain search with none of that known: terms carrying {@code TermStates}
 * fall back, and scripts naming a field or using the shapes of {@code
 * index.script.accelerated_extended} run in Java.
 */
public record EncodeContext(
    IndexReaderContext top,
    Function<String, MappedFieldType> fieldTypes,
    boolean extendedScripts,
    Reader reader,
    boolean extendedHighlight
) {
    /** Whose reading of the query is being encoded. */
    public enum Reader {
        /** The search's own. */
        SEARCH,
        /** The unified (and fast vector) highlighter's, which visits the unrewritten query. */
        HIGHLIGHT,
        /** The plain highlighter's {@code CustomWeightedSpanTermExtractor}. */
        PLAIN_HIGHLIGHT
    }

    public static final EncodeContext NONE = new EncodeContext(null, null, false, Reader.SEARCH, false);

    /** The context of a search of {@code searcherTop} in {@code ctx}'s index. */
    public static EncodeContext of(SearchContext ctx, IndexReaderContext searcherTop) {
        return new EncodeContext(
            searcherTop,
            ctx.mapperService()::fieldType,
            ExtendedScripts.ENABLED.get(ctx.getQueryShardContext().getIndexSettings().getSettings()),
            Reader.SEARCH,
            false
        );
    }

    /** As {@link #of(SearchContext, IndexReaderContext)} for an encoder that reads no reader. */
    public static EncodeContext of(SearchContext ctx) {
        return of(ctx, null);
    }

    public EncodeContext withTop(IndexReaderContext newTop) {
        return new EncodeContext(newTop, fieldTypes, extendedScripts, reader, extendedHighlight);
    }

    /** This context for a highlighter's reading of the query; {@code extended} adds the shapes of {@code index.highlight.accelerated_extended}. */
    EncodeContext forHighlight(Reader highlighter, boolean extended) {
        return new EncodeContext(top, fieldTypes, extendedScripts, highlighter, extended);
    }

    boolean highlight() {
        return reader != Reader.SEARCH;
    }

    boolean plain() {
        return reader == Reader.PLAIN_HIGHLIGHT;
    }
}
