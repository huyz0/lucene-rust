/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.join;

import org.apache.lucene.search.Query;
import org.lucenerust.opensearch.RustQueryPhaseSearcher;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.query.AbstractQueryBuilder;
import org.opensearch.index.query.InnerHitContextBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilderVisitor;
import org.opensearch.index.query.QueryRewriteContext;
import org.opensearch.index.query.QueryShardContext;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Map;
import java.util.Objects;

/**
 * A {@code has_child} or {@code has_parent} query, as the parent-join module builds it, whose
 * Lucene query on an index searched natively ({@code index.search.accelerated}) is a {@link
 * DeferredJoinQuery}: the join is then
 * run by the native search, not while OpenSearch prepares the search context. Everything else --
 * its JSON, its rewrite, its inner hits, its query for any other index -- is the wrapped query's.
 * {@link JoinRequestFilter} puts it around the joins of a search request.
 */
public final class NativeJoinQueryBuilder extends AbstractQueryBuilder<NativeJoinQueryBuilder> {
    public static final String NAME = "accelerated_join";

    private final QueryBuilder join;

    NativeJoinQueryBuilder(QueryBuilder join) {
        this.join = Objects.requireNonNull(join);
    }

    public NativeJoinQueryBuilder(StreamInput in) throws IOException {
        super(in);
        join = in.readNamedWriteable(QueryBuilder.class);
    }

    /** The {@code has_child} or {@code has_parent} query. */
    public QueryBuilder join() {
        return join;
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        out.writeNamedWriteable(join);
    }

    /** {@code {"accelerated_join": {"has_child": {...}}}}, though it prints as the join itself. */
    public static NativeJoinQueryBuilder fromXContent(XContentParser parser) throws IOException {
        return new NativeJoinQueryBuilder(parseInnerQueryBuilder(parser));
    }

    @Override
    protected void doXContent(XContentBuilder builder, Params params) throws IOException {
        // The join's own fields, in this object: the request prints as the user wrote it.
        Method m = JoinQueries.method(join.getClass(), "doXContent", XContentBuilder.class, Params.class);
        if (m == null) {
            throw new IllegalStateException("[" + NAME + "] cannot print " + join.getWriteableName());
        }
        invoke(m, builder, params);
    }

    @Override
    public void visit(QueryBuilderVisitor visitor) {
        join.visit(visitor);
    }

    @Override
    protected Query doToQuery(QueryShardContext context) throws IOException {
        Query q = join.toQuery(context);
        if (RustQueryPhaseSearcher.ENABLED.get(context.getIndexSettings().getSettings()) == false) {
            return q;
        }
        return JoinQueries.deferred(q);
    }

    @Override
    protected QueryBuilder doRewrite(QueryRewriteContext context) throws IOException {
        QueryBuilder rewritten = join.rewrite(context);
        if (rewritten == join) {
            return this;
        }
        return JoinQueries.isJoin(rewritten) ? new NativeJoinQueryBuilder(rewritten) : rewritten;
    }

    @Override
    protected void extractInnerHitBuilders(Map<String, InnerHitContextBuilder> innerHits) {
        Method m = JoinQueries.method(join.getClass(), "extractInnerHitBuilders", Map.class);
        if (m == null) {
            throw new IllegalStateException("[" + NAME + "] cannot read the inner hits of " + join.getWriteableName());
        }
        invoke(m, innerHits);
    }

    private void invoke(Method m, Object... args) {
        try {
            m.invoke(join, args);
        } catch (InvocationTargetException e) {
            if (e.getCause() instanceof RuntimeException r) {
                throw r;
            }
            throw new IllegalStateException(e.getCause());
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    protected boolean doEquals(NativeJoinQueryBuilder other) {
        return join.equals(other.join);
    }

    @Override
    protected int doHashCode() {
        return join.hashCode();
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }
}
