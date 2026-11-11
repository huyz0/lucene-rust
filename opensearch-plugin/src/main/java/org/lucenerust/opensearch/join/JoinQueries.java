/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.join;

import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.Query;
import org.opensearch.index.query.QueryBuilder;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The parent-join module's queries, which this plugin sees only as objects (the module's classes
 * are in another class loader): finding them in a request's query tree, and reading them.
 */
public final class JoinQueries {
    /** The late-parsing query's class name. */
    static final String LATE_PARSING = "org.opensearch.join.query.HasChildQueryBuilder$LateParsingQuery";

    private static final Map<String, Object> FIELDS = new ConcurrentHashMap<>();
    private static final Object NONE = new Object();

    private JoinQueries() {}

    /** Whether {@code b} is a {@code has_child} or {@code has_parent} query. */
    static boolean isJoin(QueryBuilder b) {
        String name = b.getWriteableName();
        return "has_child".equals(name) || "has_parent".equals(name);
    }

    /** Whether {@code q} is the late-parsing query the joins build. */
    static boolean isLateParsing(Query q) {
        return q.getClass().getName().equals(LATE_PARSING);
    }

    /**
     * The query a join builder built, its late-parsing query deferred ({@link DeferredJoinQuery}):
     * the query itself when it is another (a join over an unmapped field matches nothing).
     */
    static Query deferred(Query q) {
        if (isLateParsing(q)) {
            return new DeferredJoinQuery(q);
        }
        if (q instanceof BoostQuery b && isLateParsing(b.getQuery())) {
            return new BoostQuery(new DeferredJoinQuery(b.getQuery()), b.getBoost());
        }
        return q;
    }

    /** The late-parsing query's own field {@code name}, accessible, or null. */
    static Field lateField(Class<?> c, String name) {
        Object found = FIELDS.computeIfAbsent(c.getName() + "#" + name, k -> {
            try {
                Field f = c.getDeclaredField(name);
                f.setAccessible(true);
                return f;
            } catch (ReflectiveOperationException | RuntimeException e) {
                return NONE;
            }
        });
        return found instanceof Field f ? f : null;
    }

    /** The method {@code name} of {@code c} or a superclass, accessible, or null. */
    static Method method(Class<?> c, String name, Class<?>... params) {
        for (Class<?> k = c; k != null; k = k.getSuperclass()) {
            try {
                Method m = k.getDeclaredMethod(name, params);
                m.setAccessible(true);
                return m;
            } catch (NoSuchMethodException e) {
                // Further up.
            } catch (RuntimeException e) {
                return null;
            }
        }
        return null;
    }

    /**
     * {@code query} with every {@code has_child} and {@code has_parent} in it wrapped in a {@link
     * NativeJoinQueryBuilder}, in place where they are held by another query's field or list:
     * {@code bool}, {@code constant_score}, {@code dis_max}, {@code boosting}, {@code nested}, {@code
     * function_score}'s query, the joins' own inner queries. The tree's root is returned wrapped.
     */
    public static QueryBuilder wrap(QueryBuilder query) {
        return wrap(query, new IdentityHashMap<>(), 0);
    }

    /** Whether {@code query} holds a join anywhere {@link #wrap} looks. */
    public static boolean holdsJoin(QueryBuilder query) {
        return holds(query, new IdentityHashMap<>(), 0);
    }

    private static final int MAX_DEPTH = 64;

    private static boolean holds(QueryBuilder q, Map<Object, Boolean> seen, int depth) {
        if (q == null || depth > MAX_DEPTH || seen.put(q, Boolean.TRUE) != null) {
            return false;
        }
        if (isJoin(q)) {
            return true;
        }
        for (Class<?> k = q.getClass(); k != null && k != Object.class; k = k.getSuperclass()) {
            for (Field f : k.getDeclaredFields()) {
                Object v = read(q, f);
                if (v instanceof QueryBuilder child && holds(child, seen, depth + 1)) {
                    return true;
                }
                if (v instanceof List<?> list) {
                    for (Object o : list) {
                        if (o instanceof QueryBuilder child && holds(child, seen, depth + 1)) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    private static QueryBuilder wrap(QueryBuilder q, Map<Object, Boolean> seen, int depth) {
        if (q == null || q instanceof NativeJoinQueryBuilder || depth > MAX_DEPTH || seen.put(q, Boolean.TRUE) != null) {
            return q;
        }
        // The children first: a join inside a join's inner query is wrapped too.
        for (Class<?> k = q.getClass(); k != null && k != Object.class; k = k.getSuperclass()) {
            for (Field f : k.getDeclaredFields()) {
                Object v = read(q, f);
                if (v instanceof QueryBuilder child) {
                    QueryBuilder w = wrap(child, seen, depth + 1);
                    if (w != child) {
                        write(q, f, w);
                    }
                } else if (v instanceof List<?> list) {
                    wrapList(list, seen, depth);
                }
            }
        }
        return isJoin(q) ? new NativeJoinQueryBuilder(q) : q;
    }

    @SuppressWarnings("unchecked")
    private static void wrapList(List<?> list, Map<Object, Boolean> seen, int depth) {
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) instanceof QueryBuilder child) {
                QueryBuilder w = wrap(child, seen, depth + 1);
                if (w != child) {
                    try {
                        ((List<Object>) list).set(i, w);
                    } catch (RuntimeException e) {
                        // An unmodifiable list: that join stays as OpenSearch built it.
                    }
                }
            }
        }
    }

    private static Object read(Object o, Field f) {
        if (Modifier.isStatic(f.getModifiers()) || f.getType().isPrimitive()) {
            return null;
        }
        try {
            f.setAccessible(true);
            return f.get(o);
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    private static void write(Object o, Field f, Object v) {
        if (f.getType().isInstance(v) == false) {
            return;
        }
        try {
            f.set(o, v);
        } catch (ReflectiveOperationException | RuntimeException e) {
            // A field that cannot be set: that join stays as OpenSearch built it.
        }
    }
}
