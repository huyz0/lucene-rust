/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.lang.reflect.Field;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Private fields of OpenSearch's classes, looked up once per class and name. The plugin reads a
 * few of them per request (an aggregation's settings, a sort's field data); resolving one walks
 * the class hierarchy and throws on every miss, which per request cost more than the native pass
 * it was planning.
 */
final class Reflect {
    /** A class that declares no such field. */
    private static final Object NONE = new Object();
    /** A class that declares the field, which cannot be made accessible. */
    private static final Object BLOCKED = new Object();

    /** Per class, per name: its accessible {@link Field}, {@link #NONE} or {@link #BLOCKED}. */
    private static final ClassValue<ConcurrentHashMap<String, Object>> DECLARED = new ClassValue<>() {
        @Override
        protected ConcurrentHashMap<String, Object> computeValue(Class<?> c) {
            return new ConcurrentHashMap<>();
        }
    };

    private Reflect() {}

    private static Object lookup(Class<?> c, String name) {
        return DECLARED.get(c).computeIfAbsent(name, n -> {
            Field f;
            try {
                f = c.getDeclaredField(n);
            } catch (NoSuchFieldException e) {
                return NONE;
            } catch (RuntimeException e) {
                return BLOCKED;
            }
            try {
                f.setAccessible(true);
                return f;
            } catch (RuntimeException e) {
                return BLOCKED;
            }
        });
    }

    /** {@code c}'s own field {@code name}, made accessible, or null when it has none or it cannot be. */
    static Field declared(Class<?> c, String name) {
        return lookup(c, name) instanceof Field f ? f : null;
    }

    /**
     * The field {@code name} of {@code c} or the nearest superclass declaring it, or null -- also when
     * that nearest one cannot be made accessible: a field further up of the same name is a different
     * one, which it hides.
     */
    static Field inHierarchy(Class<?> c, String name) {
        for (Class<?> k = c; k != null; k = k.getSuperclass()) {
            Object found = lookup(k, name);
            if (found != NONE) {
                return found instanceof Field f ? f : null;
            }
        }
        return null;
    }
}
