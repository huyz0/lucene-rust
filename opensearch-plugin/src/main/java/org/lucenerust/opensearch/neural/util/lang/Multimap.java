/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.util.lang;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** The part of Guava's {@code ArrayListMultimap} neural-search uses: values in a list per key, keys hashed. */
public final class Multimap<K, V> {
    private final Map<K, List<V>> map = new HashMap<>();

    public static <K, V> Multimap<K, V> create() {
        return new Multimap<>();
    }

    public void put(K key, V value) {
        map.computeIfAbsent(key, k -> new ArrayList<>()).add(value);
    }

    public void putAll(Multimap<K, V> other) {
        other.map.forEach((k, vs) -> map.computeIfAbsent(k, x -> new ArrayList<>()).addAll(vs));
    }

    public boolean isEmpty() {
        return map.isEmpty();
    }

    public Map<K, List<V>> asMap() {
        return map;
    }

    public List<Map.Entry<K, V>> entries() {
        List<Map.Entry<K, V>> out = new ArrayList<>();
        map.forEach((k, vs) -> vs.forEach(v -> out.add(new AbstractMap.SimpleImmutableEntry<>(k, v))));
        return out;
    }
}
