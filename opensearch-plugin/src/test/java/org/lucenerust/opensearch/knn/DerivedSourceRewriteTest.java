/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;

/** The token-by-token rewrite writes the bytes the parsed-map route writes. */
public class DerivedSourceRewriteTest {
    private static final String[] NUMBERS = {
        "0", "-0", "1", "1.0", "1.50", "0.1", "1e10", "1E-5", "123456789012", "-9223372036854775808", "18446744073709551616",
        "3.141592653589793", "1.0e2", "0.30000000000000004", "2.5e-324", "100000000000000000000.5" };
    private static final String[] STRINGS = { "", "a", "tenant-7", "café", "日本語", "line\\nbreak", "q\\\"uote", "\\u0001ctl", "sl\\/ash", "😀" };

    private static String scalar(Random r) {
        switch (r.nextInt(6)) {
            case 0: return NUMBERS[r.nextInt(NUMBERS.length)];
            case 1: return "\"" + STRINGS[r.nextInt(STRINGS.length)] + "\"";
            case 2: return "true";
            case 3: return "false";
            case 4: return "null";
            default: return Integer.toString(r.nextInt());
        }
    }

    private static String vector(Random r, boolean nulls) {
        if (nulls && r.nextInt(12) == 0) {
            return "null";
        }
        StringBuilder b = new StringBuilder("[");
        int n = r.nextInt(6) == 0 ? r.nextInt(3) : 1 + r.nextInt(130);
        for (int i = 0; i < n; i++) {
            if (i > 0) {
                b.append(r.nextBoolean() ? "," : " , ");
            }
            switch (r.nextInt(4)) {
                case 0 -> b.append(r.nextInt(255) - 128);
                case 1 -> b.append(String.format("%.4f", r.nextGaussian()));
                case 2 -> b.append(r.nextGaussian());
                default -> b.append(NUMBERS[r.nextInt(NUMBERS.length)]);
            }
        }
        return b.append("]").toString();
    }

    private static String junk(Random r, int depth) {
        if (depth > 3 || r.nextInt(3) == 0) {
            return scalar(r);
        }
        if (r.nextBoolean()) {
            StringBuilder b = new StringBuilder("{");
            int n = r.nextInt(4);
            for (int i = 0; i < n; i++) {
                if (i > 0) b.append(',');
                b.append("\"k").append(i).append("\":").append(junk(r, depth + 1));
            }
            return b.append('}').toString();
        }
        StringBuilder b = new StringBuilder("[");
        int n = r.nextInt(4);
        for (int i = 0; i < n; i++) {
            if (i > 0) b.append(',');
            b.append(junk(r, depth + 1));
        }
        return b.append(']').toString();
    }

    /** A document with vector fields at "emb", "parts.emb" (an array of objects) and "a.b.emb", among other values. */
    private static String document(Random r, boolean dotted) {
        List<String> members = new ArrayList<>();
        members.add("\"tenant\":\"t" + r.nextInt(50) + "\"");
        if (r.nextInt(5) > 0) members.add("\"emb\":" + vector(r, true));
        if (r.nextInt(3) > 0) {
            StringBuilder p = new StringBuilder("\"parts\":[");
            int n = r.nextInt(4);
            for (int i = 0; i < n; i++) {
                if (i > 0) p.append(',');
                p.append("{\"id\":").append(i).append(",\"emb\":").append(vector(r, true)).append(",\"x\":").append(junk(r, 2)).append('}');
            }
            members.add(p.append(']').toString());
        }
        if (r.nextInt(3) == 0) members.add("\"a\":{\"b\":{\"emb\":" + vector(r, true) + ",\"z\":" + junk(r, 2) + "}}");
        if (r.nextInt(2) == 0) members.add("\"note\":" + junk(r, 0));
        if (r.nextInt(2) == 0) members.add("\"n\":" + scalar(r));
        if (dotted && r.nextInt(2) == 0) members.add("\"a.c\":" + junk(r, 2));
        if (r.nextInt(8) == 0) members.add("\"empty\":{}");
        java.util.Collections.shuffle(members, r);
        return "{" + String.join(r.nextBoolean() ? "," : " ,\n ", members) + "}";
    }

    private static Map<String, List<Object>> vectorsFor(Random r) {
        Map<String, List<Object>> m = new HashMap<>();
        for (String key : new String[] { "emb", "parts.emb", "a.b.emb" }) {
            List<Object> l = new ArrayList<>();
            for (int i = 0; i < 5; i++) {
                if (r.nextBoolean()) {
                    float[] f = new float[1 + r.nextInt(130)];
                    for (int j = 0; j < f.length; j++) f[j] = (float) r.nextGaussian();
                    l.add(f);
                } else {
                    int[] b = new int[1 + r.nextInt(130)];
                    for (int j = 0; j < b.length; j++) b[j] = r.nextInt(256) - 128;
                    l.add(b);
                }
            }
            m.put(key, l);
        }
        return m;
    }

    private static byte[] mapRoute(String json, Map<String, Function<Object, Object>> transformers, boolean mustMatch) throws IOException {
        Map<String, Object> source = XContentHelper.convertToMap(new BytesArray(json), true, (org.opensearch.core.xcontent.MediaType) XContentType.JSON).v2();
        try (BytesStreamOutput out = new BytesStreamOutput()) {
            XContentBuilder b = MediaTypeRegistry.contentBuilder(XContentType.JSON, out).map(XContentMapValues.transform(transformers, mustMatch).apply(source));
            b.close();
            return BytesReference.toBytes(out.bytes());
        }
    }

    @Test
    public void injectsAndMasksAsTheMapRouteDoes() throws IOException {
        Random r = new Random(42);
        int streamed = 0;
        int total = 3000;
        String shortest = null, shortestExpected = null, shortestGot = null;
        for (int round = 0; round < total; round++) {
            String json = document(r, round % 4 == 0);
            Map<String, List<Object>> vectors = vectorsFor(r);
            // Inject: the map route, with fresh iterators per route.
            Map<String, Function<Object, Object>> inject = new HashMap<>();
            Map<String, Iterator<Object>> next = new HashMap<>();
            for (Map.Entry<String, List<Object>> e : vectors.entrySet()) {
                Iterator<Object> it = e.getValue().iterator();
                inject.put(e.getKey(), o -> o == null || it.hasNext() == false ? o : it.next());
                next.put(e.getKey(), e.getValue().iterator());
            }
            byte[] expected = mapRoute(json, inject, false);
            BytesReference got = DerivedSourceRewrite.rewrite(new BytesArray(json), vectors.keySet(), true, path -> {
                Iterator<Object> it = next.get(path);
                return it != null && it.hasNext() ? it.next() : null;
            });
            if (got != null && new String(expected, java.nio.charset.StandardCharsets.UTF_8).equals(new String(BytesReference.toBytes(got), java.nio.charset.StandardCharsets.UTF_8)) == false
                && (shortest == null || json.length() < shortest.length())) {
                shortest = json;
                shortestExpected = new String(expected, java.nio.charset.StandardCharsets.UTF_8);
                shortestGot = new String(BytesReference.toBytes(got), java.nio.charset.StandardCharsets.UTF_8);
            }
            if (got != null) {
                streamed++;
                if (shortest != null) { continue; }
                assertEquals("inject " + json.substring(0, Math.min(json.length(), 300)), new String(expected, java.nio.charset.StandardCharsets.UTF_8), new String(BytesReference.toBytes(got), java.nio.charset.StandardCharsets.UTF_8));
            }
            Map<String, Function<Object, Object>> mask = new HashMap<>();
            for (String key : vectors.keySet()) {
                mask.put(key, o -> o == null ? null : DerivedSource.MASK);
            }
            byte[] expectedMask = mapRoute(json, mask, true);
            BytesReference masked = DerivedSourceRewrite.rewrite(new BytesArray(json), vectors.keySet(), false, path -> DerivedSource.MASK);
            if (masked != null) {
                assertEquals("mask " + json.substring(0, Math.min(json.length(), 300)), new String(expectedMask, java.nio.charset.StandardCharsets.UTF_8), new String(BytesReference.toBytes(masked), java.nio.charset.StandardCharsets.UTF_8));
            }
        }
        if (shortest != null) {
            org.junit.Assert.fail("SHORTEST " + shortest + "\nEXP " + shortestExpected + "\nGOT " + shortestGot);
        }
        assertTrue("most documents are rewritten without the map: " + streamed + " of " + total, streamed > total * 6 / 10);
    }

    @Test
    public void whatItCannotReadTheSameWayIsLeftToTheMapRoute() {
        Map<String, List<Object>> v = Map.of("emb", List.of(new float[] { 1f }));
        // a dotted key, a masked field that is not an array, an array root, and invalid JSON
        for (String json : new String[] { "{\"a.b\":1}", "{\"emb\":3}", "[1]", "{\"emb\":[1]", "{\"emb\":{\"x\":1}}" }) {
            org.junit.Assert.assertNull(json, DerivedSourceRewrite.rewrite(new BytesArray(json), v.keySet(), false, p -> new float[] { 1f }));
        }
        assertNotNull(DerivedSourceRewrite.rewrite(new BytesArray("{\"emb\":[1,2]}"), v.keySet(), false, p -> new float[] { 1f }));
    }
}
